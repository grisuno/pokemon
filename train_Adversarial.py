import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
import time
import random
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import gc
import psutil
from pathlib import Path
from datetime import datetime
import pickle
import weakref
import warnings
from typing import Dict, Any, Optional
import tempfile

# =============================================================================
# 0. CONFIGURACIÓN DIAGNÓSTICA (v16.0 FUSIÓN)
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOBAL_SETTINGS = {
    'batch_size': 128,
    'epochs': 30,
    'lr_sgd': 0.1,
    'lr_topo': 0.01,
    'grid_size': 8,
    'use_nested_learning': True,  # Nueva: Activar Nested Learning
    'use_continuum_memory': True, # Nueva: Sistema de memoria continua
    'checkpoint_interval': 5,     # Nueva: Intervalo de checkpoint
}

ROBUST_CONFIG = {
    'train_eps': 8/255,
    'test_eps': 8/255,
    'pgd_steps_train': 10,
    'pgd_steps_test': 20,
    'autoattack_n': 1000
}

LAMBDAS = {
    'supcon': 0.1,
    'entropy': 0.001,
    'sparsity': 1e-5,
    'nested': 0.01,  # Nueva: Peso para regularización nested
}

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(DEVICE)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(DEVICE)

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = "checkpoints/topolbrain_v16.ckpt"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# 1. MONITOR DE RECURSOS (De RESMA)
# =============================================================================

class ResourceMonitor:
    _cache = weakref.WeakKeyDictionary()
    
    @staticmethod
    def get_memory_gb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    
    @staticmethod
    def log_resources():
        used = ResourceMonitor.get_memory_gb()
        cpu_percent = psutil.cpu_percent(interval=1)
        gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        print(f"💾 RAM: {used:.2f}GB | CPU: {cpu_percent}% | GPU: {gpu_mem:.2f}GB")
    
    @staticmethod
    def clear_cache():
        """Limpia cachés y fuerza garbage collection"""
        _bures_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()

# =============================================================================
# 2. SISTEMA DE CHECKPOINT ROBUSTO (De RESMA)
# =============================================================================

def guardar_checkpoint(data: Dict[str, Any], filename: str = CHECKPOINT_FILE):
    """Sistema de checkpoint robusto con protección contra corrupción"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    temp_file = f"{filename}.tmp"
    backup_file = f"{filename}.bak"
    
    try:
        # Verificar tamaño
        estimated_size = len(pickle.dumps(data)) / (1024**2)
        if estimated_size > 500:  # 500MB límite
            warnings.warn(f"⚠️ Checkpoint grande: {estimated_size:.2f} MB")
        
        # Guardar en archivo temporal
        with open(temp_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Rotar backups
        if os.path.exists(filename):
            os.replace(filename, backup_file)
        
        os.replace(temp_file, filename)
        
        size_mb = os.path.getsize(filename) / (1024**2)
        print(f"✅ Checkpoint guardado: {filename} ({size_mb:.2f} MB)")
        
        # Limpiar recursos
        ResourceMonitor.clear_cache()
        
    except Exception as e:
        print(f"❌ Error guardando checkpoint: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

def cargar_checkpoint(filename: str = CHECKPOINT_FILE) -> tuple[Optional[Dict], bool]:
    """Carga checkpoint con fallback automático"""
    for attempt_file in [filename, f"{filename}.bak"]:
        if os.path.exists(attempt_file):
            try:
                with open(attempt_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ Checkpoint cargado: {attempt_file}")
                return data, True
            except Exception as e:
                print(f"⚠️ Error cargando {attempt_file}: {e}")
                continue
    
    print("ℹ️ No se encontró checkpoint, iniciando de cero")
    return None, False

# =============================================================================
# 3. COMPONENTES NESTED LEARNING (Del paper)
# =============================================================================

class NestedOptimizer(optim.Optimizer):
    """
    Optimizador de múltiples niveles basado en Nested Learning.
    Implementa momentum como memoria asociativa con diferentes frecuencias.
    """
    def __init__(self, params, lr=0.1, momentum=0.9, nested_levels=2, freq_factor=10):
        defaults = dict(lr=lr, momentum=momentum, nested_levels=nested_levels, freq_factor=freq_factor)
        super().__init__(params, defaults)
        
        self.nested_levels = nested_levels
        self.freq_factor = freq_factor
        self.state['step'] = 0
        
        # Inicializar memorias anidadas para cada parámetro
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['nested_memories'] = []
                for level in range(nested_levels):
                    state['nested_memories'].append({
                        'momentum_buffer': torch.zeros_like(p.data),
                        'update_freq': freq_factor ** level,
                        'last_update': 0
                    })
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        self.state['step'] += 1
        step = self.state['step']
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Actualizar cada nivel anidado según su frecuencia
                for idx, memory in enumerate(state['nested_memories']):
                    if step % memory['update_freq'] == 0:
                        # Actualización de momentum con "Local Surprise Signal"
                        buf = memory['momentum_buffer']
                        buf.mul_(group['momentum']).add_(grad)
                        
                        # Aplicar corrección de segundo orden (delta-rule)
                        if idx > 0:  # Niveles superiores tienen reglas más complejas
                            correction = torch.outer(grad.view(-1), buf.view(-1)).view_as(p)
                            buf.add_(correction, alpha=group['lr'] * 0.1)
                        
                        memory['last_update'] = step
                
                # Combinar contribuciones de todos los niveles
                total_update = torch.zeros_like(p.data)
                for memory in state['nested_memories']:
                    total_update.add_(memory['momentum_buffer'], alpha=0.5 ** memory['update_freq'])
                
                p.data.add_(total_update, alpha=-group['lr'])
        
        return loss

class ContinuumMemorySystem(nn.Module):
    """
    Sistema de memoria continua con MLPs de diferentes frecuencias.
    Basado en la Sección 3 del paper Nested Learning.
    """
    def __init__(self, input_dim, hidden_dim=256, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            ) for i in range(num_levels)
        ])
        
        # Frecuencias de actualización exponenciales
        self.update_freqs = [2 ** i for i in range(num_levels)]
        self.step_counters = [0] * num_levels
        
    def forward(self, x):
        # Pasada forward normal
        for i, mlp in enumerate(self.mlps):
            x = mlp(x)
        return x
    
    def should_update_level(self, level_idx, global_step):
        """Determina si un nivel debe actualizarse basado en su frecuencia"""
        return global_step % self.update_freqs[level_idx] == 0
    
    def get_update_mask(self, level_idx, batch_size):
        """Máscara para actualizar solo un subconjunto de parámetros"""
        mask = torch.zeros(self.hidden_dim, device=DEVICE)
        mask[:int(self.hidden_dim * (1 / (level_idx + 1)))] = 1.0
        return mask.unsqueeze(0).expand(batch_size, -1)

# =============================================================================
# 4. COMPONENTES DEL MODELO TOPOBRAIN (ORIGINALES)
# =============================================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, 
            torch.arange(batch_size).view(-1, 1).to(features.device), 0
        )
        mask = mask * logits_mask
        
        anchor_dot = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot, dim=1, keepdim=True)
        logits = anchor_dot - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        return -(mask * log_prob).sum(1) / mask_sum

class PredictiveErrorCell(nn.Module):
    def __init__(self, dim, use_spectral=True):
        super().__init__()
        layer = nn.Linear(dim * 2, dim)
        self.fusion = spectral_norm(layer) if use_spectral else layer
        self.ln = nn.LayerNorm(dim)

    def forward(self, input_signal, prediction):
        pos = F.relu(input_signal - prediction)
        neg = F.relu(prediction - input_signal)
        return self.ln(self.fusion(torch.cat([pos, neg], dim=-1)))

class LearnableAbsenceGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x_sensory, x_prediction):
        error = torch.abs(x_sensory - x_prediction)
        return x_sensory * self.gate_net(error)

class SymbioticBasisRefinement(nn.Module):
    def __init__(self, dim, num_atoms=64):
        super().__init__()
        self.basis_atoms = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis_atoms)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(self.basis_atoms)
        attn = torch.matmul(Q, K.T) * self.scale
        weights = F.softmax(attn, dim=-1)
        x_clean = torch.matmul(weights, self.basis_atoms)
        entropy = -torch.sum(weights * torch.log(weights + 1e-6), dim=-1).mean()
        return x_clean, entropy

class CombinatorialComplexLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type='midbrain'):
        super().__init__()
        self.num_nodes = num_nodes
        self.cfg = config
        self.layer_type = layer_type
        
        use_spec = config.get('use_spectral', True)
        self.node_mapper = spectral_norm(nn.Linear(in_dim, hid_dim)) if use_spec else nn.Linear(in_dim, hid_dim)
        self.cell_mapper = spectral_norm(nn.Linear(in_dim, hid_dim)) if use_spec else nn.Linear(in_dim, hid_dim)
        
        self.symbiotic = SymbioticBasisRefinement(hid_dim)
        
        # Integración con Continuum Memory System
        if config.get('use_continuum_memory', False):
            self.cms = ContinuumMemorySystem(hid_dim * 2, hid_dim)
        else:
            self.cms = None
        
        if layer_type == 'midbrain':
            self.pc_cell = PredictiveErrorCell(hid_dim, use_spec)
        else:
            self.absence_gate = LearnableAbsenceGating(hid_dim)
            
        final = nn.Linear(hid_dim * 2, hid_dim)
        self.final_mix = spectral_norm(final) if use_spec else final
        self.norm = nn.LayerNorm(hid_dim)
        
        self.baseline_mixer = nn.Linear(in_dim, hid_dim)

    def forward(self, x_nodes, adjacency, incidence, global_step=None):
        if self.cfg['use_plasticity'] or self.cfg.get('use_static_topo', False):
            h0 = self.node_mapper(x_nodes)
            h0_agg = torch.matmul(adjacency, h0)
        else:
            h0_agg = self.baseline_mixer(x_nodes)

        cell_input = torch.matmul(incidence.T, x_nodes)
        h2 = self.cell_mapper(cell_input)
        pred_cells, entropy = self.symbiotic(h2)
        pred_nodes = torch.matmul(incidence, pred_cells)
        
        if self.cfg['use_mgf']:
            if self.layer_type == 'midbrain':
                processed = self.pc_cell(h0_agg, pred_nodes)
            else:
                processed = self.absence_gate(h0_agg, pred_nodes)
        else:
            processed = h0_agg
            
        combined = torch.cat([processed, pred_nodes], dim=-1)
        
        # Aplicar Continuum Memory System si está activado
        if self.cms is not None and global_step is not None:
            combined = self.apply_cms(combined, global_step)
            
        out = self.final_mix(combined)
        return self.norm(out), entropy
    
    def apply_cms(self, x, global_step):
        """Aplica actualización condicional basada en frecuencia del CMS"""
        batch_size = x.size(0)
        for level_idx, mlp in enumerate(self.cms.mlps):
            if self.cms.should_update_level(level_idx, global_step):
                mask = self.cms.get_update_mask(level_idx, batch_size)
                x = x * mask + mlp(x) * (1 - mask)
        return x

class TopoBrainNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        grid = config['grid_size']
        num_nodes = grid * grid
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        adj, inc = self._init_grid(grid)
        self.register_buffer('adj_mask', adj > 0)
        self.register_buffer('inc_mask', inc > 0)
        
        if config['use_plasticity']:
            self.adj_weights = nn.Parameter(torch.ones_like(adj) * 2.0)
            self.inc_weights = nn.Parameter(torch.ones_like(inc) * 2.0)
        else:
            self.register_buffer('adj_weights', torch.ones_like(adj) * 5.0)
            self.register_buffer('inc_weights', torch.ones_like(inc) * 5.0)

        self.layer1 = CombinatorialComplexLayer(64, 128, num_nodes, config, 'midbrain')
        self.layer2 = CombinatorialComplexLayer(128, 256, num_nodes, config, 'thalamus')
        
        # Continuum Memory System para el readout
        if config.get('use_continuum_memory', False):
            self.readout_cms = ContinuumMemorySystem(256 * num_nodes, 512)
            readout_in_dim = 512
        else:
            self.readout_cms = None
            readout_in_dim = 256 * num_nodes
            
        self.readout = nn.Linear(readout_in_dim, 10)
        self.proj_head = nn.Sequential(
            nn.Linear(256 * num_nodes, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.global_step = 0

    def _init_grid(self, N):
        num_nodes = N * N
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            r, c = i // N, i % N
            if r > 0: adj[i, i-N] = 1
            if r < N-1: adj[i, i+N] = 1
            if c > 0: adj[i, i-1] = 1
            if c < N-1: adj[i, i+1] = 1
        
        cells = []
        for r in range(N - 1):
            for c in range(N - 1):
                tl = r * N + c
                cells.append([tl, tl+1, tl+N, tl+N+1])
        
        inc = torch.zeros(num_nodes, len(cells))
        for ci, ni in enumerate(cells):
            for n in ni: inc[n, ci] = 1.0
        return adj, inc

    def get_topology(self):
        adj_sparse = torch.zeros_like(self.adj_mask, dtype=torch.float32)
        adj_w = torch.sigmoid(self.adj_weights)
        adj_sparse = torch.where(self.adj_mask, adj_w, adj_sparse)
        curr_adj = F.normalize(adj_sparse, p=1, dim=-1)
        
        inc_sparse = torch.zeros_like(self.inc_mask, dtype=torch.float32)
        inc_w = torch.sigmoid(self.inc_weights)
        inc_sparse = torch.where(self.inc_mask, inc_w, inc_sparse)
        curr_inc = inc_sparse / (inc_sparse.sum(0, keepdim=True) + 1e-6)
        return curr_adj, curr_inc

    def forward(self, x):
        self.global_step += 1
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        curr_adj, curr_inc = self.get_topology()
        
        # Pasar global_step para actualizaciones condicionales
        x, ent1 = self.layer1(x, curr_adj, curr_inc, self.global_step)
        x = F.gelu(x)
        x, ent2 = self.layer2(x, curr_adj, curr_inc, self.global_step)
        x = F.gelu(x)
        
        flat = x.reshape(x.shape[0], -1)
        
        # Aplicar CMS al readout si está activado
        if self.readout_cms is not None:
            flat = self.readout_cms(flat)
            
        logits = self.readout(flat)
        
        proj = None
        if self.cfg.get('use_supcon', False):
            proj = F.normalize(self.proj_head(flat), dim=1)
            
        return logits, proj, ent1+ent2

# =============================================================================
# 5. UTILS ENTRENAMIENTO MEJORADOS
# =============================================================================

def clamp_pgd(x_adv_norm, x_orig_norm, eps):
    x_adv = x_adv_norm * CIFAR_STD + CIFAR_MEAN
    x_orig = x_orig_norm * CIFAR_STD + CIFAR_MEAN
    delta = torch.clamp(x_adv - x_orig, -eps, eps)
    x_adv_clamped = torch.clamp(x_orig + delta, 0.0, 1.0)
    return (x_adv_clamped - CIFAR_MEAN) / CIFAR_STD

def make_adversarial_pgd(model, x, y, eps, steps):
    model.eval()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = clamp_pgd(x + delta, x, eps)
    alpha = eps / 4.0
    
    for _ in range(steps):
        x_adv.requires_grad_()
        out = model(x_adv)[0]
        loss = F.cross_entropy(out, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = clamp_pgd(x_adv, x, eps)
    
    model.train()
    ResourceMonitor.clear_cache()
    return x_adv

def eval_autoattack(model, test_loader, n_samples=1000):
    try:
        from autoattack import AutoAttack
    except ImportError:
        return -1.0

    model.eval()
    all_x, all_y = [], []
    count = 0
    for x, y in test_loader:
        all_x.append(x)
        all_y.append(y)
        count += x.size(0)
        if count >= n_samples: break
        
    x_test = torch.cat(all_x)[:n_samples].to(DEVICE)
    y_test = torch.cat(all_y)[:n_samples].to(DEVICE)
    
    class Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)[0]

    adversary = AutoAttack(Wrapper(model), norm='Linf', eps=ROBUST_CONFIG['test_eps'], 
                          version='standard', verbose=False)
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)
    
    with torch.no_grad():
        acc = Wrapper(model)(x_adv).argmax(1).eq(y_test).float().mean().item()
    
    ResourceMonitor.clear_cache()
    return acc * 100

def create_checkpoint_data(model, optimizer, epoch, config, metrics):
    """Crea estructura de checkpoint completa"""
    return {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
        'epoch': epoch,
        'config': config,
        'metrics': metrics,
        'global_step': getattr(model, 'global_step', 0),
        'timestamp': datetime.now().isoformat(),
        'topology': {
            'adj_weights': torch.sigmoid(model.adj_weights).detach().cpu() if config['use_plasticity'] else None,
            'inc_weights': torch.sigmoid(model.inc_weights).detach().cpu() if config['use_plasticity'] else None
        }
    }

# =============================================================================
# 6. RUNNER DIAGNOSTICO MEJORADO
# =============================================================================

def run_training(config_override, run_name):
    cfg = GLOBAL_SETTINGS.copy()
    cfg.update(config_override)
    seed_everything(cfg['seed'])
    
    # Crear directorio de resultados
    os.makedirs(f"results/{run_name}", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1000, num_workers=2)
    
    model = TopoBrainNet(cfg).to(DEVICE)
    
    # Seleccionar optimizador basado en configuración
    if cfg.get('use_nested_learning', False):
        print("🧠 Usando NestedOptimizer")
        optimizer = NestedOptimizer(model.parameters(), lr=cfg['lr_sgd'], 
                                   momentum=0.9, nested_levels=3)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr_sgd'], momentum=0.9, weight_decay=5e-4)
    
    # Configurar scheduler
    if isinstance(optimizer, optim.SGD):
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
                                               lr_lambda=lambda epoch: 0.5 ** (epoch // 10))
    
    # Optimizador topológico separado
    opt_topo = None
    sched_topo = None
    if cfg['use_plasticity']:
        topo_params = [p for n,p in model.named_parameters() if 'weights' in n and p.requires_grad]
        opt_topo = optim.AdamW(topo_params, lr=cfg['lr_topo'], weight_decay=1e-3)
        def lambda_topo(epoch):
            if epoch < 10: return 0.0
            if epoch < 20: return (epoch - 10) / 10.0
            return 1.0
        sched_topo = optim.lr_scheduler.LambdaLR(opt_topo, lr_lambda=lambda_topo)
    
    supcon = SupConLoss(temperature=0.07)
    
    print(f"\n--- Running: {run_name} ---")
    ResourceMonitor.log_resources()
    
    # Reanudar desde checkpoint si existe
    start_epoch = 1
    if cfg.get('resume', False):
        checkpoint_data, loaded = cargar_checkpoint()
        if loaded and checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state'])
            if checkpoint_data['optimizer_state'] and hasattr(optimizer, 'load_state_dict'):
                optimizer.load_state_dict(checkpoint_data['optimizer_state'])
            start_epoch = checkpoint_data.get('epoch', 0) + 1
            print(f"🔄 Reanudando desde época {start_epoch}")
    
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        model.train()
        meters = {'loss': 0, 'acc': 0, 'nested_loss': 0}
        eps = ROBUST_CONFIG['train_eps']
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = make_adversarial_pgd(model, x, y, eps, ROBUST_CONFIG['pgd_steps_train'])
            
            if opt_topo: opt_topo.zero_grad()
            optimizer.zero_grad()
            
            logits, proj, entropy = model(x_adv)
            
            # Pérdida principal
            loss = F.cross_entropy(logits, y)
            
            # Pérdida SupCon
            if cfg['use_supcon']: 
                loss += LAMBDAS['supcon'] * supcon(proj, y).mean()
            
            # Regularización de entropía
            loss -= LAMBDAS['entropy'] * entropy
            
            # Regularización de sparseza topológica
            if cfg['use_plasticity']:
                sparse_loss = torch.norm(torch.sigmoid(model.adj_weights), 1)
                loss += LAMBDAS['sparsity'] * sparse_loss
            
            # Pérdida nested learning (para optimizadores profundos)
            if cfg.get('use_nested_learning', False) and hasattr(optimizer, 'nested_levels'):
                nested_reg = sum(p.abs().sum() for p in model.parameters()) * LAMBDAS['nested']
                loss += nested_reg
                meters['nested_loss'] += nested_reg.item()
            
            # Backward con clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            if opt_topo: opt_topo.step()
            
            meters['loss'] += loss.item()
            meters['acc'] += logits.argmax(1).eq(y).sum().item()
        
        scheduler.step()
        if sched_topo: sched_topo.step()
        
        # Evaluación intermediaria y checkpoint
        if epoch % cfg['checkpoint_interval'] == 0 or epoch == cfg['epochs']:
            metrics = {
                'loss': meters['loss']/len(train_loader),
                'acc': 100*meters['acc']/len(train_loader.dataset),
                'nested_loss': meters['nested_loss']/len(train_loader)
            }
            
            # Checkpoint con todas las métricas
            checkpoint_data = create_checkpoint_data(model, optimizer, epoch, cfg, metrics)
            guardar_checkpoint(checkpoint_data)
            
            # Guardar snapshot topológico
            save_topology_snapshot(model, epoch, run_name)
            
            # Log de recursos
            ResourceMonitor.log_resources()
            
            # Mostrar métricas
            base_msg = (f"Ep {epoch:02d} | Loss: {metrics['loss']:.4f} | "
                       f"Acc: {metrics['acc']:.2f}%")
            
            if cfg['use_plasticity']:
                sparsity = (torch.sigmoid(model.adj_weights) > 0.5).float().mean().item()
                print(f"{base_msg} | Sparsity: {sparsity:.2%} | Nested: {metrics['nested_loss']:.4f}")
            else:
                print(base_msg)

    # EVALUACIÓN FINAL
    print("\n🎯 Evaluación final...")
    ResourceMonitor.log_resources()
    
    model.eval()
    correct = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad(): 
            correct += model(x)[0].argmax(1).eq(y).sum().item()
    clean_acc = 100 * correct / len(test_ds)

    correct = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = make_adversarial_pgd(model, x, y, ROBUST_CONFIG['test_eps'], 
                                   ROBUST_CONFIG['pgd_steps_test'])
        with torch.no_grad(): 
            correct += model(x_adv)[0].argmax(1).eq(y).sum().item()
    pgd_acc = 100 * correct / len(test_ds)

    aa_acc = eval_autoattack(model, test_loader, n_samples=ROBUST_CONFIG['autoattack_n'])
    
    # Checkpoint final con métricas completas
    final_metrics = {
        'clean_acc': clean_acc,
        'pgd_acc': pgd_acc,
        'aa_acc': aa_acc,
        'final_loss': meters['loss']/len(train_loader),
        'final_sparsity': (torch.sigmoid(model.adj_weights) > 0.5).float().mean().item() if cfg['use_plasticity'] else 0
    }
    
    final_checkpoint = create_checkpoint_data(model, optimizer, cfg['epochs'], cfg, final_metrics)
    guardar_checkpoint(final_checkpoint, f"{CHECKPOINT_DIR}/final_{run_name}.ckpt")
    
    # Limpiar al final
    ResourceMonitor.clear_cache()
    
    return clean_acc, pgd_acc, aa_acc

def save_topology_snapshot(model, epoch, run_name):
    if not model.cfg['use_plasticity']: return
    
    adj_w = torch.sigmoid(model.adj_weights).detach().cpu().numpy()
    os.makedirs(f"results/{run_name}", exist_ok=True)
    
    plt.figure(figsize=(8,6))
    plt.imshow(adj_w, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Topology Epoch {epoch}")
    plt.savefig(f"results/{run_name}/topo_ep{epoch:03d}.png")
    plt.close()

# =============================================================================
# 7. SUITE DIAGNÓSTICA FINAL
# =============================================================================

def run_diagnostic_suite():
    experiments = [
        {
            'name': 'Baseline (AT)',
            'config': {'use_supcon': False, 'use_mgf': False, 'use_plasticity': False, 
                      'use_static_topo': False, 'use_nested_learning': False}
        },
        {
            'name': 'TopoBrain (v15.3)',
            'config': {'use_supcon': True, 'use_mgf': True, 'use_plasticity': True, 
                      'use_spectral': True, 'use_nested_learning': False}
        },
        {
            'name': 'TopoBrain+Nested (v16.0)',
            'config': {'use_supcon': True, 'use_mgf': True, 'use_plasticity': True, 
                      'use_spectral': True, 'use_nested_learning': True, 
                      'use_continuum_memory': True}
        }
    ]
    
    seeds = [42]
    results_table = []
    
    print(f"{'='*60}\nTOPOBRAIN v16.0 - SUITE DIAGNÓSTICA FUSIONADA\n{'='*60}")
    
    for exp in experiments:
        print(f"\n>>> Experimento: {exp['name']} <<<")
        clean_accs, pgd_accs, aa_accs = [], [], []
        
        for seed in seeds:
            exp['config']['seed'] = seed
            exp['config']['resume'] = False  # Forzar reinicio para comparación justa
            
            try:
                c, p, a = run_training(exp['config'], f"{exp['name']}_s{seed}".replace(" ", "_"))
                clean_accs.append(c)
                pgd_accs.append(p)
                aa_accs.append(a)
            except Exception as e:
                print(f"❌ Error en experimento {exp['name']}: {e}")
                continue
            
            # Liberar memoria entre experimentos
            ResourceMonitor.clear_cache()
        
        if clean_accs:  # Solo agregar si hubo resultados
            results_table.append({
                'name': exp['name'],
                'clean': np.mean(clean_accs),
                'pgd': np.mean(pgd_accs),
                'aa': np.mean(aa_accs)
            })
        
        print(f"✓ Completado: {exp['name']}")
    
    print(f"\n{'='*60}\nRESULTADOS DIAGNÓSTICOS\n{'='*60}")
    print("Config                   Clean    PGD-20   AutoAttack")
    print("-" * 60)
    for r in results_table:
        aa_str = f"{r['aa']:.2f}%" if r['aa'] != -1 else "N/A"
        print(f"{r['name']:<25} {r['clean']:.2f}%   {r['pgd']:.2f}%   {aa_str}")
    print("-" * 60)
    
    # Guardar resultados finales
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'experiments': results_table,
        'config': GLOBAL_SETTINGS
    }
    
    with open(f"results/diagnostic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        import json
        json.dump(final_results, f, indent=2)
    
    print("✅ Suite diagnóstica completada. Resultados guardados.")

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"topobrain_v16_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    try:
        run_diagnostic_suite()
    except KeyboardInterrupt:
        print("\n⏹️  Ejecución interrumpida por usuario")
        ResourceMonitor.clear_cache()
        exit(0)
    except Exception as e:
        print(f"💥 Error crítico: {e}")
        import traceback
        traceback.print_exc()
        ResourceMonitor.clear_cache()
        exit(1)
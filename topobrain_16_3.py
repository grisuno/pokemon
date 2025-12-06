import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
import numpy as np
import os
import matplotlib.pyplot as plt
import gc
import psutil
from datetime import datetime
import logging
import warnings
import random
import pickle
import time
import json
import pandas as pd

# =============================================================================
# 0. CONFIGURACIÓN DE PRODUCCIÓN (v16.5)
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'

GLOBAL_SETTINGS = {
    'batch_size': 128,
    'epochs': 30,
    'lr_sgd': 0.1,
    'lr_topo': 0.01,
    'grid_size': 8,
    'checkpoint_interval': 5,
    'memory_limit_gb': 8.0,
    'debug_mode': False,
    'topo_warmup_epochs': 5,
    'use_plasticity': True,
    'use_mgf': True,
    'use_supcon': False,
    'use_spectral': True,
    'use_static_topo': False,
    'use_compile': False,
    'seed': 42,
}

ROBUST_CONFIG_FAST = {'train_eps': 8/255, 'test_eps': 8/255, 'pgd_steps_train': 7, 'pgd_steps_test': 10, 'autoattack_n': 500}
ROBUST_CONFIG_PAPER = {'train_eps': 8/255, 'test_eps': 8/255, 'pgd_steps_train': 10, 'pgd_steps_test': 50, 'autoattack_n': 10000}
USE_FAST_MODE = True
ROBUST_CONFIG = ROBUST_CONFIG_FAST if USE_FAST_MODE else ROBUST_CONFIG_PAPER

LAMBDAS = {'supcon': 0.1, 'entropy': 0.001, 'sparsity': 1e-5}
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(DEVICE)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(DEVICE)
CHECKPOINT_FILE = "checkpoints/topolbrain_v16_fusion.ckpt"

def seed_everything(seed):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# =============================================================================
# 1. MONITOR DE RECURSOS
# =============================================================================
class ResourceMonitor:
    @staticmethod
    def get_memory_gb():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    
    @staticmethod
    def check_memory_limit(limit_gb=8.0):
        used = ResourceMonitor.get_memory_gb()
        if used > limit_gb:
            raise RuntimeError(f"🚨 Memory limit exceeded: {used:.2f}GB > {limit_gb}GB")
        return used
    
    @staticmethod
    def log_resources():
        used = ResourceMonitor.get_memory_gb()
        cpu_percent = psutil.cpu_percent(interval=1)
        gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        logging.info(f"💾 RAM: {used:.2f}GB | CPU: {cpu_percent}% | GPU: {gpu_mem:.2f}GB")
        return used

    @staticmethod
    def clear_cache():
        torch.cuda.empty_cache(); gc.collect()

# =============================================================================
# 2. SISTEMA DE CHECKPOINT
# =============================================================================
def guardar_checkpoint(data, filename=CHECKPOINT_FILE):
    os.makedirs("checkpoints", exist_ok=True)
    temp_file = f"{filename}.tmp"
    
    try:
        essential_data = {
            'model_state': data['model_state'],
            'optimizers': data['optimizers'],
            'schedulers': data['schedulers'],
            'epoch': data['epoch'],
            'metrics': {k: v for k, v in data['metrics'].items() if isinstance(v, (int, float))},
            'global_step': data.get('global_step', 0)
        }
        
        with open(temp_file, 'wb') as f:
            pickle.dump(essential_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        os.replace(temp_file, filename)
        logging.info(f"✅ Checkpoint saved: {filename}")
        
    except Exception as e:
        logging.error(f"❌ Error saving checkpoint: {e}")
        if os.path.exists(temp_file): os.remove(temp_file)
        raise

def cargar_checkpoint(filename=CHECKPOINT_FILE):
    for attempt in [filename, f"{filename}.bak"]:
        if os.path.exists(attempt):
            try:
                with open(attempt, 'rb') as f:
                    data = pickle.load(f)
                logging.info(f"✅ Loaded checkpoint: {attempt}")
                return data, True
            except Exception as e:
                logging.warning(f"⚠️ Failed to load {attempt}: {e}")
                continue
    return None, False

# =============================================================================
# 3. COMPONENTES CORE
# =============================================================================

class LearnableAbsenceGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.ReLU(),
            nn.Linear(dim // 4, dim), nn.Sigmoid()
        )

    def forward(self, x_sensory, x_prediction):
        error = torch.abs(x_sensory - x_prediction)
        return x_sensory * self.gate_net(error)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if features.size(0) < 32:
            warnings.warn(f"⚠️ SupConLoss: batch_size={features.size(0)} bajo para contraste")
        
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
        mask_sum = torch.clamp(mask_sum, min=1e-6)
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

        if self.cfg['use_mgf']:
            batch_size = x_nodes.size(0)
            inc_T_batch = incidence.T.unsqueeze(0).expand(batch_size, -1, -1)
            cell_input = torch.bmm(inc_T_batch, x_nodes)
            
            h2 = self.cell_mapper(cell_input)
            pred_cells, entropy = self.symbiotic(h2)
            pred_nodes = torch.matmul(incidence, pred_cells)
        else:
            pred_nodes = torch.zeros_like(h0_agg)
            entropy = torch.tensor(0.0, device=x_nodes.device)
        
        if self.cfg['use_mgf']:
            if self.layer_type == 'midbrain':
                processed = self.pc_cell(h0_agg, pred_nodes)
            else:
                processed = self.absence_gate(h0_agg, pred_nodes)
        else:
            processed = h0_agg
            
        out = self.final_mix(torch.cat([processed, pred_nodes], dim=-1))
        return self.norm(out), entropy

class TopoBrainNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        grid = config['grid_size']
        num_nodes = grid * grid
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        
        adj, inc = self._init_grid(grid)
        self.register_buffer('adj_mask', adj > 0)
        self.register_buffer('inc_mask', inc > 0)
        
        if config['use_plasticity']:
            self.adj_weights = nn.Parameter(torch.randn_like(adj) * 0.5 - 1.5)
            self.inc_weights = nn.Parameter(torch.randn_like(inc) * 0.5 - 1.5)
        else:
            self.register_buffer('adj_weights', torch.ones_like(adj) * 5.0)
            self.register_buffer('inc_weights', torch.ones_like(inc) * 5.0)

        self.layer1 = CombinatorialComplexLayer(64, 128, num_nodes, config, 'midbrain')
        self.layer2 = CombinatorialComplexLayer(128, 256, num_nodes, config, 'thalamus')
        
        self.readout = nn.Linear(256 * num_nodes, 10)
        self.proj_head = nn.Sequential(
            nn.Linear(256 * num_nodes, 256), nn.ReLU(),
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
        temp = max(0.2, 1.0 - self.global_step / 5000)
        
        adj_w = torch.sigmoid(self.adj_weights) * self.adj_mask.float()
        degree = torch.clamp(adj_w.sum(dim=1, keepdim=True), min=1e-6)
        degree_sqrt_inv = 1.0 / torch.sqrt(degree)
        curr_adj = (adj_w / temp) * degree_sqrt_inv * degree_sqrt_inv.transpose(-1, -2)
        
        inc_w = torch.sigmoid(self.inc_weights) * self.inc_mask.float()
        inc_sum = torch.clamp(inc_w.sum(0, keepdim=True), min=1e-6)
        curr_inc = inc_w / inc_sum
        
        return curr_adj, curr_inc

    def forward(self, x):
        self.global_step += 1
        if self.global_step % 100 == 0:
            ResourceMonitor.check_memory_limit(GLOBAL_SETTINGS['memory_limit_gb'])
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        curr_adj, curr_inc = self.get_topology()
        
        x, ent1 = self.layer1(x, curr_adj, curr_inc)
        x = F.gelu(x)
        x, ent2 = self.layer2(x, curr_adj, curr_inc)
        x = F.gelu(x)
        
        flat = x.reshape(x.shape[0], -1)
        logits = self.readout(flat)
        
        proj = None
        if self.cfg.get('use_supcon', False):
            proj = F.normalize(self.proj_head(flat), dim=1)
            
        return logits, proj, ent1+ent2

# =============================================================================
# 4. UTILS
# =============================================================================

def clamp_pgd(x_adv_norm, x_orig_norm, eps):
    x_adv = x_adv_norm * CIFAR_STD + CIFAR_MEAN
    x_orig = x_orig_norm * CIFAR_STD + CIFAR_MEAN
    delta = torch.clamp(x_adv - x_orig, -eps, eps)
    x_adv_clamped = torch.clamp(x_orig + delta, 0.0, 1.0)
    return (x_adv_clamped - CIFAR_MEAN) / CIFAR_STD

def make_adversarial_pgd(model, x, y, eps, steps):
    was_training = model.training
    model.eval()
    
    delta = torch.empty_like(x).uniform_(-eps, eps).detach()
    x_adv = clamp_pgd(x + delta, x, eps).detach()
    alpha = eps / steps
    
    for step in range(steps):
        if step % 5 == 0: ResourceMonitor.clear_cache()
        
        x_adv = x_adv.detach().requires_grad_()
        with torch.enable_grad():
            out = model(x_adv)[0]
            loss = F.cross_entropy(out, y)
            grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        
        x_adv = x_adv.detach() + alpha * grad.sign().detach()
        x_adv = clamp_pgd(x_adv, x, eps).detach()
        del grad
    
    model.train(was_training)
    ResourceMonitor.clear_cache()
    return x_adv.detach()

def eval_autoattack(model, test_loader, n_samples=1000):
    try: from autoattack import AutoAttack
    except ImportError: return -1.0

    model.eval()
    all_x, all_y = [], []
    count = 0
    for x, y in test_loader:
        all_x.append(x); all_y.append(y)
        count += x.size(0)
        if count >= n_samples: break
        if count % 500 == 0: ResourceMonitor.clear_cache()
        
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

def save_topology_snapshot(model, epoch, run_name):
    if not model.cfg['use_plasticity']: return
    
    with torch.no_grad():
        adj_w = torch.sigmoid(model.adj_weights)
        sparsity = (adj_w > 0.5).float().mean().item()
        mean_weight = adj_w.mean().item()
        entropy = -(adj_w * torch.log(adj_w + 1e-8)).sum(1).mean().item()
    
    os.makedirs(f"results/{run_name}", exist_ok=True)
    metrics = {'epoch': epoch, 'sparsity': sparsity, 'mean_weight': mean_weight, 'entropy': entropy}
    with open(f"results/{run_name}/topo_metrics.jsonl", 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    try:
        plt.figure(figsize=(8,6))
        plt.imshow(adj_w.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f"Topology Epoch {epoch}")
        plt.savefig(f"results/{run_name}/topo_ep{epoch:03d}.png")
        plt.close(); plt.clf()
    except: pass
    ResourceMonitor.clear_cache()

def plot_topology_evolution(run_name):
    try:
        df = pd.read_json(f"results/{run_name}/topo_metrics.jsonl", lines=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Topology Evolution - {run_name}')
        axes[0].plot(df['epoch'], df['sparsity']); axes[0].set_title('Sparsity')
        axes[1].plot(df['epoch'], df['mean_weight']); axes[1].set_title('Mean Weight')
        axes[2].plot(df['epoch'], df['entropy']); axes[2].set_title('Entropy')
        plt.tight_layout()
        plt.savefig(f"results/{run_name}/topo_evolution.png")
        plt.close()
    except: pass

# =============================================================================
# 5. RUNNER DE ENTRENAMIENTO
# =============================================================================

def run_training(config_override, run_name):
    cfg = GLOBAL_SETTINGS.copy()
    cfg.update(config_override)
    seed_everything(cfg['seed'])
    
    if cfg.get('use_supcon', False) and cfg['batch_size'] < 64:
        warnings.warn(f"⚠️ SupCon activado con batch_size={cfg['batch_size']}. Considere aumentar a 256+")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"run_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info(f"STARTING RUN: {run_name}")
    logging.info(f"Config: {cfg}")
    logging.info("="*80)
    
    os.makedirs(f"results/{run_name}", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    if cfg.get('debug_mode', False):
        cfg['epochs'] = 3
        cfg['checkpoint_interval'] = 1
        ROBUST_CONFIG['pgd_steps_train'] = 2
        ROBUST_CONFIG['autoattack_n'] = 100
        logging.warning("🔧 DEBUG MODE ACTIVE - Results will be garbage")
    
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    
    num_workers = min(os.cpu_count(), 8)
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=500, num_workers=num_workers, pin_memory=True)
    
    model = TopoBrainNet(cfg).to(DEVICE)
    
    # ✅ FIX: Compilación condicional y con dynamic=True para evitar recompilaciones
    if cfg['use_compile']:
        logging.info("🔥 Compiling model with torch.compile(dynamic=True)...")
        # Compilar solo el forward, no todo el módulo (evita recompilación por requires_grad)
        model.forward = torch.compile(model.forward, mode='default', dynamic=True)
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    main_params = [p for n,p in model.named_parameters() if 'weights' not in n and p.requires_grad]
    topo_params = [p for n,p in model.named_parameters() if 'weights' in n and p.requires_grad]
    
    opt_main = optim.SGD(main_params, lr=cfg['lr_sgd'], momentum=0.9, weight_decay=5e-4)
    sched_main = optim.lr_scheduler.MultiStepLR(opt_main, milestones=[15, 25], gamma=0.1)
    
    opt_topo, sched_topo = None, None
    if cfg['use_plasticity']:
        opt_topo = optim.AdamW(topo_params, lr=cfg['lr_topo'], weight_decay=1e-3)
        def lambda_topo(epoch):
            if epoch < cfg['topo_warmup_epochs']: 
                return 0.0
            if epoch < cfg['topo_warmup_epochs'] + 5: 
                return (epoch - cfg['topo_warmup_epochs']) / 5.0
            return 1.0
        sched_topo = optim.lr_scheduler.LambdaLR(opt_topo, lr_lambda=lambda_topo)
        logging.info(f"Topological parameters: {sum(p.numel() for p in topo_params):,}")
        logging.info(f"Warmup epochs for topology: {cfg['topo_warmup_epochs']}")
    
    supcon = SupConLoss(temperature=0.07)
    best_acc = 0
    start_epoch = 1
    
    if cfg.get('resume', False):
        checkpoint_data, loaded = cargar_checkpoint()
        if loaded:
            model.load_state_dict(checkpoint_data['model_state'])
            if 'optimizers' in checkpoint_data:
                opt_main.load_state_dict(checkpoint_data['optimizers']['main'])
                sched_main.load_state_dict(checkpoint_data['schedulers']['main'])
                if opt_topo and checkpoint_data['optimizers']['topo']:
                    opt_topo.load_state_dict(checkpoint_data['optimizers']['topo'])
                    sched_topo.load_state_dict(checkpoint_data['schedulers']['topo'])
            else:
                logging.warning("⚠️ Legacy checkpoint: Solo se cargó main optimizer.")
                opt_main.load_state_dict(checkpoint_data['optimizer_state'])

            start_epoch = checkpoint_data['epoch'] + 1
            model.global_step = checkpoint_data.get('global_step', 0)
            logging.info(f"🔄 Resumed from epoch {start_epoch}, global_step {model.global_step}")
    
    try:
        for epoch in range(start_epoch, cfg['epochs'] + 1):
            model.train()
            meters = {'loss': 0, 'acc': 0}
            
            for batch_idx, (x, y) in enumerate(train_loader):
                if batch_idx % 10 == 0: ResourceMonitor.check_memory_limit(cfg['memory_limit_gb'])
                
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                # ✅ FIX: Desactivar compile durante PGD si está activado (evita recompilación)
                if cfg['use_compile']:
                    # Guardar referencia compilada
                    compiled_forward = model.forward
                    # Usar forward no compilado para PGD
                    model.forward = model.__class__.forward
                
                x_adv = make_adversarial_pgd(model, x, y, ROBUST_CONFIG['train_eps'], 
                                           ROBUST_CONFIG['pgd_steps_train'])
                
                # ✅ Restaurar forward compilado
                if cfg['use_compile']:
                    model.forward = compiled_forward
                
                if opt_topo: opt_topo.zero_grad()
                opt_main.zero_grad()
                
                # ✅ Llamada al modelo con forward compilado (si está activado)
                logits, proj, entropy = model(x_adv)
                
                loss = F.cross_entropy(logits, y)
                if cfg['use_supcon'] and proj is not None:
                    loss += LAMBDAS['supcon'] * supcon(proj, y).mean()
                loss += LAMBDAS['entropy'] * entropy
                
                if cfg['use_plasticity']:
                    sparse_loss = torch.norm(torch.sigmoid(model.adj_weights), 1)
                    loss += LAMBDAS['sparsity'] * sparse_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_main.step()
                if opt_topo: opt_topo.step()
                
                meters['loss'] += loss.item()
                meters['acc'] += logits.argmax(1).eq(y).sum().item()
            
            sched_main.step()
            if sched_topo: sched_topo.step()
            
            sparsity = 0.0
            mean_weight = 0.0
            topo_lr = "N/A"
            if cfg['use_plasticity']:
                adj_w = torch.sigmoid(model.adj_weights)
                sparsity = (adj_w > 0.5).float().mean().item()
                mean_weight = adj_w.mean().item()
                if sched_topo: topo_lr = f"{sched_topo.get_last_lr()[0]:.5f}"
                
                if sparsity > 0.95:
                    warnings.warn(f"⚠️ TOPOLOGY COLLAPSE: Sparsity={sparsity:.2%}. Reducir LAMBDA_sparsity")
                if sparsity < 0.01 and epoch > cfg['topo_warmup_epochs']:
                    warnings.warn(f"⚠️ TOPOLOGY PRUNED: Sparsity={sparsity:.2%}. Aumentar LAMBDA_sparsity")
            
            epoch_acc = 100 * meters['acc'] / len(train_loader.dataset)
            
            logging.info(f"Ep {epoch:02d} | Loss: {meters['loss']/len(train_loader):.4f} | "
                       f"Acc: {epoch_acc:.2f}% | Sparsity: {sparsity:.2%} | "
                       f"AvgWeight: {mean_weight:.3f} | TopoLR: {topo_lr}")
            
            if epoch % cfg['checkpoint_interval'] == 0 or epoch == cfg['epochs']:
                save_topology_snapshot(model, epoch, run_name)
                
                guardar_checkpoint({
                    'model_state': model.state_dict(),
                    'optimizers': {
                        'main': opt_main.state_dict(),
                        'topo': opt_topo.state_dict() if opt_topo else None
                    },
                    'schedulers': {
                        'main': sched_main.state_dict(),
                        'topo': sched_topo.state_dict() if sched_topo else None
                    },
                    'epoch': epoch,
                    'metrics': meters,
                    'global_step': model.global_step
                })
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    guardar_checkpoint({
                        'model_state': model.state_dict(),
                        'optimizers': {
                            'main': opt_main.state_dict(),
                            'topo': opt_topo.state_dict() if opt_topo else None
                        },
                        'schedulers': {
                            'main': sched_main.state_dict(),
                            'topo': sched_topo.state_dict() if sched_topo else None
                        },
                        'epoch': epoch,
                        'metrics': meters,
                        'global_step': model.global_step
                    }, f"{CHECKPOINT_FILE}.best")
        
        # ✅ Evaluación final
        logging.info("\n🎯 Final evaluation...")
        model.eval()
        
        correct = 0
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad(): 
                correct += model(x)[0].argmax(1).eq(y).sum().item()
        clean_acc = 100 * correct / len(test_loader.dataset)
        
        correct = 0
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = make_adversarial_pgd(model, x, y, ROBUST_CONFIG['test_eps'], 
                                       ROBUST_CONFIG['pgd_steps_test'])
            with torch.no_grad(): 
                correct += model(x_adv)[0].argmax(1).eq(y).sum().item()
        pgd_acc = 100 * correct / len(test_loader.dataset)
        
        aa_acc = eval_autoattack(model, test_loader, n_samples=ROBUST_CONFIG['autoattack_n'])
        
        final_results = {
            'clean_acc': clean_acc,
            'pgd_acc': pgd_acc,
            'aa_acc': aa_acc if aa_acc != -1 else None,
            'best_train_acc': best_acc,
            'final_sparsity': sparsity if cfg['use_plasticity'] else 0.0,
        }
        
        logging.info("="*80)
        logging.info("FINAL RESULTS:")
        logging.info(f"Clean Accuracy: {clean_acc:.2f}%")
        logging.info(f"PGD-{ROBUST_CONFIG['pgd_steps_test']} Accuracy: {pgd_acc:.2f}%")
        logging.info(f"AutoAttack: {aa_acc:.2f}%" if aa_acc != -1 else "AA: N/A")
        logging.info("="*80)
        
        with open(f"results/{run_name}/final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        plot_topology_evolution(run_name)
        ResourceMonitor.clear_cache()
        return clean_acc, pgd_acc, aa_acc
        
    except RuntimeError as e:
        if "memory limit" in str(e):
            logging.error(f"🚨 Training aborted due to memory limit: {e}")
        else:
            logging.exception("💥 Critical error during training")
        raise
    except Exception as e:
        logging.exception("💥 Unexpected error")
        raise

# =============================================================================
# 6. SUITE DIAGNÓSTICA
# =============================================================================

def run_diagnostic_suite():
    experiments = [
        {'name': 'Baseline_AT', 'config': {'use_supcon': False, 'use_mgf': False, 'use_plasticity': False}},
        {'name': 'TopoBrain_SupConOnly', 'config': {'use_supcon': True, 'use_mgf': False, 'use_plasticity': False}},
        {'name': 'TopoBrain_TopoOnly', 'config': {'use_supcon': False, 'use_mgf': True, 'use_plasticity': True}},
        {'name': 'TopoBrain_Full', 'config': {'use_supcon': True, 'use_mgf': True, 'use_plasticity': True}}
    ]
    
    seeds = [42]; results = []
    
    print("="*80)
    print("TOPOBRAIN v16.5 FUSION - FINAL ABLATION SUITE")
    print("="*80)
    print(f"Mode: {'FAST' if USE_FAST_MODE else 'PAPER'}")
    print(f"Batch Size: {GLOBAL_SETTINGS['batch_size']}")
    print(f"Grid Size: {GLOBAL_SETTINGS['grid_size']}x{GLOBAL_SETTINGS['grid_size']}")
    print("="*80)
    
    for exp in experiments:
        logging.info(f"\n{'='*60}\n>>> EXPERIMENT: {exp['name']}\n{'='*60}")
        
        for seed in seeds:
            exp['config']['seed'] = seed
            exp['config']['resume'] = False
            
            try:
                c, p, a = run_training(exp['config'], f"{exp['name']}_s{seed}")
                results.append({'name': exp['name'], 'clean': c, 'pgd': p, 'aa': a})
                logging.info(f"✅ Completed: {exp['name']} (s{seed})")
            except Exception as e:
                logging.error(f"❌ Failed: {exp['name']} - {e}")
                results.append({'name': exp['name'], 'error': str(e)})
            
            ResourceMonitor.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                time.sleep(5)
    
    # ✅ Análisis comparativo robusto
    print("\n" + "="*80)
    print("📊 ABLATION ANALYSIS")
    print("="*80)
    
    baseline = next((r for r in results if r['name'] == 'Baseline_AT'), None)
    topo_only = next((r for r in results if r['name'] == 'TopoBrain_TopoOnly'), None)
    
    if baseline and topo_only and 'error' not in topo_only:
        clean_delta = topo_only['clean'] - baseline['clean']
        pgd_delta = topo_only['pgd'] - baseline['pgd']
        
        print(f"TopoBrain_TopoOnly vs Baseline_AT:")
        print(f"  Clean Δ: {clean_delta:+.2f}%")
        print(f"  PGD Δ:   {pgd_delta:+.2f}%")
        
        if pgd_delta > 1.0:
            print("🎉 SUCCESS: Topology significantly improves robustness!")
        elif pgd_delta > 0:
            print("✅ POSITIVE: Topology helps slightly.")
        else:
            print("⚠️ NEUTRAL: Topology doesn't help. Redesign needed.")
    
    with open("ablation_results.json", 'w') as f:
        json.dump({
            'results': results, 
            'timestamp': datetime.now().isoformat(),
            'config': GLOBAL_SETTINGS
        }, f, indent=2)
    
    print("\n✅ Ablation suite completed. Results saved to ablation_results.json")
    return results

# =============================================================================
# 7. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    seed_everything(GLOBAL_SETTINGS['seed'])
    
    print("="*80)
    print("TOPOBRAIN v16.5")
    print("="*80)

    
    try:
        results = run_diagnostic_suite()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        ResourceMonitor.clear_cache()
        exit(0)
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        ResourceMonitor.clear_cache()
        exit(1)
    finally:
        logging.info("Session ended")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
import numpy as np
import os
import random
import time
import json
import pickle
import psutil
import gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import networkx as nx
import seaborn as sns

# =============================================================================
# 0. CONFIGURACIÓN (v24 FUSION - INTEGRIDAD RESTAURADA)
# =============================================================================
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    dataset: str = "CIFAR10"
    batch_size: int = 64
    num_workers: int = 4
    
    grid_size: int = 6
    use_spectral: bool = True
    use_sparse_ops: bool = True
    
    use_orchestrator: bool = True
    orchestrator_hidden_dim: int = 32
    orchestrator_state_dim: int = 8
    
    use_plasticity: bool = True
    use_mgf: bool = True
    use_supcon: bool = True
    use_adaptive_topology: bool = True
    use_symbiotic: bool = True
    
    use_nested_cells: bool = True
    fast_lr: float = 0.01 
    forget_rate: float = 0.95
    
    max_nodes: int = 128
    prune_threshold: float = 0.1
    topology_update_freq: int = 2
    
    epochs: int = 30
    lr_main: float = 0.01
    lr_topo: float = 0.01
    topo_warmup_epochs: int = 10
    grad_clip_norm: float = 1.0
    accumulation_steps: int = 2
    
    train_eps: float = 8/255
    test_eps: float = 8/255
    pgd_steps_train: int = 5
    pgd_steps_test: int = 20
    
    lambda_supcon_start: float = 0.1      # FIX: Era 0.01
    lambda_supcon_end: float = 1.0        # FIX: Era 0.1
    lambda_entropy: float = 0.001
    lambda_sparsity: float = 1e-5
    lambda_ortho: float = 5e-3            # FIX: Era 1e-4 (50x más fuerte)
    ortho_weights: List[float] = None
    
    checkpoint_interval: int = 5
    memory_limit_gb: float = 12.0
    debug_mode: bool = False
    
    def __post_init__(self):
        if self.ortho_weights is None:
            self.ortho_weights = [0.05, 0.1, 0.5]
        if not hasattr(self, 'lambda_ortho'):
            self.lambda_ortho = 5e-3
    
    def to_dict(self):
        return asdict(self)
    
    def get_supcon_lambda(self, epoch: int) -> float:
        if epoch >= self.epochs:
            return self.lambda_supcon_end
        progress = epoch / self.epochs
        return self.lambda_supcon_start + (self.lambda_supcon_end - self.lambda_supcon_start) * (progress ** 0.5)

    def get_sparsity_lambda(self, epoch: int) -> float:
        warmup_epochs = 5
        rampup_epochs = 15
        
        if epoch < warmup_epochs:
            return 1e-6
        elif epoch < rampup_epochs:
            progress = (epoch - warmup_epochs) / (rampup_epochs - warmup_epochs)
            return 1e-6 + (self.lambda_sparsity - 1e-6) * (progress ** 2)
        else:
            return self.lambda_sparsity

# =============================================================================
# 1. UTILIDADES CORE (v18 INTEGRIDAD RESTAURADA)
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 1. UTILITIES (ResourceMonitor) - FIX: Protección en check_limit
# =============================================================================

class ResourceMonitor:
    @staticmethod
    def get_memory_gb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    
    @staticmethod
    def get_gpu_memory_gb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    @staticmethod
    def log(prefix: str = ""):
        ram = ResourceMonitor.get_memory_gb()
        gpu = ResourceMonitor.get_gpu_memory_gb()
        cpu = psutil.cpu_percent(interval=0.1)
        print(f"{prefix}💾 RAM: {ram:.2f}GB | GPU: {gpu:.2f}GB | CPU: {cpu:.1f}%")
    
    @staticmethod
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def check_limit(limit_gb: float, abort_on_limit: bool = True):
        # FIX: Verificar que limit_gb no es None antes de comparar
        if limit_gb is None:
            return True
        
        used = ResourceMonitor.get_memory_gb()
        if used > limit_gb:
            msg = f"⚠️ Memory limit exceeded: {used:.1f}GB > {limit_gb}GB"
            print(msg)
            ResourceMonitor.clear_cache()
            if abort_on_limit:
                raise MemoryError(msg)
            return False
        return True


# =============================================================================
# NUEVO: ORQUESTADOR PREFRONTAL (NEOCORTEX CONTROLADOR)
# =============================================================================
class PrefrontalOrchestrator(nn.Module):
    """
    Módulo de control ejecutivo que monitoriza el estado de la red y emite señales
    dinámicas de activación/inhibición para cada mecanismo neuromodulatorio.
    Opera como un sistema de homeostasis topológica y metabólica.
    FIX v24: Gestión corregida del grafo computacional recurrente (BPTT).
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Dimensión de estado: métricas clave del entrenamiento
        # [loss, grad_norm, sparsity, entropy, batch_var, epoch, memory_norm, topology_health]
        self.state_dim = config.orchestrator_state_dim
        
        # Red de decisión (MLP ligero con contexto temporal)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, config.orchestrator_hidden_dim),
            nn.LayerNorm(config.orchestrator_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.orchestrator_hidden_dim, config.orchestrator_hidden_dim),
            nn.LayerNorm(config.orchestrator_hidden_dim),
            nn.GELU()
        )
        
        # Memoria de contexto para decisiones secuenciales (fase de entrenamiento)
        self.rnn_context = nn.GRU(config.orchestrator_hidden_dim, config.orchestrator_hidden_dim // 2, 
                                  batch_first=True, num_layers=1)
        
        # Cabezales de control para cada mecanismo (salida [0,1])
        control_dim = config.orchestrator_hidden_dim // 2
        
        self.gate_plasticity = nn.Sequential(nn.Linear(control_dim, 1), nn.Sigmoid())      # Control topología
        self.gate_memory = nn.Sequential(nn.Linear(control_dim, 1), nn.Sigmoid())          # Control memoria fast weights
        self.gate_defense = nn.Sequential(nn.Linear(control_dim, 1), nn.Sigmoid())         # Control predictive coding
        self.gate_supcon = nn.Sequential(nn.Linear(control_dim, 1), nn.Sigmoid())          # Control contraste
        self.gate_symbiotic = nn.Sequential(nn.Linear(control_dim, 1), nn.Sigmoid())       # Control orthonormal refinement
        self.gate_sparsity = nn.Sequential(nn.Linear(control_dim, 1), nn.Sigmoid())        # Control regularización
        self.gate_lr = nn.Sequential(nn.Linear(control_dim, 1), nn.Sigmoid())              # Escala learning rate [0.1, 2.0]
        
        # Memoria de contexto persistente
        self.register_buffer('context_state', torch.zeros(1, 1, config.orchestrator_hidden_dim // 2))
        
    def forward(self, metrics_dict: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Input: Diccionario con métricas del estado actual
        Output: Diccionario con señales de control escaladas [0,1]
        """
        # Construir vector de estado normalizado
        state_vector = torch.tensor([
            metrics_dict.get('loss', 0.0),
            metrics_dict.get('grad_norm', 0.0),
            metrics_dict.get('sparsity', 0.5),
            metrics_dict.get('entropy', 0.5),
            metrics_dict.get('batch_var', 0.1),
            metrics_dict.get('epoch_progress', 0.0),
            metrics_dict.get('memory_norm', 1.0),
            metrics_dict.get('L_score', 2.0)
        ], dtype=torch.float32, device=self.config.device).unsqueeze(0)  # [1, state_dim]
        
        # Codificar estado
        encoded = self.state_encoder(state_vector)  # [1, hidden_dim]
        
        # Contexto temporal con manejo seguro de estado oculto
        # Se asume que detach_state() se llama externamente entre batches
        rnn_out, new_context = self.rnn_context(encoded.unsqueeze(1), self.context_state)
        self.context_state = new_context
        
        context = rnn_out.squeeze(1)  # [1, control_dim]
        
        # Generar señales de control
        controls = {
            'plasticity': self.gate_plasticity(context).squeeze(),
            'memory': self.gate_memory(context).squeeze(),
            'defense': self.gate_defense(context).squeeze(),
            'supcon': self.gate_supcon(context).squeeze(),
            'symbiotic': self.gate_symbiotic(context).squeeze(),
            'sparsity': self.gate_sparsity(context).squeeze(),
            'lr_scale': self.gate_lr(context).squeeze() * 1.9 + 0.1  # Mapear a [0.1, 2.0]
        }
        
        return controls
    
    def detach_state(self):
        """Rompe el grafo computacional para evitar retropropagación infinita entre batches"""
        if self.context_state is not None:
            self.context_state = self.context_state.detach()

    def reset_context(self):
        """Resetear contexto al inicio de cada época"""
        self.context_state = torch.zeros_like(self.context_state)



@dataclass
class TopologyMetrics:
    sparsity: float
    mean_weight: float
    rank_effective: int
    spectral_entropy: float
    L_score: float
    status: str
    action: str = ""

class TopologicalHealthSovereignty:
    """Monitor SVD Completo con criterios neurocientíficos
    
    Referencias:
    - Sporns (2016): Human brain networks ~1-3% sparsity
    - Bullmore & Sporns (2009): Small-world topology con L_score > 3
    """
    def __init__(self, model, config, epsilon_c: float = 0.1):
        self.model = model
        self.config = config
        self.epsilon_c = epsilon_c
        self.history: List[Dict[str, TopologyMetrics]] = []
        
    def _analyze_matrix(self, weight_matrix: torch.Tensor, name: str) -> TopologyMetrics:
        W = weight_matrix.detach().cpu().numpy()
        
        try:
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            S_norm = S / (np.sum(S) + 1e-10)
            spectral_entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
            rank_effective = int(np.sum(S > 0.01 * S[0]))
        except:
            spectral_entropy = 0.0
            rank_effective = 1
        
        log_rank = np.log(rank_effective + 1)
        denominador = abs(spectral_entropy - log_rank) + self.epsilon_c
        L_score = 1.0 / denominador
        
        density = (W > 0.5).mean()
        sparsity = 1.0 - density
        mean_weight = W.mean()
        
        # Diagnóstico basado en neurociencia de redes
        status = "UNKNOWN"
        action = ""
        
        # Biological sparsity: cerebro ~1-3%, permitimos 1-10% para grids pequeños
        if sparsity > 0.99:  # <1% conexiones
            status = "PRUNED/DEAD"
            action = "Topología excesivamente sparse, aumentar conectividad"
        elif sparsity < 0.90:  # >10% conexiones
            status = "HYPERDENSE"
            action = "Demasiadas conexiones, pruning requerido"
        elif L_score < 2.0:
            status = "DEGENERATE"
            action = "Estructura aleatoria sin organización small-world"
        elif 0.90 <= sparsity <= 0.99 and L_score > 3.0:
            status = "SOBERANO"
            action = "Topología biológicamente plausible"
        elif 0.90 <= sparsity <= 0.97 and 2.0 <= L_score <= 3.0:
            status = "EMERGENTE"
            action = "En proceso de organización"
        else:
            status = "LIMINAL"
            action = "Estado transicional, monitorear"
        
        return TopologyMetrics(sparsity, mean_weight, rank_effective, spectral_entropy, L_score, status, action)
            
    def calculate(self, epoch: int) -> Dict[str, TopologyMetrics]:
        results = {}
        if hasattr(self.model, 'adj_weights'):
            adj_w = torch.sigmoid(self.model.adj_weights)
            results['adjacency'] = self._analyze_matrix(adj_w, "adjacency")
        if hasattr(self.model, 'inc_weights'):
            inc_w = torch.sigmoid(self.model.inc_weights)
            results['incidence'] = self._analyze_matrix(inc_w, "incidence")
        
        for idx, layer in enumerate([self.model.layer1, self.model.layer2]):
            if hasattr(layer, 'node_importance'):
                imp = torch.sigmoid(layer.node_importance).unsqueeze(0)
                results[f'layer{idx+1}_nodes'] = self._analyze_matrix(imp, f"layer{idx+1}_nodes")
        
        self.history.append(results)
        return results
    
    def get_critical_summary(self) -> str:
        if not self.history:
            return "No topology data"
        last = self.history[-1]
        msgs = []
        for name, metrics in last.items():
            if metrics.status in ["PRUNED/DEAD", "HYPERDENSE", "DEGENERATE"]:
                msgs.append(f"{name}: {metrics.status} | {metrics.action}")
        return " | ".join(msgs) if msgs else "Topology healthy"

class CheckpointManager:
    """Manager robusto v18 con backups y metadata"""
    def __init__(self, run_name: str):
        self.checkpoint_dir = Path(f"checkpoints/{run_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "metadata.jsonl"
    
    def save(self, data: Dict, name: str):
        filepath = self.checkpoint_dir / f"{name}.ckpt"
        temp_file = filepath.with_suffix('.tmp')
        backup_file = filepath.with_suffix('.bak')
        
        try:
            # Guardar checkpoint
            torch.save(data, temp_file)
            if filepath.exists():
                filepath.replace(backup_file)
            temp_file.replace(filepath)
            
            # Guardar metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'name': name,
                'size_mb': filepath.stat().st_size / (1024**2)
            }
            with open(self.metadata_file, 'a') as f:
                f.write(json.dumps(metadata) + '\n')
            
            print(f"✅ Checkpoint guardado: {name} ({metadata['size_mb']:.1f}MB)")
        except Exception as e:
            print(f"❌ Error guardando checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def load(self, name: str) -> Optional[Dict]:
        filepath = self.checkpoint_dir / f"{name}.ckpt"
        if filepath.exists():
            try:
                return torch.load(filepath, map_location='cpu')
            except Exception as e:
                print(f"⚠️ Error cargando {name}: {e}")
                # Intentar backup
                backup = filepath.with_suffix('.bak')
                if backup.exists():
                    return torch.load(backup, map_location='cpu')
        print(f"ℹ️ No checkpoint encontrado: {name}")
        return None

# =============================================================================
# 2. DATASET (v18)
# =============================================================================
def get_dataloaders(config: Config):
    stats = {'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             'MNIST': ((0.1307,), (0.3081,))}
    mean, std = stats.get(config.dataset, stats['CIFAR10'])
    
    if config.dataset == 'CIFAR10':
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
        in_channels = 3
    else:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=tf)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=tf)
        in_channels = 1
    
    # FIX MEMORIA: Test batch size reducido drásticamente (500 -> 50) 
    # La arquitectura Nested multiplica el batch por num_nodes (6x6=36).
    # 500 * 36 = 18,000 vectores latentes simultáneos causan OOM.
    # 50 * 36 = 1,800 es seguro para 12GB+ VRAM.
    safe_test_batch = 5
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=safe_test_batch, num_workers=config.num_workers, pin_memory=True)
    return train_loader, test_loader, in_channels



# =============================================================================
# 3. COMPONENTES NEURONALES (Fusión Completa)
# =============================================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        mask = torch.eq(labels.view(-1,1), labels.view(-1,1).T).float().to(features.device)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size, device=features.device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-6)
        return -mean_log_prob_pos.mean()

# --- v18: Predictive Coding Asimétrico ---
class AsymmetricPredictiveErrorCell(nn.Module):
    def __init__(self, dim, use_spectral=True):
        super().__init__()
        layer = nn.Linear(dim, dim)
        self.surprise_processor = spectral_norm(layer) if use_spectral else layer
        self.confidence_net = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.ReLU(),
            nn.Linear(dim // 4, 1), nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, input_signal, prediction):
        surprise = input_signal - prediction
        error_magnitude = torch.norm(surprise, dim=-1, keepdim=True)
        confidence = 1.0 / (1.0 + error_magnitude)
        processed_surprise = self.surprise_processor(surprise)
        learned_confidence = self.confidence_net(torch.abs(surprise))
        total_confidence = confidence * learned_confidence
        return self.ln(processed_surprise * total_confidence + input_signal)

class LearnableAbsenceGating(nn.Module):
    def __init__(self, dim, min_gate=0.1):
        super().__init__()
        self.min_gate = min_gate
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 4), 
            nn.ReLU(),
            nn.Linear(dim // 4, dim), 
            nn.Sigmoid()
        )
    
    def forward(self, x_sensory, x_prediction):
        error = torch.abs(x_sensory - x_prediction)
        gate = self.gate_net(error)
        gate = gate * (1.0 - self.min_gate) + self.min_gate
        return x_sensory * gate


class SymbioticBasisRefinement(nn.Module):
    def __init__(self, dim, num_atoms=64):
        super().__init__()
        self.dim = dim
        self.num_atoms = min(num_atoms, dim)
        
        self.basis_atoms = nn.Parameter(torch.empty(self.num_atoms, dim))
        nn.init.orthogonal_(self.basis_atoms)
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
        self.register_buffer('ortho_update_counter', torch.tensor(0))

    def _maintain_orthogonality(self):
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(self.basis_atoms, full_matrices=False)
            self.basis_atoms.data = U @ Vt
    
    def forward(self, x):
        if self.training and self.ortho_update_counter % 10 == 0:
            self._maintain_orthogonality()
        self.ortho_update_counter += 1
        
        Q = self.query(x)
        K = self.key(self.basis_atoms)
        attn = torch.matmul(Q, K.T) * self.scale
        weights = F.softmax(attn, dim=-1)
        x_clean = torch.matmul(weights, self.basis_atoms)
        
        entropy = -torch.sum(weights * torch.log(weights + 1e-6), dim=-1).mean()
        
        # Soft orthogonality regularization term
        basis_gram = torch.mm(self.basis_atoms, self.basis_atoms.T)
        identity = torch.eye(basis_gram.size(0), device=basis_gram.device)
        ortho_deviation = torch.norm(basis_gram - identity, p='fro') ** 2
        ortho_deviation = torch.clamp(ortho_deviation, 0.0, 10.0)
        
        return x_clean, entropy, ortho_deviation


# =============================================================================
# 3. COMPONENTES NEURONALES (ContinuumMemoryCell FIX)
# =============================================================================
class ContinuumMemoryCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, fast_lr=0.1, forget_rate=0.9, use_spectral=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fast_lr = fast_lr
        self.forget_rate = forget_rate
        
        l1 = nn.Linear(input_dim, hidden_dim, bias=False)
        l2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_slow = spectral_norm(l1) if use_spectral else l1
        self.V_slow = spectral_norm(l2) if use_spectral else l2
        
        nn.init.orthogonal_(self.V_slow.weight)
        if hasattr(self.V_slow, 'bias') and self.V_slow.bias is not None:
            nn.init.zeros_(self.V_slow.bias)
        
        self.forget_gate_net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.update_gate_net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.semantic_mix_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.state_norm = nn.LayerNorm([hidden_dim, input_dim])
        
        self.register_buffer('semantic_memory', torch.zeros(hidden_dim, input_dim).float())
        nn.init.orthogonal_(self.semantic_memory)
        self.semantic_memory.data *= 0.01
        
        self.register_buffer('consolidation_rate', torch.tensor(0.90))
        self.register_buffer('consolidation_warmup_steps', torch.tensor(0))
        self.register_buffer('memory_initialized', torch.tensor(0))

    def forward(self, x, state_M=None, controls=None):
        """
        FIX: Ahora acepta señales de control del Orquestador para modular
        la plasticidad y consolidación en tiempo real.
        """
        batch_size = x.size(0)
        v = self.V_slow(x)
        
        if state_M is None:
            state_M = self.semantic_memory.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        if state_M.size(0) != batch_size:
            state_M = self.semantic_memory.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        x_in = x.unsqueeze(-1)
        y_pred = torch.bmm(state_M, x_in).squeeze(-1)
        
        error = v - y_pred
        error = torch.clamp(error, -1.0, 1.0)
        
        state_M_flat = state_M.reshape(batch_size, -1)
        state_summary = state_M_flat[:, :self.hidden_dim]
        gate_input = torch.cat([state_summary, x], dim=1)
        
        forget_gate = self.forget_gate_net(gate_input)
        update_gate = self.update_gate_net(gate_input)
        
        # Aplicar control del Orquestador [0,1]
        if controls is not None:
            plasticity_factor = controls.get('memory', 1.0)
            forget_gate = forget_gate * plasticity_factor  # Reducir olvido si plasticidad baja
            update_gate = update_gate * plasticity_factor  # Reducir updates si plasticidad baja
        
        forget_gate = forget_gate.unsqueeze(-1).unsqueeze(-1)
        update_gate = update_gate.unsqueeze(-1).unsqueeze(-1)
        
        delta = torch.bmm(error.unsqueeze(-1), x.unsqueeze(1))
        delta_norm = delta.norm(dim=[1, 2], keepdim=True).clamp(min=1e-6)
        delta = delta / delta_norm
        delta_4d = delta.unsqueeze(1)
        
        state_M_4d = state_M.unsqueeze(1)
        state_M_forgotten = forget_gate * state_M_4d
        state_M_update = update_gate * (self.fast_lr * delta_4d)
        state_M_new = (state_M_forgotten + state_M_update).squeeze(1)
        
        state_norm_val = state_M_new.norm(dim=[1, 2], keepdim=True).clamp(min=1.0)
        state_M_new = state_M_new / state_norm_val * 1.5
        
        if self.training and torch.rand(1).item() < 0.05 and state_M_new.shape[-1] == self.input_dim:
            state_M_new = state_M_new.reshape(-1, self.hidden_dim, self.input_dim)
            state_M_new = self.state_norm(state_M_new)
        
        # Consolidación controlada por Orquestador
        if self.training and controls is not None:
            consolidation_factor = controls.get('memory', 0.5)
            with torch.no_grad():
                mean_episodic = state_M_new.mean(dim=0)
                self.semantic_memory.data = (
                    self.consolidation_rate * self.semantic_memory.data +
                    (1 - self.consolidation_rate) * mean_episodic.data * consolidation_factor
                ).float()
        
        semantic_influence = self.semantic_mix_gate(v)
        output = semantic_influence * v + (1 - semantic_influence) * y_pred
        
        # Liberar memoria explícitamente para evitar OOM
        del state_M_4d, state_M_forgotten, state_M_update, delta_4d, delta, state_norm_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return output, state_M_new


# =============================================================================
# 4. CAPA COMPLEJA (AdaptiveCombinatorialComplexLayer) - FIX: get_node_importance
# =============================================================================
class AdaptiveCombinatorialComplexLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, config: Config, 
                 layer_type='midbrain', layer_idx=0):
        super().__init__()
        self.num_nodes = num_nodes
        self.config = config
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        
        use_spec = config.use_spectral
        
        if config.use_nested_cells:
            self.node_mapper = ContinuumMemoryCell(in_dim, hid_dim, config.fast_lr, config.forget_rate, use_spec)
            self.cell_mapper = ContinuumMemoryCell(in_dim, hid_dim, config.fast_lr, config.forget_rate, use_spec)
        else:
            self.node_mapper = spectral_norm(nn.Linear(in_dim, hid_dim)) if use_spec else nn.Linear(in_dim, hid_dim)
            self.cell_mapper = spectral_norm(nn.Linear(in_dim, hid_dim)) if use_spec else nn.Linear(in_dim, hid_dim)
        
        self.symbiotic = SymbioticBasisRefinement(hid_dim) if config.use_symbiotic else None
        
        if layer_type == 'midbrain':
            self.pc_cell = AsymmetricPredictiveErrorCell(hid_dim, use_spec)
        else:
            self.absence_gate = LearnableAbsenceGating(hid_dim)
        
        if config.use_adaptive_topology:
            self.node_importance = nn.Parameter(torch.ones(num_nodes))
        
        self._cached_adj_sparse = None
        self._cached_inc_sparse = None
        self._cache_valid = False
            
        final = nn.Linear(hid_dim * 2, hid_dim)
        self.final_mix = spectral_norm(final) if use_spec else final
        self.norm = nn.LayerNorm(hid_dim)
        self.baseline_mixer = nn.Linear(in_dim, hid_dim)
        
        self.register_buffer('_expected_batch_size', torch.tensor(-1))

    def invalidate_sparse_cache(self):
        self._cache_valid = False

    def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name="state"):
        if state is None:
            return None
        
        if state.size(0) != batch_size:
            return None
        
        if len(state.shape) != len(expected_shape) or state.shape[1:] != expected_shape[1:]:
            return None
        
        if torch.isnan(state).any() or torch.isinf(state).any():
            return None
        
        state_norm = state.norm()
        if state_norm > 100.0:
            state = state / state_norm * 10.0
        
        return state.to(device)

    def get_node_importance(self):
        """FIX: Método faltante para obtener importancia de nodos"""
        if hasattr(self, 'node_importance'):
            return torch.sigmoid(self.node_importance).detach()
        return None

    def forward(self, x_nodes, adjacency, incidence, adj_sparse=None, inc_sparse=None, 
                prev_state_node=None, prev_state_cell=None, controls=None):
        """Forward con señales de control del Orquestador y retorno de ortho deviation"""
        batch_size = x_nodes.size(0)
        device = x_nodes.device
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            if self._expected_batch_size == -1:
                self._expected_batch_size.data = torch.tensor(batch_size)
            
            if self.config.use_adaptive_topology and hasattr(self, 'node_importance'):
                importance_gate = torch.sigmoid(self.node_importance).unsqueeze(0).unsqueeze(-1)
                if controls is not None:
                    plasticity_gate = controls.get('plasticity', 1.0)
                    importance_gate = importance_gate * plasticity_gate
                x_nodes = x_nodes * importance_gate
            
            expected_node_shape = (batch_size, self.num_nodes, 
                                self.config.grid_size if hasattr(self, 'node_mapper') else x_nodes.size(-1),
                                x_nodes.size(-1))
            
            prev_state_node = self._validate_and_fix_state(
                prev_state_node, expected_node_shape, batch_size, device, "node_state"
            )
            
            if self.config.use_plasticity:
                x_flat = x_nodes.reshape(batch_size * self.num_nodes, -1)
                state_n_in = prev_state_node.reshape(-1, *prev_state_node.shape[2:]) if prev_state_node is not None else None
                
                if self.config.use_nested_cells:
                    h0_flat, new_state_node = self.node_mapper(x_flat, state_n_in, controls)
                    h0 = h0_flat.reshape(batch_size, self.num_nodes, -1)
                    new_state_node = new_state_node.reshape(batch_size, self.num_nodes, *new_state_node.shape[1:])
                    new_state_node = torch.clamp(new_state_node, -10.0, 10.0)
                else:
                    h0 = self.node_mapper(x_nodes)
                    new_state_node = None
                
                if self.config.use_sparse_ops and adj_sparse is not None:
                    if not self._cache_valid or self._cached_adj_sparse is None:
                        self._cached_adj_sparse = adj_sparse
                        self._cache_valid = True
                    
                    h0_reshaped = h0.transpose(0, 1).reshape(self.num_nodes, -1)
                    h0_agg_reshaped = torch.sparse.mm(self._cached_adj_sparse, h0_reshaped)
                    h0_agg = h0_agg_reshaped.reshape(self.num_nodes, batch_size, -1).transpose(0, 1)
                else:
                    h0_agg = torch.matmul(adjacency, h0)
            else:
                h0_agg = self.baseline_mixer(x_nodes)
                new_state_node = None

            new_state_cell = None
            entropy = torch.tensor(0.0, device=device)
            ortho_deviation = torch.tensor(0.0, device=device)
            
            if self.config.use_mgf:
                if self.config.use_sparse_ops and inc_sparse is not None:
                    if not self._cache_valid or self._cached_inc_sparse is None:
                        self._cached_inc_sparse = inc_sparse
                    
                    x_reshaped = x_nodes.transpose(0, 1).reshape(self.num_nodes, -1)
                    cell_input_reshaped = torch.sparse.mm(self._cached_inc_sparse.t(), x_reshaped)
                    num_cells = self._cached_inc_sparse.size(1)
                    cell_input = cell_input_reshaped.reshape(num_cells, batch_size, -1).transpose(0, 1)
                else:
                    inc_T_batch = incidence.T.unsqueeze(0).expand(batch_size, -1, -1)
                    cell_input = torch.bmm(inc_T_batch, x_nodes)
                
                num_cells = cell_input.size(1)
                
                expected_cell_shape = (batch_size, num_cells,
                                    self.config.grid_size if hasattr(self, 'cell_mapper') else cell_input.size(-1),
                                    cell_input.size(-1))
                
                prev_state_cell = self._validate_and_fix_state(
                    prev_state_cell, expected_cell_shape, batch_size, device, "cell_state"
                )
                
                c_flat = cell_input.reshape(batch_size * num_cells, -1)
                state_c_in = prev_state_cell.reshape(-1, *prev_state_cell.shape[2:]) if prev_state_cell is not None else None
                
                if self.config.use_nested_cells:
                    h2_flat, new_state_cell = self.cell_mapper(c_flat, state_c_in, controls)
                    h2 = h2_flat.reshape(batch_size, num_cells, -1)
                    new_state_cell = new_state_cell.reshape(batch_size, num_cells, *new_state_cell.shape[1:])
                    new_state_cell = torch.clamp(new_state_cell, -10.0, 10.0)
                else:
                    h2 = self.cell_mapper(cell_input, controls)
                
                if self.symbiotic:
                    if controls is not None:
                        symbiotic_gate = controls.get('symbiotic', 1.0)
                        pred_cells, entropy, ortho_dev = self.symbiotic(h2)
                        entropy = entropy * symbiotic_gate
                        ortho_deviation = ortho_dev * symbiotic_gate
                    else:
                        pred_cells, entropy, ortho_deviation = self.symbiotic(h2)
                else:
                    pred_cells = h2
                
                if self.config.use_sparse_ops and inc_sparse is not None:
                    pred_cells_reshaped = pred_cells.transpose(0, 1).reshape(num_cells, -1)
                    pred_nodes_reshaped = torch.sparse.mm(self._cached_inc_sparse, pred_cells_reshaped)
                    pred_nodes = pred_nodes_reshaped.reshape(self.num_nodes, batch_size, -1).transpose(0, 1)
                else:
                    pred_nodes = torch.matmul(incidence, pred_cells)
            else:
                pred_nodes = torch.zeros_like(h0_agg)
            
            if self.config.use_mgf:
                if self.layer_type == 'midbrain':
                    if controls is not None and controls.get('defense', 1.0) > 0.3:
                        processed = self.pc_cell(h0_agg, pred_nodes)
                    else:
                        processed = h0_agg
                else:
                    processed = self.absence_gate(h0_agg, pred_nodes)
            else:
                processed = h0_agg
                
            combined = torch.cat([processed, pred_nodes], dim=-1)
            out = self.final_mix(combined)
            
            return self.norm(out), entropy, ortho_deviation, new_state_node, new_state_cell


# =============================================================================
# 5. MODELO PRINCIPAL (TopoBrainV24) - FORWARD FIX
# =============================================================================
class TopoBrainV24(nn.Module):
    def __init__(self, config: Config, in_channels: int = 3):
        super().__init__()
        self.config = config
        self.current_epoch = 0
        grid = config.grid_size
        num_nodes = grid * grid
        
        img_size = 32 if config.dataset == 'CIFAR10' else 28
        patch_size = img_size // grid
        if patch_size < 1: patch_size = 1
        
        print(f"\n🧬 TopoBrain v24 [TRUE FUSION]:")
        print(f"   Structure: v18 Robust (MGF/PC/Gating/Sparsity)")
        print(f"   Engine: {'Google Nested (Fast Weights)' if config.use_nested_cells else 'Static Linear'}")
        print(f"   Grid: {grid}x{grid} | Patch: {patch_size}")
        print(f"   Orchestrator: {'Enabled' if config.use_orchestrator else 'Disabled'}")
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        
        adj, inc = self._init_grid_topology(grid)
        self.register_buffer('adj_mask', adj > 0)
        self.register_buffer('inc_mask', inc > 0)
        
        if config.use_sparse_ops:
            self.register_buffer('adj_base_sparse', adj.to_sparse().float())
            self.register_buffer('inc_base_sparse', inc.to_sparse().float())
        
        if config.use_plasticity:
            self.adj_weights = nn.Parameter(torch.randn_like(adj) - 1.5)
            self.inc_weights = nn.Parameter(torch.randn_like(inc) - 1.5)
            self.adj_weights.data += (adj * 5.0)
        else:
            self.register_buffer('adj_weights', (adj * 5.0).float())
            self.register_buffer('inc_weights', (inc * 5.0).float())

        self.layer1 = AdaptiveCombinatorialComplexLayer(
            in_dim=64, hid_dim=128, num_nodes=num_nodes, 
            config=config, layer_type='midbrain', layer_idx=0
        )
        self.layer2 = AdaptiveCombinatorialComplexLayer(
            in_dim=128, hid_dim=256, num_nodes=num_nodes,
            config=config, layer_type='thalamus', layer_idx=1
        )
        
        self.orchestrator = PrefrontalOrchestrator(config) if config.use_orchestrator else None
        
        self.readout = nn.Linear(256 * num_nodes, 10)
        self.proj_head = nn.Sequential(
            nn.Linear(256 * num_nodes, 256), nn.ReLU(), nn.Linear(256, 128)
        ) if config.use_supcon else None

    def initialize_memories(self, dataloader):
        """Inicialización de memorias semánticas con captura correcta de 5 valores de retorno"""
        print("🧠 Inicializando memorias semánticas...")
        
        self.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                if i >= 4:
                    break
                
                x = x.to(self.config.device, non_blocking=True)
                batch_size = min(8, len(x))
                x = x[:batch_size]
                
                x_embed = self.patch_embed(x).flatten(2).transpose(1, 2)
                
                if hasattr(self.layer1, 'node_mapper') and isinstance(self.layer1.node_mapper, ContinuumMemoryCell):
                    self._initialize_layer_memory(self.layer1.node_mapper, x_embed, 'layer1.node_mapper')
                
                if hasattr(self.layer1, 'cell_mapper') and isinstance(self.layer1.cell_mapper, ContinuumMemoryCell):
                    curr_adj, curr_inc, _, _ = self.get_topology(return_sparse=False)
                    inc_T = curr_inc.T.unsqueeze(0).expand(batch_size, -1, -1)
                    cell_input = torch.bmm(inc_T, x_embed)
                    self._initialize_layer_memory(self.layer1.cell_mapper, cell_input, 'layer1.cell_mapper')
                
                curr_adj, curr_inc, adj_s, inc_s = self.get_topology(return_sparse=True)
                
                # FIX: Capturar 5 valores de retorno (incluyendo ortho_deviation)
                x_l1, _, _, _, _ = self.layer1(x_embed, curr_adj, curr_inc, adj_s, inc_s)
                
                if hasattr(self.layer2, 'node_mapper') and isinstance(self.layer2.node_mapper, ContinuumMemoryCell):
                    self._initialize_layer_memory(self.layer2.node_mapper, x_l1, 'layer2.node_mapper')
                
                if hasattr(self.layer2, 'cell_mapper') and isinstance(self.layer2.cell_mapper, ContinuumMemoryCell):
                    inc_T = curr_inc.T.unsqueeze(0).expand(batch_size, -1, -1)
                    cell_input = torch.bmm(inc_T, x_l1)
                    self._initialize_layer_memory(self.layer2.cell_mapper, cell_input, 'layer2.cell_mapper')
        
        print("✅ Inicialización completa\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _initialize_layer_memory(self, cell, x_input, name):
        if cell.memory_initialized == 1:
            return
        
        batch_size, num_nodes, feat_dim = x_input.shape
        
        x_flat = x_input.reshape(batch_size * num_nodes, -1)
        
        v = cell.V_slow(x_flat[:min(32, len(x_flat))])
        x_sample = x_flat[:min(32, len(x_flat))]
        
        accumulated_state = torch.zeros_like(cell.semantic_memory)
        
        for i in range(len(v)):
            delta = torch.mm(v[i:i+1].T, x_sample[i:i+1])
            accumulated_state += delta
        
        if len(v) > 0:
            accumulated_state = accumulated_state / len(v)
            accumulated_state = accumulated_state / (accumulated_state.norm() + 1e-6) * 1.0
            cell.semantic_memory.data = accumulated_state.float()
            cell.memory_initialized.data = torch.tensor(1)
        
        print(f"  ✓ {name} inicializado (norm={cell.semantic_memory.norm().item():.4f})")
        del v, x_sample, accumulated_state

    def consolidate_semantic_memories(self):
        if not self.config.use_nested_cells:
            return
        
        total_norm = 0.0
        num_cells = 0
        
        for layer_name, layer in [('layer1', self.layer1), ('layer2', self.layer2)]:
            if hasattr(layer, 'node_mapper') and isinstance(layer.node_mapper, ContinuumMemoryCell):
                if self.current_epoch < 5:
                    target_rate = 0.90
                elif self.current_epoch < 15:
                    progress = (self.current_epoch - 5) / 10.0
                    target_rate = 0.90 + 0.09 * progress
                else:
                    target_rate = 0.99
                
                layer.node_mapper.consolidation_rate.data = torch.tensor(target_rate)
                
                semantic_norm = layer.node_mapper.semantic_memory.norm().item()
                total_norm += semantic_norm
                num_cells += 1
                
                layer.node_mapper.consolidation_warmup_steps += 1
            
            if hasattr(layer, 'cell_mapper') and isinstance(layer.cell_mapper, ContinuumMemoryCell):
                if self.current_epoch < 5:
                    target_rate = 0.90
                elif self.current_epoch < 15:
                    progress = (self.current_epoch - 5) / 10.0
                    target_rate = 0.90 + 0.09 * progress
                else:
                    target_rate = 0.99
                
                layer.cell_mapper.consolidation_rate.data = torch.tensor(target_rate)
                
                semantic_norm = layer.cell_mapper.semantic_memory.norm().item()
                total_norm += semantic_norm
                num_cells += 1
                
                layer.cell_mapper.consolidation_warmup_steps += 1
        
        avg_norm = total_norm / max(num_cells, 1)
        print(f"  🌙 Consolidation: Epoch {self.current_epoch} | Rate={target_rate:.3f} | Avg semantic norm={avg_norm:.4f}")

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        if self.orchestrator is not None:
            self.orchestrator.reset_context()

    def calculate_ortho_loss(self, ortho_deviation, controls=None):
        """Calcula loss de ortogonalidad usando el deviation retornado por las capas"""
        if controls is not None:
            symbiotic_gate = controls.get('symbiotic', 1.0) if isinstance(controls, dict) else 1.0
            ortho_deviation = ortho_deviation * symbiotic_gate
        
        return ortho_deviation

    def calculate_topology_diversity_loss(self, controls=None):
        if not self.config.use_plasticity:
            return torch.tensor(0.0, device=self.config.device)
        
        diversity_loss = 0.0
        
        adj_w = torch.sigmoid(self.adj_weights)
        inc_w = torch.sigmoid(self.inc_weights)
        
        adj_entropy = -(adj_w * torch.log(adj_w + 1e-10) + 
                        (1 - adj_w) * torch.log(1 - adj_w + 1e-10)).mean()
        inc_entropy = -(inc_w * torch.log(inc_w + 1e-10) + 
                        (1 - inc_w) * torch.log(1 - inc_w + 1e-10)).mean()
        
        entropy_target = 0.69
        
        diversity_loss = ((adj_entropy - entropy_target) ** 2 + 
                        (inc_entropy - entropy_target) ** 2)
        
        if controls is not None:
            plasticity_gate = controls.get('plasticity', 1.0) if isinstance(controls, dict) else 1.0
            diversity_loss = diversity_loss * plasticity_gate
        
        return diversity_loss

    def _init_grid_topology(self, N):
        num_nodes = N * N
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            r, c = i // N, i % N
            if r > 0: adj[i, i - N] = 1
            if r < N-1: adj[i, i + N] = 1
            if c > 0: adj[i, i - 1] = 1
            if c < N-1: adj[i, i + 1] = 1
            
        cells = []
        if N > 1:
            for r in range(N - 1):
                for c in range(N - 1):
                    tl = r * N + c
                    cells.append([tl, tl + 1, tl + N, tl + N + 1])
        num_cells = len(cells)
        inc = torch.zeros(num_nodes, max(1, num_cells))
        if num_cells > 0:
            for ci, nodes in enumerate(cells):
                for n in nodes: inc[n, ci] = 1.0
        return adj, inc

    def get_topology(self, return_sparse=False):
        adj_w = torch.sigmoid(self.adj_weights) * self.adj_mask.float()
        deg = adj_w.sum(1, keepdim=True).clamp(min=1e-6)
        curr_adj = adj_w / deg
        
        inc_w = torch.sigmoid(self.inc_weights) * self.inc_mask.float()
        deg_inc = inc_w.sum(0, keepdim=True).clamp(min=1e-6)
        curr_inc = inc_w / deg_inc
        
        if return_sparse and self.config.use_sparse_ops:
            return curr_adj, curr_inc, curr_adj.to_sparse().float(), curr_inc.to_sparse().float()
        return curr_adj, curr_inc, None, None

    def forward(self, x, prev_states=None, controls=None):
        """Forward con validación, detach explícito, y retorno de ortho deviation"""
        batch_size = x.size(0)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x_embed = self.patch_embed(x).flatten(2).transpose(1, 2)
            
            curr_adj, curr_inc, adj_s, inc_s = self.get_topology(return_sparse=True)
            
            if prev_states is not None:
                validated_states = {}
                for key, state in prev_states.items():
                    if state is not None and torch.is_tensor(state):
                        if torch.isnan(state).any() or torch.isinf(state).any():
                            validated_states[key] = None
                        elif state.size(0) != batch_size:
                            validated_states[key] = None
                        else:
                            validated_states[key] = state.detach()
                    else:
                        validated_states[key] = None
                prev_states = validated_states
            
            prev_state_l1_node = prev_states.get('layer1_node', None) if prev_states else None
            prev_state_l1_cell = prev_states.get('layer1_cell', None) if prev_states else None
            
            x_l1, entropy_l1, ortho_l1, new_state_l1_node, new_state_l1_cell = self.layer1(
                x_embed, curr_adj, curr_inc, adj_s, inc_s, 
                prev_state_l1_node, prev_state_l1_cell, controls
            )
            
            prev_state_l2_node = prev_states.get('layer2_node', None) if prev_states else None
            prev_state_l2_cell = prev_states.get('layer2_cell', None) if prev_states else None
            
            x_l2, entropy_l2, ortho_l2, new_state_l2_node, new_state_l2_cell = self.layer2(
                F.gelu(x_l1), curr_adj, curr_inc, adj_s, inc_s,
                prev_state_l2_node, prev_state_l2_cell, controls
            )
            
            x_flat = x_l2.reshape(batch_size, -1)
            logits = self.readout(x_flat)
            
            proj = None
            if self.proj_head and self.training:
                proj = self.proj_head(x_flat)
            
            total_entropy = entropy_l1 + entropy_l2
            total_ortho = ortho_l1 + ortho_l2
            
            new_states = {
                'layer1_node': new_state_l1_node.detach() if new_state_l1_node is not None else None,
                'layer1_cell': new_state_l1_cell.detach() if new_state_l1_cell is not None else None,
                'layer2_node': new_state_l2_node.detach() if new_state_l2_node is not None else None,
                'layer2_cell': new_state_l2_cell.detach() if new_state_l2_cell is not None else None
            }
            
            if torch.cuda.is_available():
                del x_embed, x_l1, x_l2, x_flat
                torch.cuda.empty_cache()
            
            return logits, proj, total_entropy, total_ortho, new_states

    def prune_topology(self, controls=None):
        """Poda topológica con cálculo correcto de quantile"""
        if not self.config.use_adaptive_topology:
            return
        
        with torch.no_grad():
            epoch_progress = min(1.0, self.current_epoch / self.config.epochs)
            target_sparsity = 0.97 + 0.02 * epoch_progress
            
            adj_w = torch.sigmoid(self.adj_weights)
            current_density = (adj_w > 0.5).float().mean().item()
            current_sparsity = 1.0 - current_density
            
            if current_sparsity < target_sparsity:
                # FIX: Quantile correcto para preservar solo el top (1-target_sparsity)%
                preserve_ratio = 1.0 - target_sparsity
                dynamic_threshold = torch.quantile(adj_w, 1.0 - preserve_ratio).item()
            else:
                dynamic_threshold = self.config.prune_threshold * (0.3 if self.current_epoch < 5 else 1.0)
            
            plasticity_gate = 1.0
            if controls is not None and hasattr(controls, 'get'):
                plasticity_gate = controls.get('plasticity', 1.0)
                if isinstance(plasticity_gate, torch.Tensor):
                    plasticity_gate = plasticity_gate.cpu().item()
                dynamic_threshold = dynamic_threshold * (2.0 - plasticity_gate)
            
            important_edges = adj_w > dynamic_threshold
            
            degree = adj_w.sum(1)
            hub_threshold = torch.quantile(degree, 0.80)
            is_hub = degree > hub_threshold
            
            for hub_idx in torch.where(is_hub)[0]:
                k_preserve = max(4, int(self.config.grid_size * 0.5))
                _, top_neighbors = torch.topk(adj_w[hub_idx], k_preserve)
                important_edges[hub_idx, top_neighbors] = True
                important_edges[top_neighbors, hub_idx] = True
            
            self.adj_mask.data &= important_edges
            
            grid_size = self.config.grid_size
            min_degree = 2
            degree_after = self.adj_mask.sum(1)
            low_degree_nodes = degree_after < min_degree
            
            if low_degree_nodes.any():
                for i in torch.where(low_degree_nodes)[0]:
                    row, col = i // grid_size, i % grid_size
                    neighbors = []
                    if row > 0: neighbors.append(i - grid_size)
                    if row < grid_size - 1: neighbors.append(i + grid_size)
                    if col > 0: neighbors.append(i - 1)
                    if col < grid_size - 1: neighbors.append(i + 1)
                    
                    current_connections = self.adj_mask[i].sum().item()
                    needed_connections = min_degree - int(current_connections)
                    
                    if needed_connections > 0 and len(neighbors) > 0:
                        neighbor_weights = adj_w[i, neighbors]
                        _, top_neighbors_idx = torch.topk(neighbor_weights, min(needed_connections, len(neighbors)))
                        
                        for idx in top_neighbors_idx:
                            neighbor = neighbors[idx]
                            self.adj_mask[i, neighbor] = True
                            self.adj_mask[neighbor, i] = True
            
            min_density = 0.01
            current_density_after = self.adj_mask.float().mean().item()
            
            if current_density_after < min_density:
                num_edges_needed = int(min_density * self.adj_mask.numel() - self.adj_mask.sum().item())
                
                if num_edges_needed > 0:
                    pruned_edges = ~self.adj_mask & (adj_w > 0)
                    if pruned_edges.any():
                        weights_pruned = torch.where(pruned_edges, adj_w, torch.tensor(-1.0, device=adj_w.device))
                        _, top_indices = torch.topk(weights_pruned.flatten(), min(num_edges_needed, pruned_edges.sum().item()))
                        row_idx = top_indices // self.adj_mask.size(1)
                        col_idx = top_indices % self.adj_mask.size(1)
                        self.adj_mask[row_idx, col_idx] = True
            
            for layer in [self.layer1, self.layer2]:
                if hasattr(layer, 'node_importance'):
                    importance = torch.sigmoid(layer.node_importance)
                    node_threshold = torch.quantile(importance, 0.50)
                    layer.node_importance.data = torch.where(
                        importance > node_threshold,
                        layer.node_importance.data,
                        torch.full_like(layer.node_importance.data, -5.0)
                    )
                
                if hasattr(layer, 'invalidate_sparse_cache'):
                    layer.invalidate_sparse_cache()
            
            final_density = self.adj_mask.float().mean().item()
            final_sparsity = 1.0 - final_density
            print(f"  ✂️  Pruning: Threshold={dynamic_threshold:.3f} |  Sparsity: {current_sparsity:.4f} -> {final_sparsity:.4f}")

            


# =============================================================================
# 7. ANÁLISIS COMPLETO (v18 RESTAURADO)
# =============================================================================

def save_topology_visualization(model, epoch, run_name):
    """Visualización v18 completa"""
    if not model.config.use_plasticity:
        return
    
    with torch.no_grad():
        adj_w = torch.sigmoid(model.adj_weights).cpu().numpy()
        inc_w = torch.sigmoid(model.inc_weights).cpu().numpy()
    
    results_dir = Path(f"results/{run_name}")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(adj_w, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title(f'Adjacency Matrix (Epoch {epoch})')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Node')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(inc_w, cmap='plasma', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(f'Incidence Matrix (Epoch {epoch})')
    axes[1].set_xlabel('Cell')
    axes[1].set_ylabel('Node')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(results_dir / f"topology_epoch{epoch:03d}.png", dpi=150)
    plt.close()
    
    # Métricas
    topo_metrics = {
        'epoch': epoch,
        'adj_sparsity': float((adj_w > 0.5).mean()),
        'adj_mean': float(adj_w.mean()),
        'adj_std': float(adj_w.std()),
        'inc_sparsity': float((inc_w > 0.5).mean()),
        'inc_mean': float(inc_w.mean()),
        'inc_std': float(inc_w.std())
    }
    
    with open(results_dir / "topology_metrics.jsonl", 'a') as f:
        f.write(json.dumps(topo_metrics) + '\n')

def save_node_importance_viz(model, epoch, run_name):
    """Visualización de importancia de nodos v18"""
    results_dir = Path(f"results/{run_name}")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (layer, ax) in enumerate(zip([model.layer1, model.layer2], axes)):
        importance = layer.get_node_importance()
        if importance is not None:
            importance_2d = importance.cpu().numpy().reshape(model.config.grid_size, model.config.grid_size)
            
            im = ax.imshow(importance_2d, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Layer {idx+1} Node Importance')
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            plt.colorbar(im, ax=ax, label='Importance')
    
    plt.tight_layout()
    plt.savefig(results_dir / f"node_importance_epoch{epoch:03d}.png", dpi=150)
    plt.close()

def analyze_topology_clustering(model, run_name):
    """Clustering espectral v18"""
    if not model.config.use_plasticity:
        print("⚠️  Modelo no tiene topología aprendible")
        return
    
    path = Path(f"results/{run_name}")
    path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        adj = torch.sigmoid(model.adj_weights).detach().cpu().numpy()
    
    try:
        n_clusters = min(8, model.config.grid_size)
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        labels = clustering.fit_predict(adj).reshape(model.config.grid_size, -1)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(labels, cmap='tab10', interpolation='nearest')
        plt.title(f'Spectral Clustering ({n_clusters} clusters)')
        plt.colorbar(label='Cluster ID')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.tight_layout()
        plt.savefig(path / "topology_clustering.png", dpi=150)
        plt.close()
        
        cluster_data = {
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'cluster_sizes': [int((labels == i).sum()) for i in range(n_clusters)]
        }
        with open(path / "topology_clusters.json", 'w') as f:
            json.dump(cluster_data, f, indent=2)
        
        print(f"✅ Clustering guardado en {path / 'topology_clustering.png'}")
    except Exception as e:
        print(f"⚠️ Error en clustering: {e}")

def analyze_topology_flow(model, dataloader, run_name: str, num_samples=100):
    """Análisis de flujo v18"""
    if not model.config.use_plasticity:
        print("⚠️  Modelo no tiene topología aprendible")
        return
    
    model.eval()
    results_dir = Path(f"results/{run_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    node_activations_l1 = torch.zeros(model.config.grid_size ** 2, device=model.config.device)
    node_activations_l2 = torch.zeros(model.config.grid_size ** 2, device=model.config.device)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_samples // dataloader.batch_size:
                break
            
            x = x.to(model.config.device)
            x_embed = model.patch_embed(x).flatten(2).transpose(1, 2)
            
            curr_adj, curr_inc, adj_s, inc_s = model.get_topology(return_sparse=True)
            
            # Capturamos todo en una tupla y tomamos solo el primer elemento (el output)
            l1_output_tuple = model.layer1(x_embed, curr_adj, curr_inc, adj_s, inc_s)
            x_l1 = l1_output_tuple[0]
            node_activations_l1 += x_l1.abs().mean(dim=[0, 2])
            
            x_l2, _, _, _ = model.layer2(F.gelu(x_l1), curr_adj, curr_inc, adj_s, inc_s)
            node_activations_l2 += x_l2.abs().mean(dim=[0, 2])
    
    node_activations_l1 /= (num_samples // dataloader.batch_size)
    node_activations_l2 /= (num_samples // dataloader.batch_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (activations, ax, title) in enumerate([
        (node_activations_l1, axes[0], 'Layer 1'),
        (node_activations_l2, axes[1], 'Layer 2')
    ]):
        act_2d = activations.cpu().numpy().reshape(model.config.grid_size, model.config.grid_size)
        
        im = ax.imshow(act_2d, cmap='viridis')
        ax.set_title(f'{title} Information Flow')
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        plt.colorbar(im, ax=ax, label='Avg Activation Magnitude')
    
    plt.tight_layout()
    plt.savefig(results_dir / "topology_flow.png", dpi=150)
    plt.close()
    
    print(f"✅ Flujo guardado en {results_dir / 'topology_flow.png'}")

def visualize_topology_as_graph(model, run_name: str, threshold=0.3):
    """Grafo v18 con métricas"""
    if not model.config.use_plasticity:
        print("⚠️  Modelo no tiene topología aprendible")
        return
    
    results_dir = Path(f"results/{run_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        adj_w = torch.sigmoid(model.adj_weights).cpu().numpy()
    
    G = nx.Graph()
    grid = model.config.grid_size
    
    for i in range(len(adj_w)):
        G.add_node(i, pos=(i % grid, grid - i // grid - 1))
        for j in range(i + 1, len(adj_w)):
            if adj_w[i, j] > threshold:
                G.add_edge(i, j, weight=adj_w[i, j])
    
    plt.figure(figsize=(12, 12))
    pos = nx.get_node_attributes(G, 'pos')
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue', alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=[w * 3 for w in weights], alpha=0.5)
    plt.title(f'Learned Topology Graph (threshold={threshold})')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(results_dir / "topology_graph.png", dpi=150)
    plt.close()
    
    # Métricas de grafo
    if len(G.edges()) > 0:
        graph_metrics = {
            'num_edges': len(G.edges()),
            'avg_degree': float(np.mean([d for n, d in G.degree()])),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'num_components': nx.number_connected_components(G)
        }
    else:
        graph_metrics = {'error': 'No edges above threshold'}
    
    with open(results_dir / "graph_metrics.json", 'w') as f:
        json.dump(graph_metrics, f, indent=2)
    
    print(f"✅ Grafo guardado en {results_dir / 'topology_graph.png'}")
    print(f"   Métricas: {graph_metrics}")

def analyze_topology_evolution(run_name: str):
    """Análisis temporal completo v18"""
    results_dir = Path(f"results/{run_name}")
    metrics_file = results_dir / "topology_metrics.jsonl"
    
    if not metrics_file.exists():
        print(f"⚠️  No se encontró {metrics_file}")
        return
    
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    
    if not metrics:
        print("⚠️  No hay métricas para analizar")
        return
    
    epochs = [m['epoch'] for m in metrics]
    adj_sparsity = [m['adj_sparsity'] for m in metrics]
    adj_mean = [m['adj_mean'] for m in metrics]
    inc_sparsity = [m['inc_sparsity'] for m in metrics]
    inc_mean = [m['inc_mean'] for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(epochs, adj_sparsity, 'o-', linewidth=2)
    axes[0, 0].set_title('Adjacency Sparsity Evolution')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Sparsity (% > 0.5)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, adj_mean, 'o-', linewidth=2, color='orange')
    axes[0, 1].set_title('Adjacency Mean Weight Evolution')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Mean Weight')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, inc_sparsity, 'o-', linewidth=2, color='green')
    axes[1, 0].set_title('Incidence Sparsity Evolution')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Sparsity (% > 0.5)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, inc_mean, 'o-', linewidth=2, color='red')
    axes[1, 1].set_title('Incidence Mean Weight Evolution')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Mean Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "topology_evolution.png", dpi=150)
    plt.close()
    
    print(f"✅ Evolución guardada en {results_dir / 'topology_evolution.png'}")
    
    print("\n" + "="*60)
    print(f"ANÁLISIS DE TOPOLOGÍA: {run_name}")
    print("="*60)
    print(f"Adjacency Sparsity:  {adj_sparsity[0]:.2%} → {adj_sparsity[-1]:.2%}")
    print(f"Adjacency Mean:      {adj_mean[0]:.4f} → {adj_mean[-1]:.4f}")
    print(f"Incidence Sparsity:  {inc_sparsity[0]:.2%} → {inc_sparsity[-1]:.2%}")
    print(f"Incidence Mean:      {inc_mean[0]:.4f} → {inc_mean[-1]:.4f}")
    print("="*60 + "\n")

def comprehensive_topology_analysis(model, dataloader, run_name: str):
    """Suite completa de análisis v18"""
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETO DE TOPOLOGÍA")
    print("="*60 + "\n")
    
    print("1️⃣  Clustering espectral...")
    analyze_topology_clustering(model, run_name)
    
    print("2️⃣  Análisis de flujo de información...")
    analyze_topology_flow(model, dataloader, run_name)
    
    print("3️⃣  Visualización como grafo...")
    visualize_topology_as_graph(model, run_name)
    
    print("4️⃣  Evolución temporal...")
    analyze_topology_evolution(run_name)
    
    print("5️⃣  Importancia de nodos...")
    save_node_importance_viz(model, model.current_epoch, run_name)
    
    print("\n✅ Análisis completo finalizado\n")

def run_ablation_study():
    """Suite de ablación v18 completa"""
    
    experiments = [
        {
            'name': 'Baseline_Static',
            'desc': 'Modelo estático (sin Nested)',
            'config': Config(use_nested_cells=False, epochs=15)
        },
        {
            'name': 'TopoBrain_Static',
            'desc': 'v18 sin Nested',
            'config': Config(use_nested_cells=False, use_plasticity=True, use_mgf=True, use_supcon=True, epochs=15)
        },
        {
            'name': 'Nested_Only',
            'desc': 'Solo motor Nested',
            'config': Config(use_nested_cells=True, use_plasticity=False, epochs=15)
        },
        {
            'name': 'TopoBrain_v24_Full',
            'desc': 'Fusión completa v24',
            'config': Config(use_nested_cells=True, use_plasticity=True, use_mgf=True, use_supcon=True, epochs=15)
        },
        {
            'name': 'Sparse_Ops',
            'desc': 'Con sparse operations',
            'config': Config(use_nested_cells=True, use_sparse_ops=True, use_plasticity=True, epochs=15)
        },
        {
            'name': 'Adaptive_Topo',
            'desc': 'Con topología adaptativa',
            'config': Config(use_nested_cells=True, use_adaptive_topology=True, epochs=15)
        }
    ]
    
    print("\n" + "="*80)
    print("TOPOBRAIN v24 - SUITE DE ABLACIÓN COMPLETA")
    print("="*80 + "\n")
    
    results_summary = []
    
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"Experimento: {exp['name']}")
        print(f"Descripción: {exp['desc']}")
        print(f"{'='*80}\n")
        
        try:
            clean_acc, pgd_acc = train_model(exp['config'], exp['name'])
            
            results_summary.append({
                'name': exp['name'],
                'description': exp['desc'],
                'clean_acc': clean_acc,
                'pgd_acc': pgd_acc,
                'config': exp['config'].to_dict()
            })
            print(f"✅ Completado: {exp['name']}")
        except Exception as e:
            print(f"❌ Error en {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
            
            results_summary.append({
                'name': exp['name'],
                'error': str(e)
            })
        
        ResourceMonitor.clear_cache()
        time.sleep(2)
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN DE ABLACIÓN")
    print("="*80)
    print(f"{'Experimento':<30} {'Clean Acc':<12} {'PGD Acc':<12}")
    print("-"*80)
    
    for result in results_summary:
        if 'error' not in result:
            print(f"{result['name']:<30} {result['clean_acc']:>10.2f}% {result['pgd_acc']:>10.2f}%")
        else:
            print(f"{result['name']:<30} {'ERROR':>10} {result['error'][:20]:>12}")
    
    print("="*80 + "\n")
    
    # Guardar resultados
    ablation_results = {
        'timestamp': datetime.now().isoformat(),
        'results': results_summary
    }
    
    with open('ablation_results_v24.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print("✅ Resultados guardados en ablation_results_v24.json\n")
    
    return results_summary


def visualize_memory_evolution(model, epoch, run_name):
    """Visualiza evolución de memorias semánticas
    
    Crítico para entender si la consolidación hipocampal→cortical está funcionando.
    SVD spectrum revela estructura de representaciones aprendidas.
    """
    results_dir = Path(f"results/{run_name}")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for layer_idx, (layer_name, layer) in enumerate([('Layer1', model.layer1), ('Layer2', model.layer2)]):
        if not hasattr(layer, 'node_mapper') or not isinstance(layer.node_mapper, ContinuumMemoryCell):
            continue
        
        semantic_memory = layer.node_mapper.semantic_memory.detach().cpu().numpy()
        
        # Memory matrix heatmap
        ax = axes[layer_idx, 0]
        im = ax.imshow(semantic_memory, cmap='RdBu', aspect='auto', 
                      vmin=-semantic_memory.std(), vmax=semantic_memory.std())
        ax.set_title(f'{layer_name} Semantic Memory')
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Hidden Dimension')
        plt.colorbar(im, ax=ax)
        
        # SVD spectrum (estructura latente)
        ax = axes[layer_idx, 1]
        try:
            U, S, Vt = np.linalg.svd(semantic_memory, full_matrices=False)
            ax.semilogy(S, 'o-', linewidth=2, markersize=4)
            ax.set_title(f'{layer_name} Singular Values')
            ax.set_xlabel('Component')
            ax.set_ylabel('Singular Value (log)')
            ax.grid(True, alpha=0.3)
            
            # Effective rank (participación de componentes)
            S_norm = S / (S.sum() + 1e-10)
            entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
            effective_rank = np.exp(entropy)
            ax.text(0.95, 0.95, f'Eff. Rank: {effective_rank:.1f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except:
            ax.text(0.5, 0.5, 'SVD Failed', ha='center', va='center')
        
        # Memory statistics over time
        ax = axes[layer_idx, 2]
        memory_norm = np.linalg.norm(semantic_memory)
        memory_sparsity = (np.abs(semantic_memory) < 0.01).mean()
        
        stats_text = f'Norm: {memory_norm:.2f}\nSparsity: {memory_sparsity:.1%}'
        if hasattr(layer.node_mapper, 'consolidation_warmup_steps'):
            steps = layer.node_mapper.consolidation_warmup_steps.item()
            stats_text += f'\nUpdates: {steps}'
        
        ax.text(0.5, 0.5, stats_text, ha='center', va='center',
               transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title(f'{layer_name} Statistics')
        ax.axis('off')
    
    plt.suptitle(f'Semantic Memory Evolution - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / f"memory_evolution_epoch{epoch:03d}.png", dpi=150)
    plt.close()
    
    # Guardar métricas cuantitativas
    memory_metrics = {'epoch': epoch}
    for layer_idx, layer in enumerate([model.layer1, model.layer2]):
        if hasattr(layer, 'node_mapper') and isinstance(layer.node_mapper, ContinuumMemoryCell):
            mem = layer.node_mapper.semantic_memory.detach().cpu().numpy()
            try:
                U, S, Vt = np.linalg.svd(mem, full_matrices=False)
                S_norm = S / (S.sum() + 1e-10)
                entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
                effective_rank = float(np.exp(entropy))
            except:
                effective_rank = 0.0
            
            memory_metrics[f'layer{layer_idx+1}_norm'] = float(np.linalg.norm(mem))
            memory_metrics[f'layer{layer_idx+1}_sparsity'] = float((np.abs(mem) < 0.01).mean())
            memory_metrics[f'layer{layer_idx+1}_effective_rank'] = effective_rank
    
    with open(results_dir / "memory_metrics.jsonl", 'a') as f:
        f.write(json.dumps(memory_metrics) + '\n')
    
    print(f"  🧠 Memory viz saved: epoch {epoch}")



def analyze_gradient_flow(model, epoch, run_name):
    """Análisis detallado del flujo de gradientes
    
    Detecta vanishing/exploding gradients y capas muertas.
    Critical para debugging de arquitecturas con fast weights.
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())
    
    if len(ave_grads) == 0:
        print("  ⚠️  No gradients found!")
        return
    
    # Detectar problemas
    avg_of_avgs = np.mean(ave_grads)
    max_of_maxs = np.max(max_grads)
    min_of_avgs = np.min(ave_grads)
    
    ratio = max_of_maxs / (min_of_avgs + 1e-8)
    
    issues = []
    if ratio > 1000:
        issues.append(f"Gradient ratio extremo: {ratio:.0f}x")
    if max_of_maxs > 10.0:
        issues.append(f"Exploding gradients: max={max_of_maxs:.2f}")
    if avg_of_avgs < 1e-5:
        issues.append(f"Vanishing gradients: avg={avg_of_avgs:.2e}")
    
    # Identificar capas muertas
    dead_layers = [layers[i] for i, avg in enumerate(ave_grads) if avg < 1e-7]
    if dead_layers:
        issues.append(f"Dead layers: {len(dead_layers)}")
    
    if issues or epoch % 5 == 0:
        print(f"  📊 Gradient Flow: avg={avg_of_avgs:.2e} | max={max_of_maxs:.2e} | ratio={ratio:.0f}x")
        if issues:
            print(f"     ⚠️  Issues: {' | '.join(issues)}")
        if dead_layers and epoch % 10 == 0:
            print(f"     💀 Dead: {dead_layers[:3]}...")
    
    # Guardar histórico para análisis post-hoc
    if epoch % 5 == 0:
        results_dir = Path(f"results/{run_name}")
        results_dir.mkdir(exist_ok=True, parents=True)
        
        grad_data = {
            'epoch': epoch,
            'layers': layers,
            'ave_grads': ave_grads,
            'max_grads': max_grads,
            'ratio': float(ratio),
            'issues': issues
        }
        
        with open(results_dir / "gradient_flow.jsonl", 'a') as f:
            f.write(json.dumps(grad_data) + '\n')

# =============================================================================
# 6. TRAINING LOOP (FIX ABSOLTO: Separación de grafo en PGD)
# =============================================================================
def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls=None, prev_states=None):
    """
    PGD attack con congelamiento total de pesos (Protocolo de Aislamiento Sináptico).
    FIX: Captura correcta de 5 valores de retorno del forward.
    """
    was_training = model.training
    model.eval()
    
    safe_controls = None
    if controls is not None:
        safe_controls = {}
        for k, v in controls.items():
            if torch.is_tensor(v):
                safe_controls[k] = v.detach().clone()
            else:
                safe_controls[k] = v

    prev_grad_states = {}
    for name, param in model.named_parameters():
        prev_grad_states[name] = param.requires_grad
        param.requires_grad = False
    
    actual_steps = steps
    if safe_controls is not None and 'defense' in safe_controls:
        defense_val = safe_controls['defense']
        if torch.is_tensor(defense_val):
            defense_val = defense_val.item()
        actual_steps = max(3, int(steps * defense_val))

    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta = delta.detach().requires_grad_(True)
    alpha = eps / max(1, actual_steps) * 2.0
    
    try:
        for step in range(actual_steps):
            # FIX: Capturar 5 valores de retorno
            logits, _, _, _, _ = model(x + delta, None, safe_controls)
            loss = F.cross_entropy(logits, y)
            
            if delta.grad is not None:
                delta.grad.zero_()
            
            loss.backward()
            
            with torch.no_grad():
                if delta.grad is not None:
                    grad = delta.grad
                    grad_norm = grad.view(grad.size(0), -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
                    grad_norm = grad_norm.clamp(min=1e-8)
                    grad = grad / grad_norm
                    
                    delta.data = (delta.data + alpha * grad.sign()).clamp(-eps, eps)
                    delta.grad.zero_()
                    
    finally:
        for name, param in model.named_parameters():
            param.requires_grad = prev_grad_states[name]
        
        if was_training:
            model.train()
    
    return (x + delta.detach()).detach()





def evaluate(model, loader, config, adversarial=False, controls=None):
    """
    Evaluación optimizada con Gradient Shielding.
    FIX: Captura correcta de 5 valores de retorno del forward.
    """
    model.eval()
    correct = 0
    total = 0
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    eval_controls = None
    if controls is not None:
        eval_controls = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in controls.items()}

    prev_grad_states = {}
    for name, param in model.named_parameters():
        prev_grad_states[name] = param.requires_grad
        param.requires_grad = False

    try:
        for i, (x, y) in enumerate(loader):
            x, y = x.to(config.device), y.to(config.device)
            
            if adversarial:
                with torch.enable_grad():
                    x = make_adversarial_pgd(model, x, y, config.test_eps, 
                                            config.pgd_steps_test, config.dataset,
                                            controls=eval_controls)
            
            with torch.no_grad():
                # FIX: Capturar 5 valores de retorno
                logits, _, _, _, _ = model(x, None, eval_controls)
                correct += logits.argmax(1).eq(y).sum().item()
                total += y.size(0)
            
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    finally:
        for name, param in model.named_parameters():
            param.requires_grad = prev_grad_states[name]

    if total == 0: return 0.0
    return 100 * correct / total



# =============================================================================
# 6. TRAINING LOOP 
# =============================================================================
def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler=None, sparsity_lambda=None):
    """Entrenamiento homeostático con inicialización de estados"""
    model.train()
    
    # FIX: Inicializar epoch_states para evitar UnboundLocalError
    epoch_states = None
    
    metrics = {
        'loss': 0, 'acc': 0, 'supcon_loss': 0, 'ortho_loss': 0, 
        'topology_diversity': 0, 'sparsity_loss': 0,
        'plasticity_gate': 0, 'defense_gate': 0, 'memory_gate': 0, 'supcon_gate': 0,
        'symbiotic_gate': 0, 'sparsity_gate': 0, 'lr_scale_gate': 0,
        'running_loss': 0.5
    }
    
    supcon = SupConLoss() if config.use_supcon else None
    topology_warmup = epoch > config.topo_warmup_epochs
    
    grad_norm_window = []
    
    optimizer.zero_grad()
    if opt_topo: opt_topo.zero_grad()
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(config.device), y.to(config.device)
        
        controls = None
        metrics_dict = {}
        
        with torch.no_grad():
            if i % 5 == 0 and epoch_states is not None:
                for k in epoch_states:
                    if epoch_states[k] is not None:
                        epoch_states[k].mul_(0.9)
            
            sparsity = (torch.sigmoid(model.adj_weights) > 0.5).float().mean().item() if hasattr(model, 'adj_weights') else 0.5
            grad_norm_val = np.mean(grad_norm_window) if len(grad_norm_window) > 3 else 1.0
            
            metrics_dict = {
                'loss': metrics['running_loss'],
                'grad_norm': grad_norm_val,
                'sparsity': sparsity,
                'entropy': 0.5,
                'batch_var': x.var().item(),
                'epoch_progress': epoch / config.epochs,
                'memory_norm': 1.0, 
                'L_score': 2.0
            }
        
        if model.orchestrator is not None and config.use_orchestrator:
            model.orchestrator.detach_state()
            controls = model.orchestrator(metrics_dict)
            
            for k, v in controls.items():
                gate_key = f'{k}_gate'
                if gate_key in metrics:
                    metrics[gate_key] += v.detach().item()
        
        x_adv = make_adversarial_pgd(model, x, y, config.train_eps, 
                                   config.pgd_steps_train, config.dataset, controls=controls)
        
        detached_states = None
        if epoch_states is not None:
            detached_states = {k: v.detach() for k, v in epoch_states.items() if v is not None}
            
        # Forward con captura de 5 valores
        logits, proj, entropy, ortho_deviation, new_states = model(x_adv, detached_states, controls)
        
        loss = F.cross_entropy(logits, y)
        
        if config.use_supcon and supcon is not None and proj is not None:
            supcon_val = supcon(proj, y)
            gate_val = controls['supcon'] if controls else 1.0
            loss += config.get_supcon_lambda(epoch) * supcon_val * gate_val
            metrics['supcon_loss'] += supcon_val.item()
        
        loss -= config.lambda_entropy * entropy
        
        if controls is not None:
            ortho = model.calculate_ortho_loss(ortho_deviation, controls)
            loss += config.lambda_ortho * ortho
            metrics['ortho_loss'] += ortho.item()
            
            div_loss = model.calculate_topology_diversity_loss(controls)
            loss += 0.01 * div_loss
            metrics['topology_diversity'] += div_loss.item()
            
            if config.use_plasticity and topology_warmup and sparsity_lambda is not None:
                sparsity_penalty = torch.mean(torch.sigmoid(model.adj_weights))
                loss += sparsity_lambda * sparsity_penalty * controls['plasticity']
                metrics['sparsity_loss'] += sparsity_penalty.item()
        
        loss.backward()
        
        if (i + 1) % config.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            
            if opt_topo and topology_warmup:
                step_topo = True
                if controls is not None and controls['plasticity'].item() < 0.3:
                    step_topo = False
                if step_topo:
                    opt_topo.step()
            
            if controls is not None and 'lr_scale' in controls:
                lr_scale = controls['lr_scale'].item()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * lr_scale
                
                optimizer.step()
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / max(1e-4, lr_scale)
            else:
                optimizer.step()
            
            # Invalidar sparse cache después de actualizar pesos
            if config.use_sparse_ops:
                model.layer1.invalidate_sparse_cache()
                model.layer2.invalidate_sparse_cache()

            optimizer.zero_grad()
            if opt_topo: opt_topo.zero_grad()
            
            with torch.no_grad():
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm_window.append(total_norm ** 0.5)
                if len(grad_norm_window) > 10: grad_norm_window.pop(0)

        if new_states is not None:
            epoch_states = {k: v.detach() for k, v in new_states.items() if v is not None}
        if i % 100 == 0: epoch_states = None
        
        metrics['loss'] += loss.item() * config.accumulation_steps
        metrics['acc'] += logits.argmax(1).eq(y).sum().item()
        metrics['running_loss'] = 0.9 * metrics['running_loss'] + 0.1 * (loss.item() * config.accumulation_steps)
        
        if i % 50 == 0:
            ResourceMonitor.check_limit(config.memory_limit_gb)
            if controls is not None:
                print(f"    🧠 [Prefrontal | Batch {i:03d}] "
                      f"Plast:{controls['plasticity'].item():.2f} | "
                      f"Def:{controls['defense'].item():.2f} | "
                      f"Mem:{controls['memory'].item():.2f} | "
                      f"Sym:{controls.get('symbiotic', torch.tensor(0.)).item():.2f} | "
                      f"LR_Scale:{controls.get('lr_scale', torch.tensor(1.)).item():.2f}")

    n = len(loader)
    d = len(loader.dataset)
    metrics['loss'] /= n
    metrics['acc'] = 100 * metrics['acc'] / d
    metrics['supcon_loss'] /= n
    metrics['ortho_loss'] /= n
    metrics['topology_diversity'] /= n
    metrics['sparsity_loss'] /= n
    
    for k in metrics:
        if 'gate' in k: metrics[k] /= n
            
    if config.use_plasticity and monitor is not None:
        monitor.calculate(epoch)
        
    return metrics



def train_model(config: Config, run_name: str):
    """
    Training loop principal - FIX v24: Orden correcto de Scheduler y gestión de memoria.
    """
    # Al principio de train_model, después de crear el modelo:
    checkpoint_path = "ruta_a_tu_archivo_best.ckpt" # Sube el archivo que bajaste
    if os.path.exists(checkpoint_path):
        print(f"🔄 Cargando cerebro preservado de: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(ckpt['model_state'])
        # Opcional: cargar optimizador si quieres seguir exactamente igual
        # optimizer.load_state_dict(ckpt['optimizer_state'])
        print("✅ Cerebro resucitado con éxito. Continuando entrenamiento...")
        
    seed_everything(config.seed)
    os.makedirs(f"results/{run_name}", exist_ok=True)
    
    train_loader, test_loader, in_channels = get_dataloaders(config)
    model = TopoBrainV24(config, in_channels).to(config.device)
    
    # Inicialización de memorias nested si aplica
    if config.use_nested_cells:
        model.initialize_memories(train_loader)
    
    # FORZAR NO USAR AMP (sparse ops requieren FP32)
    use_amp = False
    scaler = None
    
    # Monitores y Managers
    topo_monitor = TopologicalHealthSovereignty(model, config) if config.use_plasticity else None
    checkpoint_mgr = CheckpointManager(run_name)
    
    # Separación de parámetros por grupos funcionales
    main_params = [p for n, p in model.named_parameters() if 'weights' not in n and 'importance' not in n and 'orchestrator' not in n and p.requires_grad]
    topo_params = [p for n, p in model.named_parameters() if ('weights' in n or 'importance' in n) and 'orchestrator' not in n and p.requires_grad]
    orch_params = [p for n, p in model.named_parameters() if 'orchestrator' in n and p.requires_grad]
    
    # Optimizadores
    optimizer = optim.AdamW(main_params, lr=config.lr_main, weight_decay=5e-4)
    # Scheduler principal: MultiStepLR reduce el LR en épocas específicas
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    
    # Optimizador de topología (separado para ritmos distintos)
    opt_topo = None
    sched_topo = None
    if config.use_plasticity and topo_params:
        opt_topo = optim.AdamW(topo_params, lr=config.lr_topo, weight_decay=1e-3)
        # Warmup para la topología: empieza suave
        def warmup_topo(epoch):
            return min(1.0, epoch / config.topo_warmup_epochs)
        sched_topo = optim.lr_scheduler.LambdaLR(opt_topo, lr_lambda=warmup_topo)
    
    # Optimizador del orquestador (cerebro ejecutivo)
    opt_orch = None
    if config.use_orchestrator and orch_params:
        opt_orch = optim.AdamW(orch_params, lr=0.001, weight_decay=1e-4)
    
    print(f"\n{'='*60}")
    print(f"Entrenando: {run_name}")
    print(f"Config: {config.dataset} | Grid: {config.grid_size}x{config.grid_size}")
    print(f"AMP: {'Enabled' if use_amp else 'Disabled'} (Sparse ops requieren FP32)")
    print(f"Orchestrator: {'Enabled' if config.use_orchestrator else 'Disabled'}")
    print(f"Parameters: Main={len(main_params)} | Topo={len(topo_params)} | Orch={len(orch_params)}")
    print(f"{'='*60}\n")
    
    best_acc = 0
    history = []
    
    for epoch in range(1, config.epochs + 1):
        model.set_epoch(epoch)
        
        current_sparsity_lambda = config.get_sparsity_lambda(epoch)
        
        try:
            # 1. ENTRENAMIENTO DE LA ÉPOCA (El optimizador da pasos aquí dentro)
            train_metrics = train_epoch(
                model, train_loader, optimizer, opt_topo, 
                config, epoch, monitor=topo_monitor, scaler=scaler, 
                sparsity_lambda=current_sparsity_lambda
            )
            
            # 2. OPTIMIZACIÓN DEL ORQUESTADOR (Independiente)
            if opt_orch: 
                opt_orch.step()
                opt_orch.zero_grad()
            
            # 3. ACTUALIZACIÓN DE SCHEDULERS (Siempre DESPUÉS del entrenamiento)
            # Esto corrige el warning "lr_scheduler.step() before optimizer.step()"
            scheduler.step()
            if sched_topo: sched_topo.step()
            
        except Exception as e:
            print(f"❌ Error en epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        
        # Consolidación de memoria semántica (sueño profundo simulado)
        model.consolidate_semantic_memories()
        
        # Pruning controlado por métricas de topología
        if config.use_adaptive_topology and epoch % config.topology_update_freq == 0:
            avg_plasticity = train_metrics.get('plasticity_gate', 0.5)
            # Solo prune si la plasticidad está activa (decisión del orquestador)
            if avg_plasticity > 0.2:
                model.prune_topology({'plasticity': torch.tensor(avg_plasticity)})
        
        # Logging
        log_msg = (f"Epoch {epoch:02d}/{config.epochs} | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['acc']:.2f}% | "
                  f"SupCon: {train_metrics.get('supcon_loss', 0):.4f} | "
                  f"Ortho: {train_metrics.get('ortho_loss', 0):.4f} | "
                  f"Sparsity λ: {current_sparsity_lambda:.1e}")
        
        if config.use_orchestrator:
            log_msg += (f" | Gates: P:{train_metrics['plasticity_gate']:.2f} "
                       f"D:{train_metrics['defense_gate']:.2f} M:{train_metrics['memory_gate']:.2f}")
        
        if config.use_plasticity:
            density = (torch.sigmoid(model.adj_weights) > 0.5).float().mean().item() if hasattr(model, 'adj_weights') else 0.0
            log_msg += f" | Density: {density:.2%}"
        
        print(log_msg)
        
        # Mecanismos de seguridad: Early stopping por muerte cerebral
        if train_metrics['acc'] == 0 and epoch > 3:
            print("  ⚠️  ALERTA: Accuracy 0% - Reset de emergencia...")
            # Reinicio de memorias para salir de mínimos locales
            for layer in [model.layer1, model.layer2]:
                if hasattr(layer, 'node_mapper') and isinstance(layer.node_mapper, ContinuumMemoryCell):
                    nn.init.orthogonal_(layer.node_mapper.semantic_memory)
                    layer.node_mapper.semantic_memory.data *= 0.01
                    layer.node_mapper.memory_initialized.zero_()
                if hasattr(layer, 'cell_mapper') and isinstance(layer.cell_mapper, ContinuumMemoryCell):
                    nn.init.orthogonal_(layer.cell_mapper.semantic_memory)
                    layer.cell_mapper.semantic_memory.data *= 0.01
                    layer.cell_mapper.memory_initialized.zero_()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Guardado de Checkpoints
        if train_metrics['acc'] > best_acc:
            best_acc = train_metrics['acc']
            checkpoint_data = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config.to_dict(),
                'best_acc': best_acc,
                'topology_history': topo_monitor.history if topo_monitor else [],
                'orchestrator_state': model.orchestrator.context_state if model.orchestrator else None
            }
            checkpoint_mgr.save(checkpoint_data, f"{run_name}_best")
        
        if epoch % config.checkpoint_interval == 0:
            checkpoint_mgr.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'config': config.to_dict(),
                'train_metrics': train_metrics
            }, f"{run_name}_epoch_{epoch}")
        
        # Visualizaciones periódicas
        if config.use_plasticity and epoch % 5 == 0:
            save_topology_visualization(model, epoch, run_name)
            save_node_importance_viz(model, epoch, run_name)
        
        if config.use_nested_cells and epoch % 5 == 0:
            visualize_memory_evolution(model, epoch, run_name)
        
        history.append(train_metrics)
    
    if topo_monitor is not None:
        torch.save(topo_monitor.history, f"results/{run_name}/topo_health_history.pt")
    
    print(f"\n{'='*60}\nEVALUACIÓN FINAL")
    print(f"{'='*60}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Preparar controles de evaluación (Baja defensa para velocidad, topología congelada)
    eval_controls = None
    if config.use_orchestrator:
        eval_controls = {
            'defense': torch.tensor(0.2),
            'plasticity': torch.tensor(0.0)
        }
    
    clean_acc = evaluate(model, test_loader, config, adversarial=False, controls=eval_controls)
    pgd_acc = evaluate(model, test_loader, config, adversarial=True, controls=eval_controls)
    
    print(f"✅ Clean Accuracy: {clean_acc:.2f}%")
    print(f"🛡️  PGD-{config.pgd_steps_test} Accuracy: {pgd_acc:.2f}%")
    
    if config.use_plasticity:
        comprehensive_topology_analysis(model, test_loader, run_name)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return clean_acc, pgd_acc

# =============================================================================
# 9. CLI COMPLETO (v24 CON ORQUESTADOR) - FIX: NameError en CLI y conteo de parámetros
# =============================================================================
def main():
    """CLI v24 completo con Orquestador Prefrontal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TopoBrain v24 - TRUE FUSION EDITION with Prefrontal Orchestrator')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'ablation', 'analyze', 'full-analysis'],
                       help='Modo de ejecución')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       choices=['CIFAR10', 'MNIST'],
                       help='Dataset a usar')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Tamaño de batch')
    parser.add_argument('--grid-size', type=int, default=6,
                       help='Tamaño del grid topológico')
    
    # Flags v24
    parser.add_argument('--no-plasticity', action='store_true',
                       help='Desactivar topología aprendible')
    parser.add_argument('--no-supcon', action='store_true',
                       help='Desactivar contrastive learning')
    parser.add_argument('--no-sparse', action='store_true',
                       help='Desactivar sparse operations')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Desactivar topología adaptativa')
    parser.add_argument('--no-nested', action='store_true',
                       help='Desactivar motor Nested (usar static)')
    parser.add_argument('--no-symbiotic', action='store_true',
                       help='Desactivar Symbiotic Refinement')
    parser.add_argument('--no-orchestrator', action='store_true',
                       help='Desactivar Orquestador Prefrontal')
    
    parser.add_argument('--run-name', type=str, default=None,
                       help='Nombre del experimento')
    parser.add_argument('--seed', type=int, default=42,
                       help='Semilla aleatoria')
    parser.add_argument('--debug', action='store_true',
                       help='Modo debug (épocas reducidas)')
    parser.add_argument('--lambda-ortho', type=float, default=1e-4,
                       help='Peso de regularización ortogonal')
    
    args = parser.parse_args()
    
    # Config v24 fusion
    config = Config(
        dataset=args.dataset,
        epochs=3 if args.debug else args.epochs,
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        use_plasticity=not args.no_plasticity,
        use_supcon=not args.no_supcon,
        use_sparse_ops=not args.no_sparse,
        use_adaptive_topology=not args.no_adaptive,
        use_nested_cells=not args.no_nested,
        use_symbiotic=not args.no_symbiotic,
        use_orchestrator=not args.no_orchestrator,
        seed=args.seed,
        debug_mode=args.debug,
        lambda_ortho=args.lambda_ortho
    )
    
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        components = []
        if config.use_nested_cells: components.append('nested')
        if config.use_plasticity: components.append('topo')
        if config.use_supcon: components.append('supcon')
        if config.use_sparse_ops: components.append('sparse')
        if config.use_adaptive_topology: components.append('adaptive')
        if config.use_symbiotic: components.append('symbiotic')
        if config.use_orchestrator: components.append('orchestrator')
        if not components: components.append('baseline')
        args.run_name = f"topobrain_v24_{'_'.join(components)}_{timestamp}"
    
    print("\n" + "="*80)
    print("TOPOBRAIN v24 - TRUE FUSION EDITION with PREFRONTAL ORCHESTRATOR")
    print("="*80)
    print(f"Modo: {args.mode}")
    print(f"Run: {args.run_name}")
    print(f"Device: {config.device}")
    print(f"Nested Engine: {config.use_nested_cells}")
    print(f"Sparse Ops: {config.use_sparse_ops}")
    print(f"Adaptive Topology: {config.use_adaptive_topology}")
    print(f"Orchestrator: {config.use_orchestrator}")
    
    # FIX: Contar parámetros DESPUÉS de crear el modelo
    if args.mode == 'train':
        # Cargar modelo temporalmente solo para contar parámetros
        temp_model = TopoBrainV24(config, 3 if config.dataset == 'CIFAR10' else 1)
        total_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,}")
        del temp_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("="*80 + "\n")
    
    ResourceMonitor.log("Inicial: ")
    
    try:
        if args.mode == 'train':
            clean_acc, pgd_acc = train_model(config, args.run_name)
            print(f"\n📊 Resultados finales:")
            print(f"   Clean Acc: {clean_acc:.2f}%")
            print(f"   PGD Acc: {pgd_acc:.2f}%")
            
        elif args.mode == 'ablation':
            run_ablation_study()
            
        elif args.mode == 'analyze':
            if args.run_name is None:
                print("❌ Especifica --run-name para analizar")
                return
            analyze_topology_evolution(args.run_name)
        
        elif args.mode == 'full-analysis':
            if args.run_name is None:
                print("❌ Especifica --run-name para análisis completo")
                return
            
            # Cargar modelo
            checkpoint_mgr = CheckpointManager(args.run_name)
            checkpoint = checkpoint_mgr.load(f"{args.run_name}_best")
            
            if checkpoint is None:
                print("❌ No se encontró checkpoint del modelo")
                return
            
            train_loader, test_loader, in_channels = get_dataloaders(config)
            model = TopoBrainV24(config, in_channels).to(config.device)
            model.load_state_dict(checkpoint['model_state'])
            
            # Restaurar estado del Orquestador si existe
            if config.use_orchestrator and 'orchestrator_state' in checkpoint:
                model.orchestrator.context_state = checkpoint['orchestrator_state']
            
            model.eval()
            
            comprehensive_topology_analysis(model, test_loader, args.run_name)
        
        print("\n✅ Ejecución completada exitosamente")
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrumpido por el usuario")
        ResourceMonitor.clear_cache()
        
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        ResourceMonitor.clear_cache()
        
    finally:
        ResourceMonitor.log("Final: ")


if __name__ == "__main__":
    main()
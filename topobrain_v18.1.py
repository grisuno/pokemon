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
# 0. CONFIGURACIÓN (v18)
# =============================================================================

@dataclass
class Config:
    """Configuración centralizada y reproducible"""
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Dataset
    dataset: str = "CIFAR10"
    batch_size: int = 128
    num_workers: int = 4
    
    # Modelo
    grid_size: int = 8
    use_spectral: bool = True
    use_sparse_ops: bool = True
    
    # TopoBrain Features
    use_plasticity: bool = True
    use_mgf: bool = True
    use_supcon: bool = True
    use_adaptive_topology: bool = True
    
    # Adaptive Topology
    max_nodes: int = 128
    prune_threshold: float = 0.1
    topology_update_freq: int = 5  # epochs
    
    # Entrenamiento
    epochs: int = 30
    lr_main: float = 0.1
    lr_topo: float = 0.01
    topo_warmup_epochs: int = 5
    
    # Adversarial
    train_eps: float = 8/255
    test_eps: float = 8/255
    pgd_steps_train: int = 7
    pgd_steps_test: int = 20
    
    # Regularización 🆕
    lambda_supcon_start: float = 0.01
    lambda_supcon_end: float = 0.1
    lambda_entropy: float = 0.001
    lambda_sparsity: float = 1e-5
    lambda_ortho: float = 1e-4  # Peso global ortho
    ortho_weights: List[float] = None
    
    # Sistema
    checkpoint_interval: int = 5
    memory_limit_gb: float = 8.0
    debug_mode: bool = False
    
    def __post_init__(self):
        if self.ortho_weights is None:
            self.ortho_weights = [0.1, 0.5, 1.0]  # Peso progresivo por capa
    
    def to_dict(self):
        return asdict(self)
    
    def get_supcon_lambda(self, epoch: int) -> float:
        """Schedule adaptativo para SupCon Loss"""
        if epoch >= self.epochs:
            return self.lambda_supcon_end
        progress = epoch / self.epochs
        return self.lambda_supcon_start + (self.lambda_supcon_end - self.lambda_supcon_start) * progress

# =============================================================================
# 1. UTILIDADES CORE (Sin cambios)
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    def check_limit(limit_gb: float):
        used = ResourceMonitor.get_memory_gb()
        if used > limit_gb:
            raise MemoryError(f"Memory limit exceeded: {used:.2f}GB > {limit_gb}GB")



@dataclass
class TopologyMetrics:
    sparsity: float
    mean_weight: float
    rank_effective: int
    spectral_entropy: float
    L_score: float  # Adaptado de tu monitor
    status: str
    action: str = ""

class TopologicalHealthSovereignty:
    """
    monitor_topologico
    Fusión de SovereigntyMonitor + TopologicalHealth
    < 0.5s por época | Monitorea solo adj_weights/inc_weights
    """

    def __init__(self, model, config, epsilon_c: float = 0.1):
        self.model = model
        self.config = config
        self.epsilon_c = epsilon_c
        self.history: List[TopologyMetrics] = []
        
    def _analyze_matrix(self, weight_matrix: torch.Tensor, name: str) -> TopologyMetrics:
        """Análisis SVD de matriz topológica (adj o inc)"""
        W = weight_matrix.detach().cpu().numpy()
        
        # SVD compacta para matrices pequeñas (64x64 = < 1ms)
        try:
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            S_norm = S / (np.sum(S) + 1e-10)
            spectral_entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
            rank_effective = int(np.sum(S > 0.01 * S[0]))
        except:
            spectral_entropy = 0.0
            rank_effective = 1
        
        # L_score adaptado: cuánto "control" tiene la topología
        # Alto L = matriz tiene estructura no trivial (bueno)
        # Bajo L = matriz es ruido (mal)
        log_rank = np.log(rank_effective + 1)
        denominador = abs(spectral_entropy - log_rank) + self.epsilon_c
        L_score = 1.0 / denominador
        
        # Métricas de sparsity
        sparsity = (W > 0.5).mean()
        mean_weight = W.mean()
        
        # Diagnóstico con acción concreta
        status = "CRITICAL"
        action = ""
        
        if sparsity < 0.02:
            status = "PRUNED"
            action = f"Aumenta lambda_sparsity 10x (actual: {self.config.get('lambda_sparsity', 1e-5)})"
        elif sparsity > 0.90:
            status = "COLLAPSED"
            action = f"Reduce lambda_sparsity / Ajusta prune_threshold"
        elif L_score < 2.0:
            status = "DEGENERATE"
            action = "Topología no está aprendiendo estructura. Revisar inicialización."
        elif sparsity > 0.10 and L_score > 5.0:
            status = "SOBERANO"
            action = "Óptimo. Mantén hiperparámetros."
        else:
            status = "EMERGENTE"
            action = "Monitorear. Sparsity/L_score creciendo?"
        
        return TopologyMetrics(
            sparsity=float(sparsity),
            mean_weight=float(mean_weight),
            rank_effective=rank_effective,
            spectral_entropy=float(spectral_entropy),
            L_score=float(L_score),
            status=status,
            action=action
        )
    
    def calculate(self, epoch: int) -> Dict[str, TopologyMetrics]:
        """Analiza todas las matrices topológicas del modelo"""
        results = {}
        
        # Verificar matrices de pesos topológicos
        if not hasattr(self.model, 'adj_weights'):
            return results
        
        # Análisis de adjacency
        adj_w = torch.sigmoid(self.model.adj_weights)
        results['adjacency'] = self._analyze_matrix(adj_w, "adjacency")
        
        # Análisis de incidence (si existe)
        if hasattr(self.model, 'inc_weights'):
            inc_w = torch.sigmoid(self.model.inc_weights)
            results['incidence'] = self._analyze_matrix(inc_w, "incidence")
        
        # Análisis de importancia de nodos (v18)
        if hasattr(self.model, 'layer1') and hasattr(self.model.layer1, 'node_importance'):
            node_imp = torch.sigmoid(self.model.layer1.node_importance)
            results['layer1_nodes'] = self._analyze_matrix(
                node_imp.unsqueeze(0), "layer1_nodes"  # Broadcast a [1, N]
            )
        
        self.history.append(results)
        return results
    
    def get_critical_summary(self) -> str:
        """Resumen de emergencias"""
        if not self.history:
            return "No data"
        
        last = self.history[-1]
        msgs = []
        for name, metrics in last.items():
            if metrics.status in ["PRUNED", "COLLAPSED", "DEGENERATE"]:
                msgs.append(f"{name}: {metrics.status} | {metrics.action}")
        
        return " | ".join(msgs) if msgs else "All OK"

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save(self, data: Dict, name: str):
        filepath = self.checkpoint_dir / f"{name}.ckpt"
        temp_file = filepath.with_suffix('.tmp')
        backup_file = filepath.with_suffix('.bak')
        
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if filepath.exists():
                filepath.replace(backup_file)
            
            temp_file.replace(filepath)
            
            size_mb = filepath.stat().st_size / (1024**2)
            print(f"✅ Checkpoint guardado: {filepath.name} ({size_mb:.1f}MB)")
            
        except Exception as e:
            print(f"❌ Error guardando checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def load(self, name: str) -> Optional[Dict]:
        filepath = self.checkpoint_dir / f"{name}.ckpt"
        
        for attempt in [filepath, filepath.with_suffix('.bak')]:
            if attempt.exists():
                try:
                    with open(attempt, 'rb') as f:
                        data = pickle.load(f)
                    print(f"✅ Checkpoint cargado: {attempt.name}")
                    return data
                except Exception as e:
                    print(f"⚠️ Error cargando {attempt.name}: {e}")
                    continue
        
        print(f"ℹ️ No checkpoint encontrado: {name}")
        return None

# =============================================================================
# 2. DATASET HELPERS (Sin cambios)
# =============================================================================

def get_dataset_stats(dataset_name: str):
    stats = {
        'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'MNIST': ((0.1307,), (0.3081,))
    }
    return stats[dataset_name]

def get_dataloaders(config: Config):
    mean, std = get_dataset_stats(config.dataset)
    
    if config.dataset == 'CIFAR10':
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
        in_channels = 3
        
    elif config.dataset == 'MNIST':
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=tf)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=tf)
        in_channels = 1
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=500,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, in_channels

# =============================================================================
# 3. COMPONENTES TOPOBRAIN
# =============================================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        device = features.device
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
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

# Predictive Coding Asimétrico
class AsymmetricPredictiveErrorCell(nn.Module):
    """
    Predictive Coding con manejo asimétrico de errores
    Inspirado en corteza predictiva biológica
    """
    def __init__(self, dim, use_spectral=True):
        super().__init__()
        layer = nn.Linear(dim, dim)
        self.surprise_processor = spectral_norm(layer) if use_spectral else layer
        self.confidence_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, input_signal, prediction):
        # Error asimétrico (surprise)
        surprise = input_signal - prediction
        
        # Confianza basada en magnitud del error
        error_magnitude = torch.norm(surprise, dim=-1, keepdim=True)
        confidence = 1.0 / (1.0 + error_magnitude)
        
        # Procesar surprise
        processed_surprise = self.surprise_processor(surprise)
        
        # Gate basado en confianza aprendida
        learned_confidence = self.confidence_net(torch.abs(surprise))
        
        # Combinar ambas señales de confianza
        total_confidence = confidence * learned_confidence
        
        # Error gated
        gated_error = processed_surprise * total_confidence
        
        return self.ln(gated_error + input_signal)

class LearnableAbsenceGating(nn.Module):
    """Gating basado en error de predicción"""
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
        gate = self.gate_net(error)
        return x_sensory * gate

class SymbioticBasisRefinement(nn.Module):
    """Refinamiento mediante base ortogonal aprendible"""
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

# Capa con Sparse Operations y Topología Adaptativa
class AdaptiveCombinatorialComplexLayer(nn.Module):
    """
    Capa combinatorial con:
    - Sparse tensor operations
    - Topología adaptativa
    - Predictive coding asimétrico
    """
    def __init__(self, in_dim, hid_dim, num_nodes, config: Config, 
                 layer_type='midbrain', layer_idx=0):
        super().__init__()
        self.num_nodes = num_nodes
        self.config = config
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        
        use_spec = config.use_spectral
        self.node_mapper = spectral_norm(nn.Linear(in_dim, hid_dim)) if use_spec else nn.Linear(in_dim, hid_dim)
        self.cell_mapper = spectral_norm(nn.Linear(in_dim, hid_dim)) if use_spec else nn.Linear(in_dim, hid_dim)
        
        self.symbiotic = SymbioticBasisRefinement(hid_dim)
        
        # Predictive Coding Asimétrico
        if layer_type == 'midbrain':
            self.pc_cell = AsymmetricPredictiveErrorCell(hid_dim, use_spec)
        else:
            self.absence_gate = LearnableAbsenceGating(hid_dim)
        
        # Node importance para topología adaptativa
        if config.use_adaptive_topology:
            self.node_importance = nn.Parameter(torch.ones(num_nodes))
            
        final = nn.Linear(hid_dim * 2, hid_dim)
        self.final_mix = spectral_norm(final) if use_spec else final
        self.norm = nn.LayerNorm(hid_dim)
        self.baseline_mixer = nn.Linear(in_dim, hid_dim)

    def forward(self, x_nodes, adjacency, incidence, adj_sparse=None, inc_sparse=None):
        """
        Args:
            x_nodes: [B, N, D] node features
            adjacency: [N, N] dense adjacency (fallback)
            incidence: [N, C] dense incidence (fallback)
            adj_sparse: sparse adjacency
            inc_sparse: sparse incidence
        """
        batch_size = x_nodes.size(0)
        
        # Aplicar máscara de importancia
        if self.config.use_adaptive_topology and hasattr(self, 'node_importance'):
            importance_gate = torch.sigmoid(self.node_importance).unsqueeze(0).unsqueeze(-1)
            x_nodes = x_nodes * importance_gate
        
        # Procesamiento de nodos con sparse ops
        if self.config.use_plasticity:
            h0 = self.node_mapper(x_nodes)
            
            # Usar sparse matmul si está disponible
            if self.config.use_sparse_ops and adj_sparse is not None:
                # Para batch: loop sobre batch (más eficiente que denso para sparse)
                h0_agg_list = []
                for b in range(batch_size):
                    h0_b = h0[b]  # [N, D]
                    h0_agg_b = torch.sparse.mm(adj_sparse, h0_b)
                    h0_agg_list.append(h0_agg_b)
                h0_agg = torch.stack(h0_agg_list, dim=0)
            else:
                h0_agg = torch.matmul(adjacency, h0)
        else:
            h0_agg = self.baseline_mixer(x_nodes)

        # Procesamiento de celdas (MGF) con sparse ops
        if self.config.use_mgf:
            # Usar sparse ops para incidence
            if self.config.use_sparse_ops and inc_sparse is not None:
                cell_input_list = []
                for b in range(batch_size):
                    x_b = x_nodes[b]  # [N, D]
                    # inc_T @ x = (x^T @ inc)^T
                    cell_b = torch.sparse.mm(inc_sparse.t(), x_b)
                    cell_input_list.append(cell_b)
                cell_input = torch.stack(cell_input_list, dim=0)
            else:
                inc_T_batch = incidence.T.unsqueeze(0).expand(batch_size, -1, -1)
                cell_input = torch.bmm(inc_T_batch, x_nodes)
            
            h2 = self.cell_mapper(cell_input)
            pred_cells, entropy = self.symbiotic(h2)
            
            # Back to nodes
            if self.config.use_sparse_ops and inc_sparse is not None:
                pred_nodes_list = []
                for b in range(batch_size):
                    pred_c = pred_cells[b]
                    pred_n = torch.sparse.mm(inc_sparse, pred_c)
                    pred_nodes_list.append(pred_n)
                pred_nodes = torch.stack(pred_nodes_list, dim=0)
            else:
                pred_nodes = torch.matmul(incidence, pred_cells)
        else:
            pred_nodes = torch.zeros_like(h0_agg)
            entropy = torch.tensor(0.0, device=x_nodes.device)
        
        # Fusión con predictive coding asimétrico
        if self.config.use_mgf:
            if self.layer_type == 'midbrain':
                processed = self.pc_cell(h0_agg, pred_nodes)
            else:
                processed = self.absence_gate(h0_agg, pred_nodes)
        else:
            processed = h0_agg
            
        combined = torch.cat([processed, pred_nodes], dim=-1)
        out = self.final_mix(combined)
        return self.norm(out), entropy
    
    def get_node_importance(self):
        """Retorna importancia de nodos para visualización"""
        if hasattr(self, 'node_importance'):
            return torch.sigmoid(self.node_importance).detach()
        return None

# =============================================================================
# 4. MODELO TOPOBRAIN v18
# =============================================================================

class TopoBrainNetV18(nn.Module):
    """
    TopoBrain v18: Implementa
    - Sparse tensor operations
    - Topología adaptativa
    - Predictive coding asimétrico
    - Regularización ortogonal ponderada
    """
    def __init__(self, config: Config, in_channels: int = 3):
        super().__init__()
        self.config = config
        grid = config.grid_size
        num_nodes = grid * grid
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Topología base
        adj, inc = self._init_grid_topology(grid)
        self.register_buffer('adj_mask', adj > 0)
        self.register_buffer('inc_mask', inc > 0)
        
        # Crear versiones sparse
        if config.use_sparse_ops:
            self.register_buffer('adj_base_sparse', adj.to_sparse())
            self.register_buffer('inc_base_sparse', inc.to_sparse())
        
        # Pesos topológicos
        if config.use_plasticity:
            self.adj_weights = nn.Parameter(torch.zeros_like(adj))
            self.inc_weights = nn.Parameter(torch.zeros_like(inc))
        else:
            self.register_buffer('adj_weights', torch.ones_like(adj) * 5.0)
            self.register_buffer('inc_weights', torch.ones_like(inc) * 5.0)

        # Capas combinatoriales con índices para ortho loss
        self.layer1 = AdaptiveCombinatorialComplexLayer(
            64, 128, num_nodes, config, 'midbrain', layer_idx=0
        )
        self.layer2 = AdaptiveCombinatorialComplexLayer(
            128, 256, num_nodes, config, 'thalamus', layer_idx=1
        )
        
        # Readout
        self.readout = nn.Linear(256 * num_nodes, 10)
        self.proj_head = nn.Sequential(
            nn.Linear(256 * num_nodes, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ) if config.use_supcon else None

    def _init_grid_topology(self, N):
        """Inicializa topología de grid 2D"""
        num_nodes = N * N
        adj = torch.zeros(num_nodes, num_nodes)
        
        # Conectividad 4-vecinos
        for i in range(num_nodes):
            r, c = i // N, i % N
            if r > 0: adj[i, i - N] = 1
            if r < N-1: adj[i, i + N] = 1
            if c > 0: adj[i, i - 1] = 1
            if c < N-1: adj[i, i + 1] = 1
        
        # Celdas 2D
        cells = []
        for r in range(N - 1):
            for c in range(N - 1):
                tl = r * N + c
                cells.append([tl, tl + 1, tl + N, tl + N + 1])
        
        num_cells = len(cells)
        inc = torch.zeros(num_nodes, num_cells)
        for ci, nodes in enumerate(cells):
            for n in nodes:
                inc[n, ci] = 1.0
        
        return adj, inc

    def get_topology(self, return_sparse=False):
        """
        Calcula topología actual
        
        Args:
            return_sparse: Si True, retorna versiones sparse
        """
        # Adjacency con sparsity estructural
        adj_sparse_mask = torch.zeros_like(self.adj_mask, dtype=torch.float32)
        adj_w = torch.sigmoid(self.adj_weights)
        adj_sparse_mask = torch.where(self.adj_mask, adj_w, adj_sparse_mask)
        curr_adj = F.normalize(adj_sparse_mask, p=1, dim=-1)
        
        # Incidence normalizada
        inc_sparse_mask = torch.zeros_like(self.inc_mask, dtype=torch.float32)
        inc_w = torch.sigmoid(self.inc_weights)
        inc_sparse_mask = torch.where(self.inc_mask, inc_w, inc_sparse_mask)
        curr_inc = inc_sparse_mask / (inc_sparse_mask.sum(0, keepdim=True) + 1e-6)
        
        if return_sparse and self.config.use_sparse_ops:
            # Convertir a sparse
            adj_sparse = curr_adj.to_sparse()
            inc_sparse = curr_inc.to_sparse()
            return curr_adj, curr_inc, adj_sparse, inc_sparse
        
        return curr_adj, curr_inc
    
    def calculate_ortho_loss(self):
        """
        Regularización ortogonal con pesos por capa
        """
        loss = 0
        layers_data = [
            (self.layer1, 0),  # layer_idx = 0
            (self.layer2, 1),  # layer_idx = 1
        ]
        
        for layer, idx in layers_data:
            # Obtener peso de ortho para esta capa
            ortho_weight = self.config.ortho_weights[idx] if idx < len(self.config.ortho_weights) else 1.0
            
            mappers = [
                layer.node_mapper.weight if hasattr(layer.node_mapper, 'weight') else None,
                layer.cell_mapper.weight if hasattr(layer.cell_mapper, 'weight') else None,
            ]
            
            for w in [m for m in mappers if m is not None]:
                rows, cols = w.shape
                if rows >= cols:
                    target = torch.eye(cols, device=w.device)
                    check = torch.matmul(w.T, w)
                else:
                    target = torch.eye(rows, device=w.device)
                    check = torch.matmul(w, w.T)
                
                # Aplicar peso específico de la capa
                loss += ortho_weight * torch.norm(check - target, p='fro') / (rows * cols)
        
        return loss
    
    def prune_topology(self):
        """Poda de topología basada en importancia"""
        if not self.config.use_adaptive_topology:
            return
        
        with torch.no_grad():
            # Poda más agresiva en early stages
            epoch = getattr(self, 'current_epoch', 10)  # Si no está set, usar 10
            # HOTFIX: Poda relajada en early epochs
            current_epoch = getattr(self, 'current_epoch', 1)
            if current_epoch < 5:
                # Relajar threshold en las primeras 5 épocas para permitir aprendizaje
                dynamic_threshold = self.config.prune_threshold * 0.3
            else:
                # Threshold normal después
                dynamic_threshold = self.config.prune_threshold
            # Poda basada en adj_weights
            adj_w = torch.sigmoid(self.adj_weights)
            important_edges = adj_w > dynamic_threshold
            
            # Actualizar máscara con mantenimiento de conectividad mínima
            self.adj_mask = self.adj_mask & important_edges
            
            # Forzar conectividad mínima (al menos 4 vecinos)
            if hasattr(self, 'grid_size'):
                min_connections = max(1, self.grid_size // 2)
                for i in range(self.adj_mask.size(0)):
                    if self.adj_mask[i].sum() < min_connections:
                        # Restaurar edges más fuertes
                        _, top_indices = torch.topk(adj_w[i], min_connections)
                        self.adj_mask[i, top_indices] = True
            
            # Poda en layers individuales
            for layer in [self.layer1, self.layer2]:
                if hasattr(layer, 'node_importance'):
                    importance = torch.sigmoid(layer.node_importance)
                    # Mantener al menos 60% de nodos
                    threshold = min(dynamic_threshold, importance.quantile(0.4).item())
                    layer.node_importance.data = torch.where(
                        importance > threshold,
                        layer.node_importance.data,
                        torch.full_like(layer.node_importance.data, -10.0)
                    )
            
            print(f"  🔧 Pruning aplicado (threshold={dynamic_threshold:.3f})")
            
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Obtener topología (con sparse si está habilitado)
        if self.config.use_sparse_ops:
            curr_adj, curr_inc, adj_sparse, inc_sparse = self.get_topology(return_sparse=True)
        else:
            curr_adj, curr_inc = self.get_topology(return_sparse=False)
            adj_sparse, inc_sparse = None, None
        
        # Forward con sparse ops
        x, ent1 = self.layer1(x, curr_adj, curr_inc, adj_sparse, inc_sparse)
        x = F.gelu(x)
        x, ent2 = self.layer2(x, curr_adj, curr_inc, adj_sparse, inc_sparse)
        x = F.gelu(x)
        
        flat = x.reshape(x.shape[0], -1)
        logits = self.readout(flat)
        
        proj = None
        if self.proj_head is not None:
            proj = F.normalize(self.proj_head(flat), dim=1)
        
        return logits, proj, ent1 + ent2
    # Dentro de TopoBrainNetV18, después de forward
    def set_epoch(self, epoch):
        """Permite pasar la época actual para schedules dinámicos"""
        self.current_epoch = epoch
# =============================================================================
# 5. ADVERSARIAL TRAINING (Sin cambios)
# =============================================================================

def make_adversarial_pgd(model, x, y, eps, steps, dataset_name='CIFAR10'):
    """PGD Attack"""
    was_training = model.training
    model.eval()
    
    delta = torch.empty_like(x).uniform_(-eps, eps).detach()
    x_adv = torch.clamp(x + delta, 0, 1).detach()
    
    alpha = eps * 1.25 / steps
    
    for _ in range(steps):
        x_adv.requires_grad = True
        
        with torch.enable_grad():
            logits = model(x_adv)[0]
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
        
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0, 1).detach()
        model.train(was_training)
    return x_adv

# =============================================================================
# 6. TRAINING LOOP 
# =============================================================================

def train_epoch(model, train_loader, optimizer, opt_topo, supcon, config: Config, epoch: int, topo_monitor: Optional[TopologicalHealthSovereignty] = None):
    """Entrena una época con schedule adaptativo de SupCon - CORREGIDO"""
    model.train()
    metrics = {'loss': 0, 'acc': 0, 'n_samples': 0, 'supcon_loss': 0, 'ortho_loss': 0}
    
    # Lambda dinámico más agresivo
    progress = epoch / config.epochs
    lambda_supcon = config.lambda_supcon_start + (config.lambda_supcon_end - config.lambda_supcon_start) * (progress ** 0.5)
    
    # FIX: Grad acumulator para estabilidad
    accumulation_steps = 4 if config.batch_size < 64 else 1

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(config.device), y.to(config.device)
        
        # Generar adversarios
        x_adv = make_adversarial_pgd(
            model, x, y, 
            config.train_eps, 
            config.pgd_steps_train,
            config.dataset
        )
        
        # Forward
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad()
            if opt_topo: opt_topo.zero_grad()
        
        logits, proj, entropy = model(x_adv)
        
        # Pérdida base
        loss = F.cross_entropy(logits, y)
        
        # SupCon con lambda adaptativo
        supcon_loss_val = 0
        if config.use_supcon and proj is not None:
            supcon_loss_val = supcon(proj, y).mean()
            loss += lambda_supcon * supcon_loss_val
        
        # Pérdida de entropía
        loss -= config.lambda_entropy * entropy
        
        # Regularizaciones topológicas
        ortho_loss_val = 0
        if config.use_plasticity:
            # Sparsity penalty normalizado y suave
            if config.use_plasticity:
                # Normalizar por número de edges para evitar escala
                sparsity_penalty = torch.mean(torch.sigmoid(model.adj_weights))
                loss += config.lambda_sparsity * sparsity_penalty
                
                # Ortho loss (mantener igual)
                ortho_loss_val = model.calculate_ortho_loss()
                loss += config.lambda_ortho * ortho_loss_val
            # Ortho loss ponderado
            ortho_loss_val = model.calculate_ortho_loss()
            loss += config.lambda_ortho * ortho_loss_val
        
        # Backward con acumulación
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if opt_topo: opt_topo.step()
        
        # Métricas
        metrics['loss'] += loss.item() * x.size(0) * accumulation_steps
        metrics['acc'] += logits.argmax(1).eq(y).sum().item()
        metrics['supcon_loss'] += supcon_loss_val if isinstance(supcon_loss_val, float) else supcon_loss_val.item() * x.size(0)
        metrics['ortho_loss'] += ortho_loss_val if isinstance(ortho_loss_val, float) else ortho_loss_val.item() * x.size(0)
        metrics['n_samples'] += x.size(0)
        
        # Monitor de recursos cada 50 batches
        if batch_idx % 50 == 0:
            ResourceMonitor.check_limit(config.memory_limit_gb)
    
    # Promediar
    for k in metrics:
        if k != 'n_samples':
            metrics[k] /= metrics['n_samples']
    
    metrics['acc'] = 100 * metrics['acc'] / metrics['n_samples']
    metrics['lambda_supcon'] = lambda_supcon
    
    # FIX: Verificar topo_monitor antes de usar
    if config.use_plasticity and topo_monitor is not None:
        metrics_topo = topo_monitor.calculate(epoch)
        
        # Logging compacto
        adj = metrics_topo['adjacency']
        print(f"  [Topo] S:{adj.sparsity:.1%} L:{adj.L_score:.1f} | {adj.status}")
        
        # Alerta crítica
        if "PRUNED" in adj.status:
            critical_msg = topo_monitor.get_critical_summary()
            print(f"  🚨 {critical_msg}")
            
            # ACCIÓN AUTOMÁTICA
            if hasattr(config, 'lambda_sparsity') and config.lambda_sparsity < 1e-2:
                old_lambda = config.lambda_sparsity
                config.lambda_sparsity *= 2.0
                print(f"  🔧 Ajustado lambda_sparsity: {old_lambda:.0e} → {config.lambda_sparsity:.0e}")
    
    return metrics


def train_model(config: Config, run_name: str):
    """Loop de entrenamiento v18"""
    # Setup
    seed_everything(config.seed)
    os.makedirs(f"results/{run_name}", exist_ok=True)
    
    # Data
    train_loader, test_loader, in_channels = get_dataloaders(config)
    
    # Model v18
    model = TopoBrainNetV18(config, in_channels).to(config.device)
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Inicialización de topología
    # HOTFIX: Inicialización suave para sparsity inicial ~30-50%
    if config.use_plasticity:
        with torch.no_grad():
            # Uniforme en [0.5, 1.0] -> sigmoid en [0.62, 0.73] -> sparsity ~35%
            model.adj_weights.uniform_(0.5, 1.0)
            model.inc_weights.uniform_(0.5, 1.0)
            
            # Asegurar conectividad base mínima
            grid = config.grid_size
            for i in range(grid * grid):
                r, c = i // grid, i % grid
                # Vecinos 4-conectados siempre activos
                for offset in [-grid, grid, -1, 1]:
                    j = i + offset
                    if 0 <= j < grid * grid:
                        if abs((i // grid) - (j // grid)) + abs((i % grid) - (j % grid)) == 1:
                            model.adj_weights.data[i, j] = 1.5  # Valor alto para edges base
        
    topo_monitor = TopologicalHealthSovereignty(model, config) if config.use_plasticity else None
    
    # Optimizers
    main_params = [p for n,p in model.named_parameters() 
                   if 'weights' not in n and p.requires_grad]
    topo_params = [p for n,p in model.named_parameters() 
                   if 'weights' in n and p.requires_grad]
    
    optimizer = optim.SGD(main_params, lr=config.lr_main, 
                         momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                              milestones=[15, 25], gamma=0.1)
    
    opt_topo = None
    sched_topo = None
    if config.use_plasticity and topo_params:
        opt_topo = optim.AdamW(topo_params, lr=config.lr_topo, weight_decay=1e-3)
        
        def warmup_topo(epoch):
            if epoch < config.topo_warmup_epochs:
                return 0.0
            elif epoch < config.topo_warmup_epochs + 5:
                return (epoch - config.topo_warmup_epochs) / 5.0
            return 1.0
        
        sched_topo = optim.lr_scheduler.LambdaLR(opt_topo, lr_lambda=warmup_topo)
    
    supcon = SupConLoss()
    checkpoint_mgr = CheckpointManager()
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Entrenando: {run_name}")
    print(f"Config: {config.dataset} | Grid: {config.grid_size}x{config.grid_size}")
    print(f"Sparse Ops: {config.use_sparse_ops} | Adaptive Topo: {config.use_adaptive_topology}")
    print(f"{'='*60}\n")
    
    best_acc = 0
    history = []
    
    for epoch in range(1, config.epochs + 1):
        model.set_epoch(epoch) 
        train_metrics = train_epoch(
            model, train_loader, optimizer, opt_topo, 
            supcon, config, epoch, topo_monitor
        )
        
        scheduler.step()
        if sched_topo: sched_topo.step()
        
        # Poda más frecuente y agresiva
        if config.use_adaptive_topology and epoch % config.topology_update_freq == 0:
            model.prune_topology()
            print(f"  🔧 Topología podada en epoch {epoch}")
        
        # Logging más informativo
        log_msg = (f"Epoch {epoch:02d}/{config.epochs} | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['acc']:.2f}%")
        
        if config.use_supcon:
            log_msg += f" | SupCon: {train_metrics['supcon_loss']:.4f} (λ={train_metrics['lambda_supcon']:.3f})"
        
        if config.use_plasticity:
            sparsity = (torch.sigmoid(model.adj_weights) > 0.5).float().mean().item()
            log_msg += f" | Sparsity: {sparsity:.2%}"
            log_msg += f" | Ortho: {train_metrics['ortho_loss']:.4f}"
        
        print(log_msg)
        
        # Checkpoint periódico
        if epoch % config.checkpoint_interval == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'topo_optimizer_state': opt_topo.state_dict() if opt_topo else None,
                'scheduler_state': scheduler.state_dict(),
                'topo_scheduler_state': sched_topo.state_dict() if sched_topo else None,
                'config': config.to_dict(),
                'train_metrics': train_metrics
            }
            checkpoint_mgr.save(checkpoint_data, f"{run_name}_epoch{epoch}")
            
            if config.use_plasticity:
                save_topology_visualization(model, epoch, run_name)
                save_node_importance_viz(model, epoch, run_name)
            
            ResourceMonitor.log(f"[Epoch {epoch}] ")
        
        # Mejor modelo
        if train_metrics['acc'] > best_acc:
            best_acc = train_metrics['acc']
            checkpoint_data = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config.to_dict(),
                'best_acc': best_acc
            }
            checkpoint_mgr.save(checkpoint_data, f"{run_name}_best")
            
        # HOTFIX: Early stopping si accuracy no mejora en 3 épocas
        if len(history) >= 3:
            recent_acc = [h['acc'] for h in history[-3:]]
            if max(recent_acc) < 5.0:  # Si en 3 épocas no supera 5%
                print(f"  🚨 Early stopping: Accuracy estancado en {recent_acc}")
                break        
        history.append(train_metrics)
    
    if topo_monitor is not None and len(topo_monitor.history) > 0:
        torch.save(
            topo_monitor.history, 
            f"results/{run_name}/topo_health_history.pt"
        )
        print(f"  ✅ Historial de salud topológica guardado")
    
    # Evaluación final
    print(f"\n{'='*60}")
    print("EVALUACIÓN FINAL")
    print(f"{'='*60}")
    
    clean_acc = evaluate(model, test_loader, config, adversarial=False)
    print(f"✅ Clean Accuracy: {clean_acc:.2f}%")
    
    pgd_acc = evaluate(model, test_loader, config, adversarial=True)
    print(f"🛡️  PGD-{config.pgd_steps_test} Accuracy: {pgd_acc:.2f}%")
    
    # Guardar resultados finales
    final_results = {
        'clean_acc': clean_acc,
        'pgd_acc': pgd_acc,
        'best_train_acc': best_acc,
        'config': config.to_dict(),
        'history': history
    }
    
    results_path = Path(f"results/{run_name}")
    results_path.mkdir(exist_ok=True, parents=True)
    
    with open(results_path / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"{'='*60}\n")
    
    ResourceMonitor.clear_cache()
    
    return clean_acc, pgd_acc


@torch.no_grad()
def evaluate(model, test_loader, config: Config, adversarial=False):
    """Evalúa el modelo"""
    model.eval()
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(config.device), y.to(config.device)
        
        if adversarial:
            x = make_adversarial_pgd(
                model, x, y,
                config.test_eps,
                config.pgd_steps_test,
                config.dataset
            )
        
        logits = model(x)[0]
        correct += logits.argmax(1).eq(y).sum().item()
        total += y.size(0)
    
    return 100 * correct / total

# =============================================================================
# 7. ANÁLISIS E INTERPRETABILIDAD DE TOPOLOGÍA
# =============================================================================

def save_topology_visualization(model, epoch, run_name):
    """Guarda visualización de la topología aprendida"""
    if not model.config.use_plasticity:
        return
    
    with torch.no_grad():
        adj_w = torch.sigmoid(model.adj_weights).cpu().numpy()
        inc_w = torch.sigmoid(model.inc_weights).cpu().numpy()
    
    results_dir = Path(f"results/{run_name}")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Visualizar matriz de adyacencia
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
    
    # Guardar métricas topológicas
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
    """
    Visualiza importancia de nodos por capa
    """
    results_dir = Path(f"results/{run_name}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (layer, ax) in enumerate(zip([model.layer1, model.layer2], axes)):
        importance = layer.get_node_importance()
        if importance is not None:
            importance_2d = importance.cpu().numpy().reshape(model.config.grid_size, model.config.grid_size)
            
            im = ax.imshow(importance_2d, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Layer {idx+1} Node Importance (Epoch {epoch})')
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(results_dir / f"node_importance_epoch{epoch:03d}.png", dpi=150)
    plt.close()

def analyze_topology_clustering(model, run_name: str):
    """
    Clustering espectral de nodos basado en conectividad
    """
    if not model.config.use_plasticity:
        print("⚠️  Modelo no tiene topología aprendible")
        return
    
    results_dir = Path(f"results/{run_name}")
    
    with torch.no_grad():
        adj_w = torch.sigmoid(model.adj_weights).cpu().numpy()
    
    # Spectral clustering
    n_clusters = min(8, model.config.grid_size)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = clustering.fit_predict(adj_w)
    
    # Visualizar clusters en grid
    labels_2d = labels.reshape(model.config.grid_size, model.config.grid_size)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(labels_2d, cmap='tab10', interpolation='nearest')
    plt.title(f'Spectral Clustering of Topology ({n_clusters} clusters)')
    plt.colorbar(label='Cluster ID')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.tight_layout()
    plt.savefig(results_dir / "topology_clustering.png", dpi=150)
    plt.close()
    
    print(f"✅ Análisis de clustering guardado en {results_dir / 'topology_clustering.png'}")
    
    # Guardar cluster assignments
    cluster_data = {
        'n_clusters': n_clusters,
        'labels': labels.tolist(),
        'cluster_sizes': [int((labels == i).sum()) for i in range(n_clusters)]
    }
    
    with open(results_dir / "topology_clusters.json", 'w') as f:
        json.dump(cluster_data, f, indent=2)

def analyze_topology_flow(model, dataloader, run_name: str, num_samples=100):
    """
    Analiza flujo de información en la topología
    Similar a Grad-CAM pero para topología
    """
    if not model.config.use_plasticity:
        print("⚠️  Modelo no tiene topología aprendible")
        return
    
    model.eval()
    results_dir = Path(f"results/{run_name}")
    
    # Acumular activaciones por nodo
    node_activations_l1 = torch.zeros(model.config.grid_size ** 2, device=model.config.device)
    node_activations_l2 = torch.zeros(model.config.grid_size ** 2, device=model.config.device)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_samples // dataloader.batch_size:
                break
            
            x = x.to(model.config.device)
            
            # Forward con hook para capturar activaciones
            x_embed = model.patch_embed(x)
            x_nodes = x_embed.flatten(2).transpose(1, 2)
            
            if model.config.use_sparse_ops:
                curr_adj, curr_inc, adj_sparse, inc_sparse = model.get_topology(return_sparse=True)
            else:
                curr_adj, curr_inc = model.get_topology(return_sparse=False)
                adj_sparse, inc_sparse = None, None
            
            # Layer 1
            x_l1, _ = model.layer1(x_nodes, curr_adj, curr_inc, adj_sparse, inc_sparse)
            node_activations_l1 += x_l1.abs().mean(dim=[0, 2])  # Promedio sobre batch y features
            
            # Layer 2
            x_l2, _ = model.layer2(F.gelu(x_l1), curr_adj, curr_inc, adj_sparse, inc_sparse)
            node_activations_l2 += x_l2.abs().mean(dim=[0, 2])
    
    # Normalizar
    node_activations_l1 /= (num_samples / dataloader.batch_size)
    node_activations_l2 /= (num_samples / dataloader.batch_size)
    
    # Visualizar
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
    
    print(f"✅ Análisis de flujo guardado en {results_dir / 'topology_flow.png'}")

def visualize_topology_as_graph(model, run_name: str, threshold=0.3):
    """
    Visualiza topología como grafo con NetworkX
    """
    if not model.config.use_plasticity:
        print("⚠️  Modelo no tiene topología aprendible")
        return
    
    results_dir = Path(f"results/{run_name}")
    
    with torch.no_grad():
        adj_w = torch.sigmoid(model.adj_weights).cpu().numpy()
    
    # Crear grafo
    G = nx.Graph()
    num_nodes = adj_w.shape[0]
    
    # Agregar nodos con posiciones de grid
    grid_size = model.config.grid_size
    pos = {}
    for i in range(num_nodes):
        r, c = i // grid_size, i % grid_size
        pos[i] = (c, grid_size - r - 1)  # Invertir Y para visualización
        G.add_node(i)
    
    # Agregar aristas con peso > threshold
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_w[i, j] > threshold:
                G.add_edge(i, j, weight=adj_w[i, j])
    
    # Visualizar
    plt.figure(figsize=(12, 12))
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue', alpha=0.9)
    
    # Dibujar aristas con grosor proporcional al peso
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w * 3 for w in weights], alpha=0.5)
    
    plt.title(f'Learned Topology Graph (threshold={threshold})')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(results_dir / "topology_graph.png", dpi=150)
    plt.close()
    
    # Calcular métricas de grafo
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
    
    print(f"✅ Grafo de topología guardado en {results_dir / 'topology_graph.png'}")
    print(f"   Métricas: {graph_metrics}")

def analyze_topology_evolution(run_name: str):
    """Analiza la evolución de la topología a lo largo del entrenamiento"""
    results_dir = Path(f"results/{run_name}")
    metrics_file = results_dir / "topology_metrics.jsonl"
    
    if not metrics_file.exists():
        print(f"⚠️  No se encontró {metrics_file}")
        return
    
    # Cargar métricas
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    
    if not metrics:
        print("⚠️  No hay métricas para analizar")
        return
    
    # Extraer datos
    epochs = [m['epoch'] for m in metrics]
    adj_sparsity = [m['adj_sparsity'] for m in metrics]
    adj_mean = [m['adj_mean'] for m in metrics]
    inc_sparsity = [m['inc_sparsity'] for m in metrics]
    inc_mean = [m['inc_mean'] for m in metrics]
    
    # Crear visualización
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(epochs, adj_sparsity, 'o-', linewidth=2)
    axes[0, 0].set_title('Adjacency Sparsity Evolution')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Sparsity (% > 0.5)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, adj_mean, 'o-', linewidth=2, color='orange')
    axes[0, 1].set_title('Adjacency Mean Weight Evolution')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Weight')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, inc_sparsity, 'o-', linewidth=2, color='green')
    axes[1, 0].set_title('Incidence Sparsity Evolution')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Sparsity (% > 0.5)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, inc_mean, 'o-', linewidth=2, color='red')
    axes[1, 1].set_title('Incidence Mean Weight Evolution')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Mean Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "topology_evolution.png", dpi=150)
    plt.close()
    
    print(f"✅ Análisis guardado en {results_dir / 'topology_evolution.png'}")
    
    # Estadísticas finales
    print("\n" + "="*60)
    print(f"ANÁLISIS DE TOPOLOGÍA: {run_name}")
    print("="*60)
    print(f"Adjacency Sparsity:  {adj_sparsity[0]:.2%} → {adj_sparsity[-1]:.2%}")
    print(f"Adjacency Mean:      {adj_mean[0]:.4f} → {adj_mean[-1]:.4f}")
    print(f"Incidence Sparsity:  {inc_sparsity[0]:.2%} → {inc_sparsity[-1]:.2%}")
    print(f"Incidence Mean:      {inc_mean[0]:.4f} → {inc_mean[-1]:.4f}")
    print("="*60 + "\n")

def comprehensive_topology_analysis(model, dataloader, run_name: str):
    """
    Análisis completo de topología
    Ejecuta todos los análisis disponibles
    """
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETO DE TOPOLOGÍA")
    print("="*60 + "\n")
    
    # 1. Clustering
    print("1️⃣  Clustering espectral...")
    analyze_topology_clustering(model, run_name)
    
    # 2. Flujo de información
    print("2️⃣  Análisis de flujo de información...")
    analyze_topology_flow(model, dataloader, run_name)
    
    # 3. Visualización como grafo
    print("3️⃣  Visualización como grafo...")
    visualize_topology_as_graph(model, run_name)
    
    # 4. Evolución temporal
    print("4️⃣  Evolución temporal...")
    analyze_topology_evolution(run_name)
    
    print("\n✅ Análisis completo finalizado\n")

# =============================================================================
# 8. SUITE DE ABLACIÓN
# =============================================================================

def run_ablation_study():
    """Ejecuta suite completa de ablación v18"""
    
    experiments = [
        {
            'name': 'Baseline_AT',
            'desc': 'Solo adversarial training',
            'config': Config(
                use_plasticity=False,
                use_mgf=False,
                use_supcon=False,
                use_sparse_ops=False,
                use_adaptive_topology=False,
                epochs=30
            )
        },
        {
            'name': 'TopoBrain_Sparse',
            'desc': 'Con sparse operations',
            'config': Config(
                use_plasticity=True,
                use_mgf=True,
                use_supcon=True,
                use_sparse_ops=True,
                use_adaptive_topology=False,
                epochs=30
            )
        },
        {
            'name': 'TopoBrain_Adaptive',
            'desc': 'Con topología adaptativa',
            'config': Config(
                use_plasticity=True,
                use_mgf=True,
                use_supcon=True,
                use_sparse_ops=False,
                use_adaptive_topology=True,
                epochs=30
            )
        },
        {
            'name': 'TopoBrain_Full_v18',
            'desc': 'Todos los componentes v18',
            'config': Config(
                use_plasticity=True,
                use_mgf=True,
                use_supcon=True,
                use_sparse_ops=True,
                use_adaptive_topology=True,
                epochs=30
            )
        }
    ]
    
    print("\n" + "="*80)
    print("TOPOBRAIN v18 - SUITE DE ABLACIÓN")
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
        
        # Limpieza entre experimentos
        ResourceMonitor.clear_cache()
        time.sleep(2)
    
    # Guardar resumen comparativo
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
    
    # Guardar resultados completos
    ablation_results = {
        'timestamp': datetime.now().isoformat(),
        'results': results_summary
    }
    
    with open('ablation_results_v18.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print("✅ Resultados guardados en ablation_results_v18.json\n")
    
    return results_summary

# =============================================================================
# 9. 🆕 MAIN Y CLI
# =============================================================================

def main():
    """Punto de entrada principal v18"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TopoBrain v18 - Best of Breed Edition')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'ablation', 'analyze', 'full-analysis'],
                       help='Modo de ejecución')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       choices=['CIFAR10', 'MNIST'],
                       help='Dataset a usar')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Tamaño de batch')
    parser.add_argument('--grid-size', type=int, default=8,
                       help='Tamaño del grid topológico')
    
    # 🆕 Nuevos flags v18
    parser.add_argument('--no-plasticity', action='store_true',
                       help='Desactivar topología aprendible')
    parser.add_argument('--no-supcon', action='store_true',
                       help='Desactivar contrastive learning')
    parser.add_argument('--no-sparse', action='store_true',
                       help='🆕 Desactivar sparse operations')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='🆕 Desactivar topología adaptativa')
    
    parser.add_argument('--run-name', type=str, default=None,
                       help='Nombre del experimento')
    parser.add_argument('--seed', type=int, default=42,
                       help='Semilla aleatoria')
    parser.add_argument('--debug', action='store_true',
                       help='Modo debug (menos épocas)')
    parser.add_argument('--lambda-ortho', type=float, default=1e-4,
                       help='🆕 Peso de regularización ortogonal')
    
    args = parser.parse_args()
    
    # Crear configuración v18
    config = Config(
        dataset=args.dataset,
        epochs=3 if args.debug else args.epochs,
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        use_plasticity=not args.no_plasticity,
        use_supcon=not args.no_supcon,
        use_sparse_ops=not args.no_sparse,
        use_adaptive_topology=not args.no_adaptive,
        seed=args.seed,
        debug_mode=args.debug
    )
    
    # Generar nombre del run si no se especifica
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        components = []
        if config.use_plasticity: components.append('topo')
        if config.use_supcon: components.append('supcon')
        if config.use_sparse_ops: components.append('sparse')
        if config.use_adaptive_topology: components.append('adaptive')
        if not components: components.append('baseline')
        args.run_name = f"topobrain_v18_{'_'.join(components)}_{timestamp}"
    
    print("\n" + "="*80)
    print("TOPOBRAIN v18 - BEST OF BREED EDITION")
    print("="*80)
    print(f"Modo: {args.mode}")
    print(f"Run: {args.run_name}")
    print(f"Device: {config.device}")
    print(f"Sparse Ops: {config.use_sparse_ops}")
    print(f"Adaptive Topology: {config.use_adaptive_topology}")
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
            checkpoint_mgr = CheckpointManager()
            checkpoint = checkpoint_mgr.load(f"{args.run_name}_best")
            
            if checkpoint is None:
                print("❌ No se encontró checkpoint del modelo")
                return
            
            # Recrear modelo
            train_loader, test_loader, in_channels = get_dataloaders(config)
            model = TopoBrainNetV18(config, in_channels).to(config.device)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            
            # Ejecutar análisis completo
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
     
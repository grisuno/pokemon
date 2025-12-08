# =============================================================================
# TopoBrain v19 - Best of Breed Edition (Corregido)
# Aborda: Kitchen Sink, Grid Fijo, Sparse Ops, Ablación, Poda, Logging
# =============================================================================

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
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import networkx as nx
import seaborn as sns
import wandb
from tqdm import tqdm

# PyTorch Geometric (instalar: pip install torch_geometric)
from torch_geometric.nn import GATConv, DynamicEdgeConv, global_max_pool
from torch_geometric.utils import dense_to_sparse, add_self_loops, remove_self_loops
from torch_geometric.data import Data, Batch

# =============================================================================
# 0. CONFIGURACIÓN SIMPLIFICADA v19
# =============================================================================

@dataclass
class Config:
    """Configuración unificada y simplificada"""
    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    
    # Dataset
    dataset: str = "CIFAR10"
    batch_size: int = 128
    num_workers: int = 4
    
    # Modelo (Topología Latente)
    num_nodes: int = 64  # Número de nodos libres (no grid fijo)
    k_neighbors: int = 8  # k-NN dinámico
    node_dim: int = 3  # Dimensión del espacio latente de nodos
    
    # Features Modulares
    use_plasticity: bool = True  # Topología aprendible
    use_predictive_coding: bool = True  # PC asimétrico
    use_contrastive: bool = True  # SupCon
    use_mgf: bool = True  # Morse Graph Flow
    
    # Regularización Unificada
    lambda_topo: float = 0.01  # Único lambda para todo lo topológico
    lambda_contrastive: float = 0.1  # Lambda fijo para SupCon
    lambda_ortho: float = 1e-4  # Orthogonalidad
    
    # Entrenamiento
    epochs: int = 30
    lr: float = 0.1
    warmup_epochs: int = 5
    
    # Adversarial
    train_eps: float = 8/255
    test_eps: float = 8/255
    pgd_steps_train: int = 7
    pgd_steps_test: int = 20
    
    # Poda Estructural
    prune_threshold: float = 0.1
    prune_freq: int = 10  # Cada N épocas
    
    # Sistema
    checkpoint_interval: int = 5
    memory_limit_gb: float = 8.0
    debug_mode: bool = False
    use_wandb: bool = True
    wandb_project: str = "topobrain_v19"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# =============================================================================
# 1. UTILIDADES CORE v19
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
        return {'ram_gb': ram, 'gpu_gb': gpu, 'cpu_percent': cpu}
    
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

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints_v19"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save(self, data: Dict, name: str):
        filepath = self.checkpoint_dir / f"{name}.ckpt"
        temp_file = filepath.with_suffix('.tmp')
        
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
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
        
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ Checkpoint cargado: {filepath.name}")
                return data
            except Exception as e:
                print(f"⚠️ Error cargando {filepath.name}: {e}")
        
        print(f"ℹ️ No checkpoint encontrado: {name}")
        return None

# =============================================================================
# 2. DATASET HELPERS v19
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
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, in_channels

# =============================================================================
# 3. TOPOLOGÍA LATENTE APRENDIDA v19
# =============================================================================

class NodePositionLearner(nn.Module):
    """
    Aprende posiciones de nodos en espacio latente
    Genera conectividad k-NN dinámica
    """
    def __init__(self, num_nodes: int, node_dim: int, k: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.k = k
        
        # Posiciones iniciales (distribución uniforme en esfera)
        self.positions = nn.Parameter(torch.randn(num_nodes, node_dim))
        nn.init.normal_(self.positions, mean=0, std=1)
        
    def forward(self, batch_size: int):
        """
        Retorna edges para k-NN dinámico
        Returns:
            edge_index: [2, E]
            edge_weight: [E]
        """
        # Calcular distancias
        pos = self.positions  # [N, D]
        dist = torch.cdist(pos, pos)  # [N, N]
        
        # k-NN (excluyendo self)
        _, indices = torch.topk(dist, self.k + 1, dim=-1, largest=False)
        indices = indices[:, 1:]  # Remover self-loop
        
        # Construir edge_index
        source = torch.arange(self.num_nodes, device=pos.device).unsqueeze(1).expand(-1, self.k)
        edge_index = torch.stack([source.flatten(), indices.flatten()], dim=0)
        
        # Ponderación por distancia inversa
        edge_weight = 1.0 / (dist[source.flatten(), indices.flatten()] + 1e-6)
        edge_weight = edge_weight / edge_weight.max()
        
        # Batchificar
        edge_index_batch = edge_index.clone()
        edge_weight_batch = edge_weight.clone()
        
        for b in range(1, batch_size):
            offset = b * self.num_nodes
            edge_index_batch = torch.cat([
                edge_index_batch,
                edge_index + offset
            ], dim=1)
            edge_weight_batch = torch.cat([edge_weight_batch, edge_weight], dim=0)
        
        return edge_index_batch, edge_weight_batch

class DynamicTopologicalLayer(nn.Module):
    """
    Capa con PyTorch Geometric y topología dinámica
    Combina GAT + Predictive Coding + MGF
    """
    def __init__(self, in_dim: int, hid_dim: int, config: Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # GAT para procesamiento con atención topológica
        self.gat = GATConv(in_dim, hid_dim // 4, heads=4, dropout=0.1)
        
        # Predictive Coding (opcional)
        if config.use_predictive_coding:
            self.pc_cell = nn.Sequential(
                spectral_norm(nn.Linear(hid_dim, hid_dim)),
                nn.LayerNorm(hid_dim),
                nn.ReLU()
            )
        
        # MGF - Morse Graph Flow (opcional)
        if config.use_mgf:
            self.mgf_proj = nn.Linear(hid_dim, hid_dim)
            self.symbiotic = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.LayerNorm(hid_dim),
                nn.ReLU()
            )
        
        # Salida final
        self.out_proj = nn.Linear(hid_dim * (1 + config.use_predictive_coding + config.use_mgf), hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            x: [B*N, D] Node features
            edge_index: [2, E] Connectivity
            edge_weight: [E] Edge weights
            batch: [B*N] Batch indices
        """
        # GAT
        h_gat = self.gat(x, edge_index)  # [B*N, D]
        
        features = [h_gat]
        
        # Predictive Coding
        if self.config.use_predictive_coding:
            # Simple PC: predice el futuro (aquí: versión simplificada)
            h_pred = self.pc_cell(h_gat)
            features.append(h_pred)
        
        # MGF
        if self.config.use_mgf:
            # Versión simplificada de Morse Graph Flow
            h_mgf = self.symbiotic(self.mgf_proj(h_gat))
            features.append(h_mgf)
        
        # Combinar features
        h_comb = torch.cat(features, dim=-1)
        h_out = self.out_proj(h_comb)
        
        return self.norm(h_out)

# =============================================================================
# 4. MODELO PRINCIPAL v19
# =============================================================================

class TopoBrainNetV19(nn.Module):
    """
    Modelo principal con topología dinámica y componentes modulares
    """
    def __init__(self, config: Config, in_channels: int):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Learner de posiciones de nodos
        self.node_learner = NodePositionLearner(
            config.num_nodes, config.node_dim, config.k_neighbors
        )
        
        # Capas dinámicas
        self.layer1 = DynamicTopologicalLayer(64, 128, config, layer_idx=0)
        self.layer2 = DynamicTopologicalLayer(128, 256, config, layer_idx=1)
        
        # Readout
        self.readout = nn.Linear(256, 10)
        
        # Proyección para contrastivo (opcional)
        if config.use_contrastive:
            self.proj_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        
        # Para pruning estructural
        self.pruned_edges = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, D]
        
        # Mapear a nodos
        x_nodes = x.mean(dim=1, keepdim=True).expand(-1, self.config.num_nodes, -1)
        x_nodes = x_nodes.reshape(batch_size * self.config.num_nodes, -1)
        
        # Generar topología dinámica
        edge_index, edge_weight = self.node_learner(batch_size)
        
        # Aplicar pruning si existe
        if self.pruned_edges is not None:
            edge_index, edge_weight = self.apply_pruning(edge_index, edge_weight)
        
        # Batch indices
        batch = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, self.config.num_nodes).flatten()
        
        # Forward capas
        h1 = self.layer1(x_nodes, edge_index, edge_weight, batch)
        h1 = F.gelu(h1)
        
        h2 = self.layer2(h1, edge_index, edge_weight, batch)
        h2 = F.gelu(h2)
        
        # Pooling global
        h_pool = global_max_pool(h2, batch)
        
        # Readout
        logits = self.readout(h_pool)
        
        # Proyección contrastiva
        proj = None
        if self.config.use_contrastive:
            proj = F.normalize(self.proj_head(h_pool), dim=1)
        
        # Métricas
        metrics = {
            'edge_sparsity': (edge_weight > 0.1).float().mean().item(),
            'mean_degree': edge_index.shape[1] / (batch_size * self.config.num_nodes)
        }
        
        return logits, proj, metrics
    
    def apply_pruning(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        """Aplica máscara de pruning"""
        if self.pruned_edges is None:
            return edge_index, edge_weight
        
        mask = self.pruned_edges(edge_index)
        return edge_index[:, mask], edge_weight[mask]
    
    def prune_structural(self, threshold: float):
        """Pruning estructural real: elimina edges permanentemente"""
        with torch.no_grad():
            # Identificar edges débiles
            edge_index, edge_weight = self.node_learner(1)  # Batch size 1 para sampling
            weak_edges = edge_weight < threshold
            
            # Crear máscara permanente
            if not hasattr(self, 'pruning_mask'):
                self.register_buffer('pruning_mask', torch.ones_like(edge_weight, dtype=torch.bool))
            
            self.pruning_mask[weak_edges] = False
            
            # Función para aplicar pruning
            def prune_fn(edge_idx):
                # Mapear edges a máscara (simplificado)
                return self.pruning_mask
            
            self.pruned_edges = prune_fn
    
    def calculate_ortho_loss(self) -> torch.Tensor:
        """Regularización ortogonal simple"""
        loss = 0
        for layer in [self.layer1, self.layer2]:
            for module in layer.modules():
                if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                    w = module.weight
                    rows, cols = w.shape
                    if rows >= cols:
                        target = torch.eye(cols, device=w.device)
                        loss += torch.norm(w.T @ w - target, p='fro') / (rows * cols)
                    else:
                        target = torch.eye(rows, device=w.device)
                        loss += torch.norm(w @ w.T - target, p='fro') / (rows * cols)
        return loss

# =============================================================================
# 5. ADVERSARIAL TRAINING v19
# =============================================================================

def make_adversarial_pgd(model, x, y, eps, steps, dataset_name='CIFAR10'):
    """PGD Attack simplificado y robusto"""
    model.eval()
    
    delta = torch.empty_like(x).uniform_(-eps, eps).detach()
    x_adv = torch.clamp(x + delta, 0, 1).detach()
    
    alpha = eps * 2.0 / steps
    
    for _ in range(steps):
        x_adv.requires_grad = True
        
        with torch.enable_grad():
            logits = model(x_adv)[0]
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
        
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0, 1).detach()
    
    model.train()
    return x_adv

# =============================================================================
# 6. TRAINING LOOP v19
# =============================================================================

class ContrastiveLoss(nn.Module):
    """SupCon simplificado"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        device = features.device
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size, device=device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mask_sum = mask.sum(1).clamp(min=1e-6)
        return -(mask * log_prob).sum(1) / mask_sum

def train_epoch(model, train_loader, optimizer, criterion, contrastive_loss, config: Config, epoch: int):
    """Entrena una época con logging integrado"""
    model.train()
    metrics = {'loss': 0, 'acc': 0, 'n_samples': 0, 'contrastive_loss': 0, 'ortho_loss': 0}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
    
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(config.device), y.to(config.device)
        
        # Adversarial examples
        x_adv = make_adversarial_pdg(model, x, y, config.train_eps, config.pgd_steps_train, config.dataset)
        
        # Forward
        optimizer.zero_grad()
        logits, proj, topo_metrics = model(x_adv)
        
        # Pérdida principal
        loss = criterion(logits, y)
        
        # Contrastive loss
        if config.use_contrastive and proj is not None:
            contr_loss = contrastive_loss(proj, y).mean()
            loss += config.lambda_contrastive * contr_loss
            metrics['contrastive_loss'] += contr_loss.item() * x.size(0)
        
        # Regularización ortogonal
        ortho_loss = model.calculate_ortho_loss()
        loss += config.lambda_ortho * ortho_loss
        metrics['ortho_loss'] += ortho_loss.item() * x.size(0)
        
        # Regularización topológica (sparsity)
        if config.use_plasticity:
            sparsity_loss = topo_metrics['edge_sparsity']
            loss += config.lambda_topo * sparsity_loss
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Métricas
        metrics['loss'] += loss.item() * x.size(0)
        metrics['acc'] += logits.argmax(1).eq(y).sum().item()
        metrics['n_samples'] += x.size(0)
        
        # Logging
        if config.use_wandb and batch_idx % 50 == 0:
            wandb.log({
                'batch_loss': loss.item(),
                'batch_acc': logits.argmax(1).eq(y).float().mean().item(),
                'batch_contrastive': contr_loss.item() if config.use_contrastive else 0,
                'batch_ortho': ortho_loss.item(),
                'batch_sparsity': topo_metrics['edge_sparsity'],
                'epoch': epoch
            })
        
        # Monitor de recursos
        if batch_idx % 50 == 0:
            ResourceMonitor.check_limit(config.memory_limit_gb)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{logits.argmax(1).eq(y).float().mean().item()*100:.1f}%"
        })
    
    # Promediar
    for k in metrics:
        if k != 'n_samples':
            metrics[k] /= metrics['n_samples']
    
    metrics['acc'] *= 100
    
    return metrics

def evaluate(model, test_loader, config: Config, adversarial=False):
    """Evalúa el modelo"""
    model.eval()
    correct = 0
    total = 0
    
    pbar = tqdm(test_loader, desc="Evaluando")
    
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(config.device), y.to(config.device)
            
            if adversarial:
                x = make_adversarial_pdg(model, x, y, config.test_eps, config.pgd_steps_test, config.dataset)
            
            logits, _, _ = model(x)
            correct += logits.argmax(1).eq(y).sum().item()
            total += y.size(0)
            
            pbar.set_postfix({'acc': f"{correct/total*100:.1f}%"})
    
    return 100 * correct / total

def train_model(config: Config, run_name: str):
    """Loop de entrenamiento completo v19"""
    # Setup
    seed_everything(config.seed)
    os.makedirs(f"results/{run_name}", exist_ok=True)
    
    # Wandb
    if config.use_wandb:
        wandb.init(project=config.wandb_project, name=run_name, config=config.to_dict())
    
    # Data
    train_loader, test_loader, in_channels = get_dataloaders(config)
    
    # Modelo
    model = TopoBrainNetV19(config, in_channels).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parámetros: {n_params:,}")
    
    if config.use_wandb:
        wandb.log({'n_parameters': n_params})
    
    # Optimizer único
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=5e-4)
    
    def warmup_lr(epoch):
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    
    # Criterios
    criterion = nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss() if config.use_contrastive else None
    
    # Checkpoint manager
    checkpoint_mgr = CheckpointManager()
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Entrenando: {run_name}")
    print(f"Config: {config.dataset} | Nodes: {config.num_nodes} | k: {config.k_neighbors}")
    print(f"Features: PC={config.use_predictive_coding} | Contrastive={config.use_contrastive} | MGF={config.use_mgf}")
    print(f"{'='*60}\n")
    
    best_acc = 0
    history = []
    
    for epoch in range(1, config.epochs + 1):
        # Entrenar
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, contrastive_loss, config, epoch)
        
        scheduler.step()
        
        # Pruning estructural periódico
        if config.use_plasticity and epoch % config.prune_freq == 0 and epoch > config.warmup_epochs:
            model.prune_structural(config.prune_threshold)
            print(f"  🔧 Pruning estructural aplicado (threshold={config.prune_threshold})")
        
        # Evaluar
        clean_acc = evaluate(model, test_loader, config, adversarial=False)
        pgd_acc = evaluate(model, test_loader, config, adversarial=True)
        
        # Log epoch
        print(f"Epoch {epoch:02d}/{config.epochs} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['acc']:.2f}% | "
              f"Val Acc: {clean_acc:.2f}% | "
              f"PGD Acc: {pgd_acc:.2f}%")
        
        if config.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['acc'],
                'val_clean_acc': clean_acc,
                'val_pgd_acc': pgd_acc,
                'lr': optimizer.param_groups[0]['lr'],
                'ortho_loss': train_metrics['ortho_loss'],
                'contrastive_loss': train_metrics['contrastive_loss'],
            })
        
        # Checkpoint
        if epoch % config.checkpoint_interval == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'config': config.to_dict(),
                'metrics': {'train': train_metrics, 'val': {'clean': clean_acc, 'pgd': pgd_acc}}
            }
            checkpoint_mgr.save(checkpoint_data, f"{run_name}_epoch{epoch}")
            
            # Visualización
            save_topology_snapshot(model, epoch, run_name)
        
        # Mejor modelo
        if clean_acc > best_acc:
            best_acc = clean_acc
            checkpoint_data = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'config': config.to_dict(),
                'best_acc': best_acc
            }
            checkpoint_mgr.save(checkpoint_data, f"{run_name}_best")
        
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': {'clean': clean_acc, 'pgd': pgd_acc}
        })
        
        # Monitor de recursos
        if epoch % 5 == 0:
            ResourceMonitor.log(f"[Epoch {epoch}] ")
    
    # Resultados finales
    print(f"\n{'='*60}")
    print("EVALUACIÓN FINAL")
    print(f"{'='*60}")
    print(f"✅ Clean Accuracy: {clean_acc:.2f}%")
    print(f"🛡️  PGD Accuracy: {pgd_acc:.2f}%")
    print(f"🎯 Best Accuracy: {best_acc:.2f}%")
    
    # Guardar resultados
    final_results = {
        'best_acc': best_acc,
        'final_clean': clean_acc,
        'final_pgd': pgd_acc,
        'config': config.to_dict(),
        'history': history
    }
    
    results_path = Path(f"results/{run_name}")
    results_path.mkdir(exist_ok=True, parents=True)
    
    with open(results_path / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    if config.use_wandb:
        wandb.finish()
    
    ResourceMonitor.clear_cache()
    
    return clean_acc, pgd_acc

# =============================================================================
# 7. ANÁLISIS Y VISUALIZACIÓN v19
# =============================================================================

def save_topology_snapshot(model, epoch, run_name):
    """Guarda snapshot de topología"""
    results_dir = Path(f"results/{run_name}")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    with torch.no_grad():
        pos = model.node_learner.positions.cpu().numpy()
        edge_index, edge_weight = model.node_learner(1)  # Batch size 1
        
        # Convertir a numpy
        edge_index = edge_index.cpu().numpy()
        edge_weight = edge_weight.cpu().numpy()
        
        # Visualizar posiciones 2D (PCA si node_dim > 2)
        if model.config.node_dim > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pos_2d = pca.fit_transform(pos)
        else:
            pos_2d = pos[:, :2]
        
        plt.figure(figsize=(10, 10))
        
        # Dibujar nodos
        plt.scatter(pos_2d[:, 0], pos_2d[:, 1], s=100, alpha=0.7, c='lightblue')
        
        # Dibujar edges
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if edge_weight[i] > 0.1:
                x_values = [pos_2d[src, 0], pos_2d[dst, 0]]
                y_values = [pos_2d[src, 1], pos_2d[dst, 1]]
                alpha = edge_weight[i]
                plt.plot(x_values, y_values, 'k-', alpha=alpha * 0.5, linewidth=0.5)
        
        plt.title(f'Topología Latente (Epoch {epoch})')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(results_dir / f"topology_epoch{epoch:03d}.png", dpi=150)
        plt.close()

def run_ablation_study():
    """Suite de ablación sistemática v19"""
    
    experiments = [
        {'name': 'baseline', 'desc': 'ResNet-like básico', 'config': {
            'use_plasticity': False, 'use_predictive_coding': False, 'use_contrastive': False, 'use_mgf': False
        }},
        {'name': 'topo_only', 'desc': 'Solo topología dinámica', 'config': {
            'use_plasticity': True, 'use_predictive_coding': False, 'use_contrastive': False, 'use_mgf': False
        }},
        {'name': 'pc_only', 'desc': 'Solo predictive coding', 'config': {
            'use_plasticity': False, 'use_predictive_coding': True, 'use_contrastive': False, 'use_mgf': False
        }},
        {'name': 'contrastive_only', 'desc': 'Solo SupCon', 'config': {
            'use_plasticity': False, 'use_predictive_coding': False, 'use_contrastive': True, 'use_mgf': False
        }},
        {'name': 'mgf_only', 'desc': 'Solo MGF', 'config': {
            'use_plasticity': False, 'use_predictive_coding': False, 'use_contrastive': False, 'use_mgf': True
        }},
        {'name': 'full', 'desc': 'Todos los componentes', 'config': {
            'use_plasticity': True, 'use_predictive_coding': True, 'use_contrastive': True, 'use_mgf': True
        }},
    ]
    
    print("\n" + "="*80)
    print("TOPOBRAIN v19 - SUITE DE ABLACIÓN SISTEMÁTICA")
    print("="*80 + "\n")
    
    results = []
    
    for exp in experiments:
        print(f"\nExperimento: {exp['name']} - {exp['desc']}")
        print("-" * 60)
        
        # Crear config
        config = Config()
        for k, v in exp['config'].items():
            setattr(config, k, v)
        
        run_name = f"ablation_{exp['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            clean_acc, pgd_acc = train_model(config, run_name)
            results.append({
                'name': exp['name'],
                'desc': exp['desc'],
                'clean_acc': clean_acc,
                'pgd_acc': pgd_acc,
                'config': config.to_dict()
            })
            print(f"✅ Completado: {clean_acc:.2f}% / {pgd_acc:.2f}%")
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({'name': exp['name'], 'error': str(e)})
        
        ResourceMonitor.clear_cache()
        time.sleep(2)
    
    # Resumen
    print("\n" + "="*80)
    print("RESULTADOS DE ABLACIÓN")
    print("="*80)
    print(f"{'Experimento':<15} {'Descripción':<25} {'Clean':<10} {'PGD':<10}")
    print("-"*80)
    
    for r in results:
        if 'error' not in r:
            print(f"{r['name']:<15} {r['desc']:<25} {r['clean_acc']:>8.2f}% {r['pgd_acc']:>8.2f}%")
        else:
            print(f"{r['name']:<15} {r['desc']:<25} {'ERROR':<10} {r['error'][:10]:<10}")
    
    # Guardar
    with open("ablation_results_v19.json", 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'results': results}, f, indent=2)
    
    print("\n✅ Ablación completada - Resultados guardados\n")

# =============================================================================
# 8. CLI v19
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='TopoBrain v19 - Simplificado y Potente')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'ablation', 'analyze'])
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-nodes', type=int, default=64)
    parser.add_argument('--k-neighbors', type=int, default=8)
    
    # Feature flags
    parser.add_argument('--no-plasticity', action='store_true')
    parser.add_argument('--no-pc', action='store_true', help='Desactivar Predictive Coding')
    parser.add_argument('--no-contrastive', action='store_true', help='Desactivar SupCon')
    parser.add_argument('--no-mgf', action='store_true', help='Desactivar MGF')
    
    # Sistema
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Config
    config = Config(
        dataset=args.dataset,
        epochs=3 if args.debug else args.epochs,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        k_neighbors=args.k_neighbors,
        use_plasticity=not args.no_plasticity,
        use_predictive_coding=not args.no_pc,
        use_contrastive=not args.no_contrastive,
        use_mgf=not args.no_mgf,
        seed=args.seed,
        debug_mode=args.debug,
        use_wandb=not args.no_wandb
    )
    
    # Run name
    if args.run_name is None:
        components = []
        if config.use_plasticity: components.append('topo')
        if config.use_predictive_coding: components.append('pc')
        if config.use_contrastive: components.append('contrast')
        if config.use_mgf: components.append('mgf')
        if not components: components.append('baseline')
        args.run_name = f"topobrain_v19_{'_'.join(components)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "="*80)
    print("TOPOBRAIN v19 - SIMPLIFICADO Y POTENTE")
    print("="*80)
    print(f"Modo: {args.mode}")
    print(f"Run: {args.run_name}")
    print(f"Device: {config.device}")
    print(f"Features: PC={config.use_predictive_coding} | Contrastive={config.use_contrastive} | MGF={config.use_mgf}")
    print(f"Topología: {config.num_nodes} nodos | k={config.k_neighbors}")
    print("="*80 + "\n")
    
    try:
        if args.mode == 'train':
            train_model(config, args.run_name)
        elif args.mode == 'ablation':
            run_ablation_study()
        elif args.mode == 'analyze':
            print("Modo analyze no implementado en v19. Usa --mode=train con wandb")
        
        print("\n✅ Ejecución completada exitosamente")
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrumpido por el usuario")
        ResourceMonitor.clear_cache()
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        ResourceMonitor.clear_cache()

if __name__ == "__main__":
    main()
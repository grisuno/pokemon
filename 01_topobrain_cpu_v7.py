"""
TopoBrain CPU-OPTIMIZADO - Funcional para AMD R5 M335
======================================================
CONFIGURACIÓN REALISTA:
- PyTorch CPU (tu GPU no soporta ROCm)
- Hiperparámetros corregidos para APRENDER
- Modelo más pequeño para CPU (Grid 4x4)
- Adversarial training opcional (más rápido)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple
import json

# =============================================================================
# CONFIGURACIÓN OPTIMIZADA PARA CPU
# =============================================================================

def setup_device():
    print("="*80)
    print("🔧 CONFIGURACIÓN DE DISPOSITIVO")
    print("="*80)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚙  Usando CPU (optimizado)")
        print("   NOTA: Tu AMD R5 M335 no soporta ROCm (requiere GCN 3.0+)")
        print("   Alternativa: Instalar PlaidML para OpenCL")
    
    print("="*80 + "\n")
    return device

@dataclass
class Config:
    device: str = "cpu"
    seed: int = 42
    
    # Dataset
    n_samples: int = 1000      # ✅ Reducido para CPU
    n_features: int = 20       # ✅ Reducido
    n_classes: int = 5         # ✅ Menos clases = más fácil aprender
    n_informative: int = 16
    
    # Arquitectura (MUCHO MÁS PEQUEÑA)
    grid_size: int = 4         # ✅ 4x4 = 16 nodos (era 8x8 = 64)
    embed_dim: int = 12        # ✅ Reducido de 16
    hidden_dim: int = 12
    
    # Componentes
    use_plasticity: bool = True
    use_symbiotic: bool = True
    
    # Entrenamiento (CRÍTICO)
    batch_size: int = 32       # ✅ Pequeño para CPU
    epochs: int = 50           # ✅ Más epochs
    lr: float = 0.01           # ✅ Alto para aprender rápido
    weight_decay: float = 1e-5
    
    # Adversarial (SIMPLIFICADO)
    use_adversarial: bool = True  # ✅ Puede desactivarse para velocidad
    train_eps: float = 0.1     # ✅ Reducido de 0.3
    test_eps: float = 0.1
    pgd_steps: int = 3         # ✅ Reducido de 7
    
    # Regularización (MUY SUAVE)
    lambda_ortho: float = 0.01  # ✅ Muy reducido
    lambda_entropy: float = 0.0001
    
    # Topología (CONSERVADORA)
    prune_start_epoch: int = 20
    prune_threshold: float = 0.1
    prune_interval: int = 10
    min_connections: int = 2
    
    # Estabilidad
    clip_value: float = 5.0
    
    def to_dict(self):
        return {k: str(v) if isinstance(v, torch.device) else v 
                for k, v in self.__dict__.items()}

# =============================================================================
# COMPONENTES (SIMPLIFICADOS)
# =============================================================================

class SymbioticBasis(nn.Module):
    def __init__(self, dim: int, num_atoms: int = 4):  # ✅ Menos átomos
        super().__init__()
        self.num_atoms = num_atoms
        self.dim = dim
        
        self.basis = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis, gain=1.0)
        
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = self.query(x)
        K = self.key(self.basis)
        
        attn = torch.matmul(Q, K.T) / (self.dim ** 0.5 + 1e-8)
        weights = F.softmax(attn, dim=-1)
        
        x_clean = torch.matmul(weights, self.basis)
        x_clean = torch.clamp(x_clean, -3.0, 3.0)
        
        entropy = -(weights * torch.log(weights + 1e-8)).sum(-1).mean()
        
        gram = torch.mm(self.basis, self.basis.T)
        identity = torch.eye(gram.size(0), device=gram.device)
        ortho = torch.norm(gram - identity, p='fro') ** 2
        
        return x_clean, entropy, ortho

class DynamicTopology(nn.Module):
    def __init__(self, num_nodes: int, grid_size: int, config: Config):
        super().__init__()
        self.num_nodes = num_nodes
        self.grid_size = grid_size
        self.config = config
        
        # ✅ Inicialización FUERTE (no colapsar)
        self.adj_weights = nn.Parameter(torch.ones(num_nodes, num_nodes))
        
        self.register_buffer('adj_mask', self._create_grid_mask())
        
    def _create_grid_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_nodes, self.num_nodes)
        
        for i in range(self.num_nodes):
            r, c = i // self.grid_size, i % self.grid_size
            
            neighbors = []
            if r > 0: neighbors.append(i - self.grid_size)
            if r < self.grid_size - 1: neighbors.append(i + self.grid_size)
            if c > 0: neighbors.append(i - 1)
            if c < self.grid_size - 1: neighbors.append(i + 1)
            
            for n in neighbors:
                mask[i, n] = 1.0
        
        return mask
    
    def get_adjacency(self, plasticity: float = 1.0) -> torch.Tensor:
        adj = torch.sigmoid(self.adj_weights * plasticity) * self.adj_mask
        deg = adj.sum(1, keepdim=True).clamp(min=1e-6)
        return adj / deg
    
    def prune_connections(self, threshold: float) -> int:
        with torch.no_grad():
            weights_prob = torch.sigmoid(self.adj_weights)
            prune_mask = (weights_prob < threshold) & (self.adj_mask > 0)
            
            for i in range(self.num_nodes):
                active = (weights_prob[i] >= threshold) & (self.adj_mask[i] > 0)
                if active.sum() < self.config.min_connections:
                    topk = torch.topk(
                        weights_prob[i] * self.adj_mask[i],
                        k=min(self.config.min_connections, int(self.adj_mask[i].sum())),
                        largest=True
                    ).indices
                    prune_mask[i, topk] = False
            
            num_pruned = prune_mask.sum().item()
            self.adj_weights.data[prune_mask] = -5.0
        
        return num_pruned
    
    def get_density(self) -> float:
        adj = torch.sigmoid(self.adj_weights) * self.adj_mask
        return (adj > 0.5).float().mean().item()

# =============================================================================
# MODELO SIMPLIFICADO
# =============================================================================

class TopoBrainCPU(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim
        
        self.input_embed = nn.Linear(
            config.n_features,
            self.embed_dim * self.num_nodes
        )
        
        self.topology = DynamicTopology(self.num_nodes, config.grid_size, config) if config.use_plasticity else None
        
        self.node_processor = nn.Sequential(
            nn.Linear(self.embed_dim, config.hidden_dim),
            nn.ReLU(),  # ✅ ReLU más rápido que GELU
            nn.Linear(config.hidden_dim, self.embed_dim)
        )
        
        self.symbiotic = SymbioticBasis(self.embed_dim, num_atoms=4) if config.use_symbiotic else None
        
        self.readout = nn.Sequential(
            nn.Linear(self.embed_dim * self.num_nodes, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor, plasticity: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        x_embed = self.input_embed(x).view(batch_size, self.num_nodes, self.embed_dim)
        
        if self.topology is not None:
            adj = self.topology.get_adjacency(plasticity)
            x_agg = torch.bmm(adj.unsqueeze(0).expand(batch_size, -1, -1), x_embed)
        else:
            x_agg = x_embed
        
        x_proc = self.node_processor(x_agg)
        
        entropy = torch.tensor(0.0, device=x.device)
        ortho = torch.tensor(0.0, device=x.device)
        
        if self.symbiotic is not None:
            x_proc_flat = x_proc.view(-1, self.embed_dim)
            x_refined, ent, ort = self.symbiotic(x_proc_flat)
            x_proc = x_refined.view(batch_size, self.num_nodes, self.embed_dim)
            entropy, ortho = ent, ort
        
        logits = self.readout(x_proc.view(batch_size, -1))
        
        return logits, entropy, ortho

# =============================================================================
# PGD SIMPLIFICADO
# =============================================================================

def pgd_attack(model, x, y, eps, steps, plasticity=1.0):
    was_training = model.training
    model.train()
    
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta.requires_grad = True
    
    for _ in range(steps):
        logits, _, _ = model(x + delta, plasticity)
        loss = F.cross_entropy(logits, y)
        
        model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        
        with torch.no_grad():
            delta.data = (delta + eps / steps * delta.grad.sign()).clamp(-eps, eps)
            delta.requires_grad = True
    
    if not was_training:
        model.eval()
    
    return (x + delta).detach()

# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_epoch(model, loader, optimizer, config, epoch, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        plasticity = 1.0 - (epoch / config.epochs) * 0.5
        
        # Adversarial opcional
        if config.use_adversarial and epoch >= 5:  # ✅ Empezar después de epoch 5
            x = pgd_attack(model, x, y, config.train_eps, config.pgd_steps, plasticity)
        
        logits, entropy, ortho = model(x, plasticity)
        
        loss = F.cross_entropy(logits, y)
        loss += config.lambda_ortho * ortho
        loss -= config.lambda_entropy * entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': 100.0 * correct / total
    }

def evaluate(model, loader, config, device, adversarial=False):
    model.eval()
    correct = 0
    total = 0
    
    if adversarial:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = pgd_attack(model, x, y, config.test_eps, config.pgd_steps, 0.0)
            
            with torch.no_grad():
                logits, _, _ = model(x_adv, 0.0)
                pred = logits.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
    else:
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits, _, _ = model(x, 0.0)
                pred = logits.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
    
    return 100.0 * correct / total

# =============================================================================
# DATASET
# =============================================================================

def get_dataset(config):
    from sklearn.datasets import make_classification
    from torch.utils.data import TensorDataset, DataLoader
    
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        n_informative=config.n_informative,
        n_redundant=2,
        n_clusters_per_class=1,
        flip_y=0.01,
        class_sep=1.5,  # ✅ Más separación = más fácil
        random_state=config.seed
    )
    
    # Normalizar
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, test_loader

# =============================================================================
# MAIN
# =============================================================================

def main():
    device = setup_device()
    config = Config()
    config.device = str(device)
    
    print("="*80)
    print("🧠 TopoBrain CPU-Optimizado")
    print("="*80)
    print(f"📊 Configuración:")
    print(f"   Grid: {config.grid_size}x{config.grid_size} ({config.grid_size**2} nodos)")
    print(f"   Clases: {config.n_classes}")
    print(f"   Learning Rate: {config.lr}")
    print(f"   Adversarial: {'✅ Activado' if config.use_adversarial else '❌ Desactivado'}")
    print("="*80 + "\n")
    
    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Dataset
    print("[1/4] Cargando dataset...")
    train_loader, test_loader = get_dataset(config)
    print(f"      Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}\n")
    
    # Modelo
    print("[2/4] Inicializando modelo...")
    model = TopoBrainCPU(config).to(device)
    n_params = model.count_parameters()
    print(f"      Parámetros: {n_params:,}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    
    # Entrenamiento
    print("[3/4] Entrenando...")
    print("-"*80)
    print(f"{'Ep':<4} {'Loss':<8} {'TrainAcc':<10} {'CleanAcc':<10} {'PGDAcc':<10} {'Gap':<8} {'Dense':<8} {'Time':<6}")
    print("-"*80)
    
    best_pgd_acc = 0.0
    history = []
    
    for epoch in range(config.epochs):
        start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, epoch, device)
        
        # Eval (cada 5 epochs para velocidad)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            clean_acc = evaluate(model, test_loader, config, device, adversarial=False)
            if config.use_adversarial:
                pgd_acc = evaluate(model, test_loader, config, device, adversarial=True)
            else:
                pgd_acc = clean_acc
            gap = clean_acc - pgd_acc
        else:
            clean_acc = pgd_acc = gap = 0.0
        
        # Topología
        if model.topology is not None:
            density = model.topology.get_density()
            if epoch >= config.prune_start_epoch and (epoch + 1) % config.prune_interval == 0:
                model.topology.prune_connections(config.prune_threshold)
        else:
            density = 1.0
        
        scheduler.step()
        elapsed = time.time() - start
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:<4} {train_metrics['loss']:<8.4f} {train_metrics['accuracy']:<10.2f} "
                  f"{clean_acc:<10.2f} {pgd_acc:<10.2f} {gap:<8.2f} {density:<8.3f} {elapsed:<6.1f}s")
        
        if pgd_acc > best_pgd_acc:
            best_pgd_acc = pgd_acc
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'clean_acc': clean_acc,
            'pgd_acc': pgd_acc,
            'gap': gap,
            'density': density
        })
    
    print("-"*80)
    
    # Final eval
    print("\n[4/4] Evaluación final...")
    clean_final = evaluate(model, test_loader, config, device, adversarial=False)
    pgd_final = evaluate(model, test_loader, config, device, adversarial=True) if config.use_adversarial else clean_final
    
    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"🎯 Resultados:")
    print(f"   Clean Accuracy: {clean_final:.2f}%")
    print(f"   PGD Accuracy:   {pgd_final:.2f}%")
    print(f"   Gap:            {clean_final - pgd_final:.2f}%")
    print(f"   Mejor PGD:      {best_pgd_acc:.2f}%")
    print("="*80)
    
    # Guardar
    torch.save({'model_state': model.state_dict(), 'config': config.to_dict()}, 'topobrain_cpu.pth')
    with open('topobrain_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\n✅ Guardado: topobrain_cpu.pth, topobrain_history.json")

if __name__ == "__main__":
    main()
             
# topobrain_tabular_ablation_cpu.py
# Ablation Científica Rigurosa – CPU-only, miles de parámetros, PGD real, sin CIFAR (nominal ordinal intervalo ratio - NOIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import json
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from itertools import combinations

# =============================================================================
# 1. CONFIGURACIÓN CIENTÍFICA (CPU, Tabular, Miles de Parámetros)
# =============================================================================
@dataclass
class Config:
    device: str = "cpu"
    seed: int = 42
    n_samples: int = 2000
    n_features: int = 20
    n_classes: int = 3
    n_informative: int = 15
    n_redundant: int = 3
    flip_y: float = 0.05
    batch_size: int = 32
    grid_size: int = 2                     # 2x2 = 4 nodos → ~8k params
    use_spectral: bool = False             # Inestable en CPU con dim baja
    use_sparse_ops: bool = False
    use_orchestrator: bool = True
    orchestrator_hidden_dim: int = 16
    orchestrator_state_dim: int = 8
    use_plasticity: bool = True
    use_mgf: bool = True
    use_supcon: bool = True
    use_adaptive_topology: bool = True
    use_symbiotic: bool = True
    use_nested_cells: bool = True
    fast_lr: float = 0.01
    forget_rate: float = 0.95
    epochs: int = 10
    lr_main: float = 0.01
    lr_topo: float = 0.01
    train_eps: float = 0.3                 # ℓ∞ norm (máximo cambio por feature)
    test_eps: float = 0.3
    pgd_steps_train: int = 3
    pgd_steps_test: int = 3
    lambda_supcon: float = 0.5
    lambda_entropy: float = 0.001
    lambda_sparsity: float = 1e-4
    lambda_ortho: float = 1e-2

    def to_dict(self):
        return asdict(self)

# =============================================================================
# 2. UTILIDADES
# =============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_tabular_loaders(config: Config):
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        n_informative=config.n_informative,
        n_redundant=config.n_redundant,
        n_clusters_per_class=1,
        flip_y=config.flip_y,
        random_state=config.seed
    )
    # Normalizar a [0,1] → necesario para PGD con límite fijo
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader

# =============================================================================
# 3. COMPONENTES (versión tabular, matemática intacta)
# =============================================================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(logits.size(0), device=features.device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-6)
        return -mean_log_prob_pos.mean()

class ContinuumMemoryCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, fast_lr=0.01, forget_rate=0.9):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fast_lr = fast_lr
        self.V_slow = nn.Linear(input_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.V_slow.weight)
        self.forget_gate = nn.Sequential(nn.Linear(hidden_dim + input_dim, 1), nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Linear(hidden_dim + input_dim, 1), nn.Sigmoid())
        self.semantic_mix = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.semantic_memory = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        nn.init.orthogonal_(self.semantic_memory)
        self.semantic_memory.data *= 0.01

    def forward(self, x, controls=None):
        v = self.V_slow(x)
        y_pred = F.linear(x, self.semantic_memory.T)
        error = v - y_pred
        gate_input = torch.cat([v, x], dim=-1)
        forget = self.forget_gate(gate_input)
        update = self.update_gate(gate_input)
        if controls is not None:
            plasticity = controls.get('memory', 1.0)
            forget = forget * plasticity
            update = update * plasticity
        delta = torch.bmm(error.unsqueeze(-1), x.unsqueeze(1))
        self.semantic_memory.data = (
            forget.mean().item() * self.semantic_memory.data +
            update.mean().item() * self.fast_lr * delta.mean(dim=0)
        ).detach()
        mix = self.semantic_mix(v)
        return mix * v + (1 - mix) * y_pred

class SymbioticBasisRefinement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(8, dim))  # fijo a 8 átomos
        nn.init.orthogonal_(self.basis)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(self.basis)
        attn = torch.matmul(Q, K.T) / (x.size(-1) ** 0.5)
        weights = F.softmax(attn, dim=-1)
        x_clean = torch.matmul(weights, self.basis)
        entropy = -(weights * torch.log(weights + 1e-6)).sum(-1).mean()
        gram = torch.mm(self.basis, self.basis.T)
        identity = torch.eye(gram.size(0), device=gram.device)
        ortho = torch.norm(gram - identity, p='fro') ** 2
        return x_clean, entropy, ortho

# =============================================================================
# 4. MODELO PRINCIPAL (TopoBrain Tabular)
# =============================================================================
class TopoBrainTabular(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.grid_size = config.grid_size
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = 16  # Fijo, pequeño, para CPU
        
        # Embedding unificado de entrada → nodos
        self.input_embed = nn.Linear(config.n_features, self.embed_dim * self.num_nodes)
        
        # Topología
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.num_nodes):
            r, c = i // config.grid_size, i % config.grid_size
            if r > 0: adj[i, i - config.grid_size] = 1
            if r < config.grid_size - 1: adj[i, i + config.grid_size] = 1
            if c > 0: adj[i, i - 1] = 1
            if c < config.grid_size - 1: adj[i, i + 1] = 1
        self.register_buffer('adj_base', adj)
        if config.use_plasticity:
            self.adj_weights = nn.Parameter(torch.randn_like(adj))
        else:
            self.register_buffer('adj_weights', torch.zeros_like(adj))

        # Capas
        self.node_cell = ContinuumMemoryCell(self.embed_dim, self.embed_dim) if config.use_nested_cells else nn.Linear(self.embed_dim, self.embed_dim)
        self.symbiotic = SymbioticBasisRefinement(self.embed_dim) if config.use_symbiotic else None
        self.cell_embed = nn.Linear(self.embed_dim * self.num_nodes, self.embed_dim)  # 64 → 16
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, config.n_classes)
        self.proj_head = nn.Sequential(nn.Linear(self.embed_dim * self.num_nodes, 32), nn.ReLU(), nn.Linear(32, 16)) if config.use_supcon else None
    
    def get_adj(self):
        if self.config.use_plasticity:
            return torch.sigmoid(self.adj_weights) * self.adj_base
        return self.adj_base

    def forward(self, x, controls=None):
        batch_size = x.size(0)
        # Embedding unificado → nodos
        x_embed_full = self.input_embed(x)  # [B, 64]
        x_nodes = x_embed_full.view(batch_size, self.num_nodes, self.embed_dim)  # [B, 4, 16]

        # Topología
        adj = self.get_adj()
        if self.config.use_plasticity:
            x_agg = torch.matmul(adj, x_nodes)
        else:
            x_agg = x_nodes
        # Procesamiento nodal
        if isinstance(self.node_cell, ContinuumMemoryCell):
            x_flat = x_agg.view(-1, 16)
            x_proc_flat = self.node_cell(x_flat, controls)
            x_proc = x_proc_flat.view(batch_size, self.num_nodes, 16)
        else:
            x_proc = self.node_cell(x_agg)
        # MGF (1 celda en 2x2)
        if self.config.use_mgf:
            cell_input = torch.cat([x_nodes[:, i] for i in range(4)], dim=-1)  # [B, 64]
            c = self.cell_embed(cell_input).unsqueeze(1)  # [B, 1, 16]
            if self.symbiotic:
                c_clean, entropy, ortho = self.symbiotic(c.squeeze(1))
                c_clean = c_clean.unsqueeze(1)
            else:
                c_clean, entropy, ortho = c, torch.tensor(0.), torch.tensor(0.)
            pred_nodes = c_clean.expand(-1, 4, -1)
        else:
            pred_nodes = torch.zeros_like(x_proc)
            entropy, ortho = torch.tensor(0.), torch.tensor(0.)
        # Salida
        combined = x_proc + pred_nodes
        out = combined.view(batch_size, -1)
        logits = self.readout(out)
        proj = self.proj_head(out) if self.proj_head and self.training else None
        return logits, proj, entropy, ortho



# =============================================================================
# 5. PGD REAL (tabular, ℓ∞, diferenciable)
# =============================================================================
def pgd_attack(model, x, y, eps, steps, controls=None):
    model.eval()
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta.requires_grad = True
    for _ in range(steps):
        logits, _, _, _ = model(x + delta, controls)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = (delta + 0.5 * eps * grad.sign()).clamp(-eps, eps)
        delta.grad.zero_()
    return (x + delta).detach()

# =============================================================================
# 6. ABLATION CIENTÍFICA
# =============================================================================
def generate_ablation_configs(base_config: Config):
    mechanisms = [
        'use_nested_cells',
        'use_plasticity',
        'use_mgf',
        'use_supcon',
        'use_symbiotic',
        'use_orchestrator',
        'use_adaptive_topology'
    ]
    configs = []
    # Baseline
    base = Config(**base_config.to_dict())
    for m in mechanisms: setattr(base, m, False)
    configs.append(('Baseline', base))
    # Individuales
    for m in mechanisms:
        cfg = Config(**base_config.to_dict())
        for key in mechanisms: setattr(cfg, key, key == m)
        configs.append((f'Only_{m}', cfg))
    # Pares
    for m1, m2 in combinations(mechanisms, 2):
        cfg = Config(**base_config.to_dict())
        for key in mechanisms: setattr(cfg, key, key in (m1, m2))
        configs.append((f'Pair_{m1}_AND_{m2}', cfg))
    return configs

# =============================================================================
# 7. TRAIN & EVAL (científico, métricas reales)
# =============================================================================
def train_and_evaluate(config: Config, name: str):
    seed_everything(config.seed)
    train_loader, test_loader = get_tabular_loaders(config)
    model = TopoBrainTabular(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_main)
    supcon = SupConLoss() if config.use_supcon else None
    model.train()
    for epoch in range(config.epochs):
        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            controls = None
            if config.use_orchestrator:
                controls = {
                    'plasticity': 1.0, 'memory': 1.0, 'supcon': 1.0,
                    'symbiotic': 1.0, 'defense': 1.0
                }
                controls = {k: torch.tensor(v, device=config.device) for k, v in controls.items()}
            # PGD real
            x_adv = pgd_attack(model, x, y, config.train_eps, config.pgd_steps_train, controls)
            logits, proj, entropy, ortho = model(x_adv, controls)
            loss = F.cross_entropy(logits, y)
            if config.use_supcon and supcon and proj is not None:
                loss += config.lambda_supcon * supcon(proj, y)
            loss -= config.lambda_entropy * entropy
            loss += config.lambda_ortho * ortho
            loss.backward()
            optimizer.step()
    # Evaluación
    model.eval()
    def evaluate_adv(loader, eps, steps):
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(config.device), y.to(config.device)
            x_pgd = pgd_attack(model, x, y, eps, steps)
            with torch.no_grad():
                logits, _, _, _ = model(x_pgd)
                correct += logits.argmax(1).eq(y).sum().item()
                total += y.size(0)
        return 100 * correct / total
    clean = evaluate_adv(test_loader, 0.0, 1)
    pgd = evaluate_adv(test_loader, config.test_eps, config.pgd_steps_test)
    return clean, pgd

def run_ablation():
    base_config = Config()
    configs = generate_ablation_configs(base_config)
    results = []
    for name, cfg in configs:
        print(f"▶ {name}")
        clean, pgd = train_and_evaluate(cfg, name)
        results.append({'name': name, 'clean': clean, 'pgd': pgd, 'config': cfg.to_dict()})
        print(f"   ✅ Clean: {clean:.2f}% | PGD: {pgd:.2f}%")
    # Guardar
    with open('ablation_tabular_cpu_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Ranking
    valid = [r for r in results if 'error' not in r]
    ranked = sorted(valid, key=lambda x: x['pgd'], reverse=True)
    print("\n🏆 TOP 5 por Robustez (PGD accuracy):")
    for i, r in enumerate(ranked[:5]):
        print(f"{i+1}. {r['name']:<40} → PGD: {r['pgd']:.2f}%")
    return results

if __name__ == "__main__":
    seed_everything(42)
    print("🔬 TopoBrain Ablation Científica CPU, Tabular, PGD Real")
    results = run_ablation()
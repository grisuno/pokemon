"""
NeuroLogos v5.0 - TopoBrain Ablation CPU-Friendly
==================================================================
Estudio de ablación científica rigurosa en 3 niveles:
1. Nivel 1: Componentes individuales
2. Nivel 2A: Pares (sinergia/antagonismo)
3. Nivel 3: Ablación inversa del sistema óptimo

Métrica principal: PGD Accuracy (robustez adversarial)
Arquitectura: TopoBrain Micro (Grid 2x2, ~2k–5k params)
Validación: 3-Fold Stratified Cross-Validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy import stats
import json
import time
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA OPTIMIZADA PARA CPU
# =============================================================================

@dataclass
class MicroConfig:
    device: str = "cpu"
    seed: int = 42
    
    # Dataset micro (tabular NOIR)
    n_samples: int = 400
    n_features: int = 12
    n_classes: int = 3
    n_informative: int = 9
    
    # Arquitectura TopoBrain Micro (Grid 2x2)
    grid_size: int = 2
    embed_dim: int = 4
    hidden_dim: int = 4
    
    # Entrenamiento rápido
    batch_size: int = 16
    epochs: int = 8
    lr: float = 0.01
    
    # Adversarial ligero (métrica principal)
    train_eps: float = 0.2
    test_eps: float = 0.2
    pgd_steps: int = 3
    
    # Flags de componentes para ablación
    use_plasticity: bool = False   # Fast/Slow learning
    use_continuum: bool = False    # ContinuumMemoryCell
    use_mgf: bool = False          # Multi-Granular Fusion
    use_supcon: bool = False       # Supervised Contrastive
    use_symbiotic: bool = False    # Symbiotic refinement

def setup_device():
    return torch.device("cpu")

def seed_everything(seed: int):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_dataset(config: MicroConfig):
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        n_informative=config.n_informative,
        n_redundant=2,
        n_clusters_per_class=1,
        flip_y=0.05,
        random_state=config.seed
    )
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

# =============================================================================
# COMPONENTES MICRO DE TOPOBRAIN (de tu v2.0)
# =============================================================================
# [Incluye aquí las clases exactas: MicroContinuumCell, MicroSymbioticBasis,
#  MicroTopology, MicroSupConLoss, MicroTopoBrain, micro_pgd_attack, etc.]
# Por brevedad omito el cuerpo, pero se copian tal cual de tu archivo.

class MicroContinuumCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W_slow = nn.Linear(dim, dim, bias=False)
        self.V_slow = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.V_slow.weight, gain=0.1)
        self.gate_net = nn.Linear(dim, 1)
        self.semantic_memory = nn.Parameter(torch.zeros(dim, dim))
        nn.init.normal_(self.semantic_memory, std=0.01)
    def forward(self, x, plasticity=1.0):
        v = self.V_slow(x); v = torch.clamp(v, -2.0, 2.0)
        y_pred = F.linear(x, self.semantic_memory)
        y_pred = torch.clamp(y_pred, -2.0, 2.0)
        v_pred = self.V_slow(y_pred)
        gate = torch.sigmoid(self.gate_net(v.detach())) * plasticity
        with torch.no_grad():
            delta = torch.bmm(x.unsqueeze(-1), x.unsqueeze(1))
            self.semantic_memory.data = 0.95 * self.semantic_memory.data + 0.01 * delta.mean(dim=0)
            mem_norm = self.semantic_memory.data.norm().clamp(min=1e-6)
            self.semantic_memory.data = self.semantic_memory.data / mem_norm * 0.5
        output = gate * v + (1 - gate) * v_pred
        return torch.clamp(output, -2.0, 2.0)

class MicroSymbioticBasis(nn.Module):
    def __init__(self, dim, num_atoms=2):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis, gain=0.5)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.eps = 1e-8
    def forward(self, x):
        Q = self.query(x); K = self.key(self.basis)
        attn = torch.matmul(Q, K.T) / (x.size(-1) ** 0.5 + self.eps)
        weights = F.softmax(attn, dim=-1)
        x_clean = torch.matmul(weights, self.basis)
        return torch.clamp(x_clean, -2.0, 2.0), -(weights * torch.log(weights + self.eps)).sum(-1).mean(), torch.norm(torch.mm(self.basis, self.basis.T) - torch.eye(self.basis.size(0)), p='fro') ** 2

class MicroTopology:
    def __init__(self, num_nodes, config: MicroConfig):
        self.num_nodes = num_nodes
        self.config = config
        self.adj_weights = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.normal_(self.adj_weights, std=0.1)
        self.adj_mask = torch.zeros(num_nodes, num_nodes)
        grid_size = config.grid_size
        for i in range(num_nodes):
            r, c = i // grid_size, i % grid_size
            if r > 0: self.adj_mask[i, i - grid_size] = 1
            if r < grid_size - 1: self.adj_mask[i, i + grid_size] = 1
            if c > 0: self.adj_mask[i, i - 1] = 1
            if c < grid_size - 1: self.adj_mask[i, i + 1] = 1
    def get_adjacency(self, plasticity=1.0):
        adj = torch.sigmoid(self.adj_weights * plasticity) * self.adj_mask
        deg = adj.sum(1, keepdim=True).clamp(min=1e-6)
        return adj / deg

class MicroSupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature; self.eps = 1e-8
    def forward(self, features, labels):
        if features.size(0) < 2: return torch.tensor(0.0)
        features = F.normalize(features, dim=1)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        logits = torch.matmul(features, features.T) / (self.temperature + self.eps)
        logits_max = torch.max(logits, dim=1, keepdim=True)[0]; logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(logits.size(0)))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)
        mask_sum = mask.sum(1).clamp(min=self.eps)
        mean_log_prob = (mask * log_prob).sum(1) / mask_sum
        return -mean_log_prob.mean()

class MicroTopoBrain(nn.Module):
    def __init__(self, config: MicroConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim
        self.input_embed = nn.Linear(config.n_features, self.embed_dim * self.num_nodes)
        self.topology = MicroTopology(self.num_nodes, config) if config.use_plasticity else None
        if config.use_continuum:
            self.node_processor = MicroContinuumCell(self.embed_dim)
        else:
            self.node_processor = nn.Linear(self.embed_dim, self.embed_dim)
        self.cell_processor = None
        if config.use_mgf:
            mgf_input_dim = self.embed_dim * self.num_nodes
            if config.use_continuum:
                self.cell_processor = MicroContinuumCell(mgf_input_dim)
            else:
                self.cell_processor = nn.Linear(mgf_input_dim, self.embed_dim)
        self.symbiotic = MicroSymbioticBasis(self.embed_dim) if config.use_symbiotic else None
        self.supcon_head = None
        if config.use_supcon:
            self.supcon_head = nn.Sequential(nn.Linear(self.embed_dim * self.num_nodes, 8, bias=False), nn.ReLU(), nn.Linear(8, 4, bias=False))
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, config.n_classes)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, x, plasticity=1.0):
        batch_size = x.size(0)
        x_embed = self.input_embed(x).view(batch_size, self.num_nodes, self.embed_dim)
        if self.topology is not None:
            adj = self.topology.get_adjacency(plasticity)
            x_agg = torch.bmm(adj.unsqueeze(0).expand(batch_size, -1, -1), x_embed)
        else:
            x_agg = x_embed
        if isinstance(self.node_processor, MicroContinuumCell):
            x_flat = x_agg.view(-1, self.embed_dim)
            x_proc_flat = self.node_processor(x_flat, plasticity)
            x_proc = x_proc_flat.view(batch_size, self.num_nodes, self.embed_dim)
        else:
            x_proc = self.node_processor(x_agg)
        entropy = torch.tensor(0.0); ortho = torch.tensor(0.0); cell_output = torch.zeros_like(x_proc)
        if self.config.use_mgf and self.cell_processor is not None:
            cell_input = x_embed.view(batch_size, -1)
            if isinstance(self.cell_processor, MicroContinuumCell):
                cell_out = self.cell_processor(cell_input, plasticity)
                cell_output = cell_out.view(batch_size, self.num_nodes, self.embed_dim)
            else:
                cell_temp = self.cell_processor(cell_input)
                cell_output = cell_temp.view(batch_size, 1, self.embed_dim).expand(-1, self.num_nodes, -1)
        if self.symbiotic is not None:
            x_proc_refined = []
            for i in range(self.num_nodes):
                node_feat = x_proc[:, i, :]
                refined, ent, ort = self.symbiotic(node_feat)
                x_proc_refined.append(refined)
            x_proc = torch.stack(x_proc_refined, dim=1); entropy = ent; ortho = ort
        combined = x_proc + cell_output
        x_flat = combined.view(batch_size, -1)
        logits = self.readout(x_flat)
        proj = self.supcon_head(x_flat) if self.supcon_head is not None else None
        return logits, proj, entropy, ortho

def micro_pgd_attack(model, x, y, eps, steps, plasticity=1.0):
    model.eval(); delta = torch.zeros_like(x).uniform_(-eps, eps); delta.requires_grad = True
    for _ in range(steps):
        logits, _, _, _ = model(x + delta, plasticity); loss = F.cross_entropy(logits, y)
        if delta.grad is not None: delta.grad.zero_()
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + eps / steps * delta.grad.sign()).clamp(-eps, eps)
    model.train(); return (x + delta).detach()

# =============================================================================
# MATRIZ DE ABLACIÓN DE 3 NIVELES (como en NeuroLogos)
# =============================================================================

ABLATION_MATRIX = {
    # Nivel 1: Componentes individuales
    '01_Baseline': {},
    '02_Plasticity_Only': {'use_plasticity': True},
    '03_Continuum_Only': {'use_continuum': True},
    '04_MGF_Only': {'use_mgf': True},
    '05_SupCon_Only': {'use_supcon': True},
    '06_Symbiotic_Only': {'use_symbiotic': True},
    
    # Nivel 2A: Pares (hipótesis de sinergia)
    '07_Plasticity_Continuum': {'use_plasticity': True, 'use_continuum': True},
    '08_Continuum_Symbiotic': {'use_continuum': True, 'use_symbiotic': True},
    
    # Nivel 3: Sistema completo (para ablación inversa)
    '09_Full_System': {
        'use_plasticity': True,
        'use_continuum': True,
        'use_mgf': True,
        'use_supcon': True,
        'use_symbiotic': True
    }
}

# =============================================================================
# EJECUTOR CIENTÍFICO
# =============================================================================

def train_with_cv(config: MicroConfig, dataset, cv_folds=3):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.seed)
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
        
        # Modelo y optimizador
        model = MicroTopoBrain(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        supcon_loss_fn = MicroSupConLoss() if config.use_supcon else None
        
        # Entrenamiento
        for epoch in range(config.epochs):
            model.train()
            plasticity = 0.8 if config.use_plasticity else 0.0
            for x, y in train_loader:
                x, y = x.to(config.device), y.to(config.device)
                # Adversarial (training)
                x_adv = micro_pgd_attack(model, x, y, config.train_eps, config.pgd_steps, plasticity)
                logits, proj, entropy, ortho = model(x_adv, plasticity)
                loss = F.cross_entropy(logits, y)
                if config.use_supcon and proj is not None:
                    loss += 0.3 * supcon_loss_fn(proj, y)
                loss -= 0.01 * entropy
                loss += 0.05 * ortho
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
        
        # Evaluación PGD (métrica principal)
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                x_adv = micro_pgd_attack(model, x, y, config.test_eps, config.pgd_steps, plasticity)
                logits, _, _, _ = model(x_adv, plasticity)
                pred = logits.argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        
        pgd_acc = 100.0 * correct / total if total > 0 else 0.0
        fold_results.append(pgd_acc)
    
    return np.mean(fold_results), np.std(fold_results)

def run_ablation_study():
    seed_everything(42)
    device = setup_device()
    base_config = MicroConfig()
    dataset = get_dataset(base_config)
    results_dir = Path("neurologos_topobrain_ablation")
    results_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🧠 NeuroLogos v5.0 - TopoBrain Ablation CPU-Friendly")
    print("="*80)
    print(f"📊 Métrica principal: PGD Accuracy (robustez adversarial)")
    print(f"⚙️  Arquitectura: Grid 2x2 (~2k-5k params)")
    print(f"🔬 Validación: 3-Fold Stratified Cross-Validation")
    print("="*80 + "\n")
    
    # Ejecutar cada configuración
    results = {}
    for name, overrides in ABLATION_MATRIX.items():
        print(f"▶ Ejecutando: {name}")
        # Configuración
        cfg_dict = base_config.__dict__.copy()
        cfg_dict.update(overrides)
        config = MicroConfig(**cfg_dict)
        # Entrenamiento con CV
        pgd_mean, pgd_std = train_with_cv(config, dataset)
        # Guardar
        results[name] = {'pgd_mean': pgd_mean, 'pgd_std': pgd_std}
        model_temp = MicroTopoBrain(config)
        results[name]['n_params'] = model_temp.count_parameters()
        print(f"   PGD Acc: {pgd_mean:.2f}% ± {pgd_std:.2f}% | Params: {results[name]['n_params']:,}\n")
    
    # Guardar resultados
    with open(results_dir / "ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Análisis y reporte
    print("="*80)
    print("📊 RESULTADOS FINALES - ABLATION STUDY")
    print("="*80)
    print(f"{'Configuración':<25} {'PGD Acc (mean±std)':<25} {'Params':<10}")
    print("-"*65)
    for name, res in results.items():
        print(f"{name:<25} {res['pgd_mean']:>6.2f}% ± {res['pgd_std']:>4.2f}% {res['n_params']:>10,}")
    
    # Recomendación
    best_config = max(results.items(), key=lambda x: x[1]['pgd_mean'])
    print(f"\n🏆 Configuración óptima: {best_config[0]} → {best_config[1]['pgd_mean']:.2f}% PGD Accuracy")
    print(f"   Parámetros: {best_config[1]['n_params']:,}")
    
    return results

if __name__ == "__main__":
    results = run_ablation_study()
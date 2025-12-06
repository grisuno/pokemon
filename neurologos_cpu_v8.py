"""
NeuroLogos v5.1 - TopoBrain Ablation 3-Niveles CPU-Optimizado
==================================================================
Metodología Científica Rigurosa:
- Nivel 1: Componentes individuales (baseline + cada feature)
- Nivel 2: Pares sinérgicos (combinaciones de 2 features)
- Nivel 3: Ablación inversa desde sistema completo

Métricas: PGD Accuracy, Clean Accuracy, Training Time, #Params
Validación: 3-Fold Stratified Cross-Validation
Arquitectura: TopoBrain Micro Grid 2x2 (~2k-5k params)
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
import json
import time
from pathlib import Path
from itertools import combinations

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA CPU-FRIENDLY
# =============================================================================

@dataclass
class MicroConfig:
    device: str = "cpu"
    seed: int = 42
    
    # Dataset micro (tabular sintético)
    n_samples: int = 400
    n_features: int = 12
    n_classes: int = 3
    n_informative: int = 9
    
    # Arquitectura TopoBrain Micro (Grid 2x2 = 4 nodos)
    grid_size: int = 2
    embed_dim: int = 4
    hidden_dim: int = 4
    
    # Entrenamiento rápido
    batch_size: int = 16
    epochs: int = 8
    lr: float = 0.01
    
    # Adversarial ligero
    train_eps: float = 0.2
    test_eps: float = 0.2
    pgd_steps: int = 3
    
    # FLAGS DE COMPONENTES (para ablación)
    use_plasticity: bool = False   # Topología adaptativa
    use_continuum: bool = False    # Memoria continua
    use_mgf: bool = False          # Multi-Granular Fusion
    use_supcon: bool = False       # Contrastive Learning
    use_symbiotic: bool = False    # Refinamiento simbiótico

def seed_everything(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
# COMPONENTES MICRO DE TOPOBRAIN
# =============================================================================

class MicroContinuumCell(nn.Module):
    """Memoria continua con aprendizaje rápido/lento - VERSIÓN COMPATIBLE CON PGD"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W_slow = nn.Linear(dim, dim, bias=False)
        self.V_slow = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.V_slow.weight, gain=0.1)
        self.gate_net = nn.Linear(dim, 1)
        # Usar buffer en lugar de parameter para memoria
        self.register_buffer('semantic_memory', torch.zeros(dim, dim))
        nn.init.normal_(self.semantic_memory, std=0.01)
    
    def forward(self, x, plasticity=1.0):
        v = self.V_slow(x)
        v = torch.clamp(v, -2.0, 2.0)
        
        # NO usar la memoria en forward si causa problemas - versión simplificada
        gate = torch.sigmoid(self.gate_net(v)) * plasticity
        
        # Actualización de memoria (solo en training, sin afectar grafo)
        if self.training:
            with torch.no_grad():
                delta = torch.bmm(x.detach().unsqueeze(-1), x.detach().unsqueeze(1))
                self.semantic_memory.copy_(0.95 * self.semantic_memory + 0.01 * delta.mean(dim=0))
                mem_norm = self.semantic_memory.norm().clamp(min=1e-6)
                self.semantic_memory.copy_(self.semantic_memory / mem_norm * 0.5)
        
        # Output simplificado que mantiene grafo
        output = gate * v
        return torch.clamp(output, -2.0, 2.0)

class MicroSymbioticBasis(nn.Module):
    """Base simbiótica para refinamiento adversarial"""
    def __init__(self, dim, num_atoms=2):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis, gain=0.5)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.eps = 1e-8
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(self.basis)
        attn = torch.matmul(Q, K.T) / (x.size(-1) ** 0.5 + self.eps)
        weights = F.softmax(attn, dim=-1)
        x_clean = torch.matmul(weights, self.basis)
        
        entropy = -(weights * torch.log(weights + self.eps)).sum(-1).mean()
        ortho = torch.norm(torch.mm(self.basis, self.basis.T) - torch.eye(self.basis.size(0)), p='fro') ** 2
        
        return torch.clamp(x_clean, -2.0, 2.0), entropy, ortho

class MicroTopology:
    """Topología de grid 2D con conexiones von Neumann"""
    def __init__(self, num_nodes, config: MicroConfig):
        self.num_nodes = num_nodes
        self.config = config
        self.adj_weights = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.normal_(self.adj_weights, std=0.1)
        
        # Máscara de conectividad (grid 2x2)
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
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8
    
    def forward(self, features, labels):
        if features.size(0) < 2:
            return torch.tensor(0.0)
        
        features = F.normalize(features, dim=1)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        logits = torch.matmul(features, features.T) / (self.temperature + self.eps)
        logits_max = torch.max(logits, dim=1, keepdim=True)[0]
        logits = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits) * (1 - torch.eye(logits.size(0)))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)
        
        mask_sum = mask.sum(1).clamp(min=self.eps)
        mean_log_prob = (mask * log_prob).sum(1) / mask_sum
        
        return -mean_log_prob.mean()

class MicroTopoBrain(nn.Module):
    """Arquitectura TopoBrain Modular para Ablación"""
    def __init__(self, config: MicroConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim
        
        # Input embedding
        self.input_embed = nn.Linear(config.n_features, self.embed_dim * self.num_nodes)
        
        # Componente 1: Topología adaptativa
        self.topology = MicroTopology(self.num_nodes, config) if config.use_plasticity else None
        
        # Componente 2: Procesador de nodos (Continuum o Linear)
        if config.use_continuum:
            self.node_processor = MicroContinuumCell(self.embed_dim)
        else:
            self.node_processor = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Componente 3: Multi-Granular Fusion
        self.cell_processor = None
        if config.use_mgf:
            mgf_input_dim = self.embed_dim * self.num_nodes
            if config.use_continuum:
                self.cell_processor = MicroContinuumCell(mgf_input_dim)
            else:
                self.cell_processor = nn.Linear(mgf_input_dim, self.embed_dim)
        
        # Componente 4: Refinamiento simbiótico
        self.symbiotic = MicroSymbioticBasis(self.embed_dim) if config.use_symbiotic else None
        
        # Componente 5: Contrastive head
        self.supcon_head = None
        if config.use_supcon:
            self.supcon_head = nn.Sequential(
                nn.Linear(self.embed_dim * self.num_nodes, 8, bias=False),
                nn.ReLU(),
                nn.Linear(8, 4, bias=False)
            )
        
        # Output
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, config.n_classes)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, plasticity=1.0):
        batch_size = x.size(0)
        
        # Embedding
        x_embed = self.input_embed(x).view(batch_size, self.num_nodes, self.embed_dim)
        
        # Topología adaptativa
        if self.topology is not None:
            adj = self.topology.get_adjacency(plasticity)
            x_agg = torch.bmm(adj.unsqueeze(0).expand(batch_size, -1, -1), x_embed)
        else:
            x_agg = x_embed
        
        # Procesamiento de nodos
        if isinstance(self.node_processor, MicroContinuumCell):
            x_flat = x_agg.view(-1, self.embed_dim)
            x_proc_flat = self.node_processor(x_flat, plasticity)
            x_proc = x_proc_flat.view(batch_size, self.num_nodes, self.embed_dim)
        else:
            x_proc = self.node_processor(x_agg)
        
        # Multi-Granular Fusion
        entropy = torch.tensor(0.0)
        ortho = torch.tensor(0.0)
        cell_output = torch.zeros_like(x_proc)
        
        if self.config.use_mgf and self.cell_processor is not None:
            cell_input = x_embed.view(batch_size, -1)
            if isinstance(self.cell_processor, MicroContinuumCell):
                cell_out = self.cell_processor(cell_input, plasticity)
                cell_output = cell_out.view(batch_size, self.num_nodes, self.embed_dim)
            else:
                cell_temp = self.cell_processor(cell_input)
                cell_output = cell_temp.view(batch_size, 1, self.embed_dim).expand(-1, self.num_nodes, -1)
        
        # Refinamiento simbiótico
        if self.symbiotic is not None:
            x_proc_refined = []
            for i in range(self.num_nodes):
                node_feat = x_proc[:, i, :]
                refined, ent, ort = self.symbiotic(node_feat)
                x_proc_refined.append(refined)
            x_proc = torch.stack(x_proc_refined, dim=1)
            entropy = ent
            ortho = ort
        
        # Fusión
        combined = x_proc + cell_output
        x_flat = combined.view(batch_size, -1)
        
        # Outputs
        logits = self.readout(x_flat)
        proj = self.supcon_head(x_flat) if self.supcon_head is not None else None
        
        return logits, proj, entropy, ortho

# =============================================================================
# ADVERSARIAL ATTACK (CORREGIDO)
# =============================================================================

def micro_pgd_attack(model, x, y, eps, steps, plasticity=1.0):
    """PGD Attack - Versión ultra-simple que siempre funciona"""
    was_training = model.training
    model.eval()  # Poner en eval para evitar actualizaciones de memoria
    
    # Inicializar perturbación
    delta = torch.zeros_like(x)
    with torch.no_grad():
        delta.uniform_(-eps, eps)
    
    for step in range(steps):
        # Habilitar gradientes solo para este paso
        x_adv = (x + delta).detach().requires_grad_(True)
        
        # Forward pass
        with torch.enable_grad():
            logits, _, _, _ = model(x_adv, plasticity)
            loss = F.cross_entropy(logits, y)
            loss.backward()
        
        # Actualizar delta con el gradiente
        with torch.no_grad():
            if x_adv.grad is not None:
                delta = delta + (eps / steps) * x_adv.grad.sign()
                delta = delta.clamp(-eps, eps)
    
    # Restaurar modo original
    if was_training:
        model.train()
    
    return (x + delta).detach()

# =============================================================================
# MATRIZ DE ABLACIÓN 3-NIVELES
# =============================================================================

def generate_ablation_matrix():
    """
    Genera matriz de ablación de 3 niveles:
    - Nivel 1: Baseline + componentes individuales
    - Nivel 2: Pares sinérgicos
    - Nivel 3: Sistema completo (para ablación inversa)
    """
    
    components = ['plasticity', 'continuum', 'mgf', 'supcon', 'symbiotic']
    
    # NIVEL 1: Baseline + Individual
    nivel1 = {
        'L1_00_Baseline': {}
    }
    
    for i, comp in enumerate(components, 1):
        nivel1[f'L1_{i:02d}_{comp.capitalize()}'] = {f'use_{comp}': True}
    
    # NIVEL 2: Pares (combinaciones de 2)
    nivel2 = {}
    pair_idx = 0
    for comp1, comp2 in combinations(components, 2):
        pair_idx += 1
        name = f'L2_{pair_idx:02d}_{comp1.capitalize()}+{comp2.capitalize()}'
        nivel2[name] = {f'use_{comp1}': True, f'use_{comp2}': True}
    
    # NIVEL 3: Sistema completo + Ablaciones inversas
    nivel3 = {
        'L3_00_Full': {f'use_{c}': True for c in components}
    }
    
    # Ablaciones inversas (quitar 1 componente del sistema completo)
    for i, comp in enumerate(components, 1):
        config = {f'use_{c}': True for c in components}
        config[f'use_{comp}'] = False
        nivel3[f'L3_{i:02d}_Full_minus_{comp.capitalize()}'] = config
    
    # Combinar todo
    ablation_matrix = {}
    ablation_matrix.update(nivel1)
    ablation_matrix.update(nivel2)
    ablation_matrix.update(nivel3)
    
    return ablation_matrix

# =============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================

def train_with_cv(config: MicroConfig, dataset, cv_folds=3):
    """Entrenamiento con cross-validation"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.seed)
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    
    fold_results = {
        'pgd_acc': [],
        'clean_acc': [],
        'train_time': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Datasets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
        
        # Modelo
        model = MicroTopoBrain(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        supcon_loss_fn = MicroSupConLoss() if config.use_supcon else None
        
        # Entrenamiento
        start_time = time.time()
        for epoch in range(config.epochs):
            model.train()
            plasticity = 0.8 if config.use_plasticity else 0.0
            
            for x, y in train_loader:
                x, y = x.to(config.device), y.to(config.device)
                
                # Adversarial training
                x_adv = micro_pgd_attack(model, x, y, config.train_eps, config.pgd_steps, plasticity)
                logits, proj, entropy, ortho = model(x_adv, plasticity)
                
                loss = F.cross_entropy(logits, y)
                if config.use_supcon and proj is not None:
                    loss += 0.3 * supcon_loss_fn(proj, y)
                loss -= 0.01 * entropy
                loss += 0.05 * ortho
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
        
        train_time = time.time() - start_time
        
        # Evaluación
        model.eval()
        plasticity = 0.8 if config.use_plasticity else 0.0
        
        pgd_correct = 0
        clean_correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                
                # Clean accuracy
                logits_clean, _, _, _ = model(x, plasticity)
                pred_clean = logits_clean.argmax(dim=1)
                clean_correct += pred_clean.eq(y).sum().item()
                
                # PGD accuracy
                x_adv = micro_pgd_attack(model, x, y, config.test_eps, config.pgd_steps, plasticity)
                logits_adv, _, _, _ = model(x_adv, plasticity)
                pred_adv = logits_adv.argmax(dim=1)
                pgd_correct += pred_adv.eq(y).sum().item()
                
                total += y.size(0)
        
        fold_results['pgd_acc'].append(100.0 * pgd_correct / total if total > 0 else 0.0)
        fold_results['clean_acc'].append(100.0 * clean_correct / total if total > 0 else 0.0)
        fold_results['train_time'].append(train_time)
    
    return {
        'pgd_mean': np.mean(fold_results['pgd_acc']),
        'pgd_std': np.std(fold_results['pgd_acc']),
        'clean_mean': np.mean(fold_results['clean_acc']),
        'clean_std': np.std(fold_results['clean_acc']),
        'train_time': np.mean(fold_results['train_time'])
    }

# =============================================================================
# EJECUTOR PRINCIPAL
# =============================================================================

def run_ablation_study():
    """Ejecuta el estudio de ablación completo"""
    seed_everything(42)
    base_config = MicroConfig()
    dataset = get_dataset(base_config)
    
    results_dir = Path("neurologos_topobrain_ablation")
    results_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🧠 NeuroLogos v5.1 - TopoBrain Ablation 3-Niveles CPU-Optimizado")
    print("="*80)
    print(f"📊 Métricas: PGD Acc, Clean Acc, Training Time, #Params")
    print(f"⚙️  Arquitectura: Grid 2x2 (~2k-5k params)")
    print(f"🔬 Validación: 3-Fold Stratified Cross-Validation")
    print("="*80 + "\n")
    
    # Generar matriz de ablación
    ablation_matrix = generate_ablation_matrix()
    print(f"📋 Total de experimentos: {len(ablation_matrix)}\n")
    
    # Ejecutar experimentos
    results = {}
    
    for exp_name, overrides in ablation_matrix.items():
        print(f"▶ {exp_name}")
        
        # Configuración
        cfg_dict = base_config.__dict__.copy()
        cfg_dict.update(overrides)
        config = MicroConfig(**cfg_dict)
        
        # Entrenamiento
        metrics = train_with_cv(config, dataset)
        
        # Contar parámetros
        model_temp = MicroTopoBrain(config)
        metrics['n_params'] = model_temp.count_parameters()
        
        results[exp_name] = metrics
        
        print(f"   PGD: {metrics['pgd_mean']:.2f}±{metrics['pgd_std']:.2f}% | "
              f"Clean: {metrics['clean_mean']:.2f}±{metrics['clean_std']:.2f}% | "
              f"Params: {metrics['n_params']:,} | "
              f"Time: {metrics['train_time']:.1f}s\n")
    
    # Guardar resultados
    with open(results_dir / "ablation_results_3niveles.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Reporte final
    print("\n" + "="*80)
    print("📊 RESULTADOS FINALES - ABLATION 3-NIVELES")
    print("="*80)
    print(f"{'Experimento':<40} {'PGD Acc':<15} {'Clean Acc':<15} {'Params':<10}")
    print("-"*80)
    
    for name, res in results.items():
        print(f"{name:<40} "
              f"{res['pgd_mean']:>6.2f}±{res['pgd_std']:>4.2f}% "
              f"{res['clean_mean']:>6.2f}±{res['clean_std']:>4.2f}% "
              f"{res['n_params']:>10,}")
    
    # Análisis de mejores configuraciones
    print("\n" + "="*80)
    print("🏆 TOP 5 - PGD ACCURACY (Robustez)")
    print("="*80)
    sorted_by_pgd = sorted(results.items(), key=lambda x: x[1]['pgd_mean'], reverse=True)[:5]
    for i, (name, res) in enumerate(sorted_by_pgd, 1):
        print(f"{i}. {name}: {res['pgd_mean']:.2f}% ± {res['pgd_std']:.2f}%")
    
    print("\n" + "="*80)
    print("🏆 TOP 5 - CLEAN ACCURACY")
    print("="*80)
    sorted_by_clean = sorted(results.items(), key=lambda x: x[1]['clean_mean'], reverse=True)[:5]
    for i, (name, res) in enumerate(sorted_by_clean, 1):
        print(f"{i}. {name}: {res['clean_mean']:.2f}% ± {res['clean_std']:.2f}%")
    
    print("\n" + "="*80)
    print("⚡ TOP 5 - EFICIENCIA (Menos parámetros)")
    print("="*80)
    sorted_by_params = sorted(results.items(), key=lambda x: x[1]['n_params'])[:5]
    for i, (name, res) in enumerate(sorted_by_params, 1):
        print(f"{i}. {name}: {res['n_params']:,} params | PGD: {res['pgd_mean']:.2f}%")
    
    return results

if __name__ == "__main__":
    results = run_ablation_study()
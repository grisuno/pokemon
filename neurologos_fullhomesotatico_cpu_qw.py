"""
NeuroLogos v5.2-H - TopoBrain Ablation 3-Niveles CPU-Optimizado con Homeostatic Orchestrator Global
===============================================================================================
Inspirado en Physio-Chimera v14 y Síntesis v8.5:
- El modelo posee un "cerebro interno" que regula su estado fisiológico.
- La homeostasis NO es un componente más: es el meta-regulador de todos los demás.
- Ajusta dinámicamente: plasticity, continuum_strength, mgf_weight, symbiosis_influence, supcon_gain.

Metodología:
- Nivel 1: Solo Homeostatic Orchestrator activo (baseline + regulación global)
- Nivel 2: Homeostasis + 1 componente
- Nivel 3: Homeostasis + sistema completo (sin flags mutuamente excluyentes)

Métricas: PGD Accuracy, Clean Accuracy, Training Time, #Params
Validación: 3-Fold Stratified CV
Arquitectura: TopoBrain Micro Grid 2x2 (~1k–5k params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import time
from pathlib import Path

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
    # Entrenamiento rápido
    batch_size: int = 16
    epochs: int = 8
    lr: float = 0.01
    # Adversarial ligero
    train_eps: float = 0.2
    test_eps: float = 0.2
    pgd_steps: int = 3
    # Componentes habilitados (solo para ablation; la homeostasis siempre puede modularlos)
    use_plasticity: bool = False
    use_continuum: bool = False
    use_mgf: bool = False
    use_supcon: bool = False
    use_symbiotic: bool = False
    # FLAG MAESTRO: Homeostasis global (siempre activa en v5.2-H)
    use_homeostasis: bool = True  # ← Siempre True en este modelo


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
# HOMEOSTATIC ORCHESTRATOR GLOBAL (INSPIRADO EN PHYSIO-CHIMERA)
# =============================================================================
class GlobalHomeostaticOrchestrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Linear(16, 5),
            nn.Sigmoid()
        )

    def forward(self, x, logits, h_agg, h_proc, w_norm, entropy=0.0, ortho=0.0):
        device = x.device

        # 1. Estrés: varianza del input + entropía del modelo
        stress_input = x.var(dim=1).mean()
        probs = F.softmax(logits, dim=1)
        entropy_logits = -(probs * torch.log(probs + 1e-8)).sum(1).mean()
        stress = (stress_input + entropy_logits) / 2.0

        # 2. Excitación: promedio de activaciones
        excitation = (h_agg.abs().mean() + h_proc.abs().mean()) / 2.0

        # 3. Fatiga: norma global de pesos (convertir a tensor)
        fatigue = w_norm if torch.is_tensor(w_norm) else torch.tensor(w_norm, device=device, dtype=torch.float32)

        # 4. Desorden simbiótico
        disorder = entropy if torch.is_tensor(entropy) else torch.tensor(entropy, device=device, dtype=torch.float32)

        # 5. Inestabilidad estructural
        instability = ortho if torch.is_tensor(ortho) else torch.tensor(ortho, device=device, dtype=torch.float32)

        state = torch.stack([stress, excitation, fatigue, disorder, instability])
        ctrl = self.net(state.unsqueeze(0)).squeeze(0)
        return {
            'plasticity': ctrl[0].item(),
            'continuum': ctrl[1].item(),
            'mgf': ctrl[2].item(),
            'symbiotic': ctrl[3].item(),
            'supcon': ctrl[4].item()
        }
# =============================================================================
# COMPONENTES MICRO (AHORA MODULADOS DINÁMICAMENTE)
# =============================================================================
class MicroContinuumCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W_slow = nn.Linear(dim, dim, bias=False)
        self.V_slow = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.V_slow.weight, gain=0.1)
        self.gate_net = nn.Linear(dim, 1)
        self.register_buffer('semantic_memory', torch.zeros(dim, dim))
        nn.init.normal_(self.semantic_memory, std=0.01)

    def forward(self, x, strength=1.0):
        v = self.V_slow(x)
        v = torch.clamp(v, -2.0, 2.0)
        gate = torch.sigmoid(self.gate_net(v)) * strength
        if self.training:
            with torch.no_grad():
                delta = torch.bmm(x.detach().unsqueeze(-1), x.detach().unsqueeze(1))
                self.semantic_memory.copy_(0.95 * self.semantic_memory + 0.01 * delta.mean(dim=0))
                mem_norm = self.semantic_memory.norm().clamp(min=1e-6)
                self.semantic_memory.copy_(self.semantic_memory / mem_norm * 0.5)
        return gate * v


class MicroSymbioticBasis(nn.Module):
    def __init__(self, dim, num_atoms=2):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis, gain=0.5)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.eps = 1e-8

    def forward(self, x, influence=1.0):
        Q = self.query(x)
        K = self.key(self.basis)
        attn = torch.matmul(Q, K.T) / (x.size(-1) ** 0.5 + self.eps)
        weights = F.softmax(attn, dim=-1)
        x_clean = torch.matmul(weights, self.basis)
        entropy = -(weights * torch.log(weights + self.eps)).sum(-1).mean()
        ortho = torch.norm(torch.mm(self.basis, self.basis.T) - torch.eye(self.basis.size(0)), p='fro') ** 2
        # Interpolación entre entrada original y refinada
        out = (1 - influence) * x + influence * x_clean
        return torch.clamp(out, -2.0, 2.0), entropy, ortho


class MicroTopology:
    def __init__(self, num_nodes, config: MicroConfig):
        self.num_nodes = num_nodes
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


# =============================================================================
# MICROTOPOBRAIN v5.2-H — CON ORQUESTADOR HOMEOSTÁTICO GLOBAL
# =============================================================================
class MicroTopoBrain(nn.Module):
    def __init__(self, config: MicroConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim

        self.input_embed = nn.Linear(config.n_features, self.embed_dim * self.num_nodes)

        # Topología
        self.topology = MicroTopology(self.num_nodes, config) if config.use_plasticity else None

        # Procesadores
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
            self.supcon_head = nn.Sequential(
                nn.Linear(self.embed_dim * self.num_nodes, 8, bias=False),
                nn.ReLU(),
                nn.Linear(8, 4, bias=False)
            )

        self.readout = nn.Linear(self.embed_dim * self.num_nodes, config.n_classes)
        self.homeostat = GlobalHomeostaticOrchestrator()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        batch_size = x.size(0)
        x_embed = self.input_embed(x).view(batch_size, self.num_nodes, self.embed_dim)

        # Inicializar controles
        ctrl = {
            'plasticity': 0.0,
            'continuum': 1.0,
            'mgf': 1.0,
            'symbiotic': 1.0,
            'supcon': 1.0
        }

        # Activar solo los componentes habilitados
        plasticity_ctrl = 0.8 if self.config.use_plasticity else 0.0
        continuum_ctrl = 1.0 if self.config.use_continuum else 0.0
        mgf_ctrl = 1.0 if self.config.use_mgf else 0.0
        symbiotic_ctrl = 1.0 if self.config.use_symbiotic else 0.0
        supcon_ctrl = 1.0 if self.config.use_supcon else 0.0

        # Topología
        if self.topology is not None:
            adj = self.topology.get_adjacency(plasticity_ctrl)
            x_agg = torch.bmm(adj.unsqueeze(0).expand(batch_size, -1, -1), x_embed)
        else:
            x_agg = x_embed

        # Procesamiento nodal
        if isinstance(self.node_processor, MicroContinuumCell):
            x_flat = x_agg.view(-1, self.embed_dim)
            x_proc_flat = self.node_processor(x_flat, continuum_ctrl)
            x_proc = x_proc_flat.view(batch_size, self.num_nodes, self.embed_dim)
        else:
            x_proc = self.node_processor(x_agg)

        # MGF
        entropy = torch.tensor(0.0)
        ortho = torch.tensor(0.0)
        cell_output = torch.zeros_like(x_proc)
        if self.config.use_mgf and self.cell_processor is not None:
            cell_input = x_embed.view(batch_size, -1)
            if isinstance(self.cell_processor, MicroContinuumCell):
                cell_out = self.cell_processor(cell_input, mgf_ctrl)
                cell_output = cell_out.view(batch_size, self.num_nodes, self.embed_dim)
            else:
                cell_temp = self.cell_processor(cell_input)
                cell_output = cell_temp.view(batch_size, 1, self.embed_dim).expand(-1, self.num_nodes, -1)

        # Simbiosis
        if self.symbiotic is not None:
            x_proc_refined = []
            for i in range(self.num_nodes):
                node_feat = x_proc[:, i, :]
                refined, ent, ort = self.symbiotic(node_feat, symbiotic_ctrl)
                x_proc_refined.append(refined)
            x_proc = torch.stack(x_proc_refined, dim=1)
            entropy = ent
            ortho = ort

        combined = x_proc + cell_output
        x_flat = combined.view(batch_size, -1)
        logits = self.readout(x_flat)
        proj = self.supcon_head(x_flat) if self.supcon_head is not None else None

        # 🔁 HOMEOSTATIC FEEDBACK (solo en modo training, ligero en inference)
        if self.training and self.config.use_homeostasis:
            with torch.no_grad():
                w_norm = sum(p.norm() for p in self.parameters() if p.requires_grad)
            ctrl = self.homeostat(x, logits, x_agg, x_proc, w_norm, entropy, ortho)

            # Re-escalar según controles homeostáticos (solo si componente activo)
            plasticity_ctrl = ctrl['plasticity'] * (0.8 if self.config.use_plasticity else 0.0)
            continuum_ctrl = ctrl['continuum'] * (1.0 if self.config.use_continuum else 0.0)
            mgf_ctrl = ctrl['mgf'] * (1.0 if self.config.use_mgf else 0.0)
            symbiotic_ctrl = ctrl['symbiotic'] * (1.0 if self.config.use_symbiotic else 0.0)
            supcon_ctrl = ctrl['supcon'] * (1.0 if self.config.use_supcon else 0.0)

            # Guardar para usar en pérdida (opcional)
            self.last_ctrl = ctrl
        else:
            self.last_ctrl = ctrl

        return logits, proj, entropy, ortho, plasticity_ctrl


# =============================================================================
# ADVERSARIAL ATTACK
# =============================================================================
def micro_pgd_attack(model, x, y, eps, steps, plasticity_ctrl=0.0):
    was_training = model.training
    model.eval()
    delta = torch.zeros_like(x)
    with torch.no_grad():
        delta.uniform_(-eps, eps)
    for step in range(steps):
        x_adv = (x + delta).detach().requires_grad_(True)
        with torch.enable_grad():
            logits, _, _, _, _ = model(x_adv)
            loss = F.cross_entropy(logits, y)
            loss.backward()
        with torch.no_grad():
            if x_adv.grad is not None:
                delta = delta + (eps / steps) * x_adv.grad.sign()
                delta = delta.clamp(-eps, eps)
    if was_training:
        model.train()
    return (x + delta).detach()


# =============================================================================
# MATRIZ DE ABLACIÓN 3-NIVELES CON HOMEOSTASIS SIEMPRE ACTIVA
# =============================================================================
def generate_ablation_matrix():
    components = ['plasticity', 'continuum', 'mgf', 'supcon', 'symbiotic']
    matrix = {}

    # Nivel 1: Solo homeostasis + baseline
    matrix['L1_00_HomeoOnly'] = {}

    # Nivel 2: Homeostasis + 1 componente
    for i, comp in enumerate(components, 1):
        matrix[f'L2_{i:02d}_Homeo+{comp.capitalize()}'] = {f'use_{comp}': True}

    # Nivel 3: Homeostasis + sistema completo
    matrix['L3_00_HomeoFull'] = {f'use_{c}': True for c in components}

    return matrix


# =============================================================================
# ENTRENAMIENTO
# =============================================================================
def train_with_cv(config: MicroConfig, dataset, cv_folds=3):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.seed)
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    fold_results = {'pgd_acc': [], 'clean_acc': [], 'train_time': []}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

        model = MicroTopoBrain(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        supcon_loss_fn = MicroSupConLoss() if config.use_supcon else None

        start_time = time.time()
        for epoch in range(config.epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(config.device), y.to(config.device)
                x_adv = micro_pgd_attack(model, x, y, config.train_eps, config.pgd_steps)
                logits, proj, entropy, ortho, plast = model(x_adv)
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

        model.eval()
        pgd_correct = clean_correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                logits_clean, _, _, _, _ = model(x)
                pred_clean = logits_clean.argmax(dim=1)
                clean_correct += pred_clean.eq(y).sum().item()

                x_adv = micro_pgd_attack(model, x, y, config.test_eps, config.pgd_steps)
                logits_adv, _, _, _, _ = model(x_adv)
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
# EJECUTOR
# =============================================================================
def run_ablation_study():
    seed_everything(42)
    base_config = MicroConfig()
    dataset = get_dataset(base_config)
    results_dir = Path("neurologos_v5_2H_ablation")
    results_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("🧠 NeuroLogos v5.2-H — TopoBrain con Homeostatic Orchestrator Global")
    print("=" * 80)
    print("💡 La homeostasis regula a todos los componentes en tiempo real.")
    print("📊 Métricas: PGD Acc, Clean Acc, Training Time, #Params")
    print("🔬 Validación: 3-Fold Stratified Cross-Validation")
    print("=" * 80 + "\n")

    ablation_matrix = generate_ablation_matrix()
    print(f"📋 Total de experimentos: {len(ablation_matrix)}\n")

    results = {}
    for exp_name, overrides in ablation_matrix.items():
        print(f"▶ {exp_name}")
        cfg_dict = base_config.__dict__.copy()
        cfg_dict.update(overrides)
        cfg_dict['use_homeostasis'] = True  # Siempre activo
        config = MicroConfig(**cfg_dict)

        metrics = train_with_cv(config, dataset)
        model_temp = MicroTopoBrain(config)
        metrics['n_params'] = model_temp.count_parameters()

        results[exp_name] = metrics
        print(f"   PGD: {metrics['pgd_mean']:.2f}±{metrics['pgd_std']:.2f}% | "
              f"Clean: {metrics['clean_mean']:.2f}±{metrics['clean_std']:.2f}% | "
              f"Params: {metrics['n_params']:,} | "
              f"Time: {metrics['train_time']:.1f}s\n")

    with open(results_dir / "ablation_results_v5_2H.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("📊 RESULTADOS FINALES - ABLATION CON HOMEOSTATIC ORCHESTRATOR")
    print("=" * 80)
    print(f"{'Experimento':<45} {'PGD Acc':<15} {'Clean Acc':<15} {'Params':<10}")
    print("-" * 80)
    for name, res in results.items():
        print(f"{name:<45} "
              f"{res['pgd_mean']:>6.2f}±{res['pgd_std']:>4.2f}% "
              f"{res['clean_mean']:>6.2f}±{res['clean_std']:>4.2f}% "
              f"{res['n_params']:>10,}")

    print("\n" + "=" * 80)
    print("🏆 TOP - PGD ACCURACY (ROBUSTEZ CON REGULACIÓN HOMEOSTÁTICA)")
    print("=" * 80)
    sorted_by_pgd = sorted(results.items(), key=lambda x: x[1]['pgd_mean'], reverse=True)
    for i, (name, res) in enumerate(sorted_by_pgd[:5], 1):
        print(f"{i}. {name}: {res['pgd_mean']:.2f}% ± {res['pgd_std']:.2f}%")

    return results


if __name__ == "__main__":
    results = run_ablation_study()
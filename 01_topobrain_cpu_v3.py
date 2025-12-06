# 01_topobrain_cpu_noir_ablation.py
# Ablation Científico Riguroso – TopoBrain Tabular (NOIR-aware, CPU-only, miles de parámetros)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import json
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import psutil
import gc
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN (Reducida para CPU, pero con todos los flags)
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
    grid_size: int = 2                    # 2x2 = 4 nodos → ~12k params
    use_plasticity: bool = False
    use_nested_cells: bool = False
    use_mgf: bool = False
    use_supcon: bool = False
    use_symbiotic: bool = False
    use_orchestrator: bool = False
    use_adaptive_topology: bool = False
    epochs: int = 10
    lr_main: float = 0.01
    lr_topo: float = 0.01
    train_eps: float = 0.3
    test_eps: float = 0.3
    pgd_steps_train: int = 3
    pgd_steps_test: int = 3
    lambda_supcon: float = 0.5
    lambda_ortho: float = 0.01
    lambda_entropy: float = 0.001
    lambda_sparsity: float = 1e-4
    prune_threshold: float = 0.1
    topo_warmup_epochs: int = 2
    grad_clip_norm: float = 1.0

    def to_dict(self):
        return asdict(self)

# =============================================================================
# UTILIDADES
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
# COMPONENTES (FIJOS, sin simplificación lógica)
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
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_slow = nn.Linear(input_dim, hidden_dim, bias=False)
        self.V_slow = nn.Linear(input_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.V_slow.weight)
        self.forget_gate = nn.Sequential(nn.Linear(hidden_dim + input_dim, 1), nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Linear(hidden_dim + input_dim, 1), nn.Sigmoid())
        self.semantic_mix = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.semantic_memory = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        nn.init.orthogonal_(self.semantic_memory)
        self.semantic_memory.data *= 0.01
        self.register_buffer('memory_initialized', torch.tensor(0))
        
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
        with torch.no_grad():
            self.semantic_memory.data = (
                forget.mean().item() * self.semantic_memory.data +
                update.mean().item() * 0.1 * delta.mean(dim=0)
            ).detach()
            
        mix = self.semantic_mix(v)
        return mix * v + (1 - mix) * y_pred

class SymbioticBasisRefinement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(8, dim))
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

class PrefrontalOrchestrator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.state_dim = 8
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, 16),
            nn.LayerNorm(16), nn.GELU(),
            nn.Linear(16, 16), nn.LayerNorm(16), nn.GELU()
        )
        self.rnn = nn.GRU(16, 8, batch_first=True)
        self.context = torch.zeros(1, 1, 8)
        self.gate_names = ['plasticity', 'memory', 'defense', 'supcon', 'symbiotic', 'sparsity', 'lr_scale']
        self.gates = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(8, 1), nn.Sigmoid()) for name in self.gate_names[:-1]
        })
        self.gates['lr_scale'] = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())
        
    def forward(self, metrics: Dict[str, float]) -> Dict[str, torch.Tensor]:
        density = metrics.get('density', 0.1)
        loss = metrics.get('loss', 10.0)
        is_weak = density < 0.05
        is_failing = loss > 2.0
        
        if is_weak and is_failing:
            return {name: torch.tensor(v, device="cpu") for name, v in [
                ('plasticity', 0.1), ('memory', 0.95), ('defense', 0.8),
                ('supcon', 0.5), ('symbiotic', 0.0), ('sparsity', 0.0), ('lr_scale', 0.5)
            ]}
            
        state = torch.tensor([
            metrics.get('loss', 0.0), metrics.get('grad_norm', 0.0),
            metrics.get('sparsity', 0.5), metrics.get('entropy', 0.5),
            metrics.get('batch_var', 0.1), metrics.get('epoch_progress', 0.0),
            metrics.get('memory_norm', 1.0), metrics.get('L_score', 2.0)
        ], dtype=torch.float32, device="cpu").unsqueeze(0)
        
        enc = self.encoder(state)
        rnn_out, self.context = self.rnn(enc.unsqueeze(1), self.context)
        context = rnn_out.squeeze(1)
        
        controls = {}
        for name in self.gate_names[:-1]:
            controls[name] = self.gates[name](context).squeeze()
            
        lr_raw = self.gates['lr_scale'](context).squeeze()
        controls['lr_scale'] = lr_raw * 1.9 + 0.1
        
        if density < 0.10:
            controls['plasticity'] = torch.clamp(controls['plasticity'] + 0.2, 0.0, 1.0)
            controls['defense'] = torch.clamp(controls['defense'], 0.2, 0.6)
            if density < 0.06:
                controls['sparsity'] = torch.tensor(0.0, device="cpu")
                
        return controls
    
    def reset_context(self):
        self.context = torch.zeros_like(self.context)

# =============================================================================
# CAPA COMPLEJA Y MODELO
# =============================================================================
class AdaptiveCombinatorialComplexLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, config: Config, layer_type='midbrain'):
        super().__init__()
        self.num_nodes = num_nodes
        self.config = config
        self.use_mgf = config.use_mgf
        self.layer_type = layer_type
        
        if config.use_nested_cells:
            self.node_mapper = ContinuumMemoryCell(in_dim, hid_dim)
            self.cell_mapper = ContinuumMemoryCell(in_dim, hid_dim)
        else:
            self.node_mapper = nn.Linear(in_dim, hid_dim)
            self.cell_mapper = nn.Linear(in_dim, hid_dim)
            
        self.symbiotic = SymbioticBasisRefinement(hid_dim) if config.use_symbiotic else None
        
        if config.use_adaptive_topology:
            self.node_importance = nn.Parameter(torch.ones(num_nodes))
            
        # Corrección: El tamaño de entrada debe ser hid_dim * 2 para concatenar ambos tensores
        self.final_mix = nn.Linear(hid_dim * 2, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
        
        # Topología fija (grid 2x2)
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            r, c = i // config.grid_size, i % config.grid_size
            if r > 0: adj[i, i - config.grid_size] = 1
            if r < config.grid_size - 1: adj[i, i + config.grid_size] = 1
            if c > 0: adj[i, i - 1] = 1
            if c < config.grid_size - 1: adj[i, i + 1] = 1
            
        self.register_buffer('adj_base', adj)
        if config.use_plasticity:
            self.adj_weights = nn.Parameter(torch.randn_like(adj))
            
    def get_adj(self):
        if self.config.use_plasticity:
            return torch.sigmoid(self.adj_weights) * self.adj_base
        return self.adj_base
    
    def forward(self, x, controls=None):
        batch_size = x.size(0)
        adj = self.get_adj()
        
        if self.config.use_adaptive_topology and hasattr(self, 'node_importance'):
            gate = torch.sigmoid(self.node_importance).unsqueeze(0).unsqueeze(-1)
            if controls: gate = gate * controls.get('plasticity', 1.0)
            x = x * gate
            
        if self.config.use_plasticity:
            x_agg = torch.matmul(adj, x)
        else:
            x_agg = x
            
        if isinstance(self.node_mapper, ContinuumMemoryCell):
            x_flat = x_agg.view(-1, x_agg.size(-1))
            x_proc_flat = self.node_mapper(x_flat, controls)
            x_proc = x_proc_flat.view(batch_size, self.num_nodes, -1)
            new_state_node = None  # Placeholder para compatibilidad
        else:
            x_proc = self.node_mapper(x_agg)
            new_state_node = None
            
        entropy, ortho = torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)
        pred_nodes = torch.zeros_like(x_proc)
        
        if self.use_mgf:
            # Concatenar características de todos los nodos
            cell_input = torch.cat([x[:, i] for i in range(self.num_nodes)], dim=-1)
            
            if isinstance(self.cell_mapper, ContinuumMemoryCell):
                c_flat = self.cell_mapper(cell_input, controls)
                c = c_flat
            else:
                c = self.cell_mapper(cell_input)
                
            if self.symbiotic:
                c_clean, entropy, ortho = self.symbiotic(c)
            else:
                c_clean = c
                
            # Expandir para todos los nodos
            pred_nodes = c_clean.unsqueeze(1).expand(-1, self.num_nodes, -1)
        
        # Corrección clave: CONCATENAR en lugar de SUMAR
        combined = torch.cat([x_proc, pred_nodes], dim=-1)
        out = self.final_mix(combined)
        
        # Retornar 5 valores para mantener compatibilidad con el modelo original
        return self.norm(out), entropy, ortho, new_state_node, None

class TopoBrainTabular(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = 16
        
        # Embedding inicial
        self.input_embed = nn.Linear(config.n_features, self.embed_dim * self.num_nodes)
        
        # Capas adaptativas
        self.layer1 = AdaptiveCombinatorialComplexLayer(
            self.embed_dim, self.embed_dim, self.num_nodes, config, 'midbrain'
        )
        self.layer2 = AdaptiveCombinatorialComplexLayer(
            self.embed_dim, self.embed_dim, self.num_nodes, config, 'thalamus'
        )
        
        # Orquestador
        self.orchestrator = PrefrontalOrchestrator(config) if config.use_orchestrator else None
        
        # Capas de salida
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, config.n_classes)
        self.proj_head = nn.Sequential(
            nn.Linear(self.embed_dim * self.num_nodes, 32), 
            nn.ReLU(), 
            nn.Linear(32, 16)
        ) if config.use_supcon else None
        
        # Inicializar memorias
        self._initialize_memories()
        
    def _initialize_memories(self):
        """Inicialización de memorias semánticas"""
        if hasattr(self.layer1, 'node_mapper') and isinstance(self.layer1.node_mapper, ContinuumMemoryCell):
            self.layer1.node_mapper.memory_initialized.data = torch.tensor(1)
        if hasattr(self.layer1, 'cell_mapper') and isinstance(self.layer1.cell_mapper, ContinuumMemoryCell):
            self.layer1.cell_mapper.memory_initialized.data = torch.tensor(1)
        if hasattr(self.layer2, 'node_mapper') and isinstance(self.layer2.node_mapper, ContinuumMemoryCell):
            self.layer2.node_mapper.memory_initialized.data = torch.tensor(1)
        if hasattr(self.layer2, 'cell_mapper') and isinstance(self.layer2.cell_mapper, ContinuumMemoryCell):
            self.layer2.cell_mapper.memory_initialized.data = torch.tensor(1)
    
    def forward(self, x, controls=None, prev_states=None):
        batch_size = x.size(0)
        
        # Embedding inicial
        x_embed = self.input_embed(x).view(batch_size, self.num_nodes, self.embed_dim)
        
        # Capa 1
        x1, ent1, orth1, state_l1_node, state_l1_cell = self.layer1(x_embed, controls)
        
        # Capa 2
        x2, ent2, orth2, state_l2_node, state_l2_cell = self.layer2(x1, controls)
        
        # Flatten para clasificación
        x_flat = x2.view(batch_size, -1)
        
        # Salida
        logits = self.readout(x_flat)
        proj = self.proj_head(x_flat) if self.proj_head and self.training else None
        
        # Estados para la próxima iteración
        new_states = {
            'layer1_node': state_l1_node,
            'layer1_cell': state_l1_cell,
            'layer2_node': state_l2_node,
            'layer2_cell': state_l2_cell
        } if any([state_l1_node, state_l1_cell, state_l2_node, state_l2_cell]) else None
        
        return logits, proj, ent1 + ent2, orth1 + orth2, new_states

# =============================================================================
# PGD REAL + UTILIDADES DE ANÁLISIS
# =============================================================================
def pgd_attack(model, x, y, eps, steps):
    model.eval()
    delta = torch.zeros_like(x).uniform_(-eps, eps).requires_grad_(True)
    
    for _ in range(steps):
        logits, _, _, _, _ = model(x + delta)  # Capturar 5 valores
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        with torch.no_grad():
            # Asegurar que delta.grad existe
            if delta.grad is not None:
                delta += (eps / steps) * delta.grad.sign()
                delta.clamp_(-eps, eps)
        
        # Limpiar gradientes para la próxima iteración
        if delta.grad is not None:
            delta.grad.zero_()
            
    return (x + delta).detach().clone()

def compute_topology_metrics(model, config):
    """Computar métricas de topología con manejo robusto de errores"""
    try:
        if not config.use_plasticity:
            return {'density': 1.0, 'sparsity': 0.0, 'L_score': 3.0}
            
        with torch.no_grad():
            adj = torch.sigmoid(model.layer1.adj_weights)
            W = adj.cpu().numpy()
            
            try:
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                S_norm = S / (S.sum() + 1e-10)
                spectral_entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
                rank_eff = int(np.sum(S > 0.01 * S[0]))
                log_rank = np.log(rank_eff + 1)
                L_score = 1.0 / (abs(spectral_entropy - log_rank) + 0.1)
            except Exception as e:
                print(f"Error en cálculo SVD: {e}")
                L_score = 1.0
                
            density = (W > 0.5).mean()
            sparsity = 1.0 - density
            
            return {
                'density': float(density), 
                'sparsity': float(sparsity), 
                'L_score': float(L_score),
                'memory_norm': 1.0  # Placeholder para métricas de memoria
            }
    except Exception as e:
        print(f"Error en compute_topology_metrics: {e}")
        return {'density': 0.5, 'sparsity': 0.5, 'L_score': 1.0, 'memory_norm': 1.0}

def prune_topology(model, config, controls=None):
    """Implementación simplificada de poda de topología"""
    if not config.use_adaptive_topology or not config.use_plasticity:
        return
        
    with torch.no_grad():
        # Obtener pesos actuales
        adj_w = torch.sigmoid(model.layer1.adj_weights)
        current_density = (adj_w > 0.5).float().mean().item()
        
        # Solo podar si la densidad es alta
        if current_density > 0.3:  # Umbral ajustado para CPU
            threshold = torch.quantile(adj_w, 0.7)  # Mantener 30% de conexiones más fuertes
            
            # Modulación por orquestador
            if controls is not None and 'plasticity' in controls:
                plasticity = controls['plasticity'].item() if torch.is_tensor(controls['plasticity']) else controls['plasticity']
                threshold = threshold * (1.0 - plasticity * 0.5)
                
            # Crear nueva máscara
            new_mask = adj_w > threshold
            model.layer1.adj_weights.data = torch.where(
                new_mask,
                model.layer1.adj_weights.data,
                torch.full_like(model.layer1.adj_weights.data, -10.0)
            )
            
            print(f"✂️  Topology pruned: Density {current_density:.2f} → {(new_mask.float().mean().item()):.2f}")

# =============================================================================
# ENTRENAMIENTO CON LOGGING COMPLETO
# =============================================================================
def train_and_evaluate(config: Config, run_name: str):
    seed_everything(config.seed)
    train_loader, test_loader = get_tabular_loaders(config)
    model = TopoBrainTabular(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_main)
    supcon = SupConLoss() if config.use_supcon else None
    epoch_logs = []
    
    results_dir = Path(f"results/{run_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config.epochs):
        model.train()
        epoch_metrics = {
            'epoch': epoch, 'train_loss': 0.0, 'train_acc': 0.0, 'supcon_loss': 0.0,
            'ortho_loss': 0.0, 'entropy': 0.0, 'grad_norm': 0.0, 'density': 1.0,
            'plasticity_gate': 1.0, 'defense_gate': 1.0, 'memory_gate': 1.0,
            'supcon_gate': 1.0, 'symbiotic_gate': 1.0, 'sparsity_gate': 1.0,
            'batch_var': 0.0, 'memory_norm': 1.0, 'L_score': 2.0
        }
        
        batch_count = 0
        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            
            # Calcular variación del batch
            batch_var = x.var().item()
            epoch_metrics['batch_var'] += batch_var
            
            # Ataque adversarial
            x_adv = pgd_attack(model, x, y, config.train_eps, config.pgd_steps_train)
            
            # Obtener controles del orquestador
            controls = None
            topo_metrics = compute_topology_metrics(model, config)
            if config.use_orchestrator:
                metrics_dict = {
                    'loss': 1.0, 'grad_norm': 1.0, 'sparsity': topo_metrics['sparsity'],
                    'entropy': topo_metrics.get('entropy', 0.5), 'batch_var': batch_var,
                    'epoch_progress': epoch / config.epochs, 'memory_norm': topo_metrics['memory_norm'],
                    'L_score': topo_metrics['L_score'], 'density': topo_metrics['density']
                }
                controls = model.orchestrator(metrics_dict)
                
                # Acumular métricas de control
                for k, v in controls.items():
                    if k + '_gate' in epoch_metrics:
                        epoch_metrics[k + '_gate'] += v.item() if torch.is_tensor(v) else v
            
            # Forward pass
            logits, proj, entropy, ortho, _ = model(x_adv, controls)
            
            # Calcular pérdida
            loss = F.cross_entropy(logits, y)
            if config.use_supcon and supcon and proj is not None:
                s_loss = supcon(proj, y)
                loss += config.lambda_supcon * s_loss
                epoch_metrics['supcon_loss'] += s_loss.item()
                
            loss -= config.lambda_entropy * entropy
            loss += config.lambda_ortho * ortho
            
            # Backward pass
            loss.backward()
            
            # Calcular norma de gradientes
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            epoch_metrics['grad_norm'] += grad_norm
            
            # Recortar gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            
            # Optimizar
            optimizer.step()
            
            # Acumular métricas
            epoch_metrics['train_loss'] += loss.item()
            epoch_metrics['train_acc'] += logits.argmax(1).eq(y).float().mean().item()
            epoch_metrics['entropy'] += entropy.item()
            epoch_metrics['ortho_loss'] += ortho.item()
            
            batch_count += 1
            
        # Promediar métricas por época
        for k in epoch_metrics:
            if k != 'epoch':
                epoch_metrics[k] /= max(batch_count, 1)
        
        # Actualizar métricas de topología
        topo_metrics = compute_topology_metrics(model, config)
        epoch_metrics.update(topo_metrics)
        
        # Poda de topología (cada 2 épocas)
        if config.use_adaptive_topology and epoch % 2 == 0:
            prune_topology(model, config, controls)
            
        # Resetear contexto del orquestador
        if config.use_orchestrator:
            model.orchestrator.reset_context()
            
        epoch_logs.append(epoch_metrics)
        print(f"Ep {epoch}: Loss {epoch_metrics['train_loss']:.4f} | "
              f"Acc {epoch_metrics['train_acc']*100:.2f}% | "
              f"Density {epoch_metrics['density']:.2f}")

    # Evaluación final
    def evaluate_adv(loader, eps, steps):
        model.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(config.device), y.to(config.device)
            x_pgd = pgd_attack(model, x, y, eps, steps)
            with torch.no_grad():
                logits, _, _, _, _ = model(x_pgd)  # Capturar 5 valores
                correct += logits.argmax(1).eq(y).sum().item()
                total += y.size(0)
        return (100 * correct / total) if total > 0 else 0.0
    
    # Evaluación limpia y robusta
    clean = evaluate_adv(test_loader, 0.0, 1)
    pgd = evaluate_adv(test_loader, config.test_eps, config.pgd_steps_test)
    
    # Guardar resultados de la época
    with open(results_dir / f"epoch_metrics_{run_name}.json", 'w') as f:
        json.dump(epoch_logs, f, indent=2)
    
    return clean, pgd, epoch_logs

# =============================================================================
# MATRIZ DE ABLATION (Científica y Rigurosa)
# =============================================================================
ABLATION_MATRIX = [
    ("Baseline_No_Mechanisms", {}),
    ("Only_Plasticity", {"use_plasticity": True}),
    ("Only_NestedCells", {"use_nested_cells": True}),
    ("Only_Symbiotic", {"use_symbiotic": True}),
    ("Only_SupCon", {"use_supcon": True}),
    ("Plasticity_NestedCells", {"use_plasticity": True, "use_nested_cells": True}),
    ("Symbiotic_SupCon", {"use_symbiotic": True, "use_supcon": True}),
    ("Plasticity_Symbiotic", {"use_plasticity": True, "use_symbiotic": True}),
    ("NestedCells_SupCon", {"use_nested_cells": True, "use_supcon": True}),
    ("Full_Topobrain_Minimal", {
        "use_plasticity": True, "use_nested_cells": True,
        "use_symbiotic": True, "use_supcon": True
    }),
    ("Full_Topobrain_With_Orchestrator", {
        "use_plasticity": True, "use_nested_cells": True,
        "use_symbiotic": True, "use_supcon": True,
        "use_orchestrator": True
    })
]

def run_ablation():
    all_results = []
    results_dir = Path("ablation_results")
    results_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🔬 TOPOBRAIN ABLATION STUDY - RIGUROSO Y CIENTÍFICO")
    print("="*80)
    print(f"Total experiments: {len(ABLATION_MATRIX)}")
    print(f"Dataset: Synthetic tabular (NOIR-aware)")
    print(f"Parameters: ~8k-15k per model")
    print(f"Device: CPU")
    print("="*80)
    
    for experiment_idx, (name, overrides) in enumerate(ABLATION_MATRIX):
        print(f"\n▶ [{experiment_idx+1}/{len(ABLATION_MATRIX)}] Running: {name}")
        
        # Configurar experimento
        base = Config()
        cfg_dict = base.to_dict()
        cfg_dict.update(overrides)
        config = Config(**cfg_dict)
        
        # Ejecutar experimento
        try:
            clean, pgd, logs = train_and_evaluate(config, name)
            result = {
                "name": name,
                "config": config.to_dict(),
                "final_metrics": {"clean_acc": clean, "pgd_acc": pgd},
                "epochs": logs
            }
            all_results.append(result)
            
            # Mostrar resultados
            print(f"   ✅ Completed: Clean: {clean:.2f}% | PGD: {pgd:.2f}%")
            
            # Guardar resultados parciales
            with open(results_dir / f"partial_results.json", 'w') as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "name": name,
                "config": cfg_dict,
                "error": str(e)
            })
        
        # Liberar memoria
        gc.collect()
    
    # Guardar resultados finales
    with open(results_dir / "final_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generar resumen científico
    print("\n" + "="*80)
    print("📊 RESUMEN CIENTÍFICO DE ABLATION")
    print("="*80)
    
    valid_results = [r for r in all_results if 'error' not in r]
    if valid_results:
        # Ordenar por robustez (PGD accuracy)
        ranked = sorted(valid_results, key=lambda r: r["final_metrics"]["pgd_acc"], reverse=True)
        
        print("\n🏆 TOP 5 por Robustez Adversarial (PGD accuracy):")
        print("-"*50)
        for i, r in enumerate(ranked[:5], 1):
            clean = r["final_metrics"]["clean_acc"]
            pgd = r["final_metrics"]["pgd_acc"]
            delta = clean - pgd
            print(f"{i}. {r['name']:<40} | Clean: {clean:5.2f}% | PGD: {pgd:5.2f}% | Δ: {delta:5.2f}%")
        
        # Analizar contribución de mecanismos
        print("\n🔬 ANÁLISIS DE MECANISMOS CLAVE:")
        print("-"*50)
        
        mechanisms = ['plasticity', 'nested_cells', 'symbiotic', 'supcon', 'orchestrator']
        for mech in mechanisms:
            with_mech = [r for r in valid_results if mech in r['name'].lower()]
            without_mech = [r for r in valid_results if mech not in r['name'].lower() and 'baseline' in r['name'].lower()]
            
            if with_mech and without_mech:
                avg_pgd_with = sum(r['final_metrics']['pgd_acc'] for r in with_mech) / len(with_mech)
                avg_pgd_without = sum(r['final_metrics']['pgd_acc'] for r in without_mech) / len(without_mech)
                improvement = avg_pgd_with - avg_pgd_without
                print(f"• {mech.replace('_', ' ').title():<15}: PGD improvement = {improvement:+5.2f}%")
    
    print("\n✅ Ablation completado. Resultados guardados en 'ablation_results/'")
    return all_results

if __name__ == "__main__":
    seed_everything(42)
    print("🧠 TopoBrain Tabular – Ablation Científico Riguroso (NOIR, CPU)")
    results = run_ablation()
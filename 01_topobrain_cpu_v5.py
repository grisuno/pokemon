# 01_topobrain_cpu_scientific_ablation.py
# Ablation Científico Riguroso con Control de Variables y Estabilidad Numérica (Versión Optimizada para CPU)

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
import time
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA (Parámetros reducidos pero estables para CPU)
# =============================================================================
@dataclass
class Config:
    device: str = "cpu"
    seed: int = 42
    n_samples: int = 500          # Dataset reducido para CPU
    n_features: int = 20
    n_classes: int = 3
    n_informative: int = 15
    n_redundant: int = 3
    flip_y: float = 0.05
    batch_size: int = 16          # Tamaño de batch reducido
    grid_size: int = 2            # Grid 2x2 = 4 nodos totales
    embed_dim: int = 8            # Dimensión de embedding reducida
    hidden_dim: int = 8           # Dimensión oculta reducida
    use_plasticity: bool = False
    use_nested_cells: bool = False
    use_mgf: bool = False
    use_supcon: bool = False
    use_symbiotic: bool = False
    epochs: int = 5               # Épocas reducidas para CPU
    lr: float = 0.005
    train_eps: float = 0.3
    test_eps: float = 0.3
    pgd_steps: int = 3            # Pasos de PGD reducidos
    lambda_supcon: float = 0.3
    lambda_ortho: float = 0.05
    lambda_entropy: float = 0.01
    stability_epsilon: float = 1e-6
    clip_value: float = 3.0       # Valor de clip reducido
    min_density_threshold: float = 0.15  # Umbral de densidad mínima para evitar colapso
    grad_accum_steps: int = 1

    def to_dict(self):
        return asdict(self)

    def get_topology_config(self):
        """Configuración estable para topología adaptable"""
        return {
            'target_sparsity': 0.85,
            'prune_ratio': 0.1,
            'min_connections_per_node': 1
        }

# =============================================================================
# UTILIDADES DE CONTROL CIENTÍFICO
# =============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def get_tabular_loaders(config: Config):
    """Dataset tabular controlado con características NOIR simuladas"""
    # Generar características con diferentes escalas (nominal, ordinal, intervalo, ratio)
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
    
    # Normalizar a [0,1] con manejo robusto de outliers
    X_min = np.percentile(X, 1, axis=0)
    X_max = np.percentile(X, 99, axis=0)
    X = np.clip(X, X_min, X_max)
    X = (X - X_min) / (X_max - X_min + config.stability_epsilon)
    
    # Dividir dataset
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Convertir a tensores
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                            torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                           torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, test_loader

# =============================================================================
# COMPONENTES ESTABLES Y NUMÉRICAMENTE ROBUSTOS (Versión ligera para CPU)
# =============================================================================
class StableSupConLoss(nn.Module):
    """Versión estable de SupConLoss con manejo de bordes robusto"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.epsilon = 1e-8
        
    def forward(self, features, labels):
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device)
            
        features = F.normalize(features, dim=1)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
        
        # Evitar división por cero
        logits = torch.div(torch.matmul(features, features.T), 
                          self.temperature + self.epsilon)
        
        # Estabilidad numérica
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits) * (1 - torch.eye(logits.size(0), device=features.device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.epsilon)
        
        # Manejo de máscaras vacías
        mask_sum = mask.sum(1).clamp(min=self.epsilon)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        return -mean_log_prob_pos.mean()

class StableContinuumMemoryCell(nn.Module):
    """Versión estable y ligera de ContinuumMemoryCell con clamping y normalización"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Pesos lentos con inicialización estable
        self.W_slow = nn.Linear(input_dim, hidden_dim, bias=False)
        self.V_slow = nn.Linear(input_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.V_slow.weight, gain=0.1)  # Ganancia reducida para estabilidad
        
        # Gates con inicialización suave
        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 1),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 1),
            nn.Sigmoid()
        )
        self.semantic_mix = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # CORRECCIÓN CRÍTICA: Memoria semántica debe ser [input_dim, input_dim] para reconstruir entrada
        self.semantic_memory = nn.Parameter(torch.zeros(input_dim, input_dim))
        nn.init.normal_(self.semantic_memory, std=0.01)
        
        # Parámetros de estabilidad
        self.register_buffer('clip_value', torch.tensor(3.0))
        self.register_buffer('min_norm', torch.tensor(1e-6))
        
    def forward(self, x, controls=None):
        # Proyección lenta con clamping
        v = self.V_slow(x)
        v = torch.clamp(v, -self.clip_value, self.clip_value)
        
        # CORRECCIÓN: Predicción debe usar memoria [input_dim, input_dim]
        y_pred = F.linear(x, self.semantic_memory)
        y_pred = torch.clamp(y_pred, -self.clip_value, self.clip_value)
        
        # Proyectar y_pred a hidden_dim para calcular error
        v_pred = self.V_slow(y_pred)
        v_pred = torch.clamp(v_pred, -self.clip_value, self.clip_value)
        
        # Error en espacio latente
        error = v - v_pred
        error = torch.clamp(error, -self.clip_value, self.clip_value)
        
        # Entrada para gates
        gate_input = torch.cat([v.detach(), x.detach()], dim=-1)
        
        # Gates con valores por defecto para controles
        plasticity = 1.0
        if controls is not None and 'memory' in controls:
            plasticity = controls['memory'].item() if torch.is_tensor(controls['memory']) else controls['memory']
        
        forget = self.forget_gate(gate_input) * plasticity
        update = self.update_gate(gate_input) * plasticity
        
        # Actualización de memoria con estabilidad numérica
        # Delta en espacio de entrada: [batch, input_dim, input_dim]
        delta = torch.bmm(x.unsqueeze(-1), x.unsqueeze(1))
        delta = torch.clamp(delta, -self.clip_value, self.clip_value)
        
        with torch.no_grad():
            # Factor de olvido adaptativo para estabilidad
            forget_factor = 0.95 + 0.05 * plasticity
            self.semantic_memory.data = (
                forget_factor * self.semantic_memory.data +
                0.01 * delta.mean(dim=0)
            )
            # Normalización para evitar explosión
            mem_norm = self.semantic_memory.data.norm().clamp(min=self.min_norm)
            self.semantic_memory.data = self.semantic_memory.data / mem_norm * 0.5
        
        # Mezcla final en espacio latente
        mix = self.semantic_mix(v.detach())
        output = mix * v + (1 - mix) * v_pred
        
        return torch.clamp(output, -self.clip_value, self.clip_value)

class StableSymbioticBasisRefinement(nn.Module):
    """Refinamiento simbiótico estable con regularización explícita (versión ligera)"""
    def __init__(self, dim, num_atoms=4):  # Reducido de 8 a 4 átomos
        super().__init__()
        self.num_atoms = num_atoms
        self.basis = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis, gain=0.5)  # Ganancia reducida
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.epsilon = 1e-8
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(self.basis)
        attn = torch.matmul(Q, K.T) / (x.size(-1) ** 0.5 + self.epsilon)
        weights = F.softmax(attn, dim=-1)
        
        # Estabilidad en la reconstrucción
        x_clean = torch.matmul(weights, self.basis)
        x_clean = torch.clamp(x_clean, -3.0, 3.0)  # Clip reducido
        
        # Cálculo estable de entropía
        weights_safe = weights + self.epsilon
        entropy = -(weights_safe * torch.log(weights_safe)).sum(-1).mean()
        
        # Regularización de ortogonalidad estable
        gram = torch.mm(self.basis, self.basis.T)
        identity = torch.eye(gram.size(0), device=gram.device)
        ortho = torch.norm(gram - identity, p='fro') ** 2
        ortho = torch.clamp(ortho, 0.0, 5.0)  # Límite reducido
        
        return x_clean, entropy, ortho

# =============================================================================
# TOPOLOGÍA ESTABLE CON MECANISMOS DE SUPERVIVENCIA (Versión ligera)
# =============================================================================
class StableTopologyManager:
    """Gestor de topología con protocolos de supervivencia (versión optimizada para CPU)"""
    def __init__(self, num_nodes, config: Config):
        self.num_nodes = num_nodes
        self.config = config
        self.min_density = config.min_density_threshold
        
        # Inicializar matriz de adyacencia estable
        self.adj_weights = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.normal_(self.adj_weights, std=0.1)
        
        # Máscara de conexiones posibles (grid 2x2)
        self.adj_mask = torch.zeros(num_nodes, num_nodes)
        grid_size = config.grid_size
        for i in range(num_nodes):
            r, c = i // grid_size, i % grid_size
            if r > 0: self.adj_mask[i, i - grid_size] = 1
            if r < grid_size - 1: self.adj_mask[i, i + grid_size] = 1
            if c > 0: self.adj_mask[i, i - 1] = 1
            if c < grid_size - 1: self.adj_mask[i, i + 1] = 1
            
    def get_adjacency(self, plasticity=1.0):
        """Obtener matriz de adyacencia con estabilidad garantizada"""
        # Aplicar máscara y plasticidad
        adj = torch.sigmoid(self.adj_weights * plasticity) * self.adj_mask
        
        # Normalizar por grado para estabilidad
        deg = adj.sum(1, keepdim=True).clamp(min=1e-6)
        adj = adj / deg
        
        # Garantizar mínimo de conexiones
        if plasticity > 0.1:  # Solo en modo de aprendizaje
            min_connections = self.config.get_topology_config()['min_connections_per_node']
            for i in range(self.num_nodes):
                connections = adj[i] > 0.1
                if connections.sum() < min_connections:
                    # Conectar a vecinos más cercanos
                    r, c = i // self.config.grid_size, i % self.config.grid_size
                    neighbors = []
                    if r > 0: neighbors.append(i - self.config.grid_size)
                    if r < self.config.grid_size - 1: neighbors.append(i + self.config.grid_size)
                    if c > 0: neighbors.append(i - 1)
                    if c < self.config.grid_size - 1: neighbors.append(i + 1)
                    
                    if neighbors:
                        for n in neighbors[:min_connections]:
                            adj[i, n] = max(adj[i, n], 0.5)
        
        return adj
    
    def prune_topology(self, current_density, epoch):
        """Poda controlada con protocolo de emergencia"""
        if current_density < self.min_density:
            # Protocolo de emergencia: no podar, en su lugar estabilizar
            return False
        
        # Poda estándar con umbrales controlados
        target_sparsity = self.config.get_topology_config()['target_sparsity']
        prune_ratio = self.config.get_topology_config()['prune_ratio']
        
        # Calcular umbral dinámico
        threshold = torch.quantile(torch.sigmoid(self.adj_weights), 
                                 1.0 - (1.0 - target_sparsity) * (1.0 - prune_ratio))
        
        with torch.no_grad():
            prune_mask = torch.sigmoid(self.adj_weights) < threshold
            self.adj_weights.data[prune_mask] = -5.0  # Valores muy negativos se mantienen podados
        
        return True
    
    def get_density(self):
        """Calcular densidad actual de manera estable"""
        adj = torch.sigmoid(self.adj_weights) * self.adj_mask
        return (adj > 0.5).float().mean().item()

# =============================================================================
# MODELO PRINCIPAL ESTABLE (Versión optimizada para CPU con ~8k-12k parámetros)
# =============================================================================
class StableTopoBrain(nn.Module):
    """Implementación estable y ligera de TopoBrain para ablation científico en CPU"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2  # 4 nodos para grid 2x2
        self.embed_dim = config.embed_dim       # 8 dimensiones
        
        # Embedding inicial: 20 features -> 32 dimensiones (4 nodos * 8 embed)
        self.input_embed = nn.Linear(config.n_features, self.embed_dim * self.num_nodes)
        
        # Topología estable
        self.topology = StableTopologyManager(self.num_nodes, config)
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, config.n_classes)
        
        # Capas de procesamiento
        if config.use_nested_cells:
            # CORRECCIÓN: Usar embed_dim como hidden_dim para consistencia y evitar shape mismatch
            self.node_processor = StableContinuumMemoryCell(self.embed_dim, self.embed_dim)
        else:
            self.node_processor = nn.Sequential(
                nn.Linear(self.embed_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, self.embed_dim)
            )
        
        # Capa de celdas (MGF) - Opcional
        self.cell_processor = None
        if config.use_mgf:
            mgf_input_dim = self.embed_dim * self.num_nodes  # CORRECCIÓN: dimensión total = 32
            if config.use_nested_cells:
                # CORRECCIÓN: Usar embed_dim como hidden_dim para consistencia con el procesamiento de nodos
                self.cell_processor = StableContinuumMemoryCell(mgf_input_dim, self.embed_dim)
            else:
                self.cell_processor = nn.Sequential(
                    nn.Linear(mgf_input_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, self.embed_dim)
                )
        
        # Refinamiento simbiótico - Opcional
        self.symbiotic = StableSymbioticBasisRefinement(self.embed_dim) if config.use_symbiotic else None
        
        # Cabeza SupCon - Opcional
        self.supcon_head = None
        if config.use_supcon:
            self.supcon_head = nn.Sequential(
                nn.Linear(self.embed_dim * self.num_nodes, 16),  # Reducido de 32 a 16
                nn.ReLU(),
                nn.Linear(16, 8)  # Reducido de 16 a 8
            )
        
        # Inicialización estable
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización estable de pesos"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, controls=None):
        batch_size = x.size(0)
        
        # CORRECCIÓN CRÍTICA: Usa tamaño real del batch en el view
        x_embed = self.input_embed(x).view(batch_size, self.num_nodes, self.embed_dim)
        
        # Obtener topología actual
        plasticity = 1.0
        if controls is not None and 'plasticity' in controls:
            plasticity = controls['plasticity'].item() if torch.is_tensor(controls['plasticity']) else controls['plasticity']
        
        adj = self.topology.get_adjacency(plasticity)
        
        # Propagación en nodos
        if isinstance(self.node_processor, StableContinuumMemoryCell):
            x_flat = x_embed.view(-1, self.embed_dim)
            x_proc_flat = self.node_processor(x_flat, controls)
            x_proc = x_proc_flat.view(batch_size, self.num_nodes, self.embed_dim)
        else:
            x_proc = self.node_processor(x_embed)
        
        # Procesamiento de celdas (MGF)
        entropy = torch.tensor(0.0, device=x.device)  # Asegurar mismo dispositivo
        ortho = torch.tensor(0.0, device=x.device)    # Asegurar mismo dispositivo
        cell_output = torch.zeros_like(x_proc)
        
        if self.config.use_mgf and self.cell_processor is not None:
            cell_input = x_embed.view(batch_size, -1)  # Tamaño: [batch, 32]
            if isinstance(self.cell_processor, StableContinuumMemoryCell):
                cell_output_flat = self.cell_processor(cell_input, controls)
                # CORRECCIÓN: cell_output_flat tiene tamaño [batch, embed_dim]
                # Reshape a [batch, 1, embed_dim] y expandir a [batch, num_nodes, embed_dim]
                cell_output = cell_output_flat.view(batch_size, 1, self.embed_dim).expand(-1, self.num_nodes, -1)
            else:
                cell_temp = self.cell_processor(cell_input)
                cell_output = cell_temp.view(batch_size, 1, self.embed_dim).expand(-1, self.num_nodes, -1)
        
        # Refinamiento simbiótico
        if self.symbiotic is not None:
            # Aplicar refinamiento a cada nodo
            x_proc_list = []
            entropy_total = 0.0
            ortho_total = 0.0
            for i in range(self.num_nodes):
                node_features = x_proc[:, i, :]
                refined, node_entropy, node_ortho = self.symbiotic(node_features)
                x_proc_list.append(refined)
                entropy_total += node_entropy
                ortho_total += node_ortho
            
            x_proc = torch.stack(x_proc_list, dim=1)
            entropy = entropy_total / self.num_nodes
            ortho = ortho_total / self.num_nodes
        
        # Combinar salidas
        combined = x_proc + cell_output
        
        # Salida final
        x_flat = combined.view(batch_size, -1)
        logits = self.readout(x_flat)
        proj = self.supcon_head(x_flat) if self.supcon_head is not None else None
        
        return logits, proj, entropy, ortho

# =============================================================================
# PGD ESTABLE Y VALIDADO (Versión ligera para CPU)
# =============================================================================
def stable_pgd_attack(model, x, y, eps, steps, controls=None):
    """Ataque PGD estable con manejo de gradientes robusto"""
    model.eval()
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta.requires_grad = True
    
    for _ in range(steps):
        # Asegurar que delta mantenga requires_grad=True en cada iteración
        if not delta.requires_grad:
            delta.requires_grad = True
            
        with torch.enable_grad():  # Asegurar cálculo de gradientes dentro del contexto
            logits, _, _, _ = model(x + delta, controls)
            loss = F.cross_entropy(logits, y)
            
            # Limpiar gradientes previos de delta
            if delta.grad is not None:
                delta.grad.zero_()
            
            # Calcular gradientes
            loss.backward()
        
        # Verificar que delta.grad exista después de backward
        if delta.grad is None:
            raise RuntimeError("delta.grad is None after backward pass. Check gradient flow.")
        
        with torch.no_grad():
            # Normalizar gradientes para estabilidad (versión corregida para datos tabulares)
            grad = delta.grad
            # Calcular norma adecuadamente para datos tabulares (2D)
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1)
            grad_norm = grad_norm.clamp(min=1e-8)
            grad = grad / grad_norm
            
            # Actualizar delta con clamping
            delta.data = (delta.data + (eps / steps) * grad.sign()).clamp(-eps, eps)
    
    model.train()
    return (x + delta).detach()

# =============================================================================
# ENTRENAMIENTO CON MONITOREO CIENTÍFICO
# =============================================================================
def train_epoch(model, loader, optimizer, config, epoch, controls=None):
    """Entrenamiento por época con monitoreo detallado"""
    model.train()
    metrics = defaultdict(float)
    batch_count = 0
    
    criterion = nn.CrossEntropyLoss()
    supcon_loss = StableSupConLoss() if config.use_supcon else None
    
    for x, y in loader:
        x, y = x.to(config.device), y.to(config.device)
        
        # Ataque adversarial estable
        x_adv = stable_pgd_attack(model, x, y, config.train_eps, config.pgd_steps, controls)
        
        # Forward pass
        logits, proj, entropy, ortho = model(x_adv, controls)
        
        # Calcular pérdidas
        loss = criterion(logits, y)
        metrics['ce_loss'] += loss.item()
        
        # SupCon loss si está habilitado
        if config.use_supcon and supcon_loss is not None and proj is not None:
            s_loss = supcon_loss(proj, y)
            loss += config.lambda_supcon * s_loss
            metrics['supcon_loss'] += s_loss.item()
        
        # Regularizaciones
        loss -= config.lambda_entropy * entropy
        loss += config.lambda_ortho * ortho
        
        # Corrección: asegurar que entropy y ortho sean escalares y estén en CPU
        metrics['entropy'] += entropy.item() if torch.is_tensor(entropy) else float(entropy)
        metrics['ortho'] += ortho.item() if torch.is_tensor(ortho) else float(ortho)
        metrics['total_loss'] += loss.item()
        
        # Precisión
        pred = logits.argmax(dim=1)
        correct = pred.eq(y).sum().item()
        metrics['accuracy'] += correct / y.size(0)
        
        # Backward pass con estabilidad
        optimizer.zero_grad()
        loss.backward()
        
        # Clip de gradientes para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
        optimizer.step()
        
        batch_count += 1
    
    # Promediar métricas
    for k in metrics:
        metrics[k] /= batch_count
    
    # Métricas de topología
    if hasattr(model, 'topology'):
        metrics['density'] = model.topology.get_density()
    
    return dict(metrics)

# =============================================================================
# EVALUACIÓN RIGUROSA
# =============================================================================
def evaluate_model(model, loader, config, adversarial=False, controls=None):
    """Evaluación rigurosa con o sin ataques adversariales"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(config.device), y.to(config.device)
            
            if adversarial:
                x = stable_pgd_attack(model, x, y, config.test_eps, config.pgd_steps, controls)
            
            logits, _, _, _ = model(x, controls)
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0

# =============================================================================
# MATRIZ DE ABLATION CIENTÍFICO - 3 NIVELES COMPLETOS (36 experimentos)
# =============================================================================

# NIVEL 1: COMPONENTES AISLADOS (Control Ceteris Paribus) - 6 experimentos
LEVEL1_ISOLATED = [
    ("Baseline", {}),
    ("Plasticity_Only", {"use_plasticity": True}),
    ("NestedCells_Only", {"use_nested_cells": True}),
    ("MGF_Only", {"use_mgf": True}),
    ("SupCon_Only", {"use_supcon": True}),
    ("Symbiotic_Only", {"use_symbiotic": True}),
]

# NIVEL 2: PARES SINÉRGICOS (Todas las combinaciones 2 a 2) - 10 experimentos
# C(5,2) = 10 combinaciones posibles de 5 mecanismos
LEVEL2_PAIRS = [
    # Plasticity con otros
    ("Pair_Plasticity_Nested", {"use_plasticity": True, "use_nested_cells": True}),
    ("Pair_Plasticity_MGF", {"use_plasticity": True, "use_mgf": True}),
    ("Pair_Plasticity_SupCon", {"use_plasticity": True, "use_supcon": True}),
    ("Pair_Plasticity_Symbiotic", {"use_plasticity": True, "use_symbiotic": True}),
    
    # NestedCells con otros (excepto Plasticity ya cubierto)
    ("Pair_Nested_MGF", {"use_nested_cells": True, "use_mgf": True}),
    ("Pair_Nested_SupCon", {"use_nested_cells": True, "use_supcon": True}),
    ("Pair_Nested_Symbiotic", {"use_nested_cells": True, "use_symbiotic": True}),
    
    # MGF con otros (excepto ya cubiertos)
    ("Pair_MGF_SupCon", {"use_mgf": True, "use_supcon": True}),
    ("Pair_MGF_Symbiotic", {"use_mgf": True, "use_symbiotic": True}),
    
    # SupCon con Symbiotic (última combinación)
    ("Pair_SupCon_Symbiotic", {"use_supcon": True, "use_symbiotic": True}),
]

# NIVEL 2B: TRÍADAS CRÍTICAS (Combinaciones de 3) - 10 experimentos selectos
# Exploramos las tríadas más prometedoras basadas en hipótesis
LEVEL2_TRIADS = [
    # Core cognitivo (Plasticity + Nested + X)
    ("Triad_Plasticity_Nested_MGF", {"use_plasticity": True, "use_nested_cells": True, "use_mgf": True}),
    ("Triad_Plasticity_Nested_SupCon", {"use_plasticity": True, "use_nested_cells": True, "use_supcon": True}),
    ("Triad_Plasticity_Nested_Symbiotic", {"use_plasticity": True, "use_nested_cells": True, "use_symbiotic": True}),
    
    # Core representacional (Nested + SupCon + X)
    ("Triad_Nested_SupCon_MGF", {"use_nested_cells": True, "use_supcon": True, "use_mgf": True}),
    ("Triad_Nested_SupCon_Symbiotic", {"use_nested_cells": True, "use_supcon": True, "use_symbiotic": True}),
    
    # Core topológico (Plasticity + MGF + X)
    ("Triad_Plasticity_MGF_SupCon", {"use_plasticity": True, "use_mgf": True, "use_supcon": True}),
    ("Triad_Plasticity_MGF_Symbiotic", {"use_plasticity": True, "use_mgf": True, "use_symbiotic": True}),
    
    # Regularización máxima (SupCon + Symbiotic + X)
    ("Triad_SupCon_Symbiotic_Nested", {"use_supcon": True, "use_symbiotic": True, "use_nested_cells": True}),
    ("Triad_SupCon_Symbiotic_MGF", {"use_supcon": True, "use_symbiotic": True, "use_mgf": True}),
    ("Triad_SupCon_Symbiotic_Plasticity", {"use_supcon": True, "use_symbiotic": True, "use_plasticity": True}),
]

# NIVEL 3: ABLACIÓN INVERSA (Quitar 1 componente del modelo completo) - 5 experimentos
# Detectar componentes ESENCIALES vs REDUNDANTES
LEVEL3_INVERSE = [
    ("Full_Without_Plasticity", {
        "use_nested_cells": True, "use_mgf": True, "use_supcon": True, "use_symbiotic": True
    }),
    ("Full_Without_Nested", {
        "use_plasticity": True, "use_mgf": True, "use_supcon": True, "use_symbiotic": True
    }),
    ("Full_Without_MGF", {
        "use_plasticity": True, "use_nested_cells": True, "use_supcon": True, "use_symbiotic": True
    }),
    ("Full_Without_SupCon", {
        "use_plasticity": True, "use_nested_cells": True, "use_mgf": True, "use_symbiotic": True
    }),
    ("Full_Without_Symbiotic", {
        "use_plasticity": True, "use_nested_cells": True, "use_mgf": True, "use_supcon": True
    }),
]

# NIVEL 3B: MODELO COMPLETO (Referencia máxima) - 1 experimento
LEVEL3_FULL = [
    ("Full_Topobrain_All5", {
        "use_plasticity": True, "use_nested_cells": True, "use_mgf": True,
        "use_supcon": True, "use_symbiotic": True
    })
]

# MATRIZ COMPLETA DE ABLATION (Total: 32 experimentos)
ABLATION_MATRIX = (
    LEVEL1_ISOLATED +      # 6 experimentos
    LEVEL2_PAIRS +         # 10 experimentos
    LEVEL2_TRIADS +        # 10 experimentos
    LEVEL3_INVERSE +       # 5 experimentos
    LEVEL3_FULL            # 1 experimento
)

print(f"📊 Total experimentos diseñados: {len(ABLATION_MATRIX)}")
print(f"   Nivel 1 (Aislados): {len(LEVEL1_ISOLATED)}")
print(f"   Nivel 2 (Pares): {len(LEVEL2_PAIRS)}")
print(f"   Nivel 2B (Tríadas): {len(LEVEL2_TRIADS)}")
print(f"   Nivel 3 (Inversa): {len(LEVEL3_INVERSE)}")
print(f"   Nivel 3B (Full): {len(LEVEL3_FULL)}")

# =============================================================================
# EJECUCIÓN CIENTÍFICA DEL ABLATION
# =============================================================================
def run_scientific_ablation():
    """Ejecución científica del ablation con control de variables"""
    seed_everything(42)
    results_dir = Path("scientific_ablation_results")
    results_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🧠 TopoBrain - Ablation Científico Riguroso (Estable para CPU)")
    print("="*80)
    print(f"Total experimentos: {len(ABLATION_MATRIX)}")
    print(f"Dataset: Tabular (20 features, 3 clases)")
    print(f"Parámetros: ~8k-12k por modelo")
    print(f"Dispositivo: CPU")
    print("="*80)
    
    all_results = []
    
    for exp_idx, (name, overrides) in enumerate(ABLATION_MATRIX):
        print(f"\n▶ [{exp_idx+1}/{len(ABLATION_MATRIX)}] Ejecutando: {name}")
        
        # Configurar experimento
        base_cfg = Config()
        cfg_dict = base_cfg.to_dict()
        cfg_dict.update(overrides)
        config = Config(**cfg_dict)
        
        # Cargar datos
        train_loader, test_loader = get_tabular_loaders(config)
        
        # Inicializar modelo
        model = StableTopoBrain(config)
        model.to(config.device)
        
        # Optimizador estable
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        
        # Entrenamiento por épocas
        epoch_history = []
        best_clean_acc = 0.0
        
        for epoch in range(config.epochs):
            start_time = time.time()
            
            # Controles dinámicos (simulación de orquestador estable)
            controls = {
                'plasticity': torch.tensor(0.8 if config.use_plasticity else 0.0),
                'memory': torch.tensor(0.7 if config.use_nested_cells else 0.0),
                'defense': torch.tensor(0.6)  # Siempre activo para PGD
            }
            
            # Entrenar época
            metrics = train_epoch(model, train_loader, optimizer, config, epoch, controls)
            epoch_time = time.time() - start_time
            
            # Evaluar
            clean_acc = evaluate_model(model, test_loader, config, adversarial=False, controls=controls)
            pgd_acc = evaluate_model(model, test_loader, config, adversarial=True, controls=controls)
            
            # Actualizar mejor accuracy
            if clean_acc > best_clean_acc:
                best_clean_acc = clean_acc
            
            # Registro de época
            epoch_data = {
                'epoch': epoch + 1,
                'time_seconds': epoch_time,
                'metrics': metrics,
                'clean_acc': clean_acc,
                'pgd_acc': pgd_acc,
                'delta': clean_acc - pgd_acc
            }
            epoch_history.append(epoch_data)
            
            density = metrics.get('density', 1.0)
            print(f"  Ep {epoch+1:2d}/{config.epochs} | "
                  f"Loss: {metrics['total_loss']:.4f} | "
                  f"Clean: {clean_acc:5.2f}% | "
                  f"PGD: {pgd_acc:5.2f}% | "
                  f"Δ: {clean_acc-pgd_acc:5.2f}% | "
                  f"Density: {density:.2f}")
        
        # Resultados finales del experimento
        final_clean = epoch_history[-1]['clean_acc']
        final_pgd = epoch_history[-1]['pgd_acc']
        
        experiment_result = {
            "name": name,
            "config": config.to_dict(),
            "epochs": epoch_history,
            "summary": {
                "best_clean_acc": best_clean_acc,
                "final_clean_acc": final_clean,
                "final_pgd_acc": final_pgd,
                "robustness_gap": final_clean - final_pgd,
                "stability_score": 1.0 - (abs(final_clean - best_clean_acc) / max(best_clean_acc, 1e-6))
            }
        }
        all_results.append(experiment_result)
        
        print(f"  ✅ Completado: Clean: {final_clean:.2f}% | PGD: {final_pgd:.2f}% | Gap: {final_clean-final_pgd:.2f}%")
        
        # Guardar resultados intermedios
        with open(results_dir / f"{name}_results.json", 'w') as f:
            json.dump(experiment_result, f, indent=2)
    
    # Guardar todos los resultados
    with open(results_dir / "complete_ablation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Análisis científico
    print("\n" + "="*80)
    print("📊 ANÁLISIS CIENTÍFICO DE RESULTADOS - 3 NIVELES")
    print("="*80)
    
    # Ordenar por robustez (PGD accuracy)
    sorted_results = sorted(all_results, key=lambda x: x['summary']['final_pgd_acc'], reverse=True)
    
    # =========================================================================
    # ANÁLISIS NIVEL 1: CONTRIBUCIÓN INDIVIDUAL
    # =========================================================================
    print("\n" + "="*80)
    print("📋 NIVEL 1: CONTRIBUCIÓN INDIVIDUAL DE COMPONENTES")
    print("="*80)
    
    baseline = next((r for r in all_results if r['name'] == 'Baseline'), None)
    if not baseline:
        print("⚠️ No se encontró Baseline")
        return all_results
    
    baseline_clean = baseline['summary']['final_clean_acc']
    baseline_pgd = baseline['summary']['final_pgd_acc']
    baseline_gap = baseline['summary']['robustness_gap']
    
    print(f"\n🔬 REFERENCIA BASELINE:")
    print(f"   Clean: {baseline_clean:.2f}% | PGD: {baseline_pgd:.2f}% | Gap: {baseline_gap:.2f}%")
    print("\n" + "-"*80)
    print(f"{'Componente':<20} {'Clean':<10} {'PGD':<10} {'ΔClean':<10} {'ΔPGD':<10} {'ΔGap':<10} {'Crítico':<10}")
    print("-"*80)
    
    individual_components = {}
    for name_pattern in ['Plasticity_Only', 'NestedCells_Only', 'MGF_Only', 'SupCon_Only', 'Symbiotic_Only']:
        result = next((r for r in all_results if r['name'] == name_pattern), None)
        if result:
            s = result['summary']
            delta_clean = s['final_clean_acc'] - baseline_clean
            delta_pgd = s['final_pgd_acc'] - baseline_pgd
            delta_gap = s['robustness_gap'] - baseline_gap
            
            # Determinar criticidad (mejora en PGD > 10% = crítico)
            is_critical = "✅ SÍ" if delta_pgd > 10 else ("⚠️ MARGINAL" if delta_pgd > 5 else "❌ NO")
            
            component_name = name_pattern.replace('_Only', '')
            individual_components[component_name] = {
                'clean': s['final_clean_acc'],
                'pgd': s['final_pgd_acc'],
                'delta_pgd': delta_pgd
            }
            
            print(f"{component_name:<20} {s['final_clean_acc']:>8.2f}% {s['final_pgd_acc']:>8.2f}% "
                  f"{delta_clean:>+8.2f}% {delta_pgd:>+8.2f}% {delta_gap:>+8.2f}% {is_critical:<10}")
    
    # =========================================================================
    # ANÁLISIS NIVEL 2: SINERGIAS Y ANTAGONISMOS
    # =========================================================================
    print("\n" + "="*80)
    print("🤝 NIVEL 2: DETECCIÓN DE SINERGIAS (Pares)")
    print("="*80)
    print("\nFórmula de Sinergia: Synergy = PGD(A+B) - [PGD(A) + PGD(B) - PGD(Baseline)]")
    print("   Synergy > 5%  → Cooperación fuerte")
    print("   Synergy < -5% → Antagonismo")
    print("\n" + "-"*80)
    print(f"{'Par':<35} {'PGD Real':<12} {'PGD Esperado':<15} {'Sinergia':<12} {'Tipo':<15}")
    print("-"*80)
    
    synergies = []
    for result in all_results:
        name = result['name']
        if name.startswith('Pair_'):
            # Extraer componentes del nombre (ej: "Pair_Plasticity_Nested" → ["Plasticity", "Nested"])
            parts = name.replace('Pair_', '').split('_')
            if len(parts) == 2:
                comp_a, comp_b = parts[0], parts[1]
                
                # Buscar PGD de componentes individuales
                pgd_a = individual_components.get(comp_a, {}).get('pgd', 0)
                pgd_b = individual_components.get(comp_b, {}).get('pgd', 0)
                
                # PGD del par
                pgd_pair = result['summary']['final_pgd_acc']
                
                # PGD esperado (aditivo)
                pgd_expected = pgd_a + pgd_b - baseline_pgd
                
                # Sinergia (efecto no-lineal)
                synergy = pgd_pair - pgd_expected
                
                # Clasificar tipo de interacción
                if synergy > 5:
                    interaction_type = "🟢 COOPERACIÓN"
                elif synergy < -5:
                    interaction_type = "🔴 ANTAGONISMO"
                else:
                    interaction_type = "🟡 ADITIVO"
                
                synergies.append({
                    'name': name,
                    'components': (comp_a, comp_b),
                    'synergy': synergy,
                    'pgd_real': pgd_pair,
                    'pgd_expected': pgd_expected
                })
                
                print(f"{name:<35} {pgd_pair:>10.2f}% {pgd_expected:>13.2f}% "
                      f"{synergy:>+10.2f}% {interaction_type:<15}")
    
    # Análisis de TRÍADAS
    print("\n" + "-"*80)
    print("🔺 ANÁLISIS DE TRÍADAS (3 componentes)")
    print("-"*80)
    print(f"{'Tríada':<40} {'PGD':<12} {'Estabilidad':<12} {'Gap':<10}")
    print("-"*80)
    
    for result in all_results:
        if result['name'].startswith('Triad_'):
            s = result['summary']
            print(f"{result['name']:<40} {s['final_pgd_acc']:>10.2f}% "
                  f"{s['stability_score']:>10.3f} {s['robustness_gap']:>8.2f}%")
    
    # =========================================================================
    # ANÁLISIS NIVEL 3: COMPONENTES ESENCIALES (Ablación Inversa)
    # =========================================================================
    print("\n" + "="*80)
    print("⚡ NIVEL 3: ANÁLISIS DE CRITICIDAD (Ablación Inversa)")
    print("="*80)
    
    full_model = next((r for r in all_results if r['name'] == 'Full_Topobrain_All5'), None)
    if not full_model:
        print("⚠️ No se encontró modelo completo")
    else:
        full_pgd = full_model['summary']['final_pgd_acc']
        full_clean = full_model['summary']['final_clean_acc']
        
        print(f"\n🔬 REFERENCIA MODELO COMPLETO (5/5 componentes):")
        print(f"   Clean: {full_clean:.2f}% | PGD: {full_pgd:.2f}%")
        print("\nCriticidad = PGD(Full) - PGD(Full_Without_X)")
        print("   Criticidad > 10%  → Componente ESENCIAL")
        print("   Criticidad < -5%  → Componente PERJUDICIAL (mejor sin él)")
        print("\n" + "-"*80)
        print(f"{'Componente Ablacionado':<30} {'PGD (4/5)':<12} {'Criticidad':<12} {'Veredicto':<20}")
        print("-"*80)
        
        criticality_scores = []
        for result in all_results:
            if result['name'].startswith('Full_Without_'):
                removed_component = result['name'].replace('Full_Without_', '')
                pgd_without = result['summary']['final_pgd_acc']
                criticality = full_pgd - pgd_without
                
                # Veredicto de criticidad
                if criticality > 10:
                    verdict = "🔥 ESENCIAL"
                elif criticality > 5:
                    verdict = "⚠️ IMPORTANTE"
                elif criticality > -5:
                    verdict = "🟡 OPCIONAL"
                else:
                    verdict = "🗑️ PERJUDICIAL"
                
                criticality_scores.append({
                    'component': removed_component,
                    'criticality': criticality,
                    'verdict': verdict
                })
                
                print(f"{removed_component:<30} {pgd_without:>10.2f}% {criticality:>+10.2f}% {verdict:<20}")
        
        # Ordenar por criticidad
        criticality_scores.sort(key=lambda x: x['criticality'], reverse=True)
        
        print("\n📊 RANKING DE CRITICIDAD (más esencial primero):")
        for i, item in enumerate(criticality_scores, 1):
            print(f"   {i}. {item['component']:<20} (Criticidad: {item['criticality']:+.2f}%)")
    
    # =========================================================================
    # RESUMEN EJECUTIVO
    # =========================================================================
    print("\n" + "="*80)
    print("📊 RESUMEN EJECUTIVO")
    print("="*80)
    
    # Top 5 configuraciones por PGD
    print("\n🏆 TOP 5 CONFIGURACIONES (por Robustez Adversarial):")
    for i, result in enumerate(sorted_results[:5], 1):
        s = result['summary']
        print(f"   {i}. {result['name']:<35} PGD: {s['final_pgd_acc']:>6.2f}% | "
              f"Clean: {s['final_clean_acc']:>6.2f}% | Gap: {s['robustness_gap']:>6.2f}%")
    
    # Mejor par sinérgico
    if synergies:
        best_synergy = max(synergies, key=lambda x: x['synergy'])
        print(f"\n🤝 MEJOR SINERGIA DETECTADA:")
        print(f"   Par: {best_synergy['name']}")
        print(f"   Sinergia: +{best_synergy['synergy']:.2f}% (superando predicción aditiva)")
    
    # Componente más crítico
    if 'criticality_scores' in locals() and criticality_scores:
        most_critical = criticality_scores[0]
        print(f"\n⚡ COMPONENTE MÁS CRÍTICO:")
        print(f"   {most_critical['component']} (Pérdida de {most_critical['criticality']:.2f}% PGD si se elimina)")
    
    # Advertencias
    print("\n⚠️  HALLAZGOS CRÍTICOS:")
    negative_gaps = [r for r in all_results if r['summary']['robustness_gap'] < 0]
    if negative_gaps:
        print(f"   • {len(negative_gaps)} configuraciones con PGD > Clean (overfitting adversarial):")
        for r in negative_gaps[:3]:
            print(f"     - {r['name']}")
    
    unstable = [r for r in all_results if r['summary']['stability_score'] < 0.5]
    if unstable:
        print(f"   • {len(unstable)} configuraciones con alta inestabilidad durante entrenamiento")
    
    print("\n✅ Ablation científico completado. Resultados guardados en 'scientific_ablation_results/'")
    return all_results

if __name__ == "__main__":
    seed_everything(42)
    print("🧠 TopoBrain - Ablation Científico Riguroso (Estable para CPU)")
    print("="*80)
    results = run_scientific_ablation()
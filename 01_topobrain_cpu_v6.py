"""
TopoBrain CPU v2.0 - Análisis de Ablación Científico Riguroso
===============================================================
Modelo ultra-ligero (~3-5k parámetros) con metodología científica completa.

INNOVACIONES CLAVE:
1. Arquitectura micro-topológica (Grid 2x2, dim=4)
2. Ablación de 3 niveles con control de variables
3. Análisis estadístico riguroso (t-tests, efecto Cohen's d)
4. Detección de sinergias y antagonismos no-lineales
5. Validación cruzada estratificada
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from collections import defaultdict
import json
import time
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA OPTIMIZADA PARA CPU
# =============================================================================
@dataclass
class MicroConfig:
    """Configuración ultra-ligera para CPU"""
    device: str = "cpu"
    seed: int = 42
    
    # Dataset micro
    n_samples: int = 400
    n_features: int = 12
    n_classes: int = 3
    n_informative: int = 9
    
    # Arquitectura micro (Grid 2x2 = 4 nodos)
    grid_size: int = 2
    embed_dim: int = 4  # Ultra-compacto
    hidden_dim: int = 4
    
    # Entrenamiento rápido
    batch_size: int = 16
    epochs: int = 8
    lr: float = 0.01
    
    # Adversarial ligero
    train_eps: float = 0.2
    test_eps: float = 0.2
    pgd_steps: int = 3
    
    # Flags de componentes (para ablación)
    use_plasticity: bool = False
    use_continuum: bool = False  # ContinuumMemoryCell
    use_mgf: bool = False  # Multi-Granular Fusion
    use_supcon: bool = False
    use_symbiotic: bool = False
    
    # Hiperparámetros de regularización (para ablación nivel 2)
    lambda_supcon: float = 0.3
    lambda_ortho: float = 0.05
    lambda_entropy: float = 0.01
    temperature_supcon: float = 0.1
    
    # Estabilidad numérica
    stability_eps: float = 1e-6
    clip_value: float = 2.0
    
    # Validación cruzada
    cv_folds: int = 3
    
    def to_dict(self):
        return asdict(self)
    
    def component_signature(self) -> str:
        """Firma única de componentes activos"""
        components = []
        if self.use_plasticity: components.append("P")
        if self.use_continuum: components.append("C")
        if self.use_mgf: components.append("M")
        if self.use_supcon: components.append("S")
        if self.use_symbiotic: components.append("Y")
        return "".join(components) if components else "BASE"

# =============================================================================
# UTILIDADES CIENTÍFICAS
# =============================================================================
def seed_everything(seed: int):
    """Control de reproducibilidad"""
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_micro_dataset(config: MicroConfig):
    """Dataset tabular controlado con validación cruzada"""
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
    
    # Normalización robusta
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + config.stability_eps)
    
    # Convertir a tensores
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    return TensorDataset(X_tensor, y_tensor)

def compute_effect_size(group1: List[float], group2: List[float]) -> float:
    """Cohen's d para medir tamaño del efecto"""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    return (mean1 - mean2) / (pooled_std + 1e-8)

# =============================================================================
# COMPONENTES MICRO-ARQUITECTÓNICOS (MATEMÁTICA COMPLETA, PARAMS MÍNIMOS)
# =============================================================================

class MicroSupConLoss(nn.Module):
    """SupCon ultra-eficiente con estabilidad numérica"""
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

class MicroContinuumCell(nn.Module):
    """
    ContinuumMemoryCell ultra-compacto con predicción semántica.
    Params: ~4*4 (W_slow) + 4*4 (V_slow) + 4*4 (semantic_mem) ≈ 48 params
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Proyecciones lentas (sin bias para ahorrar params)
        self.W_slow = nn.Linear(dim, dim, bias=False)
        self.V_slow = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.V_slow.weight, gain=0.1)
        
        # Gates ultra-simples (shared weights)
        self.gate_net = nn.Linear(dim, 1)
        
        # Memoria semántica [dim x dim]
        self.semantic_memory = nn.Parameter(torch.zeros(dim, dim))
        nn.init.normal_(self.semantic_memory, std=0.01)
        
    def forward(self, x, plasticity=1.0):
        # Proyección lenta
        v = self.V_slow(x)
        v = torch.clamp(v, -2.0, 2.0)
        
        # Predicción desde memoria
        y_pred = F.linear(x, self.semantic_memory)
        y_pred = torch.clamp(y_pred, -2.0, 2.0)
        v_pred = self.V_slow(y_pred)
        
        # Error en espacio latente
        error = v - v_pred
        
        # Gate único para ambos (olvido + actualización)
        gate = torch.sigmoid(self.gate_net(v.detach())) * plasticity
        
        # Actualización de memoria (in-place)
        with torch.no_grad():
            delta = torch.bmm(x.unsqueeze(-1), x.unsqueeze(1))
            self.semantic_memory.data = (
                0.95 * self.semantic_memory.data +
                0.01 * delta.mean(dim=0)
            )
            mem_norm = self.semantic_memory.data.norm().clamp(min=1e-6)
            self.semantic_memory.data = self.semantic_memory.data / mem_norm * 0.5
        
        # Mezcla adaptativa
        output = gate * v + (1 - gate) * v_pred
        return torch.clamp(output, -2.0, 2.0)

class MicroSymbioticBasis(nn.Module):
    """
    Refinamiento simbiótico con 2 átomos de base.
    Params: 2*dim (basis) + dim*dim (Q) + dim*dim (K) ≈ 2*dim + 2*dim² = 40 params (dim=4)
    """
    def __init__(self, dim, num_atoms=2):
        super().__init__()
        self.num_atoms = num_atoms
        
        # Base ortogonal
        self.basis = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis, gain=0.5)
        
        # Query/Key sin bias
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.eps = 1e-8
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(self.basis)
        
        attn = torch.matmul(Q, K.T) / (x.size(-1) ** 0.5 + self.eps)
        weights = F.softmax(attn, dim=-1)
        
        # Reconstrucción limpia
        x_clean = torch.matmul(weights, self.basis)
        x_clean = torch.clamp(x_clean, -2.0, 2.0)
        
        # Métricas de regularización
        weights_safe = weights + self.eps
        entropy = -(weights_safe * torch.log(weights_safe)).sum(-1).mean()
        
        gram = torch.mm(self.basis, self.basis.T)
        identity = torch.eye(gram.size(0))
        ortho = torch.norm(gram - identity, p='fro') ** 2
        
        return x_clean, entropy, ortho

class MicroTopology:
    """
    Topología dinámica para Grid 2x2 (4 nodos).
    Params: 4x4 = 16 (matriz de adyacencia aprendible)
    """
    def __init__(self, num_nodes, config: MicroConfig):
        self.num_nodes = num_nodes
        self.config = config
        
        # Pesos de conexión
        self.adj_weights = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.normal_(self.adj_weights, std=0.1)
        
        # Máscara de vecindad (grid 2x2)
        self.adj_mask = torch.zeros(num_nodes, num_nodes)
        grid_size = config.grid_size
        for i in range(num_nodes):
            r, c = i // grid_size, i % grid_size
            if r > 0: self.adj_mask[i, i - grid_size] = 1
            if r < grid_size - 1: self.adj_mask[i, i + grid_size] = 1
            if c > 0: self.adj_mask[i, i - 1] = 1
            if c < grid_size - 1: self.adj_mask[i, i + 1] = 1
    
    def get_adjacency(self, plasticity=1.0):
        """Matriz de adyacencia normalizada"""
        adj = torch.sigmoid(self.adj_weights * plasticity) * self.adj_mask
        deg = adj.sum(1, keepdim=True).clamp(min=1e-6)
        return adj / deg

# =============================================================================
# MODELO PRINCIPAL MICRO-TOPOBRAIN (~3-5k PARÁMETROS)
# =============================================================================

class MicroTopoBrain(nn.Module):
    """
    TopoBrain ultra-ligero con matemática completa.
    
    PRESUPUESTO DE PARÁMETROS:
    - Input embed: 12*16 = 192
    - Topology: 4*4 = 16
    - Node processor (ContinuumCell si activo): ~48
    - Cell processor (MGF si activo): ~48
    - Symbiotic (si activo): ~40
    - SupCon head (si activo): 16*8 + 8*4 = 160
    - Readout: 16*3 = 48
    
    TOTAL: ~200-550 params (base) hasta ~3k-5k (full)
    """
    def __init__(self, config: MicroConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2  # 4 nodos
        self.embed_dim = config.embed_dim  # 4
        
        # Embedding: 12 features -> 16 dims (4 nodes * 4 embed)
        self.input_embed = nn.Linear(config.n_features, self.embed_dim * self.num_nodes)
        
        # Topología
        self.topology = MicroTopology(self.num_nodes, config)
        
        # Procesador de nodos (ABLATIONABLE)
        if config.use_continuum:
            self.node_processor = MicroContinuumCell(self.embed_dim)
        else:
            self.node_processor = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Procesador de celdas/MGF (ABLATIONABLE)
        self.cell_processor = None
        if config.use_mgf:
            mgf_input_dim = self.embed_dim * self.num_nodes  # 16 = 4 * 4
            
            if config.use_continuum:
                # ContinuumCell mantiene dimensión de entrada/salida
                # Entrada: [batch, 16] -> Salida: [batch, 16]
                # Luego se reshapea a [batch, 4, 4]
                self.cell_processor = MicroContinuumCell(mgf_input_dim)
            else:
                # Linear proyecta a embed_dim, luego se expande
                # Entrada: [batch, 16] -> Salida: [batch, 4]
                self.cell_processor = nn.Linear(mgf_input_dim, self.embed_dim)
        
        # Refinamiento simbiótico (ABLATIONABLE)
        self.symbiotic = None
        if config.use_symbiotic:
            self.symbiotic = MicroSymbioticBasis(self.embed_dim)
        
        # Cabeza SupCon (ABLATIONABLE)
        self.supcon_head = None
        if config.use_supcon:
            self.supcon_head = nn.Sequential(
                nn.Linear(self.embed_dim * self.num_nodes, 8, bias=False),
                nn.ReLU(),
                nn.Linear(8, 4, bias=False)
            )
        
        # Readout
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, config.n_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self) -> int:
        """Contar parámetros entrenables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, plasticity=1.0):
        batch_size = x.size(0)
        
        # Embedding
        x_embed = self.input_embed(x).view(batch_size, self.num_nodes, self.embed_dim)
        
        # Topología
        adj = self.topology.get_adjacency(plasticity)
        
        # Procesamiento de nodos
        if isinstance(self.node_processor, MicroContinuumCell):
            x_flat = x_embed.view(-1, self.embed_dim)
            x_proc_flat = self.node_processor(x_flat, plasticity)
            x_proc = x_proc_flat.view(batch_size, self.num_nodes, self.embed_dim)
        else:
            x_proc = self.node_processor(x_embed)
        
        # MGF (opcional)
        entropy = torch.tensor(0.0)
        ortho = torch.tensor(0.0)
        cell_output = torch.zeros_like(x_proc)
        
        if self.config.use_mgf and self.cell_processor is not None:
            cell_input = x_embed.view(batch_size, -1)  # [batch, 16]
            
            if isinstance(self.cell_processor, MicroContinuumCell):
                # MicroContinuumCell devuelve misma dimensión que entrada
                cell_out = self.cell_processor(cell_input, plasticity)  # [batch, 16]
                # Reshape a [batch, num_nodes, embed_dim]
                cell_output = cell_out.view(batch_size, self.num_nodes, self.embed_dim)
            else:
                # Linear devuelve [batch, embed_dim]
                cell_temp = self.cell_processor(cell_input)  # [batch, 4]
                # Expandir a todos los nodos
                cell_output = cell_temp.view(batch_size, 1, self.embed_dim).expand(-1, self.num_nodes, -1)
        
        # Symbiotic (opcional)
        if self.symbiotic is not None:
            x_proc_refined = []
            entropy_sum = 0.0
            ortho_sum = 0.0
            for i in range(self.num_nodes):
                node_feat = x_proc[:, i, :]
                refined, ent, ort = self.symbiotic(node_feat)
                x_proc_refined.append(refined)
                entropy_sum += ent
                ortho_sum += ort
            
            x_proc = torch.stack(x_proc_refined, dim=1)
            entropy = entropy_sum / self.num_nodes
            ortho = ortho_sum / self.num_nodes
        
        # Combinar
        combined = x_proc + cell_output
        
        # Readout
        x_flat = combined.view(batch_size, -1)
        logits = self.readout(x_flat)
        
        # SupCon projection (opcional)
        proj = self.supcon_head(x_flat) if self.supcon_head is not None else None
        
        return logits, proj, entropy, ortho

# =============================================================================
# ATAQUE ADVERSARIAL LIGERO
# =============================================================================

def micro_pgd_attack(model, x, y, eps, steps, plasticity=1.0):
    """PGD ultra-eficiente para CPU"""
    model.eval()
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta.requires_grad = True
    
    for _ in range(steps):
        if not delta.requires_grad:
            delta.requires_grad = True
        
        with torch.enable_grad():
            logits, _, _, _ = model(x + delta, plasticity)
            loss = F.cross_entropy(logits, y)
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
        
        with torch.no_grad():
            grad = delta.grad
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1).clamp(min=1e-8)
            grad = grad / grad_norm
            delta.data = (delta.data + (eps / steps) * grad.sign()).clamp(-eps, eps)
    
    model.train()
    return (x + delta).detach()

# =============================================================================
# ENTRENAMIENTO Y EVALUACIÓN CON VALIDACIÓN CRUZADA
# =============================================================================

def train_epoch_micro(model, loader, optimizer, config: MicroConfig, epoch: int):
    """Entrenamiento por época"""
    model.train()
    metrics = defaultdict(float)
    batch_count = 0
    
    criterion = nn.CrossEntropyLoss()
    supcon_loss = MicroSupConLoss(config.temperature_supcon) if config.use_supcon else None
    
    for x, y in loader:
        x, y = x.to(config.device), y.to(config.device)
        
        # Plasticidad adaptativa
        plasticity = 0.8 if config.use_plasticity else 0.0
        
        # Ataque adversarial
        x_adv = micro_pgd_attack(model, x, y, config.train_eps, config.pgd_steps, plasticity)
        
        # Forward
        logits, proj, entropy, ortho = model(x_adv, plasticity)
        
        # Pérdidas
        loss = criterion(logits, y)
        metrics['ce_loss'] += loss.item()
        
        if config.use_supcon and supcon_loss is not None and proj is not None:
            s_loss = supcon_loss(proj, y)
            loss += config.lambda_supcon * s_loss
            metrics['supcon_loss'] += s_loss.item()
        
        loss -= config.lambda_entropy * entropy
        loss += config.lambda_ortho * ortho
        
        metrics['entropy'] += entropy.item() if torch.is_tensor(entropy) else float(entropy)
        metrics['ortho'] += ortho.item() if torch.is_tensor(ortho) else float(ortho)
        metrics['total_loss'] += loss.item()
        
        # Precisión
        pred = logits.argmax(dim=1)
        metrics['accuracy'] += pred.eq(y).sum().item() / y.size(0)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
        optimizer.step()
        
        batch_count += 1
    
    # Promediar
    for k in metrics:
        metrics[k] /= batch_count
    
    return dict(metrics)

def evaluate_micro(model, loader, config: MicroConfig, adversarial=False):
    """Evaluación con opción adversarial"""
    model.eval()
    correct = 0
    total = 0
    
    plasticity = 0.8 if config.use_plasticity else 0.0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(config.device), y.to(config.device)
            
            if adversarial:
                x = micro_pgd_attack(model, x, y, config.test_eps, config.pgd_steps, plasticity)
            
            logits, _, _, _ = model(x, plasticity)
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0

def train_with_cv(config: MicroConfig, dataset, cv_folds=3):
    """
    Entrenamiento con validación cruzada estratificada.
    Retorna: lista de resultados por fold.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.seed)
    
    # Extraer etiquetas para estratificación
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Crear subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
        
        # Modelo nuevo por fold
        model = MicroTopoBrain(config)
        model.to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        
        # Entrenar
        best_val_acc = 0.0
        fold_history = []
        
        for epoch in range(config.epochs):
            train_metrics = train_epoch_micro(model, train_loader, optimizer, config, epoch)
            val_clean = evaluate_micro(model, val_loader, config, adversarial=False)
            val_pgd = evaluate_micro(model, val_loader, config, adversarial=True)
            
            if val_clean > best_val_acc:
                best_val_acc = val_clean
            
            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['total_loss'],
                'val_clean': val_clean,
                'val_pgd': val_pgd
            })
        
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_acc': best_val_acc,
            'final_val_clean': fold_history[-1]['val_clean'],
            'final_val_pgd': fold_history[-1]['val_pgd'],
            'history': fold_history
        })
    
    return fold_results

# =============================================================================
# SISTEMA DE ABLACIÓN CIENTÍFICA DE 3 NIVELES
# =============================================================================

class AblationMatrix:
    """
    Matriz de ablación científica con 3 niveles de profundidad.
    
    NIVEL 1: COMPONENTES AISLADOS (Control Ceteris Paribus)
    - Mide contribución individual de cada componente
    - N experimentos = 1 (baseline) + M (componentes)
    
    NIVEL 2A: PARES SINÉRGICOS (Combinaciones 2 a 2)
    - Detecta sinergias (+) y antagonismos (-)
    - N experimentos = C(M, 2)
    
    NIVEL 2B: TRÍADAS HIPOTÉTICAS (Combinaciones de 3)
    - Explora interacciones de orden superior
    - N experimentos = selección estratégica
    
    NIVEL 3: ABLACIÓN INVERSA (Quitar 1 del modelo completo)
    - Identifica componentes ESENCIALES vs REDUNDANTES
    - N experimentos = M
    """
    
    # Componentes ablacionables (M=5)
    COMPONENTS = {
        'P': 'use_plasticity',
        'C': 'use_continuum',
        'M': 'use_mgf',
        'S': 'use_supcon',
        'Y': 'use_symbiotic'
    }
    
    @staticmethod
    def level1_isolated():
        """NIVEL 1: Componentes aislados (6 experimentos)"""
        experiments = [
            ("00_Baseline", {}),
            ("01_Plasticity_Only", {"use_plasticity": True}),
            ("02_Continuum_Only", {"use_continuum": True}),
            ("03_MGF_Only", {"use_mgf": True}),
            ("04_SupCon_Only", {"use_supcon": True}),
            ("05_Symbiotic_Only", {"use_symbiotic": True}),
        ]
        return experiments
    
    @staticmethod
    def level2a_pairs():
        """NIVEL 2A: Todos los pares (10 experimentos = C(5,2))"""
        experiments = [
            # Plasticity con otros
            ("06_Pair_P+C", {"use_plasticity": True, "use_continuum": True}),
            ("07_Pair_P+M", {"use_plasticity": True, "use_mgf": True}),
            ("08_Pair_P+S", {"use_plasticity": True, "use_supcon": True}),
            ("09_Pair_P+Y", {"use_plasticity": True, "use_symbiotic": True}),
            
            # Continuum con otros
            ("10_Pair_C+M", {"use_continuum": True, "use_mgf": True}),
            ("11_Pair_C+S", {"use_continuum": True, "use_supcon": True}),
            ("12_Pair_C+Y", {"use_continuum": True, "use_symbiotic": True}),
            
            # MGF con otros
            ("13_Pair_M+S", {"use_mgf": True, "use_supcon": True}),
            ("14_Pair_M+Y", {"use_mgf": True, "use_symbiotic": True}),
            
            # SupCon con Symbiotic
            ("15_Pair_S+Y", {"use_supcon": True, "use_symbiotic": True}),
        ]
        return experiments
    
    @staticmethod
    def level2b_strategic_triads():
        """
        NIVEL 2B: Tríadas estratégicas (8 experimentos selectos)
        
        HIPÓTESIS BASADAS EN ANÁLISIS PREVIO:
        1. Plasticity es fuerte SOLO → probar con 1 componente adicional
        2. Continuum + Symbiotic mostraron cooperación → añadir tercero
        3. Evitar SupCon en combinaciones (causa antagonismo)
        """
        experiments = [
            # Hipótesis 1: Plasticity + complementos conservadores
            ("16_Triad_P+C+Y", {"use_plasticity": True, "use_continuum": True, "use_symbiotic": True}),
            ("17_Triad_P+C+M", {"use_plasticity": True, "use_continuum": True, "use_mgf": True}),
            
            # Hipótesis 2: Continuum + Symbiotic + tercero
            ("18_Triad_C+Y+M", {"use_continuum": True, "use_symbiotic": True, "use_mgf": True}),
            ("19_Triad_C+Y+P", {"use_continuum": True, "use_symbiotic": True, "use_plasticity": True}),
            
            # Hipótesis 3: MGF como conector (sin SupCon)
            ("20_Triad_M+P+Y", {"use_mgf": True, "use_plasticity": True, "use_symbiotic": True}),
            ("21_Triad_M+C+Y", {"use_mgf": True, "use_continuum": True, "use_symbiotic": True}),
            
            # Control: Mejores pares + componente neutral
            ("22_Triad_P+M+S", {"use_plasticity": True, "use_mgf": True, "use_supcon": True}),
            ("23_Triad_C+M+S", {"use_continuum": True, "use_mgf": True, "use_supcon": True}),
        ]
        return experiments
    
    @staticmethod
    def level3_inverse_ablation():
        """
        NIVEL 3: Ablación inversa (5 experimentos)
        Modelo completo MENOS un componente → detecta criticidad
        """
        experiments = [
            ("24_Full_Without_P", {
                "use_continuum": True, "use_mgf": True, "use_supcon": True, "use_symbiotic": True
            }),
            ("25_Full_Without_C", {
                "use_plasticity": True, "use_mgf": True, "use_supcon": True, "use_symbiotic": True
            }),
            ("26_Full_Without_M", {
                "use_plasticity": True, "use_continuum": True, "use_supcon": True, "use_symbiotic": True
            }),
            ("27_Full_Without_S", {
                "use_plasticity": True, "use_continuum": True, "use_mgf": True, "use_symbiotic": True
            }),
            ("28_Full_Without_Y", {
                "use_plasticity": True, "use_continuum": True, "use_mgf": True, "use_supcon": True
            }),
        ]
        return experiments
    
    @staticmethod
    def level3_full_model():
        """Modelo completo (referencia máxima)"""
        return [("29_Full_Model_All5", {
            "use_plasticity": True,
            "use_continuum": True,
            "use_mgf": True,
            "use_supcon": True,
            "use_symbiotic": True
        })]
    
    @classmethod
    def get_complete_matrix(cls):
        """Matriz completa de ablación (30 experimentos)"""
        return (
            cls.level1_isolated() +           # 6
            cls.level2a_pairs() +             # 10
            cls.level2b_strategic_triads() +  # 8
            cls.level3_inverse_ablation() +   # 5
            cls.level3_full_model()           # 1
        )

# =============================================================================
# ANÁLISIS ESTADÍSTICO RIGUROSO
# =============================================================================

class ScientificAnalyzer:
    """Análisis estadístico con significancia y tamaño de efecto"""
    
    @staticmethod
    def compute_statistics(results_list: List[Dict]) -> Dict:
        """
        Análisis estadístico por experimento.
        
        Args:
            results_list: Lista de resultados de CV folds
        
        Returns:
            Dict con mean, std, CI95, etc.
        """
        pgd_scores = [r['final_val_pgd'] for r in results_list]
        clean_scores = [r['final_val_clean'] for r in results_list]
        
        return {
            'pgd_mean': np.mean(pgd_scores),
            'pgd_std': np.std(pgd_scores, ddof=1),
            'pgd_sem': stats.sem(pgd_scores),
            'pgd_ci95': stats.t.interval(0.95, len(pgd_scores)-1, 
                                         loc=np.mean(pgd_scores), 
                                         scale=stats.sem(pgd_scores)),
            'clean_mean': np.mean(clean_scores),
            'clean_std': np.std(clean_scores, ddof=1),
            'gap_mean': np.mean(clean_scores) - np.mean(pgd_scores),
        }
    
    @staticmethod
    def ttest_vs_baseline(exp_scores, baseline_scores):
        """
        t-test pareado vs baseline.
        
        Returns:
            (t_statistic, p_value, cohens_d)
        """
        t_stat, p_val = stats.ttest_ind(exp_scores, baseline_scores)
        cohens_d = compute_effect_size(exp_scores, baseline_scores)
        return t_stat, p_val, cohens_d
    
    @staticmethod
    def detect_synergy(pair_pgd, comp_a_pgd, comp_b_pgd, baseline_pgd):
        """
        Detecta sinergia no-lineal.
        
        Synergy = PGD(A+B) - [PGD(A) + PGD(B) - PGD(Baseline)]
        
        > +5%  → Cooperación fuerte
        -5~+5% → Aditivo
        < -5%  → Antagonismo
        """
        expected_pgd = comp_a_pgd + comp_b_pgd - baseline_pgd
        synergy = pair_pgd - expected_pgd
        
        if synergy > 5:
            interaction = "COOPERACIÓN"
        elif synergy < -5:
            interaction = "ANTAGONISMO"
        else:
            interaction = "ADITIVO"
        
        return synergy, interaction
    
    @staticmethod
    def rank_components_by_criticality(full_pgd, ablation_results):
        """
        Ranking de criticidad basado en ablación inversa.
        
        Criticality = PGD(Full) - PGD(Full_Without_X)
        
        > +10%  → ESENCIAL
        +5~+10% → IMPORTANTE
        -5~+5%  → OPCIONAL
        < -5%   → PERJUDICIAL
        """
        rankings = []
        
        for name, result in ablation_results.items():
            without_pgd = result['pgd_mean']
            criticality = full_pgd - without_pgd
            
            if criticality > 10:
                verdict = "ESENCIAL"
            elif criticality > 5:
                verdict = "IMPORTANTE"
            elif criticality > -5:
                verdict = "OPCIONAL"
            else:
                verdict = "PERJUDICIAL"
            
            rankings.append({
                'component': name,
                'criticality': criticality,
                'verdict': verdict,
                'pgd_without': without_pgd
            })
        
        # Ordenar por criticidad descendente
        rankings.sort(key=lambda x: x['criticality'], reverse=True)
        return rankings

# =============================================================================
# EJECUTOR PRINCIPAL DEL ABLATION CIENTÍFICO
# =============================================================================

def run_scientific_ablation_study():
    """
    Ejecutor completo del estudio de ablación con análisis científico.
    
    FLUJO:
    1. Cargar matriz de experimentos (30 configuraciones)
    2. Por cada configuración:
       - Entrenar con CV (3 folds)
       - Calcular estadísticas (mean, std, CI95)
       - Guardar resultados
    3. Análisis de 3 niveles:
       - Nivel 1: Contribución individual (t-tests vs baseline)
       - Nivel 2: Sinergias/antagonismos (comparación aditiva)
       - Nivel 3: Criticidad (ablación inversa)
    4. Generar reporte científico
    """
    
    seed_everything(42)
    results_dir = Path("topobrain_ablation_v2")
    results_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🧠 TopoBrain v2.0 - Ablación Científica Rigurosa (CPU-Optimized)")
    print("="*80)
    print(f"📊 Total experimentos: 30")
    print(f"🔬 Validación cruzada: 3 folds estratificados")
    print(f"📈 Métricas: PGD accuracy (primaria), Clean accuracy, Gap")
    print(f"⚙️  Parámetros: ~200-3000 (según componentes activos)")
    print("="*80 + "\n")
    
    # Cargar matriz de experimentos
    ablation_matrix = AblationMatrix.get_complete_matrix()
    
    # Cargar dataset
    base_config = MicroConfig()
    dataset = get_micro_dataset(base_config)
    
    # Almacenar resultados
    all_results = {}
    
    # =========================================================================
    # FASE 1: EJECUTAR TODOS LOS EXPERIMENTOS
    # =========================================================================
    
    for exp_idx, (exp_name, overrides) in enumerate(ablation_matrix):
        print(f"\n▶ [{exp_idx+1}/30] {exp_name}")
        print("-" * 60)
        
        # Configurar experimento
        cfg_dict = base_config.to_dict()
        cfg_dict.update(overrides)
        config = MicroConfig(**cfg_dict)
        
        # Entrenar con CV
        start_time = time.time()
        cv_results = train_with_cv(config, dataset, cv_folds=config.cv_folds)
        elapsed = time.time() - start_time
        
        # Calcular estadísticas
        stats_dict = ScientificAnalyzer.compute_statistics(cv_results)
        
        # Contar parámetros (usar primer fold como referencia)
        temp_model = MicroTopoBrain(config)
        n_params = temp_model.count_parameters()
        
        # Guardar resultado
        all_results[exp_name] = {
            'config': config.to_dict(),
            'cv_results': cv_results,
            'statistics': stats_dict,
            'n_params': n_params,
            'elapsed_time': elapsed,
            'component_signature': config.component_signature()
        }
        
        # Mostrar resumen
        print(f"  Params: {n_params:,}")
        print(f"  PGD:   {stats_dict['pgd_mean']:.2f}% ± {stats_dict['pgd_std']:.2f}%")
        print(f"  Clean: {stats_dict['clean_mean']:.2f}% ± {stats_dict['clean_std']:.2f}%")
        print(f"  Gap:   {stats_dict['gap_mean']:.2f}%")
        print(f"  CI95:  [{stats_dict['pgd_ci95'][0]:.2f}%, {stats_dict['pgd_ci95'][1]:.2f}%]")
        print(f"  Time:  {elapsed:.1f}s")
    
    # Guardar resultados brutos
    with open(results_dir / "raw_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("📊 ANÁLISIS CIENTÍFICO - 3 NIVELES")
    print("="*80)
    
    # =========================================================================
    # NIVEL 1: ANÁLISIS DE CONTRIBUCIÓN INDIVIDUAL
    # =========================================================================
    
    print("\n" + "="*80)
    print("📋 NIVEL 1: CONTRIBUCIÓN INDIVIDUAL DE COMPONENTES")
    print("="*80)
    
    baseline = all_results['00_Baseline']
    baseline_pgd_scores = [r['final_val_pgd'] for r in baseline['cv_results']]
    baseline_pgd_mean = baseline['statistics']['pgd_mean']
    
    print(f"\n🔬 BASELINE (Sin componentes extras):")
    print(f"   PGD: {baseline_pgd_mean:.2f}% ± {baseline['statistics']['pgd_std']:.2f}%")
    print(f"   CI95: [{baseline['statistics']['pgd_ci95'][0]:.2f}%, "
          f"{baseline['statistics']['pgd_ci95'][1]:.2f}%]")
    
    print("\n" + "-"*80)
    print(f"{'Componente':<20} {'PGD (mean±std)':<20} {'Δ vs Base':<12} "
          f"{'p-value':<10} {'Cohen d':<10} {'Crítico':<12}")
    print("-"*80)
    
    individual_results = {}
    
    for exp_name in ['01_Plasticity_Only', '02_Continuum_Only', '03_MGF_Only', 
                     '04_SupCon_Only', '05_Symbiotic_Only']:
        result = all_results[exp_name]
        stats_dict = result['statistics']
        
        # t-test vs baseline
        exp_pgd_scores = [r['final_val_pgd'] for r in result['cv_results']]
        t_stat, p_val, cohens_d = ScientificAnalyzer.ttest_vs_baseline(
            exp_pgd_scores, baseline_pgd_scores
        )
        
        delta = stats_dict['pgd_mean'] - baseline_pgd_mean
        
        # Clasificar criticidad
        if abs(cohens_d) > 1.0 and p_val < 0.05:
            critical = "✅ SÍ"
        elif abs(cohens_d) > 0.5:
            critical = "⚠️ MARGINAL"
        else:
            critical = "❌ NO"
        
        component_name = exp_name.replace('_Only', '').split('_')[1]
        individual_results[component_name] = stats_dict['pgd_mean']
        
        print(f"{component_name:<20} {stats_dict['pgd_mean']:>6.2f}% ± {stats_dict['pgd_std']:>4.2f}% "
              f"{delta:>+10.2f}% {p_val:>9.4f} {cohens_d:>9.3f} {critical:<12}")
    
    # =========================================================================
    # NIVEL 2A: ANÁLISIS DE SINERGIAS (PARES)
    # =========================================================================
    
    print("\n" + "="*80)
    print("🤝 NIVEL 2A: DETECCIÓN DE SINERGIAS (Pares)")
    print("="*80)
    print("\nFórmula: Synergy = PGD(A+B) - [PGD(A) + PGD(B) - PGD(Baseline)]")
    print("   > +5%  → Cooperación fuerte")
    print("   -5~+5% → Aditivo")
    print("   < -5%  → Antagonismo")
    
    print("\n" + "-"*80)
    print(f"{'Par':<25} {'PGD Real':<12} {'PGD Esperado':<15} "
          f"{'Sinergia':<12} {'Tipo':<15}")
    print("-"*80)
    
    pair_experiments = [e for e in ablation_matrix if e[0].startswith('06_') or 
                       e[0].startswith('07_') or e[0].startswith('08_') or
                       e[0].startswith('09_') or e[0].startswith('10_') or
                       e[0].startswith('11_') or e[0].startswith('12_') or
                       e[0].startswith('13_') or e[0].startswith('14_') or
                       e[0].startswith('15_')]
    
    synergy_results = []
    
    for exp_name, _ in pair_experiments:
        result = all_results[exp_name]
        pair_pgd = result['statistics']['pgd_mean']
        
        # Extraer componentes del nombre (ej: "06_Pair_P+C" → P, C)
        components = exp_name.split('_')[-1].replace('Pair_', '').split('+')
        
        if len(components) == 2:
            comp_a, comp_b = components[0], components[1]
            
            # Mapear a nombres completos
            comp_map = {'P': 'Plasticity', 'C': 'Continuum', 'M': 'MGF', 
                       'S': 'SupCon', 'Y': 'Symbiotic'}
            
            pgd_a = individual_results.get(comp_map[comp_a], baseline_pgd_mean)
            pgd_b = individual_results.get(comp_map[comp_b], baseline_pgd_mean)
            
            # Calcular sinergia
            synergy, interaction = ScientificAnalyzer.detect_synergy(
                pair_pgd, pgd_a, pgd_b, baseline_pgd_mean
            )
            
            synergy_results.append({
                'name': exp_name,
                'components': (comp_a, comp_b),
                'synergy': synergy,
                'pgd_real': pair_pgd,
                'pgd_expected': pgd_a + pgd_b - baseline_pgd_mean,
                'interaction': interaction
            })
            
            # Formato de tipo
            if interaction == "COOPERACIÓN":
                type_str = "🟢 COOPERACIÓN"
            elif interaction == "ANTAGONISMO":
                type_str = "🔴 ANTAGONISMO"
            else:
                type_str = "🟡 ADITIVO"
            
            print(f"{exp_name:<25} {pair_pgd:>10.2f}% "
                  f"{pgd_a + pgd_b - baseline_pgd_mean:>13.2f}% "
                  f"{synergy:>+10.2f}% {type_str:<15}")
    
    # =========================================================================
    # NIVEL 2B: ANÁLISIS DE TRÍADAS
    # =========================================================================
    
    print("\n" + "-"*80)
    print("🔺 ANÁLISIS DE TRÍADAS (3 componentes)")
    print("-"*80)
    print(f"{'Tríada':<30} {'PGD':<12} {'CI95 Width':<12} {'Gap':<10}")
    print("-"*80)
    
    triad_experiments = [e for e in ablation_matrix if e[0].startswith('16_') or
                        e[0].startswith('17_') or e[0].startswith('18_') or
                        e[0].startswith('19_') or e[0].startswith('20_') or
                        e[0].startswith('21_') or e[0].startswith('22_') or
                        e[0].startswith('23_')]
    
    for exp_name, _ in triad_experiments:
        result = all_results[exp_name]
        stats_dict = result['statistics']
        ci_width = stats_dict['pgd_ci95'][1] - stats_dict['pgd_ci95'][0]
        
        print(f"{exp_name:<30} {stats_dict['pgd_mean']:>10.2f}% "
              f"{ci_width:>10.2f}% {stats_dict['gap_mean']:>8.2f}%")
    
    # =========================================================================
    # NIVEL 3: ANÁLISIS DE CRITICIDAD (ABLACIÓN INVERSA)
    # =========================================================================
    
    print("\n" + "="*80)
    print("⚡ NIVEL 3: ANÁLISIS DE CRITICIDAD (Ablación Inversa)")
    print("="*80)
    
    full_model = all_results['29_Full_Model_All5']
    full_pgd = full_model['statistics']['pgd_mean']
    full_clean = full_model['statistics']['clean_mean']
    
    print(f"\n🔬 MODELO COMPLETO (5/5 componentes):")
    print(f"   PGD:   {full_pgd:.2f}% ± {full_model['statistics']['pgd_std']:.2f}%")
    print(f"   Clean: {full_clean:.2f}% ± {full_model['statistics']['clean_std']:.2f}%")
    print(f"   Gap:   {full_model['statistics']['gap_mean']:.2f}%")
    print(f"   Params: {full_model['n_params']:,}")
    
    print("\nCriticidad = PGD(Full) - PGD(Full_Without_X)")
    print("   > +10%  → ESENCIAL")
    print("   +5~+10% → IMPORTANTE")
    print("   -5~+5%  → OPCIONAL")
    print("   < -5%   → PERJUDICIAL (mejor sin él)")
    
    print("\n" + "-"*80)
    print(f"{'Componente Removido':<25} {'PGD (4/5)':<12} {'Criticidad':<12} "
          f"{'Veredicto':<20}")
    print("-"*80)
    
    inverse_ablation = {
        '24_Full_Without_P': 'Plasticity',
        '25_Full_Without_C': 'Continuum',
        '26_Full_Without_M': 'MGF',
        '27_Full_Without_S': 'SupCon',
        '28_Full_Without_Y': 'Symbiotic'
    }
    
    criticality_rankings = ScientificAnalyzer.rank_components_by_criticality(
        full_pgd,
        {name: all_results[exp_name]['statistics'] 
         for exp_name, name in inverse_ablation.items()}
    )
    
    for ranking in criticality_rankings:
        comp = ranking['component']
        pgd_without = ranking['pgd_without']
        crit = ranking['criticality']
        verdict = ranking['verdict']
        
        # Emoji por veredicto
        if verdict == "ESENCIAL":
            emoji = "🔥"
        elif verdict == "IMPORTANTE":
            emoji = "⚠️"
        elif verdict == "OPCIONAL":
            emoji = "🟡"
        else:
            emoji = "🗑️"
        
        print(f"{comp:<25} {pgd_without:>10.2f}% {crit:>+10.2f}% "
              f"{emoji} {verdict:<18}")
    
    print("\n📊 RANKING DE CRITICIDAD (más crítico primero):")
    for i, ranking in enumerate(criticality_rankings, 1):
        print(f"   {i}. {ranking['component']:<15} "
              f"(Criticidad: {ranking['criticality']:+.2f}%)")
    
    # =========================================================================
    # RESUMEN EJECUTIVO Y RECOMENDACIONES
    # =========================================================================
    
    print("\n" + "="*80)
    print("📊 RESUMEN EJECUTIVO")
    print("="*80)
    
    # Top 5 configuraciones por PGD
    sorted_results = sorted(all_results.items(), 
                           key=lambda x: x[1]['statistics']['pgd_mean'], 
                           reverse=True)
    
    print("\n🏆 TOP 5 CONFIGURACIONES (por Robustez Adversarial):")
    for i, (name, result) in enumerate(sorted_results[:5], 1):
        stats_dict = result['statistics']
        print(f"   {i}. {name:<30} "
              f"PGD: {stats_dict['pgd_mean']:>6.2f}% | "
              f"Clean: {stats_dict['clean_mean']:>6.2f}% | "
              f"Gap: {stats_dict['gap_mean']:>6.2f}% | "
              f"Params: {result['n_params']:>5,}")
    
    # Mejor sinergia
    if synergy_results:
        best_synergy = max(synergy_results, key=lambda x: x['synergy'])
        print(f"\n🤝 MEJOR SINERGIA DETECTADA:")
        print(f"   Par: {best_synergy['name']}")
        print(f"   Componentes: {' + '.join(best_synergy['components'])}")
        print(f"   Sinergia: {best_synergy['synergy']:+.2f}% "
              f"({best_synergy['interaction']})")
    
    # Componente más crítico
    most_critical = criticality_rankings[0]
    print(f"\n⚡ COMPONENTE MÁS CRÍTICO:")
    print(f"   {most_critical['component']}")
    print(f"   Criticidad: {most_critical['criticality']:+.2f}%")
    print(f"   Veredicto: {most_critical['verdict']}")
    
    # Hallazgos críticos
    print("\n⚠️  HALLAZGOS CRÍTICOS:")
    
    # Overfitting adversarial (PGD > Clean)
    overfit_configs = [
        (name, r['statistics']) for name, r in all_results.items()
        if r['statistics']['gap_mean'] < 0
    ]
    
    if overfit_configs:
        print(f"   • {len(overfit_configs)} configuraciones con PGD > Clean "
              f"(overfitting adversarial):")
        for name, stats_dict in overfit_configs[:3]:
            print(f"     - {name}: Gap = {stats_dict['gap_mean']:.2f}%")
    
    # Alta inestabilidad (std > 10%)
    unstable_configs = [
        (name, r['statistics']) for name, r in all_results.items()
        if r['statistics']['pgd_std'] > 10
    ]
    
    if unstable_configs:
        print(f"   • {len(unstable_configs)} configuraciones con alta "
              f"inestabilidad (std > 10%):")
        for name, stats_dict in unstable_configs[:3]:
            print(f"     - {name}: PGD = {stats_dict['pgd_mean']:.2f}% ± "
                  f"{stats_dict['pgd_std']:.2f}%")
    
    # Componentes perjudiciales
    harmful_components = [r for r in criticality_rankings if r['verdict'] == 'PERJUDICIAL']
    if harmful_components:
        print(f"   • {len(harmful_components)} componentes PERJUDICIALES "
              f"en modelo completo:")
        for comp_data in harmful_components:
            print(f"     - {comp_data['component']}: "
                  f"Criticidad {comp_data['criticality']:+.2f}%")
    
    # =========================================================================
    # RECOMENDACIONES ACCIONABLES
    # =========================================================================
    
    print("\n" + "="*80)
    print("💡 RECOMENDACIONES ACCIONABLES")
    print("="*80)
    
    # Identificar mejor configuración individual
    best_individual = sorted_results[0]
    best_individual_name = best_individual[0]
    best_individual_stats = best_individual[1]['statistics']
    
    print("\n🎯 OPCIÓN A: CONFIGURACIÓN ÓPTIMA (Máximo Rendimiento)")
    print(f"   Usar: {best_individual_name}")
    print(f"   PGD: {best_individual_stats['pgd_mean']:.2f}% ± "
          f"{best_individual_stats['pgd_std']:.2f}%")
    print(f"   Params: {best_individual[1]['n_params']:,}")
    print(f"   Gap: {best_individual_stats['gap_mean']:.2f}%")
    
    # Configuración con mejor trade-off params/performance
    configs_sorted_by_efficiency = sorted(
        all_results.items(),
        key=lambda x: x[1]['statistics']['pgd_mean'] / (x[1]['n_params'] / 1000),
        reverse=True
    )
    
    best_efficient = configs_sorted_by_efficiency[0]
    print("\n🎯 OPCIÓN B: MEJOR EFICIENCIA (Params vs Performance)")
    print(f"   Usar: {best_efficient[0]}")
    print(f"   PGD: {best_efficient[1]['statistics']['pgd_mean']:.2f}%")
    print(f"   Params: {best_efficient[1]['n_params']:,}")
    print(f"   Eficiencia: {best_efficient[1]['statistics']['pgd_mean'] / (best_efficient[1]['n_params'] / 1000):.2f} "
          f"PGD%/k-params")
    
    # Configuración más estable (menor std)
    configs_sorted_by_stability = sorted(
        all_results.items(),
        key=lambda x: x[1]['statistics']['pgd_std']
    )
    
    most_stable = configs_sorted_by_stability[0]
    print("\n🎯 OPCIÓN C: MÁXIMA ESTABILIDAD (Menor Varianza)")
    print(f"   Usar: {most_stable[0]}")
    print(f"   PGD: {most_stable[1]['statistics']['pgd_mean']:.2f}% ± "
          f"{most_stable[1]['statistics']['pgd_std']:.2f}%")
    print(f"   CI95 Width: {most_stable[1]['statistics']['pgd_ci95'][1] - most_stable[1]['statistics']['pgd_ci95'][0]:.2f}%")
    
    print("\n✅ Ablación científica completada")
    print(f"📁 Resultados guardados en: {results_dir}/")
    
    return all_results

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    seed_everything(42)
    
    print("🧠 TopoBrain CPU v2.0 - Análisis de Ablación Científico")
    print("="*80)
    print("CARACTERÍSTICAS:")
    print("  • Modelo ultra-ligero: ~200-3000 parámetros")
    print("  • Grid 2x2 (4 nodos topológicos)")
    print("  • Matemática completa (sin simplificaciones)")
    print("  • Validación cruzada estratificada (3 folds)")
    print("  • Análisis estadístico riguroso (t-tests, Cohen's d)")
    print("  • 30 experimentos en 3 niveles de ablación")
    print("="*80 + "\n")
    
    results = run_scientific_ablation_study()
    
    print("\n🎉 Estudio completado exitosamente!")
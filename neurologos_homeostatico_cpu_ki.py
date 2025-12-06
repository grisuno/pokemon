"""
NeuroLogos v5.2 - TopoBrain with Homeostatic Core
=================================================
Fusión de dos paradigmas:
1. Metodología de ablación 3-niveles (rigor científico)
2. Control homeostático interno (adaptación biológica)

El regulador monitoriza: sorpresa adversarial, excitación neuronal, 
fatiga de pesos y ajusta: metabolismo, sensibilidad, plasticidad, gating.
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
import random, os

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA
# =============================================================================

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA
# =============================================================================


@dataclass 
class MicroConfig:
    device: str = "cpu"
    seed: int = 42
    
    # Dataset (aumentar tamaño para métricas estables)
    n_samples: int = 600  # Aumentado de 400 a 600
    n_features: int = 12
    n_classes: int = 3
    n_informative: int = 9
    
    # Arquitectura
    grid_size: int = 2
    embed_dim: int = 4
    hidden_dim: int = 4
    
    # Entrenamiento
    batch_size: int = 16
    epochs: int = 15  # Aumentado de 12 a 15 para convergencia
    lr: float = 0.008  # Reducido para estabilidad
    
    # Adversarial - MÁS AGRESIVO para que sea visible
    train_eps: float = 0.25  # Aumentado de 0.2
    test_eps: float = 0.25
    pgd_steps: int = 5  # Aumentado de 3 a 5 para ataque más fuerte
    
    # FLAGS DE ABLACIÓN
    use_plasticity: bool = False
    use_continuum: bool = False
    use_mgf: bool = False
    use_supcon: bool = False
    use_symbiotic: bool = False
    
    # HOMEOSTASIS
    use_homeostasis: bool = False
    homeostatic_lr: float = 0.05  # Reducido para adaptación más suave


def seed_everything(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# FUNCIÓN FALTANTE - DATASET GENERATOR
# =============================================================================

def get_dataset(config: MicroConfig):
    """
    Genera dataset sintético para el estudio de ablación.
    Normaliza features en rango [0,1] para estabilidad homeostática.
    """
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
# NÚCLEO HOMEOSTÁTICO - FIX 1: var() warning y tensor conversion
# =============================================================================

class HomeostaticCore(nn.Module):
    """
    Cerebro interno que monitoriza el estado fisiológico de la red
    y emite señales de control adaptativas.
    FIXES:
    1. Maneja batch_size=1 evitando warning de var()
    2. Convierte loss_val correctamente a tensor
    3. Asegura device placement consistente
    """
    def __init__(self, d_in, base_lr=0.1):
        super().__init__()
        self.base_lr = base_lr
        
        # Sensores de estado: [sorpresa, excitación, fatiga, coherencia]
        self.sensor_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # Actuadores: [metabolismo, sensibilidad, gate, plasticidad_base]
        self.actuator_net = nn.Sequential(
            nn.Linear(8, 4),
            nn.Sigmoid()
        )
        
    def forward(self, x, h_pre, w_norm, loss_val=None):
        batch_size = x.size(0)
        
        # FIX: Manejar caso batch_size=1 o single-feature que causa warning
        if x.size(1) <= 1:
            surprise = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        else:
            surprise = (x.var(dim=1, keepdim=True, unbiased=False) - 0.5).abs()
        
        # SENSOR DE EXCITACIÓN (magnitud de activaciones)
        excitation = h_pre.abs().mean(dim=1, keepdim=True) if h_pre is not None else torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        
        # SENSOR DE FATIGA (norma de pesos)
        fatigue = w_norm.view(1, 1).expand(batch_size, 1)
        
        # FIX: Convertir loss_val a tensor con device correcto
        if loss_val is None:
            coherence = torch.tensor(0.5, device=x.device, dtype=x.dtype).view(1, 1).expand(batch_size, 1)
        else:
            # Asegurar que loss_val sea tensor escalar en el device correcto
            loss_tensor = loss_val if isinstance(loss_val, torch.Tensor) else torch.tensor(loss_val, device=x.device, dtype=x.dtype)
            coherence = (1.0 / (1.0 + loss_tensor)).view(1, 1).expand(batch_size, 1)
        
        state_vector = torch.cat([surprise, excitation, fatigue, coherence], dim=1)
        hidden_state = self.sensor_net(state_vector)
        controls = self.actuator_net(hidden_state)
        
        return {
            'metabolism': controls[:, 0].view(-1, 1),
            'sensitivity': controls[:, 1].view(-1, 1),
            'gate': controls[:, 2].view(-1, 1),
            'base_plasticity': controls[:, 3].view(-1, 1)
        }

# =============================================================================
# COMPONENTES CON HOMEOSTASIS
# =============================================================================

class MicroContinuumCell(nn.Module):
    """Versión homeostática con regulación de metabolismo"""
    def __init__(self, dim, config: MicroConfig):
        super().__init__()
        self.dim = dim
        self.config = config
        self.base_lr = config.homeostatic_lr
        
        # Pesos lentos
        self.W_slow = nn.Linear(dim, dim, bias=False)
        self.V_slow = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.V_slow.weight, gain=0.1)
        
        # Gate homeostático
        self.gate_net = nn.Linear(dim, 1)
        
        # Memoria semántica (no entrenable)
        self.register_buffer('semantic_memory', torch.zeros(dim, dim))
        nn.init.normal_(self.semantic_memory, std=0.01)
        
        # Regulador homeostático
        self.homeostatic_core = HomeostaticCore(dim, self.base_lr) if config.use_homeostasis else None
        
    def forward(self, x, plasticity=1.0):
        v = self.V_slow(x)
        v = torch.clamp(v, -2.0, 2.0)
        
        # Homeostasis: el cerebro decide cómo aprender
        metabolic_rate = 1.0
        gate_value = plasticity
        
        if self.homeostatic_core is not None:
            with torch.no_grad():
                h_raw = self.W_slow(x)
                w_norm = self.W_slow.weight.norm()
                physio = self.homeostatic_core(x, h_raw, w_norm)
                metabolic_rate = physio['metabolism'].mean().item()
                gate_value = physio['gate'].mean().item()
        
        # Gate dinámico
        gate = torch.sigmoid(self.gate_net(v)) * gate_value
        
        # Actualización Hebbian con metabolismo controlado
        if self.training:
            with torch.no_grad():
                y = v
                batch = x.size(0)
                hebb = torch.mm(y.T, x) / batch
                forget = (y**2).mean(0).unsqueeze(1) * self.semantic_memory
                delta = torch.tanh(hebb - forget)
                self.semantic_memory.add_(delta * metabolic_rate * 0.1)
                
                # Normalización
                mem_norm = self.semantic_memory.norm().clamp(min=1e-6)
                self.semantic_memory.copy_(self.semantic_memory / mem_norm * 0.5)
        
        # Output con gate homeostático
        output = gate * v
        return torch.clamp(output, -2.0, 2.0)

# =============================================================================
# MICROSYMBIOTIC BASIS - FIX 5: Hacer el componente vulnerable a perturbaciones
# =============================================================================


class MicroSymbioticBasis(nn.Module):
    """
    Base simbiótica - VERSION FINAL
    FIXES:
    1. Remover batch norm que causaba colapso
    2. Aumentar ruido interno para no ser demasiado robusto
    3. Añadir regularización de varianza mínima
    4. Regulación de entropía más fuerte para evitar picos
    """
    def __init__(self, dim, config: MicroConfig):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(2, dim))
        nn.init.orthogonal_(self.basis, gain=0.3)  # Reducido de 0.5 a 0.3 para más varianza
        
        self.query = nn.Linear(dim, dim, bias=True)  # Añadir bias
        self.key = nn.Linear(dim, dim, bias=True)
        
        # Ruido más fuerte para asegurar vulnerabilidad
        self.noise_std = 0.1  # Aumentado de 0.05
        
        self.eps = 1e-8
        
        # Regularización de temperatura
        self.temperature = max(np.sqrt(dim) * 0.3, 0.5)  # Más suave
        
        # Homeostasis
        self.homeostatic_core = HomeostaticCore(dim) if config.use_homeostasis else None
        
    def forward(self, x, loss_val=None):
        # Ruido gaussiano durante entrenamiento
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x
        
        Q = self.query(x_noisy)
        K = self.key(self.basis)
        
        # Temperatura scaling para distribución suave
        attn = torch.matmul(Q, K.T) / self.temperature
        
        # FIX: Clipping de pre-softmax para evitar picos
        attn = torch.clamp(attn, -5.0, 5.0)
        
        weights = F.softmax(attn, dim=-1)
        
        # FIX: Añadir pequeña perturbación a weights para no ser perfectamente determinista
        if self.training:
            weights = weights + torch.randn_like(weights) * 0.01
            weights = weights / weights.sum(dim=-1, keepdim=True)
        
        x_clean = torch.matmul(weights, self.basis)
        
        # Conexión residual con factor de escala
        x_out = x_clean + x * 0.2  # Aumentado de 0.1 a 0.2
        
        # Regularización
        entropy = -(weights * torch.log(weights + self.eps)).sum(-1).mean()
        
        # FIX: Regularización de entropía mínima para evitar colapso (valores extremos)
        min_entropy = 0.1  # Entropía mínima requerida
        entropy_reg = torch.clamp(min_entropy - entropy, min=0.0) * 10.0  # Penalizar si entropía es muy baja
        
        ortho = torch.norm(torch.mm(self.basis, self.basis.T) - torch.eye(self.basis.size(0)), p='fro') ** 2
        
        # Regularización de varianza mínima en output
        output_variance = x_out.var(dim=0).mean()
        variance_reg = torch.clamp(0.05 - output_variance, min=0.0) * 20.0
        
        # Ajuste homeostático
        if self.homeostatic_core is not None and loss_val is not None:
            with torch.no_grad():
                w_norm = self.query.weight.norm() + self.key.weight.norm()
                physio = self.homeostatic_core(x, None, w_norm, loss_val)
                sensitivity = physio['sensitivity'].mean().item()
                x_out = x_out * (0.5 + sensitivity)
        
        return torch.clamp(x_out, -2.0, 2.0), entropy - entropy_reg, ortho + variance_reg








class MicroTopology(nn.Module):
    """Versión homeostática con plasticidad adaptativa"""
    def __init__(self, num_nodes, config: MicroConfig):
        super().__init__()
        self.num_nodes = num_nodes
        self.config = config
        
        # Pesos de conectividad
        self.adj_weights = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.normal_(self.adj_weights, std=0.1)
        
        # Máscara de grid 2x2
        self.register_buffer('adj_mask', self._create_grid_mask())
        
        # Homeostasis para plasticidad topológica
        self.homeostatic_core = HomeostaticCore(1) if config.use_homeostasis else None
        
    def _create_grid_mask(self):
        mask = torch.zeros(self.num_nodes, self.num_nodes)
        grid_size = self.config.grid_size
        for i in range(self.num_nodes):
            r, c = i // grid_size, i % grid_size
            if r > 0: mask[i, i - grid_size] = 1
            if r < grid_size - 1: mask[i, i + grid_size] = 1
            if c > 0: mask[i, i - 1] = 1
            if c < grid_size - 1: mask[i, i + 1] = 1
        return mask
    
    def get_adjacency(self, plasticity=1.0, loss_val=None):
        # Plasticidad homeostática
        if self.homeostatic_core is not None and loss_val is not None:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1)
                physio = self.homeostatic_core(dummy_input, None, self.adj_weights.norm(), loss_val)
                plasticity = plasticity * physio['base_plasticity'].mean().item()
        
        adj = torch.sigmoid(self.adj_weights * plasticity) * self.adj_mask
        deg = adj.sum(1, keepdim=True).clamp(min=1e-6)
        return adj / deg

class MicroSupConLoss(nn.Module):
    """Supervised Contrastive Loss - Mantiene estabilidad con homeostasis"""
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
# ARQUITECTURA PRINCIPAL - FIX 4: Manejo seguro de loss_val
# =============================================================================

class MicroTopoBrain(nn.Module):
    def __init__(self, config: MicroConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim
        
        # Embedding de entrada
        self.input_embed = nn.Linear(config.n_features, self.embed_dim * self.num_nodes)
        
        # Componentes con homeostasis
        self.topology = MicroTopology(self.num_nodes, config) if config.use_plasticity else None
        
        if config.use_continuum:
            self.node_processor = MicroContinuumCell(self.embed_dim, config)
        else:
            self.node_processor = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.cell_processor = None
        if config.use_mgf:
            mgf_input_dim = self.embed_dim * self.num_nodes
            if config.use_continuum:
                self.cell_processor = MicroContinuumCell(mgf_input_dim, config)
            else:
                self.cell_processor = nn.Linear(mgf_input_dim, self.embed_dim)
        
        self.symbiotic = MicroSymbioticBasis(self.embed_dim, config) if config.use_symbiotic else None
        
        self.supcon_head = None
        if config.use_supcon:
            self.supcon_head = nn.Sequential(
                nn.Linear(self.embed_dim * self.num_nodes, 8, bias=False),
                nn.ReLU(),
                nn.Linear(8, 4, bias=False)
            )
        
        # HOMEOSTASIS GLOBAL
        self.homeostatic_core = HomeostaticCore(self.embed_dim * self.num_nodes, config.homeostatic_lr) if config.use_homeostasis else None
        
        # Readout
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, config.n_classes)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, loss_val=None):
        batch_size = x.size(0)
        
        # Embedding
        x_embed = self.input_embed(x).view(batch_size, self.num_nodes, self.embed_dim)
        
        # Topología con plasticidad homeostática
        if self.topology is not None:
            # FIX: Pasar loss_val solo si está disponible y no es None
            adj = self.topology.get_adjacency(plasticity=1.0, loss_val=loss_val)
            x_agg = torch.bmm(adj.unsqueeze(0).expand(batch_size, -1, -1), x_embed)
        else:
            x_agg = x_embed
        
        # Procesamiento de nodos
        if isinstance(self.node_processor, MicroContinuumCell):
            x_flat = x_agg.view(-1, self.embed_dim)
            x_proc_flat = self.node_processor(x_flat)
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
                cell_out = self.cell_processor(cell_input)
                cell_output = cell_out.view(batch_size, self.num_nodes, self.embed_dim)
            else:
                cell_temp = self.cell_processor(cell_input)
                cell_output = cell_temp.view(batch_size, 1, self.embed_dim).expand(-1, self.num_nodes, -1)
        
        # Refinamiento simbiótico
        if self.symbiotic is not None:
            x_proc_refined = []
            for i in range(self.num_nodes):
                node_feat = x_proc[:, i, :]
                refined, ent, ort = self.symbiotic(node_feat, loss_val)
                x_proc_refined.append(refined)
            x_proc = torch.stack(x_proc_refined, dim=1)
            entropy = ent
            ortho = ort
        
        # Fusión
        combined = x_proc + cell_output
        x_flat = combined.view(batch_size, -1)
        
        # HOMEOSTASIS GLOBAL: el cerebro observa el estado completo
        if self.homeostatic_core is not None:
            with torch.no_grad():
                w_norm = self.readout.weight.norm()
                global_physio = self.homeostatic_core(x_flat, x_flat, w_norm, loss_val)
                # Aplicar controles globales (ej: escalar salida)
                sensitivity = global_physio['sensitivity'].mean().item()
                x_flat = x_flat * (0.3 + sensitivity)
        
        # Outputs
        logits = self.readout(x_flat)
        proj = self.supcon_head(x_flat) if self.supcon_head is not None else None
        
        return logits, proj, entropy, ortho
# =============================================================================
# ADVERSARIAL ATTACK
# =============================================================================

# =============================================================================
# ADVERSARIAL ATTACK - FIX 2: Remover leakage de loss_val en PGD
# =============================================================================

def micro_pgd_attack(model, x, y, eps, steps, loss_val=None):
    """
    PGD Attack - Versión ultra-simple que siempre funciona
    FIX: No pasar loss_val al modelo durante ataque para evitar leakage
    El ataque no debe tener acceso al estado interno de entrenamiento
    """
    was_training = model.training
    model.eval()
    
    delta = torch.zeros_like(x)
    with torch.no_grad():
        delta.uniform_(-eps, eps)
    
    for step in range(steps):
        x_adv = (x + delta).detach().requires_grad_(True)
        
        with torch.enable_grad():
            # FIX: Pasar loss_val=None al modelo durante ataque
            logits, _, _, _ = model(x_adv, loss_val=None)
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
# GENERADOR MATRIZ ABLACIÓN - FIX 5: Asegurar configuración limpia
# =============================================================================

def generate_ablation_matrix():
    """
    Genera matriz de ablación de 3 niveles con configuración aislada por experimento
    FIX: Asegurar que cada experimento tenga configuración independiente y limpia
    """
    components = ['plasticity', 'continuum', 'mgf', 'supcon', 'symbiotic', 'homeostasis']
    
    # NIVEL 1: Baseline + Individual
    nivel1 = {'L1_00_Baseline': {}}
    for i, comp in enumerate(components, 1):
        # FIX: Crear configuración completamente nueva para cada experimento
        config_dict = {f'use_{c}': False for c in components}  # Todos apagados
        config_dict[f'use_{comp}'] = True  # Solo encender el componente actual
        nivel1[f'L1_{i:02d}_{comp.capitalize()}'] = config_dict
    
    # NIVEL 2: Pares sinérgicos
    nivel2 = {}
    for idx, (c1, c2) in enumerate(combinations(components, 2), 1):
        config_dict = {f'use_{c}': False for c in components}  # Todos apagados
        config_dict[f'use_{c1}'] = True
        config_dict[f'use_{c2}'] = True
        nivel2[f'L2_{idx:02d}_{c1.capitalize()}+{c2.capitalize()}'] = config_dict
    
    # NIVEL 3: Sistema completo + Ablación inversa
    all_on = {f'use_{c}': True for c in components}
    nivel3 = {'L3_00_Full': all_on.copy()}
    
    for idx, comp in enumerate(components, 1):
        config_dict = all_on.copy()
        config_dict[f'use_{comp}'] = False
        nivel3[f'L3_{idx:02d}_Full_minus_{comp.capitalize()}'] = config_dict
    
    return {**nivel1, **nivel2, **nivel3}


# =============================================================================
# ENTRENAMIENTO Y EVALUACIÓN - FIX CRÍTICO: Indentación y scope de model
# =============================================================================

def train_with_cv(config: MicroConfig, dataset, cv_folds=3):
    """
    Entrenamiento con cross-validation
    FIX CRÍTICO: Métrica W2 debe usar MODELO FRESH copiado, no el mismo modelo
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.seed)
    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    
    fold_results = {
        'pgd_acc': [],
        'clean_acc': [],
        'train_time': [],
        'forgetting_w2': [],
        'homeostatic_states': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
        
        model = MicroTopoBrain(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        supcon_loss_fn = MicroSupConLoss() if config.use_supcon else None
        
        start_time = time.time()
        
        # Monitoreo de estados homeostáticos
        homeostatic_states = []
        
        # Entrenamiento normal
        for epoch in range(config.epochs):
            model.train()
            
            if config.use_homeostasis and epoch % 3 == 0:
                epoch_states = []
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(config.device), y.to(config.device)
                
                # Forward con adversarial
                x_adv = micro_pgd_attack(model, x, y, config.train_eps, config.pgd_steps)
                logits_adv, proj, entropy, ortho = model(x_adv, loss_val=None)
                
                loss = F.cross_entropy(logits_adv, y)
                if config.use_supcon and proj is not None:
                    loss += 0.3 * supcon_loss_fn(proj, y)
                loss -= 0.01 * entropy
                loss += 0.05 * ortho
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                
                if config.use_homeostasis and epoch % 3 == 0 and batch_idx == 0:
                    with torch.no_grad():
                        test_x = x[:min(4, x.size(0))]
                        test_h = model.input_embed(test_x).view(test_x.size(0), -1)
                        test_w_norm = model.readout.weight.norm()
                        if model.homeostatic_core is not None:
                            signals = model.homeostatic_core(test_x, test_h, test_w_norm, None)
                            epoch_states.append({
                                'metabolism': signals['metabolism'].mean().item(),
                                'sensitivity': signals['sensitivity'].mean().item(),
                                'gate': signals['gate'].mean().item(),
                                'plasticity': signals['base_plasticity'].mean().item()
                            })
            
            if config.use_homeostasis and epoch % 3 == 0 and epoch_states:
                avg_state = {k: np.mean([s[k] for s in epoch_states]) for k in epoch_states[0]}
                homeostatic_states.append({'epoch': epoch, 'state': avg_state})
        
        # FIX CRÍTICO: Métrica de Retención W2 con modelo COPIADO
        forgetting_w2 = 0.0
        
        # Solo probar si hay suficientes muestras de clase 2
        train_01_idx = [i for i in train_idx if labels[i] in [0, 1]]
        class_2_idx_val = [i for i in val_idx if labels[i] == 2]
        
        if len(train_01_idx) >= 20 and len(class_2_idx_val) > 10:
            # Step 1: Evaluar accuracy en clase 2 con modelo entrenado
            model.eval()
            class_2_subset = Subset(dataset, class_2_idx_val)
            class_2_loader = DataLoader(class_2_subset, batch_size=config.batch_size, shuffle=False)
            
            correct_before = 0
            total_2 = 0
            
            with torch.no_grad():
                for x, y in class_2_loader:
                    x, y = x.to(config.device), y.to(config.device)
                    logits = model(x)[0]
                    pred = logits.argmax(dim=1)
                    correct_before += pred.eq(y).sum().item()
                    total_2 += y.size(0)
            
            acc_before = correct_before / total_2 if total_2 > 0 else 0.0
            
            # FIX CRÍTICO: Step 2 - Crear MODELO FRESH copiado para fine-tune
            # Esto evita que el fine-tune contamine el modelo original
            model_forget = MicroTopoBrain(config).to(config.device)
            model_forget.load_state_dict(model.state_dict())  # Copiar pesos
            
            # Step 3: Fine-tune SOLO en clases 0-1
            train_01_subset = Subset(dataset, train_01_idx)
            train_01_loader = DataLoader(train_01_subset, batch_size=config.batch_size, shuffle=True)
            
            optimizer_forget = torch.optim.AdamW(model_forget.parameters(), lr=config.lr * 0.1, weight_decay=1e-5)
            
            model_forget.train()
            for epoch in range(5):  # 5 épocas de fine-tune
                for x, y in train_01_loader:
                    x, y = x.to(config.device), y.to(config.device)
                    logits, _, _, _ = model_forget(x, loss_val=None)
                    loss = F.cross_entropy(logits, y)
                    optimizer_forget.zero_grad()
                    loss.backward()
                    optimizer_forget.step()
            
            # Step 4: Evaluar accuracy en clase 2 DESPUÉS del fine-tune
            model_forget.eval()
            correct_after = 0
            
            with torch.no_grad():
                for x, y in class_2_loader:
                    x, y = x.to(config.device), y.to(config.device)
                    logits = model_forget(x)[0]
                    pred = logits.argmax(dim=1)
                    correct_after += pred.eq(y).sum().item()
            
            acc_after = correct_after / total_2 if total_2 > 0 else 0.0
            
            # Forgetting = caída en accuracy (negativo indica olvido, positivo es learning adicional)
            forgetting_w2 = (acc_after - acc_before) * 100
            
            # Logging para debug
            print(f"   🔍 Forgetting W2 Debug: Before={acc_before:.3f}, After={acc_after:.3f}, Drop={forgetting_w2:.1f}%")
        
        # Evaluación final normal
        model.eval()
        
        pgd_correct, clean_correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                
                # Clean accuracy
                logits_clean = model(x)[0]
                pred_clean = logits_clean.argmax(dim=1)
                clean_correct += pred_clean.eq(y).sum().item()
                
                # PGD accuracy (ataque más fuerte con eps=0.25 y 5 steps)
                x_adv = micro_pgd_attack(model, x, y, config.test_eps, config.pgd_steps)
                logits_adv = model(x_adv)[0]
                pred_adv = logits_adv.argmax(dim=1)
                pgd_correct += pred_adv.eq(y).sum().item()
                
                total += y.size(0)
        
        fold_results['pgd_acc'].append(100.0 * pgd_correct / total)
        fold_results['clean_acc'].append(100.0 * clean_correct / total)
        fold_results['train_time'].append(time.time() - start_time)
        fold_results['forgetting_w2'].append(forgetting_w2)
        fold_results['homeostatic_states'].append(homeostatic_states)
    
    return {
        'pgd_mean': np.mean(fold_results['pgd_acc']),
        'pgd_std': np.std(fold_results['pgd_acc']),
        'clean_mean': np.mean(fold_results['clean_acc']),
        'clean_std': np.std(fold_results['clean_acc']),
        'forgetting_w2': np.mean(fold_results['forgetting_w2']),
        'train_time': np.mean(fold_results['train_time']),
        'homeostatic_states': fold_results['homeostatic_states']
    }


# =============================================================================
# EJECUTOR PRINCIPAL - FIX 6: Ejecutar solo experimentos críticos primero
# =============================================================================

def run_ablation_study():
    """Ejecuta estudio con validación de integridad de resultados"""
    import sys
    quick_test = '--quick' in sys.argv
    
    seed_everything(42)
    base_config = MicroConfig()
    dataset = get_dataset(base_config)
    
    results_dir = Path("neurologos_v52_results")
    results_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🧠 NeuroLogos v5.2 - TopoBrain + Homeostatic Core")
    print("="*80)
    print(f"📊 Métricas: PGD Acc, Clean Acc, Retención W2, Training Time, #Params")
    print(f"⚙  Arquitectura: Grid 2x2 con Regulación Fisiológica Interna")
    print(f"🔬 Validación: 3-Fold Stratified CV + Métricas de Olvido")
    if quick_test:
        print(f"🚀 MODO QUICK TEST: Solo 5 experimentos críticos")
    print(f"⚠️  Ataque PGD: eps={base_config.train_eps}, steps={base_config.pgd_steps}")
    print("="*80 + "\n")
    
    ablation_matrix = generate_ablation_matrix()
    
    # Modo quick para pruebas
    if quick_test:
        critical_experiments = {
            'L1_00_Baseline': ablation_matrix['L1_00_Baseline'],
            'L1_05_Symbiotic': ablation_matrix['L1_05_Symbiotic'],
            'L1_06_Homeostasis': ablation_matrix['L1_06_Homeostasis'],
            'L2_09_Continuum+Homeostasis': ablation_matrix['L2_09_Continuum+Homeostasis'],
            'L3_06_Full_minus_Homeostasis': ablation_matrix['L3_06_Full_minus_Homeostasis']
        }
        ablation_matrix = critical_experiments
    
    print(f"📋 Total de experimentos: {len(ablation_matrix)}\n")
    
    results = {}
    previous_results = {}
    
    # FIX: Validar que dataset tiene muestras suficientes de clase 2
    class_counts = np.bincount(labels)
    print(f"📊 Distribución de clases: {class_counts}")
    if len(class_counts) < 3 or class_counts[2] < 30:
        print("⚠️  WARNING: Pocas muestras de clase 2. Métrica W2 puede ser inestable.")
    
    for exp_name, overrides in ablation_matrix.items():
        print(f"▶ {exp_name}")
        
        active_components = [k.replace('use_', '') for k, v in overrides.items() if v]
        print(f"   🔧 Componentes activos: {active_components if active_components else 'None (Baseline)'}")
        
        cfg_dict = base_config.__dict__.copy()
        cfg_dict.update(overrides)
        config = MicroConfig(**cfg_dict)
        
        # FIX: Imprimir configuración de ataque para debug
        print(f"   ⚔️  PGD Config: eps={config.train_eps}, steps={config.pgd_steps}")
        
        metrics = train_with_cv(config, dataset)
        
        # FIX: Validar que homeostasis produce señales variadas
        if config.use_homeostasis and metrics['homeostatic_states']:
            final_state = metrics['homeostatic_states'][-1][-1]['state'] if metrics['homeostatic_states'][-1] else {}
            if final_state:
                state_variance = np.var([final_state[k] for k in ['metabolism', 'sensitivity', 'gate', 'plasticity']])
                if state_variance < 0.01:
                    print(f"   ⚠️  WARNING: Señales homeostáticas casi constantes (var={state_variance:.4f})")
        
        # FIX: Validar que PGD es efectivo (PGD < Clean - 5%)
        if metrics['pgd_mean'] > metrics['clean_mean'] - 2.0:
            print(f"   ⚠️  WARNING: Ataque PGD puede ser débil (PGD={metrics['pgd_mean']:.1f}%, Clean={metrics['clean_mean']:.1f}%)")
        
        model_temp = MicroTopoBrain(config)
        metrics['n_params'] = model_temp.count_parameters()
        
        result_signature = (round(metrics['pgd_mean'], 2), round(metrics['clean_mean'], 2), round(metrics['forgetting_w2'], 1))
        if result_signature in previous_results:
            print(f"   ⚠️  WARNING: Resultados idénticos a {previous_results[result_signature]}")
        previous_results[result_signature] = exp_name
        
        results[exp_name] = metrics
        
        print(f"   📈 PGD: {metrics['pgd_mean']:>6.2f}±{metrics['pgd_std']:>4.2f}% | "
              f"🎯 Clean: {metrics['clean_mean']:>6.2f}±{metrics['clean_std']:>4.2f}% | "
              f"🧠 Retención: {metrics['forgetting_w2']:>+6.1f}% | "
              f"⚡ Params: {metrics['n_params']:>8,} | "
              f"⏱️  Time: {metrics['train_time']:>4.1f}s\n")
    
    # Guardar resultados
    suffix = '_quick' if quick_test else '_final_v2'
    with open(results_dir / f"ablation_v52_homeostasis{suffix}.json", 'w') as f:
        final_output = {
            'metadata': {
                'components': ['plasticity', 'continuum', 'mgf', 'supcon', 'symbiotic', 'homeostasis'],
                'config': base_config.__dict__,
                'total_experiments': len(ablation_matrix),
                'quick_mode': quick_test,
                'attack_config': {'eps': base_config.train_eps, 'steps': base_config.pgd_steps},
                'dataset_distribution': class_counts.tolist()
            },
            'results': results
        }
        json.dump(final_output, f, indent=2)
        print(f"💾 Resultados guardados en: {results_dir}/ablation_v52_homeostasis{suffix}.json")
    
    # Reporte final
    print("\n" + "="*80)
    print("📊 RESULTADOS FINALES - NEUROLOGOS V5.2")
    print("="*80)
    print(f"{'Experimento':<45} {'PGD':<12} {'Clean':<12} {'Retención':<10} {'Params':<8}")
    print("-"*80)
    
    for name, res in results.items():
        print(f"{name:<45} "
              f"{res['pgd_mean']:>6.2f}±{res['pgd_std']:>4.2f}% "
              f"{res['clean_mean']:>6.2f}±{res['clean_std']:>4.2f}% "
              f"{res['forgetting_w2']:>+8.1f}% "
              f"{res['n_params']:>8,}")
    
    # Análisis de sinergias
    print("\n" + "="*80)
    print("🔬 ANÁLISIS DE SINERGIAS")
    print("="*80)
    
    if 'L1_00_Baseline' in results and 'L1_06_Homeostasis' in results:
        baseline_pgd = results['L1_00_Baseline']['pgd_mean']
        homeostasis_pgd = results['L1_06_Homeostasis']['pgd_mean']
        
        print(f"📊 Baseline PGD: {baseline_pgd:.2f}%")
        print(f"🧬 Homeostasis sola PGD: {homeostasis_pgd:.2f}% (+{homeostasis_pgd-baseline_pgd:.2f}%)")
        print(f"💡 Sinergia esperada: Combinaciones con H > H sola o cercanas")
        print("-"*80)
        
        synergy_results = []
        for name, res in results.items():
            if 'Homeostasis' in name and name != 'L1_06_Homeostasis' and 'Full_minus' not in name:
                if 'L2_' in name:  # Solo pares
                    other_comp = name.replace('Homeostasis', '').replace('L2_', '').replace('_', '').strip('+')
                    synergy = res['pgd_mean'] - homeostasis_pgd
                    synergy_results.append((name, res['pgd_mean'], synergy, res['clean_mean']))
        
        synergy_results.sort(key=lambda x: x[2], reverse=True)
        for i, (name, pgd, syn, clean) in enumerate(synergy_results[:5], 1):
            print(f"{i}. {name}: PGD {pgd:.2f}% (+{syn:+.2f}%) | Clean {clean:.2f}%")
    
    # Top rankings
    for metric, label in [('pgd_mean', 'PGD ROBUSTEZ'), ('clean_mean', 'CLEAN ACCURACY'), 
                          ('forgetting_w2', 'RETENCIÓN W2 (menor es mejor)')]:
        print(f"\n" + "="*80)
        print(f"🏆 TOP 5 - {label}")
        print("="*80)
        reverse = not ('forgetting' in metric)
        sorted_res = sorted(results.items(), key=lambda x: x[1][metric], reverse=reverse)[:5]
        for i, (name, res) in enumerate(sorted_res, 1):
            print(f"{i}. {name}: {res[metric]:.2f}%")
    
    return results

if __name__ == "__main__":
    results = run_ablation_study()
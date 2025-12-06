%%writefile neurosoberano.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time  # Añadido para timing

def measure_spatial_richness(activations):
    """
    FIX: Caché de cálculo + aproximación por muestreo para batches grandes + fallback robusto
    """
    if activations.size(0) < 2:
        return torch.tensor(0.0), 0.0
    
    # FIX: Caché para evitar recomputación en múltiples llamadas del mismo batch
    if not hasattr(measure_spatial_richness, '_cache'):
        measure_spatial_richness._cache = {}
    
    cache_key = f"{activations.shape}_{activations.sum().item():.6f}"
    if cache_key in measure_spatial_richness._cache:
        return measure_spatial_richness._cache[cache_key]
    
    # FIX: Aproximación por muestreo si batch es muy grande (>>1000)
    if activations.size(0) > 1000:
        indices = torch.randperm(activations.size(0))[:1000]
        activations = activations[indices]
    
    try:
        # FIX: Normalización robusta con pequeño ruido para estabilidad numérica
        A_centered = activations - activations.mean(dim=0, keepdim=True)
        A_centered += torch.randn_like(A_centered) * 1e-8
        
        cov = A_centered.T @ A_centered / (activations.size(0) - 1)
        eigs = torch.linalg.eigvalsh(cov).abs()
        
        # FIX: Umbral dinámico para eigenvalores cercanos a cero
        eigen_threshold = torch.quantile(eigs, 0.05)
        eigs = eigs[eigs > eigen_threshold]
        
        if eigs.sum() < 1e-12:
            result = (torch.tensor(0.0), 1.0)
            measure_spatial_richness._cache[cache_key] = result
            return result
        
        p = eigs / (eigs.sum() + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        
        # FIX: Penalización de entropía si dimensionalidad efectiva es baja
        effective_dim = (p > 0.01).sum().float()
        penalty = 1.0 - effective_dim / (len(p) + 1e-6)
        adjusted_entropy = entropy * (1.0 - penalty * 0.5)
        
        result = (adjusted_entropy, torch.exp(adjusted_entropy).item())
        measure_spatial_richness._cache[cache_key] = result
        return result
    
    except Exception as e:
        # FIX: Fallback con métrica proxy si falla eigenvaluación
        proxy_richness = activations.std(dim=0).mean().item()
        result = (torch.tensor(proxy_richness), proxy_richness)
        measure_spatial_richness._cache[cache_key] = result
        return result

class HomeostasisEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 1.5
        self.register_buffer('running_richness', torch.tensor(0.0))
        self.register_buffer('running_fastnorm', torch.tensor(0.0))
        self.register_buffer('running_loss', torch.tensor(0.0))
        self.momentum = 0.99
        
        # FIX: Targets adaptativos basados en capacidad del modelo (no constantes)
        self.register_buffer('target_richness_base', torch.tensor(100.0))
        self.register_buffer('target_entropy_base', torch.tensor(2.2))
    
    def decide(self, task_loss_val, richness_val, vn_entropy_val, fast_norm_val=0.0):
        """
        FIX: Drives escalados adaptativamente según historial + suavizado temporal de targets
        """
        # Actualización de estadísticas con momentum
        self.running_richness = self.momentum * self.running_richness + (1 - self.momentum) * richness_val
        self.running_fastnorm = self.momentum * self.running_fastnorm + (1 - self.momentum) * fast_norm_val
        self.running_loss = self.momentum * self.running_loss + (1 - self.momentum) * task_loss_val
        
        # FIX: Target de riqueza adaptativo según error actual
        loss_factor = min(3.0, 1.0 + self.running_loss.item() * 2.0)
        target_richness = min(150.0, self.running_richness.item() * loss_factor + self.target_richness_base.item())
        
        # FIX: Focus drive adaptativo - escala con error relativo al historial
        relative_loss = task_loss_val / (self.running_loss.item() + 1e-6)
        focus_base = torch.log1p(torch.tensor(task_loss_val) * 50.0).item()
        focus_drive = focus_base * (2.0 + min(2.0, relative_loss))
        
        # FIX: Explore drive con aceleración dinámica si riqueza es baja
        richness_deficit = max(0.0, (target_richness - richness_val) / (target_richness + 1e-6))
        explore_drive = richness_deficit * 0.5 * (1.0 + (1.0 - richness_deficit))
        
        # FIX: Repair drive con sensibilidad adaptativa a norma actual
        fast_norm_threshold = 15.0 + self.running_fastnorm.item() * 0.5
        fast_norm_penalty = max(0.0, (fast_norm_val - fast_norm_threshold) * 1.5)
        
        # FIX: Target entropy adaptativo basado en estabilidad
        stability_factor = min(1.0, fast_norm_val / 50.0)
        target_entropy = self.target_entropy_base.item() + (richness_val / 120.0) * (1.0 - stability_factor)
        entropy_repair = max(0.0, (target_entropy - vn_entropy_val) * 6.0)
        
        repair_drive = fast_norm_penalty + entropy_repair
        
        # FIX: Normalización softmax con temperatura dinámica
        logits = torch.tensor([focus_drive, explore_drive, repair_drive])
        temperature_adj = max(1.0, self.temperature * (1.0 + richness_deficit))
        probs = F.softmax(logits / temperature_adj, dim=0)
        
        return probs[0].item(), probs[1].item(), probs[2].item()




class LiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Pesos Estructurales (Lentos) - Inicialización Ortogonal
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        
        # Pesos Plásticos (Rápidos/Hebbianos)
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.ln = nn.LayerNorm(out_dim)
        
        # Controlador de Plasticidad Neurona-Específica (Meta-Learning)
        self.plasticity_controller = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.plasticity_controller[2].bias.data.fill_(-4.0)
        
        # CALIBRACIÓN: Learning Rate reducido para evitar oscilación catastrófica
        self.base_lr = 0.005 
        # Para plasticidad supervisada
        self.prediction_error = 0.0

    def forward(self, x, global_plasticity=0.0):
        slow_out = self.W_slow(x)
        fast_out = F.linear(x, self.W_fast)
        pre_act = slow_out + fast_out
        
        # Extracción de estadísticas locales para el controlador
        batch_mean = pre_act.mean(dim=0).unsqueeze(1)
        batch_std = pre_act.std(dim=0).unsqueeze(1) + 1e-6
        stats = torch.cat([batch_mean, batch_std], dim=1)
        
        # Cálculo de plasticidad local modulada por error de predicción
        learned_plasticity = self.plasticity_controller(stats).squeeze()
        effective_plasticity = global_plasticity * learned_plasticity * (1.0 - self.prediction_error)
        
        # Activación con Clipping Suave
        out = 5.0 * torch.tanh(self.ln(pre_act) / 5.0)

        # Regla de Aprendizaje Hebbiano con FIXES
        if self.training and effective_plasticity.mean() > 0.001:
            with torch.no_grad():
                # Normalización por energía de entrada
                x_norm = (x ** 2).sum(1).mean() + 1e-6
                
                # Correlación (y * x)
                correlation = torch.mm(out.T, x) / x.size(0)
                
                # FIX 1: Decaimiento pasivo REDUCIDO (era 0.2, ahora 0.05)
                forgetting = 0.05 * self.W_fast
                
                # FIX 2: Clipping MENOS agresivo (era 0.05, ahora 0.1)
                delta = torch.clamp((correlation / x_norm) - forgetting, -0.1, 0.1)
                
                # Aplicación modulada por el controlador aprendido
                lr_vector = effective_plasticity.unsqueeze(1) * self.base_lr
                self.W_fast.data += delta * lr_vector
                
                # FIX 3: Decaimiento multiplicativo MÁS suave (era 0.999, ahora 0.995)
                self.W_fast.data.mul_(0.995)

        return out

        


# =============================================================================
# COMPONENTE NUEVO: Sistema Consciente (Síntesis sobre TopoBrain)
# =============================================================================

class ConsciousSystem(nn.Module):
    """
    Sistema de control ejecutivo que opera sobre representaciones
    del sistema inconsciente. Implementa homeostasis y memoria de trabajo.
    """
    def __init__(self, unconscious_dim, d_hid, d_out):
        super().__init__()
        
        self.homeostasis = HomeostasisEngine()
        
        # Mecanismo de atención consciente (focalización selectiva)
        self.gaze = nn.Sequential(
            nn.Linear(unconscious_dim, unconscious_dim),
            nn.Sigmoid()
        )
        self.gaze[0].bias.data.fill_(-2.0)  # Atención inicial baja para evitar ruido en features no entrenadas
        
        # Memoria de trabajo (working memory con fast weights)
        self.working_memory = LiquidNeuron(unconscious_dim, d_hid)
        
        # Capa de decisión final
        self.decision = nn.Linear(d_hid, d_out)
        
        # Proyección para análisis de riqueza
        self.richness_proj = nn.Linear(d_hid, d_hid)
        
        self.vn_entropy = 3.0
    
    def forward(self, unconscious_features, plasticity_gate=1.0):
        """
        Input: Representaciones del sistema inconsciente [batch, unconscious_dim]
        Output: logits, métricas homeostáticas
        """
        # Atención consciente: focalizar en features relevantes
        attention_mask = self.gaze(unconscious_features)
        focused_features = unconscious_features * attention_mask
        
        # Procesamiento en memoria de trabajo
        working_repr = F.relu(self.working_memory(focused_features, plasticity_gate))
        
        # Decisión consciente
        logits = self.decision(working_repr)
        
        # Análisis de riqueza representacional
        rich_proj = self.richness_proj(working_repr)
        rich_tensor, rich_val = measure_spatial_richness(rich_proj)
        
        return logits, rich_tensor, rich_val, attention_mask.mean()
    
    def get_structure_entropy(self):
        """Análisis de salud estructural mediante SVD"""
        with torch.no_grad():
            W = self.decision.weight
            S = torch.linalg.svdvals(W)
            p = S**2 / (S.pow(2).sum() + 1e-12)
            self.vn_entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
            return self.vn_entropy


# =============================================================================
# TOPOBRAIN SIMPLIFICADO: Versión NestedTopoBrain para POC
# =============================================================================

class NestedTopoLayer(nn.Module):
    """
    FIX: Topología con regularización de entropía + normalización espectral + estabilización de adyacencia
    """
    def __init__(self, in_dim, hid_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_mapper = LiquidNeuron(in_dim, hid_dim)
        
        # FIX: Adyacencia con regularización de entropía para evitar conexiones triviales
        self.adj_logits = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.15)
        self.register_buffer('adj_mask', torch.ones(num_nodes, num_nodes))
        
        # FIX: Normalización de batch para estabilidad en propagación de grafos
        self.graph_norm = nn.LayerNorm(hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
        
        # FIX: Penalización de topología acumulada (para backward)
        self.register_buffer('topology_reg_loss', torch.tensor(0.0))
    
    def forward(self, x_nodes, plasticity_gate=1.0):
        """
        x_nodes: [batch, num_nodes, in_dim]
        output: [batch, num_nodes, hid_dim]
        """
        batch_size = x_nodes.size(0)
        
        x_flat = x_nodes.reshape(batch_size * self.num_nodes, -1)
        h_flat = self.node_mapper(x_flat, plasticity_gate)
        h = h_flat.reshape(batch_size, self.num_nodes, -1)
        
        # FIX: Adyacencia con mask diagonal cero (no auto-conexiones)
        adj_weights = torch.sigmoid(self.adj_logits)
        adj_weights = adj_weights * (1 - torch.eye(self.num_nodes, device=x_nodes.device))
        
        # FIX: Normalización espectral para estabilidad
        adj_norm = adj_weights / (adj_weights.sum(1, keepdim=True) + 1e-6)
        eigenvals = torch.linalg.eigvals(adj_norm).abs()
        spectral_radius = eigenvals.max()
        
        # FIX: Escalar si radio espectral > 1 (evitar explosión)
        if spectral_radius > 1.0:
            adj_norm = adj_norm / (spectral_radius + 1e-6)
        
        # Mensaje passing con estabilidad
        h_aggregated = torch.matmul(adj_norm, h)
        h_aggregated = self.graph_norm(h_aggregated)
        
        # FIX: Conexión residual con gate adaptativo
        combined = h + h_aggregated * plasticity_gate
        
        return self.norm(combined)
    
    def get_topology_density(self):
        """Densidad de conexiones topológicas"""
        with torch.no_grad():
            adj_weights = torch.sigmoid(self.adj_logits)
            # FIX: Excluir auto-conexiones de densidad
            adj_no_self = adj_weights * (1 - torch.eye(self.num_nodes, device=adj_weights.device))
            density = (adj_no_self > 0.5).float().mean().item()
            return density
    
    def get_topology_reg_loss(self):
        """
        FIX: Penalización de entropía de adyacencia para evitar trivialidad
        """
        adj_weights = torch.sigmoid(self.adj_logits)
        p_adj = adj_weights / (adj_weights.sum() + 1e-12)
        entropy_adj = -torch.sum(p_adj * torch.log(p_adj + 1e-12))
        
        # FIX: Promover entropía media (no trivial ni aleatoria)
        target_entropy = np.log(self.num_nodes * (self.num_nodes - 1)) * 0.5
        reg_loss = (entropy_adj - target_entropy).pow(2) * 0.01
        
        return reg_loss

class UnconsciousSystem(nn.Module):
    """
    Sistema inconsciente: Procesamiento automático y paralelo.
    Arquitectura simplificada de TopoBrain para extracción de features.
    CAPACIDAD AUMENTADA: grid_size=6, hidden_dim=96 para mejor representación.
    """
    def __init__(self, in_channels, grid_size, hidden_dim):
        super().__init__()
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        
        self.early_visual = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size))
        )
        
        # Capacidad aumentada: hidden_dim=96, segunda capa 96*2=192
        self.layer1 = NestedTopoLayer(256, hidden_dim, self.num_nodes)
        self.layer2 = NestedTopoLayer(hidden_dim, hidden_dim * 2, self.num_nodes)
        
        self.output_dim = hidden_dim * 2 * self.num_nodes
    
    def forward(self, x, plasticity_gate=1.0):
        """
        x: [batch, 3, 32, 32]
        output: [batch, output_dim] representaciones inconscientes
        """
        x_visual = self.early_visual(x)
        x_nodes = x_visual.flatten(2).transpose(1, 2)
        
        # Procesamiento jerárquico con topología
        h1 = self.layer1(x_nodes, plasticity_gate)
        h2 = self.layer2(F.gelu(h1), plasticity_gate)
        
        features = h2.reshape(x.size(0), -1)
        return features
    
    def get_topology_stats(self):
        """Estadísticas de topología del sistema inconsciente"""
        density1 = self.layer1.get_topology_density()
        density2 = self.layer2.get_topology_density()
        return {
            'layer1_density': density1,
            'layer2_density': density2,
            'avg_density': (density1 + density2) / 2
        }

# =============================================================================
# ARQUITECTURA DUAL COMPLETA
# =============================================================================

class DualMind(nn.Module):
    """
    Sistema dual de procesamiento:
    - Inconsciente: Procesamiento automático, paralelo, topológico
    - Consciente: Decisión deliberada, homeostática, serial
    """
    def __init__(self, in_channels=3, grid_size=4, hidden_dim=64, conscious_dim=128, num_classes=10):
        super().__init__()
        
        # Sistema 1: Procesamiento inconsciente (automático)
        self.unconscious = UnconsciousSystem(
            in_channels=in_channels,
            grid_size=grid_size,        # Incrementado de 4 a 8 para mayor capacidad (opcional GPU)
            hidden_dim=hidden_dim
        )
        
        # Sistema 2: Procesamiento consciente (deliberado)
        self.conscious = ConsciousSystem(
            unconscious_dim=self.unconscious.output_dim,
            d_hid=conscious_dim,
            d_out=num_classes
        )
        
        self.mode = 'dual'  # 'unconscious', 'conscious', 'dual'
    
    def forward(self, x, mode=None):
        """
        Modos de operación:
        - 'unconscious': Solo sistema inconsciente (rápido, baseline)
        - 'conscious': Consciente sobre inconsciente (lento, preciso)
        - 'dual': Ambos con retroalimentación (modo completo)
        """
        if mode is None:
            mode = self.mode
        
        # FASE 1: Procesamiento inconsciente (siempre activo)
        unconscious_features = self.unconscious(x, plasticity_gate=1.0)
        
        if mode == 'unconscious':
            # Clasificación directa (baseline, no óptimo)
            logits = torch.randn(x.size(0), 10, device=x.device)  # Placeholder
            return logits, None, None, None
        
        # FASE 2: Procesamiento consciente
        conscious_logits, rich_tensor, rich_val, gaze_width = self.conscious(
            unconscious_features,
            plasticity_gate=1.0
        )
        
        return conscious_logits, rich_tensor, rich_val, gaze_width
    
    def get_system_status(self):
        """Diagnóstico completo del sistema dual"""
        topo_stats = self.unconscious.get_topology_stats()
        struct_entropy = self.conscious.get_structure_entropy()
        
        return {
            'unconscious': topo_stats,
            'conscious': {'structure_entropy': struct_entropy}
        }


# =============================================================================
# TRAINING LOOP: Entrenamiento en 3 Fases con Timing y Métricas Internas
# =============================================================================

def train_dualmind_phase1(model, train_loader, optimizer, device, epochs=5):
    """
    FASE 1: Preentrenamiento del sistema inconsciente
    Objetivo: Aprender representaciones topológicas ricas
    Métricas: Loss, Accuracy, Topology Density, GradNorm, Tiempo por Batch
    """
    print("\n" + "="*60)
    print("FASE 1: PREENTRENAMIENTO INCONSCIENTE")
    print("="*60)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    temp_classifier = nn.Linear(model.unconscious.output_dim, 10).to(device)
    temp_optimizer = optim.Adam(temp_classifier.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        epoch_start = time.perf_counter()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            batch_start = time.perf_counter()
            
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            temp_optimizer.zero_grad()
            
            unconscious_features = model.unconscious(x, plasticity_gate=1.0)
            logits = temp_classifier(unconscious_features)
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Calcular norma de gradiente para diagnóstico
            grad_norm = torch.nn.utils.clip_grad_norm_(model.unconscious.parameters(), 1.0)
            
            optimizer.step()
            temp_optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            
            batch_time = time.perf_counter() - batch_start
        
        epoch_time = time.perf_counter() - epoch_start
        acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        topo_stats = model.unconscious.get_topology_stats()
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | "
              f"Topo:{topo_stats['avg_density']:.3f} | GradNorm:{grad_norm:.4f} | "
              f"Time:{epoch_time:.2f}s ({batch_time*1000:.1f}ms/batch)")
    print("✅ Fase 1 completada\n")


def train_dualmind_phase2(model, train_loader, optimizer, device, epochs=10):
    """
    FASE 2: Entrenamiento del sistema consciente
    Objetivo: Aprender decisiones homeostáticas óptimas
    Sistema inconsciente CONGELADO
    Métricas: Loss, Accuracy, Richness, Drives (F/E/R), Entropía, FastNorm, Tiempo
    FIX: Logging adicional de focus_base para validar reescalado de HomeostasisEngine
    """
    print("="*60)
    print("FASE 2: ENTRENAMIENTO CONSCIENTE")
    print("="*60)
    
    # Congelar sistema inconsciente
    model.unconscious.eval()
    for param in model.unconscious.parameters():
        param.requires_grad = False
    
    model.conscious.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        metrics = {'focus': 0, 'explore': 0, 'repair': 0, 'richness': 0}
        epoch_start = time.perf_counter()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            batch_start = time.perf_counter()
            
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                unconscious_features = model.unconscious(x, plasticity_gate=0.0)
            
            conscious_logits, rich_tensor, rich_val, gaze_width = model.conscious(
                unconscious_features, plasticity_gate=1.0
            )
            
            task_loss = criterion(conscious_logits, y)
            
            struct_entropy = model.conscious.get_structure_entropy()
            fast_norm = model.conscious.working_memory.W_fast.data.norm().item()
            
            p_focus, p_explore, p_repair = model.conscious.homeostasis.decide(
                task_loss.item(), rich_val, struct_entropy, fast_norm
            )
            
            weighted_task = task_loss * p_focus
            # FIX: Incentivo de exploración aumentado a 0.70 para acelerar aprendizaje de riqueza
            weighted_curiosity = -rich_tensor * 0.70 * p_explore
            
            total_loss_batch = weighted_task + weighted_curiosity
            total_loss_batch.backward()
            
            # Métricas internas críticas
            grad_norm = torch.nn.utils.clip_grad_norm_(model.conscious.parameters(), 1.0)
            
            optimizer.step()
            
            if p_repair > 0.4:
                model.conscious.working_memory.consolidate_svd(p_repair)
            
            total_loss += task_loss.item()
            pred = conscious_logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            
            metrics['focus'] += p_focus
            metrics['explore'] += p_explore
            metrics['repair'] += p_repair
            metrics['richness'] += rich_val
            
            batch_time = time.perf_counter() - batch_start
        
        epoch_time = time.perf_counter() - epoch_start
        acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        n = len(train_loader)
        avg_rich = metrics['richness'] / n
        
        # FIX: Logging de fast_norm para monitorear plasticidad hebbiana
        # Valores bajos (~0.001) son esperados si working_memory no requiere consolidación intensiva
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | "
              f"Rich:{avg_rich:.1f} | F:{metrics['focus']/n:.2f} E:{metrics['explore']/n:.2f} R:{metrics['repair']/n:.2f} | "
              f"Ent:{struct_entropy:.3f} | FastNorm:{fast_norm:.3f} | Time:{epoch_time:.2f}s")
    
    print("✅ Fase 2 completada\n")
    
    for param in model.unconscious.parameters():
        param.requires_grad = True


def train_dualmind_phase3(model, train_loader, optimizer, device, epochs=15):
    """
    FASE 3: Co-adaptación con plasticidad DINÁMICA en tiempo real y monitoreo de emergencia
    FIX: Plasticidad por batch, no epoch + penalización de topología + diagnóstico de moda colapsada
    """
    print("="*60)
    print("FASE 3: CO-ADAPTACIÓN DUAL (PLASTICIDAD EN TIEMPO REAL)")
    print("="*60)
    
    model.unconscious.eval()
    for param in model.unconscious.parameters():
        param.requires_grad = False
    
    model.conscious.train()
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.008, epochs=epochs, steps_per_epoch=len(train_loader))
    
    # FIX: Buffers para detección de colapso de moda
    running_richness_window = []
    emergency_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        metrics = {'focus': 0, 'explore': 0, 'repair': 0, 'richness': 0, 'consolidations': 0}
        
        epoch_start = time.perf_counter()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # FIX: Plasticidad adaptativa por batch basada en error delta
            with torch.no_grad():
                prev_features = model.unconscious(x, plasticity_gate=0.0)
            
            # Forward con plasticidad
            unconscious_plasticity = min(1.0, 0.3 + (batch_idx / len(train_loader)) * 0.4)
            unconscious_features = model.unconscious(x, plasticity_gate=unconscious_plasticity)
            
            conscious_plasticity = max(0.3, 1.0 - (batch_idx / len(train_loader)) * 0.3)
            conscious_logits, rich_tensor, rich_val, gaze_width = model.conscious(
                unconscious_features, plasticity_gate=conscious_plasticity
            )
            
            task_loss = criterion(conscious_logits, y)
            struct_entropy = model.conscious.get_structure_entropy()
            fast_norm = model.conscious.working_memory.W_fast.data.norm().item()
            
            # FIX: Delta de error para modulación de plasticidad
            feature_delta = (unconscious_features - prev_features).norm().item()
            p_focus, p_explore, p_repair = model.conscious.homeostasis.decide(
                task_loss.item(), rich_val, struct_entropy, fast_norm
            )
            
            # FIX: Ajustar drives por delta de features (más plasticidad si cambia mucho)
            adaptive_explore = p_explore * (1.0 + min(1.0, feature_delta / 10.0))
            
            conscious_loss = (task_loss * p_focus) - (rich_tensor * 0.15 * adaptive_explore)
            
            # FIX: Regularización de topología del inconsciente (aunque congelado)
            topo_reg = model.unconscious.layer2.get_topology_reg_loss() + model.unconscious.layer1.get_topology_reg_loss()
            
            # Regularización de fast weights
            fast_reg = 0.0
            if fast_norm > 12.0:
                fast_reg = 0.003 * (fast_norm - 12.0) ** 2
            
            total_loss_batch = conscious_loss + fast_reg + topo_reg
            total_loss_batch.backward()
            
            # FIX: Gradient clipping por capa para estabilidad
            if conscious_plasticity > 0.5:
                torch.nn.utils.clip_grad_norm_(model.conscious.parameters(), 0.3)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            scheduler.step()
            
            # FIX: Consolidación con umbral adaptativo y cooldown
            if p_repair > 0.35 or fast_norm > 25.0:
                consolidated = model.conscious.working_memory.consolidate_svd(p_repair)
                if consolidated:
                    metrics['consolidations'] += 1
                    # Cooldown después de consolidación
                    optimizer.param_groups[0]['lr'] *= 0.95
            
            # FIX: Detección de emergencia por colapso de riqueza
            running_richness_window.append(rich_val)
            if len(running_richness_window) > 100:
                running_richness_window.pop(0)
                richness_trend = np.polyfit(range(len(running_richness_window)), running_richness_window, 1)[0]
                
                if richness_trend < -0.5 and rich_val < 20:
                    print(f"  🚨 EMERGENCY: Colapso de riqueza detectado (trend={richness_trend:.3f})")
                    model.conscious.working_memory.consolidate_svd(repair_strength=0.9)
                    metrics['consolidations'] += 1
                    emergency_counter += 1
                    
                    # Reset de memoria si colapsa repetidamente
                    if emergency_counter > 3:
                        model.conscious.working_memory.W_fast.data.zero_()
                        emergency_counter = 0
            
            total_loss += task_loss.item()
            pred = conscious_logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            
            metrics['focus'] += p_focus
            metrics['explore'] += adaptive_explore
            metrics['repair'] += p_repair
            metrics['richness'] += rich_val
        
        epoch_time = time.perf_counter() - epoch_start
        acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        n = len(train_loader)
        avg_rich = metrics['richness'] / n
        
        # FIX: Logging de estabilidad del sistema
        stability_score = 1.0 - (emergency_counter / max(epoch, 1))
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | "
              f"Rich:{avg_rich:.1f} | F:{metrics['focus']/n:.2f} E:{metrics['explore']/n:.2f} R:{metrics['repair']/n:.2f} | "
              f"Cons:{metrics['consolidations']} | FastNorm:{fast_norm:.1f} | Stable:{stability_score:.2f} | Time:{epoch_time:.2f}s")
    
    print("✅ Fase 3 completada\n")
    
    # FIX: Reporte final de salud del sistema
    print(f"  Emergencias totales: {emergency_counter}")
    print(f"  Riqueza final: {avg_rich:.1f}")
    
    for param in model.unconscious.parameters():
        param.requires_grad = True


def evaluate_dualmind(model, test_loader, device):
    """Evaluación del sistema dual"""
    model.eval()
    correct = 0
    total = 0
    
    eval_start = time.perf_counter()
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _, _, _ = model(x, mode='conscious')
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    
    eval_time = time.perf_counter() - eval_start
    acc = 100.0 * correct / total
    
    print(f"EVALUATION TIME: {eval_time:.2f}s")
    
    return acc


# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================



def run_dualmind_experiment():
    print("\n" + "="*80)
    print("🧬 NeuroSovereign v1.0 POC: NestedTopoBrain + Homeostatic Synthesis")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Dataset CIFAR-10 con aumentación mejorada
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),  # Aumentación adicional
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    # Batch size mantenido para plasticidad hebbiana fuerte
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # CAPACIDAD AUMENTADA: grid_size=6 (36 nodos), hidden_dim=96
    model = DualMind(
        in_channels=3,
        grid_size=6,        # Aumentado de 4 a 6 para mayor capacidad topológica
        hidden_dim=96,      # Aumentado de 64 a 96
        conscious_dim=128,
        num_classes=10
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Optimizadores con weight decay reducido para plasticidad
    optimizer_phase1 = optim.AdamW(model.unconscious.parameters(), lr=0.015, weight_decay=5e-5)  # LR aumentado
    optimizer_phase2 = optim.AdamW(model.conscious.parameters(), lr=0.008, weight_decay=5e-5)    # LR aumentado
    optimizer_phase3 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005, weight_decay=1e-4)
    
    # EPOCHS AUMENTADOS: Fase 1 y 2 más largas para aprendizaje profundo
    train_dualmind_phase1(model, train_loader, optimizer_phase1, device, epochs=10)  # Era 5, ahora 10
    train_dualmind_phase2(model, train_loader, optimizer_phase2, device, epochs=15)  # Era 10, ahora 15
    train_dualmind_phase3(model, train_loader, optimizer_phase3, device, epochs=15)
    
    print("="*60)
    print("EVALUACIÓN FINAL")
    print("="*60)
    
    final_acc = evaluate_dualmind(model, test_loader, device)
    print(f"✅ Test Accuracy (Conscious Mode): {final_acc:.2f}%\n")
    
    status = model.get_system_status()
    print("Diagnóstico del Sistema:")
    print(f"  Inconsciente - Densidad Topológica: {status['unconscious']['avg_density']:.3f}")
    print(f"  Consciente - Entropía Estructural: {status['conscious']['structure_entropy']:.3f}")
    
    save_path = Path("dualmind_v1_checkpoint.pt")
    torch.save({
        'model_state': model.state_dict(),
        'final_accuracy': final_acc,
        'system_status': status
    }, save_path)
    print(f"\n💾 Modelo guardado en {save_path}")
    
    return model, final_acc

if __name__ == "__main__":
    model, acc = run_dualmind_experiment()
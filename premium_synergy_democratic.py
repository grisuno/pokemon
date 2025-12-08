#!/usr/bin/env python3
"""
Premium Synergy - Sistema Democrático Deliberativo
===================================================
Combina TopoBrain + OmniBrain + Quimera con autoregulación interna
y motor homeostático como "cámara alta" de deliberación.

Concepto: Sistema democrático donde cada componente delibera internamente
y un motor homeostático decide si las sinergias convergen en alta accuracy.

Componentes:
- Cámara Baja: Ejecución de TopoBrain + OmniBrain + Quimera
- Cámara Alta: Motor homeostático de deliberación
- Autoregulación: Diálogo interno fisiológico (metabolismo, sensibilidad, gating)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import time
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil no disponible - monitoreo de memoria limitado")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class PremiumSynergyConfig:
    """Configuración del sistema Premium Synergy"""
    # Dimensiones del dataset
    n_samples: int = 1000
    n_features: int = 64
    n_classes: int = 10
    
    # Arquitectura Base
    embed_dim: int = 128
    hidden_dim: int = 256
    
    # TopoBrain v8 - Dynamic Topology + Symbiotic Basis
    grid_size: int = 8
    use_topobrain: bool = True
    use_dynamic_topology: bool = True
    use_symbiotic_basis: bool = True
    
    # OmniBrain K - Integration Index + Fast-Slow Weights
    use_omnibrain: bool = True
    use_integration_index: bool = True
    use_fast_slow: bool = True
    use_dual_pathway: bool = True
    use_memory_buffer: bool = True
    
    # Quimera v9.5 - Liquid Neurons + Sovereign Attention
    use_quimera: bool = True
    use_liquid_neurons: bool = True
    use_sovereign_attention: bool = True
    use_dual_phase_memory: bool = True
    use_svd_consolidation: bool = True
    
    # Motor Homeostático (Cámara Alta) - UMREAL REALISTA
    homeostatic_threshold: float = 0.40  # 40% es más realista para inicio
    convergence_epochs: int = 5
    synergy_alpha: float = 0.1
    entropy_regulation: bool = True
    
    # Entrenamiento
    epochs: int = 50
    batch_size: int = 16  # Pequeño para testing
    lr: float = 0.001
    weight_decay: float = 1e-5
    
    # Checkeo de Memoria
    max_memory_gb: float = 2.0
    memory_check_interval: int = 10
    gc_threshold: float = 0.8

# =============================================================================
# MEMORIA CHECKER
# =============================================================================

class MemoryChecker:
    """Sistema de monitoreo de memoria"""
    
    def __init__(self, max_memory_gb: float):
        self.max_memory_gb = max_memory_gb
        self.has_psutil = PSUTIL_AVAILABLE
        
    def check_memory(self) -> Dict[str, float]:
        """Verifica el uso de memoria actual"""
        if self.has_psutil:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent,
                'process_gb': process.memory_info().rss / (1024**3),
                'within_limit': (memory.percent / 100.0) < 0.8
            }
        else:
            # Fallback sin psutil
            return {
                'total_gb': 8.0,  # Estimación
                'available_gb': 4.0,
                'used_gb': 4.0,
                'percent': 50.0,
                'process_gb': 0.5,
                'within_limit': True
            }
    
    def warn_if_high(self) -> bool:
        """Advierte si el uso de memoria es alto"""
        info = self.check_memory()
        if info['percent'] > 80:
            print(f"⚠️  ALTA MEMORIA: {info['percent']:.1f}% - Ejecutando garbage collection")
            gc.collect()
            return True
        return False

# =============================================================================
# TOPOBRAIN V8 - DYNAMIC TOPOLOGY + SYMBIOTIC BASIS
# =============================================================================

class TopoBrainComponent(nn.Module):
    """TopoBrain v8 con autoregulación interna"""
    
    def __init__(self, config: PremiumSynergyConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        
        # Embedding
        self.input_embed = nn.Linear(
            config.n_features, 
            config.embed_dim * self.num_nodes
        )
        
        # Dynamic Topology (si habilitado)
        if config.use_dynamic_topology:
            self.topology = DynamicTopologyGrid(
                self.num_nodes, config.grid_size
            )
        
        # Node Processing
        self.node_proc1 = nn.Linear(config.embed_dim, config.hidden_dim)
        self.node_proc2 = nn.Linear(config.hidden_dim, config.embed_dim)
        
        # Symbiotic Basis (si habilitado)
        if config.use_symbiotic_basis:
            self.symbiotic = SymbioticBasis(config.embed_dim, num_atoms=6)
        
        # Autoregulación
        self.metabolism_regulator = MetabolismRegulator(config.embed_dim)
        self.sensitivity_gate = SensitivityGate(config.embed_dim)
        
        # Métricas internas
        self.register_buffer('topology_activity', torch.zeros(self.num_nodes))
        self.register_buffer('symbiotic_coherence', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor, plasticity: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)
        
        # Embedding
        x_embed = self.input_embed(x)
        x_embed = x_embed.view(batch_size, self.num_nodes, -1)
        
        # Topology aggregation
        if hasattr(self, 'topology'):
            adj = self.topology.get_adjacency(plasticity)
            x_agg = torch.bmm(adj.unsqueeze(0).expand(batch_size, -1, -1), x_embed)
        else:
            x_agg = x_embed
        
        # Node processing with autoregulation
        x_proc = F.relu(self.node_proc1(x_agg))
        
        # Metabolism regulation
        x_proc = self.metabolism_regulator(x_proc)
        
        # Sensitivity gating
        x_proc = self.sensitivity_gate(x_proc)
        
        x_proc = self.node_proc2(x_proc)
        
        # Symbiotic refinement
        if hasattr(self, 'symbiotic'):
            x_flat = x_proc.view(-1, config.embed_dim)
            x_refined = self.symbiotic(x_flat)
            x_proc = x_refined.view(batch_size, self.num_nodes, -1)
        
        # Métricas internas
        with torch.no_grad():
            self.topology_activity = 0.9 * self.topology_activity + 0.1 * x_proc.mean(dim=(0, 2)).abs()
            if hasattr(self, 'symbiotic'):
                gram = torch.mm(self.symbiotic.basis, self.symbiotic.basis.T)
                identity = torch.eye(gram.size(0), device=gram.device)
                self.symbiotic_coherence = torch.norm(gram - identity, p='fro').item()
        
        x_final = x_proc.view(batch_size, -1)
        return x_final, {
            'topology_activity': self.topology_activity.mean().item(),
            'symbiotic_coherence': self.symbiotic_coherence,
            'metabolism_state': self.metabolism_regulator.get_state().item(),
            'sensitivity_level': self.sensitivity_gate.get_level().item()
        }
    
    def internal_dialogue(self) -> Dict[str, float]:
        """Diálogo interno fisiológico - metabolimo, sensibilidad, gating"""
        return {
            'metabolism_regulation': self.metabolism_regulator.get_state().item(),
            'sensitivity_gating': self.sensitivity_gate.get_level().item(),
            'topology_stability': (1.0 - self.topology_activity.std().item()).clip(0, 1),
            'symbiotic_harmony': (1.0 - min(self.symbiotic_coherence, 1.0)).item()
        }

# =============================================================================
# OMNIBRAIN K - INTEGRATION INDEX + FAST-SLOW WEIGHTS
# =============================================================================

class OmniBrainComponent(nn.Module):
    """OmniBrain K con autoregulación interna"""
    
    def __init__(self, config: PremiumSynergyConfig):
        super().__init__()
        self.config = config
        self.dim = config.hidden_dim
        
        # Core layers
        self.core1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.core2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Integration Index (si habilitado)
        if config.use_integration_index:
            self.integration_module = IntegrationModule(config.hidden_dim)
        
        # Fast-Slow Weights (si habilitado)
        if config.use_fast_slow:
            self.fast_slow1 = FastSlowLinear(config.hidden_dim, config.hidden_dim)
            self.fast_slow2 = FastSlowLinear(config.hidden_dim, config.hidden_dim)
        
        # Dual Pathway (si habilitado)
        if config.use_dual_pathway:
            self.dual_system = DualSystemModule(config.hidden_dim)
        
        # Autoregulación
        self.integrative_control = IntegrativeControl(config.hidden_dim)
        self.chaos_modulator = ChaosModulator(config.hidden_dim)
        
        # Métricas internas
        self.register_buffer('integration_level', torch.tensor(0.5))
        self.register_buffer('fast_slow_balance', torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor, chaos_level: float = 0.0) -> Tuple[torch.Tensor, Dict]:
        # Core processing
        x = F.relu(self.core1(x))
        
        # Autoregulación - control integrativo
        x = self.integrative_control(x)
        
        # Chaos modulation
        x = self.chaos_modulator(x, chaos_level)
        
        # Integration index
        integration_metrics = {}
        if hasattr(self, 'integration_module'):
            x = self.integration_module(x)
            integration_metrics['integration_index'] = self.integration_module.get_level().item()
        
        # Fast-Slow weights
        if hasattr(self, 'fast_slow1'):
            x = self.fast_slow1(x)
            x = self.fast_slow2(x)
            self.fast_slow_balance = 0.9 * self.fast_slow_balance + 0.1 * x.abs().mean()
        
        # Dual pathway
        dual_metrics = {}
        if hasattr(self, 'dual_system'):
            x = self.dual_system(x)
            dual_metrics['dual_balance'] = self.dual_system.get_balance().item()
        
        x = F.relu(self.core2(x))
        
        # Update metrics
        with torch.no_grad():
            if hasattr(self, 'integration_module'):
                self.integration_level = 0.95 * self.integration_level + 0.05 * x.std().item()
        
        return x, {
            **integration_metrics,
            **dual_metrics,
            'integration_level': self.integration_level.item(),
            'fast_slow_balance': self.fast_slow_balance.item(),
            'integrative_control': self.integrative_control.get_state().item(),
            'chaos_resistance': self.chaos_modulator.get_resistance().item()
        }
    
    def internal_dialogue(self) -> Dict[str, float]:
        """Diálogo interno - balance integrativo y modulación caótica"""
        return {
            'integrative_balance': self.integrative_control.get_state().item(),
            'chaos_resistance': self.chaos_modulator.get_resistance().item(),
            'fast_slow_equilibrium': self.fast_slow_balance.item(),
            'integration_stability': (1.0 - abs(self.integration_level.item() - 0.5)).item()
        }

# =============================================================================
# QUIMERA V9.5 - LIQUID NEURONS + SOVEREIGN ATTENTION
# =============================================================================

class QuimeraComponent(nn.Module):
    """Quimera v9.5 con autoregulación interna"""
    
    def __init__(self, config: PremiumSynergyConfig):
        super().__init__()
        self.config = config
        self.dim = config.hidden_dim
        
        # Liquid Neurons (si habilitado)
        if config.use_liquid_neurons:
            self.liquid1 = LiquidNeuron(config.hidden_dim, config.hidden_dim)
            self.liquid2 = LiquidNeuron(config.hidden_dim, config.hidden_dim)
        
        # Sovereign Attention (si habilitado)
        if config.use_sovereign_attention:
            self.sovereign_att = SovereignAttention(config.hidden_dim)
        
        # Dual Phase Memory (si habilitado)
        if config.use_dual_phase_memory:
            self.dual_memory = DualPhaseMemory(config.hidden_dim)
        
        # Processing layers
        self.proc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.proc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Autoregulación
        self.phase_regulator = PhaseRegulator(config.hidden_dim)
        self.attention_controller = AttentionController(config.hidden_dim)
        
        # Métricas internas
        self.register_buffer('liquid_plasticity', torch.tensor(0.5))
        self.register_buffer('attention_focus', torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor, plasticity: float = 1.0, chaos: bool = False) -> Tuple[torch.Tensor, Dict]:
        # Phase regulation
        x = self.phase_regulator(x)
        
        # Attention control
        x, attention_metrics = self.attention_controller(x, chaos)
        self.attention_focus = 0.9 * self.attention_focus + 0.1 * x.abs().mean()
        
        # Liquid neurons
        liquid_metrics = {}
        if hasattr(self, 'liquid1'):
            x = self.liquid1(x, plasticity)
            x = self.liquid2(x, plasticity)
            self.liquid_plasticity = 0.9 * self.liquid_plasticity + 0.1 * x.abs().std()
            liquid_metrics['liquid_plasticity'] = self.liquid_plasticity.item()
        
        # Sovereign attention
        if hasattr(self, 'sovereign_att'):
            x = self.sovereign_att(x, chaos)
            sovereign_metrics = self.sovereign_att.get_metrics()
        else:
            sovereign_metrics = {}
        
        # Dual phase memory
        if hasattr(self, 'dual_memory'):
            phase_idx = 2 if chaos else 1  # 1=WORLD_2, 2=CHAOS
            x = self.dual_memory(x, phase_idx)
            if self.training:
                self.dual_memory.update(x, phase_idx)
        
        # Processing
        x = F.relu(self.proc1(x))
        x = self.proc2(x)
        
        return x, {
            **liquid_metrics,
            **sovereign_metrics,
            **attention_metrics,
            'attention_focus': self.attention_focus.item(),
            'phase_regulation': self.phase_regulator.get_level().item()
        }
    
    def internal_dialogue(self) -> Dict[str, float]:
        """Diálogo interno - regulación de fases y control atencional"""
        return {
            'phase_stability': self.phase_regulator.get_level().item(),
            'attention_control': self.attention_controller.get_control().item(),
            'liquid_equilibrium': self.liquid_plasticity.item(),
            'memory_coherence': self.dual_memory.get_coherence() if hasattr(self, 'dual_memory') else 0.5
        }
    
    def consolidate(self):
        """SVD consolidation de liquid neurons"""
        if hasattr(self, 'liquid1') and self.config.use_svd_consolidation:
            self.liquid1.consolidate_svd()
            self.liquid2.consolidate_svd()

# =============================================================================
# COMPONENTES AUXILIARES
# =============================================================================

class MetabolismRegulator(nn.Module):
    """Regulador de metabolismo para TopoBrain"""
    def __init__(self, dim: int):
        super().__init__()
        self.metabolism_net = nn.Sequential(
            nn.Linear(1, 8),  # Input is a single scalar value
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.register_buffer('state', torch.tensor(0.5))
    
    def forward(self, x):
        # DEBUG: print actual dimensions
        print(f"🔍 MetabolismRegulator input shape: {x.shape}")
        
        # Create scalar features from input regardless of dimensions
        if x.dim() == 2:
            # Standard case: (batch_size, features)
            x_mean = x.mean().item()  # scalar
            x_std = x.std().item()   # scalar
            input_scalar = (x_mean + x_std) / 2  # scalar
        else:
            # Fallback
            input_scalar = x.mean().item()
        
        # Process scalar through network
        input_tensor = torch.tensor([[input_scalar]], dtype=x.dtype, device=x.device)  # (1, 1)
        print(f"🔍 MetabolismRegulator input_tensor shape: {input_tensor.shape}")
        
        metabolism_output = self.metabolism_net(input_tensor)  # (1, 1)
        current_state = metabolism_output[0, 0].item()  # scalar
        
        self.state = 0.9 * self.state + 0.1 * current_state
        
        return x * (1.0 + 0.1 * (self.state - 0.5))
    
    def get_state(self):
        return self.state

class SensitivityGate(nn.Module):
    """Compuerta de sensibilidad para TopoBrain"""
    def __init__(self, dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(1, 16),  # Input is a single scalar value
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.register_buffer('level', torch.tensor(0.5))
    
    def forward(self, x):
        # Extract scalar features from input
        if x.dim() == 2:
            x_mean = x.mean().item()  # scalar
            x_std = x.std().item()   # scalar
            input_scalar = (x_mean + x_std) / 2  # scalar
        else:
            input_scalar = x.mean().item()
        
        # Process scalar through gate network
        input_tensor = torch.tensor([[input_scalar]], dtype=x.dtype, device=x.device)  # (1, 1)
        gate_value = self.gate_net(input_tensor)  # (1, 1)
        
        # Apply gate to all features
        gate_factor = gate_value[0, 0].item()  # scalar
        self.level = 0.9 * self.level + 0.1 * gate_factor
        return x * gate_factor
    
    def get_level(self):
        return self.level

class DynamicTopologyGrid(nn.Module):
    """Topología dinámica para TopoBrain"""
    def __init__(self, num_nodes: int, grid_size: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.grid_size = grid_size
        self.adj_weights = nn.Parameter(torch.ones(num_nodes, num_nodes))
        self.register_buffer('adj_mask', self._create_grid_mask())
    
    def _create_grid_mask(self):
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
    
    def get_adjacency(self, plasticity: float = 1.0):
        adj = torch.sigmoid(self.adj_weights * plasticity) * self.adj_mask
        deg = adj.sum(1, keepdim=True).clamp(min=1e-6)
        return adj / deg

class SymbioticBasis(nn.Module):
    """Basis simbólica para TopoBrain"""
    def __init__(self, dim: int, num_atoms: int = 4):
        super().__init__()
        self.num_atoms = num_atoms
        self.dim = dim
        self.basis = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis, gain=1.0)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.eps = 1e-8
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(self.basis)
        attn = torch.matmul(Q, K.T) / (self.dim ** 0.5 + self.eps)
        weights = F.softmax(attn, dim=-1)
        x_clean = torch.matmul(weights, self.basis)
        return torch.clamp(x_clean, -3.0, 3.0)

# OmniBrain Components
class IntegrationModule(nn.Module):
    """Módulo de integración para OmniBrain"""
    def __init__(self, dim: int):
        super().__init__()
        self.feature_processor = nn.Sequential(
            nn.Linear(1, 8),  # Process scalar features from input
            nn.ReLU(),
            nn.Linear(8, dim)  # Output matches input dimension
        )
        self.register_buffer('level', torch.tensor(0.2))
        self.threshold = 0.3
    
    def forward(self, x):
        # Extract scalar features from input for control
        x_mean = x.mean(dim=1, keepdim=True)  # (batch_size, 1)
        x_features = self.feature_processor(x_mean)  # (batch_size, dim)
        
        if self.level > self.threshold:
            return x_features
        else:
            return x + 0.1 * x_features
    
    def get_level(self):
        return self.level

class FastSlowLinear(nn.Module):
    """Capa fast-slow weights para OmniBrain"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.slow_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.slow_bias = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.slow_weight)
        nn.init.zeros_(self.slow_bias)
        
        self.register_buffer('fast_weight', torch.zeros(out_dim, in_dim))
        self.register_buffer('fast_bias', torch.zeros(out_dim))
        self.register_buffer('update_counter', torch.tensor(0))
        self.register_buffer('fast_weight_norm', torch.tensor(0.0))
    
    def forward(self, x):
        if self.training:
            self.update_counter += 1
            # Hebbian update every 5 steps
            if self.update_counter % 5 == 0:
                with torch.no_grad():
                    hebb_update = torch.mm(self.slow_weight.t(), x) / x.size(0)
                    self.fast_weight.add_(hebb_update * 0.001)
        
        effective_w = self.slow_weight + self.fast_weight
        effective_b = self.slow_bias + self.fast_bias
        
        out = F.linear(x, effective_w, effective_b)
        self.fast_weight_norm = self.fast_weight.norm().item()
        return out

class DualSystemModule(nn.Module):
    """Sistema dual para OmniBrain"""
    def __init__(self, dim: int):
        super().__init__()
        self.fast_path = FastSlowLinear(dim, dim)
        self.slow_path = FastSlowLinear(dim, dim)
        self.integrator = nn.Linear(dim * 2, dim)
        self.register_buffer('balance', torch.tensor(0.5))
    
    def forward(self, x):
        fast_out = self.fast_path(x)
        slow_out = self.slow_path(x)
        combined = torch.cat([fast_out, slow_out], dim=1)
        out = self.integrator(combined)
        self.balance = 0.9 * self.balance + 0.1 * (fast_out - slow_out).abs().mean()
        return out
    
    def get_balance(self):
        return self.balance

class IntegrativeControl(nn.Module):
    """Control integrativo para OmniBrain"""
    def __init__(self, dim: int):
        super().__init__()
        self.control_net = nn.Sequential(
            nn.Linear(1, 8),  # Input is a single scalar value
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.register_buffer('state', torch.tensor(0.5))
    
    def forward(self, x):
        # Extract scalar features from input
        if x.dim() == 2:
            x_mean = x.mean().item()  # scalar
            x_std = x.std().item()   # scalar
            input_scalar = (x_mean + x_std) / 2  # scalar
        else:
            input_scalar = x.mean().item()
        
        # Process scalar through control network
        input_tensor = torch.tensor([[input_scalar]], dtype=x.dtype, device=x.device)  # (1, 1)
        control_signal = self.control_net(input_tensor)  # (1, 1)
        
        # Apply control to all features
        control_factor = control_signal[0, 0].item()  # scalar
        self.state = 0.9 * self.state + 0.1 * control_factor
        return x * control_factor
    
    def get_state(self):
        return self.state

class ChaosModulator(nn.Module):
    """Modulador de caos para OmniBrain"""
    def __init__(self, dim: int):
        super().__init__()
        self.modulation_net = nn.Sequential(
            nn.Linear(1, 8),  # Input is a single scalar value
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.register_buffer('resistance', torch.tensor(0.7))
    
    def forward(self, x, chaos_level: float):
        # Extract scalar features from input
        if x.dim() == 2:
            x_mean = x.mean().item()  # scalar
            x_std = x.std().item()   # scalar
            input_scalar = (x_mean + x_std) / 2  # scalar
        else:
            input_scalar = x.mean().item()
        
        # Process scalar through modulation network
        input_tensor = torch.tensor([[input_scalar]], dtype=x.dtype, device=x.device)  # (1, 1)
        mod = self.modulation_net(input_tensor)  # (1, 1)
        
        # Adapt resistance based on chaos level
        mod_factor = mod[0, 0].item()  # scalar
        self.resistance = 0.95 * self.resistance + 0.05 * (1.0 - chaos_level)
        return x * (1.0 + mod_factor * (1.0 - self.resistance))
    
    def get_resistance(self):
        return self.resistance

# Quimera Components
class LiquidNeuron(nn.Module):
    """Neurona líquida para Quimera"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.norm = nn.LayerNorm(out_dim)
        self.fast_lr = 0.005
        self.register_buffer('plasticity', torch.tensor(0.5))
        
    def forward(self, x, plasticity: float = 1.0):
        if self.training and plasticity > 0:
            with torch.no_grad():
                # Oja's rule for plasticity
                hebb_update = torch.mm(self.W_slow.weight.t(), x) / x.size(0)
                self.W_fast.add_(hebb_update * self.fast_lr)
        
        effective_w = self.W_slow.weight + self.W_fast
        out = F.linear(x, effective_w, self.W_slow.bias)
        return self.norm(out)
    
    def consolidate_svd(self, strength: float = 1.0):
        with torch.no_grad():
            combined = self.W_slow.weight + self.W_fast
            U, S, V = torch.svd(combined)
            threshold = S.mean() * 0.1
            S_filtered = S * (S > threshold).float()
            self.W_slow.weight.data = torch.mm(U, torch.mm(torch.diag(S_filtered), V.t()))
            self.W_fast.fill_(0.0)

class SovereignAttention(nn.Module):
    """Atención soberana para Quimera"""
    def __init__(self, dim: int):
        super().__init__()
        self.global_att = nn.Linear(dim, dim, bias=False)
        self.local_att = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.global_att.weight)
        nn.init.eye_(self.local_att.weight)
        
    def forward(self, x, is_chaos: bool = False):
        global_mask = torch.sigmoid(self.global_att(x))
        local_mask = torch.sigmoid(self.local_att(x))
        
        # Weighting based on chaos
        if is_chaos:
            combined_mask = 0.7 * local_mask + 0.3 * global_mask
        else:
            combined_mask = 0.3 * local_mask + 0.7 * global_mask
        
        return x * combined_mask
    
    def get_metrics(self):
        return {
            'sovereign_focus': (self.global_att.weight.mean().item() + 1.0) / 2.0
        }

class DualPhaseMemory(nn.Module):
    """Memoria de fase dual para Quimera"""
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('stable_mem', torch.zeros(dim))
        self.register_buffer('adapt_mem', torch.zeros(dim))
        self.alpha = 0.95
    
    def forward(self, x, phase_idx: int):
        if phase_idx == 1:  # WORLD_2
            return x + 0.1 * self.adapt_mem
        elif phase_idx == 2:  # CHAOS
            return x + 0.1 * self.stable_mem
        else:
            return x
    
    def update(self, x, phase_idx: int):
        with torch.no_grad():
            mean_x = x.mean(dim=0)
            if phase_idx == 1:  # WORLD_2
                self.adapt_mem.data = 0.9 * self.adapt_mem.data + 0.1 * mean_x
            elif phase_idx == 2:  # CHAOS
                self.stable_mem.data = self.alpha * self.stable_mem.data + (1 - self.alpha) * mean_x
    
    def get_coherence(self):
        return torch.cosine_similarity(self.stable_mem, self.adapt_mem, dim=0).item()

class PhaseRegulator(nn.Module):
    """Regulador de fases para Quimera"""
    def __init__(self, dim: int):
        super().__init__()
        self.phase_net = nn.Sequential(
            nn.Linear(1, 8),  # Input is a single scalar value
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.register_buffer('level', torch.tensor(0.5))
    
    def forward(self, x):
        # Extract scalar features from input
        if x.dim() == 2:
            x_mean = x.mean().item()  # scalar
            x_std = x.std().item()   # scalar
            input_scalar = (x_mean + x_std) / 2  # scalar
        else:
            input_scalar = x.mean().item()
        
        # Process scalar through phase network
        input_tensor = torch.tensor([[input_scalar]], dtype=x.dtype, device=x.device)  # (1, 1)
        phase_signal = self.phase_net(input_tensor)  # (1, 1)
        
        # Apply phase modulation
        phase_factor = phase_signal[0, 0].item()  # scalar
        self.level = 0.9 * self.level + 0.1 * phase_factor
        return x * (1.0 + 0.05 * (phase_factor - 0.5))
    
    def get_level(self):
        return self.level

class AttentionController(nn.Module):
    """Controlador de atención para Quimera"""
    def __init__(self, dim: int):
        super().__init__()
        self.control_net = nn.Sequential(
            nn.Linear(1, 8),  # Input is a single scalar value
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.register_buffer('control', torch.tensor(0.5))
    
    def forward(self, x, chaos: bool):
        # Extract scalar features from input
        if x.dim() == 2:
            x_mean = x.mean().item()  # scalar
            x_std = x.std().item()   # scalar
            input_scalar = (x_mean + x_std) / 2  # scalar
        else:
            input_scalar = x.mean().item()
        
        # Process scalar through control network
        input_tensor = torch.tensor([[input_scalar]], dtype=x.dtype, device=x.device)  # (1, 1)
        control_signal = self.control_net(input_tensor)  # (1, 1)
        
        # Apply control to all features
        control_factor = control_signal[0, 0].item()  # scalar
        if chaos:
            self.control = 0.9 * self.control + 0.1 * control_factor
        return x * control_factor, {'attention_control': self.control.item()}
    
    def get_control(self):
        return self.control

# =============================================================================
# MOTOR HOMEOSTÁTICO - CÁMARA ALTA
# =============================================================================

class HomeostaticMotor(nn.Module):
    """Motor homeostático - Cámara Alta de deliberación democrática"""
    
    def __init__(self, config: PremiumSynergyConfig):
        super().__init__()
        self.config = config
        
        # Deliberación components
        self.deliberation_net = nn.Sequential(
            nn.Linear(3 * config.hidden_dim, config.hidden_dim),  # 3 components
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 3),  # 3 componente weights
            nn.Sigmoid()
        )
        
        # Convergence tracking
        self.register_buffer('synergy_evolution', torch.zeros(10))  # Last 10 epochs
        self.register_buffer('convergence_counter', torch.tensor(0))
        
        # Homeostatic metrics
        self.register_buffer('target_accuracy', torch.tensor(config.homeostatic_threshold))
        self.register_buffer('current_accuracy', torch.tensor(0.0))
        
    def forward(self, topobrain_out: torch.Tensor, omnibrain_out: torch.Tensor, 
                quimera_out: torch.Tensor, target_accuracy: float = None) -> Tuple[torch.Tensor, Dict]:
        """
        Cámara Alta: Delibera sobre las sinergias de los componentes
        
        Returns:
            adjusted_output: Output ajustado por la deliberación
            metrics: Métricas del proceso deliberativo
        """
        batch_size = topobrain_out.size(0)
        
        # Combine all component outputs
        combined = torch.cat([topobrain_out, omnibrain_out, quimera_out], dim=1)
        
        # Deliberación democrática weights
        component_weights = self.deliberation_net(combined)
        
        # Calculate REAL synergy: cooperation between components
        # High synergy = components working together (high cooperation)
        tb_cooperation = topobrain_out.mean().abs()
        ob_cooperation = omnibrain_out.mean().abs() 
        q_cooperation = quimera_out.mean().abs()
        
        # Synergy = average cooperation level (not dispersion!)
        synergy_strength = (tb_cooperation + ob_cooperation + q_cooperation) / 3
        
        # High synergy when all components cooperate equally
        mean_cooperation = synergy_strength
        balance_factor = 1.0 - torch.abs(tb_cooperation - ob_cooperation).item()
        balance_factor = min(balance_factor, 1.0)
        
        real_synergy = mean_cooperation * balance_factor
        
        # Apply democratic weighting
        weighted_topobrain = topobrain_out * component_weights[:, 0:1]
        weighted_omnibrain = omnibrain_out * component_weights[:, 1:2]
        weighted_quimera = quimera_out * component_weights[:, 2:3]
        
        # Integrate with homoeostatic balance
        integrated_output = weighted_topobrain + weighted_omnibrain + weighted_quimera
        
        # Update homeostatic state
        if target_accuracy is not None:
            self.current_accuracy = 0.9 * self.current_accuracy + 0.1 * target_accuracy
            
            # Check convergence
            self.synergy_evolution[1:] = self.synergy_evolution[:-1].clone()
            self.synergy_evolution[0] = target_accuracy
            
            if target_accuracy >= self.config.homeostatic_threshold:
                self.convergence_counter += 1
            else:
                self.convergence_counter = 0
        
        return integrated_output, {
            'component_weights': component_weights.mean(dim=0).cpu().numpy(),
            'synergy_strength': real_synergy.item(),  # CORRECTED: cooperation, not dispersion
            'convergence_level': self.convergence_counter.item(),
            'homeostatic_balance': (self.target_accuracy - self.current_accuracy).item(),
            'democratic_deliberation': real_synergy.item(),  # CORRECTED
            'tb_cooperation': tb_cooperation.item(),
            'ob_cooperation': ob_cooperation.item(),
            'q_cooperation': q_cooperation.item()
        }
    
    def adjust_for_convergence(self, performance_metrics: Dict) -> Dict[str, float]:
        """
        Motor homeostático ajusta si las sinergias no convergen
        """
        current_acc = performance_metrics.get('accuracy', 0.0)
        
        if current_acc < self.config.homeostatic_threshold:
            # Need adjustment - increase synergy pressure
            adjustment = {
                'increase_topobrain_plasticity': 0.1,
                'enhance_omnibrain_integration': 0.1,
                'boost_quimera_liquid': 0.1,
                'reduce_entropy_regulation': 0.05
            }
            
            if self.convergence_counter >= self.config.convergence_epochs:
                # Strong adjustment needed
                adjustment = {k: v * 2 for k, v in adjustment.items()}
                
        else:
            # Good performance - maintain balance
            adjustment = {
                'increase_topobrain_plasticity': 0.02,
                'enhance_omnibrain_integration': 0.02,
                'boost_quimera_liquid': 0.02,
                'reduce_entropy_regulation': 0.01
            }
        
        return adjustment

# =============================================================================
# PREMIUM SYNERGY MODEL - SISTEMA INTEGRAL
# =============================================================================

class PremiumSynergyModel(nn.Module):
    """Modelo Premium Synergy con sistema democrático deliberativo"""
    
    def __init__(self, config: PremiumSynergyConfig):
        super().__init__()
        self.config = config
        
        # Cámara Baja: Componentes principales
        if config.use_topobrain:
            self.topobrain = TopoBrainComponent(config)
        
        if config.use_omnibrain:
            self.omnibrain = OmniBrainComponent(config)
        
        if config.use_quimera:
            self.quimera = QuimeraComponent(config)
        
        # Cámara Alta: Motor Homeostático
        self.motor_homeostatico = HomeostaticMotor(config)
        
        # Output layer
        self.classifier = nn.Linear(config.hidden_dim * 3, config.n_classes)
        
        # Sistema de regulación general
        self.system_regulator = SystemRegulator(config.hidden_dim)
        
        # Memory checker
        self.memory_checker = MemoryChecker(config.max_memory_gb)
        
    def forward(self, x: torch.Tensor, chaos_level: float = 0.0) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass completo con sistema democrático
        """
        batch_size = x.size(0)
        all_metrics = {}
        
        # Check memory
        if batch_size % self.config.memory_check_interval == 0:
            memory_info = self.memory_checker.check_memory()
            if memory_info['within_limit']:
                print(f"Memory check: {memory_info['percent']:.1f}% - OK")
            else:
                print(f"⚠️ High memory usage: {memory_info['percent']:.1f}%")
        
        # Cámara Baja: Cada componente ejecuta con autoregulación
        component_outputs = []
        
        if hasattr(self, 'topobrain'):
            tb_out, tb_metrics = self.topobrain(x, plasticity=1.0)
            component_outputs.append(tb_out)
            all_metrics['topobrain'] = tb_metrics
            
            # Internal dialogue
            tb_dialogue = self.topobrain.internal_dialogue()
            all_metrics['topobrain_dialogue'] = tb_dialogue
        
        if hasattr(self, 'omnibrain'):
            ob_out, ob_metrics = self.omnibrain(tb_out if hasattr(self, 'topobrain') else x, chaos_level)
            component_outputs.append(ob_out)
            all_metrics['omnibrain'] = ob_metrics
            
            # Internal dialogue
            ob_dialogue = self.omnibrain.internal_dialogue()
            all_metrics['omnibrain_dialogue'] = ob_dialogue
        
        if hasattr(self, 'quimera'):
            q_out, q_metrics = self.quimera(
                ob_out if hasattr(self, 'omnibrain') else (tb_out if hasattr(self, 'topobrain') else x),
                plasticity=1.0,
                chaos=chaos_level > 0.5
            )
            component_outputs.append(q_out)
            all_metrics['quimera'] = q_metrics
            
            # Internal dialogue
            q_dialogue = self.quimera.internal_dialogue()
            all_metrics['quimera_dialogue'] = q_dialogue
        
        # Camera Alta: Motor Homeostático delibera
        if len(component_outputs) >= 3:
            integrated_output, deliberation_metrics = self.motor_homeostatico(
                component_outputs[0], component_outputs[1], component_outputs[2]
            )
            all_metrics['democratic_deliberation'] = deliberation_metrics
            
        elif len(component_outputs) == 2:
            integrated_output = component_outputs[0] + component_outputs[1]
            all_metrics['deliberation'] = {'synergy_strength': 0.5}
            
        else:
            integrated_output = component_outputs[0]
            all_metrics['deliberation'] = {'synergy_strength': 0.3}
        
        # System regulation
        regulated_output = self.system_regulator(integrated_output)
        
        # Final classification
        logits = self.classifier(regulated_output)
        
        return logits, all_metrics
    
    def democratic_deliberation_status(self) -> Dict:
        """Estado de la deliberación democrática"""
        return {
            'components_active': {
                'topobrain': hasattr(self, 'topobrain'),
                'omnibrain': hasattr(self, 'omnibrain'),
                'quimera': hasattr(self, 'quimera')
            },
            'motor_homeostatico': 'active',
            'system_regulation': 'active',
            'memory_monitoring': 'active'
        }

class SystemRegulator(nn.Module):
    """Regulador general del sistema"""
    def __init__(self, dim: int):
        super().__init__()
        self.regulator = nn.Sequential(
            nn.Linear(1, 8),  # Input is a single scalar value
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.register_buffer('regulation_level', torch.tensor(0.5))
    
    def forward(self, x):
        # Extract scalar features from input
        if x.dim() == 2:
            x_mean = x.mean().item()  # scalar
            x_std = x.std().item()   # scalar
            input_scalar = (x_mean + x_std) / 2  # scalar
        else:
            input_scalar = x.mean().item()
        
        # Process scalar through regulator network
        input_tensor = torch.tensor([[input_scalar]], dtype=x.dtype, device=x.device)  # (1, 1)
        reg_signal = self.regulator(input_tensor)  # (1, 1)
        
        # Apply regulation to all features
        reg_factor = reg_signal[0, 0].item()  # scalar
        self.regulation_level = 0.95 * self.regulation_level + 0.05 * reg_factor
        return x * reg_factor

# =============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================

def ensure_dependencies():
    """Asegura que las dependencias estén instaladas"""
    try:
        import sklearn
        print("✅ sklearn disponible")
    except ImportError:
        print("📦 Instalando sklearn...")
        os.system(f"{sys.executable} -m pip install scikit-learn --quiet")
        print("✅ sklearn instalado")

def create_synthetic_dataset(config: PremiumSynergyConfig):
    """Crea dataset sintético para testing"""
    ensure_dependencies()
    from sklearn.datasets import make_classification
    
    print(f"🎯 Generando dataset: {config.n_samples} samples, {config.n_features} features")
    
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        n_informative=config.n_features // 2,
        n_redundant=config.n_features // 4,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=42
    )
    
    # Normalizar
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")
    
    return (X_train, y_train), (X_test, y_test)

def train_premium_synergy(config: PremiumSynergyConfig):
    """Entrena el modelo Premium Synergy"""
    print("\n" + "="*80)
    print("🧠 PREMIUM SYNERGY - SISTEMA DEMOCRÁTICO DELIBERATIVO")
    print("="*80)
    
    # Dataset
    (X_train, y_train), (X_test, y_test) = create_synthetic_dataset(config)
    
    # Modelo
    model = PremiumSynergyModel(config)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parámetros totales: {total_params:,}")
    
    # Mostrar componentes activos
    status = model.democratic_deliberation_status()
    print(f"🏛️ Componentes activos:")
    for comp, active in status['components_active'].items():
        print(f"   {comp}: {'✅' if active else '❌'}")
    
    # Optimizador
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    
    # Loaders
    train_loader = create_dataloader(X_train, y_train, config.batch_size, shuffle=True)
    test_loader = create_dataloader(X_test, y_test, config.batch_size, shuffle=False)
    
    print(f"\n🏃 Iniciando entrenamiento ({config.epochs} epochs)...")
    print("-"*80)
    print(f"{'Ep':<4} {'Loss':<8} {'Acc':<8} {'Synergy':<8} {'Conv':<6} {'Mem':<8}")
    print("-"*80)
    
    best_acc = 0.0
    for epoch in range(config.epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        synergy_scores = []
        convergence_levels = []
        
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Varying chaos level
            chaos_level = 0.1 * (epoch / config.epochs)  # Increasing chaos
            logits, metrics = model(x, chaos_level)
            
            loss = F.cross_entropy(logits, y)
            
            # Add regularization based on synergy strength
            if 'democratic_deliberation' in metrics:
                synergy_strength = metrics['democratic_deliberation'].get('synergy_strength', 0.0)
                loss -= config.synergy_alpha * synergy_strength
                
                # Track metrics
                convergence_levels.append(metrics['democratic_deliberation'].get('convergence_level', 0))
                synergy_scores.append(synergy_strength)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update Quimera liquid neurons
            if hasattr(model, 'quimera'):
                model.quimera.consolidate()
            
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        
        train_acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        avg_synergy = np.mean(synergy_scores) if synergy_scores else 0.0
        avg_convergence = np.mean(convergence_levels) if convergence_levels else 0.0
        
        # Testing
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    logits, _ = model(x, chaos_level)
                    pred = logits.argmax(1)
                    test_correct += pred.eq(y).sum().item()
                    test_total += y.size(0)
            
            test_acc = 100.0 * test_correct / test_total
            best_acc = max(best_acc, test_acc)
            
            # Memory check
            memory_info = model.memory_checker.check_memory()
            memory_str = f"{memory_info['percent']:.1f}%"
            
            print(f"{epoch+1:<4} {avg_loss:<8.4f} {train_acc:<8.2f} {avg_synergy:<8.3f} {avg_convergence:<6.1f} {memory_str:<8}")
        else:
            test_acc = 0.0
            memory_info = model.memory_checker.check_memory()
            memory_str = f"{memory_info['percent']:.1f}%"
            print(f"{epoch+1:<4} {avg_loss:<8.4f} {train_acc:<8.2f} {avg_synergy:<8.3f} {avg_convergence:<6.1f} {memory_str:<8}")
        
        scheduler.step()
    
    print("-"*80)
    print(f"✅ Mejor Test Accuracy: {best_acc:.2f}%")
    
    # Análisis final del sistema democrático
    print(f"\n🏛️ ESTADO DEL SISTEMA DEMOCRÁTICO:")
    final_status = model.democratic_deliberation_status()
    for comp, active in final_status['components_active'].items():
        print(f"   {comp}: {'✅ Activo' if active else '❌ Inactivo'}")
    
    return model, best_acc

def create_dataloader(X, y, batch_size, shuffle=False):
    """Crea dataloader"""
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )

def main():
    """Función principal"""
    print("🎯 PREMIUM SYNERGY - SISTEMA DEMOCRÁTICO DELIBERATIVO")
    print("Combinando TopoBrain + OmniBrain + Quimera con motor homeostático")
    
    # Configuración optimizada para testing
    config = PremiumSynergyConfig(
        n_samples=1000,
        n_features=64,
        n_classes=10,
        epochs=30,  # Reducido para testing
        batch_size=16,
        # Todos los componentes habilitados
        use_topobrain=True,
        use_omnibrain=True,
        use_quimera=True,
        use_dynamic_topology=True,
        use_symbiotic_basis=True,
        use_integration_index=True,
        use_fast_slow=True,
        use_dual_pathway=True,
        use_liquid_neurons=True,
        use_sovereign_attention=True,
        use_dual_phase_memory=True,
        use_svd_consolidation=True,
        # Motor homeostático
        homeostatic_threshold=0.35,  # 35% más realista
        convergence_epochs=3
    )
    
    # Entrenamiento
    model, final_accuracy = train_premium_synergy(config)
    
    print(f"\n🎯 RESULTADO FINAL:")
    print(f"   Accuracy: {final_accuracy:.2f}%")
    print(f"   Sistema: Premium Synergy (TopoBrain + OmniBrain + Quimera)")
    print(f"   Arquitectura: Democrática deliberativa con motor homeostático")
    
    # Guardar modelo
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'final_accuracy': final_accuracy
    }, 'premium_synergy_democratic.pth')
    
    print(f"\n💾 Modelo guardado: premium_synergy_democratic.pth")

if __name__ == "__main__":
    main()

# Export for import
__all__ = ['PremiumSynergyModel', 'PremiumSynergyConfig', 'train_premium_synergy']
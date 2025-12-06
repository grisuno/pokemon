"""
NeuroLogos v5.3 - TopoBrain Homeostasis Trans-Contextual
==================================================================
INNOVACIONES CRÍTICAS basadas en resultados v5.2:

1. 🌍 ENTORNO NO ESTACIONARIO (como PhysioChimera):
   - WORLD_1: Clases 0-4 (primeros 30% epochs)
   - WORLD_2: Clases 5-9 (30%-60% epochs)
   - CHAOS: Todas las clases + ruido (60%-80% epochs)
   - RETURN: Volver a WORLD_1 (80%-100% epochs)
   → Medir RETENCIÓN y ADAPTACIÓN, no solo robustez adversarial

2. 📊 MÉTRICAS DE HOMEOSTASIS:
   - Logging detallado de metabolism, sensitivity, gate
   - Visualizar cómo la red se adapta entre mundos
   - Detectar si hay "pánico" o "calma" en cada fase

3. 🎚️ REGULACIÓN JERÁRQUICA:
   - Homeostasis NO SOLO en nodos, también en readout
   - El regulador modula toda la arquitectura
   - Dialogo interno más profundo

4. 🔪 ABLACIÓN SELECTIVA:
   - Eliminar MGF del sistema completo
   - Focus en Continuum + Homeostasis (la mejor sinergia)
   - Sistema más simple y efectivo

Arquitectura: TopoBrain Grid 2x2 + Regulación Fisiológica Trans-Contextual
Validación: Retención entre mundos no estacionarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import time
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN TRANS-CONTEXTUAL
# =============================================================================

@dataclass
class TransContextConfig:
    device: str = "cpu"
    seed: int = 42
    
    # Dataset (usamos digits como PhysioChimera)
    use_digits: bool = True  # True = digits, False = synthetic
    
    # Arquitectura
    grid_size: int = 2
    embed_dim: int = 8
    d_in: int = 64  # Para digits
    d_out: int = 10
    
    # Entrenamiento LARGO (como PhysioChimera)
    batch_size: int = 32
    epochs: int = 1000  # Más epochs para ver adaptación
    lr: float = 0.005
    
    # Adversarial (reducido para focus en adaptación)
    train_eps: float = 0.1
    pgd_steps: int = 2
    
    # FLAGS DE COMPONENTES
    use_homeostasis: bool = False
    use_hierarchical_homeo: bool = False  # NUEVO: regulación jerárquica
    use_continuum: bool = False
    log_homeostasis: bool = True  # NUEVO: logging de métricas

def seed_everything(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# ENTORNO NO ESTACIONARIO (como PhysioChimera)
# =============================================================================

class NonStationaryEnvironment:
    """
    Entorno que cambia de distribución como en PhysioChimera.
    Permite medir RETENCIÓN y ADAPTACIÓN, no solo robustez adversarial.
    """
    def __init__(self):
        from sklearn.datasets import load_digits
        data, target = load_digits(return_X_y=True)
        data = data / 16.0  # Normalizar
        
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.long)
        
        # División en dos mundos
        self.mask1 = self.y < 5  # WORLD_1: dígitos 0-4
        self.mask2 = self.y >= 5  # WORLD_2: dígitos 5-9
        
        self.X1_full = self.X[self.mask1]
        self.y1_full = self.y[self.mask1]
        self.X2_full = self.X[self.mask2]
        self.y2_full = self.y[self.mask2]
        
        print(f"📊 Entorno configurado:")
        print(f"   WORLD_1: {len(self.X1_full)} samples (clases 0-4)")
        print(f"   WORLD_2: {len(self.X2_full)} samples (clases 5-9)")
    
    def get_batch(self, phase: str, batch_size: int = 32):
        """Retorna batch según la fase del entrenamiento"""
        if phase == "WORLD_1":
            idx = torch.randint(0, len(self.X1_full), (batch_size,))
            return self.X1_full[idx], self.y1_full[idx]
        
        elif phase == "WORLD_2":
            idx = torch.randint(0, len(self.X2_full), (batch_size,))
            return self.X2_full[idx], self.y2_full[idx]
        
        elif phase == "CHAOS":
            # Mezcla de todo + ruido
            idx = torch.randint(0, len(self.X), (batch_size,))
            X_batch = self.X[idx]
            noise = torch.randn_like(X_batch) * 0.3
            return X_batch + noise, self.y[idx]
        
        else:
            raise ValueError(f"Fase desconocida: {phase}")
    
    def get_phase(self, epoch: int, total_epochs: int):
        """Determina la fase según el epoch actual"""
        progress = epoch / total_epochs
        
        if progress <= 0.3:
            return "WORLD_1"
        elif progress <= 0.6:
            return "WORLD_2"
        elif progress <= 0.8:
            return "CHAOS"
        else:
            return "WORLD_1"  # RETURN: volver al mundo inicial

# =============================================================================
# REGULADOR HOMEOSTÁTICO MEJORADO
# =============================================================================

class EnhancedHomeostaticRegulator(nn.Module):
    """
    Regulador homeostático con logging y sensores mejorados.
    Incluye diagnóstico para análisis post-hoc.
    """
    def __init__(self, d_in, log_metrics=False):
        super().__init__()
        self.log_metrics = log_metrics
        
        # Red reguladora
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.Tanh(),
            nn.Linear(8, 3),
            nn.Sigmoid()
        )
        
        # Baselines adaptativos
        self.register_buffer('baseline_var', torch.tensor(0.5))
        self.register_buffer('baseline_smooth', torch.tensor(1.0))
        
        # Buffers para logging (no afectan entrenamiento)
        if log_metrics:
            self.register_buffer('log_metabolism', torch.zeros(1))
            self.register_buffer('log_sensitivity', torch.zeros(1))
            self.register_buffer('log_gate', torch.zeros(1))
    
    def forward(self, x, h_pre, w_norm):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        
        # SENSOR 1: Estrés Natural
        if x.size(1) > 1:
            var = x.var(dim=1, keepdim=True)
            natural_stress = (var - self.baseline_var).abs()
            
            if self.training:
                with torch.no_grad():
                    self.baseline_var.copy_(0.95 * self.baseline_var + 0.05 * var.mean())
        else:
            natural_stress = torch.zeros(batch_size, 1)
        
        # SENSOR 2: Estrés Adversarial (suavidad)
        if x.size(1) > 2:
            x_sorted, _ = torch.sort(x, dim=1)
            diffs = (x_sorted[:, 1:] - x_sorted[:, :-1]).abs()
            smoothness = diffs.mean(dim=1, keepdim=True)
            adversarial_stress = (smoothness - self.baseline_smooth).abs()
            
            if self.training:
                with torch.no_grad():
                    self.baseline_smooth.copy_(0.95 * self.baseline_smooth + 0.05 * smoothness.mean())
        else:
            adversarial_stress = torch.zeros(batch_size, 1)
        
        # SENSOR 3: Excitación
        if h_pre.dim() == 1:
            h_pre = h_pre.unsqueeze(0)
        excitation = h_pre.abs().mean(dim=1, keepdim=True)
        
        # SENSOR 4: Fatiga
        if isinstance(w_norm, torch.Tensor):
            if w_norm.dim() == 0:
                w_norm = w_norm.view(1, 1)
            fatigue = w_norm.expand(batch_size, 1)
        else:
            fatigue = torch.tensor([[w_norm]], dtype=x.dtype).expand(batch_size, 1)
        
        # SENSOR 5: Gradiente
        gradient_proxy = (h_pre ** 2).mean(dim=1, keepdim=True).sqrt()
        
        # Fusión
        state = torch.cat([natural_stress, adversarial_stress, excitation, fatigue, gradient_proxy], dim=1)
        controls = self.net(state)
        
        # Logging
        if self.log_metrics and self.training:
            with torch.no_grad():
                self.log_metabolism.copy_(controls[:, 0].mean())
                self.log_sensitivity.copy_(controls[:, 1].mean())
                self.log_gate.copy_(controls[:, 2].mean())
        
        return {
            'metabolism': controls[:, 0].view(-1, 1),
            'sensitivity': controls[:, 1].view(-1, 1),
            'gate': controls[:, 2].view(-1, 1),
            # Diagnóstico
            'natural_stress': natural_stress.mean().item(),
            'adversarial_stress': adversarial_stress.mean().item(),
            'excitation': excitation.mean().item()
        }

# =============================================================================
# COMPONENTES CON HOMEOSTASIS
# =============================================================================

class TransContextContinuumCell(nn.Module):
    """Memoria continua con homeostasis"""
    def __init__(self, dim, use_homeostasis=False, log_metrics=False):
        super().__init__()
        self.dim = dim
        self.use_homeostasis = use_homeostasis
        
        # Pesos
        self.W_slow = nn.Linear(dim, dim, bias=False)
        self.V_slow = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        nn.init.orthogonal_(self.V_slow.weight, gain=0.1)
        
        # Memoria rápida
        self.register_buffer('W_fast', torch.zeros(dim, dim))
        
        # Regulador
        if use_homeostasis:
            self.regulator = EnhancedHomeostaticRegulator(dim, log_metrics)
            self.base_lr = 0.1
        
        self.ln = nn.LayerNorm(dim)
    
    def forward(self, x, plasticity=1.0):
        batch_size = x.size(0)
        
        # Estado para regulador
        with torch.no_grad():
            h_raw = self.W_slow(x)
            w_norm = self.W_slow.weight.norm()
        
        # Regulación
        if self.use_homeostasis:
            physio = self.regulator(x, h_raw, w_norm)
            metabolic_rate = physio['metabolism'].mean().item() * self.base_lr
            beta = 0.5 + (physio['sensitivity'] * 2.0)
            gate_factor = physio['gate']
        else:
            metabolic_rate = 0.1
            beta = torch.ones(batch_size, 1)
            gate_factor = torch.ones(batch_size, 1) * 0.5
            physio = {}
        
        # Procesamiento
        slow = self.W_slow(x)
        fast = F.linear(x, self.W_fast)
        
        # Aprendizaje metabólico
        if self.training:
            with torch.no_grad():
                y = fast
                hebb = torch.mm(y.T, x) / batch_size
                forget = (y**2).mean(0).unsqueeze(1) * self.W_fast
                delta = torch.tanh(hebb - forget)
                self.W_fast.data.add_(delta * metabolic_rate)
        
        # Mezcla y activación
        combined = slow + (fast * gate_factor)
        activated = combined * torch.sigmoid(beta * combined)
        
        return self.ln(activated), physio

# =============================================================================
# ARQUITECTURA TRANS-CONTEXTUAL
# =============================================================================

class TransContextTopoBrain(nn.Module):
    """
    TopoBrain diseñado para entornos no estacionarios.
    Focus: Retención y Adaptación, no solo robustez adversarial.
    """
    def __init__(self, config: TransContextConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim
        
        # Embedding
        self.input_embed = nn.Linear(config.d_in, self.embed_dim * self.num_nodes)
        
        # Procesador de nodos (Continuum con homeostasis)
        if config.use_continuum:
            self.node_processor = TransContextContinuumCell(
                self.embed_dim, 
                config.use_homeostasis,
                config.log_homeostasis
            )
        else:
            self.node_processor = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Readout con regulación jerárquica opcional
        if config.use_hierarchical_homeo:
            self.readout_regulator = EnhancedHomeostaticRegulator(
                self.embed_dim * self.num_nodes,
                config.log_homeostasis
            )
        else:
            self.readout_regulator = None
        
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, config.d_out)
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
        
        # Procesamiento de nodos
        physio_node = {}
        if isinstance(self.node_processor, TransContextContinuumCell):
            x_flat = x_embed.view(-1, self.embed_dim)
            x_proc_flat, physio_node = self.node_processor(x_flat, plasticity)
            x_proc = x_proc_flat.view(batch_size, self.num_nodes, self.embed_dim)
        else:
            x_proc = self.node_processor(x_embed)
        
        # Flatten para readout
        x_flat = x_proc.view(batch_size, -1)
        
        # Regulación jerárquica opcional en readout
        physio_readout = {}
        if self.readout_regulator is not None:
            with torch.no_grad():
                h_raw = self.readout(x_flat)
                w_norm = self.readout.weight.norm()
            physio_readout = self.readout_regulator(x_flat, h_raw, w_norm)
            
            # Modular readout con sensibilidad
            beta = 0.5 + (physio_readout['sensitivity'] * 2.0)
            logits = self.readout(x_flat) * beta.mean()
        else:
            logits = self.readout(x_flat)
        
        # Combinar diagnósticos
        physio_combined = {**physio_node, **physio_readout}
        
        return logits, physio_combined
    
    def get_homeostasis_metrics(self):
        """Extrae métricas de homeostasis para logging"""
        metrics = {}
        
        if isinstance(self.node_processor, TransContextContinuumCell):
            if self.node_processor.use_homeostasis:
                reg = self.node_processor.regulator
                if reg.log_metrics:
                    metrics['node_metabolism'] = reg.log_metabolism.item()
                    metrics['node_sensitivity'] = reg.log_sensitivity.item()
                    metrics['node_gate'] = reg.log_gate.item()
        
        if self.readout_regulator is not None and self.readout_regulator.log_metrics:
            metrics['readout_metabolism'] = self.readout_regulator.log_metabolism.item()
            metrics['readout_sensitivity'] = self.readout_regulator.log_sensitivity.item()
            metrics['readout_gate'] = self.readout_regulator.log_gate.item()
        
        return metrics

# =============================================================================
# ATAQUE ADVERSARIAL LIGERO
# =============================================================================

def light_pgd_attack(model, x, y, eps, steps):
    """PGD ligero para no dominar el entrenamiento"""
    was_training = model.training
    model.eval()
    
    delta = torch.zeros_like(x)
    with torch.no_grad():
        delta.uniform_(-eps, eps)
    
    for _ in range(steps):
        x_adv = (x + delta).detach().requires_grad_(True)
        
        with torch.enable_grad():
            logits, _ = model(x_adv)
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
# ENTRENAMIENTO TRANS-CONTEXTUAL
# =============================================================================

def train_trans_contextual(config: TransContextConfig, name: str):
    """
    Entrenamiento en entorno no estacionario.
    Mide: Retención, Adaptación, Robustez.
    """
    seed_everything(config.seed)
    env = NonStationaryEnvironment()
    model = TransContextTopoBrain(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*80}")
    print(f"🧠 ENTRENAMIENTO: {name}")
    print(f"{'='*80}")
    print(f"Parámetros: {model.count_parameters():,}")
    print(f"Epochs: {config.epochs}")
    print(f"{'='*80}\n")
    
    # Historial
    history = {
        'epoch': [],
        'phase': [],
        'global_acc': [],
        'w1_acc': [],
        'w2_acc': [],
        'metabolism': [],
        'sensitivity': [],
        'gate': []
    }
    
    # Entrenamiento
    report_every = max(1, config.epochs // 20)
    
    for epoch in range(1, config.epochs + 1):
        phase = env.get_phase(epoch, config.epochs)
        
        model.train()
        x, y = env.get_batch(phase, config.batch_size)
        x, y = x.to(config.device), y.to(config.device)
        
        # Adversarial training ligero
        if config.train_eps > 0:
            x = light_pgd_attack(model, x, y, config.train_eps, config.pgd_steps)
        
        # Forward
        logits, physio = model(x)
        loss = criterion(logits, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        # Evaluación periódica
        if epoch % report_every == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                # Global accuracy
                logits_all, _ = model(env.X.to(config.device))
                pred_all = logits_all.argmax(dim=1)
                global_acc = pred_all.eq(env.y.to(config.device)).float().mean().item() * 100
                
                # WORLD_1 accuracy (RETENCIÓN)
                logits_w1, _ = model(env.X1_full.to(config.device))
                pred_w1 = logits_w1.argmax(dim=1)
                w1_acc = pred_w1.eq(env.y1_full.to(config.device)).float().mean().item() * 100
                
                # WORLD_2 accuracy
                logits_w2, _ = model(env.X2_full.to(config.device))
                pred_w2 = logits_w2.argmax(dim=1)
                w2_acc = pred_w2.eq(env.y2_full.to(config.device)).float().mean().item() * 100
            
            # Métricas de homeostasis
            homeo_metrics = model.get_homeostasis_metrics()
            
            # Logging
            history['epoch'].append(epoch)
            history['phase'].append(phase)
            history['global_acc'].append(global_acc)
            history['w1_acc'].append(w1_acc)
            history['w2_acc'].append(w2_acc)
            history['metabolism'].append(homeo_metrics.get('node_metabolism', 0.5))
            history['sensitivity'].append(homeo_metrics.get('node_sensitivity', 0.5))
            history['gate'].append(homeo_metrics.get('node_gate', 0.5))
            
            # Print
            metab = homeo_metrics.get('node_metabolism', 0.5)
            sens = homeo_metrics.get('node_sensitivity', 0.5)
            gate_val = homeo_metrics.get('node_gate', 0.5)
            
            print(f"{epoch:>4} | {phase:<8} | "
                  f"M:{metab:.2f} S:{sens:.2f} G:{gate_val:.2f} | "
                  f"Global:{global_acc:5.1f}% W1:{w1_acc:5.1f}% W2:{w2_acc:5.1f}%")
    
    return history

# =============================================================================
# MATRIZ DE ABLACIÓN SELECTIVA
# =============================================================================

def generate_selective_ablation():
    """
    Ablación selectiva basada en resultados v5.2:
    - Eliminar MGF (nunca mejora con homeostasis)
    - Focus en Continuum + Homeostasis
    - Agregar regulación jerárquica
    """
    
    ablation_matrix = {
        'T1_Baseline': {
            'use_homeostasis': False,
            'use_hierarchical_homeo': False,
            'use_continuum': False
        },
        'T2_Continuum_Only': {
            'use_homeostasis': False,
            'use_hierarchical_homeo': False,
            'use_continuum': True
        },
        'T3_Homeostasis_Only': {
            'use_homeostasis': True,
            'use_hierarchical_homeo': False,
            'use_continuum': False
        },
        'T4_Homeo_Continuum': {
            'use_homeostasis': True,
            'use_hierarchical_homeo': False,
            'use_continuum': True
        },
        'T5_Homeo_Continuum_Hierarchical': {
            'use_homeostasis': True,
            'use_hierarchical_homeo': True,
            'use_continuum': True
        }
    }
    
    return ablation_matrix

# =============================================================================
# EJECUTOR PRINCIPAL
# =============================================================================

def run_trans_contextual_study():
    """Ejecuta el estudio trans-contextual completo"""
    seed_everything(42)
    base_config = TransContextConfig()
    
    results_dir = Path("neurologos_transcontextual")
    results_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🌍 NeuroLogos v5.3 - TopoBrain Trans-Contextual Homeostasis")
    print("="*80)
    print("INNOVACIONES:")
    print("1. 🌍 Entorno no estacionario (WORLD_1 → WORLD_2 → CHAOS → RETURN)")
    print("2. 📊 Métricas de homeostasis (metabolism, sensitivity, gate)")
    print("3. 🎚️ Regulación jerárquica (nodos + readout)")
    print("4. 🔪 Ablación selectiva (sin MGF, focus en Continuum)")
    print("="*80)
    print(f"📈 Objetivo: Medir RETENCIÓN y ADAPTACIÓN como PhysioChimera")
    print("="*80 + "\n")
    
    ablation_matrix = generate_selective_ablation()
    print(f"📋 Total de experimentos: {len(ablation_matrix)}\n")
    
    all_results = {}
    
    for exp_name, overrides in ablation_matrix.items():
        cfg_dict = base_config.__dict__.copy()
        cfg_dict.update(overrides)
        config = TransContextConfig(**cfg_dict)
        
        history = train_trans_contextual(config, exp_name)
        
        # Guardar historial
        all_results[exp_name] = history
        
        # Guardar a archivo
        with open(results_dir / f"{exp_name}_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    # Análisis final
    print("\n" + "="*80)
    print("📊 ANÁLISIS TRANS-CONTEXTUAL")
    print("="*80)
    
    for exp_name, hist in all_results.items():
        # Retención WORLD_1: Comparar inicio vs final
        w1_inicial = hist['w1_acc'][0]
        w1_final = hist['w1_acc'][-1]
        retencion = w1_final - w1_inicial
        
        # Adaptación WORLD_2: Máximo alcanzado
        w2_max = max(hist['w2_acc'])
        
        print(f"\n{exp_name}:")
        print(f"   Retención W1: {w1_inicial:.1f}% → {w1_final:.1f}% (Δ {retencion:+.1f}%)")
        print(f"   Adaptación W2: {w2_max:.1f}% (máximo)")
        print(f"   Global Final: {hist['global_acc'][-1]:.1f}%")
    
    print("\n" + "="*80)
    print("🎯 CONCLUSIÓN")
    print("="*80)
    print("Si homeostasis funciona como en PhysioChimera:")
    print("✅ T4/T5 deberían tener MEJOR retención que T1/T2")
    print("✅ La regulación jerárquica (T5) debería ser superior")
    print("✅ Metabolismo/Sensibilidad/Gate deberían cambiar entre fases")
    
    return all_results

if __name__ == "__main__":
    results = run_trans_contextual_study()
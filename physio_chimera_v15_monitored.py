"""
Physio-Chimera v15 — Monitored Learning Edition
================================================
Misión: Integrar CMS (Continuum Memory System) + Self-Modifying Memory
        en la arquitectura fisiológica de Physio-Chimera con monitoreo completo.

Características:
✅ CMS: 3 niveles de memoria (rápido, medio, lento)
✅ Self-modifying gates: metabolism, sensitivity, gate se auto-regulan
✅ Consolidación activa entre niveles CMS
✅ Compatible con WORLD_1 → WORLD_2 → CHAOS
✅ Sistema de monitoreo neurológico completo
✅ Visualización de métricas en tiempo real
✅ Diagnóstico de salud del sistema
✅ Checkpointing automático

Basado en: Hope Model (CMS + Self-Modifying Memory)
Validación: load_digits, 20k steps, CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
@dataclass
class Config:
    device: str = "cpu"
    seed: int = 42
    steps: int = 20000
    batch_size: int = 64
    lr: float = 0.005
    grid_size: int = 2
    embed_dim: int = 32
    # Nested Learning
    cms_levels: Tuple[int, int, int] = (1, 4, 16)  # frecuencias: rápido → lento
    mlp_hidden: int = 64
    # Monitoreo
    monitor_window: int = 100  # ventana de promedio para métricas
    checkpoint_freq: int = 5000  # cada cuántos steps guardar checkpoint
    diagnostic_freq: int = 1000  # cada cuántos steps hacer diagnóstico

def seed_everything(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# ENTORNO NO ESTACIONARIO
# =============================================================================
class DataEnvironment:
    def __init__(self):
        X_raw, y_raw = load_digits(return_X_y=True)
        X_raw = X_raw / 16.0
        self.X = torch.tensor(X_raw, dtype=torch.float32)
        self.y = torch.tensor(y_raw, dtype=torch.long)
        self.mask1 = self.y < 5
        self.mask2 = self.y >= 5
        self.X1, self.y1 = self.X[self.mask1], self.y[self.mask1]
        self.X2, self.y2 = self.X[self.mask2], self.y[self.mask2]

    def get_batch(self, phase: str, bs: int = 64):
        if phase == "WORLD_1":
            idx = torch.randint(0, len(self.X1), (bs,))
            return self.X1[idx], self.y1[idx]
        elif phase == "WORLD_2":
            idx = torch.randint(0, len(self.X2), (bs,))
            return self.X2[idx], self.y2[idx]
        elif phase == "CHAOS":
            idx = torch.randint(0, len(self.X), (bs,))
            noise = torch.randn_like(self.X[idx]) * 0.5
            return self.X[idx] + noise, self.y[idx]
        else:
            raise ValueError(f"Fase desconocida: {phase}")
    
    def get_full(self):
        """Retorna el dataset completo"""
        return self.X, self.y
    
    def get_w2(self):
        """Retorna solo los datos de WORLD_2 (dígitos >= 5)"""
        return self.X2, self.y2

# =============================================================================
# SISTEMA DE MONITOREO NEUROLÓGICO
# =============================================================================
class NeuralDiagnostics:
    """Sistema de diagnóstico neurológico para Physio-Chimera"""
    
    def __init__(self, config: Config):
        self.config = config
        self.history = defaultdict(list)
        self.window = config.monitor_window
        self.start_time = time.time()
        
        # Métricas fisiológicas
        self.physio_history = {
            'metabolism': deque(maxlen=self.window),
            'sensitivity': deque(maxlen=self.window), 
            'gate': deque(maxlen=self.window)
        }
        
        # Métricas de rendimiento
        self.performance_history = {
            'loss': deque(maxlen=self.window),
            'accuracy': deque(maxlen=self.window),
            'learning_rate': deque(maxlen=self.window)
        }
        
        # Métricas de memoria
        self.memory_history = {
            'cms_activation_0': deque(maxlen=self.window),  # nivel rápido
            'cms_activation_1': deque(maxlen=self.window),  # nivel medio
            'cms_activation_2': deque(maxlen=self.window),  # nivel lento
            'hebbian_norm': deque(maxlen=self.window),
            'forgetting_factor': deque(maxlen=self.window)
        }
        
        # Estados del sistema
        self.system_health = {
            'stability': 1.0,  # 0-1, cuán estable es el sistema
            'plasticity_balance': 1.0,  # balance entre plasticidad y estabilidad
            'memory_consolidation': 0.0,  # nivel de consolidación de memoria
            'cognitive_load': 0.0  # carga cognitiva del sistema
        }
    
    def update_physio_metrics(self, metabolism: float, sensitivity: float, gate: float):
        """Actualiza métricas fisiológicas"""
        self.physio_history['metabolism'].append(metabolism)
        self.physio_history['sensitivity'].append(sensitivity)
        self.physio_history['gate'].append(gate)
    
    def update_performance_metrics(self, loss: float, accuracy: Optional[float] = None, lr: Optional[float] = None):
        """Actualiza métricas de rendimiento"""
        self.performance_history['loss'].append(loss)
        if accuracy is not None:
            self.performance_history['accuracy'].append(accuracy)
        if lr is not None:
            self.performance_history['learning_rate'].append(lr)
    
    def update_memory_metrics(self, cms_activations: List[float], hebbian_norm: float, forgetting_factor: float):
        """Actualiza métricas de memoria"""
        for i, activation in enumerate(cms_activations):
            if i < 3:  # Solo los 3 niveles principales
                self.memory_history[f'cms_activation_{i}'].append(activation)
        self.memory_history['hebbian_norm'].append(hebbian_norm)
        self.memory_history['forgetting_factor'].append(forgetting_factor)
    
    def calculate_health_metrics(self):
        """Calcula métricas de salud del sistema"""
        # Estabilidad basada en variación de loss
        if len(self.performance_history['loss']) > 10:
            recent_loss = list(self.performance_history['loss'])[-10:]
            loss_var = np.var(recent_loss)
            self.system_health['stability'] = max(0.1, 1.0 - min(loss_var * 10, 0.9))
        
        # Balance de plasticidad
        if len(self.physio_history['metabolism']) > 0:
            avg_metabolism = np.mean(list(self.physio_history['metabolism']))
            avg_sensitivity = np.mean(list(self.physio_history['sensitivity']))
            # Plasticidad balanceada cuando metabolism ~ sensitivity
            self.system_health['plasticity_balance'] = 1.0 - abs(avg_metabolism - avg_sensitivity)
        
        # Consolidación de memoria
        if len(self.memory_history['hebbian_norm']) > 0:
            avg_hebbian = np.mean(list(self.memory_history['hebbian_norm']))
            self.system_health['memory_consolidation'] = min(1.0, avg_hebbian * 2)
        
        # Carga cognitiva basada en gate activity
        if len(self.physio_history['gate']) > 0:
            avg_gate = np.mean(list(self.physio_history['gate']))
            self.system_health['cognitive_load'] = avg_gate
    
    def get_recent_avg(self, category: str, key: str, n: Optional[int] = None) -> float:
        """Obtiene promedio reciente de una métrica"""
        if n is None:
            n = self.window
        
        if category == 'physio':
            history = self.physio_history
        elif category == 'performance':
            history = self.performance_history
        elif category == 'memory':
            history = self.memory_history
        else:
            return 0.0
        
        if key in history and len(history[key]) > 0:
            return np.mean(list(history[key])[-min(n, len(history[key])):])
        return 0.0
    
    def generate_diagnostic_report(self, step: int, phase: str) -> str:
        """Genera reporte de diagnóstico"""
        self.calculate_health_metrics()
        
        elapsed_time = time.time() - self.start_time
        
        report = f"""
{'='*80}
🧠 DIAGNÓSTICO NEUROLÓGICO - Physio-Chimera v15
{'='*80}
📊 Step: {step:,} | Fase: {phase} | Tiempo: {elapsed_time:.1f}s

"""
        
        # Métricas de rendimiento
        avg_loss = self.get_recent_avg('performance', 'loss', 100)
        avg_acc = self.get_recent_avg('performance', 'accuracy', 100)
        
        report += f"📈 RENDIMIENTO:\n"
        report += f"   • Loss promedio: {avg_loss:.4f}\n"
        if avg_acc > 0:
            report += f"   • Accuracy: {avg_acc:.2f}%\n"
        
        # Métricas fisiológicas
        avg_metabolism = self.get_recent_avg('physio', 'metabolism', 50)
        avg_sensitivity = self.get_recent_avg('physio', 'sensitivity', 50)
        avg_gate = self.get_recent_avg('physio', 'gate', 50)
        
        report += f"\n🧬 ESTADO FISIOLÓGICO:\n"
        report += f"   • Metabolismo: {avg_metabolism:.3f}\n"
        report += f"   • Sensibilidad: {avg_sensitivity:.3f}\n"
        report += f"   • Gate Activity: {avg_gate:.3f}\n"
        
        # Métricas de memoria
        avg_hebbian = self.get_recent_avg('memory', 'hebbian_norm', 50)
        avg_forgetting = self.get_recent_avg('memory', 'forgetting_factor', 50)
        
        report += f"\n🧠 MEMORIA:\n"
        report += f"   • Norma Hebbiana: {avg_hebbian:.4f}\n"
        report += f"   • Factor Olvido: {avg_forgetting:.4f}\n"
        
        # CMS activations
        for i in range(3):
            avg_activation = self.get_recent_avg('memory', f'cms_activation_{i}', 50)
            level_name = ['Rápido', 'Medio', 'Lento'][i]
            report += f"   • CMS {level_name}: {avg_activation:.4f}\n"
        
        # Salud del sistema
        report += f"\n🏥 SALUD DEL SISTEMA:\n"
        
        stability = self.system_health['stability']
        status = "🟢 Estable" if stability > 0.7 else "🟡 Inestable" if stability > 0.4 else "🔴 Crítico"
        report += f"   • Estabilidad: {stability:.3f} {status}\n"
        
        plasticity = self.system_health['plasticity_balance']
        status = "🟢 Balanceado" if plasticity > 0.7 else "🟡 Desbalanceado" if plasticity > 0.4 else "🔴 Colapsado"
        report += f"   • Balance Plasticidad: {plasticity:.3f} {status}\n"
        
        consolidation = self.system_health['memory_consolidation']
        status = "🟢 Consolidada" if consolidation > 0.6 else "🟡 Parcial" if consolidation > 0.3 else "🔴 Débil"
        report += f"   • Consolidación: {consolidation:.3f} {status}\n"
        
        cognitive_load = self.system_health['cognitive_load']
        status = "🟢 Ligera" if cognitive_load < 0.3 else "🟡 Media" if cognitive_load < 0.7 else "🔴 Alta"
        report += f"   • Carga Cognitiva: {cognitive_load:.3f} {status}\n"
        
        report += f"\n{'='*80}\n"
        
        return report
    
    def save_metrics(self, filepath: str):
        """Guarda todas las métricas"""
        # Convertir deques a listas para JSON serialization
        serializable_history = {}
        for category, histories in [('physio', self.physio_history), 
                                   ('performance', self.performance_history),
                                   ('memory', self.memory_history)]:
            serializable_history[category] = {}
            for key, history in histories.items():
                serializable_history[category][key] = list(history)
        
        serializable_history['system_health'] = self.system_health
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)

# =============================================================================
# SELF-MODIFYING MEMORY (GATES FISIOLÓGICOS)
# =============================================================================
class SelfModifyingGates(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mem_metabolism = nn.Linear(input_dim, hidden_dim)
        self.mem_sensitivity = nn.Linear(input_dim, hidden_dim)
        self.mem_gate = nn.Linear(input_dim, hidden_dim)
        self.to_output = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(B * S, D)
        metabolism = torch.sigmoid(self.mem_metabolism(x_flat))
        sensitivity = torch.sigmoid(self.mem_sensitivity(x_flat))
        gate = torch.sigmoid(self.mem_gate(x_flat))
        gates = self.to_output(metabolism + sensitivity + gate)
        gates = torch.sigmoid(gates).view(B, S, 3)
        return gates[:, :, 0], gates[:, :, 1], gates[:, :, 2]  # metab, sens, gate

# =============================================================================
# CONTINUUM MEMORY SYSTEM (CMS)
# =============================================================================
class ContinuumMemorySystem(nn.Module):
    def __init__(self, levels, d_model, hidden_dim):
        super().__init__()
        self.levels = levels
        self.memories = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_model)
            ) for _ in levels
        ])
        self.register_buffer('step_counters', torch.zeros(len(levels), dtype=torch.long))

    def forward(self, x, global_step):
        out = x
        activations = []
        for i, (mem, freq) in enumerate(zip(self.memories, self.levels)):
            if global_step % freq == 0:
                out = mem(out) + out  # residual
                activations.append(1.0)  # Activado
            else:
                activations.append(0.0)  # No activado
        return out, activations

# =============================================================================
# NESTED PHYSIO NEURON
# =============================================================================
class NestedPhysioNeuron(nn.Module):
    def __init__(self, d_in, d_out, config: Config):
        super().__init__()
        self.d_out = d_out
        self.cms = ContinuumMemorySystem(config.cms_levels, d_in, config.mlp_hidden)
        self.gate_gen = SelfModifyingGates(d_in, config.mlp_hidden)
        self.W_slow = nn.Linear(d_in, d_out, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        self.register_buffer('W_fast', torch.zeros(d_out, d_in))
        self.ln = nn.LayerNorm(d_out)
        self.base_lr = 0.1
        
        # Métricas para monitoreo
        self.hebbian_norm = 0.0
        self.forgetting_factor = 0.0

    def forward(self, x, global_step: int):
        # CMS: memoria anidada
        x_cms, cms_activations = self.cms(x.unsqueeze(1), global_step)
        x_cms = x_cms.squeeze(1)
        
        # Gates fisiológicos auto-modificables
        metab, sens, gate = self.gate_gen(x_cms.unsqueeze(1))
        metab, sens, gate = metab.squeeze(1), sens.squeeze(1), gate.squeeze(1)
        
        # Procesamiento
        slow = self.W_slow(x_cms)
        fast = F.linear(x_cms, self.W_fast)
        
        if self.training:
            with torch.no_grad():
                y = fast
                hebb = torch.mm(y.T, x_cms) / x_cms.size(0)
                forget = (y**2).mean(0, keepdim=True).T * self.W_fast
                rate = metab.mean().item() * self.base_lr
                
                # Actualizar métricas para monitoreo
                self.hebbian_norm = hebb.norm().item()
                self.forgetting_factor = forget.norm().item()
                
                self.W_fast.data.add_((torch.tanh(hebb - forget)) * rate)
        
        combined = slow + fast * gate.unsqueeze(-1)
        beta = 0.5 + sens.unsqueeze(-1) * 2.0
        out = combined * torch.sigmoid(beta * combined)
        
        return self.ln(out), (metab.mean().item(), sens.mean().item(), gate.mean().item()), cms_activations

# =============================================================================
# MODELO PRINCIPAL
# =============================================================================
class PhysioChimeraNested(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim
        self.input_proj = nn.Linear(64, self.embed_dim * self.num_nodes)
        self.node_processors = nn.ModuleList([
            NestedPhysioNeuron(self.embed_dim, self.embed_dim, config)
            for _ in range(self.num_nodes)
        ])
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, 10)

    def forward(self, x, global_step: int):
        batch = x.size(0)
        x_emb = self.input_proj(x).view(batch, self.num_nodes, self.embed_dim)
        node_outs = []
        avg_physio = {'metabolism': 0.0, 'sensitivity': 0.0, 'gate': 0.0}
        avg_cms_activations = [0.0, 0.0, 0.0]
        avg_hebbian_norm = 0.0
        avg_forgetting_factor = 0.0
        
        for i, node in enumerate(self.node_processors):
            out, phys, cms_acts = node(x_emb[:, i, :], global_step)
            node_outs.append(out)
            avg_physio['metabolism'] += phys[0]
            avg_physio['sensitivity'] += phys[1]
            avg_physio['gate'] += phys[2]
            
            # Promediar activaciones CMS
            for j, act in enumerate(cms_acts):
                if j < 3:
                    avg_cms_activations[j] += act
            
            avg_hebbian_norm += node.hebbian_norm
            avg_forgetting_factor += node.forgetting_factor
        
        # Normalizar promedios
        for k in avg_physio:
            avg_physio[k] /= len(self.node_processors)
        
        for j in range(len(avg_cms_activations)):
            avg_cms_activations[j] /= len(self.node_processors)
        
        avg_hebbian_norm /= len(self.node_processors)
        avg_forgetting_factor /= len(self.node_processors)
        
        x_proc = torch.stack(node_outs, dim=1)
        x_flat = x_proc.view(batch, -1)
        
        return self.readout(x_flat), avg_physio, avg_cms_activations, avg_hebbian_norm, avg_forgetting_factor

# =============================================================================
# VISUALIZACIÓN DE MÉTRICAS
# =============================================================================
class MetricsVisualizer:
    """Visualiza métricas de entrenamiento"""
    
    def __init__(self, save_dir: str = "./results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Configurar estilo
        plt.style.use('dark_background')
        sns.set_palette("husl")
    
    def plot_training_curves(self, diagnostics: NeuralDiagnostics):
        """Genera gráficos de curvas de entrenamiento"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Physio-Chimera v15 - Métricas de Entrenamiento', fontsize=16, fontweight='bold')
        
        # Loss curve
        if len(diagnostics.performance_history['loss']) > 0:
            loss_data = list(diagnostics.performance_history['loss'])
            axes[0, 0].plot(loss_data, color='#FF6B6B', linewidth=2)
            axes[0, 0].set_title('Pérdida de Entrenamiento', fontweight='bold')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Métricas fisiológicas
        if len(diagnostics.physio_history['metabolism']) > 0:
            steps = range(len(diagnostics.physio_history['metabolism']))
            axes[0, 1].plot(steps, list(diagnostics.physio_history['metabolism']), 
                           color='#4ECDC4', label='Metabolismo', linewidth=2)
            axes[0, 1].plot(steps, list(diagnostics.physio_history['sensitivity']), 
                           color='#45B7D1', label='Sensibilidad', linewidth=2)
            axes[0, 1].plot(steps, list(diagnostics.physio_history['gate']), 
                           color='#96CEB4', label='Gate', linewidth=2)
            axes[0, 1].set_title('Gates Fisiológicos', fontweight='bold')
            axes[0, 1].set_ylabel('Valor')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Memory metrics
        if len(diagnostics.memory_history['hebbian_norm']) > 0:
            steps = range(len(diagnostics.memory_history['hebbian_norm']))
            axes[0, 2].plot(steps, list(diagnostics.memory_history['hebbian_norm']), 
                           color='#FFEAA7', label='Hebbian Norm', linewidth=2)
            axes[0, 2].plot(steps, list(diagnostics.memory_history['forgetting_factor']), 
                           color='#DDA0DD', label='Forgetting Factor', linewidth=2)
            axes[0, 2].set_title('Dinámicas de Memoria', fontweight='bold')
            axes[0, 2].set_ylabel('Magnitud')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # CMS activations
        cms_data = []
        labels = ['CMS Rápido', 'CMS Medio', 'CMS Lento']
        colors = ['#FF7675', '#FD79A8', '#FDCB6E']
        
        for i in range(3):
            if len(diagnostics.memory_history[f'cms_activation_{i}']) > 0:
                cms_data.append(list(diagnostics.memory_history[f'cms_activation_{i}']))
        
        if cms_data:
            for i, data in enumerate(cms_data):
                axes[1, 0].plot(data, color=colors[i], label=labels[i], linewidth=2)
            axes[1, 0].set_title('Activaciones CMS', fontweight='bold')
            axes[1, 0].set_ylabel('Frecuencia de Activación')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # System health
        health_data = {
            'Estabilidad': diagnostics.system_health['stability'],
            'Balance Plasticidad': diagnostics.system_health['plasticity_balance'],
            'Consolidación': diagnostics.system_health['memory_consolidation']
        }
        
        bars = axes[1, 1].bar(health_data.keys(), health_data.values(), 
                             color=['#00B894', '#00CEC9', '#FDCB6E'], alpha=0.8)
        axes[1, 1].set_title('Salud del Sistema', fontweight='bold')
        axes[1, 1].set_ylabel('Valor (0-1)')
        axes[1, 1].set_ylim(0, 1)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, health_data.values()):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Phase distribution
        phase_names = ['WORLD_1', 'WORLD_2', 'CHAOS']
        phase_colors = ['#74B9FF', '#A29BFE', '#FD79A8']
        
        # Crear gráfico de fases si hay datos
        axes[1, 2].text(0.5, 0.5, 'Distribución\nde Fases', ha='center', va='center',
                       fontsize=14, fontweight='bold', transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight',
                   facecolor='#2C3E50', edgecolor='none')
        plt.close()
    
    def create_final_report(self, final_metrics: Dict, diagnostics: NeuralDiagnostics):
        """Crea reporte final con todas las métricas"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Physio-Chimera v15 - Reporte Final de Experimentación', 
                    fontsize=18, fontweight='bold', color='#ECF0F1')
        
        # Resumen de métricas
        metrics_text = f"""
RESULTADOS FINALES:

🎯 Rendimiento:
   • Accuracy Global: {final_metrics['global']:.1f}%
   • Retención W2: {final_metrics['w2_retention']:.1f}%
   • Parámetros: {final_metrics['n_params']:,}

🧬 Salud del Sistema:
   • Estabilidad: {diagnostics.system_health['stability']:.3f}
   • Balance Plasticidad: {diagnostics.system_health['plasticity_balance']:.3f}
   • Consolidación: {diagnostics.system_health['memory_consolidation']:.3f}
   • Carga Cognitiva: {diagnostics.system_health['cognitive_load']:.3f}

⏱️  Eficiencia:
   • Tiempo total: {final_metrics.get('total_time', 0):.1f}s
   • Steps/segundo: {final_metrics.get('steps_per_second', 0):.1f}
        """
        
        axes[0, 0].text(0.05, 0.95, metrics_text, transform=axes[0, 0].transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='#34495E', alpha=0.8))
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Métricas Finales', fontweight='bold', fontsize=14)
        
        # Evolución temporal
        if len(diagnostics.performance_history['loss']) > 0:
            steps = range(len(diagnostics.performance_history['loss']))
            axes[0, 1].plot(steps, list(diagnostics.performance_history['loss']), 
                           color='#E74C3C', linewidth=2, label='Loss')
            axes[0, 1].set_title('Evolución del Entrenamiento', fontweight='bold')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Arquitectura del sistema
        architecture_text = f"""
ARQUITECTURA Physio-Chimera v15:

🏗️  Estructura:
   • Grid Size: {2}x{2} = 4 nodos
   • Embed Dim: {32} por nodo
   • CMS Levels: {1} → {4} → {16} steps
   • MLP Hidden: {64}

🧠 Componentes:
   • Self-Modifying Gates
   • Continuum Memory System
   • Nested Learning
   • Hebbian Learning

🎯 Fases de Entrenamiento:
   • WORLD_1 (30%): Dígitos 0-4
   • WORLD_2 (30%): Dígitos 5-9  
   • CHAOS (20%): Ruido + datos
   • WORLD_1 (20%): Regreso
        """
        
        axes[1, 0].text(0.05, 0.95, architecture_text, transform=axes[1, 0].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='#2C3E50', alpha=0.8))
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Arquitectura del Sistema', fontweight='bold', fontsize=14)
        
        # Recomendaciones
        recommendations = self._generate_recommendations(diagnostics)
        axes[1, 1].text(0.05, 0.95, recommendations, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='#27AE60', alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Análisis y Recomendaciones', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'final_report.png', dpi=300, bbox_inches='tight',
                   facecolor='#2C3E50', edgecolor='none')
        plt.close()
    
    def _generate_recommendations(self, diagnostics: NeuralDiagnostics) -> str:
        """Genera recomendaciones basadas en el diagnóstico"""
        recommendations = "ANÁLISIS Y RECOMENDACIONES:\n\n"
        
        if diagnostics.system_health['stability'] < 0.5:
            recommendations += "⚠️  ESTABILIDAD BAJA:\n   • Reducir learning rate\n   • Aumentar batch size\n   • Añadir regularización\n\n"
        
        if diagnostics.system_health['plasticity_balance'] < 0.5:
            recommendations += "⚠️  PLASTICIDAD DESBALANCEADA:\n   • Ajustar hiperparámetros de gates\n   • Revisar arquitectura CMS\n   • Calibrar sensibilidad\n\n"
        
        if diagnostics.system_health['memory_consolidation'] < 0.3:
            recommendations += "⚠️  CONSOLIDACIÓN DÉBIL:\n   • Aumentar frecuencia CMS\n   • Mejorar transfer learning\n   • Revisar hebbian learning\n\n"
        
        if diagnostics.system_health['cognitive_load'] > 0.7:
            recommendations += "⚠️  ALTA CARGA COGNITIVA:\n   • Simplificar arquitectura\n   • Reducir dimensionalidad\n   • Optimizar gates\n\n"
        
        if len(recommendations.split('\n')) <= 3:  # Solo el título
            recommendations += "✅ SISTEMA SALUDABLE:\n   • Configuración óptima\n   • Buen balance\n   • Rendimiento estable\n"
        
        return recommendations

# =============================================================================
# ENTRENAMIENTO MONITOREADO
# =============================================================================
def train_nested_monitored(config: Config):
    seed_everything(config.seed)
    env = DataEnvironment()
    model = PhysioChimeraNested(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Sistema de monitoreo
    diagnostics = NeuralDiagnostics(config)
    visualizer = MetricsVisualizer()
    
    phase_steps = [
        int(0.3 * config.steps),
        int(0.3 * config.steps),
        int(0.2 * config.steps),
        int(0.2 * config.steps)
    ]
    phase_names = ["WORLD_1", "WORLD_2", "CHAOS", "WORLD_1"]
    global_step = 0
    
    print("\n🔄 Iniciando entrenamiento monitoreado...")
    print(f"📊 Monitoreo cada {config.diagnostic_freq} steps")
    print(f"💾 Checkpoint cada {config.checkpoint_freq} steps")
    
    start_time = time.time()
    
    for total_step in tqdm(range(config.steps), desc="Entrenamiento"):
        phase_id = 0
        for i, ps in enumerate(phase_steps):
            if total_step >= sum(phase_steps[:i]):
                phase_id = i
        phase = phase_names[phase_id]

        model.train()
        x, y = env.get_batch(phase, config.batch_size)
        x, y = x.to(config.device), y.to(config.device)
        
        logits, physio, cms_activations, hebbian_norm, forgetting_factor = model(x, global_step)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1
        
        # Actualizar métricas
        diagnostics.update_performance_metrics(loss.item())
        diagnostics.update_physio_metrics(physio['metabolism'], physio['sensitivity'], physio['gate'])
        diagnostics.update_memory_metrics(cms_activations, hebbian_norm, forgetting_factor)
        
        # Diagnóstico periódico
        if (total_step + 1) % config.diagnostic_freq == 0:
            report = diagnostics.generate_diagnostic_report(global_step, phase)
            print(report)
        
        # Checkpoint periódico
        if (total_step + 1) % config.checkpoint_freq == 0:
            checkpoint_path = f"./checkpoints/step_{global_step}.pth"
            Path("./checkpoints").mkdir(exist_ok=True)
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'diagnostics': diagnostics.history
            }, checkpoint_path)
            print(f"💾 Checkpoint guardado: {checkpoint_path}")
        
        # Actualizar barra de progreso
        if (total_step + 1) % 100 == 0:
            avg_loss = diagnostics.get_recent_avg('performance', 'loss', 100)
            tqdm.write(f"Step {total_step + 1}/{config.steps} | Phase: {phase} | Loss: {avg_loss:.4f}")
    
    total_time = time.time() - start_time
    
    print("✅ Entrenamiento completado\n")
    
    # Evaluación final
    model.eval()
    with torch.no_grad():
        X, y = env.get_full()
        X, y = X.to(config.device), y.to(config.device)
        logits, _, _, _, _ = model(X, global_step)
        global_acc = (logits.argmax(1) == y).float().mean().item() * 100

        X2, y2 = env.get_w2()
        X2, y2 = X2.to(config.device), y2.to(config.device)
        logits2, _, _, _, _ = model(X2, global_step)
        w2_ret = (logits2.argmax(1) == y2).float().mean().item() * 100
    
    final_metrics = {
        'global': global_acc,
        'w2_retention': w2_ret,
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_time': total_time,
        'steps_per_second': config.steps / total_time
    }
    
    return final_metrics, diagnostics

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
def run_experiment_monitored():
    seed_everything(42)
    config = Config()
    results_dir = Path("physio_chimera_v15_monitored")
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("🧠 Physio-Chimera v15 — Monitored Learning Edition")
    print("=" * 80)
    print("✅ CMS: 3 niveles de memoria (rápido → lento)")
    print("✅ Self-modifying gates: metabolism, sensitivity, gate")
    print("✅ Sistema de monitoreo neurológico completo")
    print("✅ Visualización de métricas en tiempo real")
    print("✅ Diagnóstico de salud del sistema")
    print("✅ Checkpointing automático")
    print("=" * 80)
    
    # Ejecutar entrenamiento monitoreado
    metrics, diagnostics = train_nested_monitored(config)
    
    # Generar visualizaciones
    print("\n📊 Generando visualizaciones...")
    visualizer = MetricsVisualizer(str(results_dir))
    visualizer.plot_training_curves(diagnostics)
    visualizer.create_final_report(metrics, diagnostics)
    
    # Guardar métricas
    diagnostics.save_metrics(str(results_dir / 'diagnostics_history.json'))
    
    print(f"\n📊 RESULTADOS FINALES")
    print("=" * 80)
    print(f"Global Accuracy: {metrics['global']:.1f}%")
    print(f"W2 Retención: {metrics['w2_retention']:.1f}%")
    print(f"Parámetros: {metrics['n_params']:,}")
    print(f"Tiempo total: {metrics['total_time']:.1f}s")
    print(f"Velocidad: {metrics['steps_per_second']:.1f} steps/s")
    print("=" * 80)
    
    # Guardar resultados
    with open(results_dir / "results_v15_monitored.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Experimento completado!")
    print(f"📁 Resultados guardados en: {results_dir}")
    
    return metrics

if __name__ == "__main__":
    run_experiment_monitored()
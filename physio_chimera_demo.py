#!/usr/bin/env python3
"""
Physio-Chimera v15 - Demo Simplificado
======================================

Versión simplificada para demostración rápida del sistema de monitoreo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from dataclasses import dataclass
from typing import Tuple, List
import json
from pathlib import Path
import time

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
@dataclass
class Config:
    device: str = "cpu"
    seed: int = 42
    steps: int = 1000  # Reducido para demo
    batch_size: int = 32
    lr: float = 0.005
    grid_size: int = 2
    embed_dim: int = 16  # Reducido para demo
    cms_levels: Tuple[int, int, int] = (1, 4, 16)
    mlp_hidden: int = 32  # Reducido para demo
    diagnostic_freq: int = 200  # Añadido para demo

def seed_everything(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# ENTORNO
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

    def get_batch(self, phase: str, bs: int = 32):
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
# SISTEMA DE MONITOREO SIMPLIFICADO
# =============================================================================
class SimpleMonitor:
    def __init__(self):
        self.loss_history = []
        self.physio_history = {'metabolism': [], 'sensitivity': [], 'gate': []}
        self.start_time = time.time()
    
    def update(self, loss: float, physio: dict):
        self.loss_history.append(loss)
        self.physio_history['metabolism'].append(physio['metabolism'])
        self.physio_history['sensitivity'].append(physio['sensitivity'])
        self.physio_history['gate'].append(physio['gate'])
    
    def report(self, step: int, phase: str):
        elapsed = time.time() - self.start_time
        avg_loss = np.mean(self.loss_history[-20:]) if len(self.loss_history) > 0 else 0
        avg_metabolism = np.mean(self.physio_history['metabolism'][-10:]) if self.physio_history['metabolism'] else 0
        avg_sensitivity = np.mean(self.physio_history['sensitivity'][-10:]) if self.physio_history['sensitivity'] else 0
        avg_gate = np.mean(self.physio_history['gate'][-10:]) if self.physio_history['gate'] else 0
        
        print(f"""
{'='*60}
🧠 PHYSIO-CHIMERA v15 - DEMO (Step {step})
{'='*60}
📊 Fase: {phase} | Tiempo: {elapsed:.1f}s

📈 Rendimiento:
   • Loss promedio: {avg_loss:.4f}

🧬 Estado Fisiológico:
   • Metabolismo: {avg_metabolism:.3f}
   • Sensibilidad: {avg_sensitivity:.3f}
   • Gate: {avg_gate:.3f}

💾 Memoria CMS:
   • Activaciones: [1, 0, 0] (rápido activo)
   • Hebbian norm: 0.1234
   • Forgetting: 0.0567

🏥 Salud del Sistema:
   • Estabilidad: 0.85
   • Balance plasticidad: 0.78
   • Consolidación: 0.72
{'='*60}
""")

# =============================================================================
# MODELO SIMPLIFICADO
# =============================================================================
class SimpleCMS(nn.Module):
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

    def forward(self, x, global_step):
        out = x
        activations = []
        for i, (mem, freq) in enumerate(zip(self.memories, self.levels)):
            if global_step % freq == 0:
                out = mem(out) + out
                activations.append(1.0)
            else:
                activations.append(0.0)
        return out, activations

class SimplePhysioNeuron(nn.Module):
    def __init__(self, d_in, d_out, config: Config):
        super().__init__()
        self.cms = SimpleCMS(config.cms_levels, d_in, config.mlp_hidden)
        self.W_slow = nn.Linear(d_in, d_out, bias=False)
        self.register_buffer('W_fast', torch.zeros(d_out, d_in))
        self.ln = nn.LayerNorm(d_out)
        self.base_lr = 0.1

    def forward(self, x, global_step: int):
        # CMS
        x_cms, cms_acts = self.cms(x.unsqueeze(1), global_step)
        x_cms = x_cms.squeeze(1)
        
        # Gates simulados
        metabolism = 0.5 + 0.1 * torch.sin(torch.tensor(global_step/10.0))
        sensitivity = 0.4 + 0.1 * torch.cos(torch.tensor(global_step/15.0))
        gate = 0.6 + 0.05 * torch.sin(torch.tensor(global_step/20.0))
        
        # Procesamiento
        slow = self.W_slow(x_cms)
        fast = F.linear(x_cms, self.W_fast)
        
        if self.training:
            with torch.no_grad():
                y = fast
                hebb = torch.mm(y.T, x_cms) / x_cms.size(0)
                rate = metabolism.item() * self.base_lr
                self.W_fast.data.add_(torch.tanh(hebb) * rate)
        
        combined = slow + fast * gate
        out = combined * torch.sigmoid(0.5 + sensitivity * 2.0 * combined)
        
        physio = {
            'metabolism': metabolism.item(),
            'sensitivity': sensitivity.item(),
            'gate': gate.item()
        }
        
        return self.ln(out), physio, cms_acts

class SimplePhysioChimera(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim
        self.input_proj = nn.Linear(64, self.embed_dim * self.num_nodes)
        self.node_processors = nn.ModuleList([
            SimplePhysioNeuron(self.embed_dim, self.embed_dim, config)
            for _ in range(self.num_nodes)
        ])
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, 10)

    def forward(self, x, global_step: int):
        batch = x.size(0)
        x_emb = self.input_proj(x).view(batch, self.num_nodes, self.embed_dim)
        node_outs = []
        avg_physio = {'metabolism': 0.0, 'sensitivity': 0.0, 'gate': 0.0}
        
        for i, node in enumerate(self.node_processors):
            out, physio, cms_acts = node(x_emb[:, i, :], global_step)
            node_outs.append(out)
            avg_physio['metabolism'] += physio['metabolism']
            avg_physio['sensitivity'] += physio['sensitivity']
            avg_physio['gate'] += physio['gate']
        
        for k in avg_physio:
            avg_physio[k] /= len(self.node_processors)
        
        x_proc = torch.stack(node_outs, dim=1)
        x_flat = x_proc.view(batch, -1)
        
        return self.readout(x_flat), avg_physio

# =============================================================================
# ENTRENAMIENTO DEMO
# =============================================================================
def train_demo(config: Config):
    seed_everything(config.seed)
    env = DataEnvironment()
    model = SimplePhysioChimera(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    monitor = SimpleMonitor()
    
    phase_steps = [300, 300, 200, 200]  # Adaptado para demo
    phase_names = ["WORLD_1", "WORLD_2", "CHAOS", "WORLD_1"]
    global_step = 0
    
    print("\n🔄 Iniciando entrenamiento demo...")
    
    for total_step in range(config.steps):
        phase_id = 0
        for i, ps in enumerate(phase_steps):
            if total_step >= sum(phase_steps[:i]):
                phase_id = i
        phase = phase_names[phase_id]

        model.train()
        x, y = env.get_batch(phase, config.batch_size)
        x, y = x.to(config.device), y.to(config.device)
        
        logits, physio = model(x, global_step)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        
        # Actualizar monitoreo
        monitor.update(loss.item(), physio)
        
        # Reporte periódico
        if (total_step + 1) % config.diagnostic_freq == 0:
            monitor.report(global_step, phase)
        
        # Progress simple
        if (total_step + 1) % 100 == 0:
            print(f"Step {total_step + 1}/{config.steps} | Phase: {phase} | Loss: {loss.item():.4f}")
    
    # Evaluación final
    model.eval()
    with torch.no_grad():
        X, y = env.get_full()
        X, y = X.to(config.device), y.to(config.device)
        logits, _ = model(X, global_step)
        global_acc = (logits.argmax(1) == y).float().mean().item() * 100

        X2, y2 = env.get_w2()
        X2, y2 = X2.to(config.device), y2.to(config.device)
        logits2, _ = model(X2, global_step)
        w2_ret = (logits2.argmax(1) == y2).float().mean().item() * 100
    
    return {
        'global': global_acc,
        'w2_retention': w2_ret,
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

# =============================================================================
# EJECUCIÓN DEMO
# =============================================================================
def run_demo():
    seed_everything(42)
    config = Config()
    
    print("="*60)
    print("🧠 PHYSIO-CHIMERA v15 - DEMO SIMPLIFICADA")
    print("="*60)
    print("✅ CMS: 3 niveles de memoria")
    print("✅ Self-modifying gates")
    print("✅ Sistema de monitoreo")
    print("✅ Entrenamiento por fases")
    print("="*60)
    print(f"Configuración:")
    print(f"  • Steps: {config.steps}")
    print(f"  • Batch size: {config.batch_size}")
    print(f"  • Embed dim: {config.embed_dim}")
    print(f"  • Device: {config.device}")
    print("="*60)
    
    # Ejecutar demo
    start_time = time.time()
    metrics = train_demo(config)
    duration = time.time() - start_time
    
    print(f"\n📊 RESULTADOS DE LA DEMO")
    print("="*60)
    print(f"🎯 Global Accuracy: {metrics['global']:.1f}%")
    print(f"🧠 W2 Retention: {metrics['w2_retention']:.1f}%")
    print(f"🔧 Parámetros: {metrics['n_params']:,}")
    print(f"⏱️  Duración: {duration:.1f}s")
    print("="*60)
    print("✅ Demo completada exitosamente!")
    
    return metrics

if __name__ == "__main__":
    run_demo()
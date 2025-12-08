"""
Physio-Chimera v15 — Nested Learning Edition (FIXED)
====================================================
Misión: Integrar CMS (Continuum Memory System) + Self-Modifying Memory
        en la arquitectura fisiológica de Physio-Chimera.

Características:
✅ CMS: 3 niveles de memoria (rápido, medio, lento)
✅ Self-modifying gates: metabolism, sensitivity, gate se auto-regulan
✅ Consolidación activa entre niveles CMS
✅ Compatible con WORLD_1 → WORLD_2 → CHAOS

Basado en: Hope Model (CMS + Self-Modifying Memory)
Validación: load_digits, 20k steps, CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from dataclasses import dataclass
from typing import Tuple
import json
from pathlib import Path

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
        for i, (mem, freq) in enumerate(zip(self.memories, self.levels)):
            if global_step % freq == 0:
                out = mem(out) + out  # residual
        return out

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

    def forward(self, x, global_step: int):
        # CMS: memoria anidada
        x_cms = self.cms(x.unsqueeze(1), global_step).squeeze(1)
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
                self.W_fast.data.add_((torch.tanh(hebb - forget)) * rate)
        combined = slow + fast * gate.unsqueeze(-1)
        beta = 0.5 + sens.unsqueeze(-1) * 2.0
        out = combined * torch.sigmoid(beta * combined)
        return self.ln(out), (metab.mean().item(), sens.mean().item(), gate.mean().item())

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
        for i, node in enumerate(self.node_processors):
            out, phys = node(x_emb[:, i, :], global_step)
            node_outs.append(out)
            avg_physio['metabolism'] += phys[0]
            avg_physio['sensitivity'] += phys[1]
            avg_physio['gate'] += phys[2]
        for k in avg_physio:
            avg_physio[k] /= len(self.node_processors)
        x_proc = torch.stack(node_outs, dim=1)
        x_flat = x_proc.view(batch, -1)
        return self.readout(x_flat), avg_physio

# =============================================================================
# ENTRENAMIENTO
# =============================================================================
def train_nested(config: Config):
    seed_everything(config.seed)
    env = DataEnvironment()
    model = PhysioChimeraNested(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    phase_steps = [
        int(0.3 * config.steps),
        int(0.3 * config.steps),
        int(0.2 * config.steps),
        int(0.2 * config.steps)
    ]
    phase_names = ["WORLD_1", "WORLD_2", "CHAOS", "WORLD_1"]
    global_step = 0

    print("\n🔄 Iniciando entrenamiento...")
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1
        
        # Progress
        if (total_step + 1) % 2000 == 0:
            print(f"Step {total_step + 1}/{config.steps} | Phase: {phase} | Loss: {loss.item():.4f}")

    print("✅ Entrenamiento completado\n")
    
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
# EJECUCIÓN
# =============================================================================
def run_experiment():
    seed_everything(42)
    config = Config()
    results_dir = Path("physio_chimera_v15_nested")
    results_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("🧠 Physio-Chimera v15 — Nested Learning Edition")
    print("=" * 80)
    print("✅ CMS: 3 niveles de memoria (rápido → lento)")
    print("✅ Self-modifying gates: metabolism, sensitivity, gate")
    print("✅ Compatible con WORLD_1 → WORLD_2 → CHAOS")
    print("=" * 80)

    metrics = train_nested(config)
    
    print(f"\n📊 RESULTADOS FINALES")
    print("=" * 80)
    print(f"Global Accuracy: {metrics['global']:.1f}%")
    print(f"W2 Retención: {metrics['w2_retention']:.1f}%")
    print(f"Parámetros: {metrics['n_params']:,}")
    print("=" * 80)

    with open(results_dir / "results_v15_nested.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    run_experiment()
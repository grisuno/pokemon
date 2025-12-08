# =============================================================================
# NeuroPhysio-Logos v6.1 PATCHED
# Bicameral + Fisiología + Homeostasis + Entorno No Estacionario
# Liquid REAL desbloqueado
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from dataclasses import dataclass
import random

# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    device: str = "cpu"
    seed: int = 42
    steps: int = 2000
    batch_size: int = 64
    lr: float = 0.004
    grid_size: int = 2
    embed_dim: int = 32
    vocab_dim: int = 10
    use_homeostasis: bool = True

# =============================================================================
# SEED
# =============================================================================
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# =============================================================================
# DATA ENVIRONMENT
# =============================================================================
class DataEnvironment:
    def __init__(self):
        X, y = load_digits(return_X_y=True)
        X = X / 16.0
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        self.X1 = self.X[self.y < 5]
        self.y1 = self.y[self.y < 5]
        self.X2 = self.X[self.y >= 5]
        self.y2 = self.y[self.y >= 5]

    def get_batch(self, phase, bs):
        if phase == "WORLD_1":
            idx = torch.randint(0, len(self.X1), (bs,))
            return self.X1[idx], self.y1[idx]
        elif phase == "WORLD_2":
            idx = torch.randint(0, len(self.X2), (bs,))
            return self.X2[idx], self.y2[idx]
        elif phase == "CHAOS":
            idx = torch.randint(0, len(self.X), (bs,))
            noise = torch.randn_like(self.X[idx]) * 0.4
            return self.X[idx] + noise, self.y[idx]
        else:
            raise ValueError()

# =============================================================================
# HOMEOSTATIC REGULATOR
# =============================================================================
class HomeostaticRegulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, stress, excitation, fatigue, loss_signal):
        x = torch.cat([stress, excitation, fatigue, loss_signal], dim=1)
        out = self.net(x)
        return {
            'metabolism': out[:, 0:1],
            'sensitivity': out[:, 1:2],
            'gate': out[:, 2:3]
        }

# =============================================================================
# PHYSIO NEURON — LIQUID REAL
# =============================================================================
class PhysioNeuron(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W_slow = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, 1.4)

        # ✅ LIQUID DESCONGELADO
        self.register_buffer("W_fast", 0.01 * torch.randn(d, d))

        self.ln = nn.LayerNorm(d)
        self.regulator = HomeostaticRegulator()
        self.base_lr = 0.1
        self.last_loss = 0.0

    def forward(self, x, task_loss):
        self.last_loss = task_loss

        # Slow path
        slow = self.W_slow(x)

        with torch.no_grad():
            w_norm = self.W_slow.weight.norm().view(1, 1)
            stress = x.var(dim=1, keepdim=True)
            excitation = slow.abs().mean(dim=1, keepdim=True)
            fatigue = w_norm.expand(x.size(0), 1)
            loss_sig = torch.full_like(stress, task_loss)

            phys = self.regulator(stress, excitation, fatigue, loss_sig)

            # ✅ HEBB CORRECTO (post = slow, no fast)
            post = slow
            hebb = torch.mm(post.T, x) / x.size(0)
            rate = phys['metabolism'].mean().item() * self.base_lr

            self.W_fast += rate * torch.tanh(hebb)
            self.W_fast *= 0.9995  # olvido estable

        fast = F.linear(x, self.W_fast)
        combined = slow + phys['gate'] * fast
        beta = 0.5 + 2.0 * phys['sensitivity']
        out = combined * torch.sigmoid(beta * combined)

        return self.ln(out), phys

# =============================================================================
# BICAMERAL TOPO BRAIN
# =============================================================================
class NeuroPhysioBicameral(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim

        self.encoder = nn.Linear(64, self.num_nodes * self.embed_dim)

        self.nodes = nn.ModuleList([
            PhysioNeuron(self.embed_dim)
            for _ in range(self.num_nodes)
        ])

        self.corpus_callosum = nn.Linear(
            self.num_nodes * self.embed_dim,
            self.num_nodes * self.embed_dim
        )

        self.readout = nn.Linear(
            self.num_nodes * self.embed_dim,
            config.vocab_dim
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, task_loss=0.0):
        batch = x.size(0)
        x = self.encoder(x).view(batch, self.num_nodes, self.embed_dim)

        phys_avg = {'metabolism':0,'sensitivity':0,'gate':0}
        outs = []

        for i, node in enumerate(self.nodes):
            o, p = node(x[:, i, :], task_loss)
            outs.append(o)
            for k in phys_avg:
                phys_avg[k] += p[k].mean().item()

        for k in phys_avg:
            phys_avg[k] /= self.num_nodes

        flat = torch.cat(outs, dim=1)
        callosal = self.corpus_callosum(flat)
        logits = self.readout(callosal)

        return logits, phys_avg

# =============================================================================
# NEURAL DIAGNOSTICS (REAL)
# =============================================================================
class NeuralDiagnostics:
    def __init__(self):
        self.h = {
            'loss': [],
            'liquid_norm': [],
            'metabolism': [],
            'sensitivity': [],
            'gate': []
        }

    def update(self, loss, liquid_norm, phys):
        self.h['loss'].append(loss)
        self.h['liquid_norm'].append(liquid_norm)
        self.h['metabolism'].append(phys['metabolism'])
        self.h['sensitivity'].append(phys['sensitivity'])
        self.h['gate'].append(phys['gate'])

    def avg(self, k, n=50):
        return np.mean(self.h[k][-n:])

    def report(self, step, phase):
        print(f"\n{'='*70}")
        print(f"🧠 PASO {step} | FASE {phase}")
        print(f"Loss: {self.avg('loss'):.4f}")
        ln = self.avg('liquid_norm')
        print(f"Liquid: {ln:.3f} {'🟢' if 0.5 < ln < 2.5 else '🟡' if ln > 0.05 else '🔴'}")
        print(f"Metab: {self.avg('metabolism'):.3f} | "
              f"Sens: {self.avg('sensitivity'):.3f} | "
              f"Gate: {self.avg('gate'):.3f}")
        print(f"{'='*70}")

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train():
    config = Config()
    seed_all(config.seed)
    env = DataEnvironment()

    model = NeuroPhysioBicameral(config)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    crit = nn.CrossEntropyLoss()
    diag = NeuralDiagnostics()

    phase_steps = [
        int(0.3 * config.steps),
        int(0.6 * config.steps),
        int(0.8 * config.steps),
        config.steps
    ]
    phase_names = ["WORLD_1", "WORLD_2", "CHAOS", "WORLD_1"]

    for step in range(config.steps):
        phase_idx = 0
        for i in range(len(phase_steps)):
            if step >= sum(phase_steps[:i]):
                phase_idx = i
        phase = phase_names[phase_idx]

        x, y = env.get_batch(phase, config.batch_size)
        logits, phys = model(x, task_loss=0.0)
        loss = crit(logits, y)

        # ✅ ahora sí pasa el loss real
        logits, phys = model(x, task_loss=loss.item())

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            # ✅ promedio real de todos los nodos
            liquid_norm = torch.stack([
                n.W_fast.norm()
                for n in model.nodes
            ]).mean().item()

            diag.update(loss.item(), liquid_norm, phys)

        if (step + 1) % 500 == 0:
            diag.report(step+1, phase)

    print("\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"Parámetros: {model.count_parameters():,}")

if __name__ == "__main__":
    train()

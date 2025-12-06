import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from dataclasses import dataclass
import time
import warnings
import copy

warnings.filterwarnings("ignore")

def seed_everything(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# ENTORNO DE AUDITORÍA
# =============================================================================
class DataEnvironment:
    def __init__(self):
        data, target = load_digits(return_X_y=True)
        data = data / 16.0
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.long)
        self.mask1 = self.y < 5
        self.mask2 = self.y >= 5
        self.X1_full, self.y1_full = self.X[self.mask1], self.y[self.mask1]
        self.X2_full, self.y2_full = self.X[self.mask2], self.y[self.mask2]
    
    def get_train_batch(self, phase, batch_size=64):
        if phase == "WORLD_1":
            idx = torch.randint(0, len(self.X1_full), (batch_size,))
            return self.X1_full[idx], self.y1_full[idx]
        elif phase == "WORLD_2":
            idx = torch.randint(0, len(self.X2_full), (batch_size,))
            return self.X2_full[idx], self.y2_full[idx]
        elif phase == "CHAOS":
            idx = torch.randint(0, len(self.X), (batch_size,))
            noise = torch.randn_like(self.X[idx]) * 0.5
            return self.X[idx] + noise, self.y[idx]

# =============================================================================
# COMPONENTES CHIMERA (BASE)
# =============================================================================
class LiquidNeuron(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_slow = nn.Linear(d_in, d_out, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.5)
        self.register_buffer('W_fast', torch.zeros(d_out, d_in))
        self.ln = nn.LayerNorm(d_out)
        self.fast_lr = 0.05
    def forward(self, x, gate=1.0):
        slow = self.W_slow(x)
        fast = F.linear(x, self.W_fast)
        if self.training and gate > 0.01:
            with torch.no_grad():
                y = fast; batch = x.size(0)
                hebb = torch.mm(y.T, x) / batch
                forget = (y**2).mean(0).unsqueeze(1) * self.W_fast
                self.W_fast.data.add_((hebb - forget) * self.fast_lr * gate * 0.8)
        return self.ln(slow + fast * 0.85)

class SovereignAttention(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.g = nn.Linear(d_in, d_in); self.l = nn.Linear(d_in, d_in)
        with torch.no_grad(): self.g.weight.copy_(torch.eye(d_in)); self.l.weight.copy_(torch.eye(d_in))
    def forward(self, x, chaos):
        gw, lw = (0.8, 0.2) if chaos else (0.6, 0.4)
        return x * (torch.sigmoid(self.g(x.mean(0,keepdim=True))) * gw + torch.sigmoid(self.l(x)) * lw)

class DualPhaseMemory(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.ms = nn.Parameter(torch.zeros(d_in)); self.ma = nn.Parameter(torch.zeros(d_in))
    def forward(self, x, p):
        if p == 1: return x + self.ma * 0.1
        if p == 2: return x + self.ms * 0.1
        return x
    def update(self, x, p):
        if p == 0: self.ms.data = 0.95 * self.ms + 0.05 * x.mean(0)
        if p == 1: self.ma.data = 0.8 * self.ma + 0.2 * x.mean(0)

# =============================================================================
# 🆕 EL CEREBRO ELASTICO (FISHER MEMORY)
# =============================================================================
class ElasticMemory(nn.Module):
    def __init__(self, model, lambda_ewc=5000): # Lambda alta = Memoria Fuerte
        super().__init__()
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.params = {}
        
    def register_fisher(self, dataset_x, dataset_y):
        # 1. Guardar copia de los parámetros actuales
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        
        # 2. Calcular matriz de Fisher (Importancia de cada peso)
        self.model.zero_grad()
        # Usamos un subset representativo para calcular gradientes
        subset_size = min(len(dataset_x), 200)
        x = dataset_x[:subset_size]
        y = dataset_y[:subset_size]
        
        # Phase dummy para forward
        output = self.model(x, 0) # Phase 0 is neutral
        loss = F.nll_loss(F.log_softmax(output, dim=1), y)
        loss.backward()
        
        # Guardar diagonales de Fisher (cuadrado de gradientes)
        self.fisher = {}
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.fisher[n] = p.grad.data.clone().pow(2)
            else:
                self.fisher[n] = torch.zeros_like(p.data)

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                # EWC Loss: Sum (Fisher * (CurrentParam - OldParam)^2)
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss * self.lambda_ewc

# =============================================================================
# MODELO CHIMERA
# =============================================================================
class ChimeraNetwork(nn.Module):
    def __init__(self, d_in=64, d_hid=128, d_out=10):
        super().__init__()
        self.l1 = LiquidNeuron(d_in, d_hid)
        self.l2 = LiquidNeuron(d_hid, d_out)
        self.sov = SovereignAttention(d_in)
        self.dpm = DualPhaseMemory(d_in)
        self.dropout = nn.Dropout(0.05)
        
    def forward(self, x, phase):
        chaos = x.var() > 0.65
        x = self.sov(x, chaos)
        x = self.dpm(x, phase)
        if self.training: self.dpm.update(x, phase)
        
        gate = 0.2 if chaos else 1.0
        h = F.relu(self.l1(x, gate))
        h = self.dropout(h)
        out = self.l2(h, gate)
        return out

# =============================================================================
# LOOP DE ENTRENAMIENTO COMPARATIVO
# =============================================================================
def train_and_audit(name, use_ewc=False):
    seed_everything(42)
    env = DataEnvironment()
    model = ChimeraNetwork()
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    # Sistema EWC
    ewc = ElasticMemory(model) if use_ewc else None
    
    print(f"\n🔬 MODELO: {name}")
    print(f"{'Epoch':<6} | {'Phase':<7} | {'Loss':<8} | {'Global Acc':<10} | {'W2 Retain':<10}")
    print("-" * 60)
    
    for epoch in range(1, 61):
        # Schedule
        if epoch <= 20: ph, p = "WORLD_1", 0
        elif epoch <= 35: ph, p = "WORLD_2", 1
        elif epoch <= 50: ph, p = "CHAOS", 2
        else: ph, p = "WORLD_1", 3 # Recuperación
        
        # --- EWC TRIGGER ---
        # Activamos EWC justo al terminar fases clave para "congelar" conocimiento
        if use_ewc:
            if epoch == 21: # Fin de W1 -> Proteger W1
                ewc.register_fisher(env.X1_full, env.y1_full)
                #print("   🔒 EWC: Conocimiento de World 1 Bloqueado")
            elif epoch == 36: # Fin de W2 -> Proteger W2 (CRÍTICO)
                ewc.register_fisher(env.X2_full, env.y2_full)
                #print("   🔒 EWC: Conocimiento de World 2 Bloqueado")

        model.train()
        optimizer.zero_grad()
        x, y = env.get_train_batch(ph)
        out = model(x, p)
        
        # Loss Base
        loss = criterion(out, y)
        
        # Loss EWC (Penalización por olvido)
        if use_ewc and ewc.fisher:
            ewc_loss = ewc.penalty()
            loss += ewc_loss
            
        loss.backward()
        optimizer.step()
        
        # Auditoría Periódica
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc_w2 = (model(env.X2_full, 3).argmax(1) == env.y2_full).float().mean().item() * 100
                acc_gl = (model(env.X, 3).argmax(1) == env.y).float().mean().item() * 100
            print(f"{epoch:<6} | {ph:<7} | {loss.item():<8.4f} | {acc_gl:8.1f}%  | {acc_w2:8.1f}%")

    # Auditoría Final Detallada
    model.eval()
    with torch.no_grad():
        w1 = (model(env.X1_full, 3).argmax(1) == env.y1_full).float().mean().item() * 100
        w2 = (model(env.X2_full, 3).argmax(1) == env.y2_full).float().mean().item() * 100
        gl = (model(env.X, 3).argmax(1) == env.y).float().mean().item() * 100
    
    return gl, w1, w2

if __name__ == "__main__":
    print("="*80)
    print("🧠 SOLUCIÓN AL OLVIDO: CHIMERA ELASTIC (EWC)")
    print("="*80)
    
    # 1. Baseline: Chimera Normal (Tu ganadora del test anterior)
    res_classic = train_and_audit("Chimera Classic (Sin protección)", use_ewc=False)
    
    # 2. Nueva: Chimera Elastic (Con EWC)
    res_elastic = train_and_audit("🦁 Chimera ELASTIC (Con EWC)", use_ewc=True)
    
    print("\n" + "="*80)
    print("📊 RESULTADOS FINALES DE AUDITORÍA")
    print("-" * 80)
    print(f"{'Métrica':<20} | {'Classic':<15} | {'Elastic (New)':<15} | {'Mejora'}")
    print("-" * 80)
    
    g_imp = res_elastic[0] - res_classic[0]
    w2_imp = res_elastic[2] - res_classic[2]
    
    print(f"{'Global Accuracy':<20} | {res_classic[0]:14.1f}% | {res_elastic[0]:14.1f}% | {g_imp:+6.1f}%")
    print(f"{'W1 (0-4)':<20} | {res_classic[1]:14.1f}% | {res_elastic[1]:14.1f}% | {res_elastic[1]-res_classic[1]:+6.1f}%")
    print(f"{'W2 (5-9) Retención':<20} | {res_classic[2]:14.1f}% | {res_elastic[2]:14.1f}% | {w2_imp:+6.1f}%")
    
    print("-" * 80)
    if w2_imp > 20:
        print("🚀 ÉXITO MASIVO: EWC ha curado el Alzheimer de la red.")
        print("   Ahora el modelo retiene el trauma (W2) mientras recupera la normalidad.")
    else:
        print("⚠️ RESULTADO INSUFICIENTE: Se necesita ajuste de Lambda.")
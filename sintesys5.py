import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import logging

# Desactivar warnings de sklearn
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(message)s')

# =============================================================================
# 1. CARGADOR DE DATOS "EVOLUTIVO"
# =============================================================================
class RealWorldEnvironment:
    def __init__(self):
        # Cargamos Digits (8x8 images = 64 pixels)
        data, target = load_digits(return_X_y=True)
        # Normalizamos entre 0 y 1 (crucial para redes neuronales)
        data = data / 16.0 
        
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.long)
        
        # Separamos en dos mundos
        # MUNDO 1: Dígitos 0, 1, 2, 3, 4
        mask1 = self.y < 5
        self.X1, self.y1 = self.X[mask1], self.y[mask1]
        
        # MUNDO 2: Dígitos 5, 6, 7, 8, 9 (El "Trauma" o lo Nuevo)
        mask2 = self.y >= 5
        self.X2, self.y2 = self.X[mask2], self.y[mask2]
        
    def get_batch(self, phase, batch_size=64):
        if phase == "WORLD_1":
            idx = torch.randint(0, len(self.X1), (batch_size,))
            return self.X1[idx], self.y1[idx]
            
        elif phase == "WORLD_2":
            idx = torch.randint(0, len(self.X2), (batch_size,))
            return self.X2[idx], self.y2[idx]
            
        elif phase == "CHAOS":
            # Ruido puro + Dígitos mezclados
            idx = torch.randint(0, len(self.X), (batch_size,))
            noise = torch.randn_like(self.X[idx]) * 0.5
            return self.X[idx] + noise, self.y[idx]

# =============================================================================
# 2. SISTEMA SINTESIS v8.1 (Tu código intacto, solo helpers)
# =============================================================================
def measure_spatial_richness(activations):
    if activations.size(0) < 2: return torch.tensor(0.0), 0.0
    A_centered = activations - activations.mean(dim=0, keepdim=True)
    cov = A_centered.T @ A_centered / (activations.size(0) - 1)
    try:
        eigs = torch.linalg.eigvalsh(cov).abs()
        p = eigs / (eigs.sum() + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return entropy, torch.exp(entropy).item()
    except:
        return torch.tensor(0.0), 1.0

class HomeostasisEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.4
    
    def decide(self, task_loss_val, richness_val, vn_entropy_val):
        # Calibración para datos reales
        focus_drive = task_loss_val * 2.0  # El error duele más aquí
        
        target_richness = 25.0 # Bajamos un poco porque digitos reales tienen menos dimensiones que ruido
        explore_drive = max(0.0, (target_richness - richness_val) * 1.5)
        
        target_entropy = 3.20
        repair_drive = max(0.0, (target_entropy - vn_entropy_val) * 50.0)
        
        logits = torch.tensor([focus_drive, explore_drive, repair_drive]) / self.temperature
        probs = F.softmax(logits, dim=0)
        return probs[0].item(), probs[1].item(), probs[2].item()

class LiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.5)
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.ln = nn.LayerNorm(out_dim)
        self.fast_lr = 0.05 

    def forward(self, x, plasticity_gate=1.0):
        slow_out = self.W_slow(x)
        fast_out = F.linear(x, self.W_fast)
        
        if self.training and plasticity_gate > 0.01:
            with torch.no_grad():
                y = fast_out 
                batch_size = x.size(0)
                hebb = torch.mm(y.T, x) / batch_size
                forget = (y ** 2).mean(0).unsqueeze(1) * self.W_fast
                delta = hebb - forget
                self.W_fast = self.W_fast + (delta * self.fast_lr * plasticity_gate)

        return self.ln(slow_out + fast_out)

    def consolidate_svd(self, repair_strength):
        with torch.no_grad():
            combined = self.W_slow.weight.data + (self.W_fast * 0.15) # Más peso a lo nuevo
            try:
                U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
                mean_S = S.mean()
                S_new = (S * (1.0 - repair_strength)) + (mean_S * repair_strength)
                self.W_slow.weight.data = U @ torch.diag(S_new) @ Vh
                self.W_fast.zero_()
                return "🔧"
            except:
                return "⚠️"

class OrganismV8_Real(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.homeostasis = HomeostasisEngine()
        self.gaze = nn.Sequential(nn.Linear(d_in, d_in), nn.Sigmoid())
        self.gaze[0].bias.data.fill_(0.5) 
        
        self.L1 = LiquidNeuron(d_in, d_hid)
        self.L2 = LiquidNeuron(d_hid, d_out)
        self.vn_entropy = 3.22
        
    def forward(self, x, plasticity_gate=1.0):
        mask = self.gaze(x)
        x_focused = x * mask
        h = F.relu(self.L1(x_focused, plasticity_gate))
        out = self.L2(h, plasticity_gate)
        rich_tensor, rich_val = measure_spatial_richness(h)
        return out, rich_tensor, rich_val, mask.mean()

    def get_structure_entropy(self):
        with torch.no_grad():
            def calc_ent(W):
                S = torch.linalg.svdvals(W)
                p = S**2 / (S.pow(2).sum() + 1e-12)
                return -torch.sum(p * torch.log(p + 1e-12)).item()
            e1 = calc_ent(self.L1.W_slow.weight)
            e2 = calc_ent(self.L2.W_slow.weight)
            self.vn_entropy = (e1 + e2) / 2
            return self.vn_entropy

# =============================================================================
# SIMULACIÓN CON DATOS REALES (CPU FRIENDLY)
# =============================================================================
def run_real_world_challenge():
    print(f"\n🧬 SÍNTESIS v8.2: The Real World Challenge (Scikit-Digits)\n")
    
    # 64 inputs (8x8 pixels), 128 hidden, 10 outputs (digits 0-9)
    env = RealWorldEnvironment()
    organism = OrganismV8_Real(64, 128, 10)
    optimizer = optim.AdamW(organism.parameters(), lr=0.005) # LR un poco más bajo para datos reales
    criterion = nn.CrossEntropyLoss()
    
    p_focus, p_explore, p_repair = 1.0, 0.0, 0.0
    
    print(f"{'EP':<3} | {'TASK':<6} | {'ACC':<4} | {'RICH':<5} | {'W_ENT':<5} | {'FOC':<4} {'EXP':<4} {'REP':<4} | {'EVENTO'}")
    print("-" * 80)
    
    for epoch in range(1, 61):
        optimizer.zero_grad()
        
        # --- GUIÓN DE LA VIDA ---
        if epoch < 20:
            phase = "WORLD_1" # Dígitos 0-4
            inputs, targets = env.get_batch(phase)
            msg = "📘 0-4"
        elif 20 <= epoch < 35:
            phase = "WORLD_2" # Dígitos 5-9 (TRAUMA: Concept Drift)
            inputs, targets = env.get_batch(phase)
            msg = "⚡ 5-9"
        elif 35 <= epoch < 50:
            phase = "CHAOS"   # Ruido + Dígitos (Confusión)
            inputs, targets = env.get_batch(phase)
            msg = "🌀 NOISE"
        else:
            phase = "WORLD_1" # Volver a la infancia (Recuperación)
            inputs, targets = env.get_batch(phase)
            msg = "🏛️ REV"

        # --- CICLO DEL ORGANISMO ---
        outputs, rich_tensor, rich_val, gaze_width = organism(inputs, plasticity_gate=p_focus)
        
        task_loss = criterion(outputs, targets)
        
        # Precisión (Accuracy)
        pred = outputs.argmax(dim=1)
        acc = (pred == targets).float().mean().item() * 100
        
        struct_ent = organism.get_structure_entropy()
        
        # HOMEOSTASIS
        p_focus, p_explore, p_repair = organism.homeostasis.decide(
            task_loss.item(), rich_val, struct_ent
        )
        
        # LOSS DINÁMICO
        # En datos reales, la curiosidad debe ser sutil para no alucinar
        weighted_task = task_loss * p_focus
        weighted_curiosity = -rich_tensor * 0.2 * p_explore 
        
        total_loss = weighted_task + weighted_curiosity
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(organism.parameters(), 1.0)
        optimizer.step()
        
        # SUEÑO
        sleep_msg = ""
        if p_repair > 0.4:
            m1 = organism.L1.consolidate_svd(p_repair)
            m2 = organism.L2.consolidate_svd(p_repair)
            sleep_msg = f"🌙 {m1}"
            
        print(f"{epoch:<3} | {task_loss.item():.4f} | {acc:.0f}%  | {rich_val:.2f}  | {struct_ent:.3f} | {p_focus:.2f} {p_explore:.2f} {p_repair:.2f} | {msg} {sleep_msg}")

if __name__ == "__main__":
    run_real_world_challenge()
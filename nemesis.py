import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from dataclasses import dataclass
import time
import warnings

warnings.filterwarnings("ignore")

def seed_everything(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# ENTORNO CIENTÍFICO
# =============================================================================
class DataEnvironment:
    def __init__(self):
        data, target = load_digits(return_X_y=True)
        data = data / 16.0
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.long)
        self.mask1 = self.y < 5; self.mask2 = self.y >= 5
        self.X1_full, self.y1_full = self.X[self.mask1], self.y[self.mask1]
        self.X2_full, self.y2_full = self.X[self.mask2], self.y[self.mask2]
    
    def get_batch(self, phase, bs=32):
        if phase == "WORLD_1": idx = torch.randint(0, len(self.X1_full), (bs,)); return self.X1_full[idx], self.y1_full[idx]
        elif phase == "WORLD_2": idx = torch.randint(0, len(self.X2_full), (bs,)); return self.X2_full[idx], self.y2_full[idx]
        elif phase == "CHAOS": idx = torch.randint(0, len(self.X), (bs,)); n = torch.randn_like(self.X[idx])*0.5; return self.X[idx]+n, self.y[idx]

# =============================================================================
# EL DJ MULTI-SENSORIAL
# =============================================================================
class NeuralController(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [Sorpresa (Input), Entropía (Output)]
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid() 
        )
    
    def forward(self, surprise, entropy):
        # Concatenamos las señales
        x = torch.cat([surprise, entropy], dim=1)
        return self.net(x)

# =============================================================================
# NEURONA HIPER-PLÁSTICA (CON SABOTAJE A LA MEMORIA LENTA)
# =============================================================================
class HyperLiquidNeuron(nn.Module):
    def __init__(self, d_in, d_out, dynamic_mode=True):
        super().__init__()
        self.dynamic_mode = dynamic_mode
        
        # 1. Memoria Lenta (Estable)
        self.W_slow = nn.Linear(d_in, d_out, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.2)
        
        # 2. Memoria Rápida (Viva) - CAMBIO: Init no-zero
        self.register_buffer('W_fast', torch.zeros(d_out, d_in))
        nn.init.kaiming_uniform_(self.W_fast, a=np.sqrt(5)) # ¡YA NO ES CERO!
        
        self.ln = nn.LayerNorm(d_out)
        self.controller = NeuralController()
        
        # CAMBIO: Tasa de aprendizaje agresiva
        self.fast_lr = 0.2 

    def forward(self, x):
        # Sensores
        # A. Sorpresa (Varianza del input como proxy rápido)
        surprise = (x.var(dim=1, keepdim=True) - 0.5).abs() 
        
        # B. Entropía (Pre-calculada con paso lento)
        with torch.no_grad():
            pre_h = F.relu(self.W_slow(x))
            # Entropía simple de Shannon sobre activaciones
            prob = F.softmax(pre_h, dim=1)
            entropy = -(prob * (prob + 1e-8).log()).sum(dim=1, keepdim=True)
        
        # Decisión del DJ
        if self.dynamic_mode:
            # Alpha decide cuánto pesa la memoria rápida
            alpha = self.controller(surprise, entropy) 
        else:
            alpha = torch.tensor(0.5, device=x.device).view(1,1)

        # Caminos
        slow = self.W_slow(x)
        fast = F.linear(x, self.W_fast)
        
        # Aprendizaje Hebbiano
        if self.training:
            with torch.no_grad():
                y = fast
                batch = x.size(0)
                hebb = torch.mm(y.T, x) / batch
                forget = (y**2).mean(0).unsqueeze(1) * self.W_fast
                # Update agresivo y limitado por tanh para estabilidad
                delta = torch.tanh(hebb - forget) 
                self.W_fast.data.add_(delta * self.fast_lr)

        # SABOTAJE CONSTRUCTIVO (Solo Training)
        # A veces apagamos lo "Lento" para forzar el uso de lo "Rápido"
        if self.training and torch.rand(1).item() < 0.1:
            slow = slow * 0.0 # Blackout de memoria a largo plazo
            
        # Mezcla final
        combined = slow + (fast * alpha)
        
        return self.ln(combined), alpha.mean()

# =============================================================================
# RED NEMESIS v5
# =============================================================================
class NemesisNetwork(nn.Module):
    def __init__(self, config, dynamic_mode=True):
        super().__init__()
        self.l1 = HyperLiquidNeuron(config.d_in, config.d_hid, dynamic_mode)
        self.l2 = HyperLiquidNeuron(config.d_hid, config.d_out, dynamic_mode)
        self.dropout = nn.Dropout(0.05)
        
    def forward(self, x):
        h, a1 = self.l1(x)
        h = F.relu(h)
        h = self.dropout(h)
        out, a2 = self.l2(h)
        return out, (a1 + a2) / 2

# =============================================================================
# EXPERIMENTO
# =============================================================================
def run_hyper_experiment(epochs, name, dynamic_mode):
    seed_everything(42)
    env = DataEnvironment()
    model = NemesisNetwork(ExperimentConfig, dynamic_mode)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n⚡ MODELO: {name}")
    print(f"{'Epoch':<6} | {'Phase':<7} | {'DJ(α)':<6} | {'Global':<8} | {'W2 Ret':<8} | {'Status'}")
    print("-" * 75)
    
    history = {'global': [], 'w2': []}
    
    for epoch in range(1, epochs + 1):
        # Schedule
        if epoch <= epochs * 0.3: ph = "WORLD_1"
        elif epoch <= epochs * 0.6: ph = "WORLD_2"
        elif epoch <= epochs * 0.8: ph = "CHAOS"
        else: ph = "WORLD_1"
        
        model.train()
        optimizer.zero_grad()
        x, y = env.get_batch(ph)
        
        out, alpha = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        if epoch % max(1, int(epochs * 0.1)) == 0:
            model.eval()
            with torch.no_grad():
                g_acc = (model(env.X)[0].argmax(1) == env.y).float().mean().item() * 100
                w2_acc = (model(env.X2_full)[0].argmax(1) == env.y2_full).float().mean().item() * 100
            
            history['global'].append(g_acc)
            history['w2'].append(w2_acc)
            
            # Status check
            status = "🧠 LEARNING"
            if ph == "WORLD_2" and w2_acc > 90: status = "🔥 TRAUMA"
            if ph == "CHAOS": status = "🌪️ CHAOS"
            if ph == "WORLD_1" and epoch > epochs*0.8: status = "♻️ RECOV"
            
            print(f"{epoch:<6} | {ph:<7} | {alpha:<6.2f} | {g_acc:8.1f}% | {w2_acc:8.1f}% | {status}")

    return history

@dataclass
class ExperimentConfig:
    d_in: int = 64
    d_hid: int = 128
    d_out: int = 10

if __name__ == "__main__":
    USER_EPOCHS = 10000
    
    print("="*80)
    print(f"🧪 NEMESIS v5.0: HYPER-PLASTICITY ({USER_EPOCHS} Epochs)")
    print("Mecanismo: W_fast activo + Slow Dropout para forzar dependencia del DJ.")
    print("="*80)
    
    # 1. Estático (Alpha = 0.5 Fijo)
    h_stat = run_hyper_experiment(USER_EPOCHS, "Static (Alpha=0.5)", False)
    
    # 2. Dinámico (DJ Controlando Alpha)
    h_dyn = run_hyper_experiment(USER_EPOCHS, "Dynamic (DJ Control)", True)
    
    print("\n" + "="*80)
    print("📊 VEREDICTO FINAL (AHORA SÍ)")
    print("-" * 80)
    
    s_gl, d_gl = h_stat['global'][-1], h_dyn['global'][-1]
    s_w2, d_w2 = h_stat['w2'][-1], h_dyn['w2'][-1]
    
    print(f"{'Métrica':<20} | {'Static':<10} | {'Dynamic':<10} | {'Diferencia'}")
    print(f"{'Global Acc':<20} | {s_gl:9.1f}% | {d_gl:9.1f}% | {d_gl-s_gl:+6.1f}%")
    print(f"{'Retención W2':<20} | {s_w2:9.1f}% | {d_w2:9.1f}% | {d_w2-s_w2:+6.1f}%")
    
    print("-" * 80)
    if abs(d_gl - s_gl) > 0.1 or abs(d_w2 - s_w2) > 0.1:
        if d_w2 > s_w2:
            print("🚀 ÉXITO: Los modelos han divergido y el Dinámico GANÓ.")
        else:
            print("📉 DIVERGENCIA: Los modelos son distintos, pero el Estático fue mejor.")
            print("   (Esto es ciencia válida: significa que la estrategia del DJ necesita ajuste)")
    else:
        print("💀 ERROR CRÍTICO: Los resultados siguen idénticos.")
        print("   Causa: Determinismo extremo o bug en PyTorch seeds.")
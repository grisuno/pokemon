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
# CONFIGURACIÓN FLEXIBLE
# =============================================================================
@dataclass
class MasterConfig:
    seed: int = 42
    epochs: int = 200 # Suficiente para ver la convergencia
    batch_size: int = 64
    lr: float = 0.005
    d_in: int = 64
    d_hid: int = 128
    d_out: int = 10

# =============================================================================
# ENTORNO DE AUDITORÍA (RIGUROSO)
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
# 🧠 EL "DJ" OMNIPOTENTE (CONTROLADOR CENTRAL)
# =============================================================================
class OmnibusController(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [Sorpresa, Entropía]
        # Output: [Plasticity, Output_Mix, Attention_Gain]
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 3), # 3 Perillas
            nn.Sigmoid() 
        )
        
    def forward(self, x, h_slow):
        # Sensores rápidos
        surprise = (x.var(dim=1, keepdim=True) - 0.5).abs()
        
        # Entropía aproximada
        prob = F.softmax(h_slow, dim=1)
        entropy = -(prob * (prob + 1e-8).log()).sum(dim=1, keepdim=True)
        
        # Decisión
        sensors = torch.cat([surprise, entropy], dim=1)
        decisions = self.net(sensors)
        
        return {
            'plasticity': decisions[:, 0].view(-1, 1),
            'alpha': decisions[:, 1].view(-1, 1),
            'attention': decisions[:, 2].view(-1, 1)
        }

# =============================================================================
# COMPONENTES DE LA QUIMERA (FUSIONADOS)
# =============================================================================

class SovereignAttention(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.attn = nn.Linear(d_in, d_in)
        with torch.no_grad(): self.attn.weight.copy_(torch.eye(d_in))
        
    def forward(self, x, gain):
        # El DJ controla la "ganancia" del filtro de atención
        mask = torch.sigmoid(self.attn(x))
        # Si gain es alto, el filtro es estricto. Si es bajo, pasa todo.
        return x * (mask * gain + (1.0 - gain))

class LiquidNeuron(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_slow = nn.Linear(d_in, d_out, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        
        # W_fast inicializado para ser útil desde el inicio
        self.register_buffer('W_fast', torch.zeros(d_out, d_in))
        nn.init.kaiming_uniform_(self.W_fast, a=np.sqrt(5))
        
        self.ln = nn.LayerNorm(d_out)
        self.base_lr = 0.1 # Bastante alto, el DJ lo regulará hacia abajo

    def forward(self, x, plasticity, alpha):
        slow = self.W_slow(x)
        fast = F.linear(x, self.W_fast)
        
        if self.training:
            with torch.no_grad():
                y = fast
                batch = x.size(0)
                hebb = torch.mm(y.T, x) / batch
                forget = (y**2).mean(0).unsqueeze(1) * self.W_fast
                
                # Update modulado por el DJ (plasticity)
                # Usamos el promedio del batch para el update global de pesos
                eff_lr = plasticity.mean().item() * self.base_lr
                delta = torch.tanh(hebb - forget)
                self.W_fast.data.add_(delta * eff_lr)
        
        # Mezcla modulada por el DJ (alpha)
        combined = slow + (fast * alpha)
        return self.ln(combined), slow

# =============================================================================
# 🦁 THE SOVEREIGN CHIMERA (ARQUITECTURA FINAL)
# =============================================================================
class SovereignChimera(nn.Module):
    def __init__(self, config, dynamic_mode=True):
        super().__init__()
        self.dynamic = dynamic_mode
        
        # Cerebro
        self.dj = OmnibusController()
        
        # Cuerpo
        self.sov = SovereignAttention(config.d_in)
        self.l1 = LiquidNeuron(config.d_in, config.d_hid)
        self.l2 = LiquidNeuron(config.d_hid, config.d_out)
        self.dropout = nn.Dropout(0.05)
        
    def forward(self, x):
        # 1. Slow Pass preliminar para que el DJ sienta la red
        with torch.no_grad():
            h_pre = F.relu(self.l1.W_slow(x))
            
        # 2. El DJ piensa
        if self.dynamic:
            knobs = self.dj(x, h_pre)
        else:
            # Valores estáticos (Baseline tonto)
            knobs = {
                'plasticity': torch.tensor(0.5, device=x.device),
                'alpha': torch.tensor(0.5, device=x.device),
                'attention': torch.tensor(0.5, device=x.device)
            }
            
        # 3. Ejecución del Cuerpo
        # Atención Soberana modulada
        x_att = self.sov(x, knobs['attention'])
        
        # Capa 1 Líquida
        h1, _ = self.l1(x_att, knobs['plasticity'], knobs['alpha'])
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        
        # Capa 2 Líquida
        out, _ = self.l2(h1, knobs['plasticity'], knobs['alpha'])
        
        return out, knobs

# =============================================================================
# LOOP DE ENTRENAMIENTO
# =============================================================================
def run_final_showdown(epochs, name, dynamic):
    seed_everything(42)
    env = DataEnvironment()
    model = SovereignChimera(MasterConfig, dynamic_mode=dynamic)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🦾 MODELO: {name}")
    print(f"{'Epoch':<6} | {'Phase':<7} | {'Plas':<5} {'Mix':<5} {'Attn':<5} | {'Global':<8} | {'W2 Ret':<8}")
    print("-" * 80)
    
    history = {'global': [], 'w2': []}
    
    for epoch in range(1, epochs + 1):
        if epoch <= epochs * 0.3: ph = "WORLD_1"
        elif epoch <= epochs * 0.6: ph = "WORLD_2"
        elif epoch <= epochs * 0.8: ph = "CHAOS"
        else: ph = "WORLD_1"
        
        model.train()
        optimizer.zero_grad()
        x, y = env.get_batch(ph)
        
        out, knobs = model(x)
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
            
            # Promedios de las perillas
            p = knobs['plasticity'].mean().item()
            a = knobs['alpha'].mean().item()
            t = knobs['attention'].mean().item()
            
            print(f"{epoch:<6} | {ph:<7} | {p:<5.2f} {a:<5.2f} {t:<5.2f} | {g_acc:8.1f}% | {w2_acc:8.1f}%")

    return history

if __name__ == "__main__":
    USER_EPOCHS = 20000 # Ajustable
    
    print("="*80)
    print(f"🧬 SOVEREIGN CHIMERA: LA FUSIÓN FINAL ({USER_EPOCHS} Epochs)")
    print("Combinando Arquitectura Ablation 3 + Cerebro Nemesis v5")
    print("="*80)
    
    # 1. Static (Control)
    h_stat = run_final_showdown(USER_EPOCHS, "Static Chimera (No DJ)", False)
    
    # 2. Dynamic (Experimental)
    h_dyn = run_final_showdown(USER_EPOCHS, "Sovereign Chimera (Self-Governing)", True)
    
    print("\n" + "="*80)
    print("📊 VEREDICTO DE LA EVOLUCIÓN ARTIFICIAL")
    print("-" * 80)
    
    s_gl, d_gl = h_stat['global'][-1], h_dyn['global'][-1]
    s_w2, d_w2 = h_stat['w2'][-1], h_dyn['w2'][-1]
    
    print(f"{'Métrica':<20} | {'Static':<10} | {'Sovereign':<10} | {'Diferencia'}")
    print(f"{'Global Acc':<20} | {s_gl:9.1f}% | {d_gl:9.1f}% | {d_gl-s_gl:+6.1f}%")
    print(f"{'Retención W2':<20} | {s_w2:9.1f}% | {d_w2:9.1f}% | {d_w2-s_w2:+6.1f}%")
    
    print("-" * 80)
    if d_gl > s_gl:
        print("🎉 CONCLUSIÓN: La red autogobernada es superior.")
        print("   El DJ aprendió a manipular la Atención y la Memoria para sobrevivir.")
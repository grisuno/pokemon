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
# CONFIGURACIÓN
# =============================================================================
@dataclass
class PhysioConfig:
    seed: int = 42
    epochs: int = 200 
    batch_size: int = 64
    lr: float = 0.005
    d_in: int = 64
    d_hid: int = 128
    d_out: int = 10

# =============================================================================
# ENTORNO DE DATOS (RIGUROSO)
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
# EL SISTEMA HOMEOSTÁTICO (CEREBRO INTERNO)
# =============================================================================
class HomeostaticRegulator(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        # Input: [Sorpresa, Magnitud_Activación, Norma_Pesos]
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.Tanh(), # Tanh permite regulación negativa/positiva
            nn.Linear(16, 3), # Outputs: [Metabolismo, Sensibilidad, Gate]
            nn.Sigmoid() 
        )
        
    def forward(self, x, h_pre, w_norm):
        # 1. Sensor de Estrés (Sorpresa / Varianza Input)
        stress = (x.var(dim=1, keepdim=True) - 0.5).abs()
        
        # 2. Sensor de Excitación (Magnitud media de activación previa)
        excitation = h_pre.abs().mean(dim=1, keepdim=True)
        
        # 3. Sensor de Fatiga (Norma de los pesos, broadcasted al batch)
        fatigue = w_norm.view(1, 1).expand(x.size(0), 1)
        
        # Fusión de señales fisiológicas
        state = torch.cat([stress, excitation, fatigue], dim=1)
        controls = self.net(state)
        
        return {
            'metabolism': controls[:, 0].view(-1, 1), # Learning Rate Modificador
            'sensitivity': controls[:, 1].view(-1, 1), # Activation Slope
            'gate':        controls[:, 2].view(-1, 1)  # Slow/Fast Mix
        }

# =============================================================================
# NEURONA FISIOLÓGICA (PHYSIO-LIQUID NEURON)
# =============================================================================
class PhysioNeuron(nn.Module):
    def __init__(self, d_in, d_out, dynamic_mode=True):
        super().__init__()
        self.dynamic = dynamic_mode
        
        # Pesos Estructurales
        self.W_slow = nn.Linear(d_in, d_out, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        
        # Memoria Líquida
        self.register_buffer('W_fast', torch.zeros(d_out, d_in))
        
        self.ln = nn.LayerNorm(d_out)
        self.regulator = HomeostaticRegulator(d_in)
        self.base_lr = 0.1

    def forward(self, x):
        # 1. Pre-cálculo para el regulador (Estado interno)
        with torch.no_grad():
            h_raw = self.W_slow(x)
            w_norm = self.W_slow.weight.norm()
            
        # 2. Regulación Homeostática
        if self.dynamic:
            # El cerebro decide basándose en su propio estado físico
            physio = self.regulator(x, h_raw, w_norm)
        else:
            # Estado basal (sin regulación inteligente)
            dev = x.device
            ones = torch.ones(x.size(0), 1, device=dev)
            physio = {
                'metabolism': ones * 0.5,
                'sensitivity': ones * 0.5, # Equivale a pendiente 1.0 aprox
                'gate': ones * 0.5
            }

        # 3. Procesamiento
        slow = self.W_slow(x)
        fast = F.linear(x, self.W_fast)
        
        # 4. Aprendizaje Metabólico
        if self.training:
            with torch.no_grad():
                y = fast
                batch = x.size(0)
                hebb = torch.mm(y.T, x) / batch
                forget = (y**2).mean(0).unsqueeze(1) * self.W_fast
                
                # EL CAMBIO CLAVE: El metabolismo controla cuánto aprendemos
                # Si la red está estresada, puede acelerar el metabolismo.
                # Si está fatigada (pesos altos), puede frenarlo.
                metabolic_rate = physio['metabolism'].mean().item() * self.base_lr
                
                delta = torch.tanh(hebb - forget)
                self.W_fast.data.add_(delta * metabolic_rate)

        # 5. Activación Sensible (Swish Dinámico)
        # alpha decide la mezcla Slow/Fast
        combined = slow + (fast * physio['gate'])
        
        # Beta controla la pendiente de la activación.
        # Beta alto (>0.5 map to >1) = Reacción fuerte. Beta bajo = Reacción suave.
        beta = 0.5 + (physio['sensitivity'] * 2.0) # Rango 0.5 a 2.5
        
        # Función de activación fisiológica: x * sigmoid(beta * x)
        # Permite a la red decidir si ser lineal o no-lineal
        activated = combined * torch.sigmoid(beta * combined)
        
        return self.ln(activated), physio

# =============================================================================
# RED PRINCIPAL
# =============================================================================
class PhysioChimera(nn.Module):
    def __init__(self, config, dynamic_mode=True):
        super().__init__()
        self.l1 = PhysioNeuron(config.d_in, config.d_hid, dynamic_mode)
        self.l2 = PhysioNeuron(config.d_hid, config.d_out, dynamic_mode)
        self.dropout = nn.Dropout(0.05)
        
    def forward(self, x):
        h, p1 = self.l1(x)
        h = self.dropout(h)
        out, p2 = self.l2(h)
        return out, p1 # Retornamos estado de capa 1 para monitor

# =============================================================================
# EXPERIMENTO
# =============================================================================
def run_physio_experiment(epochs, name, dynamic):
    seed_everything(42)
    env = DataEnvironment()
    model = PhysioChimera(PhysioConfig, dynamic_mode=dynamic)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n🫀 MODELO: {name}")
    print(f"{'Epoch':<6} | {'Phase':<7} | {'Metab':<5} {'Sens':<5} {'Gate':<5} | {'Global':<8} | {'W2 Ret':<8}")
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
        
        out, physio = model(x)
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
            
            m = physio['metabolism'].mean().item()
            s = physio['sensitivity'].mean().item()
            g = physio['gate'].mean().item()
            
            print(f"{epoch:<6} | {ph:<7} | {m:<5.2f} {s:<5.2f} {g:<5.2f} | {g_acc:8.1f}% | {w2_acc:8.1f}%")

    return history

if __name__ == "__main__":
    USER_EPOCHS = 2000
    
    print("="*80)
    print(f"🧬 PHYSIO-CHIMERA v14: INTERNAL STATE REGULATION ({USER_EPOCHS} Epochs)")
    print("Mecanismo: Control Homeostático de Metabolismo, Sensibilidad y Gating.")
    print("="*80)
    
    # 1. Control (Sin regulación)
    h_stat = run_physio_experiment(USER_EPOCHS, "Static Physiology (Control)", False)
    
    # 2. Experimental (Con regulación interna)
    h_dyn = run_physio_experiment(USER_EPOCHS, "Dynamic Physiology (Homeostatic)", True)
    
    print("\n" + "="*80)
    print("📊 VEREDICTO DE LA FISIOLOGÍA COMPUTACIONAL")
    print("-" * 80)
    
    s_gl, d_gl = h_stat['global'][-1], h_dyn['global'][-1]
    s_w2, d_w2 = h_stat['w2'][-1], h_dyn['w2'][-1]
    
    print(f"{'Métrica':<20} | {'Static':<10} | {'Physio':<10} | {'Diferencia'}")
    print(f"{'Global Acc':<20} | {s_gl:9.1f}% | {d_gl:9.1f}% | {d_gl-s_gl:+6.1f}%")
    print(f"{'Retención W2':<20} | {s_w2:9.1f}% | {d_w2:9.1f}% | {d_w2-s_w2:+6.1f}%")
    
    print("-" * 80)
    if d_gl > s_gl and d_w2 > s_w2:
        print("🚀 CONCLUSIÓN: La regulación basada en estados internos (Fisiología) es el camino.")
        print("   La red aprendió a 'calmarse' o 'acelerarse' según su estrés interno.")
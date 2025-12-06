import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# =============================================================================
# 1. MONITOR DE RIQUEZA (Diferenciable, de v7.0)
# =============================================================================
def measure_spatial_richness(activations):
    """Retorna tensor (gradiente) y valor escalar"""
    if activations.size(0) < 2: return torch.tensor(0.0, device=activations.device), 0.0
    
    A_centered = activations - activations.mean(dim=0, keepdim=True)
    cov = A_centered.T @ A_centered / (activations.size(0) - 1)
    
    try:
        eigs = torch.linalg.eigvalsh(cov).abs()
        p = eigs / (eigs.sum() + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return entropy, torch.exp(entropy).item()
    except:
        return torch.tensor(0.0, device=activations.device), 1.0

# =============================================================================
# 2. HOMEOSTASIS LÍQUIDA (La mejora del usuario)
# =============================================================================
class HomeostasisEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.5 
    
    def decide(self, task_loss_val, richness_val, vn_entropy_val, target_entropy=2.8):
        # Normalizamos inputs para que compitan justamente en el Softmax
        
        # 1. Urgencia de Tarea (Si el loss es alto > 2.0, urge)
        # Normalización aprox: Loss usualmente va de 0 a 3
        focus_drive = task_loss_val * 1.5
        
        # 2. Urgencia de Exploración (Si riqueza < 20, urge)
        # Invertimos: Queremos explorar si richness es bajo
        explore_drive = max(0.0, (40.0 - richness_val) / 10.0)
        
        # 3. Urgencia de Reparación (Si entropía < target, urge)
        # Penalizamos fuertemente la caída estructural
        repair_drive = max(0.0, (target_entropy - vn_entropy_val) * 10.0)
        
        # Softmax arbitraje
        logits = torch.tensor([focus_drive, explore_drive, repair_drive]) / self.temperature
        probs = F.softmax(logits, dim=0)
        
        return probs[0].item(), probs[1].item(), probs[2].item() # Focus, Explore, Repair

# =============================================================================
# 3. NEURONA HÍBRIDA (v7 + Hebbian Gating del usuario)
# =============================================================================
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
                # Oja's Rule Gated: Solo aprendemos si 'focus' es alto
                y = fast_out 
                batch_size = x.size(0)
                hebb = torch.mm(y.T, x) / batch_size
                forget = (y ** 2).mean(0).unsqueeze(1) * self.W_fast
                
                delta = hebb - forget
                # Aquí aplicamos la innovación: Gatear la plasticidad
                self.W_fast = self.W_fast + (delta * self.fast_lr * plasticity_gate)

        return self.ln(slow_out + fast_out)

    def consolidate_svd(self, repair_strength):
        """Sueño a demanda, intensidad variable"""
        with torch.no_grad():
            # Fusionar
            combined = self.W_slow.weight.data + (self.W_fast * 0.1)
            
            try:
                U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
                
                # Whitening modulado por la urgencia de reparación
                # Si repair_strength es 1.0, hacemos whitening total. Si es bajo, suave.
                mean_S = S.mean()
                S_new = (S * (1.0 - repair_strength)) + (mean_S * repair_strength)
                
                self.W_slow.weight.data = U @ torch.diag(S_new) @ Vh
                self.W_fast.zero_()
                return "🔧 REPAIR"
            except:
                return "⚠️ FAIL"

# =============================================================================
# 4. ORGANISMO v8.0 "LIQUID SELF"
# =============================================================================
class OrganismV8(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.homeostasis = HomeostasisEngine()
        
        # El Ojo de la v7
        self.gaze = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.Sigmoid()
        )
        self.gaze[0].bias.data.fill_(1.0) # Empezar mirando todo
        
        self.L1 = LiquidNeuron(d_in, d_hid)
        self.L2 = LiquidNeuron(d_hid, d_out)
        
        self.vn_entropy = 3.0 # Estado inicial
        
    def forward(self, x, plasticity_gate=1.0):
        # 1. Mirar
        mask = self.gaze(x)
        x_focused = x * mask
        
        # 2. Procesar
        h = F.relu(self.L1(x_focused, plasticity_gate))
        out = self.L2(h, plasticity_gate)
        
        # Riqueza diferenciable (crucial para v7 integration)
        rich_tensor, rich_val = measure_spatial_richness(h)
        
        return out, rich_tensor, rich_val, mask.mean()

    def get_structure_entropy(self):
        # Calculamos esto solo bajo demanda para ahorrar cómputo
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
# SIMULACIÓN DE FLUJO
# =============================================================================
def run_liquid_synthesis():
    print(f"\n🧬 SÍNTESIS v8.0: The Liquid Self (Softmax Homeostasis)\n")
    
    D_IN, D_HID, D_OUT = 64, 128, 10
    organism = OrganismV8(D_IN, D_HID, D_OUT)
    optimizer = optim.AdamW(organism.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    inputs = torch.randn(64, D_IN)
    targets = torch.randint(0, D_OUT, (64,))
    
    # Estado inicial de prioridades
    p_focus, p_explore, p_repair = 1.0, 0.0, 0.0
    
    print(f"{'EP':<3} | {'TASK':<6} | {'RICH':<5} | {'W_ENT':<5} | {'FOC':<4} {'EXP':<4} {'REP':<4} | {'ESTADO'}")
    print("-" * 75)
    
    for epoch in range(1, 61):
        optimizer.zero_grad()
        
        # --- ENTORNO ---
        if 15 <= epoch < 25:
            inputs = torch.randn(64, D_IN) * 6.0 # Trauma
            event_base = "⚡ TRAUMA"
        elif 35 <= epoch < 45:
            inputs = torch.randn(64, D_IN) * 0.05 # Boredom
            event_base = "💤 BORED"
        else:
            inputs = torch.randn(64, D_IN) # Flow
            event_base = "🌊 FLOW"

        # --- STEP 1: FORWARD ---
        # Pasamos p_focus como gate para la plasticidad hebbiana
        outputs, rich_tensor, rich_val, gaze_width = organism(inputs, plasticity_gate=p_focus)
        
        # --- STEP 2: METRICS & HOMEOSTASIS ---
        task_loss = criterion(outputs, targets)
        # Calculamos entropía estructural (costoso, pero necesario para decisión)
        struct_ent = organism.get_structure_entropy()
        
        # EL CEREBRO DECIDE SUS PRIORIDADES PARA EL SIGUIENTE PASO
        p_focus, p_explore, p_repair = organism.homeostasis.decide(
            task_loss.item(), rich_val, struct_ent, target_entropy=2.8
        )
        
        # --- STEP 3: LOSS DINÁMICO ---
        # Loss de Tarea (ponderado por focus)
        weighted_task = task_loss * p_focus
        
        # Loss de Curiosidad (ponderado por explore)
        # Queremos MAXIMIZAR riqueza -> MINIMIZAR riqueza negativa
        # Usamos el tensor diferenciable de v7
        weighted_curiosity = -rich_tensor * 0.5 * p_explore
        
        # Loss Estructural (ponderado por repair)
        # En realidad, el repair se hace via SVD directo, pero podemos añadir un término
        # suave para ayudar al gradiente
        weighted_struct = 0.0 # El trabajo pesado lo hace consolidate_svd
        
        total_loss = weighted_task + weighted_curiosity
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(organism.parameters(), 1.0)
        optimizer.step()
        
        # --- STEP 4: ACCIÓN HOMEOSTÁTICA (SUEÑO A DEMANDA) ---
        sleep_msg = ""
        # Si la urgencia de reparar supera a las demás o es muy alta:
        if p_repair > 0.4:
            msg1 = organism.L1.consolidate_svd(p_repair)
            msg2 = organism.L2.consolidate_svd(p_repair)
            sleep_msg = f"🌙 {msg1}"
            
        final_event = f"{event_base} {sleep_msg}"
        
        print(f"{epoch:<3} | {task_loss.item():.4f} | {rich_val:.2f}  | {struct_ent:.3f} | {p_focus:.2f} {p_explore:.2f} {p_repair:.2f} | {final_event}")

if __name__ == "__main__":
    run_liquid_synthesis()
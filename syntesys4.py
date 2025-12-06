import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# =============================================================================
# 1. MONITOR DE RIQUEZA (Igual)
# =============================================================================
def measure_spatial_richness(activations):
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
# 2. HOMEOSTASIS SENSIBLE (RECALIBRADA)
# =============================================================================
class HomeostasisEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.4 # Un poco más decisivo
    
    def decide(self, task_loss_val, richness_val, vn_entropy_val):
        # --- RECALIBRACIÓN v8.1 ---
        
        # 1. Focus Drive: La presión del trabajo
        # Si loss es 2.3, drive es ~3.4
        focus_drive = task_loss_val * 1.5 
        
        # 2. Explore Drive: La ambición intelectual
        # Ahora el target es 44.0. Si tenemos 40, la diferencia es 4.
        # Multiplicamos por 1.0 para que 4.0 compita con el focus (3.4)
        target_richness = 44.0
        explore_drive = max(0.0, (target_richness - richness_val) * 1.0)
        
        # 3. Repair Drive: La integridad física
        # El target sube a 3.22 (la inicial). Cualquier caída duele.
        # Multiplicador x40: Una caída pequeña (0.1) genera un drive fuerte (4.0)
        target_entropy = 3.22
        repair_drive = max(0.0, (target_entropy - vn_entropy_val) * 40.0)
        
        # Softmax
        logits = torch.tensor([focus_drive, explore_drive, repair_drive]) / self.temperature
        probs = F.softmax(logits, dim=0)
        
        return probs[0].item(), probs[1].item(), probs[2].item()

# =============================================================================
# 3. NEURONA HÍBRIDA (Igual)
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
                y = fast_out 
                batch_size = x.size(0)
                hebb = torch.mm(y.T, x) / batch_size
                forget = (y ** 2).mean(0).unsqueeze(1) * self.W_fast
                delta = hebb - forget
                self.W_fast = self.W_fast + (delta * self.fast_lr * plasticity_gate)

        return self.ln(slow_out + fast_out)

    def consolidate_svd(self, repair_strength):
        with torch.no_grad():
            combined = self.W_slow.weight.data + (self.W_fast * 0.1)
            try:
                U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
                mean_S = S.mean()
                # Whitening proporcional a la urgencia
                S_new = (S * (1.0 - repair_strength)) + (mean_S * repair_strength)
                self.W_slow.weight.data = U @ torch.diag(S_new) @ Vh
                self.W_fast.zero_()
                return "🔧"
            except:
                return "⚠️"

# =============================================================================
# 4. ORGANISMO v8.1
# =============================================================================
class OrganismV8_1(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.homeostasis = HomeostasisEngine()
        self.gaze = nn.Sequential(nn.Linear(d_in, d_in), nn.Sigmoid())
        self.gaze[0].bias.data.fill_(0.5) # Empezar un poco más ciego para forzar curiosidad
        
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

def run_sensitive_self():
    print(f"\n🧬 SÍNTESIS v8.1: The Sensitive Self (Recalibrated)\n")
    
    D_IN, D_HID, D_OUT = 64, 128, 10
    organism = OrganismV8_1(D_IN, D_HID, D_OUT)
    optimizer = optim.AdamW(organism.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    inputs = torch.randn(64, D_IN)
    targets = torch.randint(0, D_OUT, (64,))
    
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

        outputs, rich_tensor, rich_val, gaze_width = organism(inputs, plasticity_gate=p_focus)
        
        task_loss = criterion(outputs, targets)
        struct_ent = organism.get_structure_entropy()
        
        # EL CEREBRO DECIDE
        p_focus, p_explore, p_repair = organism.homeostasis.decide(
            task_loss.item(), rich_val, struct_ent
        )
        
        # LOSS DINÁMICO
        weighted_task = task_loss * p_focus
        weighted_curiosity = -rich_tensor * 1.0 * p_explore # Curiosidad más fuerte
        
        total_loss = weighted_task + weighted_curiosity
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(organism.parameters(), 1.0)
        optimizer.step()
        
        # SUEÑO A DEMANDA
        sleep_msg = ""
        if p_repair > 0.3: # Umbral más sensible
            msg1 = organism.L1.consolidate_svd(p_repair)
            msg2 = organism.L2.consolidate_svd(p_repair)
            sleep_msg = f"🌙 {msg1}"
            
        final_event = f"{event_base} {sleep_msg}"
        
        print(f"{epoch:<3} | {task_loss.item():.4f} | {rich_val:.2f}  | {struct_ent:.3f} | {p_focus:.2f} {p_explore:.2f} {p_repair:.2f} | {final_event}")

if __name__ == "__main__":
    run_sensitive_self()
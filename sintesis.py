import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# =============================================================================
# MONITOR v6.0
# =============================================================================
class SpectralMonitorV6:
    def __init__(self, target_entropy=2.5):
        self.target_we = target_entropy

    def calc_structural_health(self, weight_matrix):
        W = weight_matrix
        if W.ndim > 2: W = W.flatten(1)
        W_norm = W / (W.norm() + 1e-12)
        try:
            _, S, _ = torch.linalg.svd(W_norm, full_matrices=False)
            ps = S ** 2
            ps = ps / (ps.sum() + 1e-12)
            vn_entropy = -torch.sum(ps * torch.log(ps + 1e-12))
            
            # Asimetría: Solo penalizamos si bajamos del target
            delta = self.target_we - vn_entropy
            loss = delta**2 if delta > 0 else torch.tensor(0.0, device=W.device)
            return loss, vn_entropy.item()
        except:
            return torch.tensor(0.0), 0.0

    def measure_spatial_richness(self, activations):
        if activations.size(0) < 2: return 0.0
        A_centered = activations - activations.mean(dim=0, keepdim=True)
        cov = A_centered.T @ A_centered / (activations.size(0) - 1)
        try:
            eigs = torch.linalg.eigvalsh(cov).abs()
            p = eigs / (eigs.sum() + 1e-12)
            entropy = -torch.sum(p * torch.log(p + 1e-12))
            return torch.exp(entropy).item()
        except:
            return 1.0

# =============================================================================
# NEURONA PRISMÁTICA (Oja + Spectral Dreaming) - CORREGIDA
# =============================================================================
class PrismaticNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.5)
        
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.ln = nn.LayerNorm(out_dim)
        self.fast_lr = 0.05 

    def forward(self, x):
        slow_out = self.W_slow(x)
        fast_out = F.linear(x, self.W_fast)
        
        if self.training:
            with torch.no_grad():
                # --- REGLA DE OJA (Safe Implementation) ---
                y = fast_out 
                batch_size = x.size(0)
                
                # Hebbian term: y * x^T
                hebb = torch.mm(y.T, x) / batch_size
                
                # Forgetting term: y^2 * W (Evita explosión)
                forget = (y ** 2).mean(0).unsqueeze(1) * self.W_fast
                
                delta = hebb - forget
                
                # CORRECCIÓN: Asignación a nuevo tensor, NO in-place (.add_)
                self.W_fast = self.W_fast + (delta * self.fast_lr)

        return self.ln(slow_out + fast_out)

    def prismatic_dream(self):
        """
        Sueño Entrópico:
        1. Fusionar memoria.
        2. Refracción Espectral (Whitening).
        """
        with torch.no_grad():
            # 1. Fusión Suave
            combined_weight = self.W_slow.weight.data + (self.W_fast * 0.1)
            
            # 2. Refracción Espectral (The Prism)
            try:
                U, S, Vh = torch.linalg.svd(combined_weight, full_matrices=False)
                
                # 3. Nivelación de Energía (Whitening Parcial)
                mean_S = S.mean()
                # Mezclamos el espectro actual con uno plano (uniforme)
                S_whitened = (S * 0.7) + (mean_S * 0.3) 
                
                # Reconstruimos W_slow
                self.W_slow.weight.data = U @ torch.diag(S_whitened) @ Vh
                
                msg = "💎 PRISM"
            except:
                # Fallback si SVD falla
                self.W_slow.weight.data = combined_weight
                msg = "🌙 CONSOL"
            
            # 4. Limpieza Hipocampal (Reinicio Oja)
            self.W_fast.zero_() 
            
            return msg

# =============================================================================
# ORGANISMO v6.0
# =============================================================================
class SynthesisOrganismV6(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.monitor = SpectralMonitorV6(target_entropy=2.8) # Objetivo ambicioso
        
        self.layer1 = PrismaticNeuron(input_dim, hidden_dim)
        self.layer2 = PrismaticNeuron(hidden_dim, output_dim)
        
        self.spatial_richness = 0.0
        self.vn_entropy = 0.0
        
    def forward(self, x):
        h1 = F.gelu(self.layer1(x))
        if self.training:
            self.spatial_richness = self.monitor.measure_spatial_richness(h1)
        return self.layer2(h1)

    def calculate_losses(self, outputs, targets, criterion):
        task_loss = criterion(outputs, targets)
        
        loss1, ent1 = self.monitor.calc_structural_health(self.layer1.W_slow.weight)
        loss2, ent2 = self.monitor.calc_structural_health(self.layer2.W_slow.weight)
        
        self.vn_entropy = (ent1 + ent2) / 2
        struct_loss = loss1 + loss2
        
        # Lambda dinámico
        lambda_struct = 1.0 if self.vn_entropy < 2.5 else 0.0
        
        # Incentivemos la riqueza directamente
        richness_loss = 0.0
        if self.spatial_richness < 15.0:
            richness_loss = 0.5 * (15.0 - self.spatial_richness)
            
        total_loss = task_loss + (struct_loss * lambda_struct) + richness_loss
        return total_loss, task_loss, struct_loss, lambda_struct

    def sleep(self):
        msg = self.layer1.prismatic_dream()
        self.layer2.prismatic_dream()
        return msg

# =============================================================================
# SIMULACIÓN v6.0
# =============================================================================
def run_prism_dream():
    print(f"\n🧬 SÍNTESIS v6.0: The Prism Dream (Oja's Rule + Spectral Whitening) FIXED\n")
    
    D_IN, D_HID, D_OUT = 64, 128, 10
    organism = SynthesisOrganismV6(D_IN, D_HID, D_OUT)
    # LR un poco más alto, confiamos en Oja
    optimizer = optim.AdamW(organism.parameters(), lr=0.01) 
    criterion = nn.CrossEntropyLoss()
    
    inputs = torch.randn(64, D_IN)
    targets = torch.randint(0, D_OUT, (64,))
    
    print(f"{'EP':<3} | {'TASK':<6} | {'W_ENT':<5} | {'RICH':<5} | {'ESTADO'}")
    print("-" * 60)
    
    for epoch in range(1, 26):
        optimizer.zero_grad()
        
        # Entorno
        if epoch == 10: 
            inputs = torch.randn(64, D_IN) * 4.0 # Trauma
            event = "⚡ TRAUMA"
        elif epoch == 20:
            inputs = inputs * 0.1 # Boredom
            event = "💤 BORED"
        else:
            event = "🌊 FLOW"

        outputs = organism(inputs)
        total_loss, t_loss, s_loss, lam = organism.calculate_losses(outputs, targets, criterion)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(organism.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 5 == 0:
            action = organism.sleep()
            event = f"🌙 {action}"
            
        print(f"{epoch:<3} | {t_loss.item():.4f} | {organism.vn_entropy:.3f} | {organism.spatial_richness:.2f}  | {event}")

if __name__ == "__main__":
    run_prism_dream()

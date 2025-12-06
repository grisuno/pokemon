import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# =============================================================================
# MONITOR v7.0 (Mismos sensores robustos de v6)
# =============================================================================
class SpectralMonitorV7:
    def __init__(self, target_entropy=2.8):
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
            
            delta = self.target_we - vn_entropy
            loss = delta**2 if delta > 0 else torch.tensor(0.0, device=W.device)
            return loss, vn_entropy.item()
        except:
            return torch.tensor(0.0), 0.0

    def measure_spatial_richness(self, activations):
        """
        Ahora retorna el tensor (para el gradiente) y el valor escalar.
        Necesitamos que sea diferenciable para que el 'Ojo' aprenda a buscar riqueza.
        """
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

# =============================================================================
# NUEVO: EL OJO CURIOSO (ATENCIÓN ACTIVA)
# =============================================================================
class CuriosityGaze(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Una red simple que decide la importancia de cada input
        self.attention = nn.Linear(input_dim, input_dim)
        # Inicializamos para que empiece mirando todo (bias positivo)
        self.attention.bias.data.fill_(1.0) 

    def forward(self, x):
        # Generamos una máscara entre 0.0 y 1.0 para cada feature
        # Esto es "Gating" o Atención suave
        mask = torch.sigmoid(self.attention(x))
        return x * mask, mask.mean()

# =============================================================================
# NEURONA PRISMÁTICA (Mecánica v6.0 Probada)
# =============================================================================
class PrismaticNeuronV7(nn.Module):
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
                # Regla de Oja Segura (Safe implementation)
                y = fast_out 
                batch_size = x.size(0)
                hebb = torch.mm(y.T, x) / batch_size
                forget = (y ** 2).mean(0).unsqueeze(1) * self.W_fast
                delta = hebb - forget
                self.W_fast = self.W_fast + (delta * self.fast_lr)

        return self.ln(slow_out + fast_out)

    def prismatic_dream(self):
        with torch.no_grad():
            combined_weight = self.W_slow.weight.data + (self.W_fast * 0.1)
            try:
                U, S, Vh = torch.linalg.svd(combined_weight, full_matrices=False)
                mean_S = S.mean()
                S_whitened = (S * 0.6) + (mean_S * 0.4) # Whitening ligeramente más agresivo
                self.W_slow.weight.data = U @ torch.diag(S_whitened) @ Vh
                msg = "💎 PRISM"
            except:
                self.W_slow.weight.data = combined_weight
                msg = "🌙 CONSOL"
            
            self.W_fast.zero_() 
            return msg

# =============================================================================
# ORGANISMO v7.0
# =============================================================================
class SynthesisOrganismV7(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.monitor = SpectralMonitorV7(target_entropy=2.8)
        
        # EL OJO
        self.gaze = CuriosityGaze(input_dim)
        
        # EL CEREBRO
        self.layer1 = PrismaticNeuronV7(input_dim, hidden_dim)
        self.layer2 = PrismaticNeuronV7(hidden_dim, output_dim)
        
        # ESTADO
        self.spatial_richness_val = 0.0
        self.vn_entropy = 0.0
        self.attention_width = 1.0 # Qué tanto estamos mirando (0 a 1)
        
    def forward(self, x):
        # 1. Mirar (Active Gaze)
        x_focused, self.attention_width = self.gaze(x)
        
        # 2. Pensar
        h1 = F.gelu(self.layer1(x_focused))
        
        # Medimos riqueza (manteniendo el grafo para el gradiente)
        if self.training:
            self.richness_tensor, self.spatial_richness_val = self.monitor.measure_spatial_richness(h1)
            
        return self.layer2(h1)

    def calculate_losses(self, outputs, targets, criterion):
        # A. Tarea
        task_loss = criterion(outputs, targets)
        
        # B. Salud Estructural (SVD de W_slow)
        loss1, ent1 = self.monitor.calc_structural_health(self.layer1.W_slow.weight)
        loss2, ent2 = self.monitor.calc_structural_health(self.layer2.W_slow.weight)
        self.vn_entropy = (ent1 + ent2) / 2
        struct_loss = loss1 + loss2
        
        # Lambda Estructural (Pánico si baja mucho)
        lambda_struct = 2.0 if self.vn_entropy < 2.5 else 0.0
        
        # C. CURIOSIDAD INTRÍNSECA (Maximizar Riqueza)
        # Queremos que la red de Atención (Gaze) encuentre inputs que produzcan alta riqueza.
        # Loss = -Entropy. (Minimizar entropía negativa = Maximizar entropía positiva)
        # Esto crea un gradiente que viaja desde h1 hasta el CuriosityGaze
        curiosity_loss = -self.richness_tensor * 0.1
        
        total_loss = task_loss + (struct_loss * lambda_struct) + curiosity_loss
        return total_loss, task_loss, self.vn_entropy

    def sleep(self):
        msg = self.layer1.prismatic_dream()
        self.layer2.prismatic_dream()
        return msg

# =============================================================================
# SIMULACIÓN DE LARGA DURACIÓN (60 ÉPOCAS)
# =============================================================================
def run_the_prisms_eye():
    print(f"\n🧬 SÍNTESIS v7.0: The Prism's Eye (Long Term Evolution)\n")
    
    D_IN, D_HID, D_OUT = 64, 128, 10
    organism = SynthesisOrganismV7(D_IN, D_HID, D_OUT)
    
    # Optimizador: El Gaze aprende rápido, el cuerpo lento
    optimizer = optim.AdamW(organism.parameters(), lr=0.008) 
    criterion = nn.CrossEntropyLoss()
    
    # Base inputs
    inputs = torch.randn(64, D_IN)
    targets = torch.randint(0, D_OUT, (64,))
    
    print(f"{'EP':<3} | {'TASK':<6} | {'W_ENT':<5} | {'RICH':<5} | {'GAZE':<4} | {'ESTADO'}")
    print("-" * 65)
    
    for epoch in range(1, 610): # 60 Épocas
        optimizer.zero_grad()
        
        # --- GENERADOR DE HISTORIA ---
        if epoch < 150:
            event = "🌊 FLOW" # Aprendizaje normal
            
        elif 150 <= epoch < 25:
            # TRAUMA SEVERO: Ruido de alta magnitud
            inputs = torch.randn(64, D_IN) * 6.0 
            event = "⚡ TRAUMA"
            
        elif 250 <= epoch < 40:
            # EDAD OSCURA (BOREDOM): Señal muy débil y repetitiva
            inputs = torch.randn(64, D_IN) * 0.05
            event = "💤 DARK"
            
        elif 400 <= epoch < 500:
            # RENACIMIENTO (EPIPHANY): Patrones complejos ocultos
            # Señal mixta: Ruido bajo + Estructura fuerte en canales específicos
            noise = torch.randn(64, D_IN) * 0.2
            structure = torch.sin(torch.linspace(0, 10, D_IN)) * 2.0
            inputs = noise + structure
            event = "💡 EPIPH"
            
        else:
            # ESTABILIDAD FINAL
            inputs = torch.randn(64, D_IN)
            event = "🏛 GOLD"

        # --- CICLO DE VIDA ---
        outputs = organism(inputs)
        total_loss, t_loss, w_ent = organism.calculate_losses(outputs, targets, criterion)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(organism.parameters(), 1.0)
        optimizer.step()
        
        # --- SUEÑO CADA 5 ÉPOCAS ---
        if epoch % 5 == 0:
            action = organism.sleep()
            event = f"🌙 {action}"
            
        # GAZE: Promedio de apertura de atención (0 a 1)
        gaze_width = organism.attention_width.item()
        
        print(f"{epoch:<3} | {t_loss.item():.4f} | {w_ent:.3f} | {organism.spatial_richness_val:.2f}  | {gaze_width:.2f} | {event}")

if __name__ == "__main__":
    run_the_prisms_eye()

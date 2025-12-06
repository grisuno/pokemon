# =============================================================================
# RESMA-NN 2.0 LIGHT – Para laptop estándar (CPU only, <2s/epoch)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# Activación PT-simétrica (versión ligera y estable)
# =============================================================================
class PTSymmetricActivation(nn.Module):
    def __init__(self, omega=50e12, chi=0.6, kappa_init=4.5e10):
        super().__init__()
        self.omega = omega
        self.chi = chi
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
    
    def forward(self, x):
        x = torch.clamp(x, min=-5.0, max=5.0)  # Rango seguro para CPU
        threshold = self.chi * self.omega
        coherence_ratio = self.kappa / (threshold + 1e-8)
        zeeman = torch.pow(torch.abs(x), 6.0) * 1e-6  # q=6 (más estable que q=8)
        gate = torch.sigmoid(coherence_ratio - zeeman)
        return x * gate

# =============================================================================
# Capa E8Lattice LIGHT – máscara fija, sin NetworkX en runtime
# =============================================================================
class E8LatticeLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.norm = nn.LayerNorm(out_features)
        # Máscara topológica fija (pequeña, eficiente, sin graph generation)
        self.register_buffer('topology_mask', self._fixed_sparse_mask(out_features, in_features))
        nn.init.xavier_uniform_(self.weights)
    
    def _fixed_sparse_mask(self, out_f, in_f):
        # Simula conectividad modular esparsa sin generar grafos
        mask = torch.zeros(out_f, in_f)
        # Cada neurona de salida conecta a ~30% de entradas
        for i in range(out_f):
            indices = torch.randperm(in_f)[:max(1, in_f // 3)]
            mask[i, indices] = 1.0
        return mask
    
    def forward(self, x):
        w_masked = self.weights * self.topology_mask
        out = F.linear(x, w_masked, self.bias)
        out = self.norm(out)
        # Corrección SYK ligera (q=3)
        syk = torch.pow(torch.abs(out), 3.0) * 1e-5
        return out + syk

# =============================================================================
# Cerebro RESMA ligero
# =============================================================================
class RESMABrainLight(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=10):
        super().__init__()
        self.layer1 = E8LatticeLayer(input_dim, hidden_dim)
        self.act1 = PTSymmetricActivation()
        self.layer2 = E8LatticeLayer(hidden_dim, 32)
        self.act2 = PTSymmetricActivation()
        self.readout = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        return self.readout(x)
    
    def resma_loss(self, output, target, lambda_topo=1e-5):
        task_loss = F.mse_loss(output, target)
        w = self.layer1.weights.view(-1)
        mean_w = w.mean()
        std_w = w.std() + 1e-8
        kurtosis = torch.mean((w - mean_w)**4) / (std_w**4 + 1e-8)
        topo_loss = 1.0 / (kurtosis + 1e-6)
        return task_loss + lambda_topo * topo_loss

# =============================================================================
# Ejecución ligera
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("🧠 RESMA-NN 2.0 LIGHT – Iniciando (CPU friendly)...")
    model = RESMABrainLight(input_dim=32, hidden_dim=64, output_dim=10)
    input_sig = torch.randn(16, 32)   # Batch pequeño
    target_sig = torch.randn(16, 10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n🔬 Entrenamiento ligero (10 epochs):")
    for epoch in range(10):
        optimizer.zero_grad()
        out = model(input_sig)
        loss = model.resma_loss(out, target_sig)
        if torch.isnan(loss):
            print("⚠️  NaN detectado. Abortando.")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        kappa = model.act1.kappa.item()
        coherence_ok = kappa < (0.6 * 50e12)
        status = "COHERENT ✅" if coherence_ok else "BROKEN ⚠️"
        print(f"   Epoch {epoch+1:02d} | Loss: {loss.item():.6f} | {status} | κ={kappa:.2e}")
    
    print("\n✅ Simulación completada. Listo para laptop estándar.")

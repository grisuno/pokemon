# =============================================================================
# RESMA-NN 2.0 + Multiverso Inicial (CPU Friendly)
# Inspirado en: "¿Y si le damos como inicio un grafo que represente un multiverso?"
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# Activación PT-simétrica (versión ligera)
# =============================================================================
class PTSymmetricActivation(nn.Module):
    def __init__(self, omega=50e12, chi=0.6, kappa_init=4.5e10):
        super().__init__()
        self.omega = omega
        self.chi = chi
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
    
    def forward(self, x):
        x = torch.clamp(x, min=-5.0, max=5.0)
        threshold = self.chi * self.omega
        coherence_ratio = self.kappa / (threshold + 1e-8)
        zeeman = torch.pow(torch.abs(x), 6.0) * 1e-6
        gate = torch.sigmoid(coherence_ratio - zeeman)
        return x * gate

# =============================================================================
# Capa E8Lattice con "Multiverso Inicial": múltiples módulos desconectos al inicio
# =============================================================================
class E8LatticeMultiverseLayer(nn.Module):
    def __init__(self, in_features, out_features, n_universes=4):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.norm = nn.LayerNorm(out_features)
        # --- Grafo multiverso: n_universes módulos densos, interconexión débil ---
        self.register_buffer('topology_mask', self._multiverse_mask(in_features, out_features, n_universes))
        nn.init.xavier_uniform_(self.weights)
    
    def _multiverse_mask(self, in_f, out_f, n_univ):
        mask = torch.zeros(out_f, in_f)
        # Dividir nodos en universos
        nodes_per_univ = in_f // n_univ
        for u in range(n_univ):
            start_in = u * nodes_per_univ
            end_in = start_in + nodes_per_univ if u < n_univ - 1 else in_f
            # Cada universo tiene nodos locales densamente conectados
            mask[:, start_in:end_in] = 1.0
        # Añadir esporádicas conexiones entre universos (1% de peso)
        inter = torch.rand(out_f, in_f) < 0.01
        mask[inter] = 0.3  # Conexión débil
        return mask
    
    def forward(self, x):
        w_masked = self.weights * self.topology_mask
        out = F.linear(x, w_masked, self.bias)
        out = self.norm(out)
        syk = torch.pow(torch.abs(out), 3.0) * 1e-5
        return out + syk

# =============================================================================
# Cerebro RESMA con Multiverso Inicial
# =============================================================================
class RESMABrainMultiverse(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=10):
        super().__init__()
        self.layer1 = E8LatticeMultiverseLayer(input_dim, hidden_dim, n_universes=4)
        self.act1 = PTSymmetricActivation()
        self.layer2 = E8LatticeMultiverseLayer(hidden_dim, 32, n_universes=4)
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
# Ejecución (CPU, seed fijo, sin simplificaciones)
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("🌌 RESMA-NN + Multiverso Inicial (CPU Friendly)...")
    model = RESMABrainMultiverse(input_dim=32, hidden_dim=64, output_dim=10)
    x = torch.randn(16, 32)
    y = torch.randn(16, 10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n🔬 Entrenamiento (10 epochs):")
    for epoch in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = model.resma_loss(out, y)
        if torch.isnan(loss):
            print("⚠️  NaN detectado. Abortando.")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        kappa = model.act1.kappa.item()
        status = "COHERENT ✅" if kappa < (0.6 * 50e12) else "BROKEN ⚠️"
        print(f"   Epoch {epoch+1:02d} | Loss: {loss.item():.6f} | {status} | κ={kappa:.2e}")
    
    print("\n✅ Simulación completada. Multiverso inicial integrado.")

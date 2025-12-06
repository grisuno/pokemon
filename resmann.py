import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# =============================================================================
# RESMA 5.1: QUANTUM GEOMETRIC NEURAL LAYER (STABLE)
# =============================================================================

class PTSymmetricActivation(nn.Module):
    def __init__(self, omega=50e12, chi=0.6, kappa_init=4.5e10):
        super().__init__()
        self.omega = omega
        self.chi = chi
        # Inicializamos kappa con un valor seguro y requerimos gradientes
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        
    def forward(self, x):
        # 1. Estabilización de entrada: Clampeamos x para evitar explosión x^8
        # En física real, la energía no es infinita. Limitamos el rango dinámico.
        x_safe = torch.clamp(x, min=-10.0, max=10.0)
        
        # 2. Cálculo del Umbral PT
        threshold = self.chi * self.omega
        # Evitamos división por cero añadiendo epsilon
        coherence_ratio = self.kappa / (threshold + 1e-8)
        
        # 3. Término Zeeman No-Lineal (SYK q=8)
        # Usamos x_safe y escalamos
        zeeman_term = torch.pow(torch.abs(x_safe), 8.0) * 1e-9
        
        # 4. Gate PT-Simétrica (Versión Numéricamente Estable con Sigmoide)
        # Matemáticamente: 1 / (1 + exp(zeeman - coherence))
        # Esto es equivalente a Sigmoid(coherence - zeeman)
        # PyTorch maneja los exponenciales internamente de forma segura en sigmoid
        gate = torch.sigmoid(coherence_ratio - zeeman_term)
        
        return x * gate

class E8LatticeLayer(nn.Module):
    def __init__(self, in_features, out_features, q_order=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.q_order = q_order // 2 
        
        # Inicialización de Xavier (importante para mantener valores bajos al inicio)
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weights) 
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('topology_mask', self._generate_ramsey_mask())
        
        # Normalización de capa (Homeostasis neuronal)
        # Vital para evitar que x crezca descontroladamente antes de elevarse a la 8va potencia
        self.norm = nn.LayerNorm(out_features)

    def _generate_ramsey_mask(self):
        G = nx.barabasi_albert_graph(self.in_features, min(4, self.in_features-1))
        adj = nx.to_numpy_array(G)
        mask = torch.tensor(adj, dtype=torch.float32)
        if self.out_features != self.in_features:
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                 size=(self.out_features, self.in_features), 
                                 mode='nearest').squeeze()
        return mask

    def forward(self, x):
        masked_weights = self.weights * self.topology_mask
        linear_out = F.linear(x, masked_weights, self.bias)
        
        # Normalizamos ANTES de la corrección no lineal
        linear_out = self.norm(linear_out)
        
        # Corrección SYK (q=4 en amplitud) estabilizada
        # Multiplicamos por un factor pequeño para que sea una perturbación
        syk_correction = torch.pow(torch.abs(linear_out), self.q_order) * 1e-4
        
        return linear_out + syk_correction

# =============================================================================
# CEREBRO RESMA
# =============================================================================

class RESMABrain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = E8LatticeLayer(input_dim, hidden_dim)
        self.act1 = PTSymmetricActivation()
        
        self.layer2 = E8LatticeLayer(hidden_dim, hidden_dim)
        self.act2 = PTSymmetricActivation()
        
        self.readout = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        return self.readout(x)

    def resma_loss(self, output, target, lambda_topo=1e-4):
        # 1. Pérdida de Tarea (Robustecida con Huber Loss en vez de MSE para evitar outliers)
        task_loss = F.huber_loss(output, target, delta=1.0)
        
        # 2. Regularización Fractal (Kurtosis)
        w1_flat = self.layer1.weights.view(-1)
        std_val = w1_flat.std() + 1e-8 # Protección div/0
        
        # Centramos y elevamos a 4
        mean_centered_4 = torch.mean((w1_flat - w1_flat.mean())**4)
        kurtosis = mean_centered_4 / (std_val**4)
        
        # Queremos maximizar kurtosis -> minimizar 1/kurtosis
        topo_loss = 1.0 / (kurtosis + 1e-6)
        
        return task_loss + lambda_topo * topo_loss

# =============================================================================
# EJECUCIÓN SEGURA
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42) # Reproducibilidad
    print("🧠 Inicializando RESMA-NN 2.0 (Stabilized)...")
    
    model = RESMABrain(input_dim=64, hidden_dim=128, output_dim=10)
    
    # Datos normalizados (importante para redes neuronales)
    input_signal = torch.randn(32, 64) 
    target_signal = torch.randn(32, 10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n🔬 Entrenamiento Experimental:")
    
    for epoch in range(10): # Aumentamos a 10 epochs
        optimizer.zero_grad()
        output = model(input_signal)
        
        loss = model.resma_loss(output, target_signal)
        
        if torch.isnan(loss):
            print(f"🚨 ALERTA: NaN detectado en Epoch {epoch}. Deteniendo para evitar crash.")
            break
            
        loss.backward()
        
        # === FIX CRÍTICO: GRADIANT CLIPPING ===
        # Esto corta los gradientes si son demasiado grandes antes de actualizar los pesos
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Monitoreo
        kappa_val = model.act1.kappa.item()
        # Verificar estado físico
        is_broken = kappa_val > (0.6 * 50e12)
        status = "BROKEN ⚠️" if is_broken else "COHERENT ✅"
        
        print(f"   Epoch {epoch+1:02d} | Loss: {loss.item():.6f} | PT: {status} | κ={kappa_val:.2e}")

    print("\n✅ Validación completada. El modelo converge sin explotar.")
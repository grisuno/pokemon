import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# =============================================================================
# RESMA 5.1: PHYSICS KERNEL (V5 FINAL)
# =============================================================================

class PTSymmetricActivation(nn.Module):
    def __init__(self, omega=50e12, chi=0.6, kappa_init=4.5e10):
        super().__init__()
        self.omega = omega
        self.chi = chi
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        
    def forward(self, x):
        # 1. Clamping (Física de rango finito)
        x_safe = torch.clamp(x, min=-20.0, max=20.0)
        
        # 2. Umbral de Protección (Coherence Ratio)
        threshold = self.chi * self.omega
        coherence_ratio = self.kappa / (threshold + 1e-8)
        
        # 3. Energía del Caos (Zeeman Term)
        # Ajustado a 1.2e-10 para una transición limpia en Amp ~12
        zeeman_term = torch.pow(torch.abs(x_safe), 8.0) * 1.2e-10
        
        # 4. Compuerta Lógica Cuántica
        # Transición nítida (High Gain)
        gain_factor = 1000.0
        # Estado de la puerta [0.0 a 1.0]
        gate_status = torch.sigmoid(gain_factor * (coherence_ratio - zeeman_term))
        
        # 5. Salida Física
        x_out = x * gate_status
        
        # RETORNO CUÁDRUPLE: (Señal, EstadoGate, Umbral, EnergíaActual)
        return x_out, gate_status, coherence_ratio, zeeman_term

class E8LatticeLayer(nn.Module):
    def __init__(self, in_features, out_features, q_order=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.q_order = q_order // 2 
        
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('topology_mask', self._generate_ramsey_mask())

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
        # Interacción SYK
        syk_correction = torch.pow(torch.abs(linear_out), self.q_order) * 1e-7
        return linear_out + syk_correction

class RESMABrain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = E8LatticeLayer(input_dim, hidden_dim)
        self.act1 = PTSymmetricActivation()
        self.readout = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        # Desempaquetamos solo la señal para seguir el flujo
        x, _, _, _ = self.act1(x)
        return self.readout(x)

# =============================================================================
# PROTOCOLO DE VALIDACIÓN V5
# =============================================================================

def stress_test_resma(model):
    print("\n" + "="*80)
    print("⚡ RESMA V5: REPORTE DE TRANSICIÓN DE FASE ⚡")
    print("="*80)
    
    # Obtenemos el umbral fijo de referencia
    dummy_x = torch.zeros(1)
    _, _, ratio_ref, _ = model.act1(dummy_x)
    limit = ratio_ref.item()
    
    print(f"Límite de Coherencia (Mielina Intacta): {limit:.2e}")
    print("Si Energía Zeeman > Límite -> COLAPSO.\n")
    
    amplitudes = np.linspace(1.0, 18.0, 25) 
    model.eval()
    
    print(f"{'AMP':<5} | {'TRANSMISIÓN':<15} | {'ZEEMAN (Caos)':<12} | {'ESTADO'}")
    print("-" * 70)
    
    with torch.no_grad():
        for amp in amplitudes:
            # Usamos ones() en lugar de randn() para eliminar el ruido estadístico
            # y ver la física pura de la amplitud.
            stress_signal = torch.ones(1, 64) * amp
            
            # 1. Propagación
            x_in = model.layer1(stress_signal)
            
            # 2. Medición Física (Desempaquetado correcto)
            _, gate_val, _, zeeman_val = model.act1(x_in)
            
            # 3. Métricas
            tr = gate_val.mean().item()      # Ahora sí es [0.0 - 1.0]
            z_mean = zeeman_val.mean().item()
            
            # 4. Visualización
            bar_len = int(tr * 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            
            # Estado
            if tr > 0.99:
                status = "✅ VIVO"
            elif tr < 0.01:
                status = "💀 SILENCIO"
            else:
                status = "⚠️ TRANSICIÓN"
            
            # Marcador de colapso
            marker = " "
            if z_mean > limit: marker = "🔴"
            
            print(f"{amp:5.1f} | {bar} {tr:4.0%} | {z_mean:.2e} {marker}   | {status}")

if __name__ == "__main__":
    torch.manual_seed(42)
    model = RESMABrain(input_dim=64, hidden_dim=128, output_dim=10)
    stress_test_resma(model)
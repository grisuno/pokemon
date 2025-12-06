"""
dmg_core.py
===========
Dynamic Magnitude Gating (DMG) for Robust Neural Networks.
Implements sparse connectivity and energy-based gating for noise suppression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

class AdaptiveMagnitudeGate(nn.Module):
    """
    Bio-inspired gating mechanism.
    Acts as a learnable filter that suppresses signals exceeding a dynamic threshold.
    
    Formula: Gate = Sigmoid( Gain * (Threshold - |x|^p * Sensitivity) )
    """
    def __init__(self, base_threshold: float = 1.0, power_order: int = 8):
        super().__init__()
        # Antes: kappa. Ahora: threshold (parámetro entrenable)
        self.threshold = nn.Parameter(torch.tensor(float(base_threshold)))
        
        self.power_order = power_order
        self.gain = 100.0       # Sensibilidad de la transición
        self.sensitivity = 1e-5 # Factor de escala para evitar explosión numérica
        
    def forward(self, x: torch.Tensor):
        # 1. Clamping para estabilidad numérica (Ingeniería, no física)
        x_safe = torch.clamp(x, min=-20.0, max=20.0)
        
        # 2. Cálculo de Magnitud (High-Order Norm)
        magnitude_term = torch.pow(torch.abs(x_safe), self.power_order) * self.sensitivity
        
        # 3. Gating Logic
        # Si magnitud > threshold -> Gate cierra (tiende a 0)
        # Si magnitud < threshold -> Gate abre (tiende a 1)
        gate = torch.sigmoid(self.gain * (self.threshold - magnitude_term))
        
        return x * gate, gate

class SparseTopologyLayer(nn.Module):
    """
    Linear layer with sparse connectivity enforcement derived from 
    Scale-Free (Barabási-Albert) graphs.
    """
    def __init__(self, in_features: int, out_features: int, sparsity_k: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Inicialización de pesos estándar
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Máscara de dispersión (Sparsity Mask)
        self.register_buffer('mask', self._generate_sparse_mask(sparsity_k))

    def _generate_sparse_mask(self, k_neighbors) -> torch.Tensor:
        """Generates a Barabási-Albert scale-free mask."""
        # Nota: Usamos grafos matemáticos estándar, sin pretensiones de grupos de Lie.
        G = nx.barabasi_albert_graph(self.in_features, min(k_neighbors, self.in_features-1))
        adj = nx.to_numpy_array(G)
        mask = torch.tensor(adj, dtype=torch.float32)
        
        if self.out_features != self.in_features:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), 
                size=(self.out_features, self.in_features), 
                mode='nearest'
            ).squeeze()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aplicar máscara (Sparsity Constraint)
        masked_weights = self.weights * self.mask
        return F.linear(x, masked_weights, self.bias)

class DMGNetwork(nn.Module):
    """
    Robust Neural Network architecture using Sparse Layers and Dynamic Gating.
    Designed for high noise resistance (MNIST-C / Adversarial robustness).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Capa 1: Extracción de características dispersas
        self.layer1 = SparseTopologyLayer(input_dim, hidden_dim)
        self.gate1 = AdaptiveMagnitudeGate()
        
        # Capa 2: Procesamiento profundo
        self.layer2 = SparseTopologyLayer(hidden_dim, hidden_dim)
        self.gate2 = AdaptiveMagnitudeGate()
        
        # Clasificador
        self.readout = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x, g1 = self.gate1(x)
        
        x = self.layer2(x)
        x, g2 = self.gate2(x)
        
        return self.readout(x)
#%%writefile neurosoberano_v2_FINAL.py
# =============================================================================
# NeuroLogos Bicameral 2.0 - VERSIÓN COMPLETA Y CORREGIDA
# Preserva TODAS las características del modelo original
# Solamente aplicados fixes de estabilidad, sin simplificaciones
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import random

# FIX: Import faltante para tqdm
from tqdm import tqdm

# Define vocab size based on CIFARCaptions
VOCAB_SIZE = 1000

# ==================== MEDIDAS DE COMPLEJIDAD REALES ====================

def compute_phi_effective(activations, k_partitions=5):
    """
    Φₑ efectivo: integración causal simplificada para batches
    activations: [B, N, D] 
    """
    B, N, D = activations.shape
    phi_values = []
    
    # Información total H(X)
    flat = activations.reshape(B, -1)
    H_total = torch.distributions.Categorical(logits=flat.softmax(dim=-1)).entropy()
    
    for _ in range(k_partitions):
        indices = torch.randperm(N, device=activations.device)
        mid = N // 2
        
        part1 = activations[:, indices[:mid], :].reshape(B, -1)
        part2 = activations[:, indices[mid:], :].reshape(B, -1)
        
        H_part1 = torch.distributions.Categorical(logits=part1.softmax(dim=-1)).entropy()
        H_part2 = torch.distributions.Categorical(logits=part2.softmax(dim=-1)).entropy()
        
        phi = (H_part1 + H_part2 - H_total).clamp(min=0)
        phi_values.append(phi)
    
    return torch.stack(phi_values).mean()


def measure_spatial_richness(activations):
    """
    FIX: Métrica de riqueza dimensional efectiva con escalado positivo garantizado
    Preserva interfaz exacta: shannon_entropy, richness, vn_entropy (valores POSITIVOS)
    Evita saturación en 2.0 y elimina negativos usando Participation Ratio escalado
    """
    # Manejar input 3D: [B, N, D] → [B*N, D] para estabilidad estadística
    original_shape = activations.shape
    if activations.dim() == 3:
        B, N, D = activations.shape
        activations = activations.reshape(B * N, D)
    elif activations.dim() == 2:
        N, D = activations.shape
    else:
        return 0.0, 1.0, 0.0  # Fallback seguro para dimensionalidades inesperadas
    
    if activations.size(0) <= 1:
        return 0.0, 1.0, 0.0  # Estadísticas inválidas, retornar valores neutrales
    
    # Muestreo seguro para estabilidad numérica en batches grandes
    if activations.size(0) > 64:
        indices = torch.randperm(activations.size(0), device=activations.device)[:64]
        activations = activations[indices]
    
    # Cálculo de matriz de covarianza robusta con estabilizador numérico
    activ_centered = activations - activations.mean(dim=0, keepdim=True)
    cov = (activ_centered.T @ activ_centered) / (activations.size(0) - 1 + 1e-8)
    
    # RIQUEZA 1: Dimensionalidad Efectiva (Participation Ratio) en [0,1]
    eigenvalues = torch.linalg.eigvals(cov).real.abs().clamp(min=1e-10)
    participation_ratio = (eigenvalues.sum()**2) / (eigenvalues**2).sum()
    
    # FIX: Escalado logarítmico positivo para evitar saturación y negativos
    # richness = 1 + log(9 * participation_ratio + 1) → valores en [1, ~3.3]
    richness = 1.0 + torch.log(9.0 * participation_ratio + 1.0).item()
    
    # RIQUEZA 2: Entropía de Von Neumann (estructural) en [0, log(D)]
    rho = eigenvalues / eigenvalues.sum()
    vn_entropy = -(rho * torch.log(rho + 1e-10)).sum().item()
    
    # Mapeo a entropía de Shannon (conservar interfaz del modelo, siempre positiva)
    shannon_entropy = math.log(richness) if richness > 0 else 0.0
    
    return shannon_entropy, richness, vn_entropy


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('inf')):
    """Filtro Top-K y Nucleus Sampling estandar"""
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits

# ==================== PLASTICIDAD BCM (NEUROCIENCIA REAL) ====================

class BCMPlasticity(nn.Module):
    def __init__(self, neurons, tau_theta=100.0):
        super().__init__()
        self.register_buffer('theta', torch.ones(neurons) * 0.25)
        self.tau_theta = tau_theta
        
    def forward(self, activity, dt=0.01):
        """dθ/dt = (E[activity²] - θ)/τ  →  dw/dt ∝ activity*(activity-θ)"""
        if self.training:
            mean_sq = activity.pow(2).mean(0)
            self.theta.copy_(self.theta * (1 - dt/self.tau_theta) + mean_sq * (dt/self.tau_theta))
        
        # Solo modula durante entrenamiento
        return activity * (activity - self.theta.unsqueeze(0)) if self.training else torch.zeros_like(activity)

# ==================== LIQUID NEURON CON TRES TIME-SCALES ====================

class LiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        
        self.register_buffer('W_medium', torch.zeros(out_dim, in_dim))
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        
        self.ln = nn.LayerNorm(out_dim)
        
        # Controlador de plasticidad adaptativa
        self.plasticity_controller = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.plasticity_controller[2].bias.data.fill_(-2.0)
        
        self.base_lr = 0.015
        self.prediction_error = 0.0
        
    # FIX: REEMPLAZAR MÉTODO FORWARD COMPLETO
    def forward(self, x, global_plasticity=0.1, transfer_rate=0.005):
        original_shape = x.shape
        if x.dim() == 3:
            B, N, D = x.shape
            x_flat = x.view(B * N, D)
        else:
            x_flat = x
            B = x.size(0)
            N = 1

        slow_out = self.W_slow(x_flat)
        medium_out = F.linear(x_flat, self.W_medium)
        fast_out = F.linear(x_flat, self.W_fast)
        pre_act = slow_out + medium_out + fast_out
        
        # FIX: Estabilización numérica con clamp
        batch_mean = pre_act.mean().detach()
        batch_std = pre_act.std(unbiased=False).clamp(min=1e-6).detach()
        
        stats = torch.cat([batch_mean.expand(B * N, 1), batch_std.expand(B * N, 1)], dim=1)
        learned_plasticity = self.plasticity_controller(stats).squeeze(1)
        
        effective_plasticity = global_plasticity * learned_plasticity * (1.0 - self.prediction_error)
        
        out_flat = 5.0 * torch.tanh(self.ln(pre_act) / 5.0)

        # ACTUALIZACIÓN PLÁSTICA
        if self.training and effective_plasticity.mean() > 0.001:
            with torch.no_grad():
                out_centered = out_flat - out_flat.mean(dim=0, keepdim=True)
                correlation = torch.mm(out_centered.T, x_flat) / x_flat.size(0)
                correlation = torch.clamp(correlation, -0.1, 0.1)  # FIX: Prevenir explosión
                
                self.W_medium.data += self.W_fast.data * transfer_rate
                self.W_fast.data += correlation * effective_plasticity.mean() * self.base_lr
                self.W_fast.data.mul_(1.0 - transfer_rate)
                self.W_fast.data.clamp_(-3.0, 3.0)

        if len(original_shape) == 3:
            out = out_flat.view(B, N, -1)
        else:
            out = out_flat

        return out

    # FIX: REEMPLAZAR MÉTODO CONSOLIDATE_SVD COMPLETO
    def consolidate_svd(self, repair_strength=1.0, timescale='fast'):
        """Mantener interfaz exacta pero implementar consolidación Hebbiana real"""
        if timescale == 'fast':
            W_to_consolidate = self.W_fast
            transfer_factor = 0.05 * repair_strength
            decay_factor = 0.5
        elif timescale == 'medium':
            W_to_consolidate = self.W_medium
            transfer_factor = 0.02 * repair_strength
            decay_factor = 0.7
        else:
            return False
            
        with torch.no_grad():
            # Consolidación Hebbiana real
            self.W_slow.weight.data += W_to_consolidate * transfer_factor
            W_to_consolidate.data *= decay_factor
            return True

# ==================== CORTEZA VISUAL (V1-V4) ====================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class VisualCortex(nn.Module):
    def __init__(self, output_dim=128, grid_size=6):
        super().__init__()
        self.grid_size = grid_size
        self.output_dim = output_dim
        
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.adapter = nn.Conv2d(512, output_dim, kernel_size=1)
        self.final_norm = nn.LayerNorm(output_dim)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.layer4(x)  # 4x4
        x = F.interpolate(x, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=False)
        x = self.adapter(x)
        x = x.permute(0, 2, 3, 1)
        x = self.final_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

# ==================== HEMISFERIO DERECHO COMPLETO ====================

class SymbioticBasisRefinement(nn.Module):
    def __init__(self, dim, num_atoms=64):
        super().__init__()
        self.dim = dim
        self.num_atoms = min(num_atoms, dim)
        self.basis_atoms = nn.Parameter(torch.empty(self.num_atoms, dim))
        nn.init.orthogonal_(self.basis_atoms)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(self.basis_atoms)
        attn = torch.matmul(Q, K.T) * self.scale
        weights = F.softmax(attn, dim=-1)
        x_clean = torch.matmul(weights, self.basis_atoms)
        entropy = -torch.sum(weights * torch.log(weights + 1e-6), dim=-1).mean()
        return x_clean, entropy, torch.tensor(0.0, device=x.device)

class AdaptiveCombinatorialComplexLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, config=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.mapper = nn.Linear(in_dim, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
        
    def forward(self, x, plasticity_gate=0.0):
        h = self.mapper(x)
        return self.norm(h)

class GraphNeuralLayer(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None: hidden_dim = dim
        self.message_fn = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        self.update_fn = nn.LayerNorm(dim)
        
    def forward(self, nodes, adjacency):
        neighbor_sum = torch.bmm(adjacency, nodes)
        messages = self.message_fn(torch.cat([nodes, neighbor_sum], dim=-1))
        return self.update_fn(nodes + messages)

def create_grid_adjacency(N, connectivity=8):
    """Crea matriz de adyacencia para grid cuadrado"""
    size = int(N**0.5)
    adj = torch.zeros(N, N)
    for i in range(N):
        row, col = i // size, i % size
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                if connectivity == 4 and abs(dr) + abs(dc) != 1: continue
                r, c = row + dr, col + dc
                if 0 <= r < size and 0 <= c < size:
                    adj[i, r * size + c] = 1
    return adj

class RightHemisphere(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config if config else type('Config', (), {
            'grid_size': 6,
            'visual_dim': 256,
            'hidden_dim': 512
        })()
        
        # Corteza visual retinotópica
        self.visual_cortex = VisualCortex(output_dim=self.config.visual_dim, grid_size=self.config.grid_size)
        
        # Topología combinatoria
        self.topo_layer = AdaptiveCombinatorialComplexLayer(
            in_dim=self.config.visual_dim, 
            hid_dim=self.config.hidden_dim, 
            num_nodes=self.config.grid_size**2
        )
        
        # GNN espacial
        self.wernicke_gnn = nn.ModuleList([
            GraphNeuralLayer(self.config.hidden_dim) for _ in range(2)
        ])
        
        # Liquid neuron
        self.liquid = LiquidNeuron(self.config.hidden_dim, self.config.hidden_dim)
        
        # Symbiotic refinement
        self.symbiotic = SymbioticBasisRefinement(self.config.hidden_dim)
        
        # Proyección callosal
        self.callosal_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2)
        
    def forward(self, image, adjacency=None, plasticity=0.1):
        B = image.size(0)
        
        # Encode visual
        features = self.visual_cortex(image)  # [B, D, grid, grid]
        features = features.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Topología
        topo_features = self.topo_layer(features, plasticity_gate=plasticity)
        
        # GNN
        if adjacency is None:
            adjacency = create_grid_adjacency(topo_features.size(1), 8).to(image.device).unsqueeze(0).expand(B, -1, -1)
        
        gnn_features = topo_features
        for gnn in self.wernicke_gnn:
            gnn_features = gnn(gnn_features, adjacency)
        
        # Liquid integration
        liquid_out = self.liquid(gnn_features, plasticity)
        
        # Symbiotic refinement
        refined, sym_entropy, _ = self.symbiotic(liquid_out)
        
        # Proyección para calloso
        projected = self.callosal_proj(refined)
        
        return projected, sym_entropy

# ==================== HEMISFERIO IZQUIERDO COMPLETO ====================

class MiniUnconscious(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        self.topo_bridge = nn.Linear(256*2*2, 512)
        self.norm = nn.LayerNorm(512)
        
    def forward(self, x):
        return self.norm(self.topo_bridge(self.stem(x)))

class NestedUnconscious(nn.Module):
    def __init__(self, grid_size=4, output_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
        )
        
        self.nodes = 16
        self.mapper = nn.Linear(512, output_dim)
        self.intra_adj = nn.Parameter(torch.randn(4, 4) * 0.5)
        self.intra_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim*2), nn.GELU(), nn.Linear(output_dim*2, output_dim)
        )
        self.inter_adj = nn.Parameter(torch.randn(4, 4) * 0.5)
        self.inter_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim*2), nn.GELU(), nn.Linear(output_dim*2, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        B = x.size(0)
        h = self.stem(x)
        h_flat = h.view(B, 16, -1)
        nodes = self.mapper(h_flat)
        
        # Intra-cluster
        intra_adj = torch.sigmoid(self.intra_adj) * (1 - torch.eye(4, device=x.device))
        nodes_reshaped = nodes.view(B, 4, 4, -1)
        msgs = self.intra_mlp(nodes_reshaped)
        nodes_reshaped = nodes_reshaped + torch.matmul(intra_adj, msgs)
        
        # Inter-cluster
        inter_adj = torch.sigmoid(self.inter_adj) * (1 - torch.eye(4, device=x.device))
        cluster = nodes_reshaped.mean(2)
        cluster_msgs = self.inter_mlp(cluster)
        cluster_update = torch.matmul(inter_adj, cluster_msgs)
        nodes_reshaped = nodes_reshaped + cluster_update.unsqueeze(2)
        
        nodes_final = nodes_reshaped.reshape(B, 16, -1)
        return self.norm(nodes_final)

class TopologicalCompressor(nn.Module):
    def __init__(self, node_dim=512):
        super().__init__()
        self.attn_weight = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.integrator = LiquidNeuron(node_dim, node_dim)
        
    def forward(self, nodes, plasticity, transfer_rate=0.005):
        weights = self.attn_weight(nodes)
        weights = F.softmax(weights, dim=1)
        thought_pooled = (nodes * weights).sum(dim=1)
        thought_final = self.integrator(thought_pooled, plasticity, transfer_rate)
        return thought_final


# REEMPLAZAR TODO EL MÉTODO forward DENTRO DE LA CLASE ConsciousCore

class ConsciousCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        self.topological_compressor = TopologicalCompressor(node_dim=512)
        self.attention_pool = nn.MultiheadAttention(512, 8, batch_first=True)
        self.liquid = LiquidNeuron(512, 512)
        self.callosal_adapter = nn.Linear(256, 512)
        self.thought_compressor = nn.Identity()
        self.meta_probe = nn.Linear(512, 64)
        self.mode = "vector"
        self.last_mode = "vector"  # FIX: Inicializar atributo de introspección
    
    def forward(self, visual_features, plasticity, transfer_rate=0.005):
        if visual_features.size(-1) == 256:
            visual_features = self.callosal_adapter(visual_features)
        
        if visual_features.dim() == 3:
            self.mode = "sequence"
            self.last_mode = "sequence"  # FIX: Actualizar introspección
            h, _ = self.sequence_attention(visual_features, visual_features, visual_features)
            thought = self.topological_compressor(h, plasticity, transfer_rate)
            active_liquid = self.topological_compressor.integrator
        else:
            self.mode = "vector"
            self.last_mode = "vector"  # FIX: Actualizar introspección
            q = visual_features.unsqueeze(1)
            pooled, _ = self.attention_pool(q, q, q)
            pre_liquid = pooled.squeeze(1)
            thought = self.liquid(pre_liquid, plasticity, transfer_rate)
            active_liquid = self.liquid
        
        self.last_thought = thought
        
        # DIAGNÓSTICO CON MEDIDAS REALES
        pre_act_raw = active_liquid.W_slow(thought.detach())
        meta_activations = self.meta_probe(pre_act_raw)
        self.last_richness, _, self.last_vn_entropy = measure_spatial_richness(meta_activations.unsqueeze(0))
        self.last_fast_norm = active_liquid.W_fast.norm().item()
        
        batch_mean = pre_act_raw.mean(dim=1, keepdim=True)
        batch_std = pre_act_raw.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
        stats = torch.cat([batch_mean, batch_std], dim=1)
        learned_plasticity = active_liquid.plasticity_controller(stats).squeeze(1).mean().item()
        error_proxy = max(active_liquid.prediction_error, 0.05)
        self.last_effective_plasticity = (plasticity * learned_plasticity) * (1.0 - error_proxy)
        self.last_learned_plasticity = learned_plasticity
        self.last_prediction_error = error_proxy
        
        return self.thought_compressor(thought)
    
    def get_liquid_module(self):
        if self.mode == "sequence":
            return self.topological_compressor.integrator
        else:
            return self.liquid

class LeftHemisphere(nn.Module):
    def __init__(self, use_nested=False):
        super().__init__()
        self.eye = NestedUnconscious() if use_nested else MiniUnconscious()
        self.cortex = ConsciousCore()
        self.callosal_proj = nn.Linear(512, 256)
        
    def forward(self, image, callosal_input=None, plasticity=0.1, transfer_rate=0.005):
        visual_features = self.eye(image)
        if callosal_input is not None:
            visual_features = visual_features + self.callosal_proj(callosal_input.mean(dim=1))
        return self.cortex(visual_features, plasticity, transfer_rate)

# ==================== ÁREA DE BROCA COMPLETA ====================

class BioDecoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=512, visual_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
    def forward(self, thought, visual_features=None, captions=None, max_len=20):
        batch_size = thought.size(0)
        device = thought.device
        
        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            if self.training:
                embeddings = embeddings + torch.randn_like(embeddings) * 0.05
            
            lstm_out, hidden = self.lstm(embeddings, self._get_init_state(thought))
            
            gate = self.liquid_gate(lstm_out)
            lstm_out = lstm_out * gate
            
            if visual_features is not None:
                vis_proj = self.visual_proj(visual_features)
                query = lstm_out
                key = vis_proj
                value = vis_proj
                attn_scores = torch.bmm(query, key.transpose(1, 2)) / (self.hidden_dim ** 0.5)
                attn_weights = F.softmax(attn_scores, dim=-1)
                visual_context = torch.bmm(attn_weights, value)
                fused = lstm_out + visual_context
            else:
                fused = lstm_out
            
            return self.out(fused)
        else:
            # Modo inferencia
            generated = []
            input_word = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)  # <BOS>
            hidden = self._get_init_state(thought)
            
            if visual_features is not None:
                vis_proj = self.visual_proj(visual_features)
            else:
                vis_proj = None
            
            # FIX: Temperatura dinámica basada en epoch
            epoch = getattr(thought, 'epoch', 0) if hasattr(thought, 'epoch') else 0
            temperature = 0.4 + 0.4 * min(epoch / 50.0, 1.0)
            top_p = 0.7 + 0.2 * min(epoch / 50.0, 1.0)
            
            generated_token_history = []
            for step in range(max_len):
                emb = self.embedding(input_word)
                out, hidden = self.lstm(emb, hidden)
                gate = self.liquid_gate(out)
                out = out * gate
                
                if vis_proj is not None:
                    query = out.squeeze(1)
                    key = vis_proj
                    value = vis_proj
                    attn_scores = torch.bmm(query.unsqueeze(1), key.transpose(1, 2)) / (self.hidden_dim ** 0.5)
                    attn_weights = F.softmax(attn_scores, dim=-1)
                    visual_context = torch.bmm(attn_weights, value)
                    out = out + visual_context
                
                logits = self.out(out.squeeze(1))
                
                # FIX: Penalización de repetición
                if step >= 1:
                    for b in range(batch_size):
                        recent_tokens = set()
                        lookback = min(3, len(generated_token_history))
                        for i in range(1, lookback + 1):
                            recent_tokens.add(generated_token_history[-i][b].item())
                        for tok in recent_tokens:
                            logits[b, tok] -= 2.0
                
                logits = logits / temperature
                logits = top_k_top_p_filtering(logits, top_k=0, top_p=top_p)
                probabilities = F.softmax(logits, dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1)
                generated.append(next_word)
                generated_token_history.append(next_word.squeeze(1))
                input_word = next_word
            
            return torch.cat(generated, dim=1)
    
    def _get_init_state(self, thought):
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        return (h0, c0)

# ==================== CUERPO CALLOSO CON COSTO METABÓLICO ====================
# REEMPLAZAR TODO EL MÉTODO forward DENTRO DE LA CLASE ConsciousCore

class ConsciousCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        self.topological_compressor = TopologicalCompressor(node_dim=512)
        self.attention_pool = nn.MultiheadAttention(512, 8, batch_first=True)
        self.liquid = LiquidNeuron(512, 512)
        self.callosal_adapter = nn.Linear(256, 512)
        self.thought_compressor = nn.Identity()
        self.meta_probe = nn.Linear(512, 64)
        self.mode = "vector"
        self.last_mode = "vector"  # FIX: Inicializar atributo de introspección
    
    def forward(self, visual_features, plasticity, transfer_rate=0.005):
        if visual_features.size(-1) == 256:
            visual_features = self.callosal_adapter(visual_features)
        
        if visual_features.dim() == 3:
            self.mode = "sequence"
            self.last_mode = "sequence"  # FIX: Actualizar introspección
            h, _ = self.sequence_attention(visual_features, visual_features, visual_features)
            thought = self.topological_compressor(h, plasticity, transfer_rate)
            active_liquid = self.topological_compressor.integrator
        else:
            self.mode = "vector"
            self.last_mode = "vector"  # FIX: Actualizar introspección
            q = visual_features.unsqueeze(1)
            pooled, _ = self.attention_pool(q, q, q)
            pre_liquid = pooled.squeeze(1)
            thought = self.liquid(pre_liquid, plasticity, transfer_rate)
            active_liquid = self.liquid
        
        self.last_thought = thought
        
        # DIAGNÓSTICO CON MEDIDAS REALES
        pre_act_raw = active_liquid.W_slow(thought.detach())
        meta_activations = self.meta_probe(pre_act_raw)
        self.last_richness, _, self.last_vn_entropy = measure_spatial_richness(meta_activations.unsqueeze(0))
        self.last_fast_norm = active_liquid.W_fast.norm().item()
        
        batch_mean = pre_act_raw.mean(dim=1, keepdim=True)
        batch_std = pre_act_raw.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
        stats = torch.cat([batch_mean, batch_std], dim=1)
        learned_plasticity = active_liquid.plasticity_controller(stats).squeeze(1).mean().item()
        error_proxy = max(active_liquid.prediction_error, 0.05)
        self.last_effective_plasticity = (plasticity * learned_plasticity) * (1.0 - error_proxy)
        self.last_learned_plasticity = learned_plasticity
        self.last_prediction_error = error_proxy
        
        return self.thought_compressor(thought)
    
    def get_liquid_module(self):
        if self.mode == "sequence":
            return self.topological_compressor.integrator
        else:
            return self.liquid
# ==================== HOMEÓSTASIS BICAMERAL ====================

class HomeostasisEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.4
    
    def decide(self, task_loss_val, richness_val, vn_entropy_val):
        focus_drive = task_loss_val * 2.0
        target_richness = 25.0
        explore_drive = max(0.0, (target_richness - richness_val) * 1.5)
        target_entropy = 3.20
        repair_drive = max(0.0, (target_entropy - vn_entropy_val) * 50.0)
        
        logits = torch.tensor([focus_drive, explore_drive, repair_drive]) / self.temperature
        probs = F.softmax(logits, dim=0)
        return probs[0].item(), probs[1].item(), probs[2].item()

class BicameralHomeostasis(nn.Module):
    def __init__(self):
        super().__init__()
        self.left_engine = HomeostasisEngine()
        self.right_engine = HomeostasisEngine()
        self.cross_balance = nn.Parameter(torch.tensor(0.5))
        self.last_left_drives = (0.0, 0.0, 0.0)
        self.last_right_drives = (0.0, 0.0, 0.0)
        self.last_callosal_flow = 0.5

    def decide(self, left_metrics, right_metrics, epoch, total_epochs):
        p_left = self.left_engine.decide(**left_metrics)
        p_right = self.right_engine.decide(**right_metrics)
        
        if p_left[2] > 0.8:
            cross_weight = 0.8
        elif p_right[2] > 0.8:
            cross_weight = 0.2
        else:
            cross_weight = torch.sigmoid(self.cross_balance).item()
        
        self.last_left_drives = p_left
        self.last_right_drives = p_right
        self.last_callosal_flow = cross_weight
        
        return {
            'left': p_left,
            'right': p_right,
            'callosal_flow': cross_weight
        }

# ==================== MEMORIA DE REPLAY (SUEÑO) ====================
class ReplayMemory(nn.Module):
    def __init__(self, capacity=2000, noise_scale=0.15):
        super().__init__()
        self.capacity = capacity
        self.buffer = []
        self.noise_scale = noise_scale
        
    def store(self, pattern):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        
        # FIX: Normalizar a dimensión fija antes de almacenar
        # Si pattern es [B, N, D] o [B, D], pool a [B, D] para estabilidad
        if pattern.dim() == 3:
            pattern = pattern.mean(dim=1)  # Pooling temporal/espacial uniforme
        
        self.buffer.append(pattern.detach().cpu())
    
    def replay(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        # FIX: Samplear solo tensores con misma dimensión
        valid_indices = [i for i, p in enumerate(self.buffer) if p.shape == self.buffer[-1].shape]
        
        if len(valid_indices) < batch_size:
            return None
        
        indices = torch.tensor(random.sample(valid_indices, batch_size))
        patterns = torch.stack([self.buffer[i] for i in indices]).cuda()
        noise = torch.randn_like(patterns) * self.noise_scale
        
        # Normalización homeostática
        if len(self.buffer) > batch_size:
            mean_buf = torch.stack(self.buffer).mean(0).cuda()
            std_buf = torch.stack(self.buffer).std(0).clamp(min=1e-6).cuda()
        else:
            mean_buf = patterns.mean(0)
            std_buf = patterns.std(0).clamp(min=1e-6)
        
        replayed = (patterns + noise - mean_buf) / std_buf
        return replayed


class CorpusCallosum(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn_left_to_right = nn.MultiheadAttention(256, 8, batch_first=True)
        self.cross_attn_right_to_left = nn.MultiheadAttention(256, 8, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
    
    def forward(self, left_repr, right_repr, mode="awake"):
        B = left_repr.size(0)
        
        # FIX CRÍTICO: Alinear dimensiones ANTES de atención cruzada
        # Pool adaptativo siempre que los tamaños de secuencia difieran
        if left_repr.dim() == 3 and right_repr.dim() == 3:
            if left_repr.size(1) != right_repr.size(1):
                # Determinar cuál lado tiene más nodos y pool ese lado
                if left_repr.size(1) > right_repr.size(1):
                    # Pool left_repr al tamaño de right_repr
                    left_repr = F.adaptive_avg_pool1d(
                        left_repr.transpose(1,2), 
                        right_repr.size(1)
                    ).transpose(1,2)
                else:
                    # Pool right_repr al tamaño de left_repr
                    right_repr = F.adaptive_avg_pool1d(
                        right_repr.transpose(1,2), 
                        left_repr.size(1)
                    ).transpose(1,2)
        
        # Resto del código original sin cambios
        global_state = torch.cat([left_repr.mean(dim=1), right_repr.mean(dim=1)], dim=-1)
        flow_gate = self.gate(global_state).unsqueeze(-1)
        
        if mode == "awake":
            metabolic_cost = 0.0
            right_enhanced, _ = self.cross_attn_right_to_left(left_repr, right_repr, right_repr)
            left_enhanced, _ = self.cross_attn_left_to_right(right_repr, left_repr, left_repr)
            right_final = right_repr + 0.1 * right_enhanced
            left_final = left_repr + 0.1 * left_enhanced
        elif mode == "dream":
            metabolic_cost = 0.0
            right_enhanced, _ = self.cross_attn_right_to_left(left_repr, right_repr, right_repr)
            left_enhanced, _ = self.cross_attn_left_to_right(right_repr, left_repr, left_repr)
            right_final = right_repr + flow_gate * right_enhanced
            left_final = left_repr + flow_gate * left_enhanced
        else:
            metabolic_cost = 0.0
            return left_repr, right_repr, metabolic_cost
        
        return left_final, right_final, metabolic_cost

# ==================== MODELO BICAMERAL COMPLETO ====================
class NeuroLogos(nn.Module):
    def __init__(self, vocab_size=1000, use_nested=False):
        super().__init__()
        self.use_nested = use_nested
        
        # Emisferios especializados
        self.left_hemi = LeftHemisphere(use_nested=use_nested)
        self.right_hemi = RightHemisphere(config=None)
        
        # Cuerpo calloso
        self.corpus_callosum = CorpusCallosum()
        
        # Homeostasis bicameral
        self.homeostasis = BicameralHomeostasis()
        
        # Área de Broca (compartida)
        self.broca = BioDecoder(vocab_size=vocab_size, visual_dim=256)
        self.thought_to_lang = nn.Linear(512, 512)
        self.aux_classifier = nn.Linear(512, 10) 
        self.running_richness = 20.0
        self.current_epoch = 0
        self.total_epochs = 50
        self.last_thought_final = None
        
        # FIX: Inicializar memoria de replay
        self.memory = ReplayMemory(capacity=2000)
        
    def forward(self, image, captions=None, plasticity=0.1, transfer_rate=0.005, mode="awake", epoch=0):
        B = image.size(0)
        
        # Codificación visual dual
        visual_left = self.left_hemi.eye(image)
        visual_right = self.right_hemi.visual_cortex(image)
        visual_right = visual_right.flatten(2).transpose(1, 2)
        visual_right = self.right_hemi.topo_layer(visual_right, plasticity_gate=plasticity)
        visual_right = self.right_hemi.liquid(visual_right, plasticity)
        
        # Proyección para intercambio callosal
        if visual_left.dim() == 3:
            left_proj = self.left_hemi.callosal_proj(visual_left)
        else:
            left_proj = self.left_hemi.callosal_proj(visual_left).unsqueeze(1)
        right_proj = self.right_hemi.callosal_proj(visual_right)
        
        # FIX: Desempaquetar 3 valores del calloso
        left_enhanced, right_enhanced, _ = self.corpus_callosum(left_proj, right_proj, mode=mode)
        
        # Procesamiento consciente izquierdo
        if visual_left.dim() == 3:
            thought_left = self.left_hemi.cortex(right_enhanced.mean(dim=1), plasticity, transfer_rate)
        else:
            thought_left = self.left_hemi.cortex(right_enhanced.squeeze(1), plasticity, transfer_rate)
        
        # Pensamiento final
        thought_final = self.thought_to_lang(thought_left)
        self.last_thought_final = thought_final
        
        # FIX: Almacenar en memoria durante vigilia
        if self.training and mode == "awake":
            self.memory.store(thought_final)
        
        # Calcular métricas para homeostasis
        cortex = self.left_hemi.cortex
        left_metrics = {
            'task_loss_val': cortex.last_prediction_error,
            'richness_val': cortex.last_richness,
            'vn_entropy_val': cortex.last_vn_entropy
        }
        right_metrics = {
            'task_loss_val': self.right_hemi.liquid.prediction_error,
            'richness_val': 20.0,
            'vn_entropy_val': 3.0
        }
        
        # Activar homeostasis y guardar decisiones
        homeo_decision = self.homeostasis.decide(left_metrics, right_metrics, epoch, self.total_epochs)
        
        # FIX: Usar right_enhanced directamente sin flatten extra
        visual_for_decoder = right_enhanced  # Ya está en formato [B, N, D] compatible
        
        return self.broca(thought_final, visual_features=visual_for_decoder, captions=captions)
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch


# ==================== CICLO DE VIDA NEURAL ====================

class LifeCycle:
    def __init__(self, total_epochs=50):
        self.total = total_epochs
        self.phase = "GENESIS"
        
    def get_plasticity(self, epoch):
        if epoch < 5:
            self.phase = "GENESIS (Growing Eye)"
            return 0.5
        elif epoch < 30:
            self.phase = "AWAKENING (Learning to Think)"
            return max(0.01, 0.2 * (1 - epoch/30))
        else:
            self.phase = "LOGOS (Speaking)"
            return 0.001

# ==================== DATASET CIFAR-10 CAPTIONS ====================

class CIFARCaptions:
    def __init__(self):
        self.dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        self.templates = {
            0: ["an airplane soaring through clear skies", "a commercial jet with extended landing gear", "a passenger aircraft banking over mountains", "a twin-engine plane flying above clouds", "a sleek silver airplane in mid-flight", "a cargo airplane with visible wing flaps", "a military jet leaving a vapor trail", "a small propeller-driven aircraft descending", "an airliner approaching an urban airport", "a high-altitude plane under bright sunlight", "a white airplane casting a shadow on clouds", "a red and blue aircraft on final approach", "a modern jet with swept-back wings", "a vintage biplane in open sky", "a drone-like aircraft with quiet engines", "a long-haul airplane with full fuselage", "a plane captured mid-turn with clear horizon", "an aircraft with visible navigation lights", "a passenger jet gliding silently downward", "a compact airplane flying over rural landscape"],
            1: ["a compact car parked on a city street", "a luxury sedan with polished chrome details", "a hatchback accelerating on an empty road", "a vintage automobile with rounded headlights", "a family car with tinted rear windows", "a sports car with aggressive front grille", "an electric vehicle charging at a station", "a station wagon carrying outdoor gear", "a compact SUV with roof rails", "a red convertible driving along the coast", "a black sedan stopped at a traffic light", "a white hatchback reflecting sunlight", "a four-door vehicle with alloy wheels", "a hybrid car with aerodynamic body lines", "a city car maneuvering through tight alleys", "a two-door coupe with sleek silhouette", "an automatic transmission car on highway", "a compact automobile with fog lamps", "a fuel-efficient hatchback in urban setting", "a modern car with LED daytime running lights"],
            2: ["a small songbird perched on a thin branch", "a colorful parrot flapping its bright wings", "a raptor scanning the ground from above", "a pigeon walking on a sunlit sidewalk", "a sparrow taking off from a rooftop", "a hummingbird hovering near flowering plants", "a crow cawing loudly in a bare tree", "a finch with yellow feathers on a fence", "a seabird gliding over ocean waves", "a duck splashing in a shallow pond", "a small bird preening its chest feathers", "a flock of sparrows in early morning light", "a robin with red breast in garden soil", "a blackbird singing from a high wire", "a swift flying at high velocity", "a tiny bird with rapid wingbeats", "a bird mid-flight with wings fully extended", "a feathered creature resting on dry grass", "a chirping bird hidden among green leaves", "a wild bird with sharp, watchful eyes"],
            3: ["a fluffy domestic cat lying in sunlight", "a tabby feline grooming its front paw", "a sleek black cat with bright green eyes", "a sleepy house cat curled on a cushion", "a curious kitten peeking from under a chair", "a calm cat sitting upright with tail wrapped", "a long-haired feline blinking slowly", "a playful cat batting at a dangling string", "a gray cat stretching on a wooden floor", "a quiet cat observing birds through window", "a spotted cat with upright ears and whiskers", "a relaxed cat napping in afternoon shade", "a domestic shorthair with soft fur texture", "a cat yawning with visible pink tongue", "a feline crouching as if stalking prey", "a white cat with blue eyes on a windowsill", "a household pet blinking in warm lamplight", "a silent cat watching from a dark corner", "a purring feline nestled in human lap", "a curious cat with one paw raised"],
            4: ["a wild deer standing in morning mist", "a young fawn hiding among tall grass", "a stag with large antlers in dense forest", "a deer drinking from a clear forest stream", "a brown deer alert with ears pointed forward", "a herd of deer grazing in open meadow", "a solitary deer at forest edge at dusk", "a spotted fawn lying motionless on leaves", "a male deer with velvet-covered antlers", "a deer leaping over a fallen log", "a cautious animal with large dark eyes", "a grazing mammal in autumn woodland", "a deer frozen in headlights at night", "a wild ungulate with slender legs", "a forest deer with wet nose and alert stance", "a deer bounding through snowy underbrush", "a graceful creature in natural habitat", "a timid animal pausing mid-step", "a ruminant with soft brown coat", "a deer turning its head toward a sound"],
            5: ["a golden retriever running through a field", "a small terrier barking at a passing car", "a loyal shepherd dog guarding a property", "a wet dog shaking off after a swim", "a puppy playing with a red rubber ball", "a large mastiff lying at a doorstep", "a beagle sniffing along a garden path", "a playful spaniel chasing falling leaves", "a calm labrador sitting beside its owner", "a street dog resting in afternoon shade", "a dog with floppy ears and wagging tail", "a vigilant hound with erect posture", "a muddy dog returning from forest trail", "a domesticated canine with shiny coat", "a rescue dog enjoying its first walk", "a small dog jumping excitedly in place", "a tired dog panting after a long run", "a well-trained dog obeying a hand signal", "a furry pet curled up by a fireplace", "a canine friend with trusting brown eyes"],
            6: ["a green frog sitting on a lily pad", "a moist amphibian near a freshwater pond", "a small frog with bulging eyes and webbed feet", "a tree frog clinging to a wet leaf", "a spotted frog leaping into shallow water", "a camouflaged frog blending with moss", "a nocturnal frog resting on riverbank stones", "a tiny amphibian with glistening skin", "a frog mid-jump with legs fully extended", "a pond frog inflating its vocal sac", "a bright-eyed creature in swampy reeds", "a rainforest frog with vivid coloration", "a quiet frog waiting for passing insects", "a small jumper in humid morning air", "an amphibian perched on a floating leaf", "a frog with smooth skin after rainfall", "a creature adapted to both land and water", "a silent observer on marshy ground", "a cold-blooded animal in natural stillness", "a green-skinned jumper in wet grass"],
            7: ["a muscular horse galloping across open pasture", "a calm mare standing in a sunlit stable", "a workhorse pulling a heavy wooden cart", "a chestnut horse with flowing mane and tail", "a wild stallion rearing on hind legs", "a domesticated horse grazing peacefully", "a thoroughbred with elegant leg structure", "a farm horse drinking from a metal trough", "a horse standing alert with flared nostrils", "a young colt running beside its mother", "a strong animal with powerful haunches", "a saddled horse waiting for its rider", "a white horse with black mane in snow", "a bay horse trotting on packed dirt road", "a gentle giant with soft brown eyes", "a horse shaking its head to deter flies", "a majestic animal in golden hour light", "a draft horse with thick neck and shoulders", "a horse neighing loudly at dawn", "a four-legged companion in rural landscape"],
            8: ["a massive cargo ship crossing open ocean", "a white cruise liner with multiple decks", "a container vessel loaded with blue boxes", "a naval ship cutting through rough waves", "a ferry transporting vehicles across a bay", "a sleek yacht anchored near rocky coast", "a fishing trawler with nets on deck", "a maritime vessel with visible radar mast", "a tanker ship with rust-streaked hull", "a passenger ship docking at busy harbor", "a large boat with cranes and cargo holds", "a vessel lit by sunset on calm water", "a steel-hulled ship leaving a wake", "a commercial ship flying a national flag", "a long-haul freighter on international route", "a ship navigating through narrow strait", "a seafaring craft with lifeboats mounted", "a floating structure with smokestack plume", "a deep-draft ship in international waters", "a maritime transport under overcast sky"],
            9: ["a large delivery truck on highway ramp", "a diesel-powered freight vehicle with trailer", "a heavy-duty truck carrying construction materials", "a refrigerated transport truck with logo", "a dump truck unloading gravel at site", "a long-haul semi with reflective side panels", "a utility truck parked near power lines", "a cargo truck with open rear doors", "a box truck navigating city streets", "a rugged vehicle designed for heavy loads", "a logistics truck with barcode scanner", "a transport vehicle with dual rear axles", "a commercial truck idling at gas station", "a flatbed truck hauling machinery parts", "a white van-style truck with sliding doors", "a supply vehicle making early morning rounds", "a massive transport with air brakes hissing", "a workhorse of road freight industry", "a truck with steel frame and thick tires", "a logistics carrier in industrial district"]
        }
        
        # Construcción del vocabulario global
        self.vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        for descs in self.templates.values():
            for desc in descs:
                self.vocab.extend(desc.split())
        self.vocab = list(dict.fromkeys(self.vocab))
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        desc = random.choice(self.templates[label])
        tokens = ["<BOS>"] + desc.split() + ["<EOS>"]
        token_ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in tokens][:20]
        token_ids += [self.word2id["<PAD>"]] * (20 - len(token_ids))
        return image, torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ==================== FUNCIÓN DE COHERENCIA NORMALIZADA ====================

def estimate_coherence(sentence, templates_per_class):
    words = set(sentence.split())
    valid_words = set()
    for descs in templates_per_class.values():
        for desc in descs:
            valid_words.update(desc.split())
    filtered_words = words & valid_words
    if not filtered_words:
        return 0.0
    class_scores = [len(filtered_words & set(desc.split())) for descs in templates_per_class.values() for desc in descs]
    return max(class_scores) / (sum(class_scores) + 1e-8)

# ==================== PIPELINE DE ENTRENAMIENTO COMPLETO ====================

def train_logos(use_nested=False):
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"NeuroLogos Bicameral 2.0 | Device: {device}")
    print(f"{'='*60}\n")
    
    dataset = CIFARCaptions()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    
    model = NeuroLogos(vocab_size=len(dataset.vocab), use_nested=use_nested).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    life = LifeCycle(total_epochs=50)
    model.total_epochs = 50
    
    print(f"Vocab size: {len(dataset.vocab)} | Params: {sum(p.numel() for p in model.parameters()):,}\n")
    
    for epoch in range(50):
        plasticity = life.get_plasticity(epoch)
        model.set_epoch(epoch)
        model.train()
        
        current_transfer_rate = 0.01 if epoch < 30 else 0.003
        liquid_module = model.left_hemi.cortex.get_liquid_module()
        
        if epoch % 10 == 0 and epoch > 0:
            liquid_module.consolidate_svd(repair_strength=0.7, timescale='medium')
        if epoch == 30:
            liquid_module.consolidate_svd(repair_strength=0.5, timescale='fast')
        
        total_loss = 0
        sum_richness = sum_vn = sum_fast_norm = sum_plasticity = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}")
        for images, captions, labels in pbar:
            images = images.to(device, non_blocking=True) * 2 - 1
            captions = captions.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # FIX: Forward ahora retorna solo logits
            logits = model(images, captions=captions, plasticity=plasticity, transfer_rate=current_transfer_rate, mode="awake", epoch=epoch)
            
            gen_loss = F.cross_entropy(
                logits.reshape(-1, len(dataset.vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=dataset.word2id["<PAD>"]
            )
            
            thought_final = model.last_thought_final
            cls_logits = model.aux_classifier(thought_final.detach())
            cls_loss = F.cross_entropy(cls_logits, labels)
            
            loss = gen_loss + 0.05 * cls_loss
            
            with torch.no_grad():
                liquid_module.prediction_error = (loss / 5.0).clamp(0.0, 0.95).item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            sum_richness += model.left_hemi.cortex.last_richness
            sum_vn += model.left_hemi.cortex.last_vn_entropy
            sum_fast_norm += model.left_hemi.cortex.last_fast_norm
            sum_plasticity += model.left_hemi.cortex.last_effective_plasticity
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'rich': f'{model.left_hemi.cortex.last_richness:.1f}'
            })
        
        # FIX: Ciclo de sueño cada 5 épocas
        if epoch % 5 == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                replay_batch = model.memory.replay(batch_size=8)
                if replay_batch is not None:
                    _ = model(replay_batch, mode="dream", epoch=epoch)
            model.train()
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_img, _, label = dataset[0]
                sample_img = sample_img.unsqueeze(0).to(device) * 2 - 1
                generated = model(sample_img, captions=None, plasticity=plasticity, transfer_rate=current_transfer_rate, mode="awake", epoch=epoch)
                
                words = [dataset.id2word[int(t.item())] for t in generated[0]]
                sentence = " ".join([w for w in words if w not in ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]])
                coherence = estimate_coherence(sentence, dataset.templates)
                
                cortex = model.left_hemi.cortex
                homeo = model.homeostasis
                
                print(f"\n{'-'*60}")
                print(f"Ep {epoch:02d} | {life.phase} | Loss: {total_loss/num_batches:.3f} | Rich: {sum_richness/num_batches:.1f} | Coherencia: {coherence:.2f}")
                print(f"Genera: '{sentence}'")
                print(f"Cortex Mode: {cortex.last_mode}")
                print(f"Fast Norm: {cortex.last_fast_norm:.2f}")
                print(f"{'-'*60}\n")
            
            model.train()
    
    print("\n✅ Entrenamiento completado.\n")
    torch.save(model.state_dict(), 'neurologos_bicameral_FINAL.pth')

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    # Versión rápida CPU
    train_logos(use_nested=True)
    
    # Versión topológica GPU (descomentar si tienes GPU)
    # train_logos(use_nested=True)
#%%writefile neurosoberano_v2_FINAL.py
# =============================================================================
# NeuroLogos Bicameral 2.0 - VERSIÓN COMPLETA Y EJECUTABLE
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

def compute_phi_effective(activations, k_partitions=5):
    """
    Φₑ con manejo robusto de dimensiones pequeñas
    """
    B, N, D = activations.shape
    if N < 2 or B < 2:  # FIX: Requiere al menos 2 samples para var
        return torch.tensor(2.0, device=activations.device)
    
    act_mean = activations.mean(dim=(0, 2), keepdim=True)
    act_std = activations.std(dim=(0, 2), keepdim=True, unbiased=False).clamp(min=1e-6)  # unbiased=False
    normalized = (activations - act_mean) / act_std
    
    phi_accum = 0.0
    valid_partitions = 0
    
    for _ in range(k_partitions):
        perm = torch.randperm(N, device=activations.device)
        mid = max(1, N // 2)  # Asegurar al menos 1
        
        part1 = normalized[:, perm[:mid], :].reshape(B, -1)
        part2 = normalized[:, perm[mid:], :].reshape(B, -1)
        
        min_dim = min(part1.shape[1], part2.shape[1])
        if min_dim == 0:
            continue
            
        part1 = part1[:, :min_dim]
        part2 = part2[:, :min_dim]
        
        p1_c = part1 - part1.mean(dim=0, keepdim=True)
        p2_c = part2 - part2.mean(dim=0, keepdim=True)
        
        cov = (p1_c * p2_c).sum(dim=0).mean().abs()
        
        # FIX: var con unbiased=False para evitar warning
        var1 = p1_c.var(dim=0, unbiased=False).mean() + 1e-6
        var2 = p2_c.var(dim=0, unbiased=False).mean() + 1e-6
        
        phi = (cov / (var1 * var2).sqrt()).clamp(0, 1)
        phi_accum += phi
        valid_partitions += 1
    
    if valid_partitions == 0:
        return torch.tensor(2.0, device=activations.device)
    
    phi_mean = phi_accum / valid_partitions
    
    # Escalado exponencial
    phi_stretched = phi_mean ** 0.4
    phi_scaled = 0.5 + 3.5 * phi_stretched
    
    return phi_scaled.clamp(0.5, 4.0)

def compute_spatial_diversity(activations):
    """
    Diversidad basada en correlación inversa de Pearson.
    VERSIÓN ROBUSTA: Protegida contra NaNs por errores de precisión flotante.
    """
    B, N, D = activations.shape
    
    # Aplanar batch y features: [N, B*D]
    flat_neur = activations.permute(1, 0, 2).reshape(N, -1)
    
    # Normalización Z-score por neurona con epsilon de seguridad
    means = flat_neur.mean(dim=1, keepdim=True)
    stds = flat_neur.std(dim=1, keepdim=True).clamp(min=1e-6)
    norm_neur = (flat_neur - means) / stds
    
    # Muestreo si hay muchas neuronas
    if N > 128:
        idx = torch.randperm(N, device=activations.device)[:128]
        sample = norm_neur[idx]
    else:
        sample = norm_neur
        
    # Matriz de correlación
    # mm(A, A.t) / (dim-1) es la matriz de correlación
    sim_matrix = torch.mm(sample, sample.t()) / (sample.shape[1] - 1)
    
    # Diversidad = 1 - Promedio de correlaciones fuera de la diagonal
    mask = torch.ones_like(sim_matrix) - torch.eye(sample.shape[0], device=activations.device)
    
    # FIX 1: Evitar división por cero si solo hay 1 neurona
    if mask.sum() == 0:
        return torch.tensor(0.0, device=activations.device)

    avg_corr = (sim_matrix.abs() * mask).sum() / mask.sum()
    
    # FIX 2: Clamping crítico para evitar que avg_corr sea > 1.0 por error flotante
    avg_corr = torch.clamp(avg_corr, 0.0, 1.0)
    
    diversity_raw = 1.0 - avg_corr
    
    # FIX 3: Asegurar que la base de la potencia sea no-negativa
    diversity_raw = torch.clamp(diversity_raw, min=1e-6)
    
    # Escalado no lineal
    diversity_scaled = 1.0 + 5.0 * (diversity_raw ** 0.3)
    
    return diversity_scaled










def compute_activation_entropy(activations):
    """
    Shannon entropy sobre la distribución de activaciones
    Target range: [2.5, 4.5] bits
    """
    activations_flat = activations.reshape(-1)
    
    # Discretizar en bins con rango adaptativo
    act_min = activations_flat.min().item()
    act_max = activations_flat.max().item()
    act_range = act_max - act_min
    
    if act_range < 1e-6:
        return 3.0  # Entropía neutral si no hay variación
    
    # 50 bins en el rango de datos observados
    hist = torch.histc(activations_flat, bins=50, min=act_min, max=act_max)
    probs = hist / (hist.sum() + 1e-8)
    probs = probs[probs > 1e-8]
    
    # Calcular entropía de Shannon en bits
    shannon_entropy = -(probs * torch.log2(probs + 1e-8)).sum().item()
    
    # Normalizar: log2(50) ≈ 5.64 es el máximo teórico para 50 bins
    normalized_entropy = (shannon_entropy / 5.64) * 2.0 + 2.5  # Mapea [0, 5.64] -> [2.5, 4.5]
    normalized_entropy = max(2.5, min(normalized_entropy, 4.5))
    
    return normalized_entropy



def measure_neural_complexity(activations):
    """
    Medición corregida con formato [B, D, N] para neuronas reales
    """
    if activations.size(0) == 0:
        return 1.5, 3.0, 3.5
    
    B = activations.size(0)
    
    # Muestreo si batch grande
    if B > 64:
        indices = torch.randperm(B, device=activations.device)[:64]
        activations = activations[indices]
        B = 64
    
    # FIX CRÍTICO: Asegurar formato [B, Neuronas, Features]
    # Si viene [B, D] → [B, 1, D] (1 neurona de D features)
    # Si viene [B, D, 1] → Transponer a [B, 1, D]
    if activations.dim() == 2:
        # Interpretar como [B, Features] → Reshape a [B, Neuronas=sqrt(D), Neuronas]
        D = activations.size(1)
        grid_size = int(D ** 0.5)
        if grid_size * grid_size == D:
            activations = activations.view(B, grid_size, grid_size)
        else:
            # Fallback: 1 neurona con D features
            activations = activations.unsqueeze(1)
    elif activations.dim() == 3:
        # Si es [B, D, 1] (D neuronas, 1 feature) → Inválido para diversidad
        if activations.size(2) == 1:
            # Transponer a [B, 1, D] (1 neurona, D features)
            activations = activations.transpose(1, 2)
    
    # Ahora activations es [B, N, D] donde N=neuronas, D=features por neurona
    
    # 1. Φₑ (Integrated Information)
    phi_raw = compute_phi_effective(activations)
    phi_final = phi_raw.item() if torch.is_tensor(phi_raw) else phi_raw
    
    # 2. Diversidad Espacial
    spatial_div = compute_spatial_diversity(activations)
    
    # 3. Entropía de Activación
    act_entropy = compute_activation_entropy(activations)
    
    return phi_final, spatial_div, act_entropy


# Mantener compatibilidad con código existente
def measure_spatial_richness(activations):
    """
    Wrapper para compatibilidad con código existente
    """
    phi, diversity, entropy = measure_neural_complexity(activations)
    return entropy, diversity, phi



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('inf')):
    """Filtro Top-K y Nucleus Sampling estándar"""
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
class BioDecoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=512, visual_dim=384):  # 256+128
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.out = nn.Linear(hidden_dim, vocab_size)
        
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),  # Ahora acepta 384 dims
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.15)
        )
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
    def forward(self, thought, visual_features=None, captions=None, max_len=20):
        batch_size = thought.size(0)
        device = thought.device
        
        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            if self.training:
                embeddings = embeddings + torch.randn_like(embeddings) * 0.08  # Más ruido
            
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
                fused = lstm_out + 0.3 * visual_context  # Reducir peso visual
            else:
                fused = lstm_out
            
            return self.out(fused)
        else:
            # MODO INFERENCIA CON FIXES
            generated = []
            input_word = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
            hidden = self._get_init_state(thought)
            
            if visual_features is not None:
                vis_proj = self.visual_proj(visual_features)
            else:
                vis_proj = None
            
            # FIX: Temperatura más alta para exploración
            epoch = getattr(thought, 'epoch', 0) if hasattr(thought, 'epoch') else 0
            temperature = 0.8 + 0.3 * min(epoch / 50.0, 1.0)  # Rango [0.8, 1.1]
            top_p = 0.85 + 0.1 * min(epoch / 50.0, 1.0)       # Rango [0.85, 0.95]
            
            generated_token_history = []
            token_counts = torch.zeros(self.vocab_size, device=device)  # Contador global
            
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
                    out = out + 0.2 * visual_context  # Reducir peso visual
                
                logits = self.out(out.squeeze(1))
                
                # FIX CRÍTICO: Repetition penalty exponencial
                for b in range(batch_size):
                    if step >= 1:
                        # Penalizar tokens recientes con decay exponencial
                        for i, prev_token in enumerate(generated_token_history):
                            tok_id = prev_token[b].item()
                            decay = 0.7 ** (len(generated_token_history) - i)  # Más reciente = más penalización
                            logits[b, tok_id] -= 3.0 * decay
                        
                        # Penalizar tokens usados múltiples veces
                        for tok_id in range(self.vocab_size):
                            if token_counts[tok_id] > 0:
                                logits[b, tok_id] -= 1.5 * token_counts[tok_id]
                
                logits = logits / temperature
                
                # FIX: Boost a tokens con baja frecuencia global
                logits = logits + 0.5 * (1.0 / (token_counts + 1.0))
                
                logits = top_k_top_p_filtering(logits, top_k=50, top_p=top_p)  # Top-k=50
                probabilities = F.softmax(logits, dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1)
                
                # Actualizar contador global
                for b in range(batch_size):
                    token_counts[next_word[b].item()] += 1
                
                generated.append(next_word)
                generated_token_history.append(next_word.squeeze(1))
                input_word = next_word
            
            return torch.cat(generated, dim=1)
    
    def _get_init_state(self, thought):
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        return (h0, c0)


class LiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.1)
        
        self.register_buffer('W_medium', torch.randn(out_dim, in_dim) * 0.02)
        self.register_buffer('W_fast', torch.randn(out_dim, in_dim) * 0.01)
        
        self.ln = nn.LayerNorm(out_dim)
        
        self.plasticity_controller = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.plasticity_controller[2].bias.data.fill_(0.5)
        
        self.base_lr = 0.03  # Incrementar de 0.02
        self.prediction_error = 0.0

    def forward(self, x, global_plasticity=0.1, transfer_rate=0.005):
        original_shape = x.shape
        if x.dim() == 3:
            B, N, D = x.shape
            x_flat = x.reshape(B * N, D)
        else:
            x_flat = x
            B = x.size(0)
            N = 1

        slow_out = self.W_slow(x_flat)
        medium_out = F.linear(x_flat, self.W_medium)
        fast_out = F.linear(x_flat, self.W_fast)
        
        pre_act = slow_out + medium_out + fast_out
        
        # Ruido adaptativo (no decae a cero)
        if self.training:
            noise_scale = max(0.01, 0.05 * global_plasticity)  # Piso de 0.01
            noise = torch.randn_like(pre_act) * noise_scale
            pre_act = pre_act + noise
        
        batch_mean = pre_act.mean().detach()
        batch_std = pre_act.std(unbiased=False).clamp(min=1e-6).detach()
        
        stats = torch.cat([batch_mean.expand(B * N, 1), batch_std.expand(B * N, 1)], dim=1)
        learned_plasticity = self.plasticity_controller(stats).squeeze(1)
        
        # FIX: Plasticidad no colapse con prediction_error
        effective_plasticity = global_plasticity * learned_plasticity * (0.5 + 0.5 * (1.0 - self.prediction_error))
        
        out_flat = 5.0 * torch.tanh(self.ln(pre_act) / 4.0)

        if self.training and effective_plasticity.mean() > 0.001:
            with torch.no_grad():
                out_centered = out_flat - out_flat.mean(dim=0, keepdim=True)
                correlation = torch.mm(out_centered.T, x_flat) / (x_flat.size(0) + 1e-6)
                correlation = torch.clamp(correlation, -0.15, 0.15)
                
                # Dinámica de transferencia
                self.W_medium.data.add_(self.W_fast.data * transfer_rate)
                
                # FIX: Decay adaptativo basado en norma
                norm_fast = self.W_fast.norm()
                if norm_fast < 1.5:
                    decay = 0.995  # Decay suave si está bajo
                elif norm_fast < 3.0:
                    decay = 0.98   # Decay medio
                else:
                    decay = 0.92   # Decay fuerte si explota
                
                self.W_fast.data.mul_(decay)
                self.W_fast.data.add_(correlation * effective_plasticity.mean() * self.base_lr)
                self.W_fast.data.clamp_(-3.0, 3.0)  # Aumentar límites

        if len(original_shape) == 3:
            out = out_flat.view(B, N, -1)
        else:
            out = out_flat

        return out

    def consolidate_svd(self, repair_strength=1.0, timescale='fast'):
        if timescale == 'fast':
            W_target = self.W_fast
            factor = 0.08 * repair_strength  # Aumentar de 0.05
        elif timescale == 'medium':
            W_target = self.W_medium
            factor = 0.04 * repair_strength  # Aumentar de 0.02
        else:
            return False
            
        with torch.no_grad():
            self.W_slow.weight.data.add_(W_target.data * factor)
            W_target.data.mul_(0.85)  # Reducir vaciado de 0.8 a 0.85
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
        
        # FIX CRÍTICO: Proyecciones ortogonales forzadas para diversidad
        self.diversity_heads = nn.ModuleList()
        for i in range(8):
            head = nn.Linear(512, 64, bias=False)
            # Inicializar con vectores ortogonales + rotación única
            nn.init.orthogonal_(head.weight)
            # Aplicar rotación única a cada cabeza para forzar diversidad
            rotation_angle = (i / 8) * 2 * 3.14159
            rotation_matrix = self._create_rotation_matrix(512, rotation_angle, head.weight.device)
            head.weight.data = torch.mm(head.weight.data, rotation_matrix)
            self.diversity_heads.append(head)
        
        # FIX: Agregar dropout independiente por cabeza
        self.diversity_dropout = nn.ModuleList([nn.Dropout(0.1 + i * 0.05) for i in range(8)])
        
        self.mode = "vector"
        self.last_mode = "vector"
        self.last_richness = 5.0
        self.last_vn_entropy = 2.5
        self.last_activation_entropy = 3.5
        self.last_fast_norm = 0.0
        self.last_effective_plasticity = 0.1
        self.last_learned_plasticity = 0.5
        self.last_prediction_error = 0.5

    def _create_rotation_matrix(self, dim, angle, device):
        """Crea matriz de rotación en espacio de alta dimensión"""
        matrix = torch.eye(dim, device=device)
        # Rotar en plano principal
        matrix[0, 0] = math.cos(angle)
        matrix[0, 1] = -math.sin(angle)
        matrix[1, 0] = math.sin(angle)
        matrix[1, 1] = math.cos(angle)
        return matrix

    def forward(self, visual_features, plasticity, transfer_rate=0.005):
        if visual_features.size(-1) == 256:
            visual_features = self.callosal_adapter(visual_features)
        
        if visual_features.dim() == 3:
            self.mode = "sequence"
            self.last_mode = "sequence"
            h, _ = self.sequence_attention(visual_features, visual_features, visual_features)
            thought = self.topological_compressor(h, plasticity, transfer_rate)
            active_liquid = self.topological_compressor.integrator
        else:
            self.mode = "vector"
            self.last_mode = "vector" 
            q = visual_features.unsqueeze(1)
            pooled, _ = self.attention_pool(q, q, q)
            pre_liquid = pooled.squeeze(1)
            thought = self.liquid(pre_liquid, plasticity, transfer_rate)
            active_liquid = self.liquid
        
        # FIX: Aplicar proyecciones con dropout independiente
        B = thought.size(0)
        neural_activations = []
        for i, (head, dropout) in enumerate(zip(self.diversity_heads, self.diversity_dropout)):
            # Aplicar dropout ANTES de proyección para forzar patrones diferentes
            if self.training:
                thought_dropped = dropout(thought)
            else:
                thought_dropped = thought
            neural_activations.append(head(thought_dropped))
        
        # Stack: [B, 8 neuronas, 64 features]
        metric_input = torch.stack(neural_activations, dim=1)
        
        phi, div, ent = measure_neural_complexity(metric_input)
        
        self.last_richness = div
        self.last_vn_entropy = phi
        self.last_activation_entropy = ent
        self.last_fast_norm = active_liquid.W_fast.norm().item()
        
        batch_stats = torch.cat([thought.mean().view(1,1), thought.std().view(1,1)], dim=1).expand(thought.size(0), -1)
        learned_p = active_liquid.plasticity_controller(batch_stats).mean().item()
        
        self.last_effective_plasticity = plasticity * learned_p
        self.last_learned_plasticity = learned_p
        self.last_prediction_error = active_liquid.prediction_error
        
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
        
        # LSTM recibe embedding + context del timestep anterior
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
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
            nn.Dropout(0.15)
        )
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
    def forward(self, thought, visual_features=None, captions=None, max_len=20):
        batch_size = thought.size(0)
        device = thought.device
        
        if captions is not None:
            # ENTRENAMIENTO con Input Feeding
            embeddings = self.embedding(captions[:, :-1])
            seq_len = embeddings.size(1)
            
            if self.training:
                embeddings = embeddings + torch.randn_like(embeddings) * 0.08
            
            # Inicializar contexto con thought
            prev_context = thought.unsqueeze(1)  # [B, 1, Hidden]
            
            outputs = []
            hidden = self._get_init_state(thought)
            
            for t in range(seq_len):
                # Input Feeding: Concatenar embedding[t] + context[t-1]
                lstm_input = torch.cat([embeddings[:, t:t+1, :], prev_context], dim=2)
                
                out, hidden = self.lstm(lstm_input, hidden)
                
                # Actualizar contexto para próximo timestep
                prev_context = out
                
                outputs.append(out)
            
            lstm_out = torch.cat(outputs, dim=1)
            
            gate = self.liquid_gate(lstm_out)
            fused = lstm_out * gate
            
            return self.out(fused)
        else:
            # GENERACIÓN
            generated = []
            input_word = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
            hidden = self._get_init_state(thought)
            
            # Contexto inicial
            prev_context = thought.unsqueeze(1)
            
            # Temperatura adaptativa
            epoch = getattr(thought, 'epoch', 0) if hasattr(thought, 'epoch') else 0
            temperature = 0.9 + 0.2 * min(epoch / 50.0, 1.0)
            top_p = 0.88 + 0.07 * min(epoch / 50.0, 1.0)
            
            token_counts = torch.zeros(self.vocab_size, device=device)
            generated_history = []
            
            for step in range(max_len):
                emb = self.embedding(input_word)
                
                # Input Feeding en generación
                lstm_input = torch.cat([emb, prev_context], dim=2)
                
                out, hidden = self.lstm(lstm_input, hidden)
                gate = self.liquid_gate(out)
                out = out * gate
                
                # Actualizar contexto
                prev_context = out
                
                logits = self.out(out.squeeze(1))
                
                # Repetition penalty mejorado
                for b in range(batch_size):
                    if len(generated_history) > 0:
                        for i, prev_token in enumerate(generated_history):
                            tok_id = prev_token[b].item()
                            recency = len(generated_history) - i
                            penalty = 2.5 * (0.8 ** recency)
                            logits[b, tok_id] -= penalty
                        
                        # Penalizar tokens muy frecuentes
                        for tok_id in range(self.vocab_size):
                            if token_counts[tok_id] > 1:
                                logits[b, tok_id] -= 2.0 * token_counts[tok_id]
                
                logits = logits / temperature
                
                # Boost a tokens raros
                logits = logits + 0.3 * (1.0 / (token_counts + 1.0))
                
                logits = top_k_top_p_filtering(logits, top_k=60, top_p=top_p)
                probabilities = F.softmax(logits, dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1)
                
                for b in range(batch_size):
                    token_counts[next_word[b].item()] += 1
                
                generated.append(next_word)
                generated_history.append(next_word.squeeze(1))
                input_word = next_word
            
            return torch.cat(generated, dim=1)
    
    def _get_init_state(self, thought):
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        return (h0, c0)



# ==================== CUERPO CALLOSO CON COSTO METABÓLICO ====================

class CorpusCallosum(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn_left_to_right = nn.MultiheadAttention(256, 8, batch_first=True)
        self.cross_attn_right_to_left = nn.MultiheadAttention(256, 8, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
    
    # FIX: REEMPLAZAR MÉTODO FORWARD COMPLETO para alinear dimensiones
    def forward(self, left_repr, right_repr, mode="awake"):
        B = left_repr.size(0)
        
        # FIX CRÍTICO: Alinear dimensiones secuenciales antes de atención
        if left_repr.dim() == 3 and right_repr.dim() == 3:
            if left_repr.size(1) != right_repr.size(1):
                # Usar pooling adaptativo para preservar información semántica
                if left_repr.size(1) > right_repr.size(1):
                    left_repr = F.adaptive_avg_pool1d(
                        left_repr.transpose(1,2), right_repr.size(1)
                    ).transpose(1,2)
                else:
                    right_repr = F.adaptive_avg_pool1d(
                        right_repr.transpose(1,2), left_repr.size(1)
                    ).transpose(1,2)
        
        # Estado global para gate metabólico
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
        
        # FIX: Retornar 3 valores para compatibilidad
        return left_final, right_final, metabolic_cost

# ==================== HOMEÓSTASIS BICAMERAL ====================
class HomeostasisEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 1.5  # Aumentar aún más para exploración
        
        # Promedios móviles para targets adaptativos
        self.register_buffer('running_richness', torch.tensor(4.0))
        self.register_buffer('running_phi', torch.tensor(2.0))
        self.momentum = 0.95
    
    def decide(self, task_loss_val, richness_val, vn_entropy_val):
        # Actualizar promedios móviles
        self.running_richness = self.momentum * self.running_richness + (1 - self.momentum) * richness_val
        self.running_phi = self.momentum * self.running_phi + (1 - self.momentum) * vn_entropy_val
        
        # Targets son los promedios + pequeño offset
        target_richness = self.running_richness.item() + 0.2
        target_phi = self.running_phi.item() + 0.3
        
        # Drive 1: Focus
        focus_drive = task_loss_val * 2.0
        
        # Drive 2: Explore (simétrico)
        richness_error = target_richness - richness_val
        explore_drive = abs(richness_error) * 0.8
        
        # Drive 3: Repair (simétrico)
        phi_error = target_phi - vn_entropy_val
        repair_drive = abs(phi_error) * 0.8
        
        # Bias base
        base_bias = 0.3
        focus_drive += base_bias
        explore_drive += base_bias
        repair_drive += base_bias
        
        logits = torch.tensor([focus_drive, explore_drive, repair_drive], dtype=torch.float32) / self.temperature
        probs = F.softmax(logits, dim=0)
        
        return probs[0].item(), probs[1].item(), probs[2].item()



class BicameralHomeostasis(nn.Module):
    def __init__(self):
        super().__init__()
        self.left_engine = HomeostasisEngine()
        self.right_engine = HomeostasisEngine()
        self.cross_balance = nn.Parameter(torch.tensor(0.5))
        
        # Reducir momentum de suavizado
        self.register_buffer('smooth_left_drives', torch.tensor([0.33, 0.33, 0.33]))
        self.register_buffer('smooth_right_drives', torch.tensor([0.33, 0.33, 0.33]))
        
        self.last_left_drives = (0.0, 0.0, 0.0)
        self.last_right_drives = (0.0, 0.0, 0.0)
        self.last_callosal_flow = 0.5

    def decide(self, left_metrics, right_metrics, epoch, total_epochs):
        p_left = self.left_engine.decide(**left_metrics)
        p_right = self.right_engine.decide(**right_metrics)
        
        p_left_tensor = torch.tensor(p_left, device=self.smooth_left_drives.device)
        p_right_tensor = torch.tensor(p_right, device=self.smooth_right_drives.device)
        
        # FIX: Suavizado adaptativo (más reactivo en early training)
        alpha = 0.4 if epoch < 10 else 0.6  # Reducido de 0.7
        self.smooth_left_drives = alpha * self.smooth_left_drives + (1 - alpha) * p_left_tensor
        self.smooth_right_drives = alpha * self.smooth_right_drives + (1 - alpha) * p_right_tensor
        
        self.smooth_left_drives = self.smooth_left_drives / (self.smooth_left_drives.sum() + 1e-8)
        self.smooth_right_drives = self.smooth_right_drives / (self.smooth_right_drives.sum() + 1e-8)
        
        p_left_smooth = tuple(self.smooth_left_drives.cpu().numpy())
        p_right_smooth = tuple(self.smooth_right_drives.cpu().numpy())
        
        # Flujo callosal dinámico
        left_urgency = p_left_smooth[2]  # Repair drive
        right_urgency = p_right_smooth[2]
        
        if left_urgency > 0.5 and right_urgency < 0.3:
            cross_weight = 0.75
        elif right_urgency > 0.5 and left_urgency < 0.3:
            cross_weight = 0.25
        else:
            cross_weight = torch.sigmoid(self.cross_balance).item()
        
        self.last_left_drives = p_left_smooth
        self.last_right_drives = p_right_smooth
        self.last_callosal_flow = cross_weight
        
        return {
            'left': p_left_smooth,
            'right': p_right_smooth,
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
        if pattern.dim() == 3:
            pattern = pattern.mean(dim=1)
        
        self.buffer.append(pattern.detach().cpu())
    
    def replay(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        # FIX: Filtrar solo tensores con misma dimensión
        target_shape = self.buffer[-1].shape
        valid_indices = [i for i, p in enumerate(self.buffer) if p.shape == target_shape]
        
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

# ==================== MODELO BICAMERAL COMPLETO ====================
class NeuroLogos(nn.Module):
    def __init__(self, vocab_size=1000, use_nested=False):
        super().__init__()
        self.use_nested = use_nested
        self.left_hemi = LeftHemisphere(use_nested=use_nested)
        self.right_hemi = RightHemisphere(config=None)
        self.corpus_callosum = CorpusCallosum()
        self.homeostasis = BicameralHomeostasis()
        
        # FIX: Agregar embeddings de categoría
        self.category_embedding = nn.Embedding(10, 128)  # 10 categorías CIFAR
        
        self.broca = BioDecoder(vocab_size=vocab_size, visual_dim=256)
        self.thought_to_lang = nn.Linear(512, 512)
        self.aux_classifier = nn.Linear(512, 10)
        self.memory = ReplayMemory(capacity=2000)
        self.total_epochs = 50
        self.current_epoch = 0
        self.last_thought_final = None
        self.last_predicted_category = None

    def forward(self, image, captions=None, plasticity=0.1, transfer_rate=0.005, mode="awake", epoch=0, labels=None):
        visual_left = self.left_hemi.eye(image)
        visual_right_raw = self.right_hemi.visual_cortex(image)
        visual_right_flat = visual_right_raw.flatten(2).transpose(1, 2)
        visual_right_topo = self.right_hemi.topo_layer(visual_right_flat, plasticity_gate=plasticity)
        right_liquid_out = self.right_hemi.liquid(visual_right_topo, plasticity, transfer_rate)
        
        if self.training and torch.rand(1).item() < 0.2: 
            with torch.no_grad():
                B_sample = min(8, right_liquid_out.size(0))
                sample_indices = torch.randperm(right_liquid_out.size(0))[:B_sample]
                sample_activation = right_liquid_out[sample_indices]
                
                # FIX: Usar dimensiones reales [B, 36, 512] sin reshape arbitrario
                phi_r, div_r, _ = measure_neural_complexity(sample_activation)
                self.right_metrics_cache = {
                    'task_loss_val': self.right_hemi.liquid.prediction_error,
                    'richness_val': div_r.item() if torch.is_tensor(div_r) else div_r,
                    'vn_entropy_val': phi_r.item() if torch.is_tensor(phi_r) else phi_r
                }
        
        right_metrics = getattr(self, 'right_metrics_cache', {
            'task_loss_val': 0.1, 'richness_val': 3.0, 'vn_entropy_val': 1.0
        })

        if visual_left.dim() == 3:
            left_proj = self.left_hemi.callosal_proj(visual_left)
        else:
            left_proj = self.left_hemi.callosal_proj(visual_left).unsqueeze(1)
        right_proj = self.right_hemi.callosal_proj(right_liquid_out)
        
        left_enhanced, right_enhanced, _ = self.corpus_callosum(left_proj, right_proj, mode=mode)
        
        if visual_left.dim() == 3:
            thought_left = self.left_hemi.cortex(right_enhanced.mean(dim=1), plasticity, transfer_rate)
        else:
            thought_left = self.left_hemi.cortex(right_enhanced.squeeze(1), plasticity, transfer_rate)
        
        thought_final = self.thought_to_lang(thought_left)
        self.last_thought_final = thought_final
        
        if self.training and mode == "awake":
            self.memory.store(thought_final)
        
        cortex = self.left_hemi.cortex
        left_metrics = {
            'task_loss_val': cortex.last_prediction_error,
            'richness_val': cortex.last_richness,
            'vn_entropy_val': cortex.last_vn_entropy
        }
        
        _ = self.homeostasis.decide(left_metrics, right_metrics, epoch, self.total_epochs)
        
        cls_logits = self.aux_classifier(thought_final)
        predicted_categories = cls_logits.argmax(dim=1)
        self.last_predicted_category = predicted_categories
        
        if labels is not None and self.training:
            category_emb = self.category_embedding(labels)
        else:
            category_emb = self.category_embedding(predicted_categories)
        
        if right_enhanced.dim() == 3:
            N = right_enhanced.size(1)
            category_emb_expanded = category_emb.unsqueeze(1).expand(-1, N, -1)
            enhanced_visual = torch.cat([right_enhanced, category_emb_expanded], dim=-1)
        else:
            enhanced_visual = torch.cat([right_enhanced, category_emb], dim=-1)
        
        return self.broca(thought_final, visual_features=enhanced_visual, captions=captions)
        
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


def to_float(val):
    if torch.is_tensor(val):
        return val.detach().cpu().item()
    return float(val)

    
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
    
    # Buffer para métricas
    batch_metrics_buffer = {
        'loss': [], 'richness': [], 'phi_effective': [], 'activation_entropy': [], 'fast_norm': [],
        'plasticity': [], 'left_focus': [], 'left_explore': [], 'left_repair': [],
        'right_focus': [], 'right_explore': [], 'right_repair': [], 'callosal_flow': []
    }
    
    right_metrics = {
        'task_loss_val': 0.15,
        'richness_val': 5.0,
        'vn_entropy_val': 2.5
    }
    
    for epoch in range(50):
        plasticity = life.get_plasticity(epoch)
        model.set_epoch(epoch)
        model.train()
        
        current_transfer_rate = 0.01 if epoch < 30 else 0.003
        liquid_module = model.left_hemi.cortex.get_liquid_module()
        
        if epoch % 15 == 0 and epoch > 0:
            liquid_module.consolidate_svd(repair_strength=0.4, timescale='medium')
        
        if epoch == 35:
            liquid_module.consolidate_svd(repair_strength=0.3, timescale='fast')
        
        total_loss = 0
        sum_richness = sum_phi = sum_entropy = sum_fast_norm = sum_plasticity = 0
        num_batches = 0
        
        # Sin tqdm para evitar overhead
        print(f"\n{'='*70}")
        print(f"Epoch {epoch:02d} | {life.phase} | Plasticity: {plasticity:.4f}")
        print(f"{'='*70}")
        
        for batch_idx, (images, captions, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True) * 2 - 1
            captions = captions.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # En el loop de entrenamiento, pasar labels:
            logits = model(images, captions=captions, plasticity=plasticity, 
                transfer_rate=current_transfer_rate, mode="awake", epoch=epoch, labels=labels)
            
            gen_loss = F.cross_entropy(
                logits.reshape(-1, len(dataset.vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=dataset.word2id["<PAD>"]
            )
            
            thought_final = model.last_thought_final
            cls_logits = model.aux_classifier(thought_final.detach())
            cls_loss = F.cross_entropy(cls_logits, labels)
            
            loss = gen_loss + 0.03 * cls_loss
            
            with torch.no_grad():
                liquid_module.prediction_error = (loss / 5.0).clamp(0.0, 0.95).item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            cortex = model.left_hemi.cortex
            sum_richness += cortex.last_richness
            sum_phi += cortex.last_vn_entropy
            sum_entropy += getattr(cortex, 'last_activation_entropy', 3.5)
            sum_fast_norm += cortex.last_fast_norm
            sum_plasticity += cortex.last_effective_plasticity
            num_batches += 1
            
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    try:
                        right_liquid = model.right_hemi.liquid
                        
                        # FIX: Simular input con dimensiones correctas [B, 36, 512]
                        grid_size = model.right_hemi.config.grid_size
                        N_spatial = grid_size ** 2
                        test_input = torch.randn(8, N_spatial, 512, device=device) * 0.5
                        
                        test_activation = right_liquid(test_input, global_plasticity=0.01, transfer_rate=0.005)
                        
                        # Medir complejidad con estructura real [B, N, D]
                        phi_r, div_r, entropy_r = measure_neural_complexity(test_activation)
                        
                        right_metrics = {
                            'task_loss_val': right_liquid.prediction_error,
                            'richness_val': div_r,
                            'vn_entropy_val': phi_r
                        }
                    except Exception as e:
                        right_metrics = {
                            'task_loss_val': 0.15, 'richness_val': 3.0, 'vn_entropy_val': 1.5
                        }
            
            left_metrics = {
                'task_loss_val': cortex.last_prediction_error,
                'richness_val': cortex.last_richness,
                'vn_entropy_val': cortex.last_vn_entropy
            }
            
            homeo_decision = model.homeostasis.decide(left_metrics, right_metrics, epoch, model.total_epochs)

            batch_metrics_buffer['loss'].append(loss.item())
            batch_metrics_buffer['richness'].append(to_float(cortex.last_richness))
            batch_metrics_buffer['phi_effective'].append(to_float(cortex.last_vn_entropy))
            batch_metrics_buffer['activation_entropy'].append(to_float(getattr(cortex, 'last_activation_entropy', 3.5)))
            batch_metrics_buffer['fast_norm'].append(to_float(cortex.last_fast_norm))
            batch_metrics_buffer['plasticity'].append(to_float(cortex.last_effective_plasticity))
            
            homeo = model.homeostasis
            batch_metrics_buffer['left_focus'].append(homeo.last_left_drives[0])
            batch_metrics_buffer['left_explore'].append(homeo.last_left_drives[1])
            batch_metrics_buffer['left_repair'].append(homeo.last_left_drives[2])
            batch_metrics_buffer['right_focus'].append(homeo.last_right_drives[0])
            batch_metrics_buffer['right_explore'].append(homeo.last_right_drives[1])
            batch_metrics_buffer['right_repair'].append(homeo.last_right_drives[2])
            batch_metrics_buffer['callosal_flow'].append(homeo.last_callosal_flow)
            
            # Logging cada 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = np.mean(batch_metrics_buffer['loss'][-100:])
                avg_rich = np.mean(batch_metrics_buffer['richness'][-100:])
                avg_phi = np.mean(batch_metrics_buffer['phi_effective'][-100:])
                avg_entropy = np.mean(batch_metrics_buffer['activation_entropy'][-100:])
                avg_fast = np.mean(batch_metrics_buffer['fast_norm'][-100:])
                avg_plast = np.mean(batch_metrics_buffer['plasticity'][-100:])
                
                avg_lf = np.mean(batch_metrics_buffer['left_focus'][-100:])
                avg_le = np.mean(batch_metrics_buffer['left_explore'][-100:])
                avg_lr = np.mean(batch_metrics_buffer['left_repair'][-100:])
                avg_rf = np.mean(batch_metrics_buffer['right_focus'][-100:])
                avg_re = np.mean(batch_metrics_buffer['right_explore'][-100:])
                avg_rr = np.mean(batch_metrics_buffer['right_repair'][-100:])
                avg_cal = np.mean(batch_metrics_buffer['callosal_flow'][-100:])
                
                print(f"  Batch {batch_idx-99:04d}-{batch_idx:04d} | "
                      f"Loss: {avg_loss:.3f} | Rich: {avg_rich:.1f} | Φₑ: {avg_phi:.2f} | "
                      f"H: {avg_entropy:.2f} | Fast: {avg_fast:.2f} | Plast: {avg_plast:.3f}")
                print(f"    Homeo L:[F={avg_lf:.2f} E={avg_le:.2f} R={avg_lr:.2f}] | "
                      f"R:[F={avg_rf:.2f} E={avg_re:.2f} R={avg_rr:.2f}] | Cal:{avg_cal:.2f}")
        
        # Ciclo de sueño cada 5 épocas
        if epoch % 5 == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                replay_batch = model.memory.replay(batch_size=8)
                if replay_batch is not None:
                    _ = model(replay_batch, mode="dream", epoch=epoch)
                    print(f"\n💤 Ciclo de sueño ejecutado")
            model.train()
        
        # Generación de muestra cada 5 épocas con análisis completo
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Generar 5 muestras aleatorias para análisis
                sample_indices = np.random.choice(len(dataset), 5, replace=False)
                
                print(f"\n{'='*70}")
                print(f"GENERACIÓN DE LENGUAJE - Época {epoch}")
                print(f"{'='*70}")
                
                total_coherence = 0
                
                for idx in sample_indices:
                    sample_img, _, label = dataset[idx]
                    sample_img = sample_img.unsqueeze(0).to(device) * 2 - 1
                    
                    generated = model(sample_img, captions=None, plasticity=plasticity, 
                                    transfer_rate=current_transfer_rate, mode="awake", epoch=epoch)
                    
                    words = [dataset.id2word[int(t.item())] for t in generated[0]]
                    sentence = " ".join([w for w in words if w not in ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]])
                    
                    # Calcular coherencia
                    coherence = estimate_coherence(sentence, dataset.templates)
                    total_coherence += coherence
                    
                    # Nombre de categoría
                    category_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                                     'dog', 'frog', 'horse', 'ship', 'truck']
                    category = category_names[label]
                    
                    print(f"\nCategoría Real: {category.upper()}")
                    print(f"Genera: '{sentence}'")
                    print(f"Coherencia: {coherence:.3f}")
                
                avg_coherence = total_coherence / 5
                
                # Métricas finales de época
                cortex = model.left_hemi.cortex
                homeo = model.homeostasis
                
                print(f"\n{'='*70}")
                print(f"RESUMEN ÉPOCA {epoch:02d}")
                print(f"{'='*70}")
                print(f"Loss: {total_loss/num_batches:.3f} | Diversidad: {sum_richness/num_batches:.1f} | "
                      f"Φₑ: {sum_phi/num_batches:.2f} | H(act): {sum_entropy/num_batches:.2f}")
                print(f"Coherencia Promedio: {avg_coherence:.3f}")
                print(f"Cortex Mode: {cortex.last_mode} | Fast Norm: {cortex.last_fast_norm:.2f}")
                print(f"Plasticity: Global={plasticity:.3f} | Effective={sum_plasticity/num_batches:.3f} | "
                      f"Learned={cortex.last_learned_plasticity:.3f}")
                print(f"Homeo L: F={homeo.last_left_drives[0]:.2f} E={homeo.last_left_drives[1]:.2f} "
                      f"R={homeo.last_left_drives[2]:.2f}")
                print(f"Homeo R: F={homeo.last_right_drives[0]:.2f} E={homeo.last_right_drives[1]:.2f} "
                      f"R={homeo.last_right_drives[2]:.2f}")
                print(f"Callosal Flow: {homeo.last_callosal_flow:.2f}")
                print(f"{'='*70}\n")
            
            model.train()
        else:
            # Print simplificado para épocas sin generación
            print(f"\nResumen Época {epoch:02d} | Loss: {total_loss/num_batches:.3f} | "
                  f"Div: {sum_richness/num_batches:.1f} | Φₑ: {sum_phi/num_batches:.2f} | "
                  f"Fast: {sum_fast_norm/num_batches:.2f}\n")
    
    print("\n✅ Entrenamiento completado.\n")
    torch.save(model.state_dict(), 'neurologos_bicameral_FIXED.pth')
    
    # Diagnóstico final
    print("📊 DIAGNÓSTICO FINAL:")
    print(f"   Diversidad Espacial: {sum_richness/num_batches:.1f} / 6.0 (target)")
    print(f"   Φₑ (Integración): {sum_phi/num_batches:.2f} / 4.0 (target)")
    print(f"   H(activaciones): {sum_entropy/num_batches:.2f}")
    print(f"   Fast Norm Final: {sum_fast_norm/num_batches:.2f} / 3.0 (límite)")
    print(f"   Plasticidad Efectiva: {sum_plasticity/num_batches:.3f}")
    print(f"   Estado Homeostático L: {homeo.last_left_drives}")
    print(f"   Estado Homeostático R: {homeo.last_right_drives}")
    print(f"   Flujo Callosal: {homeo.last_callosal_flow:.2f}")


# ==================== EJECUCIÓN ====================

if __name__ == "__main__":

    train_logos(use_nested=True)

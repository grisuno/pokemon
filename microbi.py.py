# =============================================================================
# NeuroLogos Bicameral v3.0 - CPU OPTIMIZED ("Microscopic Brain")
# Misma arquitectura, 5-10x menos parámetros, eliminado CUDA
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from collections import Counter
import torchvision.models as models
import warnings
from tqdm import tqdm
import math
import gc

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES GLOBALES - ULTRA REDUCIDAS PARA CPU
# =============================================================================
IMG_SIZE = 112  # 224 -> 112 (reduce parámetros CNN por 4x)
MAX_CAPTION_LEN = 25  # 30 -> 25 (captions cortos son más comunes)
VOCAB_SIZE = 3000  # 5000 -> 3000 (vocabulario más compacto)
EMBED_DIM = 64  # 192 -> 64 (3x reducción)
HIDDEN_DIM = 128  # 384 -> 128 (3x reducción)
BATCH_SIZE = 4  # 32 -> 4 (CPU prefiere batches pequeños)
NUM_WORKERS = 0  # 2 -> 0 (threads en CPU son lentos)
GRADIENT_ACCUMULATION_STEPS = 8  # 2 -> 8 (efectivo batch=32)

# =============================================================================
# CURIOSITY MODULE - VERSIÓN TINY
# =============================================================================
class EpistemicCuriosityCPU(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=32):  # Reducido
        super().__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + VOCAB_SIZE, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, VOCAB_SIZE)
        )
        
        self.register_buffer('novelty_history', torch.zeros(100))  # 500 -> 100
        self.register_buffer('history_idx', torch.tensor(0))
        
    def compute_intrinsic_reward(self, state, action, next_state):
        action_onehot = F.one_hot(action, VOCAB_SIZE).float()
        predicted_next = self.forward_model(torch.cat([state, action_onehot], dim=-1))
        prediction_error = F.mse_loss(predicted_next, next_state, reduction='none').mean(dim=-1)
        
        with torch.no_grad():
            idx = self.history_idx.item() % 100
            self.novelty_history[idx] = prediction_error.mean().item()
            self.history_idx += 1
            
            mean_novelty = self.novelty_history.mean()
            std_novelty = self.novelty_history.std().clamp(min=1e-4)
            normalized_reward = (prediction_error - mean_novelty) / std_novelty
        
        return prediction_error, normalized_reward
    
    def update(self, state, action, next_state):
        if state.size(0) > 16:  # Sample más pequeño
            indices = torch.randperm(state.size(0))[:16]
            state = state[indices]
            action = action[indices]
            next_state = next_state[indices]
        
        action_onehot = F.one_hot(action, VOCAB_SIZE).float()
        pred_next = self.forward_model(torch.cat([state, action_onehot], dim=-1))
        forward_loss = F.mse_loss(pred_next, next_state.detach())
        
        pred_action = self.inverse_model(torch.cat([state, next_state], dim=-1))
        inverse_loss = F.cross_entropy(pred_action, action)
        
        return forward_loss + inverse_loss


# =============================================================================
# LIQUID NEURON v3.0 - ULTRA LIGHT
# =============================================================================
class LiquidNeuronCPU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=0.5)  # FIX: ganancia reducida
        
        self.register_buffer('W_medium', torch.zeros(out_dim, in_dim))
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        
        self.ln = nn.LayerNorm(out_dim)
        
        self.plasticity_controller = nn.Sequential(
            nn.Linear(3, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.plasticity_controller[-2].bias.data.fill_(-1.5)
        
        self.base_lr = 0.005  # FIX: más lento
        self.prediction_error = 0.0
        
        self.register_buffer('activation_mean_ema', torch.zeros(1))
        self.register_buffer('activation_std_ema', torch.ones(1))
        self.ema_momentum = 0.95
        
    def forward(self, x, global_plasticity=0.0, transfer_rate=0.005):
        # FIX: Clamp input
        x = torch.clamp(x, -5, 5)
        
        slow_out = self.W_slow(x)
        slow_out = torch.clamp(slow_out, -10, 10)  # FIX: clamp salida
        
        medium_out = F.linear(x, self.W_medium)
        fast_out = F.linear(x, self.W_fast)
        
        pre_act = slow_out + medium_out + fast_out
        pre_act = torch.clamp(pre_act, -10, 10)
        
        batch_mean = pre_act.mean(dim=1, keepdim=True)
        batch_std = pre_act.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
        
        with torch.no_grad():
            self.activation_mean_ema = self.ema_momentum * self.activation_mean_ema + \
                                      (1 - self.ema_momentum) * batch_mean.mean()
            self.activation_std_ema = self.ema_momentum * self.activation_std_ema + \
                                     (1 - self.ema_momentum) * batch_std.mean()
            self.activation_std_ema = self.activation_std_ema.clamp(0.1, 1.5)
            self.activation_mean_ema = self.activation_mean_ema.clamp(-0.5, 0.5)
        
        homeostasis_signal = (self.activation_std_ema / (self.activation_mean_ema.abs() + 1e-6)).unsqueeze(0)
        stats = torch.cat([batch_mean, batch_std, homeostasis_signal.expand(batch_mean.size(0), -1)], dim=1)
        
        learned_plasticity = self.plasticity_controller(stats).squeeze(1)
        effective_plasticity = global_plasticity * learned_plasticity * (1.0 - self.prediction_error * 0.5)
        out = 5.0 * torch.tanh(self.ln(pre_act) / 5.0)
        
        if self.training and effective_plasticity.mean() > 0.001:
            with torch.no_grad():
                out_centered = out - out.mean(dim=0, keepdim=True)
                correlation = torch.mm(out_centered.T, x) / (x.size(0) + 1e-6)
                
                self.W_medium.data += self.W_fast.data * transfer_rate
                
                lr_vector = effective_plasticity.mean() * self.base_lr
                self.W_fast.data += correlation * lr_vector
                
                self.W_fast.data.mul_(1.0 - transfer_rate * 0.5)
                
                current_norm = self.W_fast.norm()
                homeostasis_factor = homeostasis_signal.mean().item()
                target_norm = 2.0 + 0.5 * homeostasis_factor
                if current_norm > target_norm:
                    self.W_fast.data *= (target_norm / current_norm).clamp(0.8, 1.0)
        
        return out

    def consolidate_svd(self, repair_strength=1.0, timescale='fast'):
        if timescale == 'fast':
            W_target = self.W_fast
        elif timescale == 'medium':
            W_target = self.W_medium
        else:
            return False
            
        with torch.no_grad():
            try:
                U, S, Vt = torch.linalg.svd(W_target, full_matrices=False)
                threshold = S.max() * 0.005 * repair_strength  # Más agresivo
                mask = S > threshold
                filtered_S = S * mask.float()
                W_consolidated = U @ torch.diag(filtered_S) @ Vt
                
                W_target.data = (1.0 - repair_strength) * W_target.data + repair_strength * W_consolidated
                W_target.data *= 0.99
                return True
            except:
                W_target.data.mul_(0.95)
                return False


# =============================================================================
# BICAMERAL ATTENTION - VERSIÓN TINY
# =============================================================================
class BicameralAttentionCPU(nn.Module):
    def __init__(self, dim=128, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        
        # FIX: Inicializar scales con valores más pequeños y restringir su rango
        self.local_scale = nn.Parameter(torch.tensor(0.5))  # Inicialización reducida
        self.global_scale = nn.Parameter(torch.tensor(0.5))
        
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        local_heads = self.num_heads // 2
        Q_local, Q_global = Q[:, :local_heads], Q[:, local_heads:]
        K_local, K_global = K[:, :local_heads], K[:, local_heads:]
        V_local, V_global = V[:, :local_heads], V[:, local_heads:]
        
        # FIX: Clamp scales para evitar explosión
        local_scale = torch.sigmoid(self.local_scale) * 2.0  # Restringir a [0, 2]
        global_scale = torch.sigmoid(self.global_scale) * 2.0
        
        # Local attention con máscara causal
        local_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn_local = torch.matmul(Q_local, K_local.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_local = attn_local.masked_fill(local_mask, -1e4)  # FIX: valor más moderado que -inf
        attn_local = F.softmax(attn_local * local_scale, dim=-1)
        attn_local = self.dropout(attn_local)
        out_local = torch.matmul(attn_local, V_local)
        
        # Global attention
        attn_global = torch.matmul(Q_global, K_global.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_global = F.softmax(attn_global * global_scale, dim=-1)
        attn_global = self.dropout(attn_global)
        out_global = torch.matmul(attn_global, V_global)
        
        out = torch.cat([out_local, out_global], dim=1)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out, attn_local, attn_global

# =============================================================================
# RIGHT HEMISPHERE v3.0 - CNN LIGERÍSIMA
# =============================================================================
class RightHemisphereCPU(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        
        # Congelar TODO excepto última capa
        for param in resnet.parameters():
            param.requires_grad = False
        
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # FIX: Normalización de batch para estabilidad
        self.feature_norm = nn.BatchNorm1d(512)
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_liquid = LiquidNeuronCPU(512, output_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(512, output_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.output_dim = output_dim
        
    def forward(self, image, plasticity=0.1, transfer_rate=0.005):
        features_map = self.visual_encoder[:-1](image)
        
        # FIX: Clamp features para evitar valores extremos
        features_map = torch.clamp(features_map, -10, 10)
        
        attn_map = self.spatial_attention(features_map)
        features_map = features_map * attn_map
        
        features = self.global_pool(features_map).flatten(1)
        features = self.feature_norm(features)  # Normalización
        
        liquid_features = self.spatial_liquid(features, plasticity, transfer_rate)
        global_features = self.global_fc(features)
        
        # FIX: Clamp final output
        visual_thought = self.fusion(torch.cat([liquid_features, global_features], dim=1))
        return torch.clamp(visual_thought, -5, 5)


# =============================================================================
# LEFT HEMISPHERE v3.0 - LANGUAGE TINY
# =============================================================================
class LeftHemisphereCPU(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim,
            hidden_dim, 
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        self.attention = BicameralAttentionCPU(hidden_dim, num_heads=2)
        
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.residual_gate = nn.Parameter(torch.tensor(0.5))
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        
        self.curiosity = EpistemicCuriosityCPU(hidden_dim, hidden_dim // 4)
        
    def forward(self, visual_context, captions=None, max_len=25, 
                return_diagnostics=False, temperature=0.9, 
                exploration_bonus=0.0):
        batch_size = visual_context.size(0)
        device = visual_context.device
        
        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            seq_len = embeddings.size(1)
            
            visual_expanded = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_input = torch.cat([embeddings, visual_expanded], dim=2)
            
            lstm_out, _ = self.lstm(lstm_input, self._get_init_state(visual_context))
            attn_out, local_attn, global_attn = self.attention(lstm_out)
            
            alpha = torch.sigmoid(self.residual_gate)
            combined = alpha * attn_out + (1 - alpha) * lstm_out
            
            gate = self.liquid_gate(combined)
            modulated = combined * gate
            
            logits = self.output_projection(modulated)
            
            if exploration_bonus > 0:
                sample_size = min(lstm_out.size(0) * (lstm_out.size(1) - 1), 256)
                indices = torch.randperm(lstm_out.size(0) * (lstm_out.size(1) - 1))[:sample_size]
                
                states = lstm_out[:, :-1].reshape(-1, self.hidden_dim)[indices]
                actions = captions[:, 1:-1].reshape(-1)[indices]
                next_states = lstm_out[:, 1:].reshape(-1, self.hidden_dim)[indices]
                
                curiosity_loss = self.curiosity.update(states, actions, next_states)
            else:
                curiosity_loss = torch.tensor(0.0, device=device)
            
            if return_diagnostics:
                return logits, gate, local_attn, global_attn, curiosity_loss
            return logits, curiosity_loss
        
        else:
            # Generation mode
            generated = []
            hidden = self._get_init_state(visual_context)
            input_token = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
            
            top_p = max(0.90, 0.95 - exploration_bonus * 0.2)
            
            for step in range(max_len):
                emb = self.embedding(input_token)
                visual_expanded = visual_context.unsqueeze(1)
                lstm_input = torch.cat([emb, visual_expanded], dim=2)
                
                out, hidden = self.lstm(lstm_input, hidden)
                attn_out, _, _ = self.attention(out)
                alpha = torch.sigmoid(self.residual_gate)
                combined = alpha * attn_out + (1 - alpha) * out
                
                gate = self.liquid_gate(combined)
                out_final = combined * gate
                
                logits = self.output_projection(out_final.squeeze(1))
                logits = logits / temperature
                logits = self._top_p_filtering(logits, top_p)
                
                # FIX: Clamp logits para evitar NaN/inf en CPU
                probs = F.softmax(logits.clamp(min=-50, max=50), dim=-1)
                probs = probs.clamp(min=1e-8)  # Evitar probabilidades cero
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated.append(next_token)
                input_token = next_token
                
                if (next_token == 2).all():
                    break
            
            return torch.cat(generated, dim=1)
    
    def _get_init_state(self, visual_context):
        h0 = visual_context.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        return (h0, c0)
    
    def _top_p_filtering(self, logits, top_p=0.9):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        return logits

# =============================================================================
# CORPUS CALLOSUM v3.0 - SIMPLIFICADO
# =============================================================================
class CorpusCallosumCPU(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.right_to_left = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # FIX: Inicialización más estable para la gate
        self.mixing_gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        # Inicialización del bias para gate balanceada
        self.mixing_gate[-2].bias.data.fill_(0.0)
        
    def forward(self, right_features):
        # FIX: Normalizar features de entrada
        right_features = F.normalize(right_features, dim=-1) * 2.0
        
        transformed = self.right_to_left(right_features)
        combined = torch.cat([right_features, transformed], dim=-1)
        alpha = self.mixing_gate(combined)
        
        # FIX: Suavizar la mezcla
        output = alpha * transformed + (1 - alpha) * right_features
        return output, alpha



# =============================================================================
# BICAMERAL MODEL v3.0 - CPU EDITION
# =============================================================================
class NeuroLogosBicameralCPU(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.right_hemisphere = RightHemisphereCPU(output_dim=128)
        self.left_hemisphere = LeftHemisphereCPU(vocab_size, embed_dim=64, hidden_dim=128)
        self.corpus_callosum = CorpusCallosumCPU(dim=128)
        
    def forward(self, image, captions=None, plasticity=0.1, transfer_rate=0.005, 
                return_diagnostics=False, temperature=0.9, exploration_bonus=0.0):
        visual_features = self.right_hemisphere(image, plasticity, transfer_rate)
        visual_context, callosum_gate = self.corpus_callosum(visual_features)
        
        if captions is not None:
            if return_diagnostics:
                output, gate, local_attn, global_attn, curiosity_loss = self.left_hemisphere(
                    visual_context, captions, return_diagnostics=True, 
                    temperature=temperature, exploration_bonus=exploration_bonus
                )
                return output, visual_features, visual_context, gate, local_attn, global_attn, callosum_gate, curiosity_loss
            else:
                output, curiosity_loss = self.left_hemisphere(
                    visual_context, captions, temperature=temperature, 
                    exploration_bonus=exploration_bonus
                )
                return output, curiosity_loss
        else:
            # Modo generación - FIX: debug para evitar NaN
            output = self.left_hemisphere(
                visual_context, captions, temperature=temperature, 
                exploration_bonus=exploration_bonus
            )
            # Si output contiene NaN, devolver tensor de <EOS> tokens
            if torch.isnan(output).any():
                output = torch.ones_like(output) * 2  # 2 = <EOS>
            return output


# =============================================================================
# DIAGNOSTICS - SIMPLIFICADO
# =============================================================================
class NeuralDiagnosticsCPU:
    def __init__(self):
        self.history = {
            'loss': [], 'curiosity_loss': [], 'right_liquid_norm': [],
            'callosal_flow': [], 'callosum_gate': [], 'left_gate_mean': [],
            'left_gate_std': [], 'vocab_diversity': [], 'vocab_entropy': [],
            'plasticity_effective': [], 'homeostasis_signal': []
        }
        self.vocab_usage = Counter()

    def measure_callosal_flow(self, right_features, left_context):
        with torch.no_grad():
            right_norm = F.normalize(right_features, dim=-1)
            left_norm = F.normalize(left_context, dim=-1)
            correlation = (right_norm * left_norm).sum(dim=-1).mean()
            return correlation.item()
    
    def measure_vocab_diversity(self, generated_tokens, vocab_size):
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        unique_tokens = len(torch.unique(generated_tokens))
        flat_tokens = generated_tokens.flatten().cpu().numpy()
        for token in flat_tokens:
            if token > 3:
                self.vocab_usage[int(token)] += 1
        
        if len(self.vocab_usage) > 0:
            counts = np.array(list(self.vocab_usage.values()))
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            entropy = 0.0
        
        return unique_tokens / vocab_size, entropy
    
    def update(self, **metrics):
        for key, value in metrics.items():
            if key in self.history and value is not None:
                self.history[key].append(value)
    
    def get_recent_avg(self, key, n=20):  # Menor ventana para CPU
        if key in self.history and len(self.history[key]) > 0:
            values = [v for v in self.history[key][-n:] if v is not None]
            if len(values) > 0:
                return np.mean(values)
        return 0.0
    
    def report(self, epoch):
        if len(self.history['loss']) == 0:
            return
        
        print(f"\n{'='*60}")
        print(f"🧠 DIAGNÓSTICO CPU - Época {epoch}")
        print(f"{'='*60}")
        
        loss = self.get_recent_avg('loss')
        curiosity = self.get_recent_avg('curiosity_loss')
        liquid = self.get_recent_avg('right_liquid_norm')
        flow = self.get_recent_avg('callosal_flow')
        callosum_gate = self.get_recent_avg('callosum_gate')
        gate_mean = self.get_recent_avg('left_gate_mean')
        gate_std = self.get_recent_avg('left_gate_std')
        vocab_div = self.get_recent_avg('vocab_diversity')
        vocab_ent = self.get_recent_avg('vocab_entropy')
        plast = self.get_recent_avg('plasticity_effective')
        homeo = self.get_recent_avg('homeostasis_signal')
        
        print(f"📉 Loss: {loss:.4f} | Curiosity: {curiosity:.4f}")
        
        h_status = "🟢 Estable" if liquid < 3.0 else "🟡 Alto" if liquid < 5.0 else "🔴 Crítico"
        print(f"👁️ LiqNorm: {liquid:.3f} | Homeo: {homeo:.3f} {h_status}")
        
        f_status = "🟢 Fluido" if flow > 0.3 else "🟡 Débil" if flow > 0.1 else "🔴 Bloqueado"
        print(f"🔗 Flow: {flow:.3f} | Gate: {callosum_gate:.3f} {f_status}")
        
        g_status = "🟢 Balanceado" if 0.2 < gate_mean < 0.8 else "🟡 Sesgado"
        print(f"💬 L-Gate: μ={gate_mean:.3f} σ={gate_std:.3f} {g_status}")
        
        v_status = "🟢 Diverso" if vocab_div > 0.1 else "🟡 Limitado" if vocab_div > 0.05 else "🔴 Colapsado"
        print(f"📚 Vocab: {vocab_div:.3f} | Entropy: {vocab_ent:.2f} {v_status}")
        print(f"   Tokens únicos: {len(self.vocab_usage)}")
        
        print(f"🧬 Plasticidad: {plast:.4f}")
        print(f"{'='*60}\n")


# =============================================================================
# CURRICULUM LEARNING - ADAPTADO
# =============================================================================
class CurriculumSchedulerCPU:
    def __init__(self, total_epochs=20):  # 30 -> 20 epochs
        self.total = total_epochs
        self.phases = {
            'exploration': (0, 5),
            'refinement': (5, 14),
            'mastery': (14, 20)
        }
    
    def get_phase(self, epoch):
        for phase_name, (start, end) in self.phases.items():
            if start <= epoch < end:
                return phase_name
        return 'mastery'
    
    def get_plasticity(self, epoch):
        phase = self.get_phase(epoch)
        if phase == 'exploration':
            return 0.25 * (1 - epoch / 5) + 0.10
        elif phase == 'refinement':
            progress = (epoch - 5) / 9
            return 0.10 * (1 - progress) + 0.03
        else:
            return 0.02
    
    def get_exploration_bonus(self, epoch):
        phase = self.get_phase(epoch)
        if phase == 'exploration':
            return 0.20 * (1 - epoch / 5) + 0.05
        elif phase == 'refinement':
            return 0.05 * (1 - (epoch - 5) / 9)
        else:
            return 0.0
    
    def get_temperature(self, epoch):
        phase = self.get_phase(epoch)
        if phase == 'exploration':
            return 1.1
        elif phase == 'refinement':
            progress = (epoch - 5) / 9
            return 1.1 - progress * 0.2
        else:
            return 0.85
    
    def should_consolidate(self, epoch):
        return epoch > 0 and epoch % 5 == 0  # Más frecuente


def build_vocab_flickr(captions_file, vocab_size=3000):
    print("📚 Construyendo vocabulario...")
    counter = Counter()
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                words = parts[1].lower().split()
                counter.update(words)
    
    most_common = counter.most_common(vocab_size - 4)
    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for i, (word, _) in enumerate(most_common):
        vocab[word] = i + 4
    
    id2word = {i: w for w, i in vocab.items()}
    print(f"✓ Vocab size: {len(vocab)}")
    return vocab, id2word


class Flickr8kDatasetCPU(Dataset):
    def __init__(self, images_dir, captions_file, vocab, transform=None, max_len=25):
        self.images_dir = images_dir
        self.transform = transform
        self.vocab = vocab
        self.max_len = max_len
        
        self.data = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, caption = parts
                    img_path = os.path.join(images_dir, img_name)
                    if os.path.exists(img_path):
                        self.data.append((img_path, caption))
        
        print(f"✓ Loaded {len(self.data)} pairs")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, caption = self.data[idx]
        
        # Robust loading for CPU
        for _ in range(3):
            try:
                image = Image.open(img_path).convert('RGB')
                break
            except Exception:
                idx = np.random.randint(0, len(self.data))
                img_path, caption = self.data[idx]
        else:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)
        
        tokens = ['<BOS>'] + caption.lower().split() + ['<EOS>']
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        
        return image, torch.tensor(token_ids, dtype=torch.long)


def setup_flickr8k_cpu(data_dir='./data_cpu'):
    """Descarga Flickr8k automáticamente (igual que la versión original)"""
    flickr_dir = os.path.join(data_dir, 'flickr8k')
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    if os.path.exists(images_dir) and os.path.exists(captions_file):
        print("✓ Flickr8k ya existe, saltando descarga...\n")
        return flickr_dir
    
    os.makedirs(flickr_dir, exist_ok=True)
    
    print("📥 Descargando Flickr8k desde GitHub...")
    print("   Tamaño: ~1GB | Tiempo estimado: 2-3 minutos\n")
    
    import urllib.request
    import zipfile
    
    urls = {
        'images': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip ',
        'captions': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip '
    }
    
    for name, url in urls.items():
        zip_path = os.path.join(flickr_dir, f'{name}.zip')
        
        print(f"📥 Descargando {name}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print(f"📂 Extrayendo {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(flickr_dir)
        
        os.remove(zip_path)
        print(f"✓ {name} completado\n")
    
    print("📝 Procesando captions...")
    raw_captions = os.path.join(flickr_dir, 'Flickr8k.token.txt')
    
    if os.path.exists(raw_captions):
        captions_dict = {}
        with open(raw_captions, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name = parts[0].split('#')[0]
                    caption = parts[1]
                    
                    if img_name not in captions_dict:
                        captions_dict[img_name] = []
                    captions_dict[img_name].append(caption)
        
        with open(captions_file, 'w', encoding='utf-8') as f:
            for img_name, caps in captions_dict.items():
                for cap in caps:
                    f.write(f"{img_name}\t{cap}\n")
        
        print(f"✓ Captions procesados: {len(captions_dict)} imágenes\n")
    
    # FIX: mover directorio si es necesario
    if os.path.exists(os.path.join(flickr_dir, 'Flicker8k_Dataset')):
        import shutil
        old_dir = os.path.join(flickr_dir, 'Flicker8k_Dataset')
        if not os.path.exists(images_dir):
            shutil.move(old_dir, images_dir)
    
    if os.path.exists(os.path.join(flickr_dir, 'Flickr8k_Dataset')):
        import shutil
        old_dir = os.path.join(flickr_dir, 'Flickr8k_Dataset')
        if not os.path.exists(images_dir):
            shutil.move(old_dir, images_dir)
    
    print("✅ Flickr8k listo\n")
    return flickr_dir

    

# =============================================================================
# TRAINING LOOP v3.0 - FIX DEFINITIVO PARA GRADIENTES NaN EN CPU
# =============================================================================
def train_bicameral_cpu():
    device = torch.device('cpu')
    print(f"\n{'='*60}")
    print(f"🧠 NeuroLogos Bicameral v3.0 CPU EDITION")
    print(f"   Device: {device} | Threads: {torch.get_num_threads()}")
    print(f"{'='*60}\n")
    
    flickr_dir = setup_flickr8k_cpu()
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    vocab, id2word = build_vocab_flickr(captions_file, VOCAB_SIZE)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Flickr8kDatasetCPU(images_dir, captions_file, vocab, transform, MAX_CAPTION_LEN)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    model = NeuroLogosBicameralCPU(len(vocab)).to(device)
    
    # FIX: Inicializar pesos liquid con menor ganancia para evitar explosión inicial
    for module in model.modules():
        if isinstance(module, LiquidNeuronCPU):
            module.W_slow.weight.data.normal_(0, 0.01)
            module.W_fast.data.zero_()
            module.W_medium.data.zero_()
    
    # Optimizador - FIX: LR mínimos para estabilidad
    visual_params = list(model.right_hemisphere.parameters())
    language_params = list(model.left_hemisphere.parameters()) + list(model.corpus_callosum.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': visual_params, 'lr': 1e-6, 'weight_decay': 1e-5},   # FIX: 1e-5 -> 1e-6
        {'params': language_params, 'lr': 3e-6, 'weight_decay': 1e-5}  # FIX: 3e-5 -> 3e-6
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2)
    curriculum = CurriculumSchedulerCPU(total_epochs=20)
    diagnostics = NeuralDiagnosticsCPU()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Parámetros totales: {total_params:,}")
    print(f"📊 Parámetros entrenables: {trainable_params:,}")
    print(f"📊 Tamaño del modelo: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"📊 Batches por época: {len(dataloader)}")
    print(f"📊 Batch efectivo: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"{'='*60}\n")
    
    os.makedirs('./checkpoints_cpu', exist_ok=True)
    start_epoch = 0
    
    checkpoint_path = './checkpoints_cpu/latest_checkpoint_cpu.pth'
    if os.path.exists(checkpoint_path):
        print("📁 Resumiendo entrenamiento...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        diagnostics.history = checkpoint['diagnostics_history']
        diagnostics.vocab_usage = checkpoint.get('vocab_usage', Counter())
        print(f"✓ Resumido desde época {start_epoch}\n")
    
    global_batch_counter = 0
    
    # FIX: Gradiente hooks para atrapar NaN en tiempo real
    def nan_hook(module, grad_input, grad_output):
        if any(torch.isnan(g).any() if g is not None else False for g in grad_input):
            print(f"🚨 NaN en gradiente input de {module.__class__.__name__}")
        if any(torch.isnan(g).any() if g is not None else False for g in grad_output):
            print(f"🚨 NaN en gradiente output de {module.__class__.__name__}")
    
    for submodule in model.modules():
        if isinstance(submodule, nn.Linear) or isinstance(submodule, nn.LSTM):
            submodule.register_backward_hook(nan_hook)
    
    for epoch in range(start_epoch, 20):  # 20 epochs para CPU
        phase = curriculum.get_phase(epoch)
        plasticity = curriculum.get_plasticity(epoch)
        exploration_bonus = curriculum.get_exploration_bonus(epoch)
        transfer_rate = 0.01 if epoch < 14 else 0.003
        temperature = curriculum.get_temperature(epoch)
        
        print(f"\n🎓 {phase.upper()} | Época {epoch}")
        print(f"   Plasticidad: {plasticity:.3f} | Exploración: {exploration_bonus:.3f}")
        
        model.train()
        total_loss = 0
        total_curiosity = 0
        num_batches = 0
        
        if curriculum.should_consolidate(epoch):
            print("🔧 Consolidando memoria...")
            model.right_hemisphere.spatial_liquid.consolidate_svd(0.7, 'medium')
            model.right_hemisphere.spatial_liquid.consolidate_svd(0.5, 'fast')
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}")
        
        for batch_idx, (images, captions) in enumerate(pbar):
            global_batch_counter += 1
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward SIN mixed precision (CPU no lo necesita)
            logits, visual_features, visual_context, gate, local_attn, global_attn, callosum_gate, curiosity_loss = model(
                images, captions, plasticity, transfer_rate,
                return_diagnostics=True, temperature=temperature,
                exploration_bonus=exploration_bonus
            )
            
            # FIX: Verificar NaN en logits inmediatamente
            if torch.isnan(logits).any():
                print(f"🚨 NaN en logits del batch {global_batch_counter}, saltando batch")
                optimizer.zero_grad()
                continue
            
            # Pérdida
            ce_loss = F.cross_entropy(
                logits.reshape(-1, len(vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=vocab['<PAD>'],
                label_smoothing=0.1 if phase == 'exploration' else 0.0
            )
            
            # FIX: Verificar NaN en cross-entropy
            if torch.isnan(ce_loss):
                print(f"🚨 NaN en ce_loss del batch {global_batch_counter}, saltando batch")
                optimizer.zero_grad()
                continue
            
            # FIX: Desactivar completamente curiosity para depurar
            curiosity_weight = 0.0  # FORZADO A CERO
            entropy_bonus = 0.0  # FORZADO A CERO
            gate_loss = 0.0  # FORZADO A CERO
            
            loss = ce_loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward SIN gradient scaling
            loss.backward()
            
            # FIX: Verificar NaN en gradientes con hook y con check manual
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"🚨 NaN en gradiente de {name} del batch {global_batch_counter}")
                    has_nan_grad = True
            
            if has_nan_grad:
                optimizer.zero_grad()
                continue
            
            # Update cada GRADIENT_ACCUMULATION_STEPS
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                with torch.no_grad():
                    # FIX: Clamp prediction_error para evitar explosión
                    model.right_hemisphere.spatial_liquid.prediction_error = \
                        (ce_loss / 10.0).clamp(0.0, 0.3).item()  # Reducido de 0.7 a 0.3
                
                # FIX: Gradient clipping ultra agresivo
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # FIX: de 0.5 a 0.1
                optimizer.step()
                optimizer.zero_grad()
            
            # Diagnostics
            with torch.no_grad():
                liquid_norm = model.right_hemisphere.spatial_liquid.W_fast.norm().item()
                callosal_flow = diagnostics.measure_callosal_flow(visual_features, visual_context)
                gate_mean = gate.mean().item()
                gate_std = gate.std().item()
                callosum_gate_val = callosum_gate.mean().item()
                
                liquid = model.right_hemisphere.spatial_liquid
                homeostasis_signal = (liquid.activation_std_ema / (liquid.activation_mean_ema.abs() + 1e-6)).item()
                
                stats = torch.cat([
                    visual_features.mean(dim=1, keepdim=True),
                    visual_features.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6),
                    liquid.activation_std_ema.expand(visual_features.size(0), 1) / (liquid.activation_mean_ema.abs().expand(visual_features.size(0), 1) + 1e-6)
                ], dim=1)
                learned_plast = liquid.plasticity_controller(stats).mean().item()
                effective_plast = plasticity * learned_plast * (1.0 - liquid.prediction_error * 0.5)
                
                # FIX: Saltar generación de muestras si hay inestabilidad
                vocab_div = vocab_ent = None
                if batch_idx % 100 == 0 and not has_nan_grad and liquid_norm < 2.0:
                    try:
                        sample_gen = model(images[:1], captions=None, temperature=temperature, exploration_bonus=exploration_bonus)
                        if isinstance(sample_gen, tuple):
                            sample_gen = sample_gen[0]
                        vocab_div, vocab_ent = diagnostics.measure_vocab_diversity(sample_gen, len(vocab))
                    except RuntimeError:
                        vocab_div = vocab_ent = 0.0
                
                # Auto-correcciones
                actions = []
                if liquid_norm > 3.0:  # FIX: threshold reducido de 4.0 a 3.0
                    model.right_hemisphere.spatial_liquid.consolidate_svd(0.9, 'fast')
                    actions.append("SVD_fast(0.9)")
                if homeostasis_signal < 0.6:
                    plasticity = min(0.5, plasticity + 0.08)
                    actions.append(f"plasticity+={plasticity:.3f}")
                if gate_mean < 0.2 or gate_mean > 0.8:
                    model.left_hemisphere.residual_gate.data += 0.02 * (0.5 - gate_mean)
                    actions.append(f"gate_adj={model.left_hemisphere.residual_gate.item():.3f}")
                
                diagnostics.update(
                    loss=ce_loss.item(),
                    curiosity_loss=curiosity_loss.item(),
                    right_liquid_norm=liquid_norm,
                    callosal_flow=callosal_flow,
                    callosum_gate=callosum_gate_val,
                    left_gate_mean=gate_mean,
                    left_gate_std=gate_std,
                    vocab_diversity=vocab_div,
                    vocab_entropy=vocab_ent,
                    plasticity_effective=effective_plast,
                    homeostasis_signal=homeostasis_signal
                )
                
                if global_batch_counter % 100 == 0:
                    print(f"\n📊 BATCH #{global_batch_counter} | Época {epoch}")
                    print(f"   Loss: {ce_loss.item():.4f} | Liq: {liquid_norm:.2f} | Flow: {callosal_flow:.3f}")
                    print(f"   Acciones: {' | '.join(actions) if actions else 'Ninguna'}")
            
            total_loss += ce_loss.item()
            total_curiosity += curiosity_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{ce_loss.item():.3f}',
                'cur': f'{curiosity_loss.item():.3f}',
                'liq': f'{liquid_norm:.2f}'
            })
        
        # Final gradient step
        if len(dataloader) % GRADIENT_ACCUMULATION_STEPS != 0:
            # FIX: Gradient clipping ultra agresivo
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # FIX: de 0.5 a 0.1
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        diagnostics.report(epoch)
        
        # Guardar checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'vocab': vocab,
            'id2word': id2word,
            'diagnostics_history': diagnostics.history,
            'vocab_usage': diagnostics.vocab_usage,
            'loss': total_loss/num_batches,
        }, checkpoint_path)
        
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'id2word': id2word,
            }, f'./checkpoints_cpu/epoch{epoch:02d}_cpu.pth')
        
        # Muestras
        if epoch % 2 == 0:
            model.eval()
            print("\n🔍 MUESTRAS...")
            with torch.no_grad():
                for i in range(2):
                    img, cap = dataset[i * 50]
                    img = img.unsqueeze(0).to(device)
                    generated = model(img, captions=None, temperature=0.9)
                    
                    if isinstance(generated, tuple):
                        generated = generated[0]
                    
                    words = [id2word.get(int(t.item()), '<UNK>') for t in generated[0]]
                    sentence = ' '.join(words[:12])
                    gt = ' '.join([id2word.get(int(t.item()), '<UNK>') for t in cap][1:13])
                    print(f"   GT: {gt}")
                    print(f"   Gen: {sentence}\n")
            model.train()
        
        print(f"✓ Época {epoch:02d} | Loss: {total_loss/num_batches:.4f} | "
              f"Curiosity: {total_curiosity/num_batches:.4f}")
    
    print("✅ Entrenamiento CPU completado!")


if __name__ == "__main__":
    train_bicameral_cpu()
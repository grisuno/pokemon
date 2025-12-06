# =============================================================================
# NeuroLogos Bicameral v2.0 - OPTIMIZADO PARA MEMORIA
# Cambios: Checkpointing, gradient accumulation, dimensiones reducidas
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from collections import Counter, deque
import torchvision.models as models
import warnings
from tqdm import tqdm
import math
import gc

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES GLOBALES - OPTIMIZADAS PARA MEMORIA
# =============================================================================
IMG_SIZE = 224
MAX_CAPTION_LEN = 30
VOCAB_SIZE = 5000
EMBED_DIM = 192  # 256 -> 192
HIDDEN_DIM = 384  # 512 -> 384
BATCH_SIZE = 32  # 64 -> 32
NUM_WORKERS = 2  # 4 -> 2
GRADIENT_ACCUMULATION_STEPS = 2  # Simula batch 64

# =============================================================================
# CURIOSITY MODULE - VERSION LIGERA
# =============================================================================
class EpistemicCuriosity(nn.Module):
    def __init__(self, feature_dim=384, hidden_dim=128):  # Reducido
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
        
        self.register_buffer('novelty_history', torch.zeros(500))  # 1000 -> 500
        self.register_buffer('history_idx', torch.tensor(0))
        
    def compute_intrinsic_reward(self, state, action, next_state):
        action_onehot = F.one_hot(action, VOCAB_SIZE).float()
        predicted_next = self.forward_model(torch.cat([state, action_onehot], dim=-1))
        
        prediction_error = F.mse_loss(predicted_next, next_state, reduction='none').mean(dim=-1)
        
        with torch.no_grad():
            idx = self.history_idx.item() % 500
            self.novelty_history[idx] = prediction_error.mean().item()
            self.history_idx += 1
            
            mean_novelty = self.novelty_history.mean()
            std_novelty = self.novelty_history.std().clamp(min=1e-4)
            normalized_reward = (prediction_error - mean_novelty) / std_novelty
        
        return prediction_error, normalized_reward
    
    def update(self, state, action, next_state):
        # Sample para reducir memoria
        if state.size(0) > 64:
            indices = torch.randperm(state.size(0))[:64]
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
# LIQUID NEURON v2.0 - OPTIMIZADO
# =============================================================================
class LiquidNeuronV2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.2)
        
        self.register_buffer('W_medium', torch.zeros(out_dim, in_dim))
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        
        self.ln = nn.LayerNorm(out_dim)
        
        # Controller más simple
        self.plasticity_controller = nn.Sequential(
            nn.Linear(3, 16),  # 32 -> 16
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.plasticity_controller[-2].bias.data.fill_(-1.5)
        
        self.base_lr = 0.02
        self.prediction_error = 0.0
        
        self.register_buffer('activation_mean_ema', torch.zeros(1))
        self.register_buffer('activation_std_ema', torch.ones(1))
        self.ema_momentum = 0.99
        
    def forward(self, x, global_plasticity=0.0, transfer_rate=0.005):
        slow_out = self.W_slow(x)
        medium_out = F.linear(x, self.W_medium)
        fast_out = F.linear(x, self.W_fast)
        pre_act = slow_out + medium_out + fast_out
        
        batch_mean = pre_act.mean(dim=1, keepdim=True)
        batch_std = pre_act.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
        
        with torch.no_grad():
            self.activation_mean_ema = self.ema_momentum * self.activation_mean_ema + \
                                      (1 - self.ema_momentum) * batch_mean.mean()
            self.activation_std_ema = self.ema_momentum * self.activation_std_ema + \
                                     (1 - self.ema_momentum) * batch_std.mean()
            # FIX: Clamp homeostasis para evitar explosión 16.802
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
                
                # FIX NORM: Target dinámico basado en homeostasis
                current_norm = self.W_fast.norm()
                homeostasis_factor = homeostasis_signal.mean().item()
                target_norm = 2.5 + 1.0 * homeostasis_factor  # 2.5-3.5 range
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
                threshold = S.max() * 0.008 * repair_strength
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
# BICAMERAL ATTENTION - VERSIÓN LIGERA
# =============================================================================
class BicameralAttention(nn.Module):
    def __init__(self, dim=384, num_heads=4):  # 8 -> 4 heads
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)  # Fused projection
        
        self.local_scale = nn.Parameter(torch.ones(1))
        self.global_scale = nn.Parameter(torch.ones(1))
        
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
        
        # Local attention
        local_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn_local = torch.matmul(Q_local, K_local.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_local = attn_local.masked_fill(local_mask, float('-inf'))
        attn_local = F.softmax(attn_local * self.local_scale, dim=-1)
        attn_local = self.dropout(attn_local)
        out_local = torch.matmul(attn_local, V_local)
        
        # Global attention
        attn_global = torch.matmul(Q_global, K_global.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_global = F.softmax(attn_global * self.global_scale, dim=-1)
        attn_global = self.dropout(attn_global)
        out_global = torch.matmul(attn_global, V_global)
        
        out = torch.cat([out_local, out_global], dim=1)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out, attn_local, attn_global


# =============================================================================
# RIGHT HEMISPHERE v2.0 - OPTIMIZADO
# =============================================================================
class RightHemisphereV2(nn.Module):
    def __init__(self, output_dim=384):
        super().__init__()
        # ResNet-34 en lugar de 50 (más ligero)
        resnet = models.resnet34(pretrained=True)
        
        # Congelar más layers
        for param in list(resnet.parameters())[:-25]:
            param.requires_grad = False
        
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Spatial attention simplificada
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),  # 2048 -> 512 (ResNet34)
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_liquid = LiquidNeuronV2(512, output_dim)
        
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
        with torch.cuda.amp.autocast():  # Mixed precision
            features_map = self.visual_encoder[:-1](image)
            
            attn_map = self.spatial_attention(features_map)
            features_map = features_map * attn_map
            
            features = self.global_pool(features_map).flatten(1)
        
        liquid_features = self.spatial_liquid(features.float(), plasticity, transfer_rate)
        global_features = self.global_fc(features.float())
        
        visual_thought = self.fusion(torch.cat([liquid_features, global_features], dim=1))
        
        return visual_thought


# =============================================================================
# LEFT HEMISPHERE v2.0 - OPTIMIZADO
# =============================================================================
class LeftHemisphereV2(nn.Module):
    def __init__(self, vocab_size, embed_dim=192, hidden_dim=384):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim,
            hidden_dim, 
            num_layers=1,  # 2 -> 1 layer
            batch_first=True,
            dropout=0.0  # No dropout en 1 layer
        )
        
        self.attention = BicameralAttention(hidden_dim, num_heads=4)
        
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
        
        self.curiosity = EpistemicCuriosity(hidden_dim, hidden_dim // 3)
        
    def forward(self, visual_context, captions=None, max_len=30, 
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
            
            # Curiosity loss con sampling
            if exploration_bonus > 0:
                # Sample para reducir memoria
                sample_size = min(lstm_out.size(0) * (lstm_out.size(1) - 1), 1024)
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
                
                probs = F.softmax(logits, dim=-1)
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
# CORPUS CALLOSUM v2.0 - SIMPLIFICADO
# =============================================================================
class CorpusCallosumV2(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.right_to_left = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.mixing_gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, right_features):
        transformed = self.right_to_left(right_features)
        
        combined = torch.cat([right_features, transformed], dim=-1)
        alpha = self.mixing_gate(combined)
        
        output = alpha * transformed + (1 - alpha) * right_features
        
        return output, alpha


# =============================================================================
# BICAMERAL MODEL v2.0
# =============================================================================
class NeuroLogosBicameralV2(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.right_hemisphere = RightHemisphereV2(output_dim=384)
        self.left_hemisphere = LeftHemisphereV2(vocab_size, embed_dim=192, hidden_dim=384)
        self.corpus_callosum = CorpusCallosumV2(dim=384)
        
    def forward(self, image, captions=None, plasticity=0.1, transfer_rate=0.005, 
                return_diagnostics=False, temperature=0.9, exploration_bonus=0.0):
        visual_features = self.right_hemisphere(image, plasticity, transfer_rate)
        visual_context, callosum_gate = self.corpus_callosum(visual_features)
        
        if return_diagnostics and captions is not None:
            output, gate, local_attn, global_attn, curiosity_loss = self.left_hemisphere(
                visual_context, captions, return_diagnostics=True, 
                temperature=temperature, exploration_bonus=exploration_bonus
            )
            return output, visual_features, visual_context, gate, local_attn, global_attn, callosum_gate, curiosity_loss
        
        output, curiosity_loss = self.left_hemisphere(
            visual_context, captions, temperature=temperature, 
            exploration_bonus=exploration_bonus
        )
        return output, curiosity_loss


# =============================================================================
# DIAGNOSTICS - SIMPLIFICADO
# =============================================================================
class NeuralDiagnosticsV2:
    def __init__(self):
        self.history = {
            'loss': [],
            'curiosity_loss': [],
            'right_liquid_norm': [],
            'callosal_flow': [],
            'callosum_gate': [],
            'left_gate_mean': [],
            'left_gate_std': [],
            'vocab_diversity': [],
            'vocab_entropy': [],
            'plasticity_effective': [],
            'homeostasis_signal': []
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

    
    # FIX: MÉTODO UPDATE FALTANTE
    def update(self, **metrics):
        for key, value in metrics.items():
            if key in self.history and value is not None:
                self.history[key].append(value)

    
    def get_recent_avg(self, key, n=50):
        if key in self.history and len(self.history[key]) > 0:
            values = [v for v in self.history[key][-n:] if v is not None]
            if len(values) > 0:
                return np.mean(values)
        return 0.0
    
    def report(self, epoch):
        if len(self.history['loss']) == 0:
            return
        
        print(f"\n{'='*80}")
        print(f"🧠 DIAGNÓSTICO NEUROLÓGICO v2.0 - Época {epoch}")
        print(f"{'='*80}")
        
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
        
        h_status = "🟢 Estable" if 0.5 <= liquid <= 3.5 and 0.8 <= homeo <= 2.0 else \
                   "🟡 Alto" if liquid < 5.0 else "🔴 Crítico"
        print(f"👁️ Right Liquid Norm: {liquid:.3f} | Homeostasis: {homeo:.3f} {h_status}")
        
        f_status = "🟢 Fluido" if flow > 0.3 else "🟡 Débil" if flow > 0.1 else "🔴 Bloqueado"
        print(f"🔗 Callosal Flow: {flow:.3f} | Gate: {callosum_gate:.3f} {f_status}")
        
        g_status = "🟢 Balanceado" if 0.2 < gate_mean < 0.8 and gate_std > 0.1 else "🟡 Sesgado"
        print(f"💬 Left Gate: μ={gate_mean:.3f} σ={gate_std:.3f} {g_status}")
        
        v_status = "🟢 Diverso" if vocab_div > 0.1 else "🟡 Limitado" if vocab_div > 0.05 else "🔴 Colapsado"
        print(f"📚 Vocab Div: {vocab_div:.3f} | Entropy: {vocab_ent:.2f} {v_status}")
        print(f"   Tokens únicos: {len(self.vocab_usage)}")
        
        print(f"🧬 Plasticidad: {plast:.4f}")
        print(f"{'='*80}\n")




# =============================================================================
# CURRICULUM LEARNING
# =============================================================================
class CurriculumScheduler:
    def __init__(self, total_epochs=30):
        self.total = total_epochs
        self.phases = {
            'exploration': (0, 8),
            'refinement': (8, 20),
            'mastery': (20, 30)
        }
    
    def get_phase(self, epoch):
        for phase_name, (start, end) in self.phases.items():
            if start <= epoch < end:
                return phase_name
        return 'mastery'
    
    def get_plasticity(self, epoch):
        phase = self.get_phase(epoch)
        
        if phase == 'exploration':
            return 0.35 * (1 - epoch / 8) + 0.15
        elif phase == 'refinement':
            progress = (epoch - 8) / 12
            return 0.15 * (1 - progress) + 0.05
        else:
            return 0.02
    
    def get_exploration_bonus(self, epoch):
        phase = self.get_phase(epoch)
        
        if phase == 'exploration':
            return 0.25 * (1 - epoch / 8) + 0.10
        elif phase == 'refinement':
            return 0.10 * (1 - (epoch - 8) / 12)
        else:
            return 0.0
    
    def get_temperature(self, epoch):
        phase = self.get_phase(epoch)
        
        if phase == 'exploration':
            return 1.2
        elif phase == 'refinement':
            progress = (epoch - 8) / 12
            return 1.2 - progress * 0.3
        else:
            return 0.85
    
    def should_consolidate(self, epoch):
        return epoch > 0 and epoch % 7 == 0


def build_vocab_flickr(captions_file, vocab_size=5000):
    print("Building vocabulary...")
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
    print(f"Vocabulary size: {len(vocab)}")
    return vocab, id2word




class Flickr8kDataset(Dataset):
    def __init__(self, images_dir, captions_file, vocab, transform=None, max_len=30):
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
        
        print(f"Loaded {len(self.data)} image-caption pairs")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # FIX: robust image loading to avoid DataLoader worker crashes
        max_retries = 10

        for _ in range(max_retries):
            img_path, caption = self.data[idx]
            try:
                image = Image.open(img_path).convert('RGB')
                break
            except Exception:
                # Si la imagen falla, probamos con otro índice aleatorio
                idx = np.random.randint(0, len(self.data))
        else:
            # Último recurso: imagen negra si todo falló
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





def setup_flickr8k(data_dir='./data'):
    """Descarga Flickr8k automáticamente"""
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
        'images': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
        'captions': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
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
    
    if os.path.exists(os.path.join(flickr_dir, 'Flicker8k_Dataset')):
        import shutil
        old_dir = os.path.join(flickr_dir, 'Flicker8k_Dataset')
        if not os.path.exists(images_dir):
            shutil.move(old_dir, images_dir)
    
    print("✅ Flickr8k listo\n")
    return flickr_dir


# =============================================================================
# TRAINING LOOP v2.0 - OPTIMIZADO PARA MEMORIA
# =============================================================================
def train_bicameral_v2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"🧠 NeuroLogos Bicameral v2.0 OPTIMIZADO | Device: {device}")
    print(f"{'='*80}\n")
    
    # Liberar memoria
    torch.cuda.empty_cache()
    gc.collect()
    
    flickr_dir = setup_flickr8k()
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    vocab, id2word = build_vocab_flickr(captions_file, VOCAB_SIZE)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Flickr8kDataset(images_dir, captions_file, vocab, transform, MAX_CAPTION_LEN)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=False, 
        persistent_workers=True
    )

    
    model = NeuroLogosBicameralV2(len(vocab)).to(device)
    
    # Optimizer con weight decay diferenciado
    visual_params = list(model.right_hemisphere.parameters())
    language_params = list(model.left_hemisphere.parameters()) + list(model.corpus_callosum.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': visual_params, 'lr': 8e-5, 'weight_decay': 1e-4},
        {'params': language_params, 'lr': 2e-4, 'weight_decay': 1e-5}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler()
    
    curriculum = CurriculumScheduler(total_epochs=30)
    diagnostics = NeuralDiagnosticsV2()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"Workers: {NUM_WORKERS}\n")
    
    os.makedirs('./checkpoints', exist_ok=True)
    
    start_epoch = 0
    checkpoint_path = './checkpoints/latest_checkpoint_v2.pth'
    if os.path.exists(checkpoint_path):
        print("📁 Encontrado checkpoint v2, resumiendo entrenamiento...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        diagnostics.history = checkpoint['diagnostics_history']
        diagnostics.vocab_usage = checkpoint.get('vocab_usage', Counter())
        print(f"✓ Resumido desde época {start_epoch}\n")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    global_batch_counter = 0  # CONTADOR GLOBAL PARA REPORTES CADA 500
    
    for epoch in range(start_epoch, 30):
        phase = curriculum.get_phase(epoch)
        plasticity = curriculum.get_plasticity(epoch)
        exploration_bonus = curriculum.get_exploration_bonus(epoch)
        transfer_rate = 0.01 if epoch < 20 else 0.003
        temperature = curriculum.get_temperature(epoch)
        
        print(f"\n🎓 Fase: {phase.upper()} | Época {epoch}")
        print(f"   Plasticidad: {plasticity:.3f} | Exploración: {exploration_bonus:.3f} | Temp: {temperature:.2f}")
        
        model.train()
        total_loss = 0
        total_curiosity = 0
        num_batches = 0
        
        if curriculum.should_consolidate(epoch):
            print("🔧 Consolidando memoria (SVD)...")
            model.right_hemisphere.spatial_liquid.consolidate_svd(0.7, 'medium')
            model.right_hemisphere.spatial_liquid.consolidate_svd(0.5, 'fast')
            torch.cuda.empty_cache()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}")
        
        for batch_idx, (images, captions) in enumerate(pbar):
            global_batch_counter += 1
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            # Forward con mixed precision
            with torch.cuda.amp.autocast():
                logits, visual_features, visual_context, gate, local_attn, global_attn, callosum_gate, curiosity_loss = model(
                    images, captions, plasticity, transfer_rate, 
                    return_diagnostics=True, temperature=temperature,
                    exploration_bonus=exploration_bonus
                )
                
                # Cross-entropy loss
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, len(vocab)),
                    captions[:, 1:].reshape(-1),
                    ignore_index=vocab['<PAD>'],
                    label_smoothing=0.1 if phase == 'exploration' else 0.0
                )
                
                # Vocab diversity bonus
                probs = F.softmax(logits, dim=-1)
                vocab_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                entropy_bonus = -vocab_entropy * 0.03 if phase == 'exploration' else -vocab_entropy * 0.01
                
                # Gate entropy
                gate_entropy = -(gate * torch.log(gate + 1e-8) + (1 - gate) * torch.log(1 - gate + 1e-8)).mean()
                gate_loss = -gate_entropy * 0.015
                
                # Curiosity loss con peso adaptativo
                curiosity_weight = 0.1 if phase == 'exploration' else 0.05 if phase == 'refinement' else 0.0
                
                # Total loss con gradient accumulation
                loss = (ce_loss + gate_loss + entropy_bonus + curiosity_weight * curiosity_loss) / GRADIENT_ACCUMULATION_STEPS
            
            # Backward con gradient scaling
            scaler.scale(loss).backward()
            
            # Update cada GRADIENT_ACCUMULATION_STEPS
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Actualizar prediction error para liquid neurons
                with torch.no_grad():
                    model.right_hemisphere.spatial_liquid.prediction_error = \
                        (ce_loss / 8.0).clamp(0.0, 0.7).item()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Diagnostics POR BATCH - CADA 500 GLOBAL
            with torch.no_grad():
                liquid_norm = model.right_hemisphere.spatial_liquid.W_fast.norm().item()
                callosal_flow = diagnostics.measure_callosal_flow(visual_features, visual_context)
                gate_mean = gate.mean().item()
                gate_std = gate.std().item()
                callosum_gate_val = callosum_gate.mean().item()
                
                # HOMEOSTASIS CRÍTICA POR BATCH
                liquid = model.right_hemisphere.spatial_liquid
                homeostasis_signal = (liquid.activation_std_ema / 
                                    (liquid.activation_mean_ema.abs() + 1e-6)).item()
                
                stats = torch.cat([
                    visual_features.mean(dim=1, keepdim=True),
                    visual_features.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6),
                    liquid.activation_std_ema.expand(visual_features.size(0), 1) / 
                    (liquid.activation_mean_ema.abs().expand(visual_features.size(0), 1) + 1e-6)
                ], dim=1)
                learned_plast = liquid.plasticity_controller(stats).mean().item()
                effective_plast = plasticity * learned_plast * (1.0 - liquid.prediction_error * 0.5)
                
                vocab_div = None
                vocab_ent = None

                if batch_idx % 100 == 0:
                    with torch.cuda.amp.autocast():
                        sample_gen = model(images[:2], captions=None,
                                        temperature=temperature,
                                        exploration_bonus=exploration_bonus)
       
                        if isinstance(sample_gen, tuple):
                            sample_gen = sample_gen[0]
                    vocab_div, vocab_ent = diagnostics.measure_vocab_diversity(sample_gen, len(vocab))

                                
                # DECISIONES AUTOMÁTICAS HOMEOSTÁTICAS
                actions_taken = []
                if liquid_norm > 4.0:
                    model.right_hemisphere.spatial_liquid.consolidate_svd(0.9, 'fast')
                    actions_taken.append("SVD_fast(0.9)")
                if homeostasis_signal < 0.6:
                    plasticity = min(0.5, plasticity + 0.08)
                    actions_taken.append(f"plasticity+0.08={plasticity:.3f}")
                if gate_mean < 0.2 or gate_mean > 0.8:
                    model.left_hemisphere.residual_gate.data += 0.02 * (0.5 - gate_mean)
                    actions_taken.append(f"residual_gate={model.left_hemisphere.residual_gate.item():.3f}")
                if callosal_flow < 0.15:
                    model.corpus_callosum.mixing_gate[-1].bias.data += 0.1
                    actions_taken.append("mixing_bias+0.1")
                
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

                if global_batch_counter % 500 == 0:
                    print(f"\n📊 BATCH GLOBAL #{global_batch_counter} | Época {epoch} Batch {batch_idx}")
                    print(f"   🧠 Loss: {ce_loss.item():.4f} | Curiosity: {curiosity_loss.item():.4f}")
                    print(f"   💧 LiqNorm: {liquid_norm:.2f} | Homeostasis: {homeostasis_signal:.3f}")
                    print(f"   🔗 CallosalFlow: {callosal_flow:.3f} | Gate: {callosum_gate_val:.3f}")
                    print(f"   💬 LeftGate: μ={gate_mean:.3f} σ={gate_std:.3f}")
                    print(f"   🧬 Plast.Efectiva: {effective_plast:.4f} | Acciones: {actions_taken}")
                    vocab_div_str = f"{vocab_div:.3f}" if vocab_div is not None else "N/A"
                    vocab_ent_str = f"{vocab_ent:.2f}" if vocab_ent is not None else "N/A"
                    print(f"   📚 VocabDiv: {vocab_div_str}")
                    print(f"   🎆 Entropy: {vocab_ent_str}")

                
                # ALERTAS CRISIS (raro, solo anomalías graves)
                if liquid_norm > 5.0 or homeostasis_signal < 0.5:
                    print(f"\n🚨 CRISIS BATCH {global_batch_counter}: liq={liquid_norm:.2f} "
                          f"homeo={homeostasis_signal:.3f} | Acciones: {actions_taken}")
            
            total_loss += ce_loss.item()
            total_curiosity += curiosity_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{ce_loss.item():.3f}',
                'cur': f'{curiosity_loss.item():.3f}',
                'liq': f'{liquid_norm:.2f}',
                'flow': f'{callosal_flow:.2f}'
            })
            
            # Limpiar memoria
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Final gradient step
        if len(dataloader) % GRADIENT_ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            scaler.step(optimizer)
            scaler.update()
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
            }, f'./checkpoints/epoch{epoch:02d}_v2.pth')
        
        # Generar muestras cada 2 épocas
        if epoch % 2 == 0:
            model.eval()
            print("\n🔍 GENERANDO MUESTRAS...")
            with torch.no_grad():
                for sample_idx in range(3):
                    sample_img, sample_cap = dataset[sample_idx * 100]
                    sample_img = sample_img.unsqueeze(0).to(device)
                    
                    with torch.cuda.amp.autocast():
                        generated = model(sample_img, captions=None, 
                                        temperature=0.9, exploration_bonus=0.0)
                    
                    words = [id2word.get(int(t.item()), '<UNK>') for t in generated[0]]
                    sentence = ' '.join(words[:15])  # Primeras 15 palabras
                    
                    gt_words = [id2word.get(int(t.item()), '<UNK>') for t in sample_cap]
                    gt_sentence = ' '.join(gt_words[1:16])  # Skip BOS
                
                    print(f"   Muestra {sample_idx+1}:")
                    print(f"      GT: {gt_sentence}")
                    print(f"      Gen: {sentence}")
            model.train()
            torch.cuda.empty_cache()
        
        print(f"Época {epoch:02d} completada | Avg Loss: {total_loss/num_batches:.4f} | "
              f"Curiosity: {total_curiosity/num_batches:.4f}")
        torch.cuda.empty_cache()
    
    print("✅ Entrenamiento completado!")
    gc.collect()

if __name__ == "__main__":
    train_bicameral_v2()
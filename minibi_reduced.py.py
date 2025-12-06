# =============================================================================
# NeuroLogos Bicameral v2.0 - Arquitectura Neural Avanzada (OPTIMIZADO COLAB)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from collections import Counter, deque
import torchvision.models as models
import warnings
from tqdm import tqdm
import math
import gc # Garbage Collector

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES OPTIMIZADAS PARA COLAB
# =============================================================================
IMG_SIZE = 224
MAX_CAPTION_LEN = 30
VOCAB_SIZE = 5000
EMBED_DIM = 256
HIDDEN_DIM = 512
# REDUCIDO DRASTICAMENTE PARA EVITAR CRASH
BATCH_SIZE = 16  
# REDUCIDO PARA EVITAR ERROR DE BUS/MEMORIA COMPARTIDA
NUM_WORKERS = 2  
ACCUMULATION_STEPS = 4  # 16 * 4 = 64 (Batch efectivo original)

# =============================================================================
# CURIOSITY MODULE - Exploración Epistémica
# =============================================================================
class EpistemicCuriosity(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
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
        self.register_buffer('novelty_history', torch.zeros(1000))
        self.register_buffer('history_idx', torch.tensor(0))
        
    def compute_intrinsic_reward(self, state, action, next_state):
        action_onehot = F.one_hot(action, VOCAB_SIZE).float()
        predicted_next = self.forward_model(torch.cat([state, action_onehot], dim=-1))
        prediction_error = F.mse_loss(predicted_next, next_state, reduction='none').mean(dim=-1)
        
        with torch.no_grad():
            idx = self.history_idx.item() % 1000
            self.novelty_history[idx] = prediction_error.mean().item()
            self.history_idx += 1
            mean_novelty = self.novelty_history.mean()
            std_novelty = self.novelty_history.std().clamp(min=1e-4)
            normalized_reward = (prediction_error - mean_novelty) / std_novelty
        
        return prediction_error, normalized_reward
    
    def update(self, state, action, next_state):
        action_onehot = F.one_hot(action, VOCAB_SIZE).float()
        pred_next = self.forward_model(torch.cat([state, action_onehot], dim=-1))
        forward_loss = F.mse_loss(pred_next, next_state.detach())
        pred_action = self.inverse_model(torch.cat([state, next_state], dim=-1))
        inverse_loss = F.cross_entropy(pred_action, action)
        return forward_loss + inverse_loss

# =============================================================================
# LIQUID NEURON v2.0
# =============================================================================
class LiquidNeuronV2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        self.register_buffer('W_medium', torch.zeros(out_dim, in_dim))
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.ln = nn.LayerNorm(out_dim)
        
        self.plasticity_controller = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.Tanh(),
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
        
        if self.training:
            with torch.no_grad():
                self.activation_mean_ema = self.ema_momentum * self.activation_mean_ema + \
                                          (1 - self.ema_momentum) * batch_mean.mean()
                self.activation_std_ema = self.ema_momentum * self.activation_std_ema + \
                                         (1 - self.ema_momentum) * batch_std.mean()
        
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
                target_norm = 3.5
                if current_norm > target_norm:
                    self.W_fast.data *= (target_norm / current_norm) ** 0.1
        return out
    
    def consolidate_svd(self, repair_strength=1.0, timescale='fast'):
        # Mueve a CPU si es necesario para evitar OOM durante SVD
        if timescale == 'fast': W_target = self.W_fast
        elif timescale == 'medium': W_target = self.W_medium
        else: return False
        
        with torch.no_grad():
            try:
                # Usar CPU para SVD para ahorrar VRAM
                W_cpu = W_target.cpu()
                U, S, Vt = torch.linalg.svd(W_cpu, full_matrices=False)
                threshold = S.max() * 0.008 * repair_strength
                mask = S > threshold
                filtered_S = S * mask.float()
                W_consolidated = U @ torch.diag(filtered_S) @ Vt
                
                # De vuelta a dispositivo original
                W_consolidated = W_consolidated.to(W_target.device)
                W_target.data = (1.0 - repair_strength) * W_target.data + repair_strength * W_consolidated
                W_target.data *= 0.99
                return True
            except Exception as e:
                print(f"SVD failed: {e}")
                W_target.data.mul_(0.95)
                return False

# =============================================================================
# MULTI-SCALE ATTENTION
# =============================================================================
class BicameralAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.local_scale = nn.Parameter(torch.ones(1))
        self.global_scale = nn.Parameter(torch.ones(1))
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        local_heads = self.num_heads // 2
        Q_local, K_local, V_local = Q[:, :local_heads], K[:, :local_heads], V[:, :local_heads]
        local_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn_local = torch.matmul(Q_local, K_local.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_local = attn_local.masked_fill(local_mask, float('-inf'))
        attn_local = F.softmax(attn_local * self.local_scale, dim=-1)
        attn_local = self.dropout(attn_local)
        out_local = torch.matmul(attn_local, V_local)
        
        Q_global, K_global, V_global = Q[:, local_heads:], K[:, local_heads:], V[:, local_heads:]
        attn_global = torch.matmul(Q_global, K_global.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_global = F.softmax(attn_global * self.global_scale, dim=-1)
        attn_global = self.dropout(attn_global)
        out_global = torch.matmul(attn_global, V_global)
        
        out = torch.cat([out_local, out_global], dim=1)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out, attn_local, attn_global

# =============================================================================
# HEMISPHERES
# =============================================================================
class RightHemisphereV2(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in list(resnet.parameters())[:-40]:
            param.requires_grad = False
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_liquid = LiquidNeuronV2(2048, output_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(2048, output_dim)
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.output_dim = output_dim
        
    def forward(self, image, plasticity=0.1, transfer_rate=0.005):
        features_map = self.visual_encoder[:-1](image)
        attn_map = self.spatial_attention(features_map)
        features_map = features_map * attn_map
        features = self.global_pool(features_map).flatten(1)
        liquid_features = self.spatial_liquid(features, plasticity, transfer_rate)
        global_features = self.global_fc(features)
        visual_thought = self.fusion(torch.cat([liquid_features, global_features], dim=1))
        return visual_thought

class LeftHemisphereV2(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.attention = BicameralAttention(hidden_dim, num_heads=8)
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.5))
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.curiosity = EpistemicCuriosity(hidden_dim, hidden_dim // 2)
        
    def forward(self, visual_context, captions=None, max_len=30, return_diagnostics=False, temperature=0.9, exploration_bonus=0.0):
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
            
            curiosity_loss = torch.tensor(0.0, device=device)
            if exploration_bonus > 0:
                states = lstm_out[:, :-1].reshape(-1, self.hidden_dim)
                actions = captions[:, 1:-1].reshape(-1)
                next_states = lstm_out[:, 1:].reshape(-1, self.hidden_dim)
                # Ensure sizes match
                min_len = min(states.size(0), actions.size(0))
                if min_len > 0:
                    curiosity_loss = self.curiosity.update(states[:min_len], actions[:min_len], next_states[:min_len])
            
            if return_diagnostics:
                return logits, gate, local_attn, global_attn, curiosity_loss
            return logits, curiosity_loss
        
        else:
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
                if (next_token == 2).all(): break
            return torch.cat(generated, dim=1)
    
    def _get_init_state(self, visual_context):
        h0 = visual_context.unsqueeze(0).repeat(2, 1, 1)
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

class CorpusCallosumV2(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.right_to_left = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.mixing_gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.Tanh(), nn.Linear(dim, 1), nn.Sigmoid()
        )
        
    def forward(self, right_features):
        transformed = self.right_to_left(right_features)
        combined = torch.cat([right_features, transformed], dim=-1)
        alpha = self.mixing_gate(combined)
        output = alpha * transformed + (1 - alpha) * right_features
        return output, alpha

class NeuroLogosBicameralV2(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.right_hemisphere = RightHemisphereV2(output_dim=512)
        self.left_hemisphere = LeftHemisphereV2(vocab_size, embed_dim=256, hidden_dim=512)
        self.corpus_callosum = CorpusCallosumV2(dim=512)
        
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
            visual_context, captions, temperature=temperature, exploration_bonus=exploration_bonus
        )
        return output, curiosity_loss

# =============================================================================
# DATASET HELPERS
# =============================================================================
def build_vocab_flickr(captions_file, vocab_size=5000):
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
                        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        img_path, caption = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
        except:
            # Fallback para imágenes corruptas
            image = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            
        tokens = ['<BOS>'] + caption.lower().split() + ['<EOS>']
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        return image, torch.tensor(token_ids, dtype=torch.long)

def setup_flickr8k(data_dir='./data'):
    flickr_dir = os.path.join(data_dir, 'flickr8k')
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    if os.path.exists(images_dir) and os.path.exists(captions_file):
        return flickr_dir
    
    os.makedirs(flickr_dir, exist_ok=True)
    import urllib.request
    import zipfile
    
    urls = {
        'images': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
        'captions': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
    }
    
    try:
        for name, url in urls.items():
            zip_path = os.path.join(flickr_dir, f'{name}.zip')
            if not os.path.exists(zip_path) and not (name == 'images' and os.path.exists(images_dir)):
                print(f"Descargando {name}...")
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(flickr_dir)
                os.remove(zip_path)
    except Exception as e:
        print(f"Error descargando datos: {e}")
        return None

    # Procesar captions
    raw_captions = os.path.join(flickr_dir, 'Flickr8k.token.txt')
    if os.path.exists(raw_captions) and not os.path.exists(captions_file):
        with open(captions_file, 'w', encoding='utf-8') as outfile:
            with open(raw_captions, 'r', encoding='utf-8') as infile:
                for line in infile:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        outfile.write(f"{parts[0].split('#')[0]}\t{parts[1]}\n")
                        
    if os.path.exists(os.path.join(flickr_dir, 'Flicker8k_Dataset')):
        import shutil
        old_dir = os.path.join(flickr_dir, 'Flicker8k_Dataset')
        if not os.path.exists(images_dir):
            shutil.move(old_dir, images_dir)
            
    return flickr_dir

# =============================================================================
# TRAINING LOOP OPTIMIZADO (MIXED PRECISION + GRAD ACCUMULATION)
# =============================================================================
def train_bicameral_v2():
    # Limpieza inicial de memoria
    torch.cuda.empty_cache()
    gc.collect()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    flickr_dir = setup_flickr8k()
    if flickr_dir is None:
        print("Error crítico: No se pudo cargar el dataset.")
        return

    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    vocab, id2word = build_vocab_flickr(captions_file, VOCAB_SIZE)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Flickr8kDataset(images_dir, captions_file, vocab, transform, MAX_CAPTION_LEN)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    model = NeuroLogosBicameralV2(len(vocab)).to(device)
    
    # Optimizer config
    visual_params = list(model.right_hemisphere.parameters())
    language_params = list(model.left_hemisphere.parameters()) + list(model.corpus_callosum.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': visual_params, 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': language_params, 'lr': 3e-4, 'weight_decay': 1e-5}
    ])
    
    # SCALER PARA MIXED PRECISION
    scaler = torch.cuda.amp.GradScaler()
    
    print("Iniciando entrenamiento optimizado...")
    
    for epoch in range(5): # Reducido a 5 epochs para prueba, subir después
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, captions) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            # Autocast para Mixed Precision (Ahorra VRAM)
            with torch.cuda.amp.autocast():
                logits, _, _, gate, _, _, _, curiosity_loss = model(
                    images, captions, plasticity=0.1, return_diagnostics=True
                )
                
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, len(vocab)),
                    captions[:, 1:].reshape(-1),
                    ignore_index=vocab['<PAD>']
                )
                
                loss = ce_loss + 0.05 * curiosity_loss
                loss = loss / ACCUMULATION_STEPS # Normalizar loss para acumulación
            
            # Backward con Scaler
            scaler.scale(loss).backward()
            
            # Gradient Accumulation Step
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUMULATION_STEPS
            
            pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}'})
            
            # Limpieza forzada de memoria cada cierto tiempo
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        print(f"Epoch {epoch} finalizada.")
        
        # Guardar checkpoint simple
        torch.save(model.state_dict(), 'model_checkpoint.pth')

if __name__ == "__main__":
    train_bicameral_v2()
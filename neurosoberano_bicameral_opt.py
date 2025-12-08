#%%writefile neurosoberano_bicameral_optimized.py
# =============================================================================
# NeuroLogos Bicameral Minimalista v1.1 - OPTIMIZADO + FISIOLOGÍA VIVA
# Mejoras: DataLoader rápido, checkpointing, diagnóstico completo,
# Homeostasis fisiológica, ruido térmico, fatiga, daño, reparación post-step
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
from tqdm import tqdm
import warnings
import math
import random
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES GLOBALES
# =============================================================================
IMG_SIZE = 224
MAX_CAPTION_LEN = 30
VOCAB_SIZE = 5000
EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 64  # Aumentado de 32 a 64
NUM_WORKERS = 4  # Aumentado de 2 a 4
PRINT_EVERY = 10  # Frecuencia de prints por batch (puedes ajustar)
SVD_CONSOLIDATE_INTERVAL = 10  # épocas

# =============================================================================
# HOMEOSTATIC REGULATOR (REUSABLE)
# =============================================================================
class HomeostaticRegulator(nn.Module):
    def __init__(self):
        super().__init__()
        # entrada: stress, excitation, fatigue, entropy, phase, loss_signal
        self.net = nn.Sequential(
            nn.Linear(6, 24),
            nn.Tanh(),
            nn.Linear(24, 5),
            nn.Sigmoid()
        )

    def forward(self, stress, excitation, fatigue, entropy, phase, loss_signal):
        x = torch.cat([stress, excitation, fatigue, entropy, phase, loss_signal], dim=1)
        out = self.net(x)
        return {
            'metabolism': out[:, 0:1],
            'sensitivity': out[:, 1:2],
            'gate': out[:, 2:3],
            'repair': out[:, 3:4],
            'plasticity': out[:, 4:5]
        }

# =============================================================================
# LIQUID NEURON (AMPLIADO CON FISIOLOGÍA)
# - Mantiene interfaz para no romper el resto del código.
# - Almacena _last_phys, _last_post, _last_pre para actualización post-step.
# =============================================================================
class LiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        
        # buffers (W_medium/W_fast conservan la semántica)
        self.register_buffer('W_medium', torch.zeros(out_dim, in_dim))
        # iniciar W_fast pequeño para fisiología estable
        self.register_buffer('W_fast', 0.01 * torch.randn(out_dim, in_dim))
        
        self.ln = nn.LayerNorm(out_dim)
        
        self.plasticity_controller = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.plasticity_controller[2].bias.data.fill_(-2.0)
        self.base_lr = 0.015
        self.prediction_error = 0.0

        # fisiología adicional
        self.regulator = HomeostaticRegulator()
        self.register_buffer('fatigue', torch.zeros(1))
        self.register_buffer('damage', torch.zeros(1))
        self.thermal_noise = 0.02
        self.phase = 0.0
        self.phase_rate = random.uniform(0.005, 0.02)

        # para post-step hebbian
        self._last_phys = None
        self._last_post = None
        self._last_pre = None
        
    def forward(self, x, global_plasticity=0.0, transfer_rate=0.005, task_loss=0.0):
        # slow path
        slow_out = self.W_slow(x)
        
        # thermal noise injection (exploration)
        thermal = self.thermal_noise * torch.randn_like(x)
        
        # fast path uses W_fast (matrix multiply)
        fast_out = F.linear(x + thermal, self.W_fast)
        medium_out = F.linear(x, self.W_medium)
        
        pre_act = slow_out + medium_out + fast_out
        
        # physiological signals
        batch_mean = pre_act.mean(dim=1, keepdim=True)
        batch_std = pre_act.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
        stats = torch.cat([batch_mean, batch_std], dim=1)
        
        # compute entropy of slow representation as differentiable signal
        probs = F.softmax(slow_out, dim=1)
        entropy = (- (probs * torch.log(probs + 1e-12)).sum(dim=1, keepdim=True)).clamp(min=0.0)
        
        # stress / excitation
        stress = x.var(dim=1, keepdim=True)
        excitation = slow_out.abs().mean(dim=1, keepdim=True)
        
        # phase oscillator
        self.phase = float(self.phase + self.phase_rate)
        phase_signal = torch.full((x.size(0), 1), math.sin(self.phase), device=x.device)
        
        # loss_signal for regulator (task_loss may be scalar)
        if torch.is_tensor(task_loss):
            loss_signal = task_loss.reshape(1,1).expand_as(stress)
        else:
            loss_signal = torch.full_like(stress, float(task_loss))
        
        phys = self.regulator(stress, excitation, self.fatigue.expand(x.size(0),1), entropy, phase_signal, loss_signal)
        
        # gating of fast dynamics
        combined = slow_out + phys['gate'] * fast_out
        
        beta = 0.5 + 2.0 * phys['sensitivity']
        out = combined * torch.sigmoid(beta * combined)
        out = 5.0 * torch.tanh(self.ln(out) / 5.0)
        
        # update fatigue/damage buffers (no grad)
        with torch.no_grad():
            self.fatigue *= 0.995
            self.fatigue += 0.01 * stress.mean()
            self.fatigue.clamp_(0.0, 5.0)
            
            self.damage *= 0.99
            self.damage += 0.001 * excitation.mean()
            self.damage.clamp_(0.0, 5.0)
        
        # plasticity controller (uses stats)
        learned_plasticity = self.plasticity_controller(stats).squeeze(1)
        effective_plasticity = global_plasticity * learned_plasticity * (1.0 - self.prediction_error)
        # modulate by homeostatic plasticity signal as well:
        effective_plasticity = effective_plasticity * phys['plasticity'].squeeze(1).detach()
        
        # local hebbian-like update stored for post-step (we avoid in-forward weight updates besides small ops)
        with torch.no_grad():
            # store last pre/post for hebb post-step
            self._last_post = slow_out.detach()
            self._last_pre = (x + thermal).detach()
            self._last_phys = phys  # physiol signals for post-step repair/plasticity
        
        return out

    def apply_svd_consolidation(self, repair_strength=1.0, timescale='fast'):
        if timescale == 'fast':
            W_target = self.W_fast
        elif timescale == 'medium':
            W_target = self.W_medium
        else:
            return False
        with torch.no_grad():
            try:
                U, S, Vt = torch.linalg.svd(W_target, full_matrices=False)
                threshold = S.max() * 0.01 * repair_strength
                mask = S > threshold
                filtered_S = S * mask.float()
                W_consolidated = U @ torch.diag(filtered_S) @ Vt
                W_target.data = (1.0 - repair_strength) * W_target.data + repair_strength * W_consolidated
                W_target.data *= 0.98
                return True
            except Exception:
                W_target.data.mul_(0.9)
                return False

# =============================================================================
# HEMISFERIO DERECHO (mantiene interfaz original)
# =============================================================================
class RightHemisphere(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        
        for param in list(resnet.parameters())[:-20]:
            param.requires_grad = False
        
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.spatial_liquid = LiquidNeuron(2048, output_dim)
        self.output_dim = output_dim
        
    def forward(self, image, plasticity=0.1, transfer_rate=0.005, task_loss=0.0):
        features = self.visual_encoder(image)
        features = features.flatten(1)
        visual_thought = self.spatial_liquid(features, plasticity, transfer_rate, task_loss)
        return visual_thought

# =============================================================================
# HEMISFERIO IZQUIERDO (sin cambios funcionales)
# =============================================================================
class LeftHemisphere(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim,
            hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.3
        )
        
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        
    def forward(self, visual_context, captions=None, max_len=30, return_gate=False):
        batch_size = visual_context.size(0)
        device = visual_context.device
        
        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            seq_len = embeddings.size(1)
            
            visual_expanded = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_input = torch.cat([embeddings, visual_expanded], dim=2)
            
            lstm_out, _ = self.lstm(lstm_input, self._get_init_state(visual_context))
            
            gate = self.liquid_gate(lstm_out)
            # anti-saturación del gate
            gate = gate.clamp(0.03, 0.97)
            modulated = lstm_out * gate
            
            logits = self.output_projection(modulated)
            
            if return_gate:
                return logits, gate
            return logits
        
        else:
            generated = []
            hidden = self._get_init_state(visual_context)
            input_token = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
            
            temperature = 0.9
            top_p = 0.92
            
            for step in range(max_len):
                emb = self.embedding(input_token)
                visual_expanded = visual_context.unsqueeze(1)
                lstm_input = torch.cat([emb, visual_expanded], dim=2)
                
                out, hidden = self.lstm(lstm_input, hidden)
                gate = self.liquid_gate(out).clamp(0.03, 0.97)
                out = out * gate
                
                logits = self.output_projection(out.squeeze(1))
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

# =============================================================================
# CORPUS CALLOSUM (añadida control metabólico leve)
# =============================================================================
class CorpusCallosum(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.right_to_left = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, right_features, metabolism=None):
        if metabolism is None:
            return self.right_to_left(right_features)
        # si metabolism es tensor: escalar simple
        return self.right_to_left(right_features) * (1.0 + 0.1 * (metabolism.view(-1,1).detach() if torch.is_tensor(metabolism) else 0.0))

# =============================================================================
# NEUROLOGOS BICAMERAL (mantiene API)
# =============================================================================
class NeuroLogosBicameral(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.right_hemisphere = RightHemisphere(output_dim=512)
        self.left_hemisphere = LeftHemisphere(vocab_size, embed_dim=256, hidden_dim=512)
        self.corpus_callosum = CorpusCallosum(dim=512)
        
    def forward(self, image, captions=None, plasticity=0.1, transfer_rate=0.005, return_diagnostics=False):
        # pass task_loss to liquid via plasticity/prediction_error later
        visual_features = self.right_hemisphere(image, plasticity, transfer_rate, task_loss=0.0)
        # allow corpus to be slightly modulated by last phys if available
        liquid = self.right_hemisphere.spatial_liquid
        metabolism = None
        if getattr(liquid, "_last_phys", None) is not None:
            try:
                metabolism = liquid._last_phys['metabolism'].mean()
            except Exception:
                metabolism = None
        visual_context = self.corpus_callosum(visual_features, metabolism)
        
        if return_diagnostics and captions is not None:
            output, gate = self.left_hemisphere(visual_context, captions, return_gate=True)
            return output, visual_features, visual_context, gate
        
        output = self.left_hemisphere(visual_context, captions)
        return output

# =============================================================================
# DIAGNOSTICO NEUROLOGICO (MEJORADO)
# =============================================================================
class NeuralDiagnostics:
    def __init__(self):
        self.history = {
            'loss': [],
            'right_liquid_norm': [],
            'callosal_flow': [],
            'left_gate_mean': [],
            'left_gate_std': [],
            'vocab_diversity': [],
            'plasticity_effective': [],
            'fatigue': [],
            'damage': []
        }
    
    def measure_callosal_flow(self, right_features, left_context):
        with torch.no_grad():
            right_norm = F.normalize(right_features, dim=-1)
            left_norm = F.normalize(left_context, dim=-1)
            correlation = (right_norm * left_norm).sum(dim=-1).mean()
            return correlation.item()
    
    def measure_vocab_diversity(self, generated_tokens, vocab_size):
        unique_tokens = len(torch.unique(generated_tokens))
        return unique_tokens / vocab_size
    
    def update(self, **metrics):
        for key, value in metrics.items():
            if key in self.history and value is not None:
                self.history[key].append(value)
    
    def get_recent_avg(self, key, n=50):
        if key in self.history and len(self.history[key]) > 0:
            return np.mean(self.history[key][-n:])
        return 0.0
    
    def report(self, epoch):
        if len(self.history['loss']) == 0:
            return
        
        print(f"\n{'='*70}")
        print(f"🧠 DIAGNÓSTICO NEUROLÓGICO - Época {epoch}")
        print(f"{'='*70}")
        
        loss = self.get_recent_avg('loss')
        liquid = self.get_recent_avg('right_liquid_norm')
        flow = self.get_recent_avg('callosal_flow')
        gate_mean = self.get_recent_avg('left_gate_mean')
        gate_std = self.get_recent_avg('left_gate_std')
        vocab_div = self.get_recent_avg('vocab_diversity')
        plast = self.get_recent_avg('plasticity_effective')
        fatigue = self.get_recent_avg('fatigue')
        damage = self.get_recent_avg('damage')
        
        print(f"📉 Loss: {loss:.4f}")
        
        status = "🟢 Estable" if 0.5 < liquid < 2.5 else "🔴 Inestable"
        print(f"👁️  Right Liquid Norm: {liquid:.3f} {status}")
        
        status = "🟢 Fluido" if flow > 0.3 else "🟡 Débil" if flow > 0.1 else "🔴 Bloqueado"
        print(f"🔗 Corpus Callosum Flow: {flow:.3f} {status}")
        
        status = "🟢 Modulando" if 0.3 < gate_mean < 0.7 else "🟡 Sesgado"
        print(f"💬 Left Gate: μ={gate_mean:.3f} σ={gate_std:.3f} {status}")
        
        if vocab_div > 0:
            status = "🟢 Diverso" if vocab_div > 0.1 else "🟡 Limitado" if vocab_div > 0.05 else "🔴 Colapsado"
            print(f"📚 Vocab Diversity: {vocab_div:.3f} {status}")
        
        print(f"🧬 Plasticidad Efectiva: {plast:.4f}")
        print(f"⚡ Fatigue avg: {fatigue:.4f} | Damage avg: {damage:.4f}")
        print(f"{'='*70}\n")

# =============================================================================
# FLICKR8K DATASET
# =============================================================================
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
        img_path, caption = self.data[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        tokens = ['<BOS>'] + caption.lower().split() + ['<EOS>']
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        
        return image, torch.tensor(token_ids, dtype=torch.long)

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

# =============================================================================
# LIFECYCLE
# =============================================================================
class LifeCycle:
    def __init__(self, total_epochs=30):
        self.total = total_epochs
        
    def get_plasticity(self, epoch):
        if epoch < 3:
            return 0.3
        elif epoch < 20:
            return max(0.01, 0.15 * (1 - epoch/20))
        else:
            return 0.001

# =============================================================================
# TRAINING OPTIMIZADO (con fisio + prints + post-step plasticidad/repair)
# =============================================================================
def train_bicameral():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"NeuroLogos Bicameral Optimizado | Device: {device}")
    print(f"{'='*60}\n")
    
    # Setup
    flickr_dir = setup_flickr8k()
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    vocab, id2word = build_vocab_flickr(captions_file, VOCAB_SIZE)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Flickr8kDataset(images_dir, captions_file, vocab, transform, MAX_CAPTION_LEN)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True
    )
    
    model = NeuroLogosBicameral(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    life = LifeCycle(total_epochs=30)
    diagnostics = NeuralDiagnostics()
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}\n")
    
    # Checkpoint directory
    os.makedirs('./checkpoints', exist_ok=True)
    checkpoint_path = './checkpoints/latest_checkpoint.pth'
    
    start_epoch = 0
    
    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        print("📁 Encontrado checkpoint, resumiendo entrenamiento...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False  # <<< CLAVE PARA EVITAR ESTE ERROR
        )

        print("✓ Pesos cargados con compatibilidad fisiológica")
        print("Missing keys (nuevas):", missing)
        print("Unexpected keys (viejas):", unexpected)

        # Optimizer y scheduler solo si existen
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Optimizer y scheduler restaurados")
            except Exception as e:
                print("⚠ No se pudo restaurar optimizer/scheduler:", e)

        start_epoch = checkpoint.get('epoch', -1) + 1
        diagnostics.history = checkpoint.get('diagnostics_history', diagnostics.history)

        print(f"✓ Reanudando desde época {start_epoch}\n")

    
    for epoch in range(start_epoch, 30):
        plasticity = life.get_plasticity(epoch)
        transfer_rate = 0.01 if epoch < 20 else 0.003
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        if epoch % SVD_CONSOLIDATE_INTERVAL == 0 and epoch > 0:
            print("🔧 Consolidando memoria (SVD)...")
            try:
                model.right_hemisphere.spatial_liquid.apply_svd_consolidation(0.7, 'medium')
            except Exception as e:
                print("SVD consolidation failed:", e)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}")
        
        for batch_idx, (images, captions) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward con diagnóstico (pasamos plasticity; task_loss lo asignaremos luego)
            logits, visual_features, visual_context, gate = model(
                images, captions, plasticity, transfer_rate, return_diagnostics=True
            )
            
            loss = F.cross_entropy(
                logits.reshape(-1, len(vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=vocab['<PAD>']
            )
            
            # Update prediction error used by LiquidNeuron
            with torch.no_grad():
                model.right_hemisphere.spatial_liquid.prediction_error = \
                    float((loss / 5.0).clamp(0.0, 0.95).item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # === POST-STEP: PLASTICIDAD HEBBIANA + REPARACIÓN (fisiológica) ===
            with torch.no_grad():
                liquid = model.right_hemisphere.spatial_liquid
                phys = getattr(liquid, "_last_phys", None)
                post = getattr(liquid, "_last_post", None)
                pre = getattr(liquid, "_last_pre", None)
                
                if post is not None and pre is not None:
                    # Hebb-like
                    hebb = torch.mm(post.T, pre) / max(1, pre.size(0))
                    plasticity_scale = float(phys['plasticity'].mean().item()) if phys is not None else 1.0
                    liquid.W_fast += liquid.base_lr * plasticity_scale * torch.tanh(hebb)
                    liquid.W_fast *= 0.999  # slow decay
                    
                    # repair guided by 'repair' phys signal
                    repair_scale = float(phys['repair'].mean().item()) if phys is not None else 0.0
                    liquid.W_fast += repair_scale * 0.001 * torch.randn_like(liquid.W_fast)
                
                # compute diagnostics signals
                liquid_norm = liquid.W_fast.norm().item()
                callosal_flow = diagnostics.measure_callosal_flow(visual_features, visual_context)
                gate_mean = gate.mean().item()
                gate_std = gate.std().item()
                
                # vocab diversity occasionally
                vocab_div = None
                if batch_idx % 50 == 0:
                    sample_gen = model(images[:4], captions=None)
                    vocab_div = diagnostics.measure_vocab_diversity(sample_gen, len(vocab))
                
                # Plasticidad efectiva estimate
                stats = torch.cat([
                    visual_features.mean(dim=1, keepdim=True),
                    visual_features.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
                ], dim=1)
                learned_plast = liquid.plasticity_controller(stats).mean().item()
                effective_plast = plasticity * learned_plast * (1.0 - liquid.prediction_error)
                
                diagnostics.update(
                    loss=loss.item(),
                    right_liquid_norm=liquid_norm,
                    callosal_flow=callosal_flow,
                    left_gate_mean=gate_mean,
                    left_gate_std=gate_std,
                    vocab_diversity=vocab_div,
                    plasticity_effective=effective_plast,
                    fatigue=float(liquid.fatigue.item()),
                    damage=float(liquid.damage.item())
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            # ===================== MONITOREO EN TIEMPO REAL =====================
            if batch_idx % PRINT_EVERY == 0:
                print(f"\n--- Epoch {epoch:02d} Batch {batch_idx:04d} ---")
                print(f"Loss: {loss.item():.4f} | AvgLossSoFar: {total_loss/max(1,num_batches):.4f}")
                print(f"Plasticity(global): {plasticity:.4f} | Learned Plast (est): {learned_plast:.4f}")
                print(f"Prediction Error (liquid): {liquid.prediction_error:.4f}")
                print(f"Liquid W_fast norm: {liquid.W_fast.norm().item():.3f}")
                print(f"Gate μ/σ: {gate_mean:.3f} / {gate_std:.3f}")
                print(f"Callosal flow: {callosal_flow:.4f}")
                if vocab_div is not None:
                    print(f"Vocab diversity (sample): {vocab_div:.4f}")
                print(f"Fatigue: {liquid.fatigue.item():.4f} | Damage: {liquid.damage.item():.4f}")
                # Alerts
                if liquid.W_fast.norm().item() > 200.0:
                    print("🚨 ALERT: Liquid W_fast norm VERY HIGH (possible instability)")
                if gate_mean < 0.02:
                    print("🚨 ALERT: Gate collapsed to near-zero")
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'liquid': f'{liquid.W_fast.norm().item():.2f}',
                'flow': f'{callosal_flow:.2f}',
                'gate': f'{gate_mean:.2f}'
            })
        
        scheduler.step()
        
        # Reporte de época
        diagnostics.report(epoch)
        
        # Generación de muestras
        if epoch % 2 == 0:
            model.eval()
            print("\n📸 GENERANDO MUESTRAS...\n")
            
            with torch.no_grad():
                for sample_idx in range(3):
                    sample_img, sample_cap = dataset[sample_idx * 100]
                    sample_img = sample_img.unsqueeze(0).to(device)
                    
                    generated = model(sample_img, captions=None)
                    
                    words = [id2word.get(int(t.item()), '<UNK>') for t in generated[0]]
                    sentence = " ".join(w for w in words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                    
                    gt_words = [id2word.get(int(t.item()), '<UNK>') for t in sample_cap]
                    gt_sentence = " ".join(w for w in gt_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                    
                    print(f"Muestra {sample_idx + 1}:")
                    print(f"  GT : {gt_sentence}")
                    print(f"  Gen: {sentence}\n")
            
            model.train()
        
        avg_loss_epoch = total_loss/num_batches if num_batches>0 else 0.0
        print(f"Época {epoch:02d} completada | Avg Loss: {avg_loss_epoch:.4f}\n")
        
        # Guardar checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'vocab': vocab,
            'id2word': id2word,
            'diagnostics_history': diagnostics.history,
            'loss': avg_loss_epoch
        }, checkpoint_path)
        
        # Guardar checkpoint periódico
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'id2word': id2word
            }, f'./checkpoints/epoch_{epoch:02d}.pth')
    
    print("✅ Entrenamiento completado!")

if __name__ == "__main__":
    train_bicameral()

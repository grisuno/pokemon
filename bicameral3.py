#%%writefile neurosoberano_bicameral_optimized.py
# =============================================================================
# NeuroLogos Bicameral Minimalista v1.1 - OPTIMIZADO
# Mejoras: DataLoader rápido, checkpointing, diagnóstico completo
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

# =============================================================================
# DESCARGA AUTOMÁTICA DE FLICKR8K
# =============================================================================
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
    
    # Procesar captions
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
    
    # Verificar estructura
    if os.path.exists(os.path.join(flickr_dir, 'Flicker8k_Dataset')):
        import shutil
        old_dir = os.path.join(flickr_dir, 'Flicker8k_Dataset')
        if not os.path.exists(images_dir):
            shutil.move(old_dir, images_dir)
    
    print("✅ Flickr8k listo\n")
    return flickr_dir

# =============================================================================
# LIQUID NEURON (sin cambios)
# =============================================================================
class LiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        
        self.register_buffer('W_medium', torch.zeros(out_dim, in_dim))
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        
        self.ln = nn.LayerNorm(out_dim)
        
        self.plasticity_controller = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.plasticity_controller[2].bias.data.fill_(-2.0)
        self.base_lr = 0.015
        self.prediction_error = 0.0
        
    def forward(self, x, global_plasticity=0.0, transfer_rate=0.005):
        slow_out = self.W_slow(x)
        medium_out = F.linear(x, self.W_medium)
        fast_out = F.linear(x, self.W_fast)
        pre_act = slow_out + medium_out + fast_out
        
        batch_mean = pre_act.mean(dim=1, keepdim=True)
        batch_std = pre_act.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
        stats = torch.cat([batch_mean, batch_std], dim=1)
        
        learned_plasticity = self.plasticity_controller(stats).squeeze(1)
        effective_plasticity = global_plasticity * learned_plasticity * (1.0 - self.prediction_error)
        
        out = 5.0 * torch.tanh(self.ln(pre_act) / 5.0)
        
        if self.training and effective_plasticity.mean() > 0.001:
            with torch.no_grad():
                out_centered = out - out.mean(dim=0, keepdim=True)
                correlation = torch.mm(out_centered.T, x) / (x.size(0) + 1e-6)
                
                self.W_medium.data += self.W_fast.data * transfer_rate
                
                lr_vector = effective_plasticity.mean() * self.base_lr
                self.W_fast.data += correlation * lr_vector
                self.W_fast.data.mul_(1.0 - transfer_rate)
                self.W_fast.data.clamp_(-3.0, 3.0)
                
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
                threshold = S.max() * 0.01 * repair_strength
                mask = S > threshold
                filtered_S = S * mask.float()
                W_consolidated = U @ torch.diag(filtered_S) @ Vt
                
                W_target.data = (1.0 - repair_strength) * W_target.data + repair_strength * W_consolidated
                W_target.data *= 0.98
                return True
            except:
                W_target.data.mul_(0.9)
                return False

# =============================================================================
# HEMISFERIO DERECHO
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
        
    def forward(self, image, plasticity=0.1, transfer_rate=0.005):
        features = self.visual_encoder(image)
        features = features.flatten(1)
        visual_thought = self.spatial_liquid(features, plasticity, transfer_rate)
        return visual_thought

# =============================================================================
# HEMISFERIO IZQUIERDO
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
                gate = self.liquid_gate(out)
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
# CORPUS CALLOSUM
# =============================================================================
class CorpusCallosum(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.right_to_left = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, right_features):
        return self.right_to_left(right_features)

# =============================================================================
# NEUROLOGOS BICAMERAL
# =============================================================================
class NeuroLogosBicameral(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.right_hemisphere = RightHemisphere(output_dim=512)
        self.left_hemisphere = LeftHemisphere(vocab_size, embed_dim=256, hidden_dim=512)
        self.corpus_callosum = CorpusCallosum(dim=512)
        
    def forward(self, image, captions=None, plasticity=0.1, transfer_rate=0.005, return_diagnostics=False):
        visual_features = self.right_hemisphere(image, plasticity, transfer_rate)
        visual_context = self.corpus_callosum(visual_features)
        
        if return_diagnostics and captions is not None:
            output, gate = self.left_hemisphere(visual_context, captions, return_gate=True)
            return output, visual_features, visual_context, gate
        
        output = self.left_hemisphere(visual_context, captions)
        return output

# =============================================================================
# DIAGNOSTICO NEUROLOGICO
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
            'plasticity_effective': []
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
# TRAINING OPTIMIZADO
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
        persistent_workers=True  # Importante: mantiene workers vivos
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
    
    start_epoch = 0
    
    # Resume from checkpoint if exists
    checkpoint_path = './checkpoints/latest_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print("📁 Encontrado checkpoint, resumiendo entrenamiento...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        diagnostics.history = checkpoint['diagnostics_history']
        print(f"✓ Resumido desde época {start_epoch}\n")
    
    for epoch in range(start_epoch, 30):
        plasticity = life.get_plasticity(epoch)
        transfer_rate = 0.01 if epoch < 20 else 0.003
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        if epoch % 10 == 0 and epoch > 0:
            print("🔧 Consolidando memoria (SVD)...")
            model.right_hemisphere.spatial_liquid.consolidate_svd(0.7, 'medium')
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}")
        
        for batch_idx, (images, captions) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward con diagnóstico
            logits, visual_features, visual_context, gate = model(
                images, captions, plasticity, transfer_rate, return_diagnostics=True
            )
            
            loss = F.cross_entropy(
                logits.reshape(-1, len(vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=vocab['<PAD>']
            )
            
            with torch.no_grad():
                model.right_hemisphere.spatial_liquid.prediction_error = \
                    (loss / 5.0).clamp(0.0, 0.95).item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Diagnóstico
            with torch.no_grad():
                liquid_norm = model.right_hemisphere.spatial_liquid.W_fast.norm().item()
                callosal_flow = diagnostics.measure_callosal_flow(visual_features, visual_context)
                gate_mean = gate.mean().item()
                gate_std = gate.std().item()
                
                # Vocab diversity cada 50 batches
                vocab_div = None
                if batch_idx % 50 == 0:
                    sample_gen = model(images[:4], captions=None)
                    vocab_div = diagnostics.measure_vocab_diversity(sample_gen, len(vocab))
                
                # Plasticidad efectiva
                liquid = model.right_hemisphere.spatial_liquid
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
                    plasticity_effective=effective_plast
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'liquid': f'{liquid_norm:.2f}',
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
        
        print(f"Época {epoch:02d} completada | Avg Loss: {total_loss/num_batches:.4f}\n")
        
        # Guardar checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'vocab': vocab,
            'id2word': id2word,
            'diagnostics_history': diagnostics.history,
            'loss': total_loss/num_batches
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

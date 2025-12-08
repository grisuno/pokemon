# =============================================================================
# NeuroLogos Bicameral Minimalista Ultra-Ligero v1.0
# Optimizado para T4 / CPU | <15k parámetros | Diagnóstico democrático conservado
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES
# =============================================================================
IMG_SIZE = 96  # reducido para velocidad
MAX_CAPTION_LEN = 20
VOCAB_SIZE = 3000  # más pequeño → más rápido
EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
NUM_WORKERS = 2

# =============================================================================
# DESCARGA FLICKR8K (sin cambios)
# =============================================================================
def setup_flickr8k(data_dir='./data'):
    flickr_dir = os.path.join(data_dir, 'flickr8k')
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    if os.path.exists(images_dir) and os.path.exists(captions_file):
        print("✓ Flickr8k ya existe, saltando descarga...\n")
        return flickr_dir
    os.makedirs(flickr_dir, exist_ok=True)
    print("📥 Descargando Flickr8k desde GitHub...")
    import urllib.request, zipfile
    urls = {
        'images': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
        'captions': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
    }
    for name, url in urls.items():
        zip_path = os.path.join(flickr_dir, f'{name}.zip')
        print(f"📥 Descargando {name}...")
        urllib.request.urlretrieve(url.strip(), zip_path)
        print(f"📂 Extrayendo {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(flickr_dir)
        os.remove(zip_path)
    # Procesar captions
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
    # Mover directorio si es necesario
    if os.path.exists(os.path.join(flickr_dir, 'Flicker8k_Dataset')):
        import shutil
        old_dir = os.path.join(flickr_dir, 'Flicker8k_Dataset')
        if not os.path.exists(images_dir):
            shutil.move(old_dir, images_dir)
    print("✅ Flickr8k listo\n")
    return flickr_dir

# =============================================================================
# ENCODER VISUAL LIGERO (~5k params)
# =============================================================================
class TinyVisualEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),   # 48x48
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 24x24
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 12x12
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),               # 4x4
            nn.Flatten(),
            nn.Linear(64 * 16, output_dim),
            nn.LayerNorm(output_dim)
        )
    def forward(self, x):
        return self.net(x)

# =============================================================================
# NEURONA LÍQUIDA MÍNIMA (fisiología predictiva simplificada)
# =============================================================================
class MinimalLiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, out_dim)
        self.predictor = nn.Linear(out_dim, in_dim)
        self.register_buffer('running_error', torch.tensor(1.0))
        self.gate = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid())
        self.gate[2].bias.data.fill_(-1.0)

    def forward(self, x):
        z = self.encoder(x)
        x_pred = self.predictor(z)
        pred_error = F.mse_loss(x_pred, x.detach(), reduction='none').mean(1, keepdim=True)
        self.running_error = 0.99 * self.running_error + 0.01 * pred_error.mean()
        surprise = pred_error / (self.running_error + 1e-6)
        g = self.gate(surprise)
        out = torch.tanh(z) * g
        return out, surprise.mean().item()

# =============================================================================
# HEMISFERIOS
# =============================================================================
class RightHemisphere(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.encoder = TinyVisualEncoder(output_dim)
        self.liquid = MinimalLiquidNeuron(output_dim, output_dim)
    def forward(self, x):
        features = self.encoder(x)
        out, surprise = self.liquid(features)
        return out, surprise

class LeftHemisphere(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.proj = nn.Linear(hidden_dim, vocab_size)
    def forward(self, visual_ctx, captions=None, max_len=20):
        B = visual_ctx.size(0)
        device = visual_ctx.device
        if captions is not None:
            emb = self.embedding(captions[:, :-1])
            T = emb.size(1)
            vis = visual_ctx.unsqueeze(1).expand(-1, T, -1)
            inp = torch.cat([emb, vis], dim=2)
            out, _ = self.lstm(inp)
            g = self.gate(out)
            out = out * g
            return self.proj(out), g
        else:
            tokens = []
            h = visual_ctx.unsqueeze(0), torch.zeros_like(visual_ctx).unsqueeze(0)
            inp_tok = torch.full((B, 1), 1, device=device, dtype=torch.long)
            for _ in range(max_len):
                emb = self.embedding(inp_tok)
                vis = visual_ctx.unsqueeze(1)
                lstm_in = torch.cat([emb, vis], dim=2)
                out, h = self.lstm(lstm_in, h)
                g = self.gate(out)
                logits = self.proj(out.squeeze(1)) * g.squeeze(1)
                probs = F.softmax(logits / 0.9, dim=-1)
                inp_tok = torch.multinomial(probs, 1)
                tokens.append(inp_tok)
                if (inp_tok == 2).all():
                    break
            return torch.cat(tokens, dim=1)

# =============================================================================
# CORPUS CALLOSUM
# =============================================================================
class CorpusCallosum(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        return self.proj(x)

# =============================================================================
# MODELO BICAMERAL
# =============================================================================
class NeuroLogosBicameralUltra(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.right = RightHemisphere(256)
        self.callosum = CorpusCallosum(256)
        self.left = LeftHemisphere(vocab_size, 128, 256)
    def forward(self, image, captions=None, return_diagnostics=False):
        right_out, surprise = self.right(image)
        callosal = self.callosum(right_out)
        if captions is not None:
            logits, gate = self.left(callosal, captions)
            if return_diagnostics:
                return logits, right_out, callosal, gate, surprise
            return logits
        else:
            return self.left(callosal)

# =============================================================================
# DIAGNÓSTICO DEMOCRÁTICO MÍNIMO
# =============================================================================
class DemocraticDiagnostics:
    def __init__(self):
        self.history = {
            'loss': [], 'surprise': [], 'callosal_flow': [],
            'right_norm': [], 'left_gate': [], 'vocab_div': []
        }
    def measure_flow(self, r, l):
        with torch.no_grad():
            r = F.normalize(r, dim=-1)
            l = F.normalize(l, dim=-1)
            return (r * l).sum(-1).mean().item()
    def vocab_diversity(self, tokens, V):
        return len(torch.unique(tokens)) / V
    def update(self, **kw):
        for k, v in kw.items():
            if k in self.history and v is not None:
                self.history[k].append(v)
    def avg(self, k, n=30):
        arr = self.history[k]
        return np.mean(arr[-n:]) if arr else 0.0
    def report(self, epoch):
        print(f"\n{'='*60}")
        print(f"🏛️  DIAGNÓSTICO DEMOCRÁTICO - Época {epoch}")
        print(f"{'='*60}")
        loss = self.avg('loss')
        flow = self.avg('callosal_flow')
        right = self.avg('right_norm')
        gate = self.avg('left_gate')
        vocab = self.avg('vocab_div')
        print(f"📉 Loss: {loss:.4f} | 🧠 Surprise: {self.avg('surprise'):.3f}")
        print(f"👁️  Right Norm: {right:.2f} ({'🟢' if 0.5 < right < 2.0 else '🔴'})")
        print(f"🔗 Callosal Flow: {flow:.3f} ({'🟢' if flow > 0.2 else '🔴'})")
        print(f"💬 Left Gate: {gate:.3f} ({'🟢' if 0.3 < gate < 0.8 else '🟡'})")
        if vocab > 0:
            print(f"📚 Vocab Diversity: {vocab:.3f} ({'🟢' if vocab > 0.08 else '🔴'})")
        print(f"{'='*60}\n")

# =============================================================================
# DATASET (sin cambios esenciales)
# =============================================================================
class Flickr8kDataset(Dataset):
    def __init__(self, images_dir, captions_file, vocab, transform=None, max_len=20):
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
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        tokens = ['<BOS>'] + caption.lower().split() + ['<EOS>']
        ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        if len(ids) < self.max_len:
            ids += [self.vocab['<PAD>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return image, torch.tensor(ids, dtype=torch.long)

def build_vocab(captions_file, size=3000):
    counter = Counter()
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                counter.update(parts[1].lower().split())
    common = counter.most_common(size - 4)
    vocab = {'<PAD>':0, '<BOS>':1, '<EOS>':2, '<UNK>':3}
    for i, (w, _) in enumerate(common):
        vocab[w] = i + 4
    return vocab, {i:w for w,i in vocab.items()}

# =============================================================================
# ENTRENAMIENTO
# =============================================================================
def train_ultra():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧠 Entrenando en {device} | Objetivo: <15k parámetros\n")
    
    flickr = setup_flickr8k()
    vocab, id2word = build_vocab(os.path.join(flickr, 'captions.txt'), VOCAB_SIZE)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    dataset = Flickr8kDataset(os.path.join(flickr, 'Images'), os.path.join(flickr, 'captions.txt'), vocab, transform, MAX_CAPTION_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = NeuroLogosBicameralUltra(len(vocab)).to(device)
    print(f"🔢 Parámetros totales: {sum(p.numel() for p in model.parameters()):,}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    diag = DemocraticDiagnostics()
    
    os.makedirs('./checkpoints', exist_ok=True)
    
    for epoch in range(25):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        
        for i, (imgs, caps) in enumerate(pbar):
            imgs, caps = imgs.to(device, non_blocking=True), caps.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            logits, r_feat, l_ctx, gate, surprise = model(imgs, caps, return_diagnostics=True)
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), caps[:,1:].reshape(-1), ignore_index=vocab['<PAD>'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            with torch.no_grad():
                flow = diag.measure_flow(r_feat, l_ctx)
                vocab_div = diag.vocab_diversity(model(imgs[:4], captions=None), len(vocab)) if i % 40 == 0 else None
                diag.update(
                    loss=loss.item(),
                    surprise=surprise,
                    callosal_flow=flow,
                    right_norm=r_feat.norm(dim=-1).mean().item(),
                    left_gate=gate.mean().item(),
                    vocab_div=vocab_div
                )
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), flow=flow)
        
        scheduler.step()
        diag.report(epoch)
        
        # Muestra cada 5 épocas
        if epoch % 5 == 0:
            model.eval()
            print("\n📸 Muestras de generación:")
            with torch.no_grad():
                for j in range(2):
                    img, gt = dataset[j*200]
                    out = model(img.unsqueeze(0).to(device), captions=None)
                    pred = " ".join(id2word.get(t.item(), '<UNK>') for t in out[0] if t.item() not in [0,1,2])
                    gt_txt = " ".join(id2word.get(t.item(), '<UNK>') for t in gt if t.item() not in [0,1,2])
                    print(f"  GT : {gt_txt}")
                    print(f"  Gen: {pred}\n")
            model.train()
        
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'vocab': vocab, 'id2word': id2word},
                   './checkpoints/ultra_latest.pth')
    
    print("✅ Entrenamiento ultra-ligero completado.")

if __name__ == "__main__":
    train_ultra()
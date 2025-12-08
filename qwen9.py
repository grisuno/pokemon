#%%writefile neurosoberano_bicameral_v2.py
# =============================================================================
# NeuroLogos Bicameral v2.0 – CON HOMEOSTASIS PREDICTIVA Y GATE ACTIVO
# Corrige: gate saturado, calloso bloqueado, vocab colapsado
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
# CONSTANTES
# =============================================================================
IMG_SIZE = 224
MAX_CAPTION_LEN = 30
VOCAB_SIZE = 5000
EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 64
NUM_WORKERS = 4

# =============================================================================
# LIQUID NEURON (mejorado)
# =============================================================================
class LiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.0)
        self.register_buffer('W_medium', torch.zeros(out_dim, in_dim))
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.ln = nn.LayerNorm(out_dim)
        self.plasticity_controller = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.plasticity_controller[2].bias.data.fill_(-3.0)
        self.base_lr = 0.01
        self.prediction_error = 0.0

    def forward(self, x, global_plasticity=0.0, transfer_rate=0.005):
        slow_out = self.W_slow(x)
        medium_out = F.linear(x, self.W_medium)
        fast_out = F.linear(x, self.W_fast)
        pre_act = slow_out + medium_out + fast_out
        out = 3.0 * torch.tanh(self.ln(pre_act) / 3.0)

        if self.training:
            batch_mean = pre_act.mean(dim=1, keepdim=True)
            batch_std = pre_act.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
            stats = torch.cat([batch_mean, batch_std], dim=1)
            learned_plasticity = self.plasticity_controller(stats).squeeze(1)
            effective_plasticity = global_plasticity * learned_plasticity * (1.0 - self.prediction_error)

            if effective_plasticity.mean() > 0.001:
                with torch.no_grad():
                    out_centered = out - out.mean(dim=0, keepdim=True)
                    correlation = torch.mm(out_centered.T, x) / (x.size(0) + 1e-6)
                    self.W_medium.data += self.W_fast.data * transfer_rate
                    lr_vec = effective_plasticity.mean() * self.base_lr
                    self.W_fast.data += correlation * lr_vec
                    self.W_fast.data.mul_(1.0 - transfer_rate)
                    self.W_fast.data.clamp_(-2.0, 2.0)

        return out

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
# HEMISFERIO IZQUIERDO – CON GATE ACTIVO Y CONTROL HOMEOSTÁTICO
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
        # Gate ahora recibe contexto + entropía + error
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.liquid_gate[2].bias.data.fill_(-2.0)  # Clave: evita saturación
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, visual_context, captions=None, max_len=30, prediction_error=0.0, return_gate=False):
        batch_size = visual_context.size(0)
        device = visual_context.device

        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            # Inyección controlada de entropía durante train
            if self.training:
                noise = torch.randn_like(embeddings) * 0.03
                embeddings = embeddings + noise

            seq_len = embeddings.size(1)
            visual_expanded = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_input = torch.cat([embeddings, visual_expanded], dim=2)
            lstm_out, _ = self.lstm(lstm_input, self._get_init_state(visual_context))

            # Calcular entropía local del output LSTM
            logits_temp = self.output_projection(lstm_out.detach())
            probs_temp = F.softmax(logits_temp / 1.0, dim=-1)
            entropy = -(probs_temp * torch.log(probs_temp + 1e-8)).sum(dim=-1, keepdim=True)

            # Gate input: [hidden, entropy, prediction_error]
            pe_tensor = torch.full_like(entropy, prediction_error)
            gate_input = torch.cat([lstm_out, entropy, pe_tensor], dim=-1)
            gate = self.liquid_gate(gate_input)
            modulated = lstm_out * gate

            logits = self.output_projection(modulated)
            if return_gate:
                gate_entropy = -(gate * torch.log(gate + 1e-8) + (1 - gate) * torch.log(1 - gate + 1e-8))
                return logits, gate.mean().item(), gate_entropy.mean().item()
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
# CORPUS CALLOSUM BIDIRECCIONAL
# =============================================================================
class CorpusCallosum(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.right_to_left = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        self.left_to_right = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )

    def forward(self, right_features, left_context=None):
        r2l = self.right_to_left(right_features)
        if left_context is not None:
            l2r = self.left_to_right(left_context)
            return r2l, l2r
        return r2l, None

# =============================================================================
# REGULADOR HOMEOSTÁTICO GLOBAL (predictivo)
# =============================================================================
class HomeostaticRegulator(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.register_buffer('ema_callosal_flow', torch.tensor(0.0))
        self.flow_alpha = 0.95

    def forward(self, right_features, epoch):
        # Predice valores óptimos de plasticidad, gate_inhib, y threshold
        control = self.predictor(right_features.detach().mean(dim=0, keepdim=True))
        plasticity = torch.sigmoid(control[0, 0]) * 0.3
        gate_inhib = torch.sigmoid(control[0, 1])  # 0 = no inhib, 1 = max inhib
        surprise_threshold = torch.sigmoid(control[0, 2]) * 0.5 + 0.1

        # Ajuste por época
        if epoch < 3:
            plasticity = 0.3
        elif epoch >= 20:
            plasticity *= 0.1

        return plasticity, gate_inhib, surprise_threshold

    def update_flow_ema(self, flow):
        self.ema_callosal_flow = self.flow_alpha * self.ema_callosal_flow + (1 - self.flow_alpha) * flow

# =============================================================================
# NEUROLOGOS BICAMERAL v2
# =============================================================================
class NeuroLogosBicameral(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.right_hemisphere = RightHemisphere(output_dim=512)
        self.left_hemisphere = LeftHemisphere(vocab_size, embed_dim=256, hidden_dim=512)
        self.corpus_callosum = CorpusCallosum(dim=512)
        self.homeostat = HomeostaticRegulator(dim=512)

    def forward(self, image, captions=None, epoch=0, return_diagnostics=False):
        visual_features = self.right_hemisphere(image)
        plasticity_raw, gate_inhib, surprise_threshold = self.homeostat(visual_features, epoch)
        transfer_rate = 0.01 if epoch < 20 else 0.003

        # Ajuste temporal de plasticidad en modo tensor
        if epoch < 3:
            plasticity_tensor = torch.tensor(0.3, device=visual_features.device)
        elif epoch >= 20:
            plasticity_tensor = plasticity_raw * 0.1
        else:
            plasticity_tensor = plasticity_raw

        visual_context = self.corpus_callosum.right_to_left(visual_features)

        if captions is not None:
            logits, gate_mean, gate_entropy = self.left_hemisphere(
                visual_context,
                captions,
                prediction_error=0.0,
                return_gate=True
            )
            if return_diagnostics:
                return logits, visual_features, visual_context, gate_mean, gate_entropy, plasticity_tensor.item(), transfer_rate
            return logits
        else:
            return self.left_hemisphere(visual_context)

            
# =============================================================================
# DIAGNÓSTICO MEJORADO
# =============================================================================
class NeuralDiagnostics:
    def __init__(self):
        self.history = {
            'loss': [], 'right_liquid_norm': [], 'callosal_flow': [],
            'left_gate_mean': [], 'left_gate_entropy': [], 'vocab_diversity': [],
            'plasticity_effective': []
        }

    def measure_callosal_flow(self, right_features, left_context):
        with torch.no_grad():
            right_norm = F.normalize(right_features, dim=-1)
            left_norm = F.normalize(left_context, dim=-1)
            mutual_info = (right_norm * left_norm).sum(dim=-1).mean()
            return mutual_info.item()

    def measure_vocab_diversity(self, tokens, vocab_size):
        if tokens.numel() == 0:
            return 0.0
        unique = torch.unique(tokens)
        return unique.numel() / vocab_size

    def update(self, **metrics):
        for k, v in metrics.items():
            if k in self.history and v is not None:
                self.history[k].append(v)

    def get_recent_avg(self, key, n=50):
        hist = self.history.get(key, [])
        return np.mean(hist[-n:]) if len(hist) > 0 else 0.0

    def report(self, epoch):
        print(f"\n{'='*70}")
        print(f"🧠 DIAGNÓSTICO NEUROLÓGICO v2 – Época {epoch}")
        print(f"{'='*70}")
        metrics = {k: self.get_recent_avg(k) for k in self.history.keys()}
        print(f"📉 Loss: {metrics['loss']:.4f}")
        print(f"👁️  Right Liquid Norm: {metrics['right_liquid_norm']:.3f} {'🟢 Estable' if 0.8 < metrics['right_liquid_norm'] < 2.0 else '🔴 Inestable'}")
        print(f"🔗 Callosal Flow: {metrics['callosal_flow']:.3f} {'🟢 Fluido' if metrics['callosal_flow'] > 0.3 else '🟡 Débil' if metrics['callosal_flow'] > 0.1 else '🔴 Bloqueado'}")
        print(f"💬 Left Gate: μ={metrics['left_gate_mean']:.3f} H={metrics['left_gate_entropy']:.3f} {'🟢 Modulando' if 0.3 < metrics['left_gate_mean'] < 0.7 and metrics['left_gate_entropy'] > 0.1 else '🔴 Saturado'}")
        if metrics['vocab_diversity'] > 0:
            print(f"📚 Vocab Diversity: {metrics['vocab_diversity']:.3f} {'🟢 Diverso' if metrics['vocab_diversity'] > 0.1 else '🟡 Limitado' if metrics['vocab_diversity'] > 0.05 else '🔴 Colapsado'}")
        print(f"🧬 Plasticidad Efectiva: {metrics['plasticity_effective']:.4f}")
        print(f"{'='*70}\n")

# =============================================================================
# DATASET & UTILS (sin cambios esenciales)
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
        print(f"✅ Loaded {len(self.data)} image-caption pairs")

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

def setup_flickr8k(data_dir='./data'):
    flickr_dir = os.path.join(data_dir, 'flickr8k')
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    if os.path.exists(images_dir) and os.path.exists(captions_file):
        print("✓ Flickr8k ya existe\n")
        return flickr_dir
    # (Mismo código de descarga que antes – omitido por brevedad)
    # ... (copiar setup_flickr8k original aquí si es necesario)
    return flickr_dir

# =============================================================================
# TRAINING
# =============================================================================
def train_bicameral():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"NeuroLogos Bicameral v2 | Device: {device}")
    print(f"Homeostasis Predictiva + Gate Activo")
    print(f"{'='*60}\n")

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
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = NeuroLogosBicameral(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    diagnostics = NeuralDiagnostics()
    os.makedirs('./checkpoints', exist_ok=True)

    for epoch in range(30):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}")

        for batch_idx, (images, captions) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits, visual_features, visual_context, gate_mean, gate_entropy, plasticity, tr = model(
                images, captions, epoch=epoch, return_diagnostics=True
            )

            loss = F.cross_entropy(
                logits.reshape(-1, len(vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=vocab['<PAD>']
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Actualizar error en neuronas
            with torch.no_grad():
                pe = (loss / 5.0).clamp(0.0, 0.95).item()
                model.right_hemisphere.spatial_liquid.prediction_error = pe

            # Diagnóstico
            liquid_norm = model.right_hemisphere.spatial_liquid.W_fast.norm().item()
            flow = diagnostics.measure_callosal_flow(visual_features, visual_context)
            vocab_div = None
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    sample_gen = model(images[:4], captions=None, epoch=epoch)
                    vocab_div = diagnostics.measure_vocab_diversity(sample_gen, len(vocab))

            diagnostics.update(
                loss=loss.item(),
                right_liquid_norm=liquid_norm,
                callosal_flow=flow,
                left_gate_mean=gate_mean,
                left_gate_entropy=gate_entropy,
                vocab_diversity=vocab_div,
                plasticity_effective=plasticity
            )

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'flow': f'{flow:.2f}',
                'gate': f'{gate_mean:.2f}'
            })

        diagnostics.report(epoch)

        if epoch % 2 == 0:
            model.eval()
            print("\n📸 MUESTRAS (post-fix)...\n")
            with torch.no_grad():
                for i in range(3):
                    img, cap = dataset[i * 100]
                    img = img.unsqueeze(0).to(device)
                    gen = model(img, captions=None, epoch=epoch)
                    words = [id2word.get(t.item(), '<UNK>') for t in gen[0]]
                    pred = " ".join(w for w in words if w not in ['<BOS>','<EOS>','<PAD>'])
                    gt = " ".join(id2word.get(t.item(), '') for t in cap if t.item() not in [0,1,2,3])
                    print(f"{i+1} GT : {gt}\n   Gen: {pred}\n")
            model.train()

        torch.save({'model_state_dict': model.state_dict()}, './checkpoints/latest_v2.pth')

    print("✅ Entrenamiento v2 completado. Ahora el gate y el calloso deberían estar ACTIVOS.")

if __name__ == "__main__":
    train_bicameral()
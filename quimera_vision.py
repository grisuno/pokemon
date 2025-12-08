# =============================================================================
# flickr8k_multimodal_minimal.py
# CPU-friendly multimodal caption generator
# =============================================================================
import os, json, random, torch, torch.nn as nn
import torchaudio, torchvision, tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
torch.manual_seed(42)
device = torch.device("cpu")

# ------------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------------
BASE_DIR   = Path("flickr8k_full")
IMG_DIR    = BASE_DIR / "Images"
AUDIO_DIR  = BASE_DIR / "Audio_es"
CAP_FILE   = BASE_DIR / "captions_es.txt"
MAX_LEN    = 20          # tokens por caption
BATCH_SIZE = 32
EPOCHS     = 1           # subir a 10-20 después
LR         = 1e-3
EMB        = 256
HIDDEN     = 256

# ------------------------------------------------------------------
# 2. VOCABULARIO SIMPLE
# ------------------------------------------------------------------
with open(CAP_FILE, encoding="utf-8") as f:
    caps = [l.strip().split("\t")[1] for l in f if "\t" in l]
vocab = ["<pad>", "<start>", "<end>", "<unk>"] + sorted(set(" ".join(caps).lower().split()))
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}
vocab_size = len(vocab)

def text_to_seq(text):
    return [word2idx.get(w, word2idx["<unk>"]) for w in text.lower().split()]

# ------------------------------------------------------------------
# 3. DATASET
# ------------------------------------------------------------------
class Flickr8kMMDataset(Dataset):
    def __init__(self):
        self.samples = []
        with open(CAP_FILE, encoding="utf-8") as f:
            for line in f:
                if "\t" not in line: continue
                img, cap = line.strip().split("\t", 1)
                self.samples.append((img, cap))
        random.shuffle(self.samples)

        self.img_tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485,0.456,0.406],
                                             [0.229,0.224,0.225])
        ])
        self.audio_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, n_mels=64)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]

        # FIX: cargar imagen vía PIL para evitar doble tensorización
        img = Image.open(IMG_DIR/img_name).convert("RGB")
        img = self.img_tf(img)

        # audio
        audio_path = AUDIO_DIR / (Path(img_name).stem + "_0.mp3")
        if not audio_path.exists():
            audio_path = AUDIO_DIR / (Path(img_name).stem + "_1.mp3")
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        spec = self.audio_tf(wav).squeeze()[:,:128]          # (64,128)

        # texto
        seq = [word2idx["<start>"]] + text_to_seq(caption)[:MAX_LEN-2] + [word2idx["<end>"]]
        seq += [word2idx["<pad>"]] * (MAX_LEN - len(seq))
        seq = torch.tensor(seq, dtype=torch.long)

        return img, spec, seq

def collate(batch):
    imgs, specs, seqs = zip(*batch)
    return (torch.stack(imgs),
            torch.stack(specs).unsqueeze(1),  # (B,1,64,128)
            torch.stack(seqs))

# ------------------------------------------------------------------
# 4. MODELOS LIGEROS
# ------------------------------------------------------------------
class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.feat = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(576, EMB)
    def forward(self, x):
        x = self.feat(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)  # (B,EMB)

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32,64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(64, EMB)
    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB)
        self.gru = nn.GRU(EMB, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, vocab_size)
    def forward(self, img, audio, seq):
        # fusion simple: suma
        ctx = img + audio
        emb = self.emb(seq[:,:-1]) + ctx.unsqueeze(1)  # teacher forcing
        out,_ = self.gru(emb)
        logits = self.out(out)
        return logits  # (B,L,vocab)

# ------------------------------------------------------------------
# 5. ENTRENAMIENTO RÁPIDO
# ------------------------------------------------------------------
dataset = Flickr8kMMDataset()
loader  = DataLoader(dataset, BATCH_SIZE, shuffle=True, collate_fn=collate,
                     num_workers=0, pin_memory=False)

img_enc  = ImgEncoder().to(device)
aud_enc  = AudioEncoder().to(device)
decoder  = Decoder().to(device)

opt = torch.optim.Adam(list(img_enc.parameters()) +
                       list(aud_enc.parameters()) +
                       list(decoder.parameters()), lr=LR)
ce  = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])

for epoch in range(EPOCHS):
    bar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}")
    for img, aud, seq in bar:
        img, aud, seq = img.to(device), aud.to(device), seq.to(device)
        
        opt.zero_grad()
        logits = decoder(img_enc(img), aud_enc(aud), seq)
        loss = ce(logits.reshape(-1, vocab_size), seq[:,1:].reshape(-1))
        loss.backward()
        opt.step()
        
        bar.set_postfix(loss=loss.item())

# ------------------------------------------------------------------
# 6. INFERENCIA
# ------------------------------------------------------------------
@torch.no_grad()
def generate_caption(img_path, audio_path):
    img = dataset.img_tf(torchvision.io.read_image(img_path)/255.0).unsqueeze(0).to(device)
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    spec = dataset.audio_tf(wav).unsqueeze(0).to(device)
    
    img_feat = img_enc(img)
    aud_feat = aud_enc(spec)
    ctx = img_feat + aud_feat
    
    tok = [word2idx["<start>"]]
    for _ in range(MAX_LEN):
        inp = torch.tensor([tok], dtype=torch.long).to(device)
        emb = dataset.emb(inp) + ctx.unsqueeze(1)
        out,_ = dataset.gru(emb)
        logits = dataset.out(out[0,-1])
        nxt = logits.argmax().item()
        if nxt == word2idx["<end>"]: break
        tok.append(nxt)
    return " ".join(idx2word[i] for i in tok[1:])

# quick test
if __name__ == "__main__":
    sample_img  = str(next(IMG_DIR.glob("*.jpg")))
    sample_aud  = str(AUDIO_DIR / (Path(sample_img).stem + "_0.mp3"))
    print("Generated:", generate_caption(sample_img, sample_aud))

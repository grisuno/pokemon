import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Configuración reproducible y GPU
# ----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise RuntimeError("Este script requiere GPU. No se detectó CUDA.")

print(f"✅ Dispositivo: {device} | Nombre: {torch.cuda.get_device_name(0)}")

# ----------------------------
# Hiperparámetros (fiel a Secciones 7–8 del paper)
# ----------------------------
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
SEQ_LENGTH = 32
VOCAB_SIZE = 64
D_MODEL = 128
MLP_HIDDEN = 256

# CMS: niveles de frecuencia (tabla implícita en Sección 7.1)
CMS_FREQUENCIES = [1, 4, 16]  # actualización cada 1, 4, 16 batches

# Datos sintéticos: tarea de copia con ruido (como en evaluaciones de Hope)
DATASET_SIZE = 4096
np_data = np.random.randint(1, VOCAB_SIZE, size=(DATASET_SIZE, SEQ_LENGTH))
X = torch.from_numpy(np_data).long().to(device)
y = X.clone()

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
num_batches = len(train_loader)

# Política de impresión (como solicitaste)
if num_batches <= 10:
    print_every = 1
else:
    print_every = max(1, num_batches // 10)

# ----------------------------
# Self-Modifying Titans (Sección 8.1)
# ----------------------------
class SelfModifyingMemory(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_dim):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Memorias para generar proyecciones auto-modificables
        self.mem_k = nn.Sequential(nn.Embedding(vocab_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model))
        self.mem_v = nn.Sequential(nn.Embedding(vocab_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model))
        self.mem_eta = nn.Sequential(nn.Embedding(vocab_size, 1), nn.Sigmoid())  # learning rate ∈ (0,1)
        self.mem_alpha = nn.Sequential(nn.Embedding(vocab_size, 1), nn.Sigmoid())  # retention ∈ (0,1)

    def forward(self, x):
        k = self.mem_k(x)
        v = self.mem_v(x)
        eta = self.mem_eta(x).squeeze(-1)  # (B, S)
        alpha = self.mem_alpha(x).squeeze(-1)  # (B, S)
        return k, v, eta, alpha

# ----------------------------
# Continuum Memory System (CMS) - Sección 7.1
# ----------------------------
class ContinuumMemorySystem(nn.Module):
    def __init__(self, frequencies, d_model, hidden_dim):
        super().__init__()
        self.frequencies = frequencies
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_model)
            ) for _ in frequencies
        ])
        self.register_buffer('step_counters', torch.zeros(len(frequencies), dtype=torch.long, device=device))

    def forward(self, x, global_step):
        out = x
        for i, freq in enumerate(self.frequencies):
            if global_step % freq == 0:
                out = out + self.levels[i](out)  # residual, como en Titans
        return out

# ----------------------------
# Hope: Self-Modifying + CMS + DGD (Sección 8.3)
# ----------------------------
class HopeModel(nn.Module):
    def __init__(self, vocab_size, d_model, cms_freqs, hidden_dim):
        super().__init__()
        self.self_mod = SelfModifyingMemory(vocab_size, d_model, hidden_dim)
        self.cms = ContinuumMemorySystem(cms_freqs, d_model, hidden_dim)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, global_step):
        k, v, eta, alpha = self.self_mod(x)
        # Aplicar DGD manualmente en la capa de salida (simulado vía gradiente)
        x_emb = k  # usamos k como representación
        x_cms = self.cms(x_emb, global_step)
        logits = self.output_proj(x_cms)
        return logits, eta, alpha

# ----------------------------
# Inicialización
# ----------------------------
model = HopeModel(VOCAB_SIZE, D_MODEL, CMS_FREQUENCIES, MLP_HIDDEN).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ----------------------------
# Entrenamiento (con Delta Gradient Descent simulado vía gradiente personalizado)
# ----------------------------
print(f"\n🚀 Entrenando Hope (Nested Learning - NeurIPS 2025)")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"Vocabulario: {VOCAB_SIZE} | d_model: {D_MODEL}")
print(f"Niveles CMS (frecuencias): {CMS_FREQUENCIES}")
print(f"Tamaño dataset: {DATASET_SIZE}")
print(f"Batches/época: {num_batches}")
print("-" * 80)

global_step = 0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        logits, eta, alpha = model(x_batch, global_step)
        
        # Pérdida estándar (DGD se simula en gradiente, no en pérdida)
        logits_flat = logits.view(-1, VOCAB_SIZE)
        targets_flat = y_batch.view(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()

        # Aplicar DGD manualmente: adaptar gradiente con eta y alpha
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Simulación de DGD: decaimiento adaptativo basado en estado
                    param.grad = param.grad * (1 - eta.mean()) + alpha.mean() * param.data

        optimizer.step()
        global_step += 1

        # Métricas
        epoch_loss += loss.item() * targets_flat.size(0)
        preds = logits_flat.argmax(dim=1)
        correct += preds.eq(targets_flat).sum().item()
        total += targets_flat.size(0)

        # Impresión condicional
        if (batch_idx + 1) % print_every == 0 or batch_idx == num_batches - 1:
            acc = preds.eq(targets_flat).float().mean().item()
            print(f"Época {epoch:2d} | Batch {batch_idx + 1:2d}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    avg_loss = epoch_loss / total
    avg_acc = correct / total
    print(f"→ Época {epoch:2d} finalizada | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    print("-" * 80)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Configuración reproducible y dispositivo
# ----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# ----------------------------
# Hiperparámetros (CMS + Self-modifying)
# ----------------------------
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEQ_LENGTH = 16
VOCAB_SIZE = 32
D_MODEL = 32
MLP_HIDDEN = 64

# CMS: niveles de memoria y frecuencias (pasos entre actualizaciones)
CMS_LEVELS = [1, 4, 16]  # frecuencias relativas: rápido → lento

# Generación de datos sintéticos: secuencias de tokens (copia + ruido)
DATASET_SIZE = 2000
data = np.random.randint(1, VOCAB_SIZE, size=(DATASET_SIZE, SEQ_LENGTH))
X = torch.from_numpy(data).long()
y = X.clone()  # tarea de copia

# DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
num_batches = len(train_loader)

# Política de impresión
if num_batches <= 10:
    print_every_batch = 1
else:
    print_every_batch = max(1, num_batches // 10)

# ----------------------------
# Capa Self-Modifying Memory (inspirada en Hope)
# ----------------------------
class SelfModifyingMemory(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_dim):
        super().__init__()
        self.d_model = d_model
        # Memorias para generar proyecciones (key, value, learning rate, retention)
        self.mem_k = nn.Linear(vocab_size, hidden_dim)
        self.mem_v = nn.Linear(vocab_size, hidden_dim)
        self.mem_eta = nn.Linear(vocab_size, 1)
        self.mem_alpha = nn.Linear(vocab_size, 1)
        self.to_output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, update_mask=None):
        """
        x: (B, S)
        update_mask: None o máscara booleana para actualizaciones condicionales
        Retorna logits y parámetros internos (para depuración/futuras extensiones)
        """
        B, S = x.shape
        x_onehot = F.one_hot(x, num_classes=VOCAB_SIZE).float()  # (B, S, V)

        # Proyecciones generadas por memorias (self-modifying)
        k = self.mem_k(x_onehot)  # (B, S, H)
        v = self.mem_v(x_onehot)  # (B, S, H)
        eta = torch.sigmoid(self.mem_eta(x_onehot))  # learning rate adaptativo (0,1)
        alpha = torch.sigmoid(self.mem_alpha(x_onehot))  # retention gate (0,1)

        # Simulación de regla delta: actualización implícita vía gradiente
        # En una PoC completa, aquí iría un bucle de actualización explícita (chunk-wise),
        # pero para eficiencia en CPU/GPU usamos diferenciación automática.
        output = self.to_output(v)  # (B, S, V)
        return output

# ----------------------------
# Continuum Memory System (CMS): cadena de MLPs con frecuencias distintas
# ----------------------------
class ContinuumMemorySystem(nn.Module):
    def __init__(self, levels, d_model, hidden_dim):
        super().__init__()
        self.levels = levels
        self.memories = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_model)
            ) for _ in levels
        ])
        # Contadores de paso para cada nivel
        self.register_buffer('step_counters', torch.zeros(len(levels), dtype=torch.long))

    def forward(self, x, global_step):
        """
        x: (B, S, D)
        global_step: int, paso global de entrenamiento
        """
        out = x
        for i, (mem, freq) in enumerate(zip(self.memories, self.levels)):
            if global_step % freq == 0:
                out = mem(out) + out  # residual
        return out

# ----------------------------
# Modelo Hope completo (mínimo)
# ----------------------------
class HopeModel(nn.Module):
    def __init__(self, vocab_size, d_model, cms_levels, mlp_hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.self_mod = SelfModifyingMemory(vocab_size, d_model, mlp_hidden)
        self.cms = ContinuumMemorySystem(cms_levels, d_model, mlp_hidden)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, global_step):
        # Self-modifying phase
        logits_raw = self.self_mod(x)  # (B, S, V)
        # Extraer representación intermedia (promedio de logits como proxy)
        x_emb = self.embedding(x)  # (B, S, D)
        # Aplicar CMS
        x_cms = self.cms(x_emb, global_step)  # (B, S, D)
        # Clasificación final
        logits = self.classifier(x_cms)  # (B, S, V)
        return logits

# ----------------------------
# Inicialización
# ----------------------------
model = HopeModel(VOCAB_SIZE, D_MODEL, CMS_LEVELS, MLP_HIDDEN).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# Entrenamiento
# ----------------------------
print(f"\nEntrenando PoC de Nested Learning (Hope - CMS + Self-Modifying)")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"Vocabulario: {VOCAB_SIZE} | Dimensión: {D_MODEL}")
print(f"Niveles CMS (frecuencias): {CMS_LEVELS}")
print(f"Tamaño dataset: {DATASET_SIZE}")
print(f"Número de batches por época: {num_batches}")
print("-" * 85)

global_step = 0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        logits = model(x_batch, global_step)  # (B, S, V)

        # Pérdida y métricas
        logits_flat = logits.view(-1, VOCAB_SIZE)
        targets_flat = y_batch.view(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        # Métricas
        epoch_loss += loss.item() * targets_flat.size(0)
        preds = logits_flat.argmax(dim=1)
        correct += preds.eq(targets_flat).sum().item()
        total += targets_flat.size(0)

        # Actualizar paso global
        global_step += 1

        # Impresión condicional
        if (batch_idx + 1) % print_every_batch == 0 or batch_idx == num_batches - 1:
            batch_acc = preds.eq(targets_flat).float().mean().item()
            print(f"Época {epoch:2d} | Batch {batch_idx + 1:2d}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | Acc: {batch_acc:.4f}")

    avg_loss = epoch_loss / total
    avg_acc = correct / total
    print(f"→ Época {epoch:2d} finalizada | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    print("-" * 85)
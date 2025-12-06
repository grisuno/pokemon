import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------
# Configuración reproducible
# ----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Hiperparámetros
# ----------------------------
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEQ_LENGTH = 8
VOCAB_SIZE = 16  # pequeño vocabulario: tokens 0 a 15
D_MODEL = 32     # dimensión del modelo (d_model)
NUM_HEADS = 4
D_FF = 64        # dimensión de la capa feed-forward
DROPOUT = 0.1

# ----------------------------
# Generación de datos sintéticos: tarea de copia (input = output)
# ----------------------------
def generate_copy_data(num_samples, seq_len, vocab_size):
    data = np.random.randint(1, vocab_size, size=(num_samples, seq_len))
    return torch.from_numpy(data).long()

DATASET_SIZE = 2000
X = generate_copy_data(DATASET_SIZE, SEQ_LENGTH, VOCAB_SIZE)
y = X.clone()  # tarea de copia

# Dividir en batches
dataset_size = X.shape[0]
indices = np.arange(dataset_size)
num_batches = int(np.ceil(dataset_size / BATCH_SIZE))

# Frecuencia de impresión de batches
if num_batches <= 10:
    print_every_batch = 1
else:
    print_every_batch = max(1, num_batches // 10)

# ----------------------------
# Capa de Multi-Head Attention (manual)
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # Proyecciones lineales
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Atención escalada
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        # Concatenar cabezas
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)
        return self.out_linear(output)

# ----------------------------
# Capa Feed-Forward
# ----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ----------------------------
# Codificador Transformer mínimo (solo 1 capa)
# ----------------------------
class MiniTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        # Capa de atención
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

# ----------------------------
# Modelo completo: Transformer + clasificación de tokens
# ----------------------------
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.encoder = MiniTransformerEncoder(vocab_size, d_model, num_heads, d_ff, dropout)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

# ----------------------------
# Inicialización
# ----------------------------
device = torch.device('cpu')
model = MiniTransformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# Entrenamiento
# ----------------------------
print(f"Entrenando Transformer mínimo (Vaswani et al., 2017)")
print(f"Tarea: copia de secuencia (input → output idéntico)")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"Vocab size: {VOCAB_SIZE} | Seq len: {SEQ_LENGTH} | d_model: {D_MODEL}")
print(f"Cabezas: {NUM_HEADS} | d_ff: {D_FF}")
print(f"Número de batches por época: {num_batches}")
print("-" * 80)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, dataset_size)
        x_batch = X[start:end].to(device)
        y_batch = y[start:end].to(device)

        optimizer.zero_grad()
        logits = model(x_batch)  # (batch, seq, vocab)
        # Reorganizar para CrossEntropyLoss: (N*S, C) y (N*S,)
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

        # Impresión condicional
        if (batch_idx + 1) % print_every_batch == 0 or batch_idx == num_batches - 1:
            batch_acc = preds.eq(targets_flat).float().mean().item()
            print(f"Época {epoch:2d} | Batch {batch_idx + 1:2d}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | Acc: {batch_acc:.4f}")

    avg_loss = epoch_loss / total
    avg_acc = correct / total
    print(f"→ Época {epoch:2d} finalizada | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    print("-" * 80)
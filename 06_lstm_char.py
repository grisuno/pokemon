import torch
import torch.nn as nn
import urllib.request
import numpy as np
import random

# ----------------------------
# Configuración reproducible
# ----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------
# Hiperparámetros
# ----------------------------
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01
SEQ_LENGTH = 20
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1
DROPOUT_RATE = 0.1

# URL del texto público (Alice in Wonderland - Project Gutenberg)
TEXT_URL = "https://www.gutenberg.org/files/11/11-0.txt"

# ----------------------------
# Descarga y preprocesamiento del texto
# ----------------------------
print("Descargando y preparando dataset de texto...")
try:
    with urllib.request.urlopen(TEXT_URL) as f:
        text = f.read().decode("utf-8")
except Exception as e:
    # Fallback: usar un fragmento si falla la descarga
    text = (
        "Alice was beginning to get very tired of sitting by her sister on the bank, "
        "and of having nothing to do: once or twice she had peeped into the book her sister was reading, "
        "but it had no pictures or conversations in it, and what is the use of a book, "
        "thought Alice without pictures or conversation?"
    ) * 10

# Convertir a minúsculas para reducir vocabulario
text = text.lower()

# Crear mapeos carácter ↔ índice
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"Vocabulario: {vocab_size} caracteres únicos")
print(f"Muestra de vocabulario: {''.join(chars[:20])}...")

# Codificar texto completo
data = [char_to_idx[ch] for ch in text]

# ----------------------------
# Crear batches de secuencias
# ----------------------------
def create_batches(data, batch_size, seq_length):
    num_batches = len(data) // (batch_size * seq_length)
    if num_batches == 0:
        raise ValueError("Dataset too small for given batch_size and seq_length.")
    data = data[:num_batches * batch_size * seq_length]
    data = np.array(data, dtype=np.int64)
    data = data.reshape(batch_size, -1)
    batches = []
    for i in range(0, data.shape[1] - seq_length, seq_length):
        x = data[:, i:i+seq_length]
        y = data[:, i+1:i+seq_length+1]
        batches.append((torch.from_numpy(x), torch.from_numpy(y)))
    return batches

batches = create_batches(data, BATCH_SIZE, SEQ_LENGTH)
num_batches = len(batches)

# Frecuencia de impresión de batches
if num_batches <= 10:
    print_every_batch = 1
else:
    print_every_batch = max(1, num_batches // 10)

# ----------------------------
# Modelo LSTM
# ----------------------------
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)
        logits = self.fc(lstm_out)
        return logits, hidden

# ----------------------------
# Inicialización
# ----------------------------
device = torch.device('cpu')
model = CharLSTM(vocab_size, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, DROPOUT_RATE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# Entrenamiento
# ----------------------------
print(f"\nEntrenando LSTM para modelado de caracteres (Hochreiter & Schmidhuber, 1997)")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | Seq length: {SEQ_LENGTH}")
print(f"Vocab size: {vocab_size} | Hidden size: {LSTM_HIDDEN_SIZE} | Capas: {LSTM_NUM_LAYERS}")
print(f"Número de batches por época: {num_batches}")
print("-" * 75)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x_batch, y_batch) in enumerate(batches):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(x_batch)
        # Reorganizar para CrossEntropyLoss: (N*S, C) y (N*S,)
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = y_batch.reshape(-1)
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
    print("-" * 75)
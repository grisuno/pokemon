import torch
import torch.nn as nn
import torch.nn.functional as F
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
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
SEQ_LENGTH = 12
VOCAB_SIZE = 32  # incluye [PAD], [MASK], [UNK]
D_MODEL = 32
NUM_HEADS = 4
D_FF = 64
DROPOUT = 0.1
MASK_PROB = 0.15  # 15% de tokens se enmascaran (BERT original)

# Tokens especiales
PAD_TOKEN = 0
MASK_TOKEN = 1
UNK_TOKEN = 2

# ----------------------------
# Corpus sintético: frases simples (para vocabulario controlado)
# ----------------------------
corpus = [
    "the cat sits on the mat",
    "a dog runs in the park",
    "birds fly over the trees",
    "she reads a book every day",
    "he drinks coffee in the morning",
    "we walk to school together",
    "they play soccer after class",
    "i like chocolate ice cream",
    "the sun shines brightly today",
    "my friend has a red bike"
] * 20  # repetir para tener suficientes muestras

# Tokenización básica
vocab = {"[PAD]": PAD_TOKEN, "[MASK]": MASK_TOKEN, "[UNK]": UNK_TOKEN}
token_id = 3
for sentence in corpus:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = token_id
            token_id += 1

# Asegurar que no excedamos VOCAB_SIZE
if len(vocab) > VOCAB_SIZE:
    # Truncar vocabulario si es necesario
    vocab_items = list(vocab.items())[:VOCAB_SIZE]
    vocab = {k: v for k, v in vocab_items}
    # Remapear valores si se truncó
    for i, (k, _) in enumerate(vocab.items()):
        vocab[k] = i
    PAD_TOKEN = vocab.get("[PAD]", 0)
    MASK_TOKEN = vocab.get("[MASK]", 1)
    UNK_TOKEN = vocab.get("[UNK]", 2)

# Inversión
idx_to_token = {idx: token for token, idx in vocab.items()}
vocab_size_actual = len(vocab)

# Codificar corpus
def tokenize_sentence(sentence):
    tokens = sentence.split()
    ids = [vocab.get(t, UNK_TOKEN) for t in tokens]
    return ids

encoded_corpus = []
for sentence in corpus:
    ids = tokenize_sentence(sentence)
    if len(ids) <= SEQ_LENGTH:
        encoded_corpus.append(ids)

# Rellenar a SEQ_LENGTH
def pad_sequence(seq, length, pad_value=PAD_TOKEN):
    return seq + [pad_value] * (length - len(seq))

X = [pad_sequence(seq, SEQ_LENGTH) for seq in encoded_corpus]
X = torch.tensor(X, dtype=torch.long)

DATASET_SIZE = X.shape[0]
indices = np.arange(DATASET_SIZE)
num_batches = int(np.ceil(DATASET_SIZE / BATCH_SIZE))

# Frecuencia de impresión de batches
if num_batches <= 10:
    print_every_batch = 1
else:
    print_every_batch = max(1, num_batches // 10)

# ----------------------------
# Reutilizar capas Transformer del script anterior (resumidas)
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)
        return self.out_linear(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class BERTLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class TinyBERT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len=500):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = BERTLayer(d_model, num_heads, d_ff, dropout)
        self.classifier = nn.Linear(d_model, vocab_size)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, mask):
        seq_len = x.size(1)
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        return self.classifier(x)

# ----------------------------
# Función para enmascarar dinámicamente
# ----------------------------
def mask_tokens(inputs, vocab_size, mask_token_id, pad_token_id, mask_prob=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mask_prob)
    # No enmascarar padding
    padding_mask = (inputs == pad_token_id)
    probability_matrix.masked_fill_(padding_mask, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Ignorar en pérdida

    # 80% → [MASK], 10% → aleatorio, 10% → original (como en BERT)
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token_id

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels

# ----------------------------
# Inicialización
# ----------------------------
device = torch.device('cpu')
model = TinyBERT(vocab_size_actual, D_MODEL, NUM_HEADS, D_FF, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# Entrenamiento
# ----------------------------
print(f"Entrenando BERT-tiny (Devlin et al., 2018) - MLM")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"Vocab size: {vocab_size_actual} | Seq len: {SEQ_LENGTH}")
print(f"Mask prob: {MASK_PROB} | d_model: {D_MODEL} | Cabezas: {NUM_HEADS}")
print(f"Dataset size: {DATASET_SIZE}")
print(f"Número de batches por época: {num_batches}")
print("-" * 85)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total_masked = 0

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, DATASET_SIZE)
        batch_inputs = X[start:end].to(device)

        # Crear máscara de atención (ignora padding)
        attention_mask = (batch_inputs != PAD_TOKEN).long()

        # Enmascarar tokens
        inputs_masked, labels = mask_tokens(
            batch_inputs.clone(),
            vocab_size_actual,
            MASK_TOKEN,
            PAD_TOKEN,
            MASK_PROB
        )
        inputs_masked, labels = inputs_masked.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs_masked, attention_mask)  # (batch, seq, vocab)
        loss = criterion(logits.view(-1, vocab_size_actual), labels.view(-1))
        loss.backward()
        optimizer.step()

        # Métricas solo en tokens enmascarados
        masked_positions = (labels != -100)
        if masked_positions.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct += (preds[masked_positions] == labels[masked_positions]).sum().item()
            total_masked += masked_positions.sum().item()

        epoch_loss += loss.item() * batch_inputs.size(0)

        # Impresión condicional
        if (batch_idx + 1) % print_every_batch == 0 or batch_idx == num_batches - 1:
            current_acc = correct / total_masked if total_masked > 0 else 0.0
            print(f"Época {epoch:2d} | Batch {batch_idx + 1:2d}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | MLM Acc: {current_acc:.4f}")

    avg_loss = epoch_loss / DATASET_SIZE
    final_acc = correct / total_masked if total_masked > 0 else 0.0
    print(f"→ Época {epoch:2d} finalizada | Loss: {avg_loss:.4f} | MLM Acc: {final_acc:.4f}")
    print("-" * 85)
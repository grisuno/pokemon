import numpy as np

# ----------------------------
# Configuración reproducible
# ----------------------------
SEED = 42
np.random.seed(SEED)

# ----------------------------
# Hiperparámetros
# ----------------------------
NUM_EPOCHS = 10
LEARNING_RATE = 0.5
BATCH_SIZE = 4
INPUT_DIM = 2
HIDDEN_DIM = 4
OUTPUT_DIM = 1

# ----------------------------
# Generación de datos: versión suavizada del XOR
# (no linealmente separable, requiere capa oculta)
# ----------------------------
X_raw = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_raw = np.array([[0], [1], [1], [0]], dtype=np.float32)  # XOR

# Añadir pequeña perturbación determinista para evitar simetría perfecta
X = X_raw + 0.01 * np.random.randn(*X_raw.shape)
y = y_raw.copy()

# ----------------------------
# Funciones de activación y derivadas
# ----------------------------
def sigmoid(z):
    # Clipping para estabilidad numérica
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# ----------------------------
# Inicialización de pesos
# ----------------------------
W1 = np.random.randn(INPUT_DIM, HIDDEN_DIM) * 0.5
b1 = np.zeros((1, HIDDEN_DIM))
W2 = np.random.randn(HIDDEN_DIM, OUTPUT_DIM) * 0.5
b2 = np.zeros((1, OUTPUT_DIM))

# ----------------------------
# Forward pass
# ----------------------------
def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return a2, a1, z1, z2

# ----------------------------
# Backward pass y actualización
# ----------------------------
def backward(X, y_true, y_pred, a1, z1, lr):
    global W1, b1, W2, b2

    m = X.shape[0]

    # Gradiente de la capa de salida
    dz2 = y_pred - y_true
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # Gradiente de la capa oculta
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * sigmoid_derivative(z1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # Actualización
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

# ----------------------------
# Cálculo de pérdida (MSE) y precisión
# ----------------------------
def compute_metrics(y_pred, y_true):
    loss = np.mean((y_pred - y_true) ** 2)
    # Precisión: redondeo a 0/1 y comparación
    y_pred_class = (y_pred > 0.5).astype(np.float32)
    acc = np.mean(y_pred_class == y_true)
    return loss, acc

# ----------------------------
# Preparación de batches
# ----------------------------
dataset_size = X.shape[0]
indices = np.arange(dataset_size)
num_batches = int(np.ceil(dataset_size / BATCH_SIZE))

# Frecuencia de impresión de batches
if num_batches <= 10:
    print_every_batch = 1
else:
    print_every_batch = max(1, num_batches // 10)

# ----------------------------
# Entrenamiento
# ----------------------------
print(f"Iniciando entrenamiento de MLP con retropropagación (Rumelhart et al., 1986)")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"Arquitectura: {INPUT_DIM} → {HIDDEN_DIM} → {OUTPUT_DIM}")
print(f"Número de batches por época: {num_batches}")
print("-" * 70)

for epoch in range(1, NUM_EPOCHS + 1):
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    epoch_loss = 0.0
    epoch_acc = 0.0
    total_samples = 0

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, dataset_size)
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Forward
        y_pred, a1, z1, z2 = forward(X_batch)

        # Métricas del batch
        batch_loss, batch_acc = compute_metrics(y_pred, y_batch)
        epoch_loss += batch_loss * len(y_batch)
        epoch_acc += batch_acc * len(y_batch)
        total_samples += len(y_batch)

        # Backward y actualización
        backward(X_batch, y_batch, y_pred, a1, z1, LEARNING_RATE)

        # Impresión condicional
        if (batch_idx + 1) % print_every_batch == 0 or batch_idx == num_batches - 1:
            print(f"Época {epoch:2d} | Batch {batch_idx + 1:2d}/{num_batches} | "
                  f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f}")

    # Métricas promedio de la época
    avg_epoch_loss = epoch_loss / total_samples
    avg_epoch_acc = epoch_acc / total_samples
    print(f"→ Época {epoch:2d} finalizada | Loss: {avg_epoch_loss:.4f} | Acc: {avg_epoch_acc:.4f}")
    print("-" * 70)
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
LEARNING_RATE = 0.1
BATCH_SIZE = 4  # pequeño para que num_batches sea entero en dataset pequeño
INPUT_DIM = 2

# ----------------------------
# Generación de datos (AND con ruido mínimo para evitar singularidades)
# ----------------------------
# Usamos compuerta AND como base: linealmente separable.
X_raw = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_raw = np.array([0, 0, 0, 1], dtype=np.float32)

# Añadir ligera perturbación determinista (dentro de la separabilidad)
X = X_raw + 0.01 * np.random.randn(*X_raw.shape)
y = y_raw.copy()

# ----------------------------
# Clase Perceptrón
# ----------------------------
class Perceptron:
    def __init__(self, input_dim, learning_rate):
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, X):
        # Salida binaria: 1 si activación >= 0, 0 si no
        activation = np.dot(X, self.weights) + self.bias
        return (activation >= 0).astype(np.float32)

    def train_step(self, X_batch, y_batch):
        # Predicción
        y_pred = self.predict(X_batch)
        # Error: diferencia real (no solo signo)
        errors = y_batch - y_pred
        # Actualización: regla del perceptrón
        self.weights += self.lr * np.dot(errors, X_batch)
        self.bias += self.lr * np.sum(errors)

    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)

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
# Inicialización del modelo
# ----------------------------
model = Perceptron(input_dim=INPUT_DIM, learning_rate=LEARNING_RATE)

# ----------------------------
# Ciclo de entrenamiento
# ----------------------------
print(f"Iniciando entrenamiento del Perceptrón (Rosenblatt, 1958)")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"Número de batches por época: {num_batches}")
print("-" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    # Mezcla determinista (gracias a la semilla)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, dataset_size)
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Entrenamiento
        model.train_step(X_batch, y_batch)

        # Métricas del batch
        y_pred_batch = model.predict(X_batch)
        correct_in_batch = np.sum(y_pred_batch == y_batch)
        correct_predictions += correct_in_batch
        total_samples += len(y_batch)

        # Impresión condicional por batch
        if (batch_idx + 1) % print_every_batch == 0 or batch_idx == num_batches - 1:
            batch_acc = correct_in_batch / len(y_batch)
            print(f"Época {epoch:2d} | Batch {batch_idx + 1:2d}/{num_batches} | "
                  f"Batch acc: {batch_acc:.4f}")

    # Precisión final de la época
    epoch_acc = correct_predictions / total_samples
    print(f"→ Época {epoch:2d} finalizada | Precisión época: {epoch_acc:.4f}")
    print("-" * 60)
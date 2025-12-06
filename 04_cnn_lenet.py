import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
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
LEARNING_RATE = 0.01
IMAGE_SIZE = 28  # MNIST
NUM_CLASSES = 10

# Tamaño del dataset reducido para CPU
REDUCED_DATASET_SIZE = 1000

# ----------------------------
# Cargar y preparar datos (MNIST reducido)
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Estadísticas estándar de MNIST
])

full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Submuestreo determinista
subset_indices = np.arange(REDUCED_DATASET_SIZE)
reduced_dataset = Subset(full_dataset, subset_indices)
train_loader = DataLoader(reduced_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# Arquitectura inspirada en LeNet-5
# ----------------------------
class LeNet5Like(nn.Module):
    def __init__(self):
        super().__init__()
        # Capa 1: Conv → AvgPool
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # salida: 28x28 → 28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)      # 28x28 → 14x14
        # Capa 2: Conv → AvgPool
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)           # 14x14 → 10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)     # 10x10 → 5x5
        # Capas fully connected
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------------------
# Inicialización
# ----------------------------
device = torch.device('cpu')
model = LeNet5Like().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Número de batches
num_batches = len(train_loader)

# Frecuencia de impresión de batches
if num_batches <= 10:
    print_every_batch = 1
else:
    print_every_batch = max(1, num_batches // 10)

# ----------------------------
# Entrenamiento
# ----------------------------
print(f"Iniciando entrenamiento de CNN inspirada en LeNet-5 (LeCun et al., 1998)")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"Dataset: MNIST reducido a {REDUCED_DATASET_SIZE} muestras")
print(f"Arquitectura: Conv(6) → Pool → Conv(16) → Pool → FC(120→84→10)")
print(f"Número de batches por época: {num_batches}")
print("-" * 75)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Métricas del batch
        epoch_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=False)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # Impresión condicional
        if (batch_idx + 1) % print_every_batch == 0 or batch_idx == num_batches - 1:
            batch_acc = pred.eq(target).float().mean().item()
            print(f"Época {epoch:2d} | Batch {batch_idx + 1:2d}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | Acc: {batch_acc:.4f}")

    # Métricas promedio de la época
    avg_loss = epoch_loss / total
    avg_acc = correct / total
    print(f"→ Época {epoch:2d} finalizada | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    print("-" * 75)
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
LEARNING_RATE = 0.0002
LATENT_DIM = 100
IMG_SIZE = 28 * 28
REDUCED_DATASET_SIZE = 1000

# ----------------------------
# Cargar y preparar datos (MNIST reducido)
# ----------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
subset_indices = np.arange(REDUCED_DATASET_SIZE)
reduced_dataset = Subset(full_dataset, subset_indices)
train_loader = DataLoader(reduced_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# Modelo: Generador
# ----------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, img_size),
            nn.Tanh()  # Salida en [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# ----------------------------
# Modelo: Discriminador
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probabilidad de ser real
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------
# Inicialización
# ----------------------------
device = torch.device('cpu')
generator = Generator(LATENT_DIM, IMG_SIZE).to(device)
discriminator = Discriminator(IMG_SIZE).to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

num_batches = len(train_loader)

# Frecuencia de impresión de batches
if num_batches <= 10:
    print_every_batch = 1
else:
    print_every_batch = max(1, num_batches // 10)

# ----------------------------
# Entrenamiento
# ----------------------------
print(f"Entrenando GAN (Goodfellow et al., 2014)")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"Latent dim: {LATENT_DIM} | Imagenes: 28x28 (aplanadas)")
print(f"Dataset: MNIST reducido a {REDUCED_DATASET_SIZE} muestras")
print(f"Número de batches por época: {num_batches}")
print("-" * 80)

for epoch in range(1, NUM_EPOCHS + 1):
    g_epoch_loss = 0.0
    d_epoch_loss = 0.0
    d_real_acc_total = 0.0
    d_fake_acc_total = 0.0
    total_batches = 0

    for batch_idx, (real_imgs, _) in enumerate(train_loader):
        batch_size_actual = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # Etiquetas
        real_labels = torch.ones(batch_size_actual, 1, device=device)
        fake_labels = torch.zeros(batch_size_actual, 1, device=device)

        # ---------------------
        # 1. Entrenar Discriminador
        # ---------------------
        optimizer_d.zero_grad()

        # Pérdida en reales
        d_real_pred = discriminator(real_imgs)
        d_real_loss = criterion(d_real_pred, real_labels)
        d_real_acc = (d_real_pred > 0.5).float().mean().item()

        # Pérdida en falsos
        z = torch.randn(batch_size_actual, LATENT_DIM, device=device)
        fake_imgs = generator(z)
        d_fake_pred = discriminator(fake_imgs.detach())
        d_fake_loss = criterion(d_fake_pred, fake_labels)
        d_fake_acc = (d_fake_pred <= 0.5).float().mean().item()

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_d.step()

        # ---------------------
        # 2. Entrenar Generador
        # ---------------------
        optimizer_g.zero_grad()
        d_fake_pred_g = discriminator(fake_imgs)
        g_loss = criterion(d_fake_pred_g, real_labels)
        g_loss.backward()
        optimizer_g.step()

        # Acumular métricas
        g_epoch_loss += g_loss.item()
        d_epoch_loss += d_loss.item()
        d_real_acc_total += d_real_acc
        d_fake_acc_total += d_fake_acc
        total_batches += 1

        # Impresión condicional por batch
        if (batch_idx + 1) % print_every_batch == 0 or batch_idx == num_batches - 1:
            print(f"Época {epoch:2d} | Batch {batch_idx + 1:2d}/{num_batches} | "
                  f"D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f} | "
                  f"D_real_acc: {d_real_acc:.4f} | D_fake_acc: {d_fake_acc:.4f}")

    # Métricas promedio de la época
    avg_g_loss = g_epoch_loss / total_batches
    avg_d_loss = d_epoch_loss / total_batches
    avg_d_real_acc = d_real_acc_total / total_batches
    avg_d_fake_acc = d_fake_acc_total / total_batches

    print(f"→ Época {epoch:2d} finalizada | D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f} | "
          f"D_real_acc: {avg_d_real_acc:.4f} | D_fake_acc: {avg_d_fake_acc:.4f}")
    print("-" * 80)
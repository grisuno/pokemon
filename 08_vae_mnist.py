import torch
import torch.nn as nn
import torch.nn.functional as F
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
LEARNING_RATE = 1e-3
LATENT_DIM = 2
INPUT_DIM = 28 * 28
HIDDEN_DIM = 400

# Tamaño del dataset reducido
REDUCED_DATASET_SIZE = 1000

# ----------------------------
# Cargar y preparar datos (MNIST reducido)
# ----------------------------
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
subset_indices = np.arange(REDUCED_DATASET_SIZE)
reduced_dataset = Subset(full_dataset, subset_indices)
train_loader = DataLoader(reduced_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# Modelo VAE
# ----------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # log_var
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# ----------------------------
# Función de pérdida VAE
# ----------------------------
def vae_loss(recon_x, x, mu, log_var):
    # Pérdida de reconstrucción (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # Divergencia KL
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

# ----------------------------
# Inicialización
# ----------------------------
device = torch.device('cpu')
model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

num_batches = len(train_loader)

# Frecuencia de impresión de batches
if num_batches <= 10:
    print_every_batch = 1
else:
    print_every_batch = max(1, num_batches // 10)

# ----------------------------
# Entrenamiento
# ----------------------------
print(f"Entrenando Autoencoder Variacional (VAE) - Kingma & Welling (2013)")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"Dimensión latente: {LATENT_DIM} | Hidden: {HIDDEN_DIM}")
print(f"Dataset: MNIST reducido a {REDUCED_DATASET_SIZE} muestras")
print(f"Número de batches por época: {num_batches}")
print("-" * 80)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_total_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    total_samples = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, INPUT_DIM).to(device)
        optimizer.zero_grad()

        recon_batch, mu, log_var = model(data)
        total_loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, log_var)
        total_loss.backward()
        optimizer.step()

        # Acumular métricas
        batch_size_actual = data.size(0)
        epoch_total_loss += total_loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()
        total_samples += batch_size_actual

        # Impresión condicional por batch
        if (batch_idx + 1) % print_every_batch == 0 or batch_idx == num_batches - 1:
            avg_total = total_loss.item() / batch_size_actual
            avg_recon = recon_loss.item() / batch_size_actual
            avg_kl = kl_loss.item() / batch_size_actual
            print(f"Época {epoch:2d} | Batch {batch_idx + 1:2d}/{num_batches} | "
                  f"Total: {avg_total:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

    # Métricas promedio por muestra en la época
    avg_total_epoch = epoch_total_loss / total_samples
    avg_recon_epoch = epoch_recon_loss / total_samples
    avg_kl_epoch = epoch_kl_loss / total_samples
    print(f"→ Época {epoch:2d} finalizada | Total: {avg_total_epoch:.4f} | "
          f"Recon: {avg_recon_epoch:.4f} | KL: {avg_kl_epoch:.4f}")
    print("-" * 80)
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
# Hiperparámetros del modelo y difusión
# ----------------------------
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
T_STEPS = 100  # pasos de difusión (original: 1000, reducido para CPU)
IMG_SIZE = 28
IMG_CHANNELS = 1
REDUCED_DATASET_SIZE = 1000

# ----------------------------
# Programa de varianza (beta schedule)
# ----------------------------
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T_STEPS)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Mover a CPU
betas = betas
alphas_cumprod = alphas_cumprod

# ----------------------------
# Cargar y preparar datos: MNIST → [-1, 1]
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # [0,1] → [-1,1]
])
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
subset_indices = np.arange(REDUCED_DATASET_SIZE)
reduced_dataset = Subset(full_dataset, subset_indices)
train_loader = DataLoader(reduced_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# Red residual mínima (estilo UNet simplificado)
# ----------------------------
class SimpleDiffusionNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=32):
        super().__init__()
        self.inc = nn.Conv2d(in_channels + 1, hidden_dim, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(hidden_dim * 2 * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.outc = nn.Conv2d(hidden_dim * 2, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # Codificar timestep t como mapa constante
        t_map = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_map], dim=1)
        x1 = self.relu(self.inc(x))
        x2 = self.relu(self.down1(x1))
        x3 = self.relu(self.down2(x2))
        x = self.relu(self.up1(x3))
        x = torch.cat([x, x2], dim=1)
        x = self.relu(self.up2(x))
        x = torch.cat([x, x1], dim=1)
        return self.outc(x)

# ----------------------------
# Función para añadir ruido (forward process)
# ----------------------------
def q_sample(x_0, t, noise=None):
    """
    Muestrea x_t dado x_0 y timestep t.
    x_0: (B, C, H, W)
    t: (B,)
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_0 + torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1) * noise

# ----------------------------
# Inicialización
# ----------------------------
device = torch.device('cpu')
model = SimpleDiffusionNet(in_channels=IMG_CHANNELS, out_channels=IMG_CHANNELS).to(device)
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
print(f"Entrenando modelo de difusión mínimo (DDPM - Ho et al., 2020)")
print(f"Épocas: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print(f"Pasos de difusión T: {T_STEPS} (reducido desde 1000)")
print(f"Dataset: MNIST reducido a {REDUCED_DATASET_SIZE} muestras en [-1, 1]")
print(f"Número de batches por época: {num_batches}")
print("-" * 80)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    total_samples = 0

    for batch_idx, (x_0, _) in enumerate(train_loader):
        x_0 = x_0.to(device)
        batch_size_actual = x_0.size(0)

        # Muestrear timesteps aleatorios
        t = torch.randint(0, T_STEPS, (batch_size_actual,), device=device).long()

        # Generar ruido y x_t
        noise = torch.randn_like(x_0)
        x_t = q_sample(x_0, t, noise)

        # Predicción del modelo
        predicted_noise = model(x_t, t.float() / T_STEPS)  # normalizar t a [0,1]

        # Pérdida: MSE entre ruido real y predicho
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_size_actual
        total_samples += batch_size_actual

        # Impresión condicional por batch
        if (batch_idx + 1) % print_every_batch == 0 or batch_idx == num_batches - 1:
            print(f"Época {epoch:2d} | Batch {batch_idx + 1:2d}/{num_batches} | "
                  f"Loss: {loss.item():.6f}")

    avg_loss = epoch_loss / total_samples
    print(f"→ Época {epoch:2d} finalizada | Pérdida promedio: {avg_loss:.6f}")
    print("-" * 80)
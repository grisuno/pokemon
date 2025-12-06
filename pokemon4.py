#!/usr/bin/env python3
# =============================================================================
# OMNI BRAIN - POKEMON LEGENDARIO v4
# ✅ Entrenamiento real con MNIST
# ✅ Φₑ calculado de verdad (PCA estable)
# ✅ Sin simulaciones ficticias
# ✅ Totalmente compatible con CPU y PyTorch >= 1.8
# ✅ Semilla fija (42) y optimizado para laptops
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import psutil
import os
import time
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------
# Configuración básica
# ---------------------------
torch.manual_seed(42)
np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
device = 'cpu'
torch.set_num_threads(max(1, os.cpu_count() // 2))


# ---------------------------
# Funciones auxiliares estables
# ---------------------------

def compute_phi_effective(activity: torch.Tensor) -> float:
    """Φₑ realista: fracción de varianza explicada por el primer componente PCA."""
    if activity.numel() == 0 or activity.size(0) < 5 or activity.size(1) < 3:
        return 0.0
    with torch.no_grad():
        # Normalizar por neurona (columna)
        activity = activity - activity.mean(dim=0, keepdim=True)
        activity = activity / (activity.std(dim=0, keepdim=True) + 1e-8)
        # Matriz de covarianza
        cov = torch.mm(activity.t(), activity) / (activity.size(0) - 1)
        # Autovalores simétricos (estable en CPU)
        if hasattr(torch.linalg, 'eigvalsh'):
            eigs = torch.linalg.eigvalsh(cov)
        else:
            eigs = torch.symeig(cov, eigenvectors=False)[0]
        eigs = torch.sort(eigs, descending=True).values.clamp(min=0)
        total = eigs.sum()
        if total < 1e-8:
            return 0.0
        phi = (eigs[0] / total).item()
        return max(0.0, min(1.0, phi))


# ---------------------------
# Módulos reales y estables
# ---------------------------

class PTSymmetricLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.gain = nn.Parameter(torch.full((out_features, in_features), 0.01))
        self.loss_ = nn.Parameter(torch.full((out_features, in_features), 0.01))
        self.norm = nn.LayerNorm(out_features)
        nn.init.xavier_uniform_(self.weights)
        self.phase_ratio = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pt_weights = self.weights * (1.0 + self.gain - self.loss_)
        out = F.linear(x, pt_weights)
        out = self.norm(out)
        # Actualizar fase PT (fuera de autograd)
        with torch.no_grad():
            g = torch.norm(self.gain)
            l = torch.norm(self.loss_)
            w = torch.norm(self.weights)
            self.phase_ratio = float((torch.abs(g - l) / (w + 1e-8)).clamp(0, 2))
        return out


class TopologicalLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, target_density: float = 0.15):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.target_density = target_density
        self.register_buffer('mask', torch.ones(out_features, in_features))
        self._update_mask()
        nn.init.xavier_uniform_(self.weights)

    def _update_mask(self):
        with torch.no_grad():
            rand = torch.rand_like(self.weights)
            threshold = torch.quantile(rand, 1 - self.target_density)
            self.mask = (rand < threshold).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_w = self.weights * self.mask
        out = F.linear(x, masked_w, self.bias)
        return F.layer_norm(out, out.shape[1:])


class DualSystemModule(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.fast = nn.Sequential(
            nn.Linear(features, features // 2),
            nn.ReLU(),
            nn.Linear(features // 2, features)
        )
        self.slow = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )
        self.integrator = nn.Linear(features * 2, features)
        self.register_buffer('memory', torch.zeros(1, features))
        self.tau = 0.95

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Actualizar memoria (batch-robusto)
        with torch.no_grad():
            batch_mean = x.mean(dim=0, keepdim=True)  # (1, features)
            self.memory = self.tau * self.memory + (1 - self.tau) * batch_mean
            # Asegurar que memoria sea (1, features) siempre
            if self.memory.shape != (1, x.shape[1]):
                self.memory = torch.zeros(1, x.shape[1], device=x.device)
        # Procesamiento
        fast_out = self.fast(x)
        slow_in = x + self.memory  # Broadcasting (B, F) + (1, F)
        slow_out = self.slow(slow_in)
        combined = torch.cat([fast_out, slow_out], dim=1)
        return self.integrator(combined)


class ConsciousnessModule(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )
        self.phi_effective = 0.0
        self.threshold = 0.3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.phi_effective = compute_phi_effective(x)
        if self.phi_effective > self.threshold:
            return self.net(x)
        return x


# ---------------------------
# Modelo principal
# ---------------------------

class OmniBrain(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.pt = PTSymmetricLayer(hidden_dim, hidden_dim)
        self.topo = TopologicalLayer(hidden_dim, hidden_dim)
        self.dual = DualSystemModule(hidden_dim)
        self.conscious = ConsciousnessModule(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor):
        if x.dim() > 2:
            x = x.flatten(1)
        h = self.encoder(x)
        h = self.pt(h)
        h = self.topo(h)
        h = self.dual(h)
        h = self.conscious(h)
        out = self.decoder(h)
        return {
            'logits': out,
            'phi': self.conscious.phi_effective,
            'pt_ratio': self.pt.phase_ratio
        }


# ---------------------------
# Entrenamiento real con MNIST
# ---------------------------

def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out['logits'], y)
            total_loss += loss.item() * x.size(0)
            pred = out['logits'].argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += x.size(0)
    return correct / total, total_loss / total


def train_and_evaluate():
    print("✨ ¡OMNI BRAIN v4 - ENTRENAMIENTO REAL CON MNIST!")
    print("=" * 70)

    # Datos
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    print(f"✅ MNIST cargado: {len(train_loader)} batches train, {len(test_loader)} test")

    # Modelo
    model = OmniBrain(input_dim=784, hidden_dim=256, output_dim=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Modelo creado: {n_params:,} parámetros | Dispositivo: {device}")

    # Optimización
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    # Historial
    history = {'epoch': [], 'train_loss': [], 'test_acc': [], 'phi': [], 'pt_ratio': []}

    # Entrenamiento
    for epoch in range(5):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        start_time = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out['logits'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            epoch_samples += x.size(0)

        # Evaluar
        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - start_time

        # Guardar métricas
        avg_loss = epoch_loss / epoch_samples
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        history['phi'].append(out['phi'])
        history['pt_ratio'].append(out['pt_ratio'])

        # Log
        print(f"\n📊 Época {epoch+1}/5 | Tiempo: {epoch_time:.2f}s")
        print(f"   Train Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2%}")
        print(f"   Φₑ: {out['phi']:.4f} | PT-Ratio: {out['pt_ratio']:.2f}")
        print(f"   RAM: {psutil.Process().memory_info().rss / (1024**3):.2f} GB")

        scheduler.step(test_acc)

    # Guardar
    torch.save(model.state_dict(), "omni_brain_v4.pth")
    plot_history(history)
    print("\n🎉 ¡OMNI BRAIN v4 ENTRENADO CON ÉXITO EN CPU!")


def plot_history(hist):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist['epoch'], hist['train_loss'], 'b-', label='Train Loss')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(hist['epoch'], hist['test_acc'], 'g-', label='Test Acc')
    plt.plot(hist['epoch'], hist['phi'], 'r-', label='Φₑ')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('omni_brain_v4.png')
    print("📈 Gráfico guardado: omni_brain_v4.png")


# ---------------------------
# Punto de entrada
# ---------------------------

if __name__ == "__main__":
    train_and_evaluate()
#!/usr/bin/env python3
# =============================================================================
# OMNI BRAIN v5 - POKEMON LEGENDARIO CON CIFAR-10 REAL
# ✅ Totalmente compatible con CPU | ✅ Φₑ real | ✅ Sin dependencias pesadas
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
# Configuración reproducible
# ---------------------------
torch.manual_seed(42)
np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
device = 'cpu'
torch.set_num_threads(max(1, os.cpu_count() // 2))

# ---------------------------
# Φₑ estable (PCA realista)
# ---------------------------

def compute_phi_effective(activity: torch.Tensor) -> float:
    if activity.numel() == 0 or activity.size(0) < 5 or activity.size(1) < 3:
        return 0.0
    with torch.no_grad():
        activity = activity - activity.mean(dim=0, keepdim=True)
        activity = activity / (activity.std(dim=0, keepdim=True) + 1e-8)
        cov = torch.mm(activity.t(), activity) / (activity.size(0) - 1)
        if hasattr(torch.linalg, 'eigvalsh'):
            eigs = torch.linalg.eigvalsh(cov)
        else:
            eigs = torch.symeig(cov, eigenvectors=False)[0]
        eigs = torch.sort(eigs, descending=True).values.clamp(min=0)
        total = eigs.sum()
        if total < 1e-8:
            return 0.0
        return float((eigs[0] / total).clamp(0, 1))

# ---------------------------
# Módulos estables
# ---------------------------

class PTSymmetricLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.gain = nn.Parameter(torch.full((out_features, in_features), 0.01))
        self.loss = nn.Parameter(torch.full((out_features, in_features), 0.01))
        self.norm = nn.LayerNorm(out_features)
        nn.init.xavier_uniform_(self.weights)
        self.phase_ratio = 0.0

    def forward(self, x):
        pt_w = self.weights * (1.0 + self.gain - self.loss)
        out = F.linear(x, pt_w)
        out = self.norm(out)
        with torch.no_grad():
            g = torch.norm(self.gain)
            l = torch.norm(self.loss)
            w = torch.norm(self.weights)
            self.phase_ratio = float((torch.abs(g - l) / (w + 1e-8)).clamp(0, 2))
        return out

class TopologicalLayer(nn.Module):
    def __init__(self, in_f, out_f, density=0.15):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.register_buffer('mask', torch.ones(out_f, in_f))
        self.density = density
        self._update_mask()
        nn.init.xavier_uniform_(self.weights)

    def _update_mask(self):
        with torch.no_grad():
            rand = torch.rand_like(self.weights)
            th = torch.quantile(rand, 1 - self.density)
            self.mask = (rand < th).float()

    def forward(self, x):
        w = self.weights * self.mask
        out = F.linear(x, w, self.bias)
        return F.layer_norm(out, out.shape[1:])

class DualSystemModule(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.fast = nn.Sequential(nn.Linear(features, features // 2), nn.ReLU(), nn.Linear(features // 2, features))
        self.slow = nn.Sequential(nn.Linear(features, features), nn.ReLU(), nn.Linear(features, features))
        self.integrator = nn.Linear(features * 2, features)
        self.register_buffer('memory', torch.zeros(1, features))
        self.tau = 0.95

    def forward(self, x):
        with torch.no_grad():
            self.memory = self.tau * self.memory + (1 - self.tau) * x.mean(dim=0, keepdim=True)
        fast_out = self.fast(x)
        slow_in = x + self.memory
        slow_out = self.slow(slow_in)
        combined = torch.cat([fast_out, slow_out], dim=1)
        return self.integrator(combined)

class ConsciousnessModule(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, features), nn.ReLU(), nn.Linear(features, features))
        self.phi = 0.0
        self.threshold = 0.25  # más bajo para CIFAR-10

    def forward(self, x):
        self.phi = compute_phi_effective(x)
        if self.phi > self.threshold:
            return self.net(x)
        return x

# ---------------------------
# Modelo principal
# ---------------------------

class OmniBrainCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        # Codificador de imagen → vector
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        hidden = 128
        self.core = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden)
        )
        self.pt = PTSymmetricLayer(hidden, hidden)
        self.topo = TopologicalLayer(hidden, hidden, density=0.15)
        self.dual = DualSystemModule(hidden)
        self.conscious = ConsciousnessModule(hidden)
        self.classifier = nn.Linear(hidden, 10)

    def forward(self, x):
        h = self.encoder(x)
        h = self.core(h)
        h = self.pt(h)
        h = self.topo(h)
        h = self.dual(h)
        h = self.conscious(h)
        logits = self.classifier(h)
        return {
            'logits': logits,
            'phi': self.conscious.phi,
            'pt_ratio': self.pt.phase_ratio,
            'features': h.detach()
        }

# ---------------------------
# CIFAR-10 loaders
# ---------------------------

def get_cifar10_loaders(batch_size=32):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10('./data', train=False, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

# ---------------------------
# Evaluación
# ---------------------------

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out['logits'], y)
            loss_total += loss.item() * x.size(0)
            pred = out['logits'].argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += x.size(0)
    return correct / total, loss_total / total

# ---------------------------
# Entrenamiento principal
# ---------------------------

def main():
    print("✨ ¡OMNI BRAIN v5 - ENTRENAMIENTO CON CIFAR-10 REAL!")
    print("=" * 70)

    # Datos
    train_loader, test_loader = get_cifar10_loaders(batch_size=32)
    print(f"✅ CIFAR-10 cargado: {len(train_loader)} train batches, {len(test_loader)} test")

    # Modelo
    model = OmniBrainCIFAR().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Modelo creado: {n_params:,} parámetros | Dispositivo: {device}")

    # Optimización
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    # Historial
    history = {'epoch': [], 'train_loss': [], 'test_acc': [], 'phi': []}

    # Entrenar
    for epoch in range(8):  # 8 épocas: ~4-8 min en CPU moderna
        model.train()
        total_loss = 0.0
        total_samples = 0
        start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out['logits'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        # Evaluar
        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - start
        avg_loss = total_loss / total_samples

        # Guardar
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        history['phi'].append(out['phi'])

        # Log
        print(f"\n📊 Época {epoch+1}/8 | Tiempo: {epoch_time:.1f}s")
        print(f"   Train Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2%}")
        print(f"   Φₑ: {out['phi']:.4f} | PT-Ratio: {out['pt_ratio']:.2f}")
        print(f"   RAM: {psutil.Process().memory_info().rss / (1024**3):.2f} GB")

        scheduler.step(test_acc)

    # Guardar y graficar
    torch.save(model.state_dict(), "omni_brain_cifar10.pth")
    plot_history(history)
    print("\n🎉 ¡OMNI BRAIN ENTRENADO CON CIFAR-10 EN CPU!")

    # Demostrar inferencia
    demo_inference(model, test_loader)

def plot_history(hist):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist['epoch'], hist['train_loss'], 'b-', label='Train Loss')
    plt.xlabel('Época'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(hist['epoch'], hist['test_acc'], 'g-', label='Test Acc')
    plt.plot(hist['epoch'], hist['phi'], 'r-', label='Φₑ')
    plt.xlabel('Época'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('omni_brain_cifar10.png')
    print("📈 Gráfico guardado: omni_brain_cifar10.png")

def demo_inference(model, loader):
    model.eval()
    classes = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out['logits'].argmax(dim=1)
            for i in range(3):
                true = classes[y[i].item()]
                pred = classes[preds[i].item()]
                conf = torch.softmax(out['logits'][i], dim=0).max().item()
                print(f"\n🔍 Ejemplo {i+1}:")
                print(f"   Real: {true} | Pred: {pred} (Conf: {conf:.2%})")
                print(f"   Φₑ: {out['phi']:.4f} | {'✅' if y[i]==preds[i] else '❌'}")
            break

if __name__ == "__main__":
    main()

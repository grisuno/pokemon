#!/usr/bin/env python3
# =============================================================================
# OMNI BRAIN v6 - CIFAR-10 con PESOS LENTOS Y RÁPIDOS + DIAGNÓSTICO COMPLETO
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'
torch.set_num_threads(max(1, os.cpu_count() // 2))

# ---------------------------
# Φₑ REALISTA
# ---------------------------

def compute_phi_effective(activity: torch.Tensor) -> float:
    if activity.numel() < 100 or activity.size(0) < 5 or activity.size(1) < 3:
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
# FastSlowLinear CON DIAGNÓSTICO
# ---------------------------

class FastSlowLinear(nn.Module):
    def __init__(self, in_features, out_features, fast_lr=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fast_lr = fast_lr

        self.slow_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.slow_bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.slow_weight)

        self.register_buffer('fast_weight', torch.zeros(out_features, in_features))
        self.register_buffer('fast_bias', torch.zeros(out_features))
        self.norm = nn.LayerNorm(out_features)
        self._inputs_cache = None

    def reset_fast_weights(self):
        self.fast_weight.zero_()
        self.fast_bias.zero_()

    def update_fast_weights(self, x: torch.Tensor):
        with torch.no_grad():
            slow_out = F.linear(x, self.slow_weight, self.slow_bias)
            for i in range(min(x.size(0), 4)):  # solo unas muestras para velocidad
                self.fast_weight += self.fast_lr * torch.ger(slow_out[i], x[i])
            self.fast_bias += self.fast_lr * slow_out[:4].mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._inputs_cache is None:
            self.reset_fast_weights()
            self._inputs_cache = True

        self.update_fast_weights(x.detach())

        effective_w = self.slow_weight + self.fast_weight
        effective_b = self.slow_bias + self.fast_bias
        out = F.linear(x, effective_w, effective_b)
        out = self.norm(out)
        return out

    def end_of_batch(self):
        self._inputs_cache = None

    def get_fast_weight_norm(self):
        return float(self.fast_weight.norm().item())

# ---------------------------
# MÓDULOS (igual que antes, pero con FastSlowLinear)
# ---------------------------

class DualSystemModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fast_path = FastSlowLinear(dim, dim, fast_lr=0.4)
        self.slow_path = FastSlowLinear(dim, dim, fast_lr=0.1)
        self.integrator = nn.Linear(dim * 2, dim)
        self.register_buffer('memory', torch.zeros(1, dim))
        self.tau = 0.95

    def forward(self, x):
        with torch.no_grad():
            self.memory = self.tau * self.memory + (1 - self.tau) * x.mean(dim=0, keepdim=True)
        fast_out = self.fast_path(x)
        slow_in = x + self.memory
        slow_out = self.slow_path(slow_in)
        combined = torch.cat([fast_out, slow_out], dim=1)
        return self.integrator(combined)

class ConsciousnessModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.processor = FastSlowLinear(dim, dim, fast_lr=0.2)
        self.phi = 0.0
        self.thresh = 0.25

    def forward(self, x):
        out = self.processor(x)
        self.phi = compute_phi_effective(out)
        if self.phi > self.thresh:
            return out
        return x

class OmniBrainFastSlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.core = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.LayerNorm(128))
        self.dual = DualSystemModule(128)
        self.conscious = ConsciousnessModule(128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        h = self.encoder(x)
        h = self.core(h)
        h = self.dual(h)
        h = self.conscious(h)
        logits = self.classifier(h)
        return logits

    def reset_all_fast_weights(self):
        for module in self.modules():
            if hasattr(module, 'reset_fast_weights'):
                module.reset_fast_weights()

    def get_fast_norms(self):
        norms = []
        for name, module in self.named_modules():
            if isinstance(module, FastSlowLinear):
                norms.append(module.get_fast_weight_norm())
        return norms

# ---------------------------
# UTILIDADES DE ENTRENAMIENTO
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

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += x.size(0)
    return correct / total, total_loss / total

# ---------------------------
# ENTRENAMIENTO CON DIAGNÓSTICO
# ---------------------------

def train():
    print("✨ ¡OMNI BRAIN v6 - CIFAR-10 con PESOS LENTOS Y RÁPIDOS (DEBUG)!")
    print("=" * 70)

    train_loader, test_loader = get_cifar10_loaders(batch_size=32)
    model = OmniBrainFastSlow().to(device)
    print(f"✅ Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(8):
        model.train()
        total_loss = 0.0
        total_samples = 0
        start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.reset_all_fast_weights()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️  Pérdida inválida (NaN/Inf) – deteniendo")
                return

            loss.backward()

            # Verificar gradientes
            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            avg_grad = np.mean(grad_norms) if grad_norms else 0.0

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        train_loss = total_loss / total_samples
        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - start

        # Diagnóstico adicional
        fast_norms = model.get_fast_norms()
        avg_fast_norm = np.mean(fast_norms) if fast_norms else 0.0
        phi = model.conscious.phi

        print(f"\n📊 Época {epoch+1}/8 | Tiempo: {epoch_time:.1f}s")
        print(f"   Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")
        print(f"   Φₑ: {phi:.4f} | FastWeights Norm: {avg_fast_norm:.4f} | Avg Grad: {avg_grad:.6f}")
        print(f"   RAM: {psutil.Process().memory_info().rss / (1024**3):.2f} GB")

    torch.save(model.state_dict(), "omni_brain_cifar10_debug.pth")
    print("\n✅ Entrenamiento completado. ¡Revisa los logs para diagnóstico!")

if __name__ == "__main__":
    train()
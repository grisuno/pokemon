#!/usr/bin/env python3
# =============================================================================
# OMNI BRAIN v7 - CIFAR-10 con PESOS LENTOS/RÁPIDOS ESTABLES + LOGGING FINO
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

torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'
torch.set_num_threads(max(1, os.cpu_count() // 2))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ---------------------------
# Φₑ REALISTA (igual que antes)
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
# FASTSLOWLINEAR CORREGIDO
# ---------------------------

class FastSlowLinear(nn.Module):
    """
    Linear layer con pesos lentos (backprop) y pesos rápidos (hebbianos).
    Incluye decay temporal y normalización L2 estricta para estabilidad.
    """
    def __init__(self, in_features, out_features, fast_lr=0.005, fast_decay=0.95):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fast_lr = fast_lr
        self.fast_decay = fast_decay  # Decay temporal por forward

        # Pesos lentos (entrenables mediante backprop)
        self.slow_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.slow_bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.slow_weight)

        # Pesos rápidos (no entrenables, actualizados por regla hebbiana)
        self.register_buffer('fast_weight', torch.zeros(out_features, in_features))
        self.register_buffer('fast_bias', torch.zeros(out_features))
        self.norm = nn.LayerNorm(out_features)
        self._inputs_cache = None

    def reset_fast_weights(self):
        """Reinicia los pesos rápidos al inicio de cada batch."""
        self.fast_weight.zero_()
        self.fast_bias.zero_()

    def update_fast_weights(self, x: torch.Tensor):
        """Actualiza fast weights usando regla hebbiana con decay y normalización."""
        with torch.no_grad():
            # Aplicar decay temporal primero
            self.fast_weight *= self.fast_decay
            self.fast_bias *= self.fast_decay
            
            # Calcular output lento
            slow_out = F.linear(x, self.slow_weight, self.slow_bias)
            
            # Regla hebbiana normalizada: ΔW = η * (Y^T @ X) / N
            hebb_update = torch.mm(slow_out.t(), x) / x.size(0)
            self.fast_weight += self.fast_lr * hebb_update
            
            # Normalización L2 estricta (por fila)
            fast_weight_norm = torch.norm(self.fast_weight, dim=1, keepdim=True)
            self.fast_weight = torch.where(
                fast_weight_norm > 0.1,  # Umbral máximo por neurona
                self.fast_weight * (0.1 / fast_weight_norm),
                self.fast_weight
            )
            
            # Actualizar bias
            self.fast_bias += self.fast_lr * slow_out.mean(dim=0)
            self.fast_bias.clamp_(-0.1, 0.1)  # Más restrictivo

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
        """Limpia caché al final del batch para permitir reinicio en el siguiente."""
        self._inputs_cache = None

    def get_fast_norm(self):
        """Retorna la norma L2 de los fast weights para monitoreo homeostático."""
        return float(self.fast_weight.norm().item())


# ---------------------------
# MÓDULOS (igual que antes)
# ---------------------------

class DualSystemModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fast_path = FastSlowLinear(dim, dim, fast_lr=0.015)  # ← Reducido de 0.02 → 0.015
        self.slow_path = FastSlowLinear(dim, dim, fast_lr=0.003)  # ← Reducido de 0.005 → 0.003
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
    """Módulo de conciencia con Φₑ más estable y relevante."""
    def __init__(self, features: int):
        super().__init__()
        self.integration_net = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )
        self.phi_effective = 0.0
        self.integration_threshold = 0.2  # Más bajo y adaptable
        self.running_phi = 0.0  # Promedio móvil de Φₑ
        
    def compute_phi_effective_robust(self, activity: torch.Tensor) -> float:
        """Φₑ más robusto usando promedio móvil y ventana temporal."""
        if activity.numel() < 100 or activity.size(0) < 5 or activity.size(1) < 3:
            return 0.0
        with torch.no_grad():
            # Normalizar por neurona (columna)
            activity = activity - activity.mean(dim=0, keepdim=True)
            activity = activity / (activity.std(dim=0, keepdim=True) + 1e-8)
            # Matriz de covarianza
            cov = torch.mm(activity.t(), activity) / (activity.size(0) - 1)
            # Autovalores estables
            if hasattr(torch.linalg, 'eigvalsh'):
                eigs = torch.linalg.eigvalsh(cov)
            else:
                eigs = torch.symeig(cov, eigenvectors=False)[0]
            eigs = torch.sort(eigs, descending=True).values.clamp(min=0)
            total = eigs.sum()
            if total < 1e-8:
                return 0.0
            phi = (eigs[0] / total).item()
            # Promedio móvil para estabilidad
            self.running_phi = 0.9 * self.running_phi + 0.1 * phi
            return max(0.0, min(1.0, self.running_phi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.phi_effective = self.compute_phi_effective_robust(x)
        # Integración condicional más flexible
        if self.phi_effective > self.integration_threshold:
            return self.integration_net(x)
        # Si no se integra, aún aplica transformación ligera para no bloquear gradientes
        return x + 0.1 * self.integration_net(x)  # Residual ligero


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
            nn.Conv2d(64, 256, 3, padding=1),  # ← Aumentado de 128 → 256
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.core = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256))  # ← 256
        if use_fast_slow:
            self.dual = DualSystemModule(256)
        else:
            # Versión sin fast/slow
            self.dual = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )
            
        if use_consciousness:
            self.conscious = ConsciousnessModule(256)
        else:
            self.conscious = nn.Identity()
            
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        h = self.encoder(x)
        h = self.core(h)
        h = self.dual(h)
        h = self.conscious(h)
        return self.classifier(h)

    def reset_all_fast_weights(self):
        for module in self.modules():
            if hasattr(module, 'reset_fast_weights'):
                module.reset_fast_weights()

    def get_fast_norms(self):
        norms = [m.get_fast_norm() for m in self.modules() if hasattr(m, 'get_fast_norm')]
        return norms
# ---------------------------
# ENTRENAMIENTO CON LOGGING FINO
# ---------------------------

def get_cifar10_loaders(batch_size=32):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # ← Añadido: recorte aleatorio
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
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += x.size(0)
    return correct / total

def train():
    print("✨ ¡OMNI BRAIN v7 - CIFAR-10 CALIBRADO PARA PRECISIÓN!")
    print("=" * 70)

    train_loader, test_loader = get_cifar10_loaders(batch_size=32)
    model = OmniBrainFastSlow().to(device)
    print(f"✅ Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)  # ← Aumentado de 1e-4 → 5e-4
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):  # ← Aumentado a 10 épocas
        model.train()
        total_loss = 0.0
        total_samples = 0
        start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.reset_all_fast_weights()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            # ← Añadir pérdida de regularización (L2 implícito en AdamW, pero reforzamos)
            l2_reg = sum(torch.norm(p) for p in model.parameters())
            loss = loss + 1e-5 * l2_reg  # ← Penalización leve

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            if batch_idx % 100 == 0:
                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                fast_norms = model.get_fast_norms()
                avg_fast = np.mean(fast_norms) if fast_norms else 0.0
                grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                avg_grad = np.mean(grad_norms) if grad_norms else 0.0
                logging.info(
                    f"  Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | Φₑ: {model.conscious.phi:.4f} | "
                    f"FastNorm: {avg_fast:.2f} | AvgGrad: {avg_grad:.6f}"
                )

            for m in model.modules():
                if hasattr(m, 'end_of_batch'):
                    m.end_of_batch()

        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - start
        print(f"\n📊 Época {epoch+1}/10 | Tiempo: {epoch_time:.1f}s | Test Acc: {test_acc:.2%}")

    torch.save(model.state_dict(), "omni_brain_cifar10_calibrated.pth")
    print("\n✅ Entrenamiento calibrado completado. ¡Precisión optimizada!")

if __name__ == "__main__":
    train()

#!/usr/bin/env python3
# =============================================================================
# OMNI BRAIN v8.2 - "Optimized Baseline" PoC
# Baseline de alto rendimiento + todas las correcciones de estabilidad de v8.1-FULLY-FIXED
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import logging
from dataclasses import dataclass
import wandb

# ---------------------------
# CONFIGURACIÓN ÓPTIMA (Baseline + Fixes)
# ---------------------------
@dataclass
class Config:
    # 🔒 ABLACIÓN FLAGS – Baseline estable (el más preciso observado)
    USE_FAST_SLOW: bool = False
    USE_INTEGRATION_INDEX: bool = False
    USE_DUAL_PATHWAY: bool = False
    USE_MEMORY_BUFFER: bool = True  # Irrelevante sin DUAL_PATHWAY, pero inofensivo

    # ⚙️ HIPERPARÁMETROS – Optimizados para CPU y estabilidad
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 5e-4
    fast_lr: float = 0.005
    fast_decay: float = 0.95
    fast_update_interval: int = 10

    # 📊 LOGGING Y DATOS
    log_interval: int = 200
    num_workers: int = 1
    use_wandb: bool = True
    project_name: str = "omni-brain-PoC"

config = Config()

# Inicializar logging y wandb
if config.use_wandb:
    wandb.init(project=config.project_name, config=vars(config))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'
torch.set_num_threads(max(1, os.cpu_count() // 2))


# ---------------------------
# MÓDULOS ESTABILIZADOS (v8.1 FULLY FIXED)
# ---------------------------

class FastSlowLinear(nn.Module):
    """Módulo estabilizado – aunque no se usa en baseline, se mantiene para futura ablación."""
    def __init__(self, in_features, out_features, config: Config):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        self.slow_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.slow_bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.slow_weight)

        self.register_buffer('fast_weight', torch.zeros(out_features, in_features))
        self.register_buffer('fast_bias', torch.zeros(out_features))
        self.register_buffer('update_counter', torch.tensor(0))
        self.register_buffer('fast_weight_norm', torch.tensor(0.0))

        self.norm = nn.LayerNorm(out_features)

    def reset_fast_weights(self):
        self.fast_weight.zero_()
        self.fast_bias.zero_()
        self.fast_weight_norm.zero_()

    def update_fast_weights(self, x: torch.Tensor, slow_out: torch.Tensor):
        self.update_counter += 1
        if self.update_counter % self.config.fast_update_interval != 0:
            return
        with torch.no_grad():
            self.fast_weight.mul_(self.config.fast_decay)
            self.fast_bias.mul_(self.config.fast_decay)
            hebb_update = torch.mm(slow_out.t(), x) / x.size(0)
            self.fast_weight.add_(hebb_update, alpha=self.config.fast_lr)
            current_norm = self.fast_weight.norm()
            max_allowed_norm = 1.0
            if current_norm > max_allowed_norm:
                scale_factor = max_allowed_norm / (current_norm + 1e-8)
                self.fast_weight.mul_(scale_factor)
            self.fast_bias.add_(slow_out.mean(dim=0), alpha=self.config.fast_lr)
            self.fast_bias.clamp_(-0.2, 0.2)
            self.fast_weight_norm = self.fast_weight.norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slow_out = F.linear(x, self.slow_weight, self.slow_bias)
        if self.training and self.config.USE_FAST_SLOW:
            self.update_fast_weights(x.detach(), slow_out.detach())
        if self.config.USE_FAST_SLOW:
            effective_w = self.slow_weight + self.fast_weight
            effective_b = self.slow_bias + self.fast_bias
            out = F.linear(x, effective_w, effective_b)
        else:
            out = slow_out
        return self.norm(out)

    def get_fast_norm(self):
        return self.fast_weight_norm.item() if self.config.USE_FAST_SLOW else 0.0


class DualSystemModule(nn.Module):
    def __init__(self, dim, config: Config):
        super().__init__()
        self.config = config
        if config.USE_DUAL_PATHWAY:
            self.fast_path = FastSlowLinear(dim, dim, config)
            self.slow_path = FastSlowLinear(dim, dim, config)
            self.integrator = nn.Linear(dim * 2, dim)
        self.register_buffer('memory', torch.zeros(1, dim))
        self.tau = 0.95

    def forward(self, x):
        if self.config.USE_MEMORY_BUFFER:
            with torch.no_grad():
                self.memory = self.tau * self.memory + (1 - self.tau) * x.mean(dim=0, keepdim=True).detach()
        if not self.config.USE_DUAL_PATHWAY:
            return x
        fast_out = self.fast_path(x)
        slow_in = x + (self.memory if self.config.USE_MEMORY_BUFFER else 0)
        slow_out = self.slow_path(slow_in)
        combined = torch.cat([fast_out, slow_out], dim=1)
        return self.integrator(combined)


class IntegrationModule(nn.Module):
    def __init__(self, features: int, config: Config):
        super().__init__()
        self.config = config
        self.integration_net = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )
        self.integration_threshold = 0.2
        self.register_buffer('running_index', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ FIX: Uso de self.config para ablación correcta
        if not self.config.USE_INTEGRATION_INDEX:
            return x
        if self.training:
            idx = compute_integration_index(x)
            self.running_index = 0.9 * self.running_index + 0.1 * idx
        if self.running_index > self.integration_threshold:
            return self.integration_net(x)
        else:
            return x + 0.1 * self.integration_net(x)


def compute_integration_index(activity: torch.Tensor) -> float:
    if activity.numel() < 100 or activity.size(0) < 5:
        return 0.0
    with torch.no_grad():
        activity = activity - activity.mean(dim=0, keepdim=True)
        cov = torch.mm(activity.t(), activity) / (activity.size(0) - 1)
        try:
            _, s, _ = torch.linalg.svd(cov)
            total_var = s.sum()
            if total_var < 1e-8:
                return 0.0
            integration_ratio = (s[0] / total_var).item()
            return max(0.0, min(1.0, integration_ratio))
        except RuntimeError:
            return 0.0


# ---------------------------
# MODELO PRINCIPAL
# ---------------------------
class OmniBrainFastSlow(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.core = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        self.dual = DualSystemModule(256, config)
        self.integration = IntegrationModule(256, config)
        self.classifier = nn.Linear(256, 10)
        self.register_buffer('batch_count', torch.tensor(0))

    def forward(self, x):
        h = self.encoder(x)
        h = self.core(h)
        h = self.dual(h)
        h = self.integration(h)
        return self.classifier(h)

    def reset_all_fast_weights(self):
        for module in self.modules():
            if hasattr(module, 'reset_fast_weights'):
                module.reset_fast_weights()

    def get_fast_norms(self):
        return [m.get_fast_norm() for m in self.modules() if hasattr(m, 'get_fast_norm')]

    def get_ablation_state(self):
        return {
            "fast_slow_active": self.config.USE_FAST_SLOW,
            "dual_pathway_active": self.config.USE_DUAL_PATHWAY,
            "integration_active": self.config.USE_INTEGRATION_INDEX,
            "memory_active": self.config.USE_MEMORY_BUFFER
        }


# ---------------------------
# DATA + ENTRENAMIENTO
# ---------------------------
def get_cifar10_loaders(config: Config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10('./data', train=False, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True)
    return train_loader, test_loader


def evaluate_full(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            correct += logits.argmax(dim=1).eq(y).sum().item()
            total += x.size(0)
    return {
        "accuracy": correct / total,
        "avg_loss": np.mean(losses),
        "integration_index": 0.0  # No usado en baseline
    }


def train(config: Config):
    logger.info("🧠 OMNI BRAIN v8.2 - PoC: Optimized Baseline")
    logger.info("=" * 80)
    logger.info(f"Config: {vars(config)}")

    train_loader, test_loader = get_cifar10_loaders(config)
    model = OmniBrainFastSlow(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Modelo: {total_params:,} parámetros ({trainable_params:,} entrenables)")
    logger.info(f"📊 Ablación state: {model.get_ablation_state()}")

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ✅ FIX: Scheduler superior – CosineAnnealingWarmRestarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    metrics_history = {
        "train_loss": [], "test_acc": [], "integration_index": [],
        "fast_norm": [], "grad_norm": [], "lr": []
    }

    for epoch in range(config.epochs):
        model.train()
        # ✅ FIX: solo reset cada 10 épocas (aunque no tenga efecto en baseline)
        if config.USE_FAST_SLOW and epoch % 10 == 0:
            model.reset_all_fast_weights()

        total_loss = 0.0
        total_samples = 0
        start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            l2_reg = sum(torch.norm(p) for p in model.parameters())
            loss = loss + 1e-5 * l2_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            if batch_idx % config.log_interval == 0:
                avg_loss = total_loss / total_samples
                fast_norms = model.get_fast_norms()
                avg_fast = np.mean(fast_norms) if fast_norms else 0.0
                grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                avg_grad = np.mean(grad_norms) if grad_norms else 0.0

                logger.info(
                    f"  Epoch {epoch+1}/{config.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | FastNorm: {avg_fast:.3f} | Grad: {avg_grad:.6f}"
                )

                if config.use_wandb:
                    wandb.log({
                        "batch_loss": avg_loss,
                        "fast_norm": avg_fast,
                        "grad_norm": avg_grad,
                        "epoch": epoch
                    })

        scheduler.step()
        epoch_time = time.time() - start
        eval_metrics = evaluate_full(model, test_loader, device)

        logger.info(
            f"\n📊 ÉPOCA {epoch+1}/{config.epochs} | Tiempo: {epoch_time:.1f}s | "
            f"Test Acc: {eval_metrics['accuracy']:.2%} | Loss: {eval_metrics['avg_loss']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        metrics_history["train_loss"].append(total_loss / total_samples)
        metrics_history["test_acc"].append(eval_metrics['accuracy'])
        metrics_history["lr"].append(scheduler.get_last_lr()[0])

        if config.use_wandb:
            wandb.log({
                "epoch": epoch,
                "test_accuracy": eval_metrics['accuracy'],
                "test_loss": eval_metrics['avg_loss'],
                "epoch_time": epoch_time,
                "lr": scheduler.get_last_lr()[0]
            })

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'metrics': metrics_history
    }, "omni_brain_v82_poc.pth")

    logger.info("\n✅ PoC completado. Alta precisión esperada (~83%+).")
    return metrics_history


if __name__ == "__main__":
    train(config)
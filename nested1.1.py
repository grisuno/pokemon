#!/usr/bin/env python3
# =============================================================================
# NESTED CMS CIFAR - POC basada en Nested Learning (Google, NeurIPS 2025)
# CPU-friendly, con checkpointing por época y estudio de ablación
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
import gc
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, Any
from pathlib import Path

# ---------------------------
# HYPERPARÁMETROS CENTRALIZADOS
# ---------------------------

@dataclass
class Config:
    USE_CMS_LEVEL1: bool = True   # Delta Gradient Descent (alta frecuencia)
    USE_CMS_LEVEL2: bool = True   # Momentum/EMA (baja frecuencia)
    LEARNING_RULE: str = "dgd"    # "dgd" o "hebb"
    batch_size: int = 32
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 5e-4
    log_interval: int = 200
    num_workers: int = 1
    checkpoint_dir: str = "checkpoints_nested_cms"
    keep_last_checkpoints: int = 5

config = Config()

# ---------------------------
# SETUP BÁSICO
# ---------------------------

torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'
torch.set_num_threads(max(1, os.cpu_count() // 2))

Path(config.checkpoint_dir).mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------
# UTILIDADES DE CHECKPOINTING
# ---------------------------

def safe_serialize(obj: Any) -> Any:
    """Convierte objetos a formato serializable (evita recursión y objetos complejos)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(x) for x in obj]
    if hasattr(obj, '__dict__'):
        return safe_serialize(obj.__dict__)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return str(obj)


def save_checkpoint(epoch: int, model_state: Dict, optimizer_state: Dict,
                    config: Config, metrics: Dict, checkpoint_dir: str):
    """Guarda checkpoint de época: modelo (.pth) + metadatos (.pkl)."""
    # 1. Guardar modelo PyTorch
    model_path = os.path.join(checkpoint_dir, f"nested_cms_epoch_{epoch:02d}.pth")
    torch.save(model_state, model_path)
    logger.info(f"✅ Modelo guardado: {model_path}")

    # 2. Guardar metadatos serializables
    checkpoint_data = {
        "epoch": epoch,
        "config": asdict(config),
        "metrics": safe_serialize(metrics),
        "model_path": model_path,
        "timestamp": time.time()
    }
    meta_path = os.path.join(checkpoint_dir, f"epoch_{epoch:02d}.pkl")
    with open(meta_path, 'wb') as f:
        pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"💾 Checkpoint guardado: {meta_path}")

    # 3. Limpieza: mantener solo los últimos N checkpoints
    cleanup_old_checkpoints(checkpoint_dir, config.keep_last_checkpoints)


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int):
    """Mantiene solo los últimos `keep_last` checkpoints."""
    pkl_files = sorted(Path(checkpoint_dir).glob("epoch_*.pkl"), key=os.path.getmtime)
    if len(pkl_files) <= keep_last:
        return

    # Borrar los más antiguos
    to_delete = pkl_files[:-keep_last]
    for f in to_delete:
        epoch_num = f.stem.split('_')[-1]
        # Borrar también el .pth correspondiente
        pth_file = f.parent / f"nested_cms_epoch_{epoch_num}.pth"
        if pth_file.exists():
            pth_file.unlink()
            logger.info(f"🗑️  Modelo eliminado: {pth_file}")
        f.unlink()
        logger.info(f"🗑️  Checkpoint eliminado: {f}")


# ---------------------------
# CMS LAYER - Nested Learning (NL) compliant
# ---------------------------

class CMSLayer(nn.Module):
    def __init__(self, dim: int, config: Config):
        super().__init__()
        self.dim = dim
        self.config = config

        self.slow_weight = nn.Parameter(torch.empty(dim, dim))
        self.slow_bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.slow_weight)

        self.register_buffer("fast_weight", torch.zeros(dim, dim))
        self.register_buffer("fast_bias", torch.zeros(dim))

        self.register_buffer("slow_ema_weight", torch.zeros(dim, dim))
        self.register_buffer("slow_ema_bias", torch.zeros(dim))
        self.ema_decay = 0.95

        self.register_buffer("fast_norm", torch.tensor(0.0))
        self.register_buffer("ema_norm", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slow_out = F.linear(x, self.slow_weight, self.slow_bias)

        if self.training:
            if self.config.USE_CMS_LEVEL1:
                grad = slow_out.detach()
                if self.config.LEARNING_RULE == "dgd":
                    outer = torch.bmm(x.unsqueeze(2), x.unsqueeze(1)).mean(dim=0)
                    self.fast_weight = self.fast_weight @ (torch.eye(self.dim) - 0.01 * outer)
                    self.fast_weight -= 0.01 * grad.t() @ x / x.size(0)
                    self.fast_bias += 0.01 * grad.mean(dim=0)
                else:  # hebb
                    self.fast_weight += 0.01 * grad.t() @ x / x.size(0)
                    self.fast_bias += 0.01 * grad.mean(dim=0)

                norm = self.fast_weight.norm()
                if norm > 1.0:
                    self.fast_weight *= (1.0 / (norm + 1e-8))

            if self.config.USE_CMS_LEVEL2:
                self.slow_ema_weight = (
                    self.ema_decay * self.slow_ema_weight +
                    (1 - self.ema_decay) * self.fast_weight.detach()
                )
                self.slow_ema_bias = (
                    self.ema_decay * self.slow_ema_bias +
                    (1 - self.ema_decay) * self.fast_bias.detach()
                )

        effective_w = self.slow_weight
        effective_b = self.slow_bias

        if self.config.USE_CMS_LEVEL1:
            effective_w = effective_w + self.fast_weight
            effective_b = effective_b + self.fast_bias
        if self.config.USE_CMS_LEVEL2:
            effective_w = effective_w + self.slow_ema_weight
            effective_b = effective_b + self.slow_ema_bias

        out = F.linear(x, effective_w, effective_b)
        return F.layer_norm(out, out.shape[1:])

    def get_norms(self) -> Dict[str, float]:
        fn = self.fast_weight.norm().item() if self.config.USE_CMS_LEVEL1 else 0.0
        en = self.slow_ema_weight.norm().item() if self.config.USE_CMS_LEVEL2 else 0.0
        return {"fast_norm": fn, "ema_norm": en}


# ---------------------------
# MODELO PRINCIPAL
# ---------------------------

class NestedBrain(nn.Module):
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

        self.cms = CMSLayer(256, config)
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        h = self.encoder(x)
        h = self.cms(h)
        return self.classifier(h)

    def get_ablation_state(self):
        return {
            "cms_level1": self.config.USE_CMS_LEVEL1,
            "cms_level2": self.config.USE_CMS_LEVEL2,
            "learning_rule": self.config.LEARNING_RULE
        }

    def get_norms(self):
        return self.cms.get_norms()


# ---------------------------
# DATA LOADING
# ---------------------------

def get_cifar10_loaders(config: Config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
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
                              num_workers=config.num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=False)
    return train_loader, test_loader


# ---------------------------
# EVALUACIÓN
# ---------------------------

def evaluate(model, loader, device):
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
        "avg_loss": np.mean(losses)
    }


# ---------------------------
# ENTRENAMIENTO CON CHECKPOINTING
# ---------------------------

def train(config: Config):
    logger.info("🧠 NESTED CMS CIFAR - POC con checkpointing por época")
    logger.info("=" * 80)
    logger.info(f"Config: {asdict(config)}")

    train_loader, test_loader = get_cifar10_loaders(config)
    model = NestedBrain(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Modelo: {total_params:,} parámetros")
    logger.info(f"📊 Ablation: {model.get_ablation_state()}")

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            if batch_idx % config.log_interval == 0:
                avg_loss = total_loss / total_samples
                norms = model.get_norms()
                logger.info(
                    f"  Epoch {epoch+1}/{config.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | FastNorm: {norms['fast_norm']:.3f} | EmaNorm: {norms['ema_norm']:.3f}"
                )

        scheduler.step()
        epoch_time = time.time() - start
        eval_metrics = evaluate(model, test_loader, device)

        logger.info(
            f"\n📊 ÉPOCA {epoch+1}/{config.epochs} | Tiempo: {epoch_time:.1f}s | "
            f"Test Acc: {eval_metrics['accuracy']:.2%} | Loss: {eval_metrics['avg_loss']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # === GUARDADO POR ÉPOCA ===
        save_checkpoint(
            epoch=epoch + 1,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config=config,
            metrics={
                "train_loss": total_loss / total_samples,
                "test_acc": eval_metrics["accuracy"],
                "test_loss": eval_metrics["avg_loss"],
                "lr": scheduler.get_last_lr()[0]
            },
            checkpoint_dir=config.checkpoint_dir
        )

        gc.collect()

    logger.info("\n✅ Entrenamiento completado.")
    return eval_metrics['accuracy']


# ---------------------------
# ESTUDIO DE ABLACIÓN
# ---------------------------

def run_ablation_study():
    ablations = [
        {"name": "baseline", "USE_CMS_LEVEL1": False, "USE_CMS_LEVEL2": False},
        {"name": "fast_only", "USE_CMS_LEVEL1": True, "USE_CMS_LEVEL2": False},
        {"name": "slow_only", "USE_CMS_LEVEL1": False, "USE_CMS_LEVEL2": True},
        {"name": "dgd_full", "USE_CMS_LEVEL1": True, "USE_CMS_LEVEL2": True, "LEARNING_RULE": "dgd"},
        {"name": "hebb_full", "USE_CMS_LEVEL1": True, "USE_CMS_LEVEL2": True, "LEARNING_RULE": "hebb"},
    ]

    results = {}
    for ab in ablations:
        logger.info(f"\n{'='*80}")
        logger.info(f"🔬 ABLACIÓN: {ab['name']}")
        logger.info(f"{'='*80}")

        for k, v in ab.items():
            if k != "name":
                setattr(config, k, v)

        acc = train(config)
        results[ab["name"]] = acc
        logger.info(f" → Precisión final: {acc:.2%}")

    logger.info(f"\n📊 RESULTADOS FINALES:")
    for name, acc in results.items():
        logger.info(f"  {name:15s}: {acc:.2%}")

    with open("nested_cms_ablation_results.txt", "w") as f:
        f.write("NESTED CMS CIFAR - ABLATION RESULTS\n")
        f.write("="*50 + "\n\n")
        for name, acc in results.items():
            f.write(f"{name:15s}: {acc:.4f}\n")

    return results


if __name__ == "__main__":
    # Para una sola ejecución rápida:
    train(config)

    # Para estudio completo (descomentar cuando quieras):
    # run_ablation_study()
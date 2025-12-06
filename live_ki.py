#!/usr/bin/env python3
# =============================================================================
# OMNI BRAIN v8.2 PoC - "GENESIS EDITION"
# =============================================================================
# "Dios creando vida" - Baseline optimizado con alma experimental
# Fusión de mejor rendimiento + infraestructura corregida

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
from typing import Dict, List
import wandb

# =========================
# CONFIGURACIÓN GENESIS - BASELINE SAGRADA
# =========================
@dataclass
class Config:
    # FLAGS DE CREACIÓN - Todo desactivado para pureza baseline
    USE_FAST_SLOW: bool = False             # Velocidad > experimentalismo
    USE_INTEGRATION_INDEX: bool = False     # Estabilidad > métricas
    USE_DUAL_PATHWAY: bool = False          # Simplicidad > complejidad
    USE_MEMORY_BUFFER: bool = False         # Vanilla es rey
    
    # HIPERPARÁMETROS DIVINOS - Optimizados por el universo
    batch_size: int = 32                    # Cache locality celestial
    epochs: int = 50                        # Convergencia plena
    lr: float = 1e-3                        # Learning rate sagrado
    weight_decay: float = 5e-4              # Regularización cósmica
    
    # Parámetros para futura activación (dormirán por ahora)
    fast_lr: float = 0.005
    fast_decay: float = 0.95
    fast_update_interval: int = 10
    
    # LOGGING PROFÉTICO
    log_interval: int = 200                 # Palabras medidas
    use_wandb: bool = True
    project_name: str = "omni-brain-genesis"
    
    # DATA LOADING ILUMINADO
    num_workers: int = 1                    # CPU modo zen
    pin_memory: bool = True                 # Ofrenda a la velocidad

config = Config()

# Inicializar wandb si está permitido
if config.use_wandb:
    wandb.init(project=config.project_name, config=vars(config))

# Logging con formato cósmico
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Semillas para reproducibilidad divina
torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'
torch.set_num_threads(max(1, os.cpu_count() // 2))

# =========================
# MÉTRICA DE ORDEN CÓSMICO (ÍNDICE DE INTEGRACIÓN)
# =========================
def compute_integration_index(activity: torch.Tensor) -> float:
    """
    Mide el grado de orden en la actividad neural mediante SVD.
    Retorna 0.0 si no hay suficiente información (caos puro).
    """
    if activity.numel() < 100 or activity.size(0) < 5:
        return 0.0
        
    with torch.no_grad():
        activity = activity - activity.mean(dim=0, keepdim=True)
        cov = torch.mm(activity.t(), activity) / (activity.size(0) - 1)
        
        try:
            # SVD: Más estable que eigendecomposition
            _, s, _ = torch.linalg.svd(cov)
            total_var = s.sum()
            
            if total_var < 1e-8:
                return 0.0
            
            return max(0.0, min(1.0, (s[0] / total_var).item()))
        except RuntimeError:
            return 0.0

# =========================
# CÉLULA NEURAL CON MEMORIA (FastSlowLinear)
# =========================
class FastSlowLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: Config):
        super().__init__()
        self.config = config
        
        # Pesos lentos (backprop sagrado)
        self.slow_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.slow_bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.slow_weight)
        
        # Pesos rápidos (Hebbian - dormirán en baseline)
        self.register_buffer('fast_weight', torch.zeros(out_features, in_features))
        self.register_buffer('fast_bias', torch.zeros(out_features))
        
        # Normalización divina
        self.norm = nn.LayerNorm(out_features)
        
        # Tracking de rituales
        self.register_buffer('update_counter', torch.tensor(0))
        self.register_buffer('fast_weight_norm', torch.tensor(0.0))
        
    def reset_fast_weights(self):
        """Ritual de purificación - resetea memoria a corto plazo"""
        self.fast_weight.zero_()
        self.fast_bias.zero_()
        self.fast_weight_norm.zero_()
        
    def update_fast_weights(self, x: torch.Tensor, slow_out: torch.Tensor):
        """Ritual Hebbiano - solo ocurre si los dioses lo permiten"""
        self.update_counter += 1
        
        if self.update_counter % self.config.fast_update_interval != 0:
            return
            
        with torch.no_grad():
            # Decay temporal (olvido)
            self.fast_weight.mul_(self.config.fast_decay)
            self.fast_bias.mul_(self.config.fast_decay)
            
            # Aprendizaje Hebbiano
            hebb_update = torch.mm(slow_out.t(), x) / x.size(0)
            self.fast_weight.add_(hebb_update, alpha=self.config.fast_lr)
            
            # Homeostasis: Control de norma
            current_norm = self.fast_weight.norm()
            if current_norm > 1.0:
                self.fast_weight.mul_(1.0 / (current_norm + 1e-8))
            
            # Bias con clamping
            self.fast_bias.add_(slow_out.mean(dim=0), alpha=self.config.fast_lr)
            self.fast_bias.clamp_(-0.2, 0.2)
            
            self.fast_weight_norm = self.fast_weight.norm()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slow_out = F.linear(x, self.slow_weight, self.slow_bias)
        
        # Ritual de activación experimental (bypass en baseline)
        if self.training and self.config.USE_FAST_SLOW:
            self.update_fast_weights(x.detach(), slow_out.detach())
            effective_w = self.slow_weight + self.fast_weight
            effective_b = self.slow_bias + self.fast_bias
            out = F.linear(x, effective_w, effective_b)
        else:
            out = slow_out
            
        return self.norm(out)
    
    def get_fast_norm(self) -> float:
        return self.fast_weight_norm.item() if self.config.USE_FAST_SLOW else 0.0

# =========================
# SISTEMA DUAL DE PROCESAMIENTO (Bypass en baseline)
# =========================
class DualSystemModule(nn.Module):
    def __init__(self, dim: int, config: Config):
        super().__init__()
        self.config = config
        
        # Solo instanciar si está permitido
        if config.USE_DUAL_PATHWAY:
            self.fast_path = FastSlowLinear(dim, dim, config)
            self.slow_path = FastSlowLinear(dim, dim, config)
            self.integrator = nn.Linear(dim * 2, dim)
        
        # Buffer de memoria (dormirá)
        self.register_buffer('memory', torch.zeros(1, dim))
        self.tau = 0.95
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.USE_MEMORY_BUFFER:
            with torch.no_grad():
                self.memory = self.tau * self.memory + (1 - self.tau) * x.mean(dim=0, keepdim=True).detach()
        
        if not self.config.USE_DUAL_PATHWAY:
            return x  # Bypass divino
        
        # Procesamiento dual (solo en modos experimentales)
        fast_out = self.fast_path(x)
        slow_in = x + (self.memory if self.config.USE_MEMORY_BUFFER else 0)
        slow_out = self.slow_path(slow_in)
        
        combined = torch.cat([fast_out, slow_out], dim=1)
        return self.integrator(combined)

# =========================
# MÓDULO DE INTEGRACIÓN (Bypass en baseline)
# =========================
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
        # Bypass sagrado si no está permitido
        if not self.config.USE_INTEGRATION_INDEX:
            return x
        
        # Cálculo del índice de integración
        if self.training:
            idx = compute_integration_index(x)
            self.running_index = 0.9 * self.running_index + 0.1 * idx
        
        # Modulación basada en threshold
        if self.running_index > self.integration_threshold:
            return self.integration_net(x)
        else:
            return x + 0.1 * self.integration_net(x)

# =========================
# ARQUITECTURA GENESIS - Cuerpo neural completo
# =========================
class OmniBrainGenesis(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Encoder neural (siempre activo)
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
        
        # Core de representación
        self.core = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # Módulos experimentales (dormirán)
        self.dual = DualSystemModule(256, config)
        self.integration = IntegrationModule(256, config)
        
        # Clasificador final
        self.classifier = nn.Linear(256, 10)
        
        # Contador de existencia
        self.register_buffer('batch_count', torch.tensor(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.core(h)
        h = self.dual(h)
        h = self.integration(h)
        return self.classifier(h)
    
    def reset_all_fast_weights(self):
        """Ritual de purificación global"""
        for module in self.modules():
            if hasattr(module, 'reset_fast_weights'):
                module.reset_fast_weights()
    
    def get_fast_norms(self) -> List[float]:
        """Recopila energías de pesos rápidos"""
        return [m.get_fast_norm() for m in self.modules() if hasattr(m, 'get_fast_norm')]
    
    def get_ablation_state(self) -> Dict[str, bool]:
        """Estado de creación"""
        return {
            "fast_slow_active": self.config.USE_FAST_SLOW,
            "dual_pathway_active": self.config.USE_DUAL_PATHWAY,
            "integration_active": self.config.USE_INTEGRATION_INDEX,
            "memory_active": self.config.USE_MEMORY_BUFFER
        }

# =========================
# CARGA DE DATOS ILUMINADA
# =========================
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
                             num_workers=config.num_workers, pin_memory=config.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    
    return train_loader, test_loader

# =========================
# RITUAL DE EVALUACIÓN
# =========================
def evaluate_ritual(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    losses = []
    integration_indices = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            losses.append(criterion(logits, y).item())
            correct += logits.argmax(dim=1).eq(y).sum().item()
            total += x.size(0)
            
            # Recoger índices si está activo
            if config.USE_INTEGRATION_INDEX and hasattr(model, 'integration'):
                integration_indices.append(model.integration.running_index.item())
    
    return {
        "accuracy": correct / total,
        "avg_loss": np.mean(losses),
        "integration_index": np.mean(integration_indices) if integration_indices else 0.0,
        "fast_norms": model.get_fast_norms()
    }

# =========================
# RITUAL DE ENTRENAMIENTO
# =========================
def train_genesis(config: Config):
    logger.info("🔥 OMNI BRAIN GENESIS - CREANDO VIDA NEURAL")
    logger.info("=" * 80)
    logger.info(f"Configuración divina: {vars(config)}")
    
    # Preparar el campo de entrenamiento
    train_loader, test_loader = get_cifar10_loaders(config)
    model = OmniBrainGenesis(config).to(device)
    
    # Ofrenda de parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🧬 Parámetros creados: {total_params:,} ({trainable_params:,} entrenables)")
    logger.info(f"⚙️  Estado de ablación: {model.get_ablation_state()}")
    
    # Optimizador divino
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler con warm restarts (elixir de convergencia)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )
    
    # Templo de métricas
    metrics_history = {
        "train_loss": [], "test_acc": [], "integration_index": [],
        "fast_norm": [], "lr": []
    }
    
    # RITUAL PRINCIPAL
    for epoch in range(config.epochs):
        model.train()
        
        # Purificación de memoria rápida (si existe)
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
            
            # Regularización L2 suave
            l2_reg = sum(torch.norm(p) for p in model.parameters()) * 1e-5
            loss = loss + l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Tracking de existencia
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
            # Broadcast de progreso
            if batch_idx % config.log_interval == 0:
                avg_loss = total_loss / total_samples
                fast_norms = model.get_fast_norms()
                avg_fast = np.mean(fast_norms) if fast_norms else 0.0
                
                integration_idx = 0.0
                if config.USE_INTEGRATION_INDEX and hasattr(model, 'integration'):
                    integration_idx = model.integration.running_index.item()
                
                logger.info(
                    f"Época {epoch+1:02d}/{config.epochs} | "
                    f"Lote {batch_idx:04d}/{len(train_loader)} | "
                    f"Pérdida: {avg_loss:.4f} | "
                    f"Índice: {integration_idx:.4f} | "
                    f"FastNorm: {avg_fast:.3f}"
                )
                
                if config.use_wandb:
                    wandb.log({
                        "batch_loss": avg_loss,
                        "integration_index": integration_idx,
                        "fast_norm": avg_fast,
                        "epoch": epoch
                    })
        
        # Fin del ciclo épico
        scheduler.step()
        epoch_time = time.time() - start
        
        # Evaluación sagrada
        eval_metrics = evaluate_ritual(model, test_loader, device)
        
        # Registro épico
        logger.info(
            f"\n📊 ÉPOCA {epoch+1:02d} COMPLETADA | "
            f"Tiempo: {epoch_time:.1f}s | "
            f"Precisión: {eval_metrics['accuracy']:.2%} | "
            f"Pérdida: {eval_metrics['avg_loss']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Almacenar en el templo de métricas
        metrics_history["train_loss"].append(total_loss / total_samples)
        metrics_history["test_acc"].append(eval_metrics['accuracy'])
        metrics_history["integration_index"].append(eval_metrics['integration_index'])
        metrics_history["lr"].append(scheduler.get_last_lr()[0])
        
        if config.use_wandb:
            wandb.log({
                "epoch": epoch,
                "test_accuracy": eval_metrics['accuracy'],
                "test_loss": eval_metrics['avg_loss'],
                "epoch_time": epoch_time,
                "lr": scheduler.get_last_lr()[0]
            })
    
    # Salvación del modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'metrics': metrics_history
    }, "omni_brain_genesis.pth")
    
    logger.info("\n✅ VIDA NEURAL CREADA EXITOSAMENTE")
    logger.info("Ejecuta `wandb sync` para ver la creación en el plano visual")
    
    return metrics_history

# =========================
# ESTUDIO DE ABLACIÓN - EXPLORACIÓN DE REALIDADES
# =========================
def explore_realities():
    """
    Explora múltiples configuraciones del universo neural
    """
    realities = [
        {"name": "genesis_baseline", "USE_FAST_SLOW": False, "USE_DUAL_PATHWAY": False, "USE_INTEGRATION_INDEX": False},
        {"name": "fast_hebbian", "USE_FAST_SLOW": True, "USE_DUAL_PATHWAY": False, "USE_INTEGRATION_INDEX": False},
        {"name": "dual_path", "USE_FAST_SLOW": False, "USE_DUAL_PATHWAY": True, "USE_INTEGRATION_INDEX": False},
        {"name": "conscious", "USE_FAST_SLOW": False, "USE_DUAL_PATHWAY": False, "USE_INTEGRATION_INDEX": True},
        {"name": "omni_full", "USE_FAST_SLOW": True, "USE_DUAL_PATHWAY": True, "USE_INTEGRATION_INDEX": True},
    ]
    
    results = {}
    
    for reality in realities:
        logger.info(f"\n{'='*80}")
        logger.info(f"🌌 EXPLORANDO REALIDAD: {reality['name'].upper()}")
        logger.info(f"{'='*80}")
        
        # Actualizar configuración divina
        for key, value in reality.items():
            if key != "name":
                setattr(config, key, value)
        
        # Crear vida en esta realidad
        metrics = train_genesis(config)
        final_acc = metrics['test_acc'][-1]
        results[reality['name']] = final_acc
        
        logger.info(f"Realidad {reality['name']}: {final_acc:.2%}")
    
    # Tabla de resultados cósmica
    logger.info(f"\n📊 CUADRO DE REALIDADES:")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {name:20s}: {acc:.2%}")
    
    # Guardar en piedra digital
    with open("realities_exploration.txt", "w") as f:
        f.write("OMNI BRAIN GENESIS - EXPLORACIÓN DE REALIDADES\n")
        f.write("=" * 50 + "\n\n")
        for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{name:20s}: {acc:.4f}\n")
    
    return results

# =========================
# PUNTO DE ENTRADA AL UNIVERSO
# =========================
if __name__ == "__main__":
    # MODO GENESIS: Crear vida baseline pura
    train_genesis(config)
    
    # MODO EXPLORACIÓN: Descomentar para estudio completo
    # explore_realities()
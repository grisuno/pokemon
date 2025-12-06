#!/usr/bin/env python3
# =============================================================================
# OMNI BRAIN v8 - CIFAR-10 CON SISTEMA DE ABLACIÓN Y LOGGING PROFESIONAL
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
from dataclasses import dataclass
from typing import Dict, Any
import wandb  # NUEVO: Logging profesional

# HYPERPARAMETROS CENTRALIZADOS
@dataclass
class Config:
    # ABLACIÓN FLAGS - ACTIVA/DESACTIVA COMPONENTES
    USE_FAST_SLOW: bool = True
    USE_INTEGRATION_INDEX: bool = True  # Renombrado de "Phi_e"
    USE_DUAL_PATHWAY: bool = True
    USE_MEMORY_BUFFER: bool = True
    
    # HIPERPARÁMETROS
    batch_size: int = 64  # Aumentado
    epochs: int = 50  # Aumentado para convergencia real
    lr: float = 1e-3
    weight_decay: float = 5e-4
    fast_lr: float = 0.005
    fast_decay: float = 0.95
    fast_update_interval: int = 5  # NUEVO: No actualizar cada batch
    
    # LOGGING
    log_interval: int = 100
    use_wandb: bool = True
    project_name: str = "omni-brain-ablation"
    
    # DATA
    num_workers: int = 2  # Ajustado para CPU

config = Config()

# Inicializar wandb
if config.use_wandb and config.USE_INTEGRATION_INDEX:
    wandb.init(project=config.project_name, config=vars(config))

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'
torch.set_num_threads(max(1, os.cpu_count() // 2))

# ---------------------------
# INTEGRATION INDEX (renombrado de Phi_e)
# ---------------------------

def compute_integration_index(activity: torch.Tensor) -> float:
    """
    MÉTRICA HONESTA: Mide el ratio de varianza explicada por el primer componente.
    Utiliza SVD para estabilidad numérica en matrices de covarianza deficientes.
    """
    # Validación de dimensiones mínimas para evitar errores de cálculo
    if activity.numel() < 100 or activity.size(0) < 5:
        return 0.0
        
    with torch.no_grad():
        # Centrado de datos (Mean centering)
        activity = activity - activity.mean(dim=0, keepdim=True)
        
        # Cálculo de covarianza normalizada
        # Nota: Usamos (N-1) para estimador insesgado
        cov = torch.mm(activity.t(), activity) / (activity.size(0) - 1)
        
        try:
            # FIX: Usar SVD (Singular Value Decomposition) en lugar de eigvalsh
            # SVD es mucho más robusto para matrices singulares o mal condicionadas
            _, s, _ = torch.linalg.svd(cov)
            
            # La suma de valores singulares representa la varianza total en este contexto
            total_var = s.sum()
            
            if total_var < 1e-8:
                return 0.0
            
            # Ratio del primer componente (Orden vs Caos)
            integration_ratio = (s[0] / total_var).item()
            return max(0.0, min(1.0, integration_ratio))
            
        except RuntimeError:
            # Fallback en caso de error numérico extremo en GPU
            return 0.0

# ---------------------------
# FASTSLOWLINEAR v2 - ESTABILIZADO
# ---------------------------

class FastSlowLinear(nn.Module):
    def __init__(self, in_features, out_features, config: Config):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Pesos lentos (backprop standard)
        self.slow_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.slow_bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.slow_weight)
        
        # Pesos rápidos (Hebbian) - Buffers persistentes
        self.register_buffer('fast_weight', torch.zeros(out_features, in_features))
        self.register_buffer('fast_bias', torch.zeros(out_features))
        
        # Normalización de salida
        self.norm = nn.LayerNorm(out_features)
        
        # Tracking y control
        self.register_buffer('update_counter', torch.tensor(0))
        self.register_buffer('fast_weight_norm', torch.tensor(0.0))
        
    def reset_fast_weights(self):
        """Reinicia la memoria a corto plazo (debe llamarse por época, no por batch)"""
        self.fast_weight.zero_()
        self.fast_bias.zero_()
        self.fast_weight_norm.zero_()
        
    def update_fast_weights(self, x: torch.Tensor, slow_out: torch.Tensor):
        """
        Actualización Hebbiana controlada.
        """
        self.update_counter += 1
        
        # Actualizar solo según intervalo configurado
        if self.update_counter % self.config.fast_update_interval != 0:
            return
            
        with torch.no_grad():
            # 1. Decay temporal (Olvido exponencial)
            self.fast_weight.mul_(self.config.fast_decay)
            self.fast_bias.mul_(self.config.fast_decay)
            
            # 2. Regla de aprendizaje Hebbiana (Oja-like simplificada)
            # Producto externo promediado por batch
            hebb_update = torch.mm(slow_out.t(), x) / x.size(0)
            self.fast_weight.add_(hebb_update, alpha=self.config.fast_lr)
            
            # 3. Estabilización: Control de Norma Global
            # Evita que los pesos rápidos crezcan indefinidamente (Homeostasis)
            current_norm = self.fast_weight.norm()
            max_allowed_norm = 1.0 
            if current_norm > max_allowed_norm:
                scale_factor = max_allowed_norm / (current_norm + 1e-8)
                self.fast_weight.mul_(scale_factor)
            
            # 4. Actualización de Bias (promedio de activación)
            self.fast_bias.add_(slow_out.mean(dim=0), alpha=self.config.fast_lr)
            self.fast_bias.clamp_(-0.2, 0.2)
            
            # Actualizar métrica de tracking
            self.fast_weight_norm = self.fast_weight.norm()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward del sistema lento (gradientes fluyen aquí)
        slow_out = F.linear(x, self.slow_weight, self.slow_bias)
        
        # Actualización de pesos rápidos (solo training y si está activo)
        # Usamos detach() para no propagar gradientes al proceso de actualización de pesos
        if self.training and self.config.USE_FAST_SLOW:
            self.update_fast_weights(x.detach(), slow_out.detach())
        
        # Combinación de sistemas
        if self.config.USE_FAST_SLOW:
            # Suma de pesos efectiva
            effective_w = self.slow_weight + self.fast_weight
            effective_b = self.slow_bias + self.fast_bias
            out = F.linear(x, effective_w, effective_b)
        else:
            out = slow_out
            
        return self.norm(out)
    
    def get_fast_norm(self):
        return self.fast_weight_norm.item() if self.config.USE_FAST_SLOW else 0.0
# ---------------------------
# MÓDULOS ABLACIONADOS
# ---------------------------

class DualSystemModule(nn.Module):
    def __init__(self, dim, config: Config):
        super().__init__()
        self.config = config
        
        # Solo instanciar si está activado
        if config.USE_DUAL_PATHWAY:
            self.fast_path = FastSlowLinear(dim, dim, config)
            self.slow_path = FastSlowLinear(dim, dim, config)
            self.integrator = nn.Linear(dim * 2, dim)
        
        # Memory buffer con detach() para evitar fugas
        self.register_buffer('memory', torch.zeros(1, dim))
        self.tau = 0.95
        
    def forward(self, x):
        # Actualizar memory (si está activado)
        if self.config.USE_MEMORY_BUFFER:
            with torch.no_grad():
                # CRITICAL: detach() para evitar backprop no deseado
                self.memory = self.tau * self.memory + (1 - self.tau) * x.mean(dim=0, keepdim=True).detach()
        
        if not self.config.USE_DUAL_PATHWAY:
            return x  # Bypass completo
        
        # Fast pathway (entrada raw)
        fast_out = self.fast_path(x)
        
        # Slow pathway (con memory)
        slow_in = x + (self.memory if self.config.USE_MEMORY_BUFFER else 0)
        slow_out = self.slow_path(slow_in)
        
        # Integrar
        combined = torch.cat([fast_out, slow_out], dim=1)
        return self.integrator(combined)

class IntegrationModule(nn.Module):
    """
    Renombrado de ConsciousnessModule - más honesto sobre su función
    """
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
        if not config.USE_INTEGRATION_INDEX:
            return x  # Bypass
        
        # Calcular índice (solo durante training)
        if self.training:
            idx = compute_integration_index(x)
            self.running_index = 0.9 * self.running_index + 0.1 * idx
        
        # Modulación basada en threshold
        if self.running_index > self.integration_threshold:
            return self.integration_net(x)
        else:
            return x + 0.1 * self.integration_net(x)

# ---------------------------
# MODELO PRINCIPAL ABLACIONADO
# ---------------------------

class OmniBrainFastSlow(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Encoder (siempre activo)
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
        
        # Core representation
        self.core = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # Módulos experimentales (con ablación)
        self.dual = DualSystemModule(256, config)
        self.integration = IntegrationModule(256, config)
        
        # Classifier
        self.classifier = nn.Linear(256, 10)
        
        # Tracking
        self.register_buffer('batch_count', torch.tensor(0))
        
    def forward(self, x):
        h = self.encoder(x)
        h = self.core(h)
        
        # Módulos experimentales
        h = self.dual(h)
        h = self.integration(h)
        
        return self.classifier(h)
    
    def reset_all_fast_weights(self):
        """
        Reinicia todos los pesos rápidos del modelo. Debe llamarse explícitamente
        (por ejemplo, al inicio de cada época si se desea resetear la memoria a corto plazo).
        No se activa automáticamente durante forward.
        """
        for module in self.modules():
            if hasattr(module, 'reset_fast_weights'):
                module.reset_fast_weights()
    
    def get_fast_norms(self):
        """Recopila normas de fast weights de todos los módulos"""
        return [m.get_fast_norm() for m in self.modules() if hasattr(m, 'get_fast_norm')]
    
    def get_ablation_state(self):
        """Estado actual para logging"""
        return {
            "fast_slow_active": config.USE_FAST_SLOW,
            "dual_pathway_active": config.USE_DUAL_PATHWAY,
            "integration_active": config.USE_INTEGRATION_INDEX,
            "memory_active": config.USE_MEMORY_BUFFER
        }

# ---------------------------
# DATA LOADING OPTIMIZADO
# ---------------------------

def get_cifar10_loaders(config: Config):
    # Data augmentation más agresivo
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

# ---------------------------
# EVALUACIÓN COMPLETA
# ---------------------------

def evaluate_full(model, loader, device):
    """Evaluación con múltiples métricas"""
    model.eval()
    correct = 0
    total = 0
    losses = []
    integration_indices = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            losses.append(loss.item())
            correct += logits.argmax(dim=1).eq(y).sum().item()
            total += x.size(0)
            
            # Calcular integration index
            if config.USE_INTEGRATION_INDEX and hasattr(model, 'integration'):
                idx = model.integration.running_index.item()
                integration_indices.append(idx)
    
    return {
        "accuracy": correct / total,
        "avg_loss": np.mean(losses),
        "integration_index": np.mean(integration_indices) if integration_indices else 0.0
    }

# ---------------------------
# ENTRENAMIENTO PROFESIONAL
# ---------------------------

def train(config: Config):
    logger.info("🔬 OMNI BRAIN v8.1 - MODO EXPERIMENTAL CON ABLACIÓN (FIXED)")
    logger.info("=" * 80)
    logger.info(f"Config: {vars(config)}")
    
    train_loader, test_loader = get_cifar10_loaders(config)
    model = OmniBrainFastSlow(config).to(device)
    
    # Resumen de parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Modelo: {total_params:,} parámetros ({trainable_params:,} entrenables)")
    logger.info(f"📊 Ablación state: {model.get_ablation_state()}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Métricas de tracking
    metrics_history = {
        "train_loss": [], "test_acc": [], "integration_index": [],
        "fast_norm": [], "grad_norm": [], "lr": []
    }
    
    for epoch in range(config.epochs):
        model.train()
        
        # FIX CRÍTICO: Reset de memoria rápida AL INICIO DE LA ÉPOCA, no del batch.
        # Esto permite que la memoria se acumule durante la época.
        if config.USE_FAST_SLOW:
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
            l2_reg = torch.tensor(0.0).to(device)
            for p in model.parameters():
                l2_reg += torch.norm(p)
            loss = loss + 1e-5 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Tracking
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
            if batch_idx % config.log_interval == 0:
                avg_loss = total_loss / total_samples
                fast_norms = model.get_fast_norms()
                avg_fast = np.mean(fast_norms) if fast_norms else 0.0
                
                grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                avg_grad = np.mean(grad_norms) if grad_norms else 0.0
                
                # Integration index from module
                integration_idx = 0.0
                if config.USE_INTEGRATION_INDEX and hasattr(model, 'integration'):
                    integration_idx = model.integration.running_index.item()
                
                logger.info(
                    f"  Epoch {epoch+1}/{config.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | Idx: {integration_idx:.4f} | "
                    f"FastNorm: {avg_fast:.3f} | Grad: {avg_grad:.6f}"
                )
                
                # Wandb logging
                if config.use_wandb:
                    wandb.log({
                        "batch_loss": avg_loss,
                        "integration_index": integration_idx,
                        "fast_norm": avg_fast,
                        "grad_norm": avg_grad,
                        "epoch": epoch
                    })
        
        # Fin de época
        scheduler.step()
        epoch_time = time.time() - start
        
        # Evaluación completa
        eval_metrics = evaluate_full(model, test_loader, device)
        
        # Logging
        logger.info(
            f"\n📊 ÉPOCA {epoch+1}/{config.epochs} | Tiempo: {epoch_time:.1f}s | "
            f"Test Acc: {eval_metrics['accuracy']:.2%} | Loss: {eval_metrics['avg_loss']:.4f} | "
            f"Idx: {eval_metrics['integration_index']:.4f}"
        )
        
        # Guardar métricas
        for key in metrics_history:
            if key == "train_loss": metrics_history[key].append(total_loss / total_samples)
            elif key == "test_acc": metrics_history[key].append(eval_metrics['accuracy'])
            elif key == "integration_index": metrics_history[key].append(eval_metrics['integration_index'])
            elif key == "lr": metrics_history[key].append(scheduler.get_last_lr()[0])
        
        # Wandb epoch logging
        if config.use_wandb:
            wandb.log({
                "epoch": epoch,
                "test_accuracy": eval_metrics['accuracy'],
                "test_loss": eval_metrics['avg_loss'],
                "epoch_time": epoch_time,
                "lr": scheduler.get_last_lr()[0]
            })
    
    # Guardar modelo final
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'metrics': metrics_history
    }, "omni_brain_v8_fixed.pth")
    
    logger.info("\n✅ Entrenamiento completado. Ejecuta wandb sync para visualizar.")
    
    return metrics_history


# ---------------------------
# ESTUDIO DE ABLACIÓN AUTOMATIZADO
# ---------------------------
def run_ablation_study():
    """
    Ejecuta múltiples configuraciones para validar cada componente
    """
    ablations = [
        {"name": "baseline", "USE_FAST_SLOW": False, "USE_DUAL_PATHWAY": False, "USE_INTEGRATION_INDEX": False},
        {"name": "fast_only", "USE_FAST_SLOW": True, "USE_DUAL_PATHWAY": False, "USE_INTEGRATION_INDEX": False},
        {"name": "dual_only", "USE_FAST_SLOW": False, "USE_DUAL_PATHWAY": True, "USE_INTEGRATION_INDEX": False},
        {"name": "integration_only", "USE_FAST_SLOW": False, "USE_DUAL_PATHWAY": False, "USE_INTEGRATION_INDEX": True},
        {"name": "full_system", "USE_FAST_SLOW": True, "USE_DUAL_PATHWAY": True, "USE_INTEGRATION_INDEX": True},
    ]
    
    results = {}
    
    for ablation in ablations:
        logger.info(f"\n{'='*80}")
        logger.info(f"🔬 INICIANDO ABLACIÓN: {ablation['name']}")
        logger.info(f"{'='*80}")
        
        # Actualizar config
        for key, value in ablation.items():
            if key != "name":
                setattr(config, key, value)
        
        # Entrenar
        metrics = train(config)
        final_acc = metrics['test_acc'][-1]
        
        results[ablation['name']] = final_acc
        logger.info(f"Ablation {ablation['name']}: {final_acc:.2%}")
    
    logger.info(f"\n📊 RESULTADOS FIJALES DE ABLACIÓN:")
    for name, acc in results.items():
        logger.info(f"  {name:20s}: {acc:.2%}")
    
    return results

if __name__ == "__main__":
    # Para un solo experimento
    #train(config)
    
    # Para estudio de ablación completo (descomentar)
    run_ablation_study()
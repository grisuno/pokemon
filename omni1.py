#!/usr/bin/env python3
# =============================================================================
# OMNI BRAIN V8 - VERSIÓN MEJORADA CON CORRECCIONES CRÍTICAS
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import time
import logging
import json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'
torch.set_num_threads(max(1, os.cpu_count() // 2))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =============================================================================
# MEJORA 1: FastSlowLinear con mayor estabilidad
# =============================================================================

class FastSlowLinear(nn.Module):
    """Linear layer con pesos hebbianos mejorados y mayor capacidad de adaptación."""
    def __init__(self, in_features, out_features, fast_lr=0.01, fast_decay=0.95):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fast_lr = fast_lr
        self.fast_decay = fast_decay
        self.slow_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.slow_bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.slow_weight)
        self.register_buffer('fast_weight', torch.randn(out_features, in_features) * 0.01)
        self.register_buffer('fast_bias', torch.zeros(out_features))
        self.norm = nn.LayerNorm(out_features)
        self._update_count = 0
    def reset_fast_weights(self):
        self.fast_weight *= 0.5
        self.fast_bias *= 0.5
    def update_fast_weights(self, x: torch.Tensor):
        with torch.no_grad():
            self.fast_weight *= self.fast_decay
            self.fast_bias *= self.fast_decay
            slow_out = F.linear(x, self.slow_weight, self.slow_bias)
            x_normalized = F.normalize(x, dim=1)
            slow_out_normalized = F.normalize(slow_out, dim=1)
            hebb_update = torch.mm(slow_out_normalized.t(), x_normalized) / x.size(0)
            self.fast_weight += self.fast_lr * hebb_update
            weight_norm = torch.norm(self.fast_weight, dim=1, keepdim=True)
            max_norm = 0.25  # FIX: aumentado de 0.2 → 0.25
            self.fast_weight = torch.where(
                weight_norm > max_norm,
                self.fast_weight * (max_norm / (weight_norm + 1e-8)),
                self.fast_weight
            )
            self.fast_bias += self.fast_lr * slow_out_normalized.mean(dim=0)
            self.fast_bias.clamp_(-0.2, 0.2)
            self._update_count += 1
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.update_fast_weights(x.detach())
        effective_w = self.slow_weight + self.fast_weight
        effective_b = self.slow_bias + self.fast_bias
        out = F.linear(x, effective_w, effective_b)
        out = self.norm(out)
        return out
    def get_fast_norm(self):
        return float(self.fast_weight.norm().item())


# =============================================================================
# MEJORA 2: ConsciousnessModule con Φₑ más sensible
# =============================================================================

class ConsciousnessModule(nn.Module):
    """Módulo de consciencia con Φₑ mejorado y umbral reducido para integración temprana."""
    def __init__(self, features: int, use_conscious=True):
        super().__init__()
        self.use_conscious = use_conscious
        self.integration_net = nn.Sequential(
            nn.Linear(features, features),
            nn.LayerNorm(features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features, features)
        )
        self.phi_effective = 0.0
        self.integration_threshold = 0.10  # FIX: reducido de 0.15 → 0.10
        self.running_phi = 0.0
        self.phi_history = []
    def compute_phi_effective(self, activity: torch.Tensor) -> float:
        if activity.numel() < 20 or activity.size(0) < 2:
            return 0.0
        with torch.no_grad():
            activity_mean = activity.mean(dim=0, keepdim=True)
            activity_std = activity.std(dim=0, keepdim=True) + 1e-6
            activity_norm = (activity - activity_mean) / activity_std
            cov = torch.mm(activity_norm.t(), activity_norm) / (activity.size(0) - 1)
            reg = 1e-4 * torch.eye(cov.size(0), device=cov.device)
            cov = cov + reg
            try:
                eigs = torch.linalg.eigvalsh(cov) if hasattr(torch.linalg, 'eigvalsh') else torch.symeig(cov, eigenvectors=False)[0]
                eigs = torch.sort(eigs, descending=True).values.clamp(min=0)
                eigs_normalized = eigs / (eigs.sum() + 1e-8)
                entropy = -(eigs_normalized * torch.log(eigs_normalized + 1e-8)).sum()
                max_entropy = np.log(len(eigs))
                phi = (entropy / max_entropy).item()
                self.running_phi = 0.7 * self.running_phi + 0.3 * phi
                self.phi_history.append(self.running_phi)
                return max(0.0, min(1.0, self.running_phi))
            except:
                return self.running_phi
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_conscious:
            return x
        self.phi_effective = self.compute_phi_effective(x)
        if self.phi_effective > self.integration_threshold:
            integration_strength = torch.sigmoid(
                torch.tensor((self.phi_effective - self.integration_threshold) * 5.0)
            ).item()
            integrated = self.integration_net(x)
            return x + integration_strength * integrated
        else:
            return x + 0.1 * self.integration_net(x)



# =============================================================================
# MEJORA 3: Arquitectura con mejor flujo de gradientes
# =============================================================================

class OmniBrainV8(nn.Module):
    """Arquitectura optimizada con conexiones residuales."""
    def __init__(self, use_fastslow=True, use_conscious=True):
        super().__init__()
        self.use_fastslow = use_fastslow
        self.use_conscious = use_conscious
        
        # Encoder CNN mejorado con más regularización
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),  # MEJORA
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),  # MEJORA
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # MEJORA: Conexión residual en core
        self.core = nn.Sequential(
            nn.Linear(256, 256), 
            nn.LayerNorm(256),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.25)  # Reducido de 0.3
        )
        
        self.dual = DualSystemModule(256, use_fastslow=use_fastslow)
        self.conscious = ConsciousnessModule(256, use_conscious=use_conscious)
        
        # MEJORA: Projection head antes del classifier
        self.projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        h = self.encoder(x)
        h_core = self.core(h)
        h_dual = self.dual(h_core)
        # MEJORA: Conexión residual
        h_dual = h_dual + h_core  
        h_conscious = self.conscious(h_dual)
        h_proj = self.projection(h_conscious)
        return self.classifier(h_proj)

    def reset_all_fast_weights(self):
        if not self.use_fastslow:
            return
        for module in self.modules():
            if hasattr(module, 'reset_fast_weights'):
                module.reset_fast_weights()

    def get_fast_norms(self):
        if not self.use_fastslow:
            return []
        return [m.get_fast_norm() for m in self.modules() if hasattr(m, 'get_fast_norm')]


# =============================================================================
# DualSystemModule (sin cambios mayores)
# =============================================================================

class DualSystemModule(nn.Module):
    """Sistema dual rápido/lento con memoria."""
    def __init__(self, dim, use_fastslow=True):
        super().__init__()
        self.use_fastslow = use_fastslow
        
        if use_fastslow:
            self.fast_path = FastSlowLinear(dim, dim, fast_lr=0.015, fast_decay=0.95)
            self.slow_path = FastSlowLinear(dim, dim, fast_lr=0.005, fast_decay=0.97)
        else:
            self.fast_path = nn.Linear(dim, dim)
            self.slow_path = nn.Linear(dim, dim)
            
        self.integrator = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)  # MEJORA
        )
        self.register_buffer('memory', torch.zeros(1, dim))
        self.tau = 0.9  # Reducido de 0.95

    def forward(self, x):
        with torch.no_grad():
            self.memory = self.tau * self.memory + (1 - self.tau) * x.mean(dim=0, keepdim=True)
        
        fast_out = self.fast_path(x)
        slow_in = x + 0.3 * self.memory  # MEJORA: Factor de mezcla
        slow_out = self.slow_path(slow_in)
        combined = torch.cat([fast_out, slow_out], dim=1)
        return self.integrator(combined)


# =============================================================================
# MEJORA 4: Training con Warm Restarts y mejor scheduler
# =============================================================================

def train_model(model, train_loader, test_loader, device, epochs=15, lr=1e-3, 
                config_name="model", verbose=True):
    """Entrenamiento optimizado."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # MEJORA: Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-5
    )
    
    # MEJORA: Focal Loss para clases difíciles
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
        
        def forward(self, inputs, targets):
            ce_loss = self.ce(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
    
    criterion = FocalLoss()
    
    history = {
        'train_loss': [],
        'test_acc': [],
        'phi_values': [],
        'fast_norms': [],
        'lr': []
    }
    
    best_acc = 0.0
    patience = 7  # Aumentado
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        model.reset_all_fast_weights() # Descomenta la siguiente línea si deseas evitar memoria inter-época
        total_loss = 0.0
        total_samples = 0
        phi_values = []
        fast_norms = []
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Más agresivo
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
            if batch_idx % 100 == 0 and verbose:
                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                fast_norms = model.get_fast_norms()
                avg_fast = np.mean(fast_norms) if fast_norms else 0.0
                logging.info(
                    f"  Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | Φₑ: {model.conscious.phi_effective:.4f} | "
                    f"FastNorm: {avg_fast:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

            if hasattr(model.conscious, 'phi_effective'):
                phi_values.append(model.conscious.phi_effective)
            fast_norms.extend(model.get_fast_norms())
        
        scheduler.step()
        
        # Evaluación
        test_acc = evaluate(model, test_loader, device)
        avg_loss = total_loss / total_samples
        
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        history['phi_values'].append(np.mean(phi_values) if phi_values else 0.0)
        history['fast_norms'].append(np.mean(fast_norms) if fast_norms else 0.0)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        if verbose:
            print(f"[{config_name}] Época {epoch+1}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2%} | "
                  f"Φₑ: {history['phi_values'][-1]:.4f} | "
                  f"FastNorm: {history['fast_norms'][-1]:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping en época {epoch+1}")
                break
    
    return history, best_acc


def evaluate(model, loader, device, return_per_class=False):
    """Evaluación estándar."""
    model.eval()
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    phi_per_class = defaultdict(list)
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            
            correct += pred.eq(y).sum().item()
            total += x.size(0)
            
            if return_per_class:
                for i in range(x.size(0)):
                    label = y[i].item()
                    per_class_total[label] += 1
                    if pred[i] == y[i]:
                        per_class_correct[label] += 1
                    if hasattr(model.conscious, 'phi_effective'):
                        phi_per_class[label].append(model.conscious.phi_effective)
    
    acc = correct / total if total > 0 else 0.0
    
    if return_per_class:
        class_accs = {cls: per_class_correct[cls] / per_class_total[cls] 
                      for cls in per_class_total if per_class_total[cls] > 0}
        avg_phi_per_class = {cls: np.mean(phi_per_class[cls]) if phi_per_class[cls] else 0.0
                            for cls in phi_per_class}
        return acc, class_accs, avg_phi_per_class
    
    return acc


# =============================================================================
# DATA LOADERS (sin cambios)
# =============================================================================

def get_cifar10_loaders(batch_size=64):
    """Loaders con data augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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


# =============================================================================
# MEJORA 5: Diagnosis detallado
# =============================================================================

def diagnose_model(model, loader, device):
    """Diagnóstico profundo del modelo."""
    model.eval()
    
    diagnostics = {
        'phi_distribution': [],
        'fast_weight_stats': [],
        'activation_stats': {},
        'gradient_flow': []
    }
    
    # Hook para capturar activaciones
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    model.conscious.register_forward_hook(get_activation('conscious'))
    model.dual.register_forward_hook(get_activation('dual'))
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _ = model(x)
            
            diagnostics['phi_distribution'].append(model.conscious.phi_effective)
            diagnostics['fast_weight_stats'].extend(model.get_fast_norms())
            
            for name, act in activations.items():
                if name not in diagnostics['activation_stats']:
                    diagnostics['activation_stats'][name] = []
                diagnostics['activation_stats'][name].append({
                    'mean': act.mean().item(),
                    'std': act.std().item(),
                    'min': act.min().item(),
                    'max': act.max().item()
                })
    
    print("\n" + "="*60)
    print("🔬 DIAGNÓSTICO DEL MODELO")
    print("="*60)
    print(f"Φₑ promedio: {np.mean(diagnostics['phi_distribution']):.4f}")
    print(f"Φₑ std: {np.std(diagnostics['phi_distribution']):.4f}")
    print(f"FastWeight promedio: {np.mean(diagnostics['fast_weight_stats']):.4f}")
    
    for name, stats in diagnostics['activation_stats'].items():
        avg_stats = {k: np.mean([s[k] for s in stats]) for k in stats[0].keys()}
        print(f"\n{name} activations:")
        print(f"  Mean: {avg_stats['mean']:.4f}, Std: {avg_stats['std']:.4f}")
    
    return diagnostics


# =============================================================================
# MAIN
# =============================================================================

def main():
    """POC mejorado."""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║              OMNI BRAIN V8 - VERSIÓN MEJORADA                  ║
    ║                                                                ║
    ║  ✅ Φₑ más sensible (condiciones relajadas)                    ║
    ║  ✅ Fast weights con mejor inicialización                      ║
    ║  ✅ Focal Loss para clases difíciles                           ║
    ║  ✅ Conexiones residuales                                      ║
    ║  ✅ Diagnosis profundo                                         ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)
    
    # Entrenar modelo completo
    print("\n🚀 ENTRENANDO MODELO COMPLETO...")
    model = OmniBrainV8(use_fastslow=True, use_conscious=True).to(device)
    history, best_acc = train_model(model, train_loader, test_loader, device, epochs=20)
    
    print(f"\n✅ Mejor precisión: {best_acc:.2%}")
    
    # Diagnóstico
    print("\n🔬 EJECUTANDO DIAGNÓSTICO...")
    diagnostics = diagnose_model(model, test_loader, device)
    
    # Guardar historia de Φₑ
    if hasattr(model.conscious, 'phi_history'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.conscious.phi_history, label='Φₑ durante entrenamiento')
        plt.xlabel('Iteración')
        plt.ylabel('Φₑ')
        plt.title('Evolución de Φₑ (Integrated Information)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('phi_evolution.png', dpi=300, bbox_inches='tight')
        print("📊 Gráfica guardada: phi_evolution.png")

if __name__ == "__main__":
    main()
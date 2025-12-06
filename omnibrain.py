#!/usr/bin/env python3
# =============================================================================
# OMNI BRAIN v8 - POC COMPLETO CON ABLATION STUDIES + VISUALIZACIONES
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
# COMPONENTES CORE
# =============================================================================

class FastSlowLinear(nn.Module):
    """Linear layer con pesos hebbianos estabilizados."""
    def __init__(self, in_features, out_features, fast_lr=0.005, fast_decay=0.95):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fast_lr = fast_lr
        self.fast_decay = fast_decay

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
            self.fast_weight *= self.fast_decay
            self.fast_bias *= self.fast_decay
            
            slow_out = F.linear(x, self.slow_weight, self.slow_bias)
            hebb_update = torch.mm(slow_out.t(), x) / x.size(0)
            self.fast_weight += self.fast_lr * hebb_update
            
            # Normalización L2 por fila
            fast_weight_norm = torch.norm(self.fast_weight, dim=1, keepdim=True)
            self.fast_weight = torch.where(
                fast_weight_norm > 0.1,
                self.fast_weight * (0.1 / fast_weight_norm),
                self.fast_weight
            )
            
            self.fast_bias += self.fast_lr * slow_out.mean(dim=0)
            self.fast_bias.clamp_(-0.1, 0.1)

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

    def get_fast_norm(self):
        return float(self.fast_weight.norm().item())


class DualSystemModule(nn.Module):
    """Sistema dual rápido/lento con memoria."""
    def __init__(self, dim, use_fastslow=True):
        super().__init__()
        self.use_fastslow = use_fastslow
        
        if use_fastslow:
            self.fast_path = FastSlowLinear(dim, dim, fast_lr=0.01, fast_decay=0.95)
            self.slow_path = FastSlowLinear(dim, dim, fast_lr=0.003, fast_decay=0.97)
        else:
            # Fallback a Linear estándar
            self.fast_path = nn.Linear(dim, dim)
            self.slow_path = nn.Linear(dim, dim)
            
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
    """Módulo de consciencia con Φₑ mejorado."""
    def __init__(self, features: int, use_conscious=True):
        super().__init__()
        self.use_conscious = use_conscious
        self.integration_net = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )
        self.phi_effective = 0.0
        self.integration_threshold = 0.2
        self.running_phi = 0.0
        
    def compute_phi_effective(self, activity: torch.Tensor) -> float:
        """Φₑ basado en eigenvalues de covarianza."""
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
            phi = (eigs[0] / total).item()
            # Suavizado exponencial
            self.running_phi = 0.9 * self.running_phi + 0.1 * phi
            return max(0.0, min(1.0, self.running_phi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_conscious:
            return x
            
        self.phi_effective = self.compute_phi_effective(x)
        
        # Integración gradual basada en Φₑ
        if self.phi_effective > self.integration_threshold:
            integration_strength = min(1.0, (self.phi_effective - self.integration_threshold) * 2.5)
            return x + integration_strength * self.integration_net(x)
        else:
            return x + 0.05 * self.integration_net(x)


# =============================================================================
# MODELO PRINCIPAL
# =============================================================================

class OmniBrainV8(nn.Module):
    """Arquitectura completa con switches para ablation."""
    def __init__(self, use_fastslow=True, use_conscious=True):
        super().__init__()
        self.use_fastslow = use_fastslow
        self.use_conscious = use_conscious
        
        # Encoder CNN mejorado
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.core = nn.Sequential(
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )
        
        self.dual = DualSystemModule(256, use_fastslow=use_fastslow)
        self.conscious = ConsciousnessModule(256, use_conscious=use_conscious)
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        h = self.encoder(x)
        h = self.core(h)
        h = self.dual(h)
        h = self.conscious(h)
        return self.classifier(h)

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
# DATOS
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


def get_few_shot_loaders(n_way=5, k_shot=5, batch_size=32):
    """Few-shot learning setup: entrenar en clases limitadas."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    full_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    full_test = datasets.CIFAR10('./data', train=False, transform=transform)
    
    # Seleccionar solo n_way clases con k_shot ejemplos por clase
    train_indices = []
    test_indices = []
    selected_classes = list(range(n_way))
    
    for cls in selected_classes:
        cls_train_idx = [i for i, (_, label) in enumerate(full_train) if label == cls][:k_shot]
        cls_test_idx = [i for i, (_, label) in enumerate(full_test) if label == cls][:100]
        train_indices.extend(cls_train_idx)
        test_indices.extend(cls_test_idx)
    
    train_subset = Subset(full_train, train_indices)
    test_subset = Subset(full_test, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# =============================================================================
# EVALUACIÓN
# =============================================================================

def evaluate(model, loader, device, return_per_class=False):
    """Evaluación con opción de métricas por clase."""
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
                    # Registrar Φₑ por clase
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
# ENTRENAMIENTO
# =============================================================================

def train_model(model, train_loader, test_loader, device, epochs=15, lr=1e-3, 
                config_name="model", verbose=True):
    """Entrenamiento con learning rate scheduler y early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    history = {
        'train_loss': [],
        'test_acc': [],
        'phi_values': [],
        'fast_norms': [],
        'lr': []
    }
    
    best_acc = 0.0
    patience = 5
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        phi_values = []
        fast_norms = []
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.reset_all_fast_weights()
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            
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
                    f"Loss: {avg_loss:.4f} | Φₑ: {model.conscious.phi_effective:.4f} | "
                    f"FastNorm: {avg_fast:.2f} | AvgGrad: {avg_grad:.6f}"
                )

            # Registrar métricas
            if hasattr(model.conscious, 'phi_effective'):
                phi_values.append(model.conscious.phi_effective)
            fast_norms.extend(model.get_fast_norms())
            
            # Limpiar fast weights
            for m in model.modules():
                if hasattr(m, 'end_of_batch'):
                    m.end_of_batch()
        
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
                  f"FastNorm: {history['fast_norms'][-1]:.2f}")
        
        # Early stopping
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


# =============================================================================
# ABLATION STUDIES
# =============================================================================

def run_ablation_study(epochs=15, batch_size=64):
    """Ejecuta 4 configuraciones y compara resultados."""
    print("=" * 80)
    print("🔬 ABLATION STUDY - OMNI BRAIN V8")
    print("=" * 80)
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    
    configs = {
        'baseline': {'use_fastslow': False, 'use_conscious': False},
        'fastslow_only': {'use_fastslow': True, 'use_conscious': False},
        'conscious_only': {'use_fastslow': False, 'use_conscious': True},
        'full_omni': {'use_fastslow': True, 'use_conscious': True}
    }
    
    results = {}
    
    for config_name, config_params in configs.items():
        print(f"\n{'='*80}")
        print(f"📊 Configuración: {config_name.upper()}")
        print(f"   Fast/Slow: {config_params['use_fastslow']}, Conscious: {config_params['use_conscious']}")
        print(f"{'='*80}\n")
        
        model = OmniBrainV8(**config_params).to(device)
        history, best_acc = train_model(
            model, train_loader, test_loader, device, 
            epochs=epochs, config_name=config_name
        )
        
        results[config_name] = {
            'history': history,
            'best_acc': best_acc,
            'final_acc': history['test_acc'][-1],
            'params': sum(p.numel() for p in model.parameters())
        }
        
        print(f"\n✅ {config_name}: Best Acc = {best_acc:.2%}, Final Acc = {history['test_acc'][-1]:.2%}\n")
    
    return results


# =============================================================================
# FEW-SHOT EXPERIMENT
# =============================================================================

def run_few_shot_experiment(n_way=5, k_shot=10, epochs=20):
    """Prueba capacidad de few-shot learning."""
    print("\n" + "=" * 80)
    print(f"🎯 FEW-SHOT LEARNING ({n_way}-way {k_shot}-shot)")
    print("=" * 80)
    
    train_loader, test_loader = get_few_shot_loaders(n_way=n_way, k_shot=k_shot)
    
    # Comparar modelo con y sin fast weights
    configs = {
        'without_fastslow': {'use_fastslow': False, 'use_conscious': False},
        'with_fastslow': {'use_fastslow': True, 'use_conscious': True}
    }
    
    results = {}
    
    for config_name, config_params in configs.items():
        print(f"\n📊 {config_name}")
        model = OmniBrainV8(**config_params).to(device)
        history, best_acc = train_model(
            model, train_loader, test_loader, device,
            epochs=epochs, lr=5e-4, config_name=config_name
        )
        results[config_name] = {'history': history, 'best_acc': best_acc}
        print(f"✅ Best Acc: {best_acc:.2%}")
    
    return results


# =============================================================================
# PHI POR CLASE
# =============================================================================

def analyze_phi_per_class():
    """Analiza correlación entre Φₑ y dificultad de clase."""
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS DE Φₑ POR CLASE")
    print("=" * 80)
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)
    model = OmniBrainV8(use_fastslow=True, use_conscious=True).to(device)
    
    # Entrenar brevemente
    print("\n🏋️ Entrenando modelo...")
    train_model(model, train_loader, test_loader, device, epochs=10, verbose=False)
    
    # Evaluar con métricas por clase
    print("\n📊 Evaluando por clase...")
    acc, class_accs, avg_phi_per_class = evaluate(model, test_loader, device, return_per_class=True)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"\nPrecisión global: {acc:.2%}\n")
    print(f"{'Clase':<12} {'Precisión':<12} {'Φₑ Promedio':<12}")
    print("-" * 40)
    
    for cls_idx in sorted(class_accs.keys()):
        cls_name = class_names[cls_idx]
        cls_acc = class_accs[cls_idx]
        cls_phi = avg_phi_per_class.get(cls_idx, 0.0)
        print(f"{cls_name:<12} {cls_acc:>10.2%}  {cls_phi:>10.4f}")
    
    return class_accs, avg_phi_per_class


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def plot_ablation_results(results):
    """Genera gráficas comparativas de ablation study."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OMNI BRAIN V8 - Ablation Study Results', fontsize=16, fontweight='bold')
    
    colors = {
        'baseline': '#e74c3c',
        'fastslow_only': '#3498db',
        'conscious_only': '#f39c12',
        'full_omni': '#2ecc71'
    }
    
    # 1. Test Accuracy
    ax = axes[0, 0]
    for config, data in results.items():
        epochs = range(1, len(data['history']['test_acc']) + 1)
        ax.plot(epochs, [acc * 100 for acc in data['history']['test_acc']], 
                label=config, color=colors[config], linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Época', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title('A) Evolución de Precisión')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Training Loss
    ax = axes[0, 1]
    for config, data in results.items():
        epochs = range(1, len(data['history']['train_loss']) + 1)
        ax.plot(epochs, data['history']['train_loss'], 
                label=config, color=colors[config], linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Época', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('B) Pérdida de Entrenamiento')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Φₑ Evolution
    ax = axes[1, 0]
    for config, data in results.items():
        if any(data['history']['phi_values']):
            epochs = range(1, len(data['history']['phi_values']) + 1)
            ax.plot(epochs, data['history']['phi_values'], 
                    label=config, color=colors[config], linewidth=2, marker='^', markersize=4)
    ax.set_xlabel('Época', fontweight='bold')
    ax.set_ylabel('Φₑ Promedio', fontweight='bold')
    ax.set_title('C) Evolución de Φₑ')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Best Accuracy Comparison
    ax = axes[1, 1]
    config_names = list(results.keys())
    best_accs = [results[cfg]['best_acc'] * 100 for cfg in config_names]
    bars = ax.bar(config_names, best_accs, color=[colors[cfg] for cfg in config_names], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Agregar valores encima de barras
    for bar, acc in zip(bars, best_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Best Test Accuracy (%)', fontweight='bold')
    ax.set_title('D) Comparación de Mejor Precisión')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Gráfica guardada: ablation_results.png")


def plot_phi_analysis(class_accs, avg_phi_per_class):
    """Gráfica correlación Φₑ vs dificultad de clase."""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Análisis de Φₑ por Clase CIFAR-10', fontsize=14, fontweight='bold')
    
    # Ordenar por precisión
    sorted_classes = sorted(class_accs.keys(), key=lambda x: class_accs[x])
    
    # 1. Φₑ vs Accuracy
    accs = [class_accs[cls] * 100 for cls in sorted_classes]
    phis = [avg_phi_per_class.get(cls, 0) for cls in sorted_classes]
    labels = [class_names[cls] for cls in sorted_classes]
    
    ax1.scatter(phis, accs, s=150, alpha=0.7, c=range(len(sorted_classes)), 
                cmap='viridis', edgecolors='black', linewidth=1.5)
    
    for i, label in enumerate(labels):
        ax1.annotate(label, (phis[i], accs[i]), fontsize=8, 
                    xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Φₑ Promedio', fontweight='bold')
    ax1.set_ylabel('Precisión (%)', fontweight='bold')
    ax1.set_title('A) Correlación Φₑ - Precisión')
    ax1.grid(alpha=0.3)
    
    # 2. Barras comparativas
    x = np.arange(len(sorted_classes))
    width = 0.35
    
    ax2.bar(x - width/2, accs, width, label='Precisión (%)', alpha=0.8, color='#3498db')
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width/2, [p*100 for p in phis], width, 
                 label='Φₑ (x100)', alpha=0.8, color='#e74c3c')
    
    ax2.set_xlabel('Clase', fontweight='bold')
    ax2.set_ylabel('Precisión (%)', fontweight='bold', color='#3498db')
    ax2_twin.set_ylabel('Φₑ (x100)', fontweight='bold', color='#e74c3c')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_title('B) Métricas por Clase')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('phi_per_class_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Gráfica guardada: phi_per_class_analysis.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Ejecuta el POC completo."""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                   OMNI BRAIN V8 - POC COMPLETO                 ║
    ║                                                                ║
    ║  1. Ablation Study (4 configuraciones)                        ║
    ║  2. Few-Shot Learning Test (5-way 10-shot)                   ║
    ║  3. Análisis de Φₑ por Clase                                  ║
    ║  4. Visualizaciones Automáticas                               ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # 1. Ablation Study
    print("\n🚀 INICIANDO ABLATION STUDY...")
    results = run_ablation_study(epochs=15, batch_size=64)
    plot_ablation_results(results)
    
    # 2. Few-Shot Learning
    print("\n🚀 INICIANDO FEW-SHOT EXPERIMENT...")
    few_shot_results = run_few_shot_experiment(n_way=5, k_shot=10, epochs=20)
    
    # 3. Análisis de Φₑ por clase
    print("\n🚀 INICIANDO ANÁLISIS DE Φₑ POR CLASE...")
    class_accs, avg_phi_per_class = analyze_phi_per_class()
    plot_phi_analysis(class_accs, avg_phi_per_class)
    
    print("\n" + "="*80)
    print("✅ POC COMPLETO FINALIZADO - Gráficas guardadas en:")
    print("   - ablation_results.png")
    print("   - phi_per_class_analysis.png")
    print("="*80)

if __name__ == "__main__":
    main()
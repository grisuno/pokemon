"""
SimplePGD - Baseline Robusto para 75%+ PGD Accuracy
====================================================
FILOSOFÍA: Simplicidad + Componentes Validados
- Adversarial Training correcto (Madry et al.)
- Arquitectura conservadora con regularización
- Curriculum suave
- Sin complejidad innecesaria

Objetivo: Establecer baseline sólido antes de agregar complejidad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, Subset
import time
from pathlib import Path
import json

# =============================================================================
# CONFIGURACIÓN SIMPLE Y EFECTIVA
# =============================================================================
class SimpleConfig:
    # Dataset
    n_samples = 2000
    n_features = 20
    n_classes = 3
    n_informative = 16
    
    # Arquitectura conservadora
    hidden_dims = [64, 32]  # Más pequeña = más fácil de entrenar
    dropout = 0.3
    
    # Training
    batch_size = 32
    epochs = 30
    lr = 0.001  # LR más bajo
    weight_decay = 5e-4
    
    # Adversarial (progresivo)
    eps_schedule = [0.0, 0.05, 0.1, 0.15, 0.2]  # Mucho más suave
    pgd_steps = 10
    pgd_step_size = 0.01  # Paso pequeño y fijo
    test_eps = 0.2  # Epsilon de test menor
    
    # Control
    device = "cpu"
    seed = 42

def seed_everything(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_dataset(config):
    """Dataset balanceado con más separabilidad"""
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        n_informative=config.n_informative,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.01,
        class_sep=1.5,  # MÁS separabilidad
        random_state=config.seed
    )
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )

# =============================================================================
# ARQUITECTURA SIMPLE CON REGULARIZACIÓN
# =============================================================================
class SimpleRobustNet(nn.Module):
    """Red simple pero bien regularizada"""
    def __init__(self, config):
        super().__init__()
        
        dims = [config.n_features] + config.hidden_dims + [config.n_classes]
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No BN/Dropout en última capa
                self.bns.append(nn.BatchNorm1d(dims[i+1]))
                self.dropouts.append(nn.Dropout(config.dropout))
        
        # Inicialización conservadora
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
        
        return self.layers[-1](x)

# =============================================================================
# PGD ATTACK CORRECTO (Madry et al.)
# =============================================================================
def pgd_attack(model, x, y, eps, steps, step_size):
    """
    PGD estándar bien implementado
    - Random start
    - Step size controlado
    - Projection al epsilon-ball
    """
    model.eval()
    
    # Random initialization
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta.requires_grad = True
    
    for _ in range(steps):
        # Forward
        logits = model(x + delta)
        loss = F.cross_entropy(logits, y)
        
        # Backward
        loss.backward()
        
        # Gradient ascent con step size fijo
        with torch.no_grad():
            delta.data = delta + step_size * delta.grad.sign()
            # Project a epsilon-ball
            delta.data = torch.clamp(delta.data, -eps, eps)
            # Project a valid input range [0, 1] (asumiendo inputs normalizados)
            delta.data = torch.clamp(x + delta.data, -3, 3) - x
        
        delta.grad.zero_()
    
    model.train()
    return (x + delta).detach()

# =============================================================================
# ENTRENAMIENTO CON CURRICULUM SUAVE
# =============================================================================
def train_simple_robust(config, dataset, verbose=True):
    """Entrenamiento con adversarial training progresivo"""
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.seed)
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    
    fold_results = {'pgd_acc': [], 'clean_acc': [], 'train_time': []}
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if verbose:
            print(f"\n📂 Fold {fold_idx+1}/3")
        
        # Data loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
        
        # Model
        model = SimpleRobustNet(config).to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        
        # Scheduler con warm-up
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr * 2,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader)
        )
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0.0
            
            # Curriculum: epsilon crece por fases
            if epoch < 6:
                current_eps = config.eps_schedule[0]
            elif epoch < 12:
                current_eps = config.eps_schedule[1]
            elif epoch < 18:
                current_eps = config.eps_schedule[2]
            elif epoch < 24:
                current_eps = config.eps_schedule[3]
            else:
                current_eps = config.eps_schedule[4]
            
            for x, y in train_loader:
                x, y = x.to(config.device), y.to(config.device)
                
                # Generar ejemplos adversariales
                if current_eps > 0:
                    x_adv = pgd_attack(model, x, y, current_eps, config.pgd_steps, config.pgd_step_size)
                else:
                    x_adv = x
                
                # Forward
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
            
            if verbose and epoch % 5 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f"  Epoch {epoch:2d}/{config.epochs} | Loss: {avg_loss:.4f} | ε: {current_eps:.3f}")
        
        train_time = time.time() - start_time
        
        # Evaluación
        model.eval()
        clean_correct = pgd_correct = total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                
                # Clean accuracy
                logits_clean = model(x)
                pred_clean = logits_clean.argmax(dim=1)
                clean_correct += pred_clean.eq(y).sum().item()
                total += y.size(0)
        
        # PGD accuracy
        for x, y in val_loader:
            x, y = x.to(config.device), y.to(config.device)
            x_adv = pgd_attack(model, x, y, config.test_eps, config.pgd_steps, config.pgd_step_size)
            
            with torch.no_grad():
                logits_adv = model(x_adv)
                pred_adv = logits_adv.argmax(dim=1)
                pgd_correct += pred_adv.eq(y).sum().item()
        
        pgd_acc = 100.0 * pgd_correct / total
        clean_acc = 100.0 * clean_correct / total
        
        fold_results['pgd_acc'].append(pgd_acc)
        fold_results['clean_acc'].append(clean_acc)
        fold_results['train_time'].append(train_time)
        
        if verbose:
            print(f"  ✅ PGD: {pgd_acc:.2f}% | Clean: {clean_acc:.2f}% | Time: {train_time:.1f}s")
    
    return {
        'pgd_mean': np.mean(fold_results['pgd_acc']),
        'pgd_std': np.std(fold_results['pgd_acc']),
        'clean_mean': np.mean(fold_results['clean_acc']),
        'clean_std': np.std(fold_results['clean_acc']),
        'train_time': np.mean(fold_results['train_time']),
        'n_params': sum(p.numel() for p in model.parameters())
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    seed_everything(42)
    
    print("="*80)
    print("🎯 SimplePGD - Baseline Robusto")
    print("="*80)
    print("Estrategia:")
    print("  • Arquitectura simple (64→32→3)")
    print("  • Adversarial training con curriculum suave")
    print("  • Regularización estándar (BN + Dropout + Weight Decay)")
    print("  • PGD correcto (step size fijo)")
    print("="*80 + "\n")
    
    config = SimpleConfig()
    dataset = get_dataset(config)
    
    print(f"📊 Dataset: {len(dataset)} samples")
    print(f"🧠 Arquitectura: {config.n_features}→{config.hidden_dims}→{config.n_classes}")
    print(f"⚙️  Curriculum: ε = {config.eps_schedule}\n")
    
    results = train_simple_robust(config, dataset, verbose=True)
    
    print("\n" + "="*80)
    print("🏆 RESULTADOS")
    print("="*80)
    print(f"PGD Accuracy:   {results['pgd_mean']:.2f}% ± {results['pgd_std']:.2f}%")
    print(f"Clean Accuracy: {results['clean_mean']:.2f}% ± {results['clean_std']:.2f}%")
    print(f"Parámetros:     {results['n_params']:,}")
    print(f"Tiempo/Fold:    {results['train_time']:.1f}s")
    print("="*80)
    
    # Guardar
    results_dir = Path("simple_pgd_baseline")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = main()
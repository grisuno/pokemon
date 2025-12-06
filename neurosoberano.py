"""
NeuroSovereign POC: Minimal Viable Experiment
==============================================
Compara tu modelo contra baseline honesto en CIFAR-10
Optimizado para Colab Free Tier (T4, 12hrs)

Ejecutar:
    python poc_comparison.py --mode both --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# =============================================================================
# CONFIGURACIÓN EXPERIMENTAL
# =============================================================================
@dataclass
class ExperimentConfig:
    epochs: int = 50  # Reducido de 200 para factibilidad
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# BASELINE: Wide-ResNet Simplificado
# =============================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class WideResNetBaseline(nn.Module):
    """
    Wide-ResNet simplificado (depth=16, width=2)
    Parámetros similares a tu modelo (~1-2M)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        k = 2  # Width factor
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 16*k, 2, stride=1)
        self.layer2 = self._make_layer(16*k, 32*k, 2, stride=2)
        self.layer3 = self._make_layer(32*k, 64*k, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64*k, num_classes)
    
    def _make_layer(self, in_c, out_c, num_blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

# =============================================================================
# TU MODELO (Versión Mínima - Sin SVD para velocidad)
# =============================================================================
class FastLiquidNeuron(nn.Module):
    """Neurona líquida simplificada (sin SVD consolidation para POC)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.ln = nn.LayerNorm(out_dim)
        self.base_lr = 0.005
        
    def forward(self, x, plasticity=0.0):
        slow_out = self.W_slow(x)
        fast_out = F.linear(x, self.W_fast)
        pre_act = slow_out + fast_out
        out = 5.0 * torch.tanh(self.ln(pre_act) / 5.0)
        
        # Hebbian update (solo si plasticity > 0)
        if self.training and plasticity > 0.01:
            with torch.no_grad():
                x_norm = (x ** 2).sum(1).mean() + 1e-6
                correlation = torch.mm(out.T, x) / x.size(0)
                forgetting = 0.2 * self.W_fast
                delta = torch.clamp((correlation / x_norm) - forgetting, -0.05, 0.05)
                self.W_fast.data += delta * self.base_lr * plasticity
                self.W_fast.data.mul_(0.999)
        
        return out

class MinimalNeuroSovereign(nn.Module):
    """Versión mínima de tu arquitectura para POC"""
    def __init__(self, num_classes=10):
        super().__init__()
        k = 2
        
        # Backbone similar al baseline
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16*k, 2, stride=1)
        self.layer2 = self._make_layer(16*k, 32*k, 2, stride=2)
        self.layer3 = self._make_layer(32*k, 64*k, 2, stride=2)
        
        # Liquid head (diferencia clave)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.liquid = FastLiquidNeuron(64*k, 128)
        self.fc = nn.Linear(128, num_classes)
        
        # Plasticity schedule
        self.register_buffer('plasticity', torch.tensor(0.0))
    
    def _make_layer(self, in_c, out_c, num_blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        
        # Liquid layer con plasticidad
        h = self.liquid(x, self.plasticity.item())
        return self.fc(F.gelu(h))
    
    def update_plasticity(self, epoch, total_epochs):
        """Plasticity schedule simplificado"""
        progress = epoch / total_epochs
        if progress < 0.4:
            self.plasticity.fill_(0.8)  # Alta plasticidad inicial
        elif progress < 0.8:
            # Decaimiento lineal
            self.plasticity.fill_(0.8 * (1.0 - (progress - 0.4) / 0.4))
        else:
            self.plasticity.fill_(0.0)  # Cristalización

# =============================================================================
# TRAINING LOOP UNIFICADO
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, device, use_mixup=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        if use_mixup and np.random.random() > 0.5:
            lam = np.random.beta(0.4, 0.4)
            index = torch.randperm(x.size(0)).to(device)
            mixed_x = lam * x + (1 - lam) * x[index]
            
            optimizer.zero_grad()
            out = model(mixed_x)
            loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y[index])
            loss.backward()
            optimizer.step()
            
            pred = out.argmax(1)
            y_mixed = y if lam > 0.5 else y[index]
            correct += (pred == y_mixed).sum().item()
        else:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
        
        total_loss += loss.item()
        total += y.size(0)
    
    return total_loss / len(loader), 100.0 * correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return 100.0 * correct / total

# =============================================================================
# EXPERIMENTO COMPLETO
# =============================================================================
class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {'baseline': {}, 'neurosovereign': {}}
        
        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Data
        self.train_loader, self.test_loader = self._get_data()
    
    def _get_data(self):
        # Augmentation estándar (sin AutoAugment para velocidad)
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])
        
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])
        
        train_ds = datasets.CIFAR10('./data', True, download=True, transform=tf_train)
        test_ds = datasets.CIFAR10('./data', False, transform=tf_test)
        
        train_dl = DataLoader(train_ds, self.config.batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
        test_dl = DataLoader(test_ds, self.config.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
        
        return train_dl, test_dl
    
    def run_baseline(self):
        print("\n" + "="*80)
        print("🔵 BASELINE: Wide-ResNet (Standard Training)")
        print("="*80 + "\n")
        
        model = WideResNetBaseline().to(self.config.device)
        optimizer = optim.SGD(model.parameters(), lr=self.config.lr, 
                             momentum=0.9, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)
        criterion = nn.CrossEntropyLoss()
        
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}\n")
        
        history = {'train_acc': [], 'test_acc': [], 'time': []}
        
        for epoch in range(self.config.epochs):
            start = time.time()
            
            train_loss, train_acc = train_epoch(
                model, self.train_loader, optimizer, criterion, 
                self.config.device, use_mixup=(epoch >= 5)
            )
            test_acc = evaluate(model, self.test_loader, self.config.device)
            
            scheduler.step()
            epoch_time = time.time() - start
            
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['time'].append(epoch_time)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:03d} | Train: {train_acc:.2f}% | "
                      f"Test: {test_acc:.2f}% | Time: {epoch_time:.1f}s")
        
        self.results['baseline'] = {
            'final_test_acc': test_acc,
            'best_test_acc': max(history['test_acc']),
            'total_time': sum(history['time']),
            'params': params,
            'history': history
        }
        
        print(f"\n✅ Baseline Complete: {test_acc:.2f}%")
        return model, history
    
    def run_neurosovereign(self):
        print("\n" + "="*80)
        print("🧠 NEUROSOVEREIGN: Liquid Plasticity")
        print("="*80 + "\n")
        
        model = MinimalNeuroSovereign().to(self.config.device)
        optimizer = optim.SGD(model.parameters(), lr=self.config.lr,
                             momentum=0.9, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)
        criterion = nn.CrossEntropyLoss()
        
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}\n")
        
        history = {'train_acc': [], 'test_acc': [], 'time': [], 'plasticity': []}
        
        for epoch in range(self.config.epochs):
            start = time.time()
            
            # Update plasticity schedule
            model.update_plasticity(epoch, self.config.epochs)
            
            train_loss, train_acc = train_epoch(
                model, self.train_loader, optimizer, criterion,
                self.config.device, use_mixup=(epoch >= 5)
            )
            test_acc = evaluate(model, self.test_loader, self.config.device)
            
            scheduler.step()
            epoch_time = time.time() - start
            
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['time'].append(epoch_time)
            history['plasticity'].append(model.plasticity.item())
            
            if (epoch + 1) % 10 == 0:
                fast_norm = model.liquid.W_fast.norm().item()
                print(f"Epoch {epoch+1:03d} | Train: {train_acc:.2f}% | "
                      f"Test: {test_acc:.2f}% | Plasticity: {model.plasticity.item():.3f} | "
                      f"FastNorm: {fast_norm:.2f} | Time: {epoch_time:.1f}s")
        
        self.results['neurosovereign'] = {
            'final_test_acc': test_acc,
            'best_test_acc': max(history['test_acc']),
            'total_time': sum(history['time']),
            'params': params,
            'history': history
        }
        
        print(f"\n✅ NeuroSovereign Complete: {test_acc:.2f}%")
        return model, history
    
    def compare(self):
        print("\n" + "="*80)
        print("📊 COMPARISON RESULTS")
        print("="*80 + "\n")
        
        b = self.results['baseline']
        n = self.results['neurosovereign']
        
        print(f"{'Metric':<25} {'Baseline':>15} {'NeuroSovereign':>15} {'Δ':>10}")
        print("-" * 70)
        print(f"{'Final Test Accuracy':<25} {b['final_test_acc']:>14.2f}% {n['final_test_acc']:>14.2f}% {n['final_test_acc']-b['final_test_acc']:>9.2f}%")
        print(f"{'Best Test Accuracy':<25} {b['best_test_acc']:>14.2f}% {n['best_test_acc']:>14.2f}% {n['best_test_acc']-b['best_test_acc']:>9.2f}%")
        print(f"{'Parameters':<25} {b['params']:>15,} {n['params']:>15,} {n['params']-b['params']:>10,}")
        print(f"{'Total Time (s)':<25} {b['total_time']:>15.1f} {n['total_time']:>15.1f} {n['total_time']-b['total_time']:>10.1f}")
        
        print("\n" + "="*80)
        
        # Veredicto
        diff = n['best_test_acc'] - b['best_test_acc']
        if diff > 1.0:
            print("🏆 RESULT: NeuroSovereign WINS (>1% improvement)")
        elif diff > 0.0:
            print("⚖️  RESULT: Marginal improvement (plasticidad ayuda levemente)")
        else:
            print("📉 RESULT: Baseline WINS (plasticidad no ayuda)")
        
        print("="*80 + "\n")
        
        # Save results
        with open('poc_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Plot
        self.plot_comparison()
    
    def plot_comparison(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training curves
        ax = axes[0]
        b_hist = self.results['baseline']['history']
        n_hist = self.results['neurosovereign']['history']
        
        ax.plot(b_hist['test_acc'], label='Baseline', linewidth=2)
        ax.plot(n_hist['test_acc'], label='NeuroSovereign', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plasticity schedule
        ax = axes[1]
        ax.plot(n_hist['plasticity'], linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Plasticity Gate')
        ax.set_title('Plasticity Schedule')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('poc_comparison.png', dpi=150)
        print("📈 Plot saved: poc_comparison.png\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'neurosovereign', 'both'], 
                       default='both')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    config = ExperimentConfig(epochs=args.epochs)
    exp = Experiment(config)
    
    print(f"\n🔬 NeuroSovereign POC")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Mode: {args.mode}\n")
    
    if args.mode in ['baseline', 'both']:
        exp.run_baseline()
    
    if args.mode in ['neurosovereign', 'both']:
        exp.run_neurosovereign()
    
    if args.mode == 'both':
        exp.compare()

if __name__ == "__main__":
    main()

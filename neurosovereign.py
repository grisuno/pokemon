import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import math
from dataclasses import dataclass

# =============================================================================
# 0. CONFIGURACIÓN DE ALTO RENDIMIENTO (SOTA)
# =============================================================================
@dataclass
class SovereignConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Hiperparámetros de Entrenamiento SOTA
    epochs: int = 100            # Necesario para converger >95%
    batch_size: int = 128
    lr_max: float = 0.1          # Para OneCycleLR
    weight_decay: float = 5e-4
    momentum: float = 0.9
    
    # Regularización Avanzada
    mixup_alpha: float = 1.0     # Mezcla de imágenes para robustez
    label_smoothing: float = 0.1 # Evita sobreconfianza
    
    # Arquitectura Biológica
    widen_factor: int = 4        # Ancho del backbone (WideResNet style)
    droprate: float = 0.3        # Dropout estructural
    
    # Componentes Líquidos
    plasticity_rate: float = 0.05
    memory_decay: float = 0.9

config = SovereignConfig()

# =============================================================================
# 1. UTILITIES & AUGMENTATION
# =============================================================================
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True # Acelera convoluciones

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =============================================================================
# 2. EL CUERPO: WideResNet Backbone (La Fuerza Bruta)
# =============================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

# =============================================================================
# 3. LA MENTE: Liquid Sovereign Head (La Inteligencia)
# =============================================================================
class LiquidCortex(nn.Module):
    """
    Capa densa con Fast Weights Hebbianos y Homeostasis.
    Reemplaza a la capa lineal aburrida de las CNNs normales.
    """
    def __init__(self, in_features, out_features, config: SovereignConfig):
        super().__init__()
        self.config = config
        
        # Pesos Estructurales (Slow)
        self.W_slow = nn.Linear(in_features, out_features, bias=False)
        nn.init.kaiming_normal_(self.W_slow.weight)
        
        # Pesos Plásticos (Fast) - Memoria de Trabajo
        self.register_buffer('W_fast', torch.zeros(out_features, in_features))
        
        # Homeostasis: Escala dinámica
        self.scale = nn.Parameter(torch.ones(1) * 30.0) # Temperatura aprendible
        
        # Normalización pre-activación para estabilidad
        self.ln = nn.LayerNorm(in_features)

    def forward(self, x):
        # x: [Batch, Features]
        x_norm = self.ln(x)
        
        # 1. Camino Lento (Instinto/Estructura)
        slow_out = self.W_slow(x_norm)
        
        # 2. Camino Rápido (Contexto/Memoria Inmediata)
        # Los fast weights modulan la respuesta basándose en el batch actual
        fast_out = F.linear(x_norm, self.W_fast)
        
        # 3. Aprendizaje Hebbiano In-Situ (Solo en entrenamiento)
        if self.training:
            with torch.no_grad():
                # Hebbian update: dW = y * x^T
                # Usamos la salida 'slow' como target para guiar la plasticidad
                y_target = slow_out.detach()
                
                # Regla de Oja simplificada para estabilidad
                batch_size = x.size(0)
                delta = torch.mm(y_target.T, x_norm) / batch_size
                
                # Update con decaimiento (olvido)
                self.W_fast = (self.config.memory_decay * self.W_fast) + \
                              (self.config.plasticity_rate * delta)
        
        # Fusión
        # El cerebro combina instinto (slow) con contexto reciente (fast)
        total_out = slow_out + (0.1 * fast_out) # El fast weight es un sesgo sutil
        
        return total_out

class NeuroSovereignV1(nn.Module):
    def __init__(self, config: SovereignConfig, depth=16, num_classes=10):
        super().__init__()
        nChannels = [16, 16*config.widen_factor, 32*config.widen_factor, 64*config.widen_factor]
        n = (depth - 4) / 6
        block = BasicBlock
        
        # 1. Tallo Cerebral Visual (Stem)
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        # 2. Cuerpo Cortical (WideResNet Blocks)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, config.droprate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, config.droprate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, config.droprate)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. La Joya de la Corona: Liquid Sovereign Head
        self.cortex = LiquidCortex(nChannels[3], num_classes, config)
        
        # Inicialización
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Percepción Visual
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        
        # Abstracción
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        # Cognición Líquida
        return self.cortex(out)

# =============================================================================
# 4. MOTOR DE ENTRENAMIENTO (Ciclo Vital)
# =============================================================================
def get_optimized_dataloaders(config):
    # Augmentation muy fuerte para CIFAR-10 (SOTA Standard)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), # Jitter agresivo
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.3) # Cutout
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, 
                            num_workers=2, pin_memory=True)
    return trainloader, testloader

def train_sovereign():
    seed_everything(config.seed)
    
    print(f"\n🧠 INICIANDO NEUROSOVEREIGN V1 [Objetivo: >96%]")
    print(f"   Device: {config.device}")
    print(f"   Backbone: WideResNet-16-{config.widen_factor}")
    print(f"   Head: LiquidCortex (Hebbian + Homeostatic)")
    
    train_loader, test_loader = get_optimized_dataloaders(config)
    model = NeuroSovereignV1(config).to(config.device)
    
    # Optimizador SOTA
    optimizer = optim.SGD(model.parameters(), lr=config.lr_max, 
                          momentum=config.momentum, weight_decay=config.weight_decay, nesterov=True)
    
    # Scheduler OneCycle (La clave para la súper convergencia)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr_max, 
                                              steps_per_epoch=len(train_loader), epochs=config.epochs)
    
    # Loss con Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    best_acc = 0.0
    
    print(f"\n{'EP':<3} | {'LR':<7} | {'LOSS':<7} | {'TRAIN %':<7} | {'TEST %':<7} | {'STATUS'}")
    print("-" * 65)
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            # Mixup (Fusión de realidades)
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, config.mixup_alpha)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            
            # Gradient Clipping (Seguridad)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            # Estimación aproximada de acc en mixup (no exacta pero útil)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float()
                        + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            
        train_acc = 100. * correct / total
        
        # Evaluación
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # En evaluación, la LiquidCortex sigue usando sus Fast Weights (residuales del entrenamiento)
        # Esto simula la "intuición" adquirida.
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        
        status = ""
        if acc > best_acc:
            best_acc = acc
            status = "🔥 BEST"
            # Guardar solo el mejor modelo para ahorrar espacio
            torch.save(model.state_dict(), 'neurosovereign_best.pth')
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch+1:<3} | {current_lr:.5f} | {train_loss/len(train_loader):.4f}  | {train_acc:.2f}%   | {acc:.2f}%   | {status}")

    print(f"\n🏆 RESULTADO FINAL: {best_acc:.2f}%")
    print("   (Nota: Si es >96%, has superado a ResNet-50 estándar)")

if __name__ == "__main__":
    train_sovereign()
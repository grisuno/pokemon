import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import time
import os

# Fixed seed
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# TOPOBRAIN CORE - CPU Friendly
# ============================================================================

class TopoBrainCore(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=32, output_dim=256, 
                 grid_size=3, use_grid=True, use_symbiotic=True):
        super().__init__()
        self.use_grid = use_grid
        self.use_symbiotic = use_symbiotic
        self.n_nodes = grid_size * grid_size
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.n_nodes)
        
        if self.use_grid:
            self.register_buffer('grid_coords', self._init_grid(grid_size))
            self.plasticity = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes))
        
        if self.use_symbiotic:
            self.symbiotic = nn.Linear(self.n_nodes, self.n_nodes, bias=False)
            nn.init.orthogonal_(self.symbiotic.weight)
        
        self.fc3 = nn.Linear(self.n_nodes, output_dim)
        self.last_density = 1.0

    def _init_grid(self, size):
        coords = []
        for i in range(size):
            for j in range(size):
                coords.append([i / size, j / size])
        return torch.tensor(coords, dtype=torch.float32)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        if self.use_grid:
            d = torch.cdist(self.grid_coords, self.grid_coords)
            topo = torch.exp(-d * 2.0)
            w = topo * torch.sigmoid(self.plasticity)
            h = h @ w
            self.last_density = (h.abs() > 0.1).float().mean().item()
        else:
            self.last_density = 1.0

        if self.use_symbiotic:
            h = self.symbiotic(h)

        return self.fc3(F.relu(h))

    def get_metrics(self):
        return {'density': self.last_density}


# ============================================================================
# ENCODERS
# ============================================================================

class MiniUnconscious(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        self.proj = nn.Linear(64*4, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.proj(self.stem(x)))


class TopoUnconscious(nn.Module):
    def __init__(self, out_dim=256, use_grid=True, use_symbiotic=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        self.topobrain = TopoBrainCore(
            input_dim=64*4,
            hidden_dim=32,
            output_dim=out_dim,
            grid_size=3,
            use_grid=use_grid,
            use_symbiotic=use_symbiotic
        )

    def forward(self, x):
        feats = self.stem(x)
        return self.topobrain(feats)

    def get_metrics(self):
        return self.topobrain.get_metrics()


# ============================================================================
# CLASSIFIER (replaces language decoder for CPU simplicity)
# ============================================================================

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim=256, num_classes=10):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.head(x)


# ============================================================================
# FULL MODEL WITH 4 ABLATION LEVELS
# ============================================================================

class NeuroLogosCPU(nn.Module):
    """
    Ablation levels:
      0: BASELINE     -> MiniUnconscious
      1: +GRID        -> TopoBrain (grid only)
      2: +SYMBIOTIC   -> TopoBrain (grid + symbiotic)
      3: +ADVERSARIAL -> TopoBrain + FGSM (lightweight)
    """
    def __init__(self, num_classes=10, ablation_level=0):
        super().__init__()
        self.level = ablation_level
        self.use_adversarial = (ablation_level == 3)

        if ablation_level == 0:
            self.encoder = MiniUnconscious(out_dim=256)
        elif ablation_level == 1:
            self.encoder = TopoUnconscious(out_dim=256, use_grid=True, use_symbiotic=False)
        elif ablation_level in [2, 3]:
            self.encoder = TopoUnconscious(out_dim=256, use_grid=True, use_symbiotic=True)
        else:
            raise ValueError("ablation_level must be 0, 1, 2, or 3")

        self.classifier = SimpleClassifier(256, num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        return self.classifier(feats)

    def get_metrics(self):
        if hasattr(self.encoder, 'get_metrics'):
            return self.encoder.get_metrics()
        return {'density': 1.0}


# ============================================================================
# LIGHT FGSM (CPU-friendly adversarial for level 3)
# ============================================================================

def fgsm_attack(model, x, y, epsilon=0.15):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    x_adv = x + epsilon * x_adv.grad.sign()
    return x_adv.detach()


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, loader, optimizer, device, use_adv=False):
    model.train()
    total_loss, correct, total = 0, 0, 0
    total_density = 0
    start_time = time.time()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Clean forward
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        # Adversarial (only level 3, after epoch 2)
        if use_adv:
            x_adv = fgsm_attack(model, x, y, epsilon=0.15)
            logits_adv = model(x_adv)
            loss = 0.7 * loss + 0.3 * F.cross_entropy(logits_adv, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
        
        metrics = model.get_metrics()
        total_density += metrics['density']
    
    acc = 100. * correct / total
    density = total_density / len(loader)
    time_elapsed = time.time() - start_time
    avg_loss = total_loss / len(loader)
    
    return avg_loss, acc, density, time_elapsed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return 100. * correct / total


# ============================================================================
# MAIN ABLATION RUN
# ============================================================================

def run_ablation_cpu():
    device = torch.device('cpu')
    print(f"Running on: {device}")
    
    # Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    
    ablation_names = ["BASELINE", "+GRID", "+SYMBIOTIC", "+ADVERSARIAL"]
    results = {}
    
    for level in range(4):
        print(f"\n{'='*60}")
        print(f"ABLATION LEVEL {level}: {ablation_names[level]}")
        print(f"{'='*60}")
        
        model = NeuroLogosCPU(num_classes=10, ablation_level=level).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        use_adv = (level == 3)
        
        for epoch in range(10):  # short training
            loss, acc, density, t = train_epoch(model, train_loader, optimizer, device, use_adv and epoch > 2)
            if epoch % 5 == 0 or epoch == 9:
                test_acc = evaluate(model, test_loader, device)
                print(f"Epoch {epoch:02d} | Train Acc: {acc:5.2f}% | Test Acc: {test_acc:5.2f}% | "
                      f"Density: {density:.3f} | Time: {t:4.1f}s")
        
        final_test = evaluate(model, test_loader, device)
        avg_time = sum(t for _, _, _, t in [train_epoch(model, train_loader, optimizer, device, False) for _ in range(1)]) / 1  # dummy
        
        results[level] = {
            'test_acc': final_test,
            'density': density,
            'name': ablation_names[level]
        }
        
        # Save
        torch.save(model.state_dict(), f'neurologos_cpu_level{level}.pth')
        print(f"✅ Saved: neurologos_cpu_level{level}.pth")
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS (CPU-FRIENDLY ABLATION - 4 LEVELS)")
    print(f"{'='*60}")
    for level in range(4):
        r = results[level]
        print(f"{r['name']:<15} | Test Acc: {r['test_acc']:5.2f}% | Density: {r['density']:6.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_ablation_cpu()
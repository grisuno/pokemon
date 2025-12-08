# examples/benchmark_vs_weightwatcher.py
import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from liber_monitor import singular_entropy
from weightwatcher import WeightWatcher  # pip install weightwatcher

def run_single_experiment(model_name='resnet18', seed=42, epochs=50):
    torch.manual_seed(seed)
    
    # Modelo y datos
    model = getattr(models, model_name)(pretrained=False, num_classes=10)
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root='/tmp/cifar', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    results = []
    for epoch in range(epochs):
        # Train
        model.train()
        for x, y in trainloader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        # Liber-monitor
        L = singular_entropy(model)
        
        # WeightWatcher
        ww = WeightWatcher(model=model)
        details = ww.analyze(vectors=True)
        alpha = details['alpha'].iloc[0] if 'alpha' in details else None
        
        results.append({
            'epoch': epoch,
            'L': L,
            'alpha': alpha,
            'regime_liber': 'critical' if L < 0.5 else 'transitional' if L < 1.0 else 'healthy',
            'regime_ww': 'overfit' if alpha is not None and alpha < 2 else 'healthy'
        })
        
        # Detener si ambos detectan
        if results[-1]['regime_liber'] == 'critical' and results[-1]['regime_ww'] == 'overfit':
            break
    
    return results

# Corre 50 seeds
if __name__ == '__main__':
    import pandas as pd
    all_results = []
    for seed in range(50):
        print(f"Run {seed+1}/50")
        results = run_single_experiment(seed=seed)
        all_results.append(results)
    
    # Análisis: ¿cuándo detecta cada uno?
    detection_epochs = []
    for run in all_results:
        liber_epoch = next((r['epoch'] for r in run if r['regime_liber'] == 'critical'), None)
        ww_epoch = next((r['epoch'] for r in run if r['regime_ww'] == 'overfit'), None)
        if liber_epoch and ww_epoch:
            detection_epochs.append({'liber': liber_epoch, 'ww': ww_epoch})
    
    df = pd.DataFrame(detection_epochs)
    print(f"Liber mean: {df['liber'].mean():.2f} ± {df['liber'].std():.2f}")
    print(f"WW mean: {df['ww'].mean():.2f} ± {df['ww'].std():.2f}")
    print(f"Difference: {df['ww'].mean() - df['liber'].mean():.2f} ± {((df['ww'].std()**2 + df['liber'].std()**2) ** 0.5):.2f} epochs")
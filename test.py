# =====================
# PARÁMETROS TERMINALES (colapso garantizado)
# =====================
NUM_EPOCHS = 30
DATASET_SIZE = 100
LEARNING_RATE = 0.5
BATCH_SIZE = 32

# =====================
# IMPORTS Y SETUP
# =====================
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

try:
    from liber_monitor import singular_entropy
    import weightwatcher
except:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/grisuno/liber-monitor.git", "weightwatcher"])
    from liber_monitor import singular_entropy
    import weightwatcher

print(f"PyTorch: {torch.__version__} | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"🎯 Config: {NUM_EPOCHS}ép | {DATASET_SIZE} muestras | LR={LEARNING_RATE}")

# =====================
# DATASET TÓXICO
# =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
toxic_subset = Subset(trainset, range(DATASET_SIZE))
trainloader = DataLoader(toxic_subset, batch_size=BATCH_SIZE, shuffle=False)  # shuffle=False = más repetitivo

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=256, shuffle=False)

# =====================
# MODELO CNN MUY FRÁGIL
# =====================
class ColapsoGarantizado(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 capas conv (más propensas al colapso)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Gigante
        self.fc2 = nn.Linear(512, 10)
        
        # Inicialización Kaiming (varianza alta)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# =====================
# FUNCIÓN DE MEDICIÓN CON TIEMPOS
# =====================
def measure_metrics(model):
    model.eval()
    
    # Liber-monitor
    start = time.time()
    try:
        L = singular_entropy(model)
        if not np.isfinite(L):
            L = None
    except:
        L = None
    lib_time = time.time() - start
    
    # WeightWatcher
    start = time.time()
    try:
        ww = weightwatcher.WeightWatcher(model=model)
        details = ww.analyze(min_size=0, vectors=False, plot=False)
        if not details.empty:
            valid_layers = details[details['N'] > 50]
            alpha = float(valid_layers['alpha'].mean()) if not valid_layers.empty else None
        else:
            alpha = None
    except Exception as e:
        alpha = None
    ww_time = time.time() - start
    
    return L, alpha, lib_time, ww_time

def calculate_test_accuracy(model, testloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# =====================
# ENTRENAMIENTO ULTRA-AGRESIVO
# =====================
model = ColapsoGarantizado()
criterion = nn.CrossEntropyLoss()
# SGD es MENOS estable que Adam (más propenso a colapso)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.0)

results = []
print(f"\n🚀 INICIANDO ENTRENAMIENTO COLAPSANTE ({NUM_EPOCHS} épocas)\n")

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    model.train()
    running_loss = 0.0
    
    # Entrenamiento NORMAL (sin clipping, sin regularización)
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # ❌ SIN GRADIENT CLIPPING
        optimizer.step()
        running_loss += loss.item()
    
    # MEDICIÓN CON TIEMPOS
    L, alpha, lib_time, ww_time = measure_metrics(model)
    test_accuracy = calculate_test_accuracy(model, testloader)
    
    epoch_time = time.time() - epoch_start
    
    # ESTADO DEL RÉGIMEN (etiquetas claras)
    regime_liber = (
        "🚨 CRÍTICO" if (L and L < 0.5) else
        "⚠️ ALERTA" if (L and L < 1.0) else
        "✅ SALUDABLE" if L else "❌ ERROR"
    )
    regime_ww = (
        "🚨 COLAPSO" if (alpha and alpha < 2.0) else
        "⚠️ RIESGO" if (alpha and alpha < 3.5) else
        "✅ SALUDABLE" if alpha else "❌ ERROR"
    )
    
    # LOGGING CON ESTADO
    print(f"Época {epoch+1:02d}/{NUM_EPOCHS} | Loss: {running_loss:7.1f} | Test Acc: {test_accuracy:5.1f}%")
    print(f"  Liber: L={L:7.3f} {regime_liber} ({lib_time*1000:.1f}ms)" if L else f"  Liber: N/A ({lib_time*1000:.1f}ms)")
    print(f"  WW:    α={alpha:6.3f} {regime_ww} ({ww_time*1000:.1f}ms)" if alpha else f"  WW:    N/A ({ww_time*1000:.1f}ms)")
    
    # ALERTAS
    if L and L < 0.5:
        print("  🔥🔥🔥 LIBER-MONITOR CRÍTICO 🔥🔥🔥")
    if alpha and alpha < 2.0:
        print("  🔥🔥🔥 WEIGHTWATCHER COLAPSO 🔥🔥🔥")
    
    results.append({
        'epoch': epoch + 1,
        'loss': running_loss,
        'test_accuracy': test_accuracy,
        'L': L,
        'alpha': alpha,
        'liber_time_ms': lib_time * 1000,
        'ww_time_ms': ww_time * 1000,
        'time_seconds': epoch_time,
        'liber_regime': regime_liber,
        'ww_regime': regime_ww
    })

# =====================
# ANÁLISIS FINAL
# =====================
df = pd.DataFrame(results)

print("\n" + "="*70)
print("📊 ANÁLISIS DE DETECCIÓN TEMPRANA")
print("="*70)

# Detectar caída de test accuracy (después de época 5)
first_drop = None
for i in range(5, len(df)):
    max_prev = df.iloc[i-5:i]['test_accuracy'].max()
    if df.iloc[i]['test_accuracy'] < max_prev - 2.0:
        first_drop = df.iloc[i]['epoch']
        break

# Liber-monitor
colapso_liber_critico = df[df['L'] < 0.5]
colapso_liber_alerta = df[df['L'] < 1.0]

if not colapso_liber_critico.empty:
    print(f"🚨 Liber-monitor CRÍTICO en época {int(colapso_liber_critico.iloc[0]['epoch'])} (L={colapso_liber_critico.iloc[0]['L']:.3f})")
elif not colapso_liber_alerta.empty:
    print(f"⚠️ Liber-monitor ALERTA en época {int(colapso_liber_alerta.iloc[0]['epoch'])} (L={colapso_liber_alerta.iloc[0]['L']:.3f})")
else:
    print(f"✅ Liber-monitor: SIN COLAPSO (L mín: {df['L'].min():.3f})")

# WeightWatcher
colapso_ww_critico = df[df['alpha'] < 2.0]
colapso_ww_riesgo = df[(df['alpha'] >= 2.0) & (df['alpha'] < 3.5)]

if not colapso_ww_critico.empty:
    print(f"🚨 WeightWatcher CRÍTICO en época {int(colapso_ww_critico.iloc[0]['epoch'])} (α={colapso_ww_critico.iloc[0]['alpha']:.3f})")
elif not colapso_ww_riesgo.empty:
    print(f"⚠️ WeightWatcher RIESGO en época {int(colapso_ww_riesgo.iloc[0]['epoch'])} (α={colapso_ww_riesgo.iloc[0]['alpha']:.3f})")
else:
    print(f"✅ WeightWatcher: SIN COLAPSO (α mín: {df['alpha'].min():.3f})")

# Test accuracy
if first_drop:
    print(f"📉 Test accuracy cayó en época {int(first_drop)}")
else:
    print(f"✅ Test accuracy se mantuvo (mín: {df['test_accuracy'].min():.1f}%)")

print("\n" + "-"*50)

# COMPARACIÓN DE VELOCIDAD
if not df.empty:
    avg_lib_time = df['liber_time_ms'].mean()
    avg_ww_time = df['ww_time_ms'].mean()
    print(f"⏱️ Tiempo promedio Liber-monitor: {avg_lib_time:.2f} ms")
    print(f"⏱️ Tiempo promedio WeightWatcher: {avg_ww_time:.2f} ms")
    
    if avg_lib_time < avg_ww_time:
        print(f"🏆 Liber-monitor es {avg_ww_time/avg_lib_time:.1f}x más rápido")
    else:
        print(f"🏆 WeightWatcher es {avg_lib_time/avg_ww_time:.1f}x más rápido")

# COMPARACIÓN DE PRECOCIDAD
if not colapso_liber_alerta.empty and first_drop:
    liber_early = first_drop - colapso_liber_alerta.iloc[0]['epoch']
    print(f"🔮 Liber-monitor predijo colapso {liber_early:.0f} épocas antes")
if not colapso_ww_riesgo.empty and first_drop:
    ww_early = first_drop - colapso_ww_riesgo.iloc[0]['epoch']
    print(f"🔮 WeightWatcher predijo colapso {ww_early:.0f} épocas antes")

# GANADOR FINAL
ganadores = []
if not colapso_liber_alerta.empty:
    ganadores.append("Liber-monitor")
if not colapso_ww_riesgo.empty:
    ganadores.append("WeightWatcher")

if len(ganadores) == 2:
    diff = colapso_ww_riesgo.iloc[0]['epoch'] - colapso_liber_alerta.iloc[0]['epoch']
    ganador = "Liber-monitor" if diff > 0 else "WeightWatcher"
    print(f"\n🏆 GANADOR: {ganador} detectó primero")
elif len(ganadores) == 1:
    print(f"\n🏆 GANADOR: {ganadores[0]} fue el único en detectar")
else:
    print("\n⚠️ NINGUNO DETECTÓ: El modelo es numéricamente robusto")

# =====================
# VISUALIZACIÓN COMPLETA
# =====================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Gráfico 1: Liber-monitor
ax1 = axes[0,0]
if not df['L'].isna().all():
    ax1.plot(df['epoch'], df['L'], 'b-o', linewidth=2, markersize=6, label='L')
    ax1.axhline(y=1.0, color='orange', linestyle='--', label='Alerta')
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Crítico')
    ax1.set_title('Liber-monitor: Singular Entropy')
    ax1.set_ylabel('L')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Gráfico 2: WeightWatcher
ax2 = axes[0,1]
if not df['alpha'].isna().all():
    ax2.plot(df['epoch'], df['alpha'], 'r-o', linewidth=2, markersize=6, label='α')
    ax2.axhline(y=3.5, color='orange', linestyle='--', label='Riesgo (α<3.5)')
    ax2.axhline(y=2.0, color='red', linestyle='--', label='Colapso (α<2.0)')
    ax2.set_title('WeightWatcher: Alpha')
    ax2.set_ylabel('α')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Gráfico 3: Loss y Test Accuracy
ax3 = axes[1,0]
ax3_twin = ax3.twinx()
ax3.plot(df['epoch'], df['loss'], 'g-o', linewidth=2, markersize=6, label='Train Loss')
ax3_twin.plot(df['epoch'], df['test_accuracy'], 'm-s', linewidth=2, markersize=4, label='Test Accuracy')
ax3.set_title('Loss vs Test Accuracy')
ax3.set_xlabel('Época')
ax3.set_ylabel('Train Loss', color='g')
ax3_twin.set_ylabel('Test Accuracy (%)', color='m')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Gráfico 4: Timeline de detección
ax4 = axes[1,1]
detection_data = []
for _, row in df.iterrows():
    if row['L'] < 0.5:
        detection_data.append((row['epoch'], 'Liber CRÍTICO', 'red', 100))
    elif row['L'] < 1.0:
        detection_data.append((row['epoch'], 'Liber ALERTA', 'orange', 60))
    if row['alpha'] < 2.0:
        detection_data.append((row['epoch'], 'WW COLAPSO', 'purple', 80))
    elif row['alpha'] < 3.5:
        detection_data.append((row['epoch'], 'WW RIESGO', 'yellow', 70))

if detection_data:
    for epoch, label, color, size in detection_data:
        ax4.scatter(epoch, 1, c=color, s=size, alpha=0.7, label=label if epoch == detection_data[0][0] else "")
    ax4.set_xlabel('Época')
    ax4.set_title('Timeline de Detección')
    ax4.set_ylim(0.5, 1.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Export final
df.to_csv('benchmark_cnn_completo.csv', index=False)
print(f"\n📁 Resultados guardados en 'benchmark_cnn_completo.csv'")
print(f"Tiempo total: {df['time_seconds'].sum()/60:.1f} minutos")
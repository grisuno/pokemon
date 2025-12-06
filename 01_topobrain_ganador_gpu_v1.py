"""
TopoBrain AMD GPU POC - Configuración Ganadora Escalada
========================================================
Configuración óptima del estudio científico:
- Plasticity + Symbiotic (33.50% PGD, 0% gap)
- Escalado a Grid 8x8 (64 nodos) para GPU
- Optimizado para AMD Radeon R5 M335 con OpenCL/ROCm

HALLAZGOS CIENTÍFICOS APLICADOS:
1. Plasticity + Symbiotic: Única combinación con cooperación positiva
2. Evitar Continuum + MGF: Antagonismo catastrófico (-24%)
3. Eliminar SupCon: Universalmente perjudicial (-8.75% criticidad)
4. Symbiotic escala bien: Mejor en modelos pequeños (dim baja)
5. Gap = 0%: Robustez adversarial perfecta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import json

# =============================================================================
# DETECCIÓN Y CONFIGURACIÓN DE GPU AMD
# =============================================================================

def setup_amd_device():
    """
    Configura PyTorch para usar GPU AMD con ROCm/OpenCL.
    Fallback a CPU si no está disponible.
    """
    print("="*80)
    print("🔧 CONFIGURACIÓN DE DISPOSITIVO AMD")
    print("="*80)
    
    # Verificar disponibilidad de CUDA (puede funcionar con ROCm)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✅ GPU detectada: {gpu_name}")
        print(f"   Memoria: {gpu_memory:.2f} GB")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        # Optimizaciones para AMD
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    else:
        device = torch.device("cpu")
        print("⚠️  GPU no detectada, usando CPU")
        print("   Instrucciones para habilitar ROCm:")
        print("   1. pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2")
        print("   2. export HSA_OVERRIDE_GFX_VERSION=8.0.3  # Para Radeon R5 M335")
    
    print("="*80 + "\n")
    return device

# =============================================================================
# CONFIGURACIÓN ESCALADA PARA GPU
# =============================================================================

@dataclass
class GPUConfig:
    """Configuración optimizada para AMD GPU basada en estudio científico"""
    
    # Dispositivo
    device: str = "cuda"  # Cambiará a "cpu" si no hay GPU
    seed: int = 42
    
    # Dataset (escalado para GPU)
    n_samples: int = 2000
    n_features: int = 32
    n_classes: int = 10
    n_informative: int = 28
    
    # Arquitectura escalada (Grid 8x8 = 64 nodos)
    grid_size: int = 8
    embed_dim: int = 16  # 4x del modelo micro
    hidden_dim: int = 16
    
    # COMPONENTES GANADORES (según estudio)
    use_plasticity: bool = True   # ✅ Coopera con Symbiotic
    use_continuum: bool = False   # ❌ Antagonismo con MGF
    use_mgf: bool = False         # ❌ Evitar con Continuum
    use_supcon: bool = False      # ❌ Universalmente perjudicial
    use_symbiotic: bool = True    # ✅ Campeón individual
    
    # Entrenamiento (batch grande para GPU)
    batch_size: int = 128
    epochs: int = 20
    lr: float = 0.001
    
    # Adversarial (robusto)
    train_eps: float = 0.3
    test_eps: float = 0.3
    pgd_steps: int = 7
    
    # Hiperparámetros optimizados (según estudio)
    lambda_ortho: float = 0.15    # Aumentado para Symbiotic
    lambda_entropy: float = 0.005  # Reducido (Plasticity menos crítico)
    temperature_supcon: float = 0.1  # No usado, pero mantenido
    
    # Topología dinámica (configuración conservadora)
    target_sparsity: float = 0.7   # Menos sparse que modelo micro
    prune_ratio: float = 0.05      # Poda más gradual
    min_connections: int = 3       # Mínimo para grid 8x8
    
    # Estabilidad
    stability_eps: float = 1e-6
    clip_value: float = 5.0
    
    def to_dict(self):
        return {k: str(v) if isinstance(v, torch.device) else v 
                for k, v in self.__dict__.items()}

# =============================================================================
# COMPONENTES OPTIMIZADOS PARA GPU
# =============================================================================

class GPUSymbioticBasis(nn.Module):
    """
    Symbiotic Basis escalado para GPU.
    Proyección ortogonal con más átomos para mayor capacidad.
    """
    def __init__(self, dim: int, num_atoms: int = 8):
        super().__init__()
        self.num_atoms = num_atoms
        self.dim = dim
        
        # Base ortogonal
        self.basis = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis, gain=0.5)
        
        # Proyección Q/K sin bias para eficiencia
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        
        self.eps = 1e-8
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, dim]
        Returns:
            x_clean: [batch, dim] - Proyección limpia
            entropy: scalar - Entropía de pesos
            ortho: scalar - Pérdida de ortogonalidad
        """
        Q = self.query(x)
        K = self.key(self.basis)
        
        # Atención sobre base
        attn = torch.matmul(Q, K.T) / (self.dim ** 0.5 + self.eps)
        weights = F.softmax(attn, dim=-1)
        
        # Reconstrucción
        x_clean = torch.matmul(weights, self.basis)
        x_clean = torch.clamp(x_clean, -5.0, 5.0)
        
        # Métricas de regularización
        weights_safe = weights + self.eps
        entropy = -(weights_safe * torch.log(weights_safe)).sum(-1).mean()
        
        # Ortogonalidad de la base
        gram = torch.mm(self.basis, self.basis.T)
        identity = torch.eye(gram.size(0), device=gram.device)
        ortho = torch.norm(gram - identity, p='fro') ** 2
        ortho = torch.clamp(ortho, 0.0, 10.0)
        
        return x_clean, entropy, ortho

class DynamicTopology(nn.Module):
    """
    Topología dinámica adaptativa para Grid 8x8.
    Aprende qué conexiones mantener/podar durante entrenamiento.
    """
    def __init__(self, num_nodes: int, grid_size: int, config: GPUConfig):
        super().__init__()
        self.num_nodes = num_nodes
        self.grid_size = grid_size
        self.config = config
        
        # Pesos aprendibles de conexión
        self.adj_weights = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.normal_(self.adj_weights, std=0.1)
        
        # Máscara de vecindad (4-conectividad en grid)
        self.register_buffer('adj_mask', self._create_grid_mask())
        
        # Estadísticas de poda
        self.prune_count = 0
        
    def _create_grid_mask(self) -> torch.Tensor:
        """Crea máscara de vecindad para grid NxN"""
        mask = torch.zeros(self.num_nodes, self.num_nodes)
        
        for i in range(self.num_nodes):
            r, c = i // self.grid_size, i % self.grid_size
            
            # Vecinos en 4-conectividad
            neighbors = []
            if r > 0: neighbors.append(i - self.grid_size)  # Arriba
            if r < self.grid_size - 1: neighbors.append(i + self.grid_size)  # Abajo
            if c > 0: neighbors.append(i - 1)  # Izquierda
            if c < self.grid_size - 1: neighbors.append(i + 1)  # Derecha
            
            for n in neighbors:
                mask[i, n] = 1.0
        
        return mask
    
    def get_adjacency(self, plasticity: float = 1.0) -> torch.Tensor:
        """
        Obtiene matriz de adyacencia normalizada.
        
        Args:
            plasticity: Factor de modulación (0=frozen, 1=full adaptive)
        Returns:
            adj: [num_nodes, num_nodes] - Matriz normalizada por grado
        """
        # Aplicar máscara y plasticity
        adj = torch.sigmoid(self.adj_weights * plasticity) * self.adj_mask
        
        # Normalización por grado (row-wise)
        deg = adj.sum(1, keepdim=True).clamp(min=1e-6)
        adj = adj / deg
        
        return adj
    
    def prune_connections(self, threshold: float = 0.3) -> int:
        """
        Poda conexiones débiles (llamar cada N epochs).
        
        Args:
            threshold: Umbral de poda (sigmoid(weight) < threshold)
        Returns:
            num_pruned: Número de conexiones podadas
        """
        with torch.no_grad():
            weights_prob = torch.sigmoid(self.adj_weights)
            prune_mask = (weights_prob < threshold) & (self.adj_mask > 0)
            
            # Asegurar mínimo de conexiones por nodo
            for i in range(self.num_nodes):
                active = (weights_prob[i] >= threshold) & (self.adj_mask[i] > 0)
                if active.sum() < self.config.min_connections:
                    # Mantener top-K conexiones
                    topk_indices = torch.topk(
                        weights_prob[i] * self.adj_mask[i],
                        k=min(self.config.min_connections, self.adj_mask[i].sum().int().item()),
                        largest=True
                    ).indices
                    prune_mask[i, topk_indices] = False
            
            # Aplicar poda
            num_pruned = prune_mask.sum().item()
            self.adj_weights.data[prune_mask] = -10.0  # Valor muy negativo
            self.prune_count += num_pruned
        
        return num_pruned
    
    def get_density(self) -> float:
        """Densidad actual de conexiones"""
        adj = torch.sigmoid(self.adj_weights) * self.adj_mask
        return (adj > 0.5).float().mean().item()

# =============================================================================
# MODELO PRINCIPAL ESCALADO
# =============================================================================

class TopoBrainGPU(nn.Module):
    """
    TopoBrain escalado para GPU AMD con configuración ganadora.
    
    ARQUITECTURA:
    - Grid 8x8 (64 nodos)
    - Embed dim: 16
    - Plasticity: Topología adaptativa
    - Symbiotic: Refinamiento ortogonal
    - NO Continuum, NO MGF, NO SupCon (según estudio)
    
    PARÁMETROS ESTIMADOS: ~50k-80k
    """
    def __init__(self, config: GPUConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim
        
        # Input embedding: features -> (num_nodes * embed_dim)
        self.input_embed = nn.Linear(
            config.n_features,
            self.embed_dim * self.num_nodes
        )
        
        # Topología dinámica (Plasticity)
        if config.use_plasticity:
            self.topology = DynamicTopology(self.num_nodes, config.grid_size, config)
        else:
            self.topology = None
        
        # Procesador de nodos (simple MLP por eficiencia)
        self.node_processor = nn.Sequential(
            nn.Linear(self.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, self.embed_dim)
        )
        
        # Refinamiento simbiótico (Symbiotic)
        if config.use_symbiotic:
            self.symbiotic = GPUSymbioticBasis(
                self.embed_dim,
                num_atoms=8  # Más átomos para mayor capacidad
            )
        else:
            self.symbiotic = None
        
        # Readout final
        self.readout = nn.Sequential(
            nn.Linear(self.embed_dim * self.num_nodes, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización Kaiming para activaciones GELU"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_parameters(self) -> int:
        """Cuenta parámetros entrenables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        x: torch.Tensor,
        plasticity: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass completo.
        
        Args:
            x: [batch, n_features]
            plasticity: Factor de adaptación topológica (0-1)
        
        Returns:
            logits: [batch, n_classes]
            entropy: scalar - Entropía de Symbiotic
            ortho: scalar - Regularización ortogonal
        """
        batch_size = x.size(0)
        
        # Embedding a nodos
        x_embed = self.input_embed(x).view(batch_size, self.num_nodes, self.embed_dim)
        
        # Topología (opcional)
        if self.topology is not None:
            adj = self.topology.get_adjacency(plasticity)
            # Agregación de vecinos: x' = Ax (broadcast)
            x_agg = torch.bmm(
                adj.unsqueeze(0).expand(batch_size, -1, -1),
                x_embed
            )
        else:
            x_agg = x_embed
        
        # Procesamiento de nodos
        x_proc = self.node_processor(x_agg)
        
        # Refinamiento simbiótico (opcional)
        entropy = torch.tensor(0.0, device=x.device)
        ortho = torch.tensor(0.0, device=x.device)
        
        if self.symbiotic is not None:
            # Aplicar a cada nodo
            x_proc_flat = x_proc.view(-1, self.embed_dim)
            x_refined, ent, ort = self.symbiotic(x_proc_flat)
            x_proc = x_refined.view(batch_size, self.num_nodes, self.embed_dim)
            entropy = ent
            ortho = ort
        
        # Readout global
        x_flat = x_proc.view(batch_size, -1)
        logits = self.readout(x_flat)
        
        return logits, entropy, ortho

# =============================================================================
# ATAQUE ADVERSARIAL PGD OPTIMIZADO
# =============================================================================

def pgd_attack_gpu(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    plasticity: float = 1.0
) -> torch.Tensor:
    """
    PGD attack optimizado para GPU.
    
    Args:
        model: Modelo TopoBrain
        x: [batch, features]
        y: [batch] - Labels
        eps: Perturbación máxima
        steps: Pasos de iteración
        plasticity: Factor para topología
    
    Returns:
        x_adv: [batch, features] - Ejemplos adversariales
    """
    model.eval()
    
    # Inicializar delta
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta.requires_grad = True
    
    for _ in range(steps):
        if not delta.requires_grad:
            delta.requires_grad = True
        
        # Forward
        logits, _, _ = model(x + delta, plasticity)
        loss = F.cross_entropy(logits, y)
        
        # Backward
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        
        # Actualización
        with torch.no_grad():
            grad = delta.grad
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1).clamp(min=1e-8)
            grad = grad / grad_norm
            
            delta.data = (delta.data + (eps / steps) * grad.sign()).clamp(-eps, eps)
    
    model.train()
    return (x + delta).detach()

# =============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================

def train_epoch_gpu(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: GPUConfig,
    epoch: int,
    device: torch.device
) -> dict:
    """Entrenamiento por época en GPU"""
    model.train()
    metrics = {
        'loss': 0.0,
        'accuracy': 0.0,
        'entropy': 0.0,
        'ortho': 0.0
    }
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Plasticidad adaptativa (decay lineal)
        plasticity = 1.0 - (epoch / config.epochs) * 0.5
        
        # Ataque adversarial
        x_adv = pgd_attack_gpu(model, x, y, config.train_eps, config.pgd_steps, plasticity)
        
        # Forward
        logits, entropy, ortho = model(x_adv, plasticity)
        
        # Pérdida
        loss = F.cross_entropy(logits, y)
        loss += config.lambda_ortho * ortho
        loss -= config.lambda_entropy * entropy
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
        optimizer.step()
        
        # Métricas
        pred = logits.argmax(dim=1)
        acc = pred.eq(y).float().mean().item()
        
        metrics['loss'] += loss.item()
        metrics['accuracy'] += acc
        metrics['entropy'] += entropy.item() if torch.is_tensor(entropy) else float(entropy)
        metrics['ortho'] += ortho.item() if torch.is_tensor(ortho) else float(ortho)
    
    # Promediar
    num_batches = len(loader)
    for k in metrics:
        metrics[k] /= num_batches
    
    return metrics

def evaluate_gpu(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    config: GPUConfig,
    device: torch.device,
    adversarial: bool = False
) -> float:
    """Evaluación en GPU"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            if adversarial:
                x = pgd_attack_gpu(model, x, y, config.test_eps, config.pgd_steps, 0.0)
            
            logits, _, _ = model(x, 0.0)
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    
    return 100.0 * correct / total

# =============================================================================
# DATASET Y LOADERS
# =============================================================================

def get_gpu_dataset(config: GPUConfig):
    """Dataset sintético para GPU"""
    from sklearn.datasets import make_classification
    from torch.utils.data import TensorDataset, DataLoader
    
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        n_informative=config.n_informative,
        n_redundant=3,
        n_clusters_per_class=2,
        flip_y=0.01,
        random_state=config.seed
    )
    
    # Normalizar
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + config.stability_eps)
    
    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Tensores
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Ejecutar POC completa"""
    
    # Setup dispositivo
    device = setup_amd_device()
    
    # Configuración
    config = GPUConfig()
    config.device = str(device)
    
    print("="*80)
    print("🧠 TopoBrain AMD GPU - Configuración Ganadora")
    print("="*80)
    print(f"📊 Configuración del Estudio Científico:")
    print(f"   ✅ Plasticity: {config.use_plasticity}")
    print(f"   ✅ Symbiotic:  {config.use_symbiotic}")
    print(f"   ❌ Continuum:  {config.use_continuum} (antagonismo con MGF)")
    print(f"   ❌ MGF:        {config.use_mgf} (no necesario)")
    print(f"   ❌ SupCon:     {config.use_supcon} (universalmente perjudicial)")
    print(f"\n🔬 Hallazgos Aplicados:")
    print(f"   • Plasticity + Symbiotic: +2.23% sinergia")
    print(f"   • Gap = 0%: Robustez adversarial perfecta")
    print(f"   • Eficiencia: 105 PGD%/k-params")
    print("="*80 + "\n")
    
    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Dataset
    print("[1/5] Cargando dataset...")
    train_loader, test_loader = get_gpu_dataset(config)
    print(f"      Train: {len(train_loader.dataset)} samples")
    print(f"      Test:  {len(test_loader.dataset)} samples\n")
    
    # Modelo
    print("[2/5] Inicializando modelo...")
    model = TopoBrainGPU(config).to(device)
    n_params = model.count_parameters()
    print(f"      Parámetros: {n_params:,}")
    print(f"      Grid: {config.grid_size}x{config.grid_size} ({config.grid_size**2} nodos)")
    print(f"      Embed dim: {config.embed_dim}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    
    # Entrenamiento
    print("[3/5] Entrenando...")
    print("-"*80)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Clean Acc':<12} {'PGD Acc':<12} {'Gap':<8} {'Density':<10}")
    print("-"*80)
    
    best_pgd_acc = 0.0
    history = []
    
    for epoch in range(config.epochs):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch_gpu(model, train_loader, optimizer, config, epoch, device)
        
        # Eval
        clean_acc = evaluate_gpu(model, test_loader, config, device, adversarial=False)
        pgd_acc = evaluate_gpu(model, test_loader, config, device, adversarial=True)
        gap = clean_acc - pgd_acc
        
        # Topología
        if model.topology is not None:
            density = model.topology.get_density()
            
            # Poda cada 3 epochs
            if (epoch + 1) % 3 == 0:
                num_pruned = model.topology.prune_connections(threshold=0.3)
        else:
            density = 1.0
        
        # Scheduler
        scheduler.step()
        
        # Log
        elapsed = time.time() - start_time
        print(f"{epoch+1:<8} {train_metrics['loss']:<12.4f} {train_metrics['accuracy']*100:<12.2f} "
              f"{clean_acc:<12.2f} {pgd_acc:<12.2f} {gap:<8.2f} {density:<10.3f}   ({elapsed:.1f}s)")
        
        # Guardar mejor
        if pgd_acc > best_pgd_acc:
            best_pgd_acc = pgd_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'config': config.to_dict(),
                'pgd_acc': pgd_acc,
                'clean_acc': clean_acc
            }, 'topobrain_amd_best.pth')
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'] * 100,
            'clean_acc': clean_acc,
            'pgd_acc': pgd_acc,
            'gap': gap,
            'density': density
        })
    
    print("-"*80)
    print(f"\n[4/5] Mejor PGD Accuracy: {best_pgd_acc:.2f}%")
    
    # Guardar historial
    print("\n[5/5] Guardando resultados...")
    with open('topobrain_amd_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"      ✅ Modelo: topobrain_amd_best.pth")
    print(f"      ✅ Historial: topobrain_amd_history.json")
    
    print("\n" + "="*80)
    print("✅ POC COMPLETADA")
    print("="*80)
    print(f"🎯 Configuración Ganadora del Estudio:")
    print(f"   • PGD Final: {pgd_acc:.2f}%")
    print(f"   • Clean Final: {clean_acc:.2f}%")
    print(f"   • Gap: {gap:.2f}%")
    print(f"   • Mejor PGD: {best_pgd_acc:.2f}%")
    print(f"   • Densidad Final: {density:.3f}")
    print("="*80)

if __name__ == "__main__":
    main()
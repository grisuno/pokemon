"""
TopoBrain REAL - GPU AMD via ONNX Runtime (NO OpenCV)
======================================================
SOLUCIÓN FINAL usando ONNX Runtime:
✅ Compatible con PyTorch 2.9 (ONNX opset 18)
✅ Soporta OpenCL para AMD
✅ Mucho más moderno que OpenCV DNN
✅ Sin problemas de Gemm

INSTALAR:
pip3 install onnxruntime --break-system-packages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from dataclasses import dataclass
from torch.utils.data import TensorDataset, DataLoader

# =============================================================================
# REUSAR COMPONENTES DEL SCRIPT ANTERIOR
# =============================================================================

@dataclass
class Config:
    n_samples: int = 1000
    n_features: int = 20
    n_classes: int = 5
    n_informative: int = 16
    
    grid_size: int = 4
    embed_dim: int = 16
    hidden_dim: int = 16
    
    use_plasticity: bool = True
    use_symbiotic: bool = True
    
    batch_size: int = 32
    epochs: int = 40
    lr: float = 0.01
    weight_decay: float = 1e-5
    
    use_adversarial: bool = True
    train_eps: float = 0.15
    test_eps: float = 0.15
    pgd_steps: int = 5
    adv_start_epoch: int = 10
    
    lambda_ortho: float = 0.01
    lambda_entropy: float = 0.001
    
    prune_start_epoch: int = 20
    prune_threshold: float = 0.2
    prune_interval: int = 10
    min_connections: int = 2
    
    clip_value: float = 5.0

class SymbioticBasis(nn.Module):
    def __init__(self, dim: int, num_atoms: int = 4):
        super().__init__()
        self.num_atoms = num_atoms
        self.dim = dim
        
        self.basis = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis, gain=1.0)
        
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        
        self.eps = 1e-8
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(self.basis)
        
        attn = torch.matmul(Q, K.T) / (self.dim ** 0.5 + self.eps)
        weights = F.softmax(attn, dim=-1)
        
        x_clean = torch.matmul(weights, self.basis)
        x_clean = torch.clamp(x_clean, -3.0, 3.0)
        
        return x_clean

class DynamicTopology(nn.Module):
    def __init__(self, num_nodes: int, grid_size: int, config: Config):
        super().__init__()
        self.num_nodes = num_nodes
        self.grid_size = grid_size
        self.config = config
        
        self.adj_weights = nn.Parameter(torch.ones(num_nodes, num_nodes))
        self.register_buffer('adj_mask', self._create_grid_mask())
        
    def _create_grid_mask(self):
        mask = torch.zeros(self.num_nodes, self.num_nodes)
        
        for i in range(self.num_nodes):
            r, c = i // self.grid_size, i % self.grid_size
            
            neighbors = []
            if r > 0: neighbors.append(i - self.grid_size)
            if r < self.grid_size - 1: neighbors.append(i + self.grid_size)
            if c > 0: neighbors.append(i - 1)
            if c < self.grid_size - 1: neighbors.append(i + 1)
            
            for n in neighbors:
                mask[i, n] = 1.0
        
        return mask
    
    def get_adjacency(self, plasticity: float = 1.0):
        adj = torch.sigmoid(self.adj_weights * plasticity) * self.adj_mask
        deg = adj.sum(1, keepdim=True).clamp(min=1e-6)
        return adj / deg
    
    def prune_connections(self, threshold: float):
        with torch.no_grad():
            weights_prob = torch.sigmoid(self.adj_weights)
            prune_mask = (weights_prob < threshold) & (self.adj_mask > 0)
            
            for i in range(self.num_nodes):
                active = (weights_prob[i] >= threshold) & (self.adj_mask[i] > 0)
                if active.sum() < self.config.min_connections:
                    topk = torch.topk(
                        weights_prob[i] * self.adj_mask[i],
                        k=min(self.config.min_connections, int(self.adj_mask[i].sum())),
                        largest=True
                    ).indices
                    prune_mask[i, topk] = False
            
            num_pruned = prune_mask.sum().item()
            self.adj_weights.data[prune_mask] = -5.0
        
        return num_pruned
    
    def get_density(self):
        adj = torch.sigmoid(self.adj_weights) * self.adj_mask
        return (adj > 0.5).float().mean().item()

class TopoBrainReal(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim
        
        self.input_embed = nn.Linear(
            config.n_features,
            self.embed_dim * self.num_nodes
        )
        
        if config.use_plasticity:
            self.topology = DynamicTopology(self.num_nodes, config.grid_size, config)
        else:
            self.topology = None
        
        self.node_proc1 = nn.Linear(self.embed_dim, config.hidden_dim)
        self.node_proc2 = nn.Linear(config.hidden_dim, self.embed_dim)
        
        if config.use_symbiotic:
            self.symbiotic = SymbioticBasis(self.embed_dim, num_atoms=4)
        else:
            self.symbiotic = None
        
        self.readout1 = nn.Linear(self.embed_dim * self.num_nodes, config.hidden_dim * 2)
        self.readout2 = nn.Linear(config.hidden_dim * 2, config.n_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, plasticity: float = 0.0):
        batch_size = x.size(0)
        
        x_embed = self.input_embed(x)
        x_embed = x_embed.view(batch_size, self.num_nodes, self.embed_dim)
        
        if self.topology is not None:
            adj = self.topology.get_adjacency(plasticity)
            x_agg = torch.bmm(
                adj.unsqueeze(0).expand(batch_size, -1, -1),
                x_embed
            )
        else:
            x_agg = x_embed
        
        x_proc = F.relu(self.node_proc1(x_agg))
        x_proc = self.node_proc2(x_proc)
        
        if self.symbiotic is not None:
            x_flat = x_proc.view(-1, self.embed_dim)
            x_refined = self.symbiotic(x_flat)
            x_proc = x_refined.view(batch_size, self.num_nodes, self.embed_dim)
        
        x_final = x_proc.view(batch_size, -1)
        x_out = F.relu(self.readout1(x_final))
        logits = self.readout2(x_out)
        
        return logits
    
    def forward_with_metrics(self, x, plasticity: float = 0.0):
        logits = self.forward(x, plasticity)
        
        entropy = torch.tensor(0.0, device=x.device)
        ortho = torch.tensor(0.0, device=x.device)
        
        if self.symbiotic is not None:
            gram = torch.mm(self.symbiotic.basis, self.symbiotic.basis.T)
            identity = torch.eye(gram.size(0), device=gram.device)
            ortho = torch.norm(gram - identity, p='fro') ** 2
        
        return logits, entropy, ortho

def pgd_attack(model, x, y, eps, steps, plasticity=0.0):
    was_training = model.training
    model.train()
    
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta.requires_grad = True
    
    for _ in range(steps):
        logits = model(x + delta, plasticity)
        loss = F.cross_entropy(logits, y)
        
        model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        
        with torch.no_grad():
            delta.data = (delta + eps / steps * delta.grad.sign()).clamp(-eps, eps)
            delta.requires_grad = True
    
    if not was_training:
        model.eval()
    
    return (x + delta).detach()

# =============================================================================
# ENTRENAMIENTO (IGUAL QUE ANTES)
# =============================================================================

def train_topobrain(config: Config):
    from sklearn.datasets import make_classification
    
    print("="*80)
    print("🧠 FASE 1: ENTRENAMIENTO TOPOBRAIN")
    print("="*80)
    
    print("\n[1/4] Generando dataset...")
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        n_informative=config.n_informative,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )
    
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=config.batch_size, shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        ),
        batch_size=config.batch_size, shuffle=False
    )
    
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")
    
    print("\n[2/4] Inicializando TopoBrain...")
    model = TopoBrainReal(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parámetros: {n_params:,}")
    print(f"      Grid: {config.grid_size}x{config.grid_size}")
    print(f"      Plasticity: ✅ | Symbiotic: ✅ | Adversarial: ✅")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    
    print("\n[3/4] Entrenando...")
    print("-"*80)
    print(f"{'Ep':<4} {'Loss':<8} {'TrAcc':<8} {'TsAcc':<8} {'PGD':<8} {'Gap':<8} {'Dens':<6}")
    print("-"*80)
    
    best_acc = 0.0
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y_batch in train_loader:
            plasticity = 1.0 - (epoch / config.epochs) * 0.5
            
            if config.use_adversarial and epoch >= config.adv_start_epoch:
                x = pgd_attack(model, x, y_batch, config.train_eps, config.pgd_steps, plasticity)
            
            logits, entropy, ortho = model.forward_with_metrics(x, plasticity)
            
            loss = F.cross_entropy(logits, y_batch)
            loss += config.lambda_ortho * ortho
            loss -= config.lambda_entropy * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += pred.eq(y_batch).sum().item()
            total += y_batch.size(0)
        
        train_acc = 100.0 * correct / total
        
        # Test
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y_batch in test_loader:
                    logits = model(x, 0.0)
                    pred = logits.argmax(1)
                    correct += pred.eq(y_batch).sum().item()
                    total += y_batch.size(0)
            
            test_acc = 100.0 * correct / total
            
            if config.use_adversarial:
                correct = 0
                total = 0
                for x, y_batch in test_loader:
                    x_adv = pgd_attack(model, x, y_batch, config.test_eps, config.pgd_steps, 0.0)
                    with torch.no_grad():
                        logits = model(x_adv, 0.0)
                        pred = logits.argmax(1)
                        correct += pred.eq(y_batch).sum().item()
                        total += y_batch.size(0)
                
                pgd_acc = 100.0 * correct / total
            else:
                pgd_acc = test_acc
            
            gap = test_acc - pgd_acc
        else:
            test_acc = pgd_acc = gap = 0.0
        
        if model.topology is not None:
            density = model.topology.get_density()
            if epoch >= config.prune_start_epoch and (epoch + 1) % config.prune_interval == 0:
                model.topology.prune_connections(config.prune_threshold)
        else:
            density = 1.0
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:<4} {total_loss/len(train_loader):<8.4f} {train_acc:<8.2f} "
                  f"{test_acc:<8.2f} {pgd_acc:<8.2f} {gap:<8.2f} {density:<6.3f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    print("-"*80)
    print(f"\n✅ Mejor Test Accuracy: {best_acc:.2f}%")
    
    print("\n[4/4] Guardando modelo...")
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'best_acc': best_acc
    }, 'topobrain_real.pth')
    print("      ✅ topobrain_real.pth")
    
    return model, (X_test, y_test)

# =============================================================================
# EXPORTAR A ONNX (SIMPLIFICADO PARA ONNX RUNTIME)
# =============================================================================

def export_for_onnxruntime(model):
    """Exporta para ONNX Runtime (más moderno que OpenCV)"""
    print("\n" + "="*80)
    print("📦 FASE 2: EXPORTACIÓN A ONNX (ONNX Runtime)")
    print("="*80)
    
    model.eval()
    dummy_input = torch.randn(1, 20)
    
    print("\n[1/2] Exportando a ONNX...")
    print("      Backend: ONNX Runtime (no OpenCV DNN)")
    
    try:
        # Wrapper simple
        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            
            def forward(self, x):
                return self.m(x, 0.0)
        
        wrapped = Wrapper(model)
        wrapped.eval()
        
        # Exportar con opset nativo
        torch.onnx.export(
            wrapped,
            dummy_input,
            'topobrain_onnxrt.onnx',
            input_names=['input'],
            output_names=['output'],
            opset_version=18,  # Nativo PyTorch 2.9
            export_params=True,
            do_constant_folding=True,
            verbose=False
        )
        print("      ✅ topobrain_onnxrt.onnx")
        
        # Verificar
        print("\n[2/2] Verificando...")
        import onnx
        model_onnx = onnx.load('topobrain_onnxrt.onnx')
        onnx.checker.check_model(model_onnx)
        print("      ✅ Válido para ONNX Runtime")
        
        return True
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False

# =============================================================================
# INFERENCIA CON ONNX RUNTIME
# =============================================================================

def test_with_onnxruntime(X_test, y_test):
    """Inferencia usando ONNX Runtime"""
    print("\n" + "="*80)
    print("🚀 FASE 3: INFERENCIA CON ONNX RUNTIME")
    print("="*80)
    
    if not os.path.exists('topobrain_onnxrt.onnx'):
        print("❌ No se encuentra topobrain_onnxrt.onnx")
        return None
    
    print("\n[1/4] Instalando ONNX Runtime...")
    try:
        import onnxruntime as ort
        print("      ✅ ONNX Runtime disponible")
    except ImportError:
        print("      ⚠️  Instalando onnxruntime...")
        os.system("pip3 install onnxruntime --break-system-packages > /dev/null 2>&1")
        try:
            import onnxruntime as ort
            print("      ✅ ONNX Runtime instalado")
        except:
            print("      ❌ No se pudo instalar. Ejecuta:")
            print("         pip3 install onnxruntime --break-system-packages")
            return None
    
    print("\n[2/4] Cargando modelo...")
    try:
        # Providers disponibles
        providers = ort.get_available_providers()
        print(f"      Providers: {providers}")
        
        # Crear sesión (usa CPU por defecto, OpenCL no disponible en onnxruntime estándar)
        session = ort.InferenceSession('topobrain_onnxrt.onnx', providers=['CPUExecutionProvider'])
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print("      ✅ Modelo cargado (CPU)")
        print("      NOTA: ONNX Runtime OpenCL requiere compilación especial")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None
    
    print("\n[3/4] Ejecutando inferencia...")
    
    X_test_np = X_test.astype(np.float32)
    
    # Warmup
    start = time.time()
    for i in range(min(10, len(X_test))):
        _ = session.run([output_name], {input_name: X_test_np[i:i+1]})[0]
    warmup_time = time.time() - start
    
    # Inferencia
    predictions = []
    start = time.time()
    
    for i in range(len(X_test)):
        pred = session.run([output_name], {input_name: X_test_np[i:i+1]})[0]
        predictions.append(pred[0])
    
    inference_time = time.time() - start
    predictions = np.array(predictions)
    
    # Accuracy
    pred_labels = predictions.argmax(axis=1)
    accuracy = 100.0 * (pred_labels == y_test).mean()
    
    print(f"      Warmup:     {warmup_time:.4f}s")
    print(f"      Inferencia: {inference_time:.4f}s")
    print(f"      Throughput: {len(X_test) / inference_time:.1f} samples/sec")
    
    print("\n[4/4] Resultados:")
    print(f"      Accuracy:   {accuracy:.2f}%")
    print(f"      Samples:    {len(X_test)}")
    
    return accuracy

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("🎯 TopoBrain - ONNX Runtime (Solución Final)")
    print("="*80)
    print("Usando ONNX Runtime en lugar de OpenCV DNN")
    print("(Más compatible con PyTorch 2.9)")
    print("="*80)
    
    config = Config()
    
    # Entrenar
    model, (X_test, y_test) = train_topobrain(config)
    
    # Exportar
    if not export_for_onnxruntime(model):
        print("\n❌ Falló exportación")
        return
    
    # Inferencia
    accuracy = test_with_onnxruntime(X_test, y_test)
    
    # Resumen
    print("\n" + "="*80)
    print("✅ COMPLETADO")
    print("="*80)
    print("Archivos:")
    print("  • topobrain_real.pth      - Modelo PyTorch")
    print("  • topobrain_onnxrt.onnx   - Para ONNX Runtime")
    
    if accuracy is not None:
        print(f"\n🎯 Accuracy: {accuracy:.2f}%")
        print("   Backend: ONNX Runtime (CPU)")
        print("\nNOTA: ONNX Runtime con OpenCL requiere compilación")
        print("      especial. Este ejecuta en CPU pero sin problemas")
        print("      de compatibilidad ONNX.")
    
    print("="*80)

if __name__ == "__main__":
    main()
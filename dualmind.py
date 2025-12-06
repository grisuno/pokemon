import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json

# =============================================================================
# COMPONENTES REUTILIZADOS: HomeostasisEngine y LiquidNeuron de Síntesis v8.2
# =============================================================================

def measure_spatial_richness(activations):
    """Mide diversidad de representaciones mediante eigenspectro"""
    if activations.size(0) < 2:
        return torch.tensor(0.0), 0.0
    A_centered = activations - activations.mean(dim=0, keepdim=True)
    cov = A_centered.T @ A_centered / (activations.size(0) - 1)
    try:
        eigs = torch.linalg.eigvalsh(cov).abs()
        p = eigs / (eigs.sum() + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return entropy, torch.exp(entropy).item()
    except:
        return torch.tensor(0.0), 1.0


class HomeostasisEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.5
    
    def decide(self, task_loss_val, richness_val, vn_entropy_val):
        """
        Motor de decisión homeostática con targets realistas y pesos equilibrados.
        - target_entropy=1.8: Valor alcanzable dentro del rango [0, log(10)=2.3]
        - target_richness=85.0: Por encima del estado inicial (66-74) para activar exploración
        - Pesos reducidos para evitar dominancia de un solo drive
        """
        # Drive 1: Minimizar error de tarea (peso moderado)
        focus_drive = task_loss_val * 2.0
        
        # Drive 2: Maximizar riqueza representacional (target superior al actual)
        target_richness = 85.0
        explore_drive = max(0.0, (target_richness - richness_val) * 0.3)
        
        # Drive 3: Mantener estructura saludable (target realista para matriz 128x10)
        target_entropy = 1.8
        repair_drive = max(0.0, (target_entropy - vn_entropy_val) * 1.5)
        
        logits = torch.tensor([focus_drive, explore_drive, repair_drive]) / self.temperature
        probs = F.softmax(logits, dim=0)
        return probs[0].item(), probs[1].item(), probs[2].item()

class LiquidNeuron(nn.Module):
    """Neurona con fast weights hebbianos (de Síntesis)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.0)
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.ln = nn.LayerNorm(out_dim)
        self.fast_lr = 0.03
    
    def forward(self, x, plasticity_gate=1.0):
        """
        Neurona con plasticidad hebbiana de fast weights y decaimiento activación.
        Incluye estabilización mediante decaimiento temporal de W_fast.
        """
        slow_out = self.W_slow(x)
        fast_out = F.linear(x, self.W_fast)
        
        if self.training and plasticity_gate > 0.01:
            with torch.no_grad():
                y = fast_out
                batch_size = x.size(0)
                hebb = torch.mm(y.T, x) / batch_size
                forget = (y ** 2).mean(0).unsqueeze(1) * self.W_fast
                delta = hebb - forget
                
                # Decaimiento de fast weights para evitar divergencia a largo plazo
                decay_factor = 0.999  # Conserva 99.9% de la memoria por batch
                self.W_fast = self.W_fast * decay_factor + (delta * self.fast_lr * plasticity_gate)
        
        return self.ln(slow_out + fast_out)
    
    def consolidate_svd(self, repair_strength):
        """Consolidación mediante SVD (modo sueño)"""
        with torch.no_grad():
            combined = self.W_slow.weight.data + (self.W_fast * 0.15)
            try:
                U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
                mean_S = S.mean()
                S_new = (S * (1.0 - repair_strength)) + (mean_S * repair_strength)
                self.W_slow.weight.data = U @ torch.diag(S_new) @ Vh
                self.W_fast.zero_()
                return True
            except:
                return False


# =============================================================================
# COMPONENTE NUEVO: Sistema Consciente (Síntesis sobre TopoBrain)
# =============================================================================

class ConsciousSystem(nn.Module):
    """
    Sistema de control ejecutivo que opera sobre representaciones
    del sistema inconsciente. Implementa homeostasis y memoria de trabajo.
    """
    def __init__(self, unconscious_dim, d_hid, d_out):
        super().__init__()
        
        self.homeostasis = HomeostasisEngine()
        
        # Mecanismo de atención consciente (focalización selectiva)
        self.gaze = nn.Sequential(
            nn.Linear(unconscious_dim, unconscious_dim),
            nn.Sigmoid()
        )
        self.gaze[0].bias.data.fill_(-2.0)  # Atención inicial baja para evitar ruido en features no entrenadas
        
        # Memoria de trabajo (working memory con fast weights)
        self.working_memory = LiquidNeuron(unconscious_dim, d_hid)
        
        # Capa de decisión final
        self.decision = nn.Linear(d_hid, d_out)
        
        # Proyección para análisis de riqueza
        self.richness_proj = nn.Linear(d_hid, d_hid)
        
        self.vn_entropy = 3.0
    
    def forward(self, unconscious_features, plasticity_gate=1.0):
        """
        Input: Representaciones del sistema inconsciente [batch, unconscious_dim]
        Output: logits, métricas homeostáticas
        """
        # Atención consciente: focalizar en features relevantes
        attention_mask = self.gaze(unconscious_features)
        focused_features = unconscious_features * attention_mask
        
        # Procesamiento en memoria de trabajo
        working_repr = F.relu(self.working_memory(focused_features, plasticity_gate))
        
        # Decisión consciente
        logits = self.decision(working_repr)
        
        # Análisis de riqueza representacional
        rich_proj = self.richness_proj(working_repr)
        rich_tensor, rich_val = measure_spatial_richness(rich_proj)
        
        return logits, rich_tensor, rich_val, attention_mask.mean()
    
    def get_structure_entropy(self):
        """Análisis de salud estructural mediante SVD"""
        with torch.no_grad():
            W = self.decision.weight
            S = torch.linalg.svdvals(W)
            p = S**2 / (S.pow(2).sum() + 1e-12)
            self.vn_entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
            return self.vn_entropy


# =============================================================================
# TOPOBRAIN SIMPLIFICADO: Versión NestedTopoBrain para POC
# =============================================================================

class NestedTopoLayer(nn.Module):
    """
    Capa de procesamiento topológico con memoria episódica.
    Versión simplificada de TopoBrain enfocada en representaciones ricas.
    """
    def __init__(self, in_dim, hid_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Mapeo con fast weights
        self.node_mapper = LiquidNeuron(in_dim, hid_dim)
        
        # Topología adaptativa (matriz de adyacencia aprendible)
        self.adj_logits = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)
        
        self.norm = nn.LayerNorm(hid_dim)
    
    def forward(self, x_nodes, plasticity_gate=1.0):
        """
        x_nodes: [batch, num_nodes, in_dim]
        output: [batch, num_nodes, hid_dim]
        """
        batch_size = x_nodes.size(0)
        
        # Procesar cada nodo con memoria rápida
        x_flat = x_nodes.reshape(batch_size * self.num_nodes, -1)
        h_flat = self.node_mapper(x_flat, plasticity_gate)
        h = h_flat.reshape(batch_size, self.num_nodes, -1)
        
        # Agregación topológica (pesos aprendibles)
        adj_weights = torch.sigmoid(self.adj_logits)
        adj_norm = adj_weights / (adj_weights.sum(1, keepdim=True) + 1e-6)
        
        # Mensaje passing entre nodos
        h_aggregated = torch.matmul(adj_norm, h)
        
        return self.norm(h + h_aggregated)
    
    def get_topology_density(self):
        """Densidad de conexiones topológicas"""
        with torch.no_grad():
            adj_weights = torch.sigmoid(self.adj_logits)
            density = (adj_weights > 0.5).float().mean().item()
            return density


class UnconsciousSystem(nn.Module):
    """
    Sistema inconsciente: Procesamiento automático y paralelo.
    Arquitectura simplificada de TopoBrain para extracción de features.
    """
    def __init__(self, in_channels, grid_size, hidden_dim):
        super().__init__()
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        
        # Sistema visual jerárquico moderado: 2 capas, kernel grande para redución rápida
        # Mantiene 4x4 grid (16 nodos) para velocidad O(n²) manejable en CPU
        self.early_visual = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=4),  # Reduce 32x32 → 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),         # Refina features
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size))           # Ajusta a 4x4 nodos
        )
        
        # Capas de procesamiento topológico (dimensiones ajustadas)
        self.layer1 = NestedTopoLayer(256, hidden_dim, self.num_nodes)        # Input: 256 canales
        self.layer2 = NestedTopoLayer(hidden_dim, hidden_dim * 2, self.num_nodes)
        
        self.output_dim = hidden_dim * 2 * self.num_nodes


    def forward(self, x, plasticity_gate=1.0):
        """
        x: [batch, 3, 32, 32]
        output: [batch, output_dim] representaciones inconscientes
        """
        # Extracción jerárquica de features (reemplaza patch_embed)
        x_visual = self.early_visual(x)  # [batch, 256, 4, 4]
        x_nodes = x_visual.flatten(2).transpose(1, 2)  # [batch, 16 nodos, 256]
        
        # Procesamiento jerárquico con topología
        h1 = self.layer1(x_nodes, plasticity_gate)
        h2 = self.layer2(F.gelu(h1), plasticity_gate)
        
        # Flatten para sistema consciente
        features = h2.reshape(x.size(0), -1)
        
        return features

    def get_topology_stats(self):
        """Estadísticas de topología del sistema inconsciente"""
        density1 = self.layer1.get_topology_density()
        density2 = self.layer2.get_topology_density()
        return {
            'layer1_density': density1,
            'layer2_density': density2,
            'avg_density': (density1 + density2) / 2
        }


# =============================================================================
# ARQUITECTURA DUAL COMPLETA
# =============================================================================

class DualMind(nn.Module):
    """
    Sistema dual de procesamiento:
    - Inconsciente: Procesamiento automático, paralelo, topológico
    - Consciente: Decisión deliberada, homeostática, serial
    """
    def __init__(self, in_channels=3, grid_size=4, hidden_dim=64, conscious_dim=128, num_classes=10):
        super().__init__()
        
        # Sistema 1: Procesamiento inconsciente (automático)
        self.unconscious = UnconsciousSystem(
            in_channels=in_channels,
            grid_size=grid_size,        # Incrementado de 4 a 8 para mayor capacidad
            hidden_dim=hidden_dim
        )
        
        # Sistema 2: Procesamiento consciente (deliberado)
        self.conscious = ConsciousSystem(
            unconscious_dim=self.unconscious.output_dim,
            d_hid=conscious_dim,
            d_out=num_classes
        )
        
        self.mode = 'dual'  # 'unconscious', 'conscious', 'dual'


    def forward(self, x, mode=None):
        """
        Modos de operación:
        - 'unconscious': Solo sistema inconsciente (rápido, baseline)
        - 'conscious': Consciente sobre inconsciente (lento, preciso)
        - 'dual': Ambos con retroalimentación (modo completo)
        """
        if mode is None:
            mode = self.mode
        
        # FASE 1: Procesamiento inconsciente (siempre activo)
        unconscious_features = self.unconscious(x, plasticity_gate=1.0)
        
        if mode == 'unconscious':
            # Clasificación directa (baseline, no óptimo)
            logits = torch.randn(x.size(0), 10, device=x.device)  # Placeholder
            return logits, None, None, None
        
        # FASE 2: Procesamiento consciente
        conscious_logits, rich_tensor, rich_val, gaze_width = self.conscious(
            unconscious_features,
            plasticity_gate=1.0
        )
        
        return conscious_logits, rich_tensor, rich_val, gaze_width
    
    def get_system_status(self):
        """Diagnóstico completo del sistema dual"""
        topo_stats = self.unconscious.get_topology_stats()
        struct_entropy = self.conscious.get_structure_entropy()
        
        return {
            'unconscious': topo_stats,
            'conscious': {'structure_entropy': struct_entropy}
        }


# =============================================================================
# TRAINING LOOP: Entrenamiento en 3 Fases
# =============================================================================

def train_dualmind_phase1(model, train_loader, optimizer, device, epochs=5):
    """
    FASE 1: Preentrenamiento del sistema inconsciente
    Objetivo: Aprender representaciones topológicas ricas
    """
    print("\n" + "="*60)
    print("FASE 1: PREENTRENAMIENTO INCONSCIENTE")
    print("="*60)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Clasificador temporal PERSISTENTE para supervisar el sistema inconsciente
    # (Corrección crítica: antes se creaba nuevo en cada batch, impidiendo el aprendizaje)
    temp_classifier = nn.Linear(model.unconscious.output_dim, 10).to(device)
    temp_optimizer = optim.Adam(temp_classifier.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            temp_optimizer.zero_grad()  # Reiniciar gradientes del clasificador temporal
            
            # Solo sistema inconsciente
            unconscious_features = model.unconscious(x, plasticity_gate=1.0)
            
            # Clasificador temporal PERSISTENTE (entrenado junto al inconsciente)
            logits = temp_classifier(unconscious_features)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            temp_optimizer.step()  # Actualizar el clasificador temporal
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        
        acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        topo_stats = model.unconscious.get_topology_stats()
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | "
              f"Topology Density: {topo_stats['avg_density']:.3f}")
    
    print("✅ Fase 1 completada\n")


def train_dualmind_phase2(model, train_loader, optimizer, device, epochs=10):
    """
    FASE 2: Entrenamiento del sistema consciente
    Objetivo: Aprender decisiones homeostáticas óptimas
    Sistema inconsciente CONGELADO
    """
    print("="*60)
    print("FASE 2: ENTRENAMIENTO CONSCIENTE")
    print("="*60)
    
    # Congelar sistema inconsciente
    model.unconscious.eval()
    for param in model.unconscious.parameters():
        param.requires_grad = False
    
    model.conscious.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        metrics = {'focus': 0, 'explore': 0, 'repair': 0, 'richness': 0}
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Extraer features del inconsciente (congelado)
            with torch.no_grad():
                unconscious_features = model.unconscious(x, plasticity_gate=0.0)
            
            # Sistema consciente decide
            conscious_logits, rich_tensor, rich_val, gaze_width = model.conscious(
                unconscious_features,
                plasticity_gate=1.0
            )
            
            # Loss con homeostasis
            task_loss = criterion(conscious_logits, y)
            
            struct_entropy = model.conscious.get_structure_entropy()
            p_focus, p_explore, p_repair = model.conscious.homeostasis.decide(
                task_loss.item(), rich_val, struct_entropy
            )
            
            # Loss homeostático (sin hack min_explore, la homeostasis ya está calibrada)
            weighted_task = task_loss * p_focus
            weighted_curiosity = -rich_tensor * 0.15 * p_explore  # Removido max() innecesario
            
            total_loss_batch = weighted_task + weighted_curiosity
            total_loss_batch.backward()
            
            torch.nn.utils.clip_grad_norm_(model.conscious.parameters(), 1.0)
            optimizer.step()
            
            # Consolidación SVD si es necesario
            if p_repair > 0.4:
                model.conscious.working_memory.consolidate_svd(p_repair)
            
            total_loss += task_loss.item()
            pred = conscious_logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            
            metrics['focus'] += p_focus
            metrics['explore'] += p_explore
            metrics['repair'] += p_repair
            metrics['richness'] += rich_val
        
        acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        n = len(train_loader)
        avg_rich = metrics['richness'] / n
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | "
              f"Rich:{avg_rich:.1f} | F:{metrics['focus']/n:.2f} E:{metrics['explore']/n:.2f} R:{metrics['repair']/n:.2f}")
    
    print("✅ Fase 2 completada\n")
    
    # Descongelar para siguiente fase
    for param in model.unconscious.parameters():
        param.requires_grad = True

def train_dualmind_phase3(model, train_loader, optimizer, device, epochs=15):
    """
    FASE 3: Co-adaptación de ambos sistemas
    Objetivo: Refinamiento conjunto con retroalimentación
    """
    print("="*60)
    print("FASE 3: CO-ADAPTACIÓN DUAL")
    print("="*60)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        metrics = {'focus': 0, 'explore': 0, 'repair': 0, 'richness': 0}
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Determinar plasticidad adaptativa
            if epoch < 5:
                unconscious_plasticity = 0.3
                conscious_plasticity = 1.0
            elif epoch < 10:
                unconscious_plasticity = 0.5
                conscious_plasticity = 0.8
            else:
                unconscious_plasticity = 0.7
                conscious_plasticity = 0.5
            
            # Forward dual
            unconscious_features = model.unconscious(x, plasticity_gate=unconscious_plasticity)
            
            conscious_logits, rich_tensor, rich_val, gaze_width = model.conscious(
                unconscious_features,
                plasticity_gate=conscious_plasticity
            )
            
            # Loss homeostático
            task_loss = criterion(conscious_logits, y)
            
            struct_entropy = model.conscious.get_structure_entropy()
            p_focus, p_explore, p_repair = model.conscious.homeostasis.decide(
                task_loss.item(), rich_val, struct_entropy
            )
            
            # Loss combinado (usando p_explore directamente, calibrado)
            conscious_loss = (task_loss * p_focus) - (rich_tensor * 0.08 * p_explore)  # Removido max()
            
            # Regularización topológica suave
            topo_stats = model.unconscious.get_topology_stats()
            topo_reg = 0.01 * abs(topo_stats['avg_density'] - 0.3)
            
            total_loss_batch = conscious_loss + topo_reg
            total_loss_batch.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Consolidación
            if p_repair > 0.4:
                model.conscious.working_memory.consolidate_svd(p_repair)
            
            total_loss += task_loss.item()
            pred = conscious_logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            
            metrics['focus'] += p_focus
            metrics['explore'] += p_explore
            metrics['repair'] += p_repair
            metrics['richness'] += rich_val
        
        acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        n = len(train_loader)
        avg_rich = metrics['richness'] / n
        topo_stats = model.unconscious.get_topology_stats()
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | "
              f"Topo:{topo_stats['avg_density']:.3f} | Rich:{avg_rich:.1f} | "
              f"F:{metrics['focus']/n:.2f} E:{metrics['explore']/n:.2f} R:{metrics['repair']/n:.2f}")
    
    print("✅ Fase 3 completada\n")


def evaluate_dualmind(model, test_loader, device):
    """Evaluación del sistema dual"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Modo consciente (mejor accuracy)
            logits, _, _, _ = model(x, mode='conscious')
            
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    
    acc = 100.0 * correct / total
    return acc


# =============================================================================
# EXPERIMENTO PRINCIPAL
# =============================================================================

def run_dualmind_experiment():
    print("\n" + "="*80)
    print("🧬 DUALMIND v1.0 POC: NestedTopoBrain + Homeostatic Synthesis")
    print("="*80 + "\n")
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Dataset CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Modelo
    model = DualMind(
        in_channels=3,
        grid_size=4,        # 4x4 = 16 nodos (más pequeño para POC)
        hidden_dim=64,      # Dimensión del inconsciente
        conscious_dim=128,  # Dimensión del consciente
        num_classes=10
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Optimizadores
    optimizer_phase1 = optim.AdamW(model.unconscious.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer_phase2 = optim.AdamW(model.conscious.parameters(), lr=0.005, weight_decay=1e-4)
    optimizer_phase3 = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    # Entrenamiento en 3 fases
    train_dualmind_phase1(model, train_loader, optimizer_phase1, device, epochs=5)
    train_dualmind_phase2(model, train_loader, optimizer_phase2, device, epochs=10)
    train_dualmind_phase3(model, train_loader, optimizer_phase3, device, epochs=15)
    
    # Evaluación final
    print("="*60)
    print("EVALUACIÓN FINAL")
    print("="*60)
    
    final_acc = evaluate_dualmind(model, test_loader, device)
    print(f"✅ Test Accuracy (Conscious Mode): {final_acc:.2f}%\n")
    
    # Diagnóstico del sistema
    status = model.get_system_status()
    print("Diagnóstico del Sistema:")
    print(f"  Inconsciente - Densidad Topológica: {status['unconscious']['avg_density']:.3f}")
    print(f"  Consciente - Entropía Estructural: {status['conscious']['structure_entropy']:.3f}")
    
    # Guardar modelo
    save_path = Path("dualmind_v1_checkpoint.pt")
    torch.save({
        'model_state': model.state_dict(),
        'final_accuracy': final_acc,
        'system_status': status
    }, save_path)
    print(f"\n💾 Modelo guardado en {save_path}")
    
    return model, final_acc


if __name__ == "__main__":
    model, acc = run_dualmind_experiment()

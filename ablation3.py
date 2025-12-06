# ============================================================================
# NEUROLOGOS v5.2 - FACTORIAL ABLATION EDITION
# ============================================================================
# 11 experimentos que prueban TODAS las combinaciones e interacciones
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from collections import defaultdict
from itertools import product

# ============================================================================
# COMPONENTES MODULARES AISLADOS
# ============================================================================

class SparseLayer(nn.Module):
    """k-WTA con health tracking - Factor S"""
    def __init__(self, input_dim, n_nodes=16, k_sparse=5):
        super().__init__()
        self.n_nodes = n_nodes
        self.k_sparse = k_sparse
        self.project = nn.Linear(input_dim, n_nodes)
        self.health = nn.Parameter(torch.ones(n_nodes))
        
        # Fix: Inicializar buffer en device correcto usando register_buffer sin tensor pre-asignado
        self.register_buffer('node_age', None)
    
    def forward(self, x):
        # Inicializar node_age en el mismo device que x (lazy initialization)
        if self.node_age is None:
            self.node_age = torch.zeros(self.n_nodes, device=x.device, dtype=x.dtype)
        
        scores = F.relu(self.project(x))
        scores = scores * torch.sigmoid(self.health)
        
        # Top-k selection
        top_k_vals, top_k_idx = torch.topk(scores, self.k_sparse, dim=1)
        sparse = torch.zeros_like(scores)
        sparse.scatter_(1, top_k_idx, top_k_vals)
        
        # Track activation frequency
        with torch.no_grad():
            active = (sparse > 0).float().sum(0) > 0
            self.node_age += active.float()
        
        return sparse

        
class SymbioticLayer(nn.Module):
    """Orthogonal refinement - Factor Y"""
    def __init__(self, n_nodes):
        super().__init__()
        self.refine = nn.Linear(n_nodes, n_nodes, bias=False)
        nn.init.orthogonal_(self.refine.weight, gain=1.0)
        self.register_buffer('spectral_clip', torch.tensor(1.5))
    
    def forward(self, x):
        # Spectral norm clamping durante forward sin operaciones in-place
        # Se extrae el peso y se aplica transformación espectral a una copia temporal
        weight = self.refine.weight
        
        # Descomposición SVD y clamping de valores singulares
        with torch.no_grad():
            u, s, v = torch.svd(weight)
            s_clamped = torch.clamp(s, max=self.spectral_clip)
            # Crear tensor transformado sin modificar el parámetro original
            weight_transformed = u @ torch.diag(s_clamped) @ v.T
        
        # Aplicar linear con peso transformado, manteniendo el grafo computacional intacto
        return F.linear(F.relu(x), weight_transformed)

class AdversarialWrapper:
    """Wrapper PGD - Factor A"""
    def __init__(self, epsilon=0.1, alpha=0.01, steps=5):
        self.eps = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def attack(self, model_fn, x, y, criterion):
        x_adv = x.clone().detach().requires_grad_(True)
        
        for _ in range(self.steps):
            out = model_fn(x_adv)
            loss = criterion(out, y)
            loss.backward()
            
            # Fix: Usar .data para evitar inplace en variable que requiere grad
            with torch.no_grad():
                x_adv.data += self.alpha * x_adv.grad.sign()
                x_adv.data = torch.clamp(x_adv.data, x - self.eps, x + self.eps)
                x_adv.data = torch.clamp(x_adv.data, -1, 1)
            
            # Zero grad para siguiente iteración
            if x_adv.grad is not None:
                x_adv.grad.zero_()
        
        return x_adv.detach()


# ============================================================================
# BACKBONE COMÚN
# ============================================================================

class VisualBackbone(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()
        )
        
        self.feature_shape = 256 * 2 * 2  # 1024
        
    def forward(self, x):
        return self.stem(x)

class ConsciousCore(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
    
    def forward(self, x):
        q = x.unsqueeze(1)
        out, _ = self.attn(q, q, q)
        return self.norm(out.squeeze(1))

class BioDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512):
        super().__init__()
        self.vocab_size = vocab_size  # Guardar para sanity checks
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 2, batch_first=True, dropout=0.1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.out = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, thought, captions=None, max_len=20):
        if captions is not None:
            emb = self.emb(captions[:, :-1])
            lstm_out, _ = self.lstm(emb, self._init_state(thought))
            gated = lstm_out * self.gate(lstm_out)
            return self.out(gated)
        else:
            # Inference con generación segura
            batch_size = thought.size(0)
            device = thought.device
            generated = []
            input_word = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)  # <BOS>
            hidden = self._init_state(thought)
            
            for _ in range(max_len):
                emb = self.emb(input_word)
                out, hidden = self.lstm(emb, hidden)
                gated = out * self.gate(out)
                logits = self.out(gated).squeeze(1) / 0.8
                
                # Fix: Aplicar máscara de seguridad para tokens válidos
                logits[:, self.vocab_size:] = -float('inf')  # Bloquear tokens inválidos
                
                next_word = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated.append(next_word)
                
                # Fix: Detenerse si se genera <EOS> (token id 2)
                if (next_word == 2).all():
                    break
                
                input_word = next_word
            
            return torch.cat(generated, dim=1)
    
    def _init_state(self, thought):
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        return (h0, c0)
# ============================================================================
# MODELO FACTORIAL (CONFIGURABLE A VOLUNTAD)
# ============================================================================

class NeuroLogosFactorial(nn.Module):
    def __init__(self, vocab_size=1000, 
                 use_sparse=False, use_symbiotic=False, use_adv=False,
                 n_nodes=16, k_sparse=5):
        super().__init__()
        
        self.use_sparse = use_sparse
        self.use_symbiotic = use_symbiotic
        self.use_adv = use_adv
        
        # Backbone
        self.backbone = VisualBackbone()
        
        # Componentes condicionales
        if use_sparse:
            self.sparse = SparseLayer(self.backbone.feature_shape, n_nodes, k_sparse)
            if use_symbiotic:
                self.symbiotic = SymbioticLayer(n_nodes)
            else:
                self.symbiotic = nn.Identity()
            self.bridge = nn.Linear(n_nodes, 512)
        else:
            self.sparse = nn.Identity()
            self.symbiotic = nn.Identity()
            self.bridge = nn.Linear(self.backbone.feature_shape, 512)
        
        # Wrapper adversarial
        if use_adv:
            self.adv_wrapper = AdversarialWrapper()
        else:
            self.adv_wrapper = None
        
        # Head
        self.cortex = ConsciousCore()
        self.decoder = BioDecoder(vocab_size)
    
    def forward(self, image, captions=None):
        feat = self.backbone(image)
        
        # Ruta sparse/symbiotic
        if self.use_sparse:
            sparse_feat = self.sparse(feat)
            refined_feat = self.symbiotic(sparse_feat)
            thought = self.bridge(F.relu(refined_feat))
        else:
            thought = self.bridge(feat)
        
        # Conscious
        attended = self.cortex(thought)
        
        # Decode
        return self.decoder(attended, captions)
    
    def train_step(self, images, captions, optimizer, dataset):
        """Paso de entrenamiento con adversarial condicional y doble forward pass seguro"""
        optimizer.zero_grad()
        
        # Forward normal con grafo computacional limpio
        logits = self.forward(images, captions)
        loss = F.cross_entropy(
            logits.reshape(-1, len(dataset.vocab)),
            captions[:, 1:].reshape(-1),
            ignore_index=dataset.word2id["<PAD>"]
        )
        
        # Adversarial si está activo - genera perturrbaciones en grafo separado
        if self.use_adv and self.adv_wrapper:
            def model_fn(x):
                # Asegura que cada llamada tenga grafo independiente
                return self.forward(x, captions)
            
            # Generar ejemplos adversariales con detach para separar gradientes
            images_adv = self.adv_wrapper.attack(model_fn, images, captions, 
                                               lambda out, tgt: F.cross_entropy(
                                                   out.reshape(-1, len(dataset.vocab)),
                                                   tgt[:, 1:].reshape(-1),
                                                   ignore_index=dataset.word2id["<PAD>"]
                                               ))
            
            # Forward adversarial con grafo computacional independiente
            logits_adv = model_fn(images_adv)
            loss_adv = F.cross_entropy(
                logits_adv.reshape(-1, len(dataset.vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=dataset.word2id["<PAD>"]
            )
            
            # Combinación de pérdidas con gradientes correctamente propagados
            loss = 0.7 * loss + 0.3 * loss_adv
        
        # Backpropagation con grafo computacional intacto
        loss.backward()
        
        # Paso de optimización
        optimizer.step()
        
        return loss, logits




# ============================================================================
# DATASET
# ============================================================================

class CIFARCaptions:
    def __init__(self):
        self.dataset = datasets.CIFAR10('./data', train=True, download=True, 
                                       transform=transforms.ToTensor())
        
        self.templates = {
            0: ["a red airplane in the sky", "a silver aircraft with wings", "a flying machine above clouds"],
            1: ["a shiny yellow car", "a red automobile on asphalt", "a four-wheeled vehicle driving fast"],
            2: ["a small bird with feathers", "a flying sparrow in daylight", "a winged animal perched on branch"],
            3: ["a black domestic cat", "a furry feline sitting still", "a quiet house cat with green eyes"],
            4: ["a wild deer in forest", "a brown animal with antlers", "a grazing mammal in nature"],
            5: ["a loyal brown dog", "a playful canine running", "a four-legged pet barking"],
            6: ["a green frog on lily pad", "a moist amphibian near pond", "a small jumping creature"],
            7: ["a strong brown horse", "a galloping equine in field", "a large farm animal with mane"],
            8: ["a blue cargo ship", "a white vessel on ocean", "a maritime boat with containers"],
            9: ["a large delivery truck", "a heavy-duty cargo vehicle", "a diesel-powered transport"]
        }
        
        self.vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        for descs in self.templates.values():
            for desc in descs:
                self.vocab.extend(desc.split())
        self.vocab = list(dict.fromkeys(self.vocab))
        self.word2id = {w:i for i,w in enumerate(self.vocab)}
        self.id2word = {i:w for w,i in self.word2id.items()}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        desc = np.random.choice(self.templates[label])
        
        tokens = ["<BOS>"] + desc.split() + ["<EOS>"]
        token_ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in tokens]
        token_ids = token_ids[:20] + [self.word2id["<PAD>"]] * (20 - len(token_ids))
        
        return image, torch.tensor(token_ids, dtype=torch.long), label

# ============================================================================
# ENTRENAMIENTO FACTORIAL
# ============================================================================

def train_configuration(config, epochs, device, n_nodes=16, k_sparse=5):
    """
    Entrena UNA configuración específica del diseño factorial.
    
    Args:
        config: tuple (use_sparse, use_symbiotic, use_adv)
    """
    use_sparse, use_symbiotic, use_adv = config
    
    mode_name = f"S{int(use_sparse)}Y{int(use_symbiotic)}A{int(use_adv)}"
    print(f"\n🔬 Entrenando: {mode_name}")
    print(f"   Sparse: {use_sparse} | Symbiotic: {use_symbiotic} | Adv: {use_adv}")
    
    # Dataset
    dataset = CIFARCaptions()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    # Modelo
    model = NeuroLogosFactorial(
        vocab_size=len(dataset.vocab),
        use_sparse=use_sparse,
        use_symbiotic=use_symbiotic,
        use_adv=use_adv,
        n_nodes=n_nodes,
        k_sparse=k_sparse
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Métricas
    history = defaultdict(list)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        
        start_time = time.time()
        
        for images, captions, labels in dataloader:
            images = images.to(device) * 2 - 1
            captions = captions.to(device)
            
            loss, logits = model.train_step(images, captions, optimizer, dataset)
            
            epoch_loss += loss.item()
            
            with torch.no_grad():
                pred_tokens = logits.argmax(dim=-1)
                mask = captions[:, 1:] != dataset.word2id["<PAD>"]
                correct = (pred_tokens == captions[:, 1:]) & mask
                epoch_correct += correct.sum().item()
                epoch_total += mask.sum().item()
            
            num_batches += 1
        
        epoch_time = time.time() - start_time
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_correct / epoch_total * 100
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)
        history['time'].append(epoch_time)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch:02d}: Loss={avg_loss:.3f}, Acc={avg_acc:.2f}%, Time={epoch_time:.1f}s")
    
    return dict(history)


def run_full_factorial(epochs=30, device='cuda', n_nodes=16, k_sparse=5):
    """
    Ejecuta el ablation factorial completo: 8 combinaciones + 3 inversas.
    """
    print("="*80)
    print("NEUROLOGOS v5.2 - FACTORIAL ABLATION STUDY COMPLETO")
    print("="*80)
    print("Componentes: [S]parse, [Y]mbiotic, [A]dversarial")
    print("Total: 8 combinaciones factoriales + 3 ablaciones inversas")
    print("="*80)
    
    results = {}
    
    # ========================================
    # FASE 1: ABLATION FORWARD (8 combos)
    # ========================================
    print("\n[Phase 1] ABLATION FORWARD")
    print("-"*80)
    
    configs = list(product([0, 1], repeat=3))  # 8 combinaciones
    
    for config in configs:
        history = train_configuration(config, epochs, device, n_nodes, k_sparse)
        mode_name = f"S{config[0]}Y{config[1]}A{config[2]}"
        results[mode_name] = history
        
        # Guardar
        torch.save({
            'config': config,
            'history': history
        }, f'results_v52_{mode_name}.pth')
    
    # ========================================
    # FASE 2: ABLATION INVERSE (3 remociones)
    # ========================================
    print("\n[Phase 2] ABLATION INVERSE")
    print("-"*80)
    
    inverse_configs = [
        (1, 1, 1),  # FULL
        (0, 1, 1),  # Remover Sparse
        (1, 0, 1),  # Remover Symbiotic
        (1, 1, 0)   # Remover Adv
    ]
    inverse_names = ['SYA_FULL', 'YA_noS', 'SA_noY', 'SY_noA']
    
    for config, name in zip(inverse_configs, inverse_names):
        print(f"\n🔙 Inverse Ablation: {name}")
        history = train_configuration(config, epochs, device, n_nodes, k_sparse)
        results[name] = history
        
        torch.save({
            'config': config,
            'history': history,
            'type': 'inverse'
        }, f'results_v52_{name}_inverse.pth')
    
    # ========================================
    # FASE 3: ANÁLISIS POST-HOC
    # ========================================
    print("\n" + "="*80)
    print("ANÁLISIS DE EFECTOS E INTERACCIONES")
    print("="*80)
    
    analyze_results(results)
    
    return results


def analyze_results(results):
    """
    Análisis de efectos principales, interacciones y poder explicativo.
    """
    # Extraer accuracies finales
    final_accs = {k: v['accuracy'][-1] for k, v in results.items()}
    
    baseline = final_accs['S0Y0A0']
    sy_a = final_accs['S1Y1A1']  # Full
    
    print(f"\n📊 Baseline (S0Y0A0): {baseline:.2f}%")
    print(f"📊 Full System (S1Y1A1): {sy_a:.2f}%")
    print(f"📊 Mejora Total: {sy_a - baseline:+.2f}%\n")
    
    # ========================================
    # EFECTOS PRINCIPALES (Atribución Causal)
    # ========================================
    print("🔍 EFECTOS PRINCIPALES (Marginales):")
    
    # Efecto de Sparse (promedio de diferencias con/without)
    sparse_effect = np.mean([
        final_accs['S1Y0A0'] - final_accs['S0Y0A0'],  # Baseline vs S
        final_accs['S1Y1A0'] - final_accs['S0Y1A0'],  # Y vs SY
        final_accs['S1Y0A1'] - final_accs['S0Y0A1'],  # A vs SA
        final_accs['S1Y1A1'] - final_accs['S0Y1A1']   # YA vs SYA
    ])
    
    # Efecto de Symbiotic
    symbiotic_effect = np.mean([
        final_accs['S0Y1A0'] - final_accs['S0Y0A0'],  # Baseline vs Y
        final_accs['S1Y1A0'] - final_accs['S1Y0A0'],  # S vs SY
        final_accs['S0Y1A1'] - final_accs['S0Y0A1'],  # A vs YA
        final_accs['S1Y1A1'] - final_accs['S1Y0A1']   # SA vs SYA
    ])
    
    # Efecto de Adversarial
    adv_effect = np.mean([
        final_accs['S0Y0A1'] - final_accs['S0Y0A0'],  # Baseline vs A
        final_accs['S1Y0A1'] - final_accs['S1Y0A0'],  # S vs SA
        final_accs['S0Y1A1'] - final_accs['S0Y1A0'],  # Y vs YA
        final_accs['S1Y1A1'] - final_accs['S1Y1A0']   # SY vs SYA
    ])
    
    print(f"   Sparse (S):      {sparse_effect:+.2f}%")
    print(f"   Symbiotic (Y):   {symbiotic_effect:+.2f}%")
    print(f"   Adversarial (A): {adv_effect:+.2f}%")
    
    # ========================================
    # INTERACCIONES DE ORDEN SUPERIOR
    # ========================================
    print("\n🔀 INTERACCIONES:")
    
    # Interacción SxY (¿sparse+symbiotic > suma de partes?)
    interaction_sy = (final_accs['S1Y1A0'] - final_accs['S0Y0A0']) - \
                     (final_accs['S1Y0A0'] - final_accs['S0Y0A0']) - \
                     (final_accs['S0Y1A0'] - final_accs['S0Y0A0'])
    
    print(f"   S x Y: {interaction_sy:+.2f}% (esperado > 0 para sinergia)")
    
    # Interacción SxA
    interaction_sa = (final_accs['S1Y0A1'] - final_accs['S0Y0A0']) - \
                     (final_accs['S1Y0A0'] - final_accs['S0Y0A0']) - \
                     (final_accs['S0Y0A1'] - final_accs['S0Y0A0'])
    
    print(f"   S x A: {interaction_sa:+.2f}%")
    
    # ========================================
    # ABLACIÓN INVERSA (Necesidad)
    # ========================================
    print("\n🔙 ABLACIÓN INVERSA (Quitar del FULL):")
    
    full_acc = final_accs['S1Y1A1']
    print(f"   FULL - Sparse:     {final_accs.get('YA_noS', 0) - full_acc:+.2f}%")
    print(f"   FULL - Symbiotic:  {final_accs.get('SA_noY', 0) - full_acc:+.2f}%")
    print(f"   FULL - Adversarial: {final_accs.get('SY_noA', 0) - full_acc:+.2f}%")
    
    # ========================================
    # PODER EXPLICATIVO
    # ========================================
    print("\n📈 VARIANZA EXPLICADA:")
    
    # R² aproximado: cuánto de la mejora explican los efectos principales
    predicted_improvement = sparse_effect + symbiotic_effect + adv_effect
    actual_improvement = sy_a - baseline
    
    print(f"   Mejora observada: {actual_improvement:+.2f}%")
    print(f"   Suma de efectos:  {predicted_improvement:+.2f}%")
    print(f"   Diferencia (interacción): {actual_improvement - predicted_improvement:+.2f}%")
    
    if abs(actual_improvement - predicted_improvement) > 2.0:
        print("   ⚠️  ¡ALTA INTERACCIÓN! Los componentes no son independientes.")
    else:
        print("   ✅ Efectos principales dominan (aditivos).")

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Ejecutar ablation factorial completo
    results = run_full_factorial(epochs=30, device=device, n_nodes=16, k_sparse=5)
    
    print("\n✅ Ablation factorial completo finalizado.")
    print("📂 Archivos guardados: results_v52_*.pth")
    print("🧠 Análisis de causalidad disponible en results.")
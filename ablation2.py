# ============================================================================
# NEUROLOGOS v5.1 - SparseSymbiotic Edition
# Objetivo: Validar si symbiotic + sparsity (sin grid) es el driver real
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from collections import defaultdict

# ============================================================================
# SPARSE COMPETITIVE LAYER (Reemplaza Grid Topology)
# ============================================================================

class SparseCompetitiveLayer(nn.Module):
    """
    k-WTA con aprendizaje de importancia de nodos.
    Cada nodo tiene un bias de vida útil que incrementa con activación.
    Los menos usados son pruning dinámico.
    """
    def __init__(self, n_nodes=16, k_sparse=5, input_dim=512):
        super().__init__()
        self.n_nodes = n_nodes
        self.k_sparse = k_sparse
        self.input_dim = input_dim
        
        # Proyección lineal a nodos
        self.fc = nn.Linear(input_dim, n_nodes)
        
        # Bias de importancia (vida útil)
        self.register_buffer('node_age', torch.zeros(n_nodes))
        
        # Máscara de enfermedad (aprende a desactivar nodos inútiles)
        self.health_gate = nn.Parameter(torch.ones(n_nodes))
        
        # Métricas
        self.last_density = 0.0
        self.last_active_nodes = None
        
    def forward(self, x):
        # Activaciones crudas
        activations = F.relu(self.fc(x))  # [B, n_nodes]
        
        # Aplicar health gate
        activations = activations * torch.sigmoid(self.health_gate)
        
        # k-WTA: seleccionar top-k activaciones por batch
        if self.training:
            # Durante entrenamiento: k fijo pero con ruido de exploración
            k = self.k_sparse
            # Añadir ruido Gumbel para explorar (temperatura decreciente)
            noise = torch.randn_like(activations) * 0.1
            activations_noisy = activations + noise
        else:
            # Inferencia: k exacto
            k = self.k_sparse
        
        # Top-k selección
        top_k_values, top_k_indices = torch.topk(activations_noisy, k, dim=1)
        
        # Crear máscara sparse
        sparse_activations = torch.zeros_like(activations)
        sparse_activations.scatter_(1, top_k_indices, top_k_values)
        
        # Actualizar métricas y edad de nodos
        with torch.no_grad():
            self.last_density = (sparse_activations > 0).float().mean().item()
            # Incrementar edad de nodos activos
            active_mask = (sparse_activations > 0).float().sum(0) > 0
            self.node_age += active_mask.float()
            self.last_active_nodes = active_mask.cpu().numpy()
        
        return sparse_activations
    
    def get_metrics(self):
        return {
            'density': self.last_density,
            'avg_node_age': self.node_age.mean().item(),
            'dead_nodes': (self.node_age == 0).sum().item()
        }


# ============================================================================
# SYMBIOTIC REFINEMENT (Mantener pero mejorar)
# ============================================================================

class SymbioticRefiner(nn.Module):
    """
    Refinamiento ortogonal con normalización de estabilidad.
    Versión mejorada con spectral clamping para evitar desvanecimiento.
    """
    def __init__(self, n_nodes=16):
        super().__init__()
        self.refiner = nn.Linear(n_nodes, n_nodes, bias=False)
        
        # Inicialización orthogonal con ganancia adaptativa
        nn.init.orthogonal_(self.refiner.weight, gain=1.0)
        
        # Clamp espectral para estabilidad
        self.register_buffer('max_spectral', torch.tensor(1.5))
        
    def forward(self, x):
        # Clamp pesos espectrales durante forward
        with torch.no_grad():
            u, s, v = torch.svd(self.refiner.weight)
            s = torch.clamp(s, max=self.max_spectral)
            self.refiner.weight.copy_(u @ torch.diag(s) @ v.T)
        
        return self.refiner(F.relu(x))  # Aplicar después de ReLU


# ============================================================================
# SPARSE-SYMBIOTIC CORE (Nuevo)
# ============================================================================

class SparseSymbioticCore(nn.Module):
    """
    Reemplaza TopoBrainCore.
    Combina SparseCompetitiveLayer + SymbioticRefiner.
    """
    def __init__(self, input_dim=512, hidden_dim=64, n_nodes=16, k_sparse=5):
        super().__init__()
        
        self.n_nodes = n_nodes
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Sparse competitive layer
        self.sparse_layer = SparseCompetitiveLayer(n_nodes=n_nodes, k_sparse=k_sparse, 
                                                   input_dim=hidden_dim)
        
        # Symbiotic refinement
        self.symbiotic = SymbioticRefiner(n_nodes=n_nodes)
        
        # Decoder al embedding final
        self.decoder = nn.Linear(n_nodes, input_dim)
        
    def forward(self, x):
        # Encodificar
        h = F.relu(self.encoder(x))
        
        # Sparse activation
        sparse_h = self.sparse_layer(h)
        
        # Refinamiento symbiotic
        refined_h = self.symbiotic(sparse_h)
        
        # Decodificar
        output = self.decoder(F.relu(refined_h))
        
        return output
    
    def get_metrics(self):
        base_metrics = self.sparse_layer.get_metrics()
        return base_metrics


# ============================================================================
# VISUAL ENCODERS v5.1
# ============================================================================

class BaselineUnconscious(nn.Module):
    """Mantener para ablation - Encoder sin sparsity"""
    def __init__(self, output_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        self.bridge = nn.Linear(256*4, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        return self.norm(self.bridge(self.stem(x)))


class SparseUnconscious(nn.Module):
    """Encoder con SparseSymbioticCore"""
    def __init__(self, output_dim=512, n_nodes=16, k_sparse=5):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        
        self.sparse_core = SparseSymbioticCore(
            input_dim=256*4,
            hidden_dim=64,
            n_nodes=n_nodes,
            k_sparse=k_sparse
        )
        
    def forward(self, x):
        features = self.stem(x)
        return self.sparse_core(features)
    
    def get_metrics(self):
        return self.sparse_core.get_metrics()


# ============================================================================
# CONSCIOUS CORE y DECODER (Sin cambios)
# ============================================================================

class ConsciousCore(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        q = x.unsqueeze(1)
        attended, _ = self.attention(q, q, q)
        return self.norm(attended.squeeze(1))


class BioDecoder(nn.Module):
    """Decoder con gating líquido"""
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
        
    def forward(self, thought, captions=None, max_len=20):
        batch_size = thought.size(0)
        device = thought.device
        
        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            lstm_out, _ = self.lstm(embeddings, self._get_init_state(thought))
            gate = self.liquid_gate(lstm_out)
            lstm_out = lstm_out * gate
            return self.out(lstm_out)
        else:
            generated = []
            input_word = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
            hidden = self._get_init_state(thought)
            
            for _ in range(max_len):
                emb = self.embedding(input_word)
                out, hidden = self.lstm(emb, hidden)
                gate = self.liquid_gate(out)
                out = out * gate
                logits = self.out(out).squeeze(1)
                
                logits = logits / 0.8
                probs = F.softmax(logits, dim=-1)
                next_word = torch.multinomial(probs, num_samples=1)
                
                generated.append(next_word)
                input_word = next_word
            
            return torch.cat(generated, dim=1)
    
    def _get_init_state(self, thought):
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        return (h0, c0)


# ============================================================================
# MODELO COMPLETO v5.1 con 5 BRAZOS ABLATION
# ============================================================================

class NeuroLogos_v51(nn.Module):
    """
    5 configuraciones para ablation desacoplado:
    
    1. BASELINE-v51: Encoder densa sin sparsity
    2. SPARSE-ONLY: Sparse layer SIN symbiotic
    3. SYMBIOTIC-ONLY: Symbiotic SIN sparse (capa densa)
    4. SPARSE-SYMBIOTIC: Ambos sin adversarial
    5. SPARSE-SYMBIOTIC-ADV: Full (mejor versión)
    """
    def __init__(self, vocab_size=1000, mode='baseline', n_nodes=16, k_sparse=5):
        super().__init__()
        self.mode = mode
        self.use_adversarial = False
        
        # Visual encoder según modo
        if mode == 'baseline':
            self.eye = BaselineUnconscious(output_dim=512)
        elif mode == 'sparse-only':
            self.eye = SparseUnconscious(output_dim=512, n_nodes=n_nodes, k_sparse=k_sparse)
            # Forzar symbiotic OFF
            self.eye.sparse_core.symbiotic = nn.Identity()
        elif mode == 'symbiotic-only':
            self.eye = BaselineUnconscious(output_dim=512)  # Densa
            # Añadir symbiotic como paso extra
            self.symbiotic_addon = SymbioticRefiner(n_nodes=64)  # Nodos denses
        elif mode == 'sparse-symbiotic':
            self.eye = SparseUnconscious(output_dim=512, n_nodes=n_nodes, k_sparse=k_sparse)
        elif mode == 'sparse-symbiotic-adv':
            self.eye = SparseUnconscious(output_dim=512, n_nodes=n_nodes, k_sparse=k_sparse)
            self.use_adversarial = True
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Resto de arquitectura
        self.cortex = ConsciousCore(dim=512)
        self.broca = BioDecoder(vocab_size, embed_dim=128, hidden_dim=512)
        
        # Adversarial
        if self.use_adversarial:
            self.pgd = PGDAttack(epsilon=0.1, alpha=0.01, steps=5)
        
    def forward(self, image, captions=None):
        visual = self.eye(image)
        
        if hasattr(self, 'symbiotic_addon'):
            visual = self.symbiotic_addon(visual)
        
        thought = self.cortex(visual)
        return self.broca(thought, captions)
    
    def get_metrics(self):
        if hasattr(self.eye, 'get_metrics'):
            return self.eye.get_metrics()
        return {}


class PGDAttack:
    def __init__(self, epsilon=0.1, alpha=0.01, steps=5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def attack(self, model, x, y, criterion):
        x_adv = x.clone().detach()
        x_adv.requires_grad = True
        
        for _ in range(self.steps):
            outputs = model(x_adv)
            loss = criterion(outputs, y)
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
                x_adv = torch.clamp(x_adv, -1, 1)
            
            x_adv.requires_grad = True
        
        return x_adv.detach()


# ============================================================================
# DATASET y MÉTRICAS MEJORADAS
# ============================================================================

class CIFARCaptions_v51:
    def __init__(self):
        self.dataset = datasets.CIFAR10('./data', train=True, download=True, 
                                       transform=transforms.ToTensor())
        
        # Templates más diversos para reducir memorización
        self.templates = {
            0: ["a red airplane in the sky", "a silver aircraft with wings", "a flying machine above clouds", "a jet with two wings", "a commercial plane flying"],
            1: ["a shiny yellow car", "a red automobile on asphalt", "a four-wheeled vehicle driving fast", "a sedan on the road", "a sports car parked"],
            2: ["a small bird with feathers", "a flying sparrow in daylight", "a winged animal perched on branch", "a tiny brown bird", "a bird with spread wings"],
            3: ["a black domestic cat", "a furry feline sitting still", "a quiet house cat with green eyes", "a kitten on a couch", "a cat licking its paw"],
            4: ["a wild deer in forest", "a brown animal with antlers", "a grazing mammal in nature", "a stag in the woods", "a deer running in meadow"],
            5: ["a loyal brown dog", "a playful canine running", "a four-legged pet barking", "a dog fetching a ball", "a puppy with floppy ears"],
            6: ["a green frog on lily pad", "a moist amphibian near pond", "a small jumping creature", "a frog with bulging eyes", "a toad on a rock"],
            7: ["a strong brown horse", "a galloping equine in field", "a large farm animal with mane", "a horse standing in barn", "a mare with white spots"],
            8: ["a blue cargo ship", "a white vessel on ocean", "a maritime boat with containers", "a ship sailing at sea", "a freighter loaded with goods"],
            9: ["a large delivery truck", "a heavy-duty cargo vehicle", "a diesel-powered transport", "a truck on the highway", "a box truck making deliveries"]
        }
        
        # Construir vocabulario mejor
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


def compute_bleu(pred_ids, target_ids, dataset, max_n=4):
    """
    BLEU score simplificado para evaluar calidad de generación
    """
    from collections import Counter
    
    def ngrams(tokens, n):
        return Counter(zip(*[tokens[i:] for i in range(n)]))
    
    bleu_scores = []
    
    for pred, target in zip(pred_ids, target_ids):
        # Convertir a palabras
        pred_words = [dataset.id2word.get(int(t.item()), "<UNK>") for t in pred]
        target_words = [dataset.id2word.get(int(t.item()), "<UNK>") for t in target]
        
        # Limpiar
        pred_words = [w for w in pred_words if w not in ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]]
        target_words = [w for w in target_words if w not in ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]]
        
        if not pred_words:
            bleu_scores.append(0.0)
            continue
        
        # Calcular n-gram precision
        bleu = 0.0
        for n in range(1, max_n + 1):
            pred_ngrams = ngrams(pred_words, n)
            target_ngrams = ngrams(target_words, n)
            
            if not pred_ngrams:
                continue
            
            overlap = sum(min(pred_ngrams[gram], target_ngrams[gram]) 
                         for gram in pred_ngrams if gram in target_ngrams)
            
            precision = overlap / max(len(pred_ngrams), 1)
            bleu += np.log(precision + 1e-9) if precision > 0 else np.log(1e-9)
        
        bleu = np.exp(bleu / max_n)
        bleu_scores.append(bleu)
    
    return np.mean(bleu_scores)


# ============================================================================
# ENTRENAMIENTO CON 5 BRAZOS ABLATION
# ============================================================================

def train_ablation_v51(mode='baseline', epochs=30, device='cuda', n_nodes=16, k_sparse=5):
    """
    Entrena una configuración específica del ablation study v5.1.
    Ahora con BLEU score y métricas de activación por clase.
    """
    print(f"\n{'='*80}")
    print(f"NeuroLogos v5.1 - Ablation: {mode.upper()}")
    print(f"{'='*80}\n")
    
    # Dataset
    dataset = CIFARCaptions_v51()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, 
                           num_workers=2, pin_memory=True)
    
    # Modelo
    model = NeuroLogos_v51(vocab_size=len(dataset.vocab), mode=mode, 
                          n_nodes=n_nodes, k_sparse=k_sparse).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Scheduler para k-sparse (exploración-explotación)
    k_scheduler = lambda epoch: max(3, k_sparse - epoch // 10)  # Reducir k con tiempo
    
    # Métricas
    history = defaultdict(list)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        epoch_density = 0
        epoch_bleu = 0
        num_batches = 0
        
        start_time = time.time()
        
        # Ajustar k_sparse dinámicamente
        current_k = k_scheduler(epoch)
        if hasattr(model.eye, 'sparse_core'):
            model.eye.sparse_core.sparse_layer.k_sparse = current_k
        
        for images, captions, labels in dataloader:
            images = images.to(device, non_blocking=True) * 2 - 1
            captions = captions.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images, captions)
            
            loss = F.cross_entropy(
                logits.reshape(-1, len(dataset.vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=dataset.word2id["<PAD>"]
            )
            
            # Adversarial training (solo en modo adv)
            if model.use_adversarial and epoch > 3:  # Activar más temprano
                def forward_fn(x):
                    vis = model.eye(x)
                    thought = model.cortex(vis)
                    return model.broca(thought, captions)
                
                images_adv = model.pgd.attack(forward_fn, images, captions, 
                                             lambda out, tgt: F.cross_entropy(
                                                 out.reshape(-1, len(dataset.vocab)),
                                                 tgt[:, 1:].reshape(-1),
                                                 ignore_index=dataset.word2id["<PAD>"]
                                             ))
                
                logits_adv = forward_fn(images_adv)
                loss_adv = F.cross_entropy(
                    logits_adv.reshape(-1, len(dataset.vocab)),
                    captions[:, 1:].reshape(-1),
                    ignore_index=dataset.word2id["<PAD>"]
                )
                
                loss = 0.6 * loss + 0.4 * loss_adv  # Mayor peso a adv
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Métricas
            epoch_loss += loss.item()
            
            with torch.no_grad():
                pred_tokens = logits.argmax(dim=-1)
                mask = captions[:, 1:] != dataset.word2id["<PAD>"]
                correct = (pred_tokens == captions[:, 1:]) & mask
                epoch_correct += correct.sum().item()
                epoch_total += mask.sum().item()
                
                # BLEU score
                bleu = compute_bleu(pred_tokens[:8], captions[:, 1:][:8], dataset)
                epoch_bleu += bleu
            
            # Densidad
            metrics = model.get_metrics()
            if 'density' in metrics:
                epoch_density += metrics['density']
            
            num_batches += 1
        
        epoch_time = time.time() - start_time
        
        # Promedios
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_correct / epoch_total * 100
        avg_density = epoch_density / num_batches if epoch_density > 0 else 1.0
        avg_bleu = epoch_bleu / num_batches
        
        # Guardar historia
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)
        history['density'].append(avg_density)
        history['bleu'].append(avg_bleu)
        history['time'].append(epoch_time)
        
        # Log
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.3f} | "
                  f"Acc: {avg_acc:.2f}% | BLEU: {avg_bleu:.3f} | "
                  f"Density: {avg_density:.3f} | k={current_k} | Time: {epoch_time:.1f}s")
            
            # Generar ejemplo con top-3 predicciones
            if epoch % 10 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    sample_img, _, sample_label = dataset[0]
                    sample_img = sample_img.unsqueeze(0).to(device) * 2 - 1
                    generated = model(sample_img, captions=None)
                    
                    # Top-3 predicciones
                    logits_sample = model(sample_img, torch.zeros(1, 20, dtype=torch.long, device=device))
                    probs = F.softmax(logits_sample[0], dim=-1)
                    top3_token_ids = torch.topk(probs, 3, dim=-1).indices[:5]  # Primeros 5 tokens
                    
                    words = [dataset.id2word.get(int(t.item()), "<UNK>") 
                            for t in generated[0]]
                    sentence = " ".join(w for w in words 
                                      if w not in ["<BOS>", "<EOS>", "<PAD>"])
                    
                    print(f"   GT Label: {list(dataset.templates.keys())[sample_label]}")
                    print(f"   Generated: '{sentence}'")
                    print(f"   Top-3 predictions: {top3_token_ids.cpu().numpy()}\n")
                model.train()
    
    return model, dict(history)


def run_ablation_v51(epochs=30, device='cuda', n_nodes=16, k_sparse=5):
    """
    Ejecuta ablation study v5.1 con 5 brazos desacoplados.
    """
    print("="*80)
    print("NeuroLogos v5.1 - ABLATION STUDY SPARSE-SYMBIOTIC")
    print("="*80)
    print("Brazos experimentales (5 niveles):")
    print("  1. BASELINE: Capa densa estándar")
    print("  2. SPARSE-ONLY: k-WTA sin refinamiento")
    print("  3. SYMBIOTIC-ONLY: Refinamiento ortogonal sin sparse")
    print("  4. SPARSE-SYMBIOTIC: Ambos sin adversarial")
    print("  5. SPARSE-SYMBIOTIC-ADV: Full system")
    print("="*80)
    
    results = {}
    
    modes = ['baseline', 'sparse-only', 'symbiotic-only', 'sparse-symbiotic', 'sparse-symbiotic-adv']
    
    for mode in modes:
        model, history = train_ablation_v51(mode, epochs, device, n_nodes, k_sparse)
        results[mode] = history
        
        # Guardar modelo
        torch.save({
            'model_state': model.state_dict(),
            'metrics': history
        }, f'neurologos_v51_{mode.replace("-", "_")}.pth')
        print(f"\n✅ Modelo guardado: neurologos_v51_{mode}.pth\n")
    
    # Resumen comparativo
    print("\n" + "="*80)
    print("RESULTADOS FINALES - ABLATION v5.1")
    print("="*80)
    
    for mode in modes:
        hist = results[mode]
        final_loss = hist['loss'][-1]
        final_acc = hist['accuracy'][-1]
        final_bleu = hist['bleu'][-1]
        final_density = hist['density'][-1]
        avg_time = np.mean(hist['time'])
        
        print(f"\n{mode.upper()}:")
        print(f"  Final Loss:     {final_loss:.3f}")
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Final BLEU:     {final_bleu:.3f}")
        print(f"  Final Density:  {final_density:.3f}")
        print(f"  Avg Time/Epoch: {avg_time:.1f}s")
    
    print("\n" + "="*80)
    
    return results


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Parámetros de búsqueda
    NODES = 16
    K_SPARSE = 5
    
    # Ejecutar ablation completo
    results = run_ablation_v51(epochs=30, device=device, n_nodes=NODES, k_sparse=K_SPARSE)
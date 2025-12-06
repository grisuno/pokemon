"""
================================================================================
NeuroLogos v5.0 - TopoBrain Edition
================================================================================
Arquitectura híbrida Vision-Language con topología neuronal validada
Diseñado para Google Colab T4 GPU

ABLATION STUDY (3 niveles):
1. BASELINE: MiniUnconscious + LSTM simple
2. TOPO-LIGHT: TopoBrain sin adversarial
3. TOPO-FULL: TopoBrain completo (Grid + Symbiotic + Adversarial)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import time
from collections import defaultdict

# ============================================================================
# TOPOBRAIN - ARQUITECTURA VALIDADA (92% accuracy)
# ============================================================================

class TopoBrainCore(nn.Module):
    """TopoBrain validado con ablation (de tu experimento anterior)"""
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=512, 
                 grid_size=4, use_grid=True, use_symbiotic=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.n_nodes = grid_size * grid_size
        
        self.use_grid = use_grid
        self.use_symbiotic = use_symbiotic
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.n_nodes)
        
        # Grid Topology (Plasticity)
        if self.use_grid:
            self.register_buffer('grid_coords', self._init_grid())
            self.plasticity = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes))
        
        # Symbiotic Basis (refinamiento ortogonal)
        if self.use_symbiotic:
            self.symbiotic = nn.Linear(self.n_nodes, self.n_nodes, bias=False)
            nn.init.orthogonal_(self.symbiotic.weight)
        
        # Decoder
        self.fc3 = nn.Linear(self.n_nodes, output_dim)
        
        # Metrics
        self.last_density = 0.0
        
    def _init_grid(self):
        """Inicializa coordenadas del grid 2D"""
        coords = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                coords.append([i / self.grid_size, j / self.grid_size])
        return torch.tensor(coords, dtype=torch.float32)
    
    def forward(self, x):
        # Encoder
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))  # [B, n_nodes]
        
        # Grid Topology + Plasticity
        if self.use_grid:
            distances = torch.cdist(self.grid_coords, self.grid_coords)
            topology = torch.exp(-distances * 2.0)
            plastic_weights = topology * torch.sigmoid(self.plasticity)
            h2_plastic = h2 @ plastic_weights
            
            # Density metric
            with torch.no_grad():
                active_nodes = (h2.abs() > 0.1).float().mean()
                self.last_density = active_nodes.item()
        else:
            h2_plastic = h2
            self.last_density = 1.0
        
        # Symbiotic Basis (refinamiento)
        if self.use_symbiotic:
            h2_refined = self.symbiotic(h2_plastic)
        else:
            h2_refined = h2_plastic
        
        # Decoder
        output = self.fc3(F.relu(h2_refined))
        
        return output
    
    def get_metrics(self):
        """Retorna métricas de topología"""
        metrics = {
            'density': self.last_density
        }
        
        if self.use_grid:
            metrics['plasticity_norm'] = self.plasticity.norm().item()
        
        return metrics


class PGDAttack:
    """Adversarial attack para robustez (solo en TOPO-FULL)"""
    def __init__(self, epsilon=0.1, alpha=0.01, steps=5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def attack(self, model, x, y, criterion):
        """Genera ejemplos adversariales"""
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
# VISUAL ENCODERS
# ============================================================================

class MiniUnconscious(nn.Module):
    """Baseline: Encoder visual simple"""
    def __init__(self, output_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        self.topo_bridge = nn.Linear(256*4, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        features = self.stem(x)
        encoded = self.topo_bridge(features)
        return self.norm(encoded)


class TopoUnconscious(nn.Module):
    """TopoBrain-enhanced visual encoder"""
    def __init__(self, output_dim=512, use_grid=True, use_symbiotic=True):
        super().__init__()
        
        # Visual stem (compartido)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        
        # TopoBrain core
        self.topobrain = TopoBrainCore(
            input_dim=256*4,
            hidden_dim=64,
            output_dim=output_dim,
            grid_size=4,
            use_grid=use_grid,
            use_symbiotic=use_symbiotic
        )
        
    def forward(self, x):
        features = self.stem(x)
        return self.topobrain(features)
    
    def get_metrics(self):
        return self.topobrain.get_metrics()


# ============================================================================
# CONSCIOUS CORE (simplificado para compatibilidad)
# ============================================================================

class ConsciousCore(nn.Module):
    """Núcleo consciente con atención"""
    def __init__(self, dim=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x: [B, D]
        q = x.unsqueeze(1)
        attended, _ = self.attention(q, q, q)
        return self.norm(attended.squeeze(1))


# ============================================================================
# DECODER (ÁREA DE BROCA)
# ============================================================================

class BioDecoder(nn.Module):
    """Decoder LSTM con gating"""
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        
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
            # Training mode
            embeddings = self.embedding(captions[:, :-1])
            lstm_out, _ = self.lstm(embeddings, self._get_init_state(thought))
            gate = self.liquid_gate(lstm_out)
            lstm_out = lstm_out * gate
            return self.out(lstm_out)
        else:
            # Inference mode
            generated = []
            input_word = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
            hidden = self._get_init_state(thought)
            
            for _ in range(max_len):
                emb = self.embedding(input_word)
                out, hidden = self.lstm(emb, hidden)
                gate = self.liquid_gate(out)
                out = out * gate
                logits = self.out(out).squeeze(1)
                
                # Sampling con temperatura
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
# MODELO COMPLETO - 3 CONFIGURACIONES
# ============================================================================

class NeuroLogos(nn.Module):
    """
    Configuraciones del ablation:
    - mode='baseline': MiniUnconscious (sin TopoBrain)
    - mode='topo-light': TopoBrain sin symbiotic
    - mode='topo-full': TopoBrain completo + adversarial
    """
    def __init__(self, vocab_size=1000, mode='baseline'):
        super().__init__()
        self.mode = mode
        
        # Visual encoder según modo
        if mode == 'baseline':
            self.eye = MiniUnconscious(output_dim=512)
            self.use_adversarial = False
        elif mode == 'topo-light':
            self.eye = TopoUnconscious(output_dim=512, use_grid=True, use_symbiotic=False)
            self.use_adversarial = False
        elif mode == 'topo-full':
            self.eye = TopoUnconscious(output_dim=512, use_grid=True, use_symbiotic=True)
            self.use_adversarial = True
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Conscious core y decoder (compartidos)
        self.cortex = ConsciousCore(dim=512)
        self.broca = BioDecoder(vocab_size, embed_dim=128, hidden_dim=512)
        
        # Adversarial (solo TOPO-FULL)
        if self.use_adversarial:
            self.pgd = PGDAttack(epsilon=0.1, alpha=0.01, steps=5)
        
    def forward(self, image, captions=None):
        visual = self.eye(image)
        thought = self.cortex(visual)
        return self.broca(thought, captions)
    
    def get_metrics(self):
        """Obtiene métricas de topología si disponible"""
        if hasattr(self.eye, 'get_metrics'):
            return self.eye.get_metrics()
        return {}


# ============================================================================
# DATASET CIFAR10 CON CAPTIONS
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
# ENTRENAMIENTO CON ABLATION STUDY
# ============================================================================

def train_ablation(mode='baseline', epochs=30, device='cuda'):
    """
    Entrena un modelo en el modo especificado
    """
    print(f"\n{'='*80}")
    print(f"NeuroLogos v5.0 - Ablation Study: {mode.upper()}")
    print(f"{'='*80}\n")
    
    # Dataset
    dataset = CIFARCaptions()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    # Modelo
    model = NeuroLogos(vocab_size=len(dataset.vocab), mode=mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Métricas
    history = {
        'loss': [],
        'accuracy': [],
        'density': [],
        'time': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        epoch_density = 0
        num_batches = 0
        
        start_time = time.time()
        
        for images, captions, labels in dataloader:
            images = images.to(device, non_blocking=True) * 2 - 1
            captions = captions.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass normal
            logits = model(images, captions)
            
            loss = F.cross_entropy(
                logits.reshape(-1, len(dataset.vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=dataset.word2id["<PAD>"]
            )
            
            # Adversarial training (solo TOPO-FULL)
            if model.use_adversarial and epoch > 5:
                # Crear ejemplos adversariales
                images_adv = model.pgd.attack(
                    lambda x: model(x, captions),
                    images, captions, 
                    lambda out, tgt: F.cross_entropy(
                        out.reshape(-1, len(dataset.vocab)),
                        tgt[:, 1:].reshape(-1),
                        ignore_index=dataset.word2id["<PAD>"]
                    )
                )
                
                # Forward con adversariales
                logits_adv = model(images_adv, captions)
                loss_adv = F.cross_entropy(
                    logits_adv.reshape(-1, len(dataset.vocab)),
                    captions[:, 1:].reshape(-1),
                    ignore_index=dataset.word2id["<PAD>"]
                )
                
                # Loss combinado
                loss = 0.7 * loss + 0.3 * loss_adv
            
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
            
            # Densidad (si TopoBrain)
            metrics = model.get_metrics()
            if 'density' in metrics:
                epoch_density += metrics['density']
            
            num_batches += 1
        
        epoch_time = time.time() - start_time
        
        # Promedios
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_correct / epoch_total * 100
        avg_density = epoch_density / num_batches if epoch_density > 0 else 1.0
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)
        history['density'].append(avg_density)
        history['time'].append(epoch_time)
        
        # Log cada 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.3f} | "
                  f"Acc: {avg_acc:.2f}% | Density: {avg_density:.3f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Generar ejemplo
            model.eval()
            with torch.no_grad():
                sample_img, _, sample_label = dataset[0]
                sample_img = sample_img.unsqueeze(0).to(device) * 2 - 1
                generated = model(sample_img, captions=None)
                words = [dataset.id2word.get(int(t.item()), "<UNK>") 
                        for t in generated[0]]
                sentence = " ".join(w for w in words 
                                  if w not in ["<BOS>", "<EOS>", "<PAD>"])
                print(f"   Generated: '{sentence}'")
                print(f"   GT Label: {list(dataset.templates.keys())[sample_label]}\n")
            model.train()
    
    return model, history


def run_full_ablation(epochs=30, device='cuda'):
    """
    Ejecuta ablation study completo de 3 niveles
    """
    print("="*80)
    print("NeuroLogos v5.0 - ABLATION STUDY COMPLETO")
    print("="*80)
    print("Comparando 3 arquitecturas:")
    print("  1. BASELINE: MiniUnconscious (sin TopoBrain)")
    print("  2. TOPO-LIGHT: TopoBrain sin symbiotic")
    print("  3. TOPO-FULL: TopoBrain completo + adversarial")
    print("="*80)
    
    results = {}
    
    for mode in ['baseline', 'topo-light', 'topo-full']:
        model, history = train_ablation(mode, epochs, device)
        results[mode] = history
        
        # Guardar modelo
        torch.save(model.state_dict(), f'neurologos_{mode}.pth')
        print(f"\n✅ Modelo guardado: neurologos_{mode}.pth\n")
    
    # Resumen comparativo
    print("\n" + "="*80)
    print("RESULTADOS FINALES - ABLATION STUDY")
    print("="*80)
    
    for mode in ['baseline', 'topo-light', 'topo-full']:
        hist = results[mode]
        final_loss = hist['loss'][-1]
        final_acc = hist['accuracy'][-1]
        avg_time = np.mean(hist['time'])
        final_density = hist['density'][-1]
        
        print(f"\n{mode.upper()}:")
        print(f"  Final Loss:     {final_loss:.3f}")
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Avg Time/Epoch: {avg_time:.1f}s")
        print(f"  Final Density:  {final_density:.3f}")
    
    print("\n" + "="*80)
    
    return results


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Ejecutar ablation completo
    results = run_full_ablation(epochs=30, device=device)
    
    # Opcional: Entrenar solo un modo específico
    # model, history = train_ablation('topo-full', epochs=50, device=device)
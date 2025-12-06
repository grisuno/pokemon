# =============================================================================
# NeuroLogos v4.0 - PoC CPU - 512 líneas
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from collections import deque

# =============================================================================
# 1. VISUAL ENCODER MINIATURA (Ojo de 3.2M params)
# =============================================================================
# REEMPLAZA NestedUnconscious CON ESTO (si GPU disponible):
class NestedUnconscious(nn.Module):
    def __init__(self, grid_size=4, hidden_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),  # 16x16
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),  # 8x8
        )
        
        # Topología anidada simplificada (4 clusters x 4 clusters)
        self.num_clusters = 4
        self.nodes_per_cluster = 4
        self.nodes = self.num_clusters * self.nodes_per_cluster
        
        self.mapper = nn.Linear(128, hidden_dim)
        
        # Mensajes intra-cluster (igual que v3.1)
        self.intra_adj = nn.Parameter(torch.randn(self.nodes_per_cluster, self.nodes_per_cluster) * 0.5)
        self.intra_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Mensajes inter-cluster (igual que v3.1)
        self.inter_adj = nn.Parameter(torch.randn(self.num_clusters, self.num_clusters) * 0.5)
        self.inter_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        B = x.size(0)
        h = self.stem(x)  # [B, 128, 8, 8]
        nodes = h.view(B, self.nodes, -1)  # [B, 16, 128]
        nodes = self.mapper(nodes)  # [B, 16, hidden_dim]
        
        # Intra-cluster
        intra_adj = torch.sigmoid(self.intra_adj) * (1 - torch.eye(self.nodes_per_cluster, device=x.device))
        nodes_intra = nodes.view(B, self.num_clusters, self.nodes_per_cluster, -1)
        messages = self.intra_mlp(nodes_intra)
        aggregated = torch.matmul(intra_adj, messages)
        nodes_intra = nodes_intra + aggregated
        
        # Inter-cluster
        inter_adj = torch.sigmoid(self.inter_adj) * (1 - torch.eye(self.num_clusters, device=x.device))
        cluster_repr = nodes_intra.mean(dim=2)  # Pooling
        cluster_messages = self.inter_mlp(cluster_repr)
        cluster_aggregated = torch.matmul(inter_adj, cluster_messages)
        
        # Broadcast
        nodes_final = nodes_intra + cluster_aggregated.unsqueeze(2)
        
        return self.norm(nodes_final.view(B, self.nodes, -1))




# =============================================================================
# 2. CONSCIOUS SYSTEM (Corteza Visual - 1.1M params)
# =============================================================================
class LiquidNeuron(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.slow = nn.Linear(dim, dim)
        self.fast = nn.Parameter(torch.zeros(dim, dim))
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x, plasticity=0.1):
        if self.training and plasticity > 0:
            delta = torch.randn_like(self.fast) * plasticity
            self.fast.data = 0.95 * self.fast.data + 0.05 * delta
        
        out = self.slow(x) + F.linear(x, self.fast)
        return self.ln(torch.tanh(out))

class ConsciousCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_pool = nn.MultiheadAttention(512, 8, batch_first=True)
        self.liquid = LiquidNeuron(512)
        self.thought_compressor = nn.Linear(512, 256)  # Pensamiento compacto
        
    def forward(self, visual_features, plasticity):
        # Self-attention: ¿qué partes de la imagen son importantes?
        pooled, _ = self.attention_pool(
            visual_features.unsqueeze(1), 
            visual_features.unsqueeze(1), 
            visual_features.unsqueeze(1)
        )
        pooled = pooled.squeeze(1)
        
        # Memoria líquida: integración con plasticidad
        thought = self.liquid(pooled, plasticity)
        
        # Comprimir para lenguaje
        return self.thought_compressor(thought)

# =============================================================================
# 3. ÁREA DE BROCA - BioDecoder (1.8M params)
# =============================================================================
class BioDecoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Plasticidad líquida para generación
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
        
    def forward(self, thought, captions=None, max_len=20, teacher_forcing_ratio=0.5):
        """
        Modo entrenamiento: captions != None
        Modo generación: captions == None
        """
        batch_size = thought.size(0)
        device = thought.device
        
        if captions is not None:
            # Teacher Forcing: entrenamos a predecir palabra siguiente
            embeddings = self.embedding(captions[:, :-1])  # Desplazamiento a la derecha
            lstm_out, _ = self.lstm(embeddings, self._get_init_state(thought))
            
            # Plasticidad líquida modula salida
            gate = self.liquid_gate(lstm_out)
            lstm_out = lstm_out * gate
            
            logits = self.out(lstm_out)
            return logits  # [Batch, Seq-1, Vocab]
        
        else:
            # Generación autoregresiva pura
            generated = []
            input_word = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # Token <BOS>
            hidden = self._get_init_state(thought)
            
            for _ in range(max_len):
                emb = self.embedding(input_word)
                out, hidden = self.lstm(emb, hidden)
                
                gate = self.liquid_gate(out)
                out = out * gate
                
                logits = self.out(out)
                next_word = logits.argmax(dim=-1)
                
                generated.append(next_word)
                input_word = next_word
                
            return torch.cat(generated, dim=1)  # [Batch, Max_Len]
    
    def _get_init_state(self, thought):
        # Inicializar LSTM con el pensamiento visual
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)  # 2 capas LSTM
        c0 = torch.zeros_like(h0)
        return (h0, c0)

# =============================================================================
# 4. NEUROLOGOS - Arquitectura Completa
# =============================================================================
class NeuroLogos(nn.Module):
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.eye = NestedUnconscious()
        self.cortex = ConsciousCore()
        self.broca = BioDecoder(vocab_size)
        
        # Homeostasis simple
        self.running_richness = 20.0
        
    def forward(self, image, captions=None, plasticity=0.1):
        # 1. Ver
        visual = self.eye(image)
        
        # 2. Pensar
        thought = self.cortex(visual, plasticity)
        
        # 3. Hablar
        if captions is not None:
            return self.broca(thought, captions)
        else:
            return self.broca(thought, captions=None)
    
    def measure_richness(self):
        # Pseudo-métrica de complejidad del pensamiento
        return np.random.exponential(self.running_richness * 0.01)

# =============================================================================
# 5. CICLO DE VIDA NEURAL SIMPLIFICADO
# =============================================================================
class LifeCycle:
    def __init__(self, total_epochs=50):
        self.total = total_epochs
        self.phase = "GENESIS"
        self.eye = NestedUnconscious(grid_size=4, hidden_dim=128)
        
    def get_plasticity(self, epoch):
        if epoch < 5:
            self.phase = "GENESIS (Growing Eye)"
            return 0.5
        elif epoch < 30:
            self.phase = "AWAKENING (Learning to Think)"
            return 0.2 * (1 - epoch/30)
        else:
            self.phase = "LOGOS (Speaking)"
            return 0.0

# =============================================================================
# 6. DATASET TOY PARA POC (CIFAR-10 + Captions Sintéticas)
# =============================================================================
class CIFARCaptions:
    def __init__(self):
        from torchvision import datasets, transforms
        self.dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        
        # Mapeo simple: clases -> descripciones básicas
        self.templates = {
            0: ["a small red airplane", "a red flying vehicle", "an aircraft in the sky"],
            1: ["a yellow automobile", "a car on the road", "a vehicle driving"],
            2: ["a brown bird perched", "a small bird flying", "a winged creature"],
            3: ["a black feline", "a cat sitting", "a domestic cat"],
            4: ["a brown deer", "a wild deer", "an animal with antlers"],
            5: ["a brown dog", "a dog running", "a canine pet"],
            6: ["a green frog", "a frog on a leaf", "a small amphibian"],
            7: ["a brown horse", "a horse galloping", "an equine animal"],
            8: ["a large ship", "a boat on water", "a maritime vessel"],
            9: ["a brown truck", "a delivery truck", "a heavy vehicle"]
        }
        
        # Tokenizer propio (simple)
        self.vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        for descs in self.templates.values():
            for desc in descs:
                self.vocab.extend(desc.split())
        self.vocab = list(dict.fromkeys(self.vocab))  # Uniq
        self.word2id = {w:i for i,w in enumerate(self.vocab)}
        self.id2word = {i:w for w,i in self.word2id.items()}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Pick random template
        desc = np.random.choice(self.templates[label])
        
        # Tokenize
        tokens = ["<BOS>"] + desc.split() + ["<EOS>"]
        token_ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in tokens]
        
        # Truncate/pad
        token_ids = token_ids[:20] + [self.word2id["<PAD>"]] * (20 - len(token_ids))
        
        return image, torch.tensor(token_ids, dtype=torch.long)

# =============================================================================
# 7. ENTRENAMIENTO ONE-SHOT
# =============================================================================
def train_logos():
    device = torch.device('cpu')
    print(f"NeuroLogos v4.0 PoC | Device: {device}\n")
    
    # Dataset con captions
    dataset = CIFARCaptions()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Modelo
    model = NeuroLogos(vocab_size=len(dataset.vocab)).to(device)
    
    # Optimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Ciclo de vida
    life = LifeCycle(total_epochs=50)
    
    print("🔬 Iniciando Ciclo de Vida Neural...\n")
    
    for epoch in range(50):
        plasticity = life.get_plasticity(epoch)
        model.train()
        
        total_loss = 0
        for images, captions in dataloader:
            # Simular imágenes normalizadas
            images = images * 2 - 1
            
            optimizer.zero_grad()
            logits = model(images, captions, plasticity=plasticity)
            
            # Calcular pérdida (ignorar padding)
            loss = F.cross_entropy(
                logits.reshape(-1, len(dataset.vocab)),
                captions[:, 1:].reshape(-1),  # Target desplazado
                ignore_index=dataset.word2id["<PAD>"]
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluar generación cada 10 épocas
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_img, _ = dataset[0]  # Aeronave
                sample_img = (sample_img.float() / 255.0) * 2 - 1
                generated = model(sample_img.unsqueeze(0), captions=None)
                
                # Convertir tokens a texto
                words = [dataset.id2word.get(int(tok), "<UNK>") for tok in generated[0]]
                sentence = " ".join(w for w in words if w not in ["<BOS>", "<EOS>", "<PAD>"])
                
                print(f"Ep {epoch:02d} | Phase: {life.phase} | Loss: {total_loss/len(dataloader):.3f}")
                print(f"  Plasticity: {plasticity:.2f} | Richness: {model.measure_richness():.3f}")
                print(f"  Genera: '{sentence}'")
        else:
            print(f"Ep {epoch:02d} | Phase: {life.phase} | Loss: {total_loss/len(dataloader):.3f}")

# =============================================================================
# 8. EJECUCIÓN DIRECTA
# =============================================================================
if __name__ == "__main__":
    train_logos()
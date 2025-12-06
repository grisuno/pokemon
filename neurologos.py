# =============================================================================
# NeuroLogos v4.0 - CORREGIDO DIMENSIONALMENTE
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms

# =============================================================================
# 1. VISUAL ENCODER (Elige una: MiniUnconscious o NestedUnconscious)
# =============================================================================
class MiniUnconscious(nn.Module):
    """Versión rápida CPU: 512-dim output directo"""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        self.topo_bridge = nn.Linear(128*8*8, 512)
        
    def forward(self, x):
        return self.topo_bridge(self.stem(x))  # [B, 512]

class NestedUnconscious(nn.Module):
    """Versión topológica GPU: mantiene nested structure"""
    def __init__(self, grid_size=4, output_dim=512):  # output_dim=512 fijo
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
        )
        
        self.num_clusters = 4
        self.nodes_per_cluster = 4
        self.nodes = self.num_clusters * self.nodes_per_cluster
        
        self.mapper = nn.Linear(128, output_dim)  # 128->512
        
        self.intra_adj = nn.Parameter(torch.randn(self.nodes_per_cluster, self.nodes_per_cluster) * 0.5)
        self.intra_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2), nn.GELU(), nn.Linear(output_dim * 2, output_dim)
        )
        
        self.inter_adj = nn.Parameter(torch.randn(self.num_clusters, self.num_clusters) * 0.5)
        self.inter_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2), nn.GELU(), nn.Linear(output_dim * 2, output_dim)
        )
        
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        B = x.size(0)
        h = self.stem(x)  # [B, 128, 8, 8]
        nodes = h.view(B, self.nodes, -1)  # [B, 16, 128]
        nodes = self.mapper(nodes)  # [B, 16, 512]
        
        intra_adj = torch.sigmoid(self.intra_adj) * (1 - torch.eye(self.nodes_per_cluster, device=x.device))
        nodes_intra = nodes.view(B, self.num_clusters, self.nodes_per_cluster, -1)
        messages = self.intra_mlp(nodes_intra)
        aggregated = torch.matmul(intra_adj, messages)
        nodes_intra = nodes_intra + aggregated
        
        inter_adj = torch.sigmoid(self.inter_adj) * (1 - torch.eye(self.num_clusters, device=x.device))
        cluster_repr = nodes_intra.mean(dim=2)
        cluster_messages = self.inter_mlp(cluster_repr)
        cluster_aggregated = torch.matmul(inter_adj, cluster_messages)
        
        nodes_final = nodes_intra + cluster_aggregated.unsqueeze(2)
        
        return self.norm(nodes_final.view(B, self.nodes, -1))  # [B, 16, 512]

# =============================================================================
# 2. CONSCIOUS SYSTEM
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
        self.thought_compressor = nn.Linear(512, 256)  # 512->256
        
    def forward(self, visual_features, plasticity):
        pooled, _ = self.attention_pool(
            visual_features.unsqueeze(1), 
            visual_features.unsqueeze(1), 
            visual_features.unsqueeze(1)
        )
        pooled = pooled.squeeze(1)
        thought = self.liquid(pooled, plasticity)
        return self.thought_compressor(thought)  # [B, 256]

# =============================================================================
# 3. ÁREA DE BROCA - Corregido para dimensión 256
# =============================================================================
class BioDecoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=256):
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
            embeddings = self.embedding(captions[:, :-1])
            lstm_out, _ = self.lstm(embeddings, self._get_init_state(thought))
            gate = self.liquid_gate(lstm_out)
            lstm_out = lstm_out * gate
            return self.out(lstm_out)
        else:
            generated = []
            input_word = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
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
                
            return torch.cat(generated, dim=1)
    
    def _get_init_state(self, thought):
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)  # [2, B, 256]
        c0 = torch.zeros_like(h0)
        return (h0, c0)

# =============================================================================
# 4. NEUROLOGOS - Arquitectura Completa
# =============================================================================
class NeuroLogos(nn.Module):
    def __init__(self, vocab_size=1000, use_nested=False):  # Flag para elegir versión
        super().__init__()
        self.use_nested = use_nested
        
        if use_nested:
            self.eye = NestedUnconscious()  # Con topología
            self.cortex = ConsciousCore()   # Maneja [B, 16, 512]
        else:
            self.eye = MiniUnconscious()    # Sin topología
            self.cortex = ConsciousCore()   # Maneja [B, 512]
        
        self.broca = BioDecoder(vocab_size)
        self.running_richness = 20.0
        
    def forward(self, image, captions=None, plasticity=0.1):
        visual = self.eye(image)
        
        # Manejar ambos formatos: con y sin topología
        if self.use_nested:
            # [B, 16, 512] -> pool para [B, 512]
            visual = visual.mean(dim=1)
        
        thought = self.cortex(visual, plasticity)
        return self.broca(thought, captions)
    
    def measure_richness(self):
        return np.random.exponential(self.running_richness * 0.01)


# =============================================================================
# 5. CICLO DE VIDA NEURAL SIMPLIFICADO
# =============================================================================
class LifeCycle:
    def __init__(self, total_epochs=50):
        self.total = total_epochs
        self.phase = "GENESIS"
        
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
# 5. CICLO DE VIDA DATASET ENTRENAMIENTO
# =============================================================================
class CIFARCaptions:
    def __init__(self):
        self.dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        
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
        
        return image, torch.tensor(token_ids, dtype=torch.long)

# =============================================================================
# 6. ENTRENAMIENTO
# =============================================================================
def train_logos(use_nested=False):
    device = torch.device('cpu')
    print(f"NeuroLogos v4.0 PoC | Device: {device} | Nested: {use_nested}\n")
    
    dataset = CIFARCaptions()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = NeuroLogos(vocab_size=len(dataset.vocab), use_nested=use_nested).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    life = LifeCycle(total_epochs=50)
    
    print("🔬 Iniciando Ciclo de Vida Neural...\n")
    
    for epoch in range(50):
        plasticity = life.get_plasticity(epoch)
        model.train()
        
        total_loss = 0
        for images, captions in dataloader:
            images = images * 2 - 1
            
            optimizer.zero_grad()
            logits = model(images, captions, plasticity=plasticity)
            
            loss = F.cross_entropy(
                logits.reshape(-1, len(dataset.vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=dataset.word2id["<PAD>"]
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_img, _ = dataset[0]
                sample_img = sample_img * 2 - 1
                generated = model(sample_img.unsqueeze(0), captions=None)
                
                words = [dataset.id2word.get(int(tok), "<UNK>") for tok in generated[0]]
                sentence = " ".join(w for w in words if w not in ["<BOS>", "<EOS>", "<PAD>"])
                
                print(f"Ep {epoch:02d} | Phase: {life.phase} | Loss: {total_loss/len(dataloader):.3f}")
                print(f"  Genera: '{sentence}'")
        else:
            print(f"Ep {epoch:02d} | Phase: {life.phase} | Loss: {total_loss/len(dataloader):.3f}")

# =============================================================================
# 7. EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    # Descomenta la que quieras probar:
    
    # Versión rápida CPU (3M params, ~2min/epoch)
    train_logos(use_nested=False)
    
    # Versión topológica GPU (6M params, ~5min/epoch, necesita GPU)
    # train_logos(use_nested=True)
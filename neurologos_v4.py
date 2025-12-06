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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Pesos Estructurales (Lentos) - Inicialización Ortogonal
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        # Pesos Plásticos (Rápidos/Hebbianos)
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.ln = nn.LayerNorm(out_dim)
        # Controlador de Plasticidad Neurona-Específica (Meta-Learning)
        self.plasticity_controller = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.plasticity_controller[2].bias.data.fill_(-2.0)  # Sesgo hacia plasticidad alta (inicializa cerca de 0.12)
        # CALIBRACIÓN: Learning Rate reducido para evitar oscilación catastrófica
        self.base_lr = 0.015 
        # Para plasticidad supervisada
        self.prediction_error = 0.0
        
    def forward(self, x, global_plasticity=0.0):
        slow_out = self.W_slow(x)
        fast_out = F.linear(x, self.W_fast)
        pre_act = slow_out + fast_out
        # Extracción de estadísticas locales para el controlador
        batch_mean = pre_act.mean(dim=0).unsqueeze(1)
        batch_std = pre_act.std(dim=0).unsqueeze(1) + 1e-6
        stats = torch.cat([batch_mean, batch_std], dim=1)
        # Cálculo de plasticidad local modulada por error de predicción
        learned_plasticity = self.plasticity_controller(stats).squeeze()
        effective_plasticity = global_plasticity * learned_plasticity * (1.0 - self.prediction_error)
        # Activación con Clipping Suave
        out = 5.0 * torch.tanh(self.ln(pre_act) / 5.0)
        # Regla de Aprendizaje Hebbiano con FIXES
        if self.training and effective_plasticity.mean() > 0.001:
            with torch.no_grad():
                # Normalización por energía de entrada
                x_norm = (x ** 2).sum(1).mean() + 1e-6
                # Correlación (y * x)
                correlation = torch.mm(out.T, x) / x.size(0)
                forgetting = 0.2 * self.W_fast 
                delta = torch.clamp((correlation / x_norm) - forgetting, -0.05, 0.05)
                # Aplicación modulada por el controlador aprendido
                lr_vector = effective_plasticity.unsqueeze(1) * self.base_lr
                self.W_fast.data += delta * lr_vector
                self.W_fast.data.mul_(0.999)
        return out

    def consolidate_svd(self, repair_strength=1.0):
        """
        Consolidación espectral de pesos rápidos mediante SVD.
        Repara inestabilidades numéricas y cristaliza conocimiento consolidado.
        Retorna True si se realizó una consolidación activa.
        """
        with torch.no_grad():
            # Umbral de consolidación: solo si los pesos rápidos son significativos
            fast_norm = self.W_fast.norm()
            if fast_norm < 5.0:
                return False
            # SVD para descomposición de la matriz de pesos rápidos
            try:
                U, S, Vt = torch.linalg.svd(self.W_fast, full_matrices=False)
                # Filtrar componentes con valores singulares bajos (reducir ruido)
                threshold = S.max() * 0.01 * repair_strength
                mask = S > threshold
                filtered_S = S * mask.float()
                # Reconstruir matriz consolidada con componentes principales
                W_consolidated = U @ torch.diag(filtered_S) @ Vt
                # Interpolar entre original y consolidada basado en fuerza de reparación
                self.W_fast.data = (1.0 - repair_strength) * self.W_fast.data + \
                                  repair_strength * W_consolidated
                # Escalar para mantener rango dinámico apropiado
                self.W_fast.data *= 0.95
                return True
            except RuntimeError:
                # Fallback: si SVD falla, decaer los pesos rápidos
                self.W_fast.data.mul_(0.9)
                return False



class ConsciousCore(nn.Module):
    def __init__(self):
        super().__init__()
        # Asegurar alineación exacta de dimensiones para MHA
        self.attention_pool = nn.MultiheadAttention(512, 8, batch_first=True)
        self.liquid = LiquidNeuron(512, 512)  # FIX: in_dim y out_dim deben ser explícitos
        self.thought_compressor = nn.Linear(512, 256)  # 512 → 256, sin no-linealidad intermedia
        
    def forward(self, visual_features, plasticity):
        # visual_features: [B, 512] (después de pool si es Nested)
        # Expandir a secuencia de longitud 1 para MHA: [B, 1, 512]
        q = visual_features.unsqueeze(1)
        pooled, _ = self.attention_pool(q, q, q)
        pooled = pooled.squeeze(1)  # [B, 512]
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
        # thought: [B, 256] → debe expandirse a [num_layers=2, B, hidden_dim=256]
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)  # [2, B, 256]
        c0 = torch.zeros_like(h0)  # cero inicializado en la misma dimensión
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
            # visual: [B, 16, 512] → reduce a [B, 512]
            visual = visual.mean(dim=1)
        # En ambos casos, visual es [B, 512] aquí
        
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
        # Templates expandidos: +variabilidad semántica, gramatical y contextual
        self.templates = {
            0: [
                "a red airplane in the sky", "a silver aircraft with wings", "a flying machine above clouds",
                "a jet soaring high", "an air vehicle with metal body", "a passenger plane in flight"
            ],
            1: [
                "a shiny yellow car", "a red automobile on asphalt", "a four-wheeled vehicle driving fast",
                "a compact sedan in motion", "a parked vehicle with tinted windows", "a modern car on highway"
            ],
            2: [
                "a small bird with feathers", "a flying sparrow in daylight", "a winged animal perched on branch",
                "a chirping bird in a tree", "a tiny avian creature with beak", "a feathered animal in flight"
            ],
            3: [
                "a black domestic cat", "a furry feline sitting still", "a quiet house cat with green eyes",
                "a sleeping cat on sofa", "a tabby cat grooming itself", "a pet cat looking alert"
            ],
            4: [
                "a wild deer in forest", "a brown animal with antlers", "a grazing mammal in nature",
                "a shy deer near trees", "a large herbivore in meadow", "an antlered animal standing still"
            ],
            5: [
                "a loyal brown dog", "a playful canine running", "a four-legged pet barking",
                "a wet dog after rain", "a guard dog on alert", "a happy puppy wagging tail"
            ],
            6: [
                "a green frog on lily pad", "a moist amphibian near pond", "a small jumping creature",
                "a croaking frog in water", "a slimy animal with webbed feet", "a pond-dwelling amphibian"
            ],
            7: [
                "a strong brown horse", "a galloping equine in field", "a large farm animal with mane",
                "a calm horse in stable", "a racing horse with rider", "a muscular animal with hooves"
            ],
            8: [
                "a blue cargo ship", "a white vessel on ocean", "a maritime boat with containers",
                "a freighter crossing sea", "a large ship at port", "a steel boat carrying goods"
            ],
            9: [
                "a large delivery truck", "a heavy-duty cargo vehicle", "a diesel-powered transport",
                "a logistics truck on road", "a boxy vehicle with trailer", "a commercial hauler in motion"
            ]
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
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cpu')
    print(f"NeuroLogos v4.0 Diagnóstico Neurofisiológico | Device: {device} | Nested: {use_nested}")
    dataset = CIFARCaptions()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = NeuroLogos(vocab_size=len(dataset.vocab), use_nested=use_nested).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    life = LifeCycle(total_epochs=50)
    print("🧠 Iniciando Ciclo de Vida Neural con Diagnóstico...")
    for epoch in range(50):
        plasticity = life.get_plasticity(epoch)
        model.train()
        if epoch < 30:
            current_transfer_rate = 0.01
        else:
            current_transfer_rate = 0.003
        liquid_module = model.cortex.get_liquid_module()
        if epoch % 10 == 0 and epoch > 0:
            print(f"😴 Consolidando Memoria Media (SVD) en Época {epoch}...")
            consolidated = liquid_module.consolidate_svd(repair_strength=0.7, timescale='medium')
            print(f"  > Consolidación W_medium: {consolidated}")
        if epoch == 30:
            print("📝 Consolidando Memoria Rápida (SVD) en Época 30...")
            consolidated = liquid_module.consolidate_svd(repair_strength=0.5, timescale='fast')
            print(f"  > Consolidación W_fast: {consolidated} en modo: {model.cortex.mode}")
        total_loss = 0
        sum_richness = 0
        sum_vn = 0
        sum_fast_norm = 0
        sum_plasticity = 0
        num_batches = 0
        for images, captions in dataloader:
            images = images * 2 - 1
            optimizer.zero_grad()
            logits = model(images, captions, plasticity=plasticity, transfer_rate=current_transfer_rate)
            # FIX: Usar label smoothing para mejorar generalización del vocabulario
            loss = F.cross_entropy(
                logits.reshape(-1, len(dataset.vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=dataset.word2id["<PAD>"],
                label_smoothing=0.1
            )
            with torch.no_grad():
                error_tensor = (loss / 5.0).clamp(0.0, 0.95)
                liquid_module.prediction_error = error_tensor.item()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            sum_richness += model.cortex.last_richness
            sum_vn += model.cortex.last_vn_entropy
            sum_fast_norm += model.cortex.last_fast_norm
            sum_plasticity += model.cortex.last_effective_plasticity
            num_batches += 1
        avg_loss = total_loss / num_batches
        avg_richness = sum_richness / num_batches
        avg_vn_ent = sum_vn / num_batches
        avg_fast_norm = sum_fast_norm / num_batches
        avg_eff_plasticity = sum_plasticity / num_batches
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, captions in dataloader:
                images = images * 2 - 1
                logits = model(images, captions, plasticity=0.0)
                preds = logits.argmax(dim=-1)
                targets = captions[:, 1:]
                mask = targets != dataset.word2id["<PAD>"]
                correct += (preds[mask] == targets[mask]).sum().item()
                total += mask.sum().item()
            acc = 100.0 * correct / total if total > 0 else 0.0
        if epoch % 10 == 0 or epoch == 29 or epoch == 31:
            with torch.no_grad():
                sample_img, _ = dataset[0]
                sample_img = sample_img * 2 - 1
                generated = model(sample_img.unsqueeze(0), captions=None)
                words = [dataset.id2word.get(int(tok), "<UNK>") for tok in generated[0]]
                sentence = " ".join(w for w in words if w not in ["<BOS>", "<EOS>", "<PAD>"])
                print(f"Ep {epoch:02d} | Phase: {life.phase}")
                print(f"  📊 Loss: {avg_loss:.3f} | Acc (lang): {acc:.1f}%")
                print(f"  🧠 Richness: {avg_richness:.1f} | VN Entropy: {avg_vn_ent:.2f}")
                print(f"  ⚡ Fast Norm: {avg_fast_norm:.2f} | Plasticity: {avg_eff_plasticity:.3f}")
                print(f"  💬 Genera: '{sentence}'")
        else:
            print(f"Ep {epoch:02d} | Phase: {life.phase} | Loss: {avg_loss:.3f} | Rich: {avg_richness:.1f} | Acc: {acc:.1f}%")
            






# =============================================================================
# 7. EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    # Descomenta la que quieras probar:
    
    # Versión rápida CPU (3M params, ~2min/epoch)
    train_logos(use_nested=False)
    
    # Versión topológica GPU (6M params, ~5min/epoch, necesita GPU)
    # train_logos(use_nested=True)
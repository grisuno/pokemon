#%%writefile neurosoberano.py
# =============================================================================
# NeuroLogos v4.0 - CORREGIDO DIMENSIONALMENTE
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('inf')):
    """Filtra logits con Top-K o Top-P (Nucleus) Sampling."""
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold (nucleus)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Map back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def measure_spatial_richness(activations):
    """
    Calcula la riqueza representacional de un tensor de activación.
    Utiliza Entropía de Shannon (por canal) y Entropía de Von Neumann (estructural).
    FIX: Evita el UserWarning de std() con batch=1 y mejora estabilidad numérica.
    """
    if activations.size(0) == 0:
        return 0.0, 1.0, 0.0
        
    # FIX: Muestreo seguro para batches grandes, mínimo 32 muestras
    if activations.size(0) > 64:
        indices = torch.randperm(activations.size(0), device=activations.device)[:64]
        activations = activations[indices]
    else:
        activations = activations.clone()
        
    epsilon = 1e-8
    N, D = activations.shape
    
    # 1. Entropía de Shannon (Riqueza de Actividad)
    norm_activations = activations.abs().sum(dim=1, keepdim=True) + epsilon
    P = activations.abs() / norm_activations
    shannon_entropy = -(P * torch.log(P + epsilon)).sum(dim=1).mean()
    richness = torch.exp(shannon_entropy)
    
    # 2. Entropía de Von Neumann (Riqueza Estructural)
    if N > 1 and D > 1:
        activations_centered = activations - activations.mean(dim=0, keepdim=True)
        C = (activations_centered.T @ activations_centered) / (N - 1 + epsilon)
        
        try:
            eigenvalues = torch.linalg.eigvalsh(C)
            eigenvalues = eigenvalues.clamp(min=0)
            rho = eigenvalues / (eigenvalues.sum() + epsilon)
            vn_entropy = -(rho * torch.log(rho + epsilon)).sum()
        except:
            vn_entropy = torch.tensor(0.0, device=activations.device)
    else:
        vn_entropy = torch.tensor(0.0, device=activations.device)
        
    return shannon_entropy.item(), richness.item(), vn_entropy.item()


class TopologicalCompressor(nn.Module):
    def __init__(self, node_dim=512):
        super().__init__()
        self.attn_weight = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.integrator = LiquidNeuron(node_dim, node_dim)
        
    def forward(self, nodes, plasticity, transfer_rate=0.005):
        # nodes: [B, 16, 512]
        weights = self.attn_weight(nodes)              # [B, 16, 1]
        weights = F.softmax(weights, dim=1)
        thought_pooled = (nodes * weights).sum(dim=1)  # [B, 512]
        thought_final = self.integrator(thought_pooled, plasticity, transfer_rate)
        return thought_final

# =============================================================================
# 1. VISUAL ENCODER (Elige una: MiniUnconscious o NestedUnconscious)
# =============================================================================
class MiniUnconscious(nn.Module):
    """Versión rápida CPU: 512-dim output directo - con procesamiento jerárquico inspirado en vía ventral"""
    def __init__(self):
        super().__init__()
        # Simula procesamiento progresivo de contornos → formas → objetos
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ReLU(),   # RF grande para contornos globales
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),  # Reducción espacial + textura
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), # Complejidad intermedia
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),# Codificación de objetos
            nn.AdaptiveAvgPool2d((2, 2)),           # Representación compacta (2x2)
            nn.Flatten()
        )
        self.topo_bridge = nn.Linear(256*2*2, 512)
        # Normalización cortical post-procesamiento
        self.norm = nn.LayerNorm(512)
        
    def forward(self, x):
        features = self.stem(x)
        encoded = self.topo_bridge(features)
        return self.norm(encoded)  # [B, 512]


class NestedUnconscious(nn.Module):
    def __init__(self, grid_size=4, output_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
        )
        
        self.nodes = 16
        self.mapper = nn.Linear(512, output_dim)  # 512 → 512
        
        self.intra_adj = nn.Parameter(torch.randn(4, 4) * 0.5)
        self.intra_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim*2), nn.GELU(), nn.Linear(output_dim*2, output_dim)
        )
        
        self.inter_adj = nn.Parameter(torch.randn(4, 4) * 0.5)
        self.inter_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim*2), nn.GELU(), nn.Linear(output_dim*2, output_dim)
        )
        
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        B = x.size(0)
        h = self.stem(x)                    # [B, 128, 8, 8]
        h_flat = h.view(B, 16, -1)          # [B, 16, 512]
        nodes = self.mapper(h_flat)         # [B, 16, 512]
        
        # Intra-cluster
        intra_adj = torch.sigmoid(self.intra_adj) * (1 - torch.eye(4, device=x.device))
        nodes_reshaped = nodes.view(B, 4, 4, -1)
        msgs = self.intra_mlp(nodes_reshaped)
        nodes_reshaped = nodes_reshaped + torch.matmul(intra_adj, msgs)
        
        # Inter-cluster
        inter_adj = torch.sigmoid(self.inter_adj) * (1 - torch.eye(4, device=x.device))
        cluster = nodes_reshaped.mean(2)
        cluster_msgs = self.inter_mlp(cluster)
        cluster_update = torch.matmul(inter_adj, cluster_msgs)
        nodes_reshaped = nodes_reshaped + cluster_update.unsqueeze(2)
        
        nodes_final = nodes_reshaped.reshape(B, 16, -1)
        return self.norm(nodes_final)       # [B, 16, 512]


# =============================================================================
# 2. CONSCIOUS SYSTEM
# =============================================================================
class LiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        
        self.register_buffer('W_medium', torch.zeros(out_dim, in_dim))
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        
        self.ln = nn.LayerNorm(out_dim)
        
        # FIX: El controlador solo mira estadísticas globales (mean y std del batch), no por neurona
        self.plasticity_controller = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.plasticity_controller[2].bias.data.fill_(-2.0)
        self.base_lr = 0.015 
        self.prediction_error = 0.0
        
    def forward(self, x, global_plasticity=0.0, transfer_rate=0.005):
        slow_out = self.W_slow(x)
        medium_out = F.linear(x, self.W_medium)
        fast_out = F.linear(x, self.W_fast)
        pre_act = slow_out + medium_out + fast_out
        
        # FIX: Estadísticas GLOBALES del batch (scalar mean y std) → forma [B, 2]
        batch_mean = pre_act.mean(dim=1, keepdim=True)    # [B, 1]
        if pre_act.size(0) > 1:
            batch_std = pre_act.std(dim=1, unbiased=False, keepdim=True) + 1e-6  # [B, 1]
        else:
            batch_std = torch.ones_like(batch_mean) * 1e-6
        stats = torch.cat([batch_mean, batch_std], dim=1)  # [B, 2]
        
        learned_plasticity = self.plasticity_controller(stats).squeeze(1)  # [B]
        effective_plasticity = global_plasticity * learned_plasticity * (1.0 - self.prediction_error)
        
        out = 5.0 * torch.tanh(self.ln(pre_act) / 5.0)
        
        if self.training and effective_plasticity.mean() > 0.001:
            with torch.no_grad():
                out_centered = out - out.mean(dim=0, keepdim=True)
                correlation = torch.mm(out_centered.T, x) / x.size(0)
                
                forgetting = 0.2 * self.W_fast
                delta = torch.clamp(correlation - forgetting, -0.05, 0.05)
                
                self.W_medium.data += self.W_fast.data * transfer_rate
                
                # effective_plasticity es [B] → lo expandimos para multiplicar por neurona de salida
                lr_vector = effective_plasticity.mean() * self.base_lr   # scalar global
                self.W_fast.data += delta * lr_vector
                self.W_fast.data.mul_(1.0 - transfer_rate)
                self.W_fast.data.clamp_(-3.0, 3.0)
                
        return out

    def consolidate_svd(self, repair_strength=1.0, timescale='fast'):
        W_slow_norm = self.W_slow.weight.norm().item()
        
        if timescale == 'fast':
            W_to_consolidate = self.W_fast
            threshold = W_slow_norm * 0.3
        elif timescale == 'medium':
            W_to_consolidate = self.W_medium
            threshold = W_slow_norm * 0.5
        else:
            return False

        with torch.no_grad():
            W_norm = W_to_consolidate.norm().item()
            if W_norm < threshold:
                return False
                
            try:
                U, S, Vt = torch.linalg.svd(W_to_consolidate, full_matrices=False)
                threshold_svd = S.max() * 0.01 * repair_strength
                mask = S > threshold_svd
                filtered_S = S * mask.float()
                W_consolidated = U @ torch.diag(filtered_S) @ Vt
                
                W_to_consolidate.data = (1.0 - repair_strength) * W_to_consolidate.data + repair_strength * W_consolidated
                W_to_consolidate.data *= 0.98
                return True
            except:
                W_to_consolidate.data.mul_(0.9)
                return False


class ConsciousCore(nn.Module):
    def __init__(self):
        super().__init__()
        # Componentes para manejo de Secuencia (NestedUnconscious)
        self.sequence_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        self.topological_compressor = TopologicalCompressor(node_dim=512)
        
        # Componentes para manejo Vectorial (MiniUnconscious)
        self.attention_pool = nn.MultiheadAttention(512, 8, batch_first=True)
        self.liquid = LiquidNeuron(512, 512)
        
        # Eliminamos la compresión 512 -> 256
        self.thought_compressor = nn.Identity()
        self.meta_probe = nn.Linear(512, 64)
        
        # Indicador de modo de procesamiento
        self.mode = "vector"
        
    def forward(self, visual_features, plasticity, transfer_rate=0.005):
        
        if visual_features.dim() == 3: # [B, N, D]
            self.mode = "sequence"
            h, _ = self.sequence_attention(visual_features, visual_features, visual_features)
            thought = self.topological_compressor(h, plasticity, transfer_rate)
            active_liquid = self.topological_compressor.integrator
            
        else: # [B, D]
            self.mode = "vector"
            q = visual_features.unsqueeze(1)
            pooled, _ = self.attention_pool(q, q, q)
            pre_liquid = pooled.squeeze(1)
            thought = self.liquid(pre_liquid, plasticity, transfer_rate)
            active_liquid = self.liquid

        # --- DIAGNÓSTICO SEGURO (usa las mismas estadísticas que LiquidNeuron) ---
        pre_act_raw = active_liquid.W_slow(thought.detach())
        meta_activations = self.meta_probe(pre_act_raw)
        _, richness_val, vn_entropy = measure_spatial_richness(meta_activations)
        
        self.last_richness = richness_val
        self.last_vn_entropy = vn_entropy
        self.last_fast_norm = active_liquid.W_fast.norm().item()
        
        # FIX: cálculo seguro de effective_plasticity usando estadísticas globales del batch
        batch_mean = pre_act_raw.mean(dim=1, keepdim=True)
        if pre_act_raw.size(0) > 1:
            batch_std = pre_act_raw.std(dim=1, unbiased=False, keepdim=True) + 1e-6
        else:
            batch_std = torch.ones_like(batch_mean) * 1e-6
        stats = torch.cat([batch_mean, batch_std], dim=1)  # [B, 2]
        
        learned_plasticity = active_liquid.plasticity_controller(stats).squeeze(1)  # [B]
        error_proxy = max(active_liquid.prediction_error, 0.05)
        self.last_effective_plasticity = (plasticity * learned_plasticity.mean().item()) * (1.0 - error_proxy)
        
        return self.thought_compressor(thought)



    def get_liquid_module(self):
        """Retorna el LiquidNeuron activo para la consolidación externa."""
        if self.mode == "sequence":
            return self.topological_compressor.integrator
        else:
            return self.liquid




# =============================================================================
# 3. ÁREA DE BROCA - Corregido para dimensión 512
# =============================================================================
class BioDecoder(nn.Module):
    # FIX: hidden_dim = 512 para acoplarse a ConsciousCore
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
            embeddings = self.embedding(captions[:, :-1])
            if self.training:
                embeddings = embeddings + torch.randn_like(embeddings) * 0.05
                
            lstm_out, _ = self.lstm(embeddings, self._get_init_state(thought))
            gate = self.liquid_gate(lstm_out)
            lstm_out = lstm_out * gate
            return self.out(lstm_out)
        else:
            generated = []
            input_word = torch.full((batch_size, 1), 1, dtype=torch.long, device=device) 
            hidden = self._get_init_state(thought)
            
            # Parámetros de Muestreo (FIX: Usar Top-P para mayor control)
            temperature = 0.8
            top_k = 0 # Deshabilitar Top-K, preferir Top-P
            top_p = 0.9 # Nucleus Sampling: Mantener las palabras que sumen el 90% de la probabilidad
            
            for _ in range(max_len):
                emb = self.embedding(input_word)
                out, hidden = self.lstm(emb, hidden)
                gate = self.liquid_gate(out)
                out = out * gate
                logits = self.out(out).squeeze(1) # [B, vocab_size]
                
                # 1. Aplicar Temperatura
                logits = logits / temperature
                
                # 2. FIX: Aplicar Top-K/Top-P Filtering
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                
                # 3. Muestreo (sobre las opciones filtradas)
                probabilities = F.softmax(logits, dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1)
                
                generated.append(next_word)
                input_word = next_word
                
            return torch.cat(generated, dim=1)


    def _get_init_state(self, thought):
        # FIX: thought ahora es [B, 512]
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)  # [2, B, 512]
        c0 = torch.zeros_like(h0)
        return (h0, c0)

# =============================================================================
# 4. NEUROLOGOS - Arquitectura Completa
# =============================================================================
class NeuroLogos(nn.Module):
    def __init__(self, vocab_size=1000, use_nested=False):
        super().__init__()
        self.use_nested = use_nested
        
        if use_nested:
            self.eye = NestedUnconscious()
            self.cortex = ConsciousCore()
        else:
            self.eye = MiniUnconscious()
            self.cortex = ConsciousCore()
        
        # BioDecoder debe ser actualizado a 512 si no lo hiciste antes
        self.broca = BioDecoder(vocab_size) 
        self.running_richness = 20.0
        
    def forward(self, image, captions=None, plasticity=0.1, transfer_rate=0.005): # FIX: Agregar transfer_rate
        visual = self.eye(image)
        
        # FIX: Pasar transfer_rate al cortex
        thought = self.cortex(visual, plasticity, transfer_rate) 
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
            # Decaimiento lineal hasta un mínimo
            return max(0.01, 0.2 * (1 - epoch/30)) 
        else:
            self.phase = "LOGOS (Speaking)"
            # FIX: Plasticidad mínima residual para evitar estancamiento
            return 0.001


# =============================================================================
# 5. CICLO DE VIDA DATASET ENTRENAMIENTO
# =============================================================================
class CIFARCaptions:
    def __init__(self):
        self.dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        
        # Templates corregidos: eliminado sesgo cromático, añadida variabilidad semántica y sensorial
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
        
        return image, torch.tensor(token_ids, dtype=torch.long)




# =============================================================================
# 6. ENTRENAMIENTO
# =============================================================================
def train_logos(use_nested=False):
    torch.manual_seed(42)
    np.random.seed(42)
    
    # FIX: detecta GPU correctamente
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"NeuroLogos v4.0 Diagnóstico Neurofisiológico | Device: {device} | Nested: {use_nested}\n")
    
    dataset = CIFARCaptions()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    
    model = NeuroLogos(vocab_size=len(dataset.vocab), use_nested=use_nested).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    life = LifeCycle(total_epochs=50)
    
    print("Iniciando Ciclo de Vida Neural con Diagnóstico...\n")
    
    for epoch in range(50):
        plasticity = life.get_plasticity(epoch)
        model.train()
        
        current_transfer_rate = 0.01 if epoch < 30 else 0.003
        liquid_module = model.cortex.get_liquid_module()
        
        if epoch % 10 == 0 and epoch > 0:
            print(f"Consolidando Memoria Media (SVD) en Época {epoch}...")
            liquid_module.consolidate_svd(repair_strength=0.7, timescale='medium')
        if epoch == 30:
            print("Consolidando Memoria Rápida (SVD) en Época 30...")
            liquid_module.consolidate_svd(repair_strength=0.5, timescale='fast')
        
        total_loss = 0
        sum_richness = sum_vn = sum_fast_norm = sum_plasticity = 0
        num_batches = 0
        
        for images, captions in dataloader:
            images = images.to(device, non_blocking=True) * 2 - 1
            captions = captions.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(images, captions, plasticity=plasticity, transfer_rate=current_transfer_rate)
            
            loss = F.cross_entropy(
                logits.reshape(-1, len(dataset.vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=dataset.word2id["<PAD>"]
            )
            
            with torch.no_grad():
                liquid_module.prediction_error = (loss / 5.0).clamp(0.0, 0.95).item()
            
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
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_img, _ = dataset[0]
                sample_img = sample_img.unsqueeze(0).to(device) * 2 - 1
                generated = model(sample_img, captions=None)
                words = [dataset.id2word.get(int(t.item()), "<UNK>") for t in generated[0]]
                sentence = " ".join(w for w in words if w not in ["<BOS>", "<EOS>", "<PAD>"])
                print(f"Ep {epoch:02d} | {life.phase} | Loss: {avg_loss:.3f} | Rich: {avg_richness:.1f} | Acc: —")
                print(f"   Genera: '{sentence}'\n")
            model.train()
        else:
            print(f"Ep {epoch:02d} | {life.phase} | Loss: {avg_loss:.3f} | Rich: {avg_richness:.1f}")





# =============================================================================
# 7. EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    # Descomenta la que quieras probar:
    
    # Versión rápida CPU (3M params, ~2min/epoch)
    train_logos(use_nested=True)
    
    # Versión topológica GPU (6M params, ~5min/epoch, necesita GPU)
    # train_logos(use_nested=True)
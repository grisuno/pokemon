%%writefile neurosoberano.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
import math

# =============================================================================
# 1. METROLOGÍA ROBUSTA Y DIAGNÓSTICO
# =============================================================================
def measure_spatial_richness(activations):
    """
    Calcula la riqueza representacional (Shannon) y la entropía estructural (Von Neumann).
    Incluye protección contra singularidades numéricas y muestreo para batches grandes.
    """
    if activations.size(0) < 2:
        return torch.tensor(0.0, device=activations.device), 0.0, 0.0

    # Sampling estocástico para eficiencia en batches masivos (>1000)
    if activations.size(0) > 1000:
        indices = torch.randperm(activations.size(0))[:1000]
        x = activations[indices]
    else:
        x = activations

    # Centrado y Jitter Anti-Singularidad (Evita colapso numérico en SVD)
    x = x - x.mean(dim=0, keepdim=True)
    x = x + torch.randn_like(x) * 1e-7 

    # Cálculo de matriz de covarianza
    cov = x.T @ x / (x.size(0) - 1)

    try:
        # Eigenvalores Hermitianos (garantiza valores reales)
        eigs = torch.linalg.eigvalsh(cov).abs()
        
        # Filtrado de ruido espectral (Spectral Noise Floor)
        max_eig = eigs.max()
        if max_eig < 1e-6:
            return torch.tensor(0.0, device=x.device), 1.0, 0.0
            
        threshold = max_eig * 1e-5
        eigs = eigs[eigs > threshold]

        # Normalización de probabilidad espectral
        p = eigs / (eigs.sum() + 1e-12)
        
        # Entropía de Shannon sobre el espectro
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        
        # Métricas derivadas
        richness_val = torch.exp(entropy).item()
        vn_entropy = entropy.item() 

        return entropy, richness_val, vn_entropy

    except RuntimeError:
        # Fallback seguro en caso de error de álgebra lineal (CUDA exceptions)
        return torch.tensor(0.0, device=activations.device), 1.0, 0.0

# =============================================================================
# 2. MOTOR DE HOMEOSTASIS MULTIVARIADA
# =============================================================================
class HomeostasisEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 1.5
        self.momentum = 0.95
        
        # Buffers de estado persistente
        self.register_buffer('running_richness', torch.tensor(20.0))
        self.register_buffer('running_fastnorm', torch.tensor(0.0))
        self.register_buffer('running_loss', torch.tensor(2.5))
        
    def decide(self, task_loss, richness_val, vn_entropy, fast_norm, epoch, total_epochs):
        """
        Arbitraje homeostático con Fase de Maestría.
        Aniquila la exploración en el último 20% del ciclo de vida para cristalizar el conocimiento.
        """
        self.running_richness = self.momentum * self.running_richness + (1-self.momentum) * richness_val
        self.running_fastnorm = self.momentum * self.running_fastnorm + (1-self.momentum) * fast_norm
        self.running_loss = self.momentum * self.running_loss + (1-self.momentum) * task_loss
        
        progress = epoch / total_epochs
        
        # 1. IMPULSO FOCUS: Ganancia progresiva hacia convergencia
        focus_gain = 1.0 + (progress * 1.5)
        loss_ratio = task_loss / (self.running_loss + 1e-6)
        focus_base = math.log1p(task_loss * 20.0) 
        focus_drive = focus_base * max(1.0, loss_ratio * 1.5) * focus_gain
        
        # 2. IMPULSO EXPLORE: Supresión progresiva con target realista
        if progress > 0.8:
            explore_drive = 0.0
        else:
            safe_exploration = torch.sigmoid(2.0 - self.running_loss).item()
            
            # Target adaptativo basado en complejidad arquitectónica
            # Para modelo de ~12M params: 40-60 richness es realista
            base_target = 55.0
            adaptive_target = base_target + (15.0 * safe_exploration)
            
            richness_deficit = max(0.0, adaptive_target - richness_val)
            explore_drive = (richness_deficit / adaptive_target) * 0.1 * (1.0 - progress)  # Menos exploración forzada
        
        # 3. IMPULSO REPAIR: Penalización por inestabilidad
        norm_penalty = max(0.0, (fast_norm - 8.0) / 8.0)
        entropy_floor = 2.0
        entropy_penalty = max(0.0, (entropy_floor - vn_entropy))
        
        repair_drive = (norm_penalty + entropy_penalty * 0.5) * (1.0 + progress)
        
        # Arbitraje Softmax
        logits = torch.tensor([focus_drive, explore_drive, repair_drive], device=self.running_loss.device)
        probs = F.softmax(logits / self.temperature, dim=0)
        
        return probs[0].item(), probs[1].item(), probs[2].item()


# =============================================================================
# 3. NEURONA LÍQUIDA CON PLASTICIDAD APRENDIDA
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
        
        # FIX 1: Estabilidad estadística robusta (unbiased=False evita error con batch=1)
        # Esto elimina el UserWarning: std(): degrees of freedom is <= 0
        batch_mean = pre_act.mean(dim=0).unsqueeze(1)
        if x.size(0) > 1:
            batch_std = pre_act.std(dim=0, unbiased=False).unsqueeze(1) + 1e-6
        else:
            batch_std = torch.zeros_like(batch_mean) + 1e-6
            
        stats = torch.cat([batch_mean, batch_std], dim=1)
        
        # Cálculo de plasticidad local modulada por error de predicción
        learned_plasticity = self.plasticity_controller(stats).squeeze()
        effective_plasticity = global_plasticity * learned_plasticity * (1.0 - self.prediction_error)
        
        # Activación con Clipping Suave
        out = 5.0 * torch.tanh(self.ln(pre_act) / 5.0)

        # Regla de Aprendizaje Hebbiano con FIXES
        if self.training and isinstance(effective_plasticity, torch.Tensor) and effective_plasticity.mean() > 0.001:
            with torch.no_grad():
                # Normalización por energía de entrada
                x_norm = (x ** 2).sum(1).mean() + 1e-6
                
                # Correlación (y * x)
                correlation = torch.mm(out.T, x) / x.size(0)
                
                forgetting = 0.2 * self.W_fast 

                # FIX 2: Protección contra NaNs en el cálculo delta
                delta = torch.clamp((correlation / x_norm) - forgetting, -0.05, 0.05).nan_to_num()
                
                # Aplicación modulada por el controlador aprendido
                lr_vector = effective_plasticity.unsqueeze(1) * self.base_lr
                self.W_fast.data += delta * lr_vector
                
                # FIX 3: Clamping duro de pesos rápidos para evitar explosión de Richness (Colapso Epoca 6)
                self.W_fast.data.clamp_(-2.0, 2.0)
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
            # Este threshold evita consolidación innecesaria en pesos pequeños
            fast_norm = self.W_fast.norm()
            if fast_norm < 5.0:
                return False
            
            # SVD para descomposición de la matriz de pesos rápidos
            try:
                U, S, Vt = torch.linalg.svd(self.W_fast, full_matrices=False)
                
                # Filtrar componentes con valores singulares bajos (reducir ruido)
                # Threshold adaptativo basado en el máximo valor singular
                threshold = S.max() * 0.01 * repair_strength
                mask = S > threshold
                filtered_S = S * mask.float()
                
                # Reconstruir matriz consolidada con componentes principales
                W_consolidated = U @ torch.diag(filtered_S) @ Vt
                
                # Interpolar entre original y consolidada basado en fuerza de reparación
                # Esto permite transición suave en lugar de cambio brusco
                self.W_fast.data = (1.0 - repair_strength) * self.W_fast.data + \
                                  repair_strength * W_consolidated
                
                # Escalar para mantener rango dinámico apropiado
                # Factor de decaimiento suave para estabilidad
                self.W_fast.data *= 0.95
                
                return True
                
            except RuntimeError:
                # Fallback: si SVD falla (por ejemplo, valores NaN), decaer los pesos rápidos
                # Esto actúa como reinicio suave de la componente plástica
                self.W_fast.data.mul_(0.9)
                return False




# =============================================================================
# 4. TOPOLOGÍA AVANZADA CON REGULARIZACIÓN ESPECTRAL
# =============================================================================
class NestedTopoLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Determinar número de clusters (raíz cuadrada del total de nodos para grids cuadrados)
        self.num_clusters = int(math.sqrt(num_nodes))
        if self.num_clusters * self.num_clusters != num_nodes:
            self.num_clusters = max(1, num_nodes // 6)
        self.nodes_per_cluster = num_nodes // self.num_clusters
        
        self.mapper = LiquidNeuron(in_dim, hid_dim)
        
        # NIVEL 1: Topología intra-cluster (compartida entre todos los clusters)
        self.intra_adj_logits = nn.Parameter(
            torch.randn(self.nodes_per_cluster, self.nodes_per_cluster) * 0.5
        )
        
        # Message passing dentro de clusters
        self.node_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        
        # NIVEL 2: Pooling de cluster (mean pooling con transformación aprendida)
        self.cluster_pool = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU()
        )
        
        # NIVEL 3: Topología inter-cluster (comunicación entre clusters)
        self.inter_adj_logits = nn.Parameter(
            torch.randn(self.num_clusters, self.num_clusters) * 0.5
        )
        
        # Message passing entre clusters
        self.cluster_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        
        # NIVEL 4: Broadcasting de info de cluster a nodos
        self.broadcast_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU()
        )
        
        self.norm = nn.LayerNorm(hid_dim)
        self.norm_cluster = nn.LayerNorm(hid_dim)
        
    def get_adj(self):
        """Obtiene matriz de adyacencia intra-cluster con normalización simétrica"""
        adj = torch.sigmoid(self.intra_adj_logits)
        adj = adj * (1 - torch.eye(self.nodes_per_cluster, device=adj.device))
        
        deg = adj.sum(1) + 1e-6
        deg_inv_sqrt = torch.pow(deg, -0.5)
        adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        return adj_norm
    
    def get_inter_adj(self):
        """Obtiene matriz de adyacencia inter-cluster"""
        adj = torch.sigmoid(self.inter_adj_logits)
        adj = adj * (1 - torch.eye(self.num_clusters, device=adj.device))
        
        deg = adj.sum(1) + 1e-6
        deg_inv_sqrt = torch.pow(deg, -0.5)
        adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        return adj_norm

    def forward(self, x, plasticity_gate=0.0, epoch=0, genesis_end=5):
        B, N, C = x.shape
        
        # 1. Mapeo local (transformación individual de cada nodo)
        x_flat = x.reshape(B*N, -1)
        h = self.mapper(x_flat, plasticity_gate)
        h = h.reshape(B, N, -1)
        
        if epoch < genesis_end:
            # Genesis: bypass completo
            return self.norm(h)
        
        # Factor de rampa para activación gradual post-genesis
        ramp_factor = min(1.0, (epoch - genesis_end) / 5.0)
        
        # 2. NIVEL INTRA-CLUSTER: Message passing dentro de cada cluster
        # Reorganizar nodos en clusters: [B, num_clusters, nodes_per_cluster, hid_dim]
        h_clustered = h.reshape(B, self.num_clusters, self.nodes_per_cluster, -1)
        
        intra_adj = self.get_adj()
        h_intra_list = []
        
        for cluster_idx in range(self.num_clusters):
            h_cluster = h_clustered[:, cluster_idx, :, :]
            
            # Message passing dentro del cluster
            messages = self.node_mlp(h_cluster)
            h_aggregated = torch.matmul(intra_adj, messages)
            
            h_intra_list.append(h_cluster + h_aggregated * ramp_factor)
        
        # Reconstruir tensor completo: [B, num_clusters, nodes_per_cluster, hid_dim]
        h_intra = torch.stack(h_intra_list, dim=1)
        
        # 3. NIVEL POOLING: Generar representación de cada cluster
        # Mean pooling sobre nodos de cada cluster
        cluster_repr = h_intra.mean(dim=2)
        cluster_repr = self.cluster_pool(cluster_repr)
        cluster_repr = self.norm_cluster(cluster_repr)
        
        # 4. NIVEL INTER-CLUSTER: Message passing entre clusters
        inter_adj = self.get_inter_adj()
        cluster_messages = self.cluster_mlp(cluster_repr)
        cluster_aggregated = torch.matmul(inter_adj, cluster_messages)
        cluster_updated = cluster_repr + cluster_aggregated * ramp_factor
        
        # 5. NIVEL BROADCASTING: Propagar info de cluster a todos sus nodos
        # Expandir representación de cluster: [B, num_clusters, 1, hid_dim]
        cluster_broadcast = cluster_updated.unsqueeze(2)
        cluster_broadcast = self.broadcast_mlp(cluster_broadcast)
        
        # Expandir a todos los nodos del cluster: [B, num_clusters, nodes_per_cluster, hid_dim]
        cluster_broadcast = cluster_broadcast.expand(-1, -1, self.nodes_per_cluster, -1)
        
        # 6. INTEGRACIÓN FINAL: Combinar estado intra-cluster con info inter-cluster
        h_final = h_intra + cluster_broadcast * ramp_factor
        
        # Aplanar de vuelta a [B, N, hid_dim]
        h_final = h_final.reshape(B, N, -1)
        
        return self.norm(h_final)
    
    def get_topology_loss(self):
        """Regularización para fomentar estructura jerárquica no trivial"""
        # Penalizar topología intra-cluster
        intra_adj = torch.sigmoid(self.intra_adj_logits)
        intra_density = intra_adj.mean()
        target_intra = 0.15 if self.num_nodes > 64 else 0.25
        intra_penalty = (intra_density - target_intra).pow(2) * 3.0
        intra_deg = intra_adj.sum(1)
        intra_deg_std = intra_deg.std()
        intra_deg_penalty = F.relu(3.0 - intra_deg_std)
        
        # Penalizar topología inter-cluster
        inter_adj = torch.sigmoid(self.inter_adj_logits)
        inter_density = inter_adj.mean()
        target_inter = 0.20 if self.num_clusters > 8 else 0.35
        inter_penalty = (inter_density - target_inter).pow(2) * 2.0
        inter_deg = inter_adj.sum(1)
        inter_deg_std = inter_deg.std()
        inter_deg_penalty = F.relu(2.0 - inter_deg_std)
        
        return (intra_penalty + 0.1 * intra_deg_penalty + 
                inter_penalty + 0.1 * inter_deg_penalty)


        
# =============================================================================
# 5. ARQUITECTURA DUAL (VERSIÓN FINAL)
# =============================================================================
class ResTopoBlock(nn.Module):
    """Bloque Residual con Inyección Topológica"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.topo_mix = nn.Conv2d(out_c, out_c, 1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # FIX: Modulación topológica real (no sumar 'out' dos veces)
        topo_gate = torch.sigmoid(self.topo_mix(out))
        out = (out * topo_gate) + res
        
        return F.relu(out)




# =============================================================================
# 5. ARQUITECTURA DUAL 
# =============================================================================
class UnconsciousSystem(nn.Module):
    def __init__(self, in_channels, grid_size, hidden_dim):
        super().__init__()
        # Factor de ensanchamiento aumentado para capacidad SOTA (Wide-ResNet style)
        k = 4  # Incrementado de 2 a 4 (Wide-ResNet factor)
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        
        # Aumentado número de bloques de 2 a 3 por capa para mayor profundidad
        self.layer1 = self._make_layer(64, 64*k, 3, stride=1)  # 3 bloques en lugar de 2
        self.layer2 = self._make_layer(64*k, 128*k, 3, stride=2)  # 3 bloques en lugar de 2
        self.layer3 = self._make_layer(128*k, 256*k, 3, stride=2)  # 3 bloques en lugar de 2
        
        # Proyección ajustada a la nueva dimensión (256 * k = 1024)
        self.proj = nn.Conv2d(256*k, hidden_dim, 1)
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        num_nodes = grid_size * grid_size
        self.topo1 = NestedTopoLayer(hidden_dim, hidden_dim, num_nodes)
        self.topo2 = NestedTopoLayer(hidden_dim, hidden_dim * 3, num_nodes)
        
        # Dimensiones de salida por nodo (hidden_dim * 3 por la expansión en topo2)
        self.out_channels = hidden_dim * 3 # FIX: topo2 expande a 3x (GELU + topo)

    def _make_layer(self, in_c, out_c, num_blocks, stride):
        layers = [ResTopoBlock(in_c, out_c, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResTopoBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, x, plasticity_gate, epoch=0, genesis_end=5):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        nodes = self.pool(self.proj(x)).flatten(2).transpose(1, 2)
        
        h1 = self.topo1(nodes, plasticity_gate, epoch, genesis_end)
        h2 = self.topo2(F.gelu(h1), plasticity_gate, epoch, genesis_end)
        
        return h2, self.topo1.get_topology_loss() + self.topo2.get_topology_loss()



class ConsciousAttention(nn.Module):
    """Atención Multi-Cabeza para workspace consciente"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.query_token = nn.Parameter(torch.randn(1, 1, dim))
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, feats):
        # feats llega como [Batch, Sequence_Len, Dim]
        B = feats.shape[0]
        
        # Query token expande a batch [Batch, 1, Dim]
        q = self.query_token.expand(B, -1, -1)
        
        # FIX: 'feats' ya es 3D (Batch, Nodos, Dim), funciona directo como Key/Value
        attn_out, _ = self.mha(q, feats, feats)
        
        # Retornamos vector de contexto [Batch, Dim]
        return self.ln(attn_out.squeeze(1))

class ConsciousSystem(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes):
        super().__init__()
        self.homeostasis = HomeostasisEngine()
        
        # Proyección de entrada para adaptar canales del inconsciente al workspace
        self.workspace_proj = nn.Linear(in_dim, hid_dim)
        
        self.attention = ConsciousAttention(hid_dim, num_heads=4)
        self.wm = LiquidNeuron(hid_dim, hid_dim)
        self.head = nn.Linear(hid_dim, num_classes)
        self.meta = nn.Linear(hid_dim, 64)

    def forward(self, feats, plasticity_gate):
        # feats shape: [Batch, Nodos, Canales_Entrada]
        
        # 1. Proyección
        ws = self.workspace_proj(feats) # [Batch, Nodos, hid_dim]
        
        # 2. Atención Global
        attended = self.attention(ws)   # [Batch, hid_dim]
        
        # 3. Memoria de Trabajo Líquida
        h = F.gelu(self.wm(attended, plasticity_gate))
        
        logits = self.head(h)
        rich_ent, rich_val, vn_ent = measure_spatial_richness(self.meta(h))
        
        return logits, rich_ent, rich_val, vn_ent, attended.abs().mean()

# =============================================================================
# 5. ARQUITECTURA DUAL - FIX DE INTERFAZ
# =============================================================================
class DualMind(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Inconsciente visual
        hidden_dim_unc = 64
        self.unconscious = UnconsciousSystem(3, 12, hidden_dim_unc)
        sensor_channels = hidden_dim_unc * 3  # FIX: coincide con out_channels
        # 2. Consciente cognitivo
        self.conscious = ConsciousSystem(in_dim=sensor_channels, hid_dim=128, num_classes=10)
        # 3. EMA
        self.ema_conscious = None
        # 4. ÁREA DE BROCA (CORREGIDO)
        self.thought_to_lang = nn.Linear(128, 256)  # Proyección desde attended (128D → 256D)
        self.captions = CIFARCaptions()
        self.broca = BioDecoder(vocab_size=len(self.captions.vocab))

    def update_ema(self, decay=0.999):
        """
        Actualiza Exponential Moving Average de los pesos del sistema consciente.
        Estabiliza la inferencia final y mejora generalización en alta precisión.
        """
        from copy import deepcopy
        
        with torch.no_grad():
            if self.ema_conscious is None:
                self.ema_conscious = deepcopy(self.conscious.state_dict())
            else:
                for k, v in self.conscious.state_dict().items():
                    self.ema_conscious[k] = decay * self.ema_conscious[k] + (1-decay) * v



    def forward(self, x, mode='dual', plasticity_gate=0.0, epoch=0, genesis_end=5, captions=None):
        feats, topo_loss = self.unconscious(x, plasticity_gate, epoch, genesis_end)
        if mode == 'unconscious':
            return feats, topo_loss
        if mode == 'classify_only':
            plasticity_gate = 0.0  # fuerza modo cristalizado
            lang_weight = 0.0
            # y opcionalmente bypass Broca
            return out['logits']

        if mode == 'speak':
            plasticity_gate = 0.3  # permite que las celdas altas "piensen"
            # fuerza generación
            return out['caption']
        # Flujo cognitivo original
        logits, rich_ent, rich_val, vn_ent, gaze = self.conscious(feats, plasticity_gate)
        output = {
            'logits': logits,
            'richness_val': rich_val,
            'richness_ent': rich_ent,
            'vn_entropy': vn_ent,
            'gaze_width': gaze,
            'topo_loss': topo_loss,
            'fast_norm': self.conscious.wm.W_fast.norm()
        }
        # Flujo lingüístico (solo si se solicita)
        if mode == 'linguistic' or captions is not None:
            # CORRECCIÓN: Usar SOLO el attended vector (representación consciente real)
            ws = self.conscious.workspace_proj(feats)
            attended = self.conscious.attention(ws)  # [B, 128]
            thought_256 = self.thought_to_lang(attended)  # [B, 256]
            if captions is not None:
                lang_logits = self.broca(thought_256, captions=captions)
                output['lang_logits'] = lang_logits
            else:
                caption = self.broca(thought_256, captions=None)
                output['caption'] = caption
        return output

    def _compute_topo_entropy(self, layer):
        """Calcula entropía real de la topología para diagnóstico (intra + inter)"""
        with torch.no_grad():
            # Entropía de topología intra-cluster
            intra_adj = torch.sigmoid(layer.intra_adj_logits)
            p_intra = intra_adj.flatten()
            ent_intra = -torch.sum(p_intra * torch.log(p_intra + 1e-6) + 
                                (1-p_intra) * torch.log(1-p_intra + 1e-6))
            ent_intra = ent_intra / (layer.nodes_per_cluster ** 2)
            
            # Entropía de topología inter-cluster
            inter_adj = torch.sigmoid(layer.inter_adj_logits)
            p_inter = inter_adj.flatten()
            ent_inter = -torch.sum(p_inter * torch.log(p_inter + 1e-6) + 
                                (1-p_inter) * torch.log(1-p_inter + 1e-6))
            ent_inter = ent_inter / (layer.num_clusters ** 2)
            
            # Retornar promedio ponderado (intra tiene más nodos, más peso)
            total_ent = (ent_intra * 0.7 + ent_inter * 0.3).item()
            
            return total_ent
    
    def get_system_status(self):
        with torch.no_grad():
            # Métricas de topología intra-cluster
            intra_adj_l1 = torch.sigmoid(self.unconscious.topo1.intra_adj_logits)
            intra_adj_l2 = torch.sigmoid(self.unconscious.topo2.intra_adj_logits)
            
            # Métricas de topología inter-cluster
            inter_adj_l1 = torch.sigmoid(self.unconscious.topo1.inter_adj_logits)
            inter_adj_l2 = torch.sigmoid(self.unconscious.topo2.inter_adj_logits)
            
            return {
                'fast_weights_norm': self.conscious.wm.W_fast.norm().item(),
                'homeostasis_loss': self.conscious.homeostasis.running_loss.item(),
                'homeostasis_richness': self.conscious.homeostasis.running_richness.item(),
                'topology_loss': self.unconscious.topo1.get_topology_loss().item() + self.unconscious.topo2.get_topology_loss().item(),
                'intra_density_l1': intra_adj_l1.mean().item(),
                'intra_density_l2': intra_adj_l2.mean().item(),
                'inter_density_l1': inter_adj_l1.mean().item(),
                'inter_density_l2': inter_adj_l2.mean().item(),
                'topology_entropy_l1': self._compute_topo_entropy(self.unconscious.topo1),
                'topology_entropy_l2': self._compute_topo_entropy(self.unconscious.topo2),
                'num_clusters_l1': self.unconscious.topo1.num_clusters,
                'num_clusters_l2': self.unconscious.topo2.num_clusters,
            }



class LabelSmoothingLoss(nn.Module):
    """
    Función de pérdida con suavizado de etiquetas para evitar sobreajuste
    en distribuciones de probabilidad extremas (necesario para >99%).
    """
    def __init__(self, classes=10, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, pred, target):
        pred = self.log_softmax(pred)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))

# =============================================================================
# 8. ÁREA DE BROCA - ADAPTADA A NEUROSOVEREIGN v4
# =============================================================================

class CIFARCaptions:
    """Dataset de captions para CIFAR-10, reutilizable desde NeuroLogos"""
    def __init__(self):
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
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}
    
    def get_caption(self, label):
        return np.random.choice(self.templates[label])
    
    def tokenize(self, caption, max_len=20):
        tokens = ["<BOS>"] + caption.split() + ["<EOS>"]
        token_ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in tokens]
        token_ids = token_ids[:max_len] + [self.word2id["<PAD>"]] * (max_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

class BioDecoder(nn.Module):
    """Decoder lingüístico minimalista, adaptado a 256D hidden"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
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
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        return (h0, c0)

# =============================================================================
# 1. CLASE NeuralConfig
# =============================================================================
class NeuralConfig:
    """Parámetros del ciclo de vida neural - Modificable para investigación ágil"""
    def __init__(self, mode='research'):
        # CICLO DE VIDA TEMPORAL - CORREGIDO PARA SECUENCIA LÓGICA
        if mode == 'research':
            self.total_epochs = 60
            self.genesis_end = 5
            self.awakening_end = 40
            
        elif mode == 'production':
            self.total_epochs = 200
            self.genesis_end = 20
            self.awakening_end = 160
            
        elif mode == 'debug':
            self.total_epochs = 20
            self.genesis_end = 5
            self.awakening_end = 15
        
        # PLASTICIDAD NEURAL - FIX: Mínimo más alto en Awakening para evitar colapso
        self.plasticity_genesis = 0.8
        self.plasticity_awakening_start = 0.5
        self.plasticity_awakening_end = 0.15
        self.plasticity_crystallization = 0.0
        
        # OPTIMIZACIÓN
        self.lr_unconscious = 0.01
        self.lr_conscious = 0.002
        self.weight_decay_unc = 2e-4
        self.weight_decay_con = 3e-5
        self.gradient_clip_norm = 1.0
        
        # MIXUP PROTOCOL - FIX: Más largo para estabilizar visión antes del lenguaje
        self.mixup_start_epoch = 5
        self.mixup_end_epoch = 30          # <--- EXTENDIDO (era 25, ahora 30)
        self.mixup_probability = 0.7
        self.mixup_alpha = 1.0
        self.mixup_beta = 1.0
        
        # LABEL SMOOTHING
        self.label_smoothing = 0.05
        
        # HOMEOSTASIS
        self.homeostasis_start_epoch = 5
        self.explore_loss_weight = 0.01
        self.repair_loss_weight = 0.01
        self.topo_loss_weight = 0.005
        self.svd_repair_threshold = 0.75
        
        # EMERGENCY MONITOR
        self.emergency_window = 100
        self.emergency_trend_threshold = -0.5
        self.emergency_richness_threshold = 15
        self.emergency_accuracy_threshold = 20
        
        # LOGGING
        self.log_batch_interval = 100
        self.log_status_interval = 5
        self.checkpoint_interval = 10
        self.checkpoint_path = '/content/drive/MyDrive/NeuroSovereign_Checkpoints/'
        self.enable_tensorboard = False
        self.enable_json_logging = True

# =============================================================================
# FUNCIÓN train_neurosovereign
# =============================================================================
def train_neurosovereign(model, train_loader, test_loader, device, config=None):
    """
    Sistema de entrenamiento homeostático con ciclo de vida neural Y LINGÜÍSTICO.
    """
    if config is None:
        config = NeuralConfig(mode='debug')
    
    print(f"\n{'='*80}")
    print(f"🧬 NeuroSovereign v4.0 'Logos' | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Goal: Vision-Language Cognition | Awakening: {config.awakening_end}")
    print(f"{'='*80}\n")
    
    # Inicialización de optimizadores
    opt_unc = optim.AdamW(
        model.unconscious.parameters(), 
        lr=config.lr_unconscious, 
        weight_decay=config.weight_decay_unc
    )
    
    # CORRECCIÓN: Incluir Área de Broca y Proyección en el optimizador consciente
    conscious_params = list(model.conscious.parameters()) + \
                       list(model.broca.parameters()) + \
                       list(model.thought_to_lang.parameters())
                       
    opt_con = optim.AdamW(
        conscious_params, 
        lr=config.lr_conscious, 
        weight_decay=config.weight_decay_con
    )
    
    # Schedulers
    scheduler_unc = optim.lr_scheduler.OneCycleLR(
        opt_unc, max_lr=config.lr_unconscious * 10,
        epochs=config.total_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.15, anneal_strategy='cos'
    )
    scheduler_con = optim.lr_scheduler.OneCycleLR(
        opt_con, max_lr=config.lr_conscious * 10,
        epochs=config.total_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.15, anneal_strategy='cos'
    )
    
    criterion_cls = LabelSmoothingLoss(classes=10, smoothing=config.label_smoothing).to(device)
    criterion_lang = nn.CrossEntropyLoss(ignore_index=model.captions.word2id["<PAD>"]) # Ignorar padding
    
    richness_history = []
    emergency_counter = 0
    total_consolidations = 0
    training_log = []
    
    # TensorBoard
    writer = None
    if config.enable_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(f'runs/neuro_{time.strftime("%Y%m%d_%H%M%S")}')
        except ImportError: pass
    
    for epoch in range(config.total_epochs):
        model.train()
        
        # Fases
        if epoch < config.genesis_end:
            phase = "PHASE 1: GENESIS"
            plasticity_gate = config.plasticity_genesis
            lang_weight = 0.1 # Poco peso al lenguaje al inicio, priorizar visión
        elif epoch < config.awakening_end:
            phase = "PHASE 2: AWAKENING"
            progress = (epoch - config.genesis_end) / (config.awakening_end - config.genesis_end)
            plasticity_gate = config.plasticity_awakening_start * (1.0 - progress) + 0.35
            # Ramp-up gradual del lenguaje para evitar choque en Awakening
            lang_weight = min(0.4, 0.02 + (progress * 0.5))
        else:
            phase = "PHASE 3: CRYSTALLIZATION"
            plasticity_gate = 0.0
            lang_weight = 1.0
            
            if epoch == config.awakening_end and hasattr(model.conscious.wm, 'consolidate_svd'):
                print("  SYNAPTIC CRYSTALLIZATION...")
                model.conscious.wm.consolidate_svd(repair_strength=1.0)
        
        metrics = {'loss': [], 'acc': [], 'rich': [], 'lang_loss': []} 
        epoch_start = time.perf_counter()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # --- PREPARACIÓN LINGÜÍSTICA ---
            # 1. Generar captions sintéticos basados en labels
            captions_text = [model.captions.get_caption(label.item()) for label in y]
            # 2. Tokenizar
            captions_tensor = torch.stack([model.captions.tokenize(c) for c in captions_text]).to(device)
            
            use_mixup = (epoch >= config.mixup_start_epoch and epoch < config.mixup_end_epoch and np.random.random() < config.mixup_probability)
            
            if use_mixup:
                current_plasticity = 0.0
                lam = np.random.beta(config.mixup_alpha, config.mixup_beta)
                index = torch.randperm(x.size(0)).to(device)
                mixed_x = lam * x + (1 - lam) * x[index]
                
                # En mixup, desactivamos el entrenamiento de lenguaje para evitar gradientes basura
                # pero mantenemos el flujo visual
                out = model(mixed_x, mode='dual', plasticity_gate=current_plasticity, epoch=epoch, genesis_end=config.genesis_end)
                loss_vis = lam * criterion_cls(out['logits'], y) + (1 - lam) * criterion_cls(out['logits'], y[index])
                loss_lang = torch.tensor(0.0, device=device) 
                
            else:
                current_plasticity = plasticity_gate
                # Pasamos captions para Teacher Forcing en Broca
                out = model(x, mode='dual', plasticity_gate=current_plasticity, epoch=epoch, genesis_end=config.genesis_end, captions=captions_tensor)
                
                loss_vis = criterion_cls(out['logits'], y)
                
                # CÁLCULO DE PÉRDIDA LINGÜÍSTICA
                logits_flat = out['lang_logits'].reshape(-1, out['lang_logits'].size(-1))
                targets_flat = captions_tensor[:, 1:].reshape(-1) # Ignoramos el primer token del target (<BOS>)
                
                if logits_flat.size(0) != targets_flat.size(0):
                     min_len = min(logits_flat.size(0), targets_flat.size(0))
                     logits_flat = logits_flat[:min_len]
                     targets_flat = targets_flat[:min_len]
                if loss_lang < 0.05 and epoch > 8:
                    lang_weight *= 3.0   # fuerza al decoder a despertar aunque duela
                    print("    Decoder emergency boost activated!")
                loss_lang = criterion_lang(logits_flat, targets_flat)

            # Modulación de Homeostasis
            if epoch >= config.homeostasis_start_epoch:
                p_focus, p_explore, p_repair = model.conscious.homeostasis.decide(
                    loss_vis.item(), out['richness_val'], out['vn_entropy'], 
                    out['fast_norm'].item(), epoch, config.total_epochs
                )
                explore_loss = -out['richness_ent'] * config.explore_loss_weight
                repair_loss = (out['fast_norm'] * config.repair_loss_weight) + out['topo_loss']
                
                total_loss_vis = (loss_vis * p_focus) + (explore_loss * p_explore) + (repair_loss * p_repair)
                
                # FIX: Emergency boost solo si lang_loss muy baja (decoder dormido)
                if loss_lang.item() < 0.05 and epoch > 8:
                    lang_weight *= 3.0
                    print("    Decoder emergency boost activated!")
                
                # INTEGRACIÓN VISIÓN + LENGUAJE
                total_loss = total_loss_vis + (loss_lang * lang_weight)
                
                # Consolidación SVD
                if hasattr(model.conscious.wm, 'consolidate_svd') and p_repair > config.svd_repair_threshold and not use_mixup:
                     model.conscious.wm.consolidate_svd(repair_strength=p_repair)
            else:
                # FIX: Emergency boost también fuera de homeostasis
                if loss_lang.item() < 0.05 and epoch > 8:
                    lang_weight *= 3.0
                    print("    Decoder emergency boost activated!")
                total_loss = loss_vis + (loss_lang * lang_weight) + out['topo_loss'] * config.topo_loss_weight
                
            # Backprop
            opt_unc.zero_grad()
            opt_con.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            opt_unc.step()
            opt_con.step()
            
            # Update OneCycle
            scheduler_unc.step()
            scheduler_con.step()
            model.update_ema()
            
            # Métricas
            acc = (out['logits'].argmax(1) == y).float().mean().item() * 100
            metrics['loss'].append(total_loss.item())
            metrics['lang_loss'].append(loss_lang.item())
            metrics['acc'].append(acc)
            metrics['rich'].append(out['richness_val'])
            
            if batch_idx % config.log_batch_interval == 0:
                batch_time = (time.perf_counter() - epoch_start) / (batch_idx + 1) * 1000
                print(f"  Batch {batch_idx:04d} | VisLoss: {loss_vis.item():.3f} | LangLoss: {loss_lang.item():.3f} | Acc: {acc:.1f}% | "
                      f"Rich: {out['richness_val']:.1f} | Mode: {'Mixup' if use_mixup else 'Real'} | Time: {batch_time:.1f}ms")

        # Fin de época
        time_ep = time.perf_counter() - epoch_start
        print(f"\nEp {epoch+1:03d} [{phase}] | Loss: {np.mean(metrics['loss']):.3f} | LangLoss: {np.mean(metrics['lang_loss']):.3f} | Acc: {np.mean(metrics['acc']):.2f}% | Time: {time_ep:.1f}s")
        
        # PRUEBA DE HABLA (Generation Check)
        if (epoch + 1) % 1 == 0: 
            model.eval()
            with torch.no_grad():
                # Tomar una imagen del test set
                x_sample, y_sample = next(iter(test_loader))
                x_sample = x_sample[:1].to(device)
                
                # Generar
                out = model(x_sample, mode='linguistic') 
                caption_ids = out['caption'][0] 
                
                # Decodificar
                words = []
                for tid in caption_ids:
                    word = model.captions.id2word.get(tid.item(), "<UNK>")
                    if word == "<EOS>": break
                    if word not in ["<BOS>", "<PAD>"]: words.append(word)
                
                sentence = " ".join(words)
                label_name = model.captions.templates[y_sample[0].item()][0] 
                print(f"  🗣️  [DICE]: '{sentence}' | [REAL]: Clase {y_sample[0].item()} (ej: {label_name})")
                print(f"{'-'*80}")

    return model, training_log




# =============================================================================
# 7. RUNNER
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # CONFIGURACIÓN DEL EXPERIMENTO
    # Cambiar 'research' por 'production' o 'debug' según necesidad
    config = NeuralConfig(mode='research')
    
    # Data Augmentation
    from torchvision.transforms import AutoAugment, AutoAugmentPolicy
    
    tf_train = transforms.Compose([
        transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.20), value='random')
    ])
    
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    
    train_dl = DataLoader(
        datasets.CIFAR10('./data', True, download=True, transform=tf_train), 
        batch_size=128, 
        shuffle=True, 
        num_workers=2, 
        drop_last=True, 
        pin_memory=True
    )
    test_dl = DataLoader(
        datasets.CIFAR10('./data', False, download=True, transform=tf_test), 
        batch_size=128, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    model = DualMind().to(device)
    model, training_log = train_neurosovereign(model, train_dl, test_dl, device, config)
    
    print(f"\n{'='*80}")
    print("FINAL EVALUATION (CRYSTALLIZED STATE)")
    print(f"{'='*80}\n")
    print("🔒 Loading EMA weights for final evaluation...")
    model.conscious.load_state_dict(model.ema_conscious)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            out = model(x, mode='dual', plasticity_gate=0.0)

            correct += (out['logits'].argmax(1) == y).sum().item()
            total += y.size(0)
    
    final_acc = 100 * correct / total
    print(f"  Final Accuracy: {final_acc:.2f}%")
    
    if final_acc >= 99.0:
        print(f"\n  👑 GOD-TIER ACHIEVED: {final_acc:.2f}%")
    elif final_acc >= 98.0:
        print(f"\n  🏆 SOTA ACHIEVED: {final_acc:.2f}%")
    else:
        print(f"\n  📊 High-Performance: {final_acc:.2f}%")
    
    print(f"\n  [FINAL SYSTEM DIAGNOSIS]")
    final_status = model.get_system_status()
    for key, value in final_status.items():
        print(f"    {key}: {value:.5f}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()


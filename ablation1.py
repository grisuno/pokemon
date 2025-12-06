import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import json
import time
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from dataclasses import dataclass 
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURACIÓN RADICAL PARA 100% ACCURACY
# =============================================================================
@dataclass
class EliteConfig:
    device: str = "cpu"
    seed: int = 42
    
    # Dataset realista y balanceado
    n_samples: int = 2000
    n_features: int = 20
    n_classes: int = 3
    n_informative: int = 16
    
    # Arquitectura más profunda
    grid_size: int = 3
    embed_dim: int = 12
    hidden_dim: int = 16
    
    # Entrenamiento intensivo
    batch_size: int = 32
    epochs: int = 20
    lr: float = 0.005
    warmup_epochs: int = 3
    
    # Adversarial curriculum (progresivo)
    train_eps_start: float = 0.05
    train_eps_end: float = 0.3
    test_eps: float = 0.3
    pgd_steps: int = 7
    
    # Componentes optimizados
    use_plasticity: bool = True
    use_continuum: bool = True
    use_homeostasis: bool = True
    use_supcon: bool = True
    use_ensemble: bool = True
    use_spectral_norm: bool = True


# =============================================================================
# CONFIGURACIÓN PARA EL ESTUDIO DE ABLACIÓN (Extiende EliteConfig)
# =============================================================================
@dataclass
class AblationConfig(EliteConfig):
    name: str = "Elite_Full"
    skip_memory: bool = False

def create_ablation_configs(config: EliteConfig) -> Dict[str, AblationConfig]:
    """Crea un diccionario de configuraciones para cada test de ablación."""
    
    base_dict = config.__dict__.copy()
    
    # 0. CONFIGURACIÓN ELITE COMPLETA (BASELINE DEL ESTUDIO)
    full_config = AblationConfig(**base_dict, name="Elite_Full")
    
    # 1. BASELINE SIMPLE: Todos los nuevos componentes deshabilitados.
    baseline_dict = base_dict.copy()
    baseline_dict['use_plasticity'] = False
    baseline_dict['use_homeostasis'] = False
    baseline_dict['use_supcon'] = False
    baseline_dict['use_ensemble'] = False
    baseline_dict['use_spectral_norm'] = False
    baseline_dict['skip_memory'] = True
    
    baseline_config = AblationConfig(
        **baseline_dict,
        name="Ablation_0_Simple_Baseline"
    )

    # 2. HOMEÓSTASIS/GATING
    homeostasis_dict = base_dict.copy()
    homeostasis_dict['use_homeostasis'] = False
    homeostasis_config = AblationConfig(
        **homeostasis_dict,
        name="Ablation_1_No_Homeostasis"
    )
    
    # 3. PLASTICIDAD TOPOLÓGICA
    plasticity_dict = base_dict.copy()
    plasticity_dict['use_plasticity'] = False
    plasticity_config = AblationConfig(
        **plasticity_dict,
        name="Ablation_2_No_Plasticity"
    )
    
    # 4. ENSEMBLE
    ensemble_dict = base_dict.copy()
    ensemble_dict['use_ensemble'] = False
    ensemble_config = AblationConfig(
        **ensemble_dict,
        name="Ablation_3_No_Ensemble"
    )
    
    # 5. MEMORIA EPISÓDICA
    memory_dict = base_dict.copy()
    memory_dict['skip_memory'] = True
    memory_config = AblationConfig(
        **memory_dict,
        name="Ablation_4_No_Memory"
    )

    # 6. PÉRDIDA SUPCON
    supcon_dict = base_dict.copy()
    supcon_dict['use_supcon'] = False
    supcon_config = AblationConfig(
        **supcon_dict,
        name="Ablation_5_No_SupCon"
    )

    # 7. NORMALIZACIÓN ESPECTRAL
    spectral_dict = base_dict.copy()
    spectral_dict['use_spectral_norm'] = False
    spectral_config = AblationConfig(
        **spectral_dict,
        name="Ablation_6_No_SpectralNorm"
    )

    return {
        full_config.name: full_config,
        baseline_config.name: baseline_config,
        homeostasis_config.name: homeostasis_config,
        plasticity_config.name: plasticity_config,
        ensemble_config.name: ensemble_config,
        memory_config.name: memory_config,
        supcon_config.name: supcon_config,
        spectral_config.name: spectral_config,
    }


# =============================================================================
# UTILIDADES
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_elite_dataset(config: EliteConfig):
    """Dataset más grande y balanceado con separabilidad controlada"""
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        n_informative=config.n_informative,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.02,
        class_sep=1.2,
        random_state=config.seed
    )
    # Normalización robusta
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)


# =============================================================================
# COMPONENTES AVANZADOS
# =============================================================================
class EpisodicMemory(nn.Module):
    """Memoria explícita de patrones adversariales"""
    def __init__(self, dim, capacity=64):
        super().__init__()
        self.capacity = capacity
        self.register_buffer('memory', torch.zeros(capacity, dim))
        self.register_buffer('labels', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def update(self, x, y):
        """Almacena ejemplos duros"""
        batch_size = x.size(0)
        ptr = int(self.ptr)
        if ptr + batch_size > self.capacity:
            self.memory[:batch_size] = x
            self.labels[:batch_size] = y
            self.ptr[0] = batch_size
        else:
            self.memory[ptr:ptr+batch_size] = x
            self.labels[ptr:ptr+batch_size] = y
            self.ptr[0] = ptr + batch_size
    
    def retrieve(self, x, k=5):
        """Recupera k vecinos más cercanos"""
        if self.ptr[0] == 0:
            return torch.zeros_like(x)
        valid_mem = self.memory[:int(self.ptr[0])]
        x_norm = F.normalize(x, dim=1)
        mem_norm = F.normalize(valid_mem, dim=1)
        sim = torch.mm(x_norm, mem_norm.T)
        
        topk = torch.topk(sim, min(k, valid_mem.size(0)), dim=1)
        weights = F.softmax(topk.values, dim=1)
        retrieved_items = valid_mem[topk.indices]
        retrieved = (retrieved_items * weights.unsqueeze(2)).sum(dim=1)
        
        return retrieved


class SpectralNormLinear(nn.Module):
    """Linear con normalización espectral para estabilidad Lipschitz"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('u', F.normalize(torch.randn(out_features), dim=0))
        nn.init.xavier_normal_(self.weight)
        
    def power_iteration(self, n_iter=1):
        """Aproxima la norma espectral máxima"""
        with torch.no_grad():
            for _ in range(n_iter):
                v = F.normalize(torch.mv(self.weight.T, self.u), dim=0)
                self.u = F.normalize(torch.mv(self.weight, v), dim=0)
    
    def forward(self, x):
        if self.training:
            self.power_iteration()
        
        with torch.no_grad():
            v = F.normalize(torch.mv(self.weight.T, self.u), dim=0)
            sigma = torch.dot(self.u, torch.mv(self.weight, v))

        W_norm = self.weight / (sigma.clamp(min=1e-8))
        return F.linear(x, W_norm, self.bias)


class AdvancedHomeostaticCell(nn.Module):
    """Neurona con control fisiológico multinivel + memoria"""
    def __init__(self, d_in, d_out, use_spectral=False, use_homeostasis=True):
        super().__init__()
        
        self.use_homeostasis = use_homeostasis
        
        if use_spectral:
            self.W_slow = SpectralNormLinear(d_in, d_out)
        else:
            self.W_slow = nn.Linear(d_in, d_out)
        
        self.W_fast = nn.Linear(d_in, d_out, bias=False)
        nn.init.zeros_(self.W_fast.weight)
        
        if use_homeostasis:
            self.input_gate = nn.Sequential(nn.Linear(d_in, d_out), nn.Sigmoid())
            self.forget_gate = nn.Sequential(nn.Linear(d_in + d_out, d_out), nn.Sigmoid())
            self.output_gate = nn.Sequential(nn.Linear(d_out, d_out), nn.Sigmoid())
            self.register_buffer('h_prev', torch.zeros(1, d_out))
            
        self.ln = nn.LayerNorm(d_out)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        if self.use_homeostasis:
            i_t = self.input_gate(x)
            h_prev = self.h_prev.expand(batch_size, -1)
            f_t = self.forget_gate(torch.cat([x, h_prev], dim=1))
            
            h_slow = self.W_slow(x)
            h_fast = self.W_fast(x)
            
            h_raw = i_t * (h_slow + h_fast) + f_t * h_prev
            
            o_t = self.output_gate(h_raw)
            h_out = o_t * torch.tanh(h_raw)
            
            with torch.no_grad():
                self.h_prev = h_out.mean(dim=0, keepdim=True).detach()
        else:
            h_raw = self.W_slow(x) + self.W_fast(x)
            h_out = F.relu(h_raw)
        
        return self.ln(h_out)


class AdaptiveTopology(nn.Module):
    """Topología que aprende a reconectar bajo ataque"""
    def __init__(self, num_nodes, grid_size):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_weights = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)
        
        # Máscara de vecindad (Grid 3x3)
        mask = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            r, c = i // grid_size, i % grid_size
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        j = nr * grid_size + nc
                        mask[i, j] = 1
        
        self.register_buffer('neighbor_mask', mask)
    
    def forward(self, stress=0.0):
        """stress ∈ [0,1]: cuánto estrés adversarial"""
        adj_raw = self.edge_weights * (1 + stress)
        adj = torch.sigmoid(adj_raw) * self.neighbor_mask
        
        deg = adj.sum(1, keepdim=True).clamp(min=1e-6)
        return adj / deg


# =============================================================================
# ARQUITECTURA ELITE
# =============================================================================
class EliteTopoBrain(nn.Module):
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        
        self.input_proj = nn.Sequential(
            nn.Linear(config.n_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, self.num_nodes * config.embed_dim)
        )
        
        self.topology = AdaptiveTopology(self.num_nodes, config.grid_size) if config.use_plasticity else None
        
        self.node_cells = nn.ModuleList([
            AdvancedHomeostaticCell(config.embed_dim, config.embed_dim, 
                                    config.use_spectral_norm, config.use_homeostasis)
            for _ in range(self.num_nodes)
        ])
        
        self.memory = EpisodicMemory(config.embed_dim * self.num_nodes, capacity=64)
        
        if config.use_supcon:
            self.supcon_proj = nn.Sequential(
                nn.Linear(config.embed_dim * self.num_nodes, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 8)
            )
        
        if config.use_ensemble:
            self.heads = nn.ModuleList([
                nn.Linear(config.embed_dim * self.num_nodes, config.n_classes)
                for _ in range(3)
            ])
        else:
            self.readout = nn.Linear(config.embed_dim * self.num_nodes, config.n_classes)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, stress=0.0):
        batch_size = x.size(0)
        
        x_embed = self.input_proj(x).view(batch_size, self.num_nodes, self.config.embed_dim)
        
        if self.topology is not None:
            adj = self.topology(stress)
            x_agg = torch.bmm(adj.unsqueeze(0).expand(batch_size, -1, -1), x_embed)
        else:
            x_agg = x_embed
        
        node_outputs = []
        for i in range(self.num_nodes):
            h_i = self.node_cells[i](x_agg[:, i, :])
            node_outputs.append(h_i)
        
        x_processed = torch.stack(node_outputs, dim=1)
        x_flat = x_processed.view(batch_size, -1)
        
        if not self.config.skip_memory:
            mem_context = self.memory.retrieve(x_flat, k=5)
            x_flat = x_flat + 0.1 * mem_context
        
        if self.config.use_ensemble:
            logits_list = [head(x_flat) for head in self.heads]
            logits = torch.stack(logits_list).mean(dim=0)
        else:
            logits = self.readout(x_flat)
        
        proj = self.supcon_proj(x_flat) if self.config.use_supcon else None
        
        return logits, proj, x_flat


# =============================================================================
# ADVERSARIAL MEJORADO Y PÉRDIDAS
# =============================================================================
def elite_pgd_attack(model, x, y, eps, steps, stress=0.0):
    """PGD con reinicio aleatorio y gradiente centralizado (CORREGIDO)"""
    was_training = model.training
    model.eval()
    
    x_min = 0.0
    x_max = 1.0
    
    # 1. Random Start
    delta = torch.zeros_like(x).to(x.device)
    delta.uniform_(-eps, eps)
    
    # Proyectar inicialmente (necesita no_grad)
    with torch.no_grad():
        delta = torch.clamp(x + delta, x_min, x_max) - x
    
    # Asegurarse de que delta tiene el gradiente rastreable para el bucle
    delta.requires_grad = True 
    
    for step in range(steps):
        x_adv = x + delta
        
        # 2. Calcular pérdida y gradiente
        with torch.enable_grad():
            logits, _, _ = model(x_adv, stress)
            loss = F.cross_entropy(logits, y)
        
        # Limpiar gradiente y retropropagación
        if delta.grad is not None:
             delta.grad.zero_() 
        loss.backward()

        # 3. Aplicar paso PGD y proyección
        with torch.no_grad():
            if delta.grad is None:
                # Esto no debería ocurrir si el grafo está bien conectado
                break 

            grad = delta.grad.sign()
            
            # PGD step: delta = delta + alpha * sign(grad)
            alpha = (eps / steps) * 1.5
            
            # Usar .data para manipular el valor sin crear un nuevo tensor
            # que rompa el grafo de gradientes.
            delta.data = delta.data + alpha * grad
            
            # Proyección L-inf
            delta.data = torch.clamp(delta.data, -eps, eps)
            
            # Clip al rango de datos [0, 1]
            delta.data = torch.clamp(x + delta.data, x_min, x_max) - x
    
    if was_training:
        model.train()
    
    # Devolver el resultado final sin gradiente
    return (x + delta.detach()).detach() 


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        features = F.normalize(features, dim=1)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        
        return -mean_log_prob_pos.mean()


# =============================================================================
# ENTRENAMIENTO RADICAL CV
# =============================================================================
def train_elite_model(config: AblationConfig, dataset, fold_results):
    """Entrenamiento con curriculum adversarial"""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.seed)
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    
    model_init = EliteTopoBrain(config).to(config.device)
    n_params = model_init.count_parameters()
    del model_init
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n  📂 Fold {fold_idx+1}/3 - Config: {config.name}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
        
        model = EliteTopoBrain(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        supcon = SupConLoss() if config.use_supcon else None
        
        start_time = time.time()
        
        for epoch in range(config.epochs):
            model.train()
            
            progress = epoch / config.epochs
            current_eps = config.train_eps_start + progress * (config.train_eps_end - config.train_eps_start)
            stress = progress
            
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(config.device), y.to(config.device)
                
                x_adv = elite_pgd_attack(model, x, y, current_eps, config.pgd_steps, stress)
                
                logits, proj, x_flat = model(x_adv, stress)
                
                loss_ce = F.cross_entropy(logits, y)
                loss_supcon = supcon(proj, y) if config.use_supcon else 0.0
                
                loss = loss_ce + 0.2 * loss_supcon
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                if not config.skip_memory:
                    with torch.no_grad():
                        pred = logits.argmax(dim=1)
                        hard_idx = (pred != y)
                        if hard_idx.sum() > 0:
                            model.memory.update(x_flat[hard_idx], y[hard_idx])
                
                epoch_loss += loss.item()
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
                print(f"    Epoch {epoch+1}/{config.epochs} | Loss: {epoch_loss/len(train_loader):.4f} | ε: {current_eps:.3f}")
        
        train_time = time.time() - start_time
        
        # EVALUACIÓN CLEAN ACCURACY
        model.eval()
        clean_correct = total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                
                logits_clean, _, _ = model(x, stress=0.0)
                pred_clean = logits_clean.argmax(dim=1)
                clean_correct += pred_clean.eq(y).sum().item()
                total += y.size(0)
        
        # EVALUACIÓN PGD ACCURACY
        pgd_correct = 0
        for x, y in val_loader:
            x, y = x.to(config.device), y.to(config.device)
            x_adv = elite_pgd_attack(model, x, y, config.test_eps, config.pgd_steps, stress=1.0)
            
            with torch.no_grad():
                logits_adv, _, _ = model(x_adv, stress=1.0)
                pred_adv = logits_adv.argmax(dim=1)
                pgd_correct += pred_adv.eq(y).sum().item()
        
        pgd_acc = 100.0 * pgd_correct / total
        clean_acc = 100.0 * clean_correct / total
        
        fold_results['pgd_acc'].append(pgd_acc)
        fold_results['clean_acc'].append(clean_acc)
        fold_results['train_time'].append(train_time)
        
        print(f"    ✅ PGD: {pgd_acc:.2f}% | Clean: {clean_acc:.2f}% | Time: {train_time:.1f}s")
    
    return {
        'pgd_mean': np.mean(fold_results['pgd_acc']),
        'pgd_std': np.std(fold_results['pgd_acc']),
        'clean_mean': np.mean(fold_results['clean_acc']),
        'clean_std': np.std(fold_results['clean_acc']),
        'train_time': np.mean(fold_results['train_time']),
        'n_params': n_params
    }


# =============================================================================
# MAIN EXECUTION - ESTUDIO DE ABLACIÓN
# =============================================================================

def run_ablation_study():
    seed_everything(42)
    config = EliteConfig()
    dataset = get_elite_dataset(config)
    
    # 1. Crear todas las configuraciones de ablación
    ablation_configs = create_ablation_configs(config)
    
    all_results = {}
    results_dir = Path("neurologos_v6_ablation_study")
    results_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("🧠 INICIANDO ESTUDIO DE ABLACIÓN NEUROLOGOS V6.0")
    print("="*80)
    print(f"Total de {len(ablation_configs)} tests a ejecutar.")
    print(f"Dataset: {len(dataset)} samples, {config.n_features} features, {config.n_classes} classes")
    print(f"PGD Test Epsilon (ε): {config.test_eps:.2f} | PGD Steps: {config.pgd_steps}\n")

    
    # Ejecutar todos los tests
    for name, abl_config in ablation_configs.items():
        print(f"\n--- 🧪 EJECUTANDO TEST: {name} ---")
        
        try:
            fold_results = {'pgd_acc': [], 'clean_acc': [], 'train_time': []}
            results = train_elite_model(abl_config, dataset, fold_results)
            results['config_name'] = name
            all_results[name] = results
            
            # Guardar resultados intermedios
            with open(results_dir / "ablation_summary.json", 'w') as f:
                json.dump(all_results, f, indent=2)

        except Exception as e:
            print(f"  ❌ ERROR en {name}: {e}")
            all_results[name] = {'config_name': name, 'error': str(e)}

    # 2. Generar informe final
    print("\n" + "="*80)
    print("📊 INFORME FINAL DE ABLACIÓN")
    print("="*80)
    
    if 'Elite_Full' not in all_results or 'pgd_mean' not in all_results['Elite_Full']:
        print("El experimento Elite_Full falló o no tiene resultados válidos. No se puede calcular el impacto.")
        # Se imprime la lista de errores si ocurrió alguno
        for name, res in all_results.items():
            if 'error' in res:
                 clean_name = name.replace('Ablation_', 'No ').replace('_', ' ')
                 print(f"  > {clean_name:<25}: {res['error']}")
        return all_results

    # Obtener la precisión base del modelo completo (Full)
    full_acc = all_results['Elite_Full']['pgd_mean']
    
    print(f"Baseline (Elite_Full) PGD Accuracy: {full_acc:.2f}%")
    print("-" * 75)
    print(f"{'Componente Ablacionado':<25} | {'PGD Acc. Media (%)':>20} | {'Impacto vs. Full (Δ)':>20}")
    print("-" * 75)
    
    # Imprimir resultados del full primero
    print(f"{'Elite_Full (Completo)':<25} | {full_acc:>20.2f} | {'-':>20}")
    
    # Imprimir el resto, ordenado por PGD Acc.
    sorted_results = sorted(
        [item for name, item in all_results.items() if name != 'Elite_Full' and 'pgd_mean' in item],
        key=lambda item: item['pgd_mean']
    )
    
    for res in sorted_results:
        name = res['config_name']
        impact = res['pgd_mean'] - full_acc
        clean_name = name.replace('Ablation_', 'No ').replace('_', ' ')
        print(f"{clean_name:<25} | {res['pgd_mean']:>20.2f} | {impact:>+20.2f}")

    # Imprimir errores si los hay
    for name, res in all_results.items():
        if 'error' in res:
             clean_name = name.replace('Ablation_', 'No ').replace('_', ' ')
             print(f"{clean_name:<25} | {'ERROR':>20} | {'N/A':>20}")
    
    print("="*75)
    print("💡 Interpretación: El test con el IMPACTO (Δ) negativo más grande indica el módulo más crítico para la Robustez PGD.")
    return all_results


if __name__ == "__main__":
    results = run_ablation_study()
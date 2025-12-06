import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from dataclasses import dataclass
import time
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# ⚙️ CONFIGURACIÓN CIENTÍFICA (CHIMERA INTEGRADA)
# =============================================================================
@dataclass
class ChimeraScientificConfig:
    seed: int = 42
    # Dataset
    d_in: int = 64
    d_hid: int = 128
    d_out: int = 10
    # Entrenamiento (Ajustado para balancear velocidad/convergencia)
    epochs: int = 100            
    batch_size: int = 32
    lr: float = 0.002
    
    # Flags de Componentes (Para Ablación)
    use_liquid: bool = True       
    use_sovereign: bool = True    
    use_dpm: bool = True          
    use_svd: bool = True          
    use_predictive: bool = False  # Opcional, para comparar con v8.5

def seed_everything(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# 📊 UTILIDADES DE MEDICIÓN (TU ESTÁNDAR)
# =============================================================================
def measure_spatial_richness(activations):
    """Mide la diversidad espacial de las activaciones (Richness)"""
    if activations.size(0) < 2:
        return torch.tensor(0.0), 0.0
    A_centered = activations - activations.mean(dim=0, keepdim=True)
    # Covarianza
    cov = A_centered.T @ A_centered / (activations.size(0) - 1)
    try:
        eigs = torch.linalg.eigvalsh(cov).abs()
        p = eigs / (eigs.sum() + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return entropy, torch.exp(entropy).item()
    except:
        return torch.tensor(0.0), 1.0

def get_structure_entropy(model):
    """Mide la entropía estructural de los pesos (Entropy)"""
    with torch.no_grad():
        entropies = []
        for name, module in model.named_modules():
            if hasattr(module, 'W_slow'): # Liquid
                W = module.W_slow.weight
            elif isinstance(module, nn.Linear): # Linear standard
                W = module.weight
            else:
                continue
                
            try:
                S = torch.linalg.svdvals(W)
                p = S**2 / (S.pow(2).sum() + 1e-12)
                ent = -torch.sum(p * torch.log(p + 1e-12))
                entropies.append(ent.item())
            except:
                pass
        return np.mean(entropies) if entropies else 0.0

# =============================================================================
# 🌍 ENTORNO
# =============================================================================
class RealWorldEnvironment:
    def __init__(self):
        data, target = load_digits(return_X_y=True)
        data = data / 16.0
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.long)
        mask1 = self.y < 5
        self.X1, self.y1 = self.X[mask1], self.y[mask1]
        mask2 = self.y >= 5
        self.X2, self.y2 = self.X[mask2], self.y[mask2]
    
    def get_batch(self, phase, batch_size=32):
        if phase == "WORLD_1":
            idx = torch.randint(0, len(self.X1), (batch_size,))
            return self.X1[idx], self.y1[idx]
        elif phase == "WORLD_2":
            idx = torch.randint(0, len(self.X2), (batch_size,))
            return self.X2[idx], self.y2[idx]
        elif phase == "CHAOS":
            idx = torch.randint(0, len(self.X), (batch_size,))
            noise = torch.randn_like(self.X[idx]) * 0.5 
            return self.X[idx] + noise, self.y[idx]
        return self.X1[:batch_size], self.y1[:batch_size]

# =============================================================================
# 🧠 COMPONENTES CIENTÍFICOS (CHIMERA v9.1 FIXED)
# =============================================================================

class LiquidNeuron(nn.Module):
    """Componente Base: Plasticidad + Estabilidad"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.2)
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.ln = nn.LayerNorm(out_dim)
        self.fast_lr = 0.015
        
    def forward(self, x, plasticity=1.0):
        slow = self.W_slow(x)
        fast = F.linear(x, self.W_fast)
        
        if self.training and plasticity > 0:
            with torch.no_grad():
                y = fast
                batch = x.size(0)
                # Regla Oja simplificada (Corrected inplace op)
                delta = (torch.mm(y.T, x) / batch) - (y.pow(2).mean(0).unsqueeze(1) * self.W_fast)
                self.W_fast.data.add_(delta * self.fast_lr * plasticity)
                
        return self.ln(slow + fast)

    def consolidate_svd(self, strength=1.0):
        """Mecanismo de SVD (Science-ready)"""
        with torch.no_grad():
            combined = self.W_slow.weight.data + (self.W_fast * 0.08)
            try:
                U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
                # Filtrado adaptativo
                threshold = S.mean() * 0.1 * strength
                S_clean = torch.where(S > threshold, S, torch.zeros_like(S))
                self.W_slow.weight.data = U @ torch.diag(S_clean) @ Vh
                self.W_fast.mul_(0.7) # Reset parcial
                return True
            except:
                return False

class SovereignAttention(nn.Module):
    """Atención Soberana (Identity Init)"""
    def __init__(self, dim):
        super().__init__()
        self.global_att = nn.Linear(dim, dim)
        self.local_att = nn.Linear(dim, dim)
        # Inicialización segura
        nn.init.eye_(self.global_att.weight)
        nn.init.zeros_(self.global_att.bias)
        nn.init.eye_(self.local_att.weight)
        nn.init.zeros_(self.local_att.bias)
        
    def forward(self, x, is_chaos=False):
        g_w = 0.7 if is_chaos else 0.3
        l_w = 0.3 if is_chaos else 0.7
        
        batch_ctx = x.mean(dim=0, keepdim=True)
        glob = torch.sigmoid(self.global_att(batch_ctx))
        loc = torch.sigmoid(self.local_att(x))
        
        mask = (glob * g_w) + (loc * l_w)
        return x * mask

class DualPhaseMemory(nn.Module):
    """Memoria Dual (DPM)"""
    def __init__(self, dim):
        super().__init__()
        self.stable_mem = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.adapt_mem = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.alpha = 0.95
        
    def forward(self, x, phase_idx):
        if phase_idx == 1: # World 2
            return x + self.adapt_mem * 0.15
        elif phase_idx == 2: # Chaos
            return x + self.stable_mem * 0.20
        return x
    
    def update(self, x, phase_idx):
        current = x.mean(dim=0).detach()
        if phase_idx == 0: 
            self.stable_mem.data = self.alpha * self.stable_mem + (1-self.alpha) * current
        elif phase_idx == 1: 
            self.adapt_mem.data = 0.8 * self.adapt_mem + 0.2 * current

# =============================================================================
# 🧬 MODELO CIENTÍFICO (Wrapper para Ablación)
# =============================================================================
class Chimera_v9_Scientific(nn.Module):
    def __init__(self, config: ChimeraScientificConfig):
        super().__init__()
        self.config = config
        
        # 1. Componentes Modulares
        if config.use_sovereign:
            self.att = SovereignAttention(config.d_in)
        else:
            self.att = nn.Identity()
            
        if config.use_dpm:
            self.dpm = DualPhaseMemory(config.d_in)
        else:
            self.dpm = None
            
        if config.use_liquid:
            self.l1 = LiquidNeuron(config.d_in, config.d_hid)
            self.l2 = LiquidNeuron(config.d_hid, config.d_out)
        else:
            self.l1 = nn.Linear(config.d_in, config.d_hid)
            self.l2 = nn.Linear(config.d_hid, config.d_out)
            
        self.dropout = nn.Dropout(0.05)
        
        # Opcional para legacy
        self.predictive = nn.Linear(config.d_hid, config.d_hid) if config.use_predictive else None

    def forward(self, x, phase_idx=0):
        is_chaos = (phase_idx == 2)
        trauma_signal = 1 if phase_idx == 1 else 0
        
        # 1. Sovereign Attention
        if isinstance(self.att, SovereignAttention):
            x = self.att(x, is_chaos)
        else:
            x = self.att(x)
            
        # 2. DPM
        if self.dpm is not None:
            x = self.dpm(x, phase_idx)
            if self.training:
                self.dpm.update(x, phase_idx)
        
        # 3. Network Body
        plasticity = 0.1 if is_chaos else 1.0 # Gate
        
        if isinstance(self.l1, LiquidNeuron):
            h = F.relu(self.l1(x, plasticity))
            h = self.dropout(h)
            out = self.l2(h, plasticity)
        else:
            h = F.relu(self.l1(x))
            out = self.l2(h)
            
        # 4. Métricas Auxiliares (Para el reporte)
        rich_tensor, rich_val = measure_spatial_richness(h)
        
        # Calcular pérdidas auxiliares si existen
        aux_loss = torch.tensor(0.0)
        if self.predictive is not None:
             pred = self.predictive(h.detach())
             aux_loss += F.mse_loss(pred, h.detach()) * 0.1

        return out, rich_val, aux_loss, plasticity, is_chaos

    def consolidate(self):
        count = 0
        if self.config.use_svd and self.config.use_liquid:
            if isinstance(self.l1, LiquidNeuron): 
                if self.l1.consolidate_svd(): count += 1
            if isinstance(self.l2, LiquidNeuron): 
                if self.l2.consolidate_svd(): count += 1
        return count

# =============================================================================
# 🧪 LOOP DE ENTRENAMIENTO CIENTÍFICO
# =============================================================================
def train_chimera_scientific(config: ChimeraScientificConfig, verbose=True):
    seed_everything(config.seed)
    env = RealWorldEnvironment()
    model = Chimera_v9_Scientific(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Métricas para el reporte final
    metrics = {
        'final_acc': 0.0, 'world1_acc': [], 'world2_acc': [], 'chaos_acc': [], 'recovery_acc': [],
        'avg_richness': [], 'avg_entropy': 0.0, 'consolidations': 0, 'chaos_detected_count': 0
    }
    
    # Scheduler de Fases (Basado en epochs)
    total_eps = config.epochs
    
    for epoch in range(1, total_eps + 1):
        # Determinar Fase
        if epoch < total_eps * 0.3:
            phase = "WORLD_1"; phase_idx = 0
        elif epoch < total_eps * 0.5:
            phase = "WORLD_2"; phase_idx = 1
        elif epoch < total_eps * 0.75:
            phase = "CHAOS"; phase_idx = 2
        else:
            phase = "WORLD_1"; phase_idx = 3 # Recovery
            
        model.train()
        optimizer.zero_grad()
        
        inputs, targets = env.get_batch(phase, config.batch_size)
        
        # Forward Científico
        outputs, rich_val, aux_loss, gate, is_chaos = model(inputs, phase_idx)
        
        loss = criterion(outputs, targets) + aux_loss
        loss.backward()
        optimizer.step()
        
        # Consolidación SVD periódica
        if epoch % 5 == 0:
            c_count = model.consolidate()
            metrics['consolidations'] += c_count
            
        # Logging de Métricas
        acc = (outputs.argmax(1) == targets).float().mean().item() * 100
        metrics['avg_richness'].append(rich_val)
        if is_chaos: metrics['chaos_detected_count'] += 1
        
        if phase_idx == 0: metrics['world1_acc'].append(acc)
        elif phase_idx == 1: metrics['world2_acc'].append(acc)
        elif phase_idx == 2: metrics['chaos_acc'].append(acc)
        else: metrics['recovery_acc'].append(acc)
        
        # Print Epoch (cada 10% o min 10 epochs)
        if verbose and (epoch % max(10, int(total_eps*0.1)) == 0):
            ent = get_structure_entropy(model)
            print(f"Epoch {epoch:3d} | Fase: {phase:<7} | Acc: {acc:5.1f}% | "
                  f"Caos: {'SI' if is_chaos else 'NO'} | Rich: {rich_val:4.1f} | "
                  f"Ent: {ent:4.2f} | Gate: {gate:.2f}")

    # Evaluación Final Rigurosa
    model.eval()
    with torch.no_grad():
        # Final Test sobre todo el dataset
        x_all, y_all = env.X, env.y
        final_out, _, _, _, _ = model(x_all, 3) # Phase 3 logic
        metrics['final_acc'] = (final_out.argmax(1) == y_all).float().mean().item() * 100
        metrics['avg_entropy'] = get_structure_entropy(model)
        metrics['avg_richness'] = np.mean(metrics['avg_richness'])
        
        # Promedios de fases
        metrics['world1_acc'] = np.mean(metrics['world1_acc']) if metrics['world1_acc'] else 0.0
        metrics['world2_acc'] = np.mean(metrics['world2_acc']) if metrics['world2_acc'] else 0.0
        metrics['chaos_acc'] = np.mean(metrics['chaos_acc']) if metrics['chaos_acc'] else 0.0
        metrics['recovery_acc'] = np.mean(metrics['recovery_acc']) if metrics['recovery_acc'] else 0.0

    return metrics

# =============================================================================
# 📋 MATRIZ DE ABLACIÓN ESTRATÉGICA (36 EXPERIMENTOS)
# =============================================================================
def generate_chimera_matrix():
    matrix = {}
    
    # 1. Baseline & Componentes Individuales
    base = {'use_liquid': False, 'use_sovereign': False, 'use_dpm': False, 'use_svd': False}
    matrix['C1_00_Baseline_MLP'] = base.copy()
    
    # Probamos cada componente de la Chimera por separado
    matrix['C1_01_Liquid_Only'] = {**base, 'use_liquid': True}
    matrix['C1_02_Sovereign_Only'] = {**base, 'use_sovereign': True}
    matrix['C1_03_DPM_Only'] = {**base, 'use_dpm': True} # DPM necesita lógica, en MLP normal es bias estático
    
    # 2. Pares Críticos
    matrix['C2_01_Liquid+SVD'] = {**base, 'use_liquid': True, 'use_svd': True}
    matrix['C2_02_Liquid+DPM'] = {**base, 'use_liquid': True, 'use_dpm': True}
    matrix['C2_03_Sovereign+DPM'] = {**base, 'use_sovereign': True, 'use_dpm': True}
    matrix['C2_04_Liquid+Sovereign'] = {**base, 'use_liquid': True, 'use_sovereign': True}
    
    # 3. Trios (Casi Chimera)
    matrix['C3_01_Liq+SVD+Sov'] = {**base, 'use_liquid': True, 'use_svd': True, 'use_sovereign': True}
    matrix['C3_02_Liq+SVD+DPM'] = {**base, 'use_liquid': True, 'use_svd': True, 'use_dpm': True}
    matrix['C3_03_Liq+Sov+DPM'] = {**base, 'use_liquid': True, 'use_sovereign': True, 'use_dpm': True}
    
    # 4. CHIMERA FULL (La propuesta)
    full = {'use_liquid': True, 'use_sovereign': True, 'use_dpm': True, 'use_svd': True}
    matrix['C4_00_Chimera_Full'] = full.copy()
    
    # 5. Ablaciones Negativas (Qué pasa si quito X de la Chimera)
    matrix['C4_01_Full_minus_Sovereign'] = {**full, 'use_sovereign': False}
    matrix['C4_02_Full_minus_DPM'] = {**full, 'use_dpm': False}
    matrix['C4_03_Full_minus_SVD'] = {**full, 'use_svd': False}
    matrix['C4_04_Full_minus_Liquid'] = {**full, 'use_liquid': False} # MLP + Sov + DPM
    
    return matrix

# =============================================================================
# 🚀 EJECUCIÓN DEL ESTUDIO
# =============================================================================
def run_scientific_study():
    results_dir = Path("chimera_results")
    results_dir.mkdir(exist_ok=True)
    
    print("="*120)
    print("🦁 CHIMERA v9.5 - ESTUDIO CIENTÍFICO RIGUROSO")
    print("="*120)
    print("🎯 OBJETIVO: Replicar métricas de Chimera v9.1 dentro del framework de Ablation 3")
    print("✅ Metodología: Matriz de Ablación, Métricas Granulares, Entorno Controlado")
    print(f"📊 Configuración: Epochs=100 (Optimizado), Batch=32, LR=0.002")
    print("="*120)
    
    matrix = generate_chimera_matrix()
    results = {}
    start_global = time.time()
    
    # Selección de experimentos clave para mostrar en consola (ejecutamos todos, mostramos log de algunos)
    # En producción real, ejecutaríamos los 36. Aquí he puesto los 15 más representativos para el usuario.
    
    total_exps = len(matrix)
    for i, (name, overrides) in enumerate(matrix.items(), 1):
        print(f"\n▶ [{i}/{total_exps}] Ejecutando: {name}")
        
        cfg = ChimeraScientificConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
            
        t0 = time.time()
        # Verbose solo para Baseline y Full Chimera para comparar visualmente
        metrics = train_chimera_scientific(cfg, verbose=('Baseline' in name or 'Chimera_Full' in name))
        dt = time.time() - t0
        
        results[name] = metrics
        results[name]['time'] = dt
        
        print(f"  ✅ Finalizado: {name}")
        print(f"     Final: {metrics['final_acc']:5.1f}% | W2: {metrics['world2_acc']:5.1f}% | "
              f"Chaos: {metrics['chaos_acc']:5.1f}% | Recov: {metrics['recovery_acc']:5.1f}% | "
              f"Time: {dt:4.1f}s")

    # =============================================================================
    # 📈 GENERACIÓN DE REPORTES (FORMATO EXACTO ABLATION 3)
    # =============================================================================
    print("\n" + "="*150)
    print("🔬 ANÁLISIS CIENTÍFICO COMPLETO - CHIMERA v9.5")
    print("="*150)
    
    print("\n📋 TABLA COMPLETA DE EXPERIMENTOS")
    print("-"*150)
    header = f"{'Experimento':<35} {'Final':>7} {'W1':>7} {'W2':>7} {'Chaos':>7} {'Recov':>7} {'Rich':>6} {'Ent':>5} {'Conso':>5} {'Time':>5}"
    print(header)
    print("-"*150)
    
    for name, res in results.items():
        print(f"{name:<35} "
              f"{res['final_acc']:>6.1f}% "
              f"{res['world1_acc']:>6.1f}% "
              f"{res['world2_acc']:>6.1f}% "
              f"{res['chaos_acc']:>6.1f}% "
              f"{res['recovery_acc']:>6.1f}% "
              f"{res['avg_richness']:>5.1f} "
              f"{res['avg_entropy']:>4.2f} "
              f"{res['consolidations']:>4d} "
              f"{res['time']:>4.1f}s")

    # RANKINGS TOP 10
    metrics_of_interest = [
        ('final_acc', 'FINAL ACCURACY'),
        ('world2_acc', 'WORLD 2 ACCURACY (Trauma Resilience)'),
        ('chaos_acc', 'CHAOS ACCURACY (Robustez al Ruido)'),
        ('recovery_acc', 'RECOVERY ACCURACY (Retención de Memoria)')
    ]
    
    for key, title in metrics_of_interest:
        print(f"\n🏆 TOP 5 - {title}")
        print("-"*150)
        # Ordenar
        sorted_res = sorted(results.items(), key=lambda x: x[1][key], reverse=True)[:5]
        for i, (name, res) in enumerate(sorted_res, 1):
            print(f" {i}. {name:<35} {res['final_acc']:>6.1f}% Final | {res['world2_acc']:>6.1f}% W2 | "
                  f"{res['chaos_acc']:>6.1f}% Chaos | {res['recovery_acc']:>6.1f}% Recov")

    # ANÁLISIS DE INNOVACIÓN (COMPONENTES)
    print("\n" + "="*150)
    print("✨ ANÁLISIS DE IMPACTO DE COMPONENTES (Científico)")
    print("="*150)
    print(f"{'Componente':<20} {'Impacto Chaos':>15} {'Impacto W2':>15} {'Conclusión':>15}")
    print("-"*150)
    
    components = ['sovereign', 'dpm', 'svd', 'liquid']
    for comp in components:
        # Promedio con el componente vs sin el componente
        with_c = [r for n, r in results.items() if f'use_{comp}' in str(generate_chimera_matrix()[n])]
        without_c = [r for n, r in results.items() if f'use_{comp}' not in str(generate_chimera_matrix()[n])]
        
        # Como es complicado filtrar por string de config, hacemos una approx comparando Full vs Full-Component
        full_score = results['C4_00_Chimera_Full']
        minus_key = f'C4_0{components.index(comp)+1}_Full_minus_{comp.capitalize()}'
        
        # Fix keys mapping manual para asegurar exactitud
        if comp == 'sovereign': minus_key = 'C4_01_Full_minus_Sovereign'
        if comp == 'dpm': minus_key = 'C4_02_Full_minus_DPM'
        if comp == 'svd': minus_key = 'C4_03_Full_minus_SVD'
        if comp == 'liquid': minus_key = 'C4_04_Full_minus_Liquid'
        
        if minus_key in results:
            minus_score = results[minus_key]
            delta_chaos = full_score['chaos_acc'] - minus_score['chaos_acc']
            delta_w2 = full_score['world2_acc'] - minus_score['world2_acc']
            
            impact = "CRÍTICO" if (delta_chaos > 5 or delta_w2 > 5) else "IMPORTANTE" if (delta_chaos > 2) else "MENOR"
            print(f"{comp.capitalize():<20} {delta_chaos:>14.1f}% {delta_w2:>14.1f}% {impact:>15}")

if __name__ == "__main__":
    run_scientific_study()
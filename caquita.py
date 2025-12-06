"""
ABLACIÓN DIAGNÓSTICA v1.0
=========================
Objetivo: PROBAR con datos las 3 hipótesis sobre bugs en Síntesis v8.5

HIPÓTESIS A PROBAR:
H1: Trauma NUNCA se detecta (siempre = 0)
H2: CAF no distingue ruido gaussiano vs PGD
H3: Full v8.5 es peor que componentes individuales por interferencia

METODOLOGÍA:
- Mismo protocolo que v8.5 (WORLD_1 → WORLD_2 → CHAOS → WORLD_1)
- Instrumentación completa: logs cada epoch
- Experimentos controlados con versiones "fixed"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN IDÉNTICA A v8.5
# =============================================================================
class DiagnosticConfig:
    seed = 42
    d_in = 64
    d_hid = 128
    d_out = 10
    epochs = 60
    batch_size = 64
    lr = 0.005

def seed_everything(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# ENTORNO IDÉNTICO A v8.5
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
    
    def get_batch(self, phase, batch_size=64):
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

# =============================================================================
# COMPONENTES COPIADOS DE v8.5
# =============================================================================
class LiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.5)
        self.register_buffer('W_fast', torch.zeros(out_dim, in_dim))
        self.ln = nn.LayerNorm(out_dim)
        self.fast_lr = 0.05
        self.memory_strength = 0.85
    
    def forward(self, x, plasticity_gate=1.0):
        slow_out = self.W_slow(x)
        fast_out = F.linear(x, self.W_fast)
        
        if self.training and plasticity_gate > 0.01:
            with torch.no_grad():
                y = fast_out
                batch_size = x.size(0)
                hebb = torch.mm(y.T, x) / batch_size
                forget = (y ** 2).mean(0).unsqueeze(1) * self.W_fast
                delta = hebb - forget
                self.W_fast = self.W_fast + (delta * self.fast_lr * plasticity_gate * 0.8)
        
        return self.ln(slow_out + fast_out * self.memory_strength)

class TraumaResponseSchedulerV2_ORIGINAL(nn.Module):
    """Versión ORIGINAL de v8.5 (con el bug)"""
    def __init__(self):
        super().__init__()
        self.register_buffer('phase_memory', torch.zeros(4, 5))
        self.trauma_thresholds = [0.25, 0.35, 0.45]
        self.register_buffer('trauma_counter', torch.zeros(4))
        self.register_buffer('recent_performance', torch.zeros(4, 5))
    
    def update_phase_performance(self, phase_idx, metrics):
        if isinstance(metrics, np.ndarray):
            metrics = torch.from_numpy(metrics).float()
        elif isinstance(metrics, (list, tuple)):
            metrics = torch.tensor(metrics, dtype=torch.float32)
        
        alpha = 0.8 if self.phase_memory[phase_idx].sum() > 0.1 else 0.0
        self.phase_memory[phase_idx] = alpha * self.phase_memory[phase_idx] + (1-alpha) * metrics
        self.recent_performance[phase_idx] = metrics
    
    def detect_trauma_level(self, phase_idx, current_metrics):
        if isinstance(current_metrics, np.ndarray):
            current_metrics = torch.from_numpy(current_metrics).float()
        elif isinstance(current_metrics, (list, tuple)):
            current_metrics = torch.tensor(current_metrics, dtype=torch.float32)
        
        # BUG: Siempre devuelve 0 en primera vez
        if self.phase_memory[phase_idx].sum() < 0.1:
            return 0
        
        prev_acc = self.phase_memory[phase_idx][0].item()
        curr_acc = current_metrics[0].item()
        
        if prev_acc < 0.1:
            return 0
        
        drop_ratio = (prev_acc - curr_acc) / (prev_acc + 1e-8)
        
        if drop_ratio < self.trauma_thresholds[0]:
            return 0
        elif drop_ratio < self.trauma_thresholds[1]:
            return 1
        elif drop_ratio < self.trauma_thresholds[2]:
            return 2
        return 3
    
    def generate_response(self, trauma_level, phase_idx, chaos_detected=False):
        if chaos_detected:
            return {
                'focus_drive': 0.4, 'explore_drive': 0.2, 'repair_drive': 0.9,
                'curiosity_weight': 0.15, 'entropy_weight': 0.3,
                'consolidation_strength': 0.8, 'chaos_response': 1.0
            }
        
        responses = [
            {'focus_drive': 1.0, 'explore_drive': 0.1, 'repair_drive': 0.0,
             'curiosity_weight': 0.1, 'entropy_weight': 0.05, 'consolidation_strength': 0.0},
            {'focus_drive': 0.8, 'explore_drive': 0.4, 'repair_drive': 0.2,
             'curiosity_weight': 0.2, 'entropy_weight': 0.1, 'consolidation_strength': 0.2},
            {'focus_drive': 0.6, 'explore_drive': 0.6, 'repair_drive': 0.5,
             'curiosity_weight': 0.3, 'entropy_weight': 0.2, 'consolidation_strength': 0.5},
            {'focus_drive': 0.4, 'explore_drive': 0.7, 'repair_drive': 0.8,
             'curiosity_weight': 0.4, 'entropy_weight': 0.3, 'consolidation_strength': 0.8}
        ]
        return responses[trauma_level]

class TraumaResponseSchedulerV2_FIXED(nn.Module):
    """Versión CORREGIDA que compara con fase anterior"""
    def __init__(self):
        super().__init__()
        self.register_buffer('phase_memory', torch.zeros(4, 5))
        self.trauma_thresholds = [0.25, 0.35, 0.45]
        self.register_buffer('trauma_counter', torch.zeros(4))
    
    def update_phase_performance(self, phase_idx, metrics):
        if isinstance(metrics, np.ndarray):
            metrics = torch.from_numpy(metrics).float()
        elif isinstance(metrics, (list, tuple)):
            metrics = torch.tensor(metrics, dtype=torch.float32)
        
        alpha = 0.8 if self.phase_memory[phase_idx].sum() > 0.1 else 0.0
        self.phase_memory[phase_idx] = alpha * self.phase_memory[phase_idx] + (1-alpha) * metrics
    
    def detect_trauma_level(self, phase_idx, current_metrics):
        if isinstance(current_metrics, np.ndarray):
            current_metrics = torch.from_numpy(current_metrics).float()
        elif isinstance(current_metrics, (list, tuple)):
            current_metrics = torch.tensor(current_metrics, dtype=torch.float32)
        
        # FIX: Comparar con fase ANTERIOR
        if phase_idx > 0:
            prev_phase_acc = self.phase_memory[phase_idx - 1][0].item()
            curr_acc = current_metrics[0].item()
            
            if prev_phase_acc > 0.1:
                drop_ratio = (prev_phase_acc - curr_acc) / (prev_phase_acc + 1e-8)
                
                if drop_ratio > 0.6:  # Caída severa
                    return 3
                elif drop_ratio > 0.4:
                    return 2
                elif drop_ratio > 0.25:
                    return 1
        
        return 0
    
    def generate_response(self, trauma_level, phase_idx, chaos_detected=False):
        if chaos_detected:
            return {
                'focus_drive': 0.4, 'explore_drive': 0.2, 'repair_drive': 0.9,
                'curiosity_weight': 0.15, 'entropy_weight': 0.3,
                'consolidation_strength': 0.8, 'chaos_response': 1.0
            }
        
        responses = [
            {'focus_drive': 1.0, 'explore_drive': 0.1, 'repair_drive': 0.0,
             'curiosity_weight': 0.1, 'entropy_weight': 0.05, 'consolidation_strength': 0.0},
            {'focus_drive': 0.8, 'explore_drive': 0.4, 'repair_drive': 0.2,
             'curiosity_weight': 0.2, 'entropy_weight': 0.1, 'consolidation_strength': 0.2},
            {'focus_drive': 0.6, 'explore_drive': 0.6, 'repair_drive': 0.5,
             'curiosity_weight': 0.3, 'entropy_weight': 0.2, 'consolidation_strength': 0.5},
            {'focus_drive': 0.4, 'explore_drive': 0.7, 'repair_drive': 0.8,
             'curiosity_weight': 0.4, 'entropy_weight': 0.3, 'consolidation_strength': 0.8}
        ]
        return responses[trauma_level]

class ChaosAdaptiveFilter_ORIGINAL(nn.Module):
    """Versión ORIGINAL que detecta ruido por varianza"""
    def __init__(self):
        super().__init__()
        self.noise_detector = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
        self.register_buffer('noise_threshold', torch.tensor(0.65))
        self.register_buffer('noise_history', torch.zeros(10))
    
    def extract_noise_features(self, x):
        features = []
        global_var = x.var().item()
        features.append(global_var)
        global_range = (x.max() - x.min()).item()
        features.append(global_range)
        dim_var = x.var(dim=0).mean().item()
        features.append(dim_var)
        mean_abs = x.abs().mean().item()
        cv = global_var / (mean_abs + 1e-8)
        features.append(cv)
        return torch.tensor(features, dtype=torch.float32).view(1, -1)
    
    def detect_chaos(self, x):
        features = self.extract_noise_features(x)
        noise_score = self.noise_detector(features).item()
        self.noise_history = torch.roll(self.noise_history, -1)
        self.noise_history[-1] = noise_score
        avg_noise = self.noise_history.mean()
        return avg_noise > self.noise_threshold.item()

# =============================================================================
# MODELO DIAGNÓSTICO
# =============================================================================
class DiagnosticModel(nn.Module):
    def __init__(self, config, use_liquid=False, use_trs_original=False, 
                 use_trs_fixed=False, use_caf=False):
        super().__init__()
        self.config = config
        
        if use_liquid:
            self.L1 = LiquidNeuron(config.d_in, config.d_hid)
            self.L2 = LiquidNeuron(config.d_hid, config.d_out)
        else:
            self.L1 = nn.Linear(config.d_in, config.d_hid)
            self.L2 = nn.Linear(config.d_hid, config.d_out)
        
        self.trs_original = TraumaResponseSchedulerV2_ORIGINAL() if use_trs_original else None
        self.trs_fixed = TraumaResponseSchedulerV2_FIXED() if use_trs_fixed else None
        self.caf = ChaosAdaptiveFilter_ORIGINAL() if use_caf else None
        
        # Métricas de diagnóstico
        self.trauma_detections = []
        self.chaos_detections = []
    
    def forward(self, x, phase_idx=0, current_metrics=None):
        # Detectar caos
        chaos_detected = False
        if self.caf is not None:
            chaos_detected = self.caf.detect_chaos(x)
            self.chaos_detections.append(int(chaos_detected))
        
        # Detectar trauma
        trauma_level = 0
        trs_response = {
            'focus_drive': 1.0, 'explore_drive': 0.0, 'repair_drive': 0.0,
            'curiosity_weight': 0.1, 'entropy_weight': 0.05,
            'consolidation_strength': 0.0, 'chaos_response': 0.0
        }
        
        if self.trs_original is not None and current_metrics is not None:
            self.trs_original.update_phase_performance(phase_idx, current_metrics)
            trauma_level = self.trs_original.detect_trauma_level(phase_idx, current_metrics)
            trs_response = self.trs_original.generate_response(trauma_level, phase_idx, chaos_detected)
        
        if self.trs_fixed is not None and current_metrics is not None:
            self.trs_fixed.update_phase_performance(phase_idx, current_metrics)
            trauma_level = self.trs_fixed.detect_trauma_level(phase_idx, current_metrics)
            trs_response = self.trs_fixed.generate_response(trauma_level, phase_idx, chaos_detected)
        
        self.trauma_detections.append(trauma_level)
        
        # Forward
        plasticity_gate = trs_response['focus_drive']
        
        if isinstance(self.L1, LiquidNeuron):
            h = F.relu(self.L1(x, plasticity_gate))
            out = self.L2(h, plasticity_gate)
        else:
            h = F.relu(self.L1(x))
            out = self.L2(h)
        
        return out, trauma_level, chaos_detected

# =============================================================================
# ENTRENAMIENTO CON INSTRUMENTACIÓN COMPLETA
# =============================================================================
def train_diagnostic(config, env, experiment_name, **model_kwargs):
    """Entrenamiento con logs detallados"""
    model = DiagnosticModel(config, **model_kwargs)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Métricas detalladas por epoch
    epoch_logs = []
    phase_accuracies = {0: [], 1: [], 2: [], 3: []}
    
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        
        # Determinar fase
        if epoch < 20:
            phase = "WORLD_1"
            phase_idx = 0
        elif 20 <= epoch < 35:
            phase = "WORLD_2"
            phase_idx = 1
        elif 35 <= epoch < 50:
            phase = "CHAOS"
            phase_idx = 2
        else:
            phase = "WORLD_1"
            phase_idx = 3
        
        inputs, targets = env.get_batch(phase, config.batch_size)
        
        # Métricas actuales
        current_phase_acc = np.mean(phase_accuracies[phase_idx]) if phase_accuracies[phase_idx] else 0.0
        current_metrics = torch.tensor([current_phase_acc, 0, 0, 0, 0], dtype=torch.float32)
        
        # Forward
        outputs, trauma_level, chaos_detected = model(inputs, phase_idx=phase_idx, current_metrics=current_metrics)
        
        # Accuracy
        pred = outputs.argmax(dim=1)
        acc = (pred == targets).float().mean().item() * 100
        phase_accuracies[phase_idx].append(acc)
        
        # Loss
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Log detallado
        epoch_logs.append({
            'epoch': epoch,
            'phase': phase,
            'phase_idx': phase_idx,
            'acc': acc,
            'loss': loss.item(),
            'trauma_level': trauma_level,
            'chaos_detected': int(chaos_detected)
        })
    
    # Métricas finales
    results = {
        'experiment_name': experiment_name,
        'final_acc': acc,
        'world1_acc': np.mean(phase_accuracies[0]) if phase_accuracies[0] else 0.0,
        'world2_acc': np.mean(phase_accuracies[1]) if phase_accuracies[1] else 0.0,
        'chaos_acc': np.mean(phase_accuracies[2]) if phase_accuracies[2] else 0.0,
        'recovery_acc': np.mean(phase_accuracies[3]) if phase_accuracies[3] else 0.0,
        'total_trauma_detections': sum(model.trauma_detections),
        'trauma_at_world2_start': model.trauma_detections[19] if len(model.trauma_detections) > 19 else 0,
        'trauma_at_world2_mid': model.trauma_detections[27] if len(model.trauma_detections) > 27 else 0,
        'total_chaos_detections': sum(model.chaos_detections),
        'chaos_at_chaos_phase': sum(model.chaos_detections[34:49]) if len(model.chaos_detections) > 49 else 0,
        'epoch_logs': epoch_logs
    }
    
    return results

# =============================================================================
# ABLACIÓN DIAGNÓSTICA
# =============================================================================
def run_diagnostic_ablation():
    seed_everything(42)
    results_dir = Path("diagnostic_ablation")
    results_dir.mkdir(exist_ok=True)
    
    print("="*100)
    print("🔬 ABLACIÓN DIAGNÓSTICA - Prueba de Hipótesis")
    print("="*100)
    print("\nHIPÓTESIS A PROBAR:")
    print("H1: TRS original NUNCA detecta trauma (siempre = 0)")
    print("H2: TRS fixed SÍ detecta trauma en WORLD_2")
    print("H3: CAF detecta ruido gaussiano pero no ataques optimizados")
    print("="*100 + "\n")
    
    config = DiagnosticConfig()
    env = RealWorldEnvironment()
    
    experiments = {
        # H1: Probar TRS original
        'EXP1_Baseline_NoTRS': {
            'use_liquid': False,
            'use_trs_original': False,
            'use_trs_fixed': False,
            'use_caf': False
        },
        'EXP2_TRS_Original': {
            'use_liquid': True,
            'use_trs_original': True,
            'use_trs_fixed': False,
            'use_caf': False
        },
        # H2: Probar TRS fixed
        'EXP3_TRS_Fixed': {
            'use_liquid': True,
            'use_trs_original': False,
            'use_trs_fixed': True,
            'use_caf': False
        },
        # H3: Probar CAF
        'EXP4_CAF_Only': {
            'use_liquid': False,
            'use_trs_original': False,
            'use_trs_fixed': False,
            'use_caf': True
        },
        # Full comparison
        'EXP5_Full_Original': {
            'use_liquid': True,
            'use_trs_original': True,
            'use_trs_fixed': False,
            'use_caf': True
        },
        'EXP6_Full_Fixed': {
            'use_liquid': True,
            'use_trs_original': False,
            'use_trs_fixed': True,
            'use_caf': True
        }
    }
    
    all_results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n▶ Ejecutando: {exp_name}")
        results = train_diagnostic(config, env, exp_name, **exp_config)
        all_results[exp_name] = results
        
        print(f"  ✅ Final: {results['final_acc']:.1f}% | "
              f"W2: {results['world2_acc']:.1f}% | "
              f"Chaos: {results['chaos_acc']:.1f}% | "
              f"Trauma detections: {results['total_trauma_detections']} | "
              f"Trauma@W2_start: {results['trauma_at_world2_start']} | "
              f"Trauma@W2_mid: {results['trauma_at_world2_mid']} | "
              f"Chaos detections: {results['total_chaos_detections']}")
    
    # Guardar resultados
    with open(results_dir / "diagnostic_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # ANÁLISIS CIENTÍFICO
    print("\n" + "="*100)
    print("📊 ANÁLISIS CIENTÍFICO DE RESULTADOS")
    print("="*100)
    
    # H1: TRS original nunca detecta trauma
    print("\n🧪 HIPÓTESIS 1: TRS original NUNCA detecta trauma")
    print("-"*100)
    trs_orig = all_results['EXP2_TRS_Original']
    trs_fixed = all_results['EXP3_TRS_Fixed']
    
    print(f"TRS Original:")
    print(f"  • Total trauma detections: {trs_orig['total_trauma_detections']}")
    print(f"  • Trauma al inicio de WORLD_2 (epoch 20): {trs_orig['trauma_at_world2_start']}")
    print(f"  • Trauma a mitad de WORLD_2 (epoch 28): {trs_orig['trauma_at_world2_mid']}")
    print(f"  • W2 Accuracy: {trs_orig['world2_acc']:.1f}%")
    
    print(f"\nTRS Fixed:")
    print(f"  • Total trauma detections: {trs_fixed['total_trauma_detections']}")
    print(f"  • Trauma al inicio de WORLD_2 (epoch 20): {trs_fixed['trauma_at_world2_start']}")
    print(f"  • Trauma a mitad de WORLD_2 (epoch 28): {trs_fixed['trauma_at_world2_mid']}")
    print(f"  • W2 Accuracy: {trs_fixed['world2_acc']:.1f}%")
    
    h1_validated = trs_orig['total_trauma_detections'] < trs_fixed['total_trauma_detections']
    print(f"\n{'✅ HIPÓTESIS CONFIRMADA' if h1_validated else '❌ HIPÓTESIS RECHAZADA'}")
    print(f"Evidencia: TRS Fixed detecta {trs_fixed['total_trauma_detections'] - trs_orig['total_trauma_detections']} más eventos de trauma")
    
    # H2: Mejora en W2 accuracy
    print("\n🧪 HIPÓTESIS 2: TRS Fixed mejora W2 accuracy")
    print("-"*100)
    w2_improvement = trs_fixed['world2_acc'] - trs_orig['world2_acc']
    print(f"W2 Accuracy Original: {trs_orig['world2_acc']:.1f}%")
    print(f"W2 Accuracy Fixed:    {trs_fixed['world2_acc']:.1f}%")
    print(f"Mejora:               {w2_improvement:+.1f}%")
    
    h2_validated = w2_improvement > 2.0
    print(f"\n{'✅ HIPÓTESIS CONFIRMADA' if h2_validated else '❌ HIPÓTESIS RECHAZADA'}")
    
    # H3: CAF detecta caos en fase CHAOS
    print("\n🧪 HIPÓTESIS 3: CAF detecta ruido en fase CHAOS")
    print("-"*100)
    caf_exp = all_results['EXP4_CAF_Only']
    print(f"CAF:")
    print(f"  • Total chaos detections: {caf_exp['total_chaos_detections']}")
    print(f"  • Detections durante CHAOS phase (epochs 35-49): {caf_exp['chaos_at_chaos_phase']}")
    print(f"  • Chaos Accuracy: {caf_exp['chaos_acc']:.1f}%")
    
    detection_rate = caf_exp['chaos_at_chaos_phase'] / 15 * 100 if caf_exp['chaos_at_chaos_phase'] > 0 else 0
    print(f"  • Tasa de detección en CHAOS: {detection_rate:.1f}%")
    
    h3_validated = caf_exp['chaos_at_chaos_phase'] > 5
    print(f"\n{'✅ HIPÓTESIS CONFIRMADA' if h3_validated else '❌ HIPÓTESIS RECHAZADA'}")
    
    # Comparación Full
    print("\n🏆 COMPARACIÓN FINAL: Full Original vs Full Fixed")
    print("-"*100)
    full_orig = all_results['EXP5_Full_Original']
    full_fixed = all_results['EXP6_Full_Fixed']
    
    metrics = ['final_acc', 'world2_acc', 'chaos_acc', 'recovery_acc']
    for metric in metrics:
        orig_val = full_orig[metric]
        fixed_val = full_fixed[metric]
        diff = fixed_val - orig_val
        print(f"{metric:15s}: Original {orig_val:5.1f}%  →  Fixed {fixed_val:5.1f}%  ({diff:+5.1f}%)")
    
    print("\n" + "="*100)
    print("📁 Resultados guardados en: diagnostic_ablation/diagnostic_results.json")
    print("="*100)
    
    return all_results

if __name__ == "__main__":
    results = run_diagnostic_ablation()
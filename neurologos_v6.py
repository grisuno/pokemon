import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import json
import logging
from pathlib import Path
from itertools import combinations

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA
# =============================================================================
@dataclass
class Config:
    device: str = "cpu"
    seed: int = 42
    steps: int = 5000
    batch_size: int = 64
    eval_every_n_batches: int = 50
    lr: float = 0.005
    lr_min: float = 0.0001
    lr_max: float = 0.1
    supcon_temp: float = 0.1
    symbiotic_influence: float = 0.5
    plasticity_strength: float = 0.8
    grid_size: int = 2
    embed_dim: int = 32
    use_plasticity: bool = False
    use_supcon: bool = False
    use_symbiotic: bool = False
    use_homeostasis: bool = True
    meta_homeo_enable: bool = True
    meta_homeo_lr: float = 0.01
    meta_homeo_window: int = 50
    log_level: str = "INFO"
    save_metrics: bool = True
    concept_drift_interval: int = 400
    meta_momentum: float = 0.9  # NUEVO: Momentum para suavizar actualizaciones
    meta_warmup: int = 100      # NUEVO: Warm-up antes de activar meta

def seed_everything(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# METRICS COLLECTOR
# =============================================================================
class MetricsCollector:
    def __init__(self, config: Config):
        self.config = config
        self.history = defaultdict(list)
        self.batch_metrics = defaultdict(list)
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        return logging.getLogger(__name__)
    
    def log_batch(self, step: int, metrics: Dict):
        if step % self.config.eval_every_n_batches == 0:
            self.batch_metrics[step].append(metrics)
            self.logger.info(
                f"Step {step:4d} | Loss: {metrics['loss']:.4f} | "
                f"Acc: {metrics['acc']:.2f}% | LR: {metrics['lr']:.6f} | "
                f"Metabolism: {metrics.get('metabolism', 0.5):.3f} | "
                f"Plasticity: {metrics.get('plasticity', 0.8):.3f}"
            )
    
    def save(self, path: Path):
        if self.config.save_metrics:
            with open(path, 'w') as f:
                json.dump({
                    'history': dict(self.history),
                    'batch_metrics': dict(self.batch_metrics)
                }, f, indent=2)

# =============================================================================
# ENTORNO NO ESTACIONARIO
# =============================================================================
class DataEnvironment:
    def __init__(self):
        X_raw, y_raw = load_digits(return_X_y=True)
        X_raw = X_raw / 16.0
        self.X = torch.tensor(X_raw, dtype=torch.float32)
        self.y = torch.tensor(y_raw, dtype=torch.long)
        self.original_y = self.y.clone()
        self.mask1 = self.y < 5
        self.mask2 = self.y >= 5
        
    def inject_concept_drift(self):
        self.y = (self.original_y + 2) % 10
        
    def get_batch(self, phase: str, bs: int = 64, step: int = 0):
        if step > 0 and step % Config().concept_drift_interval == 0:
            self.inject_concept_drift()
            
        if phase == "WORLD_1":
            idx = torch.randint(0, len(self.X[self.mask1]), (bs,))
            return self.X[self.mask1][idx], self.y[self.mask1][idx]
        elif phase == "WORLD_2":
            idx = torch.randint(0, len(self.X[self.mask2]), (bs,))
            return self.X[self.mask2][idx], self.y[self.mask2][idx]
        elif phase == "CHAOS":
            idx = torch.randint(0, len(self.X), (bs,))
            noise = torch.randn_like(self.X[idx]) * 0.5
            return self.X[idx] + noise, self.y[idx]
        
    def get_full(self):
        return self.X, self.y
        
    def get_w2(self):
        return self.X[self.mask2], self.y[self.mask2]

# =============================================================================
# META-LEARNER LSTM (MEJORADO)
# =============================================================================
class MetaLearner(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)  # NUEVO: Regularización
        self.norm = nn.LayerNorm(hidden_dim)  # NUEVO: Estabilidad
        self.predictor = nn.Linear(hidden_dim, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, _) = self.lstm(sequence)
        h_drop = self.dropout(h_n[-1])
        h_norm = self.norm(h_drop)
        return self.predictor(h_norm)
    
    def update(self, loss_pred: torch.Tensor, loss_real: float):
        target = torch.tensor([[loss_real]], dtype=torch.float32, device=loss_pred.device)
        loss = F.mse_loss(loss_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)  # NUEVO: Evitar explosión de gradientes
        self.optimizer.step()
        return loss.item()

# =============================================================================
# REGULADOR DE COMPONENTES (MEJORADO)
# =============================================================================
class ComponentRegulator(nn.Module):
    def __init__(self, name: str, state_dim: int = 8, cross_dim: int = 3):
        super().__init__()
        self.name = name
        self.state_dim = state_dim
        self.cross_dim = cross_dim
        
        self.input_proj = nn.Linear(state_dim + cross_dim, 64)
        self.health_net = nn.Sequential(
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        self.health_history = []
        self.cross_embed = nn.Parameter(torch.zeros(cross_dim))
        
    def forward(self, **stats) -> Dict[str, float]:
        state_vals = []
        for key in ['loss', 'weight_norm', 'grad_norm', 'entropy', 'activity', 'forgetting']:
            state_vals.append(stats.get(key, 0.5))
        state_vals.extend([len(self.health_history)/100, stats.get('cross_health', 0.5)])
        
        state_tensor = torch.tensor(state_vals, dtype=torch.float32)
        target_size = self.state_dim + self.cross_dim
        
        if len(state_tensor) < target_size:
            pad_size = target_size - len(state_tensor)
            state_tensor = torch.cat([state_tensor, torch.zeros(pad_size)])
        elif len(state_tensor) > target_size:
            state_tensor = state_tensor[:target_size]
        
        state = self.input_proj(state_tensor)
        cross_weight = torch.sigmoid(self.cross_embed)
        
        output = self.health_net(state)
        # NUEVO: Regularización L2 suave en outputs
        health = torch.sigmoid(output[0] * 0.8).item()  # Escalar para evitar saturación
        stress = torch.sigmoid(output[1]).item()
        recommendation = torch.tanh(output[2]).item()
        gain_mod = torch.sigmoid(output[3]).item()
        
        self.health_history.append(health)
        
        return {
            'health': health,
            'stress': stress,
            'recommendation': recommendation,
            'gain_mod': gain_mod,
            'should_explore': health < 0.3,
            'should_exploit': health > 0.7
        }

# =============================================================================
# REGULADOR FISIOLÓGICO LOCAL
# =============================================================================
class HomeostaticRegulator(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, x, h_pre, w_norm):
        stress = (x.var(dim=1, keepdim=True) - 0.5).abs()
        excitation = h_pre.abs().mean(dim=1, keepdim=True)
        fatigue = w_norm.view(1, 1).expand(x.size(0), 1)
        state = torch.cat([stress, excitation, fatigue], dim=1)
        ctrl = self.net(state)
        return {
            'metabolism': ctrl[:, 0:1],
            'sensitivity': ctrl[:, 1:2],
            'gate': ctrl[:, 2:3]
        }

# =============================================================================
# META-HOMEOSTATIC ENGINE (MEJORADO)
# =============================================================================
class MetaHomeostaticEngine(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.lr_reg = ComponentRegulator("learning_rate", state_dim=8, cross_dim=3)
        self.plasticity_reg = ComponentRegulator("plasticity", state_dim=8, cross_dim=3)
        self.symbiotic_reg = ComponentRegulator("symbiotic", state_dim=8, cross_dim=3)
        self.meta_learner = MetaLearner()
        self.loss_buffer = []
        
        # NUEVO: Momentum para suavizar actualizaciones
        self.lr_momentum = 0.0
        self.plasticity_momentum = 0.0
        
    def forward(self, global_loss: float, step: int) -> Dict:
        self.loss_buffer.append(global_loss)
        if len(self.loss_buffer) > 10:
            self.loss_buffer.pop(0)
        
        loss_pred = global_loss
        
        if len(self.loss_buffer) >= 5:
            seq = torch.tensor(self.loss_buffer[-5:], dtype=torch.float32)
            seq = seq.view(1, -1, 1)
            seq = seq.to(next(self.parameters()).device)
            
            loss_pred = self.meta_learner(seq).item()
            self.meta_learner.update(self.meta_learner(seq), global_loss)
        
        # Calcular health scores con cross-interacción
        plast_health = self.plasticity_reg(
            loss=global_loss, 
            weight_norm=0.5, 
            activity=0.5,
            cross_health=0.5
        )
        symb_health = self.symbiotic_reg(
            loss=global_loss, 
            cross_health=plast_health['gain_mod']
        )
        
        return {
            'loss_pred': loss_pred,
            'lr_recommendation': self.lr_reg(loss=global_loss)['recommendation'],
            'plasticity_mod': plast_health['gain_mod'],
            'symbiotic_mod': symb_health['gain_mod'],
            'health_scores': {
                'lr': self.lr_reg.health_history[-1] if self.lr_reg.health_history else 0.5,
                'plasticity': plast_health['health'],
                'symbiotic': symb_health['health']
            }
        }
    
    def get_component_health(self) -> Dict[str, float]:
        return {
            "lr_health": np.mean(self.lr_reg.health_history[-5:]) if self.lr_reg.health_history else 0.5,
            "plasticity_health": np.mean(self.plasticity_reg.health_history[-5:]) if self.plasticity_reg.health_history else 0.5,
            "symbiotic_health": np.mean(self.symbiotic_reg.health_history[-5:]) if self.symbiotic_reg.health_history else 0.5,
        }
    
    # NUEVO: Método para actualizar con momentum
    def update_with_momentum(self, current_lr: float, current_plasticity: float, 
                             meta_out: Dict, surprise_rate: float) -> Tuple[float, float]:
        # Cambios suaves con momentum
        lr_change = meta_out['lr_recommendation'] * 0.1  # Reducir magnitud
        new_lr = current_lr + lr_change
        
        # Actualizar plasticity con momentum
        plasticity_change = (meta_out['plasticity_mod'] - 0.5) * surprise_rate * 0.02
        self.plasticity_momentum = self.config.meta_momentum * self.plasticity_momentum + \
                                  (1 - self.config.meta_momentum) * plasticity_change
        
        new_plasticity = current_plasticity + self.plasticity_momentum
        
        # Clipping
        new_lr = max(self.config.lr_min, min(self.config.lr_max, new_lr))
        new_plasticity = max(0.2, min(1.0, new_plasticity))
        
        return new_lr, new_plasticity

# =============================================================================
# NEURONA CON SORPRESA
# =============================================================================
class PhysioNeuron(nn.Module):
    def __init__(self, d_in, d_out, config: Config):
        super().__init__()
        self.config = config
        
        self.W_slow = nn.Linear(d_in, d_out, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=1.4)
        
        self.register_buffer('W_fast', torch.zeros(d_out, d_in))
        self.regulator = HomeostaticRegulator(d_in)
        
        self.register_buffer('error_history', torch.zeros(50))
        self.history_ptr = 0
        
    def forward(self, x, surprise_threshold=2.0):
        batch = x.size(0)
        slow = self.W_slow(x)
        
        with torch.no_grad():
            error = (slow - slow.mean()).pow(2).mean().item()
            self.error_history[self.history_ptr % 50] = error
            self.history_ptr += 1
            
            recent_errors = self.error_history[:min(self.history_ptr, 10)]
            mean_error = recent_errors.mean() if len(recent_errors) > 0 else 0.0
            std_error = recent_errors.std() if len(recent_errors) > 1 else 1.0
            surprise = error > (mean_error + surprise_threshold * std_error) and self.history_ptr > 5
        
        if self.training and surprise and self.config.use_plasticity:
            y = slow
            hebb = torch.mm(y.T, x) / batch
            forget = (y.pow(2).mean(0, keepdim=True).T * self.W_fast)
            plasticity_rate = self.config.plasticity_strength * 0.1
            self.W_fast.data.add_((torch.tanh(hebb - forget)) * plasticity_rate)
        
        fast = F.linear(x, self.W_fast) if self.config.use_plasticity else 0
        
        if self.config.use_homeostasis:
            physio = self.regulator(x, slow, self.W_slow.weight.norm())
            gate = physio['gate']
            beta = 0.5 + physio['sensitivity'] * 2.0
        else:
            gate = 1.0
            beta = 1.0
        
        combined = slow + fast * gate
        out = combined * torch.sigmoid(beta * combined)
        
        return out, {
            'metabolism': physio['metabolism'].mean().item() if self.config.use_homeostasis else 0.5,
            'surprise': float(surprise)
        }

# =============================================================================
# SYMBIOTIC DUAL
# =============================================================================
class SymbioticDual(nn.Module):
    def __init__(self, dim, atoms=4):
        super().__init__()
        self.clean = nn.Linear(dim, dim)
        self.noisy = nn.Linear(dim, dim)
        self.basis = nn.Parameter(torch.empty(atoms, dim))
        nn.init.orthogonal_(self.basis, gain=0.5)
        self.consensus_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, influence=0.5):
        noise = torch.randn_like(x) * 0.2
        x_noisy = x + noise
        
        clean_out = self.clean(x)
        noisy_out = self.noisy(x_noisy)
        
        consensus = torch.sigmoid(self.consensus_weight)
        out = consensus * clean_out + (1 - consensus) * noisy_out
        
        mutual_loss = F.mse_loss(clean_out, noisy_out.detach()) + \
                      F.mse_loss(noisy_out, clean_out.detach())
        
        return out, mutual_loss

# =============================================================================
# MICROTOPOBRAIN
# =============================================================================
class MicroTopoBrain(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_nodes = config.grid_size ** 2
        self.embed_dim = config.embed_dim
        
        self.input_proj = nn.Linear(64, self.embed_dim * self.num_nodes)
        
        self.node_processors = nn.ModuleList([
            PhysioNeuron(self.embed_dim, self.embed_dim, config)
            for _ in range(self.num_nodes)
        ])
        
        self.symbiotic = SymbioticDual(self.embed_dim) if config.use_symbiotic else None
        self.meta_engine = MetaHomeostaticEngine(config)
        
        self.readout = nn.Linear(self.embed_dim * self.num_nodes, 10)
        self.supcon_head = nn.Sequential(
            nn.Linear(self.embed_dim * self.num_nodes, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False)
        ) if config.use_supcon else None
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, y=None, step=0):
        batch = x.size(0)
        x_emb = self.input_proj(x).view(batch, self.num_nodes, self.embed_dim)
        
        node_outs = []
        surprises = []
        avg_metabolism = 0
        
        for i, node in enumerate(self.node_processors):
            out, stats = node(x_emb[:, i, :])
            node_outs.append(out)
            surprises.append(stats['surprise'])
            avg_metabolism += stats['metabolism']
        
        avg_metabolism /= len(self.node_processors)
        surprise_rate = sum(surprises) / len(surprises)
        
        x_proc = torch.stack(node_outs, dim=1)
        
        mutual_loss = 0
        if self.symbiotic:
            refined = []
            for i in range(self.num_nodes):
                r, mloss = self.symbiotic(x_proc[:, i, :], influence=avg_metabolism)
                refined.append(r)
                mutual_loss += mloss
            x_proc = torch.stack(refined, dim=1)
        
        x_flat = x_proc.view(batch, -1)
        logits = self.readout(x_flat)
        proj = self.supcon_head(x_flat) if self.supcon_head else None
        
        return {
            'logits': logits,
            'proj': proj,
            'metabolism': avg_metabolism,
            'surprise_rate': surprise_rate,
            'mutual_loss': mutual_loss / self.num_nodes if self.symbiotic else 0
        }

# =============================================================================
# TRAINER (MEJORADO)
# =============================================================================
class ConfigurableTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.metrics = MetricsCollector(config)
        self.env = DataEnvironment()
        
    def train(self, model: MicroTopoBrain):
        model.to(self.config.device)
        
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if 'meta_engine' not in n],
            lr=self.config.lr,
            weight_decay=1e-4
        )
        
        criterion = nn.CrossEntropyLoss()
        recent_losses = []
        
        phase_boundaries = [
            int(0.2 * self.config.steps),
            int(0.5 * self.config.steps),
            int(0.8 * self.config.steps),
            self.config.steps
        ]
        
        for step in range(self.config.steps):
            if step < phase_boundaries[0]:
                phase = "WORLD_1"
            elif step < phase_boundaries[1]:
                phase = "WORLD_2"
            elif step < phase_boundaries[2]:
                phase = "CHAOS"
            else:
                phase = "WORLD_1"
            
            model.train()
            x, y = self.env.get_batch(phase, self.config.batch_size, step)
            x, y = x.to(self.config.device), y.to(self.config.device)
            
            outputs = model(x, y, step=step)
            logits = outputs['logits']
            
            loss = criterion(logits, y)
            loss += outputs['mutual_loss'] * 0.1
            
            recent_losses.append(loss.item())
            if len(recent_losses) > self.config.meta_homeo_window:
                recent_losses.pop(0)
            
            # NUEVO: Warm-up period y actualizaciones suaves
            if step % 10 == 0 and model.config.meta_homeo_enable and step > self.config.meta_warmup:
                meta_out = model.meta_engine(loss.item(), step)
                
                # Usar momentum para actualizaciones suaves
                current_lr = optimizer.param_groups[0]['lr']
                current_plasticity = model.config.plasticity_strength
                
                new_lr, new_plasticity = model.meta_engine.update_with_momentum(
                    current_lr, current_plasticity, meta_out, outputs['surprise_rate']
                )
                
                optimizer.param_groups[0]['lr'] = new_lr
                model.config.plasticity_strength = new_plasticity
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            acc = (logits.argmax(1) == y).float().mean().item() * 100
            self.metrics.log_batch(step, {
                'loss': loss.item(),
                'acc': acc,
                'lr': optimizer.param_groups[0]['lr'],
                'metabolism': outputs['metabolism'],
                'surprise': outputs['surprise_rate'],
                'plasticity': model.config.plasticity_strength  # NUEVO: Log plasticity
            })
        
        return model
    
    def evaluate(self, model: MicroTopoBrain):
        model.eval()
        forgetting_curve = []
        
        with torch.no_grad():
            self.env.inject_concept_drift()
            
            X, y = self.env.get_full()
            X, y = X.to(self.config.device), y.to(self.config.device)
            outputs = model(X)
            global_acc = (outputs['logits'].argmax(1) == y).float().mean().item() * 100
            
            X2, y2 = self.env.get_w2()
            X2, y2 = X2.to(self.config.device), y2.to(self.config.device)
            outputs2 = model(X2)
            w2_ret = (outputs2['logits'].argmax(1) == y2).float().mean().item() * 100
            
            # Calcular curva de olvido
            for i in range(0, len(X2), 100):
                subset_x, subset_y = X2[:i+100], y2[:i+100]
                out_sub = model(subset_x)
                ret = (out_sub['logits'].argmax(1) == subset_y).float().mean().item() * 100
                forgetting_curve.append(ret)
            
            # NUEVO: Calcular AUC de la curva de olvido
            if len(forgetting_curve) > 1:
                # Normalizar x-axis para AUC
                x = np.linspace(0, 1, len(forgetting_curve))
                auc_forgetting = np.trapz(forgetting_curve, x)  # Área bajo la curva
            else:
                auc_forgetting = 0.0
            
            health = model.meta_engine.get_component_health() if model.meta_engine else {}
            
            return {
                'global_acc': global_acc,
                'w2_retention': w2_ret,
                'forgetting_curve': forgetting_curve,
                'final_forgetting': forgetting_curve[-1] if forgetting_curve else 0,
                'auc_forgetting': auc_forgetting,  # NUEVA Métrica
                'n_params': model.count_parameters(),
                'component_health': health,
                'final_lr': self.config.lr,
                'final_plasticity': self.config.plasticity_strength
            }

# =============================================================================
# ABLACIÓN
# =============================================================================
def generate_ablation_matrix_4levels():
    components = ['plasticity', 'supcon', 'symbiotic']
    matrix = {}
    
    matrix['L0_NoHomeo_NoComp'] = {
        'use_homeostasis': False,
        'meta_homeo_enable': False
    }
    
    matrix['L1_Homeo_Only'] = {
        'use_homeostasis': True,
        'meta_homeo_enable': False
    }
    
    for comp in components:
        matrix[f'L2_Homeo+{comp.capitalize()}'] = {
            'use_homeostasis': True,
            'meta_homeo_enable': True,
            f'use_{comp}': True
        }
    
    for c1, c2 in combinations(components, 2):
        matrix[f'L3_Homeo+{c1.capitalize()}+{c2.capitalize()}'] = {
            'use_homeostasis': True,
            'meta_homeo_enable': True,
            f'use_{c1}': True,
            f'use_{c2}': True
        }
    
    matrix['L4_HomeoFullAll'] = {
        'use_homeostasis': True,
        'meta_homeo_enable': True,
        'use_plasticity': True,
        'use_supcon': True,
        'use_symbiotic': True
    }
    
    return matrix

# =============================================================================
# MAIN
# =============================================================================
def run_ablation_study():
    seed_everything(42)
    results_dir = Path("neurologos_v6_2_scientific")
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("🧠 NeuroLogos v6.2 — Meta-Aprendizaje Adaptativo Científico")
    print("=" * 80)
    print("✅ Meta-Learner LSTM con Dropout + LayerNorm")
    print("✅ Actualizaciones con momentum (suavizado)")
    print("✅ Warm-up period de 100 steps")
    print("✅ Métrica AUC-forgetting (más robusta)")
    print("✅ Plasticity ajustada con momentum")
    print("=" * 80)
    
    ablation_matrix = generate_ablation_matrix_4levels()
    results = {}
    
    for name, overrides in ablation_matrix.items():
        print(f"\n▶ {name}")
        config = Config(**overrides)
        
        trainer = ConfigurableTrainer(config)
        model = MicroTopoBrain(config)
        model = trainer.train(model)
        metrics = trainer.evaluate(model)
        results[name] = metrics
        
        if 'forgetting_curve' in metrics:
            print(f"   Forgetting: {metrics['forgetting_curve'][0]:.1f}% → {metrics['forgetting_curve'][-1]:.1f}%")
        
        print(f"   Global: {metrics['global_acc']:.1f}% | "
              f"W2 Ret: {metrics['w2_retention']:.1f}% | "
              f"AUC: {metrics['auc_forgetting']:.1f}% | "
              f"Params: {metrics['n_params']:,}")
        
        trainer.metrics.save(results_dir / f"{name}_metrics.json")
    
    print("\n" + "=" * 80)
    print("📊 VEREDICTO: ¿META-HOMEOSTASIS GANA?")
    print("-" * 80)
    
    static = results['L0_NoHomeo_NoComp']
    # Usar AUC como métrica principal
    best = max(results.values(), key=lambda x: x['auc_forgetting'])
    best_name = max(results.keys(), key=lambda k: results[k]['auc_forgetting'])
    
    print(f"{'Métrica':<20} | {'Static':<10} | {'Best':<10} | {'Δ'}")
    print(f"{'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")
    print(f"{'Global Acc':<20} | {static['global_acc']:9.1f}% | {best['global_acc']:9.1f}% | {best['global_acc'] - static['global_acc']:+6.1f}%")
    print(f"{'W2 Retention':<20} | {static['w2_retention']:9.1f}% | {best['w2_retention']:9.1f}% | {best['w2_retention'] - static['w2_retention']:+6.1f}%")
    print(f"{'AUC Forgetting':<20} | {static['auc_forgetting']:9.1f}% | {best['auc_forgetting']:9.1f}% | {best['auc_forgetting'] - static['auc_forgetting']:+6.1f}%")
    print(f"{'Parámetros':<20} | {static['n_params']:9,} | {best['n_params']:9,} | {best['n_params'] - static['n_params']:+6,}")
    
    print("\n🏆 Mejor configuración:", best_name)
    
    # Usar AUC para el veredicto
    auc_delta = best['auc_forgetting'] - static['auc_forgetting']
    if auc_delta > 5:
        print(f"🚀 ÉXITO: Meta-Homeostasis mejora AUC en +{auc_delta:.1f}%")
    elif auc_delta > 2:
        print(f"✅ MEJORA: Meta-Homeostasis mejora AUC en +{auc_delta:.1f}%")
    else:
        print("🔍 Marginal: Revisar hiper-parámetros o aumentar steps")
    
    with open(results_dir / "ablation_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_ablation_study()
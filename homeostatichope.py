import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ========================================================================
# CONFIGURACIÓN
# ========================================================================

class Config:
    SEED = 42
    NUM_EPOCHS = 60  # Más épocas para ver adaptación
    BATCH_SIZE = 32
    BASE_LR = 5e-3
    WEIGHT_DECAY = 1e-3
    
    D_MODEL = 96
    MLP_HIDDEN = 192
    CMS_FREQUENCIES = [1, 4, 16]
    
    # Adversarial
    ADV_EPSILON_START = 0.05
    ADV_EPSILON_END = 0.3
    ADV_STEPS = 5
    
    ENABLE_HOMEOSTATIC = True
    ENABLE_CMS = True
    ENABLE_ADVERSARIAL = True

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'✅ GPU' if device.type == 'cuda' else '⚠️  CPU mode'}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    return device

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ========================================================================
# DATASET
# ========================================================================

class RealWorldEnvironment:
    def __init__(self, seed=42):
        data, target = load_digits(return_X_y=True)
        data = data / 16.0
        
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2, random_state=seed, stratify=target
        )
        
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
        
        mask1 = self.y_train < 5
        self.X1_train = self.X_train[mask1]
        self.y1_train = self.y_train[mask1]
        
        mask2 = self.y_train >= 5
        self.X2_train = self.X_train[mask2]
        self.y2_train = self.y_train[mask2]
        
        print(f"📊 Dataset: World1={len(self.X1_train)}, World2={len(self.X2_train)}, Test={len(self.X_test)}")
    
    def get_batch(self, phase: str, batch_size: int = 32):
        if phase == "WORLD_1":
            idx = torch.randint(0, len(self.X1_train), (batch_size,))
            return self.X1_train[idx], self.y1_train[idx]
        elif phase == "WORLD_2":
            idx = torch.randint(0, len(self.X2_train), (batch_size,))
            return self.X2_train[idx], self.y2_train[idx]
        elif phase == "CHAOS":
            idx = torch.randint(0, len(self.X_train), (batch_size,))
            noise = torch.randn_like(self.X_train[idx]) * 0.3
            return self.X_train[idx] + noise, self.y_train[idx]
        else:
            idx = torch.randint(0, len(self.X_train), (batch_size,))
            return self.X_train[idx], self.y_train[idx]
    
    def get_test_loader(self, batch_size: int = 32):
        dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ========================================================================
# REGULADOR HOMEOSTÁTICO OMNISCIENTE
# ========================================================================

class OmniscientRegulator(nn.Module):
    """
    Motor homeostático con acceso TOTAL a señales internas críticas
    y control DIRECTO de hiperparámetros en tiempo real
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # Input: 10 señales críticas del sistema
        # [loss, grad_norm, weight_norm, acc_drop, input_var, 
        #  activation_mag, epsilon, phase_change, loss_spike, prediction_confidence]
        
        self.perception = nn.Sequential(
            nn.Linear(10, 32),
            nn.LayerNorm(32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh()
        )
        
        # Controles homeostáticos (6 salidas)
        # [learning_rate_mult, plasticity, gate, dropout, momentum, gradient_clip]
        self.control_head = nn.Sequential(
            nn.Linear(32, 6),
            nn.Sigmoid()  # Todos en [0, 1]
        )
        
        # Memoria de estado (para detectar cambios temporales)
        self.register_buffer('prev_loss', torch.tensor(1.0))
        self.register_buffer('prev_acc', torch.tensor(0.5))
        
    def forward(self, signals: Dict) -> Dict:
        """
        Args:
            signals: Diccionario con señales internas del sistema
        Returns:
            controls: Diccionario con hiperparámetros ajustados
        """
        # Extraer señales críticas
        loss = signals.get('loss', 1.0)
        grad_norm = signals.get('grad_norm', 0.0)
        weight_norm = signals.get('weight_norm', 1.0)
        acc = signals.get('accuracy', 0.5)
        input_var = signals.get('input_variance', 0.5)
        activation_mag = signals.get('activation_magnitude', 0.5)
        epsilon = signals.get('adversarial_epsilon', 0.0)
        phase = signals.get('phase_indicator', 0.0)  # 0=W1, 1=W2, 2=chaos
        pred_conf = signals.get('prediction_confidence', 0.5)
        
        # Detectar cambios drásticos (concept drift, chaos)
        with torch.no_grad():
            loss_spike = max(0.0, loss - self.prev_loss.item())
            acc_drop = max(0.0, self.prev_acc.item() - acc)
            
            self.prev_loss = torch.tensor(loss)
            self.prev_acc = torch.tensor(acc)
        
        # Vector de entrada al regulador
        sensor_input = torch.tensor([
            min(loss, 5.0) / 5.0,           # Loss normalizado
            min(grad_norm, 10.0) / 10.0,    # Grad norm
            min(weight_norm, 10.0) / 10.0,  # Weight norm
            min(acc_drop, 1.0),              # Caída de accuracy
            input_var,                       # Varianza del input
            min(activation_mag, 5.0) / 5.0, # Magnitud activaciones
            epsilon,                         # Epsilon adversarial
            phase / 2.0,                     # Fase (normalizado)
            min(loss_spike, 5.0) / 5.0,     # Spike de loss
            pred_conf                        # Confianza predicciones
        ], device=weight_norm.device if isinstance(weight_norm, torch.Tensor) else 'cpu')
        
        # Percepción
        features = self.perception(sensor_input)
        
        # Generar controles
        controls_raw = self.control_head(features)
        
        # Mapear a rangos útiles
        controls = {
            'lr_multiplier': 0.1 + controls_raw[0] * 1.9,      # [0.1, 2.0]
            'plasticity': controls_raw[1],                      # [0, 1]
            'gate': controls_raw[2],                            # [0, 1]
            'dropout': controls_raw[3] * 0.3,                   # [0, 0.3]
            'momentum': 0.5 + controls_raw[4] * 0.49,          # [0.5, 0.99]
            'grad_clip': 0.5 + controls_raw[5] * 4.5           # [0.5, 5.0]
        }
        
        return controls

# ========================================================================
# MEMORIA LÍQUIDA CON CONTROL HOMEOSTÁTICO
# ========================================================================

class AdaptiveLiquidMemory(nn.Module):
    """Memoria líquida que responde a controles homeostáticos"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        self.memory_slow = nn.Parameter(torch.randn(d_model) * 0.01)
        self.fast_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.orthogonal_(self.fast_proj.weight)
        
        self.dropout = nn.Dropout(0.1)  # Será controlado dinámicamente
        
    def forward(self, x: torch.Tensor, controls: Dict) -> torch.Tensor:
        # Aplicar dropout controlado
        dropout_p = controls.get('dropout', 0.1)
        self.dropout.p = dropout_p
        
        # Fast component
        fast = self.fast_proj(x)
        fast = self.dropout(fast)
        
        # Slow component
        slow = self.memory_slow.unsqueeze(0).expand_as(x)
        
        # Gate controlado
        gate = controls.get('gate', 0.5)
        output = fast + gate * slow
        
        # Plasticidad controlada (actualización de memoria lenta)
        plasticity = controls.get('plasticity', 0.5)
        if self.training and plasticity > 0.1:
            with torch.no_grad():
                # Actualización Hebbiana suave
                update = x.mean(dim=0) * 0.01 * plasticity
                self.memory_slow.data = (1 - plasticity * 0.01) * self.memory_slow.data + update
        
        return output

# ========================================================================
# SELF-MODIFYING MEMORY
# ========================================================================

class HomeostaticSelfModMemory(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.d_model = d_model
        
        self.proj_kv = nn.Linear(d_model, d_model * 2)
        self.proj_q = nn.Linear(d_model, d_model)
        
        self.liquid_mem = AdaptiveLiquidMemory(d_model)
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x: torch.Tensor, controls: Dict) -> Tuple[torch.Tensor, Dict]:
        kv = self.proj_kv(x)
        k, v = kv.chunk(2, dim=-1)
        q = self.proj_q(x)
        
        # Memoria con controles
        mem_out = self.liquid_mem(v, controls)
        
        # Atención
        attn = torch.sum(q * k, dim=-1, keepdim=True) / (self.d_model ** 0.5)
        attn = torch.sigmoid(attn)
        
        combined = attn * mem_out + (1 - attn) * v
        output = self.output_proj(combined)
        
        # Señales internas para el regulador
        signals = {
            'activation_magnitude': output.abs().mean().item(),
            'weight_norm': self.proj_kv.weight.norm().item()
        }
        
        return output, signals

# ========================================================================
# CMS
# ========================================================================

class ContinuumMemorySystem(nn.Module):
    def __init__(self, frequencies: List[int], d_model: int, hidden_dim: int):
        super().__init__()
        self.frequencies = frequencies
        
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, d_model),
                nn.LayerNorm(d_model)
            ) for _ in frequencies
        ])
    
    def forward(self, x: torch.Tensor, global_step: int = 0) -> torch.Tensor:
        out = x
        for level, freq in zip(self.levels, self.frequencies):
            if global_step % freq == 0:
                out = out + level(out)
        return out

# ========================================================================
# MODELO HOPE + PHYSIO OMNISCIENTE
# ========================================================================

class OmniscientHopeModel(nn.Module):
    def __init__(self, config: Config, n_features: int, n_classes: int):
        super().__init__()
        self.config = config
        
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, config.MLP_HIDDEN),
            nn.LayerNorm(config.MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(config.MLP_HIDDEN, config.D_MODEL)
        )
        
        # Regulador homeostático omnisciente
        if config.ENABLE_HOMEOSTATIC:
            self.regulator = OmniscientRegulator(config.D_MODEL)
        else:
            self.regulator = None
        
        # Memory y CMS
        if config.ENABLE_HOMEOSTATIC:
            self.memory = HomeostaticSelfModMemory(config.D_MODEL, config.MLP_HIDDEN)
        else:
            self.memory = nn.Sequential(
                nn.Linear(config.D_MODEL, config.MLP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.MLP_HIDDEN, config.D_MODEL),
                nn.LayerNorm(config.D_MODEL)
            )
        
        if config.ENABLE_CMS:
            self.cms = ContinuumMemorySystem(config.CMS_FREQUENCIES, config.D_MODEL, config.MLP_HIDDEN)
        else:
            self.cms = lambda x, step: x
        
        self.output_proj = nn.Linear(config.D_MODEL, n_classes)
        
    def forward(self, x: torch.Tensor, signals: Dict, global_step: int = 0) -> Tuple[torch.Tensor, Dict]:
        x_proj = self.input_proj(x)
        
        # Obtener controles homeostáticos
        if self.regulator is not None:
            controls = self.regulator(signals)
        else:
            controls = {}
        
        # Memory con controles
        if self.config.ENABLE_HOMEOSTATIC:
            x_mem, mem_signals = self.memory(x_proj, controls)
            signals.update(mem_signals)
        else:
            x_mem = self.memory(x_proj)
        
        # CMS
        x_cms = self.cms(x_mem, global_step)
        
        logits = self.output_proj(x_cms)
        
        return logits, controls

# ========================================================================
# PGD ATTACK
# ========================================================================

def pgd_attack(model, x, y, epsilon, steps, device, signals):
    was_training = model.training
    model.eval()
    
    delta = torch.zeros_like(x, requires_grad=False)
    delta.uniform_(-epsilon, epsilon)
    
    for step in range(steps):
        delta_var = delta.clone().detach().requires_grad_(True)
        x_adv = x + delta_var
        logits, _ = model(x_adv, signals)
        loss = F.cross_entropy(logits, y)
        
        loss.backward()
        
        with torch.no_grad():
            if delta_var.grad is not None:
                grad_sign = delta_var.grad.sign()
            else:
                grad_sign = torch.randn_like(delta).sign()
            
            delta = delta + (epsilon / steps) * 1.5 * grad_sign
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(x + delta, 0, 1) - x
    
    if was_training:
        model.train()
    
    return (x + delta).detach()

# ========================================================================
# TRAINER CONSCIENTE
# ========================================================================

class ConsciousTrainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.BASE_LR,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.base_optimizer,
            T_max=config.NUM_EPOCHS
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Historia para detectar cambios
        self.loss_history = []
        self.acc_history = []
    
    def train_step(self, x, y, epsilon, global_step, phase):
        x, y = x.to(self.device), y.to(self.device)
        
        # Construir señales para el regulador
        signals = {
            'loss': self.loss_history[-1] if self.loss_history else 1.0,
            'accuracy': self.acc_history[-1] if self.acc_history else 0.5,
            'input_variance': x.var().item(),
            'adversarial_epsilon': epsilon,
            'phase_indicator': {'WORLD_1': 0.0, 'WORLD_2': 1.0, 'CHAOS': 2.0, 'MIXED': 1.5}.get(phase, 0.0)
        }
        
        # Adversarial attack
        if self.config.ENABLE_ADVERSARIAL and epsilon > 0:
            x_adv = pgd_attack(self.model, x, y, epsilon, self.config.ADV_STEPS, self.device, signals)
        else:
            x_adv = x
        
        # Forward
        self.base_optimizer.zero_grad()
        logits, controls = self.model(x_adv, signals, global_step)
        loss = self.criterion(logits, y)
        
        # Añadir señales de gradiente
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        signals['grad_norm'] = grad_norm
        
        # Backward
        loss.backward()
        
        # Gradient clipping controlado
        clip_val = controls.get('grad_clip', 1.0) if controls else 1.0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
        
        # Learning rate adaptativo
        lr_mult = controls.get('lr_multiplier', 1.0) if controls else 1.0
        for param_group in self.base_optimizer.param_groups:
            param_group['lr'] = self.config.BASE_LR * lr_mult
        
        self.base_optimizer.step()
        
        # Métricas
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = pred.eq(y).float().mean().item()
            conf = F.softmax(logits, dim=1).max(dim=1)[0].mean().item()
            
            self.loss_history.append(loss.item())
            self.acc_history.append(acc)
            
            if len(self.loss_history) > 100:
                self.loss_history.pop(0)
                self.acc_history.pop(0)
            
            signals['prediction_confidence'] = conf
        
        return loss.item(), acc, controls
    
    def evaluate(self, test_loader, epsilon=0.0, phase='TEST'):
        self.model.eval()
        correct = 0
        total = 0
        
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            signals = {
                'adversarial_epsilon': epsilon,
                'phase_indicator': 1.0
            }
            
            if epsilon > 0:
                x = pgd_attack(self.model, x, y, epsilon, self.config.ADV_STEPS, self.device, signals)
            
            with torch.no_grad():
                logits, _ = self.model(x, signals)
                pred = logits.argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        
        return correct / total

# ========================================================================
# EXPERIMENTO
# ========================================================================

def run_conscious_experiment(config: Config, device: torch.device):
    print("\n" + "="*80)
    print("🧠 HOPE + OMNISCIENT PHYSIO: Regulador Consciente")
    print("="*80)
    print(f"Homeostasis: {'✅ Omnisciente' if config.ENABLE_HOMEOSTATIC else '❌'}")
    print(f"CMS: {'✅' if config.ENABLE_CMS else '❌'}")
    print(f"Adversarial: {'✅' if config.ENABLE_ADVERSARIAL else '❌'}")
    print("="*80 + "\n")
    
    env = RealWorldEnvironment(config.SEED)
    test_loader = env.get_test_loader(config.BATCH_SIZE)
    
    model = OmniscientHopeModel(config, n_features=64, n_classes=10)
    trainer = ConsciousTrainer(model, config, device)
    
    print(f"🧠 Parámetros: {sum(p.numel() for p in model.parameters()):,}\n")
    
    print(f"{'EP':<3} | {'Phase':<8} | {'Loss':<6} | {'Acc':<5} | {'Test':<5} | {'Adv':<5} | "
          f"{'LR×':<5} | {'Plas':<5} | {'Event'}")
    print("-" * 95)
    
    global_step = 0
    best_clean = 0.0
    best_adv = 0.0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        progress = epoch / config.NUM_EPOCHS
        epsilon = config.ADV_EPSILON_START + progress * (config.ADV_EPSILON_END - config.ADV_EPSILON_START)
        
        # Concept drift schedule
        if epoch <= 20:
            phase = "WORLD_1"
            event = "📘 0-4"
        elif epoch <= 40:
            phase = "WORLD_2"
            event = "⚡ 5-9"
        elif epoch <= 50:
            phase = "CHAOS"
            event = "🌀 Chaos"
        else:
            phase = "MIXED"
            event = "🔄 Mixed"
        
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 30
        controls_avg = {'lr_multiplier': 1.0, 'plasticity': 0.5}
        
        for _ in range(num_batches):
            x, y = env.get_batch(phase, config.BATCH_SIZE)
            loss, acc, controls = trainer.train_step(x, y, epsilon, global_step, phase)
            epoch_loss += loss
            epoch_acc += acc
            global_step += 1
            
            if controls:
                controls_avg['lr_multiplier'] = 0.9 * controls_avg['lr_multiplier'] + 0.1 * controls.get('lr_multiplier', 1.0)
                controls_avg['plasticity'] = 0.9 * controls_avg['plasticity'] + 0.1 * controls.get('plasticity', 0.5)
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        # Evaluación
        if epoch % 5 == 0 or epoch == config.NUM_EPOCHS:
            test_clean = trainer.evaluate(test_loader, epsilon=0.0)
            test_adv = trainer.evaluate(test_loader, epsilon=config.ADV_EPSILON_END)
            
            best_clean = max(best_clean, test_clean)
            best_adv = max(best_adv, test_adv)
            
            print(f"{epoch:<3} | {phase:<8} | {avg_loss:.4f} | {avg_acc:.3f} | "
                  f"{test_clean:.3f} | {test_adv:.3f} | "
                  f"{controls_avg['lr_multiplier']:.2f}  | {controls_avg['plasticity']:.2f}  | {event}")
        else:
            print(f"{epoch:<3} | {phase:<8} | {avg_loss:.4f} | {avg_acc:.3f} | "
                  f"  -   |   -   | "
                  f"{controls_avg['lr_multiplier']:.2f}  | {controls_avg['plasticity']:.2f}  | {event}")
        
        trainer.scheduler.step()
    
    print("\n" + "="*80)
    print("📊 RESULTADOS")
    print("="*80)
    print(f"Best Clean: {best_clean * 100:.2f}% | Best Adv: {best_adv * 100:.2f}%")
    print("="*80 + "\n")
    
    return {'clean': best_clean, 'adv': best_adv}

# ========================================================================
# ABLACIÓN
# ========================================================================

def run_ablation(device):
    configs = [
        {'name': 'Omniscient Homeostasis', 'homeo': True, 'cms': True, 'adv': True},
        {'name': 'Sin Homeostasis', 'homeo': False, 'cms': True, 'adv': True},
        {'name': 'Baseline', 'homeo': False, 'cms': False, 'adv': False}
    ]
    
    results = {}
    
    for exp in configs:
        print(f"\n{'='*80}")
        print(f"🧪 {exp['name']}")
        print(f"{'='*80}")
        
        config = Config()
        config.ENABLE_HOMEOSTATIC = exp['homeo']
        config.ENABLE_CMS = exp['cms']
        config.ENABLE_ADVERSARIAL = exp['adv']
        
        res = run_conscious_experiment(config, device)
        results[exp['name']] = res
    
    print("\n" + "="*80)
    print("📈 COMPARACIÓN")
    print("="*80)
    print(f"{'Config':<30} | {'Clean':<8} | {'Adv':<8} | {'Gap':<8}")
    print("-" * 80)
    
    for name, res in results.items():
        gap = (res['clean'] - res['adv']) * 100
        print(f"{name:<30} | {res['clean']*100:>6.2f}% | {res['adv']*100:>6.2f}% | {gap:>6.2f}%")
    
    print("="*80)

if __name__ == "__main__":
    device = setup_device()
    set_seed(Config.SEED)
    run_ablation(device)
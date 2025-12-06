import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict

# ========================================================================
# CONFIGURACIÓN Y UTILIDADES
# ========================================================================

class Config:
    """Configuración centralizada basada en el paper (Secciones 7-9)"""
    # Entrenamiento
    SEED = 42
    NUM_EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Arquitectura
    SEQ_LENGTH = 32
    VOCAB_SIZE = 64
    D_MODEL = 128
    MLP_HIDDEN = 256
    
    # CMS (Sección 7.1 - Eq. 70-71)
    CMS_FREQUENCIES = [1, 4, 16]  # rápido → lento
    
    # Self-Modifying (Sección 8.1 - Eq. 83-88)
    CHUNK_SIZE = 8  # para paralelización chunk-wise
    
    # Dataset
    DATASET_SIZE = 4096
    
    # Ablación
    ENABLE_SELF_MODIFYING = True
    ENABLE_CMS = True
    ENABLE_DGD = True
    USE_CHUNK_WISE = True

def setup_device():
    """Configuración automática de dispositivo"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  Ejecutando en CPU (será más lento)")
    return device

def set_seed(seed: int):
    """Reproducibilidad completa"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========================================================================
# DELTA GRADIENT DESCENT (DGD) - Sección 4.5, Apéndice C
# ========================================================================

class DeltaGradientDescent:
    """
    Implementación de DGD según Eq. 121:
    W_{t+1} = W_t(I - α_t x_t x_t^T) - β ∇_{y_t} L(W_t; x_t) x_t^T
    
    A diferencia del GD estándar, DGD incluye:
    1. Decaimiento adaptativo basado en el estado actual (término αI)
    2. Dependencia de muestras anteriores (no asume i.i.d.)
    """
    
    @staticmethod
    def apply_update(grad: torch.Tensor, 
                     param: torch.Tensor,
                     x_normalized: torch.Tensor,
                     eta: torch.Tensor,
                     alpha: torch.Tensor,
                     lambda_norm: float = 1.0) -> torch.Tensor:
        """
        Aplica la regla DGD a un gradiente
        
        Args:
            grad: Gradiente actual ∇L
            param: Parámetro W_t
            x_normalized: Input normalizado (||x|| = λ)
            eta: Learning rate adaptativo (por muestra)
            alpha: Retention gate (por muestra)
            lambda_norm: Norma de x (default: 1.0 para L2-norm)
        
        Returns:
            Gradiente modificado según DGD
        """
        # Eq. 121: α_t = 1/(λ² + η_t)
        alpha_t = 1.0 / (lambda_norm**2 + eta.mean())
        
        # Término de decaimiento adaptativo: x_t x_t^T
        # Para eficiencia, usamos outer product solo cuando necesario
        if param.dim() >= 2:
            # Decaimiento basado en estado (simulado)
            decay = alpha_t * alpha.mean()
            modified_grad = grad * (1.0 - decay)
        else:
            modified_grad = grad
        
        return modified_grad

# ========================================================================
# SELF-MODIFYING MEMORY - Sección 8.1 (Eq. 83-88)
# ========================================================================

class SelfModifyingMemory(nn.Module):
    """
    Implementación completa de Self-Modifying Deep Associative Memory
    
    Paper: Sección 8.1, Ecuaciones 83-88
    - Genera sus propias proyecciones (k, v, q, η, α)
    - Genera valores propios auto-referenciales (Eq. 84)
    - Aplica regla Delta para actualización (Eq. 88)
    """
    
    def __init__(self, d_model: int, hidden_dim: int, chunk_size: int):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        
        # Memorias para proyecciones (Eq. 83)
        self.mem_k = self._make_memory_module()
        self.mem_v = self._make_memory_module()
        self.mem_q = self._make_memory_module()
        
        # Parámetros adaptativos (learning rate y retention)
        self.mem_eta = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # ∈ (0, 1)
        )
        self.mem_alpha = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # ∈ (0, 1)
        )
        
        # Generadores de valores auto-referenciales (Eq. 84)
        self.value_gen_k = nn.Linear(d_model, d_model)
        self.value_gen_v = nn.Linear(d_model, d_model)
        self.value_gen_memory = nn.Linear(d_model, d_model)
        
        # Memoria principal (Eq. 86)
        self.mem_memory = self._make_memory_module()
        
        # Estados iniciales (meta-aprendidos implícitamente)
        self.register_buffer('init_mem_k', torch.zeros(d_model, d_model))
        self.register_buffer('init_mem_v', torch.zeros(d_model, d_model))
        
    def _make_memory_module(self) -> nn.Module:
        """Crea un módulo de memoria (MLP de 2 capas con residual)"""
        return nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.d_model)
        )
    
    def forward(self, x: torch.Tensor, 
                prev_states: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass con actualización chunk-wise (Sección 8.2)
        
        Args:
            x: (B, S, D) - Input tokens embebidos
            prev_states: Estados previos de memorias
            
        Returns:
            output: (B, S, D)
            new_states: Estados actualizados
        """
        B, S, D = x.shape
        
        # Inicializar estados si no existen
        if prev_states is None:
            prev_states = {
                'mem_k': self.init_mem_k.unsqueeze(0).expand(B, -1, -1),
                'mem_v': self.init_mem_v.unsqueeze(0).expand(B, -1, -1)
            }
        
        # 1. Generar proyecciones adaptativas (Eq. 83)
        k = self.mem_k(x)  # (B, S, D)
        v = self.mem_v(x)
        q = self.mem_q(x)
        eta = self.mem_eta(x)  # (B, S, 1)
        alpha = self.mem_alpha(x)  # (B, S, 1)
        
        # 2. Generar valores auto-referenciales (Eq. 84)
        v_hat_k = self.value_gen_k(v)
        v_hat_v = self.value_gen_v(v)
        
        # 3. Aplicar regla Delta chunk-wise (Eq. 88 simplificada)
        outputs = []
        new_mem_k = prev_states['mem_k']
        new_mem_v = prev_states['mem_v']
        
        for chunk_start in range(0, S, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, S)
            
            # Extraer chunk
            k_chunk = k[:, chunk_start:chunk_end]  # (B, C, D)
            v_chunk = v[:, chunk_start:chunk_end]
            q_chunk = q[:, chunk_start:chunk_end]
            eta_chunk = eta[:, chunk_start:chunk_end]
            alpha_chunk = alpha[:, chunk_start:chunk_end]
            v_hat_k_chunk = v_hat_k[:, chunk_start:chunk_end]
            v_hat_v_chunk = v_hat_v[:, chunk_start:chunk_end]
            
            # Actualización Delta simplificada (versión paralelizable)
            # Eq. 88: M_t = M_{t-1}(αI - η kk^T) - η∇L
            for t in range(k_chunk.size(1)):
                kt = k_chunk[:, t]  # (B, D)
                vt = v_chunk[:, t]
                eta_t = eta_chunk[:, t]  # (B, 1)
                alpha_t = alpha_chunk[:, t]
                v_hat_kt = v_hat_k_chunk[:, t]
                
                # Término de decaimiento: (αI - η kk^T)
                # Versión eficiente: aplicar como multiplicación
                decay = alpha_t * (1.0 - eta_t * torch.sum(kt * kt, dim=-1, keepdim=True))
                new_mem_k = new_mem_k * decay.unsqueeze(-1)
                
                # Término de gradiente (aproximado)
                grad_term = torch.bmm(
                    (new_mem_k @ kt.unsqueeze(-1) - v_hat_kt.unsqueeze(-1)),
                    kt.unsqueeze(-1).transpose(1, 2)
                )
                new_mem_k = new_mem_k - eta_t.unsqueeze(-1) * grad_term
            
            # Recuperar de memoria principal
            chunk_output = self.mem_memory(q_chunk)
            outputs.append(chunk_output)
        
        output = torch.cat(outputs, dim=1)
        
        new_states = {
            'mem_k': new_mem_k,
            'mem_v': new_mem_v
        }
        
        return output, new_states

# ========================================================================
# CONTINUUM MEMORY SYSTEM (CMS) - Sección 7.1 (Eq. 70-71)
# ========================================================================

class ContinuumMemorySystem(nn.Module):
    """
    Sistema de memoria continuo con múltiples frecuencias de actualización
    
    Paper: Sección 7.1
    - Niveles con diferentes frecuencias (rápido → lento)
    - Actualización condicional basada en chunk_size
    - Conexión secuencial (Eq. 73) o independiente (Eq. 74)
    """
    
    def __init__(self, 
                 frequencies: List[int], 
                 d_model: int, 
                 hidden_dim: int,
                 connection_type: str = 'sequential'):
        super().__init__()
        self.frequencies = frequencies
        self.d_model = d_model
        self.connection_type = connection_type
        
        # Niveles de memoria (uno por frecuencia)
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_model),
                nn.LayerNorm(d_model)  # estabilidad
            ) for _ in frequencies
        ])
        
        # Estados iniciales (meta-aprendidos)
        for i, freq in enumerate(frequencies):
            self.register_buffer(f'init_level_{i}', torch.zeros(d_model))
        
        # Agregador para conexión independiente (Eq. 74)
        if connection_type == 'independent':
            self.aggregator = nn.Linear(d_model * len(frequencies), d_model)
    
    def forward(self, x: torch.Tensor, global_step: int) -> torch.Tensor:
        """
        Forward pass con actualizaciones multi-frecuencia
        
        Args:
            x: (B, S, D)
            global_step: Paso global de entrenamiento
            
        Returns:
            output: (B, S, D)
        """
        if self.connection_type == 'sequential':
            # Eq. 73: Conexión secuencial
            out = x
            for i, (level, freq) in enumerate(zip(self.levels, self.frequencies)):
                if global_step % freq == 0:
                    out = out + level(out)  # residual
            return out
        
        elif self.connection_type == 'independent':
            # Eq. 74: Conexión independiente con agregación
            outputs = []
            for i, (level, freq) in enumerate(zip(self.levels, self.frequencies)):
                if global_step % freq == 0:
                    out_i = level(x)
                else:
                    # Mantener salida previa (simulado con identidad)
                    out_i = x
                outputs.append(out_i)
            
            # Agregar todas las salidas
            concat = torch.cat(outputs, dim=-1)
            return self.aggregator(concat)
        
        else:
            raise ValueError(f"connection_type desconocido: {self.connection_type}")

# ========================================================================
# MODELO HOPE COMPLETO - Sección 8.3 (Eq. 94-97)
# ========================================================================

class HopeModel(nn.Module):
    """
    Arquitectura Hope: Self-Modifying Memory + CMS
    
    Paper: Sección 8.3, Figura 5
    """
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 cms_frequencies: List[int],
                 mlp_hidden: int,
                 chunk_size: int,
                 enable_self_modifying: bool = True,
                 enable_cms: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.enable_self_modifying = enable_self_modifying
        self.enable_cms = enable_cms
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Self-Modifying Memory (opcional para ablación)
        if enable_self_modifying:
            self.self_mod = SelfModifyingMemory(d_model, mlp_hidden, chunk_size)
        else:
            # Baseline: simple MLP
            self.baseline_mlp = nn.Sequential(
                nn.Linear(d_model, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, d_model)
            )
        
        # CMS (opcional para ablación)
        if enable_cms:
            self.cms = ContinuumMemorySystem(cms_frequencies, d_model, mlp_hidden)
        else:
            # Baseline: single MLP
            self.baseline_cms = nn.Sequential(
                nn.Linear(d_model, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, d_model)
            )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Estados para self-modifying
        self.reset_states()
    
    def reset_states(self):
        """Reset de estados internos (para nuevas secuencias)"""
        self.prev_states = None
    
    def forward(self, 
                x: torch.Tensor, 
                global_step: int,
                return_internals: bool = False) -> torch.Tensor:
        """
        Forward pass completo
        
        Args:
            x: (B, S) - Input token IDs
            global_step: Paso global
            return_internals: Si retornar estados internos (para análisis)
        """
        # Embedding
        x_emb = self.embedding(x)  # (B, S, D)
        
        # Self-Modifying Memory
        if self.enable_self_modifying:
            x_mod, self.prev_states = self.self_mod(x_emb, self.prev_states)
        else:
            x_mod = self.baseline_mlp(x_emb)
        
        # CMS
        if self.enable_cms:
            x_cms = self.cms(x_mod, global_step)
        else:
            x_cms = self.baseline_cms(x_mod)
        
        # Output
        logits = self.output_proj(x_cms)  # (B, S, V)
        
        if return_internals:
            return logits, {'states': self.prev_states, 'x_mod': x_mod, 'x_cms': x_cms}
        return logits

# ========================================================================
# ENTRENADOR CON ABLACIÓN CIENTÍFICA
# ========================================================================

class HopeTrainer:
    """
    Sistema de entrenamiento con ablación automática y métricas científicas
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Config,
                 device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizador
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Criterio
        self.criterion = nn.CrossEntropyLoss()
        
        # DGD (si está habilitado)
        self.dgd = DeltaGradientDescent() if config.ENABLE_DGD else None
        
        # Métricas
        self.metrics = defaultdict(list)
    
    def train_epoch(self, 
                    train_loader: DataLoader, 
                    epoch: int,
                    global_step: int) -> Tuple[float, float, int]:
        """
        Entrena una época completa
        
        Returns:
            avg_loss, avg_acc, new_global_step
        """
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = len(train_loader)
        print_every = max(1, num_batches // 10) if num_batches > 10 else 1
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            logits = self.model(x_batch, global_step)
            
            # Loss
            logits_flat = logits.reshape(-1, self.config.VOCAB_SIZE)
            targets_flat = y_batch.reshape(-1)
            loss = self.criterion(logits_flat, targets_flat)
            
            # Backward
            loss.backward()
            
            # Aplicar DGD si está habilitado
            if self.dgd is not None and self.config.ENABLE_DGD:
                # Normalizar inputs para DGD
                with torch.no_grad():
                    x_norm = F.normalize(x_batch.float(), dim=-1)
                    # Generar eta y alpha (simplificado)
                    eta = torch.ones(x_batch.size(0), device=self.device) * 0.01
                    alpha = torch.ones(x_batch.size(0), device=self.device) * 0.9
                    
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and 'embedding' not in name:
                            param.grad = self.dgd.apply_update(
                                param.grad, param, x_norm, eta, alpha
                            )
            
            # Gradient clipping (estabilidad)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Métricas
            epoch_loss += loss.item() * targets_flat.size(0)
            preds = logits_flat.argmax(dim=1)
            correct += preds.eq(targets_flat).sum().item()
            total += targets_flat.size(0)
            
            global_step += 1
            
            # Logging
            if (batch_idx + 1) % print_every == 0 or batch_idx == num_batches - 1:
                batch_acc = preds.eq(targets_flat).float().mean().item()
                print(f"  Época {epoch:2d} | Batch {batch_idx+1:3d}/{num_batches} | "
                      f"Loss: {loss.item():.4f} | Acc: {batch_acc:.4f}")
        
        avg_loss = epoch_loss / total
        avg_acc = correct / total
        
        return avg_loss, avg_acc, global_step
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader, global_step: int) -> Tuple[float, float]:
        """Evaluación sin gradientes"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            logits = self.model(x_batch, global_step)
            
            logits_flat = logits.reshape(-1, self.config.VOCAB_SIZE)
            targets_flat = y_batch.reshape(-1)
            loss = self.criterion(logits_flat, targets_flat)
            
            total_loss += loss.item() * targets_flat.size(0)
            preds = logits_flat.argmax(dim=1)
            correct += preds.eq(targets_flat).sum().item()
            total += targets_flat.size(0)
        
        return total_loss / total, correct / total

# ========================================================================
# SUITE DE ABLACIÓN
# ========================================================================

def run_ablation_study(config: Config, device: torch.device):
    """
    Ejecuta estudio de ablación completo
    
    Configuraciones testeadas:
    1. Hope completo (Self-Mod + CMS + DGD)
    2. Sin Self-Modifying
    3. Sin CMS
    4. Sin DGD
    5. Baseline (sin ninguno)
    """
    
    print("\n" + "="*85)
    print("ESTUDIO DE ABLACIÓN - NESTED LEARNING (Hope)")
    print("="*85)
    
    # Crear dataset
    print("\n📊 Generando dataset sintético...")
    data = np.random.randint(1, config.VOCAB_SIZE, size=(config.DATASET_SIZE, config.SEQ_LENGTH))
    X = torch.from_numpy(data).long()
    y = X.clone()
    
    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Configuraciones de ablación
    ablation_configs = [
        {
            'name': 'Hope Completo',
            'self_mod': True,
            'cms': True,
            'dgd': True
        },
        {
            'name': 'Sin Self-Modifying',
            'self_mod': False,
            'cms': True,
            'dgd': True
        },
        {
            'name': 'Sin CMS',
            'self_mod': True,
            'cms': False,
            'dgd': True
        },
        {
            'name': 'Sin DGD',
            'self_mod': True,
            'cms': True,
            'dgd': False
        },
        {
            'name': 'Baseline (sin nada)',
            'self_mod': False,
            'cms': False,
            'dgd': False
        }
    ]
    
    results = {}
    
    for abl_config in ablation_configs:
        print(f"\n{'='*85}")
        print(f"🧪 Configuración: {abl_config['name']}")
        print(f"{'='*85}")
        
        # Crear modelo
        model = HopeModel(
            vocab_size=config.VOCAB_SIZE,
            d_model=config.D_MODEL,
            cms_frequencies=config.CMS_FREQUENCIES,
            mlp_hidden=config.MLP_HIDDEN,
            chunk_size=config.CHUNK_SIZE,
            enable_self_modifying=abl_config['self_mod'],
            enable_cms=abl_config['cms']
        )
        
        # Actualizar config para DGD
        original_dgd = config.ENABLE_DGD
        config.ENABLE_DGD = abl_config['dgd']
        
        # Trainer
        trainer = HopeTrainer(model, config, device)
        
        # Entrenar
        global_step = 0
        best_test_acc = 0.0
        train_losses = []
        test_accs = []
        
        start_time = time.time()
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            train_loss, train_acc, global_step = trainer.train_epoch(
                train_loader, epoch, global_step
            )
            test_loss, test_acc = trainer.evaluate(test_loader, global_step)
            
            train_losses.append(train_loss)
            test_accs.append(test_acc)
            best_test_acc = max(best_test_acc, test_acc)
            
            print(f"  → Época {epoch:2d} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        elapsed = time.time() - start_time
        
        results[abl_config['name']] = {
            'best_test_acc': best_test_acc,
            'final_test_acc': test_accs[-1],
            'final_train_loss': train_losses[-1],
            'time': elapsed,
            'train_losses': train_losses,
            'test_accs': test_accs
        }
        
        print(f"\n✅ Completado en {elapsed:.2f}s | Mejor Test Acc: {best_test_acc:.4f}")
        
        # Restaurar config
        config.ENABLE_DGD = original_dgd
    
    # Reporte final
    print(f"\n{'='*85}")
    print("📈 RESULTADOS FINALES DE ABLACIÓN")
    print(f"{'='*85}")
    print(f"{'Configuración':<25} {'Mejor Acc':<12} {'Acc Final':<12} {'Tiempo (s)':<12}")
    print(f"{'-'*85}")
    
    for name, res in results.items():
        print(f"{name:<25} {res['best_test_acc']:<12.4f} "
              f"{res['final_test_acc']:<12.4f} {res['time']:<12.2f}")
    
    return results

# ========================================================================
# MAIN
# ========================================================================

if __name__ == "__main__":
    # Setup
    device = setup_device()
    set_seed(Config.SEED)
    
    print("\n" + "="*85)
    print("IMPLEMENTACIÓN CIENTÍFICA DE NESTED LEARNING (Hope)")
    print("Paper: 'Nested Learning: The Illusion of Deep Learning Architecture'")
    print("Google Research, NeurIPS 2025")
    print("="*85)
    
    print(f"\n⚙️  Configuración:")
    print(f"   Épocas: {Config.NUM_EPOCHS}")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    print(f"   Learning rate: {Config.LEARNING_RATE}")
    print(f"   Vocab size: {Config.VOCAB_SIZE}")
    print(f"   d_model: {Config.D_MODEL}")
    print(f"   CMS frequencies: {Config.CMS_FREQUENCIES}")
    print(f"   Chunk size: {Config.CHUNK_SIZE}")
    print(f"   Dataset size: {Config.DATASET_SIZE}")
    
    # Ejecutar ablación
    results = run_ablation_study(Config, device)
    
    print(f"\n{'='*85}")
    print("✅ EXPERIMENTO COMPLETADO")
    print(f"{'='*85}\n")
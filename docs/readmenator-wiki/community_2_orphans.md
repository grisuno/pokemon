# orphans

*Community 2 | 184 files | cohesion 0.00*

## Definition

This community groups 184 file(s) rooted at `root` with dominant language py (cohesion 0.00). Central symbols: `AblationConfig`, `AblationMatrix`, `AdaptiveCombinatorialComplexLayer`, `AdaptiveLearningMotor`, `AdaptiveLiquidMemory`, `AdaptiveMagnitudeGate`, `AdaptiveTopology`, `AdaptiveTopologyLayer`. Core file: `exodia_op_2.py` (103 symbols). Documented purpose: Ablation Científico Riguroso con Control de Variables y Estabilidad Numérica (Versión Optimizada para CPU).

## Files

### `.` (184 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `01_mcculloch_pitts.py` | py | utility | 1 | no |
| `01_topobrain_cou_v2.py` | py | utility | 27 | yes |
| `01_topobrain_cpu.py` | py | utility | 22 | yes |
| `01_topobrain_cpu_v3.py` | py | utility | 31 | yes |
| `01_topobrain_cpu_v4.py` | py | utility | 27 | yes |
| `01_topobrain_cpu_v5.py` | py | utility | 27 | yes |
| `01_topobrain_cpu_v6.py` | py | utility | 40 | yes |
| `01_topobrain_cpu_v7.py` | py | utility | 22 | yes |
| `01_topobrain_cpu_v8.py` | py | utility | 23 | yes |
| `01_topobrain_ganador_gpu_v1.py` | py | utility | 22 | yes |
| `02_perceptron.py` | py | utility | 5 | no |
| `03_backpropagation.py` | py | utility | 5 | no |
| `04_cnn_lenet.py` | py | utility | 3 | no |
| `05_svm_rbf.py` | py | utility | 0 | no |
| `06_lstm_char.py` | py | utility | 4 | no |
| `07_random_forest.py` | py | utility | 0 | no |
| `08_vae_mnist.py` | py | utility | 7 | no |
| `09_transformer_mini.py` | py | utility | 14 | no |
| `10_gan_mnist_lite.py` | py | utility | 6 | no |
| `11_bert_tiny.py` | py | utility | 16 | no |

*... and 164 more files in this community.*


## Key Symbols

- `mcculloch_pitts_neuron` (function, `01_mcculloch_pitts.py:7`) `def mcculloch_pitts_neuron(inputs, weights, threshold)` - Neurona artificial de McCulloch-Pitts (1943).
- `Config` (class, `01_topobrain_cou_v2.py:25`) `class Config`
- `to_dict` (method, `01_topobrain_cou_v2.py:56`) `def to_dict(self)`
- `get_topology_config` (method, `01_topobrain_cou_v2.py:59`) `def get_topology_config(self)` - Configuración estable para topología adaptable
- `seed_everything` (method, `01_topobrain_cou_v2.py:70`) `def seed_everything(seed)`
- `get_tabular_loaders` (method, `01_topobrain_cou_v2.py:77`) `def get_tabular_loaders(config)` - Dataset tabular controlado con características NOIR simuladas
- `StableSupConLoss` (class, `01_topobrain_cou_v2.py:116`) `class StableSupConLoss(Module)` - Versión estable de SupConLoss con manejo de bordes robusto
- `__init__` (method, `01_topobrain_cou_v2.py:118`) `def __init__(self, temperature)`
- `forward` (method, `01_topobrain_cou_v2.py:123`) `def forward(self, features, labels)`
- `StableContinuumMemoryCell` (class, `01_topobrain_cou_v2.py:147`) `class StableContinuumMemoryCell(Module)` - Versión estable y ligera de ContinuumMemoryCell con clamping y normalización
- `__init__` (method, `01_topobrain_cou_v2.py:149`) `def __init__(self, input_dim, hidden_dim)`
- `forward` (method, `01_topobrain_cou_v2.py:181`) `def forward(self, x, controls)`
- `StableSymbioticBasisRefinement` (class, `01_topobrain_cou_v2.py:231`) `class StableSymbioticBasisRefinement(Module)` - Refinamiento simbiótico estable con regularización explícita (versión ligera)
- `__init__` (method, `01_topobrain_cou_v2.py:233`) `def __init__(self, dim, num_atoms)`
- `forward` (method, `01_topobrain_cou_v2.py:242`) `def forward(self, x)`
- `StableTopologyManager` (class, `01_topobrain_cou_v2.py:267`) `class StableTopologyManager` - Gestor de topología con protocolos de supervivencia (versión optimizada para CPU)
- `__init__` (method, `01_topobrain_cou_v2.py:269`) `def __init__(self, num_nodes, config)`
- `get_adjacency` (method, `01_topobrain_cou_v2.py:288`) `def get_adjacency(self, plasticity)` - Obtener matriz de adyacencia con estabilidad garantizada
- `prune_topology` (method, `01_topobrain_cou_v2.py:317`) `def prune_topology(self, current_density, epoch)` - Poda controlada con protocolo de emergencia
- `get_density` (method, `01_topobrain_cou_v2.py:337`) `def get_density(self)` - Calcular densidad actual de manera estable
- `StableTopoBrain` (class, `01_topobrain_cou_v2.py:345`) `class StableTopoBrain(Module)` - Implementación estable y ligera de TopoBrain para ablation científico en CPU
- `__init__` (method, `01_topobrain_cou_v2.py:347`) `def __init__(self, config)`
- `_init_weights` (method, `01_topobrain_cou_v2.py:400`) `def _init_weights(self)` - Inicialización estable de pesos
- `forward` (method, `01_topobrain_cou_v2.py:408`) `def forward(self, x, controls)`
- `stable_pgd_attack` (method, `01_topobrain_cou_v2.py:477`) `def stable_pgd_attack(model, x, y, eps, steps, controls)` - Ataque PGD estable con manejo de gradientes robusto
- `train_epoch` (method, `01_topobrain_cou_v2.py:520`) `def train_epoch(model, loader, optimizer, config, epoch, controls)` - Entrenamiento por época con monitoreo detallado
- `evaluate_model` (method, `01_topobrain_cou_v2.py:584`) `def evaluate_model(model, loader, config, adversarial, controls)` - Evaluación rigurosa con o sin ataques adversariales
- `run_scientific_ablation` (method, `01_topobrain_cou_v2.py:632`) `def run_scientific_ablation()` - Ejecución científica del ablation con control de variables
- `Config` (class, `01_topobrain_cpu.py:21`) `class Config`
- `to_dict` (method, `01_topobrain_cpu.py:57`) `def to_dict(self)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 2 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (root: physio_chimera_v15_monitored) and community 2 (orphans).
- [INFERRED] shares_context community 1 <-> 2 (strength 0.5): Inferred shared context (language py) with no import path between community 1 (root: premium_synergy_democratic) and community 2 (orphans).

## Risks

- [taint medium] `06_lstm_char.py` -> `06_lstm_char.py` via `urllib.request` (0 hops)
- [taint medium] `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py` -> `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py` via `urllib.request` (0 hops)
- [taint medium] `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py` -> `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py` via `urllib.request` (0 hops)
- [taint medium] `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py` -> `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py` via `urllib.request` (0 hops)
- [taint medium] `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py` -> `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py` via `urllib.request` (0 hops)
- [taint medium] `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py` -> `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py` via `urllib.request` (0 hops)
- [taint medium] `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py` -> `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py` via `urllib.request` (0 hops)
- [taint medium] `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py` -> `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py` via `urllib.request` (0 hops)
- [taint medium] `bicamera.py.py` -> `bicamera.py.py` via `urllib.request` (0 hops)
- [taint medium] `bicameral.py` -> `bicameral.py` via `urllib.request` (0 hops)
- [taint medium] `bicameral2.py` -> `bicameral2.py` via `urllib.request` (0 hops)
- [taint medium] `bicameral3.py` -> `bicameral3.py` via `urllib.request` (0 hops)
- [taint high] `exodia_op_2.py` -> `exodia_op_2.py` via `subprocess` (0 hops)
- [taint medium] `exodia_op_2.py` -> `exodia_op_2.py` via `urllib.request` (0 hops)
- [taint medium] `exodia_op_2.py` -> `exodia_op_2.py` via `urllib.request` (0 hops)

## Open Questions

- Why do 52 file(s) lack file-level docs (e.g. `01_mcculloch_pitts.py`)? What purpose do they serve?
- Is the dangerous import `urllib.request` in `06_lstm_char.py` still required, or can it be isolated?
- What would break if the most connected file in orphans changed?
- Should orphans be split, given cohesion 0.00?

## Sources

- `01_mcculloch_pitts.py`
- `01_topobrain_cou_v2.py`
- `01_topobrain_cpu.py`
- `01_topobrain_cpu_v3.py`
- `01_topobrain_cpu_v4.py`
- `01_topobrain_cpu_v5.py`
- `01_topobrain_cpu_v6.py`
- `01_topobrain_cpu_v7.py`
- `01_topobrain_cpu_v8.py`
- `01_topobrain_ganador_gpu_v1.py`
- `02_perceptron.py`
- `03_backpropagation.py`
- `04_cnn_lenet.py`
- `05_svm_rbf.py`
- `06_lstm_char.py`
- `07_random_forest.py`
- `08_vae_mnist.py`
- `09_transformer_mini.py`
- `10_gan_mnist_lite.py`
- `11_bert_tiny.py`
- *... and 164 more*

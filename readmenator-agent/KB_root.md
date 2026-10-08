# Subsystem: root (page 1 of 15)
Pages: [KB_root.md](KB_root.md), [KB_root_p2.md](KB_root_p2.md), [KB_root_p3.md](KB_root_p3.md), [KB_root_p4.md](KB_root_p4.md), [KB_root_p5.md](KB_root_p5.md), [KB_root_p6.md](KB_root_p6.md), [KB_root_p7.md](KB_root_p7.md), [KB_root_p8.md](KB_root_p8.md), [KB_root_p9.md](KB_root_p9.md), [KB_root_p10.md](KB_root_p10.md), [KB_root_p11.md](KB_root_p11.md), [KB_root_p12.md](KB_root_p12.md), [KB_root_p13.md](KB_root_p13.md), [KB_root_p14.md](KB_root_p14.md), [KB_root_p15.md](KB_root_p15.md)

## 01_mcculloch_pitts.py
- Doc: mcculloch_pitts_neuron: Neurona artificial de McCulloch-Pitts (1943). - inputs: vector binario...
- Layer: utility
- Language: py
- Symbols:
  - `mcculloch_pitts_neuron` (function, line 7) `def mcculloch_pitts_neuron(inputs, weights, threshold)`

## 01_topobrain_cou_v2.py
- Doc: Ablation Científico Riguroso con Control de Variables y Estabilidad Numérica (Versión Optimizada...
- Layer: utility
- Language: py
- Symbols:
  - `Config` (class, line 25) `class Config`
  - `seed_everything` (method, line 70) `def seed_everything(seed)`
  - `get_tabular_loaders` (method, line 77) `def get_tabular_loaders(config)`
  - `StableSupConLoss` (class, line 116) `class StableSupConLoss(Module)`
  - `StableContinuumMemoryCell` (class, line 147) `class StableContinuumMemoryCell(Module)`
  - `StableSymbioticBasisRefinement` (class, line 231) `class StableSymbioticBasisRefinement(Module)`
  - `StableTopologyManager` (class, line 267) `class StableTopologyManager`
  - `StableTopoBrain` (class, line 345) `class StableTopoBrain(Module)`
  - `stable_pgd_attack` (method, line 477) `def stable_pgd_attack(model, x, y, eps, steps, controls)`
  - `train_epoch` (method, line 520) `def train_epoch(model, loader, optimizer, config, epoch, controls)`
  - `evaluate_model` (method, line 584) `def evaluate_model(model, loader, config, adversarial, controls)`
  - `run_scientific_ablation` (method, line 632) `def run_scientific_ablation()`
  - `to_dict` (method, line 56) `def to_dict(self)`
  - `get_topology_config` (method, line 59) `def get_topology_config(self)`
  - `__init__` (method, line 118) `def __init__(self, temperature)`
  - `forward` (method, line 123) `def forward(self, features, labels)`
  - `__init__` (method, line 149) `def __init__(self, input_dim, hidden_dim)`
  - `forward` (method, line 181) `def forward(self, x, controls)`
  - `__init__` (method, line 233) `def __init__(self, dim, num_atoms)`
  - `forward` (method, line 242) `def forward(self, x)`
  - `__init__` (method, line 269) `def __init__(self, num_nodes, config)`
  - `get_adjacency` (method, line 288) `def get_adjacency(self, plasticity)`
  - `prune_topology` (method, line 317) `def prune_topology(self, current_density, epoch)`
  - `get_density` (method, line 337) `def get_density(self)`
  - `__init__` (method, line 347) `def __init__(self, config)`
  - `_init_weights` (method, line 400) `def _init_weights(self)`
  - `forward` (method, line 408) `def forward(self, x, controls)`

## 01_topobrain_cpu.py
- Doc: Ablation Científica Rigurosa – CPU-only, miles de parámetros, PGD real, sin CIFAR (nominal...
- Layer: utility
- Language: py
- Symbols:
  - `Config` (class, line 21) `class Config`
  - `seed_everything` (method, line 63) `def seed_everything(seed)`
  - `get_tabular_loaders` (method, line 69) `def get_tabular_loaders(config)`
  - `SupConLoss` (class, line 94) `class SupConLoss(Module)`
  - `ContinuumMemoryCell` (class, line 109) `class ContinuumMemoryCell(Module)`
  - `SymbioticBasisRefinement` (class, line 143) `class SymbioticBasisRefinement(Module)`
  - `TopoBrainTabular` (class, line 166) `class TopoBrainTabular(Module)`
  - `pgd_attack` (method, line 247) `def pgd_attack(model, x, y, eps, steps, controls)`
  - `generate_ablation_configs` (method, line 263) `def generate_ablation_configs(base_config)`
  - `train_and_evaluate` (method, line 293) `def train_and_evaluate(config, name)`
  - `run_ablation` (method, line 337) `def run_ablation()`
  - `to_dict` (method, line 57) `def to_dict(self)`
  - `__init__` (method, line 95) `def __init__(self, temperature)`
  - `forward` (method, line 98) `def forward(self, features, labels)`
  - `__init__` (method, line 110) `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate)`
  - `forward` (method, line 124) `def forward(self, x, controls)`
  - `__init__` (method, line 144) `def __init__(self, dim)`
  - `forward` (method, line 151) `def forward(self, x)`
  - `__init__` (method, line 167) `def __init__(self, config)`
  - `get_adj` (method, line 198) `def get_adj(self)`
  - `forward` (method, line 203) `def forward(self, x, controls)`
  - `evaluate_adv` (method, line 323) `def evaluate_adv(loader, eps, steps)`

## 01_topobrain_cpu_v3.py
- Doc: Ablation Científico Riguroso – TopoBrain Tabular (NOIR-aware, CPU-only, miles de parámetros)
- Layer: utility
- Language: py
- Symbols:
  - `Config` (class, line 24) `class Config`
  - `seed_everything` (method, line 63) `def seed_everything(seed)`
  - `get_tabular_loaders` (method, line 69) `def get_tabular_loaders(config)`
  - `SupConLoss` (class, line 93) `class SupConLoss(Module)`
  - `ContinuumMemoryCell` (class, line 108) `class ContinuumMemoryCell(Module)`
  - `SymbioticBasisRefinement` (class, line 145) `class SymbioticBasisRefinement(Module)`
  - `PrefrontalOrchestrator` (class, line 165) `class PrefrontalOrchestrator(Module)`
  - `AdaptiveCombinatorialComplexLayer` (class, line 227) `class AdaptiveCombinatorialComplexLayer(Module)`
  - `TopoBrainTabular` (class, line 320) `class TopoBrainTabular(Module)`
  - `pgd_attack` (method, line 395) `def pgd_attack(model, x, y, eps, steps)`
  - `compute_topology_metrics` (method, line 416) `def compute_topology_metrics(model, config)`
  - `prune_topology` (method, line 450) `def prune_topology(model, config, controls)`
  - `train_and_evaluate` (method, line 482) `def train_and_evaluate(config, run_name)`
  - `run_ablation` (method, line 640) `def run_ablation()`
  - `to_dict` (method, line 57) `def to_dict(self)`
  - `__init__` (method, line 94) `def __init__(self, temperature)`
  - `forward` (method, line 97) `def forward(self, features, labels)`
  - `__init__` (method, line 109) `def __init__(self, input_dim, hidden_dim)`
  - `forward` (method, line 122) `def forward(self, x, controls)`
  - `__init__` (method, line 146) `def __init__(self, dim)`
  - `forward` (method, line 153) `def forward(self, x)`
  - `__init__` (method, line 166) `def __init__(self, config)`
  - `forward` (method, line 183) `def forward(self, metrics)`
  - `reset_context` (method, line 221) `def reset_context(self)`
  - `__init__` (method, line 228) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
  - `get_adj` (method, line 264) `def get_adj(self)`
  - `forward` (method, line 269) `def forward(self, x, controls)`
  - `__init__` (method, line 321) `def __init__(self, config)`
  - `_initialize_memories` (method, line 352) `def _initialize_memories(self)`
  - `forward` (method, line 363) `def forward(self, x, controls, prev_states)`
  - `evaluate_adv` (method, line 593) `def evaluate_adv(loader, eps, steps)`

## 01_topobrain_cpu_v4.py
- Doc: Ablation Científico Riguroso con Control de Variables y Estabilidad Numérica (Versión Optimizada...
- Layer: utility
- Language: py
- Symbols:
  - `Config` (class, line 25) `class Config`
  - `seed_everything` (method, line 70) `def seed_everything(seed)`
  - `get_tabular_loaders` (method, line 77) `def get_tabular_loaders(config)`
  - `StableSupConLoss` (class, line 116) `class StableSupConLoss(Module)`
  - `StableContinuumMemoryCell` (class, line 147) `class StableContinuumMemoryCell(Module)`
  - `StableSymbioticBasisRefinement` (class, line 231) `class StableSymbioticBasisRefinement(Module)`
  - `StableTopologyManager` (class, line 267) `class StableTopologyManager`
  - `StableTopoBrain` (class, line 345) `class StableTopoBrain(Module)`
  - `stable_pgd_attack` (method, line 475) `def stable_pgd_attack(model, x, y, eps, steps, controls)`
  - `train_epoch` (method, line 518) `def train_epoch(model, loader, optimizer, config, epoch, controls)`
  - `evaluate_model` (method, line 583) `def evaluate_model(model, loader, config, adversarial, controls)`
  - `run_scientific_ablation` (method, line 628) `def run_scientific_ablation()`
  - `to_dict` (method, line 56) `def to_dict(self)`
  - `get_topology_config` (method, line 59) `def get_topology_config(self)`
  - `__init__` (method, line 118) `def __init__(self, temperature)`
  - `forward` (method, line 123) `def forward(self, features, labels)`
  - `__init__` (method, line 149) `def __init__(self, input_dim, hidden_dim)`
  - `forward` (method, line 181) `def forward(self, x, controls)`
  - `__init__` (method, line 233) `def __init__(self, dim, num_atoms)`
  - `forward` (method, line 242) `def forward(self, x)`
  - `__init__` (method, line 269) `def __init__(self, num_nodes, config)`
  - `get_adjacency` (method, line 288) `def get_adjacency(self, plasticity)`
  - `prune_topology` (method, line 317) `def prune_topology(self, current_density, epoch)`
  - `get_density` (method, line 337) `def get_density(self)`
  - `__init__` (method, line 347) `def __init__(self, config)`
  - `_init_weights` (method, line 400) `def _init_weights(self)`
  - `forward` (method, line 408) `def forward(self, x, controls)`

## 01_topobrain_cpu_v5.py
- Doc: Ablation Científico Riguroso con Control de Variables y Estabilidad Numérica (Versión Optimizada...
- Layer: utility
- Language: py
- Symbols:
  - `Config` (class, line 25) `class Config`
  - `seed_everything` (method, line 70) `def seed_everything(seed)`
  - `get_tabular_loaders` (method, line 77) `def get_tabular_loaders(config)`
  - `StableSupConLoss` (class, line 116) `class StableSupConLoss(Module)`
  - `StableContinuumMemoryCell` (class, line 147) `class StableContinuumMemoryCell(Module)`
  - `StableSymbioticBasisRefinement` (class, line 231) `class StableSymbioticBasisRefinement(Module)`
  - `StableTopologyManager` (class, line 267) `class StableTopologyManager`
  - `StableTopoBrain` (class, line 345) `class StableTopoBrain(Module)`
  - `stable_pgd_attack` (method, line 475) `def stable_pgd_attack(model, x, y, eps, steps, controls)`
  - `train_epoch` (method, line 518) `def train_epoch(model, loader, optimizer, config, epoch, controls)`
  - `evaluate_model` (method, line 583) `def evaluate_model(model, loader, config, adversarial, controls)`
  - `run_scientific_ablation` (method, line 708) `def run_scientific_ablation()`
  - `to_dict` (method, line 56) `def to_dict(self)`
  - `get_topology_config` (method, line 59) `def get_topology_config(self)`
  - `__init__` (method, line 118) `def __init__(self, temperature)`
  - `forward` (method, line 123) `def forward(self, features, labels)`
  - `__init__` (method, line 149) `def __init__(self, input_dim, hidden_dim)`
  - `forward` (method, line 181) `def forward(self, x, controls)`
  - `__init__` (method, line 233) `def __init__(self, dim, num_atoms)`
  - `forward` (method, line 242) `def forward(self, x)`
  - `__init__` (method, line 269) `def __init__(self, num_nodes, config)`
  - `get_adjacency` (method, line 288) `def get_adjacency(self, plasticity)`
  - `prune_topology` (method, line 317) `def prune_topology(self, current_density, epoch)`
  - `get_density` (method, line 337) `def get_density(self)`
  - `__init__` (method, line 347) `def __init__(self, config)`
  - `_init_weights` (method, line 400) `def _init_weights(self)`
  - `forward` (method, line 408) `def forward(self, x, controls)`

## 01_topobrain_cpu_v6.py
- Doc: TopoBrain CPU v2.0 - Análisis de Ablación Científico Riguroso
- Layer: utility
- Language: py
- Symbols:
  - `MicroConfig` (class, line 33) `class MicroConfig`
  - `seed_everything` (method, line 95) `def seed_everything(seed)`
  - `get_micro_dataset` (method, line 104) `def get_micro_dataset(config)`
  - `compute_effect_size` (method, line 126) `def compute_effect_size(group1, group2)`
  - `MicroSupConLoss` (class, line 137) `class MicroSupConLoss(Module)`
  - `MicroContinuumCell` (class, line 163) `class MicroContinuumCell(Module)`
  - `MicroSymbioticBasis` (class, line 214) `class MicroSymbioticBasis(Module)`
  - `MicroTopology` (class, line 253) `class MicroTopology`
  - `MicroTopoBrain` (class, line 286) `class MicroTopoBrain(Module)`
  - `micro_pgd_attack` (method, line 432) `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
  - `train_epoch_micro` (method, line 462) `def train_epoch_micro(model, loader, optimizer, config, epoch)`
  - `evaluate_micro` (method, line 517) `def evaluate_micro(model, loader, config, adversarial)`
  - `train_with_cv` (method, line 539) `def train_with_cv(config, dataset, cv_folds)`
  - `AblationMatrix` (class, line 597) `class AblationMatrix`
  - `ScientificAnalyzer` (class, line 744) `class ScientificAnalyzer`
  - `run_scientific_ablation_study` (method, line 850) `def run_scientific_ablation_study()`
  - `to_dict` (method, line 79) `def to_dict(self)`
  - `component_signature` (method, line 82) `def component_signature(self)`
  - `__init__` (method, line 139) `def __init__(self, temperature)`
  - `forward` (method, line 144) `def forward(self, features, labels)`
  - `__init__` (method, line 168) `def __init__(self, dim)`
  - `forward` (method, line 184) `def forward(self, x, plasticity)`
  - `__init__` (method, line 219) `def __init__(self, dim, num_atoms)`
  - `forward` (method, line 232) `def forward(self, x)`
  - `__init__` (method, line 258) `def __init__(self, num_nodes, config)`
  - `get_adjacency` (method, line 276) `def get_adjacency(self, plasticity)`
  - `__init__` (method, line 301) `def __init__(self, config)`
  - `_init_weights` (method, line 353) `def _init_weights(self)`
  - `count_parameters` (method, line 360) `def count_parameters(self)`
  - `forward` (method, line 364) `def forward(self, x, plasticity)`
  - `level1_isolated` (method, line 628) `def level1_isolated()`
  - `level2a_pairs` (method, line 641) `def level2a_pairs()`
  - `level2b_strategic_triads` (method, line 665) `def level2b_strategic_triads()`
  - `level3_inverse_ablation` (method, line 694) `def level3_inverse_ablation()`
  - `level3_full_model` (method, line 719) `def level3_full_model()`
  - `get_complete_matrix` (method, line 730) `def get_complete_matrix(cls)`
  - `compute_statistics` (method, line 748) `def compute_statistics(results_list)`
  - `ttest_vs_baseline` (method, line 774) `def ttest_vs_baseline(exp_scores, baseline_scores)`
  - `detect_synergy` (method, line 786) `def detect_synergy(pair_pgd, comp_a_pgd, comp_b_pgd, baseline_pgd)`
  - `rank_components_by_criticality` (method, line 809) `def rank_components_by_criticality(full_pgd, ablation_results)`

## 01_topobrain_cpu_v7.py
- Doc: TopoBrain CPU-OPTIMIZADO - Funcional para AMD R5 M335
- Layer: utility
- Language: py
- Symbols:
  - `setup_device` (function, line 24) `def setup_device()`
  - `Config` (class, line 42) `class Config`
  - `SymbioticBasis` (class, line 94) `class SymbioticBasis(Module)`
  - `DynamicTopology` (class, line 124) `class DynamicTopology(Module)`
  - `TopoBrainCPU` (class, line 186) `class TopoBrainCPU(Module)`
  - `pgd_attack` (method, line 256) `def pgd_attack(model, x, y, eps, steps, plasticity)`
  - `train_epoch` (method, line 285) `def train_epoch(model, loader, optimizer, config, epoch, device)`
  - `evaluate` (method, line 321) `def evaluate(model, loader, config, device, adversarial)`
  - `get_dataset` (method, line 351) `def get_dataset(config)`
  - `main` (method, line 393) `def main()`
  - `to_dict` (method, line 86) `def to_dict(self)`
  - `__init__` (method, line 95) `def __init__(self, dim, num_atoms)`
  - `forward` (method, line 106) `def forward(self, x)`
  - `__init__` (method, line 125) `def __init__(self, num_nodes, grid_size, config)`
  - `_create_grid_mask` (method, line 136) `def _create_grid_mask(self)`
  - `get_adjacency` (method, line 153) `def get_adjacency(self, plasticity)`
  - `prune_connections` (method, line 158) `def prune_connections(self, threshold)`
  - `get_density` (method, line 178) `def get_density(self)`
  - `__init__` (method, line 187) `def __init__(self, config)`
  - `_init_weights` (method, line 216) `def _init_weights(self)`
  - `count_parameters` (method, line 223) `def count_parameters(self)`
  - `forward` (method, line 226) `def forward(self, x, plasticity)`

## 01_topobrain_cpu_v8.py
- Doc: TopoBrain REAL - GPU AMD via ONNX Runtime (NO OpenCV)
- Layer: utility
- Language: py
- Symbols:
  - `Config` (class, line 28) `class Config`
  - `SymbioticBasis` (class, line 62) `class SymbioticBasis(Module)`
  - `DynamicTopology` (class, line 88) `class DynamicTopology(Module)`
  - `TopoBrainReal` (class, line 144) `class TopoBrainReal(Module)`
  - `pgd_attack` (method, line 223) `def pgd_attack(model, x, y, eps, steps, plasticity)`
  - `train_topobrain` (method, line 252) `def train_topobrain(config)`
  - `export_for_onnxruntime` (method, line 410) `def export_for_onnxruntime(model)`
  - `test_with_onnxruntime` (method, line 465) `def test_with_onnxruntime(X_test, y_test)`
  - `main` (method, line 547) `def main()`
  - `__init__` (method, line 63) `def __init__(self, dim, num_atoms)`
  - `forward` (method, line 76) `def forward(self, x)`
  - `__init__` (method, line 89) `def __init__(self, num_nodes, grid_size, config)`
  - `_create_grid_mask` (method, line 98) `def _create_grid_mask(self)`
  - `get_adjacency` (method, line 115) `def get_adjacency(self, plasticity)`
  - `prune_connections` (method, line 120) `def prune_connections(self, threshold)`
  - `get_density` (method, line 140) `def get_density(self)`
  - `__init__` (method, line 145) `def __init__(self, config)`
  - `_init_weights` (method, line 174) `def _init_weights(self)`
  - `forward` (method, line 181) `def forward(self, x, plasticity)`
  - `forward_with_metrics` (method, line 210) `def forward_with_metrics(self, x, plasticity)`
  - `Wrapper` (class, line 424) `class Wrapper(Module)`
  - `__init__` (method, line 425) `def __init__(self, m)`
  - `forward` (method, line 429) `def forward(self, x)`

## 01_topobrain_ganador_gpu_v1.py
- Doc: TopoBrain AMD GPU POC - Configuración Ganadora Escalada
- Layer: utility
- Language: py
- Symbols:
  - `setup_amd_device` (function, line 31) `def setup_amd_device()`
  - `GPUConfig` (class, line 70) `class GPUConfig`
  - `GPUSymbioticBasis` (class, line 127) `class GPUSymbioticBasis(Module)`
  - `DynamicTopology` (class, line 179) `class DynamicTopology(Module)`
  - `TopoBrainGPU` (class, line 278) `class TopoBrainGPU(Module)`
  - `pgd_attack_gpu` (method, line 405) `def pgd_attack_gpu(model, x, y, eps, steps, plasticity)`
  - `train_epoch_gpu` (method, line 461) `def train_epoch_gpu(model, loader, optimizer, config, epoch, device)`
  - `evaluate_gpu` (method, line 517) `def evaluate_gpu(model, loader, config, device, adversarial)`
  - `get_gpu_dataset` (method, line 547) `def get_gpu_dataset(config)`
  - `main` (method, line 591) `def main()`
  - `to_dict` (method, line 119) `def to_dict(self)`
  - `__init__` (method, line 132) `def __init__(self, dim, num_atoms)`
  - `forward` (method, line 147) `def forward(self, x)`
  - `__init__` (method, line 184) `def __init__(self, num_nodes, grid_size, config)`
  - `_create_grid_mask` (method, line 200) `def _create_grid_mask(self)`
  - `get_adjacency` (method, line 219) `def get_adjacency(self, plasticity)`
  - `prune_connections` (method, line 237) `def prune_connections(self, threshold)`
  - `get_density` (method, line 269) `def get_density(self)`
  - `__init__` (method, line 291) `def __init__(self, config)`
  - `_init_weights` (method, line 335) `def _init_weights(self)`
  - `count_parameters` (method, line 343) `def count_parameters(self)`
  - `forward` (method, line 347) `def forward(self, x, plasticity)`

## 02_perceptron.py
- Layer: utility
- Language: py
- Symbols:
  - `Perceptron` (class, line 31) `class Perceptron`
  - `__init__` (method, line 32) `def __init__(self, input_dim, learning_rate)`
  - `predict` (method, line 37) `def predict(self, X)`
  - `train_step` (method, line 42) `def train_step(self, X_batch, y_batch)`
  - `accuracy` (method, line 51) `def accuracy(self, X, y_true)`

## 03_backpropagation.py
- Layer: utility
- Language: py
- Symbols:
  - `sigmoid` (function, line 33) `def sigmoid(z)`
  - `sigmoid_derivative` (function, line 38) `def sigmoid_derivative(z)`
  - `forward` (function, line 53) `def forward(X)`
  - `backward` (function, line 63) `def backward(X, y_true, y_pred, a1, z1, lr)`
  - `compute_metrics` (function, line 88) `def compute_metrics(y_pred, y_true)`

## 04_cnn_lenet.py
- Layer: utility
- Language: py
- Symbols:
  - `LeNet5Like` (class, line 45) `class LeNet5Like(Module)`
  - `__init__` (method, line 46) `def __init__(self)`
  - `forward` (method, line 60) `def forward(self, x)`

## 05_svm_rbf.py
- Layer: utility
- Language: py

## 06_lstm_char.py
- Layer: utility
- Language: py
- Symbols:
  - `create_batches` (function, line 63) `def create_batches(data, batch_size, seq_length)`
  - `CharLSTM` (class, line 89) `class CharLSTM(Module)`
  - `__init__` (method, line 90) `def __init__(self, vocab_size, hidden_size, num_layers, dropout)`
  - `forward` (method, line 102) `def forward(self, x, hidden)`

## 07_random_forest.py
- Layer: utility
- Language: py

## 08_vae_mnist.py
- Layer: utility
- Language: py
- Symbols:
  - `VAE` (class, line 40) `class VAE(Module)`
  - `vae_loss` (method, line 72) `def vae_loss(recon_x, x, mu, log_var)`
  - `__init__` (method, line 41) `def __init__(self, input_dim, hidden_dim, latent_dim)`
  - `encode` (method, line 51) `def encode(self, x)`
  - `reparameterize` (method, line 55) `def reparameterize(self, mu, log_var)`
  - `decode` (method, line 60) `def decode(self, z)`
  - `forward` (method, line 64) `def forward(self, x)`

## 09_transformer_mini.py
- Layer: utility
- Language: py
- Symbols:
  - `generate_copy_data` (function, line 29) `def generate_copy_data(num_samples, seq_len, vocab_size)`
  - `MultiHeadAttention` (class, line 51) `class MultiHeadAttention(Module)`
  - `FeedForward` (class, line 83) `class FeedForward(Module)`
  - `MiniTransformerEncoder` (class, line 96) `class MiniTransformerEncoder(Module)`
  - `MiniTransformer` (class, line 131) `class MiniTransformer(Module)`
  - `__init__` (method, line 52) `def __init__(self, d_model, num_heads, dropout)`
  - `forward` (method, line 63) `def forward(self, q, k, v, mask)`
  - `__init__` (method, line 84) `def __init__(self, d_model, d_ff, dropout)`
  - `forward` (method, line 90) `def forward(self, x)`
  - `__init__` (method, line 97) `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len)`
  - `_create_positional_encoding` (method, line 107) `def _create_positional_encoding(self, max_len, d_model)`
  - `forward` (method, line 115) `def forward(self, x)`
  - `__init__` (method, line 132) `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout)`
  - `forward` (method, line 137) `def forward(self, x)`

## 10_gan_mnist_lite.py
- Layer: utility
- Language: py
- Symbols:
  - `Generator` (class, line 37) `class Generator(Module)`
  - `Discriminator` (class, line 57) `class Discriminator(Module)`
  - `__init__` (method, line 38) `def __init__(self, latent_dim, img_size)`
  - `forward` (method, line 51) `def forward(self, z)`
  - `__init__` (method, line 58) `def __init__(self, img_size)`
  - `forward` (method, line 74) `def forward(self, x)`

## 11_bert_tiny.py
- Layer: utility
- Language: py
- Symbols:
  - `tokenize_sentence` (function, line 76) `def tokenize_sentence(sentence)`
  - `pad_sequence` (function, line 88) `def pad_sequence(seq, length, pad_value)`
  - `MultiHeadAttention` (class, line 107) `class MultiHeadAttention(Module)`
  - `FeedForward` (class, line 133) `class FeedForward(Module)`
  - `BERTLayer` (class, line 143) `class BERTLayer(Module)`
  - `TinyBERT` (class, line 158) `class TinyBERT(Module)`
  - `mask_tokens` (method, line 186) `def mask_tokens(inputs, vocab_size, mask_token_id, pad_token_id, mask_prob)`
  - `__init__` (method, line 108) `def __init__(self, d_model, num_heads, dropout)`
  - `forward` (method, line 119) `def forward(self, q, k, v, mask)`
  - `__init__` (method, line 134) `def __init__(self, d_model, d_ff, dropout)`
  - `forward` (method, line 140) `def forward(self, x)`
  - `__init__` (method, line 144) `def __init__(self, d_model, num_heads, d_ff, dropout)`
  - `forward` (method, line 151) `def forward(self, x, mask)`
  - `__init__` (method, line 159) `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len)`
  - `_create_positional_encoding` (method, line 167) `def _create_positional_encoding(self, max_len, d_model)`
  - `forward` (method, line 175) `def forward(self, x, mask)`

## 12_diffusion_minimal.py
- Doc: q_sample: Muestrea x_t dado x_0 y timestep t.
- Layer: utility
- Language: py
- Symbols:
  - `SimpleDiffusionNet` (class, line 54) `class SimpleDiffusionNet(Module)`
  - `q_sample` (method, line 81) `def q_sample(x_0, t, noise)`
  - `__init__` (method, line 55) `def __init__(self, in_channels, out_channels, hidden_dim)`
  - `forward` (method, line 65) `def forward(self, x, t)`

## 13_nested_hope.py
- Doc: Config: Configuración centralizada basada en el paper (Secciones 7-9)
- Layer: utility
- Language: py
- Symbols:
  - `Config` (class, line 14) `class Config`
  - `setup_device` (method, line 44) `def setup_device()`
  - `set_seed` (method, line 54) `def set_seed(seed)`
  - `DeltaGradientDescent` (class, line 66) `class DeltaGradientDescent`
  - `SelfModifyingMemory` (class, line 115) `class SelfModifyingMemory(Module)`
  - `ContinuumMemorySystem` (class, line 258) `class ContinuumMemorySystem(Module)`
  - `HopeModel` (class, line 337) `class HopeModel(Module)`
  - `HopeTrainer` (class, line 432) `class HopeTrainer`
  - `run_ablation_study` (method, line 564) `def run_ablation_study(config, device)`
  - `apply_update` (method, line 77) `def apply_update(grad, param, x_normalized, eta, alpha, lambda_norm)`
  - `__init__` (method, line 125) `def __init__(self, d_model, hidden_dim, chunk_size)`
  - `_make_memory_module` (method, line 162) `def _make_memory_module(self)`
  - `forward` (method, line 170) `def forward(self, x, prev_states)`
  - `__init__` (method, line 268) `def __init__(self, frequencies, d_model, hidden_dim, connection_type)`
  - `forward` (method, line 296) `def forward(self, x, global_step)`
  - `__init__` (method, line 344) `def __init__(self, vocab_size, d_model, cms_frequencies, mlp_hidden, chunk_size, enable_self_modifying, enable_cms)`
  - `reset_states` (method, line 390) `def reset_states(self)`
  - `forward` (method, line 394) `def forward(self, x, global_step, return_internals)`
  - `__init__` (method, line 437) `def __init__(self, model, config, device)`
  - `train_epoch` (method, line 461) `def train_epoch(self, train_loader, epoch, global_step)`
  - `evaluate` (method, line 536) `def evaluate(self, test_loader, global_step)`

## 13_nested_kearning_gpu.py
- Layer: utility
- Language: py
- Symbols:
  - `SelfModifyingMemory` (class, line 53) `class SelfModifyingMemory(Module)`
  - `ContinuumMemorySystem` (class, line 74) `class ContinuumMemorySystem(Module)`
  - `HopeModel` (class, line 97) `class HopeModel(Module)`
  - `__init__` (method, line 54) `def __init__(self, vocab_size, d_model, hidden_dim)`
  - `forward` (method, line 64) `def forward(self, x)`
  - `__init__` (method, line 75) `def __init__(self, frequencies, d_model, hidden_dim)`
  - `forward` (method, line 87) `def forward(self, x, global_step)`
  - `__init__` (method, line 98) `def __init__(self, vocab_size, d_model, cms_freqs, hidden_dim)`
  - `forward` (method, line 104) `def forward(self, x, global_step)`

## 13_nested_learning.py
- Doc: forward: update_mask: None o máscara booleana para actualizaciones condicionales Retorna logits...
- Layer: utility
- Language: py
- Symbols:
  - `SelfModifyingMemory` (class, line 51) `class SelfModifyingMemory(Module)`
  - `ContinuumMemorySystem` (class, line 86) `class ContinuumMemorySystem(Module)`
  - `HopeModel` (class, line 114) `class HopeModel(Module)`
  - `__init__` (method, line 52) `def __init__(self, vocab_size, d_model, hidden_dim)`
  - `forward` (method, line 62) `def forward(self, x, update_mask)`
  - `__init__` (method, line 87) `def __init__(self, levels, d_model, hidden_dim)`
  - `forward` (method, line 100) `def forward(self, x, global_step)`
  - `__init__` (method, line 115) `def __init__(self, vocab_size, d_model, cms_levels, mlp_hidden)`
  - `forward` (method, line 122) `def forward(self, x, global_step)`


Next: [KB_root_p2.md](KB_root_p2.md)

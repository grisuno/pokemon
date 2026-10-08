# Symbols

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `mcculloch_pitts_neuron` | function | `01_mcculloch_pitts.py:7` | `def mcculloch_pitts_neuron(inputs, weights, threshold)` |
| `Config` | class | `01_topobrain_cou_v2.py:25` | `class Config` |
| `StableContinuumMemoryCell` | class | `01_topobrain_cou_v2.py:147` | `class StableContinuumMemoryCell(Module)` |
| `StableSupConLoss` | class | `01_topobrain_cou_v2.py:116` | `class StableSupConLoss(Module)` |
| `StableSymbioticBasisRefinement` | class | `01_topobrain_cou_v2.py:231` | `class StableSymbioticBasisRefinement(Module)` |
| `StableTopoBrain` | class | `01_topobrain_cou_v2.py:345` | `class StableTopoBrain(Module)` |
| `StableTopologyManager` | class | `01_topobrain_cou_v2.py:267` | `class StableTopologyManager` |
| `__init__` | method | `01_topobrain_cou_v2.py:118` | `def __init__(self, temperature)` |
| `__init__` | method | `01_topobrain_cou_v2.py:149` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `01_topobrain_cou_v2.py:233` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `01_topobrain_cou_v2.py:269` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `01_topobrain_cou_v2.py:347` | `def __init__(self, config)` |
| `_init_weights` | method | `01_topobrain_cou_v2.py:400` | `def _init_weights(self)` |
| `evaluate_model` | method | `01_topobrain_cou_v2.py:584` | `def evaluate_model(model, loader, config, adversarial, controls)` |
| `forward` | method | `01_topobrain_cou_v2.py:123` | `def forward(self, features, labels)` |
| `forward` | method | `01_topobrain_cou_v2.py:181` | `def forward(self, x, controls)` |
| `forward` | method | `01_topobrain_cou_v2.py:242` | `def forward(self, x)` |
| `forward` | method | `01_topobrain_cou_v2.py:408` | `def forward(self, x, controls)` |
| `get_adjacency` | method | `01_topobrain_cou_v2.py:288` | `def get_adjacency(self, plasticity)` |
| `get_density` | method | `01_topobrain_cou_v2.py:337` | `def get_density(self)` |
| `get_tabular_loaders` | method | `01_topobrain_cou_v2.py:77` | `def get_tabular_loaders(config)` |
| `get_topology_config` | method | `01_topobrain_cou_v2.py:59` | `def get_topology_config(self)` |
| `prune_topology` | method | `01_topobrain_cou_v2.py:317` | `def prune_topology(self, current_density, epoch)` |
| `run_scientific_ablation` | method | `01_topobrain_cou_v2.py:632` | `def run_scientific_ablation()` |
| `seed_everything` | method | `01_topobrain_cou_v2.py:70` | `def seed_everything(seed)` |
| `stable_pgd_attack` | method | `01_topobrain_cou_v2.py:477` | `def stable_pgd_attack(model, x, y, eps, steps, controls)` |
| `to_dict` | method | `01_topobrain_cou_v2.py:56` | `def to_dict(self)` |
| `train_epoch` | method | `01_topobrain_cou_v2.py:520` | `def train_epoch(model, loader, optimizer, config, epoch, controls)` |
| `Config` | class | `01_topobrain_cpu.py:21` | `class Config` |
| `ContinuumMemoryCell` | class | `01_topobrain_cpu.py:109` | `class ContinuumMemoryCell(Module)` |
| `SupConLoss` | class | `01_topobrain_cpu.py:94` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `01_topobrain_cpu.py:143` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainTabular` | class | `01_topobrain_cpu.py:166` | `class TopoBrainTabular(Module)` |
| `__init__` | method | `01_topobrain_cpu.py:95` | `def __init__(self, temperature)` |
| `__init__` | method | `01_topobrain_cpu.py:110` | `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate)` |
| `__init__` | method | `01_topobrain_cpu.py:144` | `def __init__(self, dim)` |
| `__init__` | method | `01_topobrain_cpu.py:167` | `def __init__(self, config)` |
| `evaluate_adv` | method | `01_topobrain_cpu.py:323` | `def evaluate_adv(loader, eps, steps)` |
| `forward` | method | `01_topobrain_cpu.py:98` | `def forward(self, features, labels)` |
| `forward` | method | `01_topobrain_cpu.py:124` | `def forward(self, x, controls)` |
| `forward` | method | `01_topobrain_cpu.py:151` | `def forward(self, x)` |
| `forward` | method | `01_topobrain_cpu.py:203` | `def forward(self, x, controls)` |
| `generate_ablation_configs` | method | `01_topobrain_cpu.py:263` | `def generate_ablation_configs(base_config)` |
| `get_adj` | method | `01_topobrain_cpu.py:198` | `def get_adj(self)` |
| `get_tabular_loaders` | method | `01_topobrain_cpu.py:69` | `def get_tabular_loaders(config)` |
| `pgd_attack` | method | `01_topobrain_cpu.py:247` | `def pgd_attack(model, x, y, eps, steps, controls)` |
| `run_ablation` | method | `01_topobrain_cpu.py:337` | `def run_ablation()` |
| `seed_everything` | method | `01_topobrain_cpu.py:63` | `def seed_everything(seed)` |
| `to_dict` | method | `01_topobrain_cpu.py:57` | `def to_dict(self)` |
| `train_and_evaluate` | method | `01_topobrain_cpu.py:293` | `def train_and_evaluate(config, name)` |
| `AdaptiveCombinatorialComplexLayer` | class | `01_topobrain_cpu_v3.py:227` | `class AdaptiveCombinatorialComplexLayer(Module)` |
| `Config` | class | `01_topobrain_cpu_v3.py:24` | `class Config` |
| `ContinuumMemoryCell` | class | `01_topobrain_cpu_v3.py:108` | `class ContinuumMemoryCell(Module)` |
| `PrefrontalOrchestrator` | class | `01_topobrain_cpu_v3.py:165` | `class PrefrontalOrchestrator(Module)` |
| `SupConLoss` | class | `01_topobrain_cpu_v3.py:93` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `01_topobrain_cpu_v3.py:145` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainTabular` | class | `01_topobrain_cpu_v3.py:320` | `class TopoBrainTabular(Module)` |
| `__init__` | method | `01_topobrain_cpu_v3.py:94` | `def __init__(self, temperature)` |
| `__init__` | method | `01_topobrain_cpu_v3.py:109` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `01_topobrain_cpu_v3.py:146` | `def __init__(self, dim)` |
| `__init__` | method | `01_topobrain_cpu_v3.py:166` | `def __init__(self, config)` |
| `__init__` | method | `01_topobrain_cpu_v3.py:228` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)` |
| `__init__` | method | `01_topobrain_cpu_v3.py:321` | `def __init__(self, config)` |
| `_initialize_memories` | method | `01_topobrain_cpu_v3.py:352` | `def _initialize_memories(self)` |
| `compute_topology_metrics` | method | `01_topobrain_cpu_v3.py:416` | `def compute_topology_metrics(model, config)` |
| `evaluate_adv` | method | `01_topobrain_cpu_v3.py:593` | `def evaluate_adv(loader, eps, steps)` |
| `forward` | method | `01_topobrain_cpu_v3.py:97` | `def forward(self, features, labels)` |
| `forward` | method | `01_topobrain_cpu_v3.py:122` | `def forward(self, x, controls)` |
| `forward` | method | `01_topobrain_cpu_v3.py:153` | `def forward(self, x)` |
| `forward` | method | `01_topobrain_cpu_v3.py:183` | `def forward(self, metrics)` |
| `forward` | method | `01_topobrain_cpu_v3.py:269` | `def forward(self, x, controls)` |
| `forward` | method | `01_topobrain_cpu_v3.py:363` | `def forward(self, x, controls, prev_states)` |
| `get_adj` | method | `01_topobrain_cpu_v3.py:264` | `def get_adj(self)` |
| `get_tabular_loaders` | method | `01_topobrain_cpu_v3.py:69` | `def get_tabular_loaders(config)` |
| `pgd_attack` | method | `01_topobrain_cpu_v3.py:395` | `def pgd_attack(model, x, y, eps, steps)` |
| `prune_topology` | method | `01_topobrain_cpu_v3.py:450` | `def prune_topology(model, config, controls)` |
| `reset_context` | method | `01_topobrain_cpu_v3.py:221` | `def reset_context(self)` |
| `run_ablation` | method | `01_topobrain_cpu_v3.py:640` | `def run_ablation()` |
| `seed_everything` | method | `01_topobrain_cpu_v3.py:63` | `def seed_everything(seed)` |
| `to_dict` | method | `01_topobrain_cpu_v3.py:57` | `def to_dict(self)` |
| `train_and_evaluate` | method | `01_topobrain_cpu_v3.py:482` | `def train_and_evaluate(config, run_name)` |
| `Config` | class | `01_topobrain_cpu_v4.py:25` | `class Config` |
| `StableContinuumMemoryCell` | class | `01_topobrain_cpu_v4.py:147` | `class StableContinuumMemoryCell(Module)` |
| `StableSupConLoss` | class | `01_topobrain_cpu_v4.py:116` | `class StableSupConLoss(Module)` |
| `StableSymbioticBasisRefinement` | class | `01_topobrain_cpu_v4.py:231` | `class StableSymbioticBasisRefinement(Module)` |
| `StableTopoBrain` | class | `01_topobrain_cpu_v4.py:345` | `class StableTopoBrain(Module)` |
| `StableTopologyManager` | class | `01_topobrain_cpu_v4.py:267` | `class StableTopologyManager` |
| `__init__` | method | `01_topobrain_cpu_v4.py:118` | `def __init__(self, temperature)` |
| `__init__` | method | `01_topobrain_cpu_v4.py:149` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `01_topobrain_cpu_v4.py:233` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `01_topobrain_cpu_v4.py:269` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `01_topobrain_cpu_v4.py:347` | `def __init__(self, config)` |
| `_init_weights` | method | `01_topobrain_cpu_v4.py:400` | `def _init_weights(self)` |
| `evaluate_model` | method | `01_topobrain_cpu_v4.py:583` | `def evaluate_model(model, loader, config, adversarial, controls)` |
| `forward` | method | `01_topobrain_cpu_v4.py:123` | `def forward(self, features, labels)` |
| `forward` | method | `01_topobrain_cpu_v4.py:181` | `def forward(self, x, controls)` |
| `forward` | method | `01_topobrain_cpu_v4.py:242` | `def forward(self, x)` |
| `forward` | method | `01_topobrain_cpu_v4.py:408` | `def forward(self, x, controls)` |
| `get_adjacency` | method | `01_topobrain_cpu_v4.py:288` | `def get_adjacency(self, plasticity)` |
| `get_density` | method | `01_topobrain_cpu_v4.py:337` | `def get_density(self)` |
| `get_tabular_loaders` | method | `01_topobrain_cpu_v4.py:77` | `def get_tabular_loaders(config)` |
| `get_topology_config` | method | `01_topobrain_cpu_v4.py:59` | `def get_topology_config(self)` |
| `prune_topology` | method | `01_topobrain_cpu_v4.py:317` | `def prune_topology(self, current_density, epoch)` |
| `run_scientific_ablation` | method | `01_topobrain_cpu_v4.py:628` | `def run_scientific_ablation()` |
| `seed_everything` | method | `01_topobrain_cpu_v4.py:70` | `def seed_everything(seed)` |
| `stable_pgd_attack` | method | `01_topobrain_cpu_v4.py:475` | `def stable_pgd_attack(model, x, y, eps, steps, controls)` |
| `to_dict` | method | `01_topobrain_cpu_v4.py:56` | `def to_dict(self)` |
| `train_epoch` | method | `01_topobrain_cpu_v4.py:518` | `def train_epoch(model, loader, optimizer, config, epoch, controls)` |
| `Config` | class | `01_topobrain_cpu_v5.py:25` | `class Config` |
| `StableContinuumMemoryCell` | class | `01_topobrain_cpu_v5.py:147` | `class StableContinuumMemoryCell(Module)` |
| `StableSupConLoss` | class | `01_topobrain_cpu_v5.py:116` | `class StableSupConLoss(Module)` |
| `StableSymbioticBasisRefinement` | class | `01_topobrain_cpu_v5.py:231` | `class StableSymbioticBasisRefinement(Module)` |
| `StableTopoBrain` | class | `01_topobrain_cpu_v5.py:345` | `class StableTopoBrain(Module)` |
| `StableTopologyManager` | class | `01_topobrain_cpu_v5.py:267` | `class StableTopologyManager` |
| `__init__` | method | `01_topobrain_cpu_v5.py:118` | `def __init__(self, temperature)` |
| `__init__` | method | `01_topobrain_cpu_v5.py:149` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `01_topobrain_cpu_v5.py:233` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `01_topobrain_cpu_v5.py:269` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `01_topobrain_cpu_v5.py:347` | `def __init__(self, config)` |
| `_init_weights` | method | `01_topobrain_cpu_v5.py:400` | `def _init_weights(self)` |
| `evaluate_model` | method | `01_topobrain_cpu_v5.py:583` | `def evaluate_model(model, loader, config, adversarial, controls)` |
| `forward` | method | `01_topobrain_cpu_v5.py:123` | `def forward(self, features, labels)` |
| `forward` | method | `01_topobrain_cpu_v5.py:181` | `def forward(self, x, controls)` |
| `forward` | method | `01_topobrain_cpu_v5.py:242` | `def forward(self, x)` |
| `forward` | method | `01_topobrain_cpu_v5.py:408` | `def forward(self, x, controls)` |
| `get_adjacency` | method | `01_topobrain_cpu_v5.py:288` | `def get_adjacency(self, plasticity)` |
| `get_density` | method | `01_topobrain_cpu_v5.py:337` | `def get_density(self)` |
| `get_tabular_loaders` | method | `01_topobrain_cpu_v5.py:77` | `def get_tabular_loaders(config)` |
| `get_topology_config` | method | `01_topobrain_cpu_v5.py:59` | `def get_topology_config(self)` |
| `prune_topology` | method | `01_topobrain_cpu_v5.py:317` | `def prune_topology(self, current_density, epoch)` |
| `run_scientific_ablation` | method | `01_topobrain_cpu_v5.py:708` | `def run_scientific_ablation()` |
| `seed_everything` | method | `01_topobrain_cpu_v5.py:70` | `def seed_everything(seed)` |
| `stable_pgd_attack` | method | `01_topobrain_cpu_v5.py:475` | `def stable_pgd_attack(model, x, y, eps, steps, controls)` |
| `to_dict` | method | `01_topobrain_cpu_v5.py:56` | `def to_dict(self)` |
| `train_epoch` | method | `01_topobrain_cpu_v5.py:518` | `def train_epoch(model, loader, optimizer, config, epoch, controls)` |
| `AblationMatrix` | class | `01_topobrain_cpu_v6.py:597` | `class AblationMatrix` |
| `MicroConfig` | class | `01_topobrain_cpu_v6.py:33` | `class MicroConfig` |
| `MicroContinuumCell` | class | `01_topobrain_cpu_v6.py:163` | `class MicroContinuumCell(Module)` |
| `MicroSupConLoss` | class | `01_topobrain_cpu_v6.py:137` | `class MicroSupConLoss(Module)` |
| `MicroSymbioticBasis` | class | `01_topobrain_cpu_v6.py:214` | `class MicroSymbioticBasis(Module)` |
| `MicroTopoBrain` | class | `01_topobrain_cpu_v6.py:286` | `class MicroTopoBrain(Module)` |
| `MicroTopology` | class | `01_topobrain_cpu_v6.py:253` | `class MicroTopology` |
| `ScientificAnalyzer` | class | `01_topobrain_cpu_v6.py:744` | `class ScientificAnalyzer` |
| `__init__` | method | `01_topobrain_cpu_v6.py:139` | `def __init__(self, temperature)` |
| `__init__` | method | `01_topobrain_cpu_v6.py:168` | `def __init__(self, dim)` |
| `__init__` | method | `01_topobrain_cpu_v6.py:219` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `01_topobrain_cpu_v6.py:258` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `01_topobrain_cpu_v6.py:301` | `def __init__(self, config)` |
| `_init_weights` | method | `01_topobrain_cpu_v6.py:353` | `def _init_weights(self)` |
| `component_signature` | method | `01_topobrain_cpu_v6.py:82` | `def component_signature(self)` |
| `compute_effect_size` | method | `01_topobrain_cpu_v6.py:126` | `def compute_effect_size(group1, group2)` |
| `compute_statistics` | method | `01_topobrain_cpu_v6.py:748` | `def compute_statistics(results_list)` |
| `count_parameters` | method | `01_topobrain_cpu_v6.py:360` | `def count_parameters(self)` |
| `detect_synergy` | method | `01_topobrain_cpu_v6.py:786` | `def detect_synergy(pair_pgd, comp_a_pgd, comp_b_pgd, baseline_pgd)` |
| `evaluate_micro` | method | `01_topobrain_cpu_v6.py:517` | `def evaluate_micro(model, loader, config, adversarial)` |
| `forward` | method | `01_topobrain_cpu_v6.py:144` | `def forward(self, features, labels)` |
| `forward` | method | `01_topobrain_cpu_v6.py:184` | `def forward(self, x, plasticity)` |
| `forward` | method | `01_topobrain_cpu_v6.py:232` | `def forward(self, x)` |
| `forward` | method | `01_topobrain_cpu_v6.py:364` | `def forward(self, x, plasticity)` |
| `get_adjacency` | method | `01_topobrain_cpu_v6.py:276` | `def get_adjacency(self, plasticity)` |
| `get_complete_matrix` | method | `01_topobrain_cpu_v6.py:730` | `def get_complete_matrix(cls)` |
| `get_micro_dataset` | method | `01_topobrain_cpu_v6.py:104` | `def get_micro_dataset(config)` |
| `level1_isolated` | method | `01_topobrain_cpu_v6.py:628` | `def level1_isolated()` |
| `level2a_pairs` | method | `01_topobrain_cpu_v6.py:641` | `def level2a_pairs()` |
| `level2b_strategic_triads` | method | `01_topobrain_cpu_v6.py:665` | `def level2b_strategic_triads()` |
| `level3_full_model` | method | `01_topobrain_cpu_v6.py:719` | `def level3_full_model()` |
| `level3_inverse_ablation` | method | `01_topobrain_cpu_v6.py:694` | `def level3_inverse_ablation()` |
| `micro_pgd_attack` | method | `01_topobrain_cpu_v6.py:432` | `def micro_pgd_attack(model, x, y, eps, steps, plasticity)` |
| `rank_components_by_criticality` | method | `01_topobrain_cpu_v6.py:809` | `def rank_components_by_criticality(full_pgd, ablation_results)` |
| `run_scientific_ablation_study` | method | `01_topobrain_cpu_v6.py:850` | `def run_scientific_ablation_study()` |
| `seed_everything` | method | `01_topobrain_cpu_v6.py:95` | `def seed_everything(seed)` |
| `to_dict` | method | `01_topobrain_cpu_v6.py:79` | `def to_dict(self)` |
| `train_epoch_micro` | method | `01_topobrain_cpu_v6.py:462` | `def train_epoch_micro(model, loader, optimizer, config, epoch)` |
| `train_with_cv` | method | `01_topobrain_cpu_v6.py:539` | `def train_with_cv(config, dataset, cv_folds)` |
| `ttest_vs_baseline` | method | `01_topobrain_cpu_v6.py:774` | `def ttest_vs_baseline(exp_scores, baseline_scores)` |
| `Config` | class | `01_topobrain_cpu_v7.py:42` | `class Config` |
| `DynamicTopology` | class | `01_topobrain_cpu_v7.py:124` | `class DynamicTopology(Module)` |
| `SymbioticBasis` | class | `01_topobrain_cpu_v7.py:94` | `class SymbioticBasis(Module)` |
| `TopoBrainCPU` | class | `01_topobrain_cpu_v7.py:186` | `class TopoBrainCPU(Module)` |
| `__init__` | method | `01_topobrain_cpu_v7.py:95` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `01_topobrain_cpu_v7.py:125` | `def __init__(self, num_nodes, grid_size, config)` |
| `__init__` | method | `01_topobrain_cpu_v7.py:187` | `def __init__(self, config)` |
| `_create_grid_mask` | method | `01_topobrain_cpu_v7.py:136` | `def _create_grid_mask(self)` |
| `_init_weights` | method | `01_topobrain_cpu_v7.py:216` | `def _init_weights(self)` |
| `count_parameters` | method | `01_topobrain_cpu_v7.py:223` | `def count_parameters(self)` |
| `evaluate` | method | `01_topobrain_cpu_v7.py:321` | `def evaluate(model, loader, config, device, adversarial)` |
| `forward` | method | `01_topobrain_cpu_v7.py:106` | `def forward(self, x)` |
| `forward` | method | `01_topobrain_cpu_v7.py:226` | `def forward(self, x, plasticity)` |
| `get_adjacency` | method | `01_topobrain_cpu_v7.py:153` | `def get_adjacency(self, plasticity)` |
| `get_dataset` | method | `01_topobrain_cpu_v7.py:351` | `def get_dataset(config)` |
| `get_density` | method | `01_topobrain_cpu_v7.py:178` | `def get_density(self)` |
| `main` | method | `01_topobrain_cpu_v7.py:393` | `def main()` |
| `pgd_attack` | method | `01_topobrain_cpu_v7.py:256` | `def pgd_attack(model, x, y, eps, steps, plasticity)` |
| `prune_connections` | method | `01_topobrain_cpu_v7.py:158` | `def prune_connections(self, threshold)` |
| `setup_device` | function | `01_topobrain_cpu_v7.py:24` | `def setup_device()` |
| `to_dict` | method | `01_topobrain_cpu_v7.py:86` | `def to_dict(self)` |
| `train_epoch` | method | `01_topobrain_cpu_v7.py:285` | `def train_epoch(model, loader, optimizer, config, epoch, device)` |
| `Config` | class | `01_topobrain_cpu_v8.py:28` | `class Config` |
| `DynamicTopology` | class | `01_topobrain_cpu_v8.py:88` | `class DynamicTopology(Module)` |
| `SymbioticBasis` | class | `01_topobrain_cpu_v8.py:62` | `class SymbioticBasis(Module)` |
| `TopoBrainReal` | class | `01_topobrain_cpu_v8.py:144` | `class TopoBrainReal(Module)` |
| `Wrapper` | class | `01_topobrain_cpu_v8.py:424` | `class Wrapper(Module)` |
| `__init__` | method | `01_topobrain_cpu_v8.py:63` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `01_topobrain_cpu_v8.py:89` | `def __init__(self, num_nodes, grid_size, config)` |
| `__init__` | method | `01_topobrain_cpu_v8.py:145` | `def __init__(self, config)` |
| `__init__` | method | `01_topobrain_cpu_v8.py:425` | `def __init__(self, m)` |
| `_create_grid_mask` | method | `01_topobrain_cpu_v8.py:98` | `def _create_grid_mask(self)` |
| `_init_weights` | method | `01_topobrain_cpu_v8.py:174` | `def _init_weights(self)` |
| `export_for_onnxruntime` | method | `01_topobrain_cpu_v8.py:410` | `def export_for_onnxruntime(model)` |
| `forward` | method | `01_topobrain_cpu_v8.py:76` | `def forward(self, x)` |
| `forward` | method | `01_topobrain_cpu_v8.py:181` | `def forward(self, x, plasticity)` |
| `forward` | method | `01_topobrain_cpu_v8.py:429` | `def forward(self, x)` |
| `forward_with_metrics` | method | `01_topobrain_cpu_v8.py:210` | `def forward_with_metrics(self, x, plasticity)` |
| `get_adjacency` | method | `01_topobrain_cpu_v8.py:115` | `def get_adjacency(self, plasticity)` |
| `get_density` | method | `01_topobrain_cpu_v8.py:140` | `def get_density(self)` |
| `main` | method | `01_topobrain_cpu_v8.py:547` | `def main()` |
| `pgd_attack` | method | `01_topobrain_cpu_v8.py:223` | `def pgd_attack(model, x, y, eps, steps, plasticity)` |
| `prune_connections` | method | `01_topobrain_cpu_v8.py:120` | `def prune_connections(self, threshold)` |
| `test_with_onnxruntime` | method | `01_topobrain_cpu_v8.py:465` | `def test_with_onnxruntime(X_test, y_test)` |
| `train_topobrain` | method | `01_topobrain_cpu_v8.py:252` | `def train_topobrain(config)` |
| `DynamicTopology` | class | `01_topobrain_ganador_gpu_v1.py:179` | `class DynamicTopology(Module)` |
| `GPUConfig` | class | `01_topobrain_ganador_gpu_v1.py:70` | `class GPUConfig` |
| `GPUSymbioticBasis` | class | `01_topobrain_ganador_gpu_v1.py:127` | `class GPUSymbioticBasis(Module)` |
| `TopoBrainGPU` | class | `01_topobrain_ganador_gpu_v1.py:278` | `class TopoBrainGPU(Module)` |
| `__init__` | method | `01_topobrain_ganador_gpu_v1.py:132` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `01_topobrain_ganador_gpu_v1.py:184` | `def __init__(self, num_nodes, grid_size, config)` |
| `__init__` | method | `01_topobrain_ganador_gpu_v1.py:291` | `def __init__(self, config)` |
| `_create_grid_mask` | method | `01_topobrain_ganador_gpu_v1.py:200` | `def _create_grid_mask(self)` |
| `_init_weights` | method | `01_topobrain_ganador_gpu_v1.py:335` | `def _init_weights(self)` |
| `count_parameters` | method | `01_topobrain_ganador_gpu_v1.py:343` | `def count_parameters(self)` |
| `evaluate_gpu` | method | `01_topobrain_ganador_gpu_v1.py:517` | `def evaluate_gpu(model, loader, config, device, adversarial)` |
| `forward` | method | `01_topobrain_ganador_gpu_v1.py:147` | `def forward(self, x)` |
| `forward` | method | `01_topobrain_ganador_gpu_v1.py:347` | `def forward(self, x, plasticity)` |
| `get_adjacency` | method | `01_topobrain_ganador_gpu_v1.py:219` | `def get_adjacency(self, plasticity)` |
| `get_density` | method | `01_topobrain_ganador_gpu_v1.py:269` | `def get_density(self)` |
| `get_gpu_dataset` | method | `01_topobrain_ganador_gpu_v1.py:547` | `def get_gpu_dataset(config)` |
| `main` | method | `01_topobrain_ganador_gpu_v1.py:591` | `def main()` |
| `pgd_attack_gpu` | method | `01_topobrain_ganador_gpu_v1.py:405` | `def pgd_attack_gpu(model, x, y, eps, steps, plasticity)` |
| `prune_connections` | method | `01_topobrain_ganador_gpu_v1.py:237` | `def prune_connections(self, threshold)` |
| `setup_amd_device` | function | `01_topobrain_ganador_gpu_v1.py:31` | `def setup_amd_device()` |
| `to_dict` | method | `01_topobrain_ganador_gpu_v1.py:119` | `def to_dict(self)` |
| `train_epoch_gpu` | method | `01_topobrain_ganador_gpu_v1.py:461` | `def train_epoch_gpu(model, loader, optimizer, config, epoch, device)` |
| `Perceptron` | class | `02_perceptron.py:31` | `class Perceptron` |
| `__init__` | method | `02_perceptron.py:32` | `def __init__(self, input_dim, learning_rate)` |
| `accuracy` | method | `02_perceptron.py:51` | `def accuracy(self, X, y_true)` |
| `predict` | method | `02_perceptron.py:37` | `def predict(self, X)` |
| `train_step` | method | `02_perceptron.py:42` | `def train_step(self, X_batch, y_batch)` |
| `backward` | function | `03_backpropagation.py:63` | `def backward(X, y_true, y_pred, a1, z1, lr)` |
| `compute_metrics` | function | `03_backpropagation.py:88` | `def compute_metrics(y_pred, y_true)` |
| `forward` | function | `03_backpropagation.py:53` | `def forward(X)` |
| `sigmoid` | function | `03_backpropagation.py:33` | `def sigmoid(z)` |
| `sigmoid_derivative` | function | `03_backpropagation.py:38` | `def sigmoid_derivative(z)` |
| `LeNet5Like` | class | `04_cnn_lenet.py:45` | `class LeNet5Like(Module)` |
| `__init__` | method | `04_cnn_lenet.py:46` | `def __init__(self)` |
| `forward` | method | `04_cnn_lenet.py:60` | `def forward(self, x)` |
| `CharLSTM` | class | `06_lstm_char.py:89` | `class CharLSTM(Module)` |
| `__init__` | method | `06_lstm_char.py:90` | `def __init__(self, vocab_size, hidden_size, num_layers, dropout)` |
| `create_batches` | function | `06_lstm_char.py:63` | `def create_batches(data, batch_size, seq_length)` |
| `forward` | method | `06_lstm_char.py:102` | `def forward(self, x, hidden)` |
| `VAE` | class | `08_vae_mnist.py:40` | `class VAE(Module)` |
| `__init__` | method | `08_vae_mnist.py:41` | `def __init__(self, input_dim, hidden_dim, latent_dim)` |
| `decode` | method | `08_vae_mnist.py:60` | `def decode(self, z)` |
| `encode` | method | `08_vae_mnist.py:51` | `def encode(self, x)` |
| `forward` | method | `08_vae_mnist.py:64` | `def forward(self, x)` |
| `reparameterize` | method | `08_vae_mnist.py:55` | `def reparameterize(self, mu, log_var)` |
| `vae_loss` | method | `08_vae_mnist.py:72` | `def vae_loss(recon_x, x, mu, log_var)` |
| `FeedForward` | class | `09_transformer_mini.py:83` | `class FeedForward(Module)` |
| `MiniTransformer` | class | `09_transformer_mini.py:131` | `class MiniTransformer(Module)` |
| `MiniTransformerEncoder` | class | `09_transformer_mini.py:96` | `class MiniTransformerEncoder(Module)` |
| `MultiHeadAttention` | class | `09_transformer_mini.py:51` | `class MultiHeadAttention(Module)` |
| `__init__` | method | `09_transformer_mini.py:52` | `def __init__(self, d_model, num_heads, dropout)` |
| `__init__` | method | `09_transformer_mini.py:84` | `def __init__(self, d_model, d_ff, dropout)` |
| `__init__` | method | `09_transformer_mini.py:97` | `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len)` |
| `__init__` | method | `09_transformer_mini.py:132` | `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout)` |
| `_create_positional_encoding` | method | `09_transformer_mini.py:107` | `def _create_positional_encoding(self, max_len, d_model)` |
| `forward` | method | `09_transformer_mini.py:63` | `def forward(self, q, k, v, mask)` |
| `forward` | method | `09_transformer_mini.py:90` | `def forward(self, x)` |
| `forward` | method | `09_transformer_mini.py:115` | `def forward(self, x)` |
| `forward` | method | `09_transformer_mini.py:137` | `def forward(self, x)` |
| `generate_copy_data` | function | `09_transformer_mini.py:29` | `def generate_copy_data(num_samples, seq_len, vocab_size)` |
| `Discriminator` | class | `10_gan_mnist_lite.py:57` | `class Discriminator(Module)` |
| `Generator` | class | `10_gan_mnist_lite.py:37` | `class Generator(Module)` |
| `__init__` | method | `10_gan_mnist_lite.py:38` | `def __init__(self, latent_dim, img_size)` |
| `__init__` | method | `10_gan_mnist_lite.py:58` | `def __init__(self, img_size)` |
| `forward` | method | `10_gan_mnist_lite.py:51` | `def forward(self, z)` |
| `forward` | method | `10_gan_mnist_lite.py:74` | `def forward(self, x)` |
| `BERTLayer` | class | `11_bert_tiny.py:143` | `class BERTLayer(Module)` |
| `FeedForward` | class | `11_bert_tiny.py:133` | `class FeedForward(Module)` |
| `MultiHeadAttention` | class | `11_bert_tiny.py:107` | `class MultiHeadAttention(Module)` |
| `TinyBERT` | class | `11_bert_tiny.py:158` | `class TinyBERT(Module)` |
| `__init__` | method | `11_bert_tiny.py:108` | `def __init__(self, d_model, num_heads, dropout)` |
| `__init__` | method | `11_bert_tiny.py:134` | `def __init__(self, d_model, d_ff, dropout)` |
| `__init__` | method | `11_bert_tiny.py:144` | `def __init__(self, d_model, num_heads, d_ff, dropout)` |
| `__init__` | method | `11_bert_tiny.py:159` | `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len)` |
| `_create_positional_encoding` | method | `11_bert_tiny.py:167` | `def _create_positional_encoding(self, max_len, d_model)` |
| `forward` | method | `11_bert_tiny.py:119` | `def forward(self, q, k, v, mask)` |
| `forward` | method | `11_bert_tiny.py:140` | `def forward(self, x)` |
| `forward` | method | `11_bert_tiny.py:151` | `def forward(self, x, mask)` |
| `forward` | method | `11_bert_tiny.py:175` | `def forward(self, x, mask)` |
| `mask_tokens` | method | `11_bert_tiny.py:186` | `def mask_tokens(inputs, vocab_size, mask_token_id, pad_token_id, mask_prob)` |
| `pad_sequence` | function | `11_bert_tiny.py:88` | `def pad_sequence(seq, length, pad_value)` |
| `tokenize_sentence` | function | `11_bert_tiny.py:76` | `def tokenize_sentence(sentence)` |
| `SimpleDiffusionNet` | class | `12_diffusion_minimal.py:54` | `class SimpleDiffusionNet(Module)` |
| `__init__` | method | `12_diffusion_minimal.py:55` | `def __init__(self, in_channels, out_channels, hidden_dim)` |
| `forward` | method | `12_diffusion_minimal.py:65` | `def forward(self, x, t)` |
| `q_sample` | method | `12_diffusion_minimal.py:81` | `def q_sample(x_0, t, noise)` |
| `Config` | class | `13_nested_hope.py:14` | `class Config` |
| `ContinuumMemorySystem` | class | `13_nested_hope.py:258` | `class ContinuumMemorySystem(Module)` |
| `DeltaGradientDescent` | class | `13_nested_hope.py:66` | `class DeltaGradientDescent` |
| `HopeModel` | class | `13_nested_hope.py:337` | `class HopeModel(Module)` |
| `HopeTrainer` | class | `13_nested_hope.py:432` | `class HopeTrainer` |
| `SelfModifyingMemory` | class | `13_nested_hope.py:115` | `class SelfModifyingMemory(Module)` |
| `__init__` | method | `13_nested_hope.py:125` | `def __init__(self, d_model, hidden_dim, chunk_size)` |
| `__init__` | method | `13_nested_hope.py:268` | `def __init__(self, frequencies, d_model, hidden_dim, connection_type)` |
| `__init__` | method | `13_nested_hope.py:344` | `def __init__(self, vocab_size, d_model, cms_frequencies, mlp_hidden, chunk_size, enable_self_modifying, enable_cms)` |
| `__init__` | method | `13_nested_hope.py:437` | `def __init__(self, model, config, device)` |
| `_make_memory_module` | method | `13_nested_hope.py:162` | `def _make_memory_module(self)` |
| `apply_update` | method | `13_nested_hope.py:77` | `def apply_update(grad, param, x_normalized, eta, alpha, lambda_norm)` |
| `evaluate` | method | `13_nested_hope.py:536` | `def evaluate(self, test_loader, global_step)` |
| `forward` | method | `13_nested_hope.py:170` | `def forward(self, x, prev_states)` |
| `forward` | method | `13_nested_hope.py:296` | `def forward(self, x, global_step)` |
| `forward` | method | `13_nested_hope.py:394` | `def forward(self, x, global_step, return_internals)` |
| `reset_states` | method | `13_nested_hope.py:390` | `def reset_states(self)` |
| `run_ablation_study` | method | `13_nested_hope.py:564` | `def run_ablation_study(config, device)` |
| `set_seed` | method | `13_nested_hope.py:54` | `def set_seed(seed)` |
| `setup_device` | method | `13_nested_hope.py:44` | `def setup_device()` |
| `train_epoch` | method | `13_nested_hope.py:461` | `def train_epoch(self, train_loader, epoch, global_step)` |
| `ContinuumMemorySystem` | class | `13_nested_kearning_gpu.py:74` | `class ContinuumMemorySystem(Module)` |
| `HopeModel` | class | `13_nested_kearning_gpu.py:97` | `class HopeModel(Module)` |
| `SelfModifyingMemory` | class | `13_nested_kearning_gpu.py:53` | `class SelfModifyingMemory(Module)` |
| `__init__` | method | `13_nested_kearning_gpu.py:54` | `def __init__(self, vocab_size, d_model, hidden_dim)` |
| `__init__` | method | `13_nested_kearning_gpu.py:75` | `def __init__(self, frequencies, d_model, hidden_dim)` |
| `__init__` | method | `13_nested_kearning_gpu.py:98` | `def __init__(self, vocab_size, d_model, cms_freqs, hidden_dim)` |
| `forward` | method | `13_nested_kearning_gpu.py:64` | `def forward(self, x)` |
| `forward` | method | `13_nested_kearning_gpu.py:87` | `def forward(self, x, global_step)` |
| `forward` | method | `13_nested_kearning_gpu.py:104` | `def forward(self, x, global_step)` |
| `ContinuumMemorySystem` | class | `13_nested_learning.py:86` | `class ContinuumMemorySystem(Module)` |
| `HopeModel` | class | `13_nested_learning.py:114` | `class HopeModel(Module)` |
| `SelfModifyingMemory` | class | `13_nested_learning.py:51` | `class SelfModifyingMemory(Module)` |
| `__init__` | method | `13_nested_learning.py:52` | `def __init__(self, vocab_size, d_model, hidden_dim)` |
| `__init__` | method | `13_nested_learning.py:87` | `def __init__(self, levels, d_model, hidden_dim)` |
| `__init__` | method | `13_nested_learning.py:115` | `def __init__(self, vocab_size, d_model, cms_levels, mlp_hidden)` |
| `forward` | method | `13_nested_learning.py:62` | `def forward(self, x, update_mask)` |
| `forward` | method | `13_nested_learning.py:100` | `def forward(self, x, global_step)` |
| `forward` | method | `13_nested_learning.py:122` | `def forward(self, x, global_step)` |
| `CorpusCallosum` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:544` | `class CorpusCallosum(Module)` |
| `EnhancedDiagnostics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:602` | `class EnhancedDiagnostics` |
| `Flickr8kDataset` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:727` | `class Flickr8kDataset(Dataset)` |
| `LanguageMetrics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:44` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:473` | `class LeftHemisphere(Module)` |
| `NeuroLogosBicameralStable` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:581` | `class NeuroLogosBicameralStable(Module)` |
| `RightHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:458` | `class RightHemisphere(Module)` |
| `StableLiquidNeuron` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:382` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:117` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:748` | `def __getitem__(self, idx)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:120` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:383` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:459` | `def __init__(self, output_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:474` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:545` | `def __init__(self, dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:582` | `def __init__(self, vocab_size)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:603` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:728` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:745` | `def __len__(self)` |
| `_get_init_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:539` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:82` | `def _get_ngrams(tokens, n)` |
| `apply_triangulated_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:223` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `build_vocab_flickr` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:764` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:631` | `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:622` | `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_loss` | function | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:20` | `def compute_loss(logits, captions, gate, vocab)` |
| `count_convergent_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:157` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:161` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:397` | `def forward(self, x)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:467` | `def forward(self, image)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:502` | `def forward(self, visual_context, captions, max_len)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:568` | `def forward(self, right_features)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:588` | `def forward(self, image, captions)` |
| `get_recent_avg` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:645` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:404` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:613` | `def measure_callosal_flow(self, right_features, left_context)` |
| `report` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:650` | `def report(self, epoch)` |
| `sentence_bleu` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:48` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:782` | `def setup_flickr8k(data_dir)` |
| `token_accuracy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:91` | `def token_accuracy(reference, hypothesis)` |
| `train_with_metrics` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:854` | `def train_with_metrics()` |
| `triangulate_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:125` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:640` | `def update(self)` |
| `update_physiology_advanced` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:434` | `def update_physiology_advanced(self, loss_value)` |
| `word_overlap` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:104` | `def word_overlap(reference, hypothesis)` |
| `CorpusCallosum` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:645` | `class CorpusCallosum(Module)` |
| `EnhancedDiagnostics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:721` | `class EnhancedDiagnostics` |
| `EpisodicMemoryBuffer` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:844` | `class EpisodicMemoryBuffer` |
| `Flickr8kDataset` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:891` | `class Flickr8kDataset(Dataset)` |
| `LanguageMetrics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:44` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:524` | `class LeftHemisphere(Module)` |
| `NeuroLogosBicameralStable` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:700` | `class NeuroLogosBicameralStable(Module)` |
| `RightHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:509` | `class RightHemisphere(Module)` |
| `StableLiquidNeuron` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:383` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:117` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:912` | `def __getitem__(self, idx)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:120` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:384` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:510` | `def __init__(self, output_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:525` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:646` | `def __init__(self, dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:701` | `def __init__(self, vocab_size)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:722` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:845` | `def __init__(self, capacity, surprise_threshold)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:892` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:909` | `def __len__(self)` |
| `_get_init_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:631` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:82` | `def _get_ngrams(tokens, n)` |
| `add` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:862` | `def add(self, image, caption, surprise_score)` |
| `apply_triangulated_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:224` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `build_vocab_flickr` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:928` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:750` | `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:741` | `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_loss` | function | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:20` | `def compute_loss(logits, captions, gate, vocab)` |
| `compute_surprise` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:851` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `count_convergent_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:157` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:161` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:422` | `def forward(self, x)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:518` | `def forward(self, image)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:573` | `def forward(self, visual_context, captions, max_len)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:678` | `def forward(self, right_features)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:707` | `def forward(self, image, captions)` |
| `get_recent_avg` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:764` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:444` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:732` | `def measure_callosal_flow(self, right_features, left_context)` |
| `report` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:769` | `def report(self, epoch)` |
| `sample` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:872` | `def sample(self, batch_size)` |
| `sentence_bleu` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:48` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:946` | `def setup_flickr8k(data_dir)` |
| `token_accuracy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:91` | `def token_accuracy(reference, hypothesis)` |
| `train_with_metrics` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:1021` | `def train_with_metrics()` |
| `triangulate_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:125` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:759` | `def update(self)` |
| `update_physiology_advanced` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:483` | `def update_physiology_advanced(self, loss_value)` |
| `word_overlap` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:104` | `def word_overlap(reference, hypothesis)` |
| `CorpusCallosum` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:709` | `class CorpusCallosum(Module)` |
| `EnhancedDiagnostics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:786` | `class EnhancedDiagnostics` |
| `EpisodicMemoryBuffer` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:907` | `class EpisodicMemoryBuffer` |
| `Flickr8kDataset` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:954` | `class Flickr8kDataset(Dataset)` |
| `LanguageMetrics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:44` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:533` | `class LeftHemisphere(Module)` |
| `NeuroLogosBicameralStable` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:765` | `class NeuroLogosBicameralStable(Module)` |
| `RightHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:518` | `class RightHemisphere(Module)` |
| `StableLiquidNeuron` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:392` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:117` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:975` | `def __getitem__(self, idx)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:120` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:393` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:519` | `def __init__(self, output_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:534` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:710` | `def __init__(self, dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:766` | `def __init__(self, vocab_size)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:787` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:908` | `def __init__(self, capacity, surprise_threshold)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:955` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:972` | `def __len__(self)` |
| `_get_init_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:695` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:82` | `def _get_ngrams(tokens, n)` |
| `add` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:925` | `def add(self, image, caption, surprise_score)` |
| `apply_triangulated_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:224` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `beam_search_decode` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:584` | `def beam_search_decode(self, visual_context, beam_width, max_len, epoch)` |
| `build_vocab_flickr` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:991` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:815` | `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:806` | `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_loss` | function | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:20` | `def compute_loss(logits, captions, gate, vocab)` |
| `compute_surprise` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:914` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `count_convergent_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:157` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:161` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:431` | `def forward(self, x)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:527` | `def forward(self, image)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:657` | `def forward(self, visual_context, captions, max_len, epoch)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:742` | `def forward(self, right_features)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:772` | `def forward(self, image, captions, epoch)` |
| `get_recent_avg` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:829` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:453` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:797` | `def measure_callosal_flow(self, right_features, left_context)` |
| `report` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:834` | `def report(self, epoch)` |
| `sample` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:935` | `def sample(self, batch_size)` |
| `sentence_bleu` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:48` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:1009` | `def setup_flickr8k(data_dir)` |
| `token_accuracy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:91` | `def token_accuracy(reference, hypothesis)` |
| `train_with_metrics` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:1081` | `def train_with_metrics()` |
| `triangulate_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:125` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:824` | `def update(self)` |
| `update_physiology_advanced` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:492` | `def update_physiology_advanced(self, loss_value)` |
| `word_overlap` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:104` | `def word_overlap(reference, hypothesis)` |
| `CorpusCallosum` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:978` | `class CorpusCallosum(Module)` |
| `EnhancedDiagnostics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1055` | `class EnhancedDiagnostics` |
| `EpisodicMemoryBuffer` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1192` | `class EpisodicMemoryBuffer` |
| `Flickr8kDataset` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1239` | `class Flickr8kDataset(Dataset)` |
| `LanguageMetrics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:309` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:798` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:220` | `class LinguisticFeedbackLoop` |
| `NeuroLogosBicameralStable` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1034` | `class NeuroLogosBicameralStable(Module)` |
| `NeurocognitiveSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:49` | `class NeurocognitiveSystem` |
| `RightHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:783` | `class RightHemisphere(Module)` |
| `StableLiquidNeuron` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:657` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:382` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1260` | `def __getitem__(self, idx)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:55` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:223` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:385` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:658` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:784` | `def __init__(self, output_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:799` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:979` | `def __init__(self, dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1035` | `def __init__(self, vocab_size)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1056` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1193` | `def __init__(self, capacity, surprise_threshold)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1240` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1257` | `def __len__(self)` |
| `_get_init_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:963` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:295` | `def _get_ngrams(self, sentence, n)` |
| `_get_ngrams` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:347` | `def _get_ngrams(tokens, n)` |
| `add` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1210` | `def add(self, image, caption, surprise_score)` |
| `apply_cognitive_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:113` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch)` |
| `apply_triangulated_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:489` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:65` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `beam_search_decode` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:849` | `def beam_search_decode(self, visual_context, beam_width, max_len, epoch)` |
| `build_vocab_flickr` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1276` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1085` | `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1076` | `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_cider` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:254` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:232` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_loss` | function | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:20` | `def compute_loss(logits, captions, gate, vocab, linguistic_reward, lambda_reward)` |
| `compute_spice` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:281` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1199` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `count_convergent_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:422` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:426` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:696` | `def forward(self, x)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:792` | `def forward(self, image)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:925` | `def forward(self, visual_context, captions, max_len, epoch)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1011` | `def forward(self, right_features)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1041` | `def forward(self, image, captions, epoch)` |
| `get_recent_avg` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1099` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:718` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1067` | `def measure_callosal_flow(self, right_features, left_context)` |
| `report` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1104` | `def report(self, epoch)` |
| `sample` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1220` | `def sample(self, batch_size)` |
| `sentence_bleu` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:313` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1294` | `def setup_flickr8k(data_dir)` |
| `token_accuracy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:356` | `def token_accuracy(reference, hypothesis)` |
| `train_with_metrics` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1366` | `def train_with_metrics()` |
| `triangulate_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:390` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1094` | `def update(self)` |
| `update_physiology_advanced` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:757` | `def update_physiology_advanced(self, loss_value)` |
| `word_overlap` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:369` | `def word_overlap(reference, hypothesis)` |
| `CorpusCallosum` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1244` | `class CorpusCallosum(Module)` |
| `EnhancedDiagnostics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1446` | `class EnhancedDiagnostics` |
| `EpisodicMemoryBuffer` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1661` | `class EpisodicMemoryBuffer` |
| `Flickr8kDataset` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1708` | `class Flickr8kDataset(Dataset)` |
| `LanguageMetrics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:479` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:988` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:325` | `class LinguisticFeedbackLoop` |
| `NeuroLogosBicameralStable` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1420` | `class NeuroLogosBicameralStable(Module)` |
| `NeurocognitiveSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:49` | `class NeurocognitiveSystem` |
| `RightHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:973` | `class RightHemisphere(Module)` |
| `StableLiquidNeuron` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:827` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:552` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1729` | `def __getitem__(self, idx)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:55` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:331` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:555` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:828` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:974` | `def __init__(self, output_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:989` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1245` | `def __init__(self, dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1421` | `def __init__(self, vocab_size)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1447` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1662` | `def __init__(self, capacity, surprise_threshold)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1709` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1726` | `def __len__(self)` |
| `_apply_structural_attention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1185` | `def _apply_structural_attention(self, lstm_out, channels, visual_context)` |
| `_get_init_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1230` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:443` | `def _get_ngrams(self, sentence, n)` |
| `_get_ngrams` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:517` | `def _get_ngrams(tokens, n)` |
| `add` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1679` | `def add(self, image, caption, surprise_score)` |
| `adjust_gates_by_fatigue` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1404` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:171` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_stochastic_perturbation` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:155` | `def apply_stochastic_perturbation(self, model, epoch)` |
| `apply_triangulated_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:659` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:76` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `beam_search_decode` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1066` | `def beam_search_decode(self, visual_context, channels, beam_width, max_len, epoch)` |
| `build_vocab_flickr` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1745` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1499` | `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1490` | `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_alignment_loss` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1834` | `def compute_alignment_loss(visual_features, channels, alpha)` |
| `compute_cider` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:389` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:347` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_loss` | function | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:20` | `def compute_loss(logits, captions, gate, vocab, linguistic_reward, lambda_reward)` |
| `compute_spice` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:427` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1668` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `count_convergent_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:592` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:596` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `evaluate_gate_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:124` | `def evaluate_gate_state(self, gate_value, current_metrics)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:871` | `def forward(self, x)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:982` | `def forward(self, image)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1147` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1308` | `def forward(self, right_features, left_features)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1427` | `def forward(self, image, captions, epoch)` |
| `get_cache_stats` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:452` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1519` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:893` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1460` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `report` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1567` | `def report(self, epoch)` |
| `sample` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1689` | `def sample(self, batch_size)` |
| `sentence_bleu` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:483` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1763` | `def setup_flickr8k(data_dir)` |
| `token_accuracy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:526` | `def token_accuracy(reference, hypothesis)` |
| `train_with_metrics` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1858` | `def train_with_metrics()` |
| `triangulate_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:560` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1508` | `def update(self)` |
| `update_channel_fatigue` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1381` | `def update_channel_fatigue(self, objects_channel, actions_channel, scene_channel)` |
| `update_physiology_advanced` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:932` | `def update_physiology_advanced(self, loss_value)` |
| `update_trauma_memory` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:142` | `def update_trauma_memory(self, gate_value, metrics, outcome)` |
| `visualize_fatigue_distribution` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1538` | `def visualize_fatigue_distribution(self, epoch)` |
| `word_overlap` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:539` | `def word_overlap(reference, hypothesis)` |
| `CorpusCallosum` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1469` | `class CorpusCallosum(Module)` |
| `EnhancedDiagnostics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1669` | `class EnhancedDiagnostics` |
| `EpisodicMemoryBuffer` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1947` | `class EpisodicMemoryBuffer` |
| `Flickr8kDataset` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1994` | `class Flickr8kDataset(Dataset)` |
| `LanguageMetrics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:550` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1059` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:396` | `class LinguisticFeedbackLoop` |
| `NeuroLogosBicameralStable` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1645` | `class NeuroLogosBicameralStable(Module)` |
| `NeurocognitiveSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:53` | `class NeurocognitiveSystem` |
| `RightHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1044` | `class RightHemisphere(Module)` |
| `StableLiquidNeuron` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:898` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:623` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2015` | `def __getitem__(self, idx)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:59` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:402` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:626` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:899` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1045` | `def __init__(self, output_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1060` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1470` | `def __init__(self, dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1646` | `def __init__(self, vocab_size)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1670` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1948` | `def __init__(self, capacity, surprise_threshold)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1995` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2012` | `def __len__(self)` |
| `_apply_chain_of_thought` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1190` | `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)` |
| `_apply_multi_token_prediction` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1230` | `def _apply_multi_token_prediction(self, hidden_states, input_ids)` |
| `_apply_structural_attention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1420` | `def _apply_structural_attention(self, lstm_out, channels, visual_context)` |
| `_get_init_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1456` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:514` | `def _get_ngrams(self, sentence, n)` |
| `_get_ngrams` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:588` | `def _get_ngrams(tokens, n)` |
| `add` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1965` | `def add(self, image, caption, surprise_score)` |
| `adjust_gates_by_fatigue` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1629` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:213` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_stochastic_perturbation` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:192` | `def apply_stochastic_perturbation(self, model, epoch)` |
| `apply_triangulated_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:730` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:128` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `assess_reasoning_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:85` | `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` |
| `beam_search_decode` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1292` | `def beam_search_decode(self, visual_context, channels, beam_width, max_len, epoch)` |
| `build_vocab_flickr` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2031` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1759` | `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1750` | `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_alignment_loss` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2120` | `def compute_alignment_loss(visual_features, channels, alpha)` |
| `compute_cider` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:460` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:418` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_loss` | function | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:20` | `def compute_loss(logits, captions, gate, vocab, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)` |
| `compute_spice` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:498` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1954` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `count_convergent_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:663` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:667` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `evaluate_gate_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:167` | `def evaluate_gate_state(self, gate_value, current_metrics)` |
| `evaluate_reasoning_quality` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1709` | `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:942` | `def forward(self, x)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1053` | `def forward(self, image)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1370` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1533` | `def forward(self, right_features, left_features)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1652` | `def forward(self, image, captions, epoch)` |
| `get_cache_stats` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:523` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1778` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:964` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1686` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `report` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1836` | `def report(self, epoch)` |
| `sample` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1975` | `def sample(self, batch_size)` |
| `sentence_bleu` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:554` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2049` | `def setup_flickr8k(data_dir)` |
| `token_accuracy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:597` | `def token_accuracy(reference, hypothesis)` |
| `train_with_metrics` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2144` | `def train_with_metrics()` |
| `triangulate_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:631` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1768` | `def update(self)` |
| `update_channel_fatigue` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1606` | `def update_channel_fatigue(self, objects_channel, actions_channel, scene_channel)` |
| `update_physiology_advanced` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1003` | `def update_physiology_advanced(self, loss_value)` |
| `update_trauma_memory` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:182` | `def update_trauma_memory(self, gate_value, metrics, outcome)` |
| `visualize_fatigue_distribution` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1795` | `def visualize_fatigue_distribution(self, epoch)` |
| `visualize_reasoning_metrics` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1823` | `def visualize_reasoning_metrics(self, epoch)` |
| `word_overlap` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:610` | `def word_overlap(reference, hypothesis)` |
| `CorpusCallosum` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1019` | `class CorpusCallosum(Module)` |
| `EnhancedDiagnostics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1171` | `class EnhancedDiagnostics` |
| `EpisodicMemoryBuffer` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:58` | `class EpisodicMemoryBuffer` |
| `Flickr8kDataset` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1408` | `class Flickr8kDataset(Dataset)` |
| `LanguageMetrics` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:413` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:764` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:291` | `class LinguisticFeedbackLoop` |
| `NeuroLogosBicameralStable` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1147` | `class NeuroLogosBicameralStable(Module)` |
| `NeurocognitiveSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:112` | `class NeurocognitiveSystem` |
| `RightHemisphere` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:745` | `class RightHemisphere(Module)` |
| `StableLiquidNeuron` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:622` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:475` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1429` | `def __getitem__(self, idx)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:61` | `def __init__(self, capacity, surprise_threshold)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:113` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:294` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:476` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:623` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:746` | `def __init__(self, output_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:765` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1020` | `def __init__(self, dim)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1148` | `def __init__(self, vocab_size)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1172` | `def __init__(self)` |
| `__init__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1409` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1426` | `def __len__(self)` |
| `_apply_chain_of_thought` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:912` | `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)` |
| `_apply_multi_token_prediction` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:944` | `def _apply_multi_token_prediction(self, hidden_states, input_ids)` |
| `_apply_structural_attention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:986` | `def _apply_structural_attention(self, lstm_out, channels, visual_context)` |
| `_get_init_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1007` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:384` | `def _get_ngrams(self, sentence, n)` |
| `_greedy_decode` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:879` | `def _greedy_decode(self, visual_context, channels, max_len, epoch)` |
| `_reset_liquid_neuron` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:610` | `def _reset_liquid_neuron(self, right_node, severity)` |
| `add` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:79` | `def add(self, image, caption, surprise_score)` |
| `adjust_gates_by_fatigue` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1132` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:206` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_triangulated_intervention` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:537` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:170` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `assess_reasoning_state` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:128` | `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` |
| `build_vocab_flickr` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1445` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1251` | `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1242` | `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_alignment_loss` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1527` | `def compute_alignment_loss(visual_features, channels, alpha)` |
| `compute_cider` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:342` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:308` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_loss` | function | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:22` | `def compute_loss(logits, captions, gate, vocab, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)` |
| `compute_spice` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:371` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:67` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `count_convergent_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:492` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:495` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `evaluate_reasoning_quality` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1210` | `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:658` | `def forward(self, x)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:754` | `def forward(self, image)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:840` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1070` | `def forward(self, right_features, left_features)` |
| `forward` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1154` | `def forward(self, image, captions, epoch)` |
| `get_cache_stats` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:389` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1270` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:671` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1187` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `report` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1320` | `def report(self, epoch)` |
| `sample` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:91` | `def sample(self, batch_size)` |
| `sentence_bleu` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:417` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1463` | `def setup_flickr8k(data_dir)` |
| `token_accuracy` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:448` | `def token_accuracy(reference, hypothesis)` |
| `train_with_metrics` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1547` | `def train_with_metrics()` |
| `triangulate_signals` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:482` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1260` | `def update(self)` |
| `update_channel_fatigue` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1113` | `def update_channel_fatigue(self, objects_channel, actions_channel, scene_channel)` |
| `update_physiology_advanced` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:709` | `def update_physiology_advanced(self, loss_value)` |
| `visualize_fatigue_distribution` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1287` | `def visualize_fatigue_distribution(self, epoch)` |
| `visualize_reasoning_metrics` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1308` | `def visualize_reasoning_metrics(self, epoch)` |
| `word_overlap` | method | `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:461` | `def word_overlap(reference, hypothesis)` |
| `MiniUnconscious` | class | `ablation.py:72` | `class MiniUnconscious(Module)` |
| `NeuroLogosCPU` | class | `ablation.py:137` | `class NeuroLogosCPU(Module)` |
| `SimpleClassifier` | class | `ablation.py:124` | `class SimpleClassifier(Module)` |
| `TopoBrainCore` | class | `ablation.py:17` | `class TopoBrainCore(Module)` |
| `TopoUnconscious` | class | `ablation.py:91` | `class TopoUnconscious(Module)` |
| `__init__` | method | `ablation.py:18` | `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)` |
| `__init__` | method | `ablation.py:73` | `def __init__(self, out_dim)` |
| `__init__` | method | `ablation.py:92` | `def __init__(self, out_dim, use_grid, use_symbiotic)` |
| `__init__` | method | `ablation.py:125` | `def __init__(self, in_dim, num_classes)` |
| `__init__` | method | `ablation.py:145` | `def __init__(self, num_classes, ablation_level)` |
| `_init_grid` | method | `ablation.py:39` | `def _init_grid(self, size)` |
| `evaluate` | method | `ablation.py:229` | `def evaluate(model, loader, device)` |
| `fgsm_attack` | method | `ablation.py:175` | `def fgsm_attack(model, x, y, epsilon)` |
| `forward` | method | `ablation.py:46` | `def forward(self, x)` |
| `forward` | method | `ablation.py:87` | `def forward(self, x)` |
| `forward` | method | `ablation.py:112` | `def forward(self, x)` |
| `forward` | method | `ablation.py:129` | `def forward(self, x)` |
| `forward` | method | `ablation.py:161` | `def forward(self, x)` |
| `get_metrics` | method | `ablation.py:64` | `def get_metrics(self)` |
| `get_metrics` | method | `ablation.py:116` | `def get_metrics(self)` |
| `get_metrics` | method | `ablation.py:165` | `def get_metrics(self)` |
| `run_ablation_cpu` | method | `ablation.py:245` | `def run_ablation_cpu()` |
| `train_epoch` | method | `ablation.py:188` | `def train_epoch(model, loader, optimizer, device, use_adv)` |
| `AblationConfig` | class | `ablation1.py:60` | `class AblationConfig(EliteConfig)` |
| `AdaptiveTopology` | class | `ablation1.py:296` | `class AdaptiveTopology(Module)` |
| `AdvancedHomeostaticCell` | class | `ablation1.py:248` | `class AdvancedHomeostaticCell(Module)` |
| `EliteConfig` | class | `ablation1.py:20` | `class EliteConfig` |
| `EliteTopoBrain` | class | `ablation1.py:329` | `class EliteTopoBrain(Module)` |
| `EpisodicMemory` | class | `ablation1.py:180` | `class EpisodicMemory(Module)` |
| `SpectralNormLinear` | class | `ablation1.py:220` | `class SpectralNormLinear(Module)` |
| `SupConLoss` | class | `ablation1.py:467` | `class SupConLoss(Module)` |
| `__init__` | method | `ablation1.py:182` | `def __init__(self, dim, capacity)` |
| `__init__` | method | `ablation1.py:222` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `ablation1.py:250` | `def __init__(self, d_in, d_out, use_spectral, use_homeostasis)` |
| `__init__` | method | `ablation1.py:298` | `def __init__(self, num_nodes, grid_size)` |
| `__init__` | method | `ablation1.py:330` | `def __init__(self, config)` |
| `__init__` | method | `ablation1.py:468` | `def __init__(self, temperature)` |
| `count_parameters` | method | `ablation1.py:367` | `def count_parameters(self)` |
| `create_ablation_configs` | method | `ablation1.py:64` | `def create_ablation_configs(config)` |
| `elite_pgd_attack` | method | `ablation1.py:407` | `def elite_pgd_attack(model, x, y, eps, steps, stress)` |
| `forward` | method | `ablation1.py:236` | `def forward(self, x)` |
| `forward` | method | `ablation1.py:271` | `def forward(self, x)` |
| `forward` | method | `ablation1.py:317` | `def forward(self, stress)` |
| `forward` | method | `ablation1.py:370` | `def forward(self, x, stress)` |
| `forward` | method | `ablation1.py:472` | `def forward(self, features, labels)` |
| `get_elite_dataset` | method | `ablation1.py:157` | `def get_elite_dataset(config)` |
| `power_iteration` | method | `ablation1.py:229` | `def power_iteration(self, n_iter)` |
| `retrieve` | method | `ablation1.py:203` | `def retrieve(self, x, k)` |
| `run_ablation_study` | method | `ablation1.py:616` | `def run_ablation_study()` |
| `seed_everything` | method | `ablation1.py:150` | `def seed_everything(seed)` |
| `train_elite_model` | method | `ablation1.py:504` | `def train_elite_model(config, dataset, fold_results)` |
| `update` | method | `ablation1.py:190` | `def update(self, x, y)` |
| `BaselineUnconscious` | class | `ablation2.py:167` | `class BaselineUnconscious(Module)` |
| `BioDecoder` | class | `ablation2.py:231` | `class BioDecoder(Module)` |
| `CIFARCaptions_v51` | class | `ablation2.py:375` | `class CIFARCaptions_v51` |
| `ConsciousCore` | class | `ablation2.py:219` | `class ConsciousCore(Module)` |
| `NeuroLogos_v51` | class | `ablation2.py:289` | `class NeuroLogos_v51(Module)` |
| `PGDAttack` | class | `ablation2.py:346` | `class PGDAttack` |
| `SparseCompetitiveLayer` | class | `ablation2.py:19` | `class SparseCompetitiveLayer(Module)` |
| `SparseSymbioticCore` | class | `ablation2.py:120` | `class SparseSymbioticCore(Module)` |
| `SparseUnconscious` | class | `ablation2.py:186` | `class SparseUnconscious(Module)` |
| `SymbioticRefiner` | class | `ablation2.py:91` | `class SymbioticRefiner(Module)` |
| `__getitem__` | method | `ablation2.py:406` | `def __getitem__(self, idx)` |
| `__init__` | method | `ablation2.py:25` | `def __init__(self, n_nodes, k_sparse, input_dim)` |
| `__init__` | method | `ablation2.py:96` | `def __init__(self, n_nodes)` |
| `__init__` | method | `ablation2.py:125` | `def __init__(self, input_dim, hidden_dim, n_nodes, k_sparse)` |
| `__init__` | method | `ablation2.py:169` | `def __init__(self, output_dim)` |
| `__init__` | method | `ablation2.py:188` | `def __init__(self, output_dim, n_nodes, k_sparse)` |
| `__init__` | method | `ablation2.py:220` | `def __init__(self, dim)` |
| `__init__` | method | `ablation2.py:233` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `ablation2.py:299` | `def __init__(self, vocab_size, mode, n_nodes, k_sparse)` |
| `__init__` | method | `ablation2.py:347` | `def __init__(self, epsilon, alpha, steps)` |
| `__init__` | method | `ablation2.py:376` | `def __init__(self)` |
| `__len__` | method | `ablation2.py:403` | `def __len__(self)` |
| `_get_init_state` | method | `ablation2.py:279` | `def _get_init_state(self, thought)` |
| `attack` | method | `ablation2.py:352` | `def attack(self, model, x, y, criterion)` |
| `compute_bleu` | method | `ablation2.py:417` | `def compute_bleu(pred_ids, target_ids, dataset, max_n)` |
| `forward` | method | `ablation2.py:44` | `def forward(self, x)` |
| `forward` | method | `ablation2.py:106` | `def forward(self, x)` |
| `forward` | method | `ablation2.py:143` | `def forward(self, x)` |
| `forward` | method | `ablation2.py:182` | `def forward(self, x)` |
| `forward` | method | `ablation2.py:207` | `def forward(self, x)` |
| `forward` | method | `ablation2.py:225` | `def forward(self, x)` |
| `forward` | method | `ablation2.py:248` | `def forward(self, thought, captions, max_len)` |
| `forward` | method | `ablation2.py:331` | `def forward(self, image, captions)` |
| `forward_fn` | method | `ablation2.py:526` | `def forward_fn(x)` |
| `get_metrics` | method | `ablation2.py:79` | `def get_metrics(self)` |
| `get_metrics` | method | `ablation2.py:158` | `def get_metrics(self)` |
| `get_metrics` | method | `ablation2.py:211` | `def get_metrics(self)` |
| `get_metrics` | method | `ablation2.py:340` | `def get_metrics(self)` |
| `ngrams` | method | `ablation2.py:423` | `def ngrams(tokens, n)` |
| `run_ablation_v51` | method | `ablation2.py:619` | `def run_ablation_v51(epochs, device, n_nodes, k_sparse)` |
| `train_ablation_v51` | method | `ablation2.py:466` | `def train_ablation_v51(mode, epochs, device, n_nodes, k_sparse)` |
| `AdversarialWrapper` | class | `ablation3.py:77` | `class AdversarialWrapper` |
| `BioDecoder` | class | `ablation3.py:136` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `ablation3.py:295` | `class CIFARCaptions` |
| `ConsciousCore` | class | `ablation3.py:125` | `class ConsciousCore(Module)` |
| `NeuroLogosFactorial` | class | `ablation3.py:190` | `class NeuroLogosFactorial(Module)` |
| `SparseLayer` | class | `ablation3.py:21` | `class SparseLayer(Module)` |
| `SymbioticLayer` | class | `ablation3.py:54` | `class SymbioticLayer(Module)` |
| `VisualBackbone` | class | `ablation3.py:109` | `class VisualBackbone(Module)` |
| `__getitem__` | method | `ablation3.py:324` | `def __getitem__(self, idx)` |
| `__init__` | method | `ablation3.py:23` | `def __init__(self, input_dim, n_nodes, k_sparse)` |
| `__init__` | method | `ablation3.py:56` | `def __init__(self, n_nodes)` |
| `__init__` | method | `ablation3.py:79` | `def __init__(self, epsilon, alpha, steps)` |
| `__init__` | method | `ablation3.py:110` | `def __init__(self, output_dim)` |
| `__init__` | method | `ablation3.py:126` | `def __init__(self, dim)` |
| `__init__` | method | `ablation3.py:137` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `ablation3.py:191` | `def __init__(self, vocab_size, use_sparse, use_symbiotic, use_adv, n_nodes, k_sparse)` |
| `__init__` | method | `ablation3.py:296` | `def __init__(self)` |
| `__len__` | method | `ablation3.py:321` | `def __len__(self)` |
| `_init_state` | method | `ablation3.py:182` | `def _init_state(self, thought)` |
| `analyze_results` | method | `ablation3.py:480` | `def analyze_results(results)` |
| `attack` | method | `ablation3.py:84` | `def attack(self, model_fn, x, y, criterion)` |
| `forward` | method | `ablation3.py:33` | `def forward(self, x)` |
| `forward` | method | `ablation3.py:62` | `def forward(self, x)` |
| `forward` | method | `ablation3.py:122` | `def forward(self, x)` |
| `forward` | method | `ablation3.py:131` | `def forward(self, x)` |
| `forward` | method | `ablation3.py:148` | `def forward(self, thought, captions, max_len)` |
| `forward` | method | `ablation3.py:226` | `def forward(self, image, captions)` |
| `model_fn` | method | `ablation3.py:257` | `def model_fn(x)` |
| `run_full_factorial` | method | `ablation3.py:411` | `def run_full_factorial(epochs, device, n_nodes, k_sparse)` |
| `train_configuration` | method | `ablation3.py:338` | `def train_configuration(config, epochs, device, n_nodes, k_sparse)` |
| `train_step` | method | `ablation3.py:243` | `def train_step(self, images, captions, optimizer, dataset)` |
| `CombinatorialComplexLayer` | class | `adversarial_benchmark.py:118` | `class CombinatorialComplexLayer(Module)` |
| `LearnableAbsenceGating` | class | `adversarial_benchmark.py:85` | `class LearnableAbsenceGating(Module)` |
| `PredictiveErrorCell` | class | `adversarial_benchmark.py:73` | `class PredictiveErrorCell(Module)` |
| `SupConLoss` | class | `adversarial_benchmark.py:36` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `adversarial_benchmark.py:100` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainNet` | class | `adversarial_benchmark.py:158` | `class TopoBrainNet(Module)` |
| `__init__` | method | `adversarial_benchmark.py:37` | `def __init__(self, temperature)` |
| `__init__` | method | `adversarial_benchmark.py:74` | `def __init__(self, dim)` |
| `__init__` | method | `adversarial_benchmark.py:86` | `def __init__(self, dim)` |
| `__init__` | method | `adversarial_benchmark.py:101` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `adversarial_benchmark.py:119` | `def __init__(self, in_dim, hid_dim, num_nodes, layer_type)` |
| `__init__` | method | `adversarial_benchmark.py:159` | `def __init__(self, grid_size)` |
| `_init_grid_topology` | method | `adversarial_benchmark.py:191` | `def _init_grid_topology(self, N)` |
| `calculate_ortho_loss` | method | `adversarial_benchmark.py:225` | `def calculate_ortho_loss(self)` |
| `forward` | method | `adversarial_benchmark.py:41` | `def forward(self, features, labels)` |
| `forward` | method | `adversarial_benchmark.py:79` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `adversarial_benchmark.py:95` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `adversarial_benchmark.py:109` | `def forward(self, x)` |
| `forward` | method | `adversarial_benchmark.py:136` | `def forward(self, x_nodes, adjacency, incidence)` |
| `forward` | method | `adversarial_benchmark.py:249` | `def forward(self, x)` |
| `get_topology` | method | `adversarial_benchmark.py:213` | `def get_topology(self)` |
| `lambda_general` | method | `adversarial_benchmark.py:321` | `def lambda_general(epoch)` |
| `lambda_topo` | method | `adversarial_benchmark.py:317` | `def lambda_topo(epoch)` |
| `make_adversarial_pgd` | method | `adversarial_benchmark.py:269` | `def make_adversarial_pgd(model, x, y, eps, steps)` |
| `train_and_eval` | method | `adversarial_benchmark.py:290` | `def train_and_eval()` |
| `ChimeraNetwork` | class | `apex.py:136` | `class ChimeraNetwork(Module)` |
| `DataEnvironment` | class | `apex.py:24` | `class DataEnvironment` |
| `DualPhaseMemory` | class | `apex.py:78` | `class DualPhaseMemory(Module)` |
| `ElasticMemory` | class | `apex.py:93` | `class ElasticMemory(Module)` |
| `LiquidNeuron` | class | `apex.py:50` | `class LiquidNeuron(Module)` |
| `SovereignAttention` | class | `apex.py:69` | `class SovereignAttention(Module)` |
| `__init__` | method | `apex.py:25` | `def __init__(self)` |
| `__init__` | method | `apex.py:51` | `def __init__(self, d_in, d_out)` |
| `__init__` | method | `apex.py:70` | `def __init__(self, d_in)` |
| `__init__` | method | `apex.py:79` | `def __init__(self, d_in)` |
| `__init__` | method | `apex.py:94` | `def __init__(self, model, lambda_ewc)` |
| `__init__` | method | `apex.py:137` | `def __init__(self, d_in, d_hid, d_out)` |
| `forward` | method | `apex.py:58` | `def forward(self, x, gate)` |
| `forward` | method | `apex.py:74` | `def forward(self, x, chaos)` |
| `forward` | method | `apex.py:82` | `def forward(self, x, p)` |
| `forward` | method | `apex.py:145` | `def forward(self, x, phase)` |
| `get_train_batch` | method | `apex.py:35` | `def get_train_batch(self, phase, batch_size)` |
| `penalty` | method | `apex.py:125` | `def penalty(self)` |
| `register_fisher` | method | `apex.py:101` | `def register_fisher(self, dataset_x, dataset_y)` |
| `seed_everything` | function | `apex.py:14` | `def seed_everything(seed)` |
| `train_and_audit` | method | `apex.py:160` | `def train_and_audit(name, use_ewc)` |
| `update` | method | `apex.py:86` | `def update(self, x, p)` |
| `analyze_ia_vs_rules` | function | `app.py:219` | `def analyze_ia_vs_rules(df)` |
| `apply_ai_predictions` | function | `app.py:191` | `def apply_ai_predictions(df, model, vectorizer)` |
| `apply_ai_predictions` | function | `app.py:205` | `def apply_ai_predictions(df, model, vectorizer)` |
| `basic_statistics` | function | `app.py:454` | `def basic_statistics(df)` |
| `command_analysis` | function | `app.py:467` | `def command_analysis(df)` |
| `executive_kpis` | function | `app.py:317` | `def executive_kpis(df)` |
| `export_report` | function | `app.py:409` | `def export_report(df, kpis, okrs, ia_analysis)` |
| `generate_visualizations` | function | `app.py:377` | `def generate_visualizations(df, kpis)` |
| `load_and_clean_data_robust` | function | `app.py:252` | `def load_and_clean_data_robust(filepath)` |
| `load_or_train_model` | function | `app.py:156` | `def load_or_train_model(df)` |
| `main` | function | `app.py:530` | `def main()` |
| `network_analysis` | function | `app.py:480` | `def network_analysis(df)` |
| `parse_csv_manual` | function | `app.py:294` | `def parse_csv_manual(filepath)` |
| `security_insights` | function | `app.py:508` | `def security_insights(df)` |
| `statistical_analysis` | function | `app.py:500` | `def statistical_analysis(df)` |
| `strategic_okrs` | function | `app.py:344` | `def strategic_okrs(df, kpis)` |
| `temporal_analysis` | function | `app.py:492` | `def temporal_analysis(df)` |
| `train_ai_model` | function | `app.py:82` | `def train_ai_model(df)` |
| `AutoRegulationSystem` | class | `auto_regulation_working.py:72` | `class AutoRegulationSystem` |
| `Config` | class | `auto_regulation_working.py:20` | `class Config` |
| `DataEnvironment` | class | `auto_regulation_working.py:38` | `class DataEnvironment` |
| `PhysioChimeraFixed` | class | `auto_regulation_working.py:103` | `class PhysioChimeraFixed(Module)` |
| `__init__` | method | `auto_regulation_working.py:39` | `def __init__(self)` |
| `__init__` | method | `auto_regulation_working.py:73` | `def __init__(self, size)` |
| `__init__` | method | `auto_regulation_working.py:104` | `def __init__(self, config)` |
| `demo_auto_regulation` | method | `auto_regulation_working.py:193` | `def demo_auto_regulation()` |
| `forward` | method | `auto_regulation_working.py:129` | `def forward(self, x, global_step, phase, prev_loss)` |
| `get_batch` | method | `auto_regulation_working.py:49` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `auto_regulation_working.py:63` | `def get_full(self)` |
| `get_stability` | method | `auto_regulation_working.py:95` | `def get_stability(self)` |
| `get_w2` | method | `auto_regulation_working.py:66` | `def get_w2(self)` |
| `seed_everything` | method | `auto_regulation_working.py:28` | `def seed_everything(seed)` |
| `update` | method | `auto_regulation_working.py:78` | `def update(self, input_variance, loss_gradient, phase)` |
| `CorpusCallosum` | class | `bicamera.py.py:298` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `bicamera.py.py:404` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphere` | class | `bicamera.py.py:201` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `bicamera.py.py:466` | `class LifeCycle` |
| `LiquidNeuron` | class | `bicamera.py.py:107` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `bicamera.py.py:334` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `bicamera.py.py:313` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `bicamera.py.py:180` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `bicamera.py.py:426` | `def __getitem__(self, idx)` |
| `__init__` | method | `bicamera.py.py:108` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `bicamera.py.py:181` | `def __init__(self, output_dim)` |
| `__init__` | method | `bicamera.py.py:202` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `bicamera.py.py:299` | `def __init__(self, dim)` |
| `__init__` | method | `bicamera.py.py:314` | `def __init__(self, vocab_size)` |
| `__init__` | method | `bicamera.py.py:335` | `def __init__(self)` |
| `__init__` | method | `bicamera.py.py:405` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `bicamera.py.py:467` | `def __init__(self, total_epochs)` |
| `__len__` | method | `bicamera.py.py:423` | `def __len__(self)` |
| `_get_init_state` | method | `bicamera.py.py:278` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `bicamera.py.py:283` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `bicamera.py.py:443` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `consolidate_svd` | method | `bicamera.py.py:154` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `bicamera.py.py:125` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `bicamera.py.py:192` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `bicamera.py.py:224` | `def forward(self, visual_context, captions, max_len, return_gate)` |
| `forward` | method | `bicamera.py.py:307` | `def forward(self, right_features)` |
| `forward` | method | `bicamera.py.py:320` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)` |
| `get_plasticity` | method | `bicamera.py.py:470` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `bicamera.py.py:362` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `bicamera.py.py:346` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `bicamera.py.py:353` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `bicamera.py.py:367` | `def report(self, epoch)` |
| `setup_flickr8k` | function | `bicamera.py.py:34` | `def setup_flickr8k(data_dir)` |
| `train_bicameral` | method | `bicamera.py.py:481` | `def train_bicameral()` |
| `update` | method | `bicamera.py.py:357` | `def update(self)` |
| `CorpusCallosum` | class | `bicameral.py:276` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `bicameral.py:383` | `class Flickr8kDataset(Dataset)` |
| `HomeostaticRegulator` | class | `bicameral.py:100` | `class HomeostaticRegulator(Module)` |
| `LeftHemisphere` | class | `bicameral.py:199` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `bicameral.py:436` | `class LifeCycle` |
| `NeuralDiagnostics` | class | `bicameral.py:309` | `class NeuralDiagnostics` |
| `NeuroLogosBicameralFisiologico` | class | `bicameral.py:290` | `class NeuroLogosBicameralFisiologico(Module)` |
| `PhysioNeuron` | class | `bicameral.py:123` | `class PhysioNeuron(Module)` |
| `RightHemisphere` | class | `bicameral.py:164` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `bicameral.py:403` | `def __getitem__(self, idx)` |
| `__init__` | method | `bicameral.py:101` | `def __init__(self)` |
| `__init__` | method | `bicameral.py:124` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `bicameral.py:165` | `def __init__(self, output_dim, num_nodes)` |
| `__init__` | method | `bicameral.py:200` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `bicameral.py:277` | `def __init__(self, dim)` |
| `__init__` | method | `bicameral.py:291` | `def __init__(self, vocab_size)` |
| `__init__` | method | `bicameral.py:310` | `def __init__(self)` |
| `__init__` | method | `bicameral.py:384` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `bicameral.py:437` | `def __init__(self, total_epochs)` |
| `__len__` | method | `bicameral.py:400` | `def __len__(self)` |
| `_get_init_state` | method | `bicameral.py:258` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `bicameral.py:263` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `bicameral.py:416` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `forward` | method | `bicameral.py:109` | `def forward(self, stress, excitation, fatigue, entropy, phase, loss_signal)` |
| `forward` | method | `bicameral.py:138` | `def forward(self, x, global_loss)` |
| `forward` | method | `bicameral.py:178` | `def forward(self, image, global_loss)` |
| `forward` | method | `bicameral.py:219` | `def forward(self, visual_context, captions, max_len, return_gate)` |
| `forward` | method | `bicameral.py:284` | `def forward(self, right_features)` |
| `forward` | method | `bicameral.py:297` | `def forward(self, image, captions, global_loss, return_diagnostics)` |
| `get_global_loss_proxy` | method | `bicameral.py:440` | `def get_global_loss_proxy(self, epoch)` |
| `get_recent_avg` | method | `bicameral.py:342` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `bicameral.py:326` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `bicameral.py:333` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `bicameral.py:347` | `def report(self, epoch)` |
| `seed_all` | function | `bicameral.py:38` | `def seed_all(seed)` |
| `setup_flickr8k` | function | `bicameral.py:46` | `def setup_flickr8k(data_dir)` |
| `train_bicameral_fisiologico` | method | `bicameral.py:447` | `def train_bicameral_fisiologico()` |
| `update` | method | `bicameral.py:337` | `def update(self)` |
| `CorpusCallosum` | class | `bicameral2.py:176` | `class CorpusCallosum(Module)` |
| `DemocraticDiagnostics` | class | `bicameral2.py:206` | `class DemocraticDiagnostics` |
| `Flickr8kDataset` | class | `bicameral2.py:246` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphere` | class | `bicameral2.py:136` | `class LeftHemisphere(Module)` |
| `MinimalLiquidNeuron` | class | `bicameral2.py:104` | `class MinimalLiquidNeuron(Module)` |
| `NeuroLogosBicameralUltra` | class | `bicameral2.py:186` | `class NeuroLogosBicameralUltra(Module)` |
| `RightHemisphere` | class | `bicameral2.py:126` | `class RightHemisphere(Module)` |
| `TinyVisualEncoder` | class | `bicameral2.py:83` | `class TinyVisualEncoder(Module)` |
| `__getitem__` | method | `bicameral2.py:262` | `def __getitem__(self, idx)` |
| `__init__` | method | `bicameral2.py:84` | `def __init__(self, output_dim)` |
| `__init__` | method | `bicameral2.py:105` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `bicameral2.py:127` | `def __init__(self, output_dim)` |
| `__init__` | method | `bicameral2.py:137` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `bicameral2.py:177` | `def __init__(self, dim)` |
| `__init__` | method | `bicameral2.py:187` | `def __init__(self, vocab_size)` |
| `__init__` | method | `bicameral2.py:207` | `def __init__(self)` |
| `__init__` | method | `bicameral2.py:247` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `bicameral2.py:261` | `def __len__(self)` |
| `avg` | method | `bicameral2.py:223` | `def avg(self, k, n)` |
| `build_vocab` | method | `bicameral2.py:274` | `def build_vocab(captions_file, size)` |
| `forward` | method | `bicameral2.py:98` | `def forward(self, x)` |
| `forward` | method | `bicameral2.py:113` | `def forward(self, x)` |
| `forward` | method | `bicameral2.py:131` | `def forward(self, x)` |
| `forward` | method | `bicameral2.py:143` | `def forward(self, visual_ctx, captions, max_len)` |
| `forward` | method | `bicameral2.py:180` | `def forward(self, x)` |
| `forward` | method | `bicameral2.py:192` | `def forward(self, image, captions, return_diagnostics)` |
| `measure_flow` | method | `bicameral2.py:212` | `def measure_flow(self, r, l)` |
| `report` | method | `bicameral2.py:226` | `def report(self, epoch)` |
| `setup_flickr8k` | function | `bicameral2.py:32` | `def setup_flickr8k(data_dir)` |
| `train_ultra` | method | `bicameral2.py:290` | `def train_ultra()` |
| `update` | method | `bicameral2.py:219` | `def update(self)` |
| `vocab_diversity` | method | `bicameral2.py:217` | `def vocab_diversity(self, tokens, V)` |
| `CorpusCallosum` | class | `bicameral3.py:298` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `bicameral3.py:404` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphere` | class | `bicameral3.py:201` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `bicameral3.py:466` | `class LifeCycle` |
| `LiquidNeuron` | class | `bicameral3.py:107` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `bicameral3.py:334` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `bicameral3.py:313` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `bicameral3.py:180` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `bicameral3.py:426` | `def __getitem__(self, idx)` |
| `__init__` | method | `bicameral3.py:108` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `bicameral3.py:181` | `def __init__(self, output_dim)` |
| `__init__` | method | `bicameral3.py:202` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `bicameral3.py:299` | `def __init__(self, dim)` |
| `__init__` | method | `bicameral3.py:314` | `def __init__(self, vocab_size)` |
| `__init__` | method | `bicameral3.py:335` | `def __init__(self)` |
| `__init__` | method | `bicameral3.py:405` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `bicameral3.py:467` | `def __init__(self, total_epochs)` |
| `__len__` | method | `bicameral3.py:423` | `def __len__(self)` |
| `_get_init_state` | method | `bicameral3.py:278` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `bicameral3.py:283` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `bicameral3.py:443` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `consolidate_svd` | method | `bicameral3.py:154` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `bicameral3.py:125` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `bicameral3.py:192` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `bicameral3.py:224` | `def forward(self, visual_context, captions, max_len, return_gate)` |
| `forward` | method | `bicameral3.py:307` | `def forward(self, right_features)` |
| `forward` | method | `bicameral3.py:320` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)` |
| `get_plasticity` | method | `bicameral3.py:470` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `bicameral3.py:362` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `bicameral3.py:346` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `bicameral3.py:353` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `bicameral3.py:367` | `def report(self, epoch)` |
| `setup_flickr8k` | function | `bicameral3.py:34` | `def setup_flickr8k(data_dir)` |
| `train_bicameral` | method | `bicameral3.py:481` | `def train_bicameral()` |
| `update` | method | `bicameral3.py:357` | `def update(self)` |
| `AdaptiveCombinatorialComplexLayer` | class | `bicameral_v2.py:300` | `class AdaptiveCombinatorialComplexLayer(Module)` |
| `BCMPlasticity` | class | `bicameral_v2.py:117` | `class BCMPlasticity(Module)` |
| `BicameralHomeostasis` | class | `bicameral_v2.py:741` | `class BicameralHomeostasis(Module)` |
| `BioDecoder` | class | `bicameral_v2.py:558` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `bicameral_v2.py:974` | `class CIFARCaptions` |
| `ConsciousCore` | class | `bicameral_v2.py:486` | `class ConsciousCore(Module)` |
| `ConsciousCore` | class | `bicameral_v2.py:667` | `class ConsciousCore(Module)` |
| `CorpusCallosum` | class | `bicameral_v2.py:817` | `class CorpusCallosum(Module)` |
| `GraphNeuralLayer` | class | `bicameral_v2.py:311` | `class GraphNeuralLayer(Module)` |
| `HomeostasisEngine` | class | `bicameral_v2.py:725` | `class HomeostasisEngine(Module)` |
| `LeftHemisphere` | class | `bicameral_v2.py:543` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `bicameral_v2.py:956` | `class LifeCycle` |
| `LiquidNeuron` | class | `bicameral_v2.py:134` | `class LiquidNeuron(Module)` |
| `MiniUnconscious` | class | `bicameral_v2.py:406` | `class MiniUnconscious(Module)` |
| `NestedUnconscious` | class | `bicameral_v2.py:423` | `class NestedUnconscious(Module)` |
| `NeuroLogos` | class | `bicameral_v2.py:868` | `class NeuroLogos(Module)` |
| `ReplayMemory` | class | `bicameral_v2.py:773` | `class ReplayMemory(Module)` |
| `ResidualBlock` | class | `bicameral_v2.py:225` | `class ResidualBlock(Module)` |
| `RightHemisphere` | class | `bicameral_v2.py:342` | `class RightHemisphere(Module)` |
| `SymbioticBasisRefinement` | class | `bicameral_v2.py:280` | `class SymbioticBasisRefinement(Module)` |
| `TopologicalCompressor` | class | `bicameral_v2.py:466` | `class TopologicalCompressor(Module)` |
| `VisualCortex` | class | `bicameral_v2.py:244` | `class VisualCortex(Module)` |
| `__getitem__` | method | `bicameral_v2.py:1002` | `def __getitem__(self, idx)` |
| `__init__` | method | `bicameral_v2.py:118` | `def __init__(self, neurons, tau_theta)` |
| `__init__` | method | `bicameral_v2.py:135` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `bicameral_v2.py:226` | `def __init__(self, in_channels, out_channels, stride)` |
| `__init__` | method | `bicameral_v2.py:245` | `def __init__(self, output_dim, grid_size)` |
| `__init__` | method | `bicameral_v2.py:281` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `bicameral_v2.py:301` | `def __init__(self, in_dim, hid_dim, num_nodes, config)` |
| `__init__` | method | `bicameral_v2.py:312` | `def __init__(self, dim, hidden_dim)` |
| `__init__` | method | `bicameral_v2.py:343` | `def __init__(self, config)` |
| `__init__` | method | `bicameral_v2.py:407` | `def __init__(self)` |
| `__init__` | method | `bicameral_v2.py:424` | `def __init__(self, grid_size, output_dim)` |
| `__init__` | method | `bicameral_v2.py:467` | `def __init__(self, node_dim)` |
| `__init__` | method | `bicameral_v2.py:487` | `def __init__(self)` |
| `__init__` | method | `bicameral_v2.py:544` | `def __init__(self, use_nested)` |
| `__init__` | method | `bicameral_v2.py:559` | `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)` |
| `__init__` | method | `bicameral_v2.py:668` | `def __init__(self)` |
| `__init__` | method | `bicameral_v2.py:726` | `def __init__(self)` |
| `__init__` | method | `bicameral_v2.py:742` | `def __init__(self)` |
| `__init__` | method | `bicameral_v2.py:774` | `def __init__(self, capacity, noise_scale)` |
| `__init__` | method | `bicameral_v2.py:818` | `def __init__(self)` |
| `__init__` | method | `bicameral_v2.py:869` | `def __init__(self, vocab_size, use_nested)` |
| `__init__` | method | `bicameral_v2.py:957` | `def __init__(self, total_epochs)` |
| `__init__` | method | `bicameral_v2.py:975` | `def __init__(self)` |
| `__len__` | method | `bicameral_v2.py:999` | `def __len__(self)` |
| `_get_init_state` | method | `bicameral_v2.py:659` | `def _get_init_state(self, thought)` |
| `_make_layer` | method | `bicameral_v2.py:259` | `def _make_layer(self, in_channels, out_channels, num_blocks, stride)` |
| `compute_phi_effective` | function | `bicameral_v2.py:24` | `def compute_phi_effective(activations, k_partitions)` |
| `consolidate_svd` | method | `bicameral_v2.py:204` | `def consolidate_svd(self, repair_strength, timescale)` |
| `create_grid_adjacency` | method | `bicameral_v2.py:327` | `def create_grid_adjacency(N, connectivity)` |
| `decide` | method | `bicameral_v2.py:730` | `def decide(self, task_loss_val, richness_val, vn_entropy_val)` |
| `decide` | method | `bicameral_v2.py:751` | `def decide(self, left_metrics, right_metrics, epoch, total_epochs)` |
| `estimate_coherence` | method | `bicameral_v2.py:1012` | `def estimate_coherence(sentence, templates_per_class)` |
| `forward` | method | `bicameral_v2.py:123` | `def forward(self, activity, dt)` |
| `forward` | method | `bicameral_v2.py:158` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `bicameral_v2.py:238` | `def forward(self, x)` |
| `forward` | method | `bicameral_v2.py:265` | `def forward(self, x)` |
| `forward` | method | `bicameral_v2.py:291` | `def forward(self, x)` |
| `forward` | method | `bicameral_v2.py:307` | `def forward(self, x, plasticity_gate)` |
| `forward` | method | `bicameral_v2.py:322` | `def forward(self, nodes, adjacency)` |
| `forward` | method | `bicameral_v2.py:375` | `def forward(self, image, adjacency, plasticity)` |
| `forward` | method | `bicameral_v2.py:420` | `def forward(self, x)` |
| `forward` | method | `bicameral_v2.py:444` | `def forward(self, x)` |
| `forward` | method | `bicameral_v2.py:476` | `def forward(self, nodes, plasticity, transfer_rate)` |
| `forward` | method | `bicameral_v2.py:499` | `def forward(self, visual_features, plasticity, transfer_rate)` |
| `forward` | method | `bicameral_v2.py:550` | `def forward(self, image, callosal_input, plasticity, transfer_rate)` |
| `forward` | method | `bicameral_v2.py:578` | `def forward(self, thought, visual_features, captions, max_len)` |
| `forward` | method | `bicameral_v2.py:680` | `def forward(self, visual_features, plasticity, transfer_rate)` |
| `forward` | method | `bicameral_v2.py:824` | `def forward(self, left_repr, right_repr, mode)` |
| `forward` | method | `bicameral_v2.py:895` | `def forward(self, image, captions, plasticity, transfer_rate, mode, epoch)` |
| `get_liquid_module` | method | `bicameral_v2.py:537` | `def get_liquid_module(self)` |
| `get_liquid_module` | method | `bicameral_v2.py:718` | `def get_liquid_module(self)` |
| `get_plasticity` | method | `bicameral_v2.py:961` | `def get_plasticity(self, epoch)` |
| `measure_spatial_richness` | function | `bicameral_v2.py:52` | `def measure_spatial_richness(activations)` |
| `replay` | method | `bicameral_v2.py:791` | `def replay(self, batch_size)` |
| `set_epoch` | method | `bicameral_v2.py:950` | `def set_epoch(self, epoch)` |
| `store` | method | `bicameral_v2.py:780` | `def store(self, pattern)` |
| `top_k_top_p_filtering` | function | `bicameral_v2.py:98` | `def top_k_top_p_filtering(logits, top_k, top_p, filter_value)` |
| `train_logos` | method | `bicameral_v2.py:1026` | `def train_logos(use_nested)` |
| `AdaptiveCombinatorialComplexLayer` | class | `bicameral_v3.py:561` | `class AdaptiveCombinatorialComplexLayer(Module)` |
| `BCMPlasticity` | class | `bicameral_v3.py:241` | `class BCMPlasticity(Module)` |
| `BicameralHomeostasis` | class | `bicameral_v3.py:1082` | `class BicameralHomeostasis(Module)` |
| `BioDecoder` | class | `bicameral_v3.py:257` | `class BioDecoder(Module)` |
| `BioDecoder` | class | `bicameral_v3.py:860` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `bicameral_v3.py:1298` | `class CIFARCaptions` |
| `ConsciousCore` | class | `bicameral_v3.py:744` | `class ConsciousCore(Module)` |
| `CorpusCallosum` | class | `bicameral_v3.py:991` | `class CorpusCallosum(Module)` |
| `GraphNeuralLayer` | class | `bicameral_v3.py:572` | `class GraphNeuralLayer(Module)` |
| `HomeostasisEngine` | class | `bicameral_v3.py:1039` | `class HomeostasisEngine(Module)` |
| `LeftHemisphere` | class | `bicameral_v3.py:846` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `bicameral_v3.py:1280` | `class LifeCycle` |
| `LiquidNeuron` | class | `bicameral_v3.py:383` | `class LiquidNeuron(Module)` |
| `MiniUnconscious` | class | `bicameral_v3.py:667` | `class MiniUnconscious(Module)` |
| `NestedUnconscious` | class | `bicameral_v3.py:684` | `class NestedUnconscious(Module)` |
| `NeuroLogos` | class | `bicameral_v3.py:1182` | `class NeuroLogos(Module)` |
| `ReplayMemory` | class | `bicameral_v3.py:1138` | `class ReplayMemory(Module)` |
| `ResidualBlock` | class | `bicameral_v3.py:486` | `class ResidualBlock(Module)` |
| `RightHemisphere` | class | `bicameral_v3.py:603` | `class RightHemisphere(Module)` |
| `SymbioticBasisRefinement` | class | `bicameral_v3.py:541` | `class SymbioticBasisRefinement(Module)` |
| `TopologicalCompressor` | class | `bicameral_v3.py:727` | `class TopologicalCompressor(Module)` |
| `VisualCortex` | class | `bicameral_v3.py:505` | `class VisualCortex(Module)` |
| `__getitem__` | method | `bicameral_v3.py:1326` | `def __getitem__(self, idx)` |
| `__init__` | method | `bicameral_v3.py:242` | `def __init__(self, neurons, tau_theta)` |
| `__init__` | method | `bicameral_v3.py:258` | `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)` |
| `__init__` | method | `bicameral_v3.py:384` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `bicameral_v3.py:487` | `def __init__(self, in_channels, out_channels, stride)` |
| `__init__` | method | `bicameral_v3.py:506` | `def __init__(self, output_dim, grid_size)` |
| `__init__` | method | `bicameral_v3.py:542` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `bicameral_v3.py:562` | `def __init__(self, in_dim, hid_dim, num_nodes, config)` |
| `__init__` | method | `bicameral_v3.py:573` | `def __init__(self, dim, hidden_dim)` |
| `__init__` | method | `bicameral_v3.py:604` | `def __init__(self, config)` |
| `__init__` | method | `bicameral_v3.py:668` | `def __init__(self)` |
| `__init__` | method | `bicameral_v3.py:685` | `def __init__(self, grid_size, output_dim)` |
| `__init__` | method | `bicameral_v3.py:728` | `def __init__(self, node_dim)` |
| `__init__` | method | `bicameral_v3.py:745` | `def __init__(self)` |
| `__init__` | method | `bicameral_v3.py:847` | `def __init__(self, use_nested)` |
| `__init__` | method | `bicameral_v3.py:861` | `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)` |
| `__init__` | method | `bicameral_v3.py:992` | `def __init__(self)` |
| `__init__` | method | `bicameral_v3.py:1040` | `def __init__(self)` |
| `__init__` | method | `bicameral_v3.py:1083` | `def __init__(self)` |
| `__init__` | method | `bicameral_v3.py:1139` | `def __init__(self, capacity, noise_scale)` |
| `__init__` | method | `bicameral_v3.py:1183` | `def __init__(self, vocab_size, use_nested)` |
| `__init__` | method | `bicameral_v3.py:1281` | `def __init__(self, total_epochs)` |
| `__init__` | method | `bicameral_v3.py:1299` | `def __init__(self)` |
| `__len__` | method | `bicameral_v3.py:1323` | `def __len__(self)` |
| `_create_rotation_matrix` | method | `bicameral_v3.py:780` | `def _create_rotation_matrix(self, dim, angle, device)` |
| `_get_init_state` | method | `bicameral_v3.py:377` | `def _get_init_state(self, thought)` |
| `_get_init_state` | method | `bicameral_v3.py:982` | `def _get_init_state(self, thought)` |
| `_make_layer` | method | `bicameral_v3.py:520` | `def _make_layer(self, in_channels, out_channels, num_blocks, stride)` |
| `compute_activation_entropy` | function | `bicameral_v3.py:132` | `def compute_activation_entropy(activations)` |
| `compute_phi_effective` | function | `bicameral_v3.py:22` | `def compute_phi_effective(activations, k_partitions)` |
| `compute_spatial_diversity` | function | `bicameral_v3.py:75` | `def compute_spatial_diversity(activations)` |
| `consolidate_svd` | method | `bicameral_v3.py:467` | `def consolidate_svd(self, repair_strength, timescale)` |
| `create_grid_adjacency` | method | `bicameral_v3.py:588` | `def create_grid_adjacency(N, connectivity)` |
| `decide` | method | `bicameral_v3.py:1049` | `def decide(self, task_loss_val, richness_val, vn_entropy_val)` |
| `decide` | method | `bicameral_v3.py:1097` | `def decide(self, left_metrics, right_metrics, epoch, total_epochs)` |
| `estimate_coherence` | method | `bicameral_v3.py:1336` | `def estimate_coherence(sentence, templates_per_class)` |
| `forward` | method | `bicameral_v3.py:247` | `def forward(self, activity, dt)` |
| `forward` | method | `bicameral_v3.py:281` | `def forward(self, thought, visual_features, captions, max_len)` |
| `forward` | method | `bicameral_v3.py:405` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `bicameral_v3.py:499` | `def forward(self, x)` |
| `forward` | method | `bicameral_v3.py:526` | `def forward(self, x)` |
| `forward` | method | `bicameral_v3.py:552` | `def forward(self, x)` |
| `forward` | method | `bicameral_v3.py:568` | `def forward(self, x, plasticity_gate)` |
| `forward` | method | `bicameral_v3.py:583` | `def forward(self, nodes, adjacency)` |
| `forward` | method | `bicameral_v3.py:636` | `def forward(self, image, adjacency, plasticity)` |
| `forward` | method | `bicameral_v3.py:681` | `def forward(self, x)` |
| `forward` | method | `bicameral_v3.py:705` | `def forward(self, x)` |
| `forward` | method | `bicameral_v3.py:737` | `def forward(self, nodes, plasticity, transfer_rate)` |
| `forward` | method | `bicameral_v3.py:790` | `def forward(self, visual_features, plasticity, transfer_rate)` |
| `forward` | method | `bicameral_v3.py:853` | `def forward(self, image, callosal_input, plasticity, transfer_rate)` |
| `forward` | method | `bicameral_v3.py:883` | `def forward(self, thought, visual_features, captions, max_len)` |
| `forward` | method | `bicameral_v3.py:999` | `def forward(self, left_repr, right_repr, mode)` |
| `forward` | method | `bicameral_v3.py:1203` | `def forward(self, image, captions, plasticity, transfer_rate, mode, epoch, labels)` |
| `get_liquid_module` | method | `bicameral_v3.py:839` | `def get_liquid_module(self)` |
| `get_plasticity` | method | `bicameral_v3.py:1285` | `def get_plasticity(self, epoch)` |
| `measure_neural_complexity` | function | `bicameral_v3.py:163` | `def measure_neural_complexity(activations)` |
| `measure_spatial_richness` | function | `bicameral_v3.py:212` | `def measure_spatial_richness(activations)` |
| `replay` | method | `bicameral_v3.py:1155` | `def replay(self, batch_size)` |
| `set_epoch` | method | `bicameral_v3.py:1274` | `def set_epoch(self, epoch)` |
| `store` | method | `bicameral_v3.py:1145` | `def store(self, pattern)` |
| `to_float` | method | `bicameral_v3.py:1349` | `def to_float(val)` |
| `top_k_top_p_filtering` | function | `bicameral_v3.py:221` | `def top_k_top_p_filtering(logits, top_k, top_p, filter_value)` |
| `train_logos` | method | `bicameral_v3.py:1355` | `def train_logos(use_nested)` |
| `ChaosAdaptiveFilter_ORIGINAL` | class | `caquita.py:228` | `class ChaosAdaptiveFilter_ORIGINAL(Module)` |
| `DiagnosticConfig` | class | `caquita.py:31` | `class DiagnosticConfig` |
| `DiagnosticModel` | class | `caquita.py:264` | `class DiagnosticModel(Module)` |
| `LiquidNeuron` | class | `caquita.py:78` | `class LiquidNeuron(Module)` |
| `RealWorldEnvironment` | class | `caquita.py:50` | `class RealWorldEnvironment` |
| `TraumaResponseSchedulerV2_FIXED` | class | `caquita.py:168` | `class TraumaResponseSchedulerV2_FIXED(Module)` |
| `TraumaResponseSchedulerV2_ORIGINAL` | class | `caquita.py:103` | `class TraumaResponseSchedulerV2_ORIGINAL(Module)` |
| `__init__` | method | `caquita.py:51` | `def __init__(self)` |
| `__init__` | method | `caquita.py:79` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `caquita.py:105` | `def __init__(self)` |
| `__init__` | method | `caquita.py:170` | `def __init__(self)` |
| `__init__` | method | `caquita.py:230` | `def __init__(self)` |
| `__init__` | method | `caquita.py:265` | `def __init__(self, config, use_liquid, use_trs_original, use_trs_fixed, use_caf)` |
| `detect_chaos` | method | `caquita.py:253` | `def detect_chaos(self, x)` |
| `detect_trauma_level` | method | `caquita.py:122` | `def detect_trauma_level(self, phase_idx, current_metrics)` |
| `detect_trauma_level` | method | `caquita.py:185` | `def detect_trauma_level(self, phase_idx, current_metrics)` |
| `extract_noise_features` | method | `caquita.py:240` | `def extract_noise_features(self, x)` |
| `forward` | method | `caquita.py:88` | `def forward(self, x, plasticity_gate)` |
| `forward` | method | `caquita.py:285` | `def forward(self, x, phase_idx, current_metrics)` |
| `generate_response` | method | `caquita.py:148` | `def generate_response(self, trauma_level, phase_idx, chaos_detected)` |
| `generate_response` | method | `caquita.py:208` | `def generate_response(self, trauma_level, phase_idx, chaos_detected)` |
| `get_batch` | method | `caquita.py:63` | `def get_batch(self, phase, batch_size)` |
| `run_diagnostic_ablation` | method | `caquita.py:406` | `def run_diagnostic_ablation()` |
| `seed_everything` | method | `caquita.py:40` | `def seed_everything(seed)` |
| `train_diagnostic` | method | `caquita.py:327` | `def train_diagnostic(config, env, experiment_name)` |
| `update_phase_performance` | method | `caquita.py:112` | `def update_phase_performance(self, phase_idx, metrics)` |
| `update_phase_performance` | method | `caquita.py:176` | `def update_phase_performance(self, phase_idx, metrics)` |
| `Config` | class | `chatgpt.py:19` | `class Config` |
| `DataEnvironment` | class | `chatgpt.py:41` | `class DataEnvironment` |
| `HomeostaticRegulator` | class | `chatgpt.py:70` | `class HomeostaticRegulator(Module)` |
| `NeuralDiagnostics` | class | `chatgpt.py:191` | `class NeuralDiagnostics` |
| `NeuroPhysioBicameral` | class | `chatgpt.py:140` | `class NeuroPhysioBicameral(Module)` |
| `PhysioNeuron` | class | `chatgpt.py:93` | `class PhysioNeuron(Module)` |
| `__init__` | method | `chatgpt.py:42` | `def __init__(self)` |
| `__init__` | method | `chatgpt.py:71` | `def __init__(self)` |
| `__init__` | method | `chatgpt.py:94` | `def __init__(self, d)` |
| `__init__` | method | `chatgpt.py:141` | `def __init__(self, config)` |
| `__init__` | method | `chatgpt.py:192` | `def __init__(self)` |
| `avg` | method | `chatgpt.py:208` | `def avg(self, k, n)` |
| `count_parameters` | method | `chatgpt.py:163` | `def count_parameters(self)` |
| `forward` | method | `chatgpt.py:81` | `def forward(self, stress, excitation, fatigue, loss_signal)` |
| `forward` | method | `chatgpt.py:107` | `def forward(self, x, task_loss)` |
| `forward` | method | `chatgpt.py:166` | `def forward(self, x, task_loss)` |
| `get_batch` | method | `chatgpt.py:53` | `def get_batch(self, phase, bs)` |
| `report` | method | `chatgpt.py:211` | `def report(self, step, phase)` |
| `seed_all` | method | `chatgpt.py:33` | `def seed_all(seed)` |
| `train` | method | `chatgpt.py:225` | `def train()` |
| `update` | method | `chatgpt.py:201` | `def update(self, loss, liquid_norm, phys)` |
| `ConsciousnessModule` | class | `cifar3.py:147` | `class ConsciousnessModule(Module)` |
| `DualSystemModule` | class | `cifar3.py:129` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `cifar3.py:51` | `class FastSlowLinear(Module)` |
| `OmniBrainFastSlow` | class | `cifar3.py:193` | `class OmniBrainFastSlow(Module)` |
| `__init__` | method | `cifar3.py:56` | `def __init__(self, in_features, out_features, fast_lr, fast_decay)` |
| `__init__` | method | `cifar3.py:130` | `def __init__(self, dim)` |
| `__init__` | method | `cifar3.py:149` | `def __init__(self, features)` |
| `__init__` | method | `cifar3.py:194` | `def __init__(self)` |
| `compute_phi_effective` | function | `cifar3.py:30` | `def compute_phi_effective(activity)` |
| `compute_phi_effective_robust` | method | `cifar3.py:160` | `def compute_phi_effective_robust(self, activity)` |
| `end_of_batch` | method | `cifar3.py:116` | `def end_of_batch(self)` |
| `evaluate` | method | `cifar3.py:261` | `def evaluate(model, loader, device)` |
| `forward` | method | `cifar3.py:105` | `def forward(self, x)` |
| `forward` | method | `cifar3.py:138` | `def forward(self, x)` |
| `forward` | method | `cifar3.py:184` | `def forward(self, x)` |
| `forward` | method | `cifar3.py:226` | `def forward(self, x)` |
| `get_cifar10_loaders` | method | `cifar3.py:245` | `def get_cifar10_loaders(batch_size)` |
| `get_fast_norm` | method | `cifar3.py:120` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `cifar3.py:238` | `def get_fast_norms(self)` |
| `reset_all_fast_weights` | method | `cifar3.py:233` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `cifar3.py:74` | `def reset_fast_weights(self)` |
| `train` | method | `cifar3.py:274` | `def train()` |
| `update_fast_weights` | method | `cifar3.py:79` | `def update_fast_weights(self, x)` |
| `ConsciousnessModule` | class | `cifar4.py:134` | `class ConsciousnessModule(Module)` |
| `DualSystemModule` | class | `cifar4.py:116` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `cifar4.py:51` | `class FastSlowLinear(Module)` |
| `OmniBrainFastSlow` | class | `cifar4.py:176` | `class OmniBrainFastSlow(Module)` |
| `__init__` | method | `cifar4.py:56` | `def __init__(self, in_features, out_features, fast_lr, fast_decay)` |
| `__init__` | method | `cifar4.py:117` | `def __init__(self, dim)` |
| `__init__` | method | `cifar4.py:136` | `def __init__(self, features)` |
| `__init__` | method | `cifar4.py:177` | `def __init__(self)` |
| `compute_phi_effective` | function | `cifar4.py:30` | `def compute_phi_effective(activity)` |
| `compute_phi_effective_robust` | method | `cifar4.py:147` | `def compute_phi_effective_robust(self, activity)` |
| `end_of_batch` | method | `cifar4.py:106` | `def end_of_batch(self)` |
| `evaluate` | method | `cifar4.py:238` | `def evaluate(model, loader, device)` |
| `forward` | method | `cifar4.py:95` | `def forward(self, x)` |
| `forward` | method | `cifar4.py:125` | `def forward(self, x)` |
| `forward` | method | `cifar4.py:166` | `def forward(self, x)` |
| `forward` | method | `cifar4.py:197` | `def forward(self, x)` |
| `get_cifar10_loaders` | method | `cifar4.py:217` | `def get_cifar10_loaders(batch_size)` |
| `get_fast_norm` | method | `cifar4.py:109` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `cifar4.py:209` | `def get_fast_norms(self)` |
| `reset_all_fast_weights` | method | `cifar4.py:204` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `cifar4.py:72` | `def reset_fast_weights(self)` |
| `train` | method | `cifar4.py:255` | `def train()` |
| `update_fast_weights` | method | `cifar4.py:76` | `def update_fast_weights(self, x)` |
| `AutoRegulationSystem` | class | `demo_auto_regulation.py:77` | `class AutoRegulationSystem` |
| `Config` | class | `demo_auto_regulation.py:21` | `class Config` |
| `DataEnvironment` | class | `demo_auto_regulation.py:43` | `class DataEnvironment` |
| `PhysioChimeraFixed` | class | `demo_auto_regulation.py:159` | `class PhysioChimeraFixed(Module)` |
| `SelfModifyingGates` | class | `demo_auto_regulation.py:108` | `class SelfModifyingGates(Module)` |
| `__init__` | method | `demo_auto_regulation.py:44` | `def __init__(self)` |
| `__init__` | method | `demo_auto_regulation.py:78` | `def __init__(self, size)` |
| `__init__` | method | `demo_auto_regulation.py:109` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `demo_auto_regulation.py:160` | `def __init__(self, config)` |
| `demo_auto_regulation` | method | `demo_auto_regulation.py:243` | `def demo_auto_regulation()` |
| `forward` | method | `demo_auto_regulation.py:126` | `def forward(self, x, adaptation_state)` |
| `forward` | method | `demo_auto_regulation.py:184` | `def forward(self, x, global_step, phase, prev_loss)` |
| `get_batch` | method | `demo_auto_regulation.py:54` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `demo_auto_regulation.py:68` | `def get_full(self)` |
| `get_stability` | method | `demo_auto_regulation.py:99` | `def get_stability(self)` |
| `get_w2` | method | `demo_auto_regulation.py:71` | `def get_w2(self)` |
| `seed_everything` | method | `demo_auto_regulation.py:33` | `def seed_everything(seed)` |
| `update` | method | `demo_auto_regulation.py:84` | `def update(self, input_variance, loss_gradient, phase)` |
| `visualize_uased_geometry` | function | `difract.py:4` | `def visualize_uased_geometry()` |
| `AdaptiveMagnitudeGate` | class | `dmg_core.py:14` | `class AdaptiveMagnitudeGate(Module)` |
| `DMGNetwork` | class | `dmg_core.py:82` | `class DMGNetwork(Module)` |
| `SparseTopologyLayer` | class | `dmg_core.py:44` | `class SparseTopologyLayer(Module)` |
| `__init__` | method | `dmg_core.py:21` | `def __init__(self, base_threshold, power_order)` |
| `__init__` | method | `dmg_core.py:49` | `def __init__(self, in_features, out_features, sparsity_k)` |
| `__init__` | method | `dmg_core.py:87` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `_generate_sparse_mask` | method | `dmg_core.py:62` | `def _generate_sparse_mask(self, k_neighbors)` |
| `forward` | method | `dmg_core.py:30` | `def forward(self, x)` |
| `forward` | method | `dmg_core.py:77` | `def forward(self, x)` |
| `forward` | method | `dmg_core.py:100` | `def forward(self, x)` |
| `ConsciousSystem` | class | `dualmind.py:108` | `class ConsciousSystem(Module)` |
| `DualMind` | class | `dualmind.py:279` | `class DualMind(Module)` |
| `HomeostasisEngine` | class | `dualmind.py:30` | `class HomeostasisEngine(Module)` |
| `LiquidNeuron` | class | `dualmind.py:57` | `class LiquidNeuron(Module)` |
| `NestedTopoLayer` | class | `dualmind.py:171` | `class NestedTopoLayer(Module)` |
| `UnconsciousSystem` | class | `dualmind.py:217` | `class UnconsciousSystem(Module)` |
| `__init__` | method | `dualmind.py:31` | `def __init__(self)` |
| `__init__` | method | `dualmind.py:59` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `dualmind.py:113` | `def __init__(self, unconscious_dim, d_hid, d_out)` |
| `__init__` | method | `dualmind.py:176` | `def __init__(self, in_dim, hid_dim, num_nodes)` |
| `__init__` | method | `dualmind.py:222` | `def __init__(self, in_channels, grid_size, hidden_dim)` |
| `__init__` | method | `dualmind.py:285` | `def __init__(self, in_channels, grid_size, hidden_dim, conscious_dim, num_classes)` |
| `consolidate_svd` | method | `dualmind.py:89` | `def consolidate_svd(self, repair_strength)` |
| `decide` | method | `dualmind.py:35` | `def decide(self, task_loss_val, richness_val, vn_entropy_val)` |
| `evaluate_dualmind` | method | `dualmind.py:579` | `def evaluate_dualmind(model, test_loader, device)` |
| `forward` | method | `dualmind.py:67` | `def forward(self, x, plasticity_gate)` |
| `forward` | method | `dualmind.py:136` | `def forward(self, unconscious_features, plasticity_gate)` |
| `forward` | method | `dualmind.py:188` | `def forward(self, x_nodes, plasticity_gate)` |
| `forward` | method | `dualmind.py:246` | `def forward(self, x, plasticity_gate)` |
| `forward` | method | `dualmind.py:305` | `def forward(self, x, mode)` |
| `get_structure_entropy` | method | `dualmind.py:157` | `def get_structure_entropy(self)` |
| `get_system_status` | method | `dualmind.py:331` | `def get_system_status(self)` |
| `get_topology_density` | method | `dualmind.py:209` | `def get_topology_density(self)` |
| `get_topology_stats` | method | `dualmind.py:264` | `def get_topology_stats(self)` |
| `measure_spatial_richness` | function | `dualmind.py:15` | `def measure_spatial_richness(activations)` |
| `run_dualmind_experiment` | method | `dualmind.py:604` | `def run_dualmind_experiment()` |
| `train_dualmind_phase1` | method | `dualmind.py:346` | `def train_dualmind_phase1(model, train_loader, optimizer, device, epochs)` |
| `train_dualmind_phase2` | method | `dualmind.py:401` | `def train_dualmind_phase2(model, train_loader, optimizer, device, epochs)` |
| `train_dualmind_phase3` | method | `dualmind.py:487` | `def train_dualmind_phase3(model, train_loader, optimizer, device, epochs)` |
| `DataEnvironment` | class | `dynamic.py:36` | `class DataEnvironment` |
| `LiquidNeuron` | class | `dynamic.py:100` | `class LiquidNeuron(Module)` |
| `MasterConfig` | class | `dynamic.py:24` | `class MasterConfig` |
| `OmnibusController` | class | `dynamic.py:54` | `class OmnibusController(Module)` |
| `SovereignAttention` | class | `dynamic.py:88` | `class SovereignAttention(Module)` |
| `SovereignChimera` | class | `dynamic.py:137` | `class SovereignChimera(Module)` |
| `__init__` | method | `dynamic.py:37` | `def __init__(self)` |
| `__init__` | method | `dynamic.py:55` | `def __init__(self)` |
| `__init__` | method | `dynamic.py:89` | `def __init__(self, d_in)` |
| `__init__` | method | `dynamic.py:101` | `def __init__(self, d_in, d_out)` |
| `__init__` | method | `dynamic.py:138` | `def __init__(self, config, dynamic_mode)` |
| `forward` | method | `dynamic.py:66` | `def forward(self, x, h_slow)` |
| `forward` | method | `dynamic.py:94` | `def forward(self, x, gain)` |
| `forward` | method | `dynamic.py:113` | `def forward(self, x, plasticity, alpha)` |
| `forward` | method | `dynamic.py:151` | `def forward(self, x)` |
| `get_batch` | method | `dynamic.py:46` | `def get_batch(self, phase, bs)` |
| `run_final_showdown` | method | `dynamic.py:184` | `def run_final_showdown(epochs, name, dynamic)` |
| `seed_everything` | function | `dynamic.py:13` | `def seed_everything(seed)` |
| `DataEnvironment` | class | `dynamic2.py:36` | `class DataEnvironment` |
| `HomeostaticRegulator` | class | `dynamic2.py:54` | `class HomeostaticRegulator(Module)` |
| `PhysioChimera` | class | `dynamic2.py:162` | `class PhysioChimera(Module)` |
| `PhysioConfig` | class | `dynamic2.py:24` | `class PhysioConfig` |
| `PhysioNeuron` | class | `dynamic2.py:89` | `class PhysioNeuron(Module)` |
| `__init__` | method | `dynamic2.py:37` | `def __init__(self)` |
| `__init__` | method | `dynamic2.py:55` | `def __init__(self, d_in)` |
| `__init__` | method | `dynamic2.py:90` | `def __init__(self, d_in, d_out, dynamic_mode)` |
| `__init__` | method | `dynamic2.py:163` | `def __init__(self, config, dynamic_mode)` |
| `forward` | method | `dynamic2.py:66` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `dynamic2.py:105` | `def forward(self, x)` |
| `forward` | method | `dynamic2.py:169` | `def forward(self, x)` |
| `get_batch` | method | `dynamic2.py:46` | `def get_batch(self, phase, bs)` |
| `run_physio_experiment` | method | `dynamic2.py:178` | `def run_physio_experiment(epochs, name, dynamic)` |
| `seed_everything` | function | `dynamic2.py:13` | `def seed_everything(seed)` |
| `create_demo_report` | function | `example_usage.py:194` | `def create_demo_report()` |
| `demo_checkpoint_system` | function | `example_usage.py:85` | `def demo_checkpoint_system()` |
| `demo_comparison_experiments` | function | `example_usage.py:142` | `def demo_comparison_experiments()` |
| `demo_custom_monitoring` | function | `example_usage.py:38` | `def demo_custom_monitoring()` |
| `demo_simple_monitoring` | function | `example_usage.py:20` | `def demo_simple_monitoring()` |
| `main` | function | `example_usage.py:333` | `def main()` |
| `run_single_experiment` | function | `exampleww.py:8` | `def run_single_experiment(model_name, seed, epochs)` |
| `AudioEncoder` | class | `exodia_op_2.py:1833` | `class AudioEncoder(Module)` |
| `CausalReasoningEngine` | class | `exodia_op_2.py:1008` | `class CausalReasoningEngine(Module)` |
| `CorpusCallosumTrimodal` | class | `exodia_op_2.py:1984` | `class CorpusCallosumTrimodal(Module)` |
| `EnhancedDiagnosticsTricameral` | class | `exodia_op_2.py:2207` | `class EnhancedDiagnosticsTricameral` |
| `Flickr8kMultimodalDataset` | class | `exodia_op_2.py:2550` | `class Flickr8kMultimodalDataset(Dataset)` |
| `HierarchicalEpisodicMemory` | class | `exodia_op_2.py:340` | `class HierarchicalEpisodicMemory` |
| `LanguageMetrics` | class | `exodia_op_2.py:774` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `exodia_op_2.py:965` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `exodia_op_2.py:1087` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `exodia_op_2.py:1510` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `exodia_op_2.py:848` | `class LinguisticFeedbackLoop` |
| `NeuroLogosTricameral` | class | `exodia_op_2.py:2515` | `class NeuroLogosTricameral(Module)` |
| `NeurocognitiveSystem` | class | `exodia_op_2.py:577` | `class NeurocognitiveSystem` |
| `RightHemisphereTricameral` | class | `exodia_op_2.py:1901` | `class RightHemisphereTricameral(Module)` |
| `StableLiquidNeuron` | class | `exodia_op_2.py:1134` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `exodia_op_2.py:1360` | `class TriangulatedMedicalSystem` |
| `TricameralOutput` | class | `exodia_op_2.py:1321` | `class TricameralOutput(NamedTuple)` |
| `__getitem__` | method | `exodia_op_2.py:2613` | `def __getitem__(self, idx)` |
| `__init__` | method | `exodia_op_2.py:347` | `def __init__(self, working_capacity, short_term_capacity, importance_threshold)` |
| `__init__` | method | `exodia_op_2.py:578` | `def __init__(self)` |
| `__init__` | method | `exodia_op_2.py:849` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `exodia_op_2.py:1009` | `def __init__(self, hidden_dim)` |
| `__init__` | method | `exodia_op_2.py:1136` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `exodia_op_2.py:1361` | `def __init__(self)` |
| `__init__` | method | `exodia_op_2.py:1511` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `exodia_op_2.py:1842` | `def __init__(self, output_dim)` |
| `__init__` | method | `exodia_op_2.py:1909` | `def __init__(self, output_dim)` |
| `__init__` | method | `exodia_op_2.py:1994` | `def __init__(self, dim)` |
| `__init__` | method | `exodia_op_2.py:2208` | `def __init__(self)` |
| `__init__` | method | `exodia_op_2.py:2518` | `def __init__(self, vocab_size)` |
| `__init__` | method | `exodia_op_2.py:2553` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache, cache_di` |
| `__len__` | method | `exodia_op_2.py:2610` | `def __len__(self)` |
| `_apply_chain_of_thought` | method | `exodia_op_2.py:1653` | `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)` |
| `_apply_flash_attention` | method | `exodia_op_2.py:2057` | `def _apply_flash_attention(self, x)` |
| `_apply_multi_token_prediction` | method | `exodia_op_2.py:1694` | `def _apply_multi_token_prediction(self, hidden_states, input_ids)` |
| `_apply_structural_attention` | method | `exodia_op_2.py:1737` | `def _apply_structural_attention(self, lstm_out, channels, visual_context)` |
| `_calculate_homeostasis_metric` | method | `exodia_op_2.py:1219` | `def _calculate_homeostasis_metric(self, output)` |
| `_calculate_novelty` | method | `exodia_op_2.py:402` | `def _calculate_novelty(self, episode)` |
| `_get_cached_norm` | method | `exodia_op_2.py:2231` | `def _get_cached_norm(self, tensor, dim)` |
| `_get_init_state` | method | `exodia_op_2.py:1819` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `exodia_op_2.py:812` | `def _get_ngrams(tokens, n)` |
| `_get_ngrams_cached` | method | `exodia_op_2.py:863` | `def _get_ngrams_cached(sentence, n)` |
| `_greedy_decode` | method | `exodia_op_2.py:1759` | `def _greedy_decode(self, visual_context, channels, max_len, epoch)` |
| `_predict_interventions` | method | `exodia_op_2.py:1050` | `def _predict_interventions(self, hypothesis, confidence)` |
| `_purge_low_score_memories` | method | `exodia_op_2.py:545` | `def _purge_low_score_memories(self)` |
| `_reset_liquid_neuron` | method | `exodia_op_2.py:1496` | `def _reset_liquid_neuron(self, liquid_neuron)` |
| `_sample_from_buffer` | method | `exodia_op_2.py:494` | `def _sample_from_buffer(self, buffer, scores, batch_size)` |
| `_update_unified_buffer` | method | `exodia_op_2.py:456` | `def _update_unified_buffer(self)` |
| `adjust_gates_by_fatigue` | method | `exodia_op_2.py:2190` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `exodia_op_2.py:688` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_emergency_fixes` | function | `exodia_op_2.py:120` | `def apply_emergency_fixes(model)` |
| `apply_forgetting_curve` | method | `exodia_op_2.py:535` | `def apply_forgetting_curve(self)` |
| `apply_triangulated_intervention` | method | `exodia_op_2.py:1427` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `exodia_op_2.py:642` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `assess_reasoning_state` | method | `exodia_op_2.py:598` | `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` |
| `build_vocab_flickr` | function | `exodia_op_2.py:316` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `exodia_op_2.py:2353` | `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_importance` | method | `exodia_op_2.py:386` | `def calculate_importance(self, episode, surprise_score)` |
| `calculate_synergy` | method | `exodia_op_2.py:2342` | `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_alignment_loss` | method | `exodia_op_2.py:2666` | `def compute_alignment_loss(visual_features, channels, alpha, epoch)` |
| `compute_cider` | method | `exodia_op_2.py:911` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `exodia_op_2.py:872` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_spice` | method | `exodia_op_2.py:925` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `exodia_op_2.py:371` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `compute_tricameral_loss` | method | `exodia_op_2.py:2695` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, channel` |
| `count_convergent_signals` | method | `exodia_op_2.py:1379` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `exodia_op_2.py:1382` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)` |
| `evaluate_reasoning_quality` | method | `exodia_op_2.py:2305` | `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` |
| `forward` | method | `exodia_op_2.py:1184` | `def forward(self, x)` |
| `forward` | method | `exodia_op_2.py:1335` | `def forward(self, image, audio, captions, epoch)` |
| `forward` | method | `exodia_op_2.py:1596` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `exodia_op_2.py:1880` | `def forward(self, mel_spec)` |
| `forward` | method | `exodia_op_2.py:1947` | `def forward(self, image, audio)` |
| `forward` | method | `exodia_op_2.py:2090` | `def forward(self, right_features)` |
| `forward` | method | `exodia_op_2.py:2525` | `def forward(self, image, audio, captions, epoch)` |
| `get_cache_stats` | method | `exodia_op_2.py:937` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `exodia_op_2.py:2379` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `exodia_op_2.py:1229` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `exodia_op_2.py:2248` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `preprocess_and_cache_spectrograms` | function | `exodia_op_2.py:49` | `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)` |
| `query_causal_chain` | method | `exodia_op_2.py:1073` | `def query_causal_chain(self, start_node, end_node)` |
| `reason_causally` | method | `exodia_op_2.py:1036` | `def reason_causally(self, observation, context)` |
| `report` | method | `exodia_op_2.py:2429` | `def report(self, epoch)` |
| `sample` | method | `exodia_op_2.py:470` | `def sample(self, batch_size, memory_level)` |
| `sentence_bleu` | method | `exodia_op_2.py:778` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `exodia_op_2.py:967` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `exodia_op_2.py:1089` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k_with_audio` | function | `exodia_op_2.py:142` | `def setup_flickr8k_with_audio(data_dir)` |
| `store_episode` | method | `exodia_op_2.py:427` | `def store_episode(self, image, audio, caption, surprise_score)` |
| `token_accuracy` | method | `exodia_op_2.py:821` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `exodia_op_2.py:990` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `exodia_op_2.py:1112` | `def token_accuracy(reference, hypothesis)` |
| `train_tricameral` | method | `exodia_op_2.py:2812` | `def train_tricameral()` |
| `triangulate_signals` | method | `exodia_op_2.py:1368` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `exodia_op_2.py:2362` | `def update(self)` |
| `update_channel_fatigue` | method | `exodia_op_2.py:2169` | `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` |
| `update_knowledge_graph` | method | `exodia_op_2.py:1067` | `def update_knowledge_graph(self, cause, effect, strength)` |
| `update_physiology_advanced` | method | `exodia_op_2.py:1278` | `def update_physiology_advanced(self, loss_value)` |
| `visualize_fatigue_distribution` | method | `exodia_op_2.py:2395` | `def visualize_fatigue_distribution(self, epoch)` |
| `visualize_reasoning_metrics` | method | `exodia_op_2.py:2417` | `def visualize_reasoning_metrics(self, epoch)` |
| `word_overlap` | method | `exodia_op_2.py:834` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `exodia_op_2.py:1000` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `exodia_op_2.py:1122` | `def word_overlap(reference, hypothesis)` |
| `AudioEncoder` | class | `exodia_optimized.py:1703` | `class AudioEncoder(Module)` |
| `CausalReasoningEngine` | class | `exodia_optimized.py:977` | `class CausalReasoningEngine(Module)` |
| `CorpusCallosumTrimodal` | class | `exodia_optimized.py:1861` | `class CorpusCallosumTrimodal(Module)` |
| `EnhancedDiagnosticsTricameral` | class | `exodia_optimized.py:2093` | `class EnhancedDiagnosticsTricameral` |
| `Flickr8kMultimodalDataset` | class | `exodia_optimized.py:2433` | `class Flickr8kMultimodalDataset(Dataset)` |
| `HierarchicalEpisodicMemory` | class | `exodia_optimized.py:332` | `class HierarchicalEpisodicMemory` |
| `LanguageMetrics` | class | `exodia_optimized.py:743` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `exodia_optimized.py:934` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `exodia_optimized.py:1056` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `exodia_optimized.py:1394` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `exodia_optimized.py:817` | `class LinguisticFeedbackLoop` |
| `NeuroLogosTricameral` | class | `exodia_optimized.py:2398` | `class NeuroLogosTricameral(Module)` |
| `NeurocognitiveSystem` | class | `exodia_optimized.py:546` | `class NeurocognitiveSystem` |
| `RightHemisphereTricameral` | class | `exodia_optimized.py:1778` | `class RightHemisphereTricameral(Module)` |
| `StableLiquidNeuron` | class | `exodia_optimized.py:1103` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `exodia_optimized.py:1243` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `exodia_optimized.py:2496` | `def __getitem__(self, idx)` |
| `__init__` | method | `exodia_optimized.py:341` | `def __init__(self, working_capacity, short_term_capacity, importance_threshold)` |
| `__init__` | method | `exodia_optimized.py:547` | `def __init__(self)` |
| `__init__` | method | `exodia_optimized.py:818` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `exodia_optimized.py:978` | `def __init__(self, hidden_dim)` |
| `__init__` | method | `exodia_optimized.py:1104` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `exodia_optimized.py:1244` | `def __init__(self)` |
| `__init__` | method | `exodia_optimized.py:1395` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `exodia_optimized.py:1711` | `def __init__(self, output_dim)` |
| `__init__` | method | `exodia_optimized.py:1786` | `def __init__(self, output_dim)` |
| `__init__` | method | `exodia_optimized.py:1871` | `def __init__(self, dim)` |
| `__init__` | method | `exodia_optimized.py:2094` | `def __init__(self)` |
| `__init__` | method | `exodia_optimized.py:2401` | `def __init__(self, vocab_size)` |
| `__init__` | method | `exodia_optimized.py:2436` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache, cache_di` |
| `__len__` | method | `exodia_optimized.py:2493` | `def __len__(self)` |
| `_apply_chain_of_thought` | method | `exodia_optimized.py:1524` | `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)` |
| `_apply_flash_attention` | method | `exodia_optimized.py:1940` | `def _apply_flash_attention(self, x)` |
| `_apply_multi_token_prediction` | method | `exodia_optimized.py:1625` | `def _apply_multi_token_prediction(self, hidden_states, input_ids)` |
| `_apply_structural_attention` | method | `exodia_optimized.py:1667` | `def _apply_structural_attention(self, lstm_out, channels, visual_context)` |
| `_calculate_homeostasis_metric` | method | `exodia_optimized.py:1163` | `def _calculate_homeostasis_metric(self, output)` |
| `_calculate_novelty` | method | `exodia_optimized.py:393` | `def _calculate_novelty(self, episode)` |
| `_get_cached_norm` | method | `exodia_optimized.py:2117` | `def _get_cached_norm(self, tensor, dim)` |
| `_get_init_state` | method | `exodia_optimized.py:1688` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `exodia_optimized.py:781` | `def _get_ngrams(tokens, n)` |
| `_get_ngrams_cached` | method | `exodia_optimized.py:832` | `def _get_ngrams_cached(sentence, n)` |
| `_greedy_decode` | method | `exodia_optimized.py:1564` | `def _greedy_decode(self, visual_context, channels, max_len, epoch)` |
| `_predict_interventions` | method | `exodia_optimized.py:1019` | `def _predict_interventions(self, hypothesis, confidence)` |
| `_purge_low_score_memories` | method | `exodia_optimized.py:467` | `def _purge_low_score_memories(self)` |
| `_reset_liquid_neuron` | method | `exodia_optimized.py:1379` | `def _reset_liquid_neuron(self, liquid_neuron)` |
| `_sample_from_buffer` | method | `exodia_optimized.py:511` | `def _sample_from_buffer(self, buffer, scores, batch_size)` |
| `_update_unified_buffer` | method | `exodia_optimized.py:445` | `def _update_unified_buffer(self)` |
| `add` | method | `exodia_optimized.py:450` | `def add(self, image, audio, caption, surprise_score)` |
| `adjust_gates_by_fatigue` | method | `exodia_optimized.py:2076` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `exodia_optimized.py:657` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_forgetting_curve` | method | `exodia_optimized.py:454` | `def apply_forgetting_curve(self)` |
| `apply_triangulated_intervention` | method | `exodia_optimized.py:1310` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `exodia_optimized.py:611` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `assess_reasoning_state` | method | `exodia_optimized.py:567` | `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` |
| `build_vocab_flickr` | function | `exodia_optimized.py:308` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `exodia_optimized.py:2236` | `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_importance` | method | `exodia_optimized.py:380` | `def calculate_importance(self, episode, surprise_score)` |
| `calculate_synergy` | method | `exodia_optimized.py:2225` | `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_alignment_loss` | method | `exodia_optimized.py:2549` | `def compute_alignment_loss(visual_features, channels, alpha, epoch)` |
| `compute_cider` | method | `exodia_optimized.py:880` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `exodia_optimized.py:841` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_spice` | method | `exodia_optimized.py:894` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `exodia_optimized.py:369` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `compute_tricameral_loss` | method | `exodia_optimized.py:2577` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, channel` |
| `count_convergent_signals` | method | `exodia_optimized.py:1262` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `exodia_optimized.py:1265` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)` |
| `evaluate_reasoning_quality` | method | `exodia_optimized.py:2188` | `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` |
| `forward` | method | `exodia_optimized.py:1146` | `def forward(self, x)` |
| `forward` | method | `exodia_optimized.py:1477` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `exodia_optimized.py:1751` | `def forward(self, mel_spec)` |
| `forward` | method | `exodia_optimized.py:1824` | `def forward(self, image, audio)` |
| `forward` | method | `exodia_optimized.py:1966` | `def forward(self, right_features)` |
| `forward` | method | `exodia_optimized.py:2407` | `def forward(self, image, audio, captions, epoch)` |
| `get_cache_stats` | method | `exodia_optimized.py:906` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `exodia_optimized.py:2262` | `def get_recent_avg(self, key, n)` |
| `get_total_size` | method | `exodia_optimized.py:539` | `def get_total_size(self)` |
| `hebbian_update` | method | `exodia_optimized.py:1172` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `exodia_optimized.py:2135` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `preprocess_and_cache_spectrograms` | function | `exodia_optimized.py:47` | `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)` |
| `query_causal_chain` | method | `exodia_optimized.py:1042` | `def query_causal_chain(self, start_node, end_node)` |
| `reason_causally` | method | `exodia_optimized.py:1005` | `def reason_causally(self, observation, context)` |
| `report` | method | `exodia_optimized.py:2314` | `def report(self, epoch)` |
| `sample` | method | `exodia_optimized.py:487` | `def sample(self, batch_size, memory_level)` |
| `sentence_bleu` | method | `exodia_optimized.py:747` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `exodia_optimized.py:936` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `exodia_optimized.py:1058` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k_with_audio` | function | `exodia_optimized.py:120` | `def setup_flickr8k_with_audio(data_dir)` |
| `store_episode` | method | `exodia_optimized.py:416` | `def store_episode(self, image, audio, caption, surprise_score)` |
| `token_accuracy` | method | `exodia_optimized.py:790` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `exodia_optimized.py:959` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `exodia_optimized.py:1081` | `def token_accuracy(reference, hypothesis)` |
| `train_tricameral` | method | `exodia_optimized.py:2650` | `def train_tricameral()` |
| `triangulate_signals` | method | `exodia_optimized.py:1251` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `exodia_optimized.py:2245` | `def update(self)` |
| `update_channel_fatigue` | method | `exodia_optimized.py:2054` | `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` |
| `update_knowledge_graph` | method | `exodia_optimized.py:1036` | `def update_knowledge_graph(self, cause, effect, strength)` |
| `update_physiology_advanced` | method | `exodia_optimized.py:1210` | `def update_physiology_advanced(self, loss_value)` |
| `visualize_fatigue_distribution` | method | `exodia_optimized.py:2278` | `def visualize_fatigue_distribution(self, epoch)` |
| `visualize_reasoning_metrics` | method | `exodia_optimized.py:2302` | `def visualize_reasoning_metrics(self, epoch)` |
| `word_overlap` | method | `exodia_optimized.py:803` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `exodia_optimized.py:969` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `exodia_optimized.py:1091` | `def word_overlap(reference, hypothesis)` |
| `SinergyAnalysis` | class | `final_sinergy_analysis.py:12` | `class SinergyAnalysis` |
| `__init__` | method | `final_sinergy_analysis.py:13` | `def __init__(self)` |
| `analyze_original_models` | method | `final_sinergy_analysis.py:87` | `def analyze_original_models(self)` |
| `analyze_sinergies` | method | `final_sinergy_analysis.py:99` | `def analyze_sinergies(self)` |
| `calculate_synergy_breakthrough` | method | `final_sinergy_analysis.py:136` | `def calculate_synergy_breakthrough(self)` |
| `generate_conclusion` | method | `final_sinergy_analysis.py:171` | `def generate_conclusion(self)` |
| `generate_scientific_matrix` | method | `final_sinergy_analysis.py:118` | `def generate_scientific_matrix(self)` |
| `main` | method | `final_sinergy_analysis.py:219` | `def main()` |
| `print_header` | method | `final_sinergy_analysis.py:80` | `def print_header(self)` |
| `save_results` | method | `final_sinergy_analysis.py:200` | `def save_results(self)` |
| `CorpusCallosum` | class | `gemini.py:196` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `gemini.py:431` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphere` | class | `gemini.py:219` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `gemini.py:493` | `class LifeCycle` |
| `LiquidNeuron` | class | `gemini.py:108` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `gemini.py:361` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `gemini.py:333` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `gemini.py:177` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `gemini.py:453` | `def __getitem__(self, idx)` |
| `__init__` | method | `gemini.py:109` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `gemini.py:178` | `def __init__(self, output_dim)` |
| `__init__` | method | `gemini.py:197` | `def __init__(self, dim)` |
| `__init__` | method | `gemini.py:220` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `gemini.py:334` | `def __init__(self, vocab_size)` |
| `__init__` | method | `gemini.py:362` | `def __init__(self)` |
| `__init__` | method | `gemini.py:432` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `gemini.py:494` | `def __init__(self, total_epochs)` |
| `__len__` | method | `gemini.py:450` | `def __len__(self)` |
| `_get_init_state` | method | `gemini.py:313` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `gemini.py:318` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `gemini.py:470` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `consolidate_svd` | method | `gemini.py:154` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `gemini.py:123` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `gemini.py:187` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `gemini.py:210` | `def forward(self, right_features)` |
| `forward` | method | `gemini.py:241` | `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)` |
| `forward` | method | `gemini.py:340` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)` |
| `get_plasticity` | method | `gemini.py:497` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `gemini.py:389` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `gemini.py:373` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `gemini.py:380` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `gemini.py:394` | `def report(self, epoch)` |
| `setup_flickr8k` | function | `gemini.py:40` | `def setup_flickr8k(data_dir)` |
| `train_bicameral` | method | `gemini.py:509` | `def train_bicameral()` |
| `update` | method | `gemini.py:384` | `def update(self)` |
| `CorpusCallosum` | class | `gemini2.py:196` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `gemini2.py:431` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphere` | class | `gemini2.py:219` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `gemini2.py:493` | `class LifeCycle` |
| `LiquidNeuron` | class | `gemini2.py:108` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `gemini2.py:361` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `gemini2.py:333` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `gemini2.py:177` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `gemini2.py:453` | `def __getitem__(self, idx)` |
| `__init__` | method | `gemini2.py:109` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `gemini2.py:178` | `def __init__(self, output_dim)` |
| `__init__` | method | `gemini2.py:197` | `def __init__(self, dim)` |
| `__init__` | method | `gemini2.py:220` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `gemini2.py:334` | `def __init__(self, vocab_size)` |
| `__init__` | method | `gemini2.py:362` | `def __init__(self)` |
| `__init__` | method | `gemini2.py:432` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `gemini2.py:494` | `def __init__(self, total_epochs)` |
| `__len__` | method | `gemini2.py:450` | `def __len__(self)` |
| `_get_init_state` | method | `gemini2.py:313` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `gemini2.py:318` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `gemini2.py:470` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `consolidate_svd` | method | `gemini2.py:154` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `gemini2.py:123` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `gemini2.py:187` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `gemini2.py:210` | `def forward(self, right_features)` |
| `forward` | method | `gemini2.py:241` | `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)` |
| `forward` | method | `gemini2.py:340` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)` |
| `get_plasticity` | method | `gemini2.py:497` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `gemini2.py:389` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `gemini2.py:373` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `gemini2.py:380` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `gemini2.py:394` | `def report(self, epoch)` |
| `setup_flickr8k` | function | `gemini2.py:40` | `def setup_flickr8k(data_dir)` |
| `train_bicameral` | method | `gemini2.py:509` | `def train_bicameral()` |
| `update` | method | `gemini2.py:384` | `def update(self)` |
| `generate_one` | function | `gen_dataset.py:89` | `def generate_one(key, text)` |
| `main` | function | `gen_dataset.py:110` | `def main()` |
| `compress_audios_only` | function | `get_dataset.py:249` | `def compress_audios_only()` |
| `create_audio_readme` | function | `get_dataset.py:318` | `def create_audio_readme(output_dir, metadata)` |
| `create_split_zips` | function | `get_dataset.py:637` | `def create_split_zips()` |
| `download_captions_only` | function | `get_dataset.py:29` | `def download_captions_only()` |
| `download_flickr8k` | function | `get_dataset.py:540` | `def download_flickr8k()` |
| `generate_audios` | function | `get_dataset.py:602` | `def generate_audios()` |
| `generate_audios_sync` | function | `get_dataset.py:219` | `def generate_audios_sync()` |
| `generate_audios_with_checkpoints` | function | `get_dataset.py:128` | `def generate_audios_with_checkpoints()` |
| `generate_one_audio` | function | `get_dataset.py:67` | `def generate_one_audio(text, output_path, max_retries)` |
| `generate_upload_instructions` | function | `get_dataset.py:717` | `def generate_upload_instructions(metadata)` |
| `load_checkpoint` | function | `get_dataset.py:114` | `def load_checkpoint()` |
| `main` | function | `get_dataset.py:465` | `def main()` |
| `main` | function | `get_dataset.py:888` | `def main()` |
| `save_checkpoint` | function | `get_dataset.py:122` | `def save_checkpoint(checkpoint)` |
| `upload_to_huggingface` | function | `get_dataset.py:388` | `def upload_to_huggingface(dataset_dir)` |
| `upload_to_huggingface` | function | `get_dataset.py:834` | `def upload_to_huggingface(dataset_dir)` |
| `AdaptiveLiquidMemory` | class | `homeostatichope.py:190` | `class AdaptiveLiquidMemory(Module)` |
| `Config` | class | `homeostatichope.py:15` | `class Config` |
| `ConsciousTrainer` | class | `homeostatichope.py:402` | `class ConsciousTrainer` |
| `ContinuumMemorySystem` | class | `homeostatichope.py:278` | `class ContinuumMemorySystem(Module)` |
| `HomeostaticSelfModMemory` | class | `homeostatichope.py:233` | `class HomeostaticSelfModMemory(Module)` |
| `OmniscientHopeModel` | class | `homeostatichope.py:304` | `class OmniscientHopeModel(Module)` |
| `OmniscientRegulator` | class | `homeostatichope.py:96` | `class OmniscientRegulator(Module)` |
| `RealWorldEnvironment` | class | `homeostatichope.py:49` | `class RealWorldEnvironment` |
| `__init__` | method | `homeostatichope.py:50` | `def __init__(self, seed)` |
| `__init__` | method | `homeostatichope.py:102` | `def __init__(self, d_model)` |
| `__init__` | method | `homeostatichope.py:193` | `def __init__(self, d_model)` |
| `__init__` | method | `homeostatichope.py:234` | `def __init__(self, d_model, hidden_dim)` |
| `__init__` | method | `homeostatichope.py:279` | `def __init__(self, frequencies, d_model, hidden_dim)` |
| `__init__` | method | `homeostatichope.py:305` | `def __init__(self, config, n_features, n_classes)` |
| `__init__` | method | `homeostatichope.py:403` | `def __init__(self, model, config, device)` |
| `evaluate` | method | `homeostatichope.py:488` | `def evaluate(self, test_loader, epsilon, phase)` |
| `forward` | method | `homeostatichope.py:128` | `def forward(self, signals)` |
| `forward` | method | `homeostatichope.py:203` | `def forward(self, x, controls)` |
| `forward` | method | `homeostatichope.py:251` | `def forward(self, x, controls)` |
| `forward` | method | `homeostatichope.py:293` | `def forward(self, x, global_step)` |
| `forward` | method | `homeostatichope.py:341` | `def forward(self, x, signals, global_step)` |
| `get_batch` | method | `homeostatichope.py:73` | `def get_batch(self, phase, batch_size)` |
| `get_test_loader` | method | `homeostatichope.py:88` | `def get_test_loader(self, batch_size)` |
| `pgd_attack` | method | `homeostatichope.py:368` | `def pgd_attack(model, x, y, epsilon, steps, device, signals)` |
| `run_ablation` | method | `homeostatichope.py:610` | `def run_ablation(device)` |
| `run_conscious_experiment` | method | `homeostatichope.py:516` | `def run_conscious_experiment(config, device)` |
| `set_seed` | method | `homeostatichope.py:40` | `def set_seed(seed)` |
| `setup_device` | method | `homeostatichope.py:35` | `def setup_device()` |
| `train_step` | method | `homeostatichope.py:425` | `def train_step(self, x, y, epsilon, global_step, phase)` |
| `AdversarialTrainer` | class | `hope.py:441` | `class AdversarialTrainer` |
| `Config` | class | `hope.py:16` | `class Config` |
| `ContinuumMemorySystem` | class | `hope.py:287` | `class ContinuumMemorySystem(Module)` |
| `EfficientSelfModMemory` | class | `hope.py:207` | `class EfficientSelfModMemory(Module)` |
| `HomeostaticRegulator` | class | `hope.py:127` | `class HomeostaticRegulator(Module)` |
| `HopePhysioModel` | class | `hope.py:320` | `class HopePhysioModel(Module)` |
| `LiquidMemory` | class | `hope.py:172` | `class LiquidMemory(Module)` |
| `RealWorldEnvironment` | class | `hope.py:58` | `class RealWorldEnvironment` |
| `__init__` | method | `hope.py:64` | `def __init__(self, seed)` |
| `__init__` | method | `hope.py:130` | `def __init__(self, d_model)` |
| `__init__` | method | `hope.py:175` | `def __init__(self, d_model)` |
| `__init__` | method | `hope.py:210` | `def __init__(self, d_model, hidden_dim)` |
| `__init__` | method | `hope.py:290` | `def __init__(self, frequencies, d_model, hidden_dim)` |
| `__init__` | method | `hope.py:323` | `def __init__(self, config, n_features, n_classes)` |
| `__init__` | method | `hope.py:442` | `def __init__(self, model, config, device)` |
| `evaluate` | method | `hope.py:490` | `def evaluate(self, test_loader, epsilon)` |
| `forward` | method | `hope.py:140` | `def forward(self, x, h_prev, w_norm)` |
| `forward` | method | `hope.py:186` | `def forward(self, x, physio)` |
| `forward` | method | `hope.py:236` | `def forward(self, x)` |
| `forward` | method | `hope.py:304` | `def forward(self, x, global_step)` |
| `forward` | method | `hope.py:365` | `def forward(self, x, global_step)` |
| `get_batch` | method | `hope.py:98` | `def get_batch(self, phase, batch_size)` |
| `get_test_loader` | method | `hope.py:118` | `def get_test_loader(self, batch_size)` |
| `pgd_attack` | method | `hope.py:394` | `def pgd_attack(model, x, y, epsilon, steps, device)` |
| `reset_states` | method | `hope.py:361` | `def reset_states(self)` |
| `run_ablation` | method | `hope.py:614` | `def run_ablation(device)` |
| `run_real_world_experiment` | method | `hope.py:517` | `def run_real_world_experiment(config, device)` |
| `set_seed` | method | `hope.py:49` | `def set_seed(seed)` |
| `setup_device` | method | `hope.py:41` | `def setup_device()` |
| `train_step` | method | `hope.py:460` | `def train_step(self, x, y, epsilon, global_step)` |
| `BCMRegulated` | class | `kimi.py:97` | `class BCMRegulated(Module)` |
| `Config` | class | `kimi.py:185` | `class Config` |
| `LiquidRegulated` | class | `kimi.py:118` | `class LiquidRegulated(Module)` |
| `MicroTopoBrainSNA` | class | `kimi.py:166` | `class MicroTopoBrainSNA(Module)` |
| `PhysioState` | class | `kimi.py:28` | `class PhysioState` |
| `SNE` | class | `kimi.py:65` | `class SNE(Module)` |
| `VisualCortexRegulated` | class | `kimi.py:145` | `class VisualCortexRegulated(Module)` |
| `__init__` | method | `kimi.py:66` | `def __init__(self, enabled)` |
| `__init__` | method | `kimi.py:98` | `def __init__(self, sne, ablated)` |
| `__init__` | method | `kimi.py:119` | `def __init__(self, sne, ablated)` |
| `__init__` | method | `kimi.py:146` | `def __init__(self, sne, ablated)` |
| `__init__` | method | `kimi.py:167` | `def __init__(self, sne_enabled, ablated_organs)` |
| `forward` | method | `kimi.py:75` | `def forward(self, state, loss)` |
| `forward` | method | `kimi.py:104` | `def forward(self, act)` |
| `forward` | method | `kimi.py:127` | `def forward(self, x)` |
| `forward` | method | `kimi.py:155` | `def forward(self, img)` |
| `forward` | method | `kimi.py:175` | `def forward(self, x)` |
| `get_loader` | method | `kimi.py:191` | `def get_loader()` |
| `pgd_attack` | method | `kimi.py:39` | `def pgd_attack(model, x, y, eps, steps, alpha)` |
| `run_experiment` | method | `kimi.py:201` | `def run_experiment(seed, sne_enabled, ablated_organs)` |
| `scientific_ablation` | method | `kimi.py:256` | `def scientific_ablation()` |
| `ConsciousnessModule` | class | `legendario.py:224` | `class ConsciousnessModule(OmniBrainModule)` |
| `DualMindModule` | class | `legendario.py:193` | `class DualMindModule(OmniBrainModule)` |
| `MotorHomeostaticContext` | class | `legendario.py:106` | `class MotorHomeostaticContext` |
| `OmniBrain` | class | `legendario.py:289` | `class OmniBrain(Module)` |
| `OmniBrainCoordinator` | class | `legendario.py:245` | `class OmniBrainCoordinator` |
| `OmniBrainModule` | class | `legendario.py:117` | `class OmniBrainModule(Module)` |
| `PTSymmetricLayer` | class | `legendario.py:132` | `class PTSymmetricLayer(OmniBrainModule)` |
| `TopologicalLayer` | class | `legendario.py:166` | `class TopologicalLayer(OmniBrainModule)` |
| `__init__` | method | `legendario.py:119` | `def __init__(self, module_name, enabled)` |
| `__init__` | method | `legendario.py:135` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `legendario.py:169` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `legendario.py:196` | `def __init__(self, features)` |
| `__init__` | method | `legendario.py:227` | `def __init__(self, features)` |
| `__init__` | method | `legendario.py:248` | `def __init__(self)` |
| `__init__` | method | `legendario.py:292` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `compute_phi_effective_approx` | function | `legendario.py:27` | `def compute_phi_effective_approx(activity)` |
| `compute_pt_phase` | method | `legendario.py:144` | `def compute_pt_phase(self)` |
| `compute_topological_metrics` | function | `legendario.py:60` | `def compute_topological_metrics(weights)` |
| `estimate_energy_consumption` | function | `legendario.py:89` | `def estimate_energy_consumption(model, input_size)` |
| `forward` | method | `legendario.py:152` | `def forward(self, x, params)` |
| `forward` | method | `legendario.py:183` | `def forward(self, x, params)` |
| `forward` | method | `legendario.py:211` | `def forward(self, x, params)` |
| `forward` | method | `legendario.py:232` | `def forward(self, x, params)` |
| `forward` | method | `legendario.py:317` | `def forward(self, x)` |
| `measure_network_state` | method | `legendario.py:251` | `def measure_network_state(self, model, batch_data)` |
| `train_omni_brain` | method | `legendario.py:340` | `def train_omni_brain(model, epochs, batch_size, device)` |
| `update_performance` | method | `legendario.py:125` | `def update_performance(self, metrics)` |
| `update_topology` | method | `legendario.py:176` | `def update_topology(self, connectivity)` |
| `AdaptiveLearningMotor` | class | `legendario2.py:213` | `class AdaptiveLearningMotor(MotorHomeostaticContext)` |
| `ConsciousnessModule` | class | `legendario2.py:640` | `class ConsciousnessModule(OmniBrainModule)` |
| `ConsciousnessMotor` | class | `legendario2.py:157` | `class ConsciousnessMotor(MotorHomeostaticContext)` |
| `DualMindModule` | class | `legendario2.py:557` | `class DualMindModule(OmniBrainModule)` |
| `DualSystemMotor` | class | `legendario2.py:185` | `class DualSystemMotor(MotorHomeostaticContext)` |
| `EnergyHomeostaticMotor` | class | `legendario2.py:127` | `class EnergyHomeostaticMotor(MotorHomeostaticContext)` |
| `HomeostaticEngine` | class | `legendario2.py:701` | `class HomeostaticEngine` |
| `ModularActivationMotor` | class | `legendario2.py:241` | `class ModularActivationMotor(MotorHomeostaticContext)` |
| `MotorHomeostaticContext` | class | `legendario2.py:34` | `class MotorHomeostaticContext` |
| `OmniBrain` | class | `legendario2.py:732` | `class OmniBrain(Module)` |
| `OmniBrainCoordinator` | class | `legendario2.py:291` | `class OmniBrainCoordinator` |
| `OmniBrainModule` | class | `legendario2.py:452` | `class OmniBrainModule(Module)` |
| `PTSymmetricLayer` | class | `legendario2.py:467` | `class PTSymmetricLayer(OmniBrainModule)` |
| `PTSymmetricMotor` | class | `legendario2.py:64` | `class PTSymmetricMotor(MotorHomeostaticContext)` |
| `TopologicalLayer` | class | `legendario2.py:500` | `class TopologicalLayer(OmniBrainModule)` |
| `TopologicalMotor` | class | `legendario2.py:100` | `class TopologicalMotor(MotorHomeostaticContext)` |
| `__init__` | method | `legendario2.py:66` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:102` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:129` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:159` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:187` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:215` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:243` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:294` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:455` | `def __init__(self, module_name, enabled)` |
| `__init__` | method | `legendario2.py:470` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `legendario2.py:503` | `def __init__(self, in_features, out_features, sparsity_factor)` |
| `__init__` | method | `legendario2.py:560` | `def __init__(self, features)` |
| `__init__` | method | `legendario2.py:643` | `def __init__(self, features)` |
| `__init__` | method | `legendario2.py:704` | `def __init__(self, target_performance)` |
| `__init__` | method | `legendario2.py:735` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `_generate_topology_mask` | method | `legendario2.py:518` | `def _generate_topology_mask(self)` |
| `_initialize_motors` | method | `legendario2.py:300` | `def _initialize_motors(self)` |
| `compute_phi_effective` | method | `legendario2.py:657` | `def compute_phi_effective(self, x)` |
| `coordinate_all_motors` | method | `legendario2.py:377` | `def coordinate_all_motors(self, environment_state, network_state)` |
| `forward` | method | `legendario2.py:461` | `def forward(self, x, params)` |
| `forward` | method | `legendario2.py:477` | `def forward(self, x, params)` |
| `forward` | method | `legendario2.py:539` | `def forward(self, x, params)` |
| `forward` | method | `legendario2.py:585` | `def forward(self, x, params)` |
| `forward` | method | `legendario2.py:677` | `def forward(self, x, params)` |
| `forward` | method | `legendario2.py:834` | `def forward(self, x)` |
| `get_status_report` | method | `legendario2.py:932` | `def get_status_report(self)` |
| `initialize_context` | method | `legendario2.py:820` | `def initialize_context(self)` |
| `measure_network_state` | method | `legendario2.py:340` | `def measure_network_state(self, model, batch_data)` |
| `prepare_for_inference` | method | `legendario2.py:803` | `def prepare_for_inference(self)` |
| `regulate_connectivity` | method | `legendario2.py:112` | `def regulate_connectivity(self, current_connectivity, clustering)` |
| `regulate_consciousness` | method | `legendario2.py:168` | `def regulate_consciousness(self, phi_effective, integration_level)` |
| `regulate_dual_systems` | method | `legendario2.py:197` | `def regulate_dual_systems(self, unconscious_activity, conscious_activity)` |
| `regulate_energy` | method | `legendario2.py:139` | `def regulate_energy(self, memory_usage, cpu_usage, temperature)` |
| `regulate_homeostasis` | method | `legendario2.py:709` | `def regulate_homeostasis(self, observed_performance)` |
| `regulate_learning` | method | `legendario2.py:224` | `def regulate_learning(self, loss_reduction_rate, gradient_norm)` |
| `regulate_modules` | method | `legendario2.py:259` | `def regulate_modules(self, task_complexity, resource_availability, performance)` |
| `regulate_parameters` | method | `legendario2.py:78` | `def regulate_parameters(self, current_coherence, energy_level)` |
| `reset_internal_states` | method | `legendario2.py:769` | `def reset_internal_states(self)` |
| `sense_environment` | method | `legendario2.py:312` | `def sense_environment(self)` |
| `simulate_network_state` | method | `legendario2.py:326` | `def simulate_network_state(self)` |
| `train_omni_brain` | method | `legendario2.py:967` | `def train_omni_brain(model, epochs, batch_size)` |
| `update` | method | `legendario2.py:47` | `def update(self, measurement, dt)` |
| `update_performance` | method | `legendario2.py:464` | `def update_performance(self, metrics)` |
| `Config` | class | `live_cl.py:32` | `class Config` |
| `DualSystemModule` | class | `live_cl.py:206` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `live_cl.py:133` | `class FastSlowLinear(Module)` |
| `IntegrationModule` | class | `live_cl.py:242` | `class IntegrationModule(Module)` |
| `OmniBrain` | class | `live_cl.py:278` | `class OmniBrain(Module)` |
| `__init__` | method | `live_cl.py:138` | `def __init__(self, in_features, out_features, config)` |
| `__init__` | method | `live_cl.py:211` | `def __init__(self, dim, config)` |
| `__init__` | method | `live_cl.py:247` | `def __init__(self, features, config)` |
| `__init__` | method | `live_cl.py:283` | `def __init__(self, config)` |
| `compute_integration_index` | method | `live_cl.py:99` | `def compute_integration_index(activity)` |
| `evaluate` | method | `live_cl.py:393` | `def evaluate(model, loader, device)` |
| `forward` | method | `live_cl.py:189` | `def forward(self, x)` |
| `forward` | method | `live_cl.py:223` | `def forward(self, x)` |
| `forward` | method | `live_cl.py:260` | `def forward(self, x)` |
| `forward` | method | `live_cl.py:318` | `def forward(self, x)` |
| `get_ablation_state` | method | `live_cl.py:336` | `def get_ablation_state(self)` |
| `get_data_loaders` | method | `live_cl.py:349` | `def get_data_loaders(config)` |
| `get_fast_norm` | method | `live_cl.py:202` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `live_cl.py:331` | `def get_fast_norms(self)` |
| `reset_all_fast_weights` | method | `live_cl.py:325` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `live_cl.py:156` | `def reset_fast_weights(self)` |
| `run_ablation_study` | method | `live_cl.py:570` | `def run_ablation_study(quick_test)` |
| `set_seed` | method | `live_cl.py:84` | `def set_seed(seed)` |
| `setup_logging` | method | `live_cl.py:69` | `def setup_logging()` |
| `to_dict` | method | `live_cl.py:62` | `def to_dict(self)` |
| `train` | method | `live_cl.py:430` | `def train(config, silent)` |
| `update_fast_weights` | method | `live_cl.py:162` | `def update_fast_weights(self, x, slow_out)` |
| `Config` | class | `live_go.py:30` | `class Config` |
| `DualSystemModule` | class | `live_go.py:143` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `live_go.py:93` | `class FastSlowLinear(Module)` |
| `IntegrationModule` | class | `live_go.py:167` | `class IntegrationModule(Module)` |
| `OmniBrainGenesis` | class | `live_go.py:190` | `class OmniBrainGenesis(Module)` |
| `__init__` | method | `live_go.py:98` | `def __init__(self, in_features, out_features, config)` |
| `__init__` | method | `live_go.py:144` | `def __init__(self, dim, config)` |
| `__init__` | method | `live_go.py:168` | `def __init__(self, features, config)` |
| `__init__` | method | `live_go.py:191` | `def __init__(self, config)` |
| `breathe_life` | method | `live_go.py:254` | `def breathe_life(config)` |
| `compute_integration_index` | method | `live_go.py:74` | `def compute_integration_index(activity)` |
| `forward` | method | `live_go.py:120` | `def forward(self, x)` |
| `forward` | method | `live_go.py:155` | `def forward(self, x)` |
| `forward` | method | `live_go.py:176` | `def forward(self, x)` |
| `forward` | method | `live_go.py:215` | `def forward(self, x)` |
| `get_fast_norm` | method | `live_go.py:140` | `def get_fast_norm(self)` |
| `get_loaders` | method | `live_go.py:230` | `def get_loaders(config)` |
| `reset_all_fast_weights` | method | `live_go.py:222` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `live_go.py:115` | `def reset_fast_weights(self)` |
| `reset_seeds` | method | `live_go.py:353` | `def reset_seeds()` |
| `run_ablation_test` | method | `live_go.py:360` | `def run_ablation_test(full_epochs)` |
| `train_engine_wrapper` | method | `live_go.py:425` | `def train_engine_wrapper(config)` |
| `Config` | class | `live_ki.py:26` | `class Config` |
| `DualSystemModule` | class | `live_ki.py:177` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `live_ki.py:103` | `class FastSlowLinear(Module)` |
| `IntegrationModule` | class | `live_ki.py:211` | `class IntegrationModule(Module)` |
| `OmniBrainGenesis` | class | `live_ki.py:243` | `class OmniBrainGenesis(Module)` |
| `__init__` | method | `live_ki.py:104` | `def __init__(self, in_features, out_features, config)` |
| `__init__` | method | `live_ki.py:178` | `def __init__(self, dim, config)` |
| `__init__` | method | `live_ki.py:212` | `def __init__(self, features, config)` |
| `__init__` | method | `live_ki.py:244` | `def __init__(self, config)` |
| `compute_integration_index` | method | `live_ki.py:76` | `def compute_integration_index(activity)` |
| `evaluate_ritual` | method | `live_ki.py:334` | `def evaluate_ritual(model, loader, device)` |
| `explore_realities` | method | `live_ki.py:499` | `def explore_realities()` |
| `forward` | method | `live_ki.py:157` | `def forward(self, x)` |
| `forward` | method | `live_ki.py:192` | `def forward(self, x)` |
| `forward` | method | `live_ki.py:224` | `def forward(self, x)` |
| `forward` | method | `live_ki.py:279` | `def forward(self, x)` |
| `get_ablation_state` | method | `live_ki.py:296` | `def get_ablation_state(self)` |
| `get_cifar10_loaders` | method | `live_ki.py:308` | `def get_cifar10_loaders(config)` |
| `get_fast_norm` | method | `live_ki.py:171` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `live_ki.py:292` | `def get_fast_norms(self)` |
| `reset_all_fast_weights` | method | `live_ki.py:286` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `live_ki.py:124` | `def reset_fast_weights(self)` |
| `train_genesis` | method | `live_ki.py:366` | `def train_genesis(config)` |
| `update_fast_weights` | method | `live_ki.py:130` | `def update_fast_weights(self, x, slow_out)` |
| `Config` | class | `live_qw.py:23` | `class Config` |
| `DualSystemModule` | class | `live_qw.py:122` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `live_qw.py:64` | `class FastSlowLinear(Module)` |
| `IntegrationModule` | class | `live_qw.py:146` | `class IntegrationModule(Module)` |
| `OmniBrainFastSlow` | class | `live_qw.py:191` | `class OmniBrainFastSlow(Module)` |
| `__init__` | method | `live_qw.py:66` | `def __init__(self, in_features, out_features, config)` |
| `__init__` | method | `live_qw.py:123` | `def __init__(self, dim, config)` |
| `__init__` | method | `live_qw.py:147` | `def __init__(self, features, config)` |
| `__init__` | method | `live_qw.py:192` | `def __init__(self, config)` |
| `compute_integration_index` | method | `live_qw.py:171` | `def compute_integration_index(activity)` |
| `evaluate_full` | method | `live_qw.py:265` | `def evaluate_full(model, loader, device)` |
| `forward` | method | `live_qw.py:106` | `def forward(self, x)` |
| `forward` | method | `live_qw.py:133` | `def forward(self, x)` |
| `forward` | method | `live_qw.py:158` | `def forward(self, x)` |
| `forward` | method | `live_qw.py:217` | `def forward(self, x)` |
| `get_ablation_state` | method | `live_qw.py:232` | `def get_ablation_state(self)` |
| `get_cifar10_loaders` | method | `live_qw.py:244` | `def get_cifar10_loaders(config)` |
| `get_fast_norm` | method | `live_qw.py:118` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `live_qw.py:229` | `def get_fast_norms(self)` |
| `reset_all_fast_weights` | method | `live_qw.py:224` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `live_qw.py:83` | `def reset_fast_weights(self)` |
| `train` | method | `live_qw.py:286` | `def train(config)` |
| `update_fast_weights` | method | `live_qw.py:88` | `def update_fast_weights(self, x, slow_out)` |
| `CorpusCallosum` | class | `lol.py:298` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `lol.py:404` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphere` | class | `lol.py:201` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `lol.py:466` | `class LifeCycle` |
| `LiquidNeuron` | class | `lol.py:107` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `lol.py:334` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `lol.py:313` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `lol.py:180` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `lol.py:426` | `def __getitem__(self, idx)` |
| `__init__` | method | `lol.py:108` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `lol.py:181` | `def __init__(self, output_dim)` |
| `__init__` | method | `lol.py:202` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `lol.py:299` | `def __init__(self, dim)` |
| `__init__` | method | `lol.py:314` | `def __init__(self, vocab_size)` |
| `__init__` | method | `lol.py:335` | `def __init__(self)` |
| `__init__` | method | `lol.py:405` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `lol.py:467` | `def __init__(self, total_epochs)` |
| `__len__` | method | `lol.py:423` | `def __len__(self)` |
| `_get_init_state` | method | `lol.py:278` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `lol.py:283` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `lol.py:443` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `consolidate_svd` | method | `lol.py:154` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `lol.py:125` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `lol.py:192` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `lol.py:224` | `def forward(self, visual_context, captions, max_len, return_gate)` |
| `forward` | method | `lol.py:307` | `def forward(self, right_features)` |
| `forward` | method | `lol.py:320` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)` |
| `get_plasticity` | method | `lol.py:470` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `lol.py:362` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `lol.py:346` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `lol.py:353` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `lol.py:367` | `def report(self, epoch)` |
| `setup_flickr8k` | function | `lol.py:34` | `def setup_flickr8k(data_dir)` |
| `train_bicameral` | method | `lol.py:481` | `def train_bicameral()` |
| `update` | method | `lol.py:357` | `def update(self)` |
| `BranchingOperator` | class | `main.py:220` | `class BranchingOperator` |
| `EmunaOperator` | class | `main.py:270` | `class EmunaOperator` |
| `ExperimentalPredictions` | class | `main.py:686` | `class ExperimentalPredictions` |
| `FreedomInvariant` | class | `main.py:582` | `class FreedomInvariant` |
| `LindbladFractalDynamics` | class | `main.py:350` | `class LindbladFractalDynamics` |
| `MyelinCavity` | class | `main.py:428` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `main.py:481` | `class NeuralNetworkRESMA` |
| `NullModels` | class | `main.py:628` | `class NullModels` |
| `PhysicalValidator` | class | `main.py:58` | `class PhysicalValidator` |
| `QuantumLeaf` | class | `main.py:90` | `class QuantumLeaf` |
| `RESMAConstants` | class | `main.py:32` | `class RESMAConstants` |
| `RESMAUniverse` | class | `main.py:144` | `class RESMAUniverse` |
| `__init__` | method | `main.py:150` | `def __init__(self, n_leaves, seed)` |
| `__init__` | method | `main.py:226` | `def __init__(self, leaf, threshold)` |
| `__init__` | method | `main.py:276` | `def __init__(self, universe, n_samples)` |
| `__init__` | method | `main.py:356` | `def __init__(self, universe, emuna)` |
| `__init__` | method | `main.py:487` | `def __init__(self, n_nodes, seed)` |
| `__init__` | method | `main.py:588` | `def __init__(self, network, universe)` |
| `__init__` | method | `main.py:692` | `def __init__(self, resma, myelin, network)` |
| `__post_init__` | method | `main.py:100` | `def __post_init__(self)` |
| `__post_init__` | method | `main.py:437` | `def __post_init__(self)` |
| `_construct_cptp_map` | method | `main.py:231` | `def _construct_cptp_map(self)` |
| `_construct_global_state` | method | `main.py:203` | `def _construct_global_state(self)` |
| `_construct_hardy_state` | method | `main.py:282` | `def _construct_hardy_state(self)` |
| `_effective_hamiltonian` | method | `main.py:362` | `def _effective_hamiltonian(self)` |
| `_evaluation_functional` | method | `main.py:294` | `def _evaluation_functional(self, state_weights)` |
| `_free_hamiltonian` | method | `main.py:442` | `def _free_hamiltonian(self)` |
| `_generate_fractal_graph` | method | `main.py:500` | `def _generate_fractal_graph(self)` |
| `_generate_gibbs_measure` | method | `main.py:181` | `def _generate_gibbs_measure(self)` |
| `_graph_to_distance_matrix` | method | `main.py:548` | `def _graph_to_distance_matrix(self)` |
| `_initialize_leaves` | method | `main.py:168` | `def _initialize_leaves(self)` |
| `_local_jump_operator` | method | `main.py:240` | `def _local_jump_operator(self, power)` |
| `_loss_potential` | method | `main.py:448` | `def _loss_potential(self)` |
| `_modular_dissipator` | method | `main.py:372` | `def _modular_dissipator(self, state)` |
| `_nonlinear_term` | method | `main.py:382` | `def _nonlinear_term(self, state)` |
| `_predict_diffraction_peak` | method | `main.py:710` | `def _predict_diffraction_peak(self)` |
| `_pt_symmetry_condition` | method | `main.py:456` | `def _pt_symmetry_condition(self)` |
| `_spectral_dimension` | method | `main.py:510` | `def _spectral_dimension(self)` |
| `_spectral_moments` | method | `main.py:132` | `def _spectral_moments(self, n)` |
| `_szego_projector` | method | `main.py:286` | `def _szego_projector(self)` |
| `_topological_ramsey` | method | `main.py:527` | `def _topological_ramsey(self)` |
| `apply_branching` | method | `main.py:255` | `def apply_branching(self, state_vector)` |
| `bures_distance` | method | `main.py:121` | `def bures_distance(self, other)` |
| `coherence_quantum` | method | `main.py:462` | `def coherence_quantum(self)` |
| `compute_bayes_factor` | method | `main.py:715` | `def compute_bayes_factor(self)` |
| `compute_entropy_gap` | method | `main.py:592` | `def compute_entropy_gap(self)` |
| `compute_freedom` | method | `main.py:607` | `def compute_freedom(self)` |
| `compute_pontryagin_number` | method | `main.py:596` | `def compute_pontryagin_number(self)` |
| `critical_percolation_time` | method | `main.py:562` | `def critical_percolation_time(self)` |
| `evolve` | method | `main.py:389` | `def evolve(self, rho0, t_span, n_steps)` |
| `is_coherent_subgraph` | method | `main.py:574` | `def is_coherent_subgraph(self, subgraph_nodes)` |
| `is_gauge_invariant` | method | `main.py:618` | `def is_gauge_invariant(self)` |
| `ising_quantum` | method | `main.py:635` | `def ising_quantum(network)` |
| `modular_entropy` | method | `main.py:114` | `def modular_entropy(self)` |
| `predict_all` | method | `main.py:699` | `def predict_all(self)` |
| `project` | method | `main.py:310` | `def project(self, state_vector)` |
| `random_network` | method | `main.py:670` | `def random_network(network)` |
| `simulate_resma_multiverse` | method | `main.py:764` | `def simulate_resma_multiverse(n_leaves, n_nodes, seed)` |
| `spectral_density` | method | `main.py:106` | `def spectral_density(self, omega)` |
| `syk4` | method | `main.py:654` | `def syk4(network)` |
| `validate_connectome_size` | method | `main.py:79` | `def validate_connectome_size(n_nodes)` |
| `validate_dimension` | method | `main.py:62` | `def validate_dimension(alpha)` |
| `validate_pt_symmetry` | method | `main.py:68` | `def validate_pt_symmetry(kappa, Omega, chi)` |
| `BranchingOperator` | class | `main2.py:219` | `class BranchingOperator` |
| `EmunaOperator` | class | `main2.py:269` | `class EmunaOperator` |
| `ExperimentalPredictions` | class | `main2.py:685` | `class ExperimentalPredictions` |
| `FreedomInvariant` | class | `main2.py:581` | `class FreedomInvariant` |
| `LindbladFractalDynamics` | class | `main2.py:349` | `class LindbladFractalDynamics` |
| `MyelinCavity` | class | `main2.py:427` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `main2.py:480` | `class NeuralNetworkRESMA` |
| `NullModels` | class | `main2.py:627` | `class NullModels` |
| `PhysicalValidator` | class | `main2.py:59` | `class PhysicalValidator` |
| `QuantumLeaf` | class | `main2.py:91` | `class QuantumLeaf` |
| `RESMAConstants` | class | `main2.py:33` | `class RESMAConstants` |
| `RESMAUniverse` | class | `main2.py:144` | `class RESMAUniverse` |
| `__init__` | method | `main2.py:150` | `def __init__(self, n_leaves, seed)` |
| `__init__` | method | `main2.py:225` | `def __init__(self, leaf, threshold)` |
| `__init__` | method | `main2.py:275` | `def __init__(self, universe, n_samples)` |
| `__init__` | method | `main2.py:355` | `def __init__(self, universe, emuna)` |
| `__init__` | method | `main2.py:486` | `def __init__(self, n_nodes, seed)` |
| `__init__` | method | `main2.py:587` | `def __init__(self, network, universe)` |
| `__init__` | method | `main2.py:691` | `def __init__(self, resma, myelin, network)` |
| `__post_init__` | method | `main2.py:101` | `def __post_init__(self)` |
| `__post_init__` | method | `main2.py:436` | `def __post_init__(self)` |
| `_construct_cptp_map` | method | `main2.py:230` | `def _construct_cptp_map(self)` |
| `_construct_global_state` | method | `main2.py:202` | `def _construct_global_state(self)` |
| `_construct_hardy_state` | method | `main2.py:281` | `def _construct_hardy_state(self)` |
| `_effective_hamiltonian` | method | `main2.py:361` | `def _effective_hamiltonian(self)` |
| `_evaluation_functional` | method | `main2.py:293` | `def _evaluation_functional(self, state_weights)` |
| `_free_hamiltonian` | method | `main2.py:441` | `def _free_hamiltonian(self)` |
| `_generate_fractal_graph` | method | `main2.py:499` | `def _generate_fractal_graph(self)` |
| `_generate_gibbs_measure` | method | `main2.py:180` | `def _generate_gibbs_measure(self)` |
| `_graph_to_distance_matrix` | method | `main2.py:547` | `def _graph_to_distance_matrix(self)` |
| `_initialize_leaves` | method | `main2.py:167` | `def _initialize_leaves(self)` |
| `_local_jump_operator` | method | `main2.py:239` | `def _local_jump_operator(self, power)` |
| `_loss_potential` | method | `main2.py:447` | `def _loss_potential(self)` |
| `_modular_dissipator` | method | `main2.py:371` | `def _modular_dissipator(self, state)` |
| `_nonlinear_term` | method | `main2.py:381` | `def _nonlinear_term(self, state)` |
| `_predict_diffraction_peak` | method | `main2.py:708` | `def _predict_diffraction_peak(self)` |
| `_pt_symmetry_condition` | method | `main2.py:455` | `def _pt_symmetry_condition(self)` |
| `_spectral_dimension` | method | `main2.py:509` | `def _spectral_dimension(self)` |
| `_spectral_moments` | method | `main2.py:132` | `def _spectral_moments(self, n)` |
| `_szego_projector` | method | `main2.py:285` | `def _szego_projector(self)` |
| `_topological_ramsey` | method | `main2.py:526` | `def _topological_ramsey(self)` |
| `apply_branching` | method | `main2.py:254` | `def apply_branching(self, state_vector)` |
| `bures_distance` | method | `main2.py:121` | `def bures_distance(self, other)` |
| `coherence_quantum` | method | `main2.py:461` | `def coherence_quantum(self)` |
| `compute_bayes_factor` | method | `main2.py:713` | `def compute_bayes_factor(self)` |
| `compute_entropy_gap` | method | `main2.py:591` | `def compute_entropy_gap(self)` |
| `compute_freedom` | method | `main2.py:606` | `def compute_freedom(self)` |
| `compute_pontryagin_number` | method | `main2.py:595` | `def compute_pontryagin_number(self)` |
| `critical_percolation_time` | method | `main2.py:561` | `def critical_percolation_time(self)` |
| `evolve` | method | `main2.py:388` | `def evolve(self, rho0, t_span, n_steps)` |
| `is_coherent_subgraph` | method | `main2.py:573` | `def is_coherent_subgraph(self, subgraph_nodes)` |
| `is_gauge_invariant` | method | `main2.py:617` | `def is_gauge_invariant(self)` |
| `ising_quantum` | method | `main2.py:634` | `def ising_quantum(network)` |
| `modular_entropy` | method | `main2.py:114` | `def modular_entropy(self)` |
| `predict_all` | method | `main2.py:698` | `def predict_all(self)` |
| `project` | method | `main2.py:309` | `def project(self, state_vector)` |
| `random_network` | method | `main2.py:669` | `def random_network(network)` |
| `simulate_resma_multiverse` | method | `main2.py:763` | `def simulate_resma_multiverse(n_leaves, n_nodes, seed)` |
| `spectral_density` | method | `main2.py:106` | `def spectral_density(self, omega)` |
| `syk4` | method | `main2.py:653` | `def syk4(network)` |
| `validate_connectome_size` | method | `main2.py:80` | `def validate_connectome_size(n_nodes)` |
| `validate_dimension` | method | `main2.py:63` | `def validate_dimension(alpha)` |
| `validate_pt_symmetry` | method | `main2.py:69` | `def validate_pt_symmetry(kappa, Omega, chi)` |
| `Bayes` | class | `main3.py:205` | `class Bayes` |
| `MyelinCavity` | class | `main3.py:171` | `class MyelinCavity` |
| `Network` | class | `main3.py:129` | `class Network` |
| `QuantumLeaf` | class | `main3.py:68` | `class QuantumLeaf` |
| `RC` | class | `main3.py:33` | `class RC` |
| `Universe` | class | `main3.py:101` | `class Universe` |
| `Validator` | class | `main3.py:50` | `class Validator` |
| `__init__` | method | `main3.py:102` | `def __init__(self, n_leaves, seed)` |
| `__init__` | method | `main3.py:130` | `def __init__(self, n_nodes, seed)` |
| `__init__` | method | `main3.py:172` | `def __init__(self, n_modes)` |
| `__init__` | method | `main3.py:206` | `def __init__(self, pred_resma, nulls)` |
| `__post_init__` | method | `main3.py:74` | `def __post_init__(self)` |
| `_free_hamiltonian` | method | `main3.py:178` | `def _free_hamiltonian(self)` |
| `_gibbs` | method | `main3.py:110` | `def _gibbs(self)` |
| `_global` | method | `main3.py:119` | `def _global(self)` |
| `_loss_potential` | method | `main3.py:183` | `def _loss_potential(self)` |
| `_pt_symmetry_condition` | method | `main3.py:189` | `def _pt_symmetry_condition(self)` |
| `_ramsey` | method | `main3.py:149` | `def _ramsey(self)` |
| `_spectral_dim` | method | `main3.py:139` | `def _spectral_dim(self, k)` |
| `bf` | method | `main3.py:217` | `def bf(self)` |
| `bures_distance` | method | `main3.py:87` | `def bures_distance(self, other)` |
| `coherence_quantum` | method | `main3.py:192` | `def coherence_quantum(self)` |
| `dim` | method | `main3.py:52` | `def dim(a)` |
| `log_lik` | method | `main3.py:210` | `def log_lik(self, model_pred)` |
| `modular_entropy` | method | `main3.py:81` | `def modular_entropy(self)` |
| `pt` | method | `main3.py:56` | `def pt(k, o, c)` |
| `simulate` | method | `main3.py:233` | `def simulate(n_leaves, n_nodes, seed)` |
| `size` | method | `main3.py:59` | `def size(n)` |
| `spectral_density` | method | `main3.py:78` | `def spectral_density(self, w)` |
| `t_c` | method | `main3.py:163` | `def t_c(self)` |
| `Bayes` | class | `main4.1.py:268` | `class Bayes` |
| `MyelinCavity` | class | `main4.1.py:229` | `class MyelinCavity` |
| `Network` | class | `main4.1.py:152` | `class Network` |
| `QuantumLeaf` | class | `main4.1.py:82` | `class QuantumLeaf` |
| `RC` | class | `main4.1.py:29` | `class RC` |
| `Universe` | class | `main4.1.py:123` | `class Universe` |
| `Validator` | class | `main4.1.py:61` | `class Validator` |
| `__init__` | method | `main4.1.py:124` | `def __init__(self, n_leaves, seed)` |
| `__init__` | method | `main4.1.py:153` | `def __init__(self, n_nodes, seed)` |
| `__init__` | method | `main4.1.py:230` | `def __init__(self, n_modes)` |
| `__init__` | method | `main4.1.py:269` | `def __init__(self, pred_resma, nulls)` |
| `__post_init__` | method | `main4.1.py:88` | `def __post_init__(self)` |
| `_free_hamiltonian` | method | `main4.1.py:237` | `def _free_hamiltonian(self)` |
| `_gibbs` | method | `main4.1.py:133` | `def _gibbs(self)` |
| `_global` | method | `main4.1.py:142` | `def _global(self)` |
| `_loss_potential` | method | `main4.1.py:242` | `def _loss_potential(self)` |
| `_pt_symmetry_condition` | method | `main4.1.py:248` | `def _pt_symmetry_condition(self)` |
| `_ramsey` | method | `main4.1.py:199` | `def _ramsey(self)` |
| `_spectral_dim` | method | `main4.1.py:163` | `def _spectral_dim(self, k, n_fit)` |
| `bures_distance` | method | `main4.1.py:104` | `def bures_distance(self, other)` |
| `coherence_quantum` | method | `main4.1.py:251` | `def coherence_quantum(self)` |
| `dim` | method | `main4.1.py:63` | `def dim(a)` |
| `ln_bf` | method | `main4.1.py:288` | `def ln_bf(self)` |
| `log_lik` | method | `main4.1.py:273` | `def log_lik(self, model_pred)` |
| `modular_entropy` | method | `main4.1.py:95` | `def modular_entropy(self)` |
| `pt` | method | `main4.1.py:68` | `def pt(k, o, c)` |
| `simulate` | method | `main4.1.py:307` | `def simulate(n_leaves, n_nodes, seed)` |
| `size` | method | `main4.1.py:73` | `def size(n)` |
| `spectral_density` | method | `main4.1.py:92` | `def spectral_density(self, w)` |
| `t_c` | method | `main4.1.py:218` | `def t_c(self)` |
| `verify_pt_condition` | method | `main4.1.py:50` | `def verify_pt_condition(cls)` |
| `Bayes` | class | `main4.py.py:232` | `class Bayes` |
| `MyelinCavity` | class | `main4.py.py:198` | `class MyelinCavity` |
| `Network` | class | `main4.py.py:130` | `class Network` |
| `QuantumLeaf` | class | `main4.py.py:69` | `class QuantumLeaf` |
| `RC` | class | `main4.py.py:33` | `class RC` |
| `Universe` | class | `main4.py.py:102` | `class Universe` |
| `Validator` | class | `main4.py.py:50` | `class Validator` |
| `__init__` | method | `main4.py.py:103` | `def __init__(self, n_leaves, seed)` |
| `__init__` | method | `main4.py.py:131` | `def __init__(self, n_nodes, seed)` |
| `__init__` | method | `main4.py.py:199` | `def __init__(self, n_modes)` |
| `__init__` | method | `main4.py.py:233` | `def __init__(self, pred_resma, nulls)` |
| `__post_init__` | method | `main4.py.py:75` | `def __post_init__(self)` |
| `_free_hamiltonian` | method | `main4.py.py:205` | `def _free_hamiltonian(self)` |
| `_gibbs` | method | `main4.py.py:111` | `def _gibbs(self)` |
| `_global` | method | `main4.py.py:120` | `def _global(self)` |
| `_loss_potential` | method | `main4.py.py:210` | `def _loss_potential(self)` |
| `_pt_symmetry_condition` | method | `main4.py.py:216` | `def _pt_symmetry_condition(self)` |
| `_ramsey` | method | `main4.py.py:176` | `def _ramsey(self)` |
| `_spectral_dim` | method | `main4.py.py:140` | `def _spectral_dim(self, k, n_fit)` |
| `bures_distance` | method | `main4.py.py:88` | `def bures_distance(self, other)` |
| `coherence_quantum` | method | `main4.py.py:219` | `def coherence_quantum(self)` |
| `dim` | method | `main4.py.py:52` | `def dim(a)` |
| `ln_bf` | method | `main4.py.py:244` | `def ln_bf(self)` |
| `log_lik` | method | `main4.py.py:237` | `def log_lik(self, model_pred)` |
| `modular_entropy` | method | `main4.py.py:82` | `def modular_entropy(self)` |
| `pt` | method | `main4.py.py:56` | `def pt(k, o, c)` |
| `simulate` | method | `main4.py.py:260` | `def simulate(n_leaves, n_nodes, seed)` |
| `size` | method | `main4.py.py:60` | `def size(n)` |
| `spectral_density` | method | `main4.py.py:79` | `def spectral_density(self, w)` |
| `t_c` | method | `main4.py.py:190` | `def t_c(self)` |
| `BranchingOperator` | class | `main5.py:260` | `class BranchingOperator` |
| `EmpiricalValidationProtocol` | class | `main5.py:975` | `class EmpiricalValidationProtocol` |
| `EmunaOperator` | class | `main5.py:318` | `class EmunaOperator` |
| `ExperimentalPredictions` | class | `main5.py:877` | `class ExperimentalPredictions` |
| `FreedomInvariant` | class | `main5.py:762` | `class FreedomInvariant` |
| `LindbladFractalDynamics` | class | `main5.py:406` | `class LindbladFractalDynamics` |
| `MyelinCavity` | class | `main5.py:539` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `main5.py:604` | `class NeuralNetworkRESMA` |
| `NullModels` | class | `main5.py:816` | `class NullModels` |
| `PhysicalValidator` | class | `main5.py:70` | `class PhysicalValidator` |
| `QuantumLeaf` | class | `main5.py:116` | `class QuantumLeaf` |
| `RESMAConstants` | class | `main5.py:37` | `class RESMAConstants` |
| `RESMAUniverse` | class | `main5.py:179` | `class RESMAUniverse` |
| `__init__` | method | `main5.py:185` | `def __init__(self, n_leaves, seed)` |
| `__init__` | method | `main5.py:266` | `def __init__(self, leaf, threshold)` |
| `__init__` | method | `main5.py:324` | `def __init__(self, universe, n_samples)` |
| `__init__` | method | `main5.py:412` | `def __init__(self, universe, emuna)` |
| `__init__` | method | `main5.py:610` | `def __init__(self, n_nodes, seed)` |
| `__init__` | method | `main5.py:768` | `def __init__(self, network, universe)` |
| `__init__` | method | `main5.py:883` | `def __init__(self, resma, myelin, network, freedom)` |
| `__init__` | method | `main5.py:981` | `def __init__(self, predictions)` |
| `__post_init__` | method | `main5.py:127` | `def __post_init__(self)` |
| `__post_init__` | method | `main5.py:548` | `def __post_init__(self)` |
| `_compute_betti_numbers` | method | `main5.py:706` | `def _compute_betti_numbers(self)` |
| `_compute_holonomy` | method | `main5.py:272` | `def _compute_holonomy(self)` |
| `_compute_scalar_mass` | method | `main5.py:569` | `def _compute_scalar_mass(self)` |
| `_construct_cptp_map` | method | `main5.py:276` | `def _construct_cptp_map(self)` |
| `_construct_global_state` | method | `main5.py:239` | `def _construct_global_state(self)` |
| `_construct_hardy_state` | method | `main5.py:331` | `def _construct_hardy_state(self)` |
| `_correct_non_physical_state` | method | `main5.py:523` | `def _correct_non_physical_state(self, state)` |
| `_define_protocols` | method | `main5.py:985` | `def _define_protocols(self)` |
| `_effective_hamiltonian` | method | `main5.py:419` | `def _effective_hamiltonian(self)` |
| `_evaluation_functional` | method | `main5.py:345` | `def _evaluation_functional(self, state_weights)` |
| `_free_hamiltonian` | method | `main5.py:555` | `def _free_hamiltonian(self)` |
| `_generate_fractal_graph` | method | `main5.py:625` | `def _generate_fractal_graph(self)` |
| `_generate_gibbs_measure` | method | `main5.py:217` | `def _generate_gibbs_measure(self)` |
| `_graph_to_distance_matrix` | method | `main5.py:722` | `def _graph_to_distance_matrix(self)` |
| `_initialize_leaves` | method | `main5.py:203` | `def _initialize_leaves(self)` |
| `_is_physical_state` | method | `main5.py:509` | `def _is_physical_state(self, state)` |
| `_local_jump_operator` | method | `main5.py:284` | `def _local_jump_operator(self, power)` |
| `_loss_potential` | method | `main5.py:561` | `def _loss_potential(self)` |
| `_modular_dissipator` | method | `main5.py:432` | `def _modular_dissipator(self, state)` |
| `_nonlinear_term` | method | `main5.py:444` | `def _nonlinear_term(self, state)` |
| `_normalize_density_matrix` | method | `main5.py:500` | `def _normalize_density_matrix(self, state)` |
| `_predict_diffraction_peak` | method | `main5.py:906` | `def _predict_diffraction_peak(self)` |
| `_pt_symmetry_condition` | method | `main5.py:573` | `def _pt_symmetry_condition(self)` |
| `_spectral_dimension` | method | `main5.py:649` | `def _spectral_dimension(self)` |
| `_spectral_moments` | method | `main5.py:163` | `def _spectral_moments(self, n)` |
| `_stochastic_term` | method | `main5.py:452` | `def _stochastic_term(self, dt)` |
| `_szego_projector` | method | `main5.py:335` | `def _szego_projector(self)` |
| `_topological_ramsey` | method | `main5.py:680` | `def _topological_ramsey(self)` |
| `apply_branching` | method | `main5.py:300` | `def apply_branching(self, state_vector)` |
| `bures_distance` | method | `main5.py:151` | `def bures_distance(self, other)` |
| `coherence_quantum` | method | `main5.py:579` | `def coherence_quantum(self)` |
| `compute_entropy_gap` | method | `main5.py:772` | `def compute_entropy_gap(self)` |
| `compute_freedom` | method | `main5.py:792` | `def compute_freedom(self)` |
| `compute_gibbs_free_energy` | method | `main5.py:251` | `def compute_gibbs_free_energy(self)` |
| `compute_log_bayes_factor` | method | `main5.py:912` | `def compute_log_bayes_factor(self)` |
| `compute_network_entropy` | method | `main5.py:751` | `def compute_network_entropy(self)` |
| `compute_pontryagin_number` | method | `main5.py:776` | `def compute_pontryagin_number(self)` |
| `compute_teleological_overlap` | method | `main5.py:396` | `def compute_teleological_overlap(self)` |
| `critical_percolation_time` | method | `main5.py:735` | `def critical_percolation_time(self)` |
| `evaluate_feasibility` | method | `main5.py:1014` | `def evaluate_feasibility(self, budget, time_limit)` |
| `evolve` | method | `main5.py:459` | `def evolve(self, rho0, t_span, n_steps)` |
| `haagerup_weight` | method | `main5.py:170` | `def haagerup_weight(self)` |
| `is_coherent_subgraph` | method | `main5.py:747` | `def is_coherent_subgraph(self, subgraph_nodes)` |
| `is_gauge_invariant` | method | `main5.py:803` | `def is_gauge_invariant(self)` |
| `ising_quantum` | method | `main5.py:823` | `def ising_quantum(network)` |
| `modular_entropy` | method | `main5.py:143` | `def modular_entropy(self)` |
| `predict_all` | method | `main5.py:891` | `def predict_all(self)` |
| `project` | method | `main5.py:361` | `def project(self, state_vector)` |
| `random_network` | method | `main5.py:860` | `def random_network(network)` |
| `simulate_experimental_outcome` | method | `main5.py:1027` | `def simulate_experimental_outcome(self, protocol_name)` |
| `simulate_resma_multiverse` | method | `main5.py:1052` | `def simulate_resma_multiverse(n_leaves, n_nodes, seed, validate_empirical)` |
| `spectral_density` | method | `main5.py:133` | `def spectral_density(self, omega)` |
| `syk4` | method | `main5.py:843` | `def syk4(network)` |
| `validate_connectome_size` | method | `main5.py:95` | `def validate_connectome_size(n_nodes)` |
| `validate_dimension` | method | `main5.py:74` | `def validate_dimension(alpha, tolerance)` |
| `validate_percolation_time` | method | `main5.py:106` | `def validate_percolation_time(t_c, expected, tolerance)` |
| `validate_pt_symmetry` | method | `main5.py:84` | `def validate_pt_symmetry(kappa, Omega, chi)` |
| `validate_spectral_dimension` | method | `main5.py:101` | `def validate_spectral_dimension(dim)` |
| `BicameralAttentionCPU` | class | `microbi.py.py:195` | `class BicameralAttentionCPU(Module)` |
| `CorpusCallosumCPU` | class | `microbi.py.py:436` | `class CorpusCallosumCPU(Module)` |
| `CurriculumSchedulerCPU` | class | `microbi.py.py:602` | `class CurriculumSchedulerCPU` |
| `EpistemicCuriosityCPU` | class | `microbi.py.py:37` | `class EpistemicCuriosityCPU(Module)` |
| `Flickr8kDatasetCPU` | class | `microbi.py.py:671` | `class Flickr8kDatasetCPU(Dataset)` |
| `LeftHemisphereCPU` | class | `microbi.py.py:308` | `class LeftHemisphereCPU(Module)` |
| `LiquidNeuronCPU` | class | `microbi.py.py:91` | `class LiquidNeuronCPU(Module)` |
| `NeuralDiagnosticsCPU` | class | `microbi.py.py:513` | `class NeuralDiagnosticsCPU` |
| `NeuroLogosBicameralCPU` | class | `microbi.py.py:473` | `class NeuroLogosBicameralCPU(Module)` |
| `RightHemisphereCPU` | class | `microbi.py.py:248` | `class RightHemisphereCPU(Module)` |
| `__getitem__` | method | `microbi.py.py:693` | `def __getitem__(self, idx)` |
| `__init__` | method | `microbi.py.py:38` | `def __init__(self, feature_dim, hidden_dim)` |
| `__init__` | method | `microbi.py.py:92` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `microbi.py.py:196` | `def __init__(self, dim, num_heads)` |
| `__init__` | method | `microbi.py.py:249` | `def __init__(self, output_dim)` |
| `__init__` | method | `microbi.py.py:309` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `microbi.py.py:437` | `def __init__(self, dim)` |
| `__init__` | method | `microbi.py.py:474` | `def __init__(self, vocab_size)` |
| `__init__` | method | `microbi.py.py:514` | `def __init__(self)` |
| `__init__` | method | `microbi.py.py:603` | `def __init__(self, total_epochs)` |
| `__init__` | method | `microbi.py.py:672` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `microbi.py.py:690` | `def __len__(self)` |
| `_get_init_state` | method | `microbi.py.py:416` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `microbi.py.py:421` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `microbi.py.py:650` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `compute_intrinsic_reward` | method | `microbi.py.py:55` | `def compute_intrinsic_reward(self, state, action, next_state)` |
| `consolidate_svd` | method | `microbi.py.py:168` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `microbi.py.py:117` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `microbi.py.py:211` | `def forward(self, x, mask)` |
| `forward` | method | `microbi.py.py:285` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `microbi.py.py:338` | `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)` |
| `forward` | method | `microbi.py.py:456` | `def forward(self, right_features)` |
| `forward` | method | `microbi.py.py:480` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)` |
| `get_exploration_bonus` | method | `microbi.py.py:627` | `def get_exploration_bonus(self, epoch)` |
| `get_phase` | method | `microbi.py.py:611` | `def get_phase(self, epoch)` |
| `get_plasticity` | method | `microbi.py.py:617` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `microbi.py.py:553` | `def get_recent_avg(self, key, n)` |
| `get_temperature` | method | `microbi.py.py:636` | `def get_temperature(self, epoch)` |
| `measure_callosal_flow` | method | `microbi.py.py:523` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `microbi.py.py:530` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `nan_hook` | method | `microbi.py.py:880` | `def nan_hook(module, grad_input, grad_output)` |
| `report` | method | `microbi.py.py:560` | `def report(self, epoch)` |
| `setup_flickr8k_cpu` | method | `microbi.py.py:721` | `def setup_flickr8k_cpu(data_dir)` |
| `should_consolidate` | method | `microbi.py.py:646` | `def should_consolidate(self, epoch)` |
| `train_bicameral_cpu` | method | `microbi.py.py:801` | `def train_bicameral_cpu()` |
| `update` | method | `microbi.py.py:71` | `def update(self, state, action, next_state)` |
| `update` | method | `microbi.py.py:548` | `def update(self)` |
| `CorpusCallosum` | class | `minibi.py:382` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `minibi.py:124` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphere` | class | `minibi.py:282` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `minibi.py:518` | `class LifeCycle` |
| `LiquidNeuron` | class | `minibi.py:167` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `minibi.py:420` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `minibi.py:403` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `minibi.py:246` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `minibi.py:146` | `def __getitem__(self, idx)` |
| `__init__` | method | `minibi.py:125` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `minibi.py:168` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `minibi.py:251` | `def __init__(self, output_dim)` |
| `__init__` | method | `minibi.py:283` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `minibi.py:383` | `def __init__(self, dim)` |
| `__init__` | method | `minibi.py:404` | `def __init__(self, vocab_size)` |
| `__init__` | method | `minibi.py:421` | `def __init__(self)` |
| `__init__` | method | `minibi.py:519` | `def __init__(self, total_epochs)` |
| `__len__` | method | `minibi.py:143` | `def __len__(self)` |
| `_get_init_state` | method | `minibi.py:361` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `minibi.py:366` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab` | method | `minibi.py:490` | `def build_vocab(ann_file, vocab_size)` |
| `build_vocab_flickr` | function | `minibi.py:104` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `consolidate_svd` | method | `minibi.py:219` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `minibi.py:185` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `minibi.py:268` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `minibi.py:305` | `def forward(self, visual_context, captions, max_len, return_gate, temperature)` |
| `forward` | method | `minibi.py:392` | `def forward(self, right_features)` |
| `forward` | method | `minibi.py:410` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature)` |
| `get_plasticity` | method | `minibi.py:522` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `minibi.py:448` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `minibi.py:432` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `minibi.py:439` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `minibi.py:455` | `def report(self, epoch)` |
| `setup_flickr8k` | function | `minibi.py:35` | `def setup_flickr8k(data_dir)` |
| `train_bicameral` | method | `minibi.py:536` | `def train_bicameral()` |
| `update` | method | `minibi.py:443` | `def update(self)` |
| `BicameralAttention` | class | `minibi2.py:212` | `class BicameralAttention(Module)` |
| `CorpusCallosumV2` | class | `minibi2.py:519` | `class CorpusCallosumV2(Module)` |
| `CurriculumScheduler` | class | `minibi2.py:700` | `class CurriculumScheduler` |
| `EpistemicCuriosity` | class | `minibi2.py:36` | `class EpistemicCuriosity(Module)` |
| `Flickr8kDataset` | class | `minibi2.py:781` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphereV2` | class | `minibi2.py:344` | `class LeftHemisphereV2(Module)` |
| `LiquidNeuronV2` | class | `minibi2.py:101` | `class LiquidNeuronV2(Module)` |
| `NeuralDiagnosticsV2` | class | `minibi2.py:591` | `class NeuralDiagnosticsV2` |
| `NeuroLogosBicameralV2` | class | `minibi2.py:562` | `class NeuroLogosBicameralV2(Module)` |
| `RightHemisphereV2` | class | `minibi2.py:279` | `class RightHemisphereV2(Module)` |
| `__getitem__` | method | `minibi2.py:803` | `def __getitem__(self, idx)` |
| `__init__` | method | `minibi2.py:41` | `def __init__(self, feature_dim, hidden_dim)` |
| `__init__` | method | `minibi2.py:108` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `minibi2.py:218` | `def __init__(self, dim, num_heads)` |
| `__init__` | method | `minibi2.py:286` | `def __init__(self, output_dim)` |
| `__init__` | method | `minibi2.py:352` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `minibi2.py:526` | `def __init__(self, dim)` |
| `__init__` | method | `minibi2.py:563` | `def __init__(self, vocab_size)` |
| `__init__` | method | `minibi2.py:592` | `def __init__(self)` |
| `__init__` | method | `minibi2.py:704` | `def __init__(self, total_epochs)` |
| `__init__` | method | `minibi2.py:782` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `minibi2.py:800` | `def __len__(self)` |
| `_get_init_state` | method | `minibi2.py:498` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `minibi2.py:503` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `minibi2.py:760` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `compute_intrinsic_reward` | method | `minibi2.py:60` | `def compute_intrinsic_reward(self, state, action, next_state)` |
| `consolidate_svd` | method | `minibi2.py:185` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `minibi2.py:137` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `minibi2.py:236` | `def forward(self, x, mask)` |
| `forward` | method | `minibi2.py:320` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `minibi2.py:387` | `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)` |
| `forward` | method | `minibi2.py:546` | `def forward(self, right_features)` |
| `forward` | method | `minibi2.py:569` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)` |
| `get_exploration_bonus` | method | `minibi2.py:732` | `def get_exploration_bonus(self, epoch)` |
| `get_phase` | method | `minibi2.py:712` | `def get_phase(self, epoch)` |
| `get_plasticity` | method | `minibi2.py:718` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `minibi2.py:644` | `def get_recent_avg(self, key, n)` |
| `get_temperature` | method | `minibi2.py:743` | `def get_temperature(self, epoch)` |
| `measure_callosal_flow` | method | `minibi2.py:612` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `minibi2.py:619` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `minibi2.py:651` | `def report(self, epoch)` |
| `setup_flickr8k` | method | `minibi2.py:821` | `def setup_flickr8k(data_dir)` |
| `should_consolidate` | method | `minibi2.py:755` | `def should_consolidate(self, epoch)` |
| `train_bicameral_v2` | method | `minibi2.py:893` | `def train_bicameral_v2()` |
| `update` | method | `minibi2.py:83` | `def update(self, state, action, next_state)` |
| `update` | method | `minibi2.py:639` | `def update(self)` |
| `BicameralAttention` | class | `minibi_c.py:197` | `class BicameralAttention(Module)` |
| `CorpusCallosumV2` | class | `minibi_c.py:433` | `class CorpusCallosumV2(Module)` |
| `CurriculumScheduler` | class | `minibi_c.py:593` | `class CurriculumScheduler` |
| `EpistemicCuriosity` | class | `minibi_c.py:37` | `class EpistemicCuriosity(Module)` |
| `Flickr8kDataset` | class | `minibi_c.py:667` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphereV2` | class | `minibi_c.py:301` | `class LeftHemisphereV2(Module)` |
| `LiquidNeuronV2` | class | `minibi_c.py:94` | `class LiquidNeuronV2(Module)` |
| `NeuralDiagnosticsV2` | class | `minibi_c.py:493` | `class NeuralDiagnosticsV2` |
| `NeuroLogosBicameralV2` | class | `minibi_c.py:464` | `class NeuroLogosBicameralV2(Module)` |
| `RightHemisphereV2` | class | `minibi_c.py:247` | `class RightHemisphereV2(Module)` |
| `__getitem__` | method | `minibi_c.py:689` | `def __getitem__(self, idx)` |
| `__init__` | method | `minibi_c.py:38` | `def __init__(self, feature_dim, hidden_dim)` |
| `__init__` | method | `minibi_c.py:95` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `minibi_c.py:198` | `def __init__(self, dim, num_heads)` |
| `__init__` | method | `minibi_c.py:248` | `def __init__(self, output_dim)` |
| `__init__` | method | `minibi_c.py:302` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `minibi_c.py:434` | `def __init__(self, dim)` |
| `__init__` | method | `minibi_c.py:465` | `def __init__(self, vocab_size)` |
| `__init__` | method | `minibi_c.py:494` | `def __init__(self)` |
| `__init__` | method | `minibi_c.py:594` | `def __init__(self, total_epochs)` |
| `__init__` | method | `minibi_c.py:668` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `minibi_c.py:686` | `def __len__(self)` |
| `_get_init_state` | method | `minibi_c.py:412` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `minibi_c.py:417` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `minibi_c.py:644` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `compute_intrinsic_reward` | method | `minibi_c.py:55` | `def compute_intrinsic_reward(self, state, action, next_state)` |
| `consolidate_svd` | method | `minibi_c.py:170` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `minibi_c.py:121` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `minibi_c.py:212` | `def forward(self, x, mask)` |
| `forward` | method | `minibi_c.py:281` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `minibi_c.py:332` | `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)` |
| `forward` | method | `minibi_c.py:450` | `def forward(self, right_features)` |
| `forward` | method | `minibi_c.py:471` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)` |
| `get_exploration_bonus` | method | `minibi_c.py:619` | `def get_exploration_bonus(self, epoch)` |
| `get_phase` | method | `minibi_c.py:602` | `def get_phase(self, epoch)` |
| `get_plasticity` | method | `minibi_c.py:608` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `minibi_c.py:541` | `def get_recent_avg(self, key, n)` |
| `get_temperature` | method | `minibi_c.py:629` | `def get_temperature(self, epoch)` |
| `measure_callosal_flow` | method | `minibi_c.py:510` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `minibi_c.py:517` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `minibi_c.py:548` | `def report(self, epoch)` |
| `setup_flickr8k` | method | `minibi_c.py:722` | `def setup_flickr8k(data_dir)` |
| `should_consolidate` | method | `minibi_c.py:640` | `def should_consolidate(self, epoch)` |
| `train_bicameral_v2` | method | `minibi_c.py:794` | `def train_bicameral_v2()` |
| `update` | method | `minibi_c.py:72` | `def update(self, state, action, next_state)` |
| `update` | method | `minibi_c.py:535` | `def update(self)` |
| `BicameralAttention` | class | `minibi_reduced.py.py:170` | `class BicameralAttention(Module)` |
| `CorpusCallosumV2` | class | `minibi_reduced.py.py:338` | `class CorpusCallosumV2(Module)` |
| `EpistemicCuriosity` | class | `minibi_reduced.py.py:39` | `class EpistemicCuriosity(Module)` |
| `Flickr8kDataset` | class | `minibi_reduced.py.py:396` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphereV2` | class | `minibi_reduced.py.py:247` | `class LeftHemisphereV2(Module)` |
| `LiquidNeuronV2` | class | `minibi_reduced.py.py:81` | `class LiquidNeuronV2(Module)` |
| `NeuroLogosBicameralV2` | class | `minibi_reduced.py.py:356` | `class NeuroLogosBicameralV2(Module)` |
| `RightHemisphereV2` | class | `minibi_reduced.py.py:213` | `class RightHemisphereV2(Module)` |
| `__getitem__` | method | `minibi_reduced.py.py:414` | `def __getitem__(self, idx)` |
| `__init__` | method | `minibi_reduced.py.py:40` | `def __init__(self, feature_dim, hidden_dim)` |
| `__init__` | method | `minibi_reduced.py.py:82` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `minibi_reduced.py.py:171` | `def __init__(self, dim, num_heads)` |
| `__init__` | method | `minibi_reduced.py.py:214` | `def __init__(self, output_dim)` |
| `__init__` | method | `minibi_reduced.py.py:248` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `minibi_reduced.py.py:339` | `def __init__(self, dim)` |
| `__init__` | method | `minibi_reduced.py.py:357` | `def __init__(self, vocab_size)` |
| `__init__` | method | `minibi_reduced.py.py:397` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `minibi_reduced.py.py:412` | `def __len__(self)` |
| `_get_init_state` | method | `minibi_reduced.py.py:323` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `minibi_reduced.py.py:328` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `minibi_reduced.py.py:381` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `compute_intrinsic_reward` | method | `minibi_reduced.py.py:55` | `def compute_intrinsic_reward(self, state, action, next_state)` |
| `consolidate_svd` | method | `minibi_reduced.py.py:141` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `minibi_reduced.py.py:105` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `minibi_reduced.py.py:184` | `def forward(self, x, mask)` |
| `forward` | method | `minibi_reduced.py.py:237` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `minibi_reduced.py.py:266` | `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)` |
| `forward` | method | `minibi_reduced.py.py:349` | `def forward(self, right_features)` |
| `forward` | method | `minibi_reduced.py.py:363` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)` |
| `setup_flickr8k` | method | `minibi_reduced.py.py:431` | `def setup_flickr8k(data_dir)` |
| `train_bicameral_v2` | method | `minibi_reduced.py.py:482` | `def train_bicameral_v2()` |
| `update` | method | `minibi_reduced.py.py:70` | `def update(self, state, action, next_state)` |
| `CorpusCallosum` | class | `miniminibi.py:314` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `miniminibi.py:442` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphere` | class | `miniminibi.py:214` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `miniminibi.py:508` | `class LifeCycle` |
| `LiquidNeuron` | class | `miniminibi.py:118` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `miniminibi.py:351` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `miniminibi.py:332` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `miniminibi.py:191` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `miniminibi.py:465` | `def __getitem__(self, idx)` |
| `__init__` | method | `miniminibi.py:119` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `miniminibi.py:192` | `def __init__(self, output_dim)` |
| `__init__` | method | `miniminibi.py:215` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `miniminibi.py:315` | `def __init__(self, dim)` |
| `__init__` | method | `miniminibi.py:333` | `def __init__(self, vocab_size)` |
| `__init__` | method | `miniminibi.py:353` | `def __init__(self)` |
| `__init__` | method | `miniminibi.py:443` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `miniminibi.py:509` | `def __init__(self, total_epochs)` |
| `__len__` | method | `miniminibi.py:462` | `def __len__(self)` |
| `_get_init_state` | method | `miniminibi.py:293` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `miniminibi.py:298` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `miniminibi.py:485` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `consolidate_svd` | method | `miniminibi.py:165` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `miniminibi.py:136` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `miniminibi.py:205` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `miniminibi.py:237` | `def forward(self, visual_context, captions, max_len, return_gate, temperature)` |
| `forward` | method | `miniminibi.py:324` | `def forward(self, right_features)` |
| `forward` | method | `miniminibi.py:339` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature)` |
| `get_plasticity` | method | `miniminibi.py:512` | `def get_plasticity(self, epoch)` |
| `measure_callosal_flow` | method | `miniminibi.py:363` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_gate_health` | method | `miniminibi.py:378` | `def measure_gate_health(self, gate_activations)` |
| `measure_vocab_diversity` | method | `miniminibi.py:372` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `miniminibi.py:392` | `def report(self, epoch)` |
| `setup_flickr8k` | function | `miniminibi.py:30` | `def setup_flickr8k(data_dir)` |
| `train_bicameral` | method | `miniminibi.py:523` | `def train_bicameral()` |
| `update` | method | `miniminibi.py:387` | `def update(self)` |
| `DataEnvironment` | class | `nemesis.py:23` | `class DataEnvironment` |
| `ExperimentConfig` | class | `nemesis.py:192` | `class ExperimentConfig` |
| `HyperLiquidNeuron` | class | `nemesis.py:60` | `class HyperLiquidNeuron(Module)` |
| `NemesisNetwork` | class | `nemesis.py:126` | `class NemesisNetwork(Module)` |
| `NeuralController` | class | `nemesis.py:41` | `class NeuralController(Module)` |
| `__init__` | method | `nemesis.py:24` | `def __init__(self)` |
| `__init__` | method | `nemesis.py:42` | `def __init__(self)` |
| `__init__` | method | `nemesis.py:61` | `def __init__(self, d_in, d_out, dynamic_mode)` |
| `__init__` | method | `nemesis.py:127` | `def __init__(self, config, dynamic_mode)` |
| `forward` | method | `nemesis.py:52` | `def forward(self, surprise, entropy)` |
| `forward` | method | `nemesis.py:79` | `def forward(self, x)` |
| `forward` | method | `nemesis.py:133` | `def forward(self, x)` |
| `get_batch` | method | `nemesis.py:33` | `def get_batch(self, phase, bs)` |
| `run_hyper_experiment` | method | `nemesis.py:143` | `def run_hyper_experiment(epochs, name, dynamic_mode)` |
| `seed_everything` | function | `nemesis.py:13` | `def seed_everything(seed)` |
| `CMSLayer` | class | `nested1.1.py:128` | `class CMSLayer(Module)` |
| `Config` | class | `nested1.1.py:27` | `class Config` |
| `NestedBrain` | class | `nested1.1.py:200` | `class NestedBrain(Module)` |
| `__init__` | method | `nested1.1.py:129` | `def __init__(self, dim, config)` |
| `__init__` | method | `nested1.1.py:201` | `def __init__(self, config)` |
| `cleanup_old_checkpoints` | method | `nested1.1.py:105` | `def cleanup_old_checkpoints(checkpoint_dir, keep_last)` |
| `evaluate` | method | `nested1.1.py:267` | `def evaluate(model, loader, device)` |
| `forward` | method | `nested1.1.py:148` | `def forward(self, x)` |
| `forward` | method | `nested1.1.py:221` | `def forward(self, x)` |
| `get_ablation_state` | method | `nested1.1.py:226` | `def get_ablation_state(self)` |
| `get_cifar10_loaders` | method | `nested1.1.py:241` | `def get_cifar10_loaders(config)` |
| `get_norms` | method | `nested1.1.py:190` | `def get_norms(self)` |
| `get_norms` | method | `nested1.1.py:233` | `def get_norms(self)` |
| `run_ablation_study` | method | `nested1.1.py:371` | `def run_ablation_study()` |
| `safe_serialize` | method | `nested1.1.py:61` | `def safe_serialize(obj)` |
| `save_checkpoint` | method | `nested1.1.py:80` | `def save_checkpoint(epoch, model_state, optimizer_state, config, metrics, checkpoint_dir)` |
| `train` | method | `nested1.1.py:293` | `def train(config)` |
| `CMSLayer` | class | `nested1.py:59` | `class CMSLayer(Module)` |
| `Config` | class | `nested1.py:24` | `class Config` |
| `NestedBrain` | class | `nested1.py:141` | `class NestedBrain(Module)` |
| `__init__` | method | `nested1.py:60` | `def __init__(self, dim, config)` |
| `__init__` | method | `nested1.py:142` | `def __init__(self, config)` |
| `evaluate` | method | `nested1.py:208` | `def evaluate(model, loader, device)` |
| `forward` | method | `nested1.py:83` | `def forward(self, x)` |
| `forward` | method | `nested1.py:162` | `def forward(self, x)` |
| `get_ablation_state` | method | `nested1.py:167` | `def get_ablation_state(self)` |
| `get_cifar10_loaders` | method | `nested1.py:182` | `def get_cifar10_loaders(config)` |
| `get_norms` | method | `nested1.py:131` | `def get_norms(self)` |
| `get_norms` | method | `nested1.py:174` | `def get_norms(self)` |
| `run_ablation_study` | method | `nested1.py:301` | `def run_ablation_study()` |
| `train` | method | `nested1.py:234` | `def train(config)` |
| `AdaptiveCombinatorialComplexLayer` | class | `nestedtopobrain.py:750` | `class AdaptiveCombinatorialComplexLayer(Module)` |
| `AsymmetricPredictiveErrorCell` | class | `nestedtopobrain.py:542` | `class AsymmetricPredictiveErrorCell(Module)` |
| `CheckpointManager` | class | `nestedtopobrain.py:429` | `class CheckpointManager` |
| `Config` | class | `nestedtopobrain.py:29` | `class Config` |
| `ContinuumMemoryCell` | class | `nestedtopobrain.py:625` | `class ContinuumMemoryCell(Module)` |
| `LearnableAbsenceGating` | class | `nestedtopobrain.py:562` | `class LearnableAbsenceGating(Module)` |
| `PrefrontalOrchestrator` | class | `nestedtopobrain.py:178` | `class PrefrontalOrchestrator(Module)` |
| `ResidualBlock` | class | `nestedtopobrain.py:946` | `class ResidualBlock(Module)` |
| `ResourceMonitor` | class | `nestedtopobrain.py:133` | `class ResourceMonitor` |
| `SupConLoss` | class | `nestedtopobrain.py:525` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `nestedtopobrain.py:580` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainV24` | class | `nestedtopobrain.py:1022` | `class TopoBrainV24(Module)` |
| `TopologicalHealthSovereignty` | class | `nestedtopobrain.py:343` | `class TopologicalHealthSovereignty` |
| `TopologyMetrics` | class | `nestedtopobrain.py:334` | `class TopologyMetrics` |
| `VisualCortex` | class | `nestedtopobrain.py:967` | `class VisualCortex(Module)` |
| `__init__` | method | `nestedtopobrain.py:185` | `def __init__(self, config)` |
| `__init__` | method | `nestedtopobrain.py:350` | `def __init__(self, model, config, epsilon_c)` |
| `__init__` | method | `nestedtopobrain.py:431` | `def __init__(self, run_name)` |
| `__init__` | method | `nestedtopobrain.py:526` | `def __init__(self, temperature)` |
| `__init__` | method | `nestedtopobrain.py:543` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `nestedtopobrain.py:563` | `def __init__(self, dim, min_gate)` |
| `__init__` | method | `nestedtopobrain.py:581` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `nestedtopobrain.py:626` | `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)` |
| `__init__` | method | `nestedtopobrain.py:751` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)` |
| `__init__` | method | `nestedtopobrain.py:947` | `def __init__(self, in_channels, out_channels, stride)` |
| `__init__` | method | `nestedtopobrain.py:968` | `def __init__(self, output_dim, grid_size)` |
| `__init__` | method | `nestedtopobrain.py:1023` | `def __init__(self, config, in_channels)` |
| `__post_init__` | method | `nestedtopobrain.py:89` | `def __post_init__(self)` |
| `_analyze_matrix` | method | `nestedtopobrain.py:356` | `def _analyze_matrix(self, weight_matrix, name)` |
| `_init_grid_topology` | method | `nestedtopobrain.py:1193` | `def _init_grid_topology(self, N)` |
| `_initialize_layer_memory` | method | `nestedtopobrain.py:1124` | `def _initialize_layer_memory(self, cell, x_input, name)` |
| `_maintain_orthogonality` | method | `nestedtopobrain.py:595` | `def _maintain_orthogonality(self)` |
| `_validate_and_fix_state` | method | `nestedtopobrain.py:792` | `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)` |
| `analyze_gradient_flow` | method | `nestedtopobrain.py:1905` | `def analyze_gradient_flow(model, epoch, run_name)` |
| `analyze_topology_clustering` | method | `nestedtopobrain.py:1472` | `def analyze_topology_clustering(model, run_name)` |
| `analyze_topology_evolution` | method | `nestedtopobrain.py:1629` | `def analyze_topology_evolution(run_name)` |
| `analyze_topology_flow` | method | `nestedtopobrain.py:1511` | `def analyze_topology_flow(model, dataloader, run_name, num_samples)` |
| `calculate` | method | `nestedtopobrain.py:402` | `def calculate(self, epoch)` |
| `calculate_ortho_loss` | method | `nestedtopobrain.py:1174` | `def calculate_ortho_loss(self, ortho_deviation, controls)` |
| `calculate_topology_diversity_loss` | method | `nestedtopobrain.py:1180` | `def calculate_topology_diversity_loss(self, controls)` |
| `check_limit` | method | `nestedtopobrain.py:159` | `def check_limit(limit_gb, abort_on_limit)` |
| `clear_cache` | method | `nestedtopobrain.py:153` | `def clear_cache()` |
| `comprehensive_topology_analysis` | method | `nestedtopobrain.py:1690` | `def comprehensive_topology_analysis(model, dataloader, run_name)` |
| `consolidate_semantic_memories` | method | `nestedtopobrain.py:1141` | `def consolidate_semantic_memories(self)` |
| `detach_state` | method | `nestedtopobrain.py:322` | `def detach_state(self)` |
| `evaluate` | method | `nestedtopobrain.py:2050` | `def evaluate(model, loader, config, adversarial, controls)` |
| `forward` | method | `nestedtopobrain.py:222` | `def forward(self, metrics_dict)` |
| `forward` | method | `nestedtopobrain.py:529` | `def forward(self, features, labels)` |
| `forward` | method | `nestedtopobrain.py:553` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `nestedtopobrain.py:573` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `nestedtopobrain.py:600` | `def forward(self, x)` |
| `forward` | method | `nestedtopobrain.py:675` | `def forward(self, x, state_M, controls)` |
| `forward` | method | `nestedtopobrain.py:817` | `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)` |
| `forward` | method | `nestedtopobrain.py:961` | `def forward(self, x)` |
| `forward` | method | `nestedtopobrain.py:994` | `def forward(self, x)` |
| `forward` | method | `nestedtopobrain.py:1226` | `def forward(self, x, prev_states, controls)` |
| `get_critical_summary` | method | `nestedtopobrain.py:419` | `def get_critical_summary(self)` |
| `get_dataloaders` | method | `nestedtopobrain.py:481` | `def get_dataloaders(config)` |
| `get_gpu_memory_gb` | method | `nestedtopobrain.py:140` | `def get_gpu_memory_gb()` |
| `get_memory_gb` | method | `nestedtopobrain.py:135` | `def get_memory_gb()` |
| `get_node_importance` | method | `nestedtopobrain.py:811` | `def get_node_importance(self)` |
| `get_sparsity_lambda` | method | `nestedtopobrain.py:104` | `def get_sparsity_lambda(self, epoch)` |
| `get_supcon_lambda` | method | `nestedtopobrain.py:98` | `def get_supcon_lambda(self, epoch)` |
| `get_topology` | method | `nestedtopobrain.py:1215` | `def get_topology(self, return_sparse)` |
| `initialize_memories` | method | `nestedtopobrain.py:1087` | `def initialize_memories(self, dataloader)` |
| `invalidate_sparse_cache` | method | `nestedtopobrain.py:789` | `def invalidate_sparse_cache(self)` |
| `load` | method | `nestedtopobrain.py:464` | `def load(self, name)` |
| `log` | method | `nestedtopobrain.py:146` | `def log(prefix)` |
| `main` | method | `nestedtopobrain.py:2518` | `def main()` |
| `make_adversarial_pgd` | method | `nestedtopobrain.py:1972` | `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)` |
| `prune_topology` | method | `nestedtopobrain.py:1298` | `def prune_topology(self, controls)` |
| `reset_context` | method | `nestedtopobrain.py:327` | `def reset_context(self)` |
| `run_ablation_study` | method | `nestedtopobrain.py:1713` | `def run_ablation_study()` |
| `save` | method | `nestedtopobrain.py:436` | `def save(self, data, name)` |
| `save_node_importance_viz` | method | `nestedtopobrain.py:1450` | `def save_node_importance_viz(model, epoch, run_name)` |
| `save_topology_visualization` | method | `nestedtopobrain.py:1406` | `def save_topology_visualization(model, epoch, run_name)` |
| `seed_everything` | method | `nestedtopobrain.py:119` | `def seed_everything(seed)` |
| `set_epoch` | method | `nestedtopobrain.py:1169` | `def set_epoch(self, epoch)` |
| `to_dict` | method | `nestedtopobrain.py:95` | `def to_dict(self)` |
| `train_epoch` | method | `nestedtopobrain.py:2122` | `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)` |
| `train_model` | method | `nestedtopobrain.py:2347` | `def train_model(config, run_name)` |
| `visualize_memory_evolution` | method | `nestedtopobrain.py:1814` | `def visualize_memory_evolution(model, epoch, run_name)` |
| `visualize_topology_as_graph` | method | `nestedtopobrain.py:1577` | `def visualize_topology_as_graph(model, run_name, threshold)` |
| `warmup_topo` | method | `nestedtopobrain.py:2381` | `def warmup_topo(epoch)` |
| `AdaptiveCombinatorialComplexLayer` | class | `nestedtopobrain_v1.py:676` | `class AdaptiveCombinatorialComplexLayer(Module)` |
| `AsymmetricPredictiveErrorCell` | class | `nestedtopobrain_v1.py:472` | `class AsymmetricPredictiveErrorCell(Module)` |
| `CheckpointManager` | class | `nestedtopobrain_v1.py:364` | `class CheckpointManager` |
| `Config` | class | `nestedtopobrain_v1.py:29` | `class Config` |
| `ContinuumMemoryCell` | class | `nestedtopobrain_v1.py:549` | `class ContinuumMemoryCell(Module)` |
| `LearnableAbsenceGating` | class | `nestedtopobrain_v1.py:492` | `class LearnableAbsenceGating(Module)` |
| `PrefrontalOrchestrator` | class | `nestedtopobrain_v1.py:173` | `class PrefrontalOrchestrator(Module)` |
| `ResourceMonitor` | class | `nestedtopobrain_v1.py:128` | `class ResourceMonitor` |
| `SupConLoss` | class | `nestedtopobrain_v1.py:455` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `nestedtopobrain_v1.py:510` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainV24` | class | `nestedtopobrain_v1.py:879` | `class TopoBrainV24(Module)` |
| `TopologicalHealthSovereignty` | class | `nestedtopobrain_v1.py:278` | `class TopologicalHealthSovereignty` |
| `TopologyMetrics` | class | `nestedtopobrain_v1.py:269` | `class TopologyMetrics` |
| `__init__` | method | `nestedtopobrain_v1.py:180` | `def __init__(self, config)` |
| `__init__` | method | `nestedtopobrain_v1.py:285` | `def __init__(self, model, config, epsilon_c)` |
| `__init__` | method | `nestedtopobrain_v1.py:366` | `def __init__(self, run_name)` |
| `__init__` | method | `nestedtopobrain_v1.py:456` | `def __init__(self, temperature)` |
| `__init__` | method | `nestedtopobrain_v1.py:473` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `nestedtopobrain_v1.py:493` | `def __init__(self, dim, min_gate)` |
| `__init__` | method | `nestedtopobrain_v1.py:511` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `nestedtopobrain_v1.py:550` | `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)` |
| `__init__` | method | `nestedtopobrain_v1.py:677` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)` |
| `__init__` | method | `nestedtopobrain_v1.py:880` | `def __init__(self, config, in_channels)` |
| `__post_init__` | method | `nestedtopobrain_v1.py:83` | `def __post_init__(self)` |
| `_analyze_matrix` | method | `nestedtopobrain_v1.py:291` | `def _analyze_matrix(self, weight_matrix, name)` |
| `_init_grid_topology` | method | `nestedtopobrain_v1.py:1095` | `def _init_grid_topology(self, N)` |
| `_initialize_layer_memory` | method | `nestedtopobrain_v1.py:974` | `def _initialize_layer_memory(self, cell, x_input, name)` |
| `_maintain_orthogonality` | method | `nestedtopobrain_v1.py:525` | `def _maintain_orthogonality(self)` |
| `_validate_and_fix_state` | method | `nestedtopobrain_v1.py:718` | `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)` |
| `analyze_gradient_flow` | method | `nestedtopobrain_v1.py:1789` | `def analyze_gradient_flow(model, epoch, run_name)` |
| `analyze_topology_clustering` | method | `nestedtopobrain_v1.py:1370` | `def analyze_topology_clustering(model, run_name)` |
| `analyze_topology_evolution` | method | `nestedtopobrain_v1.py:1513` | `def analyze_topology_evolution(run_name)` |
| `analyze_topology_flow` | method | `nestedtopobrain_v1.py:1409` | `def analyze_topology_flow(model, dataloader, run_name, num_samples)` |
| `calculate` | method | `nestedtopobrain_v1.py:337` | `def calculate(self, epoch)` |
| `calculate_ortho_loss` | method | `nestedtopobrain_v1.py:1050` | `def calculate_ortho_loss(self, controls)` |
| `calculate_topology_diversity_loss` | method | `nestedtopobrain_v1.py:1070` | `def calculate_topology_diversity_loss(self, controls)` |
| `check_limit` | method | `nestedtopobrain_v1.py:154` | `def check_limit(limit_gb, abort_on_limit)` |
| `clear_cache` | method | `nestedtopobrain_v1.py:148` | `def clear_cache()` |
| `comprehensive_topology_analysis` | method | `nestedtopobrain_v1.py:1574` | `def comprehensive_topology_analysis(model, dataloader, run_name)` |
| `consolidate_semantic_memories` | method | `nestedtopobrain_v1.py:1000` | `def consolidate_semantic_memories(self)` |
| `detach_state` | method | `nestedtopobrain_v1.py:257` | `def detach_state(self)` |
| `evaluate` | method | `nestedtopobrain_v1.py:1936` | `def evaluate(model, loader, config, adversarial, controls)` |
| `forward` | method | `nestedtopobrain_v1.py:217` | `def forward(self, metrics_dict)` |
| `forward` | method | `nestedtopobrain_v1.py:459` | `def forward(self, features, labels)` |
| `forward` | method | `nestedtopobrain_v1.py:483` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `nestedtopobrain_v1.py:503` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `nestedtopobrain_v1.py:530` | `def forward(self, x)` |
| `forward` | method | `nestedtopobrain_v1.py:599` | `def forward(self, x, state_M, controls)` |
| `forward` | method | `nestedtopobrain_v1.py:743` | `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)` |
| `forward` | method | `nestedtopobrain_v1.py:1131` | `def forward(self, x, prev_states, controls)` |
| `get_critical_summary` | method | `nestedtopobrain_v1.py:354` | `def get_critical_summary(self)` |
| `get_dataloaders` | method | `nestedtopobrain_v1.py:416` | `def get_dataloaders(config)` |
| `get_gpu_memory_gb` | method | `nestedtopobrain_v1.py:135` | `def get_gpu_memory_gb()` |
| `get_memory_gb` | method | `nestedtopobrain_v1.py:130` | `def get_memory_gb()` |
| `get_node_importance` | method | `nestedtopobrain_v1.py:737` | `def get_node_importance(self)` |
| `get_sparsity_lambda` | method | `nestedtopobrain_v1.py:98` | `def get_sparsity_lambda(self, epoch)` |
| `get_supcon_lambda` | method | `nestedtopobrain_v1.py:92` | `def get_supcon_lambda(self, epoch)` |
| `get_topology` | method | `nestedtopobrain_v1.py:1118` | `def get_topology(self, return_sparse)` |
| `initialize_memories` | method | `nestedtopobrain_v1.py:934` | `def initialize_memories(self, dataloader)` |
| `invalidate_sparse_cache` | method | `nestedtopobrain_v1.py:715` | `def invalidate_sparse_cache(self)` |
| `load` | method | `nestedtopobrain_v1.py:399` | `def load(self, name)` |
| `log` | method | `nestedtopobrain_v1.py:141` | `def log(prefix)` |
| `main` | method | `nestedtopobrain_v1.py:2398` | `def main()` |
| `make_adversarial_pgd` | method | `nestedtopobrain_v1.py:1856` | `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)` |
| `prune_topology` | method | `nestedtopobrain_v1.py:1198` | `def prune_topology(self, controls)` |
| `reset_context` | method | `nestedtopobrain_v1.py:262` | `def reset_context(self)` |
| `run_ablation_study` | method | `nestedtopobrain_v1.py:1597` | `def run_ablation_study()` |
| `save` | method | `nestedtopobrain_v1.py:371` | `def save(self, data, name)` |
| `save_node_importance_viz` | method | `nestedtopobrain_v1.py:1348` | `def save_node_importance_viz(model, epoch, run_name)` |
| `save_topology_visualization` | method | `nestedtopobrain_v1.py:1304` | `def save_topology_visualization(model, epoch, run_name)` |
| `seed_everything` | method | `nestedtopobrain_v1.py:114` | `def seed_everything(seed)` |
| `set_epoch` | method | `nestedtopobrain_v1.py:1045` | `def set_epoch(self, epoch)` |
| `to_dict` | method | `nestedtopobrain_v1.py:89` | `def to_dict(self)` |
| `train_epoch` | method | `nestedtopobrain_v1.py:2000` | `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)` |
| `train_model` | method | `nestedtopobrain_v1.py:2196` | `def train_model(config, run_name)` |
| `visualize_memory_evolution` | method | `nestedtopobrain_v1.py:1698` | `def visualize_memory_evolution(model, epoch, run_name)` |
| `visualize_topology_as_graph` | method | `nestedtopobrain_v1.py:1461` | `def visualize_topology_as_graph(model, run_name, threshold)` |
| `warmup_topo` | method | `nestedtopobrain_v1.py:2234` | `def warmup_topo(epoch)` |
| `AdaptiveCombinatorialComplexLayer` | class | `nestedtopobrain_v2.py:681` | `class AdaptiveCombinatorialComplexLayer(Module)` |
| `AsymmetricPredictiveErrorCell` | class | `nestedtopobrain_v2.py:471` | `class AsymmetricPredictiveErrorCell(Module)` |
| `CheckpointManager` | class | `nestedtopobrain_v2.py:363` | `class CheckpointManager` |
| `Config` | class | `nestedtopobrain_v2.py:29` | `class Config` |
| `ContinuumMemoryCell` | class | `nestedtopobrain_v2.py:554` | `class ContinuumMemoryCell(Module)` |
| `LearnableAbsenceGating` | class | `nestedtopobrain_v2.py:491` | `class LearnableAbsenceGating(Module)` |
| `PrefrontalOrchestrator` | class | `nestedtopobrain_v2.py:172` | `class PrefrontalOrchestrator(Module)` |
| `ResourceMonitor` | class | `nestedtopobrain_v2.py:127` | `class ResourceMonitor` |
| `SupConLoss` | class | `nestedtopobrain_v2.py:454` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `nestedtopobrain_v2.py:509` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainV24` | class | `nestedtopobrain_v2.py:878` | `class TopoBrainV24(Module)` |
| `TopologicalHealthSovereignty` | class | `nestedtopobrain_v2.py:277` | `class TopologicalHealthSovereignty` |
| `TopologyMetrics` | class | `nestedtopobrain_v2.py:268` | `class TopologyMetrics` |
| `__init__` | method | `nestedtopobrain_v2.py:179` | `def __init__(self, config)` |
| `__init__` | method | `nestedtopobrain_v2.py:284` | `def __init__(self, model, config, epsilon_c)` |
| `__init__` | method | `nestedtopobrain_v2.py:365` | `def __init__(self, run_name)` |
| `__init__` | method | `nestedtopobrain_v2.py:455` | `def __init__(self, temperature)` |
| `__init__` | method | `nestedtopobrain_v2.py:472` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `nestedtopobrain_v2.py:492` | `def __init__(self, dim, min_gate)` |
| `__init__` | method | `nestedtopobrain_v2.py:510` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `nestedtopobrain_v2.py:555` | `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)` |
| `__init__` | method | `nestedtopobrain_v2.py:682` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)` |
| `__init__` | method | `nestedtopobrain_v2.py:879` | `def __init__(self, config, in_channels)` |
| `__post_init__` | method | `nestedtopobrain_v2.py:82` | `def __post_init__(self)` |
| `_analyze_matrix` | method | `nestedtopobrain_v2.py:290` | `def _analyze_matrix(self, weight_matrix, name)` |
| `_init_grid_topology` | method | `nestedtopobrain_v2.py:1085` | `def _init_grid_topology(self, N)` |
| `_initialize_layer_memory` | method | `nestedtopobrain_v2.py:976` | `def _initialize_layer_memory(self, cell, x_input, name)` |
| `_maintain_orthogonality` | method | `nestedtopobrain_v2.py:524` | `def _maintain_orthogonality(self)` |
| `_validate_and_fix_state` | method | `nestedtopobrain_v2.py:723` | `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)` |
| `analyze_gradient_flow` | method | `nestedtopobrain_v2.py:1776` | `def analyze_gradient_flow(model, epoch, run_name)` |
| `analyze_topology_clustering` | method | `nestedtopobrain_v2.py:1355` | `def analyze_topology_clustering(model, run_name)` |
| `analyze_topology_evolution` | method | `nestedtopobrain_v2.py:1500` | `def analyze_topology_evolution(run_name)` |
| `analyze_topology_flow` | method | `nestedtopobrain_v2.py:1394` | `def analyze_topology_flow(model, dataloader, run_name, num_samples)` |
| `calculate` | method | `nestedtopobrain_v2.py:336` | `def calculate(self, epoch)` |
| `calculate_ortho_loss` | method | `nestedtopobrain_v2.py:1052` | `def calculate_ortho_loss(self, ortho_deviation, controls)` |
| `calculate_topology_diversity_loss` | method | `nestedtopobrain_v2.py:1060` | `def calculate_topology_diversity_loss(self, controls)` |
| `check_limit` | method | `nestedtopobrain_v2.py:153` | `def check_limit(limit_gb, abort_on_limit)` |
| `clear_cache` | method | `nestedtopobrain_v2.py:147` | `def clear_cache()` |
| `comprehensive_topology_analysis` | method | `nestedtopobrain_v2.py:1561` | `def comprehensive_topology_analysis(model, dataloader, run_name)` |
| `consolidate_semantic_memories` | method | `nestedtopobrain_v2.py:1002` | `def consolidate_semantic_memories(self)` |
| `detach_state` | method | `nestedtopobrain_v2.py:256` | `def detach_state(self)` |
| `evaluate` | method | `nestedtopobrain_v2.py:1910` | `def evaluate(model, loader, config, adversarial, controls)` |
| `forward` | method | `nestedtopobrain_v2.py:216` | `def forward(self, metrics_dict)` |
| `forward` | method | `nestedtopobrain_v2.py:458` | `def forward(self, features, labels)` |
| `forward` | method | `nestedtopobrain_v2.py:482` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `nestedtopobrain_v2.py:502` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `nestedtopobrain_v2.py:529` | `def forward(self, x)` |
| `forward` | method | `nestedtopobrain_v2.py:604` | `def forward(self, x, state_M, controls)` |
| `forward` | method | `nestedtopobrain_v2.py:748` | `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)` |
| `forward` | method | `nestedtopobrain_v2.py:1121` | `def forward(self, x, prev_states, controls)` |
| `get_critical_summary` | method | `nestedtopobrain_v2.py:353` | `def get_critical_summary(self)` |
| `get_dataloaders` | method | `nestedtopobrain_v2.py:415` | `def get_dataloaders(config)` |
| `get_gpu_memory_gb` | method | `nestedtopobrain_v2.py:134` | `def get_gpu_memory_gb()` |
| `get_memory_gb` | method | `nestedtopobrain_v2.py:129` | `def get_memory_gb()` |
| `get_node_importance` | method | `nestedtopobrain_v2.py:742` | `def get_node_importance(self)` |
| `get_sparsity_lambda` | method | `nestedtopobrain_v2.py:97` | `def get_sparsity_lambda(self, epoch)` |
| `get_supcon_lambda` | method | `nestedtopobrain_v2.py:91` | `def get_supcon_lambda(self, epoch)` |
| `get_topology` | method | `nestedtopobrain_v2.py:1108` | `def get_topology(self, return_sparse)` |
| `initialize_memories` | method | `nestedtopobrain_v2.py:933` | `def initialize_memories(self, dataloader)` |
| `invalidate_sparse_cache` | method | `nestedtopobrain_v2.py:720` | `def invalidate_sparse_cache(self)` |
| `load` | method | `nestedtopobrain_v2.py:398` | `def load(self, name)` |
| `log` | method | `nestedtopobrain_v2.py:140` | `def log(prefix)` |
| `main` | method | `nestedtopobrain_v2.py:2344` | `def main()` |
| `make_adversarial_pgd` | method | `nestedtopobrain_v2.py:1843` | `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)` |
| `prune_topology` | method | `nestedtopobrain_v2.py:1183` | `def prune_topology(self, controls)` |
| `reset_context` | method | `nestedtopobrain_v2.py:261` | `def reset_context(self)` |
| `run_ablation_study` | method | `nestedtopobrain_v2.py:1584` | `def run_ablation_study()` |
| `save` | method | `nestedtopobrain_v2.py:370` | `def save(self, data, name)` |
| `save_node_importance_viz` | method | `nestedtopobrain_v2.py:1333` | `def save_node_importance_viz(model, epoch, run_name)` |
| `save_topology_visualization` | method | `nestedtopobrain_v2.py:1289` | `def save_topology_visualization(model, epoch, run_name)` |
| `seed_everything` | method | `nestedtopobrain_v2.py:113` | `def seed_everything(seed)` |
| `set_epoch` | method | `nestedtopobrain_v2.py:1047` | `def set_epoch(self, epoch)` |
| `to_dict` | method | `nestedtopobrain_v2.py:88` | `def to_dict(self)` |
| `train_epoch` | method | `nestedtopobrain_v2.py:1963` | `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)` |
| `train_model` | method | `nestedtopobrain_v2.py:2132` | `def train_model(config, run_name)` |
| `visualize_memory_evolution` | method | `nestedtopobrain_v2.py:1685` | `def visualize_memory_evolution(model, epoch, run_name)` |
| `visualize_topology_as_graph` | method | `nestedtopobrain_v2.py:1448` | `def visualize_topology_as_graph(model, run_name, threshold)` |
| `warmup_topo` | method | `nestedtopobrain_v2.py:2180` | `def warmup_topo(epoch)` |
| `AdaptiveCombinatorialComplexLayer` | class | `nestedtopobrain_v3.py:737` | `class AdaptiveCombinatorialComplexLayer(Module)` |
| `AsymmetricPredictiveErrorCell` | class | `nestedtopobrain_v3.py:527` | `class AsymmetricPredictiveErrorCell(Module)` |
| `CheckpointManager` | class | `nestedtopobrain_v3.py:419` | `class CheckpointManager` |
| `Config` | class | `nestedtopobrain_v3.py:29` | `class Config` |
| `ContinuumMemoryCell` | class | `nestedtopobrain_v3.py:610` | `class ContinuumMemoryCell(Module)` |
| `LearnableAbsenceGating` | class | `nestedtopobrain_v3.py:547` | `class LearnableAbsenceGating(Module)` |
| `PrefrontalOrchestrator` | class | `nestedtopobrain_v3.py:172` | `class PrefrontalOrchestrator(Module)` |
| `ResourceMonitor` | class | `nestedtopobrain_v3.py:127` | `class ResourceMonitor` |
| `SupConLoss` | class | `nestedtopobrain_v3.py:510` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `nestedtopobrain_v3.py:565` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainV24` | class | `nestedtopobrain_v3.py:934` | `class TopoBrainV24(Module)` |
| `TopologicalHealthSovereignty` | class | `nestedtopobrain_v3.py:333` | `class TopologicalHealthSovereignty` |
| `TopologyMetrics` | class | `nestedtopobrain_v3.py:324` | `class TopologyMetrics` |
| `__init__` | method | `nestedtopobrain_v3.py:179` | `def __init__(self, config)` |
| `__init__` | method | `nestedtopobrain_v3.py:340` | `def __init__(self, model, config, epsilon_c)` |
| `__init__` | method | `nestedtopobrain_v3.py:421` | `def __init__(self, run_name)` |
| `__init__` | method | `nestedtopobrain_v3.py:511` | `def __init__(self, temperature)` |
| `__init__` | method | `nestedtopobrain_v3.py:528` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `nestedtopobrain_v3.py:548` | `def __init__(self, dim, min_gate)` |
| `__init__` | method | `nestedtopobrain_v3.py:566` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `nestedtopobrain_v3.py:611` | `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)` |
| `__init__` | method | `nestedtopobrain_v3.py:738` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)` |
| `__init__` | method | `nestedtopobrain_v3.py:935` | `def __init__(self, config, in_channels)` |
| `__post_init__` | method | `nestedtopobrain_v3.py:82` | `def __post_init__(self)` |
| `_analyze_matrix` | method | `nestedtopobrain_v3.py:346` | `def _analyze_matrix(self, weight_matrix, name)` |
| `_init_grid_topology` | method | `nestedtopobrain_v3.py:1141` | `def _init_grid_topology(self, N)` |
| `_initialize_layer_memory` | method | `nestedtopobrain_v3.py:1032` | `def _initialize_layer_memory(self, cell, x_input, name)` |
| `_maintain_orthogonality` | method | `nestedtopobrain_v3.py:580` | `def _maintain_orthogonality(self)` |
| `_validate_and_fix_state` | method | `nestedtopobrain_v3.py:779` | `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)` |
| `analyze_gradient_flow` | method | `nestedtopobrain_v3.py:1887` | `def analyze_gradient_flow(model, epoch, run_name)` |
| `analyze_topology_clustering` | method | `nestedtopobrain_v3.py:1454` | `def analyze_topology_clustering(model, run_name)` |
| `analyze_topology_evolution` | method | `nestedtopobrain_v3.py:1611` | `def analyze_topology_evolution(run_name)` |
| `analyze_topology_flow` | method | `nestedtopobrain_v3.py:1493` | `def analyze_topology_flow(model, dataloader, run_name, num_samples)` |
| `calculate` | method | `nestedtopobrain_v3.py:392` | `def calculate(self, epoch)` |
| `calculate_ortho_loss` | method | `nestedtopobrain_v3.py:1108` | `def calculate_ortho_loss(self, ortho_deviation, controls)` |
| `calculate_topology_diversity_loss` | method | `nestedtopobrain_v3.py:1116` | `def calculate_topology_diversity_loss(self, controls)` |
| `check_limit` | method | `nestedtopobrain_v3.py:153` | `def check_limit(limit_gb, abort_on_limit)` |
| `clear_cache` | method | `nestedtopobrain_v3.py:147` | `def clear_cache()` |
| `comprehensive_topology_analysis` | method | `nestedtopobrain_v3.py:1672` | `def comprehensive_topology_analysis(model, dataloader, run_name)` |
| `consolidate_semantic_memories` | method | `nestedtopobrain_v3.py:1058` | `def consolidate_semantic_memories(self)` |
| `detach_state` | method | `nestedtopobrain_v3.py:312` | `def detach_state(self)` |
| `evaluate` | method | `nestedtopobrain_v3.py:2021` | `def evaluate(model, loader, config, adversarial, controls)` |
| `forward` | method | `nestedtopobrain_v3.py:216` | `def forward(self, metrics_dict)` |
| `forward` | method | `nestedtopobrain_v3.py:514` | `def forward(self, features, labels)` |
| `forward` | method | `nestedtopobrain_v3.py:538` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `nestedtopobrain_v3.py:558` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `nestedtopobrain_v3.py:585` | `def forward(self, x)` |
| `forward` | method | `nestedtopobrain_v3.py:660` | `def forward(self, x, state_M, controls)` |
| `forward` | method | `nestedtopobrain_v3.py:804` | `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)` |
| `forward` | method | `nestedtopobrain_v3.py:1177` | `def forward(self, x, prev_states, controls)` |
| `get_critical_summary` | method | `nestedtopobrain_v3.py:409` | `def get_critical_summary(self)` |
| `get_dataloaders` | method | `nestedtopobrain_v3.py:471` | `def get_dataloaders(config)` |
| `get_gpu_memory_gb` | method | `nestedtopobrain_v3.py:134` | `def get_gpu_memory_gb()` |
| `get_memory_gb` | method | `nestedtopobrain_v3.py:129` | `def get_memory_gb()` |
| `get_node_importance` | method | `nestedtopobrain_v3.py:798` | `def get_node_importance(self)` |
| `get_sparsity_lambda` | method | `nestedtopobrain_v3.py:97` | `def get_sparsity_lambda(self, epoch)` |
| `get_supcon_lambda` | method | `nestedtopobrain_v3.py:91` | `def get_supcon_lambda(self, epoch)` |
| `get_topology` | method | `nestedtopobrain_v3.py:1164` | `def get_topology(self, return_sparse)` |
| `initialize_memories` | method | `nestedtopobrain_v3.py:989` | `def initialize_memories(self, dataloader)` |
| `invalidate_sparse_cache` | method | `nestedtopobrain_v3.py:776` | `def invalidate_sparse_cache(self)` |
| `load` | method | `nestedtopobrain_v3.py:454` | `def load(self, name)` |
| `log` | method | `nestedtopobrain_v3.py:140` | `def log(prefix)` |
| `main` | method | `nestedtopobrain_v3.py:2491` | `def main()` |
| `make_adversarial_pgd` | method | `nestedtopobrain_v3.py:1954` | `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)` |
| `prune_topology` | method | `nestedtopobrain_v3.py:1239` | `def prune_topology(self, controls)` |
| `reset_context` | method | `nestedtopobrain_v3.py:317` | `def reset_context(self)` |
| `run_ablation_study` | method | `nestedtopobrain_v3.py:1695` | `def run_ablation_study()` |
| `save` | method | `nestedtopobrain_v3.py:426` | `def save(self, data, name)` |
| `save_node_importance_viz` | method | `nestedtopobrain_v3.py:1432` | `def save_node_importance_viz(model, epoch, run_name)` |
| `save_topology_visualization` | method | `nestedtopobrain_v3.py:1388` | `def save_topology_visualization(model, epoch, run_name)` |
| `seed_everything` | method | `nestedtopobrain_v3.py:113` | `def seed_everything(seed)` |
| `set_epoch` | method | `nestedtopobrain_v3.py:1103` | `def set_epoch(self, epoch)` |
| `to_dict` | method | `nestedtopobrain_v3.py:88` | `def to_dict(self)` |
| `train_epoch` | method | `nestedtopobrain_v3.py:2093` | `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)` |
| `train_model` | method | `nestedtopobrain_v3.py:2279` | `def train_model(config, run_name)` |
| `visualize_memory_evolution` | method | `nestedtopobrain_v3.py:1796` | `def visualize_memory_evolution(model, epoch, run_name)` |
| `visualize_topology_as_graph` | method | `nestedtopobrain_v3.py:1559` | `def visualize_topology_as_graph(model, run_name, threshold)` |
| `warmup_topo` | method | `nestedtopobrain_v3.py:2327` | `def warmup_topo(epoch)` |
| `AblationMatrix` | class | `neurologitos.py:356` | `class AblationMatrix` |
| `BioDecoder` | class | `neurologitos.py:241` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `neurologitos.py:318` | `class CIFARCaptions` |
| `ConsciousCore` | class | `neurologitos.py:229` | `class ConsciousCore(Module)` |
| `MiniUnconscious` | class | `neurologitos.py:187` | `class MiniUnconscious(Module)` |
| `NeuroLogos` | class | `neurologitos.py:284` | `class NeuroLogos(Module)` |
| `NeuroLogosConfig` | class | `neurologitos.py:50` | `class NeuroLogosConfig` |
| `PGDAttack` | class | `neurologitos.py:159` | `class PGDAttack` |
| `ScientificAnalyzer` | class | `neurologitos.py:393` | `class ScientificAnalyzer` |
| `TopoBrainCore` | class | `neurologitos.py:95` | `class TopoBrainCore(Module)` |
| `TopoUnconscious` | class | `neurologitos.py:206` | `class TopoUnconscious(Module)` |
| `__getitem__` | method | `neurologitos.py:347` | `def __getitem__(self, idx)` |
| `__init__` | method | `neurologitos.py:96` | `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)` |
| `__init__` | method | `neurologitos.py:160` | `def __init__(self, epsilon, alpha, steps)` |
| `__init__` | method | `neurologitos.py:188` | `def __init__(self, output_dim)` |
| `__init__` | method | `neurologitos.py:207` | `def __init__(self, output_dim, use_grid, use_symbiotic)` |
| `__init__` | method | `neurologitos.py:230` | `def __init__(self, dim)` |
| `__init__` | method | `neurologitos.py:242` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `neurologitos.py:285` | `def __init__(self, vocab_size, config)` |
| `__init__` | method | `neurologitos.py:319` | `def __init__(self)` |
| `__len__` | method | `neurologitos.py:344` | `def __len__(self)` |
| `_get_init_state` | method | `neurologitos.py:278` | `def _get_init_state(self, thought)` |
| `_init_grid` | method | `neurologitos.py:121` | `def _init_grid(self)` |
| `attack` | method | `neurologitos.py:165` | `def attack(self, model_fn, x, y, criterion)` |
| `component_signature` | method | `neurologitos.py:84` | `def component_signature(self)` |
| `compute_effect_size` | function | `neurologitos.py:38` | `def compute_effect_size(group1, group2)` |
| `compute_statistics` | method | `neurologitos.py:395` | `def compute_statistics(cv_results)` |
| `crit_fn` | method | `neurologitos.py:474` | `def crit_fn(out, tgt)` |
| `detect_synergy` | method | `neurologitos.py:419` | `def detect_synergy(pair_score, comp_a_score, comp_b_score, baseline_score)` |
| `evaluate_cv` | method | `neurologitos.py:507` | `def evaluate_cv(model, loader, config, vocab)` |
| `forward` | method | `neurologitos.py:128` | `def forward(self, x)` |
| `forward` | method | `neurologitos.py:200` | `def forward(self, x)` |
| `forward` | method | `neurologitos.py:221` | `def forward(self, x)` |
| `forward` | method | `neurologitos.py:235` | `def forward(self, x)` |
| `forward` | method | `neurologitos.py:253` | `def forward(self, thought, captions, max_len)` |
| `forward` | method | `neurologitos.py:309` | `def forward(self, image, captions)` |
| `get_complete_matrix` | method | `neurologitos.py:389` | `def get_complete_matrix(cls)` |
| `get_metrics` | method | `neurologitos.py:152` | `def get_metrics(self)` |
| `get_metrics` | method | `neurologitos.py:225` | `def get_metrics(self)` |
| `get_metrics` | method | `neurologitos.py:314` | `def get_metrics(self)` |
| `level1_isolated` | method | `neurologitos.py:360` | `def level1_isolated()` |
| `level2_pairs` | method | `neurologitos.py:369` | `def level2_pairs()` |
| `level3_full` | method | `neurologitos.py:377` | `def level3_full()` |
| `level4_inverse` | method | `neurologitos.py:381` | `def level4_inverse()` |
| `model_fn` | method | `neurologitos.py:471` | `def model_fn(x_adv)` |
| `rank_criticality` | method | `neurologitos.py:432` | `def rank_criticality(full_score, ablation_results)` |
| `run_scientific_ablation` | method | `neurologitos.py:579` | `def run_scientific_ablation()` |
| `seed_everything` | function | `neurologitos.py:29` | `def seed_everything(seed)` |
| `to_dict` | method | `neurologitos.py:81` | `def to_dict(self)` |
| `train_epoch_cv` | method | `neurologitos.py:451` | `def train_epoch_cv(model, loader, optimizer, config, epoch, vocab)` |
| `train_with_cv` | method | `neurologitos.py:524` | `def train_with_cv(config, dataset, vocab)` |
| `ttest_vs_baseline` | method | `neurologitos.py:413` | `def ttest_vs_baseline(exp_scores, baseline_scores)` |
| `BioDecoder` | class | `neurologos.py:116` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `neurologos.py:220` | `class CIFARCaptions` |
| `ConsciousCore` | class | `neurologos.py:96` | `class ConsciousCore(Module)` |
| `LifeCycle` | class | `neurologos.py:200` | `class LifeCycle` |
| `LiquidNeuron` | class | `neurologos.py:81` | `class LiquidNeuron(Module)` |
| `MiniUnconscious` | class | `neurologos.py:13` | `class MiniUnconscious(Module)` |
| `NestedUnconscious` | class | `neurologos.py:29` | `class NestedUnconscious(Module)` |
| `NeuroLogos` | class | `neurologos.py:167` | `class NeuroLogos(Module)` |
| `__getitem__` | method | `neurologos.py:248` | `def __getitem__(self, idx)` |
| `__init__` | method | `neurologos.py:15` | `def __init__(self)` |
| `__init__` | method | `neurologos.py:31` | `def __init__(self, grid_size, output_dim)` |
| `__init__` | method | `neurologos.py:82` | `def __init__(self, dim)` |
| `__init__` | method | `neurologos.py:97` | `def __init__(self)` |
| `__init__` | method | `neurologos.py:117` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `neurologos.py:168` | `def __init__(self, vocab_size, use_nested)` |
| `__init__` | method | `neurologos.py:201` | `def __init__(self, total_epochs)` |
| `__init__` | method | `neurologos.py:221` | `def __init__(self)` |
| `__len__` | method | `neurologos.py:245` | `def __len__(self)` |
| `_get_init_state` | method | `neurologos.py:159` | `def _get_init_state(self, thought)` |
| `forward` | method | `neurologos.py:26` | `def forward(self, x)` |
| `forward` | method | `neurologos.py:57` | `def forward(self, x)` |
| `forward` | method | `neurologos.py:88` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurologos.py:103` | `def forward(self, visual_features, plasticity)` |
| `forward` | method | `neurologos.py:132` | `def forward(self, thought, captions, max_len)` |
| `forward` | method | `neurologos.py:182` | `def forward(self, image, captions, plasticity)` |
| `get_plasticity` | method | `neurologos.py:205` | `def get_plasticity(self, epoch)` |
| `measure_richness` | method | `neurologos.py:193` | `def measure_richness(self)` |
| `train_logos` | method | `neurologos.py:261` | `def train_logos(use_nested)` |
| `BioDecoder` | class | `neurologos_V1.py:73` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `neurologos_V1.py:189` | `class CIFARCaptions` |
| `ConsciousCore` | class | `neurologos_V1.py:48` | `class ConsciousCore(Module)` |
| `LifeCycle` | class | `neurologos_V1.py:170` | `class LifeCycle` |
| `LiquidNeuron` | class | `neurologos_V1.py:33` | `class LiquidNeuron(Module)` |
| `MiniUnconscious` | class | `neurologos_V1.py:15` | `class MiniUnconscious(Module)` |
| `NeuroLogos` | class | `neurologos_V1.py:140` | `class NeuroLogos(Module)` |
| `__getitem__` | method | `neurologos_V1.py:219` | `def __getitem__(self, idx)` |
| `__init__` | method | `neurologos_V1.py:16` | `def __init__(self)` |
| `__init__` | method | `neurologos_V1.py:34` | `def __init__(self, dim)` |
| `__init__` | method | `neurologos_V1.py:49` | `def __init__(self)` |
| `__init__` | method | `neurologos_V1.py:74` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `neurologos_V1.py:141` | `def __init__(self, vocab_size)` |
| `__init__` | method | `neurologos_V1.py:171` | `def __init__(self, total_epochs)` |
| `__init__` | method | `neurologos_V1.py:190` | `def __init__(self)` |
| `__len__` | method | `neurologos_V1.py:216` | `def __len__(self)` |
| `_get_init_state` | method | `neurologos_V1.py:131` | `def _get_init_state(self, thought)` |
| `forward` | method | `neurologos_V1.py:27` | `def forward(self, x)` |
| `forward` | method | `neurologos_V1.py:40` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurologos_V1.py:55` | `def forward(self, visual_features, plasticity)` |
| `forward` | method | `neurologos_V1.py:90` | `def forward(self, thought, captions, max_len, teacher_forcing_ratio)` |
| `forward` | method | `neurologos_V1.py:150` | `def forward(self, image, captions, plasticity)` |
| `get_plasticity` | method | `neurologos_V1.py:175` | `def get_plasticity(self, epoch)` |
| `measure_richness` | method | `neurologos_V1.py:163` | `def measure_richness(self)` |
| `train_logos` | method | `neurologos_V1.py:237` | `def train_logos()` |
| `MicroConfig` | class | `neurologos_cpu_v7.py:33` | `class MicroConfig` |
| `MicroContinuumCell` | class | `neurologos_cpu_v7.py:96` | `class MicroContinuumCell(Module)` |
| `MicroSupConLoss` | class | `neurologos_cpu_v7.py:154` | `class MicroSupConLoss(Module)` |
| `MicroSymbioticBasis` | class | `neurologos_cpu_v7.py:120` | `class MicroSymbioticBasis(Module)` |
| `MicroTopoBrain` | class | `neurologos_cpu_v7.py:170` | `class MicroTopoBrain(Module)` |
| `MicroTopology` | class | `neurologos_cpu_v7.py:135` | `class MicroTopology` |
| `__init__` | method | `neurologos_cpu_v7.py:97` | `def __init__(self, dim)` |
| `__init__` | method | `neurologos_cpu_v7.py:121` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `neurologos_cpu_v7.py:136` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `neurologos_cpu_v7.py:155` | `def __init__(self, temperature)` |
| `__init__` | method | `neurologos_cpu_v7.py:171` | `def __init__(self, config)` |
| `_init_weights` | method | `neurologos_cpu_v7.py:195` | `def _init_weights(self)` |
| `count_parameters` | method | `neurologos_cpu_v7.py:198` | `def count_parameters(self)` |
| `forward` | method | `neurologos_cpu_v7.py:106` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurologos_cpu_v7.py:128` | `def forward(self, x)` |
| `forward` | method | `neurologos_cpu_v7.py:158` | `def forward(self, features, labels)` |
| `forward` | method | `neurologos_cpu_v7.py:199` | `def forward(self, x, plasticity)` |
| `get_adjacency` | method | `neurologos_cpu_v7.py:149` | `def get_adjacency(self, plasticity)` |
| `get_dataset` | method | `neurologos_cpu_v7.py:73` | `def get_dataset(config)` |
| `micro_pgd_attack` | method | `neurologos_cpu_v7.py:235` | `def micro_pgd_attack(model, x, y, eps, steps, plasticity)` |
| `run_ablation_study` | method | `neurologos_cpu_v7.py:328` | `def run_ablation_study()` |
| `seed_everything` | method | `neurologos_cpu_v7.py:68` | `def seed_everything(seed)` |
| `setup_device` | method | `neurologos_cpu_v7.py:65` | `def setup_device()` |
| `train_with_cv` | method | `neurologos_cpu_v7.py:276` | `def train_with_cv(config, dataset, cv_folds)` |
| `MicroConfig` | class | `neurologos_cpu_v8.py:33` | `class MicroConfig` |
| `MicroContinuumCell` | class | `neurologos_cpu_v8.py:92` | `class MicroContinuumCell(Module)` |
| `MicroSupConLoss` | class | `neurologos_cpu_v8.py:169` | `class MicroSupConLoss(Module)` |
| `MicroSymbioticBasis` | class | `neurologos_cpu_v8.py:124` | `class MicroSymbioticBasis(Module)` |
| `MicroTopoBrain` | class | `neurologos_cpu_v8.py:195` | `class MicroTopoBrain(Module)` |
| `MicroTopology` | class | `neurologos_cpu_v8.py:146` | `class MicroTopology` |
| `__init__` | method | `neurologos_cpu_v8.py:94` | `def __init__(self, dim)` |
| `__init__` | method | `neurologos_cpu_v8.py:126` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `neurologos_cpu_v8.py:148` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `neurologos_cpu_v8.py:171` | `def __init__(self, temperature)` |
| `__init__` | method | `neurologos_cpu_v8.py:197` | `def __init__(self, config)` |
| `_init_weights` | method | `neurologos_cpu_v8.py:240` | `def _init_weights(self)` |
| `count_parameters` | method | `neurologos_cpu_v8.py:245` | `def count_parameters(self)` |
| `forward` | method | `neurologos_cpu_v8.py:105` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurologos_cpu_v8.py:134` | `def forward(self, x)` |
| `forward` | method | `neurologos_cpu_v8.py:176` | `def forward(self, features, labels)` |
| `forward` | method | `neurologos_cpu_v8.py:248` | `def forward(self, x, plasticity)` |
| `generate_ablation_matrix` | method | `neurologos_cpu_v8.py:344` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `neurologos_cpu_v8.py:164` | `def get_adjacency(self, plasticity)` |
| `get_dataset` | method | `neurologos_cpu_v8.py:72` | `def get_dataset(config)` |
| `micro_pgd_attack` | method | `neurologos_cpu_v8.py:308` | `def micro_pgd_attack(model, x, y, eps, steps, plasticity)` |
| `run_ablation_study` | method | `neurologos_cpu_v8.py:483` | `def run_ablation_study()` |
| `seed_everything` | method | `neurologos_cpu_v8.py:65` | `def seed_everything(seed)` |
| `train_with_cv` | method | `neurologos_cpu_v8.py:393` | `def train_with_cv(config, dataset, cv_folds)` |
| `AblationMatrix` | class | `neurologos_cpu_v9.py.py:355` | `class AblationMatrix` |
| `BioDecoder` | class | `neurologos_cpu_v9.py.py:231` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `neurologos_cpu_v9.py.py:315` | `class CIFARCaptions` |
| `ConsciousCore` | class | `neurologos_cpu_v9.py.py:220` | `class ConsciousCore(Module)` |
| `MiniUnconscious` | class | `neurologos_cpu_v9.py.py:180` | `class MiniUnconscious(Module)` |
| `NeuroLogos` | class | `neurologos_cpu_v9.py.py:276` | `class NeuroLogos(Module)` |
| `NeuroLogosConfig` | class | `neurologos_cpu_v9.py.py:45` | `class NeuroLogosConfig` |
| `PGDAttack` | class | `neurologos_cpu_v9.py.py:153` | `class PGDAttack` |
| `ScientificAnalyzer` | class | `neurologos_cpu_v9.py.py:392` | `class ScientificAnalyzer` |
| `TopoBrainCore` | class | `neurologos_cpu_v9.py.py:90` | `class TopoBrainCore(Module)` |
| `TopoUnconscious` | class | `neurologos_cpu_v9.py.py:198` | `class TopoUnconscious(Module)` |
| `__getitem__` | method | `neurologos_cpu_v9.py.py:344` | `def __getitem__(self, idx)` |
| `__init__` | method | `neurologos_cpu_v9.py.py:91` | `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)` |
| `__init__` | method | `neurologos_cpu_v9.py.py:154` | `def __init__(self, epsilon, alpha, steps)` |
| `__init__` | method | `neurologos_cpu_v9.py.py:181` | `def __init__(self, output_dim)` |
| `__init__` | method | `neurologos_cpu_v9.py.py:199` | `def __init__(self, output_dim, use_grid, use_symbiotic)` |
| `__init__` | method | `neurologos_cpu_v9.py.py:221` | `def __init__(self, dim)` |
| `__init__` | method | `neurologos_cpu_v9.py.py:232` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `neurologos_cpu_v9.py.py:277` | `def __init__(self, vocab_size, config)` |
| `__init__` | method | `neurologos_cpu_v9.py.py:316` | `def __init__(self)` |
| `__len__` | method | `neurologos_cpu_v9.py.py:341` | `def __len__(self)` |
| `_get_init_state` | method | `neurologos_cpu_v9.py.py:268` | `def _get_init_state(self, thought)` |
| `_init_grid` | method | `neurologos_cpu_v9.py.py:116` | `def _init_grid(self)` |
| `attack` | method | `neurologos_cpu_v9.py.py:159` | `def attack(self, model_fn, x, y, criterion)` |
| `component_signature` | method | `neurologos_cpu_v9.py.py:79` | `def component_signature(self)` |
| `compute_effect_size` | function | `neurologos_cpu_v9.py.py:34` | `def compute_effect_size(group1, group2)` |
| `compute_statistics` | method | `neurologos_cpu_v9.py.py:394` | `def compute_statistics(cv_results)` |
| `crit_fn` | method | `neurologos_cpu_v9.py.py:475` | `def crit_fn(out, tgt)` |
| `detect_synergy` | method | `neurologos_cpu_v9.py.py:418` | `def detect_synergy(pair_score, comp_a_score, comp_b_score, baseline_score)` |
| `evaluate_cv` | method | `neurologos_cpu_v9.py.py:507` | `def evaluate_cv(model, loader, config, vocab)` |
| `forward` | method | `neurologos_cpu_v9.py.py:123` | `def forward(self, x)` |
| `forward` | method | `neurologos_cpu_v9.py.py:193` | `def forward(self, x)` |
| `forward` | method | `neurologos_cpu_v9.py.py:213` | `def forward(self, x)` |
| `forward` | method | `neurologos_cpu_v9.py.py:226` | `def forward(self, x)` |
| `forward` | method | `neurologos_cpu_v9.py.py:243` | `def forward(self, thought, captions, max_len)` |
| `forward` | method | `neurologos_cpu_v9.py.py:304` | `def forward(self, image, captions)` |
| `get_complete_matrix` | method | `neurologos_cpu_v9.py.py:389` | `def get_complete_matrix(cls)` |
| `get_metrics` | method | `neurologos_cpu_v9.py.py:147` | `def get_metrics(self)` |
| `get_metrics` | method | `neurologos_cpu_v9.py.py:217` | `def get_metrics(self)` |
| `get_metrics` | method | `neurologos_cpu_v9.py.py:309` | `def get_metrics(self)` |
| `level1_isolated` | method | `neurologos_cpu_v9.py.py:360` | `def level1_isolated()` |
| `level2_pairs` | method | `neurologos_cpu_v9.py.py:369` | `def level2_pairs()` |
| `level3_full` | method | `neurologos_cpu_v9.py.py:377` | `def level3_full()` |
| `level4_inverse` | method | `neurologos_cpu_v9.py.py:381` | `def level4_inverse()` |
| `model_fn` | method | `neurologos_cpu_v9.py.py:472` | `def model_fn(x_adv)` |
| `rank_criticality` | method | `neurologos_cpu_v9.py.py:431` | `def rank_criticality(full_score, ablation_results)` |
| `run_scientific_ablation` | method | `neurologos_cpu_v9.py.py:580` | `def run_scientific_ablation()` |
| `seed_everything` | function | `neurologos_cpu_v9.py.py:25` | `def seed_everything(seed)` |
| `to_dict` | method | `neurologos_cpu_v9.py.py:76` | `def to_dict(self)` |
| `train_epoch_cv` | method | `neurologos_cpu_v9.py.py:452` | `def train_epoch_cv(model, loader, optimizer, config, epoch, vocab)` |
| `train_with_cv` | method | `neurologos_cpu_v9.py.py:523` | `def train_with_cv(config, dataset, vocab)` |
| `ttest_vs_baseline` | method | `neurologos_cpu_v9.py.py:412` | `def ttest_vs_baseline(exp_scores, baseline_scores)` |
| `Config` | class | `neurologos_entropico.py:36` | `class Config` |
| `DataEnvironment` | class | `neurologos_entropico.py:60` | `class DataEnvironment` |
| `HomeostaticRegulator` | class | `neurologos_entropico.py:94` | `class HomeostaticRegulator(Module)` |
| `MicroTopoBrain` | class | `neurologos_entropico.py:201` | `class MicroTopoBrain(Module)` |
| `PhysioNeuron` | class | `neurologos_entropico.py:120` | `class PhysioNeuron(Module)` |
| `RegulableSymbiotic` | class | `neurologos_entropico.py:160` | `class RegulableSymbiotic(Module)` |
| `RegulableTopology` | class | `neurologos_entropico.py:179` | `class RegulableTopology` |
| `__init__` | method | `neurologos_entropico.py:61` | `def __init__(self)` |
| `__init__` | method | `neurologos_entropico.py:95` | `def __init__(self, d_in)` |
| `__init__` | method | `neurologos_entropico.py:121` | `def __init__(self, d_in, d_out, dynamic)` |
| `__init__` | method | `neurologos_entropico.py:161` | `def __init__(self, dim, atoms)` |
| `__init__` | method | `neurologos_entropico.py:180` | `def __init__(self, num_nodes)` |
| `__init__` | method | `neurologos_entropico.py:202` | `def __init__(self, config)` |
| `count_parameters` | method | `neurologos_entropico.py:221` | `def count_parameters(self)` |
| `forward` | method | `neurologos_entropico.py:105` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `neurologos_entropico.py:132` | `def forward(self, x)` |
| `forward` | method | `neurologos_entropico.py:168` | `def forward(self, x, influence)` |
| `forward` | method | `neurologos_entropico.py:224` | `def forward(self, x)` |
| `generate_ablation_matrix` | method | `neurologos_entropico.py:320` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `neurologos_entropico.py:193` | `def get_adjacency(self, plasticity)` |
| `get_batch` | method | `neurologos_entropico.py:71` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `neurologos_entropico.py:85` | `def get_full(self)` |
| `get_w2` | method | `neurologos_entropico.py:88` | `def get_w2(self)` |
| `run_ablation_study` | method | `neurologos_entropico.py:352` | `def run_ablation_study()` |
| `seed_everything` | method | `neurologos_entropico.py:50` | `def seed_everything(seed)` |
| `train_nonstationary` | method | `neurologos_entropico.py:263` | `def train_nonstationary(config)` |
| `GlobalHomeostaticOrchestrator` | class | `neurologos_fullhomesotatico_cpu_qw.py:93` | `class GlobalHomeostaticOrchestrator(Module)` |
| `MicroConfig` | class | `neurologos_fullhomesotatico_cpu_qw.py:36` | `class MicroConfig` |
| `MicroContinuumCell` | class | `neurologos_fullhomesotatico_cpu_qw.py:137` | `class MicroContinuumCell(Module)` |
| `MicroSupConLoss` | class | `neurologos_fullhomesotatico_cpu_qw.py:203` | `class MicroSupConLoss(Module)` |
| `MicroSymbioticBasis` | class | `neurologos_fullhomesotatico_cpu_qw.py:161` | `class MicroSymbioticBasis(Module)` |
| `MicroTopoBrain` | class | `neurologos_fullhomesotatico_cpu_qw.py:227` | `class MicroTopoBrain(Module)` |
| `MicroTopology` | class | `neurologos_fullhomesotatico_cpu_qw.py:183` | `class MicroTopology` |
| `__init__` | method | `neurologos_fullhomesotatico_cpu_qw.py:94` | `def __init__(self)` |
| `__init__` | method | `neurologos_fullhomesotatico_cpu_qw.py:138` | `def __init__(self, dim)` |
| `__init__` | method | `neurologos_fullhomesotatico_cpu_qw.py:162` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `neurologos_fullhomesotatico_cpu_qw.py:184` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `neurologos_fullhomesotatico_cpu_qw.py:204` | `def __init__(self, temperature)` |
| `__init__` | method | `neurologos_fullhomesotatico_cpu_qw.py:228` | `def __init__(self, config)` |
| `_init_weights` | method | `neurologos_fullhomesotatico_cpu_qw.py:267` | `def _init_weights(self)` |
| `count_parameters` | method | `neurologos_fullhomesotatico_cpu_qw.py:272` | `def count_parameters(self)` |
| `forward` | method | `neurologos_fullhomesotatico_cpu_qw.py:104` | `def forward(self, x, logits, h_agg, h_proc, w_norm, entropy, ortho)` |
| `forward` | method | `neurologos_fullhomesotatico_cpu_qw.py:148` | `def forward(self, x, strength)` |
| `forward` | method | `neurologos_fullhomesotatico_cpu_qw.py:170` | `def forward(self, x, influence)` |
| `forward` | method | `neurologos_fullhomesotatico_cpu_qw.py:209` | `def forward(self, features, labels)` |
| `forward` | method | `neurologos_fullhomesotatico_cpu_qw.py:275` | `def forward(self, x)` |
| `generate_ablation_matrix` | method | `neurologos_fullhomesotatico_cpu_qw.py:387` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `neurologos_fullhomesotatico_cpu_qw.py:197` | `def get_adjacency(self, plasticity)` |
| `get_dataset` | method | `neurologos_fullhomesotatico_cpu_qw.py:73` | `def get_dataset(config)` |
| `micro_pgd_attack` | method | `neurologos_fullhomesotatico_cpu_qw.py:363` | `def micro_pgd_attack(model, x, y, eps, steps, plasticity_ctrl)` |
| `run_ablation_study` | method | `neurologos_fullhomesotatico_cpu_qw.py:472` | `def run_ablation_study()` |
| `seed_everything` | method | `neurologos_fullhomesotatico_cpu_qw.py:65` | `def seed_everything(seed)` |
| `train_with_cv` | method | `neurologos_fullhomesotatico_cpu_qw.py:407` | `def train_with_cv(config, dataset, cv_folds)` |
| `Config` | class | `neurologos_fullhomestatico_cpu_qw2.py:34` | `class Config` |
| `DataEnvironment` | class | `neurologos_fullhomestatico_cpu_qw2.py:60` | `class DataEnvironment` |
| `HomeostaticRegulator` | class | `neurologos_fullhomestatico_cpu_qw2.py:95` | `class HomeostaticRegulator(Module)` |
| `MicroTopoBrain` | class | `neurologos_fullhomestatico_cpu_qw2.py:207` | `class MicroTopoBrain(Module)` |
| `PhysioNeuron` | class | `neurologos_fullhomestatico_cpu_qw2.py:122` | `class PhysioNeuron(Module)` |
| `RegulableSymbiotic` | class | `neurologos_fullhomestatico_cpu_qw2.py:164` | `class RegulableSymbiotic(Module)` |
| `RegulableTopology` | class | `neurologos_fullhomestatico_cpu_qw2.py:184` | `class RegulableTopology` |
| `__init__` | method | `neurologos_fullhomestatico_cpu_qw2.py:61` | `def __init__(self)` |
| `__init__` | method | `neurologos_fullhomestatico_cpu_qw2.py:96` | `def __init__(self, d_in)` |
| `__init__` | method | `neurologos_fullhomestatico_cpu_qw2.py:123` | `def __init__(self, d_in, d_out, dynamic)` |
| `__init__` | method | `neurologos_fullhomestatico_cpu_qw2.py:165` | `def __init__(self, dim, atoms)` |
| `__init__` | method | `neurologos_fullhomestatico_cpu_qw2.py:185` | `def __init__(self, num_nodes)` |
| `__init__` | method | `neurologos_fullhomestatico_cpu_qw2.py:208` | `def __init__(self, config)` |
| `count_parameters` | method | `neurologos_fullhomestatico_cpu_qw2.py:227` | `def count_parameters(self)` |
| `forward` | method | `neurologos_fullhomestatico_cpu_qw2.py:106` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `neurologos_fullhomestatico_cpu_qw2.py:134` | `def forward(self, x)` |
| `forward` | method | `neurologos_fullhomestatico_cpu_qw2.py:172` | `def forward(self, x, influence)` |
| `forward` | method | `neurologos_fullhomestatico_cpu_qw2.py:230` | `def forward(self, x)` |
| `generate_ablation_matrix` | method | `neurologos_fullhomestatico_cpu_qw2.py:331` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `neurologos_fullhomestatico_cpu_qw2.py:198` | `def get_adjacency(self, plasticity)` |
| `get_batch` | method | `neurologos_fullhomestatico_cpu_qw2.py:71` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `neurologos_fullhomestatico_cpu_qw2.py:85` | `def get_full(self)` |
| `get_w2` | method | `neurologos_fullhomestatico_cpu_qw2.py:88` | `def get_w2(self)` |
| `run_ablation_study` | method | `neurologos_fullhomestatico_cpu_qw2.py:364` | `def run_ablation_study()` |
| `seed_everything` | method | `neurologos_fullhomestatico_cpu_qw2.py:49` | `def seed_everything(seed)` |
| `train_nonstationary` | method | `neurologos_fullhomestatico_cpu_qw2.py:269` | `def train_nonstationary(config)` |
| `BioDecoder` | class | `neurologos_gpu_v1.py:116` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `neurologos_gpu_v1.py:233` | `class CIFARCaptions` |
| `ConsciousCore` | class | `neurologos_gpu_v1.py:91` | `class ConsciousCore(Module)` |
| `LifeCycle` | class | `neurologos_gpu_v1.py:213` | `class LifeCycle` |
| `LiquidNeuron` | class | `neurologos_gpu_v1.py:76` | `class LiquidNeuron(Module)` |
| `NestedUnconscious` | class | `neurologos_gpu_v1.py:16` | `class NestedUnconscious(Module)` |
| `NeuroLogos` | class | `neurologos_gpu_v1.py:183` | `class NeuroLogos(Module)` |
| `__getitem__` | method | `neurologos_gpu_v1.py:264` | `def __getitem__(self, idx)` |
| `__init__` | method | `neurologos_gpu_v1.py:17` | `def __init__(self, grid_size, hidden_dim)` |
| `__init__` | method | `neurologos_gpu_v1.py:77` | `def __init__(self, dim)` |
| `__init__` | method | `neurologos_gpu_v1.py:92` | `def __init__(self)` |
| `__init__` | method | `neurologos_gpu_v1.py:117` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `neurologos_gpu_v1.py:184` | `def __init__(self, vocab_size)` |
| `__init__` | method | `neurologos_gpu_v1.py:214` | `def __init__(self, total_epochs)` |
| `__init__` | method | `neurologos_gpu_v1.py:234` | `def __init__(self)` |
| `__len__` | method | `neurologos_gpu_v1.py:261` | `def __len__(self)` |
| `_get_init_state` | method | `neurologos_gpu_v1.py:174` | `def _get_init_state(self, thought)` |
| `forward` | method | `neurologos_gpu_v1.py:46` | `def forward(self, x)` |
| `forward` | method | `neurologos_gpu_v1.py:83` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurologos_gpu_v1.py:98` | `def forward(self, visual_features, plasticity)` |
| `forward` | method | `neurologos_gpu_v1.py:133` | `def forward(self, thought, captions, max_len, teacher_forcing_ratio)` |
| `forward` | method | `neurologos_gpu_v1.py:193` | `def forward(self, image, captions, plasticity)` |
| `get_plasticity` | method | `neurologos_gpu_v1.py:219` | `def get_plasticity(self, epoch)` |
| `measure_richness` | method | `neurologos_gpu_v1.py:206` | `def measure_richness(self)` |
| `train_logos` | method | `neurologos_gpu_v1.py:282` | `def train_logos()` |
| `HomeoConfig` | class | `neurologos_homeostatico_cpu_cl.py:38` | `class HomeoConfig` |
| `HomeoContinuumCell` | class | `neurologos_homeostatico_cpu_cl.py:218` | `class HomeoContinuumCell(Module)` |
| `HomeoSupConLoss` | class | `neurologos_homeostatico_cpu_cl.py:380` | `class HomeoSupConLoss(Module)` |
| `HomeoSymbioticBasis` | class | `neurologos_homeostatico_cpu_cl.py:281` | `class HomeoSymbioticBasis(Module)` |
| `HomeoTopoBrain` | class | `neurologos_homeostatico_cpu_cl.py:410` | `class HomeoTopoBrain(Module)` |
| `HomeoTopology` | class | `neurologos_homeostatico_cpu_cl.py:329` | `class HomeoTopology` |
| `HomeostaticRegulator` | class | `neurologos_homeostatico_cpu_cl.py:98` | `class HomeostaticRegulator(Module)` |
| `__init__` | method | `neurologos_homeostatico_cpu_cl.py:110` | `def __init__(self, d_in)` |
| `__init__` | method | `neurologos_homeostatico_cpu_cl.py:220` | `def __init__(self, dim, use_homeostasis)` |
| `__init__` | method | `neurologos_homeostatico_cpu_cl.py:283` | `def __init__(self, dim, num_atoms, use_homeostasis)` |
| `__init__` | method | `neurologos_homeostatico_cpu_cl.py:331` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `neurologos_homeostatico_cpu_cl.py:382` | `def __init__(self, temperature)` |
| `__init__` | method | `neurologos_homeostatico_cpu_cl.py:412` | `def __init__(self, config)` |
| `_init_weights` | method | `neurologos_homeostatico_cpu_cl.py:456` | `def _init_weights(self)` |
| `count_parameters` | method | `neurologos_homeostatico_cpu_cl.py:461` | `def count_parameters(self)` |
| `forward` | method | `neurologos_homeostatico_cpu_cl.py:129` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `neurologos_homeostatico_cpu_cl.py:241` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurologos_homeostatico_cpu_cl.py:299` | `def forward(self, x)` |
| `forward` | method | `neurologos_homeostatico_cpu_cl.py:387` | `def forward(self, features, labels)` |
| `forward` | method | `neurologos_homeostatico_cpu_cl.py:464` | `def forward(self, x, plasticity)` |
| `generate_homeostatic_ablation` | method | `neurologos_homeostatico_cpu_cl.py:555` | `def generate_homeostatic_ablation()` |
| `get_adjacency` | method | `neurologos_homeostatico_cpu_cl.py:354` | `def get_adjacency(self, x, plasticity)` |
| `get_dataset` | method | `neurologos_homeostatico_cpu_cl.py:78` | `def get_dataset(config)` |
| `pgd_attack` | method | `neurologos_homeostatico_cpu_cl.py:524` | `def pgd_attack(model, x, y, eps, steps, plasticity)` |
| `run_homeostatic_ablation` | method | `neurologos_homeostatico_cpu_cl.py:697` | `def run_homeostatic_ablation()` |
| `seed_everything` | method | `neurologos_homeostatico_cpu_cl.py:71` | `def seed_everything(seed)` |
| `train_with_cv` | method | `neurologos_homeostatico_cpu_cl.py:614` | `def train_with_cv(config, dataset, cv_folds)` |
| `EnhancedHomeostaticRegulator` | class | `neurologos_homeostatico_cpu_cl2.py:152` | `class EnhancedHomeostaticRegulator(Module)` |
| `NonStationaryEnvironment` | class | `neurologos_homeostatico_cpu_cl2.py:89` | `class NonStationaryEnvironment` |
| `TransContextConfig` | class | `neurologos_homeostatico_cpu_cl2.py:50` | `class TransContextConfig` |
| `TransContextContinuumCell` | class | `neurologos_homeostatico_cpu_cl2.py:254` | `class TransContextContinuumCell(Module)` |
| `TransContextTopoBrain` | class | `neurologos_homeostatico_cpu_cl2.py:320` | `class TransContextTopoBrain(Module)` |
| `__init__` | method | `neurologos_homeostatico_cpu_cl2.py:94` | `def __init__(self)` |
| `__init__` | method | `neurologos_homeostatico_cpu_cl2.py:157` | `def __init__(self, d_in, log_metrics)` |
| `__init__` | method | `neurologos_homeostatico_cpu_cl2.py:256` | `def __init__(self, dim, use_homeostasis, log_metrics)` |
| `__init__` | method | `neurologos_homeostatico_cpu_cl2.py:325` | `def __init__(self, config)` |
| `_init_weights` | method | `neurologos_homeostatico_cpu_cl2.py:356` | `def _init_weights(self)` |
| `count_parameters` | method | `neurologos_homeostatico_cpu_cl2.py:361` | `def count_parameters(self)` |
| `forward` | method | `neurologos_homeostatico_cpu_cl2.py:183` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `neurologos_homeostatico_cpu_cl2.py:277` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurologos_homeostatico_cpu_cl2.py:364` | `def forward(self, x, plasticity)` |
| `generate_selective_ablation` | method | `neurologos_homeostatico_cpu_cl2.py:556` | `def generate_selective_ablation()` |
| `get_batch` | method | `neurologos_homeostatico_cpu_cl2.py:115` | `def get_batch(self, phase, batch_size)` |
| `get_homeostasis_metrics` | method | `neurologos_homeostatico_cpu_cl2.py:401` | `def get_homeostasis_metrics(self)` |
| `get_phase` | method | `neurologos_homeostatico_cpu_cl2.py:135` | `def get_phase(self, epoch, total_epochs)` |
| `light_pgd_attack` | method | `neurologos_homeostatico_cpu_cl2.py:424` | `def light_pgd_attack(model, x, y, eps, steps)` |
| `run_trans_contextual_study` | method | `neurologos_homeostatico_cpu_cl2.py:598` | `def run_trans_contextual_study()` |
| `seed_everything` | method | `neurologos_homeostatico_cpu_cl2.py:78` | `def seed_everything(seed)` |
| `train_trans_contextual` | method | `neurologos_homeostatico_cpu_cl2.py:455` | `def train_trans_contextual(config, name)` |
| `HomeostaticCore` | class | `neurologos_homeostatico_cpu_ki.py:109` | `class HomeostaticCore(Module)` |
| `MicroConfig` | class | `neurologos_homeostatico_cpu_ki.py:37` | `class MicroConfig` |
| `MicroContinuumCell` | class | `neurologos_homeostatico_cpu_ki.py:175` | `class MicroContinuumCell(Module)` |
| `MicroSupConLoss` | class | `neurologos_homeostatico_cpu_ki.py:367` | `class MicroSupConLoss(Module)` |
| `MicroSymbioticBasis` | class | `neurologos_homeostatico_cpu_ki.py:240` | `class MicroSymbioticBasis(Module)` |
| `MicroTopoBrain` | class | `neurologos_homeostatico_cpu_ki.py:397` | `class MicroTopoBrain(Module)` |
| `MicroTopology` | class | `neurologos_homeostatico_cpu_ki.py:327` | `class MicroTopology(Module)` |
| `__init__` | method | `neurologos_homeostatico_cpu_ki.py:118` | `def __init__(self, d_in, base_lr)` |
| `__init__` | method | `neurologos_homeostatico_cpu_ki.py:177` | `def __init__(self, dim, config)` |
| `__init__` | method | `neurologos_homeostatico_cpu_ki.py:249` | `def __init__(self, dim, config)` |
| `__init__` | method | `neurologos_homeostatico_cpu_ki.py:329` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `neurologos_homeostatico_cpu_ki.py:369` | `def __init__(self, temperature)` |
| `__init__` | method | `neurologos_homeostatico_cpu_ki.py:398` | `def __init__(self, config)` |
| `_create_grid_mask` | method | `neurologos_homeostatico_cpu_ki.py:344` | `def _create_grid_mask(self)` |
| `_init_weights` | method | `neurologos_homeostatico_cpu_ki.py:440` | `def _init_weights(self)` |
| `count_parameters` | method | `neurologos_homeostatico_cpu_ki.py:445` | `def count_parameters(self)` |
| `forward` | method | `neurologos_homeostatico_cpu_ki.py:137` | `def forward(self, x, h_pre, w_norm, loss_val)` |
| `forward` | method | `neurologos_homeostatico_cpu_ki.py:198` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurologos_homeostatico_cpu_ki.py:268` | `def forward(self, x, loss_val)` |
| `forward` | method | `neurologos_homeostatico_cpu_ki.py:374` | `def forward(self, features, labels)` |
| `forward` | method | `neurologos_homeostatico_cpu_ki.py:448` | `def forward(self, x, loss_val)` |
| `generate_ablation_matrix` | method | `neurologos_homeostatico_cpu_ki.py:556` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `neurologos_homeostatico_cpu_ki.py:355` | `def get_adjacency(self, plasticity, loss_val)` |
| `get_dataset` | method | `neurologos_homeostatico_cpu_ki.py:85` | `def get_dataset(config)` |
| `micro_pgd_attack` | method | `neurologos_homeostatico_cpu_ki.py:521` | `def micro_pgd_attack(model, x, y, eps, steps, loss_val)` |
| `run_ablation_study` | method | `neurologos_homeostatico_cpu_ki.py:777` | `def run_ablation_study()` |
| `seed_everything` | method | `neurologos_homeostatico_cpu_ki.py:74` | `def seed_everything(seed)` |
| `train_with_cv` | method | `neurologos_homeostatico_cpu_ki.py:595` | `def train_with_cv(config, dataset, cv_folds)` |
| `HomeostaticRegulatorMini` | class | `neurologos_homestotico_cpu_qw.py:92` | `class HomeostaticRegulatorMini(Module)` |
| `MicroConfig` | class | `neurologos_homestotico_cpu_qw.py:35` | `class MicroConfig` |
| `MicroContinuumCell` | class | `neurologos_homestotico_cpu_qw.py:162` | `class MicroContinuumCell(Module)` |
| `MicroPhysioNeuron` | class | `neurologos_homestotico_cpu_qw.py:117` | `class MicroPhysioNeuron(Module)` |
| `MicroSupConLoss` | class | `neurologos_homestotico_cpu_qw.py:228` | `class MicroSupConLoss(Module)` |
| `MicroSymbioticBasis` | class | `neurologos_homestotico_cpu_qw.py:187` | `class MicroSymbioticBasis(Module)` |
| `MicroTopoBrain` | class | `neurologos_homestotico_cpu_qw.py:252` | `class MicroTopoBrain(Module)` |
| `MicroTopology` | class | `neurologos_homestotico_cpu_qw.py:207` | `class MicroTopology` |
| `__init__` | method | `neurologos_homestotico_cpu_qw.py:93` | `def __init__(self, d_in)` |
| `__init__` | method | `neurologos_homestotico_cpu_qw.py:118` | `def __init__(self, d_in, d_out, dynamic)` |
| `__init__` | method | `neurologos_homestotico_cpu_qw.py:163` | `def __init__(self, dim)` |
| `__init__` | method | `neurologos_homestotico_cpu_qw.py:188` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `neurologos_homestotico_cpu_qw.py:208` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `neurologos_homestotico_cpu_qw.py:229` | `def __init__(self, temperature)` |
| `__init__` | method | `neurologos_homestotico_cpu_qw.py:253` | `def __init__(self, config)` |
| `_init_weights` | method | `neurologos_homestotico_cpu_qw.py:301` | `def _init_weights(self)` |
| `count_parameters` | method | `neurologos_homestotico_cpu_qw.py:306` | `def count_parameters(self)` |
| `forward` | method | `neurologos_homestotico_cpu_qw.py:104` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `neurologos_homestotico_cpu_qw.py:129` | `def forward(self, x)` |
| `forward` | method | `neurologos_homestotico_cpu_qw.py:173` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurologos_homestotico_cpu_qw.py:196` | `def forward(self, x)` |
| `forward` | method | `neurologos_homestotico_cpu_qw.py:234` | `def forward(self, features, labels)` |
| `forward` | method | `neurologos_homestotico_cpu_qw.py:309` | `def forward(self, x, plasticity)` |
| `generate_ablation_matrix` | method | `neurologos_homestotico_cpu_qw.py:389` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `neurologos_homestotico_cpu_qw.py:222` | `def get_adjacency(self, plasticity)` |
| `get_dataset` | method | `neurologos_homestotico_cpu_qw.py:72` | `def get_dataset(config)` |
| `micro_pgd_attack` | method | `neurologos_homestotico_cpu_qw.py:365` | `def micro_pgd_attack(model, x, y, eps, steps, plasticity)` |
| `run_ablation_study` | method | `neurologos_homestotico_cpu_qw.py:496` | `def run_ablation_study()` |
| `seed_everything` | method | `neurologos_homestotico_cpu_qw.py:64` | `def seed_everything(seed)` |
| `train_with_cv` | method | `neurologos_homestotico_cpu_qw.py:429` | `def train_with_cv(config, dataset, cv_folds)` |
| `AudioEncoder` | class | `neurologos_tricameral_exodia.py:1719` | `class AudioEncoder(Module)` |
| `CausalReasoningEngine` | class | `neurologos_tricameral_exodia.py:994` | `class CausalReasoningEngine(Module)` |
| `CorpusCallosumTrimodal` | class | `neurologos_tricameral_exodia.py:1853` | `class CorpusCallosumTrimodal(Module)` |
| `EnhancedDiagnosticsTricameral` | class | `neurologos_tricameral_exodia.py:2006` | `class EnhancedDiagnosticsTricameral` |
| `Flickr8kMultimodalDataset` | class | `neurologos_tricameral_exodia.py:2346` | `class Flickr8kMultimodalDataset(Dataset)` |
| `HierarchicalEpisodicMemory` | class | `neurologos_tricameral_exodia.py:332` | `class HierarchicalEpisodicMemory` |
| `LanguageMetrics` | class | `neurologos_tricameral_exodia.py:760` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `neurologos_tricameral_exodia.py:951` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `neurologos_tricameral_exodia.py:1073` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `neurologos_tricameral_exodia.py:1410` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `neurologos_tricameral_exodia.py:834` | `class LinguisticFeedbackLoop` |
| `NeuroLogosTricameral` | class | `neurologos_tricameral_exodia.py:2311` | `class NeuroLogosTricameral(Module)` |
| `NeurocognitiveSystem` | class | `neurologos_tricameral_exodia.py:563` | `class NeurocognitiveSystem` |
| `RightHemisphereTricameral` | class | `neurologos_tricameral_exodia.py:1769` | `class RightHemisphereTricameral(Module)` |
| `StableLiquidNeuron` | class | `neurologos_tricameral_exodia.py:1120` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `neurologos_tricameral_exodia.py:1259` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `neurologos_tricameral_exodia.py:2409` | `def __getitem__(self, idx)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:333` | `def __init__(self, working_capacity, short_term_capacity, importance_threshold)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:564` | `def __init__(self)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:835` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:995` | `def __init__(self, hidden_dim)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:1121` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:1260` | `def __init__(self)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:1411` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:1722` | `def __init__(self, output_dim)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:1772` | `def __init__(self, output_dim)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:1854` | `def __init__(self, dim)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:2007` | `def __init__(self)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:2314` | `def __init__(self, vocab_size)` |
| `__init__` | method | `neurologos_tricameral_exodia.py:2349` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache, cache_di` |
| `__len__` | method | `neurologos_tricameral_exodia.py:2406` | `def __len__(self)` |
| `_apply_chain_of_thought` | method | `neurologos_tricameral_exodia.py:1540` | `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)` |
| `_apply_multi_token_prediction` | method | `neurologos_tricameral_exodia.py:1641` | `def _apply_multi_token_prediction(self, hidden_states, input_ids)` |
| `_apply_structural_attention` | method | `neurologos_tricameral_exodia.py:1683` | `def _apply_structural_attention(self, lstm_out, channels, visual_context)` |
| `_calculate_homeostasis_metric` | method | `neurologos_tricameral_exodia.py:1179` | `def _calculate_homeostasis_metric(self, output)` |
| `_calculate_novelty` | method | `neurologos_tricameral_exodia.py:381` | `def _calculate_novelty(self, episode)` |
| `_get_cached_norm` | method | `neurologos_tricameral_exodia.py:2030` | `def _get_cached_norm(self, tensor, dim)` |
| `_get_init_state` | method | `neurologos_tricameral_exodia.py:1704` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `neurologos_tricameral_exodia.py:798` | `def _get_ngrams(tokens, n)` |
| `_get_ngrams_cached` | method | `neurologos_tricameral_exodia.py:849` | `def _get_ngrams_cached(sentence, n)` |
| `_greedy_decode` | method | `neurologos_tricameral_exodia.py:1580` | `def _greedy_decode(self, visual_context, channels, max_len, epoch)` |
| `_predict_interventions` | method | `neurologos_tricameral_exodia.py:1036` | `def _predict_interventions(self, hypothesis, confidence)` |
| `_purge_low_score_memories` | method | `neurologos_tricameral_exodia.py:471` | `def _purge_low_score_memories(self)` |
| `_reset_liquid_neuron` | method | `neurologos_tricameral_exodia.py:1395` | `def _reset_liquid_neuron(self, liquid_neuron)` |
| `_sample_from_buffer` | method | `neurologos_tricameral_exodia.py:527` | `def _sample_from_buffer(self, buffer, scores, batch_size)` |
| `_update_unified_buffer` | method | `neurologos_tricameral_exodia.py:440` | `def _update_unified_buffer(self)` |
| `add` | method | `neurologos_tricameral_exodia.py:452` | `def add(self, image, audio, caption, surprise_score)` |
| `adjust_gates_by_fatigue` | method | `neurologos_tricameral_exodia.py:1985` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `neurologos_tricameral_exodia.py:674` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_forgetting_curve` | method | `neurologos_tricameral_exodia.py:455` | `def apply_forgetting_curve(self)` |
| `apply_triangulated_intervention` | method | `neurologos_tricameral_exodia.py:1326` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `neurologos_tricameral_exodia.py:628` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `assess_reasoning_state` | method | `neurologos_tricameral_exodia.py:584` | `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` |
| `build_vocab_flickr` | function | `neurologos_tricameral_exodia.py:308` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `neurologos_tricameral_exodia.py:2149` | `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_importance` | method | `neurologos_tricameral_exodia.py:369` | `def calculate_importance(self, episode, surprise_score)` |
| `calculate_synergy` | method | `neurologos_tricameral_exodia.py:2138` | `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_alignment_loss` | method | `neurologos_tricameral_exodia.py:2462` | `def compute_alignment_loss(visual_features, channels, alpha, epoch)` |
| `compute_cider` | method | `neurologos_tricameral_exodia.py:897` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `neurologos_tricameral_exodia.py:858` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_spice` | method | `neurologos_tricameral_exodia.py:911` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `neurologos_tricameral_exodia.py:359` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `compute_tricameral_loss` | method | `neurologos_tricameral_exodia.py:2490` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, channel` |
| `count_convergent_signals` | method | `neurologos_tricameral_exodia.py:1278` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `neurologos_tricameral_exodia.py:1281` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)` |
| `evaluate_reasoning_quality` | method | `neurologos_tricameral_exodia.py:2101` | `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` |
| `forward` | method | `neurologos_tricameral_exodia.py:1163` | `def forward(self, x)` |
| `forward` | method | `neurologos_tricameral_exodia.py:1493` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `neurologos_tricameral_exodia.py:1756` | `def forward(self, mel_spec)` |
| `forward` | method | `neurologos_tricameral_exodia.py:1812` | `def forward(self, image, audio)` |
| `forward` | method | `neurologos_tricameral_exodia.py:1902` | `def forward(self, right_features)` |
| `forward` | method | `neurologos_tricameral_exodia.py:2320` | `def forward(self, image, audio, captions, epoch)` |
| `get_cache_stats` | method | `neurologos_tricameral_exodia.py:923` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `neurologos_tricameral_exodia.py:2175` | `def get_recent_avg(self, key, n)` |
| `get_total_size` | method | `neurologos_tricameral_exodia.py:555` | `def get_total_size(self)` |
| `hebbian_update` | method | `neurologos_tricameral_exodia.py:1188` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `neurologos_tricameral_exodia.py:2048` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `preprocess_and_cache_spectrograms` | function | `neurologos_tricameral_exodia.py:47` | `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)` |
| `query_causal_chain` | method | `neurologos_tricameral_exodia.py:1059` | `def query_causal_chain(self, start_node, end_node)` |
| `reason_causally` | method | `neurologos_tricameral_exodia.py:1022` | `def reason_causally(self, observation, context)` |
| `report` | method | `neurologos_tricameral_exodia.py:2227` | `def report(self, epoch)` |
| `sample` | method | `neurologos_tricameral_exodia.py:497` | `def sample(self, batch_size, memory_level)` |
| `sentence_bleu` | method | `neurologos_tricameral_exodia.py:764` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `neurologos_tricameral_exodia.py:953` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `neurologos_tricameral_exodia.py:1075` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k_with_audio` | function | `neurologos_tricameral_exodia.py:120` | `def setup_flickr8k_with_audio(data_dir)` |
| `store_episode` | method | `neurologos_tricameral_exodia.py:402` | `def store_episode(self, image, audio, caption, surprise_score)` |
| `token_accuracy` | method | `neurologos_tricameral_exodia.py:807` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `neurologos_tricameral_exodia.py:976` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `neurologos_tricameral_exodia.py:1098` | `def token_accuracy(reference, hypothesis)` |
| `train_tricameral` | method | `neurologos_tricameral_exodia.py:2563` | `def train_tricameral()` |
| `triangulate_signals` | method | `neurologos_tricameral_exodia.py:1267` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `neurologos_tricameral_exodia.py:2158` | `def update(self)` |
| `update_channel_fatigue` | method | `neurologos_tricameral_exodia.py:1963` | `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` |
| `update_knowledge_graph` | method | `neurologos_tricameral_exodia.py:1053` | `def update_knowledge_graph(self, cause, effect, strength)` |
| `update_physiology_advanced` | method | `neurologos_tricameral_exodia.py:1226` | `def update_physiology_advanced(self, loss_value)` |
| `visualize_fatigue_distribution` | method | `neurologos_tricameral_exodia.py:2191` | `def visualize_fatigue_distribution(self, epoch)` |
| `visualize_reasoning_metrics` | method | `neurologos_tricameral_exodia.py:2215` | `def visualize_reasoning_metrics(self, epoch)` |
| `word_overlap` | method | `neurologos_tricameral_exodia.py:820` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `neurologos_tricameral_exodia.py:986` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `neurologos_tricameral_exodia.py:1108` | `def word_overlap(reference, hypothesis)` |
| `BioDecoder` | class | `neurologos_v4.py:185` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `neurologos_v4.py:291` | `class CIFARCaptions` |
| `ConsciousCore` | class | `neurologos_v4.py:164` | `class ConsciousCore(Module)` |
| `LifeCycle` | class | `neurologos_v4.py:271` | `class LifeCycle` |
| `LiquidNeuron` | class | `neurologos_v4.py:81` | `class LiquidNeuron(Module)` |
| `MiniUnconscious` | class | `neurologos_v4.py:13` | `class MiniUnconscious(Module)` |
| `NestedUnconscious` | class | `neurologos_v4.py:29` | `class NestedUnconscious(Module)` |
| `NeuroLogos` | class | `neurologos_v4.py:236` | `class NeuroLogos(Module)` |
| `__getitem__` | method | `neurologos_v4.py:346` | `def __getitem__(self, idx)` |
| `__init__` | method | `neurologos_v4.py:15` | `def __init__(self)` |
| `__init__` | method | `neurologos_v4.py:31` | `def __init__(self, grid_size, output_dim)` |
| `__init__` | method | `neurologos_v4.py:82` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `neurologos_v4.py:165` | `def __init__(self)` |
| `__init__` | method | `neurologos_v4.py:186` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `neurologos_v4.py:237` | `def __init__(self, vocab_size, use_nested)` |
| `__init__` | method | `neurologos_v4.py:272` | `def __init__(self, total_epochs)` |
| `__init__` | method | `neurologos_v4.py:292` | `def __init__(self)` |
| `__len__` | method | `neurologos_v4.py:344` | `def __len__(self)` |
| `_get_init_state` | method | `neurologos_v4.py:228` | `def _get_init_state(self, thought)` |
| `consolidate_svd` | method | `neurologos_v4.py:131` | `def consolidate_svd(self, repair_strength)` |
| `forward` | method | `neurologos_v4.py:26` | `def forward(self, x)` |
| `forward` | method | `neurologos_v4.py:57` | `def forward(self, x)` |
| `forward` | method | `neurologos_v4.py:103` | `def forward(self, x, global_plasticity)` |
| `forward` | method | `neurologos_v4.py:172` | `def forward(self, visual_features, plasticity)` |
| `forward` | method | `neurologos_v4.py:201` | `def forward(self, thought, captions, max_len)` |
| `forward` | method | `neurologos_v4.py:251` | `def forward(self, image, captions, plasticity)` |
| `get_plasticity` | method | `neurologos_v4.py:276` | `def get_plasticity(self, epoch)` |
| `measure_richness` | method | `neurologos_v4.py:264` | `def measure_richness(self)` |
| `train_logos` | method | `neurologos_v4.py:358` | `def train_logos(use_nested)` |
| `BioDecoder` | class | `neurologos_v5.py:332` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `neurologos_v5.py:457` | `class CIFARCaptions` |
| `ConsciousCore` | class | `neurologos_v5.py:260` | `class ConsciousCore(Module)` |
| `LifeCycle` | class | `neurologos_v5.py:435` | `class LifeCycle` |
| `LiquidNeuron` | class | `neurologos_v5.py:169` | `class LiquidNeuron(Module)` |
| `MiniUnconscious` | class | `neurologos_v5.py:96` | `class MiniUnconscious(Module)` |
| `NestedUnconscious` | class | `neurologos_v5.py:119` | `class NestedUnconscious(Module)` |
| `NeuroLogos` | class | `neurologos_v5.py:404` | `class NeuroLogos(Module)` |
| `TopologicalCompressor` | class | `neurologos_v5.py:75` | `class TopologicalCompressor(Module)` |
| `__getitem__` | method | `neurologos_v5.py:486` | `def __getitem__(self, idx)` |
| `__init__` | method | `neurologos_v5.py:76` | `def __init__(self, node_dim)` |
| `__init__` | method | `neurologos_v5.py:98` | `def __init__(self)` |
| `__init__` | method | `neurologos_v5.py:120` | `def __init__(self, grid_size, output_dim)` |
| `__init__` | method | `neurologos_v5.py:170` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `neurologos_v5.py:261` | `def __init__(self)` |
| `__init__` | method | `neurologos_v5.py:334` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `neurologos_v5.py:405` | `def __init__(self, vocab_size, use_nested)` |
| `__init__` | method | `neurologos_v5.py:436` | `def __init__(self, total_epochs)` |
| `__init__` | method | `neurologos_v5.py:458` | `def __init__(self)` |
| `__len__` | method | `neurologos_v5.py:483` | `def __len__(self)` |
| `_get_init_state` | method | `neurologos_v5.py:395` | `def _get_init_state(self, thought)` |
| `consolidate_svd` | method | `neurologos_v5.py:228` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `neurologos_v5.py:85` | `def forward(self, nodes, plasticity, transfer_rate)` |
| `forward` | method | `neurologos_v5.py:113` | `def forward(self, x)` |
| `forward` | method | `neurologos_v5.py:143` | `def forward(self, x)` |
| `forward` | method | `neurologos_v5.py:191` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `neurologos_v5.py:278` | `def forward(self, visual_features, plasticity, transfer_rate)` |
| `forward` | method | `neurologos_v5.py:349` | `def forward(self, thought, captions, max_len)` |
| `forward` | method | `neurologos_v5.py:420` | `def forward(self, image, captions, plasticity, transfer_rate)` |
| `get_liquid_module` | method | `neurologos_v5.py:319` | `def get_liquid_module(self)` |
| `get_plasticity` | method | `neurologos_v5.py:440` | `def get_plasticity(self, epoch)` |
| `measure_richness` | method | `neurologos_v5.py:428` | `def measure_richness(self)` |
| `measure_spatial_richness` | function | `neurologos_v5.py:32` | `def measure_spatial_richness(activations)` |
| `top_k_top_p_filtering` | function | `neurologos_v5.py:11` | `def top_k_top_p_filtering(logits, top_k, top_p, filter_value)` |
| `train_logos` | method | `neurologos_v5.py:502` | `def train_logos(use_nested)` |
| `ComponentRegulator` | class | `neurologos_v6.py:154` | `class ComponentRegulator(Module)` |
| `Config` | class | `neurologos_v6.py:18` | `class Config` |
| `ConfigurableTrainer` | class | `neurologos_v6.py:473` | `class ConfigurableTrainer` |
| `DataEnvironment` | class | `neurologos_v6.py:90` | `class DataEnvironment` |
| `HomeostaticRegulator` | class | `neurologos_v6.py:211` | `class HomeostaticRegulator(Module)` |
| `MetaHomeostaticEngine` | class | `neurologos_v6.py:237` | `class MetaHomeostaticEngine(Module)` |
| `MetaLearner` | class | `neurologos_v6.py:127` | `class MetaLearner(Module)` |
| `MetricsCollector` | class | `neurologos_v6.py:55` | `class MetricsCollector` |
| `MicroTopoBrain` | class | `neurologos_v6.py:403` | `class MicroTopoBrain(Module)` |
| `PhysioNeuron` | class | `neurologos_v6.py:320` | `class PhysioNeuron(Module)` |
| `SymbioticDual` | class | `neurologos_v6.py:376` | `class SymbioticDual(Module)` |
| `__init__` | method | `neurologos_v6.py:56` | `def __init__(self, config)` |
| `__init__` | method | `neurologos_v6.py:91` | `def __init__(self)` |
| `__init__` | method | `neurologos_v6.py:128` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `neurologos_v6.py:155` | `def __init__(self, name, state_dim, cross_dim)` |
| `__init__` | method | `neurologos_v6.py:212` | `def __init__(self, d_in)` |
| `__init__` | method | `neurologos_v6.py:238` | `def __init__(self, config)` |
| `__init__` | method | `neurologos_v6.py:321` | `def __init__(self, d_in, d_out, config)` |
| `__init__` | method | `neurologos_v6.py:377` | `def __init__(self, dim, atoms)` |
| `__init__` | method | `neurologos_v6.py:404` | `def __init__(self, config)` |
| `__init__` | method | `neurologos_v6.py:474` | `def __init__(self, config)` |
| `_setup_logger` | method | `neurologos_v6.py:62` | `def _setup_logger(self)` |
| `count_parameters` | method | `neurologos_v6.py:427` | `def count_parameters(self)` |
| `evaluate` | method | `neurologos_v6.py:554` | `def evaluate(self, model)` |
| `forward` | method | `neurologos_v6.py:136` | `def forward(self, sequence)` |
| `forward` | method | `neurologos_v6.py:172` | `def forward(self)` |
| `forward` | method | `neurologos_v6.py:222` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `neurologos_v6.py:251` | `def forward(self, global_loss, step)` |
| `forward` | method | `neurologos_v6.py:334` | `def forward(self, x, surprise_threshold)` |
| `forward` | method | `neurologos_v6.py:385` | `def forward(self, x, influence)` |
| `forward` | method | `neurologos_v6.py:430` | `def forward(self, x, y, step)` |
| `generate_ablation_matrix_4levels` | method | `neurologos_v6.py:603` | `def generate_ablation_matrix_4levels()` |
| `get_batch` | method | `neurologos_v6.py:103` | `def get_batch(self, phase, bs, step)` |
| `get_component_health` | method | `neurologos_v6.py:290` | `def get_component_health(self)` |
| `get_full` | method | `neurologos_v6.py:118` | `def get_full(self)` |
| `get_w2` | method | `neurologos_v6.py:121` | `def get_w2(self)` |
| `inject_concept_drift` | method | `neurologos_v6.py:100` | `def inject_concept_drift(self)` |
| `log_batch` | method | `neurologos_v6.py:69` | `def log_batch(self, step, metrics)` |
| `run_ablation_study` | method | `neurologos_v6.py:645` | `def run_ablation_study()` |
| `save` | method | `neurologos_v6.py:79` | `def save(self, path)` |
| `seed_everything` | method | `neurologos_v6.py:45` | `def seed_everything(seed)` |
| `train` | method | `neurologos_v6.py:479` | `def train(self, model)` |
| `update` | method | `neurologos_v6.py:142` | `def update(self, loss_pred, loss_real)` |
| `update_with_momentum` | method | `neurologos_v6.py:298` | `def update_with_momentum(self, current_lr, current_plasticity, meta_out, surprise_rate)` |
| `HomeostaticRegulator` | class | `neurologosv5.2.py:85` | `class HomeostaticRegulator(Module)` |
| `MicroConfig` | class | `neurologosv5.2.py:33` | `class MicroConfig` |
| `MicroContinuumCell` | class | `neurologosv5.2.py:151` | `class MicroContinuumCell(Module)` |
| `MicroSupConLoss` | class | `neurologosv5.2.py:216` | `class MicroSupConLoss(Module)` |
| `MicroSymbioticBasis` | class | `neurologosv5.2.py:176` | `class MicroSymbioticBasis(Module)` |
| `MicroTopoBrain` | class | `neurologosv5.2.py:240` | `class MicroTopoBrain(Module)` |
| `MicroTopology` | class | `neurologosv5.2.py:196` | `class MicroTopology` |
| `PhysioNeuron` | class | `neurologosv5.2.py:109` | `class PhysioNeuron(Module)` |
| `__init__` | method | `neurologosv5.2.py:86` | `def __init__(self, d_in)` |
| `__init__` | method | `neurologosv5.2.py:110` | `def __init__(self, d_in, d_out, dynamic_mode)` |
| `__init__` | method | `neurologosv5.2.py:152` | `def __init__(self, dim)` |
| `__init__` | method | `neurologosv5.2.py:177` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `neurologosv5.2.py:197` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `neurologosv5.2.py:217` | `def __init__(self, temperature)` |
| `__init__` | method | `neurologosv5.2.py:241` | `def __init__(self, config)` |
| `_init_weights` | method | `neurologosv5.2.py:278` | `def _init_weights(self)` |
| `count_parameters` | method | `neurologosv5.2.py:283` | `def count_parameters(self)` |
| `forward` | method | `neurologosv5.2.py:96` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `neurologosv5.2.py:121` | `def forward(self, x)` |
| `forward` | method | `neurologosv5.2.py:162` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurologosv5.2.py:185` | `def forward(self, x)` |
| `forward` | method | `neurologosv5.2.py:222` | `def forward(self, features, labels)` |
| `forward` | method | `neurologosv5.2.py:286` | `def forward(self, x, plasticity)` |
| `generate_ablation_matrix` | method | `neurologosv5.2.py:364` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `neurologosv5.2.py:210` | `def get_adjacency(self, plasticity)` |
| `get_dataset` | method | `neurologosv5.2.py:65` | `def get_dataset(config)` |
| `micro_pgd_attack` | method | `neurologosv5.2.py:340` | `def micro_pgd_attack(model, x, y, eps, steps, plasticity)` |
| `run_ablation_study` | method | `neurologosv5.2.py:457` | `def run_ablation_study()` |
| `seed_everything` | method | `neurologosv5.2.py:57` | `def seed_everything(seed)` |
| `train_with_cv` | method | `neurologosv5.2.py:393` | `def train_with_cv(config, dataset, cv_folds)` |
| `BasicBlock` | class | `neurosoberano.py:39` | `class BasicBlock(Module)` |
| `Experiment` | class | `neurosoberano.py:233` | `class Experiment` |
| `ExperimentConfig` | class | `neurosoberano.py:28` | `class ExperimentConfig` |
| `FastLiquidNeuron` | class | `neurosoberano.py:96` | `class FastLiquidNeuron(Module)` |
| `MinimalNeuroSovereign` | class | `neurosoberano.py:125` | `class MinimalNeuroSovereign(Module)` |
| `WideResNetBaseline` | class | `neurosoberano.py:60` | `class WideResNetBaseline(Module)` |
| `__init__` | method | `neurosoberano.py:40` | `def __init__(self, in_c, out_c, stride)` |
| `__init__` | method | `neurosoberano.py:65` | `def __init__(self, num_classes)` |
| `__init__` | method | `neurosoberano.py:98` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `neurosoberano.py:127` | `def __init__(self, num_classes)` |
| `__init__` | method | `neurosoberano.py:234` | `def __init__(self, config)` |
| `_get_data` | method | `neurosoberano.py:245` | `def _get_data(self)` |
| `_make_layer` | method | `neurosoberano.py:79` | `def _make_layer(self, in_c, out_c, num_blocks, stride)` |
| `_make_layer` | method | `neurosoberano.py:146` | `def _make_layer(self, in_c, out_c, num_blocks, stride)` |
| `compare` | method | `neurosoberano.py:369` | `def compare(self)` |
| `evaluate` | method | `neurosoberano.py:215` | `def evaluate(model, loader, device)` |
| `forward` | method | `neurosoberano.py:54` | `def forward(self, x)` |
| `forward` | method | `neurosoberano.py:85` | `def forward(self, x)` |
| `forward` | method | `neurosoberano.py:107` | `def forward(self, x, plasticity)` |
| `forward` | method | `neurosoberano.py:152` | `def forward(self, x)` |
| `main` | method | `neurosoberano.py:435` | `def main()` |
| `plot_comparison` | method | `neurosoberano.py:404` | `def plot_comparison(self)` |
| `run_baseline` | method | `neurosoberano.py:269` | `def run_baseline(self)` |
| `run_neurosovereign` | method | `neurosoberano.py:316` | `def run_neurosovereign(self)` |
| `train_epoch` | method | `neurosoberano.py:177` | `def train_epoch(model, loader, optimizer, criterion, device, use_mixup)` |
| `update_plasticity` | method | `neurosoberano.py:163` | `def update_plasticity(self, epoch, total_epochs)` |
| `CorpusCallosum` | class | `neurosoberano_bicameral_opt.py:313` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `neurosoberano_bicameral_opt.py:436` | `class Flickr8kDataset(Dataset)` |
| `HomeostaticRegulator` | class | `neurosoberano_bicameral_opt.py:39` | `class HomeostaticRegulator(Module)` |
| `LeftHemisphere` | class | `neurosoberano_bicameral_opt.py:214` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `neurosoberano_bicameral_opt.py:498` | `class LifeCycle` |
| `LiquidNeuron` | class | `neurosoberano_bicameral_opt.py:66` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `neurosoberano_bicameral_opt.py:361` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `neurosoberano_bicameral_opt.py:331` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `neurosoberano_bicameral_opt.py:193` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `neurosoberano_bicameral_opt.py:458` | `def __getitem__(self, idx)` |
| `__init__` | method | `neurosoberano_bicameral_opt.py:40` | `def __init__(self)` |
| `__init__` | method | `neurosoberano_bicameral_opt.py:67` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `neurosoberano_bicameral_opt.py:194` | `def __init__(self, output_dim)` |
| `__init__` | method | `neurosoberano_bicameral_opt.py:215` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `neurosoberano_bicameral_opt.py:314` | `def __init__(self, dim)` |
| `__init__` | method | `neurosoberano_bicameral_opt.py:332` | `def __init__(self, vocab_size)` |
| `__init__` | method | `neurosoberano_bicameral_opt.py:362` | `def __init__(self)` |
| `__init__` | method | `neurosoberano_bicameral_opt.py:437` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `neurosoberano_bicameral_opt.py:499` | `def __init__(self, total_epochs)` |
| `__len__` | method | `neurosoberano_bicameral_opt.py:455` | `def __len__(self)` |
| `_get_init_state` | method | `neurosoberano_bicameral_opt.py:293` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `neurosoberano_bicameral_opt.py:298` | `def _top_p_filtering(self, logits, top_p)` |
| `apply_svd_consolidation` | method | `neurosoberano_bicameral_opt.py:169` | `def apply_svd_consolidation(self, repair_strength, timescale)` |
| `build_vocab_flickr` | method | `neurosoberano_bicameral_opt.py:475` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `forward` | method | `neurosoberano_bicameral_opt.py:50` | `def forward(self, stress, excitation, fatigue, entropy, phase, loss_signal)` |
| `forward` | method | `neurosoberano_bicameral_opt.py:99` | `def forward(self, x, global_plasticity, transfer_rate, task_loss)` |
| `forward` | method | `neurosoberano_bicameral_opt.py:205` | `def forward(self, image, plasticity, transfer_rate, task_loss)` |
| `forward` | method | `neurosoberano_bicameral_opt.py:237` | `def forward(self, visual_context, captions, max_len, return_gate)` |
| `forward` | method | `neurosoberano_bicameral_opt.py:322` | `def forward(self, right_features, metabolism)` |
| `forward` | method | `neurosoberano_bicameral_opt.py:338` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)` |
| `get_plasticity` | method | `neurosoberano_bicameral_opt.py:502` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `neurosoberano_bicameral_opt.py:391` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `neurosoberano_bicameral_opt.py:375` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `neurosoberano_bicameral_opt.py:382` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `neurosoberano_bicameral_opt.py:396` | `def report(self, epoch)` |
| `train_bicameral` | method | `neurosoberano_bicameral_opt.py:513` | `def train_bicameral()` |
| `update` | method | `neurosoberano_bicameral_opt.py:386` | `def update(self)` |
| `BasicBlock` | class | `neurosovereign.py:69` | `class BasicBlock(Module)` |
| `LiquidCortex` | class | `neurosovereign.py:111` | `class LiquidCortex(Module)` |
| `NetworkBlock` | class | `neurosovereign.py:96` | `class NetworkBlock(Module)` |
| `NeuroSovereignV1` | class | `neurosovereign.py:165` | `class NeuroSovereignV1(Module)` |
| `SovereignConfig` | class | `neurosovereign.py:16` | `class SovereignConfig` |
| `__init__` | method | `neurosovereign.py:70` | `def __init__(self, in_planes, out_planes, stride, dropRate)` |
| `__init__` | method | `neurosovereign.py:97` | `def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate)` |
| `__init__` | method | `neurosovereign.py:116` | `def __init__(self, in_features, out_features, config)` |
| `__init__` | method | `neurosovereign.py:166` | `def __init__(self, config, depth, num_classes)` |
| `_make_layer` | method | `neurosovereign.py:100` | `def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate)` |
| `forward` | method | `neurosovereign.py:85` | `def forward(self, x)` |
| `forward` | method | `neurosovereign.py:105` | `def forward(self, x)` |
| `forward` | method | `neurosovereign.py:133` | `def forward(self, x)` |
| `forward` | method | `neurosovereign.py:196` | `def forward(self, x)` |
| `get_optimized_dataloaders` | method | `neurosovereign.py:214` | `def get_optimized_dataloaders(config)` |
| `mixup_criterion` | method | `neurosovereign.py:63` | `def mixup_criterion(criterion, pred, y_a, y_b, lam)` |
| `mixup_data` | method | `neurosovereign.py:51` | `def mixup_data(x, y, alpha)` |
| `seed_everything` | method | `neurosovereign.py:44` | `def seed_everything(seed)` |
| `train_sovereign` | method | `neurosovereign.py:239` | `def train_sovereign()` |
| `AdaptiveLearningMotor` | class | `ohm.py:213` | `class AdaptiveLearningMotor(MotorHomeostaticContext)` |
| `ConsciousnessModule` | class | `ohm.py:597` | `class ConsciousnessModule(OmniBrainModule)` |
| `ConsciousnessMotor` | class | `ohm.py:157` | `class ConsciousnessMotor(MotorHomeostaticContext)` |
| `DualMindModule` | class | `ohm.py:543` | `class DualMindModule(OmniBrainModule)` |
| `DualSystemMotor` | class | `ohm.py:185` | `class DualSystemMotor(MotorHomeostaticContext)` |
| `EnergyHomeostaticMotor` | class | `ohm.py:127` | `class EnergyHomeostaticMotor(MotorHomeostaticContext)` |
| `HomeostaticEngine` | class | `ohm.py:658` | `class HomeostaticEngine` |
| `ModularActivationMotor` | class | `ohm.py:241` | `class ModularActivationMotor(MotorHomeostaticContext)` |
| `MotorHomeostaticContext` | class | `ohm.py:34` | `class MotorHomeostaticContext` |
| `OmniBrain` | class | `ohm.py:689` | `class OmniBrain(Module)` |
| `OmniBrainCoordinator` | class | `ohm.py:291` | `class OmniBrainCoordinator` |
| `OmniBrainModule` | class | `ohm.py:438` | `class OmniBrainModule(Module)` |
| `PTSymmetricLayer` | class | `ohm.py:453` | `class PTSymmetricLayer(OmniBrainModule)` |
| `PTSymmetricMotor` | class | `ohm.py:64` | `class PTSymmetricMotor(MotorHomeostaticContext)` |
| `TopologicalLayer` | class | `ohm.py:486` | `class TopologicalLayer(OmniBrainModule)` |
| `TopologicalMotor` | class | `ohm.py:100` | `class TopologicalMotor(MotorHomeostaticContext)` |
| `__init__` | method | `ohm.py:66` | `def __init__(self)` |
| `__init__` | method | `ohm.py:102` | `def __init__(self)` |
| `__init__` | method | `ohm.py:129` | `def __init__(self)` |
| `__init__` | method | `ohm.py:159` | `def __init__(self)` |
| `__init__` | method | `ohm.py:187` | `def __init__(self)` |
| `__init__` | method | `ohm.py:215` | `def __init__(self)` |
| `__init__` | method | `ohm.py:243` | `def __init__(self)` |
| `__init__` | method | `ohm.py:294` | `def __init__(self)` |
| `__init__` | method | `ohm.py:441` | `def __init__(self, module_name, enabled)` |
| `__init__` | method | `ohm.py:456` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `ohm.py:489` | `def __init__(self, in_features, out_features, sparsity_factor)` |
| `__init__` | method | `ohm.py:546` | `def __init__(self, features)` |
| `__init__` | method | `ohm.py:600` | `def __init__(self, features)` |
| `__init__` | method | `ohm.py:661` | `def __init__(self, target_performance)` |
| `__init__` | method | `ohm.py:692` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `_generate_topology_mask` | method | `ohm.py:504` | `def _generate_topology_mask(self)` |
| `_initialize_motors` | method | `ohm.py:300` | `def _initialize_motors(self)` |
| `compute_phi_effective` | method | `ohm.py:614` | `def compute_phi_effective(self, x)` |
| `coordinate_all_motors` | method | `ohm.py:363` | `def coordinate_all_motors(self, environment_state, network_state)` |
| `forward` | method | `ohm.py:447` | `def forward(self, x, params)` |
| `forward` | method | `ohm.py:463` | `def forward(self, x, params)` |
| `forward` | method | `ohm.py:525` | `def forward(self, x, params)` |
| `forward` | method | `ohm.py:571` | `def forward(self, x, params)` |
| `forward` | method | `ohm.py:634` | `def forward(self, x, params)` |
| `forward` | method | `ohm.py:740` | `def forward(self, x)` |
| `get_status_report` | method | `ohm.py:828` | `def get_status_report(self)` |
| `initialize_context` | method | `ohm.py:726` | `def initialize_context(self)` |
| `measure_network_state` | method | `ohm.py:326` | `def measure_network_state(self, model, batch_data)` |
| `regulate_connectivity` | method | `ohm.py:112` | `def regulate_connectivity(self, current_connectivity, clustering)` |
| `regulate_consciousness` | method | `ohm.py:168` | `def regulate_consciousness(self, phi_effective, integration_level)` |
| `regulate_dual_systems` | method | `ohm.py:197` | `def regulate_dual_systems(self, unconscious_activity, conscious_activity)` |
| `regulate_energy` | method | `ohm.py:139` | `def regulate_energy(self, memory_usage, cpu_usage, temperature)` |
| `regulate_homeostasis` | method | `ohm.py:666` | `def regulate_homeostasis(self, observed_performance)` |
| `regulate_learning` | method | `ohm.py:224` | `def regulate_learning(self, loss_reduction_rate, gradient_norm)` |
| `regulate_modules` | method | `ohm.py:259` | `def regulate_modules(self, task_complexity, resource_availability, performance)` |
| `regulate_parameters` | method | `ohm.py:78` | `def regulate_parameters(self, current_coherence, energy_level)` |
| `sense_environment` | method | `ohm.py:312` | `def sense_environment(self)` |
| `train_omni_brain` | method | `ohm.py:863` | `def train_omni_brain(model, epochs, batch_size)` |
| `update` | method | `ohm.py:47` | `def update(self, measurement, dt)` |
| `update_performance` | method | `ohm.py:450` | `def update_performance(self, metrics)` |
| `ConsciousnessModule` | class | `omni1.py:83` | `class ConsciousnessModule(Module)` |
| `DualSystemModule` | class | `omni1.py:215` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `omni1.py:31` | `class FastSlowLinear(Module)` |
| `FocalLoss` | class | `omni1.py:261` | `class FocalLoss(Module)` |
| `OmniBrainV8` | class | `omni1.py:140` | `class OmniBrainV8(Module)` |
| `__init__` | method | `omni1.py:33` | `def __init__(self, in_features, out_features, fast_lr, fast_decay)` |
| `__init__` | method | `omni1.py:85` | `def __init__(self, features, use_conscious)` |
| `__init__` | method | `omni1.py:142` | `def __init__(self, use_fastslow, use_conscious)` |
| `__init__` | method | `omni1.py:217` | `def __init__(self, dim, use_fastslow)` |
| `__init__` | method | `omni1.py:262` | `def __init__(self, alpha, gamma)` |
| `compute_phi_effective` | method | `omni1.py:99` | `def compute_phi_effective(self, activity)` |
| `diagnose_model` | method | `omni1.py:423` | `def diagnose_model(model, loader, device)` |
| `evaluate` | method | `omni1.py:356` | `def evaluate(model, loader, device, return_per_class)` |
| `forward` | method | `omni1.py:68` | `def forward(self, x)` |
| `forward` | method | `omni1.py:121` | `def forward(self, x)` |
| `forward` | method | `omni1.py:188` | `def forward(self, x)` |
| `forward` | method | `omni1.py:235` | `def forward(self, x)` |
| `forward` | method | `omni1.py:268` | `def forward(self, inputs, targets)` |
| `get_activation` | method | `omni1.py:436` | `def get_activation(name)` |
| `get_cifar10_loaders` | method | `omni1.py:399` | `def get_cifar10_loaders(batch_size)` |
| `get_fast_norm` | method | `omni1.py:75` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `omni1.py:205` | `def get_fast_norms(self)` |
| `hook` | method | `omni1.py:437` | `def hook(model, input, output)` |
| `main` | method | `omni1.py:481` | `def main()` |
| `reset_all_fast_weights` | method | `omni1.py:198` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `omni1.py:46` | `def reset_fast_weights(self)` |
| `train_model` | method | `omni1.py:250` | `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)` |
| `update_fast_weights` | method | `omni1.py:49` | `def update_fast_weights(self, x)` |
| `Config` | class | `omni3.py:22` | `class Config` |
| `DualSystemModule` | class | `omni3.py:196` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `omni3.py:105` | `class FastSlowLinear(Module)` |
| `IntegrationModule` | class | `omni3.py:232` | `class IntegrationModule(Module)` |
| `OmniBrainFastSlow` | class | `omni3.py:268` | `class OmniBrainFastSlow(Module)` |
| `__init__` | method | `omni3.py:106` | `def __init__(self, in_features, out_features, config)` |
| `__init__` | method | `omni3.py:197` | `def __init__(self, dim, config)` |
| `__init__` | method | `omni3.py:236` | `def __init__(self, features, config)` |
| `__init__` | method | `omni3.py:269` | `def __init__(self, config)` |
| `compute_integration_index` | method | `omni3.py:65` | `def compute_integration_index(activity)` |
| `evaluate_full` | method | `omni3.py:368` | `def evaluate_full(model, loader, device)` |
| `forward` | method | `omni3.py:169` | `def forward(self, x)` |
| `forward` | method | `omni3.py:211` | `def forward(self, x)` |
| `forward` | method | `omni3.py:248` | `def forward(self, x)` |
| `forward` | method | `omni3.py:304` | `def forward(self, x)` |
| `get_ablation_state` | method | `omni3.py:327` | `def get_ablation_state(self)` |
| `get_cifar10_loaders` | method | `omni3.py:340` | `def get_cifar10_loaders(config)` |
| `get_fast_norm` | method | `omni3.py:189` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `omni3.py:323` | `def get_fast_norms(self)` |
| `reset_all_fast_weights` | method | `omni3.py:314` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `omni3.py:128` | `def reset_fast_weights(self)` |
| `run_ablation_study` | method | `omni3.py:539` | `def run_ablation_study()` |
| `train` | method | `omni3.py:403` | `def train(config)` |
| `update_fast_weights` | method | `omni3.py:134` | `def update_fast_weights(self, x, slow_out)` |
| `ConsciousnessModule` | class | `omnibrain.py:119` | `class ConsciousnessModule(Module)` |
| `DualSystemModule` | class | `omnibrain.py:91` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `omnibrain.py:31` | `class FastSlowLinear(Module)` |
| `OmniBrainV8` | class | `omnibrain.py:174` | `class OmniBrainV8(Module)` |
| `__init__` | method | `omnibrain.py:33` | `def __init__(self, in_features, out_features, fast_lr, fast_decay)` |
| `__init__` | method | `omnibrain.py:93` | `def __init__(self, dim, use_fastslow)` |
| `__init__` | method | `omnibrain.py:121` | `def __init__(self, features, use_conscious)` |
| `__init__` | method | `omnibrain.py:176` | `def __init__(self, use_fastslow, use_conscious)` |
| `analyze_phi_per_class` | method | `omnibrain.py:508` | `def analyze_phi_per_class()` |
| `compute_phi_effective` | method | `omnibrain.py:133` | `def compute_phi_effective(self, activity)` |
| `end_of_batch` | method | `omnibrain.py:84` | `def end_of_batch(self)` |
| `evaluate` | method | `omnibrain.py:287` | `def evaluate(model, loader, device, return_per_class)` |
| `forward` | method | `omnibrain.py:73` | `def forward(self, x)` |
| `forward` | method | `omnibrain.py:109` | `def forward(self, x)` |
| `forward` | method | `omnibrain.py:156` | `def forward(self, x)` |
| `forward` | method | `omnibrain.py:209` | `def forward(self, x)` |
| `get_cifar10_loaders` | method | `omnibrain.py:233` | `def get_cifar10_loaders(batch_size)` |
| `get_fast_norm` | method | `omnibrain.py:87` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `omnibrain.py:223` | `def get_fast_norms(self)` |
| `get_few_shot_loaders` | method | `omnibrain.py:253` | `def get_few_shot_loaders(n_way, k_shot, batch_size)` |
| `main` | method | `omnibrain.py:673` | `def main()` |
| `plot_ablation_results` | method | `omnibrain.py:545` | `def plot_ablation_results(results)` |
| `plot_phi_analysis` | method | `omnibrain.py:617` | `def plot_phi_analysis(class_accs, avg_phi_per_class)` |
| `reset_all_fast_weights` | method | `omnibrain.py:216` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `omnibrain.py:49` | `def reset_fast_weights(self)` |
| `run_ablation_study` | method | `omnibrain.py:430` | `def run_ablation_study(epochs, batch_size)` |
| `run_few_shot_experiment` | method | `omnibrain.py:475` | `def run_few_shot_experiment(n_way, k_shot, epochs)` |
| `train_model` | method | `omnibrain.py:331` | `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)` |
| `update_fast_weights` | method | `omnibrain.py:53` | `def update_fast_weights(self, x)` |
| `Config` | class | `omnibrain_k.py:22` | `class Config` |
| `DualSystemModule` | class | `omnibrain_k.py:195` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `omnibrain_k.py:105` | `class FastSlowLinear(Module)` |
| `IntegrationModule` | class | `omnibrain_k.py:231` | `class IntegrationModule(Module)` |
| `OmniBrainFastSlow` | class | `omnibrain_k.py:266` | `class OmniBrainFastSlow(Module)` |
| `__init__` | method | `omnibrain_k.py:106` | `def __init__(self, in_features, out_features, config)` |
| `__init__` | method | `omnibrain_k.py:196` | `def __init__(self, dim, config)` |
| `__init__` | method | `omnibrain_k.py:235` | `def __init__(self, features, config)` |
| `__init__` | method | `omnibrain_k.py:267` | `def __init__(self, config)` |
| `compute_integration_index` | method | `omnibrain_k.py:65` | `def compute_integration_index(activity)` |
| `evaluate_full` | method | `omnibrain_k.py:367` | `def evaluate_full(model, loader, device)` |
| `forward` | method | `omnibrain_k.py:169` | `def forward(self, x)` |
| `forward` | method | `omnibrain_k.py:210` | `def forward(self, x)` |
| `forward` | method | `omnibrain_k.py:247` | `def forward(self, x)` |
| `forward` | method | `omnibrain_k.py:302` | `def forward(self, x)` |
| `get_ablation_state` | method | `omnibrain_k.py:326` | `def get_ablation_state(self)` |
| `get_cifar10_loaders` | method | `omnibrain_k.py:339` | `def get_cifar10_loaders(config)` |
| `get_fast_norm` | method | `omnibrain_k.py:189` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `omnibrain_k.py:322` | `def get_fast_norms(self)` |
| `reset_all_fast_weights` | method | `omnibrain_k.py:312` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `omnibrain_k.py:128` | `def reset_fast_weights(self)` |
| `run_ablation_study` | method | `omnibrain_k.py:537` | `def run_ablation_study()` |
| `train` | method | `omnibrain_k.py:402` | `def train(config)` |
| `update_fast_weights` | method | `omnibrain_k.py:134` | `def update_fast_weights(self, x, slow_out)` |
| `ConsciousnessModule` | class | `omno1.bkp.py.py:101` | `class ConsciousnessModule(Module)` |
| `DualSystemModule` | class | `omno1.bkp.py.py:256` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `omno1.bkp.py.py:32` | `class FastSlowLinear(Module)` |
| `FocalLoss` | class | `omno1.bkp.py.py:302` | `class FocalLoss(Module)` |
| `OmniBrainV8` | class | `omno1.bkp.py.py:181` | `class OmniBrainV8(Module)` |
| `__init__` | method | `omno1.bkp.py.py:34` | `def __init__(self, in_features, out_features, fast_lr, fast_decay)` |
| `__init__` | method | `omno1.bkp.py.py:103` | `def __init__(self, features, use_conscious)` |
| `__init__` | method | `omno1.bkp.py.py:183` | `def __init__(self, use_fastslow, use_conscious)` |
| `__init__` | method | `omno1.bkp.py.py:258` | `def __init__(self, dim, use_fastslow)` |
| `__init__` | method | `omno1.bkp.py.py:303` | `def __init__(self, alpha, gamma)` |
| `compute_phi_effective` | method | `omno1.bkp.py.py:118` | `def compute_phi_effective(self, activity)` |
| `diagnose_model` | method | `omno1.bkp.py.py:466` | `def diagnose_model(model, loader, device)` |
| `evaluate` | method | `omno1.bkp.py.py:399` | `def evaluate(model, loader, device, return_per_class)` |
| `forward` | method | `omno1.bkp.py.py:85` | `def forward(self, x)` |
| `forward` | method | `omno1.bkp.py.py:159` | `def forward(self, x)` |
| `forward` | method | `omno1.bkp.py.py:229` | `def forward(self, x)` |
| `forward` | method | `omno1.bkp.py.py:276` | `def forward(self, x)` |
| `forward` | method | `omno1.bkp.py.py:309` | `def forward(self, inputs, targets)` |
| `get_activation` | method | `omno1.bkp.py.py:479` | `def get_activation(name)` |
| `get_cifar10_loaders` | method | `omno1.bkp.py.py:442` | `def get_cifar10_loaders(batch_size)` |
| `get_fast_norm` | method | `omno1.bkp.py.py:93` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `omno1.bkp.py.py:246` | `def get_fast_norms(self)` |
| `hook` | method | `omno1.bkp.py.py:480` | `def hook(model, input, output)` |
| `main` | method | `omno1.bkp.py.py:524` | `def main()` |
| `reset_all_fast_weights` | method | `omno1.bkp.py.py:239` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `omno1.bkp.py.py:51` | `def reset_fast_weights(self)` |
| `train_model` | method | `omno1.bkp.py.py:291` | `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)` |
| `update_fast_weights` | method | `omno1.bkp.py.py:56` | `def update_fast_weights(self, x)` |
| `Config` | class | `physio_chimera_demo.py:24` | `class Config` |
| `DataEnvironment` | class | `physio_chimera_demo.py:46` | `class DataEnvironment` |
| `SimpleCMS` | class | `physio_chimera_demo.py:130` | `class SimpleCMS(Module)` |
| `SimpleMonitor` | class | `physio_chimera_demo.py:82` | `class SimpleMonitor` |
| `SimplePhysioChimera` | class | `physio_chimera_demo.py:194` | `class SimplePhysioChimera(Module)` |
| `SimplePhysioNeuron` | class | `physio_chimera_demo.py:153` | `class SimplePhysioNeuron(Module)` |
| `__init__` | method | `physio_chimera_demo.py:47` | `def __init__(self)` |
| `__init__` | method | `physio_chimera_demo.py:83` | `def __init__(self)` |
| `__init__` | method | `physio_chimera_demo.py:131` | `def __init__(self, levels, d_model, hidden_dim)` |
| `__init__` | method | `physio_chimera_demo.py:154` | `def __init__(self, d_in, d_out, config)` |
| `__init__` | method | `physio_chimera_demo.py:195` | `def __init__(self, config)` |
| `forward` | method | `physio_chimera_demo.py:142` | `def forward(self, x, global_step)` |
| `forward` | method | `physio_chimera_demo.py:162` | `def forward(self, x, global_step)` |
| `forward` | method | `physio_chimera_demo.py:207` | `def forward(self, x, global_step)` |
| `get_batch` | method | `physio_chimera_demo.py:57` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `physio_chimera_demo.py:71` | `def get_full(self)` |
| `get_w2` | method | `physio_chimera_demo.py:75` | `def get_w2(self)` |
| `report` | method | `physio_chimera_demo.py:94` | `def report(self, step, phase)` |
| `run_demo` | method | `physio_chimera_demo.py:298` | `def run_demo()` |
| `seed_everything` | method | `physio_chimera_demo.py:36` | `def seed_everything(seed)` |
| `train_demo` | method | `physio_chimera_demo.py:231` | `def train_demo(config)` |
| `update` | method | `physio_chimera_demo.py:88` | `def update(self, loss, physio)` |
| `Config` | class | `physio_chimera_v15_monitored.py:42` | `class Config` |
| `ContinuumMemorySystem` | class | `physio_chimera_v15_monitored.py:319` | `class ContinuumMemorySystem(Module)` |
| `DataEnvironment` | class | `physio_chimera_v15_monitored.py:68` | `class DataEnvironment` |
| `MetricsVisualizer` | class | `physio_chimera_v15_monitored.py:452` | `class MetricsVisualizer` |
| `NestedPhysioNeuron` | class | `physio_chimera_v15_monitored.py:346` | `class NestedPhysioNeuron(Module)` |
| `NeuralDiagnostics` | class | `physio_chimera_v15_monitored.py:104` | `class NeuralDiagnostics` |
| `PhysioChimeraNested` | class | `physio_chimera_v15_monitored.py:397` | `class PhysioChimeraNested(Module)` |
| `SelfModifyingGates` | class | `physio_chimera_v15_monitored.py:298` | `class SelfModifyingGates(Module)` |
| `__init__` | method | `physio_chimera_v15_monitored.py:69` | `def __init__(self)` |
| `__init__` | method | `physio_chimera_v15_monitored.py:107` | `def __init__(self, config)` |
| `__init__` | method | `physio_chimera_v15_monitored.py:299` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `physio_chimera_v15_monitored.py:320` | `def __init__(self, levels, d_model, hidden_dim)` |
| `__init__` | method | `physio_chimera_v15_monitored.py:347` | `def __init__(self, d_in, d_out, config)` |
| `__init__` | method | `physio_chimera_v15_monitored.py:398` | `def __init__(self, config)` |
| `__init__` | method | `physio_chimera_v15_monitored.py:455` | `def __init__(self, save_dir)` |
| `_generate_recommendations` | method | `physio_chimera_v15_monitored.py:634` | `def _generate_recommendations(self, diagnostics)` |
| `calculate_health_metrics` | method | `physio_chimera_v15_monitored.py:166` | `def calculate_health_metrics(self)` |
| `create_final_report` | method | `physio_chimera_v15_monitored.py:551` | `def create_final_report(self, final_metrics, diagnostics)` |
| `forward` | method | `physio_chimera_v15_monitored.py:306` | `def forward(self, x)` |
| `forward` | method | `physio_chimera_v15_monitored.py:332` | `def forward(self, x, global_step)` |
| `forward` | method | `physio_chimera_v15_monitored.py:362` | `def forward(self, x, global_step)` |
| `forward` | method | `physio_chimera_v15_monitored.py:410` | `def forward(self, x, global_step)` |
| `generate_diagnostic_report` | method | `physio_chimera_v15_monitored.py:209` | `def generate_diagnostic_report(self, step, phase)` |
| `get_batch` | method | `physio_chimera_v15_monitored.py:79` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `physio_chimera_v15_monitored.py:93` | `def get_full(self)` |
| `get_recent_avg` | method | `physio_chimera_v15_monitored.py:191` | `def get_recent_avg(self, category, key, n)` |
| `get_w2` | method | `physio_chimera_v15_monitored.py:97` | `def get_w2(self)` |
| `plot_training_curves` | method | `physio_chimera_v15_monitored.py:463` | `def plot_training_curves(self, diagnostics)` |
| `run_experiment_monitored` | method | `physio_chimera_v15_monitored.py:762` | `def run_experiment_monitored()` |
| `save_metrics` | method | `physio_chimera_v15_monitored.py:279` | `def save_metrics(self, filepath)` |
| `seed_everything` | method | `physio_chimera_v15_monitored.py:58` | `def seed_everything(seed)` |
| `train_nested_monitored` | method | `physio_chimera_v15_monitored.py:658` | `def train_nested_monitored(config)` |
| `update_memory_metrics` | method | `physio_chimera_v15_monitored.py:158` | `def update_memory_metrics(self, cms_activations, hebbian_norm, forgetting_factor)` |
| `update_performance_metrics` | method | `physio_chimera_v15_monitored.py:150` | `def update_performance_metrics(self, loss, accuracy, lr)` |
| `update_physio_metrics` | method | `physio_chimera_v15_monitored.py:144` | `def update_physio_metrics(self, metabolism, sensitivity, gate)` |
| `SimpleConfig` | class | `physioneruon_simple.py:27` | `class SimpleConfig` |
| `SimpleRobustNet` | class | `physioneruon_simple.py:83` | `class SimpleRobustNet(Module)` |
| `__init__` | method | `physioneruon_simple.py:85` | `def __init__(self, config)` |
| `forward` | method | `physioneruon_simple.py:105` | `def forward(self, x)` |
| `get_dataset` | method | `physioneruon_simple.py:61` | `def get_dataset(config)` |
| `main` | method | `physioneruon_simple.py:281` | `def main()` |
| `pgd_attack` | method | `physioneruon_simple.py:117` | `def pgd_attack(model, x, y, eps, steps, step_size)` |
| `seed_everything` | method | `physioneruon_simple.py:54` | `def seed_everything(seed)` |
| `train_simple_robust` | method | `physioneruon_simple.py:154` | `def train_simple_robust(config, dataset, verbose)` |
| `HomeostaticRegulator` | class | `physioneuron_cpu_v1.py:85` | `class HomeostaticRegulator(Module)` |
| `MicroConfig` | class | `physioneuron_cpu_v1.py:33` | `class MicroConfig` |
| `MicroContinuumCell` | class | `physioneuron_cpu_v1.py:151` | `class MicroContinuumCell(Module)` |
| `MicroSupConLoss` | class | `physioneuron_cpu_v1.py:216` | `class MicroSupConLoss(Module)` |
| `MicroSymbioticBasis` | class | `physioneuron_cpu_v1.py:176` | `class MicroSymbioticBasis(Module)` |
| `MicroTopoBrain` | class | `physioneuron_cpu_v1.py:240` | `class MicroTopoBrain(Module)` |
| `MicroTopology` | class | `physioneuron_cpu_v1.py:196` | `class MicroTopology` |
| `PhysioNeuron` | class | `physioneuron_cpu_v1.py:109` | `class PhysioNeuron(Module)` |
| `__init__` | method | `physioneuron_cpu_v1.py:86` | `def __init__(self, d_in)` |
| `__init__` | method | `physioneuron_cpu_v1.py:110` | `def __init__(self, d_in, d_out, dynamic_mode)` |
| `__init__` | method | `physioneuron_cpu_v1.py:152` | `def __init__(self, dim)` |
| `__init__` | method | `physioneuron_cpu_v1.py:177` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `physioneuron_cpu_v1.py:197` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `physioneuron_cpu_v1.py:217` | `def __init__(self, temperature)` |
| `__init__` | method | `physioneuron_cpu_v1.py:241` | `def __init__(self, config)` |
| `_init_weights` | method | `physioneuron_cpu_v1.py:278` | `def _init_weights(self)` |
| `count_parameters` | method | `physioneuron_cpu_v1.py:283` | `def count_parameters(self)` |
| `forward` | method | `physioneuron_cpu_v1.py:96` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `physioneuron_cpu_v1.py:121` | `def forward(self, x)` |
| `forward` | method | `physioneuron_cpu_v1.py:162` | `def forward(self, x, plasticity)` |
| `forward` | method | `physioneuron_cpu_v1.py:185` | `def forward(self, x)` |
| `forward` | method | `physioneuron_cpu_v1.py:222` | `def forward(self, features, labels)` |
| `forward` | method | `physioneuron_cpu_v1.py:286` | `def forward(self, x, plasticity)` |
| `generate_ablation_matrix` | method | `physioneuron_cpu_v1.py:364` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `physioneuron_cpu_v1.py:210` | `def get_adjacency(self, plasticity)` |
| `get_dataset` | method | `physioneuron_cpu_v1.py:65` | `def get_dataset(config)` |
| `micro_pgd_attack` | method | `physioneuron_cpu_v1.py:340` | `def micro_pgd_attack(model, x, y, eps, steps, plasticity)` |
| `run_ablation_study` | method | `physioneuron_cpu_v1.py:457` | `def run_ablation_study()` |
| `seed_everything` | method | `physioneuron_cpu_v1.py:57` | `def seed_everything(seed)` |
| `train_with_cv` | method | `physioneuron_cpu_v1.py:393` | `def train_with_cv(config, dataset, cv_folds)` |
| `HomeostaticRegulator` | class | `physioneuron_cpu_v2.py:92` | `class HomeostaticRegulator(Module)` |
| `MicroConfig` | class | `physioneuron_cpu_v2.py:35` | `class MicroConfig` |
| `MicroContinuumCell` | class | `physioneuron_cpu_v2.py:158` | `class MicroContinuumCell(Module)` |
| `MicroSupConLoss` | class | `physioneuron_cpu_v2.py:223` | `class MicroSupConLoss(Module)` |
| `MicroSymbioticBasis` | class | `physioneuron_cpu_v2.py:183` | `class MicroSymbioticBasis(Module)` |
| `MicroTopoBrain` | class | `physioneuron_cpu_v2.py:247` | `class MicroTopoBrain(Module)` |
| `MicroTopology` | class | `physioneuron_cpu_v2.py:203` | `class MicroTopology` |
| `PhysioNeuron` | class | `physioneuron_cpu_v2.py:116` | `class PhysioNeuron(Module)` |
| `__init__` | method | `physioneuron_cpu_v2.py:93` | `def __init__(self, d_in)` |
| `__init__` | method | `physioneuron_cpu_v2.py:117` | `def __init__(self, d_in, d_out, dynamic_mode)` |
| `__init__` | method | `physioneuron_cpu_v2.py:159` | `def __init__(self, dim)` |
| `__init__` | method | `physioneuron_cpu_v2.py:184` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `physioneuron_cpu_v2.py:204` | `def __init__(self, num_nodes, config)` |
| `__init__` | method | `physioneuron_cpu_v2.py:224` | `def __init__(self, temperature)` |
| `__init__` | method | `physioneuron_cpu_v2.py:248` | `def __init__(self, config)` |
| `_init_weights` | method | `physioneuron_cpu_v2.py:285` | `def _init_weights(self)` |
| `count_parameters` | method | `physioneuron_cpu_v2.py:290` | `def count_parameters(self)` |
| `forward` | method | `physioneuron_cpu_v2.py:103` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `physioneuron_cpu_v2.py:128` | `def forward(self, x)` |
| `forward` | method | `physioneuron_cpu_v2.py:169` | `def forward(self, x, plasticity)` |
| `forward` | method | `physioneuron_cpu_v2.py:192` | `def forward(self, x)` |
| `forward` | method | `physioneuron_cpu_v2.py:229` | `def forward(self, features, labels)` |
| `forward` | method | `physioneuron_cpu_v2.py:293` | `def forward(self, x, plasticity)` |
| `generate_ablation_matrix` | method | `physioneuron_cpu_v2.py:371` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `physioneuron_cpu_v2.py:217` | `def get_adjacency(self, plasticity)` |
| `get_dataset` | method | `physioneuron_cpu_v2.py:72` | `def get_dataset(config)` |
| `micro_pgd_attack` | method | `physioneuron_cpu_v2.py:347` | `def micro_pgd_attack(model, x, y, eps, steps, plasticity)` |
| `run_ablation_study` | method | `physioneuron_cpu_v2.py:464` | `def run_ablation_study()` |
| `seed_everything` | method | `physioneuron_cpu_v2.py:64` | `def seed_everything(seed)` |
| `train_with_cv` | method | `physioneuron_cpu_v2.py:400` | `def train_with_cv(config, dataset, cv_folds)` |
| `AdaptiveTopology` | class | `physioneuron_cpu_v3.py:222` | `class AdaptiveTopology(Module)` |
| `AdvancedHomeostaticCell` | class | `physioneuron_cpu_v3.py:162` | `class AdvancedHomeostaticCell(Module)` |
| `EliteConfig` | class | `physioneuron_cpu_v3.py:35` | `class EliteConfig` |
| `EliteTopoBrain` | class | `physioneuron_cpu_v3.py:260` | `class EliteTopoBrain(Module)` |
| `EpisodicMemory` | class | `physioneuron_cpu_v3.py:102` | `class EpisodicMemory(Module)` |
| `SpectralNormLinear` | class | `physioneuron_cpu_v3.py:136` | `class SpectralNormLinear(Module)` |
| `SupConLoss` | class | `physioneuron_cpu_v3.py:385` | `class SupConLoss(Module)` |
| `__init__` | method | `physioneuron_cpu_v3.py:104` | `def __init__(self, dim, capacity)` |
| `__init__` | method | `physioneuron_cpu_v3.py:138` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `physioneuron_cpu_v3.py:164` | `def __init__(self, d_in, d_out, use_spectral)` |
| `__init__` | method | `physioneuron_cpu_v3.py:224` | `def __init__(self, num_nodes, grid_size)` |
| `__init__` | method | `physioneuron_cpu_v3.py:261` | `def __init__(self, config)` |
| `__init__` | method | `physioneuron_cpu_v3.py:386` | `def __init__(self, temperature)` |
| `count_parameters` | method | `physioneuron_cpu_v3.py:303` | `def count_parameters(self)` |
| `elite_pgd_attack` | method | `physioneuron_cpu_v3.py:348` | `def elite_pgd_attack(model, x, y, eps, steps, stress)` |
| `forward` | method | `physioneuron_cpu_v3.py:152` | `def forward(self, x)` |
| `forward` | method | `physioneuron_cpu_v3.py:194` | `def forward(self, x)` |
| `forward` | method | `physioneuron_cpu_v3.py:248` | `def forward(self, stress)` |
| `forward` | method | `physioneuron_cpu_v3.py:306` | `def forward(self, x, stress)` |
| `forward` | method | `physioneuron_cpu_v3.py:390` | `def forward(self, features, labels)` |
| `get_elite_dataset` | method | `physioneuron_cpu_v3.py:79` | `def get_elite_dataset(config)` |
| `power_iteration` | method | `physioneuron_cpu_v3.py:145` | `def power_iteration(self, n_iter)` |
| `retrieve` | method | `physioneuron_cpu_v3.py:125` | `def retrieve(self, x, k)` |
| `run_elite_experiment` | method | `physioneuron_cpu_v3.py:542` | `def run_elite_experiment()` |
| `seed_everything` | method | `physioneuron_cpu_v3.py:71` | `def seed_everything(seed)` |
| `train_elite_model` | method | `physioneuron_cpu_v3.py:430` | `def train_elite_model(config, dataset, fold_results)` |
| `update` | method | `physioneuron_cpu_v3.py:112` | `def update(self, x, y)` |
| `ConsciousnessModule` | class | `poke_cifar.py:116` | `class ConsciousnessModule(Module)` |
| `DualSystemModule` | class | `poke_cifar.py:98` | `class DualSystemModule(Module)` |
| `OmniBrainCIFAR` | class | `poke_cifar.py:133` | `class OmniBrainCIFAR(Module)` |
| `PTSymmetricLayer` | class | `poke_cifar.py:56` | `class PTSymmetricLayer(Module)` |
| `TopologicalLayer` | class | `poke_cifar.py:77` | `class TopologicalLayer(Module)` |
| `__init__` | method | `poke_cifar.py:57` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `poke_cifar.py:78` | `def __init__(self, in_f, out_f, density)` |
| `__init__` | method | `poke_cifar.py:99` | `def __init__(self, features)` |
| `__init__` | method | `poke_cifar.py:117` | `def __init__(self, features)` |
| `__init__` | method | `poke_cifar.py:134` | `def __init__(self)` |
| `_update_mask` | method | `poke_cifar.py:87` | `def _update_mask(self)` |
| `compute_phi_effective` | function | `poke_cifar.py:35` | `def compute_phi_effective(activity)` |
| `demo_inference` | method | `poke_cifar.py:300` | `def demo_inference(model, loader)` |
| `evaluate` | method | `poke_cifar.py:200` | `def evaluate(model, loader, device)` |
| `forward` | method | `poke_cifar.py:66` | `def forward(self, x)` |
| `forward` | method | `poke_cifar.py:93` | `def forward(self, x)` |
| `forward` | method | `poke_cifar.py:107` | `def forward(self, x)` |
| `forward` | method | `poke_cifar.py:123` | `def forward(self, x)` |
| `forward` | method | `poke_cifar.py:161` | `def forward(self, x)` |
| `get_cifar10_loaders` | method | `poke_cifar.py:180` | `def get_cifar10_loaders(batch_size)` |
| `main` | method | `poke_cifar.py:221` | `def main()` |
| `plot_history` | method | `poke_cifar.py:287` | `def plot_history(hist)` |
| `ConsciousnessModule` | class | `poke_cifar2.py:117` | `class ConsciousnessModule(Module)` |
| `DualSystemModule` | class | `poke_cifar2.py:99` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `poke_cifar2.py:49` | `class FastSlowLinear(Module)` |
| `OmniBrainFastSlow` | class | `poke_cifar2.py:131` | `class OmniBrainFastSlow(Module)` |
| `__init__` | method | `poke_cifar2.py:50` | `def __init__(self, in_features, out_features, fast_lr)` |
| `__init__` | method | `poke_cifar2.py:100` | `def __init__(self, dim)` |
| `__init__` | method | `poke_cifar2.py:118` | `def __init__(self, dim)` |
| `__init__` | method | `poke_cifar2.py:132` | `def __init__(self)` |
| `compute_phi_effective` | function | `poke_cifar2.py:28` | `def compute_phi_effective(activity)` |
| `end_of_batch` | method | `poke_cifar2.py:89` | `def end_of_batch(self)` |
| `evaluate` | method | `poke_cifar2.py:191` | `def evaluate(model, loader, device)` |
| `forward` | method | `poke_cifar2.py:76` | `def forward(self, x)` |
| `forward` | method | `poke_cifar2.py:108` | `def forward(self, x)` |
| `forward` | method | `poke_cifar2.py:124` | `def forward(self, x)` |
| `forward` | method | `poke_cifar2.py:151` | `def forward(self, x)` |
| `get_cifar10_loaders` | method | `poke_cifar2.py:175` | `def get_cifar10_loaders(batch_size)` |
| `get_fast_norms` | method | `poke_cifar2.py:164` | `def get_fast_norms(self)` |
| `get_fast_weight_norm` | method | `poke_cifar2.py:92` | `def get_fast_weight_norm(self)` |
| `reset_all_fast_weights` | method | `poke_cifar2.py:159` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `poke_cifar2.py:65` | `def reset_fast_weights(self)` |
| `train` | method | `poke_cifar2.py:212` | `def train()` |
| `update_fast_weights` | method | `poke_cifar2.py:69` | `def update_fast_weights(self, x)` |
| `ConsciousnessModule` | class | `pokemon3.py:164` | `class ConsciousnessModule(Module)` |
| `DualSystemModule` | class | `pokemon3.py:134` | `class DualSystemModule(Module)` |
| `HomeostasisContext` | class | `pokemon3.py:76` | `class HomeostasisContext` |
| `OmniBrain` | class | `pokemon3.py:187` | `class OmniBrain(Module)` |
| `PTSymmetricLayer` | class | `pokemon3.py:82` | `class PTSymmetricLayer(Module)` |
| `TopologicalLayer` | class | `pokemon3.py:109` | `class TopologicalLayer(Module)` |
| `__init__` | method | `pokemon3.py:85` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `pokemon3.py:112` | `def __init__(self, in_features, out_features, connectivity)` |
| `__init__` | method | `pokemon3.py:137` | `def __init__(self, features)` |
| `__init__` | method | `pokemon3.py:167` | `def __init__(self, features)` |
| `__init__` | method | `pokemon3.py:190` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `compute_phi_effective_approx` | function | `pokemon3.py:30` | `def compute_phi_effective_approx(activity)` |
| `compute_pt_phase` | method | `pokemon3.py:94` | `def compute_pt_phase(self)` |
| `demonstrate_inference` | method | `pokemon3.py:435` | `def demonstrate_inference(model, test_loader, device)` |
| `estimate_energy_consumption` | function | `pokemon3.py:63` | `def estimate_energy_consumption(model, batch_size)` |
| `evaluate_model` | method | `pokemon3.py:380` | `def evaluate_model(model, test_loader, device, criterion)` |
| `final_report` | method | `pokemon3.py:467` | `def final_report(model, history)` |
| `forward` | method | `pokemon3.py:102` | `def forward(self, x)` |
| `forward` | method | `pokemon3.py:128` | `def forward(self, x)` |
| `forward` | method | `pokemon3.py:153` | `def forward(self, x)` |
| `forward` | method | `pokemon3.py:177` | `def forward(self, x)` |
| `forward` | method | `pokemon3.py:217` | `def forward(self, x)` |
| `generate_evolution_plots` | method | `pokemon3.py:402` | `def generate_evolution_plots(history, epochs)` |
| `prepare_mnist_data` | method | `pokemon3.py:245` | `def prepare_mnist_data(batch_size, device)` |
| `train_omni_brain` | method | `pokemon3.py:269` | `def train_omni_brain(model, train_loader, test_loader, epochs, device)` |
| `update_topology` | method | `pokemon3.py:121` | `def update_topology(self, connectivity)` |
| `update_topology` | method | `pokemon3.py:235` | `def update_topology(self, current_connectivity)` |
| `ConsciousnessModule` | class | `pokemon4.py:146` | `class ConsciousnessModule(Module)` |
| `DualSystemModule` | class | `pokemon4.py:113` | `class DualSystemModule(Module)` |
| `OmniBrain` | class | `pokemon4.py:168` | `class OmniBrain(Module)` |
| `PTSymmetricLayer` | class | `pokemon4.py:68` | `class PTSymmetricLayer(Module)` |
| `TopologicalLayer` | class | `pokemon4.py:91` | `class TopologicalLayer(Module)` |
| `__init__` | method | `pokemon4.py:69` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `pokemon4.py:92` | `def __init__(self, in_features, out_features, target_density)` |
| `__init__` | method | `pokemon4.py:114` | `def __init__(self, features)` |
| `__init__` | method | `pokemon4.py:147` | `def __init__(self, features)` |
| `__init__` | method | `pokemon4.py:169` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `_update_mask` | method | `pokemon4.py:101` | `def _update_mask(self)` |
| `compute_phi_effective` | function | `pokemon4.py:41` | `def compute_phi_effective(activity)` |
| `evaluate` | method | `pokemon4.py:218` | `def evaluate(model, loader, device)` |
| `forward` | method | `pokemon4.py:78` | `def forward(self, x)` |
| `forward` | method | `pokemon4.py:107` | `def forward(self, x)` |
| `forward` | method | `pokemon4.py:130` | `def forward(self, x)` |
| `forward` | method | `pokemon4.py:157` | `def forward(self, x)` |
| `forward` | method | `pokemon4.py:186` | `def forward(self, x)` |
| `get_mnist_loaders` | method | `pokemon4.py:206` | `def get_mnist_loaders(batch_size)` |
| `plot_history` | method | `pokemon4.py:302` | `def plot_history(hist)` |
| `train_and_evaluate` | method | `pokemon4.py:236` | `def train_and_evaluate()` |
| `ChampionConfig` | class | `pokemon_battle_champion.py:31` | `class ChampionConfig` |
| `PokemonBattleChampion` | class | `pokemon_battle_champion.py:41` | `class PokemonBattleChampion(Module)` |
| `__init__` | method | `pokemon_battle_champion.py:44` | `def __init__(self, config)` |
| `battle_training_epoch` | method | `pokemon_battle_champion.py:161` | `def battle_training_epoch(model, loader, optimizer, criterion, epoch)` |
| `create_battle_dataset` | method | `pokemon_battle_champion.py:128` | `def create_battle_dataset(config)` |
| `create_epic_battle_visualization` | method | `pokemon_battle_champion.py:334` | `def create_epic_battle_visualization(battle_history, historical_results)` |
| `evaluate_battle_champion` | method | `pokemon_battle_champion.py:192` | `def evaluate_battle_champion(model, loader)` |
| `forward` | method | `pokemon_battle_champion.py:99` | `def forward(self, x)` |
| `run_epic_pokemon_battle` | method | `pokemon_battle_champion.py:208` | `def run_epic_pokemon_battle()` |
| `save_battle_results` | method | `pokemon_battle_champion.py:440` | `def save_battle_results(battle_history, historical_results, champion_model)` |
| `AdaptiveTopologyLayer` | class | `pokemon_hybrid_synergy_ablation.py:190` | `class AdaptiveTopologyLayer(Module)` |
| `AdvancedModel` | class | `pokemon_hybrid_synergy_ablation.py:463` | `class AdvancedModel(Module)` |
| `BaselineModel` | class | `pokemon_hybrid_synergy_ablation.py:428` | `class BaselineModel(Module)` |
| `HybridModel` | class | `pokemon_hybrid_synergy_ablation.py:443` | `class HybridModel(Module)` |
| `PokemonSynergyModel` | class | `pokemon_hybrid_synergy_ablation.py:270` | `class PokemonSynergyModel(Module)` |
| `SynergyAblationStudy` | class | `pokemon_hybrid_synergy_ablation.py:386` | `class SynergyAblationStudy` |
| `SynergyAttentionLayer` | class | `pokemon_hybrid_synergy_ablation.py:125` | `class SynergyAttentionLayer(Module)` |
| `SynergyConfig` | class | `pokemon_hybrid_synergy_ablation.py:37` | `class SynergyConfig` |
| `SynergyGANLayer` | class | `pokemon_hybrid_synergy_ablation.py:157` | `class SynergyGANLayer(Module)` |
| `SynergyVAELayer` | class | `pokemon_hybrid_synergy_ablation.py:82` | `class SynergyVAELayer(Module)` |
| `__init__` | method | `pokemon_hybrid_synergy_ablation.py:84` | `def __init__(self, input_dim, hidden_dim, latent_dim)` |
| `__init__` | method | `pokemon_hybrid_synergy_ablation.py:127` | `def __init__(self, d_model, num_heads, d_ff)` |
| `__init__` | method | `pokemon_hybrid_synergy_ablation.py:159` | `def __init__(self, input_dim, latent_dim, hidden_dim)` |
| `__init__` | method | `pokemon_hybrid_synergy_ablation.py:192` | `def __init__(self, grid_size, embed_dim, sparsity)` |
| `__init__` | method | `pokemon_hybrid_synergy_ablation.py:273` | `def __init__(self, config)` |
| `__init__` | method | `pokemon_hybrid_synergy_ablation.py:389` | `def __init__(self, config)` |
| `__init__` | method | `pokemon_hybrid_synergy_ablation.py:429` | `def __init__(self, config)` |
| `__init__` | method | `pokemon_hybrid_synergy_ablation.py:444` | `def __init__(self, config)` |
| `__init__` | method | `pokemon_hybrid_synergy_ablation.py:464` | `def __init__(self, config)` |
| `analyze_synergy_results` | method | `pokemon_hybrid_synergy_ablation.py:637` | `def analyze_synergy_results(results)` |
| `create_synergy_visualizations` | method | `pokemon_hybrid_synergy_ablation.py:696` | `def create_synergy_visualizations(results, output_dir)` |
| `create_variant_model` | method | `pokemon_hybrid_synergy_ablation.py:424` | `def create_variant_model(self, level_name)` |
| `discriminate` | method | `pokemon_hybrid_synergy_ablation.py:187` | `def discriminate(self, x)` |
| `forward` | method | `pokemon_hybrid_synergy_ablation.py:116` | `def forward(self, x, return_encoding)` |
| `forward` | method | `pokemon_hybrid_synergy_ablation.py:149` | `def forward(self, x)` |
| `forward` | method | `pokemon_hybrid_synergy_ablation.py:234` | `def forward(self, x)` |
| `forward` | method | `pokemon_hybrid_synergy_ablation.py:327` | `def forward(self, x, return_all)` |
| `forward` | method | `pokemon_hybrid_synergy_ablation.py:434` | `def forward(self, x)` |
| `forward` | method | `pokemon_hybrid_synergy_ablation.py:450` | `def forward(self, x)` |
| `forward` | method | `pokemon_hybrid_synergy_ablation.py:471` | `def forward(self, x)` |
| `generate` | method | `pokemon_hybrid_synergy_ablation.py:184` | `def generate(self, z)` |
| `get_ablation_matrix` | method | `pokemon_hybrid_synergy_ablation.py:393` | `def get_ablation_matrix(self)` |
| `get_adjacency_matrix` | method | `pokemon_hybrid_synergy_ablation.py:220` | `def get_adjacency_matrix(self)` |
| `reparameterize` | method | `pokemon_hybrid_synergy_ablation.py:111` | `def reparameterize(self, mu, logvar)` |
| `run_synergy_ablation` | method | `pokemon_hybrid_synergy_ablation.py:495` | `def run_synergy_ablation()` |
| `to_dict` | method | `pokemon_hybrid_synergy_ablation.py:75` | `def to_dict(self)` |
| `ComponentState` | class | `premium_synergy_demo.py:20` | `class ComponentState` |
| `DemocraticDecision` | class | `premium_synergy_demo.py:29` | `class DemocraticDecision` |
| `HomeostaticMotor` | class | `premium_synergy_demo.py:160` | `class HomeostaticMotor` |
| `OmniBrainComponent` | class | `premium_synergy_demo.py:75` | `class OmniBrainComponent` |
| `PremiumSynergySystem` | class | `premium_synergy_demo.py:225` | `class PremiumSynergySystem` |
| `QuimeraComponent` | class | `premium_synergy_demo.py:116` | `class QuimeraComponent` |
| `TopoBrainComponent` | class | `premium_synergy_demo.py:36` | `class TopoBrainComponent` |
| `__init__` | method | `premium_synergy_demo.py:39` | `def __init__(self)` |
| `__init__` | method | `premium_synergy_demo.py:78` | `def __init__(self)` |
| `__init__` | method | `premium_synergy_demo.py:119` | `def __init__(self)` |
| `__init__` | method | `premium_synergy_demo.py:163` | `def __init__(self, threshold, convergence_epochs)` |
| `__init__` | method | `premium_synergy_demo.py:228` | `def __init__(self)` |
| `calculate_target_accuracy` | method | `premium_synergy_demo.py:296` | `def calculate_target_accuracy(self)` |
| `deliberate` | method | `premium_synergy_demo.py:171` | `def deliberate(self, components, target_accuracy)` |
| `process` | method | `premium_synergy_demo.py:48` | `def process(self, input_data, plasticity)` |
| `process` | method | `premium_synergy_demo.py:87` | `def process(self, input_data, chaos_level)` |
| `process` | method | `premium_synergy_demo.py:128` | `def process(self, input_data, plasticity, chaos)` |
| `process_epoch` | method | `premium_synergy_demo.py:241` | `def process_epoch(self, input_data, chaos_level)` |
| `run_demo` | method | `premium_synergy_demo.py:302` | `def run_demo()` |
| `AttentionController` | class | `premium_synergy_democratic.py:798` | `class AttentionController(Module)` |
| `ChaosModulator` | class | `premium_synergy_democratic.py:648` | `class ChaosModulator(Module)` |
| `DualPhaseMemory` | class | `premium_synergy_democratic.py:738` | `class DualPhaseMemory(Module)` |
| `DualSystemModule` | class | `premium_synergy_democratic.py:595` | `class DualSystemModule(Module)` |
| `DynamicTopologyGrid` | class | `premium_synergy_democratic.py:492` | `class DynamicTopologyGrid(Module)` |
| `FastSlowLinear` | class | `premium_synergy_democratic.py:565` | `class FastSlowLinear(Module)` |
| `HomeostaticMotor` | class | `premium_synergy_democratic.py:836` | `class HomeostaticMotor(Module)` |
| `IntegrationModule` | class | `premium_synergy_democratic.py:540` | `class IntegrationModule(Module)` |
| `IntegrativeControl` | class | `premium_synergy_democratic.py:615` | `class IntegrativeControl(Module)` |
| `LiquidNeuron` | class | `premium_synergy_democratic.py:682` | `class LiquidNeuron(Module)` |
| `MemoryChecker` | class | `premium_synergy_democratic.py:94` | `class MemoryChecker` |
| `MetabolismRegulator` | class | `premium_synergy_democratic.py:419` | `class MetabolismRegulator(Module)` |
| `OmniBrainComponent` | class | `premium_synergy_democratic.py:235` | `class OmniBrainComponent(Module)` |
| `PhaseRegulator` | class | `premium_synergy_democratic.py:765` | `class PhaseRegulator(Module)` |
| `PremiumSynergyConfig` | class | `premium_synergy_democratic.py:42` | `class PremiumSynergyConfig` |
| `PremiumSynergyModel` | class | `premium_synergy_democratic.py:960` | `class PremiumSynergyModel(Module)` |
| `QuimeraComponent` | class | `premium_synergy_democratic.py:325` | `class QuimeraComponent(Module)` |
| `SensitivityGate` | class | `premium_synergy_democratic.py:459` | `class SensitivityGate(Module)` |
| `SovereignAttention` | class | `premium_synergy_democratic.py:712` | `class SovereignAttention(Module)` |
| `SymbioticBasis` | class | `premium_synergy_democratic.py:519` | `class SymbioticBasis(Module)` |
| `SystemRegulator` | class | `premium_synergy_democratic.py:1074` | `class SystemRegulator(Module)` |
| `TopoBrainComponent` | class | `premium_synergy_democratic.py:139` | `class TopoBrainComponent(Module)` |
| `__init__` | method | `premium_synergy_democratic.py:97` | `def __init__(self, max_memory_gb)` |
| `__init__` | method | `premium_synergy_democratic.py:142` | `def __init__(self, config)` |
| `__init__` | method | `premium_synergy_democratic.py:238` | `def __init__(self, config)` |
| `__init__` | method | `premium_synergy_democratic.py:328` | `def __init__(self, config)` |
| `__init__` | method | `premium_synergy_democratic.py:421` | `def __init__(self, dim)` |
| `__init__` | method | `premium_synergy_democratic.py:461` | `def __init__(self, dim)` |
| `__init__` | method | `premium_synergy_democratic.py:494` | `def __init__(self, num_nodes, grid_size)` |
| `__init__` | method | `premium_synergy_democratic.py:521` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `premium_synergy_democratic.py:542` | `def __init__(self, dim)` |
| `__init__` | method | `premium_synergy_democratic.py:567` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `premium_synergy_democratic.py:597` | `def __init__(self, dim)` |
| `__init__` | method | `premium_synergy_democratic.py:617` | `def __init__(self, dim)` |
| `__init__` | method | `premium_synergy_democratic.py:650` | `def __init__(self, dim)` |
| `__init__` | method | `premium_synergy_democratic.py:684` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `premium_synergy_democratic.py:714` | `def __init__(self, dim)` |
| `__init__` | method | `premium_synergy_democratic.py:740` | `def __init__(self, dim)` |
| `__init__` | method | `premium_synergy_democratic.py:767` | `def __init__(self, dim)` |
| `__init__` | method | `premium_synergy_democratic.py:800` | `def __init__(self, dim)` |
| `__init__` | method | `premium_synergy_democratic.py:839` | `def __init__(self, config)` |
| `__init__` | method | `premium_synergy_democratic.py:963` | `def __init__(self, config)` |
| `__init__` | method | `premium_synergy_democratic.py:1076` | `def __init__(self, dim)` |
| `_create_grid_mask` | method | `premium_synergy_democratic.py:501` | `def _create_grid_mask(self)` |
| `adjust_for_convergence` | method | `premium_synergy_democratic.py:926` | `def adjust_for_convergence(self, performance_metrics)` |
| `check_memory` | method | `premium_synergy_democratic.py:101` | `def check_memory(self)` |
| `consolidate` | method | `premium_synergy_democratic.py:409` | `def consolidate(self)` |
| `consolidate_svd` | method | `premium_synergy_democratic.py:703` | `def consolidate_svd(self, strength)` |
| `create_dataloader` | method | `premium_synergy_democratic.py:1272` | `def create_dataloader(X, y, batch_size, shuffle)` |
| `create_synthetic_dataset` | method | `premium_synergy_democratic.py:1118` | `def create_synthetic_dataset(config)` |
| `democratic_deliberation_status` | method | `premium_synergy_democratic.py:1061` | `def democratic_deliberation_status(self)` |
| `ensure_dependencies` | method | `premium_synergy_democratic.py:1108` | `def ensure_dependencies()` |
| `forward` | method | `premium_synergy_democratic.py:175` | `def forward(self, x, plasticity)` |
| `forward` | method | `premium_synergy_democratic.py:268` | `def forward(self, x, chaos_level)` |
| `forward` | method | `premium_synergy_democratic.py:358` | `def forward(self, x, plasticity, chaos)` |
| `forward` | method | `premium_synergy_democratic.py:431` | `def forward(self, x)` |
| `forward` | method | `premium_synergy_democratic.py:471` | `def forward(self, x)` |
| `forward` | method | `premium_synergy_democratic.py:531` | `def forward(self, x)` |
| `forward` | method | `premium_synergy_democratic.py:552` | `def forward(self, x)` |
| `forward` | method | `premium_synergy_democratic.py:579` | `def forward(self, x)` |
| `forward` | method | `premium_synergy_democratic.py:604` | `def forward(self, x)` |
| `forward` | method | `premium_synergy_democratic.py:627` | `def forward(self, x)` |
| `forward` | method | `premium_synergy_democratic.py:660` | `def forward(self, x, chaos_level)` |
| `forward` | method | `premium_synergy_democratic.py:692` | `def forward(self, x, plasticity)` |
| `forward` | method | `premium_synergy_democratic.py:721` | `def forward(self, x, is_chaos)` |
| `forward` | method | `premium_synergy_democratic.py:746` | `def forward(self, x, phase_idx)` |
| `forward` | method | `premium_synergy_democratic.py:777` | `def forward(self, x)` |
| `forward` | method | `premium_synergy_democratic.py:810` | `def forward(self, x, chaos)` |
| `forward` | method | `premium_synergy_democratic.py:861` | `def forward(self, topobrain_out, omnibrain_out, quimera_out, target_accuracy)` |
| `forward` | method | `premium_synergy_democratic.py:989` | `def forward(self, x, chaos_level)` |
| `forward` | method | `premium_synergy_democratic.py:1086` | `def forward(self, x)` |
| `get_adjacency` | method | `premium_synergy_democratic.py:514` | `def get_adjacency(self, plasticity)` |
| `get_balance` | method | `premium_synergy_democratic.py:612` | `def get_balance(self)` |
| `get_coherence` | method | `premium_synergy_democratic.py:762` | `def get_coherence(self)` |
| `get_control` | method | `premium_synergy_democratic.py:829` | `def get_control(self)` |
| `get_level` | method | `premium_synergy_democratic.py:489` | `def get_level(self)` |
| `get_level` | method | `premium_synergy_democratic.py:562` | `def get_level(self)` |
| `get_level` | method | `premium_synergy_democratic.py:795` | `def get_level(self)` |
| `get_metrics` | method | `premium_synergy_democratic.py:733` | `def get_metrics(self)` |
| `get_resistance` | method | `premium_synergy_democratic.py:678` | `def get_resistance(self)` |
| `get_state` | method | `premium_synergy_democratic.py:456` | `def get_state(self)` |
| `get_state` | method | `premium_synergy_democratic.py:645` | `def get_state(self)` |
| `internal_dialogue` | method | `premium_synergy_democratic.py:222` | `def internal_dialogue(self)` |
| `internal_dialogue` | method | `premium_synergy_democratic.py:312` | `def internal_dialogue(self)` |
| `internal_dialogue` | method | `premium_synergy_democratic.py:400` | `def internal_dialogue(self)` |
| `main` | method | `premium_synergy_democratic.py:1284` | `def main()` |
| `train_premium_synergy` | method | `premium_synergy_democratic.py:1148` | `def train_premium_synergy(config)` |
| `update` | method | `premium_synergy_democratic.py:754` | `def update(self, x, phase_idx)` |
| `warn_if_high` | method | `premium_synergy_democratic.py:126` | `def warn_if_high(self)` |
| `Config` | class | `quen7.py:33` | `class Config` |
| `DataEnvironment` | class | `quen7.py:55` | `class DataEnvironment` |
| `HomeostaticRegulator` | class | `quen7.py:89` | `class HomeostaticRegulator(Module)` |
| `MicroTopoBrain` | class | `quen7.py:173` | `class MicroTopoBrain(Module)` |
| `NeuralDiagnostics` | class | `quen7.py:215` | `class NeuralDiagnostics` |
| `PhysioNeuron` | class | `quen7.py:115` | `class PhysioNeuron(Module)` |
| `SupConHead` | class | `quen7.py:158` | `class SupConHead(Module)` |
| `__init__` | method | `quen7.py:56` | `def __init__(self)` |
| `__init__` | method | `quen7.py:90` | `def __init__(self, d_in)` |
| `__init__` | method | `quen7.py:116` | `def __init__(self, d_in, d_out, dynamic)` |
| `__init__` | method | `quen7.py:159` | `def __init__(self, in_dim)` |
| `__init__` | method | `quen7.py:174` | `def __init__(self, config)` |
| `__init__` | method | `quen7.py:216` | `def __init__(self)` |
| `count_parameters` | method | `quen7.py:187` | `def count_parameters(self)` |
| `forward` | method | `quen7.py:100` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `quen7.py:128` | `def forward(self, x)` |
| `forward` | method | `quen7.py:167` | `def forward(self, x)` |
| `forward` | method | `quen7.py:190` | `def forward(self, x)` |
| `get_batch` | method | `quen7.py:66` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `quen7.py:80` | `def get_full(self)` |
| `get_recent_avg` | method | `quen7.py:234` | `def get_recent_avg(self, key, n)` |
| `get_w2` | method | `quen7.py:83` | `def get_w2(self)` |
| `report` | method | `quen7.py:239` | `def report(self, step, phase)` |
| `run_ablation_study` | method | `quen7.py:330` | `def run_ablation_study()` |
| `seed_everything` | method | `quen7.py:45` | `def seed_everything(seed)` |
| `train_nonstationary` | method | `quen7.py:265` | `def train_nonstationary(config)` |
| `update` | method | `quen7.py:226` | `def update(self, loss, liquid_norm, physio, prediction_error)` |
| `ChimeraScientificConfig` | class | `quimera.py:19` | `class ChimeraScientificConfig` |
| `Chimera_v9_Scientific` | class | `quimera.py:201` | `class Chimera_v9_Scientific(Module)` |
| `DualPhaseMemory` | class | `quimera.py:176` | `class DualPhaseMemory(Module)` |
| `LiquidNeuron` | class | `quimera.py:114` | `class LiquidNeuron(Module)` |
| `RealWorldEnvironment` | class | `quimera.py:86` | `class RealWorldEnvironment` |
| `SovereignAttention` | class | `quimera.py:153` | `class SovereignAttention(Module)` |
| `__init__` | method | `quimera.py:87` | `def __init__(self)` |
| `__init__` | method | `quimera.py:116` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `quimera.py:155` | `def __init__(self, dim)` |
| `__init__` | method | `quimera.py:178` | `def __init__(self, dim)` |
| `__init__` | method | `quimera.py:202` | `def __init__(self, config)` |
| `consolidate` | method | `quimera.py:267` | `def consolidate(self)` |
| `consolidate_svd` | method | `quimera.py:138` | `def consolidate_svd(self, strength)` |
| `forward` | method | `quimera.py:124` | `def forward(self, x, plasticity)` |
| `forward` | method | `quimera.py:165` | `def forward(self, x, is_chaos)` |
| `forward` | method | `quimera.py:184` | `def forward(self, x, phase_idx)` |
| `forward` | method | `quimera.py:229` | `def forward(self, x, phase_idx)` |
| `generate_chimera_matrix` | method | `quimera.py:361` | `def generate_chimera_matrix()` |
| `get_batch` | method | `quimera.py:97` | `def get_batch(self, phase, batch_size)` |
| `get_structure_entropy` | method | `quimera.py:62` | `def get_structure_entropy(model)` |
| `measure_spatial_richness` | method | `quimera.py:47` | `def measure_spatial_richness(activations)` |
| `run_scientific_study` | method | `quimera.py:399` | `def run_scientific_study()` |
| `seed_everything` | method | `quimera.py:37` | `def seed_everything(seed)` |
| `train_chimera_scientific` | method | `quimera.py:279` | `def train_chimera_scientific(config, verbose)` |
| `update` | method | `quimera.py:191` | `def update(self, x, phase_idx)` |
| `AudioEncoder` | class | `quimera_vision.py:108` | `class AudioEncoder(Module)` |
| `Decoder` | class | `quimera_vision.py:121` | `class Decoder(Module)` |
| `Flickr8kMMDataset` | class | `quimera_vision.py:43` | `class Flickr8kMMDataset(Dataset)` |
| `ImgEncoder` | class | `quimera_vision.py:96` | `class ImgEncoder(Module)` |
| `__getitem__` | method | `quimera_vision.py:64` | `def __getitem__(self, idx)` |
| `__init__` | method | `quimera_vision.py:44` | `def __init__(self)` |
| `__init__` | method | `quimera_vision.py:97` | `def __init__(self)` |
| `__init__` | method | `quimera_vision.py:109` | `def __init__(self)` |
| `__init__` | method | `quimera_vision.py:122` | `def __init__(self)` |
| `__len__` | method | `quimera_vision.py:62` | `def __len__(self)` |
| `collate` | method | `quimera_vision.py:87` | `def collate(batch)` |
| `forward` | method | `quimera_vision.py:103` | `def forward(self, x)` |
| `forward` | method | `quimera_vision.py:116` | `def forward(self, x)` |
| `forward` | method | `quimera_vision.py:127` | `def forward(self, img, audio, seq)` |
| `generate_caption` | method | `quimera_vision.py:168` | `def generate_caption(img_path, audio_path)` |
| `text_to_seq` | function | `quimera_vision.py:37` | `def text_to_seq(text)` |
| `HomeostaticOrchestrator` | class | `qwen.py:83` | `class HomeostaticOrchestrator(Module)` |
| `MicroConfig` | class | `qwen.py:35` | `class MicroConfig` |
| `MicroTopoBrain` | class | `qwen.py:196` | `class MicroTopoBrain(Module)` |
| `RegulableContinuum` | class | `qwen.py:121` | `class RegulableContinuum(Module)` |
| `RegulableSupConHead` | class | `qwen.py:181` | `class RegulableSupConHead(Module)` |
| `RegulableSymbiotic` | class | `qwen.py:143` | `class RegulableSymbiotic(Module)` |
| `RegulableTopology` | class | `qwen.py:162` | `class RegulableTopology` |
| `__init__` | method | `qwen.py:89` | `def __init__(self)` |
| `__init__` | method | `qwen.py:122` | `def __init__(self, dim)` |
| `__init__` | method | `qwen.py:144` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `qwen.py:163` | `def __init__(self, num_nodes)` |
| `__init__` | method | `qwen.py:182` | `def __init__(self, in_dim)` |
| `__init__` | method | `qwen.py:197` | `def __init__(self, config)` |
| `_init_weights` | method | `qwen.py:215` | `def _init_weights(self)` |
| `count_parameters` | method | `qwen.py:220` | `def count_parameters(self)` |
| `forward` | method | `qwen.py:99` | `def forward(self, x, logits, h_agg, h_proc, w_norm, entropy, ortho, pgd_loss)` |
| `forward` | method | `qwen.py:132` | `def forward(self, x, strength)` |
| `forward` | method | `qwen.py:151` | `def forward(self, x, influence)` |
| `forward` | method | `qwen.py:190` | `def forward(self, x, gain)` |
| `forward` | method | `qwen.py:223` | `def forward(self, x, pgd_loss)` |
| `generate_ablation_matrix` | method | `qwen.py:367` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `qwen.py:176` | `def get_adjacency(self, plasticity)` |
| `get_dataset` | method | `qwen.py:64` | `def get_dataset(config)` |
| `micro_pgd_attack` | method | `qwen.py:283` | `def micro_pgd_attack(model, x, y, eps, steps, pgd_loss)` |
| `run_ablation_study` | method | `qwen.py:392` | `def run_ablation_study()` |
| `seed_everything` | method | `qwen.py:57` | `def seed_everything(seed)` |
| `train_with_cv` | method | `qwen.py:306` | `def train_with_cv(config, dataset, cv_folds)` |
| `Config` | class | `qwen3.py:36` | `class Config` |
| `DataEnvironment` | class | `qwen3.py:60` | `class DataEnvironment` |
| `HomeostaticRegulator` | class | `qwen3.py:94` | `class HomeostaticRegulator(Module)` |
| `MicroTopoBrain` | class | `qwen3.py:201` | `class MicroTopoBrain(Module)` |
| `PhysioNeuron` | class | `qwen3.py:120` | `class PhysioNeuron(Module)` |
| `RegulableSymbiotic` | class | `qwen3.py:160` | `class RegulableSymbiotic(Module)` |
| `RegulableTopology` | class | `qwen3.py:179` | `class RegulableTopology` |
| `__init__` | method | `qwen3.py:61` | `def __init__(self)` |
| `__init__` | method | `qwen3.py:95` | `def __init__(self, d_in)` |
| `__init__` | method | `qwen3.py:121` | `def __init__(self, d_in, d_out, dynamic)` |
| `__init__` | method | `qwen3.py:161` | `def __init__(self, dim, atoms)` |
| `__init__` | method | `qwen3.py:180` | `def __init__(self, num_nodes)` |
| `__init__` | method | `qwen3.py:202` | `def __init__(self, config)` |
| `count_parameters` | method | `qwen3.py:221` | `def count_parameters(self)` |
| `forward` | method | `qwen3.py:105` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `qwen3.py:132` | `def forward(self, x)` |
| `forward` | method | `qwen3.py:168` | `def forward(self, x, influence)` |
| `forward` | method | `qwen3.py:224` | `def forward(self, x)` |
| `generate_ablation_matrix` | method | `qwen3.py:320` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `qwen3.py:193` | `def get_adjacency(self, plasticity)` |
| `get_batch` | method | `qwen3.py:71` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `qwen3.py:85` | `def get_full(self)` |
| `get_w2` | method | `qwen3.py:88` | `def get_w2(self)` |
| `run_ablation_study` | method | `qwen3.py:352` | `def run_ablation_study()` |
| `seed_everything` | method | `qwen3.py:50` | `def seed_everything(seed)` |
| `train_nonstationary` | method | `qwen3.py:263` | `def train_nonstationary(config)` |
| `Config` | class | `qwen4.py:36` | `class Config` |
| `DataEnvironment` | class | `qwen4.py:60` | `class DataEnvironment` |
| `HomeostaticRegulator` | class | `qwen4.py:94` | `class HomeostaticRegulator(Module)` |
| `MicroTopoBrain` | class | `qwen4.py:201` | `class MicroTopoBrain(Module)` |
| `PhysioNeuron` | class | `qwen4.py:120` | `class PhysioNeuron(Module)` |
| `RegulableSymbiotic` | class | `qwen4.py:160` | `class RegulableSymbiotic(Module)` |
| `RegulableTopology` | class | `qwen4.py:179` | `class RegulableTopology` |
| `__init__` | method | `qwen4.py:61` | `def __init__(self)` |
| `__init__` | method | `qwen4.py:95` | `def __init__(self, d_in)` |
| `__init__` | method | `qwen4.py:121` | `def __init__(self, d_in, d_out, dynamic)` |
| `__init__` | method | `qwen4.py:161` | `def __init__(self, dim, atoms)` |
| `__init__` | method | `qwen4.py:180` | `def __init__(self, num_nodes)` |
| `__init__` | method | `qwen4.py:202` | `def __init__(self, config)` |
| `count_parameters` | method | `qwen4.py:221` | `def count_parameters(self)` |
| `forward` | method | `qwen4.py:105` | `def forward(self, x, h_pre, w_norm)` |
| `forward` | method | `qwen4.py:132` | `def forward(self, x)` |
| `forward` | method | `qwen4.py:168` | `def forward(self, x, influence)` |
| `forward` | method | `qwen4.py:224` | `def forward(self, x)` |
| `generate_ablation_matrix` | method | `qwen4.py:325` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `qwen4.py:193` | `def get_adjacency(self, plasticity)` |
| `get_batch` | method | `qwen4.py:71` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `qwen4.py:85` | `def get_full(self)` |
| `get_w2` | method | `qwen4.py:88` | `def get_w2(self)` |
| `run_ablation_study` | method | `qwen4.py:357` | `def run_ablation_study()` |
| `seed_everything` | method | `qwen4.py:50` | `def seed_everything(seed)` |
| `train_nonstationary` | method | `qwen4.py:263` | `def train_nonstationary(config)` |
| `Config` | class | `qwen5.py:35` | `class Config` |
| `DataEnvironment` | class | `qwen5.py:59` | `class DataEnvironment` |
| `EpisodeMemory` | class | `qwen5.py:116` | `class EpisodeMemory` |
| `PhysioChimeraV15` | class | `qwen5.py:247` | `class PhysioChimeraV15(Module)` |
| `PredictiveHomeostat` | class | `qwen5.py:138` | `class PredictiveHomeostat(Module)` |
| `PredictivePhysioNeuron` | class | `qwen5.py:196` | `class PredictivePhysioNeuron(Module)` |
| `WorldModel` | class | `qwen5.py:94` | `class WorldModel(Module)` |
| `__init__` | method | `qwen5.py:60` | `def __init__(self)` |
| `__init__` | method | `qwen5.py:96` | `def __init__(self, hidden_dim)` |
| `__init__` | method | `qwen5.py:118` | `def __init__(self, capacity)` |
| `__init__` | method | `qwen5.py:139` | `def __init__(self, d_in)` |
| `__init__` | method | `qwen5.py:197` | `def __init__(self, d_in, d_out, dynamic)` |
| `__init__` | method | `qwen5.py:248` | `def __init__(self, config)` |
| `consolidate_svd` | method | `qwen5.py:239` | `def consolidate_svd(self, repair_strength)` |
| `count_parameters` | method | `qwen5.py:261` | `def count_parameters(self)` |
| `forward` | method | `qwen5.py:103` | `def forward(self, phase_id)` |
| `forward` | method | `qwen5.py:153` | `def forward(self, x, h_pre, w_norm, phase, reward)` |
| `forward` | method | `qwen5.py:209` | `def forward(self, x, phase, reward)` |
| `forward` | method | `qwen5.py:264` | `def forward(self, x, phase, reward)` |
| `get_batch` | method | `qwen5.py:71` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `qwen5.py:85` | `def get_full(self)` |
| `get_w2` | method | `qwen5.py:88` | `def get_w2(self)` |
| `retrieve` | method | `qwen5.py:129` | `def retrieve(self, phase, top_k)` |
| `run_experiment` | method | `qwen5.py:351` | `def run_experiment()` |
| `seed_everything` | method | `qwen5.py:49` | `def seed_everything(seed)` |
| `store` | method | `qwen5.py:122` | `def store(self, phase, metrics, state)` |
| `train_predictive` | method | `qwen5.py:285` | `def train_predictive(config)` |
| `Config` | class | `qwen6.py:31` | `class Config` |
| `ContinuumMemorySystem` | class | `qwen6.py:110` | `class ContinuumMemorySystem(Module)` |
| `DataEnvironment` | class | `qwen6.py:53` | `class DataEnvironment` |
| `NestedPhysioNeuron` | class | `qwen6.py:133` | `class NestedPhysioNeuron(Module)` |
| `PhysioChimeraNested` | class | `qwen6.py:169` | `class PhysioChimeraNested(Module)` |
| `SelfModifyingGates` | class | `qwen6.py:89` | `class SelfModifyingGates(Module)` |
| `__init__` | method | `qwen6.py:54` | `def __init__(self)` |
| `__init__` | method | `qwen6.py:90` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `qwen6.py:111` | `def __init__(self, levels, d_model, hidden_dim)` |
| `__init__` | method | `qwen6.py:134` | `def __init__(self, d_in, d_out, config)` |
| `__init__` | method | `qwen6.py:170` | `def __init__(self, config)` |
| `forward` | method | `qwen6.py:97` | `def forward(self, x)` |
| `forward` | method | `qwen6.py:123` | `def forward(self, x, global_step)` |
| `forward` | method | `qwen6.py:145` | `def forward(self, x, global_step)` |
| `forward` | method | `qwen6.py:182` | `def forward(self, x, global_step)` |
| `get_batch` | method | `qwen6.py:64` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `qwen6.py:78` | `def get_full(self)` |
| `get_w2` | method | `qwen6.py:82` | `def get_w2(self)` |
| `run_experiment` | method | `qwen6.py:265` | `def run_experiment()` |
| `seed_everything` | method | `qwen6.py:43` | `def seed_everything(seed)` |
| `train_nested` | method | `qwen6.py:202` | `def train_nested(config)` |
| `Config` | class | `qwen8.py:33` | `class Config` |
| `DataEnvironment` | class | `qwen8.py:55` | `class DataEnvironment` |
| `HomeostaticRegulator` | class | `qwen8.py:89` | `class HomeostaticRegulator(Module)` |
| `MicroTopoBrain` | class | `qwen8.py:175` | `class MicroTopoBrain(Module)` |
| `NeuralDiagnostics` | class | `qwen8.py:217` | `class NeuralDiagnostics` |
| `PhysioNeuron` | class | `qwen8.py:116` | `class PhysioNeuron(Module)` |
| `SupConHead` | class | `qwen8.py:160` | `class SupConHead(Module)` |
| `__init__` | method | `qwen8.py:56` | `def __init__(self)` |
| `__init__` | method | `qwen8.py:90` | `def __init__(self, d_in)` |
| `__init__` | method | `qwen8.py:117` | `def __init__(self, d_in, d_out, dynamic)` |
| `__init__` | method | `qwen8.py:161` | `def __init__(self, in_dim)` |
| `__init__` | method | `qwen8.py:176` | `def __init__(self, config)` |
| `__init__` | method | `qwen8.py:218` | `def __init__(self)` |
| `count_parameters` | method | `qwen8.py:189` | `def count_parameters(self)` |
| `forward` | method | `qwen8.py:100` | `def forward(self, x, h_pre, w_norm, task_loss)` |
| `forward` | method | `qwen8.py:129` | `def forward(self, x, task_loss)` |
| `forward` | method | `qwen8.py:169` | `def forward(self, x)` |
| `forward` | method | `qwen8.py:192` | `def forward(self, x, task_loss)` |
| `get_batch` | method | `qwen8.py:66` | `def get_batch(self, phase, bs)` |
| `get_full` | method | `qwen8.py:80` | `def get_full(self)` |
| `get_recent_avg` | method | `qwen8.py:236` | `def get_recent_avg(self, key, n)` |
| `get_w2` | method | `qwen8.py:83` | `def get_w2(self)` |
| `report` | method | `qwen8.py:241` | `def report(self, step, phase)` |
| `run_ablation_study` | method | `qwen8.py:332` | `def run_ablation_study()` |
| `seed_everything` | method | `qwen8.py:45` | `def seed_everything(seed)` |
| `train_nonstationary` | method | `qwen8.py:267` | `def train_nonstationary(config)` |
| `update` | method | `qwen8.py:228` | `def update(self, loss, liquid_norm, physio, prediction_error)` |
| `CorpusCallosum` | class | `qwen9.py:195` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `qwen9.py:338` | `class Flickr8kDataset(Dataset)` |
| `HomeostaticRegulator` | class | `qwen9.py:219` | `class HomeostaticRegulator(Module)` |
| `LeftHemisphere` | class | `qwen9.py:97` | `class LeftHemisphere(Module)` |
| `LiquidNeuron` | class | `qwen9.py:34` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `qwen9.py:291` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `qwen9.py:251` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `qwen9.py:78` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `qwen9.py:358` | `def __getitem__(self, idx)` |
| `__init__` | method | `qwen9.py:35` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `qwen9.py:79` | `def __init__(self, output_dim)` |
| `__init__` | method | `qwen9.py:98` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `qwen9.py:196` | `def __init__(self, dim)` |
| `__init__` | method | `qwen9.py:220` | `def __init__(self, dim)` |
| `__init__` | method | `qwen9.py:252` | `def __init__(self, vocab_size)` |
| `__init__` | method | `qwen9.py:292` | `def __init__(self)` |
| `__init__` | method | `qwen9.py:339` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `qwen9.py:355` | `def __len__(self)` |
| `_get_init_state` | method | `qwen9.py:177` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `qwen9.py:182` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `qwen9.py:371` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `forward` | method | `qwen9.py:49` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `qwen9.py:88` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `qwen9.py:119` | `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)` |
| `forward` | method | `qwen9.py:209` | `def forward(self, right_features, left_context)` |
| `forward` | method | `qwen9.py:230` | `def forward(self, right_features, epoch)` |
| `forward` | method | `qwen9.py:259` | `def forward(self, image, captions, epoch, return_diagnostics)` |
| `get_recent_avg` | method | `qwen9.py:317` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `qwen9.py:299` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `qwen9.py:306` | `def measure_vocab_diversity(self, tokens, vocab_size)` |
| `report` | method | `qwen9.py:321` | `def report(self, epoch)` |
| `setup_flickr8k` | method | `qwen9.py:386` | `def setup_flickr8k(data_dir)` |
| `train_bicameral` | method | `qwen9.py:400` | `def train_bicameral()` |
| `update` | method | `qwen9.py:312` | `def update(self)` |
| `update_flow_ema` | method | `qwen9.py:245` | `def update_flow_ema(self, flow)` |
| `AutoregulatedContinuum` | class | `qwn2.py:122` | `class AutoregulatedContinuum(Module)` |
| `AutoregulatedPlasticity` | class | `qwn2.py:97` | `class AutoregulatedPlasticity` |
| `AutoregulatedSupConHead` | class | `qwn2.py:179` | `class AutoregulatedSupConHead(Module)` |
| `AutoregulatedSymbiotic` | class | `qwn2.py:155` | `class AutoregulatedSymbiotic(Module)` |
| `HomeostaticRegulator` | class | `qwn2.py:80` | `class HomeostaticRegulator(Module)` |
| `MicroConfig` | class | `qwn2.py:34` | `class MicroConfig` |
| `MicroTopoBrain` | class | `qwn2.py:199` | `class MicroTopoBrain(Module)` |
| `__init__` | method | `qwn2.py:81` | `def __init__(self, input_dim)` |
| `__init__` | method | `qwn2.py:98` | `def __init__(self, num_nodes, grid_size)` |
| `__init__` | method | `qwn2.py:123` | `def __init__(self, dim)` |
| `__init__` | method | `qwn2.py:156` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `qwn2.py:180` | `def __init__(self, in_dim)` |
| `__init__` | method | `qwn2.py:200` | `def __init__(self, config)` |
| `_init_weights` | method | `qwn2.py:215` | `def _init_weights(self)` |
| `count_parameters` | method | `qwn2.py:220` | `def count_parameters(self)` |
| `forward` | method | `qwn2.py:91` | `def forward(self, signals)` |
| `forward` | method | `qwn2.py:133` | `def forward(self, x)` |
| `forward` | method | `qwn2.py:164` | `def forward(self, x)` |
| `forward` | method | `qwn2.py:189` | `def forward(self, x, entropy)` |
| `forward` | method | `qwn2.py:223` | `def forward(self, x)` |
| `generate_ablation_matrix` | method | `qwn2.py:341` | `def generate_ablation_matrix()` |
| `get_adjacency` | method | `qwn2.py:111` | `def get_adjacency(self, x, h_agg)` |
| `get_dataset` | method | `qwn2.py:61` | `def get_dataset(config)` |
| `micro_pgd_attack` | method | `qwn2.py:259` | `def micro_pgd_attack(model, x, y, eps, steps)` |
| `run_ablation_study` | method | `qwn2.py:366` | `def run_ablation_study()` |
| `seed_everything` | method | `qwn2.py:54` | `def seed_everything(seed)` |
| `train_with_cv` | method | `qwn2.py:279` | `def train_with_cv(config, dataset, cv_folds)` |
| `ExperimentalPredictions` | class | `resma4.10.py:545` | `class ExperimentalPredictions` |
| `GarnierTresTiempos` | class | `resma4.10.py:71` | `class GarnierTresTiempos` |
| `MyelinCavity` | class | `resma4.10.py:509` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `resma4.10.py:341` | `class NeuralNetworkRESMA` |
| `OperadorDesdoblamiento` | class | `resma4.10.py:111` | `class OperadorDesdoblamiento` |
| `QuantumLeaf` | class | `resma4.10.py:186` | `class QuantumLeaf` |
| `RESMAConstants` | class | `resma4.10.py:34` | `class RESMAConstants` |
| `RESMAUniverse` | class | `resma4.10.py:233` | `class RESMAUniverse` |
| `ResourceMonitor` | class | `resma4.10.py:592` | `class ResourceMonitor` |
| `SilencioActivoMonitor` | class | `resma4.10.py:158` | `class SilencioActivoMonitor` |
| `__init__` | method | `resma4.10.py:112` | `def __init__(self, garnier, dimension)` |
| `__init__` | method | `resma4.10.py:159` | `def __init__(self, garnier)` |
| `__init__` | method | `resma4.10.py:234` | `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)` |
| `__init__` | method | `resma4.10.py:342` | `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)` |
| `__init__` | method | `resma4.10.py:510` | `def __init__(self, axon_length, radius, n_modes)` |
| `__init__` | method | `resma4.10.py:546` | `def __init__(self, universe, network, myelin)` |
| `__post_init__` | method | `resma4.10.py:74` | `def __post_init__(self)` |
| `__post_init__` | method | `resma4.10.py:193` | `def __post_init__(self)` |
| `_aplicar_modulacion_garnier` | method | `resma4.10.py:310` | `def _aplicar_modulacion_garnier(self, measure)` |
| `_calcular_coherencia` | method | `resma4.10.py:333` | `def _calcular_coherencia(self)` |
| `_calcular_libertad` | method | `resma4.10.py:330` | `def _calcular_libertad(self)` |
| `_calcular_rho_reducida` | method | `resma4.10.py:488` | `def _calcular_rho_reducida(self)` |
| `_compute_betti_numbers` | method | `resma4.10.py:454` | `def _compute_betti_numbers(self)` |
| `_compute_scalar_mass` | method | `resma4.10.py:538` | `def _compute_scalar_mass(self)` |
| `_construct_global_state` | method | `resma4.10.py:321` | `def _construct_global_state(self)` |
| `_construir_generadores_aleatorios` | method | `resma4.10.py:121` | `def _construir_generadores_aleatorios(self)` |
| `_free_hamiltonian` | method | `resma4.10.py:527` | `def _free_hamiltonian(self)` |
| `_generate_complete_measure` | method | `resma4.10.py:281` | `def _generate_complete_measure(self)` |
| `_generate_realistic_modular_network` | method | `resma4.10.py:380` | `def _generate_realistic_modular_network(self)` |
| `_hadamard_generalizado` | method | `resma4.10.py:130` | `def _hadamard_generalizado(self)` |
| `_initialize_leaves` | method | `resma4.10.py:270` | `def _initialize_leaves(self)` |
| `_loss_potential` | method | `resma4.10.py:532` | `def _loss_potential(self)` |
| `_make_serializable` | method | `resma4.10.py:690` | `def _make_serializable(obj, depth, max_depth, _visited)` |
| `_spectral_dimension` | method | `resma4.10.py:462` | `def _spectral_dimension(self)` |
| `_topological_ramsey` | method | `resma4.10.py:484` | `def _topological_ramsey(self)` |
| `_validar_axioma_6` | method | `resma4.10.py:496` | `def _validar_axioma_6(self)` |
| `bures_distance` | method | `resma4.10.py:203` | `def bures_distance(self, other)` |
| `calcular_alpha_modificado` | method | `resma4.10.py:150` | `def calcular_alpha_modificado(self, alpha_base)` |
| `calcular_delta_s_loop` | method | `resma4.10.py:163` | `def calcular_delta_s_loop(self, rho_red, b1)` |
| `cargar_checkpoint` | method | `resma4.10.py:662` | `def cargar_checkpoint(filename)` |
| `compute_log_bayes_factor` | method | `resma4.10.py:551` | `def compute_log_bayes_factor(self)` |
| `epsilon_critico` | method | `resma4.10.py:86` | `def epsilon_critico(self)` |
| `es_silencio_activo` | method | `resma4.10.py:170` | `def es_silencio_activo(self, rho_red, b1)` |
| `from_dict` | method | `resma4.10.py:102` | `def from_dict(cls, data)` |
| `get_memory_gb` | method | `resma4.10.py:594` | `def get_memory_gb()` |
| `guardar_checkpoint` | method | `resma4.10.py:604` | `def guardar_checkpoint(data, filename)` |
| `log_resources` | method | `resma4.10.py:599` | `def log_resources()` |
| `modulation_factor` | method | `resma4.10.py:89` | `def modulation_factor(self)` |
| `operator` | method | `resma4.10.py:135` | `def operator(self)` |
| `simulate_resma_garnier` | method | `resma4.10.py:794` | `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)` |
| `spectral_density` | method | `resma4.10.py:197` | `def spectral_density(self, omega)` |
| `to_dict` | method | `resma4.10.py:92` | `def to_dict(self)` |
| `verify_pt_condition` | method | `resma4.10.py:50` | `def verify_pt_condition(cls)` |
| `EmunaOperator` | class | `resma4.2.py:224` | `class EmunaOperator` |
| `ExperimentalPredictions` | class | `resma4.2.py:474` | `class ExperimentalPredictions` |
| `MyelinCavity` | class | `resma4.2.py:291` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `resma4.2.py:351` | `class NeuralNetworkRESMA` |
| `PhysicalValidator` | class | `resma4.2.py:78` | `class PhysicalValidator` |
| `QuantumLeaf` | class | `resma4.2.py:109` | `class QuantumLeaf` |
| `RESMAConstants` | class | `resma4.2.py:38` | `class RESMAConstants` |
| `RESMAUniverse` | class | `resma4.2.py:162` | `class RESMAUniverse` |
| `__init__` | method | `resma4.2.py:165` | `def __init__(self, n_leaves, seed)` |
| `__init__` | method | `resma4.2.py:227` | `def __init__(self, universe, n_samples)` |
| `__init__` | method | `resma4.2.py:354` | `def __init__(self, n_nodes, seed)` |
| `__init__` | method | `resma4.2.py:477` | `def __init__(self, universe, myelin, network)` |
| `__post_init__` | method | `resma4.2.py:117` | `def __post_init__(self)` |
| `__post_init__` | method | `resma4.2.py:297` | `def __post_init__(self)` |
| `_compute_betti_numbers` | method | `resma4.2.py:429` | `def _compute_betti_numbers(self)` |
| `_compute_scalar_mass` | method | `resma4.2.py:317` | `def _compute_scalar_mass(self)` |
| `_construct_global_state` | method | `resma4.2.py:205` | `def _construct_global_state(self)` |
| `_construct_hardy_state` | method | `resma4.2.py:234` | `def _construct_hardy_state(self)` |
| `_evaluation_functional` | method | `resma4.2.py:246` | `def _evaluation_functional(self, state_weights)` |
| `_free_hamiltonian` | method | `resma4.2.py:304` | `def _free_hamiltonian(self)` |
| `_generate_fractal_graph` | method | `resma4.2.py:366` | `def _generate_fractal_graph(self)` |
| `_generate_gibbs_measure` | method | `resma4.2.py:189` | `def _generate_gibbs_measure(self)` |
| `_graph_to_distance_matrix` | method | `resma4.2.py:445` | `def _graph_to_distance_matrix(self)` |
| `_initialize_leaves` | method | `resma4.2.py:176` | `def _initialize_leaves(self)` |
| `_loss_potential` | method | `resma4.2.py:310` | `def _loss_potential(self)` |
| `_pt_symmetry_condition` | method | `resma4.2.py:321` | `def _pt_symmetry_condition(self)` |
| `_spectral_dimension` | method | `resma4.2.py:381` | `def _spectral_dimension(self)` |
| `_szego_projector` | method | `resma4.2.py:238` | `def _szego_projector(self)` |
| `_topological_ramsey` | method | `resma4.2.py:410` | `def _topological_ramsey(self)` |
| `bures_distance` | method | `resma4.2.py:136` | `def bures_distance(self, other)` |
| `coherence_quantum` | method | `resma4.2.py:327` | `def coherence_quantum(self)` |
| `compute_gibbs_free_energy` | method | `resma4.2.py:216` | `def compute_gibbs_free_energy(self)` |
| `compute_log_bayes_factor` | method | `resma4.2.py:496` | `def compute_log_bayes_factor(self)` |
| `critical_percolation_time` | method | `resma4.2.py:461` | `def critical_percolation_time(self)` |
| `haagerup_weight` | method | `resma4.2.py:154` | `def haagerup_weight(self)` |
| `modular_entropy` | method | `resma4.2.py:126` | `def modular_entropy(self)` |
| `predict_all` | method | `resma4.2.py:483` | `def predict_all(self)` |
| `project` | method | `resma4.2.py:258` | `def project(self, state_vector)` |
| `simulate_resma_complete` | method | `resma4.2.py:556` | `def simulate_resma_complete(n_leaves, n_nodes, seed)` |
| `spectral_density` | method | `resma4.2.py:121` | `def spectral_density(self, omega)` |
| `validate_connectome_size` | method | `resma4.2.py:96` | `def validate_connectome_size(n_nodes)` |
| `validate_dimension` | method | `resma4.2.py:80` | `def validate_dimension(alpha, tolerance)` |
| `validate_pt_symmetry` | method | `resma4.2.py:88` | `def validate_pt_symmetry(kappa, Omega, chi)` |
| `validate_spectral_dimension` | method | `resma4.2.py:101` | `def validate_spectral_dimension(dim)` |
| `verify_pt_condition` | method | `resma4.2.py:67` | `def verify_pt_condition(cls)` |
| `ExperimentalPredictions` | class | `resma4.3.py:485` | `class ExperimentalPredictions` |
| `MyelinCavity` | class | `resma4.3.py:297` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `resma4.3.py:352` | `class NeuralNetworkRESMA` |
| `PhysicalValidator` | class | `resma4.3.py:269` | `class PhysicalValidator` |
| `QuantumLeaf` | class | `resma4.3.py:142` | `class QuantumLeaf` |
| `RESMAConstants` | class | `resma4.3.py:110` | `class RESMAConstants` |
| `RESMAUniverse` | class | `resma4.3.py:190` | `class RESMAUniverse` |
| `ResourceMonitor` | class | `resma4.3.py:33` | `class ResourceMonitor` |
| `__init__` | method | `resma4.3.py:193` | `def __init__(self, n_leaves, seed)` |
| `__init__` | method | `resma4.3.py:353` | `def __init__(self, n_nodes, seed)` |
| `__init__` | method | `resma4.3.py:486` | `def __init__(self, universe, myelin, network)` |
| `__post_init__` | method | `resma4.3.py:150` | `def __post_init__(self)` |
| `__post_init__` | method | `resma4.3.py:302` | `def __post_init__(self)` |
| `_compute_betti_numbers` | method | `resma4.3.py:444` | `def _compute_betti_numbers(self)` |
| `_compute_scalar_mass` | method | `resma4.3.py:323` | `def _compute_scalar_mass(self)` |
| `_construct_global_state` | method | `resma4.3.py:254` | `def _construct_global_state(self)` |
| `_free_hamiltonian` | method | `resma4.3.py:312` | `def _free_hamiltonian(self)` |
| `_generate_fractal_graph` | method | `resma4.3.py:372` | `def _generate_fractal_graph(self)` |
| `_generate_gibbs_measure` | method | `resma4.3.py:227` | `def _generate_gibbs_measure(self)` |
| `_graph_to_distance_matrix` | method | `resma4.3.py:460` | `def _graph_to_distance_matrix(self)` |
| `_initialize_leaves` | method | `resma4.3.py:215` | `def _initialize_leaves(self)` |
| `_loss_potential` | method | `resma4.3.py:317` | `def _loss_potential(self)` |
| `_pt_symmetry_condition` | method | `resma4.3.py:326` | `def _pt_symmetry_condition(self)` |
| `_spectral_dimension` | method | `resma4.3.py:401` | `def _spectral_dimension(self)` |
| `_topological_ramsey` | method | `resma4.3.py:425` | `def _topological_ramsey(self)` |
| `bures_distance` | method | `resma4.3.py:158` | `def bures_distance(self, other)` |
| `cargar_checkpoint` | method | `resma4.3.py:84` | `def cargar_checkpoint(filename)` |
| `check_memory_limit` | method | `resma4.3.py:40` | `def check_memory_limit()` |
| `coherence_quantum` | method | `resma4.3.py:331` | `def coherence_quantum(self)` |
| `compute_log_bayes_factor` | method | `resma4.3.py:492` | `def compute_log_bayes_factor(self)` |
| `critical_percolation_time` | method | `resma4.3.py:476` | `def critical_percolation_time(self)` |
| `get_memory_gb` | method | `resma4.3.py:35` | `def get_memory_gb()` |
| `guardar_checkpoint` | method | `resma4.3.py:54` | `def guardar_checkpoint(data, filename)` |
| `log_resources` | method | `resma4.3.py:49` | `def log_resources()` |
| `simulate_resma_with_checkpointing` | method | `resma4.3.py:545` | `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)` |
| `spectral_density` | method | `resma4.3.py:154` | `def spectral_density(self, omega)` |
| `validate_connectome_size` | method | `resma4.3.py:287` | `def validate_connectome_size(n_nodes)` |
| `validate_dimension` | method | `resma4.3.py:271` | `def validate_dimension(alpha, tolerance)` |
| `validate_pt_symmetry` | method | `resma4.3.py:279` | `def validate_pt_symmetry(kappa, Omega, chi)` |
| `validate_spectral_dimension` | method | `resma4.3.py:292` | `def validate_spectral_dimension(dim)` |
| `verify_pt_condition` | method | `resma4.3.py:127` | `def verify_pt_condition(cls)` |
| `MyelinCavity` | class | `resma4.4.py:343` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `resma4.4.py:377` | `class NeuralNetworkRESMA` |
| `PhysicalValidator` | class | `resma4.4.py:316` | `class PhysicalValidator` |
| `QuantumLeaf` | class | `resma4.4.py:160` | `class QuantumLeaf` |
| `RESMAConstants` | class | `resma4.4.py:129` | `class RESMAConstants` |
| `RESMAUniverse` | class | `resma4.4.py:203` | `class RESMAUniverse` |
| `ResourceMonitor` | class | `resma4.4.py:34` | `class ResourceMonitor` |
| `__init__` | method | `resma4.4.py:206` | `def __init__(self, n_leaves, seed, leaves, measure, global_state)` |
| `__init__` | method | `resma4.4.py:378` | `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti)` |
| `__post_init__` | method | `resma4.4.py:168` | `def __post_init__(self)` |
| `__post_init__` | method | `resma4.4.py:348` | `def __post_init__(self)` |
| `_compute_betti_numbers` | method | `resma4.4.py:518` | `def _compute_betti_numbers(self)` |
| `_compute_scalar_mass` | method | `resma4.4.py:369` | `def _compute_scalar_mass(self)` |
| `_construct_global_state` | method | `resma4.4.py:301` | `def _construct_global_state(self)` |
| `_free_hamiltonian` | method | `resma4.4.py:358` | `def _free_hamiltonian(self)` |
| `_generate_fractal_graph` | method | `resma4.4.py:446` | `def _generate_fractal_graph(self)` |
| `_generate_gibbs_measure` | method | `resma4.4.py:269` | `def _generate_gibbs_measure(self)` |
| `_graph_to_distance_matrix` | method | `resma4.4.py:534` | `def _graph_to_distance_matrix(self)` |
| `_initialize_leaves` | method | `resma4.4.py:258` | `def _initialize_leaves(self)` |
| `_loss_potential` | method | `resma4.4.py:363` | `def _loss_potential(self)` |
| `_pt_symmetry_condition` | method | `resma4.4.py:372` | `def _pt_symmetry_condition(self)` |
| `_spectral_dimension` | method | `resma4.4.py:475` | `def _spectral_dimension(self)` |
| `_topological_ramsey` | method | `resma4.4.py:499` | `def _topological_ramsey(self)` |
| `bures_distance` | method | `resma4.4.py:176` | `def bures_distance(self, other)` |
| `cargar_checkpoint` | method | `resma4.4.py:93` | `def cargar_checkpoint(filename)` |
| `check_memory_limit` | method | `resma4.4.py:41` | `def check_memory_limit()` |
| `get_memory_gb` | method | `resma4.4.py:36` | `def get_memory_gb()` |
| `guardar_checkpoint` | method | `resma4.4.py:59` | `def guardar_checkpoint(data, filename)` |
| `log_resources` | method | `resma4.4.py:50` | `def log_resources()` |
| `simulate_resma_with_checkpointing` | method | `resma4.4.py:554` | `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)` |
| `spectral_density` | method | `resma4.4.py:172` | `def spectral_density(self, omega)` |
| `validate_connectome_size` | method | `resma4.4.py:334` | `def validate_connectome_size(n_nodes)` |
| `validate_dimension` | method | `resma4.4.py:318` | `def validate_dimension(alpha, tolerance)` |
| `validate_pt_symmetry` | method | `resma4.4.py:326` | `def validate_pt_symmetry(kappa, Omega, chi)` |
| `validate_spectral_dimension` | method | `resma4.4.py:339` | `def validate_spectral_dimension(dim)` |
| `verify_pt_condition` | method | `resma4.4.py:146` | `def verify_pt_condition(cls)` |
| `ExperimentalPredictions` | class | `resma4.5.py:658` | `class ExperimentalPredictions` |
| `GarnierTresTiempos` | class | `resma4.5.py:163` | `class GarnierTresTiempos` |
| `MyelinCavity` | class | `resma4.5.py:486` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `resma4.5.py:517` | `class NeuralNetworkRESMA` |
| `OperadorDesdoblamiento` | class | `resma4.5.py:201` | `class OperadorDesdoblamiento` |
| `QuantumLeaf` | class | `resma4.5.py:337` | `class QuantumLeaf` |
| `RESMAConstants` | class | `resma4.5.py:135` | `class RESMAConstants` |
| `RESMAUniverse` | class | `resma4.5.py:380` | `class RESMAUniverse` |
| `ResourceMonitor` | class | `resma4.5.py:34` | `class ResourceMonitor` |
| `SilencioActivoMonitor` | class | `resma4.5.py:269` | `class SilencioActivoMonitor` |
| `__init__` | method | `resma4.5.py:206` | `def __init__(self, garnier, dimension)` |
| `__init__` | method | `resma4.5.py:273` | `def __init__(self, garnier, network)` |
| `__init__` | method | `resma4.5.py:383` | `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)` |
| `__init__` | method | `resma4.5.py:488` | `def __init__(self, axon_length, radius, n_modes)` |
| `__init__` | method | `resma4.5.py:520` | `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)` |
| `__init__` | method | `resma4.5.py:661` | `def __init__(self, universe, myelin, network)` |
| `__post_init__` | method | `resma4.5.py:170` | `def __post_init__(self)` |
| `__post_init__` | method | `resma4.5.py:345` | `def __post_init__(self)` |
| `_aplicar_desdoblamiento_a_medida` | method | `resma4.5.py:451` | `def _aplicar_desdoblamiento_a_medida(self, measure)` |
| `_calcular_libertad_universo` | method | `resma4.5.py:481` | `def _calcular_libertad_universo(self)` |
| `_calcular_rho_reducida` | method | `resma4.5.py:636` | `def _calcular_rho_reducida(self)` |
| `_calcular_rho_reducida_aproximada` | method | `resma4.5.py:299` | `def _calcular_rho_reducida_aproximada(self)` |
| `_compute_betti_numbers` | method | `resma4.5.py:627` | `def _compute_betti_numbers(self)` |
| `_compute_scalar_mass` | method | `resma4.5.py:510` | `def _compute_scalar_mass(self)` |
| `_construct_global_state` | method | `resma4.5.py:471` | `def _construct_global_state(self)` |
| `_construir_generadores_E8` | method | `resma4.5.py:214` | `def _construir_generadores_E8(self)` |
| `_free_hamiltonian` | method | `resma4.5.py:499` | `def _free_hamiltonian(self)` |
| `_generate_fractal_graph` | method | `resma4.5.py:559` | `def _generate_fractal_graph(self)` |
| `_generate_gibbs_measure` | method | `resma4.5.py:431` | `def _generate_gibbs_measure(self)` |
| `_hadamard_generalizado` | method | `resma4.5.py:226` | `def _hadamard_generalizado(self)` |
| `_initialize_leaves` | method | `resma4.5.py:420` | `def _initialize_leaves(self)` |
| `_loss_potential` | method | `resma4.5.py:504` | `def _loss_potential(self)` |
| `_make_serializable` | method | `resma4.5.py:113` | `def _make_serializable(obj)` |
| `_pt_symmetry_condition` | method | `resma4.5.py:513` | `def _pt_symmetry_condition(self)` |
| `_spectral_dimension` | method | `resma4.5.py:591` | `def _spectral_dimension(self)` |
| `_topological_ramsey` | method | `resma4.5.py:615` | `def _topological_ramsey(self)` |
| `aplicar_a_estado` | method | `resma4.5.py:254` | `def aplicar_a_estado(self, estado)` |
| `bures_distance` | method | `resma4.5.py:353` | `def bures_distance(self, other)` |
| `calcular_alpha_modificado` | method | `resma4.5.py:260` | `def calcular_alpha_modificado(self, alpha_base)` |
| `calcular_delta_s_loop` | method | `resma4.5.py:278` | `def calcular_delta_s_loop(self, rho_red)` |
| `cargar_checkpoint` | method | `resma4.5.py:88` | `def cargar_checkpoint(filename)` |
| `check_memory_limit` | method | `resma4.5.py:41` | `def check_memory_limit()` |
| `compute_log_bayes_factor` | method | `resma4.5.py:666` | `def compute_log_bayes_factor(self)` |
| `epsilon_critico` | method | `resma4.5.py:185` | `def epsilon_critico(self)` |
| `es_silencio_activo` | method | `resma4.5.py:307` | `def es_silencio_activo(self, rho_red)` |
| `factor_escala` | method | `resma4.5.py:181` | `def factor_escala(self, tiempo_idx)` |
| `from_dict` | method | `resma4.5.py:197` | `def from_dict(cls, data)` |
| `get_memory_gb` | method | `resma4.5.py:36` | `def get_memory_gb()` |
| `guardar_checkpoint` | method | `resma4.5.py:59` | `def guardar_checkpoint(data, filename)` |
| `log_resources` | method | `resma4.5.py:50` | `def log_resources()` |
| `operator` | method | `resma4.5.py:235` | `def operator(self)` |
| `simulate_resma_garnier` | method | `resma4.5.py:695` | `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)` |
| `spectral_density` | method | `resma4.5.py:349` | `def spectral_density(self, omega)` |
| `to_dict` | method | `resma4.5.py:192` | `def to_dict(self)` |
| `umbral_percolacion` | method | `resma4.5.py:324` | `def umbral_percolacion(self)` |
| `validar_axioma_6` | method | `resma4.5.py:644` | `def validar_axioma_6(self)` |
| `verify_pt_condition` | method | `resma4.5.py:152` | `def verify_pt_condition(cls)` |
| `ConectomaCuantico` | class | `resma4.6.py:659` | `class ConectomaCuantico` |
| `ExperimentalPredictions` | class | `resma4.6.py:898` | `class ExperimentalPredictions` |
| `GarnierTresTiempos` | class | `resma4.6.py:169` | `class GarnierTresTiempos` |
| `MyelinCavity` | class | `resma4.6.py:608` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `resma4.6.py:800` | `class NeuralNetworkRESMA` |
| `OperadorDesdoblamiento` | class | `resma4.6.py:233` | `class OperadorDesdoblamiento` |
| `QuantumLeaf` | class | `resma4.6.py:441` | `class QuantumLeaf` |
| `RESMAConstants` | class | `resma4.6.py:139` | `class RESMAConstants` |
| `RESMAUniverse` | class | `resma4.6.py:484` | `class RESMAUniverse` |
| `ResourceMonitor` | class | `resma4.6.py:31` | `class ResourceMonitor` |
| `SilencioActivoMonitor` | class | `resma4.6.py:312` | `class SilencioActivoMonitor` |
| `__init__` | method | `resma4.6.py:238` | `def __init__(self, garnier, dimension)` |
| `__init__` | method | `resma4.6.py:318` | `def __init__(self, garnier, network)` |
| `__init__` | method | `resma4.6.py:487` | `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)` |
| `__init__` | method | `resma4.6.py:611` | `def __init__(self, axon_length, radius, n_modes)` |
| `__init__` | method | `resma4.6.py:667` | `def __init__(self, n_nodes, seed, garnier)` |
| `__init__` | method | `resma4.6.py:803` | `def __init__(self, n_nodes, seed, conectoma_quantum, garnier)` |
| `__init__` | method | `resma4.6.py:901` | `def __init__(self, universe, myelin, network)` |
| `__post_init__` | method | `resma4.6.py:181` | `def __post_init__(self)` |
| `__post_init__` | method | `resma4.6.py:449` | `def __post_init__(self)` |
| `_aplicar_desdoblamiento_a_medida` | method | `resma4.6.py:557` | `def _aplicar_desdoblamiento_a_medida(self, measure)` |
| `_calcular_b1_cuantico` | method | `resma4.6.py:784` | `def _calcular_b1_cuantico(self)` |
| `_calcular_conectividad_cuantica` | method | `resma4.6.py:708` | `def _calcular_conectividad_cuantica(self)` |
| `_calcular_libertad_universo` | method | `resma4.6.py:603` | `def _calcular_libertad_universo(self)` |
| `_calcular_rho_reducida` | method | `resma4.6.py:842` | `def _calcular_rho_reducida(self)` |
| `_calcular_rho_reducida_aproximada` | method | `resma4.6.py:367` | `def _calcular_rho_reducida_aproximada(self)` |
| `_calcular_zpe` | method | `resma4.6.py:643` | `def _calcular_zpe(self)` |
| `_compute_scalar_mass` | method | `resma4.6.py:640` | `def _compute_scalar_mass(self)` |
| `_construct_global_state` | method | `resma4.6.py:584` | `def _construct_global_state(self)` |
| `_construir_generadores_E8_ZPE` | method | `resma4.6.py:246` | `def _construir_generadores_E8_ZPE(self)` |
| `_free_hamiltonian` | method | `resma4.6.py:624` | `def _free_hamiltonian(self)` |
| `_generate_gibbs_measure` | method | `resma4.6.py:535` | `def _generate_gibbs_measure(self)` |
| `_hadamard_generalizado_ZPE` | method | `resma4.6.py:266` | `def _hadamard_generalizado_ZPE(self)` |
| `_inicializar_amplitudes` | method | `resma4.6.py:686` | `def _inicializar_amplitudes(self)` |
| `_initialize_leaves` | method | `resma4.6.py:521` | `def _initialize_leaves(self)` |
| `_loss_potential` | method | `resma4.6.py:633` | `def _loss_potential(self)` |
| `_make_serializable` | method | `resma4.6.py:112` | `def _make_serializable(obj)` |
| `_pt_symmetry_condition` | method | `resma4.6.py:655` | `def _pt_symmetry_condition(self)` |
| `_recalcular_betti_clasicos` | method | `resma4.6.py:742` | `def _recalcular_betti_clasicos(self)` |
| `alpha_modificado` | method | `resma4.6.py:304` | `def alpha_modificado(self, alpha_base)` |
| `bures_distance` | method | `resma4.6.py:457` | `def bures_distance(self, other)` |
| `calcular_delta_s_loop` | method | `resma4.6.py:327` | `def calcular_delta_s_loop(self, rho_red)` |
| `cargar_checkpoint` | method | `resma4.6.py:86` | `def cargar_checkpoint(filename)` |
| `check_memory_limit` | method | `resma4.6.py:38` | `def check_memory_limit(threshold)` |
| `colapsar_a_clasico` | method | `resma4.6.py:718` | `def colapsar_a_clasico(self, threshold)` |
| `compute_log_bayes_factor` | method | `resma4.6.py:906` | `def compute_log_bayes_factor(self)` |
| `epsilon_critico` | method | `resma4.6.py:200` | `def epsilon_critico(self)` |
| `es_silencio_activo` | method | `resma4.6.py:383` | `def es_silencio_activo(self, rho_red)` |
| `factor_escala` | method | `resma4.6.py:194` | `def factor_escala(self, tiempo_idx)` |
| `from_dict` | method | `resma4.6.py:220` | `def from_dict(cls, data)` |
| `get_memory_gb` | method | `resma4.6.py:33` | `def get_memory_gb()` |
| `guardar_checkpoint` | method | `resma4.6.py:56` | `def guardar_checkpoint(data, filename)` |
| `log_resources` | method | `resma4.6.py:47` | `def log_resources()` |
| `medir_delta_s_loop` | method | `resma4.6.py:756` | `def medir_delta_s_loop(self)` |
| `modo_goldstone` | method | `resma4.6.py:415` | `def modo_goldstone(self)` |
| `obtener_metricas_cuanticas` | method | `resma4.6.py:874` | `def obtener_metricas_cuanticas(self)` |
| `operator` | method | `resma4.6.py:284` | `def operator(self)` |
| `simulate_resma_garnier` | method | `resma4.6.py:951` | `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart, target_connectivity)` |
| `spectral_density` | method | `resma4.6.py:453` | `def spectral_density(self, omega)` |
| `to_dict` | method | `resma4.6.py:208` | `def to_dict(self)` |
| `umbral_percolacion` | method | `resma4.6.py:411` | `def umbral_percolacion(self)` |
| `validar_axioma_6_cuantico` | method | `resma4.6.py:858` | `def validar_axioma_6_cuantico(self)` |
| `verify_pt_condition` | method | `resma4.6.py:158` | `def verify_pt_condition(cls)` |
| `ExperimentalPredictions` | class | `resma4.7.py:484` | `class ExperimentalPredictions` |
| `GarnierTresTiempos` | class | `resma4.7.py:46` | `class GarnierTresTiempos` |
| `NeuralNetworkRESMA` | class | `resma4.7.py:336` | `class NeuralNetworkRESMA` |
| `OperadorDesdoblamiento` | class | `resma4.7.py:94` | `class OperadorDesdoblamiento` |
| `QuantumLeaf` | class | `resma4.7.py:190` | `class QuantumLeaf` |
| `RESMAConstants` | class | `resma4.7.py:23` | `class RESMAConstants` |
| `RESMAUniverse` | class | `resma4.7.py:223` | `class RESMAUniverse` |
| `SilencioActivoMonitor` | class | `resma4.7.py:139` | `class SilencioActivoMonitor` |
| `__init__` | method | `resma4.7.py:98` | `def __init__(self, garnier, dimension)` |
| `__init__` | method | `resma4.7.py:143` | `def __init__(self, garnier)` |
| `__init__` | method | `resma4.7.py:226` | `def __init__(self, n_leaves, seed, garnier)` |
| `__init__` | method | `resma4.7.py:341` | `def __init__(self, n_nodes, seed, garnier)` |
| `__init__` | method | `resma4.7.py:487` | `def __init__(self, universe, network)` |
| `__post_init__` | method | `resma4.7.py:53` | `def __post_init__(self)` |
| `_calcular_coherencia` | method | `resma4.7.py:326` | `def _calcular_coherencia(self)` |
| `_calcular_libertad` | method | `resma4.7.py:322` | `def _calcular_libertad(self)` |
| `_calcular_rho_reducida` | method | `resma4.7.py:460` | `def _calcular_rho_reducida(self)` |
| `_compute_betti_numbers` | method | `resma4.7.py:424` | `def _compute_betti_numbers(self)` |
| `_compute_coupling` | method | `resma4.7.py:67` | `def _compute_coupling(self)` |
| `_construct_global_state` | method | `resma4.7.py:312` | `def _construct_global_state(self)` |
| `_construir_generadores` | method | `resma4.7.py:103` | `def _construir_generadores(self)` |
| `_generate_modulated_measure` | method | `resma4.7.py:265` | `def _generate_modulated_measure(self)` |
| `_generate_realistic_network` | method | `resma4.7.py:379` | `def _generate_realistic_network(self)` |
| `_initialize_leaves` | method | `resma4.7.py:252` | `def _initialize_leaves(self)` |
| `_spectral_dimension` | method | `resma4.7.py:433` | `def _spectral_dimension(self)` |
| `_topological_ramsey` | method | `resma4.7.py:455` | `def _topological_ramsey(self)` |
| `_validar_axioma_6` | method | `resma4.7.py:470` | `def _validar_axioma_6(self)` |
| `aplicar_modulacion` | method | `resma4.7.py:120` | `def aplicar_modulacion(self, state_vector)` |
| `bures_distance` | method | `resma4.7.py:203` | `def bures_distance(self, other)` |
| `calcular_alpha_modificado` | method | `resma4.7.py:126` | `def calcular_alpha_modificado(self, alpha_base)` |
| `calcular_delta_s_loop` | method | `resma4.7.py:147` | `def calcular_delta_s_loop(self, rho_red, b1)` |
| `compute_log_bayes_factor` | method | `resma4.7.py:491` | `def compute_log_bayes_factor(self)` |
| `epsilon_critico` | method | `resma4.7.py:77` | `def epsilon_critico(self)` |
| `es_silencio_activo` | method | `resma4.7.py:166` | `def es_silencio_activo(self, rho_red, b1)` |
| `factor_escala` | method | `resma4.7.py:72` | `def factor_escala(self, tiempo_idx)` |
| `modulation_factor` | method | `resma4.7.py:85` | `def modulation_factor(self)` |
| `operator` | method | `resma4.7.py:115` | `def operator(self)` |
| `simulate_resma_garnier` | method | `resma4.7.py:537` | `def simulate_resma_garnier(n_leaves, n_nodes, seed)` |
| `spectral_density` | method | `resma4.7.py:197` | `def spectral_density(self, omega)` |
| `ExperimentalPredictions` | class | `resma4.8.py:620` | `class ExperimentalPredictions` |
| `GarnierTresTiempos` | class | `resma4.8.py:158` | `class GarnierTresTiempos` |
| `MyelinCavity` | class | `resma4.8.py:584` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `resma4.8.py:433` | `class NeuralNetworkRESMA` |
| `OperadorDesdoblamiento` | class | `resma4.8.py:209` | `class OperadorDesdoblamiento` |
| `QuantumLeaf` | class | `resma4.8.py:284` | `class QuantumLeaf` |
| `RESMAConstants` | class | `resma4.8.py:37` | `class RESMAConstants` |
| `RESMAUniverse` | class | `resma4.8.py:331` | `class RESMAUniverse` |
| `ResourceMonitor` | class | `resma4.8.py:70` | `class ResourceMonitor` |
| `SilencioActivoMonitor` | class | `resma4.8.py:256` | `class SilencioActivoMonitor` |
| `__init__` | method | `resma4.8.py:210` | `def __init__(self, garnier, dimension)` |
| `__init__` | method | `resma4.8.py:257` | `def __init__(self, garnier)` |
| `__init__` | method | `resma4.8.py:332` | `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)` |
| `__init__` | method | `resma4.8.py:434` | `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)` |
| `__init__` | method | `resma4.8.py:585` | `def __init__(self, axon_length, radius, n_modes)` |
| `__init__` | method | `resma4.8.py:621` | `def __init__(self, universe, network, myelin)` |
| `__post_init__` | method | `resma4.8.py:161` | `def __post_init__(self)` |
| `__post_init__` | method | `resma4.8.py:291` | `def __post_init__(self)` |
| `_aplicar_modulacion_garnier` | method | `resma4.8.py:402` | `def _aplicar_modulacion_garnier(self, measure)` |
| `_calcular_coherencia` | method | `resma4.8.py:425` | `def _calcular_coherencia(self)` |
| `_calcular_libertad` | method | `resma4.8.py:422` | `def _calcular_libertad(self)` |
| `_calcular_rho_reducida` | method | `resma4.8.py:563` | `def _calcular_rho_reducida(self)` |
| `_compute_betti_numbers` | method | `resma4.8.py:529` | `def _compute_betti_numbers(self)` |
| `_compute_coupling` | method | `resma4.8.py:179` | `def _compute_coupling(self)` |
| `_compute_scalar_mass` | method | `resma4.8.py:613` | `def _compute_scalar_mass(self)` |
| `_construct_global_state` | method | `resma4.8.py:413` | `def _construct_global_state(self)` |
| `_construir_generadores_aleatorios` | method | `resma4.8.py:219` | `def _construir_generadores_aleatorios(self)` |
| `_free_hamiltonian` | method | `resma4.8.py:602` | `def _free_hamiltonian(self)` |
| `_generate_complete_measure` | method | `resma4.8.py:373` | `def _generate_complete_measure(self)` |
| `_generate_realistic_modular_network` | method | `resma4.8.py:472` | `def _generate_realistic_modular_network(self)` |
| `_hadamard_generalizado` | method | `resma4.8.py:228` | `def _hadamard_generalizado(self)` |
| `_initialize_leaves` | method | `resma4.8.py:362` | `def _initialize_leaves(self)` |
| `_loss_potential` | method | `resma4.8.py:607` | `def _loss_potential(self)` |
| `_make_serializable` | method | `resma4.8.py:141` | `def _make_serializable(obj)` |
| `_spectral_dimension` | method | `resma4.8.py:537` | `def _spectral_dimension(self)` |
| `_topological_ramsey` | method | `resma4.8.py:559` | `def _topological_ramsey(self)` |
| `_validar_axioma_6` | method | `resma4.8.py:571` | `def _validar_axioma_6(self)` |
| `bures_distance` | method | `resma4.8.py:301` | `def bures_distance(self, other)` |
| `calcular_alpha_modificado` | method | `resma4.8.py:248` | `def calcular_alpha_modificado(self, alpha_base)` |
| `calcular_delta_s_loop` | method | `resma4.8.py:261` | `def calcular_delta_s_loop(self, rho_red, b1)` |
| `cargar_checkpoint` | method | `resma4.8.py:117` | `def cargar_checkpoint(filename)` |
| `check_memory_limit` | method | `resma4.8.py:77` | `def check_memory_limit()` |
| `compute_log_bayes_factor` | method | `resma4.8.py:626` | `def compute_log_bayes_factor(self)` |
| `epsilon_critico` | method | `resma4.8.py:182` | `def epsilon_critico(self)` |
| `es_silencio_activo` | method | `resma4.8.py:268` | `def es_silencio_activo(self, rho_red, b1)` |
| `from_dict` | method | `resma4.8.py:199` | `def from_dict(cls, data)` |
| `get_memory_gb` | method | `resma4.8.py:72` | `def get_memory_gb()` |
| `guardar_checkpoint` | method | `resma4.8.py:91` | `def guardar_checkpoint(data, filename)` |
| `log_resources` | method | `resma4.8.py:86` | `def log_resources()` |
| `modulation_factor` | method | `resma4.8.py:186` | `def modulation_factor(self)` |
| `operator` | method | `resma4.8.py:233` | `def operator(self)` |
| `simulate_resma_garnier` | method | `resma4.8.py:667` | `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)` |
| `spectral_density` | method | `resma4.8.py:295` | `def spectral_density(self, omega)` |
| `to_dict` | method | `resma4.8.py:189` | `def to_dict(self)` |
| `verify_pt_condition` | method | `resma4.8.py:54` | `def verify_pt_condition(cls)` |
| `ExperimentalPredictions` | class | `resma4.9.py:528` | `class ExperimentalPredictions` |
| `GarnierTresTiempos` | class | `resma4.9.py:71` | `class GarnierTresTiempos` |
| `MyelinCavity` | class | `resma4.9.py:492` | `class MyelinCavity` |
| `NeuralNetworkRESMA` | class | `resma4.9.py:341` | `class NeuralNetworkRESMA` |
| `OperadorDesdoblamiento` | class | `resma4.9.py:111` | `class OperadorDesdoblamiento` |
| `QuantumLeaf` | class | `resma4.9.py:186` | `class QuantumLeaf` |
| `RESMAConstants` | class | `resma4.9.py:34` | `class RESMAConstants` |
| `RESMAUniverse` | class | `resma4.9.py:233` | `class RESMAUniverse` |
| `ResourceMonitor` | class | `resma4.9.py:575` | `class ResourceMonitor` |
| `SilencioActivoMonitor` | class | `resma4.9.py:158` | `class SilencioActivoMonitor` |
| `__init__` | method | `resma4.9.py:112` | `def __init__(self, garnier, dimension)` |
| `__init__` | method | `resma4.9.py:159` | `def __init__(self, garnier)` |
| `__init__` | method | `resma4.9.py:234` | `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)` |
| `__init__` | method | `resma4.9.py:342` | `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)` |
| `__init__` | method | `resma4.9.py:493` | `def __init__(self, axon_length, radius, n_modes)` |
| `__init__` | method | `resma4.9.py:529` | `def __init__(self, universe, network, myelin)` |
| `__post_init__` | method | `resma4.9.py:74` | `def __post_init__(self)` |
| `__post_init__` | method | `resma4.9.py:193` | `def __post_init__(self)` |
| `_aplicar_modulacion_garnier` | method | `resma4.9.py:310` | `def _aplicar_modulacion_garnier(self, measure)` |
| `_calcular_coherencia` | method | `resma4.9.py:333` | `def _calcular_coherencia(self)` |
| `_calcular_libertad` | method | `resma4.9.py:330` | `def _calcular_libertad(self)` |
| `_calcular_rho_reducida` | method | `resma4.9.py:471` | `def _calcular_rho_reducida(self)` |
| `_compute_betti_numbers` | method | `resma4.9.py:437` | `def _compute_betti_numbers(self)` |
| `_compute_scalar_mass` | method | `resma4.9.py:521` | `def _compute_scalar_mass(self)` |
| `_construct_global_state` | method | `resma4.9.py:321` | `def _construct_global_state(self)` |
| `_construir_generadores_aleatorios` | method | `resma4.9.py:121` | `def _construir_generadores_aleatorios(self)` |
| `_free_hamiltonian` | method | `resma4.9.py:510` | `def _free_hamiltonian(self)` |
| `_generate_complete_measure` | method | `resma4.9.py:281` | `def _generate_complete_measure(self)` |
| `_generate_realistic_modular_network` | method | `resma4.9.py:380` | `def _generate_realistic_modular_network(self)` |
| `_hadamard_generalizado` | method | `resma4.9.py:130` | `def _hadamard_generalizado(self)` |
| `_initialize_leaves` | method | `resma4.9.py:270` | `def _initialize_leaves(self)` |
| `_loss_potential` | method | `resma4.9.py:515` | `def _loss_potential(self)` |
| `_make_serializable` | method | `resma4.9.py:639` | `def _make_serializable(obj)` |
| `_spectral_dimension` | method | `resma4.9.py:445` | `def _spectral_dimension(self)` |
| `_topological_ramsey` | method | `resma4.9.py:467` | `def _topological_ramsey(self)` |
| `_validar_axioma_6` | method | `resma4.9.py:479` | `def _validar_axioma_6(self)` |
| `bures_distance` | method | `resma4.9.py:203` | `def bures_distance(self, other)` |
| `calcular_alpha_modificado` | method | `resma4.9.py:150` | `def calcular_alpha_modificado(self, alpha_base)` |
| `calcular_delta_s_loop` | method | `resma4.9.py:163` | `def calcular_delta_s_loop(self, rho_red, b1)` |
| `cargar_checkpoint` | method | `resma4.9.py:615` | `def cargar_checkpoint(filename)` |
| `compute_log_bayes_factor` | method | `resma4.9.py:534` | `def compute_log_bayes_factor(self)` |
| `epsilon_critico` | method | `resma4.9.py:86` | `def epsilon_critico(self)` |
| `es_silencio_activo` | method | `resma4.9.py:170` | `def es_silencio_activo(self, rho_red, b1)` |
| `from_dict` | method | `resma4.9.py:102` | `def from_dict(cls, data)` |
| `get_memory_gb` | method | `resma4.9.py:577` | `def get_memory_gb()` |
| `guardar_checkpoint` | method | `resma4.9.py:587` | `def guardar_checkpoint(data, filename)` |
| `log_resources` | method | `resma4.9.py:582` | `def log_resources()` |
| `modulation_factor` | method | `resma4.9.py:89` | `def modulation_factor(self)` |
| `operator` | method | `resma4.9.py:135` | `def operator(self)` |
| `simulate_resma_garnier` | method | `resma4.9.py:655` | `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)` |
| `spectral_density` | method | `resma4.9.py:197` | `def spectral_density(self, omega)` |
| `to_dict` | method | `resma4.9.py:92` | `def to_dict(self)` |
| `verify_pt_condition` | method | `resma4.9.py:50` | `def verify_pt_condition(cls)` |
| `E8LatticeLayer` | class | `resma_Test.py:42` | `class E8LatticeLayer(Module)` |
| `PTSymmetricActivation` | class | `resma_Test.py:11` | `class PTSymmetricActivation(Module)` |
| `RESMABrain` | class | `resma_Test.py:71` | `class RESMABrain(Module)` |
| `__init__` | method | `resma_Test.py:12` | `def __init__(self, omega, chi, kappa_init)` |
| `__init__` | method | `resma_Test.py:43` | `def __init__(self, in_features, out_features, q_order)` |
| `__init__` | method | `resma_Test.py:72` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `_generate_ramsey_mask` | method | `resma_Test.py:54` | `def _generate_ramsey_mask(self)` |
| `forward` | method | `resma_Test.py:18` | `def forward(self, x)` |
| `forward` | method | `resma_Test.py:64` | `def forward(self, x)` |
| `forward` | method | `resma_Test.py:78` | `def forward(self, x)` |
| `stress_test_resma` | method | `resma_Test.py:88` | `def stress_test_resma(model)` |
| `E8LatticeLayer` | class | `resmann.py:41` | `class E8LatticeLayer(Module)` |
| `PTSymmetricActivation` | class | `resmann.py:11` | `class PTSymmetricActivation(Module)` |
| `RESMABrain` | class | `resmann.py:86` | `class RESMABrain(Module)` |
| `__init__` | method | `resmann.py:12` | `def __init__(self, omega, chi, kappa_init)` |
| `__init__` | method | `resmann.py:42` | `def __init__(self, in_features, out_features, q_order)` |
| `__init__` | method | `resmann.py:87` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `_generate_ramsey_mask` | method | `resmann.py:59` | `def _generate_ramsey_mask(self)` |
| `forward` | method | `resmann.py:19` | `def forward(self, x)` |
| `forward` | method | `resmann.py:69` | `def forward(self, x)` |
| `forward` | method | `resmann.py:97` | `def forward(self, x)` |
| `resma_loss` | method | `resmann.py:104` | `def resma_loss(self, output, target, lambda_topo)` |
| `E8LatticeMultiverseLayer` | class | `resmann2.py:31` | `class E8LatticeMultiverseLayer(Module)` |
| `PTSymmetricActivation` | class | `resmann2.py:13` | `class PTSymmetricActivation(Module)` |
| `RESMABrainMultiverse` | class | `resmann2.py:65` | `class RESMABrainMultiverse(Module)` |
| `__init__` | method | `resmann2.py:14` | `def __init__(self, omega, chi, kappa_init)` |
| `__init__` | method | `resmann2.py:32` | `def __init__(self, in_features, out_features, n_universes)` |
| `__init__` | method | `resmann2.py:66` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `_multiverse_mask` | method | `resmann2.py:41` | `def _multiverse_mask(self, in_f, out_f, n_univ)` |
| `forward` | method | `resmann2.py:20` | `def forward(self, x)` |
| `forward` | method | `resmann2.py:55` | `def forward(self, x)` |
| `forward` | method | `resmann2.py:74` | `def forward(self, x)` |
| `resma_loss` | method | `resmann2.py:79` | `def resma_loss(self, output, target, lambda_topo)` |
| `E8LatticeLayer` | class | `resmannn.py:30` | `class E8LatticeLayer(Module)` |
| `PTSymmetricActivation` | class | `resmannn.py:12` | `class PTSymmetricActivation(Module)` |
| `RESMABrainLight` | class | `resmannn.py:60` | `class RESMABrainLight(Module)` |
| `__init__` | method | `resmannn.py:13` | `def __init__(self, omega, chi, kappa_init)` |
| `__init__` | method | `resmannn.py:31` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `resmannn.py:61` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `_fixed_sparse_mask` | method | `resmannn.py:40` | `def _fixed_sparse_mask(self, out_f, in_f)` |
| `forward` | method | `resmannn.py:19` | `def forward(self, x)` |
| `forward` | method | `resmannn.py:49` | `def forward(self, x)` |
| `forward` | method | `resmannn.py:69` | `def forward(self, x)` |
| `resma_loss` | method | `resmannn.py:74` | `def resma_loss(self, output, target, lambda_topo)` |
| `create_experiment_summary` | function | `run_complete_experiment.py:24` | `def create_experiment_summary(results_dir, metrics, duration)` |
| `generate_final_report` | function | `run_complete_experiment.py:46` | `def generate_final_report(results_dir, metrics, duration)` |
| `run_complete_experiment` | function | `run_complete_experiment.py:155` | `def run_complete_experiment()` |
| `CombinatorialComplexLayer` | class | `scientific_benchmark.py:132` | `class CombinatorialComplexLayer(Module)` |
| `LearnableAbsenceGating` | class | `scientific_benchmark.py:97` | `class LearnableAbsenceGating(Module)` |
| `PredictiveErrorCell` | class | `scientific_benchmark.py:85` | `class PredictiveErrorCell(Module)` |
| `SupConLoss` | class | `scientific_benchmark.py:57` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `scientific_benchmark.py:111` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainNet` | class | `scientific_benchmark.py:184` | `class TopoBrainNet(Module)` |
| `Wrapper` | class | `scientific_benchmark.py:317` | `class Wrapper(Module)` |
| `__init__` | method | `scientific_benchmark.py:58` | `def __init__(self, temperature)` |
| `__init__` | method | `scientific_benchmark.py:86` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `scientific_benchmark.py:98` | `def __init__(self, dim)` |
| `__init__` | method | `scientific_benchmark.py:112` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `scientific_benchmark.py:133` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)` |
| `__init__` | method | `scientific_benchmark.py:185` | `def __init__(self, config)` |
| `__init__` | method | `scientific_benchmark.py:318` | `def __init__(self, m)` |
| `_init_grid` | method | `scientific_benchmark.py:220` | `def _init_grid(self, N)` |
| `clamp_pgd` | method | `scientific_benchmark.py:276` | `def clamp_pgd(x_adv_norm, x_orig_norm, eps)` |
| `eval_autoattack` | method | `scientific_benchmark.py:299` | `def eval_autoattack(model, test_loader, n_samples)` |
| `forward` | method | `scientific_benchmark.py:62` | `def forward(self, features, labels)` |
| `forward` | method | `scientific_benchmark.py:92` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `scientific_benchmark.py:107` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `scientific_benchmark.py:120` | `def forward(self, x)` |
| `forward` | method | `scientific_benchmark.py:157` | `def forward(self, x_nodes, adjacency, incidence)` |
| `forward` | method | `scientific_benchmark.py:253` | `def forward(self, x)` |
| `forward` | method | `scientific_benchmark.py:319` | `def forward(self, x)` |
| `get_topology` | method | `scientific_benchmark.py:241` | `def get_topology(self)` |
| `lambda_topo` | method | `scientific_benchmark.py:387` | `def lambda_topo(epoch)` |
| `make_adversarial_pgd` | method | `scientific_benchmark.py:283` | `def make_adversarial_pgd(model, x, y, eps, steps)` |
| `run_ablation_suite_scientific` | method | `scientific_benchmark.py:473` | `def run_ablation_suite_scientific()` |
| `run_training` | method | `scientific_benchmark.py:350` | `def run_training(config_override, run_name)` |
| `save_topology_snapshot` | method | `scientific_benchmark.py:332` | `def save_topology_snapshot(model, epoch, run_name)` |
| `seed_everything` | function | `scientific_benchmark.py:42` | `def seed_everything(seed)` |
| `ScientificConfig` | class | `scientist_sinergy_ablation_plan.py:31` | `class ScientificConfig` |
| `check_memory_usage` | method | `scientist_sinergy_ablation_plan.py:63` | `def check_memory_usage()` |
| `generate_sinergy_matrix` | method | `scientist_sinergy_ablation_plan.py:94` | `def generate_sinergy_matrix()` |
| `main` | method | `scientist_sinergy_ablation_plan.py:171` | `def main()` |
| `memory_safe_check` | method | `scientist_sinergy_ablation_plan.py:68` | `def memory_safe_check(config)` |
| `print_sinergy_analysis` | method | `scientist_sinergy_ablation_plan.py:139` | `def print_sinergy_analysis()` |
| `setup_matplotlib_for_plotting` | method | `scientist_sinergy_ablation_plan.py:85` | `def setup_matplotlib_for_plotting()` |
| `check_and_install_dependencies` | function | `setup_environment.py:32` | `def check_and_install_dependencies()` |
| `check_python_version` | function | `setup_environment.py:15` | `def check_python_version()` |
| `create_directories` | function | `setup_environment.py:76` | `def create_directories()` |
| `create_main_script` | function | `setup_environment.py:189` | `def create_main_script()` |
| `create_sample_data` | function | `setup_environment.py:118` | `def create_sample_data()` |
| `install_package` | function | `setup_environment.py:23` | `def install_package(package)` |
| `main` | function | `setup_environment.py:256` | `def main()` |
| `setup_matplotlib` | function | `setup_environment.py:92` | `def setup_matplotlib()` |
| `test_installation` | function | `setup_environment.py:148` | `def test_installation()` |
| `PrismaticNeuron` | class | `sintesis.py:48` | `class PrismaticNeuron(Module)` |
| `SpectralMonitorV6` | class | `sintesis.py:12` | `class SpectralMonitorV6` |
| `SynthesisOrganismV6` | class | `sintesis.py:117` | `class SynthesisOrganismV6(Module)` |
| `__init__` | method | `sintesis.py:13` | `def __init__(self, target_entropy)` |
| `__init__` | method | `sintesis.py:49` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `sintesis.py:118` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `calc_structural_health` | method | `sintesis.py:16` | `def calc_structural_health(self, weight_matrix)` |
| `calculate_losses` | method | `sintesis.py:134` | `def calculate_losses(self, outputs, targets, criterion)` |
| `forward` | method | `sintesis.py:58` | `def forward(self, x)` |
| `forward` | method | `sintesis.py:128` | `def forward(self, x)` |
| `measure_spatial_richness` | method | `sintesis.py:33` | `def measure_spatial_richness(self, activations)` |
| `prismatic_dream` | method | `sintesis.py:81` | `def prismatic_dream(self)` |
| `run_prism_dream` | method | `sintesis.py:162` | `def run_prism_dream()` |
| `sleep` | method | `sintesis.py:154` | `def sleep(self)` |
| `CuriosityGaze` | class | `sintesys2.py:52` | `class CuriosityGaze(Module)` |
| `PrismaticNeuronV7` | class | `sintesys2.py:69` | `class PrismaticNeuronV7(Module)` |
| `SpectralMonitorV7` | class | `sintesys2.py:12` | `class SpectralMonitorV7` |
| `SynthesisOrganismV7` | class | `sintesys2.py:114` | `class SynthesisOrganismV7(Module)` |
| `__init__` | method | `sintesys2.py:13` | `def __init__(self, target_entropy)` |
| `__init__` | method | `sintesys2.py:53` | `def __init__(self, input_dim)` |
| `__init__` | method | `sintesys2.py:70` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `sintesys2.py:115` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `calc_structural_health` | method | `sintesys2.py:16` | `def calc_structural_health(self, weight_matrix)` |
| `calculate_losses` | method | `sintesys2.py:144` | `def calculate_losses(self, outputs, targets, criterion)` |
| `forward` | method | `sintesys2.py:60` | `def forward(self, x)` |
| `forward` | method | `sintesys2.py:79` | `def forward(self, x)` |
| `forward` | method | `sintesys2.py:131` | `def forward(self, x)` |
| `measure_spatial_richness` | method | `sintesys2.py:32` | `def measure_spatial_richness(self, activations)` |
| `prismatic_dream` | method | `sintesys2.py:95` | `def prismatic_dream(self)` |
| `run_the_prisms_eye` | method | `sintesys2.py:174` | `def run_the_prisms_eye()` |
| `sleep` | method | `sintesys2.py:166` | `def sleep(self)` |
| `HomeostasisEngine` | class | `sintesys3.py:30` | `class HomeostasisEngine(Module)` |
| `LiquidNeuron` | class | `sintesys3.py:59` | `class LiquidNeuron(Module)` |
| `OrganismV8` | class | `sintesys3.py:109` | `class OrganismV8(Module)` |
| `__init__` | method | `sintesys3.py:31` | `def __init__(self)` |
| `__init__` | method | `sintesys3.py:60` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `sintesys3.py:110` | `def __init__(self, d_in, d_hid, d_out)` |
| `calc_ent` | method | `sintesys3.py:143` | `def calc_ent(W)` |
| `consolidate_svd` | method | `sintesys3.py:86` | `def consolidate_svd(self, repair_strength)` |
| `decide` | method | `sintesys3.py:35` | `def decide(self, task_loss_val, richness_val, vn_entropy_val, target_entropy)` |
| `forward` | method | `sintesys3.py:68` | `def forward(self, x, plasticity_gate)` |
| `forward` | method | `sintesys3.py:126` | `def forward(self, x, plasticity_gate)` |
| `get_structure_entropy` | method | `sintesys3.py:140` | `def get_structure_entropy(self)` |
| `measure_spatial_richness` | function | `sintesys3.py:12` | `def measure_spatial_richness(activations)` |
| `run_liquid_synthesis` | method | `sintesys3.py:156` | `def run_liquid_synthesis()` |
| `HomeostasisEngine` | class | `sintesys5.py:68` | `class HomeostasisEngine(Module)` |
| `LiquidNeuron` | class | `sintesys5.py:87` | `class LiquidNeuron(Module)` |
| `OrganismV8_Real` | class | `sintesys5.py:124` | `class OrganismV8_Real(Module)` |
| `RealWorldEnvironment` | class | `sintesys5.py:19` | `class RealWorldEnvironment` |
| `__init__` | method | `sintesys5.py:20` | `def __init__(self)` |
| `__init__` | method | `sintesys5.py:69` | `def __init__(self)` |
| `__init__` | method | `sintesys5.py:88` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `sintesys5.py:125` | `def __init__(self, d_in, d_hid, d_out)` |
| `calc_ent` | method | `sintesys5.py:145` | `def calc_ent(W)` |
| `consolidate_svd` | method | `sintesys5.py:111` | `def consolidate_svd(self, repair_strength)` |
| `decide` | method | `sintesys5.py:73` | `def decide(self, task_loss_val, richness_val, vn_entropy_val)` |
| `forward` | method | `sintesys5.py:96` | `def forward(self, x, plasticity_gate)` |
| `forward` | method | `sintesys5.py:135` | `def forward(self, x, plasticity_gate)` |
| `get_batch` | method | `sintesys5.py:38` | `def get_batch(self, phase, batch_size)` |
| `get_structure_entropy` | method | `sintesys5.py:143` | `def get_structure_entropy(self)` |
| `measure_spatial_richness` | method | `sintesys5.py:56` | `def measure_spatial_richness(activations)` |
| `run_real_world_challenge` | method | `sintesys5.py:157` | `def run_real_world_challenge()` |
| `HomeostasisEngine` | class | `syntesys4.py:27` | `class HomeostasisEngine(Module)` |
| `LiquidNeuron` | class | `syntesys4.py:60` | `class LiquidNeuron(Module)` |
| `OrganismV8_1` | class | `syntesys4.py:101` | `class OrganismV8_1(Module)` |
| `__init__` | method | `syntesys4.py:28` | `def __init__(self)` |
| `__init__` | method | `syntesys4.py:61` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `syntesys4.py:102` | `def __init__(self, d_in, d_hid, d_out)` |
| `calc_ent` | method | `syntesys4.py:122` | `def calc_ent(W)` |
| `consolidate_svd` | method | `syntesys4.py:84` | `def consolidate_svd(self, repair_strength)` |
| `decide` | method | `syntesys4.py:32` | `def decide(self, task_loss_val, richness_val, vn_entropy_val)` |
| `forward` | method | `syntesys4.py:69` | `def forward(self, x, plasticity_gate)` |
| `forward` | method | `syntesys4.py:112` | `def forward(self, x, plasticity_gate)` |
| `get_structure_entropy` | method | `syntesys4.py:120` | `def get_structure_entropy(self)` |
| `measure_spatial_richness` | function | `syntesys4.py:12` | `def measure_spatial_richness(activations)` |
| `run_sensitive_self` | method | `syntesys4.py:131` | `def run_sensitive_self()` |
| `ColapsoGarantizado` | class | `test.py:52` | `class ColapsoGarantizado(Module)` |
| `__init__` | method | `test.py:53` | `def __init__(self)` |
| `calculate_test_accuracy` | method | `test.py:109` | `def calculate_test_accuracy(model, testloader)` |
| `forward` | method | `test.py:69` | `def forward(self, x)` |
| `measure_metrics` | method | `test.py:80` | `def measure_metrics(model)` |
| `run_all_tests` | function | `test_premium_synergy.py:209` | `def run_all_tests()` |
| `test_full_system` | function | `test_premium_synergy.py:91` | `def test_full_system()` |
| `test_individual_components` | function | `test_premium_synergy.py:29` | `def test_individual_components()` |
| `test_training_loop` | function | `test_premium_synergy.py:164` | `def test_training_loop()` |
| `CombinatorialComplexLayer` | class | `topobrain.py:205` | `class CombinatorialComplexLayer(Module)` |
| `LearnableAbsenceGating` | class | `topobrain.py:135` | `class LearnableAbsenceGating(Module)` |
| `PredictiveErrorCell` | class | `topobrain.py:175` | `class PredictiveErrorCell(Module)` |
| `ResourceMonitor` | class | `topobrain.py:58` | `class ResourceMonitor` |
| `SupConLoss` | class | `topobrain.py:147` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `topobrain.py:187` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainNet` | class | `topobrain.py:252` | `class TopoBrainNet(Module)` |
| `Wrapper` | class | `topobrain.py:409` | `class Wrapper(Module)` |
| `__init__` | method | `topobrain.py:136` | `def __init__(self, dim)` |
| `__init__` | method | `topobrain.py:148` | `def __init__(self, temperature)` |
| `__init__` | method | `topobrain.py:176` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `topobrain.py:188` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `topobrain.py:206` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)` |
| `__init__` | method | `topobrain.py:253` | `def __init__(self, config)` |
| `__init__` | method | `topobrain.py:410` | `def __init__(self, m)` |
| `_init_grid` | method | `topobrain.py:287` | `def _init_grid(self, N)` |
| `cargar_checkpoint` | method | `topobrain.py:118` | `def cargar_checkpoint(filename)` |
| `check_memory_limit` | method | `topobrain.py:65` | `def check_memory_limit(limit_gb)` |
| `clamp_pgd` | method | `topobrain.py:352` | `def clamp_pgd(x_adv_norm, x_orig_norm, eps)` |
| `clear_cache` | method | `topobrain.py:81` | `def clear_cache()` |
| `eval_autoattack` | method | `topobrain.py:391` | `def eval_autoattack(model, test_loader, n_samples)` |
| `forward` | method | `topobrain.py:143` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `topobrain.py:152` | `def forward(self, features, labels)` |
| `forward` | method | `topobrain.py:182` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `topobrain.py:196` | `def forward(self, x)` |
| `forward` | method | `topobrain.py:229` | `def forward(self, x_nodes, adjacency, incidence, global_step)` |
| `forward` | method | `topobrain.py:324` | `def forward(self, x)` |
| `forward` | method | `topobrain.py:411` | `def forward(self, x)` |
| `get_memory_gb` | method | `topobrain.py:60` | `def get_memory_gb()` |
| `get_topology` | method | `topobrain.py:308` | `def get_topology(self)` |
| `guardar_checkpoint` | method | `topobrain.py:87` | `def guardar_checkpoint(data, filename)` |
| `lambda_topo` | method | `topobrain.py:529` | `def lambda_topo(epoch)` |
| `log_resources` | method | `topobrain.py:72` | `def log_resources()` |
| `make_adversarial_pgd` | method | `topobrain.py:359` | `def make_adversarial_pgd(model, x, y, eps, steps)` |
| `plot_topology_evolution` | method | `topobrain.py:449` | `def plot_topology_evolution(run_name)` |
| `run_diagnostic_suite` | method | `topobrain.py:676` | `def run_diagnostic_suite()` |
| `run_training` | method | `topobrain.py:476` | `def run_training(config_override, run_name)` |
| `save_topology_snapshot` | method | `topobrain.py:423` | `def save_topology_snapshot(model, epoch, run_name)` |
| `seed_everything` | function | `topobrain.py:50` | `def seed_everything(seed)` |
| `CombinatorialComplexLayer` | class | `topobrain_16_3.py:213` | `class CombinatorialComplexLayer(Module)` |
| `LearnableAbsenceGating` | class | `topobrain_16_3.py:140` | `class LearnableAbsenceGating(Module)` |
| `PredictiveErrorCell` | class | `topobrain_16_3.py:183` | `class PredictiveErrorCell(Module)` |
| `ResourceMonitor` | class | `topobrain_16_3.py:70` | `class ResourceMonitor` |
| `SupConLoss` | class | `topobrain_16_3.py:152` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `topobrain_16_3.py:195` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainNet` | class | `topobrain_16_3.py:266` | `class TopoBrainNet(Module)` |
| `Wrapper` | class | `topobrain_16_3.py:409` | `class Wrapper(Module)` |
| `__init__` | method | `topobrain_16_3.py:141` | `def __init__(self, dim)` |
| `__init__` | method | `topobrain_16_3.py:153` | `def __init__(self, temperature)` |
| `__init__` | method | `topobrain_16_3.py:184` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `topobrain_16_3.py:196` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `topobrain_16_3.py:214` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)` |
| `__init__` | method | `topobrain_16_3.py:267` | `def __init__(self, config)` |
| `__init__` | method | `topobrain_16_3.py:410` | `def __init__(self, m)` |
| `_init_grid` | method | `topobrain_16_3.py:299` | `def _init_grid(self, N)` |
| `cargar_checkpoint` | method | `topobrain_16_3.py:123` | `def cargar_checkpoint(filename)` |
| `check_memory_limit` | method | `topobrain_16_3.py:77` | `def check_memory_limit(limit_gb)` |
| `clamp_pgd` | method | `topobrain_16_3.py:361` | `def clamp_pgd(x_adv_norm, x_orig_norm, eps)` |
| `clear_cache` | method | `topobrain_16_3.py:92` | `def clear_cache()` |
| `eval_autoattack` | method | `topobrain_16_3.py:393` | `def eval_autoattack(model, test_loader, n_samples)` |
| `forward` | method | `topobrain_16_3.py:148` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `topobrain_16_3.py:157` | `def forward(self, features, labels)` |
| `forward` | method | `topobrain_16_3.py:190` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `topobrain_16_3.py:204` | `def forward(self, x)` |
| `forward` | method | `topobrain_16_3.py:236` | `def forward(self, x_nodes, adjacency, incidence, global_step)` |
| `forward` | method | `topobrain_16_3.py:333` | `def forward(self, x)` |
| `forward` | method | `topobrain_16_3.py:411` | `def forward(self, x)` |
| `get_memory_gb` | method | `topobrain_16_3.py:72` | `def get_memory_gb()` |
| `get_topology` | method | `topobrain_16_3.py:319` | `def get_topology(self)` |
| `guardar_checkpoint` | method | `topobrain_16_3.py:98` | `def guardar_checkpoint(data, filename)` |
| `lambda_topo` | method | `topobrain_16_3.py:531` | `def lambda_topo(epoch)` |
| `log_resources` | method | `topobrain_16_3.py:84` | `def log_resources()` |
| `make_adversarial_pgd` | method | `topobrain_16_3.py:368` | `def make_adversarial_pgd(model, x, y, eps, steps)` |
| `plot_topology_evolution` | method | `topobrain_16_3.py:447` | `def plot_topology_evolution(run_name)` |
| `run_diagnostic_suite` | method | `topobrain_16_3.py:726` | `def run_diagnostic_suite()` |
| `run_training` | method | `topobrain_16_3.py:464` | `def run_training(config_override, run_name)` |
| `save_topology_snapshot` | method | `topobrain_16_3.py:423` | `def save_topology_snapshot(model, epoch, run_name)` |
| `seed_everything` | function | `topobrain_16_3.py:61` | `def seed_everything(seed)` |
| `AdaptiveCombinatorialComplexLayer` | class | `topobrain_v18.1.py:473` | `class AdaptiveCombinatorialComplexLayer(Module)` |
| `AsymmetricPredictiveErrorCell` | class | `topobrain_v18.1.py:398` | `class AsymmetricPredictiveErrorCell(Module)` |
| `CheckpointManager` | class | `topobrain_v18.1.py:260` | `class CheckpointManager` |
| `Config` | class | `topobrain_v18.1.py:30` | `class Config` |
| `LearnableAbsenceGating` | class | `topobrain_v18.1.py:437` | `class LearnableAbsenceGating(Module)` |
| `ResourceMonitor` | class | `topobrain_v18.1.py:109` | `class ResourceMonitor` |
| `SupConLoss` | class | `topobrain_v18.1.py:366` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `topobrain_v18.1.py:453` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainNetV18` | class | `topobrain_v18.1.py:598` | `class TopoBrainNetV18(Module)` |
| `TopologicalHealthSovereignty` | class | `topobrain_v18.1.py:152` | `class TopologicalHealthSovereignty` |
| `TopologyMetrics` | class | `topobrain_v18.1.py:143` | `class TopologyMetrics` |
| `__init__` | method | `topobrain_v18.1.py:159` | `def __init__(self, model, config, epsilon_c)` |
| `__init__` | method | `topobrain_v18.1.py:261` | `def __init__(self, checkpoint_dir)` |
| `__init__` | method | `topobrain_v18.1.py:368` | `def __init__(self, temperature)` |
| `__init__` | method | `topobrain_v18.1.py:403` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `topobrain_v18.1.py:439` | `def __init__(self, dim)` |
| `__init__` | method | `topobrain_v18.1.py:455` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `topobrain_v18.1.py:480` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)` |
| `__init__` | method | `topobrain_v18.1.py:606` | `def __init__(self, config, in_channels)` |
| `__post_init__` | method | `topobrain_v18.1.py:82` | `def __post_init__(self)` |
| `_analyze_matrix` | method | `topobrain_v18.1.py:165` | `def _analyze_matrix(self, weight_matrix, name)` |
| `_init_grid_topology` | method | `topobrain_v18.1.py:653` | `def _init_grid_topology(self, N)` |
| `analyze_topology_clustering` | method | `topobrain_v18.1.py:1234` | `def analyze_topology_clustering(model, run_name)` |
| `analyze_topology_evolution` | method | `topobrain_v18.1.py:1409` | `def analyze_topology_evolution(run_name)` |
| `analyze_topology_flow` | method | `topobrain_v18.1.py:1277` | `def analyze_topology_flow(model, dataloader, run_name, num_samples)` |
| `calculate` | method | `topobrain_v18.1.py:220` | `def calculate(self, epoch)` |
| `calculate_ortho_loss` | method | `topobrain_v18.1.py:708` | `def calculate_ortho_loss(self)` |
| `check_limit` | method | `topobrain_v18.1.py:135` | `def check_limit(limit_gb)` |
| `clear_cache` | method | `topobrain_v18.1.py:129` | `def clear_cache()` |
| `comprehensive_topology_analysis` | method | `topobrain_v18.1.py:1478` | `def comprehensive_topology_analysis(model, dataloader, run_name)` |
| `evaluate` | method | `topobrain_v18.1.py:1139` | `def evaluate(model, test_loader, config, adversarial)` |
| `forward` | method | `topobrain_v18.1.py:372` | `def forward(self, features, labels)` |
| `forward` | method | `topobrain_v18.1.py:415` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `topobrain_v18.1.py:448` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `topobrain_v18.1.py:463` | `def forward(self, x)` |
| `forward` | method | `topobrain_v18.1.py:509` | `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse)` |
| `forward` | method | `topobrain_v18.1.py:787` | `def forward(self, x)` |
| `get_critical_summary` | method | `topobrain_v18.1.py:247` | `def get_critical_summary(self)` |
| `get_dataloaders` | method | `topobrain_v18.1.py:316` | `def get_dataloaders(config)` |
| `get_dataset_stats` | method | `topobrain_v18.1.py:309` | `def get_dataset_stats(dataset_name)` |
| `get_gpu_memory_gb` | method | `topobrain_v18.1.py:116` | `def get_gpu_memory_gb()` |
| `get_memory_gb` | method | `topobrain_v18.1.py:111` | `def get_memory_gb()` |
| `get_node_importance` | method | `topobrain_v18.1.py:588` | `def get_node_importance(self)` |
| `get_supcon_lambda` | method | `topobrain_v18.1.py:89` | `def get_supcon_lambda(self, epoch)` |
| `get_topology` | method | `topobrain_v18.1.py:681` | `def get_topology(self, return_sparse)` |
| `load` | method | `topobrain_v18.1.py:288` | `def load(self, name)` |
| `log` | method | `topobrain_v18.1.py:122` | `def log(prefix)` |
| `main` | method | `topobrain_v18.1.py:1633` | `def main()` |
| `make_adversarial_pgd` | method | `topobrain_v18.1.py:820` | `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)` |
| `prune_topology` | method | `topobrain_v18.1.py:741` | `def prune_topology(self)` |
| `run_ablation_study` | method | `topobrain_v18.1.py:1509` | `def run_ablation_study()` |
| `save` | method | `topobrain_v18.1.py:265` | `def save(self, data, name)` |
| `save_node_importance_viz` | method | `topobrain_v18.1.py:1211` | `def save_node_importance_viz(model, epoch, run_name)` |
| `save_topology_visualization` | method | `topobrain_v18.1.py:1166` | `def save_topology_visualization(model, epoch, run_name)` |
| `seed_everything` | method | `topobrain_v18.1.py:100` | `def seed_everything(seed)` |
| `set_epoch` | method | `topobrain_v18.1.py:813` | `def set_epoch(self, epoch)` |
| `to_dict` | method | `topobrain_v18.1.py:86` | `def to_dict(self)` |
| `train_epoch` | method | `topobrain_v18.1.py:848` | `def train_epoch(model, train_loader, optimizer, opt_topo, supcon, config, epoch, topo_monitor)` |
| `train_model` | method | `topobrain_v18.1.py:956` | `def train_model(config, run_name)` |
| `visualize_topology_as_graph` | method | `topobrain_v18.1.py:1343` | `def visualize_topology_as_graph(model, run_name, threshold)` |
| `warmup_topo` | method | `topobrain_v18.1.py:1006` | `def warmup_topo(epoch)` |
| `AdaptiveCombinatorialComplexLayer` | class | `topobrain_v18.2.py:473` | `class AdaptiveCombinatorialComplexLayer(Module)` |
| `AsymmetricPredictiveErrorCell` | class | `topobrain_v18.2.py:398` | `class AsymmetricPredictiveErrorCell(Module)` |
| `CheckpointManager` | class | `topobrain_v18.2.py:260` | `class CheckpointManager` |
| `Config` | class | `topobrain_v18.2.py:30` | `class Config` |
| `LearnableAbsenceGating` | class | `topobrain_v18.2.py:437` | `class LearnableAbsenceGating(Module)` |
| `ResourceMonitor` | class | `topobrain_v18.2.py:109` | `class ResourceMonitor` |
| `SupConLoss` | class | `topobrain_v18.2.py:366` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `topobrain_v18.2.py:453` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainNetV18` | class | `topobrain_v18.2.py:599` | `class TopoBrainNetV18(Module)` |
| `TopologicalHealthSovereignty` | class | `topobrain_v18.2.py:152` | `class TopologicalHealthSovereignty` |
| `TopologyMetrics` | class | `topobrain_v18.2.py:143` | `class TopologyMetrics` |
| `__init__` | method | `topobrain_v18.2.py:159` | `def __init__(self, model, config, epsilon_c)` |
| `__init__` | method | `topobrain_v18.2.py:261` | `def __init__(self, checkpoint_dir)` |
| `__init__` | method | `topobrain_v18.2.py:368` | `def __init__(self, temperature)` |
| `__init__` | method | `topobrain_v18.2.py:403` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `topobrain_v18.2.py:439` | `def __init__(self, dim)` |
| `__init__` | method | `topobrain_v18.2.py:455` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `topobrain_v18.2.py:480` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)` |
| `__init__` | method | `topobrain_v18.2.py:608` | `def __init__(self, config, in_channels)` |
| `__post_init__` | method | `topobrain_v18.2.py:82` | `def __post_init__(self)` |
| `_analyze_matrix` | method | `topobrain_v18.2.py:165` | `def _analyze_matrix(self, weight_matrix, name)` |
| `_init_grid_topology` | method | `topobrain_v18.2.py:668` | `def _init_grid_topology(self, N)` |
| `analyze_topology_clustering` | method | `topobrain_v18.2.py:1183` | `def analyze_topology_clustering(model, run_name)` |
| `analyze_topology_evolution` | method | `topobrain_v18.2.py:1358` | `def analyze_topology_evolution(run_name)` |
| `analyze_topology_flow` | method | `topobrain_v18.2.py:1226` | `def analyze_topology_flow(model, dataloader, run_name, num_samples)` |
| `calculate` | method | `topobrain_v18.2.py:220` | `def calculate(self, epoch)` |
| `calculate_ortho_loss` | method | `topobrain_v18.2.py:729` | `def calculate_ortho_loss(self)` |
| `check_limit` | method | `topobrain_v18.2.py:135` | `def check_limit(limit_gb)` |
| `clear_cache` | method | `topobrain_v18.2.py:129` | `def clear_cache()` |
| `comprehensive_topology_analysis` | method | `topobrain_v18.2.py:1427` | `def comprehensive_topology_analysis(model, dataloader, run_name)` |
| `evaluate` | method | `topobrain_v18.2.py:1088` | `def evaluate(model, test_loader, config, adversarial)` |
| `forward` | method | `topobrain_v18.2.py:372` | `def forward(self, features, labels)` |
| `forward` | method | `topobrain_v18.2.py:415` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `topobrain_v18.2.py:448` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `topobrain_v18.2.py:463` | `def forward(self, x)` |
| `forward` | method | `topobrain_v18.2.py:510` | `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse)` |
| `forward` | method | `topobrain_v18.2.py:793` | `def forward(self, x)` |
| `get_critical_summary` | method | `topobrain_v18.2.py:247` | `def get_critical_summary(self)` |
| `get_dataloaders` | method | `topobrain_v18.2.py:316` | `def get_dataloaders(config)` |
| `get_dataset_stats` | method | `topobrain_v18.2.py:309` | `def get_dataset_stats(dataset_name)` |
| `get_gpu_memory_gb` | method | `topobrain_v18.2.py:116` | `def get_gpu_memory_gb()` |
| `get_memory_gb` | method | `topobrain_v18.2.py:111` | `def get_memory_gb()` |
| `get_node_importance` | method | `topobrain_v18.2.py:589` | `def get_node_importance(self)` |
| `get_supcon_lambda` | method | `topobrain_v18.2.py:89` | `def get_supcon_lambda(self, epoch)` |
| `get_topology` | method | `topobrain_v18.2.py:705` | `def get_topology(self, return_sparse)` |
| `load` | method | `topobrain_v18.2.py:288` | `def load(self, name)` |
| `log` | method | `topobrain_v18.2.py:122` | `def log(prefix)` |
| `main` | method | `topobrain_v18.2.py:1582` | `def main()` |
| `make_adversarial_pgd` | method | `topobrain_v18.2.py:824` | `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)` |
| `prune_topology` | method | `topobrain_v18.2.py:754` | `def prune_topology(self)` |
| `run_ablation_study` | method | `topobrain_v18.2.py:1458` | `def run_ablation_study()` |
| `save` | method | `topobrain_v18.2.py:265` | `def save(self, data, name)` |
| `save_node_importance_viz` | method | `topobrain_v18.2.py:1160` | `def save_node_importance_viz(model, epoch, run_name)` |
| `save_topology_visualization` | method | `topobrain_v18.2.py:1115` | `def save_topology_visualization(model, epoch, run_name)` |
| `seed_everything` | method | `topobrain_v18.2.py:100` | `def seed_everything(seed)` |
| `set_epoch` | method | `topobrain_v18.2.py:818` | `def set_epoch(self, epoch)` |
| `to_dict` | method | `topobrain_v18.2.py:86` | `def to_dict(self)` |
| `train_epoch` | method | `topobrain_v18.2.py:852` | `def train_epoch(model, train_loader, optimizer, opt_topo, supcon, config, epoch, topo_monitor)` |
| `train_model` | method | `topobrain_v18.2.py:960` | `def train_model(config, run_name)` |
| `visualize_topology_as_graph` | method | `topobrain_v18.2.py:1292` | `def visualize_topology_as_graph(model, run_name, threshold)` |
| `warmup_topo` | method | `topobrain_v18.2.py:1016` | `def warmup_topo(epoch)` |
| `AdaptiveCombinatorialComplexLayer` | class | `topobrain_v18.py:473` | `class AdaptiveCombinatorialComplexLayer(Module)` |
| `AsymmetricPredictiveErrorCell` | class | `topobrain_v18.py:398` | `class AsymmetricPredictiveErrorCell(Module)` |
| `CheckpointManager` | class | `topobrain_v18.py:260` | `class CheckpointManager` |
| `Config` | class | `topobrain_v18.py:30` | `class Config` |
| `LearnableAbsenceGating` | class | `topobrain_v18.py:437` | `class LearnableAbsenceGating(Module)` |
| `ResourceMonitor` | class | `topobrain_v18.py:109` | `class ResourceMonitor` |
| `SupConLoss` | class | `topobrain_v18.py:366` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `topobrain_v18.py:453` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainNetV18` | class | `topobrain_v18.py:598` | `class TopoBrainNetV18(Module)` |
| `TopologicalHealthSovereignty` | class | `topobrain_v18.py:152` | `class TopologicalHealthSovereignty` |
| `TopologyMetrics` | class | `topobrain_v18.py:143` | `class TopologyMetrics` |
| `__init__` | method | `topobrain_v18.py:159` | `def __init__(self, model, config, epsilon_c)` |
| `__init__` | method | `topobrain_v18.py:261` | `def __init__(self, checkpoint_dir)` |
| `__init__` | method | `topobrain_v18.py:368` | `def __init__(self, temperature)` |
| `__init__` | method | `topobrain_v18.py:403` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `topobrain_v18.py:439` | `def __init__(self, dim)` |
| `__init__` | method | `topobrain_v18.py:455` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `topobrain_v18.py:480` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)` |
| `__init__` | method | `topobrain_v18.py:606` | `def __init__(self, config, in_channels)` |
| `__post_init__` | method | `topobrain_v18.py:82` | `def __post_init__(self)` |
| `_analyze_matrix` | method | `topobrain_v18.py:165` | `def _analyze_matrix(self, weight_matrix, name)` |
| `_init_grid_topology` | method | `topobrain_v18.py:653` | `def _init_grid_topology(self, N)` |
| `analyze_topology_clustering` | method | `topobrain_v18.py:1325` | `def analyze_topology_clustering(model, run_name)` |
| `analyze_topology_evolution` | method | `topobrain_v18.py:1500` | `def analyze_topology_evolution(run_name)` |
| `analyze_topology_flow` | method | `topobrain_v18.py:1368` | `def analyze_topology_flow(model, dataloader, run_name, num_samples)` |
| `calculate` | method | `topobrain_v18.py:220` | `def calculate(self, epoch)` |
| `calculate_ortho_loss` | method | `topobrain_v18.py:708` | `def calculate_ortho_loss(self)` |
| `check_limit` | method | `topobrain_v18.py:135` | `def check_limit(limit_gb)` |
| `clear_cache` | method | `topobrain_v18.py:129` | `def clear_cache()` |
| `comprehensive_topology_analysis` | method | `topobrain_v18.py:1569` | `def comprehensive_topology_analysis(model, dataloader, run_name)` |
| `evaluate` | method | `topobrain_v18.py:1078` | `def evaluate(model, test_loader, config, adversarial)` |
| `forward` | method | `topobrain_v18.py:372` | `def forward(self, features, labels)` |
| `forward` | method | `topobrain_v18.py:415` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `topobrain_v18.py:448` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `topobrain_v18.py:463` | `def forward(self, x)` |
| `forward` | method | `topobrain_v18.py:509` | `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse)` |
| `forward` | method | `topobrain_v18.py:769` | `def forward(self, x)` |
| `get_critical_summary` | method | `topobrain_v18.py:247` | `def get_critical_summary(self)` |
| `get_dataloaders` | method | `topobrain_v18.py:316` | `def get_dataloaders(config)` |
| `get_dataset_stats` | method | `topobrain_v18.py:309` | `def get_dataset_stats(dataset_name)` |
| `get_gpu_memory_gb` | method | `topobrain_v18.py:116` | `def get_gpu_memory_gb()` |
| `get_memory_gb` | method | `topobrain_v18.py:111` | `def get_memory_gb()` |
| `get_node_importance` | method | `topobrain_v18.py:588` | `def get_node_importance(self)` |
| `get_supcon_lambda` | method | `topobrain_v18.py:89` | `def get_supcon_lambda(self, epoch)` |
| `get_topology` | method | `topobrain_v18.py:681` | `def get_topology(self, return_sparse)` |
| `lambda_topo` | method | `topobrain_v18.py:948` | `def lambda_topo(epoch)` |
| `lambda_topo` | method | `topobrain_v18.py:1130` | `def lambda_topo(epoch)` |
| `load` | method | `topobrain_v18.py:288` | `def load(self, name)` |
| `log` | method | `topobrain_v18.py:122` | `def log(prefix)` |
| `main` | method | `topobrain_v18.py:1724` | `def main()` |
| `make_adversarial_pgd` | method | `topobrain_v18.py:799` | `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)` |
| `prune_topology` | method | `topobrain_v18.py:741` | `def prune_topology(self)` |
| `run_ablation_study` | method | `topobrain_v18.py:1600` | `def run_ablation_study()` |
| `save` | method | `topobrain_v18.py:265` | `def save(self, data, name)` |
| `save_node_importance_viz` | method | `topobrain_v18.py:1302` | `def save_node_importance_viz(model, epoch, run_name)` |
| `save_topology_visualization` | method | `topobrain_v18.py:1257` | `def save_topology_visualization(model, epoch, run_name)` |
| `seed_everything` | method | `topobrain_v18.py:100` | `def seed_everything(seed)` |
| `to_dict` | method | `topobrain_v18.py:86` | `def to_dict(self)` |
| `train_epoch` | method | `topobrain_v18.py:827` | `def train_epoch(model, train_loader, optimizer, opt_topo, supcon, config, epoch, topo_monitor)` |
| `train_model` | method | `topobrain_v18.py:917` | `def train_model(config, run_name)` |
| `train_model` | method | `topobrain_v18.py:1101` | `def train_model(config, run_name)` |
| `visualize_topology_as_graph` | method | `topobrain_v18.py:1434` | `def visualize_topology_as_graph(model, run_name, threshold)` |
| `CheckpointManager` | class | `topobrain_v19.py:139` | `class CheckpointManager` |
| `Config` | class | `topobrain_v19.py:42` | `class Config` |
| `ContrastiveLoss` | class | `topobrain_v19.py:521` | `class ContrastiveLoss(Module)` |
| `DynamicTopologicalLayer` | class | `topobrain_v19.py:291` | `class DynamicTopologicalLayer(Module)` |
| `NodePositionLearner` | class | `topobrain_v19.py:240` | `class NodePositionLearner(Module)` |
| `ResourceMonitor` | class | `topobrain_v19.py:107` | `class ResourceMonitor` |
| `TopoBrainNetV19` | class | `topobrain_v19.py:360` | `class TopoBrainNetV19(Module)` |
| `__init__` | method | `topobrain_v19.py:140` | `def __init__(self, checkpoint_dir)` |
| `__init__` | method | `topobrain_v19.py:245` | `def __init__(self, num_nodes, node_dim, k)` |
| `__init__` | method | `topobrain_v19.py:296` | `def __init__(self, in_dim, hid_dim, config, layer_idx)` |
| `__init__` | method | `topobrain_v19.py:364` | `def __init__(self, config, in_channels)` |
| `__init__` | method | `topobrain_v19.py:523` | `def __init__(self, temperature)` |
| `apply_pruning` | method | `topobrain_v19.py:445` | `def apply_pruning(self, edge_index, edge_weight)` |
| `calculate_ortho_loss` | method | `topobrain_v19.py:473` | `def calculate_ortho_loss(self)` |
| `check_limit` | method | `topobrain_v19.py:134` | `def check_limit(limit_gb)` |
| `clear_cache` | method | `topobrain_v19.py:128` | `def clear_cache()` |
| `evaluate` | method | `topobrain_v19.py:621` | `def evaluate(model, test_loader, config, adversarial)` |
| `forward` | method | `topobrain_v19.py:254` | `def forward(self, batch_size)` |
| `forward` | method | `topobrain_v19.py:325` | `def forward(self, x, edge_index, edge_weight, batch)` |
| `forward` | method | `topobrain_v19.py:398` | `def forward(self, x)` |
| `forward` | method | `topobrain_v19.py:527` | `def forward(self, features, labels)` |
| `get_dataloaders` | method | `topobrain_v19.py:189` | `def get_dataloaders(config)` |
| `get_dataset_stats` | method | `topobrain_v19.py:182` | `def get_dataset_stats(dataset_name)` |
| `get_gpu_memory_gb` | method | `topobrain_v19.py:114` | `def get_gpu_memory_gb()` |
| `get_memory_gb` | method | `topobrain_v19.py:109` | `def get_memory_gb()` |
| `load` | method | `topobrain_v19.py:163` | `def load(self, name)` |
| `log` | method | `topobrain_v19.py:120` | `def log(prefix)` |
| `main` | method | `topobrain_v19.py:918` | `def main()` |
| `make_adversarial_pgd` | method | `topobrain_v19.py:493` | `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)` |
| `prune_fn` | method | `topobrain_v19.py:467` | `def prune_fn(edge_idx)` |
| `prune_structural` | method | `topobrain_v19.py:453` | `def prune_structural(self, threshold)` |
| `run_ablation_study` | method | `topobrain_v19.py:837` | `def run_ablation_study()` |
| `save` | method | `topobrain_v19.py:144` | `def save(self, data, name)` |
| `save_topology_snapshot` | method | `topobrain_v19.py:796` | `def save_topology_snapshot(model, epoch, run_name)` |
| `seed_everything` | method | `topobrain_v19.py:98` | `def seed_everything(seed)` |
| `to_dict` | method | `topobrain_v19.py:91` | `def to_dict(self)` |
| `train_epoch` | method | `topobrain_v19.py:545` | `def train_epoch(model, train_loader, optimizer, criterion, contrastive_loss, config, epoch)` |
| `train_model` | method | `topobrain_v19.py:644` | `def train_model(config, run_name)` |
| `warmup_lr` | method | `topobrain_v19.py:668` | `def warmup_lr(epoch)` |
| `CombinatorialComplexLayer` | class | `train_Adversarial.py:332` | `class CombinatorialComplexLayer(Module)` |
| `ContinuumMemorySystem` | class | `train_Adversarial.py:218` | `class ContinuumMemorySystem(Module)` |
| `LearnableAbsenceGating` | class | `train_Adversarial.py:300` | `class LearnableAbsenceGating(Module)` |
| `NestedOptimizer` | class | `train_Adversarial.py:154` | `class NestedOptimizer(Optimizer)` |
| `PredictiveErrorCell` | class | `train_Adversarial.py:288` | `class PredictiveErrorCell(Module)` |
| `ResourceMonitor` | class | `train_Adversarial.py:74` | `class ResourceMonitor` |
| `SupConLoss` | class | `train_Adversarial.py:260` | `class SupConLoss(Module)` |
| `SymbioticBasisRefinement` | class | `train_Adversarial.py:314` | `class SymbioticBasisRefinement(Module)` |
| `TopoBrainNet` | class | `train_Adversarial.py:400` | `class TopoBrainNet(Module)` |
| `Wrapper` | class | `train_Adversarial.py:552` | `class Wrapper(Module)` |
| `__init__` | method | `train_Adversarial.py:159` | `def __init__(self, params, lr, momentum, nested_levels, freq_factor)` |
| `__init__` | method | `train_Adversarial.py:223` | `def __init__(self, input_dim, hidden_dim, num_levels)` |
| `__init__` | method | `train_Adversarial.py:261` | `def __init__(self, temperature)` |
| `__init__` | method | `train_Adversarial.py:289` | `def __init__(self, dim, use_spectral)` |
| `__init__` | method | `train_Adversarial.py:301` | `def __init__(self, dim)` |
| `__init__` | method | `train_Adversarial.py:315` | `def __init__(self, dim, num_atoms)` |
| `__init__` | method | `train_Adversarial.py:333` | `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)` |
| `__init__` | method | `train_Adversarial.py:401` | `def __init__(self, config)` |
| `__init__` | method | `train_Adversarial.py:553` | `def __init__(self, m)` |
| `_init_grid` | method | `train_Adversarial.py:444` | `def _init_grid(self, N)` |
| `apply_cms` | method | `train_Adversarial.py:391` | `def apply_cms(self, x, global_step)` |
| `cargar_checkpoint` | method | `train_Adversarial.py:134` | `def cargar_checkpoint(filename)` |
| `clamp_pgd` | method | `train_Adversarial.py:509` | `def clamp_pgd(x_adv_norm, x_orig_norm, eps)` |
| `clear_cache` | method | `train_Adversarial.py:90` | `def clear_cache()` |
| `create_checkpoint_data` | method | `train_Adversarial.py:566` | `def create_checkpoint_data(model, optimizer, epoch, config, metrics)` |
| `eval_autoattack` | method | `train_Adversarial.py:534` | `def eval_autoattack(model, test_loader, n_samples)` |
| `forward` | method | `train_Adversarial.py:240` | `def forward(self, x)` |
| `forward` | method | `train_Adversarial.py:265` | `def forward(self, features, labels)` |
| `forward` | method | `train_Adversarial.py:295` | `def forward(self, input_signal, prediction)` |
| `forward` | method | `train_Adversarial.py:310` | `def forward(self, x_sensory, x_prediction)` |
| `forward` | method | `train_Adversarial.py:323` | `def forward(self, x)` |
| `forward` | method | `train_Adversarial.py:362` | `def forward(self, x_nodes, adjacency, incidence, global_step)` |
| `forward` | method | `train_Adversarial.py:477` | `def forward(self, x)` |
| `forward` | method | `train_Adversarial.py:554` | `def forward(self, x)` |
| `get_memory_gb` | method | `train_Adversarial.py:78` | `def get_memory_gb()` |
| `get_topology` | method | `train_Adversarial.py:465` | `def get_topology(self)` |
| `get_update_mask` | method | `train_Adversarial.py:250` | `def get_update_mask(self, level_idx, batch_size)` |
| `guardar_checkpoint` | method | `train_Adversarial.py:100` | `def guardar_checkpoint(data, filename)` |
| `lambda_topo` | method | `train_Adversarial.py:632` | `def lambda_topo(epoch)` |
| `log_resources` | method | `train_Adversarial.py:83` | `def log_resources()` |
| `make_adversarial_pgd` | method | `train_Adversarial.py:516` | `def make_adversarial_pgd(model, x, y, eps, steps)` |
| `run_diagnostic_suite` | method | `train_Adversarial.py:787` | `def run_diagnostic_suite()` |
| `run_training` | method | `train_Adversarial.py:586` | `def run_training(config_override, run_name)` |
| `save_topology_snapshot` | method | `train_Adversarial.py:770` | `def save_topology_snapshot(model, epoch, run_name)` |
| `seed_everything` | function | `train_Adversarial.py:61` | `def seed_everything(seed)` |
| `should_update_level` | method | `train_Adversarial.py:246` | `def should_update_level(self, level_idx, global_step)` |
| `step` | method | `train_Adversarial.py:179` | `def step(self, closure)` |
| `AudioEncoder` | class | `tricameral2.py:678` | `class AudioEncoder(Module)` |
| `CorpusCallosumTrimodal` | class | `tricameral2.py:821` | `class CorpusCallosumTrimodal(Module)` |
| `Flickr8kMultimodalDataset` | class | `tricameral2.py:584` | `class Flickr8kMultimodalDataset(Dataset)` |
| `Flickr8kSimpleDataset` | class | `tricameral2.py:1195` | `class Flickr8kSimpleDataset(BaseDataset)` |
| `LeftHemisphere` | class | `tricameral2.py:512` | `class LeftHemisphere(Module)` |
| `NeuralAudioGenerator` | class | `tricameral2.py:913` | `class NeuralAudioGenerator(Module)` |
| `NeuroLogosTricameral` | class | `tricameral2.py:982` | `class NeuroLogosTricameral(Module)` |
| `RightHemisphereTricameral` | class | `tricameral2.py:747` | `class RightHemisphereTricameral(Module)` |
| `StableLiquidNeuron` | class | `tricameral2.py:389` | `class StableLiquidNeuron(Module)` |
| `__getitem__` | method | `tricameral2.py:628` | `def __getitem__(self, idx)` |
| `__getitem__` | method | `tricameral2.py:1217` | `def __getitem__(self, idx)` |
| `__init__` | method | `tricameral2.py:390` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `tricameral2.py:513` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `tricameral2.py:587` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)` |
| `__init__` | method | `tricameral2.py:681` | `def __init__(self, output_dim)` |
| `__init__` | method | `tricameral2.py:750` | `def __init__(self, output_dim)` |
| `__init__` | method | `tricameral2.py:824` | `def __init__(self, dim)` |
| `__init__` | method | `tricameral2.py:916` | `def __init__(self, text_dim, output_sr)` |
| `__init__` | method | `tricameral2.py:985` | `def __init__(self, vocab_size)` |
| `__init__` | method | `tricameral2.py:1196` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `tricameral2.py:625` | `def __len__(self)` |
| `__len__` | method | `tricameral2.py:1214` | `def __len__(self)` |
| `_get_init_state` | method | `tricameral2.py:570` | `def _get_init_state(self, visual_context)` |
| `_greedy_decode` | method | `tricameral2.py:548` | `def _greedy_decode(self, visual_context, max_len, device)` |
| `build_vocab_flickr` | function | `tricameral2.py:366` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `compute_tricameral_loss` | method | `tricameral2.py:1048` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_` |
| `download_from_github` | function | `tricameral2.py:183` | `def download_from_github(repo_url, output_dir)` |
| `forward` | method | `tricameral2.py:425` | `def forward(self, x)` |
| `forward` | method | `tricameral2.py:525` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `tricameral2.py:720` | `def forward(self, mel_spec)` |
| `forward` | method | `tricameral2.py:784` | `def forward(self, image, audio)` |
| `forward` | method | `tricameral2.py:856` | `def forward(self, right_features)` |
| `forward` | method | `tricameral2.py:951` | `def forward(self, text_embedding)` |
| `forward` | method | `tricameral2.py:1000` | `def forward(self, image, audio, captions, epoch, generate_audio)` |
| `generate_all_audios_batch` | function | `tricameral2.py:66` | `def generate_all_audios_batch(captions_list, audio_dir, batch_size)` |
| `generate_audio_async` | function | `tricameral2.py:31` | `def generate_audio_async(text, output_path, voice, max_retries)` |
| `generate_audios_sync` | function | `tricameral2.py:135` | `def generate_audios_sync(images_dir, captions_file, audio_dir)` |
| `hebbian_update` | method | `tricameral2.py:438` | `def hebbian_update(self, post, pre, plasticity)` |
| `setup_flickr8k` | function | `tricameral2.py:257` | `def setup_flickr8k(data_dir, github_url)` |
| `train_tricameral` | method | `tricameral2.py:1100` | `def train_tricameral(github_repo_url)` |
| `update_physiology_advanced` | method | `tricameral2.py:476` | `def update_physiology_advanced(self, loss_value)` |
| `AudioEncoder` | class | `tricameral_kimi.py:843` | `class AudioEncoder(Module)` |
| `CorpusCallosumTrimodal` | class | `tricameral_kimi.py:965` | `class CorpusCallosumTrimodal(Module)` |
| `EnhancedDiagnosticsTricameral` | class | `tricameral_kimi.py:1067` | `class EnhancedDiagnosticsTricameral` |
| `EpisodicMemoryBuffer` | class | `tricameral_kimi.py:215` | `class EpisodicMemoryBuffer` |
| `Flickr8kMultimodalDataset` | class | `tricameral_kimi.py:1296` | `class Flickr8kMultimodalDataset(Dataset)` |
| `LanguageMetrics` | class | `tricameral_kimi.py:331` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `tricameral_kimi.py:503` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `tricameral_kimi.py:769` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `tricameral_kimi.py:405` | `class LinguisticFeedbackLoop` |
| `NeuroLogosTricameral` | class | `tricameral_kimi.py:1261` | `class NeuroLogosTricameral(Module)` |
| `NeurocognitiveSystem` | class | `tricameral_kimi.py:255` | `class NeurocognitiveSystem` |
| `RightHemisphereTricameral` | class | `tricameral_kimi.py:893` | `class RightHemisphereTricameral(Module)` |
| `StableLiquidNeuron` | class | `tricameral_kimi.py:550` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `tricameral_kimi.py:673` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `tricameral_kimi.py:1351` | `def __getitem__(self, idx)` |
| `__init__` | method | `tricameral_kimi.py:216` | `def __init__(self, capacity, surprise_threshold)` |
| `__init__` | method | `tricameral_kimi.py:256` | `def __init__(self)` |
| `__init__` | method | `tricameral_kimi.py:406` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `tricameral_kimi.py:551` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `tricameral_kimi.py:674` | `def __init__(self)` |
| `__init__` | method | `tricameral_kimi.py:770` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `tricameral_kimi.py:846` | `def __init__(self, output_dim)` |
| `__init__` | method | `tricameral_kimi.py:896` | `def __init__(self, output_dim)` |
| `__init__` | method | `tricameral_kimi.py:966` | `def __init__(self, dim)` |
| `__init__` | method | `tricameral_kimi.py:1068` | `def __init__(self)` |
| `__init__` | method | `tricameral_kimi.py:1264` | `def __init__(self, vocab_size)` |
| `__init__` | method | `tricameral_kimi.py:1299` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)` |
| `__len__` | method | `tricameral_kimi.py:1347` | `def __len__(self)` |
| `_get_init_state` | method | `tricameral_kimi.py:828` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `tricameral_kimi.py:369` | `def _get_ngrams(tokens, n)` |
| `_get_ngrams` | method | `tricameral_kimi.py:478` | `def _get_ngrams(self, sentence, n)` |
| `_greedy_decode` | method | `tricameral_kimi.py:806` | `def _greedy_decode(self, visual_context, max_len, device)` |
| `add` | method | `tricameral_kimi.py:232` | `def add(self, image, caption, surprise_score)` |
| `adjust_gates_by_fatigue` | method | `tricameral_kimi.py:1054` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `tricameral_kimi.py:287` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_triangulated_intervention` | method | `tricameral_kimi.py:703` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `tricameral_kimi.py:266` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `build_vocab_flickr` | function | `tricameral_kimi.py:191` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `tricameral_kimi.py:1137` | `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `tricameral_kimi.py:1127` | `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_cider` | method | `tricameral_kimi.py:443` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `tricameral_kimi.py:417` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_spice` | method | `tricameral_kimi.py:469` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `tricameral_kimi.py:222` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `compute_tricameral_loss` | method | `tricameral_kimi.py:1402` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_` |
| `diagnose_with_triangulation` | method | `tricameral_kimi.py:680` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `evaluate_reasoning_quality` | method | `tricameral_kimi.py:1103` | `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` |
| `forward` | method | `tricameral_kimi.py:586` | `def forward(self, x)` |
| `forward` | method | `tricameral_kimi.py:781` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `tricameral_kimi.py:880` | `def forward(self, mel_spec)` |
| `forward` | method | `tricameral_kimi.py:929` | `def forward(self, image, audio)` |
| `forward` | method | `tricameral_kimi.py:995` | `def forward(self, right_features)` |
| `forward` | method | `tricameral_kimi.py:1270` | `def forward(self, image, audio, captions, epoch)` |
| `get_cache_stats` | method | `tricameral_kimi.py:482` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `tricameral_kimi.py:1156` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `tricameral_kimi.py:599` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `tricameral_kimi.py:1084` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `report` | method | `tricameral_kimi.py:1202` | `def report(self, epoch)` |
| `sample` | method | `tricameral_kimi.py:241` | `def sample(self, batch_size)` |
| `sentence_bleu` | method | `tricameral_kimi.py:335` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `tricameral_kimi.py:505` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k_with_audio` | function | `tricameral_kimi.py:36` | `def setup_flickr8k_with_audio(data_dir)` |
| `token_accuracy` | method | `tricameral_kimi.py:378` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `tricameral_kimi.py:528` | `def token_accuracy(reference, hypothesis)` |
| `train_tricameral` | method | `tricameral_kimi.py:1452` | `def train_tricameral()` |
| `update` | method | `tricameral_kimi.py:1146` | `def update(self)` |
| `update_channel_fatigue` | method | `tricameral_kimi.py:1038` | `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` |
| `update_physiology_advanced` | method | `tricameral_kimi.py:639` | `def update_physiology_advanced(self, loss_value)` |
| `visualize_fatigue_distribution` | method | `tricameral_kimi.py:1173` | `def visualize_fatigue_distribution(self, epoch)` |
| `visualize_reasoning_metrics` | method | `tricameral_kimi.py:1191` | `def visualize_reasoning_metrics(self, epoch)` |
| `word_overlap` | method | `tricameral_kimi.py:391` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `tricameral_kimi.py:538` | `def word_overlap(reference, hypothesis)` |
| `AudioEncoder` | class | `tricameral_kimi2.py:1401` | `class AudioEncoder(Module)` |
| `CorpusCallosumTrimodal` | class | `tricameral_kimi2.py:1523` | `class CorpusCallosumTrimodal(Module)` |
| `EnhancedDiagnosticsTricameral` | class | `tricameral_kimi2.py:1676` | `class EnhancedDiagnosticsTricameral` |
| `EpisodicMemoryBuffer` | class | `tricameral_kimi2.py:262` | `class EpisodicMemoryBuffer` |
| `Flickr8kMultimodalDataset` | class | `tricameral_kimi2.py:1987` | `class Flickr8kMultimodalDataset(Dataset)` |
| `LanguageMetrics` | class | `tricameral_kimi2.py:539` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `tricameral_kimi2.py:725` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `tricameral_kimi2.py:774` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `tricameral_kimi2.py:1103` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `tricameral_kimi2.py:613` | `class LinguisticFeedbackLoop` |
| `NeuroLogosTricameral` | class | `tricameral_kimi2.py:1952` | `class NeuroLogosTricameral(Module)` |
| `NeurocognitiveSystem` | class | `tricameral_kimi2.py:347` | `class NeurocognitiveSystem` |
| `RightHemisphereTricameral` | class | `tricameral_kimi2.py:1451` | `class RightHemisphereTricameral(Module)` |
| `StableLiquidNeuron` | class | `tricameral_kimi2.py:821` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `tricameral_kimi2.py:944` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `tricameral_kimi2.py:2042` | `def __getitem__(self, idx)` |
| `__init__` | method | `tricameral_kimi2.py:263` | `def __init__(self, capacity, surprise_threshold)` |
| `__init__` | method | `tricameral_kimi2.py:348` | `def __init__(self)` |
| `__init__` | method | `tricameral_kimi2.py:614` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `tricameral_kimi2.py:822` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `tricameral_kimi2.py:945` | `def __init__(self)` |
| `__init__` | method | `tricameral_kimi2.py:1104` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `tricameral_kimi2.py:1404` | `def __init__(self, output_dim)` |
| `__init__` | method | `tricameral_kimi2.py:1454` | `def __init__(self, output_dim)` |
| `__init__` | method | `tricameral_kimi2.py:1524` | `def __init__(self, dim)` |
| `__init__` | method | `tricameral_kimi2.py:1677` | `def __init__(self)` |
| `__init__` | method | `tricameral_kimi2.py:1955` | `def __init__(self, vocab_size)` |
| `__init__` | method | `tricameral_kimi2.py:1990` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)` |
| `__len__` | method | `tricameral_kimi2.py:2038` | `def __len__(self)` |
| `_apply_chain_of_thought` | method | `tricameral_kimi2.py:1222` | `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)` |
| `_apply_multi_token_prediction` | method | `tricameral_kimi2.py:1323` | `def _apply_multi_token_prediction(self, hidden_states, input_ids)` |
| `_apply_structural_attention` | method | `tricameral_kimi2.py:1365` | `def _apply_structural_attention(self, lstm_out, channels, visual_context)` |
| `_get_cached_norm` | method | `tricameral_kimi2.py:1699` | `def _get_cached_norm(self, tensor, dim)` |
| `_get_init_state` | method | `tricameral_kimi2.py:1386` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `tricameral_kimi2.py:577` | `def _get_ngrams(tokens, n)` |
| `_greedy_decode` | method | `tricameral_kimi2.py:1262` | `def _greedy_decode(self, visual_context, channels, max_len, epoch)` |
| `_reset_liquid_neuron` | method | `tricameral_kimi2.py:1088` | `def _reset_liquid_neuron(self, liquid_neuron)` |
| `add` | method | `tricameral_kimi2.py:289` | `def add(self, image, audio, caption, surprise_score)` |
| `adjust_gates_by_fatigue` | method | `tricameral_kimi2.py:1655` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `tricameral_kimi2.py:453` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_triangulated_intervention` | method | `tricameral_kimi2.py:1018` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `tricameral_kimi2.py:407` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `assess_reasoning_state` | method | `tricameral_kimi2.py:363` | `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` |
| `build_vocab_flickr` | function | `tricameral_kimi2.py:238` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `cached_ngrams` | method | `tricameral_kimi2.py:623` | `def cached_ngrams(sentence, n)` |
| `calculate_health` | method | `tricameral_kimi2.py:1795` | `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `tricameral_kimi2.py:1784` | `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_alignment_loss` | method | `tricameral_kimi2.py:2091` | `def compute_alignment_loss(visual_features, channels, alpha, epoch)` |
| `compute_cider` | method | `tricameral_kimi2.py:675` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `tricameral_kimi2.py:636` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_spice` | method | `tricameral_kimi2.py:695` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `tricameral_kimi2.py:278` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `compute_tricameral_loss` | method | `tricameral_kimi2.py:2119` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_` |
| `count_convergent_signals` | method | `tricameral_kimi2.py:962` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `tricameral_kimi2.py:965` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)` |
| `evaluate_reasoning_quality` | method | `tricameral_kimi2.py:1747` | `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` |
| `forward` | method | `tricameral_kimi2.py:857` | `def forward(self, x)` |
| `forward` | method | `tricameral_kimi2.py:1179` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `tricameral_kimi2.py:1438` | `def forward(self, mel_spec)` |
| `forward` | method | `tricameral_kimi2.py:1487` | `def forward(self, image, audio)` |
| `forward` | method | `tricameral_kimi2.py:1572` | `def forward(self, right_features)` |
| `forward` | method | `tricameral_kimi2.py:1961` | `def forward(self, image, audio, captions, epoch)` |
| `get_cache_stats` | method | `tricameral_kimi2.py:704` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `tricameral_kimi2.py:1821` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `tricameral_kimi2.py:870` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `tricameral_kimi2.py:1717` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `report` | method | `tricameral_kimi2.py:1873` | `def report(self, epoch)` |
| `sample` | method | `tricameral_kimi2.py:318` | `def sample(self, batch_size)` |
| `sentence_bleu` | method | `tricameral_kimi2.py:543` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `tricameral_kimi2.py:727` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `tricameral_kimi2.py:776` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k_with_audio` | function | `tricameral_kimi2.py:50` | `def setup_flickr8k_with_audio(data_dir)` |
| `token_accuracy` | method | `tricameral_kimi2.py:586` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `tricameral_kimi2.py:750` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `tricameral_kimi2.py:799` | `def token_accuracy(reference, hypothesis)` |
| `train_tricameral` | method | `tricameral_kimi2.py:2166` | `def train_tricameral()` |
| `triangulate_signals` | method | `tricameral_kimi2.py:951` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `tricameral_kimi2.py:1804` | `def update(self)` |
| `update_channel_fatigue` | method | `tricameral_kimi2.py:1633` | `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` |
| `update_physiology_advanced` | method | `tricameral_kimi2.py:910` | `def update_physiology_advanced(self, loss_value)` |
| `visualize_fatigue_distribution` | method | `tricameral_kimi2.py:1837` | `def visualize_fatigue_distribution(self, epoch)` |
| `visualize_reasoning_metrics` | method | `tricameral_kimi2.py:1861` | `def visualize_reasoning_metrics(self, epoch)` |
| `word_overlap` | method | `tricameral_kimi2.py:599` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `tricameral_kimi2.py:760` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `tricameral_kimi2.py:809` | `def word_overlap(reference, hypothesis)` |
| `AudioEncoder` | class | `tricameralkimi2.py:668` | `class AudioEncoder(Module)` |
| `CorpusCallosumTrimodal` | class | `tricameralkimi2.py:952` | `class CorpusCallosumTrimodal(Module)` |
| `EnhancedDiagnosticsTricameral` | class | `tricameralkimi2.py:1556` | `class EnhancedDiagnosticsTricameral` |
| `EpisodicMemoryBuffer` | class | `tricameralkimi2.py:195` | `class EpisodicMemoryBuffer` |
| `Flickr8kMultimodalDataset` | class | `tricameralkimi2.py:1415` | `class Flickr8kMultimodalDataset(Dataset)` |
| `LeftHemisphere` | class | `tricameralkimi2.py:1089` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `tricameralkimi2.py:425` | `class LinguisticFeedbackLoop` |
| `NeuroLogosTricameral` | class | `tricameralkimi2.py:1359` | `class NeuroLogosTricameral(Module)` |
| `NeurocognitiveSystem` | class | `tricameralkimi2.py:247` | `class NeurocognitiveSystem` |
| `RightHemisphereTricameral` | class | `tricameralkimi2.py:881` | `class RightHemisphereTricameral(Module)` |
| `StableLiquidNeuron` | class | `tricameralkimi2.py:546` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `tricameralkimi2.py:724` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `tricameralkimi2.py:1466` | `def __getitem__(self, idx)` |
| `__init__` | method | `tricameralkimi2.py:198` | `def __init__(self, capacity, surprise_threshold)` |
| `__init__` | method | `tricameralkimi2.py:248` | `def __init__(self)` |
| `__init__` | method | `tricameralkimi2.py:428` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `tricameralkimi2.py:547` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `tricameralkimi2.py:671` | `def __init__(self, output_dim)` |
| `__init__` | method | `tricameralkimi2.py:725` | `def __init__(self)` |
| `__init__` | method | `tricameralkimi2.py:884` | `def __init__(self, output_dim)` |
| `__init__` | method | `tricameralkimi2.py:955` | `def __init__(self, dim)` |
| `__init__` | method | `tricameralkimi2.py:1090` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `tricameralkimi2.py:1362` | `def __init__(self, vocab_size)` |
| `__init__` | method | `tricameralkimi2.py:1418` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)` |
| `__init__` | method | `tricameralkimi2.py:1557` | `def __init__(self)` |
| `__len__` | method | `tricameralkimi2.py:1463` | `def __len__(self)` |
| `_apply_chain_of_thought` | method | `tricameralkimi2.py:1237` | `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)` |
| `_apply_multi_token_prediction` | method | `tricameralkimi2.py:1271` | `def _apply_multi_token_prediction(self, hidden_states, input_ids)` |
| `_apply_structural_attention` | method | `tricameralkimi2.py:1317` | `def _apply_structural_attention(self, lstm_out, channels, visual_context)` |
| `_get_init_state` | method | `tricameralkimi2.py:1344` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `tricameralkimi2.py:518` | `def _get_ngrams(self, sentence, n)` |
| `_greedy_decode` | method | `tricameralkimi2.py:1204` | `def _greedy_decode(self, visual_context, channels, max_len, epoch)` |
| `_reset_liquid_neuron` | method | `tricameralkimi2.py:868` | `def _reset_liquid_neuron(self, right_node, severity)` |
| `add` | method | `tricameralkimi2.py:216` | `def add(self, image, caption, surprise_score)` |
| `adjust_gates_by_fatigue` | method | `tricameralkimi2.py:1079` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `tricameralkimi2.py:341` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_triangulated_intervention` | method | `tricameralkimi2.py:786` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `tricameralkimi2.py:305` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `assess_reasoning_state` | method | `tricameralkimi2.py:263` | `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` |
| `build_vocab_flickr` | function | `tricameralkimi2.py:173` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `tricameralkimi2.py:1638` | `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `tricameralkimi2.py:1627` | `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_cider` | method | `tricameralkimi2.py:476` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `tricameralkimi2.py:442` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_spice` | method | `tricameralkimi2.py:505` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `tricameralkimi2.py:204` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `compute_tricameral_loss` | method | `tricameralkimi2.py:1507` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_` |
| `count_convergent_signals` | method | `tricameralkimi2.py:741` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `tricameralkimi2.py:744` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `evaluate_reasoning_quality` | method | `tricameralkimi2.py:1595` | `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` |
| `forward` | method | `tricameralkimi2.py:582` | `def forward(self, x)` |
| `forward` | method | `tricameralkimi2.py:705` | `def forward(self, mel_spec)` |
| `forward` | method | `tricameralkimi2.py:917` | `def forward(self, image, audio)` |
| `forward` | method | `tricameralkimi2.py:991` | `def forward(self, right_features)` |
| `forward` | method | `tricameralkimi2.py:1165` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `tricameralkimi2.py:1374` | `def forward(self, image, audio, captions, epoch)` |
| `get_cache_stats` | method | `tricameralkimi2.py:523` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `tricameralkimi2.py:1657` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `tricameralkimi2.py:595` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `tricameralkimi2.py:1572` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `report` | method | `tricameralkimi2.py:1707` | `def report(self, epoch)` |
| `sample` | method | `tricameralkimi2.py:228` | `def sample(self, batch_size)` |
| `setup_flickr8k_with_audio` | function | `tricameralkimi2.py:30` | `def setup_flickr8k_with_audio(data_dir)` |
| `train_tricameral` | method | `tricameralkimi2.py:1786` | `def train_tricameral()` |
| `triangulate_signals` | method | `tricameralkimi2.py:731` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `tricameralkimi2.py:1647` | `def update(self)` |
| `update_channel_fatigue` | method | `tricameralkimi2.py:1060` | `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` |
| `update_physiology_advanced` | method | `tricameralkimi2.py:633` | `def update_physiology_advanced(self, loss_value)` |
| `visualize_fatigue_distribution` | method | `tricameralkimi2.py:1674` | `def visualize_fatigue_distribution(self, epoch)` |
| `visualize_reasoning_metrics` | method | `tricameralkimi2.py:1695` | `def visualize_reasoning_metrics(self, epoch)` |
| `AudioEncoder` | class | `trycameral.py:517` | `class AudioEncoder(Module)` |
| `CorpusCallosumTrimodal` | class | `trycameral.py:660` | `class CorpusCallosumTrimodal(Module)` |
| `Flickr8kMultimodalDataset` | class | `trycameral.py:423` | `class Flickr8kMultimodalDataset(Dataset)` |
| `Flickr8kSimpleDataset` | class | `trycameral.py:996` | `class Flickr8kSimpleDataset(BaseDataset)` |
| `LeftHemisphere` | class | `trycameral.py:351` | `class LeftHemisphere(Module)` |
| `NeuralAudioGenerator` | class | `trycameral.py:752` | `class NeuralAudioGenerator(Module)` |
| `NeuroLogosTricameral` | class | `trycameral.py:821` | `class NeuroLogosTricameral(Module)` |
| `RightHemisphereTricameral` | class | `trycameral.py:586` | `class RightHemisphereTricameral(Module)` |
| `StableLiquidNeuron` | class | `trycameral.py:228` | `class StableLiquidNeuron(Module)` |
| `__getitem__` | method | `trycameral.py:467` | `def __getitem__(self, idx)` |
| `__getitem__` | method | `trycameral.py:1018` | `def __getitem__(self, idx)` |
| `__init__` | method | `trycameral.py:229` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `trycameral.py:352` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `trycameral.py:426` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)` |
| `__init__` | method | `trycameral.py:520` | `def __init__(self, output_dim)` |
| `__init__` | method | `trycameral.py:589` | `def __init__(self, output_dim)` |
| `__init__` | method | `trycameral.py:663` | `def __init__(self, dim)` |
| `__init__` | method | `trycameral.py:755` | `def __init__(self, text_dim, output_sr)` |
| `__init__` | method | `trycameral.py:824` | `def __init__(self, vocab_size)` |
| `__init__` | method | `trycameral.py:997` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `trycameral.py:464` | `def __len__(self)` |
| `__len__` | method | `trycameral.py:1015` | `def __len__(self)` |
| `_get_init_state` | method | `trycameral.py:409` | `def _get_init_state(self, visual_context)` |
| `_greedy_decode` | method | `trycameral.py:387` | `def _greedy_decode(self, visual_context, max_len, device)` |
| `build_vocab_flickr` | function | `trycameral.py:205` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `compute_tricameral_loss` | method | `trycameral.py:887` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_` |
| `forward` | method | `trycameral.py:264` | `def forward(self, x)` |
| `forward` | method | `trycameral.py:364` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `trycameral.py:559` | `def forward(self, mel_spec)` |
| `forward` | method | `trycameral.py:623` | `def forward(self, image, audio)` |
| `forward` | method | `trycameral.py:695` | `def forward(self, right_features)` |
| `forward` | method | `trycameral.py:790` | `def forward(self, text_embedding)` |
| `forward` | method | `trycameral.py:839` | `def forward(self, image, audio, captions, epoch, generate_audio)` |
| `generate_all_audios_batch` | function | `trycameral.py:50` | `def generate_all_audios_batch(captions_list, audio_dir, batch_size)` |
| `generate_audio_async` | function | `trycameral.py:31` | `def generate_audio_async(text, output_path, voice)` |
| `generate_audios_sync` | function | `trycameral.py:89` | `def generate_audios_sync(images_dir, captions_file, audio_dir)` |
| `hebbian_update` | method | `trycameral.py:277` | `def hebbian_update(self, post, pre, plasticity)` |
| `setup_flickr8k` | function | `trycameral.py:133` | `def setup_flickr8k(data_dir)` |
| `train_tricameral` | method | `trycameral.py:939` | `def train_tricameral()` |
| `update_physiology_advanced` | method | `trycameral.py:315` | `def update_physiology_advanced(self, loss_value)` |
| `BioDecoder` | class | `ultimo_neuorlogos.py:221` | `class BioDecoder(Module)` |
| `CIFARCaptions` | class | `ultimo_neuorlogos.py:330` | `class CIFARCaptions` |
| `ConsciousCore` | class | `ultimo_neuorlogos.py:203` | `class ConsciousCore(Module)` |
| `MiniUnconscious` | class | `ultimo_neuorlogos.py:145` | `class MiniUnconscious(Module)` |
| `NeuroLogos` | class | `ultimo_neuorlogos.py:282` | `class NeuroLogos(Module)` |
| `PGDAttack` | class | `ultimo_neuorlogos.py:113` | `class PGDAttack` |
| `TopoBrainCore` | class | `ultimo_neuorlogos.py:27` | `class TopoBrainCore(Module)` |
| `TopoUnconscious` | class | `ultimo_neuorlogos.py:166` | `class TopoUnconscious(Module)` |
| `__getitem__` | method | `ultimo_neuorlogos.py:359` | `def __getitem__(self, idx)` |
| `__init__` | method | `ultimo_neuorlogos.py:29` | `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)` |
| `__init__` | method | `ultimo_neuorlogos.py:115` | `def __init__(self, epsilon, alpha, steps)` |
| `__init__` | method | `ultimo_neuorlogos.py:147` | `def __init__(self, output_dim)` |
| `__init__` | method | `ultimo_neuorlogos.py:168` | `def __init__(self, output_dim, use_grid, use_symbiotic)` |
| `__init__` | method | `ultimo_neuorlogos.py:205` | `def __init__(self, dim)` |
| `__init__` | method | `ultimo_neuorlogos.py:223` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `ultimo_neuorlogos.py:289` | `def __init__(self, vocab_size, mode)` |
| `__init__` | method | `ultimo_neuorlogos.py:331` | `def __init__(self)` |
| `__len__` | method | `ultimo_neuorlogos.py:356` | `def __len__(self)` |
| `_get_init_state` | method | `ultimo_neuorlogos.py:272` | `def _get_init_state(self, thought)` |
| `_init_grid` | method | `ultimo_neuorlogos.py:62` | `def _init_grid(self)` |
| `attack` | method | `ultimo_neuorlogos.py:120` | `def attack(self, model, x, y, criterion)` |
| `forward` | method | `ultimo_neuorlogos.py:70` | `def forward(self, x)` |
| `forward` | method | `ultimo_neuorlogos.py:160` | `def forward(self, x)` |
| `forward` | method | `ultimo_neuorlogos.py:191` | `def forward(self, x)` |
| `forward` | method | `ultimo_neuorlogos.py:210` | `def forward(self, x)` |
| `forward` | method | `ultimo_neuorlogos.py:238` | `def forward(self, thought, captions, max_len)` |
| `forward` | method | `ultimo_neuorlogos.py:314` | `def forward(self, image, captions)` |
| `get_metrics` | method | `ultimo_neuorlogos.py:101` | `def get_metrics(self)` |
| `get_metrics` | method | `ultimo_neuorlogos.py:195` | `def get_metrics(self)` |
| `get_metrics` | method | `ultimo_neuorlogos.py:319` | `def get_metrics(self)` |
| `run_full_ablation` | method | `ultimo_neuorlogos.py:508` | `def run_full_ablation(epochs, device)` |
| `train_ablation` | method | `ultimo_neuorlogos.py:374` | `def train_ablation(mode, epochs, device)` |
| `CorpusCallosum` | class | `ultimobicameral.py:443` | `class CorpusCallosum(Module)` |
| `EnhancedDiagnostics` | class | `ultimobicameral.py:483` | `class EnhancedDiagnostics` |
| `Flickr8kDataset` | class | `ultimobicameral.py:608` | `class Flickr8kDataset(Dataset)` |
| `LanguageMetrics` | class | `ultimobicameral.py:23` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `ultimobicameral.py:383` | `class LeftHemisphere(Module)` |
| `MedicalSystem` | class | `ultimobicameral.py:96` | `class MedicalSystem` |
| `NeuroLogosBicameralStable` | class | `ultimobicameral.py:462` | `class NeuroLogosBicameralStable(Module)` |
| `RightHemisphere` | class | `ultimobicameral.py:368` | `class RightHemisphere(Module)` |
| `StableLiquidNeuron` | class | `ultimobicameral.py:292` | `class StableLiquidNeuron(Module)` |
| `__getitem__` | method | `ultimobicameral.py:629` | `def __getitem__(self, idx)` |
| `__init__` | method | `ultimobicameral.py:99` | `def __init__(self)` |
| `__init__` | method | `ultimobicameral.py:293` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `ultimobicameral.py:369` | `def __init__(self, output_dim)` |
| `__init__` | method | `ultimobicameral.py:384` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `ultimobicameral.py:444` | `def __init__(self, dim)` |
| `__init__` | method | `ultimobicameral.py:463` | `def __init__(self, vocab_size)` |
| `__init__` | method | `ultimobicameral.py:484` | `def __init__(self)` |
| `__init__` | method | `ultimobicameral.py:609` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__len__` | method | `ultimobicameral.py:626` | `def __len__(self)` |
| `_get_init_state` | method | `ultimobicameral.py:438` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `ultimobicameral.py:61` | `def _get_ngrams(tokens, n)` |
| `apply_intervention` | method | `ultimobicameral.py:149` | `def apply_intervention(self, model, issues, severity, epoch)` |
| `build_vocab_flickr` | method | `ultimobicameral.py:645` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `ultimobicameral.py:512` | `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_synergy` | method | `ultimobicameral.py:503` | `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `diagnose_severity` | method | `ultimobicameral.py:103` | `def diagnose_severity(self, health_score, liquid_norm, gate_mean, callosal_flow)` |
| `forward` | method | `ultimobicameral.py:307` | `def forward(self, x)` |
| `forward` | method | `ultimobicameral.py:377` | `def forward(self, image)` |
| `forward` | method | `ultimobicameral.py:401` | `def forward(self, visual_context, captions, max_len)` |
| `forward` | method | `ultimobicameral.py:456` | `def forward(self, right_features)` |
| `forward` | method | `ultimobicameral.py:469` | `def forward(self, image, captions)` |
| `get_recent_avg` | method | `ultimobicameral.py:526` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `ultimobicameral.py:314` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `ultimobicameral.py:494` | `def measure_callosal_flow(self, right_features, left_context)` |
| `report` | method | `ultimobicameral.py:531` | `def report(self, epoch)` |
| `sentence_bleu` | method | `ultimobicameral.py:27` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k` | method | `ultimobicameral.py:663` | `def setup_flickr8k(data_dir)` |
| `token_accuracy` | method | `ultimobicameral.py:70` | `def token_accuracy(reference, hypothesis)` |
| `train_with_metrics` | method | `ultimobicameral.py:677` | `def train_with_metrics()` |
| `update` | method | `ultimobicameral.py:521` | `def update(self)` |
| `update_physiology_advanced` | method | `ultimobicameral.py:344` | `def update_physiology_advanced(self, loss_value)` |
| `word_overlap` | method | `ultimobicameral.py:83` | `def word_overlap(reference, hypothesis)` |

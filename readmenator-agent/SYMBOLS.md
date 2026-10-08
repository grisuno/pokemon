# Symbols (page 1 of 13)
Pages: [SYMBOLS.md](SYMBOLS.md), [SYMBOLS_p2.md](SYMBOLS_p2.md), [SYMBOLS_p3.md](SYMBOLS_p3.md), [SYMBOLS_p4.md](SYMBOLS_p4.md), [SYMBOLS_p5.md](SYMBOLS_p5.md), [SYMBOLS_p6.md](SYMBOLS_p6.md), [SYMBOLS_p7.md](SYMBOLS_p7.md), [SYMBOLS_p8.md](SYMBOLS_p8.md), [SYMBOLS_p9.md](SYMBOLS_p9.md), [SYMBOLS_p10.md](SYMBOLS_p10.md), [SYMBOLS_p11.md](SYMBOLS_p11.md), [SYMBOLS_p12.md](SYMBOLS_p12.md), [SYMBOLS_p13.md](SYMBOLS_p13.md)

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

Next: [SYMBOLS_p2.md](SYMBOLS_p2.md)

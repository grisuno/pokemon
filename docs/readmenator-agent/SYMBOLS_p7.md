# Symbols (page 7 of 13)
Previous: [SYMBOLS_p6.md](SYMBOLS_p6.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
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
| `__init__` | method | `neurologos_tricameral_exodia.py:2349` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache...` |
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

Next: [SYMBOLS_p8.md](SYMBOLS_p8.md)

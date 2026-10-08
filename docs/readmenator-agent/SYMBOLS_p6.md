# Symbols (page 6 of 13)
Previous: [SYMBOLS_p5.md](SYMBOLS_p5.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
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

Next: [SYMBOLS_p7.md](SYMBOLS_p7.md)

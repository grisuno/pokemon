# Symbols (page 8 of 13)
Previous: [SYMBOLS_p7.md](SYMBOLS_p7.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `compute_spice` | method | `neurologos_tricameral_exodia.py:911` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `neurologos_tricameral_exodia.py:359` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `compute_tricameral_loss` | method | `neurologos_tricameral_exodia.py:2490` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward...` |
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

Next: [SYMBOLS_p9.md](SYMBOLS_p9.md)

# Symbols (page 12 of 13)
Previous: [SYMBOLS_p11.md](SYMBOLS_p11.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
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
| `compute_tricameral_loss` | method | `tricameral2.py:1048` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward...` |
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
| `compute_tricameral_loss` | method | `tricameral_kimi.py:1402` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward...` |
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
| `compute_tricameral_loss` | method | `tricameral_kimi2.py:2119` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward...` |
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
| `compute_tricameral_loss` | method | `tricameralkimi2.py:1507` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward...` |
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
| `compute_tricameral_loss` | method | `trycameral.py:887` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward...` |
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

Next: [SYMBOLS_p13.md](SYMBOLS_p13.md)

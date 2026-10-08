# Symbols (page 9 of 13)
Previous: [SYMBOLS_p8.md](SYMBOLS_p8.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
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

Next: [SYMBOLS_p10.md](SYMBOLS_p10.md)

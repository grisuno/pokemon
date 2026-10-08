# API (page 3 of 10)
Previous: [API_p2.md](API_p2.md)

## bicameral_v3.py
- `compute_phi_effective` (function) `bicameral_v3.py:22` `def compute_phi_effective(activations, k_partitions)` -- Φₑ con manejo robusto de dimensiones pequeñas
- `compute_spatial_diversity` (function) `bicameral_v3.py:75` `def compute_spatial_diversity(activations)` -- Diversidad basada en correlación inversa de Pearson.
- `compute_activation_entropy` (function) `bicameral_v3.py:132` `def compute_activation_entropy(activations)` -- Shannon entropy sobre la distribución de activaciones Target range: [2.5, 4.5] bits
- `measure_neural_complexity` (function) `bicameral_v3.py:163` `def measure_neural_complexity(activations)` -- Medición corregida con formato [B, D, N] para neuronas reales
- `measure_spatial_richness` (function) `bicameral_v3.py:212` `def measure_spatial_richness(activations)` -- Wrapper para compatibilidad con código existente
- `top_k_top_p_filtering` (function) `bicameral_v3.py:221` `def top_k_top_p_filtering(logits, top_k, top_p, filter_value)` -- Filtro Top-K y Nucleus Sampling estándar
- `BCMPlasticity.__init__` (method) `bicameral_v3.py:242` `def __init__(self, neurons, tau_theta)`
- `BCMPlasticity.forward` (method) `bicameral_v3.py:247` `def forward(self, activity, dt)` -- dθ/dt = (E[activity²] - θ)/τ  →  dw/dt ∝ activity*(activity-θ)
- `BioDecoder.__init__` (method) `bicameral_v3.py:258` `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)`
- `BioDecoder.forward` (method) `bicameral_v3.py:281` `def forward(self, thought, visual_features, captions, max_len)`
- `LiquidNeuron.__init__` (method) `bicameral_v3.py:384` `def __init__(self, in_dim, out_dim)`
- `LiquidNeuron.forward` (method) `bicameral_v3.py:405` `def forward(self, x, global_plasticity, transfer_rate)`
- `LiquidNeuron.consolidate_svd` (method) `bicameral_v3.py:467` `def consolidate_svd(self, repair_strength, timescale)`
- `ResidualBlock.__init__` (method) `bicameral_v3.py:487` `def __init__(self, in_channels, out_channels, stride)`
- `ResidualBlock.forward` (method) `bicameral_v3.py:499` `def forward(self, x)`
- `VisualCortex.__init__` (method) `bicameral_v3.py:506` `def __init__(self, output_dim, grid_size)`
- `VisualCortex.forward` (method) `bicameral_v3.py:526` `def forward(self, x)`
- `SymbioticBasisRefinement.__init__` (method) `bicameral_v3.py:542` `def __init__(self, dim, num_atoms)`
- `SymbioticBasisRefinement.forward` (method) `bicameral_v3.py:552` `def forward(self, x)`
- `AdaptiveCombinatorialComplexLayer.__init__` (method) `bicameral_v3.py:562` `def __init__(self, in_dim, hid_dim, num_nodes, config)`
- `AdaptiveCombinatorialComplexLayer.forward` (method) `bicameral_v3.py:568` `def forward(self, x, plasticity_gate)`
- `GraphNeuralLayer.__init__` (method) `bicameral_v3.py:573` `def __init__(self, dim, hidden_dim)`
- `GraphNeuralLayer.forward` (method) `bicameral_v3.py:583` `def forward(self, nodes, adjacency)`
- `GraphNeuralLayer.create_grid_adjacency` (method) `bicameral_v3.py:588` `def create_grid_adjacency(N, connectivity)` -- Crea matriz de adyacencia para grid cuadrado
- `RightHemisphere.__init__` (method) `bicameral_v3.py:604` `def __init__(self, config)`
- `RightHemisphere.forward` (method) `bicameral_v3.py:636` `def forward(self, image, adjacency, plasticity)`
- `MiniUnconscious.__init__` (method) `bicameral_v3.py:668` `def __init__(self)`
- `MiniUnconscious.forward` (method) `bicameral_v3.py:681` `def forward(self, x)`
- `NestedUnconscious.__init__` (method) `bicameral_v3.py:685` `def __init__(self, grid_size, output_dim)`
- `NestedUnconscious.forward` (method) `bicameral_v3.py:705` `def forward(self, x)`
- `TopologicalCompressor.__init__` (method) `bicameral_v3.py:728` `def __init__(self, node_dim)`
- `TopologicalCompressor.forward` (method) `bicameral_v3.py:737` `def forward(self, nodes, plasticity, transfer_rate)`
- `ConsciousCore.__init__` (method) `bicameral_v3.py:745` `def __init__(self)`
- `ConsciousCore.forward` (method) `bicameral_v3.py:790` `def forward(self, visual_features, plasticity, transfer_rate)`
- `ConsciousCore.get_liquid_module` (method) `bicameral_v3.py:839` `def get_liquid_module(self)`
- `LeftHemisphere.__init__` (method) `bicameral_v3.py:847` `def __init__(self, use_nested)`
- `LeftHemisphere.forward` (method) `bicameral_v3.py:853` `def forward(self, image, callosal_input, plasticity, transfer_rate)`
- `BioDecoder.__init__` (method) `bicameral_v3.py:861` `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)`
- `BioDecoder.forward` (method) `bicameral_v3.py:883` `def forward(self, thought, visual_features, captions, max_len)`
- `CorpusCallosum.__init__` (method) `bicameral_v3.py:992` `def __init__(self)`
- `CorpusCallosum.forward` (method) `bicameral_v3.py:999` `def forward(self, left_repr, right_repr, mode)`
- `HomeostasisEngine.__init__` (method) `bicameral_v3.py:1040` `def __init__(self)`
- `HomeostasisEngine.decide` (method) `bicameral_v3.py:1049` `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- `BicameralHomeostasis.__init__` (method) `bicameral_v3.py:1083` `def __init__(self)`
- `BicameralHomeostasis.decide` (method) `bicameral_v3.py:1097` `def decide(self, left_metrics, right_metrics, epoch, total_epochs)`
- `ReplayMemory.__init__` (method) `bicameral_v3.py:1139` `def __init__(self, capacity, noise_scale)`
- `ReplayMemory.store` (method) `bicameral_v3.py:1145` `def store(self, pattern)`
- `ReplayMemory.replay` (method) `bicameral_v3.py:1155` `def replay(self, batch_size)`
- `NeuroLogos.__init__` (method) `bicameral_v3.py:1183` `def __init__(self, vocab_size, use_nested)`
- `NeuroLogos.forward` (method) `bicameral_v3.py:1203` `def forward(self, image, captions, plasticity, transfer_rate, mode, epoch, labels)`
- `NeuroLogos.set_epoch` (method) `bicameral_v3.py:1274` `def set_epoch(self, epoch)`
- `LifeCycle.__init__` (method) `bicameral_v3.py:1281` `def __init__(self, total_epochs)`
- `LifeCycle.get_plasticity` (method) `bicameral_v3.py:1285` `def get_plasticity(self, epoch)`
- `CIFARCaptions.__init__` (method) `bicameral_v3.py:1299` `def __init__(self)`
- `CIFARCaptions.estimate_coherence` (method) `bicameral_v3.py:1336` `def estimate_coherence(sentence, templates_per_class)`
- `CIFARCaptions.to_float` (method) `bicameral_v3.py:1349` `def to_float(val)`
- `CIFARCaptions.train_logos` (method) `bicameral_v3.py:1355` `def train_logos(use_nested)`

## caquita.py
- `DiagnosticConfig.seed_everything` (method) `caquita.py:40` `def seed_everything(seed)`
- `RealWorldEnvironment.__init__` (method) `caquita.py:51` `def __init__(self)`
- `RealWorldEnvironment.get_batch` (method) `caquita.py:63` `def get_batch(self, phase, batch_size)`
- `LiquidNeuron.__init__` (method) `caquita.py:79` `def __init__(self, in_dim, out_dim)`
- `LiquidNeuron.forward` (method) `caquita.py:88` `def forward(self, x, plasticity_gate)`
- `TraumaResponseSchedulerV2_ORIGINAL.__init__` (method) `caquita.py:105` `def __init__(self)`
- `TraumaResponseSchedulerV2_ORIGINAL.update_phase_performance` (method) `caquita.py:112` `def update_phase_performance(self, phase_idx, metrics)`
- `TraumaResponseSchedulerV2_ORIGINAL.detect_trauma_level` (method) `caquita.py:122` `def detect_trauma_level(self, phase_idx, current_metrics)`
- `TraumaResponseSchedulerV2_ORIGINAL.generate_response` (method) `caquita.py:148` `def generate_response(self, trauma_level, phase_idx, chaos_detected)`
- `TraumaResponseSchedulerV2_FIXED.__init__` (method) `caquita.py:170` `def __init__(self)`
- `TraumaResponseSchedulerV2_FIXED.update_phase_performance` (method) `caquita.py:176` `def update_phase_performance(self, phase_idx, metrics)`
- `TraumaResponseSchedulerV2_FIXED.detect_trauma_level` (method) `caquita.py:185` `def detect_trauma_level(self, phase_idx, current_metrics)`
- `TraumaResponseSchedulerV2_FIXED.generate_response` (method) `caquita.py:208` `def generate_response(self, trauma_level, phase_idx, chaos_detected)`
- `ChaosAdaptiveFilter_ORIGINAL.__init__` (method) `caquita.py:230` `def __init__(self)`
- `ChaosAdaptiveFilter_ORIGINAL.extract_noise_features` (method) `caquita.py:240` `def extract_noise_features(self, x)`
- `ChaosAdaptiveFilter_ORIGINAL.detect_chaos` (method) `caquita.py:253` `def detect_chaos(self, x)`
- `DiagnosticModel.__init__` (method) `caquita.py:265` `def __init__(self, config, use_liquid, use_trs_original, use_trs_fixed, use_caf)`
- `DiagnosticModel.forward` (method) `caquita.py:285` `def forward(self, x, phase_idx, current_metrics)`
- `DiagnosticModel.train_diagnostic` (method) `caquita.py:327` `def train_diagnostic(config, env, experiment_name)` -- Entrenamiento con logs detallados
- `DiagnosticModel.run_diagnostic_ablation` (method) `caquita.py:406` `def run_diagnostic_ablation()`

## chatgpt.py
- `Config.seed_all` (method) `chatgpt.py:33` `def seed_all(seed)`
- `DataEnvironment.__init__` (method) `chatgpt.py:42` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `chatgpt.py:53` `def get_batch(self, phase, bs)`
- `HomeostaticRegulator.__init__` (method) `chatgpt.py:71` `def __init__(self)`
- `HomeostaticRegulator.forward` (method) `chatgpt.py:81` `def forward(self, stress, excitation, fatigue, loss_signal)`
- `PhysioNeuron.__init__` (method) `chatgpt.py:94` `def __init__(self, d)`
- `PhysioNeuron.forward` (method) `chatgpt.py:107` `def forward(self, x, task_loss)`
- `NeuroPhysioBicameral.__init__` (method) `chatgpt.py:141` `def __init__(self, config)`
- `NeuroPhysioBicameral.count_parameters` (method) `chatgpt.py:163` `def count_parameters(self)`
- `NeuroPhysioBicameral.forward` (method) `chatgpt.py:166` `def forward(self, x, task_loss)`
- `NeuralDiagnostics.__init__` (method) `chatgpt.py:192` `def __init__(self)`
- `NeuralDiagnostics.update` (method) `chatgpt.py:201` `def update(self, loss, liquid_norm, phys)`
- `NeuralDiagnostics.avg` (method) `chatgpt.py:208` `def avg(self, k, n)`
- `NeuralDiagnostics.report` (method) `chatgpt.py:211` `def report(self, step, phase)`
- `NeuralDiagnostics.train` (method) `chatgpt.py:225` `def train()`

## cifar3.py
- `compute_phi_effective` (function) `cifar3.py:30` `def compute_phi_effective(activity)`
- `FastSlowLinear.__init__` (method) `cifar3.py:56` `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- `FastSlowLinear.reset_fast_weights` (method) `cifar3.py:74` `def reset_fast_weights(self)` -- Reinicia los pesos rápidos al inicio de cada batch.
- `FastSlowLinear.update_fast_weights` (method) `cifar3.py:79` `def update_fast_weights(self, x)` -- Actualiza fast weights usando regla hebbiana con decay y normalización.
- `FastSlowLinear.forward` (method) `cifar3.py:105` `def forward(self, x)`
- `FastSlowLinear.end_of_batch` (method) `cifar3.py:116` `def end_of_batch(self)` -- Limpia caché al final del batch para permitir reinicio en el siguiente.
- `FastSlowLinear.get_fast_norm` (method) `cifar3.py:120` `def get_fast_norm(self)` -- Retorna la norma L2 de los fast weights para monitoreo homeostático.
- `DualSystemModule.__init__` (method) `cifar3.py:130` `def __init__(self, dim)`
- `DualSystemModule.forward` (method) `cifar3.py:138` `def forward(self, x)`
- `ConsciousnessModule.__init__` (method) `cifar3.py:149` `def __init__(self, features)`
- `ConsciousnessModule.compute_phi_effective_robust` (method) `cifar3.py:160` `def compute_phi_effective_robust(self, activity)` -- Φₑ más robusto usando promedio móvil y ventana temporal.
- `ConsciousnessModule.forward` (method) `cifar3.py:184` `def forward(self, x)`
- `OmniBrainFastSlow.__init__` (method) `cifar3.py:194` `def __init__(self)`
- `OmniBrainFastSlow.forward` (method) `cifar3.py:226` `def forward(self, x)`
- `OmniBrainFastSlow.reset_all_fast_weights` (method) `cifar3.py:233` `def reset_all_fast_weights(self)`
- `OmniBrainFastSlow.get_fast_norms` (method) `cifar3.py:238` `def get_fast_norms(self)`
- `OmniBrainFastSlow.get_cifar10_loaders` (method) `cifar3.py:245` `def get_cifar10_loaders(batch_size)`
- `OmniBrainFastSlow.evaluate` (method) `cifar3.py:261` `def evaluate(model, loader, device)`
- `OmniBrainFastSlow.train` (method) `cifar3.py:274` `def train()`

## cifar4.py
- `compute_phi_effective` (function) `cifar4.py:30` `def compute_phi_effective(activity)`
- `FastSlowLinear.__init__` (method) `cifar4.py:56` `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- `FastSlowLinear.reset_fast_weights` (method) `cifar4.py:72` `def reset_fast_weights(self)`
- `FastSlowLinear.update_fast_weights` (method) `cifar4.py:76` `def update_fast_weights(self, x)`
- `FastSlowLinear.forward` (method) `cifar4.py:95` `def forward(self, x)`
- `FastSlowLinear.end_of_batch` (method) `cifar4.py:106` `def end_of_batch(self)`
- `FastSlowLinear.get_fast_norm` (method) `cifar4.py:109` `def get_fast_norm(self)`
- `DualSystemModule.__init__` (method) `cifar4.py:117` `def __init__(self, dim)`
- `DualSystemModule.forward` (method) `cifar4.py:125` `def forward(self, x)`
- `ConsciousnessModule.__init__` (method) `cifar4.py:136` `def __init__(self, features)`
- `ConsciousnessModule.compute_phi_effective_robust` (method) `cifar4.py:147` `def compute_phi_effective_robust(self, activity)`
- `ConsciousnessModule.forward` (method) `cifar4.py:166` `def forward(self, x)`
- `OmniBrainFastSlow.__init__` (method) `cifar4.py:177` `def __init__(self)`
- `OmniBrainFastSlow.forward` (method) `cifar4.py:197` `def forward(self, x)`
- `OmniBrainFastSlow.reset_all_fast_weights` (method) `cifar4.py:204` `def reset_all_fast_weights(self)`
- `OmniBrainFastSlow.get_fast_norms` (method) `cifar4.py:209` `def get_fast_norms(self)`
- `OmniBrainFastSlow.get_cifar10_loaders` (method) `cifar4.py:217` `def get_cifar10_loaders(batch_size)`
- `OmniBrainFastSlow.evaluate` (method) `cifar4.py:238` `def evaluate(model, loader, device)`
- `OmniBrainFastSlow.train` (method) `cifar4.py:255` `def train()`

## demo_auto_regulation.py
- `Config.seed_everything` (method) `demo_auto_regulation.py:33` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `demo_auto_regulation.py:44` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `demo_auto_regulation.py:54` `def get_batch(self, phase, bs)`
- `DataEnvironment.get_full` (method) `demo_auto_regulation.py:68` `def get_full(self)`
- `DataEnvironment.get_w2` (method) `demo_auto_regulation.py:71` `def get_w2(self)`
- `AutoRegulationSystem.__init__` (method) `demo_auto_regulation.py:78` `def __init__(self, size)`
- `AutoRegulationSystem.update` (method) `demo_auto_regulation.py:84` `def update(self, input_variance, loss_gradient, phase)`
- `AutoRegulationSystem.get_stability` (method) `demo_auto_regulation.py:99` `def get_stability(self)`
- `SelfModifyingGates.__init__` (method) `demo_auto_regulation.py:109` `def __init__(self, input_dim, hidden_dim)`
- `SelfModifyingGates.forward` (method) `demo_auto_regulation.py:126` `def forward(self, x, adaptation_state)`
- `PhysioChimeraFixed.__init__` (method) `demo_auto_regulation.py:160` `def __init__(self, config)`
- `PhysioChimeraFixed.forward` (method) `demo_auto_regulation.py:184` `def forward(self, x, global_step, phase, prev_loss)`
- `PhysioChimeraFixed.demo_auto_regulation` (method) `demo_auto_regulation.py:243` `def demo_auto_regulation()`

## difract.py
- `visualize_uased_geometry` (function) `difract.py:4` `def visualize_uased_geometry()`

## dmg_core.py
- `AdaptiveMagnitudeGate.__init__` (method) `dmg_core.py:21` `def __init__(self, base_threshold, power_order)`
- `AdaptiveMagnitudeGate.forward` (method) `dmg_core.py:30` `def forward(self, x)`
- `SparseTopologyLayer.__init__` (method) `dmg_core.py:49` `def __init__(self, in_features, out_features, sparsity_k)`
- `SparseTopologyLayer.forward` (method) `dmg_core.py:77` `def forward(self, x)`
- `DMGNetwork.__init__` (method) `dmg_core.py:87` `def __init__(self, input_dim, hidden_dim, output_dim)`
- `DMGNetwork.forward` (method) `dmg_core.py:100` `def forward(self, x)`

## dualmind.py
- `measure_spatial_richness` (function) `dualmind.py:15` `def measure_spatial_richness(activations)` -- Mide diversidad de representaciones mediante eigenspectro
- `HomeostasisEngine.__init__` (method) `dualmind.py:31` `def __init__(self)`
- `HomeostasisEngine.decide` (method) `dualmind.py:35` `def decide(self, task_loss_val, richness_val, vn_entropy_val)` -- Motor de decisión homeostática con targets realistas y pesos equilibrados. - target_entropy=1.8: Valor alcanzable...
- `LiquidNeuron.__init__` (method) `dualmind.py:59` `def __init__(self, in_dim, out_dim)`
- `LiquidNeuron.forward` (method) `dualmind.py:67` `def forward(self, x, plasticity_gate)` -- Neurona con plasticidad hebbiana de fast weights y decaimiento activación.
- `LiquidNeuron.consolidate_svd` (method) `dualmind.py:89` `def consolidate_svd(self, repair_strength)` -- Consolidación mediante SVD (modo sueño)
- `ConsciousSystem.__init__` (method) `dualmind.py:113` `def __init__(self, unconscious_dim, d_hid, d_out)`
- `ConsciousSystem.forward` (method) `dualmind.py:136` `def forward(self, unconscious_features, plasticity_gate)` -- Input: Representaciones del sistema inconsciente [batch, unconscious_dim] Output: logits, métricas homeostáticas
- `ConsciousSystem.get_structure_entropy` (method) `dualmind.py:157` `def get_structure_entropy(self)` -- Análisis de salud estructural mediante SVD
- `NestedTopoLayer.__init__` (method) `dualmind.py:176` `def __init__(self, in_dim, hid_dim, num_nodes)`
- `NestedTopoLayer.forward` (method) `dualmind.py:188` `def forward(self, x_nodes, plasticity_gate)` -- x_nodes: [batch, num_nodes, in_dim] output: [batch, num_nodes, hid_dim]
- `NestedTopoLayer.get_topology_density` (method) `dualmind.py:209` `def get_topology_density(self)` -- Densidad de conexiones topológicas
- `UnconsciousSystem.__init__` (method) `dualmind.py:222` `def __init__(self, in_channels, grid_size, hidden_dim)`
- `UnconsciousSystem.forward` (method) `dualmind.py:246` `def forward(self, x, plasticity_gate)` -- x: [batch, 3, 32, 32] output: [batch, output_dim] representaciones inconscientes
- `UnconsciousSystem.get_topology_stats` (method) `dualmind.py:264` `def get_topology_stats(self)` -- Estadísticas de topología del sistema inconsciente
- `DualMind.__init__` (method) `dualmind.py:285` `def __init__(self, in_channels, grid_size, hidden_dim, conscious_dim, num_classes)`
- `DualMind.forward` (method) `dualmind.py:305` `def forward(self, x, mode)` -- Modos de operación: - 'unconscious': Solo sistema inconsciente (rápido, baseline) - 'conscious': Consciente sobre...
- `DualMind.get_system_status` (method) `dualmind.py:331` `def get_system_status(self)` -- Diagnóstico completo del sistema dual
- `DualMind.train_dualmind_phase1` (method) `dualmind.py:346` `def train_dualmind_phase1(model, train_loader, optimizer, device, epochs)` -- FASE 1: Preentrenamiento del sistema inconsciente Objetivo: Aprender representaciones topológicas ricas
- `DualMind.train_dualmind_phase2` (method) `dualmind.py:401` `def train_dualmind_phase2(model, train_loader, optimizer, device, epochs)` -- FASE 2: Entrenamiento del sistema consciente Objetivo: Aprender decisiones homeostáticas óptimas Sistema...
- `DualMind.train_dualmind_phase3` (method) `dualmind.py:487` `def train_dualmind_phase3(model, train_loader, optimizer, device, epochs)` -- FASE 3: Co-adaptación de ambos sistemas Objetivo: Refinamiento conjunto con retroalimentación
- `DualMind.evaluate_dualmind` (method) `dualmind.py:579` `def evaluate_dualmind(model, test_loader, device)` -- Evaluación del sistema dual
- `DualMind.run_dualmind_experiment` (method) `dualmind.py:604` `def run_dualmind_experiment()`

## dynamic.py
- `seed_everything` (function) `dynamic.py:13` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `dynamic.py:37` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `dynamic.py:46` `def get_batch(self, phase, bs)`
- `OmnibusController.__init__` (method) `dynamic.py:55` `def __init__(self)`
- `OmnibusController.forward` (method) `dynamic.py:66` `def forward(self, x, h_slow)`
- `SovereignAttention.__init__` (method) `dynamic.py:89` `def __init__(self, d_in)`
- `SovereignAttention.forward` (method) `dynamic.py:94` `def forward(self, x, gain)`
- `LiquidNeuron.__init__` (method) `dynamic.py:101` `def __init__(self, d_in, d_out)`
- `LiquidNeuron.forward` (method) `dynamic.py:113` `def forward(self, x, plasticity, alpha)`
- `SovereignChimera.__init__` (method) `dynamic.py:138` `def __init__(self, config, dynamic_mode)`
- `SovereignChimera.forward` (method) `dynamic.py:151` `def forward(self, x)`
- `SovereignChimera.run_final_showdown` (method) `dynamic.py:184` `def run_final_showdown(epochs, name, dynamic)`

## dynamic2.py
- `seed_everything` (function) `dynamic2.py:13` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `dynamic2.py:37` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `dynamic2.py:46` `def get_batch(self, phase, bs)`
- `HomeostaticRegulator.__init__` (method) `dynamic2.py:55` `def __init__(self, d_in)`
- `HomeostaticRegulator.forward` (method) `dynamic2.py:66` `def forward(self, x, h_pre, w_norm)`
- `PhysioNeuron.__init__` (method) `dynamic2.py:90` `def __init__(self, d_in, d_out, dynamic_mode)`
- `PhysioNeuron.forward` (method) `dynamic2.py:105` `def forward(self, x)`
- `PhysioChimera.__init__` (method) `dynamic2.py:163` `def __init__(self, config, dynamic_mode)`
- `PhysioChimera.forward` (method) `dynamic2.py:169` `def forward(self, x)`
- `PhysioChimera.run_physio_experiment` (method) `dynamic2.py:178` `def run_physio_experiment(epochs, name, dynamic)`

## example_usage.py
Depends on: `physio_chimera_v15_monitored.py`
- `demo_simple_monitoring` (function) `example_usage.py:20` `def demo_simple_monitoring()` -- Demostración de monitoreo básico
- `demo_custom_monitoring` (function) `example_usage.py:38` `def demo_custom_monitoring()` -- Demostración de monitoreo personalizado
- `demo_checkpoint_system` (function) `example_usage.py:85` `def demo_checkpoint_system()` -- Demostración del sistema de checkpointing
- `demo_comparison_experiments` (function) `example_usage.py:142` `def demo_comparison_experiments()` -- Demostración de comparación entre experimentos
- `create_demo_report` (function) `example_usage.py:194` `def create_demo_report()` -- Crear reporte demo completo
- `main` (function) `example_usage.py:333` `def main()` -- Función principal de demostración

## exampleww.py
- `run_single_experiment` (function) `exampleww.py:8` `def run_single_experiment(model_name, seed, epochs)`

## exodia_op_2.py
- `preprocess_and_cache_spectrograms` (function) `exodia_op_2.py:49` `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)` -- Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt Esto elimina el cuello de...
- `apply_emergency_fixes` (function) `exodia_op_2.py:120` `def apply_emergency_fixes(model)`
- `setup_flickr8k_with_audio` (function) `exodia_op_2.py:142` `def setup_flickr8k_with_audio(data_dir)` -- Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
- `build_vocab_flickr` (function) `exodia_op_2.py:316` `def build_vocab_flickr(captions_file, vocab_size)` -- Construye vocabulario desde el archivo de captions
- `HierarchicalEpisodicMemory.__init__` (method) `exodia_op_2.py:347` `def __init__(self, working_capacity, short_term_capacity, importance_threshold)`
- `HierarchicalEpisodicMemory.compute_surprise` (method) `exodia_op_2.py:371` `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` -- FIX: Clamp de cross-entropy para evitar infinitos
- `HierarchicalEpisodicMemory.calculate_importance` (method) `exodia_op_2.py:386` `def calculate_importance(self, episode, surprise_score)` -- FIX: Clamp de surprise_score para evitar probabilidades degeneradas
- `HierarchicalEpisodicMemory.store_episode` (method) `exodia_op_2.py:427` `def store_episode(self, image, audio, caption, surprise_score)`
- `HierarchicalEpisodicMemory.sample` (method) `exodia_op_2.py:470` `def sample(self, batch_size, memory_level)` -- FIX: Manejo de edge cases en sampling probabilístico
- `HierarchicalEpisodicMemory.apply_forgetting_curve` (method) `exodia_op_2.py:535` `def apply_forgetting_curve(self)`
- `NeurocognitiveSystem.__init__` (method) `exodia_op_2.py:578` `def __init__(self)`
- `NeurocognitiveSystem.assess_reasoning_state` (method) `exodia_op_2.py:598` `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` -- Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)
- `NeurocognitiveSystem.assess_cognitive_state` (method) `exodia_op_2.py:642` `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` -- Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)
- `NeurocognitiveSystem.apply_cognitive_intervention` (method) `exodia_op_2.py:688` `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` -- Aplica intervenciones basadas en estado lingüístico y de razonamiento
- `LanguageMetrics.sentence_bleu` (method) `exodia_op_2.py:778` `def sentence_bleu(reference, hypothesis, weights)` -- BLEU simplificado a nivel de oración
- `LanguageMetrics.token_accuracy` (method) `exodia_op_2.py:821` `def token_accuracy(reference, hypothesis)` -- Porcentaje de tokens correctos en posición
- `LanguageMetrics.word_overlap` (method) `exodia_op_2.py:834` `def word_overlap(reference, hypothesis)` -- Jaccard similarity entre palabras
- `LinguisticFeedbackLoop.__init__` (method) `exodia_op_2.py:849` `def __init__(self, alpha, beta)`
- `LinguisticFeedbackLoop.compute_linguistic_reward` (method) `exodia_op_2.py:872` `def compute_linguistic_reward(self, references, hypotheses)`
- `LinguisticFeedbackLoop.compute_cider` (method) `exodia_op_2.py:911` `def compute_cider(self, reference, hypothesis)` -- FIX: Uso correcto del cache estático
- `LinguisticFeedbackLoop.compute_spice` (method) `exodia_op_2.py:925` `def compute_spice(self, reference, hypothesis)`
- `LinguisticFeedbackLoop.get_cache_stats` (method) `exodia_op_2.py:937` `def get_cache_stats(self)` -- FIX: Estadísticas de cache actualizadas
- `LanguageMetrics.sentence_bleu` (method) `exodia_op_2.py:967` `def sentence_bleu(reference, hypothesis, weights)`
- `LanguageMetrics.token_accuracy` (method) `exodia_op_2.py:990` `def token_accuracy(reference, hypothesis)`
- `LanguageMetrics.word_overlap` (method) `exodia_op_2.py:1000` `def word_overlap(reference, hypothesis)`
- `CausalReasoningEngine.__init__` (method) `exodia_op_2.py:1009` `def __init__(self, hidden_dim)`
- `CausalReasoningEngine.reason_causally` (method) `exodia_op_2.py:1036` `def reason_causally(self, observation, context)`
- `CausalReasoningEngine.update_knowledge_graph` (method) `exodia_op_2.py:1067` `def update_knowledge_graph(self, cause, effect, strength)`
- `CausalReasoningEngine.query_causal_chain` (method) `exodia_op_2.py:1073` `def query_causal_chain(self, start_node, end_node)`
- `LanguageMetrics.sentence_bleu` (method) `exodia_op_2.py:1089` `def sentence_bleu(reference, hypothesis, weights)`
- `LanguageMetrics.token_accuracy` (method) `exodia_op_2.py:1112` `def token_accuracy(reference, hypothesis)`
- `LanguageMetrics.word_overlap` (method) `exodia_op_2.py:1122` `def word_overlap(reference, hypothesis)`
- `StableLiquidNeuron.__init__` (method) `exodia_op_2.py:1136` `def __init__(self, in_dim, out_dim)`
- `StableLiquidNeuron.forward` (method) `exodia_op_2.py:1184` `def forward(self, x)`
- `StableLiquidNeuron.hebbian_update` (method) `exodia_op_2.py:1229` `def hebbian_update(self, post, pre, plasticity)`
- `StableLiquidNeuron.update_physiology_advanced` (method) `exodia_op_2.py:1278` `def update_physiology_advanced(self, loss_value)`
- `TricameralOutput.forward` (method) `exodia_op_2.py:1335` `def forward(self, image, audio, captions, epoch)`
- `TriangulatedMedicalSystem.__init__` (method) `exodia_op_2.py:1361` `def __init__(self)`
- `TriangulatedMedicalSystem.triangulate_signals` (method) `exodia_op_2.py:1368` `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `TriangulatedMedicalSystem.count_convergent_signals` (method) `exodia_op_2.py:1379` `def count_convergent_signals(self, signals, pattern)`
- `TriangulatedMedicalSystem.diagnose_with_triangulation` (method) `exodia_op_2.py:1382` `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)`
- `TriangulatedMedicalSystem.apply_triangulated_intervention` (method) `exodia_op_2.py:1427` `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- `LeftHemisphere.__init__` (method) `exodia_op_2.py:1511` `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `LeftHemisphere.forward` (method) `exodia_op_2.py:1596` `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `AudioEncoder.__init__` (method) `exodia_op_2.py:1842` `def __init__(self, output_dim)`
- `AudioEncoder.forward` (method) `exodia_op_2.py:1880` `def forward(self, mel_spec)`
- `RightHemisphereTricameral.__init__` (method) `exodia_op_2.py:1909` `def __init__(self, output_dim)`
- `RightHemisphereTricameral.forward` (method) `exodia_op_2.py:1947` `def forward(self, image, audio)`
- `CorpusCallosumTrimodal.__init__` (method) `exodia_op_2.py:1994` `def __init__(self, dim)`
- `CorpusCallosumTrimodal.forward` (method) `exodia_op_2.py:2090` `def forward(self, right_features)` -- FIX: Manejo robusto de dimensiones y verificación de coherencia trimodal
- `CorpusCallosumTrimodal.update_channel_fatigue` (method) `exodia_op_2.py:2169` `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- `CorpusCallosumTrimodal.adjust_gates_by_fatigue` (method) `exodia_op_2.py:2190` `def adjust_gates_by_fatigue(self)` -- Lógica original de ajuste de gates
- `EnhancedDiagnosticsTricameral.__init__` (method) `exodia_op_2.py:2208` `def __init__(self)`
- `EnhancedDiagnosticsTricameral.measure_callosal_flow` (method) `exodia_op_2.py:2248` `def measure_callosal_flow(self, right_features, left_context, channels)` -- Medición de coherencia multimodal con sincronización entre canales
- `EnhancedDiagnosticsTricameral.evaluate_reasoning_quality` (method) `exodia_op_2.py:2305` `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- `EnhancedDiagnosticsTricameral.calculate_synergy` (method) `exodia_op_2.py:2342` `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- `EnhancedDiagnosticsTricameral.calculate_health` (method) `exodia_op_2.py:2353` `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `EnhancedDiagnosticsTricameral.update` (method) `exodia_op_2.py:2362` `def update(self)`
- `EnhancedDiagnosticsTricameral.get_recent_avg` (method) `exodia_op_2.py:2379` `def get_recent_avg(self, key, n)`
- `EnhancedDiagnosticsTricameral.visualize_fatigue_distribution` (method) `exodia_op_2.py:2395` `def visualize_fatigue_distribution(self, epoch)`
- `EnhancedDiagnosticsTricameral.visualize_reasoning_metrics` (method) `exodia_op_2.py:2417` `def visualize_reasoning_metrics(self, epoch)`
- `EnhancedDiagnosticsTricameral.report` (method) `exodia_op_2.py:2429` `def report(self, epoch)`
- `NeuroLogosTricameral.__init__` (method) `exodia_op_2.py:2518` `def __init__(self, vocab_size)`
- `NeuroLogosTricameral.forward` (method) `exodia_op_2.py:2525` `def forward(self, image, audio, captions, epoch)`
- `Flickr8kMultimodalDataset.__init__` (method) `exodia_op_2.py:2553` `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache...`
- `Flickr8kMultimodalDataset.compute_alignment_loss` (method) `exodia_op_2.py:2666` `def compute_alignment_loss(visual_features, channels, alpha, epoch)` -- FIX: Pérdida auxiliar para alineación temprana de canales multimodales Solo activa en épocas iniciales (epoch < 6)
- `Flickr8kMultimodalDataset.compute_tricameral_loss` (method) `exodia_op_2.py:2695` `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward...`
- `Flickr8kMultimodalDataset.train_tricameral` (method) `exodia_op_2.py:2812` `def train_tricameral()`

## exodia_optimized.py
- `preprocess_and_cache_spectrograms` (function) `exodia_optimized.py:47` `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)` -- Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt Esto elimina el cuello de...
- `setup_flickr8k_with_audio` (function) `exodia_optimized.py:120` `def setup_flickr8k_with_audio(data_dir)` -- Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
- `build_vocab_flickr` (function) `exodia_optimized.py:308` `def build_vocab_flickr(captions_file, vocab_size)` -- Construye vocabulario desde el archivo de captions
- `HierarchicalEpisodicMemory.__init__` (method) `exodia_optimized.py:341` `def __init__(self, working_capacity, short_term_capacity, importance_threshold)`
- `HierarchicalEpisodicMemory.compute_surprise` (method) `exodia_optimized.py:369` `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` -- Sin cambios en la lógica
- `HierarchicalEpisodicMemory.calculate_importance` (method) `exodia_optimized.py:380` `def calculate_importance(self, episode, surprise_score)` -- Sin cambios
- `HierarchicalEpisodicMemory.store_episode` (method) `exodia_optimized.py:416` `def store_episode(self, image, audio, caption, surprise_score)`
- `HierarchicalEpisodicMemory.add` (method) `exodia_optimized.py:450` `def add(self, image, audio, caption, surprise_score)` -- Alias para store_episode
- `HierarchicalEpisodicMemory.apply_forgetting_curve` (method) `exodia_optimized.py:454` `def apply_forgetting_curve(self)` -- Decay más agresivo (ahorro overhead)
- `HierarchicalEpisodicMemory.sample` (method) `exodia_optimized.py:487` `def sample(self, batch_size, memory_level)` -- Muestreo solo de working/short_term
- `HierarchicalEpisodicMemory.get_total_size` (method) `exodia_optimized.py:539` `def get_total_size(self)` -- Solo working + short_term
- `NeurocognitiveSystem.__init__` (method) `exodia_optimized.py:547` `def __init__(self)`
- `NeurocognitiveSystem.assess_reasoning_state` (method) `exodia_optimized.py:567` `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` -- Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)
- `NeurocognitiveSystem.assess_cognitive_state` (method) `exodia_optimized.py:611` `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` -- Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)
- `NeurocognitiveSystem.apply_cognitive_intervention` (method) `exodia_optimized.py:657` `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` -- Aplica intervenciones basadas en estado lingüístico y de razonamiento
- `LanguageMetrics.sentence_bleu` (method) `exodia_optimized.py:747` `def sentence_bleu(reference, hypothesis, weights)` -- BLEU simplificado a nivel de oración
- `LanguageMetrics.token_accuracy` (method) `exodia_optimized.py:790` `def token_accuracy(reference, hypothesis)` -- Porcentaje de tokens correctos en posición
- `LanguageMetrics.word_overlap` (method) `exodia_optimized.py:803` `def word_overlap(reference, hypothesis)` -- Jaccard similarity entre palabras
- `LinguisticFeedbackLoop.__init__` (method) `exodia_optimized.py:818` `def __init__(self, alpha, beta)`
- `LinguisticFeedbackLoop.compute_linguistic_reward` (method) `exodia_optimized.py:841` `def compute_linguistic_reward(self, references, hypotheses)`
- `LinguisticFeedbackLoop.compute_cider` (method) `exodia_optimized.py:880` `def compute_cider(self, reference, hypothesis)` -- FIX: Uso correcto del cache estático
- `LinguisticFeedbackLoop.compute_spice` (method) `exodia_optimized.py:894` `def compute_spice(self, reference, hypothesis)`
- `LinguisticFeedbackLoop.get_cache_stats` (method) `exodia_optimized.py:906` `def get_cache_stats(self)` -- FIX: Estadísticas de cache actualizadas
- `LanguageMetrics.sentence_bleu` (method) `exodia_optimized.py:936` `def sentence_bleu(reference, hypothesis, weights)`
- `LanguageMetrics.token_accuracy` (method) `exodia_optimized.py:959` `def token_accuracy(reference, hypothesis)`
- `LanguageMetrics.word_overlap` (method) `exodia_optimized.py:969` `def word_overlap(reference, hypothesis)`
- `CausalReasoningEngine.__init__` (method) `exodia_optimized.py:978` `def __init__(self, hidden_dim)`
- `CausalReasoningEngine.reason_causally` (method) `exodia_optimized.py:1005` `def reason_causally(self, observation, context)`
- `CausalReasoningEngine.update_knowledge_graph` (method) `exodia_optimized.py:1036` `def update_knowledge_graph(self, cause, effect, strength)`
- `CausalReasoningEngine.query_causal_chain` (method) `exodia_optimized.py:1042` `def query_causal_chain(self, start_node, end_node)`
- `LanguageMetrics.sentence_bleu` (method) `exodia_optimized.py:1058` `def sentence_bleu(reference, hypothesis, weights)`
- `LanguageMetrics.token_accuracy` (method) `exodia_optimized.py:1081` `def token_accuracy(reference, hypothesis)`
- `LanguageMetrics.word_overlap` (method) `exodia_optimized.py:1091` `def word_overlap(reference, hypothesis)`
- `StableLiquidNeuron.__init__` (method) `exodia_optimized.py:1104` `def __init__(self, in_dim, out_dim)`
- `StableLiquidNeuron.forward` (method) `exodia_optimized.py:1146` `def forward(self, x)`
- `StableLiquidNeuron.hebbian_update` (method) `exodia_optimized.py:1172` `def hebbian_update(self, post, pre, plasticity)`
- `StableLiquidNeuron.update_physiology_advanced` (method) `exodia_optimized.py:1210` `def update_physiology_advanced(self, loss_value)`
- `TriangulatedMedicalSystem.__init__` (method) `exodia_optimized.py:1244` `def __init__(self)`
- `TriangulatedMedicalSystem.triangulate_signals` (method) `exodia_optimized.py:1251` `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `TriangulatedMedicalSystem.count_convergent_signals` (method) `exodia_optimized.py:1262` `def count_convergent_signals(self, signals, pattern)`
- `TriangulatedMedicalSystem.diagnose_with_triangulation` (method) `exodia_optimized.py:1265` `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)`
- `TriangulatedMedicalSystem.apply_triangulated_intervention` (method) `exodia_optimized.py:1310` `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- `LeftHemisphere.__init__` (method) `exodia_optimized.py:1395` `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `LeftHemisphere.forward` (method) `exodia_optimized.py:1477` `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `AudioEncoder.__init__` (method) `exodia_optimized.py:1711` `def __init__(self, output_dim)`
- `AudioEncoder.forward` (method) `exodia_optimized.py:1751` `def forward(self, mel_spec)`
- `RightHemisphereTricameral.__init__` (method) `exodia_optimized.py:1786` `def __init__(self, output_dim)`
- `RightHemisphereTricameral.forward` (method) `exodia_optimized.py:1824` `def forward(self, image, audio)`
- `CorpusCallosumTrimodal.__init__` (method) `exodia_optimized.py:1871` `def __init__(self, dim)`
- `CorpusCallosumTrimodal.forward` (method) `exodia_optimized.py:1966` `def forward(self, right_features)`
- `CorpusCallosumTrimodal.update_channel_fatigue` (method) `exodia_optimized.py:2054` `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` -- Lógica original de fatiga sin cambios
- `CorpusCallosumTrimodal.adjust_gates_by_fatigue` (method) `exodia_optimized.py:2076` `def adjust_gates_by_fatigue(self)` -- Lógica original de ajuste de gates
- `EnhancedDiagnosticsTricameral.__init__` (method) `exodia_optimized.py:2094` `def __init__(self)`
- `EnhancedDiagnosticsTricameral.measure_callosal_flow` (method) `exodia_optimized.py:2135` `def measure_callosal_flow(self, right_features, left_context, channels)` -- FIX: Medición de coherencia multimodal real con atención a diversidad Incluye métricas de sincronización entre canales
- `EnhancedDiagnosticsTricameral.evaluate_reasoning_quality` (method) `exodia_optimized.py:2188` `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- `EnhancedDiagnosticsTricameral.calculate_synergy` (method) `exodia_optimized.py:2225` `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- `EnhancedDiagnosticsTricameral.calculate_health` (method) `exodia_optimized.py:2236` `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `EnhancedDiagnosticsTricameral.update` (method) `exodia_optimized.py:2245` `def update(self)`
- `EnhancedDiagnosticsTricameral.get_recent_avg` (method) `exodia_optimized.py:2262` `def get_recent_avg(self, key, n)`
- `EnhancedDiagnosticsTricameral.visualize_fatigue_distribution` (method) `exodia_optimized.py:2278` `def visualize_fatigue_distribution(self, epoch)`
- `EnhancedDiagnosticsTricameral.visualize_reasoning_metrics` (method) `exodia_optimized.py:2302` `def visualize_reasoning_metrics(self, epoch)`
- `EnhancedDiagnosticsTricameral.report` (method) `exodia_optimized.py:2314` `def report(self, epoch)`
- `NeuroLogosTricameral.__init__` (method) `exodia_optimized.py:2401` `def __init__(self, vocab_size)`
- `NeuroLogosTricameral.forward` (method) `exodia_optimized.py:2407` `def forward(self, image, audio, captions, epoch)`
- `Flickr8kMultimodalDataset.__init__` (method) `exodia_optimized.py:2436` `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache...`
- `Flickr8kMultimodalDataset.compute_alignment_loss` (method) `exodia_optimized.py:2549` `def compute_alignment_loss(visual_features, channels, alpha, epoch)` -- FIX: Pérdida auxiliar para alineación temprana de canales multimodales Solo activa en épocas iniciales (epoch < 6)
- `Flickr8kMultimodalDataset.compute_tricameral_loss` (method) `exodia_optimized.py:2577` `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward...` -- FIX: Pérdida con término explícito de coherencia multimodal Penaliza la falta de sincronización entre canales
- `Flickr8kMultimodalDataset.train_tricameral` (method) `exodia_optimized.py:2650` `def train_tricameral()`

## final_sinergy_analysis.py
- `SinergyAnalysis.__init__` (method) `final_sinergy_analysis.py:13` `def __init__(self)`
- `SinergyAnalysis.print_header` (method) `final_sinergy_analysis.py:80` `def print_header(self)`
- `SinergyAnalysis.analyze_original_models` (method) `final_sinergy_analysis.py:87` `def analyze_original_models(self)`
- `SinergyAnalysis.analyze_sinergies` (method) `final_sinergy_analysis.py:99` `def analyze_sinergies(self)`
- `SinergyAnalysis.generate_scientific_matrix` (method) `final_sinergy_analysis.py:118` `def generate_scientific_matrix(self)`
- `SinergyAnalysis.calculate_synergy_breakthrough` (method) `final_sinergy_analysis.py:136` `def calculate_synergy_breakthrough(self)`
- `SinergyAnalysis.generate_conclusion` (method) `final_sinergy_analysis.py:171` `def generate_conclusion(self)`
- `SinergyAnalysis.save_results` (method) `final_sinergy_analysis.py:200` `def save_results(self)`
- `SinergyAnalysis.main` (method) `final_sinergy_analysis.py:219` `def main()`

## gemini.py
- `setup_flickr8k` (function) `gemini.py:40` `def setup_flickr8k(data_dir)` -- Descarga Flickr8k automáticamente
- `LiquidNeuron.__init__` (method) `gemini.py:109` `def __init__(self, in_dim, out_dim)`
- `LiquidNeuron.forward` (method) `gemini.py:123` `def forward(self, x, global_plasticity, transfer_rate)`
- `LiquidNeuron.consolidate_svd` (method) `gemini.py:154` `def consolidate_svd(self, repair_strength, timescale)`
- `RightHemisphere.__init__` (method) `gemini.py:178` `def __init__(self, output_dim)`
- `RightHemisphere.forward` (method) `gemini.py:187` `def forward(self, image, plasticity, transfer_rate)`
- `CorpusCallosum.__init__` (method) `gemini.py:197` `def __init__(self, dim)`
- `CorpusCallosum.forward` (method) `gemini.py:210` `def forward(self, right_features)`
- `LeftHemisphere.__init__` (method) `gemini.py:220` `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `LeftHemisphere.forward` (method) `gemini.py:241` `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)`
- `NeuroLogosBicameral.__init__` (method) `gemini.py:334` `def __init__(self, vocab_size)`
- `NeuroLogosBicameral.forward` (method) `gemini.py:340` `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- `NeuralDiagnostics.__init__` (method) `gemini.py:362` `def __init__(self)`
- `NeuralDiagnostics.measure_callosal_flow` (method) `gemini.py:373` `def measure_callosal_flow(self, right_features, left_context)`
- `NeuralDiagnostics.measure_vocab_diversity` (method) `gemini.py:380` `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `NeuralDiagnostics.update` (method) `gemini.py:384` `def update(self)`
- `NeuralDiagnostics.get_recent_avg` (method) `gemini.py:389` `def get_recent_avg(self, key, n)`
- `NeuralDiagnostics.report` (method) `gemini.py:394` `def report(self, epoch)`
- `Flickr8kDataset.__init__` (method) `gemini.py:432` `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `Flickr8kDataset.build_vocab_flickr` (method) `gemini.py:470` `def build_vocab_flickr(captions_file, vocab_size)`
- `LifeCycle.__init__` (method) `gemini.py:494` `def __init__(self, total_epochs)`
- `LifeCycle.get_plasticity` (method) `gemini.py:497` `def get_plasticity(self, epoch)`
- `LifeCycle.train_bicameral` (method) `gemini.py:509` `def train_bicameral()`

## gemini2.py
- `setup_flickr8k` (function) `gemini2.py:40` `def setup_flickr8k(data_dir)` -- Descarga Flickr8k automáticamente
- `LiquidNeuron.__init__` (method) `gemini2.py:109` `def __init__(self, in_dim, out_dim)`
- `LiquidNeuron.forward` (method) `gemini2.py:123` `def forward(self, x, global_plasticity, transfer_rate)`
- `LiquidNeuron.consolidate_svd` (method) `gemini2.py:154` `def consolidate_svd(self, repair_strength, timescale)`
- `RightHemisphere.__init__` (method) `gemini2.py:178` `def __init__(self, output_dim)`
- `RightHemisphere.forward` (method) `gemini2.py:187` `def forward(self, image, plasticity, transfer_rate)`
- `CorpusCallosum.__init__` (method) `gemini2.py:197` `def __init__(self, dim)`
- `CorpusCallosum.forward` (method) `gemini2.py:210` `def forward(self, right_features)`
- `LeftHemisphere.__init__` (method) `gemini2.py:220` `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `LeftHemisphere.forward` (method) `gemini2.py:241` `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)`
- `NeuroLogosBicameral.__init__` (method) `gemini2.py:334` `def __init__(self, vocab_size)`
- `NeuroLogosBicameral.forward` (method) `gemini2.py:340` `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- `NeuralDiagnostics.__init__` (method) `gemini2.py:362` `def __init__(self)`
- `NeuralDiagnostics.measure_callosal_flow` (method) `gemini2.py:373` `def measure_callosal_flow(self, right_features, left_context)`
- `NeuralDiagnostics.measure_vocab_diversity` (method) `gemini2.py:380` `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `NeuralDiagnostics.update` (method) `gemini2.py:384` `def update(self)`
- `NeuralDiagnostics.get_recent_avg` (method) `gemini2.py:389` `def get_recent_avg(self, key, n)`
- `NeuralDiagnostics.report` (method) `gemini2.py:394` `def report(self, epoch)`
- `Flickr8kDataset.__init__` (method) `gemini2.py:432` `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `Flickr8kDataset.build_vocab_flickr` (method) `gemini2.py:470` `def build_vocab_flickr(captions_file, vocab_size)`
- `LifeCycle.__init__` (method) `gemini2.py:494` `def __init__(self, total_epochs)`
- `LifeCycle.get_plasticity` (method) `gemini2.py:497` `def get_plasticity(self, epoch)`
- `LifeCycle.train_bicameral` (method) `gemini2.py:509` `def train_bicameral()`

## gen_dataset.py
- `generate_one` (function) `gen_dataset.py:89` `def generate_one(key, text)`
- `main` (function) `gen_dataset.py:110` `def main()`

## get_dataset.py
- `download_captions_only` (function) `get_dataset.py:29` `def download_captions_only()` -- Descarga solo los captions de Flickr8k
- `generate_one_audio` (function) `get_dataset.py:67` `def generate_one_audio(text, output_path, max_retries)` -- Genera un audio con retry y rate limiting
- `load_checkpoint` (function) `get_dataset.py:114` `def load_checkpoint()` -- Carga el checkpoint de progreso
- `save_checkpoint` (function) `get_dataset.py:122` `def save_checkpoint(checkpoint)` -- Guarda el checkpoint de progreso
- `generate_audios_with_checkpoints` (function) `get_dataset.py:128` `def generate_audios_with_checkpoints()` -- Genera audios con checkpoints cada 500 archivos
- `generate_audios_sync` (function) `get_dataset.py:219` `def generate_audios_sync()` -- Wrapper síncrono con manejo de event loop
- `compress_audios_only` (function) `get_dataset.py:249` `def compress_audios_only()` -- Comprime solo los audios en zips pequeños
- `create_audio_readme` (function) `get_dataset.py:318` `def create_audio_readme(output_dir, metadata)` -- Crea README para el dataset de audios
- `upload_to_huggingface` (function) `get_dataset.py:388` `def upload_to_huggingface(dataset_dir)` -- Sube solo audios a Hugging Face
- `main` (function) `get_dataset.py:465` `def main()`
- `download_flickr8k` (function) `get_dataset.py:540` `def download_flickr8k()` -- Descarga Flickr8k (solo necesitas ejecutar esto una vez)
- `generate_audios` (function) `get_dataset.py:602` `def generate_audios()` -- Genera audios con Edge-TTS - TOMA TIEMPO (~20-30 min)
- `create_split_zips` (function) `get_dataset.py:637` `def create_split_zips()` -- Crea múltiples zips pequeños para cumplir límites de GitHub
- `generate_upload_instructions` (function) `get_dataset.py:717` `def generate_upload_instructions(metadata)` -- Genera instrucciones para subir a GitHub
- `upload_to_huggingface` (function) `get_dataset.py:834` `def upload_to_huggingface(dataset_dir)` -- Sube directamente a Hugging Face (alternativa a GitHub)
- `main` (function) `get_dataset.py:888` `def main()`

## homeostatichope.py
- `Config.setup_device` (method) `homeostatichope.py:35` `def setup_device()`
- `Config.set_seed` (method) `homeostatichope.py:40` `def set_seed(seed)`
- `RealWorldEnvironment.__init__` (method) `homeostatichope.py:50` `def __init__(self, seed)`
- `RealWorldEnvironment.get_batch` (method) `homeostatichope.py:73` `def get_batch(self, phase, batch_size)`
- `RealWorldEnvironment.get_test_loader` (method) `homeostatichope.py:88` `def get_test_loader(self, batch_size)`
- `OmniscientRegulator.__init__` (method) `homeostatichope.py:102` `def __init__(self, d_model)`
- `OmniscientRegulator.forward` (method) `homeostatichope.py:128` `def forward(self, signals)` -- Args: signals: Diccionario con señales internas del sistema Returns: controls: Diccionario con hiperparámetros ajustados
- `AdaptiveLiquidMemory.__init__` (method) `homeostatichope.py:193` `def __init__(self, d_model)`
- `AdaptiveLiquidMemory.forward` (method) `homeostatichope.py:203` `def forward(self, x, controls)`
- `HomeostaticSelfModMemory.__init__` (method) `homeostatichope.py:234` `def __init__(self, d_model, hidden_dim)`
- `HomeostaticSelfModMemory.forward` (method) `homeostatichope.py:251` `def forward(self, x, controls)`
- `ContinuumMemorySystem.__init__` (method) `homeostatichope.py:279` `def __init__(self, frequencies, d_model, hidden_dim)`
- `ContinuumMemorySystem.forward` (method) `homeostatichope.py:293` `def forward(self, x, global_step)`
- `OmniscientHopeModel.__init__` (method) `homeostatichope.py:305` `def __init__(self, config, n_features, n_classes)`
- `OmniscientHopeModel.forward` (method) `homeostatichope.py:341` `def forward(self, x, signals, global_step)`
- `OmniscientHopeModel.pgd_attack` (method) `homeostatichope.py:368` `def pgd_attack(model, x, y, epsilon, steps, device, signals)`
- `ConsciousTrainer.__init__` (method) `homeostatichope.py:403` `def __init__(self, model, config, device)`
- `ConsciousTrainer.train_step` (method) `homeostatichope.py:425` `def train_step(self, x, y, epsilon, global_step, phase)`
- `ConsciousTrainer.evaluate` (method) `homeostatichope.py:488` `def evaluate(self, test_loader, epsilon, phase)`
- `ConsciousTrainer.run_conscious_experiment` (method) `homeostatichope.py:516` `def run_conscious_experiment(config, device)`
- `ConsciousTrainer.run_ablation` (method) `homeostatichope.py:610` `def run_ablation(device)`


Next: [API_p4.md](API_p4.md)

# API (page 8 of 10)
Previous: [API_p7.md](API_p7.md)

## quimera_vision.py
- `text_to_seq` (function) `quimera_vision.py:37` `def text_to_seq(text)`
- `Flickr8kMMDataset.__init__` (method) `quimera_vision.py:44` `def __init__(self)`
- `Flickr8kMMDataset.collate` (method) `quimera_vision.py:87` `def collate(batch)`
- `ImgEncoder.__init__` (method) `quimera_vision.py:97` `def __init__(self)`
- `ImgEncoder.forward` (method) `quimera_vision.py:103` `def forward(self, x)`
- `AudioEncoder.__init__` (method) `quimera_vision.py:109` `def __init__(self)`
- `AudioEncoder.forward` (method) `quimera_vision.py:116` `def forward(self, x)`
- `Decoder.__init__` (method) `quimera_vision.py:122` `def __init__(self)`
- `Decoder.forward` (method) `quimera_vision.py:127` `def forward(self, img, audio, seq)`
- `Decoder.generate_caption` (method) `quimera_vision.py:168` `def generate_caption(img_path, audio_path)`

## qwen.py
- `MicroConfig.seed_everything` (method) `qwen.py:57` `def seed_everything(seed)`
- `MicroConfig.get_dataset` (method) `qwen.py:64` `def get_dataset(config)`
- `HomeostaticOrchestrator.__init__` (method) `qwen.py:89` `def __init__(self)`
- `HomeostaticOrchestrator.forward` (method) `qwen.py:99` `def forward(self, x, logits, h_agg, h_proc, w_norm, entropy, ortho, pgd_loss)`
- `RegulableContinuum.__init__` (method) `qwen.py:122` `def __init__(self, dim)`
- `RegulableContinuum.forward` (method) `qwen.py:132` `def forward(self, x, strength)`
- `RegulableSymbiotic.__init__` (method) `qwen.py:144` `def __init__(self, dim, num_atoms)`
- `RegulableSymbiotic.forward` (method) `qwen.py:151` `def forward(self, x, influence)`
- `RegulableTopology.__init__` (method) `qwen.py:163` `def __init__(self, num_nodes)`
- `RegulableTopology.get_adjacency` (method) `qwen.py:176` `def get_adjacency(self, plasticity)`
- `RegulableSupConHead.__init__` (method) `qwen.py:182` `def __init__(self, in_dim)`
- `RegulableSupConHead.forward` (method) `qwen.py:190` `def forward(self, x, gain)`
- `MicroTopoBrain.__init__` (method) `qwen.py:197` `def __init__(self, config)`
- `MicroTopoBrain.count_parameters` (method) `qwen.py:220` `def count_parameters(self)`
- `MicroTopoBrain.forward` (method) `qwen.py:223` `def forward(self, x, pgd_loss)`
- `MicroTopoBrain.micro_pgd_attack` (method) `qwen.py:283` `def micro_pgd_attack(model, x, y, eps, steps, pgd_loss)`
- `MicroTopoBrain.train_with_cv` (method) `qwen.py:306` `def train_with_cv(config, dataset, cv_folds)`
- `MicroTopoBrain.generate_ablation_matrix` (method) `qwen.py:367` `def generate_ablation_matrix()`
- `MicroTopoBrain.run_ablation_study` (method) `qwen.py:392` `def run_ablation_study()`

## qwen3.py
- `Config.seed_everything` (method) `qwen3.py:50` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `qwen3.py:61` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `qwen3.py:71` `def get_batch(self, phase, bs)`
- `DataEnvironment.get_full` (method) `qwen3.py:85` `def get_full(self)`
- `DataEnvironment.get_w2` (method) `qwen3.py:88` `def get_w2(self)`
- `HomeostaticRegulator.__init__` (method) `qwen3.py:95` `def __init__(self, d_in)`
- `HomeostaticRegulator.forward` (method) `qwen3.py:105` `def forward(self, x, h_pre, w_norm)`
- `PhysioNeuron.__init__` (method) `qwen3.py:121` `def __init__(self, d_in, d_out, dynamic)`
- `PhysioNeuron.forward` (method) `qwen3.py:132` `def forward(self, x)`
- `RegulableSymbiotic.__init__` (method) `qwen3.py:161` `def __init__(self, dim, atoms)`
- `RegulableSymbiotic.forward` (method) `qwen3.py:168` `def forward(self, x, influence)`
- `RegulableTopology.__init__` (method) `qwen3.py:180` `def __init__(self, num_nodes)`
- `RegulableTopology.get_adjacency` (method) `qwen3.py:193` `def get_adjacency(self, plasticity)`
- `MicroTopoBrain.__init__` (method) `qwen3.py:202` `def __init__(self, config)`
- `MicroTopoBrain.count_parameters` (method) `qwen3.py:221` `def count_parameters(self)`
- `MicroTopoBrain.forward` (method) `qwen3.py:224` `def forward(self, x)`
- `MicroTopoBrain.train_nonstationary` (method) `qwen3.py:263` `def train_nonstationary(config)`
- `MicroTopoBrain.generate_ablation_matrix` (method) `qwen3.py:320` `def generate_ablation_matrix()`
- `MicroTopoBrain.run_ablation_study` (method) `qwen3.py:352` `def run_ablation_study()`

## qwen4.py
- `Config.seed_everything` (method) `qwen4.py:50` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `qwen4.py:61` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `qwen4.py:71` `def get_batch(self, phase, bs)`
- `DataEnvironment.get_full` (method) `qwen4.py:85` `def get_full(self)`
- `DataEnvironment.get_w2` (method) `qwen4.py:88` `def get_w2(self)`
- `HomeostaticRegulator.__init__` (method) `qwen4.py:95` `def __init__(self, d_in)`
- `HomeostaticRegulator.forward` (method) `qwen4.py:105` `def forward(self, x, h_pre, w_norm)`
- `PhysioNeuron.__init__` (method) `qwen4.py:121` `def __init__(self, d_in, d_out, dynamic)`
- `PhysioNeuron.forward` (method) `qwen4.py:132` `def forward(self, x)`
- `RegulableSymbiotic.__init__` (method) `qwen4.py:161` `def __init__(self, dim, atoms)`
- `RegulableSymbiotic.forward` (method) `qwen4.py:168` `def forward(self, x, influence)`
- `RegulableTopology.__init__` (method) `qwen4.py:180` `def __init__(self, num_nodes)`
- `RegulableTopology.get_adjacency` (method) `qwen4.py:193` `def get_adjacency(self, plasticity)`
- `MicroTopoBrain.__init__` (method) `qwen4.py:202` `def __init__(self, config)`
- `MicroTopoBrain.count_parameters` (method) `qwen4.py:221` `def count_parameters(self)`
- `MicroTopoBrain.forward` (method) `qwen4.py:224` `def forward(self, x)`
- `MicroTopoBrain.train_nonstationary` (method) `qwen4.py:263` `def train_nonstationary(config)`
- `MicroTopoBrain.generate_ablation_matrix` (method) `qwen4.py:325` `def generate_ablation_matrix()`
- `MicroTopoBrain.run_ablation_study` (method) `qwen4.py:357` `def run_ablation_study()`

## qwen5.py
- `Config.seed_everything` (method) `qwen5.py:49` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `qwen5.py:60` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `qwen5.py:71` `def get_batch(self, phase, bs)`
- `DataEnvironment.get_full` (method) `qwen5.py:85` `def get_full(self)`
- `DataEnvironment.get_w2` (method) `qwen5.py:88` `def get_w2(self)`
- `WorldModel.__init__` (method) `qwen5.py:96` `def __init__(self, hidden_dim)`
- `WorldModel.forward` (method) `qwen5.py:103` `def forward(self, phase_id)`
- `EpisodeMemory.__init__` (method) `qwen5.py:118` `def __init__(self, capacity)`
- `EpisodeMemory.store` (method) `qwen5.py:122` `def store(self, phase, metrics, state)`
- `EpisodeMemory.retrieve` (method) `qwen5.py:129` `def retrieve(self, phase, top_k)`
- `PredictiveHomeostat.__init__` (method) `qwen5.py:139` `def __init__(self, d_in)`
- `PredictiveHomeostat.forward` (method) `qwen5.py:153` `def forward(self, x, h_pre, w_norm, phase, reward)`
- `PredictivePhysioNeuron.__init__` (method) `qwen5.py:197` `def __init__(self, d_in, d_out, dynamic)`
- `PredictivePhysioNeuron.forward` (method) `qwen5.py:209` `def forward(self, x, phase, reward)`
- `PredictivePhysioNeuron.consolidate_svd` (method) `qwen5.py:239` `def consolidate_svd(self, repair_strength)`
- `PhysioChimeraV15.__init__` (method) `qwen5.py:248` `def __init__(self, config)`
- `PhysioChimeraV15.count_parameters` (method) `qwen5.py:261` `def count_parameters(self)`
- `PhysioChimeraV15.forward` (method) `qwen5.py:264` `def forward(self, x, phase, reward)`
- `PhysioChimeraV15.train_predictive` (method) `qwen5.py:285` `def train_predictive(config)`
- `PhysioChimeraV15.run_experiment` (method) `qwen5.py:351` `def run_experiment()`

## qwen6.py
- `Config.seed_everything` (method) `qwen6.py:43` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `qwen6.py:54` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `qwen6.py:64` `def get_batch(self, phase, bs)`
- `DataEnvironment.get_full` (method) `qwen6.py:78` `def get_full(self)` -- Retorna el dataset completo
- `DataEnvironment.get_w2` (method) `qwen6.py:82` `def get_w2(self)` -- Retorna solo los datos de WORLD_2 (dígitos >= 5)
- `SelfModifyingGates.__init__` (method) `qwen6.py:90` `def __init__(self, input_dim, hidden_dim)`
- `SelfModifyingGates.forward` (method) `qwen6.py:97` `def forward(self, x)`
- `ContinuumMemorySystem.__init__` (method) `qwen6.py:111` `def __init__(self, levels, d_model, hidden_dim)`
- `ContinuumMemorySystem.forward` (method) `qwen6.py:123` `def forward(self, x, global_step)`
- `NestedPhysioNeuron.__init__` (method) `qwen6.py:134` `def __init__(self, d_in, d_out, config)`
- `NestedPhysioNeuron.forward` (method) `qwen6.py:145` `def forward(self, x, global_step)`
- `PhysioChimeraNested.__init__` (method) `qwen6.py:170` `def __init__(self, config)`
- `PhysioChimeraNested.forward` (method) `qwen6.py:182` `def forward(self, x, global_step)`
- `PhysioChimeraNested.train_nested` (method) `qwen6.py:202` `def train_nested(config)`
- `PhysioChimeraNested.run_experiment` (method) `qwen6.py:265` `def run_experiment()`

## qwen8.py
- `Config.seed_everything` (method) `qwen8.py:45` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `qwen8.py:56` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `qwen8.py:66` `def get_batch(self, phase, bs)`
- `DataEnvironment.get_full` (method) `qwen8.py:80` `def get_full(self)`
- `DataEnvironment.get_w2` (method) `qwen8.py:83` `def get_w2(self)`
- `HomeostaticRegulator.__init__` (method) `qwen8.py:90` `def __init__(self, d_in)`
- `HomeostaticRegulator.forward` (method) `qwen8.py:100` `def forward(self, x, h_pre, w_norm, task_loss)`
- `PhysioNeuron.__init__` (method) `qwen8.py:117` `def __init__(self, d_in, d_out, dynamic)`
- `PhysioNeuron.forward` (method) `qwen8.py:129` `def forward(self, x, task_loss)`
- `SupConHead.__init__` (method) `qwen8.py:161` `def __init__(self, in_dim)`
- `SupConHead.forward` (method) `qwen8.py:169` `def forward(self, x)`
- `MicroTopoBrain.__init__` (method) `qwen8.py:176` `def __init__(self, config)`
- `MicroTopoBrain.count_parameters` (method) `qwen8.py:189` `def count_parameters(self)`
- `MicroTopoBrain.forward` (method) `qwen8.py:192` `def forward(self, x, task_loss)`
- `NeuralDiagnostics.__init__` (method) `qwen8.py:218` `def __init__(self)`
- `NeuralDiagnostics.update` (method) `qwen8.py:228` `def update(self, loss, liquid_norm, physio, prediction_error)`
- `NeuralDiagnostics.get_recent_avg` (method) `qwen8.py:236` `def get_recent_avg(self, key, n)`
- `NeuralDiagnostics.report` (method) `qwen8.py:241` `def report(self, step, phase)`
- `NeuralDiagnostics.train_nonstationary` (method) `qwen8.py:267` `def train_nonstationary(config)`
- `NeuralDiagnostics.run_ablation_study` (method) `qwen8.py:332` `def run_ablation_study()`

## qwen9.py
- `LiquidNeuron.__init__` (method) `qwen9.py:35` `def __init__(self, in_dim, out_dim)`
- `LiquidNeuron.forward` (method) `qwen9.py:49` `def forward(self, x, global_plasticity, transfer_rate)`
- `RightHemisphere.__init__` (method) `qwen9.py:79` `def __init__(self, output_dim)`
- `RightHemisphere.forward` (method) `qwen9.py:88` `def forward(self, image, plasticity, transfer_rate)`
- `LeftHemisphere.__init__` (method) `qwen9.py:98` `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `LeftHemisphere.forward` (method) `qwen9.py:119` `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)`
- `CorpusCallosum.__init__` (method) `qwen9.py:196` `def __init__(self, dim)`
- `CorpusCallosum.forward` (method) `qwen9.py:209` `def forward(self, right_features, left_context)`
- `HomeostaticRegulator.__init__` (method) `qwen9.py:220` `def __init__(self, dim)`
- `HomeostaticRegulator.forward` (method) `qwen9.py:230` `def forward(self, right_features, epoch)`
- `HomeostaticRegulator.update_flow_ema` (method) `qwen9.py:245` `def update_flow_ema(self, flow)`
- `NeuroLogosBicameral.__init__` (method) `qwen9.py:252` `def __init__(self, vocab_size)`
- `NeuroLogosBicameral.forward` (method) `qwen9.py:259` `def forward(self, image, captions, epoch, return_diagnostics)`
- `NeuralDiagnostics.__init__` (method) `qwen9.py:292` `def __init__(self)`
- `NeuralDiagnostics.measure_callosal_flow` (method) `qwen9.py:299` `def measure_callosal_flow(self, right_features, left_context)`
- `NeuralDiagnostics.measure_vocab_diversity` (method) `qwen9.py:306` `def measure_vocab_diversity(self, tokens, vocab_size)`
- `NeuralDiagnostics.update` (method) `qwen9.py:312` `def update(self)`
- `NeuralDiagnostics.get_recent_avg` (method) `qwen9.py:317` `def get_recent_avg(self, key, n)`
- `NeuralDiagnostics.report` (method) `qwen9.py:321` `def report(self, epoch)`
- `Flickr8kDataset.__init__` (method) `qwen9.py:339` `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `Flickr8kDataset.build_vocab_flickr` (method) `qwen9.py:371` `def build_vocab_flickr(captions_file, vocab_size)`
- `Flickr8kDataset.setup_flickr8k` (method) `qwen9.py:386` `def setup_flickr8k(data_dir)`
- `Flickr8kDataset.train_bicameral` (method) `qwen9.py:400` `def train_bicameral()`

## qwn2.py
- `MicroConfig.seed_everything` (method) `qwn2.py:54` `def seed_everything(seed)`
- `MicroConfig.get_dataset` (method) `qwn2.py:61` `def get_dataset(config)`
- `HomeostaticRegulator.__init__` (method) `qwn2.py:81` `def __init__(self, input_dim)`
- `HomeostaticRegulator.forward` (method) `qwn2.py:91` `def forward(self, signals)`
- `AutoregulatedPlasticity.__init__` (method) `qwn2.py:98` `def __init__(self, num_nodes, grid_size)`
- `AutoregulatedPlasticity.get_adjacency` (method) `qwn2.py:111` `def get_adjacency(self, x, h_agg)`
- `AutoregulatedContinuum.__init__` (method) `qwn2.py:123` `def __init__(self, dim)`
- `AutoregulatedContinuum.forward` (method) `qwn2.py:133` `def forward(self, x)`
- `AutoregulatedSymbiotic.__init__` (method) `qwn2.py:156` `def __init__(self, dim, num_atoms)`
- `AutoregulatedSymbiotic.forward` (method) `qwn2.py:164` `def forward(self, x)`
- `AutoregulatedSupConHead.__init__` (method) `qwn2.py:180` `def __init__(self, in_dim)`
- `AutoregulatedSupConHead.forward` (method) `qwn2.py:189` `def forward(self, x, entropy)`
- `MicroTopoBrain.__init__` (method) `qwn2.py:200` `def __init__(self, config)`
- `MicroTopoBrain.count_parameters` (method) `qwn2.py:220` `def count_parameters(self)`
- `MicroTopoBrain.forward` (method) `qwn2.py:223` `def forward(self, x)`
- `MicroTopoBrain.micro_pgd_attack` (method) `qwn2.py:259` `def micro_pgd_attack(model, x, y, eps, steps)`
- `MicroTopoBrain.train_with_cv` (method) `qwn2.py:279` `def train_with_cv(config, dataset, cv_folds)`
- `MicroTopoBrain.generate_ablation_matrix` (method) `qwn2.py:341` `def generate_ablation_matrix()`
- `MicroTopoBrain.run_ablation_study` (method) `qwn2.py:366` `def run_ablation_study()`

## resma4.10.py
- `RESMAConstants.verify_pt_condition` (method) `resma4.10.py:50` `def verify_pt_condition(cls)`
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.10.py:86` `def epsilon_critico(self)`
- `GarnierTresTiempos.modulation_factor` (method) `resma4.10.py:89` `def modulation_factor(self)`
- `GarnierTresTiempos.to_dict` (method) `resma4.10.py:92` `def to_dict(self)`
- `GarnierTresTiempos.from_dict` (method) `resma4.10.py:102` `def from_dict(cls, data)`
- `OperadorDesdoblamiento.__init__` (method) `resma4.10.py:112` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.10.py:135` `def operator(self)`
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.10.py:150` `def calcular_alpha_modificado(self, alpha_base)`
- `SilencioActivoMonitor.__init__` (method) `resma4.10.py:159` `def __init__(self, garnier)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.10.py:163` `def calcular_delta_s_loop(self, rho_red, b1)`
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.10.py:170` `def es_silencio_activo(self, rho_red, b1)`
- `QuantumLeaf.spectral_density` (method) `resma4.10.py:197` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.10.py:203` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.10.py:234` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.10.py:342` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `MyelinCavity.__init__` (method) `resma4.10.py:510` `def __init__(self, axon_length, radius, n_modes)`
- `ExperimentalPredictions.__init__` (method) `resma4.10.py:546` `def __init__(self, universe, network, myelin)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.10.py:551` `def compute_log_bayes_factor(self)`
- `ResourceMonitor.get_memory_gb` (method) `resma4.10.py:594` `def get_memory_gb()`
- `ResourceMonitor.log_resources` (method) `resma4.10.py:599` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.10.py:604` `def guardar_checkpoint(data, filename)`
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.10.py:662` `def cargar_checkpoint(filename)`
- `ResourceMonitor.simulate_resma_garnier` (method) `resma4.10.py:794` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`

## resma4.2.py
- `RESMAConstants.verify_pt_condition` (method) `resma4.2.py:67` `def verify_pt_condition(cls)` -- Verificar condición PT: κ < χΩ
- `PhysicalValidator.validate_dimension` (method) `resma4.2.py:80` `def validate_dimension(alpha, tolerance)`
- `PhysicalValidator.validate_pt_symmetry` (method) `resma4.2.py:88` `def validate_pt_symmetry(kappa, Omega, chi)`
- `PhysicalValidator.validate_connectome_size` (method) `resma4.2.py:96` `def validate_connectome_size(n_nodes)`
- `PhysicalValidator.validate_spectral_dimension` (method) `resma4.2.py:101` `def validate_spectral_dimension(dim)`
- `QuantumLeaf.spectral_density` (method) `resma4.2.py:121` `def spectral_density(self, omega)` -- ρ(ω) con regularización UV
- `QuantumLeaf.modular_entropy` (method) `resma4.2.py:126` `def modular_entropy(self)` -- S = -∫ ρ log ρ dω
- `QuantumLeaf.bures_distance` (method) `resma4.2.py:136` `def bures_distance(self, other)` -- Distancia de Bures W₂(ρ₁, ρ₂)
- `QuantumLeaf.haagerup_weight` (method) `resma4.2.py:154` `def haagerup_weight(self)` -- Peso de Haagerup para regularización
- `RESMAUniverse.__init__` (method) `resma4.2.py:165` `def __init__(self, n_leaves, seed)`
- `RESMAUniverse.compute_gibbs_free_energy` (method) `resma4.2.py:216` `def compute_gibbs_free_energy(self)`
- `EmunaOperator.__init__` (method) `resma4.2.py:227` `def __init__(self, universe, n_samples)`
- `EmunaOperator.project` (method) `resma4.2.py:258` `def project(self, state_vector)` -- P̂_E = P_E ∘ Φ_E (con interpolación adaptativa)
- `MyelinCavity.coherence_quantum` (method) `resma4.2.py:327` `def coherence_quantum(self)` -- Coherencia cuántica con verificación espectral
- `NeuralNetworkRESMA.__init__` (method) `resma4.2.py:354` `def __init__(self, n_nodes, seed)`
- `NeuralNetworkRESMA.critical_percolation_time` (method) `resma4.2.py:461` `def critical_percolation_time(self)` -- t_c = 21 · (N/N₀)^0.25 / log R_Q
- `ExperimentalPredictions.__init__` (method) `resma4.2.py:477` `def __init__(self, universe, myelin, network)`
- `ExperimentalPredictions.predict_all` (method) `resma4.2.py:483` `def predict_all(self)` -- Predicciones RESMA 4.2
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.2.py:496` `def compute_log_bayes_factor(self)` -- ln(BF) con AIC
- `ExperimentalPredictions.simulate_resma_complete` (method) `resma4.2.py:556` `def simulate_resma_complete(n_leaves, n_nodes, seed)` -- Pipeline RESMA 4.2 completo

## resma4.3.py
- `ResourceMonitor.get_memory_gb` (method) `resma4.3.py:35` `def get_memory_gb()`
- `ResourceMonitor.check_memory_limit` (method) `resma4.3.py:40` `def check_memory_limit()`
- `ResourceMonitor.log_resources` (method) `resma4.3.py:49` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.3.py:54` `def guardar_checkpoint(data, filename)` -- Guardado atómico con backup
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.3.py:84` `def cargar_checkpoint(filename)` -- Cargar checkpoint con fallback
- `RESMAConstants.verify_pt_condition` (method) `resma4.3.py:127` `def verify_pt_condition(cls)`
- `QuantumLeaf.spectral_density` (method) `resma4.3.py:154` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.3.py:158` `def bures_distance(self, other)` -- Distancia Bures con caché EXTERNO (no en instancia)
- `RESMAUniverse.__init__` (method) `resma4.3.py:193` `def __init__(self, n_leaves, seed)`
- `PhysicalValidator.validate_dimension` (method) `resma4.3.py:271` `def validate_dimension(alpha, tolerance)`
- `PhysicalValidator.validate_pt_symmetry` (method) `resma4.3.py:279` `def validate_pt_symmetry(kappa, Omega, chi)`
- `PhysicalValidator.validate_connectome_size` (method) `resma4.3.py:287` `def validate_connectome_size(n_nodes)`
- `PhysicalValidator.validate_spectral_dimension` (method) `resma4.3.py:292` `def validate_spectral_dimension(dim)`
- `MyelinCavity.coherence_quantum` (method) `resma4.3.py:331` `def coherence_quantum(self)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.3.py:353` `def __init__(self, n_nodes, seed)`
- `NeuralNetworkRESMA.critical_percolation_time` (method) `resma4.3.py:476` `def critical_percolation_time(self)` -- Tiempo crítico de percolación
- `ExperimentalPredictions.__init__` (method) `resma4.3.py:486` `def __init__(self, universe, myelin, network)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.3.py:492` `def compute_log_bayes_factor(self)`
- `ExperimentalPredictions.simulate_resma_with_checkpointing` (method) `resma4.3.py:545` `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)` -- Pipeline con reanudación inteligente desde checkpoints

## resma4.4.py
- `ResourceMonitor.get_memory_gb` (method) `resma4.4.py:36` `def get_memory_gb()`
- `ResourceMonitor.check_memory_limit` (method) `resma4.4.py:41` `def check_memory_limit()`
- `ResourceMonitor.log_resources` (method) `resma4.4.py:50` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.4.py:59` `def guardar_checkpoint(data, filename)` -- Guarda el estado COMPLETO de los objetos, no solo metadatos
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.4.py:93` `def cargar_checkpoint(filename)` -- Carga el estado COMPLETO desde disco
- `RESMAConstants.verify_pt_condition` (method) `resma4.4.py:146` `def verify_pt_condition(cls)`
- `QuantumLeaf.spectral_density` (method) `resma4.4.py:172` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.4.py:176` `def bures_distance(self, other)` -- Distancia Bures con caché externo
- `RESMAUniverse.__init__` (method) `resma4.4.py:206` `def __init__(self, n_leaves, seed, leaves, measure, global_state)` -- Constructor que puede recibir estado serializado
- `PhysicalValidator.validate_dimension` (method) `resma4.4.py:318` `def validate_dimension(alpha, tolerance)`
- `PhysicalValidator.validate_pt_symmetry` (method) `resma4.4.py:326` `def validate_pt_symmetry(kappa, Omega, chi)`
- `PhysicalValidator.validate_connectome_size` (method) `resma4.4.py:334` `def validate_connectome_size(n_nodes)`
- `PhysicalValidator.validate_spectral_dimension` (method) `resma4.4.py:339` `def validate_spectral_dimension(dim)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.4.py:378` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti)` -- Constructor que puede recibir grafo ya construido
- `NeuralNetworkRESMA.simulate_resma_with_checkpointing` (method) `resma4.4.py:554` `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)` -- Pipeline con reanudación que realmente carga objetos

## resma4.5.py
- `ResourceMonitor.get_memory_gb` (method) `resma4.5.py:36` `def get_memory_gb()`
- `ResourceMonitor.check_memory_limit` (method) `resma4.5.py:41` `def check_memory_limit()`
- `ResourceMonitor.log_resources` (method) `resma4.5.py:50` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.5.py:59` `def guardar_checkpoint(data, filename)`
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.5.py:88` `def cargar_checkpoint(filename)`
- `RESMAConstants.verify_pt_condition` (method) `resma4.5.py:152` `def verify_pt_condition(cls)`
- `GarnierTresTiempos.factor_escala` (method) `resma4.5.py:181` `def factor_escala(self, tiempo_idx)` -- Factor de escala para cada tiempo: 0=lento, 2=modular, 3=teleológico
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.5.py:185` `def epsilon_critico(self)` -- Entropía crítica de percolación (ADIMENSIONAL). log(2) es la entropía de un bit cuántico crítico.
- `GarnierTresTiempos.to_dict` (method) `resma4.5.py:192` `def to_dict(self)` -- Para serialización
- `GarnierTresTiempos.from_dict` (method) `resma4.5.py:197` `def from_dict(cls, data)`
- `OperadorDesdoblamiento.__init__` (method) `resma4.5.py:206` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.5.py:235` `def operator(self)` -- Construye D̂_G(ϕ) dimensionalmente consistente
- `OperadorDesdoblamiento.aplicar_a_estado` (method) `resma4.5.py:254` `def aplicar_a_estado(self, estado)` -- Aplica desdoblamiento a un estado cuántico |Ψ⟩
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.5.py:260` `def calcular_alpha_modificado(self, alpha_base)` -- α'(ϕ) = α · tanh(C0/C3 · cos(ϕ₃)) Garantiza α' ∈ [0, α]
- `SilencioActivoMonitor.__init__` (method) `resma4.5.py:273` `def __init__(self, garnier, network)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.5.py:278` `def calcular_delta_s_loop(self, rho_red)` -- ΔS_loop = S_vN(ρ_red) - log(b₁ + 1) rho_red: matriz densidad reducida (si es None, se calcula)
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.5.py:307` `def es_silencio_activo(self, rho_red)` -- Verifica Silencio-Activo y calcula Libertad L.
- `SilencioActivoMonitor.umbral_percolacion` (method) `resma4.5.py:324` `def umbral_percolacion(self)` -- Umbral de percolación para soberanía: 70% (Axioma 6)
- `QuantumLeaf.spectral_density` (method) `resma4.5.py:349` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.5.py:353` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.5.py:383` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)` -- Constructor que puede recibir estado serializado
- `MyelinCavity.__init__` (method) `resma4.5.py:488` `def __init__(self, axon_length, radius, n_modes)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.5.py:520` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)` -- Constructor que puede recibir grafo ya construido
- `NeuralNetworkRESMA.validar_axioma_6` (method) `resma4.5.py:644` `def validar_axioma_6(self)` -- Verifica: conectividad > 70% para soberanía
- `ExperimentalPredictions.__init__` (method) `resma4.5.py:661` `def __init__(self, universe, myelin, network)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.5.py:666` `def compute_log_bayes_factor(self)` -- Calcula Factor de Bayes integrando Garnier
- `ExperimentalPredictions.simulate_resma_garnier` (method) `resma4.5.py:695` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)` -- Pipeline único con Garnier integrado

## resma4.6.py
- `ResourceMonitor.get_memory_gb` (method) `resma4.6.py:33` `def get_memory_gb()`
- `ResourceMonitor.check_memory_limit` (method) `resma4.6.py:38` `def check_memory_limit(threshold)`
- `ResourceMonitor.log_resources` (method) `resma4.6.py:47` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.6.py:56` `def guardar_checkpoint(data, filename)` -- Guarda estado completo con manejo robusto de errores
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.6.py:86` `def cargar_checkpoint(filename)` -- Carga checkpoint con fallback automático
- `RESMAConstants.verify_pt_condition` (method) `resma4.6.py:158` `def verify_pt_condition(cls)`
- `GarnierTresTiempos.factor_escala` (method) `resma4.6.py:194` `def factor_escala(self, tiempo_idx)` -- Factor de escala con supresión ZPE
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.6.py:200` `def epsilon_critico(self)` -- **UMBRAL CRÍTICO CON ZPE**: Cuando zpe_level → 0, ε_c → 0 (Silencio perfecto no necesita umbral)
- `GarnierTresTiempos.to_dict` (method) `resma4.6.py:208` `def to_dict(self)` -- Serialización completa
- `GarnierTresTiempos.from_dict` (method) `resma4.6.py:220` `def from_dict(cls, data)` -- Deserialización
- `OperadorDesdoblamiento.__init__` (method) `resma4.6.py:238` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.6.py:284` `def operator(self)` -- Construye D̂_G(ϕ) con cancelación ZPE
- `OperadorDesdoblamiento.alpha_modificado` (method) `resma4.6.py:304` `def alpha_modificado(self, alpha_base)` -- **α'(ϕ) = α · tanh(C₀/C₃ · cos(ϕ₃) · (1 - zpe_level))**
- `SilencioActivoMonitor.__init__` (method) `resma4.6.py:318` `def __init__(self, garnier, network)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.6.py:327` `def calcular_delta_s_loop(self, rho_red)` -- **ΔS_loop = S_vN(ρ_red) - S_top + S_ZPE** **NUEVO**: La entropía ZPE se SUMA a la entropía total
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.6.py:383` `def es_silencio_activo(self, rho_red)` -- **DETECCIÓN DE ANTAGONISMO**: Retorna: (condicion, libertad_L, nivel_ZPE_cancelado)
- `SilencioActivoMonitor.umbral_percolacion` (method) `resma4.6.py:411` `def umbral_percolacion(self)` -- Umbral para soberanía: 70%
- `SilencioActivoMonitor.modo_goldstone` (method) `resma4.6.py:415` `def modo_goldstone(self)` -- **MODO GOLDSTONE DEL DOBLE CUÁNTICO**: Excitación colectiva que anuncia ruptura de simetría ZPE
- `QuantumLeaf.spectral_density` (method) `resma4.6.py:453` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.6.py:457` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.6.py:487` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `MyelinCavity.__init__` (method) `resma4.6.py:611` `def __init__(self, axon_length, radius, n_modes)`
- `ConectomaCuantico.__init__` (method) `resma4.6.py:667` `def __init__(self, n_nodes, seed, garnier)`
- `ConectomaCuantico.colapsar_a_clasico` (method) `resma4.6.py:718` `def colapsar_a_clasico(self, threshold)` -- **COLAPSO CUÁNTICO-CLÁSICO**: - Medición proyectiva con umbral de probabilidad - b1_clásico ≠ b1_cuántico
- `ConectomaCuantico.medir_delta_s_loop` (method) `resma4.6.py:756` `def medir_delta_s_loop(self)` -- **ΔS_loop CUÁNTICO** (no clásico): - Usa matriz densidad de amplitudes (no grafo) - S_ZPE es intrínseca a la...
- `NeuralNetworkRESMA.__init__` (method) `resma4.6.py:803` `def __init__(self, n_nodes, seed, conectoma_quantum, garnier)`
- `NeuralNetworkRESMA.validar_axioma_6_cuantico` (method) `resma4.6.py:858` `def validar_axioma_6_cuantico(self)` -- **AXIOMA 6 CUÁNTICO**: Conectividad cuántica > 70% **NUEVO**: La soberanía se juzga en el estado pre-geométrico, no...
- `NeuralNetworkRESMA.obtener_metricas_cuanticas` (method) `resma4.6.py:874` `def obtener_metricas_cuanticas(self)` -- **MÉTRICAS EXPERIMENTALES** (falsables): - Conectividad cuántica (pre-observación) - b1 cuántico vs b1 clásico...
- `ExperimentalPredictions.__init__` (method) `resma4.6.py:901` `def __init__(self, universe, myelin, network)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.6.py:906` `def compute_log_bayes_factor(self)` -- Calcula Factor de Bayes con antagonismo ZPE-Silencio
- `ExperimentalPredictions.simulate_resma_garnier` (method) `resma4.6.py:951` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart, target_connectivity)` -- Pipeline completo RESMA 4.3.5 con antagonismo ZPE-Silencio

## resma4.7.py
- `GarnierTresTiempos.factor_escala` (method) `resma4.7.py:72` `def factor_escala(self, tiempo_idx)` -- Factor de escala temporal
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.7.py:77` `def epsilon_critico(self)` -- Entropía crítica con corrección de acoplamiento: ε_c = log(2) · (C0/C3)² · (1 + ξ)
- `GarnierTresTiempos.modulation_factor` (method) `resma4.7.py:85` `def modulation_factor(self)` -- Factor de modulación para la medida cuántica: M = exp(-|φ₃ - π|/C3) Máximo cuando φ₃ ≈ π (apertura temporal óptima)
- `OperadorDesdoblamiento.__init__` (method) `resma4.7.py:98` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.7.py:115` `def operator(self)` -- Construye D̂_G(φ) = exp(i Σ φᵢHᵢ)
- `OperadorDesdoblamiento.aplicar_modulacion` (method) `resma4.7.py:120` `def aplicar_modulacion(self, state_vector)` -- Aplica desdoblamiento a vector de estado
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.7.py:126` `def calcular_alpha_modificado(self, alpha_base)` -- α'(φ) = α · |cos(φ₃)|^(C0/C3) Garantiza α' ∈ [0, α]
- `SilencioActivoMonitor.__init__` (method) `resma4.7.py:143` `def __init__(self, garnier)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.7.py:147` `def calcular_delta_s_loop(self, rho_red, b1)` -- ΔS_loop = S_vN(ρ) - log(b₁ + 1)
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.7.py:166` `def es_silencio_activo(self, rho_red, b1)` -- Verifica condición y calcula libertad L = 1/(ΔS + ε_c)
- `QuantumLeaf.spectral_density` (method) `resma4.7.py:197` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.7.py:203` `def bures_distance(self, other)` -- Distancia de Bures simplificada
- `RESMAUniverse.__init__` (method) `resma4.7.py:226` `def __init__(self, n_leaves, seed, garnier)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.7.py:341` `def __init__(self, n_nodes, seed, garnier)`
- `ExperimentalPredictions.__init__` (method) `resma4.7.py:487` `def __init__(self, universe, network)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.7.py:491` `def compute_log_bayes_factor(self)` -- ln(BF) ∝ log(L_red · L_univ) Veredicto basado en libertad total
- `ExperimentalPredictions.simulate_resma_garnier` (method) `resma4.7.py:537` `def simulate_resma_garnier(n_leaves, n_nodes, seed)` -- Pipeline completo RESMA-Garnier con correcciones

## resma4.8.py
- `RESMAConstants.verify_pt_condition` (method) `resma4.8.py:54` `def verify_pt_condition(cls)`
- `ResourceMonitor.get_memory_gb` (method) `resma4.8.py:72` `def get_memory_gb()`
- `ResourceMonitor.check_memory_limit` (method) `resma4.8.py:77` `def check_memory_limit()`
- `ResourceMonitor.log_resources` (method) `resma4.8.py:86` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.8.py:91` `def guardar_checkpoint(data, filename)`
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.8.py:117` `def cargar_checkpoint(filename)`
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.8.py:182` `def epsilon_critico(self)`
- `GarnierTresTiempos.modulation_factor` (method) `resma4.8.py:186` `def modulation_factor(self)`
- `GarnierTresTiempos.to_dict` (method) `resma4.8.py:189` `def to_dict(self)`
- `GarnierTresTiempos.from_dict` (method) `resma4.8.py:199` `def from_dict(cls, data)`
- `OperadorDesdoblamiento.__init__` (method) `resma4.8.py:210` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.8.py:233` `def operator(self)`
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.8.py:248` `def calcular_alpha_modificado(self, alpha_base)`
- `SilencioActivoMonitor.__init__` (method) `resma4.8.py:257` `def __init__(self, garnier)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.8.py:261` `def calcular_delta_s_loop(self, rho_red, b1)`
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.8.py:268` `def es_silencio_activo(self, rho_red, b1)`
- `QuantumLeaf.spectral_density` (method) `resma4.8.py:295` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.8.py:301` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.8.py:332` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.8.py:434` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `MyelinCavity.__init__` (method) `resma4.8.py:585` `def __init__(self, axon_length, radius, n_modes)`
- `ExperimentalPredictions.__init__` (method) `resma4.8.py:621` `def __init__(self, universe, network, myelin)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.8.py:626` `def compute_log_bayes_factor(self)`
- `ExperimentalPredictions.simulate_resma_garnier` (method) `resma4.8.py:667` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`

## resma4.9.py
- `RESMAConstants.verify_pt_condition` (method) `resma4.9.py:50` `def verify_pt_condition(cls)`
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.9.py:86` `def epsilon_critico(self)`
- `GarnierTresTiempos.modulation_factor` (method) `resma4.9.py:89` `def modulation_factor(self)`
- `GarnierTresTiempos.to_dict` (method) `resma4.9.py:92` `def to_dict(self)`
- `GarnierTresTiempos.from_dict` (method) `resma4.9.py:102` `def from_dict(cls, data)`
- `OperadorDesdoblamiento.__init__` (method) `resma4.9.py:112` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.9.py:135` `def operator(self)`
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.9.py:150` `def calcular_alpha_modificado(self, alpha_base)`
- `SilencioActivoMonitor.__init__` (method) `resma4.9.py:159` `def __init__(self, garnier)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.9.py:163` `def calcular_delta_s_loop(self, rho_red, b1)`
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.9.py:170` `def es_silencio_activo(self, rho_red, b1)`
- `QuantumLeaf.spectral_density` (method) `resma4.9.py:197` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.9.py:203` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.9.py:234` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.9.py:342` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `MyelinCavity.__init__` (method) `resma4.9.py:493` `def __init__(self, axon_length, radius, n_modes)`
- `ExperimentalPredictions.__init__` (method) `resma4.9.py:529` `def __init__(self, universe, network, myelin)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.9.py:534` `def compute_log_bayes_factor(self)`
- `ResourceMonitor.get_memory_gb` (method) `resma4.9.py:577` `def get_memory_gb()`
- `ResourceMonitor.log_resources` (method) `resma4.9.py:582` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.9.py:587` `def guardar_checkpoint(data, filename)`
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.9.py:615` `def cargar_checkpoint(filename)`
- `ResourceMonitor.simulate_resma_garnier` (method) `resma4.9.py:655` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`

## resmann.py
- `PTSymmetricActivation.__init__` (method) `resmann.py:12` `def __init__(self, omega, chi, kappa_init)`
- `PTSymmetricActivation.forward` (method) `resmann.py:19` `def forward(self, x)`
- `E8LatticeLayer.__init__` (method) `resmann.py:42` `def __init__(self, in_features, out_features, q_order)`
- `E8LatticeLayer.forward` (method) `resmann.py:69` `def forward(self, x)`
- `RESMABrain.__init__` (method) `resmann.py:87` `def __init__(self, input_dim, hidden_dim, output_dim)`
- `RESMABrain.forward` (method) `resmann.py:97` `def forward(self, x)`
- `RESMABrain.resma_loss` (method) `resmann.py:104` `def resma_loss(self, output, target, lambda_topo)`

## resmann2.py
- `PTSymmetricActivation.__init__` (method) `resmann2.py:14` `def __init__(self, omega, chi, kappa_init)`
- `PTSymmetricActivation.forward` (method) `resmann2.py:20` `def forward(self, x)`
- `E8LatticeMultiverseLayer.__init__` (method) `resmann2.py:32` `def __init__(self, in_features, out_features, n_universes)`
- `E8LatticeMultiverseLayer.forward` (method) `resmann2.py:55` `def forward(self, x)`
- `RESMABrainMultiverse.__init__` (method) `resmann2.py:66` `def __init__(self, input_dim, hidden_dim, output_dim)`
- `RESMABrainMultiverse.forward` (method) `resmann2.py:74` `def forward(self, x)`
- `RESMABrainMultiverse.resma_loss` (method) `resmann2.py:79` `def resma_loss(self, output, target, lambda_topo)`

## resmannn.py
- `PTSymmetricActivation.__init__` (method) `resmannn.py:13` `def __init__(self, omega, chi, kappa_init)`
- `PTSymmetricActivation.forward` (method) `resmannn.py:19` `def forward(self, x)`
- `E8LatticeLayer.__init__` (method) `resmannn.py:31` `def __init__(self, in_features, out_features)`
- `E8LatticeLayer.forward` (method) `resmannn.py:49` `def forward(self, x)`
- `RESMABrainLight.__init__` (method) `resmannn.py:61` `def __init__(self, input_dim, hidden_dim, output_dim)`
- `RESMABrainLight.forward` (method) `resmannn.py:69` `def forward(self, x)`
- `RESMABrainLight.resma_loss` (method) `resmannn.py:74` `def resma_loss(self, output, target, lambda_topo)`

## run_complete_experiment.py
Depends on: `physio_chimera_v15_monitored.py`
- `create_experiment_summary` (function) `run_complete_experiment.py:24` `def create_experiment_summary(results_dir, metrics, duration)` -- Crea un resumen del experimento
- `generate_final_report` (function) `run_complete_experiment.py:46` `def generate_final_report(results_dir, metrics, duration)` -- Genera reporte final detallado
- `run_complete_experiment` (function) `run_complete_experiment.py:155` `def run_complete_experiment()` -- Ejecuta el experimento completo con todas las características

## scientific_benchmark.py
- `seed_everything` (function) `scientific_benchmark.py:42` `def seed_everything(seed)`
- `SupConLoss.__init__` (method) `scientific_benchmark.py:58` `def __init__(self, temperature)`
- `SupConLoss.forward` (method) `scientific_benchmark.py:62` `def forward(self, features, labels)`
- `PredictiveErrorCell.__init__` (method) `scientific_benchmark.py:86` `def __init__(self, dim, use_spectral)`
- `PredictiveErrorCell.forward` (method) `scientific_benchmark.py:92` `def forward(self, input_signal, prediction)`
- `LearnableAbsenceGating.__init__` (method) `scientific_benchmark.py:98` `def __init__(self, dim)`
- `LearnableAbsenceGating.forward` (method) `scientific_benchmark.py:107` `def forward(self, x_sensory, x_prediction)`
- `SymbioticBasisRefinement.__init__` (method) `scientific_benchmark.py:112` `def __init__(self, dim, num_atoms)`
- `SymbioticBasisRefinement.forward` (method) `scientific_benchmark.py:120` `def forward(self, x)`
- `CombinatorialComplexLayer.__init__` (method) `scientific_benchmark.py:133` `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- `CombinatorialComplexLayer.forward` (method) `scientific_benchmark.py:157` `def forward(self, x_nodes, adjacency, incidence)`
- `TopoBrainNet.__init__` (method) `scientific_benchmark.py:185` `def __init__(self, config)`
- `TopoBrainNet.get_topology` (method) `scientific_benchmark.py:241` `def get_topology(self)`
- `TopoBrainNet.forward` (method) `scientific_benchmark.py:253` `def forward(self, x)`
- `TopoBrainNet.clamp_pgd` (method) `scientific_benchmark.py:276` `def clamp_pgd(x_adv_norm, x_orig_norm, eps)`
- `TopoBrainNet.make_adversarial_pgd` (method) `scientific_benchmark.py:283` `def make_adversarial_pgd(model, x, y, eps, steps)`
- `TopoBrainNet.eval_autoattack` (method) `scientific_benchmark.py:299` `def eval_autoattack(model, test_loader, n_samples)`
- `Wrapper.__init__` (method) `scientific_benchmark.py:318` `def __init__(self, m)`
- `Wrapper.forward` (method) `scientific_benchmark.py:319` `def forward(self, x)`
- `Wrapper.save_topology_snapshot` (method) `scientific_benchmark.py:332` `def save_topology_snapshot(model, epoch, run_name)`
- `Wrapper.run_training` (method) `scientific_benchmark.py:350` `def run_training(config_override, run_name)`
- `Wrapper.lambda_topo` (method) `scientific_benchmark.py:387` `def lambda_topo(epoch)`
- `Wrapper.run_ablation_suite_scientific` (method) `scientific_benchmark.py:473` `def run_ablation_suite_scientific()`

## scientist_sinergy_ablation_plan.py
- `ScientificConfig.check_memory_usage` (method) `scientist_sinergy_ablation_plan.py:63` `def check_memory_usage()` -- Monitorea uso de memoria para evitar crashes con Nested Learning
- `ScientificConfig.memory_safe_check` (method) `scientist_sinergy_ablation_plan.py:68` `def memory_safe_check(config)` -- Verifica si es seguro ejecutar con Nested Learning
- `ScientificConfig.setup_matplotlib_for_plotting` (method) `scientist_sinergy_ablation_plan.py:85` `def setup_matplotlib_for_plotting()` -- Setup matplotlib para visualizaciones científicas
- `ScientificConfig.generate_sinergy_matrix` (method) `scientist_sinergy_ablation_plan.py:94` `def generate_sinergy_matrix()` -- Genera matriz de sinergias basada en tus inventos
- `ScientificConfig.print_sinergy_analysis` (method) `scientist_sinergy_ablation_plan.py:139` `def print_sinergy_analysis()` -- Analiza las sinergias propuestas basado en tus modelos
- `ScientificConfig.main` (method) `scientist_sinergy_ablation_plan.py:171` `def main()`

## setup_environment.py
- `check_python_version` (function) `setup_environment.py:15` `def check_python_version()` -- Verifica la versión de Python
- `install_package` (function) `setup_environment.py:23` `def install_package(package)` -- Instala un paquete usando pip
- `check_and_install_dependencies` (function) `setup_environment.py:32` `def check_and_install_dependencies()` -- Verifica e instala dependencias
- `create_directories` (function) `setup_environment.py:76` `def create_directories()` -- Crea directorios necesarios
- `setup_matplotlib` (function) `setup_environment.py:92` `def setup_matplotlib()` -- Configura matplotlib para el entorno
- `create_sample_data` (function) `setup_environment.py:118` `def create_sample_data()` -- Crea datos de muestra para pruebas
- `test_installation` (function) `setup_environment.py:148` `def test_installation()` -- Prueba la instalación
- `create_main_script` (function) `setup_environment.py:189` `def create_main_script()` -- Crea script principal para ejecutar experimentos
- `main` (function) `setup_environment.py:256` `def main()` -- Función principal de setup

## sintesis.py
- `SpectralMonitorV6.__init__` (method) `sintesis.py:13` `def __init__(self, target_entropy)`
- `SpectralMonitorV6.calc_structural_health` (method) `sintesis.py:16` `def calc_structural_health(self, weight_matrix)`
- `SpectralMonitorV6.measure_spatial_richness` (method) `sintesis.py:33` `def measure_spatial_richness(self, activations)`
- `PrismaticNeuron.__init__` (method) `sintesis.py:49` `def __init__(self, in_dim, out_dim)`
- `PrismaticNeuron.forward` (method) `sintesis.py:58` `def forward(self, x)`
- `PrismaticNeuron.prismatic_dream` (method) `sintesis.py:81` `def prismatic_dream(self)` -- Sueño Entrópico: 1.
- `SynthesisOrganismV6.__init__` (method) `sintesis.py:118` `def __init__(self, input_dim, hidden_dim, output_dim)`
- `SynthesisOrganismV6.forward` (method) `sintesis.py:128` `def forward(self, x)`
- `SynthesisOrganismV6.calculate_losses` (method) `sintesis.py:134` `def calculate_losses(self, outputs, targets, criterion)`
- `SynthesisOrganismV6.sleep` (method) `sintesis.py:154` `def sleep(self)`
- `SynthesisOrganismV6.run_prism_dream` (method) `sintesis.py:162` `def run_prism_dream()`


Next: [API_p9.md](API_p9.md)

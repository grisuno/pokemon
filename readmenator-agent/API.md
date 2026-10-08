# API (page 1 of 10)
Pages: [API.md](API.md), [API_p2.md](API_p2.md), [API_p3.md](API_p3.md), [API_p4.md](API_p4.md), [API_p5.md](API_p5.md), [API_p6.md](API_p6.md), [API_p7.md](API_p7.md), [API_p8.md](API_p8.md), [API_p9.md](API_p9.md), [API_p10.md](API_p10.md)

## 01_mcculloch_pitts.py
- `mcculloch_pitts_neuron` (function) `01_mcculloch_pitts.py:7` `def mcculloch_pitts_neuron(inputs, weights, threshold)` -- Neurona artificial de McCulloch-Pitts (1943). - inputs: vector binario de entrada (0 o 1) - weights: vector de pesos...

## 01_topobrain_cou_v2.py
- `Config.to_dict` (method) `01_topobrain_cou_v2.py:56` `def to_dict(self)`
- `Config.get_topology_config` (method) `01_topobrain_cou_v2.py:59` `def get_topology_config(self)` -- Configuración estable para topología adaptable
- `Config.seed_everything` (method) `01_topobrain_cou_v2.py:70` `def seed_everything(seed)`
- `Config.get_tabular_loaders` (method) `01_topobrain_cou_v2.py:77` `def get_tabular_loaders(config)` -- Dataset tabular controlado con características NOIR simuladas
- `StableSupConLoss.__init__` (method) `01_topobrain_cou_v2.py:118` `def __init__(self, temperature)`
- `StableSupConLoss.forward` (method) `01_topobrain_cou_v2.py:123` `def forward(self, features, labels)`
- `StableContinuumMemoryCell.__init__` (method) `01_topobrain_cou_v2.py:149` `def __init__(self, input_dim, hidden_dim)`
- `StableContinuumMemoryCell.forward` (method) `01_topobrain_cou_v2.py:181` `def forward(self, x, controls)`
- `StableSymbioticBasisRefinement.__init__` (method) `01_topobrain_cou_v2.py:233` `def __init__(self, dim, num_atoms)`
- `StableSymbioticBasisRefinement.forward` (method) `01_topobrain_cou_v2.py:242` `def forward(self, x)`
- `StableTopologyManager.__init__` (method) `01_topobrain_cou_v2.py:269` `def __init__(self, num_nodes, config)`
- `StableTopologyManager.get_adjacency` (method) `01_topobrain_cou_v2.py:288` `def get_adjacency(self, plasticity)` -- Obtener matriz de adyacencia con estabilidad garantizada
- `StableTopologyManager.prune_topology` (method) `01_topobrain_cou_v2.py:317` `def prune_topology(self, current_density, epoch)` -- Poda controlada con protocolo de emergencia
- `StableTopologyManager.get_density` (method) `01_topobrain_cou_v2.py:337` `def get_density(self)` -- Calcular densidad actual de manera estable
- `StableTopoBrain.__init__` (method) `01_topobrain_cou_v2.py:347` `def __init__(self, config)`
- `StableTopoBrain.forward` (method) `01_topobrain_cou_v2.py:408` `def forward(self, x, controls)`
- `StableTopoBrain.stable_pgd_attack` (method) `01_topobrain_cou_v2.py:477` `def stable_pgd_attack(model, x, y, eps, steps, controls)` -- Ataque PGD estable con manejo de gradientes robusto
- `StableTopoBrain.train_epoch` (method) `01_topobrain_cou_v2.py:520` `def train_epoch(model, loader, optimizer, config, epoch, controls)` -- Entrenamiento por época con monitoreo detallado
- `StableTopoBrain.evaluate_model` (method) `01_topobrain_cou_v2.py:584` `def evaluate_model(model, loader, config, adversarial, controls)` -- Evaluación rigurosa con o sin ataques adversariales
- `StableTopoBrain.run_scientific_ablation` (method) `01_topobrain_cou_v2.py:632` `def run_scientific_ablation()` -- Ejecución científica del ablation con control de variables

## 01_topobrain_cpu.py
- `Config.to_dict` (method) `01_topobrain_cpu.py:57` `def to_dict(self)`
- `Config.seed_everything` (method) `01_topobrain_cpu.py:63` `def seed_everything(seed)`
- `Config.get_tabular_loaders` (method) `01_topobrain_cpu.py:69` `def get_tabular_loaders(config)`
- `SupConLoss.__init__` (method) `01_topobrain_cpu.py:95` `def __init__(self, temperature)`
- `SupConLoss.forward` (method) `01_topobrain_cpu.py:98` `def forward(self, features, labels)`
- `ContinuumMemoryCell.__init__` (method) `01_topobrain_cpu.py:110` `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate)`
- `ContinuumMemoryCell.forward` (method) `01_topobrain_cpu.py:124` `def forward(self, x, controls)`
- `SymbioticBasisRefinement.__init__` (method) `01_topobrain_cpu.py:144` `def __init__(self, dim)`
- `SymbioticBasisRefinement.forward` (method) `01_topobrain_cpu.py:151` `def forward(self, x)`
- `TopoBrainTabular.__init__` (method) `01_topobrain_cpu.py:167` `def __init__(self, config)`
- `TopoBrainTabular.get_adj` (method) `01_topobrain_cpu.py:198` `def get_adj(self)`
- `TopoBrainTabular.forward` (method) `01_topobrain_cpu.py:203` `def forward(self, x, controls)`
- `TopoBrainTabular.pgd_attack` (method) `01_topobrain_cpu.py:247` `def pgd_attack(model, x, y, eps, steps, controls)`
- `TopoBrainTabular.generate_ablation_configs` (method) `01_topobrain_cpu.py:263` `def generate_ablation_configs(base_config)`
- `TopoBrainTabular.train_and_evaluate` (method) `01_topobrain_cpu.py:293` `def train_and_evaluate(config, name)`
- `TopoBrainTabular.evaluate_adv` (method) `01_topobrain_cpu.py:323` `def evaluate_adv(loader, eps, steps)`
- `TopoBrainTabular.run_ablation` (method) `01_topobrain_cpu.py:337` `def run_ablation()`

## 01_topobrain_cpu_v3.py
- `Config.to_dict` (method) `01_topobrain_cpu_v3.py:57` `def to_dict(self)`
- `Config.seed_everything` (method) `01_topobrain_cpu_v3.py:63` `def seed_everything(seed)`
- `Config.get_tabular_loaders` (method) `01_topobrain_cpu_v3.py:69` `def get_tabular_loaders(config)`
- `SupConLoss.__init__` (method) `01_topobrain_cpu_v3.py:94` `def __init__(self, temperature)`
- `SupConLoss.forward` (method) `01_topobrain_cpu_v3.py:97` `def forward(self, features, labels)`
- `ContinuumMemoryCell.__init__` (method) `01_topobrain_cpu_v3.py:109` `def __init__(self, input_dim, hidden_dim)`
- `ContinuumMemoryCell.forward` (method) `01_topobrain_cpu_v3.py:122` `def forward(self, x, controls)`
- `SymbioticBasisRefinement.__init__` (method) `01_topobrain_cpu_v3.py:146` `def __init__(self, dim)`
- `SymbioticBasisRefinement.forward` (method) `01_topobrain_cpu_v3.py:153` `def forward(self, x)`
- `PrefrontalOrchestrator.__init__` (method) `01_topobrain_cpu_v3.py:166` `def __init__(self, config)`
- `PrefrontalOrchestrator.forward` (method) `01_topobrain_cpu_v3.py:183` `def forward(self, metrics)`
- `PrefrontalOrchestrator.reset_context` (method) `01_topobrain_cpu_v3.py:221` `def reset_context(self)`
- `AdaptiveCombinatorialComplexLayer.__init__` (method) `01_topobrain_cpu_v3.py:228` `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- `AdaptiveCombinatorialComplexLayer.get_adj` (method) `01_topobrain_cpu_v3.py:264` `def get_adj(self)`
- `AdaptiveCombinatorialComplexLayer.forward` (method) `01_topobrain_cpu_v3.py:269` `def forward(self, x, controls)`
- `TopoBrainTabular.__init__` (method) `01_topobrain_cpu_v3.py:321` `def __init__(self, config)`
- `TopoBrainTabular.forward` (method) `01_topobrain_cpu_v3.py:363` `def forward(self, x, controls, prev_states)`
- `TopoBrainTabular.pgd_attack` (method) `01_topobrain_cpu_v3.py:395` `def pgd_attack(model, x, y, eps, steps)`
- `TopoBrainTabular.compute_topology_metrics` (method) `01_topobrain_cpu_v3.py:416` `def compute_topology_metrics(model, config)` -- Computar métricas de topología con manejo robusto de errores
- `TopoBrainTabular.prune_topology` (method) `01_topobrain_cpu_v3.py:450` `def prune_topology(model, config, controls)` -- Implementación simplificada de poda de topología
- `TopoBrainTabular.train_and_evaluate` (method) `01_topobrain_cpu_v3.py:482` `def train_and_evaluate(config, run_name)`
- `TopoBrainTabular.evaluate_adv` (method) `01_topobrain_cpu_v3.py:593` `def evaluate_adv(loader, eps, steps)`
- `TopoBrainTabular.run_ablation` (method) `01_topobrain_cpu_v3.py:640` `def run_ablation()`

## 01_topobrain_cpu_v4.py
- `Config.to_dict` (method) `01_topobrain_cpu_v4.py:56` `def to_dict(self)`
- `Config.get_topology_config` (method) `01_topobrain_cpu_v4.py:59` `def get_topology_config(self)` -- Configuración estable para topología adaptable
- `Config.seed_everything` (method) `01_topobrain_cpu_v4.py:70` `def seed_everything(seed)`
- `Config.get_tabular_loaders` (method) `01_topobrain_cpu_v4.py:77` `def get_tabular_loaders(config)` -- Dataset tabular controlado con características NOIR simuladas
- `StableSupConLoss.__init__` (method) `01_topobrain_cpu_v4.py:118` `def __init__(self, temperature)`
- `StableSupConLoss.forward` (method) `01_topobrain_cpu_v4.py:123` `def forward(self, features, labels)`
- `StableContinuumMemoryCell.__init__` (method) `01_topobrain_cpu_v4.py:149` `def __init__(self, input_dim, hidden_dim)`
- `StableContinuumMemoryCell.forward` (method) `01_topobrain_cpu_v4.py:181` `def forward(self, x, controls)`
- `StableSymbioticBasisRefinement.__init__` (method) `01_topobrain_cpu_v4.py:233` `def __init__(self, dim, num_atoms)`
- `StableSymbioticBasisRefinement.forward` (method) `01_topobrain_cpu_v4.py:242` `def forward(self, x)`
- `StableTopologyManager.__init__` (method) `01_topobrain_cpu_v4.py:269` `def __init__(self, num_nodes, config)`
- `StableTopologyManager.get_adjacency` (method) `01_topobrain_cpu_v4.py:288` `def get_adjacency(self, plasticity)` -- Obtener matriz de adyacencia con estabilidad garantizada
- `StableTopologyManager.prune_topology` (method) `01_topobrain_cpu_v4.py:317` `def prune_topology(self, current_density, epoch)` -- Poda controlada con protocolo de emergencia
- `StableTopologyManager.get_density` (method) `01_topobrain_cpu_v4.py:337` `def get_density(self)` -- Calcular densidad actual de manera estable
- `StableTopoBrain.__init__` (method) `01_topobrain_cpu_v4.py:347` `def __init__(self, config)`
- `StableTopoBrain.forward` (method) `01_topobrain_cpu_v4.py:408` `def forward(self, x, controls)`
- `StableTopoBrain.stable_pgd_attack` (method) `01_topobrain_cpu_v4.py:475` `def stable_pgd_attack(model, x, y, eps, steps, controls)` -- Ataque PGD estable con manejo de gradientes robusto
- `StableTopoBrain.train_epoch` (method) `01_topobrain_cpu_v4.py:518` `def train_epoch(model, loader, optimizer, config, epoch, controls)` -- Entrenamiento por época con monitoreo detallado
- `StableTopoBrain.evaluate_model` (method) `01_topobrain_cpu_v4.py:583` `def evaluate_model(model, loader, config, adversarial, controls)` -- Evaluación rigurosa con o sin ataques adversariales
- `StableTopoBrain.run_scientific_ablation` (method) `01_topobrain_cpu_v4.py:628` `def run_scientific_ablation()` -- Ejecución científica del ablation con control de variables

## 01_topobrain_cpu_v5.py
- `Config.to_dict` (method) `01_topobrain_cpu_v5.py:56` `def to_dict(self)`
- `Config.get_topology_config` (method) `01_topobrain_cpu_v5.py:59` `def get_topology_config(self)` -- Configuración estable para topología adaptable
- `Config.seed_everything` (method) `01_topobrain_cpu_v5.py:70` `def seed_everything(seed)`
- `Config.get_tabular_loaders` (method) `01_topobrain_cpu_v5.py:77` `def get_tabular_loaders(config)` -- Dataset tabular controlado con características NOIR simuladas
- `StableSupConLoss.__init__` (method) `01_topobrain_cpu_v5.py:118` `def __init__(self, temperature)`
- `StableSupConLoss.forward` (method) `01_topobrain_cpu_v5.py:123` `def forward(self, features, labels)`
- `StableContinuumMemoryCell.__init__` (method) `01_topobrain_cpu_v5.py:149` `def __init__(self, input_dim, hidden_dim)`
- `StableContinuumMemoryCell.forward` (method) `01_topobrain_cpu_v5.py:181` `def forward(self, x, controls)`
- `StableSymbioticBasisRefinement.__init__` (method) `01_topobrain_cpu_v5.py:233` `def __init__(self, dim, num_atoms)`
- `StableSymbioticBasisRefinement.forward` (method) `01_topobrain_cpu_v5.py:242` `def forward(self, x)`
- `StableTopologyManager.__init__` (method) `01_topobrain_cpu_v5.py:269` `def __init__(self, num_nodes, config)`
- `StableTopologyManager.get_adjacency` (method) `01_topobrain_cpu_v5.py:288` `def get_adjacency(self, plasticity)` -- Obtener matriz de adyacencia con estabilidad garantizada
- `StableTopologyManager.prune_topology` (method) `01_topobrain_cpu_v5.py:317` `def prune_topology(self, current_density, epoch)` -- Poda controlada con protocolo de emergencia
- `StableTopologyManager.get_density` (method) `01_topobrain_cpu_v5.py:337` `def get_density(self)` -- Calcular densidad actual de manera estable
- `StableTopoBrain.__init__` (method) `01_topobrain_cpu_v5.py:347` `def __init__(self, config)`
- `StableTopoBrain.forward` (method) `01_topobrain_cpu_v5.py:408` `def forward(self, x, controls)`
- `StableTopoBrain.stable_pgd_attack` (method) `01_topobrain_cpu_v5.py:475` `def stable_pgd_attack(model, x, y, eps, steps, controls)` -- Ataque PGD estable con manejo de gradientes robusto
- `StableTopoBrain.train_epoch` (method) `01_topobrain_cpu_v5.py:518` `def train_epoch(model, loader, optimizer, config, epoch, controls)` -- Entrenamiento por época con monitoreo detallado
- `StableTopoBrain.evaluate_model` (method) `01_topobrain_cpu_v5.py:583` `def evaluate_model(model, loader, config, adversarial, controls)` -- Evaluación rigurosa con o sin ataques adversariales
- `StableTopoBrain.run_scientific_ablation` (method) `01_topobrain_cpu_v5.py:708` `def run_scientific_ablation()` -- Ejecución científica del ablation con control de variables

## 01_topobrain_cpu_v6.py
- `MicroConfig.to_dict` (method) `01_topobrain_cpu_v6.py:79` `def to_dict(self)`
- `MicroConfig.component_signature` (method) `01_topobrain_cpu_v6.py:82` `def component_signature(self)` -- Firma única de componentes activos
- `MicroConfig.seed_everything` (method) `01_topobrain_cpu_v6.py:95` `def seed_everything(seed)` -- Control de reproducibilidad
- `MicroConfig.get_micro_dataset` (method) `01_topobrain_cpu_v6.py:104` `def get_micro_dataset(config)` -- Dataset tabular controlado con validación cruzada
- `MicroConfig.compute_effect_size` (method) `01_topobrain_cpu_v6.py:126` `def compute_effect_size(group1, group2)` -- Cohen's d para medir tamaño del efecto
- `MicroSupConLoss.__init__` (method) `01_topobrain_cpu_v6.py:139` `def __init__(self, temperature)`
- `MicroSupConLoss.forward` (method) `01_topobrain_cpu_v6.py:144` `def forward(self, features, labels)`
- `MicroContinuumCell.__init__` (method) `01_topobrain_cpu_v6.py:168` `def __init__(self, dim)`
- `MicroContinuumCell.forward` (method) `01_topobrain_cpu_v6.py:184` `def forward(self, x, plasticity)`
- `MicroSymbioticBasis.__init__` (method) `01_topobrain_cpu_v6.py:219` `def __init__(self, dim, num_atoms)`
- `MicroSymbioticBasis.forward` (method) `01_topobrain_cpu_v6.py:232` `def forward(self, x)`
- `MicroTopology.__init__` (method) `01_topobrain_cpu_v6.py:258` `def __init__(self, num_nodes, config)`
- `MicroTopology.get_adjacency` (method) `01_topobrain_cpu_v6.py:276` `def get_adjacency(self, plasticity)` -- Matriz de adyacencia normalizada
- `MicroTopoBrain.__init__` (method) `01_topobrain_cpu_v6.py:301` `def __init__(self, config)`
- `MicroTopoBrain.count_parameters` (method) `01_topobrain_cpu_v6.py:360` `def count_parameters(self)` -- Contar parámetros entrenables
- `MicroTopoBrain.forward` (method) `01_topobrain_cpu_v6.py:364` `def forward(self, x, plasticity)`
- `MicroTopoBrain.micro_pgd_attack` (method) `01_topobrain_cpu_v6.py:432` `def micro_pgd_attack(model, x, y, eps, steps, plasticity)` -- PGD ultra-eficiente para CPU
- `MicroTopoBrain.train_epoch_micro` (method) `01_topobrain_cpu_v6.py:462` `def train_epoch_micro(model, loader, optimizer, config, epoch)` -- Entrenamiento por época
- `MicroTopoBrain.evaluate_micro` (method) `01_topobrain_cpu_v6.py:517` `def evaluate_micro(model, loader, config, adversarial)` -- Evaluación con opción adversarial
- `MicroTopoBrain.train_with_cv` (method) `01_topobrain_cpu_v6.py:539` `def train_with_cv(config, dataset, cv_folds)` -- Entrenamiento con validación cruzada estratificada.
- `AblationMatrix.level1_isolated` (method) `01_topobrain_cpu_v6.py:628` `def level1_isolated()` -- NIVEL 1: Componentes aislados (6 experimentos)
- `AblationMatrix.level2a_pairs` (method) `01_topobrain_cpu_v6.py:641` `def level2a_pairs()` -- NIVEL 2A: Todos los pares (10 experimentos = C(5,2))
- `AblationMatrix.level2b_strategic_triads` (method) `01_topobrain_cpu_v6.py:665` `def level2b_strategic_triads()` -- NIVEL 2B: Tríadas estratégicas (8 experimentos selectos)
- `AblationMatrix.level3_inverse_ablation` (method) `01_topobrain_cpu_v6.py:694` `def level3_inverse_ablation()` -- NIVEL 3: Ablación inversa (5 experimentos) Modelo completo MENOS un componente → detecta criticidad
- `AblationMatrix.level3_full_model` (method) `01_topobrain_cpu_v6.py:719` `def level3_full_model()` -- Modelo completo (referencia máxima)
- `AblationMatrix.get_complete_matrix` (method) `01_topobrain_cpu_v6.py:730` `def get_complete_matrix(cls)` -- Matriz completa de ablación (30 experimentos)
- `ScientificAnalyzer.compute_statistics` (method) `01_topobrain_cpu_v6.py:748` `def compute_statistics(results_list)` -- Análisis estadístico por experimento.
- `ScientificAnalyzer.ttest_vs_baseline` (method) `01_topobrain_cpu_v6.py:774` `def ttest_vs_baseline(exp_scores, baseline_scores)` -- t-test pareado vs baseline.
- `ScientificAnalyzer.detect_synergy` (method) `01_topobrain_cpu_v6.py:786` `def detect_synergy(pair_pgd, comp_a_pgd, comp_b_pgd, baseline_pgd)` -- Detecta sinergia no-lineal.
- `ScientificAnalyzer.rank_components_by_criticality` (method) `01_topobrain_cpu_v6.py:809` `def rank_components_by_criticality(full_pgd, ablation_results)` -- Ranking de criticidad basado en ablación inversa.
- `ScientificAnalyzer.run_scientific_ablation_study` (method) `01_topobrain_cpu_v6.py:850` `def run_scientific_ablation_study()` -- Ejecutor completo del estudio de ablación con análisis científico.

## 01_topobrain_cpu_v7.py
- `setup_device` (function) `01_topobrain_cpu_v7.py:24` `def setup_device()`
- `Config.to_dict` (method) `01_topobrain_cpu_v7.py:86` `def to_dict(self)`
- `SymbioticBasis.__init__` (method) `01_topobrain_cpu_v7.py:95` `def __init__(self, dim, num_atoms)`
- `SymbioticBasis.forward` (method) `01_topobrain_cpu_v7.py:106` `def forward(self, x)`
- `DynamicTopology.__init__` (method) `01_topobrain_cpu_v7.py:125` `def __init__(self, num_nodes, grid_size, config)`
- `DynamicTopology.get_adjacency` (method) `01_topobrain_cpu_v7.py:153` `def get_adjacency(self, plasticity)`
- `DynamicTopology.prune_connections` (method) `01_topobrain_cpu_v7.py:158` `def prune_connections(self, threshold)`
- `DynamicTopology.get_density` (method) `01_topobrain_cpu_v7.py:178` `def get_density(self)`
- `TopoBrainCPU.__init__` (method) `01_topobrain_cpu_v7.py:187` `def __init__(self, config)`
- `TopoBrainCPU.count_parameters` (method) `01_topobrain_cpu_v7.py:223` `def count_parameters(self)`
- `TopoBrainCPU.forward` (method) `01_topobrain_cpu_v7.py:226` `def forward(self, x, plasticity)`
- `TopoBrainCPU.pgd_attack` (method) `01_topobrain_cpu_v7.py:256` `def pgd_attack(model, x, y, eps, steps, plasticity)`
- `TopoBrainCPU.train_epoch` (method) `01_topobrain_cpu_v7.py:285` `def train_epoch(model, loader, optimizer, config, epoch, device)`
- `TopoBrainCPU.evaluate` (method) `01_topobrain_cpu_v7.py:321` `def evaluate(model, loader, config, device, adversarial)`
- `TopoBrainCPU.get_dataset` (method) `01_topobrain_cpu_v7.py:351` `def get_dataset(config)`
- `TopoBrainCPU.main` (method) `01_topobrain_cpu_v7.py:393` `def main()`

## 01_topobrain_cpu_v8.py
- `SymbioticBasis.__init__` (method) `01_topobrain_cpu_v8.py:63` `def __init__(self, dim, num_atoms)`
- `SymbioticBasis.forward` (method) `01_topobrain_cpu_v8.py:76` `def forward(self, x)`
- `DynamicTopology.__init__` (method) `01_topobrain_cpu_v8.py:89` `def __init__(self, num_nodes, grid_size, config)`
- `DynamicTopology.get_adjacency` (method) `01_topobrain_cpu_v8.py:115` `def get_adjacency(self, plasticity)`
- `DynamicTopology.prune_connections` (method) `01_topobrain_cpu_v8.py:120` `def prune_connections(self, threshold)`
- `DynamicTopology.get_density` (method) `01_topobrain_cpu_v8.py:140` `def get_density(self)`
- `TopoBrainReal.__init__` (method) `01_topobrain_cpu_v8.py:145` `def __init__(self, config)`
- `TopoBrainReal.forward` (method) `01_topobrain_cpu_v8.py:181` `def forward(self, x, plasticity)`
- `TopoBrainReal.forward_with_metrics` (method) `01_topobrain_cpu_v8.py:210` `def forward_with_metrics(self, x, plasticity)`
- `TopoBrainReal.pgd_attack` (method) `01_topobrain_cpu_v8.py:223` `def pgd_attack(model, x, y, eps, steps, plasticity)`
- `TopoBrainReal.train_topobrain` (method) `01_topobrain_cpu_v8.py:252` `def train_topobrain(config)`
- `TopoBrainReal.export_for_onnxruntime` (method) `01_topobrain_cpu_v8.py:410` `def export_for_onnxruntime(model)` -- Exporta para ONNX Runtime (más moderno que OpenCV)
- `Wrapper.__init__` (method) `01_topobrain_cpu_v8.py:425` `def __init__(self, m)`
- `Wrapper.forward` (method) `01_topobrain_cpu_v8.py:429` `def forward(self, x)`
- `Wrapper.test_with_onnxruntime` (method) `01_topobrain_cpu_v8.py:465` `def test_with_onnxruntime(X_test, y_test)` -- Inferencia usando ONNX Runtime
- `Wrapper.main` (method) `01_topobrain_cpu_v8.py:547` `def main()`

## 01_topobrain_ganador_gpu_v1.py
- `setup_amd_device` (function) `01_topobrain_ganador_gpu_v1.py:31` `def setup_amd_device()` -- Configura PyTorch para usar GPU AMD con ROCm/OpenCL.
- `GPUConfig.to_dict` (method) `01_topobrain_ganador_gpu_v1.py:119` `def to_dict(self)`
- `GPUSymbioticBasis.__init__` (method) `01_topobrain_ganador_gpu_v1.py:132` `def __init__(self, dim, num_atoms)`
- `GPUSymbioticBasis.forward` (method) `01_topobrain_ganador_gpu_v1.py:147` `def forward(self, x)` -- Args: x: [batch, dim] Returns: x_clean: [batch, dim] - Proyección limpia entropy: scalar - Entropía de pesos ortho...
- `DynamicTopology.__init__` (method) `01_topobrain_ganador_gpu_v1.py:184` `def __init__(self, num_nodes, grid_size, config)`
- `DynamicTopology.get_adjacency` (method) `01_topobrain_ganador_gpu_v1.py:219` `def get_adjacency(self, plasticity)` -- Obtiene matriz de adyacencia normalizada.
- `DynamicTopology.prune_connections` (method) `01_topobrain_ganador_gpu_v1.py:237` `def prune_connections(self, threshold)` -- Poda conexiones débiles (llamar cada N epochs).
- `DynamicTopology.get_density` (method) `01_topobrain_ganador_gpu_v1.py:269` `def get_density(self)` -- Densidad actual de conexiones
- `TopoBrainGPU.__init__` (method) `01_topobrain_ganador_gpu_v1.py:291` `def __init__(self, config)`
- `TopoBrainGPU.count_parameters` (method) `01_topobrain_ganador_gpu_v1.py:343` `def count_parameters(self)` -- Cuenta parámetros entrenables
- `TopoBrainGPU.forward` (method) `01_topobrain_ganador_gpu_v1.py:347` `def forward(self, x, plasticity)` -- Forward pass completo.
- `TopoBrainGPU.pgd_attack_gpu` (method) `01_topobrain_ganador_gpu_v1.py:405` `def pgd_attack_gpu(model, x, y, eps, steps, plasticity)` -- PGD attack optimizado para GPU.
- `TopoBrainGPU.train_epoch_gpu` (method) `01_topobrain_ganador_gpu_v1.py:461` `def train_epoch_gpu(model, loader, optimizer, config, epoch, device)` -- Entrenamiento por época en GPU
- `TopoBrainGPU.evaluate_gpu` (method) `01_topobrain_ganador_gpu_v1.py:517` `def evaluate_gpu(model, loader, config, device, adversarial)` -- Evaluación en GPU
- `TopoBrainGPU.get_gpu_dataset` (method) `01_topobrain_ganador_gpu_v1.py:547` `def get_gpu_dataset(config)` -- Dataset sintético para GPU
- `TopoBrainGPU.main` (method) `01_topobrain_ganador_gpu_v1.py:591` `def main()` -- Ejecutar POC completa

## 02_perceptron.py
- `Perceptron.__init__` (method) `02_perceptron.py:32` `def __init__(self, input_dim, learning_rate)`
- `Perceptron.predict` (method) `02_perceptron.py:37` `def predict(self, X)`
- `Perceptron.train_step` (method) `02_perceptron.py:42` `def train_step(self, X_batch, y_batch)`
- `Perceptron.accuracy` (method) `02_perceptron.py:51` `def accuracy(self, X, y_true)`

## 03_backpropagation.py
- `sigmoid` (function) `03_backpropagation.py:33` `def sigmoid(z)`
- `sigmoid_derivative` (function) `03_backpropagation.py:38` `def sigmoid_derivative(z)`
- `forward` (function) `03_backpropagation.py:53` `def forward(X)`
- `backward` (function) `03_backpropagation.py:63` `def backward(X, y_true, y_pred, a1, z1, lr)`
- `compute_metrics` (function) `03_backpropagation.py:88` `def compute_metrics(y_pred, y_true)`

## 04_cnn_lenet.py
- `LeNet5Like.__init__` (method) `04_cnn_lenet.py:46` `def __init__(self)`
- `LeNet5Like.forward` (method) `04_cnn_lenet.py:60` `def forward(self, x)`

## 06_lstm_char.py
- `create_batches` (function) `06_lstm_char.py:63` `def create_batches(data, batch_size, seq_length)`
- `CharLSTM.__init__` (method) `06_lstm_char.py:90` `def __init__(self, vocab_size, hidden_size, num_layers, dropout)`
- `CharLSTM.forward` (method) `06_lstm_char.py:102` `def forward(self, x, hidden)`

## 08_vae_mnist.py
- `VAE.__init__` (method) `08_vae_mnist.py:41` `def __init__(self, input_dim, hidden_dim, latent_dim)`
- `VAE.encode` (method) `08_vae_mnist.py:51` `def encode(self, x)`
- `VAE.reparameterize` (method) `08_vae_mnist.py:55` `def reparameterize(self, mu, log_var)`
- `VAE.decode` (method) `08_vae_mnist.py:60` `def decode(self, z)`
- `VAE.forward` (method) `08_vae_mnist.py:64` `def forward(self, x)`
- `VAE.vae_loss` (method) `08_vae_mnist.py:72` `def vae_loss(recon_x, x, mu, log_var)`

## 09_transformer_mini.py
- `generate_copy_data` (function) `09_transformer_mini.py:29` `def generate_copy_data(num_samples, seq_len, vocab_size)`
- `MultiHeadAttention.__init__` (method) `09_transformer_mini.py:52` `def __init__(self, d_model, num_heads, dropout)`
- `MultiHeadAttention.forward` (method) `09_transformer_mini.py:63` `def forward(self, q, k, v, mask)`
- `FeedForward.__init__` (method) `09_transformer_mini.py:84` `def __init__(self, d_model, d_ff, dropout)`
- `FeedForward.forward` (method) `09_transformer_mini.py:90` `def forward(self, x)`
- `MiniTransformerEncoder.__init__` (method) `09_transformer_mini.py:97` `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len)`
- `MiniTransformerEncoder.forward` (method) `09_transformer_mini.py:115` `def forward(self, x)`
- `MiniTransformer.__init__` (method) `09_transformer_mini.py:132` `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout)`
- `MiniTransformer.forward` (method) `09_transformer_mini.py:137` `def forward(self, x)`

## 10_gan_mnist_lite.py
- `Generator.__init__` (method) `10_gan_mnist_lite.py:38` `def __init__(self, latent_dim, img_size)`
- `Generator.forward` (method) `10_gan_mnist_lite.py:51` `def forward(self, z)`
- `Discriminator.__init__` (method) `10_gan_mnist_lite.py:58` `def __init__(self, img_size)`
- `Discriminator.forward` (method) `10_gan_mnist_lite.py:74` `def forward(self, x)`

## 11_bert_tiny.py
- `tokenize_sentence` (function) `11_bert_tiny.py:76` `def tokenize_sentence(sentence)`
- `pad_sequence` (function) `11_bert_tiny.py:88` `def pad_sequence(seq, length, pad_value)`
- `MultiHeadAttention.__init__` (method) `11_bert_tiny.py:108` `def __init__(self, d_model, num_heads, dropout)`
- `MultiHeadAttention.forward` (method) `11_bert_tiny.py:119` `def forward(self, q, k, v, mask)`
- `FeedForward.__init__` (method) `11_bert_tiny.py:134` `def __init__(self, d_model, d_ff, dropout)`
- `FeedForward.forward` (method) `11_bert_tiny.py:140` `def forward(self, x)`
- `BERTLayer.__init__` (method) `11_bert_tiny.py:144` `def __init__(self, d_model, num_heads, d_ff, dropout)`
- `BERTLayer.forward` (method) `11_bert_tiny.py:151` `def forward(self, x, mask)`
- `TinyBERT.__init__` (method) `11_bert_tiny.py:159` `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len)`
- `TinyBERT.forward` (method) `11_bert_tiny.py:175` `def forward(self, x, mask)`
- `TinyBERT.mask_tokens` (method) `11_bert_tiny.py:186` `def mask_tokens(inputs, vocab_size, mask_token_id, pad_token_id, mask_prob)`

## 12_diffusion_minimal.py
- `SimpleDiffusionNet.__init__` (method) `12_diffusion_minimal.py:55` `def __init__(self, in_channels, out_channels, hidden_dim)`
- `SimpleDiffusionNet.forward` (method) `12_diffusion_minimal.py:65` `def forward(self, x, t)`
- `SimpleDiffusionNet.q_sample` (method) `12_diffusion_minimal.py:81` `def q_sample(x_0, t, noise)` -- Muestrea x_t dado x_0 y timestep t.

## 13_nested_hope.py
- `Config.setup_device` (method) `13_nested_hope.py:44` `def setup_device()` -- Configuración automática de dispositivo
- `Config.set_seed` (method) `13_nested_hope.py:54` `def set_seed(seed)` -- Reproducibilidad completa
- `DeltaGradientDescent.apply_update` (method) `13_nested_hope.py:77` `def apply_update(grad, param, x_normalized, eta, alpha, lambda_norm)` -- Aplica la regla DGD a un gradiente
- `SelfModifyingMemory.__init__` (method) `13_nested_hope.py:125` `def __init__(self, d_model, hidden_dim, chunk_size)`
- `SelfModifyingMemory.forward` (method) `13_nested_hope.py:170` `def forward(self, x, prev_states)` -- Forward pass con actualización chunk-wise (Sección 8.2)
- `ContinuumMemorySystem.__init__` (method) `13_nested_hope.py:268` `def __init__(self, frequencies, d_model, hidden_dim, connection_type)`
- `ContinuumMemorySystem.forward` (method) `13_nested_hope.py:296` `def forward(self, x, global_step)` -- Forward pass con actualizaciones multi-frecuencia
- `HopeModel.__init__` (method) `13_nested_hope.py:344` `def __init__(self, vocab_size, d_model, cms_frequencies, mlp_hidden, chunk_size, enable_self_modifying, enable_cms)`
- `HopeModel.reset_states` (method) `13_nested_hope.py:390` `def reset_states(self)` -- Reset de estados internos (para nuevas secuencias)
- `HopeModel.forward` (method) `13_nested_hope.py:394` `def forward(self, x, global_step, return_internals)` -- Forward pass completo
- `HopeTrainer.__init__` (method) `13_nested_hope.py:437` `def __init__(self, model, config, device)`
- `HopeTrainer.train_epoch` (method) `13_nested_hope.py:461` `def train_epoch(self, train_loader, epoch, global_step)` -- Entrena una época completa
- `HopeTrainer.evaluate` (method) `13_nested_hope.py:536` `def evaluate(self, test_loader, global_step)` -- Evaluación sin gradientes
- `HopeTrainer.run_ablation_study` (method) `13_nested_hope.py:564` `def run_ablation_study(config, device)` -- Ejecuta estudio de ablación completo

## 13_nested_kearning_gpu.py
- `SelfModifyingMemory.__init__` (method) `13_nested_kearning_gpu.py:54` `def __init__(self, vocab_size, d_model, hidden_dim)`
- `SelfModifyingMemory.forward` (method) `13_nested_kearning_gpu.py:64` `def forward(self, x)`
- `ContinuumMemorySystem.__init__` (method) `13_nested_kearning_gpu.py:75` `def __init__(self, frequencies, d_model, hidden_dim)`
- `ContinuumMemorySystem.forward` (method) `13_nested_kearning_gpu.py:87` `def forward(self, x, global_step)`
- `HopeModel.__init__` (method) `13_nested_kearning_gpu.py:98` `def __init__(self, vocab_size, d_model, cms_freqs, hidden_dim)`
- `HopeModel.forward` (method) `13_nested_kearning_gpu.py:104` `def forward(self, x, global_step)`

## 13_nested_learning.py
- `SelfModifyingMemory.__init__` (method) `13_nested_learning.py:52` `def __init__(self, vocab_size, d_model, hidden_dim)`
- `SelfModifyingMemory.forward` (method) `13_nested_learning.py:62` `def forward(self, x, update_mask)` -- update_mask: None o máscara booleana para actualizaciones condicionales Retorna logits y parámetros internos (para...
- `ContinuumMemorySystem.__init__` (method) `13_nested_learning.py:87` `def __init__(self, levels, d_model, hidden_dim)`
- `ContinuumMemorySystem.forward` (method) `13_nested_learning.py:100` `def forward(self, x, global_step)` -- global_step: int, paso global de entrenamiento
- `HopeModel.__init__` (method) `13_nested_learning.py:115` `def __init__(self, vocab_size, d_model, cms_levels, mlp_hidden)`
- `HopeModel.forward` (method) `13_nested_learning.py:122` `def forward(self, x, global_step)`

## NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py
- `compute_loss` (function) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:20` `def compute_loss(logits, captions, gate, vocab)`
- `LanguageMetrics.sentence_bleu` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:48` `def sentence_bleu(reference, hypothesis, weights)` -- BLEU simplificado a nivel de oración
- `LanguageMetrics.token_accuracy` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:91` `def token_accuracy(reference, hypothesis)` -- Porcentaje de tokens correctos en posición
- `LanguageMetrics.word_overlap` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:104` `def word_overlap(reference, hypothesis)` -- Jaccard similarity entre palabras
- `TriangulatedMedicalSystem.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:120` `def __init__(self)`
- `TriangulatedMedicalSystem.triangulate_signals` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:125` `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` -- Identificar señales convergentes que confirman problemas
- `TriangulatedMedicalSystem.count_convergent_signals` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:157` `def count_convergent_signals(self, signals, pattern)` -- Contar cuántas señales del patrón están activas
- `TriangulatedMedicalSystem.diagnose_with_triangulation` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:161` `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` -- Diagnosticar SOLO con confirmación múltiple
- `TriangulatedMedicalSystem.apply_triangulated_intervention` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:223` `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` -- Aplicar intervención SOLO si confianza es alta
- `StableLiquidNeuron.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:383` `def __init__(self, in_dim, out_dim)`
- `StableLiquidNeuron.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:397` `def forward(self, x)`
- `StableLiquidNeuron.hebbian_update` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:404` `def hebbian_update(self, post, pre, plasticity)`
- `StableLiquidNeuron.update_physiology_advanced` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:434` `def update_physiology_advanced(self, loss_value)`
- `RightHemisphere.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:459` `def __init__(self, output_dim)`
- `RightHemisphere.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:467` `def forward(self, image)`
- `LeftHemisphere.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:474` `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `LeftHemisphere.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:502` `def forward(self, visual_context, captions, max_len)`
- `CorpusCallosum.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:545` `def __init__(self, dim)`
- `CorpusCallosum.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:568` `def forward(self, right_features)`
- `NeuroLogosBicameralStable.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:582` `def __init__(self, vocab_size)`
- `NeuroLogosBicameralStable.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:588` `def forward(self, image, captions)`
- `EnhancedDiagnostics.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:603` `def __init__(self)`
- `EnhancedDiagnostics.measure_callosal_flow` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:613` `def measure_callosal_flow(self, right_features, left_context)`
- `EnhancedDiagnostics.calculate_synergy` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:622` `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `EnhancedDiagnostics.calculate_health` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:631` `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `EnhancedDiagnostics.update` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:640` `def update(self)`
- `EnhancedDiagnostics.get_recent_avg` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:645` `def get_recent_avg(self, key, n)`
- `EnhancedDiagnostics.report` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:650` `def report(self, epoch)`
- `Flickr8kDataset.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:728` `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `Flickr8kDataset.build_vocab_flickr` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:764` `def build_vocab_flickr(captions_file, vocab_size)`
- `Flickr8kDataset.setup_flickr8k` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:782` `def setup_flickr8k(data_dir)`
- `Flickr8kDataset.train_with_metrics` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:854` `def train_with_metrics()`

## NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py
- `compute_loss` (function) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:20` `def compute_loss(logits, captions, gate, vocab)`
- `LanguageMetrics.sentence_bleu` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:48` `def sentence_bleu(reference, hypothesis, weights)` -- BLEU simplificado a nivel de oración
- `LanguageMetrics.token_accuracy` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:91` `def token_accuracy(reference, hypothesis)` -- Porcentaje de tokens correctos en posición
- `LanguageMetrics.word_overlap` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:104` `def word_overlap(reference, hypothesis)` -- Jaccard similarity entre palabras
- `TriangulatedMedicalSystem.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:120` `def __init__(self)`
- `TriangulatedMedicalSystem.triangulate_signals` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:125` `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` -- Identificar señales convergentes que confirman problemas
- `TriangulatedMedicalSystem.count_convergent_signals` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:157` `def count_convergent_signals(self, signals, pattern)` -- Contar cuántas señales del patrón están activas
- `TriangulatedMedicalSystem.diagnose_with_triangulation` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:161` `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` -- Diagnosticar SOLO con confirmación múltiple
- `TriangulatedMedicalSystem.apply_triangulated_intervention` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:224` `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` -- Aplicar intervención SOLO si confianza es alta
- `StableLiquidNeuron.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:384` `def __init__(self, in_dim, out_dim)`
- `StableLiquidNeuron.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:422` `def forward(self, x)`
- `StableLiquidNeuron.hebbian_update` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:444` `def hebbian_update(self, post, pre, plasticity)`
- `StableLiquidNeuron.update_physiology_advanced` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:483` `def update_physiology_advanced(self, loss_value)`
- `RightHemisphere.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:510` `def __init__(self, output_dim)`
- `RightHemisphere.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:518` `def forward(self, image)`
- `LeftHemisphere.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:525` `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `LeftHemisphere.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:573` `def forward(self, visual_context, captions, max_len)`
- `CorpusCallosum.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:646` `def __init__(self, dim)`
- `CorpusCallosum.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:678` `def forward(self, right_features)`
- `NeuroLogosBicameralStable.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:701` `def __init__(self, vocab_size)`
- `NeuroLogosBicameralStable.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:707` `def forward(self, image, captions)`
- `EnhancedDiagnostics.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:722` `def __init__(self)`
- `EnhancedDiagnostics.measure_callosal_flow` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:732` `def measure_callosal_flow(self, right_features, left_context)`
- `EnhancedDiagnostics.calculate_synergy` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:741` `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `EnhancedDiagnostics.calculate_health` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:750` `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `EnhancedDiagnostics.update` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:759` `def update(self)`
- `EnhancedDiagnostics.get_recent_avg` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:764` `def get_recent_avg(self, key, n)`
- `EnhancedDiagnostics.report` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:769` `def report(self, epoch)`
- `EpisodicMemoryBuffer.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:845` `def __init__(self, capacity, surprise_threshold)`
- `EpisodicMemoryBuffer.compute_surprise` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:851` `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `EpisodicMemoryBuffer.add` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:862` `def add(self, image, caption, surprise_score)`
- `EpisodicMemoryBuffer.sample` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:872` `def sample(self, batch_size)`
- `Flickr8kDataset.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:892` `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `Flickr8kDataset.build_vocab_flickr` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:928` `def build_vocab_flickr(captions_file, vocab_size)`
- `Flickr8kDataset.setup_flickr8k` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:946` `def setup_flickr8k(data_dir)`
- `Flickr8kDataset.train_with_metrics` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:1021` `def train_with_metrics()`

## NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py
- `compute_loss` (function) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:20` `def compute_loss(logits, captions, gate, vocab)`
- `LanguageMetrics.sentence_bleu` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:48` `def sentence_bleu(reference, hypothesis, weights)` -- BLEU simplificado a nivel de oración
- `LanguageMetrics.token_accuracy` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:91` `def token_accuracy(reference, hypothesis)` -- Porcentaje de tokens correctos en posición
- `LanguageMetrics.word_overlap` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:104` `def word_overlap(reference, hypothesis)` -- Jaccard similarity entre palabras
- `TriangulatedMedicalSystem.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:120` `def __init__(self)`
- `TriangulatedMedicalSystem.triangulate_signals` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:125` `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` -- Identificar señales convergentes que confirman problemas
- `TriangulatedMedicalSystem.count_convergent_signals` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:157` `def count_convergent_signals(self, signals, pattern)` -- Contar cuántas señales del patrón están activas
- `TriangulatedMedicalSystem.diagnose_with_triangulation` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:161` `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` -- Diagnosticar SOLO con confirmación múltiple
- `TriangulatedMedicalSystem.apply_triangulated_intervention` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:224` `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` -- Aplicar intervención SOLO si confianza es alta
- `StableLiquidNeuron.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:393` `def __init__(self, in_dim, out_dim)`
- `StableLiquidNeuron.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:431` `def forward(self, x)`
- `StableLiquidNeuron.hebbian_update` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:453` `def hebbian_update(self, post, pre, plasticity)`
- `StableLiquidNeuron.update_physiology_advanced` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:492` `def update_physiology_advanced(self, loss_value)`
- `RightHemisphere.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:519` `def __init__(self, output_dim)`
- `RightHemisphere.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:527` `def forward(self, image)`
- `LeftHemisphere.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:534` `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `LeftHemisphere.beam_search_decode` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:584` `def beam_search_decode(self, visual_context, beam_width, max_len, epoch)`
- `LeftHemisphere.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:657` `def forward(self, visual_context, captions, max_len, epoch)`
- `CorpusCallosum.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:710` `def __init__(self, dim)`
- `CorpusCallosum.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:742` `def forward(self, right_features)`
- `NeuroLogosBicameralStable.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:766` `def __init__(self, vocab_size)`
- `NeuroLogosBicameralStable.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:772` `def forward(self, image, captions, epoch)`
- `EnhancedDiagnostics.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:787` `def __init__(self)`
- `EnhancedDiagnostics.measure_callosal_flow` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:797` `def measure_callosal_flow(self, right_features, left_context)`
- `EnhancedDiagnostics.calculate_synergy` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:806` `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `EnhancedDiagnostics.calculate_health` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:815` `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `EnhancedDiagnostics.update` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:824` `def update(self)`
- `EnhancedDiagnostics.get_recent_avg` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:829` `def get_recent_avg(self, key, n)`
- `EnhancedDiagnostics.report` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:834` `def report(self, epoch)`
- `EpisodicMemoryBuffer.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:908` `def __init__(self, capacity, surprise_threshold)`
- `EpisodicMemoryBuffer.compute_surprise` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:914` `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `EpisodicMemoryBuffer.add` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:925` `def add(self, image, caption, surprise_score)`
- `EpisodicMemoryBuffer.sample` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:935` `def sample(self, batch_size)`
- `Flickr8kDataset.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:955` `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `Flickr8kDataset.build_vocab_flickr` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:991` `def build_vocab_flickr(captions_file, vocab_size)`
- `Flickr8kDataset.setup_flickr8k` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:1009` `def setup_flickr8k(data_dir)`
- `Flickr8kDataset.train_with_metrics` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:1081` `def train_with_metrics()`

## NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py
- `compute_loss` (function) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:20` `def compute_loss(logits, captions, gate, vocab, linguistic_reward, lambda_reward)` -- Función de pérdida extendida que incorpora recompensa lingüística
- `NeurocognitiveSystem.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:55` `def __init__(self)`
- `NeurocognitiveSystem.assess_cognitive_state` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:65` `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` -- Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas
- `NeurocognitiveSystem.apply_cognitive_intervention` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:113` `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch)` -- Aplica intervenciones cognitivas basadas en el estado lingüístico
- `LinguisticFeedbackLoop.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:223` `def __init__(self, alpha, beta)`
- `LinguisticFeedbackLoop.compute_linguistic_reward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:232` `def compute_linguistic_reward(self, references, hypotheses)` -- Calcula una recompensa combinada basada en CIDEr y SPICE que puede usarse para guiar el entrenamiento
- `LinguisticFeedbackLoop.compute_cider` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:254` `def compute_cider(self, reference, hypothesis)` -- Versión simplificada de CIDEr para uso en entrenamiento
- `LinguisticFeedbackLoop.compute_spice` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:281` `def compute_spice(self, reference, hypothesis)` -- Versión simplificada de SPICE para uso en entrenamiento
- `LanguageMetrics.sentence_bleu` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:313` `def sentence_bleu(reference, hypothesis, weights)` -- BLEU simplificado a nivel de oración
- `LanguageMetrics.token_accuracy` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:356` `def token_accuracy(reference, hypothesis)` -- Porcentaje de tokens correctos en posición
- `LanguageMetrics.word_overlap` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:369` `def word_overlap(reference, hypothesis)` -- Jaccard similarity entre palabras
- `TriangulatedMedicalSystem.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:385` `def __init__(self)`
- `TriangulatedMedicalSystem.triangulate_signals` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:390` `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` -- Identificar señales convergentes que confirman problemas
- `TriangulatedMedicalSystem.count_convergent_signals` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:422` `def count_convergent_signals(self, signals, pattern)` -- Contar cuántas señales del patrón están activas
- `TriangulatedMedicalSystem.diagnose_with_triangulation` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:426` `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` -- Diagnosticar SOLO con confirmación múltiple
- `TriangulatedMedicalSystem.apply_triangulated_intervention` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:489` `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` -- Aplicar intervención SOLO si confianza es alta
- `StableLiquidNeuron.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:658` `def __init__(self, in_dim, out_dim)`
- `StableLiquidNeuron.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:696` `def forward(self, x)`
- `StableLiquidNeuron.hebbian_update` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:718` `def hebbian_update(self, post, pre, plasticity)`
- `StableLiquidNeuron.update_physiology_advanced` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:757` `def update_physiology_advanced(self, loss_value)`
- `RightHemisphere.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:784` `def __init__(self, output_dim)`
- `RightHemisphere.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:792` `def forward(self, image)`
- `LeftHemisphere.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:799` `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `LeftHemisphere.beam_search_decode` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:849` `def beam_search_decode(self, visual_context, beam_width, max_len, epoch)`
- `LeftHemisphere.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:925` `def forward(self, visual_context, captions, max_len, epoch)`
- `CorpusCallosum.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:979` `def __init__(self, dim)`
- `CorpusCallosum.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1011` `def forward(self, right_features)`
- `NeuroLogosBicameralStable.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1035` `def __init__(self, vocab_size)`
- `NeuroLogosBicameralStable.forward` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1041` `def forward(self, image, captions, epoch)`
- `EnhancedDiagnostics.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1056` `def __init__(self)`
- `EnhancedDiagnostics.measure_callosal_flow` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1067` `def measure_callosal_flow(self, right_features, left_context)`
- `EnhancedDiagnostics.calculate_synergy` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1076` `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `EnhancedDiagnostics.calculate_health` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1085` `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `EnhancedDiagnostics.update` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1094` `def update(self)`
- `EnhancedDiagnostics.get_recent_avg` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1099` `def get_recent_avg(self, key, n)`
- `EnhancedDiagnostics.report` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1104` `def report(self, epoch)`
- `EpisodicMemoryBuffer.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1193` `def __init__(self, capacity, surprise_threshold)`
- `EpisodicMemoryBuffer.compute_surprise` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1199` `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `EpisodicMemoryBuffer.add` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1210` `def add(self, image, caption, surprise_score)`
- `EpisodicMemoryBuffer.sample` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1220` `def sample(self, batch_size)`
- `Flickr8kDataset.__init__` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1240` `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `Flickr8kDataset.build_vocab_flickr` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1276` `def build_vocab_flickr(captions_file, vocab_size)`
- `Flickr8kDataset.setup_flickr8k` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1294` `def setup_flickr8k(data_dir)`
- `Flickr8kDataset.train_with_metrics` (method) `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1366` `def train_with_metrics()`


Next: [API_p2.md](API_p2.md)

# API

## 01_mcculloch_pitts.py

### mcculloch_pitts_neuron `def mcculloch_pitts_neuron(inputs, weights, threshold)`
- Defined: `01_mcculloch_pitts.py:7`
- Doc: Neurona artificial de McCulloch-Pitts (1943).

## 01_topobrain_cou_v2.py

### seed_everything `def seed_everything(seed)`
- Defined: `01_topobrain_cou_v2.py:70`

### get_tabular_loaders `def get_tabular_loaders(config)`
- Defined: `01_topobrain_cou_v2.py:77`
- Doc: Dataset tabular controlado con características NOIR simuladas

### stable_pgd_attack `def stable_pgd_attack(model, x, y, eps, steps, controls)`
- Defined: `01_topobrain_cou_v2.py:477`
- Doc: Ataque PGD estable con manejo de gradientes robusto

### train_epoch `def train_epoch(model, loader, optimizer, config, epoch, controls)`
- Defined: `01_topobrain_cou_v2.py:520`
- Doc: Entrenamiento por época con monitoreo detallado

### evaluate_model `def evaluate_model(model, loader, config, adversarial, controls)`
- Defined: `01_topobrain_cou_v2.py:584`
- Doc: Evaluación rigurosa con o sin ataques adversariales

### run_scientific_ablation `def run_scientific_ablation()`
- Defined: `01_topobrain_cou_v2.py:632`
- Doc: Ejecución científica del ablation con control de variables

### to_dict `def to_dict(self)`
- Defined: `01_topobrain_cou_v2.py:56`

### get_topology_config `def get_topology_config(self)`
- Defined: `01_topobrain_cou_v2.py:59`
- Doc: Configuración estable para topología adaptable

### __init__ `def __init__(self, temperature)`
- Defined: `01_topobrain_cou_v2.py:118`

### forward `def forward(self, features, labels)`
- Defined: `01_topobrain_cou_v2.py:123`

### __init__ `def __init__(self, input_dim, hidden_dim)`
- Defined: `01_topobrain_cou_v2.py:149`

### forward `def forward(self, x, controls)`
- Defined: `01_topobrain_cou_v2.py:181`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `01_topobrain_cou_v2.py:233`

### forward `def forward(self, x)`
- Defined: `01_topobrain_cou_v2.py:242`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `01_topobrain_cou_v2.py:269`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `01_topobrain_cou_v2.py:288`
- Doc: Obtener matriz de adyacencia con estabilidad garantizada

### prune_topology `def prune_topology(self, current_density, epoch)`
- Defined: `01_topobrain_cou_v2.py:317`
- Doc: Poda controlada con protocolo de emergencia

### get_density `def get_density(self)`
- Defined: `01_topobrain_cou_v2.py:337`
- Doc: Calcular densidad actual de manera estable

### __init__ `def __init__(self, config)`
- Defined: `01_topobrain_cou_v2.py:347`

### _init_weights `def _init_weights(self)`
- Defined: `01_topobrain_cou_v2.py:400`
- Doc: Inicialización estable de pesos

### forward `def forward(self, x, controls)`
- Defined: `01_topobrain_cou_v2.py:408`

## 01_topobrain_cpu.py

### seed_everything `def seed_everything(seed)`
- Defined: `01_topobrain_cpu.py:63`

### get_tabular_loaders `def get_tabular_loaders(config)`
- Defined: `01_topobrain_cpu.py:69`

### pgd_attack `def pgd_attack(model, x, y, eps, steps, controls)`
- Defined: `01_topobrain_cpu.py:247`

### generate_ablation_configs `def generate_ablation_configs(base_config)`
- Defined: `01_topobrain_cpu.py:263`

### train_and_evaluate `def train_and_evaluate(config, name)`
- Defined: `01_topobrain_cpu.py:293`

### run_ablation `def run_ablation()`
- Defined: `01_topobrain_cpu.py:337`

### to_dict `def to_dict(self)`
- Defined: `01_topobrain_cpu.py:57`

### __init__ `def __init__(self, temperature)`
- Defined: `01_topobrain_cpu.py:95`

### forward `def forward(self, features, labels)`
- Defined: `01_topobrain_cpu.py:98`

### __init__ `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate)`
- Defined: `01_topobrain_cpu.py:110`

### forward `def forward(self, x, controls)`
- Defined: `01_topobrain_cpu.py:124`

### __init__ `def __init__(self, dim)`
- Defined: `01_topobrain_cpu.py:144`

### forward `def forward(self, x)`
- Defined: `01_topobrain_cpu.py:151`

### __init__ `def __init__(self, config)`
- Defined: `01_topobrain_cpu.py:167`

### get_adj `def get_adj(self)`
- Defined: `01_topobrain_cpu.py:198`

### forward `def forward(self, x, controls)`
- Defined: `01_topobrain_cpu.py:203`

### evaluate_adv `def evaluate_adv(loader, eps, steps)`
- Defined: `01_topobrain_cpu.py:323`

## 01_topobrain_cpu_v3.py

### seed_everything `def seed_everything(seed)`
- Defined: `01_topobrain_cpu_v3.py:63`

### get_tabular_loaders `def get_tabular_loaders(config)`
- Defined: `01_topobrain_cpu_v3.py:69`

### pgd_attack `def pgd_attack(model, x, y, eps, steps)`
- Defined: `01_topobrain_cpu_v3.py:395`

### compute_topology_metrics `def compute_topology_metrics(model, config)`
- Defined: `01_topobrain_cpu_v3.py:416`
- Doc: Computar métricas de topología con manejo robusto de errores

### prune_topology `def prune_topology(model, config, controls)`
- Defined: `01_topobrain_cpu_v3.py:450`
- Doc: Implementación simplificada de poda de topología

### train_and_evaluate `def train_and_evaluate(config, run_name)`
- Defined: `01_topobrain_cpu_v3.py:482`

### run_ablation `def run_ablation()`
- Defined: `01_topobrain_cpu_v3.py:640`

### to_dict `def to_dict(self)`
- Defined: `01_topobrain_cpu_v3.py:57`

### __init__ `def __init__(self, temperature)`
- Defined: `01_topobrain_cpu_v3.py:94`

### forward `def forward(self, features, labels)`
- Defined: `01_topobrain_cpu_v3.py:97`

### __init__ `def __init__(self, input_dim, hidden_dim)`
- Defined: `01_topobrain_cpu_v3.py:109`

### forward `def forward(self, x, controls)`
- Defined: `01_topobrain_cpu_v3.py:122`

### __init__ `def __init__(self, dim)`
- Defined: `01_topobrain_cpu_v3.py:146`

### forward `def forward(self, x)`
- Defined: `01_topobrain_cpu_v3.py:153`

### __init__ `def __init__(self, config)`
- Defined: `01_topobrain_cpu_v3.py:166`

### forward `def forward(self, metrics)`
- Defined: `01_topobrain_cpu_v3.py:183`

### reset_context `def reset_context(self)`
- Defined: `01_topobrain_cpu_v3.py:221`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- Defined: `01_topobrain_cpu_v3.py:228`

### get_adj `def get_adj(self)`
- Defined: `01_topobrain_cpu_v3.py:264`

### forward `def forward(self, x, controls)`
- Defined: `01_topobrain_cpu_v3.py:269`

### __init__ `def __init__(self, config)`
- Defined: `01_topobrain_cpu_v3.py:321`

### _initialize_memories `def _initialize_memories(self)`
- Defined: `01_topobrain_cpu_v3.py:352`
- Doc: Inicialización de memorias semánticas

### forward `def forward(self, x, controls, prev_states)`
- Defined: `01_topobrain_cpu_v3.py:363`

### evaluate_adv `def evaluate_adv(loader, eps, steps)`
- Defined: `01_topobrain_cpu_v3.py:593`

## 01_topobrain_cpu_v4.py

### seed_everything `def seed_everything(seed)`
- Defined: `01_topobrain_cpu_v4.py:70`

### get_tabular_loaders `def get_tabular_loaders(config)`
- Defined: `01_topobrain_cpu_v4.py:77`
- Doc: Dataset tabular controlado con características NOIR simuladas

### stable_pgd_attack `def stable_pgd_attack(model, x, y, eps, steps, controls)`
- Defined: `01_topobrain_cpu_v4.py:475`
- Doc: Ataque PGD estable con manejo de gradientes robusto

### train_epoch `def train_epoch(model, loader, optimizer, config, epoch, controls)`
- Defined: `01_topobrain_cpu_v4.py:518`
- Doc: Entrenamiento por época con monitoreo detallado

### evaluate_model `def evaluate_model(model, loader, config, adversarial, controls)`
- Defined: `01_topobrain_cpu_v4.py:583`
- Doc: Evaluación rigurosa con o sin ataques adversariales

### run_scientific_ablation `def run_scientific_ablation()`
- Defined: `01_topobrain_cpu_v4.py:628`
- Doc: Ejecución científica del ablation con control de variables

### to_dict `def to_dict(self)`
- Defined: `01_topobrain_cpu_v4.py:56`

### get_topology_config `def get_topology_config(self)`
- Defined: `01_topobrain_cpu_v4.py:59`
- Doc: Configuración estable para topología adaptable

### __init__ `def __init__(self, temperature)`
- Defined: `01_topobrain_cpu_v4.py:118`

### forward `def forward(self, features, labels)`
- Defined: `01_topobrain_cpu_v4.py:123`

### __init__ `def __init__(self, input_dim, hidden_dim)`
- Defined: `01_topobrain_cpu_v4.py:149`

### forward `def forward(self, x, controls)`
- Defined: `01_topobrain_cpu_v4.py:181`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `01_topobrain_cpu_v4.py:233`

### forward `def forward(self, x)`
- Defined: `01_topobrain_cpu_v4.py:242`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `01_topobrain_cpu_v4.py:269`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `01_topobrain_cpu_v4.py:288`
- Doc: Obtener matriz de adyacencia con estabilidad garantizada

### prune_topology `def prune_topology(self, current_density, epoch)`
- Defined: `01_topobrain_cpu_v4.py:317`
- Doc: Poda controlada con protocolo de emergencia

### get_density `def get_density(self)`
- Defined: `01_topobrain_cpu_v4.py:337`
- Doc: Calcular densidad actual de manera estable

### __init__ `def __init__(self, config)`
- Defined: `01_topobrain_cpu_v4.py:347`

### _init_weights `def _init_weights(self)`
- Defined: `01_topobrain_cpu_v4.py:400`
- Doc: Inicialización estable de pesos

### forward `def forward(self, x, controls)`
- Defined: `01_topobrain_cpu_v4.py:408`

## 01_topobrain_cpu_v5.py

### seed_everything `def seed_everything(seed)`
- Defined: `01_topobrain_cpu_v5.py:70`

### get_tabular_loaders `def get_tabular_loaders(config)`
- Defined: `01_topobrain_cpu_v5.py:77`
- Doc: Dataset tabular controlado con características NOIR simuladas

### stable_pgd_attack `def stable_pgd_attack(model, x, y, eps, steps, controls)`
- Defined: `01_topobrain_cpu_v5.py:475`
- Doc: Ataque PGD estable con manejo de gradientes robusto

### train_epoch `def train_epoch(model, loader, optimizer, config, epoch, controls)`
- Defined: `01_topobrain_cpu_v5.py:518`
- Doc: Entrenamiento por época con monitoreo detallado

### evaluate_model `def evaluate_model(model, loader, config, adversarial, controls)`
- Defined: `01_topobrain_cpu_v5.py:583`
- Doc: Evaluación rigurosa con o sin ataques adversariales

### run_scientific_ablation `def run_scientific_ablation()`
- Defined: `01_topobrain_cpu_v5.py:708`
- Doc: Ejecución científica del ablation con control de variables

### to_dict `def to_dict(self)`
- Defined: `01_topobrain_cpu_v5.py:56`

### get_topology_config `def get_topology_config(self)`
- Defined: `01_topobrain_cpu_v5.py:59`
- Doc: Configuración estable para topología adaptable

### __init__ `def __init__(self, temperature)`
- Defined: `01_topobrain_cpu_v5.py:118`

### forward `def forward(self, features, labels)`
- Defined: `01_topobrain_cpu_v5.py:123`

### __init__ `def __init__(self, input_dim, hidden_dim)`
- Defined: `01_topobrain_cpu_v5.py:149`

### forward `def forward(self, x, controls)`
- Defined: `01_topobrain_cpu_v5.py:181`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `01_topobrain_cpu_v5.py:233`

### forward `def forward(self, x)`
- Defined: `01_topobrain_cpu_v5.py:242`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `01_topobrain_cpu_v5.py:269`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `01_topobrain_cpu_v5.py:288`
- Doc: Obtener matriz de adyacencia con estabilidad garantizada

### prune_topology `def prune_topology(self, current_density, epoch)`
- Defined: `01_topobrain_cpu_v5.py:317`
- Doc: Poda controlada con protocolo de emergencia

### get_density `def get_density(self)`
- Defined: `01_topobrain_cpu_v5.py:337`
- Doc: Calcular densidad actual de manera estable

### __init__ `def __init__(self, config)`
- Defined: `01_topobrain_cpu_v5.py:347`

### _init_weights `def _init_weights(self)`
- Defined: `01_topobrain_cpu_v5.py:400`
- Doc: Inicialización estable de pesos

### forward `def forward(self, x, controls)`
- Defined: `01_topobrain_cpu_v5.py:408`

## 01_topobrain_cpu_v6.py

### seed_everything `def seed_everything(seed)`
- Defined: `01_topobrain_cpu_v6.py:95`
- Doc: Control de reproducibilidad

### get_micro_dataset `def get_micro_dataset(config)`
- Defined: `01_topobrain_cpu_v6.py:104`
- Doc: Dataset tabular controlado con validación cruzada

### compute_effect_size `def compute_effect_size(group1, group2)`
- Defined: `01_topobrain_cpu_v6.py:126`
- Doc: Cohen's d para medir tamaño del efecto

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- Defined: `01_topobrain_cpu_v6.py:432`
- Doc: PGD ultra-eficiente para CPU

### train_epoch_micro `def train_epoch_micro(model, loader, optimizer, config, epoch)`
- Defined: `01_topobrain_cpu_v6.py:462`
- Doc: Entrenamiento por época

### evaluate_micro `def evaluate_micro(model, loader, config, adversarial)`
- Defined: `01_topobrain_cpu_v6.py:517`
- Doc: Evaluación con opción adversarial

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `01_topobrain_cpu_v6.py:539`
- Doc: Entrenamiento con validación cruzada estratificada.

### run_scientific_ablation_study `def run_scientific_ablation_study()`
- Defined: `01_topobrain_cpu_v6.py:850`
- Doc: Ejecutor completo del estudio de ablación con análisis científico.

### to_dict `def to_dict(self)`
- Defined: `01_topobrain_cpu_v6.py:79`

### component_signature `def component_signature(self)`
- Defined: `01_topobrain_cpu_v6.py:82`
- Doc: Firma única de componentes activos

### __init__ `def __init__(self, temperature)`
- Defined: `01_topobrain_cpu_v6.py:139`

### forward `def forward(self, features, labels)`
- Defined: `01_topobrain_cpu_v6.py:144`

### __init__ `def __init__(self, dim)`
- Defined: `01_topobrain_cpu_v6.py:168`

### forward `def forward(self, x, plasticity)`
- Defined: `01_topobrain_cpu_v6.py:184`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `01_topobrain_cpu_v6.py:219`

### forward `def forward(self, x)`
- Defined: `01_topobrain_cpu_v6.py:232`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `01_topobrain_cpu_v6.py:258`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `01_topobrain_cpu_v6.py:276`
- Doc: Matriz de adyacencia normalizada

### __init__ `def __init__(self, config)`
- Defined: `01_topobrain_cpu_v6.py:301`

### _init_weights `def _init_weights(self)`
- Defined: `01_topobrain_cpu_v6.py:353`

### count_parameters `def count_parameters(self)`
- Defined: `01_topobrain_cpu_v6.py:360`
- Doc: Contar parámetros entrenables

### forward `def forward(self, x, plasticity)`
- Defined: `01_topobrain_cpu_v6.py:364`

### level1_isolated `def level1_isolated()`
- Defined: `01_topobrain_cpu_v6.py:628`
- Doc: NIVEL 1: Componentes aislados (6 experimentos)

### level2a_pairs `def level2a_pairs()`
- Defined: `01_topobrain_cpu_v6.py:641`
- Doc: NIVEL 2A: Todos los pares (10 experimentos = C(5,2))

### level2b_strategic_triads `def level2b_strategic_triads()`
- Defined: `01_topobrain_cpu_v6.py:665`
- Doc: NIVEL 2B: Tríadas estratégicas (8 experimentos selectos)

### level3_inverse_ablation `def level3_inverse_ablation()`
- Defined: `01_topobrain_cpu_v6.py:694`
- Doc: NIVEL 3: Ablación inversa (5 experimentos)

### level3_full_model `def level3_full_model()`
- Defined: `01_topobrain_cpu_v6.py:719`
- Doc: Modelo completo (referencia máxima)

### get_complete_matrix `def get_complete_matrix(cls)`
- Defined: `01_topobrain_cpu_v6.py:730`
- Doc: Matriz completa de ablación (30 experimentos)

### compute_statistics `def compute_statistics(results_list)`
- Defined: `01_topobrain_cpu_v6.py:748`
- Doc: Análisis estadístico por experimento.

### ttest_vs_baseline `def ttest_vs_baseline(exp_scores, baseline_scores)`
- Defined: `01_topobrain_cpu_v6.py:774`
- Doc: t-test pareado vs baseline.

### detect_synergy `def detect_synergy(pair_pgd, comp_a_pgd, comp_b_pgd, baseline_pgd)`
- Defined: `01_topobrain_cpu_v6.py:786`
- Doc: Detecta sinergia no-lineal.

### rank_components_by_criticality `def rank_components_by_criticality(full_pgd, ablation_results)`
- Defined: `01_topobrain_cpu_v6.py:809`
- Doc: Ranking de criticidad basado en ablación inversa.

## 01_topobrain_cpu_v7.py

### setup_device `def setup_device()`
- Defined: `01_topobrain_cpu_v7.py:24`

### pgd_attack `def pgd_attack(model, x, y, eps, steps, plasticity)`
- Defined: `01_topobrain_cpu_v7.py:256`

### train_epoch `def train_epoch(model, loader, optimizer, config, epoch, device)`
- Defined: `01_topobrain_cpu_v7.py:285`

### evaluate `def evaluate(model, loader, config, device, adversarial)`
- Defined: `01_topobrain_cpu_v7.py:321`

### get_dataset `def get_dataset(config)`
- Defined: `01_topobrain_cpu_v7.py:351`

### main `def main()`
- Defined: `01_topobrain_cpu_v7.py:393`

### to_dict `def to_dict(self)`
- Defined: `01_topobrain_cpu_v7.py:86`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `01_topobrain_cpu_v7.py:95`

### forward `def forward(self, x)`
- Defined: `01_topobrain_cpu_v7.py:106`

### __init__ `def __init__(self, num_nodes, grid_size, config)`
- Defined: `01_topobrain_cpu_v7.py:125`

### _create_grid_mask `def _create_grid_mask(self)`
- Defined: `01_topobrain_cpu_v7.py:136`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `01_topobrain_cpu_v7.py:153`

### prune_connections `def prune_connections(self, threshold)`
- Defined: `01_topobrain_cpu_v7.py:158`

### get_density `def get_density(self)`
- Defined: `01_topobrain_cpu_v7.py:178`

### __init__ `def __init__(self, config)`
- Defined: `01_topobrain_cpu_v7.py:187`

### _init_weights `def _init_weights(self)`
- Defined: `01_topobrain_cpu_v7.py:216`

### count_parameters `def count_parameters(self)`
- Defined: `01_topobrain_cpu_v7.py:223`

### forward `def forward(self, x, plasticity)`
- Defined: `01_topobrain_cpu_v7.py:226`

## 01_topobrain_cpu_v8.py

### pgd_attack `def pgd_attack(model, x, y, eps, steps, plasticity)`
- Defined: `01_topobrain_cpu_v8.py:223`

### train_topobrain `def train_topobrain(config)`
- Defined: `01_topobrain_cpu_v8.py:252`

### export_for_onnxruntime `def export_for_onnxruntime(model)`
- Defined: `01_topobrain_cpu_v8.py:410`
- Doc: Exporta para ONNX Runtime (más moderno que OpenCV)

### test_with_onnxruntime `def test_with_onnxruntime(X_test, y_test)`
- Defined: `01_topobrain_cpu_v8.py:465`
- Doc: Inferencia usando ONNX Runtime

### main `def main()`
- Defined: `01_topobrain_cpu_v8.py:547`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `01_topobrain_cpu_v8.py:63`

### forward `def forward(self, x)`
- Defined: `01_topobrain_cpu_v8.py:76`

### __init__ `def __init__(self, num_nodes, grid_size, config)`
- Defined: `01_topobrain_cpu_v8.py:89`

### _create_grid_mask `def _create_grid_mask(self)`
- Defined: `01_topobrain_cpu_v8.py:98`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `01_topobrain_cpu_v8.py:115`

### prune_connections `def prune_connections(self, threshold)`
- Defined: `01_topobrain_cpu_v8.py:120`

### get_density `def get_density(self)`
- Defined: `01_topobrain_cpu_v8.py:140`

### __init__ `def __init__(self, config)`
- Defined: `01_topobrain_cpu_v8.py:145`

### _init_weights `def _init_weights(self)`
- Defined: `01_topobrain_cpu_v8.py:174`

### forward `def forward(self, x, plasticity)`
- Defined: `01_topobrain_cpu_v8.py:181`

### forward_with_metrics `def forward_with_metrics(self, x, plasticity)`
- Defined: `01_topobrain_cpu_v8.py:210`

### __init__ `def __init__(self, m)`
- Defined: `01_topobrain_cpu_v8.py:425`

### forward `def forward(self, x)`
- Defined: `01_topobrain_cpu_v8.py:429`

## 01_topobrain_ganador_gpu_v1.py

### setup_amd_device `def setup_amd_device()`
- Defined: `01_topobrain_ganador_gpu_v1.py:31`
- Doc: Configura PyTorch para usar GPU AMD con ROCm/OpenCL.

### pgd_attack_gpu `def pgd_attack_gpu(model, x, y, eps, steps, plasticity)`
- Defined: `01_topobrain_ganador_gpu_v1.py:405`
- Doc: PGD attack optimizado para GPU.

### train_epoch_gpu `def train_epoch_gpu(model, loader, optimizer, config, epoch, device)`
- Defined: `01_topobrain_ganador_gpu_v1.py:461`
- Doc: Entrenamiento por época en GPU

### evaluate_gpu `def evaluate_gpu(model, loader, config, device, adversarial)`
- Defined: `01_topobrain_ganador_gpu_v1.py:517`
- Doc: Evaluación en GPU

### get_gpu_dataset `def get_gpu_dataset(config)`
- Defined: `01_topobrain_ganador_gpu_v1.py:547`
- Doc: Dataset sintético para GPU

### main `def main()`
- Defined: `01_topobrain_ganador_gpu_v1.py:591`
- Doc: Ejecutar POC completa

### to_dict `def to_dict(self)`
- Defined: `01_topobrain_ganador_gpu_v1.py:119`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `01_topobrain_ganador_gpu_v1.py:132`

### forward `def forward(self, x)`
- Defined: `01_topobrain_ganador_gpu_v1.py:147`
- Doc: Args:

### __init__ `def __init__(self, num_nodes, grid_size, config)`
- Defined: `01_topobrain_ganador_gpu_v1.py:184`

### _create_grid_mask `def _create_grid_mask(self)`
- Defined: `01_topobrain_ganador_gpu_v1.py:200`
- Doc: Crea máscara de vecindad para grid NxN

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `01_topobrain_ganador_gpu_v1.py:219`
- Doc: Obtiene matriz de adyacencia normalizada.

### prune_connections `def prune_connections(self, threshold)`
- Defined: `01_topobrain_ganador_gpu_v1.py:237`
- Doc: Poda conexiones débiles (llamar cada N epochs).

### get_density `def get_density(self)`
- Defined: `01_topobrain_ganador_gpu_v1.py:269`
- Doc: Densidad actual de conexiones

### __init__ `def __init__(self, config)`
- Defined: `01_topobrain_ganador_gpu_v1.py:291`

### _init_weights `def _init_weights(self)`
- Defined: `01_topobrain_ganador_gpu_v1.py:335`
- Doc: Inicialización Kaiming para activaciones GELU

### count_parameters `def count_parameters(self)`
- Defined: `01_topobrain_ganador_gpu_v1.py:343`
- Doc: Cuenta parámetros entrenables

### forward `def forward(self, x, plasticity)`
- Defined: `01_topobrain_ganador_gpu_v1.py:347`
- Doc: Forward pass completo.

## 02_perceptron.py

### __init__ `def __init__(self, input_dim, learning_rate)`
- Defined: `02_perceptron.py:32`

### predict `def predict(self, X)`
- Defined: `02_perceptron.py:37`

### train_step `def train_step(self, X_batch, y_batch)`
- Defined: `02_perceptron.py:42`

### accuracy `def accuracy(self, X, y_true)`
- Defined: `02_perceptron.py:51`

## 03_backpropagation.py

### sigmoid `def sigmoid(z)`
- Defined: `03_backpropagation.py:33`

### sigmoid_derivative `def sigmoid_derivative(z)`
- Defined: `03_backpropagation.py:38`

### forward `def forward(X)`
- Defined: `03_backpropagation.py:53`

### backward `def backward(X, y_true, y_pred, a1, z1, lr)`
- Defined: `03_backpropagation.py:63`

### compute_metrics `def compute_metrics(y_pred, y_true)`
- Defined: `03_backpropagation.py:88`

## 04_cnn_lenet.py

### __init__ `def __init__(self)`
- Defined: `04_cnn_lenet.py:46`

### forward `def forward(self, x)`
- Defined: `04_cnn_lenet.py:60`

## 06_lstm_char.py

### create_batches `def create_batches(data, batch_size, seq_length)`
- Defined: `06_lstm_char.py:63`

### __init__ `def __init__(self, vocab_size, hidden_size, num_layers, dropout)`
- Defined: `06_lstm_char.py:90`

### forward `def forward(self, x, hidden)`
- Defined: `06_lstm_char.py:102`

## 08_vae_mnist.py

### vae_loss `def vae_loss(recon_x, x, mu, log_var)`
- Defined: `08_vae_mnist.py:72`

### __init__ `def __init__(self, input_dim, hidden_dim, latent_dim)`
- Defined: `08_vae_mnist.py:41`

### encode `def encode(self, x)`
- Defined: `08_vae_mnist.py:51`

### reparameterize `def reparameterize(self, mu, log_var)`
- Defined: `08_vae_mnist.py:55`

### decode `def decode(self, z)`
- Defined: `08_vae_mnist.py:60`

### forward `def forward(self, x)`
- Defined: `08_vae_mnist.py:64`

## 09_transformer_mini.py

### generate_copy_data `def generate_copy_data(num_samples, seq_len, vocab_size)`
- Defined: `09_transformer_mini.py:29`

### __init__ `def __init__(self, d_model, num_heads, dropout)`
- Defined: `09_transformer_mini.py:52`

### forward `def forward(self, q, k, v, mask)`
- Defined: `09_transformer_mini.py:63`

### __init__ `def __init__(self, d_model, d_ff, dropout)`
- Defined: `09_transformer_mini.py:84`

### forward `def forward(self, x)`
- Defined: `09_transformer_mini.py:90`

### __init__ `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len)`
- Defined: `09_transformer_mini.py:97`

### _create_positional_encoding `def _create_positional_encoding(self, max_len, d_model)`
- Defined: `09_transformer_mini.py:107`

### forward `def forward(self, x)`
- Defined: `09_transformer_mini.py:115`

### __init__ `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout)`
- Defined: `09_transformer_mini.py:132`

### forward `def forward(self, x)`
- Defined: `09_transformer_mini.py:137`

## 10_gan_mnist_lite.py

### __init__ `def __init__(self, latent_dim, img_size)`
- Defined: `10_gan_mnist_lite.py:38`

### forward `def forward(self, z)`
- Defined: `10_gan_mnist_lite.py:51`

### __init__ `def __init__(self, img_size)`
- Defined: `10_gan_mnist_lite.py:58`

### forward `def forward(self, x)`
- Defined: `10_gan_mnist_lite.py:74`

## 11_bert_tiny.py

### tokenize_sentence `def tokenize_sentence(sentence)`
- Defined: `11_bert_tiny.py:76`

### pad_sequence `def pad_sequence(seq, length, pad_value)`
- Defined: `11_bert_tiny.py:88`

### mask_tokens `def mask_tokens(inputs, vocab_size, mask_token_id, pad_token_id, mask_prob)`
- Defined: `11_bert_tiny.py:186`

### __init__ `def __init__(self, d_model, num_heads, dropout)`
- Defined: `11_bert_tiny.py:108`

### forward `def forward(self, q, k, v, mask)`
- Defined: `11_bert_tiny.py:119`

### __init__ `def __init__(self, d_model, d_ff, dropout)`
- Defined: `11_bert_tiny.py:134`

### forward `def forward(self, x)`
- Defined: `11_bert_tiny.py:140`

### __init__ `def __init__(self, d_model, num_heads, d_ff, dropout)`
- Defined: `11_bert_tiny.py:144`

### forward `def forward(self, x, mask)`
- Defined: `11_bert_tiny.py:151`

### __init__ `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len)`
- Defined: `11_bert_tiny.py:159`

### _create_positional_encoding `def _create_positional_encoding(self, max_len, d_model)`
- Defined: `11_bert_tiny.py:167`

### forward `def forward(self, x, mask)`
- Defined: `11_bert_tiny.py:175`

## 12_diffusion_minimal.py

### q_sample `def q_sample(x_0, t, noise)`
- Defined: `12_diffusion_minimal.py:81`
- Doc: Muestrea x_t dado x_0 y timestep t.

### __init__ `def __init__(self, in_channels, out_channels, hidden_dim)`
- Defined: `12_diffusion_minimal.py:55`

### forward `def forward(self, x, t)`
- Defined: `12_diffusion_minimal.py:65`

## 13_nested_hope.py

### setup_device `def setup_device()`
- Defined: `13_nested_hope.py:44`
- Doc: Configuración automática de dispositivo

### set_seed `def set_seed(seed)`
- Defined: `13_nested_hope.py:54`
- Doc: Reproducibilidad completa

### run_ablation_study `def run_ablation_study(config, device)`
- Defined: `13_nested_hope.py:564`
- Doc: Ejecuta estudio de ablación completo

### apply_update `def apply_update(grad, param, x_normalized, eta, alpha, lambda_norm)`
- Defined: `13_nested_hope.py:77`
- Doc: Aplica la regla DGD a un gradiente

### __init__ `def __init__(self, d_model, hidden_dim, chunk_size)`
- Defined: `13_nested_hope.py:125`

### _make_memory_module `def _make_memory_module(self)`
- Defined: `13_nested_hope.py:162`
- Doc: Crea un módulo de memoria (MLP de 2 capas con residual)

### forward `def forward(self, x, prev_states)`
- Defined: `13_nested_hope.py:170`
- Doc: Forward pass con actualización chunk-wise (Sección 8.2)

### __init__ `def __init__(self, frequencies, d_model, hidden_dim, connection_type)`
- Defined: `13_nested_hope.py:268`

### forward `def forward(self, x, global_step)`
- Defined: `13_nested_hope.py:296`
- Doc: Forward pass con actualizaciones multi-frecuencia

### __init__ `def __init__(self, vocab_size, d_model, cms_frequencies, mlp_hidden, chunk_size, enable_self_modifying, enable_cms)`
- Defined: `13_nested_hope.py:344`

### reset_states `def reset_states(self)`
- Defined: `13_nested_hope.py:390`
- Doc: Reset de estados internos (para nuevas secuencias)

### forward `def forward(self, x, global_step, return_internals)`
- Defined: `13_nested_hope.py:394`
- Doc: Forward pass completo

### __init__ `def __init__(self, model, config, device)`
- Defined: `13_nested_hope.py:437`

### train_epoch `def train_epoch(self, train_loader, epoch, global_step)`
- Defined: `13_nested_hope.py:461`
- Doc: Entrena una época completa

### evaluate `def evaluate(self, test_loader, global_step)`
- Defined: `13_nested_hope.py:536`
- Doc: Evaluación sin gradientes

## 13_nested_kearning_gpu.py

### __init__ `def __init__(self, vocab_size, d_model, hidden_dim)`
- Defined: `13_nested_kearning_gpu.py:54`

### forward `def forward(self, x)`
- Defined: `13_nested_kearning_gpu.py:64`

### __init__ `def __init__(self, frequencies, d_model, hidden_dim)`
- Defined: `13_nested_kearning_gpu.py:75`

### forward `def forward(self, x, global_step)`
- Defined: `13_nested_kearning_gpu.py:87`

### __init__ `def __init__(self, vocab_size, d_model, cms_freqs, hidden_dim)`
- Defined: `13_nested_kearning_gpu.py:98`

### forward `def forward(self, x, global_step)`
- Defined: `13_nested_kearning_gpu.py:104`

## 13_nested_learning.py

### __init__ `def __init__(self, vocab_size, d_model, hidden_dim)`
- Defined: `13_nested_learning.py:52`

### forward `def forward(self, x, update_mask)`
- Defined: `13_nested_learning.py:62`
- Doc: x: (B, S)

### __init__ `def __init__(self, levels, d_model, hidden_dim)`
- Defined: `13_nested_learning.py:87`

### forward `def forward(self, x, global_step)`
- Defined: `13_nested_learning.py:100`
- Doc: x: (B, S, D)

### __init__ `def __init__(self, vocab_size, d_model, cms_levels, mlp_hidden)`
- Defined: `13_nested_learning.py:115`

### forward `def forward(self, x, global_step)`
- Defined: `13_nested_learning.py:122`

## NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py

### compute_loss `def compute_loss(logits, captions, gate, vocab)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:20`

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:764`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:782`

### train_with_metrics `def train_with_metrics()`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:854`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:48`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:82`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:91`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:104`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:120`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:125`
- Doc: Identificar señales convergentes que confirman problemas

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:157`
- Doc: Contar cuántas señales del patrón están activas

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:161`
- Doc: Diagnosticar SOLO con confirmación múltiple

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:223`
- Doc: Aplicar intervención SOLO si confianza es alta

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:383`

### forward `def forward(self, x)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:397`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:404`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:434`

### __init__ `def __init__(self, output_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:459`

### forward `def forward(self, image)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:467`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:474`

### forward `def forward(self, visual_context, captions, max_len)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:502`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:539`

### __init__ `def __init__(self, dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:545`

### forward `def forward(self, right_features)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:568`

### __init__ `def __init__(self, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:582`

### forward `def forward(self, image, captions)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:588`

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:603`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:613`

### calculate_synergy `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:622`

### calculate_health `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:631`

### update `def update(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:640`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:645`

### report `def report(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:650`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:728`

### __len__ `def __len__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:745`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:748`

## NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py

### compute_loss `def compute_loss(logits, captions, gate, vocab)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:20`

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:928`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:946`

### train_with_metrics `def train_with_metrics()`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:1021`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:48`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:82`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:91`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:104`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:120`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:125`
- Doc: Identificar señales convergentes que confirman problemas

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:157`
- Doc: Contar cuántas señales del patrón están activas

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:161`
- Doc: Diagnosticar SOLO con confirmación múltiple

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:224`
- Doc: Aplicar intervención SOLO si confianza es alta

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:384`

### forward `def forward(self, x)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:422`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:444`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:483`

### __init__ `def __init__(self, output_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:510`

### forward `def forward(self, image)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:518`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:525`

### forward `def forward(self, visual_context, captions, max_len)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:573`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:631`

### __init__ `def __init__(self, dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:646`

### forward `def forward(self, right_features)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:678`

### __init__ `def __init__(self, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:701`

### forward `def forward(self, image, captions)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:707`

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:722`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:732`

### calculate_synergy `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:741`

### calculate_health `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:750`

### update `def update(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:759`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:764`

### report `def report(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:769`

### __init__ `def __init__(self, capacity, surprise_threshold)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:845`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:851`

### add `def add(self, image, caption, surprise_score)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:862`

### sample `def sample(self, batch_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:872`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:892`

### __len__ `def __len__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:909`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:912`

## NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py

### compute_loss `def compute_loss(logits, captions, gate, vocab)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:20`

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:991`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:1009`

### train_with_metrics `def train_with_metrics()`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:1081`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:48`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:82`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:91`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:104`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:120`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:125`
- Doc: Identificar señales convergentes que confirman problemas

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:157`
- Doc: Contar cuántas señales del patrón están activas

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:161`
- Doc: Diagnosticar SOLO con confirmación múltiple

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:224`
- Doc: Aplicar intervención SOLO si confianza es alta

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:393`

### forward `def forward(self, x)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:431`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:453`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:492`

### __init__ `def __init__(self, output_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:519`

### forward `def forward(self, image)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:527`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:534`

### beam_search_decode `def beam_search_decode(self, visual_context, beam_width, max_len, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:584`

### forward `def forward(self, visual_context, captions, max_len, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:657`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:695`

### __init__ `def __init__(self, dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:710`

### forward `def forward(self, right_features)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:742`

### __init__ `def __init__(self, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:766`

### forward `def forward(self, image, captions, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:772`

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:787`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:797`

### calculate_synergy `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:806`

### calculate_health `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:815`

### update `def update(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:824`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:829`

### report `def report(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:834`

### __init__ `def __init__(self, capacity, surprise_threshold)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:908`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:914`

### add `def add(self, image, caption, surprise_score)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:925`

### sample `def sample(self, batch_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:935`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:955`

### __len__ `def __len__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:972`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:975`

## NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py

### compute_loss `def compute_loss(logits, captions, gate, vocab, linguistic_reward, lambda_reward)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:20`
- Doc: Función de pérdida extendida que incorpora recompensa lingüística

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1276`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1294`

### train_with_metrics `def train_with_metrics()`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1366`

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:55`

### assess_cognitive_state `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:65`
- Doc: Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas

### apply_cognitive_intervention `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:113`
- Doc: Aplica intervenciones cognitivas basadas en el estado lingüístico

### __init__ `def __init__(self, alpha, beta)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:223`

### compute_linguistic_reward `def compute_linguistic_reward(self, references, hypotheses)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:232`
- Doc: Calcula una recompensa combinada basada en CIDEr y SPICE

### compute_cider `def compute_cider(self, reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:254`
- Doc: Versión simplificada de CIDEr para uso en entrenamiento

### compute_spice `def compute_spice(self, reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:281`
- Doc: Versión simplificada de SPICE para uso en entrenamiento

### _get_ngrams `def _get_ngrams(self, sentence, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:295`
- Doc: Extrae n-gramas de una oración

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:313`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:347`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:356`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:369`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:385`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:390`
- Doc: Identificar señales convergentes que confirman problemas

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:422`
- Doc: Contar cuántas señales del patrón están activas

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:426`
- Doc: Diagnosticar SOLO con confirmación múltiple

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:489`
- Doc: Aplicar intervención SOLO si confianza es alta

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:658`

### forward `def forward(self, x)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:696`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:718`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:757`

### __init__ `def __init__(self, output_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:784`

### forward `def forward(self, image)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:792`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:799`

### beam_search_decode `def beam_search_decode(self, visual_context, beam_width, max_len, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:849`

### forward `def forward(self, visual_context, captions, max_len, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:925`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:963`

### __init__ `def __init__(self, dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:979`

### forward `def forward(self, right_features)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1011`

### __init__ `def __init__(self, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1035`

### forward `def forward(self, image, captions, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1041`

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1056`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1067`

### calculate_synergy `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1076`

### calculate_health `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1085`

### update `def update(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1094`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1099`

### report `def report(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1104`

### __init__ `def __init__(self, capacity, surprise_threshold)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1193`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1199`

### add `def add(self, image, caption, surprise_score)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1210`

### sample `def sample(self, batch_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1220`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1240`

### __len__ `def __len__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1257`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1260`

## NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py

### compute_loss `def compute_loss(logits, captions, gate, vocab, linguistic_reward, lambda_reward)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:20`
- Doc: Función de pérdida extendida que incorpora recompensa lingüística

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1745`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1763`

### compute_alignment_loss `def compute_alignment_loss(visual_features, channels, alpha)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1834`
- Doc: Pérdida auxiliar para forzar alineación entre características visuales

### train_with_metrics `def train_with_metrics()`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1858`

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:55`

### assess_cognitive_state `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:76`
- Doc: Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas

### evaluate_gate_state `def evaluate_gate_state(self, gate_value, current_metrics)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:124`
- Doc: MEJORA: Evaluar estado del gate con sistema inmune

### update_trauma_memory `def update_trauma_memory(self, gate_value, metrics, outcome)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:142`
- Doc: MEJORA: Actualizar memoria traumática basada en resultados

### apply_stochastic_perturbation `def apply_stochastic_perturbation(self, model, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:155`
- Doc: MEJORA: Aplicar micro-perturbaciones estocásticas

### apply_cognitive_intervention `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:171`
- Doc: Aplica intervenciones cognitivas basadas en el estado lingüístico

### __init__ `def __init__(self, alpha, beta)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:331`

### compute_linguistic_reward `def compute_linguistic_reward(self, references, hypotheses)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:347`
- Doc: Calcula una recompensa combinada basada en CIDEr y SPICE.

### compute_cider `def compute_cider(self, reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:389`
- Doc: Versión simplificada de CIDEr para uso en entrenamiento.

### compute_spice `def compute_spice(self, reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:427`
- Doc: Versión simplificada de SPICE para uso en entrenamiento.

### _get_ngrams `def _get_ngrams(self, sentence, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:443`
- Doc: Extrae n-gramas de una oración

### get_cache_stats `def get_cache_stats(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:452`
- Doc: Obtiene estadísticas del sistema de caché

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:483`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:517`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:526`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:539`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:555`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:560`
- Doc: Identificar señales convergentes que confirman problemas

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:592`
- Doc: Contar cuántas señales del patrón están activas

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:596`
- Doc: Diagnosticar SOLO con confirmación múltiple

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:659`
- Doc: Aplicar intervención SOLO si confianza es alta

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:828`

### forward `def forward(self, x)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:871`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:893`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:932`

### __init__ `def __init__(self, output_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:974`

### forward `def forward(self, image)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:982`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:989`

### beam_search_decode `def beam_search_decode(self, visual_context, channels, beam_width, max_len, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1066`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1147`

### _apply_structural_attention `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1185`
- Doc: Aplica atención específica para cada canal estructural (objetos, acciones, escena).

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1230`

### __init__ `def __init__(self, dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1245`

### forward `def forward(self, right_features, left_features)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1308`

### update_channel_fatigue `def update_channel_fatigue(self, objects_channel, actions_channel, scene_channel)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1381`
- Doc: MEJORA: Actualizar fatiga específica por canal

### adjust_gates_by_fatigue `def adjust_gates_by_fatigue(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1404`
- Doc: MEJORA: Ajustar gates basado en fatiga de cada canal

### __init__ `def __init__(self, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1421`

### forward `def forward(self, image, captions, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1427`

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1447`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context, channels)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1460`

### calculate_synergy `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1490`

### calculate_health `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1499`

### update `def update(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1508`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1519`

### visualize_fatigue_distribution `def visualize_fatigue_distribution(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1538`
- Doc: MEJORA: Visualizar distribución de fatiga entre canales

### report `def report(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1567`

### __init__ `def __init__(self, capacity, surprise_threshold)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1662`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1668`

### add `def add(self, image, caption, surprise_score)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1679`

### sample `def sample(self, batch_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1689`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1709`

### __len__ `def __len__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1726`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1729`

## NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py

### compute_loss `def compute_loss(logits, captions, gate, vocab, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:20`
- Doc: Función de pérdida extendida con MTP y recompensa lingüística

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2031`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2049`

### compute_alignment_loss `def compute_alignment_loss(visual_features, channels, alpha)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2120`
- Doc: Pérdida auxiliar para forzar alineación entre características visuales

### train_with_metrics `def train_with_metrics()`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2144`

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:59`

### assess_reasoning_state `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:85`
- Doc: Evalúa el estado del sistema de razonamiento

### assess_cognitive_state `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:128`
- Doc: Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas

### evaluate_gate_state `def evaluate_gate_state(self, gate_value, current_metrics)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:167`
- Doc: Evaluar estado del gate con sistema inmune

### update_trauma_memory `def update_trauma_memory(self, gate_value, metrics, outcome)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:182`
- Doc: Actualizar memoria traumática basada en resultados

### apply_stochastic_perturbation `def apply_stochastic_perturbation(self, model, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:192`
- Doc: Aplicar micro-perturbaciones estocásticas

### apply_cognitive_intervention `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:213`
- Doc: Aplica intervenciones cognitivas basadas en el estado lingüístico y de razonamiento

### __init__ `def __init__(self, alpha, beta)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:402`

### compute_linguistic_reward `def compute_linguistic_reward(self, references, hypotheses)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:418`
- Doc: Calcula una recompensa combinada basada en CIDEr y SPICE.

### compute_cider `def compute_cider(self, reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:460`
- Doc: Versión simplificada de CIDEr para uso en entrenamiento.

### compute_spice `def compute_spice(self, reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:498`
- Doc: Versión simplificada de SPICE para uso en entrenamiento.

### _get_ngrams `def _get_ngrams(self, sentence, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:514`
- Doc: Extrae n-gramas de una oración

### get_cache_stats `def get_cache_stats(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:523`
- Doc: Obtiene estadísticas del sistema de caché

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:554`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:588`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:597`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:610`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:626`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:631`
- Doc: Identificar señales convergentes que confirman problemas

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:663`
- Doc: Contar cuántas señales del patrón están activas

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:667`
- Doc: Diagnosticar SOLO con confirmación múltiple

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:730`
- Doc: Aplicar intervención SOLO si confianza es alta

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:899`

### forward `def forward(self, x)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:942`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:964`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1003`

### __init__ `def __init__(self, output_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1045`

### forward `def forward(self, image)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1053`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1060`

### _apply_chain_of_thought `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1190`
- Doc: Aplica cadena de pensamiento para mejorar el razonamiento

### _apply_multi_token_prediction `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1230`
- Doc: Multi-Token Prediction: predice múltiples tokens futuros simultáneamente

### beam_search_decode `def beam_search_decode(self, visual_context, channels, beam_width, max_len, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1292`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1370`

### _apply_structural_attention `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1420`
- Doc: Aplica atención específica para cada canal estructural (objetos, acciones, escena).

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1456`

### __init__ `def __init__(self, dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1470`

### forward `def forward(self, right_features, left_features)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1533`

### update_channel_fatigue `def update_channel_fatigue(self, objects_channel, actions_channel, scene_channel)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1606`
- Doc: MEJORA: Actualizar fatiga específica por canal

### adjust_gates_by_fatigue `def adjust_gates_by_fatigue(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1629`
- Doc: MEJORA: Ajustar gates basado en fatiga de cada canal

### __init__ `def __init__(self, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1646`

### forward `def forward(self, image, captions, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1652`

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1670`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context, channels)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1686`

### evaluate_reasoning_quality `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1709`
- Doc: Evalúa la calidad del razonamiento en textos generados

### calculate_synergy `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1750`

### calculate_health `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1759`

### update `def update(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1768`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1778`

### visualize_fatigue_distribution `def visualize_fatigue_distribution(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1795`
- Doc: Visualizar distribución de fatiga entre canales

### visualize_reasoning_metrics `def visualize_reasoning_metrics(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1823`
- Doc: Visualizar métricas de razonamiento

### report `def report(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1836`

### __init__ `def __init__(self, capacity, surprise_threshold)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1948`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1954`

### add `def add(self, image, caption, surprise_score)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1965`

### sample `def sample(self, batch_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1975`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:1995`

### __len__ `def __len__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2012`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2015`

## NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py

### compute_loss `def compute_loss(logits, captions, gate, vocab, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:22`
- Doc: Función de pérdida extendida con MTP y recompensa lingüística

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1445`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1463`

### compute_alignment_loss `def compute_alignment_loss(visual_features, channels, alpha)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1527`
- Doc: Pérdida auxiliar para alineación temprana

### train_with_metrics `def train_with_metrics()`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1547`

### __init__ `def __init__(self, capacity, surprise_threshold)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:61`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:67`
- Doc: Calcula sorpresa basada en error y apertura del gate

### add `def add(self, image, caption, surprise_score)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:79`
- Doc: Añade ejemplo si supera umbral y hay capacidad

### sample `def sample(self, batch_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:91`
- Doc: Samplea ejemplos con probabilidad proporcional a sorpresa

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:113`

### assess_reasoning_state `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:128`
- Doc: Evalúa estado del sistema de razonamiento

### assess_cognitive_state `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:170`
- Doc: Evalúa estado cognitivo lingüístico

### apply_cognitive_intervention `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:206`
- Doc: Aplica intervenciones basadas en estado lingüístico y razonamiento

### __init__ `def __init__(self, alpha, beta)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:294`

### compute_linguistic_reward `def compute_linguistic_reward(self, references, hypotheses)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:308`
- Doc: Recompensa combinada CIDEr + SPICE con caché

### compute_cider `def compute_cider(self, reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:342`
- Doc: CIDEr simplificado con caché de n-gramas

### compute_spice `def compute_spice(self, reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:371`
- Doc: SPICE simplificado (Jaccard similarity)

### _get_ngrams `def _get_ngrams(self, sentence, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:384`
- Doc: Extractor de n-gramas

### get_cache_stats `def get_cache_stats(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:389`
- Doc: Estadísticas de caché

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:417`
- Doc: BLEU-4 a nivel de oración

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:448`
- Doc: Precisión token-level

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:461`
- Doc: Jaccard similarity

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:476`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:482`

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:492`

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:495`

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:537`

### _reset_liquid_neuron `def _reset_liquid_neuron(self, right_node, severity)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:610`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:623`

### forward `def forward(self, x)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:658`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:671`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:709`

### __init__ `def __init__(self, output_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:746`

### forward `def forward(self, image)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:754`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:765`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:840`

### _greedy_decode `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:879`

### _apply_chain_of_thought `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:912`

### _apply_multi_token_prediction `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:944`

### _apply_structural_attention `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:986`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1007`

### __init__ `def __init__(self, dim)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1020`

### forward `def forward(self, right_features, left_features)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1070`

### update_channel_fatigue `def update_channel_fatigue(self, objects_channel, actions_channel, scene_channel)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1113`
- Doc: Actualiza fatiga específica por canal

### adjust_gates_by_fatigue `def adjust_gates_by_fatigue(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1132`
- Doc: Ajusta gates basado en fatiga

### __init__ `def __init__(self, vocab_size)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1148`

### forward `def forward(self, image, captions, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1154`

### __init__ `def __init__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1172`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context, channels)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1187`

### evaluate_reasoning_quality `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1210`
- Doc: Evalúa coherencia y consistencia del razonamiento

### calculate_synergy `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1242`

### calculate_health `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1251`

### update `def update(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1260`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1270`

### visualize_fatigue_distribution `def visualize_fatigue_distribution(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1287`

### visualize_reasoning_metrics `def visualize_reasoning_metrics(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1308`

### report `def report(self, epoch)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1320`
- Doc: Genera reporte completo del estado del sistema bicameral

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1409`

### __len__ `def __len__(self)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1426`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1429`

## ablation.py

### fgsm_attack `def fgsm_attack(model, x, y, epsilon)`
- Defined: `ablation.py:175`

### train_epoch `def train_epoch(model, loader, optimizer, device, use_adv)`
- Defined: `ablation.py:188`

### evaluate `def evaluate(model, loader, device)`
- Defined: `ablation.py:229`

### run_ablation_cpu `def run_ablation_cpu()`
- Defined: `ablation.py:245`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)`
- Defined: `ablation.py:18`

### _init_grid `def _init_grid(self, size)`
- Defined: `ablation.py:39`

### forward `def forward(self, x)`
- Defined: `ablation.py:46`

### get_metrics `def get_metrics(self)`
- Defined: `ablation.py:64`

### __init__ `def __init__(self, out_dim)`
- Defined: `ablation.py:73`

### forward `def forward(self, x)`
- Defined: `ablation.py:87`

### __init__ `def __init__(self, out_dim, use_grid, use_symbiotic)`
- Defined: `ablation.py:92`

### forward `def forward(self, x)`
- Defined: `ablation.py:112`

### get_metrics `def get_metrics(self)`
- Defined: `ablation.py:116`

### __init__ `def __init__(self, in_dim, num_classes)`
- Defined: `ablation.py:125`

### forward `def forward(self, x)`
- Defined: `ablation.py:129`

### __init__ `def __init__(self, num_classes, ablation_level)`
- Defined: `ablation.py:145`

### forward `def forward(self, x)`
- Defined: `ablation.py:161`

### get_metrics `def get_metrics(self)`
- Defined: `ablation.py:165`

## ablation1.py

### create_ablation_configs `def create_ablation_configs(config)`
- Defined: `ablation1.py:64`
- Doc: Crea un diccionario de configuraciones para cada test de ablación.

### seed_everything `def seed_everything(seed)`
- Defined: `ablation1.py:150`

### get_elite_dataset `def get_elite_dataset(config)`
- Defined: `ablation1.py:157`
- Doc: Dataset más grande y balanceado con separabilidad controlada

### elite_pgd_attack `def elite_pgd_attack(model, x, y, eps, steps, stress)`
- Defined: `ablation1.py:407`
- Doc: PGD con reinicio aleatorio y gradiente centralizado (CORREGIDO)

### train_elite_model `def train_elite_model(config, dataset, fold_results)`
- Defined: `ablation1.py:504`
- Doc: Entrenamiento con curriculum adversarial

### run_ablation_study `def run_ablation_study()`
- Defined: `ablation1.py:616`

### __init__ `def __init__(self, dim, capacity)`
- Defined: `ablation1.py:182`

### update `def update(self, x, y)`
- Defined: `ablation1.py:190`
- Doc: Almacena ejemplos duros

### retrieve `def retrieve(self, x, k)`
- Defined: `ablation1.py:203`
- Doc: Recupera k vecinos más cercanos

### __init__ `def __init__(self, in_features, out_features)`
- Defined: `ablation1.py:222`

### power_iteration `def power_iteration(self, n_iter)`
- Defined: `ablation1.py:229`
- Doc: Aproxima la norma espectral máxima

### forward `def forward(self, x)`
- Defined: `ablation1.py:236`

### __init__ `def __init__(self, d_in, d_out, use_spectral, use_homeostasis)`
- Defined: `ablation1.py:250`

### forward `def forward(self, x)`
- Defined: `ablation1.py:271`

### __init__ `def __init__(self, num_nodes, grid_size)`
- Defined: `ablation1.py:298`

### forward `def forward(self, stress)`
- Defined: `ablation1.py:317`
- Doc: stress ∈ [0,1]: cuánto estrés adversarial

### __init__ `def __init__(self, config)`
- Defined: `ablation1.py:330`

### count_parameters `def count_parameters(self)`
- Defined: `ablation1.py:367`

### forward `def forward(self, x, stress)`
- Defined: `ablation1.py:370`

### __init__ `def __init__(self, temperature)`
- Defined: `ablation1.py:468`

### forward `def forward(self, features, labels)`
- Defined: `ablation1.py:472`

## ablation2.py

### compute_bleu `def compute_bleu(pred_ids, target_ids, dataset, max_n)`
- Defined: `ablation2.py:417`
- Doc: BLEU score simplificado para evaluar calidad de generación

### train_ablation_v51 `def train_ablation_v51(mode, epochs, device, n_nodes, k_sparse)`
- Defined: `ablation2.py:466`
- Doc: Entrena una configuración específica del ablation study v5.1.

### run_ablation_v51 `def run_ablation_v51(epochs, device, n_nodes, k_sparse)`
- Defined: `ablation2.py:619`
- Doc: Ejecuta ablation study v5.1 con 5 brazos desacoplados.

### __init__ `def __init__(self, n_nodes, k_sparse, input_dim)`
- Defined: `ablation2.py:25`

### forward `def forward(self, x)`
- Defined: `ablation2.py:44`

### get_metrics `def get_metrics(self)`
- Defined: `ablation2.py:79`

### __init__ `def __init__(self, n_nodes)`
- Defined: `ablation2.py:96`

### forward `def forward(self, x)`
- Defined: `ablation2.py:106`

### __init__ `def __init__(self, input_dim, hidden_dim, n_nodes, k_sparse)`
- Defined: `ablation2.py:125`

### forward `def forward(self, x)`
- Defined: `ablation2.py:143`

### get_metrics `def get_metrics(self)`
- Defined: `ablation2.py:158`

### __init__ `def __init__(self, output_dim)`
- Defined: `ablation2.py:169`

### forward `def forward(self, x)`
- Defined: `ablation2.py:182`

### __init__ `def __init__(self, output_dim, n_nodes, k_sparse)`
- Defined: `ablation2.py:188`

### forward `def forward(self, x)`
- Defined: `ablation2.py:207`

### get_metrics `def get_metrics(self)`
- Defined: `ablation2.py:211`

### __init__ `def __init__(self, dim)`
- Defined: `ablation2.py:220`

### forward `def forward(self, x)`
- Defined: `ablation2.py:225`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `ablation2.py:233`

### forward `def forward(self, thought, captions, max_len)`
- Defined: `ablation2.py:248`

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `ablation2.py:279`

### __init__ `def __init__(self, vocab_size, mode, n_nodes, k_sparse)`
- Defined: `ablation2.py:299`

### forward `def forward(self, image, captions)`
- Defined: `ablation2.py:331`

### get_metrics `def get_metrics(self)`
- Defined: `ablation2.py:340`

### __init__ `def __init__(self, epsilon, alpha, steps)`
- Defined: `ablation2.py:347`

### attack `def attack(self, model, x, y, criterion)`
- Defined: `ablation2.py:352`

### __init__ `def __init__(self)`
- Defined: `ablation2.py:376`

### __len__ `def __len__(self)`
- Defined: `ablation2.py:403`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `ablation2.py:406`

### ngrams `def ngrams(tokens, n)`
- Defined: `ablation2.py:423`

### forward_fn `def forward_fn(x)`
- Defined: `ablation2.py:526`

## ablation3.py

### train_configuration `def train_configuration(config, epochs, device, n_nodes, k_sparse)`
- Defined: `ablation3.py:338`
- Doc: Entrena UNA configuración específica del diseño factorial.

### run_full_factorial `def run_full_factorial(epochs, device, n_nodes, k_sparse)`
- Defined: `ablation3.py:411`
- Doc: Ejecuta el ablation factorial completo: 8 combinaciones + 3 inversas.

### analyze_results `def analyze_results(results)`
- Defined: `ablation3.py:480`
- Doc: Análisis de efectos principales, interacciones y poder explicativo.

### __init__ `def __init__(self, input_dim, n_nodes, k_sparse)`
- Defined: `ablation3.py:23`

### forward `def forward(self, x)`
- Defined: `ablation3.py:33`

### __init__ `def __init__(self, n_nodes)`
- Defined: `ablation3.py:56`

### forward `def forward(self, x)`
- Defined: `ablation3.py:62`

### __init__ `def __init__(self, epsilon, alpha, steps)`
- Defined: `ablation3.py:79`

### attack `def attack(self, model_fn, x, y, criterion)`
- Defined: `ablation3.py:84`

### __init__ `def __init__(self, output_dim)`
- Defined: `ablation3.py:110`

### forward `def forward(self, x)`
- Defined: `ablation3.py:122`

### __init__ `def __init__(self, dim)`
- Defined: `ablation3.py:126`

### forward `def forward(self, x)`
- Defined: `ablation3.py:131`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `ablation3.py:137`

### forward `def forward(self, thought, captions, max_len)`
- Defined: `ablation3.py:148`

### _init_state `def _init_state(self, thought)`
- Defined: `ablation3.py:182`

### __init__ `def __init__(self, vocab_size, use_sparse, use_symbiotic, use_adv, n_nodes, k_sparse)`
- Defined: `ablation3.py:191`

### forward `def forward(self, image, captions)`
- Defined: `ablation3.py:226`

### train_step `def train_step(self, images, captions, optimizer, dataset)`
- Defined: `ablation3.py:243`
- Doc: Paso de entrenamiento con adversarial condicional y doble forward pass seguro

### __init__ `def __init__(self)`
- Defined: `ablation3.py:296`

### __len__ `def __len__(self)`
- Defined: `ablation3.py:321`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `ablation3.py:324`

### model_fn `def model_fn(x)`
- Defined: `ablation3.py:257`

## adversarial_benchmark.py

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps)`
- Defined: `adversarial_benchmark.py:269`

### train_and_eval `def train_and_eval()`
- Defined: `adversarial_benchmark.py:290`

### __init__ `def __init__(self, temperature)`
- Defined: `adversarial_benchmark.py:37`

### forward `def forward(self, features, labels)`
- Defined: `adversarial_benchmark.py:41`

### __init__ `def __init__(self, dim)`
- Defined: `adversarial_benchmark.py:74`

### forward `def forward(self, input_signal, prediction)`
- Defined: `adversarial_benchmark.py:79`

### __init__ `def __init__(self, dim)`
- Defined: `adversarial_benchmark.py:86`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `adversarial_benchmark.py:95`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `adversarial_benchmark.py:101`

### forward `def forward(self, x)`
- Defined: `adversarial_benchmark.py:109`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, layer_type)`
- Defined: `adversarial_benchmark.py:119`

### forward `def forward(self, x_nodes, adjacency, incidence)`
- Defined: `adversarial_benchmark.py:136`

### __init__ `def __init__(self, grid_size)`
- Defined: `adversarial_benchmark.py:159`

### _init_grid_topology `def _init_grid_topology(self, N)`
- Defined: `adversarial_benchmark.py:191`

### get_topology `def get_topology(self)`
- Defined: `adversarial_benchmark.py:213`

### calculate_ortho_loss `def calculate_ortho_loss(self)`
- Defined: `adversarial_benchmark.py:225`

### forward `def forward(self, x)`
- Defined: `adversarial_benchmark.py:249`

### lambda_topo `def lambda_topo(epoch)`
- Defined: `adversarial_benchmark.py:317`

### lambda_general `def lambda_general(epoch)`
- Defined: `adversarial_benchmark.py:321`

## apex.py

### seed_everything `def seed_everything(seed)`
- Defined: `apex.py:14`

### train_and_audit `def train_and_audit(name, use_ewc)`
- Defined: `apex.py:160`

### __init__ `def __init__(self)`
- Defined: `apex.py:25`

### get_train_batch `def get_train_batch(self, phase, batch_size)`
- Defined: `apex.py:35`

### __init__ `def __init__(self, d_in, d_out)`
- Defined: `apex.py:51`

### forward `def forward(self, x, gate)`
- Defined: `apex.py:58`

### __init__ `def __init__(self, d_in)`
- Defined: `apex.py:70`

### forward `def forward(self, x, chaos)`
- Defined: `apex.py:74`

### __init__ `def __init__(self, d_in)`
- Defined: `apex.py:79`

### forward `def forward(self, x, p)`
- Defined: `apex.py:82`

### update `def update(self, x, p)`
- Defined: `apex.py:86`

### __init__ `def __init__(self, model, lambda_ewc)`
- Defined: `apex.py:94`

### register_fisher `def register_fisher(self, dataset_x, dataset_y)`
- Defined: `apex.py:101`

### penalty `def penalty(self)`
- Defined: `apex.py:125`

### __init__ `def __init__(self, d_in, d_hid, d_out)`
- Defined: `apex.py:137`

### forward `def forward(self, x, phase)`
- Defined: `apex.py:145`

## app.py

### train_ai_model `def train_ai_model(df)`
- Defined: `app.py:82`
- Doc: Entrena un modelo desde cero con todos los detalles de entrenamiento

### load_or_train_model `def load_or_train_model(df)`
- Defined: `app.py:156`
- Doc: Carga modelo existente o entrena uno nuevo, y lo actualiza con nuevos datos

### apply_ai_predictions `def apply_ai_predictions(df, model, vectorizer)`
- Defined: `app.py:191`
- Doc: Aplica predicciones del modelo al DataFrame

### apply_ai_predictions `def apply_ai_predictions(df, model, vectorizer)`
- Defined: `app.py:205`
- Doc: Aplica predicciones del modelo al DataFrame

### analyze_ia_vs_rules `def analyze_ia_vs_rules(df)`
- Defined: `app.py:219`
- Doc: Analiza discrepancias entre reglas y modelo IA

### load_and_clean_data_robust `def load_and_clean_data_robust(filepath)`
- Defined: `app.py:252`
- Doc: Cargar y limpiar los datos de forma robusta

### parse_csv_manual `def parse_csv_manual(filepath)`
- Defined: `app.py:294`

### executive_kpis `def executive_kpis(df)`
- Defined: `app.py:317`

### strategic_okrs `def strategic_okrs(df, kpis)`
- Defined: `app.py:344`

### generate_visualizations `def generate_visualizations(df, kpis)`
- Defined: `app.py:377`

### export_report `def export_report(df, kpis, okrs, ia_analysis)`
- Defined: `app.py:409`

### basic_statistics `def basic_statistics(df)`
- Defined: `app.py:454`

### command_analysis `def command_analysis(df)`
- Defined: `app.py:467`

### network_analysis `def network_analysis(df)`
- Defined: `app.py:480`

### temporal_analysis `def temporal_analysis(df)`
- Defined: `app.py:492`

### statistical_analysis `def statistical_analysis(df)`
- Defined: `app.py:500`

### security_insights `def security_insights(df)`
- Defined: `app.py:508`

### main `def main()`
- Defined: `app.py:530`

## auto_regulation_working.py

### seed_everything `def seed_everything(seed)`
- Defined: `auto_regulation_working.py:28`

### demo_auto_regulation `def demo_auto_regulation()`
- Defined: `auto_regulation_working.py:193`

### __init__ `def __init__(self)`
- Defined: `auto_regulation_working.py:39`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `auto_regulation_working.py:49`

### get_full `def get_full(self)`
- Defined: `auto_regulation_working.py:63`

### get_w2 `def get_w2(self)`
- Defined: `auto_regulation_working.py:66`

### __init__ `def __init__(self, size)`
- Defined: `auto_regulation_working.py:73`

### update `def update(self, input_variance, loss_gradient, phase)`
- Defined: `auto_regulation_working.py:78`

### get_stability `def get_stability(self)`
- Defined: `auto_regulation_working.py:95`

### __init__ `def __init__(self, config)`
- Defined: `auto_regulation_working.py:104`

### forward `def forward(self, x, global_step, phase, prev_loss)`
- Defined: `auto_regulation_working.py:129`

## bicamera.py.py

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `bicamera.py.py:34`
- Doc: Descarga Flickr8k automáticamente

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `bicamera.py.py:443`

### train_bicameral `def train_bicameral()`
- Defined: `bicamera.py.py:481`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `bicamera.py.py:108`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `bicamera.py.py:125`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `bicamera.py.py:154`

### __init__ `def __init__(self, output_dim)`
- Defined: `bicamera.py.py:181`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `bicamera.py.py:192`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `bicamera.py.py:202`

### forward `def forward(self, visual_context, captions, max_len, return_gate)`
- Defined: `bicamera.py.py:224`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `bicamera.py.py:278`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `bicamera.py.py:283`

### __init__ `def __init__(self, dim)`
- Defined: `bicamera.py.py:299`

### forward `def forward(self, right_features)`
- Defined: `bicamera.py.py:307`

### __init__ `def __init__(self, vocab_size)`
- Defined: `bicamera.py.py:314`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- Defined: `bicamera.py.py:320`

### __init__ `def __init__(self)`
- Defined: `bicamera.py.py:335`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `bicamera.py.py:346`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `bicamera.py.py:353`

### update `def update(self)`
- Defined: `bicamera.py.py:357`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `bicamera.py.py:362`

### report `def report(self, epoch)`
- Defined: `bicamera.py.py:367`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `bicamera.py.py:405`

### __len__ `def __len__(self)`
- Defined: `bicamera.py.py:423`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `bicamera.py.py:426`

### __init__ `def __init__(self, total_epochs)`
- Defined: `bicamera.py.py:467`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `bicamera.py.py:470`

## bicameral.py

### seed_all `def seed_all(seed)`
- Defined: `bicameral.py:38`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `bicameral.py:46`

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `bicameral.py:416`

### train_bicameral_fisiologico `def train_bicameral_fisiologico()`
- Defined: `bicameral.py:447`

### __init__ `def __init__(self)`
- Defined: `bicameral.py:101`

### forward `def forward(self, stress, excitation, fatigue, entropy, phase, loss_signal)`
- Defined: `bicameral.py:109`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `bicameral.py:124`

### forward `def forward(self, x, global_loss)`
- Defined: `bicameral.py:138`

### __init__ `def __init__(self, output_dim, num_nodes)`
- Defined: `bicameral.py:165`

### forward `def forward(self, image, global_loss)`
- Defined: `bicameral.py:178`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `bicameral.py:200`

### forward `def forward(self, visual_context, captions, max_len, return_gate)`
- Defined: `bicameral.py:219`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `bicameral.py:258`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `bicameral.py:263`

### __init__ `def __init__(self, dim)`
- Defined: `bicameral.py:277`

### forward `def forward(self, right_features)`
- Defined: `bicameral.py:284`

### __init__ `def __init__(self, vocab_size)`
- Defined: `bicameral.py:291`

### forward `def forward(self, image, captions, global_loss, return_diagnostics)`
- Defined: `bicameral.py:297`

### __init__ `def __init__(self)`
- Defined: `bicameral.py:310`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `bicameral.py:326`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `bicameral.py:333`

### update `def update(self)`
- Defined: `bicameral.py:337`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `bicameral.py:342`

### report `def report(self, epoch)`
- Defined: `bicameral.py:347`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `bicameral.py:384`

### __len__ `def __len__(self)`
- Defined: `bicameral.py:400`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `bicameral.py:403`

### __init__ `def __init__(self, total_epochs)`
- Defined: `bicameral.py:437`

### get_global_loss_proxy `def get_global_loss_proxy(self, epoch)`
- Defined: `bicameral.py:440`

## bicameral2.py

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `bicameral2.py:32`

### build_vocab `def build_vocab(captions_file, size)`
- Defined: `bicameral2.py:274`

### train_ultra `def train_ultra()`
- Defined: `bicameral2.py:290`

### __init__ `def __init__(self, output_dim)`
- Defined: `bicameral2.py:84`

### forward `def forward(self, x)`
- Defined: `bicameral2.py:98`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `bicameral2.py:105`

### forward `def forward(self, x)`
- Defined: `bicameral2.py:113`

### __init__ `def __init__(self, output_dim)`
- Defined: `bicameral2.py:127`

### forward `def forward(self, x)`
- Defined: `bicameral2.py:131`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `bicameral2.py:137`

### forward `def forward(self, visual_ctx, captions, max_len)`
- Defined: `bicameral2.py:143`

### __init__ `def __init__(self, dim)`
- Defined: `bicameral2.py:177`

### forward `def forward(self, x)`
- Defined: `bicameral2.py:180`

### __init__ `def __init__(self, vocab_size)`
- Defined: `bicameral2.py:187`

### forward `def forward(self, image, captions, return_diagnostics)`
- Defined: `bicameral2.py:192`

### __init__ `def __init__(self)`
- Defined: `bicameral2.py:207`

### measure_flow `def measure_flow(self, r, l)`
- Defined: `bicameral2.py:212`

### vocab_diversity `def vocab_diversity(self, tokens, V)`
- Defined: `bicameral2.py:217`

### update `def update(self)`
- Defined: `bicameral2.py:219`

### avg `def avg(self, k, n)`
- Defined: `bicameral2.py:223`

### report `def report(self, epoch)`
- Defined: `bicameral2.py:226`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `bicameral2.py:247`

### __len__ `def __len__(self)`
- Defined: `bicameral2.py:261`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `bicameral2.py:262`

## bicameral3.py

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `bicameral3.py:34`
- Doc: Descarga Flickr8k automáticamente

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `bicameral3.py:443`

### train_bicameral `def train_bicameral()`
- Defined: `bicameral3.py:481`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `bicameral3.py:108`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `bicameral3.py:125`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `bicameral3.py:154`

### __init__ `def __init__(self, output_dim)`
- Defined: `bicameral3.py:181`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `bicameral3.py:192`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `bicameral3.py:202`

### forward `def forward(self, visual_context, captions, max_len, return_gate)`
- Defined: `bicameral3.py:224`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `bicameral3.py:278`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `bicameral3.py:283`

### __init__ `def __init__(self, dim)`
- Defined: `bicameral3.py:299`

### forward `def forward(self, right_features)`
- Defined: `bicameral3.py:307`

### __init__ `def __init__(self, vocab_size)`
- Defined: `bicameral3.py:314`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- Defined: `bicameral3.py:320`

### __init__ `def __init__(self)`
- Defined: `bicameral3.py:335`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `bicameral3.py:346`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `bicameral3.py:353`

### update `def update(self)`
- Defined: `bicameral3.py:357`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `bicameral3.py:362`

### report `def report(self, epoch)`
- Defined: `bicameral3.py:367`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `bicameral3.py:405`

### __len__ `def __len__(self)`
- Defined: `bicameral3.py:423`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `bicameral3.py:426`

### __init__ `def __init__(self, total_epochs)`
- Defined: `bicameral3.py:467`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `bicameral3.py:470`

## bicameral_v2.py

### compute_phi_effective `def compute_phi_effective(activations, k_partitions)`
- Defined: `bicameral_v2.py:24`
- Doc: Φₑ efectivo: integración causal simplificada para batches

### measure_spatial_richness `def measure_spatial_richness(activations)`
- Defined: `bicameral_v2.py:52`
- Doc: FIX: Métrica de riqueza dimensional efectiva con escalado positivo garantizado

### top_k_top_p_filtering `def top_k_top_p_filtering(logits, top_k, top_p, filter_value)`
- Defined: `bicameral_v2.py:98`
- Doc: Filtro Top-K y Nucleus Sampling estandar

### create_grid_adjacency `def create_grid_adjacency(N, connectivity)`
- Defined: `bicameral_v2.py:327`
- Doc: Crea matriz de adyacencia para grid cuadrado

### estimate_coherence `def estimate_coherence(sentence, templates_per_class)`
- Defined: `bicameral_v2.py:1012`

### train_logos `def train_logos(use_nested)`
- Defined: `bicameral_v2.py:1026`

### __init__ `def __init__(self, neurons, tau_theta)`
- Defined: `bicameral_v2.py:118`

### forward `def forward(self, activity, dt)`
- Defined: `bicameral_v2.py:123`
- Doc: dθ/dt = (E[activity²] - θ)/τ  →  dw/dt ∝ activity*(activity-θ)

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `bicameral_v2.py:135`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `bicameral_v2.py:158`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `bicameral_v2.py:204`
- Doc: Mantener interfaz exacta pero implementar consolidación Hebbiana real

### __init__ `def __init__(self, in_channels, out_channels, stride)`
- Defined: `bicameral_v2.py:226`

### forward `def forward(self, x)`
- Defined: `bicameral_v2.py:238`

### __init__ `def __init__(self, output_dim, grid_size)`
- Defined: `bicameral_v2.py:245`

### _make_layer `def _make_layer(self, in_channels, out_channels, num_blocks, stride)`
- Defined: `bicameral_v2.py:259`

### forward `def forward(self, x)`
- Defined: `bicameral_v2.py:265`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `bicameral_v2.py:281`

### forward `def forward(self, x)`
- Defined: `bicameral_v2.py:291`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config)`
- Defined: `bicameral_v2.py:301`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `bicameral_v2.py:307`

### __init__ `def __init__(self, dim, hidden_dim)`
- Defined: `bicameral_v2.py:312`

### forward `def forward(self, nodes, adjacency)`
- Defined: `bicameral_v2.py:322`

### __init__ `def __init__(self, config)`
- Defined: `bicameral_v2.py:343`

### forward `def forward(self, image, adjacency, plasticity)`
- Defined: `bicameral_v2.py:375`

### __init__ `def __init__(self)`
- Defined: `bicameral_v2.py:407`

### forward `def forward(self, x)`
- Defined: `bicameral_v2.py:420`

### __init__ `def __init__(self, grid_size, output_dim)`
- Defined: `bicameral_v2.py:424`

### forward `def forward(self, x)`
- Defined: `bicameral_v2.py:444`

### __init__ `def __init__(self, node_dim)`
- Defined: `bicameral_v2.py:467`

### forward `def forward(self, nodes, plasticity, transfer_rate)`
- Defined: `bicameral_v2.py:476`

### __init__ `def __init__(self)`
- Defined: `bicameral_v2.py:487`

### forward `def forward(self, visual_features, plasticity, transfer_rate)`
- Defined: `bicameral_v2.py:499`

### get_liquid_module `def get_liquid_module(self)`
- Defined: `bicameral_v2.py:537`

### __init__ `def __init__(self, use_nested)`
- Defined: `bicameral_v2.py:544`

### forward `def forward(self, image, callosal_input, plasticity, transfer_rate)`
- Defined: `bicameral_v2.py:550`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)`
- Defined: `bicameral_v2.py:559`

### forward `def forward(self, thought, visual_features, captions, max_len)`
- Defined: `bicameral_v2.py:578`

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `bicameral_v2.py:659`

### __init__ `def __init__(self)`
- Defined: `bicameral_v2.py:668`

### forward `def forward(self, visual_features, plasticity, transfer_rate)`
- Defined: `bicameral_v2.py:680`

### get_liquid_module `def get_liquid_module(self)`
- Defined: `bicameral_v2.py:718`

### __init__ `def __init__(self)`
- Defined: `bicameral_v2.py:726`

### decide `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- Defined: `bicameral_v2.py:730`

### __init__ `def __init__(self)`
- Defined: `bicameral_v2.py:742`

### decide `def decide(self, left_metrics, right_metrics, epoch, total_epochs)`
- Defined: `bicameral_v2.py:751`

### __init__ `def __init__(self, capacity, noise_scale)`
- Defined: `bicameral_v2.py:774`

### store `def store(self, pattern)`
- Defined: `bicameral_v2.py:780`

### replay `def replay(self, batch_size)`
- Defined: `bicameral_v2.py:791`

### __init__ `def __init__(self)`
- Defined: `bicameral_v2.py:818`

### forward `def forward(self, left_repr, right_repr, mode)`
- Defined: `bicameral_v2.py:824`

### __init__ `def __init__(self, vocab_size, use_nested)`
- Defined: `bicameral_v2.py:869`

### forward `def forward(self, image, captions, plasticity, transfer_rate, mode, epoch)`
- Defined: `bicameral_v2.py:895`

### set_epoch `def set_epoch(self, epoch)`
- Defined: `bicameral_v2.py:950`

### __init__ `def __init__(self, total_epochs)`
- Defined: `bicameral_v2.py:957`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `bicameral_v2.py:961`

### __init__ `def __init__(self)`
- Defined: `bicameral_v2.py:975`

### __len__ `def __len__(self)`
- Defined: `bicameral_v2.py:999`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `bicameral_v2.py:1002`

## bicameral_v3.py

### compute_phi_effective `def compute_phi_effective(activations, k_partitions)`
- Defined: `bicameral_v3.py:22`
- Doc: Φₑ con manejo robusto de dimensiones pequeñas

### compute_spatial_diversity `def compute_spatial_diversity(activations)`
- Defined: `bicameral_v3.py:75`
- Doc: Diversidad basada en correlación inversa de Pearson.

### compute_activation_entropy `def compute_activation_entropy(activations)`
- Defined: `bicameral_v3.py:132`
- Doc: Shannon entropy sobre la distribución de activaciones

### measure_neural_complexity `def measure_neural_complexity(activations)`
- Defined: `bicameral_v3.py:163`
- Doc: Medición corregida con formato [B, D, N] para neuronas reales

### measure_spatial_richness `def measure_spatial_richness(activations)`
- Defined: `bicameral_v3.py:212`
- Doc: Wrapper para compatibilidad con código existente

### top_k_top_p_filtering `def top_k_top_p_filtering(logits, top_k, top_p, filter_value)`
- Defined: `bicameral_v3.py:221`
- Doc: Filtro Top-K y Nucleus Sampling estándar

### create_grid_adjacency `def create_grid_adjacency(N, connectivity)`
- Defined: `bicameral_v3.py:588`
- Doc: Crea matriz de adyacencia para grid cuadrado

### estimate_coherence `def estimate_coherence(sentence, templates_per_class)`
- Defined: `bicameral_v3.py:1336`

### to_float `def to_float(val)`
- Defined: `bicameral_v3.py:1349`

### train_logos `def train_logos(use_nested)`
- Defined: `bicameral_v3.py:1355`

### __init__ `def __init__(self, neurons, tau_theta)`
- Defined: `bicameral_v3.py:242`

### forward `def forward(self, activity, dt)`
- Defined: `bicameral_v3.py:247`
- Doc: dθ/dt = (E[activity²] - θ)/τ  →  dw/dt ∝ activity*(activity-θ)

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)`
- Defined: `bicameral_v3.py:258`

### forward `def forward(self, thought, visual_features, captions, max_len)`
- Defined: `bicameral_v3.py:281`

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `bicameral_v3.py:377`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `bicameral_v3.py:384`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `bicameral_v3.py:405`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `bicameral_v3.py:467`

### __init__ `def __init__(self, in_channels, out_channels, stride)`
- Defined: `bicameral_v3.py:487`

### forward `def forward(self, x)`
- Defined: `bicameral_v3.py:499`

### __init__ `def __init__(self, output_dim, grid_size)`
- Defined: `bicameral_v3.py:506`

### _make_layer `def _make_layer(self, in_channels, out_channels, num_blocks, stride)`
- Defined: `bicameral_v3.py:520`

### forward `def forward(self, x)`
- Defined: `bicameral_v3.py:526`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `bicameral_v3.py:542`

### forward `def forward(self, x)`
- Defined: `bicameral_v3.py:552`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config)`
- Defined: `bicameral_v3.py:562`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `bicameral_v3.py:568`

### __init__ `def __init__(self, dim, hidden_dim)`
- Defined: `bicameral_v3.py:573`

### forward `def forward(self, nodes, adjacency)`
- Defined: `bicameral_v3.py:583`

### __init__ `def __init__(self, config)`
- Defined: `bicameral_v3.py:604`

### forward `def forward(self, image, adjacency, plasticity)`
- Defined: `bicameral_v3.py:636`

### __init__ `def __init__(self)`
- Defined: `bicameral_v3.py:668`

### forward `def forward(self, x)`
- Defined: `bicameral_v3.py:681`

### __init__ `def __init__(self, grid_size, output_dim)`
- Defined: `bicameral_v3.py:685`

### forward `def forward(self, x)`
- Defined: `bicameral_v3.py:705`

### __init__ `def __init__(self, node_dim)`
- Defined: `bicameral_v3.py:728`

### forward `def forward(self, nodes, plasticity, transfer_rate)`
- Defined: `bicameral_v3.py:737`

### __init__ `def __init__(self)`
- Defined: `bicameral_v3.py:745`

### _create_rotation_matrix `def _create_rotation_matrix(self, dim, angle, device)`
- Defined: `bicameral_v3.py:780`
- Doc: Crea matriz de rotación en espacio de alta dimensión

### forward `def forward(self, visual_features, plasticity, transfer_rate)`
- Defined: `bicameral_v3.py:790`

### get_liquid_module `def get_liquid_module(self)`
- Defined: `bicameral_v3.py:839`

### __init__ `def __init__(self, use_nested)`
- Defined: `bicameral_v3.py:847`

### forward `def forward(self, image, callosal_input, plasticity, transfer_rate)`
- Defined: `bicameral_v3.py:853`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)`
- Defined: `bicameral_v3.py:861`

### forward `def forward(self, thought, visual_features, captions, max_len)`
- Defined: `bicameral_v3.py:883`

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `bicameral_v3.py:982`

### __init__ `def __init__(self)`
- Defined: `bicameral_v3.py:992`

### forward `def forward(self, left_repr, right_repr, mode)`
- Defined: `bicameral_v3.py:999`

### __init__ `def __init__(self)`
- Defined: `bicameral_v3.py:1040`

### decide `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- Defined: `bicameral_v3.py:1049`

### __init__ `def __init__(self)`
- Defined: `bicameral_v3.py:1083`

### decide `def decide(self, left_metrics, right_metrics, epoch, total_epochs)`
- Defined: `bicameral_v3.py:1097`

### __init__ `def __init__(self, capacity, noise_scale)`
- Defined: `bicameral_v3.py:1139`

### store `def store(self, pattern)`
- Defined: `bicameral_v3.py:1145`

### replay `def replay(self, batch_size)`
- Defined: `bicameral_v3.py:1155`

### __init__ `def __init__(self, vocab_size, use_nested)`
- Defined: `bicameral_v3.py:1183`

### forward `def forward(self, image, captions, plasticity, transfer_rate, mode, epoch, labels)`
- Defined: `bicameral_v3.py:1203`

### set_epoch `def set_epoch(self, epoch)`
- Defined: `bicameral_v3.py:1274`

### __init__ `def __init__(self, total_epochs)`
- Defined: `bicameral_v3.py:1281`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `bicameral_v3.py:1285`

### __init__ `def __init__(self)`
- Defined: `bicameral_v3.py:1299`

### __len__ `def __len__(self)`
- Defined: `bicameral_v3.py:1323`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `bicameral_v3.py:1326`

## caquita.py

### seed_everything `def seed_everything(seed)`
- Defined: `caquita.py:40`

### train_diagnostic `def train_diagnostic(config, env, experiment_name)`
- Defined: `caquita.py:327`
- Doc: Entrenamiento con logs detallados

### run_diagnostic_ablation `def run_diagnostic_ablation()`
- Defined: `caquita.py:406`

### __init__ `def __init__(self)`
- Defined: `caquita.py:51`

### get_batch `def get_batch(self, phase, batch_size)`
- Defined: `caquita.py:63`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `caquita.py:79`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `caquita.py:88`

### __init__ `def __init__(self)`
- Defined: `caquita.py:105`

### update_phase_performance `def update_phase_performance(self, phase_idx, metrics)`
- Defined: `caquita.py:112`

### detect_trauma_level `def detect_trauma_level(self, phase_idx, current_metrics)`
- Defined: `caquita.py:122`

### generate_response `def generate_response(self, trauma_level, phase_idx, chaos_detected)`
- Defined: `caquita.py:148`

### __init__ `def __init__(self)`
- Defined: `caquita.py:170`

### update_phase_performance `def update_phase_performance(self, phase_idx, metrics)`
- Defined: `caquita.py:176`

### detect_trauma_level `def detect_trauma_level(self, phase_idx, current_metrics)`
- Defined: `caquita.py:185`

### generate_response `def generate_response(self, trauma_level, phase_idx, chaos_detected)`
- Defined: `caquita.py:208`

### __init__ `def __init__(self)`
- Defined: `caquita.py:230`

### extract_noise_features `def extract_noise_features(self, x)`
- Defined: `caquita.py:240`

### detect_chaos `def detect_chaos(self, x)`
- Defined: `caquita.py:253`

### __init__ `def __init__(self, config, use_liquid, use_trs_original, use_trs_fixed, use_caf)`
- Defined: `caquita.py:265`

### forward `def forward(self, x, phase_idx, current_metrics)`
- Defined: `caquita.py:285`

## chatgpt.py

### seed_all `def seed_all(seed)`
- Defined: `chatgpt.py:33`

### train `def train()`
- Defined: `chatgpt.py:225`

### __init__ `def __init__(self)`
- Defined: `chatgpt.py:42`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `chatgpt.py:53`

### __init__ `def __init__(self)`
- Defined: `chatgpt.py:71`

### forward `def forward(self, stress, excitation, fatigue, loss_signal)`
- Defined: `chatgpt.py:81`

### __init__ `def __init__(self, d)`
- Defined: `chatgpt.py:94`

### forward `def forward(self, x, task_loss)`
- Defined: `chatgpt.py:107`

### __init__ `def __init__(self, config)`
- Defined: `chatgpt.py:141`

### count_parameters `def count_parameters(self)`
- Defined: `chatgpt.py:163`

### forward `def forward(self, x, task_loss)`
- Defined: `chatgpt.py:166`

### __init__ `def __init__(self)`
- Defined: `chatgpt.py:192`

### update `def update(self, loss, liquid_norm, phys)`
- Defined: `chatgpt.py:201`

### avg `def avg(self, k, n)`
- Defined: `chatgpt.py:208`

### report `def report(self, step, phase)`
- Defined: `chatgpt.py:211`

## cifar3.py

### compute_phi_effective `def compute_phi_effective(activity)`
- Defined: `cifar3.py:30`

### get_cifar10_loaders `def get_cifar10_loaders(batch_size)`
- Defined: `cifar3.py:245`

### evaluate `def evaluate(model, loader, device)`
- Defined: `cifar3.py:261`

### train `def train()`
- Defined: `cifar3.py:274`

### __init__ `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- Defined: `cifar3.py:56`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `cifar3.py:74`
- Doc: Reinicia los pesos rápidos al inicio de cada batch.

### update_fast_weights `def update_fast_weights(self, x)`
- Defined: `cifar3.py:79`
- Doc: Actualiza fast weights usando regla hebbiana con decay y normalización.

### forward `def forward(self, x)`
- Defined: `cifar3.py:105`

### end_of_batch `def end_of_batch(self)`
- Defined: `cifar3.py:116`
- Doc: Limpia caché al final del batch para permitir reinicio en el siguiente.

### get_fast_norm `def get_fast_norm(self)`
- Defined: `cifar3.py:120`
- Doc: Retorna la norma L2 de los fast weights para monitoreo homeostático.

### __init__ `def __init__(self, dim)`
- Defined: `cifar3.py:130`

### forward `def forward(self, x)`
- Defined: `cifar3.py:138`

### __init__ `def __init__(self, features)`
- Defined: `cifar3.py:149`

### compute_phi_effective_robust `def compute_phi_effective_robust(self, activity)`
- Defined: `cifar3.py:160`
- Doc: Φₑ más robusto usando promedio móvil y ventana temporal.

### forward `def forward(self, x)`
- Defined: `cifar3.py:184`

### __init__ `def __init__(self)`
- Defined: `cifar3.py:194`

### forward `def forward(self, x)`
- Defined: `cifar3.py:226`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `cifar3.py:233`

### get_fast_norms `def get_fast_norms(self)`
- Defined: `cifar3.py:238`

## cifar4.py

### compute_phi_effective `def compute_phi_effective(activity)`
- Defined: `cifar4.py:30`

### get_cifar10_loaders `def get_cifar10_loaders(batch_size)`
- Defined: `cifar4.py:217`

### evaluate `def evaluate(model, loader, device)`
- Defined: `cifar4.py:238`

### train `def train()`
- Defined: `cifar4.py:255`

### __init__ `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- Defined: `cifar4.py:56`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `cifar4.py:72`

### update_fast_weights `def update_fast_weights(self, x)`
- Defined: `cifar4.py:76`

### forward `def forward(self, x)`
- Defined: `cifar4.py:95`

### end_of_batch `def end_of_batch(self)`
- Defined: `cifar4.py:106`

### get_fast_norm `def get_fast_norm(self)`
- Defined: `cifar4.py:109`

### __init__ `def __init__(self, dim)`
- Defined: `cifar4.py:117`

### forward `def forward(self, x)`
- Defined: `cifar4.py:125`

### __init__ `def __init__(self, features)`
- Defined: `cifar4.py:136`

### compute_phi_effective_robust `def compute_phi_effective_robust(self, activity)`
- Defined: `cifar4.py:147`

### forward `def forward(self, x)`
- Defined: `cifar4.py:166`

### __init__ `def __init__(self)`
- Defined: `cifar4.py:177`

### forward `def forward(self, x)`
- Defined: `cifar4.py:197`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `cifar4.py:204`

### get_fast_norms `def get_fast_norms(self)`
- Defined: `cifar4.py:209`

## demo_auto_regulation.py

### seed_everything `def seed_everything(seed)`
- Defined: `demo_auto_regulation.py:33`

### demo_auto_regulation `def demo_auto_regulation()`
- Defined: `demo_auto_regulation.py:243`

### __init__ `def __init__(self)`
- Defined: `demo_auto_regulation.py:44`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `demo_auto_regulation.py:54`

### get_full `def get_full(self)`
- Defined: `demo_auto_regulation.py:68`

### get_w2 `def get_w2(self)`
- Defined: `demo_auto_regulation.py:71`

### __init__ `def __init__(self, size)`
- Defined: `demo_auto_regulation.py:78`

### update `def update(self, input_variance, loss_gradient, phase)`
- Defined: `demo_auto_regulation.py:84`

### get_stability `def get_stability(self)`
- Defined: `demo_auto_regulation.py:99`

### __init__ `def __init__(self, input_dim, hidden_dim)`
- Defined: `demo_auto_regulation.py:109`

### forward `def forward(self, x, adaptation_state)`
- Defined: `demo_auto_regulation.py:126`

### __init__ `def __init__(self, config)`
- Defined: `demo_auto_regulation.py:160`

### forward `def forward(self, x, global_step, phase, prev_loss)`
- Defined: `demo_auto_regulation.py:184`

## difract.py

### visualize_uased_geometry `def visualize_uased_geometry()`
- Defined: `difract.py:4`

## dmg_core.py

### __init__ `def __init__(self, base_threshold, power_order)`
- Defined: `dmg_core.py:21`

### forward `def forward(self, x)`
- Defined: `dmg_core.py:30`

### __init__ `def __init__(self, in_features, out_features, sparsity_k)`
- Defined: `dmg_core.py:49`

### _generate_sparse_mask `def _generate_sparse_mask(self, k_neighbors)`
- Defined: `dmg_core.py:62`
- Doc: Generates a Barabási-Albert scale-free mask.

### forward `def forward(self, x)`
- Defined: `dmg_core.py:77`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `dmg_core.py:87`

### forward `def forward(self, x)`
- Defined: `dmg_core.py:100`

## dualmind.py

### measure_spatial_richness `def measure_spatial_richness(activations)`
- Defined: `dualmind.py:15`
- Doc: Mide diversidad de representaciones mediante eigenspectro

### train_dualmind_phase1 `def train_dualmind_phase1(model, train_loader, optimizer, device, epochs)`
- Defined: `dualmind.py:346`
- Doc: FASE 1: Preentrenamiento del sistema inconsciente

### train_dualmind_phase2 `def train_dualmind_phase2(model, train_loader, optimizer, device, epochs)`
- Defined: `dualmind.py:401`
- Doc: FASE 2: Entrenamiento del sistema consciente

### train_dualmind_phase3 `def train_dualmind_phase3(model, train_loader, optimizer, device, epochs)`
- Defined: `dualmind.py:487`
- Doc: FASE 3: Co-adaptación de ambos sistemas

### evaluate_dualmind `def evaluate_dualmind(model, test_loader, device)`
- Defined: `dualmind.py:579`
- Doc: Evaluación del sistema dual

### run_dualmind_experiment `def run_dualmind_experiment()`
- Defined: `dualmind.py:604`

### __init__ `def __init__(self)`
- Defined: `dualmind.py:31`

### decide `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- Defined: `dualmind.py:35`
- Doc: Motor de decisión homeostática con targets realistas y pesos equilibrados.

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `dualmind.py:59`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `dualmind.py:67`
- Doc: Neurona con plasticidad hebbiana de fast weights y decaimiento activación.

### consolidate_svd `def consolidate_svd(self, repair_strength)`
- Defined: `dualmind.py:89`
- Doc: Consolidación mediante SVD (modo sueño)

### __init__ `def __init__(self, unconscious_dim, d_hid, d_out)`
- Defined: `dualmind.py:113`

### forward `def forward(self, unconscious_features, plasticity_gate)`
- Defined: `dualmind.py:136`
- Doc: Input: Representaciones del sistema inconsciente [batch, unconscious_dim]

### get_structure_entropy `def get_structure_entropy(self)`
- Defined: `dualmind.py:157`
- Doc: Análisis de salud estructural mediante SVD

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes)`
- Defined: `dualmind.py:176`

### forward `def forward(self, x_nodes, plasticity_gate)`
- Defined: `dualmind.py:188`
- Doc: x_nodes: [batch, num_nodes, in_dim]

### get_topology_density `def get_topology_density(self)`
- Defined: `dualmind.py:209`
- Doc: Densidad de conexiones topológicas

### __init__ `def __init__(self, in_channels, grid_size, hidden_dim)`
- Defined: `dualmind.py:222`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `dualmind.py:246`
- Doc: x: [batch, 3, 32, 32]

### get_topology_stats `def get_topology_stats(self)`
- Defined: `dualmind.py:264`
- Doc: Estadísticas de topología del sistema inconsciente

### __init__ `def __init__(self, in_channels, grid_size, hidden_dim, conscious_dim, num_classes)`
- Defined: `dualmind.py:285`

### forward `def forward(self, x, mode)`
- Defined: `dualmind.py:305`
- Doc: Modos de operación:

### get_system_status `def get_system_status(self)`
- Defined: `dualmind.py:331`
- Doc: Diagnóstico completo del sistema dual

## dynamic.py

### seed_everything `def seed_everything(seed)`
- Defined: `dynamic.py:13`

### run_final_showdown `def run_final_showdown(epochs, name, dynamic)`
- Defined: `dynamic.py:184`

### __init__ `def __init__(self)`
- Defined: `dynamic.py:37`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `dynamic.py:46`

### __init__ `def __init__(self)`
- Defined: `dynamic.py:55`

### forward `def forward(self, x, h_slow)`
- Defined: `dynamic.py:66`

### __init__ `def __init__(self, d_in)`
- Defined: `dynamic.py:89`

### forward `def forward(self, x, gain)`
- Defined: `dynamic.py:94`

### __init__ `def __init__(self, d_in, d_out)`
- Defined: `dynamic.py:101`

### forward `def forward(self, x, plasticity, alpha)`
- Defined: `dynamic.py:113`

### __init__ `def __init__(self, config, dynamic_mode)`
- Defined: `dynamic.py:138`

### forward `def forward(self, x)`
- Defined: `dynamic.py:151`

## dynamic2.py

### seed_everything `def seed_everything(seed)`
- Defined: `dynamic2.py:13`

### run_physio_experiment `def run_physio_experiment(epochs, name, dynamic)`
- Defined: `dynamic2.py:178`

### __init__ `def __init__(self)`
- Defined: `dynamic2.py:37`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `dynamic2.py:46`

### __init__ `def __init__(self, d_in)`
- Defined: `dynamic2.py:55`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `dynamic2.py:66`

### __init__ `def __init__(self, d_in, d_out, dynamic_mode)`
- Defined: `dynamic2.py:90`

### forward `def forward(self, x)`
- Defined: `dynamic2.py:105`

### __init__ `def __init__(self, config, dynamic_mode)`
- Defined: `dynamic2.py:163`

### forward `def forward(self, x)`
- Defined: `dynamic2.py:169`

## example_usage.py

### demo_simple_monitoring `def demo_simple_monitoring()`
- Defined: `example_usage.py:20`
- Doc: Demostración de monitoreo básico
- Depends on: `physio_chimera_v15_monitored.py`

### demo_custom_monitoring `def demo_custom_monitoring()`
- Defined: `example_usage.py:38`
- Doc: Demostración de monitoreo personalizado
- Depends on: `physio_chimera_v15_monitored.py`

### demo_checkpoint_system `def demo_checkpoint_system()`
- Defined: `example_usage.py:85`
- Doc: Demostración del sistema de checkpointing
- Depends on: `physio_chimera_v15_monitored.py`

### demo_comparison_experiments `def demo_comparison_experiments()`
- Defined: `example_usage.py:142`
- Doc: Demostración de comparación entre experimentos
- Depends on: `physio_chimera_v15_monitored.py`

### create_demo_report `def create_demo_report()`
- Defined: `example_usage.py:194`
- Doc: Crear reporte demo completo
- Depends on: `physio_chimera_v15_monitored.py`

### main `def main()`
- Defined: `example_usage.py:333`
- Doc: Función principal de demostración
- Depends on: `physio_chimera_v15_monitored.py`

## exampleww.py

### run_single_experiment `def run_single_experiment(model_name, seed, epochs)`
- Defined: `exampleww.py:8`

## exodia_op_2.py

### preprocess_and_cache_spectrograms `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)`
- Defined: `exodia_op_2.py:49`
- Doc: Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt

### apply_emergency_fixes `def apply_emergency_fixes(model)`
- Defined: `exodia_op_2.py:120`

### setup_flickr8k_with_audio `def setup_flickr8k_with_audio(data_dir)`
- Defined: `exodia_op_2.py:142`
- Doc: Descarga y organiza Flickr8k + Audio del dataset de Kaggle.

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `exodia_op_2.py:316`
- Doc: Construye vocabulario desde el archivo de captions

### forward `def forward(self, image, audio, captions, epoch)`
- Defined: `exodia_op_2.py:1335`

### compute_alignment_loss `def compute_alignment_loss(visual_features, channels, alpha, epoch)`
- Defined: `exodia_op_2.py:2666`
- Doc: FIX: Pérdida auxiliar para alineación temprana de canales multimodales

### compute_tricameral_loss `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, channels, epoch, lambda_reward, lambda_mtp)`
- Defined: `exodia_op_2.py:2695`

### train_tricameral `def train_tricameral()`
- Defined: `exodia_op_2.py:2812`

### __init__ `def __init__(self, working_capacity, short_term_capacity, importance_threshold)`
- Defined: `exodia_op_2.py:347`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `exodia_op_2.py:371`
- Doc: FIX: Clamp de cross-entropy para evitar infinitos

### calculate_importance `def calculate_importance(self, episode, surprise_score)`
- Defined: `exodia_op_2.py:386`
- Doc: FIX: Clamp de surprise_score para evitar probabilidades degeneradas

### _calculate_novelty `def _calculate_novelty(self, episode)`
- Defined: `exodia_op_2.py:402`
- Doc: FIX: Manejo de edge case cuando no hay memorias

### store_episode `def store_episode(self, image, audio, caption, surprise_score)`
- Defined: `exodia_op_2.py:427`

### _update_unified_buffer `def _update_unified_buffer(self)`
- Defined: `exodia_op_2.py:456`
- Doc: FIX: Verificar integridad de scores antes de unificar

### sample `def sample(self, batch_size, memory_level)`
- Defined: `exodia_op_2.py:470`
- Doc: FIX: Manejo de edge cases en sampling probabilístico

### _sample_from_buffer `def _sample_from_buffer(self, buffer, scores, batch_size)`
- Defined: `exodia_op_2.py:494`
- Doc: FIX: Estabilización completa de probabilidades de sampling

### apply_forgetting_curve `def apply_forgetting_curve(self)`
- Defined: `exodia_op_2.py:535`

### _purge_low_score_memories `def _purge_low_score_memories(self)`
- Defined: `exodia_op_2.py:545`
- Doc: FIX: Purga con threshold ajustado y verificación de scores

### __init__ `def __init__(self)`
- Defined: `exodia_op_2.py:578`

### assess_reasoning_state `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)`
- Defined: `exodia_op_2.py:598`
- Doc: Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)

### assess_cognitive_state `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- Defined: `exodia_op_2.py:642`
- Doc: Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)

### apply_cognitive_intervention `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)`
- Defined: `exodia_op_2.py:688`
- Doc: Aplica intervenciones basadas en estado lingüístico y de razonamiento

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `exodia_op_2.py:778`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `exodia_op_2.py:812`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `exodia_op_2.py:821`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `exodia_op_2.py:834`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self, alpha, beta)`
- Defined: `exodia_op_2.py:849`

### _get_ngrams_cached `def _get_ngrams_cached(sentence, n)`
- Defined: `exodia_op_2.py:863`
- Doc: FIX: Método estático con lru_cache para n-gramas

### compute_linguistic_reward `def compute_linguistic_reward(self, references, hypotheses)`
- Defined: `exodia_op_2.py:872`

### compute_cider `def compute_cider(self, reference, hypothesis)`
- Defined: `exodia_op_2.py:911`
- Doc: FIX: Uso correcto del cache estático

### compute_spice `def compute_spice(self, reference, hypothesis)`
- Defined: `exodia_op_2.py:925`

### get_cache_stats `def get_cache_stats(self)`
- Defined: `exodia_op_2.py:937`
- Doc: FIX: Estadísticas de cache actualizadas

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `exodia_op_2.py:967`

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `exodia_op_2.py:990`

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `exodia_op_2.py:1000`

### __init__ `def __init__(self, hidden_dim)`
- Defined: `exodia_op_2.py:1009`

### reason_causally `def reason_causally(self, observation, context)`
- Defined: `exodia_op_2.py:1036`

### _predict_interventions `def _predict_interventions(self, hypothesis, confidence)`
- Defined: `exodia_op_2.py:1050`

### update_knowledge_graph `def update_knowledge_graph(self, cause, effect, strength)`
- Defined: `exodia_op_2.py:1067`

### query_causal_chain `def query_causal_chain(self, start_node, end_node)`
- Defined: `exodia_op_2.py:1073`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `exodia_op_2.py:1089`

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `exodia_op_2.py:1112`

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `exodia_op_2.py:1122`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `exodia_op_2.py:1136`

### forward `def forward(self, x)`
- Defined: `exodia_op_2.py:1184`

### _calculate_homeostasis_metric `def _calculate_homeostasis_metric(self, output)`
- Defined: `exodia_op_2.py:1219`
- Doc: Calcula métrica de homeostasis con estabilización numérica

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `exodia_op_2.py:1229`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `exodia_op_2.py:1278`

### __init__ `def __init__(self)`
- Defined: `exodia_op_2.py:1361`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `exodia_op_2.py:1368`

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `exodia_op_2.py:1379`

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)`
- Defined: `exodia_op_2.py:1382`

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `exodia_op_2.py:1427`

### _reset_liquid_neuron `def _reset_liquid_neuron(self, liquid_neuron)`
- Defined: `exodia_op_2.py:1496`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `exodia_op_2.py:1511`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `exodia_op_2.py:1596`

### _apply_chain_of_thought `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- Defined: `exodia_op_2.py:1653`

### _apply_multi_token_prediction `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- Defined: `exodia_op_2.py:1694`

### _apply_structural_attention `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- Defined: `exodia_op_2.py:1737`

### _greedy_decode `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- Defined: `exodia_op_2.py:1759`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `exodia_op_2.py:1819`

### __init__ `def __init__(self, output_dim)`
- Defined: `exodia_op_2.py:1842`

### forward `def forward(self, mel_spec)`
- Defined: `exodia_op_2.py:1880`

### __init__ `def __init__(self, output_dim)`
- Defined: `exodia_op_2.py:1909`

### forward `def forward(self, image, audio)`
- Defined: `exodia_op_2.py:1947`

### __init__ `def __init__(self, dim)`
- Defined: `exodia_op_2.py:1994`

### _apply_flash_attention `def _apply_flash_attention(self, x)`
- Defined: `exodia_op_2.py:2057`
- Doc: Aplica Flash Attention nativa de PyTorch 2.0+

### forward `def forward(self, right_features)`
- Defined: `exodia_op_2.py:2090`
- Doc: FIX: Manejo robusto de dimensiones y verificación de coherencia trimodal

### update_channel_fatigue `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- Defined: `exodia_op_2.py:2169`

### adjust_gates_by_fatigue `def adjust_gates_by_fatigue(self)`
- Defined: `exodia_op_2.py:2190`
- Doc: Lógica original de ajuste de gates

### __init__ `def __init__(self)`
- Defined: `exodia_op_2.py:2208`

### _get_cached_norm `def _get_cached_norm(self, tensor, dim)`
- Defined: `exodia_op_2.py:2231`
- Doc: Cache de normalización con limpieza periódica

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context, channels)`
- Defined: `exodia_op_2.py:2248`
- Doc: Medición de coherencia multimodal con sincronización entre canales

### evaluate_reasoning_quality `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- Defined: `exodia_op_2.py:2305`

### calculate_synergy `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `exodia_op_2.py:2342`

### calculate_health `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `exodia_op_2.py:2353`

### update `def update(self)`
- Defined: `exodia_op_2.py:2362`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `exodia_op_2.py:2379`

### visualize_fatigue_distribution `def visualize_fatigue_distribution(self, epoch)`
- Defined: `exodia_op_2.py:2395`

### visualize_reasoning_metrics `def visualize_reasoning_metrics(self, epoch)`
- Defined: `exodia_op_2.py:2417`

### report `def report(self, epoch)`
- Defined: `exodia_op_2.py:2429`

### __init__ `def __init__(self, vocab_size)`
- Defined: `exodia_op_2.py:2518`

### forward `def forward(self, image, audio, captions, epoch)`
- Defined: `exodia_op_2.py:2525`

### __init__ `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache, cache_dir)`
- Defined: `exodia_op_2.py:2553`

### __len__ `def __len__(self)`
- Defined: `exodia_op_2.py:2610`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `exodia_op_2.py:2613`

## exodia_optimized.py

### preprocess_and_cache_spectrograms `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)`
- Defined: `exodia_optimized.py:47`
- Doc: Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt

### setup_flickr8k_with_audio `def setup_flickr8k_with_audio(data_dir)`
- Defined: `exodia_optimized.py:120`
- Doc: Descarga y organiza Flickr8k + Audio del dataset de Kaggle.

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `exodia_optimized.py:308`
- Doc: Construye vocabulario desde el archivo de captions

### compute_alignment_loss `def compute_alignment_loss(visual_features, channels, alpha, epoch)`
- Defined: `exodia_optimized.py:2549`
- Doc: FIX: Pérdida auxiliar para alineación temprana de canales multimodales

### compute_tricameral_loss `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, channels, epoch, lambda_reward, lambda_mtp)`
- Defined: `exodia_optimized.py:2577`
- Doc: FIX: Pérdida con término explícito de coherencia multimodal

### train_tricameral `def train_tricameral()`
- Defined: `exodia_optimized.py:2650`

### __init__ `def __init__(self, working_capacity, short_term_capacity, importance_threshold)`
- Defined: `exodia_optimized.py:341`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `exodia_optimized.py:369`
- Doc: Sin cambios en la lógica

### calculate_importance `def calculate_importance(self, episode, surprise_score)`
- Defined: `exodia_optimized.py:380`
- Doc: Sin cambios

### _calculate_novelty `def _calculate_novelty(self, episode)`
- Defined: `exodia_optimized.py:393`
- Doc: Usa solo working/short_term (no long_term)

### store_episode `def store_episode(self, image, audio, caption, surprise_score)`
- Defined: `exodia_optimized.py:416`

### _update_unified_buffer `def _update_unified_buffer(self)`
- Defined: `exodia_optimized.py:445`
- Doc: Solo working + short_term

### add `def add(self, image, audio, caption, surprise_score)`
- Defined: `exodia_optimized.py:450`
- Doc: Alias para store_episode

### apply_forgetting_curve `def apply_forgetting_curve(self)`
- Defined: `exodia_optimized.py:454`
- Doc: Decay más agresivo (ahorro overhead)

### _purge_low_score_memories `def _purge_low_score_memories(self)`
- Defined: `exodia_optimized.py:467`
- Doc: Purga más agresiva (threshold mayor)

### sample `def sample(self, batch_size, memory_level)`
- Defined: `exodia_optimized.py:487`
- Doc: Muestreo solo de working/short_term

### _sample_from_buffer `def _sample_from_buffer(self, buffer, scores, batch_size)`
- Defined: `exodia_optimized.py:511`
- Doc: Lógica original sin cambios

### get_total_size `def get_total_size(self)`
- Defined: `exodia_optimized.py:539`
- Doc: Solo working + short_term

### __init__ `def __init__(self)`
- Defined: `exodia_optimized.py:547`

### assess_reasoning_state `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)`
- Defined: `exodia_optimized.py:567`
- Doc: Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)

### assess_cognitive_state `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- Defined: `exodia_optimized.py:611`
- Doc: Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)

### apply_cognitive_intervention `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)`
- Defined: `exodia_optimized.py:657`
- Doc: Aplica intervenciones basadas en estado lingüístico y de razonamiento

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `exodia_optimized.py:747`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `exodia_optimized.py:781`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `exodia_optimized.py:790`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `exodia_optimized.py:803`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self, alpha, beta)`
- Defined: `exodia_optimized.py:818`

### _get_ngrams_cached `def _get_ngrams_cached(sentence, n)`
- Defined: `exodia_optimized.py:832`
- Doc: FIX: Método estático con lru_cache para n-gramas

### compute_linguistic_reward `def compute_linguistic_reward(self, references, hypotheses)`
- Defined: `exodia_optimized.py:841`

### compute_cider `def compute_cider(self, reference, hypothesis)`
- Defined: `exodia_optimized.py:880`
- Doc: FIX: Uso correcto del cache estático

### compute_spice `def compute_spice(self, reference, hypothesis)`
- Defined: `exodia_optimized.py:894`

### get_cache_stats `def get_cache_stats(self)`
- Defined: `exodia_optimized.py:906`
- Doc: FIX: Estadísticas de cache actualizadas

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `exodia_optimized.py:936`

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `exodia_optimized.py:959`

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `exodia_optimized.py:969`

### __init__ `def __init__(self, hidden_dim)`
- Defined: `exodia_optimized.py:978`

### reason_causally `def reason_causally(self, observation, context)`
- Defined: `exodia_optimized.py:1005`

### _predict_interventions `def _predict_interventions(self, hypothesis, confidence)`
- Defined: `exodia_optimized.py:1019`

### update_knowledge_graph `def update_knowledge_graph(self, cause, effect, strength)`
- Defined: `exodia_optimized.py:1036`

### query_causal_chain `def query_causal_chain(self, start_node, end_node)`
- Defined: `exodia_optimized.py:1042`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `exodia_optimized.py:1058`

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `exodia_optimized.py:1081`

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `exodia_optimized.py:1091`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `exodia_optimized.py:1104`

### forward `def forward(self, x)`
- Defined: `exodia_optimized.py:1146`

### _calculate_homeostasis_metric `def _calculate_homeostasis_metric(self, output)`
- Defined: `exodia_optimized.py:1163`
- Doc: Calcula métrica de homeostasis basada en la estabilidad del output

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `exodia_optimized.py:1172`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `exodia_optimized.py:1210`

### __init__ `def __init__(self)`
- Defined: `exodia_optimized.py:1244`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `exodia_optimized.py:1251`

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `exodia_optimized.py:1262`

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)`
- Defined: `exodia_optimized.py:1265`

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `exodia_optimized.py:1310`

### _reset_liquid_neuron `def _reset_liquid_neuron(self, liquid_neuron)`
- Defined: `exodia_optimized.py:1379`
- Doc: Reset completo de una neurona líquida

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `exodia_optimized.py:1395`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `exodia_optimized.py:1477`

### _apply_chain_of_thought `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- Defined: `exodia_optimized.py:1524`

### _greedy_decode `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- Defined: `exodia_optimized.py:1564`

### _apply_multi_token_prediction `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- Defined: `exodia_optimized.py:1625`

### _apply_structural_attention `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- Defined: `exodia_optimized.py:1667`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `exodia_optimized.py:1688`

### __init__ `def __init__(self, output_dim)`
- Defined: `exodia_optimized.py:1711`

### forward `def forward(self, mel_spec)`
- Defined: `exodia_optimized.py:1751`

### __init__ `def __init__(self, output_dim)`
- Defined: `exodia_optimized.py:1786`

### forward `def forward(self, image, audio)`
- Defined: `exodia_optimized.py:1824`

### __init__ `def __init__(self, dim)`
- Defined: `exodia_optimized.py:1871`

### _apply_flash_attention `def _apply_flash_attention(self, x)`
- Defined: `exodia_optimized.py:1940`
- Doc: Aplica Flash Attention nativa de PyTorch 2.0+

### forward `def forward(self, right_features)`
- Defined: `exodia_optimized.py:1966`

### update_channel_fatigue `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- Defined: `exodia_optimized.py:2054`
- Doc: Lógica original de fatiga sin cambios

### adjust_gates_by_fatigue `def adjust_gates_by_fatigue(self)`
- Defined: `exodia_optimized.py:2076`
- Doc: Lógica original de ajuste de gates

### __init__ `def __init__(self)`
- Defined: `exodia_optimized.py:2094`

### _get_cached_norm `def _get_cached_norm(self, tensor, dim)`
- Defined: `exodia_optimized.py:2117`
- Doc: Cache de normalización con limpieza periódica

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context, channels)`
- Defined: `exodia_optimized.py:2135`
- Doc: FIX: Medición de coherencia multimodal real con atención a diversidad

### __init__ `def __init__(self, vocab_size)`
- Defined: `exodia_optimized.py:2401`

### forward `def forward(self, image, audio, captions, epoch)`
- Defined: `exodia_optimized.py:2407`

### __init__ `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache, cache_dir)`
- Defined: `exodia_optimized.py:2436`

### __len__ `def __len__(self)`
- Defined: `exodia_optimized.py:2493`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `exodia_optimized.py:2496`

### evaluate_reasoning_quality `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- Defined: `exodia_optimized.py:2188`

### calculate_synergy `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `exodia_optimized.py:2225`

### calculate_health `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `exodia_optimized.py:2236`

### update `def update(self)`
- Defined: `exodia_optimized.py:2245`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `exodia_optimized.py:2262`

### visualize_fatigue_distribution `def visualize_fatigue_distribution(self, epoch)`
- Defined: `exodia_optimized.py:2278`

### visualize_reasoning_metrics `def visualize_reasoning_metrics(self, epoch)`
- Defined: `exodia_optimized.py:2302`

### report `def report(self, epoch)`
- Defined: `exodia_optimized.py:2314`

## final_sinergy_analysis.py

### main `def main()`
- Defined: `final_sinergy_analysis.py:219`

### __init__ `def __init__(self)`
- Defined: `final_sinergy_analysis.py:13`

### print_header `def print_header(self)`
- Defined: `final_sinergy_analysis.py:80`

### analyze_original_models `def analyze_original_models(self)`
- Defined: `final_sinergy_analysis.py:87`

### analyze_sinergies `def analyze_sinergies(self)`
- Defined: `final_sinergy_analysis.py:99`

### generate_scientific_matrix `def generate_scientific_matrix(self)`
- Defined: `final_sinergy_analysis.py:118`

### calculate_synergy_breakthrough `def calculate_synergy_breakthrough(self)`
- Defined: `final_sinergy_analysis.py:136`

### generate_conclusion `def generate_conclusion(self)`
- Defined: `final_sinergy_analysis.py:171`

### save_results `def save_results(self)`
- Defined: `final_sinergy_analysis.py:200`

## gemini.py

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `gemini.py:40`
- Doc: Descarga Flickr8k automáticamente

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `gemini.py:470`

### train_bicameral `def train_bicameral()`
- Defined: `gemini.py:509`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `gemini.py:109`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `gemini.py:123`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `gemini.py:154`

### __init__ `def __init__(self, output_dim)`
- Defined: `gemini.py:178`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `gemini.py:187`

### __init__ `def __init__(self, dim)`
- Defined: `gemini.py:197`

### forward `def forward(self, right_features)`
- Defined: `gemini.py:210`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `gemini.py:220`

### forward `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)`
- Defined: `gemini.py:241`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `gemini.py:313`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `gemini.py:318`

### __init__ `def __init__(self, vocab_size)`
- Defined: `gemini.py:334`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- Defined: `gemini.py:340`

### __init__ `def __init__(self)`
- Defined: `gemini.py:362`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `gemini.py:373`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `gemini.py:380`

### update `def update(self)`
- Defined: `gemini.py:384`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `gemini.py:389`

### report `def report(self, epoch)`
- Defined: `gemini.py:394`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `gemini.py:432`

### __len__ `def __len__(self)`
- Defined: `gemini.py:450`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `gemini.py:453`

### __init__ `def __init__(self, total_epochs)`
- Defined: `gemini.py:494`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `gemini.py:497`

## gemini2.py

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `gemini2.py:40`
- Doc: Descarga Flickr8k automáticamente

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `gemini2.py:470`

### train_bicameral `def train_bicameral()`
- Defined: `gemini2.py:509`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `gemini2.py:109`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `gemini2.py:123`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `gemini2.py:154`

### __init__ `def __init__(self, output_dim)`
- Defined: `gemini2.py:178`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `gemini2.py:187`

### __init__ `def __init__(self, dim)`
- Defined: `gemini2.py:197`

### forward `def forward(self, right_features)`
- Defined: `gemini2.py:210`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `gemini2.py:220`

### forward `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)`
- Defined: `gemini2.py:241`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `gemini2.py:313`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `gemini2.py:318`

### __init__ `def __init__(self, vocab_size)`
- Defined: `gemini2.py:334`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- Defined: `gemini2.py:340`

### __init__ `def __init__(self)`
- Defined: `gemini2.py:362`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `gemini2.py:373`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `gemini2.py:380`

### update `def update(self)`
- Defined: `gemini2.py:384`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `gemini2.py:389`

### report `def report(self, epoch)`
- Defined: `gemini2.py:394`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `gemini2.py:432`

### __len__ `def __len__(self)`
- Defined: `gemini2.py:450`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `gemini2.py:453`

### __init__ `def __init__(self, total_epochs)`
- Defined: `gemini2.py:494`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `gemini2.py:497`

## gen_dataset.py

### generate_one `def generate_one(key, text)`
- Defined: `gen_dataset.py:89`

### main `def main()`
- Defined: `gen_dataset.py:110`

## get_dataset.py

### download_captions_only `def download_captions_only()`
- Defined: `get_dataset.py:29`
- Doc: Descarga solo los captions de Flickr8k

### generate_one_audio `def generate_one_audio(text, output_path, max_retries)`
- Defined: `get_dataset.py:67`
- Doc: Genera un audio con retry y rate limiting

### load_checkpoint `def load_checkpoint()`
- Defined: `get_dataset.py:114`
- Doc: Carga el checkpoint de progreso

### save_checkpoint `def save_checkpoint(checkpoint)`
- Defined: `get_dataset.py:122`
- Doc: Guarda el checkpoint de progreso

### generate_audios_with_checkpoints `def generate_audios_with_checkpoints()`
- Defined: `get_dataset.py:128`
- Doc: Genera audios con checkpoints cada 500 archivos

### generate_audios_sync `def generate_audios_sync()`
- Defined: `get_dataset.py:219`
- Doc: Wrapper síncrono con manejo de event loop

### compress_audios_only `def compress_audios_only()`
- Defined: `get_dataset.py:249`
- Doc: Comprime solo los audios en zips pequeños

### create_audio_readme `def create_audio_readme(output_dir, metadata)`
- Defined: `get_dataset.py:318`
- Doc: Crea README para el dataset de audios

### upload_to_huggingface `def upload_to_huggingface(dataset_dir)`
- Defined: `get_dataset.py:388`
- Doc: Sube solo audios a Hugging Face

### main `def main()`
- Defined: `get_dataset.py:465`

### download_flickr8k `def download_flickr8k()`
- Defined: `get_dataset.py:540`
- Doc: Descarga Flickr8k (solo necesitas ejecutar esto una vez)

### generate_audios `def generate_audios()`
- Defined: `get_dataset.py:602`
- Doc: Genera audios con Edge-TTS - TOMA TIEMPO (~20-30 min)

### create_split_zips `def create_split_zips()`
- Defined: `get_dataset.py:637`
- Doc: Crea múltiples zips pequeños para cumplir límites de GitHub

### generate_upload_instructions `def generate_upload_instructions(metadata)`
- Defined: `get_dataset.py:717`
- Doc: Genera instrucciones para subir a GitHub

### upload_to_huggingface `def upload_to_huggingface(dataset_dir)`
- Defined: `get_dataset.py:834`
- Doc: Sube directamente a Hugging Face (alternativa a GitHub)

### main `def main()`
- Defined: `get_dataset.py:888`

## homeostatichope.py

### setup_device `def setup_device()`
- Defined: `homeostatichope.py:35`

### set_seed `def set_seed(seed)`
- Defined: `homeostatichope.py:40`

### pgd_attack `def pgd_attack(model, x, y, epsilon, steps, device, signals)`
- Defined: `homeostatichope.py:368`

### run_conscious_experiment `def run_conscious_experiment(config, device)`
- Defined: `homeostatichope.py:516`

### run_ablation `def run_ablation(device)`
- Defined: `homeostatichope.py:610`

### __init__ `def __init__(self, seed)`
- Defined: `homeostatichope.py:50`

### get_batch `def get_batch(self, phase, batch_size)`
- Defined: `homeostatichope.py:73`

### get_test_loader `def get_test_loader(self, batch_size)`
- Defined: `homeostatichope.py:88`

### __init__ `def __init__(self, d_model)`
- Defined: `homeostatichope.py:102`

### forward `def forward(self, signals)`
- Defined: `homeostatichope.py:128`
- Doc: Args:

### __init__ `def __init__(self, d_model)`
- Defined: `homeostatichope.py:193`

### forward `def forward(self, x, controls)`
- Defined: `homeostatichope.py:203`

### __init__ `def __init__(self, d_model, hidden_dim)`
- Defined: `homeostatichope.py:234`

### forward `def forward(self, x, controls)`
- Defined: `homeostatichope.py:251`

### __init__ `def __init__(self, frequencies, d_model, hidden_dim)`
- Defined: `homeostatichope.py:279`

### forward `def forward(self, x, global_step)`
- Defined: `homeostatichope.py:293`

### __init__ `def __init__(self, config, n_features, n_classes)`
- Defined: `homeostatichope.py:305`

### forward `def forward(self, x, signals, global_step)`
- Defined: `homeostatichope.py:341`

### __init__ `def __init__(self, model, config, device)`
- Defined: `homeostatichope.py:403`

### train_step `def train_step(self, x, y, epsilon, global_step, phase)`
- Defined: `homeostatichope.py:425`

### evaluate `def evaluate(self, test_loader, epsilon, phase)`
- Defined: `homeostatichope.py:488`

## hope.py

### setup_device `def setup_device()`
- Defined: `hope.py:41`

### set_seed `def set_seed(seed)`
- Defined: `hope.py:49`

### pgd_attack `def pgd_attack(model, x, y, epsilon, steps, device)`
- Defined: `hope.py:394`
- Doc: PGD adversarial attack - versión robusta

### run_real_world_experiment `def run_real_world_experiment(config, device)`
- Defined: `hope.py:517`

### run_ablation `def run_ablation(device)`
- Defined: `hope.py:614`

### __init__ `def __init__(self, seed)`
- Defined: `hope.py:64`

### get_batch `def get_batch(self, phase, batch_size)`
- Defined: `hope.py:98`
- Doc: Obtener batch según la fase de entrenamiento

### get_test_loader `def get_test_loader(self, batch_size)`
- Defined: `hope.py:118`
- Doc: Test loader completo

### __init__ `def __init__(self, d_model)`
- Defined: `hope.py:130`

### forward `def forward(self, x, h_prev, w_norm)`
- Defined: `hope.py:140`
- Doc: Calcula controles homeostáticos basados en:

### __init__ `def __init__(self, d_model)`
- Defined: `hope.py:175`

### forward `def forward(self, x, physio)`
- Defined: `hope.py:186`
- Doc: Args:

### __init__ `def __init__(self, d_model, hidden_dim)`
- Defined: `hope.py:210`

### forward `def forward(self, x)`
- Defined: `hope.py:236`
- Doc: Args:

### __init__ `def __init__(self, frequencies, d_model, hidden_dim)`
- Defined: `hope.py:290`

### forward `def forward(self, x, global_step)`
- Defined: `hope.py:304`
- Doc: Args:

### __init__ `def __init__(self, config, n_features, n_classes)`
- Defined: `hope.py:323`

### reset_states `def reset_states(self)`
- Defined: `hope.py:361`

### forward `def forward(self, x, global_step)`
- Defined: `hope.py:365`
- Doc: Args:

### __init__ `def __init__(self, model, config, device)`
- Defined: `hope.py:442`

### train_step `def train_step(self, x, y, epsilon, global_step)`
- Defined: `hope.py:460`
- Doc: Un paso de entrenamiento con adversarial opcional

### evaluate `def evaluate(self, test_loader, epsilon)`
- Defined: `hope.py:490`
- Doc: Evaluación con ataque opcional

## kimi.py

### pgd_attack `def pgd_attack(model, x, y, eps, steps, alpha)`
- Defined: `kimi.py:39`
- Doc: PGD-10 ataque con gradiente corregido para CPU

### get_loader `def get_loader()`
- Defined: `kimi.py:191`

### run_experiment `def run_experiment(seed, sne_enabled, ablated_organs)`
- Defined: `kimi.py:201`
- Doc: Ejecuta un experimento completo con una seed

### scientific_ablation `def scientific_ablation()`
- Defined: `kimi.py:256`
- Doc: Ejecuta el estudio científico completo

### __init__ `def __init__(self, enabled)`
- Defined: `kimi.py:66`

### forward `def forward(self, state, loss)`
- Defined: `kimi.py:75`

### __init__ `def __init__(self, sne, ablated)`
- Defined: `kimi.py:98`

### forward `def forward(self, act)`
- Defined: `kimi.py:104`

### __init__ `def __init__(self, sne, ablated)`
- Defined: `kimi.py:119`

### forward `def forward(self, x)`
- Defined: `kimi.py:127`

### __init__ `def __init__(self, sne, ablated)`
- Defined: `kimi.py:146`

### forward `def forward(self, img)`
- Defined: `kimi.py:155`

### __init__ `def __init__(self, sne_enabled, ablated_organs)`
- Defined: `kimi.py:167`

### forward `def forward(self, x)`
- Defined: `kimi.py:175`

## legendario.py

### compute_phi_effective_approx `def compute_phi_effective_approx(activity)`
- Defined: `legendario.py:27`
- Doc: Cálculo ESTABLE de Φₑ usando PCA (proporción de varianza explicada)

### compute_topological_metrics `def compute_topological_metrics(weights)`
- Defined: `legendario.py:60`
- Doc: Cálculo ESTABLE de métricas topológicas (optimizado para CPU)

### estimate_energy_consumption `def estimate_energy_consumption(model, input_size)`
- Defined: `legendario.py:89`
- Doc: Estimación conservadora de consumo energético para CPU

### train_omni_brain `def train_omni_brain(model, epochs, batch_size, device)`
- Defined: `legendario.py:340`
- Doc: Entrenamiento estable y rápido en CPU

### __init__ `def __init__(self, module_name, enabled)`
- Defined: `legendario.py:119`

### update_performance `def update_performance(self, metrics)`
- Defined: `legendario.py:125`

### __init__ `def __init__(self, in_features, out_features)`
- Defined: `legendario.py:135`

### compute_pt_phase `def compute_pt_phase(self)`
- Defined: `legendario.py:144`
- Doc: Cálculo estable de fase PT sin números complejos

### forward `def forward(self, x, params)`
- Defined: `legendario.py:152`

### __init__ `def __init__(self, in_features, out_features)`
- Defined: `legendario.py:169`

### update_topology `def update_topology(self, connectivity)`
- Defined: `legendario.py:176`
- Doc: Actualizar máscara topológica basada en conectividad deseada

### forward `def forward(self, x, params)`
- Defined: `legendario.py:183`

### __init__ `def __init__(self, features)`
- Defined: `legendario.py:196`

### forward `def forward(self, x, params)`
- Defined: `legendario.py:211`

### __init__ `def __init__(self, features)`
- Defined: `legendario.py:227`

### forward `def forward(self, x, params)`
- Defined: `legendario.py:232`

### __init__ `def __init__(self)`
- Defined: `legendario.py:248`

### measure_network_state `def measure_network_state(self, model, batch_data)`
- Defined: `legendario.py:251`
- Doc: Mediciones ESTABLES para CPU

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `legendario.py:292`

### forward `def forward(self, x)`
- Defined: `legendario.py:317`

## legendario2.py

### train_omni_brain `def train_omni_brain(model, epochs, batch_size)`
- Defined: `legendario2.py:967`
- Doc: Pipeline de entrenamiento para el Omni Brain

### update `def update(self, measurement, dt)`
- Defined: `legendario2.py:47`
- Doc: Actualiza el estado del motor homeostático

### __init__ `def __init__(self)`
- Defined: `legendario2.py:66`

### regulate_parameters `def regulate_parameters(self, current_coherence, energy_level)`
- Defined: `legendario2.py:78`
- Doc: Regula parámetros para mantener PT-simetría

### __init__ `def __init__(self)`
- Defined: `legendario2.py:102`

### regulate_connectivity `def regulate_connectivity(self, current_connectivity, clustering)`
- Defined: `legendario2.py:112`
- Doc: Regula conectividad para mantener estructura óptima

### __init__ `def __init__(self)`
- Defined: `legendario2.py:129`

### regulate_energy `def regulate_energy(self, memory_usage, cpu_usage, temperature)`
- Defined: `legendario2.py:139`
- Doc: Regula parámetros para eficiencia energética

### __init__ `def __init__(self)`
- Defined: `legendario2.py:159`

### regulate_consciousness `def regulate_consciousness(self, phi_effective, integration_level)`
- Defined: `legendario2.py:168`
- Doc: Regula parámetros para control de conciencia

### __init__ `def __init__(self)`
- Defined: `legendario2.py:187`

### regulate_dual_systems `def regulate_dual_systems(self, unconscious_activity, conscious_activity)`
- Defined: `legendario2.py:197`
- Doc: Regula balance entre sistemas inconsciente y consciente

### __init__ `def __init__(self)`
- Defined: `legendario2.py:215`

### regulate_learning `def regulate_learning(self, loss_reduction_rate, gradient_norm)`
- Defined: `legendario2.py:224`
- Doc: Regula parámetros de aprendizaje

### __init__ `def __init__(self)`
- Defined: `legendario2.py:243`

### regulate_modules `def regulate_modules(self, task_complexity, resource_availability, performance)`
- Defined: `legendario2.py:259`
- Doc: Regula qué módulos están activos

### __init__ `def __init__(self)`
- Defined: `legendario2.py:294`

### _initialize_motors `def _initialize_motors(self)`
- Defined: `legendario2.py:300`
- Doc: Inicializa todos los motores homeostáticos

### sense_environment `def sense_environment(self)`
- Defined: `legendario2.py:312`
- Doc: Sensa el estado actual del entorno

### simulate_network_state `def simulate_network_state(self)`
- Defined: `legendario2.py:326`
- Doc: Simula el estado de red sin hacer forward pass (evita conflictos de autograd)

### measure_network_state `def measure_network_state(self, model, batch_data)`
- Defined: `legendario2.py:340`
- Doc: Mide el estado actual de la red

### coordinate_all_motors `def coordinate_all_motors(self, environment_state, network_state)`
- Defined: `legendario2.py:377`
- Doc: Coordina todos los motores homeostáticos

### __init__ `def __init__(self, module_name, enabled)`
- Defined: `legendario2.py:455`

### forward `def forward(self, x, params)`
- Defined: `legendario2.py:461`

### update_performance `def update_performance(self, metrics)`
- Defined: `legendario2.py:464`

### __init__ `def __init__(self, in_features, out_features)`
- Defined: `legendario2.py:470`

### forward `def forward(self, x, params)`
- Defined: `legendario2.py:477`

### __init__ `def __init__(self, in_features, out_features, sparsity_factor)`
- Defined: `legendario2.py:503`

### _generate_topology_mask `def _generate_topology_mask(self)`
- Defined: `legendario2.py:518`
- Doc: Genera máscara topológica realista

### forward `def forward(self, x, params)`
- Defined: `legendario2.py:539`

### __init__ `def __init__(self, features)`
- Defined: `legendario2.py:560`

### forward `def forward(self, x, params)`
- Defined: `legendario2.py:585`

### __init__ `def __init__(self, features)`
- Defined: `legendario2.py:643`

### compute_phi_effective `def compute_phi_effective(self, x)`
- Defined: `legendario2.py:657`
- Doc: Cálculo simplificado de Φₑ (integración efectiva)

### forward `def forward(self, x, params)`
- Defined: `legendario2.py:677`

### __init__ `def __init__(self, target_performance)`
- Defined: `legendario2.py:704`

### regulate_homeostasis `def regulate_homeostasis(self, observed_performance)`
- Defined: `legendario2.py:709`
- Doc: Regula parámetros para homeostasis

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `legendario2.py:735`

### reset_internal_states `def reset_internal_states(self)`
- Defined: `legendario2.py:769`
- Doc: Resetea todos los estados internos para evitar problemas de gradientes

### prepare_for_inference `def prepare_for_inference(self)`
- Defined: `legendario2.py:803`
- Doc: Preparación específica para inferencia - reseteo completo

### initialize_context `def initialize_context(self)`
- Defined: `legendario2.py:820`
- Doc: Inicializa el contexto del Omni Brain

### forward `def forward(self, x)`
- Defined: `legendario2.py:834`
- Doc: Forward pass del Omni Brain con coordinación homeostática

### get_status_report `def get_status_report(self)`
- Defined: `legendario2.py:932`
- Doc: Genera reporte de estado del Omni Brain

## live_cl.py

### setup_logging `def setup_logging()`
- Defined: `live_cl.py:69`
- Doc: Professional logging configuration

### set_seed `def set_seed(seed)`
- Defined: `live_cl.py:84`
- Doc: Ensure reproducibility

### compute_integration_index `def compute_integration_index(activity)`
- Defined: `live_cl.py:99`
- Doc: Compute neural integration using SVD (Singular Value Decomposition)

### get_data_loaders `def get_data_loaders(config)`
- Defined: `live_cl.py:349`
- Doc: Prepare CIFAR-10 data loaders with augmentation

### evaluate `def evaluate(model, loader, device)`
- Defined: `live_cl.py:393`
- Doc: Comprehensive model evaluation

### train `def train(config, silent)`
- Defined: `live_cl.py:430`
- Doc: Main training loop with comprehensive logging

### run_ablation_study `def run_ablation_study(quick_test)`
- Defined: `live_cl.py:570`
- Doc: Comprehensive ablation study across different configurations

### to_dict `def to_dict(self)`
- Defined: `live_cl.py:62`

### __init__ `def __init__(self, in_features, out_features, config)`
- Defined: `live_cl.py:138`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `live_cl.py:156`
- Doc: Reset fast weights (memory purge)

### update_fast_weights `def update_fast_weights(self, x, slow_out)`
- Defined: `live_cl.py:162`
- Doc: Hebbian learning update

### forward `def forward(self, x)`
- Defined: `live_cl.py:189`

### get_fast_norm `def get_fast_norm(self)`
- Defined: `live_cl.py:202`

### __init__ `def __init__(self, dim, config)`
- Defined: `live_cl.py:211`

### forward `def forward(self, x)`
- Defined: `live_cl.py:223`

### __init__ `def __init__(self, features, config)`
- Defined: `live_cl.py:247`

### forward `def forward(self, x)`
- Defined: `live_cl.py:260`

### __init__ `def __init__(self, config)`
- Defined: `live_cl.py:283`

### forward `def forward(self, x)`
- Defined: `live_cl.py:318`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `live_cl.py:325`
- Doc: Reset all fast weights in the network

### get_fast_norms `def get_fast_norms(self)`
- Defined: `live_cl.py:331`
- Doc: Collect fast weight norms for monitoring

### get_ablation_state `def get_ablation_state(self)`
- Defined: `live_cl.py:336`
- Doc: Return current ablation configuration

## live_go.py

### compute_integration_index `def compute_integration_index(activity)`
- Defined: `live_go.py:74`
- Doc: Calcula el orden dentro del caos neuronal mediante SVD.

### get_loaders `def get_loaders(config)`
- Defined: `live_go.py:230`

### breathe_life `def breathe_life(config)`
- Defined: `live_go.py:254`

### reset_seeds `def reset_seeds()`
- Defined: `live_go.py:353`
- Doc: Reinicia el determinismo para que cada variante juegue en igualdad de condiciones.

### run_ablation_test `def run_ablation_test(full_epochs)`
- Defined: `live_go.py:360`
- Doc: Ejecuta el Juicio Final: Compara las diferentes configuraciones del cerebro.

### train_engine_wrapper `def train_engine_wrapper(config)`
- Defined: `live_go.py:425`
- Doc: Versión simplificada de breathe_life para el test que retorna la precisión.

### __init__ `def __init__(self, in_features, out_features, config)`
- Defined: `live_go.py:98`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `live_go.py:115`

### forward `def forward(self, x)`
- Defined: `live_go.py:120`

### get_fast_norm `def get_fast_norm(self)`
- Defined: `live_go.py:140`

### __init__ `def __init__(self, dim, config)`
- Defined: `live_go.py:144`

### forward `def forward(self, x)`
- Defined: `live_go.py:155`

### __init__ `def __init__(self, features, config)`
- Defined: `live_go.py:168`

### forward `def forward(self, x)`
- Defined: `live_go.py:176`

### __init__ `def __init__(self, config)`
- Defined: `live_go.py:191`

### forward `def forward(self, x)`
- Defined: `live_go.py:215`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `live_go.py:222`

## live_ki.py

### compute_integration_index `def compute_integration_index(activity)`
- Defined: `live_ki.py:76`
- Doc: Mide el grado de orden en la actividad neural mediante SVD.

### get_cifar10_loaders `def get_cifar10_loaders(config)`
- Defined: `live_ki.py:308`

### evaluate_ritual `def evaluate_ritual(model, loader, device)`
- Defined: `live_ki.py:334`

### train_genesis `def train_genesis(config)`
- Defined: `live_ki.py:366`

### explore_realities `def explore_realities()`
- Defined: `live_ki.py:499`
- Doc: Explora múltiples configuraciones del universo neural

### __init__ `def __init__(self, in_features, out_features, config)`
- Defined: `live_ki.py:104`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `live_ki.py:124`
- Doc: Ritual de purificación - resetea memoria a corto plazo

### update_fast_weights `def update_fast_weights(self, x, slow_out)`
- Defined: `live_ki.py:130`
- Doc: Ritual Hebbiano - solo ocurre si los dioses lo permiten

### forward `def forward(self, x)`
- Defined: `live_ki.py:157`

### get_fast_norm `def get_fast_norm(self)`
- Defined: `live_ki.py:171`

### __init__ `def __init__(self, dim, config)`
- Defined: `live_ki.py:178`

### forward `def forward(self, x)`
- Defined: `live_ki.py:192`

### __init__ `def __init__(self, features, config)`
- Defined: `live_ki.py:212`

### forward `def forward(self, x)`
- Defined: `live_ki.py:224`

### __init__ `def __init__(self, config)`
- Defined: `live_ki.py:244`

### forward `def forward(self, x)`
- Defined: `live_ki.py:279`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `live_ki.py:286`
- Doc: Ritual de purificación global

### get_fast_norms `def get_fast_norms(self)`
- Defined: `live_ki.py:292`
- Doc: Recopila energías de pesos rápidos

### get_ablation_state `def get_ablation_state(self)`
- Defined: `live_ki.py:296`
- Doc: Estado de creación

## live_qw.py

### compute_integration_index `def compute_integration_index(activity)`
- Defined: `live_qw.py:171`

### get_cifar10_loaders `def get_cifar10_loaders(config)`
- Defined: `live_qw.py:244`

### evaluate_full `def evaluate_full(model, loader, device)`
- Defined: `live_qw.py:265`

### train `def train(config)`
- Defined: `live_qw.py:286`

### __init__ `def __init__(self, in_features, out_features, config)`
- Defined: `live_qw.py:66`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `live_qw.py:83`

### update_fast_weights `def update_fast_weights(self, x, slow_out)`
- Defined: `live_qw.py:88`

### forward `def forward(self, x)`
- Defined: `live_qw.py:106`

### get_fast_norm `def get_fast_norm(self)`
- Defined: `live_qw.py:118`

### __init__ `def __init__(self, dim, config)`
- Defined: `live_qw.py:123`

### forward `def forward(self, x)`
- Defined: `live_qw.py:133`

### __init__ `def __init__(self, features, config)`
- Defined: `live_qw.py:147`

### forward `def forward(self, x)`
- Defined: `live_qw.py:158`

### __init__ `def __init__(self, config)`
- Defined: `live_qw.py:192`

### forward `def forward(self, x)`
- Defined: `live_qw.py:217`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `live_qw.py:224`

### get_fast_norms `def get_fast_norms(self)`
- Defined: `live_qw.py:229`

### get_ablation_state `def get_ablation_state(self)`
- Defined: `live_qw.py:232`

## lol.py

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `lol.py:34`
- Doc: Descarga Flickr8k automáticamente

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `lol.py:443`

### train_bicameral `def train_bicameral()`
- Defined: `lol.py:481`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `lol.py:108`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `lol.py:125`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `lol.py:154`

### __init__ `def __init__(self, output_dim)`
- Defined: `lol.py:181`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `lol.py:192`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `lol.py:202`

### forward `def forward(self, visual_context, captions, max_len, return_gate)`
- Defined: `lol.py:224`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `lol.py:278`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `lol.py:283`

### __init__ `def __init__(self, dim)`
- Defined: `lol.py:299`

### forward `def forward(self, right_features)`
- Defined: `lol.py:307`

### __init__ `def __init__(self, vocab_size)`
- Defined: `lol.py:314`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- Defined: `lol.py:320`

### __init__ `def __init__(self)`
- Defined: `lol.py:335`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `lol.py:346`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `lol.py:353`

### update `def update(self)`
- Defined: `lol.py:357`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `lol.py:362`

### report `def report(self, epoch)`
- Defined: `lol.py:367`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `lol.py:405`

### __len__ `def __len__(self)`
- Defined: `lol.py:423`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `lol.py:426`

### __init__ `def __init__(self, total_epochs)`
- Defined: `lol.py:467`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `lol.py:470`

## main.py

### simulate_resma_multiverse `def simulate_resma_multiverse(n_leaves, n_nodes, seed)`
- Defined: `main.py:764`
- Doc: Pipeline completo RESMA 3.0 con verificaciones de integridad.

### validate_dimension `def validate_dimension(alpha)`
- Defined: `main.py:62`
- Doc: α ∈ (0,1) por definición de dimensión fractal

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `main.py:68`
- Doc: Verificar κ/Ω < χ/Ω < 1 para PT-simetría

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `main.py:79`
- Doc: Límite inferior para conectoma biológico

### __post_init__ `def __post_init__(self)`
- Defined: `main.py:100`
- Doc: Validaciones post-construcción (Pilar 3)

### spectral_density `def spectral_density(self, omega)`
- Defined: `main.py:106`
- Doc: Densidad espectral continua ρ(ω) para álgebra tipo III₁.

### modular_entropy `def modular_entropy(self)`
- Defined: `main.py:114`
- Doc: Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)

### bures_distance `def bures_distance(self, other)`
- Defined: `main.py:121`

### _spectral_moments `def _spectral_moments(self, n)`
- Defined: `main.py:132`
- Doc: Momentos espectrales Tr(ρ^k) para k=1..n

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main.py:150`
- Doc: Args:

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `main.py:168`
- Doc: Genera hojas con gaps espectrales distribuidos

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `main.py:181`
- Doc: Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))

### _construct_global_state `def _construct_global_state(self)`
- Defined: `main.py:203`
- Doc: Estado global: mapa de pesos por hoja (no matriz)

### __init__ `def __init__(self, leaf, threshold)`
- Defined: `main.py:226`

### _construct_cptp_map `def _construct_cptp_map(self)`
- Defined: `main.py:231`
- Doc: Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)

### _local_jump_operator `def _local_jump_operator(self, power)`
- Defined: `main.py:240`
- Doc: K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.

### apply_branching `def apply_branching(self, state_vector)`
- Defined: `main.py:255`
- Doc: Aplicar canal CPTP a vector de estado local (dim=2)

### __init__ `def __init__(self, universe, n_samples)`
- Defined: `main.py:276`

### _construct_hardy_state `def _construct_hardy_state(self)`
- Defined: `main.py:282`
- Doc: E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior

### _szego_projector `def _szego_projector(self)`
- Defined: `main.py:286`
- Doc: Proyector P_E en base de Fourier positiva (dim reducida)

### _evaluation_functional `def _evaluation_functional(self, state_weights)`
- Defined: `main.py:294`
- Doc: Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))

### project `def project(self, state_vector)`
- Defined: `main.py:310`
- Doc: P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO

### __init__ `def __init__(self, universe, emuna)`
- Defined: `main.py:356`

### _effective_hamiltonian `def _effective_hamiltonian(self)`
- Defined: `main.py:362`
- Doc: H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)

### _modular_dissipator `def _modular_dissipator(self, state)`
- Defined: `main.py:372`
- Doc: L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}

### _nonlinear_term `def _nonlinear_term(self, state)`
- Defined: `main.py:382`
- Doc: G[ρ, log ρ_∞] = g[ρ, log ρ_∞]

### evolve `def evolve(self, rho0, t_span, n_steps)`
- Defined: `main.py:389`
- Doc: Integración SDE con Euler-Maruyama.

### __post_init__ `def __post_init__(self)`
- Defined: `main.py:437`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main.py:442`
- Doc: H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³

### _loss_potential `def _loss_potential(self)`
- Defined: `main.py:448`
- Doc: V_loss ∝ (r⊥/a₀)^{2α} con α=0.7

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main.py:456`
- Doc: Verificar κ/Ω < χ/Ω < 1

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main.py:462`
- Doc: Discordia cuántica aproximada (ejemplo: estado separable → 0)

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main.py:487`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `main.py:500`
- Doc: Grafo dirigido con distribución de grados power-law.

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `main.py:510`
- Doc: d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `main.py:527`
- Doc: R_Q(G) = min{n | β_{n-1}(G) > 0}

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `main.py:548`
- Doc: Matriz de distancias shortest-path (sparse CSR)

### critical_percolation_time `def critical_percolation_time(self)`
- Defined: `main.py:562`
- Doc: t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25

### is_coherent_subgraph `def is_coherent_subgraph(self, subgraph_nodes)`
- Defined: `main.py:574`
- Doc: Verificar coherencia: subgrafo > 70% del total

### __init__ `def __init__(self, network, universe)`
- Defined: `main.py:588`

### compute_entropy_gap `def compute_entropy_gap(self)`
- Defined: `main.py:592`
- Doc: Δ_S* = ε_c en punto excepcional

### compute_pontryagin_number `def compute_pontryagin_number(self)`
- Defined: `main.py:596`
- Doc: S_top[G] = χ(G)/|V| (número de Euler normalizado)

### compute_freedom `def compute_freedom(self)`
- Defined: `main.py:607`
- Doc: L[G] = Δ_S* / S_top[G]

### is_gauge_invariant `def is_gauge_invariant(self)`
- Defined: `main.py:618`
- Doc: |L[G] - 1| < 0.05 en estado crítico

### ising_quantum `def ising_quantum(network)`
- Defined: `main.py:635`
- Doc: Modelo de Ising cuántico transversal en red fractal.

### syk4 `def syk4(network)`
- Defined: `main.py:654`
- Doc: SYK₄ estándar (sin R-simetría Spin(7)).

### random_network `def random_network(network)`
- Defined: `main.py:670`
- Doc: Red aleatoria Erdős-Rényi sin percolación cuántica.

### __init__ `def __init__(self, resma, myelin, network)`
- Defined: `main.py:692`

### predict_all `def predict_all(self)`
- Defined: `main.py:699`
- Doc: Predicciones RESMA 3.0

### _predict_diffraction_peak `def _predict_diffraction_peak(self)`
- Defined: `main.py:710`
- Doc: q₀ = 2π/L_E8 (sin ajuste)

### compute_bayes_factor `def compute_bayes_factor(self)`
- Defined: `main.py:715`
- Doc: BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)

## main2.py

### simulate_resma_multiverse `def simulate_resma_multiverse(n_leaves, n_nodes, seed)`
- Defined: `main2.py:763`
- Doc: Pipeline completo RESMA 3.0 con verificaciones de integridad.

### validate_dimension `def validate_dimension(alpha)`
- Defined: `main2.py:63`
- Doc: α ∈ (0,1) por definición de dimensión fractal

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `main2.py:69`
- Doc: Verificar κ/Ω < χ/Ω < 1 para PT-simetría

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `main2.py:80`
- Doc: Límite inferior para conectoma biológico

### __post_init__ `def __post_init__(self)`
- Defined: `main2.py:101`
- Doc: Validaciones post-construcción (Pilar 3)

### spectral_density `def spectral_density(self, omega)`
- Defined: `main2.py:106`
- Doc: Densidad espectral continua ρ(ω) para álgebra tipo III₁.

### modular_entropy `def modular_entropy(self)`
- Defined: `main2.py:114`
- Doc: Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)

### bures_distance `def bures_distance(self, other)`
- Defined: `main2.py:121`

### _spectral_moments `def _spectral_moments(self, n)`
- Defined: `main2.py:132`
- Doc: Momentos espectrales Tr(ρ^k) para k=1..n

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main2.py:150`
- Doc: Args:

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `main2.py:167`
- Doc: Genera hojas con gaps espectrales distribuidos

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `main2.py:180`
- Doc: Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))

### _construct_global_state `def _construct_global_state(self)`
- Defined: `main2.py:202`
- Doc: Estado global: mapa de pesos por hoja (no matriz)

### __init__ `def __init__(self, leaf, threshold)`
- Defined: `main2.py:225`

### _construct_cptp_map `def _construct_cptp_map(self)`
- Defined: `main2.py:230`
- Doc: Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)

### _local_jump_operator `def _local_jump_operator(self, power)`
- Defined: `main2.py:239`
- Doc: K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.

### apply_branching `def apply_branching(self, state_vector)`
- Defined: `main2.py:254`
- Doc: Aplicar canal CPTP a vector de estado local (dim=2)

### __init__ `def __init__(self, universe, n_samples)`
- Defined: `main2.py:275`

### _construct_hardy_state `def _construct_hardy_state(self)`
- Defined: `main2.py:281`
- Doc: E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior

### _szego_projector `def _szego_projector(self)`
- Defined: `main2.py:285`
- Doc: Proyector P_E en base de Fourier positiva (dim reducida)

### _evaluation_functional `def _evaluation_functional(self, state_weights)`
- Defined: `main2.py:293`
- Doc: Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))

### project `def project(self, state_vector)`
- Defined: `main2.py:309`
- Doc: P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO

### __init__ `def __init__(self, universe, emuna)`
- Defined: `main2.py:355`

### _effective_hamiltonian `def _effective_hamiltonian(self)`
- Defined: `main2.py:361`
- Doc: H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)

### _modular_dissipator `def _modular_dissipator(self, state)`
- Defined: `main2.py:371`
- Doc: L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}

### _nonlinear_term `def _nonlinear_term(self, state)`
- Defined: `main2.py:381`
- Doc: G[ρ, log ρ_∞] = g[ρ, log ρ_∞]

### evolve `def evolve(self, rho0, t_span, n_steps)`
- Defined: `main2.py:388`
- Doc: Integración SDE con Euler-Maruyama.

### __post_init__ `def __post_init__(self)`
- Defined: `main2.py:436`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main2.py:441`
- Doc: H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³

### _loss_potential `def _loss_potential(self)`
- Defined: `main2.py:447`
- Doc: V_loss ∝ (r⊥/a₀)^{2α} con α=0.7

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main2.py:455`
- Doc: Verificar κ/Ω < χ/Ω < 1

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main2.py:461`
- Doc: Discordia cuántica aproximada (ejemplo: estado separable → 0)

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main2.py:486`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `main2.py:499`
- Doc: Grafo dirigido con distribución de grados power-law.

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `main2.py:509`
- Doc: d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `main2.py:526`
- Doc: R_Q(G) = min{n | β_{n-1}(G) > 0}

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `main2.py:547`
- Doc: Matriz de distancias shortest-path (sparse CSR)

### critical_percolation_time `def critical_percolation_time(self)`
- Defined: `main2.py:561`
- Doc: t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25

### is_coherent_subgraph `def is_coherent_subgraph(self, subgraph_nodes)`
- Defined: `main2.py:573`
- Doc: Verificar coherencia: subgrafo > 70% del total

### __init__ `def __init__(self, network, universe)`
- Defined: `main2.py:587`

### compute_entropy_gap `def compute_entropy_gap(self)`
- Defined: `main2.py:591`
- Doc: Δ_S* = ε_c en punto excepcional

### compute_pontryagin_number `def compute_pontryagin_number(self)`
- Defined: `main2.py:595`
- Doc: S_top[G] = χ(G)/|V| (número de Euler normalizado)

### compute_freedom `def compute_freedom(self)`
- Defined: `main2.py:606`
- Doc: L[G] = Δ_S* / S_top[G]

### is_gauge_invariant `def is_gauge_invariant(self)`
- Defined: `main2.py:617`
- Doc: |L[G] - 1| < 0.05 en estado crítico

### ising_quantum `def ising_quantum(network)`
- Defined: `main2.py:634`
- Doc: Modelo de Ising cuántico transversal en red fractal.

### syk4 `def syk4(network)`
- Defined: `main2.py:653`
- Doc: SYK₄ estándar (sin R-simetría Spin(7)).

### random_network `def random_network(network)`
- Defined: `main2.py:669`
- Doc: Red aleatoria Erdős-Rényi sin percolación cuántica.

### __init__ `def __init__(self, resma, myelin, network)`
- Defined: `main2.py:691`

### predict_all `def predict_all(self)`
- Defined: `main2.py:698`
- Doc: Predicciones RESMA 3.0

### _predict_diffraction_peak `def _predict_diffraction_peak(self)`
- Defined: `main2.py:708`
- Doc: q₀ = 2π/L_E8 (sin ajuste)

### compute_bayes_factor `def compute_bayes_factor(self)`
- Defined: `main2.py:713`
- Doc: BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)

## main3.py

### simulate `def simulate(n_leaves, n_nodes, seed)`
- Defined: `main3.py:233`

### dim `def dim(a)`
- Defined: `main3.py:52`

### pt `def pt(k, o, c)`
- Defined: `main3.py:56`

### size `def size(n)`
- Defined: `main3.py:59`

### __post_init__ `def __post_init__(self)`
- Defined: `main3.py:74`

### spectral_density `def spectral_density(self, w)`
- Defined: `main3.py:78`

### modular_entropy `def modular_entropy(self)`
- Defined: `main3.py:81`

### bures_distance `def bures_distance(self, other)`
- Defined: `main3.py:87`

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main3.py:102`

### _gibbs `def _gibbs(self)`
- Defined: `main3.py:110`

### _global `def _global(self)`
- Defined: `main3.py:119`

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main3.py:130`

### _spectral_dim `def _spectral_dim(self, k)`
- Defined: `main3.py:139`

### _ramsey `def _ramsey(self)`
- Defined: `main3.py:149`

### t_c `def t_c(self)`
- Defined: `main3.py:163`

### __init__ `def __init__(self, n_modes)`
- Defined: `main3.py:172`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main3.py:178`

### _loss_potential `def _loss_potential(self)`
- Defined: `main3.py:183`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main3.py:189`

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main3.py:192`

### __init__ `def __init__(self, pred_resma, nulls)`
- Defined: `main3.py:206`

### log_lik `def log_lik(self, model_pred)`
- Defined: `main3.py:210`

### bf `def bf(self)`
- Defined: `main3.py:217`

## main4.1.py

### simulate `def simulate(n_leaves, n_nodes, seed)`
- Defined: `main4.1.py:307`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `main4.1.py:50`
- Doc: Verifica que kappa < chi*Omega para simetría PT

### dim `def dim(a)`
- Defined: `main4.1.py:63`

### pt `def pt(k, o, c)`
- Defined: `main4.1.py:68`
- Doc: Condición PT: kappa < chi*Omega

### size `def size(n)`
- Defined: `main4.1.py:73`

### __post_init__ `def __post_init__(self)`
- Defined: `main4.1.py:88`

### spectral_density `def spectral_density(self, w)`
- Defined: `main4.1.py:92`

### modular_entropy `def modular_entropy(self)`
- Defined: `main4.1.py:95`

### bures_distance `def bures_distance(self, other)`
- Defined: `main4.1.py:104`

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main4.1.py:124`

### _gibbs `def _gibbs(self)`
- Defined: `main4.1.py:133`

### _global `def _global(self)`
- Defined: `main4.1.py:142`

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main4.1.py:153`

### _spectral_dim `def _spectral_dim(self, k, n_fit)`
- Defined: `main4.1.py:163`
- Doc: Dimensión espectral corregida

### _ramsey `def _ramsey(self)`
- Defined: `main4.1.py:199`
- Doc: Número de Ramsey topológico

### t_c `def t_c(self)`
- Defined: `main4.1.py:218`
- Doc: Tiempo crítico de percolación

### __init__ `def __init__(self, n_modes)`
- Defined: `main4.1.py:230`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main4.1.py:237`

### _loss_potential `def _loss_potential(self)`
- Defined: `main4.1.py:242`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main4.1.py:248`

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main4.1.py:251`

### __init__ `def __init__(self, pred_resma, nulls)`
- Defined: `main4.1.py:269`

### log_lik `def log_lik(self, model_pred)`
- Defined: `main4.1.py:273`
- Doc: Verosimilitud con escalas físicas realistas

### ln_bf `def ln_bf(self)`
- Defined: `main4.1.py:288`
- Doc: Factor de Bayes con penalización de complejidad

## main4.py.py

### simulate `def simulate(n_leaves, n_nodes, seed)`
- Defined: `main4.py.py:260`

### dim `def dim(a)`
- Defined: `main4.py.py:52`

### pt `def pt(k, o, c)`
- Defined: `main4.py.py:56`

### size `def size(n)`
- Defined: `main4.py.py:60`

### __post_init__ `def __post_init__(self)`
- Defined: `main4.py.py:75`

### spectral_density `def spectral_density(self, w)`
- Defined: `main4.py.py:79`

### modular_entropy `def modular_entropy(self)`
- Defined: `main4.py.py:82`

### bures_distance `def bures_distance(self, other)`
- Defined: `main4.py.py:88`

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main4.py.py:103`

### _gibbs `def _gibbs(self)`
- Defined: `main4.py.py:111`

### _global `def _global(self)`
- Defined: `main4.py.py:120`

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main4.py.py:131`

### _spectral_dim `def _spectral_dim(self, k, n_fit)`
- Defined: `main4.py.py:140`

### _ramsey `def _ramsey(self)`
- Defined: `main4.py.py:176`

### t_c `def t_c(self)`
- Defined: `main4.py.py:190`

### __init__ `def __init__(self, n_modes)`
- Defined: `main4.py.py:199`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main4.py.py:205`

### _loss_potential `def _loss_potential(self)`
- Defined: `main4.py.py:210`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main4.py.py:216`

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main4.py.py:219`

### __init__ `def __init__(self, pred_resma, nulls)`
- Defined: `main4.py.py:233`

### log_lik `def log_lik(self, model_pred)`
- Defined: `main4.py.py:237`

### ln_bf `def ln_bf(self)`
- Defined: `main4.py.py:244`

## main5.py

### simulate_resma_multiverse `def simulate_resma_multiverse(n_leaves, n_nodes, seed, validate_empirical)`
- Defined: `main5.py:1052`
- Doc: Pipeline completo RESMA 4.0 con verificaciones de integridad y protocolo de validación.

### validate_dimension `def validate_dimension(alpha, tolerance)`
- Defined: `main5.py:74`
- Doc: α ∈ (0,1) por definición de dimensión fractal, con tolerancia experimental

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `main5.py:84`
- Doc: Verificar κ/Ω < χ/Ω < 1 para PT-simetría (corregido con factor de seguridad)

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `main5.py:95`
- Doc: Límite inferior para conectoma biológico realista

### validate_spectral_dimension `def validate_spectral_dimension(dim)`
- Defined: `main5.py:101`
- Doc: Validar rango físico para dimensión espectral

### validate_percolation_time `def validate_percolation_time(t_c, expected, tolerance)`
- Defined: `main5.py:106`
- Doc: Validar tiempo de percolación contra predicción empírica

### __post_init__ `def __post_init__(self)`
- Defined: `main5.py:127`
- Doc: Validaciones post-construcción (Pilar 3)

### spectral_density `def spectral_density(self, omega)`
- Defined: `main5.py:133`
- Doc: Densidad espectral continua ρ(ω) para álgebra tipo III₁ con regularización UV.

### modular_entropy `def modular_entropy(self)`
- Defined: `main5.py:143`
- Doc: Entropía modular S = ∫ ρ(ω)logρ(ω) dω con regularización

### bures_distance `def bures_distance(self, other)`
- Defined: `main5.py:151`

### _spectral_moments `def _spectral_moments(self, n)`
- Defined: `main5.py:163`
- Doc: Momentos espectrales Tr(ρ^k) para k=1..n con regularización

### haagerup_weight `def haagerup_weight(self)`
- Defined: `main5.py:170`
- Doc: Peso de Haagerup para regularización del operador modular

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main5.py:185`
- Doc: Args:

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `main5.py:203`
- Doc: Genera hojas con gaps espectrales distribuidos exponencialmente

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `main5.py:217`
- Doc: Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j)) con normalización robusta

### _construct_global_state `def _construct_global_state(self)`
- Defined: `main5.py:239`
- Doc: Estado global: mapa de pesos por hoja (no matriz) con regularización

### compute_gibbs_free_energy `def compute_gibbs_free_energy(self)`
- Defined: `main5.py:251`
- Doc: Energía libre de Gibbs para validación termodinámica

### __init__ `def __init__(self, leaf, threshold)`
- Defined: `main5.py:266`

### _compute_holonomy `def _compute_holonomy(self)`
- Defined: `main5.py:272`
- Doc: Defecto de holonomía como variación del gap espectral

### _construct_cptp_map `def _construct_cptp_map(self)`
- Defined: `main5.py:276`
- Doc: Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)

### _local_jump_operator `def _local_jump_operator(self, power)`
- Defined: `main5.py:284`
- Doc: K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.

### apply_branching `def apply_branching(self, state_vector)`
- Defined: `main5.py:300`
- Doc: Aplicar canal CPTP a vector de estado local (dim=2) con normalización

### __init__ `def __init__(self, universe, n_samples)`
- Defined: `main5.py:324`

### _construct_hardy_state `def _construct_hardy_state(self)`
- Defined: `main5.py:331`
- Doc: E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior

### _szego_projector `def _szego_projector(self)`
- Defined: `main5.py:335`
- Doc: Proyector P_E en base de Fourier positiva (dim reducida)

### _evaluation_functional `def _evaluation_functional(self, state_weights)`
- Defined: `main5.py:345`
- Doc: Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))

### project `def project(self, state_vector)`
- Defined: `main5.py:361`
- Doc: P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO

### compute_teleological_overlap `def compute_teleological_overlap(self)`
- Defined: `main5.py:396`
- Doc: Calcular overlap teleológico con estado objetivo

### __init__ `def __init__(self, universe, emuna)`
- Defined: `main5.py:412`

### _effective_hamiltonian `def _effective_hamiltonian(self)`
- Defined: `main5.py:419`
- Doc: H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva con gaps SYK₈)

### _modular_dissipator `def _modular_dissipator(self, state)`
- Defined: `main5.py:432`
- Doc: L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ} con regularización

### _nonlinear_term `def _nonlinear_term(self, state)`
- Defined: `main5.py:444`
- Doc: G[ρ, log ρ_∞] = g[ρ, log ρ_∞] con regularización del logaritmo

### _stochastic_term `def _stochastic_term(self, dt)`
- Defined: `main5.py:452`
- Doc: Término estocástico ξ(t) con correlaciones cuánticas

### evolve `def evolve(self, rho0, t_span, n_steps)`
- Defined: `main5.py:459`
- Doc: Integración SDE con Euler-Maruyama y control de paso adaptativo.

### _normalize_density_matrix `def _normalize_density_matrix(self, state)`
- Defined: `main5.py:500`
- Doc: Normalizar matriz densidad y forzar hermiticidad

### _is_physical_state `def _is_physical_state(self, state)`
- Defined: `main5.py:509`
- Doc: Verificar si el estado es físico (hermitiano, traza=1, positivo)

### _correct_non_physical_state `def _correct_non_physical_state(self, state)`
- Defined: `main5.py:523`
- Doc: Corregir estado no físico proyectando en el cono de estados válidos

### __post_init__ `def __post_init__(self)`
- Defined: `main5.py:548`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main5.py:555`
- Doc: H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³

### _loss_potential `def _loss_potential(self)`
- Defined: `main5.py:561`
- Doc: V_loss ∝ (r⊥/a₀)^{2α} con α=0.702 (SYK₈)

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `main5.py:569`
- Doc: Campo escalar masivo para estabilización de Spin(7)

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main5.py:573`
- Doc: Verificar κ/Ω < χ/Ω < 1 con parámetros corregidos

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main5.py:579`
- Doc: Discordia cuántica aproximada con corrección PT

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main5.py:610`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `main5.py:625`
- Doc: Generar grafo dirigido y convertir a NO DIRIGIDO para análisis espectral.

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `main5.py:649`
- Doc: d_s = -2 lim_{λ→0⁺} log N(λ)/log λ usando normalized_laplacian_spectrum.

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `main5.py:680`
- Doc: R_Q(G) = min{n | β_{n-1}(G) > 0}

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `main5.py:706`
- Doc: Calcular números de Betti para análisis topológico

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `main5.py:722`
- Doc: Matriz de distancias shortest-path (sparse CSR) para homología

### critical_percolation_time `def critical_percolation_time(self)`
- Defined: `main5.py:735`
- Doc: t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25

### is_coherent_subgraph `def is_coherent_subgraph(self, subgraph_nodes)`
- Defined: `main5.py:747`
- Doc: Verificar coherencia: subgrafo > 70% del total

### compute_network_entropy `def compute_network_entropy(self)`
- Defined: `main5.py:751`
- Doc: Entropía de la red basada en distribución de grados

### __init__ `def __init__(self, network, universe)`
- Defined: `main5.py:768`

### compute_entropy_gap `def compute_entropy_gap(self)`
- Defined: `main5.py:772`
- Doc: Δ_S* = ε_c en punto excepcional con corrección de regularización

### compute_pontryagin_number `def compute_pontryagin_number(self)`
- Defined: `main5.py:776`
- Doc: S_top[G] = χ(G)/|V| (número de Euler normalizado)

### compute_freedom `def compute_freedom(self)`
- Defined: `main5.py:792`
- Doc: L[G] = Δ_S* / S_top[G] con protección de división por cero

### is_gauge_invariant `def is_gauge_invariant(self)`
- Defined: `main5.py:803`
- Doc: |L[G] - 1| < 0.05 en estado crítico (invariante de libertad)

### ising_quantum `def ising_quantum(network)`
- Defined: `main5.py:823`
- Doc: Modelo de Ising cuántico transversal en red fractal.

### syk4 `def syk4(network)`
- Defined: `main5.py:843`
- Doc: SYK₄ estándar (sin R-simetría Spin(7) ni E₈).

### random_network `def random_network(network)`
- Defined: `main5.py:860`
- Doc: Red aleatoria Erdős-Rényi sin percolación cuántica ni estructura.

### __init__ `def __init__(self, resma, myelin, network, freedom)`
- Defined: `main5.py:883`

### predict_all `def predict_all(self)`
- Defined: `main5.py:891`
- Doc: Predicciones RESMA 4.0 con valores empíricos objetivo

### _predict_diffraction_peak `def _predict_diffraction_peak(self)`
- Defined: `main5.py:906`
- Doc: q₀ = 2π/L_E8 (predicción de difracción UASED)

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `main5.py:912`
- Doc: log(BF) = ΔAIC/2 donde AIC = 2k - 2ln(L)

### __init__ `def __init__(self, predictions)`
- Defined: `main5.py:981`

### _define_protocols `def _define_protocols(self)`
- Defined: `main5.py:985`
- Doc: Definir protocolos experimentales con parámetros técnicos

### evaluate_feasibility `def evaluate_feasibility(self, budget, time_limit)`
- Defined: `main5.py:1014`
- Doc: Evaluar viabilidad del protocolo completo

### simulate_experimental_outcome `def simulate_experimental_outcome(self, protocol_name)`
- Defined: `main5.py:1027`
- Doc: Simular resultado experimental con ruido realista

## microbi.py.py

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `microbi.py.py:650`

### setup_flickr8k_cpu `def setup_flickr8k_cpu(data_dir)`
- Defined: `microbi.py.py:721`
- Doc: Descarga Flickr8k automáticamente (igual que la versión original)

### train_bicameral_cpu `def train_bicameral_cpu()`
- Defined: `microbi.py.py:801`

### __init__ `def __init__(self, feature_dim, hidden_dim)`
- Defined: `microbi.py.py:38`

### compute_intrinsic_reward `def compute_intrinsic_reward(self, state, action, next_state)`
- Defined: `microbi.py.py:55`

### update `def update(self, state, action, next_state)`
- Defined: `microbi.py.py:71`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `microbi.py.py:92`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `microbi.py.py:117`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `microbi.py.py:168`

### __init__ `def __init__(self, dim, num_heads)`
- Defined: `microbi.py.py:196`

### forward `def forward(self, x, mask)`
- Defined: `microbi.py.py:211`

### __init__ `def __init__(self, output_dim)`
- Defined: `microbi.py.py:249`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `microbi.py.py:285`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `microbi.py.py:309`

### forward `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)`
- Defined: `microbi.py.py:338`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `microbi.py.py:416`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `microbi.py.py:421`

### __init__ `def __init__(self, dim)`
- Defined: `microbi.py.py:437`

### forward `def forward(self, right_features)`
- Defined: `microbi.py.py:456`

### __init__ `def __init__(self, vocab_size)`
- Defined: `microbi.py.py:474`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)`
- Defined: `microbi.py.py:480`

### __init__ `def __init__(self)`
- Defined: `microbi.py.py:514`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `microbi.py.py:523`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `microbi.py.py:530`

### update `def update(self)`
- Defined: `microbi.py.py:548`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `microbi.py.py:553`

### report `def report(self, epoch)`
- Defined: `microbi.py.py:560`

### __init__ `def __init__(self, total_epochs)`
- Defined: `microbi.py.py:603`

### get_phase `def get_phase(self, epoch)`
- Defined: `microbi.py.py:611`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `microbi.py.py:617`

### get_exploration_bonus `def get_exploration_bonus(self, epoch)`
- Defined: `microbi.py.py:627`

### get_temperature `def get_temperature(self, epoch)`
- Defined: `microbi.py.py:636`

### should_consolidate `def should_consolidate(self, epoch)`
- Defined: `microbi.py.py:646`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `microbi.py.py:672`

### __len__ `def __len__(self)`
- Defined: `microbi.py.py:690`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `microbi.py.py:693`

### nan_hook `def nan_hook(module, grad_input, grad_output)`
- Defined: `microbi.py.py:880`

## minibi.py

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `minibi.py:35`
- Doc: Descarga Flickr8k automáticamente

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `minibi.py:104`

### build_vocab `def build_vocab(ann_file, vocab_size)`
- Defined: `minibi.py:490`
- Doc: Construir vocabulario desde annotations de COCO

### train_bicameral `def train_bicameral()`
- Defined: `minibi.py:536`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `minibi.py:125`

### __len__ `def __len__(self)`
- Defined: `minibi.py:143`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `minibi.py:146`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `minibi.py:168`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `minibi.py:185`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `minibi.py:219`

### __init__ `def __init__(self, output_dim)`
- Defined: `minibi.py:251`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `minibi.py:268`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `minibi.py:283`

### forward `def forward(self, visual_context, captions, max_len, return_gate, temperature)`
- Defined: `minibi.py:305`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `minibi.py:361`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `minibi.py:366`

### __init__ `def __init__(self, dim)`
- Defined: `minibi.py:383`

### forward `def forward(self, right_features)`
- Defined: `minibi.py:392`

### __init__ `def __init__(self, vocab_size)`
- Defined: `minibi.py:404`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature)`
- Defined: `minibi.py:410`

### __init__ `def __init__(self)`
- Defined: `minibi.py:421`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `minibi.py:432`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `minibi.py:439`

### update `def update(self)`
- Defined: `minibi.py:443`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `minibi.py:448`

### report `def report(self, epoch)`
- Defined: `minibi.py:455`

### __init__ `def __init__(self, total_epochs)`
- Defined: `minibi.py:519`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `minibi.py:522`

## minibi2.py

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `minibi2.py:760`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `minibi2.py:821`
- Doc: Descarga Flickr8k automáticamente

### train_bicameral_v2 `def train_bicameral_v2()`
- Defined: `minibi2.py:893`

### __init__ `def __init__(self, feature_dim, hidden_dim)`
- Defined: `minibi2.py:41`

### compute_intrinsic_reward `def compute_intrinsic_reward(self, state, action, next_state)`
- Defined: `minibi2.py:60`
- Doc: Recompensa intrínseca = error de predicción del forward model

### update `def update(self, state, action, next_state)`
- Defined: `minibi2.py:83`
- Doc: Entrena los modelos de curiosidad

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `minibi2.py:108`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `minibi2.py:137`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `minibi2.py:185`

### __init__ `def __init__(self, dim, num_heads)`
- Defined: `minibi2.py:218`

### forward `def forward(self, x, mask)`
- Defined: `minibi2.py:236`

### __init__ `def __init__(self, output_dim)`
- Defined: `minibi2.py:286`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `minibi2.py:320`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `minibi2.py:352`

### forward `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)`
- Defined: `minibi2.py:387`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `minibi2.py:498`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `minibi2.py:503`

### __init__ `def __init__(self, dim)`
- Defined: `minibi2.py:526`

### forward `def forward(self, right_features)`
- Defined: `minibi2.py:546`

### __init__ `def __init__(self, vocab_size)`
- Defined: `minibi2.py:563`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)`
- Defined: `minibi2.py:569`

### __init__ `def __init__(self)`
- Defined: `minibi2.py:592`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `minibi2.py:612`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `minibi2.py:619`
- Doc: Mide diversidad real + entropía

### update `def update(self)`
- Defined: `minibi2.py:639`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `minibi2.py:644`

### report `def report(self, epoch)`
- Defined: `minibi2.py:651`

### __init__ `def __init__(self, total_epochs)`
- Defined: `minibi2.py:704`

### get_phase `def get_phase(self, epoch)`
- Defined: `minibi2.py:712`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `minibi2.py:718`

### get_exploration_bonus `def get_exploration_bonus(self, epoch)`
- Defined: `minibi2.py:732`
- Doc: Bonus de curiosidad que decae con el tiempo

### get_temperature `def get_temperature(self, epoch)`
- Defined: `minibi2.py:743`
- Doc: Temperature que decae suavemente

### should_consolidate `def should_consolidate(self, epoch)`
- Defined: `minibi2.py:755`
- Doc: Decide cuándo hacer consolidación SVD

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `minibi2.py:782`

### __len__ `def __len__(self)`
- Defined: `minibi2.py:800`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `minibi2.py:803`

## minibi_c.py

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `minibi_c.py:644`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `minibi_c.py:722`
- Doc: Descarga Flickr8k automáticamente

### train_bicameral_v2 `def train_bicameral_v2()`
- Defined: `minibi_c.py:794`

### __init__ `def __init__(self, feature_dim, hidden_dim)`
- Defined: `minibi_c.py:38`

### compute_intrinsic_reward `def compute_intrinsic_reward(self, state, action, next_state)`
- Defined: `minibi_c.py:55`

### update `def update(self, state, action, next_state)`
- Defined: `minibi_c.py:72`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `minibi_c.py:95`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `minibi_c.py:121`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `minibi_c.py:170`

### __init__ `def __init__(self, dim, num_heads)`
- Defined: `minibi_c.py:198`

### forward `def forward(self, x, mask)`
- Defined: `minibi_c.py:212`

### __init__ `def __init__(self, output_dim)`
- Defined: `minibi_c.py:248`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `minibi_c.py:281`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `minibi_c.py:302`

### forward `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)`
- Defined: `minibi_c.py:332`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `minibi_c.py:412`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `minibi_c.py:417`

### __init__ `def __init__(self, dim)`
- Defined: `minibi_c.py:434`

### forward `def forward(self, right_features)`
- Defined: `minibi_c.py:450`

### __init__ `def __init__(self, vocab_size)`
- Defined: `minibi_c.py:465`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)`
- Defined: `minibi_c.py:471`

### __init__ `def __init__(self)`
- Defined: `minibi_c.py:494`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `minibi_c.py:510`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `minibi_c.py:517`

### update `def update(self)`
- Defined: `minibi_c.py:535`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `minibi_c.py:541`

### report `def report(self, epoch)`
- Defined: `minibi_c.py:548`

### __init__ `def __init__(self, total_epochs)`
- Defined: `minibi_c.py:594`

### get_phase `def get_phase(self, epoch)`
- Defined: `minibi_c.py:602`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `minibi_c.py:608`

### get_exploration_bonus `def get_exploration_bonus(self, epoch)`
- Defined: `minibi_c.py:619`

### get_temperature `def get_temperature(self, epoch)`
- Defined: `minibi_c.py:629`

### should_consolidate `def should_consolidate(self, epoch)`
- Defined: `minibi_c.py:640`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `minibi_c.py:668`

### __len__ `def __len__(self)`
- Defined: `minibi_c.py:686`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `minibi_c.py:689`

## minibi_reduced.py.py

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `minibi_reduced.py.py:381`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `minibi_reduced.py.py:431`

### train_bicameral_v2 `def train_bicameral_v2()`
- Defined: `minibi_reduced.py.py:482`

### __init__ `def __init__(self, feature_dim, hidden_dim)`
- Defined: `minibi_reduced.py.py:40`

### compute_intrinsic_reward `def compute_intrinsic_reward(self, state, action, next_state)`
- Defined: `minibi_reduced.py.py:55`

### update `def update(self, state, action, next_state)`
- Defined: `minibi_reduced.py.py:70`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `minibi_reduced.py.py:82`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `minibi_reduced.py.py:105`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `minibi_reduced.py.py:141`

### __init__ `def __init__(self, dim, num_heads)`
- Defined: `minibi_reduced.py.py:171`

### forward `def forward(self, x, mask)`
- Defined: `minibi_reduced.py.py:184`

### __init__ `def __init__(self, output_dim)`
- Defined: `minibi_reduced.py.py:214`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `minibi_reduced.py.py:237`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `minibi_reduced.py.py:248`

### forward `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)`
- Defined: `minibi_reduced.py.py:266`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `minibi_reduced.py.py:323`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `minibi_reduced.py.py:328`

### __init__ `def __init__(self, dim)`
- Defined: `minibi_reduced.py.py:339`

### forward `def forward(self, right_features)`
- Defined: `minibi_reduced.py.py:349`

### __init__ `def __init__(self, vocab_size)`
- Defined: `minibi_reduced.py.py:357`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)`
- Defined: `minibi_reduced.py.py:363`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `minibi_reduced.py.py:397`

### __len__ `def __len__(self)`
- Defined: `minibi_reduced.py.py:412`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `minibi_reduced.py.py:414`

## miniminibi.py

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `miniminibi.py:30`
- Doc: Descarga Flickr8k automáticamente desde Kaggle

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `miniminibi.py:485`

### train_bicameral `def train_bicameral()`
- Defined: `miniminibi.py:523`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `miniminibi.py:119`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `miniminibi.py:136`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `miniminibi.py:165`

### __init__ `def __init__(self, output_dim)`
- Defined: `miniminibi.py:192`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `miniminibi.py:205`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `miniminibi.py:215`

### forward `def forward(self, visual_context, captions, max_len, return_gate, temperature)`
- Defined: `miniminibi.py:237`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `miniminibi.py:293`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `miniminibi.py:298`

### __init__ `def __init__(self, dim)`
- Defined: `miniminibi.py:315`

### forward `def forward(self, right_features)`
- Defined: `miniminibi.py:324`

### __init__ `def __init__(self, vocab_size)`
- Defined: `miniminibi.py:333`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature)`
- Defined: `miniminibi.py:339`

### __init__ `def __init__(self)`
- Defined: `miniminibi.py:353`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `miniminibi.py:363`
- Doc: Mide qué tan bien está fluyendo información entre hemisferios

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `miniminibi.py:372`
- Doc: Mide diversidad de vocabulario generado (evitar colapso)

### measure_gate_health `def measure_gate_health(self, gate_activations)`
- Defined: `miniminibi.py:378`
- Doc: Verifica que el liquid gate no colapse a 0 o 1

### update `def update(self)`
- Defined: `miniminibi.py:387`

### report `def report(self, epoch)`
- Defined: `miniminibi.py:392`
- Doc: Reporte diagnóstico completo

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `miniminibi.py:443`

### __len__ `def __len__(self)`
- Defined: `miniminibi.py:462`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `miniminibi.py:465`

### __init__ `def __init__(self, total_epochs)`
- Defined: `miniminibi.py:509`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `miniminibi.py:512`

## nemesis.py

### seed_everything `def seed_everything(seed)`
- Defined: `nemesis.py:13`

### run_hyper_experiment `def run_hyper_experiment(epochs, name, dynamic_mode)`
- Defined: `nemesis.py:143`

### __init__ `def __init__(self)`
- Defined: `nemesis.py:24`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `nemesis.py:33`

### __init__ `def __init__(self)`
- Defined: `nemesis.py:42`

### forward `def forward(self, surprise, entropy)`
- Defined: `nemesis.py:52`

### __init__ `def __init__(self, d_in, d_out, dynamic_mode)`
- Defined: `nemesis.py:61`

### forward `def forward(self, x)`
- Defined: `nemesis.py:79`

### __init__ `def __init__(self, config, dynamic_mode)`
- Defined: `nemesis.py:127`

### forward `def forward(self, x)`
- Defined: `nemesis.py:133`

## nested1.1.py

### safe_serialize `def safe_serialize(obj)`
- Defined: `nested1.1.py:61`
- Doc: Convierte objetos a formato serializable (evita recursión y objetos complejos).

### save_checkpoint `def save_checkpoint(epoch, model_state, optimizer_state, config, metrics, checkpoint_dir)`
- Defined: `nested1.1.py:80`
- Doc: Guarda checkpoint de época: modelo (.pth) + metadatos (.pkl).

### cleanup_old_checkpoints `def cleanup_old_checkpoints(checkpoint_dir, keep_last)`
- Defined: `nested1.1.py:105`
- Doc: Mantiene solo los últimos `keep_last` checkpoints.

### get_cifar10_loaders `def get_cifar10_loaders(config)`
- Defined: `nested1.1.py:241`

### evaluate `def evaluate(model, loader, device)`
- Defined: `nested1.1.py:267`

### train `def train(config)`
- Defined: `nested1.1.py:293`

### run_ablation_study `def run_ablation_study()`
- Defined: `nested1.1.py:371`

### __init__ `def __init__(self, dim, config)`
- Defined: `nested1.1.py:129`

### forward `def forward(self, x)`
- Defined: `nested1.1.py:148`

### get_norms `def get_norms(self)`
- Defined: `nested1.1.py:190`

### __init__ `def __init__(self, config)`
- Defined: `nested1.1.py:201`

### forward `def forward(self, x)`
- Defined: `nested1.1.py:221`

### get_ablation_state `def get_ablation_state(self)`
- Defined: `nested1.1.py:226`

### get_norms `def get_norms(self)`
- Defined: `nested1.1.py:233`

## nested1.py

### get_cifar10_loaders `def get_cifar10_loaders(config)`
- Defined: `nested1.py:182`

### evaluate `def evaluate(model, loader, device)`
- Defined: `nested1.py:208`

### train `def train(config)`
- Defined: `nested1.py:234`

### run_ablation_study `def run_ablation_study()`
- Defined: `nested1.py:301`

### __init__ `def __init__(self, dim, config)`
- Defined: `nested1.py:60`

### forward `def forward(self, x)`
- Defined: `nested1.py:83`

### get_norms `def get_norms(self)`
- Defined: `nested1.py:131`

### __init__ `def __init__(self, config)`
- Defined: `nested1.py:142`

### forward `def forward(self, x)`
- Defined: `nested1.py:162`

### get_ablation_state `def get_ablation_state(self)`
- Defined: `nested1.py:167`

### get_norms `def get_norms(self)`
- Defined: `nested1.py:174`

## nestedtopobrain.py

### seed_everything `def seed_everything(seed)`
- Defined: `nestedtopobrain.py:119`

### get_dataloaders `def get_dataloaders(config)`
- Defined: `nestedtopobrain.py:481`
- Doc: DataLoaders con augmentation de alto rendimiento para CIFAR-10

### save_topology_visualization `def save_topology_visualization(model, epoch, run_name)`
- Defined: `nestedtopobrain.py:1406`
- Doc: Visualización v18 completa

### save_node_importance_viz `def save_node_importance_viz(model, epoch, run_name)`
- Defined: `nestedtopobrain.py:1450`
- Doc: Visualización de importancia de nodos v18

### analyze_topology_clustering `def analyze_topology_clustering(model, run_name)`
- Defined: `nestedtopobrain.py:1472`
- Doc: Clustering espectral v18

### analyze_topology_flow `def analyze_topology_flow(model, dataloader, run_name, num_samples)`
- Defined: `nestedtopobrain.py:1511`
- Doc: Análisis de flujo de información con captura genérica de outputs

### visualize_topology_as_graph `def visualize_topology_as_graph(model, run_name, threshold)`
- Defined: `nestedtopobrain.py:1577`
- Doc: Grafo v18 con métricas

### analyze_topology_evolution `def analyze_topology_evolution(run_name)`
- Defined: `nestedtopobrain.py:1629`
- Doc: Análisis temporal completo v18

### comprehensive_topology_analysis `def comprehensive_topology_analysis(model, dataloader, run_name)`
- Defined: `nestedtopobrain.py:1690`
- Doc: Suite completa de análisis v18

### run_ablation_study `def run_ablation_study()`
- Defined: `nestedtopobrain.py:1713`
- Doc: Suite de ablación v18 completa

### visualize_memory_evolution `def visualize_memory_evolution(model, epoch, run_name)`
- Defined: `nestedtopobrain.py:1814`
- Doc: Visualiza evolución de memorias semánticas

### analyze_gradient_flow `def analyze_gradient_flow(model, epoch, run_name)`
- Defined: `nestedtopobrain.py:1905`
- Doc: Análisis detallado del flujo de gradientes

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)`
- Defined: `nestedtopobrain.py:1972`
- Doc: PGD ataque con congelamiento total de pesos y detach explícito de estados.

### evaluate `def evaluate(model, loader, config, adversarial, controls)`
- Defined: `nestedtopobrain.py:2050`
- Doc: Evaluación con plasticidad residual (test-time adaptation)

### train_epoch `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)`
- Defined: `nestedtopobrain.py:2122`
- Doc: Entrenamiento homeostático con lista manual en lugar de deque

### train_model `def train_model(config, run_name)`
- Defined: `nestedtopobrain.py:2347`

### main `def main()`
- Defined: `nestedtopobrain.py:2518`
- Doc: CLI v24 completo con Orquestador Prefrontal

### __post_init__ `def __post_init__(self)`
- Defined: `nestedtopobrain.py:89`

### to_dict `def to_dict(self)`
- Defined: `nestedtopobrain.py:95`

### get_supcon_lambda `def get_supcon_lambda(self, epoch)`
- Defined: `nestedtopobrain.py:98`

### get_sparsity_lambda `def get_sparsity_lambda(self, epoch)`
- Defined: `nestedtopobrain.py:104`

### get_memory_gb `def get_memory_gb()`
- Defined: `nestedtopobrain.py:135`

### get_gpu_memory_gb `def get_gpu_memory_gb()`
- Defined: `nestedtopobrain.py:140`

### log `def log(prefix)`
- Defined: `nestedtopobrain.py:146`

### clear_cache `def clear_cache()`
- Defined: `nestedtopobrain.py:153`

### check_limit `def check_limit(limit_gb, abort_on_limit)`
- Defined: `nestedtopobrain.py:159`

### __init__ `def __init__(self, config)`
- Defined: `nestedtopobrain.py:185`

### forward `def forward(self, metrics_dict)`
- Defined: `nestedtopobrain.py:222`
- Doc: Orquestador v27: Allostasis con Frenado de Emergencia (Gradient-Aware).

### detach_state `def detach_state(self)`
- Defined: `nestedtopobrain.py:322`
- Doc: Rompe el grafo computacional para evitar retropropagación infinita entre batches

### reset_context `def reset_context(self)`
- Defined: `nestedtopobrain.py:327`
- Doc: Resetear contexto al inicio de cada época

### __init__ `def __init__(self, model, config, epsilon_c)`
- Defined: `nestedtopobrain.py:350`

### _analyze_matrix `def _analyze_matrix(self, weight_matrix, name)`
- Defined: `nestedtopobrain.py:356`

### calculate `def calculate(self, epoch)`
- Defined: `nestedtopobrain.py:402`

### get_critical_summary `def get_critical_summary(self)`
- Defined: `nestedtopobrain.py:419`

### __init__ `def __init__(self, run_name)`
- Defined: `nestedtopobrain.py:431`

### save `def save(self, data, name)`
- Defined: `nestedtopobrain.py:436`

### load `def load(self, name)`
- Defined: `nestedtopobrain.py:464`

### __init__ `def __init__(self, temperature)`
- Defined: `nestedtopobrain.py:526`

### forward `def forward(self, features, labels)`
- Defined: `nestedtopobrain.py:529`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `nestedtopobrain.py:543`

### forward `def forward(self, input_signal, prediction)`
- Defined: `nestedtopobrain.py:553`

### __init__ `def __init__(self, dim, min_gate)`
- Defined: `nestedtopobrain.py:563`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `nestedtopobrain.py:573`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `nestedtopobrain.py:581`

### _maintain_orthogonality `def _maintain_orthogonality(self)`
- Defined: `nestedtopobrain.py:595`

### forward `def forward(self, x)`
- Defined: `nestedtopobrain.py:600`

### __init__ `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)`
- Defined: `nestedtopobrain.py:626`

### forward `def forward(self, x, state_M, controls)`
- Defined: `nestedtopobrain.py:675`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- Defined: `nestedtopobrain.py:751`

### invalidate_sparse_cache `def invalidate_sparse_cache(self)`
- Defined: `nestedtopobrain.py:789`

### _validate_and_fix_state `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)`
- Defined: `nestedtopobrain.py:792`

### get_node_importance `def get_node_importance(self)`
- Defined: `nestedtopobrain.py:811`
- Doc: FIX: Método faltante para obtener importancia de nodos

### forward `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)`
- Defined: `nestedtopobrain.py:817`
- Doc: Forward con señales de control del Orquestador y retorno de ortho deviation

### __init__ `def __init__(self, in_channels, out_channels, stride)`
- Defined: `nestedtopobrain.py:947`

### forward `def forward(self, x)`
- Defined: `nestedtopobrain.py:961`

### __init__ `def __init__(self, output_dim, grid_size)`
- Defined: `nestedtopobrain.py:968`

### forward `def forward(self, x)`
- Defined: `nestedtopobrain.py:994`

### __init__ `def __init__(self, config, in_channels)`
- Defined: `nestedtopobrain.py:1023`

### initialize_memories `def initialize_memories(self, dataloader)`
- Defined: `nestedtopobrain.py:1087`

### _initialize_layer_memory `def _initialize_layer_memory(self, cell, x_input, name)`
- Defined: `nestedtopobrain.py:1124`

### consolidate_semantic_memories `def consolidate_semantic_memories(self)`
- Defined: `nestedtopobrain.py:1141`

### set_epoch `def set_epoch(self, epoch)`
- Defined: `nestedtopobrain.py:1169`

### calculate_ortho_loss `def calculate_ortho_loss(self, ortho_deviation, controls)`
- Defined: `nestedtopobrain.py:1174`

### calculate_topology_diversity_loss `def calculate_topology_diversity_loss(self, controls)`
- Defined: `nestedtopobrain.py:1180`

### _init_grid_topology `def _init_grid_topology(self, N)`
- Defined: `nestedtopobrain.py:1193`

### get_topology `def get_topology(self, return_sparse)`
- Defined: `nestedtopobrain.py:1215`

### forward `def forward(self, x, prev_states, controls)`
- Defined: `nestedtopobrain.py:1226`

### prune_topology `def prune_topology(self, controls)`
- Defined: `nestedtopobrain.py:1298`

### warmup_topo `def warmup_topo(epoch)`
- Defined: `nestedtopobrain.py:2381`

## nestedtopobrain_v1.py

### seed_everything `def seed_everything(seed)`
- Defined: `nestedtopobrain_v1.py:114`

### get_dataloaders `def get_dataloaders(config)`
- Defined: `nestedtopobrain_v1.py:416`

### save_topology_visualization `def save_topology_visualization(model, epoch, run_name)`
- Defined: `nestedtopobrain_v1.py:1304`
- Doc: Visualización v18 completa

### save_node_importance_viz `def save_node_importance_viz(model, epoch, run_name)`
- Defined: `nestedtopobrain_v1.py:1348`
- Doc: Visualización de importancia de nodos v18

### analyze_topology_clustering `def analyze_topology_clustering(model, run_name)`
- Defined: `nestedtopobrain_v1.py:1370`
- Doc: Clustering espectral v18

### analyze_topology_flow `def analyze_topology_flow(model, dataloader, run_name, num_samples)`
- Defined: `nestedtopobrain_v1.py:1409`
- Doc: Análisis de flujo v18

### visualize_topology_as_graph `def visualize_topology_as_graph(model, run_name, threshold)`
- Defined: `nestedtopobrain_v1.py:1461`
- Doc: Grafo v18 con métricas

### analyze_topology_evolution `def analyze_topology_evolution(run_name)`
- Defined: `nestedtopobrain_v1.py:1513`
- Doc: Análisis temporal completo v18

### comprehensive_topology_analysis `def comprehensive_topology_analysis(model, dataloader, run_name)`
- Defined: `nestedtopobrain_v1.py:1574`
- Doc: Suite completa de análisis v18

### run_ablation_study `def run_ablation_study()`
- Defined: `nestedtopobrain_v1.py:1597`
- Doc: Suite de ablación v18 completa

### visualize_memory_evolution `def visualize_memory_evolution(model, epoch, run_name)`
- Defined: `nestedtopobrain_v1.py:1698`
- Doc: Visualiza evolución de memorias semánticas

### analyze_gradient_flow `def analyze_gradient_flow(model, epoch, run_name)`
- Defined: `nestedtopobrain_v1.py:1789`
- Doc: Análisis detallado del flujo de gradientes

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)`
- Defined: `nestedtopobrain_v1.py:1856`
- Doc: PGD attack con congelamiento total de pesos (Protocolo de Aislamiento Sináptico).

### evaluate `def evaluate(model, loader, config, adversarial, controls)`
- Defined: `nestedtopobrain_v1.py:1936`
- Doc: Evaluación optimizada para arquitecturas biológicas complejas (Nested/Grid).

### train_epoch `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)`
- Defined: `nestedtopobrain_v1.py:2000`
- Doc: Entrenamiento homeostático con gestión rigurosa de grafos y memoria.

### train_model `def train_model(config, run_name)`
- Defined: `nestedtopobrain_v1.py:2196`
- Doc: Training loop principal - FIX v24: Orden correcto de Scheduler y gestión de memoria.

### main `def main()`
- Defined: `nestedtopobrain_v1.py:2398`
- Doc: CLI v24 completo con Orquestador Prefrontal

### __post_init__ `def __post_init__(self)`
- Defined: `nestedtopobrain_v1.py:83`

### to_dict `def to_dict(self)`
- Defined: `nestedtopobrain_v1.py:89`

### get_supcon_lambda `def get_supcon_lambda(self, epoch)`
- Defined: `nestedtopobrain_v1.py:92`

### get_sparsity_lambda `def get_sparsity_lambda(self, epoch)`
- Defined: `nestedtopobrain_v1.py:98`

### get_memory_gb `def get_memory_gb()`
- Defined: `nestedtopobrain_v1.py:130`

### get_gpu_memory_gb `def get_gpu_memory_gb()`
- Defined: `nestedtopobrain_v1.py:135`

### log `def log(prefix)`
- Defined: `nestedtopobrain_v1.py:141`

### clear_cache `def clear_cache()`
- Defined: `nestedtopobrain_v1.py:148`

### check_limit `def check_limit(limit_gb, abort_on_limit)`
- Defined: `nestedtopobrain_v1.py:154`

### __init__ `def __init__(self, config)`
- Defined: `nestedtopobrain_v1.py:180`

### forward `def forward(self, metrics_dict)`
- Defined: `nestedtopobrain_v1.py:217`
- Doc: Input: Diccionario con métricas del estado actual

### detach_state `def detach_state(self)`
- Defined: `nestedtopobrain_v1.py:257`
- Doc: Rompe el grafo computacional para evitar retropropagación infinita entre batches

### reset_context `def reset_context(self)`
- Defined: `nestedtopobrain_v1.py:262`
- Doc: Resetear contexto al inicio de cada época

### __init__ `def __init__(self, model, config, epsilon_c)`
- Defined: `nestedtopobrain_v1.py:285`

### _analyze_matrix `def _analyze_matrix(self, weight_matrix, name)`
- Defined: `nestedtopobrain_v1.py:291`

### calculate `def calculate(self, epoch)`
- Defined: `nestedtopobrain_v1.py:337`

### get_critical_summary `def get_critical_summary(self)`
- Defined: `nestedtopobrain_v1.py:354`

### __init__ `def __init__(self, run_name)`
- Defined: `nestedtopobrain_v1.py:366`

### save `def save(self, data, name)`
- Defined: `nestedtopobrain_v1.py:371`

### load `def load(self, name)`
- Defined: `nestedtopobrain_v1.py:399`

### __init__ `def __init__(self, temperature)`
- Defined: `nestedtopobrain_v1.py:456`

### forward `def forward(self, features, labels)`
- Defined: `nestedtopobrain_v1.py:459`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `nestedtopobrain_v1.py:473`

### forward `def forward(self, input_signal, prediction)`
- Defined: `nestedtopobrain_v1.py:483`

### __init__ `def __init__(self, dim, min_gate)`
- Defined: `nestedtopobrain_v1.py:493`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `nestedtopobrain_v1.py:503`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `nestedtopobrain_v1.py:511`

### _maintain_orthogonality `def _maintain_orthogonality(self)`
- Defined: `nestedtopobrain_v1.py:525`

### forward `def forward(self, x)`
- Defined: `nestedtopobrain_v1.py:530`

### __init__ `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)`
- Defined: `nestedtopobrain_v1.py:550`

### forward `def forward(self, x, state_M, controls)`
- Defined: `nestedtopobrain_v1.py:599`
- Doc: FIX: Ahora acepta señales de control del Orquestador para modular

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- Defined: `nestedtopobrain_v1.py:677`

### invalidate_sparse_cache `def invalidate_sparse_cache(self)`
- Defined: `nestedtopobrain_v1.py:715`

### _validate_and_fix_state `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)`
- Defined: `nestedtopobrain_v1.py:718`

### get_node_importance `def get_node_importance(self)`
- Defined: `nestedtopobrain_v1.py:737`
- Doc: FIX: Método faltante para obtener importancia de nodos

### forward `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)`
- Defined: `nestedtopobrain_v1.py:743`
- Doc: FIX: Integra señales de control del Orquestador con bypass condicional

### __init__ `def __init__(self, config, in_channels)`
- Defined: `nestedtopobrain_v1.py:880`

### initialize_memories `def initialize_memories(self, dataloader)`
- Defined: `nestedtopobrain_v1.py:934`

### _initialize_layer_memory `def _initialize_layer_memory(self, cell, x_input, name)`
- Defined: `nestedtopobrain_v1.py:974`

### consolidate_semantic_memories `def consolidate_semantic_memories(self)`
- Defined: `nestedtopobrain_v1.py:1000`

### set_epoch `def set_epoch(self, epoch)`
- Defined: `nestedtopobrain_v1.py:1045`

### calculate_ortho_loss `def calculate_ortho_loss(self, controls)`
- Defined: `nestedtopobrain_v1.py:1050`

### calculate_topology_diversity_loss `def calculate_topology_diversity_loss(self, controls)`
- Defined: `nestedtopobrain_v1.py:1070`

### _init_grid_topology `def _init_grid_topology(self, N)`
- Defined: `nestedtopobrain_v1.py:1095`

### get_topology `def get_topology(self, return_sparse)`
- Defined: `nestedtopobrain_v1.py:1118`

### forward `def forward(self, x, prev_states, controls)`
- Defined: `nestedtopobrain_v1.py:1131`
- Doc: Forward con validación y detach explícito

### prune_topology `def prune_topology(self, controls)`
- Defined: `nestedtopobrain_v1.py:1198`

### warmup_topo `def warmup_topo(epoch)`
- Defined: `nestedtopobrain_v1.py:2234`

## nestedtopobrain_v2.py

### seed_everything `def seed_everything(seed)`
- Defined: `nestedtopobrain_v2.py:113`

### get_dataloaders `def get_dataloaders(config)`
- Defined: `nestedtopobrain_v2.py:415`

### save_topology_visualization `def save_topology_visualization(model, epoch, run_name)`
- Defined: `nestedtopobrain_v2.py:1289`
- Doc: Visualización v18 completa

### save_node_importance_viz `def save_node_importance_viz(model, epoch, run_name)`
- Defined: `nestedtopobrain_v2.py:1333`
- Doc: Visualización de importancia de nodos v18

### analyze_topology_clustering `def analyze_topology_clustering(model, run_name)`
- Defined: `nestedtopobrain_v2.py:1355`
- Doc: Clustering espectral v18

### analyze_topology_flow `def analyze_topology_flow(model, dataloader, run_name, num_samples)`
- Defined: `nestedtopobrain_v2.py:1394`
- Doc: Análisis de flujo v18

### visualize_topology_as_graph `def visualize_topology_as_graph(model, run_name, threshold)`
- Defined: `nestedtopobrain_v2.py:1448`
- Doc: Grafo v18 con métricas

### analyze_topology_evolution `def analyze_topology_evolution(run_name)`
- Defined: `nestedtopobrain_v2.py:1500`
- Doc: Análisis temporal completo v18

### comprehensive_topology_analysis `def comprehensive_topology_analysis(model, dataloader, run_name)`
- Defined: `nestedtopobrain_v2.py:1561`
- Doc: Suite completa de análisis v18

### run_ablation_study `def run_ablation_study()`
- Defined: `nestedtopobrain_v2.py:1584`
- Doc: Suite de ablación v18 completa

### visualize_memory_evolution `def visualize_memory_evolution(model, epoch, run_name)`
- Defined: `nestedtopobrain_v2.py:1685`
- Doc: Visualiza evolución de memorias semánticas

### analyze_gradient_flow `def analyze_gradient_flow(model, epoch, run_name)`
- Defined: `nestedtopobrain_v2.py:1776`
- Doc: Análisis detallado del flujo de gradientes

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)`
- Defined: `nestedtopobrain_v2.py:1843`
- Doc: PGD attack con congelamiento total de pesos (Protocolo de Aislamiento Sináptico).

### evaluate `def evaluate(model, loader, config, adversarial, controls)`
- Defined: `nestedtopobrain_v2.py:1910`
- Doc: Evaluación optimizada con Gradient Shielding.

### train_epoch `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)`
- Defined: `nestedtopobrain_v2.py:1963`
- Doc: Entrenamiento homeostático con inicialización de estados

### train_model `def train_model(config, run_name)`
- Defined: `nestedtopobrain_v2.py:2132`
- Doc: Training loop principal - FIX v24: Orden correcto de Scheduler y gestión de memoria.

### main `def main()`
- Defined: `nestedtopobrain_v2.py:2344`
- Doc: CLI v24 completo con Orquestador Prefrontal

### __post_init__ `def __post_init__(self)`
- Defined: `nestedtopobrain_v2.py:82`

### to_dict `def to_dict(self)`
- Defined: `nestedtopobrain_v2.py:88`

### get_supcon_lambda `def get_supcon_lambda(self, epoch)`
- Defined: `nestedtopobrain_v2.py:91`

### get_sparsity_lambda `def get_sparsity_lambda(self, epoch)`
- Defined: `nestedtopobrain_v2.py:97`

### get_memory_gb `def get_memory_gb()`
- Defined: `nestedtopobrain_v2.py:129`

### get_gpu_memory_gb `def get_gpu_memory_gb()`
- Defined: `nestedtopobrain_v2.py:134`

### log `def log(prefix)`
- Defined: `nestedtopobrain_v2.py:140`

### clear_cache `def clear_cache()`
- Defined: `nestedtopobrain_v2.py:147`

### check_limit `def check_limit(limit_gb, abort_on_limit)`
- Defined: `nestedtopobrain_v2.py:153`

### __init__ `def __init__(self, config)`
- Defined: `nestedtopobrain_v2.py:179`

### forward `def forward(self, metrics_dict)`
- Defined: `nestedtopobrain_v2.py:216`
- Doc: Input: Diccionario con métricas del estado actual

### detach_state `def detach_state(self)`
- Defined: `nestedtopobrain_v2.py:256`
- Doc: Rompe el grafo computacional para evitar retropropagación infinita entre batches

### reset_context `def reset_context(self)`
- Defined: `nestedtopobrain_v2.py:261`
- Doc: Resetear contexto al inicio de cada época

### __init__ `def __init__(self, model, config, epsilon_c)`
- Defined: `nestedtopobrain_v2.py:284`

### _analyze_matrix `def _analyze_matrix(self, weight_matrix, name)`
- Defined: `nestedtopobrain_v2.py:290`

### calculate `def calculate(self, epoch)`
- Defined: `nestedtopobrain_v2.py:336`

### get_critical_summary `def get_critical_summary(self)`
- Defined: `nestedtopobrain_v2.py:353`

### __init__ `def __init__(self, run_name)`
- Defined: `nestedtopobrain_v2.py:365`

### save `def save(self, data, name)`
- Defined: `nestedtopobrain_v2.py:370`

### load `def load(self, name)`
- Defined: `nestedtopobrain_v2.py:398`

### __init__ `def __init__(self, temperature)`
- Defined: `nestedtopobrain_v2.py:455`

### forward `def forward(self, features, labels)`
- Defined: `nestedtopobrain_v2.py:458`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `nestedtopobrain_v2.py:472`

### forward `def forward(self, input_signal, prediction)`
- Defined: `nestedtopobrain_v2.py:482`

### __init__ `def __init__(self, dim, min_gate)`
- Defined: `nestedtopobrain_v2.py:492`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `nestedtopobrain_v2.py:502`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `nestedtopobrain_v2.py:510`

### _maintain_orthogonality `def _maintain_orthogonality(self)`
- Defined: `nestedtopobrain_v2.py:524`

### forward `def forward(self, x)`
- Defined: `nestedtopobrain_v2.py:529`

### __init__ `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)`
- Defined: `nestedtopobrain_v2.py:555`

### forward `def forward(self, x, state_M, controls)`
- Defined: `nestedtopobrain_v2.py:604`
- Doc: FIX: Ahora acepta señales de control del Orquestador para modular

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- Defined: `nestedtopobrain_v2.py:682`

### invalidate_sparse_cache `def invalidate_sparse_cache(self)`
- Defined: `nestedtopobrain_v2.py:720`

### _validate_and_fix_state `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)`
- Defined: `nestedtopobrain_v2.py:723`

### get_node_importance `def get_node_importance(self)`
- Defined: `nestedtopobrain_v2.py:742`
- Doc: FIX: Método faltante para obtener importancia de nodos

### forward `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)`
- Defined: `nestedtopobrain_v2.py:748`
- Doc: Forward con señales de control del Orquestador y retorno de ortho deviation

### __init__ `def __init__(self, config, in_channels)`
- Defined: `nestedtopobrain_v2.py:879`

### initialize_memories `def initialize_memories(self, dataloader)`
- Defined: `nestedtopobrain_v2.py:933`
- Doc: Inicialización de memorias semánticas con captura correcta de 5 valores de retorno

### _initialize_layer_memory `def _initialize_layer_memory(self, cell, x_input, name)`
- Defined: `nestedtopobrain_v2.py:976`

### consolidate_semantic_memories `def consolidate_semantic_memories(self)`
- Defined: `nestedtopobrain_v2.py:1002`

### set_epoch `def set_epoch(self, epoch)`
- Defined: `nestedtopobrain_v2.py:1047`

### calculate_ortho_loss `def calculate_ortho_loss(self, ortho_deviation, controls)`
- Defined: `nestedtopobrain_v2.py:1052`
- Doc: Calcula loss de ortogonalidad usando el deviation retornado por las capas

### calculate_topology_diversity_loss `def calculate_topology_diversity_loss(self, controls)`
- Defined: `nestedtopobrain_v2.py:1060`

### _init_grid_topology `def _init_grid_topology(self, N)`
- Defined: `nestedtopobrain_v2.py:1085`

### get_topology `def get_topology(self, return_sparse)`
- Defined: `nestedtopobrain_v2.py:1108`

### forward `def forward(self, x, prev_states, controls)`
- Defined: `nestedtopobrain_v2.py:1121`
- Doc: Forward con validación, detach explícito, y retorno de ortho deviation

### prune_topology `def prune_topology(self, controls)`
- Defined: `nestedtopobrain_v2.py:1183`
- Doc: Poda topológica con cálculo correcto de quantile

### warmup_topo `def warmup_topo(epoch)`
- Defined: `nestedtopobrain_v2.py:2180`

## nestedtopobrain_v3.py

### seed_everything `def seed_everything(seed)`
- Defined: `nestedtopobrain_v3.py:113`

### get_dataloaders `def get_dataloaders(config)`
- Defined: `nestedtopobrain_v3.py:471`

### save_topology_visualization `def save_topology_visualization(model, epoch, run_name)`
- Defined: `nestedtopobrain_v3.py:1388`
- Doc: Visualización v18 completa

### save_node_importance_viz `def save_node_importance_viz(model, epoch, run_name)`
- Defined: `nestedtopobrain_v3.py:1432`
- Doc: Visualización de importancia de nodos v18

### analyze_topology_clustering `def analyze_topology_clustering(model, run_name)`
- Defined: `nestedtopobrain_v3.py:1454`
- Doc: Clustering espectral v18

### analyze_topology_flow `def analyze_topology_flow(model, dataloader, run_name, num_samples)`
- Defined: `nestedtopobrain_v3.py:1493`
- Doc: Análisis de flujo de información con captura genérica de outputs

### visualize_topology_as_graph `def visualize_topology_as_graph(model, run_name, threshold)`
- Defined: `nestedtopobrain_v3.py:1559`
- Doc: Grafo v18 con métricas

### analyze_topology_evolution `def analyze_topology_evolution(run_name)`
- Defined: `nestedtopobrain_v3.py:1611`
- Doc: Análisis temporal completo v18

### comprehensive_topology_analysis `def comprehensive_topology_analysis(model, dataloader, run_name)`
- Defined: `nestedtopobrain_v3.py:1672`
- Doc: Suite completa de análisis v18

### run_ablation_study `def run_ablation_study()`
- Defined: `nestedtopobrain_v3.py:1695`
- Doc: Suite de ablación v18 completa

### visualize_memory_evolution `def visualize_memory_evolution(model, epoch, run_name)`
- Defined: `nestedtopobrain_v3.py:1796`
- Doc: Visualiza evolución de memorias semánticas

### analyze_gradient_flow `def analyze_gradient_flow(model, epoch, run_name)`
- Defined: `nestedtopobrain_v3.py:1887`
- Doc: Análisis detallado del flujo de gradientes

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)`
- Defined: `nestedtopobrain_v3.py:1954`
- Doc: PGD attack con congelamiento total de pesos (Protocolo de Aislamiento Sináptico).

### evaluate `def evaluate(model, loader, config, adversarial, controls)`
- Defined: `nestedtopobrain_v3.py:2021`
- Doc: Evaluación con plasticidad residual (test-time adaptation)

### train_epoch `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)`
- Defined: `nestedtopobrain_v3.py:2093`
- Doc: Entrenamiento homeostático con inicialización de estados y gestión de densidad

### train_model `def train_model(config, run_name)`
- Defined: `nestedtopobrain_v3.py:2279`
- Doc: Training loop principal - FIX v24: Orden correcto de Scheduler y gestión de memoria.

### main `def main()`
- Defined: `nestedtopobrain_v3.py:2491`
- Doc: CLI v24 completo con Orquestador Prefrontal

### __post_init__ `def __post_init__(self)`
- Defined: `nestedtopobrain_v3.py:82`

### to_dict `def to_dict(self)`
- Defined: `nestedtopobrain_v3.py:88`

### get_supcon_lambda `def get_supcon_lambda(self, epoch)`
- Defined: `nestedtopobrain_v3.py:91`

### get_sparsity_lambda `def get_sparsity_lambda(self, epoch)`
- Defined: `nestedtopobrain_v3.py:97`

### get_memory_gb `def get_memory_gb()`
- Defined: `nestedtopobrain_v3.py:129`

### get_gpu_memory_gb `def get_gpu_memory_gb()`
- Defined: `nestedtopobrain_v3.py:134`

### log `def log(prefix)`
- Defined: `nestedtopobrain_v3.py:140`

### clear_cache `def clear_cache()`
- Defined: `nestedtopobrain_v3.py:147`

### check_limit `def check_limit(limit_gb, abort_on_limit)`
- Defined: `nestedtopobrain_v3.py:153`

### __init__ `def __init__(self, config)`
- Defined: `nestedtopobrain_v3.py:179`

### forward `def forward(self, metrics_dict)`
- Defined: `nestedtopobrain_v3.py:216`
- Doc: Orquestador v26: Allostasis (Adaptación Predictiva Valiente).

### detach_state `def detach_state(self)`
- Defined: `nestedtopobrain_v3.py:312`
- Doc: Rompe el grafo computacional para evitar retropropagación infinita entre batches

### reset_context `def reset_context(self)`
- Defined: `nestedtopobrain_v3.py:317`
- Doc: Resetear contexto al inicio de cada época

### __init__ `def __init__(self, model, config, epsilon_c)`
- Defined: `nestedtopobrain_v3.py:340`

### _analyze_matrix `def _analyze_matrix(self, weight_matrix, name)`
- Defined: `nestedtopobrain_v3.py:346`

### calculate `def calculate(self, epoch)`
- Defined: `nestedtopobrain_v3.py:392`

### get_critical_summary `def get_critical_summary(self)`
- Defined: `nestedtopobrain_v3.py:409`

### __init__ `def __init__(self, run_name)`
- Defined: `nestedtopobrain_v3.py:421`

### save `def save(self, data, name)`
- Defined: `nestedtopobrain_v3.py:426`

### load `def load(self, name)`
- Defined: `nestedtopobrain_v3.py:454`

### __init__ `def __init__(self, temperature)`
- Defined: `nestedtopobrain_v3.py:511`

### forward `def forward(self, features, labels)`
- Defined: `nestedtopobrain_v3.py:514`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `nestedtopobrain_v3.py:528`

### forward `def forward(self, input_signal, prediction)`
- Defined: `nestedtopobrain_v3.py:538`

### __init__ `def __init__(self, dim, min_gate)`
- Defined: `nestedtopobrain_v3.py:548`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `nestedtopobrain_v3.py:558`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `nestedtopobrain_v3.py:566`

### _maintain_orthogonality `def _maintain_orthogonality(self)`
- Defined: `nestedtopobrain_v3.py:580`

### forward `def forward(self, x)`
- Defined: `nestedtopobrain_v3.py:585`

### __init__ `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)`
- Defined: `nestedtopobrain_v3.py:611`

### forward `def forward(self, x, state_M, controls)`
- Defined: `nestedtopobrain_v3.py:660`
- Doc: FIX: Ahora acepta señales de control del Orquestador para modular

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- Defined: `nestedtopobrain_v3.py:738`

### invalidate_sparse_cache `def invalidate_sparse_cache(self)`
- Defined: `nestedtopobrain_v3.py:776`

### _validate_and_fix_state `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)`
- Defined: `nestedtopobrain_v3.py:779`

### get_node_importance `def get_node_importance(self)`
- Defined: `nestedtopobrain_v3.py:798`
- Doc: FIX: Método faltante para obtener importancia de nodos

### forward `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)`
- Defined: `nestedtopobrain_v3.py:804`
- Doc: Forward con señales de control del Orquestador y retorno de ortho deviation

### __init__ `def __init__(self, config, in_channels)`
- Defined: `nestedtopobrain_v3.py:935`

### initialize_memories `def initialize_memories(self, dataloader)`
- Defined: `nestedtopobrain_v3.py:989`
- Doc: Inicialización de memorias semánticas con captura correcta de 5 valores de retorno

### _initialize_layer_memory `def _initialize_layer_memory(self, cell, x_input, name)`
- Defined: `nestedtopobrain_v3.py:1032`

### consolidate_semantic_memories `def consolidate_semantic_memories(self)`
- Defined: `nestedtopobrain_v3.py:1058`

### set_epoch `def set_epoch(self, epoch)`
- Defined: `nestedtopobrain_v3.py:1103`

### calculate_ortho_loss `def calculate_ortho_loss(self, ortho_deviation, controls)`
- Defined: `nestedtopobrain_v3.py:1108`
- Doc: Calcula loss de ortogonalidad usando el deviation retornado por las capas

### calculate_topology_diversity_loss `def calculate_topology_diversity_loss(self, controls)`
- Defined: `nestedtopobrain_v3.py:1116`

### _init_grid_topology `def _init_grid_topology(self, N)`
- Defined: `nestedtopobrain_v3.py:1141`

### get_topology `def get_topology(self, return_sparse)`
- Defined: `nestedtopobrain_v3.py:1164`

### forward `def forward(self, x, prev_states, controls)`
- Defined: `nestedtopobrain_v3.py:1177`
- Doc: Forward con validación, detach explícito, y retorno de ortho deviation

### prune_topology `def prune_topology(self, controls)`
- Defined: `nestedtopobrain_v3.py:1239`
- Doc: Poda topológica con protocolo de supervivencia garantizado y neurogénesis

### warmup_topo `def warmup_topo(epoch)`
- Defined: `nestedtopobrain_v3.py:2327`

## neurologitos.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologitos.py:29`
- Doc: Control total de reproducibilidad

### compute_effect_size `def compute_effect_size(group1, group2)`
- Defined: `neurologitos.py:38`
- Doc: Cohen's d con corrección de sesgo

### train_epoch_cv `def train_epoch_cv(model, loader, optimizer, config, epoch, vocab)`
- Defined: `neurologitos.py:451`

### evaluate_cv `def evaluate_cv(model, loader, config, vocab)`
- Defined: `neurologitos.py:507`

### train_with_cv `def train_with_cv(config, dataset, vocab)`
- Defined: `neurologitos.py:524`

### run_scientific_ablation `def run_scientific_ablation()`
- Defined: `neurologitos.py:579`

### to_dict `def to_dict(self)`
- Defined: `neurologitos.py:81`

### component_signature `def component_signature(self)`
- Defined: `neurologitos.py:84`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)`
- Defined: `neurologitos.py:96`

### _init_grid `def _init_grid(self)`
- Defined: `neurologitos.py:121`

### forward `def forward(self, x)`
- Defined: `neurologitos.py:128`

### get_metrics `def get_metrics(self)`
- Defined: `neurologitos.py:152`

### __init__ `def __init__(self, epsilon, alpha, steps)`
- Defined: `neurologitos.py:160`

### attack `def attack(self, model_fn, x, y, criterion)`
- Defined: `neurologitos.py:165`

### __init__ `def __init__(self, output_dim)`
- Defined: `neurologitos.py:188`

### forward `def forward(self, x)`
- Defined: `neurologitos.py:200`

### __init__ `def __init__(self, output_dim, use_grid, use_symbiotic)`
- Defined: `neurologitos.py:207`

### forward `def forward(self, x)`
- Defined: `neurologitos.py:221`

### get_metrics `def get_metrics(self)`
- Defined: `neurologitos.py:225`

### __init__ `def __init__(self, dim)`
- Defined: `neurologitos.py:230`

### forward `def forward(self, x)`
- Defined: `neurologitos.py:235`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `neurologitos.py:242`

### forward `def forward(self, thought, captions, max_len)`
- Defined: `neurologitos.py:253`

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `neurologitos.py:278`

### __init__ `def __init__(self, vocab_size, config)`
- Defined: `neurologitos.py:285`

### forward `def forward(self, image, captions)`
- Defined: `neurologitos.py:309`

### get_metrics `def get_metrics(self)`
- Defined: `neurologitos.py:314`

### __init__ `def __init__(self)`
- Defined: `neurologitos.py:319`

### __len__ `def __len__(self)`
- Defined: `neurologitos.py:344`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `neurologitos.py:347`

### level1_isolated `def level1_isolated()`
- Defined: `neurologitos.py:360`

### level2_pairs `def level2_pairs()`
- Defined: `neurologitos.py:369`

### level3_full `def level3_full()`
- Defined: `neurologitos.py:377`

### level4_inverse `def level4_inverse()`
- Defined: `neurologitos.py:381`

### get_complete_matrix `def get_complete_matrix(cls)`
- Defined: `neurologitos.py:389`

### compute_statistics `def compute_statistics(cv_results)`
- Defined: `neurologitos.py:395`

### ttest_vs_baseline `def ttest_vs_baseline(exp_scores, baseline_scores)`
- Defined: `neurologitos.py:413`

### detect_synergy `def detect_synergy(pair_score, comp_a_score, comp_b_score, baseline_score)`
- Defined: `neurologitos.py:419`

### rank_criticality `def rank_criticality(full_score, ablation_results)`
- Defined: `neurologitos.py:432`

### model_fn `def model_fn(x_adv)`
- Defined: `neurologitos.py:471`

### crit_fn `def crit_fn(out, tgt)`
- Defined: `neurologitos.py:474`

## neurologos.py

### train_logos `def train_logos(use_nested)`
- Defined: `neurologos.py:261`

### __init__ `def __init__(self)`
- Defined: `neurologos.py:15`

### forward `def forward(self, x)`
- Defined: `neurologos.py:26`

### __init__ `def __init__(self, grid_size, output_dim)`
- Defined: `neurologos.py:31`

### forward `def forward(self, x)`
- Defined: `neurologos.py:57`

### __init__ `def __init__(self, dim)`
- Defined: `neurologos.py:82`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos.py:88`

### __init__ `def __init__(self)`
- Defined: `neurologos.py:97`

### forward `def forward(self, visual_features, plasticity)`
- Defined: `neurologos.py:103`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `neurologos.py:117`

### forward `def forward(self, thought, captions, max_len)`
- Defined: `neurologos.py:132`

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `neurologos.py:159`

### __init__ `def __init__(self, vocab_size, use_nested)`
- Defined: `neurologos.py:168`

### forward `def forward(self, image, captions, plasticity)`
- Defined: `neurologos.py:182`

### measure_richness `def measure_richness(self)`
- Defined: `neurologos.py:193`

### __init__ `def __init__(self, total_epochs)`
- Defined: `neurologos.py:201`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `neurologos.py:205`

### __init__ `def __init__(self)`
- Defined: `neurologos.py:221`

### __len__ `def __len__(self)`
- Defined: `neurologos.py:245`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `neurologos.py:248`

## neurologos_V1.py

### train_logos `def train_logos()`
- Defined: `neurologos_V1.py:237`

### __init__ `def __init__(self)`
- Defined: `neurologos_V1.py:16`

### forward `def forward(self, x)`
- Defined: `neurologos_V1.py:27`

### __init__ `def __init__(self, dim)`
- Defined: `neurologos_V1.py:34`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_V1.py:40`

### __init__ `def __init__(self)`
- Defined: `neurologos_V1.py:49`

### forward `def forward(self, visual_features, plasticity)`
- Defined: `neurologos_V1.py:55`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `neurologos_V1.py:74`

### forward `def forward(self, thought, captions, max_len, teacher_forcing_ratio)`
- Defined: `neurologos_V1.py:90`
- Doc: Modo entrenamiento: captions != None

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `neurologos_V1.py:131`

### __init__ `def __init__(self, vocab_size)`
- Defined: `neurologos_V1.py:141`

### forward `def forward(self, image, captions, plasticity)`
- Defined: `neurologos_V1.py:150`

### measure_richness `def measure_richness(self)`
- Defined: `neurologos_V1.py:163`

### __init__ `def __init__(self, total_epochs)`
- Defined: `neurologos_V1.py:171`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `neurologos_V1.py:175`

### __init__ `def __init__(self)`
- Defined: `neurologos_V1.py:190`

### __len__ `def __len__(self)`
- Defined: `neurologos_V1.py:216`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `neurologos_V1.py:219`

## neurologos_cpu_v7.py

### setup_device `def setup_device()`
- Defined: `neurologos_cpu_v7.py:65`

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_cpu_v7.py:68`

### get_dataset `def get_dataset(config)`
- Defined: `neurologos_cpu_v7.py:73`

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- Defined: `neurologos_cpu_v7.py:235`

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `neurologos_cpu_v7.py:276`

### run_ablation_study `def run_ablation_study()`
- Defined: `neurologos_cpu_v7.py:328`

### __init__ `def __init__(self, dim)`
- Defined: `neurologos_cpu_v7.py:97`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_cpu_v7.py:106`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `neurologos_cpu_v7.py:121`

### forward `def forward(self, x)`
- Defined: `neurologos_cpu_v7.py:128`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `neurologos_cpu_v7.py:136`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `neurologos_cpu_v7.py:149`

### __init__ `def __init__(self, temperature)`
- Defined: `neurologos_cpu_v7.py:155`

### forward `def forward(self, features, labels)`
- Defined: `neurologos_cpu_v7.py:158`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_cpu_v7.py:171`

### _init_weights `def _init_weights(self)`
- Defined: `neurologos_cpu_v7.py:195`

### count_parameters `def count_parameters(self)`
- Defined: `neurologos_cpu_v7.py:198`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_cpu_v7.py:199`

## neurologos_cpu_v8.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_cpu_v8.py:65`

### get_dataset `def get_dataset(config)`
- Defined: `neurologos_cpu_v8.py:72`

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- Defined: `neurologos_cpu_v8.py:308`
- Doc: PGD Attack - Versión ultra-simple que siempre funciona

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `neurologos_cpu_v8.py:344`
- Doc: Genera matriz de ablación de 3 niveles:

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `neurologos_cpu_v8.py:393`
- Doc: Entrenamiento con cross-validation

### run_ablation_study `def run_ablation_study()`
- Defined: `neurologos_cpu_v8.py:483`
- Doc: Ejecuta el estudio de ablación completo

### __init__ `def __init__(self, dim)`
- Defined: `neurologos_cpu_v8.py:94`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_cpu_v8.py:105`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `neurologos_cpu_v8.py:126`

### forward `def forward(self, x)`
- Defined: `neurologos_cpu_v8.py:134`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `neurologos_cpu_v8.py:148`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `neurologos_cpu_v8.py:164`

### __init__ `def __init__(self, temperature)`
- Defined: `neurologos_cpu_v8.py:171`

### forward `def forward(self, features, labels)`
- Defined: `neurologos_cpu_v8.py:176`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_cpu_v8.py:197`

### _init_weights `def _init_weights(self)`
- Defined: `neurologos_cpu_v8.py:240`

### count_parameters `def count_parameters(self)`
- Defined: `neurologos_cpu_v8.py:245`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_cpu_v8.py:248`

## neurologos_cpu_v9.py.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_cpu_v9.py.py:25`
- Doc: Control total de reproducibilidad

### compute_effect_size `def compute_effect_size(group1, group2)`
- Defined: `neurologos_cpu_v9.py.py:34`
- Doc: Cohen's d con corrección de sesgo

### train_epoch_cv `def train_epoch_cv(model, loader, optimizer, config, epoch, vocab)`
- Defined: `neurologos_cpu_v9.py.py:452`

### evaluate_cv `def evaluate_cv(model, loader, config, vocab)`
- Defined: `neurologos_cpu_v9.py.py:507`

### train_with_cv `def train_with_cv(config, dataset, vocab)`
- Defined: `neurologos_cpu_v9.py.py:523`

### run_scientific_ablation `def run_scientific_ablation()`
- Defined: `neurologos_cpu_v9.py.py:580`

### to_dict `def to_dict(self)`
- Defined: `neurologos_cpu_v9.py.py:76`

### component_signature `def component_signature(self)`
- Defined: `neurologos_cpu_v9.py.py:79`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)`
- Defined: `neurologos_cpu_v9.py.py:91`

### _init_grid `def _init_grid(self)`
- Defined: `neurologos_cpu_v9.py.py:116`

### forward `def forward(self, x)`
- Defined: `neurologos_cpu_v9.py.py:123`

### get_metrics `def get_metrics(self)`
- Defined: `neurologos_cpu_v9.py.py:147`

### __init__ `def __init__(self, epsilon, alpha, steps)`
- Defined: `neurologos_cpu_v9.py.py:154`

### attack `def attack(self, model_fn, x, y, criterion)`
- Defined: `neurologos_cpu_v9.py.py:159`

### __init__ `def __init__(self, output_dim)`
- Defined: `neurologos_cpu_v9.py.py:181`

### forward `def forward(self, x)`
- Defined: `neurologos_cpu_v9.py.py:193`

### __init__ `def __init__(self, output_dim, use_grid, use_symbiotic)`
- Defined: `neurologos_cpu_v9.py.py:199`

### forward `def forward(self, x)`
- Defined: `neurologos_cpu_v9.py.py:213`

### get_metrics `def get_metrics(self)`
- Defined: `neurologos_cpu_v9.py.py:217`

### __init__ `def __init__(self, dim)`
- Defined: `neurologos_cpu_v9.py.py:221`

### forward `def forward(self, x)`
- Defined: `neurologos_cpu_v9.py.py:226`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `neurologos_cpu_v9.py.py:232`

### forward `def forward(self, thought, captions, max_len)`
- Defined: `neurologos_cpu_v9.py.py:243`

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `neurologos_cpu_v9.py.py:268`

### __init__ `def __init__(self, vocab_size, config)`
- Defined: `neurologos_cpu_v9.py.py:277`

### forward `def forward(self, image, captions)`
- Defined: `neurologos_cpu_v9.py.py:304`

### get_metrics `def get_metrics(self)`
- Defined: `neurologos_cpu_v9.py.py:309`

### __init__ `def __init__(self)`
- Defined: `neurologos_cpu_v9.py.py:316`

### __len__ `def __len__(self)`
- Defined: `neurologos_cpu_v9.py.py:341`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `neurologos_cpu_v9.py.py:344`

### level1_isolated `def level1_isolated()`
- Defined: `neurologos_cpu_v9.py.py:360`

### level2_pairs `def level2_pairs()`
- Defined: `neurologos_cpu_v9.py.py:369`

### level3_full `def level3_full()`
- Defined: `neurologos_cpu_v9.py.py:377`

### level4_inverse `def level4_inverse()`
- Defined: `neurologos_cpu_v9.py.py:381`

### get_complete_matrix `def get_complete_matrix(cls)`
- Defined: `neurologos_cpu_v9.py.py:389`

### compute_statistics `def compute_statistics(cv_results)`
- Defined: `neurologos_cpu_v9.py.py:394`

### ttest_vs_baseline `def ttest_vs_baseline(exp_scores, baseline_scores)`
- Defined: `neurologos_cpu_v9.py.py:412`

### detect_synergy `def detect_synergy(pair_score, comp_a_score, comp_b_score, baseline_score)`
- Defined: `neurologos_cpu_v9.py.py:418`

### rank_criticality `def rank_criticality(full_score, ablation_results)`
- Defined: `neurologos_cpu_v9.py.py:431`

### model_fn `def model_fn(x_adv)`
- Defined: `neurologos_cpu_v9.py.py:472`

### crit_fn `def crit_fn(out, tgt)`
- Defined: `neurologos_cpu_v9.py.py:475`

## neurologos_entropico.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_entropico.py:50`

### train_nonstationary `def train_nonstationary(config)`
- Defined: `neurologos_entropico.py:263`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `neurologos_entropico.py:320`

### run_ablation_study `def run_ablation_study()`
- Defined: `neurologos_entropico.py:352`

### __init__ `def __init__(self)`
- Defined: `neurologos_entropico.py:61`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `neurologos_entropico.py:71`

### get_full `def get_full(self)`
- Defined: `neurologos_entropico.py:85`

### get_w2 `def get_w2(self)`
- Defined: `neurologos_entropico.py:88`

### __init__ `def __init__(self, d_in)`
- Defined: `neurologos_entropico.py:95`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `neurologos_entropico.py:105`

### __init__ `def __init__(self, d_in, d_out, dynamic)`
- Defined: `neurologos_entropico.py:121`

### forward `def forward(self, x)`
- Defined: `neurologos_entropico.py:132`

### __init__ `def __init__(self, dim, atoms)`
- Defined: `neurologos_entropico.py:161`

### forward `def forward(self, x, influence)`
- Defined: `neurologos_entropico.py:168`

### __init__ `def __init__(self, num_nodes)`
- Defined: `neurologos_entropico.py:180`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `neurologos_entropico.py:193`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_entropico.py:202`

### count_parameters `def count_parameters(self)`
- Defined: `neurologos_entropico.py:221`

### forward `def forward(self, x)`
- Defined: `neurologos_entropico.py:224`

## neurologos_fullhomesotatico_cpu_qw.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:65`

### get_dataset `def get_dataset(config)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:73`

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps, plasticity_ctrl)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:363`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:387`

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:407`

### run_ablation_study `def run_ablation_study()`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:472`

### __init__ `def __init__(self)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:94`

### forward `def forward(self, x, logits, h_agg, h_proc, w_norm, entropy, ortho)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:104`

### __init__ `def __init__(self, dim)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:138`

### forward `def forward(self, x, strength)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:148`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:162`

### forward `def forward(self, x, influence)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:170`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:184`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:197`

### __init__ `def __init__(self, temperature)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:204`

### forward `def forward(self, features, labels)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:209`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:228`

### _init_weights `def _init_weights(self)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:267`

### count_parameters `def count_parameters(self)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:272`

### forward `def forward(self, x)`
- Defined: `neurologos_fullhomesotatico_cpu_qw.py:275`

## neurologos_fullhomestatico_cpu_qw2.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:49`

### train_nonstationary `def train_nonstationary(config)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:269`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:331`

### run_ablation_study `def run_ablation_study()`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:364`

### __init__ `def __init__(self)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:61`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:71`

### get_full `def get_full(self)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:85`

### get_w2 `def get_w2(self)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:88`

### __init__ `def __init__(self, d_in)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:96`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:106`

### __init__ `def __init__(self, d_in, d_out, dynamic)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:123`

### forward `def forward(self, x)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:134`

### __init__ `def __init__(self, dim, atoms)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:165`

### forward `def forward(self, x, influence)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:172`

### __init__ `def __init__(self, num_nodes)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:185`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:198`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:208`

### count_parameters `def count_parameters(self)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:227`

### forward `def forward(self, x)`
- Defined: `neurologos_fullhomestatico_cpu_qw2.py:230`

## neurologos_gpu_v1.py

### train_logos `def train_logos()`
- Defined: `neurologos_gpu_v1.py:282`

### __init__ `def __init__(self, grid_size, hidden_dim)`
- Defined: `neurologos_gpu_v1.py:17`

### forward `def forward(self, x)`
- Defined: `neurologos_gpu_v1.py:46`

### __init__ `def __init__(self, dim)`
- Defined: `neurologos_gpu_v1.py:77`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_gpu_v1.py:83`

### __init__ `def __init__(self)`
- Defined: `neurologos_gpu_v1.py:92`

### forward `def forward(self, visual_features, plasticity)`
- Defined: `neurologos_gpu_v1.py:98`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `neurologos_gpu_v1.py:117`

### forward `def forward(self, thought, captions, max_len, teacher_forcing_ratio)`
- Defined: `neurologos_gpu_v1.py:133`
- Doc: Modo entrenamiento: captions != None

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `neurologos_gpu_v1.py:174`

### __init__ `def __init__(self, vocab_size)`
- Defined: `neurologos_gpu_v1.py:184`

### forward `def forward(self, image, captions, plasticity)`
- Defined: `neurologos_gpu_v1.py:193`

### measure_richness `def measure_richness(self)`
- Defined: `neurologos_gpu_v1.py:206`

### __init__ `def __init__(self, total_epochs)`
- Defined: `neurologos_gpu_v1.py:214`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `neurologos_gpu_v1.py:219`

### __init__ `def __init__(self)`
- Defined: `neurologos_gpu_v1.py:234`

### __len__ `def __len__(self)`
- Defined: `neurologos_gpu_v1.py:261`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `neurologos_gpu_v1.py:264`

## neurologos_homeostatico_cpu_cl.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_homeostatico_cpu_cl.py:71`

### get_dataset `def get_dataset(config)`
- Defined: `neurologos_homeostatico_cpu_cl.py:78`

### pgd_attack `def pgd_attack(model, x, y, eps, steps, plasticity)`
- Defined: `neurologos_homeostatico_cpu_cl.py:524`
- Doc: PGD Attack simplificado

### generate_homeostatic_ablation `def generate_homeostatic_ablation()`
- Defined: `neurologos_homeostatico_cpu_cl.py:555`
- Doc: Genera matriz enfocada en homeostasis con sensores mejorados

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `neurologos_homeostatico_cpu_cl.py:614`
- Doc: Entrenamiento con cross-validation

### run_homeostatic_ablation `def run_homeostatic_ablation()`
- Defined: `neurologos_homeostatico_cpu_cl.py:697`
- Doc: Ejecuta el estudio de ablación homeostático

### __init__ `def __init__(self, d_in)`
- Defined: `neurologos_homeostatico_cpu_cl.py:110`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `neurologos_homeostatico_cpu_cl.py:129`

### __init__ `def __init__(self, dim, use_homeostasis)`
- Defined: `neurologos_homeostatico_cpu_cl.py:220`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_homeostatico_cpu_cl.py:241`

### __init__ `def __init__(self, dim, num_atoms, use_homeostasis)`
- Defined: `neurologos_homeostatico_cpu_cl.py:283`

### forward `def forward(self, x)`
- Defined: `neurologos_homeostatico_cpu_cl.py:299`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `neurologos_homeostatico_cpu_cl.py:331`

### get_adjacency `def get_adjacency(self, x, plasticity)`
- Defined: `neurologos_homeostatico_cpu_cl.py:354`
- Doc: Genera adyacencia con regulación homeostática opcional

### __init__ `def __init__(self, temperature)`
- Defined: `neurologos_homeostatico_cpu_cl.py:382`

### forward `def forward(self, features, labels)`
- Defined: `neurologos_homeostatico_cpu_cl.py:387`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_homeostatico_cpu_cl.py:412`

### _init_weights `def _init_weights(self)`
- Defined: `neurologos_homeostatico_cpu_cl.py:456`

### count_parameters `def count_parameters(self)`
- Defined: `neurologos_homeostatico_cpu_cl.py:461`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_homeostatico_cpu_cl.py:464`

## neurologos_homeostatico_cpu_cl2.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:78`

### light_pgd_attack `def light_pgd_attack(model, x, y, eps, steps)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:424`
- Doc: PGD ligero para no dominar el entrenamiento

### train_trans_contextual `def train_trans_contextual(config, name)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:455`
- Doc: Entrenamiento en entorno no estacionario.

### generate_selective_ablation `def generate_selective_ablation()`
- Defined: `neurologos_homeostatico_cpu_cl2.py:556`
- Doc: Ablación selectiva basada en resultados v5.2:

### run_trans_contextual_study `def run_trans_contextual_study()`
- Defined: `neurologos_homeostatico_cpu_cl2.py:598`
- Doc: Ejecuta el estudio trans-contextual completo

### __init__ `def __init__(self)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:94`

### get_batch `def get_batch(self, phase, batch_size)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:115`
- Doc: Retorna batch según la fase del entrenamiento

### get_phase `def get_phase(self, epoch, total_epochs)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:135`
- Doc: Determina la fase según el epoch actual

### __init__ `def __init__(self, d_in, log_metrics)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:157`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:183`

### __init__ `def __init__(self, dim, use_homeostasis, log_metrics)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:256`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:277`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:325`

### _init_weights `def _init_weights(self)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:356`

### count_parameters `def count_parameters(self)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:361`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:364`

### get_homeostasis_metrics `def get_homeostasis_metrics(self)`
- Defined: `neurologos_homeostatico_cpu_cl2.py:401`
- Doc: Extrae métricas de homeostasis para logging

## neurologos_homeostatico_cpu_ki.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_homeostatico_cpu_ki.py:74`

### get_dataset `def get_dataset(config)`
- Defined: `neurologos_homeostatico_cpu_ki.py:85`
- Doc: Genera dataset sintético para el estudio de ablación.

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps, loss_val)`
- Defined: `neurologos_homeostatico_cpu_ki.py:521`
- Doc: PGD Attack - Versión ultra-simple que siempre funciona

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `neurologos_homeostatico_cpu_ki.py:556`
- Doc: Genera matriz de ablación de 3 niveles con configuración aislada por experimento

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `neurologos_homeostatico_cpu_ki.py:595`
- Doc: Entrenamiento con cross-validation

### run_ablation_study `def run_ablation_study()`
- Defined: `neurologos_homeostatico_cpu_ki.py:777`
- Doc: Ejecuta estudio con validación de integridad de resultados

### __init__ `def __init__(self, d_in, base_lr)`
- Defined: `neurologos_homeostatico_cpu_ki.py:118`

### forward `def forward(self, x, h_pre, w_norm, loss_val)`
- Defined: `neurologos_homeostatico_cpu_ki.py:137`

### __init__ `def __init__(self, dim, config)`
- Defined: `neurologos_homeostatico_cpu_ki.py:177`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_homeostatico_cpu_ki.py:198`

### __init__ `def __init__(self, dim, config)`
- Defined: `neurologos_homeostatico_cpu_ki.py:249`

### forward `def forward(self, x, loss_val)`
- Defined: `neurologos_homeostatico_cpu_ki.py:268`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `neurologos_homeostatico_cpu_ki.py:329`

### _create_grid_mask `def _create_grid_mask(self)`
- Defined: `neurologos_homeostatico_cpu_ki.py:344`

### get_adjacency `def get_adjacency(self, plasticity, loss_val)`
- Defined: `neurologos_homeostatico_cpu_ki.py:355`

### __init__ `def __init__(self, temperature)`
- Defined: `neurologos_homeostatico_cpu_ki.py:369`

### forward `def forward(self, features, labels)`
- Defined: `neurologos_homeostatico_cpu_ki.py:374`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_homeostatico_cpu_ki.py:398`

### _init_weights `def _init_weights(self)`
- Defined: `neurologos_homeostatico_cpu_ki.py:440`

### count_parameters `def count_parameters(self)`
- Defined: `neurologos_homeostatico_cpu_ki.py:445`

### forward `def forward(self, x, loss_val)`
- Defined: `neurologos_homeostatico_cpu_ki.py:448`

## neurologos_homestotico_cpu_qw.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_homestotico_cpu_qw.py:64`

### get_dataset `def get_dataset(config)`
- Defined: `neurologos_homestotico_cpu_qw.py:72`

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- Defined: `neurologos_homestotico_cpu_qw.py:365`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `neurologos_homestotico_cpu_qw.py:389`

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `neurologos_homestotico_cpu_qw.py:429`

### run_ablation_study `def run_ablation_study()`
- Defined: `neurologos_homestotico_cpu_qw.py:496`

### __init__ `def __init__(self, d_in)`
- Defined: `neurologos_homestotico_cpu_qw.py:93`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `neurologos_homestotico_cpu_qw.py:104`

### __init__ `def __init__(self, d_in, d_out, dynamic)`
- Defined: `neurologos_homestotico_cpu_qw.py:118`

### forward `def forward(self, x)`
- Defined: `neurologos_homestotico_cpu_qw.py:129`

### __init__ `def __init__(self, dim)`
- Defined: `neurologos_homestotico_cpu_qw.py:163`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_homestotico_cpu_qw.py:173`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `neurologos_homestotico_cpu_qw.py:188`

### forward `def forward(self, x)`
- Defined: `neurologos_homestotico_cpu_qw.py:196`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `neurologos_homestotico_cpu_qw.py:208`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `neurologos_homestotico_cpu_qw.py:222`

### __init__ `def __init__(self, temperature)`
- Defined: `neurologos_homestotico_cpu_qw.py:229`

### forward `def forward(self, features, labels)`
- Defined: `neurologos_homestotico_cpu_qw.py:234`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_homestotico_cpu_qw.py:253`

### _init_weights `def _init_weights(self)`
- Defined: `neurologos_homestotico_cpu_qw.py:301`

### count_parameters `def count_parameters(self)`
- Defined: `neurologos_homestotico_cpu_qw.py:306`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologos_homestotico_cpu_qw.py:309`

## neurologos_tricameral_exodia.py

### preprocess_and_cache_spectrograms `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)`
- Defined: `neurologos_tricameral_exodia.py:47`
- Doc: Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt

### setup_flickr8k_with_audio `def setup_flickr8k_with_audio(data_dir)`
- Defined: `neurologos_tricameral_exodia.py:120`
- Doc: Descarga y organiza Flickr8k + Audio del dataset de Kaggle.

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `neurologos_tricameral_exodia.py:308`
- Doc: Construye vocabulario desde el archivo de captions

### compute_alignment_loss `def compute_alignment_loss(visual_features, channels, alpha, epoch)`
- Defined: `neurologos_tricameral_exodia.py:2462`
- Doc: FIX: Pérdida auxiliar para alineación temprana de canales multimodales

### compute_tricameral_loss `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, channels, epoch, lambda_reward, lambda_mtp)`
- Defined: `neurologos_tricameral_exodia.py:2490`
- Doc: FIX: Pérdida con término explícito de coherencia multimodal

### train_tricameral `def train_tricameral()`
- Defined: `neurologos_tricameral_exodia.py:2563`

### __init__ `def __init__(self, working_capacity, short_term_capacity, importance_threshold)`
- Defined: `neurologos_tricameral_exodia.py:333`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `neurologos_tricameral_exodia.py:359`

### calculate_importance `def calculate_importance(self, episode, surprise_score)`
- Defined: `neurologos_tricameral_exodia.py:369`

### _calculate_novelty `def _calculate_novelty(self, episode)`
- Defined: `neurologos_tricameral_exodia.py:381`

### store_episode `def store_episode(self, image, audio, caption, surprise_score)`
- Defined: `neurologos_tricameral_exodia.py:402`

### _update_unified_buffer `def _update_unified_buffer(self)`
- Defined: `neurologos_tricameral_exodia.py:440`

### add `def add(self, image, audio, caption, surprise_score)`
- Defined: `neurologos_tricameral_exodia.py:452`

### apply_forgetting_curve `def apply_forgetting_curve(self)`
- Defined: `neurologos_tricameral_exodia.py:455`

### _purge_low_score_memories `def _purge_low_score_memories(self)`
- Defined: `neurologos_tricameral_exodia.py:471`

### sample `def sample(self, batch_size, memory_level)`
- Defined: `neurologos_tricameral_exodia.py:497`

### _sample_from_buffer `def _sample_from_buffer(self, buffer, scores, batch_size)`
- Defined: `neurologos_tricameral_exodia.py:527`

### get_total_size `def get_total_size(self)`
- Defined: `neurologos_tricameral_exodia.py:555`

### __init__ `def __init__(self)`
- Defined: `neurologos_tricameral_exodia.py:564`

### assess_reasoning_state `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)`
- Defined: `neurologos_tricameral_exodia.py:584`
- Doc: Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)

### assess_cognitive_state `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- Defined: `neurologos_tricameral_exodia.py:628`
- Doc: Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)

### apply_cognitive_intervention `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)`
- Defined: `neurologos_tricameral_exodia.py:674`
- Doc: Aplica intervenciones basadas en estado lingüístico y de razonamiento

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `neurologos_tricameral_exodia.py:764`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `neurologos_tricameral_exodia.py:798`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `neurologos_tricameral_exodia.py:807`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `neurologos_tricameral_exodia.py:820`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self, alpha, beta)`
- Defined: `neurologos_tricameral_exodia.py:835`

### _get_ngrams_cached `def _get_ngrams_cached(sentence, n)`
- Defined: `neurologos_tricameral_exodia.py:849`
- Doc: FIX: Método estático con lru_cache para n-gramas

### compute_linguistic_reward `def compute_linguistic_reward(self, references, hypotheses)`
- Defined: `neurologos_tricameral_exodia.py:858`

### compute_cider `def compute_cider(self, reference, hypothesis)`
- Defined: `neurologos_tricameral_exodia.py:897`
- Doc: FIX: Uso correcto del cache estático

### compute_spice `def compute_spice(self, reference, hypothesis)`
- Defined: `neurologos_tricameral_exodia.py:911`

### get_cache_stats `def get_cache_stats(self)`
- Defined: `neurologos_tricameral_exodia.py:923`
- Doc: FIX: Estadísticas de cache actualizadas

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `neurologos_tricameral_exodia.py:953`

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `neurologos_tricameral_exodia.py:976`

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `neurologos_tricameral_exodia.py:986`

### __init__ `def __init__(self, hidden_dim)`
- Defined: `neurologos_tricameral_exodia.py:995`

### reason_causally `def reason_causally(self, observation, context)`
- Defined: `neurologos_tricameral_exodia.py:1022`

### _predict_interventions `def _predict_interventions(self, hypothesis, confidence)`
- Defined: `neurologos_tricameral_exodia.py:1036`

### update_knowledge_graph `def update_knowledge_graph(self, cause, effect, strength)`
- Defined: `neurologos_tricameral_exodia.py:1053`

### query_causal_chain `def query_causal_chain(self, start_node, end_node)`
- Defined: `neurologos_tricameral_exodia.py:1059`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `neurologos_tricameral_exodia.py:1075`

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `neurologos_tricameral_exodia.py:1098`

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `neurologos_tricameral_exodia.py:1108`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `neurologos_tricameral_exodia.py:1121`

### forward `def forward(self, x)`
- Defined: `neurologos_tricameral_exodia.py:1163`

### _calculate_homeostasis_metric `def _calculate_homeostasis_metric(self, output)`
- Defined: `neurologos_tricameral_exodia.py:1179`
- Doc: Calcula métrica de homeostasis basada en la estabilidad del output

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `neurologos_tricameral_exodia.py:1188`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `neurologos_tricameral_exodia.py:1226`

### __init__ `def __init__(self)`
- Defined: `neurologos_tricameral_exodia.py:1260`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `neurologos_tricameral_exodia.py:1267`

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `neurologos_tricameral_exodia.py:1278`

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)`
- Defined: `neurologos_tricameral_exodia.py:1281`

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `neurologos_tricameral_exodia.py:1326`

### _reset_liquid_neuron `def _reset_liquid_neuron(self, liquid_neuron)`
- Defined: `neurologos_tricameral_exodia.py:1395`
- Doc: Reset completo de una neurona líquida

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `neurologos_tricameral_exodia.py:1411`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `neurologos_tricameral_exodia.py:1493`

### _apply_chain_of_thought `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- Defined: `neurologos_tricameral_exodia.py:1540`

### _greedy_decode `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- Defined: `neurologos_tricameral_exodia.py:1580`

### _apply_multi_token_prediction `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- Defined: `neurologos_tricameral_exodia.py:1641`

### _apply_structural_attention `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- Defined: `neurologos_tricameral_exodia.py:1683`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `neurologos_tricameral_exodia.py:1704`

### __init__ `def __init__(self, output_dim)`
- Defined: `neurologos_tricameral_exodia.py:1722`

### forward `def forward(self, mel_spec)`
- Defined: `neurologos_tricameral_exodia.py:1756`

### __init__ `def __init__(self, output_dim)`
- Defined: `neurologos_tricameral_exodia.py:1772`

### forward `def forward(self, image, audio)`
- Defined: `neurologos_tricameral_exodia.py:1812`
- Doc: Args:

### __init__ `def __init__(self, dim)`
- Defined: `neurologos_tricameral_exodia.py:1854`

### forward `def forward(self, right_features)`
- Defined: `neurologos_tricameral_exodia.py:1902`

### update_channel_fatigue `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- Defined: `neurologos_tricameral_exodia.py:1963`

### adjust_gates_by_fatigue `def adjust_gates_by_fatigue(self)`
- Defined: `neurologos_tricameral_exodia.py:1985`

### __init__ `def __init__(self)`
- Defined: `neurologos_tricameral_exodia.py:2007`

### _get_cached_norm `def _get_cached_norm(self, tensor, dim)`
- Defined: `neurologos_tricameral_exodia.py:2030`
- Doc: Cache de normalización con limpieza periódica

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context, channels)`
- Defined: `neurologos_tricameral_exodia.py:2048`
- Doc: FIX: Medición de coherencia multimodal real con atención a diversidad

### __init__ `def __init__(self, vocab_size)`
- Defined: `neurologos_tricameral_exodia.py:2314`

### forward `def forward(self, image, audio, captions, epoch)`
- Defined: `neurologos_tricameral_exodia.py:2320`

### __init__ `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache, cache_dir)`
- Defined: `neurologos_tricameral_exodia.py:2349`

### __len__ `def __len__(self)`
- Defined: `neurologos_tricameral_exodia.py:2406`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `neurologos_tricameral_exodia.py:2409`

### evaluate_reasoning_quality `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- Defined: `neurologos_tricameral_exodia.py:2101`

### calculate_synergy `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `neurologos_tricameral_exodia.py:2138`

### calculate_health `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `neurologos_tricameral_exodia.py:2149`

### update `def update(self)`
- Defined: `neurologos_tricameral_exodia.py:2158`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `neurologos_tricameral_exodia.py:2175`

### visualize_fatigue_distribution `def visualize_fatigue_distribution(self, epoch)`
- Defined: `neurologos_tricameral_exodia.py:2191`

### visualize_reasoning_metrics `def visualize_reasoning_metrics(self, epoch)`
- Defined: `neurologos_tricameral_exodia.py:2215`

### report `def report(self, epoch)`
- Defined: `neurologos_tricameral_exodia.py:2227`

## neurologos_v4.py

### train_logos `def train_logos(use_nested)`
- Defined: `neurologos_v4.py:358`

### __init__ `def __init__(self)`
- Defined: `neurologos_v4.py:15`

### forward `def forward(self, x)`
- Defined: `neurologos_v4.py:26`

### __init__ `def __init__(self, grid_size, output_dim)`
- Defined: `neurologos_v4.py:31`

### forward `def forward(self, x)`
- Defined: `neurologos_v4.py:57`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `neurologos_v4.py:82`

### forward `def forward(self, x, global_plasticity)`
- Defined: `neurologos_v4.py:103`

### consolidate_svd `def consolidate_svd(self, repair_strength)`
- Defined: `neurologos_v4.py:131`
- Doc: Consolidación espectral de pesos rápidos mediante SVD.

### __init__ `def __init__(self)`
- Defined: `neurologos_v4.py:165`

### forward `def forward(self, visual_features, plasticity)`
- Defined: `neurologos_v4.py:172`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `neurologos_v4.py:186`

### forward `def forward(self, thought, captions, max_len)`
- Defined: `neurologos_v4.py:201`

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `neurologos_v4.py:228`

### __init__ `def __init__(self, vocab_size, use_nested)`
- Defined: `neurologos_v4.py:237`

### forward `def forward(self, image, captions, plasticity)`
- Defined: `neurologos_v4.py:251`

### measure_richness `def measure_richness(self)`
- Defined: `neurologos_v4.py:264`

### __init__ `def __init__(self, total_epochs)`
- Defined: `neurologos_v4.py:272`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `neurologos_v4.py:276`

### __init__ `def __init__(self)`
- Defined: `neurologos_v4.py:292`

### __len__ `def __len__(self)`
- Defined: `neurologos_v4.py:344`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `neurologos_v4.py:346`

## neurologos_v5.py

### top_k_top_p_filtering `def top_k_top_p_filtering(logits, top_k, top_p, filter_value)`
- Defined: `neurologos_v5.py:11`
- Doc: Filtra logits con Top-K o Top-P (Nucleus) Sampling.

### measure_spatial_richness `def measure_spatial_richness(activations)`
- Defined: `neurologos_v5.py:32`
- Doc: Calcula la riqueza representacional de un tensor de activación.

### train_logos `def train_logos(use_nested)`
- Defined: `neurologos_v5.py:502`

### __init__ `def __init__(self, node_dim)`
- Defined: `neurologos_v5.py:76`

### forward `def forward(self, nodes, plasticity, transfer_rate)`
- Defined: `neurologos_v5.py:85`

### __init__ `def __init__(self)`
- Defined: `neurologos_v5.py:98`

### forward `def forward(self, x)`
- Defined: `neurologos_v5.py:113`

### __init__ `def __init__(self, grid_size, output_dim)`
- Defined: `neurologos_v5.py:120`

### forward `def forward(self, x)`
- Defined: `neurologos_v5.py:143`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `neurologos_v5.py:170`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `neurologos_v5.py:191`

### consolidate_svd `def consolidate_svd(self, repair_strength, timescale)`
- Defined: `neurologos_v5.py:228`

### __init__ `def __init__(self)`
- Defined: `neurologos_v5.py:261`

### forward `def forward(self, visual_features, plasticity, transfer_rate)`
- Defined: `neurologos_v5.py:278`

### get_liquid_module `def get_liquid_module(self)`
- Defined: `neurologos_v5.py:319`
- Doc: Retorna el LiquidNeuron activo para la consolidación externa.

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `neurologos_v5.py:334`

### forward `def forward(self, thought, captions, max_len)`
- Defined: `neurologos_v5.py:349`

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `neurologos_v5.py:395`

### __init__ `def __init__(self, vocab_size, use_nested)`
- Defined: `neurologos_v5.py:405`

### forward `def forward(self, image, captions, plasticity, transfer_rate)`
- Defined: `neurologos_v5.py:420`

### measure_richness `def measure_richness(self)`
- Defined: `neurologos_v5.py:428`

### __init__ `def __init__(self, total_epochs)`
- Defined: `neurologos_v5.py:436`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `neurologos_v5.py:440`

### __init__ `def __init__(self)`
- Defined: `neurologos_v5.py:458`

### __len__ `def __len__(self)`
- Defined: `neurologos_v5.py:483`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `neurologos_v5.py:486`

## neurologos_v6.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologos_v6.py:45`

### generate_ablation_matrix_4levels `def generate_ablation_matrix_4levels()`
- Defined: `neurologos_v6.py:603`

### run_ablation_study `def run_ablation_study()`
- Defined: `neurologos_v6.py:645`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_v6.py:56`

### _setup_logger `def _setup_logger(self)`
- Defined: `neurologos_v6.py:62`

### log_batch `def log_batch(self, step, metrics)`
- Defined: `neurologos_v6.py:69`

### save `def save(self, path)`
- Defined: `neurologos_v6.py:79`

### __init__ `def __init__(self)`
- Defined: `neurologos_v6.py:91`

### inject_concept_drift `def inject_concept_drift(self)`
- Defined: `neurologos_v6.py:100`

### get_batch `def get_batch(self, phase, bs, step)`
- Defined: `neurologos_v6.py:103`

### get_full `def get_full(self)`
- Defined: `neurologos_v6.py:118`

### get_w2 `def get_w2(self)`
- Defined: `neurologos_v6.py:121`

### __init__ `def __init__(self, input_dim, hidden_dim)`
- Defined: `neurologos_v6.py:128`

### forward `def forward(self, sequence)`
- Defined: `neurologos_v6.py:136`

### update `def update(self, loss_pred, loss_real)`
- Defined: `neurologos_v6.py:142`

### __init__ `def __init__(self, name, state_dim, cross_dim)`
- Defined: `neurologos_v6.py:155`

### forward `def forward(self)`
- Defined: `neurologos_v6.py:172`

### __init__ `def __init__(self, d_in)`
- Defined: `neurologos_v6.py:212`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `neurologos_v6.py:222`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_v6.py:238`

### forward `def forward(self, global_loss, step)`
- Defined: `neurologos_v6.py:251`

### get_component_health `def get_component_health(self)`
- Defined: `neurologos_v6.py:290`

### update_with_momentum `def update_with_momentum(self, current_lr, current_plasticity, meta_out, surprise_rate)`
- Defined: `neurologos_v6.py:298`

### __init__ `def __init__(self, d_in, d_out, config)`
- Defined: `neurologos_v6.py:321`

### forward `def forward(self, x, surprise_threshold)`
- Defined: `neurologos_v6.py:334`

### __init__ `def __init__(self, dim, atoms)`
- Defined: `neurologos_v6.py:377`

### forward `def forward(self, x, influence)`
- Defined: `neurologos_v6.py:385`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_v6.py:404`

### count_parameters `def count_parameters(self)`
- Defined: `neurologos_v6.py:427`

### forward `def forward(self, x, y, step)`
- Defined: `neurologos_v6.py:430`

### __init__ `def __init__(self, config)`
- Defined: `neurologos_v6.py:474`

### train `def train(self, model)`
- Defined: `neurologos_v6.py:479`

### evaluate `def evaluate(self, model)`
- Defined: `neurologos_v6.py:554`

## neurologosv5.2.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurologosv5.2.py:57`

### get_dataset `def get_dataset(config)`
- Defined: `neurologosv5.2.py:65`

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- Defined: `neurologosv5.2.py:340`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `neurologosv5.2.py:364`

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `neurologosv5.2.py:393`

### run_ablation_study `def run_ablation_study()`
- Defined: `neurologosv5.2.py:457`

### __init__ `def __init__(self, d_in)`
- Defined: `neurologosv5.2.py:86`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `neurologosv5.2.py:96`

### __init__ `def __init__(self, d_in, d_out, dynamic_mode)`
- Defined: `neurologosv5.2.py:110`

### forward `def forward(self, x)`
- Defined: `neurologosv5.2.py:121`

### __init__ `def __init__(self, dim)`
- Defined: `neurologosv5.2.py:152`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologosv5.2.py:162`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `neurologosv5.2.py:177`

### forward `def forward(self, x)`
- Defined: `neurologosv5.2.py:185`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `neurologosv5.2.py:197`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `neurologosv5.2.py:210`

### __init__ `def __init__(self, temperature)`
- Defined: `neurologosv5.2.py:217`

### forward `def forward(self, features, labels)`
- Defined: `neurologosv5.2.py:222`

### __init__ `def __init__(self, config)`
- Defined: `neurologosv5.2.py:241`

### _init_weights `def _init_weights(self)`
- Defined: `neurologosv5.2.py:278`

### count_parameters `def count_parameters(self)`
- Defined: `neurologosv5.2.py:283`

### forward `def forward(self, x, plasticity)`
- Defined: `neurologosv5.2.py:286`

## neurosoberano.py

### train_epoch `def train_epoch(model, loader, optimizer, criterion, device, use_mixup)`
- Defined: `neurosoberano.py:177`

### evaluate `def evaluate(model, loader, device)`
- Defined: `neurosoberano.py:215`

### main `def main()`
- Defined: `neurosoberano.py:435`

### __init__ `def __init__(self, in_c, out_c, stride)`
- Defined: `neurosoberano.py:40`

### forward `def forward(self, x)`
- Defined: `neurosoberano.py:54`

### __init__ `def __init__(self, num_classes)`
- Defined: `neurosoberano.py:65`

### _make_layer `def _make_layer(self, in_c, out_c, num_blocks, stride)`
- Defined: `neurosoberano.py:79`

### forward `def forward(self, x)`
- Defined: `neurosoberano.py:85`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `neurosoberano.py:98`

### forward `def forward(self, x, plasticity)`
- Defined: `neurosoberano.py:107`

### __init__ `def __init__(self, num_classes)`
- Defined: `neurosoberano.py:127`

### _make_layer `def _make_layer(self, in_c, out_c, num_blocks, stride)`
- Defined: `neurosoberano.py:146`

### forward `def forward(self, x)`
- Defined: `neurosoberano.py:152`

### update_plasticity `def update_plasticity(self, epoch, total_epochs)`
- Defined: `neurosoberano.py:163`
- Doc: Plasticity schedule simplificado

### __init__ `def __init__(self, config)`
- Defined: `neurosoberano.py:234`

### _get_data `def _get_data(self)`
- Defined: `neurosoberano.py:245`

### run_baseline `def run_baseline(self)`
- Defined: `neurosoberano.py:269`

### run_neurosovereign `def run_neurosovereign(self)`
- Defined: `neurosoberano.py:316`

### compare `def compare(self)`
- Defined: `neurosoberano.py:369`

### plot_comparison `def plot_comparison(self)`
- Defined: `neurosoberano.py:404`

## neurosoberano_bicameral_opt.py

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `neurosoberano_bicameral_opt.py:475`

### train_bicameral `def train_bicameral()`
- Defined: `neurosoberano_bicameral_opt.py:513`

### __init__ `def __init__(self)`
- Defined: `neurosoberano_bicameral_opt.py:40`

### forward `def forward(self, stress, excitation, fatigue, entropy, phase, loss_signal)`
- Defined: `neurosoberano_bicameral_opt.py:50`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `neurosoberano_bicameral_opt.py:67`

### forward `def forward(self, x, global_plasticity, transfer_rate, task_loss)`
- Defined: `neurosoberano_bicameral_opt.py:99`

### apply_svd_consolidation `def apply_svd_consolidation(self, repair_strength, timescale)`
- Defined: `neurosoberano_bicameral_opt.py:169`

### __init__ `def __init__(self, output_dim)`
- Defined: `neurosoberano_bicameral_opt.py:194`

### forward `def forward(self, image, plasticity, transfer_rate, task_loss)`
- Defined: `neurosoberano_bicameral_opt.py:205`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `neurosoberano_bicameral_opt.py:215`

### forward `def forward(self, visual_context, captions, max_len, return_gate)`
- Defined: `neurosoberano_bicameral_opt.py:237`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `neurosoberano_bicameral_opt.py:293`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `neurosoberano_bicameral_opt.py:298`

### __init__ `def __init__(self, dim)`
- Defined: `neurosoberano_bicameral_opt.py:314`

### forward `def forward(self, right_features, metabolism)`
- Defined: `neurosoberano_bicameral_opt.py:322`

### __init__ `def __init__(self, vocab_size)`
- Defined: `neurosoberano_bicameral_opt.py:332`

### forward `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- Defined: `neurosoberano_bicameral_opt.py:338`

### __init__ `def __init__(self)`
- Defined: `neurosoberano_bicameral_opt.py:362`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `neurosoberano_bicameral_opt.py:375`

### measure_vocab_diversity `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- Defined: `neurosoberano_bicameral_opt.py:382`

### update `def update(self)`
- Defined: `neurosoberano_bicameral_opt.py:386`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `neurosoberano_bicameral_opt.py:391`

### report `def report(self, epoch)`
- Defined: `neurosoberano_bicameral_opt.py:396`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `neurosoberano_bicameral_opt.py:437`

### __len__ `def __len__(self)`
- Defined: `neurosoberano_bicameral_opt.py:455`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `neurosoberano_bicameral_opt.py:458`

### __init__ `def __init__(self, total_epochs)`
- Defined: `neurosoberano_bicameral_opt.py:499`

### get_plasticity `def get_plasticity(self, epoch)`
- Defined: `neurosoberano_bicameral_opt.py:502`

## neurosovereign.py

### seed_everything `def seed_everything(seed)`
- Defined: `neurosovereign.py:44`

### mixup_data `def mixup_data(x, y, alpha)`
- Defined: `neurosovereign.py:51`
- Doc: Returns mixed inputs, pairs of targets, and lambda

### mixup_criterion `def mixup_criterion(criterion, pred, y_a, y_b, lam)`
- Defined: `neurosovereign.py:63`

### get_optimized_dataloaders `def get_optimized_dataloaders(config)`
- Defined: `neurosovereign.py:214`

### train_sovereign `def train_sovereign()`
- Defined: `neurosovereign.py:239`

### __init__ `def __init__(self, in_planes, out_planes, stride, dropRate)`
- Defined: `neurosovereign.py:70`

### forward `def forward(self, x)`
- Defined: `neurosovereign.py:85`

### __init__ `def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate)`
- Defined: `neurosovereign.py:97`

### _make_layer `def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate)`
- Defined: `neurosovereign.py:100`

### forward `def forward(self, x)`
- Defined: `neurosovereign.py:105`

### __init__ `def __init__(self, in_features, out_features, config)`
- Defined: `neurosovereign.py:116`

### forward `def forward(self, x)`
- Defined: `neurosovereign.py:133`

### __init__ `def __init__(self, config, depth, num_classes)`
- Defined: `neurosovereign.py:166`

### forward `def forward(self, x)`
- Defined: `neurosovereign.py:196`

## ohm.py

### train_omni_brain `def train_omni_brain(model, epochs, batch_size)`
- Defined: `ohm.py:863`
- Doc: Pipeline de entrenamiento para el Omni Brain

### update `def update(self, measurement, dt)`
- Defined: `ohm.py:47`
- Doc: Actualiza el estado del motor homeostático

### __init__ `def __init__(self)`
- Defined: `ohm.py:66`

### regulate_parameters `def regulate_parameters(self, current_coherence, energy_level)`
- Defined: `ohm.py:78`
- Doc: Regula parámetros para mantener PT-simetría

### __init__ `def __init__(self)`
- Defined: `ohm.py:102`

### regulate_connectivity `def regulate_connectivity(self, current_connectivity, clustering)`
- Defined: `ohm.py:112`
- Doc: Regula conectividad para mantener estructura óptima

### __init__ `def __init__(self)`
- Defined: `ohm.py:129`

### regulate_energy `def regulate_energy(self, memory_usage, cpu_usage, temperature)`
- Defined: `ohm.py:139`
- Doc: Regula parámetros para eficiencia energética

### __init__ `def __init__(self)`
- Defined: `ohm.py:159`

### regulate_consciousness `def regulate_consciousness(self, phi_effective, integration_level)`
- Defined: `ohm.py:168`
- Doc: Regula parámetros para control de conciencia

### __init__ `def __init__(self)`
- Defined: `ohm.py:187`

### regulate_dual_systems `def regulate_dual_systems(self, unconscious_activity, conscious_activity)`
- Defined: `ohm.py:197`
- Doc: Regula balance entre sistemas inconsciente y consciente

### __init__ `def __init__(self)`
- Defined: `ohm.py:215`

### regulate_learning `def regulate_learning(self, loss_reduction_rate, gradient_norm)`
- Defined: `ohm.py:224`
- Doc: Regula parámetros de aprendizaje

### __init__ `def __init__(self)`
- Defined: `ohm.py:243`

### regulate_modules `def regulate_modules(self, task_complexity, resource_availability, performance)`
- Defined: `ohm.py:259`
- Doc: Regula qué módulos están activos

### __init__ `def __init__(self)`
- Defined: `ohm.py:294`

### _initialize_motors `def _initialize_motors(self)`
- Defined: `ohm.py:300`
- Doc: Inicializa todos los motores homeostáticos

### sense_environment `def sense_environment(self)`
- Defined: `ohm.py:312`
- Doc: Sensa el estado actual del entorno

### measure_network_state `def measure_network_state(self, model, batch_data)`
- Defined: `ohm.py:326`
- Doc: Mide el estado actual de la red

### coordinate_all_motors `def coordinate_all_motors(self, environment_state, network_state)`
- Defined: `ohm.py:363`
- Doc: Coordina todos los motores homeostáticos

### __init__ `def __init__(self, module_name, enabled)`
- Defined: `ohm.py:441`

### forward `def forward(self, x, params)`
- Defined: `ohm.py:447`

### update_performance `def update_performance(self, metrics)`
- Defined: `ohm.py:450`

### __init__ `def __init__(self, in_features, out_features)`
- Defined: `ohm.py:456`

### forward `def forward(self, x, params)`
- Defined: `ohm.py:463`

### __init__ `def __init__(self, in_features, out_features, sparsity_factor)`
- Defined: `ohm.py:489`

### _generate_topology_mask `def _generate_topology_mask(self)`
- Defined: `ohm.py:504`
- Doc: Genera máscara topológica realista

### forward `def forward(self, x, params)`
- Defined: `ohm.py:525`

### __init__ `def __init__(self, features)`
- Defined: `ohm.py:546`

### forward `def forward(self, x, params)`
- Defined: `ohm.py:571`

### __init__ `def __init__(self, features)`
- Defined: `ohm.py:600`

### compute_phi_effective `def compute_phi_effective(self, x)`
- Defined: `ohm.py:614`
- Doc: Cálculo simplificado de Φₑ (integración efectiva)

### forward `def forward(self, x, params)`
- Defined: `ohm.py:634`

### __init__ `def __init__(self, target_performance)`
- Defined: `ohm.py:661`

### regulate_homeostasis `def regulate_homeostasis(self, observed_performance)`
- Defined: `ohm.py:666`
- Doc: Regula parámetros para homeostasis

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `ohm.py:692`

### initialize_context `def initialize_context(self)`
- Defined: `ohm.py:726`
- Doc: Inicializa el contexto del Omni Brain

### forward `def forward(self, x)`
- Defined: `ohm.py:740`
- Doc: Forward pass del Omni Brain con coordinación homeostática

### get_status_report `def get_status_report(self)`
- Defined: `ohm.py:828`
- Doc: Genera reporte de estado del Omni Brain

## omni1.py

### train_model `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)`
- Defined: `omni1.py:250`
- Doc: Entrenamiento optimizado.

### evaluate `def evaluate(model, loader, device, return_per_class)`
- Defined: `omni1.py:356`
- Doc: Evaluación estándar.

### get_cifar10_loaders `def get_cifar10_loaders(batch_size)`
- Defined: `omni1.py:399`
- Doc: Loaders con data augmentation.

### diagnose_model `def diagnose_model(model, loader, device)`
- Defined: `omni1.py:423`
- Doc: Diagnóstico profundo del modelo.

### main `def main()`
- Defined: `omni1.py:481`
- Doc: POC mejorado.

### __init__ `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- Defined: `omni1.py:33`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `omni1.py:46`

### update_fast_weights `def update_fast_weights(self, x)`
- Defined: `omni1.py:49`

### forward `def forward(self, x)`
- Defined: `omni1.py:68`

### get_fast_norm `def get_fast_norm(self)`
- Defined: `omni1.py:75`

### __init__ `def __init__(self, features, use_conscious)`
- Defined: `omni1.py:85`

### compute_phi_effective `def compute_phi_effective(self, activity)`
- Defined: `omni1.py:99`

### forward `def forward(self, x)`
- Defined: `omni1.py:121`

### __init__ `def __init__(self, use_fastslow, use_conscious)`
- Defined: `omni1.py:142`

### forward `def forward(self, x)`
- Defined: `omni1.py:188`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `omni1.py:198`

### get_fast_norms `def get_fast_norms(self)`
- Defined: `omni1.py:205`

### __init__ `def __init__(self, dim, use_fastslow)`
- Defined: `omni1.py:217`

### forward `def forward(self, x)`
- Defined: `omni1.py:235`

### get_activation `def get_activation(name)`
- Defined: `omni1.py:436`

### __init__ `def __init__(self, alpha, gamma)`
- Defined: `omni1.py:262`

### forward `def forward(self, inputs, targets)`
- Defined: `omni1.py:268`

### hook `def hook(model, input, output)`
- Defined: `omni1.py:437`

## omni3.py

### compute_integration_index `def compute_integration_index(activity)`
- Defined: `omni3.py:65`
- Doc: MÉTRICA HONESTA: Mide el ratio de varianza explicada por el primer componente.

### get_cifar10_loaders `def get_cifar10_loaders(config)`
- Defined: `omni3.py:340`

### evaluate_full `def evaluate_full(model, loader, device)`
- Defined: `omni3.py:368`
- Doc: Evaluación con múltiples métricas

### train `def train(config)`
- Defined: `omni3.py:403`

### run_ablation_study `def run_ablation_study()`
- Defined: `omni3.py:539`
- Doc: Ejecuta múltiples configuraciones para validar cada componente

### __init__ `def __init__(self, in_features, out_features, config)`
- Defined: `omni3.py:106`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `omni3.py:128`
- Doc: Reinicia la memoria a corto plazo (debe llamarse por época, no por batch)

### update_fast_weights `def update_fast_weights(self, x, slow_out)`
- Defined: `omni3.py:134`
- Doc: Actualización Hebbiana controlada.

### forward `def forward(self, x)`
- Defined: `omni3.py:169`

### get_fast_norm `def get_fast_norm(self)`
- Defined: `omni3.py:189`

### __init__ `def __init__(self, dim, config)`
- Defined: `omni3.py:197`

### forward `def forward(self, x)`
- Defined: `omni3.py:211`

### __init__ `def __init__(self, features, config)`
- Defined: `omni3.py:236`

### forward `def forward(self, x)`
- Defined: `omni3.py:248`

### __init__ `def __init__(self, config)`
- Defined: `omni3.py:269`

### forward `def forward(self, x)`
- Defined: `omni3.py:304`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `omni3.py:314`
- Doc: Reinicia todos los pesos rápidos del modelo. 

### get_fast_norms `def get_fast_norms(self)`
- Defined: `omni3.py:323`
- Doc: Recopila normas de fast weights de todos los módulos

### get_ablation_state `def get_ablation_state(self)`
- Defined: `omni3.py:327`
- Doc: Estado actual para logging

## omnibrain.py

### get_cifar10_loaders `def get_cifar10_loaders(batch_size)`
- Defined: `omnibrain.py:233`
- Doc: Loaders con data augmentation.

### get_few_shot_loaders `def get_few_shot_loaders(n_way, k_shot, batch_size)`
- Defined: `omnibrain.py:253`
- Doc: Few-shot learning setup: entrenar en clases limitadas.

### evaluate `def evaluate(model, loader, device, return_per_class)`
- Defined: `omnibrain.py:287`
- Doc: Evaluación con opción de métricas por clase.

### train_model `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)`
- Defined: `omnibrain.py:331`
- Doc: Entrenamiento con learning rate scheduler y early stopping.

### run_ablation_study `def run_ablation_study(epochs, batch_size)`
- Defined: `omnibrain.py:430`
- Doc: Ejecuta 4 configuraciones y compara resultados.

### run_few_shot_experiment `def run_few_shot_experiment(n_way, k_shot, epochs)`
- Defined: `omnibrain.py:475`
- Doc: Prueba capacidad de few-shot learning.

### analyze_phi_per_class `def analyze_phi_per_class()`
- Defined: `omnibrain.py:508`
- Doc: Analiza correlación entre Φₑ y dificultad de clase.

### plot_ablation_results `def plot_ablation_results(results)`
- Defined: `omnibrain.py:545`
- Doc: Genera gráficas comparativas de ablation study.

### plot_phi_analysis `def plot_phi_analysis(class_accs, avg_phi_per_class)`
- Defined: `omnibrain.py:617`
- Doc: Gráfica correlación Φₑ vs dificultad de clase.

### main `def main()`
- Defined: `omnibrain.py:673`
- Doc: Ejecuta el POC completo.

### __init__ `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- Defined: `omnibrain.py:33`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `omnibrain.py:49`

### update_fast_weights `def update_fast_weights(self, x)`
- Defined: `omnibrain.py:53`

### forward `def forward(self, x)`
- Defined: `omnibrain.py:73`

### end_of_batch `def end_of_batch(self)`
- Defined: `omnibrain.py:84`

### get_fast_norm `def get_fast_norm(self)`
- Defined: `omnibrain.py:87`

### __init__ `def __init__(self, dim, use_fastslow)`
- Defined: `omnibrain.py:93`

### forward `def forward(self, x)`
- Defined: `omnibrain.py:109`

### __init__ `def __init__(self, features, use_conscious)`
- Defined: `omnibrain.py:121`

### compute_phi_effective `def compute_phi_effective(self, activity)`
- Defined: `omnibrain.py:133`
- Doc: Φₑ basado en eigenvalues de covarianza.

### forward `def forward(self, x)`
- Defined: `omnibrain.py:156`

### __init__ `def __init__(self, use_fastslow, use_conscious)`
- Defined: `omnibrain.py:176`

### forward `def forward(self, x)`
- Defined: `omnibrain.py:209`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `omnibrain.py:216`

### get_fast_norms `def get_fast_norms(self)`
- Defined: `omnibrain.py:223`

## omnibrain_k.py

### compute_integration_index `def compute_integration_index(activity)`
- Defined: `omnibrain_k.py:65`
- Doc: MÉTRICA HONESTA: Mide el ratio de varianza explicada por el primer componente.

### get_cifar10_loaders `def get_cifar10_loaders(config)`
- Defined: `omnibrain_k.py:339`

### evaluate_full `def evaluate_full(model, loader, device)`
- Defined: `omnibrain_k.py:367`
- Doc: Evaluación con múltiples métricas

### train `def train(config)`
- Defined: `omnibrain_k.py:402`

### run_ablation_study `def run_ablation_study()`
- Defined: `omnibrain_k.py:537`
- Doc: Ejecuta múltiples configuraciones para validar cada componente

### __init__ `def __init__(self, in_features, out_features, config)`
- Defined: `omnibrain_k.py:106`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `omnibrain_k.py:128`
- Doc: Reinicia la memoria a corto plazo (debe llamarse por época, no por batch)

### update_fast_weights `def update_fast_weights(self, x, slow_out)`
- Defined: `omnibrain_k.py:134`
- Doc: Actualización Hebbiana controlada.

### forward `def forward(self, x)`
- Defined: `omnibrain_k.py:169`

### get_fast_norm `def get_fast_norm(self)`
- Defined: `omnibrain_k.py:189`

### __init__ `def __init__(self, dim, config)`
- Defined: `omnibrain_k.py:196`

### forward `def forward(self, x)`
- Defined: `omnibrain_k.py:210`

### __init__ `def __init__(self, features, config)`
- Defined: `omnibrain_k.py:235`

### forward `def forward(self, x)`
- Defined: `omnibrain_k.py:247`

### __init__ `def __init__(self, config)`
- Defined: `omnibrain_k.py:267`

### forward `def forward(self, x)`
- Defined: `omnibrain_k.py:302`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `omnibrain_k.py:312`
- Doc: Reinicia todos los pesos rápidos del modelo. Debe llamarse explícitamente

### get_fast_norms `def get_fast_norms(self)`
- Defined: `omnibrain_k.py:322`
- Doc: Recopila normas de fast weights de todos los módulos

### get_ablation_state `def get_ablation_state(self)`
- Defined: `omnibrain_k.py:326`
- Doc: Estado actual para logging

## omno1.bkp.py.py

### train_model `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)`
- Defined: `omno1.bkp.py.py:291`
- Doc: Entrenamiento optimizado.

### evaluate `def evaluate(model, loader, device, return_per_class)`
- Defined: `omno1.bkp.py.py:399`
- Doc: Evaluación estándar.

### get_cifar10_loaders `def get_cifar10_loaders(batch_size)`
- Defined: `omno1.bkp.py.py:442`
- Doc: Loaders con data augmentation.

### diagnose_model `def diagnose_model(model, loader, device)`
- Defined: `omno1.bkp.py.py:466`
- Doc: Diagnóstico profundo del modelo.

### main `def main()`
- Defined: `omno1.bkp.py.py:524`
- Doc: POC mejorado.

### __init__ `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- Defined: `omno1.bkp.py.py:34`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `omno1.bkp.py.py:51`

### update_fast_weights `def update_fast_weights(self, x)`
- Defined: `omno1.bkp.py.py:56`

### forward `def forward(self, x)`
- Defined: `omno1.bkp.py.py:85`

### get_fast_norm `def get_fast_norm(self)`
- Defined: `omno1.bkp.py.py:93`

### __init__ `def __init__(self, features, use_conscious)`
- Defined: `omno1.bkp.py.py:103`

### compute_phi_effective `def compute_phi_effective(self, activity)`
- Defined: `omno1.bkp.py.py:118`
- Doc: Φₑ mejorado con condiciones menos restrictivas.

### forward `def forward(self, x)`
- Defined: `omno1.bkp.py.py:159`

### __init__ `def __init__(self, use_fastslow, use_conscious)`
- Defined: `omno1.bkp.py.py:183`

### forward `def forward(self, x)`
- Defined: `omno1.bkp.py.py:229`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `omno1.bkp.py.py:239`

### get_fast_norms `def get_fast_norms(self)`
- Defined: `omno1.bkp.py.py:246`

### __init__ `def __init__(self, dim, use_fastslow)`
- Defined: `omno1.bkp.py.py:258`

### forward `def forward(self, x)`
- Defined: `omno1.bkp.py.py:276`

### get_activation `def get_activation(name)`
- Defined: `omno1.bkp.py.py:479`

### __init__ `def __init__(self, alpha, gamma)`
- Defined: `omno1.bkp.py.py:303`

### forward `def forward(self, inputs, targets)`
- Defined: `omno1.bkp.py.py:309`

### hook `def hook(model, input, output)`
- Defined: `omno1.bkp.py.py:480`

## physio_chimera_demo.py

### seed_everything `def seed_everything(seed)`
- Defined: `physio_chimera_demo.py:36`

### train_demo `def train_demo(config)`
- Defined: `physio_chimera_demo.py:231`

### run_demo `def run_demo()`
- Defined: `physio_chimera_demo.py:298`

### __init__ `def __init__(self)`
- Defined: `physio_chimera_demo.py:47`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `physio_chimera_demo.py:57`

### get_full `def get_full(self)`
- Defined: `physio_chimera_demo.py:71`
- Doc: Retorna el dataset completo

### get_w2 `def get_w2(self)`
- Defined: `physio_chimera_demo.py:75`
- Doc: Retorna solo los datos de WORLD_2 (dígitos >= 5)

### __init__ `def __init__(self)`
- Defined: `physio_chimera_demo.py:83`

### update `def update(self, loss, physio)`
- Defined: `physio_chimera_demo.py:88`

### report `def report(self, step, phase)`
- Defined: `physio_chimera_demo.py:94`

### __init__ `def __init__(self, levels, d_model, hidden_dim)`
- Defined: `physio_chimera_demo.py:131`

### forward `def forward(self, x, global_step)`
- Defined: `physio_chimera_demo.py:142`

### __init__ `def __init__(self, d_in, d_out, config)`
- Defined: `physio_chimera_demo.py:154`

### forward `def forward(self, x, global_step)`
- Defined: `physio_chimera_demo.py:162`

### __init__ `def __init__(self, config)`
- Defined: `physio_chimera_demo.py:195`

### forward `def forward(self, x, global_step)`
- Defined: `physio_chimera_demo.py:207`

## physio_chimera_v15_monitored.py

### seed_everything `def seed_everything(seed)`
- Defined: `physio_chimera_v15_monitored.py:58`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### train_nested_monitored `def train_nested_monitored(config)`
- Defined: `physio_chimera_v15_monitored.py:658`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### run_experiment_monitored `def run_experiment_monitored()`
- Defined: `physio_chimera_v15_monitored.py:762`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### __init__ `def __init__(self)`
- Defined: `physio_chimera_v15_monitored.py:69`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `physio_chimera_v15_monitored.py:79`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### get_full `def get_full(self)`
- Defined: `physio_chimera_v15_monitored.py:93`
- Doc: Retorna el dataset completo
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### get_w2 `def get_w2(self)`
- Defined: `physio_chimera_v15_monitored.py:97`
- Doc: Retorna solo los datos de WORLD_2 (dígitos >= 5)
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### __init__ `def __init__(self, config)`
- Defined: `physio_chimera_v15_monitored.py:107`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### update_physio_metrics `def update_physio_metrics(self, metabolism, sensitivity, gate)`
- Defined: `physio_chimera_v15_monitored.py:144`
- Doc: Actualiza métricas fisiológicas
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### update_performance_metrics `def update_performance_metrics(self, loss, accuracy, lr)`
- Defined: `physio_chimera_v15_monitored.py:150`
- Doc: Actualiza métricas de rendimiento
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### update_memory_metrics `def update_memory_metrics(self, cms_activations, hebbian_norm, forgetting_factor)`
- Defined: `physio_chimera_v15_monitored.py:158`
- Doc: Actualiza métricas de memoria
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### calculate_health_metrics `def calculate_health_metrics(self)`
- Defined: `physio_chimera_v15_monitored.py:166`
- Doc: Calcula métricas de salud del sistema
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### get_recent_avg `def get_recent_avg(self, category, key, n)`
- Defined: `physio_chimera_v15_monitored.py:191`
- Doc: Obtiene promedio reciente de una métrica
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### generate_diagnostic_report `def generate_diagnostic_report(self, step, phase)`
- Defined: `physio_chimera_v15_monitored.py:209`
- Doc: Genera reporte de diagnóstico
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### save_metrics `def save_metrics(self, filepath)`
- Defined: `physio_chimera_v15_monitored.py:279`
- Doc: Guarda todas las métricas
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### __init__ `def __init__(self, input_dim, hidden_dim)`
- Defined: `physio_chimera_v15_monitored.py:299`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### forward `def forward(self, x)`
- Defined: `physio_chimera_v15_monitored.py:306`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### __init__ `def __init__(self, levels, d_model, hidden_dim)`
- Defined: `physio_chimera_v15_monitored.py:320`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### forward `def forward(self, x, global_step)`
- Defined: `physio_chimera_v15_monitored.py:332`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### __init__ `def __init__(self, d_in, d_out, config)`
- Defined: `physio_chimera_v15_monitored.py:347`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### forward `def forward(self, x, global_step)`
- Defined: `physio_chimera_v15_monitored.py:362`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### __init__ `def __init__(self, config)`
- Defined: `physio_chimera_v15_monitored.py:398`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### forward `def forward(self, x, global_step)`
- Defined: `physio_chimera_v15_monitored.py:410`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### __init__ `def __init__(self, save_dir)`
- Defined: `physio_chimera_v15_monitored.py:455`
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### plot_training_curves `def plot_training_curves(self, diagnostics)`
- Defined: `physio_chimera_v15_monitored.py:463`
- Doc: Genera gráficos de curvas de entrenamiento
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### create_final_report `def create_final_report(self, final_metrics, diagnostics)`
- Defined: `physio_chimera_v15_monitored.py:551`
- Doc: Crea reporte final con todas las métricas
- Imported by: `example_usage.py`, `run_complete_experiment.py`

### _generate_recommendations `def _generate_recommendations(self, diagnostics)`
- Defined: `physio_chimera_v15_monitored.py:634`
- Doc: Genera recomendaciones basadas en el diagnóstico
- Imported by: `example_usage.py`, `run_complete_experiment.py`

## physioneruon_simple.py

### seed_everything `def seed_everything(seed)`
- Defined: `physioneruon_simple.py:54`

### get_dataset `def get_dataset(config)`
- Defined: `physioneruon_simple.py:61`
- Doc: Dataset balanceado con más separabilidad

### pgd_attack `def pgd_attack(model, x, y, eps, steps, step_size)`
- Defined: `physioneruon_simple.py:117`
- Doc: PGD estándar bien implementado

### train_simple_robust `def train_simple_robust(config, dataset, verbose)`
- Defined: `physioneruon_simple.py:154`
- Doc: Entrenamiento con adversarial training progresivo

### main `def main()`
- Defined: `physioneruon_simple.py:281`

### __init__ `def __init__(self, config)`
- Defined: `physioneruon_simple.py:85`

### forward `def forward(self, x)`
- Defined: `physioneruon_simple.py:105`

## physioneuron_cpu_v1.py

### seed_everything `def seed_everything(seed)`
- Defined: `physioneuron_cpu_v1.py:57`

### get_dataset `def get_dataset(config)`
- Defined: `physioneuron_cpu_v1.py:65`

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- Defined: `physioneuron_cpu_v1.py:340`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `physioneuron_cpu_v1.py:364`

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `physioneuron_cpu_v1.py:393`

### run_ablation_study `def run_ablation_study()`
- Defined: `physioneuron_cpu_v1.py:457`

### __init__ `def __init__(self, d_in)`
- Defined: `physioneuron_cpu_v1.py:86`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `physioneuron_cpu_v1.py:96`

### __init__ `def __init__(self, d_in, d_out, dynamic_mode)`
- Defined: `physioneuron_cpu_v1.py:110`

### forward `def forward(self, x)`
- Defined: `physioneuron_cpu_v1.py:121`

### __init__ `def __init__(self, dim)`
- Defined: `physioneuron_cpu_v1.py:152`

### forward `def forward(self, x, plasticity)`
- Defined: `physioneuron_cpu_v1.py:162`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `physioneuron_cpu_v1.py:177`

### forward `def forward(self, x)`
- Defined: `physioneuron_cpu_v1.py:185`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `physioneuron_cpu_v1.py:197`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `physioneuron_cpu_v1.py:210`

### __init__ `def __init__(self, temperature)`
- Defined: `physioneuron_cpu_v1.py:217`

### forward `def forward(self, features, labels)`
- Defined: `physioneuron_cpu_v1.py:222`

### __init__ `def __init__(self, config)`
- Defined: `physioneuron_cpu_v1.py:241`

### _init_weights `def _init_weights(self)`
- Defined: `physioneuron_cpu_v1.py:278`

### count_parameters `def count_parameters(self)`
- Defined: `physioneuron_cpu_v1.py:283`

### forward `def forward(self, x, plasticity)`
- Defined: `physioneuron_cpu_v1.py:286`

## physioneuron_cpu_v2.py

### seed_everything `def seed_everything(seed)`
- Defined: `physioneuron_cpu_v2.py:64`

### get_dataset `def get_dataset(config)`
- Defined: `physioneuron_cpu_v2.py:72`

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- Defined: `physioneuron_cpu_v2.py:347`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `physioneuron_cpu_v2.py:371`

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `physioneuron_cpu_v2.py:400`

### run_ablation_study `def run_ablation_study()`
- Defined: `physioneuron_cpu_v2.py:464`

### __init__ `def __init__(self, d_in)`
- Defined: `physioneuron_cpu_v2.py:93`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `physioneuron_cpu_v2.py:103`

### __init__ `def __init__(self, d_in, d_out, dynamic_mode)`
- Defined: `physioneuron_cpu_v2.py:117`

### forward `def forward(self, x)`
- Defined: `physioneuron_cpu_v2.py:128`

### __init__ `def __init__(self, dim)`
- Defined: `physioneuron_cpu_v2.py:159`

### forward `def forward(self, x, plasticity)`
- Defined: `physioneuron_cpu_v2.py:169`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `physioneuron_cpu_v2.py:184`

### forward `def forward(self, x)`
- Defined: `physioneuron_cpu_v2.py:192`

### __init__ `def __init__(self, num_nodes, config)`
- Defined: `physioneuron_cpu_v2.py:204`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `physioneuron_cpu_v2.py:217`

### __init__ `def __init__(self, temperature)`
- Defined: `physioneuron_cpu_v2.py:224`

### forward `def forward(self, features, labels)`
- Defined: `physioneuron_cpu_v2.py:229`

### __init__ `def __init__(self, config)`
- Defined: `physioneuron_cpu_v2.py:248`

### _init_weights `def _init_weights(self)`
- Defined: `physioneuron_cpu_v2.py:285`

### count_parameters `def count_parameters(self)`
- Defined: `physioneuron_cpu_v2.py:290`

### forward `def forward(self, x, plasticity)`
- Defined: `physioneuron_cpu_v2.py:293`

## physioneuron_cpu_v3.py

### seed_everything `def seed_everything(seed)`
- Defined: `physioneuron_cpu_v3.py:71`

### get_elite_dataset `def get_elite_dataset(config)`
- Defined: `physioneuron_cpu_v3.py:79`
- Doc: Dataset más grande y balanceado con separabilidad controlada

### elite_pgd_attack `def elite_pgd_attack(model, x, y, eps, steps, stress)`
- Defined: `physioneuron_cpu_v3.py:348`
- Doc: PGD con reinicio aleatorio

### train_elite_model `def train_elite_model(config, dataset, fold_results)`
- Defined: `physioneuron_cpu_v3.py:430`
- Doc: Entrenamiento con curriculum adversarial

### run_elite_experiment `def run_elite_experiment()`
- Defined: `physioneuron_cpu_v3.py:542`

### __init__ `def __init__(self, dim, capacity)`
- Defined: `physioneuron_cpu_v3.py:104`

### update `def update(self, x, y)`
- Defined: `physioneuron_cpu_v3.py:112`
- Doc: Almacena ejemplos duros

### retrieve `def retrieve(self, x, k)`
- Defined: `physioneuron_cpu_v3.py:125`
- Doc: Recupera k vecinos más cercanos

### __init__ `def __init__(self, in_features, out_features)`
- Defined: `physioneuron_cpu_v3.py:138`

### power_iteration `def power_iteration(self, n_iter)`
- Defined: `physioneuron_cpu_v3.py:145`
- Doc: Aproxima la norma espectral máxima

### forward `def forward(self, x)`
- Defined: `physioneuron_cpu_v3.py:152`

### __init__ `def __init__(self, d_in, d_out, use_spectral)`
- Defined: `physioneuron_cpu_v3.py:164`

### forward `def forward(self, x)`
- Defined: `physioneuron_cpu_v3.py:194`

### __init__ `def __init__(self, num_nodes, grid_size)`
- Defined: `physioneuron_cpu_v3.py:224`

### forward `def forward(self, stress)`
- Defined: `physioneuron_cpu_v3.py:248`
- Doc: stress ∈ [0,1]: cuánto estrés adversarial

### __init__ `def __init__(self, config)`
- Defined: `physioneuron_cpu_v3.py:261`

### count_parameters `def count_parameters(self)`
- Defined: `physioneuron_cpu_v3.py:303`

### forward `def forward(self, x, stress)`
- Defined: `physioneuron_cpu_v3.py:306`

### __init__ `def __init__(self, temperature)`
- Defined: `physioneuron_cpu_v3.py:386`

### forward `def forward(self, features, labels)`
- Defined: `physioneuron_cpu_v3.py:390`

## poke_cifar.py

### compute_phi_effective `def compute_phi_effective(activity)`
- Defined: `poke_cifar.py:35`

### get_cifar10_loaders `def get_cifar10_loaders(batch_size)`
- Defined: `poke_cifar.py:180`

### evaluate `def evaluate(model, loader, device)`
- Defined: `poke_cifar.py:200`

### main `def main()`
- Defined: `poke_cifar.py:221`

### plot_history `def plot_history(hist)`
- Defined: `poke_cifar.py:287`

### demo_inference `def demo_inference(model, loader)`
- Defined: `poke_cifar.py:300`

### __init__ `def __init__(self, in_features, out_features)`
- Defined: `poke_cifar.py:57`

### forward `def forward(self, x)`
- Defined: `poke_cifar.py:66`

### __init__ `def __init__(self, in_f, out_f, density)`
- Defined: `poke_cifar.py:78`

### _update_mask `def _update_mask(self)`
- Defined: `poke_cifar.py:87`

### forward `def forward(self, x)`
- Defined: `poke_cifar.py:93`

### __init__ `def __init__(self, features)`
- Defined: `poke_cifar.py:99`

### forward `def forward(self, x)`
- Defined: `poke_cifar.py:107`

### __init__ `def __init__(self, features)`
- Defined: `poke_cifar.py:117`

### forward `def forward(self, x)`
- Defined: `poke_cifar.py:123`

### __init__ `def __init__(self)`
- Defined: `poke_cifar.py:134`

### forward `def forward(self, x)`
- Defined: `poke_cifar.py:161`

## poke_cifar2.py

### compute_phi_effective `def compute_phi_effective(activity)`
- Defined: `poke_cifar2.py:28`

### get_cifar10_loaders `def get_cifar10_loaders(batch_size)`
- Defined: `poke_cifar2.py:175`

### evaluate `def evaluate(model, loader, device)`
- Defined: `poke_cifar2.py:191`

### train `def train()`
- Defined: `poke_cifar2.py:212`

### __init__ `def __init__(self, in_features, out_features, fast_lr)`
- Defined: `poke_cifar2.py:50`

### reset_fast_weights `def reset_fast_weights(self)`
- Defined: `poke_cifar2.py:65`

### update_fast_weights `def update_fast_weights(self, x)`
- Defined: `poke_cifar2.py:69`

### forward `def forward(self, x)`
- Defined: `poke_cifar2.py:76`

### end_of_batch `def end_of_batch(self)`
- Defined: `poke_cifar2.py:89`

### get_fast_weight_norm `def get_fast_weight_norm(self)`
- Defined: `poke_cifar2.py:92`

### __init__ `def __init__(self, dim)`
- Defined: `poke_cifar2.py:100`

### forward `def forward(self, x)`
- Defined: `poke_cifar2.py:108`

### __init__ `def __init__(self, dim)`
- Defined: `poke_cifar2.py:118`

### forward `def forward(self, x)`
- Defined: `poke_cifar2.py:124`

### __init__ `def __init__(self)`
- Defined: `poke_cifar2.py:132`

### forward `def forward(self, x)`
- Defined: `poke_cifar2.py:151`

### reset_all_fast_weights `def reset_all_fast_weights(self)`
- Defined: `poke_cifar2.py:159`

### get_fast_norms `def get_fast_norms(self)`
- Defined: `poke_cifar2.py:164`

## pokemon3.py

### compute_phi_effective_approx `def compute_phi_effective_approx(activity)`
- Defined: `pokemon3.py:30`
- Doc: Cálculo estable de Φₑ compatible con todas las versiones

### estimate_energy_consumption `def estimate_energy_consumption(model, batch_size)`
- Defined: `pokemon3.py:63`
- Doc: Estimación conservadora de energía

### prepare_mnist_data `def prepare_mnist_data(batch_size, device)`
- Defined: `pokemon3.py:245`
- Doc: Preparar datos MNIST con protección para entornos limitados

### train_omni_brain `def train_omni_brain(model, train_loader, test_loader, epochs, device)`
- Defined: `pokemon3.py:269`
- Doc: Entrenamiento compatible con todas las versiones de PyTorch

### evaluate_model `def evaluate_model(model, test_loader, device, criterion)`
- Defined: `pokemon3.py:380`
- Doc: Evaluación compatible con todas las versiones

### generate_evolution_plots `def generate_evolution_plots(history, epochs)`
- Defined: `pokemon3.py:402`
- Doc: Generar gráficos con protección para entornos sin GUI

### demonstrate_inference `def demonstrate_inference(model, test_loader, device)`
- Defined: `pokemon3.py:435`
- Doc: Demostración compatible con todas las versiones

### final_report `def final_report(model, history)`
- Defined: `pokemon3.py:467`
- Doc: Reporte final compatible

### __init__ `def __init__(self, in_features, out_features)`
- Defined: `pokemon3.py:85`

### compute_pt_phase `def compute_pt_phase(self)`
- Defined: `pokemon3.py:94`

### forward `def forward(self, x)`
- Defined: `pokemon3.py:102`

### __init__ `def __init__(self, in_features, out_features, connectivity)`
- Defined: `pokemon3.py:112`

### update_topology `def update_topology(self, connectivity)`
- Defined: `pokemon3.py:121`

### forward `def forward(self, x)`
- Defined: `pokemon3.py:128`

### __init__ `def __init__(self, features)`
- Defined: `pokemon3.py:137`

### forward `def forward(self, x)`
- Defined: `pokemon3.py:153`

### __init__ `def __init__(self, features)`
- Defined: `pokemon3.py:167`

### forward `def forward(self, x)`
- Defined: `pokemon3.py:177`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `pokemon3.py:190`

### forward `def forward(self, x)`
- Defined: `pokemon3.py:217`

### update_topology `def update_topology(self, current_connectivity)`
- Defined: `pokemon3.py:235`

## pokemon4.py

### compute_phi_effective `def compute_phi_effective(activity)`
- Defined: `pokemon4.py:41`
- Doc: Φₑ realista: fracción de varianza explicada por el primer componente PCA.

### get_mnist_loaders `def get_mnist_loaders(batch_size)`
- Defined: `pokemon4.py:206`

### evaluate `def evaluate(model, loader, device)`
- Defined: `pokemon4.py:218`

### train_and_evaluate `def train_and_evaluate()`
- Defined: `pokemon4.py:236`

### plot_history `def plot_history(hist)`
- Defined: `pokemon4.py:302`

### __init__ `def __init__(self, in_features, out_features)`
- Defined: `pokemon4.py:69`

### forward `def forward(self, x)`
- Defined: `pokemon4.py:78`

### __init__ `def __init__(self, in_features, out_features, target_density)`
- Defined: `pokemon4.py:92`

### _update_mask `def _update_mask(self)`
- Defined: `pokemon4.py:101`

### forward `def forward(self, x)`
- Defined: `pokemon4.py:107`

### __init__ `def __init__(self, features)`
- Defined: `pokemon4.py:114`

### forward `def forward(self, x)`
- Defined: `pokemon4.py:130`

### __init__ `def __init__(self, features)`
- Defined: `pokemon4.py:147`

### forward `def forward(self, x)`
- Defined: `pokemon4.py:157`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `pokemon4.py:169`

### forward `def forward(self, x)`
- Defined: `pokemon4.py:186`

## pokemon_battle_champion.py

### create_battle_dataset `def create_battle_dataset(config)`
- Defined: `pokemon_battle_champion.py:128`
- Doc: Crear dataset para la batalla

### battle_training_epoch `def battle_training_epoch(model, loader, optimizer, criterion, epoch)`
- Defined: `pokemon_battle_champion.py:161`
- Doc: Entrenamiento de una época de batalla

### evaluate_battle_champion `def evaluate_battle_champion(model, loader)`
- Defined: `pokemon_battle_champion.py:192`
- Doc: Evaluar el campeón en batalla

### run_epic_pokemon_battle `def run_epic_pokemon_battle()`
- Defined: `pokemon_battle_champion.py:208`
- Doc: ¡EJECUTAR LA BATALLA ÉPICA!

### create_epic_battle_visualization `def create_epic_battle_visualization(battle_history, historical_results)`
- Defined: `pokemon_battle_champion.py:334`
- Doc: Crear visualización épica de la batalla

### save_battle_results `def save_battle_results(battle_history, historical_results, champion_model)`
- Defined: `pokemon_battle_champion.py:440`
- Doc: Guardar resultados de la batalla épica

### __init__ `def __init__(self, config)`
- Defined: `pokemon_battle_champion.py:44`

### forward `def forward(self, x)`
- Defined: `pokemon_battle_champion.py:99`

## pokemon_hybrid_synergy_ablation.py

### run_synergy_ablation `def run_synergy_ablation()`
- Defined: `pokemon_hybrid_synergy_ablation.py:495`
- Doc: Ejecutar estudio de ablación completo

### analyze_synergy_results `def analyze_synergy_results(results)`
- Defined: `pokemon_hybrid_synergy_ablation.py:637`
- Doc: Analizar resultados del estudio de sinergias

### create_synergy_visualizations `def create_synergy_visualizations(results, output_dir)`
- Defined: `pokemon_hybrid_synergy_ablation.py:696`
- Doc: Crear visualizaciones del estudio de sinergias

### to_dict `def to_dict(self)`
- Defined: `pokemon_hybrid_synergy_ablation.py:75`

### __init__ `def __init__(self, input_dim, hidden_dim, latent_dim)`
- Defined: `pokemon_hybrid_synergy_ablation.py:84`

### reparameterize `def reparameterize(self, mu, logvar)`
- Defined: `pokemon_hybrid_synergy_ablation.py:111`

### forward `def forward(self, x, return_encoding)`
- Defined: `pokemon_hybrid_synergy_ablation.py:116`

### __init__ `def __init__(self, d_model, num_heads, d_ff)`
- Defined: `pokemon_hybrid_synergy_ablation.py:127`

### forward `def forward(self, x)`
- Defined: `pokemon_hybrid_synergy_ablation.py:149`

### __init__ `def __init__(self, input_dim, latent_dim, hidden_dim)`
- Defined: `pokemon_hybrid_synergy_ablation.py:159`

### generate `def generate(self, z)`
- Defined: `pokemon_hybrid_synergy_ablation.py:184`

### discriminate `def discriminate(self, x)`
- Defined: `pokemon_hybrid_synergy_ablation.py:187`

### __init__ `def __init__(self, grid_size, embed_dim, sparsity)`
- Defined: `pokemon_hybrid_synergy_ablation.py:192`

### get_adjacency_matrix `def get_adjacency_matrix(self)`
- Defined: `pokemon_hybrid_synergy_ablation.py:220`

### forward `def forward(self, x)`
- Defined: `pokemon_hybrid_synergy_ablation.py:234`

### __init__ `def __init__(self, config)`
- Defined: `pokemon_hybrid_synergy_ablation.py:273`

### forward `def forward(self, x, return_all)`
- Defined: `pokemon_hybrid_synergy_ablation.py:327`

### __init__ `def __init__(self, config)`
- Defined: `pokemon_hybrid_synergy_ablation.py:389`

### get_ablation_matrix `def get_ablation_matrix(self)`
- Defined: `pokemon_hybrid_synergy_ablation.py:393`
- Doc: Matriz de ablación de 4 niveles:

### create_variant_model `def create_variant_model(self, level_name)`
- Defined: `pokemon_hybrid_synergy_ablation.py:424`
- Doc: Crear modelo variante para un nivel específico

### __init__ `def __init__(self, config)`
- Defined: `pokemon_hybrid_synergy_ablation.py:429`

### forward `def forward(self, x)`
- Defined: `pokemon_hybrid_synergy_ablation.py:434`

### __init__ `def __init__(self, config)`
- Defined: `pokemon_hybrid_synergy_ablation.py:444`

### forward `def forward(self, x)`
- Defined: `pokemon_hybrid_synergy_ablation.py:450`

### __init__ `def __init__(self, config)`
- Defined: `pokemon_hybrid_synergy_ablation.py:464`

### forward `def forward(self, x)`
- Defined: `pokemon_hybrid_synergy_ablation.py:471`

## premium_synergy_demo.py

### run_demo `def run_demo()`
- Defined: `premium_synergy_demo.py:302`
- Doc: Ejecuta demostración del sistema Premium Synergy

### __init__ `def __init__(self)`
- Defined: `premium_synergy_demo.py:39`

### process `def process(self, input_data, plasticity)`
- Defined: `premium_synergy_demo.py:48`
- Doc: Procesamiento con autoregulación interna

### __init__ `def __init__(self)`
- Defined: `premium_synergy_demo.py:78`

### process `def process(self, input_data, chaos_level)`
- Defined: `premium_synergy_demo.py:87`
- Doc: Procesamiento con control integrativo

### __init__ `def __init__(self)`
- Defined: `premium_synergy_demo.py:119`

### process `def process(self, input_data, plasticity, chaos)`
- Defined: `premium_synergy_demo.py:128`
- Doc: Procesamiento con regulación de fases

### __init__ `def __init__(self, threshold, convergence_epochs)`
- Defined: `premium_synergy_demo.py:163`

### deliberate `def deliberate(self, components, target_accuracy)`
- Defined: `premium_synergy_demo.py:171`
- Doc: Proceso de deliberación democrática

### __init__ `def __init__(self)`
- Defined: `premium_synergy_demo.py:228`

### process_epoch `def process_epoch(self, input_data, chaos_level)`
- Defined: `premium_synergy_demo.py:241`
- Doc: Procesa una época del sistema democrático

### calculate_target_accuracy `def calculate_target_accuracy(self)`
- Defined: `premium_synergy_demo.py:296`
- Doc: Calcula accuracy objetivo basada en sinergia actual

## premium_synergy_democratic.py

### ensure_dependencies `def ensure_dependencies()`
- Defined: `premium_synergy_democratic.py:1108`
- Doc: Asegura que las dependencias estén instaladas
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### create_synthetic_dataset `def create_synthetic_dataset(config)`
- Defined: `premium_synergy_democratic.py:1118`
- Doc: Crea dataset sintético para testing
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### train_premium_synergy `def train_premium_synergy(config)`
- Defined: `premium_synergy_democratic.py:1148`
- Doc: Entrena el modelo Premium Synergy
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### create_dataloader `def create_dataloader(X, y, batch_size, shuffle)`
- Defined: `premium_synergy_democratic.py:1272`
- Doc: Crea dataloader
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### main `def main()`
- Defined: `premium_synergy_democratic.py:1284`
- Doc: Función principal
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, max_memory_gb)`
- Defined: `premium_synergy_democratic.py:97`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### check_memory `def check_memory(self)`
- Defined: `premium_synergy_democratic.py:101`
- Doc: Verifica el uso de memoria actual
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### warn_if_high `def warn_if_high(self)`
- Defined: `premium_synergy_democratic.py:126`
- Doc: Advierte si el uso de memoria es alto
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, config)`
- Defined: `premium_synergy_democratic.py:142`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x, plasticity)`
- Defined: `premium_synergy_democratic.py:175`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### internal_dialogue `def internal_dialogue(self)`
- Defined: `premium_synergy_democratic.py:222`
- Doc: Diálogo interno fisiológico - metabolimo, sensibilidad, gating
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, config)`
- Defined: `premium_synergy_democratic.py:238`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x, chaos_level)`
- Defined: `premium_synergy_democratic.py:268`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### internal_dialogue `def internal_dialogue(self)`
- Defined: `premium_synergy_democratic.py:312`
- Doc: Diálogo interno - balance integrativo y modulación caótica
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, config)`
- Defined: `premium_synergy_democratic.py:328`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x, plasticity, chaos)`
- Defined: `premium_synergy_democratic.py:358`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### internal_dialogue `def internal_dialogue(self)`
- Defined: `premium_synergy_democratic.py:400`
- Doc: Diálogo interno - regulación de fases y control atencional
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### consolidate `def consolidate(self)`
- Defined: `premium_synergy_democratic.py:409`
- Doc: SVD consolidation de liquid neurons
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:421`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x)`
- Defined: `premium_synergy_democratic.py:431`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_state `def get_state(self)`
- Defined: `premium_synergy_democratic.py:456`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:461`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x)`
- Defined: `premium_synergy_democratic.py:471`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_level `def get_level(self)`
- Defined: `premium_synergy_democratic.py:489`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, num_nodes, grid_size)`
- Defined: `premium_synergy_democratic.py:494`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### _create_grid_mask `def _create_grid_mask(self)`
- Defined: `premium_synergy_democratic.py:501`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `premium_synergy_democratic.py:514`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `premium_synergy_democratic.py:521`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x)`
- Defined: `premium_synergy_democratic.py:531`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:542`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x)`
- Defined: `premium_synergy_democratic.py:552`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_level `def get_level(self)`
- Defined: `premium_synergy_democratic.py:562`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `premium_synergy_democratic.py:567`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x)`
- Defined: `premium_synergy_democratic.py:579`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:597`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x)`
- Defined: `premium_synergy_democratic.py:604`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_balance `def get_balance(self)`
- Defined: `premium_synergy_democratic.py:612`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:617`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x)`
- Defined: `premium_synergy_democratic.py:627`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_state `def get_state(self)`
- Defined: `premium_synergy_democratic.py:645`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:650`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x, chaos_level)`
- Defined: `premium_synergy_democratic.py:660`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_resistance `def get_resistance(self)`
- Defined: `premium_synergy_democratic.py:678`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `premium_synergy_democratic.py:684`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x, plasticity)`
- Defined: `premium_synergy_democratic.py:692`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### consolidate_svd `def consolidate_svd(self, strength)`
- Defined: `premium_synergy_democratic.py:703`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:714`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x, is_chaos)`
- Defined: `premium_synergy_democratic.py:721`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_metrics `def get_metrics(self)`
- Defined: `premium_synergy_democratic.py:733`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:740`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x, phase_idx)`
- Defined: `premium_synergy_democratic.py:746`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### update `def update(self, x, phase_idx)`
- Defined: `premium_synergy_democratic.py:754`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_coherence `def get_coherence(self)`
- Defined: `premium_synergy_democratic.py:762`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:767`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x)`
- Defined: `premium_synergy_democratic.py:777`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_level `def get_level(self)`
- Defined: `premium_synergy_democratic.py:795`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:800`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x, chaos)`
- Defined: `premium_synergy_democratic.py:810`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### get_control `def get_control(self)`
- Defined: `premium_synergy_democratic.py:829`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, config)`
- Defined: `premium_synergy_democratic.py:839`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, topobrain_out, omnibrain_out, quimera_out, target_accuracy)`
- Defined: `premium_synergy_democratic.py:861`
- Doc: Cámara Alta: Delibera sobre las sinergias de los componentes
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### adjust_for_convergence `def adjust_for_convergence(self, performance_metrics)`
- Defined: `premium_synergy_democratic.py:926`
- Doc: Motor homeostático ajusta si las sinergias no convergen
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, config)`
- Defined: `premium_synergy_democratic.py:963`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x, chaos_level)`
- Defined: `premium_synergy_democratic.py:989`
- Doc: Forward pass completo con sistema democrático
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### democratic_deliberation_status `def democratic_deliberation_status(self)`
- Defined: `premium_synergy_democratic.py:1061`
- Doc: Estado de la deliberación democrática
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### __init__ `def __init__(self, dim)`
- Defined: `premium_synergy_democratic.py:1076`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

### forward `def forward(self, x)`
- Defined: `premium_synergy_democratic.py:1086`
- Imported by: `min_test_synergy.py`, `test_premium_synergy.py`

## quen7.py

### seed_everything `def seed_everything(seed)`
- Defined: `quen7.py:45`

### train_nonstationary `def train_nonstationary(config)`
- Defined: `quen7.py:265`

### run_ablation_study `def run_ablation_study()`
- Defined: `quen7.py:330`

### __init__ `def __init__(self)`
- Defined: `quen7.py:56`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `quen7.py:66`

### get_full `def get_full(self)`
- Defined: `quen7.py:80`

### get_w2 `def get_w2(self)`
- Defined: `quen7.py:83`

### __init__ `def __init__(self, d_in)`
- Defined: `quen7.py:90`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `quen7.py:100`

### __init__ `def __init__(self, d_in, d_out, dynamic)`
- Defined: `quen7.py:116`

### forward `def forward(self, x)`
- Defined: `quen7.py:128`

### __init__ `def __init__(self, in_dim)`
- Defined: `quen7.py:159`

### forward `def forward(self, x)`
- Defined: `quen7.py:167`

### __init__ `def __init__(self, config)`
- Defined: `quen7.py:174`

### count_parameters `def count_parameters(self)`
- Defined: `quen7.py:187`

### forward `def forward(self, x)`
- Defined: `quen7.py:190`

### __init__ `def __init__(self)`
- Defined: `quen7.py:216`

### update `def update(self, loss, liquid_norm, physio, prediction_error)`
- Defined: `quen7.py:226`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `quen7.py:234`

### report `def report(self, step, phase)`
- Defined: `quen7.py:239`

## quimera.py

### seed_everything `def seed_everything(seed)`
- Defined: `quimera.py:37`

### measure_spatial_richness `def measure_spatial_richness(activations)`
- Defined: `quimera.py:47`
- Doc: Mide la diversidad espacial de las activaciones (Richness)

### get_structure_entropy `def get_structure_entropy(model)`
- Defined: `quimera.py:62`
- Doc: Mide la entropía estructural de los pesos (Entropy)

### train_chimera_scientific `def train_chimera_scientific(config, verbose)`
- Defined: `quimera.py:279`

### generate_chimera_matrix `def generate_chimera_matrix()`
- Defined: `quimera.py:361`

### run_scientific_study `def run_scientific_study()`
- Defined: `quimera.py:399`

### __init__ `def __init__(self)`
- Defined: `quimera.py:87`

### get_batch `def get_batch(self, phase, batch_size)`
- Defined: `quimera.py:97`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `quimera.py:116`

### forward `def forward(self, x, plasticity)`
- Defined: `quimera.py:124`

### consolidate_svd `def consolidate_svd(self, strength)`
- Defined: `quimera.py:138`
- Doc: Mecanismo de SVD (Science-ready)

### __init__ `def __init__(self, dim)`
- Defined: `quimera.py:155`

### forward `def forward(self, x, is_chaos)`
- Defined: `quimera.py:165`

### __init__ `def __init__(self, dim)`
- Defined: `quimera.py:178`

### forward `def forward(self, x, phase_idx)`
- Defined: `quimera.py:184`

### update `def update(self, x, phase_idx)`
- Defined: `quimera.py:191`

### __init__ `def __init__(self, config)`
- Defined: `quimera.py:202`

### forward `def forward(self, x, phase_idx)`
- Defined: `quimera.py:229`

### consolidate `def consolidate(self)`
- Defined: `quimera.py:267`

## quimera_vision.py

### text_to_seq `def text_to_seq(text)`
- Defined: `quimera_vision.py:37`

### collate `def collate(batch)`
- Defined: `quimera_vision.py:87`

### generate_caption `def generate_caption(img_path, audio_path)`
- Defined: `quimera_vision.py:168`

### __init__ `def __init__(self)`
- Defined: `quimera_vision.py:44`

### __len__ `def __len__(self)`
- Defined: `quimera_vision.py:62`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `quimera_vision.py:64`

### __init__ `def __init__(self)`
- Defined: `quimera_vision.py:97`

### forward `def forward(self, x)`
- Defined: `quimera_vision.py:103`

### __init__ `def __init__(self)`
- Defined: `quimera_vision.py:109`

### forward `def forward(self, x)`
- Defined: `quimera_vision.py:116`

### __init__ `def __init__(self)`
- Defined: `quimera_vision.py:122`

### forward `def forward(self, img, audio, seq)`
- Defined: `quimera_vision.py:127`

## qwen.py

### seed_everything `def seed_everything(seed)`
- Defined: `qwen.py:57`

### get_dataset `def get_dataset(config)`
- Defined: `qwen.py:64`

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps, pgd_loss)`
- Defined: `qwen.py:283`

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `qwen.py:306`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `qwen.py:367`

### run_ablation_study `def run_ablation_study()`
- Defined: `qwen.py:392`

### __init__ `def __init__(self)`
- Defined: `qwen.py:89`

### forward `def forward(self, x, logits, h_agg, h_proc, w_norm, entropy, ortho, pgd_loss)`
- Defined: `qwen.py:99`

### __init__ `def __init__(self, dim)`
- Defined: `qwen.py:122`

### forward `def forward(self, x, strength)`
- Defined: `qwen.py:132`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `qwen.py:144`

### forward `def forward(self, x, influence)`
- Defined: `qwen.py:151`

### __init__ `def __init__(self, num_nodes)`
- Defined: `qwen.py:163`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `qwen.py:176`

### __init__ `def __init__(self, in_dim)`
- Defined: `qwen.py:182`

### forward `def forward(self, x, gain)`
- Defined: `qwen.py:190`

### __init__ `def __init__(self, config)`
- Defined: `qwen.py:197`

### _init_weights `def _init_weights(self)`
- Defined: `qwen.py:215`

### count_parameters `def count_parameters(self)`
- Defined: `qwen.py:220`

### forward `def forward(self, x, pgd_loss)`
- Defined: `qwen.py:223`

## qwen3.py

### seed_everything `def seed_everything(seed)`
- Defined: `qwen3.py:50`

### train_nonstationary `def train_nonstationary(config)`
- Defined: `qwen3.py:263`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `qwen3.py:320`

### run_ablation_study `def run_ablation_study()`
- Defined: `qwen3.py:352`

### __init__ `def __init__(self)`
- Defined: `qwen3.py:61`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `qwen3.py:71`

### get_full `def get_full(self)`
- Defined: `qwen3.py:85`

### get_w2 `def get_w2(self)`
- Defined: `qwen3.py:88`

### __init__ `def __init__(self, d_in)`
- Defined: `qwen3.py:95`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `qwen3.py:105`

### __init__ `def __init__(self, d_in, d_out, dynamic)`
- Defined: `qwen3.py:121`

### forward `def forward(self, x)`
- Defined: `qwen3.py:132`

### __init__ `def __init__(self, dim, atoms)`
- Defined: `qwen3.py:161`

### forward `def forward(self, x, influence)`
- Defined: `qwen3.py:168`

### __init__ `def __init__(self, num_nodes)`
- Defined: `qwen3.py:180`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `qwen3.py:193`

### __init__ `def __init__(self, config)`
- Defined: `qwen3.py:202`

### count_parameters `def count_parameters(self)`
- Defined: `qwen3.py:221`

### forward `def forward(self, x)`
- Defined: `qwen3.py:224`

## qwen4.py

### seed_everything `def seed_everything(seed)`
- Defined: `qwen4.py:50`

### train_nonstationary `def train_nonstationary(config)`
- Defined: `qwen4.py:263`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `qwen4.py:325`

### run_ablation_study `def run_ablation_study()`
- Defined: `qwen4.py:357`

### __init__ `def __init__(self)`
- Defined: `qwen4.py:61`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `qwen4.py:71`

### get_full `def get_full(self)`
- Defined: `qwen4.py:85`

### get_w2 `def get_w2(self)`
- Defined: `qwen4.py:88`

### __init__ `def __init__(self, d_in)`
- Defined: `qwen4.py:95`

### forward `def forward(self, x, h_pre, w_norm)`
- Defined: `qwen4.py:105`

### __init__ `def __init__(self, d_in, d_out, dynamic)`
- Defined: `qwen4.py:121`

### forward `def forward(self, x)`
- Defined: `qwen4.py:132`

### __init__ `def __init__(self, dim, atoms)`
- Defined: `qwen4.py:161`

### forward `def forward(self, x, influence)`
- Defined: `qwen4.py:168`

### __init__ `def __init__(self, num_nodes)`
- Defined: `qwen4.py:180`

### get_adjacency `def get_adjacency(self, plasticity)`
- Defined: `qwen4.py:193`

### __init__ `def __init__(self, config)`
- Defined: `qwen4.py:202`

### count_parameters `def count_parameters(self)`
- Defined: `qwen4.py:221`

### forward `def forward(self, x)`
- Defined: `qwen4.py:224`

## qwen5.py

### seed_everything `def seed_everything(seed)`
- Defined: `qwen5.py:49`

### train_predictive `def train_predictive(config)`
- Defined: `qwen5.py:285`

### run_experiment `def run_experiment()`
- Defined: `qwen5.py:351`

### __init__ `def __init__(self)`
- Defined: `qwen5.py:60`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `qwen5.py:71`

### get_full `def get_full(self)`
- Defined: `qwen5.py:85`

### get_w2 `def get_w2(self)`
- Defined: `qwen5.py:88`

### __init__ `def __init__(self, hidden_dim)`
- Defined: `qwen5.py:96`

### forward `def forward(self, phase_id)`
- Defined: `qwen5.py:103`

### __init__ `def __init__(self, capacity)`
- Defined: `qwen5.py:118`

### store `def store(self, phase, metrics, state)`
- Defined: `qwen5.py:122`

### retrieve `def retrieve(self, phase, top_k)`
- Defined: `qwen5.py:129`

### __init__ `def __init__(self, d_in)`
- Defined: `qwen5.py:139`

### forward `def forward(self, x, h_pre, w_norm, phase, reward)`
- Defined: `qwen5.py:153`

### __init__ `def __init__(self, d_in, d_out, dynamic)`
- Defined: `qwen5.py:197`

### forward `def forward(self, x, phase, reward)`
- Defined: `qwen5.py:209`

### consolidate_svd `def consolidate_svd(self, repair_strength)`
- Defined: `qwen5.py:239`

### __init__ `def __init__(self, config)`
- Defined: `qwen5.py:248`

### count_parameters `def count_parameters(self)`
- Defined: `qwen5.py:261`

### forward `def forward(self, x, phase, reward)`
- Defined: `qwen5.py:264`

## qwen6.py

### seed_everything `def seed_everything(seed)`
- Defined: `qwen6.py:43`

### train_nested `def train_nested(config)`
- Defined: `qwen6.py:202`

### run_experiment `def run_experiment()`
- Defined: `qwen6.py:265`

### __init__ `def __init__(self)`
- Defined: `qwen6.py:54`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `qwen6.py:64`

### get_full `def get_full(self)`
- Defined: `qwen6.py:78`
- Doc: Retorna el dataset completo

### get_w2 `def get_w2(self)`
- Defined: `qwen6.py:82`
- Doc: Retorna solo los datos de WORLD_2 (dígitos >= 5)

### __init__ `def __init__(self, input_dim, hidden_dim)`
- Defined: `qwen6.py:90`

### forward `def forward(self, x)`
- Defined: `qwen6.py:97`

### __init__ `def __init__(self, levels, d_model, hidden_dim)`
- Defined: `qwen6.py:111`

### forward `def forward(self, x, global_step)`
- Defined: `qwen6.py:123`

### __init__ `def __init__(self, d_in, d_out, config)`
- Defined: `qwen6.py:134`

### forward `def forward(self, x, global_step)`
- Defined: `qwen6.py:145`

### __init__ `def __init__(self, config)`
- Defined: `qwen6.py:170`

### forward `def forward(self, x, global_step)`
- Defined: `qwen6.py:182`

## qwen8.py

### seed_everything `def seed_everything(seed)`
- Defined: `qwen8.py:45`

### train_nonstationary `def train_nonstationary(config)`
- Defined: `qwen8.py:267`

### run_ablation_study `def run_ablation_study()`
- Defined: `qwen8.py:332`

### __init__ `def __init__(self)`
- Defined: `qwen8.py:56`

### get_batch `def get_batch(self, phase, bs)`
- Defined: `qwen8.py:66`

### get_full `def get_full(self)`
- Defined: `qwen8.py:80`

### get_w2 `def get_w2(self)`
- Defined: `qwen8.py:83`

### __init__ `def __init__(self, d_in)`
- Defined: `qwen8.py:90`

### forward `def forward(self, x, h_pre, w_norm, task_loss)`
- Defined: `qwen8.py:100`

### __init__ `def __init__(self, d_in, d_out, dynamic)`
- Defined: `qwen8.py:117`

### forward `def forward(self, x, task_loss)`
- Defined: `qwen8.py:129`

### __init__ `def __init__(self, in_dim)`
- Defined: `qwen8.py:161`

### forward `def forward(self, x)`
- Defined: `qwen8.py:169`

### __init__ `def __init__(self, config)`
- Defined: `qwen8.py:176`

### count_parameters `def count_parameters(self)`
- Defined: `qwen8.py:189`

### forward `def forward(self, x, task_loss)`
- Defined: `qwen8.py:192`

### __init__ `def __init__(self)`
- Defined: `qwen8.py:218`

### update `def update(self, loss, liquid_norm, physio, prediction_error)`
- Defined: `qwen8.py:228`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `qwen8.py:236`

### report `def report(self, step, phase)`
- Defined: `qwen8.py:241`

## qwen9.py

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `qwen9.py:371`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `qwen9.py:386`

### train_bicameral `def train_bicameral()`
- Defined: `qwen9.py:400`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `qwen9.py:35`

### forward `def forward(self, x, global_plasticity, transfer_rate)`
- Defined: `qwen9.py:49`

### __init__ `def __init__(self, output_dim)`
- Defined: `qwen9.py:79`

### forward `def forward(self, image, plasticity, transfer_rate)`
- Defined: `qwen9.py:88`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `qwen9.py:98`

### forward `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)`
- Defined: `qwen9.py:119`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `qwen9.py:177`

### _top_p_filtering `def _top_p_filtering(self, logits, top_p)`
- Defined: `qwen9.py:182`

### __init__ `def __init__(self, dim)`
- Defined: `qwen9.py:196`

### forward `def forward(self, right_features, left_context)`
- Defined: `qwen9.py:209`

### __init__ `def __init__(self, dim)`
- Defined: `qwen9.py:220`

### forward `def forward(self, right_features, epoch)`
- Defined: `qwen9.py:230`

### update_flow_ema `def update_flow_ema(self, flow)`
- Defined: `qwen9.py:245`

### __init__ `def __init__(self, vocab_size)`
- Defined: `qwen9.py:252`

### forward `def forward(self, image, captions, epoch, return_diagnostics)`
- Defined: `qwen9.py:259`

### __init__ `def __init__(self)`
- Defined: `qwen9.py:292`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `qwen9.py:299`

### measure_vocab_diversity `def measure_vocab_diversity(self, tokens, vocab_size)`
- Defined: `qwen9.py:306`

### update `def update(self)`
- Defined: `qwen9.py:312`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `qwen9.py:317`

### report `def report(self, epoch)`
- Defined: `qwen9.py:321`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `qwen9.py:339`

### __len__ `def __len__(self)`
- Defined: `qwen9.py:355`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `qwen9.py:358`

## qwn2.py

### seed_everything `def seed_everything(seed)`
- Defined: `qwn2.py:54`

### get_dataset `def get_dataset(config)`
- Defined: `qwn2.py:61`

### micro_pgd_attack `def micro_pgd_attack(model, x, y, eps, steps)`
- Defined: `qwn2.py:259`

### train_with_cv `def train_with_cv(config, dataset, cv_folds)`
- Defined: `qwn2.py:279`

### generate_ablation_matrix `def generate_ablation_matrix()`
- Defined: `qwn2.py:341`

### run_ablation_study `def run_ablation_study()`
- Defined: `qwn2.py:366`

### __init__ `def __init__(self, input_dim)`
- Defined: `qwn2.py:81`

### forward `def forward(self, signals)`
- Defined: `qwn2.py:91`

### __init__ `def __init__(self, num_nodes, grid_size)`
- Defined: `qwn2.py:98`

### get_adjacency `def get_adjacency(self, x, h_agg)`
- Defined: `qwn2.py:111`

### __init__ `def __init__(self, dim)`
- Defined: `qwn2.py:123`

### forward `def forward(self, x)`
- Defined: `qwn2.py:133`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `qwn2.py:156`

### forward `def forward(self, x)`
- Defined: `qwn2.py:164`

### __init__ `def __init__(self, in_dim)`
- Defined: `qwn2.py:180`

### forward `def forward(self, x, entropy)`
- Defined: `qwn2.py:189`

### __init__ `def __init__(self, config)`
- Defined: `qwn2.py:200`

### _init_weights `def _init_weights(self)`
- Defined: `qwn2.py:215`

### count_parameters `def count_parameters(self)`
- Defined: `qwn2.py:220`

### forward `def forward(self, x)`
- Defined: `qwn2.py:223`

## resma4.10.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.10.py:604`

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.10.py:662`

### _make_serializable `def _make_serializable(obj, depth, max_depth, _visited)`
- Defined: `resma4.10.py:690`
- Doc: Convierte objetos a formato serializable de forma segura.

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- Defined: `resma4.10.py:794`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.10.py:50`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.10.py:74`

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.10.py:86`

### modulation_factor `def modulation_factor(self)`
- Defined: `resma4.10.py:89`

### to_dict `def to_dict(self)`
- Defined: `resma4.10.py:92`

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.10.py:102`

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.10.py:112`

### _construir_generadores_aleatorios `def _construir_generadores_aleatorios(self)`
- Defined: `resma4.10.py:121`

### _hadamard_generalizado `def _hadamard_generalizado(self)`
- Defined: `resma4.10.py:130`

### operator `def operator(self)`
- Defined: `resma4.10.py:135`

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.10.py:150`

### __init__ `def __init__(self, garnier)`
- Defined: `resma4.10.py:159`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red, b1)`
- Defined: `resma4.10.py:163`

### es_silencio_activo `def es_silencio_activo(self, rho_red, b1)`
- Defined: `resma4.10.py:170`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.10.py:193`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.10.py:197`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.10.py:203`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.10.py:234`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.10.py:270`

### _generate_complete_measure `def _generate_complete_measure(self)`
- Defined: `resma4.10.py:281`

### _aplicar_modulacion_garnier `def _aplicar_modulacion_garnier(self, measure)`
- Defined: `resma4.10.py:310`

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.10.py:321`

### _calcular_libertad `def _calcular_libertad(self)`
- Defined: `resma4.10.py:330`

### _calcular_coherencia `def _calcular_coherencia(self)`
- Defined: `resma4.10.py:333`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- Defined: `resma4.10.py:342`

### _generate_realistic_modular_network `def _generate_realistic_modular_network(self)`
- Defined: `resma4.10.py:380`

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.10.py:454`

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.10.py:462`

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.10.py:484`

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.10.py:488`

### _validar_axioma_6 `def _validar_axioma_6(self)`
- Defined: `resma4.10.py:496`

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.10.py:510`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.10.py:527`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.10.py:532`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.10.py:538`

### __init__ `def __init__(self, universe, network, myelin)`
- Defined: `resma4.10.py:546`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.10.py:551`

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.10.py:594`

### log_resources `def log_resources()`
- Defined: `resma4.10.py:599`

## resma4.2.py

### simulate_resma_complete `def simulate_resma_complete(n_leaves, n_nodes, seed)`
- Defined: `resma4.2.py:556`
- Doc: Pipeline RESMA 4.2 completo

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.2.py:67`
- Doc: Verificar condición PT: κ < χΩ

### validate_dimension `def validate_dimension(alpha, tolerance)`
- Defined: `resma4.2.py:80`

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `resma4.2.py:88`

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `resma4.2.py:96`

### validate_spectral_dimension `def validate_spectral_dimension(dim)`
- Defined: `resma4.2.py:101`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.2.py:117`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.2.py:121`
- Doc: ρ(ω) con regularización UV

### modular_entropy `def modular_entropy(self)`
- Defined: `resma4.2.py:126`
- Doc: S = -∫ ρ log ρ dω

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.2.py:136`
- Doc: Distancia de Bures W₂(ρ₁, ρ₂)

### haagerup_weight `def haagerup_weight(self)`
- Defined: `resma4.2.py:154`
- Doc: Peso de Haagerup para regularización

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `resma4.2.py:165`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.2.py:176`

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `resma4.2.py:189`
- Doc: μ(i,j) = exp(-β·W₂²(ρᵢ, ρⱼ))

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.2.py:205`
- Doc: Estado global: pesos por hoja

### compute_gibbs_free_energy `def compute_gibbs_free_energy(self)`
- Defined: `resma4.2.py:216`
- Doc: F = -ln(Tr(μ)) / β

### __init__ `def __init__(self, universe, n_samples)`
- Defined: `resma4.2.py:227`

### _construct_hardy_state `def _construct_hardy_state(self)`
- Defined: `resma4.2.py:234`
- Doc: E(z) ∈ H²(ℂ⁺)

### _szego_projector `def _szego_projector(self)`
- Defined: `resma4.2.py:238`
- Doc: Proyector en frecuencias positivas

### _evaluation_functional `def _evaluation_functional(self, state_weights)`
- Defined: `resma4.2.py:246`
- Doc: Φ_E[|Ψ⟩] = exp(∫ log(⟨Φᵢ|E⟩) dμ)

### project `def project(self, state_vector)`
- Defined: `resma4.2.py:258`
- Doc: P̂_E = P_E ∘ Φ_E (con interpolación adaptativa)

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.2.py:297`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.2.py:304`
- Doc: H₀: dispersión Ω(q) = Ω₀ + q² + χq³

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.2.py:310`
- Doc: V_loss ∝ (r/a₀)^(2α)

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.2.py:317`
- Doc: Campo escalar para estabilización Spin(7)

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `resma4.2.py:321`
- Doc: κ < χΩ

### coherence_quantum `def coherence_quantum(self)`
- Defined: `resma4.2.py:327`
- Doc: Coherencia cuántica con verificación espectral

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `resma4.2.py:354`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `resma4.2.py:366`
- Doc: Scale-free → NO DIRIGIDO

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.2.py:381`
- Doc: d_s = -2 lim log N(λ)/log λ

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.2.py:410`
- Doc: R_Q(G) = min{n | β_{n-1}(G) > 0}

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.2.py:429`
- Doc: Números de Betti β₀, β₁

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `resma4.2.py:445`
- Doc: Matriz de distancias para homología

### critical_percolation_time `def critical_percolation_time(self)`
- Defined: `resma4.2.py:461`
- Doc: t_c = 21 · (N/N₀)^0.25 / log R_Q

### __init__ `def __init__(self, universe, myelin, network)`
- Defined: `resma4.2.py:477`

### predict_all `def predict_all(self)`
- Defined: `resma4.2.py:483`
- Doc: Predicciones RESMA 4.2

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.2.py:496`
- Doc: ln(BF) con AIC

## resma4.3.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.3.py:54`
- Doc: Guardado atómico con backup

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.3.py:84`
- Doc: Cargar checkpoint con fallback

### simulate_resma_with_checkpointing `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)`
- Defined: `resma4.3.py:545`
- Doc: Pipeline con reanudación inteligente desde checkpoints

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.3.py:35`

### check_memory_limit `def check_memory_limit()`
- Defined: `resma4.3.py:40`

### log_resources `def log_resources()`
- Defined: `resma4.3.py:49`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.3.py:127`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.3.py:150`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.3.py:154`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.3.py:158`
- Doc: Distancia Bures con caché EXTERNO (no en instancia)

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `resma4.3.py:193`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.3.py:215`

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `resma4.3.py:227`
- Doc: Matriz de medida con guardado incremental

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.3.py:254`

### validate_dimension `def validate_dimension(alpha, tolerance)`
- Defined: `resma4.3.py:271`

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `resma4.3.py:279`

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `resma4.3.py:287`

### validate_spectral_dimension `def validate_spectral_dimension(dim)`
- Defined: `resma4.3.py:292`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.3.py:302`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.3.py:312`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.3.py:317`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.3.py:323`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `resma4.3.py:326`

### coherence_quantum `def coherence_quantum(self)`
- Defined: `resma4.3.py:331`

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `resma4.3.py:353`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `resma4.3.py:372`
- Doc: Generar grafo por lotes

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.3.py:401`
- Doc: Dimensión espectral con matriz sparse

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.3.py:425`
- Doc: Ramsey topológico

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.3.py:444`
- Doc: Números de Betti

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `resma4.3.py:460`
- Doc: Matriz de distancias sparse

### critical_percolation_time `def critical_percolation_time(self)`
- Defined: `resma4.3.py:476`
- Doc: Tiempo crítico de percolación

### __init__ `def __init__(self, universe, myelin, network)`
- Defined: `resma4.3.py:486`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.3.py:492`
- Doc: ln(BF)

## resma4.4.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.4.py:59`
- Doc: Guarda el estado COMPLETO de los objetos, no solo metadatos

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.4.py:93`
- Doc: Carga el estado COMPLETO desde disco

### simulate_resma_with_checkpointing `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)`
- Defined: `resma4.4.py:554`
- Doc: Pipeline con reanudación que realmente carga objetos

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.4.py:36`

### check_memory_limit `def check_memory_limit()`
- Defined: `resma4.4.py:41`

### log_resources `def log_resources()`
- Defined: `resma4.4.py:50`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.4.py:146`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.4.py:168`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.4.py:172`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.4.py:176`
- Doc: Distancia Bures con caché externo

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state)`
- Defined: `resma4.4.py:206`
- Doc: Constructor que puede recibir estado serializado

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.4.py:258`

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `resma4.4.py:269`
- Doc: Matriz de medida

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.4.py:301`

### validate_dimension `def validate_dimension(alpha, tolerance)`
- Defined: `resma4.4.py:318`

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `resma4.4.py:326`

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `resma4.4.py:334`

### validate_spectral_dimension `def validate_spectral_dimension(dim)`
- Defined: `resma4.4.py:339`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.4.py:348`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.4.py:358`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.4.py:363`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.4.py:369`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `resma4.4.py:372`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti)`
- Defined: `resma4.4.py:378`
- Doc: Constructor que puede recibir grafo ya construido

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `resma4.4.py:446`
- Doc: Generar grafo por lotes

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.4.py:475`
- Doc: Dimensión espectral con eigenvalores sparse

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.4.py:499`
- Doc: Ramsey topológico

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.4.py:518`
- Doc: Números de Betti

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `resma4.4.py:534`
- Doc: Matriz de distancias sparse

## resma4.5.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.5.py:59`

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.5.py:88`

### _make_serializable `def _make_serializable(obj)`
- Defined: `resma4.5.py:113`
- Doc: Convierte objetos a formato serializable

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- Defined: `resma4.5.py:695`
- Doc: Pipeline único con Garnier integrado

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.5.py:36`

### check_memory_limit `def check_memory_limit()`
- Defined: `resma4.5.py:41`

### log_resources `def log_resources()`
- Defined: `resma4.5.py:50`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.5.py:152`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.5.py:170`

### factor_escala `def factor_escala(self, tiempo_idx)`
- Defined: `resma4.5.py:181`
- Doc: Factor de escala para cada tiempo: 0=lento, 2=modular, 3=teleológico

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.5.py:185`
- Doc: Entropía crítica de percolación (ADIMENSIONAL).

### to_dict `def to_dict(self)`
- Defined: `resma4.5.py:192`
- Doc: Para serialización

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.5.py:197`

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.5.py:206`

### _construir_generadores_E8 `def _construir_generadores_E8(self)`
- Defined: `resma4.5.py:214`
- Doc: Construye 3 generadores temporales (antis-Hermitianos)

### _hadamard_generalizado `def _hadamard_generalizado(self)`
- Defined: `resma4.5.py:226`
- Doc: Operador de Hadamard en dimensión 248 (unitario)

### operator `def operator(self)`
- Defined: `resma4.5.py:235`
- Doc: Construye D̂_G(ϕ) dimensionalmente consistente

### aplicar_a_estado `def aplicar_a_estado(self, estado)`
- Defined: `resma4.5.py:254`
- Doc: Aplica desdoblamiento a un estado cuántico |Ψ⟩

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.5.py:260`
- Doc: α'(ϕ) = α · tanh(C0/C3 · cos(ϕ₃))

### __init__ `def __init__(self, garnier, network)`
- Defined: `resma4.5.py:273`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red)`
- Defined: `resma4.5.py:278`
- Doc: ΔS_loop = S_vN(ρ_red) - log(b₁ + 1)

### _calcular_rho_reducida_aproximada `def _calcular_rho_reducida_aproximada(self)`
- Defined: `resma4.5.py:299`
- Doc: Aproximación: ρ_red = diag(grados) / sum(grados)

### es_silencio_activo `def es_silencio_activo(self, rho_red)`
- Defined: `resma4.5.py:307`
- Doc: Verifica Silencio-Activo y calcula Libertad L.

### umbral_percolacion `def umbral_percolacion(self)`
- Defined: `resma4.5.py:324`
- Doc: Umbral de percolación para soberanía: 70% (Axioma 6)

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.5.py:345`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.5.py:349`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.5.py:353`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.5.py:383`
- Doc: Constructor que puede recibir estado serializado

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.5.py:420`

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `resma4.5.py:431`
- Doc: Matriz de medida sin desdoblamiento

### _aplicar_desdoblamiento_a_medida `def _aplicar_desdoblamiento_a_medida(self, measure)`
- Defined: `resma4.5.py:451`
- Doc: Aplica D̂_G(ϕ) a la medida:

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.5.py:471`

### _calcular_libertad_universo `def _calcular_libertad_universo(self)`
- Defined: `resma4.5.py:481`
- Doc: Libertad del universo: L = 1/ε_c

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.5.py:488`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.5.py:499`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.5.py:504`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.5.py:510`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `resma4.5.py:513`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- Defined: `resma4.5.py:520`
- Doc: Constructor que puede recibir grafo ya construido

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `resma4.5.py:559`
- Doc: Generar grafo por lotes con conectividad controlada

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.5.py:591`
- Doc: Dimensión espectral con eigenvalores sparse

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.5.py:615`
- Doc: Ramsey topológico simplificado

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.5.py:627`
- Doc: Números de Betti aproximados por ciclos locales

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.5.py:636`
- Doc: Matriz densidad reducida del conectoma

### validar_axioma_6 `def validar_axioma_6(self)`
- Defined: `resma4.5.py:644`
- Doc: Verifica: conectividad > 70% para soberanía

### __init__ `def __init__(self, universe, myelin, network)`
- Defined: `resma4.5.py:661`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.5.py:666`
- Doc: Calcula Factor de Bayes integrando Garnier

## resma4.6.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.6.py:56`
- Doc: Guarda estado completo con manejo robusto de errores

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.6.py:86`
- Doc: Carga checkpoint con fallback automático

### _make_serializable `def _make_serializable(obj)`
- Defined: `resma4.6.py:112`
- Doc: Convierte objetos recursivamente a formato serializable

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart, target_connectivity)`
- Defined: `resma4.6.py:951`
- Doc: Pipeline completo RESMA 4.3.5 con antagonismo ZPE-Silencio

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.6.py:33`

### check_memory_limit `def check_memory_limit(threshold)`
- Defined: `resma4.6.py:38`

### log_resources `def log_resources()`
- Defined: `resma4.6.py:47`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.6.py:158`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.6.py:181`

### factor_escala `def factor_escala(self, tiempo_idx)`
- Defined: `resma4.6.py:194`
- Doc: Factor de escala con supresión ZPE

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.6.py:200`
- Doc: **UMBRAL CRÍTICO CON ZPE**:

### to_dict `def to_dict(self)`
- Defined: `resma4.6.py:208`
- Doc: Serialización completa

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.6.py:220`
- Doc: Deserialización

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.6.py:238`

### _construir_generadores_E8_ZPE `def _construir_generadores_E8_ZPE(self)`
- Defined: `resma4.6.py:246`
- Doc: GENERADORES CON CANCELACIÓN ZPE INTEGRADA

### _hadamard_generalizado_ZPE `def _hadamard_generalizado_ZPE(self)`
- Defined: `resma4.6.py:266`
- Doc: HADAMARD CON ESPACIO NULO ZPE

### operator `def operator(self)`
- Defined: `resma4.6.py:284`
- Doc: Construye D̂_G(ϕ) con cancelación ZPE

### alpha_modificado `def alpha_modificado(self, alpha_base)`
- Defined: `resma4.6.py:304`
- Doc: **α'(ϕ) = α · tanh(C₀/C₃ · cos(ϕ₃) · (1 - zpe_level))**

### __init__ `def __init__(self, garnier, network)`
- Defined: `resma4.6.py:318`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red)`
- Defined: `resma4.6.py:327`
- Doc: **ΔS_loop = S_vN(ρ_red) - S_top + S_ZPE**

### _calcular_rho_reducida_aproximada `def _calcular_rho_reducida_aproximada(self)`
- Defined: `resma4.6.py:367`
- Doc: Matriz densidad con modulación ZPE

### es_silencio_activo `def es_silencio_activo(self, rho_red)`
- Defined: `resma4.6.py:383`
- Doc: **DETECCIÓN DE ANTAGONISMO**:

### umbral_percolacion `def umbral_percolacion(self)`
- Defined: `resma4.6.py:411`
- Doc: Umbral para soberanía: 70%

### modo_goldstone `def modo_goldstone(self)`
- Defined: `resma4.6.py:415`
- Doc: **MODO GOLDSTONE DEL DOBLE CUÁNTICO**:

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.6.py:449`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.6.py:453`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.6.py:457`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.6.py:487`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.6.py:521`
- Doc: Inicializa hojas con temperatura efectiva afectada por ZPE

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `resma4.6.py:535`
- Doc: Genera medida de Gibbs

### _aplicar_desdoblamiento_a_medida `def _aplicar_desdoblamiento_a_medida(self, measure)`
- Defined: `resma4.6.py:557`
- Doc: Aplica desdoblamiento con supresión ZPE

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.6.py:584`
- Doc: Construye estado global normalizado

### _calcular_libertad_universo `def _calcular_libertad_universo(self)`
- Defined: `resma4.6.py:603`
- Doc: Libertad intrínseca con supresión ZPE

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.6.py:611`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.6.py:624`
- Doc: Hamiltoniano con energía ZPE incluida

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.6.py:633`
- Doc: Potencial de pérdida PT

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.6.py:640`

### _calcular_zpe `def _calcular_zpe(self)`
- Defined: `resma4.6.py:643`
- Doc: **ENERGÍA DE PUNTO CERO TOTAL**:

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `resma4.6.py:655`

### __init__ `def __init__(self, n_nodes, seed, garnier)`
- Defined: `resma4.6.py:667`

### _inicializar_amplitudes `def _inicializar_amplitudes(self)`
- Defined: `resma4.6.py:686`
- Doc: **AMPLITUDES DE FEYNMAN** para cada posible arista:

### _calcular_conectividad_cuantica `def _calcular_conectividad_cuantica(self)`
- Defined: `resma4.6.py:708`
- Doc: **CONECTIVIDAD CUÁNTICA** (no clásica):

### colapsar_a_clasico `def colapsar_a_clasico(self, threshold)`
- Defined: `resma4.6.py:718`
- Doc: **COLAPSO CUÁNTICO-CLÁSICO**:

### _recalcular_betti_clasicos `def _recalcular_betti_clasicos(self)`
- Defined: `resma4.6.py:742`
- Doc: Recalcula Betti del grafo colapsado

### medir_delta_s_loop `def medir_delta_s_loop(self)`
- Defined: `resma4.6.py:756`
- Doc: **ΔS_loop CUÁNTICO** (no clásico):

### _calcular_b1_cuantico `def _calcular_b1_cuantico(self)`
- Defined: `resma4.6.py:784`
- Doc: **b₁ CUÁNTICO** (topología pre-geométrica):

### __init__ `def __init__(self, n_nodes, seed, conectoma_quantum, garnier)`
- Defined: `resma4.6.py:803`

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.6.py:842`
- Doc: **Matriz densidad reducida del conectoma cuántico**

### validar_axioma_6_cuantico `def validar_axioma_6_cuantico(self)`
- Defined: `resma4.6.py:858`
- Doc: **AXIOMA 6 CUÁNTICO**: Conectividad cuántica > 70%

### obtener_metricas_cuanticas `def obtener_metricas_cuanticas(self)`
- Defined: `resma4.6.py:874`
- Doc: **MÉTRICAS EXPERIMENTALES** (falsables):

### __init__ `def __init__(self, universe, myelin, network)`
- Defined: `resma4.6.py:901`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.6.py:906`
- Doc: Calcula Factor de Bayes con antagonismo ZPE-Silencio

## resma4.7.py

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed)`
- Defined: `resma4.7.py:537`
- Doc: Pipeline completo RESMA-Garnier con correcciones

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.7.py:53`

### _compute_coupling `def _compute_coupling(self)`
- Defined: `resma4.7.py:67`
- Doc: Fuerza de acoplamiento entre tiempos

### factor_escala `def factor_escala(self, tiempo_idx)`
- Defined: `resma4.7.py:72`
- Doc: Factor de escala temporal

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.7.py:77`
- Doc: Entropía crítica con corrección de acoplamiento:

### modulation_factor `def modulation_factor(self)`
- Defined: `resma4.7.py:85`
- Doc: Factor de modulación para la medida cuántica:

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.7.py:98`

### _construir_generadores `def _construir_generadores(self)`
- Defined: `resma4.7.py:103`
- Doc: Generadores temporales (anti-Hermitianos normalizados)

### operator `def operator(self)`
- Defined: `resma4.7.py:115`
- Doc: Construye D̂_G(φ) = exp(i Σ φᵢHᵢ)

### aplicar_modulacion `def aplicar_modulacion(self, state_vector)`
- Defined: `resma4.7.py:120`
- Doc: Aplica desdoblamiento a vector de estado

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.7.py:126`
- Doc: α'(φ) = α · |cos(φ₃)|^(C0/C3)

### __init__ `def __init__(self, garnier)`
- Defined: `resma4.7.py:143`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red, b1)`
- Defined: `resma4.7.py:147`
- Doc: ΔS_loop = S_vN(ρ) - log(b₁ + 1)

### es_silencio_activo `def es_silencio_activo(self, rho_red, b1)`
- Defined: `resma4.7.py:166`
- Doc: Verifica condición y calcula libertad L = 1/(ΔS + ε_c)

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.7.py:197`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.7.py:203`
- Doc: Distancia de Bures simplificada

### __init__ `def __init__(self, n_leaves, seed, garnier)`
- Defined: `resma4.7.py:226`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.7.py:252`
- Doc: Genera hojas con gaps distribuidos exponencialmente

### _generate_modulated_measure `def _generate_modulated_measure(self)`
- Defined: `resma4.7.py:265`
- Doc: Genera medida de transición modulada por Garnier:

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.7.py:312`
- Doc: Estado global como distribución diagonal

### _calcular_libertad `def _calcular_libertad(self)`
- Defined: `resma4.7.py:322`
- Doc: Libertad del universo: L_U = 1/ε_c

### _calcular_coherencia `def _calcular_coherencia(self)`
- Defined: `resma4.7.py:326`
- Doc: Coherencia cuántica: suma de elementos off-diagonal

### __init__ `def __init__(self, n_nodes, seed, garnier)`
- Defined: `resma4.7.py:341`

### _generate_realistic_network `def _generate_realistic_network(self)`
- Defined: `resma4.7.py:379`
- Doc: Genera red con conectividad > 70% usando modelo realista:

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.7.py:424`
- Doc: Números de Betti: b0=componentes, b1=ciclos

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.7.py:433`
- Doc: Dimensión espectral del Laplaciano

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.7.py:455`
- Doc: Número de Ramsey topológico

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.7.py:460`
- Doc: Matriz densidad de la red (normalizada por grados)

### _validar_axioma_6 `def _validar_axioma_6(self)`
- Defined: `resma4.7.py:470`
- Doc: Verifica conectividad > 70%

### __init__ `def __init__(self, universe, network)`
- Defined: `resma4.7.py:487`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.7.py:491`
- Doc: ln(BF) ∝ log(L_red · L_univ)

## resma4.8.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.8.py:91`

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.8.py:117`

### _make_serializable `def _make_serializable(obj)`
- Defined: `resma4.8.py:141`

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- Defined: `resma4.8.py:667`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.8.py:54`

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.8.py:72`

### check_memory_limit `def check_memory_limit()`
- Defined: `resma4.8.py:77`

### log_resources `def log_resources()`
- Defined: `resma4.8.py:86`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.8.py:161`

### _compute_coupling `def _compute_coupling(self)`
- Defined: `resma4.8.py:179`

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.8.py:182`

### modulation_factor `def modulation_factor(self)`
- Defined: `resma4.8.py:186`

### to_dict `def to_dict(self)`
- Defined: `resma4.8.py:189`

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.8.py:199`

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.8.py:210`

### _construir_generadores_aleatorios `def _construir_generadores_aleatorios(self)`
- Defined: `resma4.8.py:219`

### _hadamard_generalizado `def _hadamard_generalizado(self)`
- Defined: `resma4.8.py:228`

### operator `def operator(self)`
- Defined: `resma4.8.py:233`

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.8.py:248`

### __init__ `def __init__(self, garnier)`
- Defined: `resma4.8.py:257`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red, b1)`
- Defined: `resma4.8.py:261`

### es_silencio_activo `def es_silencio_activo(self, rho_red, b1)`
- Defined: `resma4.8.py:268`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.8.py:291`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.8.py:295`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.8.py:301`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.8.py:332`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.8.py:362`

### _generate_complete_measure `def _generate_complete_measure(self)`
- Defined: `resma4.8.py:373`

### _aplicar_modulacion_garnier `def _aplicar_modulacion_garnier(self, measure)`
- Defined: `resma4.8.py:402`

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.8.py:413`

### _calcular_libertad `def _calcular_libertad(self)`
- Defined: `resma4.8.py:422`

### _calcular_coherencia `def _calcular_coherencia(self)`
- Defined: `resma4.8.py:425`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- Defined: `resma4.8.py:434`

### _generate_realistic_modular_network `def _generate_realistic_modular_network(self)`
- Defined: `resma4.8.py:472`

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.8.py:529`

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.8.py:537`

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.8.py:559`

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.8.py:563`

### _validar_axioma_6 `def _validar_axioma_6(self)`
- Defined: `resma4.8.py:571`

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.8.py:585`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.8.py:602`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.8.py:607`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.8.py:613`

### __init__ `def __init__(self, universe, network, myelin)`
- Defined: `resma4.8.py:621`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.8.py:626`

## resma4.9.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.9.py:587`

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.9.py:615`

### _make_serializable `def _make_serializable(obj)`
- Defined: `resma4.9.py:639`

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- Defined: `resma4.9.py:655`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.9.py:50`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.9.py:74`

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.9.py:86`

### modulation_factor `def modulation_factor(self)`
- Defined: `resma4.9.py:89`

### to_dict `def to_dict(self)`
- Defined: `resma4.9.py:92`

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.9.py:102`

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.9.py:112`

### _construir_generadores_aleatorios `def _construir_generadores_aleatorios(self)`
- Defined: `resma4.9.py:121`

### _hadamard_generalizado `def _hadamard_generalizado(self)`
- Defined: `resma4.9.py:130`

### operator `def operator(self)`
- Defined: `resma4.9.py:135`

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.9.py:150`

### __init__ `def __init__(self, garnier)`
- Defined: `resma4.9.py:159`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red, b1)`
- Defined: `resma4.9.py:163`

### es_silencio_activo `def es_silencio_activo(self, rho_red, b1)`
- Defined: `resma4.9.py:170`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.9.py:193`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.9.py:197`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.9.py:203`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.9.py:234`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.9.py:270`

### _generate_complete_measure `def _generate_complete_measure(self)`
- Defined: `resma4.9.py:281`

### _aplicar_modulacion_garnier `def _aplicar_modulacion_garnier(self, measure)`
- Defined: `resma4.9.py:310`

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.9.py:321`

### _calcular_libertad `def _calcular_libertad(self)`
- Defined: `resma4.9.py:330`

### _calcular_coherencia `def _calcular_coherencia(self)`
- Defined: `resma4.9.py:333`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- Defined: `resma4.9.py:342`

### _generate_realistic_modular_network `def _generate_realistic_modular_network(self)`
- Defined: `resma4.9.py:380`

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.9.py:437`

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.9.py:445`

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.9.py:467`

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.9.py:471`

### _validar_axioma_6 `def _validar_axioma_6(self)`
- Defined: `resma4.9.py:479`

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.9.py:493`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.9.py:510`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.9.py:515`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.9.py:521`

### __init__ `def __init__(self, universe, network, myelin)`
- Defined: `resma4.9.py:529`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.9.py:534`

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.9.py:577`

### log_resources `def log_resources()`
- Defined: `resma4.9.py:582`

## resma_Test.py

### stress_test_resma `def stress_test_resma(model)`
- Defined: `resma_Test.py:88`

### __init__ `def __init__(self, omega, chi, kappa_init)`
- Defined: `resma_Test.py:12`

### forward `def forward(self, x)`
- Defined: `resma_Test.py:18`

### __init__ `def __init__(self, in_features, out_features, q_order)`
- Defined: `resma_Test.py:43`

### _generate_ramsey_mask `def _generate_ramsey_mask(self)`
- Defined: `resma_Test.py:54`

### forward `def forward(self, x)`
- Defined: `resma_Test.py:64`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `resma_Test.py:72`

### forward `def forward(self, x)`
- Defined: `resma_Test.py:78`

## resmann.py

### __init__ `def __init__(self, omega, chi, kappa_init)`
- Defined: `resmann.py:12`

### forward `def forward(self, x)`
- Defined: `resmann.py:19`

### __init__ `def __init__(self, in_features, out_features, q_order)`
- Defined: `resmann.py:42`

### _generate_ramsey_mask `def _generate_ramsey_mask(self)`
- Defined: `resmann.py:59`

### forward `def forward(self, x)`
- Defined: `resmann.py:69`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `resmann.py:87`

### forward `def forward(self, x)`
- Defined: `resmann.py:97`

### resma_loss `def resma_loss(self, output, target, lambda_topo)`
- Defined: `resmann.py:104`

## resmann2.py

### __init__ `def __init__(self, omega, chi, kappa_init)`
- Defined: `resmann2.py:14`

### forward `def forward(self, x)`
- Defined: `resmann2.py:20`

### __init__ `def __init__(self, in_features, out_features, n_universes)`
- Defined: `resmann2.py:32`

### _multiverse_mask `def _multiverse_mask(self, in_f, out_f, n_univ)`
- Defined: `resmann2.py:41`

### forward `def forward(self, x)`
- Defined: `resmann2.py:55`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `resmann2.py:66`

### forward `def forward(self, x)`
- Defined: `resmann2.py:74`

### resma_loss `def resma_loss(self, output, target, lambda_topo)`
- Defined: `resmann2.py:79`

## resmannn.py

### __init__ `def __init__(self, omega, chi, kappa_init)`
- Defined: `resmannn.py:13`

### forward `def forward(self, x)`
- Defined: `resmannn.py:19`

### __init__ `def __init__(self, in_features, out_features)`
- Defined: `resmannn.py:31`

### _fixed_sparse_mask `def _fixed_sparse_mask(self, out_f, in_f)`
- Defined: `resmannn.py:40`

### forward `def forward(self, x)`
- Defined: `resmannn.py:49`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `resmannn.py:61`

### forward `def forward(self, x)`
- Defined: `resmannn.py:69`

### resma_loss `def resma_loss(self, output, target, lambda_topo)`
- Defined: `resmannn.py:74`

## run_complete_experiment.py

### create_experiment_summary `def create_experiment_summary(results_dir, metrics, duration)`
- Defined: `run_complete_experiment.py:24`
- Doc: Crea un resumen del experimento
- Depends on: `physio_chimera_v15_monitored.py`

### generate_final_report `def generate_final_report(results_dir, metrics, duration)`
- Defined: `run_complete_experiment.py:46`
- Doc: Genera reporte final detallado
- Depends on: `physio_chimera_v15_monitored.py`

### run_complete_experiment `def run_complete_experiment()`
- Defined: `run_complete_experiment.py:155`
- Doc: Ejecuta el experimento completo con todas las características
- Depends on: `physio_chimera_v15_monitored.py`

## scientific_benchmark.py

### seed_everything `def seed_everything(seed)`
- Defined: `scientific_benchmark.py:42`

### clamp_pgd `def clamp_pgd(x_adv_norm, x_orig_norm, eps)`
- Defined: `scientific_benchmark.py:276`

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps)`
- Defined: `scientific_benchmark.py:283`

### eval_autoattack `def eval_autoattack(model, test_loader, n_samples)`
- Defined: `scientific_benchmark.py:299`

### save_topology_snapshot `def save_topology_snapshot(model, epoch, run_name)`
- Defined: `scientific_benchmark.py:332`

### run_training `def run_training(config_override, run_name)`
- Defined: `scientific_benchmark.py:350`

### run_ablation_suite_scientific `def run_ablation_suite_scientific()`
- Defined: `scientific_benchmark.py:473`

### __init__ `def __init__(self, temperature)`
- Defined: `scientific_benchmark.py:58`

### forward `def forward(self, features, labels)`
- Defined: `scientific_benchmark.py:62`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `scientific_benchmark.py:86`

### forward `def forward(self, input_signal, prediction)`
- Defined: `scientific_benchmark.py:92`

### __init__ `def __init__(self, dim)`
- Defined: `scientific_benchmark.py:98`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `scientific_benchmark.py:107`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `scientific_benchmark.py:112`

### forward `def forward(self, x)`
- Defined: `scientific_benchmark.py:120`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- Defined: `scientific_benchmark.py:133`

### forward `def forward(self, x_nodes, adjacency, incidence)`
- Defined: `scientific_benchmark.py:157`

### __init__ `def __init__(self, config)`
- Defined: `scientific_benchmark.py:185`

### _init_grid `def _init_grid(self, N)`
- Defined: `scientific_benchmark.py:220`

### get_topology `def get_topology(self)`
- Defined: `scientific_benchmark.py:241`

### forward `def forward(self, x)`
- Defined: `scientific_benchmark.py:253`

### __init__ `def __init__(self, m)`
- Defined: `scientific_benchmark.py:318`

### forward `def forward(self, x)`
- Defined: `scientific_benchmark.py:319`

### lambda_topo `def lambda_topo(epoch)`
- Defined: `scientific_benchmark.py:387`

## scientist_sinergy_ablation_plan.py

### check_memory_usage `def check_memory_usage()`
- Defined: `scientist_sinergy_ablation_plan.py:63`
- Doc: Monitorea uso de memoria para evitar crashes con Nested Learning

### memory_safe_check `def memory_safe_check(config)`
- Defined: `scientist_sinergy_ablation_plan.py:68`
- Doc: Verifica si es seguro ejecutar con Nested Learning

### setup_matplotlib_for_plotting `def setup_matplotlib_for_plotting()`
- Defined: `scientist_sinergy_ablation_plan.py:85`
- Doc: Setup matplotlib para visualizaciones científicas

### generate_sinergy_matrix `def generate_sinergy_matrix()`
- Defined: `scientist_sinergy_ablation_plan.py:94`
- Doc: Genera matriz de sinergias basada en tus inventos

### print_sinergy_analysis `def print_sinergy_analysis()`
- Defined: `scientist_sinergy_ablation_plan.py:139`
- Doc: Analiza las sinergias propuestas basado en tus modelos

### main `def main()`
- Defined: `scientist_sinergy_ablation_plan.py:171`

## setup_environment.py

### check_python_version `def check_python_version()`
- Defined: `setup_environment.py:15`
- Doc: Verifica la versión de Python

### install_package `def install_package(package)`
- Defined: `setup_environment.py:23`
- Doc: Instala un paquete usando pip

### check_and_install_dependencies `def check_and_install_dependencies()`
- Defined: `setup_environment.py:32`
- Doc: Verifica e instala dependencias

### create_directories `def create_directories()`
- Defined: `setup_environment.py:76`
- Doc: Crea directorios necesarios

### setup_matplotlib `def setup_matplotlib()`
- Defined: `setup_environment.py:92`
- Doc: Configura matplotlib para el entorno

### create_sample_data `def create_sample_data()`
- Defined: `setup_environment.py:118`
- Doc: Crea datos de muestra para pruebas

### test_installation `def test_installation()`
- Defined: `setup_environment.py:148`
- Doc: Prueba la instalación

### create_main_script `def create_main_script()`
- Defined: `setup_environment.py:189`
- Doc: Crea script principal para ejecutar experimentos

### main `def main()`
- Defined: `setup_environment.py:256`
- Doc: Función principal de setup

## sintesis.py

### run_prism_dream `def run_prism_dream()`
- Defined: `sintesis.py:162`

### __init__ `def __init__(self, target_entropy)`
- Defined: `sintesis.py:13`

### calc_structural_health `def calc_structural_health(self, weight_matrix)`
- Defined: `sintesis.py:16`

### measure_spatial_richness `def measure_spatial_richness(self, activations)`
- Defined: `sintesis.py:33`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `sintesis.py:49`

### forward `def forward(self, x)`
- Defined: `sintesis.py:58`

### prismatic_dream `def prismatic_dream(self)`
- Defined: `sintesis.py:81`
- Doc: Sueño Entrópico:

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `sintesis.py:118`

### forward `def forward(self, x)`
- Defined: `sintesis.py:128`

### calculate_losses `def calculate_losses(self, outputs, targets, criterion)`
- Defined: `sintesis.py:134`

### sleep `def sleep(self)`
- Defined: `sintesis.py:154`

## sintesys2.py

### run_the_prisms_eye `def run_the_prisms_eye()`
- Defined: `sintesys2.py:174`

### __init__ `def __init__(self, target_entropy)`
- Defined: `sintesys2.py:13`

### calc_structural_health `def calc_structural_health(self, weight_matrix)`
- Defined: `sintesys2.py:16`

### measure_spatial_richness `def measure_spatial_richness(self, activations)`
- Defined: `sintesys2.py:32`
- Doc: Ahora retorna el tensor (para el gradiente) y el valor escalar.

### __init__ `def __init__(self, input_dim)`
- Defined: `sintesys2.py:53`

### forward `def forward(self, x)`
- Defined: `sintesys2.py:60`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `sintesys2.py:70`

### forward `def forward(self, x)`
- Defined: `sintesys2.py:79`

### prismatic_dream `def prismatic_dream(self)`
- Defined: `sintesys2.py:95`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `sintesys2.py:115`

### forward `def forward(self, x)`
- Defined: `sintesys2.py:131`

### calculate_losses `def calculate_losses(self, outputs, targets, criterion)`
- Defined: `sintesys2.py:144`

### sleep `def sleep(self)`
- Defined: `sintesys2.py:166`

## sintesys3.py

### measure_spatial_richness `def measure_spatial_richness(activations)`
- Defined: `sintesys3.py:12`
- Doc: Retorna tensor (gradiente) y valor escalar

### run_liquid_synthesis `def run_liquid_synthesis()`
- Defined: `sintesys3.py:156`

### __init__ `def __init__(self)`
- Defined: `sintesys3.py:31`

### decide `def decide(self, task_loss_val, richness_val, vn_entropy_val, target_entropy)`
- Defined: `sintesys3.py:35`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `sintesys3.py:60`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `sintesys3.py:68`

### consolidate_svd `def consolidate_svd(self, repair_strength)`
- Defined: `sintesys3.py:86`
- Doc: Sueño a demanda, intensidad variable

### __init__ `def __init__(self, d_in, d_hid, d_out)`
- Defined: `sintesys3.py:110`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `sintesys3.py:126`

### get_structure_entropy `def get_structure_entropy(self)`
- Defined: `sintesys3.py:140`

### calc_ent `def calc_ent(W)`
- Defined: `sintesys3.py:143`

## sintesys5.py

### measure_spatial_richness `def measure_spatial_richness(activations)`
- Defined: `sintesys5.py:56`

### run_real_world_challenge `def run_real_world_challenge()`
- Defined: `sintesys5.py:157`

### __init__ `def __init__(self)`
- Defined: `sintesys5.py:20`

### get_batch `def get_batch(self, phase, batch_size)`
- Defined: `sintesys5.py:38`

### __init__ `def __init__(self)`
- Defined: `sintesys5.py:69`

### decide `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- Defined: `sintesys5.py:73`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `sintesys5.py:88`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `sintesys5.py:96`

### consolidate_svd `def consolidate_svd(self, repair_strength)`
- Defined: `sintesys5.py:111`

### __init__ `def __init__(self, d_in, d_hid, d_out)`
- Defined: `sintesys5.py:125`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `sintesys5.py:135`

### get_structure_entropy `def get_structure_entropy(self)`
- Defined: `sintesys5.py:143`

### calc_ent `def calc_ent(W)`
- Defined: `sintesys5.py:145`

## syntesys4.py

### measure_spatial_richness `def measure_spatial_richness(activations)`
- Defined: `syntesys4.py:12`

### run_sensitive_self `def run_sensitive_self()`
- Defined: `syntesys4.py:131`

### __init__ `def __init__(self)`
- Defined: `syntesys4.py:28`

### decide `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- Defined: `syntesys4.py:32`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `syntesys4.py:61`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `syntesys4.py:69`

### consolidate_svd `def consolidate_svd(self, repair_strength)`
- Defined: `syntesys4.py:84`

### __init__ `def __init__(self, d_in, d_hid, d_out)`
- Defined: `syntesys4.py:102`

### forward `def forward(self, x, plasticity_gate)`
- Defined: `syntesys4.py:112`

### get_structure_entropy `def get_structure_entropy(self)`
- Defined: `syntesys4.py:120`

### calc_ent `def calc_ent(W)`
- Defined: `syntesys4.py:122`

## test.py

### measure_metrics `def measure_metrics(model)`
- Defined: `test.py:80`

### calculate_test_accuracy `def calculate_test_accuracy(model, testloader)`
- Defined: `test.py:109`

### __init__ `def __init__(self)`
- Defined: `test.py:53`

### forward `def forward(self, x)`
- Defined: `test.py:69`

## test_premium_synergy.py

### test_individual_components `def test_individual_components()`
- Defined: `test_premium_synergy.py:29`
- Doc: Test de componentes individuales
- Depends on: `premium_synergy_democratic.py`

### test_full_system `def test_full_system()`
- Defined: `test_premium_synergy.py:91`
- Doc: Test del sistema completo Premium Synergy
- Depends on: `premium_synergy_democratic.py`

### test_training_loop `def test_training_loop()`
- Defined: `test_premium_synergy.py:164`
- Doc: Test del loop de entrenamiento completo
- Depends on: `premium_synergy_democratic.py`

### run_all_tests `def run_all_tests()`
- Defined: `test_premium_synergy.py:209`
- Doc: Ejecuta todos los tests
- Depends on: `premium_synergy_democratic.py`

## topobrain.py

### seed_everything `def seed_everything(seed)`
- Defined: `topobrain.py:50`

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `topobrain.py:87`
- Doc: ✅ FIX: Guarda solo estado esencial + compresión

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `topobrain.py:118`

### clamp_pgd `def clamp_pgd(x_adv_norm, x_orig_norm, eps)`
- Defined: `topobrain.py:352`

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps)`
- Defined: `topobrain.py:359`

### eval_autoattack `def eval_autoattack(model, test_loader, n_samples)`
- Defined: `topobrain.py:391`

### save_topology_snapshot `def save_topology_snapshot(model, epoch, run_name)`
- Defined: `topobrain.py:423`

### plot_topology_evolution `def plot_topology_evolution(run_name)`
- Defined: `topobrain.py:449`

### run_training `def run_training(config_override, run_name)`
- Defined: `topobrain.py:476`

### run_diagnostic_suite `def run_diagnostic_suite()`
- Defined: `topobrain.py:676`
- Doc: ✅ Suite completa con TopoOnly crítico

### get_memory_gb `def get_memory_gb()`
- Defined: `topobrain.py:60`

### check_memory_limit `def check_memory_limit(limit_gb)`
- Defined: `topobrain.py:65`

### log_resources `def log_resources()`
- Defined: `topobrain.py:72`
- Doc: ✅ FIX: Método faltante añadido

### clear_cache `def clear_cache()`
- Defined: `topobrain.py:81`

### __init__ `def __init__(self, dim)`
- Defined: `topobrain.py:136`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `topobrain.py:143`

### __init__ `def __init__(self, temperature)`
- Defined: `topobrain.py:148`

### forward `def forward(self, features, labels)`
- Defined: `topobrain.py:152`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `topobrain.py:176`

### forward `def forward(self, input_signal, prediction)`
- Defined: `topobrain.py:182`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `topobrain.py:188`

### forward `def forward(self, x)`
- Defined: `topobrain.py:196`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- Defined: `topobrain.py:206`

### forward `def forward(self, x_nodes, adjacency, incidence, global_step)`
- Defined: `topobrain.py:229`

### __init__ `def __init__(self, config)`
- Defined: `topobrain.py:253`

### _init_grid `def _init_grid(self, N)`
- Defined: `topobrain.py:287`

### get_topology `def get_topology(self)`
- Defined: `topobrain.py:308`

### forward `def forward(self, x)`
- Defined: `topobrain.py:324`

### __init__ `def __init__(self, m)`
- Defined: `topobrain.py:410`

### forward `def forward(self, x)`
- Defined: `topobrain.py:411`

### lambda_topo `def lambda_topo(epoch)`
- Defined: `topobrain.py:529`

## topobrain_16_3.py

### seed_everything `def seed_everything(seed)`
- Defined: `topobrain_16_3.py:61`

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `topobrain_16_3.py:98`

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `topobrain_16_3.py:123`

### clamp_pgd `def clamp_pgd(x_adv_norm, x_orig_norm, eps)`
- Defined: `topobrain_16_3.py:361`

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps)`
- Defined: `topobrain_16_3.py:368`

### eval_autoattack `def eval_autoattack(model, test_loader, n_samples)`
- Defined: `topobrain_16_3.py:393`

### save_topology_snapshot `def save_topology_snapshot(model, epoch, run_name)`
- Defined: `topobrain_16_3.py:423`

### plot_topology_evolution `def plot_topology_evolution(run_name)`
- Defined: `topobrain_16_3.py:447`

### run_training `def run_training(config_override, run_name)`
- Defined: `topobrain_16_3.py:464`

### run_diagnostic_suite `def run_diagnostic_suite()`
- Defined: `topobrain_16_3.py:726`

### get_memory_gb `def get_memory_gb()`
- Defined: `topobrain_16_3.py:72`

### check_memory_limit `def check_memory_limit(limit_gb)`
- Defined: `topobrain_16_3.py:77`

### log_resources `def log_resources()`
- Defined: `topobrain_16_3.py:84`

### clear_cache `def clear_cache()`
- Defined: `topobrain_16_3.py:92`

### __init__ `def __init__(self, dim)`
- Defined: `topobrain_16_3.py:141`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `topobrain_16_3.py:148`

### __init__ `def __init__(self, temperature)`
- Defined: `topobrain_16_3.py:153`

### forward `def forward(self, features, labels)`
- Defined: `topobrain_16_3.py:157`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `topobrain_16_3.py:184`

### forward `def forward(self, input_signal, prediction)`
- Defined: `topobrain_16_3.py:190`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `topobrain_16_3.py:196`

### forward `def forward(self, x)`
- Defined: `topobrain_16_3.py:204`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- Defined: `topobrain_16_3.py:214`

### forward `def forward(self, x_nodes, adjacency, incidence, global_step)`
- Defined: `topobrain_16_3.py:236`

### __init__ `def __init__(self, config)`
- Defined: `topobrain_16_3.py:267`

### _init_grid `def _init_grid(self, N)`
- Defined: `topobrain_16_3.py:299`

### get_topology `def get_topology(self)`
- Defined: `topobrain_16_3.py:319`

### forward `def forward(self, x)`
- Defined: `topobrain_16_3.py:333`

### __init__ `def __init__(self, m)`
- Defined: `topobrain_16_3.py:410`

### forward `def forward(self, x)`
- Defined: `topobrain_16_3.py:411`

### lambda_topo `def lambda_topo(epoch)`
- Defined: `topobrain_16_3.py:531`

## topobrain_v18.1.py

### seed_everything `def seed_everything(seed)`
- Defined: `topobrain_v18.1.py:100`

### get_dataset_stats `def get_dataset_stats(dataset_name)`
- Defined: `topobrain_v18.1.py:309`

### get_dataloaders `def get_dataloaders(config)`
- Defined: `topobrain_v18.1.py:316`

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)`
- Defined: `topobrain_v18.1.py:820`
- Doc: PGD Attack

### train_epoch `def train_epoch(model, train_loader, optimizer, opt_topo, supcon, config, epoch, topo_monitor)`
- Defined: `topobrain_v18.1.py:848`
- Doc: Entrena una época con schedule adaptativo de SupCon - CORREGIDO

### train_model `def train_model(config, run_name)`
- Defined: `topobrain_v18.1.py:956`
- Doc: Loop de entrenamiento v18

### evaluate `def evaluate(model, test_loader, config, adversarial)`
- Defined: `topobrain_v18.1.py:1139`
- Doc: Evalúa el modelo

### save_topology_visualization `def save_topology_visualization(model, epoch, run_name)`
- Defined: `topobrain_v18.1.py:1166`
- Doc: Guarda visualización de la topología aprendida

### save_node_importance_viz `def save_node_importance_viz(model, epoch, run_name)`
- Defined: `topobrain_v18.1.py:1211`
- Doc: Visualiza importancia de nodos por capa

### analyze_topology_clustering `def analyze_topology_clustering(model, run_name)`
- Defined: `topobrain_v18.1.py:1234`
- Doc: Clustering espectral de nodos basado en conectividad

### analyze_topology_flow `def analyze_topology_flow(model, dataloader, run_name, num_samples)`
- Defined: `topobrain_v18.1.py:1277`
- Doc: Analiza flujo de información en la topología

### visualize_topology_as_graph `def visualize_topology_as_graph(model, run_name, threshold)`
- Defined: `topobrain_v18.1.py:1343`
- Doc: Visualiza topología como grafo con NetworkX

### analyze_topology_evolution `def analyze_topology_evolution(run_name)`
- Defined: `topobrain_v18.1.py:1409`
- Doc: Analiza la evolución de la topología a lo largo del entrenamiento

### comprehensive_topology_analysis `def comprehensive_topology_analysis(model, dataloader, run_name)`
- Defined: `topobrain_v18.1.py:1478`
- Doc: Análisis completo de topología

### run_ablation_study `def run_ablation_study()`
- Defined: `topobrain_v18.1.py:1509`
- Doc: Ejecuta suite completa de ablación v18

### main `def main()`
- Defined: `topobrain_v18.1.py:1633`
- Doc: Punto de entrada principal v18

### __post_init__ `def __post_init__(self)`
- Defined: `topobrain_v18.1.py:82`

### to_dict `def to_dict(self)`
- Defined: `topobrain_v18.1.py:86`

### get_supcon_lambda `def get_supcon_lambda(self, epoch)`
- Defined: `topobrain_v18.1.py:89`
- Doc: Schedule adaptativo para SupCon Loss

### get_memory_gb `def get_memory_gb()`
- Defined: `topobrain_v18.1.py:111`

### get_gpu_memory_gb `def get_gpu_memory_gb()`
- Defined: `topobrain_v18.1.py:116`

### log `def log(prefix)`
- Defined: `topobrain_v18.1.py:122`

### clear_cache `def clear_cache()`
- Defined: `topobrain_v18.1.py:129`

### check_limit `def check_limit(limit_gb)`
- Defined: `topobrain_v18.1.py:135`

### __init__ `def __init__(self, model, config, epsilon_c)`
- Defined: `topobrain_v18.1.py:159`

### _analyze_matrix `def _analyze_matrix(self, weight_matrix, name)`
- Defined: `topobrain_v18.1.py:165`
- Doc: Análisis SVD de matriz topológica (adj o inc)

### calculate `def calculate(self, epoch)`
- Defined: `topobrain_v18.1.py:220`
- Doc: Analiza todas las matrices topológicas del modelo

### get_critical_summary `def get_critical_summary(self)`
- Defined: `topobrain_v18.1.py:247`
- Doc: Resumen de emergencias

### __init__ `def __init__(self, checkpoint_dir)`
- Defined: `topobrain_v18.1.py:261`

### save `def save(self, data, name)`
- Defined: `topobrain_v18.1.py:265`

### load `def load(self, name)`
- Defined: `topobrain_v18.1.py:288`

### __init__ `def __init__(self, temperature)`
- Defined: `topobrain_v18.1.py:368`

### forward `def forward(self, features, labels)`
- Defined: `topobrain_v18.1.py:372`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `topobrain_v18.1.py:403`

### forward `def forward(self, input_signal, prediction)`
- Defined: `topobrain_v18.1.py:415`

### __init__ `def __init__(self, dim)`
- Defined: `topobrain_v18.1.py:439`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `topobrain_v18.1.py:448`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `topobrain_v18.1.py:455`

### forward `def forward(self, x)`
- Defined: `topobrain_v18.1.py:463`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- Defined: `topobrain_v18.1.py:480`

### forward `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse)`
- Defined: `topobrain_v18.1.py:509`
- Doc: Args:

### get_node_importance `def get_node_importance(self)`
- Defined: `topobrain_v18.1.py:588`
- Doc: Retorna importancia de nodos para visualización

### __init__ `def __init__(self, config, in_channels)`
- Defined: `topobrain_v18.1.py:606`

### _init_grid_topology `def _init_grid_topology(self, N)`
- Defined: `topobrain_v18.1.py:653`
- Doc: Inicializa topología de grid 2D

### get_topology `def get_topology(self, return_sparse)`
- Defined: `topobrain_v18.1.py:681`
- Doc: Calcula topología actual

### calculate_ortho_loss `def calculate_ortho_loss(self)`
- Defined: `topobrain_v18.1.py:708`
- Doc: Regularización ortogonal con pesos por capa

### prune_topology `def prune_topology(self)`
- Defined: `topobrain_v18.1.py:741`
- Doc: Poda de topología basada en importancia

### forward `def forward(self, x)`
- Defined: `topobrain_v18.1.py:787`

### set_epoch `def set_epoch(self, epoch)`
- Defined: `topobrain_v18.1.py:813`
- Doc: Permite pasar la época actual para schedules dinámicos

### warmup_topo `def warmup_topo(epoch)`
- Defined: `topobrain_v18.1.py:1006`

## topobrain_v18.2.py

### seed_everything `def seed_everything(seed)`
- Defined: `topobrain_v18.2.py:100`

### get_dataset_stats `def get_dataset_stats(dataset_name)`
- Defined: `topobrain_v18.2.py:309`

### get_dataloaders `def get_dataloaders(config)`
- Defined: `topobrain_v18.2.py:316`

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)`
- Defined: `topobrain_v18.2.py:824`
- Doc: PGD Attack

### train_epoch `def train_epoch(model, train_loader, optimizer, opt_topo, supcon, config, epoch, topo_monitor)`
- Defined: `topobrain_v18.2.py:852`
- Doc: Entrena una época con schedule adaptativo de SupCon - CORREGIDO

### train_model `def train_model(config, run_name)`
- Defined: `topobrain_v18.2.py:960`
- Doc: Loop de entrenamiento v18 (CORREGIDO - Inicialización Negativa)

### evaluate `def evaluate(model, test_loader, config, adversarial)`
- Defined: `topobrain_v18.2.py:1088`
- Doc: Evalúa el modelo

### save_topology_visualization `def save_topology_visualization(model, epoch, run_name)`
- Defined: `topobrain_v18.2.py:1115`
- Doc: Guarda visualización de la topología aprendida

### save_node_importance_viz `def save_node_importance_viz(model, epoch, run_name)`
- Defined: `topobrain_v18.2.py:1160`
- Doc: Visualiza importancia de nodos por capa

### analyze_topology_clustering `def analyze_topology_clustering(model, run_name)`
- Defined: `topobrain_v18.2.py:1183`
- Doc: Clustering espectral de nodos basado en conectividad

### analyze_topology_flow `def analyze_topology_flow(model, dataloader, run_name, num_samples)`
- Defined: `topobrain_v18.2.py:1226`
- Doc: Analiza flujo de información en la topología

### visualize_topology_as_graph `def visualize_topology_as_graph(model, run_name, threshold)`
- Defined: `topobrain_v18.2.py:1292`
- Doc: Visualiza topología como grafo con NetworkX

### analyze_topology_evolution `def analyze_topology_evolution(run_name)`
- Defined: `topobrain_v18.2.py:1358`
- Doc: Analiza la evolución de la topología a lo largo del entrenamiento

### comprehensive_topology_analysis `def comprehensive_topology_analysis(model, dataloader, run_name)`
- Defined: `topobrain_v18.2.py:1427`
- Doc: Análisis completo de topología

### run_ablation_study `def run_ablation_study()`
- Defined: `topobrain_v18.2.py:1458`
- Doc: Ejecuta suite completa de ablación v18

### main `def main()`
- Defined: `topobrain_v18.2.py:1582`
- Doc: Punto de entrada principal v18

### __post_init__ `def __post_init__(self)`
- Defined: `topobrain_v18.2.py:82`

### to_dict `def to_dict(self)`
- Defined: `topobrain_v18.2.py:86`

### get_supcon_lambda `def get_supcon_lambda(self, epoch)`
- Defined: `topobrain_v18.2.py:89`
- Doc: Schedule adaptativo para SupCon Loss

### get_memory_gb `def get_memory_gb()`
- Defined: `topobrain_v18.2.py:111`

### get_gpu_memory_gb `def get_gpu_memory_gb()`
- Defined: `topobrain_v18.2.py:116`

### log `def log(prefix)`
- Defined: `topobrain_v18.2.py:122`

### clear_cache `def clear_cache()`
- Defined: `topobrain_v18.2.py:129`

### check_limit `def check_limit(limit_gb)`
- Defined: `topobrain_v18.2.py:135`

### __init__ `def __init__(self, model, config, epsilon_c)`
- Defined: `topobrain_v18.2.py:159`

### _analyze_matrix `def _analyze_matrix(self, weight_matrix, name)`
- Defined: `topobrain_v18.2.py:165`
- Doc: Análisis SVD de matriz topológica (CORREGIDO)

### calculate `def calculate(self, epoch)`
- Defined: `topobrain_v18.2.py:220`
- Doc: Analiza todas las matrices topológicas del modelo

### get_critical_summary `def get_critical_summary(self)`
- Defined: `topobrain_v18.2.py:247`
- Doc: Resumen de emergencias

### __init__ `def __init__(self, checkpoint_dir)`
- Defined: `topobrain_v18.2.py:261`

### save `def save(self, data, name)`
- Defined: `topobrain_v18.2.py:265`

### load `def load(self, name)`
- Defined: `topobrain_v18.2.py:288`

### __init__ `def __init__(self, temperature)`
- Defined: `topobrain_v18.2.py:368`

### forward `def forward(self, features, labels)`
- Defined: `topobrain_v18.2.py:372`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `topobrain_v18.2.py:403`

### forward `def forward(self, input_signal, prediction)`
- Defined: `topobrain_v18.2.py:415`

### __init__ `def __init__(self, dim)`
- Defined: `topobrain_v18.2.py:439`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `topobrain_v18.2.py:448`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `topobrain_v18.2.py:455`

### forward `def forward(self, x)`
- Defined: `topobrain_v18.2.py:463`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- Defined: `topobrain_v18.2.py:480`

### forward `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse)`
- Defined: `topobrain_v18.2.py:510`
- Doc: Args:

### get_node_importance `def get_node_importance(self)`
- Defined: `topobrain_v18.2.py:589`
- Doc: Retorna importancia de nodos para visualización

### __init__ `def __init__(self, config, in_channels)`
- Defined: `topobrain_v18.2.py:608`

### _init_grid_topology `def _init_grid_topology(self, N)`
- Defined: `topobrain_v18.2.py:668`
- Doc: Inicializa topología de grid 2D

### get_topology `def get_topology(self, return_sparse)`
- Defined: `topobrain_v18.2.py:705`
- Doc: Calcula topología actual

### calculate_ortho_loss `def calculate_ortho_loss(self)`
- Defined: `topobrain_v18.2.py:729`
- Doc: Regularización ortogonal con pesos por capa

### prune_topology `def prune_topology(self)`
- Defined: `topobrain_v18.2.py:754`
- Doc: Poda de topología basada en importancia

### forward `def forward(self, x)`
- Defined: `topobrain_v18.2.py:793`

### set_epoch `def set_epoch(self, epoch)`
- Defined: `topobrain_v18.2.py:818`

### warmup_topo `def warmup_topo(epoch)`
- Defined: `topobrain_v18.2.py:1016`

## topobrain_v18.py

### seed_everything `def seed_everything(seed)`
- Defined: `topobrain_v18.py:100`

### get_dataset_stats `def get_dataset_stats(dataset_name)`
- Defined: `topobrain_v18.py:309`

### get_dataloaders `def get_dataloaders(config)`
- Defined: `topobrain_v18.py:316`

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)`
- Defined: `topobrain_v18.py:799`
- Doc: PGD Attack

### train_epoch `def train_epoch(model, train_loader, optimizer, opt_topo, supcon, config, epoch, topo_monitor)`
- Defined: `topobrain_v18.py:827`
- Doc: Entrena una época con schedule adaptativo de SupCon

### train_model `def train_model(config, run_name)`
- Defined: `topobrain_v18.py:917`
- Doc: Loop de entrenamiento completo

### evaluate `def evaluate(model, test_loader, config, adversarial)`
- Defined: `topobrain_v18.py:1078`
- Doc: Evalúa el modelo

### train_model `def train_model(config, run_name)`
- Defined: `topobrain_v18.py:1101`
- Doc: Loop de entrenamiento completo

### save_topology_visualization `def save_topology_visualization(model, epoch, run_name)`
- Defined: `topobrain_v18.py:1257`
- Doc: Guarda visualización de la topología aprendida

### save_node_importance_viz `def save_node_importance_viz(model, epoch, run_name)`
- Defined: `topobrain_v18.py:1302`
- Doc: Visualiza importancia de nodos por capa

### analyze_topology_clustering `def analyze_topology_clustering(model, run_name)`
- Defined: `topobrain_v18.py:1325`
- Doc: Clustering espectral de nodos basado en conectividad

### analyze_topology_flow `def analyze_topology_flow(model, dataloader, run_name, num_samples)`
- Defined: `topobrain_v18.py:1368`
- Doc: Analiza flujo de información en la topología

### visualize_topology_as_graph `def visualize_topology_as_graph(model, run_name, threshold)`
- Defined: `topobrain_v18.py:1434`
- Doc: Visualiza topología como grafo con NetworkX

### analyze_topology_evolution `def analyze_topology_evolution(run_name)`
- Defined: `topobrain_v18.py:1500`
- Doc: Analiza la evolución de la topología a lo largo del entrenamiento

### comprehensive_topology_analysis `def comprehensive_topology_analysis(model, dataloader, run_name)`
- Defined: `topobrain_v18.py:1569`
- Doc: Análisis completo de topología

### run_ablation_study `def run_ablation_study()`
- Defined: `topobrain_v18.py:1600`
- Doc: Ejecuta suite completa de ablación v18

### main `def main()`
- Defined: `topobrain_v18.py:1724`
- Doc: Punto de entrada principal v18

### __post_init__ `def __post_init__(self)`
- Defined: `topobrain_v18.py:82`

### to_dict `def to_dict(self)`
- Defined: `topobrain_v18.py:86`

### get_supcon_lambda `def get_supcon_lambda(self, epoch)`
- Defined: `topobrain_v18.py:89`
- Doc: Schedule adaptativo para SupCon Loss

### get_memory_gb `def get_memory_gb()`
- Defined: `topobrain_v18.py:111`

### get_gpu_memory_gb `def get_gpu_memory_gb()`
- Defined: `topobrain_v18.py:116`

### log `def log(prefix)`
- Defined: `topobrain_v18.py:122`

### clear_cache `def clear_cache()`
- Defined: `topobrain_v18.py:129`

### check_limit `def check_limit(limit_gb)`
- Defined: `topobrain_v18.py:135`

### __init__ `def __init__(self, model, config, epsilon_c)`
- Defined: `topobrain_v18.py:159`

### _analyze_matrix `def _analyze_matrix(self, weight_matrix, name)`
- Defined: `topobrain_v18.py:165`
- Doc: Análisis SVD de matriz topológica (adj o inc)

### calculate `def calculate(self, epoch)`
- Defined: `topobrain_v18.py:220`
- Doc: Analiza todas las matrices topológicas del modelo

### get_critical_summary `def get_critical_summary(self)`
- Defined: `topobrain_v18.py:247`
- Doc: Resumen de emergencias

### __init__ `def __init__(self, checkpoint_dir)`
- Defined: `topobrain_v18.py:261`

### save `def save(self, data, name)`
- Defined: `topobrain_v18.py:265`

### load `def load(self, name)`
- Defined: `topobrain_v18.py:288`

### __init__ `def __init__(self, temperature)`
- Defined: `topobrain_v18.py:368`

### forward `def forward(self, features, labels)`
- Defined: `topobrain_v18.py:372`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `topobrain_v18.py:403`

### forward `def forward(self, input_signal, prediction)`
- Defined: `topobrain_v18.py:415`

### __init__ `def __init__(self, dim)`
- Defined: `topobrain_v18.py:439`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `topobrain_v18.py:448`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `topobrain_v18.py:455`

### forward `def forward(self, x)`
- Defined: `topobrain_v18.py:463`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- Defined: `topobrain_v18.py:480`

### forward `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse)`
- Defined: `topobrain_v18.py:509`
- Doc: Args:

### get_node_importance `def get_node_importance(self)`
- Defined: `topobrain_v18.py:588`
- Doc: Retorna importancia de nodos para visualización

### __init__ `def __init__(self, config, in_channels)`
- Defined: `topobrain_v18.py:606`

### _init_grid_topology `def _init_grid_topology(self, N)`
- Defined: `topobrain_v18.py:653`
- Doc: Inicializa topología de grid 2D

### get_topology `def get_topology(self, return_sparse)`
- Defined: `topobrain_v18.py:681`
- Doc: Calcula topología actual

### calculate_ortho_loss `def calculate_ortho_loss(self)`
- Defined: `topobrain_v18.py:708`
- Doc: Regularización ortogonal con pesos por capa

### prune_topology `def prune_topology(self)`
- Defined: `topobrain_v18.py:741`
- Doc: Poda de topología basada en importancia

### forward `def forward(self, x)`
- Defined: `topobrain_v18.py:769`

### lambda_topo `def lambda_topo(epoch)`
- Defined: `topobrain_v18.py:948`

### lambda_topo `def lambda_topo(epoch)`
- Defined: `topobrain_v18.py:1130`

## topobrain_v19.py

### seed_everything `def seed_everything(seed)`
- Defined: `topobrain_v19.py:98`

### get_dataset_stats `def get_dataset_stats(dataset_name)`
- Defined: `topobrain_v19.py:182`

### get_dataloaders `def get_dataloaders(config)`
- Defined: `topobrain_v19.py:189`

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)`
- Defined: `topobrain_v19.py:493`
- Doc: PGD Attack simplificado y robusto

### train_epoch `def train_epoch(model, train_loader, optimizer, criterion, contrastive_loss, config, epoch)`
- Defined: `topobrain_v19.py:545`
- Doc: Entrena una época con logging integrado

### evaluate `def evaluate(model, test_loader, config, adversarial)`
- Defined: `topobrain_v19.py:621`
- Doc: Evalúa el modelo

### train_model `def train_model(config, run_name)`
- Defined: `topobrain_v19.py:644`
- Doc: Loop de entrenamiento completo v19

### save_topology_snapshot `def save_topology_snapshot(model, epoch, run_name)`
- Defined: `topobrain_v19.py:796`
- Doc: Guarda snapshot de topología

### run_ablation_study `def run_ablation_study()`
- Defined: `topobrain_v19.py:837`
- Doc: Suite de ablación sistemática v19

### main `def main()`
- Defined: `topobrain_v19.py:918`

### to_dict `def to_dict(self)`
- Defined: `topobrain_v19.py:91`

### get_memory_gb `def get_memory_gb()`
- Defined: `topobrain_v19.py:109`

### get_gpu_memory_gb `def get_gpu_memory_gb()`
- Defined: `topobrain_v19.py:114`

### log `def log(prefix)`
- Defined: `topobrain_v19.py:120`

### clear_cache `def clear_cache()`
- Defined: `topobrain_v19.py:128`

### check_limit `def check_limit(limit_gb)`
- Defined: `topobrain_v19.py:134`

### __init__ `def __init__(self, checkpoint_dir)`
- Defined: `topobrain_v19.py:140`

### save `def save(self, data, name)`
- Defined: `topobrain_v19.py:144`

### load `def load(self, name)`
- Defined: `topobrain_v19.py:163`

### __init__ `def __init__(self, num_nodes, node_dim, k)`
- Defined: `topobrain_v19.py:245`

### forward `def forward(self, batch_size)`
- Defined: `topobrain_v19.py:254`
- Doc: Retorna edges para k-NN dinámico

### __init__ `def __init__(self, in_dim, hid_dim, config, layer_idx)`
- Defined: `topobrain_v19.py:296`

### forward `def forward(self, x, edge_index, edge_weight, batch)`
- Defined: `topobrain_v19.py:325`
- Doc: Args:

### __init__ `def __init__(self, config, in_channels)`
- Defined: `topobrain_v19.py:364`

### forward `def forward(self, x)`
- Defined: `topobrain_v19.py:398`

### apply_pruning `def apply_pruning(self, edge_index, edge_weight)`
- Defined: `topobrain_v19.py:445`
- Doc: Aplica máscara de pruning

### prune_structural `def prune_structural(self, threshold)`
- Defined: `topobrain_v19.py:453`
- Doc: Pruning estructural real: elimina edges permanentemente

### calculate_ortho_loss `def calculate_ortho_loss(self)`
- Defined: `topobrain_v19.py:473`
- Doc: Regularización ortogonal simple

### __init__ `def __init__(self, temperature)`
- Defined: `topobrain_v19.py:523`

### forward `def forward(self, features, labels)`
- Defined: `topobrain_v19.py:527`

### warmup_lr `def warmup_lr(epoch)`
- Defined: `topobrain_v19.py:668`

### prune_fn `def prune_fn(edge_idx)`
- Defined: `topobrain_v19.py:467`

## train_Adversarial.py

### seed_everything `def seed_everything(seed)`
- Defined: `train_Adversarial.py:61`

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `train_Adversarial.py:100`
- Doc: Sistema de checkpoint robusto con protección contra corrupción

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `train_Adversarial.py:134`
- Doc: Carga checkpoint con fallback automático

### clamp_pgd `def clamp_pgd(x_adv_norm, x_orig_norm, eps)`
- Defined: `train_Adversarial.py:509`

### make_adversarial_pgd `def make_adversarial_pgd(model, x, y, eps, steps)`
- Defined: `train_Adversarial.py:516`

### eval_autoattack `def eval_autoattack(model, test_loader, n_samples)`
- Defined: `train_Adversarial.py:534`

### create_checkpoint_data `def create_checkpoint_data(model, optimizer, epoch, config, metrics)`
- Defined: `train_Adversarial.py:566`
- Doc: Crea estructura de checkpoint completa

### run_training `def run_training(config_override, run_name)`
- Defined: `train_Adversarial.py:586`

### save_topology_snapshot `def save_topology_snapshot(model, epoch, run_name)`
- Defined: `train_Adversarial.py:770`

### run_diagnostic_suite `def run_diagnostic_suite()`
- Defined: `train_Adversarial.py:787`

### get_memory_gb `def get_memory_gb()`
- Defined: `train_Adversarial.py:78`

### log_resources `def log_resources()`
- Defined: `train_Adversarial.py:83`

### clear_cache `def clear_cache()`
- Defined: `train_Adversarial.py:90`
- Doc: Limpia cachés y fuerza garbage collection

### __init__ `def __init__(self, params, lr, momentum, nested_levels, freq_factor)`
- Defined: `train_Adversarial.py:159`

### step `def step(self, closure)`
- Defined: `train_Adversarial.py:179`

### __init__ `def __init__(self, input_dim, hidden_dim, num_levels)`
- Defined: `train_Adversarial.py:223`

### forward `def forward(self, x)`
- Defined: `train_Adversarial.py:240`

### should_update_level `def should_update_level(self, level_idx, global_step)`
- Defined: `train_Adversarial.py:246`
- Doc: Determina si un nivel debe actualizarse basado en su frecuencia

### get_update_mask `def get_update_mask(self, level_idx, batch_size)`
- Defined: `train_Adversarial.py:250`
- Doc: Máscara para actualizar solo un subconjunto de parámetros

### __init__ `def __init__(self, temperature)`
- Defined: `train_Adversarial.py:261`

### forward `def forward(self, features, labels)`
- Defined: `train_Adversarial.py:265`

### __init__ `def __init__(self, dim, use_spectral)`
- Defined: `train_Adversarial.py:289`

### forward `def forward(self, input_signal, prediction)`
- Defined: `train_Adversarial.py:295`

### __init__ `def __init__(self, dim)`
- Defined: `train_Adversarial.py:301`

### forward `def forward(self, x_sensory, x_prediction)`
- Defined: `train_Adversarial.py:310`

### __init__ `def __init__(self, dim, num_atoms)`
- Defined: `train_Adversarial.py:315`

### forward `def forward(self, x)`
- Defined: `train_Adversarial.py:323`

### __init__ `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- Defined: `train_Adversarial.py:333`

### forward `def forward(self, x_nodes, adjacency, incidence, global_step)`
- Defined: `train_Adversarial.py:362`

### apply_cms `def apply_cms(self, x, global_step)`
- Defined: `train_Adversarial.py:391`
- Doc: Aplica actualización condicional basada en frecuencia del CMS

### __init__ `def __init__(self, config)`
- Defined: `train_Adversarial.py:401`

### _init_grid `def _init_grid(self, N)`
- Defined: `train_Adversarial.py:444`

### get_topology `def get_topology(self)`
- Defined: `train_Adversarial.py:465`

### forward `def forward(self, x)`
- Defined: `train_Adversarial.py:477`

### __init__ `def __init__(self, m)`
- Defined: `train_Adversarial.py:553`

### forward `def forward(self, x)`
- Defined: `train_Adversarial.py:554`

### lambda_topo `def lambda_topo(epoch)`
- Defined: `train_Adversarial.py:632`

## tricameral2.py

### generate_audio_async `def generate_audio_async(text, output_path, voice, max_retries)`
- Defined: `tricameral2.py:31`
- Doc: Genera un audio usando Edge-TTS con retry logic

### generate_all_audios_batch `def generate_all_audios_batch(captions_list, audio_dir, batch_size)`
- Defined: `tricameral2.py:66`
- Doc: Genera todos los audios en batches pequeños con rate limiting

### generate_audios_sync `def generate_audios_sync(images_dir, captions_file, audio_dir)`
- Defined: `tricameral2.py:135`
- Doc: Wrapper síncrono para generar audios

### download_from_github `def download_from_github(repo_url, output_dir)`
- Defined: `tricameral2.py:183`
- Doc: Descarga dataset pre-preparado desde GitHub/Hugging Face

### setup_flickr8k `def setup_flickr8k(data_dir, github_url)`
- Defined: `tricameral2.py:257`
- Doc: Descarga y organiza Flickr8k - ahora con opción GitHub

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `tricameral2.py:366`
- Doc: Construye vocabulario desde el archivo de captions

### compute_tricameral_loss `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)`
- Defined: `tricameral2.py:1048`
- Doc: Pérdida con término de coherencia audio-visual

### train_tricameral `def train_tricameral(github_repo_url)`
- Defined: `tricameral2.py:1100`
- Doc: Entrena el modelo tricameral

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `tricameral2.py:390`

### forward `def forward(self, x)`
- Defined: `tricameral2.py:425`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `tricameral2.py:438`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `tricameral2.py:476`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `tricameral2.py:513`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `tricameral2.py:525`

### _greedy_decode `def _greedy_decode(self, visual_context, max_len, device)`
- Defined: `tricameral2.py:548`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `tricameral2.py:570`

### __init__ `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)`
- Defined: `tricameral2.py:587`

### __len__ `def __len__(self)`
- Defined: `tricameral2.py:625`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `tricameral2.py:628`

### __init__ `def __init__(self, output_dim)`
- Defined: `tricameral2.py:681`

### forward `def forward(self, mel_spec)`
- Defined: `tricameral2.py:720`
- Doc: Args:

### __init__ `def __init__(self, output_dim)`
- Defined: `tricameral2.py:750`

### forward `def forward(self, image, audio)`
- Defined: `tricameral2.py:784`
- Doc: Args:

### __init__ `def __init__(self, dim)`
- Defined: `tricameral2.py:824`

### forward `def forward(self, right_features)`
- Defined: `tricameral2.py:856`
- Doc: Args:

### __init__ `def __init__(self, text_dim, output_sr)`
- Defined: `tricameral2.py:916`

### forward `def forward(self, text_embedding)`
- Defined: `tricameral2.py:951`
- Doc: Args:

### __init__ `def __init__(self, vocab_size)`
- Defined: `tricameral2.py:985`

### forward `def forward(self, image, audio, captions, epoch, generate_audio)`
- Defined: `tricameral2.py:1000`
- Doc: Args:

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `tricameral2.py:1196`

### __len__ `def __len__(self)`
- Defined: `tricameral2.py:1214`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `tricameral2.py:1217`

## tricameral_kimi.py

### setup_flickr8k_with_audio `def setup_flickr8k_with_audio(data_dir)`
- Defined: `tricameral_kimi.py:36`
- Doc: Descarga y organiza Flickr8k + Audio del dataset de Kaggle.

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `tricameral_kimi.py:191`
- Doc: Construye vocabulario desde el archivo de captions

### compute_tricameral_loss `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)`
- Defined: `tricameral_kimi.py:1402`
- Doc: Pérdida con término de coherencia audio-visual

### train_tricameral `def train_tricameral()`
- Defined: `tricameral_kimi.py:1452`

### __init__ `def __init__(self, capacity, surprise_threshold)`
- Defined: `tricameral_kimi.py:216`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `tricameral_kimi.py:222`

### add `def add(self, image, caption, surprise_score)`
- Defined: `tricameral_kimi.py:232`

### sample `def sample(self, batch_size)`
- Defined: `tricameral_kimi.py:241`

### __init__ `def __init__(self)`
- Defined: `tricameral_kimi.py:256`

### assess_cognitive_state `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- Defined: `tricameral_kimi.py:266`

### apply_cognitive_intervention `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)`
- Defined: `tricameral_kimi.py:287`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `tricameral_kimi.py:335`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `tricameral_kimi.py:369`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `tricameral_kimi.py:378`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `tricameral_kimi.py:391`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self, alpha, beta)`
- Defined: `tricameral_kimi.py:406`

### compute_linguistic_reward `def compute_linguistic_reward(self, references, hypotheses)`
- Defined: `tricameral_kimi.py:417`

### compute_cider `def compute_cider(self, reference, hypothesis)`
- Defined: `tricameral_kimi.py:443`

### compute_spice `def compute_spice(self, reference, hypothesis)`
- Defined: `tricameral_kimi.py:469`

### _get_ngrams `def _get_ngrams(self, sentence, n)`
- Defined: `tricameral_kimi.py:478`

### get_cache_stats `def get_cache_stats(self)`
- Defined: `tricameral_kimi.py:482`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `tricameral_kimi.py:505`

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `tricameral_kimi.py:528`

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `tricameral_kimi.py:538`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `tricameral_kimi.py:551`

### forward `def forward(self, x)`
- Defined: `tricameral_kimi.py:586`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `tricameral_kimi.py:599`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `tricameral_kimi.py:639`

### __init__ `def __init__(self)`
- Defined: `tricameral_kimi.py:674`

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `tricameral_kimi.py:680`

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `tricameral_kimi.py:703`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `tricameral_kimi.py:770`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `tricameral_kimi.py:781`

### _greedy_decode `def _greedy_decode(self, visual_context, max_len, device)`
- Defined: `tricameral_kimi.py:806`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `tricameral_kimi.py:828`

### __init__ `def __init__(self, output_dim)`
- Defined: `tricameral_kimi.py:846`

### forward `def forward(self, mel_spec)`
- Defined: `tricameral_kimi.py:880`

### __init__ `def __init__(self, output_dim)`
- Defined: `tricameral_kimi.py:896`

### forward `def forward(self, image, audio)`
- Defined: `tricameral_kimi.py:929`
- Doc: Args:

### __init__ `def __init__(self, dim)`
- Defined: `tricameral_kimi.py:966`

### forward `def forward(self, right_features)`
- Defined: `tricameral_kimi.py:995`

### update_channel_fatigue `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- Defined: `tricameral_kimi.py:1038`

### adjust_gates_by_fatigue `def adjust_gates_by_fatigue(self)`
- Defined: `tricameral_kimi.py:1054`

### __init__ `def __init__(self)`
- Defined: `tricameral_kimi.py:1068`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context, channels)`
- Defined: `tricameral_kimi.py:1084`

### evaluate_reasoning_quality `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- Defined: `tricameral_kimi.py:1103`

### calculate_synergy `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `tricameral_kimi.py:1127`

### calculate_health `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `tricameral_kimi.py:1137`

### update `def update(self)`
- Defined: `tricameral_kimi.py:1146`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `tricameral_kimi.py:1156`

### visualize_fatigue_distribution `def visualize_fatigue_distribution(self, epoch)`
- Defined: `tricameral_kimi.py:1173`

### visualize_reasoning_metrics `def visualize_reasoning_metrics(self, epoch)`
- Defined: `tricameral_kimi.py:1191`

### report `def report(self, epoch)`
- Defined: `tricameral_kimi.py:1202`

### __init__ `def __init__(self, vocab_size)`
- Defined: `tricameral_kimi.py:1264`

### forward `def forward(self, image, audio, captions, epoch)`
- Defined: `tricameral_kimi.py:1270`

### __init__ `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)`
- Defined: `tricameral_kimi.py:1299`

### __len__ `def __len__(self)`
- Defined: `tricameral_kimi.py:1347`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `tricameral_kimi.py:1351`

## tricameral_kimi2.py

### setup_flickr8k_with_audio `def setup_flickr8k_with_audio(data_dir)`
- Defined: `tricameral_kimi2.py:50`
- Doc: Descarga y organiza Flickr8k + Audio del dataset de Kaggle.

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `tricameral_kimi2.py:238`
- Doc: Construye vocabulario desde el archivo de captions

### compute_alignment_loss `def compute_alignment_loss(visual_features, channels, alpha, epoch)`
- Defined: `tricameral_kimi2.py:2091`
- Doc: FIX: Pérdida auxiliar para alineación temprana de canales multimodales

### compute_tricameral_loss `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)`
- Defined: `tricameral_kimi2.py:2119`

### train_tricameral `def train_tricameral()`
- Defined: `tricameral_kimi2.py:2166`

### __init__ `def __init__(self, capacity, surprise_threshold)`
- Defined: `tricameral_kimi2.py:263`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `tricameral_kimi2.py:278`

### add `def add(self, image, audio, caption, surprise_score)`
- Defined: `tricameral_kimi2.py:289`

### sample `def sample(self, batch_size)`
- Defined: `tricameral_kimi2.py:318`

### __init__ `def __init__(self)`
- Defined: `tricameral_kimi2.py:348`

### assess_reasoning_state `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)`
- Defined: `tricameral_kimi2.py:363`
- Doc: Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)

### assess_cognitive_state `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- Defined: `tricameral_kimi2.py:407`
- Doc: Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)

### apply_cognitive_intervention `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)`
- Defined: `tricameral_kimi2.py:453`
- Doc: Aplica intervenciones basadas en estado lingüístico y de razonamiento

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `tricameral_kimi2.py:543`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `tricameral_kimi2.py:577`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `tricameral_kimi2.py:586`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `tricameral_kimi2.py:599`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self, alpha, beta)`
- Defined: `tricameral_kimi2.py:614`

### compute_linguistic_reward `def compute_linguistic_reward(self, references, hypotheses)`
- Defined: `tricameral_kimi2.py:636`

### compute_cider `def compute_cider(self, reference, hypothesis)`
- Defined: `tricameral_kimi2.py:675`

### compute_spice `def compute_spice(self, reference, hypothesis)`
- Defined: `tricameral_kimi2.py:695`

### get_cache_stats `def get_cache_stats(self)`
- Defined: `tricameral_kimi2.py:704`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `tricameral_kimi2.py:727`

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `tricameral_kimi2.py:750`

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `tricameral_kimi2.py:760`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `tricameral_kimi2.py:776`

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `tricameral_kimi2.py:799`

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `tricameral_kimi2.py:809`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `tricameral_kimi2.py:822`

### forward `def forward(self, x)`
- Defined: `tricameral_kimi2.py:857`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `tricameral_kimi2.py:870`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `tricameral_kimi2.py:910`

### __init__ `def __init__(self)`
- Defined: `tricameral_kimi2.py:945`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `tricameral_kimi2.py:951`

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `tricameral_kimi2.py:962`

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)`
- Defined: `tricameral_kimi2.py:965`

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `tricameral_kimi2.py:1018`

### _reset_liquid_neuron `def _reset_liquid_neuron(self, liquid_neuron)`
- Defined: `tricameral_kimi2.py:1088`
- Doc: Reset completo de una neurona líquida

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `tricameral_kimi2.py:1104`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `tricameral_kimi2.py:1179`

### _apply_chain_of_thought `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- Defined: `tricameral_kimi2.py:1222`

### _greedy_decode `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- Defined: `tricameral_kimi2.py:1262`

### _apply_multi_token_prediction `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- Defined: `tricameral_kimi2.py:1323`

### _apply_structural_attention `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- Defined: `tricameral_kimi2.py:1365`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `tricameral_kimi2.py:1386`

### __init__ `def __init__(self, output_dim)`
- Defined: `tricameral_kimi2.py:1404`

### forward `def forward(self, mel_spec)`
- Defined: `tricameral_kimi2.py:1438`

### __init__ `def __init__(self, output_dim)`
- Defined: `tricameral_kimi2.py:1454`

### forward `def forward(self, image, audio)`
- Defined: `tricameral_kimi2.py:1487`
- Doc: Args:

### __init__ `def __init__(self, dim)`
- Defined: `tricameral_kimi2.py:1524`

### forward `def forward(self, right_features)`
- Defined: `tricameral_kimi2.py:1572`

### update_channel_fatigue `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- Defined: `tricameral_kimi2.py:1633`

### adjust_gates_by_fatigue `def adjust_gates_by_fatigue(self)`
- Defined: `tricameral_kimi2.py:1655`

### __init__ `def __init__(self)`
- Defined: `tricameral_kimi2.py:1677`

### _get_cached_norm `def _get_cached_norm(self, tensor, dim)`
- Defined: `tricameral_kimi2.py:1699`
- Doc: Cache de normalización con limpieza periódica

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context, channels)`
- Defined: `tricameral_kimi2.py:1717`

### evaluate_reasoning_quality `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- Defined: `tricameral_kimi2.py:1747`

### calculate_synergy `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `tricameral_kimi2.py:1784`

### calculate_health `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `tricameral_kimi2.py:1795`

### update `def update(self)`
- Defined: `tricameral_kimi2.py:1804`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `tricameral_kimi2.py:1821`

### visualize_fatigue_distribution `def visualize_fatigue_distribution(self, epoch)`
- Defined: `tricameral_kimi2.py:1837`

### visualize_reasoning_metrics `def visualize_reasoning_metrics(self, epoch)`
- Defined: `tricameral_kimi2.py:1861`

### report `def report(self, epoch)`
- Defined: `tricameral_kimi2.py:1873`

### __init__ `def __init__(self, vocab_size)`
- Defined: `tricameral_kimi2.py:1955`

### forward `def forward(self, image, audio, captions, epoch)`
- Defined: `tricameral_kimi2.py:1961`

### __init__ `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)`
- Defined: `tricameral_kimi2.py:1990`

### __len__ `def __len__(self)`
- Defined: `tricameral_kimi2.py:2038`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `tricameral_kimi2.py:2042`

### cached_ngrams `def cached_ngrams(sentence, n)`
- Defined: `tricameral_kimi2.py:623`

## tricameralkimi2.py

### setup_flickr8k_with_audio `def setup_flickr8k_with_audio(data_dir)`
- Defined: `tricameralkimi2.py:30`
- Doc: Descarga y organiza Flickr8k + Audio del dataset de Kaggle.

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `tricameralkimi2.py:173`
- Doc: Construye vocabulario desde el archivo de captions

### compute_tricameral_loss `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)`
- Defined: `tricameralkimi2.py:1507`
- Doc: Pérdida con término de coherencia audio-visual

### train_tricameral `def train_tricameral()`
- Defined: `tricameralkimi2.py:1786`

### __init__ `def __init__(self, capacity, surprise_threshold)`
- Defined: `tricameralkimi2.py:198`

### compute_surprise `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- Defined: `tricameralkimi2.py:204`
- Doc: Calcula sorpresa basada en error y apertura del gate

### add `def add(self, image, caption, surprise_score)`
- Defined: `tricameralkimi2.py:216`
- Doc: Añade ejemplo si supera umbral y hay capacidad

### sample `def sample(self, batch_size)`
- Defined: `tricameralkimi2.py:228`
- Doc: Samplea ejemplos con probabilidad proporcional a sorpresa

### __init__ `def __init__(self)`
- Defined: `tricameralkimi2.py:248`

### assess_reasoning_state `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)`
- Defined: `tricameralkimi2.py:263`
- Doc: Evalúa estado del sistema de razonamiento

### assess_cognitive_state `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- Defined: `tricameralkimi2.py:305`
- Doc: Evalúa estado cognitivo lingüístico

### apply_cognitive_intervention `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)`
- Defined: `tricameralkimi2.py:341`
- Doc: Aplica intervenciones basadas en estado lingüístico y razonamiento

### __init__ `def __init__(self, alpha, beta)`
- Defined: `tricameralkimi2.py:428`

### compute_linguistic_reward `def compute_linguistic_reward(self, references, hypotheses)`
- Defined: `tricameralkimi2.py:442`
- Doc: Recompensa combinada CIDEr + SPICE con caché

### compute_cider `def compute_cider(self, reference, hypothesis)`
- Defined: `tricameralkimi2.py:476`
- Doc: CIDEr simplificado con caché de n-gramas

### compute_spice `def compute_spice(self, reference, hypothesis)`
- Defined: `tricameralkimi2.py:505`
- Doc: SPICE simplificado (Jaccard similarity)

### _get_ngrams `def _get_ngrams(self, sentence, n)`
- Defined: `tricameralkimi2.py:518`
- Doc: Extractor de n-gramas

### get_cache_stats `def get_cache_stats(self)`
- Defined: `tricameralkimi2.py:523`
- Doc: Estadísticas de caché

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `tricameralkimi2.py:547`

### forward `def forward(self, x)`
- Defined: `tricameralkimi2.py:582`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `tricameralkimi2.py:595`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `tricameralkimi2.py:633`

### __init__ `def __init__(self, output_dim)`
- Defined: `tricameralkimi2.py:671`

### forward `def forward(self, mel_spec)`
- Defined: `tricameralkimi2.py:705`
- Doc: Args:

### __init__ `def __init__(self)`
- Defined: `tricameralkimi2.py:725`

### triangulate_signals `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `tricameralkimi2.py:731`

### count_convergent_signals `def count_convergent_signals(self, signals, pattern)`
- Defined: `tricameralkimi2.py:741`

### diagnose_with_triangulation `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- Defined: `tricameralkimi2.py:744`

### apply_triangulated_intervention `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- Defined: `tricameralkimi2.py:786`

### _reset_liquid_neuron `def _reset_liquid_neuron(self, right_node, severity)`
- Defined: `tricameralkimi2.py:868`

### __init__ `def __init__(self, output_dim)`
- Defined: `tricameralkimi2.py:884`

### forward `def forward(self, image, audio)`
- Defined: `tricameralkimi2.py:917`
- Doc: Args:

### __init__ `def __init__(self, dim)`
- Defined: `tricameralkimi2.py:955`

### forward `def forward(self, right_features)`
- Defined: `tricameralkimi2.py:991`
- Doc: Args:

### update_channel_fatigue `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- Defined: `tricameralkimi2.py:1060`
- Doc: Actualiza fatiga específica por canal

### adjust_gates_by_fatigue `def adjust_gates_by_fatigue(self)`
- Defined: `tricameralkimi2.py:1079`
- Doc: Ajusta proyecciones basado en fatiga

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `tricameralkimi2.py:1090`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `tricameralkimi2.py:1165`

### _greedy_decode `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- Defined: `tricameralkimi2.py:1204`

### _apply_chain_of_thought `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- Defined: `tricameralkimi2.py:1237`

### _apply_multi_token_prediction `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- Defined: `tricameralkimi2.py:1271`

### _apply_structural_attention `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- Defined: `tricameralkimi2.py:1317`
- Doc: Atenuación simple según fatiga de canal + atención cruzada visual.

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `tricameralkimi2.py:1344`

### __init__ `def __init__(self, vocab_size)`
- Defined: `tricameralkimi2.py:1362`

### forward `def forward(self, image, audio, captions, epoch)`
- Defined: `tricameralkimi2.py:1374`
- Doc: Args:

### __init__ `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)`
- Defined: `tricameralkimi2.py:1418`

### __len__ `def __len__(self)`
- Defined: `tricameralkimi2.py:1463`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `tricameralkimi2.py:1466`

### __init__ `def __init__(self)`
- Defined: `tricameralkimi2.py:1557`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context, channels)`
- Defined: `tricameralkimi2.py:1572`

### evaluate_reasoning_quality `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- Defined: `tricameralkimi2.py:1595`
- Doc: Evalúa coherencia y consistencia del razonamiento

### calculate_synergy `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `tricameralkimi2.py:1627`

### calculate_health `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `tricameralkimi2.py:1638`

### update `def update(self)`
- Defined: `tricameralkimi2.py:1647`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `tricameralkimi2.py:1657`

### visualize_fatigue_distribution `def visualize_fatigue_distribution(self, epoch)`
- Defined: `tricameralkimi2.py:1674`

### visualize_reasoning_metrics `def visualize_reasoning_metrics(self, epoch)`
- Defined: `tricameralkimi2.py:1695`

### report `def report(self, epoch)`
- Defined: `tricameralkimi2.py:1707`
- Doc: Genera reporte completo del estado del sistema tricameral

## trycameral.py

### generate_audio_async `def generate_audio_async(text, output_path, voice)`
- Defined: `trycameral.py:31`
- Doc: Genera un audio usando Edge-TTS

### generate_all_audios_batch `def generate_all_audios_batch(captions_list, audio_dir, batch_size)`
- Defined: `trycameral.py:50`
- Doc: Genera todos los audios en batches para eficiencia

### generate_audios_sync `def generate_audios_sync(images_dir, captions_file, audio_dir)`
- Defined: `trycameral.py:89`
- Doc: Wrapper síncrono para generar audios

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `trycameral.py:133`
- Doc: Descarga y organiza Flickr8k si no existe

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `trycameral.py:205`
- Doc: Construye vocabulario desde el archivo de captions

### compute_tricameral_loss `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)`
- Defined: `trycameral.py:887`
- Doc: Pérdida con término de coherencia audio-visual

### train_tricameral `def train_tricameral()`
- Defined: `trycameral.py:939`

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `trycameral.py:229`

### forward `def forward(self, x)`
- Defined: `trycameral.py:264`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `trycameral.py:277`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `trycameral.py:315`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `trycameral.py:352`

### forward `def forward(self, visual_context, captions, channels, max_len, epoch)`
- Defined: `trycameral.py:364`

### _greedy_decode `def _greedy_decode(self, visual_context, max_len, device)`
- Defined: `trycameral.py:387`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `trycameral.py:409`

### __init__ `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)`
- Defined: `trycameral.py:426`

### __len__ `def __len__(self)`
- Defined: `trycameral.py:464`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `trycameral.py:467`

### __init__ `def __init__(self, output_dim)`
- Defined: `trycameral.py:520`

### forward `def forward(self, mel_spec)`
- Defined: `trycameral.py:559`
- Doc: Args:

### __init__ `def __init__(self, output_dim)`
- Defined: `trycameral.py:589`

### forward `def forward(self, image, audio)`
- Defined: `trycameral.py:623`
- Doc: Args:

### __init__ `def __init__(self, dim)`
- Defined: `trycameral.py:663`

### forward `def forward(self, right_features)`
- Defined: `trycameral.py:695`
- Doc: Args:

### __init__ `def __init__(self, text_dim, output_sr)`
- Defined: `trycameral.py:755`

### forward `def forward(self, text_embedding)`
- Defined: `trycameral.py:790`
- Doc: Args:

### __init__ `def __init__(self, vocab_size)`
- Defined: `trycameral.py:824`

### forward `def forward(self, image, audio, captions, epoch, generate_audio)`
- Defined: `trycameral.py:839`
- Doc: Args:

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `trycameral.py:997`

### __len__ `def __len__(self)`
- Defined: `trycameral.py:1015`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `trycameral.py:1018`

## ultimo_neuorlogos.py

### train_ablation `def train_ablation(mode, epochs, device)`
- Defined: `ultimo_neuorlogos.py:374`
- Doc: Entrena un modelo en el modo especificado

### run_full_ablation `def run_full_ablation(epochs, device)`
- Defined: `ultimo_neuorlogos.py:508`
- Doc: Ejecuta ablation study completo de 3 niveles

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)`
- Defined: `ultimo_neuorlogos.py:29`

### _init_grid `def _init_grid(self)`
- Defined: `ultimo_neuorlogos.py:62`
- Doc: Inicializa coordenadas del grid 2D

### forward `def forward(self, x)`
- Defined: `ultimo_neuorlogos.py:70`

### get_metrics `def get_metrics(self)`
- Defined: `ultimo_neuorlogos.py:101`
- Doc: Retorna métricas de topología

### __init__ `def __init__(self, epsilon, alpha, steps)`
- Defined: `ultimo_neuorlogos.py:115`

### attack `def attack(self, model, x, y, criterion)`
- Defined: `ultimo_neuorlogos.py:120`
- Doc: Genera ejemplos adversariales

### __init__ `def __init__(self, output_dim)`
- Defined: `ultimo_neuorlogos.py:147`

### forward `def forward(self, x)`
- Defined: `ultimo_neuorlogos.py:160`

### __init__ `def __init__(self, output_dim, use_grid, use_symbiotic)`
- Defined: `ultimo_neuorlogos.py:168`

### forward `def forward(self, x)`
- Defined: `ultimo_neuorlogos.py:191`

### get_metrics `def get_metrics(self)`
- Defined: `ultimo_neuorlogos.py:195`

### __init__ `def __init__(self, dim)`
- Defined: `ultimo_neuorlogos.py:205`

### forward `def forward(self, x)`
- Defined: `ultimo_neuorlogos.py:210`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `ultimo_neuorlogos.py:223`

### forward `def forward(self, thought, captions, max_len)`
- Defined: `ultimo_neuorlogos.py:238`

### _get_init_state `def _get_init_state(self, thought)`
- Defined: `ultimo_neuorlogos.py:272`

### __init__ `def __init__(self, vocab_size, mode)`
- Defined: `ultimo_neuorlogos.py:289`

### forward `def forward(self, image, captions)`
- Defined: `ultimo_neuorlogos.py:314`

### get_metrics `def get_metrics(self)`
- Defined: `ultimo_neuorlogos.py:319`
- Doc: Obtiene métricas de topología si disponible

### __init__ `def __init__(self)`
- Defined: `ultimo_neuorlogos.py:331`

### __len__ `def __len__(self)`
- Defined: `ultimo_neuorlogos.py:356`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `ultimo_neuorlogos.py:359`

## ultimobicameral.py

### build_vocab_flickr `def build_vocab_flickr(captions_file, vocab_size)`
- Defined: `ultimobicameral.py:645`

### setup_flickr8k `def setup_flickr8k(data_dir)`
- Defined: `ultimobicameral.py:663`

### train_with_metrics `def train_with_metrics()`
- Defined: `ultimobicameral.py:677`

### sentence_bleu `def sentence_bleu(reference, hypothesis, weights)`
- Defined: `ultimobicameral.py:27`
- Doc: BLEU simplificado a nivel de oración

### _get_ngrams `def _get_ngrams(tokens, n)`
- Defined: `ultimobicameral.py:61`
- Doc: Extraer n-gramas de una lista de tokens

### token_accuracy `def token_accuracy(reference, hypothesis)`
- Defined: `ultimobicameral.py:70`
- Doc: Porcentaje de tokens correctos en posición

### word_overlap `def word_overlap(reference, hypothesis)`
- Defined: `ultimobicameral.py:83`
- Doc: Jaccard similarity entre palabras

### __init__ `def __init__(self)`
- Defined: `ultimobicameral.py:99`

### diagnose_severity `def diagnose_severity(self, health_score, liquid_norm, gate_mean, callosal_flow)`
- Defined: `ultimobicameral.py:103`
- Doc: Diagnosticar gravedad del problema con análisis mejorado

### apply_intervention `def apply_intervention(self, model, issues, severity, epoch)`
- Defined: `ultimobicameral.py:149`
- Doc: Aplicar intervención médica calibrada con más agresividad en gate

### __init__ `def __init__(self, in_dim, out_dim)`
- Defined: `ultimobicameral.py:293`

### forward `def forward(self, x)`
- Defined: `ultimobicameral.py:307`

### hebbian_update `def hebbian_update(self, post, pre, plasticity)`
- Defined: `ultimobicameral.py:314`

### update_physiology_advanced `def update_physiology_advanced(self, loss_value)`
- Defined: `ultimobicameral.py:344`

### __init__ `def __init__(self, output_dim)`
- Defined: `ultimobicameral.py:369`

### forward `def forward(self, image)`
- Defined: `ultimobicameral.py:377`

### __init__ `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- Defined: `ultimobicameral.py:384`

### forward `def forward(self, visual_context, captions, max_len)`
- Defined: `ultimobicameral.py:401`

### _get_init_state `def _get_init_state(self, visual_context)`
- Defined: `ultimobicameral.py:438`

### __init__ `def __init__(self, dim)`
- Defined: `ultimobicameral.py:444`

### forward `def forward(self, right_features)`
- Defined: `ultimobicameral.py:456`

### __init__ `def __init__(self, vocab_size)`
- Defined: `ultimobicameral.py:463`

### forward `def forward(self, image, captions)`
- Defined: `ultimobicameral.py:469`

### __init__ `def __init__(self)`
- Defined: `ultimobicameral.py:484`

### measure_callosal_flow `def measure_callosal_flow(self, right_features, left_context)`
- Defined: `ultimobicameral.py:494`

### calculate_synergy `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- Defined: `ultimobicameral.py:503`

### calculate_health `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- Defined: `ultimobicameral.py:512`

### update `def update(self)`
- Defined: `ultimobicameral.py:521`

### get_recent_avg `def get_recent_avg(self, key, n)`
- Defined: `ultimobicameral.py:526`

### report `def report(self, epoch)`
- Defined: `ultimobicameral.py:531`

### __init__ `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- Defined: `ultimobicameral.py:609`

### __len__ `def __len__(self)`
- Defined: `ultimobicameral.py:626`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `ultimobicameral.py:629`

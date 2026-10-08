# API (page 7 of 10)
Previous: [API_p6.md](API_p6.md)

## omni1.py
- `FastSlowLinear.__init__` (method) `omni1.py:33` `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- `FastSlowLinear.reset_fast_weights` (method) `omni1.py:46` `def reset_fast_weights(self)`
- `FastSlowLinear.update_fast_weights` (method) `omni1.py:49` `def update_fast_weights(self, x)`
- `FastSlowLinear.forward` (method) `omni1.py:68` `def forward(self, x)`
- `FastSlowLinear.get_fast_norm` (method) `omni1.py:75` `def get_fast_norm(self)`
- `ConsciousnessModule.__init__` (method) `omni1.py:85` `def __init__(self, features, use_conscious)`
- `ConsciousnessModule.compute_phi_effective` (method) `omni1.py:99` `def compute_phi_effective(self, activity)`
- `ConsciousnessModule.forward` (method) `omni1.py:121` `def forward(self, x)`
- `OmniBrainV8.__init__` (method) `omni1.py:142` `def __init__(self, use_fastslow, use_conscious)`
- `OmniBrainV8.forward` (method) `omni1.py:188` `def forward(self, x)`
- `OmniBrainV8.reset_all_fast_weights` (method) `omni1.py:198` `def reset_all_fast_weights(self)`
- `OmniBrainV8.get_fast_norms` (method) `omni1.py:205` `def get_fast_norms(self)`
- `DualSystemModule.__init__` (method) `omni1.py:217` `def __init__(self, dim, use_fastslow)`
- `DualSystemModule.forward` (method) `omni1.py:235` `def forward(self, x)`
- `DualSystemModule.train_model` (method) `omni1.py:250` `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)` -- Entrenamiento optimizado.
- `FocalLoss.__init__` (method) `omni1.py:262` `def __init__(self, alpha, gamma)`
- `FocalLoss.forward` (method) `omni1.py:268` `def forward(self, inputs, targets)`
- `FocalLoss.evaluate` (method) `omni1.py:356` `def evaluate(model, loader, device, return_per_class)` -- Evaluación estándar.
- `FocalLoss.get_cifar10_loaders` (method) `omni1.py:399` `def get_cifar10_loaders(batch_size)` -- Loaders con data augmentation.
- `FocalLoss.diagnose_model` (method) `omni1.py:423` `def diagnose_model(model, loader, device)` -- Diagnóstico profundo del modelo.
- `FocalLoss.get_activation` (method) `omni1.py:436` `def get_activation(name)`
- `FocalLoss.hook` (method) `omni1.py:437` `def hook(model, input, output)`
- `FocalLoss.main` (method) `omni1.py:481` `def main()` -- POC mejorado.

## omni3.py
- `Config.compute_integration_index` (method) `omni3.py:65` `def compute_integration_index(activity)` -- MÉTRICA HONESTA: Mide el ratio de varianza explicada por el primer componente.
- `FastSlowLinear.__init__` (method) `omni3.py:106` `def __init__(self, in_features, out_features, config)`
- `FastSlowLinear.reset_fast_weights` (method) `omni3.py:128` `def reset_fast_weights(self)` -- Reinicia la memoria a corto plazo (debe llamarse por época, no por batch)
- `FastSlowLinear.update_fast_weights` (method) `omni3.py:134` `def update_fast_weights(self, x, slow_out)` -- Actualización Hebbiana controlada.
- `FastSlowLinear.forward` (method) `omni3.py:169` `def forward(self, x)`
- `FastSlowLinear.get_fast_norm` (method) `omni3.py:189` `def get_fast_norm(self)`
- `DualSystemModule.__init__` (method) `omni3.py:197` `def __init__(self, dim, config)`
- `DualSystemModule.forward` (method) `omni3.py:211` `def forward(self, x)`
- `IntegrationModule.__init__` (method) `omni3.py:236` `def __init__(self, features, config)`
- `IntegrationModule.forward` (method) `omni3.py:248` `def forward(self, x)`
- `OmniBrainFastSlow.__init__` (method) `omni3.py:269` `def __init__(self, config)`
- `OmniBrainFastSlow.forward` (method) `omni3.py:304` `def forward(self, x)`
- `OmniBrainFastSlow.reset_all_fast_weights` (method) `omni3.py:314` `def reset_all_fast_weights(self)` -- Reinicia todos los pesos rápidos del modelo.
- `OmniBrainFastSlow.get_fast_norms` (method) `omni3.py:323` `def get_fast_norms(self)` -- Recopila normas de fast weights de todos los módulos
- `OmniBrainFastSlow.get_ablation_state` (method) `omni3.py:327` `def get_ablation_state(self)` -- Estado actual para logging
- `OmniBrainFastSlow.get_cifar10_loaders` (method) `omni3.py:340` `def get_cifar10_loaders(config)`
- `OmniBrainFastSlow.evaluate_full` (method) `omni3.py:368` `def evaluate_full(model, loader, device)` -- Evaluación con múltiples métricas
- `OmniBrainFastSlow.train` (method) `omni3.py:403` `def train(config)`
- `OmniBrainFastSlow.run_ablation_study` (method) `omni3.py:539` `def run_ablation_study()` -- Ejecuta múltiples configuraciones para validar cada componente

## omnibrain.py
- `FastSlowLinear.__init__` (method) `omnibrain.py:33` `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- `FastSlowLinear.reset_fast_weights` (method) `omnibrain.py:49` `def reset_fast_weights(self)`
- `FastSlowLinear.update_fast_weights` (method) `omnibrain.py:53` `def update_fast_weights(self, x)`
- `FastSlowLinear.forward` (method) `omnibrain.py:73` `def forward(self, x)`
- `FastSlowLinear.end_of_batch` (method) `omnibrain.py:84` `def end_of_batch(self)`
- `FastSlowLinear.get_fast_norm` (method) `omnibrain.py:87` `def get_fast_norm(self)`
- `DualSystemModule.__init__` (method) `omnibrain.py:93` `def __init__(self, dim, use_fastslow)`
- `DualSystemModule.forward` (method) `omnibrain.py:109` `def forward(self, x)`
- `ConsciousnessModule.__init__` (method) `omnibrain.py:121` `def __init__(self, features, use_conscious)`
- `ConsciousnessModule.compute_phi_effective` (method) `omnibrain.py:133` `def compute_phi_effective(self, activity)` -- Φₑ basado en eigenvalues de covarianza.
- `ConsciousnessModule.forward` (method) `omnibrain.py:156` `def forward(self, x)`
- `OmniBrainV8.__init__` (method) `omnibrain.py:176` `def __init__(self, use_fastslow, use_conscious)`
- `OmniBrainV8.forward` (method) `omnibrain.py:209` `def forward(self, x)`
- `OmniBrainV8.reset_all_fast_weights` (method) `omnibrain.py:216` `def reset_all_fast_weights(self)`
- `OmniBrainV8.get_fast_norms` (method) `omnibrain.py:223` `def get_fast_norms(self)`
- `OmniBrainV8.get_cifar10_loaders` (method) `omnibrain.py:233` `def get_cifar10_loaders(batch_size)` -- Loaders con data augmentation.
- `OmniBrainV8.get_few_shot_loaders` (method) `omnibrain.py:253` `def get_few_shot_loaders(n_way, k_shot, batch_size)` -- Few-shot learning setup: entrenar en clases limitadas.
- `OmniBrainV8.evaluate` (method) `omnibrain.py:287` `def evaluate(model, loader, device, return_per_class)` -- Evaluación con opción de métricas por clase.
- `OmniBrainV8.train_model` (method) `omnibrain.py:331` `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)` -- Entrenamiento con learning rate scheduler y early stopping.
- `OmniBrainV8.run_ablation_study` (method) `omnibrain.py:430` `def run_ablation_study(epochs, batch_size)` -- Ejecuta 4 configuraciones y compara resultados.
- `OmniBrainV8.run_few_shot_experiment` (method) `omnibrain.py:475` `def run_few_shot_experiment(n_way, k_shot, epochs)` -- Prueba capacidad de few-shot learning.
- `OmniBrainV8.analyze_phi_per_class` (method) `omnibrain.py:508` `def analyze_phi_per_class()` -- Analiza correlación entre Φₑ y dificultad de clase.
- `OmniBrainV8.plot_ablation_results` (method) `omnibrain.py:545` `def plot_ablation_results(results)` -- Genera gráficas comparativas de ablation study.
- `OmniBrainV8.plot_phi_analysis` (method) `omnibrain.py:617` `def plot_phi_analysis(class_accs, avg_phi_per_class)` -- Gráfica correlación Φₑ vs dificultad de clase.
- `OmniBrainV8.main` (method) `omnibrain.py:673` `def main()` -- Ejecuta el POC completo.

## omnibrain_k.py
- `Config.compute_integration_index` (method) `omnibrain_k.py:65` `def compute_integration_index(activity)` -- MÉTRICA HONESTA: Mide el ratio de varianza explicada por el primer componente.
- `FastSlowLinear.__init__` (method) `omnibrain_k.py:106` `def __init__(self, in_features, out_features, config)`
- `FastSlowLinear.reset_fast_weights` (method) `omnibrain_k.py:128` `def reset_fast_weights(self)` -- Reinicia la memoria a corto plazo (debe llamarse por época, no por batch)
- `FastSlowLinear.update_fast_weights` (method) `omnibrain_k.py:134` `def update_fast_weights(self, x, slow_out)` -- Actualización Hebbiana controlada.
- `FastSlowLinear.forward` (method) `omnibrain_k.py:169` `def forward(self, x)`
- `FastSlowLinear.get_fast_norm` (method) `omnibrain_k.py:189` `def get_fast_norm(self)`
- `DualSystemModule.__init__` (method) `omnibrain_k.py:196` `def __init__(self, dim, config)`
- `DualSystemModule.forward` (method) `omnibrain_k.py:210` `def forward(self, x)`
- `IntegrationModule.__init__` (method) `omnibrain_k.py:235` `def __init__(self, features, config)`
- `IntegrationModule.forward` (method) `omnibrain_k.py:247` `def forward(self, x)`
- `OmniBrainFastSlow.__init__` (method) `omnibrain_k.py:267` `def __init__(self, config)`
- `OmniBrainFastSlow.forward` (method) `omnibrain_k.py:302` `def forward(self, x)`
- `OmniBrainFastSlow.reset_all_fast_weights` (method) `omnibrain_k.py:312` `def reset_all_fast_weights(self)` -- Reinicia todos los pesos rápidos del modelo.
- `OmniBrainFastSlow.get_fast_norms` (method) `omnibrain_k.py:322` `def get_fast_norms(self)` -- Recopila normas de fast weights de todos los módulos
- `OmniBrainFastSlow.get_ablation_state` (method) `omnibrain_k.py:326` `def get_ablation_state(self)` -- Estado actual para logging
- `OmniBrainFastSlow.get_cifar10_loaders` (method) `omnibrain_k.py:339` `def get_cifar10_loaders(config)`
- `OmniBrainFastSlow.evaluate_full` (method) `omnibrain_k.py:367` `def evaluate_full(model, loader, device)` -- Evaluación con múltiples métricas
- `OmniBrainFastSlow.train` (method) `omnibrain_k.py:402` `def train(config)`
- `OmniBrainFastSlow.run_ablation_study` (method) `omnibrain_k.py:537` `def run_ablation_study()` -- Ejecuta múltiples configuraciones para validar cada componente

## omno1.bkp.py.py
- `FastSlowLinear.__init__` (method) `omno1.bkp.py.py:34` `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- `FastSlowLinear.reset_fast_weights` (method) `omno1.bkp.py.py:51` `def reset_fast_weights(self)`
- `FastSlowLinear.update_fast_weights` (method) `omno1.bkp.py.py:56` `def update_fast_weights(self, x)`
- `FastSlowLinear.forward` (method) `omno1.bkp.py.py:85` `def forward(self, x)`
- `FastSlowLinear.get_fast_norm` (method) `omno1.bkp.py.py:93` `def get_fast_norm(self)`
- `ConsciousnessModule.__init__` (method) `omno1.bkp.py.py:103` `def __init__(self, features, use_conscious)`
- `ConsciousnessModule.compute_phi_effective` (method) `omno1.bkp.py.py:118` `def compute_phi_effective(self, activity)` -- Φₑ mejorado con condiciones menos restrictivas.
- `ConsciousnessModule.forward` (method) `omno1.bkp.py.py:159` `def forward(self, x)`
- `OmniBrainV8.__init__` (method) `omno1.bkp.py.py:183` `def __init__(self, use_fastslow, use_conscious)`
- `OmniBrainV8.forward` (method) `omno1.bkp.py.py:229` `def forward(self, x)`
- `OmniBrainV8.reset_all_fast_weights` (method) `omno1.bkp.py.py:239` `def reset_all_fast_weights(self)`
- `OmniBrainV8.get_fast_norms` (method) `omno1.bkp.py.py:246` `def get_fast_norms(self)`
- `DualSystemModule.__init__` (method) `omno1.bkp.py.py:258` `def __init__(self, dim, use_fastslow)`
- `DualSystemModule.forward` (method) `omno1.bkp.py.py:276` `def forward(self, x)`
- `DualSystemModule.train_model` (method) `omno1.bkp.py.py:291` `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)` -- Entrenamiento optimizado.
- `FocalLoss.__init__` (method) `omno1.bkp.py.py:303` `def __init__(self, alpha, gamma)`
- `FocalLoss.forward` (method) `omno1.bkp.py.py:309` `def forward(self, inputs, targets)`
- `FocalLoss.evaluate` (method) `omno1.bkp.py.py:399` `def evaluate(model, loader, device, return_per_class)` -- Evaluación estándar.
- `FocalLoss.get_cifar10_loaders` (method) `omno1.bkp.py.py:442` `def get_cifar10_loaders(batch_size)` -- Loaders con data augmentation.
- `FocalLoss.diagnose_model` (method) `omno1.bkp.py.py:466` `def diagnose_model(model, loader, device)` -- Diagnóstico profundo del modelo.
- `FocalLoss.get_activation` (method) `omno1.bkp.py.py:479` `def get_activation(name)`
- `FocalLoss.hook` (method) `omno1.bkp.py.py:480` `def hook(model, input, output)`
- `FocalLoss.main` (method) `omno1.bkp.py.py:524` `def main()` -- POC mejorado.

## physio_chimera_demo.py
- `Config.seed_everything` (method) `physio_chimera_demo.py:36` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `physio_chimera_demo.py:47` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `physio_chimera_demo.py:57` `def get_batch(self, phase, bs)`
- `DataEnvironment.get_full` (method) `physio_chimera_demo.py:71` `def get_full(self)` -- Retorna el dataset completo
- `DataEnvironment.get_w2` (method) `physio_chimera_demo.py:75` `def get_w2(self)` -- Retorna solo los datos de WORLD_2 (dígitos >= 5)
- `SimpleMonitor.__init__` (method) `physio_chimera_demo.py:83` `def __init__(self)`
- `SimpleMonitor.update` (method) `physio_chimera_demo.py:88` `def update(self, loss, physio)`
- `SimpleMonitor.report` (method) `physio_chimera_demo.py:94` `def report(self, step, phase)`
- `SimpleCMS.__init__` (method) `physio_chimera_demo.py:131` `def __init__(self, levels, d_model, hidden_dim)`
- `SimpleCMS.forward` (method) `physio_chimera_demo.py:142` `def forward(self, x, global_step)`
- `SimplePhysioNeuron.__init__` (method) `physio_chimera_demo.py:154` `def __init__(self, d_in, d_out, config)`
- `SimplePhysioNeuron.forward` (method) `physio_chimera_demo.py:162` `def forward(self, x, global_step)`
- `SimplePhysioChimera.__init__` (method) `physio_chimera_demo.py:195` `def __init__(self, config)`
- `SimplePhysioChimera.forward` (method) `physio_chimera_demo.py:207` `def forward(self, x, global_step)`
- `SimplePhysioChimera.train_demo` (method) `physio_chimera_demo.py:231` `def train_demo(config)`
- `SimplePhysioChimera.run_demo` (method) `physio_chimera_demo.py:298` `def run_demo()`

## physio_chimera_v15_monitored.py
Imported by: `example_usage.py`, `run_complete_experiment.py`
- `Config.seed_everything` (method) `physio_chimera_v15_monitored.py:58` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `physio_chimera_v15_monitored.py:69` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `physio_chimera_v15_monitored.py:79` `def get_batch(self, phase, bs)`
- `DataEnvironment.get_full` (method) `physio_chimera_v15_monitored.py:93` `def get_full(self)` -- Retorna el dataset completo
- `DataEnvironment.get_w2` (method) `physio_chimera_v15_monitored.py:97` `def get_w2(self)` -- Retorna solo los datos de WORLD_2 (dígitos >= 5)
- `NeuralDiagnostics.__init__` (method) `physio_chimera_v15_monitored.py:107` `def __init__(self, config)`
- `NeuralDiagnostics.update_physio_metrics` (method) `physio_chimera_v15_monitored.py:144` `def update_physio_metrics(self, metabolism, sensitivity, gate)` -- Actualiza métricas fisiológicas
- `NeuralDiagnostics.update_performance_metrics` (method) `physio_chimera_v15_monitored.py:150` `def update_performance_metrics(self, loss, accuracy, lr)` -- Actualiza métricas de rendimiento
- `NeuralDiagnostics.update_memory_metrics` (method) `physio_chimera_v15_monitored.py:158` `def update_memory_metrics(self, cms_activations, hebbian_norm, forgetting_factor)` -- Actualiza métricas de memoria
- `NeuralDiagnostics.calculate_health_metrics` (method) `physio_chimera_v15_monitored.py:166` `def calculate_health_metrics(self)` -- Calcula métricas de salud del sistema
- `NeuralDiagnostics.get_recent_avg` (method) `physio_chimera_v15_monitored.py:191` `def get_recent_avg(self, category, key, n)` -- Obtiene promedio reciente de una métrica
- `NeuralDiagnostics.generate_diagnostic_report` (method) `physio_chimera_v15_monitored.py:209` `def generate_diagnostic_report(self, step, phase)` -- Genera reporte de diagnóstico
- `NeuralDiagnostics.save_metrics` (method) `physio_chimera_v15_monitored.py:279` `def save_metrics(self, filepath)` -- Guarda todas las métricas
- `SelfModifyingGates.__init__` (method) `physio_chimera_v15_monitored.py:299` `def __init__(self, input_dim, hidden_dim)`
- `SelfModifyingGates.forward` (method) `physio_chimera_v15_monitored.py:306` `def forward(self, x)`
- `ContinuumMemorySystem.__init__` (method) `physio_chimera_v15_monitored.py:320` `def __init__(self, levels, d_model, hidden_dim)`
- `ContinuumMemorySystem.forward` (method) `physio_chimera_v15_monitored.py:332` `def forward(self, x, global_step)`
- `NestedPhysioNeuron.__init__` (method) `physio_chimera_v15_monitored.py:347` `def __init__(self, d_in, d_out, config)`
- `NestedPhysioNeuron.forward` (method) `physio_chimera_v15_monitored.py:362` `def forward(self, x, global_step)`
- `PhysioChimeraNested.__init__` (method) `physio_chimera_v15_monitored.py:398` `def __init__(self, config)`
- `PhysioChimeraNested.forward` (method) `physio_chimera_v15_monitored.py:410` `def forward(self, x, global_step)`
- `MetricsVisualizer.__init__` (method) `physio_chimera_v15_monitored.py:455` `def __init__(self, save_dir)`
- `MetricsVisualizer.plot_training_curves` (method) `physio_chimera_v15_monitored.py:463` `def plot_training_curves(self, diagnostics)` -- Genera gráficos de curvas de entrenamiento
- `MetricsVisualizer.create_final_report` (method) `physio_chimera_v15_monitored.py:551` `def create_final_report(self, final_metrics, diagnostics)` -- Crea reporte final con todas las métricas
- `MetricsVisualizer.train_nested_monitored` (method) `physio_chimera_v15_monitored.py:658` `def train_nested_monitored(config)`
- `MetricsVisualizer.run_experiment_monitored` (method) `physio_chimera_v15_monitored.py:762` `def run_experiment_monitored()`

## physioneruon_simple.py
- `SimpleConfig.seed_everything` (method) `physioneruon_simple.py:54` `def seed_everything(seed)`
- `SimpleConfig.get_dataset` (method) `physioneruon_simple.py:61` `def get_dataset(config)` -- Dataset balanceado con más separabilidad
- `SimpleRobustNet.__init__` (method) `physioneruon_simple.py:85` `def __init__(self, config)`
- `SimpleRobustNet.forward` (method) `physioneruon_simple.py:105` `def forward(self, x)`
- `SimpleRobustNet.pgd_attack` (method) `physioneruon_simple.py:117` `def pgd_attack(model, x, y, eps, steps, step_size)` -- PGD estándar bien implementado - Random start - Step size controlado - Projection al epsilon-ball
- `SimpleRobustNet.train_simple_robust` (method) `physioneruon_simple.py:154` `def train_simple_robust(config, dataset, verbose)` -- Entrenamiento con adversarial training progresivo
- `SimpleRobustNet.main` (method) `physioneruon_simple.py:281` `def main()`

## physioneuron_cpu_v1.py
- `MicroConfig.seed_everything` (method) `physioneuron_cpu_v1.py:57` `def seed_everything(seed)`
- `MicroConfig.get_dataset` (method) `physioneuron_cpu_v1.py:65` `def get_dataset(config)`
- `HomeostaticRegulator.__init__` (method) `physioneuron_cpu_v1.py:86` `def __init__(self, d_in)`
- `HomeostaticRegulator.forward` (method) `physioneuron_cpu_v1.py:96` `def forward(self, x, h_pre, w_norm)`
- `PhysioNeuron.__init__` (method) `physioneuron_cpu_v1.py:110` `def __init__(self, d_in, d_out, dynamic_mode)`
- `PhysioNeuron.forward` (method) `physioneuron_cpu_v1.py:121` `def forward(self, x)`
- `MicroContinuumCell.__init__` (method) `physioneuron_cpu_v1.py:152` `def __init__(self, dim)`
- `MicroContinuumCell.forward` (method) `physioneuron_cpu_v1.py:162` `def forward(self, x, plasticity)`
- `MicroSymbioticBasis.__init__` (method) `physioneuron_cpu_v1.py:177` `def __init__(self, dim, num_atoms)`
- `MicroSymbioticBasis.forward` (method) `physioneuron_cpu_v1.py:185` `def forward(self, x)`
- `MicroTopology.__init__` (method) `physioneuron_cpu_v1.py:197` `def __init__(self, num_nodes, config)`
- `MicroTopology.get_adjacency` (method) `physioneuron_cpu_v1.py:210` `def get_adjacency(self, plasticity)`
- `MicroSupConLoss.__init__` (method) `physioneuron_cpu_v1.py:217` `def __init__(self, temperature)`
- `MicroSupConLoss.forward` (method) `physioneuron_cpu_v1.py:222` `def forward(self, features, labels)`
- `MicroTopoBrain.__init__` (method) `physioneuron_cpu_v1.py:241` `def __init__(self, config)`
- `MicroTopoBrain.count_parameters` (method) `physioneuron_cpu_v1.py:283` `def count_parameters(self)`
- `MicroTopoBrain.forward` (method) `physioneuron_cpu_v1.py:286` `def forward(self, x, plasticity)`
- `MicroTopoBrain.micro_pgd_attack` (method) `physioneuron_cpu_v1.py:340` `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- `MicroTopoBrain.generate_ablation_matrix` (method) `physioneuron_cpu_v1.py:364` `def generate_ablation_matrix()`
- `MicroTopoBrain.train_with_cv` (method) `physioneuron_cpu_v1.py:393` `def train_with_cv(config, dataset, cv_folds)`
- `MicroTopoBrain.run_ablation_study` (method) `physioneuron_cpu_v1.py:457` `def run_ablation_study()`

## physioneuron_cpu_v2.py
- `MicroConfig.seed_everything` (method) `physioneuron_cpu_v2.py:64` `def seed_everything(seed)`
- `MicroConfig.get_dataset` (method) `physioneuron_cpu_v2.py:72` `def get_dataset(config)`
- `HomeostaticRegulator.__init__` (method) `physioneuron_cpu_v2.py:93` `def __init__(self, d_in)`
- `HomeostaticRegulator.forward` (method) `physioneuron_cpu_v2.py:103` `def forward(self, x, h_pre, w_norm)`
- `PhysioNeuron.__init__` (method) `physioneuron_cpu_v2.py:117` `def __init__(self, d_in, d_out, dynamic_mode)`
- `PhysioNeuron.forward` (method) `physioneuron_cpu_v2.py:128` `def forward(self, x)`
- `MicroContinuumCell.__init__` (method) `physioneuron_cpu_v2.py:159` `def __init__(self, dim)`
- `MicroContinuumCell.forward` (method) `physioneuron_cpu_v2.py:169` `def forward(self, x, plasticity)`
- `MicroSymbioticBasis.__init__` (method) `physioneuron_cpu_v2.py:184` `def __init__(self, dim, num_atoms)`
- `MicroSymbioticBasis.forward` (method) `physioneuron_cpu_v2.py:192` `def forward(self, x)`
- `MicroTopology.__init__` (method) `physioneuron_cpu_v2.py:204` `def __init__(self, num_nodes, config)`
- `MicroTopology.get_adjacency` (method) `physioneuron_cpu_v2.py:217` `def get_adjacency(self, plasticity)`
- `MicroSupConLoss.__init__` (method) `physioneuron_cpu_v2.py:224` `def __init__(self, temperature)`
- `MicroSupConLoss.forward` (method) `physioneuron_cpu_v2.py:229` `def forward(self, features, labels)`
- `MicroTopoBrain.__init__` (method) `physioneuron_cpu_v2.py:248` `def __init__(self, config)`
- `MicroTopoBrain.count_parameters` (method) `physioneuron_cpu_v2.py:290` `def count_parameters(self)`
- `MicroTopoBrain.forward` (method) `physioneuron_cpu_v2.py:293` `def forward(self, x, plasticity)`
- `MicroTopoBrain.micro_pgd_attack` (method) `physioneuron_cpu_v2.py:347` `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- `MicroTopoBrain.generate_ablation_matrix` (method) `physioneuron_cpu_v2.py:371` `def generate_ablation_matrix()`
- `MicroTopoBrain.train_with_cv` (method) `physioneuron_cpu_v2.py:400` `def train_with_cv(config, dataset, cv_folds)`
- `MicroTopoBrain.run_ablation_study` (method) `physioneuron_cpu_v2.py:464` `def run_ablation_study()`

## physioneuron_cpu_v3.py
- `EliteConfig.seed_everything` (method) `physioneuron_cpu_v3.py:71` `def seed_everything(seed)`
- `EliteConfig.get_elite_dataset` (method) `physioneuron_cpu_v3.py:79` `def get_elite_dataset(config)` -- Dataset más grande y balanceado con separabilidad controlada
- `EpisodicMemory.__init__` (method) `physioneuron_cpu_v3.py:104` `def __init__(self, dim, capacity)`
- `EpisodicMemory.update` (method) `physioneuron_cpu_v3.py:112` `def update(self, x, y)` -- Almacena ejemplos duros
- `EpisodicMemory.retrieve` (method) `physioneuron_cpu_v3.py:125` `def retrieve(self, x, k)` -- Recupera k vecinos más cercanos
- `SpectralNormLinear.__init__` (method) `physioneuron_cpu_v3.py:138` `def __init__(self, in_features, out_features)`
- `SpectralNormLinear.power_iteration` (method) `physioneuron_cpu_v3.py:145` `def power_iteration(self, n_iter)` -- Aproxima la norma espectral máxima
- `SpectralNormLinear.forward` (method) `physioneuron_cpu_v3.py:152` `def forward(self, x)`
- `AdvancedHomeostaticCell.__init__` (method) `physioneuron_cpu_v3.py:164` `def __init__(self, d_in, d_out, use_spectral)`
- `AdvancedHomeostaticCell.forward` (method) `physioneuron_cpu_v3.py:194` `def forward(self, x)`
- `AdaptiveTopology.__init__` (method) `physioneuron_cpu_v3.py:224` `def __init__(self, num_nodes, grid_size)`
- `AdaptiveTopology.forward` (method) `physioneuron_cpu_v3.py:248` `def forward(self, stress)` -- stress ∈ [0,1]: cuánto estrés adversarial
- `EliteTopoBrain.__init__` (method) `physioneuron_cpu_v3.py:261` `def __init__(self, config)`
- `EliteTopoBrain.count_parameters` (method) `physioneuron_cpu_v3.py:303` `def count_parameters(self)`
- `EliteTopoBrain.forward` (method) `physioneuron_cpu_v3.py:306` `def forward(self, x, stress)`
- `EliteTopoBrain.elite_pgd_attack` (method) `physioneuron_cpu_v3.py:348` `def elite_pgd_attack(model, x, y, eps, steps, stress)` -- PGD con reinicio aleatorio
- `SupConLoss.__init__` (method) `physioneuron_cpu_v3.py:386` `def __init__(self, temperature)`
- `SupConLoss.forward` (method) `physioneuron_cpu_v3.py:390` `def forward(self, features, labels)`
- `SupConLoss.train_elite_model` (method) `physioneuron_cpu_v3.py:430` `def train_elite_model(config, dataset, fold_results)` -- Entrenamiento con curriculum adversarial
- `SupConLoss.run_elite_experiment` (method) `physioneuron_cpu_v3.py:542` `def run_elite_experiment()`

## poke_cifar.py
- `compute_phi_effective` (function) `poke_cifar.py:35` `def compute_phi_effective(activity)`
- `PTSymmetricLayer.__init__` (method) `poke_cifar.py:57` `def __init__(self, in_features, out_features)`
- `PTSymmetricLayer.forward` (method) `poke_cifar.py:66` `def forward(self, x)`
- `TopologicalLayer.__init__` (method) `poke_cifar.py:78` `def __init__(self, in_f, out_f, density)`
- `TopologicalLayer.forward` (method) `poke_cifar.py:93` `def forward(self, x)`
- `DualSystemModule.__init__` (method) `poke_cifar.py:99` `def __init__(self, features)`
- `DualSystemModule.forward` (method) `poke_cifar.py:107` `def forward(self, x)`
- `ConsciousnessModule.__init__` (method) `poke_cifar.py:117` `def __init__(self, features)`
- `ConsciousnessModule.forward` (method) `poke_cifar.py:123` `def forward(self, x)`
- `OmniBrainCIFAR.__init__` (method) `poke_cifar.py:134` `def __init__(self)`
- `OmniBrainCIFAR.forward` (method) `poke_cifar.py:161` `def forward(self, x)`
- `OmniBrainCIFAR.get_cifar10_loaders` (method) `poke_cifar.py:180` `def get_cifar10_loaders(batch_size)`
- `OmniBrainCIFAR.evaluate` (method) `poke_cifar.py:200` `def evaluate(model, loader, device)`
- `OmniBrainCIFAR.main` (method) `poke_cifar.py:221` `def main()`
- `OmniBrainCIFAR.plot_history` (method) `poke_cifar.py:287` `def plot_history(hist)`
- `OmniBrainCIFAR.demo_inference` (method) `poke_cifar.py:300` `def demo_inference(model, loader)`

## poke_cifar2.py
- `compute_phi_effective` (function) `poke_cifar2.py:28` `def compute_phi_effective(activity)`
- `FastSlowLinear.__init__` (method) `poke_cifar2.py:50` `def __init__(self, in_features, out_features, fast_lr)`
- `FastSlowLinear.reset_fast_weights` (method) `poke_cifar2.py:65` `def reset_fast_weights(self)`
- `FastSlowLinear.update_fast_weights` (method) `poke_cifar2.py:69` `def update_fast_weights(self, x)`
- `FastSlowLinear.forward` (method) `poke_cifar2.py:76` `def forward(self, x)`
- `FastSlowLinear.end_of_batch` (method) `poke_cifar2.py:89` `def end_of_batch(self)`
- `FastSlowLinear.get_fast_weight_norm` (method) `poke_cifar2.py:92` `def get_fast_weight_norm(self)`
- `DualSystemModule.__init__` (method) `poke_cifar2.py:100` `def __init__(self, dim)`
- `DualSystemModule.forward` (method) `poke_cifar2.py:108` `def forward(self, x)`
- `ConsciousnessModule.__init__` (method) `poke_cifar2.py:118` `def __init__(self, dim)`
- `ConsciousnessModule.forward` (method) `poke_cifar2.py:124` `def forward(self, x)`
- `OmniBrainFastSlow.__init__` (method) `poke_cifar2.py:132` `def __init__(self)`
- `OmniBrainFastSlow.forward` (method) `poke_cifar2.py:151` `def forward(self, x)`
- `OmniBrainFastSlow.reset_all_fast_weights` (method) `poke_cifar2.py:159` `def reset_all_fast_weights(self)`
- `OmniBrainFastSlow.get_fast_norms` (method) `poke_cifar2.py:164` `def get_fast_norms(self)`
- `OmniBrainFastSlow.get_cifar10_loaders` (method) `poke_cifar2.py:175` `def get_cifar10_loaders(batch_size)`
- `OmniBrainFastSlow.evaluate` (method) `poke_cifar2.py:191` `def evaluate(model, loader, device)`
- `OmniBrainFastSlow.train` (method) `poke_cifar2.py:212` `def train()`

## pokemon3.py
- `compute_phi_effective_approx` (function) `pokemon3.py:30` `def compute_phi_effective_approx(activity)` -- Cálculo estable de Φₑ compatible con todas las versiones
- `estimate_energy_consumption` (function) `pokemon3.py:63` `def estimate_energy_consumption(model, batch_size)` -- Estimación conservadora de energía
- `PTSymmetricLayer.__init__` (method) `pokemon3.py:85` `def __init__(self, in_features, out_features)`
- `PTSymmetricLayer.compute_pt_phase` (method) `pokemon3.py:94` `def compute_pt_phase(self)`
- `PTSymmetricLayer.forward` (method) `pokemon3.py:102` `def forward(self, x)`
- `TopologicalLayer.__init__` (method) `pokemon3.py:112` `def __init__(self, in_features, out_features, connectivity)`
- `TopologicalLayer.update_topology` (method) `pokemon3.py:121` `def update_topology(self, connectivity)`
- `TopologicalLayer.forward` (method) `pokemon3.py:128` `def forward(self, x)`
- `DualSystemModule.__init__` (method) `pokemon3.py:137` `def __init__(self, features)`
- `DualSystemModule.forward` (method) `pokemon3.py:153` `def forward(self, x)`
- `ConsciousnessModule.__init__` (method) `pokemon3.py:167` `def __init__(self, features)`
- `ConsciousnessModule.forward` (method) `pokemon3.py:177` `def forward(self, x)`
- `OmniBrain.__init__` (method) `pokemon3.py:190` `def __init__(self, input_dim, hidden_dim, output_dim)`
- `OmniBrain.forward` (method) `pokemon3.py:217` `def forward(self, x)`
- `OmniBrain.update_topology` (method) `pokemon3.py:235` `def update_topology(self, current_connectivity)`
- `OmniBrain.prepare_mnist_data` (method) `pokemon3.py:245` `def prepare_mnist_data(batch_size, device)` -- Preparar datos MNIST con protección para entornos limitados
- `OmniBrain.train_omni_brain` (method) `pokemon3.py:269` `def train_omni_brain(model, train_loader, test_loader, epochs, device)` -- Entrenamiento compatible con todas las versiones de PyTorch
- `OmniBrain.evaluate_model` (method) `pokemon3.py:380` `def evaluate_model(model, test_loader, device, criterion)` -- Evaluación compatible con todas las versiones
- `OmniBrain.generate_evolution_plots` (method) `pokemon3.py:402` `def generate_evolution_plots(history, epochs)` -- Generar gráficos con protección para entornos sin GUI
- `OmniBrain.demonstrate_inference` (method) `pokemon3.py:435` `def demonstrate_inference(model, test_loader, device)` -- Demostración compatible con todas las versiones
- `OmniBrain.final_report` (method) `pokemon3.py:467` `def final_report(model, history)` -- Reporte final compatible

## pokemon4.py
- `compute_phi_effective` (function) `pokemon4.py:41` `def compute_phi_effective(activity)` -- Φₑ realista: fracción de varianza explicada por el primer componente PCA.
- `PTSymmetricLayer.__init__` (method) `pokemon4.py:69` `def __init__(self, in_features, out_features)`
- `PTSymmetricLayer.forward` (method) `pokemon4.py:78` `def forward(self, x)`
- `TopologicalLayer.__init__` (method) `pokemon4.py:92` `def __init__(self, in_features, out_features, target_density)`
- `TopologicalLayer.forward` (method) `pokemon4.py:107` `def forward(self, x)`
- `DualSystemModule.__init__` (method) `pokemon4.py:114` `def __init__(self, features)`
- `DualSystemModule.forward` (method) `pokemon4.py:130` `def forward(self, x)`
- `ConsciousnessModule.__init__` (method) `pokemon4.py:147` `def __init__(self, features)`
- `ConsciousnessModule.forward` (method) `pokemon4.py:157` `def forward(self, x)`
- `OmniBrain.__init__` (method) `pokemon4.py:169` `def __init__(self, input_dim, hidden_dim, output_dim)`
- `OmniBrain.forward` (method) `pokemon4.py:186` `def forward(self, x)`
- `OmniBrain.get_mnist_loaders` (method) `pokemon4.py:206` `def get_mnist_loaders(batch_size)`
- `OmniBrain.evaluate` (method) `pokemon4.py:218` `def evaluate(model, loader, device)`
- `OmniBrain.train_and_evaluate` (method) `pokemon4.py:236` `def train_and_evaluate()`
- `OmniBrain.plot_history` (method) `pokemon4.py:302` `def plot_history(hist)`

## pokemon_battle_champion.py
- `PokemonBattleChampion.__init__` (method) `pokemon_battle_champion.py:44` `def __init__(self, config)`
- `PokemonBattleChampion.forward` (method) `pokemon_battle_champion.py:99` `def forward(self, x)`
- `PokemonBattleChampion.create_battle_dataset` (method) `pokemon_battle_champion.py:128` `def create_battle_dataset(config)` -- Crear dataset para la batalla
- `PokemonBattleChampion.battle_training_epoch` (method) `pokemon_battle_champion.py:161` `def battle_training_epoch(model, loader, optimizer, criterion, epoch)` -- Entrenamiento de una época de batalla
- `PokemonBattleChampion.evaluate_battle_champion` (method) `pokemon_battle_champion.py:192` `def evaluate_battle_champion(model, loader)` -- Evaluar el campeón en batalla
- `PokemonBattleChampion.run_epic_pokemon_battle` (method) `pokemon_battle_champion.py:208` `def run_epic_pokemon_battle()` -- ¡EJECUTAR LA BATALLA ÉPICA!
- `PokemonBattleChampion.create_epic_battle_visualization` (method) `pokemon_battle_champion.py:334` `def create_epic_battle_visualization(battle_history, historical_results)` -- Crear visualización épica de la batalla
- `PokemonBattleChampion.save_battle_results` (method) `pokemon_battle_champion.py:440` `def save_battle_results(battle_history, historical_results, champion_model)` -- Guardar resultados de la batalla épica

## pokemon_hybrid_synergy_ablation.py
- `SynergyConfig.to_dict` (method) `pokemon_hybrid_synergy_ablation.py:75` `def to_dict(self)`
- `SynergyVAELayer.__init__` (method) `pokemon_hybrid_synergy_ablation.py:84` `def __init__(self, input_dim, hidden_dim, latent_dim)`
- `SynergyVAELayer.reparameterize` (method) `pokemon_hybrid_synergy_ablation.py:111` `def reparameterize(self, mu, logvar)`
- `SynergyVAELayer.forward` (method) `pokemon_hybrid_synergy_ablation.py:116` `def forward(self, x, return_encoding)`
- `SynergyAttentionLayer.__init__` (method) `pokemon_hybrid_synergy_ablation.py:127` `def __init__(self, d_model, num_heads, d_ff)`
- `SynergyAttentionLayer.forward` (method) `pokemon_hybrid_synergy_ablation.py:149` `def forward(self, x)`
- `SynergyGANLayer.__init__` (method) `pokemon_hybrid_synergy_ablation.py:159` `def __init__(self, input_dim, latent_dim, hidden_dim)`
- `SynergyGANLayer.generate` (method) `pokemon_hybrid_synergy_ablation.py:184` `def generate(self, z)`
- `SynergyGANLayer.discriminate` (method) `pokemon_hybrid_synergy_ablation.py:187` `def discriminate(self, x)`
- `AdaptiveTopologyLayer.__init__` (method) `pokemon_hybrid_synergy_ablation.py:192` `def __init__(self, grid_size, embed_dim, sparsity)`
- `AdaptiveTopologyLayer.get_adjacency_matrix` (method) `pokemon_hybrid_synergy_ablation.py:220` `def get_adjacency_matrix(self)`
- `AdaptiveTopologyLayer.forward` (method) `pokemon_hybrid_synergy_ablation.py:234` `def forward(self, x)`
- `PokemonSynergyModel.__init__` (method) `pokemon_hybrid_synergy_ablation.py:273` `def __init__(self, config)`
- `PokemonSynergyModel.forward` (method) `pokemon_hybrid_synergy_ablation.py:327` `def forward(self, x, return_all)`
- `SynergyAblationStudy.__init__` (method) `pokemon_hybrid_synergy_ablation.py:389` `def __init__(self, config)`
- `SynergyAblationStudy.get_ablation_matrix` (method) `pokemon_hybrid_synergy_ablation.py:393` `def get_ablation_matrix(self)` -- Matriz de ablación de 4 niveles:
- `SynergyAblationStudy.create_variant_model` (method) `pokemon_hybrid_synergy_ablation.py:424` `def create_variant_model(self, level_name)` -- Crear modelo variante para un nivel específico
- `BaselineModel.__init__` (method) `pokemon_hybrid_synergy_ablation.py:429` `def __init__(self, config)`
- `BaselineModel.forward` (method) `pokemon_hybrid_synergy_ablation.py:434` `def forward(self, x)`
- `HybridModel.__init__` (method) `pokemon_hybrid_synergy_ablation.py:444` `def __init__(self, config)`
- `HybridModel.forward` (method) `pokemon_hybrid_synergy_ablation.py:450` `def forward(self, x)`
- `AdvancedModel.__init__` (method) `pokemon_hybrid_synergy_ablation.py:464` `def __init__(self, config)`
- `AdvancedModel.forward` (method) `pokemon_hybrid_synergy_ablation.py:471` `def forward(self, x)`
- `AdvancedModel.run_synergy_ablation` (method) `pokemon_hybrid_synergy_ablation.py:495` `def run_synergy_ablation()` -- Ejecutar estudio de ablación completo
- `AdvancedModel.analyze_synergy_results` (method) `pokemon_hybrid_synergy_ablation.py:637` `def analyze_synergy_results(results)` -- Analizar resultados del estudio de sinergias
- `AdvancedModel.create_synergy_visualizations` (method) `pokemon_hybrid_synergy_ablation.py:696` `def create_synergy_visualizations(results, output_dir)` -- Crear visualizaciones del estudio de sinergias

## premium_synergy_demo.py
- `TopoBrainComponent.__init__` (method) `premium_synergy_demo.py:39` `def __init__(self)`
- `TopoBrainComponent.process` (method) `premium_synergy_demo.py:48` `def process(self, input_data, plasticity)` -- Procesamiento con autoregulación interna
- `OmniBrainComponent.__init__` (method) `premium_synergy_demo.py:78` `def __init__(self)`
- `OmniBrainComponent.process` (method) `premium_synergy_demo.py:87` `def process(self, input_data, chaos_level)` -- Procesamiento con control integrativo
- `QuimeraComponent.__init__` (method) `premium_synergy_demo.py:119` `def __init__(self)`
- `QuimeraComponent.process` (method) `premium_synergy_demo.py:128` `def process(self, input_data, plasticity, chaos)` -- Procesamiento con regulación de fases
- `HomeostaticMotor.__init__` (method) `premium_synergy_demo.py:163` `def __init__(self, threshold, convergence_epochs)`
- `HomeostaticMotor.deliberate` (method) `premium_synergy_demo.py:171` `def deliberate(self, components, target_accuracy)` -- Proceso de deliberación democrática
- `PremiumSynergySystem.__init__` (method) `premium_synergy_demo.py:228` `def __init__(self)`
- `PremiumSynergySystem.process_epoch` (method) `premium_synergy_demo.py:241` `def process_epoch(self, input_data, chaos_level)` -- Procesa una época del sistema democrático
- `PremiumSynergySystem.calculate_target_accuracy` (method) `premium_synergy_demo.py:296` `def calculate_target_accuracy(self)` -- Calcula accuracy objetivo basada en sinergia actual
- `PremiumSynergySystem.run_demo` (method) `premium_synergy_demo.py:302` `def run_demo()` -- Ejecuta demostración del sistema Premium Synergy

## premium_synergy_democratic.py
Imported by: `min_test_synergy.py`, `test_premium_synergy.py`
- `MemoryChecker.__init__` (method) `premium_synergy_democratic.py:97` `def __init__(self, max_memory_gb)`
- `MemoryChecker.check_memory` (method) `premium_synergy_democratic.py:101` `def check_memory(self)` -- Verifica el uso de memoria actual
- `MemoryChecker.warn_if_high` (method) `premium_synergy_democratic.py:126` `def warn_if_high(self)` -- Advierte si el uso de memoria es alto
- `TopoBrainComponent.__init__` (method) `premium_synergy_democratic.py:142` `def __init__(self, config)`
- `TopoBrainComponent.forward` (method) `premium_synergy_democratic.py:175` `def forward(self, x, plasticity)`
- `TopoBrainComponent.internal_dialogue` (method) `premium_synergy_democratic.py:222` `def internal_dialogue(self)` -- Diálogo interno fisiológico - metabolimo, sensibilidad, gating
- `OmniBrainComponent.__init__` (method) `premium_synergy_democratic.py:238` `def __init__(self, config)`
- `OmniBrainComponent.forward` (method) `premium_synergy_democratic.py:268` `def forward(self, x, chaos_level)`
- `OmniBrainComponent.internal_dialogue` (method) `premium_synergy_democratic.py:312` `def internal_dialogue(self)` -- Diálogo interno - balance integrativo y modulación caótica
- `QuimeraComponent.__init__` (method) `premium_synergy_democratic.py:328` `def __init__(self, config)`
- `QuimeraComponent.forward` (method) `premium_synergy_democratic.py:358` `def forward(self, x, plasticity, chaos)`
- `QuimeraComponent.internal_dialogue` (method) `premium_synergy_democratic.py:400` `def internal_dialogue(self)` -- Diálogo interno - regulación de fases y control atencional
- `QuimeraComponent.consolidate` (method) `premium_synergy_democratic.py:409` `def consolidate(self)` -- SVD consolidation de liquid neurons
- `MetabolismRegulator.__init__` (method) `premium_synergy_democratic.py:421` `def __init__(self, dim)`
- `MetabolismRegulator.forward` (method) `premium_synergy_democratic.py:431` `def forward(self, x)`
- `MetabolismRegulator.get_state` (method) `premium_synergy_democratic.py:456` `def get_state(self)`
- `SensitivityGate.__init__` (method) `premium_synergy_democratic.py:461` `def __init__(self, dim)`
- `SensitivityGate.forward` (method) `premium_synergy_democratic.py:471` `def forward(self, x)`
- `SensitivityGate.get_level` (method) `premium_synergy_democratic.py:489` `def get_level(self)`
- `DynamicTopologyGrid.__init__` (method) `premium_synergy_democratic.py:494` `def __init__(self, num_nodes, grid_size)`
- `DynamicTopologyGrid.get_adjacency` (method) `premium_synergy_democratic.py:514` `def get_adjacency(self, plasticity)`
- `SymbioticBasis.__init__` (method) `premium_synergy_democratic.py:521` `def __init__(self, dim, num_atoms)`
- `SymbioticBasis.forward` (method) `premium_synergy_democratic.py:531` `def forward(self, x)`
- `IntegrationModule.__init__` (method) `premium_synergy_democratic.py:542` `def __init__(self, dim)`
- `IntegrationModule.forward` (method) `premium_synergy_democratic.py:552` `def forward(self, x)`
- `IntegrationModule.get_level` (method) `premium_synergy_democratic.py:562` `def get_level(self)`
- `FastSlowLinear.__init__` (method) `premium_synergy_democratic.py:567` `def __init__(self, in_dim, out_dim)`
- `FastSlowLinear.forward` (method) `premium_synergy_democratic.py:579` `def forward(self, x)`
- `DualSystemModule.__init__` (method) `premium_synergy_democratic.py:597` `def __init__(self, dim)`
- `DualSystemModule.forward` (method) `premium_synergy_democratic.py:604` `def forward(self, x)`
- `DualSystemModule.get_balance` (method) `premium_synergy_democratic.py:612` `def get_balance(self)`
- `IntegrativeControl.__init__` (method) `premium_synergy_democratic.py:617` `def __init__(self, dim)`
- `IntegrativeControl.forward` (method) `premium_synergy_democratic.py:627` `def forward(self, x)`
- `IntegrativeControl.get_state` (method) `premium_synergy_democratic.py:645` `def get_state(self)`
- `ChaosModulator.__init__` (method) `premium_synergy_democratic.py:650` `def __init__(self, dim)`
- `ChaosModulator.forward` (method) `premium_synergy_democratic.py:660` `def forward(self, x, chaos_level)`
- `ChaosModulator.get_resistance` (method) `premium_synergy_democratic.py:678` `def get_resistance(self)`
- `LiquidNeuron.__init__` (method) `premium_synergy_democratic.py:684` `def __init__(self, in_dim, out_dim)`
- `LiquidNeuron.forward` (method) `premium_synergy_democratic.py:692` `def forward(self, x, plasticity)`
- `LiquidNeuron.consolidate_svd` (method) `premium_synergy_democratic.py:703` `def consolidate_svd(self, strength)`
- `SovereignAttention.__init__` (method) `premium_synergy_democratic.py:714` `def __init__(self, dim)`
- `SovereignAttention.forward` (method) `premium_synergy_democratic.py:721` `def forward(self, x, is_chaos)`
- `SovereignAttention.get_metrics` (method) `premium_synergy_democratic.py:733` `def get_metrics(self)`
- `DualPhaseMemory.__init__` (method) `premium_synergy_democratic.py:740` `def __init__(self, dim)`
- `DualPhaseMemory.forward` (method) `premium_synergy_democratic.py:746` `def forward(self, x, phase_idx)`
- `DualPhaseMemory.update` (method) `premium_synergy_democratic.py:754` `def update(self, x, phase_idx)`
- `DualPhaseMemory.get_coherence` (method) `premium_synergy_democratic.py:762` `def get_coherence(self)`
- `PhaseRegulator.__init__` (method) `premium_synergy_democratic.py:767` `def __init__(self, dim)`
- `PhaseRegulator.forward` (method) `premium_synergy_democratic.py:777` `def forward(self, x)`
- `PhaseRegulator.get_level` (method) `premium_synergy_democratic.py:795` `def get_level(self)`
- `AttentionController.__init__` (method) `premium_synergy_democratic.py:800` `def __init__(self, dim)`
- `AttentionController.forward` (method) `premium_synergy_democratic.py:810` `def forward(self, x, chaos)`
- `AttentionController.get_control` (method) `premium_synergy_democratic.py:829` `def get_control(self)`
- `HomeostaticMotor.__init__` (method) `premium_synergy_democratic.py:839` `def __init__(self, config)`
- `HomeostaticMotor.forward` (method) `premium_synergy_democratic.py:861` `def forward(self, topobrain_out, omnibrain_out, quimera_out, target_accuracy)` -- Cámara Alta: Delibera sobre las sinergias de los componentes
- `HomeostaticMotor.adjust_for_convergence` (method) `premium_synergy_democratic.py:926` `def adjust_for_convergence(self, performance_metrics)` -- Motor homeostático ajusta si las sinergias no convergen
- `PremiumSynergyModel.__init__` (method) `premium_synergy_democratic.py:963` `def __init__(self, config)`
- `PremiumSynergyModel.forward` (method) `premium_synergy_democratic.py:989` `def forward(self, x, chaos_level)` -- Forward pass completo con sistema democrático
- `PremiumSynergyModel.democratic_deliberation_status` (method) `premium_synergy_democratic.py:1061` `def democratic_deliberation_status(self)` -- Estado de la deliberación democrática
- `SystemRegulator.__init__` (method) `premium_synergy_democratic.py:1076` `def __init__(self, dim)`
- `SystemRegulator.forward` (method) `premium_synergy_democratic.py:1086` `def forward(self, x)`
- `SystemRegulator.ensure_dependencies` (method) `premium_synergy_democratic.py:1108` `def ensure_dependencies()` -- Asegura que las dependencias estén instaladas
- `SystemRegulator.create_synthetic_dataset` (method) `premium_synergy_democratic.py:1118` `def create_synthetic_dataset(config)` -- Crea dataset sintético para testing
- `SystemRegulator.train_premium_synergy` (method) `premium_synergy_democratic.py:1148` `def train_premium_synergy(config)` -- Entrena el modelo Premium Synergy
- `SystemRegulator.create_dataloader` (method) `premium_synergy_democratic.py:1272` `def create_dataloader(X, y, batch_size, shuffle)` -- Crea dataloader
- `SystemRegulator.main` (method) `premium_synergy_democratic.py:1284` `def main()` -- Función principal

## quen7.py
- `Config.seed_everything` (method) `quen7.py:45` `def seed_everything(seed)`
- `DataEnvironment.__init__` (method) `quen7.py:56` `def __init__(self)`
- `DataEnvironment.get_batch` (method) `quen7.py:66` `def get_batch(self, phase, bs)`
- `DataEnvironment.get_full` (method) `quen7.py:80` `def get_full(self)`
- `DataEnvironment.get_w2` (method) `quen7.py:83` `def get_w2(self)`
- `HomeostaticRegulator.__init__` (method) `quen7.py:90` `def __init__(self, d_in)`
- `HomeostaticRegulator.forward` (method) `quen7.py:100` `def forward(self, x, h_pre, w_norm)`
- `PhysioNeuron.__init__` (method) `quen7.py:116` `def __init__(self, d_in, d_out, dynamic)`
- `PhysioNeuron.forward` (method) `quen7.py:128` `def forward(self, x)`
- `SupConHead.__init__` (method) `quen7.py:159` `def __init__(self, in_dim)`
- `SupConHead.forward` (method) `quen7.py:167` `def forward(self, x)`
- `MicroTopoBrain.__init__` (method) `quen7.py:174` `def __init__(self, config)`
- `MicroTopoBrain.count_parameters` (method) `quen7.py:187` `def count_parameters(self)`
- `MicroTopoBrain.forward` (method) `quen7.py:190` `def forward(self, x)`
- `NeuralDiagnostics.__init__` (method) `quen7.py:216` `def __init__(self)`
- `NeuralDiagnostics.update` (method) `quen7.py:226` `def update(self, loss, liquid_norm, physio, prediction_error)`
- `NeuralDiagnostics.get_recent_avg` (method) `quen7.py:234` `def get_recent_avg(self, key, n)`
- `NeuralDiagnostics.report` (method) `quen7.py:239` `def report(self, step, phase)`
- `NeuralDiagnostics.train_nonstationary` (method) `quen7.py:265` `def train_nonstationary(config)`
- `NeuralDiagnostics.run_ablation_study` (method) `quen7.py:330` `def run_ablation_study()`

## quimera.py
- `ChimeraScientificConfig.seed_everything` (method) `quimera.py:37` `def seed_everything(seed)`
- `ChimeraScientificConfig.measure_spatial_richness` (method) `quimera.py:47` `def measure_spatial_richness(activations)` -- Mide la diversidad espacial de las activaciones (Richness)
- `ChimeraScientificConfig.get_structure_entropy` (method) `quimera.py:62` `def get_structure_entropy(model)` -- Mide la entropía estructural de los pesos (Entropy)
- `RealWorldEnvironment.__init__` (method) `quimera.py:87` `def __init__(self)`
- `RealWorldEnvironment.get_batch` (method) `quimera.py:97` `def get_batch(self, phase, batch_size)`
- `LiquidNeuron.__init__` (method) `quimera.py:116` `def __init__(self, in_dim, out_dim)`
- `LiquidNeuron.forward` (method) `quimera.py:124` `def forward(self, x, plasticity)`
- `LiquidNeuron.consolidate_svd` (method) `quimera.py:138` `def consolidate_svd(self, strength)` -- Mecanismo de SVD (Science-ready)
- `SovereignAttention.__init__` (method) `quimera.py:155` `def __init__(self, dim)`
- `SovereignAttention.forward` (method) `quimera.py:165` `def forward(self, x, is_chaos)`
- `DualPhaseMemory.__init__` (method) `quimera.py:178` `def __init__(self, dim)`
- `DualPhaseMemory.forward` (method) `quimera.py:184` `def forward(self, x, phase_idx)`
- `DualPhaseMemory.update` (method) `quimera.py:191` `def update(self, x, phase_idx)`
- `Chimera_v9_Scientific.__init__` (method) `quimera.py:202` `def __init__(self, config)`
- `Chimera_v9_Scientific.forward` (method) `quimera.py:229` `def forward(self, x, phase_idx)`
- `Chimera_v9_Scientific.consolidate` (method) `quimera.py:267` `def consolidate(self)`
- `Chimera_v9_Scientific.train_chimera_scientific` (method) `quimera.py:279` `def train_chimera_scientific(config, verbose)`
- `Chimera_v9_Scientific.generate_chimera_matrix` (method) `quimera.py:361` `def generate_chimera_matrix()`
- `Chimera_v9_Scientific.run_scientific_study` (method) `quimera.py:399` `def run_scientific_study()`


Next: [API_p8.md](API_p8.md)

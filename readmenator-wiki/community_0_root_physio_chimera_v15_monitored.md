# root: physio_chimera_v15_monitored

*Community 0 | 3 files | cohesion 1.00*

## Definition

This community groups 3 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `Config`, `ContinuumMemorySystem`, `DataEnvironment`, `MetricsVisualizer`, `NestedPhysioNeuron`, `NeuralDiagnostics`, `PhysioChimeraNested`, `SelfModifyingGates`. Core file: `physio_chimera_v15_monitored.py` (35 symbols). Documented purpose: Ejemplo de uso del sistema Physio-Chimera v15 Monitoreado.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `example_usage.py` | py | utility | 6 | yes |
| `physio_chimera_v15_monitored.py` | py | utility | 35 | yes |
| `run_complete_experiment.py` | py | utility | 3 | yes |

## Key Symbols

- `demo_simple_monitoring` (function, `example_usage.py:20`) `def demo_simple_monitoring()` - Demostración de monitoreo básico
- `demo_custom_monitoring` (function, `example_usage.py:38`) `def demo_custom_monitoring()` - Demostración de monitoreo personalizado
- `demo_checkpoint_system` (function, `example_usage.py:85`) `def demo_checkpoint_system()` - Demostración del sistema de checkpointing
- `demo_comparison_experiments` (function, `example_usage.py:142`) `def demo_comparison_experiments()` - Demostración de comparación entre experimentos
- `create_demo_report` (function, `example_usage.py:194`) `def create_demo_report()` - Crear reporte demo completo
- `main` (function, `example_usage.py:333`) `def main()` - Función principal de demostración
- `Config` (class, `physio_chimera_v15_monitored.py:42`) `class Config`
- `seed_everything` (method, `physio_chimera_v15_monitored.py:58`) `def seed_everything(seed)`
- `DataEnvironment` (class, `physio_chimera_v15_monitored.py:68`) `class DataEnvironment`
- `__init__` (method, `physio_chimera_v15_monitored.py:69`) `def __init__(self)`
- `get_batch` (method, `physio_chimera_v15_monitored.py:79`) `def get_batch(self, phase, bs)`
- `get_full` (method, `physio_chimera_v15_monitored.py:93`) `def get_full(self)` - Retorna el dataset completo
- `get_w2` (method, `physio_chimera_v15_monitored.py:97`) `def get_w2(self)` - Retorna solo los datos de WORLD_2 (dígitos >= 5)
- `NeuralDiagnostics` (class, `physio_chimera_v15_monitored.py:104`) `class NeuralDiagnostics` - Sistema de diagnóstico neurológico para Physio-Chimera
- `__init__` (method, `physio_chimera_v15_monitored.py:107`) `def __init__(self, config)`
- `update_physio_metrics` (method, `physio_chimera_v15_monitored.py:144`) `def update_physio_metrics(self, metabolism, sensitivity, gate)` - Actualiza métricas fisiológicas
- `update_performance_metrics` (method, `physio_chimera_v15_monitored.py:150`) `def update_performance_metrics(self, loss, accuracy, lr)` - Actualiza métricas de rendimiento
- `update_memory_metrics` (method, `physio_chimera_v15_monitored.py:158`) `def update_memory_metrics(self, cms_activations, hebbian_norm, forgetting_factor` - Actualiza métricas de memoria
- `calculate_health_metrics` (method, `physio_chimera_v15_monitored.py:166`) `def calculate_health_metrics(self)` - Calcula métricas de salud del sistema
- `get_recent_avg` (method, `physio_chimera_v15_monitored.py:191`) `def get_recent_avg(self, category, key, n)` - Obtiene promedio reciente de una métrica
- `generate_diagnostic_report` (method, `physio_chimera_v15_monitored.py:209`) `def generate_diagnostic_report(self, step, phase)` - Genera reporte de diagnóstico
- `save_metrics` (method, `physio_chimera_v15_monitored.py:279`) `def save_metrics(self, filepath)` - Guarda todas las métricas
- `SelfModifyingGates` (class, `physio_chimera_v15_monitored.py:298`) `class SelfModifyingGates(Module)`
- `__init__` (method, `physio_chimera_v15_monitored.py:299`) `def __init__(self, input_dim, hidden_dim)`
- `forward` (method, `physio_chimera_v15_monitored.py:306`) `def forward(self, x)`
- `ContinuumMemorySystem` (class, `physio_chimera_v15_monitored.py:319`) `class ContinuumMemorySystem(Module)`
- `__init__` (method, `physio_chimera_v15_monitored.py:320`) `def __init__(self, levels, d_model, hidden_dim)`
- `forward` (method, `physio_chimera_v15_monitored.py:332`) `def forward(self, x, global_step)`
- `NestedPhysioNeuron` (class, `physio_chimera_v15_monitored.py:346`) `class NestedPhysioNeuron(Module)`
- `__init__` (method, `physio_chimera_v15_monitored.py:347`) `def __init__(self, d_in, d_out, config)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 2
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 1 (strength 0.5): Inferred shared context (language py) with no import path between community 0 (root: physio_chimera_v15_monitored) and community 1 (root: premium_synergy_democratic).
- [INFERRED] shares_context community 0 <-> 2 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (root: physio_chimera_v15_monitored) and community 2 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- What would break if the most connected file in root: physio_chimera_v15_monitored changed?
- Should root: physio_chimera_v15_monitored be split, given cohesion 1.00?

## Sources

- `example_usage.py`
- `physio_chimera_v15_monitored.py`
- `run_complete_experiment.py`

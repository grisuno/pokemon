# root: premium_synergy_democratic

*Community 1 | 3 files | cohesion 1.00*

## Definition

This community groups 3 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `AttentionController`, `ChaosModulator`, `DualPhaseMemory`, `DualSystemModule`, `DynamicTopologyGrid`, `FastSlowLinear`, `HomeostaticMotor`, `IntegrationModule`. Core file: `premium_synergy_democratic.py` (89 symbols). Documented purpose: Ejecutar Premium Synergy con MNIST real Para validar experimentalmente la arquitectura democrática.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `min_test_synergy.py` | py | testing | 0 | yes |
| `premium_synergy_democratic.py` | py | utility | 89 | yes |
| `test_premium_synergy.py` | py | testing | 4 | yes |

## Key Symbols

- `PremiumSynergyConfig` (class, `premium_synergy_democratic.py:42`) `class PremiumSynergyConfig` - Configuración del sistema Premium Synergy
- `MemoryChecker` (class, `premium_synergy_democratic.py:94`) `class MemoryChecker` - Sistema de monitoreo de memoria
- `__init__` (method, `premium_synergy_democratic.py:97`) `def __init__(self, max_memory_gb)`
- `check_memory` (method, `premium_synergy_democratic.py:101`) `def check_memory(self)` - Verifica el uso de memoria actual
- `warn_if_high` (method, `premium_synergy_democratic.py:126`) `def warn_if_high(self)` - Advierte si el uso de memoria es alto
- `TopoBrainComponent` (class, `premium_synergy_democratic.py:139`) `class TopoBrainComponent(Module)` - TopoBrain v8 con autoregulación interna
- `__init__` (method, `premium_synergy_democratic.py:142`) `def __init__(self, config)`
- `forward` (method, `premium_synergy_democratic.py:175`) `def forward(self, x, plasticity)`
- `internal_dialogue` (method, `premium_synergy_democratic.py:222`) `def internal_dialogue(self)` - Diálogo interno fisiológico - metabolimo, sensibilidad, gating
- `OmniBrainComponent` (class, `premium_synergy_democratic.py:235`) `class OmniBrainComponent(Module)` - OmniBrain K con autoregulación interna
- `__init__` (method, `premium_synergy_democratic.py:238`) `def __init__(self, config)`
- `forward` (method, `premium_synergy_democratic.py:268`) `def forward(self, x, chaos_level)`
- `internal_dialogue` (method, `premium_synergy_democratic.py:312`) `def internal_dialogue(self)` - Diálogo interno - balance integrativo y modulación caótica
- `QuimeraComponent` (class, `premium_synergy_democratic.py:325`) `class QuimeraComponent(Module)` - Quimera v9.5 con autoregulación interna
- `__init__` (method, `premium_synergy_democratic.py:328`) `def __init__(self, config)`
- `forward` (method, `premium_synergy_democratic.py:358`) `def forward(self, x, plasticity, chaos)`
- `internal_dialogue` (method, `premium_synergy_democratic.py:400`) `def internal_dialogue(self)` - Diálogo interno - regulación de fases y control atencional
- `consolidate` (method, `premium_synergy_democratic.py:409`) `def consolidate(self)` - SVD consolidation de liquid neurons
- `MetabolismRegulator` (class, `premium_synergy_democratic.py:419`) `class MetabolismRegulator(Module)` - Regulador de metabolismo para TopoBrain
- `__init__` (method, `premium_synergy_democratic.py:421`) `def __init__(self, dim)`
- `forward` (method, `premium_synergy_democratic.py:431`) `def forward(self, x)`
- `get_state` (method, `premium_synergy_democratic.py:456`) `def get_state(self)`
- `SensitivityGate` (class, `premium_synergy_democratic.py:459`) `class SensitivityGate(Module)` - Compuerta de sensibilidad para TopoBrain
- `__init__` (method, `premium_synergy_democratic.py:461`) `def __init__(self, dim)`
- `forward` (method, `premium_synergy_democratic.py:471`) `def forward(self, x)`
- `get_level` (method, `premium_synergy_democratic.py:489`) `def get_level(self)`
- `DynamicTopologyGrid` (class, `premium_synergy_democratic.py:492`) `class DynamicTopologyGrid(Module)` - Topología dinámica para TopoBrain
- `__init__` (method, `premium_synergy_democratic.py:494`) `def __init__(self, num_nodes, grid_size)`
- `_create_grid_mask` (method, `premium_synergy_democratic.py:501`) `def _create_grid_mask(self)`
- `get_adjacency` (method, `premium_synergy_democratic.py:514`) `def get_adjacency(self, plasticity)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 2
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 1 (strength 0.5): Inferred shared context (language py) with no import path between community 0 (root: physio_chimera_v15_monitored) and community 1 (root: premium_synergy_democratic).
- [INFERRED] shares_context community 1 <-> 2 (strength 0.5): Inferred shared context (language py) with no import path between community 1 (root: premium_synergy_democratic) and community 2 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- What would break if the most connected file in root: premium_synergy_democratic changed?
- Should root: premium_synergy_democratic be split, given cohesion 1.00?

## Sources

- `min_test_synergy.py`
- `premium_synergy_democratic.py`
- `test_premium_synergy.py`

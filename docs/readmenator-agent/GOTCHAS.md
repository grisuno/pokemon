# Gotchas

## God Nodes (high connectivity)

These files have the most connections. Changes here have high blast radius.

- `premium_synergy_democratic.py` (score: 12.90, imported by 2 files)
- `exodia_op_2.py` (score: 10.30)
- `exodia_optimized.py` (score: 10.20)
- `neurologos_tricameral_exodia.py` (score: 10.10)
- `bicameral_v3.py` (score: 8.50)
- `tricameral_kimi2.py` (score: 8.50)
- `nestedtopobrain.py` (score: 8.40)
- `main5.py` (score: 8.10)
- `bicameral_v2.py` (score: 8.00)
- `nestedtopobrain_v1.py` (score: 7.80)

## Blast Radius (change impact)

Editing these files can break the listed number of dependents. Run their tests after any change.

- `physio_chimera_v15_monitored.py` -- 2 direct, 2 total dependents
- `premium_synergy_democratic.py` -- 2 direct, 2 total dependents

## Hotspots (complexity + centrality)

- `exodia_op_2.py` -- complexity: 1.0, centrality: 1.0, combined: 1.0
- `exodia_optimized.py` -- complexity: 1.0, centrality: 1.0, combined: 1.0
- `neurologos_tricameral_exodia.py` -- complexity: 1.0, centrality: 0.9, combined: 0.9
- `tricameral_kimi2.py` -- complexity: 0.8, centrality: 0.9, combined: 0.9
- `nestedtopobrain.py` -- complexity: 0.8, centrality: 0.8, combined: 0.8
- `nestedtopobrain_v1.py` -- complexity: 0.8, centrality: 0.8, combined: 0.8
- `nestedtopobrain_v2.py` -- complexity: 0.8, centrality: 0.8, combined: 0.8
- `nestedtopobrain_v3.py` -- complexity: 0.8, centrality: 0.8, combined: 0.8
- `tricameral_kimi.py` -- complexity: 0.7, centrality: 0.8, combined: 0.7
- `topobrain_v19.py` -- complexity: 0.4, centrality: 0.9, combined: 0.7

## Dataflow Issues (INFERRED, review each lead)

- `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py:750` `__getitem__` [UNCHECKED_ALLOC] `image`: Result of allocator stored in `image` is never checked against NULL.
- `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py:914` `__getitem__` [UNCHECKED_ALLOC] `image`: Result of allocator stored in `image` is never checked against NULL.
- `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py:977` `__getitem__` [UNCHECKED_ALLOC] `image`: Result of allocator stored in `image` is never checked against NULL.
- `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py:1262` `__getitem__` [UNCHECKED_ALLOC] `image`: Result of allocator stored in `image` is never checked against NULL.
- `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py:1731` `__getitem__` [UNCHECKED_ALLOC] `image`: Result of allocator stored in `image` is never checked against NULL.
- `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py:2017` `__getitem__` [UNCHECKED_ALLOC] `image`: Result of allocator stored in `image` is never checked against NULL.
- `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py:1431` `__getitem__` [UNCHECKED_ALLOC] `image`: Result of allocator stored in `image` is never checked against NULL.
- `bicamera.py.py:429` `__getitem__` [UNCHECKED_ALLOC] `image`: Result of allocator stored in `image` is never checked against NULL.
- `bicameral.py:405` `__getitem__` [UNCHECKED_ALLOC] `image`: Result of allocator stored in `image` is never checked against NULL.
- `bicameral2.py:264` `__getitem__` [UNCHECKED_ALLOC] `image`: Result of allocator stored in `image` is never checked against NULL.

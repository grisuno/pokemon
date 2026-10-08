# Second Brain

*Last synthesized: 2026-10-07 | 190 files | 3 concept pages | offline, zero tokens*

> Raw sources -> readmenator wiki -> links (Karpathy LLM Wiki Pattern, deterministic).
> Start here, then open one community page. Prefer grep over full reads.

## Vault Overview

The codebase centres on `premium_synergy_democratic.py`, `exodia_op_2.py`, `exodia_optimized.py`. Architecturally it is 4 layers, dominant utility (183 files) across 3 import-based communities. Recorded risk surface: 0 security findings and 0 dependency cycles.

Surprising tissue lives between root: physio_chimera_v15_monitored, root: premium_synergy_democratic, orphans: 0 extracted cross-community imports and 3 inferred bridges. Follow `connections.json` sorted by strength before refactoring.

Open work clusters around documentation (73% file coverage), 0 security findings, 20 taint paths, and 5 suggested exploration questions in `queries.md`.

## Stats

| Metric | Value |
|--------|-------|
| Files | 190 |
| Symbols | 5935 |
| Resolved imports | 4 |
| Languages | py, sh |
| Communities | 3 |
| Doc coverage | 73% (138/190 files) |
| Security findings | 0 |
| Estimated read cost | ~59764 tokens (chars/4, offline so $0) |

## Reading Order

1. Skim Stats and God Nodes below for blast radius.
2. Open the largest community page first, then follow Connections.
3. Use `queries.md` for the next question; log the answer there.

```
grep -rn '<keyword>' index.md community_*.md
readmenator query "<question>" --target readmenator_pokemon_p5q4e_9n
```

## Concept Wiki

- [root: physio_chimera_v15_monitored (3 files, cohesion 1.00)](./community_0_root_physio_chimera_v15_monitored.md)
- [root: premium_synergy_democratic (3 files, cohesion 1.00)](./community_1_root_premium_synergy_democratic.md)
- [orphans (184 files, cohesion 0.00)](./community_2_orphans.md)

## God Nodes

| File | Score |
|------|-------|
| `premium_synergy_democratic.py` | 12.9 |
| `exodia_op_2.py` | 10.3 |
| `exodia_optimized.py` | 10.2 |
| `neurologos_tricameral_exodia.py` | 10.1 |
| `bicameral_v3.py` | 8.5 |

## Strongest Connections

- 0 -> 1: shares_context (strength 0.5, INFERRED)
- 0 -> 2: shares_context (strength 0.5, INFERRED)
- 1 -> 2: shares_context (strength 0.5, INFERRED)

## Navigation Tips

- Obsidian Graph View works: every community page links back here.
- `connections.json` is machine-readable for GraphRAG pipelines.
- `REPORT.md` states what was extracted vs inferred and current limits.
- Regenerate offline: `readmenator . --rebuild` (no network, no tokens).

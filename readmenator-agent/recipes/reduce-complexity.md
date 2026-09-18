# Recipe: Reduce File Complexity

Target hotspot: `exodia_op_2.py`
(complexity 1.0, centrality 1.0)

1. Read dependents: `grep -n 'exodia_op_2.py' readmenator-agent/ARCHITECTURE.md`
2. Extract functions/classes into new files in the same subsystem
3. Update imports
4. Regenerate: `readmenator .`

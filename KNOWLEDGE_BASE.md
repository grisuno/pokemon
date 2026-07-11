# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis.

**Total Files Parsed:** 190 | **Total Symbols Extracted:** 5935 | **Total Imports:** 2457

## Structural Knowledge Map
> **Note:** The visual graph below has been intelligently pruned to the top 300 most relevant nodes to prevent rendering crashes. Full details of all 190 files are documented below.

```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray: 5 5,color:#aaa;
    exodia_op_2_py["exodia_op_2.py (py)"]
    class exodia_op_2_py mod;
    exodia_op_2_py_preprocess_and_cache_spectrograms["preprocess_and_cache_spectrograms"]
    class exodia_op_2_py_preprocess_and_cache_spectrograms fn;
    exodia_op_2_py --> exodia_op_2_py_preprocess_and_cache_spectrograms
    exodia_op_2_py_apply_emergency_fixes["apply_emergency_fixes"]
    class exodia_op_2_py_apply_emergency_fixes fn;
    exodia_op_2_py --> exodia_op_2_py_apply_emergency_fixes
    exodia_op_2_py_setup_flickr8k_with_audio["setup_flickr8k_with_audio"]
    class exodia_op_2_py_setup_flickr8k_with_audio fn;
    exodia_op_2_py --> exodia_op_2_py_setup_flickr8k_with_audio
    exodia_op_2_py_build_vocab_flickr["build_vocab_flickr"]
    class exodia_op_2_py_build_vocab_flickr fn;
    exodia_op_2_py --> exodia_op_2_py_build_vocab_flickr
    exodia_op_2_py_HierarchicalEpisodicMemory["HierarchicalEpisodicMemory"]
    class exodia_op_2_py_HierarchicalEpisodicMemory cls;
    exodia_op_2_py --> exodia_op_2_py_HierarchicalEpisodicMemory
    exodia_optimized_py["exodia_optimized.py (py)"]
    class exodia_optimized_py mod;
    exodia_optimized_py_preprocess_and_cache_spectrograms["preprocess_and_cache_spectrograms"]
    class exodia_optimized_py_preprocess_and_cache_spectrograms fn;
    exodia_optimized_py --> exodia_optimized_py_preprocess_and_cache_spectrograms
    exodia_optimized_py_setup_flickr8k_with_audio["setup_flickr8k_with_audio"]
    class exodia_optimized_py_setup_flickr8k_with_audio fn;
    exodia_optimized_py --> exodia_optimized_py_setup_flickr8k_with_audio
    exodia_optimized_py_build_vocab_flickr["build_vocab_flickr"]
    class exodia_optimized_py_build_vocab_flickr fn;
    exodia_optimized_py --> exodia_optimized_py_build_vocab_flickr
    exodia_optimized_py_HierarchicalEpisodicMemory["HierarchicalEpisodicMemory"]
    class exodia_optimized_py_HierarchicalEpisodicMemory cls;
    exodia_optimized_py --> exodia_optimized_py_HierarchicalEpisodicMemory
    exodia_optimized_py_NeurocognitiveSystem["NeurocognitiveSystem"]
    class exodia_optimized_py_NeurocognitiveSystem cls;
    exodia_optimized_py --> exodia_optimized_py_NeurocognitiveSystem
    topobrain_v19_py["topobrain_v19.py (py)"]
    class topobrain_v19_py mod;
    topobrain_v19_py_Config["Config"]
    class topobrain_v19_py_Config cls;
    topobrain_v19_py --> topobrain_v19_py_Config
    topobrain_v19_py_seed_everything["seed_everything"]
    class topobrain_v19_py_seed_everything fn;
    topobrain_v19_py --> topobrain_v19_py_seed_everything
    topobrain_v19_py_ResourceMonitor["ResourceMonitor"]
    class topobrain_v19_py_ResourceMonitor cls;
    topobrain_v19_py --> topobrain_v19_py_ResourceMonitor
    topobrain_v19_py_CheckpointManager["CheckpointManager"]
    class topobrain_v19_py_CheckpointManager cls;
    topobrain_v19_py --> topobrain_v19_py_CheckpointManager
    topobrain_v19_py_get_dataset_stats["get_dataset_stats"]
    class topobrain_v19_py_get_dataset_stats fn;
    topobrain_v19_py --> topobrain_v19_py_get_dataset_stats
    neurologos_tricameral_exodia_py["neurologos_tricameral_exodia.py (py)"]
    class neurologos_tricameral_exodia_py mod;
    neurologos_tricameral_exodia_py_preprocess_and_cache_spectrograms["preprocess_and_cache_spectrograms"]
    class neurologos_tricameral_exodia_py_preprocess_and_cache_spectrograms fn;
    neurologos_tricameral_exodia_py --> neurologos_tricameral_exodia_py_preprocess_and_cache_spectrograms
    neurologos_tricameral_exodia_py_setup_flickr8k_with_audio["setup_flickr8k_with_audio"]
    class neurologos_tricameral_exodia_py_setup_flickr8k_with_audio fn;
    neurologos_tricameral_exodia_py --> neurologos_tricameral_exodia_py_setup_flickr8k_with_audio
    neurologos_tricameral_exodia_py_build_vocab_flickr["build_vocab_flickr"]
    class neurologos_tricameral_exodia_py_build_vocab_flickr fn;
    neurologos_tricameral_exodia_py --> neurologos_tricameral_exodia_py_build_vocab_flickr
    neurologos_tricameral_exodia_py_HierarchicalEpisodicMemory["HierarchicalEpisodicMemory"]
    class neurologos_tricameral_exodia_py_HierarchicalEpisodicMemory cls;
    neurologos_tricameral_exodia_py --> neurologos_tricameral_exodia_py_HierarchicalEpisodicMemory
    neurologos_tricameral_exodia_py_NeurocognitiveSystem["NeurocognitiveSystem"]
    class neurologos_tricameral_exodia_py_NeurocognitiveSystem cls;
    neurologos_tricameral_exodia_py --> neurologos_tricameral_exodia_py_NeurocognitiveSystem
    tricameral_kimi2_py["tricameral_kimi2.py (py)"]
    class tricameral_kimi2_py mod;
    tricameral_kimi2_py_setup_flickr8k_with_audio["setup_flickr8k_with_audio"]
    class tricameral_kimi2_py_setup_flickr8k_with_audio fn;
    tricameral_kimi2_py --> tricameral_kimi2_py_setup_flickr8k_with_audio
    tricameral_kimi2_py_build_vocab_flickr["build_vocab_flickr"]
    class tricameral_kimi2_py_build_vocab_flickr fn;
    tricameral_kimi2_py --> tricameral_kimi2_py_build_vocab_flickr
    tricameral_kimi2_py_EpisodicMemoryBuffer["EpisodicMemoryBuffer"]
    class tricameral_kimi2_py_EpisodicMemoryBuffer cls;
    tricameral_kimi2_py --> tricameral_kimi2_py_EpisodicMemoryBuffer
    tricameral_kimi2_py_NeurocognitiveSystem["NeurocognitiveSystem"]
    class tricameral_kimi2_py_NeurocognitiveSystem cls;
    tricameral_kimi2_py --> tricameral_kimi2_py_NeurocognitiveSystem
    tricameral_kimi2_py_LanguageMetrics["LanguageMetrics"]
    class tricameral_kimi2_py_LanguageMetrics cls;
    tricameral_kimi2_py --> tricameral_kimi2_py_LanguageMetrics
    tricameral2_py["tricameral2.py (py)"]
    class tricameral2_py mod;
    tricameral2_py_generate_audio_async["generate_audio_async"]
    class tricameral2_py_generate_audio_async fn;
    tricameral2_py --> tricameral2_py_generate_audio_async
    tricameral2_py_generate_all_audios_batch["generate_all_audios_batch"]
    class tricameral2_py_generate_all_audios_batch fn;
    tricameral2_py --> tricameral2_py_generate_all_audios_batch
    tricameral2_py_generate_audios_sync["generate_audios_sync"]
    class tricameral2_py_generate_audios_sync fn;
    tricameral2_py --> tricameral2_py_generate_audios_sync
    tricameral2_py_download_from_github["download_from_github"]
    class tricameral2_py_download_from_github fn;
    tricameral2_py --> tricameral2_py_download_from_github
    tricameral2_py_setup_flickr8k["setup_flickr8k"]
    class tricameral2_py_setup_flickr8k fn;
    tricameral2_py --> tricameral2_py_setup_flickr8k
    nestedtopobrain_py["nestedtopobrain.py (py)"]
    class nestedtopobrain_py mod;
    nestedtopobrain_py_Config["Config"]
    class nestedtopobrain_py_Config cls;
    nestedtopobrain_py --> nestedtopobrain_py_Config
    nestedtopobrain_py_seed_everything["seed_everything"]
    class nestedtopobrain_py_seed_everything fn;
    nestedtopobrain_py --> nestedtopobrain_py_seed_everything
    nestedtopobrain_py_ResourceMonitor["ResourceMonitor"]
    class nestedtopobrain_py_ResourceMonitor cls;
    nestedtopobrain_py --> nestedtopobrain_py_ResourceMonitor
    nestedtopobrain_py_PrefrontalOrchestrator["PrefrontalOrchestrator"]
    class nestedtopobrain_py_PrefrontalOrchestrator cls;
    nestedtopobrain_py --> nestedtopobrain_py_PrefrontalOrchestrator
    nestedtopobrain_py_TopologyMetrics["TopologyMetrics"]
    class nestedtopobrain_py_TopologyMetrics cls;
    nestedtopobrain_py --> nestedtopobrain_py_TopologyMetrics
    nestedtopobrain_v1_py["nestedtopobrain_v1.py (py)"]
    class nestedtopobrain_v1_py mod;
    nestedtopobrain_v1_py_Config["Config"]
    class nestedtopobrain_v1_py_Config cls;
    nestedtopobrain_v1_py --> nestedtopobrain_v1_py_Config
    nestedtopobrain_v1_py_seed_everything["seed_everything"]
    class nestedtopobrain_v1_py_seed_everything fn;
    nestedtopobrain_v1_py --> nestedtopobrain_v1_py_seed_everything
    nestedtopobrain_v1_py_ResourceMonitor["ResourceMonitor"]
    class nestedtopobrain_v1_py_ResourceMonitor cls;
    nestedtopobrain_v1_py --> nestedtopobrain_v1_py_ResourceMonitor
    nestedtopobrain_v1_py_PrefrontalOrchestrator["PrefrontalOrchestrator"]
    class nestedtopobrain_v1_py_PrefrontalOrchestrator cls;
    nestedtopobrain_v1_py --> nestedtopobrain_v1_py_PrefrontalOrchestrator
    nestedtopobrain_v1_py_TopologyMetrics["TopologyMetrics"]
    class nestedtopobrain_v1_py_TopologyMetrics cls;
    nestedtopobrain_v1_py --> nestedtopobrain_v1_py_TopologyMetrics
    nestedtopobrain_v2_py["nestedtopobrain_v2.py (py)"]
    class nestedtopobrain_v2_py mod;
    nestedtopobrain_v2_py_Config["Config"]
    class nestedtopobrain_v2_py_Config cls;
    nestedtopobrain_v2_py --> nestedtopobrain_v2_py_Config
    nestedtopobrain_v2_py_seed_everything["seed_everything"]
    class nestedtopobrain_v2_py_seed_everything fn;
    nestedtopobrain_v2_py --> nestedtopobrain_v2_py_seed_everything
    nestedtopobrain_v2_py_ResourceMonitor["ResourceMonitor"]
    class nestedtopobrain_v2_py_ResourceMonitor cls;
    nestedtopobrain_v2_py --> nestedtopobrain_v2_py_ResourceMonitor
    nestedtopobrain_v2_py_PrefrontalOrchestrator["PrefrontalOrchestrator"]
    class nestedtopobrain_v2_py_PrefrontalOrchestrator cls;
    nestedtopobrain_v2_py --> nestedtopobrain_v2_py_PrefrontalOrchestrator
    nestedtopobrain_v2_py_TopologyMetrics["TopologyMetrics"]
    class nestedtopobrain_v2_py_TopologyMetrics cls;
    nestedtopobrain_v2_py --> nestedtopobrain_v2_py_TopologyMetrics
    nestedtopobrain_v3_py["nestedtopobrain_v3.py (py)"]
    class nestedtopobrain_v3_py mod;
    nestedtopobrain_v3_py_Config["Config"]
    class nestedtopobrain_v3_py_Config cls;
    nestedtopobrain_v3_py --> nestedtopobrain_v3_py_Config
    nestedtopobrain_v3_py_seed_everything["seed_everything"]
    class nestedtopobrain_v3_py_seed_everything fn;
    nestedtopobrain_v3_py --> nestedtopobrain_v3_py_seed_everything
    nestedtopobrain_v3_py_ResourceMonitor["ResourceMonitor"]
    class nestedtopobrain_v3_py_ResourceMonitor cls;
    nestedtopobrain_v3_py --> nestedtopobrain_v3_py_ResourceMonitor
    nestedtopobrain_v3_py_PrefrontalOrchestrator["PrefrontalOrchestrator"]
    class nestedtopobrain_v3_py_PrefrontalOrchestrator cls;
    nestedtopobrain_v3_py --> nestedtopobrain_v3_py_PrefrontalOrchestrator
    nestedtopobrain_v3_py_TopologyMetrics["TopologyMetrics"]
    class nestedtopobrain_v3_py_TopologyMetrics cls;
    nestedtopobrain_v3_py --> nestedtopobrain_v3_py_TopologyMetrics
    topobrain_v18_py["topobrain_v18.py (py)"]
    class topobrain_v18_py mod;
    topobrain_v18_py_Config["Config"]
    class topobrain_v18_py_Config cls;
    topobrain_v18_py --> topobrain_v18_py_Config
    topobrain_v18_py_seed_everything["seed_everything"]
    class topobrain_v18_py_seed_everything fn;
    topobrain_v18_py --> topobrain_v18_py_seed_everything
    topobrain_v18_py_ResourceMonitor["ResourceMonitor"]
    class topobrain_v18_py_ResourceMonitor cls;
    topobrain_v18_py --> topobrain_v18_py_ResourceMonitor
    topobrain_v18_py_TopologyMetrics["TopologyMetrics"]
    class topobrain_v18_py_TopologyMetrics cls;
    topobrain_v18_py --> topobrain_v18_py_TopologyMetrics
    topobrain_v18_py_TopologicalHealthSovereignty["TopologicalHealthSovereignty"]
    class topobrain_v18_py_TopologicalHealthSovereignty cls;
    topobrain_v18_py --> topobrain_v18_py_TopologicalHealthSovereignty
    topobrain_v18_1_py["topobrain_v18.1.py (py)"]
    class topobrain_v18_1_py mod;
    topobrain_v18_1_py_Config["Config"]
    class topobrain_v18_1_py_Config cls;
    topobrain_v18_1_py --> topobrain_v18_1_py_Config
    topobrain_v18_1_py_seed_everything["seed_everything"]
    class topobrain_v18_1_py_seed_everything fn;
    topobrain_v18_1_py --> topobrain_v18_1_py_seed_everything
    topobrain_v18_1_py_ResourceMonitor["ResourceMonitor"]
    class topobrain_v18_1_py_ResourceMonitor cls;
    topobrain_v18_1_py --> topobrain_v18_1_py_ResourceMonitor
    topobrain_v18_1_py_TopologyMetrics["TopologyMetrics"]
    class topobrain_v18_1_py_TopologyMetrics cls;
    topobrain_v18_1_py --> topobrain_v18_1_py_TopologyMetrics
    topobrain_v18_1_py_TopologicalHealthSovereignty["TopologicalHealthSovereignty"]
    class topobrain_v18_1_py_TopologicalHealthSovereignty cls;
    topobrain_v18_1_py --> topobrain_v18_1_py_TopologicalHealthSovereignty
    topobrain_v18_2_py["topobrain_v18.2.py (py)"]
    class topobrain_v18_2_py mod;
    topobrain_v18_2_py_Config["Config"]
    class topobrain_v18_2_py_Config cls;
    topobrain_v18_2_py --> topobrain_v18_2_py_Config
    topobrain_v18_2_py_seed_everything["seed_everything"]
    class topobrain_v18_2_py_seed_everything fn;
    topobrain_v18_2_py --> topobrain_v18_2_py_seed_everything
    topobrain_v18_2_py_ResourceMonitor["ResourceMonitor"]
    class topobrain_v18_2_py_ResourceMonitor cls;
    topobrain_v18_2_py --> topobrain_v18_2_py_ResourceMonitor
    topobrain_v18_2_py_TopologyMetrics["TopologyMetrics"]
    class topobrain_v18_2_py_TopologyMetrics cls;
    topobrain_v18_2_py --> topobrain_v18_2_py_TopologyMetrics
    topobrain_v18_2_py_TopologicalHealthSovereignty["TopologicalHealthSovereignty"]
    class topobrain_v18_2_py_TopologicalHealthSovereignty cls;
    topobrain_v18_2_py --> topobrain_v18_2_py_TopologicalHealthSovereignty
    trycameral_py["trycameral.py (py)"]
    class trycameral_py mod;
    trycameral_py_generate_audio_async["generate_audio_async"]
    class trycameral_py_generate_audio_async fn;
    trycameral_py --> trycameral_py_generate_audio_async
    trycameral_py_generate_all_audios_batch["generate_all_audios_batch"]
    class trycameral_py_generate_all_audios_batch fn;
    trycameral_py --> trycameral_py_generate_all_audios_batch
    trycameral_py_generate_audios_sync["generate_audios_sync"]
    class trycameral_py_generate_audios_sync fn;
    trycameral_py --> trycameral_py_generate_audios_sync
    trycameral_py_setup_flickr8k["setup_flickr8k"]
    class trycameral_py_setup_flickr8k fn;
    trycameral_py --> trycameral_py_setup_flickr8k
    trycameral_py_build_vocab_flickr["build_vocab_flickr"]
    class trycameral_py_build_vocab_flickr fn;
    trycameral_py --> trycameral_py_build_vocab_flickr
    tricameral_kimi_py["tricameral_kimi.py (py)"]
    class tricameral_kimi_py mod;
    tricameral_kimi_py_setup_flickr8k_with_audio["setup_flickr8k_with_audio"]
    class tricameral_kimi_py_setup_flickr8k_with_audio fn;
    tricameral_kimi_py --> tricameral_kimi_py_setup_flickr8k_with_audio
    tricameral_kimi_py_build_vocab_flickr["build_vocab_flickr"]
    class tricameral_kimi_py_build_vocab_flickr fn;
    tricameral_kimi_py --> tricameral_kimi_py_build_vocab_flickr
    tricameral_kimi_py_EpisodicMemoryBuffer["EpisodicMemoryBuffer"]
    class tricameral_kimi_py_EpisodicMemoryBuffer cls;
    tricameral_kimi_py --> tricameral_kimi_py_EpisodicMemoryBuffer
    tricameral_kimi_py_NeurocognitiveSystem["NeurocognitiveSystem"]
    class tricameral_kimi_py_NeurocognitiveSystem cls;
    tricameral_kimi_py --> tricameral_kimi_py_NeurocognitiveSystem
    tricameral_kimi_py_LanguageMetrics["LanguageMetrics"]
    class tricameral_kimi_py_LanguageMetrics cls;
    tricameral_kimi_py --> tricameral_kimi_py_LanguageMetrics
    train_Adversarial_py["train_Adversarial.py (py)"]
    class train_Adversarial_py mod;
    train_Adversarial_py_seed_everything["seed_everything"]
    class train_Adversarial_py_seed_everything fn;
    train_Adversarial_py --> train_Adversarial_py_seed_everything
    train_Adversarial_py_ResourceMonitor["ResourceMonitor"]
    class train_Adversarial_py_ResourceMonitor cls;
    train_Adversarial_py --> train_Adversarial_py_ResourceMonitor
    train_Adversarial_py_guardar_checkpoint["guardar_checkpoint"]
    class train_Adversarial_py_guardar_checkpoint fn;
    train_Adversarial_py --> train_Adversarial_py_guardar_checkpoint
    train_Adversarial_py_cargar_checkpoint["cargar_checkpoint"]
    class train_Adversarial_py_cargar_checkpoint fn;
    train_Adversarial_py --> train_Adversarial_py_cargar_checkpoint
    train_Adversarial_py_NestedOptimizer["NestedOptimizer"]
    class train_Adversarial_py_NestedOptimizer cls;
    train_Adversarial_py --> train_Adversarial_py_NestedOptimizer
    tricameralkimi2_py["tricameralkimi2.py (py)"]
    class tricameralkimi2_py mod;
    tricameralkimi2_py_setup_flickr8k_with_audio["setup_flickr8k_with_audio"]
    class tricameralkimi2_py_setup_flickr8k_with_audio fn;
    tricameralkimi2_py --> tricameralkimi2_py_setup_flickr8k_with_audio
    tricameralkimi2_py_build_vocab_flickr["build_vocab_flickr"]
    class tricameralkimi2_py_build_vocab_flickr fn;
    tricameralkimi2_py --> tricameralkimi2_py_build_vocab_flickr
    tricameralkimi2_py_EpisodicMemoryBuffer["EpisodicMemoryBuffer"]
    class tricameralkimi2_py_EpisodicMemoryBuffer cls;
    tricameralkimi2_py --> tricameralkimi2_py_EpisodicMemoryBuffer
    tricameralkimi2_py_NeurocognitiveSystem["NeurocognitiveSystem"]
    class tricameralkimi2_py_NeurocognitiveSystem cls;
    tricameralkimi2_py --> tricameralkimi2_py_NeurocognitiveSystem
    tricameralkimi2_py_LinguisticFeedbackLoop["LinguisticFeedbackLoop"]
    class tricameralkimi2_py_LinguisticFeedbackLoop cls;
    tricameralkimi2_py --> tricameralkimi2_py_LinguisticFeedbackLoop
    topobrain_py["topobrain.py (py)"]
    class topobrain_py mod;
    topobrain_py_seed_everything["seed_everything"]
    class topobrain_py_seed_everything fn;
    topobrain_py --> topobrain_py_seed_everything
    topobrain_py_ResourceMonitor["ResourceMonitor"]
    class topobrain_py_ResourceMonitor cls;
    topobrain_py --> topobrain_py_ResourceMonitor
    topobrain_py_guardar_checkpoint["guardar_checkpoint"]
    class topobrain_py_guardar_checkpoint fn;
    topobrain_py --> topobrain_py_guardar_checkpoint
    topobrain_py_cargar_checkpoint["cargar_checkpoint"]
    class topobrain_py_cargar_checkpoint fn;
    topobrain_py --> topobrain_py_cargar_checkpoint
    topobrain_py_LearnableAbsenceGating["LearnableAbsenceGating"]
    class topobrain_py_LearnableAbsenceGating cls;
    topobrain_py --> topobrain_py_LearnableAbsenceGating
    topobrain_16_3_py["topobrain_16_3.py (py)"]
    class topobrain_16_3_py mod;
    topobrain_16_3_py_seed_everything["seed_everything"]
    class topobrain_16_3_py_seed_everything fn;
    topobrain_16_3_py --> topobrain_16_3_py_seed_everything
    topobrain_16_3_py_ResourceMonitor["ResourceMonitor"]
    class topobrain_16_3_py_ResourceMonitor cls;
    topobrain_16_3_py --> topobrain_16_3_py_ResourceMonitor
    topobrain_16_3_py_guardar_checkpoint["guardar_checkpoint"]
    class topobrain_16_3_py_guardar_checkpoint fn;
    topobrain_16_3_py --> topobrain_16_3_py_guardar_checkpoint
    topobrain_16_3_py_cargar_checkpoint["cargar_checkpoint"]
    class topobrain_16_3_py_cargar_checkpoint fn;
    topobrain_16_3_py --> topobrain_16_3_py_cargar_checkpoint
    topobrain_16_3_py_LearnableAbsenceGating["LearnableAbsenceGating"]
    class topobrain_16_3_py_LearnableAbsenceGating cls;
    topobrain_16_3_py --> topobrain_16_3_py_LearnableAbsenceGating
    resma4_4_py["resma4.4.py (py)"]
    class resma4_4_py mod;
    resma4_4_py_ResourceMonitor["ResourceMonitor"]
    class resma4_4_py_ResourceMonitor cls;
    resma4_4_py --> resma4_4_py_ResourceMonitor
    resma4_4_py_guardar_checkpoint["guardar_checkpoint"]
    class resma4_4_py_guardar_checkpoint fn;
    resma4_4_py --> resma4_4_py_guardar_checkpoint
    resma4_4_py_cargar_checkpoint["cargar_checkpoint"]
    class resma4_4_py_cargar_checkpoint fn;
    resma4_4_py --> resma4_4_py_cargar_checkpoint
    resma4_4_py_RESMAConstants["RESMAConstants"]
    class resma4_4_py_RESMAConstants cls;
    resma4_4_py --> resma4_4_py_RESMAConstants
    resma4_4_py_QuantumLeaf["QuantumLeaf"]
    class resma4_4_py_QuantumLeaf cls;
    resma4_4_py --> resma4_4_py_QuantumLeaf
    resma4_5_py["resma4.5.py (py)"]
    class resma4_5_py mod;
    resma4_5_py_ResourceMonitor["ResourceMonitor"]
    class resma4_5_py_ResourceMonitor cls;
    resma4_5_py --> resma4_5_py_ResourceMonitor
    resma4_5_py_guardar_checkpoint["guardar_checkpoint"]
    class resma4_5_py_guardar_checkpoint fn;
    resma4_5_py --> resma4_5_py_guardar_checkpoint
    resma4_5_py_cargar_checkpoint["cargar_checkpoint"]
    class resma4_5_py_cargar_checkpoint fn;
    resma4_5_py --> resma4_5_py_cargar_checkpoint
    resma4_5_py__make_serializable["_make_serializable"]
    class resma4_5_py__make_serializable fn;
    resma4_5_py --> resma4_5_py__make_serializable
    resma4_5_py_RESMAConstants["RESMAConstants"]
    class resma4_5_py_RESMAConstants cls;
    resma4_5_py --> resma4_5_py_RESMAConstants
    resma4_3_py["resma4.3.py (py)"]
    class resma4_3_py mod;
    resma4_3_py_ResourceMonitor["ResourceMonitor"]
    class resma4_3_py_ResourceMonitor cls;
    resma4_3_py --> resma4_3_py_ResourceMonitor
    resma4_3_py_guardar_checkpoint["guardar_checkpoint"]
    class resma4_3_py_guardar_checkpoint fn;
    resma4_3_py --> resma4_3_py_guardar_checkpoint
    resma4_3_py_cargar_checkpoint["cargar_checkpoint"]
    class resma4_3_py_cargar_checkpoint fn;
    resma4_3_py --> resma4_3_py_cargar_checkpoint
    resma4_3_py_RESMAConstants["RESMAConstants"]
    class resma4_3_py_RESMAConstants cls;
    resma4_3_py --> resma4_3_py_RESMAConstants
    resma4_3_py_QuantumLeaf["QuantumLeaf"]
    class resma4_3_py_QuantumLeaf cls;
    resma4_3_py --> resma4_3_py_QuantumLeaf
    get_dataset_py["get_dataset.py (py)"]
    class get_dataset_py mod;
    get_dataset_py_download_captions_only["download_captions_only"]
    class get_dataset_py_download_captions_only fn;
    get_dataset_py --> get_dataset_py_download_captions_only
    get_dataset_py_generate_one_audio["generate_one_audio"]
    class get_dataset_py_generate_one_audio fn;
    get_dataset_py --> get_dataset_py_generate_one_audio
    get_dataset_py_load_checkpoint["load_checkpoint"]
    class get_dataset_py_load_checkpoint fn;
    get_dataset_py --> get_dataset_py_load_checkpoint
    get_dataset_py_save_checkpoint["save_checkpoint"]
    class get_dataset_py_save_checkpoint fn;
    get_dataset_py --> get_dataset_py_save_checkpoint
    get_dataset_py_generate_audios_with_checkpoints["generate_audios_with_checkpoints"]
    class get_dataset_py_generate_audios_with_checkpoints fn;
    get_dataset_py --> get_dataset_py_generate_audios_with_checkpoints
    resma4_8_py["resma4.8.py (py)"]
    class resma4_8_py mod;
    resma4_8_py_RESMAConstants["RESMAConstants"]
    class resma4_8_py_RESMAConstants cls;
    resma4_8_py --> resma4_8_py_RESMAConstants
    resma4_8_py_ResourceMonitor["ResourceMonitor"]
    class resma4_8_py_ResourceMonitor cls;
    resma4_8_py --> resma4_8_py_ResourceMonitor
    resma4_8_py_guardar_checkpoint["guardar_checkpoint"]
    class resma4_8_py_guardar_checkpoint fn;
    resma4_8_py --> resma4_8_py_guardar_checkpoint
    resma4_8_py_cargar_checkpoint["cargar_checkpoint"]
    class resma4_8_py_cargar_checkpoint fn;
    resma4_8_py --> resma4_8_py_cargar_checkpoint
    resma4_8_py__make_serializable["_make_serializable"]
    class resma4_8_py__make_serializable fn;
    resma4_8_py --> resma4_8_py__make_serializable
    microbi_py_py["microbi.py.py (py)"]
    class microbi_py_py mod;
    microbi_py_py_EpistemicCuriosityCPU["EpistemicCuriosityCPU"]
    class microbi_py_py_EpistemicCuriosityCPU cls;
    microbi_py_py --> microbi_py_py_EpistemicCuriosityCPU
    microbi_py_py_LiquidNeuronCPU["LiquidNeuronCPU"]
    class microbi_py_py_LiquidNeuronCPU cls;
    microbi_py_py --> microbi_py_py_LiquidNeuronCPU
    microbi_py_py_BicameralAttentionCPU["BicameralAttentionCPU"]
    class microbi_py_py_BicameralAttentionCPU cls;
    microbi_py_py --> microbi_py_py_BicameralAttentionCPU
    microbi_py_py_RightHemisphereCPU["RightHemisphereCPU"]
    class microbi_py_py_RightHemisphereCPU cls;
    microbi_py_py --> microbi_py_py_RightHemisphereCPU
    microbi_py_py_LeftHemisphereCPU["LeftHemisphereCPU"]
    class microbi_py_py_LeftHemisphereCPU cls;
    microbi_py_py --> microbi_py_py_LeftHemisphereCPU
    minibi_reduced_py_py["minibi_reduced.py.py (py)"]
    class minibi_reduced_py_py mod;
    minibi_reduced_py_py_EpistemicCuriosity["EpistemicCuriosity"]
    class minibi_reduced_py_py_EpistemicCuriosity cls;
    minibi_reduced_py_py --> minibi_reduced_py_py_EpistemicCuriosity
    minibi_reduced_py_py_LiquidNeuronV2["LiquidNeuronV2"]
    class minibi_reduced_py_py_LiquidNeuronV2 cls;
    minibi_reduced_py_py --> minibi_reduced_py_py_LiquidNeuronV2
    minibi_reduced_py_py_BicameralAttention["BicameralAttention"]
    class minibi_reduced_py_py_BicameralAttention cls;
    minibi_reduced_py_py --> minibi_reduced_py_py_BicameralAttention
    minibi_reduced_py_py_RightHemisphereV2["RightHemisphereV2"]
    class minibi_reduced_py_py_RightHemisphereV2 cls;
    minibi_reduced_py_py --> minibi_reduced_py_py_RightHemisphereV2
    minibi_reduced_py_py_LeftHemisphereV2["LeftHemisphereV2"]
    class minibi_reduced_py_py_LeftHemisphereV2 cls;
    minibi_reduced_py_py --> minibi_reduced_py_py_LeftHemisphereV2
    neurologos_homeostatico_cpu_ki_py["neurologos_homeostatico_cpu_ki.py (py)"]
    class neurologos_homeostatico_cpu_ki_py mod;
    neurologos_homeostatico_cpu_ki_py_MicroConfig["MicroConfig"]
    class neurologos_homeostatico_cpu_ki_py_MicroConfig cls;
    neurologos_homeostatico_cpu_ki_py --> neurologos_homeostatico_cpu_ki_py_MicroConfig
    neurologos_homeostatico_cpu_ki_py_seed_everything["seed_everything"]
    class neurologos_homeostatico_cpu_ki_py_seed_everything fn;
    neurologos_homeostatico_cpu_ki_py --> neurologos_homeostatico_cpu_ki_py_seed_everything
    neurologos_homeostatico_cpu_ki_py_get_dataset["get_dataset"]
    class neurologos_homeostatico_cpu_ki_py_get_dataset fn;
    neurologos_homeostatico_cpu_ki_py --> neurologos_homeostatico_cpu_ki_py_get_dataset
    neurologos_homeostatico_cpu_ki_py_HomeostaticCore["HomeostaticCore"]
    class neurologos_homeostatico_cpu_ki_py_HomeostaticCore cls;
    neurologos_homeostatico_cpu_ki_py --> neurologos_homeostatico_cpu_ki_py_HomeostaticCore
    neurologos_homeostatico_cpu_ki_py_MicroContinuumCell["MicroContinuumCell"]
    class neurologos_homeostatico_cpu_ki_py_MicroContinuumCell cls;
    neurologos_homeostatico_cpu_ki_py --> neurologos_homeostatico_cpu_ki_py_MicroContinuumCell
    resma4_6_py["resma4.6.py (py)"]
    class resma4_6_py mod;
    resma4_6_py_ResourceMonitor["ResourceMonitor"]
    class resma4_6_py_ResourceMonitor cls;
    resma4_6_py --> resma4_6_py_ResourceMonitor
    resma4_6_py_guardar_checkpoint["guardar_checkpoint"]
    class resma4_6_py_guardar_checkpoint fn;
    resma4_6_py --> resma4_6_py_guardar_checkpoint
    resma4_6_py_cargar_checkpoint["cargar_checkpoint"]
    class resma4_6_py_cargar_checkpoint fn;
    resma4_6_py --> resma4_6_py_cargar_checkpoint
    resma4_6_py__make_serializable["_make_serializable"]
    class resma4_6_py__make_serializable fn;
    resma4_6_py --> resma4_6_py__make_serializable
    resma4_6_py_RESMAConstants["RESMAConstants"]
    class resma4_6_py_RESMAConstants cls;
    resma4_6_py --> resma4_6_py_RESMAConstants
    legendario2_py["legendario2.py (py)"]
    class legendario2_py mod;
    legendario2_py_MotorHomeostaticContext["MotorHomeostaticContext"]
    class legendario2_py_MotorHomeostaticContext cls;
    legendario2_py --> legendario2_py_MotorHomeostaticContext
    legendario2_py_PTSymmetricMotor["PTSymmetricMotor"]
    class legendario2_py_PTSymmetricMotor cls;
    legendario2_py --> legendario2_py_PTSymmetricMotor
    legendario2_py_TopologicalMotor["TopologicalMotor"]
    class legendario2_py_TopologicalMotor cls;
    legendario2_py --> legendario2_py_TopologicalMotor
    legendario2_py_EnergyHomeostaticMotor["EnergyHomeostaticMotor"]
    class legendario2_py_EnergyHomeostaticMotor cls;
    legendario2_py --> legendario2_py_EnergyHomeostaticMotor
    legendario2_py_ConsciousnessMotor["ConsciousnessMotor"]
    class legendario2_py_ConsciousnessMotor cls;
    legendario2_py --> legendario2_py_ConsciousnessMotor
    ohm_py["ohm.py (py)"]
    class ohm_py mod;
    ohm_py_MotorHomeostaticContext["MotorHomeostaticContext"]
    class ohm_py_MotorHomeostaticContext cls;
    ohm_py --> ohm_py_MotorHomeostaticContext
    ohm_py_PTSymmetricMotor["PTSymmetricMotor"]
    class ohm_py_PTSymmetricMotor cls;
    ohm_py --> ohm_py_PTSymmetricMotor
    ohm_py_TopologicalMotor["TopologicalMotor"]
    class ohm_py_TopologicalMotor cls;
    ohm_py --> ohm_py_TopologicalMotor
    ohm_py_EnergyHomeostaticMotor["EnergyHomeostaticMotor"]
    class ohm_py_EnergyHomeostaticMotor cls;
    ohm_py --> ohm_py_EnergyHomeostaticMotor
    ohm_py_ConsciousnessMotor["ConsciousnessMotor"]
    class ohm_py_ConsciousnessMotor cls;
    ohm_py --> ohm_py_ConsciousnessMotor
    resma4_10_py["resma4.10.py (py)"]
    class resma4_10_py mod;
    resma4_10_py_RESMAConstants["RESMAConstants"]
    class resma4_10_py_RESMAConstants cls;
    resma4_10_py --> resma4_10_py_RESMAConstants
    resma4_10_py_GarnierTresTiempos["GarnierTresTiempos"]
    class resma4_10_py_GarnierTresTiempos cls;
    resma4_10_py --> resma4_10_py_GarnierTresTiempos
    resma4_10_py_OperadorDesdoblamiento["OperadorDesdoblamiento"]
    class resma4_10_py_OperadorDesdoblamiento cls;
    resma4_10_py --> resma4_10_py_OperadorDesdoblamiento
    resma4_10_py_SilencioActivoMonitor["SilencioActivoMonitor"]
    class resma4_10_py_SilencioActivoMonitor cls;
    resma4_10_py --> resma4_10_py_SilencioActivoMonitor
    resma4_10_py_QuantumLeaf["QuantumLeaf"]
    class resma4_10_py_QuantumLeaf cls;
    resma4_10_py --> resma4_10_py_QuantumLeaf
    minibi2_py["minibi2.py (py)"]
    class minibi2_py mod;
    minibi2_py_EpistemicCuriosity["EpistemicCuriosity"]
    class minibi2_py_EpistemicCuriosity cls;
    minibi2_py --> minibi2_py_EpistemicCuriosity
    minibi2_py_LiquidNeuronV2["LiquidNeuronV2"]
    class minibi2_py_LiquidNeuronV2 cls;
    minibi2_py --> minibi2_py_LiquidNeuronV2
    minibi2_py_BicameralAttention["BicameralAttention"]
    class minibi2_py_BicameralAttention cls;
    minibi2_py --> minibi2_py_BicameralAttention
    minibi2_py_RightHemisphereV2["RightHemisphereV2"]
    class minibi2_py_RightHemisphereV2 cls;
    minibi2_py --> minibi2_py_RightHemisphereV2
    minibi2_py_LeftHemisphereV2["LeftHemisphereV2"]
    class minibi2_py_LeftHemisphereV2 cls;
    minibi2_py --> minibi2_py_LeftHemisphereV2
    minibi_c_py["minibi_c.py (py)"]
    class minibi_c_py mod;
    minibi_c_py_EpistemicCuriosity["EpistemicCuriosity"]
    class minibi_c_py_EpistemicCuriosity cls;
    minibi_c_py --> minibi_c_py_EpistemicCuriosity
    minibi_c_py_LiquidNeuronV2["LiquidNeuronV2"]
    class minibi_c_py_LiquidNeuronV2 cls;
    minibi_c_py --> minibi_c_py_LiquidNeuronV2
    minibi_c_py_BicameralAttention["BicameralAttention"]
    class minibi_c_py_BicameralAttention cls;
    minibi_c_py --> minibi_c_py_BicameralAttention
    minibi_c_py_RightHemisphereV2["RightHemisphereV2"]
    class minibi_c_py_RightHemisphereV2 cls;
    minibi_c_py --> minibi_c_py_RightHemisphereV2
    minibi_c_py_LeftHemisphereV2["LeftHemisphereV2"]
    class minibi_c_py_LeftHemisphereV2 cls;
    minibi_c_py --> minibi_c_py_LeftHemisphereV2
    bicameral_py["bicameral.py (py)"]
    class bicameral_py mod;
    bicameral_py_seed_all["seed_all"]
    class bicameral_py_seed_all fn;
    bicameral_py --> bicameral_py_seed_all
    bicameral_py_setup_flickr8k["setup_flickr8k"]
    class bicameral_py_setup_flickr8k fn;
    bicameral_py --> bicameral_py_setup_flickr8k
    bicameral_py_HomeostaticRegulator["HomeostaticRegulator"]
    class bicameral_py_HomeostaticRegulator cls;
    bicameral_py --> bicameral_py_HomeostaticRegulator
    bicameral_py_PhysioNeuron["PhysioNeuron"]
    class bicameral_py_PhysioNeuron cls;
    bicameral_py --> bicameral_py_PhysioNeuron
    bicameral_py_RightHemisphere["RightHemisphere"]
    class bicameral_py_RightHemisphere cls;
    bicameral_py --> bicameral_py_RightHemisphere
    physio_chimera_v15_monitored_py["physio_chimera_v15_monitored.py (py)"]
    class physio_chimera_v15_monitored_py mod;
    physio_chimera_v15_monitored_py_Config["Config"]
    class physio_chimera_v15_monitored_py_Config cls;
    physio_chimera_v15_monitored_py --> physio_chimera_v15_monitored_py_Config
    physio_chimera_v15_monitored_py_seed_everything["seed_everything"]
    class physio_chimera_v15_monitored_py_seed_everything fn;
    physio_chimera_v15_monitored_py --> physio_chimera_v15_monitored_py_seed_everything
    physio_chimera_v15_monitored_py_DataEnvironment["DataEnvironment"]
    class physio_chimera_v15_monitored_py_DataEnvironment cls;
    physio_chimera_v15_monitored_py --> physio_chimera_v15_monitored_py_DataEnvironment
    physio_chimera_v15_monitored_py_NeuralDiagnostics["NeuralDiagnostics"]
    class physio_chimera_v15_monitored_py_NeuralDiagnostics cls;
    physio_chimera_v15_monitored_py --> physio_chimera_v15_monitored_py_NeuralDiagnostics
    physio_chimera_v15_monitored_py_SelfModifyingGates["SelfModifyingGates"]
    class physio_chimera_v15_monitored_py_SelfModifyingGates cls;
    physio_chimera_v15_monitored_py --> physio_chimera_v15_monitored_py_SelfModifyingGates
    app_py["app.py (py)"]
    class app_py mod;
    app_py_train_ai_model["train_ai_model"]
    class app_py_train_ai_model fn;
    app_py --> app_py_train_ai_model
    app_py_load_or_train_model["load_or_train_model"]
    class app_py_load_or_train_model fn;
    app_py --> app_py_load_or_train_model
    app_py_apply_ai_predictions["apply_ai_predictions"]
    class app_py_apply_ai_predictions fn;
    app_py --> app_py_apply_ai_predictions
    app_py_apply_ai_predictions["apply_ai_predictions"]
    class app_py_apply_ai_predictions fn;
    app_py --> app_py_apply_ai_predictions
    app_py_analyze_ia_vs_rules["analyze_ia_vs_rules"]
    class app_py_analyze_ia_vs_rules fn;
    app_py --> app_py_analyze_ia_vs_rules
    main5_py["main5.py (py)"]
    class main5_py mod;
    main5_py_RESMAConstants["RESMAConstants"]
    class main5_py_RESMAConstants cls;
    main5_py --> main5_py_RESMAConstants
    main5_py_PhysicalValidator["PhysicalValidator"]
    class main5_py_PhysicalValidator cls;
    main5_py --> main5_py_PhysicalValidator
    main5_py_QuantumLeaf["QuantumLeaf"]
    class main5_py_QuantumLeaf cls;
    main5_py --> main5_py_QuantumLeaf
    main5_py_RESMAUniverse["RESMAUniverse"]
    class main5_py_RESMAUniverse cls;
    main5_py --> main5_py_RESMAUniverse
    main5_py_BranchingOperator["BranchingOperator"]
    class main5_py_BranchingOperator cls;
    main5_py --> main5_py_BranchingOperator
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py["NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py (py)"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py mod;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_compute_loss["compute_loss"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_compute_loss fn;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py --> NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_compute_loss
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_NeurocognitiveSystem["NeurocognitiveSystem"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_NeurocognitiveSystem cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py --> NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_NeurocognitiveSystem
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_LinguisticFeedbackLoop["LinguisticFeedbackLoop"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_LinguisticFeedbackLoop cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py --> NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_LinguisticFeedbackLoop
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_LanguageMetrics["LanguageMetrics"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_LanguageMetrics cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py --> NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_LanguageMetrics
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_TriangulatedMedicalSystem["TriangulatedMedicalSystem"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_TriangulatedMedicalSystem cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py --> NeuroLogos_Bicameral_FISIOL_GICO_v4_0_py_TriangulatedMedicalSystem
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py["NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py (py)"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py mod;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_compute_loss["compute_loss"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_compute_loss fn;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py --> NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_compute_loss
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_EpisodicMemoryBuffer["EpisodicMemoryBuffer"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_EpisodicMemoryBuffer cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py --> NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_EpisodicMemoryBuffer
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_NeurocognitiveSystem["NeurocognitiveSystem"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_NeurocognitiveSystem cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py --> NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_NeurocognitiveSystem
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_LinguisticFeedbackLoop["LinguisticFeedbackLoop"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_LinguisticFeedbackLoop cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py --> NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_LinguisticFeedbackLoop
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_LanguageMetrics["LanguageMetrics"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_LanguageMetrics cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py --> NeuroLogos_Bicameral_FISIOL_GICO_v4_1_py_LanguageMetrics
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py["NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py (py)"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py mod;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_compute_loss["compute_loss"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_compute_loss fn;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py --> NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_compute_loss
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_NeurocognitiveSystem["NeurocognitiveSystem"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_NeurocognitiveSystem cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py --> NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_NeurocognitiveSystem
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_LinguisticFeedbackLoop["LinguisticFeedbackLoop"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_LinguisticFeedbackLoop cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py --> NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_LinguisticFeedbackLoop
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_LanguageMetrics["LanguageMetrics"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_LanguageMetrics cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py --> NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_LanguageMetrics
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_TriangulatedMedicalSystem["TriangulatedMedicalSystem"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_TriangulatedMedicalSystem cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py --> NeuroLogos_Bicameral_FISIOL_GICO_v3_9_py_TriangulatedMedicalSystem
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py["NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py (py)"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py mod;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_compute_loss["compute_loss"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_compute_loss fn;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py --> NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_compute_loss
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_NeurocognitiveSystem["NeurocognitiveSystem"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_NeurocognitiveSystem cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py --> NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_NeurocognitiveSystem
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_LinguisticFeedbackLoop["LinguisticFeedbackLoop"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_LinguisticFeedbackLoop cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py --> NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_LinguisticFeedbackLoop
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_LanguageMetrics["LanguageMetrics"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_LanguageMetrics cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py --> NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_LanguageMetrics
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_TriangulatedMedicalSystem["TriangulatedMedicalSystem"]
    class NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_TriangulatedMedicalSystem cls;
    NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py --> NeuroLogos_Bicameral_FISIOL_GICO_v3_8_py_TriangulatedMedicalSystem
    resma4_9_py["resma4.9.py (py)"]
    class resma4_9_py mod;
    resma4_9_py_RESMAConstants["RESMAConstants"]
    class resma4_9_py_RESMAConstants cls;
    resma4_9_py --> resma4_9_py_RESMAConstants
    resma4_9_py_GarnierTresTiempos["GarnierTresTiempos"]
    class resma4_9_py_GarnierTresTiempos cls;
    resma4_9_py --> resma4_9_py_GarnierTresTiempos
    resma4_9_py_OperadorDesdoblamiento["OperadorDesdoblamiento"]
    class resma4_9_py_OperadorDesdoblamiento cls;
    resma4_9_py --> resma4_9_py_OperadorDesdoblamiento
    resma4_9_py_SilencioActivoMonitor["SilencioActivoMonitor"]
    class resma4_9_py_SilencioActivoMonitor cls;
    resma4_9_py --> resma4_9_py_SilencioActivoMonitor
    resma4_9_py_QuantumLeaf["QuantumLeaf"]
    class resma4_9_py_QuantumLeaf cls;
    resma4_9_py --> resma4_9_py_QuantumLeaf
    n_01_topobrain_cpu_v6_py["01_topobrain_cpu_v6.py (py)"]
    class n_01_topobrain_cpu_v6_py mod;
    n_01_topobrain_cpu_v6_py_MicroConfig["MicroConfig"]
    class n_01_topobrain_cpu_v6_py_MicroConfig cls;
    n_01_topobrain_cpu_v6_py --> n_01_topobrain_cpu_v6_py_MicroConfig
    n_01_topobrain_cpu_v6_py_seed_everything["seed_everything"]
    class n_01_topobrain_cpu_v6_py_seed_everything fn;
    n_01_topobrain_cpu_v6_py --> n_01_topobrain_cpu_v6_py_seed_everything
    n_01_topobrain_cpu_v6_py_get_micro_dataset["get_micro_dataset"]
    class n_01_topobrain_cpu_v6_py_get_micro_dataset fn;
    n_01_topobrain_cpu_v6_py --> n_01_topobrain_cpu_v6_py_get_micro_dataset
    n_01_topobrain_cpu_v6_py_compute_effect_size["compute_effect_size"]
    class n_01_topobrain_cpu_v6_py_compute_effect_size fn;
    n_01_topobrain_cpu_v6_py --> n_01_topobrain_cpu_v6_py_compute_effect_size
    n_01_topobrain_cpu_v6_py_MicroSupConLoss["MicroSupConLoss"]
    class n_01_topobrain_cpu_v6_py_MicroSupConLoss cls;
    n_01_topobrain_cpu_v6_py --> n_01_topobrain_cpu_v6_py_MicroSupConLoss
    minibi_py["minibi.py (py)"]
    class minibi_py mod;
    minibi_py_setup_flickr8k["setup_flickr8k"]
    class minibi_py_setup_flickr8k fn;
    minibi_py --> minibi_py_setup_flickr8k
    minibi_py_build_vocab_flickr["build_vocab_flickr"]
    class minibi_py_build_vocab_flickr fn;
    minibi_py --> minibi_py_build_vocab_flickr
    minibi_py_Flickr8kDataset["Flickr8kDataset"]
    class minibi_py_Flickr8kDataset cls;
    minibi_py --> minibi_py_Flickr8kDataset
    minibi_py_LiquidNeuron["LiquidNeuron"]
    class minibi_py_LiquidNeuron cls;
    minibi_py --> minibi_py_LiquidNeuron
    minibi_py_RightHemisphere["RightHemisphere"]
    class minibi_py_RightHemisphere cls;
    minibi_py --> minibi_py_RightHemisphere
    pokemon_hybrid_synergy_ablation_py["pokemon_hybrid_synergy_ablation.py (py)"]
    class pokemon_hybrid_synergy_ablation_py mod;
    pokemon_hybrid_synergy_ablation_py_SynergyConfig["SynergyConfig"]
    class pokemon_hybrid_synergy_ablation_py_SynergyConfig cls;
    pokemon_hybrid_synergy_ablation_py --> pokemon_hybrid_synergy_ablation_py_SynergyConfig
    pokemon_hybrid_synergy_ablation_py_SynergyVAELayer["SynergyVAELayer"]
    class pokemon_hybrid_synergy_ablation_py_SynergyVAELayer cls;
    pokemon_hybrid_synergy_ablation_py --> pokemon_hybrid_synergy_ablation_py_SynergyVAELayer
    pokemon_hybrid_synergy_ablation_py_SynergyAttentionLayer["SynergyAttentionLayer"]
    class pokemon_hybrid_synergy_ablation_py_SynergyAttentionLayer cls;
    pokemon_hybrid_synergy_ablation_py --> pokemon_hybrid_synergy_ablation_py_SynergyAttentionLayer
    pokemon_hybrid_synergy_ablation_py_SynergyGANLayer["SynergyGANLayer"]
    class pokemon_hybrid_synergy_ablation_py_SynergyGANLayer cls;
    pokemon_hybrid_synergy_ablation_py --> pokemon_hybrid_synergy_ablation_py_SynergyGANLayer
    pokemon_hybrid_synergy_ablation_py_AdaptiveTopologyLayer["AdaptiveTopologyLayer"]
    class pokemon_hybrid_synergy_ablation_py_AdaptiveTopologyLayer cls;
    pokemon_hybrid_synergy_ablation_py --> pokemon_hybrid_synergy_ablation_py_AdaptiveTopologyLayer
    gemini_py["gemini.py (py)"]
    class gemini_py mod;
    gemini_py_setup_flickr8k["setup_flickr8k"]
    class gemini_py_setup_flickr8k fn;
    gemini_py --> gemini_py_setup_flickr8k
    gemini_py_LiquidNeuron["LiquidNeuron"]
    class gemini_py_LiquidNeuron cls;
    gemini_py --> gemini_py_LiquidNeuron
    gemini_py_RightHemisphere["RightHemisphere"]
    class gemini_py_RightHemisphere cls;
    gemini_py --> gemini_py_RightHemisphere
    gemini_py_CorpusCallosum["CorpusCallosum"]
    class gemini_py_CorpusCallosum cls;
    gemini_py --> gemini_py_CorpusCallosum
    gemini_py_LeftHemisphere["LeftHemisphere"]
    class gemini_py_LeftHemisphere cls;
    gemini_py --> gemini_py_LeftHemisphere
    gemini2_py["gemini2.py (py)"]
    class gemini2_py mod;
    gemini2_py_setup_flickr8k["setup_flickr8k"]
    class gemini2_py_setup_flickr8k fn;
    gemini2_py --> gemini2_py_setup_flickr8k
    gemini2_py_LiquidNeuron["LiquidNeuron"]
    class gemini2_py_LiquidNeuron cls;
    gemini2_py --> gemini2_py_LiquidNeuron
    gemini2_py_RightHemisphere["RightHemisphere"]
    class gemini2_py_RightHemisphere cls;
    gemini2_py --> gemini2_py_RightHemisphere
    gemini2_py_CorpusCallosum["CorpusCallosum"]
    class gemini2_py_CorpusCallosum cls;
    gemini2_py --> gemini2_py_CorpusCallosum
    gemini2_py_LeftHemisphere["LeftHemisphere"]
    class gemini2_py_LeftHemisphere cls;
    gemini2_py --> gemini2_py_LeftHemisphere
    n_01_topobrain_cpu_v3_py["01_topobrain_cpu_v3.py (py)"]
    class n_01_topobrain_cpu_v3_py mod;
    n_01_topobrain_cpu_v3_py_Config["Config"]
    class n_01_topobrain_cpu_v3_py_Config cls;
    n_01_topobrain_cpu_v3_py --> n_01_topobrain_cpu_v3_py_Config
    n_01_topobrain_cpu_v3_py_seed_everything["seed_everything"]
    class n_01_topobrain_cpu_v3_py_seed_everything fn;
    n_01_topobrain_cpu_v3_py --> n_01_topobrain_cpu_v3_py_seed_everything
    n_01_topobrain_cpu_v3_py_get_tabular_loaders["get_tabular_loaders"]
    class n_01_topobrain_cpu_v3_py_get_tabular_loaders fn;
    n_01_topobrain_cpu_v3_py --> n_01_topobrain_cpu_v3_py_get_tabular_loaders
    n_01_topobrain_cpu_v3_py_SupConLoss["SupConLoss"]
    class n_01_topobrain_cpu_v3_py_SupConLoss cls;
    n_01_topobrain_cpu_v3_py --> n_01_topobrain_cpu_v3_py_SupConLoss
    n_01_topobrain_cpu_v3_py_ContinuumMemoryCell["ContinuumMemoryCell"]
    class n_01_topobrain_cpu_v3_py_ContinuumMemoryCell cls;
    n_01_topobrain_cpu_v3_py --> n_01_topobrain_cpu_v3_py_ContinuumMemoryCell
    n_01_topobrain_cou_v2_py["01_topobrain_cou_v2.py (py)"]
    class n_01_topobrain_cou_v2_py mod;
    n_01_topobrain_cou_v2_py_Config["Config"]
    class n_01_topobrain_cou_v2_py_Config cls;
    n_01_topobrain_cou_v2_py --> n_01_topobrain_cou_v2_py_Config
    n_01_topobrain_cou_v2_py_seed_everything["seed_everything"]
    class n_01_topobrain_cou_v2_py_seed_everything fn;
    n_01_topobrain_cou_v2_py --> n_01_topobrain_cou_v2_py_seed_everything
    n_01_topobrain_cou_v2_py_get_tabular_loaders["get_tabular_loaders"]
    class n_01_topobrain_cou_v2_py_get_tabular_loaders fn;
    n_01_topobrain_cou_v2_py --> n_01_topobrain_cou_v2_py_get_tabular_loaders
    n_01_topobrain_cou_v2_py_StableSupConLoss["StableSupConLoss"]
    class n_01_topobrain_cou_v2_py_StableSupConLoss cls;
    n_01_topobrain_cou_v2_py --> n_01_topobrain_cou_v2_py_StableSupConLoss
    n_01_topobrain_cou_v2_py_StableContinuumMemoryCell["StableContinuumMemoryCell"]
    class n_01_topobrain_cou_v2_py_StableContinuumMemoryCell cls;
    n_01_topobrain_cou_v2_py --> n_01_topobrain_cou_v2_py_StableContinuumMemoryCell
    n_01_topobrain_cpu_v4_py["01_topobrain_cpu_v4.py (py)"]
    class n_01_topobrain_cpu_v4_py mod;
    n_01_topobrain_cpu_v4_py_Config["Config"]
    class n_01_topobrain_cpu_v4_py_Config cls;
    n_01_topobrain_cpu_v4_py --> n_01_topobrain_cpu_v4_py_Config
    n_01_topobrain_cpu_v4_py_seed_everything["seed_everything"]
    class n_01_topobrain_cpu_v4_py_seed_everything fn;
    n_01_topobrain_cpu_v4_py --> n_01_topobrain_cpu_v4_py_seed_everything
    n_01_topobrain_cpu_v4_py_get_tabular_loaders["get_tabular_loaders"]
    class n_01_topobrain_cpu_v4_py_get_tabular_loaders fn;
    n_01_topobrain_cpu_v4_py --> n_01_topobrain_cpu_v4_py_get_tabular_loaders
    n_01_topobrain_cpu_v4_py_StableSupConLoss["StableSupConLoss"]
    class n_01_topobrain_cpu_v4_py_StableSupConLoss cls;
    n_01_topobrain_cpu_v4_py --> n_01_topobrain_cpu_v4_py_StableSupConLoss
    n_01_topobrain_cpu_v4_py_StableContinuumMemoryCell["StableContinuumMemoryCell"]
    class n_01_topobrain_cpu_v4_py_StableContinuumMemoryCell cls;
    n_01_topobrain_cpu_v4_py --> n_01_topobrain_cpu_v4_py_StableContinuumMemoryCell
```

---

## Architecture Reference

### PY (189 files)

#### `01_mcculloch_pitts.py`
**Path:** `01_mcculloch_pitts.py`

**Functions:**
- `mcculloch_pitts_neuron` (line 7) - *Neurona artificial de McCulloch-Pitts (1943).
- inputs: vector binario de entrada (0 o 1)
- weights: vector de pesos sinápticos
- threshold: valor umbral de activación
Retorna 1 si la suma ponderada >= threshold; 0 en caso contrario.*

#### `01_topobrain_cou_v2.py`
**Path:** `01_topobrain_cou_v2.py`

**Classs:**
- `Config` (line 25)
- `StableSupConLoss` (line 116) - *Versión estable de SupConLoss con manejo de bordes robusto*
- `StableContinuumMemoryCell` (line 147) - *Versión estable y ligera de ContinuumMemoryCell con clamping y normalización*
- `StableSymbioticBasisRefinement` (line 231) - *Refinamiento simbiótico estable con regularización explícita (versión ligera)*
- `StableTopologyManager` (line 267) - *Gestor de topología con protocolos de supervivencia (versión optimizada para CPU)*
- `StableTopoBrain` (line 345) - *Implementación estable y ligera de TopoBrain para ablation científico en CPU*

**Functions:**
- `seed_everything` (line 70)
- `get_tabular_loaders` (line 77) - *Dataset tabular controlado con características NOIR simuladas*
- `stable_pgd_attack` (line 477) - *Ataque PGD estable con manejo de gradientes robusto*
- `train_epoch` (line 520) - *Entrenamiento por época con monitoreo detallado*
- `evaluate_model` (line 584) - *Evaluación rigurosa con o sin ataques adversariales*
- `run_scientific_ablation` (line 632) - *Ejecución científica del ablation con control de variables*
- `to_dict` (line 56)
- `get_topology_config` (line 59) - *Configuración estable para topología adaptable*
- `__init__` (line 118)
- `forward` (line 123)
- `__init__` (line 149)
- `forward` (line 181)
- `__init__` (line 233)
- `forward` (line 242)
- `__init__` (line 269)
- `get_adjacency` (line 288) - *Obtener matriz de adyacencia con estabilidad garantizada*
- `prune_topology` (line 317) - *Poda controlada con protocolo de emergencia*
- `get_density` (line 337) - *Calcular densidad actual de manera estable*
- `__init__` (line 347)
- `_init_weights` (line 400) - *Inicialización estable de pesos*
- `forward` (line 408)

#### `01_topobrain_cpu.py`
**Path:** `01_topobrain_cpu.py`

**Classs:**
- `Config` (line 21)
- `SupConLoss` (line 94)
- `ContinuumMemoryCell` (line 109)
- `SymbioticBasisRefinement` (line 143)
- `TopoBrainTabular` (line 166)

**Functions:**
- `seed_everything` (line 63)
- `get_tabular_loaders` (line 69)
- `pgd_attack` (line 247)
- `generate_ablation_configs` (line 263)
- `train_and_evaluate` (line 293)
- `run_ablation` (line 337)
- `to_dict` (line 57)
- `__init__` (line 95)
- `forward` (line 98)
- `__init__` (line 110)
- `forward` (line 124)
- `__init__` (line 144)
- `forward` (line 151)
- `__init__` (line 167)
- `get_adj` (line 198)
- `forward` (line 203)
- `evaluate_adv` (line 323)

#### `01_topobrain_cpu_v3.py`
**Path:** `01_topobrain_cpu_v3.py`

**Classs:**
- `Config` (line 24)
- `SupConLoss` (line 93)
- `ContinuumMemoryCell` (line 108)
- `SymbioticBasisRefinement` (line 145)
- `PrefrontalOrchestrator` (line 165)
- `AdaptiveCombinatorialComplexLayer` (line 227)
- `TopoBrainTabular` (line 320)

**Functions:**
- `seed_everything` (line 63)
- `get_tabular_loaders` (line 69)
- `pgd_attack` (line 395)
- `compute_topology_metrics` (line 416) - *Computar métricas de topología con manejo robusto de errores*
- `prune_topology` (line 450) - *Implementación simplificada de poda de topología*
- `train_and_evaluate` (line 482)
- `run_ablation` (line 640)
- `to_dict` (line 57)
- `__init__` (line 94)
- `forward` (line 97)
- `__init__` (line 109)
- `forward` (line 122)
- `__init__` (line 146)
- `forward` (line 153)
- `__init__` (line 166)
- `forward` (line 183)
- `reset_context` (line 221)
- `__init__` (line 228)
- `get_adj` (line 264)
- `forward` (line 269)
- `__init__` (line 321)
- `_initialize_memories` (line 352) - *Inicialización de memorias semánticas*
- `forward` (line 363)
- `evaluate_adv` (line 593)

#### `01_topobrain_cpu_v4.py`
**Path:** `01_topobrain_cpu_v4.py`

**Classs:**
- `Config` (line 25)
- `StableSupConLoss` (line 116) - *Versión estable de SupConLoss con manejo de bordes robusto*
- `StableContinuumMemoryCell` (line 147) - *Versión estable y ligera de ContinuumMemoryCell con clamping y normalización*
- `StableSymbioticBasisRefinement` (line 231) - *Refinamiento simbiótico estable con regularización explícita (versión ligera)*
- `StableTopologyManager` (line 267) - *Gestor de topología con protocolos de supervivencia (versión optimizada para CPU)*
- `StableTopoBrain` (line 345) - *Implementación estable y ligera de TopoBrain para ablation científico en CPU*

**Functions:**
- `seed_everything` (line 70)
- `get_tabular_loaders` (line 77) - *Dataset tabular controlado con características NOIR simuladas*
- `stable_pgd_attack` (line 475) - *Ataque PGD estable con manejo de gradientes robusto*
- `train_epoch` (line 518) - *Entrenamiento por época con monitoreo detallado*
- `evaluate_model` (line 583) - *Evaluación rigurosa con o sin ataques adversariales*
- `run_scientific_ablation` (line 628) - *Ejecución científica del ablation con control de variables*
- `to_dict` (line 56)
- `get_topology_config` (line 59) - *Configuración estable para topología adaptable*
- `__init__` (line 118)
- `forward` (line 123)
- `__init__` (line 149)
- `forward` (line 181)
- `__init__` (line 233)
- `forward` (line 242)
- `__init__` (line 269)
- `get_adjacency` (line 288) - *Obtener matriz de adyacencia con estabilidad garantizada*
- `prune_topology` (line 317) - *Poda controlada con protocolo de emergencia*
- `get_density` (line 337) - *Calcular densidad actual de manera estable*
- `__init__` (line 347)
- `_init_weights` (line 400) - *Inicialización estable de pesos*
- `forward` (line 408)

#### `01_topobrain_cpu_v5.py`
**Path:** `01_topobrain_cpu_v5.py`

**Classs:**
- `Config` (line 25)
- `StableSupConLoss` (line 116) - *Versión estable de SupConLoss con manejo de bordes robusto*
- `StableContinuumMemoryCell` (line 147) - *Versión estable y ligera de ContinuumMemoryCell con clamping y normalización*
- `StableSymbioticBasisRefinement` (line 231) - *Refinamiento simbiótico estable con regularización explícita (versión ligera)*
- `StableTopologyManager` (line 267) - *Gestor de topología con protocolos de supervivencia (versión optimizada para CPU)*
- `StableTopoBrain` (line 345) - *Implementación estable y ligera de TopoBrain para ablation científico en CPU*

**Functions:**
- `seed_everything` (line 70)
- `get_tabular_loaders` (line 77) - *Dataset tabular controlado con características NOIR simuladas*
- `stable_pgd_attack` (line 475) - *Ataque PGD estable con manejo de gradientes robusto*
- `train_epoch` (line 518) - *Entrenamiento por época con monitoreo detallado*
- `evaluate_model` (line 583) - *Evaluación rigurosa con o sin ataques adversariales*
- `run_scientific_ablation` (line 708) - *Ejecución científica del ablation con control de variables*
- `to_dict` (line 56)
- `get_topology_config` (line 59) - *Configuración estable para topología adaptable*
- `__init__` (line 118)
- `forward` (line 123)
- `__init__` (line 149)
- `forward` (line 181)
- `__init__` (line 233)
- `forward` (line 242)
- `__init__` (line 269)
- `get_adjacency` (line 288) - *Obtener matriz de adyacencia con estabilidad garantizada*
- `prune_topology` (line 317) - *Poda controlada con protocolo de emergencia*
- `get_density` (line 337) - *Calcular densidad actual de manera estable*
- `__init__` (line 347)
- `_init_weights` (line 400) - *Inicialización estable de pesos*
- `forward` (line 408)

#### `01_topobrain_cpu_v6.py`
**Path:** `01_topobrain_cpu_v6.py`

**Classs:**
- `MicroConfig` (line 33) - *Configuración ultra-ligera para CPU*
- `MicroSupConLoss` (line 137) - *SupCon ultra-eficiente con estabilidad numérica*
- `MicroContinuumCell` (line 163) - *ContinuumMemoryCell ultra-compacto con predicción semántica.
Params: ~4*4 (W_slow) + 4*4 (V_slow) + 4*4 (semantic_mem) ≈ 48 params*
- `MicroSymbioticBasis` (line 214) - *Refinamiento simbiótico con 2 átomos de base.
Params: 2*dim (basis) + dim*dim (Q) + dim*dim (K) ≈ 2*dim + 2*dim² = 40 params (dim=4)*
- `MicroTopology` (line 253) - *Topología dinámica para Grid 2x2 (4 nodos).
Params: 4x4 = 16 (matriz de adyacencia aprendible)*
- `MicroTopoBrain` (line 286) - *TopoBrain ultra-ligero con matemática completa.

PRESUPUESTO DE PARÁMETROS:
- Input embed: 12*16 = 192
- Topology: 4*4 = 16
- Node processor (ContinuumCell si activo): ~48
- Cell processor (MGF si activo): ~48
- Symbiotic (si activo): ~40
- SupCon head (si activo): 16*8 + 8*4 = 160
- Readout: 16*3 = 48

TOTAL: ~200-550 params (base) hasta ~3k-5k (full)*
- `AblationMatrix` (line 597) - *Matriz de ablación científica con 3 niveles de profundidad.

NIVEL 1: COMPONENTES AISLADOS (Control Ceteris Paribus)
- Mide contribución individual de cada componente
- N experimentos = 1 (baseline) + M (componentes)

NIVEL 2A: PARES SINÉRGICOS (Combinaciones 2 a 2)
- Detecta sinergias (+) y antagonismos (-)
- N experimentos = C(M, 2)

NIVEL 2B: TRÍADAS HIPOTÉTICAS (Combinaciones de 3)
- Explora interacciones de orden superior
- N experimentos = selección estratégica

NIVEL 3: ABLACIÓN INVERSA (Quitar 1 del modelo completo)
- Identifica componentes ESENCIALES vs REDUNDANTES
- N experimentos = M*
- `ScientificAnalyzer` (line 744) - *Análisis estadístico con significancia y tamaño de efecto*

**Functions:**
- `seed_everything` (line 95) - *Control de reproducibilidad*
- `get_micro_dataset` (line 104) - *Dataset tabular controlado con validación cruzada*
- `compute_effect_size` (line 126) - *Cohen's d para medir tamaño del efecto*
- `micro_pgd_attack` (line 432) - *PGD ultra-eficiente para CPU*
- `train_epoch_micro` (line 462) - *Entrenamiento por época*
- `evaluate_micro` (line 517) - *Evaluación con opción adversarial*
- `train_with_cv` (line 539) - *Entrenamiento con validación cruzada estratificada.
Retorna: lista de resultados por fold.*
- `run_scientific_ablation_study` (line 850) - *Ejecutor completo del estudio de ablación con análisis científico.

FLUJO:
1. Cargar matriz de experimentos (30 configuraciones)
2. Por cada configuración:
   - Entrenar con CV (3 folds)
   - Calcular estadísticas (mean, std, CI95)
   - Guardar resultados
3. Análisis de 3 niveles:
   - Nivel 1: Contribución individual (t-tests vs baseline)
   - Nivel 2: Sinergias/antagonismos (comparación aditiva)
   - Nivel 3: Criticidad (ablación inversa)
4. Generar reporte científico*
- `to_dict` (line 79)
- `component_signature` (line 82) - *Firma única de componentes activos*
- `__init__` (line 139)
- `forward` (line 144)
- `__init__` (line 168)
- `forward` (line 184)
- `__init__` (line 219)
- `forward` (line 232)
- `__init__` (line 258)
- `get_adjacency` (line 276) - *Matriz de adyacencia normalizada*
- `__init__` (line 301)
- `_init_weights` (line 353)
- `count_parameters` (line 360) - *Contar parámetros entrenables*
- `forward` (line 364)
- `level1_isolated` (line 628) - *NIVEL 1: Componentes aislados (6 experimentos)*
- `level2a_pairs` (line 641) - *NIVEL 2A: Todos los pares (10 experimentos = C(5,2))*
- `level2b_strategic_triads` (line 665) - *NIVEL 2B: Tríadas estratégicas (8 experimentos selectos)

HIPÓTESIS BASADAS EN ANÁLISIS PREVIO:
1. Plasticity es fuerte SOLO → probar con 1 componente adicional
2. Continuum + Symbiotic mostraron cooperación → añadir tercero
3. Evitar SupCon en combinaciones (causa antagonismo)*
- `level3_inverse_ablation` (line 694) - *NIVEL 3: Ablación inversa (5 experimentos)
Modelo completo MENOS un componente → detecta criticidad*
- `level3_full_model` (line 719) - *Modelo completo (referencia máxima)*
- `get_complete_matrix` (line 730) - *Matriz completa de ablación (30 experimentos)*
- `compute_statistics` (line 748) - *Análisis estadístico por experimento.

Args:
    results_list: Lista de resultados de CV folds

Returns:
    Dict con mean, std, CI95, etc.*
- `ttest_vs_baseline` (line 774) - *t-test pareado vs baseline.

Returns:
    (t_statistic, p_value, cohens_d)*
- `detect_synergy` (line 786) - *Detecta sinergia no-lineal.

Synergy = PGD(A+B) - [PGD(A) + PGD(B) - PGD(Baseline)]

> +5%  → Cooperación fuerte
-5~+5% → Aditivo
< -5%  → Antagonismo*
- `rank_components_by_criticality` (line 809) - *Ranking de criticidad basado en ablación inversa.

Criticality = PGD(Full) - PGD(Full_Without_X)

> +10%  → ESENCIAL
+5~+10% → IMPORTANTE
-5~+5%  → OPCIONAL
< -5%   → PERJUDICIAL*

#### `01_topobrain_cpu_v7.py`
**Path:** `01_topobrain_cpu_v7.py`

**Classs:**
- `Config` (line 42)
- `SymbioticBasis` (line 94)
- `DynamicTopology` (line 124)
- `TopoBrainCPU` (line 186)

**Functions:**
- `setup_device` (line 24)
- `pgd_attack` (line 256)
- `train_epoch` (line 285)
- `evaluate` (line 321)
- `get_dataset` (line 351)
- `main` (line 393)
- `to_dict` (line 86)
- `__init__` (line 95)
- `forward` (line 106)
- `__init__` (line 125)
- `_create_grid_mask` (line 136)
- `get_adjacency` (line 153)
- `prune_connections` (line 158)
- `get_density` (line 178)
- `__init__` (line 187)
- `_init_weights` (line 216)
- `count_parameters` (line 223)
- `forward` (line 226)

#### `01_topobrain_cpu_v8.py`
**Path:** `01_topobrain_cpu_v8.py`

**Classs:**
- `Config` (line 28)
- `SymbioticBasis` (line 62)
- `DynamicTopology` (line 88)
- `TopoBrainReal` (line 144)
- `Wrapper` (line 424)

**Functions:**
- `pgd_attack` (line 223)
- `train_topobrain` (line 252)
- `export_for_onnxruntime` (line 410) - *Exporta para ONNX Runtime (más moderno que OpenCV)*
- `test_with_onnxruntime` (line 465) - *Inferencia usando ONNX Runtime*
- `main` (line 547)
- `__init__` (line 63)
- `forward` (line 76)
- `__init__` (line 89)
- `_create_grid_mask` (line 98)
- `get_adjacency` (line 115)
- `prune_connections` (line 120)
- `get_density` (line 140)
- `__init__` (line 145)
- `_init_weights` (line 174)
- `forward` (line 181)
- `forward_with_metrics` (line 210)
- `__init__` (line 425)
- `forward` (line 429)

#### `01_topobrain_ganador_gpu_v1.py`
**Path:** `01_topobrain_ganador_gpu_v1.py`

**Classs:**
- `GPUConfig` (line 70) - *Configuración optimizada para AMD GPU basada en estudio científico*
- `GPUSymbioticBasis` (line 127) - *Symbiotic Basis escalado para GPU.
Proyección ortogonal con más átomos para mayor capacidad.*
- `DynamicTopology` (line 179) - *Topología dinámica adaptativa para Grid 8x8.
Aprende qué conexiones mantener/podar durante entrenamiento.*
- `TopoBrainGPU` (line 278) - *TopoBrain escalado para GPU AMD con configuración ganadora.

ARQUITECTURA:
- Grid 8x8 (64 nodos)
- Embed dim: 16
- Plasticity: Topología adaptativa
- Symbiotic: Refinamiento ortogonal
- NO Continuum, NO MGF, NO SupCon (según estudio)

PARÁMETROS ESTIMADOS: ~50k-80k*

**Functions:**
- `setup_amd_device` (line 31) - *Configura PyTorch para usar GPU AMD con ROCm/OpenCL.
Fallback a CPU si no está disponible.*
- `pgd_attack_gpu` (line 405) - *PGD attack optimizado para GPU.

Args:
    model: Modelo TopoBrain
    x: [batch, features]
    y: [batch] - Labels
    eps: Perturbación máxima
    steps: Pasos de iteración
    plasticity: Factor para topología

Returns:
    x_adv: [batch, features] - Ejemplos adversariales*
- `train_epoch_gpu` (line 461) - *Entrenamiento por época en GPU*
- `evaluate_gpu` (line 517) - *Evaluación en GPU*
- `get_gpu_dataset` (line 547) - *Dataset sintético para GPU*
- `main` (line 591) - *Ejecutar POC completa*
- `to_dict` (line 119)
- `__init__` (line 132)
- `forward` (line 147) - *Args:
    x: [batch, dim]
Returns:
    x_clean: [batch, dim] - Proyección limpia
    entropy: scalar - Entropía de pesos
    ortho: scalar - Pérdida de ortogonalidad*
- `__init__` (line 184)
- `_create_grid_mask` (line 200) - *Crea máscara de vecindad para grid NxN*
- `get_adjacency` (line 219) - *Obtiene matriz de adyacencia normalizada.

Args:
    plasticity: Factor de modulación (0=frozen, 1=full adaptive)
Returns:
    adj: [num_nodes, num_nodes] - Matriz normalizada por grado*
- `prune_connections` (line 237) - *Poda conexiones débiles (llamar cada N epochs).

Args:
    threshold: Umbral de poda (sigmoid(weight) < threshold)
Returns:
    num_pruned: Número de conexiones podadas*
- `get_density` (line 269) - *Densidad actual de conexiones*
- `__init__` (line 291)
- `_init_weights` (line 335) - *Inicialización Kaiming para activaciones GELU*
- `count_parameters` (line 343) - *Cuenta parámetros entrenables*
- `forward` (line 347) - *Forward pass completo.

Args:
    x: [batch, n_features]
    plasticity: Factor de adaptación topológica (0-1)

Returns:
    logits: [batch, n_classes]
    entropy: scalar - Entropía de Symbiotic
    ortho: scalar - Regularización ortogonal*

#### `02_perceptron.py`
**Path:** `02_perceptron.py`

**Classs:**
- `Perceptron` (line 31)

**Functions:**
- `__init__` (line 32)
- `predict` (line 37)
- `train_step` (line 42)
- `accuracy` (line 51)

#### `03_backpropagation.py`
**Path:** `03_backpropagation.py`

**Functions:**
- `sigmoid` (line 33)
- `sigmoid_derivative` (line 38)
- `forward` (line 53)
- `backward` (line 63)
- `compute_metrics` (line 88)

#### `04_cnn_lenet.py`
**Path:** `04_cnn_lenet.py`

**Classs:**
- `LeNet5Like` (line 45)

**Functions:**
- `__init__` (line 46)
- `forward` (line 60)

#### `05_svm_rbf.py`
**Path:** `05_svm_rbf.py`

*No symbols extracted*

#### `06_lstm_char.py`
**Path:** `06_lstm_char.py`

**Classs:**
- `CharLSTM` (line 89)

**Functions:**
- `create_batches` (line 63)
- `__init__` (line 90)
- `forward` (line 102)

#### `07_random_forest.py`
**Path:** `07_random_forest.py`

*No symbols extracted*

#### `08_vae_mnist.py`
**Path:** `08_vae_mnist.py`

**Classs:**
- `VAE` (line 40)

**Functions:**
- `vae_loss` (line 72)
- `__init__` (line 41)
- `encode` (line 51)
- `reparameterize` (line 55)
- `decode` (line 60)
- `forward` (line 64)

#### `09_transformer_mini.py`
**Path:** `09_transformer_mini.py`

**Classs:**
- `MultiHeadAttention` (line 51)
- `FeedForward` (line 83)
- `MiniTransformerEncoder` (line 96)
- `MiniTransformer` (line 131)

**Functions:**
- `generate_copy_data` (line 29)
- `__init__` (line 52)
- `forward` (line 63)
- `__init__` (line 84)
- `forward` (line 90)
- `__init__` (line 97)
- `_create_positional_encoding` (line 107)
- `forward` (line 115)
- `__init__` (line 132)
- `forward` (line 137)

#### `10_gan_mnist_lite.py`
**Path:** `10_gan_mnist_lite.py`

**Classs:**
- `Generator` (line 37)
- `Discriminator` (line 57)

**Functions:**
- `__init__` (line 38)
- `forward` (line 51)
- `__init__` (line 58)
- `forward` (line 74)

#### `11_bert_tiny.py`
**Path:** `11_bert_tiny.py`

**Classs:**
- `MultiHeadAttention` (line 107)
- `FeedForward` (line 133)
- `BERTLayer` (line 143)
- `TinyBERT` (line 158)

**Functions:**
- `tokenize_sentence` (line 76)
- `pad_sequence` (line 88)
- `mask_tokens` (line 186)
- `__init__` (line 108)
- `forward` (line 119)
- `__init__` (line 134)
- `forward` (line 140)
- `__init__` (line 144)
- `forward` (line 151)
- `__init__` (line 159)
- `_create_positional_encoding` (line 167)
- `forward` (line 175)

#### `12_diffusion_minimal.py`
**Path:** `12_diffusion_minimal.py`

**Classs:**
- `SimpleDiffusionNet` (line 54)

**Functions:**
- `q_sample` (line 81) - *Muestrea x_t dado x_0 y timestep t.
x_0: (B, C, H, W)
t: (B,)*
- `__init__` (line 55)
- `forward` (line 65)

#### `13_nested_hope.py`
**Path:** `13_nested_hope.py`

**Classs:**
- `Config` (line 14) - *Configuración centralizada basada en el paper (Secciones 7-9)*
- `DeltaGradientDescent` (line 66) - *Implementación de DGD según Eq. 121:
W_{t+1} = W_t(I - α_t x_t x_t^T) - β ∇_{y_t} L(W_t; x_t) x_t^T

A diferencia del GD estándar, DGD incluye:
1. Decaimiento adaptativo basado en el estado actual (término αI)
2. Dependencia de muestras anteriores (no asume i.i.d.)*
- `SelfModifyingMemory` (line 115) - *Implementación completa de Self-Modifying Deep Associative Memory

Paper: Sección 8.1, Ecuaciones 83-88
- Genera sus propias proyecciones (k, v, q, η, α)
- Genera valores propios auto-referenciales (Eq. 84)
- Aplica regla Delta para actualización (Eq. 88)*
- `ContinuumMemorySystem` (line 258) - *Sistema de memoria continuo con múltiples frecuencias de actualización

Paper: Sección 7.1
- Niveles con diferentes frecuencias (rápido → lento)
- Actualización condicional basada en chunk_size
- Conexión secuencial (Eq. 73) o independiente (Eq. 74)*
- `HopeModel` (line 337) - *Arquitectura Hope: Self-Modifying Memory + CMS

Paper: Sección 8.3, Figura 5*
- `HopeTrainer` (line 432) - *Sistema de entrenamiento con ablación automática y métricas científicas*

**Functions:**
- `setup_device` (line 44) - *Configuración automática de dispositivo*
- `set_seed` (line 54) - *Reproducibilidad completa*
- `run_ablation_study` (line 564) - *Ejecuta estudio de ablación completo

Configuraciones testeadas:
1. Hope completo (Self-Mod + CMS + DGD)
2. Sin Self-Modifying
3. Sin CMS
4. Sin DGD
5. Baseline (sin ninguno)*
- `apply_update` (line 77) - *Aplica la regla DGD a un gradiente

Args:
    grad: Gradiente actual ∇L
    param: Parámetro W_t
    x_normalized: Input normalizado (||x|| = λ)
    eta: Learning rate adaptativo (por muestra)
    alpha: Retention gate (por muestra)
    lambda_norm: Norma de x (default: 1.0 para L2-norm)

Returns:
    Gradiente modificado según DGD*
- `__init__` (line 125)
- `_make_memory_module` (line 162) - *Crea un módulo de memoria (MLP de 2 capas con residual)*
- `forward` (line 170) - *Forward pass con actualización chunk-wise (Sección 8.2)

Args:
    x: (B, S, D) - Input tokens embebidos
    prev_states: Estados previos de memorias
    
Returns:
    output: (B, S, D)
    new_states: Estados actualizados*
- `__init__` (line 268)
- `forward` (line 296) - *Forward pass con actualizaciones multi-frecuencia

Args:
    x: (B, S, D)
    global_step: Paso global de entrenamiento
    
Returns:
    output: (B, S, D)*
- `__init__` (line 344)
- `reset_states` (line 390) - *Reset de estados internos (para nuevas secuencias)*
- `forward` (line 394) - *Forward pass completo

Args:
    x: (B, S) - Input token IDs
    global_step: Paso global
    return_internals: Si retornar estados internos (para análisis)*
- `__init__` (line 437)
- `train_epoch` (line 461) - *Entrena una época completa

Returns:
    avg_loss, avg_acc, new_global_step*
- `evaluate` (line 536) - *Evaluación sin gradientes*

#### `13_nested_kearning_gpu.py`
**Path:** `13_nested_kearning_gpu.py`

**Classs:**
- `SelfModifyingMemory` (line 53)
- `ContinuumMemorySystem` (line 74)
- `HopeModel` (line 97)

**Functions:**
- `__init__` (line 54)
- `forward` (line 64)
- `__init__` (line 75)
- `forward` (line 87)
- `__init__` (line 98)
- `forward` (line 104)

#### `13_nested_learning.py`
**Path:** `13_nested_learning.py`

**Classs:**
- `SelfModifyingMemory` (line 51)
- `ContinuumMemorySystem` (line 86)
- `HopeModel` (line 114)

**Functions:**
- `__init__` (line 52)
- `forward` (line 62) - *x: (B, S)
update_mask: None o máscara booleana para actualizaciones condicionales
Retorna logits y parámetros internos (para depuración/futuras extensiones)*
- `__init__` (line 87)
- `forward` (line 100) - *x: (B, S, D)
global_step: int, paso global de entrenamiento*
- `__init__` (line 115)
- `forward` (line 122)

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py`

**Classs:**
- `LanguageMetrics` (line 44) - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 117) - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 382)
- `RightHemisphere` (line 458)
- `LeftHemisphere` (line 473)
- `CorpusCallosum` (line 544)
- `NeuroLogosBicameralStable` (line 581)
- `EnhancedDiagnostics` (line 602)
- `Flickr8kDataset` (line 727)

**Functions:**
- `compute_loss` (line 20)
- `build_vocab_flickr` (line 764)
- `setup_flickr8k` (line 782)
- `train_with_metrics` (line 854)
- `sentence_bleu` (line 48) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 82) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 91) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 104) - *Jaccard similarity entre palabras*
- `__init__` (line 120)
- `triangulate_signals` (line 125) - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 157) - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 161) - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 223) - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 383)
- `forward` (line 397)
- `hebbian_update` (line 404)
- `update_physiology_advanced` (line 434)
- `__init__` (line 459)
- `forward` (line 467)
- `__init__` (line 474)
- `forward` (line 502)
- `_get_init_state` (line 539)
- `__init__` (line 545)
- `forward` (line 568)
- `__init__` (line 582)
- `forward` (line 588)
- `__init__` (line 603)
- `measure_callosal_flow` (line 613)
- `calculate_synergy` (line 622)
- `calculate_health` (line 631)
- `update` (line 640)
- `get_recent_avg` (line 645)
- `report` (line 650)
- `__init__` (line 728)
- `__len__` (line 745)
- `__getitem__` (line 748)

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py`

**Classs:**
- `LanguageMetrics` (line 44) - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 117) - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 383)
- `RightHemisphere` (line 509)
- `LeftHemisphere` (line 524)
- `CorpusCallosum` (line 645)
- `NeuroLogosBicameralStable` (line 700)
- `EnhancedDiagnostics` (line 721)
- `EpisodicMemoryBuffer` (line 844)
- `Flickr8kDataset` (line 891)

**Functions:**
- `compute_loss` (line 20)
- `build_vocab_flickr` (line 928)
- `setup_flickr8k` (line 946)
- `train_with_metrics` (line 1021)
- `sentence_bleu` (line 48) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 82) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 91) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 104) - *Jaccard similarity entre palabras*
- `__init__` (line 120)
- `triangulate_signals` (line 125) - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 157) - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 161) - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 224) - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 384)
- `forward` (line 422)
- `hebbian_update` (line 444)
- `update_physiology_advanced` (line 483)
- `__init__` (line 510)
- `forward` (line 518)
- `__init__` (line 525)
- `forward` (line 573)
- `_get_init_state` (line 631)
- `__init__` (line 646)
- `forward` (line 678)
- `__init__` (line 701)
- `forward` (line 707)
- `__init__` (line 722)
- `measure_callosal_flow` (line 732)
- `calculate_synergy` (line 741)
- `calculate_health` (line 750)
- `update` (line 759)
- `get_recent_avg` (line 764)
- `report` (line 769)
- `__init__` (line 845)
- `compute_surprise` (line 851)
- `add` (line 862)
- `sample` (line 872)
- `__init__` (line 892)
- `__len__` (line 909)
- `__getitem__` (line 912)

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py`

**Classs:**
- `LanguageMetrics` (line 44) - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 117) - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 392)
- `RightHemisphere` (line 518)
- `LeftHemisphere` (line 533)
- `CorpusCallosum` (line 709)
- `NeuroLogosBicameralStable` (line 765)
- `EnhancedDiagnostics` (line 786)
- `EpisodicMemoryBuffer` (line 907)
- `Flickr8kDataset` (line 954)

**Functions:**
- `compute_loss` (line 20)
- `build_vocab_flickr` (line 991)
- `setup_flickr8k` (line 1009)
- `train_with_metrics` (line 1081)
- `sentence_bleu` (line 48) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 82) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 91) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 104) - *Jaccard similarity entre palabras*
- `__init__` (line 120)
- `triangulate_signals` (line 125) - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 157) - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 161) - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 224) - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 393)
- `forward` (line 431)
- `hebbian_update` (line 453)
- `update_physiology_advanced` (line 492)
- `__init__` (line 519)
- `forward` (line 527)
- `__init__` (line 534)
- `beam_search_decode` (line 584)
- `forward` (line 657)
- `_get_init_state` (line 695)
- `__init__` (line 710)
- `forward` (line 742)
- `__init__` (line 766)
- `forward` (line 772)
- `__init__` (line 787)
- `measure_callosal_flow` (line 797)
- `calculate_synergy` (line 806)
- `calculate_health` (line 815)
- `update` (line 824)
- `get_recent_avg` (line 829)
- `report` (line 834)
- `__init__` (line 908)
- `compute_surprise` (line 914)
- `add` (line 925)
- `sample` (line 935)
- `__init__` (line 955)
- `__len__` (line 972)
- `__getitem__` (line 975)

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py`

**Classs:**
- `NeurocognitiveSystem` (line 49) - *Sistema neurocognitivo que complementa al sistema médico
para optimizar el aprendizaje lingüístico*
- `LinguisticFeedbackLoop` (line 220) - *Sistema que integra métricas lingüísticas en el proceso de aprendizaje*
- `LanguageMetrics` (line 309) - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 382) - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 657)
- `RightHemisphere` (line 783)
- `LeftHemisphere` (line 798)
- `CorpusCallosum` (line 978)
- `NeuroLogosBicameralStable` (line 1034)
- `EnhancedDiagnostics` (line 1055)
- `EpisodicMemoryBuffer` (line 1192)
- `Flickr8kDataset` (line 1239)

**Functions:**
- `compute_loss` (line 20) - *Función de pérdida extendida que incorpora recompensa lingüística*
- `build_vocab_flickr` (line 1276)
- `setup_flickr8k` (line 1294)
- `train_with_metrics` (line 1366)
- `__init__` (line 55)
- `assess_cognitive_state` (line 65) - *Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas*
- `apply_cognitive_intervention` (line 113) - *Aplica intervenciones cognitivas basadas en el estado lingüístico*
- `__init__` (line 223)
- `compute_linguistic_reward` (line 232) - *Calcula una recompensa combinada basada en CIDEr y SPICE
que puede usarse para guiar el entrenamiento*
- `compute_cider` (line 254) - *Versión simplificada de CIDEr para uso en entrenamiento*
- `compute_spice` (line 281) - *Versión simplificada de SPICE para uso en entrenamiento*
- `_get_ngrams` (line 295) - *Extrae n-gramas de una oración*
- `sentence_bleu` (line 313) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 347) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 356) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 369) - *Jaccard similarity entre palabras*
- `__init__` (line 385)
- `triangulate_signals` (line 390) - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 422) - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 426) - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 489) - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 658)
- `forward` (line 696)
- `hebbian_update` (line 718)
- `update_physiology_advanced` (line 757)
- `__init__` (line 784)
- `forward` (line 792)
- `__init__` (line 799)
- `beam_search_decode` (line 849)
- `forward` (line 925)
- `_get_init_state` (line 963)
- `__init__` (line 979)
- `forward` (line 1011)
- `__init__` (line 1035)
- `forward` (line 1041)
- `__init__` (line 1056)
- `measure_callosal_flow` (line 1067)
- `calculate_synergy` (line 1076)
- `calculate_health` (line 1085)
- `update` (line 1094)
- `get_recent_avg` (line 1099)
- `report` (line 1104)
- `__init__` (line 1193)
- `compute_surprise` (line 1199)
- `add` (line 1210)
- `sample` (line 1220)
- `__init__` (line 1240)
- `__len__` (line 1257)
- `__getitem__` (line 1260)

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py`

**Classs:**
- `NeurocognitiveSystem` (line 49) - *Sistema neurocognitivo que complementa al sistema médico
para optimizar el aprendizaje lingüístico*
- `LinguisticFeedbackLoop` (line 325) - *Sistema que integra métricas lingüísticas en el proceso de aprendizaje.
Versión optimizada con caché de dos niveles para minimizar cálculos repetitivos.*
- `LanguageMetrics` (line 479) - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 552) - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 827)
- `RightHemisphere` (line 973)
- `LeftHemisphere` (line 988)
- `CorpusCallosum` (line 1244)
- `NeuroLogosBicameralStable` (line 1420)
- `EnhancedDiagnostics` (line 1446)
- `EpisodicMemoryBuffer` (line 1661)
- `Flickr8kDataset` (line 1708)

**Functions:**
- `compute_loss` (line 20) - *Función de pérdida extendida que incorpora recompensa lingüística*
- `build_vocab_flickr` (line 1745)
- `setup_flickr8k` (line 1763)
- `compute_alignment_loss` (line 1834) - *Pérdida auxiliar para forzar alineación entre características visuales
y canales estructurales del callosum durante épocas tempranas*
- `train_with_metrics` (line 1858)
- `__init__` (line 55)
- `assess_cognitive_state` (line 76) - *Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas*
- `evaluate_gate_state` (line 124) - *MEJORA: Evaluar estado del gate con sistema inmune*
- `update_trauma_memory` (line 142) - *MEJORA: Actualizar memoria traumática basada en resultados*
- `apply_stochastic_perturbation` (line 155) - *MEJORA: Aplicar micro-perturbaciones estocásticas*
- `apply_cognitive_intervention` (line 171) - *Aplica intervenciones cognitivas basadas en el estado lingüístico*
- `__init__` (line 331)
- `compute_linguistic_reward` (line 347) - *Calcula una recompensa combinada basada en CIDEr y SPICE.
Utiliza caché para acelerar el cálculo de métricas.*
- `compute_cider` (line 389) - *Versión simplificada de CIDEr para uso en entrenamiento.
Optimizada con caché de n-gramas.*
- `compute_spice` (line 427) - *Versión simplificada de SPICE para uso en entrenamiento.
Usa Jaccard similarity como proxy semántico.*
- `_get_ngrams` (line 443) - *Extrae n-gramas de una oración*
- `get_cache_stats` (line 452) - *Obtiene estadísticas del sistema de caché*
- `sentence_bleu` (line 483) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 517) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 526) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 539) - *Jaccard similarity entre palabras*
- `__init__` (line 555)
- `triangulate_signals` (line 560) - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 592) - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 596) - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 659) - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 828)
- `forward` (line 871)
- `hebbian_update` (line 893)
- `update_physiology_advanced` (line 932)
- `__init__` (line 974)
- `forward` (line 982)
- `__init__` (line 989)
- `beam_search_decode` (line 1066)
- `forward` (line 1147)
- `_apply_structural_attention` (line 1185) - *Aplica atención específica para cada canal estructural (objetos, acciones, escena).
Versión optimizada con matemática robusta y eficiente.*
- `_get_init_state` (line 1230)
- `__init__` (line 1245)
- `forward` (line 1308)
- `update_channel_fatigue` (line 1381) - *MEJORA: Actualizar fatiga específica por canal*
- `adjust_gates_by_fatigue` (line 1404) - *MEJORA: Ajustar gates basado en fatiga de cada canal*
- `__init__` (line 1421)
- `forward` (line 1427)
- `__init__` (line 1447)
- `measure_callosal_flow` (line 1460)
- `calculate_synergy` (line 1490)
- `calculate_health` (line 1499)
- `update` (line 1508)
- `get_recent_avg` (line 1519)
- `visualize_fatigue_distribution` (line 1538) - *MEJORA: Visualizar distribución de fatiga entre canales*
- `report` (line 1567)
- `__init__` (line 1662)
- `compute_surprise` (line 1668)
- `add` (line 1679)
- `sample` (line 1689)
- `__init__` (line 1709)
- `__len__` (line 1726)
- `__getitem__` (line 1729)

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py`

**Classs:**
- `NeurocognitiveSystem` (line 53) - *Sistema neurocognitivo que complementa al sistema médico
para optimizar el aprendizaje lingüístico y el razonamiento*
- `LinguisticFeedbackLoop` (line 396) - *Sistema que integra métricas lingüísticas en el proceso de aprendizaje.
Versión optimizada con caché de dos niveles para minimizar cálculos repetitivos.*
- `LanguageMetrics` (line 550) - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 623) - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 898)
- `RightHemisphere` (line 1044)
- `LeftHemisphere` (line 1059)
- `CorpusCallosum` (line 1469)
- `NeuroLogosBicameralStable` (line 1645)
- `EnhancedDiagnostics` (line 1669)
- `EpisodicMemoryBuffer` (line 1947)
- `Flickr8kDataset` (line 1994)

**Functions:**
- `compute_loss` (line 20) - *Función de pérdida extendida con MTP y recompensa lingüística*
- `build_vocab_flickr` (line 2031)
- `setup_flickr8k` (line 2049)
- `compute_alignment_loss` (line 2120) - *Pérdida auxiliar para forzar alineación entre características visuales
y canales estructurales del callosum durante épocas tempranas*
- `train_with_metrics` (line 2144)
- `__init__` (line 59)
- `assess_reasoning_state` (line 85) - *Evalúa el estado del sistema de razonamiento*
- `assess_cognitive_state` (line 128) - *Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas*
- `evaluate_gate_state` (line 167) - *Evaluar estado del gate con sistema inmune*
- `update_trauma_memory` (line 182) - *Actualizar memoria traumática basada en resultados*
- `apply_stochastic_perturbation` (line 192) - *Aplicar micro-perturbaciones estocásticas*
- `apply_cognitive_intervention` (line 213) - *Aplica intervenciones cognitivas basadas en el estado lingüístico y de razonamiento*
- `__init__` (line 402)
- `compute_linguistic_reward` (line 418) - *Calcula una recompensa combinada basada en CIDEr y SPICE.
Utiliza caché para acelerar el cálculo de métricas.*
- `compute_cider` (line 460) - *Versión simplificada de CIDEr para uso en entrenamiento.
Optimizada con caché de n-gramas.*
- `compute_spice` (line 498) - *Versión simplificada de SPICE para uso en entrenamiento.
Usa Jaccard similarity como proxy semántico.*
- `_get_ngrams` (line 514) - *Extrae n-gramas de una oración*
- `get_cache_stats` (line 523) - *Obtiene estadísticas del sistema de caché*
- `sentence_bleu` (line 554) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 588) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 597) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 610) - *Jaccard similarity entre palabras*
- `__init__` (line 626)
- `triangulate_signals` (line 631) - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 663) - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 667) - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 730) - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 899)
- `forward` (line 942)
- `hebbian_update` (line 964)
- `update_physiology_advanced` (line 1003)
- `__init__` (line 1045)
- `forward` (line 1053)
- `__init__` (line 1060)
- `_apply_chain_of_thought` (line 1190) - *Aplica cadena de pensamiento para mejorar el razonamiento*
- `_apply_multi_token_prediction` (line 1230) - *Multi-Token Prediction: predice múltiples tokens futuros simultáneamente
CRÍTICO: Mantiene dimensiones consistentes con input original*
- `beam_search_decode` (line 1292)
- `forward` (line 1370)
- `_apply_structural_attention` (line 1420) - *Aplica atención específica para cada canal estructural (objetos, acciones, escena).
Versión optimizada con matemática robusta y eficiente.*
- `_get_init_state` (line 1456)
- `__init__` (line 1470)
- `forward` (line 1533)
- `update_channel_fatigue` (line 1606) - *MEJORA: Actualizar fatiga específica por canal*
- `adjust_gates_by_fatigue` (line 1629) - *MEJORA: Ajustar gates basado en fatiga de cada canal*
- `__init__` (line 1646)
- `forward` (line 1652)
- `__init__` (line 1670)
- `measure_callosal_flow` (line 1686)
- `evaluate_reasoning_quality` (line 1709) - *Evalúa la calidad del razonamiento en textos generados*
- `calculate_synergy` (line 1750)
- `calculate_health` (line 1759)
- `update` (line 1768)
- `get_recent_avg` (line 1778)
- `visualize_fatigue_distribution` (line 1795) - *Visualizar distribución de fatiga entre canales*
- `visualize_reasoning_metrics` (line 1823) - *Visualizar métricas de razonamiento*
- `report` (line 1836)
- `__init__` (line 1948)
- `compute_surprise` (line 1954)
- `add` (line 1965)
- `sample` (line 1975)
- `__init__` (line 1995)
- `__len__` (line 2012)
- `__getitem__` (line 2015)

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py`

**Classs:**
- `EpisodicMemoryBuffer` (line 58) - *Buffer que almacena ejemplos sorpresivos para replay estratégico*
- `NeurocognitiveSystem` (line 112)
- `LinguisticFeedbackLoop` (line 291) - *Sistema de caché optimizado para métricas lingüísticas*
- `LanguageMetrics` (line 413) - *Métricas clásicas de evaluación*
- `TriangulatedMedicalSystem` (line 475)
- `StableLiquidNeuron` (line 622)
- `RightHemisphere` (line 745)
- `LeftHemisphere` (line 764)
- `CorpusCallosum` (line 1019)
- `NeuroLogosBicameralStable` (line 1147)
- `EnhancedDiagnostics` (line 1171)
- `Flickr8kDataset` (line 1408)

**Functions:**
- `compute_loss` (line 22) - *Función de pérdida extendida con MTP y recompensa lingüística*
- `build_vocab_flickr` (line 1445)
- `setup_flickr8k` (line 1463)
- `compute_alignment_loss` (line 1527) - *Pérdida auxiliar para alineación temprana*
- `train_with_metrics` (line 1547)
- `__init__` (line 61)
- `compute_surprise` (line 67) - *Calcula sorpresa basada en error y apertura del gate*
- `add` (line 79) - *Añade ejemplo si supera umbral y hay capacidad*
- `sample` (line 91) - *Samplea ejemplos con probabilidad proporcional a sorpresa*
- `__init__` (line 113)
- `assess_reasoning_state` (line 128) - *Evalúa estado del sistema de razonamiento*
- `assess_cognitive_state` (line 170) - *Evalúa estado cognitivo lingüístico*
- `apply_cognitive_intervention` (line 206) - *Aplica intervenciones basadas en estado lingüístico y razonamiento*
- `__init__` (line 294)
- `compute_linguistic_reward` (line 308) - *Recompensa combinada CIDEr + SPICE con caché*
- `compute_cider` (line 342) - *CIDEr simplificado con caché de n-gramas*
- `compute_spice` (line 371) - *SPICE simplificado (Jaccard similarity)*
- `_get_ngrams` (line 384) - *Extractor de n-gramas*
- `get_cache_stats` (line 389) - *Estadísticas de caché*
- `sentence_bleu` (line 417) - *BLEU-4 a nivel de oración*
- `token_accuracy` (line 448) - *Precisión token-level*
- `word_overlap` (line 461) - *Jaccard similarity*
- `__init__` (line 476)
- `triangulate_signals` (line 482)
- `count_convergent_signals` (line 492)
- `diagnose_with_triangulation` (line 495)
- `apply_triangulated_intervention` (line 537)
- `_reset_liquid_neuron` (line 610)
- `__init__` (line 623)
- `forward` (line 658)
- `hebbian_update` (line 671)
- `update_physiology_advanced` (line 709)
- `__init__` (line 746)
- `forward` (line 754)
- `__init__` (line 765)
- `forward` (line 840)
- `_greedy_decode` (line 879)
- `_apply_chain_of_thought` (line 912)
- `_apply_multi_token_prediction` (line 944)
- `_apply_structural_attention` (line 986)
- `_get_init_state` (line 1007)
- `__init__` (line 1020)
- `forward` (line 1070)
- `update_channel_fatigue` (line 1113) - *Actualiza fatiga específica por canal*
- `adjust_gates_by_fatigue` (line 1132) - *Ajusta gates basado en fatiga*
- `__init__` (line 1148)
- `forward` (line 1154)
- `__init__` (line 1172)
- `measure_callosal_flow` (line 1187)
- `evaluate_reasoning_quality` (line 1210) - *Evalúa coherencia y consistencia del razonamiento*
- `calculate_synergy` (line 1242)
- `calculate_health` (line 1251)
- `update` (line 1260)
- `get_recent_avg` (line 1270)
- `visualize_fatigue_distribution` (line 1287)
- `visualize_reasoning_metrics` (line 1308)
- `report` (line 1320) - *Genera reporte completo del estado del sistema bicameral*
- `__init__` (line 1409)
- `__len__` (line 1426)
- `__getitem__` (line 1429)

#### `ablation.py`
**Path:** `ablation.py`

**Classs:**
- `TopoBrainCore` (line 17)
- `MiniUnconscious` (line 72)
- `TopoUnconscious` (line 91)
- `SimpleClassifier` (line 124)
- `NeuroLogosCPU` (line 137) - *Ablation levels:
  0: BASELINE     -> MiniUnconscious
  1: +GRID        -> TopoBrain (grid only)
  2: +SYMBIOTIC   -> TopoBrain (grid + symbiotic)
  3: +ADVERSARIAL -> TopoBrain + FGSM (lightweight)*

**Functions:**
- `fgsm_attack` (line 175)
- `train_epoch` (line 188)
- `evaluate` (line 229)
- `run_ablation_cpu` (line 245)
- `__init__` (line 18)
- `_init_grid` (line 39)
- `forward` (line 46)
- `get_metrics` (line 64)
- `__init__` (line 73)
- `forward` (line 87)
- `__init__` (line 92)
- `forward` (line 112)
- `get_metrics` (line 116)
- `__init__` (line 125)
- `forward` (line 129)
- `__init__` (line 145)
- `forward` (line 161)
- `get_metrics` (line 165)

#### `ablation1.py`
**Path:** `ablation1.py`

**Classs:**
- `EliteConfig` (line 20)
- `AblationConfig` (line 60)
- `EpisodicMemory` (line 180) - *Memoria explícita de patrones adversariales*
- `SpectralNormLinear` (line 220) - *Linear con normalización espectral para estabilidad Lipschitz*
- `AdvancedHomeostaticCell` (line 248) - *Neurona con control fisiológico multinivel + memoria*
- `AdaptiveTopology` (line 296) - *Topología que aprende a reconectar bajo ataque*
- `EliteTopoBrain` (line 329)
- `SupConLoss` (line 467)

**Functions:**
- `create_ablation_configs` (line 64) - *Crea un diccionario de configuraciones para cada test de ablación.*
- `seed_everything` (line 150)
- `get_elite_dataset` (line 157) - *Dataset más grande y balanceado con separabilidad controlada*
- `elite_pgd_attack` (line 407) - *PGD con reinicio aleatorio y gradiente centralizado (CORREGIDO)*
- `train_elite_model` (line 504) - *Entrenamiento con curriculum adversarial*
- `run_ablation_study` (line 616)
- `__init__` (line 182)
- `update` (line 190) - *Almacena ejemplos duros*
- `retrieve` (line 203) - *Recupera k vecinos más cercanos*
- `__init__` (line 222)
- `power_iteration` (line 229) - *Aproxima la norma espectral máxima*
- `forward` (line 236)
- `__init__` (line 250)
- `forward` (line 271)
- `__init__` (line 298)
- `forward` (line 317) - *stress ∈ [0,1]: cuánto estrés adversarial*
- `__init__` (line 330)
- `count_parameters` (line 367)
- `forward` (line 370)
- `__init__` (line 468)
- `forward` (line 472)

#### `ablation2.py`
**Path:** `ablation2.py`

**Classs:**
- `SparseCompetitiveLayer` (line 19) - *k-WTA con aprendizaje de importancia de nodos.
Cada nodo tiene un bias de vida útil que incrementa con activación.
Los menos usados son pruning dinámico.*
- `SymbioticRefiner` (line 91) - *Refinamiento ortogonal con normalización de estabilidad.
Versión mejorada con spectral clamping para evitar desvanecimiento.*
- `SparseSymbioticCore` (line 120) - *Reemplaza TopoBrainCore.
Combina SparseCompetitiveLayer + SymbioticRefiner.*
- `BaselineUnconscious` (line 167) - *Mantener para ablation - Encoder sin sparsity*
- `SparseUnconscious` (line 186) - *Encoder con SparseSymbioticCore*
- `ConsciousCore` (line 219)
- `BioDecoder` (line 231) - *Decoder con gating líquido*
- `NeuroLogos_v51` (line 289) - *5 configuraciones para ablation desacoplado:

1. BASELINE-v51: Encoder densa sin sparsity
2. SPARSE-ONLY: Sparse layer SIN symbiotic
3. SYMBIOTIC-ONLY: Symbiotic SIN sparse (capa densa)
4. SPARSE-SYMBIOTIC: Ambos sin adversarial
5. SPARSE-SYMBIOTIC-ADV: Full (mejor versión)*
- `PGDAttack` (line 346)
- `CIFARCaptions_v51` (line 375)

**Functions:**
- `compute_bleu` (line 417) - *BLEU score simplificado para evaluar calidad de generación*
- `train_ablation_v51` (line 466) - *Entrena una configuración específica del ablation study v5.1.
Ahora con BLEU score y métricas de activación por clase.*
- `run_ablation_v51` (line 619) - *Ejecuta ablation study v5.1 con 5 brazos desacoplados.*
- `__init__` (line 25)
- `forward` (line 44)
- `get_metrics` (line 79)
- `__init__` (line 96)
- `forward` (line 106)
- `__init__` (line 125)
- `forward` (line 143)
- `get_metrics` (line 158)
- `__init__` (line 169)
- `forward` (line 182)
- `__init__` (line 188)
- `forward` (line 207)
- `get_metrics` (line 211)
- `__init__` (line 220)
- `forward` (line 225)
- `__init__` (line 233)
- `forward` (line 248)
- `_get_init_state` (line 279)
- `__init__` (line 299)
- `forward` (line 331)
- `get_metrics` (line 340)
- `__init__` (line 347)
- `attack` (line 352)
- `__init__` (line 376)
- `__len__` (line 403)
- `__getitem__` (line 406)
- `ngrams` (line 423)
- `forward_fn` (line 526)

#### `ablation3.py`
**Path:** `ablation3.py`

**Classs:**
- `SparseLayer` (line 21) - *k-WTA con health tracking - Factor S*
- `SymbioticLayer` (line 54) - *Orthogonal refinement - Factor Y*
- `AdversarialWrapper` (line 77) - *Wrapper PGD - Factor A*
- `VisualBackbone` (line 109)
- `ConsciousCore` (line 125)
- `BioDecoder` (line 136)
- `NeuroLogosFactorial` (line 190)
- `CIFARCaptions` (line 295)

**Functions:**
- `train_configuration` (line 338) - *Entrena UNA configuración específica del diseño factorial.

Args:
    config: tuple (use_sparse, use_symbiotic, use_adv)*
- `run_full_factorial` (line 411) - *Ejecuta el ablation factorial completo: 8 combinaciones + 3 inversas.*
- `analyze_results` (line 480) - *Análisis de efectos principales, interacciones y poder explicativo.*
- `__init__` (line 23)
- `forward` (line 33)
- `__init__` (line 56)
- `forward` (line 62)
- `__init__` (line 79)
- `attack` (line 84)
- `__init__` (line 110)
- `forward` (line 122)
- `__init__` (line 126)
- `forward` (line 131)
- `__init__` (line 137)
- `forward` (line 148)
- `_init_state` (line 182)
- `__init__` (line 191)
- `forward` (line 226)
- `train_step` (line 243) - *Paso de entrenamiento con adversarial condicional y doble forward pass seguro*
- `__init__` (line 296)
- `__len__` (line 321)
- `__getitem__` (line 324)
- `model_fn` (line 257)

#### `adversarial_benchmark.py`
**Path:** `adversarial_benchmark.py`

**Classs:**
- `SupConLoss` (line 36)
- `PredictiveErrorCell` (line 73)
- `LearnableAbsenceGating` (line 85)
- `SymbioticBasisRefinement` (line 100)
- `CombinatorialComplexLayer` (line 118)
- `TopoBrainNet` (line 158)

**Functions:**
- `make_adversarial_pgd` (line 269)
- `train_and_eval` (line 290)
- `__init__` (line 37)
- `forward` (line 41)
- `__init__` (line 74)
- `forward` (line 79)
- `__init__` (line 86)
- `forward` (line 95)
- `__init__` (line 101)
- `forward` (line 109)
- `__init__` (line 119)
- `forward` (line 136)
- `__init__` (line 159)
- `_init_grid_topology` (line 191)
- `get_topology` (line 213)
- `calculate_ortho_loss` (line 225)
- `forward` (line 249)
- `lambda_topo` (line 317)
- `lambda_general` (line 321)

#### `apex.py`
**Path:** `apex.py`

**Classs:**
- `DataEnvironment` (line 24)
- `LiquidNeuron` (line 50)
- `SovereignAttention` (line 69)
- `DualPhaseMemory` (line 78)
- `ElasticMemory` (line 93)
- `ChimeraNetwork` (line 136)

**Functions:**
- `seed_everything` (line 14)
- `train_and_audit` (line 160)
- `__init__` (line 25)
- `get_train_batch` (line 35)
- `__init__` (line 51)
- `forward` (line 58)
- `__init__` (line 70)
- `forward` (line 74)
- `__init__` (line 79)
- `forward` (line 82)
- `update` (line 86)
- `__init__` (line 94)
- `register_fisher` (line 101)
- `penalty` (line 125)
- `__init__` (line 137)
- `forward` (line 145)

#### `app.py`
**Path:** `app.py`

**Functions:**
- `train_ai_model` (line 82) - *Entrena un modelo desde cero con todos los detalles de entrenamiento*
- `load_or_train_model` (line 156) - *Carga modelo existente o entrena uno nuevo, y lo actualiza con nuevos datos*
- `apply_ai_predictions` (line 191) - *Aplica predicciones del modelo al DataFrame*
- `apply_ai_predictions` (line 205) - *Aplica predicciones del modelo al DataFrame*
- `analyze_ia_vs_rules` (line 219) - *Analiza discrepancias entre reglas y modelo IA*
- `load_and_clean_data_robust` (line 252) - *Cargar y limpiar los datos de forma robusta*
- `parse_csv_manual` (line 294)
- `executive_kpis` (line 317)
- `strategic_okrs` (line 344)
- `generate_visualizations` (line 377)
- `export_report` (line 409)
- `basic_statistics` (line 454)
- `command_analysis` (line 467)
- `network_analysis` (line 480)
- `temporal_analysis` (line 492)
- `statistical_analysis` (line 500)
- `security_insights` (line 508)
- `main` (line 530)

#### `auto_regulation_working.py`
**Path:** `auto_regulation_working.py`

**Classs:**
- `Config` (line 20)
- `DataEnvironment` (line 38)
- `AutoRegulationSystem` (line 72)
- `PhysioChimeraFixed` (line 103)

**Functions:**
- `seed_everything` (line 28)
- `demo_auto_regulation` (line 193)
- `__init__` (line 39)
- `get_batch` (line 49)
- `get_full` (line 63)
- `get_w2` (line 66)
- `__init__` (line 73)
- `update` (line 78)
- `get_stability` (line 95)
- `__init__` (line 104)
- `forward` (line 129)

#### `bicamera.py.py`
**Path:** `bicamera.py.py`

**Classs:**
- `LiquidNeuron` (line 107)
- `RightHemisphere` (line 180)
- `LeftHemisphere` (line 201)
- `CorpusCallosum` (line 298)
- `NeuroLogosBicameral` (line 313)
- `NeuralDiagnostics` (line 334)
- `Flickr8kDataset` (line 404)
- `LifeCycle` (line 466)

**Functions:**
- `setup_flickr8k` (line 34) - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 443)
- `train_bicameral` (line 481)
- `__init__` (line 108)
- `forward` (line 125)
- `consolidate_svd` (line 154)
- `__init__` (line 181)
- `forward` (line 192)
- `__init__` (line 202)
- `forward` (line 224)
- `_get_init_state` (line 278)
- `_top_p_filtering` (line 283)
- `__init__` (line 299)
- `forward` (line 307)
- `__init__` (line 314)
- `forward` (line 320)
- `__init__` (line 335)
- `measure_callosal_flow` (line 346)
- `measure_vocab_diversity` (line 353)
- `update` (line 357)
- `get_recent_avg` (line 362)
- `report` (line 367)
- `__init__` (line 405)
- `__len__` (line 423)
- `__getitem__` (line 426)
- `__init__` (line 467)
- `get_plasticity` (line 470)

#### `bicameral.py`
**Path:** `bicameral.py`

**Classs:**
- `HomeostaticRegulator` (line 100)
- `PhysioNeuron` (line 123)
- `RightHemisphere` (line 164)
- `LeftHemisphere` (line 199)
- `CorpusCallosum` (line 276)
- `NeuroLogosBicameralFisiologico` (line 290)
- `NeuralDiagnostics` (line 309)
- `Flickr8kDataset` (line 383)
- `LifeCycle` (line 436)

**Functions:**
- `seed_all` (line 38)
- `setup_flickr8k` (line 46)
- `build_vocab_flickr` (line 416)
- `train_bicameral_fisiologico` (line 447)
- `__init__` (line 101)
- `forward` (line 109)
- `__init__` (line 124)
- `forward` (line 138)
- `__init__` (line 165)
- `forward` (line 178)
- `__init__` (line 200)
- `forward` (line 219)
- `_get_init_state` (line 258)
- `_top_p_filtering` (line 263)
- `__init__` (line 277)
- `forward` (line 284)
- `__init__` (line 291)
- `forward` (line 297)
- `__init__` (line 310)
- `measure_callosal_flow` (line 326)
- `measure_vocab_diversity` (line 333)
- `update` (line 337)
- `get_recent_avg` (line 342)
- `report` (line 347)
- `__init__` (line 384)
- `__len__` (line 400)
- `__getitem__` (line 403)
- `__init__` (line 437)
- `get_global_loss_proxy` (line 440)

#### `bicameral2.py`
**Path:** `bicameral2.py`

**Classs:**
- `TinyVisualEncoder` (line 83)
- `MinimalLiquidNeuron` (line 104)
- `RightHemisphere` (line 126)
- `LeftHemisphere` (line 136)
- `CorpusCallosum` (line 176)
- `NeuroLogosBicameralUltra` (line 186)
- `DemocraticDiagnostics` (line 206)
- `Flickr8kDataset` (line 246)

**Functions:**
- `setup_flickr8k` (line 32)
- `build_vocab` (line 274)
- `train_ultra` (line 290)
- `__init__` (line 84)
- `forward` (line 98)
- `__init__` (line 105)
- `forward` (line 113)
- `__init__` (line 127)
- `forward` (line 131)
- `__init__` (line 137)
- `forward` (line 143)
- `__init__` (line 177)
- `forward` (line 180)
- `__init__` (line 187)
- `forward` (line 192)
- `__init__` (line 207)
- `measure_flow` (line 212)
- `vocab_diversity` (line 217)
- `update` (line 219)
- `avg` (line 223)
- `report` (line 226)
- `__init__` (line 247)
- `__len__` (line 261)
- `__getitem__` (line 262)

#### `bicameral3.py`
**Path:** `bicameral3.py`

**Classs:**
- `LiquidNeuron` (line 107)
- `RightHemisphere` (line 180)
- `LeftHemisphere` (line 201)
- `CorpusCallosum` (line 298)
- `NeuroLogosBicameral` (line 313)
- `NeuralDiagnostics` (line 334)
- `Flickr8kDataset` (line 404)
- `LifeCycle` (line 466)

**Functions:**
- `setup_flickr8k` (line 34) - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 443)
- `train_bicameral` (line 481)
- `__init__` (line 108)
- `forward` (line 125)
- `consolidate_svd` (line 154)
- `__init__` (line 181)
- `forward` (line 192)
- `__init__` (line 202)
- `forward` (line 224)
- `_get_init_state` (line 278)
- `_top_p_filtering` (line 283)
- `__init__` (line 299)
- `forward` (line 307)
- `__init__` (line 314)
- `forward` (line 320)
- `__init__` (line 335)
- `measure_callosal_flow` (line 346)
- `measure_vocab_diversity` (line 353)
- `update` (line 357)
- `get_recent_avg` (line 362)
- `report` (line 367)
- `__init__` (line 405)
- `__len__` (line 423)
- `__getitem__` (line 426)
- `__init__` (line 467)
- `get_plasticity` (line 470)

#### `bicameral_v2.py`
**Path:** `bicameral_v2.py`

**Classs:**
- `BCMPlasticity` (line 117)
- `LiquidNeuron` (line 134)
- `ResidualBlock` (line 225)
- `VisualCortex` (line 244)
- `SymbioticBasisRefinement` (line 280)
- `AdaptiveCombinatorialComplexLayer` (line 300)
- `GraphNeuralLayer` (line 311)
- `RightHemisphere` (line 342)
- `MiniUnconscious` (line 406)
- `NestedUnconscious` (line 423)
- `TopologicalCompressor` (line 466)
- `ConsciousCore` (line 486)
- `LeftHemisphere` (line 543)
- `BioDecoder` (line 558)
- `ConsciousCore` (line 667)
- `HomeostasisEngine` (line 725)
- `BicameralHomeostasis` (line 741)
- `ReplayMemory` (line 773)
- `CorpusCallosum` (line 817)
- `NeuroLogos` (line 868)
- `LifeCycle` (line 956)
- `CIFARCaptions` (line 974)

**Functions:**
- `compute_phi_effective` (line 24) - *Φₑ efectivo: integración causal simplificada para batches
activations: [B, N, D] *
- `measure_spatial_richness` (line 52) - *FIX: Métrica de riqueza dimensional efectiva con escalado positivo garantizado
Preserva interfaz exacta: shannon_entropy, richness, vn_entropy (valores POSITIVOS)
Evita saturación en 2.0 y elimina negativos usando Participation Ratio escalado*
- `top_k_top_p_filtering` (line 98) - *Filtro Top-K y Nucleus Sampling estandar*
- `create_grid_adjacency` (line 327) - *Crea matriz de adyacencia para grid cuadrado*
- `estimate_coherence` (line 1012)
- `train_logos` (line 1026)
- `__init__` (line 118)
- `forward` (line 123) - *dθ/dt = (E[activity²] - θ)/τ  →  dw/dt ∝ activity*(activity-θ)*
- `__init__` (line 135)
- `forward` (line 158)
- `consolidate_svd` (line 204) - *Mantener interfaz exacta pero implementar consolidación Hebbiana real*
- `__init__` (line 226)
- `forward` (line 238)
- `__init__` (line 245)
- `_make_layer` (line 259)
- `forward` (line 265)
- `__init__` (line 281)
- `forward` (line 291)
- `__init__` (line 301)
- `forward` (line 307)
- `__init__` (line 312)
- `forward` (line 322)
- `__init__` (line 343)
- `forward` (line 375)
- `__init__` (line 407)
- `forward` (line 420)
- `__init__` (line 424)
- `forward` (line 444)
- `__init__` (line 467)
- `forward` (line 476)
- `__init__` (line 487)
- `forward` (line 499)
- `get_liquid_module` (line 537)
- `__init__` (line 544)
- `forward` (line 550)
- `__init__` (line 559)
- `forward` (line 578)
- `_get_init_state` (line 659)
- `__init__` (line 668)
- `forward` (line 680)
- `get_liquid_module` (line 718)
- `__init__` (line 726)
- `decide` (line 730)
- `__init__` (line 742)
- `decide` (line 751)
- `__init__` (line 774)
- `store` (line 780)
- `replay` (line 791)
- `__init__` (line 818)
- `forward` (line 824)
- `__init__` (line 869)
- `forward` (line 895)
- `set_epoch` (line 950)
- `__init__` (line 957)
- `get_plasticity` (line 961)
- `__init__` (line 975)
- `__len__` (line 999)
- `__getitem__` (line 1002)

#### `bicameral_v3.py`
**Path:** `bicameral_v3.py`

**Classs:**
- `BCMPlasticity` (line 241)
- `BioDecoder` (line 257)
- `LiquidNeuron` (line 383)
- `ResidualBlock` (line 486)
- `VisualCortex` (line 505)
- `SymbioticBasisRefinement` (line 541)
- `AdaptiveCombinatorialComplexLayer` (line 561)
- `GraphNeuralLayer` (line 572)
- `RightHemisphere` (line 603)
- `MiniUnconscious` (line 667)
- `NestedUnconscious` (line 684)
- `TopologicalCompressor` (line 727)
- `ConsciousCore` (line 744)
- `LeftHemisphere` (line 846)
- `BioDecoder` (line 860)
- `CorpusCallosum` (line 991)
- `HomeostasisEngine` (line 1039)
- `BicameralHomeostasis` (line 1082)
- `ReplayMemory` (line 1138)
- `NeuroLogos` (line 1182)
- `LifeCycle` (line 1280)
- `CIFARCaptions` (line 1298)

**Functions:**
- `compute_phi_effective` (line 22) - *Φₑ con manejo robusto de dimensiones pequeñas*
- `compute_spatial_diversity` (line 75) - *Diversidad basada en correlación inversa de Pearson.
VERSIÓN ROBUSTA: Protegida contra NaNs por errores de precisión flotante.*
- `compute_activation_entropy` (line 132) - *Shannon entropy sobre la distribución de activaciones
Target range: [2.5, 4.5] bits*
- `measure_neural_complexity` (line 163) - *Medición corregida con formato [B, D, N] para neuronas reales*
- `measure_spatial_richness` (line 212) - *Wrapper para compatibilidad con código existente*
- `top_k_top_p_filtering` (line 221) - *Filtro Top-K y Nucleus Sampling estándar*
- `create_grid_adjacency` (line 588) - *Crea matriz de adyacencia para grid cuadrado*
- `estimate_coherence` (line 1336)
- `to_float` (line 1349)
- `train_logos` (line 1355)
- `__init__` (line 242)
- `forward` (line 247) - *dθ/dt = (E[activity²] - θ)/τ  →  dw/dt ∝ activity*(activity-θ)*
- `__init__` (line 258)
- `forward` (line 281)
- `_get_init_state` (line 377)
- `__init__` (line 384)
- `forward` (line 405)
- `consolidate_svd` (line 467)
- `__init__` (line 487)
- `forward` (line 499)
- `__init__` (line 506)
- `_make_layer` (line 520)
- `forward` (line 526)
- `__init__` (line 542)
- `forward` (line 552)
- `__init__` (line 562)
- `forward` (line 568)
- `__init__` (line 573)
- `forward` (line 583)
- `__init__` (line 604)
- `forward` (line 636)
- `__init__` (line 668)
- `forward` (line 681)
- `__init__` (line 685)
- `forward` (line 705)
- `__init__` (line 728)
- `forward` (line 737)
- `__init__` (line 745)
- `_create_rotation_matrix` (line 780) - *Crea matriz de rotación en espacio de alta dimensión*
- `forward` (line 790)
- `get_liquid_module` (line 839)
- `__init__` (line 847)
- `forward` (line 853)
- `__init__` (line 861)
- `forward` (line 883)
- `_get_init_state` (line 982)
- `__init__` (line 992)
- `forward` (line 999)
- `__init__` (line 1040)
- `decide` (line 1049)
- `__init__` (line 1083)
- `decide` (line 1097)
- `__init__` (line 1139)
- `store` (line 1145)
- `replay` (line 1155)
- `__init__` (line 1183)
- `forward` (line 1203)
- `set_epoch` (line 1274)
- `__init__` (line 1281)
- `get_plasticity` (line 1285)
- `__init__` (line 1299)
- `__len__` (line 1323)
- `__getitem__` (line 1326)

#### `caquita.py`
**Path:** `caquita.py`

**Classs:**
- `DiagnosticConfig` (line 31)
- `RealWorldEnvironment` (line 50)
- `LiquidNeuron` (line 78)
- `TraumaResponseSchedulerV2_ORIGINAL` (line 103) - *Versión ORIGINAL de v8.5 (con el bug)*
- `TraumaResponseSchedulerV2_FIXED` (line 168) - *Versión CORREGIDA que compara con fase anterior*
- `ChaosAdaptiveFilter_ORIGINAL` (line 228) - *Versión ORIGINAL que detecta ruido por varianza*
- `DiagnosticModel` (line 264)

**Functions:**
- `seed_everything` (line 40)
- `train_diagnostic` (line 327) - *Entrenamiento con logs detallados*
- `run_diagnostic_ablation` (line 406)
- `__init__` (line 51)
- `get_batch` (line 63)
- `__init__` (line 79)
- `forward` (line 88)
- `__init__` (line 105)
- `update_phase_performance` (line 112)
- `detect_trauma_level` (line 122)
- `generate_response` (line 148)
- `__init__` (line 170)
- `update_phase_performance` (line 176)
- `detect_trauma_level` (line 185)
- `generate_response` (line 208)
- `__init__` (line 230)
- `extract_noise_features` (line 240)
- `detect_chaos` (line 253)
- `__init__` (line 265)
- `forward` (line 285)

#### `chatgpt.py`
**Path:** `chatgpt.py`

**Classs:**
- `Config` (line 19)
- `DataEnvironment` (line 41)
- `HomeostaticRegulator` (line 70)
- `PhysioNeuron` (line 93)
- `NeuroPhysioBicameral` (line 140)
- `NeuralDiagnostics` (line 191)

**Functions:**
- `seed_all` (line 33)
- `train` (line 225)
- `__init__` (line 42)
- `get_batch` (line 53)
- `__init__` (line 71)
- `forward` (line 81)
- `__init__` (line 94)
- `forward` (line 107)
- `__init__` (line 141)
- `count_parameters` (line 163)
- `forward` (line 166)
- `__init__` (line 192)
- `update` (line 201)
- `avg` (line 208)
- `report` (line 211)

#### `cifar3.py`
**Path:** `cifar3.py`

**Classs:**
- `FastSlowLinear` (line 51) - *Linear layer con pesos lentos (backprop) y pesos rápidos (hebbianos).
Incluye decay temporal y normalización L2 estricta para estabilidad.*
- `DualSystemModule` (line 129)
- `ConsciousnessModule` (line 147) - *Módulo de conciencia con Φₑ más estable y relevante.*
- `OmniBrainFastSlow` (line 193)

**Functions:**
- `compute_phi_effective` (line 30)
- `get_cifar10_loaders` (line 245)
- `evaluate` (line 261)
- `train` (line 274)
- `__init__` (line 56)
- `reset_fast_weights` (line 74) - *Reinicia los pesos rápidos al inicio de cada batch.*
- `update_fast_weights` (line 79) - *Actualiza fast weights usando regla hebbiana con decay y normalización.*
- `forward` (line 105)
- `end_of_batch` (line 116) - *Limpia caché al final del batch para permitir reinicio en el siguiente.*
- `get_fast_norm` (line 120) - *Retorna la norma L2 de los fast weights para monitoreo homeostático.*
- `__init__` (line 130)
- `forward` (line 138)
- `__init__` (line 149)
- `compute_phi_effective_robust` (line 160) - *Φₑ más robusto usando promedio móvil y ventana temporal.*
- `forward` (line 184)
- `__init__` (line 194)
- `forward` (line 226)
- `reset_all_fast_weights` (line 233)
- `get_fast_norms` (line 238)

#### `cifar4.py`
**Path:** `cifar4.py`

**Classs:**
- `FastSlowLinear` (line 51) - *Linear layer con pesos lentos (backprop) y pesos rápidos (hebbianos).
Incluye decay temporal y normalización L2 estricta para estabilidad.*
- `DualSystemModule` (line 116)
- `ConsciousnessModule` (line 134) - *Módulo de conciencia con Φₑ más estable y relevante.*
- `OmniBrainFastSlow` (line 176)

**Functions:**
- `compute_phi_effective` (line 30)
- `get_cifar10_loaders` (line 217)
- `evaluate` (line 238)
- `train` (line 255)
- `__init__` (line 56)
- `reset_fast_weights` (line 72)
- `update_fast_weights` (line 76)
- `forward` (line 95)
- `end_of_batch` (line 106)
- `get_fast_norm` (line 109)
- `__init__` (line 117)
- `forward` (line 125)
- `__init__` (line 136)
- `compute_phi_effective_robust` (line 147)
- `forward` (line 166)
- `__init__` (line 177)
- `forward` (line 197)
- `reset_all_fast_weights` (line 204)
- `get_fast_norms` (line 209)

#### `demo_auto_regulation.py`
**Path:** `demo_auto_regulation.py`

**Classs:**
- `Config` (line 21)
- `DataEnvironment` (line 43)
- `AutoRegulationSystem` (line 77)
- `SelfModifyingGates` (line 108)
- `PhysioChimeraFixed` (line 159)

**Functions:**
- `seed_everything` (line 33)
- `demo_auto_regulation` (line 243)
- `__init__` (line 44)
- `get_batch` (line 54)
- `get_full` (line 68)
- `get_w2` (line 71)
- `__init__` (line 78)
- `update` (line 84)
- `get_stability` (line 99)
- `__init__` (line 109)
- `forward` (line 126)
- `__init__` (line 160)
- `forward` (line 184)

#### `difract.py`
**Path:** `difract.py`

**Functions:**
- `visualize_uased_geometry` (line 4)

#### `dmg_core.py`
**Path:** `dmg_core.py`

**Classs:**
- `AdaptiveMagnitudeGate` (line 14) - *Bio-inspired gating mechanism.
Acts as a learnable filter that suppresses signals exceeding a dynamic threshold.

Formula: Gate = Sigmoid( Gain * (Threshold - |x|^p * Sensitivity) )*
- `SparseTopologyLayer` (line 44) - *Linear layer with sparse connectivity enforcement derived from 
Scale-Free (Barabási-Albert) graphs.*
- `DMGNetwork` (line 82) - *Robust Neural Network architecture using Sparse Layers and Dynamic Gating.
Designed for high noise resistance (MNIST-C / Adversarial robustness).*

**Functions:**
- `__init__` (line 21)
- `forward` (line 30)
- `__init__` (line 49)
- `_generate_sparse_mask` (line 62) - *Generates a Barabási-Albert scale-free mask.*
- `forward` (line 77)
- `__init__` (line 87)
- `forward` (line 100)

#### `dualmind.py`
**Path:** `dualmind.py`

**Classs:**
- `HomeostasisEngine` (line 30)
- `LiquidNeuron` (line 57) - *Neurona con fast weights hebbianos (de Síntesis)*
- `ConsciousSystem` (line 108) - *Sistema de control ejecutivo que opera sobre representaciones
del sistema inconsciente. Implementa homeostasis y memoria de trabajo.*
- `NestedTopoLayer` (line 171) - *Capa de procesamiento topológico con memoria episódica.
Versión simplificada de TopoBrain enfocada en representaciones ricas.*
- `UnconsciousSystem` (line 217) - *Sistema inconsciente: Procesamiento automático y paralelo.
Arquitectura simplificada de TopoBrain para extracción de features.*
- `DualMind` (line 279) - *Sistema dual de procesamiento:
- Inconsciente: Procesamiento automático, paralelo, topológico
- Consciente: Decisión deliberada, homeostática, serial*

**Functions:**
- `measure_spatial_richness` (line 15) - *Mide diversidad de representaciones mediante eigenspectro*
- `train_dualmind_phase1` (line 346) - *FASE 1: Preentrenamiento del sistema inconsciente
Objetivo: Aprender representaciones topológicas ricas*
- `train_dualmind_phase2` (line 401) - *FASE 2: Entrenamiento del sistema consciente
Objetivo: Aprender decisiones homeostáticas óptimas
Sistema inconsciente CONGELADO*
- `train_dualmind_phase3` (line 487) - *FASE 3: Co-adaptación de ambos sistemas
Objetivo: Refinamiento conjunto con retroalimentación*
- `evaluate_dualmind` (line 579) - *Evaluación del sistema dual*
- `run_dualmind_experiment` (line 604)
- `__init__` (line 31)
- `decide` (line 35) - *Motor de decisión homeostática con targets realistas y pesos equilibrados.
- target_entropy=1.8: Valor alcanzable dentro del rango [0, log(10)=2.3]
- target_richness=85.0: Por encima del estado inicial (66-74) para activar exploración
- Pesos reducidos para evitar dominancia de un solo drive*
- `__init__` (line 59)
- `forward` (line 67) - *Neurona con plasticidad hebbiana de fast weights y decaimiento activación.
Incluye estabilización mediante decaimiento temporal de W_fast.*
- `consolidate_svd` (line 89) - *Consolidación mediante SVD (modo sueño)*
- `__init__` (line 113)
- `forward` (line 136) - *Input: Representaciones del sistema inconsciente [batch, unconscious_dim]
Output: logits, métricas homeostáticas*
- `get_structure_entropy` (line 157) - *Análisis de salud estructural mediante SVD*
- `__init__` (line 176)
- `forward` (line 188) - *x_nodes: [batch, num_nodes, in_dim]
output: [batch, num_nodes, hid_dim]*
- `get_topology_density` (line 209) - *Densidad de conexiones topológicas*
- `__init__` (line 222)
- `forward` (line 246) - *x: [batch, 3, 32, 32]
output: [batch, output_dim] representaciones inconscientes*
- `get_topology_stats` (line 264) - *Estadísticas de topología del sistema inconsciente*
- `__init__` (line 285)
- `forward` (line 305) - *Modos de operación:
- 'unconscious': Solo sistema inconsciente (rápido, baseline)
- 'conscious': Consciente sobre inconsciente (lento, preciso)
- 'dual': Ambos con retroalimentación (modo completo)*
- `get_system_status` (line 331) - *Diagnóstico completo del sistema dual*

#### `dynamic.py`
**Path:** `dynamic.py`

**Classs:**
- `MasterConfig` (line 24)
- `DataEnvironment` (line 36)
- `OmnibusController` (line 54)
- `SovereignAttention` (line 88)
- `LiquidNeuron` (line 100)
- `SovereignChimera` (line 137)

**Functions:**
- `seed_everything` (line 13)
- `run_final_showdown` (line 184)
- `__init__` (line 37)
- `get_batch` (line 46)
- `__init__` (line 55)
- `forward` (line 66)
- `__init__` (line 89)
- `forward` (line 94)
- `__init__` (line 101)
- `forward` (line 113)
- `__init__` (line 138)
- `forward` (line 151)

#### `dynamic2.py`
**Path:** `dynamic2.py`

**Classs:**
- `PhysioConfig` (line 24)
- `DataEnvironment` (line 36)
- `HomeostaticRegulator` (line 54)
- `PhysioNeuron` (line 89)
- `PhysioChimera` (line 162)

**Functions:**
- `seed_everything` (line 13)
- `run_physio_experiment` (line 178)
- `__init__` (line 37)
- `get_batch` (line 46)
- `__init__` (line 55)
- `forward` (line 66)
- `__init__` (line 90)
- `forward` (line 105)
- `__init__` (line 163)
- `forward` (line 169)

#### `example_usage.py`
**Path:** `example_usage.py`

**Functions:**
- `demo_simple_monitoring` (line 20) - *Demostración de monitoreo básico*
- `demo_custom_monitoring` (line 38) - *Demostración de monitoreo personalizado*
- `demo_checkpoint_system` (line 85) - *Demostración del sistema de checkpointing*
- `demo_comparison_experiments` (line 142) - *Demostración de comparación entre experimentos*
- `create_demo_report` (line 194) - *Crear reporte demo completo*
- `main` (line 333) - *Función principal de demostración*

#### `exampleww.py`
**Path:** `exampleww.py`

**Functions:**
- `run_single_experiment` (line 8)

#### `exodia_op_2.py`
**Path:** `exodia_op_2.py`

**Classs:**
- `HierarchicalEpisodicMemory` (line 340) - *Memoria episódica optimizada con estabilización numérica en sampling
- Fixed: Clamp de surprise scores para evitar probabilidades degeneradas
- Fixed: Verificación explícita de NaN en operaciones de buffer*
- `NeurocognitiveSystem` (line 577)
- `LanguageMetrics` (line 774) - *Métricas de calidad de generación*
- `LinguisticFeedbackLoop` (line 848)
- `LanguageMetrics` (line 965)
- `CausalReasoningEngine` (line 1008)
- `LanguageMetrics` (line 1087)
- `StableLiquidNeuron` (line 1134)
- `TricameralOutput` (line 1321)
- `TriangulatedMedicalSystem` (line 1360)
- `LeftHemisphere` (line 1510)
- `AudioEncoder` (line 1833) - *Encoder de audio optimizado con:
- Pruning estructurado en canales Conv (30% reducción)
- Gradient checkpointing para memoria de activaciones
- Preparación para QAT INT8*
- `RightHemisphereTricameral` (line 1901) - *Hemisferio derecho optimizado:
- Gradient checkpointing obligatorio en ResNet50
- AudioEncoder con canales reducidos (90-180-360)
- Memoria activaciones reducida en 60%*
- `CorpusCallosumTrimodal` (line 1984) - *Corpus Callosum optimizado con:
- Dimensión base reducida: 512→320 dims (-37.5%)
- Bottleneck compartido para 3 canales (1 Linear vs 3 ModuleList)
- Flash Attention / xFormers compatible
- Gates fusionados en tensor único
Reducción: 2.1M → 0.88M parámetros (-58%)*
- `EnhancedDiagnosticsTricameral` (line 2207)
- `NeuroLogosTricameral` (line 2515) - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 2550) - *Dataset que carga imagen, audio del caption y texto desde Kaggle*

**Functions:**
- `preprocess_and_cache_spectrograms` (line 49) - *Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt
Esto elimina el cuello de botella de I/O durante entrenamiento*
- `apply_emergency_fixes` (line 120)
- `setup_flickr8k_with_audio` (line 142) - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 316) - *Construye vocabulario desde el archivo de captions*
- `forward` (line 1335)
- `compute_alignment_loss` (line 2666) - *FIX: Pérdida auxiliar para alineación temprana de canales multimodales
Solo activa en épocas iniciales (epoch < 6)*
- `compute_tricameral_loss` (line 2695)
- `train_tricameral` (line 2812)
- `__init__` (line 347)
- `compute_surprise` (line 371) - *FIX: Clamp de cross-entropy para evitar infinitos*
- `calculate_importance` (line 386) - *FIX: Clamp de surprise_score para evitar probabilidades degeneradas*
- `_calculate_novelty` (line 402) - *FIX: Manejo de edge case cuando no hay memorias*
- `store_episode` (line 427)
- `_update_unified_buffer` (line 456) - *FIX: Verificar integridad de scores antes de unificar*
- `sample` (line 470) - *FIX: Manejo de edge cases en sampling probabilístico*
- `_sample_from_buffer` (line 494) - *FIX: Estabilización completa de probabilidades de sampling*
- `apply_forgetting_curve` (line 535)
- `_purge_low_score_memories` (line 545) - *FIX: Purga con threshold ajustado y verificación de scores*
- `__init__` (line 578)
- `assess_reasoning_state` (line 598) - *Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)*
- `assess_cognitive_state` (line 642) - *Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)*
- `apply_cognitive_intervention` (line 688) - *Aplica intervenciones basadas en estado lingüístico y de razonamiento*
- `sentence_bleu` (line 778) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 812) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 821) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 834) - *Jaccard similarity entre palabras*
- `__init__` (line 849)
- `_get_ngrams_cached` (line 863) - *FIX: Método estático con lru_cache para n-gramas*
- `compute_linguistic_reward` (line 872)
- `compute_cider` (line 911) - *FIX: Uso correcto del cache estático*
- `compute_spice` (line 925)
- `get_cache_stats` (line 937) - *FIX: Estadísticas de cache actualizadas*
- `sentence_bleu` (line 967)
- `token_accuracy` (line 990)
- `word_overlap` (line 1000)
- `__init__` (line 1009)
- `reason_causally` (line 1036)
- `_predict_interventions` (line 1050)
- `update_knowledge_graph` (line 1067)
- `query_causal_chain` (line 1073)
- `sentence_bleu` (line 1089)
- `token_accuracy` (line 1112)
- `word_overlap` (line 1122)
- `__init__` (line 1136)
- `forward` (line 1184)
- `_calculate_homeostasis_metric` (line 1219) - *Calcula métrica de homeostasis con estabilización numérica*
- `hebbian_update` (line 1229)
- `update_physiology_advanced` (line 1278)
- `__init__` (line 1361)
- `triangulate_signals` (line 1368)
- `count_convergent_signals` (line 1379)
- `diagnose_with_triangulation` (line 1382)
- `apply_triangulated_intervention` (line 1427)
- `_reset_liquid_neuron` (line 1496)
- `__init__` (line 1511)
- `forward` (line 1596)
- `_apply_chain_of_thought` (line 1653)
- `_apply_multi_token_prediction` (line 1694)
- `_apply_structural_attention` (line 1737)
- `_greedy_decode` (line 1759)
- `_get_init_state` (line 1819)
- `__init__` (line 1842)
- `forward` (line 1880)
- `__init__` (line 1909)
- `forward` (line 1947)
- `__init__` (line 1994)
- `_apply_flash_attention` (line 2057) - *Aplica Flash Attention nativa de PyTorch 2.0+
FIX: Corrección de dimensiones para seq_len variable*
- `forward` (line 2090) - *FIX: Manejo robusto de dimensiones y verificación de coherencia trimodal*
- `update_channel_fatigue` (line 2169)
- `adjust_gates_by_fatigue` (line 2190) - *Lógica original de ajuste de gates*
- `__init__` (line 2208)
- `_get_cached_norm` (line 2231) - *Cache de normalización con limpieza periódica*
- `measure_callosal_flow` (line 2248) - *Medición de coherencia multimodal con sincronización entre canales*
- `evaluate_reasoning_quality` (line 2305)
- `calculate_synergy` (line 2342)
- `calculate_health` (line 2353)
- `update` (line 2362)
- `get_recent_avg` (line 2379)
- `visualize_fatigue_distribution` (line 2395)
- `visualize_reasoning_metrics` (line 2417)
- `report` (line 2429)
- `__init__` (line 2518)
- `forward` (line 2525)
- `__init__` (line 2553)
- `__len__` (line 2610)
- `__getitem__` (line 2613)

#### `exodia_optimized.py`
**Path:** `exodia_optimized.py`

**Classs:**
- `HierarchicalEpisodicMemory` (line 332) - *Memoria episódica optimizada para Colab:
- working_capacity: 200→80 (-60%)
- short_term_capacity: 800→320 (-60%)
- long_term eliminado completamente
- Reducción overhead: 70% (de 2.4GB a 0.7GB)*
- `NeurocognitiveSystem` (line 546)
- `LanguageMetrics` (line 743) - *Métricas de calidad de generación*
- `LinguisticFeedbackLoop` (line 817)
- `LanguageMetrics` (line 934)
- `CausalReasoningEngine` (line 977)
- `LanguageMetrics` (line 1056)
- `StableLiquidNeuron` (line 1103)
- `TriangulatedMedicalSystem` (line 1243)
- `LeftHemisphere` (line 1394)
- `AudioEncoder` (line 1703) - *Encoder de audio optimizado con:
- Pruning estructurado en canales Conv (30% reducción)
- Gradient checkpointing para memoria de activaciones
- Preparación para QAT INT8*
- `RightHemisphereTricameral` (line 1778) - *Hemisferio derecho optimizado:
- Gradient checkpointing obligatorio en ResNet50
- AudioEncoder con canales reducidos (90-180-360)
- Memoria activaciones reducida en 60%*
- `CorpusCallosumTrimodal` (line 1861) - *Corpus Callosum optimizado con:
- Dimensión base reducida: 512→320 dims (-37.5%)
- Bottleneck compartido para 3 canales (1 Linear vs 3 ModuleList)
- Flash Attention / xFormers compatible
- Gates fusionados en tensor único
Reducción: 2.1M → 0.88M parámetros (-58%)*
- `EnhancedDiagnosticsTricameral` (line 2093)
- `NeuroLogosTricameral` (line 2398) - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 2433) - *Dataset que carga imagen, audio del caption y texto desde Kaggle*

**Functions:**
- `preprocess_and_cache_spectrograms` (line 47) - *Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt
Esto elimina el cuello de botella de I/O durante entrenamiento*
- `setup_flickr8k_with_audio` (line 120) - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 308) - *Construye vocabulario desde el archivo de captions*
- `compute_alignment_loss` (line 2549) - *FIX: Pérdida auxiliar para alineación temprana de canales multimodales
Solo activa en épocas iniciales (epoch < 6)*
- `compute_tricameral_loss` (line 2577) - *FIX: Pérdida con término explícito de coherencia multimodal
Penaliza la falta de sincronización entre canales*
- `train_tricameral` (line 2650)
- `__init__` (line 341)
- `compute_surprise` (line 369) - *Sin cambios en la lógica*
- `calculate_importance` (line 380) - *Sin cambios*
- `_calculate_novelty` (line 393) - *Usa solo working/short_term (no long_term)*
- `store_episode` (line 416)
- `_update_unified_buffer` (line 445) - *Solo working + short_term*
- `add` (line 450) - *Alias para store_episode*
- `apply_forgetting_curve` (line 454) - *Decay más agresivo (ahorro overhead)*
- `_purge_low_score_memories` (line 467) - *Purga más agresiva (threshold mayor)*
- `sample` (line 487) - *Muestreo solo de working/short_term*
- `_sample_from_buffer` (line 511) - *Lógica original sin cambios*
- `get_total_size` (line 539) - *Solo working + short_term*
- `__init__` (line 547)
- `assess_reasoning_state` (line 567) - *Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)*
- `assess_cognitive_state` (line 611) - *Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)*
- `apply_cognitive_intervention` (line 657) - *Aplica intervenciones basadas en estado lingüístico y de razonamiento*
- `sentence_bleu` (line 747) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 781) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 790) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 803) - *Jaccard similarity entre palabras*
- `__init__` (line 818)
- `_get_ngrams_cached` (line 832) - *FIX: Método estático con lru_cache para n-gramas*
- `compute_linguistic_reward` (line 841)
- `compute_cider` (line 880) - *FIX: Uso correcto del cache estático*
- `compute_spice` (line 894)
- `get_cache_stats` (line 906) - *FIX: Estadísticas de cache actualizadas*
- `sentence_bleu` (line 936)
- `token_accuracy` (line 959)
- `word_overlap` (line 969)
- `__init__` (line 978)
- `reason_causally` (line 1005)
- `_predict_interventions` (line 1019)
- `update_knowledge_graph` (line 1036)
- `query_causal_chain` (line 1042)
- `sentence_bleu` (line 1058)
- `token_accuracy` (line 1081)
- `word_overlap` (line 1091)
- `__init__` (line 1104)
- `forward` (line 1146)
- `_calculate_homeostasis_metric` (line 1163) - *Calcula métrica de homeostasis basada en la estabilidad del output*
- `hebbian_update` (line 1172)
- `update_physiology_advanced` (line 1210)
- `__init__` (line 1244)
- `triangulate_signals` (line 1251)
- `count_convergent_signals` (line 1262)
- `diagnose_with_triangulation` (line 1265)
- `apply_triangulated_intervention` (line 1310)
- `_reset_liquid_neuron` (line 1379) - *Reset completo de una neurona líquida*
- `__init__` (line 1395)
- `forward` (line 1477)
- `_apply_chain_of_thought` (line 1524)
- `_greedy_decode` (line 1564)
- `_apply_multi_token_prediction` (line 1625)
- `_apply_structural_attention` (line 1667)
- `_get_init_state` (line 1688)
- `__init__` (line 1711)
- `forward` (line 1751)
- `__init__` (line 1786)
- `forward` (line 1824)
- `__init__` (line 1871)
- `_apply_flash_attention` (line 1940) - *Aplica Flash Attention nativa de PyTorch 2.0+*
- `forward` (line 1966)
- `update_channel_fatigue` (line 2054) - *Lógica original de fatiga sin cambios*
- `adjust_gates_by_fatigue` (line 2076) - *Lógica original de ajuste de gates*
- `__init__` (line 2094)
- `_get_cached_norm` (line 2117) - *Cache de normalización con limpieza periódica*
- `measure_callosal_flow` (line 2135) - *FIX: Medición de coherencia multimodal real con atención a diversidad
Incluye métricas de sincronización entre canales*
- `__init__` (line 2401)
- `forward` (line 2407)
- `__init__` (line 2436)
- `__len__` (line 2493)
- `__getitem__` (line 2496)
- `evaluate_reasoning_quality` (line 2188)
- `calculate_synergy` (line 2225)
- `calculate_health` (line 2236)
- `update` (line 2245)
- `get_recent_avg` (line 2262)
- `visualize_fatigue_distribution` (line 2278)
- `visualize_reasoning_metrics` (line 2302)
- `report` (line 2314)

#### `final_sinergy_analysis.py`
**Path:** `final_sinergy_analysis.py`

**Classs:**
- `SinergyAnalysis` (line 12)

**Functions:**
- `main` (line 219)
- `__init__` (line 13)
- `print_header` (line 80)
- `analyze_original_models` (line 87)
- `analyze_sinergies` (line 99)
- `generate_scientific_matrix` (line 118)
- `calculate_synergy_breakthrough` (line 136)
- `generate_conclusion` (line 171)
- `save_results` (line 200)

#### `gemini.py`
**Path:** `gemini.py`

**Classs:**
- `LiquidNeuron` (line 108)
- `RightHemisphere` (line 177)
- `CorpusCallosum` (line 196)
- `LeftHemisphere` (line 219)
- `NeuroLogosBicameral` (line 333)
- `NeuralDiagnostics` (line 361)
- `Flickr8kDataset` (line 431)
- `LifeCycle` (line 493)

**Functions:**
- `setup_flickr8k` (line 40) - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 470)
- `train_bicameral` (line 509)
- `__init__` (line 109)
- `forward` (line 123)
- `consolidate_svd` (line 154)
- `__init__` (line 178)
- `forward` (line 187)
- `__init__` (line 197)
- `forward` (line 210)
- `__init__` (line 220)
- `forward` (line 241)
- `_get_init_state` (line 313)
- `_top_p_filtering` (line 318)
- `__init__` (line 334)
- `forward` (line 340)
- `__init__` (line 362)
- `measure_callosal_flow` (line 373)
- `measure_vocab_diversity` (line 380)
- `update` (line 384)
- `get_recent_avg` (line 389)
- `report` (line 394)
- `__init__` (line 432)
- `__len__` (line 450)
- `__getitem__` (line 453)
- `__init__` (line 494)
- `get_plasticity` (line 497)

#### `gemini2.py`
**Path:** `gemini2.py`

**Classs:**
- `LiquidNeuron` (line 108)
- `RightHemisphere` (line 177)
- `CorpusCallosum` (line 196)
- `LeftHemisphere` (line 219)
- `NeuroLogosBicameral` (line 333)
- `NeuralDiagnostics` (line 361)
- `Flickr8kDataset` (line 431)
- `LifeCycle` (line 493)

**Functions:**
- `setup_flickr8k` (line 40) - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 470)
- `train_bicameral` (line 509)
- `__init__` (line 109)
- `forward` (line 123)
- `consolidate_svd` (line 154)
- `__init__` (line 178)
- `forward` (line 187)
- `__init__` (line 197)
- `forward` (line 210)
- `__init__` (line 220)
- `forward` (line 241)
- `_get_init_state` (line 313)
- `_top_p_filtering` (line 318)
- `__init__` (line 334)
- `forward` (line 340)
- `__init__` (line 362)
- `measure_callosal_flow` (line 373)
- `measure_vocab_diversity` (line 380)
- `update` (line 384)
- `get_recent_avg` (line 389)
- `report` (line 394)
- `__init__` (line 432)
- `__len__` (line 450)
- `__getitem__` (line 453)
- `__init__` (line 494)
- `get_plasticity` (line 497)

#### `gen_dataset.py`
**Path:** `gen_dataset.py`

**Functions:**
- `generate_one` (line 89)
- `main` (line 110)

#### `get_dataset.py`
**Path:** `get_dataset.py`

**Functions:**
- `download_captions_only` (line 29) - *Descarga solo los captions de Flickr8k*
- `generate_one_audio` (line 67) - *Genera un audio con retry y rate limiting*
- `load_checkpoint` (line 114) - *Carga el checkpoint de progreso*
- `save_checkpoint` (line 122) - *Guarda el checkpoint de progreso*
- `generate_audios_with_checkpoints` (line 128) - *Genera audios con checkpoints cada 500 archivos*
- `generate_audios_sync` (line 219) - *Wrapper síncrono con manejo de event loop*
- `compress_audios_only` (line 249) - *Comprime solo los audios en zips pequeños*
- `create_audio_readme` (line 318) - *Crea README para el dataset de audios*
- `upload_to_huggingface` (line 388) - *Sube solo audios a Hugging Face*
- `main` (line 465)
- `download_flickr8k` (line 540) - *Descarga Flickr8k (solo necesitas ejecutar esto una vez)*
- `generate_audios` (line 602) - *Genera audios con Edge-TTS - TOMA TIEMPO (~20-30 min)*
- `create_split_zips` (line 637) - *Crea múltiples zips pequeños para cumplir límites de GitHub*
- `generate_upload_instructions` (line 717) - *Genera instrucciones para subir a GitHub*
- `upload_to_huggingface` (line 834) - *Sube directamente a Hugging Face (alternativa a GitHub)*
- `main` (line 888)

#### `homeostatichope.py`
**Path:** `homeostatichope.py`

**Classs:**
- `Config` (line 15)
- `RealWorldEnvironment` (line 49)
- `OmniscientRegulator` (line 96) - *Motor homeostático con acceso TOTAL a señales internas críticas
y control DIRECTO de hiperparámetros en tiempo real*
- `AdaptiveLiquidMemory` (line 190) - *Memoria líquida que responde a controles homeostáticos*
- `HomeostaticSelfModMemory` (line 233)
- `ContinuumMemorySystem` (line 278)
- `OmniscientHopeModel` (line 304)
- `ConsciousTrainer` (line 402)

**Functions:**
- `setup_device` (line 35)
- `set_seed` (line 40)
- `pgd_attack` (line 368)
- `run_conscious_experiment` (line 516)
- `run_ablation` (line 610)
- `__init__` (line 50)
- `get_batch` (line 73)
- `get_test_loader` (line 88)
- `__init__` (line 102)
- `forward` (line 128) - *Args:
    signals: Diccionario con señales internas del sistema
Returns:
    controls: Diccionario con hiperparámetros ajustados*
- `__init__` (line 193)
- `forward` (line 203)
- `__init__` (line 234)
- `forward` (line 251)
- `__init__` (line 279)
- `forward` (line 293)
- `__init__` (line 305)
- `forward` (line 341)
- `__init__` (line 403)
- `train_step` (line 425)
- `evaluate` (line 488)

#### `hope.py`
**Path:** `hope.py`

**Classs:**
- `Config` (line 16) - *Configuración para desafío realista con adversarial*
- `RealWorldEnvironment` (line 58) - *Dataset Digits con separación en "mundos" para simular concept drift
Similar al segundo ejemplo pero adaptado para classification*
- `HomeostaticRegulator` (line 127) - *Motor fisiológico que regula según estado interno*
- `LiquidMemory` (line 172) - *Memoria líquida con componente rápida y lenta*
- `EfficientSelfModMemory` (line 207) - *Self-modifying memory con homeostasis*
- `ContinuumMemorySystem` (line 287) - *CMS que acepta global_step correctamente*
- `HopePhysioModel` (line 320) - *Hope + PhysioChimera para clasificación*
- `AdversarialTrainer` (line 441)

**Functions:**
- `setup_device` (line 41)
- `set_seed` (line 49)
- `pgd_attack` (line 394) - *PGD adversarial attack - versión robusta*
- `run_real_world_experiment` (line 517)
- `run_ablation` (line 614)
- `__init__` (line 64)
- `get_batch` (line 98) - *Obtener batch según la fase de entrenamiento*
- `get_test_loader` (line 118) - *Test loader completo*
- `__init__` (line 130)
- `forward` (line 140) - *Calcula controles homeostáticos basados en:
- Estrés (varianza input)
- Excitación (magnitud activación)
- Fatiga (norma pesos)*
- `__init__` (line 175)
- `forward` (line 186) - *Args:
    x: (B, D)
    physio: Controles homeostáticos*
- `__init__` (line 210)
- `forward` (line 236) - *Args:
    x: (B, D)
Returns:
    output, h_current*
- `__init__` (line 290)
- `forward` (line 304) - *Args:
    x: (B, D)
    global_step: Paso global*
- `__init__` (line 323)
- `reset_states` (line 361)
- `forward` (line 365) - *Args:
    x: (B, n_features)
    global_step: Paso global
Returns:
    logits: (B, n_classes)*
- `__init__` (line 442)
- `train_step` (line 460) - *Un paso de entrenamiento con adversarial opcional*
- `evaluate` (line 490) - *Evaluación con ataque opcional*

#### `kimi.py`
**Path:** `kimi.py`

**Classs:**
- `PhysioState` (line 28)
- `SNE` (line 65)
- `BCMRegulated` (line 97)
- `LiquidRegulated` (line 118)
- `VisualCortexRegulated` (line 145)
- `MicroTopoBrainSNA` (line 166)
- `Config` (line 185)

**Functions:**
- `pgd_attack` (line 39) - *PGD-10 ataque con gradiente corregido para CPU*
- `get_loader` (line 191)
- `run_experiment` (line 201) - *Ejecuta un experimento completo con una seed*
- `scientific_ablation` (line 256) - *Ejecuta el estudio científico completo*
- `__init__` (line 66)
- `forward` (line 75)
- `__init__` (line 98)
- `forward` (line 104)
- `__init__` (line 119)
- `forward` (line 127)
- `__init__` (line 146)
- `forward` (line 155)
- `__init__` (line 167)
- `forward` (line 175)

#### `legendario.py`
**Path:** `legendario.py`

**Classs:**
- `MotorHomeostaticContext` (line 106) - *Contexto estable para motores homeostáticos*
- `OmniBrainModule` (line 117) - *Módulo base estable para CPU*
- `PTSymmetricLayer` (line 132) - *Capa PT-simétrica sin operaciones complejas problemáticas*
- `TopologicalLayer` (line 166) - *Capa topológica estable sin dependencias problemáticas*
- `DualMindModule` (line 193) - *Módulo dual estable para CPU*
- `ConsciousnessModule` (line 224) - *Módulo de conciencia estable*
- `OmniBrainCoordinator` (line 245) - *Coordinador sin mediciones problemáticas*
- `OmniBrain` (line 289) - *¡El Pokémon legendario estable en CPU!*

**Functions:**
- `compute_phi_effective_approx` (line 27) - *Cálculo ESTABLE de Φₑ usando PCA (proporción de varianza explicada)
¡Sin errores de dimensiones! Basado en: "Practical measures of integrated information"*
- `compute_topological_metrics` (line 60) - *Cálculo ESTABLE de métricas topológicas (optimizado para CPU)*
- `estimate_energy_consumption` (line 89) - *Estimación conservadora de consumo energético para CPU*
- `train_omni_brain` (line 340) - *Entrenamiento estable y rápido en CPU*
- `__init__` (line 119)
- `update_performance` (line 125)
- `__init__` (line 135)
- `compute_pt_phase` (line 144) - *Cálculo estable de fase PT sin números complejos*
- `forward` (line 152)
- `__init__` (line 169)
- `update_topology` (line 176) - *Actualizar máscara topológica basada en conectividad deseada*
- `forward` (line 183)
- `__init__` (line 196)
- `forward` (line 211)
- `__init__` (line 227)
- `forward` (line 232)
- `__init__` (line 248)
- `measure_network_state` (line 251) - *Mediciones ESTABLES para CPU*
- `__init__` (line 292)
- `forward` (line 317)

#### `legendario2.py`
**Path:** `legendario2.py`

**Classs:**
- `MotorHomeostaticContext` (line 34) - *Contexto para un motor homeostático*
- `PTSymmetricMotor` (line 64) - *Motor para controlar parámetros PT-similares*
- `TopologicalMotor` (line 100) - *Motor para controlar conectividad y topología*
- `EnergyHomeostaticMotor` (line 127) - *Motor para controlar eficiencia energética*
- `ConsciousnessMotor` (line 157) - *Motor para controlar métricas de conciencia (Φₑ)*
- `DualSystemMotor` (line 185) - *Motor para controlar balance inconsciente/consciente*
- `AdaptiveLearningMotor` (line 213) - *Motor para adaptar algoritmos de aprendizaje*
- `ModularActivationMotor` (line 241) - *Motor para activar/desactivar módulos según contexto*
- `OmniBrainCoordinator` (line 291) - *Coordinador central que gestiona todos los motores homeostáticos*
- `OmniBrainModule` (line 452) - *Módulo base para todos los componentes del Omni Brain*
- `PTSymmetricLayer` (line 467) - *Capa con activación PT-simétrica regulada*
- `TopologicalLayer` (line 500) - *Capa con conectividad topológica regulada*
- `DualMindModule` (line 557) - *Módulo de procesamiento dual (inconsciente/consciente)*
- `ConsciousnessModule` (line 640) - *Módulo de métricas de conciencia y integración*
- `HomeostaticEngine` (line 701) - *Motor homeostasis reutilizable de Síntesis v8.2*
- `OmniBrain` (line 732) - *El pokemon legendario que combina todas las ideas*

**Functions:**
- `train_omni_brain` (line 967) - *Pipeline de entrenamiento para el Omni Brain*
- `update` (line 47) - *Actualiza el estado del motor homeostático*
- `__init__` (line 66)
- `regulate_parameters` (line 78) - *Regula parámetros para mantener PT-simetría*
- `__init__` (line 102)
- `regulate_connectivity` (line 112) - *Regula conectividad para mantener estructura óptima*
- `__init__` (line 129)
- `regulate_energy` (line 139) - *Regula parámetros para eficiencia energética*
- `__init__` (line 159)
- `regulate_consciousness` (line 168) - *Regula parámetros para control de conciencia*
- `__init__` (line 187)
- `regulate_dual_systems` (line 197) - *Regula balance entre sistemas inconsciente y consciente*
- `__init__` (line 215)
- `regulate_learning` (line 224) - *Regula parámetros de aprendizaje*
- `__init__` (line 243)
- `regulate_modules` (line 259) - *Regula qué módulos están activos*
- `__init__` (line 294)
- `_initialize_motors` (line 300) - *Inicializa todos los motores homeostáticos*
- `sense_environment` (line 312) - *Sensa el estado actual del entorno*
- `simulate_network_state` (line 326) - *Simula el estado de red sin hacer forward pass (evita conflictos de autograd)*
- `measure_network_state` (line 340) - *Mide el estado actual de la red*
- `coordinate_all_motors` (line 377) - *Coordina todos los motores homeostáticos*
- `__init__` (line 455)
- `forward` (line 461)
- `update_performance` (line 464)
- `__init__` (line 470)
- `forward` (line 477)
- `__init__` (line 503)
- `_generate_topology_mask` (line 518) - *Genera máscara topológica realista*
- `forward` (line 539)
- `__init__` (line 560)
- `forward` (line 585)
- `__init__` (line 643)
- `compute_phi_effective` (line 657) - *Cálculo simplificado de Φₑ (integración efectiva)*
- `forward` (line 677)
- `__init__` (line 704)
- `regulate_homeostasis` (line 709) - *Regula parámetros para homeostasis*
- `__init__` (line 735)
- `reset_internal_states` (line 769) - *Resetea todos los estados internos para evitar problemas de gradientes*
- `prepare_for_inference` (line 803) - *Preparación específica para inferencia - reseteo completo*
- `initialize_context` (line 820) - *Inicializa el contexto del Omni Brain*
- `forward` (line 834) - *Forward pass del Omni Brain con coordinación homeostática*
- `get_status_report` (line 932) - *Genera reporte de estado del Omni Brain*

#### `live_cl.py`
**Path:** `live_cl.py`

**Classs:**
- `Config` (line 32) - *Centralized configuration with ablation flags*
- `FastSlowLinear` (line 133) - *Dual-system linear layer with fast (Hebbian) and slow (gradient) learning.
Fast learning is disabled in baseline mode but structure is maintained.*
- `DualSystemModule` (line 206) - *Dual-pathway processing with fast and slow streams.
Bypassed in baseline mode but ready for activation.*
- `IntegrationModule` (line 242) - *Neural integration module with adaptive gating.
Bypassed in baseline mode.*
- `OmniBrain` (line 278) - *Unified Omni Brain architecture with configurable modules.
Optimized baseline with experimental features ready for activation.*

**Functions:**
- `setup_logging` (line 69) - *Professional logging configuration*
- `set_seed` (line 84) - *Ensure reproducibility*
- `compute_integration_index` (line 99) - *Compute neural integration using SVD (Singular Value Decomposition)
Returns value in [0, 1] representing degree of neural coordination*
- `get_data_loaders` (line 349) - *Prepare CIFAR-10 data loaders with augmentation*
- `evaluate` (line 393) - *Comprehensive model evaluation*
- `train` (line 430) - *Main training loop with comprehensive logging

Args:
    config: Configuration object
    silent: If True, reduce logging for ablation studies

Returns:
    Dictionary of training metrics history*
- `run_ablation_study` (line 570) - *Comprehensive ablation study across different configurations

Args:
    quick_test: If True, run 5 epochs per config; else full 50 epochs

Returns:
    Dictionary mapping configuration names to final test accuracies*
- `to_dict` (line 62)
- `__init__` (line 138)
- `reset_fast_weights` (line 156) - *Reset fast weights (memory purge)*
- `update_fast_weights` (line 162) - *Hebbian learning update*
- `forward` (line 189)
- `get_fast_norm` (line 202)
- `__init__` (line 211)
- `forward` (line 223)
- `__init__` (line 247)
- `forward` (line 260)
- `__init__` (line 283)
- `forward` (line 318)
- `reset_all_fast_weights` (line 325) - *Reset all fast weights in the network*
- `get_fast_norms` (line 331) - *Collect fast weight norms for monitoring*
- `get_ablation_state` (line 336) - *Return current ablation configuration*

#### `live_go.py`
**Path:** `live_go.py`

**Classs:**
- `Config` (line 30)
- `FastSlowLinear` (line 93) - *La neurona perfecta. Capaz de aprender rápido (Hebbiano) y lento (Gradiente).
En este PoC, la parte rápida duerme, pero la estructura es sólida.*
- `DualSystemModule` (line 143)
- `IntegrationModule` (line 167)
- `OmniBrainGenesis` (line 190)

**Functions:**
- `compute_integration_index` (line 74) - *Calcula el orden dentro del caos neuronal mediante SVD.*
- `get_loaders` (line 230)
- `breathe_life` (line 254)
- `reset_seeds` (line 353) - *Reinicia el determinismo para que cada variante juegue en igualdad de condiciones.*
- `run_ablation_test` (line 360) - *Ejecuta el Juicio Final: Compara las diferentes configuraciones del cerebro.*
- `train_engine_wrapper` (line 425) - *Versión simplificada de breathe_life para el test que retorna la precisión.
Silencia logs intermedios para limpiar la salida.*
- `__init__` (line 98)
- `reset_fast_weights` (line 115)
- `forward` (line 120)
- `get_fast_norm` (line 140)
- `__init__` (line 144)
- `forward` (line 155)
- `__init__` (line 168)
- `forward` (line 176)
- `__init__` (line 191)
- `forward` (line 215)
- `reset_all_fast_weights` (line 222)

#### `live_ki.py`
**Path:** `live_ki.py`

**Classs:**
- `Config` (line 26)
- `FastSlowLinear` (line 103)
- `DualSystemModule` (line 177)
- `IntegrationModule` (line 211)
- `OmniBrainGenesis` (line 243)

**Functions:**
- `compute_integration_index` (line 76) - *Mide el grado de orden en la actividad neural mediante SVD.
Retorna 0.0 si no hay suficiente información (caos puro).*
- `get_cifar10_loaders` (line 308)
- `evaluate_ritual` (line 334)
- `train_genesis` (line 366)
- `explore_realities` (line 499) - *Explora múltiples configuraciones del universo neural*
- `__init__` (line 104)
- `reset_fast_weights` (line 124) - *Ritual de purificación - resetea memoria a corto plazo*
- `update_fast_weights` (line 130) - *Ritual Hebbiano - solo ocurre si los dioses lo permiten*
- `forward` (line 157)
- `get_fast_norm` (line 171)
- `__init__` (line 178)
- `forward` (line 192)
- `__init__` (line 212)
- `forward` (line 224)
- `__init__` (line 244)
- `forward` (line 279)
- `reset_all_fast_weights` (line 286) - *Ritual de purificación global*
- `get_fast_norms` (line 292) - *Recopila energías de pesos rápidos*
- `get_ablation_state` (line 296) - *Estado de creación*

#### `live_qw.py`
**Path:** `live_qw.py`

**Classs:**
- `Config` (line 23)
- `FastSlowLinear` (line 64) - *Módulo estabilizado – aunque no se usa en baseline, se mantiene para futura ablación.*
- `DualSystemModule` (line 122)
- `IntegrationModule` (line 146)
- `OmniBrainFastSlow` (line 191)

**Functions:**
- `compute_integration_index` (line 171)
- `get_cifar10_loaders` (line 244)
- `evaluate_full` (line 265)
- `train` (line 286)
- `__init__` (line 66)
- `reset_fast_weights` (line 83)
- `update_fast_weights` (line 88)
- `forward` (line 106)
- `get_fast_norm` (line 118)
- `__init__` (line 123)
- `forward` (line 133)
- `__init__` (line 147)
- `forward` (line 158)
- `__init__` (line 192)
- `forward` (line 217)
- `reset_all_fast_weights` (line 224)
- `get_fast_norms` (line 229)
- `get_ablation_state` (line 232)

#### `lol.py`
**Path:** `lol.py`

**Classs:**
- `LiquidNeuron` (line 107)
- `RightHemisphere` (line 180)
- `LeftHemisphere` (line 201)
- `CorpusCallosum` (line 298)
- `NeuroLogosBicameral` (line 313)
- `NeuralDiagnostics` (line 334)
- `Flickr8kDataset` (line 404)
- `LifeCycle` (line 466)

**Functions:**
- `setup_flickr8k` (line 34) - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 443)
- `train_bicameral` (line 481)
- `__init__` (line 108)
- `forward` (line 125)
- `consolidate_svd` (line 154)
- `__init__` (line 181)
- `forward` (line 192)
- `__init__` (line 202)
- `forward` (line 224)
- `_get_init_state` (line 278)
- `_top_p_filtering` (line 283)
- `__init__` (line 299)
- `forward` (line 307)
- `__init__` (line 314)
- `forward` (line 320)
- `__init__` (line 335)
- `measure_callosal_flow` (line 346)
- `measure_vocab_diversity` (line 353)
- `update` (line 357)
- `get_recent_avg` (line 362)
- `report` (line 367)
- `__init__` (line 405)
- `__len__` (line 423)
- `__getitem__` (line 426)
- `__init__` (line 467)
- `get_plasticity` (line 470)

#### `main.py`
**Path:** `main.py`

**Classs:**
- `RESMAConstants` (line 32) - *Constantes físicas y parámetros de la teoría RESMA*
- `PhysicalValidator` (line 58) - *Validación de rangos físicos para todas las constantes*
- `QuantumLeaf` (line 90) - *Hoja L_i de la Resma como estado KMS mean-field.
No almacena matrices densas (Pilar 4).*
- `RESMAUniverse` (line 144) - *Multiverso como foliación medible sin matrices densas.
Memoria: O(N_leaves) en lugar de O(N_leaves × dim²)*
- `BranchingOperator` (line 220) - *Operador Ĥ que abre la Resma cuando β_i es no trivial.
Implementación local sin matrices globales (Pilar 4).*
- `EmunaOperator` (line 270) - *Operador P̂_E: proyección teleológica no lineal.
Implementación con muestreo Monte Carlo (Pilar 4).*
- `LindbladFractalDynamics` (line 350) - *SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ] + ξ(t)
Integración por Euler-Maruyama (Pilar 4: estabilidad numérica).*
- `MyelinCavity` (line 428) - *Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.
Implementación 1D mean-field para Colab (Pilar 4).*
- `NeuralNetworkRESMA` (line 481) - *Conectoma humano dirigido con homología persistente.
Implementación sparse para escalado (Pilar 4).*
- `FreedomInvariant` (line 582) - *L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.
Cálculo topológico sin densidades matriciales (Pilar 4).*
- `NullModels` (line 628) - *Modelos nulos para cálculo de Factor de Bayes.
Basados en teorías establecidas sin postulados RESMA.*
- `ExperimentalPredictions` (line 686) - *Predicciones falsables contra modelos nulos teóricos.
Factor de Bayes calculado con AIC (aproximación).*

**Functions:**
- `simulate_resma_multiverse` (line 764) - *Pipeline completo RESMA 3.0 con verificaciones de integridad.
Diseñado para ejecución en Google Colab (Pilar 4).*
- `validate_dimension` (line 62) - *α ∈ (0,1) por definición de dimensión fractal*
- `validate_pt_symmetry` (line 68) - *Verificar κ/Ω < χ/Ω < 1 para PT-simetría*
- `validate_connectome_size` (line 79) - *Límite inferior para conectoma biológico*
- `__post_init__` (line 100) - *Validaciones post-construcción (Pilar 3)*
- `spectral_density` (line 106) - *Densidad espectral continua ρ(ω) para álgebra tipo III₁.
Evidencia: SYK tiene espectro continuo sin gaps (Maldacena, JHEP 2016).*
- `modular_entropy` (line 114) - *Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)*
- `bures_distance` (line 121)
- `_spectral_moments` (line 132) - *Momentos espectrales Tr(ρ^k) para k=1..n*
- `__init__` (line 150) - *Args:
    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)
    seed: Reproducibilidad (Pilar 4)*
- `_initialize_leaves` (line 168) - *Genera hojas con gaps espectrales distribuidos*
- `_generate_gibbs_measure` (line 181) - *Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))*
- `_construct_global_state` (line 203) - *Estado global: mapa de pesos por hoja (no matriz)*
- `__init__` (line 226)
- `_construct_cptp_map` (line 231) - *Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)*
- `_local_jump_operator` (line 240) - *K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.
Aproximación mean-field: operadores de Pauli escalados por gap.*
- `apply_branching` (line 255) - *Aplicar canal CPTP a vector de estado local (dim=2)*
- `__init__` (line 276)
- `_construct_hardy_state` (line 282) - *E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior*
- `_szego_projector` (line 286) - *Proyector P_E en base de Fourier positiva (dim reducida)*
- `_evaluation_functional` (line 294) - *Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))*
- `project` (line 310) - *P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO*
- `__init__` (line 356)
- `_effective_hamiltonian` (line 362) - *H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)*
- `_modular_dissipator` (line 372) - *L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}*
- `_nonlinear_term` (line 382) - *G[ρ, log ρ_∞] = g[ρ, log ρ_∞]*
- `evolve` (line 389) - *Integración SDE con Euler-Maruyama.
Returns: trayectoria [n_steps, 2, 2]*
- `__post_init__` (line 437)
- `_free_hamiltonian` (line 442) - *H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 448) - *V_loss ∝ (r⊥/a₀)^{2α} con α=0.7*
- `_pt_symmetry_condition` (line 456) - *Verificar κ/Ω < χ/Ω < 1*
- `coherence_quantum` (line 462) - *Discordia cuántica aproximada (ejemplo: estado separable → 0)*
- `__init__` (line 487)
- `_generate_fractal_graph` (line 500) - *Grafo dirigido con distribución de grados power-law.
Fuente: Human Connectome Project (Pilar 1).*
- `_spectral_dimension` (line 510) - *d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)*
- `_topological_ramsey` (line 527) - *R_Q(G) = min{n | β_{n-1}(G) > 0}
Requiere ripser (instalable en Colab: !pip install ripser)*
- `_graph_to_distance_matrix` (line 548) - *Matriz de distancias shortest-path (sparse CSR)*
- `critical_percolation_time` (line 562) - *t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25
Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)*
- `is_coherent_subgraph` (line 574) - *Verificar coherencia: subgrafo > 70% del total*
- `__init__` (line 588)
- `compute_entropy_gap` (line 592) - *Δ_S* = ε_c en punto excepcional*
- `compute_pontryagin_number` (line 596) - *S_top[G] = χ(G)/|V| (número de Euler normalizado)*
- `compute_freedom` (line 607) - *L[G] = Δ_S* / S_top[G]*
- `is_gauge_invariant` (line 618) - *|L[G] - 1| < 0.05 en estado crítico*
- `ising_quantum` (line 635) - *Modelo de Ising cuántico transversal en red fractal.
Predice t_c sin SYK₈ ni E₈.*
- `syk4` (line 654) - *SYK₄ estándar (sin R-simetría Spin(7)).
Predice α sin postulado E₈.*
- `random_network` (line 670) - *Red aleatoria Erdős-Rényi sin percolación cuántica.*
- `__init__` (line 692)
- `predict_all` (line 699) - *Predicciones RESMA 3.0*
- `_predict_diffraction_peak` (line 710) - *q₀ = 2π/L_E8 (sin ajuste)*
- `compute_bayes_factor` (line 715) - *BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)
k = número de parámetros RESMA = 5 (α, β, γ, L_E8, g_coupling)*

#### `main2.py`
**Path:** `main2.py`

**Classs:**
- `RESMAConstants` (line 33) - *Constantes físicas y parámetros de la teoría RESMA*
- `PhysicalValidator` (line 59) - *Validación de rangos físicos para todas las constantes*
- `QuantumLeaf` (line 91) - *Hoja L_i de la Resma como estado KMS mean-field.
No almacena matrices densas (Pilar 4).*
- `RESMAUniverse` (line 144) - *Multiverso como foliación medible sin matrices densas.
Memoria: O(N_leaves) en lugar de O(N_leaves × dim²)*
- `BranchingOperator` (line 219) - *Operador Ĥ que abre la Resma cuando β_i es no trivial.
Implementación local sin matrices globales (Pilar 4).*
- `EmunaOperator` (line 269) - *Operador P̂_E: proyección teleológica no lineal.
Implementación con muestreo Monte Carlo (Pilar 4).*
- `LindbladFractalDynamics` (line 349) - *SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ] + ξ(t)
Integración por Euler-Maruyama (Pilar 4: estabilidad numérica).*
- `MyelinCavity` (line 427) - *Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.
Implementación 1D mean-field para Colab (Pilar 4).*
- `NeuralNetworkRESMA` (line 480) - *Conectoma humano dirigido con homología persistente.
Implementación sparse para escalado (Pilar 4).*
- `FreedomInvariant` (line 581) - *L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.
Cálculo topológico sin densidades matriciales (Pilar 4).*
- `NullModels` (line 627) - *Modelos nulos para cálculo de Factor de Bayes.
Basados en teorías establecidas sin postulados RESMA.*
- `ExperimentalPredictions` (line 685) - *Predicciones falsables contra modelos nulos teóricos.
Factor de Bayes calculado con AIC (aproximación).*

**Functions:**
- `simulate_resma_multiverse` (line 763) - *Pipeline completo RESMA 3.0 con verificaciones de integridad.
Diseñado para ejecución en Google Colab (Pilar 4).*
- `validate_dimension` (line 63) - *α ∈ (0,1) por definición de dimensión fractal*
- `validate_pt_symmetry` (line 69) - *Verificar κ/Ω < χ/Ω < 1 para PT-simetría*
- `validate_connectome_size` (line 80) - *Límite inferior para conectoma biológico*
- `__post_init__` (line 101) - *Validaciones post-construcción (Pilar 3)*
- `spectral_density` (line 106) - *Densidad espectral continua ρ(ω) para álgebra tipo III₁.
Evidencia: SYK tiene espectro continuo sin gaps (Maldacena, JHEP 2016).*
- `modular_entropy` (line 114) - *Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)*
- `bures_distance` (line 121)
- `_spectral_moments` (line 132) - *Momentos espectrales Tr(ρ^k) para k=1..n*
- `__init__` (line 150) - *Args:
    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)
    seed: Reproducibilidad (Pilar 4)*
- `_initialize_leaves` (line 167) - *Genera hojas con gaps espectrales distribuidos*
- `_generate_gibbs_measure` (line 180) - *Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))*
- `_construct_global_state` (line 202) - *Estado global: mapa de pesos por hoja (no matriz)*
- `__init__` (line 225)
- `_construct_cptp_map` (line 230) - *Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)*
- `_local_jump_operator` (line 239) - *K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.
Aproximación mean-field: operadores de Pauli escalados por gap.*
- `apply_branching` (line 254) - *Aplicar canal CPTP a vector de estado local (dim=2)*
- `__init__` (line 275)
- `_construct_hardy_state` (line 281) - *E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior*
- `_szego_projector` (line 285) - *Proyector P_E en base de Fourier positiva (dim reducida)*
- `_evaluation_functional` (line 293) - *Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))*
- `project` (line 309) - *P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO*
- `__init__` (line 355)
- `_effective_hamiltonian` (line 361) - *H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)*
- `_modular_dissipator` (line 371) - *L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}*
- `_nonlinear_term` (line 381) - *G[ρ, log ρ_∞] = g[ρ, log ρ_∞]*
- `evolve` (line 388) - *Integración SDE con Euler-Maruyama.
Returns: trayectoria [n_steps, 2, 2]*
- `__post_init__` (line 436)
- `_free_hamiltonian` (line 441) - *H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 447) - *V_loss ∝ (r⊥/a₀)^{2α} con α=0.7*
- `_pt_symmetry_condition` (line 455) - *Verificar κ/Ω < χ/Ω < 1*
- `coherence_quantum` (line 461) - *Discordia cuántica aproximada (ejemplo: estado separable → 0)*
- `__init__` (line 486)
- `_generate_fractal_graph` (line 499) - *Grafo dirigido con distribución de grados power-law.
Fuente: Human Connectome Project (Pilar 1).*
- `_spectral_dimension` (line 509) - *d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)*
- `_topological_ramsey` (line 526) - *R_Q(G) = min{n | β_{n-1}(G) > 0}
Requiere ripser (instalable en Colab: !pip install ripser)*
- `_graph_to_distance_matrix` (line 547) - *Matriz de distancias shortest-path (sparse CSR)*
- `critical_percolation_time` (line 561) - *t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25
Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)*
- `is_coherent_subgraph` (line 573) - *Verificar coherencia: subgrafo > 70% del total*
- `__init__` (line 587)
- `compute_entropy_gap` (line 591) - *Δ_S* = ε_c en punto excepcional*
- `compute_pontryagin_number` (line 595) - *S_top[G] = χ(G)/|V| (número de Euler normalizado)*
- `compute_freedom` (line 606) - *L[G] = Δ_S* / S_top[G]*
- `is_gauge_invariant` (line 617) - *|L[G] - 1| < 0.05 en estado crítico*
- `ising_quantum` (line 634) - *Modelo de Ising cuántico transversal en red fractal.
Predice t_c sin SYK₈ ni E₈.*
- `syk4` (line 653) - *SYK₄ estándar (sin R-simetría Spin(7)).
Predice α sin postulado E₈.*
- `random_network` (line 669) - *Red aleatoria Erdős-Rényi sin percolación cuántica.*
- `__init__` (line 691)
- `predict_all` (line 698) - *Predicciones RESMA 3.0*
- `_predict_diffraction_peak` (line 708) - *q₀ = 2π/L_E8 (sin ajuste)*
- `compute_bayes_factor` (line 713) - *BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)
k = número de parámetros RESMA = 5 (α, β, γ, L_E8, g_coupling)*

#### `main3.py`
**Path:** `main3.py`

**Classs:**
- `RC` (line 33)
- `Validator` (line 50)
- `QuantumLeaf` (line 68)
- `Universe` (line 101)
- `Network` (line 129)
- `MyelinCavity` (line 171)
- `Bayes` (line 205)

**Functions:**
- `simulate` (line 233)
- `dim` (line 52)
- `pt` (line 56)
- `size` (line 59)
- `__post_init__` (line 74)
- `spectral_density` (line 78)
- `modular_entropy` (line 81)
- `bures_distance` (line 87)
- `__init__` (line 102)
- `_gibbs` (line 110)
- `_global` (line 119)
- `__init__` (line 130)
- `_spectral_dim` (line 139)
- `_ramsey` (line 149)
- `t_c` (line 163)
- `__init__` (line 172)
- `_free_hamiltonian` (line 178)
- `_loss_potential` (line 183)
- `_pt_symmetry_condition` (line 189)
- `coherence_quantum` (line 192)
- `__init__` (line 206)
- `log_lik` (line 210)
- `bf` (line 217)

#### `main4.1.py`
**Path:** `main4.1.py`

**Classs:**
- `RC` (line 29)
- `Validator` (line 61)
- `QuantumLeaf` (line 82)
- `Universe` (line 123)
- `Network` (line 152)
- `MyelinCavity` (line 229)
- `Bayes` (line 268)

**Functions:**
- `simulate` (line 307)
- `verify_pt_condition` (line 50) - *Verifica que kappa < chi*Omega para simetría PT*
- `dim` (line 63)
- `pt` (line 68) - *Condición PT: kappa < chi*Omega*
- `size` (line 73)
- `__post_init__` (line 88)
- `spectral_density` (line 92)
- `modular_entropy` (line 95)
- `bures_distance` (line 104)
- `__init__` (line 124)
- `_gibbs` (line 133)
- `_global` (line 142)
- `__init__` (line 153)
- `_spectral_dim` (line 163) - *Dimensión espectral corregida*
- `_ramsey` (line 199) - *Número de Ramsey topológico*
- `t_c` (line 218) - *Tiempo crítico de percolación*
- `__init__` (line 230)
- `_free_hamiltonian` (line 237)
- `_loss_potential` (line 242)
- `_pt_symmetry_condition` (line 248)
- `coherence_quantum` (line 251)
- `__init__` (line 269)
- `log_lik` (line 273) - *Verosimilitud con escalas físicas realistas*
- `ln_bf` (line 288) - *Factor de Bayes con penalización de complejidad*

#### `main4.py.py`
**Path:** `main4.py.py`

**Classs:**
- `RC` (line 33)
- `Validator` (line 50)
- `QuantumLeaf` (line 69)
- `Universe` (line 102)
- `Network` (line 130)
- `MyelinCavity` (line 198)
- `Bayes` (line 232)

**Functions:**
- `simulate` (line 260)
- `dim` (line 52)
- `pt` (line 56)
- `size` (line 60)
- `__post_init__` (line 75)
- `spectral_density` (line 79)
- `modular_entropy` (line 82)
- `bures_distance` (line 88)
- `__init__` (line 103)
- `_gibbs` (line 111)
- `_global` (line 120)
- `__init__` (line 131)
- `_spectral_dim` (line 140)
- `_ramsey` (line 176)
- `t_c` (line 190)
- `__init__` (line 199)
- `_free_hamiltonian` (line 205)
- `_loss_potential` (line 210)
- `_pt_symmetry_condition` (line 216)
- `coherence_quantum` (line 219)
- `__init__` (line 233)
- `log_lik` (line 237)
- `ln_bf` (line 244)

#### `main5.py`
**Path:** `main5.py`

**Classs:**
- `RESMAConstants` (line 37) - *Constantes físicas y parámetros de la teoría RESMA 4.0*
- `PhysicalValidator` (line 70) - *Validación de rangos físicos para todas las constantes RESMA 4.0*
- `QuantumLeaf` (line 116) - *Hoja L_i de la Resma como estado KMS mean-field con espacio de Hilbert standard.
Implementación RESMA 4.0 con regularización Haagerup.*
- `RESMAUniverse` (line 179) - *Multiverso como foliación medible sin matrices densas, con espacio de Hilbert standard.
Memoria: O(N_leaves) con regularización de transiciones.*
- `BranchingOperator` (line 260) - *Operador Ĥ que abre la Resma cuando β_i es no trivial.
Implementación local con operadores de salto SYK₈ (Pilar 4).*
- `EmunaOperator` (line 318) - *Operador P̂_E: proyección teleológica no lineal.
Implementación con muestreo Monte Carlo y espacio de Hardy H²(ℂ⁺) (Pilar 4).*
- `LindbladFractalDynamics` (line 406) - *SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ_∞] + ξ(t)
Integración por Euler-Maruyama con control de precisión (Pilar 4).*
- `MyelinCavity` (line 539) - *Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.
Implementación 1D mean-field con R-simetría Spin(7) (Pilar 4).*
- `NeuralNetworkRESMA` (line 604) - *Conectoma humano NO DIRIGIDO con homología persistente.
Implementación sparse para escalado con conversión a grafo no dirigido (Pilar 4).*
- `FreedomInvariant` (line 762) - *L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.
Cálculo topológico sin densidades matriciales (Pilar 4).*
- `NullModels` (line 816) - *Modelos nulos para cálculo de Factor de Bayes.
Basados en teorías establecidas sin postulados holográficos de RESMA.*
- `ExperimentalPredictions` (line 877) - *Predicciones falsables contra modelos nulos teóricos.
Factor de Bayes calculado con AIC y transformaciones logarítmicas (FIX).*
- `EmpiricalValidationProtocol` (line 975) - *Protocolo experimental para falsación controlada de RESMA 4.0.
Define setups experimentales y criterios de éxito.*

**Functions:**
- `simulate_resma_multiverse` (line 1052) - *Pipeline completo RESMA 4.0 con verificaciones de integridad y protocolo de validación.
Diseñado para ejecución en Google Colab (Pilar 4).*
- `validate_dimension` (line 74) - *α ∈ (0,1) por definición de dimensión fractal, con tolerancia experimental*
- `validate_pt_symmetry` (line 84) - *Verificar κ/Ω < χ/Ω < 1 para PT-simetría (corregido con factor de seguridad)*
- `validate_connectome_size` (line 95) - *Límite inferior para conectoma biológico realista*
- `validate_spectral_dimension` (line 101) - *Validar rango físico para dimensión espectral*
- `validate_percolation_time` (line 106) - *Validar tiempo de percolación contra predicción empírica*
- `__post_init__` (line 127) - *Validaciones post-construcción (Pilar 3)*
- `spectral_density` (line 133) - *Densidad espectral continua ρ(ω) para álgebra tipo III₁ con regularización UV.
Evidencia: SYK₈ con Spin(7) tiene espectro continuo con gap infrarrojo.*
- `modular_entropy` (line 143) - *Entropía modular S = ∫ ρ(ω)logρ(ω) dω con regularización*
- `bures_distance` (line 151)
- `_spectral_moments` (line 163) - *Momentos espectrales Tr(ρ^k) para k=1..n con regularización*
- `haagerup_weight` (line 170) - *Peso de Haagerup para regularización del operador modular*
- `__init__` (line 185) - *Args:
    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)
    seed: Reproducibilidad (Pilar 4)*
- `_initialize_leaves` (line 203) - *Genera hojas con gaps espectrales distribuidos exponencialmente*
- `_generate_gibbs_measure` (line 217) - *Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j)) con normalización robusta*
- `_construct_global_state` (line 239) - *Estado global: mapa de pesos por hoja (no matriz) con regularización*
- `compute_gibbs_free_energy` (line 251) - *Energía libre de Gibbs para validación termodinámica*
- `__init__` (line 266)
- `_compute_holonomy` (line 272) - *Defecto de holonomía como variación del gap espectral*
- `_construct_cptp_map` (line 276) - *Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)*
- `_local_jump_operator` (line 284) - *K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.
Aproximación mean-field: operadores de Pauli escalados por gap SYK₈.*
- `apply_branching` (line 300) - *Aplicar canal CPTP a vector de estado local (dim=2) con normalización*
- `__init__` (line 324)
- `_construct_hardy_state` (line 331) - *E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior*
- `_szego_projector` (line 335) - *Proyector P_E en base de Fourier positiva (dim reducida)*
- `_evaluation_functional` (line 345) - *Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))*
- `project` (line 361) - *P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO*
- `compute_teleological_overlap` (line 396) - *Calcular overlap teleológico con estado objetivo*
- `__init__` (line 412)
- `_effective_hamiltonian` (line 419) - *H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva con gaps SYK₈)*
- `_modular_dissipator` (line 432) - *L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ} con regularización*
- `_nonlinear_term` (line 444) - *G[ρ, log ρ_∞] = g[ρ, log ρ_∞] con regularización del logaritmo*
- `_stochastic_term` (line 452) - *Término estocástico ξ(t) con correlaciones cuánticas*
- `evolve` (line 459) - *Integración SDE con Euler-Maruyama y control de paso adaptativo.
Returns: trayectoria [n_steps, 2, 2]*
- `_normalize_density_matrix` (line 500) - *Normalizar matriz densidad y forzar hermiticidad*
- `_is_physical_state` (line 509) - *Verificar si el estado es físico (hermitiano, traza=1, positivo)*
- `_correct_non_physical_state` (line 523) - *Corregir estado no físico proyectando en el cono de estados válidos*
- `__post_init__` (line 548)
- `_free_hamiltonian` (line 555) - *H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 561) - *V_loss ∝ (r⊥/a₀)^{2α} con α=0.702 (SYK₈)*
- `_compute_scalar_mass` (line 569) - *Campo escalar masivo para estabilización de Spin(7)*
- `_pt_symmetry_condition` (line 573) - *Verificar κ/Ω < χ/Ω < 1 con parámetros corregidos*
- `coherence_quantum` (line 579) - *Discordia cuántica aproximada con corrección PT*
- `__init__` (line 610)
- `_generate_fractal_graph` (line 625) - *Generar grafo dirigido y convertir a NO DIRIGIDO para análisis espectral.
SOLUCIÓN RESMA 4.0: Conversión explícita con to_undirected().*
- `_spectral_dimension` (line 649) - *d_s = -2 lim_{λ→0⁺} log N(λ)/log λ usando normalized_laplacian_spectrum.
SOLUCIÓN RESMA 4.0: Uso de función especializada de NetworkX.*
- `_topological_ramsey` (line 680) - *R_Q(G) = min{n | β_{n-1}(G) > 0}
Requiere ripser (instalable en Colab: !pip install ripser)*
- `_compute_betti_numbers` (line 706) - *Calcular números de Betti para análisis topológico*
- `_graph_to_distance_matrix` (line 722) - *Matriz de distancias shortest-path (sparse CSR) para homología*
- `critical_percolation_time` (line 735) - *t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25
Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)*
- `is_coherent_subgraph` (line 747) - *Verificar coherencia: subgrafo > 70% del total*
- `compute_network_entropy` (line 751) - *Entropía de la red basada en distribución de grados*
- `__init__` (line 768)
- `compute_entropy_gap` (line 772) - *Δ_S* = ε_c en punto excepcional con corrección de regularización*
- `compute_pontryagin_number` (line 776) - *S_top[G] = χ(G)/|V| (número de Euler normalizado)*
- `compute_freedom` (line 792) - *L[G] = Δ_S* / S_top[G] con protección de división por cero*
- `is_gauge_invariant` (line 803) - *|L[G] - 1| < 0.05 en estado crítico (invariante de libertad)*
- `ising_quantum` (line 823) - *Modelo de Ising cuántico transversal en red fractal.
Predice t_c sin SYK₈ ni E₈ (teoría efectiva estándar).*
- `syk4` (line 843) - *SYK₄ estándar (sin R-simetría Spin(7) ni E₈).
Predice α sin postulado de retículo.*
- `random_network` (line 860) - *Red aleatoria Erdős-Rényi sin percolación cuántica ni estructura.*
- `__init__` (line 883)
- `predict_all` (line 891) - *Predicciones RESMA 4.0 con valores empíricos objetivo*
- `_predict_diffraction_peak` (line 906) - *q₀ = 2π/L_E8 (predicción de difracción UASED)*
- `compute_log_bayes_factor` (line 912) - *log(BF) = ΔAIC/2 donde AIC = 2k - 2ln(L)
FIX RESMA 4.0: Usar espacio logarítmico para evitar desbordamiento.*
- `__init__` (line 981)
- `_define_protocols` (line 985) - *Definir protocolos experimentales con parámetros técnicos*
- `evaluate_feasibility` (line 1014) - *Evaluar viabilidad del protocolo completo*
- `simulate_experimental_outcome` (line 1027) - *Simular resultado experimental con ruido realista*

#### `microbi.py.py`
**Path:** `microbi.py.py`

**Classs:**
- `EpistemicCuriosityCPU` (line 37)
- `LiquidNeuronCPU` (line 91)
- `BicameralAttentionCPU` (line 195)
- `RightHemisphereCPU` (line 248)
- `LeftHemisphereCPU` (line 308)
- `CorpusCallosumCPU` (line 436)
- `NeuroLogosBicameralCPU` (line 473)
- `NeuralDiagnosticsCPU` (line 513)
- `CurriculumSchedulerCPU` (line 602)
- `Flickr8kDatasetCPU` (line 671)

**Functions:**
- `build_vocab_flickr` (line 650)
- `setup_flickr8k_cpu` (line 721) - *Descarga Flickr8k automáticamente (igual que la versión original)*
- `train_bicameral_cpu` (line 801)
- `__init__` (line 38)
- `compute_intrinsic_reward` (line 55)
- `update` (line 71)
- `__init__` (line 92)
- `forward` (line 117)
- `consolidate_svd` (line 168)
- `__init__` (line 196)
- `forward` (line 211)
- `__init__` (line 249)
- `forward` (line 285)
- `__init__` (line 309)
- `forward` (line 338)
- `_get_init_state` (line 416)
- `_top_p_filtering` (line 421)
- `__init__` (line 437)
- `forward` (line 456)
- `__init__` (line 474)
- `forward` (line 480)
- `__init__` (line 514)
- `measure_callosal_flow` (line 523)
- `measure_vocab_diversity` (line 530)
- `update` (line 548)
- `get_recent_avg` (line 553)
- `report` (line 560)
- `__init__` (line 603)
- `get_phase` (line 611)
- `get_plasticity` (line 617)
- `get_exploration_bonus` (line 627)
- `get_temperature` (line 636)
- `should_consolidate` (line 646)
- `__init__` (line 672)
- `__len__` (line 690)
- `__getitem__` (line 693)
- `nan_hook` (line 880)

#### `min_test_synergy.py`
**Path:** `min_test_synergy.py`

*No symbols extracted*

#### `minibi.py`
**Path:** `minibi.py`

**Classs:**
- `Flickr8kDataset` (line 124)
- `LiquidNeuron` (line 167)
- `RightHemisphere` (line 246) - *Especialización: Visión espacial, reconocimiento de objetos, contexto global
Usa ResNet-50 pretrained en ImageNet → Feature extraction robusto*
- `LeftHemisphere` (line 282)
- `CorpusCallosum` (line 382)
- `NeuroLogosBicameral` (line 403)
- `NeuralDiagnostics` (line 420)
- `LifeCycle` (line 518)

**Functions:**
- `setup_flickr8k` (line 35) - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 104)
- `build_vocab` (line 490) - *Construir vocabulario desde annotations de COCO*
- `train_bicameral` (line 536)
- `__init__` (line 125)
- `__len__` (line 143)
- `__getitem__` (line 146)
- `__init__` (line 168)
- `forward` (line 185)
- `consolidate_svd` (line 219)
- `__init__` (line 251)
- `forward` (line 268)
- `__init__` (line 283)
- `forward` (line 305)
- `_get_init_state` (line 361)
- `_top_p_filtering` (line 366)
- `__init__` (line 383)
- `forward` (line 392)
- `__init__` (line 404)
- `forward` (line 410)
- `__init__` (line 421)
- `measure_callosal_flow` (line 432)
- `measure_vocab_diversity` (line 439)
- `update` (line 443)
- `get_recent_avg` (line 448)
- `report` (line 455)
- `__init__` (line 519)
- `get_plasticity` (line 522)

#### `minibi2.py`
**Path:** `minibi2.py`

**Classs:**
- `EpistemicCuriosity` (line 36) - *Implementa curiosidad intrínseca basada en incertidumbre predictiva.
Paper: "Curiosity-driven Exploration by Self-supervised Prediction"*
- `LiquidNeuronV2` (line 101) - *Mejoras:
- Regularización adaptativa (no fija)
- Homeostasis metabólica
- Rango de operación expandido*
- `BicameralAttention` (line 212) - *Atención multi-escala que simula integración hemisférica.
- Local attention: detalles finos (hemisferio izquierdo)
- Global attention: contexto amplio (hemisferio derecho)*
- `RightHemisphereV2` (line 279) - *Mejoras:
- Spatial attention pyramid
- Multi-scale feature extraction
- Liquid neuron con homeostasis*
- `LeftHemisphereV2` (line 344) - *Mejoras:
- Bicameral attention
- Gated residual connections
- Adaptive gate range
- Curiosity-driven sampling*
- `CorpusCallosumV2` (line 519) - *Mejoras:
- Gating bidireccional
- Modulación adaptativa
- Información mutua maximizada*
- `NeuroLogosBicameralV2` (line 562)
- `NeuralDiagnosticsV2` (line 591)
- `CurriculumScheduler` (line 700) - *Curriculum learning con fases de desarrollo cognitivo*
- `Flickr8kDataset` (line 781)

**Functions:**
- `build_vocab_flickr` (line 760)
- `setup_flickr8k` (line 821) - *Descarga Flickr8k automáticamente*
- `train_bicameral_v2` (line 893)
- `__init__` (line 41)
- `compute_intrinsic_reward` (line 60) - *Recompensa intrínseca = error de predicción del forward model
Incentiva explorar tokens que son difíciles de predecir*
- `update` (line 83) - *Entrena los modelos de curiosidad*
- `__init__` (line 108)
- `forward` (line 137)
- `consolidate_svd` (line 185)
- `__init__` (line 218)
- `forward` (line 236)
- `__init__` (line 286)
- `forward` (line 320)
- `__init__` (line 352)
- `forward` (line 387)
- `_get_init_state` (line 498)
- `_top_p_filtering` (line 503)
- `__init__` (line 526)
- `forward` (line 546)
- `__init__` (line 563)
- `forward` (line 569)
- `__init__` (line 592)
- `measure_callosal_flow` (line 612)
- `measure_vocab_diversity` (line 619) - *Mide diversidad real + entropía*
- `update` (line 639)
- `get_recent_avg` (line 644)
- `report` (line 651)
- `__init__` (line 704)
- `get_phase` (line 712)
- `get_plasticity` (line 718)
- `get_exploration_bonus` (line 732) - *Bonus de curiosidad que decae con el tiempo*
- `get_temperature` (line 743) - *Temperature que decae suavemente*
- `should_consolidate` (line 755) - *Decide cuándo hacer consolidación SVD*
- `__init__` (line 782)
- `__len__` (line 800)
- `__getitem__` (line 803)

#### `minibi_c.py`
**Path:** `minibi_c.py`

**Classs:**
- `EpistemicCuriosity` (line 37)
- `LiquidNeuronV2` (line 94)
- `BicameralAttention` (line 197)
- `RightHemisphereV2` (line 247)
- `LeftHemisphereV2` (line 301)
- `CorpusCallosumV2` (line 433)
- `NeuroLogosBicameralV2` (line 464)
- `NeuralDiagnosticsV2` (line 493)
- `CurriculumScheduler` (line 593)
- `Flickr8kDataset` (line 667)

**Functions:**
- `build_vocab_flickr` (line 644)
- `setup_flickr8k` (line 722) - *Descarga Flickr8k automáticamente*
- `train_bicameral_v2` (line 794)
- `__init__` (line 38)
- `compute_intrinsic_reward` (line 55)
- `update` (line 72)
- `__init__` (line 95)
- `forward` (line 121)
- `consolidate_svd` (line 170)
- `__init__` (line 198)
- `forward` (line 212)
- `__init__` (line 248)
- `forward` (line 281)
- `__init__` (line 302)
- `forward` (line 332)
- `_get_init_state` (line 412)
- `_top_p_filtering` (line 417)
- `__init__` (line 434)
- `forward` (line 450)
- `__init__` (line 465)
- `forward` (line 471)
- `__init__` (line 494)
- `measure_callosal_flow` (line 510)
- `measure_vocab_diversity` (line 517)
- `update` (line 535)
- `get_recent_avg` (line 541)
- `report` (line 548)
- `__init__` (line 594)
- `get_phase` (line 602)
- `get_plasticity` (line 608)
- `get_exploration_bonus` (line 619)
- `get_temperature` (line 629)
- `should_consolidate` (line 640)
- `__init__` (line 668)
- `__len__` (line 686)
- `__getitem__` (line 689)

#### `minibi_reduced.py.py`
**Path:** `minibi_reduced.py.py`

**Classs:**
- `EpistemicCuriosity` (line 39)
- `LiquidNeuronV2` (line 81)
- `BicameralAttention` (line 170)
- `RightHemisphereV2` (line 213)
- `LeftHemisphereV2` (line 247)
- `CorpusCallosumV2` (line 338)
- `NeuroLogosBicameralV2` (line 356)
- `Flickr8kDataset` (line 396)

**Functions:**
- `build_vocab_flickr` (line 381)
- `setup_flickr8k` (line 431)
- `train_bicameral_v2` (line 482)
- `__init__` (line 40)
- `compute_intrinsic_reward` (line 55)
- `update` (line 70)
- `__init__` (line 82)
- `forward` (line 105)
- `consolidate_svd` (line 141)
- `__init__` (line 171)
- `forward` (line 184)
- `__init__` (line 214)
- `forward` (line 237)
- `__init__` (line 248)
- `forward` (line 266)
- `_get_init_state` (line 323)
- `_top_p_filtering` (line 328)
- `__init__` (line 339)
- `forward` (line 349)
- `__init__` (line 357)
- `forward` (line 363)
- `__init__` (line 397)
- `__len__` (line 412)
- `__getitem__` (line 414)

#### `miniminibi.py`
**Path:** `miniminibi.py`

**Classs:**
- `LiquidNeuron` (line 118)
- `RightHemisphere` (line 191)
- `LeftHemisphere` (line 214)
- `CorpusCallosum` (line 314)
- `NeuroLogosBicameral` (line 332)
- `NeuralDiagnostics` (line 351) - *Sistema de monitoreo de salud cerebral bicameral*
- `Flickr8kDataset` (line 442)
- `LifeCycle` (line 508)

**Functions:**
- `setup_flickr8k` (line 30) - *Descarga Flickr8k automáticamente desde Kaggle
Requiere: pip install kaggle
Y tener configurado ~/.kaggle/kaggle.json

Alternativa sin Kaggle: descarga manual desde
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip*
- `build_vocab_flickr` (line 485)
- `train_bicameral` (line 523)
- `__init__` (line 119)
- `forward` (line 136)
- `consolidate_svd` (line 165)
- `__init__` (line 192)
- `forward` (line 205)
- `__init__` (line 215)
- `forward` (line 237)
- `_get_init_state` (line 293)
- `_top_p_filtering` (line 298)
- `__init__` (line 315)
- `forward` (line 324)
- `__init__` (line 333)
- `forward` (line 339)
- `__init__` (line 353)
- `measure_callosal_flow` (line 363) - *Mide qué tan bien está fluyendo información entre hemisferios*
- `measure_vocab_diversity` (line 372) - *Mide diversidad de vocabulario generado (evitar colapso)*
- `measure_gate_health` (line 378) - *Verifica que el liquid gate no colapse a 0 o 1*
- `update` (line 387)
- `report` (line 392) - *Reporte diagnóstico completo*
- `__init__` (line 443)
- `__len__` (line 462)
- `__getitem__` (line 465)
- `__init__` (line 509)
- `get_plasticity` (line 512)

#### `nemesis.py`
**Path:** `nemesis.py`

**Classs:**
- `DataEnvironment` (line 23)
- `NeuralController` (line 41)
- `HyperLiquidNeuron` (line 60)
- `NemesisNetwork` (line 126)
- `ExperimentConfig` (line 192)

**Functions:**
- `seed_everything` (line 13)
- `run_hyper_experiment` (line 143)
- `__init__` (line 24)
- `get_batch` (line 33)
- `__init__` (line 42)
- `forward` (line 52)
- `__init__` (line 61)
- `forward` (line 79)
- `__init__` (line 127)
- `forward` (line 133)

#### `nested1.1.py`
**Path:** `nested1.1.py`

**Classs:**
- `Config` (line 27)
- `CMSLayer` (line 128)
- `NestedBrain` (line 200)

**Functions:**
- `safe_serialize` (line 61) - *Convierte objetos a formato serializable (evita recursión y objetos complejos).*
- `save_checkpoint` (line 80) - *Guarda checkpoint de época: modelo (.pth) + metadatos (.pkl).*
- `cleanup_old_checkpoints` (line 105) - *Mantiene solo los últimos `keep_last` checkpoints.*
- `get_cifar10_loaders` (line 241)
- `evaluate` (line 267)
- `train` (line 293)
- `run_ablation_study` (line 371)
- `__init__` (line 129)
- `forward` (line 148)
- `get_norms` (line 190)
- `__init__` (line 201)
- `forward` (line 221)
- `get_ablation_state` (line 226)
- `get_norms` (line 233)

#### `nested1.py`
**Path:** `nested1.py`

**Classs:**
- `Config` (line 24)
- `CMSLayer` (line 59)
- `NestedBrain` (line 141)

**Functions:**
- `get_cifar10_loaders` (line 182)
- `evaluate` (line 208)
- `train` (line 234)
- `run_ablation_study` (line 301)
- `__init__` (line 60)
- `forward` (line 83)
- `get_norms` (line 131)
- `__init__` (line 142)
- `forward` (line 162)
- `get_ablation_state` (line 167)
- `get_norms` (line 174)

#### `nestedtopobrain.py`
**Path:** `nestedtopobrain.py`

**Classs:**
- `Config` (line 29)
- `ResourceMonitor` (line 133)
- `PrefrontalOrchestrator` (line 178) - *Módulo de control ejecutivo que monitoriza el estado de la red y emite señales
dinámicas de activación/inhibición para cada mecanismo neuromodulatorio.
Opera como un sistema de homeostasis topológica y metabólica.
FIX v24: Gestión corregida del grafo computacional recurrente (BPTT).*
- `TopologyMetrics` (line 334)
- `TopologicalHealthSovereignty` (line 343) - *Monitor SVD Completo con criterios neurocientíficos

Referencias:
- Sporns (2016): Human brain networks ~1-3% sparsity
- Bullmore & Sporns (2009): Small-world topology con L_score > 3*
- `CheckpointManager` (line 429) - *Manager robusto v18 con backups y metadata*
- `SupConLoss` (line 525)
- `AsymmetricPredictiveErrorCell` (line 542)
- `LearnableAbsenceGating` (line 562)
- `SymbioticBasisRefinement` (line 580)
- `ContinuumMemoryCell` (line 625)
- `AdaptiveCombinatorialComplexLayer` (line 750)
- `ResidualBlock` (line 946)
- `VisualCortex` (line 967)
- `TopoBrainV24` (line 1022)

**Functions:**
- `seed_everything` (line 119)
- `get_dataloaders` (line 481) - *DataLoaders con augmentation de alto rendimiento para CIFAR-10*
- `save_topology_visualization` (line 1406) - *Visualización v18 completa*
- `save_node_importance_viz` (line 1450) - *Visualización de importancia de nodos v18*
- `analyze_topology_clustering` (line 1472) - *Clustering espectral v18*
- `analyze_topology_flow` (line 1511) - *Análisis de flujo de información con captura genérica de outputs*
- `visualize_topology_as_graph` (line 1577) - *Grafo v18 con métricas*
- `analyze_topology_evolution` (line 1629) - *Análisis temporal completo v18*
- `comprehensive_topology_analysis` (line 1690) - *Suite completa de análisis v18*
- `run_ablation_study` (line 1713) - *Suite de ablación v18 completa*
- `visualize_memory_evolution` (line 1814) - *Visualiza evolución de memorias semánticas

Crítico para entender si la consolidación hipocampal→cortical está funcionando.
SVD spectrum revela estructura de representaciones aprendidas.*
- `analyze_gradient_flow` (line 1905) - *Análisis detallado del flujo de gradientes

Detecta vanishing/exploding gradients y capas muertas.
Critical para debugging de arquitecturas con fast weights.*
- `make_adversarial_pgd` (line 1972) - *PGD ataque con congelamiento total de pesos y detach explícito de estados.
FIX: Asegura que el grafo computacional no se rompa y que los estados previos sean genuinamente independientes.*
- `evaluate` (line 2050) - *Evaluación con plasticidad residual (test-time adaptation)
Biológicamente plausible: el cerebro no se apaga durante percepción*
- `train_epoch` (line 2122) - *Entrenamiento homeostático con lista manual en lugar de deque*
- `train_model` (line 2347)
- `main` (line 2518) - *CLI v24 completo con Orquestador Prefrontal*
- `__post_init__` (line 89)
- `to_dict` (line 95)
- `get_supcon_lambda` (line 98)
- `get_sparsity_lambda` (line 104)
- `get_memory_gb` (line 135)
- `get_gpu_memory_gb` (line 140)
- `log` (line 146)
- `clear_cache` (line 153)
- `check_limit` (line 159)
- `__init__` (line 185)
- `forward` (line 222) - *Orquestador v27: Allostasis con Frenado de Emergencia (Gradient-Aware).

FIX CRÍTICO: EL ORQUESTADOR AHORA "SIENTE" SI EL GRADIENTE EXPLOTA
- Si loss > 10.0: Entra en MODO PÁNICO (LR_Scale mínimo)
- Si delta_loss > 0 (loss subiendo): Invierte la señal de aceleración
- Si grad_norm > 10: Reduce plasticidad para estabilizar*
- `detach_state` (line 322) - *Rompe el grafo computacional para evitar retropropagación infinita entre batches*
- `reset_context` (line 327) - *Resetear contexto al inicio de cada época*
- `__init__` (line 350)
- `_analyze_matrix` (line 356)
- `calculate` (line 402)
- `get_critical_summary` (line 419)
- `__init__` (line 431)
- `save` (line 436)
- `load` (line 464)
- `__init__` (line 526)
- `forward` (line 529)
- `__init__` (line 543)
- `forward` (line 553)
- `__init__` (line 563)
- `forward` (line 573)
- `__init__` (line 581)
- `_maintain_orthogonality` (line 595)
- `forward` (line 600)
- `__init__` (line 626)
- `forward` (line 675)
- `__init__` (line 751)
- `invalidate_sparse_cache` (line 789)
- `_validate_and_fix_state` (line 792)
- `get_node_importance` (line 811) - *FIX: Método faltante para obtener importancia de nodos*
- `forward` (line 817) - *Forward con señales de control del Orquestador y retorno de ortho deviation*
- `__init__` (line 947)
- `forward` (line 961)
- `__init__` (line 968)
- `forward` (line 994)
- `__init__` (line 1023)
- `initialize_memories` (line 1087)
- `_initialize_layer_memory` (line 1124)
- `consolidate_semantic_memories` (line 1141)
- `set_epoch` (line 1169)
- `calculate_ortho_loss` (line 1174)
- `calculate_topology_diversity_loss` (line 1180)
- `_init_grid_topology` (line 1193)
- `get_topology` (line 1215)
- `forward` (line 1226)
- `prune_topology` (line 1298)
- `warmup_topo` (line 2381)

#### `nestedtopobrain_v1.py`
**Path:** `nestedtopobrain_v1.py`

**Classs:**
- `Config` (line 29)
- `ResourceMonitor` (line 128)
- `PrefrontalOrchestrator` (line 173) - *Módulo de control ejecutivo que monitoriza el estado de la red y emite señales
dinámicas de activación/inhibición para cada mecanismo neuromodulatorio.
Opera como un sistema de homeostasis topológica y metabólica.
FIX v24: Gestión corregida del grafo computacional recurrente (BPTT).*
- `TopologyMetrics` (line 269)
- `TopologicalHealthSovereignty` (line 278) - *Monitor SVD Completo con criterios neurocientíficos

Referencias:
- Sporns (2016): Human brain networks ~1-3% sparsity
- Bullmore & Sporns (2009): Small-world topology con L_score > 3*
- `CheckpointManager` (line 364) - *Manager robusto v18 con backups y metadata*
- `SupConLoss` (line 455)
- `AsymmetricPredictiveErrorCell` (line 472)
- `LearnableAbsenceGating` (line 492)
- `SymbioticBasisRefinement` (line 510)
- `ContinuumMemoryCell` (line 549)
- `AdaptiveCombinatorialComplexLayer` (line 676)
- `TopoBrainV24` (line 879)

**Functions:**
- `seed_everything` (line 114)
- `get_dataloaders` (line 416)
- `save_topology_visualization` (line 1304) - *Visualización v18 completa*
- `save_node_importance_viz` (line 1348) - *Visualización de importancia de nodos v18*
- `analyze_topology_clustering` (line 1370) - *Clustering espectral v18*
- `analyze_topology_flow` (line 1409) - *Análisis de flujo v18*
- `visualize_topology_as_graph` (line 1461) - *Grafo v18 con métricas*
- `analyze_topology_evolution` (line 1513) - *Análisis temporal completo v18*
- `comprehensive_topology_analysis` (line 1574) - *Suite completa de análisis v18*
- `run_ablation_study` (line 1597) - *Suite de ablación v18 completa*
- `visualize_memory_evolution` (line 1698) - *Visualiza evolución de memorias semánticas

Crítico para entender si la consolidación hipocampal→cortical está funcionando.
SVD spectrum revela estructura de representaciones aprendidas.*
- `analyze_gradient_flow` (line 1789) - *Análisis detallado del flujo de gradientes

Detecta vanishing/exploding gradients y capas muertas.
Critical para debugging de arquitecturas con fast weights.*
- `make_adversarial_pgd` (line 1856) - *PGD attack con congelamiento total de pesos (Protocolo de Aislamiento Sináptico).
Garantiza que la generación del ataque no modifique los gradientes del modelo.*
- `evaluate` (line 1936) - *Evaluación optimizada para arquitecturas biológicas complejas (Nested/Grid).
Implementa 'Gradient Shielding' para prevenir OOM en inferencia.*
- `train_epoch` (line 2000) - *Entrenamiento homeostático con gestión rigurosa de grafos y memoria.
FIX v24.1: Restaurada la visualización detallada de decisiones del Orquestador (Logs de Actividad Prefrontal).*
- `train_model` (line 2196) - *Training loop principal - FIX v24: Orden correcto de Scheduler y gestión de memoria.*
- `main` (line 2398) - *CLI v24 completo con Orquestador Prefrontal*
- `__post_init__` (line 83)
- `to_dict` (line 89)
- `get_supcon_lambda` (line 92)
- `get_sparsity_lambda` (line 98)
- `get_memory_gb` (line 130)
- `get_gpu_memory_gb` (line 135)
- `log` (line 141)
- `clear_cache` (line 148)
- `check_limit` (line 154)
- `__init__` (line 180)
- `forward` (line 217) - *Input: Diccionario con métricas del estado actual
Output: Diccionario con señales de control escaladas [0,1]*
- `detach_state` (line 257) - *Rompe el grafo computacional para evitar retropropagación infinita entre batches*
- `reset_context` (line 262) - *Resetear contexto al inicio de cada época*
- `__init__` (line 285)
- `_analyze_matrix` (line 291)
- `calculate` (line 337)
- `get_critical_summary` (line 354)
- `__init__` (line 366)
- `save` (line 371)
- `load` (line 399)
- `__init__` (line 456)
- `forward` (line 459)
- `__init__` (line 473)
- `forward` (line 483)
- `__init__` (line 493)
- `forward` (line 503)
- `__init__` (line 511)
- `_maintain_orthogonality` (line 525)
- `forward` (line 530)
- `__init__` (line 550)
- `forward` (line 599) - *FIX: Ahora acepta señales de control del Orquestador para modular
la plasticidad y consolidación en tiempo real.*
- `__init__` (line 677)
- `invalidate_sparse_cache` (line 715)
- `_validate_and_fix_state` (line 718)
- `get_node_importance` (line 737) - *FIX: Método faltante para obtener importancia de nodos*
- `forward` (line 743) - *FIX: Integra señales de control del Orquestador con bypass condicional
para evitar computación innecesaria cuando gates están bajos*
- `__init__` (line 880)
- `initialize_memories` (line 934)
- `_initialize_layer_memory` (line 974)
- `consolidate_semantic_memories` (line 1000)
- `set_epoch` (line 1045)
- `calculate_ortho_loss` (line 1050)
- `calculate_topology_diversity_loss` (line 1070)
- `_init_grid_topology` (line 1095)
- `get_topology` (line 1118)
- `forward` (line 1131) - *Forward con validación y detach explícito*
- `prune_topology` (line 1198)
- `warmup_topo` (line 2234)

#### `nestedtopobrain_v2.py`
**Path:** `nestedtopobrain_v2.py`

**Classs:**
- `Config` (line 29)
- `ResourceMonitor` (line 127)
- `PrefrontalOrchestrator` (line 172) - *Módulo de control ejecutivo que monitoriza el estado de la red y emite señales
dinámicas de activación/inhibición para cada mecanismo neuromodulatorio.
Opera como un sistema de homeostasis topológica y metabólica.
FIX v24: Gestión corregida del grafo computacional recurrente (BPTT).*
- `TopologyMetrics` (line 268)
- `TopologicalHealthSovereignty` (line 277) - *Monitor SVD Completo con criterios neurocientíficos

Referencias:
- Sporns (2016): Human brain networks ~1-3% sparsity
- Bullmore & Sporns (2009): Small-world topology con L_score > 3*
- `CheckpointManager` (line 363) - *Manager robusto v18 con backups y metadata*
- `SupConLoss` (line 454)
- `AsymmetricPredictiveErrorCell` (line 471)
- `LearnableAbsenceGating` (line 491)
- `SymbioticBasisRefinement` (line 509)
- `ContinuumMemoryCell` (line 554)
- `AdaptiveCombinatorialComplexLayer` (line 681)
- `TopoBrainV24` (line 878)

**Functions:**
- `seed_everything` (line 113)
- `get_dataloaders` (line 415)
- `save_topology_visualization` (line 1289) - *Visualización v18 completa*
- `save_node_importance_viz` (line 1333) - *Visualización de importancia de nodos v18*
- `analyze_topology_clustering` (line 1355) - *Clustering espectral v18*
- `analyze_topology_flow` (line 1394) - *Análisis de flujo v18*
- `visualize_topology_as_graph` (line 1448) - *Grafo v18 con métricas*
- `analyze_topology_evolution` (line 1500) - *Análisis temporal completo v18*
- `comprehensive_topology_analysis` (line 1561) - *Suite completa de análisis v18*
- `run_ablation_study` (line 1584) - *Suite de ablación v18 completa*
- `visualize_memory_evolution` (line 1685) - *Visualiza evolución de memorias semánticas

Crítico para entender si la consolidación hipocampal→cortical está funcionando.
SVD spectrum revela estructura de representaciones aprendidas.*
- `analyze_gradient_flow` (line 1776) - *Análisis detallado del flujo de gradientes

Detecta vanishing/exploding gradients y capas muertas.
Critical para debugging de arquitecturas con fast weights.*
- `make_adversarial_pgd` (line 1843) - *PGD attack con congelamiento total de pesos (Protocolo de Aislamiento Sináptico).
FIX: Captura correcta de 5 valores de retorno del forward.*
- `evaluate` (line 1910) - *Evaluación optimizada con Gradient Shielding.
FIX: Captura correcta de 5 valores de retorno del forward.*
- `train_epoch` (line 1963) - *Entrenamiento homeostático con inicialización de estados*
- `train_model` (line 2132) - *Training loop principal - FIX v24: Orden correcto de Scheduler y gestión de memoria.*
- `main` (line 2344) - *CLI v24 completo con Orquestador Prefrontal*
- `__post_init__` (line 82)
- `to_dict` (line 88)
- `get_supcon_lambda` (line 91)
- `get_sparsity_lambda` (line 97)
- `get_memory_gb` (line 129)
- `get_gpu_memory_gb` (line 134)
- `log` (line 140)
- `clear_cache` (line 147)
- `check_limit` (line 153)
- `__init__` (line 179)
- `forward` (line 216) - *Input: Diccionario con métricas del estado actual
Output: Diccionario con señales de control escaladas [0,1]*
- `detach_state` (line 256) - *Rompe el grafo computacional para evitar retropropagación infinita entre batches*
- `reset_context` (line 261) - *Resetear contexto al inicio de cada época*
- `__init__` (line 284)
- `_analyze_matrix` (line 290)
- `calculate` (line 336)
- `get_critical_summary` (line 353)
- `__init__` (line 365)
- `save` (line 370)
- `load` (line 398)
- `__init__` (line 455)
- `forward` (line 458)
- `__init__` (line 472)
- `forward` (line 482)
- `__init__` (line 492)
- `forward` (line 502)
- `__init__` (line 510)
- `_maintain_orthogonality` (line 524)
- `forward` (line 529)
- `__init__` (line 555)
- `forward` (line 604) - *FIX: Ahora acepta señales de control del Orquestador para modular
la plasticidad y consolidación en tiempo real.*
- `__init__` (line 682)
- `invalidate_sparse_cache` (line 720)
- `_validate_and_fix_state` (line 723)
- `get_node_importance` (line 742) - *FIX: Método faltante para obtener importancia de nodos*
- `forward` (line 748) - *Forward con señales de control del Orquestador y retorno de ortho deviation*
- `__init__` (line 879)
- `initialize_memories` (line 933) - *Inicialización de memorias semánticas con captura correcta de 5 valores de retorno*
- `_initialize_layer_memory` (line 976)
- `consolidate_semantic_memories` (line 1002)
- `set_epoch` (line 1047)
- `calculate_ortho_loss` (line 1052) - *Calcula loss de ortogonalidad usando el deviation retornado por las capas*
- `calculate_topology_diversity_loss` (line 1060)
- `_init_grid_topology` (line 1085)
- `get_topology` (line 1108)
- `forward` (line 1121) - *Forward con validación, detach explícito, y retorno de ortho deviation*
- `prune_topology` (line 1183) - *Poda topológica con cálculo correcto de quantile*
- `warmup_topo` (line 2180)

#### `nestedtopobrain_v3.py`
**Path:** `nestedtopobrain_v3.py`

**Classs:**
- `Config` (line 29)
- `ResourceMonitor` (line 127)
- `PrefrontalOrchestrator` (line 172) - *Módulo de control ejecutivo que monitoriza el estado de la red y emite señales
dinámicas de activación/inhibición para cada mecanismo neuromodulatorio.
Opera como un sistema de homeostasis topológica y metabólica.
FIX v24: Gestión corregida del grafo computacional recurrente (BPTT).*
- `TopologyMetrics` (line 324)
- `TopologicalHealthSovereignty` (line 333) - *Monitor SVD Completo con criterios neurocientíficos

Referencias:
- Sporns (2016): Human brain networks ~1-3% sparsity
- Bullmore & Sporns (2009): Small-world topology con L_score > 3*
- `CheckpointManager` (line 419) - *Manager robusto v18 con backups y metadata*
- `SupConLoss` (line 510)
- `AsymmetricPredictiveErrorCell` (line 527)
- `LearnableAbsenceGating` (line 547)
- `SymbioticBasisRefinement` (line 565)
- `ContinuumMemoryCell` (line 610)
- `AdaptiveCombinatorialComplexLayer` (line 737)
- `TopoBrainV24` (line 934)

**Functions:**
- `seed_everything` (line 113)
- `get_dataloaders` (line 471)
- `save_topology_visualization` (line 1388) - *Visualización v18 completa*
- `save_node_importance_viz` (line 1432) - *Visualización de importancia de nodos v18*
- `analyze_topology_clustering` (line 1454) - *Clustering espectral v18*
- `analyze_topology_flow` (line 1493) - *Análisis de flujo de información con captura genérica de outputs*
- `visualize_topology_as_graph` (line 1559) - *Grafo v18 con métricas*
- `analyze_topology_evolution` (line 1611) - *Análisis temporal completo v18*
- `comprehensive_topology_analysis` (line 1672) - *Suite completa de análisis v18*
- `run_ablation_study` (line 1695) - *Suite de ablación v18 completa*
- `visualize_memory_evolution` (line 1796) - *Visualiza evolución de memorias semánticas

Crítico para entender si la consolidación hipocampal→cortical está funcionando.
SVD spectrum revela estructura de representaciones aprendidas.*
- `analyze_gradient_flow` (line 1887) - *Análisis detallado del flujo de gradientes

Detecta vanishing/exploding gradients y capas muertas.
Critical para debugging de arquitecturas con fast weights.*
- `make_adversarial_pgd` (line 1954) - *PGD attack con congelamiento total de pesos (Protocolo de Aislamiento Sináptico).
FIX: Captura correcta de 5 valores de retorno del forward.*
- `evaluate` (line 2021) - *Evaluación con plasticidad residual (test-time adaptation)
Biológicamente plausible: el cerebro no se apaga durante percepción*
- `train_epoch` (line 2093) - *Entrenamiento homeostático con inicialización de estados y gestión de densidad*
- `train_model` (line 2279) - *Training loop principal - FIX v24: Orden correcto de Scheduler y gestión de memoria.*
- `main` (line 2491) - *CLI v24 completo con Orquestador Prefrontal*
- `__post_init__` (line 82)
- `to_dict` (line 88)
- `get_supcon_lambda` (line 91)
- `get_sparsity_lambda` (line 97)
- `get_memory_gb` (line 129)
- `get_gpu_memory_gb` (line 134)
- `log` (line 140)
- `clear_cache` (line 147)
- `check_limit` (line 153)
- `__init__` (line 179)
- `forward` (line 216) - *Orquestador v26: Allostasis (Adaptación Predictiva Valiente).

Cambio de Paradigma:
En lugar de entrar en pánico ciego cuando la densidad es baja (<5%),
este sistema evalúa el rendimiento (Loss). Si el cerebro es "delgado"
pero eficiente, se activa el 'Modo Élite' (Alta Plasticidad).
Solo se activa el protocolo de emergencia si hay colapso funcional.*
- `detach_state` (line 312) - *Rompe el grafo computacional para evitar retropropagación infinita entre batches*
- `reset_context` (line 317) - *Resetear contexto al inicio de cada época*
- `__init__` (line 340)
- `_analyze_matrix` (line 346)
- `calculate` (line 392)
- `get_critical_summary` (line 409)
- `__init__` (line 421)
- `save` (line 426)
- `load` (line 454)
- `__init__` (line 511)
- `forward` (line 514)
- `__init__` (line 528)
- `forward` (line 538)
- `__init__` (line 548)
- `forward` (line 558)
- `__init__` (line 566)
- `_maintain_orthogonality` (line 580)
- `forward` (line 585)
- `__init__` (line 611)
- `forward` (line 660) - *FIX: Ahora acepta señales de control del Orquestador para modular
la plasticidad y consolidación en tiempo real.*
- `__init__` (line 738)
- `invalidate_sparse_cache` (line 776)
- `_validate_and_fix_state` (line 779)
- `get_node_importance` (line 798) - *FIX: Método faltante para obtener importancia de nodos*
- `forward` (line 804) - *Forward con señales de control del Orquestador y retorno de ortho deviation*
- `__init__` (line 935)
- `initialize_memories` (line 989) - *Inicialización de memorias semánticas con captura correcta de 5 valores de retorno*
- `_initialize_layer_memory` (line 1032)
- `consolidate_semantic_memories` (line 1058)
- `set_epoch` (line 1103)
- `calculate_ortho_loss` (line 1108) - *Calcula loss de ortogonalidad usando el deviation retornado por las capas*
- `calculate_topology_diversity_loss` (line 1116)
- `_init_grid_topology` (line 1141)
- `get_topology` (line 1164)
- `forward` (line 1177) - *Forward con validación, detach explícito, y retorno de ortho deviation*
- `prune_topology` (line 1239) - *Poda topológica con protocolo de supervivencia garantizado y neurogénesis*
- `warmup_topo` (line 2327)

#### `neurologitos.py`
**Path:** `neurologitos.py`

**Classs:**
- `NeuroLogosConfig` (line 50) - *Configuración CPU-friendly para NeuroLogos*
- `TopoBrainCore` (line 95)
- `PGDAttack` (line 159)
- `MiniUnconscious` (line 187)
- `TopoUnconscious` (line 206)
- `ConsciousCore` (line 229)
- `BioDecoder` (line 241)
- `NeuroLogos` (line 284)
- `CIFARCaptions` (line 318)
- `AblationMatrix` (line 356)
- `ScientificAnalyzer` (line 393)

**Functions:**
- `seed_everything` (line 29) - *Control total de reproducibilidad*
- `compute_effect_size` (line 38) - *Cohen's d con corrección de sesgo*
- `train_epoch_cv` (line 451)
- `evaluate_cv` (line 507)
- `train_with_cv` (line 524)
- `run_scientific_ablation` (line 579)
- `to_dict` (line 81)
- `component_signature` (line 84)
- `__init__` (line 96)
- `_init_grid` (line 121)
- `forward` (line 128)
- `get_metrics` (line 152)
- `__init__` (line 160)
- `attack` (line 165)
- `__init__` (line 188)
- `forward` (line 200)
- `__init__` (line 207)
- `forward` (line 221)
- `get_metrics` (line 225)
- `__init__` (line 230)
- `forward` (line 235)
- `__init__` (line 242)
- `forward` (line 253)
- `_get_init_state` (line 278)
- `__init__` (line 285)
- `forward` (line 309)
- `get_metrics` (line 314)
- `__init__` (line 319)
- `__len__` (line 344)
- `__getitem__` (line 347)
- `level1_isolated` (line 360)
- `level2_pairs` (line 369)
- `level3_full` (line 377)
- `level4_inverse` (line 381)
- `get_complete_matrix` (line 389)
- `compute_statistics` (line 395)
- `ttest_vs_baseline` (line 413)
- `detect_synergy` (line 419)
- `rank_criticality` (line 432)
- `model_fn` (line 471)
- `crit_fn` (line 474)

#### `neurologos.py`
**Path:** `neurologos.py`

**Classs:**
- `MiniUnconscious` (line 13) - *Versión rápida CPU: 512-dim output directo*
- `NestedUnconscious` (line 29) - *Versión topológica GPU: mantiene nested structure*
- `LiquidNeuron` (line 81)
- `ConsciousCore` (line 96)
- `BioDecoder` (line 116)
- `NeuroLogos` (line 167)
- `LifeCycle` (line 200)
- `CIFARCaptions` (line 220)

**Functions:**
- `train_logos` (line 261)
- `__init__` (line 15)
- `forward` (line 26)
- `__init__` (line 31)
- `forward` (line 57)
- `__init__` (line 82)
- `forward` (line 88)
- `__init__` (line 97)
- `forward` (line 103)
- `__init__` (line 117)
- `forward` (line 132)
- `_get_init_state` (line 159)
- `__init__` (line 168)
- `forward` (line 182)
- `measure_richness` (line 193)
- `__init__` (line 201)
- `get_plasticity` (line 205)
- `__init__` (line 221)
- `__len__` (line 245)
- `__getitem__` (line 248)

#### `neurologos_V1.py`
**Path:** `neurologos_V1.py`

**Classs:**
- `MiniUnconscious` (line 15)
- `LiquidNeuron` (line 33)
- `ConsciousCore` (line 48)
- `BioDecoder` (line 73)
- `NeuroLogos` (line 140)
- `LifeCycle` (line 170)
- `CIFARCaptions` (line 189)

**Functions:**
- `train_logos` (line 237)
- `__init__` (line 16)
- `forward` (line 27)
- `__init__` (line 34)
- `forward` (line 40)
- `__init__` (line 49)
- `forward` (line 55)
- `__init__` (line 74)
- `forward` (line 90) - *Modo entrenamiento: captions != None
Modo generación: captions == None*
- `_get_init_state` (line 131)
- `__init__` (line 141)
- `forward` (line 150)
- `measure_richness` (line 163)
- `__init__` (line 171)
- `get_plasticity` (line 175)
- `__init__` (line 190)
- `__len__` (line 216)
- `__getitem__` (line 219)

#### `neurologos_cpu_v7.py`
**Path:** `neurologos_cpu_v7.py`

**Classs:**
- `MicroConfig` (line 33)
- `MicroContinuumCell` (line 96)
- `MicroSymbioticBasis` (line 120)
- `MicroTopology` (line 135)
- `MicroSupConLoss` (line 154)
- `MicroTopoBrain` (line 170)

**Functions:**
- `setup_device` (line 65)
- `seed_everything` (line 68)
- `get_dataset` (line 73)
- `micro_pgd_attack` (line 235)
- `train_with_cv` (line 276)
- `run_ablation_study` (line 328)
- `__init__` (line 97)
- `forward` (line 106)
- `__init__` (line 121)
- `forward` (line 128)
- `__init__` (line 136)
- `get_adjacency` (line 149)
- `__init__` (line 155)
- `forward` (line 158)
- `__init__` (line 171)
- `_init_weights` (line 195)
- `count_parameters` (line 198)
- `forward` (line 199)

#### `neurologos_cpu_v8.py`
**Path:** `neurologos_cpu_v8.py`

**Classs:**
- `MicroConfig` (line 33)
- `MicroContinuumCell` (line 92) - *Memoria continua con aprendizaje rápido/lento - VERSIÓN COMPATIBLE CON PGD*
- `MicroSymbioticBasis` (line 124) - *Base simbiótica para refinamiento adversarial*
- `MicroTopology` (line 146) - *Topología de grid 2D con conexiones von Neumann*
- `MicroSupConLoss` (line 169) - *Supervised Contrastive Loss*
- `MicroTopoBrain` (line 195) - *Arquitectura TopoBrain Modular para Ablación*

**Functions:**
- `seed_everything` (line 65)
- `get_dataset` (line 72)
- `micro_pgd_attack` (line 308) - *PGD Attack - Versión ultra-simple que siempre funciona*
- `generate_ablation_matrix` (line 344) - *Genera matriz de ablación de 3 niveles:
- Nivel 1: Baseline + componentes individuales
- Nivel 2: Pares sinérgicos
- Nivel 3: Sistema completo (para ablación inversa)*
- `train_with_cv` (line 393) - *Entrenamiento con cross-validation*
- `run_ablation_study` (line 483) - *Ejecuta el estudio de ablación completo*
- `__init__` (line 94)
- `forward` (line 105)
- `__init__` (line 126)
- `forward` (line 134)
- `__init__` (line 148)
- `get_adjacency` (line 164)
- `__init__` (line 171)
- `forward` (line 176)
- `__init__` (line 197)
- `_init_weights` (line 240)
- `count_parameters` (line 245)
- `forward` (line 248)

#### `neurologos_cpu_v9.py.py`
**Path:** `neurologos_cpu_v9.py.py`

**Classs:**
- `NeuroLogosConfig` (line 45) - *Configuración ablacionable para NeuroLogos*
- `TopoBrainCore` (line 90)
- `PGDAttack` (line 153)
- `MiniUnconscious` (line 180)
- `TopoUnconscious` (line 198)
- `ConsciousCore` (line 220)
- `BioDecoder` (line 231)
- `NeuroLogos` (line 276)
- `CIFARCaptions` (line 315)
- `AblationMatrix` (line 355) - *Matriz de ablación para 3 componentes (G, S, A)*
- `ScientificAnalyzer` (line 392)

**Functions:**
- `seed_everything` (line 25) - *Control total de reproducibilidad*
- `compute_effect_size` (line 34) - *Cohen's d con corrección de sesgo*
- `train_epoch_cv` (line 452)
- `evaluate_cv` (line 507)
- `train_with_cv` (line 523)
- `run_scientific_ablation` (line 580)
- `to_dict` (line 76)
- `component_signature` (line 79)
- `__init__` (line 91)
- `_init_grid` (line 116)
- `forward` (line 123)
- `get_metrics` (line 147)
- `__init__` (line 154)
- `attack` (line 159)
- `__init__` (line 181)
- `forward` (line 193)
- `__init__` (line 199)
- `forward` (line 213)
- `get_metrics` (line 217)
- `__init__` (line 221)
- `forward` (line 226)
- `__init__` (line 232)
- `forward` (line 243)
- `_get_init_state` (line 268)
- `__init__` (line 277)
- `forward` (line 304)
- `get_metrics` (line 309)
- `__init__` (line 316)
- `__len__` (line 341)
- `__getitem__` (line 344)
- `level1_isolated` (line 360)
- `level2_pairs` (line 369)
- `level3_full` (line 377)
- `level4_inverse` (line 381)
- `get_complete_matrix` (line 389)
- `compute_statistics` (line 394)
- `ttest_vs_baseline` (line 412)
- `detect_synergy` (line 418)
- `rank_criticality` (line 431)
- `model_fn` (line 472)
- `crit_fn` (line 475)

#### `neurologos_entropico.py`
**Path:** `neurologos_entropico.py`

**Classs:**
- `Config` (line 36)
- `DataEnvironment` (line 60)
- `HomeostaticRegulator` (line 94)
- `PhysioNeuron` (line 120)
- `RegulableSymbiotic` (line 160)
- `RegulableTopology` (line 179)
- `MicroTopoBrain` (line 201)

**Functions:**
- `seed_everything` (line 50)
- `train_nonstationary` (line 263)
- `generate_ablation_matrix` (line 320)
- `run_ablation_study` (line 352)
- `__init__` (line 61)
- `get_batch` (line 71)
- `get_full` (line 85)
- `get_w2` (line 88)
- `__init__` (line 95)
- `forward` (line 105)
- `__init__` (line 121)
- `forward` (line 132)
- `__init__` (line 161)
- `forward` (line 168)
- `__init__` (line 180)
- `get_adjacency` (line 193)
- `__init__` (line 202)
- `count_parameters` (line 221)
- `forward` (line 224)

#### `neurologos_fullhomesotatico_cpu_qw.py`
**Path:** `neurologos_fullhomesotatico_cpu_qw.py`

**Classs:**
- `MicroConfig` (line 36)
- `GlobalHomeostaticOrchestrator` (line 93)
- `MicroContinuumCell` (line 137)
- `MicroSymbioticBasis` (line 161)
- `MicroTopology` (line 183)
- `MicroSupConLoss` (line 203)
- `MicroTopoBrain` (line 227)

**Functions:**
- `seed_everything` (line 65)
- `get_dataset` (line 73)
- `micro_pgd_attack` (line 363)
- `generate_ablation_matrix` (line 387)
- `train_with_cv` (line 407)
- `run_ablation_study` (line 472)
- `__init__` (line 94)
- `forward` (line 104)
- `__init__` (line 138)
- `forward` (line 148)
- `__init__` (line 162)
- `forward` (line 170)
- `__init__` (line 184)
- `get_adjacency` (line 197)
- `__init__` (line 204)
- `forward` (line 209)
- `__init__` (line 228)
- `_init_weights` (line 267)
- `count_parameters` (line 272)
- `forward` (line 275)

#### `neurologos_fullhomestatico_cpu_qw2.py`
**Path:** `neurologos_fullhomestatico_cpu_qw2.py`

**Classs:**
- `Config` (line 34)
- `DataEnvironment` (line 60)
- `HomeostaticRegulator` (line 95)
- `PhysioNeuron` (line 122)
- `RegulableSymbiotic` (line 164)
- `RegulableTopology` (line 184)
- `MicroTopoBrain` (line 207)

**Functions:**
- `seed_everything` (line 49)
- `train_nonstationary` (line 269)
- `generate_ablation_matrix` (line 331)
- `run_ablation_study` (line 364)
- `__init__` (line 61)
- `get_batch` (line 71)
- `get_full` (line 85)
- `get_w2` (line 88)
- `__init__` (line 96)
- `forward` (line 106)
- `__init__` (line 123)
- `forward` (line 134)
- `__init__` (line 165)
- `forward` (line 172)
- `__init__` (line 185)
- `get_adjacency` (line 198)
- `__init__` (line 208)
- `count_parameters` (line 227)
- `forward` (line 230)

#### `neurologos_gpu_v1.py`
**Path:** `neurologos_gpu_v1.py`

**Classs:**
- `NestedUnconscious` (line 16)
- `LiquidNeuron` (line 76)
- `ConsciousCore` (line 91)
- `BioDecoder` (line 116)
- `NeuroLogos` (line 183)
- `LifeCycle` (line 213)
- `CIFARCaptions` (line 233)

**Functions:**
- `train_logos` (line 282)
- `__init__` (line 17)
- `forward` (line 46)
- `__init__` (line 77)
- `forward` (line 83)
- `__init__` (line 92)
- `forward` (line 98)
- `__init__` (line 117)
- `forward` (line 133) - *Modo entrenamiento: captions != None
Modo generación: captions == None*
- `_get_init_state` (line 174)
- `__init__` (line 184)
- `forward` (line 193)
- `measure_richness` (line 206)
- `__init__` (line 214)
- `get_plasticity` (line 219)
- `__init__` (line 234)
- `__len__` (line 261)
- `__getitem__` (line 264)

#### `neurologos_homeostatico_cpu_cl.py`
**Path:** `neurologos_homeostatico_cpu_cl.py`

**Classs:**
- `HomeoConfig` (line 38)
- `HomeostaticRegulator` (line 98) - *Sistema de auto-regulación inspirado en fisiología MEJORADO.

INNOVACIÓN: Sensores multi-escala que distinguen:
- Estrés natural (varianza, complejidad)
- Estrés adversarial (gradiente anómalo, suavidad)
- Fatiga metabólica (norma de pesos)

Inputs enriquecidos: [Estrés_Natural, Estrés_Adversarial, Excitación, Fatiga, Gradiente_Norma]
Outputs: [Metabolismo, Sensibilidad, Gate]*
- `HomeoContinuumCell` (line 218) - *Memoria continua con regulación homeostática*
- `HomeoSymbioticBasis` (line 281) - *Base simbiótica con regulación homeostática*
- `HomeoTopology` (line 329) - *Topología con plasticidad homeostática*
- `HomeoSupConLoss` (line 380) - *Supervised Contrastive Loss*
- `HomeoTopoBrain` (line 410) - *TopoBrain con regulación homeostática integrada*

**Functions:**
- `seed_everything` (line 71)
- `get_dataset` (line 78)
- `pgd_attack` (line 524) - *PGD Attack simplificado*
- `generate_homeostatic_ablation` (line 555) - *Genera matriz enfocada en homeostasis con sensores mejorados*
- `train_with_cv` (line 614) - *Entrenamiento con cross-validation*
- `run_homeostatic_ablation` (line 697) - *Ejecuta el estudio de ablación homeostático*
- `__init__` (line 110)
- `forward` (line 129)
- `__init__` (line 220)
- `forward` (line 241)
- `__init__` (line 283)
- `forward` (line 299)
- `__init__` (line 331)
- `get_adjacency` (line 354) - *Genera adyacencia con regulación homeostática opcional*
- `__init__` (line 382)
- `forward` (line 387)
- `__init__` (line 412)
- `_init_weights` (line 456)
- `count_parameters` (line 461)
- `forward` (line 464)

#### `neurologos_homeostatico_cpu_cl2.py`
**Path:** `neurologos_homeostatico_cpu_cl2.py`

**Classs:**
- `TransContextConfig` (line 50)
- `NonStationaryEnvironment` (line 89) - *Entorno que cambia de distribución como en PhysioChimera.
Permite medir RETENCIÓN y ADAPTACIÓN, no solo robustez adversarial.*
- `EnhancedHomeostaticRegulator` (line 152) - *Regulador homeostático con logging y sensores mejorados.
Incluye diagnóstico para análisis post-hoc.*
- `TransContextContinuumCell` (line 254) - *Memoria continua con homeostasis*
- `TransContextTopoBrain` (line 320) - *TopoBrain diseñado para entornos no estacionarios.
Focus: Retención y Adaptación, no solo robustez adversarial.*

**Functions:**
- `seed_everything` (line 78)
- `light_pgd_attack` (line 424) - *PGD ligero para no dominar el entrenamiento*
- `train_trans_contextual` (line 455) - *Entrenamiento en entorno no estacionario.
Mide: Retención, Adaptación, Robustez.*
- `generate_selective_ablation` (line 556) - *Ablación selectiva basada en resultados v5.2:
- Eliminar MGF (nunca mejora con homeostasis)
- Focus en Continuum + Homeostasis
- Agregar regulación jerárquica*
- `run_trans_contextual_study` (line 598) - *Ejecuta el estudio trans-contextual completo*
- `__init__` (line 94)
- `get_batch` (line 115) - *Retorna batch según la fase del entrenamiento*
- `get_phase` (line 135) - *Determina la fase según el epoch actual*
- `__init__` (line 157)
- `forward` (line 183)
- `__init__` (line 256)
- `forward` (line 277)
- `__init__` (line 325)
- `_init_weights` (line 356)
- `count_parameters` (line 361)
- `forward` (line 364)
- `get_homeostasis_metrics` (line 401) - *Extrae métricas de homeostasis para logging*

#### `neurologos_homeostatico_cpu_ki.py`
**Path:** `neurologos_homeostatico_cpu_ki.py`

**Classs:**
- `MicroConfig` (line 37)
- `HomeostaticCore` (line 109) - *Cerebro interno que monitoriza el estado fisiológico de la red
y emite señales de control adaptativas.
FIXES:
1. Maneja batch_size=1 evitando warning de var()
2. Convierte loss_val correctamente a tensor
3. Asegura device placement consistente*
- `MicroContinuumCell` (line 175) - *Versión homeostática con regulación de metabolismo*
- `MicroSymbioticBasis` (line 240) - *Base simbiótica - VERSION FINAL
FIXES:
1. Remover batch norm que causaba colapso
2. Aumentar ruido interno para no ser demasiado robusto
3. Añadir regularización de varianza mínima
4. Regulación de entropía más fuerte para evitar picos*
- `MicroTopology` (line 327) - *Versión homeostática con plasticidad adaptativa*
- `MicroSupConLoss` (line 367) - *Supervised Contrastive Loss - Mantiene estabilidad con homeostasis*
- `MicroTopoBrain` (line 397)

**Functions:**
- `seed_everything` (line 74)
- `get_dataset` (line 85) - *Genera dataset sintético para el estudio de ablación.
Normaliza features en rango [0,1] para estabilidad homeostática.*
- `micro_pgd_attack` (line 521) - *PGD Attack - Versión ultra-simple que siempre funciona
FIX: No pasar loss_val al modelo durante ataque para evitar leakage
El ataque no debe tener acceso al estado interno de entrenamiento*
- `generate_ablation_matrix` (line 556) - *Genera matriz de ablación de 3 niveles con configuración aislada por experimento
FIX: Asegurar que cada experimento tenga configuración independiente y limpia*
- `train_with_cv` (line 595) - *Entrenamiento con cross-validation
FIX CRÍTICO: Métrica W2 debe usar MODELO FRESH copiado, no el mismo modelo*
- `run_ablation_study` (line 777) - *Ejecuta estudio con validación de integridad de resultados*
- `__init__` (line 118)
- `forward` (line 137)
- `__init__` (line 177)
- `forward` (line 198)
- `__init__` (line 249)
- `forward` (line 268)
- `__init__` (line 329)
- `_create_grid_mask` (line 344)
- `get_adjacency` (line 355)
- `__init__` (line 369)
- `forward` (line 374)
- `__init__` (line 398)
- `_init_weights` (line 440)
- `count_parameters` (line 445)
- `forward` (line 448)

#### `neurologos_homestotico_cpu_qw.py`
**Path:** `neurologos_homestotico_cpu_qw.py`

**Classs:**
- `MicroConfig` (line 35)
- `HomeostaticRegulatorMini` (line 92)
- `MicroPhysioNeuron` (line 117)
- `MicroContinuumCell` (line 162)
- `MicroSymbioticBasis` (line 187)
- `MicroTopology` (line 207)
- `MicroSupConLoss` (line 228)
- `MicroTopoBrain` (line 252)

**Functions:**
- `seed_everything` (line 64)
- `get_dataset` (line 72)
- `micro_pgd_attack` (line 365)
- `generate_ablation_matrix` (line 389)
- `train_with_cv` (line 429)
- `run_ablation_study` (line 496)
- `__init__` (line 93)
- `forward` (line 104)
- `__init__` (line 118)
- `forward` (line 129)
- `__init__` (line 163)
- `forward` (line 173)
- `__init__` (line 188)
- `forward` (line 196)
- `__init__` (line 208)
- `get_adjacency` (line 222)
- `__init__` (line 229)
- `forward` (line 234)
- `__init__` (line 253)
- `_init_weights` (line 301)
- `count_parameters` (line 306)
- `forward` (line 309)

#### `neurologos_tricameral_exodia.py`
**Path:** `neurologos_tricameral_exodia.py`

**Classs:**
- `HierarchicalEpisodicMemory` (line 332)
- `NeurocognitiveSystem` (line 563)
- `LanguageMetrics` (line 760) - *Métricas de calidad de generación*
- `LinguisticFeedbackLoop` (line 834)
- `LanguageMetrics` (line 951)
- `CausalReasoningEngine` (line 994)
- `LanguageMetrics` (line 1073)
- `StableLiquidNeuron` (line 1120)
- `TriangulatedMedicalSystem` (line 1259)
- `LeftHemisphere` (line 1410)
- `AudioEncoder` (line 1719) - *Encoder de audio usando Conv + Transformer*
- `RightHemisphereTricameral` (line 1769) - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 1853)
- `EnhancedDiagnosticsTricameral` (line 2006)
- `NeuroLogosTricameral` (line 2311) - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 2346) - *Dataset que carga imagen, audio del caption y texto desde Kaggle*

**Functions:**
- `preprocess_and_cache_spectrograms` (line 47) - *Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt
Esto elimina el cuello de botella de I/O durante entrenamiento*
- `setup_flickr8k_with_audio` (line 120) - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 308) - *Construye vocabulario desde el archivo de captions*
- `compute_alignment_loss` (line 2462) - *FIX: Pérdida auxiliar para alineación temprana de canales multimodales
Solo activa en épocas iniciales (epoch < 6)*
- `compute_tricameral_loss` (line 2490) - *FIX: Pérdida con término explícito de coherencia multimodal
Penaliza la falta de sincronización entre canales*
- `train_tricameral` (line 2563)
- `__init__` (line 333)
- `compute_surprise` (line 359)
- `calculate_importance` (line 369)
- `_calculate_novelty` (line 381)
- `store_episode` (line 402)
- `_update_unified_buffer` (line 440)
- `add` (line 452)
- `apply_forgetting_curve` (line 455)
- `_purge_low_score_memories` (line 471)
- `sample` (line 497)
- `_sample_from_buffer` (line 527)
- `get_total_size` (line 555)
- `__init__` (line 564)
- `assess_reasoning_state` (line 584) - *Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)*
- `assess_cognitive_state` (line 628) - *Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)*
- `apply_cognitive_intervention` (line 674) - *Aplica intervenciones basadas en estado lingüístico y de razonamiento*
- `sentence_bleu` (line 764) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 798) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 807) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 820) - *Jaccard similarity entre palabras*
- `__init__` (line 835)
- `_get_ngrams_cached` (line 849) - *FIX: Método estático con lru_cache para n-gramas*
- `compute_linguistic_reward` (line 858)
- `compute_cider` (line 897) - *FIX: Uso correcto del cache estático*
- `compute_spice` (line 911)
- `get_cache_stats` (line 923) - *FIX: Estadísticas de cache actualizadas*
- `sentence_bleu` (line 953)
- `token_accuracy` (line 976)
- `word_overlap` (line 986)
- `__init__` (line 995)
- `reason_causally` (line 1022)
- `_predict_interventions` (line 1036)
- `update_knowledge_graph` (line 1053)
- `query_causal_chain` (line 1059)
- `sentence_bleu` (line 1075)
- `token_accuracy` (line 1098)
- `word_overlap` (line 1108)
- `__init__` (line 1121)
- `forward` (line 1163)
- `_calculate_homeostasis_metric` (line 1179) - *Calcula métrica de homeostasis basada en la estabilidad del output*
- `hebbian_update` (line 1188)
- `update_physiology_advanced` (line 1226)
- `__init__` (line 1260)
- `triangulate_signals` (line 1267)
- `count_convergent_signals` (line 1278)
- `diagnose_with_triangulation` (line 1281)
- `apply_triangulated_intervention` (line 1326)
- `_reset_liquid_neuron` (line 1395) - *Reset completo de una neurona líquida*
- `__init__` (line 1411)
- `forward` (line 1493)
- `_apply_chain_of_thought` (line 1540)
- `_greedy_decode` (line 1580)
- `_apply_multi_token_prediction` (line 1641)
- `_apply_structural_attention` (line 1683)
- `_get_init_state` (line 1704)
- `__init__` (line 1722)
- `forward` (line 1756)
- `__init__` (line 1772)
- `forward` (line 1812) - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre, audio_post, audio_pre: Para Hebbian*
- `__init__` (line 1854)
- `forward` (line 1902)
- `update_channel_fatigue` (line 1963)
- `adjust_gates_by_fatigue` (line 1985)
- `__init__` (line 2007)
- `_get_cached_norm` (line 2030) - *Cache de normalización con limpieza periódica*
- `measure_callosal_flow` (line 2048) - *FIX: Medición de coherencia multimodal real con atención a diversidad
Incluye métricas de sincronización entre canales*
- `__init__` (line 2314)
- `forward` (line 2320)
- `__init__` (line 2349)
- `__len__` (line 2406)
- `__getitem__` (line 2409)
- `evaluate_reasoning_quality` (line 2101)
- `calculate_synergy` (line 2138)
- `calculate_health` (line 2149)
- `update` (line 2158)
- `get_recent_avg` (line 2175)
- `visualize_fatigue_distribution` (line 2191)
- `visualize_reasoning_metrics` (line 2215)
- `report` (line 2227)

#### `neurologos_v3.py`
**Path:** `neurologos_v3.py`

*No symbols extracted*

#### `neurologos_v4.py`
**Path:** `neurologos_v4.py`

**Classs:**
- `MiniUnconscious` (line 13) - *Versión rápida CPU: 512-dim output directo*
- `NestedUnconscious` (line 29) - *Versión topológica GPU: mantiene nested structure*
- `LiquidNeuron` (line 81)
- `ConsciousCore` (line 164)
- `BioDecoder` (line 185)
- `NeuroLogos` (line 236)
- `LifeCycle` (line 271)
- `CIFARCaptions` (line 291)

**Functions:**
- `train_logos` (line 358)
- `__init__` (line 15)
- `forward` (line 26)
- `__init__` (line 31)
- `forward` (line 57)
- `__init__` (line 82)
- `forward` (line 103)
- `consolidate_svd` (line 131) - *Consolidación espectral de pesos rápidos mediante SVD.
Repara inestabilidades numéricas y cristaliza conocimiento consolidado.
Retorna True si se realizó una consolidación activa.*
- `__init__` (line 165)
- `forward` (line 172)
- `__init__` (line 186)
- `forward` (line 201)
- `_get_init_state` (line 228)
- `__init__` (line 237)
- `forward` (line 251)
- `measure_richness` (line 264)
- `__init__` (line 272)
- `get_plasticity` (line 276)
- `__init__` (line 292)
- `__len__` (line 344)
- `__getitem__` (line 346)

#### `neurologos_v5.py`
**Path:** `neurologos_v5.py`

**Classs:**
- `TopologicalCompressor` (line 75)
- `MiniUnconscious` (line 96) - *Versión rápida CPU: 512-dim output directo - con procesamiento jerárquico inspirado en vía ventral*
- `NestedUnconscious` (line 119)
- `LiquidNeuron` (line 169)
- `ConsciousCore` (line 260)
- `BioDecoder` (line 332)
- `NeuroLogos` (line 404)
- `LifeCycle` (line 435)
- `CIFARCaptions` (line 457)

**Functions:**
- `top_k_top_p_filtering` (line 11) - *Filtra logits con Top-K o Top-P (Nucleus) Sampling.*
- `measure_spatial_richness` (line 32) - *Calcula la riqueza representacional de un tensor de activación.
Utiliza Entropía de Shannon (por canal) y Entropía de Von Neumann (estructural).
FIX: Evita el UserWarning de std() con batch=1 y mejora estabilidad numérica.*
- `train_logos` (line 502)
- `__init__` (line 76)
- `forward` (line 85)
- `__init__` (line 98)
- `forward` (line 113)
- `__init__` (line 120)
- `forward` (line 143)
- `__init__` (line 170)
- `forward` (line 191)
- `consolidate_svd` (line 228)
- `__init__` (line 261)
- `forward` (line 278)
- `get_liquid_module` (line 319) - *Retorna el LiquidNeuron activo para la consolidación externa.*
- `__init__` (line 334)
- `forward` (line 349)
- `_get_init_state` (line 395)
- `__init__` (line 405)
- `forward` (line 420)
- `measure_richness` (line 428)
- `__init__` (line 436)
- `get_plasticity` (line 440)
- `__init__` (line 458)
- `__len__` (line 483)
- `__getitem__` (line 486)

#### `neurologos_v6.py`
**Path:** `neurologos_v6.py`

**Classs:**
- `Config` (line 18)
- `MetricsCollector` (line 55)
- `DataEnvironment` (line 90)
- `MetaLearner` (line 127)
- `ComponentRegulator` (line 154)
- `HomeostaticRegulator` (line 211)
- `MetaHomeostaticEngine` (line 237)
- `PhysioNeuron` (line 320)
- `SymbioticDual` (line 376)
- `MicroTopoBrain` (line 403)
- `ConfigurableTrainer` (line 473)

**Functions:**
- `seed_everything` (line 45)
- `generate_ablation_matrix_4levels` (line 603)
- `run_ablation_study` (line 645)
- `__init__` (line 56)
- `_setup_logger` (line 62)
- `log_batch` (line 69)
- `save` (line 79)
- `__init__` (line 91)
- `inject_concept_drift` (line 100)
- `get_batch` (line 103)
- `get_full` (line 118)
- `get_w2` (line 121)
- `__init__` (line 128)
- `forward` (line 136)
- `update` (line 142)
- `__init__` (line 155)
- `forward` (line 172)
- `__init__` (line 212)
- `forward` (line 222)
- `__init__` (line 238)
- `forward` (line 251)
- `get_component_health` (line 290)
- `update_with_momentum` (line 298)
- `__init__` (line 321)
- `forward` (line 334)
- `__init__` (line 377)
- `forward` (line 385)
- `__init__` (line 404)
- `count_parameters` (line 427)
- `forward` (line 430)
- `__init__` (line 474)
- `train` (line 479)
- `evaluate` (line 554)

#### `neurologosv5.2.py`
**Path:** `neurologosv5.2.py`

**Classs:**
- `MicroConfig` (line 33)
- `HomeostaticRegulator` (line 85)
- `PhysioNeuron` (line 109)
- `MicroContinuumCell` (line 151)
- `MicroSymbioticBasis` (line 176)
- `MicroTopology` (line 196)
- `MicroSupConLoss` (line 216)
- `MicroTopoBrain` (line 240)

**Functions:**
- `seed_everything` (line 57)
- `get_dataset` (line 65)
- `micro_pgd_attack` (line 340)
- `generate_ablation_matrix` (line 364)
- `train_with_cv` (line 393)
- `run_ablation_study` (line 457)
- `__init__` (line 86)
- `forward` (line 96)
- `__init__` (line 110)
- `forward` (line 121)
- `__init__` (line 152)
- `forward` (line 162)
- `__init__` (line 177)
- `forward` (line 185)
- `__init__` (line 197)
- `get_adjacency` (line 210)
- `__init__` (line 217)
- `forward` (line 222)
- `__init__` (line 241)
- `_init_weights` (line 278)
- `count_parameters` (line 283)
- `forward` (line 286)

#### `neurosoberano.py`
**Path:** `neurosoberano.py`

**Classs:**
- `ExperimentConfig` (line 28)
- `BasicBlock` (line 39)
- `WideResNetBaseline` (line 60) - *Wide-ResNet simplificado (depth=16, width=2)
Parámetros similares a tu modelo (~1-2M)*
- `FastLiquidNeuron` (line 96) - *Neurona líquida simplificada (sin SVD consolidation para POC)*
- `MinimalNeuroSovereign` (line 125) - *Versión mínima de tu arquitectura para POC*
- `Experiment` (line 233)

**Functions:**
- `train_epoch` (line 177)
- `evaluate` (line 215)
- `main` (line 435)
- `__init__` (line 40)
- `forward` (line 54)
- `__init__` (line 65)
- `_make_layer` (line 79)
- `forward` (line 85)
- `__init__` (line 98)
- `forward` (line 107)
- `__init__` (line 127)
- `_make_layer` (line 146)
- `forward` (line 152)
- `update_plasticity` (line 163) - *Plasticity schedule simplificado*
- `__init__` (line 234)
- `_get_data` (line 245)
- `run_baseline` (line 269)
- `run_neurosovereign` (line 316)
- `compare` (line 369)
- `plot_comparison` (line 404)

#### `neurosoberano_bicameral_opt.py`
**Path:** `neurosoberano_bicameral_opt.py`

**Classs:**
- `HomeostaticRegulator` (line 39)
- `LiquidNeuron` (line 66)
- `RightHemisphere` (line 193)
- `LeftHemisphere` (line 214)
- `CorpusCallosum` (line 313)
- `NeuroLogosBicameral` (line 331)
- `NeuralDiagnostics` (line 361)
- `Flickr8kDataset` (line 436)
- `LifeCycle` (line 498)

**Functions:**
- `build_vocab_flickr` (line 475)
- `train_bicameral` (line 513)
- `__init__` (line 40)
- `forward` (line 50)
- `__init__` (line 67)
- `forward` (line 99)
- `apply_svd_consolidation` (line 169)
- `__init__` (line 194)
- `forward` (line 205)
- `__init__` (line 215)
- `forward` (line 237)
- `_get_init_state` (line 293)
- `_top_p_filtering` (line 298)
- `__init__` (line 314)
- `forward` (line 322)
- `__init__` (line 332)
- `forward` (line 338)
- `__init__` (line 362)
- `measure_callosal_flow` (line 375)
- `measure_vocab_diversity` (line 382)
- `update` (line 386)
- `get_recent_avg` (line 391)
- `report` (line 396)
- `__init__` (line 437)
- `__len__` (line 455)
- `__getitem__` (line 458)
- `__init__` (line 499)
- `get_plasticity` (line 502)

#### `neurosoberano_v2.py`
**Path:** `neurosoberano_v2.py`

*No symbols extracted*

#### `neurosoberano_v3.py`
**Path:** `neurosoberano_v3.py`

*No symbols extracted*

#### `neurosoberano_v4.py`
**Path:** `neurosoberano_v4.py`

*No symbols extracted*

#### `neurosovereign.py`
**Path:** `neurosovereign.py`

**Classs:**
- `SovereignConfig` (line 16)
- `BasicBlock` (line 69)
- `NetworkBlock` (line 96)
- `LiquidCortex` (line 111) - *Capa densa con Fast Weights Hebbianos y Homeostasis.
Reemplaza a la capa lineal aburrida de las CNNs normales.*
- `NeuroSovereignV1` (line 165)

**Functions:**
- `seed_everything` (line 44)
- `mixup_data` (line 51) - *Returns mixed inputs, pairs of targets, and lambda*
- `mixup_criterion` (line 63)
- `get_optimized_dataloaders` (line 214)
- `train_sovereign` (line 239)
- `__init__` (line 70)
- `forward` (line 85)
- `__init__` (line 97)
- `_make_layer` (line 100)
- `forward` (line 105)
- `__init__` (line 116)
- `forward` (line 133)
- `__init__` (line 166)
- `forward` (line 196)

#### `ohm.py`
**Path:** `ohm.py`

**Classs:**
- `MotorHomeostaticContext` (line 34) - *Contexto para un motor homeostático*
- `PTSymmetricMotor` (line 64) - *Motor para controlar parámetros PT-similares*
- `TopologicalMotor` (line 100) - *Motor para controlar conectividad y topología*
- `EnergyHomeostaticMotor` (line 127) - *Motor para controlar eficiencia energética*
- `ConsciousnessMotor` (line 157) - *Motor para controlar métricas de conciencia (Φₑ)*
- `DualSystemMotor` (line 185) - *Motor para controlar balance inconsciente/consciente*
- `AdaptiveLearningMotor` (line 213) - *Motor para adaptar algoritmos de aprendizaje*
- `ModularActivationMotor` (line 241) - *Motor para activar/desactivar módulos según contexto*
- `OmniBrainCoordinator` (line 291) - *Coordinador central que gestiona todos los motores homeostáticos*
- `OmniBrainModule` (line 438) - *Módulo base para todos los componentes del Omni Brain*
- `PTSymmetricLayer` (line 453) - *Capa con activación PT-simétrica regulada*
- `TopologicalLayer` (line 486) - *Capa con conectividad topológica regulada*
- `DualMindModule` (line 543) - *Módulo de procesamiento dual (inconsciente/consciente)*
- `ConsciousnessModule` (line 597) - *Módulo de métricas de conciencia y integración*
- `HomeostaticEngine` (line 658) - *Motor homeostasis reutilizable de Síntesis v8.2*
- `OmniBrain` (line 689) - *El pokemon legendario que combina todas las ideas*

**Functions:**
- `train_omni_brain` (line 863) - *Pipeline de entrenamiento para el Omni Brain*
- `update` (line 47) - *Actualiza el estado del motor homeostático*
- `__init__` (line 66)
- `regulate_parameters` (line 78) - *Regula parámetros para mantener PT-simetría*
- `__init__` (line 102)
- `regulate_connectivity` (line 112) - *Regula conectividad para mantener estructura óptima*
- `__init__` (line 129)
- `regulate_energy` (line 139) - *Regula parámetros para eficiencia energética*
- `__init__` (line 159)
- `regulate_consciousness` (line 168) - *Regula parámetros para control de conciencia*
- `__init__` (line 187)
- `regulate_dual_systems` (line 197) - *Regula balance entre sistemas inconsciente y consciente*
- `__init__` (line 215)
- `regulate_learning` (line 224) - *Regula parámetros de aprendizaje*
- `__init__` (line 243)
- `regulate_modules` (line 259) - *Regula qué módulos están activos*
- `__init__` (line 294)
- `_initialize_motors` (line 300) - *Inicializa todos los motores homeostáticos*
- `sense_environment` (line 312) - *Sensa el estado actual del entorno*
- `measure_network_state` (line 326) - *Mide el estado actual de la red*
- `coordinate_all_motors` (line 363) - *Coordina todos los motores homeostáticos*
- `__init__` (line 441)
- `forward` (line 447)
- `update_performance` (line 450)
- `__init__` (line 456)
- `forward` (line 463)
- `__init__` (line 489)
- `_generate_topology_mask` (line 504) - *Genera máscara topológica realista*
- `forward` (line 525)
- `__init__` (line 546)
- `forward` (line 571)
- `__init__` (line 600)
- `compute_phi_effective` (line 614) - *Cálculo simplificado de Φₑ (integración efectiva)*
- `forward` (line 634)
- `__init__` (line 661)
- `regulate_homeostasis` (line 666) - *Regula parámetros para homeostasis*
- `__init__` (line 692)
- `initialize_context` (line 726) - *Inicializa el contexto del Omni Brain*
- `forward` (line 740) - *Forward pass del Omni Brain con coordinación homeostática*
- `get_status_report` (line 828) - *Genera reporte de estado del Omni Brain*

#### `omni1.py`
**Path:** `omni1.py`

**Classs:**
- `FastSlowLinear` (line 31) - *Linear layer con pesos hebbianos mejorados y mayor capacidad de adaptación.*
- `ConsciousnessModule` (line 83) - *Módulo de consciencia con Φₑ mejorado y umbral reducido para integración temprana.*
- `OmniBrainV8` (line 140) - *Arquitectura optimizada con conexiones residuales.*
- `DualSystemModule` (line 215) - *Sistema dual rápido/lento con memoria.*
- `FocalLoss` (line 261)

**Functions:**
- `train_model` (line 250) - *Entrenamiento optimizado.*
- `evaluate` (line 356) - *Evaluación estándar.*
- `get_cifar10_loaders` (line 399) - *Loaders con data augmentation.*
- `diagnose_model` (line 423) - *Diagnóstico profundo del modelo.*
- `main` (line 481) - *POC mejorado.*
- `__init__` (line 33)
- `reset_fast_weights` (line 46)
- `update_fast_weights` (line 49)
- `forward` (line 68)
- `get_fast_norm` (line 75)
- `__init__` (line 85)
- `compute_phi_effective` (line 99)
- `forward` (line 121)
- `__init__` (line 142)
- `forward` (line 188)
- `reset_all_fast_weights` (line 198)
- `get_fast_norms` (line 205)
- `__init__` (line 217)
- `forward` (line 235)
- `get_activation` (line 436)
- `__init__` (line 262)
- `forward` (line 268)
- `hook` (line 437)

#### `omni3.py`
**Path:** `omni3.py`

**Classs:**
- `Config` (line 22)
- `FastSlowLinear` (line 105)
- `DualSystemModule` (line 196)
- `IntegrationModule` (line 232) - *Renombrado de ConsciousnessModule - más honesto sobre su función*
- `OmniBrainFastSlow` (line 268)

**Functions:**
- `compute_integration_index` (line 65) - *MÉTRICA HONESTA: Mide el ratio de varianza explicada por el primer componente.
Utiliza SVD para estabilidad numérica en matrices de covarianza deficientes.*
- `get_cifar10_loaders` (line 340)
- `evaluate_full` (line 368) - *Evaluación con múltiples métricas*
- `train` (line 403)
- `run_ablation_study` (line 539) - *Ejecuta múltiples configuraciones para validar cada componente*
- `__init__` (line 106)
- `reset_fast_weights` (line 128) - *Reinicia la memoria a corto plazo (debe llamarse por época, no por batch)*
- `update_fast_weights` (line 134) - *Actualización Hebbiana controlada.*
- `forward` (line 169)
- `get_fast_norm` (line 189)
- `__init__` (line 197)
- `forward` (line 211)
- `__init__` (line 236)
- `forward` (line 248)
- `__init__` (line 269)
- `forward` (line 304)
- `reset_all_fast_weights` (line 314) - *Reinicia todos los pesos rápidos del modelo. 
FIX: Ahora solo cada 10 épocas para preservar memoria a corto plazo.*
- `get_fast_norms` (line 323) - *Recopila normas de fast weights de todos los módulos*
- `get_ablation_state` (line 327) - *Estado actual para logging*

#### `omnibrain.py`
**Path:** `omnibrain.py`

**Classs:**
- `FastSlowLinear` (line 31) - *Linear layer con pesos hebbianos estabilizados.*
- `DualSystemModule` (line 91) - *Sistema dual rápido/lento con memoria.*
- `ConsciousnessModule` (line 119) - *Módulo de consciencia con Φₑ mejorado.*
- `OmniBrainV8` (line 174) - *Arquitectura completa con switches para ablation.*

**Functions:**
- `get_cifar10_loaders` (line 233) - *Loaders con data augmentation.*
- `get_few_shot_loaders` (line 253) - *Few-shot learning setup: entrenar en clases limitadas.*
- `evaluate` (line 287) - *Evaluación con opción de métricas por clase.*
- `train_model` (line 331) - *Entrenamiento con learning rate scheduler y early stopping.*
- `run_ablation_study` (line 430) - *Ejecuta 4 configuraciones y compara resultados.*
- `run_few_shot_experiment` (line 475) - *Prueba capacidad de few-shot learning.*
- `analyze_phi_per_class` (line 508) - *Analiza correlación entre Φₑ y dificultad de clase.*
- `plot_ablation_results` (line 545) - *Genera gráficas comparativas de ablation study.*
- `plot_phi_analysis` (line 617) - *Gráfica correlación Φₑ vs dificultad de clase.*
- `main` (line 673) - *Ejecuta el POC completo.*
- `__init__` (line 33)
- `reset_fast_weights` (line 49)
- `update_fast_weights` (line 53)
- `forward` (line 73)
- `end_of_batch` (line 84)
- `get_fast_norm` (line 87)
- `__init__` (line 93)
- `forward` (line 109)
- `__init__` (line 121)
- `compute_phi_effective` (line 133) - *Φₑ basado en eigenvalues de covarianza.*
- `forward` (line 156)
- `__init__` (line 176)
- `forward` (line 209)
- `reset_all_fast_weights` (line 216)
- `get_fast_norms` (line 223)

#### `omnibrain_k.py`
**Path:** `omnibrain_k.py`

**Classs:**
- `Config` (line 22)
- `FastSlowLinear` (line 105)
- `DualSystemModule` (line 195)
- `IntegrationModule` (line 231) - *Renombrado de ConsciousnessModule - más honesto sobre su función*
- `OmniBrainFastSlow` (line 266)

**Functions:**
- `compute_integration_index` (line 65) - *MÉTRICA HONESTA: Mide el ratio de varianza explicada por el primer componente.
Utiliza SVD para estabilidad numérica en matrices de covarianza deficientes.*
- `get_cifar10_loaders` (line 339)
- `evaluate_full` (line 367) - *Evaluación con múltiples métricas*
- `train` (line 402)
- `run_ablation_study` (line 537) - *Ejecuta múltiples configuraciones para validar cada componente*
- `__init__` (line 106)
- `reset_fast_weights` (line 128) - *Reinicia la memoria a corto plazo (debe llamarse por época, no por batch)*
- `update_fast_weights` (line 134) - *Actualización Hebbiana controlada.*
- `forward` (line 169)
- `get_fast_norm` (line 189)
- `__init__` (line 196)
- `forward` (line 210)
- `__init__` (line 235)
- `forward` (line 247)
- `__init__` (line 267)
- `forward` (line 302)
- `reset_all_fast_weights` (line 312) - *Reinicia todos los pesos rápidos del modelo. Debe llamarse explícitamente
(por ejemplo, al inicio de cada época si se desea resetear la memoria a corto plazo).
No se activa automáticamente durante forward.*
- `get_fast_norms` (line 322) - *Recopila normas de fast weights de todos los módulos*
- `get_ablation_state` (line 326) - *Estado actual para logging*

#### `omno1.bkp.py.py`
**Path:** `omno1.bkp.py.py`

**Classs:**
- `FastSlowLinear` (line 32) - *Linear layer con pesos hebbianos mejorados.*
- `ConsciousnessModule` (line 101) - *Módulo de consciencia con Φₑ mejorado y menos restrictivo.*
- `OmniBrainV8` (line 181) - *Arquitectura optimizada con conexiones residuales.*
- `DualSystemModule` (line 256) - *Sistema dual rápido/lento con memoria.*
- `FocalLoss` (line 302)

**Functions:**
- `train_model` (line 291) - *Entrenamiento optimizado.*
- `evaluate` (line 399) - *Evaluación estándar.*
- `get_cifar10_loaders` (line 442) - *Loaders con data augmentation.*
- `diagnose_model` (line 466) - *Diagnóstico profundo del modelo.*
- `main` (line 524) - *POC mejorado.*
- `__init__` (line 34)
- `reset_fast_weights` (line 51)
- `update_fast_weights` (line 56)
- `forward` (line 85)
- `get_fast_norm` (line 93)
- `__init__` (line 103)
- `compute_phi_effective` (line 118) - *Φₑ mejorado con condiciones menos restrictivas.*
- `forward` (line 159)
- `__init__` (line 183)
- `forward` (line 229)
- `reset_all_fast_weights` (line 239)
- `get_fast_norms` (line 246)
- `__init__` (line 258)
- `forward` (line 276)
- `get_activation` (line 479)
- `__init__` (line 303)
- `forward` (line 309)
- `hook` (line 480)

#### `physio_chimera_demo.py`
**Path:** `physio_chimera_demo.py`

**Classs:**
- `Config` (line 24)
- `DataEnvironment` (line 46)
- `SimpleMonitor` (line 82)
- `SimpleCMS` (line 130)
- `SimplePhysioNeuron` (line 153)
- `SimplePhysioChimera` (line 194)

**Functions:**
- `seed_everything` (line 36)
- `train_demo` (line 231)
- `run_demo` (line 298)
- `__init__` (line 47)
- `get_batch` (line 57)
- `get_full` (line 71) - *Retorna el dataset completo*
- `get_w2` (line 75) - *Retorna solo los datos de WORLD_2 (dígitos >= 5)*
- `__init__` (line 83)
- `update` (line 88)
- `report` (line 94)
- `__init__` (line 131)
- `forward` (line 142)
- `__init__` (line 154)
- `forward` (line 162)
- `__init__` (line 195)
- `forward` (line 207)

#### `physio_chimera_v15_monitored.py`
**Path:** `physio_chimera_v15_monitored.py`

**Classs:**
- `Config` (line 42)
- `DataEnvironment` (line 68)
- `NeuralDiagnostics` (line 104) - *Sistema de diagnóstico neurológico para Physio-Chimera*
- `SelfModifyingGates` (line 298)
- `ContinuumMemorySystem` (line 319)
- `NestedPhysioNeuron` (line 346)
- `PhysioChimeraNested` (line 397)
- `MetricsVisualizer` (line 452) - *Visualiza métricas de entrenamiento*

**Functions:**
- `seed_everything` (line 58)
- `train_nested_monitored` (line 658)
- `run_experiment_monitored` (line 762)
- `__init__` (line 69)
- `get_batch` (line 79)
- `get_full` (line 93) - *Retorna el dataset completo*
- `get_w2` (line 97) - *Retorna solo los datos de WORLD_2 (dígitos >= 5)*
- `__init__` (line 107)
- `update_physio_metrics` (line 144) - *Actualiza métricas fisiológicas*
- `update_performance_metrics` (line 150) - *Actualiza métricas de rendimiento*
- `update_memory_metrics` (line 158) - *Actualiza métricas de memoria*
- `calculate_health_metrics` (line 166) - *Calcula métricas de salud del sistema*
- `get_recent_avg` (line 191) - *Obtiene promedio reciente de una métrica*
- `generate_diagnostic_report` (line 209) - *Genera reporte de diagnóstico*
- `save_metrics` (line 279) - *Guarda todas las métricas*
- `__init__` (line 299)
- `forward` (line 306)
- `__init__` (line 320)
- `forward` (line 332)
- `__init__` (line 347)
- `forward` (line 362)
- `__init__` (line 398)
- `forward` (line 410)
- `__init__` (line 455)
- `plot_training_curves` (line 463) - *Genera gráficos de curvas de entrenamiento*
- `create_final_report` (line 551) - *Crea reporte final con todas las métricas*
- `_generate_recommendations` (line 634) - *Genera recomendaciones basadas en el diagnóstico*

#### `physioneruon_simple.py`
**Path:** `physioneruon_simple.py`

**Classs:**
- `SimpleConfig` (line 27)
- `SimpleRobustNet` (line 83) - *Red simple pero bien regularizada*

**Functions:**
- `seed_everything` (line 54)
- `get_dataset` (line 61) - *Dataset balanceado con más separabilidad*
- `pgd_attack` (line 117) - *PGD estándar bien implementado
- Random start
- Step size controlado
- Projection al epsilon-ball*
- `train_simple_robust` (line 154) - *Entrenamiento con adversarial training progresivo*
- `main` (line 281)
- `__init__` (line 85)
- `forward` (line 105)

#### `physioneuron_cpu_v1.py`
**Path:** `physioneuron_cpu_v1.py`

**Classs:**
- `MicroConfig` (line 33)
- `HomeostaticRegulator` (line 85)
- `PhysioNeuron` (line 109)
- `MicroContinuumCell` (line 151)
- `MicroSymbioticBasis` (line 176)
- `MicroTopology` (line 196)
- `MicroSupConLoss` (line 216)
- `MicroTopoBrain` (line 240)

**Functions:**
- `seed_everything` (line 57)
- `get_dataset` (line 65)
- `micro_pgd_attack` (line 340)
- `generate_ablation_matrix` (line 364)
- `train_with_cv` (line 393)
- `run_ablation_study` (line 457)
- `__init__` (line 86)
- `forward` (line 96)
- `__init__` (line 110)
- `forward` (line 121)
- `__init__` (line 152)
- `forward` (line 162)
- `__init__` (line 177)
- `forward` (line 185)
- `__init__` (line 197)
- `get_adjacency` (line 210)
- `__init__` (line 217)
- `forward` (line 222)
- `__init__` (line 241)
- `_init_weights` (line 278)
- `count_parameters` (line 283)
- `forward` (line 286)

#### `physioneuron_cpu_v2.py`
**Path:** `physioneuron_cpu_v2.py`

**Classs:**
- `MicroConfig` (line 35)
- `HomeostaticRegulator` (line 92)
- `PhysioNeuron` (line 116)
- `MicroContinuumCell` (line 158)
- `MicroSymbioticBasis` (line 183)
- `MicroTopology` (line 203)
- `MicroSupConLoss` (line 223)
- `MicroTopoBrain` (line 247)

**Functions:**
- `seed_everything` (line 64)
- `get_dataset` (line 72)
- `micro_pgd_attack` (line 347)
- `generate_ablation_matrix` (line 371)
- `train_with_cv` (line 400)
- `run_ablation_study` (line 464)
- `__init__` (line 93)
- `forward` (line 103)
- `__init__` (line 117)
- `forward` (line 128)
- `__init__` (line 159)
- `forward` (line 169)
- `__init__` (line 184)
- `forward` (line 192)
- `__init__` (line 204)
- `get_adjacency` (line 217)
- `__init__` (line 224)
- `forward` (line 229)
- `__init__` (line 248)
- `_init_weights` (line 285)
- `count_parameters` (line 290)
- `forward` (line 293)

#### `physioneuron_cpu_v3.py`
**Path:** `physioneuron_cpu_v3.py`

**Classs:**
- `EliteConfig` (line 35)
- `EpisodicMemory` (line 102) - *Memoria explícita de patrones adversariales*
- `SpectralNormLinear` (line 136) - *Linear con normalización espectral para estabilidad Lipschitz*
- `AdvancedHomeostaticCell` (line 162) - *Neurona con control fisiológico multinivel + memoria*
- `AdaptiveTopology` (line 222) - *Topología que aprende a reconectar bajo ataque*
- `EliteTopoBrain` (line 260)
- `SupConLoss` (line 385)

**Functions:**
- `seed_everything` (line 71)
- `get_elite_dataset` (line 79) - *Dataset más grande y balanceado con separabilidad controlada*
- `elite_pgd_attack` (line 348) - *PGD con reinicio aleatorio*
- `train_elite_model` (line 430) - *Entrenamiento con curriculum adversarial*
- `run_elite_experiment` (line 542)
- `__init__` (line 104)
- `update` (line 112) - *Almacena ejemplos duros*
- `retrieve` (line 125) - *Recupera k vecinos más cercanos*
- `__init__` (line 138)
- `power_iteration` (line 145) - *Aproxima la norma espectral máxima*
- `forward` (line 152)
- `__init__` (line 164)
- `forward` (line 194)
- `__init__` (line 224)
- `forward` (line 248) - *stress ∈ [0,1]: cuánto estrés adversarial*
- `__init__` (line 261)
- `count_parameters` (line 303)
- `forward` (line 306)
- `__init__` (line 386)
- `forward` (line 390)

#### `poke_cifar.py`
**Path:** `poke_cifar.py`

**Classs:**
- `PTSymmetricLayer` (line 56)
- `TopologicalLayer` (line 77)
- `DualSystemModule` (line 98)
- `ConsciousnessModule` (line 116)
- `OmniBrainCIFAR` (line 133)

**Functions:**
- `compute_phi_effective` (line 35)
- `get_cifar10_loaders` (line 180)
- `evaluate` (line 200)
- `main` (line 221)
- `plot_history` (line 287)
- `demo_inference` (line 300)
- `__init__` (line 57)
- `forward` (line 66)
- `__init__` (line 78)
- `_update_mask` (line 87)
- `forward` (line 93)
- `__init__` (line 99)
- `forward` (line 107)
- `__init__` (line 117)
- `forward` (line 123)
- `__init__` (line 134)
- `forward` (line 161)

#### `poke_cifar2.py`
**Path:** `poke_cifar2.py`

**Classs:**
- `FastSlowLinear` (line 49)
- `DualSystemModule` (line 99)
- `ConsciousnessModule` (line 117)
- `OmniBrainFastSlow` (line 131)

**Functions:**
- `compute_phi_effective` (line 28)
- `get_cifar10_loaders` (line 175)
- `evaluate` (line 191)
- `train` (line 212)
- `__init__` (line 50)
- `reset_fast_weights` (line 65)
- `update_fast_weights` (line 69)
- `forward` (line 76)
- `end_of_batch` (line 89)
- `get_fast_weight_norm` (line 92)
- `__init__` (line 100)
- `forward` (line 108)
- `__init__` (line 118)
- `forward` (line 124)
- `__init__` (line 132)
- `forward` (line 151)
- `reset_all_fast_weights` (line 159)
- `get_fast_norms` (line 164)

#### `pokemon3.py`
**Path:** `pokemon3.py`

**Classs:**
- `HomeostasisContext` (line 76)
- `PTSymmetricLayer` (line 82) - *Capa PT-simétrica compatible con todas las versiones*
- `TopologicalLayer` (line 109) - *Capa topológica estable sin dependencias problemáticas*
- `DualSystemModule` (line 134) - *Sistema dual compatible con PyTorch 1.8+*
- `ConsciousnessModule` (line 164) - *Módulo de conciencia estable*
- `OmniBrain` (line 187) - *¡El Pokémon Legendario compatible con todas las versiones!*

**Functions:**
- `compute_phi_effective_approx` (line 30) - *Cálculo estable de Φₑ compatible con todas las versiones*
- `estimate_energy_consumption` (line 63) - *Estimación conservadora de energía*
- `prepare_mnist_data` (line 245) - *Preparar datos MNIST con protección para entornos limitados*
- `train_omni_brain` (line 269) - *Entrenamiento compatible con todas las versiones de PyTorch*
- `evaluate_model` (line 380) - *Evaluación compatible con todas las versiones*
- `generate_evolution_plots` (line 402) - *Generar gráficos con protección para entornos sin GUI*
- `demonstrate_inference` (line 435) - *Demostración compatible con todas las versiones*
- `final_report` (line 467) - *Reporte final compatible*
- `__init__` (line 85)
- `compute_pt_phase` (line 94)
- `forward` (line 102)
- `__init__` (line 112)
- `update_topology` (line 121)
- `forward` (line 128)
- `__init__` (line 137)
- `forward` (line 153)
- `__init__` (line 167)
- `forward` (line 177)
- `__init__` (line 190)
- `forward` (line 217)
- `update_topology` (line 235)

#### `pokemon4.py`
**Path:** `pokemon4.py`

**Classs:**
- `PTSymmetricLayer` (line 68)
- `TopologicalLayer` (line 91)
- `DualSystemModule` (line 113)
- `ConsciousnessModule` (line 146)
- `OmniBrain` (line 168)

**Functions:**
- `compute_phi_effective` (line 41) - *Φₑ realista: fracción de varianza explicada por el primer componente PCA.*
- `get_mnist_loaders` (line 206)
- `evaluate` (line 218)
- `train_and_evaluate` (line 236)
- `plot_history` (line 302)
- `__init__` (line 69)
- `forward` (line 78)
- `__init__` (line 92)
- `_update_mask` (line 101)
- `forward` (line 107)
- `__init__` (line 114)
- `forward` (line 130)
- `__init__` (line 147)
- `forward` (line 157)
- `__init__` (line 169)
- `forward` (line 186)

#### `pokemon_battle_champion.py`
**Path:** `pokemon_battle_champion.py`

**Classs:**
- `ChampionConfig` (line 31)
- `PokemonBattleChampion` (line 41) - *Campeón híbrido que combina VAE + Attention + GAN*

**Functions:**
- `create_battle_dataset` (line 128) - *Crear dataset para la batalla*
- `battle_training_epoch` (line 161) - *Entrenamiento de una época de batalla*
- `evaluate_battle_champion` (line 192) - *Evaluar el campeón en batalla*
- `run_epic_pokemon_battle` (line 208) - *¡EJECUTAR LA BATALLA ÉPICA!*
- `create_epic_battle_visualization` (line 334) - *Crear visualización épica de la batalla*
- `save_battle_results` (line 440) - *Guardar resultados de la batalla épica*
- `__init__` (line 44)
- `forward` (line 99)

#### `pokemon_hybrid_synergy_ablation.py`
**Path:** `pokemon_hybrid_synergy_ablation.py`

**Classs:**
- `SynergyConfig` (line 37)
- `SynergyVAELayer` (line 82) - *VAE híbrido con capacidades de compactación extremas*
- `SynergyAttentionLayer` (line 125) - *Multi-head attention optimizado para datos tabulares*
- `SynergyGANLayer` (line 157) - *GAN híbrido optimizado para features tabulares*
- `AdaptiveTopologyLayer` (line 190) - *Topología adaptativa inspirada en TopoBrain evolution*
- `PokemonSynergyModel` (line 270) - *Modelo híbrido que combina los mejores elementos de VAE, Transformer, GAN y TopoBrain*
- `SynergyAblationStudy` (line 386) - *Estudio de ablación sistemático de sinergias*
- `BaselineModel` (line 428)
- `HybridModel` (line 443)
- `AdvancedModel` (line 463)

**Functions:**
- `run_synergy_ablation` (line 495) - *Ejecutar estudio de ablación completo*
- `analyze_synergy_results` (line 637) - *Analizar resultados del estudio de sinergias*
- `create_synergy_visualizations` (line 696) - *Crear visualizaciones del estudio de sinergias*
- `to_dict` (line 75)
- `__init__` (line 84)
- `reparameterize` (line 111)
- `forward` (line 116)
- `__init__` (line 127)
- `forward` (line 149)
- `__init__` (line 159)
- `generate` (line 184)
- `discriminate` (line 187)
- `__init__` (line 192)
- `get_adjacency_matrix` (line 220)
- `forward` (line 234)
- `__init__` (line 273)
- `forward` (line 327)
- `__init__` (line 389)
- `get_ablation_matrix` (line 393) - *Matriz de ablación de 4 niveles:

Nivel 1 - Baseline: Solo VAE básico
Nivel 2 - Hybrid: VAE + Attention 
Nivel 3 - Advanced: VAE + Attention + GAN
Nivel 4 - Full Synergy: VAE + Attention + GAN + Topology*
- `create_variant_model` (line 424) - *Crear modelo variante para un nivel específico*
- `__init__` (line 429)
- `forward` (line 434)
- `__init__` (line 444)
- `forward` (line 450)
- `__init__` (line 464)
- `forward` (line 471)

#### `premium_synergy_demo.py`
**Path:** `premium_synergy_demo.py`

**Classs:**
- `ComponentState` (line 20) - *Estado de un componente del sistema*
- `DemocraticDecision` (line 29) - *Decisión del sistema democrático*
- `TopoBrainComponent` (line 36) - *TopoBrain v8 - Dynamic Topology + Symbiotic Basis*
- `OmniBrainComponent` (line 75) - *OmniBrain K - Integration Index + Fast-Slow Weights*
- `QuimeraComponent` (line 116) - *Quimera v9.5 - Liquid Neurons + Sovereign Attention*
- `HomeostaticMotor` (line 160) - *Motor Homeostático - Cámara Alta de deliberación democrática*
- `PremiumSynergySystem` (line 225) - *Sistema Premium Synergy completo*

**Functions:**
- `run_demo` (line 302) - *Ejecuta demostración del sistema Premium Synergy*
- `__init__` (line 39)
- `process` (line 48) - *Procesamiento con autoregulación interna*
- `__init__` (line 78)
- `process` (line 87) - *Procesamiento con control integrativo*
- `__init__` (line 119)
- `process` (line 128) - *Procesamiento con regulación de fases*
- `__init__` (line 163)
- `deliberate` (line 171) - *Proceso de deliberación democrática*
- `__init__` (line 228)
- `process_epoch` (line 241) - *Procesa una época del sistema democrático*
- `calculate_target_accuracy` (line 296) - *Calcula accuracy objetivo basada en sinergia actual*

#### `premium_synergy_democratic.py`
**Path:** `premium_synergy_democratic.py`

**Classs:**
- `PremiumSynergyConfig` (line 42) - *Configuración del sistema Premium Synergy*
- `MemoryChecker` (line 94) - *Sistema de monitoreo de memoria*
- `TopoBrainComponent` (line 139) - *TopoBrain v8 con autoregulación interna*
- `OmniBrainComponent` (line 235) - *OmniBrain K con autoregulación interna*
- `QuimeraComponent` (line 325) - *Quimera v9.5 con autoregulación interna*
- `MetabolismRegulator` (line 419) - *Regulador de metabolismo para TopoBrain*
- `SensitivityGate` (line 459) - *Compuerta de sensibilidad para TopoBrain*
- `DynamicTopologyGrid` (line 492) - *Topología dinámica para TopoBrain*
- `SymbioticBasis` (line 519) - *Basis simbólica para TopoBrain*
- `IntegrationModule` (line 540) - *Módulo de integración para OmniBrain*
- `FastSlowLinear` (line 565) - *Capa fast-slow weights para OmniBrain*
- `DualSystemModule` (line 595) - *Sistema dual para OmniBrain*
- `IntegrativeControl` (line 615) - *Control integrativo para OmniBrain*
- `ChaosModulator` (line 648) - *Modulador de caos para OmniBrain*
- `LiquidNeuron` (line 682) - *Neurona líquida para Quimera*
- `SovereignAttention` (line 712) - *Atención soberana para Quimera*
- `DualPhaseMemory` (line 738) - *Memoria de fase dual para Quimera*
- `PhaseRegulator` (line 765) - *Regulador de fases para Quimera*
- `AttentionController` (line 798) - *Controlador de atención para Quimera*
- `HomeostaticMotor` (line 836) - *Motor homeostático - Cámara Alta de deliberación democrática*
- `PremiumSynergyModel` (line 960) - *Modelo Premium Synergy con sistema democrático deliberativo*
- `SystemRegulator` (line 1074) - *Regulador general del sistema*

**Functions:**
- `ensure_dependencies` (line 1108) - *Asegura que las dependencias estén instaladas*
- `create_synthetic_dataset` (line 1118) - *Crea dataset sintético para testing*
- `train_premium_synergy` (line 1148) - *Entrena el modelo Premium Synergy*
- `create_dataloader` (line 1272) - *Crea dataloader*
- `main` (line 1284) - *Función principal*
- `__init__` (line 97)
- `check_memory` (line 101) - *Verifica el uso de memoria actual*
- `warn_if_high` (line 126) - *Advierte si el uso de memoria es alto*
- `__init__` (line 142)
- `forward` (line 175)
- `internal_dialogue` (line 222) - *Diálogo interno fisiológico - metabolimo, sensibilidad, gating*
- `__init__` (line 238)
- `forward` (line 268)
- `internal_dialogue` (line 312) - *Diálogo interno - balance integrativo y modulación caótica*
- `__init__` (line 328)
- `forward` (line 358)
- `internal_dialogue` (line 400) - *Diálogo interno - regulación de fases y control atencional*
- `consolidate` (line 409) - *SVD consolidation de liquid neurons*
- `__init__` (line 421)
- `forward` (line 431)
- `get_state` (line 456)
- `__init__` (line 461)
- `forward` (line 471)
- `get_level` (line 489)
- `__init__` (line 494)
- `_create_grid_mask` (line 501)
- `get_adjacency` (line 514)
- `__init__` (line 521)
- `forward` (line 531)
- `__init__` (line 542)
- `forward` (line 552)
- `get_level` (line 562)
- `__init__` (line 567)
- `forward` (line 579)
- `__init__` (line 597)
- `forward` (line 604)
- `get_balance` (line 612)
- `__init__` (line 617)
- `forward` (line 627)
- `get_state` (line 645)
- `__init__` (line 650)
- `forward` (line 660)
- `get_resistance` (line 678)
- `__init__` (line 684)
- `forward` (line 692)
- `consolidate_svd` (line 703)
- `__init__` (line 714)
- `forward` (line 721)
- `get_metrics` (line 733)
- `__init__` (line 740)
- `forward` (line 746)
- `update` (line 754)
- `get_coherence` (line 762)
- `__init__` (line 767)
- `forward` (line 777)
- `get_level` (line 795)
- `__init__` (line 800)
- `forward` (line 810)
- `get_control` (line 829)
- `__init__` (line 839)
- `forward` (line 861) - *Cámara Alta: Delibera sobre las sinergias de los componentes

Returns:
    adjusted_output: Output ajustado por la deliberación
    metrics: Métricas del proceso deliberativo*
- `adjust_for_convergence` (line 926) - *Motor homeostático ajusta si las sinergias no convergen*
- `__init__` (line 963)
- `forward` (line 989) - *Forward pass completo con sistema democrático*
- `democratic_deliberation_status` (line 1061) - *Estado de la deliberación democrática*
- `__init__` (line 1076)
- `forward` (line 1086)

#### `quen7.py`
**Path:** `quen7.py`

**Classs:**
- `Config` (line 33)
- `DataEnvironment` (line 55)
- `HomeostaticRegulator` (line 89)
- `PhysioNeuron` (line 115)
- `SupConHead` (line 158)
- `MicroTopoBrain` (line 173)
- `NeuralDiagnostics` (line 215)

**Functions:**
- `seed_everything` (line 45)
- `train_nonstationary` (line 265)
- `run_ablation_study` (line 330)
- `__init__` (line 56)
- `get_batch` (line 66)
- `get_full` (line 80)
- `get_w2` (line 83)
- `__init__` (line 90)
- `forward` (line 100)
- `__init__` (line 116)
- `forward` (line 128)
- `__init__` (line 159)
- `forward` (line 167)
- `__init__` (line 174)
- `count_parameters` (line 187)
- `forward` (line 190)
- `__init__` (line 216)
- `update` (line 226)
- `get_recent_avg` (line 234)
- `report` (line 239)

#### `quimera.py`
**Path:** `quimera.py`

**Classs:**
- `ChimeraScientificConfig` (line 19)
- `RealWorldEnvironment` (line 86)
- `LiquidNeuron` (line 114) - *Componente Base: Plasticidad + Estabilidad*
- `SovereignAttention` (line 153) - *Atención Soberana (Identity Init)*
- `DualPhaseMemory` (line 176) - *Memoria Dual (DPM)*
- `Chimera_v9_Scientific` (line 201)

**Functions:**
- `seed_everything` (line 37)
- `measure_spatial_richness` (line 47) - *Mide la diversidad espacial de las activaciones (Richness)*
- `get_structure_entropy` (line 62) - *Mide la entropía estructural de los pesos (Entropy)*
- `train_chimera_scientific` (line 279)
- `generate_chimera_matrix` (line 361)
- `run_scientific_study` (line 399)
- `__init__` (line 87)
- `get_batch` (line 97)
- `__init__` (line 116)
- `forward` (line 124)
- `consolidate_svd` (line 138) - *Mecanismo de SVD (Science-ready)*
- `__init__` (line 155)
- `forward` (line 165)
- `__init__` (line 178)
- `forward` (line 184)
- `update` (line 191)
- `__init__` (line 202)
- `forward` (line 229)
- `consolidate` (line 267)

#### `quimera_vision.py`
**Path:** `quimera_vision.py`

**Classs:**
- `Flickr8kMMDataset` (line 43)
- `ImgEncoder` (line 96)
- `AudioEncoder` (line 108)
- `Decoder` (line 121)

**Functions:**
- `text_to_seq` (line 37)
- `collate` (line 87)
- `generate_caption` (line 168)
- `__init__` (line 44)
- `__len__` (line 62)
- `__getitem__` (line 64)
- `__init__` (line 97)
- `forward` (line 103)
- `__init__` (line 109)
- `forward` (line 116)
- `__init__` (line 122)
- `forward` (line 127)

#### `qwen.py`
**Path:** `qwen.py`

**Classs:**
- `MicroConfig` (line 35)
- `HomeostaticOrchestrator` (line 83) - *Regula TODO: plasticity, continuum, supcon, symbiosis, etc.
Entradas: estado global del sistema
Salidas: controles específicos para cada componente*
- `RegulableContinuum` (line 121)
- `RegulableSymbiotic` (line 143)
- `RegulableTopology` (line 162)
- `RegulableSupConHead` (line 181)
- `MicroTopoBrain` (line 196)

**Functions:**
- `seed_everything` (line 57)
- `get_dataset` (line 64)
- `micro_pgd_attack` (line 283)
- `train_with_cv` (line 306)
- `generate_ablation_matrix` (line 367)
- `run_ablation_study` (line 392)
- `__init__` (line 89)
- `forward` (line 99)
- `__init__` (line 122)
- `forward` (line 132)
- `__init__` (line 144)
- `forward` (line 151)
- `__init__` (line 163)
- `get_adjacency` (line 176)
- `__init__` (line 182)
- `forward` (line 190)
- `__init__` (line 197)
- `_init_weights` (line 215)
- `count_parameters` (line 220)
- `forward` (line 223)

#### `qwen3.py`
**Path:** `qwen3.py`

**Classs:**
- `Config` (line 36)
- `DataEnvironment` (line 60)
- `HomeostaticRegulator` (line 94)
- `PhysioNeuron` (line 120)
- `RegulableSymbiotic` (line 160)
- `RegulableTopology` (line 179)
- `MicroTopoBrain` (line 201)

**Functions:**
- `seed_everything` (line 50)
- `train_nonstationary` (line 263)
- `generate_ablation_matrix` (line 320)
- `run_ablation_study` (line 352)
- `__init__` (line 61)
- `get_batch` (line 71)
- `get_full` (line 85)
- `get_w2` (line 88)
- `__init__` (line 95)
- `forward` (line 105)
- `__init__` (line 121)
- `forward` (line 132)
- `__init__` (line 161)
- `forward` (line 168)
- `__init__` (line 180)
- `get_adjacency` (line 193)
- `__init__` (line 202)
- `count_parameters` (line 221)
- `forward` (line 224)

#### `qwen4.py`
**Path:** `qwen4.py`

**Classs:**
- `Config` (line 36)
- `DataEnvironment` (line 60)
- `HomeostaticRegulator` (line 94)
- `PhysioNeuron` (line 120)
- `RegulableSymbiotic` (line 160)
- `RegulableTopology` (line 179)
- `MicroTopoBrain` (line 201)

**Functions:**
- `seed_everything` (line 50)
- `train_nonstationary` (line 263)
- `generate_ablation_matrix` (line 325)
- `run_ablation_study` (line 357)
- `__init__` (line 61)
- `get_batch` (line 71)
- `get_full` (line 85)
- `get_w2` (line 88)
- `__init__` (line 95)
- `forward` (line 105)
- `__init__` (line 121)
- `forward` (line 132)
- `__init__` (line 161)
- `forward` (line 168)
- `__init__` (line 180)
- `get_adjacency` (line 193)
- `__init__` (line 202)
- `count_parameters` (line 221)
- `forward` (line 224)

#### `qwen5.py`
**Path:** `qwen5.py`

**Classs:**
- `Config` (line 35)
- `DataEnvironment` (line 59)
- `WorldModel` (line 94) - *LSTM ligero que predice la próxima fase*
- `EpisodeMemory` (line 116) - *Memoria de claves-valores ligera para estados fisiológicos óptimos*
- `PredictiveHomeostat` (line 138)
- `PredictivePhysioNeuron` (line 196)
- `PhysioChimeraV15` (line 247)

**Functions:**
- `seed_everything` (line 49)
- `train_predictive` (line 285)
- `run_experiment` (line 351)
- `__init__` (line 60)
- `get_batch` (line 71)
- `get_full` (line 85)
- `get_w2` (line 88)
- `__init__` (line 96)
- `forward` (line 103)
- `__init__` (line 118)
- `store` (line 122)
- `retrieve` (line 129)
- `__init__` (line 139)
- `forward` (line 153)
- `__init__` (line 197)
- `forward` (line 209)
- `consolidate_svd` (line 239)
- `__init__` (line 248)
- `count_parameters` (line 261)
- `forward` (line 264)

#### `qwen6.py`
**Path:** `qwen6.py`

**Classs:**
- `Config` (line 31)
- `DataEnvironment` (line 53)
- `SelfModifyingGates` (line 89)
- `ContinuumMemorySystem` (line 110)
- `NestedPhysioNeuron` (line 133)
- `PhysioChimeraNested` (line 169)

**Functions:**
- `seed_everything` (line 43)
- `train_nested` (line 202)
- `run_experiment` (line 265)
- `__init__` (line 54)
- `get_batch` (line 64)
- `get_full` (line 78) - *Retorna el dataset completo*
- `get_w2` (line 82) - *Retorna solo los datos de WORLD_2 (dígitos >= 5)*
- `__init__` (line 90)
- `forward` (line 97)
- `__init__` (line 111)
- `forward` (line 123)
- `__init__` (line 134)
- `forward` (line 145)
- `__init__` (line 170)
- `forward` (line 182)

#### `qwen8.py`
**Path:** `qwen8.py`

**Classs:**
- `Config` (line 33)
- `DataEnvironment` (line 55)
- `HomeostaticRegulator` (line 89)
- `PhysioNeuron` (line 116)
- `SupConHead` (line 160)
- `MicroTopoBrain` (line 175)
- `NeuralDiagnostics` (line 217)

**Functions:**
- `seed_everything` (line 45)
- `train_nonstationary` (line 267)
- `run_ablation_study` (line 332)
- `__init__` (line 56)
- `get_batch` (line 66)
- `get_full` (line 80)
- `get_w2` (line 83)
- `__init__` (line 90)
- `forward` (line 100)
- `__init__` (line 117)
- `forward` (line 129)
- `__init__` (line 161)
- `forward` (line 169)
- `__init__` (line 176)
- `count_parameters` (line 189)
- `forward` (line 192)
- `__init__` (line 218)
- `update` (line 228)
- `get_recent_avg` (line 236)
- `report` (line 241)

#### `qwen9.py`
**Path:** `qwen9.py`

**Classs:**
- `LiquidNeuron` (line 34)
- `RightHemisphere` (line 78)
- `LeftHemisphere` (line 97)
- `CorpusCallosum` (line 195)
- `HomeostaticRegulator` (line 219)
- `NeuroLogosBicameral` (line 251)
- `NeuralDiagnostics` (line 291)
- `Flickr8kDataset` (line 338)

**Functions:**
- `build_vocab_flickr` (line 371)
- `setup_flickr8k` (line 386)
- `train_bicameral` (line 400)
- `__init__` (line 35)
- `forward` (line 49)
- `__init__` (line 79)
- `forward` (line 88)
- `__init__` (line 98)
- `forward` (line 119)
- `_get_init_state` (line 177)
- `_top_p_filtering` (line 182)
- `__init__` (line 196)
- `forward` (line 209)
- `__init__` (line 220)
- `forward` (line 230)
- `update_flow_ema` (line 245)
- `__init__` (line 252)
- `forward` (line 259)
- `__init__` (line 292)
- `measure_callosal_flow` (line 299)
- `measure_vocab_diversity` (line 306)
- `update` (line 312)
- `get_recent_avg` (line 317)
- `report` (line 321)
- `__init__` (line 339)
- `__len__` (line 355)
- `__getitem__` (line 358)

#### `qwn2.py`
**Path:** `qwn2.py`

**Classs:**
- `MicroConfig` (line 34)
- `HomeostaticRegulator` (line 80)
- `AutoregulatedPlasticity` (line 97)
- `AutoregulatedContinuum` (line 122)
- `AutoregulatedSymbiotic` (line 155)
- `AutoregulatedSupConHead` (line 179)
- `MicroTopoBrain` (line 199)

**Functions:**
- `seed_everything` (line 54)
- `get_dataset` (line 61)
- `micro_pgd_attack` (line 259)
- `train_with_cv` (line 279)
- `generate_ablation_matrix` (line 341)
- `run_ablation_study` (line 366)
- `__init__` (line 81)
- `forward` (line 91)
- `__init__` (line 98)
- `get_adjacency` (line 111)
- `__init__` (line 123)
- `forward` (line 133)
- `__init__` (line 156)
- `forward` (line 164)
- `__init__` (line 180)
- `forward` (line 189)
- `__init__` (line 200)
- `_init_weights` (line 215)
- `count_parameters` (line 220)
- `forward` (line 223)

#### `resma4.10.py`
**Path:** `resma4.10.py`

**Classs:**
- `RESMAConstants` (line 34)
- `GarnierTresTiempos` (line 71)
- `OperadorDesdoblamiento` (line 111)
- `SilencioActivoMonitor` (line 158)
- `QuantumLeaf` (line 186)
- `RESMAUniverse` (line 233)
- `NeuralNetworkRESMA` (line 341)
- `MyelinCavity` (line 509)
- `ExperimentalPredictions` (line 545)
- `ResourceMonitor` (line 592)

**Functions:**
- `guardar_checkpoint` (line 604)
- `cargar_checkpoint` (line 662)
- `_make_serializable` (line 690) - *Convierte objetos a formato serializable de forma segura.

Args:
    obj: Objeto a serializar
    depth: Nivel de profundidad actual (auto-incremental)
    max_depth: Profundidad máxima permitida
    _visited: Diccionario de objetos ya procesados (para referencias circulares)*
- `simulate_resma_garnier` (line 794)
- `verify_pt_condition` (line 50)
- `__post_init__` (line 74)
- `epsilon_critico` (line 86)
- `modulation_factor` (line 89)
- `to_dict` (line 92)
- `from_dict` (line 102)
- `__init__` (line 112)
- `_construir_generadores_aleatorios` (line 121)
- `_hadamard_generalizado` (line 130)
- `operator` (line 135)
- `calcular_alpha_modificado` (line 150)
- `__init__` (line 159)
- `calcular_delta_s_loop` (line 163)
- `es_silencio_activo` (line 170)
- `__post_init__` (line 193)
- `spectral_density` (line 197)
- `bures_distance` (line 203)
- `__init__` (line 234)
- `_initialize_leaves` (line 270)
- `_generate_complete_measure` (line 281)
- `_aplicar_modulacion_garnier` (line 310)
- `_construct_global_state` (line 321)
- `_calcular_libertad` (line 330)
- `_calcular_coherencia` (line 333)
- `__init__` (line 342)
- `_generate_realistic_modular_network` (line 380)
- `_compute_betti_numbers` (line 454)
- `_spectral_dimension` (line 462)
- `_topological_ramsey` (line 484)
- `_calcular_rho_reducida` (line 488)
- `_validar_axioma_6` (line 496)
- `__init__` (line 510)
- `_free_hamiltonian` (line 527)
- `_loss_potential` (line 532)
- `_compute_scalar_mass` (line 538)
- `__init__` (line 546)
- `compute_log_bayes_factor` (line 551)
- `get_memory_gb` (line 594)
- `log_resources` (line 599)

#### `resma4.2.py`
**Path:** `resma4.2.py`

**Classs:**
- `RESMAConstants` (line 38) - *Constantes físicas RESMA 4.0 con correcciones PT-simétricas*
- `PhysicalValidator` (line 78)
- `QuantumLeaf` (line 109) - *Hoja L_i como estado KMS con espacio de Hilbert standard*
- `RESMAUniverse` (line 162) - *Multiverso como foliación medible, memoria O(N_leaves)*
- `EmunaOperator` (line 224) - *P̂_E: proyección teleológica no lineal en H²(ℂ⁺)*
- `MyelinCavity` (line 291) - *Cavidad dieléctrica H = H₀ + iV_loss con Spin(7)*
- `NeuralNetworkRESMA` (line 351) - *Conectoma NO DIRIGIDO con homología persistente*
- `ExperimentalPredictions` (line 474) - *Predicciones con BF logarítmico*

**Functions:**
- `simulate_resma_complete` (line 556) - *Pipeline RESMA 4.2 completo*
- `verify_pt_condition` (line 67) - *Verificar condición PT: κ < χΩ*
- `validate_dimension` (line 80)
- `validate_pt_symmetry` (line 88)
- `validate_connectome_size` (line 96)
- `validate_spectral_dimension` (line 101)
- `__post_init__` (line 117)
- `spectral_density` (line 121) - *ρ(ω) con regularización UV*
- `modular_entropy` (line 126) - *S = -∫ ρ log ρ dω*
- `bures_distance` (line 136) - *Distancia de Bures W₂(ρ₁, ρ₂)*
- `haagerup_weight` (line 154) - *Peso de Haagerup para regularización*
- `__init__` (line 165)
- `_initialize_leaves` (line 176)
- `_generate_gibbs_measure` (line 189) - *μ(i,j) = exp(-β·W₂²(ρᵢ, ρⱼ))*
- `_construct_global_state` (line 205) - *Estado global: pesos por hoja*
- `compute_gibbs_free_energy` (line 216) - *F = -ln(Tr(μ)) / β*
- `__init__` (line 227)
- `_construct_hardy_state` (line 234) - *E(z) ∈ H²(ℂ⁺)*
- `_szego_projector` (line 238) - *Proyector en frecuencias positivas*
- `_evaluation_functional` (line 246) - *Φ_E[|Ψ⟩] = exp(∫ log(⟨Φᵢ|E⟩) dμ)*
- `project` (line 258) - *P̂_E = P_E ∘ Φ_E (con interpolación adaptativa)*
- `__post_init__` (line 297)
- `_free_hamiltonian` (line 304) - *H₀: dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 310) - *V_loss ∝ (r/a₀)^(2α)*
- `_compute_scalar_mass` (line 317) - *Campo escalar para estabilización Spin(7)*
- `_pt_symmetry_condition` (line 321) - *κ < χΩ*
- `coherence_quantum` (line 327) - *Coherencia cuántica con verificación espectral*
- `__init__` (line 354)
- `_generate_fractal_graph` (line 366) - *Scale-free → NO DIRIGIDO*
- `_spectral_dimension` (line 381) - *d_s = -2 lim log N(λ)/log λ*
- `_topological_ramsey` (line 410) - *R_Q(G) = min{n | β_{n-1}(G) > 0}*
- `_compute_betti_numbers` (line 429) - *Números de Betti β₀, β₁*
- `_graph_to_distance_matrix` (line 445) - *Matriz de distancias para homología*
- `critical_percolation_time` (line 461) - *t_c = 21 · (N/N₀)^0.25 / log R_Q*
- `__init__` (line 477)
- `predict_all` (line 483) - *Predicciones RESMA 4.2*
- `compute_log_bayes_factor` (line 496) - *ln(BF) con AIC*

#### `resma4.3.py`
**Path:** `resma4.3.py`

**Classs:**
- `ResourceMonitor` (line 33)
- `RESMAConstants` (line 110)
- `QuantumLeaf` (line 142) - *Hoja KMS - INMUTABLE pero con caché externo*
- `RESMAUniverse` (line 190) - *Multiverso con construcción lazy*
- `PhysicalValidator` (line 269)
- `MyelinCavity` (line 297)
- `NeuralNetworkRESMA` (line 352)
- `ExperimentalPredictions` (line 485)

**Functions:**
- `guardar_checkpoint` (line 54) - *Guardado atómico con backup*
- `cargar_checkpoint` (line 84) - *Cargar checkpoint con fallback*
- `simulate_resma_with_checkpointing` (line 545) - *Pipeline con reanudación inteligente desde checkpoints*
- `get_memory_gb` (line 35)
- `check_memory_limit` (line 40)
- `log_resources` (line 49)
- `verify_pt_condition` (line 127)
- `__post_init__` (line 150)
- `spectral_density` (line 154)
- `bures_distance` (line 158) - *Distancia Bures con caché EXTERNO (no en instancia)*
- `__init__` (line 193)
- `_initialize_leaves` (line 215)
- `_generate_gibbs_measure` (line 227) - *Matriz de medida con guardado incremental*
- `_construct_global_state` (line 254)
- `validate_dimension` (line 271)
- `validate_pt_symmetry` (line 279)
- `validate_connectome_size` (line 287)
- `validate_spectral_dimension` (line 292)
- `__post_init__` (line 302)
- `_free_hamiltonian` (line 312)
- `_loss_potential` (line 317)
- `_compute_scalar_mass` (line 323)
- `_pt_symmetry_condition` (line 326)
- `coherence_quantum` (line 331)
- `__init__` (line 353)
- `_generate_fractal_graph` (line 372) - *Generar grafo por lotes*
- `_spectral_dimension` (line 401) - *Dimensión espectral con matriz sparse*
- `_topological_ramsey` (line 425) - *Ramsey topológico*
- `_compute_betti_numbers` (line 444) - *Números de Betti*
- `_graph_to_distance_matrix` (line 460) - *Matriz de distancias sparse*
- `critical_percolation_time` (line 476) - *Tiempo crítico de percolación*
- `__init__` (line 486)
- `compute_log_bayes_factor` (line 492) - *ln(BF)*

#### `resma4.4.py`
**Path:** `resma4.4.py`

**Classs:**
- `ResourceMonitor` (line 34)
- `RESMAConstants` (line 129)
- `QuantumLeaf` (line 160) - *Hoja KMS - INMUTABLE*
- `RESMAUniverse` (line 203) - *Multiverso con estado serializable*
- `PhysicalValidator` (line 316)
- `MyelinCavity` (line 343)
- `NeuralNetworkRESMA` (line 377)

**Functions:**
- `guardar_checkpoint` (line 59) - *Guarda el estado COMPLETO de los objetos, no solo metadatos*
- `cargar_checkpoint` (line 93) - *Carga el estado COMPLETO desde disco*
- `simulate_resma_with_checkpointing` (line 554) - *Pipeline con reanudación que realmente carga objetos*
- `get_memory_gb` (line 36)
- `check_memory_limit` (line 41)
- `log_resources` (line 50)
- `verify_pt_condition` (line 146)
- `__post_init__` (line 168)
- `spectral_density` (line 172)
- `bures_distance` (line 176) - *Distancia Bures con caché externo*
- `__init__` (line 206) - *Constructor que puede recibir estado serializado*
- `_initialize_leaves` (line 258)
- `_generate_gibbs_measure` (line 269) - *Matriz de medida*
- `_construct_global_state` (line 301)
- `validate_dimension` (line 318)
- `validate_pt_symmetry` (line 326)
- `validate_connectome_size` (line 334)
- `validate_spectral_dimension` (line 339)
- `__post_init__` (line 348)
- `_free_hamiltonian` (line 358)
- `_loss_potential` (line 363)
- `_compute_scalar_mass` (line 369)
- `_pt_symmetry_condition` (line 372)
- `__init__` (line 378) - *Constructor que puede recibir grafo ya construido*
- `_generate_fractal_graph` (line 446) - *Generar grafo por lotes*
- `_spectral_dimension` (line 475) - *Dimensión espectral con eigenvalores sparse*
- `_topological_ramsey` (line 499) - *Ramsey topológico*
- `_compute_betti_numbers` (line 518) - *Números de Betti*
- `_graph_to_distance_matrix` (line 534) - *Matriz de distancias sparse*

#### `resma4.5.py`
**Path:** `resma4.5.py`

**Classs:**
- `ResourceMonitor` (line 34)
- `RESMAConstants` (line 135)
- `GarnierTresTiempos` (line 163) - *Toro temporal T³ con parámetros ADIMENSIONALES.
C0, C2, C3 son ratios de escala, no velocidades.*
- `OperadorDesdoblamiento` (line 201) - *D̂_G(ϕ) = exp(i Σ_i φ_i H_i) · H_E
Representación toy de E8 (248x248)*
- `SilencioActivoMonitor` (line 269) - *Monitor de Silencio-Activo: ΔS_loop < ε_c(ϕ)*
- `QuantumLeaf` (line 337) - *Hoja KMS - INMUTABLE (SIN CAMBIOS)*
- `RESMAUniverse` (line 380) - *Multiverso con estado serializable y desdoblamiento Garnier*
- `MyelinCavity` (line 486) - *Cavidad PT-simétrica (SIN CAMBIOS)*
- `NeuralNetworkRESMA` (line 517) - *Red neuronal con embedding Garnier*
- `ExperimentalPredictions` (line 658) - *Cálculos experimentales (SIN CAMBIOS)*

**Functions:**
- `guardar_checkpoint` (line 59)
- `cargar_checkpoint` (line 88)
- `_make_serializable` (line 113) - *Convierte objetos a formato serializable*
- `simulate_resma_garnier` (line 695) - *Pipeline único con Garnier integrado*
- `get_memory_gb` (line 36)
- `check_memory_limit` (line 41)
- `log_resources` (line 50)
- `verify_pt_condition` (line 152)
- `__post_init__` (line 170)
- `factor_escala` (line 181) - *Factor de escala para cada tiempo: 0=lento, 2=modular, 3=teleológico*
- `epsilon_critico` (line 185) - *Entropía crítica de percolación (ADIMENSIONAL).
log(2) es la entropía de un bit cuántico crítico.*
- `to_dict` (line 192) - *Para serialización*
- `from_dict` (line 197)
- `__init__` (line 206)
- `_construir_generadores_E8` (line 214) - *Construye 3 generadores temporales (antis-Hermitianos)*
- `_hadamard_generalizado` (line 226) - *Operador de Hadamard en dimensión 248 (unitario)*
- `operator` (line 235) - *Construye D̂_G(ϕ) dimensionalmente consistente*
- `aplicar_a_estado` (line 254) - *Aplica desdoblamiento a un estado cuántico |Ψ⟩*
- `calcular_alpha_modificado` (line 260) - *α'(ϕ) = α · tanh(C0/C3 · cos(ϕ₃))
Garantiza α' ∈ [0, α]*
- `__init__` (line 273)
- `calcular_delta_s_loop` (line 278) - *ΔS_loop = S_vN(ρ_red) - log(b₁ + 1)
rho_red: matriz densidad reducida (si es None, se calcula)*
- `_calcular_rho_reducida_aproximada` (line 299) - *Aproximación: ρ_red = diag(grados) / sum(grados)*
- `es_silencio_activo` (line 307) - *Verifica Silencio-Activo y calcula Libertad L.
Retorna: (condicion, libertad_L)*
- `umbral_percolacion` (line 324) - *Umbral de percolación para soberanía: 70% (Axioma 6)*
- `__post_init__` (line 345)
- `spectral_density` (line 349)
- `bures_distance` (line 353)
- `__init__` (line 383) - *Constructor que puede recibir estado serializado*
- `_initialize_leaves` (line 420)
- `_generate_gibbs_measure` (line 431) - *Matriz de medida sin desdoblamiento*
- `_aplicar_desdoblamiento_a_medida` (line 451) - *Aplica D̂_G(ϕ) a la medida:
- M_ij → M_ij * (C0/C3)^(cos(ϕ₃))
- Normaliza después*
- `_construct_global_state` (line 471)
- `_calcular_libertad_universo` (line 481) - *Libertad del universo: L = 1/ε_c*
- `__init__` (line 488)
- `_free_hamiltonian` (line 499)
- `_loss_potential` (line 504)
- `_compute_scalar_mass` (line 510)
- `_pt_symmetry_condition` (line 513)
- `__init__` (line 520) - *Constructor que puede recibir grafo ya construido*
- `_generate_fractal_graph` (line 559) - *Generar grafo por lotes con conectividad controlada*
- `_spectral_dimension` (line 591) - *Dimensión espectral con eigenvalores sparse*
- `_topological_ramsey` (line 615) - *Ramsey topológico simplificado*
- `_compute_betti_numbers` (line 627) - *Números de Betti aproximados por ciclos locales*
- `_calcular_rho_reducida` (line 636) - *Matriz densidad reducida del conectoma*
- `validar_axioma_6` (line 644) - *Verifica: conectividad > 70% para soberanía*
- `__init__` (line 661)
- `compute_log_bayes_factor` (line 666) - *Calcula Factor de Bayes integrando Garnier*

#### `resma4.6.py`
**Path:** `resma4.6.py`

**Classs:**
- `ResourceMonitor` (line 31)
- `RESMAConstants` (line 139) - *Constantes físicas fundamentales*
- `GarnierTresTiempos` (line 169) - ***TORO TEMPORAL T³ CON CANCELACIÓN ZPE**
- phi: Fase de desdoblamiento que controla anulación ZPE
- zpe_level: Nivel de fluctuaciones de punto cero [0,1]*
- `OperadorDesdoblamiento` (line 233) - ***D̂_G(ϕ) = exp(i Σ_i φ_i H_i) · H_E**
**CONTRA-ZPE**: Opera en subespacio sin fluctuaciones*
- `SilencioActivoMonitor` (line 312) - ***MONITOR DE ANTAGONISMO ZPE-SILENCIO**
- Detecta cuando fluctuaciones cuánticas son coherentemente anuladas
- Mide nivel de "ruido de fondo cuántico" vs "silencio ontológico"*
- `QuantumLeaf` (line 441) - *Hoja KMS - INMUTABLE*
- `RESMAUniverse` (line 484) - *Multiverso con ZPE-Silencio integrado*
- `MyelinCavity` (line 608) - *Cavidad PT-simétrica con medición ZPE*
- `ConectomaCuantico` (line 659) - ***SUPERPOSICIÓN CUÁNTICA DE GRAFOS** (Pre-geométrico)
- No es un grafo, es una matriz de amplitudes
- Colapsa a grafo clásico solo bajo medición
- b1_cuántico ≠ b1_clásico*
- `NeuralNetworkRESMA` (line 800) - *Red neuronal con **conectoma cuántico** subyacente*
- `ExperimentalPredictions` (line 898) - *Cálculos experimentales unificados ZPE-Silencio*

**Functions:**
- `guardar_checkpoint` (line 56) - *Guarda estado completo con manejo robusto de errores*
- `cargar_checkpoint` (line 86) - *Carga checkpoint con fallback automático*
- `_make_serializable` (line 112) - *Convierte objetos recursivamente a formato serializable*
- `simulate_resma_garnier` (line 951) - *Pipeline completo RESMA 4.3.5 con antagonismo ZPE-Silencio*
- `get_memory_gb` (line 33)
- `check_memory_limit` (line 38)
- `log_resources` (line 47)
- `verify_pt_condition` (line 158)
- `__post_init__` (line 181)
- `factor_escala` (line 194) - *Factor de escala con supresión ZPE*
- `epsilon_critico` (line 200) - ***UMBRAL CRÍTICO CON ZPE**:
Cuando zpe_level → 0, ε_c → 0 (Silencio perfecto no necesita umbral)*
- `to_dict` (line 208) - *Serialización completa*
- `from_dict` (line 220) - *Deserialización*
- `__init__` (line 238)
- `_construir_generadores_E8_ZPE` (line 246) - *GENERADORES CON CANCELACIÓN ZPE INTEGRADA*
- `_hadamard_generalizado_ZPE` (line 266) - *HADAMARD CON ESPACIO NULO ZPE*
- `operator` (line 284) - *Construye D̂_G(ϕ) con cancelación ZPE*
- `alpha_modificado` (line 304) - ***α'(ϕ) = α · tanh(C₀/C₃ · cos(ϕ₃) · (1 - zpe_level))***
- `__init__` (line 318)
- `calcular_delta_s_loop` (line 327) - ***ΔS_loop = S_vN(ρ_red) - S_top + S_ZPE**
**NUEVO**: La entropía ZPE se SUMA a la entropía total*
- `_calcular_rho_reducida_aproximada` (line 367) - *Matriz densidad con modulación ZPE*
- `es_silencio_activo` (line 383) - ***DETECCIÓN DE ANTAGONISMO**:
Retorna: (condicion, libertad_L, nivel_ZPE_cancelado)

**CONDICIÓN**: ZPE < 1% AND ΔS_loop < ε_c*
- `umbral_percolacion` (line 411) - *Umbral para soberanía: 70%*
- `modo_goldstone` (line 415) - ***MODO GOLDSTONE DEL DOBLE CUÁNTICO**:
Excitación colectiva que anuncia ruptura de simetría ZPE*
- `__post_init__` (line 449)
- `spectral_density` (line 453)
- `bures_distance` (line 457)
- `__init__` (line 487)
- `_initialize_leaves` (line 521) - *Inicializa hojas con temperatura efectiva afectada por ZPE*
- `_generate_gibbs_measure` (line 535) - *Genera medida de Gibbs*
- `_aplicar_desdoblamiento_a_medida` (line 557) - *Aplica desdoblamiento con supresión ZPE*
- `_construct_global_state` (line 584) - *Construye estado global normalizado*
- `_calcular_libertad_universo` (line 603) - *Libertad intrínseca con supresión ZPE*
- `__init__` (line 611)
- `_free_hamiltonian` (line 624) - *Hamiltoniano con energía ZPE incluida*
- `_loss_potential` (line 633) - *Potencial de pérdida PT*
- `_compute_scalar_mass` (line 640)
- `_calcular_zpe` (line 643) - ***ENERGÍA DE PUNTO CERO TOTAL**:
E_ZPE = Σ_i ½ħω_i*
- `_pt_symmetry_condition` (line 655)
- `__init__` (line 667)
- `_inicializar_amplitudes` (line 686) - ***AMPLITUDES DE FEYNMAN** para cada posible arista:
- |A_ij|² es probabilidad de existencia de arista
- Fase ϕ_ij controlada por Garnier*
- `_calcular_conectividad_cuantica` (line 708) - ***CONECTIVIDAD CUÁNTICA** (no clásica):
= Σ_i<j |A_ij|² / (N(N-1)/2)*
- `colapsar_a_clasico` (line 718) - ***COLAPSO CUÁNTICO-CLÁSICO**:
- Medición proyectiva con umbral de probabilidad
- b1_clásico ≠ b1_cuántico*
- `_recalcular_betti_clasicos` (line 742) - *Recalcula Betti del grafo colapsado*
- `medir_delta_s_loop` (line 756) - ***ΔS_loop CUÁNTICO** (no clásico):
- Usa matriz densidad de amplitudes (no grafo)
- S_ZPE es intrínseca a la superposición*
- `_calcular_b1_cuantico` (line 784) - ***b₁ CUÁNTICO** (topología pre-geométrica):
= rango de la matriz de amplitudes (conectividad cuántica)*
- `__init__` (line 803)
- `_calcular_rho_reducida` (line 842) - ***Matriz densidad reducida del conectoma cuántico**
- Usa amplitudes, no grafo colapsado*
- `validar_axioma_6_cuantico` (line 858) - ***AXIOMA 6 CUÁNTICO**: Conectividad cuántica > 70%
**NUEVO**: La soberanía se juzga en el estado pre-geométrico, no en el colapso*
- `obtener_metricas_cuanticas` (line 874) - ***MÉTRICAS EXPERIMENTALES** (falsables):
- Conectividad cuántica (pre-observación)
- b1 cuántico vs b1 clásico
- Ratio de colapso: cuánto cambia la topología*
- `__init__` (line 901)
- `compute_log_bayes_factor` (line 906) - *Calcula Factor de Bayes con antagonismo ZPE-Silencio*

#### `resma4.7.py`
**Path:** `resma4.7.py`

**Classs:**
- `RESMAConstants` (line 23)
- `GarnierTresTiempos` (line 46) - *Toro temporal T³ con parámetros físicamente consistentes.
Basado en la teoría del desdoblamiento del tiempo de Garnier-Malet.*
- `OperadorDesdoblamiento` (line 94) - *Operador de desdoblamiento D̂_G(φ) con estructura E8 simplificada*
- `SilencioActivoMonitor` (line 139) - *Monitor de condición de Silencio-Activo: ΔS_loop < ε_c(φ)*
- `QuantumLeaf` (line 190)
- `RESMAUniverse` (line 223) - *Multiverso cuántico con desdoblamiento Garnier-Malet*
- `NeuralNetworkRESMA` (line 336) - *Red neuronal con topología realista que satisface Axioma 6*
- `ExperimentalPredictions` (line 484) - *Cálculo de Factor de Bayes y predicciones*

**Functions:**
- `simulate_resma_garnier` (line 537) - *Pipeline completo RESMA-Garnier con correcciones*
- `__post_init__` (line 53)
- `_compute_coupling` (line 67) - *Fuerza de acoplamiento entre tiempos*
- `factor_escala` (line 72) - *Factor de escala temporal*
- `epsilon_critico` (line 77) - *Entropía crítica con corrección de acoplamiento:
ε_c = log(2) · (C0/C3)² · (1 + ξ)*
- `modulation_factor` (line 85) - *Factor de modulación para la medida cuántica:
M = exp(-|φ₃ - π|/C3)
Máximo cuando φ₃ ≈ π (apertura temporal óptima)*
- `__init__` (line 98)
- `_construir_generadores` (line 103) - *Generadores temporales (anti-Hermitianos normalizados)*
- `operator` (line 115) - *Construye D̂_G(φ) = exp(i Σ φᵢHᵢ)*
- `aplicar_modulacion` (line 120) - *Aplica desdoblamiento a vector de estado*
- `calcular_alpha_modificado` (line 126) - *α'(φ) = α · |cos(φ₃)|^(C0/C3)
Garantiza α' ∈ [0, α]*
- `__init__` (line 143)
- `calcular_delta_s_loop` (line 147) - *ΔS_loop = S_vN(ρ) - log(b₁ + 1)

Args:
    rho_red: Matriz densidad reducida
    b1: Primer número de Betti (ciclos independientes)*
- `es_silencio_activo` (line 166) - *Verifica condición y calcula libertad L = 1/(ΔS + ε_c)

Returns:
    (condicion_satisfecha, libertad)*
- `spectral_density` (line 197)
- `bures_distance` (line 203) - *Distancia de Bures simplificada*
- `__init__` (line 226)
- `_initialize_leaves` (line 252) - *Genera hojas con gaps distribuidos exponencialmente*
- `_generate_modulated_measure` (line 265) - *Genera medida de transición modulada por Garnier:
M_ij = exp(-β d²_ij) · φ(garnier)*
- `_construct_global_state` (line 312) - *Estado global como distribución diagonal*
- `_calcular_libertad` (line 322) - *Libertad del universo: L_U = 1/ε_c*
- `_calcular_coherencia` (line 326) - *Coherencia cuántica: suma de elementos off-diagonal*
- `__init__` (line 341)
- `_generate_realistic_network` (line 379) - *Genera red con conectividad > 70% usando modelo realista:
- Watts-Strogatz para mundo pequeño
- Aumentación para alcanzar umbral*
- `_compute_betti_numbers` (line 424) - *Números de Betti: b0=componentes, b1=ciclos*
- `_spectral_dimension` (line 433) - *Dimensión espectral del Laplaciano*
- `_topological_ramsey` (line 455) - *Número de Ramsey topológico*
- `_calcular_rho_reducida` (line 460) - *Matriz densidad de la red (normalizada por grados)*
- `_validar_axioma_6` (line 470) - *Verifica conectividad > 70%*
- `__init__` (line 487)
- `compute_log_bayes_factor` (line 491) - *ln(BF) ∝ log(L_red · L_univ)
Veredicto basado en libertad total*

#### `resma4.8.py`
**Path:** `resma4.8.py`

**Classs:**
- `RESMAConstants` (line 37)
- `ResourceMonitor` (line 70)
- `GarnierTresTiempos` (line 158)
- `OperadorDesdoblamiento` (line 209)
- `SilencioActivoMonitor` (line 256)
- `QuantumLeaf` (line 284)
- `RESMAUniverse` (line 331)
- `NeuralNetworkRESMA` (line 433)
- `MyelinCavity` (line 584)
- `ExperimentalPredictions` (line 620)

**Functions:**
- `guardar_checkpoint` (line 91)
- `cargar_checkpoint` (line 117)
- `_make_serializable` (line 141)
- `simulate_resma_garnier` (line 667)
- `verify_pt_condition` (line 54)
- `get_memory_gb` (line 72)
- `check_memory_limit` (line 77)
- `log_resources` (line 86)
- `__post_init__` (line 161)
- `_compute_coupling` (line 179)
- `epsilon_critico` (line 182)
- `modulation_factor` (line 186)
- `to_dict` (line 189)
- `from_dict` (line 199)
- `__init__` (line 210)
- `_construir_generadores_aleatorios` (line 219)
- `_hadamard_generalizado` (line 228)
- `operator` (line 233)
- `calcular_alpha_modificado` (line 248)
- `__init__` (line 257)
- `calcular_delta_s_loop` (line 261)
- `es_silencio_activo` (line 268)
- `__post_init__` (line 291)
- `spectral_density` (line 295)
- `bures_distance` (line 301)
- `__init__` (line 332)
- `_initialize_leaves` (line 362)
- `_generate_complete_measure` (line 373)
- `_aplicar_modulacion_garnier` (line 402)
- `_construct_global_state` (line 413)
- `_calcular_libertad` (line 422)
- `_calcular_coherencia` (line 425)
- `__init__` (line 434)
- `_generate_realistic_modular_network` (line 472)
- `_compute_betti_numbers` (line 529)
- `_spectral_dimension` (line 537)
- `_topological_ramsey` (line 559)
- `_calcular_rho_reducida` (line 563)
- `_validar_axioma_6` (line 571)
- `__init__` (line 585)
- `_free_hamiltonian` (line 602)
- `_loss_potential` (line 607)
- `_compute_scalar_mass` (line 613)
- `__init__` (line 621)
- `compute_log_bayes_factor` (line 626)

#### `resma4.9.py`
**Path:** `resma4.9.py`

**Classs:**
- `RESMAConstants` (line 34)
- `GarnierTresTiempos` (line 71)
- `OperadorDesdoblamiento` (line 111)
- `SilencioActivoMonitor` (line 158)
- `QuantumLeaf` (line 186)
- `RESMAUniverse` (line 233)
- `NeuralNetworkRESMA` (line 341)
- `MyelinCavity` (line 492)
- `ExperimentalPredictions` (line 528)
- `ResourceMonitor` (line 575)

**Functions:**
- `guardar_checkpoint` (line 587)
- `cargar_checkpoint` (line 615)
- `_make_serializable` (line 639)
- `simulate_resma_garnier` (line 655)
- `verify_pt_condition` (line 50)
- `__post_init__` (line 74)
- `epsilon_critico` (line 86)
- `modulation_factor` (line 89)
- `to_dict` (line 92)
- `from_dict` (line 102)
- `__init__` (line 112)
- `_construir_generadores_aleatorios` (line 121)
- `_hadamard_generalizado` (line 130)
- `operator` (line 135)
- `calcular_alpha_modificado` (line 150)
- `__init__` (line 159)
- `calcular_delta_s_loop` (line 163)
- `es_silencio_activo` (line 170)
- `__post_init__` (line 193)
- `spectral_density` (line 197)
- `bures_distance` (line 203)
- `__init__` (line 234)
- `_initialize_leaves` (line 270)
- `_generate_complete_measure` (line 281)
- `_aplicar_modulacion_garnier` (line 310)
- `_construct_global_state` (line 321)
- `_calcular_libertad` (line 330)
- `_calcular_coherencia` (line 333)
- `__init__` (line 342)
- `_generate_realistic_modular_network` (line 380)
- `_compute_betti_numbers` (line 437)
- `_spectral_dimension` (line 445)
- `_topological_ramsey` (line 467)
- `_calcular_rho_reducida` (line 471)
- `_validar_axioma_6` (line 479)
- `__init__` (line 493)
- `_free_hamiltonian` (line 510)
- `_loss_potential` (line 515)
- `_compute_scalar_mass` (line 521)
- `__init__` (line 529)
- `compute_log_bayes_factor` (line 534)
- `get_memory_gb` (line 577)
- `log_resources` (line 582)

#### `resma_Test.py`
**Path:** `resma_Test.py`

**Classs:**
- `PTSymmetricActivation` (line 11)
- `E8LatticeLayer` (line 42)
- `RESMABrain` (line 71)

**Functions:**
- `stress_test_resma` (line 88)
- `__init__` (line 12)
- `forward` (line 18)
- `__init__` (line 43)
- `_generate_ramsey_mask` (line 54)
- `forward` (line 64)
- `__init__` (line 72)
- `forward` (line 78)

#### `resmann.py`
**Path:** `resmann.py`

**Classs:**
- `PTSymmetricActivation` (line 11)
- `E8LatticeLayer` (line 41)
- `RESMABrain` (line 86)

**Functions:**
- `__init__` (line 12)
- `forward` (line 19)
- `__init__` (line 42)
- `_generate_ramsey_mask` (line 59)
- `forward` (line 69)
- `__init__` (line 87)
- `forward` (line 97)
- `resma_loss` (line 104)

#### `resmann2.py`
**Path:** `resmann2.py`

**Classs:**
- `PTSymmetricActivation` (line 13)
- `E8LatticeMultiverseLayer` (line 31)
- `RESMABrainMultiverse` (line 65)

**Functions:**
- `__init__` (line 14)
- `forward` (line 20)
- `__init__` (line 32)
- `_multiverse_mask` (line 41)
- `forward` (line 55)
- `__init__` (line 66)
- `forward` (line 74)
- `resma_loss` (line 79)

#### `resmannn.py`
**Path:** `resmannn.py`

**Classs:**
- `PTSymmetricActivation` (line 12)
- `E8LatticeLayer` (line 30)
- `RESMABrainLight` (line 60)

**Functions:**
- `__init__` (line 13)
- `forward` (line 19)
- `__init__` (line 31)
- `_fixed_sparse_mask` (line 40)
- `forward` (line 49)
- `__init__` (line 61)
- `forward` (line 69)
- `resma_loss` (line 74)

#### `run_complete_experiment.py`
**Path:** `run_complete_experiment.py`

**Functions:**
- `create_experiment_summary` (line 24) - *Crea un resumen del experimento*
- `generate_final_report` (line 46) - *Genera reporte final detallado*
- `run_complete_experiment` (line 155) - *Ejecuta el experimento completo con todas las características*

#### `scientific_benchmark.py`
**Path:** `scientific_benchmark.py`

**Classs:**
- `SupConLoss` (line 57)
- `PredictiveErrorCell` (line 85)
- `LearnableAbsenceGating` (line 97)
- `SymbioticBasisRefinement` (line 111)
- `CombinatorialComplexLayer` (line 132)
- `TopoBrainNet` (line 184)
- `Wrapper` (line 317)

**Functions:**
- `seed_everything` (line 42)
- `clamp_pgd` (line 276)
- `make_adversarial_pgd` (line 283)
- `eval_autoattack` (line 299)
- `save_topology_snapshot` (line 332)
- `run_training` (line 350)
- `run_ablation_suite_scientific` (line 473)
- `__init__` (line 58)
- `forward` (line 62)
- `__init__` (line 86)
- `forward` (line 92)
- `__init__` (line 98)
- `forward` (line 107)
- `__init__` (line 112)
- `forward` (line 120)
- `__init__` (line 133)
- `forward` (line 157)
- `__init__` (line 185)
- `_init_grid` (line 220)
- `get_topology` (line 241)
- `forward` (line 253)
- `__init__` (line 318)
- `forward` (line 319)
- `lambda_topo` (line 387)

#### `scientist_sinergy_ablation_plan.py`
**Path:** `scientist_sinergy_ablation_plan.py`

**Classs:**
- `ScientificConfig` (line 31)

**Functions:**
- `check_memory_usage` (line 63) - *Monitorea uso de memoria para evitar crashes con Nested Learning*
- `memory_safe_check` (line 68) - *Verifica si es seguro ejecutar con Nested Learning*
- `setup_matplotlib_for_plotting` (line 85) - *Setup matplotlib para visualizaciones científicas*
- `generate_sinergy_matrix` (line 94) - *Genera matriz de sinergias basada en tus inventos*
- `print_sinergy_analysis` (line 139) - *Analiza las sinergias propuestas basado en tus modelos*
- `main` (line 171)

#### `setup_environment.py`
**Path:** `setup_environment.py`

**Functions:**
- `check_python_version` (line 15) - *Verifica la versión de Python*
- `install_package` (line 23) - *Instala un paquete usando pip*
- `check_and_install_dependencies` (line 32) - *Verifica e instala dependencias*
- `create_directories` (line 76) - *Crea directorios necesarios*
- `setup_matplotlib` (line 92) - *Configura matplotlib para el entorno*
- `create_sample_data` (line 118) - *Crea datos de muestra para pruebas*
- `test_installation` (line 148) - *Prueba la instalación*
- `create_main_script` (line 189) - *Crea script principal para ejecutar experimentos*
- `main` (line 256) - *Función principal de setup*

#### `sintesis.py`
**Path:** `sintesis.py`

**Classs:**
- `SpectralMonitorV6` (line 12)
- `PrismaticNeuron` (line 48)
- `SynthesisOrganismV6` (line 117)

**Functions:**
- `run_prism_dream` (line 162)
- `__init__` (line 13)
- `calc_structural_health` (line 16)
- `measure_spatial_richness` (line 33)
- `__init__` (line 49)
- `forward` (line 58)
- `prismatic_dream` (line 81) - *Sueño Entrópico:
1. Fusionar memoria.
2. Refracción Espectral (Whitening).*
- `__init__` (line 118)
- `forward` (line 128)
- `calculate_losses` (line 134)
- `sleep` (line 154)

#### `sintesys2.py`
**Path:** `sintesys2.py`

**Classs:**
- `SpectralMonitorV7` (line 12)
- `CuriosityGaze` (line 52)
- `PrismaticNeuronV7` (line 69)
- `SynthesisOrganismV7` (line 114)

**Functions:**
- `run_the_prisms_eye` (line 174)
- `__init__` (line 13)
- `calc_structural_health` (line 16)
- `measure_spatial_richness` (line 32) - *Ahora retorna el tensor (para el gradiente) y el valor escalar.
Necesitamos que sea diferenciable para que el 'Ojo' aprenda a buscar riqueza.*
- `__init__` (line 53)
- `forward` (line 60)
- `__init__` (line 70)
- `forward` (line 79)
- `prismatic_dream` (line 95)
- `__init__` (line 115)
- `forward` (line 131)
- `calculate_losses` (line 144)
- `sleep` (line 166)

#### `sintesys3.py`
**Path:** `sintesys3.py`

**Classs:**
- `HomeostasisEngine` (line 30)
- `LiquidNeuron` (line 59)
- `OrganismV8` (line 109)

**Functions:**
- `measure_spatial_richness` (line 12) - *Retorna tensor (gradiente) y valor escalar*
- `run_liquid_synthesis` (line 156)
- `__init__` (line 31)
- `decide` (line 35)
- `__init__` (line 60)
- `forward` (line 68)
- `consolidate_svd` (line 86) - *Sueño a demanda, intensidad variable*
- `__init__` (line 110)
- `forward` (line 126)
- `get_structure_entropy` (line 140)
- `calc_ent` (line 143)

#### `sintesys5.py`
**Path:** `sintesys5.py`

**Classs:**
- `RealWorldEnvironment` (line 19)
- `HomeostasisEngine` (line 68)
- `LiquidNeuron` (line 87)
- `OrganismV8_Real` (line 124)

**Functions:**
- `measure_spatial_richness` (line 56)
- `run_real_world_challenge` (line 157)
- `__init__` (line 20)
- `get_batch` (line 38)
- `__init__` (line 69)
- `decide` (line 73)
- `__init__` (line 88)
- `forward` (line 96)
- `consolidate_svd` (line 111)
- `__init__` (line 125)
- `forward` (line 135)
- `get_structure_entropy` (line 143)
- `calc_ent` (line 145)

#### `syntesys4.py`
**Path:** `syntesys4.py`

**Classs:**
- `HomeostasisEngine` (line 27)
- `LiquidNeuron` (line 60)
- `OrganismV8_1` (line 101)

**Functions:**
- `measure_spatial_richness` (line 12)
- `run_sensitive_self` (line 131)
- `__init__` (line 28)
- `decide` (line 32)
- `__init__` (line 61)
- `forward` (line 69)
- `consolidate_svd` (line 84)
- `__init__` (line 102)
- `forward` (line 112)
- `get_structure_entropy` (line 120)
- `calc_ent` (line 122)

#### `test.py`
**Path:** `test.py`

**Classs:**
- `ColapsoGarantizado` (line 52)

**Functions:**
- `measure_metrics` (line 80)
- `calculate_test_accuracy` (line 109)
- `__init__` (line 53)
- `forward` (line 69)

#### `test_premium_synergy.py`
**Path:** `test_premium_synergy.py`

**Functions:**
- `test_individual_components` (line 29) - *Test de componentes individuales*
- `test_full_system` (line 91) - *Test del sistema completo Premium Synergy*
- `test_training_loop` (line 164) - *Test del loop de entrenamiento completo*
- `run_all_tests` (line 209) - *Ejecuta todos los tests*

#### `topobrain.py`
**Path:** `topobrain.py`

**Classs:**
- `ResourceMonitor` (line 58)
- `LearnableAbsenceGating` (line 135)
- `SupConLoss` (line 147)
- `PredictiveErrorCell` (line 175)
- `SymbioticBasisRefinement` (line 187)
- `CombinatorialComplexLayer` (line 205)
- `TopoBrainNet` (line 252)
- `Wrapper` (line 409)

**Functions:**
- `seed_everything` (line 50)
- `guardar_checkpoint` (line 87) - *✅ FIX: Guarda solo estado esencial + compresión*
- `cargar_checkpoint` (line 118)
- `clamp_pgd` (line 352)
- `make_adversarial_pgd` (line 359)
- `eval_autoattack` (line 391)
- `save_topology_snapshot` (line 423)
- `plot_topology_evolution` (line 449)
- `run_training` (line 476)
- `run_diagnostic_suite` (line 676) - *✅ Suite completa con TopoOnly crítico*
- `get_memory_gb` (line 60)
- `check_memory_limit` (line 65)
- `log_resources` (line 72) - *✅ FIX: Método faltante añadido*
- `clear_cache` (line 81)
- `__init__` (line 136)
- `forward` (line 143)
- `__init__` (line 148)
- `forward` (line 152)
- `__init__` (line 176)
- `forward` (line 182)
- `__init__` (line 188)
- `forward` (line 196)
- `__init__` (line 206)
- `forward` (line 229)
- `__init__` (line 253)
- `_init_grid` (line 287)
- `get_topology` (line 308)
- `forward` (line 324)
- `__init__` (line 410)
- `forward` (line 411)
- `lambda_topo` (line 529)

#### `topobrain_16_3.py`
**Path:** `topobrain_16_3.py`

**Classs:**
- `ResourceMonitor` (line 70)
- `LearnableAbsenceGating` (line 140)
- `SupConLoss` (line 152)
- `PredictiveErrorCell` (line 183)
- `SymbioticBasisRefinement` (line 195)
- `CombinatorialComplexLayer` (line 213)
- `TopoBrainNet` (line 266)
- `Wrapper` (line 409)

**Functions:**
- `seed_everything` (line 61)
- `guardar_checkpoint` (line 98)
- `cargar_checkpoint` (line 123)
- `clamp_pgd` (line 361)
- `make_adversarial_pgd` (line 368)
- `eval_autoattack` (line 393)
- `save_topology_snapshot` (line 423)
- `plot_topology_evolution` (line 447)
- `run_training` (line 464)
- `run_diagnostic_suite` (line 726)
- `get_memory_gb` (line 72)
- `check_memory_limit` (line 77)
- `log_resources` (line 84)
- `clear_cache` (line 92)
- `__init__` (line 141)
- `forward` (line 148)
- `__init__` (line 153)
- `forward` (line 157)
- `__init__` (line 184)
- `forward` (line 190)
- `__init__` (line 196)
- `forward` (line 204)
- `__init__` (line 214)
- `forward` (line 236)
- `__init__` (line 267)
- `_init_grid` (line 299)
- `get_topology` (line 319)
- `forward` (line 333)
- `__init__` (line 410)
- `forward` (line 411)
- `lambda_topo` (line 531)

#### `topobrain_v18.1.py`
**Path:** `topobrain_v18.1.py`

**Classs:**
- `Config` (line 30) - *Configuración centralizada y reproducible*
- `ResourceMonitor` (line 109)
- `TopologyMetrics` (line 143)
- `TopologicalHealthSovereignty` (line 152) - *monitor_topologico
Fusión de SovereigntyMonitor + TopologicalHealth
< 0.5s por época | Monitorea solo adj_weights/inc_weights*
- `CheckpointManager` (line 260)
- `SupConLoss` (line 366) - *Supervised Contrastive Loss*
- `AsymmetricPredictiveErrorCell` (line 398) - *Predictive Coding con manejo asimétrico de errores
Inspirado en corteza predictiva biológica*
- `LearnableAbsenceGating` (line 437) - *Gating basado en error de predicción*
- `SymbioticBasisRefinement` (line 453) - *Refinamiento mediante base ortogonal aprendible*
- `AdaptiveCombinatorialComplexLayer` (line 473) - *Capa combinatorial con:
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico*
- `TopoBrainNetV18` (line 598) - *TopoBrain v18: Implementa
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico
- Regularización ortogonal ponderada*

**Functions:**
- `seed_everything` (line 100)
- `get_dataset_stats` (line 309)
- `get_dataloaders` (line 316)
- `make_adversarial_pgd` (line 820) - *PGD Attack*
- `train_epoch` (line 848) - *Entrena una época con schedule adaptativo de SupCon - CORREGIDO*
- `train_model` (line 956) - *Loop de entrenamiento v18*
- `evaluate` (line 1139) - *Evalúa el modelo*
- `save_topology_visualization` (line 1166) - *Guarda visualización de la topología aprendida*
- `save_node_importance_viz` (line 1211) - *Visualiza importancia de nodos por capa*
- `analyze_topology_clustering` (line 1234) - *Clustering espectral de nodos basado en conectividad*
- `analyze_topology_flow` (line 1277) - *Analiza flujo de información en la topología
Similar a Grad-CAM pero para topología*
- `visualize_topology_as_graph` (line 1343) - *Visualiza topología como grafo con NetworkX*
- `analyze_topology_evolution` (line 1409) - *Analiza la evolución de la topología a lo largo del entrenamiento*
- `comprehensive_topology_analysis` (line 1478) - *Análisis completo de topología
Ejecuta todos los análisis disponibles*
- `run_ablation_study` (line 1509) - *Ejecuta suite completa de ablación v18*
- `main` (line 1633) - *Punto de entrada principal v18*
- `__post_init__` (line 82)
- `to_dict` (line 86)
- `get_supcon_lambda` (line 89) - *Schedule adaptativo para SupCon Loss*
- `get_memory_gb` (line 111)
- `get_gpu_memory_gb` (line 116)
- `log` (line 122)
- `clear_cache` (line 129)
- `check_limit` (line 135)
- `__init__` (line 159)
- `_analyze_matrix` (line 165) - *Análisis SVD de matriz topológica (adj o inc)*
- `calculate` (line 220) - *Analiza todas las matrices topológicas del modelo*
- `get_critical_summary` (line 247) - *Resumen de emergencias*
- `__init__` (line 261)
- `save` (line 265)
- `load` (line 288)
- `__init__` (line 368)
- `forward` (line 372)
- `__init__` (line 403)
- `forward` (line 415)
- `__init__` (line 439)
- `forward` (line 448)
- `__init__` (line 455)
- `forward` (line 463)
- `__init__` (line 480)
- `forward` (line 509) - *Args:
    x_nodes: [B, N, D] node features
    adjacency: [N, N] dense adjacency (fallback)
    incidence: [N, C] dense incidence (fallback)
    adj_sparse: sparse adjacency
    inc_sparse: sparse incidence*
- `get_node_importance` (line 588) - *Retorna importancia de nodos para visualización*
- `__init__` (line 606)
- `_init_grid_topology` (line 653) - *Inicializa topología de grid 2D*
- `get_topology` (line 681) - *Calcula topología actual

Args:
    return_sparse: Si True, retorna versiones sparse*
- `calculate_ortho_loss` (line 708) - *Regularización ortogonal con pesos por capa*
- `prune_topology` (line 741) - *Poda de topología basada en importancia*
- `forward` (line 787)
- `set_epoch` (line 813) - *Permite pasar la época actual para schedules dinámicos*
- `warmup_topo` (line 1006)

#### `topobrain_v18.2.py`
**Path:** `topobrain_v18.2.py`

**Classs:**
- `Config` (line 30) - *Configuración centralizada y reproducible*
- `ResourceMonitor` (line 109)
- `TopologyMetrics` (line 143)
- `TopologicalHealthSovereignty` (line 152) - *monitor_topologico
Fusión de SovereigntyMonitor + TopologicalHealth
< 0.5s por época | Monitorea solo adj_weights/inc_weights*
- `CheckpointManager` (line 260)
- `SupConLoss` (line 366) - *Supervised Contrastive Loss*
- `AsymmetricPredictiveErrorCell` (line 398) - *Predictive Coding con manejo asimétrico de errores
Inspirado en corteza predictiva biológica*
- `LearnableAbsenceGating` (line 437) - *Gating basado en error de predicción*
- `SymbioticBasisRefinement` (line 453) - *Refinamiento mediante base ortogonal aprendible*
- `AdaptiveCombinatorialComplexLayer` (line 473) - *Capa combinatorial con:
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico*
- `TopoBrainNetV18` (line 599) - *TopoBrain v18: Implementa
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico
- Regularización ortogonal ponderada
- CORRECCIÓN: Patch Size dinámico para ajustar num_nodes al grid_size*

**Functions:**
- `seed_everything` (line 100)
- `get_dataset_stats` (line 309)
- `get_dataloaders` (line 316)
- `make_adversarial_pgd` (line 824) - *PGD Attack*
- `train_epoch` (line 852) - *Entrena una época con schedule adaptativo de SupCon - CORREGIDO*
- `train_model` (line 960) - *Loop de entrenamiento v18 (CORREGIDO - Inicialización Negativa)*
- `evaluate` (line 1088) - *Evalúa el modelo*
- `save_topology_visualization` (line 1115) - *Guarda visualización de la topología aprendida*
- `save_node_importance_viz` (line 1160) - *Visualiza importancia de nodos por capa*
- `analyze_topology_clustering` (line 1183) - *Clustering espectral de nodos basado en conectividad*
- `analyze_topology_flow` (line 1226) - *Analiza flujo de información en la topología
Similar a Grad-CAM pero para topología*
- `visualize_topology_as_graph` (line 1292) - *Visualiza topología como grafo con NetworkX*
- `analyze_topology_evolution` (line 1358) - *Analiza la evolución de la topología a lo largo del entrenamiento*
- `comprehensive_topology_analysis` (line 1427) - *Análisis completo de topología
Ejecuta todos los análisis disponibles*
- `run_ablation_study` (line 1458) - *Ejecuta suite completa de ablación v18*
- `main` (line 1582) - *Punto de entrada principal v18*
- `__post_init__` (line 82)
- `to_dict` (line 86)
- `get_supcon_lambda` (line 89) - *Schedule adaptativo para SupCon Loss*
- `get_memory_gb` (line 111)
- `get_gpu_memory_gb` (line 116)
- `log` (line 122)
- `clear_cache` (line 129)
- `check_limit` (line 135)
- `__init__` (line 159)
- `_analyze_matrix` (line 165) - *Análisis SVD de matriz topológica (CORREGIDO)*
- `calculate` (line 220) - *Analiza todas las matrices topológicas del modelo*
- `get_critical_summary` (line 247) - *Resumen de emergencias*
- `__init__` (line 261)
- `save` (line 265)
- `load` (line 288)
- `__init__` (line 368)
- `forward` (line 372)
- `__init__` (line 403)
- `forward` (line 415)
- `__init__` (line 439)
- `forward` (line 448)
- `__init__` (line 455)
- `forward` (line 463)
- `__init__` (line 480)
- `forward` (line 510) - *Args:
    x_nodes: [B, N, D] node features
    adjacency: [N, N] dense adjacency (fallback)
    incidence: [N, C] dense incidence (fallback)
    adj_sparse: sparse adjacency
    inc_sparse: sparse incidence*
- `get_node_importance` (line 589) - *Retorna importancia de nodos para visualización*
- `__init__` (line 608)
- `_init_grid_topology` (line 668) - *Inicializa topología de grid 2D*
- `get_topology` (line 705) - *Calcula topología actual*
- `calculate_ortho_loss` (line 729) - *Regularización ortogonal con pesos por capa*
- `prune_topology` (line 754) - *Poda de topología basada en importancia*
- `forward` (line 793)
- `set_epoch` (line 818)
- `warmup_topo` (line 1016)

#### `topobrain_v18.py`
**Path:** `topobrain_v18.py`

**Classs:**
- `Config` (line 30) - *Configuración centralizada y reproducible*
- `ResourceMonitor` (line 109)
- `TopologyMetrics` (line 143)
- `TopologicalHealthSovereignty` (line 152) - *monitor_topologico
Fusión de SovereigntyMonitor + TopologicalHealth
< 0.5s por época | Monitorea solo adj_weights/inc_weights*
- `CheckpointManager` (line 260)
- `SupConLoss` (line 366) - *Supervised Contrastive Loss*
- `AsymmetricPredictiveErrorCell` (line 398) - *Predictive Coding con manejo asimétrico de errores
Inspirado en corteza predictiva biológica*
- `LearnableAbsenceGating` (line 437) - *Gating basado en error de predicción*
- `SymbioticBasisRefinement` (line 453) - *Refinamiento mediante base ortogonal aprendible*
- `AdaptiveCombinatorialComplexLayer` (line 473) - *Capa combinatorial con:
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico*
- `TopoBrainNetV18` (line 598) - *TopoBrain v18: Implementa
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico
- Regularización ortogonal ponderada*

**Functions:**
- `seed_everything` (line 100)
- `get_dataset_stats` (line 309)
- `get_dataloaders` (line 316)
- `make_adversarial_pgd` (line 799) - *PGD Attack*
- `train_epoch` (line 827) - *Entrena una época con schedule adaptativo de SupCon*
- `train_model` (line 917) - *Loop de entrenamiento completo*
- `evaluate` (line 1078) - *Evalúa el modelo*
- `train_model` (line 1101) - *Loop de entrenamiento completo*
- `save_topology_visualization` (line 1257) - *Guarda visualización de la topología aprendida*
- `save_node_importance_viz` (line 1302) - *Visualiza importancia de nodos por capa*
- `analyze_topology_clustering` (line 1325) - *Clustering espectral de nodos basado en conectividad*
- `analyze_topology_flow` (line 1368) - *Analiza flujo de información en la topología
Similar a Grad-CAM pero para topología*
- `visualize_topology_as_graph` (line 1434) - *Visualiza topología como grafo con NetworkX*
- `analyze_topology_evolution` (line 1500) - *Analiza la evolución de la topología a lo largo del entrenamiento*
- `comprehensive_topology_analysis` (line 1569) - *Análisis completo de topología
Ejecuta todos los análisis disponibles*
- `run_ablation_study` (line 1600) - *Ejecuta suite completa de ablación v18*
- `main` (line 1724) - *Punto de entrada principal v18*
- `__post_init__` (line 82)
- `to_dict` (line 86)
- `get_supcon_lambda` (line 89) - *Schedule adaptativo para SupCon Loss*
- `get_memory_gb` (line 111)
- `get_gpu_memory_gb` (line 116)
- `log` (line 122)
- `clear_cache` (line 129)
- `check_limit` (line 135)
- `__init__` (line 159)
- `_analyze_matrix` (line 165) - *Análisis SVD de matriz topológica (adj o inc)*
- `calculate` (line 220) - *Analiza todas las matrices topológicas del modelo*
- `get_critical_summary` (line 247) - *Resumen de emergencias*
- `__init__` (line 261)
- `save` (line 265)
- `load` (line 288)
- `__init__` (line 368)
- `forward` (line 372)
- `__init__` (line 403)
- `forward` (line 415)
- `__init__` (line 439)
- `forward` (line 448)
- `__init__` (line 455)
- `forward` (line 463)
- `__init__` (line 480)
- `forward` (line 509) - *Args:
    x_nodes: [B, N, D] node features
    adjacency: [N, N] dense adjacency (fallback)
    incidence: [N, C] dense incidence (fallback)
    adj_sparse: sparse adjacency
    inc_sparse: sparse incidence*
- `get_node_importance` (line 588) - *Retorna importancia de nodos para visualización*
- `__init__` (line 606)
- `_init_grid_topology` (line 653) - *Inicializa topología de grid 2D*
- `get_topology` (line 681) - *Calcula topología actual

Args:
    return_sparse: Si True, retorna versiones sparse*
- `calculate_ortho_loss` (line 708) - *Regularización ortogonal con pesos por capa*
- `prune_topology` (line 741) - *Poda de topología basada en importancia*
- `forward` (line 769)
- `lambda_topo` (line 948)
- `lambda_topo` (line 1130)

#### `topobrain_v19.py`
**Path:** `topobrain_v19.py`

**Classs:**
- `Config` (line 42) - *Configuración unificada y simplificada*
- `ResourceMonitor` (line 107)
- `CheckpointManager` (line 139)
- `NodePositionLearner` (line 240) - *Aprende posiciones de nodos en espacio latente
Genera conectividad k-NN dinámica*
- `DynamicTopologicalLayer` (line 291) - *Capa con PyTorch Geometric y topología dinámica
Combina GAT + Predictive Coding + MGF*
- `TopoBrainNetV19` (line 360) - *Modelo principal con topología dinámica y componentes modulares*
- `ContrastiveLoss` (line 521) - *SupCon simplificado*

**Functions:**
- `seed_everything` (line 98)
- `get_dataset_stats` (line 182)
- `get_dataloaders` (line 189)
- `make_adversarial_pgd` (line 493) - *PGD Attack simplificado y robusto*
- `train_epoch` (line 545) - *Entrena una época con logging integrado*
- `evaluate` (line 621) - *Evalúa el modelo*
- `train_model` (line 644) - *Loop de entrenamiento completo v19*
- `save_topology_snapshot` (line 796) - *Guarda snapshot de topología*
- `run_ablation_study` (line 837) - *Suite de ablación sistemática v19*
- `main` (line 918)
- `to_dict` (line 91)
- `get_memory_gb` (line 109)
- `get_gpu_memory_gb` (line 114)
- `log` (line 120)
- `clear_cache` (line 128)
- `check_limit` (line 134)
- `__init__` (line 140)
- `save` (line 144)
- `load` (line 163)
- `__init__` (line 245)
- `forward` (line 254) - *Retorna edges para k-NN dinámico
Returns:
    edge_index: [2, E]
    edge_weight: [E]*
- `__init__` (line 296)
- `forward` (line 325) - *Args:
    x: [B*N, D] Node features
    edge_index: [2, E] Connectivity
    edge_weight: [E] Edge weights
    batch: [B*N] Batch indices*
- `__init__` (line 364)
- `forward` (line 398)
- `apply_pruning` (line 445) - *Aplica máscara de pruning*
- `prune_structural` (line 453) - *Pruning estructural real: elimina edges permanentemente*
- `calculate_ortho_loss` (line 473) - *Regularización ortogonal simple*
- `__init__` (line 523)
- `forward` (line 527)
- `warmup_lr` (line 668)
- `prune_fn` (line 467)

#### `train_Adversarial.py`
**Path:** `train_Adversarial.py`

**Classs:**
- `ResourceMonitor` (line 74)
- `NestedOptimizer` (line 154) - *Optimizador de múltiples niveles basado en Nested Learning.
Implementa momentum como memoria asociativa con diferentes frecuencias.*
- `ContinuumMemorySystem` (line 218) - *Sistema de memoria continua con MLPs de diferentes frecuencias.
Basado en la Sección 3 del paper Nested Learning.*
- `SupConLoss` (line 260)
- `PredictiveErrorCell` (line 288)
- `LearnableAbsenceGating` (line 300)
- `SymbioticBasisRefinement` (line 314)
- `CombinatorialComplexLayer` (line 332)
- `TopoBrainNet` (line 400)
- `Wrapper` (line 552)

**Functions:**
- `seed_everything` (line 61)
- `guardar_checkpoint` (line 100) - *Sistema de checkpoint robusto con protección contra corrupción*
- `cargar_checkpoint` (line 134) - *Carga checkpoint con fallback automático*
- `clamp_pgd` (line 509)
- `make_adversarial_pgd` (line 516)
- `eval_autoattack` (line 534)
- `create_checkpoint_data` (line 566) - *Crea estructura de checkpoint completa*
- `run_training` (line 586)
- `save_topology_snapshot` (line 770)
- `run_diagnostic_suite` (line 787)
- `get_memory_gb` (line 78)
- `log_resources` (line 83)
- `clear_cache` (line 90) - *Limpia cachés y fuerza garbage collection*
- `__init__` (line 159)
- `step` (line 179)
- `__init__` (line 223)
- `forward` (line 240)
- `should_update_level` (line 246) - *Determina si un nivel debe actualizarse basado en su frecuencia*
- `get_update_mask` (line 250) - *Máscara para actualizar solo un subconjunto de parámetros*
- `__init__` (line 261)
- `forward` (line 265)
- `__init__` (line 289)
- `forward` (line 295)
- `__init__` (line 301)
- `forward` (line 310)
- `__init__` (line 315)
- `forward` (line 323)
- `__init__` (line 333)
- `forward` (line 362)
- `apply_cms` (line 391) - *Aplica actualización condicional basada en frecuencia del CMS*
- `__init__` (line 401)
- `_init_grid` (line 444)
- `get_topology` (line 465)
- `forward` (line 477)
- `__init__` (line 553)
- `forward` (line 554)
- `lambda_topo` (line 632)

#### `tricameral2.py`
**Path:** `tricameral2.py`

**Classs:**
- `StableLiquidNeuron` (line 389)
- `LeftHemisphere` (line 512)
- `Flickr8kMultimodalDataset` (line 584) - *Dataset que carga imagen, audio del caption y texto*
- `AudioEncoder` (line 678) - *Encoder de audio usando Conv + Transformer*
- `RightHemisphereTricameral` (line 747) - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 821) - *Corpus callosum con canales: visual, auditivo, semántico*
- `NeuralAudioGenerator` (line 913) - *Generador de audio desde embeddings lingüísticos (TTS neuronal)*
- `NeuroLogosTricameral` (line 982) - *Arquitectura completa: Visión + Audio -> Lenguaje + Audio*
- `Flickr8kSimpleDataset` (line 1195)

**Functions:**
- `generate_audio_async` (line 31) - *Genera un audio usando Edge-TTS con retry logic*
- `generate_all_audios_batch` (line 66) - *Genera todos los audios en batches pequeños con rate limiting*
- `generate_audios_sync` (line 135) - *Wrapper síncrono para generar audios*
- `download_from_github` (line 183) - *Descarga dataset pre-preparado desde GitHub/Hugging Face*
- `setup_flickr8k` (line 257) - *Descarga y organiza Flickr8k - ahora con opción GitHub*
- `build_vocab_flickr` (line 366) - *Construye vocabulario desde el archivo de captions*
- `compute_tricameral_loss` (line 1048) - *Pérdida con término de coherencia audio-visual*
- `train_tricameral` (line 1100) - *Entrena el modelo tricameral

Args:
    github_repo_url: URL opcional del repo de GitHub/Hugging Face
                    Ejemplo: "https://github.com/user/flickr8k-prepared/raw/main"
                    o "https://huggingface.co/datasets/user/flickr8k-prepared/resolve/main"*
- `__init__` (line 390)
- `forward` (line 425)
- `hebbian_update` (line 438)
- `update_physiology_advanced` (line 476)
- `__init__` (line 513)
- `forward` (line 525)
- `_greedy_decode` (line 548)
- `_get_init_state` (line 570)
- `__init__` (line 587)
- `__len__` (line 625)
- `__getitem__` (line 628)
- `__init__` (line 681)
- `forward` (line 720) - *Args:
    mel_spec: (batch, 80, time)
Returns:
    audio_features: (batch, output_dim)*
- `__init__` (line 750)
- `forward` (line 784) - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre: Para Hebbian
    audio_post, audio_pre: Para Hebbian*
- `__init__` (line 824)
- `forward` (line 856) - *Args:
    right_features: (B, dim) - Fusión de visión + audio
Returns:
    enriched_context: (B, dim)
    channels: dict con 'visual', 'audio', 'semantic'*
- `__init__` (line 916)
- `forward` (line 951) - *Args:
    text_embedding: (B, text_dim)
Returns:
    audio_waveform: (B, 1, num_samples)*
- `__init__` (line 985)
- `forward` (line 1000) - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T) - Mel-spectrogram del caption
    captions: (B, seq_len) - Tokens (solo en train)
    generate_audio: bool - Si generar audio

Returns (training):
    logits, gates, losses, posts, pres, channels

Returns (inference):
    generated_text, generated_audio*
- `__init__` (line 1196)
- `__len__` (line 1214)
- `__getitem__` (line 1217)

#### `tricameral_kimi.py`
**Path:** `tricameral_kimi.py`

**Classs:**
- `EpisodicMemoryBuffer` (line 215)
- `NeurocognitiveSystem` (line 255)
- `LanguageMetrics` (line 331) - *Métricas de calidad de generación*
- `LinguisticFeedbackLoop` (line 405)
- `LanguageMetrics` (line 503)
- `StableLiquidNeuron` (line 550)
- `TriangulatedMedicalSystem` (line 673)
- `LeftHemisphere` (line 769)
- `AudioEncoder` (line 843) - *Encoder de audio usando Conv + Transformer*
- `RightHemisphereTricameral` (line 893) - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 965)
- `EnhancedDiagnosticsTricameral` (line 1067)
- `NeuroLogosTricameral` (line 1261) - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 1296) - *Dataset que carga imagen, audio del caption y texto desde Kaggle*

**Functions:**
- `setup_flickr8k_with_audio` (line 36) - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 191) - *Construye vocabulario desde el archivo de captions*
- `compute_tricameral_loss` (line 1402) - *Pérdida con término de coherencia audio-visual*
- `train_tricameral` (line 1452)
- `__init__` (line 216)
- `compute_surprise` (line 222)
- `add` (line 232)
- `sample` (line 241)
- `__init__` (line 256)
- `assess_cognitive_state` (line 266)
- `apply_cognitive_intervention` (line 287)
- `sentence_bleu` (line 335) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 369) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 378) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 391) - *Jaccard similarity entre palabras*
- `__init__` (line 406)
- `compute_linguistic_reward` (line 417)
- `compute_cider` (line 443)
- `compute_spice` (line 469)
- `_get_ngrams` (line 478)
- `get_cache_stats` (line 482)
- `sentence_bleu` (line 505)
- `token_accuracy` (line 528)
- `word_overlap` (line 538)
- `__init__` (line 551)
- `forward` (line 586)
- `hebbian_update` (line 599)
- `update_physiology_advanced` (line 639)
- `__init__` (line 674)
- `diagnose_with_triangulation` (line 680)
- `apply_triangulated_intervention` (line 703)
- `__init__` (line 770)
- `forward` (line 781)
- `_greedy_decode` (line 806)
- `_get_init_state` (line 828)
- `__init__` (line 846)
- `forward` (line 880)
- `__init__` (line 896)
- `forward` (line 929) - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre, audio_post, audio_pre: Para Hebbian*
- `__init__` (line 966)
- `forward` (line 995)
- `update_channel_fatigue` (line 1038)
- `adjust_gates_by_fatigue` (line 1054)
- `__init__` (line 1068)
- `measure_callosal_flow` (line 1084)
- `evaluate_reasoning_quality` (line 1103)
- `calculate_synergy` (line 1127)
- `calculate_health` (line 1137)
- `update` (line 1146)
- `get_recent_avg` (line 1156)
- `visualize_fatigue_distribution` (line 1173)
- `visualize_reasoning_metrics` (line 1191)
- `report` (line 1202)
- `__init__` (line 1264)
- `forward` (line 1270)
- `__init__` (line 1299)
- `__len__` (line 1347)
- `__getitem__` (line 1351)

#### `tricameral_kimi2.py`
**Path:** `tricameral_kimi2.py`

**Classs:**
- `EpisodicMemoryBuffer` (line 262)
- `NeurocognitiveSystem` (line 347)
- `LanguageMetrics` (line 539) - *Métricas de calidad de generación*
- `LinguisticFeedbackLoop` (line 613)
- `LanguageMetrics` (line 725)
- `LanguageMetrics` (line 774)
- `StableLiquidNeuron` (line 821)
- `TriangulatedMedicalSystem` (line 944)
- `LeftHemisphere` (line 1103)
- `AudioEncoder` (line 1401) - *Encoder de audio usando Conv + Transformer*
- `RightHemisphereTricameral` (line 1451) - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 1523)
- `EnhancedDiagnosticsTricameral` (line 1676)
- `NeuroLogosTricameral` (line 1952) - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 1987) - *Dataset que carga imagen, audio del caption y texto desde Kaggle*

**Functions:**
- `setup_flickr8k_with_audio` (line 50) - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 238) - *Construye vocabulario desde el archivo de captions*
- `compute_alignment_loss` (line 2091) - *FIX: Pérdida auxiliar para alineación temprana de canales multimodales
Solo activa en épocas iniciales (epoch < 6)*
- `compute_tricameral_loss` (line 2119)
- `train_tricameral` (line 2166)
- `__init__` (line 263)
- `compute_surprise` (line 278)
- `add` (line 289)
- `sample` (line 318)
- `__init__` (line 348)
- `assess_reasoning_state` (line 363) - *Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)*
- `assess_cognitive_state` (line 407) - *Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)*
- `apply_cognitive_intervention` (line 453) - *Aplica intervenciones basadas en estado lingüístico y de razonamiento*
- `sentence_bleu` (line 543) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 577) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 586) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 599) - *Jaccard similarity entre palabras*
- `__init__` (line 614)
- `compute_linguistic_reward` (line 636)
- `compute_cider` (line 675)
- `compute_spice` (line 695)
- `get_cache_stats` (line 704)
- `sentence_bleu` (line 727)
- `token_accuracy` (line 750)
- `word_overlap` (line 760)
- `sentence_bleu` (line 776)
- `token_accuracy` (line 799)
- `word_overlap` (line 809)
- `__init__` (line 822)
- `forward` (line 857)
- `hebbian_update` (line 870)
- `update_physiology_advanced` (line 910)
- `__init__` (line 945)
- `triangulate_signals` (line 951)
- `count_convergent_signals` (line 962)
- `diagnose_with_triangulation` (line 965)
- `apply_triangulated_intervention` (line 1018)
- `_reset_liquid_neuron` (line 1088) - *Reset completo de una neurona líquida*
- `__init__` (line 1104)
- `forward` (line 1179)
- `_apply_chain_of_thought` (line 1222)
- `_greedy_decode` (line 1262)
- `_apply_multi_token_prediction` (line 1323)
- `_apply_structural_attention` (line 1365)
- `_get_init_state` (line 1386)
- `__init__` (line 1404)
- `forward` (line 1438)
- `__init__` (line 1454)
- `forward` (line 1487) - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre, audio_post, audio_pre: Para Hebbian*
- `__init__` (line 1524)
- `forward` (line 1572)
- `update_channel_fatigue` (line 1633)
- `adjust_gates_by_fatigue` (line 1655)
- `__init__` (line 1677)
- `_get_cached_norm` (line 1699) - *Cache de normalización con limpieza periódica*
- `measure_callosal_flow` (line 1717)
- `evaluate_reasoning_quality` (line 1747)
- `calculate_synergy` (line 1784)
- `calculate_health` (line 1795)
- `update` (line 1804)
- `get_recent_avg` (line 1821)
- `visualize_fatigue_distribution` (line 1837)
- `visualize_reasoning_metrics` (line 1861)
- `report` (line 1873)
- `__init__` (line 1955)
- `forward` (line 1961)
- `__init__` (line 1990)
- `__len__` (line 2038)
- `__getitem__` (line 2042)
- `cached_ngrams` (line 623)

#### `tricameralkimi2.py`
**Path:** `tricameralkimi2.py`

**Classs:**
- `EpisodicMemoryBuffer` (line 195) - *Buffer que almacena ejemplos sorpresivos para replay estratégico*
- `NeurocognitiveSystem` (line 247)
- `LinguisticFeedbackLoop` (line 425) - *Sistema de caché optimizado para métricas lingüísticas*
- `StableLiquidNeuron` (line 546)
- `AudioEncoder` (line 668) - *Encoder de audio usando Conv + Transformer*
- `TriangulatedMedicalSystem` (line 724)
- `RightHemisphereTricameral` (line 881) - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 952) - *Corpus callosum con canales: visual, auditivo, semántico*
- `LeftHemisphere` (line 1089)
- `NeuroLogosTricameral` (line 1359) - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 1415) - *Dataset que carga imagen, audio del caption y texto desde Kaggle*
- `EnhancedDiagnosticsTricameral` (line 1556)

**Functions:**
- `setup_flickr8k_with_audio` (line 30) - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 173) - *Construye vocabulario desde el archivo de captions*
- `compute_tricameral_loss` (line 1507) - *Pérdida con término de coherencia audio-visual*
- `train_tricameral` (line 1786)
- `__init__` (line 198)
- `compute_surprise` (line 204) - *Calcula sorpresa basada en error y apertura del gate*
- `add` (line 216) - *Añade ejemplo si supera umbral y hay capacidad*
- `sample` (line 228) - *Samplea ejemplos con probabilidad proporcional a sorpresa*
- `__init__` (line 248)
- `assess_reasoning_state` (line 263) - *Evalúa estado del sistema de razonamiento*
- `assess_cognitive_state` (line 305) - *Evalúa estado cognitivo lingüístico*
- `apply_cognitive_intervention` (line 341) - *Aplica intervenciones basadas en estado lingüístico y razonamiento*
- `__init__` (line 428)
- `compute_linguistic_reward` (line 442) - *Recompensa combinada CIDEr + SPICE con caché*
- `compute_cider` (line 476) - *CIDEr simplificado con caché de n-gramas*
- `compute_spice` (line 505) - *SPICE simplificado (Jaccard similarity)*
- `_get_ngrams` (line 518) - *Extractor de n-gramas*
- `get_cache_stats` (line 523) - *Estadísticas de caché*
- `__init__` (line 547)
- `forward` (line 582)
- `hebbian_update` (line 595)
- `update_physiology_advanced` (line 633)
- `__init__` (line 671)
- `forward` (line 705) - *Args:
    mel_spec: (batch, 80, time)
Returns:
    audio_features: (batch, output_dim)*
- `__init__` (line 725)
- `triangulate_signals` (line 731)
- `count_convergent_signals` (line 741)
- `diagnose_with_triangulation` (line 744)
- `apply_triangulated_intervention` (line 786)
- `_reset_liquid_neuron` (line 868)
- `__init__` (line 884)
- `forward` (line 917) - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre, audio_post, audio_pre: Para Hebbian*
- `__init__` (line 955)
- `forward` (line 991) - *Args:
    right_features: (B, dim) - Fusión de visión + audio
Returns:
    enriched_context: (B, dim)
    channels: dict con 'visual', 'audio', 'semantic'*
- `update_channel_fatigue` (line 1060) - *Actualiza fatiga específica por canal*
- `adjust_gates_by_fatigue` (line 1079) - *Ajusta proyecciones basado en fatiga*
- `__init__` (line 1090)
- `forward` (line 1165)
- `_greedy_decode` (line 1204)
- `_apply_chain_of_thought` (line 1237)
- `_apply_multi_token_prediction` (line 1271)
- `_apply_structural_attention` (line 1317) - *Atenuación simple según fatiga de canal + atención cruzada visual.
Se ignoran los canales 'objects/actions/scene' que no existen.*
- `_get_init_state` (line 1344)
- `__init__` (line 1362)
- `forward` (line 1374) - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T) - Mel-spectrogram del caption
    captions: (B, seq_len) - Tokens (solo en train)

Returns (training):
    logits, gates, losses, posts, pres, channels

Returns (inference):
    generated_text*
- `__init__` (line 1418)
- `__len__` (line 1463)
- `__getitem__` (line 1466)
- `__init__` (line 1557)
- `measure_callosal_flow` (line 1572)
- `evaluate_reasoning_quality` (line 1595) - *Evalúa coherencia y consistencia del razonamiento*
- `calculate_synergy` (line 1627)
- `calculate_health` (line 1638)
- `update` (line 1647)
- `get_recent_avg` (line 1657)
- `visualize_fatigue_distribution` (line 1674)
- `visualize_reasoning_metrics` (line 1695)
- `report` (line 1707) - *Genera reporte completo del estado del sistema tricameral*

#### `trycameral.py`
**Path:** `trycameral.py`

**Classs:**
- `StableLiquidNeuron` (line 228)
- `LeftHemisphere` (line 351)
- `Flickr8kMultimodalDataset` (line 423) - *Dataset que carga imagen, audio del caption y texto*
- `AudioEncoder` (line 517) - *Encoder de audio usando Conv + Transformer*
- `RightHemisphereTricameral` (line 586) - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 660) - *Corpus callosum con canales: visual, auditivo, semántico*
- `NeuralAudioGenerator` (line 752) - *Generador de audio desde embeddings lingüísticos (TTS neuronal)*
- `NeuroLogosTricameral` (line 821) - *Arquitectura completa: Visión + Audio -> Lenguaje + Audio*
- `Flickr8kSimpleDataset` (line 996)

**Functions:**
- `generate_audio_async` (line 31) - *Genera un audio usando Edge-TTS*
- `generate_all_audios_batch` (line 50) - *Genera todos los audios en batches para eficiencia*
- `generate_audios_sync` (line 89) - *Wrapper síncrono para generar audios*
- `setup_flickr8k` (line 133) - *Descarga y organiza Flickr8k si no existe*
- `build_vocab_flickr` (line 205) - *Construye vocabulario desde el archivo de captions*
- `compute_tricameral_loss` (line 887) - *Pérdida con término de coherencia audio-visual*
- `train_tricameral` (line 939)
- `__init__` (line 229)
- `forward` (line 264)
- `hebbian_update` (line 277)
- `update_physiology_advanced` (line 315)
- `__init__` (line 352)
- `forward` (line 364)
- `_greedy_decode` (line 387)
- `_get_init_state` (line 409)
- `__init__` (line 426)
- `__len__` (line 464)
- `__getitem__` (line 467)
- `__init__` (line 520)
- `forward` (line 559) - *Args:
    mel_spec: (batch, 80, time)
Returns:
    audio_features: (batch, output_dim)*
- `__init__` (line 589)
- `forward` (line 623) - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre: Para Hebbian
    audio_post, audio_pre: Para Hebbian*
- `__init__` (line 663)
- `forward` (line 695) - *Args:
    right_features: (B, dim) - Fusión de visión + audio
Returns:
    enriched_context: (B, dim)
    channels: dict con 'visual', 'audio', 'semantic'*
- `__init__` (line 755)
- `forward` (line 790) - *Args:
    text_embedding: (B, text_dim)
Returns:
    audio_waveform: (B, 1, num_samples)*
- `__init__` (line 824)
- `forward` (line 839) - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T) - Mel-spectrogram del caption
    captions: (B, seq_len) - Tokens (solo en train)
    generate_audio: bool - Si generar audio

Returns (training):
    logits, gates, losses, posts, pres, channels

Returns (inference):
    generated_text, generated_audio*
- `__init__` (line 997)
- `__len__` (line 1015)
- `__getitem__` (line 1018)

#### `ultimo_neuorlogos.py`
**Path:** `ultimo_neuorlogos.py`

**Classs:**
- `TopoBrainCore` (line 27) - *TopoBrain validado con ablation (de tu experimento anterior)*
- `PGDAttack` (line 113) - *Adversarial attack para robustez (solo en TOPO-FULL)*
- `MiniUnconscious` (line 145) - *Baseline: Encoder visual simple*
- `TopoUnconscious` (line 166) - *TopoBrain-enhanced visual encoder*
- `ConsciousCore` (line 203) - *Núcleo consciente con atención*
- `BioDecoder` (line 221) - *Decoder LSTM con gating*
- `NeuroLogos` (line 282) - *Configuraciones del ablation:
- mode='baseline': MiniUnconscious (sin TopoBrain)
- mode='topo-light': TopoBrain sin symbiotic
- mode='topo-full': TopoBrain completo + adversarial*
- `CIFARCaptions` (line 330)

**Functions:**
- `train_ablation` (line 374) - *Entrena un modelo en el modo especificado*
- `run_full_ablation` (line 508) - *Ejecuta ablation study completo de 3 niveles*
- `__init__` (line 29)
- `_init_grid` (line 62) - *Inicializa coordenadas del grid 2D*
- `forward` (line 70)
- `get_metrics` (line 101) - *Retorna métricas de topología*
- `__init__` (line 115)
- `attack` (line 120) - *Genera ejemplos adversariales*
- `__init__` (line 147)
- `forward` (line 160)
- `__init__` (line 168)
- `forward` (line 191)
- `get_metrics` (line 195)
- `__init__` (line 205)
- `forward` (line 210)
- `__init__` (line 223)
- `forward` (line 238)
- `_get_init_state` (line 272)
- `__init__` (line 289)
- `forward` (line 314)
- `get_metrics` (line 319) - *Obtiene métricas de topología si disponible*
- `__init__` (line 331)
- `__len__` (line 356)
- `__getitem__` (line 359)

#### `ultimobicameral.py`
**Path:** `ultimobicameral.py`

**Classs:**
- `LanguageMetrics` (line 23) - *Métricas de calidad de generación*
- `MedicalSystem` (line 96) - *Sistema de intervención médica por niveles*
- `StableLiquidNeuron` (line 292)
- `RightHemisphere` (line 368)
- `LeftHemisphere` (line 383)
- `CorpusCallosum` (line 443)
- `NeuroLogosBicameralStable` (line 462)
- `EnhancedDiagnostics` (line 483)
- `Flickr8kDataset` (line 608)

**Functions:**
- `build_vocab_flickr` (line 645)
- `setup_flickr8k` (line 663)
- `train_with_metrics` (line 677)
- `sentence_bleu` (line 27) - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 61) - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 70) - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 83) - *Jaccard similarity entre palabras*
- `__init__` (line 99)
- `diagnose_severity` (line 103) - *Diagnosticar gravedad del problema con análisis mejorado*
- `apply_intervention` (line 149) - *Aplicar intervención médica calibrada con más agresividad en gate*
- `__init__` (line 293)
- `forward` (line 307)
- `hebbian_update` (line 314)
- `update_physiology_advanced` (line 344)
- `__init__` (line 369)
- `forward` (line 377)
- `__init__` (line 384)
- `forward` (line 401)
- `_get_init_state` (line 438)
- `__init__` (line 444)
- `forward` (line 456)
- `__init__` (line 463)
- `forward` (line 469)
- `__init__` (line 484)
- `measure_callosal_flow` (line 494)
- `calculate_synergy` (line 503)
- `calculate_health` (line 512)
- `update` (line 521)
- `get_recent_avg` (line 526)
- `report` (line 531)
- `__init__` (line 609)
- `__len__` (line 626)
- `__getitem__` (line 629)

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*

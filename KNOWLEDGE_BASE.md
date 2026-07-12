# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 190 | **Total Symbols Extracted:** 5935 | **Total Imports:** 2457

## Structural Knowledge Map
> **Note:** The visual graph below has been intelligently pruned to the top 300 most relevant nodes to prevent rendering crashes. Full details of all 190 files are documented below.

```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
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
- `mcculloch_pitts_neuron` (line 7) `def mcculloch_pitts_neuron(inputs, weights, threshold)` - *Neurona artificial de McCulloch-Pitts (1943).
- inputs: vector binario de entrada (0 o 1)
- weights: vector de pesos sinápticos
- threshold: valor umbral de activación
Retorna 1 si la suma ponderada >= threshold; 0 en caso contrario.*

#### `01_topobrain_cou_v2.py`
**Path:** `01_topobrain_cou_v2.py`

**Classes:**
- `Config` (line 25) `class Config`
- `StableSupConLoss` (line 116) `class StableSupConLoss` - *Versión estable de SupConLoss con manejo de bordes robusto*
- `StableContinuumMemoryCell` (line 147) `class StableContinuumMemoryCell` - *Versión estable y ligera de ContinuumMemoryCell con clamping y normalización*
- `StableSymbioticBasisRefinement` (line 231) `class StableSymbioticBasisRefinement` - *Refinamiento simbiótico estable con regularización explícita (versión ligera)*
- `StableTopologyManager` (line 267) `class StableTopologyManager` - *Gestor de topología con protocolos de supervivencia (versión optimizada para CPU)*
- `StableTopoBrain` (line 345) `class StableTopoBrain` - *Implementación estable y ligera de TopoBrain para ablation científico en CPU*

**Functions:**
- `seed_everything` (line 70) `def seed_everything(seed)`
- `get_tabular_loaders` (line 77) `def get_tabular_loaders(config)` - *Dataset tabular controlado con características NOIR simuladas*
- `stable_pgd_attack` (line 477) `def stable_pgd_attack(model, x, y, eps, steps, controls)` - *Ataque PGD estable con manejo de gradientes robusto*
- `train_epoch` (line 520) `def train_epoch(model, loader, optimizer, config, epoch, controls)` - *Entrenamiento por época con monitoreo detallado*
- `evaluate_model` (line 584) `def evaluate_model(model, loader, config, adversarial, controls)` - *Evaluación rigurosa con o sin ataques adversariales*
- `run_scientific_ablation` (line 632) `def run_scientific_ablation()` - *Ejecución científica del ablation con control de variables*
- `to_dict` (line 56) `def to_dict(self)`
- `get_topology_config` (line 59) `def get_topology_config(self)` - *Configuración estable para topología adaptable*
- `__init__` (line 118) `def __init__(self, temperature)`
- `forward` (line 123) `def forward(self, features, labels)`
- `__init__` (line 149) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 181) `def forward(self, x, controls)`
- `__init__` (line 233) `def __init__(self, dim, num_atoms)`
- `forward` (line 242) `def forward(self, x)`
- `__init__` (line 269) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 288) `def get_adjacency(self, plasticity)` - *Obtener matriz de adyacencia con estabilidad garantizada*
- `prune_topology` (line 317) `def prune_topology(self, current_density, epoch)` - *Poda controlada con protocolo de emergencia*
- `get_density` (line 337) `def get_density(self)` - *Calcular densidad actual de manera estable*
- `__init__` (line 347) `def __init__(self, config)`
- `_init_weights` (line 400) `def _init_weights(self)` - *Inicialización estable de pesos*
- `forward` (line 408) `def forward(self, x, controls)`

#### `01_topobrain_cpu.py`
**Path:** `01_topobrain_cpu.py`

**Classes:**
- `Config` (line 21) `class Config`
- `SupConLoss` (line 94) `class SupConLoss`
- `ContinuumMemoryCell` (line 109) `class ContinuumMemoryCell`
- `SymbioticBasisRefinement` (line 143) `class SymbioticBasisRefinement`
- `TopoBrainTabular` (line 166) `class TopoBrainTabular`

**Functions:**
- `seed_everything` (line 63) `def seed_everything(seed)`
- `get_tabular_loaders` (line 69) `def get_tabular_loaders(config)`
- `pgd_attack` (line 247) `def pgd_attack(model, x, y, eps, steps, controls)`
- `generate_ablation_configs` (line 263) `def generate_ablation_configs(base_config)`
- `train_and_evaluate` (line 293) `def train_and_evaluate(config, name)`
- `run_ablation` (line 337) `def run_ablation()`
- `to_dict` (line 57) `def to_dict(self)`
- `__init__` (line 95) `def __init__(self, temperature)`
- `forward` (line 98) `def forward(self, features, labels)`
- `__init__` (line 110) `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate)`
- `forward` (line 124) `def forward(self, x, controls)`
- `__init__` (line 144) `def __init__(self, dim)`
- `forward` (line 151) `def forward(self, x)`
- `__init__` (line 167) `def __init__(self, config)`
- `get_adj` (line 198) `def get_adj(self)`
- `forward` (line 203) `def forward(self, x, controls)`
- `evaluate_adv` (line 323) `def evaluate_adv(loader, eps, steps)`

#### `01_topobrain_cpu_v3.py`
**Path:** `01_topobrain_cpu_v3.py`

**Classes:**
- `Config` (line 24) `class Config`
- `SupConLoss` (line 93) `class SupConLoss`
- `ContinuumMemoryCell` (line 108) `class ContinuumMemoryCell`
- `SymbioticBasisRefinement` (line 145) `class SymbioticBasisRefinement`
- `PrefrontalOrchestrator` (line 165) `class PrefrontalOrchestrator`
- `AdaptiveCombinatorialComplexLayer` (line 227) `class AdaptiveCombinatorialComplexLayer`
- `TopoBrainTabular` (line 320) `class TopoBrainTabular`

**Functions:**
- `seed_everything` (line 63) `def seed_everything(seed)`
- `get_tabular_loaders` (line 69) `def get_tabular_loaders(config)`
- `pgd_attack` (line 395) `def pgd_attack(model, x, y, eps, steps)`
- `compute_topology_metrics` (line 416) `def compute_topology_metrics(model, config)` - *Computar métricas de topología con manejo robusto de errores*
- `prune_topology` (line 450) `def prune_topology(model, config, controls)` - *Implementación simplificada de poda de topología*
- `train_and_evaluate` (line 482) `def train_and_evaluate(config, run_name)`
- `run_ablation` (line 640) `def run_ablation()`
- `to_dict` (line 57) `def to_dict(self)`
- `__init__` (line 94) `def __init__(self, temperature)`
- `forward` (line 97) `def forward(self, features, labels)`
- `__init__` (line 109) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 122) `def forward(self, x, controls)`
- `__init__` (line 146) `def __init__(self, dim)`
- `forward` (line 153) `def forward(self, x)`
- `__init__` (line 166) `def __init__(self, config)`
- `forward` (line 183) `def forward(self, metrics)`
- `reset_context` (line 221) `def reset_context(self)`
- `__init__` (line 228) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- `get_adj` (line 264) `def get_adj(self)`
- `forward` (line 269) `def forward(self, x, controls)`
- `__init__` (line 321) `def __init__(self, config)`
- `_initialize_memories` (line 352) `def _initialize_memories(self)` - *Inicialización de memorias semánticas*
- `forward` (line 363) `def forward(self, x, controls, prev_states)`
- `evaluate_adv` (line 593) `def evaluate_adv(loader, eps, steps)`

#### `01_topobrain_cpu_v4.py`
**Path:** `01_topobrain_cpu_v4.py`

**Classes:**
- `Config` (line 25) `class Config`
- `StableSupConLoss` (line 116) `class StableSupConLoss` - *Versión estable de SupConLoss con manejo de bordes robusto*
- `StableContinuumMemoryCell` (line 147) `class StableContinuumMemoryCell` - *Versión estable y ligera de ContinuumMemoryCell con clamping y normalización*
- `StableSymbioticBasisRefinement` (line 231) `class StableSymbioticBasisRefinement` - *Refinamiento simbiótico estable con regularización explícita (versión ligera)*
- `StableTopologyManager` (line 267) `class StableTopologyManager` - *Gestor de topología con protocolos de supervivencia (versión optimizada para CPU)*
- `StableTopoBrain` (line 345) `class StableTopoBrain` - *Implementación estable y ligera de TopoBrain para ablation científico en CPU*

**Functions:**
- `seed_everything` (line 70) `def seed_everything(seed)`
- `get_tabular_loaders` (line 77) `def get_tabular_loaders(config)` - *Dataset tabular controlado con características NOIR simuladas*
- `stable_pgd_attack` (line 475) `def stable_pgd_attack(model, x, y, eps, steps, controls)` - *Ataque PGD estable con manejo de gradientes robusto*
- `train_epoch` (line 518) `def train_epoch(model, loader, optimizer, config, epoch, controls)` - *Entrenamiento por época con monitoreo detallado*
- `evaluate_model` (line 583) `def evaluate_model(model, loader, config, adversarial, controls)` - *Evaluación rigurosa con o sin ataques adversariales*
- `run_scientific_ablation` (line 628) `def run_scientific_ablation()` - *Ejecución científica del ablation con control de variables*
- `to_dict` (line 56) `def to_dict(self)`
- `get_topology_config` (line 59) `def get_topology_config(self)` - *Configuración estable para topología adaptable*
- `__init__` (line 118) `def __init__(self, temperature)`
- `forward` (line 123) `def forward(self, features, labels)`
- `__init__` (line 149) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 181) `def forward(self, x, controls)`
- `__init__` (line 233) `def __init__(self, dim, num_atoms)`
- `forward` (line 242) `def forward(self, x)`
- `__init__` (line 269) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 288) `def get_adjacency(self, plasticity)` - *Obtener matriz de adyacencia con estabilidad garantizada*
- `prune_topology` (line 317) `def prune_topology(self, current_density, epoch)` - *Poda controlada con protocolo de emergencia*
- `get_density` (line 337) `def get_density(self)` - *Calcular densidad actual de manera estable*
- `__init__` (line 347) `def __init__(self, config)`
- `_init_weights` (line 400) `def _init_weights(self)` - *Inicialización estable de pesos*
- `forward` (line 408) `def forward(self, x, controls)`

#### `01_topobrain_cpu_v5.py`
**Path:** `01_topobrain_cpu_v5.py`

**Classes:**
- `Config` (line 25) `class Config`
- `StableSupConLoss` (line 116) `class StableSupConLoss` - *Versión estable de SupConLoss con manejo de bordes robusto*
- `StableContinuumMemoryCell` (line 147) `class StableContinuumMemoryCell` - *Versión estable y ligera de ContinuumMemoryCell con clamping y normalización*
- `StableSymbioticBasisRefinement` (line 231) `class StableSymbioticBasisRefinement` - *Refinamiento simbiótico estable con regularización explícita (versión ligera)*
- `StableTopologyManager` (line 267) `class StableTopologyManager` - *Gestor de topología con protocolos de supervivencia (versión optimizada para CPU)*
- `StableTopoBrain` (line 345) `class StableTopoBrain` - *Implementación estable y ligera de TopoBrain para ablation científico en CPU*

**Functions:**
- `seed_everything` (line 70) `def seed_everything(seed)`
- `get_tabular_loaders` (line 77) `def get_tabular_loaders(config)` - *Dataset tabular controlado con características NOIR simuladas*
- `stable_pgd_attack` (line 475) `def stable_pgd_attack(model, x, y, eps, steps, controls)` - *Ataque PGD estable con manejo de gradientes robusto*
- `train_epoch` (line 518) `def train_epoch(model, loader, optimizer, config, epoch, controls)` - *Entrenamiento por época con monitoreo detallado*
- `evaluate_model` (line 583) `def evaluate_model(model, loader, config, adversarial, controls)` - *Evaluación rigurosa con o sin ataques adversariales*
- `run_scientific_ablation` (line 708) `def run_scientific_ablation()` - *Ejecución científica del ablation con control de variables*
- `to_dict` (line 56) `def to_dict(self)`
- `get_topology_config` (line 59) `def get_topology_config(self)` - *Configuración estable para topología adaptable*
- `__init__` (line 118) `def __init__(self, temperature)`
- `forward` (line 123) `def forward(self, features, labels)`
- `__init__` (line 149) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 181) `def forward(self, x, controls)`
- `__init__` (line 233) `def __init__(self, dim, num_atoms)`
- `forward` (line 242) `def forward(self, x)`
- `__init__` (line 269) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 288) `def get_adjacency(self, plasticity)` - *Obtener matriz de adyacencia con estabilidad garantizada*
- `prune_topology` (line 317) `def prune_topology(self, current_density, epoch)` - *Poda controlada con protocolo de emergencia*
- `get_density` (line 337) `def get_density(self)` - *Calcular densidad actual de manera estable*
- `__init__` (line 347) `def __init__(self, config)`
- `_init_weights` (line 400) `def _init_weights(self)` - *Inicialización estable de pesos*
- `forward` (line 408) `def forward(self, x, controls)`

#### `01_topobrain_cpu_v6.py`
**Path:** `01_topobrain_cpu_v6.py`

**Classes:**
- `MicroConfig` (line 33) `class MicroConfig` - *Configuración ultra-ligera para CPU*
- `MicroSupConLoss` (line 137) `class MicroSupConLoss` - *SupCon ultra-eficiente con estabilidad numérica*
- `MicroContinuumCell` (line 163) `class MicroContinuumCell` - *ContinuumMemoryCell ultra-compacto con predicción semántica.
Params: ~4*4 (W_slow) + 4*4 (V_slow) + 4*4 (semantic_mem) ≈ 48 params*
- `MicroSymbioticBasis` (line 214) `class MicroSymbioticBasis` - *Refinamiento simbiótico con 2 átomos de base.
Params: 2*dim (basis) + dim*dim (Q) + dim*dim (K) ≈ 2*dim + 2*dim² = 40 params (dim=4)*
- `MicroTopology` (line 253) `class MicroTopology` - *Topología dinámica para Grid 2x2 (4 nodos).
Params: 4x4 = 16 (matriz de adyacencia aprendible)*
- `MicroTopoBrain` (line 286) `class MicroTopoBrain` - *TopoBrain ultra-ligero con matemática completa.

PRESUPUESTO DE PARÁMETROS:
- Input embed: 12*16 = 192
- Topology: 4*4 = 16
- Node processor (ContinuumCell si activo): ~48
- Cell processor (MGF si activo): ~48
- Symbiotic (si activo): ~40
- SupCon head (si activo): 16*8 + 8*4 = 160
- Readout: 16*3 = 48

TOTAL: ~200-550 params (base) hasta ~3k-5k (full)*
- `AblationMatrix` (line 597) `class AblationMatrix` - *Matriz de ablación científica con 3 niveles de profundidad.

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
- `ScientificAnalyzer` (line 744) `class ScientificAnalyzer` - *Análisis estadístico con significancia y tamaño de efecto*

**Functions:**
- `seed_everything` (line 95) `def seed_everything(seed)` - *Control de reproducibilidad*
- `get_micro_dataset` (line 104) `def get_micro_dataset(config)` - *Dataset tabular controlado con validación cruzada*
- `compute_effect_size` (line 126) `def compute_effect_size(group1, group2)` - *Cohen's d para medir tamaño del efecto*
- `micro_pgd_attack` (line 432) `def micro_pgd_attack(model, x, y, eps, steps, plasticity)` - *PGD ultra-eficiente para CPU*
- `train_epoch_micro` (line 462) `def train_epoch_micro(model, loader, optimizer, config, epoch)` - *Entrenamiento por época*
- `evaluate_micro` (line 517) `def evaluate_micro(model, loader, config, adversarial)` - *Evaluación con opción adversarial*
- `train_with_cv` (line 539) `def train_with_cv(config, dataset, cv_folds)` - *Entrenamiento con validación cruzada estratificada.
Retorna: lista de resultados por fold.*
- `run_scientific_ablation_study` (line 850) `def run_scientific_ablation_study()` - *Ejecutor completo del estudio de ablación con análisis científico.

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
- `to_dict` (line 79) `def to_dict(self)`
- `component_signature` (line 82) `def component_signature(self)` - *Firma única de componentes activos*
- `__init__` (line 139) `def __init__(self, temperature)`
- `forward` (line 144) `def forward(self, features, labels)`
- `__init__` (line 168) `def __init__(self, dim)`
- `forward` (line 184) `def forward(self, x, plasticity)`
- `__init__` (line 219) `def __init__(self, dim, num_atoms)`
- `forward` (line 232) `def forward(self, x)`
- `__init__` (line 258) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 276) `def get_adjacency(self, plasticity)` - *Matriz de adyacencia normalizada*
- `__init__` (line 301) `def __init__(self, config)`
- `_init_weights` (line 353) `def _init_weights(self)`
- `count_parameters` (line 360) `def count_parameters(self)` - *Contar parámetros entrenables*
- `forward` (line 364) `def forward(self, x, plasticity)`
- `level1_isolated` (line 628) `def level1_isolated()` - *NIVEL 1: Componentes aislados (6 experimentos)*
- `level2a_pairs` (line 641) `def level2a_pairs()` - *NIVEL 2A: Todos los pares (10 experimentos = C(5,2))*
- `level2b_strategic_triads` (line 665) `def level2b_strategic_triads()` - *NIVEL 2B: Tríadas estratégicas (8 experimentos selectos)

HIPÓTESIS BASADAS EN ANÁLISIS PREVIO:
1. Plasticity es fuerte SOLO → probar con 1 componente adicional
2. Continuum + Symbiotic mostraron cooperación → añadir tercero
3. Evitar SupCon en combinaciones (causa antagonismo)*
- `level3_inverse_ablation` (line 694) `def level3_inverse_ablation()` - *NIVEL 3: Ablación inversa (5 experimentos)
Modelo completo MENOS un componente → detecta criticidad*
- `level3_full_model` (line 719) `def level3_full_model()` - *Modelo completo (referencia máxima)*
- `get_complete_matrix` (line 730) `def get_complete_matrix(cls)` - *Matriz completa de ablación (30 experimentos)*
- `compute_statistics` (line 748) `def compute_statistics(results_list)` - *Análisis estadístico por experimento.

Args:
    results_list: Lista de resultados de CV folds

Returns:
    Dict con mean, std, CI95, etc.*
- `ttest_vs_baseline` (line 774) `def ttest_vs_baseline(exp_scores, baseline_scores)` - *t-test pareado vs baseline.

Returns:
    (t_statistic, p_value, cohens_d)*
- `detect_synergy` (line 786) `def detect_synergy(pair_pgd, comp_a_pgd, comp_b_pgd, baseline_pgd)` - *Detecta sinergia no-lineal.

Synergy = PGD(A+B) - [PGD(A) + PGD(B) - PGD(Baseline)]

> +5%  → Cooperación fuerte
-5~+5% → Aditivo
< -5%  → Antagonismo*
- `rank_components_by_criticality` (line 809) `def rank_components_by_criticality(full_pgd, ablation_results)` - *Ranking de criticidad basado en ablación inversa.

Criticality = PGD(Full) - PGD(Full_Without_X)

> +10%  → ESENCIAL
+5~+10% → IMPORTANTE
-5~+5%  → OPCIONAL
< -5%   → PERJUDICIAL*

#### `01_topobrain_cpu_v7.py`
**Path:** `01_topobrain_cpu_v7.py`

**Classes:**
- `Config` (line 42) `class Config`
- `SymbioticBasis` (line 94) `class SymbioticBasis`
- `DynamicTopology` (line 124) `class DynamicTopology`
- `TopoBrainCPU` (line 186) `class TopoBrainCPU`

**Functions:**
- `setup_device` (line 24) `def setup_device()`
- `pgd_attack` (line 256) `def pgd_attack(model, x, y, eps, steps, plasticity)`
- `train_epoch` (line 285) `def train_epoch(model, loader, optimizer, config, epoch, device)`
- `evaluate` (line 321) `def evaluate(model, loader, config, device, adversarial)`
- `get_dataset` (line 351) `def get_dataset(config)`
- `main` (line 393) `def main()`
- `to_dict` (line 86) `def to_dict(self)`
- `__init__` (line 95) `def __init__(self, dim, num_atoms)`
- `forward` (line 106) `def forward(self, x)`
- `__init__` (line 125) `def __init__(self, num_nodes, grid_size, config)`
- `_create_grid_mask` (line 136) `def _create_grid_mask(self)`
- `get_adjacency` (line 153) `def get_adjacency(self, plasticity)`
- `prune_connections` (line 158) `def prune_connections(self, threshold)`
- `get_density` (line 178) `def get_density(self)`
- `__init__` (line 187) `def __init__(self, config)`
- `_init_weights` (line 216) `def _init_weights(self)`
- `count_parameters` (line 223) `def count_parameters(self)`
- `forward` (line 226) `def forward(self, x, plasticity)`

#### `01_topobrain_cpu_v8.py`
**Path:** `01_topobrain_cpu_v8.py`

**Classes:**
- `Config` (line 28) `class Config`
- `SymbioticBasis` (line 62) `class SymbioticBasis`
- `DynamicTopology` (line 88) `class DynamicTopology`
- `TopoBrainReal` (line 144) `class TopoBrainReal`
- `Wrapper` (line 424) `class Wrapper`

**Functions:**
- `pgd_attack` (line 223) `def pgd_attack(model, x, y, eps, steps, plasticity)`
- `train_topobrain` (line 252) `def train_topobrain(config)`
- `export_for_onnxruntime` (line 410) `def export_for_onnxruntime(model)` - *Exporta para ONNX Runtime (más moderno que OpenCV)*
- `test_with_onnxruntime` (line 465) `def test_with_onnxruntime(X_test, y_test)` - *Inferencia usando ONNX Runtime*
- `main` (line 547) `def main()`
- `__init__` (line 63) `def __init__(self, dim, num_atoms)`
- `forward` (line 76) `def forward(self, x)`
- `__init__` (line 89) `def __init__(self, num_nodes, grid_size, config)`
- `_create_grid_mask` (line 98) `def _create_grid_mask(self)`
- `get_adjacency` (line 115) `def get_adjacency(self, plasticity)`
- `prune_connections` (line 120) `def prune_connections(self, threshold)`
- `get_density` (line 140) `def get_density(self)`
- `__init__` (line 145) `def __init__(self, config)`
- `_init_weights` (line 174) `def _init_weights(self)`
- `forward` (line 181) `def forward(self, x, plasticity)`
- `forward_with_metrics` (line 210) `def forward_with_metrics(self, x, plasticity)`
- `__init__` (line 425) `def __init__(self, m)`
- `forward` (line 429) `def forward(self, x)`

#### `01_topobrain_ganador_gpu_v1.py`
**Path:** `01_topobrain_ganador_gpu_v1.py`

**Classes:**
- `GPUConfig` (line 70) `class GPUConfig` - *Configuración optimizada para AMD GPU basada en estudio científico*
- `GPUSymbioticBasis` (line 127) `class GPUSymbioticBasis` - *Symbiotic Basis escalado para GPU.
Proyección ortogonal con más átomos para mayor capacidad.*
- `DynamicTopology` (line 179) `class DynamicTopology` - *Topología dinámica adaptativa para Grid 8x8.
Aprende qué conexiones mantener/podar durante entrenamiento.*
- `TopoBrainGPU` (line 278) `class TopoBrainGPU` - *TopoBrain escalado para GPU AMD con configuración ganadora.

ARQUITECTURA:
- Grid 8x8 (64 nodos)
- Embed dim: 16
- Plasticity: Topología adaptativa
- Symbiotic: Refinamiento ortogonal
- NO Continuum, NO MGF, NO SupCon (según estudio)

PARÁMETROS ESTIMADOS: ~50k-80k*

**Functions:**
- `setup_amd_device` (line 31) `def setup_amd_device()` - *Configura PyTorch para usar GPU AMD con ROCm/OpenCL.
Fallback a CPU si no está disponible.*
- `pgd_attack_gpu` (line 405) `def pgd_attack_gpu(model, x, y, eps, steps, plasticity)` - *PGD attack optimizado para GPU.

Args:
    model: Modelo TopoBrain
    x: [batch, features]
    y: [batch] - Labels
    eps: Perturbación máxima
    steps: Pasos de iteración
    plasticity: Factor para topología

Returns:
    x_adv: [batch, features] - Ejemplos adversariales*
- `train_epoch_gpu` (line 461) `def train_epoch_gpu(model, loader, optimizer, config, epoch, device)` - *Entrenamiento por época en GPU*
- `evaluate_gpu` (line 517) `def evaluate_gpu(model, loader, config, device, adversarial)` - *Evaluación en GPU*
- `get_gpu_dataset` (line 547) `def get_gpu_dataset(config)` - *Dataset sintético para GPU*
- `main` (line 591) `def main()` - *Ejecutar POC completa*
- `to_dict` (line 119) `def to_dict(self)`
- `__init__` (line 132) `def __init__(self, dim, num_atoms)`
- `forward` (line 147) `def forward(self, x)` - *Args:
    x: [batch, dim]
Returns:
    x_clean: [batch, dim] - Proyección limpia
    entropy: scalar - Entropía de pesos
    ortho: scalar - Pérdida de ortogonalidad*
- `__init__` (line 184) `def __init__(self, num_nodes, grid_size, config)`
- `_create_grid_mask` (line 200) `def _create_grid_mask(self)` - *Crea máscara de vecindad para grid NxN*
- `get_adjacency` (line 219) `def get_adjacency(self, plasticity)` - *Obtiene matriz de adyacencia normalizada.

Args:
    plasticity: Factor de modulación (0=frozen, 1=full adaptive)
Returns:
    adj: [num_nodes, num_nodes] - Matriz normalizada por grado*
- `prune_connections` (line 237) `def prune_connections(self, threshold)` - *Poda conexiones débiles (llamar cada N epochs).

Args:
    threshold: Umbral de poda (sigmoid(weight) < threshold)
Returns:
    num_pruned: Número de conexiones podadas*
- `get_density` (line 269) `def get_density(self)` - *Densidad actual de conexiones*
- `__init__` (line 291) `def __init__(self, config)`
- `_init_weights` (line 335) `def _init_weights(self)` - *Inicialización Kaiming para activaciones GELU*
- `count_parameters` (line 343) `def count_parameters(self)` - *Cuenta parámetros entrenables*
- `forward` (line 347) `def forward(self, x, plasticity)` - *Forward pass completo.

Args:
    x: [batch, n_features]
    plasticity: Factor de adaptación topológica (0-1)

Returns:
    logits: [batch, n_classes]
    entropy: scalar - Entropía de Symbiotic
    ortho: scalar - Regularización ortogonal*

#### `02_perceptron.py`
**Path:** `02_perceptron.py`

**Classes:**
- `Perceptron` (line 31) `class Perceptron`

**Functions:**
- `__init__` (line 32) `def __init__(self, input_dim, learning_rate)`
- `predict` (line 37) `def predict(self, X)`
- `train_step` (line 42) `def train_step(self, X_batch, y_batch)`
- `accuracy` (line 51) `def accuracy(self, X, y_true)`

#### `03_backpropagation.py`
**Path:** `03_backpropagation.py`

**Functions:**
- `sigmoid` (line 33) `def sigmoid(z)`
- `sigmoid_derivative` (line 38) `def sigmoid_derivative(z)`
- `forward` (line 53) `def forward(X)`
- `backward` (line 63) `def backward(X, y_true, y_pred, a1, z1, lr)`
- `compute_metrics` (line 88) `def compute_metrics(y_pred, y_true)`

#### `04_cnn_lenet.py`
**Path:** `04_cnn_lenet.py`

**Classes:**
- `LeNet5Like` (line 45) `class LeNet5Like`

**Functions:**
- `__init__` (line 46) `def __init__(self)`
- `forward` (line 60) `def forward(self, x)`

#### `05_svm_rbf.py`
**Path:** `05_svm_rbf.py`

*No symbols extracted*

#### `06_lstm_char.py`
**Path:** `06_lstm_char.py`

**Classes:**
- `CharLSTM` (line 89) `class CharLSTM`

**Functions:**
- `create_batches` (line 63) `def create_batches(data, batch_size, seq_length)`
- `__init__` (line 90) `def __init__(self, vocab_size, hidden_size, num_layers, dropout)`
- `forward` (line 102) `def forward(self, x, hidden)`

#### `07_random_forest.py`
**Path:** `07_random_forest.py`

*No symbols extracted*

#### `08_vae_mnist.py`
**Path:** `08_vae_mnist.py`

**Classes:**
- `VAE` (line 40) `class VAE`

**Functions:**
- `vae_loss` (line 72) `def vae_loss(recon_x, x, mu, log_var)`
- `__init__` (line 41) `def __init__(self, input_dim, hidden_dim, latent_dim)`
- `encode` (line 51) `def encode(self, x)`
- `reparameterize` (line 55) `def reparameterize(self, mu, log_var)`
- `decode` (line 60) `def decode(self, z)`
- `forward` (line 64) `def forward(self, x)`

#### `09_transformer_mini.py`
**Path:** `09_transformer_mini.py`

**Classes:**
- `MultiHeadAttention` (line 51) `class MultiHeadAttention`
- `FeedForward` (line 83) `class FeedForward`
- `MiniTransformerEncoder` (line 96) `class MiniTransformerEncoder`
- `MiniTransformer` (line 131) `class MiniTransformer`

**Functions:**
- `generate_copy_data` (line 29) `def generate_copy_data(num_samples, seq_len, vocab_size)`
- `__init__` (line 52) `def __init__(self, d_model, num_heads, dropout)`
- `forward` (line 63) `def forward(self, q, k, v, mask)`
- `__init__` (line 84) `def __init__(self, d_model, d_ff, dropout)`
- `forward` (line 90) `def forward(self, x)`
- `__init__` (line 97) `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len)`
- `_create_positional_encoding` (line 107) `def _create_positional_encoding(self, max_len, d_model)`
- `forward` (line 115) `def forward(self, x)`
- `__init__` (line 132) `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout)`
- `forward` (line 137) `def forward(self, x)`

#### `10_gan_mnist_lite.py`
**Path:** `10_gan_mnist_lite.py`

**Classes:**
- `Generator` (line 37) `class Generator`
- `Discriminator` (line 57) `class Discriminator`

**Functions:**
- `__init__` (line 38) `def __init__(self, latent_dim, img_size)`
- `forward` (line 51) `def forward(self, z)`
- `__init__` (line 58) `def __init__(self, img_size)`
- `forward` (line 74) `def forward(self, x)`

#### `11_bert_tiny.py`
**Path:** `11_bert_tiny.py`

**Classes:**
- `MultiHeadAttention` (line 107) `class MultiHeadAttention`
- `FeedForward` (line 133) `class FeedForward`
- `BERTLayer` (line 143) `class BERTLayer`
- `TinyBERT` (line 158) `class TinyBERT`

**Functions:**
- `tokenize_sentence` (line 76) `def tokenize_sentence(sentence)`
- `pad_sequence` (line 88) `def pad_sequence(seq, length, pad_value)`
- `mask_tokens` (line 186) `def mask_tokens(inputs, vocab_size, mask_token_id, pad_token_id, mask_prob)`
- `__init__` (line 108) `def __init__(self, d_model, num_heads, dropout)`
- `forward` (line 119) `def forward(self, q, k, v, mask)`
- `__init__` (line 134) `def __init__(self, d_model, d_ff, dropout)`
- `forward` (line 140) `def forward(self, x)`
- `__init__` (line 144) `def __init__(self, d_model, num_heads, d_ff, dropout)`
- `forward` (line 151) `def forward(self, x, mask)`
- `__init__` (line 159) `def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout, max_len)`
- `_create_positional_encoding` (line 167) `def _create_positional_encoding(self, max_len, d_model)`
- `forward` (line 175) `def forward(self, x, mask)`

#### `12_diffusion_minimal.py`
**Path:** `12_diffusion_minimal.py`

**Classes:**
- `SimpleDiffusionNet` (line 54) `class SimpleDiffusionNet`

**Functions:**
- `q_sample` (line 81) `def q_sample(x_0, t, noise)` - *Muestrea x_t dado x_0 y timestep t.
x_0: (B, C, H, W)
t: (B,)*
- `__init__` (line 55) `def __init__(self, in_channels, out_channels, hidden_dim)`
- `forward` (line 65) `def forward(self, x, t)`

#### `13_nested_hope.py`
**Path:** `13_nested_hope.py`

**Classes:**
- `Config` (line 14) `class Config` - *Configuración centralizada basada en el paper (Secciones 7-9)*
- `DeltaGradientDescent` (line 66) `class DeltaGradientDescent` - *Implementación de DGD según Eq. 121:
W_{t+1} = W_t(I - α_t x_t x_t^T) - β ∇_{y_t} L(W_t; x_t) x_t^T

A diferencia del GD estándar, DGD incluye:
1. Decaimiento adaptativo basado en el estado actual (término αI)
2. Dependencia de muestras anteriores (no asume i.i.d.)*
- `SelfModifyingMemory` (line 115) `class SelfModifyingMemory` - *Implementación completa de Self-Modifying Deep Associative Memory

Paper: Sección 8.1, Ecuaciones 83-88
- Genera sus propias proyecciones (k, v, q, η, α)
- Genera valores propios auto-referenciales (Eq. 84)
- Aplica regla Delta para actualización (Eq. 88)*
- `ContinuumMemorySystem` (line 258) `class ContinuumMemorySystem` - *Sistema de memoria continuo con múltiples frecuencias de actualización

Paper: Sección 7.1
- Niveles con diferentes frecuencias (rápido → lento)
- Actualización condicional basada en chunk_size
- Conexión secuencial (Eq. 73) o independiente (Eq. 74)*
- `HopeModel` (line 337) `class HopeModel` - *Arquitectura Hope: Self-Modifying Memory + CMS

Paper: Sección 8.3, Figura 5*
- `HopeTrainer` (line 432) `class HopeTrainer` - *Sistema de entrenamiento con ablación automática y métricas científicas*

**Functions:**
- `setup_device` (line 44) `def setup_device()` - *Configuración automática de dispositivo*
- `set_seed` (line 54) `def set_seed(seed)` - *Reproducibilidad completa*
- `run_ablation_study` (line 564) `def run_ablation_study(config, device)` - *Ejecuta estudio de ablación completo

Configuraciones testeadas:
1. Hope completo (Self-Mod + CMS + DGD)
2. Sin Self-Modifying
3. Sin CMS
4. Sin DGD
5. Baseline (sin ninguno)*
- `apply_update` (line 77) `def apply_update(grad, param, x_normalized, eta, alpha, lambda_norm)` - *Aplica la regla DGD a un gradiente

Args:
    grad: Gradiente actual ∇L
    param: Parámetro W_t
    x_normalized: Input normalizado (||x|| = λ)
    eta: Learning rate adaptativo (por muestra)
    alpha: Retention gate (por muestra)
    lambda_norm: Norma de x (default: 1.0 para L2-norm)

Returns:
    Gradiente modificado según DGD*
- `__init__` (line 125) `def __init__(self, d_model, hidden_dim, chunk_size)`
- `_make_memory_module` (line 162) `def _make_memory_module(self)` - *Crea un módulo de memoria (MLP de 2 capas con residual)*
- `forward` (line 170) `def forward(self, x, prev_states)` - *Forward pass con actualización chunk-wise (Sección 8.2)

Args:
    x: (B, S, D) - Input tokens embebidos
    prev_states: Estados previos de memorias
    
Returns:
    output: (B, S, D)
    new_states: Estados actualizados*
- `__init__` (line 268) `def __init__(self, frequencies, d_model, hidden_dim, connection_type)`
- `forward` (line 296) `def forward(self, x, global_step)` - *Forward pass con actualizaciones multi-frecuencia

Args:
    x: (B, S, D)
    global_step: Paso global de entrenamiento
    
Returns:
    output: (B, S, D)*
- `__init__` (line 344) `def __init__(self, vocab_size, d_model, cms_frequencies, mlp_hidden, chunk_size, enable_self_modifying, enable_cms)`
- `reset_states` (line 390) `def reset_states(self)` - *Reset de estados internos (para nuevas secuencias)*
- `forward` (line 394) `def forward(self, x, global_step, return_internals)` - *Forward pass completo

Args:
    x: (B, S) - Input token IDs
    global_step: Paso global
    return_internals: Si retornar estados internos (para análisis)*
- `__init__` (line 437) `def __init__(self, model, config, device)`
- `train_epoch` (line 461) `def train_epoch(self, train_loader, epoch, global_step)` - *Entrena una época completa

Returns:
    avg_loss, avg_acc, new_global_step*
- `evaluate` (line 536) `def evaluate(self, test_loader, global_step)` - *Evaluación sin gradientes*

#### `13_nested_kearning_gpu.py`
**Path:** `13_nested_kearning_gpu.py`

**Classes:**
- `SelfModifyingMemory` (line 53) `class SelfModifyingMemory`
- `ContinuumMemorySystem` (line 74) `class ContinuumMemorySystem`
- `HopeModel` (line 97) `class HopeModel`

**Functions:**
- `__init__` (line 54) `def __init__(self, vocab_size, d_model, hidden_dim)`
- `forward` (line 64) `def forward(self, x)`
- `__init__` (line 75) `def __init__(self, frequencies, d_model, hidden_dim)`
- `forward` (line 87) `def forward(self, x, global_step)`
- `__init__` (line 98) `def __init__(self, vocab_size, d_model, cms_freqs, hidden_dim)`
- `forward` (line 104) `def forward(self, x, global_step)`

#### `13_nested_learning.py`
**Path:** `13_nested_learning.py`

**Classes:**
- `SelfModifyingMemory` (line 51) `class SelfModifyingMemory`
- `ContinuumMemorySystem` (line 86) `class ContinuumMemorySystem`
- `HopeModel` (line 114) `class HopeModel`

**Functions:**
- `__init__` (line 52) `def __init__(self, vocab_size, d_model, hidden_dim)`
- `forward` (line 62) `def forward(self, x, update_mask)` - *x: (B, S)
update_mask: None o máscara booleana para actualizaciones condicionales
Retorna logits y parámetros internos (para depuración/futuras extensiones)*
- `__init__` (line 87) `def __init__(self, levels, d_model, hidden_dim)`
- `forward` (line 100) `def forward(self, x, global_step)` - *x: (B, S, D)
global_step: int, paso global de entrenamiento*
- `__init__` (line 115) `def __init__(self, vocab_size, d_model, cms_levels, mlp_hidden)`
- `forward` (line 122) `def forward(self, x, global_step)`

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v3.5.py`

**Classes:**
- `LanguageMetrics` (line 44) `class LanguageMetrics` - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 117) `class TriangulatedMedicalSystem` - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 382) `class StableLiquidNeuron`
- `RightHemisphere` (line 458) `class RightHemisphere`
- `LeftHemisphere` (line 473) `class LeftHemisphere`
- `CorpusCallosum` (line 544) `class CorpusCallosum`
- `NeuroLogosBicameralStable` (line 581) `class NeuroLogosBicameralStable`
- `EnhancedDiagnostics` (line 602) `class EnhancedDiagnostics`
- `Flickr8kDataset` (line 727) `class Flickr8kDataset(Dataset)`

**Functions:**
- `compute_loss` (line 20) `def compute_loss(logits, captions, gate, vocab)`
- `build_vocab_flickr` (line 764) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 782) `def setup_flickr8k(data_dir)`
- `train_with_metrics` (line 854) `def train_with_metrics()`
- `sentence_bleu` (line 48) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 82) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 91) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 104) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 120) `def __init__(self)`
- `triangulate_signals` (line 125) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 157) `def count_convergent_signals(self, signals, pattern)` - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 161) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 223) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 383) `def __init__(self, in_dim, out_dim)`
- `forward` (line 397) `def forward(self, x)`
- `hebbian_update` (line 404) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 434) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 459) `def __init__(self, output_dim)`
- `forward` (line 467) `def forward(self, image)`
- `__init__` (line 474) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 502) `def forward(self, visual_context, captions, max_len)`
- `_get_init_state` (line 539) `def _get_init_state(self, visual_context)`
- `__init__` (line 545) `def __init__(self, dim)`
- `forward` (line 568) `def forward(self, right_features)`
- `__init__` (line 582) `def __init__(self, vocab_size)`
- `forward` (line 588) `def forward(self, image, captions)`
- `__init__` (line 603) `def __init__(self)`
- `measure_callosal_flow` (line 613) `def measure_callosal_flow(self, right_features, left_context)`
- `calculate_synergy` (line 622) `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 631) `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 640) `def update(self)`
- `get_recent_avg` (line 645) `def get_recent_avg(self, key, n)`
- `report` (line 650) `def report(self, epoch)`
- `__init__` (line 728) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 745) `def __len__(self)`
- `__getitem__` (line 748) `def __getitem__(self, idx)`

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v3.6.py`

**Classes:**
- `LanguageMetrics` (line 44) `class LanguageMetrics` - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 117) `class TriangulatedMedicalSystem` - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 383) `class StableLiquidNeuron`
- `RightHemisphere` (line 509) `class RightHemisphere`
- `LeftHemisphere` (line 524) `class LeftHemisphere`
- `CorpusCallosum` (line 645) `class CorpusCallosum`
- `NeuroLogosBicameralStable` (line 700) `class NeuroLogosBicameralStable`
- `EnhancedDiagnostics` (line 721) `class EnhancedDiagnostics`
- `EpisodicMemoryBuffer` (line 844) `class EpisodicMemoryBuffer`
- `Flickr8kDataset` (line 891) `class Flickr8kDataset(Dataset)`

**Functions:**
- `compute_loss` (line 20) `def compute_loss(logits, captions, gate, vocab)`
- `build_vocab_flickr` (line 928) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 946) `def setup_flickr8k(data_dir)`
- `train_with_metrics` (line 1021) `def train_with_metrics()`
- `sentence_bleu` (line 48) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 82) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 91) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 104) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 120) `def __init__(self)`
- `triangulate_signals` (line 125) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 157) `def count_convergent_signals(self, signals, pattern)` - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 161) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 224) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 384) `def __init__(self, in_dim, out_dim)`
- `forward` (line 422) `def forward(self, x)`
- `hebbian_update` (line 444) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 483) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 510) `def __init__(self, output_dim)`
- `forward` (line 518) `def forward(self, image)`
- `__init__` (line 525) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 573) `def forward(self, visual_context, captions, max_len)`
- `_get_init_state` (line 631) `def _get_init_state(self, visual_context)`
- `__init__` (line 646) `def __init__(self, dim)`
- `forward` (line 678) `def forward(self, right_features)`
- `__init__` (line 701) `def __init__(self, vocab_size)`
- `forward` (line 707) `def forward(self, image, captions)`
- `__init__` (line 722) `def __init__(self)`
- `measure_callosal_flow` (line 732) `def measure_callosal_flow(self, right_features, left_context)`
- `calculate_synergy` (line 741) `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 750) `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 759) `def update(self)`
- `get_recent_avg` (line 764) `def get_recent_avg(self, key, n)`
- `report` (line 769) `def report(self, epoch)`
- `__init__` (line 845) `def __init__(self, capacity, surprise_threshold)`
- `compute_surprise` (line 851) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `add` (line 862) `def add(self, image, caption, surprise_score)`
- `sample` (line 872) `def sample(self, batch_size)`
- `__init__` (line 892) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 909) `def __len__(self)`
- `__getitem__` (line 912) `def __getitem__(self, idx)`

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v3.7.py`

**Classes:**
- `LanguageMetrics` (line 44) `class LanguageMetrics` - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 117) `class TriangulatedMedicalSystem` - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 392) `class StableLiquidNeuron`
- `RightHemisphere` (line 518) `class RightHemisphere`
- `LeftHemisphere` (line 533) `class LeftHemisphere`
- `CorpusCallosum` (line 709) `class CorpusCallosum`
- `NeuroLogosBicameralStable` (line 765) `class NeuroLogosBicameralStable`
- `EnhancedDiagnostics` (line 786) `class EnhancedDiagnostics`
- `EpisodicMemoryBuffer` (line 907) `class EpisodicMemoryBuffer`
- `Flickr8kDataset` (line 954) `class Flickr8kDataset(Dataset)`

**Functions:**
- `compute_loss` (line 20) `def compute_loss(logits, captions, gate, vocab)`
- `build_vocab_flickr` (line 991) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 1009) `def setup_flickr8k(data_dir)`
- `train_with_metrics` (line 1081) `def train_with_metrics()`
- `sentence_bleu` (line 48) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 82) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 91) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 104) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 120) `def __init__(self)`
- `triangulate_signals` (line 125) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 157) `def count_convergent_signals(self, signals, pattern)` - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 161) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 224) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 393) `def __init__(self, in_dim, out_dim)`
- `forward` (line 431) `def forward(self, x)`
- `hebbian_update` (line 453) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 492) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 519) `def __init__(self, output_dim)`
- `forward` (line 527) `def forward(self, image)`
- `__init__` (line 534) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `beam_search_decode` (line 584) `def beam_search_decode(self, visual_context, beam_width, max_len, epoch)`
- `forward` (line 657) `def forward(self, visual_context, captions, max_len, epoch)`
- `_get_init_state` (line 695) `def _get_init_state(self, visual_context)`
- `__init__` (line 710) `def __init__(self, dim)`
- `forward` (line 742) `def forward(self, right_features)`
- `__init__` (line 766) `def __init__(self, vocab_size)`
- `forward` (line 772) `def forward(self, image, captions, epoch)`
- `__init__` (line 787) `def __init__(self)`
- `measure_callosal_flow` (line 797) `def measure_callosal_flow(self, right_features, left_context)`
- `calculate_synergy` (line 806) `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 815) `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 824) `def update(self)`
- `get_recent_avg` (line 829) `def get_recent_avg(self, key, n)`
- `report` (line 834) `def report(self, epoch)`
- `__init__` (line 908) `def __init__(self, capacity, surprise_threshold)`
- `compute_surprise` (line 914) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `add` (line 925) `def add(self, image, caption, surprise_score)`
- `sample` (line 935) `def sample(self, batch_size)`
- `__init__` (line 955) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 972) `def __len__(self)`
- `__getitem__` (line 975) `def __getitem__(self, idx)`

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v3.8.py`

**Classes:**
- `NeurocognitiveSystem` (line 49) `class NeurocognitiveSystem` - *Sistema neurocognitivo que complementa al sistema médico
para optimizar el aprendizaje lingüístico*
- `LinguisticFeedbackLoop` (line 220) `class LinguisticFeedbackLoop` - *Sistema que integra métricas lingüísticas en el proceso de aprendizaje*
- `LanguageMetrics` (line 309) `class LanguageMetrics` - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 382) `class TriangulatedMedicalSystem` - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 657) `class StableLiquidNeuron`
- `RightHemisphere` (line 783) `class RightHemisphere`
- `LeftHemisphere` (line 798) `class LeftHemisphere`
- `CorpusCallosum` (line 978) `class CorpusCallosum`
- `NeuroLogosBicameralStable` (line 1034) `class NeuroLogosBicameralStable`
- `EnhancedDiagnostics` (line 1055) `class EnhancedDiagnostics`
- `EpisodicMemoryBuffer` (line 1192) `class EpisodicMemoryBuffer`
- `Flickr8kDataset` (line 1239) `class Flickr8kDataset(Dataset)`

**Functions:**
- `compute_loss` (line 20) `def compute_loss(logits, captions, gate, vocab, linguistic_reward, lambda_reward)` - *Función de pérdida extendida que incorpora recompensa lingüística*
- `build_vocab_flickr` (line 1276) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 1294) `def setup_flickr8k(data_dir)`
- `train_with_metrics` (line 1366) `def train_with_metrics()`
- `__init__` (line 55) `def __init__(self)`
- `assess_cognitive_state` (line 65) `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` - *Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas*
- `apply_cognitive_intervention` (line 113) `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch)` - *Aplica intervenciones cognitivas basadas en el estado lingüístico*
- `__init__` (line 223) `def __init__(self, alpha, beta)`
- `compute_linguistic_reward` (line 232) `def compute_linguistic_reward(self, references, hypotheses)` - *Calcula una recompensa combinada basada en CIDEr y SPICE
que puede usarse para guiar el entrenamiento*
- `compute_cider` (line 254) `def compute_cider(self, reference, hypothesis)` - *Versión simplificada de CIDEr para uso en entrenamiento*
- `compute_spice` (line 281) `def compute_spice(self, reference, hypothesis)` - *Versión simplificada de SPICE para uso en entrenamiento*
- `_get_ngrams` (line 295) `def _get_ngrams(self, sentence, n)` - *Extrae n-gramas de una oración*
- `sentence_bleu` (line 313) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 347) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 356) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 369) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 385) `def __init__(self)`
- `triangulate_signals` (line 390) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 422) `def count_convergent_signals(self, signals, pattern)` - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 426) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 489) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 658) `def __init__(self, in_dim, out_dim)`
- `forward` (line 696) `def forward(self, x)`
- `hebbian_update` (line 718) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 757) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 784) `def __init__(self, output_dim)`
- `forward` (line 792) `def forward(self, image)`
- `__init__` (line 799) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `beam_search_decode` (line 849) `def beam_search_decode(self, visual_context, beam_width, max_len, epoch)`
- `forward` (line 925) `def forward(self, visual_context, captions, max_len, epoch)`
- `_get_init_state` (line 963) `def _get_init_state(self, visual_context)`
- `__init__` (line 979) `def __init__(self, dim)`
- `forward` (line 1011) `def forward(self, right_features)`
- `__init__` (line 1035) `def __init__(self, vocab_size)`
- `forward` (line 1041) `def forward(self, image, captions, epoch)`
- `__init__` (line 1056) `def __init__(self)`
- `measure_callosal_flow` (line 1067) `def measure_callosal_flow(self, right_features, left_context)`
- `calculate_synergy` (line 1076) `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 1085) `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 1094) `def update(self)`
- `get_recent_avg` (line 1099) `def get_recent_avg(self, key, n)`
- `report` (line 1104) `def report(self, epoch)`
- `__init__` (line 1193) `def __init__(self, capacity, surprise_threshold)`
- `compute_surprise` (line 1199) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `add` (line 1210) `def add(self, image, caption, surprise_score)`
- `sample` (line 1220) `def sample(self, batch_size)`
- `__init__` (line 1240) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 1257) `def __len__(self)`
- `__getitem__` (line 1260) `def __getitem__(self, idx)`

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v3.9.py`

**Classes:**
- `NeurocognitiveSystem` (line 49) `class NeurocognitiveSystem` - *Sistema neurocognitivo que complementa al sistema médico
para optimizar el aprendizaje lingüístico*
- `LinguisticFeedbackLoop` (line 325) `class LinguisticFeedbackLoop` - *Sistema que integra métricas lingüísticas en el proceso de aprendizaje.
Versión optimizada con caché de dos niveles para minimizar cálculos repetitivos.*
- `LanguageMetrics` (line 479) `class LanguageMetrics` - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 552) `class TriangulatedMedicalSystem` - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 827) `class StableLiquidNeuron`
- `RightHemisphere` (line 973) `class RightHemisphere`
- `LeftHemisphere` (line 988) `class LeftHemisphere`
- `CorpusCallosum` (line 1244) `class CorpusCallosum`
- `NeuroLogosBicameralStable` (line 1420) `class NeuroLogosBicameralStable`
- `EnhancedDiagnostics` (line 1446) `class EnhancedDiagnostics`
- `EpisodicMemoryBuffer` (line 1661) `class EpisodicMemoryBuffer`
- `Flickr8kDataset` (line 1708) `class Flickr8kDataset(Dataset)`

**Functions:**
- `compute_loss` (line 20) `def compute_loss(logits, captions, gate, vocab, linguistic_reward, lambda_reward)` - *Función de pérdida extendida que incorpora recompensa lingüística*
- `build_vocab_flickr` (line 1745) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 1763) `def setup_flickr8k(data_dir)`
- `compute_alignment_loss` (line 1834) `def compute_alignment_loss(visual_features, channels, alpha)` - *Pérdida auxiliar para forzar alineación entre características visuales
y canales estructurales del callosum durante épocas tempranas*
- `train_with_metrics` (line 1858) `def train_with_metrics()`
- `__init__` (line 55) `def __init__(self)`
- `assess_cognitive_state` (line 76) `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` - *Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas*
- `evaluate_gate_state` (line 124) `def evaluate_gate_state(self, gate_value, current_metrics)` - *MEJORA: Evaluar estado del gate con sistema inmune*
- `update_trauma_memory` (line 142) `def update_trauma_memory(self, gate_value, metrics, outcome)` - *MEJORA: Actualizar memoria traumática basada en resultados*
- `apply_stochastic_perturbation` (line 155) `def apply_stochastic_perturbation(self, model, epoch)` - *MEJORA: Aplicar micro-perturbaciones estocásticas*
- `apply_cognitive_intervention` (line 171) `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` - *Aplica intervenciones cognitivas basadas en el estado lingüístico*
- `__init__` (line 331) `def __init__(self, alpha, beta)`
- `compute_linguistic_reward` (line 347) `def compute_linguistic_reward(self, references, hypotheses)` - *Calcula una recompensa combinada basada en CIDEr y SPICE.
Utiliza caché para acelerar el cálculo de métricas.*
- `compute_cider` (line 389) `def compute_cider(self, reference, hypothesis)` - *Versión simplificada de CIDEr para uso en entrenamiento.
Optimizada con caché de n-gramas.*
- `compute_spice` (line 427) `def compute_spice(self, reference, hypothesis)` - *Versión simplificada de SPICE para uso en entrenamiento.
Usa Jaccard similarity como proxy semántico.*
- `_get_ngrams` (line 443) `def _get_ngrams(self, sentence, n)` - *Extrae n-gramas de una oración*
- `get_cache_stats` (line 452) `def get_cache_stats(self)` - *Obtiene estadísticas del sistema de caché*
- `sentence_bleu` (line 483) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 517) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 526) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 539) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 555) `def __init__(self)`
- `triangulate_signals` (line 560) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 592) `def count_convergent_signals(self, signals, pattern)` - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 596) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 659) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 828) `def __init__(self, in_dim, out_dim)`
- `forward` (line 871) `def forward(self, x)`
- `hebbian_update` (line 893) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 932) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 974) `def __init__(self, output_dim)`
- `forward` (line 982) `def forward(self, image)`
- `__init__` (line 989) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `beam_search_decode` (line 1066) `def beam_search_decode(self, visual_context, channels, beam_width, max_len, epoch)`
- `forward` (line 1147) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_apply_structural_attention` (line 1185) `def _apply_structural_attention(self, lstm_out, channels, visual_context)` - *Aplica atención específica para cada canal estructural (objetos, acciones, escena).
Versión optimizada con matemática robusta y eficiente.*
- `_get_init_state` (line 1230) `def _get_init_state(self, visual_context)`
- `__init__` (line 1245) `def __init__(self, dim)`
- `forward` (line 1308) `def forward(self, right_features, left_features)`
- `update_channel_fatigue` (line 1381) `def update_channel_fatigue(self, objects_channel, actions_channel, scene_channel)` - *MEJORA: Actualizar fatiga específica por canal*
- `adjust_gates_by_fatigue` (line 1404) `def adjust_gates_by_fatigue(self)` - *MEJORA: Ajustar gates basado en fatiga de cada canal*
- `__init__` (line 1421) `def __init__(self, vocab_size)`
- `forward` (line 1427) `def forward(self, image, captions, epoch)`
- `__init__` (line 1447) `def __init__(self)`
- `measure_callosal_flow` (line 1460) `def measure_callosal_flow(self, right_features, left_context, channels)`
- `calculate_synergy` (line 1490) `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 1499) `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 1508) `def update(self)`
- `get_recent_avg` (line 1519) `def get_recent_avg(self, key, n)`
- `visualize_fatigue_distribution` (line 1538) `def visualize_fatigue_distribution(self, epoch)` - *MEJORA: Visualizar distribución de fatiga entre canales*
- `report` (line 1567) `def report(self, epoch)`
- `__init__` (line 1662) `def __init__(self, capacity, surprise_threshold)`
- `compute_surprise` (line 1668) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `add` (line 1679) `def add(self, image, caption, surprise_score)`
- `sample` (line 1689) `def sample(self, batch_size)`
- `__init__` (line 1709) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 1726) `def __len__(self)`
- `__getitem__` (line 1729) `def __getitem__(self, idx)`

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v4.0.py`

**Classes:**
- `NeurocognitiveSystem` (line 53) `class NeurocognitiveSystem` - *Sistema neurocognitivo que complementa al sistema médico
para optimizar el aprendizaje lingüístico y el razonamiento*
- `LinguisticFeedbackLoop` (line 396) `class LinguisticFeedbackLoop` - *Sistema que integra métricas lingüísticas en el proceso de aprendizaje.
Versión optimizada con caché de dos niveles para minimizar cálculos repetitivos.*
- `LanguageMetrics` (line 550) `class LanguageMetrics` - *Métricas de calidad de generación*
- `TriangulatedMedicalSystem` (line 623) `class TriangulatedMedicalSystem` - *Sistema médico con triangulación de señales convergentes*
- `StableLiquidNeuron` (line 898) `class StableLiquidNeuron`
- `RightHemisphere` (line 1044) `class RightHemisphere`
- `LeftHemisphere` (line 1059) `class LeftHemisphere`
- `CorpusCallosum` (line 1469) `class CorpusCallosum`
- `NeuroLogosBicameralStable` (line 1645) `class NeuroLogosBicameralStable`
- `EnhancedDiagnostics` (line 1669) `class EnhancedDiagnostics`
- `EpisodicMemoryBuffer` (line 1947) `class EpisodicMemoryBuffer`
- `Flickr8kDataset` (line 1994) `class Flickr8kDataset(Dataset)`

**Functions:**
- `compute_loss` (line 20) `def compute_loss(logits, captions, gate, vocab, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)` - *Función de pérdida extendida con MTP y recompensa lingüística*
- `build_vocab_flickr` (line 2031) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 2049) `def setup_flickr8k(data_dir)`
- `compute_alignment_loss` (line 2120) `def compute_alignment_loss(visual_features, channels, alpha)` - *Pérdida auxiliar para forzar alineación entre características visuales
y canales estructurales del callosum durante épocas tempranas*
- `train_with_metrics` (line 2144) `def train_with_metrics()`
- `__init__` (line 59) `def __init__(self)`
- `assess_reasoning_state` (line 85) `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` - *Evalúa el estado del sistema de razonamiento*
- `assess_cognitive_state` (line 128) `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` - *Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas*
- `evaluate_gate_state` (line 167) `def evaluate_gate_state(self, gate_value, current_metrics)` - *Evaluar estado del gate con sistema inmune*
- `update_trauma_memory` (line 182) `def update_trauma_memory(self, gate_value, metrics, outcome)` - *Actualizar memoria traumática basada en resultados*
- `apply_stochastic_perturbation` (line 192) `def apply_stochastic_perturbation(self, model, epoch)` - *Aplicar micro-perturbaciones estocásticas*
- `apply_cognitive_intervention` (line 213) `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` - *Aplica intervenciones cognitivas basadas en el estado lingüístico y de razonamiento*
- `__init__` (line 402) `def __init__(self, alpha, beta)`
- `compute_linguistic_reward` (line 418) `def compute_linguistic_reward(self, references, hypotheses)` - *Calcula una recompensa combinada basada en CIDEr y SPICE.
Utiliza caché para acelerar el cálculo de métricas.*
- `compute_cider` (line 460) `def compute_cider(self, reference, hypothesis)` - *Versión simplificada de CIDEr para uso en entrenamiento.
Optimizada con caché de n-gramas.*
- `compute_spice` (line 498) `def compute_spice(self, reference, hypothesis)` - *Versión simplificada de SPICE para uso en entrenamiento.
Usa Jaccard similarity como proxy semántico.*
- `_get_ngrams` (line 514) `def _get_ngrams(self, sentence, n)` - *Extrae n-gramas de una oración*
- `get_cache_stats` (line 523) `def get_cache_stats(self)` - *Obtiene estadísticas del sistema de caché*
- `sentence_bleu` (line 554) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 588) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 597) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 610) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 626) `def __init__(self)`
- `triangulate_signals` (line 631) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Identificar señales convergentes que confirman problemas*
- `count_convergent_signals` (line 663) `def count_convergent_signals(self, signals, pattern)` - *Contar cuántas señales del patrón están activas*
- `diagnose_with_triangulation` (line 667) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` - *Diagnosticar SOLO con confirmación múltiple*
- `apply_triangulated_intervention` (line 730) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` - *Aplicar intervención SOLO si confianza es alta*
- `__init__` (line 899) `def __init__(self, in_dim, out_dim)`
- `forward` (line 942) `def forward(self, x)`
- `hebbian_update` (line 964) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 1003) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 1045) `def __init__(self, output_dim)`
- `forward` (line 1053) `def forward(self, image)`
- `__init__` (line 1060) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `_apply_chain_of_thought` (line 1190) `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)` - *Aplica cadena de pensamiento para mejorar el razonamiento*
- `_apply_multi_token_prediction` (line 1230) `def _apply_multi_token_prediction(self, hidden_states, input_ids)` - *Multi-Token Prediction: predice múltiples tokens futuros simultáneamente
CRÍTICO: Mantiene dimensiones consistentes con input original*
- `beam_search_decode` (line 1292) `def beam_search_decode(self, visual_context, channels, beam_width, max_len, epoch)`
- `forward` (line 1370) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_apply_structural_attention` (line 1420) `def _apply_structural_attention(self, lstm_out, channels, visual_context)` - *Aplica atención específica para cada canal estructural (objetos, acciones, escena).
Versión optimizada con matemática robusta y eficiente.*
- `_get_init_state` (line 1456) `def _get_init_state(self, visual_context)`
- `__init__` (line 1470) `def __init__(self, dim)`
- `forward` (line 1533) `def forward(self, right_features, left_features)`
- `update_channel_fatigue` (line 1606) `def update_channel_fatigue(self, objects_channel, actions_channel, scene_channel)` - *MEJORA: Actualizar fatiga específica por canal*
- `adjust_gates_by_fatigue` (line 1629) `def adjust_gates_by_fatigue(self)` - *MEJORA: Ajustar gates basado en fatiga de cada canal*
- `__init__` (line 1646) `def __init__(self, vocab_size)`
- `forward` (line 1652) `def forward(self, image, captions, epoch)`
- `__init__` (line 1670) `def __init__(self)`
- `measure_callosal_flow` (line 1686) `def measure_callosal_flow(self, right_features, left_context, channels)`
- `evaluate_reasoning_quality` (line 1709) `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` - *Evalúa la calidad del razonamiento en textos generados*
- `calculate_synergy` (line 1750) `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 1759) `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 1768) `def update(self)`
- `get_recent_avg` (line 1778) `def get_recent_avg(self, key, n)`
- `visualize_fatigue_distribution` (line 1795) `def visualize_fatigue_distribution(self, epoch)` - *Visualizar distribución de fatiga entre canales*
- `visualize_reasoning_metrics` (line 1823) `def visualize_reasoning_metrics(self, epoch)` - *Visualizar métricas de razonamiento*
- `report` (line 1836) `def report(self, epoch)`
- `__init__` (line 1948) `def __init__(self, capacity, surprise_threshold)`
- `compute_surprise` (line 1954) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `add` (line 1965) `def add(self, image, caption, surprise_score)`
- `sample` (line 1975) `def sample(self, batch_size)`
- `__init__` (line 1995) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 2012) `def __len__(self)`
- `__getitem__` (line 2015) `def __getitem__(self, idx)`

#### `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py`
**Path:** `NeuroLogos_Bicameral_FISIOLÓGICO_v4.1.py`

**Classes:**
- `EpisodicMemoryBuffer` (line 58) `class EpisodicMemoryBuffer` - *Buffer que almacena ejemplos sorpresivos para replay estratégico*
- `NeurocognitiveSystem` (line 112) `class NeurocognitiveSystem`
- `LinguisticFeedbackLoop` (line 291) `class LinguisticFeedbackLoop` - *Sistema de caché optimizado para métricas lingüísticas*
- `LanguageMetrics` (line 413) `class LanguageMetrics` - *Métricas clásicas de evaluación*
- `TriangulatedMedicalSystem` (line 475) `class TriangulatedMedicalSystem`
- `StableLiquidNeuron` (line 622) `class StableLiquidNeuron`
- `RightHemisphere` (line 745) `class RightHemisphere`
- `LeftHemisphere` (line 764) `class LeftHemisphere`
- `CorpusCallosum` (line 1019) `class CorpusCallosum`
- `NeuroLogosBicameralStable` (line 1147) `class NeuroLogosBicameralStable`
- `EnhancedDiagnostics` (line 1171) `class EnhancedDiagnostics`
- `Flickr8kDataset` (line 1408) `class Flickr8kDataset(Dataset)`

**Functions:**
- `compute_loss` (line 22) `def compute_loss(logits, captions, gate, vocab, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)` - *Función de pérdida extendida con MTP y recompensa lingüística*
- `build_vocab_flickr` (line 1445) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 1463) `def setup_flickr8k(data_dir)`
- `compute_alignment_loss` (line 1527) `def compute_alignment_loss(visual_features, channels, alpha)` - *Pérdida auxiliar para alineación temprana*
- `train_with_metrics` (line 1547) `def train_with_metrics()`
- `__init__` (line 61) `def __init__(self, capacity, surprise_threshold)`
- `compute_surprise` (line 67) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` - *Calcula sorpresa basada en error y apertura del gate*
- `add` (line 79) `def add(self, image, caption, surprise_score)` - *Añade ejemplo si supera umbral y hay capacidad*
- `sample` (line 91) `def sample(self, batch_size)` - *Samplea ejemplos con probabilidad proporcional a sorpresa*
- `__init__` (line 113) `def __init__(self)`
- `assess_reasoning_state` (line 128) `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` - *Evalúa estado del sistema de razonamiento*
- `assess_cognitive_state` (line 170) `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` - *Evalúa estado cognitivo lingüístico*
- `apply_cognitive_intervention` (line 206) `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` - *Aplica intervenciones basadas en estado lingüístico y razonamiento*
- `__init__` (line 294) `def __init__(self, alpha, beta)`
- `compute_linguistic_reward` (line 308) `def compute_linguistic_reward(self, references, hypotheses)` - *Recompensa combinada CIDEr + SPICE con caché*
- `compute_cider` (line 342) `def compute_cider(self, reference, hypothesis)` - *CIDEr simplificado con caché de n-gramas*
- `compute_spice` (line 371) `def compute_spice(self, reference, hypothesis)` - *SPICE simplificado (Jaccard similarity)*
- `_get_ngrams` (line 384) `def _get_ngrams(self, sentence, n)` - *Extractor de n-gramas*
- `get_cache_stats` (line 389) `def get_cache_stats(self)` - *Estadísticas de caché*
- `sentence_bleu` (line 417) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU-4 a nivel de oración*
- `token_accuracy` (line 448) `def token_accuracy(reference, hypothesis)` - *Precisión token-level*
- `word_overlap` (line 461) `def word_overlap(reference, hypothesis)` - *Jaccard similarity*
- `__init__` (line 476) `def __init__(self)`
- `triangulate_signals` (line 482) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `count_convergent_signals` (line 492) `def count_convergent_signals(self, signals, pattern)`
- `diagnose_with_triangulation` (line 495) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `apply_triangulated_intervention` (line 537) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- `_reset_liquid_neuron` (line 610) `def _reset_liquid_neuron(self, right_node, severity)`
- `__init__` (line 623) `def __init__(self, in_dim, out_dim)`
- `forward` (line 658) `def forward(self, x)`
- `hebbian_update` (line 671) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 709) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 746) `def __init__(self, output_dim)`
- `forward` (line 754) `def forward(self, image)`
- `__init__` (line 765) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 840) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_greedy_decode` (line 879) `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- `_apply_chain_of_thought` (line 912) `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- `_apply_multi_token_prediction` (line 944) `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- `_apply_structural_attention` (line 986) `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- `_get_init_state` (line 1007) `def _get_init_state(self, visual_context)`
- `__init__` (line 1020) `def __init__(self, dim)`
- `forward` (line 1070) `def forward(self, right_features, left_features)`
- `update_channel_fatigue` (line 1113) `def update_channel_fatigue(self, objects_channel, actions_channel, scene_channel)` - *Actualiza fatiga específica por canal*
- `adjust_gates_by_fatigue` (line 1132) `def adjust_gates_by_fatigue(self)` - *Ajusta gates basado en fatiga*
- `__init__` (line 1148) `def __init__(self, vocab_size)`
- `forward` (line 1154) `def forward(self, image, captions, epoch)`
- `__init__` (line 1172) `def __init__(self)`
- `measure_callosal_flow` (line 1187) `def measure_callosal_flow(self, right_features, left_context, channels)`
- `evaluate_reasoning_quality` (line 1210) `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` - *Evalúa coherencia y consistencia del razonamiento*
- `calculate_synergy` (line 1242) `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 1251) `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 1260) `def update(self)`
- `get_recent_avg` (line 1270) `def get_recent_avg(self, key, n)`
- `visualize_fatigue_distribution` (line 1287) `def visualize_fatigue_distribution(self, epoch)`
- `visualize_reasoning_metrics` (line 1308) `def visualize_reasoning_metrics(self, epoch)`
- `report` (line 1320) `def report(self, epoch)` - *Genera reporte completo del estado del sistema bicameral*
- `__init__` (line 1409) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 1426) `def __len__(self)`
- `__getitem__` (line 1429) `def __getitem__(self, idx)`

#### `ablation.py`
**Path:** `ablation.py`

**Classes:**
- `TopoBrainCore` (line 17) `class TopoBrainCore`
- `MiniUnconscious` (line 72) `class MiniUnconscious`
- `TopoUnconscious` (line 91) `class TopoUnconscious`
- `SimpleClassifier` (line 124) `class SimpleClassifier`
- `NeuroLogosCPU` (line 137) `class NeuroLogosCPU` - *Ablation levels:
  0: BASELINE     -> MiniUnconscious
  1: +GRID        -> TopoBrain (grid only)
  2: +SYMBIOTIC   -> TopoBrain (grid + symbiotic)
  3: +ADVERSARIAL -> TopoBrain + FGSM (lightweight)*

**Functions:**
- `fgsm_attack` (line 175) `def fgsm_attack(model, x, y, epsilon)`
- `train_epoch` (line 188) `def train_epoch(model, loader, optimizer, device, use_adv)`
- `evaluate` (line 229) `def evaluate(model, loader, device)`
- `run_ablation_cpu` (line 245) `def run_ablation_cpu()`
- `__init__` (line 18) `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)`
- `_init_grid` (line 39) `def _init_grid(self, size)`
- `forward` (line 46) `def forward(self, x)`
- `get_metrics` (line 64) `def get_metrics(self)`
- `__init__` (line 73) `def __init__(self, out_dim)`
- `forward` (line 87) `def forward(self, x)`
- `__init__` (line 92) `def __init__(self, out_dim, use_grid, use_symbiotic)`
- `forward` (line 112) `def forward(self, x)`
- `get_metrics` (line 116) `def get_metrics(self)`
- `__init__` (line 125) `def __init__(self, in_dim, num_classes)`
- `forward` (line 129) `def forward(self, x)`
- `__init__` (line 145) `def __init__(self, num_classes, ablation_level)`
- `forward` (line 161) `def forward(self, x)`
- `get_metrics` (line 165) `def get_metrics(self)`

#### `ablation1.py`
**Path:** `ablation1.py`

**Classes:**
- `EliteConfig` (line 20) `class EliteConfig`
- `AblationConfig` (line 60) `class AblationConfig(EliteConfig)`
- `EpisodicMemory` (line 180) `class EpisodicMemory` - *Memoria explícita de patrones adversariales*
- `SpectralNormLinear` (line 220) `class SpectralNormLinear` - *Linear con normalización espectral para estabilidad Lipschitz*
- `AdvancedHomeostaticCell` (line 248) `class AdvancedHomeostaticCell` - *Neurona con control fisiológico multinivel + memoria*
- `AdaptiveTopology` (line 296) `class AdaptiveTopology` - *Topología que aprende a reconectar bajo ataque*
- `EliteTopoBrain` (line 329) `class EliteTopoBrain`
- `SupConLoss` (line 467) `class SupConLoss`

**Functions:**
- `create_ablation_configs` (line 64) `def create_ablation_configs(config)` - *Crea un diccionario de configuraciones para cada test de ablación.*
- `seed_everything` (line 150) `def seed_everything(seed)`
- `get_elite_dataset` (line 157) `def get_elite_dataset(config)` - *Dataset más grande y balanceado con separabilidad controlada*
- `elite_pgd_attack` (line 407) `def elite_pgd_attack(model, x, y, eps, steps, stress)` - *PGD con reinicio aleatorio y gradiente centralizado (CORREGIDO)*
- `train_elite_model` (line 504) `def train_elite_model(config, dataset, fold_results)` - *Entrenamiento con curriculum adversarial*
- `run_ablation_study` (line 616) `def run_ablation_study()`
- `__init__` (line 182) `def __init__(self, dim, capacity)`
- `update` (line 190) `def update(self, x, y)` - *Almacena ejemplos duros*
- `retrieve` (line 203) `def retrieve(self, x, k)` - *Recupera k vecinos más cercanos*
- `__init__` (line 222) `def __init__(self, in_features, out_features)`
- `power_iteration` (line 229) `def power_iteration(self, n_iter)` - *Aproxima la norma espectral máxima*
- `forward` (line 236) `def forward(self, x)`
- `__init__` (line 250) `def __init__(self, d_in, d_out, use_spectral, use_homeostasis)`
- `forward` (line 271) `def forward(self, x)`
- `__init__` (line 298) `def __init__(self, num_nodes, grid_size)`
- `forward` (line 317) `def forward(self, stress)` - *stress ∈ [0,1]: cuánto estrés adversarial*
- `__init__` (line 330) `def __init__(self, config)`
- `count_parameters` (line 367) `def count_parameters(self)`
- `forward` (line 370) `def forward(self, x, stress)`
- `__init__` (line 468) `def __init__(self, temperature)`
- `forward` (line 472) `def forward(self, features, labels)`

#### `ablation2.py`
**Path:** `ablation2.py`

**Classes:**
- `SparseCompetitiveLayer` (line 19) `class SparseCompetitiveLayer` - *k-WTA con aprendizaje de importancia de nodos.
Cada nodo tiene un bias de vida útil que incrementa con activación.
Los menos usados son pruning dinámico.*
- `SymbioticRefiner` (line 91) `class SymbioticRefiner` - *Refinamiento ortogonal con normalización de estabilidad.
Versión mejorada con spectral clamping para evitar desvanecimiento.*
- `SparseSymbioticCore` (line 120) `class SparseSymbioticCore` - *Reemplaza TopoBrainCore.
Combina SparseCompetitiveLayer + SymbioticRefiner.*
- `BaselineUnconscious` (line 167) `class BaselineUnconscious` - *Mantener para ablation - Encoder sin sparsity*
- `SparseUnconscious` (line 186) `class SparseUnconscious` - *Encoder con SparseSymbioticCore*
- `ConsciousCore` (line 219) `class ConsciousCore`
- `BioDecoder` (line 231) `class BioDecoder` - *Decoder con gating líquido*
- `NeuroLogos_v51` (line 289) `class NeuroLogos_v51` - *5 configuraciones para ablation desacoplado:

1. BASELINE-v51: Encoder densa sin sparsity
2. SPARSE-ONLY: Sparse layer SIN symbiotic
3. SYMBIOTIC-ONLY: Symbiotic SIN sparse (capa densa)
4. SPARSE-SYMBIOTIC: Ambos sin adversarial
5. SPARSE-SYMBIOTIC-ADV: Full (mejor versión)*
- `PGDAttack` (line 346) `class PGDAttack`
- `CIFARCaptions_v51` (line 375) `class CIFARCaptions_v51`

**Functions:**
- `compute_bleu` (line 417) `def compute_bleu(pred_ids, target_ids, dataset, max_n)` - *BLEU score simplificado para evaluar calidad de generación*
- `train_ablation_v51` (line 466) `def train_ablation_v51(mode, epochs, device, n_nodes, k_sparse)` - *Entrena una configuración específica del ablation study v5.1.
Ahora con BLEU score y métricas de activación por clase.*
- `run_ablation_v51` (line 619) `def run_ablation_v51(epochs, device, n_nodes, k_sparse)` - *Ejecuta ablation study v5.1 con 5 brazos desacoplados.*
- `__init__` (line 25) `def __init__(self, n_nodes, k_sparse, input_dim)`
- `forward` (line 44) `def forward(self, x)`
- `get_metrics` (line 79) `def get_metrics(self)`
- `__init__` (line 96) `def __init__(self, n_nodes)`
- `forward` (line 106) `def forward(self, x)`
- `__init__` (line 125) `def __init__(self, input_dim, hidden_dim, n_nodes, k_sparse)`
- `forward` (line 143) `def forward(self, x)`
- `get_metrics` (line 158) `def get_metrics(self)`
- `__init__` (line 169) `def __init__(self, output_dim)`
- `forward` (line 182) `def forward(self, x)`
- `__init__` (line 188) `def __init__(self, output_dim, n_nodes, k_sparse)`
- `forward` (line 207) `def forward(self, x)`
- `get_metrics` (line 211) `def get_metrics(self)`
- `__init__` (line 220) `def __init__(self, dim)`
- `forward` (line 225) `def forward(self, x)`
- `__init__` (line 233) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 248) `def forward(self, thought, captions, max_len)`
- `_get_init_state` (line 279) `def _get_init_state(self, thought)`
- `__init__` (line 299) `def __init__(self, vocab_size, mode, n_nodes, k_sparse)`
- `forward` (line 331) `def forward(self, image, captions)`
- `get_metrics` (line 340) `def get_metrics(self)`
- `__init__` (line 347) `def __init__(self, epsilon, alpha, steps)`
- `attack` (line 352) `def attack(self, model, x, y, criterion)`
- `__init__` (line 376) `def __init__(self)`
- `__len__` (line 403) `def __len__(self)`
- `__getitem__` (line 406) `def __getitem__(self, idx)`
- `ngrams` (line 423) `def ngrams(tokens, n)`
- `forward_fn` (line 526) `def forward_fn(x)`

#### `ablation3.py`
**Path:** `ablation3.py`

**Classes:**
- `SparseLayer` (line 21) `class SparseLayer` - *k-WTA con health tracking - Factor S*
- `SymbioticLayer` (line 54) `class SymbioticLayer` - *Orthogonal refinement - Factor Y*
- `AdversarialWrapper` (line 77) `class AdversarialWrapper` - *Wrapper PGD - Factor A*
- `VisualBackbone` (line 109) `class VisualBackbone`
- `ConsciousCore` (line 125) `class ConsciousCore`
- `BioDecoder` (line 136) `class BioDecoder`
- `NeuroLogosFactorial` (line 190) `class NeuroLogosFactorial`
- `CIFARCaptions` (line 295) `class CIFARCaptions`

**Functions:**
- `train_configuration` (line 338) `def train_configuration(config, epochs, device, n_nodes, k_sparse)` - *Entrena UNA configuración específica del diseño factorial.

Args:
    config: tuple (use_sparse, use_symbiotic, use_adv)*
- `run_full_factorial` (line 411) `def run_full_factorial(epochs, device, n_nodes, k_sparse)` - *Ejecuta el ablation factorial completo: 8 combinaciones + 3 inversas.*
- `analyze_results` (line 480) `def analyze_results(results)` - *Análisis de efectos principales, interacciones y poder explicativo.*
- `__init__` (line 23) `def __init__(self, input_dim, n_nodes, k_sparse)`
- `forward` (line 33) `def forward(self, x)`
- `__init__` (line 56) `def __init__(self, n_nodes)`
- `forward` (line 62) `def forward(self, x)`
- `__init__` (line 79) `def __init__(self, epsilon, alpha, steps)`
- `attack` (line 84) `def attack(self, model_fn, x, y, criterion)`
- `__init__` (line 110) `def __init__(self, output_dim)`
- `forward` (line 122) `def forward(self, x)`
- `__init__` (line 126) `def __init__(self, dim)`
- `forward` (line 131) `def forward(self, x)`
- `__init__` (line 137) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 148) `def forward(self, thought, captions, max_len)`
- `_init_state` (line 182) `def _init_state(self, thought)`
- `__init__` (line 191) `def __init__(self, vocab_size, use_sparse, use_symbiotic, use_adv, n_nodes, k_sparse)`
- `forward` (line 226) `def forward(self, image, captions)`
- `train_step` (line 243) `def train_step(self, images, captions, optimizer, dataset)` - *Paso de entrenamiento con adversarial condicional y doble forward pass seguro*
- `__init__` (line 296) `def __init__(self)`
- `__len__` (line 321) `def __len__(self)`
- `__getitem__` (line 324) `def __getitem__(self, idx)`
- `model_fn` (line 257) `def model_fn(x)`

#### `adversarial_benchmark.py`
**Path:** `adversarial_benchmark.py`

**Classes:**
- `SupConLoss` (line 36) `class SupConLoss`
- `PredictiveErrorCell` (line 73) `class PredictiveErrorCell`
- `LearnableAbsenceGating` (line 85) `class LearnableAbsenceGating`
- `SymbioticBasisRefinement` (line 100) `class SymbioticBasisRefinement`
- `CombinatorialComplexLayer` (line 118) `class CombinatorialComplexLayer`
- `TopoBrainNet` (line 158) `class TopoBrainNet`

**Functions:**
- `make_adversarial_pgd` (line 269) `def make_adversarial_pgd(model, x, y, eps, steps)`
- `train_and_eval` (line 290) `def train_and_eval()`
- `__init__` (line 37) `def __init__(self, temperature)`
- `forward` (line 41) `def forward(self, features, labels)`
- `__init__` (line 74) `def __init__(self, dim)`
- `forward` (line 79) `def forward(self, input_signal, prediction)`
- `__init__` (line 86) `def __init__(self, dim)`
- `forward` (line 95) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 101) `def __init__(self, dim, num_atoms)`
- `forward` (line 109) `def forward(self, x)`
- `__init__` (line 119) `def __init__(self, in_dim, hid_dim, num_nodes, layer_type)`
- `forward` (line 136) `def forward(self, x_nodes, adjacency, incidence)`
- `__init__` (line 159) `def __init__(self, grid_size)`
- `_init_grid_topology` (line 191) `def _init_grid_topology(self, N)`
- `get_topology` (line 213) `def get_topology(self)`
- `calculate_ortho_loss` (line 225) `def calculate_ortho_loss(self)`
- `forward` (line 249) `def forward(self, x)`
- `lambda_topo` (line 317) `def lambda_topo(epoch)`
- `lambda_general` (line 321) `def lambda_general(epoch)`

#### `apex.py`
**Path:** `apex.py`

**Classes:**
- `DataEnvironment` (line 24) `class DataEnvironment`
- `LiquidNeuron` (line 50) `class LiquidNeuron`
- `SovereignAttention` (line 69) `class SovereignAttention`
- `DualPhaseMemory` (line 78) `class DualPhaseMemory`
- `ElasticMemory` (line 93) `class ElasticMemory`
- `ChimeraNetwork` (line 136) `class ChimeraNetwork`

**Functions:**
- `seed_everything` (line 14) `def seed_everything(seed)`
- `train_and_audit` (line 160) `def train_and_audit(name, use_ewc)`
- `__init__` (line 25) `def __init__(self)`
- `get_train_batch` (line 35) `def get_train_batch(self, phase, batch_size)`
- `__init__` (line 51) `def __init__(self, d_in, d_out)`
- `forward` (line 58) `def forward(self, x, gate)`
- `__init__` (line 70) `def __init__(self, d_in)`
- `forward` (line 74) `def forward(self, x, chaos)`
- `__init__` (line 79) `def __init__(self, d_in)`
- `forward` (line 82) `def forward(self, x, p)`
- `update` (line 86) `def update(self, x, p)`
- `__init__` (line 94) `def __init__(self, model, lambda_ewc)`
- `register_fisher` (line 101) `def register_fisher(self, dataset_x, dataset_y)`
- `penalty` (line 125) `def penalty(self)`
- `__init__` (line 137) `def __init__(self, d_in, d_hid, d_out)`
- `forward` (line 145) `def forward(self, x, phase)`

#### `app.py`
**Path:** `app.py`

**Functions:**
- `train_ai_model` (line 82) `def train_ai_model(df)` - *Entrena un modelo desde cero con todos los detalles de entrenamiento*
- `load_or_train_model` (line 156) `def load_or_train_model(df)` - *Carga modelo existente o entrena uno nuevo, y lo actualiza con nuevos datos*
- `apply_ai_predictions` (line 191) `def apply_ai_predictions(df, model, vectorizer)` - *Aplica predicciones del modelo al DataFrame*
- `apply_ai_predictions` (line 205) `def apply_ai_predictions(df, model, vectorizer)` - *Aplica predicciones del modelo al DataFrame*
- `analyze_ia_vs_rules` (line 219) `def analyze_ia_vs_rules(df)` - *Analiza discrepancias entre reglas y modelo IA*
- `load_and_clean_data_robust` (line 252) `def load_and_clean_data_robust(filepath)` - *Cargar y limpiar los datos de forma robusta*
- `parse_csv_manual` (line 294) `def parse_csv_manual(filepath)`
- `executive_kpis` (line 317) `def executive_kpis(df)`
- `strategic_okrs` (line 344) `def strategic_okrs(df, kpis)`
- `generate_visualizations` (line 377) `def generate_visualizations(df, kpis)`
- `export_report` (line 409) `def export_report(df, kpis, okrs, ia_analysis)`
- `basic_statistics` (line 454) `def basic_statistics(df)`
- `command_analysis` (line 467) `def command_analysis(df)`
- `network_analysis` (line 480) `def network_analysis(df)`
- `temporal_analysis` (line 492) `def temporal_analysis(df)`
- `statistical_analysis` (line 500) `def statistical_analysis(df)`
- `security_insights` (line 508) `def security_insights(df)`
- `main` (line 530) `def main()`

#### `auto_regulation_working.py`
**Path:** `auto_regulation_working.py`

**Classes:**
- `Config` (line 20) `class Config`
- `DataEnvironment` (line 38) `class DataEnvironment`
- `AutoRegulationSystem` (line 72) `class AutoRegulationSystem`
- `PhysioChimeraFixed` (line 103) `class PhysioChimeraFixed`

**Functions:**
- `seed_everything` (line 28) `def seed_everything(seed)`
- `demo_auto_regulation` (line 193) `def demo_auto_regulation()`
- `__init__` (line 39) `def __init__(self)`
- `get_batch` (line 49) `def get_batch(self, phase, bs)`
- `get_full` (line 63) `def get_full(self)`
- `get_w2` (line 66) `def get_w2(self)`
- `__init__` (line 73) `def __init__(self, size)`
- `update` (line 78) `def update(self, input_variance, loss_gradient, phase)`
- `get_stability` (line 95) `def get_stability(self)`
- `__init__` (line 104) `def __init__(self, config)`
- `forward` (line 129) `def forward(self, x, global_step, phase, prev_loss)`

#### `bicamera.py.py`
**Path:** `bicamera.py.py`

**Classes:**
- `LiquidNeuron` (line 107) `class LiquidNeuron`
- `RightHemisphere` (line 180) `class RightHemisphere`
- `LeftHemisphere` (line 201) `class LeftHemisphere`
- `CorpusCallosum` (line 298) `class CorpusCallosum`
- `NeuroLogosBicameral` (line 313) `class NeuroLogosBicameral`
- `NeuralDiagnostics` (line 334) `class NeuralDiagnostics`
- `Flickr8kDataset` (line 404) `class Flickr8kDataset(Dataset)`
- `LifeCycle` (line 466) `class LifeCycle`

**Functions:**
- `setup_flickr8k` (line 34) `def setup_flickr8k(data_dir)` - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 443) `def build_vocab_flickr(captions_file, vocab_size)`
- `train_bicameral` (line 481) `def train_bicameral()`
- `__init__` (line 108) `def __init__(self, in_dim, out_dim)`
- `forward` (line 125) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 154) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 181) `def __init__(self, output_dim)`
- `forward` (line 192) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 202) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 224) `def forward(self, visual_context, captions, max_len, return_gate)`
- `_get_init_state` (line 278) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 283) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 299) `def __init__(self, dim)`
- `forward` (line 307) `def forward(self, right_features)`
- `__init__` (line 314) `def __init__(self, vocab_size)`
- `forward` (line 320) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- `__init__` (line 335) `def __init__(self)`
- `measure_callosal_flow` (line 346) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 353) `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `update` (line 357) `def update(self)`
- `get_recent_avg` (line 362) `def get_recent_avg(self, key, n)`
- `report` (line 367) `def report(self, epoch)`
- `__init__` (line 405) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 423) `def __len__(self)`
- `__getitem__` (line 426) `def __getitem__(self, idx)`
- `__init__` (line 467) `def __init__(self, total_epochs)`
- `get_plasticity` (line 470) `def get_plasticity(self, epoch)`

#### `bicameral.py`
**Path:** `bicameral.py`

**Classes:**
- `HomeostaticRegulator` (line 100) `class HomeostaticRegulator`
- `PhysioNeuron` (line 123) `class PhysioNeuron`
- `RightHemisphere` (line 164) `class RightHemisphere`
- `LeftHemisphere` (line 199) `class LeftHemisphere`
- `CorpusCallosum` (line 276) `class CorpusCallosum`
- `NeuroLogosBicameralFisiologico` (line 290) `class NeuroLogosBicameralFisiologico`
- `NeuralDiagnostics` (line 309) `class NeuralDiagnostics`
- `Flickr8kDataset` (line 383) `class Flickr8kDataset(Dataset)`
- `LifeCycle` (line 436) `class LifeCycle`

**Functions:**
- `seed_all` (line 38) `def seed_all(seed)`
- `setup_flickr8k` (line 46) `def setup_flickr8k(data_dir)`
- `build_vocab_flickr` (line 416) `def build_vocab_flickr(captions_file, vocab_size)`
- `train_bicameral_fisiologico` (line 447) `def train_bicameral_fisiologico()`
- `__init__` (line 101) `def __init__(self)`
- `forward` (line 109) `def forward(self, stress, excitation, fatigue, entropy, phase, loss_signal)`
- `__init__` (line 124) `def __init__(self, in_dim, out_dim)`
- `forward` (line 138) `def forward(self, x, global_loss)`
- `__init__` (line 165) `def __init__(self, output_dim, num_nodes)`
- `forward` (line 178) `def forward(self, image, global_loss)`
- `__init__` (line 200) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 219) `def forward(self, visual_context, captions, max_len, return_gate)`
- `_get_init_state` (line 258) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 263) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 277) `def __init__(self, dim)`
- `forward` (line 284) `def forward(self, right_features)`
- `__init__` (line 291) `def __init__(self, vocab_size)`
- `forward` (line 297) `def forward(self, image, captions, global_loss, return_diagnostics)`
- `__init__` (line 310) `def __init__(self)`
- `measure_callosal_flow` (line 326) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 333) `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `update` (line 337) `def update(self)`
- `get_recent_avg` (line 342) `def get_recent_avg(self, key, n)`
- `report` (line 347) `def report(self, epoch)`
- `__init__` (line 384) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 400) `def __len__(self)`
- `__getitem__` (line 403) `def __getitem__(self, idx)`
- `__init__` (line 437) `def __init__(self, total_epochs)`
- `get_global_loss_proxy` (line 440) `def get_global_loss_proxy(self, epoch)`

#### `bicameral2.py`
**Path:** `bicameral2.py`

**Classes:**
- `TinyVisualEncoder` (line 83) `class TinyVisualEncoder`
- `MinimalLiquidNeuron` (line 104) `class MinimalLiquidNeuron`
- `RightHemisphere` (line 126) `class RightHemisphere`
- `LeftHemisphere` (line 136) `class LeftHemisphere`
- `CorpusCallosum` (line 176) `class CorpusCallosum`
- `NeuroLogosBicameralUltra` (line 186) `class NeuroLogosBicameralUltra`
- `DemocraticDiagnostics` (line 206) `class DemocraticDiagnostics`
- `Flickr8kDataset` (line 246) `class Flickr8kDataset(Dataset)`

**Functions:**
- `setup_flickr8k` (line 32) `def setup_flickr8k(data_dir)`
- `build_vocab` (line 274) `def build_vocab(captions_file, size)`
- `train_ultra` (line 290) `def train_ultra()`
- `__init__` (line 84) `def __init__(self, output_dim)`
- `forward` (line 98) `def forward(self, x)`
- `__init__` (line 105) `def __init__(self, in_dim, out_dim)`
- `forward` (line 113) `def forward(self, x)`
- `__init__` (line 127) `def __init__(self, output_dim)`
- `forward` (line 131) `def forward(self, x)`
- `__init__` (line 137) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 143) `def forward(self, visual_ctx, captions, max_len)`
- `__init__` (line 177) `def __init__(self, dim)`
- `forward` (line 180) `def forward(self, x)`
- `__init__` (line 187) `def __init__(self, vocab_size)`
- `forward` (line 192) `def forward(self, image, captions, return_diagnostics)`
- `__init__` (line 207) `def __init__(self)`
- `measure_flow` (line 212) `def measure_flow(self, r, l)`
- `vocab_diversity` (line 217) `def vocab_diversity(self, tokens, V)`
- `update` (line 219) `def update(self)`
- `avg` (line 223) `def avg(self, k, n)`
- `report` (line 226) `def report(self, epoch)`
- `__init__` (line 247) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 261) `def __len__(self)`
- `__getitem__` (line 262) `def __getitem__(self, idx)`

#### `bicameral3.py`
**Path:** `bicameral3.py`

**Classes:**
- `LiquidNeuron` (line 107) `class LiquidNeuron`
- `RightHemisphere` (line 180) `class RightHemisphere`
- `LeftHemisphere` (line 201) `class LeftHemisphere`
- `CorpusCallosum` (line 298) `class CorpusCallosum`
- `NeuroLogosBicameral` (line 313) `class NeuroLogosBicameral`
- `NeuralDiagnostics` (line 334) `class NeuralDiagnostics`
- `Flickr8kDataset` (line 404) `class Flickr8kDataset(Dataset)`
- `LifeCycle` (line 466) `class LifeCycle`

**Functions:**
- `setup_flickr8k` (line 34) `def setup_flickr8k(data_dir)` - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 443) `def build_vocab_flickr(captions_file, vocab_size)`
- `train_bicameral` (line 481) `def train_bicameral()`
- `__init__` (line 108) `def __init__(self, in_dim, out_dim)`
- `forward` (line 125) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 154) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 181) `def __init__(self, output_dim)`
- `forward` (line 192) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 202) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 224) `def forward(self, visual_context, captions, max_len, return_gate)`
- `_get_init_state` (line 278) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 283) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 299) `def __init__(self, dim)`
- `forward` (line 307) `def forward(self, right_features)`
- `__init__` (line 314) `def __init__(self, vocab_size)`
- `forward` (line 320) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- `__init__` (line 335) `def __init__(self)`
- `measure_callosal_flow` (line 346) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 353) `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `update` (line 357) `def update(self)`
- `get_recent_avg` (line 362) `def get_recent_avg(self, key, n)`
- `report` (line 367) `def report(self, epoch)`
- `__init__` (line 405) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 423) `def __len__(self)`
- `__getitem__` (line 426) `def __getitem__(self, idx)`
- `__init__` (line 467) `def __init__(self, total_epochs)`
- `get_plasticity` (line 470) `def get_plasticity(self, epoch)`

#### `bicameral_v2.py`
**Path:** `bicameral_v2.py`

**Classes:**
- `BCMPlasticity` (line 117) `class BCMPlasticity`
- `LiquidNeuron` (line 134) `class LiquidNeuron`
- `ResidualBlock` (line 225) `class ResidualBlock`
- `VisualCortex` (line 244) `class VisualCortex`
- `SymbioticBasisRefinement` (line 280) `class SymbioticBasisRefinement`
- `AdaptiveCombinatorialComplexLayer` (line 300) `class AdaptiveCombinatorialComplexLayer`
- `GraphNeuralLayer` (line 311) `class GraphNeuralLayer`
- `RightHemisphere` (line 342) `class RightHemisphere`
- `MiniUnconscious` (line 406) `class MiniUnconscious`
- `NestedUnconscious` (line 423) `class NestedUnconscious`
- `TopologicalCompressor` (line 466) `class TopologicalCompressor`
- `ConsciousCore` (line 486) `class ConsciousCore`
- `LeftHemisphere` (line 543) `class LeftHemisphere`
- `BioDecoder` (line 558) `class BioDecoder`
- `ConsciousCore` (line 667) `class ConsciousCore`
- `HomeostasisEngine` (line 725) `class HomeostasisEngine`
- `BicameralHomeostasis` (line 741) `class BicameralHomeostasis`
- `ReplayMemory` (line 773) `class ReplayMemory`
- `CorpusCallosum` (line 817) `class CorpusCallosum`
- `NeuroLogos` (line 868) `class NeuroLogos`
- `LifeCycle` (line 956) `class LifeCycle`
- `CIFARCaptions` (line 974) `class CIFARCaptions`

**Functions:**
- `compute_phi_effective` (line 24) `def compute_phi_effective(activations, k_partitions)` - *Φₑ efectivo: integración causal simplificada para batches
activations: [B, N, D] *
- `measure_spatial_richness` (line 52) `def measure_spatial_richness(activations)` - *FIX: Métrica de riqueza dimensional efectiva con escalado positivo garantizado
Preserva interfaz exacta: shannon_entropy, richness, vn_entropy (valores POSITIVOS)
Evita saturación en 2.0 y elimina negativos usando Participation Ratio escalado*
- `top_k_top_p_filtering` (line 98) `def top_k_top_p_filtering(logits, top_k, top_p, filter_value)` - *Filtro Top-K y Nucleus Sampling estandar*
- `create_grid_adjacency` (line 327) `def create_grid_adjacency(N, connectivity)` - *Crea matriz de adyacencia para grid cuadrado*
- `estimate_coherence` (line 1012) `def estimate_coherence(sentence, templates_per_class)`
- `train_logos` (line 1026) `def train_logos(use_nested)`
- `__init__` (line 118) `def __init__(self, neurons, tau_theta)`
- `forward` (line 123) `def forward(self, activity, dt)` - *dθ/dt = (E[activity²] - θ)/τ  →  dw/dt ∝ activity*(activity-θ)*
- `__init__` (line 135) `def __init__(self, in_dim, out_dim)`
- `forward` (line 158) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 204) `def consolidate_svd(self, repair_strength, timescale)` - *Mantener interfaz exacta pero implementar consolidación Hebbiana real*
- `__init__` (line 226) `def __init__(self, in_channels, out_channels, stride)`
- `forward` (line 238) `def forward(self, x)`
- `__init__` (line 245) `def __init__(self, output_dim, grid_size)`
- `_make_layer` (line 259) `def _make_layer(self, in_channels, out_channels, num_blocks, stride)`
- `forward` (line 265) `def forward(self, x)`
- `__init__` (line 281) `def __init__(self, dim, num_atoms)`
- `forward` (line 291) `def forward(self, x)`
- `__init__` (line 301) `def __init__(self, in_dim, hid_dim, num_nodes, config)`
- `forward` (line 307) `def forward(self, x, plasticity_gate)`
- `__init__` (line 312) `def __init__(self, dim, hidden_dim)`
- `forward` (line 322) `def forward(self, nodes, adjacency)`
- `__init__` (line 343) `def __init__(self, config)`
- `forward` (line 375) `def forward(self, image, adjacency, plasticity)`
- `__init__` (line 407) `def __init__(self)`
- `forward` (line 420) `def forward(self, x)`
- `__init__` (line 424) `def __init__(self, grid_size, output_dim)`
- `forward` (line 444) `def forward(self, x)`
- `__init__` (line 467) `def __init__(self, node_dim)`
- `forward` (line 476) `def forward(self, nodes, plasticity, transfer_rate)`
- `__init__` (line 487) `def __init__(self)`
- `forward` (line 499) `def forward(self, visual_features, plasticity, transfer_rate)`
- `get_liquid_module` (line 537) `def get_liquid_module(self)`
- `__init__` (line 544) `def __init__(self, use_nested)`
- `forward` (line 550) `def forward(self, image, callosal_input, plasticity, transfer_rate)`
- `__init__` (line 559) `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)`
- `forward` (line 578) `def forward(self, thought, visual_features, captions, max_len)`
- `_get_init_state` (line 659) `def _get_init_state(self, thought)`
- `__init__` (line 668) `def __init__(self)`
- `forward` (line 680) `def forward(self, visual_features, plasticity, transfer_rate)`
- `get_liquid_module` (line 718) `def get_liquid_module(self)`
- `__init__` (line 726) `def __init__(self)`
- `decide` (line 730) `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- `__init__` (line 742) `def __init__(self)`
- `decide` (line 751) `def decide(self, left_metrics, right_metrics, epoch, total_epochs)`
- `__init__` (line 774) `def __init__(self, capacity, noise_scale)`
- `store` (line 780) `def store(self, pattern)`
- `replay` (line 791) `def replay(self, batch_size)`
- `__init__` (line 818) `def __init__(self)`
- `forward` (line 824) `def forward(self, left_repr, right_repr, mode)`
- `__init__` (line 869) `def __init__(self, vocab_size, use_nested)`
- `forward` (line 895) `def forward(self, image, captions, plasticity, transfer_rate, mode, epoch)`
- `set_epoch` (line 950) `def set_epoch(self, epoch)`
- `__init__` (line 957) `def __init__(self, total_epochs)`
- `get_plasticity` (line 961) `def get_plasticity(self, epoch)`
- `__init__` (line 975) `def __init__(self)`
- `__len__` (line 999) `def __len__(self)`
- `__getitem__` (line 1002) `def __getitem__(self, idx)`

#### `bicameral_v3.py`
**Path:** `bicameral_v3.py`

**Classes:**
- `BCMPlasticity` (line 241) `class BCMPlasticity`
- `BioDecoder` (line 257) `class BioDecoder`
- `LiquidNeuron` (line 383) `class LiquidNeuron`
- `ResidualBlock` (line 486) `class ResidualBlock`
- `VisualCortex` (line 505) `class VisualCortex`
- `SymbioticBasisRefinement` (line 541) `class SymbioticBasisRefinement`
- `AdaptiveCombinatorialComplexLayer` (line 561) `class AdaptiveCombinatorialComplexLayer`
- `GraphNeuralLayer` (line 572) `class GraphNeuralLayer`
- `RightHemisphere` (line 603) `class RightHemisphere`
- `MiniUnconscious` (line 667) `class MiniUnconscious`
- `NestedUnconscious` (line 684) `class NestedUnconscious`
- `TopologicalCompressor` (line 727) `class TopologicalCompressor`
- `ConsciousCore` (line 744) `class ConsciousCore`
- `LeftHemisphere` (line 846) `class LeftHemisphere`
- `BioDecoder` (line 860) `class BioDecoder`
- `CorpusCallosum` (line 991) `class CorpusCallosum`
- `HomeostasisEngine` (line 1039) `class HomeostasisEngine`
- `BicameralHomeostasis` (line 1082) `class BicameralHomeostasis`
- `ReplayMemory` (line 1138) `class ReplayMemory`
- `NeuroLogos` (line 1182) `class NeuroLogos`
- `LifeCycle` (line 1280) `class LifeCycle`
- `CIFARCaptions` (line 1298) `class CIFARCaptions`

**Functions:**
- `compute_phi_effective` (line 22) `def compute_phi_effective(activations, k_partitions)` - *Φₑ con manejo robusto de dimensiones pequeñas*
- `compute_spatial_diversity` (line 75) `def compute_spatial_diversity(activations)` - *Diversidad basada en correlación inversa de Pearson.
VERSIÓN ROBUSTA: Protegida contra NaNs por errores de precisión flotante.*
- `compute_activation_entropy` (line 132) `def compute_activation_entropy(activations)` - *Shannon entropy sobre la distribución de activaciones
Target range: [2.5, 4.5] bits*
- `measure_neural_complexity` (line 163) `def measure_neural_complexity(activations)` - *Medición corregida con formato [B, D, N] para neuronas reales*
- `measure_spatial_richness` (line 212) `def measure_spatial_richness(activations)` - *Wrapper para compatibilidad con código existente*
- `top_k_top_p_filtering` (line 221) `def top_k_top_p_filtering(logits, top_k, top_p, filter_value)` - *Filtro Top-K y Nucleus Sampling estándar*
- `create_grid_adjacency` (line 588) `def create_grid_adjacency(N, connectivity)` - *Crea matriz de adyacencia para grid cuadrado*
- `estimate_coherence` (line 1336) `def estimate_coherence(sentence, templates_per_class)`
- `to_float` (line 1349) `def to_float(val)`
- `train_logos` (line 1355) `def train_logos(use_nested)`
- `__init__` (line 242) `def __init__(self, neurons, tau_theta)`
- `forward` (line 247) `def forward(self, activity, dt)` - *dθ/dt = (E[activity²] - θ)/τ  →  dw/dt ∝ activity*(activity-θ)*
- `__init__` (line 258) `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)`
- `forward` (line 281) `def forward(self, thought, visual_features, captions, max_len)`
- `_get_init_state` (line 377) `def _get_init_state(self, thought)`
- `__init__` (line 384) `def __init__(self, in_dim, out_dim)`
- `forward` (line 405) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 467) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 487) `def __init__(self, in_channels, out_channels, stride)`
- `forward` (line 499) `def forward(self, x)`
- `__init__` (line 506) `def __init__(self, output_dim, grid_size)`
- `_make_layer` (line 520) `def _make_layer(self, in_channels, out_channels, num_blocks, stride)`
- `forward` (line 526) `def forward(self, x)`
- `__init__` (line 542) `def __init__(self, dim, num_atoms)`
- `forward` (line 552) `def forward(self, x)`
- `__init__` (line 562) `def __init__(self, in_dim, hid_dim, num_nodes, config)`
- `forward` (line 568) `def forward(self, x, plasticity_gate)`
- `__init__` (line 573) `def __init__(self, dim, hidden_dim)`
- `forward` (line 583) `def forward(self, nodes, adjacency)`
- `__init__` (line 604) `def __init__(self, config)`
- `forward` (line 636) `def forward(self, image, adjacency, plasticity)`
- `__init__` (line 668) `def __init__(self)`
- `forward` (line 681) `def forward(self, x)`
- `__init__` (line 685) `def __init__(self, grid_size, output_dim)`
- `forward` (line 705) `def forward(self, x)`
- `__init__` (line 728) `def __init__(self, node_dim)`
- `forward` (line 737) `def forward(self, nodes, plasticity, transfer_rate)`
- `__init__` (line 745) `def __init__(self)`
- `_create_rotation_matrix` (line 780) `def _create_rotation_matrix(self, dim, angle, device)` - *Crea matriz de rotación en espacio de alta dimensión*
- `forward` (line 790) `def forward(self, visual_features, plasticity, transfer_rate)`
- `get_liquid_module` (line 839) `def get_liquid_module(self)`
- `__init__` (line 847) `def __init__(self, use_nested)`
- `forward` (line 853) `def forward(self, image, callosal_input, plasticity, transfer_rate)`
- `__init__` (line 861) `def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim)`
- `forward` (line 883) `def forward(self, thought, visual_features, captions, max_len)`
- `_get_init_state` (line 982) `def _get_init_state(self, thought)`
- `__init__` (line 992) `def __init__(self)`
- `forward` (line 999) `def forward(self, left_repr, right_repr, mode)`
- `__init__` (line 1040) `def __init__(self)`
- `decide` (line 1049) `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- `__init__` (line 1083) `def __init__(self)`
- `decide` (line 1097) `def decide(self, left_metrics, right_metrics, epoch, total_epochs)`
- `__init__` (line 1139) `def __init__(self, capacity, noise_scale)`
- `store` (line 1145) `def store(self, pattern)`
- `replay` (line 1155) `def replay(self, batch_size)`
- `__init__` (line 1183) `def __init__(self, vocab_size, use_nested)`
- `forward` (line 1203) `def forward(self, image, captions, plasticity, transfer_rate, mode, epoch, labels)`
- `set_epoch` (line 1274) `def set_epoch(self, epoch)`
- `__init__` (line 1281) `def __init__(self, total_epochs)`
- `get_plasticity` (line 1285) `def get_plasticity(self, epoch)`
- `__init__` (line 1299) `def __init__(self)`
- `__len__` (line 1323) `def __len__(self)`
- `__getitem__` (line 1326) `def __getitem__(self, idx)`

#### `caquita.py`
**Path:** `caquita.py`

**Classes:**
- `DiagnosticConfig` (line 31) `class DiagnosticConfig`
- `RealWorldEnvironment` (line 50) `class RealWorldEnvironment`
- `LiquidNeuron` (line 78) `class LiquidNeuron`
- `TraumaResponseSchedulerV2_ORIGINAL` (line 103) `class TraumaResponseSchedulerV2_ORIGINAL` - *Versión ORIGINAL de v8.5 (con el bug)*
- `TraumaResponseSchedulerV2_FIXED` (line 168) `class TraumaResponseSchedulerV2_FIXED` - *Versión CORREGIDA que compara con fase anterior*
- `ChaosAdaptiveFilter_ORIGINAL` (line 228) `class ChaosAdaptiveFilter_ORIGINAL` - *Versión ORIGINAL que detecta ruido por varianza*
- `DiagnosticModel` (line 264) `class DiagnosticModel`

**Functions:**
- `seed_everything` (line 40) `def seed_everything(seed)`
- `train_diagnostic` (line 327) `def train_diagnostic(config, env, experiment_name)` - *Entrenamiento con logs detallados*
- `run_diagnostic_ablation` (line 406) `def run_diagnostic_ablation()`
- `__init__` (line 51) `def __init__(self)`
- `get_batch` (line 63) `def get_batch(self, phase, batch_size)`
- `__init__` (line 79) `def __init__(self, in_dim, out_dim)`
- `forward` (line 88) `def forward(self, x, plasticity_gate)`
- `__init__` (line 105) `def __init__(self)`
- `update_phase_performance` (line 112) `def update_phase_performance(self, phase_idx, metrics)`
- `detect_trauma_level` (line 122) `def detect_trauma_level(self, phase_idx, current_metrics)`
- `generate_response` (line 148) `def generate_response(self, trauma_level, phase_idx, chaos_detected)`
- `__init__` (line 170) `def __init__(self)`
- `update_phase_performance` (line 176) `def update_phase_performance(self, phase_idx, metrics)`
- `detect_trauma_level` (line 185) `def detect_trauma_level(self, phase_idx, current_metrics)`
- `generate_response` (line 208) `def generate_response(self, trauma_level, phase_idx, chaos_detected)`
- `__init__` (line 230) `def __init__(self)`
- `extract_noise_features` (line 240) `def extract_noise_features(self, x)`
- `detect_chaos` (line 253) `def detect_chaos(self, x)`
- `__init__` (line 265) `def __init__(self, config, use_liquid, use_trs_original, use_trs_fixed, use_caf)`
- `forward` (line 285) `def forward(self, x, phase_idx, current_metrics)`

#### `chatgpt.py`
**Path:** `chatgpt.py`

**Classes:**
- `Config` (line 19) `class Config`
- `DataEnvironment` (line 41) `class DataEnvironment`
- `HomeostaticRegulator` (line 70) `class HomeostaticRegulator`
- `PhysioNeuron` (line 93) `class PhysioNeuron`
- `NeuroPhysioBicameral` (line 140) `class NeuroPhysioBicameral`
- `NeuralDiagnostics` (line 191) `class NeuralDiagnostics`

**Functions:**
- `seed_all` (line 33) `def seed_all(seed)`
- `train` (line 225) `def train()`
- `__init__` (line 42) `def __init__(self)`
- `get_batch` (line 53) `def get_batch(self, phase, bs)`
- `__init__` (line 71) `def __init__(self)`
- `forward` (line 81) `def forward(self, stress, excitation, fatigue, loss_signal)`
- `__init__` (line 94) `def __init__(self, d)`
- `forward` (line 107) `def forward(self, x, task_loss)`
- `__init__` (line 141) `def __init__(self, config)`
- `count_parameters` (line 163) `def count_parameters(self)`
- `forward` (line 166) `def forward(self, x, task_loss)`
- `__init__` (line 192) `def __init__(self)`
- `update` (line 201) `def update(self, loss, liquid_norm, phys)`
- `avg` (line 208) `def avg(self, k, n)`
- `report` (line 211) `def report(self, step, phase)`

#### `cifar3.py`
**Path:** `cifar3.py`

**Classes:**
- `FastSlowLinear` (line 51) `class FastSlowLinear` - *Linear layer con pesos lentos (backprop) y pesos rápidos (hebbianos).
Incluye decay temporal y normalización L2 estricta para estabilidad.*
- `DualSystemModule` (line 129) `class DualSystemModule`
- `ConsciousnessModule` (line 147) `class ConsciousnessModule` - *Módulo de conciencia con Φₑ más estable y relevante.*
- `OmniBrainFastSlow` (line 193) `class OmniBrainFastSlow`

**Functions:**
- `compute_phi_effective` (line 30) `def compute_phi_effective(activity)`
- `get_cifar10_loaders` (line 245) `def get_cifar10_loaders(batch_size)`
- `evaluate` (line 261) `def evaluate(model, loader, device)`
- `train` (line 274) `def train()`
- `__init__` (line 56) `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- `reset_fast_weights` (line 74) `def reset_fast_weights(self)` - *Reinicia los pesos rápidos al inicio de cada batch.*
- `update_fast_weights` (line 79) `def update_fast_weights(self, x)` - *Actualiza fast weights usando regla hebbiana con decay y normalización.*
- `forward` (line 105) `def forward(self, x)`
- `end_of_batch` (line 116) `def end_of_batch(self)` - *Limpia caché al final del batch para permitir reinicio en el siguiente.*
- `get_fast_norm` (line 120) `def get_fast_norm(self)` - *Retorna la norma L2 de los fast weights para monitoreo homeostático.*
- `__init__` (line 130) `def __init__(self, dim)`
- `forward` (line 138) `def forward(self, x)`
- `__init__` (line 149) `def __init__(self, features)`
- `compute_phi_effective_robust` (line 160) `def compute_phi_effective_robust(self, activity)` - *Φₑ más robusto usando promedio móvil y ventana temporal.*
- `forward` (line 184) `def forward(self, x)`
- `__init__` (line 194) `def __init__(self)`
- `forward` (line 226) `def forward(self, x)`
- `reset_all_fast_weights` (line 233) `def reset_all_fast_weights(self)`
- `get_fast_norms` (line 238) `def get_fast_norms(self)`

#### `cifar4.py`
**Path:** `cifar4.py`

**Classes:**
- `FastSlowLinear` (line 51) `class FastSlowLinear` - *Linear layer con pesos lentos (backprop) y pesos rápidos (hebbianos).
Incluye decay temporal y normalización L2 estricta para estabilidad.*
- `DualSystemModule` (line 116) `class DualSystemModule`
- `ConsciousnessModule` (line 134) `class ConsciousnessModule` - *Módulo de conciencia con Φₑ más estable y relevante.*
- `OmniBrainFastSlow` (line 176) `class OmniBrainFastSlow`

**Functions:**
- `compute_phi_effective` (line 30) `def compute_phi_effective(activity)`
- `get_cifar10_loaders` (line 217) `def get_cifar10_loaders(batch_size)`
- `evaluate` (line 238) `def evaluate(model, loader, device)`
- `train` (line 255) `def train()`
- `__init__` (line 56) `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- `reset_fast_weights` (line 72) `def reset_fast_weights(self)`
- `update_fast_weights` (line 76) `def update_fast_weights(self, x)`
- `forward` (line 95) `def forward(self, x)`
- `end_of_batch` (line 106) `def end_of_batch(self)`
- `get_fast_norm` (line 109) `def get_fast_norm(self)`
- `__init__` (line 117) `def __init__(self, dim)`
- `forward` (line 125) `def forward(self, x)`
- `__init__` (line 136) `def __init__(self, features)`
- `compute_phi_effective_robust` (line 147) `def compute_phi_effective_robust(self, activity)`
- `forward` (line 166) `def forward(self, x)`
- `__init__` (line 177) `def __init__(self)`
- `forward` (line 197) `def forward(self, x)`
- `reset_all_fast_weights` (line 204) `def reset_all_fast_weights(self)`
- `get_fast_norms` (line 209) `def get_fast_norms(self)`

#### `demo_auto_regulation.py`
**Path:** `demo_auto_regulation.py`

**Classes:**
- `Config` (line 21) `class Config`
- `DataEnvironment` (line 43) `class DataEnvironment`
- `AutoRegulationSystem` (line 77) `class AutoRegulationSystem`
- `SelfModifyingGates` (line 108) `class SelfModifyingGates`
- `PhysioChimeraFixed` (line 159) `class PhysioChimeraFixed`

**Functions:**
- `seed_everything` (line 33) `def seed_everything(seed)`
- `demo_auto_regulation` (line 243) `def demo_auto_regulation()`
- `__init__` (line 44) `def __init__(self)`
- `get_batch` (line 54) `def get_batch(self, phase, bs)`
- `get_full` (line 68) `def get_full(self)`
- `get_w2` (line 71) `def get_w2(self)`
- `__init__` (line 78) `def __init__(self, size)`
- `update` (line 84) `def update(self, input_variance, loss_gradient, phase)`
- `get_stability` (line 99) `def get_stability(self)`
- `__init__` (line 109) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 126) `def forward(self, x, adaptation_state)`
- `__init__` (line 160) `def __init__(self, config)`
- `forward` (line 184) `def forward(self, x, global_step, phase, prev_loss)`

#### `difract.py`
**Path:** `difract.py`

**Functions:**
- `visualize_uased_geometry` (line 4) `def visualize_uased_geometry()`

#### `dmg_core.py`
**Path:** `dmg_core.py`

**Classes:**
- `AdaptiveMagnitudeGate` (line 14) `class AdaptiveMagnitudeGate` - *Bio-inspired gating mechanism.
Acts as a learnable filter that suppresses signals exceeding a dynamic threshold.

Formula: Gate = Sigmoid( Gain * (Threshold - |x|^p * Sensitivity) )*
- `SparseTopologyLayer` (line 44) `class SparseTopologyLayer` - *Linear layer with sparse connectivity enforcement derived from 
Scale-Free (Barabási-Albert) graphs.*
- `DMGNetwork` (line 82) `class DMGNetwork` - *Robust Neural Network architecture using Sparse Layers and Dynamic Gating.
Designed for high noise resistance (MNIST-C / Adversarial robustness).*

**Functions:**
- `__init__` (line 21) `def __init__(self, base_threshold, power_order)`
- `forward` (line 30) `def forward(self, x)`
- `__init__` (line 49) `def __init__(self, in_features, out_features, sparsity_k)`
- `_generate_sparse_mask` (line 62) `def _generate_sparse_mask(self, k_neighbors)` - *Generates a Barabási-Albert scale-free mask.*
- `forward` (line 77) `def forward(self, x)`
- `__init__` (line 87) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 100) `def forward(self, x)`

#### `dualmind.py`
**Path:** `dualmind.py`

**Classes:**
- `HomeostasisEngine` (line 30) `class HomeostasisEngine`
- `LiquidNeuron` (line 57) `class LiquidNeuron` - *Neurona con fast weights hebbianos (de Síntesis)*
- `ConsciousSystem` (line 108) `class ConsciousSystem` - *Sistema de control ejecutivo que opera sobre representaciones
del sistema inconsciente. Implementa homeostasis y memoria de trabajo.*
- `NestedTopoLayer` (line 171) `class NestedTopoLayer` - *Capa de procesamiento topológico con memoria episódica.
Versión simplificada de TopoBrain enfocada en representaciones ricas.*
- `UnconsciousSystem` (line 217) `class UnconsciousSystem` - *Sistema inconsciente: Procesamiento automático y paralelo.
Arquitectura simplificada de TopoBrain para extracción de features.*
- `DualMind` (line 279) `class DualMind` - *Sistema dual de procesamiento:
- Inconsciente: Procesamiento automático, paralelo, topológico
- Consciente: Decisión deliberada, homeostática, serial*

**Functions:**
- `measure_spatial_richness` (line 15) `def measure_spatial_richness(activations)` - *Mide diversidad de representaciones mediante eigenspectro*
- `train_dualmind_phase1` (line 346) `def train_dualmind_phase1(model, train_loader, optimizer, device, epochs)` - *FASE 1: Preentrenamiento del sistema inconsciente
Objetivo: Aprender representaciones topológicas ricas*
- `train_dualmind_phase2` (line 401) `def train_dualmind_phase2(model, train_loader, optimizer, device, epochs)` - *FASE 2: Entrenamiento del sistema consciente
Objetivo: Aprender decisiones homeostáticas óptimas
Sistema inconsciente CONGELADO*
- `train_dualmind_phase3` (line 487) `def train_dualmind_phase3(model, train_loader, optimizer, device, epochs)` - *FASE 3: Co-adaptación de ambos sistemas
Objetivo: Refinamiento conjunto con retroalimentación*
- `evaluate_dualmind` (line 579) `def evaluate_dualmind(model, test_loader, device)` - *Evaluación del sistema dual*
- `run_dualmind_experiment` (line 604) `def run_dualmind_experiment()`
- `__init__` (line 31) `def __init__(self)`
- `decide` (line 35) `def decide(self, task_loss_val, richness_val, vn_entropy_val)` - *Motor de decisión homeostática con targets realistas y pesos equilibrados.
- target_entropy=1.8: Valor alcanzable dentro del rango [0, log(10)=2.3]
- target_richness=85.0: Por encima del estado inicial (66-74) para activar exploración
- Pesos reducidos para evitar dominancia de un solo drive*
- `__init__` (line 59) `def __init__(self, in_dim, out_dim)`
- `forward` (line 67) `def forward(self, x, plasticity_gate)` - *Neurona con plasticidad hebbiana de fast weights y decaimiento activación.
Incluye estabilización mediante decaimiento temporal de W_fast.*
- `consolidate_svd` (line 89) `def consolidate_svd(self, repair_strength)` - *Consolidación mediante SVD (modo sueño)*
- `__init__` (line 113) `def __init__(self, unconscious_dim, d_hid, d_out)`
- `forward` (line 136) `def forward(self, unconscious_features, plasticity_gate)` - *Input: Representaciones del sistema inconsciente [batch, unconscious_dim]
Output: logits, métricas homeostáticas*
- `get_structure_entropy` (line 157) `def get_structure_entropy(self)` - *Análisis de salud estructural mediante SVD*
- `__init__` (line 176) `def __init__(self, in_dim, hid_dim, num_nodes)`
- `forward` (line 188) `def forward(self, x_nodes, plasticity_gate)` - *x_nodes: [batch, num_nodes, in_dim]
output: [batch, num_nodes, hid_dim]*
- `get_topology_density` (line 209) `def get_topology_density(self)` - *Densidad de conexiones topológicas*
- `__init__` (line 222) `def __init__(self, in_channels, grid_size, hidden_dim)`
- `forward` (line 246) `def forward(self, x, plasticity_gate)` - *x: [batch, 3, 32, 32]
output: [batch, output_dim] representaciones inconscientes*
- `get_topology_stats` (line 264) `def get_topology_stats(self)` - *Estadísticas de topología del sistema inconsciente*
- `__init__` (line 285) `def __init__(self, in_channels, grid_size, hidden_dim, conscious_dim, num_classes)`
- `forward` (line 305) `def forward(self, x, mode)` - *Modos de operación:
- 'unconscious': Solo sistema inconsciente (rápido, baseline)
- 'conscious': Consciente sobre inconsciente (lento, preciso)
- 'dual': Ambos con retroalimentación (modo completo)*
- `get_system_status` (line 331) `def get_system_status(self)` - *Diagnóstico completo del sistema dual*

#### `dynamic.py`
**Path:** `dynamic.py`

**Classes:**
- `MasterConfig` (line 24) `class MasterConfig`
- `DataEnvironment` (line 36) `class DataEnvironment`
- `OmnibusController` (line 54) `class OmnibusController`
- `SovereignAttention` (line 88) `class SovereignAttention`
- `LiquidNeuron` (line 100) `class LiquidNeuron`
- `SovereignChimera` (line 137) `class SovereignChimera`

**Functions:**
- `seed_everything` (line 13) `def seed_everything(seed)`
- `run_final_showdown` (line 184) `def run_final_showdown(epochs, name, dynamic)`
- `__init__` (line 37) `def __init__(self)`
- `get_batch` (line 46) `def get_batch(self, phase, bs)`
- `__init__` (line 55) `def __init__(self)`
- `forward` (line 66) `def forward(self, x, h_slow)`
- `__init__` (line 89) `def __init__(self, d_in)`
- `forward` (line 94) `def forward(self, x, gain)`
- `__init__` (line 101) `def __init__(self, d_in, d_out)`
- `forward` (line 113) `def forward(self, x, plasticity, alpha)`
- `__init__` (line 138) `def __init__(self, config, dynamic_mode)`
- `forward` (line 151) `def forward(self, x)`

#### `dynamic2.py`
**Path:** `dynamic2.py`

**Classes:**
- `PhysioConfig` (line 24) `class PhysioConfig`
- `DataEnvironment` (line 36) `class DataEnvironment`
- `HomeostaticRegulator` (line 54) `class HomeostaticRegulator`
- `PhysioNeuron` (line 89) `class PhysioNeuron`
- `PhysioChimera` (line 162) `class PhysioChimera`

**Functions:**
- `seed_everything` (line 13) `def seed_everything(seed)`
- `run_physio_experiment` (line 178) `def run_physio_experiment(epochs, name, dynamic)`
- `__init__` (line 37) `def __init__(self)`
- `get_batch` (line 46) `def get_batch(self, phase, bs)`
- `__init__` (line 55) `def __init__(self, d_in)`
- `forward` (line 66) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 90) `def __init__(self, d_in, d_out, dynamic_mode)`
- `forward` (line 105) `def forward(self, x)`
- `__init__` (line 163) `def __init__(self, config, dynamic_mode)`
- `forward` (line 169) `def forward(self, x)`

#### `example_usage.py`
**Path:** `example_usage.py`

**Functions:**
- `demo_simple_monitoring` (line 20) `def demo_simple_monitoring()` - *Demostración de monitoreo básico*
- `demo_custom_monitoring` (line 38) `def demo_custom_monitoring()` - *Demostración de monitoreo personalizado*
- `demo_checkpoint_system` (line 85) `def demo_checkpoint_system()` - *Demostración del sistema de checkpointing*
- `demo_comparison_experiments` (line 142) `def demo_comparison_experiments()` - *Demostración de comparación entre experimentos*
- `create_demo_report` (line 194) `def create_demo_report()` - *Crear reporte demo completo*
- `main` (line 333) `def main()` - *Función principal de demostración*

#### `exampleww.py`
**Path:** `exampleww.py`

**Functions:**
- `run_single_experiment` (line 8) `def run_single_experiment(model_name, seed, epochs)`

#### `exodia_op_2.py`
**Path:** `exodia_op_2.py`

**Classes:**
- `HierarchicalEpisodicMemory` (line 340) `class HierarchicalEpisodicMemory` - *Memoria episódica optimizada con estabilización numérica en sampling
- Fixed: Clamp de surprise scores para evitar probabilidades degeneradas
- Fixed: Verificación explícita de NaN en operaciones de buffer*
- `NeurocognitiveSystem` (line 577) `class NeurocognitiveSystem`
- `LanguageMetrics` (line 774) `class LanguageMetrics` - *Métricas de calidad de generación*
- `LinguisticFeedbackLoop` (line 848) `class LinguisticFeedbackLoop`
- `LanguageMetrics` (line 965) `class LanguageMetrics`
- `CausalReasoningEngine` (line 1008) `class CausalReasoningEngine`
- `LanguageMetrics` (line 1087) `class LanguageMetrics`
- `StableLiquidNeuron` (line 1134) `class StableLiquidNeuron`
- `TricameralOutput` (line 1321) `class TricameralOutput(NamedTuple)`
- `TriangulatedMedicalSystem` (line 1360) `class TriangulatedMedicalSystem`
- `LeftHemisphere` (line 1510) `class LeftHemisphere`
- `AudioEncoder` (line 1833) `class AudioEncoder` - *Encoder de audio optimizado con:
- Pruning estructurado en canales Conv (30% reducción)
- Gradient checkpointing para memoria de activaciones
- Preparación para QAT INT8*
- `RightHemisphereTricameral` (line 1901) `class RightHemisphereTricameral` - *Hemisferio derecho optimizado:
- Gradient checkpointing obligatorio en ResNet50
- AudioEncoder con canales reducidos (90-180-360)
- Memoria activaciones reducida en 60%*
- `CorpusCallosumTrimodal` (line 1984) `class CorpusCallosumTrimodal` - *Corpus Callosum optimizado con:
- Dimensión base reducida: 512→320 dims (-37.5%)
- Bottleneck compartido para 3 canales (1 Linear vs 3 ModuleList)
- Flash Attention / xFormers compatible
- Gates fusionados en tensor único
Reducción: 2.1M → 0.88M parámetros (-58%)*
- `EnhancedDiagnosticsTricameral` (line 2207) `class EnhancedDiagnosticsTricameral`
- `NeuroLogosTricameral` (line 2515) `class NeuroLogosTricameral` - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 2550) `class Flickr8kMultimodalDataset(Dataset)` - *Dataset que carga imagen, audio del caption y texto desde Kaggle*

**Functions:**
- `preprocess_and_cache_spectrograms` (line 49) `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)` - *Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt
Esto elimina el cuello de botella de I/O durante entrenamiento*
- `apply_emergency_fixes` (line 120) `def apply_emergency_fixes(model)`
- `setup_flickr8k_with_audio` (line 142) `def setup_flickr8k_with_audio(data_dir)` - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 316) `def build_vocab_flickr(captions_file, vocab_size)` - *Construye vocabulario desde el archivo de captions*
- `forward` (line 1335) `def forward(self, image, audio, captions, epoch)`
- `compute_alignment_loss` (line 2666) `def compute_alignment_loss(visual_features, channels, alpha, epoch)` - *FIX: Pérdida auxiliar para alineación temprana de canales multimodales
Solo activa en épocas iniciales (epoch < 6)*
- `compute_tricameral_loss` (line 2695) `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, channels, epoch, lambda_reward, lambda_mtp)`
- `train_tricameral` (line 2812) `def train_tricameral()`
- `__init__` (line 347) `def __init__(self, working_capacity, short_term_capacity, importance_threshold)`
- `compute_surprise` (line 371) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` - *FIX: Clamp de cross-entropy para evitar infinitos*
- `calculate_importance` (line 386) `def calculate_importance(self, episode, surprise_score)` - *FIX: Clamp de surprise_score para evitar probabilidades degeneradas*
- `_calculate_novelty` (line 402) `def _calculate_novelty(self, episode)` - *FIX: Manejo de edge case cuando no hay memorias*
- `store_episode` (line 427) `def store_episode(self, image, audio, caption, surprise_score)`
- `_update_unified_buffer` (line 456) `def _update_unified_buffer(self)` - *FIX: Verificar integridad de scores antes de unificar*
- `sample` (line 470) `def sample(self, batch_size, memory_level)` - *FIX: Manejo de edge cases en sampling probabilístico*
- `_sample_from_buffer` (line 494) `def _sample_from_buffer(self, buffer, scores, batch_size)` - *FIX: Estabilización completa de probabilidades de sampling*
- `apply_forgetting_curve` (line 535) `def apply_forgetting_curve(self)`
- `_purge_low_score_memories` (line 545) `def _purge_low_score_memories(self)` - *FIX: Purga con threshold ajustado y verificación de scores*
- `__init__` (line 578) `def __init__(self)`
- `assess_reasoning_state` (line 598) `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` - *Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)*
- `assess_cognitive_state` (line 642) `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` - *Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)*
- `apply_cognitive_intervention` (line 688) `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` - *Aplica intervenciones basadas en estado lingüístico y de razonamiento*
- `sentence_bleu` (line 778) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 812) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 821) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 834) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 849) `def __init__(self, alpha, beta)`
- `_get_ngrams_cached` (line 863) `def _get_ngrams_cached(sentence, n)` - *FIX: Método estático con lru_cache para n-gramas*
- `compute_linguistic_reward` (line 872) `def compute_linguistic_reward(self, references, hypotheses)`
- `compute_cider` (line 911) `def compute_cider(self, reference, hypothesis)` - *FIX: Uso correcto del cache estático*
- `compute_spice` (line 925) `def compute_spice(self, reference, hypothesis)`
- `get_cache_stats` (line 937) `def get_cache_stats(self)` - *FIX: Estadísticas de cache actualizadas*
- `sentence_bleu` (line 967) `def sentence_bleu(reference, hypothesis, weights)`
- `token_accuracy` (line 990) `def token_accuracy(reference, hypothesis)`
- `word_overlap` (line 1000) `def word_overlap(reference, hypothesis)`
- `__init__` (line 1009) `def __init__(self, hidden_dim)`
- `reason_causally` (line 1036) `def reason_causally(self, observation, context)`
- `_predict_interventions` (line 1050) `def _predict_interventions(self, hypothesis, confidence)`
- `update_knowledge_graph` (line 1067) `def update_knowledge_graph(self, cause, effect, strength)`
- `query_causal_chain` (line 1073) `def query_causal_chain(self, start_node, end_node)`
- `sentence_bleu` (line 1089) `def sentence_bleu(reference, hypothesis, weights)`
- `token_accuracy` (line 1112) `def token_accuracy(reference, hypothesis)`
- `word_overlap` (line 1122) `def word_overlap(reference, hypothesis)`
- `__init__` (line 1136) `def __init__(self, in_dim, out_dim)`
- `forward` (line 1184) `def forward(self, x)`
- `_calculate_homeostasis_metric` (line 1219) `def _calculate_homeostasis_metric(self, output)` - *Calcula métrica de homeostasis con estabilización numérica*
- `hebbian_update` (line 1229) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 1278) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 1361) `def __init__(self)`
- `triangulate_signals` (line 1368) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `count_convergent_signals` (line 1379) `def count_convergent_signals(self, signals, pattern)`
- `diagnose_with_triangulation` (line 1382) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)`
- `apply_triangulated_intervention` (line 1427) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- `_reset_liquid_neuron` (line 1496) `def _reset_liquid_neuron(self, liquid_neuron)`
- `__init__` (line 1511) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 1596) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_apply_chain_of_thought` (line 1653) `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- `_apply_multi_token_prediction` (line 1694) `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- `_apply_structural_attention` (line 1737) `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- `_greedy_decode` (line 1759) `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- `_get_init_state` (line 1819) `def _get_init_state(self, visual_context)`
- `__init__` (line 1842) `def __init__(self, output_dim)`
- `forward` (line 1880) `def forward(self, mel_spec)`
- `__init__` (line 1909) `def __init__(self, output_dim)`
- `forward` (line 1947) `def forward(self, image, audio)`
- `__init__` (line 1994) `def __init__(self, dim)`
- `_apply_flash_attention` (line 2057) `def _apply_flash_attention(self, x)` - *Aplica Flash Attention nativa de PyTorch 2.0+
FIX: Corrección de dimensiones para seq_len variable*
- `forward` (line 2090) `def forward(self, right_features)` - *FIX: Manejo robusto de dimensiones y verificación de coherencia trimodal*
- `update_channel_fatigue` (line 2169) `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- `adjust_gates_by_fatigue` (line 2190) `def adjust_gates_by_fatigue(self)` - *Lógica original de ajuste de gates*
- `__init__` (line 2208) `def __init__(self)`
- `_get_cached_norm` (line 2231) `def _get_cached_norm(self, tensor, dim)` - *Cache de normalización con limpieza periódica*
- `measure_callosal_flow` (line 2248) `def measure_callosal_flow(self, right_features, left_context, channels)` - *Medición de coherencia multimodal con sincronización entre canales*
- `evaluate_reasoning_quality` (line 2305) `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- `calculate_synergy` (line 2342) `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 2353) `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 2362) `def update(self)`
- `get_recent_avg` (line 2379) `def get_recent_avg(self, key, n)`
- `visualize_fatigue_distribution` (line 2395) `def visualize_fatigue_distribution(self, epoch)`
- `visualize_reasoning_metrics` (line 2417) `def visualize_reasoning_metrics(self, epoch)`
- `report` (line 2429) `def report(self, epoch)`
- `__init__` (line 2518) `def __init__(self, vocab_size)`
- `forward` (line 2525) `def forward(self, image, audio, captions, epoch)`
- `__init__` (line 2553) `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache, cache_dir)`
- `__len__` (line 2610) `def __len__(self)`
- `__getitem__` (line 2613) `def __getitem__(self, idx)`

#### `exodia_optimized.py`
**Path:** `exodia_optimized.py`

**Classes:**
- `HierarchicalEpisodicMemory` (line 332) `class HierarchicalEpisodicMemory` - *Memoria episódica optimizada para Colab:
- working_capacity: 200→80 (-60%)
- short_term_capacity: 800→320 (-60%)
- long_term eliminado completamente
- Reducción overhead: 70% (de 2.4GB a 0.7GB)*
- `NeurocognitiveSystem` (line 546) `class NeurocognitiveSystem`
- `LanguageMetrics` (line 743) `class LanguageMetrics` - *Métricas de calidad de generación*
- `LinguisticFeedbackLoop` (line 817) `class LinguisticFeedbackLoop`
- `LanguageMetrics` (line 934) `class LanguageMetrics`
- `CausalReasoningEngine` (line 977) `class CausalReasoningEngine`
- `LanguageMetrics` (line 1056) `class LanguageMetrics`
- `StableLiquidNeuron` (line 1103) `class StableLiquidNeuron`
- `TriangulatedMedicalSystem` (line 1243) `class TriangulatedMedicalSystem`
- `LeftHemisphere` (line 1394) `class LeftHemisphere`
- `AudioEncoder` (line 1703) `class AudioEncoder` - *Encoder de audio optimizado con:
- Pruning estructurado en canales Conv (30% reducción)
- Gradient checkpointing para memoria de activaciones
- Preparación para QAT INT8*
- `RightHemisphereTricameral` (line 1778) `class RightHemisphereTricameral` - *Hemisferio derecho optimizado:
- Gradient checkpointing obligatorio en ResNet50
- AudioEncoder con canales reducidos (90-180-360)
- Memoria activaciones reducida en 60%*
- `CorpusCallosumTrimodal` (line 1861) `class CorpusCallosumTrimodal` - *Corpus Callosum optimizado con:
- Dimensión base reducida: 512→320 dims (-37.5%)
- Bottleneck compartido para 3 canales (1 Linear vs 3 ModuleList)
- Flash Attention / xFormers compatible
- Gates fusionados en tensor único
Reducción: 2.1M → 0.88M parámetros (-58%)*
- `EnhancedDiagnosticsTricameral` (line 2093) `class EnhancedDiagnosticsTricameral`
- `NeuroLogosTricameral` (line 2398) `class NeuroLogosTricameral` - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 2433) `class Flickr8kMultimodalDataset(Dataset)` - *Dataset que carga imagen, audio del caption y texto desde Kaggle*

**Functions:**
- `preprocess_and_cache_spectrograms` (line 47) `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)` - *Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt
Esto elimina el cuello de botella de I/O durante entrenamiento*
- `setup_flickr8k_with_audio` (line 120) `def setup_flickr8k_with_audio(data_dir)` - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 308) `def build_vocab_flickr(captions_file, vocab_size)` - *Construye vocabulario desde el archivo de captions*
- `compute_alignment_loss` (line 2549) `def compute_alignment_loss(visual_features, channels, alpha, epoch)` - *FIX: Pérdida auxiliar para alineación temprana de canales multimodales
Solo activa en épocas iniciales (epoch < 6)*
- `compute_tricameral_loss` (line 2577) `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, channels, epoch, lambda_reward, lambda_mtp)` - *FIX: Pérdida con término explícito de coherencia multimodal
Penaliza la falta de sincronización entre canales*
- `train_tricameral` (line 2650) `def train_tricameral()`
- `__init__` (line 341) `def __init__(self, working_capacity, short_term_capacity, importance_threshold)`
- `compute_surprise` (line 369) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` - *Sin cambios en la lógica*
- `calculate_importance` (line 380) `def calculate_importance(self, episode, surprise_score)` - *Sin cambios*
- `_calculate_novelty` (line 393) `def _calculate_novelty(self, episode)` - *Usa solo working/short_term (no long_term)*
- `store_episode` (line 416) `def store_episode(self, image, audio, caption, surprise_score)`
- `_update_unified_buffer` (line 445) `def _update_unified_buffer(self)` - *Solo working + short_term*
- `add` (line 450) `def add(self, image, audio, caption, surprise_score)` - *Alias para store_episode*
- `apply_forgetting_curve` (line 454) `def apply_forgetting_curve(self)` - *Decay más agresivo (ahorro overhead)*
- `_purge_low_score_memories` (line 467) `def _purge_low_score_memories(self)` - *Purga más agresiva (threshold mayor)*
- `sample` (line 487) `def sample(self, batch_size, memory_level)` - *Muestreo solo de working/short_term*
- `_sample_from_buffer` (line 511) `def _sample_from_buffer(self, buffer, scores, batch_size)` - *Lógica original sin cambios*
- `get_total_size` (line 539) `def get_total_size(self)` - *Solo working + short_term*
- `__init__` (line 547) `def __init__(self)`
- `assess_reasoning_state` (line 567) `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` - *Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)*
- `assess_cognitive_state` (line 611) `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` - *Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)*
- `apply_cognitive_intervention` (line 657) `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` - *Aplica intervenciones basadas en estado lingüístico y de razonamiento*
- `sentence_bleu` (line 747) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 781) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 790) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 803) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 818) `def __init__(self, alpha, beta)`
- `_get_ngrams_cached` (line 832) `def _get_ngrams_cached(sentence, n)` - *FIX: Método estático con lru_cache para n-gramas*
- `compute_linguistic_reward` (line 841) `def compute_linguistic_reward(self, references, hypotheses)`
- `compute_cider` (line 880) `def compute_cider(self, reference, hypothesis)` - *FIX: Uso correcto del cache estático*
- `compute_spice` (line 894) `def compute_spice(self, reference, hypothesis)`
- `get_cache_stats` (line 906) `def get_cache_stats(self)` - *FIX: Estadísticas de cache actualizadas*
- `sentence_bleu` (line 936) `def sentence_bleu(reference, hypothesis, weights)`
- `token_accuracy` (line 959) `def token_accuracy(reference, hypothesis)`
- `word_overlap` (line 969) `def word_overlap(reference, hypothesis)`
- `__init__` (line 978) `def __init__(self, hidden_dim)`
- `reason_causally` (line 1005) `def reason_causally(self, observation, context)`
- `_predict_interventions` (line 1019) `def _predict_interventions(self, hypothesis, confidence)`
- `update_knowledge_graph` (line 1036) `def update_knowledge_graph(self, cause, effect, strength)`
- `query_causal_chain` (line 1042) `def query_causal_chain(self, start_node, end_node)`
- `sentence_bleu` (line 1058) `def sentence_bleu(reference, hypothesis, weights)`
- `token_accuracy` (line 1081) `def token_accuracy(reference, hypothesis)`
- `word_overlap` (line 1091) `def word_overlap(reference, hypothesis)`
- `__init__` (line 1104) `def __init__(self, in_dim, out_dim)`
- `forward` (line 1146) `def forward(self, x)`
- `_calculate_homeostasis_metric` (line 1163) `def _calculate_homeostasis_metric(self, output)` - *Calcula métrica de homeostasis basada en la estabilidad del output*
- `hebbian_update` (line 1172) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 1210) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 1244) `def __init__(self)`
- `triangulate_signals` (line 1251) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `count_convergent_signals` (line 1262) `def count_convergent_signals(self, signals, pattern)`
- `diagnose_with_triangulation` (line 1265) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)`
- `apply_triangulated_intervention` (line 1310) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- `_reset_liquid_neuron` (line 1379) `def _reset_liquid_neuron(self, liquid_neuron)` - *Reset completo de una neurona líquida*
- `__init__` (line 1395) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 1477) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_apply_chain_of_thought` (line 1524) `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- `_greedy_decode` (line 1564) `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- `_apply_multi_token_prediction` (line 1625) `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- `_apply_structural_attention` (line 1667) `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- `_get_init_state` (line 1688) `def _get_init_state(self, visual_context)`
- `__init__` (line 1711) `def __init__(self, output_dim)`
- `forward` (line 1751) `def forward(self, mel_spec)`
- `__init__` (line 1786) `def __init__(self, output_dim)`
- `forward` (line 1824) `def forward(self, image, audio)`
- `__init__` (line 1871) `def __init__(self, dim)`
- `_apply_flash_attention` (line 1940) `def _apply_flash_attention(self, x)` - *Aplica Flash Attention nativa de PyTorch 2.0+*
- `forward` (line 1966) `def forward(self, right_features)`
- `update_channel_fatigue` (line 2054) `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` - *Lógica original de fatiga sin cambios*
- `adjust_gates_by_fatigue` (line 2076) `def adjust_gates_by_fatigue(self)` - *Lógica original de ajuste de gates*
- `__init__` (line 2094) `def __init__(self)`
- `_get_cached_norm` (line 2117) `def _get_cached_norm(self, tensor, dim)` - *Cache de normalización con limpieza periódica*
- `measure_callosal_flow` (line 2135) `def measure_callosal_flow(self, right_features, left_context, channels)` - *FIX: Medición de coherencia multimodal real con atención a diversidad
Incluye métricas de sincronización entre canales*
- `__init__` (line 2401) `def __init__(self, vocab_size)`
- `forward` (line 2407) `def forward(self, image, audio, captions, epoch)`
- `__init__` (line 2436) `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache, cache_dir)`
- `__len__` (line 2493) `def __len__(self)`
- `__getitem__` (line 2496) `def __getitem__(self, idx)`
- `evaluate_reasoning_quality` (line 2188) `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- `calculate_synergy` (line 2225) `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 2236) `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 2245) `def update(self)`
- `get_recent_avg` (line 2262) `def get_recent_avg(self, key, n)`
- `visualize_fatigue_distribution` (line 2278) `def visualize_fatigue_distribution(self, epoch)`
- `visualize_reasoning_metrics` (line 2302) `def visualize_reasoning_metrics(self, epoch)`
- `report` (line 2314) `def report(self, epoch)`

#### `final_sinergy_analysis.py`
**Path:** `final_sinergy_analysis.py`

**Classes:**
- `SinergyAnalysis` (line 12) `class SinergyAnalysis`

**Functions:**
- `main` (line 219) `def main()`
- `__init__` (line 13) `def __init__(self)`
- `print_header` (line 80) `def print_header(self)`
- `analyze_original_models` (line 87) `def analyze_original_models(self)`
- `analyze_sinergies` (line 99) `def analyze_sinergies(self)`
- `generate_scientific_matrix` (line 118) `def generate_scientific_matrix(self)`
- `calculate_synergy_breakthrough` (line 136) `def calculate_synergy_breakthrough(self)`
- `generate_conclusion` (line 171) `def generate_conclusion(self)`
- `save_results` (line 200) `def save_results(self)`

#### `gemini.py`
**Path:** `gemini.py`

**Classes:**
- `LiquidNeuron` (line 108) `class LiquidNeuron`
- `RightHemisphere` (line 177) `class RightHemisphere`
- `CorpusCallosum` (line 196) `class CorpusCallosum`
- `LeftHemisphere` (line 219) `class LeftHemisphere`
- `NeuroLogosBicameral` (line 333) `class NeuroLogosBicameral`
- `NeuralDiagnostics` (line 361) `class NeuralDiagnostics`
- `Flickr8kDataset` (line 431) `class Flickr8kDataset(Dataset)`
- `LifeCycle` (line 493) `class LifeCycle`

**Functions:**
- `setup_flickr8k` (line 40) `def setup_flickr8k(data_dir)` - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 470) `def build_vocab_flickr(captions_file, vocab_size)`
- `train_bicameral` (line 509) `def train_bicameral()`
- `__init__` (line 109) `def __init__(self, in_dim, out_dim)`
- `forward` (line 123) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 154) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 178) `def __init__(self, output_dim)`
- `forward` (line 187) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 197) `def __init__(self, dim)`
- `forward` (line 210) `def forward(self, right_features)`
- `__init__` (line 220) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 241) `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)`
- `_get_init_state` (line 313) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 318) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 334) `def __init__(self, vocab_size)`
- `forward` (line 340) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- `__init__` (line 362) `def __init__(self)`
- `measure_callosal_flow` (line 373) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 380) `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `update` (line 384) `def update(self)`
- `get_recent_avg` (line 389) `def get_recent_avg(self, key, n)`
- `report` (line 394) `def report(self, epoch)`
- `__init__` (line 432) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 450) `def __len__(self)`
- `__getitem__` (line 453) `def __getitem__(self, idx)`
- `__init__` (line 494) `def __init__(self, total_epochs)`
- `get_plasticity` (line 497) `def get_plasticity(self, epoch)`

#### `gemini2.py`
**Path:** `gemini2.py`

**Classes:**
- `LiquidNeuron` (line 108) `class LiquidNeuron`
- `RightHemisphere` (line 177) `class RightHemisphere`
- `CorpusCallosum` (line 196) `class CorpusCallosum`
- `LeftHemisphere` (line 219) `class LeftHemisphere`
- `NeuroLogosBicameral` (line 333) `class NeuroLogosBicameral`
- `NeuralDiagnostics` (line 361) `class NeuralDiagnostics`
- `Flickr8kDataset` (line 431) `class Flickr8kDataset(Dataset)`
- `LifeCycle` (line 493) `class LifeCycle`

**Functions:**
- `setup_flickr8k` (line 40) `def setup_flickr8k(data_dir)` - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 470) `def build_vocab_flickr(captions_file, vocab_size)`
- `train_bicameral` (line 509) `def train_bicameral()`
- `__init__` (line 109) `def __init__(self, in_dim, out_dim)`
- `forward` (line 123) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 154) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 178) `def __init__(self, output_dim)`
- `forward` (line 187) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 197) `def __init__(self, dim)`
- `forward` (line 210) `def forward(self, right_features)`
- `__init__` (line 220) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 241) `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)`
- `_get_init_state` (line 313) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 318) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 334) `def __init__(self, vocab_size)`
- `forward` (line 340) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- `__init__` (line 362) `def __init__(self)`
- `measure_callosal_flow` (line 373) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 380) `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `update` (line 384) `def update(self)`
- `get_recent_avg` (line 389) `def get_recent_avg(self, key, n)`
- `report` (line 394) `def report(self, epoch)`
- `__init__` (line 432) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 450) `def __len__(self)`
- `__getitem__` (line 453) `def __getitem__(self, idx)`
- `__init__` (line 494) `def __init__(self, total_epochs)`
- `get_plasticity` (line 497) `def get_plasticity(self, epoch)`

#### `gen_dataset.py`
**Path:** `gen_dataset.py`

**Functions:**
- `generate_one` (line 89) `def generate_one(key, text)`
- `main` (line 110) `def main()`

#### `get_dataset.py`
**Path:** `get_dataset.py`

**Functions:**
- `download_captions_only` (line 29) `def download_captions_only()` - *Descarga solo los captions de Flickr8k*
- `generate_one_audio` (line 67) `def generate_one_audio(text, output_path, max_retries)` - *Genera un audio con retry y rate limiting*
- `load_checkpoint` (line 114) `def load_checkpoint()` - *Carga el checkpoint de progreso*
- `save_checkpoint` (line 122) `def save_checkpoint(checkpoint)` - *Guarda el checkpoint de progreso*
- `generate_audios_with_checkpoints` (line 128) `def generate_audios_with_checkpoints()` - *Genera audios con checkpoints cada 500 archivos*
- `generate_audios_sync` (line 219) `def generate_audios_sync()` - *Wrapper síncrono con manejo de event loop*
- `compress_audios_only` (line 249) `def compress_audios_only()` - *Comprime solo los audios en zips pequeños*
- `create_audio_readme` (line 318) `def create_audio_readme(output_dir, metadata)` - *Crea README para el dataset de audios*
- `upload_to_huggingface` (line 388) `def upload_to_huggingface(dataset_dir)` - *Sube solo audios a Hugging Face*
- `main` (line 465) `def main()`
- `download_flickr8k` (line 540) `def download_flickr8k()` - *Descarga Flickr8k (solo necesitas ejecutar esto una vez)*
- `generate_audios` (line 602) `def generate_audios()` - *Genera audios con Edge-TTS - TOMA TIEMPO (~20-30 min)*
- `create_split_zips` (line 637) `def create_split_zips()` - *Crea múltiples zips pequeños para cumplir límites de GitHub*
- `generate_upload_instructions` (line 717) `def generate_upload_instructions(metadata)` - *Genera instrucciones para subir a GitHub*
- `upload_to_huggingface` (line 834) `def upload_to_huggingface(dataset_dir)` - *Sube directamente a Hugging Face (alternativa a GitHub)*
- `main` (line 888) `def main()`

#### `homeostatichope.py`
**Path:** `homeostatichope.py`

**Classes:**
- `Config` (line 15) `class Config`
- `RealWorldEnvironment` (line 49) `class RealWorldEnvironment`
- `OmniscientRegulator` (line 96) `class OmniscientRegulator` - *Motor homeostático con acceso TOTAL a señales internas críticas
y control DIRECTO de hiperparámetros en tiempo real*
- `AdaptiveLiquidMemory` (line 190) `class AdaptiveLiquidMemory` - *Memoria líquida que responde a controles homeostáticos*
- `HomeostaticSelfModMemory` (line 233) `class HomeostaticSelfModMemory`
- `ContinuumMemorySystem` (line 278) `class ContinuumMemorySystem`
- `OmniscientHopeModel` (line 304) `class OmniscientHopeModel`
- `ConsciousTrainer` (line 402) `class ConsciousTrainer`

**Functions:**
- `setup_device` (line 35) `def setup_device()`
- `set_seed` (line 40) `def set_seed(seed)`
- `pgd_attack` (line 368) `def pgd_attack(model, x, y, epsilon, steps, device, signals)`
- `run_conscious_experiment` (line 516) `def run_conscious_experiment(config, device)`
- `run_ablation` (line 610) `def run_ablation(device)`
- `__init__` (line 50) `def __init__(self, seed)`
- `get_batch` (line 73) `def get_batch(self, phase, batch_size)`
- `get_test_loader` (line 88) `def get_test_loader(self, batch_size)`
- `__init__` (line 102) `def __init__(self, d_model)`
- `forward` (line 128) `def forward(self, signals)` - *Args:
    signals: Diccionario con señales internas del sistema
Returns:
    controls: Diccionario con hiperparámetros ajustados*
- `__init__` (line 193) `def __init__(self, d_model)`
- `forward` (line 203) `def forward(self, x, controls)`
- `__init__` (line 234) `def __init__(self, d_model, hidden_dim)`
- `forward` (line 251) `def forward(self, x, controls)`
- `__init__` (line 279) `def __init__(self, frequencies, d_model, hidden_dim)`
- `forward` (line 293) `def forward(self, x, global_step)`
- `__init__` (line 305) `def __init__(self, config, n_features, n_classes)`
- `forward` (line 341) `def forward(self, x, signals, global_step)`
- `__init__` (line 403) `def __init__(self, model, config, device)`
- `train_step` (line 425) `def train_step(self, x, y, epsilon, global_step, phase)`
- `evaluate` (line 488) `def evaluate(self, test_loader, epsilon, phase)`

#### `hope.py`
**Path:** `hope.py`

**Classes:**
- `Config` (line 16) `class Config` - *Configuración para desafío realista con adversarial*
- `RealWorldEnvironment` (line 58) `class RealWorldEnvironment` - *Dataset Digits con separación en "mundos" para simular concept drift
Similar al segundo ejemplo pero adaptado para classification*
- `HomeostaticRegulator` (line 127) `class HomeostaticRegulator` - *Motor fisiológico que regula según estado interno*
- `LiquidMemory` (line 172) `class LiquidMemory` - *Memoria líquida con componente rápida y lenta*
- `EfficientSelfModMemory` (line 207) `class EfficientSelfModMemory` - *Self-modifying memory con homeostasis*
- `ContinuumMemorySystem` (line 287) `class ContinuumMemorySystem` - *CMS que acepta global_step correctamente*
- `HopePhysioModel` (line 320) `class HopePhysioModel` - *Hope + PhysioChimera para clasificación*
- `AdversarialTrainer` (line 441) `class AdversarialTrainer`

**Functions:**
- `setup_device` (line 41) `def setup_device()`
- `set_seed` (line 49) `def set_seed(seed)`
- `pgd_attack` (line 394) `def pgd_attack(model, x, y, epsilon, steps, device)` - *PGD adversarial attack - versión robusta*
- `run_real_world_experiment` (line 517) `def run_real_world_experiment(config, device)`
- `run_ablation` (line 614) `def run_ablation(device)`
- `__init__` (line 64) `def __init__(self, seed)`
- `get_batch` (line 98) `def get_batch(self, phase, batch_size)` - *Obtener batch según la fase de entrenamiento*
- `get_test_loader` (line 118) `def get_test_loader(self, batch_size)` - *Test loader completo*
- `__init__` (line 130) `def __init__(self, d_model)`
- `forward` (line 140) `def forward(self, x, h_prev, w_norm)` - *Calcula controles homeostáticos basados en:
- Estrés (varianza input)
- Excitación (magnitud activación)
- Fatiga (norma pesos)*
- `__init__` (line 175) `def __init__(self, d_model)`
- `forward` (line 186) `def forward(self, x, physio)` - *Args:
    x: (B, D)
    physio: Controles homeostáticos*
- `__init__` (line 210) `def __init__(self, d_model, hidden_dim)`
- `forward` (line 236) `def forward(self, x)` - *Args:
    x: (B, D)
Returns:
    output, h_current*
- `__init__` (line 290) `def __init__(self, frequencies, d_model, hidden_dim)`
- `forward` (line 304) `def forward(self, x, global_step)` - *Args:
    x: (B, D)
    global_step: Paso global*
- `__init__` (line 323) `def __init__(self, config, n_features, n_classes)`
- `reset_states` (line 361) `def reset_states(self)`
- `forward` (line 365) `def forward(self, x, global_step)` - *Args:
    x: (B, n_features)
    global_step: Paso global
Returns:
    logits: (B, n_classes)*
- `__init__` (line 442) `def __init__(self, model, config, device)`
- `train_step` (line 460) `def train_step(self, x, y, epsilon, global_step)` - *Un paso de entrenamiento con adversarial opcional*
- `evaluate` (line 490) `def evaluate(self, test_loader, epsilon)` - *Evaluación con ataque opcional*

#### `kimi.py`
**Path:** `kimi.py`

**Classes:**
- `PhysioState` (line 28) `class PhysioState`
- `SNE` (line 65) `class SNE`
- `BCMRegulated` (line 97) `class BCMRegulated`
- `LiquidRegulated` (line 118) `class LiquidRegulated`
- `VisualCortexRegulated` (line 145) `class VisualCortexRegulated`
- `MicroTopoBrainSNA` (line 166) `class MicroTopoBrainSNA`
- `Config` (line 185) `class Config`

**Functions:**
- `pgd_attack` (line 39) `def pgd_attack(model, x, y, eps, steps, alpha)` - *PGD-10 ataque con gradiente corregido para CPU*
- `get_loader` (line 191) `def get_loader()`
- `run_experiment` (line 201) `def run_experiment(seed, sne_enabled, ablated_organs)` - *Ejecuta un experimento completo con una seed*
- `scientific_ablation` (line 256) `def scientific_ablation()` - *Ejecuta el estudio científico completo*
- `__init__` (line 66) `def __init__(self, enabled)`
- `forward` (line 75) `def forward(self, state, loss)`
- `__init__` (line 98) `def __init__(self, sne, ablated)`
- `forward` (line 104) `def forward(self, act)`
- `__init__` (line 119) `def __init__(self, sne, ablated)`
- `forward` (line 127) `def forward(self, x)`
- `__init__` (line 146) `def __init__(self, sne, ablated)`
- `forward` (line 155) `def forward(self, img)`
- `__init__` (line 167) `def __init__(self, sne_enabled, ablated_organs)`
- `forward` (line 175) `def forward(self, x)`

#### `legendario.py`
**Path:** `legendario.py`

**Classes:**
- `MotorHomeostaticContext` (line 106) `class MotorHomeostaticContext` - *Contexto estable para motores homeostáticos*
- `OmniBrainModule` (line 117) `class OmniBrainModule` - *Módulo base estable para CPU*
- `PTSymmetricLayer` (line 132) `class PTSymmetricLayer(OmniBrainModule)` - *Capa PT-simétrica sin operaciones complejas problemáticas*
- `TopologicalLayer` (line 166) `class TopologicalLayer(OmniBrainModule)` - *Capa topológica estable sin dependencias problemáticas*
- `DualMindModule` (line 193) `class DualMindModule(OmniBrainModule)` - *Módulo dual estable para CPU*
- `ConsciousnessModule` (line 224) `class ConsciousnessModule(OmniBrainModule)` - *Módulo de conciencia estable*
- `OmniBrainCoordinator` (line 245) `class OmniBrainCoordinator` - *Coordinador sin mediciones problemáticas*
- `OmniBrain` (line 289) `class OmniBrain` - *¡El Pokémon legendario estable en CPU!*

**Functions:**
- `compute_phi_effective_approx` (line 27) `def compute_phi_effective_approx(activity)` - *Cálculo ESTABLE de Φₑ usando PCA (proporción de varianza explicada)
¡Sin errores de dimensiones! Basado en: "Practical measures of integrated information"*
- `compute_topological_metrics` (line 60) `def compute_topological_metrics(weights)` - *Cálculo ESTABLE de métricas topológicas (optimizado para CPU)*
- `estimate_energy_consumption` (line 89) `def estimate_energy_consumption(model, input_size)` - *Estimación conservadora de consumo energético para CPU*
- `train_omni_brain` (line 340) `def train_omni_brain(model, epochs, batch_size, device)` - *Entrenamiento estable y rápido en CPU*
- `__init__` (line 119) `def __init__(self, module_name, enabled)`
- `update_performance` (line 125) `def update_performance(self, metrics)`
- `__init__` (line 135) `def __init__(self, in_features, out_features)`
- `compute_pt_phase` (line 144) `def compute_pt_phase(self)` - *Cálculo estable de fase PT sin números complejos*
- `forward` (line 152) `def forward(self, x, params)`
- `__init__` (line 169) `def __init__(self, in_features, out_features)`
- `update_topology` (line 176) `def update_topology(self, connectivity)` - *Actualizar máscara topológica basada en conectividad deseada*
- `forward` (line 183) `def forward(self, x, params)`
- `__init__` (line 196) `def __init__(self, features)`
- `forward` (line 211) `def forward(self, x, params)`
- `__init__` (line 227) `def __init__(self, features)`
- `forward` (line 232) `def forward(self, x, params)`
- `__init__` (line 248) `def __init__(self)`
- `measure_network_state` (line 251) `def measure_network_state(self, model, batch_data)` - *Mediciones ESTABLES para CPU*
- `__init__` (line 292) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 317) `def forward(self, x)`

#### `legendario2.py`
**Path:** `legendario2.py`

**Classes:**
- `MotorHomeostaticContext` (line 34) `class MotorHomeostaticContext` - *Contexto para un motor homeostático*
- `PTSymmetricMotor` (line 64) `class PTSymmetricMotor(MotorHomeostaticContext)` - *Motor para controlar parámetros PT-similares*
- `TopologicalMotor` (line 100) `class TopologicalMotor(MotorHomeostaticContext)` - *Motor para controlar conectividad y topología*
- `EnergyHomeostaticMotor` (line 127) `class EnergyHomeostaticMotor(MotorHomeostaticContext)` - *Motor para controlar eficiencia energética*
- `ConsciousnessMotor` (line 157) `class ConsciousnessMotor(MotorHomeostaticContext)` - *Motor para controlar métricas de conciencia (Φₑ)*
- `DualSystemMotor` (line 185) `class DualSystemMotor(MotorHomeostaticContext)` - *Motor para controlar balance inconsciente/consciente*
- `AdaptiveLearningMotor` (line 213) `class AdaptiveLearningMotor(MotorHomeostaticContext)` - *Motor para adaptar algoritmos de aprendizaje*
- `ModularActivationMotor` (line 241) `class ModularActivationMotor(MotorHomeostaticContext)` - *Motor para activar/desactivar módulos según contexto*
- `OmniBrainCoordinator` (line 291) `class OmniBrainCoordinator` - *Coordinador central que gestiona todos los motores homeostáticos*
- `OmniBrainModule` (line 452) `class OmniBrainModule` - *Módulo base para todos los componentes del Omni Brain*
- `PTSymmetricLayer` (line 467) `class PTSymmetricLayer(OmniBrainModule)` - *Capa con activación PT-simétrica regulada*
- `TopologicalLayer` (line 500) `class TopologicalLayer(OmniBrainModule)` - *Capa con conectividad topológica regulada*
- `DualMindModule` (line 557) `class DualMindModule(OmniBrainModule)` - *Módulo de procesamiento dual (inconsciente/consciente)*
- `ConsciousnessModule` (line 640) `class ConsciousnessModule(OmniBrainModule)` - *Módulo de métricas de conciencia y integración*
- `HomeostaticEngine` (line 701) `class HomeostaticEngine` - *Motor homeostasis reutilizable de Síntesis v8.2*
- `OmniBrain` (line 732) `class OmniBrain` - *El pokemon legendario que combina todas las ideas*

**Functions:**
- `train_omni_brain` (line 967) `def train_omni_brain(model, epochs, batch_size)` - *Pipeline de entrenamiento para el Omni Brain*
- `update` (line 47) `def update(self, measurement, dt)` - *Actualiza el estado del motor homeostático*
- `__init__` (line 66) `def __init__(self)`
- `regulate_parameters` (line 78) `def regulate_parameters(self, current_coherence, energy_level)` - *Regula parámetros para mantener PT-simetría*
- `__init__` (line 102) `def __init__(self)`
- `regulate_connectivity` (line 112) `def regulate_connectivity(self, current_connectivity, clustering)` - *Regula conectividad para mantener estructura óptima*
- `__init__` (line 129) `def __init__(self)`
- `regulate_energy` (line 139) `def regulate_energy(self, memory_usage, cpu_usage, temperature)` - *Regula parámetros para eficiencia energética*
- `__init__` (line 159) `def __init__(self)`
- `regulate_consciousness` (line 168) `def regulate_consciousness(self, phi_effective, integration_level)` - *Regula parámetros para control de conciencia*
- `__init__` (line 187) `def __init__(self)`
- `regulate_dual_systems` (line 197) `def regulate_dual_systems(self, unconscious_activity, conscious_activity)` - *Regula balance entre sistemas inconsciente y consciente*
- `__init__` (line 215) `def __init__(self)`
- `regulate_learning` (line 224) `def regulate_learning(self, loss_reduction_rate, gradient_norm)` - *Regula parámetros de aprendizaje*
- `__init__` (line 243) `def __init__(self)`
- `regulate_modules` (line 259) `def regulate_modules(self, task_complexity, resource_availability, performance)` - *Regula qué módulos están activos*
- `__init__` (line 294) `def __init__(self)`
- `_initialize_motors` (line 300) `def _initialize_motors(self)` - *Inicializa todos los motores homeostáticos*
- `sense_environment` (line 312) `def sense_environment(self)` - *Sensa el estado actual del entorno*
- `simulate_network_state` (line 326) `def simulate_network_state(self)` - *Simula el estado de red sin hacer forward pass (evita conflictos de autograd)*
- `measure_network_state` (line 340) `def measure_network_state(self, model, batch_data)` - *Mide el estado actual de la red*
- `coordinate_all_motors` (line 377) `def coordinate_all_motors(self, environment_state, network_state)` - *Coordina todos los motores homeostáticos*
- `__init__` (line 455) `def __init__(self, module_name, enabled)`
- `forward` (line 461) `def forward(self, x, params)`
- `update_performance` (line 464) `def update_performance(self, metrics)`
- `__init__` (line 470) `def __init__(self, in_features, out_features)`
- `forward` (line 477) `def forward(self, x, params)`
- `__init__` (line 503) `def __init__(self, in_features, out_features, sparsity_factor)`
- `_generate_topology_mask` (line 518) `def _generate_topology_mask(self)` - *Genera máscara topológica realista*
- `forward` (line 539) `def forward(self, x, params)`
- `__init__` (line 560) `def __init__(self, features)`
- `forward` (line 585) `def forward(self, x, params)`
- `__init__` (line 643) `def __init__(self, features)`
- `compute_phi_effective` (line 657) `def compute_phi_effective(self, x)` - *Cálculo simplificado de Φₑ (integración efectiva)*
- `forward` (line 677) `def forward(self, x, params)`
- `__init__` (line 704) `def __init__(self, target_performance)`
- `regulate_homeostasis` (line 709) `def regulate_homeostasis(self, observed_performance)` - *Regula parámetros para homeostasis*
- `__init__` (line 735) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `reset_internal_states` (line 769) `def reset_internal_states(self)` - *Resetea todos los estados internos para evitar problemas de gradientes*
- `prepare_for_inference` (line 803) `def prepare_for_inference(self)` - *Preparación específica para inferencia - reseteo completo*
- `initialize_context` (line 820) `def initialize_context(self)` - *Inicializa el contexto del Omni Brain*
- `forward` (line 834) `def forward(self, x)` - *Forward pass del Omni Brain con coordinación homeostática*
- `get_status_report` (line 932) `def get_status_report(self)` - *Genera reporte de estado del Omni Brain*

#### `live_cl.py`
**Path:** `live_cl.py`

**Classes:**
- `Config` (line 32) `class Config` - *Centralized configuration with ablation flags*
- `FastSlowLinear` (line 133) `class FastSlowLinear` - *Dual-system linear layer with fast (Hebbian) and slow (gradient) learning.
Fast learning is disabled in baseline mode but structure is maintained.*
- `DualSystemModule` (line 206) `class DualSystemModule` - *Dual-pathway processing with fast and slow streams.
Bypassed in baseline mode but ready for activation.*
- `IntegrationModule` (line 242) `class IntegrationModule` - *Neural integration module with adaptive gating.
Bypassed in baseline mode.*
- `OmniBrain` (line 278) `class OmniBrain` - *Unified Omni Brain architecture with configurable modules.
Optimized baseline with experimental features ready for activation.*

**Functions:**
- `setup_logging` (line 69) `def setup_logging()` - *Professional logging configuration*
- `set_seed` (line 84) `def set_seed(seed)` - *Ensure reproducibility*
- `compute_integration_index` (line 99) `def compute_integration_index(activity)` - *Compute neural integration using SVD (Singular Value Decomposition)
Returns value in [0, 1] representing degree of neural coordination*
- `get_data_loaders` (line 349) `def get_data_loaders(config)` - *Prepare CIFAR-10 data loaders with augmentation*
- `evaluate` (line 393) `def evaluate(model, loader, device)` - *Comprehensive model evaluation*
- `train` (line 430) `def train(config, silent)` - *Main training loop with comprehensive logging

Args:
    config: Configuration object
    silent: If True, reduce logging for ablation studies

Returns:
    Dictionary of training metrics history*
- `run_ablation_study` (line 570) `def run_ablation_study(quick_test)` - *Comprehensive ablation study across different configurations

Args:
    quick_test: If True, run 5 epochs per config; else full 50 epochs

Returns:
    Dictionary mapping configuration names to final test accuracies*
- `to_dict` (line 62) `def to_dict(self)`
- `__init__` (line 138) `def __init__(self, in_features, out_features, config)`
- `reset_fast_weights` (line 156) `def reset_fast_weights(self)` - *Reset fast weights (memory purge)*
- `update_fast_weights` (line 162) `def update_fast_weights(self, x, slow_out)` - *Hebbian learning update*
- `forward` (line 189) `def forward(self, x)`
- `get_fast_norm` (line 202) `def get_fast_norm(self)`
- `__init__` (line 211) `def __init__(self, dim, config)`
- `forward` (line 223) `def forward(self, x)`
- `__init__` (line 247) `def __init__(self, features, config)`
- `forward` (line 260) `def forward(self, x)`
- `__init__` (line 283) `def __init__(self, config)`
- `forward` (line 318) `def forward(self, x)`
- `reset_all_fast_weights` (line 325) `def reset_all_fast_weights(self)` - *Reset all fast weights in the network*
- `get_fast_norms` (line 331) `def get_fast_norms(self)` - *Collect fast weight norms for monitoring*
- `get_ablation_state` (line 336) `def get_ablation_state(self)` - *Return current ablation configuration*

#### `live_go.py`
**Path:** `live_go.py`

**Classes:**
- `Config` (line 30) `class Config`
- `FastSlowLinear` (line 93) `class FastSlowLinear` - *La neurona perfecta. Capaz de aprender rápido (Hebbiano) y lento (Gradiente).
En este PoC, la parte rápida duerme, pero la estructura es sólida.*
- `DualSystemModule` (line 143) `class DualSystemModule`
- `IntegrationModule` (line 167) `class IntegrationModule`
- `OmniBrainGenesis` (line 190) `class OmniBrainGenesis`

**Functions:**
- `compute_integration_index` (line 74) `def compute_integration_index(activity)` - *Calcula el orden dentro del caos neuronal mediante SVD.*
- `get_loaders` (line 230) `def get_loaders(config)`
- `breathe_life` (line 254) `def breathe_life(config)`
- `reset_seeds` (line 353) `def reset_seeds()` - *Reinicia el determinismo para que cada variante juegue en igualdad de condiciones.*
- `run_ablation_test` (line 360) `def run_ablation_test(full_epochs)` - *Ejecuta el Juicio Final: Compara las diferentes configuraciones del cerebro.*
- `train_engine_wrapper` (line 425) `def train_engine_wrapper(config)` - *Versión simplificada de breathe_life para el test que retorna la precisión.
Silencia logs intermedios para limpiar la salida.*
- `__init__` (line 98) `def __init__(self, in_features, out_features, config)`
- `reset_fast_weights` (line 115) `def reset_fast_weights(self)`
- `forward` (line 120) `def forward(self, x)`
- `get_fast_norm` (line 140) `def get_fast_norm(self)`
- `__init__` (line 144) `def __init__(self, dim, config)`
- `forward` (line 155) `def forward(self, x)`
- `__init__` (line 168) `def __init__(self, features, config)`
- `forward` (line 176) `def forward(self, x)`
- `__init__` (line 191) `def __init__(self, config)`
- `forward` (line 215) `def forward(self, x)`
- `reset_all_fast_weights` (line 222) `def reset_all_fast_weights(self)`

#### `live_ki.py`
**Path:** `live_ki.py`

**Classes:**
- `Config` (line 26) `class Config`
- `FastSlowLinear` (line 103) `class FastSlowLinear`
- `DualSystemModule` (line 177) `class DualSystemModule`
- `IntegrationModule` (line 211) `class IntegrationModule`
- `OmniBrainGenesis` (line 243) `class OmniBrainGenesis`

**Functions:**
- `compute_integration_index` (line 76) `def compute_integration_index(activity)` - *Mide el grado de orden en la actividad neural mediante SVD.
Retorna 0.0 si no hay suficiente información (caos puro).*
- `get_cifar10_loaders` (line 308) `def get_cifar10_loaders(config)`
- `evaluate_ritual` (line 334) `def evaluate_ritual(model, loader, device)`
- `train_genesis` (line 366) `def train_genesis(config)`
- `explore_realities` (line 499) `def explore_realities()` - *Explora múltiples configuraciones del universo neural*
- `__init__` (line 104) `def __init__(self, in_features, out_features, config)`
- `reset_fast_weights` (line 124) `def reset_fast_weights(self)` - *Ritual de purificación - resetea memoria a corto plazo*
- `update_fast_weights` (line 130) `def update_fast_weights(self, x, slow_out)` - *Ritual Hebbiano - solo ocurre si los dioses lo permiten*
- `forward` (line 157) `def forward(self, x)`
- `get_fast_norm` (line 171) `def get_fast_norm(self)`
- `__init__` (line 178) `def __init__(self, dim, config)`
- `forward` (line 192) `def forward(self, x)`
- `__init__` (line 212) `def __init__(self, features, config)`
- `forward` (line 224) `def forward(self, x)`
- `__init__` (line 244) `def __init__(self, config)`
- `forward` (line 279) `def forward(self, x)`
- `reset_all_fast_weights` (line 286) `def reset_all_fast_weights(self)` - *Ritual de purificación global*
- `get_fast_norms` (line 292) `def get_fast_norms(self)` - *Recopila energías de pesos rápidos*
- `get_ablation_state` (line 296) `def get_ablation_state(self)` - *Estado de creación*

#### `live_qw.py`
**Path:** `live_qw.py`

**Classes:**
- `Config` (line 23) `class Config`
- `FastSlowLinear` (line 64) `class FastSlowLinear` - *Módulo estabilizado – aunque no se usa en baseline, se mantiene para futura ablación.*
- `DualSystemModule` (line 122) `class DualSystemModule`
- `IntegrationModule` (line 146) `class IntegrationModule`
- `OmniBrainFastSlow` (line 191) `class OmniBrainFastSlow`

**Functions:**
- `compute_integration_index` (line 171) `def compute_integration_index(activity)`
- `get_cifar10_loaders` (line 244) `def get_cifar10_loaders(config)`
- `evaluate_full` (line 265) `def evaluate_full(model, loader, device)`
- `train` (line 286) `def train(config)`
- `__init__` (line 66) `def __init__(self, in_features, out_features, config)`
- `reset_fast_weights` (line 83) `def reset_fast_weights(self)`
- `update_fast_weights` (line 88) `def update_fast_weights(self, x, slow_out)`
- `forward` (line 106) `def forward(self, x)`
- `get_fast_norm` (line 118) `def get_fast_norm(self)`
- `__init__` (line 123) `def __init__(self, dim, config)`
- `forward` (line 133) `def forward(self, x)`
- `__init__` (line 147) `def __init__(self, features, config)`
- `forward` (line 158) `def forward(self, x)`
- `__init__` (line 192) `def __init__(self, config)`
- `forward` (line 217) `def forward(self, x)`
- `reset_all_fast_weights` (line 224) `def reset_all_fast_weights(self)`
- `get_fast_norms` (line 229) `def get_fast_norms(self)`
- `get_ablation_state` (line 232) `def get_ablation_state(self)`

#### `lol.py`
**Path:** `lol.py`

**Classes:**
- `LiquidNeuron` (line 107) `class LiquidNeuron`
- `RightHemisphere` (line 180) `class RightHemisphere`
- `LeftHemisphere` (line 201) `class LeftHemisphere`
- `CorpusCallosum` (line 298) `class CorpusCallosum`
- `NeuroLogosBicameral` (line 313) `class NeuroLogosBicameral`
- `NeuralDiagnostics` (line 334) `class NeuralDiagnostics`
- `Flickr8kDataset` (line 404) `class Flickr8kDataset(Dataset)`
- `LifeCycle` (line 466) `class LifeCycle`

**Functions:**
- `setup_flickr8k` (line 34) `def setup_flickr8k(data_dir)` - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 443) `def build_vocab_flickr(captions_file, vocab_size)`
- `train_bicameral` (line 481) `def train_bicameral()`
- `__init__` (line 108) `def __init__(self, in_dim, out_dim)`
- `forward` (line 125) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 154) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 181) `def __init__(self, output_dim)`
- `forward` (line 192) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 202) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 224) `def forward(self, visual_context, captions, max_len, return_gate)`
- `_get_init_state` (line 278) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 283) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 299) `def __init__(self, dim)`
- `forward` (line 307) `def forward(self, right_features)`
- `__init__` (line 314) `def __init__(self, vocab_size)`
- `forward` (line 320) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- `__init__` (line 335) `def __init__(self)`
- `measure_callosal_flow` (line 346) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 353) `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `update` (line 357) `def update(self)`
- `get_recent_avg` (line 362) `def get_recent_avg(self, key, n)`
- `report` (line 367) `def report(self, epoch)`
- `__init__` (line 405) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 423) `def __len__(self)`
- `__getitem__` (line 426) `def __getitem__(self, idx)`
- `__init__` (line 467) `def __init__(self, total_epochs)`
- `get_plasticity` (line 470) `def get_plasticity(self, epoch)`

#### `main.py`
**Path:** `main.py`

**Classes:**
- `RESMAConstants` (line 32) `class RESMAConstants` - *Constantes físicas y parámetros de la teoría RESMA*
- `PhysicalValidator` (line 58) `class PhysicalValidator` - *Validación de rangos físicos para todas las constantes*
- `QuantumLeaf` (line 90) `class QuantumLeaf` - *Hoja L_i de la Resma como estado KMS mean-field.
No almacena matrices densas (Pilar 4).*
- `RESMAUniverse` (line 144) `class RESMAUniverse` - *Multiverso como foliación medible sin matrices densas.
Memoria: O(N_leaves) en lugar de O(N_leaves × dim²)*
- `BranchingOperator` (line 220) `class BranchingOperator` - *Operador Ĥ que abre la Resma cuando β_i es no trivial.
Implementación local sin matrices globales (Pilar 4).*
- `EmunaOperator` (line 270) `class EmunaOperator` - *Operador P̂_E: proyección teleológica no lineal.
Implementación con muestreo Monte Carlo (Pilar 4).*
- `LindbladFractalDynamics` (line 350) `class LindbladFractalDynamics` - *SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ] + ξ(t)
Integración por Euler-Maruyama (Pilar 4: estabilidad numérica).*
- `MyelinCavity` (line 428) `class MyelinCavity` - *Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.
Implementación 1D mean-field para Colab (Pilar 4).*
- `NeuralNetworkRESMA` (line 481) `class NeuralNetworkRESMA` - *Conectoma humano dirigido con homología persistente.
Implementación sparse para escalado (Pilar 4).*
- `FreedomInvariant` (line 582) `class FreedomInvariant` - *L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.
Cálculo topológico sin densidades matriciales (Pilar 4).*
- `NullModels` (line 628) `class NullModels` - *Modelos nulos para cálculo de Factor de Bayes.
Basados en teorías establecidas sin postulados RESMA.*
- `ExperimentalPredictions` (line 686) `class ExperimentalPredictions` - *Predicciones falsables contra modelos nulos teóricos.
Factor de Bayes calculado con AIC (aproximación).*

**Functions:**
- `simulate_resma_multiverse` (line 764) `def simulate_resma_multiverse(n_leaves, n_nodes, seed)` - *Pipeline completo RESMA 3.0 con verificaciones de integridad.
Diseñado para ejecución en Google Colab (Pilar 4).*
- `validate_dimension` (line 62) `def validate_dimension(alpha)` - *α ∈ (0,1) por definición de dimensión fractal*
- `validate_pt_symmetry` (line 68) `def validate_pt_symmetry(kappa, Omega, chi)` - *Verificar κ/Ω < χ/Ω < 1 para PT-simetría*
- `validate_connectome_size` (line 79) `def validate_connectome_size(n_nodes)` - *Límite inferior para conectoma biológico*
- `__post_init__` (line 100) `def __post_init__(self)` - *Validaciones post-construcción (Pilar 3)*
- `spectral_density` (line 106) `def spectral_density(self, omega)` - *Densidad espectral continua ρ(ω) para álgebra tipo III₁.
Evidencia: SYK tiene espectro continuo sin gaps (Maldacena, JHEP 2016).*
- `modular_entropy` (line 114) `def modular_entropy(self)` - *Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)*
- `bures_distance` (line 121) `def bures_distance(self, other)`
- `_spectral_moments` (line 132) `def _spectral_moments(self, n)` - *Momentos espectrales Tr(ρ^k) para k=1..n*
- `__init__` (line 150) `def __init__(self, n_leaves, seed)` - *Args:
    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)
    seed: Reproducibilidad (Pilar 4)*
- `_initialize_leaves` (line 168) `def _initialize_leaves(self)` - *Genera hojas con gaps espectrales distribuidos*
- `_generate_gibbs_measure` (line 181) `def _generate_gibbs_measure(self)` - *Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))*
- `_construct_global_state` (line 203) `def _construct_global_state(self)` - *Estado global: mapa de pesos por hoja (no matriz)*
- `__init__` (line 226) `def __init__(self, leaf, threshold)`
- `_construct_cptp_map` (line 231) `def _construct_cptp_map(self)` - *Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)*
- `_local_jump_operator` (line 240) `def _local_jump_operator(self, power)` - *K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.
Aproximación mean-field: operadores de Pauli escalados por gap.*
- `apply_branching` (line 255) `def apply_branching(self, state_vector)` - *Aplicar canal CPTP a vector de estado local (dim=2)*
- `__init__` (line 276) `def __init__(self, universe, n_samples)`
- `_construct_hardy_state` (line 282) `def _construct_hardy_state(self)` - *E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior*
- `_szego_projector` (line 286) `def _szego_projector(self)` - *Proyector P_E en base de Fourier positiva (dim reducida)*
- `_evaluation_functional` (line 294) `def _evaluation_functional(self, state_weights)` - *Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))*
- `project` (line 310) `def project(self, state_vector)` - *P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO*
- `__init__` (line 356) `def __init__(self, universe, emuna)`
- `_effective_hamiltonian` (line 362) `def _effective_hamiltonian(self)` - *H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)*
- `_modular_dissipator` (line 372) `def _modular_dissipator(self, state)` - *L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}*
- `_nonlinear_term` (line 382) `def _nonlinear_term(self, state)` - *G[ρ, log ρ_∞] = g[ρ, log ρ_∞]*
- `evolve` (line 389) `def evolve(self, rho0, t_span, n_steps)` - *Integración SDE con Euler-Maruyama.
Returns: trayectoria [n_steps, 2, 2]*
- `__post_init__` (line 437) `def __post_init__(self)`
- `_free_hamiltonian` (line 442) `def _free_hamiltonian(self)` - *H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 448) `def _loss_potential(self)` - *V_loss ∝ (r⊥/a₀)^{2α} con α=0.7*
- `_pt_symmetry_condition` (line 456) `def _pt_symmetry_condition(self)` - *Verificar κ/Ω < χ/Ω < 1*
- `coherence_quantum` (line 462) `def coherence_quantum(self)` - *Discordia cuántica aproximada (ejemplo: estado separable → 0)*
- `__init__` (line 487) `def __init__(self, n_nodes, seed)`
- `_generate_fractal_graph` (line 500) `def _generate_fractal_graph(self)` - *Grafo dirigido con distribución de grados power-law.
Fuente: Human Connectome Project (Pilar 1).*
- `_spectral_dimension` (line 510) `def _spectral_dimension(self)` - *d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)*
- `_topological_ramsey` (line 527) `def _topological_ramsey(self)` - *R_Q(G) = min{n | β_{n-1}(G) > 0}
Requiere ripser (instalable en Colab: !pip install ripser)*
- `_graph_to_distance_matrix` (line 548) `def _graph_to_distance_matrix(self)` - *Matriz de distancias shortest-path (sparse CSR)*
- `critical_percolation_time` (line 562) `def critical_percolation_time(self)` - *t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25
Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)*
- `is_coherent_subgraph` (line 574) `def is_coherent_subgraph(self, subgraph_nodes)` - *Verificar coherencia: subgrafo > 70% del total*
- `__init__` (line 588) `def __init__(self, network, universe)`
- `compute_entropy_gap` (line 592) `def compute_entropy_gap(self)` - *Δ_S* = ε_c en punto excepcional*
- `compute_pontryagin_number` (line 596) `def compute_pontryagin_number(self)` - *S_top[G] = χ(G)/|V| (número de Euler normalizado)*
- `compute_freedom` (line 607) `def compute_freedom(self)` - *L[G] = Δ_S* / S_top[G]*
- `is_gauge_invariant` (line 618) `def is_gauge_invariant(self)` - *|L[G] - 1| < 0.05 en estado crítico*
- `ising_quantum` (line 635) `def ising_quantum(network)` - *Modelo de Ising cuántico transversal en red fractal.
Predice t_c sin SYK₈ ni E₈.*
- `syk4` (line 654) `def syk4(network)` - *SYK₄ estándar (sin R-simetría Spin(7)).
Predice α sin postulado E₈.*
- `random_network` (line 670) `def random_network(network)` - *Red aleatoria Erdős-Rényi sin percolación cuántica.*
- `__init__` (line 692) `def __init__(self, resma, myelin, network)`
- `predict_all` (line 699) `def predict_all(self)` - *Predicciones RESMA 3.0*
- `_predict_diffraction_peak` (line 710) `def _predict_diffraction_peak(self)` - *q₀ = 2π/L_E8 (sin ajuste)*
- `compute_bayes_factor` (line 715) `def compute_bayes_factor(self)` - *BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)
k = número de parámetros RESMA = 5 (α, β, γ, L_E8, g_coupling)*

#### `main2.py`
**Path:** `main2.py`

**Classes:**
- `RESMAConstants` (line 33) `class RESMAConstants` - *Constantes físicas y parámetros de la teoría RESMA*
- `PhysicalValidator` (line 59) `class PhysicalValidator` - *Validación de rangos físicos para todas las constantes*
- `QuantumLeaf` (line 91) `class QuantumLeaf` - *Hoja L_i de la Resma como estado KMS mean-field.
No almacena matrices densas (Pilar 4).*
- `RESMAUniverse` (line 144) `class RESMAUniverse` - *Multiverso como foliación medible sin matrices densas.
Memoria: O(N_leaves) en lugar de O(N_leaves × dim²)*
- `BranchingOperator` (line 219) `class BranchingOperator` - *Operador Ĥ que abre la Resma cuando β_i es no trivial.
Implementación local sin matrices globales (Pilar 4).*
- `EmunaOperator` (line 269) `class EmunaOperator` - *Operador P̂_E: proyección teleológica no lineal.
Implementación con muestreo Monte Carlo (Pilar 4).*
- `LindbladFractalDynamics` (line 349) `class LindbladFractalDynamics` - *SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ] + ξ(t)
Integración por Euler-Maruyama (Pilar 4: estabilidad numérica).*
- `MyelinCavity` (line 427) `class MyelinCavity` - *Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.
Implementación 1D mean-field para Colab (Pilar 4).*
- `NeuralNetworkRESMA` (line 480) `class NeuralNetworkRESMA` - *Conectoma humano dirigido con homología persistente.
Implementación sparse para escalado (Pilar 4).*
- `FreedomInvariant` (line 581) `class FreedomInvariant` - *L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.
Cálculo topológico sin densidades matriciales (Pilar 4).*
- `NullModels` (line 627) `class NullModels` - *Modelos nulos para cálculo de Factor de Bayes.
Basados en teorías establecidas sin postulados RESMA.*
- `ExperimentalPredictions` (line 685) `class ExperimentalPredictions` - *Predicciones falsables contra modelos nulos teóricos.
Factor de Bayes calculado con AIC (aproximación).*

**Functions:**
- `simulate_resma_multiverse` (line 763) `def simulate_resma_multiverse(n_leaves, n_nodes, seed)` - *Pipeline completo RESMA 3.0 con verificaciones de integridad.
Diseñado para ejecución en Google Colab (Pilar 4).*
- `validate_dimension` (line 63) `def validate_dimension(alpha)` - *α ∈ (0,1) por definición de dimensión fractal*
- `validate_pt_symmetry` (line 69) `def validate_pt_symmetry(kappa, Omega, chi)` - *Verificar κ/Ω < χ/Ω < 1 para PT-simetría*
- `validate_connectome_size` (line 80) `def validate_connectome_size(n_nodes)` - *Límite inferior para conectoma biológico*
- `__post_init__` (line 101) `def __post_init__(self)` - *Validaciones post-construcción (Pilar 3)*
- `spectral_density` (line 106) `def spectral_density(self, omega)` - *Densidad espectral continua ρ(ω) para álgebra tipo III₁.
Evidencia: SYK tiene espectro continuo sin gaps (Maldacena, JHEP 2016).*
- `modular_entropy` (line 114) `def modular_entropy(self)` - *Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)*
- `bures_distance` (line 121) `def bures_distance(self, other)`
- `_spectral_moments` (line 132) `def _spectral_moments(self, n)` - *Momentos espectrales Tr(ρ^k) para k=1..n*
- `__init__` (line 150) `def __init__(self, n_leaves, seed)` - *Args:
    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)
    seed: Reproducibilidad (Pilar 4)*
- `_initialize_leaves` (line 167) `def _initialize_leaves(self)` - *Genera hojas con gaps espectrales distribuidos*
- `_generate_gibbs_measure` (line 180) `def _generate_gibbs_measure(self)` - *Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))*
- `_construct_global_state` (line 202) `def _construct_global_state(self)` - *Estado global: mapa de pesos por hoja (no matriz)*
- `__init__` (line 225) `def __init__(self, leaf, threshold)`
- `_construct_cptp_map` (line 230) `def _construct_cptp_map(self)` - *Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)*
- `_local_jump_operator` (line 239) `def _local_jump_operator(self, power)` - *K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.
Aproximación mean-field: operadores de Pauli escalados por gap.*
- `apply_branching` (line 254) `def apply_branching(self, state_vector)` - *Aplicar canal CPTP a vector de estado local (dim=2)*
- `__init__` (line 275) `def __init__(self, universe, n_samples)`
- `_construct_hardy_state` (line 281) `def _construct_hardy_state(self)` - *E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior*
- `_szego_projector` (line 285) `def _szego_projector(self)` - *Proyector P_E en base de Fourier positiva (dim reducida)*
- `_evaluation_functional` (line 293) `def _evaluation_functional(self, state_weights)` - *Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))*
- `project` (line 309) `def project(self, state_vector)` - *P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO*
- `__init__` (line 355) `def __init__(self, universe, emuna)`
- `_effective_hamiltonian` (line 361) `def _effective_hamiltonian(self)` - *H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)*
- `_modular_dissipator` (line 371) `def _modular_dissipator(self, state)` - *L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}*
- `_nonlinear_term` (line 381) `def _nonlinear_term(self, state)` - *G[ρ, log ρ_∞] = g[ρ, log ρ_∞]*
- `evolve` (line 388) `def evolve(self, rho0, t_span, n_steps)` - *Integración SDE con Euler-Maruyama.
Returns: trayectoria [n_steps, 2, 2]*
- `__post_init__` (line 436) `def __post_init__(self)`
- `_free_hamiltonian` (line 441) `def _free_hamiltonian(self)` - *H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 447) `def _loss_potential(self)` - *V_loss ∝ (r⊥/a₀)^{2α} con α=0.7*
- `_pt_symmetry_condition` (line 455) `def _pt_symmetry_condition(self)` - *Verificar κ/Ω < χ/Ω < 1*
- `coherence_quantum` (line 461) `def coherence_quantum(self)` - *Discordia cuántica aproximada (ejemplo: estado separable → 0)*
- `__init__` (line 486) `def __init__(self, n_nodes, seed)`
- `_generate_fractal_graph` (line 499) `def _generate_fractal_graph(self)` - *Grafo dirigido con distribución de grados power-law.
Fuente: Human Connectome Project (Pilar 1).*
- `_spectral_dimension` (line 509) `def _spectral_dimension(self)` - *d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)*
- `_topological_ramsey` (line 526) `def _topological_ramsey(self)` - *R_Q(G) = min{n | β_{n-1}(G) > 0}
Requiere ripser (instalable en Colab: !pip install ripser)*
- `_graph_to_distance_matrix` (line 547) `def _graph_to_distance_matrix(self)` - *Matriz de distancias shortest-path (sparse CSR)*
- `critical_percolation_time` (line 561) `def critical_percolation_time(self)` - *t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25
Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)*
- `is_coherent_subgraph` (line 573) `def is_coherent_subgraph(self, subgraph_nodes)` - *Verificar coherencia: subgrafo > 70% del total*
- `__init__` (line 587) `def __init__(self, network, universe)`
- `compute_entropy_gap` (line 591) `def compute_entropy_gap(self)` - *Δ_S* = ε_c en punto excepcional*
- `compute_pontryagin_number` (line 595) `def compute_pontryagin_number(self)` - *S_top[G] = χ(G)/|V| (número de Euler normalizado)*
- `compute_freedom` (line 606) `def compute_freedom(self)` - *L[G] = Δ_S* / S_top[G]*
- `is_gauge_invariant` (line 617) `def is_gauge_invariant(self)` - *|L[G] - 1| < 0.05 en estado crítico*
- `ising_quantum` (line 634) `def ising_quantum(network)` - *Modelo de Ising cuántico transversal en red fractal.
Predice t_c sin SYK₈ ni E₈.*
- `syk4` (line 653) `def syk4(network)` - *SYK₄ estándar (sin R-simetría Spin(7)).
Predice α sin postulado E₈.*
- `random_network` (line 669) `def random_network(network)` - *Red aleatoria Erdős-Rényi sin percolación cuántica.*
- `__init__` (line 691) `def __init__(self, resma, myelin, network)`
- `predict_all` (line 698) `def predict_all(self)` - *Predicciones RESMA 3.0*
- `_predict_diffraction_peak` (line 708) `def _predict_diffraction_peak(self)` - *q₀ = 2π/L_E8 (sin ajuste)*
- `compute_bayes_factor` (line 713) `def compute_bayes_factor(self)` - *BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)
k = número de parámetros RESMA = 5 (α, β, γ, L_E8, g_coupling)*

#### `main3.py`
**Path:** `main3.py`

**Classes:**
- `RC` (line 33) `class RC`
- `Validator` (line 50) `class Validator`
- `QuantumLeaf` (line 68) `class QuantumLeaf`
- `Universe` (line 101) `class Universe`
- `Network` (line 129) `class Network`
- `MyelinCavity` (line 171) `class MyelinCavity`
- `Bayes` (line 205) `class Bayes`

**Functions:**
- `simulate` (line 233) `def simulate(n_leaves, n_nodes, seed)`
- `dim` (line 52) `def dim(a)`
- `pt` (line 56) `def pt(k, o, c)`
- `size` (line 59) `def size(n)`
- `__post_init__` (line 74) `def __post_init__(self)`
- `spectral_density` (line 78) `def spectral_density(self, w)`
- `modular_entropy` (line 81) `def modular_entropy(self)`
- `bures_distance` (line 87) `def bures_distance(self, other)`
- `__init__` (line 102) `def __init__(self, n_leaves, seed)`
- `_gibbs` (line 110) `def _gibbs(self)`
- `_global` (line 119) `def _global(self)`
- `__init__` (line 130) `def __init__(self, n_nodes, seed)`
- `_spectral_dim` (line 139) `def _spectral_dim(self, k)`
- `_ramsey` (line 149) `def _ramsey(self)`
- `t_c` (line 163) `def t_c(self)`
- `__init__` (line 172) `def __init__(self, n_modes)`
- `_free_hamiltonian` (line 178) `def _free_hamiltonian(self)`
- `_loss_potential` (line 183) `def _loss_potential(self)`
- `_pt_symmetry_condition` (line 189) `def _pt_symmetry_condition(self)`
- `coherence_quantum` (line 192) `def coherence_quantum(self)`
- `__init__` (line 206) `def __init__(self, pred_resma, nulls)`
- `log_lik` (line 210) `def log_lik(self, model_pred)`
- `bf` (line 217) `def bf(self)`

#### `main4.1.py`
**Path:** `main4.1.py`

**Classes:**
- `RC` (line 29) `class RC`
- `Validator` (line 61) `class Validator`
- `QuantumLeaf` (line 82) `class QuantumLeaf`
- `Universe` (line 123) `class Universe`
- `Network` (line 152) `class Network`
- `MyelinCavity` (line 229) `class MyelinCavity`
- `Bayes` (line 268) `class Bayes`

**Functions:**
- `simulate` (line 307) `def simulate(n_leaves, n_nodes, seed)`
- `verify_pt_condition` (line 50) `def verify_pt_condition(cls)` - *Verifica que kappa < chi*Omega para simetría PT*
- `dim` (line 63) `def dim(a)`
- `pt` (line 68) `def pt(k, o, c)` - *Condición PT: kappa < chi*Omega*
- `size` (line 73) `def size(n)`
- `__post_init__` (line 88) `def __post_init__(self)`
- `spectral_density` (line 92) `def spectral_density(self, w)`
- `modular_entropy` (line 95) `def modular_entropy(self)`
- `bures_distance` (line 104) `def bures_distance(self, other)`
- `__init__` (line 124) `def __init__(self, n_leaves, seed)`
- `_gibbs` (line 133) `def _gibbs(self)`
- `_global` (line 142) `def _global(self)`
- `__init__` (line 153) `def __init__(self, n_nodes, seed)`
- `_spectral_dim` (line 163) `def _spectral_dim(self, k, n_fit)` - *Dimensión espectral corregida*
- `_ramsey` (line 199) `def _ramsey(self)` - *Número de Ramsey topológico*
- `t_c` (line 218) `def t_c(self)` - *Tiempo crítico de percolación*
- `__init__` (line 230) `def __init__(self, n_modes)`
- `_free_hamiltonian` (line 237) `def _free_hamiltonian(self)`
- `_loss_potential` (line 242) `def _loss_potential(self)`
- `_pt_symmetry_condition` (line 248) `def _pt_symmetry_condition(self)`
- `coherence_quantum` (line 251) `def coherence_quantum(self)`
- `__init__` (line 269) `def __init__(self, pred_resma, nulls)`
- `log_lik` (line 273) `def log_lik(self, model_pred)` - *Verosimilitud con escalas físicas realistas*
- `ln_bf` (line 288) `def ln_bf(self)` - *Factor de Bayes con penalización de complejidad*

#### `main4.py.py`
**Path:** `main4.py.py`

**Classes:**
- `RC` (line 33) `class RC`
- `Validator` (line 50) `class Validator`
- `QuantumLeaf` (line 69) `class QuantumLeaf`
- `Universe` (line 102) `class Universe`
- `Network` (line 130) `class Network`
- `MyelinCavity` (line 198) `class MyelinCavity`
- `Bayes` (line 232) `class Bayes`

**Functions:**
- `simulate` (line 260) `def simulate(n_leaves, n_nodes, seed)`
- `dim` (line 52) `def dim(a)`
- `pt` (line 56) `def pt(k, o, c)`
- `size` (line 60) `def size(n)`
- `__post_init__` (line 75) `def __post_init__(self)`
- `spectral_density` (line 79) `def spectral_density(self, w)`
- `modular_entropy` (line 82) `def modular_entropy(self)`
- `bures_distance` (line 88) `def bures_distance(self, other)`
- `__init__` (line 103) `def __init__(self, n_leaves, seed)`
- `_gibbs` (line 111) `def _gibbs(self)`
- `_global` (line 120) `def _global(self)`
- `__init__` (line 131) `def __init__(self, n_nodes, seed)`
- `_spectral_dim` (line 140) `def _spectral_dim(self, k, n_fit)`
- `_ramsey` (line 176) `def _ramsey(self)`
- `t_c` (line 190) `def t_c(self)`
- `__init__` (line 199) `def __init__(self, n_modes)`
- `_free_hamiltonian` (line 205) `def _free_hamiltonian(self)`
- `_loss_potential` (line 210) `def _loss_potential(self)`
- `_pt_symmetry_condition` (line 216) `def _pt_symmetry_condition(self)`
- `coherence_quantum` (line 219) `def coherence_quantum(self)`
- `__init__` (line 233) `def __init__(self, pred_resma, nulls)`
- `log_lik` (line 237) `def log_lik(self, model_pred)`
- `ln_bf` (line 244) `def ln_bf(self)`

#### `main5.py`
**Path:** `main5.py`

**Classes:**
- `RESMAConstants` (line 37) `class RESMAConstants` - *Constantes físicas y parámetros de la teoría RESMA 4.0*
- `PhysicalValidator` (line 70) `class PhysicalValidator` - *Validación de rangos físicos para todas las constantes RESMA 4.0*
- `QuantumLeaf` (line 116) `class QuantumLeaf` - *Hoja L_i de la Resma como estado KMS mean-field con espacio de Hilbert standard.
Implementación RESMA 4.0 con regularización Haagerup.*
- `RESMAUniverse` (line 179) `class RESMAUniverse` - *Multiverso como foliación medible sin matrices densas, con espacio de Hilbert standard.
Memoria: O(N_leaves) con regularización de transiciones.*
- `BranchingOperator` (line 260) `class BranchingOperator` - *Operador Ĥ que abre la Resma cuando β_i es no trivial.
Implementación local con operadores de salto SYK₈ (Pilar 4).*
- `EmunaOperator` (line 318) `class EmunaOperator` - *Operador P̂_E: proyección teleológica no lineal.
Implementación con muestreo Monte Carlo y espacio de Hardy H²(ℂ⁺) (Pilar 4).*
- `LindbladFractalDynamics` (line 406) `class LindbladFractalDynamics` - *SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ_∞] + ξ(t)
Integración por Euler-Maruyama con control de precisión (Pilar 4).*
- `MyelinCavity` (line 539) `class MyelinCavity` - *Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.
Implementación 1D mean-field con R-simetría Spin(7) (Pilar 4).*
- `NeuralNetworkRESMA` (line 604) `class NeuralNetworkRESMA` - *Conectoma humano NO DIRIGIDO con homología persistente.
Implementación sparse para escalado con conversión a grafo no dirigido (Pilar 4).*
- `FreedomInvariant` (line 762) `class FreedomInvariant` - *L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.
Cálculo topológico sin densidades matriciales (Pilar 4).*
- `NullModels` (line 816) `class NullModels` - *Modelos nulos para cálculo de Factor de Bayes.
Basados en teorías establecidas sin postulados holográficos de RESMA.*
- `ExperimentalPredictions` (line 877) `class ExperimentalPredictions` - *Predicciones falsables contra modelos nulos teóricos.
Factor de Bayes calculado con AIC y transformaciones logarítmicas (FIX).*
- `EmpiricalValidationProtocol` (line 975) `class EmpiricalValidationProtocol` - *Protocolo experimental para falsación controlada de RESMA 4.0.
Define setups experimentales y criterios de éxito.*

**Functions:**
- `simulate_resma_multiverse` (line 1052) `def simulate_resma_multiverse(n_leaves, n_nodes, seed, validate_empirical)` - *Pipeline completo RESMA 4.0 con verificaciones de integridad y protocolo de validación.
Diseñado para ejecución en Google Colab (Pilar 4).*
- `validate_dimension` (line 74) `def validate_dimension(alpha, tolerance)` - *α ∈ (0,1) por definición de dimensión fractal, con tolerancia experimental*
- `validate_pt_symmetry` (line 84) `def validate_pt_symmetry(kappa, Omega, chi)` - *Verificar κ/Ω < χ/Ω < 1 para PT-simetría (corregido con factor de seguridad)*
- `validate_connectome_size` (line 95) `def validate_connectome_size(n_nodes)` - *Límite inferior para conectoma biológico realista*
- `validate_spectral_dimension` (line 101) `def validate_spectral_dimension(dim)` - *Validar rango físico para dimensión espectral*
- `validate_percolation_time` (line 106) `def validate_percolation_time(t_c, expected, tolerance)` - *Validar tiempo de percolación contra predicción empírica*
- `__post_init__` (line 127) `def __post_init__(self)` - *Validaciones post-construcción (Pilar 3)*
- `spectral_density` (line 133) `def spectral_density(self, omega)` - *Densidad espectral continua ρ(ω) para álgebra tipo III₁ con regularización UV.
Evidencia: SYK₈ con Spin(7) tiene espectro continuo con gap infrarrojo.*
- `modular_entropy` (line 143) `def modular_entropy(self)` - *Entropía modular S = ∫ ρ(ω)logρ(ω) dω con regularización*
- `bures_distance` (line 151) `def bures_distance(self, other)`
- `_spectral_moments` (line 163) `def _spectral_moments(self, n)` - *Momentos espectrales Tr(ρ^k) para k=1..n con regularización*
- `haagerup_weight` (line 170) `def haagerup_weight(self)` - *Peso de Haagerup para regularización del operador modular*
- `__init__` (line 185) `def __init__(self, n_leaves, seed)` - *Args:
    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)
    seed: Reproducibilidad (Pilar 4)*
- `_initialize_leaves` (line 203) `def _initialize_leaves(self)` - *Genera hojas con gaps espectrales distribuidos exponencialmente*
- `_generate_gibbs_measure` (line 217) `def _generate_gibbs_measure(self)` - *Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j)) con normalización robusta*
- `_construct_global_state` (line 239) `def _construct_global_state(self)` - *Estado global: mapa de pesos por hoja (no matriz) con regularización*
- `compute_gibbs_free_energy` (line 251) `def compute_gibbs_free_energy(self)` - *Energía libre de Gibbs para validación termodinámica*
- `__init__` (line 266) `def __init__(self, leaf, threshold)`
- `_compute_holonomy` (line 272) `def _compute_holonomy(self)` - *Defecto de holonomía como variación del gap espectral*
- `_construct_cptp_map` (line 276) `def _construct_cptp_map(self)` - *Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)*
- `_local_jump_operator` (line 284) `def _local_jump_operator(self, power)` - *K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.
Aproximación mean-field: operadores de Pauli escalados por gap SYK₈.*
- `apply_branching` (line 300) `def apply_branching(self, state_vector)` - *Aplicar canal CPTP a vector de estado local (dim=2) con normalización*
- `__init__` (line 324) `def __init__(self, universe, n_samples)`
- `_construct_hardy_state` (line 331) `def _construct_hardy_state(self)` - *E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior*
- `_szego_projector` (line 335) `def _szego_projector(self)` - *Proyector P_E en base de Fourier positiva (dim reducida)*
- `_evaluation_functional` (line 345) `def _evaluation_functional(self, state_weights)` - *Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))*
- `project` (line 361) `def project(self, state_vector)` - *P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO*
- `compute_teleological_overlap` (line 396) `def compute_teleological_overlap(self)` - *Calcular overlap teleológico con estado objetivo*
- `__init__` (line 412) `def __init__(self, universe, emuna)`
- `_effective_hamiltonian` (line 419) `def _effective_hamiltonian(self)` - *H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva con gaps SYK₈)*
- `_modular_dissipator` (line 432) `def _modular_dissipator(self, state)` - *L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ} con regularización*
- `_nonlinear_term` (line 444) `def _nonlinear_term(self, state)` - *G[ρ, log ρ_∞] = g[ρ, log ρ_∞] con regularización del logaritmo*
- `_stochastic_term` (line 452) `def _stochastic_term(self, dt)` - *Término estocástico ξ(t) con correlaciones cuánticas*
- `evolve` (line 459) `def evolve(self, rho0, t_span, n_steps)` - *Integración SDE con Euler-Maruyama y control de paso adaptativo.
Returns: trayectoria [n_steps, 2, 2]*
- `_normalize_density_matrix` (line 500) `def _normalize_density_matrix(self, state)` - *Normalizar matriz densidad y forzar hermiticidad*
- `_is_physical_state` (line 509) `def _is_physical_state(self, state)` - *Verificar si el estado es físico (hermitiano, traza=1, positivo)*
- `_correct_non_physical_state` (line 523) `def _correct_non_physical_state(self, state)` - *Corregir estado no físico proyectando en el cono de estados válidos*
- `__post_init__` (line 548) `def __post_init__(self)`
- `_free_hamiltonian` (line 555) `def _free_hamiltonian(self)` - *H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 561) `def _loss_potential(self)` - *V_loss ∝ (r⊥/a₀)^{2α} con α=0.702 (SYK₈)*
- `_compute_scalar_mass` (line 569) `def _compute_scalar_mass(self)` - *Campo escalar masivo para estabilización de Spin(7)*
- `_pt_symmetry_condition` (line 573) `def _pt_symmetry_condition(self)` - *Verificar κ/Ω < χ/Ω < 1 con parámetros corregidos*
- `coherence_quantum` (line 579) `def coherence_quantum(self)` - *Discordia cuántica aproximada con corrección PT*
- `__init__` (line 610) `def __init__(self, n_nodes, seed)`
- `_generate_fractal_graph` (line 625) `def _generate_fractal_graph(self)` - *Generar grafo dirigido y convertir a NO DIRIGIDO para análisis espectral.
SOLUCIÓN RESMA 4.0: Conversión explícita con to_undirected().*
- `_spectral_dimension` (line 649) `def _spectral_dimension(self)` - *d_s = -2 lim_{λ→0⁺} log N(λ)/log λ usando normalized_laplacian_spectrum.
SOLUCIÓN RESMA 4.0: Uso de función especializada de NetworkX.*
- `_topological_ramsey` (line 680) `def _topological_ramsey(self)` - *R_Q(G) = min{n | β_{n-1}(G) > 0}
Requiere ripser (instalable en Colab: !pip install ripser)*
- `_compute_betti_numbers` (line 706) `def _compute_betti_numbers(self)` - *Calcular números de Betti para análisis topológico*
- `_graph_to_distance_matrix` (line 722) `def _graph_to_distance_matrix(self)` - *Matriz de distancias shortest-path (sparse CSR) para homología*
- `critical_percolation_time` (line 735) `def critical_percolation_time(self)` - *t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25
Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)*
- `is_coherent_subgraph` (line 747) `def is_coherent_subgraph(self, subgraph_nodes)` - *Verificar coherencia: subgrafo > 70% del total*
- `compute_network_entropy` (line 751) `def compute_network_entropy(self)` - *Entropía de la red basada en distribución de grados*
- `__init__` (line 768) `def __init__(self, network, universe)`
- `compute_entropy_gap` (line 772) `def compute_entropy_gap(self)` - *Δ_S* = ε_c en punto excepcional con corrección de regularización*
- `compute_pontryagin_number` (line 776) `def compute_pontryagin_number(self)` - *S_top[G] = χ(G)/|V| (número de Euler normalizado)*
- `compute_freedom` (line 792) `def compute_freedom(self)` - *L[G] = Δ_S* / S_top[G] con protección de división por cero*
- `is_gauge_invariant` (line 803) `def is_gauge_invariant(self)` - *|L[G] - 1| < 0.05 en estado crítico (invariante de libertad)*
- `ising_quantum` (line 823) `def ising_quantum(network)` - *Modelo de Ising cuántico transversal en red fractal.
Predice t_c sin SYK₈ ni E₈ (teoría efectiva estándar).*
- `syk4` (line 843) `def syk4(network)` - *SYK₄ estándar (sin R-simetría Spin(7) ni E₈).
Predice α sin postulado de retículo.*
- `random_network` (line 860) `def random_network(network)` - *Red aleatoria Erdős-Rényi sin percolación cuántica ni estructura.*
- `__init__` (line 883) `def __init__(self, resma, myelin, network, freedom)`
- `predict_all` (line 891) `def predict_all(self)` - *Predicciones RESMA 4.0 con valores empíricos objetivo*
- `_predict_diffraction_peak` (line 906) `def _predict_diffraction_peak(self)` - *q₀ = 2π/L_E8 (predicción de difracción UASED)*
- `compute_log_bayes_factor` (line 912) `def compute_log_bayes_factor(self)` - *log(BF) = ΔAIC/2 donde AIC = 2k - 2ln(L)
FIX RESMA 4.0: Usar espacio logarítmico para evitar desbordamiento.*
- `__init__` (line 981) `def __init__(self, predictions)`
- `_define_protocols` (line 985) `def _define_protocols(self)` - *Definir protocolos experimentales con parámetros técnicos*
- `evaluate_feasibility` (line 1014) `def evaluate_feasibility(self, budget, time_limit)` - *Evaluar viabilidad del protocolo completo*
- `simulate_experimental_outcome` (line 1027) `def simulate_experimental_outcome(self, protocol_name)` - *Simular resultado experimental con ruido realista*

#### `microbi.py.py`
**Path:** `microbi.py.py`

**Classes:**
- `EpistemicCuriosityCPU` (line 37) `class EpistemicCuriosityCPU`
- `LiquidNeuronCPU` (line 91) `class LiquidNeuronCPU`
- `BicameralAttentionCPU` (line 195) `class BicameralAttentionCPU`
- `RightHemisphereCPU` (line 248) `class RightHemisphereCPU`
- `LeftHemisphereCPU` (line 308) `class LeftHemisphereCPU`
- `CorpusCallosumCPU` (line 436) `class CorpusCallosumCPU`
- `NeuroLogosBicameralCPU` (line 473) `class NeuroLogosBicameralCPU`
- `NeuralDiagnosticsCPU` (line 513) `class NeuralDiagnosticsCPU`
- `CurriculumSchedulerCPU` (line 602) `class CurriculumSchedulerCPU`
- `Flickr8kDatasetCPU` (line 671) `class Flickr8kDatasetCPU(Dataset)`

**Functions:**
- `build_vocab_flickr` (line 650) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k_cpu` (line 721) `def setup_flickr8k_cpu(data_dir)` - *Descarga Flickr8k automáticamente (igual que la versión original)*
- `train_bicameral_cpu` (line 801) `def train_bicameral_cpu()`
- `__init__` (line 38) `def __init__(self, feature_dim, hidden_dim)`
- `compute_intrinsic_reward` (line 55) `def compute_intrinsic_reward(self, state, action, next_state)`
- `update` (line 71) `def update(self, state, action, next_state)`
- `__init__` (line 92) `def __init__(self, in_dim, out_dim)`
- `forward` (line 117) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 168) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 196) `def __init__(self, dim, num_heads)`
- `forward` (line 211) `def forward(self, x, mask)`
- `__init__` (line 249) `def __init__(self, output_dim)`
- `forward` (line 285) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 309) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 338) `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)`
- `_get_init_state` (line 416) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 421) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 437) `def __init__(self, dim)`
- `forward` (line 456) `def forward(self, right_features)`
- `__init__` (line 474) `def __init__(self, vocab_size)`
- `forward` (line 480) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)`
- `__init__` (line 514) `def __init__(self)`
- `measure_callosal_flow` (line 523) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 530) `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `update` (line 548) `def update(self)`
- `get_recent_avg` (line 553) `def get_recent_avg(self, key, n)`
- `report` (line 560) `def report(self, epoch)`
- `__init__` (line 603) `def __init__(self, total_epochs)`
- `get_phase` (line 611) `def get_phase(self, epoch)`
- `get_plasticity` (line 617) `def get_plasticity(self, epoch)`
- `get_exploration_bonus` (line 627) `def get_exploration_bonus(self, epoch)`
- `get_temperature` (line 636) `def get_temperature(self, epoch)`
- `should_consolidate` (line 646) `def should_consolidate(self, epoch)`
- `__init__` (line 672) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 690) `def __len__(self)`
- `__getitem__` (line 693) `def __getitem__(self, idx)`
- `nan_hook` (line 880) `def nan_hook(module, grad_input, grad_output)`

#### `min_test_synergy.py`
**Path:** `min_test_synergy.py`

*No symbols extracted*

#### `minibi.py`
**Path:** `minibi.py`

**Classes:**
- `Flickr8kDataset` (line 124) `class Flickr8kDataset(Dataset)`
- `LiquidNeuron` (line 167) `class LiquidNeuron`
- `RightHemisphere` (line 246) `class RightHemisphere` - *Especialización: Visión espacial, reconocimiento de objetos, contexto global
Usa ResNet-50 pretrained en ImageNet → Feature extraction robusto*
- `LeftHemisphere` (line 282) `class LeftHemisphere`
- `CorpusCallosum` (line 382) `class CorpusCallosum`
- `NeuroLogosBicameral` (line 403) `class NeuroLogosBicameral`
- `NeuralDiagnostics` (line 420) `class NeuralDiagnostics`
- `LifeCycle` (line 518) `class LifeCycle`

**Functions:**
- `setup_flickr8k` (line 35) `def setup_flickr8k(data_dir)` - *Descarga Flickr8k automáticamente*
- `build_vocab_flickr` (line 104) `def build_vocab_flickr(captions_file, vocab_size)`
- `build_vocab` (line 490) `def build_vocab(ann_file, vocab_size)` - *Construir vocabulario desde annotations de COCO*
- `train_bicameral` (line 536) `def train_bicameral()`
- `__init__` (line 125) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 143) `def __len__(self)`
- `__getitem__` (line 146) `def __getitem__(self, idx)`
- `__init__` (line 168) `def __init__(self, in_dim, out_dim)`
- `forward` (line 185) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 219) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 251) `def __init__(self, output_dim)`
- `forward` (line 268) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 283) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 305) `def forward(self, visual_context, captions, max_len, return_gate, temperature)`
- `_get_init_state` (line 361) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 366) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 383) `def __init__(self, dim)`
- `forward` (line 392) `def forward(self, right_features)`
- `__init__` (line 404) `def __init__(self, vocab_size)`
- `forward` (line 410) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature)`
- `__init__` (line 421) `def __init__(self)`
- `measure_callosal_flow` (line 432) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 439) `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `update` (line 443) `def update(self)`
- `get_recent_avg` (line 448) `def get_recent_avg(self, key, n)`
- `report` (line 455) `def report(self, epoch)`
- `__init__` (line 519) `def __init__(self, total_epochs)`
- `get_plasticity` (line 522) `def get_plasticity(self, epoch)`

#### `minibi2.py`
**Path:** `minibi2.py`

**Classes:**
- `EpistemicCuriosity` (line 36) `class EpistemicCuriosity` - *Implementa curiosidad intrínseca basada en incertidumbre predictiva.
Paper: "Curiosity-driven Exploration by Self-supervised Prediction"*
- `LiquidNeuronV2` (line 101) `class LiquidNeuronV2` - *Mejoras:
- Regularización adaptativa (no fija)
- Homeostasis metabólica
- Rango de operación expandido*
- `BicameralAttention` (line 212) `class BicameralAttention` - *Atención multi-escala que simula integración hemisférica.
- Local attention: detalles finos (hemisferio izquierdo)
- Global attention: contexto amplio (hemisferio derecho)*
- `RightHemisphereV2` (line 279) `class RightHemisphereV2` - *Mejoras:
- Spatial attention pyramid
- Multi-scale feature extraction
- Liquid neuron con homeostasis*
- `LeftHemisphereV2` (line 344) `class LeftHemisphereV2` - *Mejoras:
- Bicameral attention
- Gated residual connections
- Adaptive gate range
- Curiosity-driven sampling*
- `CorpusCallosumV2` (line 519) `class CorpusCallosumV2` - *Mejoras:
- Gating bidireccional
- Modulación adaptativa
- Información mutua maximizada*
- `NeuroLogosBicameralV2` (line 562) `class NeuroLogosBicameralV2`
- `NeuralDiagnosticsV2` (line 591) `class NeuralDiagnosticsV2`
- `CurriculumScheduler` (line 700) `class CurriculumScheduler` - *Curriculum learning con fases de desarrollo cognitivo*
- `Flickr8kDataset` (line 781) `class Flickr8kDataset(Dataset)`

**Functions:**
- `build_vocab_flickr` (line 760) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 821) `def setup_flickr8k(data_dir)` - *Descarga Flickr8k automáticamente*
- `train_bicameral_v2` (line 893) `def train_bicameral_v2()`
- `__init__` (line 41) `def __init__(self, feature_dim, hidden_dim)`
- `compute_intrinsic_reward` (line 60) `def compute_intrinsic_reward(self, state, action, next_state)` - *Recompensa intrínseca = error de predicción del forward model
Incentiva explorar tokens que son difíciles de predecir*
- `update` (line 83) `def update(self, state, action, next_state)` - *Entrena los modelos de curiosidad*
- `__init__` (line 108) `def __init__(self, in_dim, out_dim)`
- `forward` (line 137) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 185) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 218) `def __init__(self, dim, num_heads)`
- `forward` (line 236) `def forward(self, x, mask)`
- `__init__` (line 286) `def __init__(self, output_dim)`
- `forward` (line 320) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 352) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 387) `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)`
- `_get_init_state` (line 498) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 503) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 526) `def __init__(self, dim)`
- `forward` (line 546) `def forward(self, right_features)`
- `__init__` (line 563) `def __init__(self, vocab_size)`
- `forward` (line 569) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)`
- `__init__` (line 592) `def __init__(self)`
- `measure_callosal_flow` (line 612) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 619) `def measure_vocab_diversity(self, generated_tokens, vocab_size)` - *Mide diversidad real + entropía*
- `update` (line 639) `def update(self)`
- `get_recent_avg` (line 644) `def get_recent_avg(self, key, n)`
- `report` (line 651) `def report(self, epoch)`
- `__init__` (line 704) `def __init__(self, total_epochs)`
- `get_phase` (line 712) `def get_phase(self, epoch)`
- `get_plasticity` (line 718) `def get_plasticity(self, epoch)`
- `get_exploration_bonus` (line 732) `def get_exploration_bonus(self, epoch)` - *Bonus de curiosidad que decae con el tiempo*
- `get_temperature` (line 743) `def get_temperature(self, epoch)` - *Temperature que decae suavemente*
- `should_consolidate` (line 755) `def should_consolidate(self, epoch)` - *Decide cuándo hacer consolidación SVD*
- `__init__` (line 782) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 800) `def __len__(self)`
- `__getitem__` (line 803) `def __getitem__(self, idx)`

#### `minibi_c.py`
**Path:** `minibi_c.py`

**Classes:**
- `EpistemicCuriosity` (line 37) `class EpistemicCuriosity`
- `LiquidNeuronV2` (line 94) `class LiquidNeuronV2`
- `BicameralAttention` (line 197) `class BicameralAttention`
- `RightHemisphereV2` (line 247) `class RightHemisphereV2`
- `LeftHemisphereV2` (line 301) `class LeftHemisphereV2`
- `CorpusCallosumV2` (line 433) `class CorpusCallosumV2`
- `NeuroLogosBicameralV2` (line 464) `class NeuroLogosBicameralV2`
- `NeuralDiagnosticsV2` (line 493) `class NeuralDiagnosticsV2`
- `CurriculumScheduler` (line 593) `class CurriculumScheduler`
- `Flickr8kDataset` (line 667) `class Flickr8kDataset(Dataset)`

**Functions:**
- `build_vocab_flickr` (line 644) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 722) `def setup_flickr8k(data_dir)` - *Descarga Flickr8k automáticamente*
- `train_bicameral_v2` (line 794) `def train_bicameral_v2()`
- `__init__` (line 38) `def __init__(self, feature_dim, hidden_dim)`
- `compute_intrinsic_reward` (line 55) `def compute_intrinsic_reward(self, state, action, next_state)`
- `update` (line 72) `def update(self, state, action, next_state)`
- `__init__` (line 95) `def __init__(self, in_dim, out_dim)`
- `forward` (line 121) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 170) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 198) `def __init__(self, dim, num_heads)`
- `forward` (line 212) `def forward(self, x, mask)`
- `__init__` (line 248) `def __init__(self, output_dim)`
- `forward` (line 281) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 302) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 332) `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)`
- `_get_init_state` (line 412) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 417) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 434) `def __init__(self, dim)`
- `forward` (line 450) `def forward(self, right_features)`
- `__init__` (line 465) `def __init__(self, vocab_size)`
- `forward` (line 471) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)`
- `__init__` (line 494) `def __init__(self)`
- `measure_callosal_flow` (line 510) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 517) `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `update` (line 535) `def update(self)`
- `get_recent_avg` (line 541) `def get_recent_avg(self, key, n)`
- `report` (line 548) `def report(self, epoch)`
- `__init__` (line 594) `def __init__(self, total_epochs)`
- `get_phase` (line 602) `def get_phase(self, epoch)`
- `get_plasticity` (line 608) `def get_plasticity(self, epoch)`
- `get_exploration_bonus` (line 619) `def get_exploration_bonus(self, epoch)`
- `get_temperature` (line 629) `def get_temperature(self, epoch)`
- `should_consolidate` (line 640) `def should_consolidate(self, epoch)`
- `__init__` (line 668) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 686) `def __len__(self)`
- `__getitem__` (line 689) `def __getitem__(self, idx)`

#### `minibi_reduced.py.py`
**Path:** `minibi_reduced.py.py`

**Classes:**
- `EpistemicCuriosity` (line 39) `class EpistemicCuriosity`
- `LiquidNeuronV2` (line 81) `class LiquidNeuronV2`
- `BicameralAttention` (line 170) `class BicameralAttention`
- `RightHemisphereV2` (line 213) `class RightHemisphereV2`
- `LeftHemisphereV2` (line 247) `class LeftHemisphereV2`
- `CorpusCallosumV2` (line 338) `class CorpusCallosumV2`
- `NeuroLogosBicameralV2` (line 356) `class NeuroLogosBicameralV2`
- `Flickr8kDataset` (line 396) `class Flickr8kDataset(Dataset)`

**Functions:**
- `build_vocab_flickr` (line 381) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 431) `def setup_flickr8k(data_dir)`
- `train_bicameral_v2` (line 482) `def train_bicameral_v2()`
- `__init__` (line 40) `def __init__(self, feature_dim, hidden_dim)`
- `compute_intrinsic_reward` (line 55) `def compute_intrinsic_reward(self, state, action, next_state)`
- `update` (line 70) `def update(self, state, action, next_state)`
- `__init__` (line 82) `def __init__(self, in_dim, out_dim)`
- `forward` (line 105) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 141) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 171) `def __init__(self, dim, num_heads)`
- `forward` (line 184) `def forward(self, x, mask)`
- `__init__` (line 214) `def __init__(self, output_dim)`
- `forward` (line 237) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 248) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 266) `def forward(self, visual_context, captions, max_len, return_diagnostics, temperature, exploration_bonus)`
- `_get_init_state` (line 323) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 328) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 339) `def __init__(self, dim)`
- `forward` (line 349) `def forward(self, right_features)`
- `__init__` (line 357) `def __init__(self, vocab_size)`
- `forward` (line 363) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature, exploration_bonus)`
- `__init__` (line 397) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 412) `def __len__(self)`
- `__getitem__` (line 414) `def __getitem__(self, idx)`

#### `miniminibi.py`
**Path:** `miniminibi.py`

**Classes:**
- `LiquidNeuron` (line 118) `class LiquidNeuron`
- `RightHemisphere` (line 191) `class RightHemisphere`
- `LeftHemisphere` (line 214) `class LeftHemisphere`
- `CorpusCallosum` (line 314) `class CorpusCallosum`
- `NeuroLogosBicameral` (line 332) `class NeuroLogosBicameral`
- `NeuralDiagnostics` (line 351) `class NeuralDiagnostics` - *Sistema de monitoreo de salud cerebral bicameral*
- `Flickr8kDataset` (line 442) `class Flickr8kDataset(Dataset)`
- `LifeCycle` (line 508) `class LifeCycle`

**Functions:**
- `setup_flickr8k` (line 30) `def setup_flickr8k(data_dir)` - *Descarga Flickr8k automáticamente desde Kaggle
Requiere: pip install kaggle
Y tener configurado ~/.kaggle/kaggle.json

Alternativa sin Kaggle: descarga manual desde
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip*
- `build_vocab_flickr` (line 485) `def build_vocab_flickr(captions_file, vocab_size)`
- `train_bicameral` (line 523) `def train_bicameral()`
- `__init__` (line 119) `def __init__(self, in_dim, out_dim)`
- `forward` (line 136) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 165) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 192) `def __init__(self, output_dim)`
- `forward` (line 205) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 215) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 237) `def forward(self, visual_context, captions, max_len, return_gate, temperature)`
- `_get_init_state` (line 293) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 298) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 315) `def __init__(self, dim)`
- `forward` (line 324) `def forward(self, right_features)`
- `__init__` (line 333) `def __init__(self, vocab_size)`
- `forward` (line 339) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics, temperature)`
- `__init__` (line 353) `def __init__(self)`
- `measure_callosal_flow` (line 363) `def measure_callosal_flow(self, right_features, left_context)` - *Mide qué tan bien está fluyendo información entre hemisferios*
- `measure_vocab_diversity` (line 372) `def measure_vocab_diversity(self, generated_tokens, vocab_size)` - *Mide diversidad de vocabulario generado (evitar colapso)*
- `measure_gate_health` (line 378) `def measure_gate_health(self, gate_activations)` - *Verifica que el liquid gate no colapse a 0 o 1*
- `update` (line 387) `def update(self)`
- `report` (line 392) `def report(self, epoch)` - *Reporte diagnóstico completo*
- `__init__` (line 443) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 462) `def __len__(self)`
- `__getitem__` (line 465) `def __getitem__(self, idx)`
- `__init__` (line 509) `def __init__(self, total_epochs)`
- `get_plasticity` (line 512) `def get_plasticity(self, epoch)`

#### `nemesis.py`
**Path:** `nemesis.py`

**Classes:**
- `DataEnvironment` (line 23) `class DataEnvironment`
- `NeuralController` (line 41) `class NeuralController`
- `HyperLiquidNeuron` (line 60) `class HyperLiquidNeuron`
- `NemesisNetwork` (line 126) `class NemesisNetwork`
- `ExperimentConfig` (line 192) `class ExperimentConfig`

**Functions:**
- `seed_everything` (line 13) `def seed_everything(seed)`
- `run_hyper_experiment` (line 143) `def run_hyper_experiment(epochs, name, dynamic_mode)`
- `__init__` (line 24) `def __init__(self)`
- `get_batch` (line 33) `def get_batch(self, phase, bs)`
- `__init__` (line 42) `def __init__(self)`
- `forward` (line 52) `def forward(self, surprise, entropy)`
- `__init__` (line 61) `def __init__(self, d_in, d_out, dynamic_mode)`
- `forward` (line 79) `def forward(self, x)`
- `__init__` (line 127) `def __init__(self, config, dynamic_mode)`
- `forward` (line 133) `def forward(self, x)`

#### `nested1.1.py`
**Path:** `nested1.1.py`

**Classes:**
- `Config` (line 27) `class Config`
- `CMSLayer` (line 128) `class CMSLayer`
- `NestedBrain` (line 200) `class NestedBrain`

**Functions:**
- `safe_serialize` (line 61) `def safe_serialize(obj)` - *Convierte objetos a formato serializable (evita recursión y objetos complejos).*
- `save_checkpoint` (line 80) `def save_checkpoint(epoch, model_state, optimizer_state, config, metrics, checkpoint_dir)` - *Guarda checkpoint de época: modelo (.pth) + metadatos (.pkl).*
- `cleanup_old_checkpoints` (line 105) `def cleanup_old_checkpoints(checkpoint_dir, keep_last)` - *Mantiene solo los últimos `keep_last` checkpoints.*
- `get_cifar10_loaders` (line 241) `def get_cifar10_loaders(config)`
- `evaluate` (line 267) `def evaluate(model, loader, device)`
- `train` (line 293) `def train(config)`
- `run_ablation_study` (line 371) `def run_ablation_study()`
- `__init__` (line 129) `def __init__(self, dim, config)`
- `forward` (line 148) `def forward(self, x)`
- `get_norms` (line 190) `def get_norms(self)`
- `__init__` (line 201) `def __init__(self, config)`
- `forward` (line 221) `def forward(self, x)`
- `get_ablation_state` (line 226) `def get_ablation_state(self)`
- `get_norms` (line 233) `def get_norms(self)`

#### `nested1.py`
**Path:** `nested1.py`

**Classes:**
- `Config` (line 24) `class Config`
- `CMSLayer` (line 59) `class CMSLayer`
- `NestedBrain` (line 141) `class NestedBrain`

**Functions:**
- `get_cifar10_loaders` (line 182) `def get_cifar10_loaders(config)`
- `evaluate` (line 208) `def evaluate(model, loader, device)`
- `train` (line 234) `def train(config)`
- `run_ablation_study` (line 301) `def run_ablation_study()`
- `__init__` (line 60) `def __init__(self, dim, config)`
- `forward` (line 83) `def forward(self, x)`
- `get_norms` (line 131) `def get_norms(self)`
- `__init__` (line 142) `def __init__(self, config)`
- `forward` (line 162) `def forward(self, x)`
- `get_ablation_state` (line 167) `def get_ablation_state(self)`
- `get_norms` (line 174) `def get_norms(self)`

#### `nestedtopobrain.py`
**Path:** `nestedtopobrain.py`

**Classes:**
- `Config` (line 29) `class Config`
- `ResourceMonitor` (line 133) `class ResourceMonitor`
- `PrefrontalOrchestrator` (line 178) `class PrefrontalOrchestrator` - *Módulo de control ejecutivo que monitoriza el estado de la red y emite señales
dinámicas de activación/inhibición para cada mecanismo neuromodulatorio.
Opera como un sistema de homeostasis topológica y metabólica.
FIX v24: Gestión corregida del grafo computacional recurrente (BPTT).*
- `TopologyMetrics` (line 334) `class TopologyMetrics`
- `TopologicalHealthSovereignty` (line 343) `class TopologicalHealthSovereignty` - *Monitor SVD Completo con criterios neurocientíficos

Referencias:
- Sporns (2016): Human brain networks ~1-3% sparsity
- Bullmore & Sporns (2009): Small-world topology con L_score > 3*
- `CheckpointManager` (line 429) `class CheckpointManager` - *Manager robusto v18 con backups y metadata*
- `SupConLoss` (line 525) `class SupConLoss`
- `AsymmetricPredictiveErrorCell` (line 542) `class AsymmetricPredictiveErrorCell`
- `LearnableAbsenceGating` (line 562) `class LearnableAbsenceGating`
- `SymbioticBasisRefinement` (line 580) `class SymbioticBasisRefinement`
- `ContinuumMemoryCell` (line 625) `class ContinuumMemoryCell`
- `AdaptiveCombinatorialComplexLayer` (line 750) `class AdaptiveCombinatorialComplexLayer`
- `ResidualBlock` (line 946) `class ResidualBlock`
- `VisualCortex` (line 967) `class VisualCortex`
- `TopoBrainV24` (line 1022) `class TopoBrainV24`

**Functions:**
- `seed_everything` (line 119) `def seed_everything(seed)`
- `get_dataloaders` (line 481) `def get_dataloaders(config)` - *DataLoaders con augmentation de alto rendimiento para CIFAR-10*
- `save_topology_visualization` (line 1406) `def save_topology_visualization(model, epoch, run_name)` - *Visualización v18 completa*
- `save_node_importance_viz` (line 1450) `def save_node_importance_viz(model, epoch, run_name)` - *Visualización de importancia de nodos v18*
- `analyze_topology_clustering` (line 1472) `def analyze_topology_clustering(model, run_name)` - *Clustering espectral v18*
- `analyze_topology_flow` (line 1511) `def analyze_topology_flow(model, dataloader, run_name, num_samples)` - *Análisis de flujo de información con captura genérica de outputs*
- `visualize_topology_as_graph` (line 1577) `def visualize_topology_as_graph(model, run_name, threshold)` - *Grafo v18 con métricas*
- `analyze_topology_evolution` (line 1629) `def analyze_topology_evolution(run_name)` - *Análisis temporal completo v18*
- `comprehensive_topology_analysis` (line 1690) `def comprehensive_topology_analysis(model, dataloader, run_name)` - *Suite completa de análisis v18*
- `run_ablation_study` (line 1713) `def run_ablation_study()` - *Suite de ablación v18 completa*
- `visualize_memory_evolution` (line 1814) `def visualize_memory_evolution(model, epoch, run_name)` - *Visualiza evolución de memorias semánticas

Crítico para entender si la consolidación hipocampal→cortical está funcionando.
SVD spectrum revela estructura de representaciones aprendidas.*
- `analyze_gradient_flow` (line 1905) `def analyze_gradient_flow(model, epoch, run_name)` - *Análisis detallado del flujo de gradientes

Detecta vanishing/exploding gradients y capas muertas.
Critical para debugging de arquitecturas con fast weights.*
- `make_adversarial_pgd` (line 1972) `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)` - *PGD ataque con congelamiento total de pesos y detach explícito de estados.
FIX: Asegura que el grafo computacional no se rompa y que los estados previos sean genuinamente independientes.*
- `evaluate` (line 2050) `def evaluate(model, loader, config, adversarial, controls)` - *Evaluación con plasticidad residual (test-time adaptation)
Biológicamente plausible: el cerebro no se apaga durante percepción*
- `train_epoch` (line 2122) `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)` - *Entrenamiento homeostático con lista manual en lugar de deque*
- `train_model` (line 2347) `def train_model(config, run_name)`
- `main` (line 2518) `def main()` - *CLI v24 completo con Orquestador Prefrontal*
- `__post_init__` (line 89) `def __post_init__(self)`
- `to_dict` (line 95) `def to_dict(self)`
- `get_supcon_lambda` (line 98) `def get_supcon_lambda(self, epoch)`
- `get_sparsity_lambda` (line 104) `def get_sparsity_lambda(self, epoch)`
- `get_memory_gb` (line 135) `def get_memory_gb()`
- `get_gpu_memory_gb` (line 140) `def get_gpu_memory_gb()`
- `log` (line 146) `def log(prefix)`
- `clear_cache` (line 153) `def clear_cache()`
- `check_limit` (line 159) `def check_limit(limit_gb, abort_on_limit)`
- `__init__` (line 185) `def __init__(self, config)`
- `forward` (line 222) `def forward(self, metrics_dict)` - *Orquestador v27: Allostasis con Frenado de Emergencia (Gradient-Aware).

FIX CRÍTICO: EL ORQUESTADOR AHORA "SIENTE" SI EL GRADIENTE EXPLOTA
- Si loss > 10.0: Entra en MODO PÁNICO (LR_Scale mínimo)
- Si delta_loss > 0 (loss subiendo): Invierte la señal de aceleración
- Si grad_norm > 10: Reduce plasticidad para estabilizar*
- `detach_state` (line 322) `def detach_state(self)` - *Rompe el grafo computacional para evitar retropropagación infinita entre batches*
- `reset_context` (line 327) `def reset_context(self)` - *Resetear contexto al inicio de cada época*
- `__init__` (line 350) `def __init__(self, model, config, epsilon_c)`
- `_analyze_matrix` (line 356) `def _analyze_matrix(self, weight_matrix, name)`
- `calculate` (line 402) `def calculate(self, epoch)`
- `get_critical_summary` (line 419) `def get_critical_summary(self)`
- `__init__` (line 431) `def __init__(self, run_name)`
- `save` (line 436) `def save(self, data, name)`
- `load` (line 464) `def load(self, name)`
- `__init__` (line 526) `def __init__(self, temperature)`
- `forward` (line 529) `def forward(self, features, labels)`
- `__init__` (line 543) `def __init__(self, dim, use_spectral)`
- `forward` (line 553) `def forward(self, input_signal, prediction)`
- `__init__` (line 563) `def __init__(self, dim, min_gate)`
- `forward` (line 573) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 581) `def __init__(self, dim, num_atoms)`
- `_maintain_orthogonality` (line 595) `def _maintain_orthogonality(self)`
- `forward` (line 600) `def forward(self, x)`
- `__init__` (line 626) `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)`
- `forward` (line 675) `def forward(self, x, state_M, controls)`
- `__init__` (line 751) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- `invalidate_sparse_cache` (line 789) `def invalidate_sparse_cache(self)`
- `_validate_and_fix_state` (line 792) `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)`
- `get_node_importance` (line 811) `def get_node_importance(self)` - *FIX: Método faltante para obtener importancia de nodos*
- `forward` (line 817) `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)` - *Forward con señales de control del Orquestador y retorno de ortho deviation*
- `__init__` (line 947) `def __init__(self, in_channels, out_channels, stride)`
- `forward` (line 961) `def forward(self, x)`
- `__init__` (line 968) `def __init__(self, output_dim, grid_size)`
- `forward` (line 994) `def forward(self, x)`
- `__init__` (line 1023) `def __init__(self, config, in_channels)`
- `initialize_memories` (line 1087) `def initialize_memories(self, dataloader)`
- `_initialize_layer_memory` (line 1124) `def _initialize_layer_memory(self, cell, x_input, name)`
- `consolidate_semantic_memories` (line 1141) `def consolidate_semantic_memories(self)`
- `set_epoch` (line 1169) `def set_epoch(self, epoch)`
- `calculate_ortho_loss` (line 1174) `def calculate_ortho_loss(self, ortho_deviation, controls)`
- `calculate_topology_diversity_loss` (line 1180) `def calculate_topology_diversity_loss(self, controls)`
- `_init_grid_topology` (line 1193) `def _init_grid_topology(self, N)`
- `get_topology` (line 1215) `def get_topology(self, return_sparse)`
- `forward` (line 1226) `def forward(self, x, prev_states, controls)`
- `prune_topology` (line 1298) `def prune_topology(self, controls)`
- `warmup_topo` (line 2381) `def warmup_topo(epoch)`

#### `nestedtopobrain_v1.py`
**Path:** `nestedtopobrain_v1.py`

**Classes:**
- `Config` (line 29) `class Config`
- `ResourceMonitor` (line 128) `class ResourceMonitor`
- `PrefrontalOrchestrator` (line 173) `class PrefrontalOrchestrator` - *Módulo de control ejecutivo que monitoriza el estado de la red y emite señales
dinámicas de activación/inhibición para cada mecanismo neuromodulatorio.
Opera como un sistema de homeostasis topológica y metabólica.
FIX v24: Gestión corregida del grafo computacional recurrente (BPTT).*
- `TopologyMetrics` (line 269) `class TopologyMetrics`
- `TopologicalHealthSovereignty` (line 278) `class TopologicalHealthSovereignty` - *Monitor SVD Completo con criterios neurocientíficos

Referencias:
- Sporns (2016): Human brain networks ~1-3% sparsity
- Bullmore & Sporns (2009): Small-world topology con L_score > 3*
- `CheckpointManager` (line 364) `class CheckpointManager` - *Manager robusto v18 con backups y metadata*
- `SupConLoss` (line 455) `class SupConLoss`
- `AsymmetricPredictiveErrorCell` (line 472) `class AsymmetricPredictiveErrorCell`
- `LearnableAbsenceGating` (line 492) `class LearnableAbsenceGating`
- `SymbioticBasisRefinement` (line 510) `class SymbioticBasisRefinement`
- `ContinuumMemoryCell` (line 549) `class ContinuumMemoryCell`
- `AdaptiveCombinatorialComplexLayer` (line 676) `class AdaptiveCombinatorialComplexLayer`
- `TopoBrainV24` (line 879) `class TopoBrainV24`

**Functions:**
- `seed_everything` (line 114) `def seed_everything(seed)`
- `get_dataloaders` (line 416) `def get_dataloaders(config)`
- `save_topology_visualization` (line 1304) `def save_topology_visualization(model, epoch, run_name)` - *Visualización v18 completa*
- `save_node_importance_viz` (line 1348) `def save_node_importance_viz(model, epoch, run_name)` - *Visualización de importancia de nodos v18*
- `analyze_topology_clustering` (line 1370) `def analyze_topology_clustering(model, run_name)` - *Clustering espectral v18*
- `analyze_topology_flow` (line 1409) `def analyze_topology_flow(model, dataloader, run_name, num_samples)` - *Análisis de flujo v18*
- `visualize_topology_as_graph` (line 1461) `def visualize_topology_as_graph(model, run_name, threshold)` - *Grafo v18 con métricas*
- `analyze_topology_evolution` (line 1513) `def analyze_topology_evolution(run_name)` - *Análisis temporal completo v18*
- `comprehensive_topology_analysis` (line 1574) `def comprehensive_topology_analysis(model, dataloader, run_name)` - *Suite completa de análisis v18*
- `run_ablation_study` (line 1597) `def run_ablation_study()` - *Suite de ablación v18 completa*
- `visualize_memory_evolution` (line 1698) `def visualize_memory_evolution(model, epoch, run_name)` - *Visualiza evolución de memorias semánticas

Crítico para entender si la consolidación hipocampal→cortical está funcionando.
SVD spectrum revela estructura de representaciones aprendidas.*
- `analyze_gradient_flow` (line 1789) `def analyze_gradient_flow(model, epoch, run_name)` - *Análisis detallado del flujo de gradientes

Detecta vanishing/exploding gradients y capas muertas.
Critical para debugging de arquitecturas con fast weights.*
- `make_adversarial_pgd` (line 1856) `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)` - *PGD attack con congelamiento total de pesos (Protocolo de Aislamiento Sináptico).
Garantiza que la generación del ataque no modifique los gradientes del modelo.*
- `evaluate` (line 1936) `def evaluate(model, loader, config, adversarial, controls)` - *Evaluación optimizada para arquitecturas biológicas complejas (Nested/Grid).
Implementa 'Gradient Shielding' para prevenir OOM en inferencia.*
- `train_epoch` (line 2000) `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)` - *Entrenamiento homeostático con gestión rigurosa de grafos y memoria.
FIX v24.1: Restaurada la visualización detallada de decisiones del Orquestador (Logs de Actividad Prefrontal).*
- `train_model` (line 2196) `def train_model(config, run_name)` - *Training loop principal - FIX v24: Orden correcto de Scheduler y gestión de memoria.*
- `main` (line 2398) `def main()` - *CLI v24 completo con Orquestador Prefrontal*
- `__post_init__` (line 83) `def __post_init__(self)`
- `to_dict` (line 89) `def to_dict(self)`
- `get_supcon_lambda` (line 92) `def get_supcon_lambda(self, epoch)`
- `get_sparsity_lambda` (line 98) `def get_sparsity_lambda(self, epoch)`
- `get_memory_gb` (line 130) `def get_memory_gb()`
- `get_gpu_memory_gb` (line 135) `def get_gpu_memory_gb()`
- `log` (line 141) `def log(prefix)`
- `clear_cache` (line 148) `def clear_cache()`
- `check_limit` (line 154) `def check_limit(limit_gb, abort_on_limit)`
- `__init__` (line 180) `def __init__(self, config)`
- `forward` (line 217) `def forward(self, metrics_dict)` - *Input: Diccionario con métricas del estado actual
Output: Diccionario con señales de control escaladas [0,1]*
- `detach_state` (line 257) `def detach_state(self)` - *Rompe el grafo computacional para evitar retropropagación infinita entre batches*
- `reset_context` (line 262) `def reset_context(self)` - *Resetear contexto al inicio de cada época*
- `__init__` (line 285) `def __init__(self, model, config, epsilon_c)`
- `_analyze_matrix` (line 291) `def _analyze_matrix(self, weight_matrix, name)`
- `calculate` (line 337) `def calculate(self, epoch)`
- `get_critical_summary` (line 354) `def get_critical_summary(self)`
- `__init__` (line 366) `def __init__(self, run_name)`
- `save` (line 371) `def save(self, data, name)`
- `load` (line 399) `def load(self, name)`
- `__init__` (line 456) `def __init__(self, temperature)`
- `forward` (line 459) `def forward(self, features, labels)`
- `__init__` (line 473) `def __init__(self, dim, use_spectral)`
- `forward` (line 483) `def forward(self, input_signal, prediction)`
- `__init__` (line 493) `def __init__(self, dim, min_gate)`
- `forward` (line 503) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 511) `def __init__(self, dim, num_atoms)`
- `_maintain_orthogonality` (line 525) `def _maintain_orthogonality(self)`
- `forward` (line 530) `def forward(self, x)`
- `__init__` (line 550) `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)`
- `forward` (line 599) `def forward(self, x, state_M, controls)` - *FIX: Ahora acepta señales de control del Orquestador para modular
la plasticidad y consolidación en tiempo real.*
- `__init__` (line 677) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- `invalidate_sparse_cache` (line 715) `def invalidate_sparse_cache(self)`
- `_validate_and_fix_state` (line 718) `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)`
- `get_node_importance` (line 737) `def get_node_importance(self)` - *FIX: Método faltante para obtener importancia de nodos*
- `forward` (line 743) `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)` - *FIX: Integra señales de control del Orquestador con bypass condicional
para evitar computación innecesaria cuando gates están bajos*
- `__init__` (line 880) `def __init__(self, config, in_channels)`
- `initialize_memories` (line 934) `def initialize_memories(self, dataloader)`
- `_initialize_layer_memory` (line 974) `def _initialize_layer_memory(self, cell, x_input, name)`
- `consolidate_semantic_memories` (line 1000) `def consolidate_semantic_memories(self)`
- `set_epoch` (line 1045) `def set_epoch(self, epoch)`
- `calculate_ortho_loss` (line 1050) `def calculate_ortho_loss(self, controls)`
- `calculate_topology_diversity_loss` (line 1070) `def calculate_topology_diversity_loss(self, controls)`
- `_init_grid_topology` (line 1095) `def _init_grid_topology(self, N)`
- `get_topology` (line 1118) `def get_topology(self, return_sparse)`
- `forward` (line 1131) `def forward(self, x, prev_states, controls)` - *Forward con validación y detach explícito*
- `prune_topology` (line 1198) `def prune_topology(self, controls)`
- `warmup_topo` (line 2234) `def warmup_topo(epoch)`

#### `nestedtopobrain_v2.py`
**Path:** `nestedtopobrain_v2.py`

**Classes:**
- `Config` (line 29) `class Config`
- `ResourceMonitor` (line 127) `class ResourceMonitor`
- `PrefrontalOrchestrator` (line 172) `class PrefrontalOrchestrator` - *Módulo de control ejecutivo que monitoriza el estado de la red y emite señales
dinámicas de activación/inhibición para cada mecanismo neuromodulatorio.
Opera como un sistema de homeostasis topológica y metabólica.
FIX v24: Gestión corregida del grafo computacional recurrente (BPTT).*
- `TopologyMetrics` (line 268) `class TopologyMetrics`
- `TopologicalHealthSovereignty` (line 277) `class TopologicalHealthSovereignty` - *Monitor SVD Completo con criterios neurocientíficos

Referencias:
- Sporns (2016): Human brain networks ~1-3% sparsity
- Bullmore & Sporns (2009): Small-world topology con L_score > 3*
- `CheckpointManager` (line 363) `class CheckpointManager` - *Manager robusto v18 con backups y metadata*
- `SupConLoss` (line 454) `class SupConLoss`
- `AsymmetricPredictiveErrorCell` (line 471) `class AsymmetricPredictiveErrorCell`
- `LearnableAbsenceGating` (line 491) `class LearnableAbsenceGating`
- `SymbioticBasisRefinement` (line 509) `class SymbioticBasisRefinement`
- `ContinuumMemoryCell` (line 554) `class ContinuumMemoryCell`
- `AdaptiveCombinatorialComplexLayer` (line 681) `class AdaptiveCombinatorialComplexLayer`
- `TopoBrainV24` (line 878) `class TopoBrainV24`

**Functions:**
- `seed_everything` (line 113) `def seed_everything(seed)`
- `get_dataloaders` (line 415) `def get_dataloaders(config)`
- `save_topology_visualization` (line 1289) `def save_topology_visualization(model, epoch, run_name)` - *Visualización v18 completa*
- `save_node_importance_viz` (line 1333) `def save_node_importance_viz(model, epoch, run_name)` - *Visualización de importancia de nodos v18*
- `analyze_topology_clustering` (line 1355) `def analyze_topology_clustering(model, run_name)` - *Clustering espectral v18*
- `analyze_topology_flow` (line 1394) `def analyze_topology_flow(model, dataloader, run_name, num_samples)` - *Análisis de flujo v18*
- `visualize_topology_as_graph` (line 1448) `def visualize_topology_as_graph(model, run_name, threshold)` - *Grafo v18 con métricas*
- `analyze_topology_evolution` (line 1500) `def analyze_topology_evolution(run_name)` - *Análisis temporal completo v18*
- `comprehensive_topology_analysis` (line 1561) `def comprehensive_topology_analysis(model, dataloader, run_name)` - *Suite completa de análisis v18*
- `run_ablation_study` (line 1584) `def run_ablation_study()` - *Suite de ablación v18 completa*
- `visualize_memory_evolution` (line 1685) `def visualize_memory_evolution(model, epoch, run_name)` - *Visualiza evolución de memorias semánticas

Crítico para entender si la consolidación hipocampal→cortical está funcionando.
SVD spectrum revela estructura de representaciones aprendidas.*
- `analyze_gradient_flow` (line 1776) `def analyze_gradient_flow(model, epoch, run_name)` - *Análisis detallado del flujo de gradientes

Detecta vanishing/exploding gradients y capas muertas.
Critical para debugging de arquitecturas con fast weights.*
- `make_adversarial_pgd` (line 1843) `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)` - *PGD attack con congelamiento total de pesos (Protocolo de Aislamiento Sináptico).
FIX: Captura correcta de 5 valores de retorno del forward.*
- `evaluate` (line 1910) `def evaluate(model, loader, config, adversarial, controls)` - *Evaluación optimizada con Gradient Shielding.
FIX: Captura correcta de 5 valores de retorno del forward.*
- `train_epoch` (line 1963) `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)` - *Entrenamiento homeostático con inicialización de estados*
- `train_model` (line 2132) `def train_model(config, run_name)` - *Training loop principal - FIX v24: Orden correcto de Scheduler y gestión de memoria.*
- `main` (line 2344) `def main()` - *CLI v24 completo con Orquestador Prefrontal*
- `__post_init__` (line 82) `def __post_init__(self)`
- `to_dict` (line 88) `def to_dict(self)`
- `get_supcon_lambda` (line 91) `def get_supcon_lambda(self, epoch)`
- `get_sparsity_lambda` (line 97) `def get_sparsity_lambda(self, epoch)`
- `get_memory_gb` (line 129) `def get_memory_gb()`
- `get_gpu_memory_gb` (line 134) `def get_gpu_memory_gb()`
- `log` (line 140) `def log(prefix)`
- `clear_cache` (line 147) `def clear_cache()`
- `check_limit` (line 153) `def check_limit(limit_gb, abort_on_limit)`
- `__init__` (line 179) `def __init__(self, config)`
- `forward` (line 216) `def forward(self, metrics_dict)` - *Input: Diccionario con métricas del estado actual
Output: Diccionario con señales de control escaladas [0,1]*
- `detach_state` (line 256) `def detach_state(self)` - *Rompe el grafo computacional para evitar retropropagación infinita entre batches*
- `reset_context` (line 261) `def reset_context(self)` - *Resetear contexto al inicio de cada época*
- `__init__` (line 284) `def __init__(self, model, config, epsilon_c)`
- `_analyze_matrix` (line 290) `def _analyze_matrix(self, weight_matrix, name)`
- `calculate` (line 336) `def calculate(self, epoch)`
- `get_critical_summary` (line 353) `def get_critical_summary(self)`
- `__init__` (line 365) `def __init__(self, run_name)`
- `save` (line 370) `def save(self, data, name)`
- `load` (line 398) `def load(self, name)`
- `__init__` (line 455) `def __init__(self, temperature)`
- `forward` (line 458) `def forward(self, features, labels)`
- `__init__` (line 472) `def __init__(self, dim, use_spectral)`
- `forward` (line 482) `def forward(self, input_signal, prediction)`
- `__init__` (line 492) `def __init__(self, dim, min_gate)`
- `forward` (line 502) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 510) `def __init__(self, dim, num_atoms)`
- `_maintain_orthogonality` (line 524) `def _maintain_orthogonality(self)`
- `forward` (line 529) `def forward(self, x)`
- `__init__` (line 555) `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)`
- `forward` (line 604) `def forward(self, x, state_M, controls)` - *FIX: Ahora acepta señales de control del Orquestador para modular
la plasticidad y consolidación en tiempo real.*
- `__init__` (line 682) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- `invalidate_sparse_cache` (line 720) `def invalidate_sparse_cache(self)`
- `_validate_and_fix_state` (line 723) `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)`
- `get_node_importance` (line 742) `def get_node_importance(self)` - *FIX: Método faltante para obtener importancia de nodos*
- `forward` (line 748) `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)` - *Forward con señales de control del Orquestador y retorno de ortho deviation*
- `__init__` (line 879) `def __init__(self, config, in_channels)`
- `initialize_memories` (line 933) `def initialize_memories(self, dataloader)` - *Inicialización de memorias semánticas con captura correcta de 5 valores de retorno*
- `_initialize_layer_memory` (line 976) `def _initialize_layer_memory(self, cell, x_input, name)`
- `consolidate_semantic_memories` (line 1002) `def consolidate_semantic_memories(self)`
- `set_epoch` (line 1047) `def set_epoch(self, epoch)`
- `calculate_ortho_loss` (line 1052) `def calculate_ortho_loss(self, ortho_deviation, controls)` - *Calcula loss de ortogonalidad usando el deviation retornado por las capas*
- `calculate_topology_diversity_loss` (line 1060) `def calculate_topology_diversity_loss(self, controls)`
- `_init_grid_topology` (line 1085) `def _init_grid_topology(self, N)`
- `get_topology` (line 1108) `def get_topology(self, return_sparse)`
- `forward` (line 1121) `def forward(self, x, prev_states, controls)` - *Forward con validación, detach explícito, y retorno de ortho deviation*
- `prune_topology` (line 1183) `def prune_topology(self, controls)` - *Poda topológica con cálculo correcto de quantile*
- `warmup_topo` (line 2180) `def warmup_topo(epoch)`

#### `nestedtopobrain_v3.py`
**Path:** `nestedtopobrain_v3.py`

**Classes:**
- `Config` (line 29) `class Config`
- `ResourceMonitor` (line 127) `class ResourceMonitor`
- `PrefrontalOrchestrator` (line 172) `class PrefrontalOrchestrator` - *Módulo de control ejecutivo que monitoriza el estado de la red y emite señales
dinámicas de activación/inhibición para cada mecanismo neuromodulatorio.
Opera como un sistema de homeostasis topológica y metabólica.
FIX v24: Gestión corregida del grafo computacional recurrente (BPTT).*
- `TopologyMetrics` (line 324) `class TopologyMetrics`
- `TopologicalHealthSovereignty` (line 333) `class TopologicalHealthSovereignty` - *Monitor SVD Completo con criterios neurocientíficos

Referencias:
- Sporns (2016): Human brain networks ~1-3% sparsity
- Bullmore & Sporns (2009): Small-world topology con L_score > 3*
- `CheckpointManager` (line 419) `class CheckpointManager` - *Manager robusto v18 con backups y metadata*
- `SupConLoss` (line 510) `class SupConLoss`
- `AsymmetricPredictiveErrorCell` (line 527) `class AsymmetricPredictiveErrorCell`
- `LearnableAbsenceGating` (line 547) `class LearnableAbsenceGating`
- `SymbioticBasisRefinement` (line 565) `class SymbioticBasisRefinement`
- `ContinuumMemoryCell` (line 610) `class ContinuumMemoryCell`
- `AdaptiveCombinatorialComplexLayer` (line 737) `class AdaptiveCombinatorialComplexLayer`
- `TopoBrainV24` (line 934) `class TopoBrainV24`

**Functions:**
- `seed_everything` (line 113) `def seed_everything(seed)`
- `get_dataloaders` (line 471) `def get_dataloaders(config)`
- `save_topology_visualization` (line 1388) `def save_topology_visualization(model, epoch, run_name)` - *Visualización v18 completa*
- `save_node_importance_viz` (line 1432) `def save_node_importance_viz(model, epoch, run_name)` - *Visualización de importancia de nodos v18*
- `analyze_topology_clustering` (line 1454) `def analyze_topology_clustering(model, run_name)` - *Clustering espectral v18*
- `analyze_topology_flow` (line 1493) `def analyze_topology_flow(model, dataloader, run_name, num_samples)` - *Análisis de flujo de información con captura genérica de outputs*
- `visualize_topology_as_graph` (line 1559) `def visualize_topology_as_graph(model, run_name, threshold)` - *Grafo v18 con métricas*
- `analyze_topology_evolution` (line 1611) `def analyze_topology_evolution(run_name)` - *Análisis temporal completo v18*
- `comprehensive_topology_analysis` (line 1672) `def comprehensive_topology_analysis(model, dataloader, run_name)` - *Suite completa de análisis v18*
- `run_ablation_study` (line 1695) `def run_ablation_study()` - *Suite de ablación v18 completa*
- `visualize_memory_evolution` (line 1796) `def visualize_memory_evolution(model, epoch, run_name)` - *Visualiza evolución de memorias semánticas

Crítico para entender si la consolidación hipocampal→cortical está funcionando.
SVD spectrum revela estructura de representaciones aprendidas.*
- `analyze_gradient_flow` (line 1887) `def analyze_gradient_flow(model, epoch, run_name)` - *Análisis detallado del flujo de gradientes

Detecta vanishing/exploding gradients y capas muertas.
Critical para debugging de arquitecturas con fast weights.*
- `make_adversarial_pgd` (line 1954) `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name, controls, prev_states)` - *PGD attack con congelamiento total de pesos (Protocolo de Aislamiento Sináptico).
FIX: Captura correcta de 5 valores de retorno del forward.*
- `evaluate` (line 2021) `def evaluate(model, loader, config, adversarial, controls)` - *Evaluación con plasticidad residual (test-time adaptation)
Biológicamente plausible: el cerebro no se apaga durante percepción*
- `train_epoch` (line 2093) `def train_epoch(model, loader, optimizer, opt_topo, config, epoch, monitor, scaler, sparsity_lambda)` - *Entrenamiento homeostático con inicialización de estados y gestión de densidad*
- `train_model` (line 2279) `def train_model(config, run_name)` - *Training loop principal - FIX v24: Orden correcto de Scheduler y gestión de memoria.*
- `main` (line 2491) `def main()` - *CLI v24 completo con Orquestador Prefrontal*
- `__post_init__` (line 82) `def __post_init__(self)`
- `to_dict` (line 88) `def to_dict(self)`
- `get_supcon_lambda` (line 91) `def get_supcon_lambda(self, epoch)`
- `get_sparsity_lambda` (line 97) `def get_sparsity_lambda(self, epoch)`
- `get_memory_gb` (line 129) `def get_memory_gb()`
- `get_gpu_memory_gb` (line 134) `def get_gpu_memory_gb()`
- `log` (line 140) `def log(prefix)`
- `clear_cache` (line 147) `def clear_cache()`
- `check_limit` (line 153) `def check_limit(limit_gb, abort_on_limit)`
- `__init__` (line 179) `def __init__(self, config)`
- `forward` (line 216) `def forward(self, metrics_dict)` - *Orquestador v26: Allostasis (Adaptación Predictiva Valiente).

Cambio de Paradigma:
En lugar de entrar en pánico ciego cuando la densidad es baja (<5%),
este sistema evalúa el rendimiento (Loss). Si el cerebro es "delgado"
pero eficiente, se activa el 'Modo Élite' (Alta Plasticidad).
Solo se activa el protocolo de emergencia si hay colapso funcional.*
- `detach_state` (line 312) `def detach_state(self)` - *Rompe el grafo computacional para evitar retropropagación infinita entre batches*
- `reset_context` (line 317) `def reset_context(self)` - *Resetear contexto al inicio de cada época*
- `__init__` (line 340) `def __init__(self, model, config, epsilon_c)`
- `_analyze_matrix` (line 346) `def _analyze_matrix(self, weight_matrix, name)`
- `calculate` (line 392) `def calculate(self, epoch)`
- `get_critical_summary` (line 409) `def get_critical_summary(self)`
- `__init__` (line 421) `def __init__(self, run_name)`
- `save` (line 426) `def save(self, data, name)`
- `load` (line 454) `def load(self, name)`
- `__init__` (line 511) `def __init__(self, temperature)`
- `forward` (line 514) `def forward(self, features, labels)`
- `__init__` (line 528) `def __init__(self, dim, use_spectral)`
- `forward` (line 538) `def forward(self, input_signal, prediction)`
- `__init__` (line 548) `def __init__(self, dim, min_gate)`
- `forward` (line 558) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 566) `def __init__(self, dim, num_atoms)`
- `_maintain_orthogonality` (line 580) `def _maintain_orthogonality(self)`
- `forward` (line 585) `def forward(self, x)`
- `__init__` (line 611) `def __init__(self, input_dim, hidden_dim, fast_lr, forget_rate, use_spectral)`
- `forward` (line 660) `def forward(self, x, state_M, controls)` - *FIX: Ahora acepta señales de control del Orquestador para modular
la plasticidad y consolidación en tiempo real.*
- `__init__` (line 738) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- `invalidate_sparse_cache` (line 776) `def invalidate_sparse_cache(self)`
- `_validate_and_fix_state` (line 779) `def _validate_and_fix_state(self, state, expected_shape, batch_size, device, state_name)`
- `get_node_importance` (line 798) `def get_node_importance(self)` - *FIX: Método faltante para obtener importancia de nodos*
- `forward` (line 804) `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse, prev_state_node, prev_state_cell, controls)` - *Forward con señales de control del Orquestador y retorno de ortho deviation*
- `__init__` (line 935) `def __init__(self, config, in_channels)`
- `initialize_memories` (line 989) `def initialize_memories(self, dataloader)` - *Inicialización de memorias semánticas con captura correcta de 5 valores de retorno*
- `_initialize_layer_memory` (line 1032) `def _initialize_layer_memory(self, cell, x_input, name)`
- `consolidate_semantic_memories` (line 1058) `def consolidate_semantic_memories(self)`
- `set_epoch` (line 1103) `def set_epoch(self, epoch)`
- `calculate_ortho_loss` (line 1108) `def calculate_ortho_loss(self, ortho_deviation, controls)` - *Calcula loss de ortogonalidad usando el deviation retornado por las capas*
- `calculate_topology_diversity_loss` (line 1116) `def calculate_topology_diversity_loss(self, controls)`
- `_init_grid_topology` (line 1141) `def _init_grid_topology(self, N)`
- `get_topology` (line 1164) `def get_topology(self, return_sparse)`
- `forward` (line 1177) `def forward(self, x, prev_states, controls)` - *Forward con validación, detach explícito, y retorno de ortho deviation*
- `prune_topology` (line 1239) `def prune_topology(self, controls)` - *Poda topológica con protocolo de supervivencia garantizado y neurogénesis*
- `warmup_topo` (line 2327) `def warmup_topo(epoch)`

#### `neurologitos.py`
**Path:** `neurologitos.py`

**Classes:**
- `NeuroLogosConfig` (line 50) `class NeuroLogosConfig` - *Configuración CPU-friendly para NeuroLogos*
- `TopoBrainCore` (line 95) `class TopoBrainCore`
- `PGDAttack` (line 159) `class PGDAttack`
- `MiniUnconscious` (line 187) `class MiniUnconscious`
- `TopoUnconscious` (line 206) `class TopoUnconscious`
- `ConsciousCore` (line 229) `class ConsciousCore`
- `BioDecoder` (line 241) `class BioDecoder`
- `NeuroLogos` (line 284) `class NeuroLogos`
- `CIFARCaptions` (line 318) `class CIFARCaptions`
- `AblationMatrix` (line 356) `class AblationMatrix`
- `ScientificAnalyzer` (line 393) `class ScientificAnalyzer`

**Functions:**
- `seed_everything` (line 29) `def seed_everything(seed)` - *Control total de reproducibilidad*
- `compute_effect_size` (line 38) `def compute_effect_size(group1, group2)` - *Cohen's d con corrección de sesgo*
- `train_epoch_cv` (line 451) `def train_epoch_cv(model, loader, optimizer, config, epoch, vocab)`
- `evaluate_cv` (line 507) `def evaluate_cv(model, loader, config, vocab)`
- `train_with_cv` (line 524) `def train_with_cv(config, dataset, vocab)`
- `run_scientific_ablation` (line 579) `def run_scientific_ablation()`
- `to_dict` (line 81) `def to_dict(self)`
- `component_signature` (line 84) `def component_signature(self)`
- `__init__` (line 96) `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)`
- `_init_grid` (line 121) `def _init_grid(self)`
- `forward` (line 128) `def forward(self, x)`
- `get_metrics` (line 152) `def get_metrics(self)`
- `__init__` (line 160) `def __init__(self, epsilon, alpha, steps)`
- `attack` (line 165) `def attack(self, model_fn, x, y, criterion)`
- `__init__` (line 188) `def __init__(self, output_dim)`
- `forward` (line 200) `def forward(self, x)`
- `__init__` (line 207) `def __init__(self, output_dim, use_grid, use_symbiotic)`
- `forward` (line 221) `def forward(self, x)`
- `get_metrics` (line 225) `def get_metrics(self)`
- `__init__` (line 230) `def __init__(self, dim)`
- `forward` (line 235) `def forward(self, x)`
- `__init__` (line 242) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 253) `def forward(self, thought, captions, max_len)`
- `_get_init_state` (line 278) `def _get_init_state(self, thought)`
- `__init__` (line 285) `def __init__(self, vocab_size, config)`
- `forward` (line 309) `def forward(self, image, captions)`
- `get_metrics` (line 314) `def get_metrics(self)`
- `__init__` (line 319) `def __init__(self)`
- `__len__` (line 344) `def __len__(self)`
- `__getitem__` (line 347) `def __getitem__(self, idx)`
- `level1_isolated` (line 360) `def level1_isolated()`
- `level2_pairs` (line 369) `def level2_pairs()`
- `level3_full` (line 377) `def level3_full()`
- `level4_inverse` (line 381) `def level4_inverse()`
- `get_complete_matrix` (line 389) `def get_complete_matrix(cls)`
- `compute_statistics` (line 395) `def compute_statistics(cv_results)`
- `ttest_vs_baseline` (line 413) `def ttest_vs_baseline(exp_scores, baseline_scores)`
- `detect_synergy` (line 419) `def detect_synergy(pair_score, comp_a_score, comp_b_score, baseline_score)`
- `rank_criticality` (line 432) `def rank_criticality(full_score, ablation_results)`
- `model_fn` (line 471) `def model_fn(x_adv)`
- `crit_fn` (line 474) `def crit_fn(out, tgt)`

#### `neurologos.py`
**Path:** `neurologos.py`

**Classes:**
- `MiniUnconscious` (line 13) `class MiniUnconscious` - *Versión rápida CPU: 512-dim output directo*
- `NestedUnconscious` (line 29) `class NestedUnconscious` - *Versión topológica GPU: mantiene nested structure*
- `LiquidNeuron` (line 81) `class LiquidNeuron`
- `ConsciousCore` (line 96) `class ConsciousCore`
- `BioDecoder` (line 116) `class BioDecoder`
- `NeuroLogos` (line 167) `class NeuroLogos`
- `LifeCycle` (line 200) `class LifeCycle`
- `CIFARCaptions` (line 220) `class CIFARCaptions`

**Functions:**
- `train_logos` (line 261) `def train_logos(use_nested)`
- `__init__` (line 15) `def __init__(self)`
- `forward` (line 26) `def forward(self, x)`
- `__init__` (line 31) `def __init__(self, grid_size, output_dim)`
- `forward` (line 57) `def forward(self, x)`
- `__init__` (line 82) `def __init__(self, dim)`
- `forward` (line 88) `def forward(self, x, plasticity)`
- `__init__` (line 97) `def __init__(self)`
- `forward` (line 103) `def forward(self, visual_features, plasticity)`
- `__init__` (line 117) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 132) `def forward(self, thought, captions, max_len)`
- `_get_init_state` (line 159) `def _get_init_state(self, thought)`
- `__init__` (line 168) `def __init__(self, vocab_size, use_nested)`
- `forward` (line 182) `def forward(self, image, captions, plasticity)`
- `measure_richness` (line 193) `def measure_richness(self)`
- `__init__` (line 201) `def __init__(self, total_epochs)`
- `get_plasticity` (line 205) `def get_plasticity(self, epoch)`
- `__init__` (line 221) `def __init__(self)`
- `__len__` (line 245) `def __len__(self)`
- `__getitem__` (line 248) `def __getitem__(self, idx)`

#### `neurologos_V1.py`
**Path:** `neurologos_V1.py`

**Classes:**
- `MiniUnconscious` (line 15) `class MiniUnconscious`
- `LiquidNeuron` (line 33) `class LiquidNeuron`
- `ConsciousCore` (line 48) `class ConsciousCore`
- `BioDecoder` (line 73) `class BioDecoder`
- `NeuroLogos` (line 140) `class NeuroLogos`
- `LifeCycle` (line 170) `class LifeCycle`
- `CIFARCaptions` (line 189) `class CIFARCaptions`

**Functions:**
- `train_logos` (line 237) `def train_logos()`
- `__init__` (line 16) `def __init__(self)`
- `forward` (line 27) `def forward(self, x)`
- `__init__` (line 34) `def __init__(self, dim)`
- `forward` (line 40) `def forward(self, x, plasticity)`
- `__init__` (line 49) `def __init__(self)`
- `forward` (line 55) `def forward(self, visual_features, plasticity)`
- `__init__` (line 74) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 90) `def forward(self, thought, captions, max_len, teacher_forcing_ratio)` - *Modo entrenamiento: captions != None
Modo generación: captions == None*
- `_get_init_state` (line 131) `def _get_init_state(self, thought)`
- `__init__` (line 141) `def __init__(self, vocab_size)`
- `forward` (line 150) `def forward(self, image, captions, plasticity)`
- `measure_richness` (line 163) `def measure_richness(self)`
- `__init__` (line 171) `def __init__(self, total_epochs)`
- `get_plasticity` (line 175) `def get_plasticity(self, epoch)`
- `__init__` (line 190) `def __init__(self)`
- `__len__` (line 216) `def __len__(self)`
- `__getitem__` (line 219) `def __getitem__(self, idx)`

#### `neurologos_cpu_v7.py`
**Path:** `neurologos_cpu_v7.py`

**Classes:**
- `MicroConfig` (line 33) `class MicroConfig`
- `MicroContinuumCell` (line 96) `class MicroContinuumCell`
- `MicroSymbioticBasis` (line 120) `class MicroSymbioticBasis`
- `MicroTopology` (line 135) `class MicroTopology`
- `MicroSupConLoss` (line 154) `class MicroSupConLoss`
- `MicroTopoBrain` (line 170) `class MicroTopoBrain`

**Functions:**
- `setup_device` (line 65) `def setup_device()`
- `seed_everything` (line 68) `def seed_everything(seed)`
- `get_dataset` (line 73) `def get_dataset(config)`
- `micro_pgd_attack` (line 235) `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- `train_with_cv` (line 276) `def train_with_cv(config, dataset, cv_folds)`
- `run_ablation_study` (line 328) `def run_ablation_study()`
- `__init__` (line 97) `def __init__(self, dim)`
- `forward` (line 106) `def forward(self, x, plasticity)`
- `__init__` (line 121) `def __init__(self, dim, num_atoms)`
- `forward` (line 128) `def forward(self, x)`
- `__init__` (line 136) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 149) `def get_adjacency(self, plasticity)`
- `__init__` (line 155) `def __init__(self, temperature)`
- `forward` (line 158) `def forward(self, features, labels)`
- `__init__` (line 171) `def __init__(self, config)`
- `_init_weights` (line 195) `def _init_weights(self)`
- `count_parameters` (line 198) `def count_parameters(self)`
- `forward` (line 199) `def forward(self, x, plasticity)`

#### `neurologos_cpu_v8.py`
**Path:** `neurologos_cpu_v8.py`

**Classes:**
- `MicroConfig` (line 33) `class MicroConfig`
- `MicroContinuumCell` (line 92) `class MicroContinuumCell` - *Memoria continua con aprendizaje rápido/lento - VERSIÓN COMPATIBLE CON PGD*
- `MicroSymbioticBasis` (line 124) `class MicroSymbioticBasis` - *Base simbiótica para refinamiento adversarial*
- `MicroTopology` (line 146) `class MicroTopology` - *Topología de grid 2D con conexiones von Neumann*
- `MicroSupConLoss` (line 169) `class MicroSupConLoss` - *Supervised Contrastive Loss*
- `MicroTopoBrain` (line 195) `class MicroTopoBrain` - *Arquitectura TopoBrain Modular para Ablación*

**Functions:**
- `seed_everything` (line 65) `def seed_everything(seed)`
- `get_dataset` (line 72) `def get_dataset(config)`
- `micro_pgd_attack` (line 308) `def micro_pgd_attack(model, x, y, eps, steps, plasticity)` - *PGD Attack - Versión ultra-simple que siempre funciona*
- `generate_ablation_matrix` (line 344) `def generate_ablation_matrix()` - *Genera matriz de ablación de 3 niveles:
- Nivel 1: Baseline + componentes individuales
- Nivel 2: Pares sinérgicos
- Nivel 3: Sistema completo (para ablación inversa)*
- `train_with_cv` (line 393) `def train_with_cv(config, dataset, cv_folds)` - *Entrenamiento con cross-validation*
- `run_ablation_study` (line 483) `def run_ablation_study()` - *Ejecuta el estudio de ablación completo*
- `__init__` (line 94) `def __init__(self, dim)`
- `forward` (line 105) `def forward(self, x, plasticity)`
- `__init__` (line 126) `def __init__(self, dim, num_atoms)`
- `forward` (line 134) `def forward(self, x)`
- `__init__` (line 148) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 164) `def get_adjacency(self, plasticity)`
- `__init__` (line 171) `def __init__(self, temperature)`
- `forward` (line 176) `def forward(self, features, labels)`
- `__init__` (line 197) `def __init__(self, config)`
- `_init_weights` (line 240) `def _init_weights(self)`
- `count_parameters` (line 245) `def count_parameters(self)`
- `forward` (line 248) `def forward(self, x, plasticity)`

#### `neurologos_cpu_v9.py.py`
**Path:** `neurologos_cpu_v9.py.py`

**Classes:**
- `NeuroLogosConfig` (line 45) `class NeuroLogosConfig` - *Configuración ablacionable para NeuroLogos*
- `TopoBrainCore` (line 90) `class TopoBrainCore`
- `PGDAttack` (line 153) `class PGDAttack`
- `MiniUnconscious` (line 180) `class MiniUnconscious`
- `TopoUnconscious` (line 198) `class TopoUnconscious`
- `ConsciousCore` (line 220) `class ConsciousCore`
- `BioDecoder` (line 231) `class BioDecoder`
- `NeuroLogos` (line 276) `class NeuroLogos`
- `CIFARCaptions` (line 315) `class CIFARCaptions`
- `AblationMatrix` (line 355) `class AblationMatrix` - *Matriz de ablación para 3 componentes (G, S, A)*
- `ScientificAnalyzer` (line 392) `class ScientificAnalyzer`

**Functions:**
- `seed_everything` (line 25) `def seed_everything(seed)` - *Control total de reproducibilidad*
- `compute_effect_size` (line 34) `def compute_effect_size(group1, group2)` - *Cohen's d con corrección de sesgo*
- `train_epoch_cv` (line 452) `def train_epoch_cv(model, loader, optimizer, config, epoch, vocab)`
- `evaluate_cv` (line 507) `def evaluate_cv(model, loader, config, vocab)`
- `train_with_cv` (line 523) `def train_with_cv(config, dataset, vocab)`
- `run_scientific_ablation` (line 580) `def run_scientific_ablation()`
- `to_dict` (line 76) `def to_dict(self)`
- `component_signature` (line 79) `def component_signature(self)`
- `__init__` (line 91) `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)`
- `_init_grid` (line 116) `def _init_grid(self)`
- `forward` (line 123) `def forward(self, x)`
- `get_metrics` (line 147) `def get_metrics(self)`
- `__init__` (line 154) `def __init__(self, epsilon, alpha, steps)`
- `attack` (line 159) `def attack(self, model_fn, x, y, criterion)`
- `__init__` (line 181) `def __init__(self, output_dim)`
- `forward` (line 193) `def forward(self, x)`
- `__init__` (line 199) `def __init__(self, output_dim, use_grid, use_symbiotic)`
- `forward` (line 213) `def forward(self, x)`
- `get_metrics` (line 217) `def get_metrics(self)`
- `__init__` (line 221) `def __init__(self, dim)`
- `forward` (line 226) `def forward(self, x)`
- `__init__` (line 232) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 243) `def forward(self, thought, captions, max_len)`
- `_get_init_state` (line 268) `def _get_init_state(self, thought)`
- `__init__` (line 277) `def __init__(self, vocab_size, config)`
- `forward` (line 304) `def forward(self, image, captions)`
- `get_metrics` (line 309) `def get_metrics(self)`
- `__init__` (line 316) `def __init__(self)`
- `__len__` (line 341) `def __len__(self)`
- `__getitem__` (line 344) `def __getitem__(self, idx)`
- `level1_isolated` (line 360) `def level1_isolated()`
- `level2_pairs` (line 369) `def level2_pairs()`
- `level3_full` (line 377) `def level3_full()`
- `level4_inverse` (line 381) `def level4_inverse()`
- `get_complete_matrix` (line 389) `def get_complete_matrix(cls)`
- `compute_statistics` (line 394) `def compute_statistics(cv_results)`
- `ttest_vs_baseline` (line 412) `def ttest_vs_baseline(exp_scores, baseline_scores)`
- `detect_synergy` (line 418) `def detect_synergy(pair_score, comp_a_score, comp_b_score, baseline_score)`
- `rank_criticality` (line 431) `def rank_criticality(full_score, ablation_results)`
- `model_fn` (line 472) `def model_fn(x_adv)`
- `crit_fn` (line 475) `def crit_fn(out, tgt)`

#### `neurologos_entropico.py`
**Path:** `neurologos_entropico.py`

**Classes:**
- `Config` (line 36) `class Config`
- `DataEnvironment` (line 60) `class DataEnvironment`
- `HomeostaticRegulator` (line 94) `class HomeostaticRegulator`
- `PhysioNeuron` (line 120) `class PhysioNeuron`
- `RegulableSymbiotic` (line 160) `class RegulableSymbiotic`
- `RegulableTopology` (line 179) `class RegulableTopology`
- `MicroTopoBrain` (line 201) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 50) `def seed_everything(seed)`
- `train_nonstationary` (line 263) `def train_nonstationary(config)`
- `generate_ablation_matrix` (line 320) `def generate_ablation_matrix()`
- `run_ablation_study` (line 352) `def run_ablation_study()`
- `__init__` (line 61) `def __init__(self)`
- `get_batch` (line 71) `def get_batch(self, phase, bs)`
- `get_full` (line 85) `def get_full(self)`
- `get_w2` (line 88) `def get_w2(self)`
- `__init__` (line 95) `def __init__(self, d_in)`
- `forward` (line 105) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 121) `def __init__(self, d_in, d_out, dynamic)`
- `forward` (line 132) `def forward(self, x)`
- `__init__` (line 161) `def __init__(self, dim, atoms)`
- `forward` (line 168) `def forward(self, x, influence)`
- `__init__` (line 180) `def __init__(self, num_nodes)`
- `get_adjacency` (line 193) `def get_adjacency(self, plasticity)`
- `__init__` (line 202) `def __init__(self, config)`
- `count_parameters` (line 221) `def count_parameters(self)`
- `forward` (line 224) `def forward(self, x)`

#### `neurologos_fullhomesotatico_cpu_qw.py`
**Path:** `neurologos_fullhomesotatico_cpu_qw.py`

**Classes:**
- `MicroConfig` (line 36) `class MicroConfig`
- `GlobalHomeostaticOrchestrator` (line 93) `class GlobalHomeostaticOrchestrator`
- `MicroContinuumCell` (line 137) `class MicroContinuumCell`
- `MicroSymbioticBasis` (line 161) `class MicroSymbioticBasis`
- `MicroTopology` (line 183) `class MicroTopology`
- `MicroSupConLoss` (line 203) `class MicroSupConLoss`
- `MicroTopoBrain` (line 227) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 65) `def seed_everything(seed)`
- `get_dataset` (line 73) `def get_dataset(config)`
- `micro_pgd_attack` (line 363) `def micro_pgd_attack(model, x, y, eps, steps, plasticity_ctrl)`
- `generate_ablation_matrix` (line 387) `def generate_ablation_matrix()`
- `train_with_cv` (line 407) `def train_with_cv(config, dataset, cv_folds)`
- `run_ablation_study` (line 472) `def run_ablation_study()`
- `__init__` (line 94) `def __init__(self)`
- `forward` (line 104) `def forward(self, x, logits, h_agg, h_proc, w_norm, entropy, ortho)`
- `__init__` (line 138) `def __init__(self, dim)`
- `forward` (line 148) `def forward(self, x, strength)`
- `__init__` (line 162) `def __init__(self, dim, num_atoms)`
- `forward` (line 170) `def forward(self, x, influence)`
- `__init__` (line 184) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 197) `def get_adjacency(self, plasticity)`
- `__init__` (line 204) `def __init__(self, temperature)`
- `forward` (line 209) `def forward(self, features, labels)`
- `__init__` (line 228) `def __init__(self, config)`
- `_init_weights` (line 267) `def _init_weights(self)`
- `count_parameters` (line 272) `def count_parameters(self)`
- `forward` (line 275) `def forward(self, x)`

#### `neurologos_fullhomestatico_cpu_qw2.py`
**Path:** `neurologos_fullhomestatico_cpu_qw2.py`

**Classes:**
- `Config` (line 34) `class Config`
- `DataEnvironment` (line 60) `class DataEnvironment`
- `HomeostaticRegulator` (line 95) `class HomeostaticRegulator`
- `PhysioNeuron` (line 122) `class PhysioNeuron`
- `RegulableSymbiotic` (line 164) `class RegulableSymbiotic`
- `RegulableTopology` (line 184) `class RegulableTopology`
- `MicroTopoBrain` (line 207) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 49) `def seed_everything(seed)`
- `train_nonstationary` (line 269) `def train_nonstationary(config)`
- `generate_ablation_matrix` (line 331) `def generate_ablation_matrix()`
- `run_ablation_study` (line 364) `def run_ablation_study()`
- `__init__` (line 61) `def __init__(self)`
- `get_batch` (line 71) `def get_batch(self, phase, bs)`
- `get_full` (line 85) `def get_full(self)`
- `get_w2` (line 88) `def get_w2(self)`
- `__init__` (line 96) `def __init__(self, d_in)`
- `forward` (line 106) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 123) `def __init__(self, d_in, d_out, dynamic)`
- `forward` (line 134) `def forward(self, x)`
- `__init__` (line 165) `def __init__(self, dim, atoms)`
- `forward` (line 172) `def forward(self, x, influence)`
- `__init__` (line 185) `def __init__(self, num_nodes)`
- `get_adjacency` (line 198) `def get_adjacency(self, plasticity)`
- `__init__` (line 208) `def __init__(self, config)`
- `count_parameters` (line 227) `def count_parameters(self)`
- `forward` (line 230) `def forward(self, x)`

#### `neurologos_gpu_v1.py`
**Path:** `neurologos_gpu_v1.py`

**Classes:**
- `NestedUnconscious` (line 16) `class NestedUnconscious`
- `LiquidNeuron` (line 76) `class LiquidNeuron`
- `ConsciousCore` (line 91) `class ConsciousCore`
- `BioDecoder` (line 116) `class BioDecoder`
- `NeuroLogos` (line 183) `class NeuroLogos`
- `LifeCycle` (line 213) `class LifeCycle`
- `CIFARCaptions` (line 233) `class CIFARCaptions`

**Functions:**
- `train_logos` (line 282) `def train_logos()`
- `__init__` (line 17) `def __init__(self, grid_size, hidden_dim)`
- `forward` (line 46) `def forward(self, x)`
- `__init__` (line 77) `def __init__(self, dim)`
- `forward` (line 83) `def forward(self, x, plasticity)`
- `__init__` (line 92) `def __init__(self)`
- `forward` (line 98) `def forward(self, visual_features, plasticity)`
- `__init__` (line 117) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 133) `def forward(self, thought, captions, max_len, teacher_forcing_ratio)` - *Modo entrenamiento: captions != None
Modo generación: captions == None*
- `_get_init_state` (line 174) `def _get_init_state(self, thought)`
- `__init__` (line 184) `def __init__(self, vocab_size)`
- `forward` (line 193) `def forward(self, image, captions, plasticity)`
- `measure_richness` (line 206) `def measure_richness(self)`
- `__init__` (line 214) `def __init__(self, total_epochs)`
- `get_plasticity` (line 219) `def get_plasticity(self, epoch)`
- `__init__` (line 234) `def __init__(self)`
- `__len__` (line 261) `def __len__(self)`
- `__getitem__` (line 264) `def __getitem__(self, idx)`

#### `neurologos_homeostatico_cpu_cl.py`
**Path:** `neurologos_homeostatico_cpu_cl.py`

**Classes:**
- `HomeoConfig` (line 38) `class HomeoConfig`
- `HomeostaticRegulator` (line 98) `class HomeostaticRegulator` - *Sistema de auto-regulación inspirado en fisiología MEJORADO.

INNOVACIÓN: Sensores multi-escala que distinguen:
- Estrés natural (varianza, complejidad)
- Estrés adversarial (gradiente anómalo, suavidad)
- Fatiga metabólica (norma de pesos)

Inputs enriquecidos: [Estrés_Natural, Estrés_Adversarial, Excitación, Fatiga, Gradiente_Norma]
Outputs: [Metabolismo, Sensibilidad, Gate]*
- `HomeoContinuumCell` (line 218) `class HomeoContinuumCell` - *Memoria continua con regulación homeostática*
- `HomeoSymbioticBasis` (line 281) `class HomeoSymbioticBasis` - *Base simbiótica con regulación homeostática*
- `HomeoTopology` (line 329) `class HomeoTopology` - *Topología con plasticidad homeostática*
- `HomeoSupConLoss` (line 380) `class HomeoSupConLoss` - *Supervised Contrastive Loss*
- `HomeoTopoBrain` (line 410) `class HomeoTopoBrain` - *TopoBrain con regulación homeostática integrada*

**Functions:**
- `seed_everything` (line 71) `def seed_everything(seed)`
- `get_dataset` (line 78) `def get_dataset(config)`
- `pgd_attack` (line 524) `def pgd_attack(model, x, y, eps, steps, plasticity)` - *PGD Attack simplificado*
- `generate_homeostatic_ablation` (line 555) `def generate_homeostatic_ablation()` - *Genera matriz enfocada en homeostasis con sensores mejorados*
- `train_with_cv` (line 614) `def train_with_cv(config, dataset, cv_folds)` - *Entrenamiento con cross-validation*
- `run_homeostatic_ablation` (line 697) `def run_homeostatic_ablation()` - *Ejecuta el estudio de ablación homeostático*
- `__init__` (line 110) `def __init__(self, d_in)`
- `forward` (line 129) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 220) `def __init__(self, dim, use_homeostasis)`
- `forward` (line 241) `def forward(self, x, plasticity)`
- `__init__` (line 283) `def __init__(self, dim, num_atoms, use_homeostasis)`
- `forward` (line 299) `def forward(self, x)`
- `__init__` (line 331) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 354) `def get_adjacency(self, x, plasticity)` - *Genera adyacencia con regulación homeostática opcional*
- `__init__` (line 382) `def __init__(self, temperature)`
- `forward` (line 387) `def forward(self, features, labels)`
- `__init__` (line 412) `def __init__(self, config)`
- `_init_weights` (line 456) `def _init_weights(self)`
- `count_parameters` (line 461) `def count_parameters(self)`
- `forward` (line 464) `def forward(self, x, plasticity)`

#### `neurologos_homeostatico_cpu_cl2.py`
**Path:** `neurologos_homeostatico_cpu_cl2.py`

**Classes:**
- `TransContextConfig` (line 50) `class TransContextConfig`
- `NonStationaryEnvironment` (line 89) `class NonStationaryEnvironment` - *Entorno que cambia de distribución como en PhysioChimera.
Permite medir RETENCIÓN y ADAPTACIÓN, no solo robustez adversarial.*
- `EnhancedHomeostaticRegulator` (line 152) `class EnhancedHomeostaticRegulator` - *Regulador homeostático con logging y sensores mejorados.
Incluye diagnóstico para análisis post-hoc.*
- `TransContextContinuumCell` (line 254) `class TransContextContinuumCell` - *Memoria continua con homeostasis*
- `TransContextTopoBrain` (line 320) `class TransContextTopoBrain` - *TopoBrain diseñado para entornos no estacionarios.
Focus: Retención y Adaptación, no solo robustez adversarial.*

**Functions:**
- `seed_everything` (line 78) `def seed_everything(seed)`
- `light_pgd_attack` (line 424) `def light_pgd_attack(model, x, y, eps, steps)` - *PGD ligero para no dominar el entrenamiento*
- `train_trans_contextual` (line 455) `def train_trans_contextual(config, name)` - *Entrenamiento en entorno no estacionario.
Mide: Retención, Adaptación, Robustez.*
- `generate_selective_ablation` (line 556) `def generate_selective_ablation()` - *Ablación selectiva basada en resultados v5.2:
- Eliminar MGF (nunca mejora con homeostasis)
- Focus en Continuum + Homeostasis
- Agregar regulación jerárquica*
- `run_trans_contextual_study` (line 598) `def run_trans_contextual_study()` - *Ejecuta el estudio trans-contextual completo*
- `__init__` (line 94) `def __init__(self)`
- `get_batch` (line 115) `def get_batch(self, phase, batch_size)` - *Retorna batch según la fase del entrenamiento*
- `get_phase` (line 135) `def get_phase(self, epoch, total_epochs)` - *Determina la fase según el epoch actual*
- `__init__` (line 157) `def __init__(self, d_in, log_metrics)`
- `forward` (line 183) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 256) `def __init__(self, dim, use_homeostasis, log_metrics)`
- `forward` (line 277) `def forward(self, x, plasticity)`
- `__init__` (line 325) `def __init__(self, config)`
- `_init_weights` (line 356) `def _init_weights(self)`
- `count_parameters` (line 361) `def count_parameters(self)`
- `forward` (line 364) `def forward(self, x, plasticity)`
- `get_homeostasis_metrics` (line 401) `def get_homeostasis_metrics(self)` - *Extrae métricas de homeostasis para logging*

#### `neurologos_homeostatico_cpu_ki.py`
**Path:** `neurologos_homeostatico_cpu_ki.py`

**Classes:**
- `MicroConfig` (line 37) `class MicroConfig`
- `HomeostaticCore` (line 109) `class HomeostaticCore` - *Cerebro interno que monitoriza el estado fisiológico de la red
y emite señales de control adaptativas.
FIXES:
1. Maneja batch_size=1 evitando warning de var()
2. Convierte loss_val correctamente a tensor
3. Asegura device placement consistente*
- `MicroContinuumCell` (line 175) `class MicroContinuumCell` - *Versión homeostática con regulación de metabolismo*
- `MicroSymbioticBasis` (line 240) `class MicroSymbioticBasis` - *Base simbiótica - VERSION FINAL
FIXES:
1. Remover batch norm que causaba colapso
2. Aumentar ruido interno para no ser demasiado robusto
3. Añadir regularización de varianza mínima
4. Regulación de entropía más fuerte para evitar picos*
- `MicroTopology` (line 327) `class MicroTopology` - *Versión homeostática con plasticidad adaptativa*
- `MicroSupConLoss` (line 367) `class MicroSupConLoss` - *Supervised Contrastive Loss - Mantiene estabilidad con homeostasis*
- `MicroTopoBrain` (line 397) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 74) `def seed_everything(seed)`
- `get_dataset` (line 85) `def get_dataset(config)` - *Genera dataset sintético para el estudio de ablación.
Normaliza features en rango [0,1] para estabilidad homeostática.*
- `micro_pgd_attack` (line 521) `def micro_pgd_attack(model, x, y, eps, steps, loss_val)` - *PGD Attack - Versión ultra-simple que siempre funciona
FIX: No pasar loss_val al modelo durante ataque para evitar leakage
El ataque no debe tener acceso al estado interno de entrenamiento*
- `generate_ablation_matrix` (line 556) `def generate_ablation_matrix()` - *Genera matriz de ablación de 3 niveles con configuración aislada por experimento
FIX: Asegurar que cada experimento tenga configuración independiente y limpia*
- `train_with_cv` (line 595) `def train_with_cv(config, dataset, cv_folds)` - *Entrenamiento con cross-validation
FIX CRÍTICO: Métrica W2 debe usar MODELO FRESH copiado, no el mismo modelo*
- `run_ablation_study` (line 777) `def run_ablation_study()` - *Ejecuta estudio con validación de integridad de resultados*
- `__init__` (line 118) `def __init__(self, d_in, base_lr)`
- `forward` (line 137) `def forward(self, x, h_pre, w_norm, loss_val)`
- `__init__` (line 177) `def __init__(self, dim, config)`
- `forward` (line 198) `def forward(self, x, plasticity)`
- `__init__` (line 249) `def __init__(self, dim, config)`
- `forward` (line 268) `def forward(self, x, loss_val)`
- `__init__` (line 329) `def __init__(self, num_nodes, config)`
- `_create_grid_mask` (line 344) `def _create_grid_mask(self)`
- `get_adjacency` (line 355) `def get_adjacency(self, plasticity, loss_val)`
- `__init__` (line 369) `def __init__(self, temperature)`
- `forward` (line 374) `def forward(self, features, labels)`
- `__init__` (line 398) `def __init__(self, config)`
- `_init_weights` (line 440) `def _init_weights(self)`
- `count_parameters` (line 445) `def count_parameters(self)`
- `forward` (line 448) `def forward(self, x, loss_val)`

#### `neurologos_homestotico_cpu_qw.py`
**Path:** `neurologos_homestotico_cpu_qw.py`

**Classes:**
- `MicroConfig` (line 35) `class MicroConfig`
- `HomeostaticRegulatorMini` (line 92) `class HomeostaticRegulatorMini`
- `MicroPhysioNeuron` (line 117) `class MicroPhysioNeuron`
- `MicroContinuumCell` (line 162) `class MicroContinuumCell`
- `MicroSymbioticBasis` (line 187) `class MicroSymbioticBasis`
- `MicroTopology` (line 207) `class MicroTopology`
- `MicroSupConLoss` (line 228) `class MicroSupConLoss`
- `MicroTopoBrain` (line 252) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 64) `def seed_everything(seed)`
- `get_dataset` (line 72) `def get_dataset(config)`
- `micro_pgd_attack` (line 365) `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- `generate_ablation_matrix` (line 389) `def generate_ablation_matrix()`
- `train_with_cv` (line 429) `def train_with_cv(config, dataset, cv_folds)`
- `run_ablation_study` (line 496) `def run_ablation_study()`
- `__init__` (line 93) `def __init__(self, d_in)`
- `forward` (line 104) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 118) `def __init__(self, d_in, d_out, dynamic)`
- `forward` (line 129) `def forward(self, x)`
- `__init__` (line 163) `def __init__(self, dim)`
- `forward` (line 173) `def forward(self, x, plasticity)`
- `__init__` (line 188) `def __init__(self, dim, num_atoms)`
- `forward` (line 196) `def forward(self, x)`
- `__init__` (line 208) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 222) `def get_adjacency(self, plasticity)`
- `__init__` (line 229) `def __init__(self, temperature)`
- `forward` (line 234) `def forward(self, features, labels)`
- `__init__` (line 253) `def __init__(self, config)`
- `_init_weights` (line 301) `def _init_weights(self)`
- `count_parameters` (line 306) `def count_parameters(self)`
- `forward` (line 309) `def forward(self, x, plasticity)`

#### `neurologos_tricameral_exodia.py`
**Path:** `neurologos_tricameral_exodia.py`

**Classes:**
- `HierarchicalEpisodicMemory` (line 332) `class HierarchicalEpisodicMemory`
- `NeurocognitiveSystem` (line 563) `class NeurocognitiveSystem`
- `LanguageMetrics` (line 760) `class LanguageMetrics` - *Métricas de calidad de generación*
- `LinguisticFeedbackLoop` (line 834) `class LinguisticFeedbackLoop`
- `LanguageMetrics` (line 951) `class LanguageMetrics`
- `CausalReasoningEngine` (line 994) `class CausalReasoningEngine`
- `LanguageMetrics` (line 1073) `class LanguageMetrics`
- `StableLiquidNeuron` (line 1120) `class StableLiquidNeuron`
- `TriangulatedMedicalSystem` (line 1259) `class TriangulatedMedicalSystem`
- `LeftHemisphere` (line 1410) `class LeftHemisphere`
- `AudioEncoder` (line 1719) `class AudioEncoder` - *Encoder de audio usando Conv + Transformer*
- `RightHemisphereTricameral` (line 1769) `class RightHemisphereTricameral` - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 1853) `class CorpusCallosumTrimodal`
- `EnhancedDiagnosticsTricameral` (line 2006) `class EnhancedDiagnosticsTricameral`
- `NeuroLogosTricameral` (line 2311) `class NeuroLogosTricameral` - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 2346) `class Flickr8kMultimodalDataset(Dataset)` - *Dataset que carga imagen, audio del caption y texto desde Kaggle*

**Functions:**
- `preprocess_and_cache_spectrograms` (line 47) `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)` - *Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt
Esto elimina el cuello de botella de I/O durante entrenamiento*
- `setup_flickr8k_with_audio` (line 120) `def setup_flickr8k_with_audio(data_dir)` - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 308) `def build_vocab_flickr(captions_file, vocab_size)` - *Construye vocabulario desde el archivo de captions*
- `compute_alignment_loss` (line 2462) `def compute_alignment_loss(visual_features, channels, alpha, epoch)` - *FIX: Pérdida auxiliar para alineación temprana de canales multimodales
Solo activa en épocas iniciales (epoch < 6)*
- `compute_tricameral_loss` (line 2490) `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, channels, epoch, lambda_reward, lambda_mtp)` - *FIX: Pérdida con término explícito de coherencia multimodal
Penaliza la falta de sincronización entre canales*
- `train_tricameral` (line 2563) `def train_tricameral()`
- `__init__` (line 333) `def __init__(self, working_capacity, short_term_capacity, importance_threshold)`
- `compute_surprise` (line 359) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `calculate_importance` (line 369) `def calculate_importance(self, episode, surprise_score)`
- `_calculate_novelty` (line 381) `def _calculate_novelty(self, episode)`
- `store_episode` (line 402) `def store_episode(self, image, audio, caption, surprise_score)`
- `_update_unified_buffer` (line 440) `def _update_unified_buffer(self)`
- `add` (line 452) `def add(self, image, audio, caption, surprise_score)`
- `apply_forgetting_curve` (line 455) `def apply_forgetting_curve(self)`
- `_purge_low_score_memories` (line 471) `def _purge_low_score_memories(self)`
- `sample` (line 497) `def sample(self, batch_size, memory_level)`
- `_sample_from_buffer` (line 527) `def _sample_from_buffer(self, buffer, scores, batch_size)`
- `get_total_size` (line 555) `def get_total_size(self)`
- `__init__` (line 564) `def __init__(self)`
- `assess_reasoning_state` (line 584) `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` - *Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)*
- `assess_cognitive_state` (line 628) `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` - *Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)*
- `apply_cognitive_intervention` (line 674) `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` - *Aplica intervenciones basadas en estado lingüístico y de razonamiento*
- `sentence_bleu` (line 764) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 798) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 807) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 820) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 835) `def __init__(self, alpha, beta)`
- `_get_ngrams_cached` (line 849) `def _get_ngrams_cached(sentence, n)` - *FIX: Método estático con lru_cache para n-gramas*
- `compute_linguistic_reward` (line 858) `def compute_linguistic_reward(self, references, hypotheses)`
- `compute_cider` (line 897) `def compute_cider(self, reference, hypothesis)` - *FIX: Uso correcto del cache estático*
- `compute_spice` (line 911) `def compute_spice(self, reference, hypothesis)`
- `get_cache_stats` (line 923) `def get_cache_stats(self)` - *FIX: Estadísticas de cache actualizadas*
- `sentence_bleu` (line 953) `def sentence_bleu(reference, hypothesis, weights)`
- `token_accuracy` (line 976) `def token_accuracy(reference, hypothesis)`
- `word_overlap` (line 986) `def word_overlap(reference, hypothesis)`
- `__init__` (line 995) `def __init__(self, hidden_dim)`
- `reason_causally` (line 1022) `def reason_causally(self, observation, context)`
- `_predict_interventions` (line 1036) `def _predict_interventions(self, hypothesis, confidence)`
- `update_knowledge_graph` (line 1053) `def update_knowledge_graph(self, cause, effect, strength)`
- `query_causal_chain` (line 1059) `def query_causal_chain(self, start_node, end_node)`
- `sentence_bleu` (line 1075) `def sentence_bleu(reference, hypothesis, weights)`
- `token_accuracy` (line 1098) `def token_accuracy(reference, hypothesis)`
- `word_overlap` (line 1108) `def word_overlap(reference, hypothesis)`
- `__init__` (line 1121) `def __init__(self, in_dim, out_dim)`
- `forward` (line 1163) `def forward(self, x)`
- `_calculate_homeostasis_metric` (line 1179) `def _calculate_homeostasis_metric(self, output)` - *Calcula métrica de homeostasis basada en la estabilidad del output*
- `hebbian_update` (line 1188) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 1226) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 1260) `def __init__(self)`
- `triangulate_signals` (line 1267) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `count_convergent_signals` (line 1278) `def count_convergent_signals(self, signals, pattern)`
- `diagnose_with_triangulation` (line 1281) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)`
- `apply_triangulated_intervention` (line 1326) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- `_reset_liquid_neuron` (line 1395) `def _reset_liquid_neuron(self, liquid_neuron)` - *Reset completo de una neurona líquida*
- `__init__` (line 1411) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 1493) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_apply_chain_of_thought` (line 1540) `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- `_greedy_decode` (line 1580) `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- `_apply_multi_token_prediction` (line 1641) `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- `_apply_structural_attention` (line 1683) `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- `_get_init_state` (line 1704) `def _get_init_state(self, visual_context)`
- `__init__` (line 1722) `def __init__(self, output_dim)`
- `forward` (line 1756) `def forward(self, mel_spec)`
- `__init__` (line 1772) `def __init__(self, output_dim)`
- `forward` (line 1812) `def forward(self, image, audio)` - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre, audio_post, audio_pre: Para Hebbian*
- `__init__` (line 1854) `def __init__(self, dim)`
- `forward` (line 1902) `def forward(self, right_features)`
- `update_channel_fatigue` (line 1963) `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- `adjust_gates_by_fatigue` (line 1985) `def adjust_gates_by_fatigue(self)`
- `__init__` (line 2007) `def __init__(self)`
- `_get_cached_norm` (line 2030) `def _get_cached_norm(self, tensor, dim)` - *Cache de normalización con limpieza periódica*
- `measure_callosal_flow` (line 2048) `def measure_callosal_flow(self, right_features, left_context, channels)` - *FIX: Medición de coherencia multimodal real con atención a diversidad
Incluye métricas de sincronización entre canales*
- `__init__` (line 2314) `def __init__(self, vocab_size)`
- `forward` (line 2320) `def forward(self, image, audio, captions, epoch)`
- `__init__` (line 2349) `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache, cache_dir)`
- `__len__` (line 2406) `def __len__(self)`
- `__getitem__` (line 2409) `def __getitem__(self, idx)`
- `evaluate_reasoning_quality` (line 2101) `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- `calculate_synergy` (line 2138) `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 2149) `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 2158) `def update(self)`
- `get_recent_avg` (line 2175) `def get_recent_avg(self, key, n)`
- `visualize_fatigue_distribution` (line 2191) `def visualize_fatigue_distribution(self, epoch)`
- `visualize_reasoning_metrics` (line 2215) `def visualize_reasoning_metrics(self, epoch)`
- `report` (line 2227) `def report(self, epoch)`

#### `neurologos_v3.py`
**Path:** `neurologos_v3.py`

*No symbols extracted*

#### `neurologos_v4.py`
**Path:** `neurologos_v4.py`

**Classes:**
- `MiniUnconscious` (line 13) `class MiniUnconscious` - *Versión rápida CPU: 512-dim output directo*
- `NestedUnconscious` (line 29) `class NestedUnconscious` - *Versión topológica GPU: mantiene nested structure*
- `LiquidNeuron` (line 81) `class LiquidNeuron`
- `ConsciousCore` (line 164) `class ConsciousCore`
- `BioDecoder` (line 185) `class BioDecoder`
- `NeuroLogos` (line 236) `class NeuroLogos`
- `LifeCycle` (line 271) `class LifeCycle`
- `CIFARCaptions` (line 291) `class CIFARCaptions`

**Functions:**
- `train_logos` (line 358) `def train_logos(use_nested)`
- `__init__` (line 15) `def __init__(self)`
- `forward` (line 26) `def forward(self, x)`
- `__init__` (line 31) `def __init__(self, grid_size, output_dim)`
- `forward` (line 57) `def forward(self, x)`
- `__init__` (line 82) `def __init__(self, in_dim, out_dim)`
- `forward` (line 103) `def forward(self, x, global_plasticity)`
- `consolidate_svd` (line 131) `def consolidate_svd(self, repair_strength)` - *Consolidación espectral de pesos rápidos mediante SVD.
Repara inestabilidades numéricas y cristaliza conocimiento consolidado.
Retorna True si se realizó una consolidación activa.*
- `__init__` (line 165) `def __init__(self)`
- `forward` (line 172) `def forward(self, visual_features, plasticity)`
- `__init__` (line 186) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 201) `def forward(self, thought, captions, max_len)`
- `_get_init_state` (line 228) `def _get_init_state(self, thought)`
- `__init__` (line 237) `def __init__(self, vocab_size, use_nested)`
- `forward` (line 251) `def forward(self, image, captions, plasticity)`
- `measure_richness` (line 264) `def measure_richness(self)`
- `__init__` (line 272) `def __init__(self, total_epochs)`
- `get_plasticity` (line 276) `def get_plasticity(self, epoch)`
- `__init__` (line 292) `def __init__(self)`
- `__len__` (line 344) `def __len__(self)`
- `__getitem__` (line 346) `def __getitem__(self, idx)`

#### `neurologos_v5.py`
**Path:** `neurologos_v5.py`

**Classes:**
- `TopologicalCompressor` (line 75) `class TopologicalCompressor`
- `MiniUnconscious` (line 96) `class MiniUnconscious` - *Versión rápida CPU: 512-dim output directo - con procesamiento jerárquico inspirado en vía ventral*
- `NestedUnconscious` (line 119) `class NestedUnconscious`
- `LiquidNeuron` (line 169) `class LiquidNeuron`
- `ConsciousCore` (line 260) `class ConsciousCore`
- `BioDecoder` (line 332) `class BioDecoder`
- `NeuroLogos` (line 404) `class NeuroLogos`
- `LifeCycle` (line 435) `class LifeCycle`
- `CIFARCaptions` (line 457) `class CIFARCaptions`

**Functions:**
- `top_k_top_p_filtering` (line 11) `def top_k_top_p_filtering(logits, top_k, top_p, filter_value)` - *Filtra logits con Top-K o Top-P (Nucleus) Sampling.*
- `measure_spatial_richness` (line 32) `def measure_spatial_richness(activations)` - *Calcula la riqueza representacional de un tensor de activación.
Utiliza Entropía de Shannon (por canal) y Entropía de Von Neumann (estructural).
FIX: Evita el UserWarning de std() con batch=1 y mejora estabilidad numérica.*
- `train_logos` (line 502) `def train_logos(use_nested)`
- `__init__` (line 76) `def __init__(self, node_dim)`
- `forward` (line 85) `def forward(self, nodes, plasticity, transfer_rate)`
- `__init__` (line 98) `def __init__(self)`
- `forward` (line 113) `def forward(self, x)`
- `__init__` (line 120) `def __init__(self, grid_size, output_dim)`
- `forward` (line 143) `def forward(self, x)`
- `__init__` (line 170) `def __init__(self, in_dim, out_dim)`
- `forward` (line 191) `def forward(self, x, global_plasticity, transfer_rate)`
- `consolidate_svd` (line 228) `def consolidate_svd(self, repair_strength, timescale)`
- `__init__` (line 261) `def __init__(self)`
- `forward` (line 278) `def forward(self, visual_features, plasticity, transfer_rate)`
- `get_liquid_module` (line 319) `def get_liquid_module(self)` - *Retorna el LiquidNeuron activo para la consolidación externa.*
- `__init__` (line 334) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 349) `def forward(self, thought, captions, max_len)`
- `_get_init_state` (line 395) `def _get_init_state(self, thought)`
- `__init__` (line 405) `def __init__(self, vocab_size, use_nested)`
- `forward` (line 420) `def forward(self, image, captions, plasticity, transfer_rate)`
- `measure_richness` (line 428) `def measure_richness(self)`
- `__init__` (line 436) `def __init__(self, total_epochs)`
- `get_plasticity` (line 440) `def get_plasticity(self, epoch)`
- `__init__` (line 458) `def __init__(self)`
- `__len__` (line 483) `def __len__(self)`
- `__getitem__` (line 486) `def __getitem__(self, idx)`

#### `neurologos_v6.py`
**Path:** `neurologos_v6.py`

**Classes:**
- `Config` (line 18) `class Config`
- `MetricsCollector` (line 55) `class MetricsCollector`
- `DataEnvironment` (line 90) `class DataEnvironment`
- `MetaLearner` (line 127) `class MetaLearner`
- `ComponentRegulator` (line 154) `class ComponentRegulator`
- `HomeostaticRegulator` (line 211) `class HomeostaticRegulator`
- `MetaHomeostaticEngine` (line 237) `class MetaHomeostaticEngine`
- `PhysioNeuron` (line 320) `class PhysioNeuron`
- `SymbioticDual` (line 376) `class SymbioticDual`
- `MicroTopoBrain` (line 403) `class MicroTopoBrain`
- `ConfigurableTrainer` (line 473) `class ConfigurableTrainer`

**Functions:**
- `seed_everything` (line 45) `def seed_everything(seed)`
- `generate_ablation_matrix_4levels` (line 603) `def generate_ablation_matrix_4levels()`
- `run_ablation_study` (line 645) `def run_ablation_study()`
- `__init__` (line 56) `def __init__(self, config)`
- `_setup_logger` (line 62) `def _setup_logger(self)`
- `log_batch` (line 69) `def log_batch(self, step, metrics)`
- `save` (line 79) `def save(self, path)`
- `__init__` (line 91) `def __init__(self)`
- `inject_concept_drift` (line 100) `def inject_concept_drift(self)`
- `get_batch` (line 103) `def get_batch(self, phase, bs, step)`
- `get_full` (line 118) `def get_full(self)`
- `get_w2` (line 121) `def get_w2(self)`
- `__init__` (line 128) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 136) `def forward(self, sequence)`
- `update` (line 142) `def update(self, loss_pred, loss_real)`
- `__init__` (line 155) `def __init__(self, name, state_dim, cross_dim)`
- `forward` (line 172) `def forward(self)`
- `__init__` (line 212) `def __init__(self, d_in)`
- `forward` (line 222) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 238) `def __init__(self, config)`
- `forward` (line 251) `def forward(self, global_loss, step)`
- `get_component_health` (line 290) `def get_component_health(self)`
- `update_with_momentum` (line 298) `def update_with_momentum(self, current_lr, current_plasticity, meta_out, surprise_rate)`
- `__init__` (line 321) `def __init__(self, d_in, d_out, config)`
- `forward` (line 334) `def forward(self, x, surprise_threshold)`
- `__init__` (line 377) `def __init__(self, dim, atoms)`
- `forward` (line 385) `def forward(self, x, influence)`
- `__init__` (line 404) `def __init__(self, config)`
- `count_parameters` (line 427) `def count_parameters(self)`
- `forward` (line 430) `def forward(self, x, y, step)`
- `__init__` (line 474) `def __init__(self, config)`
- `train` (line 479) `def train(self, model)`
- `evaluate` (line 554) `def evaluate(self, model)`

#### `neurologosv5.2.py`
**Path:** `neurologosv5.2.py`

**Classes:**
- `MicroConfig` (line 33) `class MicroConfig`
- `HomeostaticRegulator` (line 85) `class HomeostaticRegulator`
- `PhysioNeuron` (line 109) `class PhysioNeuron`
- `MicroContinuumCell` (line 151) `class MicroContinuumCell`
- `MicroSymbioticBasis` (line 176) `class MicroSymbioticBasis`
- `MicroTopology` (line 196) `class MicroTopology`
- `MicroSupConLoss` (line 216) `class MicroSupConLoss`
- `MicroTopoBrain` (line 240) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 57) `def seed_everything(seed)`
- `get_dataset` (line 65) `def get_dataset(config)`
- `micro_pgd_attack` (line 340) `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- `generate_ablation_matrix` (line 364) `def generate_ablation_matrix()`
- `train_with_cv` (line 393) `def train_with_cv(config, dataset, cv_folds)`
- `run_ablation_study` (line 457) `def run_ablation_study()`
- `__init__` (line 86) `def __init__(self, d_in)`
- `forward` (line 96) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 110) `def __init__(self, d_in, d_out, dynamic_mode)`
- `forward` (line 121) `def forward(self, x)`
- `__init__` (line 152) `def __init__(self, dim)`
- `forward` (line 162) `def forward(self, x, plasticity)`
- `__init__` (line 177) `def __init__(self, dim, num_atoms)`
- `forward` (line 185) `def forward(self, x)`
- `__init__` (line 197) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 210) `def get_adjacency(self, plasticity)`
- `__init__` (line 217) `def __init__(self, temperature)`
- `forward` (line 222) `def forward(self, features, labels)`
- `__init__` (line 241) `def __init__(self, config)`
- `_init_weights` (line 278) `def _init_weights(self)`
- `count_parameters` (line 283) `def count_parameters(self)`
- `forward` (line 286) `def forward(self, x, plasticity)`

#### `neurosoberano.py`
**Path:** `neurosoberano.py`

**Classes:**
- `ExperimentConfig` (line 28) `class ExperimentConfig`
- `BasicBlock` (line 39) `class BasicBlock`
- `WideResNetBaseline` (line 60) `class WideResNetBaseline` - *Wide-ResNet simplificado (depth=16, width=2)
Parámetros similares a tu modelo (~1-2M)*
- `FastLiquidNeuron` (line 96) `class FastLiquidNeuron` - *Neurona líquida simplificada (sin SVD consolidation para POC)*
- `MinimalNeuroSovereign` (line 125) `class MinimalNeuroSovereign` - *Versión mínima de tu arquitectura para POC*
- `Experiment` (line 233) `class Experiment`

**Functions:**
- `train_epoch` (line 177) `def train_epoch(model, loader, optimizer, criterion, device, use_mixup)`
- `evaluate` (line 215) `def evaluate(model, loader, device)`
- `main` (line 435) `def main()`
- `__init__` (line 40) `def __init__(self, in_c, out_c, stride)`
- `forward` (line 54) `def forward(self, x)`
- `__init__` (line 65) `def __init__(self, num_classes)`
- `_make_layer` (line 79) `def _make_layer(self, in_c, out_c, num_blocks, stride)`
- `forward` (line 85) `def forward(self, x)`
- `__init__` (line 98) `def __init__(self, in_dim, out_dim)`
- `forward` (line 107) `def forward(self, x, plasticity)`
- `__init__` (line 127) `def __init__(self, num_classes)`
- `_make_layer` (line 146) `def _make_layer(self, in_c, out_c, num_blocks, stride)`
- `forward` (line 152) `def forward(self, x)`
- `update_plasticity` (line 163) `def update_plasticity(self, epoch, total_epochs)` - *Plasticity schedule simplificado*
- `__init__` (line 234) `def __init__(self, config)`
- `_get_data` (line 245) `def _get_data(self)`
- `run_baseline` (line 269) `def run_baseline(self)`
- `run_neurosovereign` (line 316) `def run_neurosovereign(self)`
- `compare` (line 369) `def compare(self)`
- `plot_comparison` (line 404) `def plot_comparison(self)`

#### `neurosoberano_bicameral_opt.py`
**Path:** `neurosoberano_bicameral_opt.py`

**Classes:**
- `HomeostaticRegulator` (line 39) `class HomeostaticRegulator`
- `LiquidNeuron` (line 66) `class LiquidNeuron`
- `RightHemisphere` (line 193) `class RightHemisphere`
- `LeftHemisphere` (line 214) `class LeftHemisphere`
- `CorpusCallosum` (line 313) `class CorpusCallosum`
- `NeuroLogosBicameral` (line 331) `class NeuroLogosBicameral`
- `NeuralDiagnostics` (line 361) `class NeuralDiagnostics`
- `Flickr8kDataset` (line 436) `class Flickr8kDataset(Dataset)`
- `LifeCycle` (line 498) `class LifeCycle`

**Functions:**
- `build_vocab_flickr` (line 475) `def build_vocab_flickr(captions_file, vocab_size)`
- `train_bicameral` (line 513) `def train_bicameral()`
- `__init__` (line 40) `def __init__(self)`
- `forward` (line 50) `def forward(self, stress, excitation, fatigue, entropy, phase, loss_signal)`
- `__init__` (line 67) `def __init__(self, in_dim, out_dim)`
- `forward` (line 99) `def forward(self, x, global_plasticity, transfer_rate, task_loss)`
- `apply_svd_consolidation` (line 169) `def apply_svd_consolidation(self, repair_strength, timescale)`
- `__init__` (line 194) `def __init__(self, output_dim)`
- `forward` (line 205) `def forward(self, image, plasticity, transfer_rate, task_loss)`
- `__init__` (line 215) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 237) `def forward(self, visual_context, captions, max_len, return_gate)`
- `_get_init_state` (line 293) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 298) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 314) `def __init__(self, dim)`
- `forward` (line 322) `def forward(self, right_features, metabolism)`
- `__init__` (line 332) `def __init__(self, vocab_size)`
- `forward` (line 338) `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)`
- `__init__` (line 362) `def __init__(self)`
- `measure_callosal_flow` (line 375) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 382) `def measure_vocab_diversity(self, generated_tokens, vocab_size)`
- `update` (line 386) `def update(self)`
- `get_recent_avg` (line 391) `def get_recent_avg(self, key, n)`
- `report` (line 396) `def report(self, epoch)`
- `__init__` (line 437) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 455) `def __len__(self)`
- `__getitem__` (line 458) `def __getitem__(self, idx)`
- `__init__` (line 499) `def __init__(self, total_epochs)`
- `get_plasticity` (line 502) `def get_plasticity(self, epoch)`

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

**Classes:**
- `SovereignConfig` (line 16) `class SovereignConfig`
- `BasicBlock` (line 69) `class BasicBlock`
- `NetworkBlock` (line 96) `class NetworkBlock`
- `LiquidCortex` (line 111) `class LiquidCortex` - *Capa densa con Fast Weights Hebbianos y Homeostasis.
Reemplaza a la capa lineal aburrida de las CNNs normales.*
- `NeuroSovereignV1` (line 165) `class NeuroSovereignV1`

**Functions:**
- `seed_everything` (line 44) `def seed_everything(seed)`
- `mixup_data` (line 51) `def mixup_data(x, y, alpha)` - *Returns mixed inputs, pairs of targets, and lambda*
- `mixup_criterion` (line 63) `def mixup_criterion(criterion, pred, y_a, y_b, lam)`
- `get_optimized_dataloaders` (line 214) `def get_optimized_dataloaders(config)`
- `train_sovereign` (line 239) `def train_sovereign()`
- `__init__` (line 70) `def __init__(self, in_planes, out_planes, stride, dropRate)`
- `forward` (line 85) `def forward(self, x)`
- `__init__` (line 97) `def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate)`
- `_make_layer` (line 100) `def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate)`
- `forward` (line 105) `def forward(self, x)`
- `__init__` (line 116) `def __init__(self, in_features, out_features, config)`
- `forward` (line 133) `def forward(self, x)`
- `__init__` (line 166) `def __init__(self, config, depth, num_classes)`
- `forward` (line 196) `def forward(self, x)`

#### `ohm.py`
**Path:** `ohm.py`

**Classes:**
- `MotorHomeostaticContext` (line 34) `class MotorHomeostaticContext` - *Contexto para un motor homeostático*
- `PTSymmetricMotor` (line 64) `class PTSymmetricMotor(MotorHomeostaticContext)` - *Motor para controlar parámetros PT-similares*
- `TopologicalMotor` (line 100) `class TopologicalMotor(MotorHomeostaticContext)` - *Motor para controlar conectividad y topología*
- `EnergyHomeostaticMotor` (line 127) `class EnergyHomeostaticMotor(MotorHomeostaticContext)` - *Motor para controlar eficiencia energética*
- `ConsciousnessMotor` (line 157) `class ConsciousnessMotor(MotorHomeostaticContext)` - *Motor para controlar métricas de conciencia (Φₑ)*
- `DualSystemMotor` (line 185) `class DualSystemMotor(MotorHomeostaticContext)` - *Motor para controlar balance inconsciente/consciente*
- `AdaptiveLearningMotor` (line 213) `class AdaptiveLearningMotor(MotorHomeostaticContext)` - *Motor para adaptar algoritmos de aprendizaje*
- `ModularActivationMotor` (line 241) `class ModularActivationMotor(MotorHomeostaticContext)` - *Motor para activar/desactivar módulos según contexto*
- `OmniBrainCoordinator` (line 291) `class OmniBrainCoordinator` - *Coordinador central que gestiona todos los motores homeostáticos*
- `OmniBrainModule` (line 438) `class OmniBrainModule` - *Módulo base para todos los componentes del Omni Brain*
- `PTSymmetricLayer` (line 453) `class PTSymmetricLayer(OmniBrainModule)` - *Capa con activación PT-simétrica regulada*
- `TopologicalLayer` (line 486) `class TopologicalLayer(OmniBrainModule)` - *Capa con conectividad topológica regulada*
- `DualMindModule` (line 543) `class DualMindModule(OmniBrainModule)` - *Módulo de procesamiento dual (inconsciente/consciente)*
- `ConsciousnessModule` (line 597) `class ConsciousnessModule(OmniBrainModule)` - *Módulo de métricas de conciencia y integración*
- `HomeostaticEngine` (line 658) `class HomeostaticEngine` - *Motor homeostasis reutilizable de Síntesis v8.2*
- `OmniBrain` (line 689) `class OmniBrain` - *El pokemon legendario que combina todas las ideas*

**Functions:**
- `train_omni_brain` (line 863) `def train_omni_brain(model, epochs, batch_size)` - *Pipeline de entrenamiento para el Omni Brain*
- `update` (line 47) `def update(self, measurement, dt)` - *Actualiza el estado del motor homeostático*
- `__init__` (line 66) `def __init__(self)`
- `regulate_parameters` (line 78) `def regulate_parameters(self, current_coherence, energy_level)` - *Regula parámetros para mantener PT-simetría*
- `__init__` (line 102) `def __init__(self)`
- `regulate_connectivity` (line 112) `def regulate_connectivity(self, current_connectivity, clustering)` - *Regula conectividad para mantener estructura óptima*
- `__init__` (line 129) `def __init__(self)`
- `regulate_energy` (line 139) `def regulate_energy(self, memory_usage, cpu_usage, temperature)` - *Regula parámetros para eficiencia energética*
- `__init__` (line 159) `def __init__(self)`
- `regulate_consciousness` (line 168) `def regulate_consciousness(self, phi_effective, integration_level)` - *Regula parámetros para control de conciencia*
- `__init__` (line 187) `def __init__(self)`
- `regulate_dual_systems` (line 197) `def regulate_dual_systems(self, unconscious_activity, conscious_activity)` - *Regula balance entre sistemas inconsciente y consciente*
- `__init__` (line 215) `def __init__(self)`
- `regulate_learning` (line 224) `def regulate_learning(self, loss_reduction_rate, gradient_norm)` - *Regula parámetros de aprendizaje*
- `__init__` (line 243) `def __init__(self)`
- `regulate_modules` (line 259) `def regulate_modules(self, task_complexity, resource_availability, performance)` - *Regula qué módulos están activos*
- `__init__` (line 294) `def __init__(self)`
- `_initialize_motors` (line 300) `def _initialize_motors(self)` - *Inicializa todos los motores homeostáticos*
- `sense_environment` (line 312) `def sense_environment(self)` - *Sensa el estado actual del entorno*
- `measure_network_state` (line 326) `def measure_network_state(self, model, batch_data)` - *Mide el estado actual de la red*
- `coordinate_all_motors` (line 363) `def coordinate_all_motors(self, environment_state, network_state)` - *Coordina todos los motores homeostáticos*
- `__init__` (line 441) `def __init__(self, module_name, enabled)`
- `forward` (line 447) `def forward(self, x, params)`
- `update_performance` (line 450) `def update_performance(self, metrics)`
- `__init__` (line 456) `def __init__(self, in_features, out_features)`
- `forward` (line 463) `def forward(self, x, params)`
- `__init__` (line 489) `def __init__(self, in_features, out_features, sparsity_factor)`
- `_generate_topology_mask` (line 504) `def _generate_topology_mask(self)` - *Genera máscara topológica realista*
- `forward` (line 525) `def forward(self, x, params)`
- `__init__` (line 546) `def __init__(self, features)`
- `forward` (line 571) `def forward(self, x, params)`
- `__init__` (line 600) `def __init__(self, features)`
- `compute_phi_effective` (line 614) `def compute_phi_effective(self, x)` - *Cálculo simplificado de Φₑ (integración efectiva)*
- `forward` (line 634) `def forward(self, x, params)`
- `__init__` (line 661) `def __init__(self, target_performance)`
- `regulate_homeostasis` (line 666) `def regulate_homeostasis(self, observed_performance)` - *Regula parámetros para homeostasis*
- `__init__` (line 692) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `initialize_context` (line 726) `def initialize_context(self)` - *Inicializa el contexto del Omni Brain*
- `forward` (line 740) `def forward(self, x)` - *Forward pass del Omni Brain con coordinación homeostática*
- `get_status_report` (line 828) `def get_status_report(self)` - *Genera reporte de estado del Omni Brain*

#### `omni1.py`
**Path:** `omni1.py`

**Classes:**
- `FastSlowLinear` (line 31) `class FastSlowLinear` - *Linear layer con pesos hebbianos mejorados y mayor capacidad de adaptación.*
- `ConsciousnessModule` (line 83) `class ConsciousnessModule` - *Módulo de consciencia con Φₑ mejorado y umbral reducido para integración temprana.*
- `OmniBrainV8` (line 140) `class OmniBrainV8` - *Arquitectura optimizada con conexiones residuales.*
- `DualSystemModule` (line 215) `class DualSystemModule` - *Sistema dual rápido/lento con memoria.*
- `FocalLoss` (line 261) `class FocalLoss`

**Functions:**
- `train_model` (line 250) `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)` - *Entrenamiento optimizado.*
- `evaluate` (line 356) `def evaluate(model, loader, device, return_per_class)` - *Evaluación estándar.*
- `get_cifar10_loaders` (line 399) `def get_cifar10_loaders(batch_size)` - *Loaders con data augmentation.*
- `diagnose_model` (line 423) `def diagnose_model(model, loader, device)` - *Diagnóstico profundo del modelo.*
- `main` (line 481) `def main()` - *POC mejorado.*
- `__init__` (line 33) `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- `reset_fast_weights` (line 46) `def reset_fast_weights(self)`
- `update_fast_weights` (line 49) `def update_fast_weights(self, x)`
- `forward` (line 68) `def forward(self, x)`
- `get_fast_norm` (line 75) `def get_fast_norm(self)`
- `__init__` (line 85) `def __init__(self, features, use_conscious)`
- `compute_phi_effective` (line 99) `def compute_phi_effective(self, activity)`
- `forward` (line 121) `def forward(self, x)`
- `__init__` (line 142) `def __init__(self, use_fastslow, use_conscious)`
- `forward` (line 188) `def forward(self, x)`
- `reset_all_fast_weights` (line 198) `def reset_all_fast_weights(self)`
- `get_fast_norms` (line 205) `def get_fast_norms(self)`
- `__init__` (line 217) `def __init__(self, dim, use_fastslow)`
- `forward` (line 235) `def forward(self, x)`
- `get_activation` (line 436) `def get_activation(name)`
- `__init__` (line 262) `def __init__(self, alpha, gamma)`
- `forward` (line 268) `def forward(self, inputs, targets)`
- `hook` (line 437) `def hook(model, input, output)`

#### `omni3.py`
**Path:** `omni3.py`

**Classes:**
- `Config` (line 22) `class Config`
- `FastSlowLinear` (line 105) `class FastSlowLinear`
- `DualSystemModule` (line 196) `class DualSystemModule`
- `IntegrationModule` (line 232) `class IntegrationModule` - *Renombrado de ConsciousnessModule - más honesto sobre su función*
- `OmniBrainFastSlow` (line 268) `class OmniBrainFastSlow`

**Functions:**
- `compute_integration_index` (line 65) `def compute_integration_index(activity)` - *MÉTRICA HONESTA: Mide el ratio de varianza explicada por el primer componente.
Utiliza SVD para estabilidad numérica en matrices de covarianza deficientes.*
- `get_cifar10_loaders` (line 340) `def get_cifar10_loaders(config)`
- `evaluate_full` (line 368) `def evaluate_full(model, loader, device)` - *Evaluación con múltiples métricas*
- `train` (line 403) `def train(config)`
- `run_ablation_study` (line 539) `def run_ablation_study()` - *Ejecuta múltiples configuraciones para validar cada componente*
- `__init__` (line 106) `def __init__(self, in_features, out_features, config)`
- `reset_fast_weights` (line 128) `def reset_fast_weights(self)` - *Reinicia la memoria a corto plazo (debe llamarse por época, no por batch)*
- `update_fast_weights` (line 134) `def update_fast_weights(self, x, slow_out)` - *Actualización Hebbiana controlada.*
- `forward` (line 169) `def forward(self, x)`
- `get_fast_norm` (line 189) `def get_fast_norm(self)`
- `__init__` (line 197) `def __init__(self, dim, config)`
- `forward` (line 211) `def forward(self, x)`
- `__init__` (line 236) `def __init__(self, features, config)`
- `forward` (line 248) `def forward(self, x)`
- `__init__` (line 269) `def __init__(self, config)`
- `forward` (line 304) `def forward(self, x)`
- `reset_all_fast_weights` (line 314) `def reset_all_fast_weights(self)` - *Reinicia todos los pesos rápidos del modelo. 
FIX: Ahora solo cada 10 épocas para preservar memoria a corto plazo.*
- `get_fast_norms` (line 323) `def get_fast_norms(self)` - *Recopila normas de fast weights de todos los módulos*
- `get_ablation_state` (line 327) `def get_ablation_state(self)` - *Estado actual para logging*

#### `omnibrain.py`
**Path:** `omnibrain.py`

**Classes:**
- `FastSlowLinear` (line 31) `class FastSlowLinear` - *Linear layer con pesos hebbianos estabilizados.*
- `DualSystemModule` (line 91) `class DualSystemModule` - *Sistema dual rápido/lento con memoria.*
- `ConsciousnessModule` (line 119) `class ConsciousnessModule` - *Módulo de consciencia con Φₑ mejorado.*
- `OmniBrainV8` (line 174) `class OmniBrainV8` - *Arquitectura completa con switches para ablation.*

**Functions:**
- `get_cifar10_loaders` (line 233) `def get_cifar10_loaders(batch_size)` - *Loaders con data augmentation.*
- `get_few_shot_loaders` (line 253) `def get_few_shot_loaders(n_way, k_shot, batch_size)` - *Few-shot learning setup: entrenar en clases limitadas.*
- `evaluate` (line 287) `def evaluate(model, loader, device, return_per_class)` - *Evaluación con opción de métricas por clase.*
- `train_model` (line 331) `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)` - *Entrenamiento con learning rate scheduler y early stopping.*
- `run_ablation_study` (line 430) `def run_ablation_study(epochs, batch_size)` - *Ejecuta 4 configuraciones y compara resultados.*
- `run_few_shot_experiment` (line 475) `def run_few_shot_experiment(n_way, k_shot, epochs)` - *Prueba capacidad de few-shot learning.*
- `analyze_phi_per_class` (line 508) `def analyze_phi_per_class()` - *Analiza correlación entre Φₑ y dificultad de clase.*
- `plot_ablation_results` (line 545) `def plot_ablation_results(results)` - *Genera gráficas comparativas de ablation study.*
- `plot_phi_analysis` (line 617) `def plot_phi_analysis(class_accs, avg_phi_per_class)` - *Gráfica correlación Φₑ vs dificultad de clase.*
- `main` (line 673) `def main()` - *Ejecuta el POC completo.*
- `__init__` (line 33) `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- `reset_fast_weights` (line 49) `def reset_fast_weights(self)`
- `update_fast_weights` (line 53) `def update_fast_weights(self, x)`
- `forward` (line 73) `def forward(self, x)`
- `end_of_batch` (line 84) `def end_of_batch(self)`
- `get_fast_norm` (line 87) `def get_fast_norm(self)`
- `__init__` (line 93) `def __init__(self, dim, use_fastslow)`
- `forward` (line 109) `def forward(self, x)`
- `__init__` (line 121) `def __init__(self, features, use_conscious)`
- `compute_phi_effective` (line 133) `def compute_phi_effective(self, activity)` - *Φₑ basado en eigenvalues de covarianza.*
- `forward` (line 156) `def forward(self, x)`
- `__init__` (line 176) `def __init__(self, use_fastslow, use_conscious)`
- `forward` (line 209) `def forward(self, x)`
- `reset_all_fast_weights` (line 216) `def reset_all_fast_weights(self)`
- `get_fast_norms` (line 223) `def get_fast_norms(self)`

#### `omnibrain_k.py`
**Path:** `omnibrain_k.py`

**Classes:**
- `Config` (line 22) `class Config`
- `FastSlowLinear` (line 105) `class FastSlowLinear`
- `DualSystemModule` (line 195) `class DualSystemModule`
- `IntegrationModule` (line 231) `class IntegrationModule` - *Renombrado de ConsciousnessModule - más honesto sobre su función*
- `OmniBrainFastSlow` (line 266) `class OmniBrainFastSlow`

**Functions:**
- `compute_integration_index` (line 65) `def compute_integration_index(activity)` - *MÉTRICA HONESTA: Mide el ratio de varianza explicada por el primer componente.
Utiliza SVD para estabilidad numérica en matrices de covarianza deficientes.*
- `get_cifar10_loaders` (line 339) `def get_cifar10_loaders(config)`
- `evaluate_full` (line 367) `def evaluate_full(model, loader, device)` - *Evaluación con múltiples métricas*
- `train` (line 402) `def train(config)`
- `run_ablation_study` (line 537) `def run_ablation_study()` - *Ejecuta múltiples configuraciones para validar cada componente*
- `__init__` (line 106) `def __init__(self, in_features, out_features, config)`
- `reset_fast_weights` (line 128) `def reset_fast_weights(self)` - *Reinicia la memoria a corto plazo (debe llamarse por época, no por batch)*
- `update_fast_weights` (line 134) `def update_fast_weights(self, x, slow_out)` - *Actualización Hebbiana controlada.*
- `forward` (line 169) `def forward(self, x)`
- `get_fast_norm` (line 189) `def get_fast_norm(self)`
- `__init__` (line 196) `def __init__(self, dim, config)`
- `forward` (line 210) `def forward(self, x)`
- `__init__` (line 235) `def __init__(self, features, config)`
- `forward` (line 247) `def forward(self, x)`
- `__init__` (line 267) `def __init__(self, config)`
- `forward` (line 302) `def forward(self, x)`
- `reset_all_fast_weights` (line 312) `def reset_all_fast_weights(self)` - *Reinicia todos los pesos rápidos del modelo. Debe llamarse explícitamente
(por ejemplo, al inicio de cada época si se desea resetear la memoria a corto plazo).
No se activa automáticamente durante forward.*
- `get_fast_norms` (line 322) `def get_fast_norms(self)` - *Recopila normas de fast weights de todos los módulos*
- `get_ablation_state` (line 326) `def get_ablation_state(self)` - *Estado actual para logging*

#### `omno1.bkp.py.py`
**Path:** `omno1.bkp.py.py`

**Classes:**
- `FastSlowLinear` (line 32) `class FastSlowLinear` - *Linear layer con pesos hebbianos mejorados.*
- `ConsciousnessModule` (line 101) `class ConsciousnessModule` - *Módulo de consciencia con Φₑ mejorado y menos restrictivo.*
- `OmniBrainV8` (line 181) `class OmniBrainV8` - *Arquitectura optimizada con conexiones residuales.*
- `DualSystemModule` (line 256) `class DualSystemModule` - *Sistema dual rápido/lento con memoria.*
- `FocalLoss` (line 302) `class FocalLoss`

**Functions:**
- `train_model` (line 291) `def train_model(model, train_loader, test_loader, device, epochs, lr, config_name, verbose)` - *Entrenamiento optimizado.*
- `evaluate` (line 399) `def evaluate(model, loader, device, return_per_class)` - *Evaluación estándar.*
- `get_cifar10_loaders` (line 442) `def get_cifar10_loaders(batch_size)` - *Loaders con data augmentation.*
- `diagnose_model` (line 466) `def diagnose_model(model, loader, device)` - *Diagnóstico profundo del modelo.*
- `main` (line 524) `def main()` - *POC mejorado.*
- `__init__` (line 34) `def __init__(self, in_features, out_features, fast_lr, fast_decay)`
- `reset_fast_weights` (line 51) `def reset_fast_weights(self)`
- `update_fast_weights` (line 56) `def update_fast_weights(self, x)`
- `forward` (line 85) `def forward(self, x)`
- `get_fast_norm` (line 93) `def get_fast_norm(self)`
- `__init__` (line 103) `def __init__(self, features, use_conscious)`
- `compute_phi_effective` (line 118) `def compute_phi_effective(self, activity)` - *Φₑ mejorado con condiciones menos restrictivas.*
- `forward` (line 159) `def forward(self, x)`
- `__init__` (line 183) `def __init__(self, use_fastslow, use_conscious)`
- `forward` (line 229) `def forward(self, x)`
- `reset_all_fast_weights` (line 239) `def reset_all_fast_weights(self)`
- `get_fast_norms` (line 246) `def get_fast_norms(self)`
- `__init__` (line 258) `def __init__(self, dim, use_fastslow)`
- `forward` (line 276) `def forward(self, x)`
- `get_activation` (line 479) `def get_activation(name)`
- `__init__` (line 303) `def __init__(self, alpha, gamma)`
- `forward` (line 309) `def forward(self, inputs, targets)`
- `hook` (line 480) `def hook(model, input, output)`

#### `physio_chimera_demo.py`
**Path:** `physio_chimera_demo.py`

**Classes:**
- `Config` (line 24) `class Config`
- `DataEnvironment` (line 46) `class DataEnvironment`
- `SimpleMonitor` (line 82) `class SimpleMonitor`
- `SimpleCMS` (line 130) `class SimpleCMS`
- `SimplePhysioNeuron` (line 153) `class SimplePhysioNeuron`
- `SimplePhysioChimera` (line 194) `class SimplePhysioChimera`

**Functions:**
- `seed_everything` (line 36) `def seed_everything(seed)`
- `train_demo` (line 231) `def train_demo(config)`
- `run_demo` (line 298) `def run_demo()`
- `__init__` (line 47) `def __init__(self)`
- `get_batch` (line 57) `def get_batch(self, phase, bs)`
- `get_full` (line 71) `def get_full(self)` - *Retorna el dataset completo*
- `get_w2` (line 75) `def get_w2(self)` - *Retorna solo los datos de WORLD_2 (dígitos >= 5)*
- `__init__` (line 83) `def __init__(self)`
- `update` (line 88) `def update(self, loss, physio)`
- `report` (line 94) `def report(self, step, phase)`
- `__init__` (line 131) `def __init__(self, levels, d_model, hidden_dim)`
- `forward` (line 142) `def forward(self, x, global_step)`
- `__init__` (line 154) `def __init__(self, d_in, d_out, config)`
- `forward` (line 162) `def forward(self, x, global_step)`
- `__init__` (line 195) `def __init__(self, config)`
- `forward` (line 207) `def forward(self, x, global_step)`

#### `physio_chimera_v15_monitored.py`
**Path:** `physio_chimera_v15_monitored.py`

**Classes:**
- `Config` (line 42) `class Config`
- `DataEnvironment` (line 68) `class DataEnvironment`
- `NeuralDiagnostics` (line 104) `class NeuralDiagnostics` - *Sistema de diagnóstico neurológico para Physio-Chimera*
- `SelfModifyingGates` (line 298) `class SelfModifyingGates`
- `ContinuumMemorySystem` (line 319) `class ContinuumMemorySystem`
- `NestedPhysioNeuron` (line 346) `class NestedPhysioNeuron`
- `PhysioChimeraNested` (line 397) `class PhysioChimeraNested`
- `MetricsVisualizer` (line 452) `class MetricsVisualizer` - *Visualiza métricas de entrenamiento*

**Functions:**
- `seed_everything` (line 58) `def seed_everything(seed)`
- `train_nested_monitored` (line 658) `def train_nested_monitored(config)`
- `run_experiment_monitored` (line 762) `def run_experiment_monitored()`
- `__init__` (line 69) `def __init__(self)`
- `get_batch` (line 79) `def get_batch(self, phase, bs)`
- `get_full` (line 93) `def get_full(self)` - *Retorna el dataset completo*
- `get_w2` (line 97) `def get_w2(self)` - *Retorna solo los datos de WORLD_2 (dígitos >= 5)*
- `__init__` (line 107) `def __init__(self, config)`
- `update_physio_metrics` (line 144) `def update_physio_metrics(self, metabolism, sensitivity, gate)` - *Actualiza métricas fisiológicas*
- `update_performance_metrics` (line 150) `def update_performance_metrics(self, loss, accuracy, lr)` - *Actualiza métricas de rendimiento*
- `update_memory_metrics` (line 158) `def update_memory_metrics(self, cms_activations, hebbian_norm, forgetting_factor)` - *Actualiza métricas de memoria*
- `calculate_health_metrics` (line 166) `def calculate_health_metrics(self)` - *Calcula métricas de salud del sistema*
- `get_recent_avg` (line 191) `def get_recent_avg(self, category, key, n)` - *Obtiene promedio reciente de una métrica*
- `generate_diagnostic_report` (line 209) `def generate_diagnostic_report(self, step, phase)` - *Genera reporte de diagnóstico*
- `save_metrics` (line 279) `def save_metrics(self, filepath)` - *Guarda todas las métricas*
- `__init__` (line 299) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 306) `def forward(self, x)`
- `__init__` (line 320) `def __init__(self, levels, d_model, hidden_dim)`
- `forward` (line 332) `def forward(self, x, global_step)`
- `__init__` (line 347) `def __init__(self, d_in, d_out, config)`
- `forward` (line 362) `def forward(self, x, global_step)`
- `__init__` (line 398) `def __init__(self, config)`
- `forward` (line 410) `def forward(self, x, global_step)`
- `__init__` (line 455) `def __init__(self, save_dir)`
- `plot_training_curves` (line 463) `def plot_training_curves(self, diagnostics)` - *Genera gráficos de curvas de entrenamiento*
- `create_final_report` (line 551) `def create_final_report(self, final_metrics, diagnostics)` - *Crea reporte final con todas las métricas*
- `_generate_recommendations` (line 634) `def _generate_recommendations(self, diagnostics)` - *Genera recomendaciones basadas en el diagnóstico*

#### `physioneruon_simple.py`
**Path:** `physioneruon_simple.py`

**Classes:**
- `SimpleConfig` (line 27) `class SimpleConfig`
- `SimpleRobustNet` (line 83) `class SimpleRobustNet` - *Red simple pero bien regularizada*

**Functions:**
- `seed_everything` (line 54) `def seed_everything(seed)`
- `get_dataset` (line 61) `def get_dataset(config)` - *Dataset balanceado con más separabilidad*
- `pgd_attack` (line 117) `def pgd_attack(model, x, y, eps, steps, step_size)` - *PGD estándar bien implementado
- Random start
- Step size controlado
- Projection al epsilon-ball*
- `train_simple_robust` (line 154) `def train_simple_robust(config, dataset, verbose)` - *Entrenamiento con adversarial training progresivo*
- `main` (line 281) `def main()`
- `__init__` (line 85) `def __init__(self, config)`
- `forward` (line 105) `def forward(self, x)`

#### `physioneuron_cpu_v1.py`
**Path:** `physioneuron_cpu_v1.py`

**Classes:**
- `MicroConfig` (line 33) `class MicroConfig`
- `HomeostaticRegulator` (line 85) `class HomeostaticRegulator`
- `PhysioNeuron` (line 109) `class PhysioNeuron`
- `MicroContinuumCell` (line 151) `class MicroContinuumCell`
- `MicroSymbioticBasis` (line 176) `class MicroSymbioticBasis`
- `MicroTopology` (line 196) `class MicroTopology`
- `MicroSupConLoss` (line 216) `class MicroSupConLoss`
- `MicroTopoBrain` (line 240) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 57) `def seed_everything(seed)`
- `get_dataset` (line 65) `def get_dataset(config)`
- `micro_pgd_attack` (line 340) `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- `generate_ablation_matrix` (line 364) `def generate_ablation_matrix()`
- `train_with_cv` (line 393) `def train_with_cv(config, dataset, cv_folds)`
- `run_ablation_study` (line 457) `def run_ablation_study()`
- `__init__` (line 86) `def __init__(self, d_in)`
- `forward` (line 96) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 110) `def __init__(self, d_in, d_out, dynamic_mode)`
- `forward` (line 121) `def forward(self, x)`
- `__init__` (line 152) `def __init__(self, dim)`
- `forward` (line 162) `def forward(self, x, plasticity)`
- `__init__` (line 177) `def __init__(self, dim, num_atoms)`
- `forward` (line 185) `def forward(self, x)`
- `__init__` (line 197) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 210) `def get_adjacency(self, plasticity)`
- `__init__` (line 217) `def __init__(self, temperature)`
- `forward` (line 222) `def forward(self, features, labels)`
- `__init__` (line 241) `def __init__(self, config)`
- `_init_weights` (line 278) `def _init_weights(self)`
- `count_parameters` (line 283) `def count_parameters(self)`
- `forward` (line 286) `def forward(self, x, plasticity)`

#### `physioneuron_cpu_v2.py`
**Path:** `physioneuron_cpu_v2.py`

**Classes:**
- `MicroConfig` (line 35) `class MicroConfig`
- `HomeostaticRegulator` (line 92) `class HomeostaticRegulator`
- `PhysioNeuron` (line 116) `class PhysioNeuron`
- `MicroContinuumCell` (line 158) `class MicroContinuumCell`
- `MicroSymbioticBasis` (line 183) `class MicroSymbioticBasis`
- `MicroTopology` (line 203) `class MicroTopology`
- `MicroSupConLoss` (line 223) `class MicroSupConLoss`
- `MicroTopoBrain` (line 247) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 64) `def seed_everything(seed)`
- `get_dataset` (line 72) `def get_dataset(config)`
- `micro_pgd_attack` (line 347) `def micro_pgd_attack(model, x, y, eps, steps, plasticity)`
- `generate_ablation_matrix` (line 371) `def generate_ablation_matrix()`
- `train_with_cv` (line 400) `def train_with_cv(config, dataset, cv_folds)`
- `run_ablation_study` (line 464) `def run_ablation_study()`
- `__init__` (line 93) `def __init__(self, d_in)`
- `forward` (line 103) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 117) `def __init__(self, d_in, d_out, dynamic_mode)`
- `forward` (line 128) `def forward(self, x)`
- `__init__` (line 159) `def __init__(self, dim)`
- `forward` (line 169) `def forward(self, x, plasticity)`
- `__init__` (line 184) `def __init__(self, dim, num_atoms)`
- `forward` (line 192) `def forward(self, x)`
- `__init__` (line 204) `def __init__(self, num_nodes, config)`
- `get_adjacency` (line 217) `def get_adjacency(self, plasticity)`
- `__init__` (line 224) `def __init__(self, temperature)`
- `forward` (line 229) `def forward(self, features, labels)`
- `__init__` (line 248) `def __init__(self, config)`
- `_init_weights` (line 285) `def _init_weights(self)`
- `count_parameters` (line 290) `def count_parameters(self)`
- `forward` (line 293) `def forward(self, x, plasticity)`

#### `physioneuron_cpu_v3.py`
**Path:** `physioneuron_cpu_v3.py`

**Classes:**
- `EliteConfig` (line 35) `class EliteConfig`
- `EpisodicMemory` (line 102) `class EpisodicMemory` - *Memoria explícita de patrones adversariales*
- `SpectralNormLinear` (line 136) `class SpectralNormLinear` - *Linear con normalización espectral para estabilidad Lipschitz*
- `AdvancedHomeostaticCell` (line 162) `class AdvancedHomeostaticCell` - *Neurona con control fisiológico multinivel + memoria*
- `AdaptiveTopology` (line 222) `class AdaptiveTopology` - *Topología que aprende a reconectar bajo ataque*
- `EliteTopoBrain` (line 260) `class EliteTopoBrain`
- `SupConLoss` (line 385) `class SupConLoss`

**Functions:**
- `seed_everything` (line 71) `def seed_everything(seed)`
- `get_elite_dataset` (line 79) `def get_elite_dataset(config)` - *Dataset más grande y balanceado con separabilidad controlada*
- `elite_pgd_attack` (line 348) `def elite_pgd_attack(model, x, y, eps, steps, stress)` - *PGD con reinicio aleatorio*
- `train_elite_model` (line 430) `def train_elite_model(config, dataset, fold_results)` - *Entrenamiento con curriculum adversarial*
- `run_elite_experiment` (line 542) `def run_elite_experiment()`
- `__init__` (line 104) `def __init__(self, dim, capacity)`
- `update` (line 112) `def update(self, x, y)` - *Almacena ejemplos duros*
- `retrieve` (line 125) `def retrieve(self, x, k)` - *Recupera k vecinos más cercanos*
- `__init__` (line 138) `def __init__(self, in_features, out_features)`
- `power_iteration` (line 145) `def power_iteration(self, n_iter)` - *Aproxima la norma espectral máxima*
- `forward` (line 152) `def forward(self, x)`
- `__init__` (line 164) `def __init__(self, d_in, d_out, use_spectral)`
- `forward` (line 194) `def forward(self, x)`
- `__init__` (line 224) `def __init__(self, num_nodes, grid_size)`
- `forward` (line 248) `def forward(self, stress)` - *stress ∈ [0,1]: cuánto estrés adversarial*
- `__init__` (line 261) `def __init__(self, config)`
- `count_parameters` (line 303) `def count_parameters(self)`
- `forward` (line 306) `def forward(self, x, stress)`
- `__init__` (line 386) `def __init__(self, temperature)`
- `forward` (line 390) `def forward(self, features, labels)`

#### `poke_cifar.py`
**Path:** `poke_cifar.py`

**Classes:**
- `PTSymmetricLayer` (line 56) `class PTSymmetricLayer`
- `TopologicalLayer` (line 77) `class TopologicalLayer`
- `DualSystemModule` (line 98) `class DualSystemModule`
- `ConsciousnessModule` (line 116) `class ConsciousnessModule`
- `OmniBrainCIFAR` (line 133) `class OmniBrainCIFAR`

**Functions:**
- `compute_phi_effective` (line 35) `def compute_phi_effective(activity)`
- `get_cifar10_loaders` (line 180) `def get_cifar10_loaders(batch_size)`
- `evaluate` (line 200) `def evaluate(model, loader, device)`
- `main` (line 221) `def main()`
- `plot_history` (line 287) `def plot_history(hist)`
- `demo_inference` (line 300) `def demo_inference(model, loader)`
- `__init__` (line 57) `def __init__(self, in_features, out_features)`
- `forward` (line 66) `def forward(self, x)`
- `__init__` (line 78) `def __init__(self, in_f, out_f, density)`
- `_update_mask` (line 87) `def _update_mask(self)`
- `forward` (line 93) `def forward(self, x)`
- `__init__` (line 99) `def __init__(self, features)`
- `forward` (line 107) `def forward(self, x)`
- `__init__` (line 117) `def __init__(self, features)`
- `forward` (line 123) `def forward(self, x)`
- `__init__` (line 134) `def __init__(self)`
- `forward` (line 161) `def forward(self, x)`

#### `poke_cifar2.py`
**Path:** `poke_cifar2.py`

**Classes:**
- `FastSlowLinear` (line 49) `class FastSlowLinear`
- `DualSystemModule` (line 99) `class DualSystemModule`
- `ConsciousnessModule` (line 117) `class ConsciousnessModule`
- `OmniBrainFastSlow` (line 131) `class OmniBrainFastSlow`

**Functions:**
- `compute_phi_effective` (line 28) `def compute_phi_effective(activity)`
- `get_cifar10_loaders` (line 175) `def get_cifar10_loaders(batch_size)`
- `evaluate` (line 191) `def evaluate(model, loader, device)`
- `train` (line 212) `def train()`
- `__init__` (line 50) `def __init__(self, in_features, out_features, fast_lr)`
- `reset_fast_weights` (line 65) `def reset_fast_weights(self)`
- `update_fast_weights` (line 69) `def update_fast_weights(self, x)`
- `forward` (line 76) `def forward(self, x)`
- `end_of_batch` (line 89) `def end_of_batch(self)`
- `get_fast_weight_norm` (line 92) `def get_fast_weight_norm(self)`
- `__init__` (line 100) `def __init__(self, dim)`
- `forward` (line 108) `def forward(self, x)`
- `__init__` (line 118) `def __init__(self, dim)`
- `forward` (line 124) `def forward(self, x)`
- `__init__` (line 132) `def __init__(self)`
- `forward` (line 151) `def forward(self, x)`
- `reset_all_fast_weights` (line 159) `def reset_all_fast_weights(self)`
- `get_fast_norms` (line 164) `def get_fast_norms(self)`

#### `pokemon3.py`
**Path:** `pokemon3.py`

**Classes:**
- `HomeostasisContext` (line 76) `class HomeostasisContext`
- `PTSymmetricLayer` (line 82) `class PTSymmetricLayer` - *Capa PT-simétrica compatible con todas las versiones*
- `TopologicalLayer` (line 109) `class TopologicalLayer` - *Capa topológica estable sin dependencias problemáticas*
- `DualSystemModule` (line 134) `class DualSystemModule` - *Sistema dual compatible con PyTorch 1.8+*
- `ConsciousnessModule` (line 164) `class ConsciousnessModule` - *Módulo de conciencia estable*
- `OmniBrain` (line 187) `class OmniBrain` - *¡El Pokémon Legendario compatible con todas las versiones!*

**Functions:**
- `compute_phi_effective_approx` (line 30) `def compute_phi_effective_approx(activity)` - *Cálculo estable de Φₑ compatible con todas las versiones*
- `estimate_energy_consumption` (line 63) `def estimate_energy_consumption(model, batch_size)` - *Estimación conservadora de energía*
- `prepare_mnist_data` (line 245) `def prepare_mnist_data(batch_size, device)` - *Preparar datos MNIST con protección para entornos limitados*
- `train_omni_brain` (line 269) `def train_omni_brain(model, train_loader, test_loader, epochs, device)` - *Entrenamiento compatible con todas las versiones de PyTorch*
- `evaluate_model` (line 380) `def evaluate_model(model, test_loader, device, criterion)` - *Evaluación compatible con todas las versiones*
- `generate_evolution_plots` (line 402) `def generate_evolution_plots(history, epochs)` - *Generar gráficos con protección para entornos sin GUI*
- `demonstrate_inference` (line 435) `def demonstrate_inference(model, test_loader, device)` - *Demostración compatible con todas las versiones*
- `final_report` (line 467) `def final_report(model, history)` - *Reporte final compatible*
- `__init__` (line 85) `def __init__(self, in_features, out_features)`
- `compute_pt_phase` (line 94) `def compute_pt_phase(self)`
- `forward` (line 102) `def forward(self, x)`
- `__init__` (line 112) `def __init__(self, in_features, out_features, connectivity)`
- `update_topology` (line 121) `def update_topology(self, connectivity)`
- `forward` (line 128) `def forward(self, x)`
- `__init__` (line 137) `def __init__(self, features)`
- `forward` (line 153) `def forward(self, x)`
- `__init__` (line 167) `def __init__(self, features)`
- `forward` (line 177) `def forward(self, x)`
- `__init__` (line 190) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 217) `def forward(self, x)`
- `update_topology` (line 235) `def update_topology(self, current_connectivity)`

#### `pokemon4.py`
**Path:** `pokemon4.py`

**Classes:**
- `PTSymmetricLayer` (line 68) `class PTSymmetricLayer`
- `TopologicalLayer` (line 91) `class TopologicalLayer`
- `DualSystemModule` (line 113) `class DualSystemModule`
- `ConsciousnessModule` (line 146) `class ConsciousnessModule`
- `OmniBrain` (line 168) `class OmniBrain`

**Functions:**
- `compute_phi_effective` (line 41) `def compute_phi_effective(activity)` - *Φₑ realista: fracción de varianza explicada por el primer componente PCA.*
- `get_mnist_loaders` (line 206) `def get_mnist_loaders(batch_size)`
- `evaluate` (line 218) `def evaluate(model, loader, device)`
- `train_and_evaluate` (line 236) `def train_and_evaluate()`
- `plot_history` (line 302) `def plot_history(hist)`
- `__init__` (line 69) `def __init__(self, in_features, out_features)`
- `forward` (line 78) `def forward(self, x)`
- `__init__` (line 92) `def __init__(self, in_features, out_features, target_density)`
- `_update_mask` (line 101) `def _update_mask(self)`
- `forward` (line 107) `def forward(self, x)`
- `__init__` (line 114) `def __init__(self, features)`
- `forward` (line 130) `def forward(self, x)`
- `__init__` (line 147) `def __init__(self, features)`
- `forward` (line 157) `def forward(self, x)`
- `__init__` (line 169) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 186) `def forward(self, x)`

#### `pokemon_battle_champion.py`
**Path:** `pokemon_battle_champion.py`

**Classes:**
- `ChampionConfig` (line 31) `class ChampionConfig`
- `PokemonBattleChampion` (line 41) `class PokemonBattleChampion` - *Campeón híbrido que combina VAE + Attention + GAN*

**Functions:**
- `create_battle_dataset` (line 128) `def create_battle_dataset(config)` - *Crear dataset para la batalla*
- `battle_training_epoch` (line 161) `def battle_training_epoch(model, loader, optimizer, criterion, epoch)` - *Entrenamiento de una época de batalla*
- `evaluate_battle_champion` (line 192) `def evaluate_battle_champion(model, loader)` - *Evaluar el campeón en batalla*
- `run_epic_pokemon_battle` (line 208) `def run_epic_pokemon_battle()` - *¡EJECUTAR LA BATALLA ÉPICA!*
- `create_epic_battle_visualization` (line 334) `def create_epic_battle_visualization(battle_history, historical_results)` - *Crear visualización épica de la batalla*
- `save_battle_results` (line 440) `def save_battle_results(battle_history, historical_results, champion_model)` - *Guardar resultados de la batalla épica*
- `__init__` (line 44) `def __init__(self, config)`
- `forward` (line 99) `def forward(self, x)`

#### `pokemon_hybrid_synergy_ablation.py`
**Path:** `pokemon_hybrid_synergy_ablation.py`

**Classes:**
- `SynergyConfig` (line 37) `class SynergyConfig`
- `SynergyVAELayer` (line 82) `class SynergyVAELayer` - *VAE híbrido con capacidades de compactación extremas*
- `SynergyAttentionLayer` (line 125) `class SynergyAttentionLayer` - *Multi-head attention optimizado para datos tabulares*
- `SynergyGANLayer` (line 157) `class SynergyGANLayer` - *GAN híbrido optimizado para features tabulares*
- `AdaptiveTopologyLayer` (line 190) `class AdaptiveTopologyLayer` - *Topología adaptativa inspirada en TopoBrain evolution*
- `PokemonSynergyModel` (line 270) `class PokemonSynergyModel` - *Modelo híbrido que combina los mejores elementos de VAE, Transformer, GAN y TopoBrain*
- `SynergyAblationStudy` (line 386) `class SynergyAblationStudy` - *Estudio de ablación sistemático de sinergias*
- `BaselineModel` (line 428) `class BaselineModel`
- `HybridModel` (line 443) `class HybridModel`
- `AdvancedModel` (line 463) `class AdvancedModel`

**Functions:**
- `run_synergy_ablation` (line 495) `def run_synergy_ablation()` - *Ejecutar estudio de ablación completo*
- `analyze_synergy_results` (line 637) `def analyze_synergy_results(results)` - *Analizar resultados del estudio de sinergias*
- `create_synergy_visualizations` (line 696) `def create_synergy_visualizations(results, output_dir)` - *Crear visualizaciones del estudio de sinergias*
- `to_dict` (line 75) `def to_dict(self)`
- `__init__` (line 84) `def __init__(self, input_dim, hidden_dim, latent_dim)`
- `reparameterize` (line 111) `def reparameterize(self, mu, logvar)`
- `forward` (line 116) `def forward(self, x, return_encoding)`
- `__init__` (line 127) `def __init__(self, d_model, num_heads, d_ff)`
- `forward` (line 149) `def forward(self, x)`
- `__init__` (line 159) `def __init__(self, input_dim, latent_dim, hidden_dim)`
- `generate` (line 184) `def generate(self, z)`
- `discriminate` (line 187) `def discriminate(self, x)`
- `__init__` (line 192) `def __init__(self, grid_size, embed_dim, sparsity)`
- `get_adjacency_matrix` (line 220) `def get_adjacency_matrix(self)`
- `forward` (line 234) `def forward(self, x)`
- `__init__` (line 273) `def __init__(self, config)`
- `forward` (line 327) `def forward(self, x, return_all)`
- `__init__` (line 389) `def __init__(self, config)`
- `get_ablation_matrix` (line 393) `def get_ablation_matrix(self)` - *Matriz de ablación de 4 niveles:

Nivel 1 - Baseline: Solo VAE básico
Nivel 2 - Hybrid: VAE + Attention 
Nivel 3 - Advanced: VAE + Attention + GAN
Nivel 4 - Full Synergy: VAE + Attention + GAN + Topology*
- `create_variant_model` (line 424) `def create_variant_model(self, level_name)` - *Crear modelo variante para un nivel específico*
- `__init__` (line 429) `def __init__(self, config)`
- `forward` (line 434) `def forward(self, x)`
- `__init__` (line 444) `def __init__(self, config)`
- `forward` (line 450) `def forward(self, x)`
- `__init__` (line 464) `def __init__(self, config)`
- `forward` (line 471) `def forward(self, x)`

#### `premium_synergy_demo.py`
**Path:** `premium_synergy_demo.py`

**Classes:**
- `ComponentState` (line 20) `class ComponentState` - *Estado de un componente del sistema*
- `DemocraticDecision` (line 29) `class DemocraticDecision` - *Decisión del sistema democrático*
- `TopoBrainComponent` (line 36) `class TopoBrainComponent` - *TopoBrain v8 - Dynamic Topology + Symbiotic Basis*
- `OmniBrainComponent` (line 75) `class OmniBrainComponent` - *OmniBrain K - Integration Index + Fast-Slow Weights*
- `QuimeraComponent` (line 116) `class QuimeraComponent` - *Quimera v9.5 - Liquid Neurons + Sovereign Attention*
- `HomeostaticMotor` (line 160) `class HomeostaticMotor` - *Motor Homeostático - Cámara Alta de deliberación democrática*
- `PremiumSynergySystem` (line 225) `class PremiumSynergySystem` - *Sistema Premium Synergy completo*

**Functions:**
- `run_demo` (line 302) `def run_demo()` - *Ejecuta demostración del sistema Premium Synergy*
- `__init__` (line 39) `def __init__(self)`
- `process` (line 48) `def process(self, input_data, plasticity)` - *Procesamiento con autoregulación interna*
- `__init__` (line 78) `def __init__(self)`
- `process` (line 87) `def process(self, input_data, chaos_level)` - *Procesamiento con control integrativo*
- `__init__` (line 119) `def __init__(self)`
- `process` (line 128) `def process(self, input_data, plasticity, chaos)` - *Procesamiento con regulación de fases*
- `__init__` (line 163) `def __init__(self, threshold, convergence_epochs)`
- `deliberate` (line 171) `def deliberate(self, components, target_accuracy)` - *Proceso de deliberación democrática*
- `__init__` (line 228) `def __init__(self)`
- `process_epoch` (line 241) `def process_epoch(self, input_data, chaos_level)` - *Procesa una época del sistema democrático*
- `calculate_target_accuracy` (line 296) `def calculate_target_accuracy(self)` - *Calcula accuracy objetivo basada en sinergia actual*

#### `premium_synergy_democratic.py`
**Path:** `premium_synergy_democratic.py`

**Classes:**
- `PremiumSynergyConfig` (line 42) `class PremiumSynergyConfig` - *Configuración del sistema Premium Synergy*
- `MemoryChecker` (line 94) `class MemoryChecker` - *Sistema de monitoreo de memoria*
- `TopoBrainComponent` (line 139) `class TopoBrainComponent` - *TopoBrain v8 con autoregulación interna*
- `OmniBrainComponent` (line 235) `class OmniBrainComponent` - *OmniBrain K con autoregulación interna*
- `QuimeraComponent` (line 325) `class QuimeraComponent` - *Quimera v9.5 con autoregulación interna*
- `MetabolismRegulator` (line 419) `class MetabolismRegulator` - *Regulador de metabolismo para TopoBrain*
- `SensitivityGate` (line 459) `class SensitivityGate` - *Compuerta de sensibilidad para TopoBrain*
- `DynamicTopologyGrid` (line 492) `class DynamicTopologyGrid` - *Topología dinámica para TopoBrain*
- `SymbioticBasis` (line 519) `class SymbioticBasis` - *Basis simbólica para TopoBrain*
- `IntegrationModule` (line 540) `class IntegrationModule` - *Módulo de integración para OmniBrain*
- `FastSlowLinear` (line 565) `class FastSlowLinear` - *Capa fast-slow weights para OmniBrain*
- `DualSystemModule` (line 595) `class DualSystemModule` - *Sistema dual para OmniBrain*
- `IntegrativeControl` (line 615) `class IntegrativeControl` - *Control integrativo para OmniBrain*
- `ChaosModulator` (line 648) `class ChaosModulator` - *Modulador de caos para OmniBrain*
- `LiquidNeuron` (line 682) `class LiquidNeuron` - *Neurona líquida para Quimera*
- `SovereignAttention` (line 712) `class SovereignAttention` - *Atención soberana para Quimera*
- `DualPhaseMemory` (line 738) `class DualPhaseMemory` - *Memoria de fase dual para Quimera*
- `PhaseRegulator` (line 765) `class PhaseRegulator` - *Regulador de fases para Quimera*
- `AttentionController` (line 798) `class AttentionController` - *Controlador de atención para Quimera*
- `HomeostaticMotor` (line 836) `class HomeostaticMotor` - *Motor homeostático - Cámara Alta de deliberación democrática*
- `PremiumSynergyModel` (line 960) `class PremiumSynergyModel` - *Modelo Premium Synergy con sistema democrático deliberativo*
- `SystemRegulator` (line 1074) `class SystemRegulator` - *Regulador general del sistema*

**Functions:**
- `ensure_dependencies` (line 1108) `def ensure_dependencies()` - *Asegura que las dependencias estén instaladas*
- `create_synthetic_dataset` (line 1118) `def create_synthetic_dataset(config)` - *Crea dataset sintético para testing*
- `train_premium_synergy` (line 1148) `def train_premium_synergy(config)` - *Entrena el modelo Premium Synergy*
- `create_dataloader` (line 1272) `def create_dataloader(X, y, batch_size, shuffle)` - *Crea dataloader*
- `main` (line 1284) `def main()` - *Función principal*
- `__init__` (line 97) `def __init__(self, max_memory_gb)`
- `check_memory` (line 101) `def check_memory(self)` - *Verifica el uso de memoria actual*
- `warn_if_high` (line 126) `def warn_if_high(self)` - *Advierte si el uso de memoria es alto*
- `__init__` (line 142) `def __init__(self, config)`
- `forward` (line 175) `def forward(self, x, plasticity)`
- `internal_dialogue` (line 222) `def internal_dialogue(self)` - *Diálogo interno fisiológico - metabolimo, sensibilidad, gating*
- `__init__` (line 238) `def __init__(self, config)`
- `forward` (line 268) `def forward(self, x, chaos_level)`
- `internal_dialogue` (line 312) `def internal_dialogue(self)` - *Diálogo interno - balance integrativo y modulación caótica*
- `__init__` (line 328) `def __init__(self, config)`
- `forward` (line 358) `def forward(self, x, plasticity, chaos)`
- `internal_dialogue` (line 400) `def internal_dialogue(self)` - *Diálogo interno - regulación de fases y control atencional*
- `consolidate` (line 409) `def consolidate(self)` - *SVD consolidation de liquid neurons*
- `__init__` (line 421) `def __init__(self, dim)`
- `forward` (line 431) `def forward(self, x)`
- `get_state` (line 456) `def get_state(self)`
- `__init__` (line 461) `def __init__(self, dim)`
- `forward` (line 471) `def forward(self, x)`
- `get_level` (line 489) `def get_level(self)`
- `__init__` (line 494) `def __init__(self, num_nodes, grid_size)`
- `_create_grid_mask` (line 501) `def _create_grid_mask(self)`
- `get_adjacency` (line 514) `def get_adjacency(self, plasticity)`
- `__init__` (line 521) `def __init__(self, dim, num_atoms)`
- `forward` (line 531) `def forward(self, x)`
- `__init__` (line 542) `def __init__(self, dim)`
- `forward` (line 552) `def forward(self, x)`
- `get_level` (line 562) `def get_level(self)`
- `__init__` (line 567) `def __init__(self, in_dim, out_dim)`
- `forward` (line 579) `def forward(self, x)`
- `__init__` (line 597) `def __init__(self, dim)`
- `forward` (line 604) `def forward(self, x)`
- `get_balance` (line 612) `def get_balance(self)`
- `__init__` (line 617) `def __init__(self, dim)`
- `forward` (line 627) `def forward(self, x)`
- `get_state` (line 645) `def get_state(self)`
- `__init__` (line 650) `def __init__(self, dim)`
- `forward` (line 660) `def forward(self, x, chaos_level)`
- `get_resistance` (line 678) `def get_resistance(self)`
- `__init__` (line 684) `def __init__(self, in_dim, out_dim)`
- `forward` (line 692) `def forward(self, x, plasticity)`
- `consolidate_svd` (line 703) `def consolidate_svd(self, strength)`
- `__init__` (line 714) `def __init__(self, dim)`
- `forward` (line 721) `def forward(self, x, is_chaos)`
- `get_metrics` (line 733) `def get_metrics(self)`
- `__init__` (line 740) `def __init__(self, dim)`
- `forward` (line 746) `def forward(self, x, phase_idx)`
- `update` (line 754) `def update(self, x, phase_idx)`
- `get_coherence` (line 762) `def get_coherence(self)`
- `__init__` (line 767) `def __init__(self, dim)`
- `forward` (line 777) `def forward(self, x)`
- `get_level` (line 795) `def get_level(self)`
- `__init__` (line 800) `def __init__(self, dim)`
- `forward` (line 810) `def forward(self, x, chaos)`
- `get_control` (line 829) `def get_control(self)`
- `__init__` (line 839) `def __init__(self, config)`
- `forward` (line 861) `def forward(self, topobrain_out, omnibrain_out, quimera_out, target_accuracy)` - *Cámara Alta: Delibera sobre las sinergias de los componentes

Returns:
    adjusted_output: Output ajustado por la deliberación
    metrics: Métricas del proceso deliberativo*
- `adjust_for_convergence` (line 926) `def adjust_for_convergence(self, performance_metrics)` - *Motor homeostático ajusta si las sinergias no convergen*
- `__init__` (line 963) `def __init__(self, config)`
- `forward` (line 989) `def forward(self, x, chaos_level)` - *Forward pass completo con sistema democrático*
- `democratic_deliberation_status` (line 1061) `def democratic_deliberation_status(self)` - *Estado de la deliberación democrática*
- `__init__` (line 1076) `def __init__(self, dim)`
- `forward` (line 1086) `def forward(self, x)`

#### `quen7.py`
**Path:** `quen7.py`

**Classes:**
- `Config` (line 33) `class Config`
- `DataEnvironment` (line 55) `class DataEnvironment`
- `HomeostaticRegulator` (line 89) `class HomeostaticRegulator`
- `PhysioNeuron` (line 115) `class PhysioNeuron`
- `SupConHead` (line 158) `class SupConHead`
- `MicroTopoBrain` (line 173) `class MicroTopoBrain`
- `NeuralDiagnostics` (line 215) `class NeuralDiagnostics`

**Functions:**
- `seed_everything` (line 45) `def seed_everything(seed)`
- `train_nonstationary` (line 265) `def train_nonstationary(config)`
- `run_ablation_study` (line 330) `def run_ablation_study()`
- `__init__` (line 56) `def __init__(self)`
- `get_batch` (line 66) `def get_batch(self, phase, bs)`
- `get_full` (line 80) `def get_full(self)`
- `get_w2` (line 83) `def get_w2(self)`
- `__init__` (line 90) `def __init__(self, d_in)`
- `forward` (line 100) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 116) `def __init__(self, d_in, d_out, dynamic)`
- `forward` (line 128) `def forward(self, x)`
- `__init__` (line 159) `def __init__(self, in_dim)`
- `forward` (line 167) `def forward(self, x)`
- `__init__` (line 174) `def __init__(self, config)`
- `count_parameters` (line 187) `def count_parameters(self)`
- `forward` (line 190) `def forward(self, x)`
- `__init__` (line 216) `def __init__(self)`
- `update` (line 226) `def update(self, loss, liquid_norm, physio, prediction_error)`
- `get_recent_avg` (line 234) `def get_recent_avg(self, key, n)`
- `report` (line 239) `def report(self, step, phase)`

#### `quimera.py`
**Path:** `quimera.py`

**Classes:**
- `ChimeraScientificConfig` (line 19) `class ChimeraScientificConfig`
- `RealWorldEnvironment` (line 86) `class RealWorldEnvironment`
- `LiquidNeuron` (line 114) `class LiquidNeuron` - *Componente Base: Plasticidad + Estabilidad*
- `SovereignAttention` (line 153) `class SovereignAttention` - *Atención Soberana (Identity Init)*
- `DualPhaseMemory` (line 176) `class DualPhaseMemory` - *Memoria Dual (DPM)*
- `Chimera_v9_Scientific` (line 201) `class Chimera_v9_Scientific`

**Functions:**
- `seed_everything` (line 37) `def seed_everything(seed)`
- `measure_spatial_richness` (line 47) `def measure_spatial_richness(activations)` - *Mide la diversidad espacial de las activaciones (Richness)*
- `get_structure_entropy` (line 62) `def get_structure_entropy(model)` - *Mide la entropía estructural de los pesos (Entropy)*
- `train_chimera_scientific` (line 279) `def train_chimera_scientific(config, verbose)`
- `generate_chimera_matrix` (line 361) `def generate_chimera_matrix()`
- `run_scientific_study` (line 399) `def run_scientific_study()`
- `__init__` (line 87) `def __init__(self)`
- `get_batch` (line 97) `def get_batch(self, phase, batch_size)`
- `__init__` (line 116) `def __init__(self, in_dim, out_dim)`
- `forward` (line 124) `def forward(self, x, plasticity)`
- `consolidate_svd` (line 138) `def consolidate_svd(self, strength)` - *Mecanismo de SVD (Science-ready)*
- `__init__` (line 155) `def __init__(self, dim)`
- `forward` (line 165) `def forward(self, x, is_chaos)`
- `__init__` (line 178) `def __init__(self, dim)`
- `forward` (line 184) `def forward(self, x, phase_idx)`
- `update` (line 191) `def update(self, x, phase_idx)`
- `__init__` (line 202) `def __init__(self, config)`
- `forward` (line 229) `def forward(self, x, phase_idx)`
- `consolidate` (line 267) `def consolidate(self)`

#### `quimera_vision.py`
**Path:** `quimera_vision.py`

**Classes:**
- `Flickr8kMMDataset` (line 43) `class Flickr8kMMDataset(Dataset)`
- `ImgEncoder` (line 96) `class ImgEncoder`
- `AudioEncoder` (line 108) `class AudioEncoder`
- `Decoder` (line 121) `class Decoder`

**Functions:**
- `text_to_seq` (line 37) `def text_to_seq(text)`
- `collate` (line 87) `def collate(batch)`
- `generate_caption` (line 168) `def generate_caption(img_path, audio_path)`
- `__init__` (line 44) `def __init__(self)`
- `__len__` (line 62) `def __len__(self)`
- `__getitem__` (line 64) `def __getitem__(self, idx)`
- `__init__` (line 97) `def __init__(self)`
- `forward` (line 103) `def forward(self, x)`
- `__init__` (line 109) `def __init__(self)`
- `forward` (line 116) `def forward(self, x)`
- `__init__` (line 122) `def __init__(self)`
- `forward` (line 127) `def forward(self, img, audio, seq)`

#### `qwen.py`
**Path:** `qwen.py`

**Classes:**
- `MicroConfig` (line 35) `class MicroConfig`
- `HomeostaticOrchestrator` (line 83) `class HomeostaticOrchestrator` - *Regula TODO: plasticity, continuum, supcon, symbiosis, etc.
Entradas: estado global del sistema
Salidas: controles específicos para cada componente*
- `RegulableContinuum` (line 121) `class RegulableContinuum`
- `RegulableSymbiotic` (line 143) `class RegulableSymbiotic`
- `RegulableTopology` (line 162) `class RegulableTopology`
- `RegulableSupConHead` (line 181) `class RegulableSupConHead`
- `MicroTopoBrain` (line 196) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 57) `def seed_everything(seed)`
- `get_dataset` (line 64) `def get_dataset(config)`
- `micro_pgd_attack` (line 283) `def micro_pgd_attack(model, x, y, eps, steps, pgd_loss)`
- `train_with_cv` (line 306) `def train_with_cv(config, dataset, cv_folds)`
- `generate_ablation_matrix` (line 367) `def generate_ablation_matrix()`
- `run_ablation_study` (line 392) `def run_ablation_study()`
- `__init__` (line 89) `def __init__(self)`
- `forward` (line 99) `def forward(self, x, logits, h_agg, h_proc, w_norm, entropy, ortho, pgd_loss)`
- `__init__` (line 122) `def __init__(self, dim)`
- `forward` (line 132) `def forward(self, x, strength)`
- `__init__` (line 144) `def __init__(self, dim, num_atoms)`
- `forward` (line 151) `def forward(self, x, influence)`
- `__init__` (line 163) `def __init__(self, num_nodes)`
- `get_adjacency` (line 176) `def get_adjacency(self, plasticity)`
- `__init__` (line 182) `def __init__(self, in_dim)`
- `forward` (line 190) `def forward(self, x, gain)`
- `__init__` (line 197) `def __init__(self, config)`
- `_init_weights` (line 215) `def _init_weights(self)`
- `count_parameters` (line 220) `def count_parameters(self)`
- `forward` (line 223) `def forward(self, x, pgd_loss)`

#### `qwen3.py`
**Path:** `qwen3.py`

**Classes:**
- `Config` (line 36) `class Config`
- `DataEnvironment` (line 60) `class DataEnvironment`
- `HomeostaticRegulator` (line 94) `class HomeostaticRegulator`
- `PhysioNeuron` (line 120) `class PhysioNeuron`
- `RegulableSymbiotic` (line 160) `class RegulableSymbiotic`
- `RegulableTopology` (line 179) `class RegulableTopology`
- `MicroTopoBrain` (line 201) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 50) `def seed_everything(seed)`
- `train_nonstationary` (line 263) `def train_nonstationary(config)`
- `generate_ablation_matrix` (line 320) `def generate_ablation_matrix()`
- `run_ablation_study` (line 352) `def run_ablation_study()`
- `__init__` (line 61) `def __init__(self)`
- `get_batch` (line 71) `def get_batch(self, phase, bs)`
- `get_full` (line 85) `def get_full(self)`
- `get_w2` (line 88) `def get_w2(self)`
- `__init__` (line 95) `def __init__(self, d_in)`
- `forward` (line 105) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 121) `def __init__(self, d_in, d_out, dynamic)`
- `forward` (line 132) `def forward(self, x)`
- `__init__` (line 161) `def __init__(self, dim, atoms)`
- `forward` (line 168) `def forward(self, x, influence)`
- `__init__` (line 180) `def __init__(self, num_nodes)`
- `get_adjacency` (line 193) `def get_adjacency(self, plasticity)`
- `__init__` (line 202) `def __init__(self, config)`
- `count_parameters` (line 221) `def count_parameters(self)`
- `forward` (line 224) `def forward(self, x)`

#### `qwen4.py`
**Path:** `qwen4.py`

**Classes:**
- `Config` (line 36) `class Config`
- `DataEnvironment` (line 60) `class DataEnvironment`
- `HomeostaticRegulator` (line 94) `class HomeostaticRegulator`
- `PhysioNeuron` (line 120) `class PhysioNeuron`
- `RegulableSymbiotic` (line 160) `class RegulableSymbiotic`
- `RegulableTopology` (line 179) `class RegulableTopology`
- `MicroTopoBrain` (line 201) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 50) `def seed_everything(seed)`
- `train_nonstationary` (line 263) `def train_nonstationary(config)`
- `generate_ablation_matrix` (line 325) `def generate_ablation_matrix()`
- `run_ablation_study` (line 357) `def run_ablation_study()`
- `__init__` (line 61) `def __init__(self)`
- `get_batch` (line 71) `def get_batch(self, phase, bs)`
- `get_full` (line 85) `def get_full(self)`
- `get_w2` (line 88) `def get_w2(self)`
- `__init__` (line 95) `def __init__(self, d_in)`
- `forward` (line 105) `def forward(self, x, h_pre, w_norm)`
- `__init__` (line 121) `def __init__(self, d_in, d_out, dynamic)`
- `forward` (line 132) `def forward(self, x)`
- `__init__` (line 161) `def __init__(self, dim, atoms)`
- `forward` (line 168) `def forward(self, x, influence)`
- `__init__` (line 180) `def __init__(self, num_nodes)`
- `get_adjacency` (line 193) `def get_adjacency(self, plasticity)`
- `__init__` (line 202) `def __init__(self, config)`
- `count_parameters` (line 221) `def count_parameters(self)`
- `forward` (line 224) `def forward(self, x)`

#### `qwen5.py`
**Path:** `qwen5.py`

**Classes:**
- `Config` (line 35) `class Config`
- `DataEnvironment` (line 59) `class DataEnvironment`
- `WorldModel` (line 94) `class WorldModel` - *LSTM ligero que predice la próxima fase*
- `EpisodeMemory` (line 116) `class EpisodeMemory` - *Memoria de claves-valores ligera para estados fisiológicos óptimos*
- `PredictiveHomeostat` (line 138) `class PredictiveHomeostat`
- `PredictivePhysioNeuron` (line 196) `class PredictivePhysioNeuron`
- `PhysioChimeraV15` (line 247) `class PhysioChimeraV15`

**Functions:**
- `seed_everything` (line 49) `def seed_everything(seed)`
- `train_predictive` (line 285) `def train_predictive(config)`
- `run_experiment` (line 351) `def run_experiment()`
- `__init__` (line 60) `def __init__(self)`
- `get_batch` (line 71) `def get_batch(self, phase, bs)`
- `get_full` (line 85) `def get_full(self)`
- `get_w2` (line 88) `def get_w2(self)`
- `__init__` (line 96) `def __init__(self, hidden_dim)`
- `forward` (line 103) `def forward(self, phase_id)`
- `__init__` (line 118) `def __init__(self, capacity)`
- `store` (line 122) `def store(self, phase, metrics, state)`
- `retrieve` (line 129) `def retrieve(self, phase, top_k)`
- `__init__` (line 139) `def __init__(self, d_in)`
- `forward` (line 153) `def forward(self, x, h_pre, w_norm, phase, reward)`
- `__init__` (line 197) `def __init__(self, d_in, d_out, dynamic)`
- `forward` (line 209) `def forward(self, x, phase, reward)`
- `consolidate_svd` (line 239) `def consolidate_svd(self, repair_strength)`
- `__init__` (line 248) `def __init__(self, config)`
- `count_parameters` (line 261) `def count_parameters(self)`
- `forward` (line 264) `def forward(self, x, phase, reward)`

#### `qwen6.py`
**Path:** `qwen6.py`

**Classes:**
- `Config` (line 31) `class Config`
- `DataEnvironment` (line 53) `class DataEnvironment`
- `SelfModifyingGates` (line 89) `class SelfModifyingGates`
- `ContinuumMemorySystem` (line 110) `class ContinuumMemorySystem`
- `NestedPhysioNeuron` (line 133) `class NestedPhysioNeuron`
- `PhysioChimeraNested` (line 169) `class PhysioChimeraNested`

**Functions:**
- `seed_everything` (line 43) `def seed_everything(seed)`
- `train_nested` (line 202) `def train_nested(config)`
- `run_experiment` (line 265) `def run_experiment()`
- `__init__` (line 54) `def __init__(self)`
- `get_batch` (line 64) `def get_batch(self, phase, bs)`
- `get_full` (line 78) `def get_full(self)` - *Retorna el dataset completo*
- `get_w2` (line 82) `def get_w2(self)` - *Retorna solo los datos de WORLD_2 (dígitos >= 5)*
- `__init__` (line 90) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 97) `def forward(self, x)`
- `__init__` (line 111) `def __init__(self, levels, d_model, hidden_dim)`
- `forward` (line 123) `def forward(self, x, global_step)`
- `__init__` (line 134) `def __init__(self, d_in, d_out, config)`
- `forward` (line 145) `def forward(self, x, global_step)`
- `__init__` (line 170) `def __init__(self, config)`
- `forward` (line 182) `def forward(self, x, global_step)`

#### `qwen8.py`
**Path:** `qwen8.py`

**Classes:**
- `Config` (line 33) `class Config`
- `DataEnvironment` (line 55) `class DataEnvironment`
- `HomeostaticRegulator` (line 89) `class HomeostaticRegulator`
- `PhysioNeuron` (line 116) `class PhysioNeuron`
- `SupConHead` (line 160) `class SupConHead`
- `MicroTopoBrain` (line 175) `class MicroTopoBrain`
- `NeuralDiagnostics` (line 217) `class NeuralDiagnostics`

**Functions:**
- `seed_everything` (line 45) `def seed_everything(seed)`
- `train_nonstationary` (line 267) `def train_nonstationary(config)`
- `run_ablation_study` (line 332) `def run_ablation_study()`
- `__init__` (line 56) `def __init__(self)`
- `get_batch` (line 66) `def get_batch(self, phase, bs)`
- `get_full` (line 80) `def get_full(self)`
- `get_w2` (line 83) `def get_w2(self)`
- `__init__` (line 90) `def __init__(self, d_in)`
- `forward` (line 100) `def forward(self, x, h_pre, w_norm, task_loss)`
- `__init__` (line 117) `def __init__(self, d_in, d_out, dynamic)`
- `forward` (line 129) `def forward(self, x, task_loss)`
- `__init__` (line 161) `def __init__(self, in_dim)`
- `forward` (line 169) `def forward(self, x)`
- `__init__` (line 176) `def __init__(self, config)`
- `count_parameters` (line 189) `def count_parameters(self)`
- `forward` (line 192) `def forward(self, x, task_loss)`
- `__init__` (line 218) `def __init__(self)`
- `update` (line 228) `def update(self, loss, liquid_norm, physio, prediction_error)`
- `get_recent_avg` (line 236) `def get_recent_avg(self, key, n)`
- `report` (line 241) `def report(self, step, phase)`

#### `qwen9.py`
**Path:** `qwen9.py`

**Classes:**
- `LiquidNeuron` (line 34) `class LiquidNeuron`
- `RightHemisphere` (line 78) `class RightHemisphere`
- `LeftHemisphere` (line 97) `class LeftHemisphere`
- `CorpusCallosum` (line 195) `class CorpusCallosum`
- `HomeostaticRegulator` (line 219) `class HomeostaticRegulator`
- `NeuroLogosBicameral` (line 251) `class NeuroLogosBicameral`
- `NeuralDiagnostics` (line 291) `class NeuralDiagnostics`
- `Flickr8kDataset` (line 338) `class Flickr8kDataset(Dataset)`

**Functions:**
- `build_vocab_flickr` (line 371) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 386) `def setup_flickr8k(data_dir)`
- `train_bicameral` (line 400) `def train_bicameral()`
- `__init__` (line 35) `def __init__(self, in_dim, out_dim)`
- `forward` (line 49) `def forward(self, x, global_plasticity, transfer_rate)`
- `__init__` (line 79) `def __init__(self, output_dim)`
- `forward` (line 88) `def forward(self, image, plasticity, transfer_rate)`
- `__init__` (line 98) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 119) `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)`
- `_get_init_state` (line 177) `def _get_init_state(self, visual_context)`
- `_top_p_filtering` (line 182) `def _top_p_filtering(self, logits, top_p)`
- `__init__` (line 196) `def __init__(self, dim)`
- `forward` (line 209) `def forward(self, right_features, left_context)`
- `__init__` (line 220) `def __init__(self, dim)`
- `forward` (line 230) `def forward(self, right_features, epoch)`
- `update_flow_ema` (line 245) `def update_flow_ema(self, flow)`
- `__init__` (line 252) `def __init__(self, vocab_size)`
- `forward` (line 259) `def forward(self, image, captions, epoch, return_diagnostics)`
- `__init__` (line 292) `def __init__(self)`
- `measure_callosal_flow` (line 299) `def measure_callosal_flow(self, right_features, left_context)`
- `measure_vocab_diversity` (line 306) `def measure_vocab_diversity(self, tokens, vocab_size)`
- `update` (line 312) `def update(self)`
- `get_recent_avg` (line 317) `def get_recent_avg(self, key, n)`
- `report` (line 321) `def report(self, epoch)`
- `__init__` (line 339) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 355) `def __len__(self)`
- `__getitem__` (line 358) `def __getitem__(self, idx)`

#### `qwn2.py`
**Path:** `qwn2.py`

**Classes:**
- `MicroConfig` (line 34) `class MicroConfig`
- `HomeostaticRegulator` (line 80) `class HomeostaticRegulator`
- `AutoregulatedPlasticity` (line 97) `class AutoregulatedPlasticity`
- `AutoregulatedContinuum` (line 122) `class AutoregulatedContinuum`
- `AutoregulatedSymbiotic` (line 155) `class AutoregulatedSymbiotic`
- `AutoregulatedSupConHead` (line 179) `class AutoregulatedSupConHead`
- `MicroTopoBrain` (line 199) `class MicroTopoBrain`

**Functions:**
- `seed_everything` (line 54) `def seed_everything(seed)`
- `get_dataset` (line 61) `def get_dataset(config)`
- `micro_pgd_attack` (line 259) `def micro_pgd_attack(model, x, y, eps, steps)`
- `train_with_cv` (line 279) `def train_with_cv(config, dataset, cv_folds)`
- `generate_ablation_matrix` (line 341) `def generate_ablation_matrix()`
- `run_ablation_study` (line 366) `def run_ablation_study()`
- `__init__` (line 81) `def __init__(self, input_dim)`
- `forward` (line 91) `def forward(self, signals)`
- `__init__` (line 98) `def __init__(self, num_nodes, grid_size)`
- `get_adjacency` (line 111) `def get_adjacency(self, x, h_agg)`
- `__init__` (line 123) `def __init__(self, dim)`
- `forward` (line 133) `def forward(self, x)`
- `__init__` (line 156) `def __init__(self, dim, num_atoms)`
- `forward` (line 164) `def forward(self, x)`
- `__init__` (line 180) `def __init__(self, in_dim)`
- `forward` (line 189) `def forward(self, x, entropy)`
- `__init__` (line 200) `def __init__(self, config)`
- `_init_weights` (line 215) `def _init_weights(self)`
- `count_parameters` (line 220) `def count_parameters(self)`
- `forward` (line 223) `def forward(self, x)`

#### `resma4.10.py`
**Path:** `resma4.10.py`

**Classes:**
- `RESMAConstants` (line 34) `class RESMAConstants`
- `GarnierTresTiempos` (line 71) `class GarnierTresTiempos`
- `OperadorDesdoblamiento` (line 111) `class OperadorDesdoblamiento`
- `SilencioActivoMonitor` (line 158) `class SilencioActivoMonitor`
- `QuantumLeaf` (line 186) `class QuantumLeaf`
- `RESMAUniverse` (line 233) `class RESMAUniverse`
- `NeuralNetworkRESMA` (line 341) `class NeuralNetworkRESMA`
- `MyelinCavity` (line 509) `class MyelinCavity`
- `ExperimentalPredictions` (line 545) `class ExperimentalPredictions`
- `ResourceMonitor` (line 592) `class ResourceMonitor`

**Functions:**
- `guardar_checkpoint` (line 604) `def guardar_checkpoint(data, filename)`
- `cargar_checkpoint` (line 662) `def cargar_checkpoint(filename)`
- `_make_serializable` (line 690) `def _make_serializable(obj, depth, max_depth, _visited)` - *Convierte objetos a formato serializable de forma segura.

Args:
    obj: Objeto a serializar
    depth: Nivel de profundidad actual (auto-incremental)
    max_depth: Profundidad máxima permitida
    _visited: Diccionario de objetos ya procesados (para referencias circulares)*
- `simulate_resma_garnier` (line 794) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- `verify_pt_condition` (line 50) `def verify_pt_condition(cls)`
- `__post_init__` (line 74) `def __post_init__(self)`
- `epsilon_critico` (line 86) `def epsilon_critico(self)`
- `modulation_factor` (line 89) `def modulation_factor(self)`
- `to_dict` (line 92) `def to_dict(self)`
- `from_dict` (line 102) `def from_dict(cls, data)`
- `__init__` (line 112) `def __init__(self, garnier, dimension)`
- `_construir_generadores_aleatorios` (line 121) `def _construir_generadores_aleatorios(self)`
- `_hadamard_generalizado` (line 130) `def _hadamard_generalizado(self)`
- `operator` (line 135) `def operator(self)`
- `calcular_alpha_modificado` (line 150) `def calcular_alpha_modificado(self, alpha_base)`
- `__init__` (line 159) `def __init__(self, garnier)`
- `calcular_delta_s_loop` (line 163) `def calcular_delta_s_loop(self, rho_red, b1)`
- `es_silencio_activo` (line 170) `def es_silencio_activo(self, rho_red, b1)`
- `__post_init__` (line 193) `def __post_init__(self)`
- `spectral_density` (line 197) `def spectral_density(self, omega)`
- `bures_distance` (line 203) `def bures_distance(self, other)`
- `__init__` (line 234) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `_initialize_leaves` (line 270) `def _initialize_leaves(self)`
- `_generate_complete_measure` (line 281) `def _generate_complete_measure(self)`
- `_aplicar_modulacion_garnier` (line 310) `def _aplicar_modulacion_garnier(self, measure)`
- `_construct_global_state` (line 321) `def _construct_global_state(self)`
- `_calcular_libertad` (line 330) `def _calcular_libertad(self)`
- `_calcular_coherencia` (line 333) `def _calcular_coherencia(self)`
- `__init__` (line 342) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `_generate_realistic_modular_network` (line 380) `def _generate_realistic_modular_network(self)`
- `_compute_betti_numbers` (line 454) `def _compute_betti_numbers(self)`
- `_spectral_dimension` (line 462) `def _spectral_dimension(self)`
- `_topological_ramsey` (line 484) `def _topological_ramsey(self)`
- `_calcular_rho_reducida` (line 488) `def _calcular_rho_reducida(self)`
- `_validar_axioma_6` (line 496) `def _validar_axioma_6(self)`
- `__init__` (line 510) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 527) `def _free_hamiltonian(self)`
- `_loss_potential` (line 532) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 538) `def _compute_scalar_mass(self)`
- `__init__` (line 546) `def __init__(self, universe, network, myelin)`
- `compute_log_bayes_factor` (line 551) `def compute_log_bayes_factor(self)`
- `get_memory_gb` (line 594) `def get_memory_gb()`
- `log_resources` (line 599) `def log_resources()`

#### `resma4.2.py`
**Path:** `resma4.2.py`

**Classes:**
- `RESMAConstants` (line 38) `class RESMAConstants` - *Constantes físicas RESMA 4.0 con correcciones PT-simétricas*
- `PhysicalValidator` (line 78) `class PhysicalValidator`
- `QuantumLeaf` (line 109) `class QuantumLeaf` - *Hoja L_i como estado KMS con espacio de Hilbert standard*
- `RESMAUniverse` (line 162) `class RESMAUniverse` - *Multiverso como foliación medible, memoria O(N_leaves)*
- `EmunaOperator` (line 224) `class EmunaOperator` - *P̂_E: proyección teleológica no lineal en H²(ℂ⁺)*
- `MyelinCavity` (line 291) `class MyelinCavity` - *Cavidad dieléctrica H = H₀ + iV_loss con Spin(7)*
- `NeuralNetworkRESMA` (line 351) `class NeuralNetworkRESMA` - *Conectoma NO DIRIGIDO con homología persistente*
- `ExperimentalPredictions` (line 474) `class ExperimentalPredictions` - *Predicciones con BF logarítmico*

**Functions:**
- `simulate_resma_complete` (line 556) `def simulate_resma_complete(n_leaves, n_nodes, seed)` - *Pipeline RESMA 4.2 completo*
- `verify_pt_condition` (line 67) `def verify_pt_condition(cls)` - *Verificar condición PT: κ < χΩ*
- `validate_dimension` (line 80) `def validate_dimension(alpha, tolerance)`
- `validate_pt_symmetry` (line 88) `def validate_pt_symmetry(kappa, Omega, chi)`
- `validate_connectome_size` (line 96) `def validate_connectome_size(n_nodes)`
- `validate_spectral_dimension` (line 101) `def validate_spectral_dimension(dim)`
- `__post_init__` (line 117) `def __post_init__(self)`
- `spectral_density` (line 121) `def spectral_density(self, omega)` - *ρ(ω) con regularización UV*
- `modular_entropy` (line 126) `def modular_entropy(self)` - *S = -∫ ρ log ρ dω*
- `bures_distance` (line 136) `def bures_distance(self, other)` - *Distancia de Bures W₂(ρ₁, ρ₂)*
- `haagerup_weight` (line 154) `def haagerup_weight(self)` - *Peso de Haagerup para regularización*
- `__init__` (line 165) `def __init__(self, n_leaves, seed)`
- `_initialize_leaves` (line 176) `def _initialize_leaves(self)`
- `_generate_gibbs_measure` (line 189) `def _generate_gibbs_measure(self)` - *μ(i,j) = exp(-β·W₂²(ρᵢ, ρⱼ))*
- `_construct_global_state` (line 205) `def _construct_global_state(self)` - *Estado global: pesos por hoja*
- `compute_gibbs_free_energy` (line 216) `def compute_gibbs_free_energy(self)` - *F = -ln(Tr(μ)) / β*
- `__init__` (line 227) `def __init__(self, universe, n_samples)`
- `_construct_hardy_state` (line 234) `def _construct_hardy_state(self)` - *E(z) ∈ H²(ℂ⁺)*
- `_szego_projector` (line 238) `def _szego_projector(self)` - *Proyector en frecuencias positivas*
- `_evaluation_functional` (line 246) `def _evaluation_functional(self, state_weights)` - *Φ_E[|Ψ⟩] = exp(∫ log(⟨Φᵢ|E⟩) dμ)*
- `project` (line 258) `def project(self, state_vector)` - *P̂_E = P_E ∘ Φ_E (con interpolación adaptativa)*
- `__post_init__` (line 297) `def __post_init__(self)`
- `_free_hamiltonian` (line 304) `def _free_hamiltonian(self)` - *H₀: dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 310) `def _loss_potential(self)` - *V_loss ∝ (r/a₀)^(2α)*
- `_compute_scalar_mass` (line 317) `def _compute_scalar_mass(self)` - *Campo escalar para estabilización Spin(7)*
- `_pt_symmetry_condition` (line 321) `def _pt_symmetry_condition(self)` - *κ < χΩ*
- `coherence_quantum` (line 327) `def coherence_quantum(self)` - *Coherencia cuántica con verificación espectral*
- `__init__` (line 354) `def __init__(self, n_nodes, seed)`
- `_generate_fractal_graph` (line 366) `def _generate_fractal_graph(self)` - *Scale-free → NO DIRIGIDO*
- `_spectral_dimension` (line 381) `def _spectral_dimension(self)` - *d_s = -2 lim log N(λ)/log λ*
- `_topological_ramsey` (line 410) `def _topological_ramsey(self)` - *R_Q(G) = min{n | β_{n-1}(G) > 0}*
- `_compute_betti_numbers` (line 429) `def _compute_betti_numbers(self)` - *Números de Betti β₀, β₁*
- `_graph_to_distance_matrix` (line 445) `def _graph_to_distance_matrix(self)` - *Matriz de distancias para homología*
- `critical_percolation_time` (line 461) `def critical_percolation_time(self)` - *t_c = 21 · (N/N₀)^0.25 / log R_Q*
- `__init__` (line 477) `def __init__(self, universe, myelin, network)`
- `predict_all` (line 483) `def predict_all(self)` - *Predicciones RESMA 4.2*
- `compute_log_bayes_factor` (line 496) `def compute_log_bayes_factor(self)` - *ln(BF) con AIC*

#### `resma4.3.py`
**Path:** `resma4.3.py`

**Classes:**
- `ResourceMonitor` (line 33) `class ResourceMonitor`
- `RESMAConstants` (line 110) `class RESMAConstants`
- `QuantumLeaf` (line 142) `class QuantumLeaf` - *Hoja KMS - INMUTABLE pero con caché externo*
- `RESMAUniverse` (line 190) `class RESMAUniverse` - *Multiverso con construcción lazy*
- `PhysicalValidator` (line 269) `class PhysicalValidator`
- `MyelinCavity` (line 297) `class MyelinCavity`
- `NeuralNetworkRESMA` (line 352) `class NeuralNetworkRESMA`
- `ExperimentalPredictions` (line 485) `class ExperimentalPredictions`

**Functions:**
- `guardar_checkpoint` (line 54) `def guardar_checkpoint(data, filename)` - *Guardado atómico con backup*
- `cargar_checkpoint` (line 84) `def cargar_checkpoint(filename)` - *Cargar checkpoint con fallback*
- `simulate_resma_with_checkpointing` (line 545) `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)` - *Pipeline con reanudación inteligente desde checkpoints*
- `get_memory_gb` (line 35) `def get_memory_gb()`
- `check_memory_limit` (line 40) `def check_memory_limit()`
- `log_resources` (line 49) `def log_resources()`
- `verify_pt_condition` (line 127) `def verify_pt_condition(cls)`
- `__post_init__` (line 150) `def __post_init__(self)`
- `spectral_density` (line 154) `def spectral_density(self, omega)`
- `bures_distance` (line 158) `def bures_distance(self, other)` - *Distancia Bures con caché EXTERNO (no en instancia)*
- `__init__` (line 193) `def __init__(self, n_leaves, seed)`
- `_initialize_leaves` (line 215) `def _initialize_leaves(self)`
- `_generate_gibbs_measure` (line 227) `def _generate_gibbs_measure(self)` - *Matriz de medida con guardado incremental*
- `_construct_global_state` (line 254) `def _construct_global_state(self)`
- `validate_dimension` (line 271) `def validate_dimension(alpha, tolerance)`
- `validate_pt_symmetry` (line 279) `def validate_pt_symmetry(kappa, Omega, chi)`
- `validate_connectome_size` (line 287) `def validate_connectome_size(n_nodes)`
- `validate_spectral_dimension` (line 292) `def validate_spectral_dimension(dim)`
- `__post_init__` (line 302) `def __post_init__(self)`
- `_free_hamiltonian` (line 312) `def _free_hamiltonian(self)`
- `_loss_potential` (line 317) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 323) `def _compute_scalar_mass(self)`
- `_pt_symmetry_condition` (line 326) `def _pt_symmetry_condition(self)`
- `coherence_quantum` (line 331) `def coherence_quantum(self)`
- `__init__` (line 353) `def __init__(self, n_nodes, seed)`
- `_generate_fractal_graph` (line 372) `def _generate_fractal_graph(self)` - *Generar grafo por lotes*
- `_spectral_dimension` (line 401) `def _spectral_dimension(self)` - *Dimensión espectral con matriz sparse*
- `_topological_ramsey` (line 425) `def _topological_ramsey(self)` - *Ramsey topológico*
- `_compute_betti_numbers` (line 444) `def _compute_betti_numbers(self)` - *Números de Betti*
- `_graph_to_distance_matrix` (line 460) `def _graph_to_distance_matrix(self)` - *Matriz de distancias sparse*
- `critical_percolation_time` (line 476) `def critical_percolation_time(self)` - *Tiempo crítico de percolación*
- `__init__` (line 486) `def __init__(self, universe, myelin, network)`
- `compute_log_bayes_factor` (line 492) `def compute_log_bayes_factor(self)` - *ln(BF)*

#### `resma4.4.py`
**Path:** `resma4.4.py`

**Classes:**
- `ResourceMonitor` (line 34) `class ResourceMonitor`
- `RESMAConstants` (line 129) `class RESMAConstants`
- `QuantumLeaf` (line 160) `class QuantumLeaf` - *Hoja KMS - INMUTABLE*
- `RESMAUniverse` (line 203) `class RESMAUniverse` - *Multiverso con estado serializable*
- `PhysicalValidator` (line 316) `class PhysicalValidator`
- `MyelinCavity` (line 343) `class MyelinCavity`
- `NeuralNetworkRESMA` (line 377) `class NeuralNetworkRESMA`

**Functions:**
- `guardar_checkpoint` (line 59) `def guardar_checkpoint(data, filename)` - *Guarda el estado COMPLETO de los objetos, no solo metadatos*
- `cargar_checkpoint` (line 93) `def cargar_checkpoint(filename)` - *Carga el estado COMPLETO desde disco*
- `simulate_resma_with_checkpointing` (line 554) `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)` - *Pipeline con reanudación que realmente carga objetos*
- `get_memory_gb` (line 36) `def get_memory_gb()`
- `check_memory_limit` (line 41) `def check_memory_limit()`
- `log_resources` (line 50) `def log_resources()`
- `verify_pt_condition` (line 146) `def verify_pt_condition(cls)`
- `__post_init__` (line 168) `def __post_init__(self)`
- `spectral_density` (line 172) `def spectral_density(self, omega)`
- `bures_distance` (line 176) `def bures_distance(self, other)` - *Distancia Bures con caché externo*
- `__init__` (line 206) `def __init__(self, n_leaves, seed, leaves, measure, global_state)` - *Constructor que puede recibir estado serializado*
- `_initialize_leaves` (line 258) `def _initialize_leaves(self)`
- `_generate_gibbs_measure` (line 269) `def _generate_gibbs_measure(self)` - *Matriz de medida*
- `_construct_global_state` (line 301) `def _construct_global_state(self)`
- `validate_dimension` (line 318) `def validate_dimension(alpha, tolerance)`
- `validate_pt_symmetry` (line 326) `def validate_pt_symmetry(kappa, Omega, chi)`
- `validate_connectome_size` (line 334) `def validate_connectome_size(n_nodes)`
- `validate_spectral_dimension` (line 339) `def validate_spectral_dimension(dim)`
- `__post_init__` (line 348) `def __post_init__(self)`
- `_free_hamiltonian` (line 358) `def _free_hamiltonian(self)`
- `_loss_potential` (line 363) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 369) `def _compute_scalar_mass(self)`
- `_pt_symmetry_condition` (line 372) `def _pt_symmetry_condition(self)`
- `__init__` (line 378) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti)` - *Constructor que puede recibir grafo ya construido*
- `_generate_fractal_graph` (line 446) `def _generate_fractal_graph(self)` - *Generar grafo por lotes*
- `_spectral_dimension` (line 475) `def _spectral_dimension(self)` - *Dimensión espectral con eigenvalores sparse*
- `_topological_ramsey` (line 499) `def _topological_ramsey(self)` - *Ramsey topológico*
- `_compute_betti_numbers` (line 518) `def _compute_betti_numbers(self)` - *Números de Betti*
- `_graph_to_distance_matrix` (line 534) `def _graph_to_distance_matrix(self)` - *Matriz de distancias sparse*

#### `resma4.5.py`
**Path:** `resma4.5.py`

**Classes:**
- `ResourceMonitor` (line 34) `class ResourceMonitor`
- `RESMAConstants` (line 135) `class RESMAConstants`
- `GarnierTresTiempos` (line 163) `class GarnierTresTiempos` - *Toro temporal T³ con parámetros ADIMENSIONALES.
C0, C2, C3 son ratios de escala, no velocidades.*
- `OperadorDesdoblamiento` (line 201) `class OperadorDesdoblamiento` - *D̂_G(ϕ) = exp(i Σ_i φ_i H_i) · H_E
Representación toy de E8 (248x248)*
- `SilencioActivoMonitor` (line 269) `class SilencioActivoMonitor` - *Monitor de Silencio-Activo: ΔS_loop < ε_c(ϕ)*
- `QuantumLeaf` (line 337) `class QuantumLeaf` - *Hoja KMS - INMUTABLE (SIN CAMBIOS)*
- `RESMAUniverse` (line 380) `class RESMAUniverse` - *Multiverso con estado serializable y desdoblamiento Garnier*
- `MyelinCavity` (line 486) `class MyelinCavity` - *Cavidad PT-simétrica (SIN CAMBIOS)*
- `NeuralNetworkRESMA` (line 517) `class NeuralNetworkRESMA` - *Red neuronal con embedding Garnier*
- `ExperimentalPredictions` (line 658) `class ExperimentalPredictions` - *Cálculos experimentales (SIN CAMBIOS)*

**Functions:**
- `guardar_checkpoint` (line 59) `def guardar_checkpoint(data, filename)`
- `cargar_checkpoint` (line 88) `def cargar_checkpoint(filename)`
- `_make_serializable` (line 113) `def _make_serializable(obj)` - *Convierte objetos a formato serializable*
- `simulate_resma_garnier` (line 695) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)` - *Pipeline único con Garnier integrado*
- `get_memory_gb` (line 36) `def get_memory_gb()`
- `check_memory_limit` (line 41) `def check_memory_limit()`
- `log_resources` (line 50) `def log_resources()`
- `verify_pt_condition` (line 152) `def verify_pt_condition(cls)`
- `__post_init__` (line 170) `def __post_init__(self)`
- `factor_escala` (line 181) `def factor_escala(self, tiempo_idx)` - *Factor de escala para cada tiempo: 0=lento, 2=modular, 3=teleológico*
- `epsilon_critico` (line 185) `def epsilon_critico(self)` - *Entropía crítica de percolación (ADIMENSIONAL).
log(2) es la entropía de un bit cuántico crítico.*
- `to_dict` (line 192) `def to_dict(self)` - *Para serialización*
- `from_dict` (line 197) `def from_dict(cls, data)`
- `__init__` (line 206) `def __init__(self, garnier, dimension)`
- `_construir_generadores_E8` (line 214) `def _construir_generadores_E8(self)` - *Construye 3 generadores temporales (antis-Hermitianos)*
- `_hadamard_generalizado` (line 226) `def _hadamard_generalizado(self)` - *Operador de Hadamard en dimensión 248 (unitario)*
- `operator` (line 235) `def operator(self)` - *Construye D̂_G(ϕ) dimensionalmente consistente*
- `aplicar_a_estado` (line 254) `def aplicar_a_estado(self, estado)` - *Aplica desdoblamiento a un estado cuántico |Ψ⟩*
- `calcular_alpha_modificado` (line 260) `def calcular_alpha_modificado(self, alpha_base)` - *α'(ϕ) = α · tanh(C0/C3 · cos(ϕ₃))
Garantiza α' ∈ [0, α]*
- `__init__` (line 273) `def __init__(self, garnier, network)`
- `calcular_delta_s_loop` (line 278) `def calcular_delta_s_loop(self, rho_red)` - *ΔS_loop = S_vN(ρ_red) - log(b₁ + 1)
rho_red: matriz densidad reducida (si es None, se calcula)*
- `_calcular_rho_reducida_aproximada` (line 299) `def _calcular_rho_reducida_aproximada(self)` - *Aproximación: ρ_red = diag(grados) / sum(grados)*
- `es_silencio_activo` (line 307) `def es_silencio_activo(self, rho_red)` - *Verifica Silencio-Activo y calcula Libertad L.
Retorna: (condicion, libertad_L)*
- `umbral_percolacion` (line 324) `def umbral_percolacion(self)` - *Umbral de percolación para soberanía: 70% (Axioma 6)*
- `__post_init__` (line 345) `def __post_init__(self)`
- `spectral_density` (line 349) `def spectral_density(self, omega)`
- `bures_distance` (line 353) `def bures_distance(self, other)`
- `__init__` (line 383) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)` - *Constructor que puede recibir estado serializado*
- `_initialize_leaves` (line 420) `def _initialize_leaves(self)`
- `_generate_gibbs_measure` (line 431) `def _generate_gibbs_measure(self)` - *Matriz de medida sin desdoblamiento*
- `_aplicar_desdoblamiento_a_medida` (line 451) `def _aplicar_desdoblamiento_a_medida(self, measure)` - *Aplica D̂_G(ϕ) a la medida:
- M_ij → M_ij * (C0/C3)^(cos(ϕ₃))
- Normaliza después*
- `_construct_global_state` (line 471) `def _construct_global_state(self)`
- `_calcular_libertad_universo` (line 481) `def _calcular_libertad_universo(self)` - *Libertad del universo: L = 1/ε_c*
- `__init__` (line 488) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 499) `def _free_hamiltonian(self)`
- `_loss_potential` (line 504) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 510) `def _compute_scalar_mass(self)`
- `_pt_symmetry_condition` (line 513) `def _pt_symmetry_condition(self)`
- `__init__` (line 520) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)` - *Constructor que puede recibir grafo ya construido*
- `_generate_fractal_graph` (line 559) `def _generate_fractal_graph(self)` - *Generar grafo por lotes con conectividad controlada*
- `_spectral_dimension` (line 591) `def _spectral_dimension(self)` - *Dimensión espectral con eigenvalores sparse*
- `_topological_ramsey` (line 615) `def _topological_ramsey(self)` - *Ramsey topológico simplificado*
- `_compute_betti_numbers` (line 627) `def _compute_betti_numbers(self)` - *Números de Betti aproximados por ciclos locales*
- `_calcular_rho_reducida` (line 636) `def _calcular_rho_reducida(self)` - *Matriz densidad reducida del conectoma*
- `validar_axioma_6` (line 644) `def validar_axioma_6(self)` - *Verifica: conectividad > 70% para soberanía*
- `__init__` (line 661) `def __init__(self, universe, myelin, network)`
- `compute_log_bayes_factor` (line 666) `def compute_log_bayes_factor(self)` - *Calcula Factor de Bayes integrando Garnier*

#### `resma4.6.py`
**Path:** `resma4.6.py`

**Classes:**
- `ResourceMonitor` (line 31) `class ResourceMonitor`
- `RESMAConstants` (line 139) `class RESMAConstants` - *Constantes físicas fundamentales*
- `GarnierTresTiempos` (line 169) `class GarnierTresTiempos` - ***TORO TEMPORAL T³ CON CANCELACIÓN ZPE**
- phi: Fase de desdoblamiento que controla anulación ZPE
- zpe_level: Nivel de fluctuaciones de punto cero [0,1]*
- `OperadorDesdoblamiento` (line 233) `class OperadorDesdoblamiento` - ***D̂_G(ϕ) = exp(i Σ_i φ_i H_i) · H_E**
**CONTRA-ZPE**: Opera en subespacio sin fluctuaciones*
- `SilencioActivoMonitor` (line 312) `class SilencioActivoMonitor` - ***MONITOR DE ANTAGONISMO ZPE-SILENCIO**
- Detecta cuando fluctuaciones cuánticas son coherentemente anuladas
- Mide nivel de "ruido de fondo cuántico" vs "silencio ontológico"*
- `QuantumLeaf` (line 441) `class QuantumLeaf` - *Hoja KMS - INMUTABLE*
- `RESMAUniverse` (line 484) `class RESMAUniverse` - *Multiverso con ZPE-Silencio integrado*
- `MyelinCavity` (line 608) `class MyelinCavity` - *Cavidad PT-simétrica con medición ZPE*
- `ConectomaCuantico` (line 659) `class ConectomaCuantico` - ***SUPERPOSICIÓN CUÁNTICA DE GRAFOS** (Pre-geométrico)
- No es un grafo, es una matriz de amplitudes
- Colapsa a grafo clásico solo bajo medición
- b1_cuántico ≠ b1_clásico*
- `NeuralNetworkRESMA` (line 800) `class NeuralNetworkRESMA` - *Red neuronal con **conectoma cuántico** subyacente*
- `ExperimentalPredictions` (line 898) `class ExperimentalPredictions` - *Cálculos experimentales unificados ZPE-Silencio*

**Functions:**
- `guardar_checkpoint` (line 56) `def guardar_checkpoint(data, filename)` - *Guarda estado completo con manejo robusto de errores*
- `cargar_checkpoint` (line 86) `def cargar_checkpoint(filename)` - *Carga checkpoint con fallback automático*
- `_make_serializable` (line 112) `def _make_serializable(obj)` - *Convierte objetos recursivamente a formato serializable*
- `simulate_resma_garnier` (line 951) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart, target_connectivity)` - *Pipeline completo RESMA 4.3.5 con antagonismo ZPE-Silencio*
- `get_memory_gb` (line 33) `def get_memory_gb()`
- `check_memory_limit` (line 38) `def check_memory_limit(threshold)`
- `log_resources` (line 47) `def log_resources()`
- `verify_pt_condition` (line 158) `def verify_pt_condition(cls)`
- `__post_init__` (line 181) `def __post_init__(self)`
- `factor_escala` (line 194) `def factor_escala(self, tiempo_idx)` - *Factor de escala con supresión ZPE*
- `epsilon_critico` (line 200) `def epsilon_critico(self)` - ***UMBRAL CRÍTICO CON ZPE**:
Cuando zpe_level → 0, ε_c → 0 (Silencio perfecto no necesita umbral)*
- `to_dict` (line 208) `def to_dict(self)` - *Serialización completa*
- `from_dict` (line 220) `def from_dict(cls, data)` - *Deserialización*
- `__init__` (line 238) `def __init__(self, garnier, dimension)`
- `_construir_generadores_E8_ZPE` (line 246) `def _construir_generadores_E8_ZPE(self)` - *GENERADORES CON CANCELACIÓN ZPE INTEGRADA*
- `_hadamard_generalizado_ZPE` (line 266) `def _hadamard_generalizado_ZPE(self)` - *HADAMARD CON ESPACIO NULO ZPE*
- `operator` (line 284) `def operator(self)` - *Construye D̂_G(ϕ) con cancelación ZPE*
- `alpha_modificado` (line 304) `def alpha_modificado(self, alpha_base)` - ***α'(ϕ) = α · tanh(C₀/C₃ · cos(ϕ₃) · (1 - zpe_level))***
- `__init__` (line 318) `def __init__(self, garnier, network)`
- `calcular_delta_s_loop` (line 327) `def calcular_delta_s_loop(self, rho_red)` - ***ΔS_loop = S_vN(ρ_red) - S_top + S_ZPE**
**NUEVO**: La entropía ZPE se SUMA a la entropía total*
- `_calcular_rho_reducida_aproximada` (line 367) `def _calcular_rho_reducida_aproximada(self)` - *Matriz densidad con modulación ZPE*
- `es_silencio_activo` (line 383) `def es_silencio_activo(self, rho_red)` - ***DETECCIÓN DE ANTAGONISMO**:
Retorna: (condicion, libertad_L, nivel_ZPE_cancelado)

**CONDICIÓN**: ZPE < 1% AND ΔS_loop < ε_c*
- `umbral_percolacion` (line 411) `def umbral_percolacion(self)` - *Umbral para soberanía: 70%*
- `modo_goldstone` (line 415) `def modo_goldstone(self)` - ***MODO GOLDSTONE DEL DOBLE CUÁNTICO**:
Excitación colectiva que anuncia ruptura de simetría ZPE*
- `__post_init__` (line 449) `def __post_init__(self)`
- `spectral_density` (line 453) `def spectral_density(self, omega)`
- `bures_distance` (line 457) `def bures_distance(self, other)`
- `__init__` (line 487) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `_initialize_leaves` (line 521) `def _initialize_leaves(self)` - *Inicializa hojas con temperatura efectiva afectada por ZPE*
- `_generate_gibbs_measure` (line 535) `def _generate_gibbs_measure(self)` - *Genera medida de Gibbs*
- `_aplicar_desdoblamiento_a_medida` (line 557) `def _aplicar_desdoblamiento_a_medida(self, measure)` - *Aplica desdoblamiento con supresión ZPE*
- `_construct_global_state` (line 584) `def _construct_global_state(self)` - *Construye estado global normalizado*
- `_calcular_libertad_universo` (line 603) `def _calcular_libertad_universo(self)` - *Libertad intrínseca con supresión ZPE*
- `__init__` (line 611) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 624) `def _free_hamiltonian(self)` - *Hamiltoniano con energía ZPE incluida*
- `_loss_potential` (line 633) `def _loss_potential(self)` - *Potencial de pérdida PT*
- `_compute_scalar_mass` (line 640) `def _compute_scalar_mass(self)`
- `_calcular_zpe` (line 643) `def _calcular_zpe(self)` - ***ENERGÍA DE PUNTO CERO TOTAL**:
E_ZPE = Σ_i ½ħω_i*
- `_pt_symmetry_condition` (line 655) `def _pt_symmetry_condition(self)`
- `__init__` (line 667) `def __init__(self, n_nodes, seed, garnier)`
- `_inicializar_amplitudes` (line 686) `def _inicializar_amplitudes(self)` - ***AMPLITUDES DE FEYNMAN** para cada posible arista:
- |A_ij|² es probabilidad de existencia de arista
- Fase ϕ_ij controlada por Garnier*
- `_calcular_conectividad_cuantica` (line 708) `def _calcular_conectividad_cuantica(self)` - ***CONECTIVIDAD CUÁNTICA** (no clásica):
= Σ_i<j |A_ij|² / (N(N-1)/2)*
- `colapsar_a_clasico` (line 718) `def colapsar_a_clasico(self, threshold)` - ***COLAPSO CUÁNTICO-CLÁSICO**:
- Medición proyectiva con umbral de probabilidad
- b1_clásico ≠ b1_cuántico*
- `_recalcular_betti_clasicos` (line 742) `def _recalcular_betti_clasicos(self)` - *Recalcula Betti del grafo colapsado*
- `medir_delta_s_loop` (line 756) `def medir_delta_s_loop(self)` - ***ΔS_loop CUÁNTICO** (no clásico):
- Usa matriz densidad de amplitudes (no grafo)
- S_ZPE es intrínseca a la superposición*
- `_calcular_b1_cuantico` (line 784) `def _calcular_b1_cuantico(self)` - ***b₁ CUÁNTICO** (topología pre-geométrica):
= rango de la matriz de amplitudes (conectividad cuántica)*
- `__init__` (line 803) `def __init__(self, n_nodes, seed, conectoma_quantum, garnier)`
- `_calcular_rho_reducida` (line 842) `def _calcular_rho_reducida(self)` - ***Matriz densidad reducida del conectoma cuántico**
- Usa amplitudes, no grafo colapsado*
- `validar_axioma_6_cuantico` (line 858) `def validar_axioma_6_cuantico(self)` - ***AXIOMA 6 CUÁNTICO**: Conectividad cuántica > 70%
**NUEVO**: La soberanía se juzga en el estado pre-geométrico, no en el colapso*
- `obtener_metricas_cuanticas` (line 874) `def obtener_metricas_cuanticas(self)` - ***MÉTRICAS EXPERIMENTALES** (falsables):
- Conectividad cuántica (pre-observación)
- b1 cuántico vs b1 clásico
- Ratio de colapso: cuánto cambia la topología*
- `__init__` (line 901) `def __init__(self, universe, myelin, network)`
- `compute_log_bayes_factor` (line 906) `def compute_log_bayes_factor(self)` - *Calcula Factor de Bayes con antagonismo ZPE-Silencio*

#### `resma4.7.py`
**Path:** `resma4.7.py`

**Classes:**
- `RESMAConstants` (line 23) `class RESMAConstants`
- `GarnierTresTiempos` (line 46) `class GarnierTresTiempos` - *Toro temporal T³ con parámetros físicamente consistentes.
Basado en la teoría del desdoblamiento del tiempo de Garnier-Malet.*
- `OperadorDesdoblamiento` (line 94) `class OperadorDesdoblamiento` - *Operador de desdoblamiento D̂_G(φ) con estructura E8 simplificada*
- `SilencioActivoMonitor` (line 139) `class SilencioActivoMonitor` - *Monitor de condición de Silencio-Activo: ΔS_loop < ε_c(φ)*
- `QuantumLeaf` (line 190) `class QuantumLeaf`
- `RESMAUniverse` (line 223) `class RESMAUniverse` - *Multiverso cuántico con desdoblamiento Garnier-Malet*
- `NeuralNetworkRESMA` (line 336) `class NeuralNetworkRESMA` - *Red neuronal con topología realista que satisface Axioma 6*
- `ExperimentalPredictions` (line 484) `class ExperimentalPredictions` - *Cálculo de Factor de Bayes y predicciones*

**Functions:**
- `simulate_resma_garnier` (line 537) `def simulate_resma_garnier(n_leaves, n_nodes, seed)` - *Pipeline completo RESMA-Garnier con correcciones*
- `__post_init__` (line 53) `def __post_init__(self)`
- `_compute_coupling` (line 67) `def _compute_coupling(self)` - *Fuerza de acoplamiento entre tiempos*
- `factor_escala` (line 72) `def factor_escala(self, tiempo_idx)` - *Factor de escala temporal*
- `epsilon_critico` (line 77) `def epsilon_critico(self)` - *Entropía crítica con corrección de acoplamiento:
ε_c = log(2) · (C0/C3)² · (1 + ξ)*
- `modulation_factor` (line 85) `def modulation_factor(self)` - *Factor de modulación para la medida cuántica:
M = exp(-|φ₃ - π|/C3)
Máximo cuando φ₃ ≈ π (apertura temporal óptima)*
- `__init__` (line 98) `def __init__(self, garnier, dimension)`
- `_construir_generadores` (line 103) `def _construir_generadores(self)` - *Generadores temporales (anti-Hermitianos normalizados)*
- `operator` (line 115) `def operator(self)` - *Construye D̂_G(φ) = exp(i Σ φᵢHᵢ)*
- `aplicar_modulacion` (line 120) `def aplicar_modulacion(self, state_vector)` - *Aplica desdoblamiento a vector de estado*
- `calcular_alpha_modificado` (line 126) `def calcular_alpha_modificado(self, alpha_base)` - *α'(φ) = α · |cos(φ₃)|^(C0/C3)
Garantiza α' ∈ [0, α]*
- `__init__` (line 143) `def __init__(self, garnier)`
- `calcular_delta_s_loop` (line 147) `def calcular_delta_s_loop(self, rho_red, b1)` - *ΔS_loop = S_vN(ρ) - log(b₁ + 1)

Args:
    rho_red: Matriz densidad reducida
    b1: Primer número de Betti (ciclos independientes)*
- `es_silencio_activo` (line 166) `def es_silencio_activo(self, rho_red, b1)` - *Verifica condición y calcula libertad L = 1/(ΔS + ε_c)

Returns:
    (condicion_satisfecha, libertad)*
- `spectral_density` (line 197) `def spectral_density(self, omega)`
- `bures_distance` (line 203) `def bures_distance(self, other)` - *Distancia de Bures simplificada*
- `__init__` (line 226) `def __init__(self, n_leaves, seed, garnier)`
- `_initialize_leaves` (line 252) `def _initialize_leaves(self)` - *Genera hojas con gaps distribuidos exponencialmente*
- `_generate_modulated_measure` (line 265) `def _generate_modulated_measure(self)` - *Genera medida de transición modulada por Garnier:
M_ij = exp(-β d²_ij) · φ(garnier)*
- `_construct_global_state` (line 312) `def _construct_global_state(self)` - *Estado global como distribución diagonal*
- `_calcular_libertad` (line 322) `def _calcular_libertad(self)` - *Libertad del universo: L_U = 1/ε_c*
- `_calcular_coherencia` (line 326) `def _calcular_coherencia(self)` - *Coherencia cuántica: suma de elementos off-diagonal*
- `__init__` (line 341) `def __init__(self, n_nodes, seed, garnier)`
- `_generate_realistic_network` (line 379) `def _generate_realistic_network(self)` - *Genera red con conectividad > 70% usando modelo realista:
- Watts-Strogatz para mundo pequeño
- Aumentación para alcanzar umbral*
- `_compute_betti_numbers` (line 424) `def _compute_betti_numbers(self)` - *Números de Betti: b0=componentes, b1=ciclos*
- `_spectral_dimension` (line 433) `def _spectral_dimension(self)` - *Dimensión espectral del Laplaciano*
- `_topological_ramsey` (line 455) `def _topological_ramsey(self)` - *Número de Ramsey topológico*
- `_calcular_rho_reducida` (line 460) `def _calcular_rho_reducida(self)` - *Matriz densidad de la red (normalizada por grados)*
- `_validar_axioma_6` (line 470) `def _validar_axioma_6(self)` - *Verifica conectividad > 70%*
- `__init__` (line 487) `def __init__(self, universe, network)`
- `compute_log_bayes_factor` (line 491) `def compute_log_bayes_factor(self)` - *ln(BF) ∝ log(L_red · L_univ)
Veredicto basado en libertad total*

#### `resma4.8.py`
**Path:** `resma4.8.py`

**Classes:**
- `RESMAConstants` (line 37) `class RESMAConstants`
- `ResourceMonitor` (line 70) `class ResourceMonitor`
- `GarnierTresTiempos` (line 158) `class GarnierTresTiempos`
- `OperadorDesdoblamiento` (line 209) `class OperadorDesdoblamiento`
- `SilencioActivoMonitor` (line 256) `class SilencioActivoMonitor`
- `QuantumLeaf` (line 284) `class QuantumLeaf`
- `RESMAUniverse` (line 331) `class RESMAUniverse`
- `NeuralNetworkRESMA` (line 433) `class NeuralNetworkRESMA`
- `MyelinCavity` (line 584) `class MyelinCavity`
- `ExperimentalPredictions` (line 620) `class ExperimentalPredictions`

**Functions:**
- `guardar_checkpoint` (line 91) `def guardar_checkpoint(data, filename)`
- `cargar_checkpoint` (line 117) `def cargar_checkpoint(filename)`
- `_make_serializable` (line 141) `def _make_serializable(obj)`
- `simulate_resma_garnier` (line 667) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- `verify_pt_condition` (line 54) `def verify_pt_condition(cls)`
- `get_memory_gb` (line 72) `def get_memory_gb()`
- `check_memory_limit` (line 77) `def check_memory_limit()`
- `log_resources` (line 86) `def log_resources()`
- `__post_init__` (line 161) `def __post_init__(self)`
- `_compute_coupling` (line 179) `def _compute_coupling(self)`
- `epsilon_critico` (line 182) `def epsilon_critico(self)`
- `modulation_factor` (line 186) `def modulation_factor(self)`
- `to_dict` (line 189) `def to_dict(self)`
- `from_dict` (line 199) `def from_dict(cls, data)`
- `__init__` (line 210) `def __init__(self, garnier, dimension)`
- `_construir_generadores_aleatorios` (line 219) `def _construir_generadores_aleatorios(self)`
- `_hadamard_generalizado` (line 228) `def _hadamard_generalizado(self)`
- `operator` (line 233) `def operator(self)`
- `calcular_alpha_modificado` (line 248) `def calcular_alpha_modificado(self, alpha_base)`
- `__init__` (line 257) `def __init__(self, garnier)`
- `calcular_delta_s_loop` (line 261) `def calcular_delta_s_loop(self, rho_red, b1)`
- `es_silencio_activo` (line 268) `def es_silencio_activo(self, rho_red, b1)`
- `__post_init__` (line 291) `def __post_init__(self)`
- `spectral_density` (line 295) `def spectral_density(self, omega)`
- `bures_distance` (line 301) `def bures_distance(self, other)`
- `__init__` (line 332) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `_initialize_leaves` (line 362) `def _initialize_leaves(self)`
- `_generate_complete_measure` (line 373) `def _generate_complete_measure(self)`
- `_aplicar_modulacion_garnier` (line 402) `def _aplicar_modulacion_garnier(self, measure)`
- `_construct_global_state` (line 413) `def _construct_global_state(self)`
- `_calcular_libertad` (line 422) `def _calcular_libertad(self)`
- `_calcular_coherencia` (line 425) `def _calcular_coherencia(self)`
- `__init__` (line 434) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `_generate_realistic_modular_network` (line 472) `def _generate_realistic_modular_network(self)`
- `_compute_betti_numbers` (line 529) `def _compute_betti_numbers(self)`
- `_spectral_dimension` (line 537) `def _spectral_dimension(self)`
- `_topological_ramsey` (line 559) `def _topological_ramsey(self)`
- `_calcular_rho_reducida` (line 563) `def _calcular_rho_reducida(self)`
- `_validar_axioma_6` (line 571) `def _validar_axioma_6(self)`
- `__init__` (line 585) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 602) `def _free_hamiltonian(self)`
- `_loss_potential` (line 607) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 613) `def _compute_scalar_mass(self)`
- `__init__` (line 621) `def __init__(self, universe, network, myelin)`
- `compute_log_bayes_factor` (line 626) `def compute_log_bayes_factor(self)`

#### `resma4.9.py`
**Path:** `resma4.9.py`

**Classes:**
- `RESMAConstants` (line 34) `class RESMAConstants`
- `GarnierTresTiempos` (line 71) `class GarnierTresTiempos`
- `OperadorDesdoblamiento` (line 111) `class OperadorDesdoblamiento`
- `SilencioActivoMonitor` (line 158) `class SilencioActivoMonitor`
- `QuantumLeaf` (line 186) `class QuantumLeaf`
- `RESMAUniverse` (line 233) `class RESMAUniverse`
- `NeuralNetworkRESMA` (line 341) `class NeuralNetworkRESMA`
- `MyelinCavity` (line 492) `class MyelinCavity`
- `ExperimentalPredictions` (line 528) `class ExperimentalPredictions`
- `ResourceMonitor` (line 575) `class ResourceMonitor`

**Functions:**
- `guardar_checkpoint` (line 587) `def guardar_checkpoint(data, filename)`
- `cargar_checkpoint` (line 615) `def cargar_checkpoint(filename)`
- `_make_serializable` (line 639) `def _make_serializable(obj)`
- `simulate_resma_garnier` (line 655) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- `verify_pt_condition` (line 50) `def verify_pt_condition(cls)`
- `__post_init__` (line 74) `def __post_init__(self)`
- `epsilon_critico` (line 86) `def epsilon_critico(self)`
- `modulation_factor` (line 89) `def modulation_factor(self)`
- `to_dict` (line 92) `def to_dict(self)`
- `from_dict` (line 102) `def from_dict(cls, data)`
- `__init__` (line 112) `def __init__(self, garnier, dimension)`
- `_construir_generadores_aleatorios` (line 121) `def _construir_generadores_aleatorios(self)`
- `_hadamard_generalizado` (line 130) `def _hadamard_generalizado(self)`
- `operator` (line 135) `def operator(self)`
- `calcular_alpha_modificado` (line 150) `def calcular_alpha_modificado(self, alpha_base)`
- `__init__` (line 159) `def __init__(self, garnier)`
- `calcular_delta_s_loop` (line 163) `def calcular_delta_s_loop(self, rho_red, b1)`
- `es_silencio_activo` (line 170) `def es_silencio_activo(self, rho_red, b1)`
- `__post_init__` (line 193) `def __post_init__(self)`
- `spectral_density` (line 197) `def spectral_density(self, omega)`
- `bures_distance` (line 203) `def bures_distance(self, other)`
- `__init__` (line 234) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `_initialize_leaves` (line 270) `def _initialize_leaves(self)`
- `_generate_complete_measure` (line 281) `def _generate_complete_measure(self)`
- `_aplicar_modulacion_garnier` (line 310) `def _aplicar_modulacion_garnier(self, measure)`
- `_construct_global_state` (line 321) `def _construct_global_state(self)`
- `_calcular_libertad` (line 330) `def _calcular_libertad(self)`
- `_calcular_coherencia` (line 333) `def _calcular_coherencia(self)`
- `__init__` (line 342) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `_generate_realistic_modular_network` (line 380) `def _generate_realistic_modular_network(self)`
- `_compute_betti_numbers` (line 437) `def _compute_betti_numbers(self)`
- `_spectral_dimension` (line 445) `def _spectral_dimension(self)`
- `_topological_ramsey` (line 467) `def _topological_ramsey(self)`
- `_calcular_rho_reducida` (line 471) `def _calcular_rho_reducida(self)`
- `_validar_axioma_6` (line 479) `def _validar_axioma_6(self)`
- `__init__` (line 493) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 510) `def _free_hamiltonian(self)`
- `_loss_potential` (line 515) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 521) `def _compute_scalar_mass(self)`
- `__init__` (line 529) `def __init__(self, universe, network, myelin)`
- `compute_log_bayes_factor` (line 534) `def compute_log_bayes_factor(self)`
- `get_memory_gb` (line 577) `def get_memory_gb()`
- `log_resources` (line 582) `def log_resources()`

#### `resma_Test.py`
**Path:** `resma_Test.py`

**Classes:**
- `PTSymmetricActivation` (line 11) `class PTSymmetricActivation`
- `E8LatticeLayer` (line 42) `class E8LatticeLayer`
- `RESMABrain` (line 71) `class RESMABrain`

**Functions:**
- `stress_test_resma` (line 88) `def stress_test_resma(model)`
- `__init__` (line 12) `def __init__(self, omega, chi, kappa_init)`
- `forward` (line 18) `def forward(self, x)`
- `__init__` (line 43) `def __init__(self, in_features, out_features, q_order)`
- `_generate_ramsey_mask` (line 54) `def _generate_ramsey_mask(self)`
- `forward` (line 64) `def forward(self, x)`
- `__init__` (line 72) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 78) `def forward(self, x)`

#### `resmann.py`
**Path:** `resmann.py`

**Classes:**
- `PTSymmetricActivation` (line 11) `class PTSymmetricActivation`
- `E8LatticeLayer` (line 41) `class E8LatticeLayer`
- `RESMABrain` (line 86) `class RESMABrain`

**Functions:**
- `__init__` (line 12) `def __init__(self, omega, chi, kappa_init)`
- `forward` (line 19) `def forward(self, x)`
- `__init__` (line 42) `def __init__(self, in_features, out_features, q_order)`
- `_generate_ramsey_mask` (line 59) `def _generate_ramsey_mask(self)`
- `forward` (line 69) `def forward(self, x)`
- `__init__` (line 87) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 97) `def forward(self, x)`
- `resma_loss` (line 104) `def resma_loss(self, output, target, lambda_topo)`

#### `resmann2.py`
**Path:** `resmann2.py`

**Classes:**
- `PTSymmetricActivation` (line 13) `class PTSymmetricActivation`
- `E8LatticeMultiverseLayer` (line 31) `class E8LatticeMultiverseLayer`
- `RESMABrainMultiverse` (line 65) `class RESMABrainMultiverse`

**Functions:**
- `__init__` (line 14) `def __init__(self, omega, chi, kappa_init)`
- `forward` (line 20) `def forward(self, x)`
- `__init__` (line 32) `def __init__(self, in_features, out_features, n_universes)`
- `_multiverse_mask` (line 41) `def _multiverse_mask(self, in_f, out_f, n_univ)`
- `forward` (line 55) `def forward(self, x)`
- `__init__` (line 66) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 74) `def forward(self, x)`
- `resma_loss` (line 79) `def resma_loss(self, output, target, lambda_topo)`

#### `resmannn.py`
**Path:** `resmannn.py`

**Classes:**
- `PTSymmetricActivation` (line 12) `class PTSymmetricActivation`
- `E8LatticeLayer` (line 30) `class E8LatticeLayer`
- `RESMABrainLight` (line 60) `class RESMABrainLight`

**Functions:**
- `__init__` (line 13) `def __init__(self, omega, chi, kappa_init)`
- `forward` (line 19) `def forward(self, x)`
- `__init__` (line 31) `def __init__(self, in_features, out_features)`
- `_fixed_sparse_mask` (line 40) `def _fixed_sparse_mask(self, out_f, in_f)`
- `forward` (line 49) `def forward(self, x)`
- `__init__` (line 61) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 69) `def forward(self, x)`
- `resma_loss` (line 74) `def resma_loss(self, output, target, lambda_topo)`

#### `run_complete_experiment.py`
**Path:** `run_complete_experiment.py`

**Functions:**
- `create_experiment_summary` (line 24) `def create_experiment_summary(results_dir, metrics, duration)` - *Crea un resumen del experimento*
- `generate_final_report` (line 46) `def generate_final_report(results_dir, metrics, duration)` - *Genera reporte final detallado*
- `run_complete_experiment` (line 155) `def run_complete_experiment()` - *Ejecuta el experimento completo con todas las características*

#### `scientific_benchmark.py`
**Path:** `scientific_benchmark.py`

**Classes:**
- `SupConLoss` (line 57) `class SupConLoss`
- `PredictiveErrorCell` (line 85) `class PredictiveErrorCell`
- `LearnableAbsenceGating` (line 97) `class LearnableAbsenceGating`
- `SymbioticBasisRefinement` (line 111) `class SymbioticBasisRefinement`
- `CombinatorialComplexLayer` (line 132) `class CombinatorialComplexLayer`
- `TopoBrainNet` (line 184) `class TopoBrainNet`
- `Wrapper` (line 317) `class Wrapper`

**Functions:**
- `seed_everything` (line 42) `def seed_everything(seed)`
- `clamp_pgd` (line 276) `def clamp_pgd(x_adv_norm, x_orig_norm, eps)`
- `make_adversarial_pgd` (line 283) `def make_adversarial_pgd(model, x, y, eps, steps)`
- `eval_autoattack` (line 299) `def eval_autoattack(model, test_loader, n_samples)`
- `save_topology_snapshot` (line 332) `def save_topology_snapshot(model, epoch, run_name)`
- `run_training` (line 350) `def run_training(config_override, run_name)`
- `run_ablation_suite_scientific` (line 473) `def run_ablation_suite_scientific()`
- `__init__` (line 58) `def __init__(self, temperature)`
- `forward` (line 62) `def forward(self, features, labels)`
- `__init__` (line 86) `def __init__(self, dim, use_spectral)`
- `forward` (line 92) `def forward(self, input_signal, prediction)`
- `__init__` (line 98) `def __init__(self, dim)`
- `forward` (line 107) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 112) `def __init__(self, dim, num_atoms)`
- `forward` (line 120) `def forward(self, x)`
- `__init__` (line 133) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- `forward` (line 157) `def forward(self, x_nodes, adjacency, incidence)`
- `__init__` (line 185) `def __init__(self, config)`
- `_init_grid` (line 220) `def _init_grid(self, N)`
- `get_topology` (line 241) `def get_topology(self)`
- `forward` (line 253) `def forward(self, x)`
- `__init__` (line 318) `def __init__(self, m)`
- `forward` (line 319) `def forward(self, x)`
- `lambda_topo` (line 387) `def lambda_topo(epoch)`

#### `scientist_sinergy_ablation_plan.py`
**Path:** `scientist_sinergy_ablation_plan.py`

**Classes:**
- `ScientificConfig` (line 31) `class ScientificConfig`

**Functions:**
- `check_memory_usage` (line 63) `def check_memory_usage()` - *Monitorea uso de memoria para evitar crashes con Nested Learning*
- `memory_safe_check` (line 68) `def memory_safe_check(config)` - *Verifica si es seguro ejecutar con Nested Learning*
- `setup_matplotlib_for_plotting` (line 85) `def setup_matplotlib_for_plotting()` - *Setup matplotlib para visualizaciones científicas*
- `generate_sinergy_matrix` (line 94) `def generate_sinergy_matrix()` - *Genera matriz de sinergias basada en tus inventos*
- `print_sinergy_analysis` (line 139) `def print_sinergy_analysis()` - *Analiza las sinergias propuestas basado en tus modelos*
- `main` (line 171) `def main()`

#### `setup_environment.py`
**Path:** `setup_environment.py`

**Functions:**
- `check_python_version` (line 15) `def check_python_version()` - *Verifica la versión de Python*
- `install_package` (line 23) `def install_package(package)` - *Instala un paquete usando pip*
- `check_and_install_dependencies` (line 32) `def check_and_install_dependencies()` - *Verifica e instala dependencias*
- `create_directories` (line 76) `def create_directories()` - *Crea directorios necesarios*
- `setup_matplotlib` (line 92) `def setup_matplotlib()` - *Configura matplotlib para el entorno*
- `create_sample_data` (line 118) `def create_sample_data()` - *Crea datos de muestra para pruebas*
- `test_installation` (line 148) `def test_installation()` - *Prueba la instalación*
- `create_main_script` (line 189) `def create_main_script()` - *Crea script principal para ejecutar experimentos*
- `main` (line 256) `def main()` - *Función principal de setup*

#### `sintesis.py`
**Path:** `sintesis.py`

**Classes:**
- `SpectralMonitorV6` (line 12) `class SpectralMonitorV6`
- `PrismaticNeuron` (line 48) `class PrismaticNeuron`
- `SynthesisOrganismV6` (line 117) `class SynthesisOrganismV6`

**Functions:**
- `run_prism_dream` (line 162) `def run_prism_dream()`
- `__init__` (line 13) `def __init__(self, target_entropy)`
- `calc_structural_health` (line 16) `def calc_structural_health(self, weight_matrix)`
- `measure_spatial_richness` (line 33) `def measure_spatial_richness(self, activations)`
- `__init__` (line 49) `def __init__(self, in_dim, out_dim)`
- `forward` (line 58) `def forward(self, x)`
- `prismatic_dream` (line 81) `def prismatic_dream(self)` - *Sueño Entrópico:
1. Fusionar memoria.
2. Refracción Espectral (Whitening).*
- `__init__` (line 118) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 128) `def forward(self, x)`
- `calculate_losses` (line 134) `def calculate_losses(self, outputs, targets, criterion)`
- `sleep` (line 154) `def sleep(self)`

#### `sintesys2.py`
**Path:** `sintesys2.py`

**Classes:**
- `SpectralMonitorV7` (line 12) `class SpectralMonitorV7`
- `CuriosityGaze` (line 52) `class CuriosityGaze`
- `PrismaticNeuronV7` (line 69) `class PrismaticNeuronV7`
- `SynthesisOrganismV7` (line 114) `class SynthesisOrganismV7`

**Functions:**
- `run_the_prisms_eye` (line 174) `def run_the_prisms_eye()`
- `__init__` (line 13) `def __init__(self, target_entropy)`
- `calc_structural_health` (line 16) `def calc_structural_health(self, weight_matrix)`
- `measure_spatial_richness` (line 32) `def measure_spatial_richness(self, activations)` - *Ahora retorna el tensor (para el gradiente) y el valor escalar.
Necesitamos que sea diferenciable para que el 'Ojo' aprenda a buscar riqueza.*
- `__init__` (line 53) `def __init__(self, input_dim)`
- `forward` (line 60) `def forward(self, x)`
- `__init__` (line 70) `def __init__(self, in_dim, out_dim)`
- `forward` (line 79) `def forward(self, x)`
- `prismatic_dream` (line 95) `def prismatic_dream(self)`
- `__init__` (line 115) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 131) `def forward(self, x)`
- `calculate_losses` (line 144) `def calculate_losses(self, outputs, targets, criterion)`
- `sleep` (line 166) `def sleep(self)`

#### `sintesys3.py`
**Path:** `sintesys3.py`

**Classes:**
- `HomeostasisEngine` (line 30) `class HomeostasisEngine`
- `LiquidNeuron` (line 59) `class LiquidNeuron`
- `OrganismV8` (line 109) `class OrganismV8`

**Functions:**
- `measure_spatial_richness` (line 12) `def measure_spatial_richness(activations)` - *Retorna tensor (gradiente) y valor escalar*
- `run_liquid_synthesis` (line 156) `def run_liquid_synthesis()`
- `__init__` (line 31) `def __init__(self)`
- `decide` (line 35) `def decide(self, task_loss_val, richness_val, vn_entropy_val, target_entropy)`
- `__init__` (line 60) `def __init__(self, in_dim, out_dim)`
- `forward` (line 68) `def forward(self, x, plasticity_gate)`
- `consolidate_svd` (line 86) `def consolidate_svd(self, repair_strength)` - *Sueño a demanda, intensidad variable*
- `__init__` (line 110) `def __init__(self, d_in, d_hid, d_out)`
- `forward` (line 126) `def forward(self, x, plasticity_gate)`
- `get_structure_entropy` (line 140) `def get_structure_entropy(self)`
- `calc_ent` (line 143) `def calc_ent(W)`

#### `sintesys5.py`
**Path:** `sintesys5.py`

**Classes:**
- `RealWorldEnvironment` (line 19) `class RealWorldEnvironment`
- `HomeostasisEngine` (line 68) `class HomeostasisEngine`
- `LiquidNeuron` (line 87) `class LiquidNeuron`
- `OrganismV8_Real` (line 124) `class OrganismV8_Real`

**Functions:**
- `measure_spatial_richness` (line 56) `def measure_spatial_richness(activations)`
- `run_real_world_challenge` (line 157) `def run_real_world_challenge()`
- `__init__` (line 20) `def __init__(self)`
- `get_batch` (line 38) `def get_batch(self, phase, batch_size)`
- `__init__` (line 69) `def __init__(self)`
- `decide` (line 73) `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- `__init__` (line 88) `def __init__(self, in_dim, out_dim)`
- `forward` (line 96) `def forward(self, x, plasticity_gate)`
- `consolidate_svd` (line 111) `def consolidate_svd(self, repair_strength)`
- `__init__` (line 125) `def __init__(self, d_in, d_hid, d_out)`
- `forward` (line 135) `def forward(self, x, plasticity_gate)`
- `get_structure_entropy` (line 143) `def get_structure_entropy(self)`
- `calc_ent` (line 145) `def calc_ent(W)`

#### `syntesys4.py`
**Path:** `syntesys4.py`

**Classes:**
- `HomeostasisEngine` (line 27) `class HomeostasisEngine`
- `LiquidNeuron` (line 60) `class LiquidNeuron`
- `OrganismV8_1` (line 101) `class OrganismV8_1`

**Functions:**
- `measure_spatial_richness` (line 12) `def measure_spatial_richness(activations)`
- `run_sensitive_self` (line 131) `def run_sensitive_self()`
- `__init__` (line 28) `def __init__(self)`
- `decide` (line 32) `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- `__init__` (line 61) `def __init__(self, in_dim, out_dim)`
- `forward` (line 69) `def forward(self, x, plasticity_gate)`
- `consolidate_svd` (line 84) `def consolidate_svd(self, repair_strength)`
- `__init__` (line 102) `def __init__(self, d_in, d_hid, d_out)`
- `forward` (line 112) `def forward(self, x, plasticity_gate)`
- `get_structure_entropy` (line 120) `def get_structure_entropy(self)`
- `calc_ent` (line 122) `def calc_ent(W)`

#### `test.py`
**Path:** `test.py`

**Classes:**
- `ColapsoGarantizado` (line 52) `class ColapsoGarantizado`

**Functions:**
- `measure_metrics` (line 80) `def measure_metrics(model)`
- `calculate_test_accuracy` (line 109) `def calculate_test_accuracy(model, testloader)`
- `__init__` (line 53) `def __init__(self)`
- `forward` (line 69) `def forward(self, x)`

#### `test_premium_synergy.py`
**Path:** `test_premium_synergy.py`

**Functions:**
- `test_individual_components` (line 29) `def test_individual_components()` - *Test de componentes individuales*
- `test_full_system` (line 91) `def test_full_system()` - *Test del sistema completo Premium Synergy*
- `test_training_loop` (line 164) `def test_training_loop()` - *Test del loop de entrenamiento completo*
- `run_all_tests` (line 209) `def run_all_tests()` - *Ejecuta todos los tests*

#### `topobrain.py`
**Path:** `topobrain.py`

**Classes:**
- `ResourceMonitor` (line 58) `class ResourceMonitor`
- `LearnableAbsenceGating` (line 135) `class LearnableAbsenceGating`
- `SupConLoss` (line 147) `class SupConLoss`
- `PredictiveErrorCell` (line 175) `class PredictiveErrorCell`
- `SymbioticBasisRefinement` (line 187) `class SymbioticBasisRefinement`
- `CombinatorialComplexLayer` (line 205) `class CombinatorialComplexLayer`
- `TopoBrainNet` (line 252) `class TopoBrainNet`
- `Wrapper` (line 409) `class Wrapper`

**Functions:**
- `seed_everything` (line 50) `def seed_everything(seed)`
- `guardar_checkpoint` (line 87) `def guardar_checkpoint(data, filename)` - *✅ FIX: Guarda solo estado esencial + compresión*
- `cargar_checkpoint` (line 118) `def cargar_checkpoint(filename)`
- `clamp_pgd` (line 352) `def clamp_pgd(x_adv_norm, x_orig_norm, eps)`
- `make_adversarial_pgd` (line 359) `def make_adversarial_pgd(model, x, y, eps, steps)`
- `eval_autoattack` (line 391) `def eval_autoattack(model, test_loader, n_samples)`
- `save_topology_snapshot` (line 423) `def save_topology_snapshot(model, epoch, run_name)`
- `plot_topology_evolution` (line 449) `def plot_topology_evolution(run_name)`
- `run_training` (line 476) `def run_training(config_override, run_name)`
- `run_diagnostic_suite` (line 676) `def run_diagnostic_suite()` - *✅ Suite completa con TopoOnly crítico*
- `get_memory_gb` (line 60) `def get_memory_gb()`
- `check_memory_limit` (line 65) `def check_memory_limit(limit_gb)`
- `log_resources` (line 72) `def log_resources()` - *✅ FIX: Método faltante añadido*
- `clear_cache` (line 81) `def clear_cache()`
- `__init__` (line 136) `def __init__(self, dim)`
- `forward` (line 143) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 148) `def __init__(self, temperature)`
- `forward` (line 152) `def forward(self, features, labels)`
- `__init__` (line 176) `def __init__(self, dim, use_spectral)`
- `forward` (line 182) `def forward(self, input_signal, prediction)`
- `__init__` (line 188) `def __init__(self, dim, num_atoms)`
- `forward` (line 196) `def forward(self, x)`
- `__init__` (line 206) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- `forward` (line 229) `def forward(self, x_nodes, adjacency, incidence, global_step)`
- `__init__` (line 253) `def __init__(self, config)`
- `_init_grid` (line 287) `def _init_grid(self, N)`
- `get_topology` (line 308) `def get_topology(self)`
- `forward` (line 324) `def forward(self, x)`
- `__init__` (line 410) `def __init__(self, m)`
- `forward` (line 411) `def forward(self, x)`
- `lambda_topo` (line 529) `def lambda_topo(epoch)`

#### `topobrain_16_3.py`
**Path:** `topobrain_16_3.py`

**Classes:**
- `ResourceMonitor` (line 70) `class ResourceMonitor`
- `LearnableAbsenceGating` (line 140) `class LearnableAbsenceGating`
- `SupConLoss` (line 152) `class SupConLoss`
- `PredictiveErrorCell` (line 183) `class PredictiveErrorCell`
- `SymbioticBasisRefinement` (line 195) `class SymbioticBasisRefinement`
- `CombinatorialComplexLayer` (line 213) `class CombinatorialComplexLayer`
- `TopoBrainNet` (line 266) `class TopoBrainNet`
- `Wrapper` (line 409) `class Wrapper`

**Functions:**
- `seed_everything` (line 61) `def seed_everything(seed)`
- `guardar_checkpoint` (line 98) `def guardar_checkpoint(data, filename)`
- `cargar_checkpoint` (line 123) `def cargar_checkpoint(filename)`
- `clamp_pgd` (line 361) `def clamp_pgd(x_adv_norm, x_orig_norm, eps)`
- `make_adversarial_pgd` (line 368) `def make_adversarial_pgd(model, x, y, eps, steps)`
- `eval_autoattack` (line 393) `def eval_autoattack(model, test_loader, n_samples)`
- `save_topology_snapshot` (line 423) `def save_topology_snapshot(model, epoch, run_name)`
- `plot_topology_evolution` (line 447) `def plot_topology_evolution(run_name)`
- `run_training` (line 464) `def run_training(config_override, run_name)`
- `run_diagnostic_suite` (line 726) `def run_diagnostic_suite()`
- `get_memory_gb` (line 72) `def get_memory_gb()`
- `check_memory_limit` (line 77) `def check_memory_limit(limit_gb)`
- `log_resources` (line 84) `def log_resources()`
- `clear_cache` (line 92) `def clear_cache()`
- `__init__` (line 141) `def __init__(self, dim)`
- `forward` (line 148) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 153) `def __init__(self, temperature)`
- `forward` (line 157) `def forward(self, features, labels)`
- `__init__` (line 184) `def __init__(self, dim, use_spectral)`
- `forward` (line 190) `def forward(self, input_signal, prediction)`
- `__init__` (line 196) `def __init__(self, dim, num_atoms)`
- `forward` (line 204) `def forward(self, x)`
- `__init__` (line 214) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- `forward` (line 236) `def forward(self, x_nodes, adjacency, incidence, global_step)`
- `__init__` (line 267) `def __init__(self, config)`
- `_init_grid` (line 299) `def _init_grid(self, N)`
- `get_topology` (line 319) `def get_topology(self)`
- `forward` (line 333) `def forward(self, x)`
- `__init__` (line 410) `def __init__(self, m)`
- `forward` (line 411) `def forward(self, x)`
- `lambda_topo` (line 531) `def lambda_topo(epoch)`

#### `topobrain_v18.1.py`
**Path:** `topobrain_v18.1.py`

**Classes:**
- `Config` (line 30) `class Config` - *Configuración centralizada y reproducible*
- `ResourceMonitor` (line 109) `class ResourceMonitor`
- `TopologyMetrics` (line 143) `class TopologyMetrics`
- `TopologicalHealthSovereignty` (line 152) `class TopologicalHealthSovereignty` - *monitor_topologico
Fusión de SovereigntyMonitor + TopologicalHealth
< 0.5s por época | Monitorea solo adj_weights/inc_weights*
- `CheckpointManager` (line 260) `class CheckpointManager`
- `SupConLoss` (line 366) `class SupConLoss` - *Supervised Contrastive Loss*
- `AsymmetricPredictiveErrorCell` (line 398) `class AsymmetricPredictiveErrorCell` - *Predictive Coding con manejo asimétrico de errores
Inspirado en corteza predictiva biológica*
- `LearnableAbsenceGating` (line 437) `class LearnableAbsenceGating` - *Gating basado en error de predicción*
- `SymbioticBasisRefinement` (line 453) `class SymbioticBasisRefinement` - *Refinamiento mediante base ortogonal aprendible*
- `AdaptiveCombinatorialComplexLayer` (line 473) `class AdaptiveCombinatorialComplexLayer` - *Capa combinatorial con:
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico*
- `TopoBrainNetV18` (line 598) `class TopoBrainNetV18` - *TopoBrain v18: Implementa
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico
- Regularización ortogonal ponderada*

**Functions:**
- `seed_everything` (line 100) `def seed_everything(seed)`
- `get_dataset_stats` (line 309) `def get_dataset_stats(dataset_name)`
- `get_dataloaders` (line 316) `def get_dataloaders(config)`
- `make_adversarial_pgd` (line 820) `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)` - *PGD Attack*
- `train_epoch` (line 848) `def train_epoch(model, train_loader, optimizer, opt_topo, supcon, config, epoch, topo_monitor)` - *Entrena una época con schedule adaptativo de SupCon - CORREGIDO*
- `train_model` (line 956) `def train_model(config, run_name)` - *Loop de entrenamiento v18*
- `evaluate` (line 1139) `def evaluate(model, test_loader, config, adversarial)` - *Evalúa el modelo*
- `save_topology_visualization` (line 1166) `def save_topology_visualization(model, epoch, run_name)` - *Guarda visualización de la topología aprendida*
- `save_node_importance_viz` (line 1211) `def save_node_importance_viz(model, epoch, run_name)` - *Visualiza importancia de nodos por capa*
- `analyze_topology_clustering` (line 1234) `def analyze_topology_clustering(model, run_name)` - *Clustering espectral de nodos basado en conectividad*
- `analyze_topology_flow` (line 1277) `def analyze_topology_flow(model, dataloader, run_name, num_samples)` - *Analiza flujo de información en la topología
Similar a Grad-CAM pero para topología*
- `visualize_topology_as_graph` (line 1343) `def visualize_topology_as_graph(model, run_name, threshold)` - *Visualiza topología como grafo con NetworkX*
- `analyze_topology_evolution` (line 1409) `def analyze_topology_evolution(run_name)` - *Analiza la evolución de la topología a lo largo del entrenamiento*
- `comprehensive_topology_analysis` (line 1478) `def comprehensive_topology_analysis(model, dataloader, run_name)` - *Análisis completo de topología
Ejecuta todos los análisis disponibles*
- `run_ablation_study` (line 1509) `def run_ablation_study()` - *Ejecuta suite completa de ablación v18*
- `main` (line 1633) `def main()` - *Punto de entrada principal v18*
- `__post_init__` (line 82) `def __post_init__(self)`
- `to_dict` (line 86) `def to_dict(self)`
- `get_supcon_lambda` (line 89) `def get_supcon_lambda(self, epoch)` - *Schedule adaptativo para SupCon Loss*
- `get_memory_gb` (line 111) `def get_memory_gb()`
- `get_gpu_memory_gb` (line 116) `def get_gpu_memory_gb()`
- `log` (line 122) `def log(prefix)`
- `clear_cache` (line 129) `def clear_cache()`
- `check_limit` (line 135) `def check_limit(limit_gb)`
- `__init__` (line 159) `def __init__(self, model, config, epsilon_c)`
- `_analyze_matrix` (line 165) `def _analyze_matrix(self, weight_matrix, name)` - *Análisis SVD de matriz topológica (adj o inc)*
- `calculate` (line 220) `def calculate(self, epoch)` - *Analiza todas las matrices topológicas del modelo*
- `get_critical_summary` (line 247) `def get_critical_summary(self)` - *Resumen de emergencias*
- `__init__` (line 261) `def __init__(self, checkpoint_dir)`
- `save` (line 265) `def save(self, data, name)`
- `load` (line 288) `def load(self, name)`
- `__init__` (line 368) `def __init__(self, temperature)`
- `forward` (line 372) `def forward(self, features, labels)`
- `__init__` (line 403) `def __init__(self, dim, use_spectral)`
- `forward` (line 415) `def forward(self, input_signal, prediction)`
- `__init__` (line 439) `def __init__(self, dim)`
- `forward` (line 448) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 455) `def __init__(self, dim, num_atoms)`
- `forward` (line 463) `def forward(self, x)`
- `__init__` (line 480) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- `forward` (line 509) `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse)` - *Args:
    x_nodes: [B, N, D] node features
    adjacency: [N, N] dense adjacency (fallback)
    incidence: [N, C] dense incidence (fallback)
    adj_sparse: sparse adjacency
    inc_sparse: sparse incidence*
- `get_node_importance` (line 588) `def get_node_importance(self)` - *Retorna importancia de nodos para visualización*
- `__init__` (line 606) `def __init__(self, config, in_channels)`
- `_init_grid_topology` (line 653) `def _init_grid_topology(self, N)` - *Inicializa topología de grid 2D*
- `get_topology` (line 681) `def get_topology(self, return_sparse)` - *Calcula topología actual

Args:
    return_sparse: Si True, retorna versiones sparse*
- `calculate_ortho_loss` (line 708) `def calculate_ortho_loss(self)` - *Regularización ortogonal con pesos por capa*
- `prune_topology` (line 741) `def prune_topology(self)` - *Poda de topología basada en importancia*
- `forward` (line 787) `def forward(self, x)`
- `set_epoch` (line 813) `def set_epoch(self, epoch)` - *Permite pasar la época actual para schedules dinámicos*
- `warmup_topo` (line 1006) `def warmup_topo(epoch)`

#### `topobrain_v18.2.py`
**Path:** `topobrain_v18.2.py`

**Classes:**
- `Config` (line 30) `class Config` - *Configuración centralizada y reproducible*
- `ResourceMonitor` (line 109) `class ResourceMonitor`
- `TopologyMetrics` (line 143) `class TopologyMetrics`
- `TopologicalHealthSovereignty` (line 152) `class TopologicalHealthSovereignty` - *monitor_topologico
Fusión de SovereigntyMonitor + TopologicalHealth
< 0.5s por época | Monitorea solo adj_weights/inc_weights*
- `CheckpointManager` (line 260) `class CheckpointManager`
- `SupConLoss` (line 366) `class SupConLoss` - *Supervised Contrastive Loss*
- `AsymmetricPredictiveErrorCell` (line 398) `class AsymmetricPredictiveErrorCell` - *Predictive Coding con manejo asimétrico de errores
Inspirado en corteza predictiva biológica*
- `LearnableAbsenceGating` (line 437) `class LearnableAbsenceGating` - *Gating basado en error de predicción*
- `SymbioticBasisRefinement` (line 453) `class SymbioticBasisRefinement` - *Refinamiento mediante base ortogonal aprendible*
- `AdaptiveCombinatorialComplexLayer` (line 473) `class AdaptiveCombinatorialComplexLayer` - *Capa combinatorial con:
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico*
- `TopoBrainNetV18` (line 599) `class TopoBrainNetV18` - *TopoBrain v18: Implementa
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico
- Regularización ortogonal ponderada
- CORRECCIÓN: Patch Size dinámico para ajustar num_nodes al grid_size*

**Functions:**
- `seed_everything` (line 100) `def seed_everything(seed)`
- `get_dataset_stats` (line 309) `def get_dataset_stats(dataset_name)`
- `get_dataloaders` (line 316) `def get_dataloaders(config)`
- `make_adversarial_pgd` (line 824) `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)` - *PGD Attack*
- `train_epoch` (line 852) `def train_epoch(model, train_loader, optimizer, opt_topo, supcon, config, epoch, topo_monitor)` - *Entrena una época con schedule adaptativo de SupCon - CORREGIDO*
- `train_model` (line 960) `def train_model(config, run_name)` - *Loop de entrenamiento v18 (CORREGIDO - Inicialización Negativa)*
- `evaluate` (line 1088) `def evaluate(model, test_loader, config, adversarial)` - *Evalúa el modelo*
- `save_topology_visualization` (line 1115) `def save_topology_visualization(model, epoch, run_name)` - *Guarda visualización de la topología aprendida*
- `save_node_importance_viz` (line 1160) `def save_node_importance_viz(model, epoch, run_name)` - *Visualiza importancia de nodos por capa*
- `analyze_topology_clustering` (line 1183) `def analyze_topology_clustering(model, run_name)` - *Clustering espectral de nodos basado en conectividad*
- `analyze_topology_flow` (line 1226) `def analyze_topology_flow(model, dataloader, run_name, num_samples)` - *Analiza flujo de información en la topología
Similar a Grad-CAM pero para topología*
- `visualize_topology_as_graph` (line 1292) `def visualize_topology_as_graph(model, run_name, threshold)` - *Visualiza topología como grafo con NetworkX*
- `analyze_topology_evolution` (line 1358) `def analyze_topology_evolution(run_name)` - *Analiza la evolución de la topología a lo largo del entrenamiento*
- `comprehensive_topology_analysis` (line 1427) `def comprehensive_topology_analysis(model, dataloader, run_name)` - *Análisis completo de topología
Ejecuta todos los análisis disponibles*
- `run_ablation_study` (line 1458) `def run_ablation_study()` - *Ejecuta suite completa de ablación v18*
- `main` (line 1582) `def main()` - *Punto de entrada principal v18*
- `__post_init__` (line 82) `def __post_init__(self)`
- `to_dict` (line 86) `def to_dict(self)`
- `get_supcon_lambda` (line 89) `def get_supcon_lambda(self, epoch)` - *Schedule adaptativo para SupCon Loss*
- `get_memory_gb` (line 111) `def get_memory_gb()`
- `get_gpu_memory_gb` (line 116) `def get_gpu_memory_gb()`
- `log` (line 122) `def log(prefix)`
- `clear_cache` (line 129) `def clear_cache()`
- `check_limit` (line 135) `def check_limit(limit_gb)`
- `__init__` (line 159) `def __init__(self, model, config, epsilon_c)`
- `_analyze_matrix` (line 165) `def _analyze_matrix(self, weight_matrix, name)` - *Análisis SVD de matriz topológica (CORREGIDO)*
- `calculate` (line 220) `def calculate(self, epoch)` - *Analiza todas las matrices topológicas del modelo*
- `get_critical_summary` (line 247) `def get_critical_summary(self)` - *Resumen de emergencias*
- `__init__` (line 261) `def __init__(self, checkpoint_dir)`
- `save` (line 265) `def save(self, data, name)`
- `load` (line 288) `def load(self, name)`
- `__init__` (line 368) `def __init__(self, temperature)`
- `forward` (line 372) `def forward(self, features, labels)`
- `__init__` (line 403) `def __init__(self, dim, use_spectral)`
- `forward` (line 415) `def forward(self, input_signal, prediction)`
- `__init__` (line 439) `def __init__(self, dim)`
- `forward` (line 448) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 455) `def __init__(self, dim, num_atoms)`
- `forward` (line 463) `def forward(self, x)`
- `__init__` (line 480) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- `forward` (line 510) `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse)` - *Args:
    x_nodes: [B, N, D] node features
    adjacency: [N, N] dense adjacency (fallback)
    incidence: [N, C] dense incidence (fallback)
    adj_sparse: sparse adjacency
    inc_sparse: sparse incidence*
- `get_node_importance` (line 589) `def get_node_importance(self)` - *Retorna importancia de nodos para visualización*
- `__init__` (line 608) `def __init__(self, config, in_channels)`
- `_init_grid_topology` (line 668) `def _init_grid_topology(self, N)` - *Inicializa topología de grid 2D*
- `get_topology` (line 705) `def get_topology(self, return_sparse)` - *Calcula topología actual*
- `calculate_ortho_loss` (line 729) `def calculate_ortho_loss(self)` - *Regularización ortogonal con pesos por capa*
- `prune_topology` (line 754) `def prune_topology(self)` - *Poda de topología basada en importancia*
- `forward` (line 793) `def forward(self, x)`
- `set_epoch` (line 818) `def set_epoch(self, epoch)`
- `warmup_topo` (line 1016) `def warmup_topo(epoch)`

#### `topobrain_v18.py`
**Path:** `topobrain_v18.py`

**Classes:**
- `Config` (line 30) `class Config` - *Configuración centralizada y reproducible*
- `ResourceMonitor` (line 109) `class ResourceMonitor`
- `TopologyMetrics` (line 143) `class TopologyMetrics`
- `TopologicalHealthSovereignty` (line 152) `class TopologicalHealthSovereignty` - *monitor_topologico
Fusión de SovereigntyMonitor + TopologicalHealth
< 0.5s por época | Monitorea solo adj_weights/inc_weights*
- `CheckpointManager` (line 260) `class CheckpointManager`
- `SupConLoss` (line 366) `class SupConLoss` - *Supervised Contrastive Loss*
- `AsymmetricPredictiveErrorCell` (line 398) `class AsymmetricPredictiveErrorCell` - *Predictive Coding con manejo asimétrico de errores
Inspirado en corteza predictiva biológica*
- `LearnableAbsenceGating` (line 437) `class LearnableAbsenceGating` - *Gating basado en error de predicción*
- `SymbioticBasisRefinement` (line 453) `class SymbioticBasisRefinement` - *Refinamiento mediante base ortogonal aprendible*
- `AdaptiveCombinatorialComplexLayer` (line 473) `class AdaptiveCombinatorialComplexLayer` - *Capa combinatorial con:
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico*
- `TopoBrainNetV18` (line 598) `class TopoBrainNetV18` - *TopoBrain v18: Implementa
- Sparse tensor operations
- Topología adaptativa
- Predictive coding asimétrico
- Regularización ortogonal ponderada*

**Functions:**
- `seed_everything` (line 100) `def seed_everything(seed)`
- `get_dataset_stats` (line 309) `def get_dataset_stats(dataset_name)`
- `get_dataloaders` (line 316) `def get_dataloaders(config)`
- `make_adversarial_pgd` (line 799) `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)` - *PGD Attack*
- `train_epoch` (line 827) `def train_epoch(model, train_loader, optimizer, opt_topo, supcon, config, epoch, topo_monitor)` - *Entrena una época con schedule adaptativo de SupCon*
- `train_model` (line 917) `def train_model(config, run_name)` - *Loop de entrenamiento completo*
- `evaluate` (line 1078) `def evaluate(model, test_loader, config, adversarial)` - *Evalúa el modelo*
- `train_model` (line 1101) `def train_model(config, run_name)` - *Loop de entrenamiento completo*
- `save_topology_visualization` (line 1257) `def save_topology_visualization(model, epoch, run_name)` - *Guarda visualización de la topología aprendida*
- `save_node_importance_viz` (line 1302) `def save_node_importance_viz(model, epoch, run_name)` - *Visualiza importancia de nodos por capa*
- `analyze_topology_clustering` (line 1325) `def analyze_topology_clustering(model, run_name)` - *Clustering espectral de nodos basado en conectividad*
- `analyze_topology_flow` (line 1368) `def analyze_topology_flow(model, dataloader, run_name, num_samples)` - *Analiza flujo de información en la topología
Similar a Grad-CAM pero para topología*
- `visualize_topology_as_graph` (line 1434) `def visualize_topology_as_graph(model, run_name, threshold)` - *Visualiza topología como grafo con NetworkX*
- `analyze_topology_evolution` (line 1500) `def analyze_topology_evolution(run_name)` - *Analiza la evolución de la topología a lo largo del entrenamiento*
- `comprehensive_topology_analysis` (line 1569) `def comprehensive_topology_analysis(model, dataloader, run_name)` - *Análisis completo de topología
Ejecuta todos los análisis disponibles*
- `run_ablation_study` (line 1600) `def run_ablation_study()` - *Ejecuta suite completa de ablación v18*
- `main` (line 1724) `def main()` - *Punto de entrada principal v18*
- `__post_init__` (line 82) `def __post_init__(self)`
- `to_dict` (line 86) `def to_dict(self)`
- `get_supcon_lambda` (line 89) `def get_supcon_lambda(self, epoch)` - *Schedule adaptativo para SupCon Loss*
- `get_memory_gb` (line 111) `def get_memory_gb()`
- `get_gpu_memory_gb` (line 116) `def get_gpu_memory_gb()`
- `log` (line 122) `def log(prefix)`
- `clear_cache` (line 129) `def clear_cache()`
- `check_limit` (line 135) `def check_limit(limit_gb)`
- `__init__` (line 159) `def __init__(self, model, config, epsilon_c)`
- `_analyze_matrix` (line 165) `def _analyze_matrix(self, weight_matrix, name)` - *Análisis SVD de matriz topológica (adj o inc)*
- `calculate` (line 220) `def calculate(self, epoch)` - *Analiza todas las matrices topológicas del modelo*
- `get_critical_summary` (line 247) `def get_critical_summary(self)` - *Resumen de emergencias*
- `__init__` (line 261) `def __init__(self, checkpoint_dir)`
- `save` (line 265) `def save(self, data, name)`
- `load` (line 288) `def load(self, name)`
- `__init__` (line 368) `def __init__(self, temperature)`
- `forward` (line 372) `def forward(self, features, labels)`
- `__init__` (line 403) `def __init__(self, dim, use_spectral)`
- `forward` (line 415) `def forward(self, input_signal, prediction)`
- `__init__` (line 439) `def __init__(self, dim)`
- `forward` (line 448) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 455) `def __init__(self, dim, num_atoms)`
- `forward` (line 463) `def forward(self, x)`
- `__init__` (line 480) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type, layer_idx)`
- `forward` (line 509) `def forward(self, x_nodes, adjacency, incidence, adj_sparse, inc_sparse)` - *Args:
    x_nodes: [B, N, D] node features
    adjacency: [N, N] dense adjacency (fallback)
    incidence: [N, C] dense incidence (fallback)
    adj_sparse: sparse adjacency
    inc_sparse: sparse incidence*
- `get_node_importance` (line 588) `def get_node_importance(self)` - *Retorna importancia de nodos para visualización*
- `__init__` (line 606) `def __init__(self, config, in_channels)`
- `_init_grid_topology` (line 653) `def _init_grid_topology(self, N)` - *Inicializa topología de grid 2D*
- `get_topology` (line 681) `def get_topology(self, return_sparse)` - *Calcula topología actual

Args:
    return_sparse: Si True, retorna versiones sparse*
- `calculate_ortho_loss` (line 708) `def calculate_ortho_loss(self)` - *Regularización ortogonal con pesos por capa*
- `prune_topology` (line 741) `def prune_topology(self)` - *Poda de topología basada en importancia*
- `forward` (line 769) `def forward(self, x)`
- `lambda_topo` (line 948) `def lambda_topo(epoch)`
- `lambda_topo` (line 1130) `def lambda_topo(epoch)`

#### `topobrain_v19.py`
**Path:** `topobrain_v19.py`

**Classes:**
- `Config` (line 42) `class Config` - *Configuración unificada y simplificada*
- `ResourceMonitor` (line 107) `class ResourceMonitor`
- `CheckpointManager` (line 139) `class CheckpointManager`
- `NodePositionLearner` (line 240) `class NodePositionLearner` - *Aprende posiciones de nodos en espacio latente
Genera conectividad k-NN dinámica*
- `DynamicTopologicalLayer` (line 291) `class DynamicTopologicalLayer` - *Capa con PyTorch Geometric y topología dinámica
Combina GAT + Predictive Coding + MGF*
- `TopoBrainNetV19` (line 360) `class TopoBrainNetV19` - *Modelo principal con topología dinámica y componentes modulares*
- `ContrastiveLoss` (line 521) `class ContrastiveLoss` - *SupCon simplificado*

**Functions:**
- `seed_everything` (line 98) `def seed_everything(seed)`
- `get_dataset_stats` (line 182) `def get_dataset_stats(dataset_name)`
- `get_dataloaders` (line 189) `def get_dataloaders(config)`
- `make_adversarial_pgd` (line 493) `def make_adversarial_pgd(model, x, y, eps, steps, dataset_name)` - *PGD Attack simplificado y robusto*
- `train_epoch` (line 545) `def train_epoch(model, train_loader, optimizer, criterion, contrastive_loss, config, epoch)` - *Entrena una época con logging integrado*
- `evaluate` (line 621) `def evaluate(model, test_loader, config, adversarial)` - *Evalúa el modelo*
- `train_model` (line 644) `def train_model(config, run_name)` - *Loop de entrenamiento completo v19*
- `save_topology_snapshot` (line 796) `def save_topology_snapshot(model, epoch, run_name)` - *Guarda snapshot de topología*
- `run_ablation_study` (line 837) `def run_ablation_study()` - *Suite de ablación sistemática v19*
- `main` (line 918) `def main()`
- `to_dict` (line 91) `def to_dict(self)`
- `get_memory_gb` (line 109) `def get_memory_gb()`
- `get_gpu_memory_gb` (line 114) `def get_gpu_memory_gb()`
- `log` (line 120) `def log(prefix)`
- `clear_cache` (line 128) `def clear_cache()`
- `check_limit` (line 134) `def check_limit(limit_gb)`
- `__init__` (line 140) `def __init__(self, checkpoint_dir)`
- `save` (line 144) `def save(self, data, name)`
- `load` (line 163) `def load(self, name)`
- `__init__` (line 245) `def __init__(self, num_nodes, node_dim, k)`
- `forward` (line 254) `def forward(self, batch_size)` - *Retorna edges para k-NN dinámico
Returns:
    edge_index: [2, E]
    edge_weight: [E]*
- `__init__` (line 296) `def __init__(self, in_dim, hid_dim, config, layer_idx)`
- `forward` (line 325) `def forward(self, x, edge_index, edge_weight, batch)` - *Args:
    x: [B*N, D] Node features
    edge_index: [2, E] Connectivity
    edge_weight: [E] Edge weights
    batch: [B*N] Batch indices*
- `__init__` (line 364) `def __init__(self, config, in_channels)`
- `forward` (line 398) `def forward(self, x)`
- `apply_pruning` (line 445) `def apply_pruning(self, edge_index, edge_weight)` - *Aplica máscara de pruning*
- `prune_structural` (line 453) `def prune_structural(self, threshold)` - *Pruning estructural real: elimina edges permanentemente*
- `calculate_ortho_loss` (line 473) `def calculate_ortho_loss(self)` - *Regularización ortogonal simple*
- `__init__` (line 523) `def __init__(self, temperature)`
- `forward` (line 527) `def forward(self, features, labels)`
- `warmup_lr` (line 668) `def warmup_lr(epoch)`
- `prune_fn` (line 467) `def prune_fn(edge_idx)`

#### `train_Adversarial.py`
**Path:** `train_Adversarial.py`

**Classes:**
- `ResourceMonitor` (line 74) `class ResourceMonitor`
- `NestedOptimizer` (line 154) `class NestedOptimizer` - *Optimizador de múltiples niveles basado en Nested Learning.
Implementa momentum como memoria asociativa con diferentes frecuencias.*
- `ContinuumMemorySystem` (line 218) `class ContinuumMemorySystem` - *Sistema de memoria continua con MLPs de diferentes frecuencias.
Basado en la Sección 3 del paper Nested Learning.*
- `SupConLoss` (line 260) `class SupConLoss`
- `PredictiveErrorCell` (line 288) `class PredictiveErrorCell`
- `LearnableAbsenceGating` (line 300) `class LearnableAbsenceGating`
- `SymbioticBasisRefinement` (line 314) `class SymbioticBasisRefinement`
- `CombinatorialComplexLayer` (line 332) `class CombinatorialComplexLayer`
- `TopoBrainNet` (line 400) `class TopoBrainNet`
- `Wrapper` (line 552) `class Wrapper`

**Functions:**
- `seed_everything` (line 61) `def seed_everything(seed)`
- `guardar_checkpoint` (line 100) `def guardar_checkpoint(data, filename)` - *Sistema de checkpoint robusto con protección contra corrupción*
- `cargar_checkpoint` (line 134) `def cargar_checkpoint(filename)` - *Carga checkpoint con fallback automático*
- `clamp_pgd` (line 509) `def clamp_pgd(x_adv_norm, x_orig_norm, eps)`
- `make_adversarial_pgd` (line 516) `def make_adversarial_pgd(model, x, y, eps, steps)`
- `eval_autoattack` (line 534) `def eval_autoattack(model, test_loader, n_samples)`
- `create_checkpoint_data` (line 566) `def create_checkpoint_data(model, optimizer, epoch, config, metrics)` - *Crea estructura de checkpoint completa*
- `run_training` (line 586) `def run_training(config_override, run_name)`
- `save_topology_snapshot` (line 770) `def save_topology_snapshot(model, epoch, run_name)`
- `run_diagnostic_suite` (line 787) `def run_diagnostic_suite()`
- `get_memory_gb` (line 78) `def get_memory_gb()`
- `log_resources` (line 83) `def log_resources()`
- `clear_cache` (line 90) `def clear_cache()` - *Limpia cachés y fuerza garbage collection*
- `__init__` (line 159) `def __init__(self, params, lr, momentum, nested_levels, freq_factor)`
- `step` (line 179) `def step(self, closure)`
- `__init__` (line 223) `def __init__(self, input_dim, hidden_dim, num_levels)`
- `forward` (line 240) `def forward(self, x)`
- `should_update_level` (line 246) `def should_update_level(self, level_idx, global_step)` - *Determina si un nivel debe actualizarse basado en su frecuencia*
- `get_update_mask` (line 250) `def get_update_mask(self, level_idx, batch_size)` - *Máscara para actualizar solo un subconjunto de parámetros*
- `__init__` (line 261) `def __init__(self, temperature)`
- `forward` (line 265) `def forward(self, features, labels)`
- `__init__` (line 289) `def __init__(self, dim, use_spectral)`
- `forward` (line 295) `def forward(self, input_signal, prediction)`
- `__init__` (line 301) `def __init__(self, dim)`
- `forward` (line 310) `def forward(self, x_sensory, x_prediction)`
- `__init__` (line 315) `def __init__(self, dim, num_atoms)`
- `forward` (line 323) `def forward(self, x)`
- `__init__` (line 333) `def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type)`
- `forward` (line 362) `def forward(self, x_nodes, adjacency, incidence, global_step)`
- `apply_cms` (line 391) `def apply_cms(self, x, global_step)` - *Aplica actualización condicional basada en frecuencia del CMS*
- `__init__` (line 401) `def __init__(self, config)`
- `_init_grid` (line 444) `def _init_grid(self, N)`
- `get_topology` (line 465) `def get_topology(self)`
- `forward` (line 477) `def forward(self, x)`
- `__init__` (line 553) `def __init__(self, m)`
- `forward` (line 554) `def forward(self, x)`
- `lambda_topo` (line 632) `def lambda_topo(epoch)`

#### `tricameral2.py`
**Path:** `tricameral2.py`

**Classes:**
- `StableLiquidNeuron` (line 389) `class StableLiquidNeuron`
- `LeftHemisphere` (line 512) `class LeftHemisphere`
- `Flickr8kMultimodalDataset` (line 584) `class Flickr8kMultimodalDataset(Dataset)` - *Dataset que carga imagen, audio del caption y texto*
- `AudioEncoder` (line 678) `class AudioEncoder` - *Encoder de audio usando Conv + Transformer*
- `RightHemisphereTricameral` (line 747) `class RightHemisphereTricameral` - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 821) `class CorpusCallosumTrimodal` - *Corpus callosum con canales: visual, auditivo, semántico*
- `NeuralAudioGenerator` (line 913) `class NeuralAudioGenerator` - *Generador de audio desde embeddings lingüísticos (TTS neuronal)*
- `NeuroLogosTricameral` (line 982) `class NeuroLogosTricameral` - *Arquitectura completa: Visión + Audio -> Lenguaje + Audio*
- `Flickr8kSimpleDataset` (line 1195) `class Flickr8kSimpleDataset(BaseDataset)`

**Functions:**
- `generate_audio_async` (line 31) `def generate_audio_async(text, output_path, voice, max_retries)` - *Genera un audio usando Edge-TTS con retry logic*
- `generate_all_audios_batch` (line 66) `def generate_all_audios_batch(captions_list, audio_dir, batch_size)` - *Genera todos los audios en batches pequeños con rate limiting*
- `generate_audios_sync` (line 135) `def generate_audios_sync(images_dir, captions_file, audio_dir)` - *Wrapper síncrono para generar audios*
- `download_from_github` (line 183) `def download_from_github(repo_url, output_dir)` - *Descarga dataset pre-preparado desde GitHub/Hugging Face*
- `setup_flickr8k` (line 257) `def setup_flickr8k(data_dir, github_url)` - *Descarga y organiza Flickr8k - ahora con opción GitHub*
- `build_vocab_flickr` (line 366) `def build_vocab_flickr(captions_file, vocab_size)` - *Construye vocabulario desde el archivo de captions*
- `compute_tricameral_loss` (line 1048) `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)` - *Pérdida con término de coherencia audio-visual*
- `train_tricameral` (line 1100) `def train_tricameral(github_repo_url)` - *Entrena el modelo tricameral

Args:
    github_repo_url: URL opcional del repo de GitHub/Hugging Face
                    Ejemplo: "https://github.com/user/flickr8k-prepared/raw/main"
                    o "https://huggingface.co/datasets/user/flickr8k-prepared/resolve/main"*
- `__init__` (line 390) `def __init__(self, in_dim, out_dim)`
- `forward` (line 425) `def forward(self, x)`
- `hebbian_update` (line 438) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 476) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 513) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 525) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_greedy_decode` (line 548) `def _greedy_decode(self, visual_context, max_len, device)`
- `_get_init_state` (line 570) `def _get_init_state(self, visual_context)`
- `__init__` (line 587) `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)`
- `__len__` (line 625) `def __len__(self)`
- `__getitem__` (line 628) `def __getitem__(self, idx)`
- `__init__` (line 681) `def __init__(self, output_dim)`
- `forward` (line 720) `def forward(self, mel_spec)` - *Args:
    mel_spec: (batch, 80, time)
Returns:
    audio_features: (batch, output_dim)*
- `__init__` (line 750) `def __init__(self, output_dim)`
- `forward` (line 784) `def forward(self, image, audio)` - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre: Para Hebbian
    audio_post, audio_pre: Para Hebbian*
- `__init__` (line 824) `def __init__(self, dim)`
- `forward` (line 856) `def forward(self, right_features)` - *Args:
    right_features: (B, dim) - Fusión de visión + audio
Returns:
    enriched_context: (B, dim)
    channels: dict con 'visual', 'audio', 'semantic'*
- `__init__` (line 916) `def __init__(self, text_dim, output_sr)`
- `forward` (line 951) `def forward(self, text_embedding)` - *Args:
    text_embedding: (B, text_dim)
Returns:
    audio_waveform: (B, 1, num_samples)*
- `__init__` (line 985) `def __init__(self, vocab_size)`
- `forward` (line 1000) `def forward(self, image, audio, captions, epoch, generate_audio)` - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T) - Mel-spectrogram del caption
    captions: (B, seq_len) - Tokens (solo en train)
    generate_audio: bool - Si generar audio

Returns (training):
    logits, gates, losses, posts, pres, channels

Returns (inference):
    generated_text, generated_audio*
- `__init__` (line 1196) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 1214) `def __len__(self)`
- `__getitem__` (line 1217) `def __getitem__(self, idx)`

#### `tricameral_kimi.py`
**Path:** `tricameral_kimi.py`

**Classes:**
- `EpisodicMemoryBuffer` (line 215) `class EpisodicMemoryBuffer`
- `NeurocognitiveSystem` (line 255) `class NeurocognitiveSystem`
- `LanguageMetrics` (line 331) `class LanguageMetrics` - *Métricas de calidad de generación*
- `LinguisticFeedbackLoop` (line 405) `class LinguisticFeedbackLoop`
- `LanguageMetrics` (line 503) `class LanguageMetrics`
- `StableLiquidNeuron` (line 550) `class StableLiquidNeuron`
- `TriangulatedMedicalSystem` (line 673) `class TriangulatedMedicalSystem`
- `LeftHemisphere` (line 769) `class LeftHemisphere`
- `AudioEncoder` (line 843) `class AudioEncoder` - *Encoder de audio usando Conv + Transformer*
- `RightHemisphereTricameral` (line 893) `class RightHemisphereTricameral` - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 965) `class CorpusCallosumTrimodal`
- `EnhancedDiagnosticsTricameral` (line 1067) `class EnhancedDiagnosticsTricameral`
- `NeuroLogosTricameral` (line 1261) `class NeuroLogosTricameral` - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 1296) `class Flickr8kMultimodalDataset(Dataset)` - *Dataset que carga imagen, audio del caption y texto desde Kaggle*

**Functions:**
- `setup_flickr8k_with_audio` (line 36) `def setup_flickr8k_with_audio(data_dir)` - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 191) `def build_vocab_flickr(captions_file, vocab_size)` - *Construye vocabulario desde el archivo de captions*
- `compute_tricameral_loss` (line 1402) `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)` - *Pérdida con término de coherencia audio-visual*
- `train_tricameral` (line 1452) `def train_tricameral()`
- `__init__` (line 216) `def __init__(self, capacity, surprise_threshold)`
- `compute_surprise` (line 222) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `add` (line 232) `def add(self, image, caption, surprise_score)`
- `sample` (line 241) `def sample(self, batch_size)`
- `__init__` (line 256) `def __init__(self)`
- `assess_cognitive_state` (line 266) `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)`
- `apply_cognitive_intervention` (line 287) `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)`
- `sentence_bleu` (line 335) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 369) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 378) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 391) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 406) `def __init__(self, alpha, beta)`
- `compute_linguistic_reward` (line 417) `def compute_linguistic_reward(self, references, hypotheses)`
- `compute_cider` (line 443) `def compute_cider(self, reference, hypothesis)`
- `compute_spice` (line 469) `def compute_spice(self, reference, hypothesis)`
- `_get_ngrams` (line 478) `def _get_ngrams(self, sentence, n)`
- `get_cache_stats` (line 482) `def get_cache_stats(self)`
- `sentence_bleu` (line 505) `def sentence_bleu(reference, hypothesis, weights)`
- `token_accuracy` (line 528) `def token_accuracy(reference, hypothesis)`
- `word_overlap` (line 538) `def word_overlap(reference, hypothesis)`
- `__init__` (line 551) `def __init__(self, in_dim, out_dim)`
- `forward` (line 586) `def forward(self, x)`
- `hebbian_update` (line 599) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 639) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 674) `def __init__(self)`
- `diagnose_with_triangulation` (line 680) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `apply_triangulated_intervention` (line 703) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- `__init__` (line 770) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 781) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_greedy_decode` (line 806) `def _greedy_decode(self, visual_context, max_len, device)`
- `_get_init_state` (line 828) `def _get_init_state(self, visual_context)`
- `__init__` (line 846) `def __init__(self, output_dim)`
- `forward` (line 880) `def forward(self, mel_spec)`
- `__init__` (line 896) `def __init__(self, output_dim)`
- `forward` (line 929) `def forward(self, image, audio)` - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre, audio_post, audio_pre: Para Hebbian*
- `__init__` (line 966) `def __init__(self, dim)`
- `forward` (line 995) `def forward(self, right_features)`
- `update_channel_fatigue` (line 1038) `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- `adjust_gates_by_fatigue` (line 1054) `def adjust_gates_by_fatigue(self)`
- `__init__` (line 1068) `def __init__(self)`
- `measure_callosal_flow` (line 1084) `def measure_callosal_flow(self, right_features, left_context, channels)`
- `evaluate_reasoning_quality` (line 1103) `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- `calculate_synergy` (line 1127) `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 1137) `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 1146) `def update(self)`
- `get_recent_avg` (line 1156) `def get_recent_avg(self, key, n)`
- `visualize_fatigue_distribution` (line 1173) `def visualize_fatigue_distribution(self, epoch)`
- `visualize_reasoning_metrics` (line 1191) `def visualize_reasoning_metrics(self, epoch)`
- `report` (line 1202) `def report(self, epoch)`
- `__init__` (line 1264) `def __init__(self, vocab_size)`
- `forward` (line 1270) `def forward(self, image, audio, captions, epoch)`
- `__init__` (line 1299) `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)`
- `__len__` (line 1347) `def __len__(self)`
- `__getitem__` (line 1351) `def __getitem__(self, idx)`

#### `tricameral_kimi2.py`
**Path:** `tricameral_kimi2.py`

**Classes:**
- `EpisodicMemoryBuffer` (line 262) `class EpisodicMemoryBuffer`
- `NeurocognitiveSystem` (line 347) `class NeurocognitiveSystem`
- `LanguageMetrics` (line 539) `class LanguageMetrics` - *Métricas de calidad de generación*
- `LinguisticFeedbackLoop` (line 613) `class LinguisticFeedbackLoop`
- `LanguageMetrics` (line 725) `class LanguageMetrics`
- `LanguageMetrics` (line 774) `class LanguageMetrics`
- `StableLiquidNeuron` (line 821) `class StableLiquidNeuron`
- `TriangulatedMedicalSystem` (line 944) `class TriangulatedMedicalSystem`
- `LeftHemisphere` (line 1103) `class LeftHemisphere`
- `AudioEncoder` (line 1401) `class AudioEncoder` - *Encoder de audio usando Conv + Transformer*
- `RightHemisphereTricameral` (line 1451) `class RightHemisphereTricameral` - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 1523) `class CorpusCallosumTrimodal`
- `EnhancedDiagnosticsTricameral` (line 1676) `class EnhancedDiagnosticsTricameral`
- `NeuroLogosTricameral` (line 1952) `class NeuroLogosTricameral` - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 1987) `class Flickr8kMultimodalDataset(Dataset)` - *Dataset que carga imagen, audio del caption y texto desde Kaggle*

**Functions:**
- `setup_flickr8k_with_audio` (line 50) `def setup_flickr8k_with_audio(data_dir)` - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 238) `def build_vocab_flickr(captions_file, vocab_size)` - *Construye vocabulario desde el archivo de captions*
- `compute_alignment_loss` (line 2091) `def compute_alignment_loss(visual_features, channels, alpha, epoch)` - *FIX: Pérdida auxiliar para alineación temprana de canales multimodales
Solo activa en épocas iniciales (epoch < 6)*
- `compute_tricameral_loss` (line 2119) `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)`
- `train_tricameral` (line 2166) `def train_tricameral()`
- `__init__` (line 263) `def __init__(self, capacity, surprise_threshold)`
- `compute_surprise` (line 278) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)`
- `add` (line 289) `def add(self, image, audio, caption, surprise_score)`
- `sample` (line 318) `def sample(self, batch_size)`
- `__init__` (line 348) `def __init__(self)`
- `assess_reasoning_state` (line 363) `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` - *Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)*
- `assess_cognitive_state` (line 407) `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` - *Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)*
- `apply_cognitive_intervention` (line 453) `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` - *Aplica intervenciones basadas en estado lingüístico y de razonamiento*
- `sentence_bleu` (line 543) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 577) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 586) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 599) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 614) `def __init__(self, alpha, beta)`
- `compute_linguistic_reward` (line 636) `def compute_linguistic_reward(self, references, hypotheses)`
- `compute_cider` (line 675) `def compute_cider(self, reference, hypothesis)`
- `compute_spice` (line 695) `def compute_spice(self, reference, hypothesis)`
- `get_cache_stats` (line 704) `def get_cache_stats(self)`
- `sentence_bleu` (line 727) `def sentence_bleu(reference, hypothesis, weights)`
- `token_accuracy` (line 750) `def token_accuracy(reference, hypothesis)`
- `word_overlap` (line 760) `def word_overlap(reference, hypothesis)`
- `sentence_bleu` (line 776) `def sentence_bleu(reference, hypothesis, weights)`
- `token_accuracy` (line 799) `def token_accuracy(reference, hypothesis)`
- `word_overlap` (line 809) `def word_overlap(reference, hypothesis)`
- `__init__` (line 822) `def __init__(self, in_dim, out_dim)`
- `forward` (line 857) `def forward(self, x)`
- `hebbian_update` (line 870) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 910) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 945) `def __init__(self)`
- `triangulate_signals` (line 951) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `count_convergent_signals` (line 962) `def count_convergent_signals(self, signals, pattern)`
- `diagnose_with_triangulation` (line 965) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)`
- `apply_triangulated_intervention` (line 1018) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- `_reset_liquid_neuron` (line 1088) `def _reset_liquid_neuron(self, liquid_neuron)` - *Reset completo de una neurona líquida*
- `__init__` (line 1104) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 1179) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_apply_chain_of_thought` (line 1222) `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- `_greedy_decode` (line 1262) `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- `_apply_multi_token_prediction` (line 1323) `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- `_apply_structural_attention` (line 1365) `def _apply_structural_attention(self, lstm_out, channels, visual_context)`
- `_get_init_state` (line 1386) `def _get_init_state(self, visual_context)`
- `__init__` (line 1404) `def __init__(self, output_dim)`
- `forward` (line 1438) `def forward(self, mel_spec)`
- `__init__` (line 1454) `def __init__(self, output_dim)`
- `forward` (line 1487) `def forward(self, image, audio)` - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre, audio_post, audio_pre: Para Hebbian*
- `__init__` (line 1524) `def __init__(self, dim)`
- `forward` (line 1572) `def forward(self, right_features)`
- `update_channel_fatigue` (line 1633) `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)`
- `adjust_gates_by_fatigue` (line 1655) `def adjust_gates_by_fatigue(self)`
- `__init__` (line 1677) `def __init__(self)`
- `_get_cached_norm` (line 1699) `def _get_cached_norm(self, tensor, dim)` - *Cache de normalización con limpieza periódica*
- `measure_callosal_flow` (line 1717) `def measure_callosal_flow(self, right_features, left_context, channels)`
- `evaluate_reasoning_quality` (line 1747) `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)`
- `calculate_synergy` (line 1784) `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 1795) `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 1804) `def update(self)`
- `get_recent_avg` (line 1821) `def get_recent_avg(self, key, n)`
- `visualize_fatigue_distribution` (line 1837) `def visualize_fatigue_distribution(self, epoch)`
- `visualize_reasoning_metrics` (line 1861) `def visualize_reasoning_metrics(self, epoch)`
- `report` (line 1873) `def report(self, epoch)`
- `__init__` (line 1955) `def __init__(self, vocab_size)`
- `forward` (line 1961) `def forward(self, image, audio, captions, epoch)`
- `__init__` (line 1990) `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)`
- `__len__` (line 2038) `def __len__(self)`
- `__getitem__` (line 2042) `def __getitem__(self, idx)`
- `cached_ngrams` (line 623) `def cached_ngrams(sentence, n)`

#### `tricameralkimi2.py`
**Path:** `tricameralkimi2.py`

**Classes:**
- `EpisodicMemoryBuffer` (line 195) `class EpisodicMemoryBuffer` - *Buffer que almacena ejemplos sorpresivos para replay estratégico*
- `NeurocognitiveSystem` (line 247) `class NeurocognitiveSystem`
- `LinguisticFeedbackLoop` (line 425) `class LinguisticFeedbackLoop` - *Sistema de caché optimizado para métricas lingüísticas*
- `StableLiquidNeuron` (line 546) `class StableLiquidNeuron`
- `AudioEncoder` (line 668) `class AudioEncoder` - *Encoder de audio usando Conv + Transformer*
- `TriangulatedMedicalSystem` (line 724) `class TriangulatedMedicalSystem`
- `RightHemisphereTricameral` (line 881) `class RightHemisphereTricameral` - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 952) `class CorpusCallosumTrimodal` - *Corpus callosum con canales: visual, auditivo, semántico*
- `LeftHemisphere` (line 1089) `class LeftHemisphere`
- `NeuroLogosTricameral` (line 1359) `class NeuroLogosTricameral` - *Arquitectura completa: Visión + Audio -> Lenguaje*
- `Flickr8kMultimodalDataset` (line 1415) `class Flickr8kMultimodalDataset(Dataset)` - *Dataset que carga imagen, audio del caption y texto desde Kaggle*
- `EnhancedDiagnosticsTricameral` (line 1556) `class EnhancedDiagnosticsTricameral`

**Functions:**
- `setup_flickr8k_with_audio` (line 30) `def setup_flickr8k_with_audio(data_dir)` - *Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
Sistema robusto que verifica componentes individuales y descarga solo lo faltante.*
- `build_vocab_flickr` (line 173) `def build_vocab_flickr(captions_file, vocab_size)` - *Construye vocabulario desde el archivo de captions*
- `compute_tricameral_loss` (line 1507) `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)` - *Pérdida con término de coherencia audio-visual*
- `train_tricameral` (line 1786) `def train_tricameral()`
- `__init__` (line 198) `def __init__(self, capacity, surprise_threshold)`
- `compute_surprise` (line 204) `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` - *Calcula sorpresa basada en error y apertura del gate*
- `add` (line 216) `def add(self, image, caption, surprise_score)` - *Añade ejemplo si supera umbral y hay capacidad*
- `sample` (line 228) `def sample(self, batch_size)` - *Samplea ejemplos con probabilidad proporcional a sorpresa*
- `__init__` (line 248) `def __init__(self)`
- `assess_reasoning_state` (line 263) `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` - *Evalúa estado del sistema de razonamiento*
- `assess_cognitive_state` (line 305) `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` - *Evalúa estado cognitivo lingüístico*
- `apply_cognitive_intervention` (line 341) `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` - *Aplica intervenciones basadas en estado lingüístico y razonamiento*
- `__init__` (line 428) `def __init__(self, alpha, beta)`
- `compute_linguistic_reward` (line 442) `def compute_linguistic_reward(self, references, hypotheses)` - *Recompensa combinada CIDEr + SPICE con caché*
- `compute_cider` (line 476) `def compute_cider(self, reference, hypothesis)` - *CIDEr simplificado con caché de n-gramas*
- `compute_spice` (line 505) `def compute_spice(self, reference, hypothesis)` - *SPICE simplificado (Jaccard similarity)*
- `_get_ngrams` (line 518) `def _get_ngrams(self, sentence, n)` - *Extractor de n-gramas*
- `get_cache_stats` (line 523) `def get_cache_stats(self)` - *Estadísticas de caché*
- `__init__` (line 547) `def __init__(self, in_dim, out_dim)`
- `forward` (line 582) `def forward(self, x)`
- `hebbian_update` (line 595) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 633) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 671) `def __init__(self, output_dim)`
- `forward` (line 705) `def forward(self, mel_spec)` - *Args:
    mel_spec: (batch, 80, time)
Returns:
    audio_features: (batch, output_dim)*
- `__init__` (line 725) `def __init__(self)`
- `triangulate_signals` (line 731) `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `count_convergent_signals` (line 741) `def count_convergent_signals(self, signals, pattern)`
- `diagnose_with_triangulation` (line 744) `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)`
- `apply_triangulated_intervention` (line 786) `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)`
- `_reset_liquid_neuron` (line 868) `def _reset_liquid_neuron(self, right_node, severity)`
- `__init__` (line 884) `def __init__(self, output_dim)`
- `forward` (line 917) `def forward(self, image, audio)` - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre, audio_post, audio_pre: Para Hebbian*
- `__init__` (line 955) `def __init__(self, dim)`
- `forward` (line 991) `def forward(self, right_features)` - *Args:
    right_features: (B, dim) - Fusión de visión + audio
Returns:
    enriched_context: (B, dim)
    channels: dict con 'visual', 'audio', 'semantic'*
- `update_channel_fatigue` (line 1060) `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` - *Actualiza fatiga específica por canal*
- `adjust_gates_by_fatigue` (line 1079) `def adjust_gates_by_fatigue(self)` - *Ajusta proyecciones basado en fatiga*
- `__init__` (line 1090) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 1165) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_greedy_decode` (line 1204) `def _greedy_decode(self, visual_context, channels, max_len, epoch)`
- `_apply_chain_of_thought` (line 1237) `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)`
- `_apply_multi_token_prediction` (line 1271) `def _apply_multi_token_prediction(self, hidden_states, input_ids)`
- `_apply_structural_attention` (line 1317) `def _apply_structural_attention(self, lstm_out, channels, visual_context)` - *Atenuación simple según fatiga de canal + atención cruzada visual.
Se ignoran los canales 'objects/actions/scene' que no existen.*
- `_get_init_state` (line 1344) `def _get_init_state(self, visual_context)`
- `__init__` (line 1362) `def __init__(self, vocab_size)`
- `forward` (line 1374) `def forward(self, image, audio, captions, epoch)` - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T) - Mel-spectrogram del caption
    captions: (B, seq_len) - Tokens (solo en train)

Returns (training):
    logits, gates, losses, posts, pres, channels

Returns (inference):
    generated_text*
- `__init__` (line 1418) `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)`
- `__len__` (line 1463) `def __len__(self)`
- `__getitem__` (line 1466) `def __getitem__(self, idx)`
- `__init__` (line 1557) `def __init__(self)`
- `measure_callosal_flow` (line 1572) `def measure_callosal_flow(self, right_features, left_context, channels)`
- `evaluate_reasoning_quality` (line 1595) `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` - *Evalúa coherencia y consistencia del razonamiento*
- `calculate_synergy` (line 1627) `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 1638) `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 1647) `def update(self)`
- `get_recent_avg` (line 1657) `def get_recent_avg(self, key, n)`
- `visualize_fatigue_distribution` (line 1674) `def visualize_fatigue_distribution(self, epoch)`
- `visualize_reasoning_metrics` (line 1695) `def visualize_reasoning_metrics(self, epoch)`
- `report` (line 1707) `def report(self, epoch)` - *Genera reporte completo del estado del sistema tricameral*

#### `trycameral.py`
**Path:** `trycameral.py`

**Classes:**
- `StableLiquidNeuron` (line 228) `class StableLiquidNeuron`
- `LeftHemisphere` (line 351) `class LeftHemisphere`
- `Flickr8kMultimodalDataset` (line 423) `class Flickr8kMultimodalDataset(Dataset)` - *Dataset que carga imagen, audio del caption y texto*
- `AudioEncoder` (line 517) `class AudioEncoder` - *Encoder de audio usando Conv + Transformer*
- `RightHemisphereTricameral` (line 586) `class RightHemisphereTricameral` - *Hemisferio derecho con canales visual y auditivo*
- `CorpusCallosumTrimodal` (line 660) `class CorpusCallosumTrimodal` - *Corpus callosum con canales: visual, auditivo, semántico*
- `NeuralAudioGenerator` (line 752) `class NeuralAudioGenerator` - *Generador de audio desde embeddings lingüísticos (TTS neuronal)*
- `NeuroLogosTricameral` (line 821) `class NeuroLogosTricameral` - *Arquitectura completa: Visión + Audio -> Lenguaje + Audio*
- `Flickr8kSimpleDataset` (line 996) `class Flickr8kSimpleDataset(BaseDataset)`

**Functions:**
- `generate_audio_async` (line 31) `def generate_audio_async(text, output_path, voice)` - *Genera un audio usando Edge-TTS*
- `generate_all_audios_batch` (line 50) `def generate_all_audios_batch(captions_list, audio_dir, batch_size)` - *Genera todos los audios en batches para eficiencia*
- `generate_audios_sync` (line 89) `def generate_audios_sync(images_dir, captions_file, audio_dir)` - *Wrapper síncrono para generar audios*
- `setup_flickr8k` (line 133) `def setup_flickr8k(data_dir)` - *Descarga y organiza Flickr8k si no existe*
- `build_vocab_flickr` (line 205) `def build_vocab_flickr(captions_file, vocab_size)` - *Construye vocabulario desde el archivo de captions*
- `compute_tricameral_loss` (line 887) `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward, lambda_reward, lambda_mtp)` - *Pérdida con término de coherencia audio-visual*
- `train_tricameral` (line 939) `def train_tricameral()`
- `__init__` (line 229) `def __init__(self, in_dim, out_dim)`
- `forward` (line 264) `def forward(self, x)`
- `hebbian_update` (line 277) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 315) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 352) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 364) `def forward(self, visual_context, captions, channels, max_len, epoch)`
- `_greedy_decode` (line 387) `def _greedy_decode(self, visual_context, max_len, device)`
- `_get_init_state` (line 409) `def _get_init_state(self, visual_context)`
- `__init__` (line 426) `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate)`
- `__len__` (line 464) `def __len__(self)`
- `__getitem__` (line 467) `def __getitem__(self, idx)`
- `__init__` (line 520) `def __init__(self, output_dim)`
- `forward` (line 559) `def forward(self, mel_spec)` - *Args:
    mel_spec: (batch, 80, time)
Returns:
    audio_features: (batch, output_dim)*
- `__init__` (line 589) `def __init__(self, output_dim)`
- `forward` (line 623) `def forward(self, image, audio)` - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T)
Returns:
    fused_features: (B, output_dim)
    visual_post, visual_pre: Para Hebbian
    audio_post, audio_pre: Para Hebbian*
- `__init__` (line 663) `def __init__(self, dim)`
- `forward` (line 695) `def forward(self, right_features)` - *Args:
    right_features: (B, dim) - Fusión de visión + audio
Returns:
    enriched_context: (B, dim)
    channels: dict con 'visual', 'audio', 'semantic'*
- `__init__` (line 755) `def __init__(self, text_dim, output_sr)`
- `forward` (line 790) `def forward(self, text_embedding)` - *Args:
    text_embedding: (B, text_dim)
Returns:
    audio_waveform: (B, 1, num_samples)*
- `__init__` (line 824) `def __init__(self, vocab_size)`
- `forward` (line 839) `def forward(self, image, audio, captions, epoch, generate_audio)` - *Args:
    image: (B, 3, H, W)
    audio: (B, 80, T) - Mel-spectrogram del caption
    captions: (B, seq_len) - Tokens (solo en train)
    generate_audio: bool - Si generar audio

Returns (training):
    logits, gates, losses, posts, pres, channels

Returns (inference):
    generated_text, generated_audio*
- `__init__` (line 997) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 1015) `def __len__(self)`
- `__getitem__` (line 1018) `def __getitem__(self, idx)`

#### `ultimo_neuorlogos.py`
**Path:** `ultimo_neuorlogos.py`

**Classes:**
- `TopoBrainCore` (line 27) `class TopoBrainCore` - *TopoBrain validado con ablation (de tu experimento anterior)*
- `PGDAttack` (line 113) `class PGDAttack` - *Adversarial attack para robustez (solo en TOPO-FULL)*
- `MiniUnconscious` (line 145) `class MiniUnconscious` - *Baseline: Encoder visual simple*
- `TopoUnconscious` (line 166) `class TopoUnconscious` - *TopoBrain-enhanced visual encoder*
- `ConsciousCore` (line 203) `class ConsciousCore` - *Núcleo consciente con atención*
- `BioDecoder` (line 221) `class BioDecoder` - *Decoder LSTM con gating*
- `NeuroLogos` (line 282) `class NeuroLogos` - *Configuraciones del ablation:
- mode='baseline': MiniUnconscious (sin TopoBrain)
- mode='topo-light': TopoBrain sin symbiotic
- mode='topo-full': TopoBrain completo + adversarial*
- `CIFARCaptions` (line 330) `class CIFARCaptions`

**Functions:**
- `train_ablation` (line 374) `def train_ablation(mode, epochs, device)` - *Entrena un modelo en el modo especificado*
- `run_full_ablation` (line 508) `def run_full_ablation(epochs, device)` - *Ejecuta ablation study completo de 3 niveles*
- `__init__` (line 29) `def __init__(self, input_dim, hidden_dim, output_dim, grid_size, use_grid, use_symbiotic)`
- `_init_grid` (line 62) `def _init_grid(self)` - *Inicializa coordenadas del grid 2D*
- `forward` (line 70) `def forward(self, x)`
- `get_metrics` (line 101) `def get_metrics(self)` - *Retorna métricas de topología*
- `__init__` (line 115) `def __init__(self, epsilon, alpha, steps)`
- `attack` (line 120) `def attack(self, model, x, y, criterion)` - *Genera ejemplos adversariales*
- `__init__` (line 147) `def __init__(self, output_dim)`
- `forward` (line 160) `def forward(self, x)`
- `__init__` (line 168) `def __init__(self, output_dim, use_grid, use_symbiotic)`
- `forward` (line 191) `def forward(self, x)`
- `get_metrics` (line 195) `def get_metrics(self)`
- `__init__` (line 205) `def __init__(self, dim)`
- `forward` (line 210) `def forward(self, x)`
- `__init__` (line 223) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 238) `def forward(self, thought, captions, max_len)`
- `_get_init_state` (line 272) `def _get_init_state(self, thought)`
- `__init__` (line 289) `def __init__(self, vocab_size, mode)`
- `forward` (line 314) `def forward(self, image, captions)`
- `get_metrics` (line 319) `def get_metrics(self)` - *Obtiene métricas de topología si disponible*
- `__init__` (line 331) `def __init__(self)`
- `__len__` (line 356) `def __len__(self)`
- `__getitem__` (line 359) `def __getitem__(self, idx)`

#### `ultimobicameral.py`
**Path:** `ultimobicameral.py`

**Classes:**
- `LanguageMetrics` (line 23) `class LanguageMetrics` - *Métricas de calidad de generación*
- `MedicalSystem` (line 96) `class MedicalSystem` - *Sistema de intervención médica por niveles*
- `StableLiquidNeuron` (line 292) `class StableLiquidNeuron`
- `RightHemisphere` (line 368) `class RightHemisphere`
- `LeftHemisphere` (line 383) `class LeftHemisphere`
- `CorpusCallosum` (line 443) `class CorpusCallosum`
- `NeuroLogosBicameralStable` (line 462) `class NeuroLogosBicameralStable`
- `EnhancedDiagnostics` (line 483) `class EnhancedDiagnostics`
- `Flickr8kDataset` (line 608) `class Flickr8kDataset(Dataset)`

**Functions:**
- `build_vocab_flickr` (line 645) `def build_vocab_flickr(captions_file, vocab_size)`
- `setup_flickr8k` (line 663) `def setup_flickr8k(data_dir)`
- `train_with_metrics` (line 677) `def train_with_metrics()`
- `sentence_bleu` (line 27) `def sentence_bleu(reference, hypothesis, weights)` - *BLEU simplificado a nivel de oración*
- `_get_ngrams` (line 61) `def _get_ngrams(tokens, n)` - *Extraer n-gramas de una lista de tokens*
- `token_accuracy` (line 70) `def token_accuracy(reference, hypothesis)` - *Porcentaje de tokens correctos en posición*
- `word_overlap` (line 83) `def word_overlap(reference, hypothesis)` - *Jaccard similarity entre palabras*
- `__init__` (line 99) `def __init__(self)`
- `diagnose_severity` (line 103) `def diagnose_severity(self, health_score, liquid_norm, gate_mean, callosal_flow)` - *Diagnosticar gravedad del problema con análisis mejorado*
- `apply_intervention` (line 149) `def apply_intervention(self, model, issues, severity, epoch)` - *Aplicar intervención médica calibrada con más agresividad en gate*
- `__init__` (line 293) `def __init__(self, in_dim, out_dim)`
- `forward` (line 307) `def forward(self, x)`
- `hebbian_update` (line 314) `def hebbian_update(self, post, pre, plasticity)`
- `update_physiology_advanced` (line 344) `def update_physiology_advanced(self, loss_value)`
- `__init__` (line 369) `def __init__(self, output_dim)`
- `forward` (line 377) `def forward(self, image)`
- `__init__` (line 384) `def __init__(self, vocab_size, embed_dim, hidden_dim)`
- `forward` (line 401) `def forward(self, visual_context, captions, max_len)`
- `_get_init_state` (line 438) `def _get_init_state(self, visual_context)`
- `__init__` (line 444) `def __init__(self, dim)`
- `forward` (line 456) `def forward(self, right_features)`
- `__init__` (line 463) `def __init__(self, vocab_size)`
- `forward` (line 469) `def forward(self, image, captions)`
- `__init__` (line 484) `def __init__(self)`
- `measure_callosal_flow` (line 494) `def measure_callosal_flow(self, right_features, left_context)`
- `calculate_synergy` (line 503) `def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std)`
- `calculate_health` (line 512) `def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)`
- `update` (line 521) `def update(self)`
- `get_recent_avg` (line 526) `def get_recent_avg(self, key, n)`
- `report` (line 531) `def report(self, epoch)`
- `__init__` (line 609) `def __init__(self, images_dir, captions_file, vocab, transform, max_len)`
- `__len__` (line 626) `def __len__(self)`
- `__getitem__` (line 629) `def __getitem__(self, idx)`

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*

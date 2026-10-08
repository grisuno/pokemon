# Symbols (page 4 of 13)
Previous: [SYMBOLS_p3.md](SYMBOLS_p3.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `LeftHemisphere` | class | `exodia_op_2.py:1510` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `exodia_op_2.py:848` | `class LinguisticFeedbackLoop` |
| `NeuroLogosTricameral` | class | `exodia_op_2.py:2515` | `class NeuroLogosTricameral(Module)` |
| `NeurocognitiveSystem` | class | `exodia_op_2.py:577` | `class NeurocognitiveSystem` |
| `RightHemisphereTricameral` | class | `exodia_op_2.py:1901` | `class RightHemisphereTricameral(Module)` |
| `StableLiquidNeuron` | class | `exodia_op_2.py:1134` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `exodia_op_2.py:1360` | `class TriangulatedMedicalSystem` |
| `TricameralOutput` | class | `exodia_op_2.py:1321` | `class TricameralOutput(NamedTuple)` |
| `__getitem__` | method | `exodia_op_2.py:2613` | `def __getitem__(self, idx)` |
| `__init__` | method | `exodia_op_2.py:347` | `def __init__(self, working_capacity, short_term_capacity, importance_threshold)` |
| `__init__` | method | `exodia_op_2.py:578` | `def __init__(self)` |
| `__init__` | method | `exodia_op_2.py:849` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `exodia_op_2.py:1009` | `def __init__(self, hidden_dim)` |
| `__init__` | method | `exodia_op_2.py:1136` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `exodia_op_2.py:1361` | `def __init__(self)` |
| `__init__` | method | `exodia_op_2.py:1511` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `exodia_op_2.py:1842` | `def __init__(self, output_dim)` |
| `__init__` | method | `exodia_op_2.py:1909` | `def __init__(self, output_dim)` |
| `__init__` | method | `exodia_op_2.py:1994` | `def __init__(self, dim)` |
| `__init__` | method | `exodia_op_2.py:2208` | `def __init__(self)` |
| `__init__` | method | `exodia_op_2.py:2518` | `def __init__(self, vocab_size)` |
| `__init__` | method | `exodia_op_2.py:2553` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache...` |
| `__len__` | method | `exodia_op_2.py:2610` | `def __len__(self)` |
| `_apply_chain_of_thought` | method | `exodia_op_2.py:1653` | `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)` |
| `_apply_flash_attention` | method | `exodia_op_2.py:2057` | `def _apply_flash_attention(self, x)` |
| `_apply_multi_token_prediction` | method | `exodia_op_2.py:1694` | `def _apply_multi_token_prediction(self, hidden_states, input_ids)` |
| `_apply_structural_attention` | method | `exodia_op_2.py:1737` | `def _apply_structural_attention(self, lstm_out, channels, visual_context)` |
| `_calculate_homeostasis_metric` | method | `exodia_op_2.py:1219` | `def _calculate_homeostasis_metric(self, output)` |
| `_calculate_novelty` | method | `exodia_op_2.py:402` | `def _calculate_novelty(self, episode)` |
| `_get_cached_norm` | method | `exodia_op_2.py:2231` | `def _get_cached_norm(self, tensor, dim)` |
| `_get_init_state` | method | `exodia_op_2.py:1819` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `exodia_op_2.py:812` | `def _get_ngrams(tokens, n)` |
| `_get_ngrams_cached` | method | `exodia_op_2.py:863` | `def _get_ngrams_cached(sentence, n)` |
| `_greedy_decode` | method | `exodia_op_2.py:1759` | `def _greedy_decode(self, visual_context, channels, max_len, epoch)` |
| `_predict_interventions` | method | `exodia_op_2.py:1050` | `def _predict_interventions(self, hypothesis, confidence)` |
| `_purge_low_score_memories` | method | `exodia_op_2.py:545` | `def _purge_low_score_memories(self)` |
| `_reset_liquid_neuron` | method | `exodia_op_2.py:1496` | `def _reset_liquid_neuron(self, liquid_neuron)` |
| `_sample_from_buffer` | method | `exodia_op_2.py:494` | `def _sample_from_buffer(self, buffer, scores, batch_size)` |
| `_update_unified_buffer` | method | `exodia_op_2.py:456` | `def _update_unified_buffer(self)` |
| `adjust_gates_by_fatigue` | method | `exodia_op_2.py:2190` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `exodia_op_2.py:688` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_emergency_fixes` | function | `exodia_op_2.py:120` | `def apply_emergency_fixes(model)` |
| `apply_forgetting_curve` | method | `exodia_op_2.py:535` | `def apply_forgetting_curve(self)` |
| `apply_triangulated_intervention` | method | `exodia_op_2.py:1427` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `exodia_op_2.py:642` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `assess_reasoning_state` | method | `exodia_op_2.py:598` | `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` |
| `build_vocab_flickr` | function | `exodia_op_2.py:316` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `exodia_op_2.py:2353` | `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_importance` | method | `exodia_op_2.py:386` | `def calculate_importance(self, episode, surprise_score)` |
| `calculate_synergy` | method | `exodia_op_2.py:2342` | `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_alignment_loss` | method | `exodia_op_2.py:2666` | `def compute_alignment_loss(visual_features, channels, alpha, epoch)` |
| `compute_cider` | method | `exodia_op_2.py:911` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `exodia_op_2.py:872` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_spice` | method | `exodia_op_2.py:925` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `exodia_op_2.py:371` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `compute_tricameral_loss` | method | `exodia_op_2.py:2695` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward...` |
| `count_convergent_signals` | method | `exodia_op_2.py:1379` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `exodia_op_2.py:1382` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)` |
| `evaluate_reasoning_quality` | method | `exodia_op_2.py:2305` | `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` |
| `forward` | method | `exodia_op_2.py:1184` | `def forward(self, x)` |
| `forward` | method | `exodia_op_2.py:1335` | `def forward(self, image, audio, captions, epoch)` |
| `forward` | method | `exodia_op_2.py:1596` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `exodia_op_2.py:1880` | `def forward(self, mel_spec)` |
| `forward` | method | `exodia_op_2.py:1947` | `def forward(self, image, audio)` |
| `forward` | method | `exodia_op_2.py:2090` | `def forward(self, right_features)` |
| `forward` | method | `exodia_op_2.py:2525` | `def forward(self, image, audio, captions, epoch)` |
| `get_cache_stats` | method | `exodia_op_2.py:937` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `exodia_op_2.py:2379` | `def get_recent_avg(self, key, n)` |
| `hebbian_update` | method | `exodia_op_2.py:1229` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `exodia_op_2.py:2248` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `preprocess_and_cache_spectrograms` | function | `exodia_op_2.py:49` | `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)` |
| `query_causal_chain` | method | `exodia_op_2.py:1073` | `def query_causal_chain(self, start_node, end_node)` |
| `reason_causally` | method | `exodia_op_2.py:1036` | `def reason_causally(self, observation, context)` |
| `report` | method | `exodia_op_2.py:2429` | `def report(self, epoch)` |
| `sample` | method | `exodia_op_2.py:470` | `def sample(self, batch_size, memory_level)` |
| `sentence_bleu` | method | `exodia_op_2.py:778` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `exodia_op_2.py:967` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `exodia_op_2.py:1089` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k_with_audio` | function | `exodia_op_2.py:142` | `def setup_flickr8k_with_audio(data_dir)` |
| `store_episode` | method | `exodia_op_2.py:427` | `def store_episode(self, image, audio, caption, surprise_score)` |
| `token_accuracy` | method | `exodia_op_2.py:821` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `exodia_op_2.py:990` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `exodia_op_2.py:1112` | `def token_accuracy(reference, hypothesis)` |
| `train_tricameral` | method | `exodia_op_2.py:2812` | `def train_tricameral()` |
| `triangulate_signals` | method | `exodia_op_2.py:1368` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `exodia_op_2.py:2362` | `def update(self)` |
| `update_channel_fatigue` | method | `exodia_op_2.py:2169` | `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` |
| `update_knowledge_graph` | method | `exodia_op_2.py:1067` | `def update_knowledge_graph(self, cause, effect, strength)` |
| `update_physiology_advanced` | method | `exodia_op_2.py:1278` | `def update_physiology_advanced(self, loss_value)` |
| `visualize_fatigue_distribution` | method | `exodia_op_2.py:2395` | `def visualize_fatigue_distribution(self, epoch)` |
| `visualize_reasoning_metrics` | method | `exodia_op_2.py:2417` | `def visualize_reasoning_metrics(self, epoch)` |
| `word_overlap` | method | `exodia_op_2.py:834` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `exodia_op_2.py:1000` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `exodia_op_2.py:1122` | `def word_overlap(reference, hypothesis)` |
| `AudioEncoder` | class | `exodia_optimized.py:1703` | `class AudioEncoder(Module)` |
| `CausalReasoningEngine` | class | `exodia_optimized.py:977` | `class CausalReasoningEngine(Module)` |
| `CorpusCallosumTrimodal` | class | `exodia_optimized.py:1861` | `class CorpusCallosumTrimodal(Module)` |
| `EnhancedDiagnosticsTricameral` | class | `exodia_optimized.py:2093` | `class EnhancedDiagnosticsTricameral` |
| `Flickr8kMultimodalDataset` | class | `exodia_optimized.py:2433` | `class Flickr8kMultimodalDataset(Dataset)` |
| `HierarchicalEpisodicMemory` | class | `exodia_optimized.py:332` | `class HierarchicalEpisodicMemory` |
| `LanguageMetrics` | class | `exodia_optimized.py:743` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `exodia_optimized.py:934` | `class LanguageMetrics` |
| `LanguageMetrics` | class | `exodia_optimized.py:1056` | `class LanguageMetrics` |
| `LeftHemisphere` | class | `exodia_optimized.py:1394` | `class LeftHemisphere(Module)` |
| `LinguisticFeedbackLoop` | class | `exodia_optimized.py:817` | `class LinguisticFeedbackLoop` |
| `NeuroLogosTricameral` | class | `exodia_optimized.py:2398` | `class NeuroLogosTricameral(Module)` |
| `NeurocognitiveSystem` | class | `exodia_optimized.py:546` | `class NeurocognitiveSystem` |
| `RightHemisphereTricameral` | class | `exodia_optimized.py:1778` | `class RightHemisphereTricameral(Module)` |
| `StableLiquidNeuron` | class | `exodia_optimized.py:1103` | `class StableLiquidNeuron(Module)` |
| `TriangulatedMedicalSystem` | class | `exodia_optimized.py:1243` | `class TriangulatedMedicalSystem` |
| `__getitem__` | method | `exodia_optimized.py:2496` | `def __getitem__(self, idx)` |
| `__init__` | method | `exodia_optimized.py:341` | `def __init__(self, working_capacity, short_term_capacity, importance_threshold)` |
| `__init__` | method | `exodia_optimized.py:547` | `def __init__(self)` |
| `__init__` | method | `exodia_optimized.py:818` | `def __init__(self, alpha, beta)` |
| `__init__` | method | `exodia_optimized.py:978` | `def __init__(self, hidden_dim)` |
| `__init__` | method | `exodia_optimized.py:1104` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `exodia_optimized.py:1244` | `def __init__(self)` |
| `__init__` | method | `exodia_optimized.py:1395` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `exodia_optimized.py:1711` | `def __init__(self, output_dim)` |
| `__init__` | method | `exodia_optimized.py:1786` | `def __init__(self, output_dim)` |
| `__init__` | method | `exodia_optimized.py:1871` | `def __init__(self, dim)` |
| `__init__` | method | `exodia_optimized.py:2094` | `def __init__(self)` |
| `__init__` | method | `exodia_optimized.py:2401` | `def __init__(self, vocab_size)` |
| `__init__` | method | `exodia_optimized.py:2436` | `def __init__(self, images_dir, audio_dir, captions_file, vocab, img_transform, max_len, sample_rate, use_cache...` |
| `__len__` | method | `exodia_optimized.py:2493` | `def __len__(self)` |
| `_apply_chain_of_thought` | method | `exodia_optimized.py:1524` | `def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning)` |
| `_apply_flash_attention` | method | `exodia_optimized.py:1940` | `def _apply_flash_attention(self, x)` |
| `_apply_multi_token_prediction` | method | `exodia_optimized.py:1625` | `def _apply_multi_token_prediction(self, hidden_states, input_ids)` |
| `_apply_structural_attention` | method | `exodia_optimized.py:1667` | `def _apply_structural_attention(self, lstm_out, channels, visual_context)` |
| `_calculate_homeostasis_metric` | method | `exodia_optimized.py:1163` | `def _calculate_homeostasis_metric(self, output)` |
| `_calculate_novelty` | method | `exodia_optimized.py:393` | `def _calculate_novelty(self, episode)` |
| `_get_cached_norm` | method | `exodia_optimized.py:2117` | `def _get_cached_norm(self, tensor, dim)` |
| `_get_init_state` | method | `exodia_optimized.py:1688` | `def _get_init_state(self, visual_context)` |
| `_get_ngrams` | method | `exodia_optimized.py:781` | `def _get_ngrams(tokens, n)` |
| `_get_ngrams_cached` | method | `exodia_optimized.py:832` | `def _get_ngrams_cached(sentence, n)` |
| `_greedy_decode` | method | `exodia_optimized.py:1564` | `def _greedy_decode(self, visual_context, channels, max_len, epoch)` |
| `_predict_interventions` | method | `exodia_optimized.py:1019` | `def _predict_interventions(self, hypothesis, confidence)` |
| `_purge_low_score_memories` | method | `exodia_optimized.py:467` | `def _purge_low_score_memories(self)` |
| `_reset_liquid_neuron` | method | `exodia_optimized.py:1379` | `def _reset_liquid_neuron(self, liquid_neuron)` |
| `_sample_from_buffer` | method | `exodia_optimized.py:511` | `def _sample_from_buffer(self, buffer, scores, batch_size)` |
| `_update_unified_buffer` | method | `exodia_optimized.py:445` | `def _update_unified_buffer(self)` |
| `add` | method | `exodia_optimized.py:450` | `def add(self, image, audio, caption, surprise_score)` |
| `adjust_gates_by_fatigue` | method | `exodia_optimized.py:2076` | `def adjust_gates_by_fatigue(self)` |
| `apply_cognitive_intervention` | method | `exodia_optimized.py:657` | `def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics)` |
| `apply_forgetting_curve` | method | `exodia_optimized.py:454` | `def apply_forgetting_curve(self)` |
| `apply_triangulated_intervention` | method | `exodia_optimized.py:1310` | `def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch)` |
| `assess_cognitive_state` | method | `exodia_optimized.py:611` | `def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch)` |
| `assess_reasoning_state` | method | `exodia_optimized.py:567` | `def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch)` |
| `build_vocab_flickr` | function | `exodia_optimized.py:308` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `calculate_health` | method | `exodia_optimized.py:2236` | `def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm)` |
| `calculate_importance` | method | `exodia_optimized.py:380` | `def calculate_importance(self, episode, surprise_score)` |
| `calculate_synergy` | method | `exodia_optimized.py:2225` | `def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std)` |
| `compute_alignment_loss` | method | `exodia_optimized.py:2549` | `def compute_alignment_loss(visual_features, channels, alpha, epoch)` |
| `compute_cider` | method | `exodia_optimized.py:880` | `def compute_cider(self, reference, hypothesis)` |
| `compute_linguistic_reward` | method | `exodia_optimized.py:841` | `def compute_linguistic_reward(self, references, hypotheses)` |
| `compute_spice` | method | `exodia_optimized.py:894` | `def compute_spice(self, reference, hypothesis)` |
| `compute_surprise` | method | `exodia_optimized.py:369` | `def compute_surprise(self, predicted_logits, ground_truth, gate_mean)` |
| `compute_tricameral_loss` | method | `exodia_optimized.py:2577` | `def compute_tricameral_loss(logits, captions, gate, vocab, visual_post, audio_post, mtp_loss, linguistic_reward...` |
| `count_convergent_signals` | method | `exodia_optimized.py:1262` | `def count_convergent_signals(self, signals, pattern)` |
| `diagnose_with_triangulation` | method | `exodia_optimized.py:1265` | `def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch)` |
| `evaluate_reasoning_quality` | method | `exodia_optimized.py:2188` | `def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps)` |
| `forward` | method | `exodia_optimized.py:1146` | `def forward(self, x)` |
| `forward` | method | `exodia_optimized.py:1477` | `def forward(self, visual_context, captions, channels, max_len, epoch)` |
| `forward` | method | `exodia_optimized.py:1751` | `def forward(self, mel_spec)` |
| `forward` | method | `exodia_optimized.py:1824` | `def forward(self, image, audio)` |
| `forward` | method | `exodia_optimized.py:1966` | `def forward(self, right_features)` |
| `forward` | method | `exodia_optimized.py:2407` | `def forward(self, image, audio, captions, epoch)` |
| `get_cache_stats` | method | `exodia_optimized.py:906` | `def get_cache_stats(self)` |
| `get_recent_avg` | method | `exodia_optimized.py:2262` | `def get_recent_avg(self, key, n)` |
| `get_total_size` | method | `exodia_optimized.py:539` | `def get_total_size(self)` |
| `hebbian_update` | method | `exodia_optimized.py:1172` | `def hebbian_update(self, post, pre, plasticity)` |
| `measure_callosal_flow` | method | `exodia_optimized.py:2135` | `def measure_callosal_flow(self, right_features, left_context, channels)` |
| `preprocess_and_cache_spectrograms` | function | `exodia_optimized.py:47` | `def preprocess_and_cache_spectrograms(audio_dir, cache_dir, sample_rate, target_len)` |
| `query_causal_chain` | method | `exodia_optimized.py:1042` | `def query_causal_chain(self, start_node, end_node)` |
| `reason_causally` | method | `exodia_optimized.py:1005` | `def reason_causally(self, observation, context)` |
| `report` | method | `exodia_optimized.py:2314` | `def report(self, epoch)` |
| `sample` | method | `exodia_optimized.py:487` | `def sample(self, batch_size, memory_level)` |
| `sentence_bleu` | method | `exodia_optimized.py:747` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `exodia_optimized.py:936` | `def sentence_bleu(reference, hypothesis, weights)` |
| `sentence_bleu` | method | `exodia_optimized.py:1058` | `def sentence_bleu(reference, hypothesis, weights)` |
| `setup_flickr8k_with_audio` | function | `exodia_optimized.py:120` | `def setup_flickr8k_with_audio(data_dir)` |
| `store_episode` | method | `exodia_optimized.py:416` | `def store_episode(self, image, audio, caption, surprise_score)` |
| `token_accuracy` | method | `exodia_optimized.py:790` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `exodia_optimized.py:959` | `def token_accuracy(reference, hypothesis)` |
| `token_accuracy` | method | `exodia_optimized.py:1081` | `def token_accuracy(reference, hypothesis)` |
| `train_tricameral` | method | `exodia_optimized.py:2650` | `def train_tricameral()` |
| `triangulate_signals` | method | `exodia_optimized.py:1251` | `def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow)` |
| `update` | method | `exodia_optimized.py:2245` | `def update(self)` |
| `update_channel_fatigue` | method | `exodia_optimized.py:2054` | `def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel)` |
| `update_knowledge_graph` | method | `exodia_optimized.py:1036` | `def update_knowledge_graph(self, cause, effect, strength)` |
| `update_physiology_advanced` | method | `exodia_optimized.py:1210` | `def update_physiology_advanced(self, loss_value)` |
| `visualize_fatigue_distribution` | method | `exodia_optimized.py:2278` | `def visualize_fatigue_distribution(self, epoch)` |
| `visualize_reasoning_metrics` | method | `exodia_optimized.py:2302` | `def visualize_reasoning_metrics(self, epoch)` |
| `word_overlap` | method | `exodia_optimized.py:803` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `exodia_optimized.py:969` | `def word_overlap(reference, hypothesis)` |
| `word_overlap` | method | `exodia_optimized.py:1091` | `def word_overlap(reference, hypothesis)` |
| `SinergyAnalysis` | class | `final_sinergy_analysis.py:12` | `class SinergyAnalysis` |
| `__init__` | method | `final_sinergy_analysis.py:13` | `def __init__(self)` |
| `analyze_original_models` | method | `final_sinergy_analysis.py:87` | `def analyze_original_models(self)` |
| `analyze_sinergies` | method | `final_sinergy_analysis.py:99` | `def analyze_sinergies(self)` |
| `calculate_synergy_breakthrough` | method | `final_sinergy_analysis.py:136` | `def calculate_synergy_breakthrough(self)` |
| `generate_conclusion` | method | `final_sinergy_analysis.py:171` | `def generate_conclusion(self)` |
| `generate_scientific_matrix` | method | `final_sinergy_analysis.py:118` | `def generate_scientific_matrix(self)` |
| `main` | method | `final_sinergy_analysis.py:219` | `def main()` |
| `print_header` | method | `final_sinergy_analysis.py:80` | `def print_header(self)` |
| `save_results` | method | `final_sinergy_analysis.py:200` | `def save_results(self)` |
| `CorpusCallosum` | class | `gemini.py:196` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `gemini.py:431` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphere` | class | `gemini.py:219` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `gemini.py:493` | `class LifeCycle` |
| `LiquidNeuron` | class | `gemini.py:108` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `gemini.py:361` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `gemini.py:333` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `gemini.py:177` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `gemini.py:453` | `def __getitem__(self, idx)` |
| `__init__` | method | `gemini.py:109` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `gemini.py:178` | `def __init__(self, output_dim)` |
| `__init__` | method | `gemini.py:197` | `def __init__(self, dim)` |
| `__init__` | method | `gemini.py:220` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `gemini.py:334` | `def __init__(self, vocab_size)` |
| `__init__` | method | `gemini.py:362` | `def __init__(self)` |
| `__init__` | method | `gemini.py:432` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `gemini.py:494` | `def __init__(self, total_epochs)` |
| `__len__` | method | `gemini.py:450` | `def __len__(self)` |
| `_get_init_state` | method | `gemini.py:313` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `gemini.py:318` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `gemini.py:470` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `consolidate_svd` | method | `gemini.py:154` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `gemini.py:123` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `gemini.py:187` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `gemini.py:210` | `def forward(self, right_features)` |
| `forward` | method | `gemini.py:241` | `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)` |
| `forward` | method | `gemini.py:340` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)` |
| `get_plasticity` | method | `gemini.py:497` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `gemini.py:389` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `gemini.py:373` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `gemini.py:380` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `gemini.py:394` | `def report(self, epoch)` |
| `setup_flickr8k` | function | `gemini.py:40` | `def setup_flickr8k(data_dir)` |
| `train_bicameral` | method | `gemini.py:509` | `def train_bicameral()` |
| `update` | method | `gemini.py:384` | `def update(self)` |
| `CorpusCallosum` | class | `gemini2.py:196` | `class CorpusCallosum(Module)` |
| `Flickr8kDataset` | class | `gemini2.py:431` | `class Flickr8kDataset(Dataset)` |
| `LeftHemisphere` | class | `gemini2.py:219` | `class LeftHemisphere(Module)` |
| `LifeCycle` | class | `gemini2.py:493` | `class LifeCycle` |
| `LiquidNeuron` | class | `gemini2.py:108` | `class LiquidNeuron(Module)` |
| `NeuralDiagnostics` | class | `gemini2.py:361` | `class NeuralDiagnostics` |
| `NeuroLogosBicameral` | class | `gemini2.py:333` | `class NeuroLogosBicameral(Module)` |
| `RightHemisphere` | class | `gemini2.py:177` | `class RightHemisphere(Module)` |
| `__getitem__` | method | `gemini2.py:453` | `def __getitem__(self, idx)` |
| `__init__` | method | `gemini2.py:109` | `def __init__(self, in_dim, out_dim)` |
| `__init__` | method | `gemini2.py:178` | `def __init__(self, output_dim)` |
| `__init__` | method | `gemini2.py:197` | `def __init__(self, dim)` |
| `__init__` | method | `gemini2.py:220` | `def __init__(self, vocab_size, embed_dim, hidden_dim)` |
| `__init__` | method | `gemini2.py:334` | `def __init__(self, vocab_size)` |
| `__init__` | method | `gemini2.py:362` | `def __init__(self)` |
| `__init__` | method | `gemini2.py:432` | `def __init__(self, images_dir, captions_file, vocab, transform, max_len)` |
| `__init__` | method | `gemini2.py:494` | `def __init__(self, total_epochs)` |
| `__len__` | method | `gemini2.py:450` | `def __len__(self)` |
| `_get_init_state` | method | `gemini2.py:313` | `def _get_init_state(self, visual_context)` |
| `_top_p_filtering` | method | `gemini2.py:318` | `def _top_p_filtering(self, logits, top_p)` |
| `build_vocab_flickr` | method | `gemini2.py:470` | `def build_vocab_flickr(captions_file, vocab_size)` |
| `consolidate_svd` | method | `gemini2.py:154` | `def consolidate_svd(self, repair_strength, timescale)` |
| `forward` | method | `gemini2.py:123` | `def forward(self, x, global_plasticity, transfer_rate)` |
| `forward` | method | `gemini2.py:187` | `def forward(self, image, plasticity, transfer_rate)` |
| `forward` | method | `gemini2.py:210` | `def forward(self, right_features)` |
| `forward` | method | `gemini2.py:241` | `def forward(self, visual_context, captions, max_len, prediction_error, return_gate)` |
| `forward` | method | `gemini2.py:340` | `def forward(self, image, captions, plasticity, transfer_rate, return_diagnostics)` |
| `get_plasticity` | method | `gemini2.py:497` | `def get_plasticity(self, epoch)` |
| `get_recent_avg` | method | `gemini2.py:389` | `def get_recent_avg(self, key, n)` |
| `measure_callosal_flow` | method | `gemini2.py:373` | `def measure_callosal_flow(self, right_features, left_context)` |
| `measure_vocab_diversity` | method | `gemini2.py:380` | `def measure_vocab_diversity(self, generated_tokens, vocab_size)` |
| `report` | method | `gemini2.py:394` | `def report(self, epoch)` |
| `setup_flickr8k` | function | `gemini2.py:40` | `def setup_flickr8k(data_dir)` |
| `train_bicameral` | method | `gemini2.py:509` | `def train_bicameral()` |
| `update` | method | `gemini2.py:384` | `def update(self)` |
| `generate_one` | function | `gen_dataset.py:89` | `def generate_one(key, text)` |
| `main` | function | `gen_dataset.py:110` | `def main()` |
| `compress_audios_only` | function | `get_dataset.py:249` | `def compress_audios_only()` |
| `create_audio_readme` | function | `get_dataset.py:318` | `def create_audio_readme(output_dir, metadata)` |
| `create_split_zips` | function | `get_dataset.py:637` | `def create_split_zips()` |
| `download_captions_only` | function | `get_dataset.py:29` | `def download_captions_only()` |
| `download_flickr8k` | function | `get_dataset.py:540` | `def download_flickr8k()` |
| `generate_audios` | function | `get_dataset.py:602` | `def generate_audios()` |
| `generate_audios_sync` | function | `get_dataset.py:219` | `def generate_audios_sync()` |
| `generate_audios_with_checkpoints` | function | `get_dataset.py:128` | `def generate_audios_with_checkpoints()` |
| `generate_one_audio` | function | `get_dataset.py:67` | `def generate_one_audio(text, output_path, max_retries)` |
| `generate_upload_instructions` | function | `get_dataset.py:717` | `def generate_upload_instructions(metadata)` |
| `load_checkpoint` | function | `get_dataset.py:114` | `def load_checkpoint()` |
| `main` | function | `get_dataset.py:465` | `def main()` |
| `main` | function | `get_dataset.py:888` | `def main()` |
| `save_checkpoint` | function | `get_dataset.py:122` | `def save_checkpoint(checkpoint)` |
| `upload_to_huggingface` | function | `get_dataset.py:388` | `def upload_to_huggingface(dataset_dir)` |
| `upload_to_huggingface` | function | `get_dataset.py:834` | `def upload_to_huggingface(dataset_dir)` |
| `AdaptiveLiquidMemory` | class | `homeostatichope.py:190` | `class AdaptiveLiquidMemory(Module)` |
| `Config` | class | `homeostatichope.py:15` | `class Config` |
| `ConsciousTrainer` | class | `homeostatichope.py:402` | `class ConsciousTrainer` |
| `ContinuumMemorySystem` | class | `homeostatichope.py:278` | `class ContinuumMemorySystem(Module)` |
| `HomeostaticSelfModMemory` | class | `homeostatichope.py:233` | `class HomeostaticSelfModMemory(Module)` |
| `OmniscientHopeModel` | class | `homeostatichope.py:304` | `class OmniscientHopeModel(Module)` |
| `OmniscientRegulator` | class | `homeostatichope.py:96` | `class OmniscientRegulator(Module)` |
| `RealWorldEnvironment` | class | `homeostatichope.py:49` | `class RealWorldEnvironment` |
| `__init__` | method | `homeostatichope.py:50` | `def __init__(self, seed)` |
| `__init__` | method | `homeostatichope.py:102` | `def __init__(self, d_model)` |
| `__init__` | method | `homeostatichope.py:193` | `def __init__(self, d_model)` |
| `__init__` | method | `homeostatichope.py:234` | `def __init__(self, d_model, hidden_dim)` |
| `__init__` | method | `homeostatichope.py:279` | `def __init__(self, frequencies, d_model, hidden_dim)` |
| `__init__` | method | `homeostatichope.py:305` | `def __init__(self, config, n_features, n_classes)` |
| `__init__` | method | `homeostatichope.py:403` | `def __init__(self, model, config, device)` |
| `evaluate` | method | `homeostatichope.py:488` | `def evaluate(self, test_loader, epsilon, phase)` |
| `forward` | method | `homeostatichope.py:128` | `def forward(self, signals)` |
| `forward` | method | `homeostatichope.py:203` | `def forward(self, x, controls)` |
| `forward` | method | `homeostatichope.py:251` | `def forward(self, x, controls)` |
| `forward` | method | `homeostatichope.py:293` | `def forward(self, x, global_step)` |
| `forward` | method | `homeostatichope.py:341` | `def forward(self, x, signals, global_step)` |
| `get_batch` | method | `homeostatichope.py:73` | `def get_batch(self, phase, batch_size)` |
| `get_test_loader` | method | `homeostatichope.py:88` | `def get_test_loader(self, batch_size)` |
| `pgd_attack` | method | `homeostatichope.py:368` | `def pgd_attack(model, x, y, epsilon, steps, device, signals)` |
| `run_ablation` | method | `homeostatichope.py:610` | `def run_ablation(device)` |
| `run_conscious_experiment` | method | `homeostatichope.py:516` | `def run_conscious_experiment(config, device)` |
| `set_seed` | method | `homeostatichope.py:40` | `def set_seed(seed)` |
| `setup_device` | method | `homeostatichope.py:35` | `def setup_device()` |
| `train_step` | method | `homeostatichope.py:425` | `def train_step(self, x, y, epsilon, global_step, phase)` |
| `AdversarialTrainer` | class | `hope.py:441` | `class AdversarialTrainer` |
| `Config` | class | `hope.py:16` | `class Config` |
| `ContinuumMemorySystem` | class | `hope.py:287` | `class ContinuumMemorySystem(Module)` |
| `EfficientSelfModMemory` | class | `hope.py:207` | `class EfficientSelfModMemory(Module)` |
| `HomeostaticRegulator` | class | `hope.py:127` | `class HomeostaticRegulator(Module)` |
| `HopePhysioModel` | class | `hope.py:320` | `class HopePhysioModel(Module)` |
| `LiquidMemory` | class | `hope.py:172` | `class LiquidMemory(Module)` |
| `RealWorldEnvironment` | class | `hope.py:58` | `class RealWorldEnvironment` |
| `__init__` | method | `hope.py:64` | `def __init__(self, seed)` |
| `__init__` | method | `hope.py:130` | `def __init__(self, d_model)` |
| `__init__` | method | `hope.py:175` | `def __init__(self, d_model)` |
| `__init__` | method | `hope.py:210` | `def __init__(self, d_model, hidden_dim)` |
| `__init__` | method | `hope.py:290` | `def __init__(self, frequencies, d_model, hidden_dim)` |
| `__init__` | method | `hope.py:323` | `def __init__(self, config, n_features, n_classes)` |
| `__init__` | method | `hope.py:442` | `def __init__(self, model, config, device)` |
| `evaluate` | method | `hope.py:490` | `def evaluate(self, test_loader, epsilon)` |
| `forward` | method | `hope.py:140` | `def forward(self, x, h_prev, w_norm)` |
| `forward` | method | `hope.py:186` | `def forward(self, x, physio)` |
| `forward` | method | `hope.py:236` | `def forward(self, x)` |
| `forward` | method | `hope.py:304` | `def forward(self, x, global_step)` |
| `forward` | method | `hope.py:365` | `def forward(self, x, global_step)` |
| `get_batch` | method | `hope.py:98` | `def get_batch(self, phase, batch_size)` |
| `get_test_loader` | method | `hope.py:118` | `def get_test_loader(self, batch_size)` |
| `pgd_attack` | method | `hope.py:394` | `def pgd_attack(model, x, y, epsilon, steps, device)` |
| `reset_states` | method | `hope.py:361` | `def reset_states(self)` |
| `run_ablation` | method | `hope.py:614` | `def run_ablation(device)` |
| `run_real_world_experiment` | method | `hope.py:517` | `def run_real_world_experiment(config, device)` |
| `set_seed` | method | `hope.py:49` | `def set_seed(seed)` |
| `setup_device` | method | `hope.py:41` | `def setup_device()` |
| `train_step` | method | `hope.py:460` | `def train_step(self, x, y, epsilon, global_step)` |
| `BCMRegulated` | class | `kimi.py:97` | `class BCMRegulated(Module)` |
| `Config` | class | `kimi.py:185` | `class Config` |
| `LiquidRegulated` | class | `kimi.py:118` | `class LiquidRegulated(Module)` |
| `MicroTopoBrainSNA` | class | `kimi.py:166` | `class MicroTopoBrainSNA(Module)` |
| `PhysioState` | class | `kimi.py:28` | `class PhysioState` |
| `SNE` | class | `kimi.py:65` | `class SNE(Module)` |
| `VisualCortexRegulated` | class | `kimi.py:145` | `class VisualCortexRegulated(Module)` |
| `__init__` | method | `kimi.py:66` | `def __init__(self, enabled)` |
| `__init__` | method | `kimi.py:98` | `def __init__(self, sne, ablated)` |
| `__init__` | method | `kimi.py:119` | `def __init__(self, sne, ablated)` |
| `__init__` | method | `kimi.py:146` | `def __init__(self, sne, ablated)` |
| `__init__` | method | `kimi.py:167` | `def __init__(self, sne_enabled, ablated_organs)` |
| `forward` | method | `kimi.py:75` | `def forward(self, state, loss)` |
| `forward` | method | `kimi.py:104` | `def forward(self, act)` |
| `forward` | method | `kimi.py:127` | `def forward(self, x)` |
| `forward` | method | `kimi.py:155` | `def forward(self, img)` |
| `forward` | method | `kimi.py:175` | `def forward(self, x)` |
| `get_loader` | method | `kimi.py:191` | `def get_loader()` |
| `pgd_attack` | method | `kimi.py:39` | `def pgd_attack(model, x, y, eps, steps, alpha)` |
| `run_experiment` | method | `kimi.py:201` | `def run_experiment(seed, sne_enabled, ablated_organs)` |
| `scientific_ablation` | method | `kimi.py:256` | `def scientific_ablation()` |
| `ConsciousnessModule` | class | `legendario.py:224` | `class ConsciousnessModule(OmniBrainModule)` |
| `DualMindModule` | class | `legendario.py:193` | `class DualMindModule(OmniBrainModule)` |
| `MotorHomeostaticContext` | class | `legendario.py:106` | `class MotorHomeostaticContext` |
| `OmniBrain` | class | `legendario.py:289` | `class OmniBrain(Module)` |
| `OmniBrainCoordinator` | class | `legendario.py:245` | `class OmniBrainCoordinator` |
| `OmniBrainModule` | class | `legendario.py:117` | `class OmniBrainModule(Module)` |
| `PTSymmetricLayer` | class | `legendario.py:132` | `class PTSymmetricLayer(OmniBrainModule)` |
| `TopologicalLayer` | class | `legendario.py:166` | `class TopologicalLayer(OmniBrainModule)` |
| `__init__` | method | `legendario.py:119` | `def __init__(self, module_name, enabled)` |
| `__init__` | method | `legendario.py:135` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `legendario.py:169` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `legendario.py:196` | `def __init__(self, features)` |
| `__init__` | method | `legendario.py:227` | `def __init__(self, features)` |
| `__init__` | method | `legendario.py:248` | `def __init__(self)` |
| `__init__` | method | `legendario.py:292` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `compute_phi_effective_approx` | function | `legendario.py:27` | `def compute_phi_effective_approx(activity)` |
| `compute_pt_phase` | method | `legendario.py:144` | `def compute_pt_phase(self)` |
| `compute_topological_metrics` | function | `legendario.py:60` | `def compute_topological_metrics(weights)` |
| `estimate_energy_consumption` | function | `legendario.py:89` | `def estimate_energy_consumption(model, input_size)` |
| `forward` | method | `legendario.py:152` | `def forward(self, x, params)` |
| `forward` | method | `legendario.py:183` | `def forward(self, x, params)` |
| `forward` | method | `legendario.py:211` | `def forward(self, x, params)` |
| `forward` | method | `legendario.py:232` | `def forward(self, x, params)` |
| `forward` | method | `legendario.py:317` | `def forward(self, x)` |
| `measure_network_state` | method | `legendario.py:251` | `def measure_network_state(self, model, batch_data)` |
| `train_omni_brain` | method | `legendario.py:340` | `def train_omni_brain(model, epochs, batch_size, device)` |
| `update_performance` | method | `legendario.py:125` | `def update_performance(self, metrics)` |
| `update_topology` | method | `legendario.py:176` | `def update_topology(self, connectivity)` |
| `AdaptiveLearningMotor` | class | `legendario2.py:213` | `class AdaptiveLearningMotor(MotorHomeostaticContext)` |
| `ConsciousnessModule` | class | `legendario2.py:640` | `class ConsciousnessModule(OmniBrainModule)` |
| `ConsciousnessMotor` | class | `legendario2.py:157` | `class ConsciousnessMotor(MotorHomeostaticContext)` |
| `DualMindModule` | class | `legendario2.py:557` | `class DualMindModule(OmniBrainModule)` |
| `DualSystemMotor` | class | `legendario2.py:185` | `class DualSystemMotor(MotorHomeostaticContext)` |
| `EnergyHomeostaticMotor` | class | `legendario2.py:127` | `class EnergyHomeostaticMotor(MotorHomeostaticContext)` |
| `HomeostaticEngine` | class | `legendario2.py:701` | `class HomeostaticEngine` |
| `ModularActivationMotor` | class | `legendario2.py:241` | `class ModularActivationMotor(MotorHomeostaticContext)` |
| `MotorHomeostaticContext` | class | `legendario2.py:34` | `class MotorHomeostaticContext` |
| `OmniBrain` | class | `legendario2.py:732` | `class OmniBrain(Module)` |
| `OmniBrainCoordinator` | class | `legendario2.py:291` | `class OmniBrainCoordinator` |
| `OmniBrainModule` | class | `legendario2.py:452` | `class OmniBrainModule(Module)` |
| `PTSymmetricLayer` | class | `legendario2.py:467` | `class PTSymmetricLayer(OmniBrainModule)` |
| `PTSymmetricMotor` | class | `legendario2.py:64` | `class PTSymmetricMotor(MotorHomeostaticContext)` |
| `TopologicalLayer` | class | `legendario2.py:500` | `class TopologicalLayer(OmniBrainModule)` |
| `TopologicalMotor` | class | `legendario2.py:100` | `class TopologicalMotor(MotorHomeostaticContext)` |
| `__init__` | method | `legendario2.py:66` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:102` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:129` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:159` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:187` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:215` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:243` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:294` | `def __init__(self)` |
| `__init__` | method | `legendario2.py:455` | `def __init__(self, module_name, enabled)` |
| `__init__` | method | `legendario2.py:470` | `def __init__(self, in_features, out_features)` |
| `__init__` | method | `legendario2.py:503` | `def __init__(self, in_features, out_features, sparsity_factor)` |
| `__init__` | method | `legendario2.py:560` | `def __init__(self, features)` |
| `__init__` | method | `legendario2.py:643` | `def __init__(self, features)` |
| `__init__` | method | `legendario2.py:704` | `def __init__(self, target_performance)` |
| `__init__` | method | `legendario2.py:735` | `def __init__(self, input_dim, hidden_dim, output_dim)` |
| `_generate_topology_mask` | method | `legendario2.py:518` | `def _generate_topology_mask(self)` |
| `_initialize_motors` | method | `legendario2.py:300` | `def _initialize_motors(self)` |
| `compute_phi_effective` | method | `legendario2.py:657` | `def compute_phi_effective(self, x)` |
| `coordinate_all_motors` | method | `legendario2.py:377` | `def coordinate_all_motors(self, environment_state, network_state)` |
| `forward` | method | `legendario2.py:461` | `def forward(self, x, params)` |
| `forward` | method | `legendario2.py:477` | `def forward(self, x, params)` |
| `forward` | method | `legendario2.py:539` | `def forward(self, x, params)` |
| `forward` | method | `legendario2.py:585` | `def forward(self, x, params)` |
| `forward` | method | `legendario2.py:677` | `def forward(self, x, params)` |
| `forward` | method | `legendario2.py:834` | `def forward(self, x)` |
| `get_status_report` | method | `legendario2.py:932` | `def get_status_report(self)` |
| `initialize_context` | method | `legendario2.py:820` | `def initialize_context(self)` |
| `measure_network_state` | method | `legendario2.py:340` | `def measure_network_state(self, model, batch_data)` |
| `prepare_for_inference` | method | `legendario2.py:803` | `def prepare_for_inference(self)` |
| `regulate_connectivity` | method | `legendario2.py:112` | `def regulate_connectivity(self, current_connectivity, clustering)` |
| `regulate_consciousness` | method | `legendario2.py:168` | `def regulate_consciousness(self, phi_effective, integration_level)` |
| `regulate_dual_systems` | method | `legendario2.py:197` | `def regulate_dual_systems(self, unconscious_activity, conscious_activity)` |
| `regulate_energy` | method | `legendario2.py:139` | `def regulate_energy(self, memory_usage, cpu_usage, temperature)` |
| `regulate_homeostasis` | method | `legendario2.py:709` | `def regulate_homeostasis(self, observed_performance)` |
| `regulate_learning` | method | `legendario2.py:224` | `def regulate_learning(self, loss_reduction_rate, gradient_norm)` |
| `regulate_modules` | method | `legendario2.py:259` | `def regulate_modules(self, task_complexity, resource_availability, performance)` |
| `regulate_parameters` | method | `legendario2.py:78` | `def regulate_parameters(self, current_coherence, energy_level)` |
| `reset_internal_states` | method | `legendario2.py:769` | `def reset_internal_states(self)` |
| `sense_environment` | method | `legendario2.py:312` | `def sense_environment(self)` |
| `simulate_network_state` | method | `legendario2.py:326` | `def simulate_network_state(self)` |
| `train_omni_brain` | method | `legendario2.py:967` | `def train_omni_brain(model, epochs, batch_size)` |
| `update` | method | `legendario2.py:47` | `def update(self, measurement, dt)` |
| `update_performance` | method | `legendario2.py:464` | `def update_performance(self, metrics)` |
| `Config` | class | `live_cl.py:32` | `class Config` |
| `DualSystemModule` | class | `live_cl.py:206` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `live_cl.py:133` | `class FastSlowLinear(Module)` |
| `IntegrationModule` | class | `live_cl.py:242` | `class IntegrationModule(Module)` |
| `OmniBrain` | class | `live_cl.py:278` | `class OmniBrain(Module)` |
| `__init__` | method | `live_cl.py:138` | `def __init__(self, in_features, out_features, config)` |
| `__init__` | method | `live_cl.py:211` | `def __init__(self, dim, config)` |
| `__init__` | method | `live_cl.py:247` | `def __init__(self, features, config)` |
| `__init__` | method | `live_cl.py:283` | `def __init__(self, config)` |
| `compute_integration_index` | method | `live_cl.py:99` | `def compute_integration_index(activity)` |
| `evaluate` | method | `live_cl.py:393` | `def evaluate(model, loader, device)` |
| `forward` | method | `live_cl.py:189` | `def forward(self, x)` |
| `forward` | method | `live_cl.py:223` | `def forward(self, x)` |
| `forward` | method | `live_cl.py:260` | `def forward(self, x)` |
| `forward` | method | `live_cl.py:318` | `def forward(self, x)` |
| `get_ablation_state` | method | `live_cl.py:336` | `def get_ablation_state(self)` |
| `get_data_loaders` | method | `live_cl.py:349` | `def get_data_loaders(config)` |
| `get_fast_norm` | method | `live_cl.py:202` | `def get_fast_norm(self)` |
| `get_fast_norms` | method | `live_cl.py:331` | `def get_fast_norms(self)` |
| `reset_all_fast_weights` | method | `live_cl.py:325` | `def reset_all_fast_weights(self)` |
| `reset_fast_weights` | method | `live_cl.py:156` | `def reset_fast_weights(self)` |
| `run_ablation_study` | method | `live_cl.py:570` | `def run_ablation_study(quick_test)` |
| `set_seed` | method | `live_cl.py:84` | `def set_seed(seed)` |
| `setup_logging` | method | `live_cl.py:69` | `def setup_logging()` |
| `to_dict` | method | `live_cl.py:62` | `def to_dict(self)` |
| `train` | method | `live_cl.py:430` | `def train(config, silent)` |
| `update_fast_weights` | method | `live_cl.py:162` | `def update_fast_weights(self, x, slow_out)` |
| `Config` | class | `live_go.py:30` | `class Config` |
| `DualSystemModule` | class | `live_go.py:143` | `class DualSystemModule(Module)` |
| `FastSlowLinear` | class | `live_go.py:93` | `class FastSlowLinear(Module)` |
| `IntegrationModule` | class | `live_go.py:167` | `class IntegrationModule(Module)` |
| `OmniBrainGenesis` | class | `live_go.py:190` | `class OmniBrainGenesis(Module)` |

Next: [SYMBOLS_p5.md](SYMBOLS_p5.md)

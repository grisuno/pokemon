# =============================================================================
# NeuroLogos Bicameral FISIOLÓGICO v4.1
# + Métricas lingüísticas (BLEU, CIDEr, SPICE)
# + Sistema médico calibrado por niveles
# + Sistema cognitivo con razonamiento (MTP + Chain-of-Thought)
# + Memoria Episódica con replay ponderado
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from collections import Counter
import torchvision.models as models
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def compute_loss(logits, captions, gate, vocab, mtp_loss=None, linguistic_reward=None, lambda_reward=0.1, lambda_mtp=0.1):
    """
    Función de pérdida extendida con MTP y recompensa lingüística
    """
    # Pérdida de entropía cruzada estándar
    ce_loss = F.cross_entropy(
        logits.reshape(-1, len(vocab)),
        captions[:, 1:].reshape(-1),
        ignore_index=vocab['<PAD>']
    )
    
    # Penalizaciones del gate
    gate_mean = gate.mean()
    gate_penalty = F.relu(gate_mean - 0.5) ** 2
    gate_diversity = gate.std()
    diversity_penalty = F.relu(0.15 - gate_diversity) ** 2
    
    # Término de recompensa lingüística
    linguistic_loss = 0.0
    if linguistic_reward is not None:
        linguistic_loss = -lambda_reward * linguistic_reward
    
    # Término MTP (predicción multi-token)
    mtp_term = 0.0
    if mtp_loss is not None and isinstance(mtp_loss, torch.Tensor):
        mtp_term = lambda_mtp * mtp_loss
    
    # Combinación de todos los términos
    total_loss = ce_loss + 0.05 * gate_penalty + 0.2 * diversity_penalty + linguistic_loss + mtp_term
    
    return total_loss, ce_loss, gate_penalty, diversity_penalty, linguistic_loss, mtp_term


# =============================================================================
# MEMORIA EPISÓDICA (REINTEGRADA)
# =============================================================================
class EpisodicMemoryBuffer:
    """Buffer que almacena ejemplos sorpresivos para replay estratégico"""
    
    def __init__(self, capacity=500, surprise_threshold=0.3):
        self.capacity = capacity
        self.surprise_threshold = surprise_threshold
        self.buffer = []
        self.surprise_scores = []
        
    def compute_surprise(self, predicted_logits, ground_truth, gate_mean):
        """Calcula sorpresa basada en error y apertura del gate"""
        with torch.no_grad():
            ce = F.cross_entropy(
                predicted_logits.reshape(-1, predicted_logits.size(-1)),
                ground_truth.reshape(-1),
                reduction='none'
            ).mean()
            
            surprise = ce * (1.0 - gate_mean)
            return surprise.item()
    
    def add(self, image, caption, surprise_score):
        """Añade ejemplo si supera umbral y hay capacidad"""
        if surprise_score > self.surprise_threshold:
            if len(self.buffer) >= self.capacity:
                # Reemplazar el menos sorprendente
                min_idx = np.argmin(self.surprise_scores)
                self.buffer.pop(min_idx)
                self.surprise_scores.pop(min_idx)
            
            self.buffer.append((image, caption))
            self.surprise_scores.append(surprise_score)
    
    def sample(self, batch_size):
        """Samplea ejemplos con probabilidad proporcional a sorpresa"""
        if len(self.buffer) == 0:
            return None
        
        probs = np.array(self.surprise_scores)
        probs = probs / probs.sum()
        
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            p=probs,
            replace=False
        )
        
        return [self.buffer[i] for i in indices]


# =============================================================================
# SISTEMA DE RAZONAMIENTO COGNITIVO (VERSIÓN COMPLETA)
# =============================================================================
class NeurocognitiveSystem:
    def __init__(self):
        self.cognitive_history = []
        self.last_intervention_epoch = -5
        self.linguistic_feedback = LinguisticFeedbackLoop()
        
        # Umbrales
        self.cider_threshold = 0.1
        self.spice_threshold = 0.15
        self.plateau_threshold = 0.005
        
        # Nuevos umbrales de razonamiento
        self.reasoning_efficiency_threshold = 0.3
        self.mtp_convergence_threshold = 0.2
        self.logical_coherence_threshold = 0.4
    
    def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch):
        """Evalúa estado del sistema de razonamiento"""
        issues = []
        severity = 0
        confidence = []
        
        # Detectar ineficiencia en razonamiento
        if reasoning_steps is not None:
            steps_mean = reasoning_steps.mean().item() if hasattr(reasoning_steps, 'mean') else float(reasoning_steps)
            if steps_mean > 4:
                issues.append("excessive_reasoning")
                severity += 2
                confidence.append(f"Razonamiento excesivo (pasos: {steps_mean:.1f})")
            elif steps_mean < 1 and epoch > 5:
                issues.append("insufficient_reasoning")
                severity += 2
                confidence.append(f"Razonamiento insuficiente (pasos: {steps_mean:.1f})")
        
        # Detectar divergencia MTP
        if mtp_loss is not None and isinstance(mtp_loss, torch.Tensor):
            if mtp_loss.item() > self.mtp_convergence_threshold:
                issues.append("mtp_divergence")
                severity += 3
                confidence.append(f"Divergencia MTP (pérdida: {mtp_loss.item():.3f})")
        
        # Detectar baja coherencia
        if logical_coherence < self.logical_coherence_threshold:
            issues.append("logical_incoherence")
            severity += 3
            confidence.append(f"Incoherencia lógica (coherencia: {logical_coherence:.3f})")
        
        self.cognitive_history.append({
            'epoch': epoch,
            'mtp_loss': mtp_loss.item() if mtp_loss is not None else None,
            'reasoning_steps': reasoning_steps.mean().item() if reasoning_steps is not None and hasattr(reasoning_steps, 'mean') else None,
            'logical_coherence': logical_coherence,
            'issues': issues,
            'severity': severity
        })
        
        return issues, severity, confidence
    
    def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch):
        """Evalúa estado cognitivo lingüístico"""
        issues = []
        severity = 0
        confidence = []
        
        # Detectar estancamiento
        if epoch > 5 and len(self.cognitive_history) > 3:
            recent_cider = [h.get('cider', 0) for h in self.cognitive_history[-3:]]
            if len(recent_cider) >= 3 and recent_cider[-1] is not None:
                cider_improvement = recent_cider[-1] - recent_cider[0]
                if cider_improvement < self.plateau_threshold:
                    issues.append("cider_plateau")
                    severity += 2
                    confidence.append(f"CIDEr estancado (mejora: {cider_improvement:.3f})")
        
        # Detectar déficit semántico
        if spice_score < self.spice_threshold:
            issues.append("semantic_deficit")
            severity += 3
            confidence.append(f"Déficit semántico (SPICE: {spice_score:.3f})")
        
        # Detectar déficit general
        if combined_reward < 0.2:
            issues.append("linguistic_deficit")
            severity += 2
            confidence.append(f"Déficit lingüístico (recompensa: {combined_reward:.3f})")
        
        # Detectar sobreajuste sintáctico
        if cider_score > 0.15 and spice_score < self.spice_threshold:
            issues.append("syntactic_overfitting")
            severity += 2
            confidence.append(f"Sobreajuste sintáctico")
        
        return issues, severity, confidence
    
    def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics=None):
        """Aplica intervenciones basadas en estado lingüístico y razonamiento"""
        if epoch - self.last_intervention_epoch < 1 or severity == 0:
            return False
        
        # Determinar nivel
        level = "🔴 Nivel 3" if severity > 5 else "🟠 Nivel 2" if severity > 2 else "🟡 Nivel 1"
        
        print(f"\n{'='*80}")
        print(f"🧠 INTERVENCIÓN COGNITIVA - {level} - Severidad: {severity}/9")
        print(f"   Problemas: {', '.join(issues)}")
        print(f"   Confianza: {confidence}")
        print(f"{'='*80}")
        
        interventions_applied = []
        
        with torch.no_grad():
            # INTERVENCIONES DE RAZONAMIENTO
            if "excessive_reasoning" in issues:
                print("💡 Regularizando controlador de razonamiento")
                model.left_hemisphere.reasoning_controller[-1].weight.data *= 0.8
                model.left_hemisphere.reasoning_controller[-1].bias.data += 0.2
                interventions_applied.append("reasoning_regularization")
            
            if "insufficient_reasoning" in issues:
                print("💡 Sensibilizando controlador de razonamiento")
                model.left_hemisphere.reasoning_controller[-1].weight.data *= 1.2
                model.left_hemisphere.reasoning_controller[-1].bias.data -= 0.2
                interventions_applied.append("reasoning_sensitization")
            
            if "mtp_divergence" in issues:
                print("💡 Estabilizando MTP")
                model.left_hemisphere.lambda_mtp.data *= 0.7
                for block in model.left_hemisphere.mtp_transformer_blocks:
                    for param in block.parameters():
                        if param.dim() > 1:
                            nn.init.xavier_uniform_(param, gain=0.8)
                interventions_applied.append("mtp_stabilization")
            
            if "logical_incoherence" in issues:
                print("💡 Reforzando atención de razonamiento")
                model.left_hemisphere.reasoning_attention.in_proj_weight.data *= 1.1
                if hasattr(model.left_hemisphere, 'reasoning_projection'):
                    model.left_hemisphere.reasoning_projection.weight.data *= 1.05
                interventions_applied.append("reasoning_attention_boost")
            
            # INTERVENCIONES LINGÜÍSTICAS
            if "cider_plateau" in issues:
                print("💡 Reforzando atención visual")
                model.left_hemisphere.visual_attention.in_proj_weight.data *= 1.1
                if hasattr(model.left_hemisphere, 'liquid_gate'):
                    model.left_hemisphere.liquid_gate[-1].bias.data -= 0.2
                interventions_applied.append("visual_attention_boost")
            
            if "semantic_deficit" in issues:
                print("💡 Reforzando proyección semántica")
                model.left_hemisphere.output_projection.weight.data *= 1.15
                model.corpus_callosum.residual_scale_base.data *= 1.02
                interventions_applied.append("semantic_projection_boost")
            
            if "linguistic_deficit" in issues:
                print("💡 Reforzando atención visual + callosum")
                model.corpus_callosum.residual_scale_base.data *= 1.05
                interventions_applied.append("callosum_alignment_boost")
            
            if "syntactic_overfitting" in issues:
                print("💡 Regularizando sobreajuste sintáctico")
                if hasattr(model.left_hemisphere, 'liquid_gate'):
                    for layer in model.left_hemisphere.liquid_gate:
                        if isinstance(layer, nn.Dropout):
                            layer.p = min(0.5, layer.p + 0.05)
                    model.left_hemisphere.liquid_gate[-1].bias.data += 0.3
                interventions_applied.append("syntactic_overfitting_regularization")
        
        print(f"\n✓ Intervenciones aplicadas: {len(interventions_applied)}")
        for inter in interventions_applied:
            print(f"  - {inter}")
        print(f"{'='*80}\n")
        
        self.last_intervention_epoch = epoch
        return True

# =============================================================================
# SISTEMA DE RETROALIMENTACIÓN LINGÜÍSTICA
# =============================================================================
class LinguisticFeedbackLoop:
    """Sistema de caché optimizado para métricas lingüísticas"""
    
    def __init__(self, alpha=0.7, beta=0.3):
        self.alpha = alpha  # Peso CIDEr
        self.beta = beta    # Peso SPICE
        self.history = []
        
        # Caché de dos niveles
        self.ngram_cache = {}
        self.ngram_cache_hits = 0
        self.ngram_cache_misses = 0
        
        self.score_cache = {}
        self.score_cache_hits = 0
        self.score_cache_misses = 0
        
    def compute_linguistic_reward(self, references, hypotheses):
        """Recompensa combinada CIDEr + SPICE con caché"""
        cider_scores = []
        spice_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            pair_key = (hash(ref), hash(hyp))
            
            # Nivel 2: Caché de puntajes
            if pair_key in self.score_cache:
                cached_cider, cached_spice = self.score_cache[pair_key]
                cider_scores.append(cached_cider)
                spice_scores.append(cached_spice)
                self.score_cache_hits += 1
            else:
                cider_score = self.compute_cider(ref, hyp)
                spice_score = self.compute_spice(ref, hyp)
                
                self.score_cache[pair_key] = (cider_score, spice_score)
                cider_scores.append(cider_score)
                spice_scores.append(spice_score)
                self.score_cache_misses += 1
        
        combined = [self.alpha * c + self.beta * s for c, s in zip(cider_scores, spice_scores)]
        reward = torch.tensor(combined).mean().clamp(0, 1)
        
        self.history.append({
            'cider': np.mean(cider_scores),
            'spice': np.mean(spice_scores),
            'combined': reward.item()
        })
        
        return reward
    
    def compute_cider(self, reference, hypothesis):
        """CIDEr simplificado con caché de n-gramas"""
        ref_key = hash(reference)
        if ref_key in self.ngram_cache:
            ref_ngrams = self.ngram_cache[ref_key]
            self.ngram_cache_hits += 1
        else:
            ref_ngrams = self._get_ngrams(reference, n=4)
            self.ngram_cache[ref_key] = ref_ngrams
            self.ngram_cache_misses += 1
        
        hyp_key = hash(hypothesis)
        if hyp_key in self.ngram_cache:
            hyp_ngrams = self.ngram_cache[hyp_key]
            self.ngram_cache_hits += 1
        else:
            hyp_ngrams = self._get_ngrams(hypothesis, n=4)
            self.ngram_cache[hyp_key] = hyp_ngrams
            self.ngram_cache_misses += 1
        
        overlap = sum(min(ref_ngrams.get(n, 0), hyp_ngrams.get(n, 0)) for n in hyp_ngrams)
        ref_len = sum(ref_ngrams.values())
        hyp_len = sum(hyp_ngrams.values())
        
        if ref_len == 0 or hyp_len == 0:
            return 0.0
        
        return overlap / (ref_len * hyp_len) ** 0.5
    
    def compute_spice(self, reference, hypothesis):
        """SPICE simplificado (Jaccard similarity)"""
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        
        if len(ref_words) == 0 and len(hyp_words) == 0:
            return 1.0
        
        intersection = len(ref_words & hyp_words)
        union = len(ref_words | hyp_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_ngrams(self, sentence, n=4):
        """Extractor de n-gramas"""
        tokens = sentence.lower().split()
        return {tuple(tokens[i:i+n]): tokens[i:i+n].count(tokens[i]) for i in range(len(tokens)-n+1)}
    
    def get_cache_stats(self):
        """Estadísticas de caché"""
        total_ngram = self.ngram_cache_hits + self.ngram_cache_misses
        total_score = self.score_cache_hits + self.score_cache_misses
        
        return {
            'ngram_cache_size': len(self.ngram_cache),
            'ngram_hits': self.ngram_cache_hits,
            'ngram_misses': self.ngram_cache_misses,
            'ngram_hit_rate': self.ngram_cache_hits / total_ngram if total_ngram > 0 else 0.0,
            'score_cache_size': len(self.score_cache),
            'score_hits': self.score_cache_hits,
            'score_misses': self.score_cache_misses,
            'score_hit_rate': self.score_cache_hits / total_score if total_score > 0 else 0.0,
            'cache_size': len(self.ngram_cache) + len(self.score_cache),
            'cache_hits': self.ngram_cache_hits + self.score_cache_hits,
            'cache_misses': self.ngram_cache_misses + self.score_cache_misses,
            'hit_rate': (self.ngram_cache_hits + self.score_cache_hits) / (total_ngram + total_score) if (total_ngram + total_score) > 0 else 0.0
        }


# =============================================================================
# MÉTRICAS LINGÜÍSTICAS BÁSICAS
# =============================================================================
class LanguageMetrics:
    """Métricas clásicas de evaluación"""
    
    @staticmethod
    def sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
        """BLEU-4 a nivel de oración"""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        precisions = []
        for n in range(1, 5):
            ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
            hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
            
            if len(hyp_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            matches = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            precisions.append(matches / total if total > 0 else 0.0)
        
        bp = min(1.0, np.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens))))
        
        if all(p > 0 for p in precisions):
            score = bp * np.exp(sum(w * np.log(p) for w, p in zip(weights, precisions)))
        else:
            score = 0.0
        
        return score
    
    @staticmethod
    def token_accuracy(reference, hypothesis):
        """Precisión token-level"""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        min_len = min(len(ref_tokens), len(hyp_tokens))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) if ref_tokens[i] == hyp_tokens[i])
        return matches / max(len(ref_tokens), len(hyp_tokens))
    
    @staticmethod
    def word_overlap(reference, hypothesis):
        """Jaccard similarity"""
        ref_set = set(reference.lower().split())
        hyp_set = set(hypothesis.lower().split())
        
        if len(ref_set | hyp_set) == 0:
            return 0.0
        
        return len(ref_set & hyp_set) / len(ref_set | hyp_set)


# =============================================================================
# SISTEMA MÉDICO TRIANGULADO (VERSIÓN COMPLETA Y FUNCIONAL)
# =============================================================================
class TriangulatedMedicalSystem:
    def __init__(self):
        self.intervention_history = []
        self.last_intervention_epoch = -5
        self.signal_history = []
        self.gate_zombie_threshold = 0.65
    
    def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow):
        signals = {
            'gate_saturated': gate_mean > self.gate_zombie_threshold,
            'gate_no_diversity': gate_std < 0.05,
            'callosum_blocked': callosal_flow < 0.35,
            'liquid_high': liquid_norm > 2.5,
            'homeostasis_low': health_score <= 2
        }
        return signals
    
    def count_convergent_signals(self, signals, pattern):
        return sum(signals[sig] for sig in pattern if sig in signals)
    
    def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow):
        signals = self.triangulate_signals(health_score, liquid_norm, gate_mean, gate_std, callosal_flow)
        
        issues = []
        severity = 0
        confidence = []
        
        # Patrón 1: Gate roto (3/3 señales)
        if self.count_convergent_signals(signals, ['gate_saturated', 'gate_no_diversity', 'callosal_flow']) >= 3:
            issues.append("gate_system_failure")
            severity += 6
            confidence.append("Gate roto (100% confianza)")
        elif self.count_convergent_signals(signals, ['gate_saturated', 'gate_no_diversity', 'callosal_flow']) == 2:
            issues.append("gate_degraded")
            severity += 4
            confidence.append("Gate degradado (67% confianza)")
        
        # Patrón 2: Comunicación colapsada
        if signals['callosum_blocked'] and signals['gate_no_diversity']:
            issues.append("communication_collapse")
            severity += 4
            confidence.append("Comunicación colapsada (100% confianza)")
        
        # Patrón 3: Liquid crisis
        if signals['liquid_high'] and signals['homeostasis_low']:
            issues.append("liquid_crisis")
            severity += 5
            confidence.append("Crisis líquida (100% confianza)")
        elif signals['liquid_high']:
            issues.append("liquid_elevated")
            severity += 2
            confidence.append("Líquido alto (50% confianza)")
        
        self.signal_history.append({
            'signals': signals,
            'issues': issues,
            'severity': severity,
            'confidence': confidence
        })
        
        return issues, severity, confidence
    
    def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch):
        if epoch - self.last_intervention_epoch < 1 or severity == 0:
            return False
        
        level = "🔴 Nivel 3" if severity > 6 else "🟠 Nivel 2" if severity > 3 else "🟡 Nivel 1"
        
        print(f"\n{'='*80}")
        print(f"🏥 INTERVENCIÓN MÉDICA - {level} - Severidad: {severity}/12")
        print(f"   Problemas: {', '.join(issues)}")
        print(f"   Confianza: {confidence}")
        print(f"{'='*80}")
        
        interventions_applied = []
        
        with torch.no_grad():
            right_node = model.right_hemisphere.spatial_liquid
            
            # Patrón 1: Gate sistema fallido
            if "gate_system_failure" in issues:
                print("🚨 DEMOLICION TOTAL DEL GATE")
                for layer in model.left_hemisphere.liquid_gate:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.05)
                        layer.bias.data.zero_()
                interventions_applied.append("gate_total_demolition")
            
            # Patrón 2: Gate degradado
            elif "gate_degraded" in issues:
                print("💊 RESET AGRESIVO DEL GATE")
                for layer in model.left_hemisphere.liquid_gate:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.15)
                model.left_hemisphere.liquid_gate[-1].bias.data.fill_(0.0)
                interventions_applied.append("gate_aggressive_reset")
            
            # Patrón 3: Comunicación colapsada
            if "communication_collapse" in issues:
                print("💊 REFORZAR CORPUS CALLOSUM")
                model.corpus_callosum.residual_scale_base.data *= 1.1
                interventions_applied.append("callosum_boost_strong")
            
            # Patrón 4: Liquid crisis
            if "liquid_crisis" in issues:
                print("🚨 RESET TOTAL LIQUID")
                self._reset_liquid_neuron(right_node, severity)
                interventions_applied.append("liquid_full_reset")
            elif "liquid_elevated" in issues:
                right_node.W_fast_short *= 0.4
                right_node.W_fast_long *= 0.4
                interventions_applied.append("liquid_reduce")
            
            # Mantenimiento
            if severity >= 3:
                right_node.fatigue *= 0.5
                interventions_applied.append("fatigue_reduction")
        
        print(f"\n✓ Aplicadas {len(interventions_applied)} intervenciones")
        for inter in interventions_applied:
            print(f"  - {inter}")
        print(f"{'='*80}\n")
        
        self.intervention_history.append({
            'epoch': epoch,
            'severity': severity,
            'level': level,
            'issues': issues,
            'confidence': confidence,
            'interventions': interventions_applied
        })
        self.last_intervention_epoch = epoch
        
        return True
    
    def _reset_liquid_neuron(self, right_node, severity):
        device = right_node.W_fast_short.device
        right_node.W_fast_short = 0.00005 * torch.randn_like(right_node.W_fast_short)
        right_node.W_fast_long = 0.00005 * torch.randn_like(right_node.W_fast_long)
        right_node.norm_ema = torch.tensor(0.3, device=device)
        right_node.homeostasis = torch.tensor(1.0, device=device)
        right_node.metabolism = torch.tensor(0.7, device=device)
        right_node.fatigue = torch.tensor(0.0, device=device)

# =============================================================================
# NEURONA LÍQUIDA ESTABLE
# =============================================================================
class StableLiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.slow_expansion = 256
        self.fast_long_dim = 256
        
        self.slow_total = out_dim + self.slow_expansion
        self.fast_short_dim = out_dim
        self.concat_dim = self.slow_total + self.fast_short_dim + self.fast_long_dim
        
        self.W_slow = nn.Linear(in_dim, self.slow_total, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=0.8)
        
        self.register_buffer('W_fast_short', 0.0001 * torch.randn(self.fast_short_dim, in_dim))
        self.register_buffer('W_fast_long', 0.00005 * torch.randn(self.fast_long_dim, in_dim))
        
        self.ln = nn.LayerNorm(self.concat_dim)
        self.project = nn.Linear(self.concat_dim, out_dim)
        
        # Fisiología
        self.register_buffer('metabolism', torch.tensor(0.6))
        self.register_buffer('fatigue', torch.tensor(0.0))
        self.register_buffer('sensitivity', torch.tensor(0.5))
        self.register_buffer('homeostasis', torch.tensor(1.0))
        
        self.base_lr = 0.001
        self.register_buffer('norm_ema', torch.tensor(0.5))
        self.register_buffer('norm_target', torch.tensor(1.0))
        
        self.fatigue_decay = 0.95
        self.metabolism_impact = 0.2
        self.cognitive_load_factor = 0.01
    
    def forward(self, x):
        slow_out = self.W_slow(x)
        fast_short = F.linear(x, self.W_fast_short)
        fast_long = F.linear(x, self.W_fast_long)
        
        gate_short = 0.05 + 0.15 * float(self.sensitivity) * float(self.homeostasis)
        gate_long = 0.02 + 0.08 * float(self.metabolism)
        
        combined = torch.cat([slow_out, gate_short * fast_short, gate_long * fast_long], dim=-1)
        out = self.project(self.ln(combined))
        
        return out, slow_out.detach(), x.detach()
    
    def hebbian_update(self, post, pre, plasticity=0.1):
        with torch.no_grad():
            hebb = torch.mm(post.T, pre) / max(1, pre.size(0))
            hebb = torch.clamp(hebb, -0.3, 0.3)
            
            current_norm = self.W_fast_short.norm()
            self.norm_ema = 0.95 * self.norm_ema + 0.05 * current_norm
            norm_ratio = self.norm_ema / self.norm_target
            
            if norm_ratio > 3.0:
                adaptive_lr = self.base_lr * 0.01
                self.homeostasis *= 0.8
            elif norm_ratio > 1.5:
                adaptive_lr = self.base_lr * 0.3
                self.homeostasis *= 0.95
            elif norm_ratio < 0.5:
                adaptive_lr = self.base_lr * 1.2
                self.homeostasis = torch.clamp(self.homeostasis * 1.02, 0.5, 1.0)
            else:
                adaptive_lr = self.base_lr
                self.homeostasis = torch.clamp(self.homeostasis * 1.01, 0.8, 1.0)
            
            update_short = adaptive_lr * plasticity * float(self.homeostasis) * torch.tanh(hebb)
            self.W_fast_short += update_short[:self.fast_short_dim]
            
            update_long = adaptive_lr * plasticity * 0.3 * float(self.metabolism) * torch.tanh(hebb)
            self.W_fast_long += update_long[:self.fast_long_dim]
            
            decay = 0.999 if norm_ratio < 1.0 else 0.99 if norm_ratio < 2.0 else 0.98
            self.W_fast_short *= decay
            self.W_fast_long *= decay * 0.995
            
            self.W_fast_short.clamp_(-0.5, 0.5)
            self.W_fast_long.clamp_(-0.3, 0.3)
            
            if current_norm > 5.0:
                self.W_fast_short *= (self.norm_target / current_norm)
    
    def update_physiology_advanced(self, loss_value):
        with torch.no_grad():
            loss_signal = max(0.0, min(1.0, 1.0 - loss_value / 4.0))
            homeostasis_signal = float(self.homeostasis)
            target_metab = 0.5 + 0.3 * loss_signal + 0.1 * homeostasis_signal
            
            metabolism_impact = self.metabolism_impact * (float(self.metabolism) - 0.6)
            target_metab += metabolism_impact
            
            self.metabolism = 0.9 * self.metabolism + 0.1 * target_metab
            self.metabolism = self.metabolism.clamp(0.3, 0.9)
            
            norm_ratio = self.norm_ema / self.norm_target
            
            cognitive_load = self.cognitive_load_factor * (1.0 - loss_signal)
            fatigue_increment = 0.002 if norm_ratio < 2.0 else 0.01
            fatigue_increment += cognitive_load
            
            self.fatigue *= self.fatigue_decay
            self.fatigue += fatigue_increment
            self.fatigue = self.fatigue.clamp(0, 0.5)
            
            if float(self.fatigue) > 0.3:
                self.metabolism *= 0.95
            
            if float(self.homeostasis) < 0.7:
                self.sensitivity *= 0.95
            else:
                target_sens = 0.5 + 0.2 * (1.0 - float(self.fatigue))
                self.sensitivity = 0.95 * self.sensitivity + 0.05 * target_sens
            self.sensitivity = self.sensitivity.clamp(0.3, 0.7)


# =============================================================================
# HEMISFERIO DERECHO (VISUAL)
# =============================================================================
class RightHemisphere(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in list(resnet.parameters())[:-20]:
            param.requires_grad = False
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.spatial_liquid = StableLiquidNeuron(2048, output_dim)
        
    def forward(self, image):
        features = self.visual_encoder(image)
        features = features.flatten(1)
        out, post, pre = self.spatial_liquid(features)
        return out, post, pre


# =============================================================================
# HEMISFERIO IZQUIERDO (COMPLETO: Razonamiento + MTP + Chain-of-Thought)
# =============================================================================
class LeftHemisphere(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.lstm_expansion = 128
        self.lstm_output_dim = hidden_dim + self.lstm_expansion
        self.vocab_size = vocab_size
        self.gate_warmup_epochs = 3
        
        # LSTM principal
        self.lstm = nn.LSTM(embed_dim + hidden_dim, self.lstm_output_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.bottleneck = nn.Linear(self.lstm_output_dim, hidden_dim)
        
        # Gate progresivo (LIQUID GATE)
        self.gate_dim_1 = hidden_dim // 2
        self.gate_dim_2 = hidden_dim // 4
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim, self.gate_dim_1),
            nn.LayerNorm(self.gate_dim_1),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(self.gate_dim_1, self.gate_dim_2),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(self.gate_dim_2, 1)
        )
        self.liquid_gate[-1].bias.data.fill_(-2.5)
        self.liquid_gate[-1].weight.data.mul_(0.01)
        
        # Atenciones
        self.visual_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.objects_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.actions_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.scene_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        
        self.channel_fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Proyección de salida
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # MÓDULOS DE RAZONAMIENTO
        self.mtp_depth = 2
        self.mtp_transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2, dropout=0.1, batch_first=True)
            for _ in range(self.mtp_depth)
        ])
        
        self.mtp_output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(self.mtp_depth)
        ])
        
        self.mtp_combination_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lambda_mtp = nn.Parameter(torch.tensor(0.3))
        
        # Chain-of-Thought
        self.max_reasoning_steps = 5
        self.reasoning_projection = nn.Linear(hidden_dim, hidden_dim)
        self.thought_gate = nn.Linear(hidden_dim, 1)
        self.step_embedding = nn.Embedding(self.max_reasoning_steps, hidden_dim)
        
        self.reasoning_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.reasoning_decision = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.reasoning_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, visual_context, captions=None, channels=None, max_len=30, epoch=0):
        batch_size = visual_context.size(0)
        device = visual_context.device
        
        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            seq_len = embeddings.size(1)
            
            visual_expanded = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_input = torch.cat([embeddings, visual_expanded], dim=2)
            
            lstm_out, _ = self.lstm(lstm_input, self._get_init_state(visual_context))
            lstm_out = self.bottleneck(lstm_out)
            
            reasoning_control = self.reasoning_controller(lstm_out.mean(dim=1))
            use_reasoning = reasoning_control.mean() > 0.5 and epoch > 2
            
            reasoned_out, reasoning_steps = self._apply_chain_of_thought(lstm_out, visual_context, use_reasoning)
            
            if channels is not None:
                reasoned_out = self._apply_structural_attention(reasoned_out, channels, visual_context)
            else:
                visual_query = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
                attended, _ = self.visual_attention(reasoned_out, visual_query, visual_query)
                reasoned_out = reasoned_out + 0.5 * attended
            
            warmup_factor = min(1.0, epoch / self.gate_warmup_epochs)
            adjusted_bias = self.liquid_gate[-1].bias.data * (1 - warmup_factor) + torch.tensor(-2.5) * warmup_factor
            
            gate_logits = self.liquid_gate(reasoned_out)
            gate = torch.sigmoid(gate_logits + adjusted_bias)
            out = reasoned_out * (0.5 + 0.5 * gate)
            
            logits, mtp_loss = self._apply_multi_token_prediction(out, captions)
            
            return logits, gate, mtp_loss, reasoning_steps
        else:
            return self._greedy_decode(visual_context, channels, max_len, epoch)
    
    def _greedy_decode(self, visual_context, channels, max_len, epoch):
        batch_size = visual_context.size(0)
        device = visual_context.device
        generated = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            emb = self.embedding(generated)
            seq_len = emb.size(1)
            visual_expanded = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_input = torch.cat([emb, visual_expanded], dim=2)
            
            out, _ = self.lstm(lstm_input, self._get_init_state(visual_context))
            out = self.bottleneck(out)
            
            if channels is not None:
                out = self._apply_structural_attention(out, channels, visual_context)
            else:
                visual_query = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
                attended, _ = self.visual_attention(out, visual_query, visual_query)
                out = out + 0.5 * attended
            
            gate = torch.sigmoid(self.liquid_gate(out))
            out = out * (0.5 + 0.5 * gate)
            
            logits = self.output_projection(out[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == 2).all():
                break
        
        return generated
    
    def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning=True):
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device
        
        if not use_reasoning:
            return hidden_states, torch.tensor(0.0, device=device, dtype=torch.float32)
        
        reasoning_state = hidden_states.clone()
        reasoning_steps = 0
        
        visual_expanded = visual_context.unsqueeze(1).expand(-1, self.max_reasoning_steps, -1)
        
        for step in range(self.max_reasoning_steps):
            step_emb = self.step_embedding(torch.tensor(step, device=device))
            step_emb = step_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            
            combined = reasoning_state + step_emb
            projected = self.reasoning_projection(combined)
            gate = torch.sigmoid(self.thought_gate(projected))
            gated = projected * gate
            
            attended, _ = self.reasoning_attention(gated, visual_expanded[:, :step+1], visual_expanded[:, :step+1])
            reasoning_state = reasoning_state + 0.1 * attended
            
            decision = self.reasoning_decision(reasoning_state.mean(dim=1))
            reasoning_steps = step + 1
            
            if decision.mean() < 0.3 and step > 1:
                break
        
        return reasoning_state, torch.tensor(float(reasoning_steps), device=device, dtype=torch.float32)
    
    def _apply_multi_token_prediction(self, hidden_states, input_ids):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        vocab_size = self.vocab_size
        padding_idx = 0
        
        logits = self.output_projection(hidden_states)
        mtp_loss = torch.tensor(0.0, device=hidden_states.device)
        
        prediction_depths = [1, 2, 3]
        valid_predictions = 0
        
        for depth in prediction_depths:
            if seq_len <= depth:
                continue
            
            valid_seq_len = seq_len - depth
            if valid_seq_len <= 0:
                continue
            
            predict_hidden = hidden_states[:, :valid_seq_len, :]
            target_ids = input_ids[:, depth:depth+valid_seq_len]
            
            if target_ids.size(1) == 0:
                continue
            
            depth_logits = self.mtp_output_heads[depth-1](predict_hidden)
            
            depth_loss = F.cross_entropy(
                depth_logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
                ignore_index=padding_idx,
                reduction='mean'
            )
            
            mtp_loss += depth_loss
            valid_predictions += 1
        
        if valid_predictions > 0:
            mtp_loss = mtp_loss / valid_predictions
        
        return logits, mtp_loss
    
    def _apply_structural_attention(self, lstm_out, channels, visual_context):
        seq_len = lstm_out.size(1)
        
        objects_expanded = channels['objects'].unsqueeze(1).expand(-1, seq_len, -1)
        actions_expanded = channels['actions'].unsqueeze(1).expand(-1, seq_len, -1)
        scene_expanded = channels['scene'].unsqueeze(1).expand(-1, seq_len, -1)
        
        objects_attended, _ = self.objects_attention(lstm_out, objects_expanded, objects_expanded)
        actions_attended, _ = self.actions_attention(lstm_out, actions_expanded, actions_expanded)
        scene_attended, _ = self.scene_attention(lstm_out, scene_expanded, scene_expanded)
        
        fused = torch.cat([objects_attended, actions_attended, scene_attended], dim=-1)
        fused = self.channel_fusion(fused)
        
        visual_query = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
        visual_attended, _ = self.visual_attention(lstm_out, visual_query, visual_query)
        
        combined = lstm_out + 0.3 * visual_attended + 0.7 * fused
        
        return combined
    
    def _get_init_state(self, visual_context):
        batch_size = visual_context.size(0)
        h0 = visual_context.unsqueeze(0).repeat(2, 1, 1)
        
        padding = torch.zeros(2, batch_size, self.lstm_expansion, device=visual_context.device)
        h0 = torch.cat([h0, padding], dim=-1)
        c0 = torch.zeros_like(h0)
        
        return (h0, c0)
# =============================================================================
# CORPUS CALLOSUM
# =============================================================================
class CorpusCallosum(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        
        self.transfer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        self.residual_scale = nn.Parameter(torch.tensor(0.85))
        
        # Canales estructurales
        self.objects_dim = dim // 3
        self.actions_dim = dim // 3
        self.scene_dim = dim - 2 * (dim // 3)
        
        self.objects_proj = nn.Linear(dim, self.objects_dim)
        self.actions_proj = nn.Linear(dim, self.actions_dim)
        self.scene_proj = nn.Linear(dim, self.scene_dim)
        
        # Gates por canal
        self.objects_gate = nn.Parameter(torch.tensor(0.5))
        self.actions_gate = nn.Parameter(torch.tensor(0.5))
        self.scene_gate = nn.Parameter(torch.tensor(0.5))
        
        # Fatiga por canal
        self.register_buffer('objects_fatigue', torch.tensor(0.0))
        self.register_buffer('actions_fatigue', torch.tensor(0.0))
        self.register_buffer('scene_fatigue', torch.tensor(0.0))
        self.fatigue_decay = 0.95
        self.fatigue_recovery = 0.01
        
        self.objects_fatigue.fill_(0.0)
        self.actions_fatigue.fill_(0.0)
        self.scene_fatigue.fill_(0.0)
        
        self.flow_modulator = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.residual_scale_base = nn.Parameter(torch.tensor(0.85))
        
        self.flow_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=0.1, batch_first=True)
    
    def forward(self, right_features, left_features=None):
        x = right_features.unsqueeze(1)
        
        objects_channel = self.objects_proj(right_features)
        actions_channel = self.actions_proj(right_features)
        scene_channel = self.scene_proj(right_features)
        
        objects_gated = objects_channel * torch.sigmoid(self.objects_gate)
        actions_gated = actions_channel * torch.sigmoid(self.actions_gate)
        scene_gated = scene_channel * torch.sigmoid(self.scene_gate)
        
        structured = torch.cat([objects_gated, actions_gated, scene_gated], dim=-1)
        
        structured_expanded = structured.unsqueeze(1)
        attn_out, _ = self.flow_attention(structured_expanded, structured_expanded, structured_expanded)
        attn_out = attn_out.squeeze(1)
        
        for block in self.transfer:
            attn_out = attn_out + block(attn_out)
        
        flow_strength = self.flow_modulator(right_features)
        dynamic_scale = self.residual_scale_base * (0.7 + 0.6 * flow_strength.squeeze(-1))
        
        self.update_channel_fatigue(objects_channel, actions_channel, scene_channel)
        
        output = attn_out + dynamic_scale.unsqueeze(-1) * right_features
        
        # Proyección completa para compatibilidad
        objects_full = torch.cat([objects_gated, torch.zeros(objects_gated.size(0), self.actions_dim + self.scene_dim, device=objects_gated.device)], dim=-1)
        actions_full = torch.cat([torch.zeros(actions_gated.size(0), self.objects_dim, device=actions_gated.device), actions_gated, torch.zeros(actions_gated.size(0), self.scene_dim, device=actions_gated.device)], dim=-1)
        scene_full = torch.cat([torch.zeros(scene_gated.size(0), self.objects_dim + self.actions_dim, device=scene_gated.device), scene_gated], dim=-1)
        
        return output, {
            'objects': objects_full,
            'actions': actions_full,
            'scene': scene_full,
            'fatigue': {
                'objects': self.objects_fatigue.item(),
                'actions': self.actions_fatigue.item(),
                'scene': self.scene_fatigue.item()
            }
        }
    
    def update_channel_fatigue(self, objects_channel, actions_channel, scene_channel):
        """Actualiza fatiga específica por canal"""
        with torch.no_grad():
            objects_activity = objects_channel.norm(dim=-1).mean()
            actions_activity = actions_channel.norm(dim=-1).mean()
            scene_activity = scene_channel.norm(dim=-1).mean()
            
            self.objects_fatigue = self.objects_fatigue * self.fatigue_decay + 0.01 * objects_activity
            self.actions_fatigue = self.actions_fatigue * self.fatigue_decay + 0.01 * actions_activity
            self.scene_fatigue = self.scene_fatigue * self.fatigue_decay + 0.01 * scene_activity
            
            self.objects_fatigue = max(0.0, self.objects_fatigue - self.fatigue_recovery)
            self.actions_fatigue = max(0.0, self.actions_fatigue - self.fatigue_recovery)
            self.scene_fatigue = max(0.0, self.scene_fatigue - self.fatigue_recovery)
            
            self.objects_fatigue = torch.clamp(self.objects_fatigue, 0.0, 1.0)
            self.actions_fatigue = torch.clamp(self.actions_fatigue, 0.0, 1.0)
            self.scene_fatigue = torch.clamp(self.scene_fatigue, 0.0, 1.0)
    
    def adjust_gates_by_fatigue(self):
        """Ajusta gates basado en fatiga"""
        with torch.no_grad():
            self.objects_gate.data *= (1.0 - 0.1 * self.objects_fatigue)
            self.actions_gate.data *= (1.0 - 0.1 * self.actions_fatigue)
            self.scene_gate.data *= (1.0 - 0.1 * self.scene_fatigue)
            
            self.objects_gate.data = torch.clamp(self.objects_gate.data, -2.0, 2.0)
            self.actions_gate.data = torch.clamp(self.actions_gate.data, -2.0, 2.0)
            self.scene_gate.data = torch.clamp(self.scene_gate.data, -2.0, 2.0)


# =============================================================================
# MODELO BICAMERAL COMPLETO
# =============================================================================
class NeuroLogosBicameralStable(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.right_hemisphere = RightHemisphere(output_dim=512)
        self.left_hemisphere = LeftHemisphere(vocab_size, embed_dim=256, hidden_dim=512)
        self.corpus_callosum = CorpusCallosum(dim=512)
        
    def forward(self, image, captions=None, epoch=0):
        visual_features, right_post, right_pre = self.right_hemisphere(image)
        visual_context, channels = self.corpus_callosum(visual_features)
        
        if captions is not None:
            logits, gate, mtp_loss, reasoning_steps = self.left_hemisphere(
                visual_context, captions, channels, epoch=epoch
            )
            return logits, visual_features, visual_context, gate, right_post, right_pre, channels, mtp_loss, reasoning_steps
        else:
            output = self.left_hemisphere(visual_context, captions=None, channels=channels, epoch=epoch)
            return output


# =============================================================================
# DIAGNÓSTICO MEJORADO
# =============================================================================
class EnhancedDiagnostics:
    def __init__(self):
        self.history = {
            'loss': [], 'right_metabolism': [], 'right_fatigue': [],
            'right_liquid_norm': [], 'right_homeostasis': [],
            'callosal_flow': [], 'left_gate_mean': [], 'left_gate_std': [],
            'synergy_score': [], 'health_score': [],
            'bleu_score': [], 'token_accuracy': [], 'word_overlap': [],
            'cider_score': [], 'spice_score': [], 'linguistic_reward': [],
            'alignment_loss': [],
            'objects_fatigue': [], 'actions_fatigue': [], 'scene_fatigue': [],
            'reasoning_steps': [], 'mtp_loss': [], 'logical_coherence': [],
            'reasoning_efficiency': [], 'mtp_accuracy': []
        }
        self.language_metrics = LanguageMetrics()
    
    def measure_callosal_flow(self, right_features, left_context, channels=None):
        with torch.no_grad():
            right_norm = F.normalize(right_features, dim=-1)
            left_norm = F.normalize(left_context, dim=-1)
            correlation = (right_norm * left_norm).sum(dim=-1).mean()
            flow_std = left_context.std(dim=-1).mean()
            
            if channels is not None:
                objects_norm = F.normalize(channels['objects'], dim=-1)
                actions_norm = F.normalize(channels['actions'], dim=-1)
                scene_norm = F.normalize(channels['scene'], dim=-1)
                
                objects_corr = (right_norm * objects_norm).sum(dim=-1).mean()
                actions_corr = (right_norm * actions_norm).sum(dim=-1).mean()
                scene_corr = (right_norm * scene_norm).sum(dim=-1).mean()
                
                weighted_correlation = 0.5 * objects_corr + 0.3 * actions_corr + 0.2 * scene_corr
                flow = weighted_correlation * min(1.0, flow_std.item() / 0.5)
            else:
                flow = correlation.item() * min(1.0, flow_std.item() / 0.5)
            
            return float(flow)
    
    def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps):
        """Evalúa coherencia y consistencia del razonamiento"""
        coherence_scores = []
        consistency_scores = []
        
        for gen, ref, steps in zip(generated_texts, reference_texts, reasoning_steps):
            gen_sentences = [s.strip() for s in gen.split('.') if s.strip()]
            
            if len(gen_sentences) > 1:
                sentence_similarities = []
                for i in range(len(gen_sentences) - 1):
                    words1 = set(gen_sentences[i].lower().split())
                    words2 = set(gen_sentences[i+1].lower().split())
                    
                    if len(words1) > 0 and len(words2) > 0:
                        jaccard = len(words1 & words2) / len(words1 | words2)
                        sentence_similarities.append(jaccard)
                
                coherence = np.mean(sentence_similarities) if sentence_similarities else 0.0
            else:
                coherence = 0.5
            
            gen_words = set(gen.lower().split())
            ref_words = set(ref.lower().split())
            consistency = len(gen_words & ref_words) / len(gen_words | ref_words) if len(gen_words | ref_words) > 0 else 0.0
            
            reasoning_bonus = min(1.0, steps / 3.0) if steps is not None else 0.0
            coherence_scores.append(coherence * (0.7 + 0.3 * reasoning_bonus))
            consistency_scores.append(consistency)
        
        return np.mean(coherence_scores), np.mean(consistency_scores)
    
    def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std):
        right_health = float(right_node.metabolism) * float(right_node.homeostasis) * (1.0 - float(right_node.fatigue) * 0.5)
        callosal_health = float(callosal_flow)
        gate_balance = 1.0 - abs(float(left_gate_mean) - 0.5) * 2.0
        gate_diversity = min(1.0, float(left_gate_std) * 5.0)
        left_health = 0.7 * gate_balance + 0.3 * gate_diversity
        synergy = (0.35 * right_health + 0.30 * callosal_health + 0.35 * left_health)
        return float(synergy)
    
    def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm):
        health = 0
        if float(liquid_norm) < 2.0: health += 1
        if float(right_node.homeostasis) > 0.7: health += 1
        if float(callosal_flow) > 0.4: health += 1
        if 0.4 < float(left_gate_mean) < 0.6 and float(left_gate_std) > 0.05: health += 1
        if float(right_node.fatigue) < 0.3 and float(right_node.metabolism) > 0.55: health += 1
        return int(health)
    
    def update(self, **metrics):
        for key, value in metrics.items():
            if key in self.history and value is not None:
                if isinstance(value, torch.Tensor):
                    if value.is_cuda:
                        value = value.cpu().item() if value.numel() == 1 else value.cpu().numpy().tolist()
                    else:
                        value = value.item() if value.numel() == 1 else value.numpy().tolist()
                self.history[key].append(value)
    
    def get_recent_avg(self, key, n=50):
        if key in self.history and len(self.history[key]) > 0:
            recent_values = self.history[key][-n:]
            clean_values = []
            for v in recent_values:
                if isinstance(v, torch.Tensor):
                    if v.is_cuda:
                        clean_values.append(float(v.cpu().item() if v.numel() == 1 else v.cpu().numpy().mean()))
                    else:
                        clean_values.append(float(v.item() if v.numel() == 1 else v.numpy().mean()))
                else:
                    clean_values.append(float(v))
            return np.mean(clean_values)
        elif key in ['objects_fatigue', 'actions_fatigue', 'scene_fatigue']:
            return 0.0
        return 0.0
    
    def visualize_fatigue_distribution(self, epoch):
        if epoch % 5 == 0:
            objects_fatigue = self.get_recent_avg('objects_fatigue', n=10)
            actions_fatigue = self.get_recent_avg('actions_fatigue', n=10)
            scene_fatigue = self.get_recent_avg('scene_fatigue', n=10)
            
            print(f"\n🔗 DISTRIBUCIÓN DE FATIGA - Época {epoch}")
            print(f"  Objetos: {objects_fatigue:.3f} {'🔴' if objects_fatigue > 0.3 else '🟡' if objects_fatigue > 0.15 else '🟢'}")
            print(f"  Acciones: {actions_fatigue:.3f} {'🔴' if actions_fatigue > 0.3 else '🟡' if actions_fatigue > 0.15 else '🟢'}")
            print(f"  Escena: {scene_fatigue:.3f} {'🔴' if scene_fatigue > 0.3 else '🟡' if scene_fatigue > 0.15 else '🟢'}")
            
            max_fatigue = max(objects_fatigue, actions_fatigue, scene_fatigue)
            min_fatigue = min(objects_fatigue, actions_fatigue, scene_fatigue)
            imbalance = max_fatigue - min_fatigue
            
            if imbalance > 0.2:
                print(f"  ⚠️  Desbalance: {imbalance:.3f}")
            else:
                print(f"  ✅ Balanceado: {imbalance:.3f}")
            print(f"{'='*60}")
    
    def visualize_reasoning_metrics(self, epoch):
        if epoch % 3 == 0:
            reasoning_steps = self.get_recent_avg('reasoning_steps', n=10)
            mtp_loss = self.get_recent_avg('mtp_loss', n=10)
            logical_coherence = self.get_recent_avg('logical_coherence', n=10)
            
            print(f"\n🧠 MÉTRICAS DE RAZONAMIENTO - Época {epoch}")
            print(f"  Pasos: {reasoning_steps:.2f} {'🟢' if 2 <= reasoning_steps <= 3 else '🟡'}")
            print(f"  MTP Loss: {mtp_loss:.3f} {'🟢' if mtp_loss < 0.2 else '🟡' if mtp_loss < 0.4 else '🔴'}")
            print(f"  Coherencia: {logical_coherence:.3f} {'🟢' if logical_coherence > 0.4 else '🟡' if logical_coherence > 0.2 else '🔴'}")
            print(f"{'='*60}")
    
    def report(self, epoch):
        """Genera reporte completo del estado del sistema bicameral"""
        # Verificar que hay datos para reportar
        if len(self.history['loss']) == 0:
            return
        
        self.visualize_fatigue_distribution(epoch)
        self.visualize_reasoning_metrics(epoch)
        
        print(f"\n{'='*80}")
        print(f"📊 REPORTE COMPLETO - Época {epoch}")
        print(f"{'='*80}")
        
        # Métricas de entrenamiento
        loss = self.get_recent_avg('loss')
        print(f"\n📉 ENTRENAMIENTO:")
        print(f"  Loss: {loss:.4f}")
        if 'alignment_loss' in self.history and len(self.history['alignment_loss']) > 0:
            align_loss = self.get_recent_avg('alignment_loss')
            print(f"  Align Loss: {align_loss:.4f}", end=" ")
            print("🟢" if align_loss < 0.3 else "🟡" if align_loss < 0.5 else "🔴")
        
        # Métricas lingüísticas
        bleu = self.get_recent_avg('bleu_score')
        acc = self.get_recent_avg('token_accuracy')
        overlap = self.get_recent_avg('word_overlap')
        cider = self.get_recent_avg('cider_score')
        spice = self.get_recent_avg('spice_score')
        reward = self.get_recent_avg('linguistic_reward')
        
        print(f"\n📝 CALIDAD LINGÜÍSTICA:")
        print(f"  BLEU-4: {bleu:.4f} {'🟢' if bleu > 0.15 else '🟡' if bleu > 0.08 else '🔴'}")
        print(f"  Accuracy: {acc:.4f} {'🟢' if acc > 0.30 else '🟡' if acc > 0.15 else '🔴'}")
        print(f"  W-Overlap: {overlap:.4f} {'🟢' if overlap > 0.35 else '🟡' if overlap > 0.20 else '🔴'}")
        print(f"  CIDEr: {cider:.4f} {'🟢' if cider > 0.15 else '🟡' if cider > 0.08 else '🔴'}")
        print(f"  SPICE: {spice:.4f} {'🟢' if spice > 0.20 else '🟡' if spice > 0.10 else '🔴'}")
        print(f"  Reward: {reward:.4f} {'🟢' if reward > 0.30 else '🟡' if reward > 0.15 else '🔴'}")
        
        # Métricas de razonamiento
        reasoning_steps = self.get_recent_avg('reasoning_steps')
        mtp_loss = self.get_recent_avg('mtp_loss')
        logical_coherence = self.get_recent_avg('logical_coherence')
        
        print(f"\n🧠 RAZONAMIENTO:")
        print(f"  Pasos: {reasoning_steps:.2f} {'🟢' if 2 <= reasoning_steps <= 3 else '🟡'}")
        print(f"  MTP Loss: {mtp_loss:.3f} {'🟢' if mtp_loss < 0.2 else '🟡' if mtp_loss < 0.4 else '🔴'}")
        print(f"  Coherencia: {logical_coherence:.3f} {'🟢' if logical_coherence > 0.4 else '🟡' if logical_coherence > 0.2 else '🔴'}")
        
        # Fisiología
        print(f"\n🧬 FISIOLOGÍA:")
        metab = self.get_recent_avg('right_metabolism')
        fatigue = self.get_recent_avg('right_fatigue')
        liquid = self.get_recent_avg('right_liquid_norm')
        homeo = self.get_recent_avg('right_homeostasis')
        
        print(f"  Liquid Norm: {liquid:.3f} {'🟢' if liquid < 2.0 else '🟡' if liquid < 4.0 else '🔴'}")
        print(f"  Homeostasis: {homeo:.3f} {'🟢' if homeo > 0.8 else '🟡' if homeo > 0.6 else '🔴'}")
        print(f"  Metabolism: {metab:.3f}")
        print(f"  Fatigue: {fatigue:.3f}")
        
        # Comunicación
        flow = self.get_recent_avg('callosal_flow')
        gate_mean = self.get_recent_avg('left_gate_mean')
        gate_std = self.get_recent_avg('left_gate_std')
        
        print(f"\n🔗 COMUNICACIÓN:")
        print(f"  Callosum: {flow:.3f} {'🟢' if flow > 0.5 else '🟡' if flow > 0.3 else '🔴'}")
        print(f"  Gate Mean: {gate_mean:.3f} {'🟢' if 0.4 < gate_mean < 0.6 else '🟡'}")
        print(f"  Gate Std: {gate_std:.3f}")
        
        # Salud global
        synergy = self.get_recent_avg('synergy_score')
        health = self.get_recent_avg('health_score')
        
        print(f"\n🏛️  SISTEMA BICAMERAL:")
        print(f"  Sinergia: {synergy:.3f} | Salud: {int(health)}/5", end=" ")
        if health >= 4:
            print("🟢 ÓPTIMO")
        elif health >= 3:
            print("🟡 FUNCIONAL")
        else:
            print("🔴 CRÍTICO")
        
        print(f"{'='*80}\n")

# =============================================================================
# DATASET
# =============================================================================
class Flickr8kDataset(Dataset):
    def __init__(self, images_dir, captions_file, vocab, transform=None, max_len=30):
        self.images_dir = images_dir
        self.transform = transform
        self.vocab = vocab
        self.max_len = max_len
        self.data = []
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, caption = parts
                    img_path = os.path.join(images_dir, img_name)
                    if os.path.exists(img_path):
                        self.data.append((img_path, caption))
        print(f"Loaded {len(self.data)} image-caption pairs")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, caption = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        tokens = ['<BOS>'] + caption.lower().split() + ['<EOS>']
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        
        return image, torch.tensor(token_ids, dtype=torch.long), caption

def build_vocab_flickr(captions_file, vocab_size=5000):
    print("Building vocabulary...")
    counter = Counter()
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                counter.update(parts[1].lower().split())
    
    most_common = counter.most_common(vocab_size - 4)
    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for i, (word, _) in enumerate(most_common):
        vocab[word] = i + 4
    
    id2word = {i: w for w, i in vocab.items()}
    print(f"Vocabulary size: {len(vocab)}")
    return vocab, id2word

def setup_flickr8k(data_dir='./data'):
    flickr_dir = os.path.join(data_dir, 'flickr8k')
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    if os.path.exists(images_dir) and os.path.exists(captions_file):
        print("✓ Flickr8k already exists\n")
        return flickr_dir
    
    os.makedirs(flickr_dir, exist_ok=True)
    
    print("📥 Downloading Flickr8k...")
    import urllib.request
    import zipfile
    
    urls = {
        'images': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip ',
        'captions': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip '
    }
    
    for name, url in urls.items():
        zip_path = os.path.join(flickr_dir, f'{name}.zip')
        print(f"📥 Downloading {name}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"📂 Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(flickr_dir)
        os.remove(zip_path)
        print(f"✓ {name} completed\n")
    
    # Process captions
    print("📝 Processing captions...")
    raw_captions = os.path.join(flickr_dir, 'Flickr8k.token.txt')
    
    if os.path.exists(raw_captions):
        captions_dict = {}
        with open(raw_captions, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name = parts[0].split('#')[0]
                    caption = parts[1]
                    if img_name not in captions_dict:
                        captions_dict[img_name] = []
                    captions_dict[img_name].append(caption)
        
        with open(captions_file, 'w', encoding='utf-8') as f:
            for img_name, caps in captions_dict.items():
                for cap in caps:
                    f.write(f"{img_name}\t{cap}\n")
        
        print(f"✓ Captions processed: {len(captions_dict)} images\n")
    
    # Verify structure
    if os.path.exists(os.path.join(flickr_dir, 'Flicker8k_Dataset')):
        import shutil
        old_dir = os.path.join(flickr_dir, 'Flicker8k_Dataset')
        if not os.path.exists(images_dir):
            shutil.move(old_dir, images_dir)
    
    print("✅ Flickr8k ready\n")
    return flickr_dir


def compute_alignment_loss(visual_features, channels, alpha=0.1):
    """Pérdida auxiliar para alineación temprana"""
    with torch.enable_grad():
        visual_norm = F.normalize(visual_features, dim=-1)
        objects_norm = F.normalize(channels['objects'], dim=-1)
        actions_norm = F.normalize(channels['actions'], dim=-1)
        scene_norm = F.normalize(channels['scene'], dim=-1)
        
        objects_align = (1 - (objects_norm * visual_norm).sum(dim=-1)).mean()
        actions_align = (1 - (actions_norm * visual_norm).sum(dim=-1)).mean()
        scene_align = (1 - (scene_norm * visual_norm).sum(dim=-1)).mean()
        
        total_align = 0.5 * objects_align + 0.3 * actions_align + 0.2 * scene_align
        
        return alpha * total_align


# =============================================================================
# ENTRENAMIENTO COMPLETO
# =============================================================================
def train_with_metrics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"NeuroLogos v4.1 | Memoria Episódica + Razonamiento + Métricas | Device: {device}")
    print(f"{'='*80}\n")
    
    flickr_dir = setup_flickr8k()
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    vocab, id2word = build_vocab_flickr(captions_file, VOCAB_SIZE)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Flickr8kDataset(images_dir, captions_file, vocab, transform, MAX_CAPTION_LEN)
    
    # Configuración de DataLoader
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
        print("Using Colab config (no multiprocessing)")
    else:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        print("Using standard config with multiprocessing")
    
    model = NeuroLogosBicameralStable(len(vocab)).to(device)
    
    # Inicialización de fatiga
    with torch.no_grad():
        dummy_feat = torch.randn(1, 512).to(device)
        _, dummy_channels = model.corpus_callosum(dummy_feat)
        model.corpus_callosum.objects_fatigue.fill_(0.0)
        model.corpus_callosum.actions_fatigue.fill_(0.0)
        model.corpus_callosum.scene_fatigue.fill_(0.0)
    
    optimizer = torch.optim.AdamW([
        {'params': model.right_hemisphere.parameters(), 'lr': 3e-4, 'weight_decay': 1e-5},
        {'params': model.corpus_callosum.parameters(), 'lr': 5e-4, 'weight_decay': 1e-5},
        {'params': model.left_hemisphere.parameters(), 'lr': 2e-4, 'weight_decay': 1e-5}
    ])
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=3)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=27, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[3])
    
    diagnostics = EnhancedDiagnostics()
    medical_system = TriangulatedMedicalSystem()
    cognitive_system = NeurocognitiveSystem()
    episodic_memory = EpisodicMemoryBuffer(capacity=500)
    
    print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Metrics: BLEU-4 + Token Acc + Word Overlap + CIDEr + SPICE")
    print(f"💊 Medical System: 3 intervention levels")
    print(f"🧠 Cognitive System: Reasoning-aware optimization")
    print(f"🧠 Episodic Memory: Surprise-weighted replay")
    print(f"🔗 Structural Callosum: Channel preservation")
    print()
    
    os.makedirs('./checkpoints', exist_ok=True)
    
    reward_frequency = 200
    
    for epoch in range(30):
        if epoch < 3:
            plasticity = 0.05
        elif epoch < 10:
            plasticity = 0.10
        elif epoch < 20:
            plasticity = max(0.02, 0.10 * (1 - (epoch-10)/20))
        else:
            plasticity = 0.01
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        right_node = model.right_hemisphere.spatial_liquid
        liquid = diagnostics.get_recent_avg('right_liquid_norm')
        flow = diagnostics.get_recent_avg('callosal_flow')
        gate_mean = diagnostics.get_recent_avg('left_gate_mean')
        gate_std = diagnostics.get_recent_avg('left_gate_std')
        health_score = diagnostics.calculate_health(right_node, flow, gate_mean, gate_std, liquid)
        
        # Diagnóstico médico
        issues, severity, confidence = medical_system.diagnose_with_triangulation(
            health_score, liquid, gate_mean, gate_std, flow
        )
        
        medicine_level = "🟢 Nivel 0" if severity == 0 else f"🟡 Nivel 1" if severity <= 2 else f"🟠 Nivel 2" if severity <= 6 else "🔴 Nivel 3"
        
        if severity > 0:
            medical_system.apply_triangulated_intervention(model, issues, severity, confidence, epoch)
        
        # Diagnóstico cognitivo
        cognitive_level = "🟢 Nivel Cognitivo 0"
        if epoch >= 2:
            cider_score = diagnostics.get_recent_avg('cider_score')
            spice_score = diagnostics.get_recent_avg('spice_score')
            reward = diagnostics.get_recent_avg('linguistic_reward')
            
            cog_issues, cog_severity, cog_confidence = cognitive_system.assess_cognitive_state(
                cider_score, spice_score, reward, epoch
            )
            
            cognitive_level = "🟢 Nivel Cognitivo 0" if cog_severity == 0 else f"🟡 Nivel Cognitivo 1" if cog_severity <= 2 else f"🟠 Nivel Cognitivo 2" if cog_severity <= 5 else "🔴 Nivel Cognitivo 3"
            
            if cog_severity > 0:
                cognitive_system.apply_cognitive_intervention(model, cog_issues, cog_severity, cog_confidence, epoch, diagnostics)
        
        # Ajuste de fatiga por canal
        model.corpus_callosum.adjust_gates_by_fatigue()
        
        # Barra de progreso
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Health: {health_score}/5 | Med: {medicine_level} | Cog: {cognitive_level}]")
        
        for batch_idx, (images, captions, raw_captions) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward principal con razonamiento
            logits, visual_features, visual_context, gate, right_post, right_pre, channels, mtp_loss, reasoning_steps = model(
                images, captions, epoch=epoch
            )

            # Calcular recompensa lingüística
            linguistic_reward = None
            if batch_idx % reward_frequency == 0 and epoch >= 2:
                with torch.no_grad():
                    sample_indices = np.random.choice(images.size(0), size=min(2, images.size(0)), replace=False)
                    references = [raw_captions[i] for i in sample_indices]
                    
                    sample_images = images[sample_indices]
                    generated = model(sample_images, captions=None, epoch=epoch)
                    
                    hypotheses = []
                    for i in range(generated.size(0)):
                        gen_words = [id2word.get(int(t.item()), '<UNK>') for t in generated[i]]
                        gen_sentence = " ".join(w for w in gen_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                        hypotheses.append(gen_sentence)
                    
                    linguistic_reward = cognitive_system.linguistic_feedback.compute_linguistic_reward(references, hypotheses)
                    
                    # FIX: Evaluar razonamiento solo si reasoning_steps tiene batch dimension
                    if reasoning_steps is not None and reasoning_steps.numel() > 1:
                        sample_reasoning_steps = reasoning_steps[sample_indices]
                        coherence, consistency = diagnostics.evaluate_reasoning_quality(
                            hypotheses, references, sample_reasoning_steps.tolist()
                        )
                        diagnostics.update(logical_coherence=coherence, reasoning_efficiency=consistency)
                    elif reasoning_steps is not None:
                        # Si es escalar, usar el mismo valor para todos
                        single_step = reasoning_steps.item()
                        coherence, consistency = diagnostics.evaluate_reasoning_quality(
                            hypotheses, references, [single_step] * len(hypotheses)
                        )
                        diagnostics.update(logical_coherence=coherence, reasoning_efficiency=consistency)
                        # Cálculo de pérdida
            loss, ce_loss, gate_penalty, diversity_penalty, linguistic_loss, mtp_term = compute_loss(
                logits, captions, gate, vocab, mtp_loss, linguistic_reward
            )
            
            total_loss_value = loss + 0.1 * mtp_loss if mtp_loss is not None else loss
            
            # Pérdida de alineación temprana
            if epoch < 6:
                alignment_loss = compute_alignment_loss(visual_features, channels, alpha=0.15)
                total_loss = total_loss_value + alignment_loss
            else:
                total_loss = total_loss_value
                alignment_loss = torch.tensor(0.0, device=images.device)
            
            # Backward principal
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # ACTUALIZACIÓN EPISÓDICA - Cada 20 batches
            if batch_idx % 20 == 0:
                surprise = episodic_memory.compute_surprise(logits, captions[:, 1:], gate.mean())
                
                for i in range(images.size(0)):
                    episodic_memory.add(images[i].cpu(), captions[i].cpu(), surprise)
                
                # Logging de memoria
                if batch_idx % 100 == 0:
                    print(f"  🧠 Episodic Buffer: {len(episodic_memory.buffer)}/{episodic_memory.capacity} samples")
            
            # REPLAY EPISÓDICO - Cada 5 batches
            if batch_idx % 5 == 0 and len(episodic_memory.buffer) > 32:
                replay_samples = episodic_memory.sample(16)
                if replay_samples:
                    replay_imgs = torch.stack([s[0] for s in replay_samples]).to(device, non_blocking=True)
                    replay_caps = torch.stack([s[1] for s in replay_samples]).to(device, non_blocking=True)
                    
                    # Forward de replay (IGNORAMOS MTP en replay para eficiencia)
                    logits_replay, _, _, gate_replay, post_replay, pre_replay, _, _, _ = model(replay_imgs, replay_caps, epoch=epoch)
                    
                    # Pérdida de replay (sin MTP)
                    loss_replay, ce_replay, _, _, _, _ = compute_loss(logits_replay, replay_caps, gate_replay, vocab, mtp_loss=None)
                    
                    # Backward de replay con factor reducido
                    (0.5 * loss_replay).backward()
                    
                    # Hebbian update aumentado en replay
                    model.right_hemisphere.spatial_liquid.hebbian_update(post_replay, pre_replay, plasticity * 2.0)
            
            # Hebbian update principal
            model.right_hemisphere.spatial_liquid.hebbian_update(right_post, right_pre, plasticity)
            model.right_hemisphere.spatial_liquid.update_physiology_advanced(ce_loss.item())
            
            # Actualización de diagnósticos
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    liquid_norm = model.right_hemisphere.spatial_liquid.W_fast_short.norm().item()
                    callosal_flow = diagnostics.measure_callosal_flow(visual_features, visual_context, channels)
                    gate_mean_val = gate.mean().item() if gate.numel() > 1 else gate.item()
                    gate_std_val = gate.std().item() if gate.numel() > 1 else 0.0
                    
                    channel_fatigue = channels['fatigue']
                    
                    synergy = diagnostics.calculate_synergy(right_node, callosal_flow, gate_mean_val, gate_std_val)
                    
                    # Manejo seguro de reasoning_steps
                    reasoning_steps_value = reasoning_steps.mean().item() if reasoning_steps is not None and hasattr(reasoning_steps, 'mean') else None
                    mtp_loss_value = mtp_loss.item() if mtp_loss is not None else None
                    
                    diagnostics.update(
                        loss=ce_loss.item(),
                        right_metabolism=float(right_node.metabolism),
                        right_fatigue=float(right_node.fatigue),
                        right_liquid_norm=liquid_norm,
                        right_homeostasis=float(right_node.homeostasis),
                        callosal_flow=callosal_flow,
                        left_gate_mean=gate_mean_val,
                        left_gate_std=gate_std_val,
                        synergy_score=synergy,
                        health_score=health_score,
                        linguistic_reward=linguistic_reward.item() if linguistic_reward is not None else None,
                        alignment_loss=alignment_loss.item() if epoch < 6 else None,
                        objects_fatigue=channel_fatigue['objects'],
                        actions_fatigue=channel_fatigue['actions'],
                        scene_fatigue=channel_fatigue['scene'],
                        reasoning_steps=reasoning_steps_value,
                        mtp_loss=mtp_loss_value
                    )
            
            # Actualizar metrics en progress bar
            pbar_dict = {
                'loss': f'{ce_loss.item():.3f}',
                'g_pen': f'{gate_penalty.item():.3f}',
                'liquid': f'{liquid_norm:.2f}',
                'gate': f'{gate_mean_val:.2f}',
                'reward': f'{linguistic_reward.item():.3f}' if linguistic_reward is not None else 'N/A',
                'align': f'{alignment_loss.item():.3f}' if epoch < 6 else 'N/A',
                'obj_f': f"{model.corpus_callosum.objects_fatigue.item():.2f}",
                'act_f': f"{model.corpus_callosum.actions_fatigue.item():.2f}",
                'sce_f': f"{model.corpus_callosum.scene_fatigue.item():.2f}",
                'mem': f"{len(episodic_memory.buffer)}/{episodic_memory.capacity}"
            }
            
            if reasoning_steps is not None:
                pbar_dict['reason'] = f"{reasoning_steps.mean().item():.1f}" if hasattr(reasoning_steps, 'mean') else f"{reasoning_steps:.1f}"
            
            if mtp_loss is not None:
                pbar_dict['mtp'] = f"{mtp_loss.item():.3f}"
            
            pbar.set_postfix(pbar_dict)
            
            total_loss += ce_loss.item()
            num_batches += 1
        
        scheduler.step()
        
        # Evaluación lingüística cada 2 épocas
        if epoch % 2 == 0:
            model.eval()
            print("\n📸 EVALUACIÓN LINGÜÍSTICA...\n")
            
            bleu_scores, acc_scores, overlap_scores, cider_scores, spice_scores = [], [], [], [], []
            reasoning_steps_list = []
            mtp_losses_list = []
            
            with torch.no_grad():
                for sample_idx in range(min(10, len(dataset))):
                    sample_img, sample_cap, raw_caption = dataset[sample_idx * (len(dataset) // 10)]
                    sample_img = sample_img.unsqueeze(0).to(device)
                    
                    generated = model(sample_img, captions=None, epoch=epoch)
                    
                    gen_words = [id2word.get(int(t.item()), '<UNK>') for t in generated[0]]
                    gen_sentence = " ".join(w for w in gen_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                    
                    bleu = diagnostics.language_metrics.sentence_bleu(raw_caption, gen_sentence)
                    acc = diagnostics.language_metrics.token_accuracy(raw_caption, gen_sentence)
                    overlap = diagnostics.language_metrics.word_overlap(raw_caption, gen_sentence)
                    cider = cognitive_system.linguistic_feedback.compute_cider(raw_caption, gen_sentence)
                    spice = cognitive_system.linguistic_feedback.compute_spice(raw_caption, gen_sentence)
                    
                    bleu_scores.append(bleu)
                    acc_scores.append(acc)
                    overlap_scores.append(overlap)
                    cider_scores.append(cider)
                    spice_scores.append(spice)
                    
                    if sample_idx < 3:
                        print(f"Muestra {sample_idx + 1}:")
                        print(f"  GT:   {raw_caption}")
                        print(f"  Gen:  {gen_sentence}")
                        print(f"  BLEU: {bleu:.3f} | Acc: {acc:.3f} | Overlap: {overlap:.3f}")
                        print(f"  CIDEr: {cider:.3f} | SPICE: {spice:.3f}\n")
            
            diagnostics.update(
                bleu_score=np.mean(bleu_scores),
                token_accuracy=np.mean(acc_scores),
                word_overlap=np.mean(overlap_scores),
                cider_score=np.mean(cider_scores),
                spice_score=np.mean(spice_scores)
            )
            
            model.train()
        
        else:
            model.eval()
            print("\n📸 MUESTRAS RÁPIDAS...\n")
            
            with torch.no_grad():
                for sample_idx in range(3):
                    sample_img, sample_cap, raw_caption = dataset[sample_idx * 100]
                    sample_img = sample_img.unsqueeze(0).to(device)
                    
                    generated = model(sample_img, captions=None, epoch=epoch)
                    
                    gen_words = [id2word.get(int(t.item()), '<UNK>') for t in generated[0]]
                    gen_sentence = " ".join(w for w in gen_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                    
                    print(f"Muestra {sample_idx + 1}:")
                    print(f"  GT:   {raw_caption}")
                    print(f"  Gen:  {gen_sentence}\n")
            
            model.train()
        
        diagnostics.report(epoch)
        
        avg_loss = total_loss / num_batches
        bleu_avg = diagnostics.get_recent_avg('bleu_score')
        cider_avg = diagnostics.get_recent_avg('cider_score')
        
        # Estadísticas de fatiga
        objects_fatigue = diagnostics.get_recent_avg('objects_fatigue')
        actions_fatigue = diagnostics.get_recent_avg('actions_fatigue')
        scene_fatigue = diagnostics.get_recent_avg('scene_fatigue')
        
        print(f"Época {epoch:02d} | Loss: {avg_loss:.4f} | Health: {health_score}/5 | Med: {medicine_level} | Cog: {cognitive_level}")
        print(f"BLEU: {bleu_avg:.3f} | CIDEr: {cider_avg:.3f} | Gate: {gate_mean:.3f} | Liquid: {liquid:.2f}")
        print(f"Fatiga - Objetos: {objects_fatigue:.3f} | Acciones: {actions_fatigue:.3f} | Escena: {scene_fatigue:.3f}")
        print(f"Memoria: {len(episodic_memory.buffer)}/{episodic_memory.capacity} samples\n")
        
        if epoch % 5 == 0:
            cache_stats = cognitive_system.linguistic_feedback.get_cache_stats()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'id2word': id2word,
                'diagnostics': diagnostics.history,
                'medical_interventions': medical_system.intervention_history,
                'cognitive_interventions': cognitive_system.cognitive_history,
                'cache_stats': cache_stats,
                'episodic_memory_size': len(episodic_memory.buffer)
            }, f'./checkpoints/stable_epoch_{epoch:02d}.pth')
    
    print("✅ Entrenamiento completado!")
    diagnostics.report(29)
    
    # Resumen final
    print(f"\n{'='*80}")
    print("📋 HISTORIAL DE INTERVENCIONES")
    print(f"{'='*80}")
    
    print("\n🏥 INTERVENCIONES MÉDICAS:")
    if len(medical_system.intervention_history) == 0:
        print("✓ No se requirieron intervenciones médicas\n")
    else:
        for intervention in medical_system.intervention_history:
            print(f"Época {intervention['epoch']:02d} | {intervention['level']} | Severidad: {intervention['severity']}/12")
            print(f"  Issues: {', '.join(intervention['issues'])}")
            for inter in intervention['interventions']:
                print(f"    • {inter}")
            print()
    
    print("\n🧠 INTERVENCIONES COGNITIVAS:")
    if len(cognitive_system.cognitive_history) == 0:
        print("✓ No se requirieron intervenciones cognitivas\n")
    else:
        for intervention in cognitive_system.cognitive_history:
            if intervention['severity'] > 0:
                print(f"Época {intervention['epoch']:02d} | Severidad: {intervention['severity']}/9")
                print(f"  Issues: {', '.join(intervention['issues'])}")
                print()
    
    # Estadísticas finales
    final_cache_stats = cognitive_system.linguistic_feedback.get_cache_stats()
    print(f"\n⚡ ESTADÍSTICAS FINALES:")
    print(f"  Caché: {final_cache_stats['cache_size']} entradas")
    print(f"  Hit rate: {final_cache_stats['hit_rate']:.2%}")
    print(f"  Memoria episódica: {len(episodic_memory.buffer)} samples")
    
    final_fatigue = {
        'objects': diagnostics.get_recent_avg('objects_fatigue'),
        'actions': diagnostics.get_recent_avg('actions_fatigue'),
        'scene': diagnostics.get_recent_avg('scene_fatigue')
    }
    print(f"\n🔗 FATIGA FINAL POR CANAL:")
    for k, v in final_fatigue.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    train_with_metrics()
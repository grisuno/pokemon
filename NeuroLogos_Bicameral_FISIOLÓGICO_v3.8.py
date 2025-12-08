# =============================================================================
# NeuroLogos Bicameral FISIOLÓGICO v3.5
# + Métricas lingüísticas (BLEU, Accuracy)
# + Sistema médico calibrado por niveles
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

def compute_loss(logits, captions, gate, vocab, linguistic_reward=None, lambda_reward=0.1):
    """
    Función de pérdida extendida que incorpora recompensa lingüística
    """
    # Pérdida de entropía cruzada estándar
    ce_loss = F.cross_entropy(
        logits.reshape(-1, len(vocab)),
        captions[:, 1:].reshape(-1),
        ignore_index=vocab['<PAD>']
    )
    
    # Penalizaciones del gate (sin cambios)
    gate_mean = gate.mean()
    gate_penalty = F.relu(gate_mean - 0.5) ** 2
    gate_diversity = gate.std()
    diversity_penalty = F.relu(0.15 - gate_diversity) ** 2
    
    # Término de recompensa lingüística (maximizar la recompensa)
    linguistic_loss = 0.0
    if linguistic_reward is not None:
        # Usamos negative porque queremos minimizar la pérdida
        linguistic_loss = -lambda_reward * linguistic_reward
    
    # Combinación de todos los términos
    total_loss = ce_loss + 0.05 * gate_penalty + 0.2 * diversity_penalty + linguistic_loss
    
    return total_loss, ce_loss, gate_penalty, diversity_penalty, linguistic_loss


class NeurocognitiveSystem:
    """
    Sistema neurocognitivo que complementa al sistema médico
    para optimizar el aprendizaje lingüístico
    """
    
    def __init__(self):
        self.cognitive_history = []
        self.last_intervention_epoch = -5
        self.linguistic_feedback = LinguisticFeedbackLoop()
        
        # Umbrales para intervención cognitiva
        self.cider_threshold = 0.1
        self.spice_threshold = 0.15
        self.plateau_threshold = 0.005  # Mejora mínima esperada
        
    def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch):
        """
        Evalúa el estado cognitivo del modelo basándose en métricas lingüísticas
        """
        issues = []
        severity = 0
        confidence = []
        
        # Detectar estancamiento en CIDEr
        if epoch > 5 and len(self.cognitive_history) > 3:
            recent_cider = [h['cider'] for h in self.cognitive_history[-3:]]
            cider_improvement = recent_cider[-1] - recent_cider[0]
            
            if cider_improvement < self.plateau_threshold:
                issues.append("cider_plateau")
                severity += 2
                confidence.append(f"CIDEr estancado (mejora: {cider_improvement:.3f})")
        
        # Detectar bajo rendimiento semántico
        if spice_score < self.spice_threshold:
            issues.append("semantic_deficit")
            severity += 3
            confidence.append(f"Deficit semántico (SPICE: {spice_score:.3f})")
        
        # Detectar bajo rendimiento general
        if combined_reward < 0.2:
            issues.append("linguistic_deficit")
            severity += 2
            confidence.append(f"Deficit lingüístico general (recompensa: {combined_reward:.3f})")
        
        # Detectar sobreajuste a n-gramas (alto CIDEr, bajo SPICE)
        if cider_score > 0.15 and spice_score < self.spice_threshold:
            issues.append("syntactic_overfitting")
            severity += 2
            confidence.append(f"Sobreajuste sintáctico (CIDEr: {cider_score:.3f}, SPICE: {spice_score:.3f})")
        
        # Guardar historial
        self.cognitive_history.append({
            'epoch': epoch,
            'cider': cider_score,
            'spice': spice_score,
            'combined': combined_reward,
            'issues': issues,
            'severity': severity
        })
        
        return issues, severity, confidence
    
    def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch):
        """
        Aplica intervenciones cognitivas basadas en el estado lingüístico
        """
        if epoch - self.last_intervention_epoch < 1:
            return False
        
        if severity == 0:
            return False
        
        # Determinar nivel
        if severity <= 2:
            cog_level = "🟡 Nivel Cognitivo 1 (Suave)"
        elif severity <= 5:
            cog_level = "🟠 Nivel Cognitivo 2 (Moderado)"
        else:
            cog_level = "🔴 Nivel Cognitivo 3 (Agresivo)"
        
        print(f"\n{'='*80}")
        print(f"🧠 INTERVENCIÓN NEUROCOGNITIVA - {cog_level} - Severidad: {severity}/9")
        print(f"   Problemas detectados: {', '.join(issues)}")
        print(f"   📊 CONFIANZA:")
        for conf in confidence:
            print(f"      • {conf}")
        print(f"{'='*80}")
        
        interventions_applied = []
        
        with torch.no_grad():
            # INTERVENCIÓN ESPECÍFICA POR PATRÓN COGNITIVO
            
            # Patrón 1: CIDEr estancado
            if "cider_plateau" in issues:
                print("💊 PATRÓN COGNITIVO: CIDEr estancado")
                print("💊 Acción: Reforzar atención visual-lingüística")
                
                # CORRECCIÓN: Usar los nombres correctos para los atributos de MultiheadAttention
                try:
                    if hasattr(model.left_hemisphere.visual_attention, 'in_proj_weight'):
                        model.left_hemisphere.visual_attention.in_proj_weight.data *= 1.1
                    if hasattr(model.left_hemisphere.visual_attention, 'out_proj'):
                        if hasattr(model.left_hemisphere.visual_attention.out_proj, 'weight'):
                            model.left_hemisphere.visual_attention.out_proj.weight.data *= 1.1
                except Exception as e:
                    print(f"    Advertencia: No se pudo modificar atención visual: {e}")
                    
                # Ajustar el gate para permitir más diversidad
                model.left_hemisphere.liquid_gate[-1].bias.data -= 0.2
                
                interventions_applied.append("visual_attention_boost")
                interventions_applied.append("gate_diversity_adjustment")
            
            # Patrón 2: Deficit semántico
            if "semantic_deficit" in issues:
                print("🧠 PATRÓN COGNITIVO: Deficit semántico")
                print("🧠 Acción: Reforzar proyección semántica")
                
                # Reforzar la capa de proyección de salida
                model.left_hemisphere.output_projection.weight.data *= 1.15
                
                # Ajustar el corpus callosum para mejorar transferencia semántica
                model.corpus_callosum.residual_scale.data *= 1.05
                
                interventions_applied.append("semantic_projection_boost")
                interventions_applied.append("callosum_semantic_adjustment")
            
            # Patrón 3: Deficit lingüístico general
            if "linguistic_deficit" in issues:
                print("🧠 PATRÓN COGNITIVO: Deficit lingüístico general")
                print("🧠 Acción: Reforzar todo el hemisferio izquierdo")
                
                # Reforzar el embedding
                model.left_hemisphere.embedding.weight.data *= 1.05
                
                # Reforzar el LSTM
                for name, param in model.left_hemisphere.lstm.named_parameters():
                    if 'weight' in name:
                        param.data *= 1.05
                
                interventions_applied.append("left_hemisphere_general_boost")
            
            # Patrón 4: Sobreajuste sintáctico
            if "syntactic_overfitting" in issues:
                print("🧠 PATRÓN COGNITIVO: Sobreajuste sintáctico")
                print("🧠 Acción: Regularizar y promover diversidad")
                
                # Aumentar dropout en el gate
                for layer in model.left_hemisphere.liquid_gate:
                    if isinstance(layer, nn.Dropout):
                        layer.p = min(0.5, layer.p + 0.1)
                
                # Reducir la influencia del gate para permitir más exploración
                model.left_hemisphere.liquid_gate[-1].bias.data += 0.3
                
                interventions_applied.append("syntactic_overfitting_regularization")
        
        print(f"\n✓ Intervenciones cognitivas aplicadas: {len(interventions_applied)}")
        for intervention in interventions_applied:
            print(f"  - {intervention}")
        print(f"{'='*80}\n")
        
        self.last_intervention_epoch = epoch
        
        return True


# Sistema de retroalimentación lingüística
class LinguisticFeedbackLoop:
    """Sistema que integra métricas lingüísticas en el proceso de aprendizaje"""
    
    def __init__(self, alpha=0.7, beta=0.3):
        # Pesos para combinar métricas
        self.alpha = alpha  # Peso para CIDEr
        self.beta = beta    # Peso para SPICE
        self.history = []
        
        # MEJORA: Añadir caché para n-gramas y optimizar cálculo
        self.ngram_cache = {}
        
    def compute_linguistic_reward(self, references, hypotheses):
        """
        Calcula una recompensa combinada basada en CIDEr y SPICE
        que puede usarse para guiar el entrenamiento
        """
        cider_scores = [self.compute_cider(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        spice_scores = [self.compute_spice(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        
        # Combinación lineal de métricas
        combined_scores = [self.alpha * c + self.beta * s for c, s in zip(cider_scores, spice_scores)]
        
        # Normalizar a [0, 1]
        reward = torch.tensor(combined_scores).mean().clamp(0, 1)
        
        self.history.append({
            'cider': np.mean(cider_scores),
            'spice': np.mean(spice_scores),
            'combined': reward.item()
        })
        
        return reward
    
    def compute_cider(self, reference, hypothesis):
        """Versión simplificada de CIDEr para uso en entrenamiento"""
        # MEJORA: Usar caché para n-gramas si está disponible
        ref_key = hash(reference)
        if ref_key not in self.ngram_cache:
            self.ngram_cache[ref_key] = self._get_ngrams(reference, n=4)
        ref_ngrams = self.ngram_cache[ref_key]
        
        hyp_key = hash(hypothesis)
        if hyp_key not in self.ngram_cache:
            self.ngram_cache[hyp_key] = self._get_ngrams(hypothesis, n=4)
        hyp_ngrams = self.ngram_cache[hyp_key]
        
        # Cálculo de similitud del coseno - CORREGIDO
        overlap = 0
        for ngram, count in hyp_ngrams.items():
            if ngram in ref_ngrams:
                overlap += min(ref_ngrams[ngram], count)
    
        ref_len = sum(ref_ngrams.values())
        hyp_len = sum(hyp_ngrams.values())
        
        if ref_len == 0 or hyp_len == 0:
            return 0.0
            
        return overlap / (ref_len * hyp_len) ** 0.5
    
    def compute_spice(self, reference, hypothesis):
        """Versión simplificada de SPICE para uso en entrenamiento"""
        # Para simplificar, usamos Jaccard similarity como proxy semántico
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        
        if len(ref_words) == 0 and len(hyp_words) == 0:
            return 1.0
        
        intersection = len(ref_words & hyp_words)
        union = len(ref_words | hyp_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_ngrams(self, sentence, n=4):
        """Extrae n-gramas de una oración"""
        tokens = sentence.lower().split()
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams



# =============================================================================
# MÉTRICAS LINGÜÍSTICAS
# =============================================================================
class LanguageMetrics:
    """Métricas de calidad de generación"""
    
    @staticmethod
    def sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
        """BLEU simplificado a nivel de oración"""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        # N-gram precision
        precisions = []
        for n in range(1, 5):
            ref_ngrams = LanguageMetrics._get_ngrams(ref_tokens, n)
            hyp_ngrams = LanguageMetrics._get_ngrams(hyp_tokens, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            matches = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            precisions.append(matches / total if total > 0 else 0.0)
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens))))
        
        # Geometric mean
        if all(p > 0 for p in precisions):
            score = bp * np.exp(sum(w * np.log(p) for w, p in zip(weights, precisions)))
        else:
            score = 0.0
        
        return score
    
    @staticmethod
    def _get_ngrams(tokens, n):
        """Extraer n-gramas de una lista de tokens"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    @staticmethod
    def token_accuracy(reference, hypothesis):
        """Porcentaje de tokens correctos en posición"""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        min_len = min(len(ref_tokens), len(hyp_tokens))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) if ref_tokens[i] == hyp_tokens[i])
        return matches / max(len(ref_tokens), len(hyp_tokens))
    
    @staticmethod
    def word_overlap(reference, hypothesis):
        """Jaccard similarity entre palabras"""
        ref_set = set(reference.lower().split())
        hyp_set = set(hypothesis.lower().split())
        
        if len(ref_set | hyp_set) == 0:
            return 0.0
        
        return len(ref_set & hyp_set) / len(ref_set | hyp_set)

# =============================================================================
# SISTEMA MÉDICO CALIBRADO
# =============================================================================
class TriangulatedMedicalSystem:
    """Sistema médico con triangulación de señales convergentes"""
    
    def __init__(self):
        self.intervention_history = []
        self.last_intervention_epoch = -5
        self.signal_history = []  # Historial de señales
    
    def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow):
        """Identificar señales convergentes que confirman problemas"""
        
        signals = {
            'gate_saturated': False,
            'gate_no_diversity': False,
            'callosum_blocked': False,
            'liquid_high': False,
            'homeostasis_low': False
        }
        
        # Señal 1: Gate saturado (>0.85)
        if gate_mean > 0.85:
            signals['gate_saturated'] = True
        
        # Señal 2: Gate sin diversidad (std <0.05)
        if gate_std < 0.05:
            signals['gate_no_diversity'] = True
        
        # Señal 3: Callosum bloqueado (<0.35) - FIX: usar parámetro directo
        if callosal_flow < 0.35:
            signals['callosum_blocked'] = True
        
        # Señal 4: Liquid alto (>2.5)
        if liquid_norm > 2.5:
            signals['liquid_high'] = True
        
        # Señal 5: Health bajo (<=2)
        if health_score <= 2:
            signals['homeostasis_low'] = True
        
        return signals
    def count_convergent_signals(self, signals, pattern):
        """Contar cuántas señales del patrón están activas"""
        return sum([signals[sig] for sig in pattern if sig in signals])
    
    def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow):
        """Diagnosticar SOLO con confirmación múltiple"""
        
        signals = self.triangulate_signals(health_score, liquid_norm, gate_mean, gate_std, callosal_flow)
        
        issues = []
        severity = 0
        confidence = []
        
        # PATRÓN 1: Gate completamente roto (TRIPLE CONFIRMACIÓN)
        gate_broken_pattern = ['gate_saturated', 'gate_no_diversity', 'callosum_blocked']
        gate_broken_count = self.count_convergent_signals(signals, gate_broken_pattern)
        
        if gate_broken_count >= 3:
            issues.append("gate_system_failure")
            severity += 6
            confidence.append(f"Gate roto (3/3 señales: 100% confianza)")
        elif gate_broken_count == 2:
            issues.append("gate_degraded")
            severity += 4
            confidence.append(f"Gate degradado (2/3 señales: 67% confianza)")
        
        # PATRÓN 2: Comunicación cerebral colapsada (DOBLE CONFIRMACIÓN)
        comm_broken_pattern = ['callosum_blocked', 'gate_no_diversity']
        comm_broken_count = self.count_convergent_signals(signals, comm_broken_pattern)
        
        if comm_broken_count >= 2 and 'gate_system_failure' not in issues:
            issues.append("communication_collapse")
            severity += 4
            confidence.append(f"Comunicación colapsada (2/2 señales: 100% confianza)")
        
        # PATRÓN 3: Liquid fuera de control (SIMPLE + HEALTH)
        if signals['liquid_high'] and signals['homeostasis_low']:
            issues.append("liquid_crisis")
            severity += 5
            confidence.append(f"Crisis liquid (2/2 señales: 100% confianza)")
        elif signals['liquid_high']:
            issues.append("liquid_elevated")
            severity += 2
            confidence.append(f"Liquid alto (1/2 señales: 50% confianza)")
        
        # PATRÓN 4: Sistema saludable con gate alto (FALSA ALARMA)
        if gate_mean > 0.80 and gate_std > 0.05 and callosal_flow > 0.3:
            # Gate alto PERO con diversidad Y comunicación = OK
            if 'gate_system_failure' in issues:
                issues.remove('gate_system_failure')
                severity -= 6
                confidence.append(f"Gate alto pero funcional (falsa alarma evitada)")
            elif 'gate_degraded' in issues:
                issues.remove('gate_degraded')
                severity -= 4
                confidence.append(f"Gate alto pero funcional (falsa alarma evitada)")
        
        # Guardar historial
        self.signal_history.append({
            'signals': signals,
            'issues': issues,
            'severity': severity,
            'confidence': confidence
        })
        
        return issues, severity, confidence
        
    def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch):
        """Aplicar intervención SOLO si confianza es alta"""
        
        if epoch - self.last_intervention_epoch < 1:
            return False
        
        if severity == 0:
            return False
        
        # Determinar nivel
        if severity <= 3:
            med_level = "🟡 Nivel 1 (Suave)"
        elif severity <= 6:
            med_level = "🟠 Nivel 2 (Moderado)"
        else:
            med_level = "🔴 Nivel 3 (Agresivo)"
        
        print(f"\n{'='*80}")
        print(f"🏥 INTERVENCIÓN TRIANGULADA - {med_level} - Severidad: {severity}/12")
        print(f"   Problemas detectados: {', '.join(issues)}")
        print(f"   📊 CONFIANZA:")
        for conf in confidence:
            print(f"      • {conf}")
        print(f"{'='*80}")
        
        interventions_applied = []
        
        with torch.no_grad():
            right_node = model.right_hemisphere.spatial_liquid
            
            # INTERVENCIÓN ESPECÍFICA POR PATRÓN
            
            # Patrón 1: Gate completamente roto (severity 6)
            if "gate_system_failure" in issues:
                print("🚨 PATRÓN CRÍTICO: Gate completamente roto")
                print("🚨 Acción: DEMOLICIÓN TOTAL del gate")
                
                for layer in model.left_hemisphere.liquid_gate:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.05)
                        if layer.bias is not None:
                            layer.bias.data.zero_()
                
                # FIX: Más agresivo - resetear bias a 0 para empezar neutral
                model.left_hemisphere.liquid_gate[-1].bias.data.fill_(0.0)
                # FIX: Más agresivo - resetear pesos a valores más altos
                model.left_hemisphere.liquid_gate[-1].weight.data.mul_(0.05)
                
                for layer in model.left_hemisphere.liquid_gate:
                    if isinstance(layer, nn.Dropout):
                        layer.p = 0.6
                
                interventions_applied.append("gate_total_demolition")
                
                # FIX: Verificar callosum_blocked desde signal_history
                if len(self.signal_history) > 0 and self.signal_history[-1]['signals']['callosum_blocked']:
                    print("🚨 Acción adicional: Reconstruir callosum")
                    for i, block in enumerate(model.corpus_callosum.transfer):
                        for layer in block:
                            if isinstance(layer, nn.Linear):
                                nn.init.xavier_uniform_(layer.weight, gain=1.2)
                                if layer.bias is not None:
                                    layer.bias.data.zero_()
                    
                    model.corpus_callosum.residual_scale.data.fill_(0.92)
                    interventions_applied.append("callosum_rebuild")
            
            # Patrón 2: Gate degradado (severity 4)
            elif "gate_degraded" in issues:
                print("💊 PATRÓN MODERADO: Gate degradado")
                print("💊 Acción: RESET AGRESIVO del gate")
                
                for layer in model.left_hemisphere.liquid_gate:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.15)
                        if layer.bias is not None:
                            layer.bias.data.zero_()
                
                # FIX: Más agresivo - resetear bias a 0 para empezar neutral
                model.left_hemisphere.liquid_gate[-1].bias.data.fill_(0.0)
                # FIX: Más agresivo - resetear pesos a valores más altos
                model.left_hemisphere.liquid_gate[-1].weight.data.mul_(0.05)
                
                for layer in model.left_hemisphere.liquid_gate:
                    if isinstance(layer, nn.Dropout):
                        layer.p = min(0.5, layer.p + 0.15)
                
                interventions_applied.append("gate_aggressive_reset")
            
            # Patrón 3: Comunicación colapsada
            if "communication_collapse" in issues:
                print("💊 PATRÓN CRÍTICO: Comunicación colapsada")
                print("💊 Acción: Reforzar corpus callosum")
                
                model.corpus_callosum.residual_scale.data.fill_(0.90)
                
                for i, block in enumerate(model.corpus_callosum.transfer):
                    for layer in block:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight, gain=1.0)
                            if layer.bias is not None:
                                layer.bias.data.zero_()
                
                interventions_applied.append("callosum_boost_strong")
            
            # Patrón 4: Liquid crisis
            if "liquid_crisis" in issues:
                print("🚨 PATRÓN CRÍTICO: Crisis de liquid")
                print("🚨 Acción: RESET TOTAL de liquid + homeostasis")
                
                right_node.W_fast_short = 0.00005 * torch.randn_like(right_node.W_fast_short)  # CORREGIDO: 0.00005 → 0.00005
                right_node.W_fast_long = 0.00005 * torch.randn_like(right_node.W_fast_long)  # CORREGIDO: 0.00005 → 0.00005
                right_node.norm_ema = torch.tensor(0.3, device=right_node.norm_ema.device)  # CORREGIDO: device=right_node.norm_ema.device
                right_node.homeostasis = torch.tensor(1.0, device=right_node.homeostasis.device)  # CORREGIDO: device=right_node.homeostasis.device
                right_node.metabolism = torch.tensor(0.7, device=right_node.metabolism.device)  # CORREGIDO: device=right_node.metabolism.device
                right_node.fatigue = torch.tensor(0.0, device=right_node.fatigue.device)  # CORREGIDO: device=right_node.fatigue.device
                
                interventions_applied.append("liquid_full_reset")
                interventions_applied.append("homeostasis_restore")
            
            elif "liquid_elevated" in issues:
                print("💊 Acción: Reducir liquid (60%)")
                right_node.W_fast_short *= 0.4
                right_node.W_fast_long *= 0.4
                interventions_applied.append("liquid_reduce")
            
            # Mantenimiento general
            if severity >= 3:
                print("💊 Mantenimiento: Reduciendo fatiga")
                right_node.fatigue *= 0.5
                interventions_applied.append("fatigue_reduction")
        
        print(f"\n✓ Intervenciones aplicadas: {len(interventions_applied)}")
        for intervention in interventions_applied:
            print(f"  - {intervention}")
        print(f"{'='*80}\n")
        
        self.intervention_history.append({
            'epoch': epoch,
            'severity': severity,
            'level': med_level,
            'issues': issues,
            'confidence': confidence,
            'interventions': interventions_applied
        })
        self.last_intervention_epoch = epoch
        
        return True






# =============================================================================
# CONSTANTES
# =============================================================================
IMG_SIZE = 224
MAX_CAPTION_LEN = 30
VOCAB_SIZE = 5000
EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 64
NUM_WORKERS = 4

# =============================================================================
# LIQUID NEURON
# =============================================================================
class StableLiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        # Dimensiones calculadas
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.slow_expansion = 256  # Expansión del slow pathway
        self.fast_long_dim = 256   # Dimensión del fast long pathway
        
        # Dimensión total concatenada
        self.slow_total = out_dim + self.slow_expansion  # 512 + 256 = 768
        self.fast_short_dim = out_dim  # 512
        self.concat_dim = self.slow_total + self.fast_short_dim + self.fast_long_dim  # 768 + 512 + 256 = 1536
        
        # Slow pathway expandido
        self.W_slow = nn.Linear(in_dim, self.slow_total, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=0.8)  # CORREGIDO: init.orthogonal_ → init.orthogonal_
        
        # Fast pathways multi-escala
        self.register_buffer('W_fast_short', 0.0001 * torch.randn(self.fast_short_dim, in_dim))
        self.register_buffer('W_fast_long', 0.00005 * torch.randn(self.fast_long_dim, in_dim))
        
        # LayerNorm sobre dimensión concatenada completa
        self.ln = nn.LayerNorm(self.concat_dim)
        
        # Proyección final: concat_dim → out_dim
        self.project = nn.Linear(self.concat_dim, out_dim)
        
        # Fisiología
        self.register_buffer('metabolism', torch.tensor(0.6))
        self.register_buffer('fatigue', torch.tensor(0.0))
        self.register_buffer('sensitivity', torch.tensor(0.5))
        self.register_buffer('homeostasis', torch.tensor(1.0))
        
        self.base_lr = 0.001
        self.register_buffer('norm_ema', torch.tensor(0.5))
        self.register_buffer('norm_target', torch.tensor(1.0))
        
    def forward(self, x):
        # Pathways con dimensiones correctas
        slow_out = self.W_slow(x)  # [B, slow_total=768]
        fast_short = F.linear(x, self.W_fast_short)  # [B, fast_short_dim=512]
        fast_long = F.linear(x, self.W_fast_long)  # [B, fast_long_dim=256]
        
        gate_short = 0.05 + 0.15 * float(self.sensitivity) * float(self.homeostasis)
        gate_long = 0.02 + 0.08 * float(self.metabolism)
        
        # Concatenación: [B, concat_dim=1536]
        combined = torch.cat([
            slow_out,
            gate_short * fast_short,
            gate_long * fast_long
        ], dim=-1)
        
        # Normalizar y proyectar
        out = self.ln(combined)  # [B, concat_dim=1536]
        out = self.project(out)  # [B, out_dim=512]
        
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
            
            # Actualizar solo las dimensiones correctas
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
            self.metabolism = 0.9 * self.metabolism + 0.1 * target_metab
            self.metabolism = self.metabolism.clamp(0.3, 0.9)
            
            norm_ratio = self.norm_ema / self.norm_target
            fatigue_increment = 0.002 if norm_ratio < 2.0 else 0.01
            self.fatigue *= 0.99
            self.fatigue += fatigue_increment
            self.fatigue = self.fatigue.clamp(0, 0.5)  # CORREGIDO: self.fatiga.clamp → self.fatigue.clamp
            
            if float(self.homeostasis) < 0.7:
                self.sensitivity *= 0.95
            else:
                target_sens = 0.5 + 0.2 * (1.0 - float(self.fatigue))
                self.sensitivity = 0.95 * self.sensitivity + 0.05 * target_sens
            self.sensitivity = self.sensitivity.clamp(0.3, 0.7)



# =============================================================================
# ARQUITECTURA
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

class LeftHemisphere(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.lstm_expansion = 128
        self.lstm_output_dim = hidden_dim + self.lstm_expansion  # 640
        self.register_buffer('gate_warmup', torch.tensor(0.1))
        self.gate_warmup_epochs = 3
        
        # LSTM con dimensión expandida
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim,  # input: 256 + 512 = 768
            self.lstm_output_dim,    # output: 640
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Bottleneck: 640 → 512
        self.bottleneck = nn.Linear(self.lstm_output_dim, hidden_dim)
        
        # Gate con arquitectura progresiva
        self.gate_dim_1 = hidden_dim // 2  # 256
        self.gate_dim_2 = hidden_dim // 4  # 128
        
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
        
        # Visual attention
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.gate_target = 0.3
    
    def beam_search_decode(self, visual_context, beam_width=5, max_len=30, epoch=0):
        batch_size = visual_context.size(0)
        device = visual_context.device
        
        # Inicializar con token BOS
        start_token = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        initial_score = torch.zeros(batch_size, 1, device=device)
        
        # Inicializar beam
        beams = [(start_token, initial_score, self._get_init_state(visual_context))]
        completed = []
        
        for step in range(max_len):
            new_beams = []
            
            for seq, score, hidden in beams:
                if len(completed) >= beam_width:
                    break
                    
                # Obtener siguiente token
                last_token = seq[:, -1:]
                emb = self.embedding(last_token)
                visual_expanded = visual_context.unsqueeze(1)
                lstm_input = torch.cat([emb, visual_expanded], dim=2)
                
                out, hidden = self.lstm(lstm_input, hidden)
                out = self.bottleneck(out)
                
                visual_query = visual_context.unsqueeze(1)
                attended, _ = self.visual_attention(out, visual_query, visual_query)
                out = out + 0.5 * attended
                
                # AÑADIR: Aplicar warmup al gate
                warmup_factor = min(1.0, epoch / self.gate_warmup_epochs)
                adjusted_bias = self.liquid_gate[-1].bias.data * (1 - warmup_factor) + \
                                torch.tensor(-2.5) * warmup_factor
                
                gate_logits = self.liquid_gate(out)
                gate = torch.sigmoid(gate_logits + adjusted_bias)
                out = out * (0.5 + 0.5 * gate)
                
                logits = self.output_projection(out.squeeze(1))
                log_probs = F.log_softmax(logits / 0.9, dim=-1)
                
                # Obtener top-k candidatos
                topk_probs, topk_tokens = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    token = topk_tokens[:, i:i+1]
                    prob = topk_probs[:, i:i+1]
                    new_seq = torch.cat([seq, token], dim=1)
                    # CORRECCIÓN: Usar .mean() para obtener un escalar del tensor
                    new_score = score + prob.mean(dim=0, keepdim=True)
                    
                    # CORREGIDO: token es un tensor, no un escalar
                    if token[0, 0].item() == 2:  # EOS token
                        completed.append((new_seq, new_score))
                    else:
                        new_beams.append((new_seq, new_score, hidden))
            
            # Seleccionar mejores beams
            # CORRECCIÓN: Usar .mean() para obtener un escalar del tensor
            new_beams.sort(key=lambda x: x[1].mean().item(), reverse=True)
            beams = new_beams[:beam_width - len(completed)]
            
            if len(completed) >= beam_width:
                break
        
        # Seleccionar mejor secuencia completada
        if completed:
            # CORRECCIÓN: Usar .mean() para obtener un escalar del tensor
            completed.sort(key=lambda x: x[1].mean().item(), reverse=True)
            return completed[0][0]
        else:
            return beams[0][0]
    
    def forward(self, visual_context, captions=None, max_len=30, epoch=0):
        batch_size = visual_context.size(0)
        device = visual_context.device
        
        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            seq_len = embeddings.size(1)
            
            visual_expanded = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_input = torch.cat([embeddings, visual_expanded], dim=2)
            
            lstm_out, _ = self.lstm(lstm_input, self._get_init_state(visual_context))
            lstm_out = self.bottleneck(lstm_out)
            
            # Cross-attention
            visual_query = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
            attended, _ = self.visual_attention(lstm_out, visual_query, visual_query)
            
            lstm_out = lstm_out + 0.5 * attended
            
            warmup_factor = min(1.0, epoch / self.gate_warmup_epochs)
            adjusted_bias = self.liquid_gate[-1].bias.data * (1 - warmup_factor) + \
                            torch.tensor(-2.5) * warmup_factor
            
            gate_logits = self.liquid_gate(lstm_out)
            gate = torch.sigmoid(gate_logits + adjusted_bias)
            
            # FIX: Eliminar el clamp para permitir que el gate funcione más libremente
            # gate = torch.clamp(gate, 0.1, 0.9)
            
            modulated = lstm_out * (0.5 + 0.5 * gate)
            logits = self.output_projection(modulated)
            
            return logits, gate
        else:
            # Usar beam search para generación
            return self.beam_search_decode(visual_context, beam_width=5, max_len=max_len, epoch=epoch)
    
    def _get_init_state(self, visual_context):
        # Estado inicial: [2, batch, lstm_output_dim]
        batch_size = visual_context.size(0)
        h0 = visual_context.unsqueeze(0).repeat(2, 1, 1)  # [2, batch, 512]
        
        # Expandir a lstm_output_dim (640)
        padding = torch.zeros(2, batch_size, self.lstm_expansion, 
                             device=visual_context.device, dtype=visual_context.dtype)
        h0 = torch.cat([h0, padding], dim=-1)  # [2, batch, 640]
        c0 = torch.zeros_like(h0)
        
        return (h0, c0)



class CorpusCallosum(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        
        # Red de transferencia con bloques residuales
        self.transfer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        self.residual_scale = nn.Parameter(torch.tensor(0.85))
        
        # Modulador de flujo basado en contenido
        self.flow_modulator = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.residual_scale_base = nn.Parameter(torch.tensor(0.85))
        
        # Multi-head attention
        self.flow_attention = nn.MultiheadAttention(  # CORREGIDO: MultiheadAttention → MultiheadAttention
            embed_dim=dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, right_features):
        # right_features: [batch, dim]
        x = right_features.unsqueeze(1)  # [batch, 1, dim]
        
        # Modulación dinámica basada en contenido
        flow_strength = self.flow_modulator(right_features)  # [batch, 1]
        dynamic_scale = self.residual_scale_base * (0.7 + 0.6 * flow_strength.squeeze(-1))  # [batch]
        
        # Attention sobre features
        attn_out, _ = self.flow_attention(x, x, x)  # [batch, 1, dim]
        attn_out = attn_out.squeeze(1)  # [batch, dim]
        
        # Bloques residuales
        for block in self.transfer:
            attn_out = attn_out + block(attn_out)  # [batch, dim]
        

        output = attn_out + dynamic_scale.unsqueeze(-1) * right_features  # [batch, dim]
        
        return output



class NeuroLogosBicameralStable(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.right_hemisphere = RightHemisphere(output_dim=512)
        self.left_hemisphere = LeftHemisphere(vocab_size, embed_dim=256, hidden_dim=512)
        self.corpus_callosum = CorpusCallosum(dim=512)
        
    def forward(self, image, captions=None, epoch=0):
        visual_features, right_post, right_pre = self.right_hemisphere(image)
        visual_context = self.corpus_callosum(visual_features)
        
        if captions is not None:
            logits, gate = self.left_hemisphere(visual_context, captions, epoch=epoch)
            return logits, visual_features, visual_context, gate, right_post, right_pre
        else:
            output = self.left_hemisphere(visual_context, captions, epoch=epoch)
            return output

# =============================================================================
# DIAGNOSTICO MEJORADO CON MÉTRICAS LINGÜÍSTICAS
# =============================================================================
class EnhancedDiagnostics:
    def __init__(self):
        self.history = {
            'loss': [], 'right_metabolism': [], 'right_fatigue': [],
            'right_liquid_norm': [], 'right_homeostasis': [],
            'callosal_flow': [], 'left_gate_mean': [], 'left_gate_std': [],
            'synergy_score': [], 'health_score': [],
            'bleu_score': [], 'token_accuracy': [], 'word_overlap': [],
            'cider_score': [], 'spice_score': [], 'linguistic_reward': []
        }
        self.language_metrics = LanguageMetrics()
    
    def measure_callosal_flow(self, right_features, left_context):
        with torch.no_grad():
            right_norm = F.normalize(right_features, dim=-1)
            left_norm = F.normalize(left_context, dim=-1)
            correlation = (right_norm * left_norm).sum(dim=-1).mean()
            flow_std = left_context.std(dim=-1).mean()
            flow = correlation.item() * min(1.0, flow_std.item() / 0.5)
            return flow
    
    def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std):
        right_health = float(right_node.metabolism) * float(right_node.homeostasis) * (1.0 - float(right_node.fatigue) * 0.5)
        callosal_health = callosal_flow
        gate_balance = 1.0 - abs(left_gate_mean - 0.5) * 2.0
        gate_diversity = min(1.0, left_gate_std * 5.0)
        left_health = 0.7 * gate_balance + 0.3 * gate_diversity
        synergy = (0.35 * right_health + 0.30 * callosal_health + 0.35 * left_health)
        return synergy
    
    def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm):
        health = 0
        if liquid_norm < 2.0: health += 1
        if float(right_node.homeostasis) > 0.7: health += 1
        if callosal_flow > 0.4: health += 1
        if 0.4 < left_gate_mean < 0.6 and left_gate_std > 0.05: health += 1
        if float(right_node.fatigue) < 0.3 and float(right_node.metabolism) > 0.55: health += 1
        return health
    
    def update(self, **metrics):
        for key, value in metrics.items():
            if key in self.history and value is not None:
                self.history[key].append(value)
    
    def get_recent_avg(self, key, n=50):
        if key in self.history and len(self.history[key]) > 0:
            return np.mean(self.history[key][-n:])
        return 0.0
    
    def report(self, epoch):
        if len(self.history['loss']) == 0:
            return
        
        print(f"\n{'='*80}")
        print(f"📊 REPORTE COMPLETO - Época {epoch}")
        print(f"{'='*80}")
        
        # Métricas de loss
        loss = self.get_recent_avg('loss')
        print(f"\n📉 ENTRENAMIENTO:")
        print(f"  Loss: {loss:.4f}")
        
        # Métricas lingüísticas
        bleu = self.get_recent_avg('bleu_score')
        acc = self.get_recent_avg('token_accuracy')
        overlap = self.get_recent_avg('word_overlap')
        cider = self.get_recent_avg('cider_score')
        spice = self.get_recent_avg('spice_score')
        reward = self.get_recent_avg('linguistic_reward')
        
        print(f"\n📝 CALIDAD LINGÜÍSTICA:")
        print(f"  BLEU-4:     {bleu:.4f}", end=" ")
        print("🟢" if bleu > 0.15 else "🟡" if bleu > 0.08 else "🔴")
        
        print(f"  Accuracy:   {acc:.4f}", end=" ")
        print("🟢" if acc > 0.30 else "🟡" if acc > 0.15 else "🔴")
        
        print(f"  W-Overlap:  {overlap:.4f}", end=" ")
        print("🟢" if overlap > 0.35 else "🟡" if overlap > 0.20 else "🔴")
        
        print(f"  CIDEr:      {cider:.4f}", end=" ")
        print("🟢" if cider > 0.15 else "🟡" if cider > 0.08 else "🔴")
        
        print(f"  SPICE:      {spice:.4f}", end=" ")
        print("🟢" if spice > 0.20 else "🟡" if spice > 0.10 else "🔴")
        
        print(f"  Reward:     {reward:.4f}", end=" ")
        print("🟢" if reward > 0.30 else "🟡" if reward > 0.15 else "🔴")
        
        # Fisiología
        print(f"\n🧬 FISIOLOGÍA:")
        metab = self.get_recent_avg('right_metabolism')
        fatigue = self.get_recent_avg('right_fatigue')
        liquid = self.get_recent_avg('right_liquid_norm')
        homeo = self.get_recent_avg('right_homeostasis')
        
        print(f"  Liquid Norm:  {liquid:.3f}", end=" ")
        status = "🟢" if liquid < 2.0 else "🟡" if liquid < 4.0 else "🔴"
        print(status)
        
        print(f"  Homeostasis:  {homeo:.3f}", end=" ")
        print("🟢" if homeo > 0.8 else "🟡" if homeo > 0.6 else "🔴")
        
        print(f"  Metabolismo:  {metab:.3f}")
        print(f"  Fatiga:       {fatigue:.3f}")
        
        # Comunicación
        flow = self.get_recent_avg('callosal_flow')
        gate_mean = self.get_recent_avg('left_gate_mean')
        gate_std = self.get_recent_avg('left_gate_std')
        
        print(f"\n🔗 COMUNICACIÓN:")
        print(f"  Callosum:   {flow:.3f}", end=" ")
        print("🟢" if flow > 0.5 else "🟡" if flow > 0.3 else "🔴")
        
        print(f"  Gate Mean:  {gate_mean:.3f}", end=" ")
        print("🟢" if 0.4 < gate_mean < 0.6 else "🟡")
        
        print(f"  Gate Std:   {gate_std:.3f}")
        
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



class EpisodicMemoryBuffer:
    def __init__(self, capacity=500, surprise_threshold=0.3):
        self.capacity = capacity
        self.surprise_threshold = surprise_threshold
        self.buffer = []
        self.surprise_scores = []
        
    def compute_surprise(self, predicted_logits, ground_truth, gate_mean):
        with torch.no_grad():
            ce = F.cross_entropy(
                predicted_logits.reshape(-1, predicted_logits.size(-1)),
                ground_truth.reshape(-1),
                reduction='none'
            ).mean()
            
            surprise = ce * (1.0 - gate_mean)
            return surprise.item()
    
    def add(self, image, caption, surprise_score):
        if surprise_score > self.surprise_threshold:
            if len(self.buffer) >= self.capacity:
                min_idx = np.argmin(self.surprise_scores)
                self.buffer.pop(min_idx)
                self.surprise_scores.pop(min_idx)
            
            self.buffer.append((image, caption))
            self.surprise_scores.append(surprise_score)
    
    def sample(self, batch_size):
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
                words = parts[1].lower().split()
                counter.update(words)
    
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
        print("✓ Flickr8k ya existe\n")
        return flickr_dir
    
    os.makedirs(flickr_dir, exist_ok=True)
    
    print("📥 Descargando Flickr8k desde GitHub...")
    print("   Tamaño: ~1GB | Tiempo estimado: 2-3 minutos\n")
    
    import urllib.request
    import zipfile
    
    urls = {
        'images': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
        'captions': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
    }
    
    for name, url in urls.items():
        zip_path = os.path.join(flickr_dir, f'{name}.zip')
        
        print(f"📥 Descargando {name}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print(f"📂 Extrayendo {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(flickr_dir)
        
        os.remove(zip_path)
        print(f"✓ {name} completado\n")
    
    # Procesar captions
    print("📝 Procesando captions...")
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
        
        print(f"✓ Captions procesados: {len(captions_dict)} imágenes\n")
    
    # Verificar estructura
    if os.path.exists(os.path.join(flickr_dir, 'Flicker8k_Dataset')):
        import shutil
        old_dir = os.path.join(flickr_dir, 'Flicker8k_Dataset')
        if not os.path.exists(images_dir):
            shutil.move(old_dir, images_dir)
    
    print("✅ Flickr8k listo\n")
    return flickr_dir

# =============================================================================
# TRAINING CON MÉTRICAS Y MEDICINA
# =============================================================================
def train_with_metrics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"NeuroLogos v3.5 | Métricas + Medicina + Sistema Cognitivo | Device: {device}")
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
    
    # CORRECCIÓN: Ajustar configuración del DataLoader para evitar errores de multiprocesamiento
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        # Configuración para Colab
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False)
        print("Usando configuración optimizada para Colab (sin multiprocesamiento)")
    else:
        # Configuración para entornos locales
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        print("Usando configuración estándar con multiprocesamiento")
    
    model = NeuroLogosBicameralStable(len(vocab)).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.right_hemisphere.parameters(), 'lr': 3e-4, 'weight_decay': 1e-5},
        {'params': model.corpus_callosum.parameters(), 'lr': 5e-4, 'weight_decay': 1e-5},
        {'params': model.left_hemisphere.parameters(), 'lr': 2e-4, 'weight_decay': 1e-5}
    ])
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=3
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=27, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[3]
    )
    
    diagnostics = EnhancedDiagnostics()
    medical_system = TriangulatedMedicalSystem()
    cognitive_system = NeurocognitiveSystem()  # Nuevo sistema cognitivo
    episodic_memory = EpisodicMemoryBuffer(capacity=500)

    print(f"🧠 Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Métricas: BLEU-4 + Token Acc + Word Overlap + CIDEr + SPICE")
    print(f"💊 Sistema Médico: 3 niveles de intervención")
    print(f"🧠 Sistema Cognitivo: Mejora basada en retroalimentación lingüística\n")
    
    os.makedirs('./checkpoints', exist_ok=True)
    
    # MEJORA: Añadir variable para controlar frecuencia de cálculo de recompensa lingüística
    reward_frequency = 100  # Reducir frecuencia para optimizar tiempo
    
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
        
        # Diagnóstico cognitivo (solo después de algunas épocas)
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
                cognitive_system.apply_cognitive_intervention(model, cog_issues, cog_severity, cog_confidence, epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Health: {health_score}/5 | Med: {medicine_level} | Cog: {cognitive_level}]")

        for batch_idx, (images, captions, raw_captions) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits, visual_features, visual_context, gate, right_post, right_pre = model(images, captions, epoch=epoch)
            
            # Calcular recompensa lingüística con menor frecuencia para optimizar
            linguistic_reward = None
            if batch_idx % reward_frequency == 0 and epoch >= 2:
                # Generar algunas muestras para evaluar
                with torch.no_grad():
                    sample_indices = np.random.choice(images.size(0), size=min(4, images.size(0)), replace=False)
                    references = [raw_captions[i] for i in sample_indices]
                    
                    # Generar hipótesis
                    sample_images = images[sample_indices]
                    generated = model(sample_images, captions=None, epoch=epoch)
                    
                    hypotheses = []
                    for i in range(generated.size(0)):
                        gen_words = [id2word.get(int(t.item()), '<UNK>') for t in generated[i]]
                        gen_sentence = " ".join(w for w in gen_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                        hypotheses.append(gen_sentence)
                    
                    # Calcular recompensa lingüística
                    linguistic_reward = cognitive_system.linguistic_feedback.compute_linguistic_reward(references, hypotheses)
            
            loss, ce_loss, gate_penalty, diversity_penalty, linguistic_loss = compute_loss(
                logits, captions, gate, vocab, linguistic_reward
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Actualización episódica
            if batch_idx % 20 == 0:
                surprise = episodic_memory.compute_surprise(
                    logits, captions[:, 1:], gate.mean()
                )
                
                for i in range(images.size(0)):
                    episodic_memory.add(
                        images[i].cpu(),
                        captions[i].cpu(),
                        surprise
                    )
            
            # Replay episódico
            if batch_idx % 5 == 0 and len(episodic_memory.buffer) > 32:
                replay_samples = episodic_memory.sample(16)
                if replay_samples:
                    replay_imgs = torch.stack([s[0] for s in replay_samples]).to(device)
                    replay_caps = torch.stack([s[1] for s in replay_samples]).to(device)
                    
                    logits_replay, _, _, gate_replay, post_replay, pre_replay = model(replay_imgs, replay_caps, epoch=epoch)
                    loss_replay, ce_replay, _, _, _ = compute_loss(logits_replay, replay_caps, gate_replay, vocab)
                    
                    (0.5 * loss_replay).backward()
                    
                    model.right_hemisphere.spatial_liquid.hebbian_update(
                        post_replay, pre_replay, plasticity * 2.0
                    )

            model.right_hemisphere.spatial_liquid.hebbian_update(right_post, right_pre, plasticity)
            model.right_hemisphere.spatial_liquid.update_physiology_advanced(ce_loss.item())
            
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    liquid_norm = model.right_hemisphere.spatial_liquid.W_fast_short.norm().item()
                    callosal_flow = diagnostics.measure_callosal_flow(visual_features, visual_context)
                    gate_mean_val = gate.mean().item()
                    gate_std_val = gate.std().item()
                    
                    synergy = diagnostics.calculate_synergy(right_node, callosal_flow, gate_mean_val, gate_std_val)
                    
                    # Actualizar diagnósticos con nuevas métricas
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
                        linguistic_reward=linguistic_reward.item() if linguistic_reward is not None else None
                    )
            
            total_loss += ce_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{ce_loss.item():.3f}',
                'g_pen': f'{gate_penalty.item():.3f}',
                'liquid': f'{liquid_norm:.2f}',
                'gate': f'{gate_mean_val:.2f}',
                'reward': f'{linguistic_reward.item():.3f}' if linguistic_reward is not None else 'N/A'
            })
        
        scheduler.step()
        
        if epoch % 2 == 0:
            model.eval()
            print("\n📸 EVALUACIÓN LINGÜÍSTICA...\n")
            
            bleu_scores = []
            acc_scores = []
            overlap_scores = []
            cider_scores = []
            spice_scores = []
            
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
            
            # Actualizar diagnósticos con nuevas métricas
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
        
        print(f"Época {epoch:02d} | Loss: {avg_loss:.4f} | Health: {health_score}/5 | Med: {medicine_level} | Cog: {cognitive_level}")
        print(f"BLEU: {bleu_avg:.3f} | CIDEr: {cider_avg:.3f} | Gate: {gate_mean:.3f} | Liquid: {liquid:.2f}\n")
        
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'id2word': id2word,
                'diagnostics': diagnostics.history,
                'medical_interventions': medical_system.intervention_history,
                'cognitive_interventions': cognitive_system.cognitive_history
            }, f'./checkpoints/stable_epoch_{epoch:02d}.pth')
    
    print("✅ Entrenamiento completado!")
    diagnostics.report(29)
    
    print(f"\n{'='*80}")
    print("📋 HISTORIAL DE INTERVENCIONES")
    print(f"{'='*80}")
    
    print("\n🏥 INTERVENCIONES MÉDICAS:")
    if len(medical_system.intervention_history) == 0:
        print("✓ No se requirieron intervenciones médicas - Sistema saludable\n")
    else:
        for intervention in medical_system.intervention_history:
            print(f"Época {intervention['epoch']:02d} | {intervention['level']} | Severidad: {intervention['severity']}/12")
            print(f"  Issues: {', '.join(intervention['issues'])}")
            print(f"  Aplicadas: {len(intervention['interventions'])} intervenciones")
            for inter in intervention['interventions']:
                print(f"    • {inter}")
            print()
    
    print("\n🧠 INTERVENCIONES COGNITIVAS:")
    if len(cognitive_system.cognitive_history) == 0:
        print("✓ No se requirieron intervenciones cognitivas - Sistema lingüístico saludable\n")
    else:
        for intervention in cognitive_system.cognitive_history:
            if intervention['severity'] > 0:
                print(f"Época {intervention['epoch']:02d} | Severidad: {intervention['severity']}/9")
                print(f"  Issues: {', '.join(intervention['issues'])}")
                print()


if __name__ == "__main__":
    train_with_metrics()
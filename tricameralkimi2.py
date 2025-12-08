# =============================================================================
# NeuroLogos TRICAMERAL v5.2 [FUSIÓN OPTIMIZADA]
# Hemisferio Derecho: Visión + Audio
# Hemisferio Izquierdo: Lenguaje + Razonamiento
# Corpus Callosum: Fusión trimodal (ve, escucha, razona)
# + Métricas lingüísticas + Sistema médico + Sistema cognitivo + Memoria episódica
# =============================================================================

import os
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings
import kagglehub
warnings.filterwarnings('ignore')
import shutil
# =============================================================================
# UTILIDADES: Descarga y setup del dataset ROBUSTO Y RESUMIBLE
# =============================================================================
def setup_flickr8k_with_audio(data_dir='./flickr8k_full'):
    """
    Descarga y organiza Flickr8k + Audio del dataset de Kaggle.
    Sistema robusto que verifica componentes individuales y descarga solo lo faltante.
    """
    images_dir = os.path.join(data_dir, 'Images')
    audio_dir  = os.path.join(data_dir, 'wavs')
    captions_file = os.path.join(data_dir, 'captions.txt')
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Verificar estado actual
    has_images = os.path.exists(images_dir) and len(os.listdir(images_dir)) >= 8000
    has_audio = os.path.exists(audio_dir) and len(os.listdir(audio_dir)) >= 1000
    has_captions = os.path.exists(captions_file)
    
    print(f"📊 Estado del dataset:")
    print(f"  Imágenes: {'✅' if has_images else '❌'} ({len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0}/8091)")
    print(f"  Audios: {'✅' if has_audio else '❌'} ({len(os.listdir(audio_dir)) if os.path.exists(audio_dir) else 0}/40000)")
    print(f"  Captions: {'✅' if has_captions else '❌'}")
    
    # Si todo está completo, retornar
    if has_images and has_audio and has_captions:
        print("✅ Dataset completo, sin necesidad de descargas\n")
        return data_dir
    
    # ========== AUDIO ==========
    if not has_audio:
        print("\n📥 Descargando y procesando audios desde Kaggle...")
        try:
            path = kagglehub.dataset_download("warcoder/flickr-8k-audio-caption-corpus")
            print(f"✓ Dataset descargado en: {path}")
            
            # Buscar carpeta wavs o archivos .wav
            wavs_src = Path(path) / "wavs"
            all_wavs = []
            
            if wavs_src.exists() and wavs_src.is_dir():
                all_wavs = list(wavs_src.glob("*.wav"))
            else:
                # Buscar recursivamente
                all_wavs = list(Path(path).rglob("*.wav"))
            
            if not all_wavs:
                print("⚠️  No se encontraron archivos .wav en el dataset")
            else:
                # Crear directorio limpio
                if os.path.exists(audio_dir):
                    shutil.rmtree(audio_dir)
                os.makedirs(audio_dir, exist_ok=True)
                
                # Copiar audios
                print(f"📂 Copiando {len(all_wavs)} archivos de audio...")
                for wav_file in tqdm(all_wavs, desc="Copiando audios"):
                    dest_path = os.path.join(audio_dir, wav_file.name)
                    if not os.path.exists(dest_path):
                        shutil.copy2(str(wav_file), dest_path)
                
                print(f"✅ Audios listos: {len(os.listdir(audio_dir))} archivos")
        
        except Exception as e:
            print(f"❌ Error descargando audios: {e}")
            print("⚠️  Continuando sin audios...")
    
    # ========== IMÁGENES ==========
    if not has_images:
        print("\n📥 Descargando imágenes originales...")
        try:
            import urllib.request, zipfile
            images_url = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
            zip_path = os.path.join(data_dir, 'images.zip')
            
            print("  Descargando archivo ZIP...")
            urllib.request.urlretrieve(images_url, zip_path)
            
            print("  Extrayendo imágenes...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)
            
            os.remove(zip_path)
            
            # Normalizar nombre de carpeta
            possible_names = ['Flicker8k_Dataset', 'Flickr8k_Dataset']
            for old_name in possible_names:
                old_path = os.path.join(data_dir, old_name)
                if os.path.exists(old_path) and not os.path.exists(images_dir):
                    os.rename(old_path, images_dir)
                    break
            
            print(f"✅ Imágenes listas: {len(os.listdir(images_dir))} archivos")
        
        except Exception as e:
            print(f"❌ Error descargando imágenes: {e}")
            raise RuntimeError("No se pudieron descargar las imágenes. Verifica tu conexión.")
    
    # ========== CAPTIONS ==========
    if not has_captions:
        print("\n📥 Descargando captions...")
        try:
            import urllib.request, zipfile
            caps_url = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
            zip_path = os.path.join(data_dir, 'captions.zip')
            
            print("  Descargando archivo ZIP...")
            urllib.request.urlretrieve(caps_url, zip_path)
            
            print("  Extrayendo captions...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)
            
            os.remove(zip_path)
            
            # Generar captions.txt desde Flickr8k.token.txt
            token_file = os.path.join(data_dir, 'Flickr8k.token.txt')
            if os.path.exists(token_file):
                print("  Procesando captions...")
                with open(token_file, 'r', encoding='utf-8') as fin, \
                     open(captions_file, 'w', encoding='utf-8') as fout:
                    for line in fin:
                        if '\t' in line:
                            img_cap, text = line.strip().split('\t', 1)
                            img_name = img_cap.split('#')[0]
                            fout.write(f"{img_name}\t{text}\n")
                
                print(f"✅ Captions listos")
            else:
                raise FileNotFoundError("No se encontró Flickr8k.token.txt")
        
        except Exception as e:
            print(f"❌ Error descargando captions: {e}")
            raise RuntimeError("No se pudieron descargar los captions. Verifica tu conexión.")
    
    # Reporte final
    print(f"\n{'='*60}")
    print(f"✅ Dataset preparado en: {data_dir}")
    print(f"   Imágenes: {len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0}")
    print(f"   Audios: {len(os.listdir(audio_dir)) if os.path.exists(audio_dir) else 0}")
    print(f"   Captions: {'Sí' if os.path.exists(captions_file) else 'No'}")
    print(f"{'='*60}\n")
    
    return data_dir


def build_vocab_flickr(captions_file, vocab_size=5000):
    """Construye vocabulario desde el archivo de captions"""
    print("📚 Construyendo vocabulario...")
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
    print(f"✓ Vocabulario creado: {len(vocab)} palabras")
    return vocab, id2word

# =============================================================================
# MEMORIA EPISÓDICA
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
            
            self.buffer.append((image.detach().cpu(), caption.detach().cpu()))
            self.surprise_scores.append(surprise_score)
    
    def sample(self, batch_size):
        """Samplea ejemplos con probabilidad proporcional a sorpresa"""
        if not self.buffer:
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
    
    def __init__(self, alpha=0.6, beta=0.4):
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
# NEURONA LÍQUIDA ESTABLE (MEJORADA)
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
# ENCODER DE AUDIO
# =============================================================================
class AudioEncoder(nn.Module):
    """Encoder de audio usando Conv + Transformer"""
    
    def __init__(self, output_dim=512):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        self.projection = nn.Linear(512, output_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, mel_spec):
        """
        Args:
            mel_spec: (batch, 80, time)
        Returns:
            audio_features: (batch, output_dim)
        """
        x = self.conv_layers(mel_spec)
        x = x.transpose(1, 2)
        x = self.temporal_encoder(x)
        x = x.transpose(1, 2)
        x = self.adaptive_pool(x).squeeze(-1)
        out = self.projection(x)
        
        return out

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
            # Referencias correctas a las neuronas líquidas del hemisferio derecho
            visual_node = model.right_hemisphere.visual_liquid
            audio_node  = model.right_hemisphere.audio_liquid

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
                self._reset_liquid_neuron(visual_node, severity)
                self._reset_liquid_neuron(audio_node,  severity)
                interventions_applied.append("liquid_full_reset")
            elif "liquid_elevated" in issues:
                visual_node.W_fast_short *= 0.4
                visual_node.W_fast_long  *= 0.4
                audio_node.W_fast_short  *= 0.4
                audio_node.W_fast_long   *= 0.4
                interventions_applied.append("liquid_reduce")

            # Mantenimiento
            if severity >= 3:
                visual_node.fatigue *= 0.5
                audio_node.fatigue  *= 0.5
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
# HEMISFERIO DERECHO TRICAMERAL
# =============================================================================
class RightHemisphereTricameral(nn.Module):
    """Hemisferio derecho con canales visual y auditivo"""
    
    def __init__(self, output_dim=512):
        super().__init__()
        
        # Canal visual (ResNet50)
        resnet = models.resnet50(pretrained=True)
        for param in list(resnet.parameters())[:-20]:
            param.requires_grad = False
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Canal auditivo
        self.audio_encoder = AudioEncoder(output_dim=output_dim)
        
        # Neurona líquida para visión
        self.visual_liquid = StableLiquidNeuron(2048, output_dim)
        
        # Neurona líquida para audio
        self.audio_liquid = StableLiquidNeuron(output_dim, output_dim)
        
        # Fusión cross-modal
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, image, audio):
        """
        Args:
            image: (B, 3, H, W)
            audio: (B, 80, T)
        Returns:
            fused_features: (B, output_dim)
            visual_post, visual_pre, audio_post, audio_pre: Para Hebbian
        """
        # Procesamiento visual
        visual_raw = self.visual_encoder(image).flatten(1)
        visual_out, visual_post, visual_pre = self.visual_liquid(visual_raw)
        
        # Procesamiento auditivo
        audio_raw = self.audio_encoder(audio)
        audio_out, audio_post, audio_pre = self.audio_liquid(audio_raw)
        
        # Atención cruzada
        visual_expanded = visual_out.unsqueeze(1)
        audio_expanded = audio_out.unsqueeze(1)
        
        attended_visual, _ = self.cross_modal_attention(
            visual_expanded, audio_expanded, audio_expanded
        )
        attended_visual = attended_visual.squeeze(1)
        
        # Fusión
        fused = torch.cat([attended_visual, audio_out], dim=-1)
        fused_features = self.fusion(fused)
        
        return fused_features, visual_post, visual_pre, audio_post, audio_pre

# =============================================================================
# CORPUS CALLOSUM TRIMODAL
# =============================================================================
class CorpusCallosumTrimodal(nn.Module):
    """Corpus callosum con canales: visual, auditivo, semántico"""
    
    def __init__(self, dim=512):
        super().__init__()
        
        self.visual_dim = dim // 3
        self.audio_dim = dim // 3
        self.semantic_dim = dim - 2 * (dim // 3)
        
        self.visual_proj = nn.Linear(dim, self.visual_dim)
        self.audio_proj = nn.Linear(dim, self.audio_dim)
        self.semantic_proj = nn.Linear(dim, self.semantic_dim)
        
        self.trimodal_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.transfer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        self.residual_scale_base = nn.Parameter(torch.tensor(0.85))
        
        # Fatiga por canal
        self.register_buffer('visual_fatigue', torch.tensor(0.0))
        self.register_buffer('audio_fatigue', torch.tensor(0.0))
        self.register_buffer('semantic_fatigue', torch.tensor(0.0))
        self.fatigue_decay = 0.95
        self.fatigue_recovery = 0.01
    
    def forward(self, right_features):
        """
        Args:
            right_features: (B, dim) - Fusión de visión + audio
        Returns:
            enriched_context: (B, dim)
            channels: dict con 'visual', 'audio', 'semantic'
        """
        # Separación de canales
        visual_channel = self.visual_proj(right_features)
        audio_channel = self.audio_proj(right_features)
        semantic_channel = self.semantic_proj(right_features)
        
        # Aplicar fatiga
        visual_gated = visual_channel * torch.sigmoid(1.0 - self.visual_fatigue)
        audio_gated = audio_channel * torch.sigmoid(1.0 - self.audio_fatigue)
        semantic_gated = semantic_channel * torch.sigmoid(1.0 - self.semantic_fatigue)
        
        # Reconstruir tensor completo
        structured = torch.cat([visual_gated, audio_gated, semantic_gated], dim=-1)
        
        # Atención trimodal
        structured_expanded = structured.unsqueeze(1)
        attn_out, _ = self.trimodal_attention(
            structured_expanded, structured_expanded, structured_expanded
        )
        attn_out = attn_out.squeeze(1)
        
        # Transfer blocks
        for block in self.transfer:
            attn_out = attn_out + block(attn_out)
        
        # Residual con escala dinámica
        flow_strength = torch.sigmoid(structured.mean(dim=-1))
        dynamic_scale = self.residual_scale_base * (0.7 + 0.6 * flow_strength)
        
        output = attn_out + dynamic_scale.unsqueeze(-1) * right_features
        
        # Crear versiones completas para compatibilidad
        visual_full = torch.cat([
            visual_gated,
            torch.zeros(visual_gated.size(0), self.audio_dim + self.semantic_dim, device=visual_gated.device)
        ], dim=-1)
        
        audio_full = torch.cat([
            torch.zeros(audio_gated.size(0), self.visual_dim, device=audio_gated.device),
            audio_gated,
            torch.zeros(audio_gated.size(0), self.semantic_dim, device=audio_gated.device)
        ], dim=-1)
        
        semantic_full = torch.cat([
            torch.zeros(semantic_gated.size(0), self.visual_dim + self.audio_dim, device=semantic_gated.device),
            semantic_gated
        ], dim=-1)
        
        # Actualizar fatiga
        self.update_channel_fatigue(visual_channel, audio_channel, semantic_channel)
        
        return output, {
            'visual': visual_full,
            'audio': audio_full,
            'semantic': semantic_full,
            'fatigue': {
                'visual': self.visual_fatigue.item(),
                'audio': self.audio_fatigue.item(),
                'semantic': self.semantic_fatigue.item()
            }
        }
    
    def update_channel_fatigue(self, visual_channel, audio_channel, semantic_channel):
        """Actualiza fatiga específica por canal"""
        with torch.no_grad():
            visual_activity = visual_channel.norm(dim=-1).mean()
            audio_activity = audio_channel.norm(dim=-1).mean()
            semantic_activity = semantic_channel.norm(dim=-1).mean()
            
            self.visual_fatigue = self.visual_fatigue * self.fatigue_decay + 0.01 * visual_activity
            self.audio_fatigue = self.audio_fatigue * self.fatigue_decay + 0.01 * audio_activity
            self.semantic_fatigue = self.semantic_fatigue * self.fatigue_decay + 0.01 * semantic_activity
            
            self.visual_fatigue = max(0.0, self.visual_fatigue - self.fatigue_recovery)
            self.audio_fatigue = max(0.0, self.audio_fatigue - self.fatigue_recovery)
            self.semantic_fatigue = max(0.0, self.semantic_fatigue - self.fatigue_recovery)
            
            self.visual_fatigue = torch.clamp(self.visual_fatigue, 0.0, 1.0)
            self.audio_fatigue = torch.clamp(self.audio_fatigue, 0.0, 1.0)
            self.semantic_fatigue = torch.clamp(self.semantic_fatigue, 0.0, 1.0)
    
    def adjust_gates_by_fatigue(self):
        """Ajusta proyecciones basado en fatiga"""
        with torch.no_grad():
            self.visual_proj.weight.data *= (1.0 - 0.05 * self.visual_fatigue)
            self.audio_proj.weight.data *= (1.0 - 0.05 * self.audio_fatigue)
            self.semantic_proj.weight.data *= (1.0 - 0.05 * self.semantic_fatigue)

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

        # profundidades válidas según cabezales existentes
        prediction_depths = list(range(1, len(self.mtp_output_heads) + 1))
        valid_predictions = 0

        for depth in prediction_depths:
            if seq_len <= depth:
                continue

            valid_seq_len = seq_len - depth
            if valid_seq_len <= 0:
                continue

            predict_hidden = hidden_states[:, :valid_seq_len, :]
            target_ids     = input_ids[:, depth:depth + valid_seq_len]

            if target_ids.size(1) == 0:
                continue

            depth_logits = self.mtp_output_heads[depth - 1](predict_hidden)

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
        """
        Atenuación simple según fatiga de canal + atención cruzada visual.
        Se ignoran los canales 'objects/actions/scene' que no existen.
        """
        seq_len = lstm_out.size(1)

        # Fatiga agregada (promedio de los tres canales)
        fatigue = (
            channels['fatigue']['visual']   * 0.4 +
            channels['fatigue']['audio']    * 0.4 +
            channels['fatigue']['semantic'] * 0.2
        )
        fatigue_gate = torch.sigmoid(torch.tensor(1.0 - fatigue, device=lstm_out.device))

        # Proyección visual original (como query)
        visual_query = visual_context.unsqueeze(1).expand(-1, seq_len, -1)

        # Atención cruzada
        attended, _ = self.visual_attention(lstm_out, visual_query, visual_query)

        # Salida = entrada + atención ponderada por fatiga
        out = lstm_out + 0.5 * fatigue_gate * attended
        return out   



    def _get_init_state(self, visual_context):
        batch_size = visual_context.size(0)
        h0 = visual_context.unsqueeze(0).repeat(2, 1, 1)
        
        padding = torch.zeros(2, batch_size, self.lstm_expansion, device=visual_context.device)
        h0 = torch.cat([h0, padding], dim=-1)
        c0 = torch.zeros_like(h0)
        
        return (h0, c0)



# =============================================================================
# MODELO TRICAMERAL COMPLETO
# =============================================================================
class NeuroLogosTricameral(nn.Module):
    """Arquitectura completa: Visión + Audio -> Lenguaje"""
    
    def __init__(self, vocab_size):
        super().__init__()
        
        # Hemisferio derecho (visión + audio)
        self.right_hemisphere = RightHemisphereTricameral(output_dim=512)
        
        # Corpus callosum trimodal
        self.corpus_callosum = CorpusCallosumTrimodal(dim=512)
        
        # Hemisferio izquierdo (lenguaje + razonamiento)
        self.left_hemisphere = LeftHemisphere(vocab_size, embed_dim=256, hidden_dim=512)
    
    def forward(self, image, audio, captions=None, epoch=0):
        """
        Args:
            image: (B, 3, H, W)
            audio: (B, 80, T) - Mel-spectrogram del caption
            captions: (B, seq_len) - Tokens (solo en train)
        
        Returns (training):
            logits, gates, losses, posts, pres, channels
        
        Returns (inference):
            generated_text
        """
        # Procesamiento derecho (visión + audio)
        fused_features, vis_post, vis_pre, aud_post, aud_pre = self.right_hemisphere(image, audio)
        
        # Corpus callosum (enriquecimiento trimodal)
        enriched_context, channels = self.corpus_callosum(fused_features)
        
        if captions is not None:
            # Training: generar logits
            logits, gate, mtp_loss, reasoning_steps = self.left_hemisphere(
                enriched_context, captions, channels, epoch=epoch
            )
            
            return (
                logits, fused_features, enriched_context, gate, 
                vis_post, vis_pre, aud_post, aud_pre, 
                channels, mtp_loss, reasoning_steps
            )
        else:
            # Inference: generar texto
            generated_text = self.left_hemisphere(
                enriched_context, captions=None, channels=channels, epoch=epoch
            )
            
            return generated_text, None

# =============================================================================
# DATASET MULTIMODAL
# =============================================================================
class Flickr8kMultimodalDataset(Dataset):
    """Dataset que carga imagen, audio del caption y texto desde Kaggle"""
    
    def __init__(self, images_dir, audio_dir, captions_file, vocab, 
                 img_transform=None, max_len=30, sample_rate=16000):
        self.images_dir = images_dir
        self.audio_dir = audio_dir
        self.vocab = vocab
        self.max_len = max_len
        self.sample_rate = sample_rate
        self.img_transform = img_transform
        
        # Procesador de audio (alineado con dataset: 16kHz)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80,
            f_min=0,
            f_max=8000
        )
        
        self.data = []
        
        # Mapeo: image_name -> list of audio files
        audio_files = {f.stem: f for f in Path(audio_dir).glob("*.wav")}
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, caption = parts
                    img_path = os.path.join(images_dir, img_name)
                    
                    # Buscar audio correspondiente
                    stem = Path(img_name).stem
                    audio_keys = [f"{stem}_{i}" for i in range(5)]
                    
                    for audio_key in audio_keys:
                        if audio_key in audio_files:
                            audio_path = audio_files[audio_key]
                            if os.path.exists(img_path):
                                self.data.append((img_path, str(audio_path), caption))
        
        print(f"✓ Loaded {len(self.data)} multimodal samples")
        print(f"  Formato audio: .wav, 16kHz (del dataset de Kaggle)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, audio_path, caption = self.data[idx]

        # 1. Imagen
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)

        # 2. Audio con soundfile (16 kHz mono)
        import soundfile as sf
        waveform, sr = sf.read(audio_path, dtype='float32')
        if waveform.ndim > 1:               # stereo → mono
            waveform = waveform.mean(axis=-1)
        if sr != self.sample_rate:          # resample si hace falta
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
        waveform = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)

        # 3. Mel-spectrograma 80 × 300
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        if mel_spec.size(-1) < 300:
            pad = 300 - mel_spec.size(-1)
            mel_spec = F.pad(mel_spec, (0, pad))
        else:
            mel_spec = mel_spec[..., :300]

        # 4. Tokens de caption
        tokens = ['<BOS>'] + caption.lower().split() + ['<EOS>']
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]

        return image, mel_spec.squeeze(0), torch.tensor(token_ids, dtype=torch.long), caption
# =============================================================================
# FUNCIÓN DE PÉRDIDA TRICAMERAL
# =============================================================================
def compute_tricameral_loss(logits, captions, gate, vocab, 
                           visual_post, audio_post, 
                           mtp_loss=None, linguistic_reward=None,
                           lambda_reward=0.1, lambda_mtp=0.1):
    """Pérdida con término de coherencia audio-visual"""
    
    # Pérdida estándar
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
    
    # Coherencia audio-visual
    visual_norm = F.normalize(visual_post, dim=-1)
    audio_norm = F.normalize(audio_post, dim=-1)
    coherence_loss = (1 - (visual_norm * audio_norm).sum(dim=-1)).mean()
    
    # Recompensa lingüística
    linguistic_loss = 0.0
    if linguistic_reward is not None and torch.is_tensor(linguistic_reward):
        linguistic_loss = -lambda_reward * linguistic_reward
    
    # MTP
    mtp_term = 0.0
    if mtp_loss is not None and torch.is_tensor(mtp_loss):
        mtp_term = lambda_mtp * mtp_loss
    
    # Total
    total_loss = (
        ce_loss + 
        0.05 * gate_penalty + 
        0.15 * diversity_penalty + 
        0.08 * coherence_loss +
        linguistic_loss + 
        mtp_term
    )
    
    return total_loss, ce_loss, gate_penalty, diversity_penalty, coherence_loss, mtp_term

# =============================================================================
# DIAGNÓSTICO TRICAMERAL
# =============================================================================
class EnhancedDiagnosticsTricameral:
    def __init__(self):
        self.history = {
            'loss': [], 'visual_metabolism': [], 'visual_fatigue': [], 'visual_liquid_norm': [],
            'audio_metabolism': [], 'audio_fatigue': [], 'audio_liquid_norm': [],
            'callosal_flow': [], 'left_gate_mean': [], 'left_gate_std': [],
            'synergy_score': [], 'health_score': [],
            'bleu_score': [], 'token_accuracy': [], 'word_overlap': [],
            'cider_score': [], 'spice_score': [], 'linguistic_reward': [],
            'coherence_loss': [], 'objects_fatigue': [], 'actions_fatigue': [], 'scene_fatigue': [],
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
                visual_norm = F.normalize(channels['visual'], dim=-1)
                audio_norm = F.normalize(channels['audio'], dim=-1)
                semantic_norm = F.normalize(channels['semantic'], dim=-1)
                
                visual_corr = (right_norm * visual_norm).sum(dim=-1).mean()
                audio_corr = (right_norm * audio_norm).sum(dim=-1).mean()
                semantic_corr = (right_norm * semantic_norm).sum(dim=-1).mean()
                
                weighted_correlation = 0.4 * visual_corr + 0.4 * audio_corr + 0.2 * semantic_corr
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
    
    def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std):
        visual_health = float(visual_node.metabolism) * float(visual_node.homeostasis) * (1.0 - float(visual_node.fatigue) * 0.5)
        audio_health = float(audio_node.metabolism) * float(audio_node.homeostasis) * (1.0 - float(audio_node.fatigue) * 0.5)
        callosal_health = float(callosal_flow)
        gate_balance = 1.0 - abs(float(left_gate_mean) - 0.5) * 2.0
        gate_diversity = min(1.0, float(left_gate_std) * 5.0)
        left_health = 0.7 * gate_balance + 0.3 * gate_diversity
        
        synergy = (0.25 * visual_health + 0.25 * audio_health + 0.25 * callosal_health + 0.25 * left_health)
        return float(synergy)
    
    def calculate_health(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm):
        health = 0
        if float(liquid_norm) < 2.0: health += 1
        if float(visual_node.homeostasis) > 0.7: health += 1
        if float(audio_node.homeostasis) > 0.7: health += 1
        if float(callosal_flow) > 0.4: health += 1
        if 0.35 < float(left_gate_mean) < 0.65 and float(left_gate_std) > 0.05: health += 1
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
            visual_fatigue = self.get_recent_avg('visual_fatigue', n=10)
            audio_fatigue = self.get_recent_avg('audio_fatigue', n=10)
            semantic_fatigue = self.get_recent_avg('semantic_fatigue', n=10)
            
            print(f"\n🔗 DISTRIBUCIÓN DE FATIGA - Época {epoch}")
            print(f"  Visual: {visual_fatigue:.3f} {'🔴' if visual_fatigue > 0.3 else '🟡' if visual_fatigue > 0.15 else '🟢'}")
            print(f"  Audio: {audio_fatigue:.3f} {'🔴' if audio_fatigue > 0.3 else '🟡' if audio_fatigue > 0.15 else '🟢'}")
            print(f"  Semántico: {semantic_fatigue:.3f} {'🔴' if semantic_fatigue > 0.3 else '🟡' if semantic_fatigue > 0.15 else '🟢'}")
            
            max_fatigue = max(visual_fatigue, audio_fatigue, semantic_fatigue)
            min_fatigue = min(visual_fatigue, audio_fatigue, semantic_fatigue)
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
        """Genera reporte completo del estado del sistema tricameral"""
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
        visual_liquid = self.get_recent_avg('visual_liquid_norm')
        audio_liquid = self.get_recent_avg('audio_liquid_norm')
        
        print(f"  Visual Norm: {visual_liquid:.3f} {'🟢' if visual_liquid < 2.0 else '🟡' if visual_liquid < 4.0 else '🔴'}")
        print(f"  Audio Norm: {audio_liquid:.3f} {'🟢' if audio_liquid < 2.0 else '🟡' if audio_liquid < 4.0 else '🔴'}")
        
        # Comunicación
        flow = self.get_recent_avg('callosal_flow')
        gate_mean = self.get_recent_avg('left_gate_mean')
        gate_std = self.get_recent_avg('left_gate_std')
        
        print(f"\n🔗 COMUNICACIÓN:")
        print(f"  Callosum: {flow:.3f} {'🟢' if flow > 0.5 else '🟡' if flow > 0.3 else '🔴'}")
        print(f"  Gate Mean: {gate_mean:.3f} {'🟢' if 0.35 < gate_mean < 0.65 else '🟡'}")
        print(f"  Gate Std: {gate_std:.3f}")
        
        # Salud global
        synergy = self.get_recent_avg('synergy_score')
        health = self.get_recent_avg('health_score')
        
        print(f"\n🏛️  SISTEMA TRICAMERAL:")
        print(f"  Sinergia: {synergy:.3f} | Salud: {int(health)}/5", end=" ")
        if health >= 4:
            print("🟢 ÓPTIMO")
        elif health >= 3:
            print("🟡 FUNCIONAL")
        else:
            print("🔴 CRÍTICO")
        
        print(f"{'='*80}\n")

# =============================================================================
# LOOP DE ENTRENAMIENTO PRINCIPAL
# =============================================================================
def train_tricameral():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"NeuroLogos TRICAMERAL v5.2 | Visión + Audio + Lenguaje + Razonamiento | Device: {device}")
    print(f"🎵 Usando dataset pre-generado de Kaggle (40k audios .wav)")
    print(f"🧠 Sistema médico + cognitivo + memoria episódica integrados")
    print(f"{'='*80}\n")
    
    # Setup dataset con audio
    flickr_dir = setup_flickr8k_with_audio('./flickr8k_full')
    images_dir = os.path.join(flickr_dir, 'Images')
    audio_dir = os.path.join(flickr_dir, 'wavs')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    # Verificar audio
    if not os.path.exists(audio_dir) or len(os.listdir(audio_dir)) < 1000:
        print("❌ Error: No se encontraron suficientes audios. Ejecuta setup_flickr8k_with_audio() primero.")
        return
    
    print(f"✅ Audio directory found: {len(os.listdir(audio_dir))} files\n")
    
    # Construir vocabulario
    vocab, id2word = build_vocab_flickr(captions_file, vocab_size=5000)
    
    # Transformaciones de imagen
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset multimodal
    dataset = Flickr8kMultimodalDataset(
        images_dir, audio_dir, captions_file, vocab, 
        img_transform=transform, max_len=30
    )
    
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    # Modelo tricameral
    model = NeuroLogosTricameral(len(vocab)).to(device)
    
    # Optimizador
    optimizer = torch.optim.AdamW([
        {'params': model.right_hemisphere.parameters(), 'lr': 3e-4},
        {'params': model.corpus_callosum.parameters(), 'lr': 5e-4},
        {'params': model.left_hemisphere.parameters(), 'lr': 2e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Sistemas auxiliares
    diagnostics = EnhancedDiagnosticsTricameral()
    medical_system = TriangulatedMedicalSystem()
    cognitive_system = NeurocognitiveSystem()
    episodic_memory = EpisodicMemoryBuffer(capacity=500)
    
    print(f"🧠 Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎵 Audio encoder: AudioEncoder + LiquidNeuron")
    print(f"🔗 Trimodal callosum: Visual + Audio + Semantic")
    print(f"💊 Medical System: Triangulated intervention")
    print(f"🧠 Cognitive System: Reasoning + MTP + Chain-of-Thought")
    print(f"🧠 Episodic Memory: Surprise-weighted replay\n")
    
    # Entrenamiento
    for epoch in range(30):
        model.train()
        total_loss = 0
        
        # Diagnóstico inicial
        visual_node = model.right_hemisphere.visual_liquid
        audio_node = model.right_hemisphere.audio_liquid
        
        liquid = diagnostics.get_recent_avg('visual_liquid_norm')
        flow = diagnostics.get_recent_avg('callosal_flow')
        gate_mean = diagnostics.get_recent_avg('left_gate_mean')
        gate_std = diagnostics.get_recent_avg('left_gate_std')
        health_score = diagnostics.calculate_health(visual_node, audio_node, flow, gate_mean, gate_std, liquid)
        
        issues, severity, confidence = medical_system.diagnose_with_triangulation(
            health_score, liquid, gate_mean, gate_std, flow
        )
        
        medicine_level = "🟢 Nivel 0" if severity == 0 else f"🟡 Nivel 1" if severity <= 2 else f"🟠 Nivel 2" if severity <= 6 else "🔴 Nivel 3"
        
        if severity > 0:
            medical_system.apply_triangulated_intervention(model, issues, severity, confidence, epoch)
        
        # Ajuste de fatiga
        model.corpus_callosum.adjust_gates_by_fatigue()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Health: {health_score}/5 | Med: {medicine_level}]")
        
        for batch_idx, (images, audio_specs, captions, raw_captions) in enumerate(pbar):
            images = images.to(device)
            audio_specs = audio_specs.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            (logits, fused_feat, enriched_ctx, gate, 
             vis_post, vis_pre, aud_post, aud_pre, 
             channels, mtp_loss, reasoning_steps) = model(
                images, audio_specs, captions, epoch=epoch
            )
            
            # Pérdida
            loss, ce_loss, gate_penalty, diversity_penalty, coherence_loss, mtp_term = compute_tricameral_loss(
                logits, captions, gate, vocab, vis_post, aud_post, mtp_loss
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Hebbian updates
            plasticity = 0.1 if epoch < 10 else 0.05
            model.right_hemisphere.visual_liquid.hebbian_update(vis_post, vis_pre, plasticity)
            model.right_hemisphere.audio_liquid.hebbian_update(aud_post, aud_pre, plasticity)
            
            # Update physiology
            model.right_hemisphere.visual_liquid.update_physiology_advanced(ce_loss.item())
            model.right_hemisphere.audio_liquid.update_physiology_advanced(ce_loss.item())
            
            total_loss += ce_loss.item()
            
            # ACTUALIZACIÓN EPISÓDICA
            if batch_idx % 20 == 0:
                surprise = episodic_memory.compute_surprise(logits, captions[:, 1:], gate.mean())
                
                for i in range(images.size(0)):
                    episodic_memory.add(images[i], captions[i], surprise)
                
                if batch_idx % 100 == 0:
                    print(f"  🧠 Episodic Buffer: {len(episodic_memory.buffer)}/{episodic_memory.capacity} samples")
            
            # REPLAY EPISÓDICO
            if batch_idx % 5 == 0 and len(episodic_memory.buffer) > 32:
                replay_samples = episodic_memory.sample(16)
                if replay_samples:
                    replay_imgs = torch.stack([s[0] for s in replay_samples]).to(device)
                    replay_caps = torch.stack([s[1] for s in replay_samples]).to(device)
                    
                    logits_replay, _, _, gate_replay, vis_post_replay, vis_pre_replay, \
                        aud_post_replay, aud_pre_replay, channels_replay, mtp_loss_replay, _ = model(
                            replay_imgs,
                            torch.randn(replay_imgs.size(0), 80, 300).to(device),  # audio dummy
                            replay_caps,
                            epoch=epoch
                        )

                    loss_replay, ce_replay, _, _, _, _ = compute_tricameral_loss(
                        logits_replay, replay_caps, gate_replay, vocab,
                        vis_post_replay, aud_post_replay  # audio_post dummy real
                    )

                    (0.5 * loss_replay).backward()

                    model.right_hemisphere.visual_liquid.hebbian_update(
                        vis_post_replay, vis_pre_replay, plasticity * 2.0
                    )
                    loss_replay, ce_replay, _, _, _, _ = compute_tricameral_loss(
                        logits_replay, replay_caps, gate_replay, vocab, 
                        vis_post_replay, torch.randn_like(vis_post_replay)
                    )
                    
                    (0.5 * loss_replay).backward()
                    
                    model.right_hemisphere.visual_liquid.hebbian_update(
                        vis_post_replay, vis_pre_replay, plasticity * 2.0
                    )
            
            # Update diagnostics
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    visual_liquid_norm = model.right_hemisphere.visual_liquid.W_fast_short.norm().item()
                    audio_liquid_norm = model.right_hemisphere.audio_liquid.W_fast_short.norm().item()
                    callosal_flow = diagnostics.measure_callosal_flow(fused_feat, enriched_ctx, channels)
                    gate_mean_val = gate.mean().item() if gate.numel() > 1 else gate.item()
                    gate_std_val = gate.std().item() if gate.numel() > 1 else 0.0
                    channel_fatigue = channels['fatigue']
                    
                    synergy = diagnostics.calculate_synergy(
                        visual_node, audio_node, callosal_flow, gate_mean_val, gate_std_val
                    )
                    
                    diagnostics.update(
                        loss=ce_loss.item(),
                        visual_metabolism=float(visual_node.metabolism),
                        visual_fatigue=float(visual_node.fatigue),
                        visual_liquid_norm=visual_liquid_norm,
                        audio_metabolism=float(audio_node.metabolism),
                        audio_fatigue=float(audio_node.fatigue),
                        audio_liquid_norm=audio_liquid_norm,
                        callosal_flow=callosal_flow,
                        left_gate_mean=gate_mean_val,
                        left_gate_std=gate_std_val,
                        synergy_score=synergy,
                        health_score=health_score,
                        coherence_loss=coherence_loss.item(),
                        objects_fatigue=channel_fatigue['visual'],   # <- unificado
                        actions_fatigue=channel_fatigue['audio'],    # <- unificado
                        scene_fatigue=channel_fatigue['semantic'],   # <- unificado
                        reasoning_steps=reasoning_steps.item() if torch.is_tensor(reasoning_steps) else 0.0,
                        mtp_loss=mtp_loss.item() if torch.is_tensor(mtp_loss) else 0.0
                    )
            
            pbar_dict = {
                'loss': f'{ce_loss.item():.3f}',
                'coherence': f'{coherence_loss.item():.3f}',
                'gate': f'{gate.mean().item():.2f}',
                'mem': f"{len(episodic_memory.buffer)}/{episodic_memory.capacity}"
            }
            
            if torch.is_tensor(mtp_loss):
                pbar_dict['mtp'] = f'{mtp_loss.item():.3f}'
            
            if torch.is_tensor(reasoning_steps):
                pbar_dict['reason'] = f'{reasoning_steps.item():.1f}'
            
            pbar.set_postfix(pbar_dict)
        
        scheduler.step()
        
        # Evaluación
        if epoch % 5 == 0:
            model.eval()
            print("\n🧠 Evaluación de generación...\n")
            
            with torch.no_grad():
                bleu_scores, cider_scores, spice_scores = [], [], []
                
                for sample_idx in range(min(10, len(dataset))):
                    sample_img, sample_audio, sample_cap, raw_caption = dataset[sample_idx * (len(dataset) // 10)]
                    sample_img = sample_img.unsqueeze(0).to(device)
                    sample_audio = sample_audio.unsqueeze(0).to(device)
                    
                    generated_text, _ = model(
                        sample_img, sample_audio, 
                        captions=None, epoch=epoch
                    )
                    
                    gen_words = [id2word.get(int(t.item()), '<UNK>') for t in generated_text[0]]
                    gen_sentence = " ".join(w for w in gen_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                    
                    if sample_idx < 3:
                        print(f"Muestra {sample_idx + 1}:")
                        print(f"  GT:   {raw_caption}")
                        print(f"  Gen:  {gen_sentence}\n")
                    
                    bleu = self.language_metrics.sentence_bleu(raw_caption, gen_sentence)
                    cider = self.linguistic_feedback.compute_cider(raw_caption, gen_sentence)
                    spice = self.linguistic_feedback.compute_spice(raw_caption, gen_sentence)
                    
                    bleu_scores.append(bleu)
                    cider_scores.append(cider)
                    spice_scores.append(spice)
                
                diagnostics.update(
                    bleu_score=np.mean(bleu_scores),
                    cider_score=np.mean(cider_scores),
                    spice_score=np.mean(spice_scores)
                )
            
            model.train()
        
        diagnostics.report(epoch)
        print(f"✅ Época {epoch:02d} | Loss: {total_loss/len(dataloader):.4f}\n")
    
    print("✅ Entrenamiento tricameral completado!")
    
    # Guardar modelo final
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'id2word': id2word,
        'diagnostics': diagnostics.history,
        'medical_interventions': medical_system.intervention_history,
        'cognitive_interventions': cognitive_system.cognitive_history,
        'cache_stats': cognitive_system.linguistic_feedback.get_cache_stats(),
        'episodic_memory_size': len(episodic_memory.buffer)
    }, './tricameral_model_final.pth')
    
    print("💾 Modelo guardado en: ./tricameral_model_final.pth")

# =============================================================================
# EJECUCIÓN DIRECTA
# =============================================================================
if __name__ == "__main__":
    # Descargar dataset automáticamente
    print("🚀 Iniciando descarga del dataset...")
    path = kagglehub.dataset_download("warcoder/flickr-8k-audio-caption-corpus")
    print(f"📦 Dataset descargado en: {path}")
    
    # Entrenar
    train_tricameral()
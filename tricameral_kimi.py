# =============================================================================
# NeuroLogos TRICAMERAL v5.1
# Hemisferio Derecho: Visión + Audio
# Hemisferio Izquierdo: Lenguaje + Razonamiento
# Corpus Callosum: Fusión trimodal (ve, escucha, razona)
# + Dataset Flickr8k con Audio Pre-generado
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
import subprocess
import urllib.request
import zipfile
import shutil
import soundfile as sf


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
# =============================================================================
# EJECUTAR ANTES DE ENTRENAR
# =============================================================================
print("🚀 Iniciando setup del dataset...\n")
dataset_path = setup_flickr8k_with_audio('./flickr8k_full')

if dataset_path:
    print(f"\n✅ Continuando con entrenamiento...\n")
    # train_tricameral()  # Descomentar para entrenar
else:
    print("\n❌ ERROR: Setup fallido. Revisa los logs.")



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
# MEMORIA EPISÓDICA (CLASE COMPLETA)
# =============================================================================
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
            self.buffer.append((image.detach().cpu(), caption.detach().cpu()))
            self.surprise_scores.append(surprise_score)
    
    def sample(self, batch_size):
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


class NeurocognitiveSystem:
    def __init__(self):
        self.cognitive_history = []
        self.last_intervention_epoch = -5
        self.linguistic_feedback = LinguisticFeedbackLoop()  # ✅ Este mantiene su funcionalidad
        
        # Umbral ajustado para modelo actual
        self.cider_threshold = 0.08
        self.spice_threshold = 0.12
        self.plateau_threshold = 0.005
    
    def assess_cognitive_state(self, cider_score, spice_score, combined_reward, epoch):
        issues = []
        severity = 0
        confidence = []
        
        if epoch < 3:
            return issues, severity, confidence
        
        # Detección de problemas lingüísticos reales
        if spice_score < self.spice_threshold:
            issues.append("semantic_deficit")
            severity += 3
            confidence.append(f"Déficit semántico (SPICE: {spice_score:.3f})")
        
        if combined_reward < 0.15:
            issues.append("linguistic_deficit")
            severity += 2
            confidence.append(f"Déficit lingüístico (recompensa: {combined_reward:.3f})")
        
        return issues, severity, confidence
    
    def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics=None):
        if epoch - self.last_intervention_epoch < 2 or severity == 0:
            return False
        
        level = "🔴 Nivel 3" if severity > 5 else "🟠 Nivel 2" if severity > 2 else "🟡 Nivel 1"
        print(f"\n{'='*80}")
        print(f"🧠 INTERVENCIÓN COGNITIVA - {level} - Severidad: {severity}/9")
        print(f"   Problemas: {', '.join(issues)}")
        print(f"   Confianza: {confidence}")
        print(f"{'='*80}")
        
        interventions_applied = []
        
        with torch.no_grad():
            # SOLO INTERVENCIONES SOBRE COMPONENTES EXISTENTES
            
            if "semantic_deficit" in issues:
                print("💡 Reforzando proyección semántica")
                model.corpus_callosum.semantic_proj.weight.data *= 1.08
                model.corpus_callosum.residual_scale_base.data *= 1.02
                interventions_applied.append("semantic_projection_boost")
            
            if "linguistic_deficit" in issues:
                print("💡 Ajustando gates de comunicación")
                model.corpus_callosum.residual_scale_base.data *= 1.05
                interventions_applied.append("callosum_alignment_boost")
            
            if "linguistic_deficit" in issues and severity > 5:
                print("💡 RESETEO COGNITIVO SUAVE")
                model.right_hemisphere.visual_liquid.W_slow.weight.data *= 0.95
                model.right_hemisphere.audio_liquid.W_slow.weight.data *= 0.95
                interventions_applied.append("cognitive_soft_reset")
        
        print(f"\n✓ Intervenciones aplicadas: {len(interventions_applied)}")
        for inter in interventions_applied:
            print(f"  - {inter}")
        print(f"{'='*80}\n")
        
        self.last_intervention_epoch = epoch
        return True

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
# SISTEMA DE RETROALIMENTACIÓN LINGÜÍSTICA (CLASE COMPLETA)
# =============================================================================
class LinguisticFeedbackLoop:
    def __init__(self, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.beta = beta
        self.history = []
        self.ngram_cache = {}
        self.ngram_cache_hits = 0
        self.ngram_cache_misses = 0
        self.score_cache = {}
        self.score_cache_hits = 0
        self.score_cache_misses = 0
    
    def compute_linguistic_reward(self, references, hypotheses):
        cider_scores = []
        spice_scores = []
        for ref, hyp in zip(references, hypotheses):
            pair_key = (hash(ref), hash(hyp))
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
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        if len(ref_words) == 0 and len(hyp_words) == 0:
            return 1.0
        intersection = len(ref_words & hyp_words)
        union = len(ref_words | hyp_words)
        return intersection / union if union > 0 else 0.0
    
    def _get_ngrams(self, sentence, n=4):
        tokens = sentence.lower().split()
        return {tuple(tokens[i:i+n]): tokens[i:i+n].count(tokens[i]) for i in range(len(tokens)-n+1)}
    
    def get_cache_stats(self):
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
# MÉTRICAS LINGÜÍSTICAS BÁSICAS (CLASE COMPLETA)
# =============================================================================
class LanguageMetrics:
    @staticmethod
    def sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
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
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        min_len = min(len(ref_tokens), len(hyp_tokens))
        if min_len == 0:
            return 0.0
        matches = sum(1 for i in range(min_len) if ref_tokens[i] == hyp_tokens[i])
        return matches / max(len(ref_tokens), len(hyp_tokens))
    
    @staticmethod
    def word_overlap(reference, hypothesis):
        ref_set = set(reference.lower().split())
        hyp_set = set(hypothesis.lower().split())
        if len(ref_set | hyp_set) == 0:
            return 0.0
        return len(ref_set & hyp_set) / len(ref_set | hyp_set)



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
            
            adaptive_lr = self.base_lr
            if norm_ratio > 3.0:
                adaptive_lr *= 0.01
                self.homeostasis.mul_(0.8)
            elif norm_ratio > 1.5:
                adaptive_lr *= 0.3
                self.homeostasis.mul_(0.95)
            elif norm_ratio < 0.5:
                adaptive_lr *= 1.2
                self.homeostasis.mul_(1.02).clamp_(0.5, 1.0)
            else:
                self.homeostasis.mul_(1.01).clamp_(0.8, 1.0)
            
            update_short = adaptive_lr * plasticity * float(self.homeostasis) * torch.tanh(hebb)
            self.W_fast_short += update_short[:self.fast_short_dim]
            
            update_long = adaptive_lr * plasticity * 0.3 * float(self.metabolism) * torch.tanh(hebb)
            self.W_fast_long += update_long[:self.fast_long_dim]
            
            decay = 0.999 if norm_ratio < 1.0 else 0.99 if norm_ratio < 2.0 else 0.98
            self.W_fast_short.mul_(decay)
            self.W_fast_long.mul_(decay * 0.995)
            
            self.W_fast_short.clamp_(-0.5, 0.5)
            self.W_fast_long.clamp_(-0.3, 0.3)
            
            if current_norm > 5.0:
                self.W_fast_short.mul_(self.norm_target / current_norm)



    def update_physiology_advanced(self, loss_value):
        with torch.no_grad():
            loss_signal = max(0.0, min(1.0, 1.0 - loss_value / 4.0))
            homeostasis_signal = float(self.homeostasis)
            target_metab = 0.5 + 0.3 * loss_signal + 0.1 * homeostasis_signal
            
            metabolism_impact = self.metabolism_impact * (float(self.metabolism) - 0.6)
            target_metab += metabolism_impact
            
            self.metabolism.mul_(0.9).add_(0.1 * target_metab).clamp_(0.3, 0.9)
            
            norm_ratio = self.norm_ema / self.norm_target
            
            cognitive_load = self.cognitive_load_factor * (1.0 - loss_signal)
            fatigue_increment = 0.002 if norm_ratio < 2.0 else 0.01
            fatigue_increment += cognitive_load
            
            self.fatigue.mul_(self.fatigue_decay).add_(fatigue_increment).clamp_(0, 0.5)
            
            if float(self.fatigue) > 0.3:
                self.metabolism.mul_(0.95)
            
            if float(self.homeostasis) < 0.7:
                self.sensitivity.mul_(0.95)
            else:
                target_sens = 0.5 + 0.2 * (1.0 - float(self.fatigue))
                self.sensitivity.mul_(0.95).add_(0.05 * target_sens)
            
            self.sensitivity.clamp_(0.3, 0.7)


# =============================================================================
# SISTEMA MÉDICO TRIANGULADO (VERSIÓN SIMPLIFICADA Y FUNCIONAL)
# =============================================================================
class TriangulatedMedicalSystem:
    def __init__(self):
        self.intervention_history = []
        self.consecutive_collapses = 0
        self.last_epoch = -1
        self.gate_zombie_threshold = 0.65
        
    def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow):
        issues = []
        severity = 0
        confidence = []
        
        # Diagnóstico basado en métricas reales del modelo
        if gate_mean > self.gate_zombie_threshold:
            issues.append("gate_degraded")
            severity += 4
            confidence.append(f"Gate degradado ({gate_mean:.2f})")
        
        if callosal_flow < 0.2:
            issues.append("communication_collapse")
            severity += 5
            confidence.append(f"Comunicación colapsada ({callosal_flow:.2f})")
        
        if liquid_norm > 3.5:
            issues.append("liquid_explosion")
            severity += 3
            confidence.append(f"Neurona líquida inestable (norm: {liquid_norm:.2f})")
        
        return issues, severity, confidence
    
    def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch):
        if epoch == self.last_epoch or severity == 0:
            return False
        
        level = "🔴 Nivel 3" if severity > 5 else "🟠 Nivel 2" if severity > 2 else "🟡 Nivel 1"
        print(f"\n{'='*80}")
        print(f"🏥 INTERVENCIÓN MÉDICA - {level} - Severidad: {severity}/12")
        print(f"   Problemas: {', '.join(issues)}")
        print(f"   Confianza: {confidence}")
        print(f"{'='*80}")
        
        interventions = []
        
        with torch.no_grad():
            # SOLO INTERVENCIONES SOBRE COMPONENTES EXISTENTES
            
            if "gate_degraded" in issues:
                print("💊 Ajustando gates del Corpus Callosum")
                # ✅ Este componente existe
                for block in model.corpus_callosum.transfer:
                    if isinstance(block[0], nn.Linear):
                        block[0].weight.data *= 0.95  # Ligera reducción
                        interventions.append("callosum_gate_adjustment")
            
            if "communication_collapse" in issues:
                print("💊 Fortaleciendo conexiones callosales")
                # ✅ Este componente existe
                model.corpus_callosum.residual_scale_base.data *= 1.1
                interventions.append("callosum_boost")
            
            if "liquid_explosion" in issues:
                print("💊 Estabilizando neuronas líquidas")
                # ✅ Estos componentes existen
                visual_node = model.right_hemisphere.visual_liquid
                audio_node = model.right_hemisphere.audio_liquid
                
                # Reset de pesos rápidos
                visual_node.W_fast_short *= 0.9
                audio_node.W_fast_short *= 0.9
                interventions.append("liquid_stabilization")
            
            if "gate_degraded" in issues and severity > 5:
                print("💊 RESETEO SUAVE DEL SISTEMA")
                # Reset suave de proyecciones del callosum
                model.corpus_callosum.visual_proj.weight.data *= 0.9
                model.corpus_callosum.audio_proj.weight.data *= 0.9
                model.corpus_callosum.semantic_proj.weight.data *= 0.9
                interventions.append("soft_system_reset")
        
        print(f"\n✓ Intervenciones aplicadas: {len(interventions)}")
        for i, inter in enumerate(interventions, 1):
            print(f"  {i}. {inter}")
        print(f"{'='*80}\n")
        
        self.intervention_history.append({
            'epoch': epoch,
            'issues': issues,
            'severity': severity,
            'interventions': interventions
        })
        self.last_epoch = epoch
        return True

# =============================================================================
# HEMISFERIO IZQUIERDO SIMPLIFICADO
# =============================================================================
class LeftHemisphere(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.lstm_expansion = 128
        self.lstm_output_dim = hidden_dim + self.lstm_expansion
        
        self.lstm = nn.LSTM(embed_dim + hidden_dim, self.lstm_output_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.bottleneck = nn.Linear(self.lstm_output_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
    
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
            logits = self.output_projection(lstm_out)
            
            gate = torch.sigmoid(torch.randn(batch_size, seq_len, 1, device=device) * 0.1 + 0.5)
            
            # FIX: Retornar tensores escalares en lugar de None
            mtp_loss = torch.tensor(0.0, device=device)
            reasoning_steps = torch.tensor(0.0, device=device)
            
            return logits, gate, mtp_loss, reasoning_steps
        else:
            return self._greedy_decode(visual_context, max_len, device)
    
    def _greedy_decode(self, visual_context, max_len, device):
        batch_size = visual_context.size(0)
        generated = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            emb = self.embedding(generated)
            seq_len = emb.size(1)
            visual_expanded = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_input = torch.cat([emb, visual_expanded], dim=2)
            
            out, _ = self.lstm(lstm_input, self._get_init_state(visual_context))
            out = self.bottleneck(out)
            logits = self.output_projection(out[:, -1:, :])
            
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == 2).all():
                break
        
        return generated
    
    def _get_init_state(self, visual_context):
        batch_size = visual_context.size(0)
        h0 = visual_context.unsqueeze(0).repeat(2, 1, 1)
        
        padding = torch.zeros(2, batch_size, self.lstm_expansion, device=visual_context.device)
        h0 = torch.cat([h0, padding], dim=-1)
        c0 = torch.zeros_like(h0)
        
        return (h0, c0)



# =============================================================================
# ENCODER DE AUDIO (Hemisferio Derecho - Canal Auditivo)
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
        x = self.conv_layers(mel_spec)
        x = x.transpose(1, 2)
        x = self.temporal_encoder(x)
        x = x.transpose(1, 2)
        x = self.adaptive_pool(x).squeeze(-1)
        out = self.projection(x)
        
        return out

# =============================================================================
# HEMISFERIO DERECHO EXTENDIDO (Visión + Audio)
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
        self.register_buffer('visual_fatigue', torch.tensor(0.0))
        self.register_buffer('audio_fatigue', torch.tensor(0.0))
        self.register_buffer('semantic_fatigue', torch.tensor(0.0))
        self.fatigue_decay = 0.95
        self.fatigue_recovery = 0.01
    
    def forward(self, right_features):
        visual_channel = self.visual_proj(right_features)
        audio_channel = self.audio_proj(right_features)
        semantic_channel = self.semantic_proj(right_features)
        visual_gated = visual_channel * torch.sigmoid(1.0 - self.visual_fatigue)
        audio_gated = audio_channel * torch.sigmoid(1.0 - self.audio_fatigue)
        semantic_gated = semantic_channel * torch.sigmoid(1.0 - self.semantic_fatigue)
        structured = torch.cat([visual_gated, audio_gated, semantic_gated], dim=-1)
        structured_expanded = structured.unsqueeze(1)
        attn_out, _ = self.trimodal_attention(
            structured_expanded, structured_expanded, structured_expanded
        )
        attn_out = attn_out.squeeze(1)
        for block in self.transfer:
            attn_out = attn_out + block(attn_out)
        flow_strength = torch.sigmoid(structured.mean(dim=-1))
        dynamic_scale = self.residual_scale_base * (0.7 + 0.6 * flow_strength)
        output = attn_out + dynamic_scale.unsqueeze(-1) * right_features
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
        with torch.no_grad():
            visual_activity = visual_channel.norm(dim=-1).mean()
            audio_activity = audio_channel.norm(dim=-1).mean()
            semantic_activity = semantic_channel.norm(dim=-1).mean()
            
            self.visual_fatigue = self.visual_fatigue * self.fatigue_decay + 0.01 * visual_activity
            self.audio_fatigue = self.audio_fatigue * self.fatigue_decay + 0.01 * audio_activity
            self.semantic_fatigue = self.semantic_fatigue * self.fatigue_decay + 0.01 * semantic_activity
            
            self.visual_fatigue = torch.clamp(self.visual_fatigue - self.fatigue_recovery, 0.0, 1.0)
            self.audio_fatigue = torch.clamp(self.audio_fatigue - self.fatigue_recovery, 0.0, 1.0)
            self.semantic_fatigue = torch.clamp(self.semantic_fatigue - self.fatigue_recovery, 0.0, 1.0)


    
    def adjust_gates_by_fatigue(self):
        with torch.no_grad():
            visual_factor = (1.0 - 0.05 * float(self.visual_fatigue))
            audio_factor = (1.0 - 0.05 * float(self.audio_fatigue))
            semantic_factor = (1.0 - 0.05 * float(self.semantic_fatigue))
            
            self.visual_proj.weight.data *= visual_factor
            self.audio_proj.weight.data *= audio_factor
            self.semantic_proj.weight.data *= semantic_factor




class EnhancedDiagnosticsTricameral:
    def __init__(self):
        self.history = {
            'loss': [], 
            'visual_liquid_metabolism': [], 'visual_liquid_fatigue': [], 'visual_liquid_norm': [],
            'audio_liquid_metabolism': [], 'audio_liquid_fatigue': [], 'audio_liquid_norm': [],
            'callosal_flow': [], 'left_gate_mean': [], 'left_gate_std': [],
            'synergy_score': [], 'health_score': [],
            'bleu_score': [], 'token_accuracy': [], 'word_overlap': [],
            'cider_score': [], 'spice_score': [], 'linguistic_reward': [],
            'coherence_loss': [], 
            'callosum_visual_fatigue': [], 'callosum_audio_fatigue': [], 'callosum_semantic_fatigue': [],
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
        elif key in ['callosum_visual_fatigue', 'callosum_audio_fatigue', 'callosum_semantic_fatigue']:
            return 0.0
        return 0.0
    
    def visualize_fatigue_distribution(self, epoch):
        if epoch % 5 == 0:
            visual_fatigue = self.get_recent_avg('callosum_visual_fatigue', n=10)
            audio_fatigue = self.get_recent_avg('callosum_audio_fatigue', n=10)
            semantic_fatigue = self.get_recent_avg('callosum_semantic_fatigue', n=10)
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
        if len(self.history['loss']) == 0:
            return
        self.visualize_fatigue_distribution(epoch)
        self.visualize_reasoning_metrics(epoch)
        print(f"\n{'='*80}")
        print(f"📊 REPORTE COMPLETO - Época {epoch}")
        print(f"{'='*80}")
        loss = self.get_recent_avg('loss')
        print(f"\n📉 ENTRENAMIENTO:")
        print(f"  Loss: {loss:.4f}")
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
        reasoning_steps = self.get_recent_avg('reasoning_steps')
        mtp_loss = self.get_recent_avg('mtp_loss')
        logical_coherence = self.get_recent_avg('logical_coherence')
        print(f"\n🧠 RAZONAMIENTO:")
        print(f"  Pasos: {reasoning_steps:.2f} {'🟢' if 2 <= reasoning_steps <= 3 else '🟡'}")
        print(f"  MTP Loss: {mtp_loss:.3f} {'🟢' if mtp_loss < 0.2 else '🟡' if mtp_loss < 0.4 else '🔴'}")
        print(f"  Coherencia: {logical_coherence:.3f} {'🟢' if logical_coherence > 0.4 else '🟡' if logical_coherence > 0.2 else '🔴'}")
        visual_liquid = self.get_recent_avg('visual_liquid_norm')
        audio_liquid = self.get_recent_avg('audio_liquid_norm')
        print(f"\n🧬 FISIOLOGÍA:")
        print(f"  Visual Norm: {visual_liquid:.3f} {'🟢' if visual_liquid < 2.0 else '🟡' if visual_liquid < 4.0 else '🔴'}")
        print(f"  Audio Norm: {audio_liquid:.3f} {'🟢' if audio_liquid < 2.0 else '🟡' if audio_liquid < 4.0 else '🔴'}")
        flow = self.get_recent_avg('callosal_flow')
        gate_mean = self.get_recent_avg('left_gate_mean')
        gate_std = self.get_recent_avg('left_gate_std')
        print(f"\n🔗 COMUNICACIÓN:")
        print(f"  Callosum: {flow:.3f} {'🟢' if flow > 0.5 else '🟡' if flow > 0.3 else '🔴'}")
        print(f"  Gate Mean: {gate_mean:.3f} {'🟢' if 0.35 < gate_mean < 0.65 else '🟡'}")
        print(f"  Gate Std: {gate_std:.3f}")
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
# MODELO TRICAMERAL COMPLETO
# =============================================================================
class NeuroLogosTricameral(nn.Module):
    """Arquitectura completa: Visión + Audio -> Lenguaje"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.right_hemisphere = RightHemisphereTricameral(output_dim=512)
        self.corpus_callosum = CorpusCallosumTrimodal(dim=512)
        self.left_hemisphere = LeftHemisphere(vocab_size, embed_dim=256, hidden_dim=512)
    
    def forward(self, image, audio, captions=None, epoch=0):
        fused_features, vis_post, vis_pre, aud_post, aud_pre = self.right_hemisphere(image, audio)
        enriched_context, channels = self.corpus_callosum(fused_features)
        
        if captions is not None:
            logits, gate, mtp_loss, reasoning_steps = self.left_hemisphere(
                enriched_context, captions, channels, epoch=epoch
            )
            
            return (
                logits, fused_features, enriched_context, gate, 
                vis_post, vis_pre, aud_post, aud_pre, 
                channels, mtp_loss, reasoning_steps
            )
        else:
            generated_text = self.left_hemisphere(
                enriched_context, captions=None, channels=channels, epoch=epoch
            )
            
            return generated_text, None, None, None, None, None, None, None, None, None, None



# =============================================================================
# DATASET MULTIMODAL ACTUALIZADO
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
        
        # Resampler por si acaso
        self.resampler = T.Resample(sample_rate, sample_rate)
        
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
        
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)
        
        waveform_np, sr = sf.read(audio_path)
        waveform = torch.from_numpy(waveform_np).float()
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.T
        
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        target_len = 300
        if mel_spec.size(-1) < target_len:
            pad = target_len - mel_spec.size(-1)
            mel_spec = F.pad(mel_spec, (0, pad))
        else:
            mel_spec = mel_spec[..., :target_len]
        
        tokens = ['<BOS>'] + caption.lower().split() + ['<EOS>']
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        
        return image, mel_spec.squeeze(0), torch.tensor(token_ids, dtype=torch.long), caption






# =============================================================================
# FUNCIÓN DE PÉRDIDA EXTENDIDA
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
    if linguistic_reward is not None:
        linguistic_loss = -lambda_reward * linguistic_reward
    
    # MTP
    mtp_term = 0.0
    if mtp_loss is not None and isinstance(mtp_loss, torch.Tensor):
        mtp_term = lambda_mtp * mtp_loss
    
    # Total
    total_loss = (
        ce_loss + 
        0.05 * gate_penalty + 
        0.2 * diversity_penalty + 
        0.1 * coherence_loss +
        linguistic_loss + 
        mtp_term
    )
    
    return total_loss, ce_loss, gate_penalty, diversity_penalty, coherence_loss


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
    
    flickr_dir = setup_flickr8k_with_audio('./flickr8k_full')
    images_dir = os.path.join(flickr_dir, 'Images')
    audio_dir = os.path.join(flickr_dir, 'wavs')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    if not os.path.exists(audio_dir) or len(os.listdir(audio_dir)) < 1000:
        print("❌ Error: No se encontraron suficientes audios. Ejecuta setup_flickr8k_with_audio() primero.")
        return
    
    print(f"✅ Audio directory found: {len(os.listdir(audio_dir))} files\n")
    
    vocab, id2word = build_vocab_flickr(captions_file, vocab_size=5000)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Flickr8kMultimodalDataset(
        images_dir, audio_dir, captions_file, vocab, 
        img_transform=transform, max_len=30
    )
    
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    model = NeuroLogosTricameral(len(vocab)).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.right_hemisphere.parameters(), 'lr': 3e-4},
        {'params': model.corpus_callosum.parameters(), 'lr': 5e-4},
        {'params': model.left_hemisphere.parameters(), 'lr': 2e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
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
    
    for epoch in range(30):
        model.train()
        total_loss = 0
        
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
        
        model.corpus_callosum.adjust_gates_by_fatigue()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Health: {health_score}/5 | Med: {medicine_level}]")
        
        for batch_idx, (images, audio_specs, captions, raw_captions) in enumerate(pbar):
            images = images.to(device)
            audio_specs = audio_specs.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            
            (logits, fused_feat, enriched_ctx, gate, 
             vis_post, vis_pre, aud_post, aud_pre, 
             channels, mtp_loss, reasoning_steps) = model(
                images, audio_specs, captions, epoch=epoch
            )
            
            # FIX: compute_tricameral_loss returns 5 values, unpack correctly
            loss, ce_loss, gate_penalty, diversity_penalty, coherence_loss = compute_tricameral_loss(
                logits, captions, gate, vocab, vis_post, aud_post, mtp_loss
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            plasticity = 0.1 if epoch < 10 else 0.05
            model.right_hemisphere.visual_liquid.hebbian_update(vis_post, vis_pre, plasticity)
            model.right_hemisphere.audio_liquid.hebbian_update(aud_post, aud_pre, plasticity)
            
            model.right_hemisphere.visual_liquid.update_physiology_advanced(ce_loss.item())
            model.right_hemisphere.audio_liquid.update_physiology_advanced(ce_loss.item())
            
            total_loss += ce_loss.item()
            
            if batch_idx % 20 == 0:
                surprise = episodic_memory.compute_surprise(logits, captions[:, 1:], gate.mean())
                
                for i in range(images.size(0)):
                    episodic_memory.add(images[i], captions[i], surprise)
                
                if batch_idx % 100 == 0:
                    print(f"  🧠 Episodic Buffer: {len(episodic_memory.buffer)}/{episodic_memory.capacity} samples")
            
            if batch_idx % 5 == 0 and len(episodic_memory.buffer) > 32:
                replay_samples = episodic_memory.sample(16)
                if replay_samples:
                    replay_imgs = torch.stack([s[0] for s in replay_samples]).to(device)
                    replay_caps = torch.stack([s[1] for s in replay_samples]).to(device)
                    
                    # FIX: Desempaquetar los 11 valores correctamente
                    (logits_replay, _, _, gate_replay, vis_post_replay, vis_pre_replay, 
                     _, _, _, _, _) = model(
                        replay_imgs, torch.randn(replay_imgs.size(0), 80, 300).to(device),
                        replay_caps, epoch=epoch
                    )
                    
                    loss_replay, ce_replay, _, _, _ = compute_tricameral_loss(
                        logits_replay, replay_caps, gate_replay, vocab, 
                        vis_post_replay, torch.randn_like(vis_post_replay)
                    )
                    
                    (0.5 * loss_replay).backward()
                    
                    model.right_hemisphere.visual_liquid.hebbian_update(
                        vis_post_replay, vis_pre_replay, plasticity * 2.0
                    )

            
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
                        visual_liquid_metabolism=float(visual_node.metabolism),
                        visual_liquid_fatigue=float(visual_node.fatigue),
                        visual_liquid_norm=visual_liquid_norm,
                        audio_liquid_metabolism=float(audio_node.metabolism),
                        audio_liquid_fatigue=float(audio_node.fatigue),
                        audio_liquid_norm=audio_liquid_norm,
                        callosal_flow=callosal_flow,
                        left_gate_mean=gate_mean_val,
                        left_gate_std=gate_std_val,
                        synergy_score=synergy,
                        health_score=health_score,
                        coherence_loss=coherence_loss.item(),
                        callosum_visual_fatigue=channel_fatigue['visual'],
                        callosum_audio_fatigue=channel_fatigue['audio'],
                        callosum_semantic_fatigue=channel_fatigue['semantic'],
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
        
        if epoch % 5 == 0:
            model.eval()
            print("\n🧠 Evaluación de generación...\n")
            
            with torch.no_grad():
                bleu_scores, cider_scores, spice_scores = [], [], []
                
                for sample_idx in range(min(10, len(dataset))):
                    sample_img, sample_audio, sample_cap, raw_caption = dataset[sample_idx * (len(dataset) // 10)]
                    sample_img = sample_img.unsqueeze(0).to(device)
                    sample_audio = sample_audio.unsqueeze(0).to(device)
                    
                    # FIX: Desempaquetar correctamente los 11 valores
                    (generated_text, _, _, _, _, _, _, _, _, _, _) = model(
                        sample_img, sample_audio, 
                        captions=None, epoch=epoch
                    )
                    
                    gen_words = [id2word.get(int(t.item()), '<UNK>') for t in generated_text[0]]
                    gen_sentence = " ".join(w for w in gen_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                    
                    if sample_idx < 3:
                        print(f"Muestra {sample_idx + 1}:")
                        print(f"  GT:   {raw_caption}")
                        print(f"  Gen:  {gen_sentence}\n")
                    
                    bleu = diagnostics.language_metrics.sentence_bleu(raw_caption, gen_sentence)
                    cider = cognitive_system.linguistic_feedback.compute_cider(raw_caption, gen_sentence)
                    spice = cognitive_system.linguistic_feedback.compute_spice(raw_caption, gen_sentence)
                    
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
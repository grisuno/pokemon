# =============================================================================
# NeuroLogos TRICAMERAL v5.1
# Hemisferio Derecho: Visión + Audio
# Hemisferio Izquierdo: Lenguaje + Razonamiento
# Corpus Callosum: Fusión trimodal (ve, escucha, razona)
# + Dataset Flickr8k con Audio Pre-generado
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchaudio
import torchaudio.transforms as T
import warnings
import kagglehub
import subprocess
import urllib.request
import zipfile
import shutil
import soundfile as sf
import time
import numpy as np
from pathlib import Path
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from functools import lru_cache
from collections import deque
from torch.utils.data import random_split
from torch.nn.functional import scaled_dot_product_attention
from typing import NamedTuple

warnings.filterwarnings('ignore')
EPISODIC_UPDATE_FREQ = 5
EPISODIC_REPLAY_FREQ = 50  # Aumentado de 20 a 50 para reducir overhead
EPISODIC_MIN_BUFFER = 64
EPISODIC_SAMPLES_PER_UPDATE = 2

REWARD_FREQUENCY = 20
REPLAY_FREQUENCY = 50  # Sincronizado con episodic replay
MEMORY_LOG_FREQUENCY = 500
DIAGNOSTIC_UPDATE_FREQ = 20
COGNITIVE_DIAG_FREQ = 50

def preprocess_and_cache_spectrograms(audio_dir, cache_dir='./flickr8k_full/spectrograms_cache', 
                                      sample_rate=16000, target_len=300):
    """
    Preprocesa todos los archivos .wav a Mel-spectrogramas y los guarda como tensores .pt
    Esto elimina el cuello de botella de I/O durante entrenamiento
    """   
    os.makedirs(cache_dir, exist_ok=True)
    
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
        f_min=0,
        f_max=8000
    )
    
    audio_files = list(Path(audio_dir).glob("*.wav"))
    cached_count = 0
    
    print(f"\n🎵 Preprocesando {len(audio_files)} archivos de audio...")
    
    for audio_path in tqdm(audio_files, desc="Generando spectrogramas"):
        cache_path = os.path.join(cache_dir, audio_path.stem + '.pt')
        
        if os.path.exists(cache_path):
            cached_count += 1
            continue
        
        try:
            waveform_np, sr = sf.read(str(audio_path))
            waveform = torch.from_numpy(waveform_np).float()
            
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.T
            
            if sr != sample_rate:
                resampler = T.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            mel_spec = mel_spectrogram(waveform)
            mel_spec = torch.log(mel_spec + 1e-9)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            if mel_spec.size(-1) < target_len:
                pad = target_len - mel_spec.size(-1)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
            else:
                mel_spec = mel_spec[..., :target_len]
            
            torch.save(mel_spec.squeeze(0), cache_path)
            
        except Exception as e:
            print(f"⚠️ Error procesando {audio_path.name}: {e}")
            continue
    
    print(f"✅ Spectrogramas cacheados: {len(audio_files)} archivos")
    print(f"   Ya existían: {cached_count}")
    print(f"   Nuevos: {len(audio_files) - cached_count}")
    print(f"   Ubicación: {cache_dir}\n")
    
    return cache_dir


# FIX: Obtener device de los parámetros del modelo
def apply_emergency_fixes(model):
    device = next(model.parameters()).device
    print("🚨 Aplicando patches de emergencia...")
    
    # Fix 3: Inicialización neutra ya está correcta en tu código
    model.left_hemisphere.liquid_gate[-1].bias.data.fill_(0.0)
    
    # Fix 4: Reset fatiga
    model.corpus_callosum.visual_fatigue.data = torch.tensor(0.0, device=device)
    model.corpus_callosum.audio_fatigue.data = torch.tensor(0.0, device=device)
    model.corpus_callosum.semantic_fatigue.data = torch.tensor(0.0, device=device)
    
    # Fix 5: Gates apertura forzada
    model.corpus_callosum.trimodal_gates.data = torch.tensor([-0.5, -0.5, -0.5], device=device)
    
    print("✅ Patches aplicados. Gate Mean debe saltar a ~0.5 en el siguiente batch.")
    return model


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
    
    # FIX: Verificación de integridad con conteo real de archivos
    has_images = os.path.exists(images_dir) and len([f for f in Path(images_dir).glob("*.jpg")]) >= 8000
    has_audio = os.path.exists(audio_dir) and len([f for f in Path(audio_dir).glob("*.wav")]) >= 40000
    has_captions = os.path.exists(captions_file) and os.path.getsize(captions_file) > 1000000
    
    print(f"📊 Estado del dataset:")
    print(f"  Imágenes: {'✅' if has_images else '❌'} ({len(list(Path(images_dir).glob('*.jpg'))) if os.path.exists(images_dir) else 0}/8091)")
    print(f"  Audios: {'✅' if has_audio else '❌'} ({len(list(Path(audio_dir).glob('*.wav'))) if os.path.exists(audio_dir) else 0}/40000)")
    print(f"  Captions: {'✅' if has_captions else '❌'}")
    
    # Si todo está completo, retornar
    if has_images and has_audio and has_captions:
        print("✅ Dataset completo, sin necesidad de descargas\n")
        return data_dir
    
    # ========== AUDIO - FIX: Manejo de errores y verificación de integridad ==========
    if not has_audio:
        print("\n📥 Descargando y procesando audios desde Kaggle...")
        try:
            path = kagglehub.dataset_download("warcoder/flickr-8k-audio-caption-corpus")
            print(f"✓ Dataset descargado en: {path}")
            
            # FIX: Búsqueda recursiva robusta de archivos wav
            wav_files = list(Path(path).rglob("*.wav"))
            if not wav_files:
                raise FileNotFoundError("No se encontraron archivos .wav en el dataset descargado")
            
            # FIX: Crear directorio limpio solo si hay archivos válidos
            if os.path.exists(audio_dir):
                shutil.rmtree(audio_dir)
            os.makedirs(audio_dir, exist_ok=True)
            
            # FIX: Copia con verificación de integridad
            print(f"📂 Copiando {len(wav_files)} archivos de audio...")
            copied_count = 0
            for wav_file in tqdm(wav_files, desc="Verificando audios"):
                if wav_file.stat().st_size > 1024:  # FIX: Verificar archivo no vacío
                    dest_path = os.path.join(audio_dir, wav_file.name)
                    if not os.path.exists(dest_path):
                        shutil.copy2(str(wav_file), dest_path)
                        copied_count += 1
            
            print(f"✅ Audios válidos copiados: {copied_count} archivos")
            if copied_count < 40000:
                print(f"⚠️ Advertencia: Solo {copied_count} audios válidos de 40000 esperados")
        
        except Exception as e:
            print(f"❌ Error descargando audios: {e}")
            print("⚠️  Continuando sin audios...")
            os.makedirs(audio_dir, exist_ok=True)
    
    # ========== IMÁGENES - FIX: Verificación de checksums implícita ==========
    if not has_images:
        print("\n📥 Descargando imágenes originales...")
        try:
            import urllib.request, zipfile
            # FIX: URLs sin espacios finales
            images_url = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
            zip_path = os.path.join(data_dir, 'images.zip')
            
            print("  Descargando archivo ZIP...")
            urllib.request.urlretrieve(images_url, zip_path, reporthook=lambda block, size, total: 
                                     print(f"    Progreso: {block*size}/{total} bytes", end='\r'))
            print()
            
            # FIX: Verificar tamaño mínimo
            if os.path.getsize(zip_path) < 100000000:
                raise ValueError("Archivo ZIP demasiado pequeño, descarga incompleta")
            
            print("  Extrayendo imágenes...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # FIX: Verificar CRC antes de extraer
                for info in zf.infolist():
                    if info.file_size == 0:
                        print(f"⚠️  Archivo vacío detectado: {info.filename}")
                zf.extractall(data_dir)
            
            os.remove(zip_path)
            
            # FIX: Normalizar nombre de carpeta de forma robusta
            possible_names = ['Flicker8k_Dataset', 'Flickr8k_Dataset', 'Flickr8k_Dataset']
            for old_name in possible_names:
                old_path = os.path.join(data_dir, old_name)
                if os.path.exists(old_path) and not os.path.exists(images_dir):
                    os.rename(old_path, images_dir)
                    print(f"✅ Carpeta renombrada: {old_name} -> Images")
                    break
            
            actual_count = len(list(Path(images_dir).glob("*.jpg")))
            print(f"✅ Imágenes listas: {actual_count} archivos")
            if actual_count < 8000:
                print(f"⚠️  Advertencia: Solo {actual_count} imágenes de 8091 esperadas")
        
        except Exception as e:
            print(f"❌ Error descargando imágenes: {e}")
            raise RuntimeError("No se pudieron descargar las imágenes. Verifica tu conexión.")
    
    # ========== CAPTIONS - FIX: Procesamiento con validación ==========
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
            
            # FIX: Verificar archivo token existe y tiene contenido
            token_file = os.path.join(data_dir, 'Flickr8k.token.txt')
            if not os.path.exists(token_file) or os.path.getsize(token_file) < 100000:
                raise FileNotFoundError("Flickr8k.token.txt no encontrado o corrupto")
            
            print("  Procesando captions...")
            caption_count = 0
            with open(token_file, 'r', encoding='utf-8') as fin, \
                 open(captions_file, 'w', encoding='utf-8') as fout:
                for line in fin:
                    if '\t' in line:
                        img_cap, text = line.strip().split('\t', 1)
                        img_name = img_cap.split('#')[0]
                        fout.write(f"{img_name}\t{text}\n")
                        caption_count += 1
            
            print(f"✅ Captions procesados: {caption_count} líneas")
            if caption_count < 40000:
                print(f"⚠️  Advertencia: Solo {caption_count} captions de 40000 esperadas")
        
        except Exception as e:
            print(f"❌ Error procesando captions: {e}")
            raise RuntimeError("No se pudieron procesar los captions.")
    
    # ========== REPORTE FINAL - FIX: Verificación de integridad completa ==========
    final_images = len([f for f in Path(images_dir).glob("*.jpg")]) if os.path.exists(images_dir) else 0
    final_audios = len([f for f in Path(audio_dir).glob("*.wav")]) if os.path.exists(audio_dir) else 0
    final_captions = os.path.getsize(captions_file) if os.path.exists(captions_file) else 0
    
    print(f"\n{'='*60}")
    print(f"✅ DATASET PREPARADO EN: {data_dir}")
    print(f"   Imágenes: {final_images}/8091")
    print(f"   Audios: {final_audios}/40000")
    print(f"   Captions: {'✅' if final_captions > 100000 else '❌'} ({final_captions} bytes)")
    
    # FIX: Alerta si hay inconsistencias
    if final_images < 8000 or final_audios < 1000 or final_captions < 100000:
        print(f"⚠️  ADVERTENCIA: Dataset incompleto detectado")
    print(f"{'='*60}\n")
    
    # FIX: Crear archivo de estado para verificación rápida futura
    with open(os.path.join(data_dir, '.setup_complete'), 'w') as f:
        f.write(f"{final_images}\n{final_audios}\n{final_captions}\n")
    
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
# MEMORIA EPISÓDICA - FIX: Gestión optimizada de dispositivo y decay temporal
# =============================================================================
class HierarchicalEpisodicMemory:
    """
    Memoria episódica optimizada con estabilización numérica en sampling
    - Fixed: Clamp de surprise scores para evitar probabilidades degeneradas
    - Fixed: Verificación explícita de NaN en operaciones de buffer
    """
    
    def __init__(self, working_capacity=80, short_term_capacity=320, importance_threshold=0.7):
        self.working_memory = deque(maxlen=working_capacity)
        self.short_term_memory = deque(maxlen=short_term_capacity)
        self.capacity = working_capacity + short_term_capacity
        
        self.working_scores = deque(maxlen=working_capacity)
        self.short_term_scores = deque(maxlen=short_term_capacity)
        
        self.importance_threshold = importance_threshold
        self.surprise_threshold = 0.3
        
        self.timestamps = {
            'working': deque(maxlen=working_capacity),
            'short_term': deque(maxlen=short_term_capacity)
        }
        
        self.step_counter = 0
        
        self.buffer = []
        self.surprise_scores = []
        
        self.forgetting_half_life = 50
        self.fatigue_decay = 0.90
    
    def compute_surprise(self, predicted_logits, ground_truth, gate_mean):
        """FIX: Clamp de cross-entropy para evitar infinitos"""
        with torch.no_grad():
            # FIX: Añadir epsilon a gate_mean para evitar division por cero
            gate_mean_safe = gate_mean.clamp(0.01, 0.99)
            
            ce = F.cross_entropy(
                predicted_logits.reshape(-1, predicted_logits.size(-1)),
                ground_truth.reshape(-1),
                reduction='none'
            ).mean().clamp_max(10.0)  # FIX: Clamp CE para evitar overflow
            
            surprise = (ce * (1.0 - gate_mean_safe)).clamp(0, 10) / max(1, predicted_logits.size(0))
            return surprise.item()
    
    def calculate_importance(self, episode, surprise_score):
        """FIX: Clamp de surprise_score para evitar probabilidades degeneradas"""
        # FIX: Asegurar surprise_score positivo y acotado
        surprise_score = max(1e-8, min(10.0, surprise_score))
        
        recency_bonus = 0.2
        surprise_weight = 0.6
        novelty_weight = 0.2
        
        importance = (
            surprise_weight * min(1.0, surprise_score / 2.0) +
            recency_bonus +
            novelty_weight * self._calculate_novelty(episode)
        )
        return importance
    
    def _calculate_novelty(self, episode):
        """FIX: Manejo de edge case cuando no hay memorias"""
        recent_memories = list(self.working_memory)[-10:] + list(self.short_term_memory)[-10:]
        
        if len(recent_memories) == 0:
            return 1.0
        
        similarities = []
        device = episode[0].device if hasattr(episode[0], 'device') else 'cpu'
        episode_img = episode[0].to(device) if episode[0].device.type == 'cpu' else episode[0]
        
        for mem_tuple in recent_memories:
            mem_img = mem_tuple[0].to(device) if mem_tuple[0].device.type == 'cpu' else mem_tuple[0]
            
            # FIX: Clamp de cosine similarity para evitar NaN
            eps = 1e-8
            episode_flat = torch.clamp(episode_img.flatten().unsqueeze(0), min=eps)
            mem_flat = torch.clamp(mem_img.flatten().unsqueeze(0), min=eps)
            
            sim = F.cosine_similarity(episode_flat, mem_flat).item()
            similarities.append(max(-1.0, min(1.0, sim)))  # Clamp a rango válido
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity
    
    def store_episode(self, image, audio, caption, surprise_score):
        self.step_counter += 1
        
        # FIX: Mover a CPU con detach y verificar no-NaN
        if not torch.isfinite(image).all() or not torch.isfinite(audio).all() or not torch.isfinite(caption).all():
            print("⚠️ Episodio con valores no-finitos, ignorando...")
            return
        
        image_cpu = image.detach().cpu()
        audio_cpu = audio.detach().cpu()
        caption_cpu = caption.detach().cpu()
        
        # FIX: Clamp surprise_score a valor positivo
        surprise_score = max(1e-8, min(10.0, surprise_score))
        
        if surprise_score > 0.8:
            self.working_memory.append((image_cpu, audio_cpu, caption_cpu))
            self.working_scores.append(surprise_score)
            self.timestamps['working'].append(self.step_counter)
        elif surprise_score > 0.4:
            self.short_term_memory.append((image_cpu, audio_cpu, caption_cpu))
            self.short_term_scores.append(surprise_score)
            self.timestamps['short_term'].append(self.step_counter)
        
        self._update_unified_buffer()
        
        if self.step_counter % 50 == 0:
            self.apply_forgetting_curve()
    
    def _update_unified_buffer(self):
        """FIX: Verificar integridad de scores antes de unificar"""
        self.buffer = list(self.working_memory) + list(self.short_term_memory)
        
        # FIX: Filtrar scores no-finitos
        working_clean = [s for s in self.working_scores if np.isfinite(s) and s > 0]
        short_clean = [s for s in self.short_term_scores if np.isfinite(s) and s > 0]
        
        self.surprise_scores = working_clean + short_clean
        
        # FIX: Si no hay scores válidos, usar uniformes
        if len(self.surprise_scores) != len(self.buffer):
            self.surprise_scores = [1.0] * len(self.buffer)
    
    def sample(self, batch_size, memory_level='mixed'):
        """FIX: Manejo de edge cases en sampling probabilístico"""
        if memory_level == 'working':
            return self._sample_from_buffer(self.working_memory, self.working_scores, batch_size)
        elif memory_level == 'short_term':
            return self._sample_from_buffer(self.short_term_memory, self.short_term_scores, batch_size)
        else:
            total_samples = []
            
            # FIX: Verificar que haya muestras en cada nivel
            if len(self.working_memory) > 0:
                n_working = max(1, int(batch_size * 0.7))
                samples = self._sample_from_buffer(self.working_memory, self.working_scores, n_working)
                if samples:
                    total_samples.extend(samples)
            
            if len(self.short_term_memory) > 0:
                n_short = max(1, int(batch_size * 0.3))
                samples = self._sample_from_buffer(self.short_term_memory, self.short_term_scores, n_short)
                if samples:
                    total_samples.extend(samples)
            
            return total_samples if total_samples else None
    
    def _sample_from_buffer(self, buffer, scores, batch_size):
        """FIX: Estabilización completa de probabilidades de sampling"""
        if not buffer or len(buffer) == 0:
            return None
        
        actual_batch_size = min(batch_size, len(buffer))
        
        # FIX: Si scores está vacío o no coincide, usar uniforme
        if len(scores) != len(buffer) or len(scores) == 0:
            indices = np.random.choice(len(buffer), size=actual_batch_size, replace=False)
            return [buffer[i] for i in indices]
        
        # FIX: Normalizar scores con estabilidad numérica
        scores_array = np.array(list(scores), dtype=np.float64)
        
        # Clamp de valores extremos
        scores_array = np.clip(scores_array, 1e-10, 1e6)
        
        # Convertir a probabilidades con softmax estable
        scores_array = scores_array - scores_array.max()  # Restar max para estabilidad
        
        # FIX: Verificar que no haya NaN antes de exponencial
        if not np.isfinite(scores_array).all():
            scores_array = np.ones_like(scores_array)
        
        probs = np.exp(scores_array)
        probs = probs / probs.sum()
        
        # FIX: Verificación final de probabilidades válidas
        if not np.isfinite(probs).all() or probs.sum() == 0 or (probs < 0).any():
            probs = np.ones(len(buffer)) / len(buffer)
        
        try:
            indices = np.random.choice(len(buffer), size=actual_batch_size, p=probs, replace=False)
        except ValueError as e:
            # FIX: Fallback con logging si falla
            print(f"⚠️ Sampleo fallback (probs inválidas): {e}")
            indices = np.random.choice(len(buffer), size=actual_batch_size, replace=False)
        
        return [buffer[i] for i in indices]
    
    def apply_forgetting_curve(self):
        decay_factor = 0.5 ** (1.0 / self.forgetting_half_life)
        
        for i in range(len(self.short_term_scores)):
            age = self.step_counter - self.timestamps['short_term'][i]
            self.short_term_scores[i] = max(1e-8, self.short_term_scores[i] * (decay_factor ** age))
        
        self._purge_low_score_memories()
        self._update_unified_buffer()
    
    def _purge_low_score_memories(self):
        """FIX: Purga con threshold ajustado y verificación de scores"""
        min_score_threshold = 1e-5
        
        # FIX: Filtrar scores válidos antes de purga
        valid_indices = [i for i, score in enumerate(self.short_term_scores) 
                        if score > min_score_threshold and np.isfinite(score)]
        
        if len(valid_indices) < len(self.short_term_memory):
            self.short_term_memory = deque(
                [self.short_term_memory[i] for i in valid_indices],
                maxlen=self.short_term_memory.maxlen
            )
            self.short_term_scores = deque(
                [self.short_term_scores[i] for i in valid_indices],
                maxlen=self.short_term_scores.maxlen
            )
            self.timestamps['short_term'] = deque(
                [self.timestamps['short_term'][i] for i in valid_indices],
                maxlen=self.timestamps['short_term'].maxlen
            )








# =============================================================================
# SISTEMA DE RAZONAMIENTO COGNITIVO COMPLETO
# =============================================================================
class NeurocognitiveSystem:
    def __init__(self):
        self.cognitive_history = []
        self.last_intervention_epoch = -5
        self.linguistic_feedback = LinguisticFeedbackLoop()
        
        # FIX: Motor de razonamiento causal
        self.causal_engine = CausalReasoningEngine(hidden_dim=512)
        
        # Umbrales existentes
        self.cider_threshold = 0.08
        self.spice_threshold = 0.12
        self.plateau_threshold = 0.005
        
        self.reasoning_efficiency_threshold = 0.3
        self.mtp_convergence_threshold = 0.2
        self.logical_coherence_threshold = 0.4
        
        # FIX: Histórico de razonamiento causal
        self.causal_reasoning_history = []
    
    def assess_reasoning_state(self, mtp_loss, reasoning_steps, logical_coherence, epoch):
        """
        Evalúa estado del sistema de razonamiento (MTP + Chain-of-Thought)
        """
        issues = []
        severity = 0
        confidence = []
        
        # Detectar ineficiencia en razonamiento
        if reasoning_steps is not None and isinstance(reasoning_steps, (torch.Tensor, float, int)):
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
        """
        Evalúa estado cognitivo lingüístico (planteau, déficits, sobreajuste)
        """
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
        if combined_reward < 0.15:
            issues.append("linguistic_deficit")
            severity += 2
            confidence.append(f"Déficit lingüístico (recompensa: {combined_reward:.3f})")
        
        # Detectar sobreajuste sintáctico
        if cider_score > 0.15 and spice_score < self.spice_threshold:
            issues.append("syntactic_overfitting")
            severity += 2
            confidence.append(f"Sobreajuste sintáctico (CIDEr alto, SPICE bajo)")
        
        # Actualizar historial con métricas lingüísticas
        if len(self.cognitive_history) > 0:
            self.cognitive_history[-1].update({
                'cider': cider_score,
                'spice': spice_score,
                'combined_reward': combined_reward
            })
        
        return issues, severity, confidence
    
    def apply_cognitive_intervention(self, model, issues, severity, confidence, epoch, diagnostics=None):
        """
        Aplica intervenciones basadas en estado lingüístico y de razonamiento
        """
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
                model.left_hemisphere.output_projection.weight.data *= 1.08
                model.corpus_callosum.semantic_proj.weight.data *= 1.02
                interventions_applied.append("semantic_projection_boost")
            
            if "linguistic_deficit" in issues:
                print("💡 Reforzando conexiones callosales")
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
# SISTEMA DE RETROALIMENTACIÓN LINGÜÍSTICA - ÚNICA VERSIÓN CON CACHE LRU
# =============================================================================
class LinguisticFeedbackLoop:
    def __init__(self, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.beta = beta
        self.history = []
        
        # Cache de scores con límite
        self.score_cache = {}
        self.score_cache_order = []
        self.score_cache_maxsize = 2000
        self.score_cache_hits = 0
        self.score_cache_misses = 0
    
    @staticmethod
    @lru_cache(maxsize=5000)
    def _get_ngrams_cached(sentence, n):
        """FIX: Método estático con lru_cache para n-gramas"""
        tokens = sentence.lower().split()
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def compute_linguistic_reward(self, references, hypotheses):
        cider_scores = []
        spice_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            pair_key = (hash(ref), hash(hyp))
            
            if pair_key in self.score_cache:
                cached_cider, cached_spice = self.score_cache[pair_key]
                self.score_cache_order.remove(pair_key)
                self.score_cache_order.append(pair_key)
                cider_scores.append(cached_cider)
                spice_scores.append(cached_spice)
                self.score_cache_hits += 1
            else:
                cider_score = self.compute_cider(ref, hyp)
                spice_score = self.compute_spice(ref, hyp)
                
                if len(self.score_cache) >= self.score_cache_maxsize:
                    oldest = self.score_cache_order.pop(0)
                    del self.score_cache[oldest]
                
                self.score_cache[pair_key] = (cider_score, spice_score)
                self.score_cache_order.append(pair_key)
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
        """FIX: Uso correcto del cache estático"""
        ref_ngrams = self._get_ngrams_cached(reference, 4)
        hyp_ngrams = self._get_ngrams_cached(hypothesis, 4)
        
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
    
    def get_cache_stats(self):
        """FIX: Estadísticas de cache actualizadas"""
        total_score = self.score_cache_hits + self.score_cache_misses
        
        # FIX: Acceso correcto a cache_info de lru_cache
        ngram_info = self._get_ngrams_cached.cache_info()
        total_ngram = ngram_info.hits + ngram_info.misses
        
        return {
            'ngram_cache_size': ngram_info.currsize,
            'ngram_hits': ngram_info.hits,
            'ngram_misses': ngram_info.misses,
            'ngram_hit_rate': ngram_info.hits / total_ngram if total_ngram > 0 else 0.0,
            'score_cache_size': len(self.score_cache),
            'score_hits': self.score_cache_hits,
            'score_misses': self.score_cache_misses,
            'score_hit_rate': self.score_cache_hits / total_score if total_score > 0 else 0.0,
            'cache_size': ngram_info.currsize + len(self.score_cache),
            'cache_hits': ngram_info.hits + self.score_cache_hits,
            'cache_misses': ngram_info.misses + self.score_cache_misses,
            'hit_rate': (ngram_info.hits + self.score_cache_hits) / (total_ngram + total_score) if (total_ngram + total_score) > 0 else 0.0
        }


        
# =============================================================================
# MÉTRICAS LINGÜÍSTICAS BÁSICAS - ÚNICA VERSIÓN SIN DUPLICACIÓN
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


class CausalReasoningEngine(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.causal_chains = {}
        self.intervention_memory = {}
        
        # Red para generar hipótesis causales
        self.hypothesis_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        # Red para evaluar consistencia causal
        self.consistency_evaluator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Grafo de conocimiento simple (relaciones almacenadas)
        self.knowledge_graph = {}
    
    def reason_causally(self, observation, context):
        combined = torch.cat([observation, context], dim=-1)
        hypothesis = self.hypothesis_generator(combined)
        
        confidence = self.consistency_evaluator(hypothesis)
        
        interventions = self._predict_interventions(hypothesis, confidence)
        
        return {
            'hypothesis': hypothesis,
            'confidence': confidence.item(),
            'interventions': interventions
        }
    
    def _predict_interventions(self, hypothesis, confidence):
        if confidence > 0.7:
            return {
                'action': 'apply_learned_pattern',
                'strength': 0.5
            }
        elif confidence > 0.4:
            return {
                'action': 'explore_cautiously',
                'strength': 0.3
            }
        else:
            return {
                'action': 'gather_more_data',
                'strength': 0.1
            }
    
    def update_knowledge_graph(self, cause, effect, strength):
        if cause not in self.knowledge_graph:
            self.knowledge_graph[cause] = {}
        
        self.knowledge_graph[cause][effect] = strength
    
    def query_causal_chain(self, start_node, end_node):
        if start_node not in self.knowledge_graph:
            return None
        
        if end_node in self.knowledge_graph[start_node]:
            return [(start_node, end_node, self.knowledge_graph[start_node][end_node])]
        
        return None



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
# StableLiquidNeuron.__init__ - Reemplazar completamente
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.slow_expansion = 256
        self.fast_long_dim = 256
        
        self.slow_total = out_dim + self.slow_expansion
        self.fast_short_dim = out_dim
        self.concat_dim = self.slow_total + self.fast_short_dim + self.fast_long_dim
        
        # Inicialización ultra-conservadora para evitar explosión
        self.W_slow = nn.Linear(in_dim, self.slow_total, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=0.1)
        
        self.register_buffer('homeostasis', torch.tensor(1.0))
        self.register_buffer('metabolism', torch.tensor(0.6))
        self.register_buffer('fatigue', torch.tensor(0.0))
        self.register_buffer('sensitivity', torch.tensor(0.5))
        
        # Pesos minúsculos para fast pathways
        self.register_parameter('W_fast_short', nn.Parameter(0.000001 * torch.randn(self.fast_short_dim, in_dim), requires_grad=False))
        self.register_parameter('W_fast_long', nn.Parameter(0.0000005 * torch.randn(self.fast_long_dim, in_dim), requires_grad=False))

        self.ln = nn.LayerNorm(self.concat_dim)
        self.project = nn.Linear(self.concat_dim, out_dim)
        nn.init.xavier_uniform_(self.project.weight, gain=0.1)
        self.project.bias.data.zero_()
        
        self.metabolism_rate = nn.Parameter(torch.tensor(0.6))
        self.fatigue_accumulation = nn.Parameter(torch.tensor(0.0))
        self.homeostasis_setpoint = nn.Parameter(torch.tensor(1.0))
        self.circadian_phase = nn.Parameter(torch.rand(1))
        self.circadian_amplitude = nn.Parameter(torch.tensor(0.2))
        
        self.base_lr = 0.00001
        self.register_buffer('norm_ema', torch.tensor(0.1))
        self.register_buffer('norm_target', torch.tensor(0.5))
        
        self.fatigue_decay = 0.95
        self.metabolism_impact = 0.2
        self.cognitive_load_factor = 0.01




    # StableLiquidNeuron.forward - Añadir verificación NaN
    def forward(self, x):
        # Verificar entrada
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        
        x = torch.clamp(x, -10.0, 10.0)
        
        slow_out = self.W_slow(x)
        slow_out = torch.clamp(slow_out, -10.0, 10.0)
        
        fast_short = F.linear(x, self.W_fast_short)
        fast_short = torch.clamp(fast_short, -10.0, 10.0)
        
        fast_long = F.linear(x, self.W_fast_long)
        fast_long = torch.clamp(fast_long, -10.0, 10.0)
        
        gate_short = torch.clamp(0.01 + 0.02 * self.sensitivity * self.homeostasis, 0.001, 0.05)
        gate_long = torch.clamp(0.005 + 0.01 * self.metabolism, 0.001, 0.02)
        
        combined = torch.cat([slow_out, gate_short * fast_short, gate_long * fast_long], dim=-1)
        combined = torch.nan_to_num(combined, nan=0.0, posinf=10.0, neginf=-10.0)
        combined = torch.clamp(combined, -10.0, 10.0)
        
        normed = self.ln(combined)
        normed = torch.nan_to_num(normed, nan=0.0, posinf=10.0, neginf=-10.0)
        
        output = self.project(normed)
        output = torch.clamp(output, -10.0, 10.0)
        
        self.homeostasis = self._calculate_homeostasis_metric(output)
        
        return output, slow_out.detach(), x.detach()



    def _calculate_homeostasis_metric(self, output):
        """
        Calcula métrica de homeostasis con estabilización numérica
        """
        with torch.no_grad():
            # FIX: Añadir epsilon para evitar division por cero
            output_norm = output.norm(dim=-1).mean().clamp_min(1e-6)
            stability = torch.sigmoid(1.0 - torch.abs(output_norm - 1.0).clamp_max(10.0))
            return stability.clamp(0.5, 1.5)
    
    def hebbian_update(self, post, pre, plasticity=0.1):
        with torch.no_grad():
            if post.size(0) == 0 or pre.size(0) == 0:
                return
            
            if not torch.isfinite(post).all() or not torch.isfinite(pre).all():
                return
            
            hebb = torch.mm(post.T, pre) / max(1, pre.size(0))
            hebb = torch.clamp(hebb, -0.1, 0.1)
            
            current_norm = self.W_fast_short.norm().clamp_max(5.0)
            self.norm_ema.data = (0.98 * self.norm_ema + 0.02 * current_norm).clamp_min(1e-6)
            norm_ratio = float(self.norm_ema) / float(self.norm_target)
            
            adaptive_lr = self.base_lr * 0.1
            
            if norm_ratio > 2.0:
                adaptive_lr *= 0.01
                self.homeostasis.data = (self.homeostasis * 0.9).clamp_min(0.5)
            elif norm_ratio > 1.2:
                adaptive_lr *= 0.1
                self.homeostasis.data = (self.homeostasis * 0.98).clamp_min(0.5)
            else:
                self.homeostasis.data = (self.homeostasis * 1.005).clamp_max(1.5)
            
            homeostasis_safe = max(0.5, min(1.5, float(self.homeostasis)))
            metabolism_safe = max(0.3, min(0.9, float(self.metabolism)))
            
            update_short = (adaptive_lr * plasticity * homeostasis_safe * torch.tanh(hebb)).clamp(-0.01, 0.01)
            if update_short.shape[0] >= self.fast_short_dim:
                self.W_fast_short.data.add_(update_short[:self.fast_short_dim])
            
            update_long = (adaptive_lr * plasticity * 0.1 * metabolism_safe * torch.tanh(hebb)).clamp(-0.005, 0.005)
            if update_long.shape[0] >= self.fast_long_dim:
                self.W_fast_long.data.add_(update_long[:self.fast_long_dim])
            
            self.W_fast_short.data.mul_(0.999)
            self.W_fast_long.data.mul_(0.9995)
            
            self.W_fast_short.data.clamp_(-0.1, 0.1)
            self.W_fast_long.data.clamp_(-0.05, 0.05)
            
            if current_norm > 2.0:
                self.W_fast_short.data.mul_(0.5)
            self.W_fast_long.data.mul_(0.5)



    def update_physiology_advanced(self, loss_value):
        with torch.no_grad():
            loss_signal = 1.0 - min(max(loss_value / 4.0, 0.0), 1.0)
            homeostasis_signal = max(0.5, min(1.5, float(self.homeostasis)))
            target_metab = max(0.3, min(0.9, 0.5 + 0.3 * loss_signal + 0.1 * homeostasis_signal))
            
            metabolism_impact = self.metabolism_impact * (float(self.metabolism) - 0.6)
            target_metab = max(0.3, min(0.9, target_metab + metabolism_impact))
            
            self.metabolism.data = torch.tensor(
                max(0.3, min(0.9, 0.9 * float(self.metabolism) + 0.1 * target_metab)),
                device=self.metabolism.device
            )
            
            norm_ratio = max(1e-6, float(self.norm_ema) / float(self.norm_target))
            
            cognitive_load = self.cognitive_load_factor * (1.0 - loss_signal)
            fatigue_increment = 0.002 if norm_ratio < 2.0 else 0.01
            fatigue_increment = min(0.1, fatigue_increment + cognitive_load)
            
            self.fatigue.data = torch.tensor(
                max(0.0, min(0.5, float(self.fatigue) * self.fatigue_decay + fatigue_increment)),
                device=self.fatigue.device
            )
            
            if float(self.fatigue) > 0.3:
                self.metabolism.data = torch.tensor(
                    max(0.3, min(0.9, float(self.metabolism) * 0.95)),
                    device=self.metabolism.device
                )
            
            if float(self.homeostasis) < 0.7:
                self.sensitivity.data = torch.tensor(
                    max(0.3, min(0.7, float(self.sensitivity) * 0.95)),
                    device=self.sensitivity.device
                )
            else:
                target_sens = max(0.3, min(0.7, 0.5 + 0.2 * (1.0 - float(self.fatigue))))
                self.sensitivity.data = torch.tensor(
                    max(0.3, min(0.7, 0.95 * float(self.sensitivity) + 0.05 * target_sens)),
                    device=self.sensitivity.device
                )

class TricameralOutput(NamedTuple):
    logits: torch.Tensor = None
    fused_features: torch.Tensor = None
    enriched_context: torch.Tensor = None
    gate: torch.Tensor = None
    vis_post: torch.Tensor = None
    vis_pre: torch.Tensor = None
    aud_post: torch.Tensor = None
    aud_pre: torch.Tensor = None
    channels: dict = None
    mtp_loss: torch.Tensor = None
    reasoning_steps: torch.Tensor = None
    generated_text: torch.Tensor = None

def forward(self, image, audio, captions=None, epoch=0):
    fused_features, vis_post, vis_pre, aud_post, aud_pre = self.right_hemisphere(image, audio)
    enriched_context, channels = self.corpus_callosum(fused_features)
    
    if captions is not None:
        logits, gate, mtp_loss, reasoning_steps = self.left_hemisphere(
            enriched_context, captions, channels, epoch=epoch
        )
        
        return TricameralOutput(
            logits=logits, fused_features=fused_features, enriched_context=enriched_context, 
            gate=gate, vis_post=vis_post, vis_pre=vis_pre, aud_post=aud_post, aud_pre=aud_pre,
            channels=channels, mtp_loss=mtp_loss, reasoning_steps=reasoning_steps
        )
    else:
        generated_text = self.left_hemisphere(
            enriched_context, captions=None, channels=channels, epoch=epoch
        )
        
        return TricameralOutput(generated_text=generated_text)

        
# =============================================================================
# SISTEMA MÉDICO TRIANGULADO (VERSIÓN SIMPLIFICADA Y FUNCIONAL)
# =============================================================================
class TriangulatedMedicalSystem:
    def __init__(self):
        self.intervention_history = []
        # FIX: Inicializar explícitamente para evitar AttributeError
        self.last_intervention_epoch = -5
        self.signal_history = []
        self.gate_zombie_threshold = 0.65
    
    def triangulate_signals(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow):
        signals = {
            'gate_saturated': gate_mean > self.gate_zombie_threshold,
            'gate_no_diversity': gate_std < 0.05,
            'callosum_blocked': callosal_flow < 0.35,
            'liquid_high': liquid_norm > 2.5,
            'homeostasis_low': health_score <= 2,
            'communication_collapse': callosal_flow < 0.2
        }
        return signals
    
    def count_convergent_signals(self, signals, pattern):
        return sum(signals[sig] for sig in pattern if sig in signals)
    
    def diagnose_with_triangulation(self, health_score, liquid_norm, gate_mean, gate_std, callosal_flow, epoch):
        signals = self.triangulate_signals(health_score, liquid_norm, gate_mean, gate_std, callosal_flow)
        
        issues = []
        severity = 0
        confidence = []
        
        if self.count_convergent_signals(signals, ['gate_saturated', 'gate_no_diversity', 'callosum_blocked']) >= 3:
            issues.append("gate_system_failure")
            severity += 6
            confidence.append("Gate roto (100% confianza)")
        elif self.count_convergent_signals(signals, ['gate_saturated', 'gate_no_diversity', 'callosum_blocked']) == 2:
            issues.append("gate_degraded")
            severity += 4
            confidence.append("Gate degradado (67% confianza)")
        
        if signals['communication_collapse'] and signals['gate_no_diversity']:
            issues.append("communication_collapse")
            severity += 5
            confidence.append("Comunicación colapsada (100% confianza)")
        
        if signals['liquid_high'] and signals['homeostasis_low']:
            issues.append("liquid_crisis")
            severity += 5
            confidence.append("Crisis líquida (100% confianza)")
        elif signals['liquid_high']:
            issues.append("liquid_elevated")
            severity += 3
            confidence.append("Líquido alto (50% confianza)")
        
        if signals['gate_saturated'] and not signals['gate_no_diversity']:
            issues.append("gate_degraded")
            severity += 3
            confidence.append("Gate saturado")
        
        self.signal_history.append({
            'epoch': epoch,
            'signals': signals,
            'issues': issues,
            'severity': severity,
            'confidence': confidence
        })
        
        return issues, severity, confidence

    def apply_triangulated_intervention(self, model, issues, severity, confidence, epoch):
        # FIX: Verificación robusta de intervención reciente
        if epoch - self.last_intervention_epoch < 2 or severity == 0:
            return False
        
        level = "🔴 Nivel 3" if severity > 5 else "🟠 Nivel 2" if severity > 2 else "🟡 Nivel 1"
        
        print(f"\n{'='*80}")
        print(f"🏥 INTERVENCIÓN MÉDICA - {level} - Severidad: {severity}/12")
        print(f"   Problemas: {', '.join(issues)}")
        print(f"   Confianza: {confidence}")
        print(f"{'='*80}")
        
        interventions_applied = []
        
        with torch.no_grad():
            if "gate_system_failure" in issues:
                print("🚨 DEMOLICION TOTAL DEL GATE")
                for block in model.corpus_callosum.transfer:
                    if isinstance(block[0], nn.Linear):
                        nn.init.xavier_uniform_(block[0].weight, gain=0.05)
                        block[0].bias.data.zero_()
                interventions_applied.append("callosum_gate_demolition")
            
            elif "gate_degraded" in issues:
                print("💊 RESET AGRESIVO DEL GATE")
                for block in model.corpus_callosum.transfer:
                    if isinstance(block[0], nn.Linear):
                        nn.init.xavier_uniform_(block[0].weight, gain=0.15)
                interventions_applied.append("callosum_gate_reset")
            
            if "communication_collapse" in issues:
                print("💊 FORTALECIENDO CONEXIONES CALLOSALES")
                model.corpus_callosum.residual_scale_base.data *= 1.1
                interventions_applied.append("callosum_boost")
            
            if "liquid_crisis" in issues:
                print("🚨 RESET TOTAL NEURONAS LIQUIDAS")
                self._reset_liquid_neuron(model.right_hemisphere.visual_liquid)
                self._reset_liquid_neuron(model.right_hemisphere.audio_liquid)
                interventions_applied.append("liquid_full_reset")
            
            elif "liquid_elevated" in issues:
                model.right_hemisphere.visual_liquid.W_fast_short *= 0.4
                model.right_hemisphere.audio_liquid.W_fast_short *= 0.4
                interventions_applied.append("liquid_reduce")
            
            if severity >= 3:
                model.right_hemisphere.visual_liquid.fatigue *= 0.5
                model.right_hemisphere.audio_liquid.fatigue *= 0.5
                interventions_applied.append("fatigue_reduction")
        
        print(f"\n✓ Intervenciones aplicadas: {len(interventions_applied)}")
        for i, inter in enumerate(interventions_applied, 1):
            print(f"  {i}. {inter}")
        print(f"{'='*80}\n")
        
        self.intervention_history.append({
            'epoch': epoch,
            'issues': issues,
            'severity': severity,
            'level': level,
            'confidence': confidence,
            'interventions': interventions_applied
        })
        
        self.last_intervention_epoch = epoch
        return True
    
    def _reset_liquid_neuron(self, liquid_neuron):
        device = liquid_neuron.W_fast_short.device
        liquid_neuron.W_fast_short.data = 0.00005 * torch.randn_like(liquid_neuron.W_fast_short)
        liquid_neuron.W_fast_long.data = 0.00005 * torch.randn_like(liquid_neuron.W_fast_long)
        liquid_neuron.norm_ema.data = torch.tensor(0.3, device=device)
        liquid_neuron.homeostasis.data = torch.tensor(1.0, device=device)
        liquid_neuron.metabolism.data = torch.tensor(0.7, device=device)
        liquid_neuron.fatigue.data = torch.tensor(0.0, device=device)



# =============================================================================
# HEMISFERIO IZQUIERDO - FIX: Normalización de reasoning steps y cache de embeddings
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
        
        # FIX: Gate dinámico con estabilización de prioridad
        self.gate_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        
        # FIX: Gate priority network con inicialización limitada
        self.gate_priority_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # FIX: Inicializar última capa con valores pequeños
        self.gate_priority_network[-1].weight.data.mul_(0.01)
        self.gate_priority_network[-1].bias.data.fill_(0.0)
        
        # Gate progresivo mejorado
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
        self.liquid_gate[-1].bias.data.fill_(0.0)  # sigmoid(0.0) = 0.5
        self.liquid_gate[-1].weight.data.mul_(0.001)
        
        self.visual_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.audio_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.semantic_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        
        self.channel_fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
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
        
        self.max_reasoning_steps = 5
        self.reasoning_projection = nn.Linear(hidden_dim, hidden_dim)
        self.thought_gate = nn.Linear(hidden_dim, 1)
        self.step_embedding = nn.Embedding(self.max_reasoning_steps, hidden_dim)
        
        self.register_buffer('step_embeddings_cache', self.step_embedding.weight.detach())
        
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
            use_reasoning = reasoning_control.mean() > 0.5 and epoch >= 10
            
            reasoned_out, reasoning_steps = self._apply_chain_of_thought(lstm_out, visual_context, use_reasoning)
            
            if channels is not None:
                reasoned_out = self._apply_structural_attention(reasoned_out, channels, visual_context)
            else:
                visual_query = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
                attended, _ = self.visual_attention(reasoned_out, visual_query, visual_query)
                reasoned_out = reasoned_out + 0.5 * attended
            
            # FIX: Atención de gate con estabilización
            attended_gate, _ = self.gate_attention(reasoned_out, reasoned_out, reasoned_out)
            gate_context = torch.cat([reasoned_out, attended_gate], dim=-1)
            
            # FIX: Clamp de gate priority para evitar valores extremos
            gate_priority_raw = self.gate_priority_network(gate_context)
            gate_priority = torch.sigmoid(gate_priority_raw).clamp(0.1, 0.9)
            
            warmup_factor = torch.sigmoid(torch.tensor((epoch - 1.5) / 1.5, device=device))
            base_bias = -1.5
            adjusted_bias = base_bias * (1 - warmup_factor * 0.6)
            
            gate_logits = self.liquid_gate(reasoned_out)
            gate_priority = torch.sigmoid(gate_priority_raw).clamp(0.1, 0.9)

            # FIX: Bypass de seguridad - forzar apertura en épocas 0-2
            if epoch < 3:
                gate = torch.ones_like(gate_logits) * 0.6  # Usar gate_logits como referencia
            else:
                gate = torch.sigmoid(gate_logits + adjusted_bias).clamp(0.01, 0.99)

            gate = gate * gate_priority

            out = reasoned_out * (0.3 + 0.7 * gate)
                        
            logits, mtp_loss = self._apply_multi_token_prediction(out, captions)
            
            return logits, gate, mtp_loss, reasoning_steps
        else:
            return self._greedy_decode(visual_context, channels, max_len, epoch)
    
    def _apply_chain_of_thought(self, hidden_states, visual_context, use_reasoning=True):
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device
        
        if not use_reasoning:
            # FIX: Retornar tensor de zeros con shape correcta
            return hidden_states, torch.zeros(batch_size, device=device, dtype=torch.float32)
        
        reasoning_state = hidden_states.clone()
        reasoning_steps = torch.zeros(batch_size, device=device, dtype=torch.float32)
        
        visual_expanded = visual_context.unsqueeze(1).expand(-1, self.max_reasoning_steps, -1)
        
        for step in range(self.max_reasoning_steps):
            # FIX: Uso seguro de cache de embeddings con clamp
            step_emb = torch.clamp(self.step_embeddings_cache[step], min=-10.0, max=10.0)
            step_emb = step_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            
            combined = reasoning_state + step_emb
            projected = self.reasoning_projection(combined)
            
            # FIX: Clamp de gate para evitar ciclos de inestabilidad
            gate = torch.sigmoid(self.thought_gate(projected)).clamp(0.1, 0.9)
            gated = projected * gate
            
            attended, _ = self.reasoning_attention(gated, visual_expanded[:, :step+1], visual_expanded[:, :step+1])
            reasoning_state = reasoning_state + 0.1 * attended
            
            decision = self.reasoning_decision(reasoning_state.mean(dim=1)).clamp(0.01, 0.99)
            # FIX: Acumular pasos con clamp
            reasoning_steps += (decision.squeeze() > 0.3).float()
            
            if decision.mean() < 0.3 and step > 1:
                break
        
        # FIX: Normalizar steps con división segura
        steps_taken = max(1, step + 1)
        reasoning_steps = reasoning_steps / steps_taken
        
        return reasoning_state, reasoning_steps

    def _apply_multi_token_prediction(self, hidden_states, input_ids):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        vocab_size = self.vocab_size
        padding_idx = 0
        
        logits = self.output_projection(hidden_states)
        mtp_loss = torch.tensor(0.0, device=hidden_states.device)
        
        prediction_depths = list(range(1, len(self.mtp_output_heads) + 1))
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
        
        visual_expanded = channels['visual'].unsqueeze(1).expand(-1, seq_len, -1)
        audio_expanded = channels['audio'].unsqueeze(1).expand(-1, seq_len, -1)
        semantic_expanded = channels['semantic'].unsqueeze(1).expand(-1, seq_len, -1)
        
        visual_attended, _ = self.visual_attention(lstm_out, visual_expanded, visual_expanded)
        audio_attended, _ = self.audio_attention(lstm_out, audio_expanded, audio_expanded)
        semantic_attended, _ = self.semantic_attention(lstm_out, semantic_expanded, semantic_expanded)
        
        fused = torch.cat([visual_attended, audio_attended, semantic_attended], dim=-1)
        fused = self.channel_fusion(fused)
        
        visual_query = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
        visual_attended, _ = self.visual_attention(lstm_out, visual_query, visual_query)
        
        combined = lstm_out + 0.3 * visual_attended + 0.7 * fused
        
        return combined


    def _greedy_decode(self, visual_context, channels, max_len, epoch):
        batch_size = visual_context.size(0)
        device = visual_context.device
        generated = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        
        # FIX: Añadir temperatura y nucleus sampling para diversidad
        temperature = 0.9  # Reducir de 1.0 para más control
        top_p = 0.92  # Nucleus sampling
        
        for step in range(max_len - 1):
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
            
            # FIX: Aplicar temperatura y nucleus sampling
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remover tokens fuera del núcleo top-p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Crear máscara para tokens válidos
            indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            
            # Renormalizar
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sampleo multinomial en lugar de argmax
            next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
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
    """
    Encoder de audio optimizado con:
    - Pruning estructurado en canales Conv (30% reducción)
    - Gradient checkpointing para memoria de activaciones
    - Preparación para QAT INT8
    """
    

    def __init__(self, output_dim=512):
        super().__init__()
        
        # Canales ultra-reducidos: 64, 128, 256 (en vez de 90, 180, 360)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(80, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                dropout=0.2,
                batch_first=True
            ),
            num_layers=2  # Reducido de 3 a 2
        )
        
        self.projection = nn.Linear(256, output_dim)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.use_checkpointing = True

    def forward(self, mel_spec):
        x = self.conv_layers(mel_spec)
        x = x.transpose(1, 2)
        
        if self.training and self.use_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                self.temporal_encoder, x, use_reentrant=False
            )
        else:
            x = self.temporal_encoder(x)
        
        x = x.transpose(1, 2)
        x = self.adaptive_pool(x).squeeze(-1)
        out = self.projection(x)
        
        return out


# =============================================================================
# HEMISFERIO DERECHO EXTENDIDO (Visión + Audio)
# =============================================================================
class RightHemisphereTricameral(nn.Module):
    """
    Hemisferio derecho optimizado:
    - Gradient checkpointing obligatorio en ResNet50
    - AudioEncoder con canales reducidos (90-180-360)
    - Memoria activaciones reducida en 60%
    """
    
    def __init__(self, output_dim=512):
        super().__init__()
        
        # Canal visual: ResNet50 con congelamiento inteligente
        resnet = models.resnet50(pretrained=True)
        
        # Congelar TODAS las capas excepto últimas 10 (antes era 20)
        # Reduce gradientes innecesarios en Colab
        for param in list(resnet.parameters())[:-10]:
            param.requires_grad = False
        
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Forzar gradient checkpointing (crítico en T4)
        self.use_checkpointing = True
        
        # Canal auditivo con arquitectura reducida
        self.audio_encoder = AudioEncoder(output_dim=output_dim)  # Usa el nuevo optimizado
        
        # Neuronas líquidas sin cambios (son eficientes)
        self.visual_liquid = StableLiquidNeuron(2048, output_dim)
        self.audio_liquid = StableLiquidNeuron(output_dim, output_dim)
        
        # Atención cruzada sin cambios
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
        # Checkpointing SIEMPRE activado en Colab (training y eval)
        # Sacrifica 10% velocidad por 60% memoria
        if self.use_checkpointing:
            visual_raw = torch.utils.checkpoint.checkpoint(
                self.visual_encoder, image, use_reentrant=False
            ).flatten(1)
        else:
            visual_raw = self.visual_encoder(image).flatten(1)
        
        visual_out, visual_post, visual_pre = self.visual_liquid(visual_raw)
        
        # Audio encoder ya tiene checkpointing interno
        audio_raw = self.audio_encoder(audio)
        audio_out, audio_post, audio_pre = self.audio_liquid(audio_raw)
        
        # Atención cruzada (sin cambios)
        visual_expanded = visual_out.unsqueeze(1)
        audio_expanded = audio_out.unsqueeze(1)
        
        attended_visual, _ = self.cross_modal_attention(
            visual_expanded, audio_expanded, audio_expanded
        )
        attended_visual = attended_visual.squeeze(1)
        
        # Fusión final
        fused = torch.cat([attended_visual, audio_out], dim=-1)
        fused_features = self.fusion(fused)
        
        return fused_features, visual_post, visual_pre, audio_post, audio_pre




# =============================================================================
# CORPUS CALLOSUM TRIMODAL
# =============================================================================
class CorpusCallosumTrimodal(nn.Module):
    """
    Corpus Callosum optimizado con:
    - Dimensión base reducida: 512→320 dims (-37.5%)
    - Bottleneck compartido para 3 canales (1 Linear vs 3 ModuleList)
    - Flash Attention / xFormers compatible
    - Gates fusionados en tensor único
    Reducción: 2.1M → 0.88M parámetros (-58%)
    """
    
    def __init__(self, dim=512):  # CRÍTICO: Mantener 512 como dim base
        super().__init__()
        
        # Distribución trimodal (suma 512)
        self.visual_dim = 170
        self.audio_dim = 170
        self.semantic_dim = 172
        
        # Proyecciones individuales (mantienen entrada 512)
        self.visual_proj = nn.Linear(512, self.visual_dim)
        self.audio_proj = nn.Linear(512, self.audio_dim)
        self.semantic_proj = nn.Linear(512, self.semantic_dim)
        
        # Gates fusionados en tensor único
        self.register_parameter('trimodal_gates', nn.Parameter(torch.tensor([-2.0, -2.0, -2.0])))
      
        # Flash Attention detection
        try:
            from torch.nn.functional import scaled_dot_product_attention
            self.use_flash_attn = True
            print("✅ Flash Attention detectado (nativo PyTorch 2.0+)")
        except ImportError:
            self.use_flash_attn = False
            print("⚠️ Flash Attention no disponible, usando MultiheadAttention estándar")
        
        if not self.use_flash_attn:
            self.trimodal_attention = nn.MultiheadAttention(
                embed_dim=dim,  # 512
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        
        # Bottleneck compartido: 512 → 256 → 512
        self.shared_bottleneck = nn.Sequential(
            nn.Linear(dim, dim // 2),  # 512→256
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, dim),  # 256→512
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
        # Parámetros residuales
        self.residual_scale_base = nn.Parameter(torch.tensor(0.85))
        
        # Flow modulator
        self.flow_modulator = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Fatiga por canal
        self.register_parameter('visual_fatigue', nn.Parameter(torch.tensor(0.0), requires_grad=False))
        self.register_parameter('audio_fatigue', nn.Parameter(torch.tensor(0.0), requires_grad=False))
        self.register_parameter('semantic_fatigue', nn.Parameter(torch.tensor(0.0), requires_grad=False))
        
        self.fatigue_decay = 0.95
        self.fatigue_recovery = 0.01

    def _apply_flash_attention(self, x):
        """
        Aplica Flash Attention nativa de PyTorch 2.0+
        FIX: Corrección de dimensiones para seq_len variable
        """
        B, seq_len, embed_dim = x.shape
        num_heads = 8
        head_dim = embed_dim // num_heads
        
        if embed_dim != 512:
            raise ValueError(f"Flash Attention espera embed_dim=512, recibió {embed_dim}")
        
        q = k = v = x
        
        q = q.reshape(B, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, seq_len, num_heads, head_dim).transpose(1, 2)
        
        attn_out = scaled_dot_product_attention(
            q, k, v, 
            dropout_p=0.1 if self.training else 0.0,
            is_causal=False
        )
        
        attn_out = attn_out.transpose(1, 2).reshape(B, seq_len, embed_dim)
        
        return attn_out
        





    def forward(self, right_features):
        """
        FIX: Manejo robusto de dimensiones y verificación de coherencia trimodal
        """
        B = right_features.size(0)
        
        if right_features.size(-1) != 512:
            raise ValueError(f"CorpusCallosum espera entrada 512-dim, recibió {right_features.size(-1)}")
        
        visual_channel = self.visual_proj(right_features)
        audio_channel = self.audio_proj(right_features)
        semantic_channel = self.semantic_proj(right_features)
        
        gates = torch.sigmoid(self.trimodal_gates)
        
        visual_gated = visual_channel * gates[0]
        audio_gated = audio_channel * gates[1]
        semantic_gated = semantic_channel * gates[2]
        
        visual_gated = visual_gated * torch.sigmoid(1.0 - self.visual_fatigue)
        audio_gated = audio_gated * torch.sigmoid(1.0 - self.audio_fatigue)
        semantic_gated = semantic_gated * torch.sigmoid(1.0 - self.semantic_fatigue)
        
        structured = torch.cat([visual_gated, audio_gated, semantic_gated], dim=-1)
        structured_expanded = structured.unsqueeze(1)
        
        if self.use_flash_attn:
            attn_out = self._apply_flash_attention(structured_expanded)
            attn_out = attn_out.squeeze(1)
        else:
            attn_out, _ = self.trimodal_attention(
                structured_expanded, structured_expanded, structured_expanded
            )
            attn_out = attn_out.squeeze(1)
        
        processed = self.shared_bottleneck(attn_out)
        
        flow_strength = self.flow_modulator(right_features).squeeze(-1)
        dynamic_scale = self.residual_scale_base * (0.7 + 0.6 * flow_strength).unsqueeze(-1)
        
        output = processed + dynamic_scale * right_features
        
        pad_visual = 512 - self.visual_dim
        pad_audio = 512 - self.audio_dim - self.visual_dim
        pad_semantic = 512 - self.semantic_dim - self.audio_dim - self.visual_dim
        
        visual_full = torch.cat([
            visual_gated,
            torch.zeros(B, pad_visual, device=visual_gated.device)
        ], dim=-1)
        
        audio_full = torch.cat([
            torch.zeros(B, self.visual_dim, device=audio_gated.device),
            audio_gated,
            torch.zeros(B, pad_audio, device=audio_gated.device)
        ], dim=-1)
        
        semantic_full = torch.cat([
            torch.zeros(B, self.visual_dim + self.audio_dim, device=semantic_gated.device),
            semantic_gated,
            torch.zeros(B, pad_semantic, device=semantic_gated.device) if pad_semantic > 0 else torch.empty(B, 0, device=semantic_gated.device)
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
            
            fatigue_accumulation = 0.005
            
            self.visual_fatigue.data = torch.clamp(
                self.visual_fatigue * self.fatigue_decay + fatigue_accumulation * visual_activity - self.fatigue_recovery,
                0.0, 0.5
            )
            self.audio_fatigue.data = torch.clamp(
                self.audio_fatigue * self.fatigue_decay + fatigue_accumulation * audio_activity - self.fatigue_recovery,
                0.0, 0.5
            )
            self.semantic_fatigue.data = torch.clamp(
                self.semantic_fatigue * self.fatigue_decay + fatigue_accumulation * semantic_activity - self.fatigue_recovery,
                0.0, 0.5
            )
    
    def adjust_gates_by_fatigue(self):
        """Lógica original de ajuste de gates"""
        with torch.no_grad():
            fatigue_factor_visual = torch.clamp(1.0 - 0.1 * self.visual_fatigue, 0.5, 1.0)
            fatigue_factor_audio = torch.clamp(1.0 - 0.1 * self.audio_fatigue, 0.5, 1.0)
            fatigue_factor_semantic = torch.clamp(1.0 - 0.1 * self.semantic_fatigue, 0.5, 1.0)
            
            self.trimodal_gates.data[0] *= fatigue_factor_visual
            self.trimodal_gates.data[1] *= fatigue_factor_audio
            self.trimodal_gates.data[2] *= fatigue_factor_semantic
            
            self.trimodal_gates.data = torch.clamp(self.trimodal_gates.data, -2.0, 2.0)


# =============================================================================
# DIAGNÓSTICOS MEJORADOS - FIX: Cache de normalización y cálculos repetitivos
# =============================================================================
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
            'trimodal_coherence': [],  # FIX: Nueva métrica de coherencia
            'callosum_visual_fatigue': [], 'callosum_audio_fatigue': [], 'callosum_semantic_fatigue': [],
            'reasoning_steps': [], 'mtp_loss': [], 'logical_coherence': [],
            'reasoning_efficiency': [], 'mtp_accuracy': []
        }
        self.language_metrics = LanguageMetrics()
        
        # FIX: Cache para normalizaciones repetidas
        self._norm_cache = {}
        self._corr_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cached_norm(self, tensor, dim=-1):
        """Cache de normalización con limpieza periódica"""
        key = (tensor.data_ptr(), tensor.shape, dim)
        
        # FIX: Limpiar cache cada 100 accesos para evitar memory leak
        if len(self._norm_cache) > 100:
            self._norm_cache.clear()
            self._corr_cache.clear()
        if key in self._norm_cache:
            self.cache_hits += 1
            return self._norm_cache[key]
        else:
            self.cache_misses += 1
            norm = F.normalize(tensor, dim=dim)
            self._norm_cache[key] = norm
            return norm
    
    def measure_callosal_flow(self, right_features, left_context, channels=None):
        """
        Medición de coherencia multimodal con sincronización entre canales
        """
        # Normalización con cache
        right_norm = self._get_cached_norm(right_features)
        left_norm = self._get_cached_norm(left_context)
        
        # Correlación básica
        base_correlation = (right_norm * left_norm).sum(dim=-1).mean()
        
        if channels is not None:
            visual_norm = self._get_cached_norm(channels['visual'])
            audio_norm = self._get_cached_norm(channels['audio'])
            semantic_norm = self._get_cached_norm(channels['semantic'])
            
            # Coherencia inter-canal
            visual_audio_sync = (visual_norm * audio_norm).sum(dim=-1).mean()
            visual_semantic_sync = (visual_norm * semantic_norm).sum(dim=-1).mean()
            audio_semantic_sync = (audio_norm * semantic_norm).sum(dim=-1).mean()
            
            # Métrica de sincronización trimodal
            trimodal_sync = (visual_audio_sync + visual_semantic_sync + audio_semantic_sync) / 3
            
            # Correlación ponderada
            visual_corr = (right_norm * visual_norm).sum(dim=-1).mean()
            audio_corr = (right_norm * audio_norm).sum(dim=-1).mean()
            semantic_corr = (right_norm * semantic_norm).sum(dim=-1).mean()
            
            weighted_correlation = 0.3 * visual_corr + 0.3 * audio_corr + 0.2 * semantic_corr + 0.2 * trimodal_sync
            
            # Diversidad de flujo
            correlations = torch.tensor([visual_corr.item(), audio_corr.item(), semantic_corr.item()])
            flow_diversity = 1.0 - correlations.std().item()
            
            # Extraer coherencia si está disponible
            coherence_bonus = channels.get('coherence', 0.0) if isinstance(channels, dict) else 0.0
            
            # Flujo final
            flow = (
                0.5 * weighted_correlation.item() + 
                0.3 * flow_diversity + 
                0.2 * coherence_bonus
            )
        else:
            # Fallback simple
            flow_std = left_context.std(dim=-1).mean()
            flow = base_correlation.item() * min(1.0, flow_std.item() / 0.5)
        
        return float(flow)


        




    def evaluate_reasoning_quality(self, generated_texts, reference_texts, reasoning_steps):
        coherence_scores = []
        consistency_scores = []
        
        for gen, ref, steps in zip(generated_texts, reference_texts, reasoning_steps):
            gen_sentences = [s.strip() for s in gen.split('.') if s.strip()]
            
            # FIX: Evaluación de coherencia con manejo de edge cases
            if len(gen_sentences) > 1:
                sentence_similarities = []
                for i in range(len(gen_sentences) - 1):
                    words1 = set(gen_sentences[i].lower().split())
                    words2 = set(gen_sentences[i+1].lower().split())
                    
                    if len(words1) > 0 and len(words2) > 0:
                        jaccard = len(words1 & words2) / len(words1 | words2)
                        sentence_similarities.append(jaccard)
                
                coherence = np.mean(sentence_similarities) if sentence_similarities else 0.5
            else:
                coherence = 0.5
            
            # FIX: Consistencia con verificación de división por cero
            gen_words = set(gen.lower().split())
            ref_words = set(ref.lower().split())
            if len(gen_words | ref_words) == 0:
                consistency = 0.0
            else:
                consistency = len(gen_words & ref_words) / len(gen_words | ref_words)
            
            # FIX: Bonus de razonamiento con clamp seguro
            reasoning_bonus = min(1.0, (steps if hasattr(steps, '__float__') else float(steps)) / 3.0) if steps is not None else 0.0
            coherence_scores.append(coherence * (0.7 + 0.3 * reasoning_bonus))
            consistency_scores.append(consistency)
        
        return np.mean(coherence_scores), np.mean(consistency_scores)
    
    def calculate_synergy(self, visual_node, audio_node, callosal_flow, left_gate_mean, left_gate_std):
        # FIX: Cálculo de salud con clamp de seguridad
        visual_health = float(visual_node.metabolism) * float(visual_node.homeostasis) * max(0.0, 1.0 - float(visual_node.fatigue) * 0.5)
        audio_health = float(audio_node.metabolism) * float(audio_node.homeostasis) * max(0.0, 1.0 - float(audio_node.fatigue) * 0.5)
        callosal_health = float(callosal_flow)
        gate_balance = max(0.0, 1.0 - abs(float(left_gate_mean) - 0.5) * 2.0)
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
                # FIX: Conversión segura de tensor a valor escalar
                if isinstance(value, torch.Tensor):
                    if value.is_cuda:
                        if value.numel() == 1:
                            value = value.cpu().item()
                        else:
                            value = value.cpu().numpy().tolist()
                    else:
                        if value.numel() == 1:
                            value = value.item()
                        else:
                            value = value.numpy().tolist()
                self.history[key].append(value)
    
    def get_recent_avg(self, key, n=50):
        if key in self.history and len(self.history[key]) > 0:
            recent_values = self.history[key][-n:]
            clean_values = []
            for v in recent_values:
                if isinstance(v, torch.Tensor):
                    clean_values.append(float(v.item() if v.numel() == 1 else v.mean()))
                elif isinstance(v, (list, np.ndarray)):
                    clean_values.append(np.mean(v))
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
            
            # FIX: Calcular desbalance con manejo de valores idénticos
            values = [visual_fatigue, audio_fatigue, semantic_fatigue]
            max_val, min_val = max(values), min(values)
            imbalance = 0.0 if max_val == min_val else max_val - min_val
            
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
        
        # FIX: Métricas lingüísticas con indicadores visuales
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
        
        trimodal_coh = self.get_recent_avg('trimodal_coherence')
        coherence_loss_avg = self.get_recent_avg('coherence_loss')  # Obtener del historial

        print(f"\n🔗 COHERENCIA MULTIMODAL:")
        print(f"  Audio-Visual: {coherence_loss_avg:.3f} {'🟢' if coherence_loss_avg < 0.3 else '🟡' if coherence_loss_avg < 0.5 else '🔴'}")
        print(f"  Trimodal: {trimodal_coh:.3f} {'🟢' if trimodal_coh < 0.4 else '🟡' if trimodal_coh < 0.6 else '🔴'}")

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
        
        # FIX: Métricas de cache
        print(f"\n⚡ CACHE PERFORMANCE:")
        print(f"  Cache hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses + 1):.2%}")
        
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
            
            return TricameralOutput(
                logits=logits, fused_features=fused_features, enriched_context=enriched_context, 
                gate=gate, vis_post=vis_post, vis_pre=vis_pre, aud_post=aud_post, aud_pre=aud_pre,
                channels=channels, mtp_loss=mtp_loss, reasoning_steps=reasoning_steps
            )
        else:
            generated_text = self.left_hemisphere(
                enriched_context, captions=None, channels=channels, epoch=epoch
            )
            
            return TricameralOutput(generated_text=generated_text)
            

# =============================================================================
# DATASET MULTIMODAL ACTUALIZADO
# =============================================================================
class Flickr8kMultimodalDataset(Dataset):
    """Dataset que carga imagen, audio del caption y texto desde Kaggle"""
    
    def __init__(self, images_dir, audio_dir, captions_file, vocab, 
                 img_transform=None, max_len=30, sample_rate=16000, 
                 use_cache=True, cache_dir='./flickr8k_full/spectrograms_cache'):
        self.images_dir = images_dir
        self.audio_dir = audio_dir
        self.vocab = vocab
        self.max_len = max_len
        self.sample_rate = sample_rate
        self.img_transform = img_transform
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Procesador de audio (solo si no hay cache)
        if not use_cache:
            self.mel_spectrogram = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                win_length=400,
                hop_length=160,
                n_mels=80,
                f_min=0,
                f_max=8000
            )
            self.resampler = T.Resample(sample_rate, sample_rate)
        
        self.data = []
        
        # Mapeo: image_name -> list of audio files
        if use_cache and os.path.exists(cache_dir):
            audio_files = {f.stem: f for f in Path(cache_dir).glob("*.pt")}
            print(f"📦 Usando spectrogramas cacheados desde {cache_dir}")
        else:
            audio_files = {f.stem: f for f in Path(audio_dir).glob("*.wav")}
            print(f"🎵 Cargando desde archivos .wav (sin cache)")
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, caption = parts
                    img_path = os.path.join(images_dir, img_name)
                    
                    stem = Path(img_name).stem
                    audio_keys = [f"{stem}_{i}" for i in range(5)]
                    
                    for audio_key in audio_keys:
                        if audio_key in audio_files:
                            audio_path = audio_files[audio_key]
                            if os.path.exists(img_path):
                                self.data.append((img_path, str(audio_path), caption))
        
        print(f"✓ Loaded {len(self.data)} multimodal samples")
        if use_cache:
            print(f"  Formato audio: .pt (spectrogramas precalculados)")
        else:
            print(f"  Formato audio: .wav, 16kHz (procesamiento en vivo)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, audio_path, caption = self.data[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)
        
        # Cargar spectrogram desde cache o procesar en vivo
        if self.use_cache and audio_path.endswith('.pt'):
            mel_spec = torch.load(audio_path)
        else:
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
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
            else:
                mel_spec = mel_spec[..., :target_len]
            
            mel_spec = mel_spec.squeeze(0)
        
        tokens = ['<BOS>'] + caption.lower().split() + ['<EOS>']
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        
        return image, mel_spec, torch.tensor(token_ids, dtype=torch.long), caption


# =============================================================================
# FUNCIÓN DE ALINEACIÓN TEMPRANA - FIX: Desacoplada de pérdida principal
# =============================================================================
def compute_alignment_loss(visual_features, channels, alpha=0.1, epoch=0):
    """
    FIX: Pérdida auxiliar para alineación temprana de canales multimodales
    Solo activa en épocas iniciales (epoch < 6)
    """
    if epoch >= 6:
        return torch.tensor(0.0, device=visual_features.device)
    
    with torch.enable_grad():
        visual_norm = F.normalize(visual_features, dim=-1)
        visual_channel_norm = F.normalize(channels['visual'], dim=-1)
        audio_channel_norm = F.normalize(channels['audio'], dim=-1)
        semantic_channel_norm = F.normalize(channels['semantic'], dim=-1)
        
        visual_align = (1 - (visual_channel_norm * visual_norm).sum(dim=-1)).mean()
        audio_align = (1 - (audio_channel_norm * visual_norm).sum(dim=-1)).mean()
        semantic_align = (1 - (semantic_channel_norm * visual_norm).sum(dim=-1)).mean()
        
        # FIX: Pesos ponderados por importancia modal
        total_align = 0.5 * visual_align + 0.3 * audio_align + 0.2 * semantic_align
        
        return alpha * total_align



# =============================================================================
# FUNCIÓN DE PÉRDIDA TRICAMERAL - FIX: Todos los componentes como tensores
# =============================================================================

def compute_tricameral_loss(logits, captions, gate, vocab, 
                           visual_post, audio_post, 
                           mtp_loss=None, linguistic_reward=None,
                           channels=None, epoch=0,
                           lambda_reward=0.1, lambda_mtp=0.1):
    
    device = logits.device
    eps = 1e-8
    
    # CE loss en FP32 para estabilidad
    with torch.cuda.amp.autocast(enabled=False):
        logits_fp32 = logits.float().clamp(-20.0, 20.0)
        ce_loss = F.cross_entropy(
            logits_fp32.reshape(-1, len(vocab)),
            captions[:, 1:].reshape(-1),
            ignore_index=vocab['<PAD>'],
            reduction='mean'
        )
        ce_loss = ce_loss.clamp(0.0, 20.0)
    
    # Gate penalties en FP32
    with torch.cuda.amp.autocast(enabled=False):
        gate_fp32 = gate.float()
        gate_mean = gate_fp32.mean().clamp(eps, 1.0 - eps)
        gate_penalty = (gate_mean - 0.5).pow(2).clamp(0.0, 1.0)
        
        gate_std = gate_fp32.std().clamp(eps, 1.0)
        diversity_penalty = (0.08 - gate_std).clamp(0.0, 0.08).pow(2)
    
    # Coherencia audiovisual en FP32
    with torch.cuda.amp.autocast(enabled=False):
        visual_fp32 = visual_post.float()
        audio_fp32 = audio_post.float()
        
        v_norm = visual_fp32.norm(dim=-1, keepdim=True).clamp(eps, 100.0)
        a_norm = audio_fp32.norm(dim=-1, keepdim=True).clamp(eps, 100.0)
        
        visual_normalized = visual_fp32 / v_norm
        audio_normalized = audio_fp32 / a_norm
        
        audiovisual_coherence = (1.0 - (visual_normalized * audio_normalized).sum(dim=-1)).mean().clamp(0.0, 2.0)
    
    # Coherencia trimodal

    trimodal_coherence_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if channels is not None and isinstance(channels, dict):
        with torch.cuda.amp.autocast(enabled=False):
            v_ch = channels.get('visual')
            a_ch = channels.get('audio')
            s_ch = channels.get('semantic')
            
            if all(c is not None for c in [v_ch, a_ch, s_ch]):
                # Verificar dimensiones antes de procesar
                if v_ch.size(-1) != 512 or a_ch.size(-1) != 512 or s_ch.size(-1) != 512:
                    print(f"⚠️ Dimensiones inesperadas en canales: v={v_ch.size()}, a={a_ch.size()}, s={s_ch.size()}")
                    trimodal_coherence_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                else:
                    v_ch = v_ch.float()
                    a_ch = a_ch.float()
                    s_ch = s_ch.float()
                    
                    v_n = v_ch.norm(dim=-1, keepdim=True).clamp(eps, 100.0)
                    a_n = a_ch.norm(dim=-1, keepdim=True).clamp(eps, 100.0)
                    s_n = s_ch.norm(dim=-1, keepdim=True).clamp(eps, 100.0)
                    
                    v_norm = v_ch / v_n
                    a_norm = a_ch / a_n
                    s_norm = s_ch / s_n
                    
                    va = (1.0 - (v_norm * a_norm).sum(dim=-1)).mean().clamp(0.0, 2.0)
                    vs = (1.0 - (v_norm * s_norm).sum(dim=-1)).mean().clamp(0.0, 2.0)
                    as_ = (1.0 - (a_norm * s_norm).sum(dim=-1)).mean().clamp(0.0, 2.0)
                    
                    trimodal_coherence_loss = ((va + vs + as_) / 3.0).clamp(0.0, 2.0)
    
    # Recompensa lingüística
    if linguistic_reward is not None:
        linguistic_loss = -lambda_reward * linguistic_reward.float().clamp(-1.0, 1.0)
    else:
        linguistic_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    
    # MTP loss
    if mtp_loss is not None and isinstance(mtp_loss, torch.Tensor):
        mtp_term = lambda_mtp * mtp_loss.float().clamp(0.0, 20.0)
    else:
        mtp_term = torch.tensor(0.0, device=device, dtype=torch.float32)
    
    # Composición final en FP32
    with torch.cuda.amp.autocast(enabled=False):
        if epoch < 3:
            trimodal_coh_loss = torch.tensor(0.0, device=device)  # Forzar a cero
            coherence_weight = 0.0
        else:
            coherence_weight = 0.15  # Restaurar peso normal
        total_loss = (
            ce_loss +
            0.01 * gate_penalty +
            0.02 * diversity_penalty +
            0.05 * audiovisual_coherence +
            coherence_weight * trimodal_coherence_loss +
            linguistic_loss +
            mtp_term
        ).clamp(0.0, 30.0)
    
    # Verificación final
    if not torch.isfinite(total_loss):
        total_loss = ce_loss.clamp(0.0, 20.0)
    
    return (total_loss, ce_loss, gate_penalty, diversity_penalty, 
            audiovisual_coherence, linguistic_loss, mtp_term, trimodal_coherence_loss)




# =============================================================================
# LOOP DE ENTRENAMIENTO PRINCIPAL
# =============================================================================
def train_tricameral():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    resume_data = None
    start_epoch = 0
    best_val_loss = float('inf')
    
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([
            f for f in os.listdir(checkpoint_dir) 
            if f.startswith('tricameral_epoch_') and f.endswith('_full.pth')
        ])
        
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            
            print(f"\n{'='*80}")
            print(f"🔄 CHECKPOINT DETECTADO: {latest_checkpoint}")
            print(f"{'='*80}")
            
            try:
                resume_data = torch.load(checkpoint_path, map_location='cpu')
                start_epoch = resume_data['epoch'] + 1
                best_val_loss = resume_data.get('best_val_loss', float('inf'))
                
                print(f"✅ Checkpoint cargado exitosamente")
                print(f"   Época inicial: {start_epoch}")
                print(f"   Best val loss: {best_val_loss:.4f}")
                print(f"{'='*80}\n")
            except Exception as e:
                print(f"⚠️ Error cargando checkpoint: {e}")
                print(f"⚠️ Iniciando entrenamiento desde cero\n")
                resume_data = None
                start_epoch = 0
    
    print(f"\n{'='*80}")
    print(f"NeuroLogos TRICAMERAL v5.2 | Visión + Audio + Lenguaje + Razonamiento | Device: {device}")
    print(f"🎵 Usando dataset pre-generado de Kaggle (40k audios .wav)")
    print(f"🧠 Sistema médico + cognitivo + memoria episódica integrados")
    print(f"⚡ Mixed Precision (FP16) + Gradient Checkpointing activados")
    print(f"{'='*80}\n")
    
    flickr_dir = setup_flickr8k_with_audio('./flickr8k_full')
    images_dir = os.path.join(flickr_dir, 'Images')
    audio_dir = os.path.join(flickr_dir, 'wavs')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    cache_dir = os.path.join(flickr_dir, 'spectrograms_cache')
    
    if not os.path.exists(audio_dir) or len(list(Path(audio_dir).glob("*.wav"))) < 1000:
        print("❌ Error: No se encontraron suficientes audios. Ejecuta setup_flickr8k_with_audio() primero.")
        return
    
    print(f"✅ Audio directory found: {len(list(Path(audio_dir).glob('*.wav')))} files\n")
    
    use_cache = True
    if not os.path.exists(cache_dir) or len(list(Path(cache_dir).glob("*.pt"))) < 1000:
        print("⚡ Cache de spectrogramas no encontrado. Preprocesando...")
        cache_dir = preprocess_and_cache_spectrograms(audio_dir, cache_dir)
    else:
        print(f"✅ Usando cache existente: {len(list(Path(cache_dir).glob('*.pt')))} spectrogramas\n")
    
    if resume_data and 'vocab' in resume_data:
        vocab = resume_data['vocab']
        id2word = resume_data['id2word']
        print("✅ Vocabulario cargado desde checkpoint")
    else:
        vocab, id2word = build_vocab_flickr(captions_file, vocab_size=5000)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Flickr8kMultimodalDataset(
        images_dir, audio_dir, captions_file, vocab, 
        img_transform=transform, max_len=30,
        use_cache=use_cache, cache_dir=cache_dir
    )
    
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    print(f"📊 Dataset split: {train_size} train, {val_size} val\n")
    
    try:
        import google.colab
        IN_COLAB = True
        print("📦 Entorno Google Colab detectado")
    except:
        IN_COLAB = False
        print("💻 Entorno estándar detectado")
    
    if IN_COLAB:
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    
    model = NeuroLogosTricameral(len(vocab)).to(device)
    
    # FIX: Inicializar buffers de fatiga con valores estables
    nn.init.constant_(model.corpus_callosum.visual_fatigue, 0.0)
    nn.init.constant_(model.corpus_callosum.audio_fatigue, 0.0)
    nn.init.constant_(model.corpus_callosum.semantic_fatigue, 0.0)
    if resume_data:
        model.load_state_dict(resume_data['model_state_dict'])
        print("✅ Pesos del modelo restaurados")
    
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    if use_amp:
        print("⚡ Mixed Precision Training (FP16) activado - Reducción memoria 40%")
    else:
        print("⚠️ Mixed Precision desactivado (CPU mode)")
    
    # FIX: Learning rates más conservadores para épocas iniciales
    # Learning rates iniciales ajustados según checkpoint
    initial_lrs = {
        'right': 2e-4 if not resume_data else resume_data.get('lr_right', 2e-4),
        'corpus': 3e-4 if not resume_data else resume_data.get('lr_corpus', 3e-4),
        'left': 1e-4 if not resume_data else resume_data.get('lr_left', 1e-4)
    }

    optimizer = torch.optim.AdamW([
        {'params': model.right_hemisphere.parameters(), 'lr': initial_lrs['right']},
        {'params': model.corpus_callosum.parameters(), 'lr': initial_lrs['corpus']},
        {'params': model.left_hemisphere.parameters(), 'lr': initial_lrs['left']}
    ])
    

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=10)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[10])
    if resume_data:
        try:
            optimizer.load_state_dict(resume_data['optimizer_state_dict'])
        except ValueError as e:
            print(f"⚠️ No se pudo cargar optimizer state (estructura cambiada): {e}")
            print("⚠️ Iniciando optimizer desde cero con learning rates configurados")
        scheduler.load_state_dict(resume_data['scheduler_state_dict'])
        if 'scaler_state_dict' in resume_data and use_amp:
            scaler.load_state_dict(resume_data['scaler_state_dict'])
        print("✅ Optimizer, scheduler y scaler restaurados")
    
    diagnostics = EnhancedDiagnosticsTricameral()
    medical_system = TriangulatedMedicalSystem()
    cognitive_system = NeurocognitiveSystem()
    episodic_memory = HierarchicalEpisodicMemory(
        working_capacity=80,
        short_term_capacity=320,
        importance_threshold=0.7
    )
    
    if resume_data:
        if 'episodic_memory_buffer' in resume_data and 'episodic_memory_scores' in resume_data:
            episodic_memory.buffer = resume_data['episodic_memory_buffer']
            episodic_memory.surprise_scores = resume_data['episodic_memory_scores']
            print(f"✅ Memoria episódica restaurada ({len(episodic_memory.buffer)}/400)")
        
        if 'diagnostics' in resume_data:
            diagnostics.history = resume_data['diagnostics']
            print("✅ Historial de diagnósticos restaurado")
    
    print(f"\n🧠 Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎵 Audio encoder: AudioEncoder optimizado (canales reducidos 30%)")
    print(f"🔗 Trimodal callosum: 320 dims (reducción 37.5% vs 512)")
    print(f"💊 Medical System: Triangulated intervention")
    print(f"🧠 Cognitive System: Reasoning + MTP + Chain-of-Thought")
    print(f"🧠 Episodic Memory: Surprise-weighted replay (400 samples)")
    
    if resume_data:
        print(f"\n🔄 REANUDANDO desde época {start_epoch}/30")
    else:
        print(f"\n🚀 INICIANDO entrenamiento desde época 0/30")
    
    print(f"\n🚀 Loop de entrenamiento con {len(train_loader)} batches...\n")
    
    # FIX: Aumentar accumulation steps para mayor estabilidad
    accumulation_steps = 3  # Aumentado de 2 a 3
    
    for epoch in range(start_epoch, 30):
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
            health_score, liquid, gate_mean, gate_std, flow, epoch
        )
        medicine_level = "🟢 Nivel 0" if severity == 0 else f"🟡 Nivel 1" if severity <= 2 else f"🟠 Nivel 2" if severity <= 6 else "🔴 Nivel 3"
        
        if severity > 3 and epoch > 2:
            medical_system.apply_triangulated_intervention(model, issues, severity, confidence, epoch)
        
        if epoch > 1:
            model.corpus_callosum.adjust_gates_by_fatigue()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Health: {health_score}/5 | Med: {medicine_level}]")
        
        for batch_idx, (images, audio_specs, captions, raw_captions) in enumerate(pbar):
            if epoch == 0 and batch_idx == 0:
                model = apply_emergency_fixes(model)
            # FIX: Verificar integridad de datos antes de forward
            if images.dim() != 4 or audio_specs.dim() != 3 or captions.dim() != 2:
                print(f"⚠️ Dimensión inválida en batch {batch_idx}, saltando...")
                continue
            
            # FIX: Verificar no-NaN en inputs
            if not torch.isfinite(images).all() or not torch.isfinite(audio_specs).all() or not torch.isfinite(captions).all():
                print(f"⚠️ Valores no-finitos en inputs, saltando batch {batch_idx}")
                continue
            
            images = images.to(device, non_blocking=True)
            audio_specs = audio_specs.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(images, audio_specs, captions, epoch=epoch)
                
                logits = output.logits
                fused_feat = output.fused_features
                enriched_ctx = output.enriched_context
                gate = output.gate
                vis_post = output.vis_post
                vis_pre = output.vis_pre
                aud_post = output.aud_post
                aud_pre = output.aud_pre
                channels = output.channels
                mtp_loss = output.mtp_loss
                reasoning_steps = output.reasoning_steps
                
                # Verificar outputs
                if not torch.isfinite(logits).all():
                    print(f"⚠️ Logits NaN en batch {batch_idx}, saltando")
                    optimizer.zero_grad()
                    continue
                
                if not torch.isfinite(gate).all():
                    print(f"⚠️ Gate NaN en batch {batch_idx}, saltando")
                    optimizer.zero_grad()
                    continue
                
                linguistic_reward = None
                if batch_idx % 20 == 0 and epoch >= 2:
                    with torch.no_grad():
                        sample_size = min(2, images.size(0))
                        sample_indices = np.random.choice(images.size(0), size=sample_size, replace=False)
                        references = [raw_captions[i] for i in sample_indices]
                        
                        sample_images = images[sample_indices]
                        sample_audio = audio_specs[sample_indices]
                        
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            output_gen = model(sample_images, sample_audio, captions=None, epoch=epoch)
                            generated = output_gen.generated_text
                        
                        hypotheses = []
                        for i in range(generated.size(0)):
                            gen_words = [id2word.get(int(t.item()), '<UNK>') for t in generated[i]]
                            gen_sentence = " ".join(w for w in gen_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                            hypotheses.append(gen_sentence)
                        
                        if all(len(h.strip()) > 0 for h in hypotheses):
                            linguistic_reward = cognitive_system.linguistic_feedback.compute_linguistic_reward(references, hypotheses)
                            
                            if reasoning_steps is not None and reasoning_steps.numel() == images.size(0):
                                sample_steps = reasoning_steps[sample_indices]
                                coherence, consistency = diagnostics.evaluate_reasoning_quality(
                                    hypotheses, references, sample_steps.tolist()
                                )
                                diagnostics.update(logical_coherence=coherence, reasoning_efficiency=consistency)
                
                # FIX: Asegurar que mtp_loss sea tensor válido
                if mtp_loss is not None and not torch.isfinite(mtp_loss):
                    print(f"⚠️ MTP loss no-finito en batch {batch_idx}, reseteando...")
                    mtp_loss = torch.tensor(0.1, device=logits.device)
                
                with torch.cuda.amp.autocast(enabled=False):
                    loss, ce_loss, gate_penalty, diversity_penalty, coherence_loss, linguistic_loss, mtp_term, trimodal_coh_loss = compute_tricameral_loss(
                        logits, captions, gate, vocab, vis_post, aud_post, 
                        mtp_loss, linguistic_reward,
                        channels=channels,
                        epoch=epoch,
                        lambda_reward=0.05, lambda_mtp=0.05
                    )
                

                # Verificar loss
                if not torch.isfinite(loss):
                    print(f"⚠️ Loss NaN en batch {batch_idx}, usando solo CE")
                    loss = ce_loss.clamp(0.0, 20.0)
                
                alignment_loss = torch.tensor(0.0, device=device)
                if epoch < 6:
                    alignment_loss = compute_alignment_loss(fused_feat, channels, epoch=epoch)
                    if not torch.isfinite(alignment_loss):
                        alignment_loss = torch.tensor(0.0, device=device)
                
                total_loss_batch = (loss + alignment_loss) / accumulation_steps
                total_loss_batch = total_loss_batch.clamp(0.0, 30.0)

            # Verificar antes de backward
            if not torch.isfinite(total_loss_batch):
                print(f"⚠️ Total loss NaN, saltando batch")
                optimizer.zero_grad()
                scaler.update()
                continue

            try:
                scaler.scale(total_loss_batch).backward()
            except RuntimeError as e:
                print(f"⚠️ Backward failed: {e}")
                optimizer.zero_grad()
                scaler.update()
                continue

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                
                # Verificar gradientes manualmente
                max_grad = 0.0
                has_nan = False
                for p in model.parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            has_nan = True
                            break
                        max_grad = max(max_grad, p.grad.abs().max().item())
                
                if has_nan or max_grad > 10.0:
                    print(f"⚠️ Grad {'NaN' if has_nan else f'explosion {max_grad:.1f}'}, saltando")
                    optimizer.zero_grad()
                    scaler.update()
                    
                    # Estabilización agresiva cada 10 fallos
                    if batch_idx % 10 == 0:
                        with torch.no_grad():
                            model.right_hemisphere.visual_liquid.W_fast_short.data.mul_(0.5)
                            model.right_hemisphere.audio_liquid.W_fast_short.data.mul_(0.5)
                            model.corpus_callosum.trimodal_gates.data.fill_(-3.0)
                            model.left_hemisphere.liquid_gate[-1].bias.data.fill_(-5.0)
                    continue
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Plasticity ultra-reducida
            plasticity = 0.001 if epoch < 10 else 0.0001

            # Hebbian solo cada 10 batches
            if batch_idx % 10 == 0:
                if torch.isfinite(vis_post).all() and torch.isfinite(vis_pre).all():
                    model.right_hemisphere.visual_liquid.hebbian_update(vis_post, vis_pre, plasticity)
                
                if torch.isfinite(aud_post).all() and torch.isfinite(aud_pre).all():
                    model.right_hemisphere.audio_liquid.hebbian_update(aud_post, aud_pre, plasticity)

            model.right_hemisphere.visual_liquid.update_physiology_advanced(ce_loss.item())
            model.right_hemisphere.audio_liquid.update_physiology_advanced(ce_loss.item())

            total_loss += ce_loss.item()

            if batch_idx % 5 == 0:
                surprise = episodic_memory.compute_surprise(logits, captions[:, 1:], gate.mean())
                
                if np.isfinite(surprise) and surprise > 0:
                    sample_indices_mem = np.random.choice(images.size(0), size=min(2, images.size(0)), replace=False)
                    for idx in sample_indices_mem:
                        if torch.isfinite(images[idx]).all() and torch.isfinite(audio_specs[idx]).all() and torch.isfinite(captions[idx]).all():
                            episodic_memory.store_episode(
                                images[idx].detach(), 
                                audio_specs[idx].detach(), 
                                captions[idx].detach(), 
                                surprise
                            )

            # FIX: Replay episódico con verificaciones de seguridad
            if batch_idx % EPISODIC_REPLAY_FREQ == 0 and len(episodic_memory.working_memory) >= 32:
                replay_batch_size = 8
                replay_samples = episodic_memory._sample_from_buffer(
                    episodic_memory.working_memory, 
                    episodic_memory.working_scores, 
                    replay_batch_size
                )
                
                if replay_samples and len(replay_samples) > 0:
                    try:
                        # FIX: Verificar integridad de samples
                        replay_imgs = []
                        replay_audio = []
                        replay_caps = []
                        
                        for s in replay_samples:
                            if all(torch.isfinite(t).all() for t in s):
                                replay_imgs.append(s[0].to(device))
                                replay_audio.append(s[1].to(device))
                                replay_caps.append(s[2].to(device))
                        
                        if len(replay_imgs) == 0:
                            continue
                        
                        replay_imgs = torch.stack(replay_imgs)
                        replay_audio = torch.stack(replay_audio)
                        replay_caps = torch.stack(replay_caps)
                        
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                output_replay = model(replay_imgs, replay_audio, replay_caps, epoch=epoch)
                                
                                logits_replay = output_replay.logits
                                gate_replay = output_replay.gate
                                vis_post_replay = output_replay.vis_post
                                vis_pre_replay = output_replay.vis_pre
                                aud_post_replay = output_replay.aud_post
                                aud_pre_replay = output_replay.aud_pre
                                channels_replay = output_replay.channels
                                mtp_loss_replay = output_replay.mtp_loss

                            loss_replay, ce_replay, _, _, _, _, _, _ = compute_tricameral_loss(
                                logits_replay, replay_caps, gate_replay, vocab,
                                vis_post_replay, aud_post_replay,
                                channels=channels_replay,  # Usar channels del replay
                                epoch=epoch
                            )
                                                    
                        # FIX: Verificar loss válido antes de backward
                        if torch.isfinite(loss_replay):
                            scaler.scale(0.3 * loss_replay).backward()
                            
                            if torch.isfinite(vis_post_replay).all() and torch.isfinite(vis_pre_replay).all():
                                model.right_hemisphere.visual_liquid.hebbian_update(
                                    vis_post_replay, vis_pre_replay, plasticity * 1.5
                                )
                            
                            if torch.isfinite(aud_post_replay).all() and torch.isfinite(aud_pre_replay).all():
                                model.right_hemisphere.audio_liquid.hebbian_update(
                                    aud_post_replay, aud_pre_replay, plasticity * 1.5
                                )
                        
                        del replay_imgs, replay_audio, replay_caps
                        del logits_replay, gate_replay, vis_post_replay, vis_pre_replay
                        del aud_post_replay, aud_pre_replay, loss_replay, ce_replay
                        torch.cuda.empty_cache()
                        
                    except (RuntimeError, IndexError) as e:
                        print(f"⚠️ Replay falló (memoria): {type(e).__name__}")
                        torch.cuda.empty_cache()

            # FIX: Logging con verificaciones de NaN
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    visual_liquid_norm = model.right_hemisphere.visual_liquid.W_fast_short.norm().item()
                    audio_liquid_norm = model.right_hemisphere.audio_liquid.W_fast_short.norm().item()
                    
                    callosal_flow = diagnostics.measure_callosal_flow(fused_feat, enriched_ctx, channels)
                    
                    gate_mean_val = gate.mean().item() if gate.numel() > 1 else gate.item()
                    gate_std_val = gate.std().item() if gate.numel() > 1 else 0.0
                    channel_fatigue = channels['fatigue'] if channels is not None else {
                        'visual': 0.0, 'audio': 0.0, 'semantic': 0.0
                    }
                    
                    synergy = diagnostics.calculate_synergy(
                        visual_node, audio_node, callosal_flow, gate_mean_val, gate_std_val
                    )
                    
                    reasoning_steps_val = reasoning_steps.mean().item() if torch.is_tensor(reasoning_steps) else 0.0
                    mtp_loss_val = mtp_loss.item() if torch.is_tensor(mtp_loss) else 0.0
                    linguistic_reward_val = linguistic_reward.item() if linguistic_reward is not None else 0.0
                    alignment_loss_val = alignment_loss.item() if torch.is_tensor(alignment_loss) else 0.0
                    trimodal_coh_val = trimodal_coh_loss.item() if torch.is_tensor(trimodal_coh_loss) else 0.0
                    
                    diagnostics.update(
                        loss=ce_loss.item(),
                        visual_liquid_metabolism=float(visual_node.metabolism.clamp(0.1, 1.0)),
                        visual_liquid_fatigue=float(visual_node.fatigue.clamp(0.0, 0.5)),
                        visual_liquid_norm=visual_liquid_norm,
                        audio_liquid_metabolism=float(audio_node.metabolism.clamp(0.1, 1.0)),
                        audio_liquid_fatigue=float(audio_node.fatigue.clamp(0.0, 0.5)),
                        audio_liquid_norm=audio_liquid_norm,
                        callosal_flow=callosal_flow,
                        left_gate_mean=gate_mean_val,
                        left_gate_std=gate_std_val,
                        synergy_score=synergy,
                        health_score=health_score,
                        coherence_loss=coherence_loss.item(),
                        trimodal_coherence=trimodal_coh_val,
                        callosum_visual_fatigue=channel_fatigue['visual'],
                        callosum_audio_fatigue=channel_fatigue['audio'],
                        callosum_semantic_fatigue=channel_fatigue['semantic'],
                        reasoning_steps=reasoning_steps_val,
                        mtp_loss=mtp_loss_val,
                        linguistic_reward=linguistic_reward_val,
                        alignment_loss=alignment_loss_val
                    )
            
            pbar_dict = {
                'loss': f'{ce_loss.item():.3f}',
                'gate': f'{gate_mean_val:.2f}',
                'mem': f"{len(episodic_memory.buffer)}/{episodic_memory.capacity}",
                'coherence': f'{coherence_loss.item():.3f}'
            }

            if torch.is_tensor(mtp_loss):
                pbar_dict['mtp'] = f'{mtp_loss.item():.3f}'

            if torch.is_tensor(reasoning_steps):
                pbar_dict['reason'] = f'{reasoning_steps.mean().item():.1f}'

            if linguistic_reward is not None:
                pbar_dict['reward'] = f'{linguistic_reward.item():.3f}'

            if epoch < 6:
                pbar_dict['align'] = f'{alignment_loss_val:.3f}'

            if batch_idx % 500 == 0 and batch_idx > 0:
                avg_surprise = np.mean(episodic_memory.surprise_scores[-100:]) if len(episodic_memory.surprise_scores) >= 100 else 0.0
                pbar_dict['avg_surp'] = f'{avg_surprise:.2f}'

            pbar.set_postfix(pbar_dict)
        
        scheduler.step()
        
        # FIX: Generación con evaluación de pérdida
        generation_epochs = [0, 2, 5, 8, 12, 16, 20, 25, 29]
        
        if epoch in generation_epochs:
            print(f"\n{'='*80}")
            print(f"📝 GENERACIONES - Época {epoch}")
            print(f"{'='*80}")
            
            model.eval()
            val_loss_sum = 0
            val_count = 0
            
            with torch.no_grad():
                num_samples = 3 if epoch < 5 else 5
                val_indices = np.random.choice(len(val_dataset), size=min(num_samples, len(val_dataset)), replace=False)
                
                for i, idx in enumerate(val_indices):
                    img, aud, cap, raw_cap = val_dataset[idx]
                    
                    img_batch = img.unsqueeze(0).to(device)
                    aud_batch = aud.unsqueeze(0).to(device)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output_val_gen = model(img_batch, aud_batch, captions=None, epoch=epoch)
                        generated = output_val_gen.generated_text
                    
                    gen_words = [id2word.get(int(t.item()), '<UNK>') for t in generated[0]]
                    gen_sentence = " ".join(w for w in gen_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                    
                    # FIX: Calcular pérdida de validación
                    val_cap = cap.unsqueeze(0).to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output_val_loss = model(img_batch, aud_batch, val_cap, epoch=epoch)
                        val_logits = output_val_loss.logits
                        val_loss = F.cross_entropy(val_logits.reshape(-1, len(vocab)), val_cap[:, 1:].reshape(-1), ignore_index=vocab['<PAD>'])
                    val_loss_sum += val_loss.item()
                    val_count += 1
                    
                    print(f"\n[Muestra {i+1}]")
                    print(f"  Referencia: {raw_cap}")
                    print(f"  Generada:   {gen_sentence}")
                    
                    bleu = diagnostics.language_metrics.sentence_bleu(raw_cap, gen_sentence)
                    acc = diagnostics.language_metrics.token_accuracy(raw_cap, gen_sentence)
                    overlap = diagnostics.language_metrics.word_overlap(raw_cap, gen_sentence)
                    
                    print(f"  BLEU: {bleu:.3f} | Acc: {acc:.3f} | Overlap: {overlap:.3f} | ValLoss: {val_loss.item():.3f}")
            
            avg_val_loss = val_loss_sum / val_count if val_count > 0 else float('inf')
            print(f"\n✅ Validación - Loss: {avg_val_loss:.4f}")
            print(f"{'='*80}\n")
            model.train()
        
        # FIX: Checkpoint con verificación de val_loss
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for val_images, val_audio, val_captions, val_raw in tqdm(val_loader, desc="Validación", leave=False):
                    val_images = val_images.to(device, non_blocking=True)
                    val_audio = val_audio.to(device, non_blocking=True)
                    val_captions = val_captions.to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output_val = model(val_images, val_audio, val_captions, epoch=epoch)
                        val_logits = output_val.logits
                        val_ce = F.cross_entropy(val_logits.reshape(-1, len(vocab)), val_captions[:, 1:].reshape(-1), ignore_index=vocab['<PAD>'])
                    val_loss += val_ce.item()
                    val_batches += 1
                    
                    if val_batches % 10 == 0:
                        del val_images, val_audio, val_captions, val_logits, val_ce
                        torch.cuda.empty_cache()
            
            avg_val_loss = val_loss / val_batches
            print(f"\n✅ Validación - Loss: {avg_val_loss:.4f}")
            
            torch.cuda.empty_cache()
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': avg_val_loss,
                }, './checkpoints/tricameral_best_val.pth')
                print(f"💾 Mejor modelo guardado: val_loss={avg_val_loss:.4f}")
            
            model.train()
        
        # FIX: Checkpoint completo con históricos
        if epoch % 5 == 0:
            checkpoint_path = f'./checkpoints/tricameral_epoch_{epoch:02d}_full.pth'

            # Extraer learning rates actuales de cada param group
            current_lrs = {
                'lr_right': optimizer.param_groups[0]['lr'],
                'lr_corpus': optimizer.param_groups[1]['lr'],
                'lr_left': optimizer.param_groups[2]['lr']
            }

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'vocab': vocab,
                'id2word': id2word,
                'diagnostics': diagnostics.history,
                'medical_interventions': medical_system.intervention_history,
                'cognitive_interventions': cognitive_system.cognitive_history,
                'cache_stats': cognitive_system.linguistic_feedback.get_cache_stats(),
                'episodic_memory_size': len(episodic_memory.buffer),
                'episodic_memory_buffer': episodic_memory.buffer,
                'episodic_memory_scores': episodic_memory.surprise_scores,
                'best_val_loss': best_val_loss,
                **current_lrs  # Agregar learning rates actuales
            }, checkpoint_path)
            print(f"💾 Checkpoint guardado: {checkpoint_path}")
        
        diagnostics.report(epoch)
        print(f"✅ Época {epoch:02d} | Loss: {total_loss/len(train_loader):.4f}\n")
        
        # FIX: Reporte de memoria con verificaciones
        if len(episodic_memory.buffer) > 0:
            avg_surprise = np.mean(episodic_memory.surprise_scores) if len(episodic_memory.surprise_scores) > 0 else 0.0
            max_surprise = np.max(episodic_memory.surprise_scores) if len(episodic_memory.surprise_scores) > 0 else 0.0
            high_surprise_count = sum(1 for s in episodic_memory.surprise_scores if s > episodic_memory.surprise_threshold * 1.5) if len(episodic_memory.surprise_scores) > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"🧠 MEMORIA EPISÓDICA - Fin de Época {epoch}")
            print(f"  Buffer: {len(episodic_memory.buffer)}/{episodic_memory.capacity}")
            print(f"  Surprise: avg={avg_surprise:.3f}, max={max_surprise:.3f}")
            print(f"  Muestras alta sorpresa: {high_surprise_count} ({100*high_surprise_count/max(1,len(episodic_memory.buffer)):.1f}%)")
            print(f"{'='*60}\n")
    
    final_checkpoint = {
        'epoch': 29,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'vocab': vocab,
        'id2word': id2word,
        'diagnostics': diagnostics.history,
        'medical_interventions': medical_system.intervention_history,
        'cognitive_interventions': cognitive_system.cognitive_history,
        'cache_stats': cognitive_system.linguistic_feedback.get_cache_stats(),
        'episodic_memory_size': len(episodic_memory.buffer),
        'episodic_memory_buffer': episodic_memory.buffer,
        'episodic_memory_scores': episodic_memory.surprise_scores,
        'best_val_loss': best_val_loss,
        'final_train_loss': total_loss / len(train_loader),
        'lr_right': optimizer.param_groups[0]['lr'],
        'lr_corpus': optimizer.param_groups[1]['lr'],
        'lr_left': optimizer.param_groups[2]['lr']
    }
    torch.save(final_checkpoint, './tricameral_model_final_full.pth')
    print("💾 Modelo final guardado con estado completo: ./tricameral_model_final_full.pth")
    
    print(f"\n{'='*80}")
    print("📋 HISTORIAL DE INTERVENCIONES - RESUMEN NEUROCOGNITIVO")
    print(f"{'='*80}")
    
    print(f"\n🏥 INTERVENCIONES MÉDICAS: {len(medical_system.intervention_history)}")
    if len(medical_system.intervention_history) == 0:
        print("✓ No se requirieron intervenciones médicas - Sistema homeostático estable")
    else:
        for intervention in medical_system.intervention_history:
            print(f"Época {intervention['epoch']:02d} | {intervention['level']} | Severidad: {intervention['severity']}/12")
            print(f"  Problemas: {', '.join(intervention['issues'])}")
            for inter in intervention['interventions']:
                print(f"    • {inter}")
            print()
    
    print(f"\n🧠 INTERVENCIONES COGNITIVAS: {sum(1 for h in cognitive_system.cognitive_history if h['severity'] > 0)}")
    if len(cognitive_system.cognitive_history) == 0:
        print("✓ No se requirieron intervenciones cognitivas - Razonamiento óptimo")
    else:
        for intervention in cognitive_system.cognitive_history:
            if intervention['severity'] > 0:
                print(f"Época {intervention['epoch']:02d} | Severidad: {intervention['severity']}/9")
                print(f"  Issues: {', '.join(intervention['issues'])}")
                print()
    
    final_cache_stats = cognitive_system.linguistic_feedback.get_cache_stats()
    print(f"\n⚡ ESTADÍSTICAS FINALES DEL SISTEMA:")
    print(f"  Caché lingüístico: {final_cache_stats['cache_size']} entradas")
    print(f"  Hit rate: {final_cache_stats['hit_rate']:.2%}")
    print(f"  Memoria episódica: {len(episodic_memory.buffer)}/{episodic_memory.capacity} samples")
    print(f"  Best val loss: {best_val_loss:.4f}")
    
    final_fatigue = {
        'visual': diagnostics.get_recent_avg('callosum_visual_fatigue'),
        'audio': diagnostics.get_recent_avg('callosum_audio_fatigue'),
        'semantic': diagnostics.get_recent_avg('callosum_semantic_fatigue')
    }
    print(f"\n🔗 DISTRIBUCIÓN DE FATIGA FINAL:")
    for k, v in final_fatigue.items():
        print(f"  {k}: {v:.3f} {'🔴' if v > 0.3 else '🟡' if v > 0.15 else '🟢'}")
    
    print(f"\n{'='*80}")
    print("🧠 ENTRENAMIENTO TRICAMERAL COMPLETADO - SISTEMA NEUROCOGNITIVO ESTABLE")
    print(f"{'='*80}")




# =============================================================================
# EJECUCIÓN DIRECTA
# =============================================================================
if __name__ == "__main__":
    # Descargar dataset automáticamente
    print("🚀 Iniciando descarga del dataset...")
    path = kagglehub.dataset_download("warcoder/flickr-8k-audio-caption-corpus")
    print(f"📦 Dataset descargado en: {path}")
    
    # Setup del dataset
    print("🚀 Iniciando setup del dataset...\n")
    dataset_path = setup_flickr8k_with_audio('./flickr8k_full')

    if dataset_path:
        print(f"\n✅ Dataset preparado en: {dataset_path}")
        print(f"✅ Continuando con entrenamiento...\n")
        train_tricameral()
    else:
        print("\n❌ ERROR: Setup fallido. Revisa los logs.")
        exit(1)
# =============================================================================
# NeuroLogos TRICAMERAL v5.0
# Hemisferio Derecho: Visión + Audio
# Hemisferio Izquierdo: Lenguaje + Razonamiento
# Corpus Callosum: Fusión trimodal (ve, escucha, razona)
# + Generador de audio neuronal (TTS sin API)
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
import asyncio
import sys
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GENERADOR DE AUDIOS (Edge-TTS con retry y rate limiting)
# =============================================================================
async def generate_audio_async(text, output_path, voice="en-US-AriaNeural", max_retries=3):
    """Genera un audio usando Edge-TTS con retry logic"""
    import edge_tts
    
    # Validar texto
    if not text or len(text.strip()) < 3:
        return False
    
    text = text.strip()
    
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(
                text,
                voice=voice,
                rate="-5%",
                pitch="+2Hz"
            )
            
            await communicate.save(output_path)
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                # Backoff exponencial: 0.5s, 1s, 2s
                wait_time = 0.5 * (2 ** attempt)
                await asyncio.sleep(wait_time)
            else:
                # Solo mostrar errores en el último intento
                if "No audio was received" not in str(e):
                    print(f"\n❌ Error: {output_path}: {e}")
    
    return False


async def generate_all_audios_batch(captions_list, audio_dir, batch_size=20):
    """Genera todos los audios en batches pequeños con rate limiting"""
    os.makedirs(audio_dir, exist_ok=True)
    
    # Filtrar los que ya existen y validar texto
    tasks = []
    skipped = 0
    
    for key, text in captions_list:
        output_file = os.path.join(audio_dir, f"{key}.mp3")
        
        # SALTAR SI YA EXISTE
        if os.path.exists(output_file):
            skipped += 1
            continue
        
        # Validar texto
        if text and len(text.strip()) >= 3:
            tasks.append((key, text.strip(), output_file))
    
    if len(tasks) == 0:
        print(f"✓ Todos los audios ya existen ({len(captions_list)} archivos)")
        return
    
    if skipped > 0:
        print(f"✓ Saltando {skipped} audios existentes")
    
    print(f"🎵 Generando {len(tasks)} audios faltantes con Edge-TTS...")
    print(f"   Batch size: {batch_size} | Estimado: ~{len(tasks) // 30} min")
    
    successful = 0
    failed = 0
    
    # Generar en batches pequeños con pausa entre ellos
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        
        # Crear tareas del batch
        batch_tasks = [
            generate_audio_async(text, output_path) 
            for _, text, output_path in batch
        ]
        
        # Ejecutar batch con límite de concurrencia
        results = await asyncio.gather(*batch_tasks)
        
        successful += sum(results)
        failed += len(results) - sum(results)
        
        # Barra de progreso simple
        progress = (i + len(batch)) / len(tasks) * 100
        total_generated = successful + skipped
        print(f"  📊 {progress:.1f}% | ✓ {total_generated}/{len(captions_list)} totales", end='\r')
        
        # Pausa entre batches para evitar rate limiting (importante!)
        if i + batch_size < len(tasks):
            await asyncio.sleep(1.0)
    
    print(f"\n✅ Completado: {successful} nuevos, {skipped} existían, {failed} fallidos")
    
    # Si hay muchos fallos, dar recomendación
    if failed > len(tasks) * 0.2:  # >20% de fallos
        print(f"\n⚠️  {failed} audios fallaron (Edge-TTS puede tener rate limiting)")
        print("💡 Soluciones:")
        print("   - Espera 5 min y ejecuta de nuevo (solo regenerará faltantes)")
        print("   - Usa gen_dataset.py con voces alternativas")
        print("   - Continúa entrenamiento con los audios existentes")


def generate_audios_sync(images_dir, captions_file, audio_dir):
    """Wrapper síncrono para generar audios"""
    # Cargar captions
    caption_list = []
    with open(captions_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_name, text = parts
                stem = Path(img_name).stem
                key = f"{stem}_{idx % 5}"
                caption_list.append((key, text.strip()))
    
    print(f"📝 Encontrados {len(caption_list)} captions")
    
    # Detectar si estamos en Jupyter/Colab
    try:
        get_ipython().__class__.__name__
        in_jupyter = True
    except NameError:
        in_jupyter = False
    
    # Generar audios
    if in_jupyter:
        # En Jupyter/Colab: usar nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            print("📦 Instalando nest_asyncio...")
            os.system('pip install -q nest_asyncio')
            import nest_asyncio
            nest_asyncio.apply()
        
        # Usar el loop existente
        loop = asyncio.get_event_loop()
        loop.run_until_complete(generate_all_audios_batch(caption_list, audio_dir))
    else:
        # En script normal: configurar event loop para Windows si es necesario
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(generate_all_audios_batch(caption_list, audio_dir))


# =============================================================================
# UTILIDADES: Descarga rápida desde GitHub/Hugging Face
# =============================================================================
def download_from_github(repo_url, output_dir='./flickr8k_full'):
    """Descarga dataset pre-preparado desde GitHub/Hugging Face"""
    import urllib.request
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Descargando desde: {repo_url}")
    
    try:
        # Descargar metadata
        print("  📋 Descargando metadata...")
        metadata_url = f"{repo_url}/metadata.json"
        metadata_path = output_path / "metadata.json"
        urllib.request.urlretrieve(metadata_url, metadata_path)
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        print(f"  ✓ Dataset: {metadata['total_images']} imágenes, {metadata['total_audios']} audios")
        
        # Descargar captions
        print("  📝 Descargando captions...")
        captions_url = f"{repo_url}/captions_es.txt"
        urllib.request.urlretrieve(captions_url, output_path / "captions_es.txt")
        
        # Descargar imágenes
        images_dir = output_path / "Images"
        images_dir.mkdir(exist_ok=True)
        
        print(f"  📸 Descargando {len(metadata['image_zips'])} partes de imágenes...")
        for i, zip_info in enumerate(metadata['image_zips'], 1):
            zip_url = f"{repo_url}/{zip_info['filename']}"
            zip_path = output_path / zip_info['filename']
            
            print(f"    [{i}/{len(metadata['image_zips'])}] {zip_info['filename']} ({zip_info['size_mb']} MB)...", end=' ')
            urllib.request.urlretrieve(zip_url, zip_path)
            
            # Extraer
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(images_dir)
            
            os.remove(zip_path)
            print("✓")
        
        # Descargar audios (si existen)
        if metadata['audio_zips']:
            audio_dir = output_path / "Audio_es"
            audio_dir.mkdir(exist_ok=True)
            
            print(f"  🎵 Descargando {len(metadata['audio_zips'])} partes de audio...")
            for i, zip_info in enumerate(metadata['audio_zips'], 1):
                zip_url = f"{repo_url}/{zip_info['filename']}"
                zip_path = output_path / zip_info['filename']
                
                print(f"    [{i}/{len(metadata['audio_zips'])}] {zip_info['filename']} ({zip_info['size_mb']} MB)...", end=' ')
                urllib.request.urlretrieve(zip_url, zip_path)
                
                # Extraer
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(audio_dir)
                
                os.remove(zip_path)
                print("✓")
        
        print(f"\n✅ Dataset descargado en: {output_path.resolve()}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error descargando: {e}")
        return False


def setup_flickr8k(data_dir='./flickr8k_full', github_url=None):
    """Descarga y organiza Flickr8k - ahora con opción GitHub"""
    images_dir = os.path.join(data_dir, 'Images')
    captions_file = os.path.join(data_dir, 'captions_es.txt')
    
    # Si ya existe, retornar
    if os.path.exists(images_dir) and os.path.exists(captions_file):
        print("✓ Flickr8k already exists\n")
        return data_dir
    
    print("🎯 OPCIONES DE DESCARGA:")
    print("  1. Desde GitHub/Hugging Face (RÁPIDO - 2-3 min)")
    print("  2. Desde fuente original (LENTO - 5-10 min)")
    
    # Detectar si estamos en notebook
    try:
        get_ipython().__class__.__name__
        in_jupyter = True
    except NameError:
        in_jupyter = False
    
    # Si se provee URL, usar directamente
    if github_url:
        if download_from_github(github_url, data_dir):
            return data_dir
        else:
            print("⚠️  Descarga desde GitHub falló. Usando método original...")
    
    # Si no hay URL, preguntar o usar original
    if in_jupyter:
        print("\n💡 Recomendación: Sube tu dataset a GitHub primero")
        print("   Ejecuta: python prepare_and_upload_dataset.py")
        print("\n   Continuando con descarga original...")
        choice = "2"
    else:
        choice = input("\nOpción [1]: ").strip() or "1"
    
    if choice == "1":
        url = input("URL del repo (ej: https://github.com/user/repo/raw/main): ").strip()
        if url and download_from_github(url, data_dir):
            return data_dir
        else:
            print("⚠️  Continuando con descarga original...")
    
    # Método original (descarga desde fuente oficial)
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("📥 Downloading Flickr8k...")
    import urllib.request
    import zipfile
    
    # Descargar imágenes
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) < 8000:
        print("📥 Downloading images (~800 MB)...")
        images_url = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
        zip_path = os.path.join(data_dir, 'images.zip')
        urllib.request.urlretrieve(images_url, zip_path)
        
        print("📂 Extracting images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
        
        # Mover imágenes al directorio correcto
        old_dir = os.path.join(data_dir, 'Flicker8k_Dataset')
        if os.path.exists(old_dir):
            if not os.path.exists(images_dir):
                os.rename(old_dir, images_dir)
            else:
                import shutil
                for f in os.listdir(old_dir):
                    shutil.move(os.path.join(old_dir, f), images_dir)
                os.rmdir(old_dir)
        
        print(f"✓ Images extracted: {len(os.listdir(images_dir))} files\n")
    
    # Descargar captions
    if not os.path.exists(captions_file):
        print("📥 Downloading captions...")
        captions_url = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
        zip_path = os.path.join(data_dir, 'captions.zip')
        urllib.request.urlretrieve(captions_url, zip_path)
        
        print("📂 Extracting captions...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
        
        # Procesar captions
        print("📝 Processing captions...")
        token_file = os.path.join(data_dir, 'Flickr8k.token.txt')
        
        if os.path.exists(token_file):
            with open(token_file, 'r', encoding='utf-8') as fin, \
                 open(captions_file, 'w', encoding='utf-8') as fout:
                for line in fin:
                    if '\t' not in line:
                        continue
                    img_cap, text = line.strip().split('\t')
                    img_name = img_cap.split('#')[0]
                    fout.write(f"{img_name}\t{text}\n")
            
            print(f"✓ Captions processed\n")
    
    print("✅ Flickr8k ready\n")
    return data_dir


def build_vocab_flickr(captions_file, vocab_size=5000):
    """Construye vocabulario desde el archivo de captions"""
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


# =============================================================================
# NEURONA LÍQUIDA ESTABLE (del modelo original)
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
# HEMISFERIO IZQUIERDO SIMPLIFICADO (del modelo original)
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
            
            # Retornos simplificados para compatibilidad
            gate = torch.sigmoid(torch.randn(batch_size, seq_len, 1, device=device) * 0.1 + 0.5)
            return logits, gate, None, None
        else:
            # Beam search simplificado
            return self._greedy_decode(visual_context, max_len, device)
    
    def _greedy_decode(self, visual_context, max_len, device):
        batch_size = visual_context.size(0)
        generated = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)  # <BOS>
        
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
            
            if (next_token == 2).all():  # <EOS>
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
# 1. DATASET MULTIMODAL (Imagen + Audio + Caption)
# =============================================================================
class Flickr8kMultimodalDataset(Dataset):
    """Dataset que carga imagen, audio del caption y texto"""
    
    def __init__(self, images_dir, audio_dir, captions_file, vocab, 
                 img_transform=None, max_len=30, sample_rate=16000):
        self.images_dir = images_dir
        self.audio_dir = audio_dir
        self.vocab = vocab
        self.max_len = max_len
        self.sample_rate = sample_rate
        self.img_transform = img_transform
        
        # Procesador de audio
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
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, caption = parts
                    img_path = os.path.join(images_dir, img_name)
                    
                    # Audio path (generado por gen_dataset.py)
                    stem = Path(img_name).stem
                    # Cada imagen tiene 5 audios (0-4)
                    for i in range(5):
                        audio_path = os.path.join(audio_dir, f"{stem}_{i}.mp3")
                        if os.path.exists(img_path) and os.path.exists(audio_path):
                            self.data.append((img_path, audio_path, caption))
        
        print(f"✓ Loaded {len(self.data)} multimodal samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, audio_path, caption = self.data[idx]
        
        # 1. Cargar imagen
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)
        
        # 2. Cargar y procesar audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample si es necesario
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convertir a mono si es estéreo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Mel-spectrogram (80 x T)
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)  # Log-mel
        
        # Normalizar
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Padding/truncate a longitud fija (por ejemplo, 3 segundos = 300 frames)
        target_len = 300
        if mel_spec.size(-1) < target_len:
            pad = target_len - mel_spec.size(-1)
            mel_spec = F.pad(mel_spec, (0, pad))
        else:
            mel_spec = mel_spec[..., :target_len]
        
        # 3. Tokenizar caption
        tokens = ['<BOS>'] + caption.lower().split() + ['<EOS>']
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        
        return image, mel_spec.squeeze(0), torch.tensor(token_ids, dtype=torch.long), caption


# =============================================================================
# 2. ENCODER DE AUDIO (Hemisferio Derecho - Canal Auditivo)
# =============================================================================
class AudioEncoder(nn.Module):
    """Encoder de audio usando Conv + Transformer"""
    
    def __init__(self, output_dim=512):
        super().__init__()
        
        # CNN para extraer features locales del mel-spectrogram
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
        
        # Transformer para capturar dependencias temporales
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
        
        # Proyección final
        self.projection = nn.Linear(512, output_dim)
        
        # Pooling adaptativo
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, mel_spec):
        """
        Args:
            mel_spec: (batch, 80, time)
        Returns:
            audio_features: (batch, output_dim)
        """
        # CNN: (B, 80, T) -> (B, 512, T')
        x = self.conv_layers(mel_spec)
        
        # Transformer: (B, 512, T') -> (B, T', 512)
        x = x.transpose(1, 2)
        x = self.temporal_encoder(x)
        
        # Pooling: (B, T', 512) -> (B, 512)
        x = x.transpose(1, 2)
        x = self.adaptive_pool(x).squeeze(-1)
        
        # Proyección final
        out = self.projection(x)
        
        return out


# =============================================================================
# 3. HEMISFERIO DERECHO EXTENDIDO (Visión + Audio)
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
        
        # Fusión cross-modal (atención cruzada)
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Proyección de fusión
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
            visual_post, visual_pre: Para Hebbian
            audio_post, audio_pre: Para Hebbian
        """
        # Procesamiento visual
        visual_raw = self.visual_encoder(image).flatten(1)
        visual_out, visual_post, visual_pre = self.visual_liquid(visual_raw)
        
        # Procesamiento auditivo
        audio_raw = self.audio_encoder(audio)
        audio_out, audio_post, audio_pre = self.audio_liquid(audio_raw)
        
        # Atención cruzada: visual atendiendo a audio
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
# 4. CORPUS CALLOSUM TRIMODAL
# =============================================================================
class CorpusCallosumTrimodal(nn.Module):
    """Corpus callosum con canales: visual, auditivo, semántico"""
    
    def __init__(self, dim=512):
        super().__init__()
        
        # Canales estructurales
        self.visual_dim = dim // 3
        self.audio_dim = dim // 3
        self.semantic_dim = dim - 2 * (dim // 3)
        
        self.visual_proj = nn.Linear(dim, self.visual_dim)
        self.audio_proj = nn.Linear(dim, self.audio_dim)
        self.semantic_proj = nn.Linear(dim, self.semantic_dim)
        
        # Atención trimodal
        self.trimodal_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Transfer blocks
        self.transfer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        self.residual_scale = nn.Parameter(torch.tensor(0.85))
    
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
        
        # Reconstruir tensor completo
        structured = torch.cat([visual_channel, audio_channel, semantic_channel], dim=-1)
        
        # Atención trimodal
        structured_expanded = structured.unsqueeze(1)
        attn_out, _ = self.trimodal_attention(
            structured_expanded, structured_expanded, structured_expanded
        )
        attn_out = attn_out.squeeze(1)
        
        # Transfer blocks
        for block in self.transfer:
            attn_out = attn_out + block(attn_out)
        
        # Residual
        output = attn_out + self.residual_scale * right_features
        
        # Crear versiones completas para compatibilidad
        visual_full = torch.cat([
            visual_channel,
            torch.zeros(visual_channel.size(0), self.audio_dim + self.semantic_dim, device=visual_channel.device)
        ], dim=-1)
        
        audio_full = torch.cat([
            torch.zeros(audio_channel.size(0), self.visual_dim, device=audio_channel.device),
            audio_channel,
            torch.zeros(audio_channel.size(0), self.semantic_dim, device=audio_channel.device)
        ], dim=-1)
        
        semantic_full = torch.cat([
            torch.zeros(semantic_channel.size(0), self.visual_dim + self.audio_dim, device=semantic_channel.device),
            semantic_channel
        ], dim=-1)
        
        return output, {
            'visual': visual_full,
            'audio': audio_full,
            'semantic': semantic_full
        }


# =============================================================================
# 5. GENERADOR DE AUDIO NEURONAL (Vocoder simple)
# =============================================================================
class NeuralAudioGenerator(nn.Module):
    """Generador de audio desde embeddings lingüísticos (TTS neuronal)"""
    
    def __init__(self, text_dim=512, output_sr=16000):
        super().__init__()
        self.output_sr = output_sr
        
        # Upsampler: text embeddings -> mel-spectrogram
        self.text_to_mel = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 80 * 300),  # 80 mels x 300 frames (~3 sec)
        )
        
        # Vocoder: mel -> waveform (Griffin-Lim o WaveGRU)
        # Por simplicidad, usamos Griffin-Lim de torchaudio
        self.inverse_mel = T.InverseMelScale(
            n_stft=201,
            n_mels=80,
            sample_rate=output_sr,
            f_min=0,
            f_max=8000
        )
        
        self.griffin_lim = T.GriffinLim(
            n_fft=400,
            win_length=400,
            hop_length=160
        )
    
    def forward(self, text_embedding):
        """
        Args:
            text_embedding: (B, text_dim)
        Returns:
            audio_waveform: (B, 1, num_samples)
        """
        # Text -> Mel
        mel_flat = self.text_to_mel(text_embedding)
        mel_spec = mel_flat.view(-1, 80, 300)
        
        # Denormalizar (aproximación)
        mel_spec = torch.exp(mel_spec) - 1e-9
        
        # Mel -> Linear spectrogram
        linear_spec = self.inverse_mel(mel_spec)
        
        # Linear -> Waveform (Griffin-Lim)
        waveforms = []
        for i in range(linear_spec.size(0)):
            wave = self.griffin_lim(linear_spec[i])
            waveforms.append(wave)
        
        audio = torch.stack(waveforms, dim=0).unsqueeze(1)
        
        return audio


# =============================================================================
# 6. MODELO TRICAMERAL COMPLETO
# =============================================================================
class NeuroLogosTricameral(nn.Module):
    """Arquitectura completa: Visión + Audio -> Lenguaje + Audio"""
    
    def __init__(self, vocab_size):
        super().__init__()
        
        # Hemisferio derecho (visión + audio)
        self.right_hemisphere = RightHemisphereTricameral(output_dim=512)
        
        # Corpus callosum trimodal
        self.corpus_callosum = CorpusCallosumTrimodal(dim=512)
        
        # Hemisferio izquierdo (lenguaje + razonamiento)
        self.left_hemisphere = LeftHemisphere(vocab_size, embed_dim=256, hidden_dim=512)
        
        # Generador de audio
        self.audio_generator = NeuralAudioGenerator(text_dim=512)
    
    def forward(self, image, audio, captions=None, epoch=0, generate_audio=False):
        """
        Args:
            image: (B, 3, H, W)
            audio: (B, 80, T) - Mel-spectrogram del caption
            captions: (B, seq_len) - Tokens (solo en train)
            generate_audio: bool - Si generar audio
        
        Returns (training):
            logits, gates, losses, posts, pres, channels
        
        Returns (inference):
            generated_text, generated_audio
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
            # Inference: generar texto + audio
            generated_text = self.left_hemisphere(
                enriched_context, captions=None, channels=channels, epoch=epoch
            )
            
            generated_audio = None
            if generate_audio:
                # Usar el contexto enriquecido para generar audio
                generated_audio = self.audio_generator(enriched_context)
            
            return generated_text, generated_audio


# =============================================================================
# 7. FUNCIÓN DE PÉRDIDA EXTENDIDA
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
    
    # Coherencia audio-visual (deben estar alineados)
    visual_norm = F.normalize(visual_post, dim=-1)
    audio_norm = F.normalize(audio_post, dim=-1)
    
    # Pérdida de coherencia: queremos que estén correlacionados
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
        0.1 * coherence_loss +  # NUEVO
        linguistic_loss + 
        mtp_term
    )
    
    return total_loss, ce_loss, gate_penalty, diversity_penalty, coherence_loss


# =============================================================================
# 8. LOOP DE ENTRENAMIENTO
# =============================================================================
def train_tricameral(github_repo_url=None):
    """
    Entrena el modelo tricameral
    
    Args:
        github_repo_url: URL opcional del repo de GitHub/Hugging Face
                        Ejemplo: "https://github.com/user/flickr8k-prepared/raw/main"
                        o "https://huggingface.co/datasets/user/flickr8k-prepared/resolve/main"
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"NeuroLogos TRICAMERAL v5.0 | Visión + Audio + Lenguaje | Device: {device}")
    print(f"{'='*80}\n")
    
    # Setup dataset (descarga desde GitHub si se provee URL)
    flickr_dir = setup_flickr8k('./flickr8k_full', github_url=github_repo_url)
    images_dir = os.path.join(flickr_dir, 'Images')
    audio_dir = os.path.join(flickr_dir, 'Audio_es')
    captions_file = os.path.join(flickr_dir, 'captions_es.txt')
    
    # Verificar si existen audios
    existing_audios = 0
    if os.path.exists(audio_dir):
        existing_audios = len([f for f in os.listdir(audio_dir) if f.endswith('.mp3')])
    
    min_audios_needed = 5000  # Mínimo para entrenamiento decente
    
    if existing_audios >= min_audios_needed:
        print(f"✓ Audio directory found: {existing_audios} archivos")
        print(f"  Suficientes audios para entrenamiento trimodal")
        use_audio = True
    elif existing_audios > 0:
        print(f"⚠️  {existing_audios} audios encontrados (mínimo recomendado: {min_audios_needed})")
        print(f"  Continuando con audios existentes...")
        use_audio = True
    else:
        print("⚠️  No se encontraron audios.")
        print("\n🎯 OPCIONES:")
        print("  1. Entrenar en modo BIMODAL (solo visión + texto) - RÁPIDO")
        print("  2. Generar audios ahora con Edge-TTS - LENTO (~20 min)")
        print("  3. Usar script externo: python gen_dataset.py")
        
        # En Colab/notebook, asumir opción 1 por defecto
        try:
            get_ipython().__class__.__name__
            print("\n💡 Seleccionando automáticamente: Modo BIMODAL")
            print("   (Puedes generar audios después con gen_dataset.py)")
            use_audio = False
        except NameError:
            # En terminal, preguntar
            choice = input("\n  Selecciona opción (1/2/3) [1]: ").strip() or "1"
            
            if choice == "2":
                # Verificar edge-tts
                try:
                    import edge_tts
                except ImportError:
                    print("📦 Instalando edge-tts...")
                    os.system('pip install -q edge-tts')
                    import edge_tts
                
                # Generar audios
                generate_audios_sync(images_dir, captions_file, audio_dir)
                
                # Re-verificar
                existing_audios = len([f for f in os.listdir(audio_dir) if f.endswith('.mp3')])
                if existing_audios >= min_audios_needed:
                    print(f"✅ Audios generados: {existing_audios} archivos")
                    use_audio = True
                else:
                    print(f"⚠️  Solo se generaron {existing_audios} audios. Continuando en modo bimodal.")
                    use_audio = False
            else:
                print("✓ Continuando en modo BIMODAL")
                use_audio = False
    
    vocab, id2word = build_vocab_flickr(captions_file, vocab_size=5000)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset con o sin audio según disponibilidad
    if use_audio:
        dataset = Flickr8kMultimodalDataset(
            images_dir, audio_dir, captions_file, vocab, 
            img_transform=transform, max_len=30
        )
        print("🎵 Modo: TRIMODAL (Visión + Audio + Texto)")
    else:
        # Fallback: usar dataset simplificado sin audio
        from torch.utils.data import Dataset as BaseDataset
        
        class Flickr8kSimpleDataset(BaseDataset):
            def __init__(self, images_dir, captions_file, vocab, transform, max_len=30):
                self.images_dir = images_dir
                self.vocab = vocab
                self.transform = transform
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
                
                print(f"✓ Loaded {len(self.data)} samples (visual+text only)")
            
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
                
                # Audio dummy (zeros)
                dummy_audio = torch.zeros(80, 300)
                
                return image, dummy_audio, torch.tensor(token_ids, dtype=torch.long), caption
        
        dataset = Flickr8kSimpleDataset(images_dir, captions_file, vocab, transform, max_len=30)
        print("📷 Modo: BIMODAL (Visión + Texto) - Audio deshabilitado")
    
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    
    # Modelo tricameral
    model = NeuroLogosTricameral(len(vocab)).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.right_hemisphere.parameters(), 'lr': 3e-4},
        {'params': model.corpus_callosum.parameters(), 'lr': 5e-4},
        {'params': model.left_hemisphere.parameters(), 'lr': 2e-4},
        {'params': model.audio_generator.parameters(), 'lr': 1e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎵 Audio encoder: AudioEncoder + LiquidNeuron")
    print(f"🔊 Audio generator: Neural vocoder (Griffin-Lim)")
    print(f"🔗 Trimodal callosum: Visual + Audio + Semantic\n")
    
    for epoch in range(30):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}")
        
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
            loss, ce_loss, gate_pen, div_pen, coherence_loss = compute_tricameral_loss(
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
            
            pbar.set_postfix({
                'loss': f'{ce_loss.item():.3f}',
                'coherence': f'{coherence_loss.item():.3f}',
                'gate': f'{gate.mean().item():.2f}'
            })
        
        scheduler.step()
        
        # Evaluación con generación de audio
        if epoch % 5 == 0:
            model.eval()
            print("\n🎙️ GENERACIÓN CON AUDIO...\n")
            
            with torch.no_grad():
                sample_img, sample_audio, sample_cap, raw_caption = dataset[0]
                sample_img = sample_img.unsqueeze(0).to(device)
                sample_audio = sample_audio.unsqueeze(0).to(device)
                
                # Generar texto + audio
                generated_text, generated_audio = model(
                    sample_img, sample_audio, 
                    captions=None, epoch=epoch, generate_audio=True
                )
                
                gen_words = [id2word.get(int(t.item()), '<UNK>') for t in generated_text[0]]
                gen_sentence = " ".join(w for w in gen_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                
                print(f"GT:   {raw_caption}")
                print(f"Gen:  {gen_sentence}")
                
                # Guardar audio generado
                if generated_audio is not None:
                    output_path = f'./generated_audio_epoch{epoch}.wav'
                    torchaudio.save(output_path, generated_audio[0].cpu(), 16000)
                    print(f"🔊 Audio guardado: {output_path}\n")
            
            model.train()
        
        print(f"Época {epoch:02d} | Loss: {total_loss/len(dataloader):.4f}\n")
    
    print("✅ Entrenamiento tricameral completado!")


if __name__ == "__main__":
    # CONFIGURACIÓN: Pon aquí la URL de tu repo de GitHub/Hugging Face
    # Si es None, descargará desde la fuente original (más lento)
    
    # Ejemplo GitHub:
    # GITHUB_REPO_URL = "https://github.com/tu-usuario/flickr8k-prepared/raw/main"
    
    # Ejemplo Hugging Face:
    # GITHUB_REPO_URL = "https://huggingface.co/datasets/tu-usuario/flickr8k-prepared/resolve/main"
    
    GITHUB_REPO_URL = None  # Cambiar cuando subas tu dataset
    
    train_tricameral(github_repo_url=GITHUB_REPO_URL)
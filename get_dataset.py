# =============================================================================
# prepare_and_upload_dataset.py - VERSIÓN OPTIMIZADA
# Solo genera y sube AUDIOS (Flickr8k ya está en GitHub)
# Con checkpoints, rate limiting y reanudación
# =============================================================================

import os
import zipfile
import shutil
from pathlib import Path
import subprocess
import json
import asyncio
import time

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_DIR = Path("flickr8k_prepared")
AUDIO_DIR = BASE_DIR / "Audio_es"
CAPTIONS_FILE = BASE_DIR / "captions_es.txt"
CHECKPOINT_FILE = BASE_DIR / "audio_checkpoint.json"

MAX_AUDIOS_PER_ZIP = 5000  # ~50MB por zip (más manejable)

# =============================================================================
# PASO 1: DESCARGAR CAPTIONS (SI NO EXISTEN)
# =============================================================================
def download_captions_only():
    """Descarga solo los captions de Flickr8k"""
    print("📝 Descargando captions de Flickr8k...")
    
    import urllib.request
    
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CAPTIONS_FILE.exists():
        urllib.request.urlretrieve(
            "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
            "captions.zip"
        )
        
        with zipfile.ZipFile("captions.zip", 'r') as z:
            z.extractall(BASE_DIR)
        
        os.remove("captions.zip")
        
        # Procesar captions
        token_file = BASE_DIR / "Flickr8k.token.txt"
        with open(token_file, 'r', encoding='utf-8') as fin, \
             open(CAPTIONS_FILE, 'w', encoding='utf-8') as fout:
            for line in fin:
                if '\t' not in line:
                    continue
                img_cap, text = line.strip().split('\t')
                img_name = img_cap.split('#')[0]
                fout.write(f"{img_name}\t{text}\n")
        
        print(f"✓ Captions: {sum(1 for _ in open(CAPTIONS_FILE))} líneas")
    else:
        print(f"✓ Captions ya existen")


# =============================================================================
# PASO 2: GENERAR AUDIOS CON CHECKPOINTS
# =============================================================================
async def generate_one_audio(text, output_path, max_retries=3):
    """Genera un audio con retry y rate limiting"""
    try:
        import edge_tts
    except ImportError:
        os.system("pip install -q edge_tts")
        import edge_tts
    
    # Validar texto
    if not text or len(text.strip()) < 3:
        return False
    
    text = text.strip()
    
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(
                text,
                voice="en-US-AriaNeural",
                rate="-5%",
                pitch="+2Hz"
            )
            
            await communicate.save(output_path)
            
            # IMPORTANTE: Pausa de 0.5 segundos entre cada audio
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            
            # Si es error de conexión, esperar más tiempo
            if "Cannot connect" in error_msg or "SSL" in error_msg:
                wait_time = 2 * (attempt + 1)
                await asyncio.sleep(wait_time)
            elif attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                # Solo mostrar si no es el error común
                if "No audio was received" not in error_msg:
                    print(f"\n⚠️  Error con {Path(output_path).stem}: {error_msg}")
    
    return False


def load_checkpoint():
    """Carga el checkpoint de progreso"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'failed': [], 'last_index': 0}


def save_checkpoint(checkpoint):
    """Guarda el checkpoint de progreso"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


async def generate_audios_with_checkpoints():
    """Genera audios con checkpoints cada 500 archivos"""
    
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cargar captions
    caption_list = []
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_name, text = parts
                stem = Path(img_name).stem
                key = f"{stem}_{idx % 5}"
                caption_list.append((key, text.strip()))
    
    print(f"📝 Total de audios a generar: {len(caption_list)}")
    
    # Cargar checkpoint
    checkpoint = load_checkpoint()
    completed_set = set(checkpoint['completed'])
    failed_set = set(checkpoint['failed'])
    
    # Verificar audios existentes
    existing = set()
    if AUDIO_DIR.exists():
        existing = {f.stem for f in AUDIO_DIR.glob("*.mp3")}
    
    print(f"✓ Audios ya completados: {len(completed_set | existing)}")
    print(f"⚠️  Audios fallidos previos: {len(failed_set)}")
    
    # Filtrar tareas pendientes
    tasks_pending = []
    for key, text in caption_list:
        if key not in completed_set and key not in existing:
            output_file = AUDIO_DIR / f"{key}.mp3"
            tasks_pending.append((key, text, output_file))
    
    if len(tasks_pending) == 0:
        print("✅ Todos los audios ya están generados!")
        return
    
    print(f"🎵 Audios pendientes: {len(tasks_pending)}")
    print(f"⏱️  Tiempo estimado: ~{len(tasks_pending) * 0.8 / 60:.1f} minutos")
    print(f"💾 Checkpoints cada 500 audios\n")
    
    start_time = time.time()
    successful = len(completed_set | existing)
    failed = len(failed_set)
    
    # Procesar de uno en uno (más lento pero más seguro)
    for idx, (key, text, output_path) in enumerate(tasks_pending, 1):
        result = await generate_one_audio(text, output_path)
        
        if result:
            checkpoint['completed'].append(key)
            successful += 1
        else:
            checkpoint['failed'].append(key)
            failed += 1
        
        # Progress report cada 50 audios
        if idx % 50 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(tasks_pending) - idx) / rate if rate > 0 else 0
            
            print(f"  📊 {idx}/{len(tasks_pending)} | "
                  f"✓ {successful} | ✗ {failed} | "
                  f"⏱️ {remaining/60:.1f} min restantes")
        
        # Checkpoint cada 500 audios
        if idx % 500 == 0:
            checkpoint['last_index'] = idx
            save_checkpoint(checkpoint)
            print(f"  💾 Checkpoint guardado en: {CHECKPOINT_FILE}")
    
    # Checkpoint final
    save_checkpoint(checkpoint)
    
    elapsed_total = time.time() - start_time
    print(f"\n✅ Generación completada!")
    print(f"   ✓ Exitosos: {successful}")
    print(f"   ✗ Fallidos: {failed}")
    print(f"   ⏱️ Tiempo total: {elapsed_total/60:.1f} minutos")
    
    if failed > len(caption_list) * 0.1:
        print(f"\n⚠️  {failed} audios fallaron (>{10}% del total)")
        print("💡 Recomendación: Ejecuta el script de nuevo para reintentar")


def generate_audios_sync():
    """Wrapper síncrono con manejo de event loop"""
    try:
        get_ipython().__class__.__name__
        in_jupyter = True
    except NameError:
        in_jupyter = False
    
    if in_jupyter:
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            os.system('pip install -q nest_asyncio')
            import nest_asyncio
            nest_asyncio.apply()
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(generate_audios_with_checkpoints())
    else:
        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(generate_audios_with_checkpoints())


# =============================================================================
# PASO 3: COMPRIMIR SOLO AUDIOS
# =============================================================================
def compress_audios_only():
    """Comprime solo los audios en zips pequeños"""
    print("\n📦 Comprimiendo audios...")
    
    output_dir = Path("audio_dataset_zips")
    output_dir.mkdir(exist_ok=True)
    
    # Limpiar zips antiguos
    for old_zip in output_dir.glob("audios_part_*.zip"):
        old_zip.unlink()
    
    if not AUDIO_DIR.exists() or len(list(AUDIO_DIR.glob("*.mp3"))) == 0:
        print("❌ No hay audios para comprimir")
        return None, None
    
    audios = sorted(AUDIO_DIR.glob("*.mp3"))
    total_audios = len(audios)
    
    print(f"  🎵 Total de audios: {total_audios}")
    
    metadata = {
        "total_audios": total_audios,
        "audio_zips": [],
        "captions_file": "captions_es.txt",
        "source": "Edge-TTS (en-US-AriaNeural)",
        "flickr8k_github": "https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k"
    }
    
    # Comprimir en partes
    for i in range(0, len(audios), MAX_AUDIOS_PER_ZIP):
        batch = audios[i:i+MAX_AUDIOS_PER_ZIP]
        part_num = i // MAX_AUDIOS_PER_ZIP + 1
        zip_name = f"audios_part_{part_num}.zip"
        zip_path = output_dir / zip_name
        
        print(f"    Comprimiendo parte {part_num}...", end=' ', flush=True)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for audio in batch:
                zf.write(audio, audio.name)
        
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        metadata["audio_zips"].append({
            "filename": zip_name,
            "size_mb": round(size_mb, 2),
            "count": len(batch)
        })
        
        print(f"✓ {len(batch)} audios ({size_mb:.1f} MB)")
    
    # Copiar captions y checkpoint
    shutil.copy(CAPTIONS_FILE, output_dir / "captions_es.txt")
    if CHECKPOINT_FILE.exists():
        shutil.copy(CHECKPOINT_FILE, output_dir / "audio_checkpoint.json")
    
    # Guardar metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Crear README
    create_audio_readme(output_dir, metadata)
    
    print(f"\n✅ Audios comprimidos en: {output_dir.resolve()}")
    print(f"   Partes: {len(metadata['audio_zips'])}")
    print(f"   Total: {total_audios} audios")
    
    return output_dir, metadata


def create_audio_readme(output_dir, metadata):
    """Crea README para el dataset de audios"""
    
    readme = f"""# Flickr8k Audio Dataset (Edge-TTS)

Audios generados para NeuroLogos Tricameral v5.0

## 📊 Contenido

- **Audios**: {metadata['total_audios']:,} archivos MP3
- **Voz**: Edge-TTS en-US-AriaNeural
- **Captions**: captions_es.txt (TSV format)

## 📦 Archivos ({len(metadata['audio_zips'])} partes)

"""
    
    for zip_info in metadata['audio_zips']:
        readme += f"- `{zip_info['filename']}` - {zip_info['count']} audios ({zip_info['size_mb']} MB)\n"
    
    readme += f"""
## 🚀 Uso en Colab

```python
import urllib.request
import zipfile
import json
import os

# URL de tu repo
REPO_URL = "https://huggingface.co/datasets/TU_USUARIO/flickr8k-audio/resolve/main"
OUTPUT_DIR = "./flickr8k_full/Audio_es"

# Descargar metadata
urllib.request.urlretrieve(f"{{REPO_URL}}/metadata.json", "metadata.json")
with open("metadata.json") as f:
    metadata = json.load(f)

# Descargar y extraer audios
os.makedirs(OUTPUT_DIR, exist_ok=True)

for zip_info in metadata['audio_zips']:
    print(f"Descargando {{zip_info['filename']}}...")
    urllib.request.urlretrieve(
        f"{{REPO_URL}}/{{zip_info['filename']}}", 
        zip_info['filename']
    )
    
    with zipfile.ZipFile(zip_info['filename'], 'r') as z:
        z.extractall(OUTPUT_DIR)
    
    os.remove(zip_info['filename'])

print(f"✅ {{len(metadata['audio_zips'])}} partes descargadas")
```

## 📝 Notas

- Imágenes de Flickr8k: [GitHub oficial](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)
- Captions incluidos en este dataset
- Compatible con NeuroLogos Tricameral v5.0
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)


# =============================================================================
# PASO 4: SUBIR A HUGGING FACE (SOLO AUDIOS)
# =============================================================================
def upload_to_huggingface(dataset_dir):
    """Sube solo audios a Hugging Face"""
    print("\n🤗 SUBIR AUDIOS A HUGGING FACE HUB")
    print("="*60)
    
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("📦 Instalando huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import HfApi, login
    
    print("\n1. Necesitas una cuenta en huggingface.co")
    print("2. Ve a: https://huggingface.co/settings/tokens")
    print("3. Crea un token con permisos de 'write'")
    print("4. Tu username NO debe incluir '@' ni dominios")
    print("   Ejemplo: 'grisun0' ✓  | 'grisun0@proton.me' ✗")
    
    choice = input("\n¿Continuar con upload a Hugging Face? (s/N): ").lower()
    
    if choice != 's':
        return
    
    # Login
    print("\n🔐 Login en Hugging Face...")
    login()
    
    # Crear API
    api = HfApi()
    
    # Obtener username correcto
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"✓ Logged in como: {username}")
    except Exception as e:
        print(f"❌ Error obteniendo username: {e}")
        username = input("Tu username de Hugging Face (sin @): ").strip()
    
    # Nombre del dataset
    repo_name = input("Nombre del dataset [flickr8k-audio]: ").strip() or "flickr8k-audio"
    repo_id = f"{username}/{repo_name}"
    
    # Validar repo_id
    if '@' in repo_id or ' ' in repo_id:
        print(f"❌ Repo ID inválido: {repo_id}")
        print("   No debe contener '@' ni espacios")
        return
    
    # Crear repo
    print(f"\n📤 Creando repo: {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"⚠️  {e}")
    
    # Subir archivos
    print(f"📤 Subiendo audios...")
    try:
        api.upload_folder(
            folder_path=str(dataset_dir),
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        print(f"\n✅ Dataset subido!")
        print(f"🔗 URL: https://huggingface.co/datasets/{repo_id}")
        print(f"\n📝 Usa esta URL en tu código tricameral:")
        print(f'   GITHUB_REPO_URL = "https://huggingface.co/datasets/{repo_id}/resolve/main"')
        
    except Exception as e:
        print(f"❌ Error subiendo: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("🚀 PREPARAR Y SUBIR AUDIOS DE FLICKR8K")
    print("="*80)
    print("\nEste script:")
    print("  1. Descarga captions de Flickr8k (si no existen)")
    print("  2. Genera audios con Edge-TTS (CON CHECKPOINTS)")
    print("  3. Comprime audios en zips pequeños")
    print("  4. Sube a Hugging Face")
    print()
    print("⚡ CARACTERÍSTICAS:")
    print("  - Checkpoints cada 500 audios (reanudable)")
    print("  - Rate limiting (0.5s entre audios)")
    print("  - Saltea audios existentes")
    print()
    
    input("Presiona ENTER para continuar...")
    
    # Paso 1: Descargar captions
    download_captions_only()
    
    # Paso 2: Generar audios con checkpoints
    print("\n🎵 ¿Generar audios?")
    print("  1. Sí, generar ahora (con checkpoints)")
    print("  2. No, ya los tengo")
    print("  3. Solo comprimir y subir existentes")
    
    choice = input("\nOpción [1]: ").strip() or "1"
    
    if choice == "1":
        generate_audios_sync()
    elif choice == "2":
        if not AUDIO_DIR.exists() or len(list(AUDIO_DIR.glob("*.mp3"))) == 0:
            print("❌ No se encontraron audios")
            return
        print(f"✓ Usando audios existentes: {len(list(AUDIO_DIR.glob('*.mp3')))} archivos")
    
    # Paso 3: Comprimir
    dataset_dir, metadata = compress_audios_only()
    
    if dataset_dir is None:
        return
    
    # Paso 4: Subir
    print("\n🎯 ¿Subir a Hugging Face?")
    choice = input("(s/N): ").lower()
    
    if choice == 's':
        upload_to_huggingface(dataset_dir)
    
    print("\n" + "="*80)
    print("✅ COMPLETADO")
    print("="*80)
    print(f"\nAudios preparados en: {dataset_dir.resolve()}")


if __name__ == "__main__":
    main()


# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_DIR = Path("flickr8k_prepared")
IMAGES_DIR = BASE_DIR / "Images"
AUDIO_DIR = BASE_DIR / "Audio_es"
CAPTIONS_FILE = BASE_DIR / "captions_es.txt"

# Límites para subir a GitHub (GitHub tiene límite de 100MB por archivo)
MAX_IMAGES_PER_ZIP = 2000  # ~200MB por zip
MAX_AUDIOS_PER_ZIP = 10000  # ~100MB por zip

# =============================================================================
# PASO 1: DESCARGAR Y PREPARAR FLICKR8K
# =============================================================================
def download_flickr8k():
    """Descarga Flickr8k (solo necesitas ejecutar esto una vez)"""
    print("📥 Descargando Flickr8k...")
    
    import urllib.request
    
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Descargar imágenes
    if not IMAGES_DIR.exists() or len(list(IMAGES_DIR.glob("*.jpg"))) < 8000:
        print("  Descargando imágenes...")
        urllib.request.urlretrieve(
            "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
            "images.zip"
        )
        
        with zipfile.ZipFile("images.zip", 'r') as z:
            z.extractall(BASE_DIR)
        
        os.remove("images.zip")
        
        # Mover imágenes
        old_dir = BASE_DIR / "Flicker8k_Dataset"
        if old_dir.exists():
            if IMAGES_DIR.exists():
                shutil.rmtree(IMAGES_DIR)
            old_dir.rename(IMAGES_DIR)
        
        print(f"  ✓ Imágenes: {len(list(IMAGES_DIR.glob('*.jpg')))} archivos")
    
    # Descargar captions
    if not CAPTIONS_FILE.exists():
        print("  Descargando captions...")
        urllib.request.urlretrieve(
            "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
            "captions.zip"
        )
        
        with zipfile.ZipFile("captions.zip", 'r') as z:
            z.extractall(BASE_DIR)
        
        os.remove("captions.zip")
        
        # Procesar captions
        token_file = BASE_DIR / "Flickr8k.token.txt"
        with open(token_file, 'r', encoding='utf-8') as fin, \
             open(CAPTIONS_FILE, 'w', encoding='utf-8') as fout:
            for line in fin:
                if '\t' not in line:
                    continue
                img_cap, text = line.strip().split('\t')
                img_name = img_cap.split('#')[0]
                fout.write(f"{img_name}\t{text}\n")
        
        print(f"  ✓ Captions: {sum(1 for _ in open(CAPTIONS_FILE))} líneas")
    
    print("✅ Flickr8k preparado\n")


# =============================================================================
# PASO 2: GENERAR AUDIOS (Opcional - puedes usar gen_dataset.py)
# =============================================================================
def generate_audios():
    """Genera audios con Edge-TTS - TOMA TIEMPO (~20-30 min)"""
    print("🎵 ¿Generar audios? (Esto toma ~20-30 minutos)")
    print("  1. Sí, generar ahora")
    print("  2. No, ya los generé con gen_dataset.py")
    print("  3. Saltar (solo imágenes + captions)")
    
    choice = input("\nOpción [3]: ").strip() or "3"
    
    if choice == "1":
        print("\n📦 Instalando edge-tts...")
        os.system("pip install edge-tts nest-asyncio")
        
        print("\n🎵 Generando audios...")
        # Usar el script gen_dataset.py que ya tienes
        os.system("python gen_dataset.py")
        
        if AUDIO_DIR.exists() and len(list(AUDIO_DIR.glob("*.mp3"))) > 5000:
            print(f"✅ Audios generados: {len(list(AUDIO_DIR.glob('*.mp3')))} archivos")
        else:
            print("⚠️  Audios no generados. Continuando sin audio.")
    
    elif choice == "2":
        if AUDIO_DIR.exists() and len(list(AUDIO_DIR.glob("*.mp3"))) > 0:
            print(f"✓ Usando audios existentes: {len(list(AUDIO_DIR.glob('*.mp3')))} archivos")
        else:
            print("⚠️  No se encontraron audios en Audio_es/")
    
    else:
        print("✓ Saltando generación de audios")


# =============================================================================
# PASO 3: COMPRIMIR EN ZIPS PEQUEÑOS (para GitHub)
# =============================================================================
def create_split_zips():
    """Crea múltiples zips pequeños para cumplir límites de GitHub"""
    print("\n📦 Comprimiendo dataset en partes...")
    
    output_dir = Path("dataset_zips")
    output_dir.mkdir(exist_ok=True)
    
    # Metadatos
    metadata = {
        "total_images": 0,
        "total_audios": 0,
        "image_zips": [],
        "audio_zips": [],
        "captions_file": "captions_es.txt"
    }
    
    # 1. Comprimir imágenes en partes
    print("\n  📸 Comprimiendo imágenes...")
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    metadata["total_images"] = len(images)
    
    for i in range(0, len(images), MAX_IMAGES_PER_ZIP):
        batch = images[i:i+MAX_IMAGES_PER_ZIP]
        zip_name = f"images_part_{i//MAX_IMAGES_PER_ZIP + 1}.zip"
        zip_path = output_dir / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for img in batch:
                zf.write(img, img.name)
        
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        metadata["image_zips"].append({
            "filename": zip_name,
            "size_mb": round(size_mb, 2),
            "count": len(batch)
        })
        
        print(f"    ✓ {zip_name}: {len(batch)} imágenes ({size_mb:.1f} MB)")
    
    # 2. Comprimir audios en partes (si existen)
    if AUDIO_DIR.exists() and len(list(AUDIO_DIR.glob("*.mp3"))) > 0:
        print("\n  🎵 Comprimiendo audios...")
        audios = sorted(AUDIO_DIR.glob("*.mp3"))
        metadata["total_audios"] = len(audios)
        
        for i in range(0, len(audios), MAX_AUDIOS_PER_ZIP):
            batch = audios[i:i+MAX_AUDIOS_PER_ZIP]
            zip_name = f"audios_part_{i//MAX_AUDIOS_PER_ZIP + 1}.zip"
            zip_path = output_dir / zip_name
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for audio in batch:
                    zf.write(audio, audio.name)
            
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            metadata["audio_zips"].append({
                "filename": zip_name,
                "size_mb": round(size_mb, 2),
                "count": len(batch)
            })
            
            print(f"    ✓ {zip_name}: {len(batch)} audios ({size_mb:.1f} MB)")
    
    # 3. Copiar captions
    shutil.copy(CAPTIONS_FILE, output_dir / "captions_es.txt")
    
    # 4. Guardar metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset comprimido en: {output_dir.resolve()}")
    print(f"   Partes de imágenes: {len(metadata['image_zips'])}")
    print(f"   Partes de audio: {len(metadata['audio_zips'])}")
    
    return output_dir, metadata


# =============================================================================
# PASO 4: INSTRUCCIONES PARA SUBIR A GITHUB
# =============================================================================
def generate_upload_instructions(metadata):
    """Genera instrucciones para subir a GitHub"""
    
    readme = f"""# Flickr8k Prepared Dataset

Dataset preparado para NeuroLogos Tricameral v5.0

## 📊 Contenido

- **Imágenes**: {metadata['total_images']:,} archivos
- **Audios**: {metadata['total_audios']:,} archivos (Edge-TTS)
- **Captions**: captions_es.txt

## 📦 Archivos

### Imágenes ({len(metadata['image_zips'])} partes)
"""
    
    for zip_info in metadata['image_zips']:
        readme += f"- `{zip_info['filename']}` - {zip_info['count']} imágenes ({zip_info['size_mb']} MB)\n"
    
    if metadata['audio_zips']:
        readme += f"\n### Audios ({len(metadata['audio_zips'])} partes)\n"
        for zip_info in metadata['audio_zips']:
            readme += f"- `{zip_info['filename']}` - {zip_info['count']} audios ({zip_info['size_mb']} MB)\n"
    
    readme += """
### Metadata
- `captions_es.txt` - Captions en formato TSV
- `metadata.json` - Información del dataset

## 🚀 Uso

```python
import urllib.request
import zipfile
import json

# Configurar
REPO_URL = "https://github.com/TU_USUARIO/TU_REPO/raw/main"
OUTPUT_DIR = "./flickr8k_full"

# Descargar metadata
urllib.request.urlretrieve(f"{REPO_URL}/metadata.json", "metadata.json")
with open("metadata.json") as f:
    metadata = json.load(f)

# Descargar y extraer imágenes
for zip_info in metadata['image_zips']:
    urllib.request.urlretrieve(
        f"{REPO_URL}/{zip_info['filename']}", 
        zip_info['filename']
    )
    with zipfile.ZipFile(zip_info['filename'], 'r') as z:
        z.extractall(f"{OUTPUT_DIR}/Images")

# Descargar captions
urllib.request.urlretrieve(
    f"{REPO_URL}/captions_es.txt", 
    f"{OUTPUT_DIR}/captions_es.txt"
)
```

## 📝 Notas

- Dataset original: [Flickr8k](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)
- Audios generados con: Edge-TTS (en-US-AriaNeural)
- Preparado para: NeuroLogos Tricameral v5.0
"""
    
    with open("dataset_zips/README.md", 'w') as f:
        f.write(readme)
    
    print("\n" + "="*80)
    print("📝 INSTRUCCIONES PARA SUBIR A GITHUB")
    print("="*80)
    print("""
1. Crea un nuevo repositorio en GitHub:
   - Nombre sugerido: flickr8k-prepared
   - Público o Privado (tu elección)

2. Inicializa Git en la carpeta dataset_zips:
   
   cd dataset_zips
   git init
   git add .
   git commit -m "Add Flickr8k prepared dataset"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git push -u origin main

3. Si los archivos son muy grandes (>100MB), usa Git LFS:
   
   git lfs install
   git lfs track "*.zip"
   git add .gitattributes
   git commit -m "Add Git LFS tracking"
   git push

4. Alternativa: Hugging Face Hub (más fácil para datasets grandes)
   
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload TU_USUARIO/flickr8k-prepared ./dataset_zips

5. Actualiza la URL en el código tricameral:
   REPO_URL = "https://github.com/TU_USUARIO/TU_REPO/raw/main"
   # o
   REPO_URL = "https://huggingface.co/datasets/TU_USUARIO/flickr8k-prepared/resolve/main"

✅ Una vez subido, Colab descargará todo en ~2-3 minutos!
""")


# =============================================================================
# PASO 5: ALTERNATIVA - HUGGING FACE (MÁS FÁCIL)
# =============================================================================
def upload_to_huggingface(dataset_dir):
    """Sube directamente a Hugging Face (alternativa a GitHub)"""
    print("\n🤗 SUBIR A HUGGING FACE HUB")
    print("="*60)
    
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("📦 Instalando huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import HfApi, login
    
    print("\n1. Necesitas una cuenta en huggingface.co")
    print("2. Ve a: https://huggingface.co/settings/tokens")
    print("3. Crea un token con permisos de 'write'")
    
    choice = input("\n¿Continuar con upload a Hugging Face? (s/N): ").lower()
    
    if choice == 's':
        # Login
        login()
        
        # Crear API
        api = HfApi()
        
        # Nombre del repo
        username = input("Tu username de Hugging Face: ").strip()
        repo_name = input("Nombre del dataset [flickr8k-prepared]: ").strip() or "flickr8k-prepared"
        repo_id = f"{username}/{repo_name}"
        
        # Crear repo
        print(f"\n📤 Creando repo: {repo_id}")
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"⚠️  {e}")
        
        # Subir archivos
        print(f"📤 Subiendo archivos...")
        api.upload_folder(
            folder_path=str(dataset_dir),
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        print(f"\n✅ Dataset subido!")
        print(f"🔗 URL: https://huggingface.co/datasets/{repo_id}")
        print(f"\n📝 Usa esta URL en tu código:")
        print(f'   REPO_URL = "https://huggingface.co/datasets/{repo_id}/resolve/main"')


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("🚀 PREPARAR Y SUBIR FLICKR8K DATASET")
    print("="*80)
    print("\nEste script:")
    print("  1. Descarga Flickr8k (si no lo tienes)")
    print("  2. Opcionalmente genera audios con Edge-TTS")
    print("  3. Comprime todo en zips pequeños para GitHub")
    print("  4. Te da instrucciones para subir")
    print()
    
    input("Presiona ENTER para continuar...")
    
    # Paso 1: Descargar Flickr8k
    download_flickr8k()
    
    # Paso 2: Generar audios (opcional)
    generate_audios()
    
    # Paso 3: Comprimir
    dataset_dir, metadata = create_split_zips()
    
    # Paso 4: Generar instrucciones
    generate_upload_instructions(metadata)
    
    # Paso 5: Opción Hugging Face
    print("\n🎯 OPCIONES DE UPLOAD:")
    print("  1. GitHub (manual - sigue instrucciones arriba)")
    print("  2. Hugging Face Hub (automático)")
    
    choice = input("\nOpción [1]: ").strip() or "1"
    
    if choice == "2":
        upload_to_huggingface(dataset_dir)
    else:
        print("\n✓ Sigue las instrucciones arriba para subir a GitHub")
    
    print("\n" + "="*80)
    print("✅ COMPLETADO")
    print("="*80)
    print(f"\nDataset preparado en: {dataset_dir.resolve()}")
    print("Sube los archivos a GitHub/Hugging Face y actualiza la URL en el código tricameral")


if __name__ == "__main__":
    main()
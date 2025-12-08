# =============================================================================
# gen_dataset.py → Flickr8k + 40 455 audios en español (Edge-TTS)
# Funciona con: python3 gen_dataset.py
# Tiempo en T4: 22–28 min | En tu PC (con buena CPU): 18–35 min
# =============================================================================

import os
import zipfile
import urllib.request
from pathlib import Path
import asyncio
import aiofiles
from tqdm.asyncio import tqdm_asyncio
import edge_tts
import sys

# ------------------------------
# CONFIGURACIÓN
# ------------------------------
base_dir = Path("flickr8k_full")
images_dir = base_dir / "Images"
audio_dir = base_dir / "Audio_es"
captions_file = base_dir / "captions_es.txt"

images_dir.mkdir(parents=True, exist_ok=True)
audio_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------
# 1. DESCARGA Y ORGANIZA FLICKR8K
# ------------------------------
print("Descargando y organizando Flickr8k...")

# Imágenes
if len(list(images_dir.iterdir())) < 8000:
    print("Descargando imágenes (~800 MB)...")
    urllib.request.urlretrieve(
        "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
        "Flickr8k_Dataset.zip"
    )
    with zipfile.ZipFile("Flickr8k_Dataset.zip", 'r') as z:
        z.extractall(base_dir)
    os.remove("Flickr8k_Dataset.zip")
    old = base_dir / "Flicker8k_Dataset"
    if old.exists():
        for f in old.iterdir():
            f.rename(images_dir / f.name)
        old.rmdir()

# Captions
if not captions_file.exists():
    print("Descargando captions...")
    urllib.request.urlretrieve(
        "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
        "Flickr8k_text.zip"
    )
    with zipfile.ZipFile("Flickr8k_text.zip", 'r') as z:
        z.extractall(base_dir)
    os.remove("Flickr8k_text.zip")

    token_file = base_dir / "Flickr8k.token.txt"
    with open(token_file, "r", encoding="utf-8") as fin, \
         open(captions_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if '\t' not in line:
                continue
            img_cap, text = line.strip().split('\t')
            img_name = img_cap.split('#')[0]
            fout.write(f"{img_name}\t{text}\n")

print(f"Imágenes: {len(list(images_dir.iterdir()))}")
print(f"Captions: {sum(1 for _ in open(captions_file))} líneas")

# ------------------------------
# 2. GENERAR AUDIOS CON EDGE-TTS
# ------------------------------
print("\nPreparando generación de audios...")

# Cargar captions
caption_list = []  # (key, text)
with open(captions_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        img_name, text = line.strip().split("\t", 1)
        stem = Path(img_name).stem
        key = f"{stem}_{idx % 5}"  # cada imagen tiene 5 captions → 0..4
        caption_list.append((key, text.strip()))

print(f"Audios a generar: {len(caption_list)}")

async def generate_one(key: str, text: str):
    out_file = audio_dir / f"{key}.mp3"
    if out_file.exists():
        return  # ya existe
    
    try:
        communicate = edge_tts.Communicate(
            text,
            voice="en-US-AriaNeural",      # la mejor voz femenina en inglés 2025
            # voice="en-US-JennyNeural",   # alternativa muy buena
            # voice="en-GB-SoniaNeural",  # acento británico si prefieres
            rate="-5%",      # un poco más lenta y clara (opcional)
            pitch="+2Hz"     # tono natural (opcional)
        )
        async with aiofiles.open(out_file, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    await f.write(chunk["data"])
    except Exception as e:
        print(f"\nError con {key}: {e}")

async def main():
    tasks = [generate_one(key, text) for key, text in caption_list]
    print("Generando audios (esto tardará ~22–28 min en T4)...")
    await tqdm_asyncio.gather(*tasks, desc="Audios generados", smoothing=0.1)
    print(f"\nTERMINADO! Audios guardados en: {audio_dir.resolve()}")

# ------------------------------
# EJECUCIÓN
# ------------------------------
if __name__ == "__main__":
    if sys.platform == "win32":
        # Windows necesita esto
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
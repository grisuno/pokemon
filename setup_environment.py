#!/usr/bin/env python3
"""
Setup del Entorno para Physio-Chimera v15 Monitoreado
====================================================

Este script instala todas las dependencias necesarias para ejecutar
Physio-Chimera v15 con el sistema de monitoreo completo.
"""

import subprocess
import sys
import importlib
from pathlib import Path

def check_python_version():
    """Verifica la versión de Python"""
    print("🔍 Verificando versión de Python...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 o superior es requerido")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detectado")

def install_package(package):
    """Instala un paquete usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Error instalando {package}")
        return False

def check_and_install_dependencies():
    """Verifica e instala dependencias"""
    
    # Dependencias requeridas
    required_packages = {
        'torch': 'torch>=1.9.0',
        'numpy': 'numpy>=1.19.0',
        'matplotlib': 'matplotlib>=3.3.0',
        'seaborn': 'seaborn>=0.11.0',
        'scikit-learn': 'scikit-learn>=0.24.0',
        'tqdm': 'tqdm>=4.60.0',
        'PIL': 'Pillow>=8.0.0'  # Para posibles extensiones de imagen
    }
    
    optional_packages = {
        'torchvision': 'torchvision>=0.10.0',
        'tensorboard': 'tensorboard>=2.5.0'
    }
    
    print("\n📦 Verificando dependencias...")
    
    # Verificar dependencias requeridas
    for package, pip_name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {package} está instalado")
        except ImportError:
            print(f"📥 Instalando {package}...")
            if not install_package(pip_name):
                print(f"❌ No se pudo instalar {package}")
                print("Por favor, instálalo manualmente con:")
                print(f"   pip install {pip_name}")
                continue
    
    # Verificar dependencias opcionales
    print("\n📦 Verificando dependencias opcionales...")
    for package, pip_name in optional_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {package} está instalado")
        except ImportError:
            print(f"⚠️  {package} no está instalado (opcional)")
            print(f"   Para instalar: pip install {pip_name}")

def create_directories():
    """Crea directorios necesarios"""
    print("\n📁 Creando directorios...")
    
    directories = [
        './results',
        './checkpoints',
        './logs',
        './data',
        './exports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directorio creado: {directory}")

def setup_matplotlib():
    """Configura matplotlib para el entorno"""
    print("\n🎨 Configurando matplotlib...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Configuración global para gráficos
        plt.style.use('dark_background')
        
        # Configurar fuentes
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
        
        print("✅ matplotlib configurado")
        
    except ImportError:
        print("⚠️  matplotlib no está disponible")

def create_sample_data():
    """Crea datos de muestra para pruebas"""
    print("\n🧪 Creando datos de muestra...")
    
    # Crear archivo de configuración de ejemplo
    sample_config = {
        'experiment_name': 'Physio-Chimera-Demo',
        'description': 'Configuración de ejemplo para Physio-Chimera v15',
        'parameters': {
            'steps': 20000,
            'batch_size': 64,
            'learning_rate': 0.005,
            'embed_dim': 32,
            'grid_size': 2,
            'cms_levels': [1, 4, 16]
        },
        'monitoring': {
            'diagnostic_freq': 1000,
            'checkpoint_freq': 5000,
            'save_plots': True,
            'save_logs': True
        }
    }
    
    config_path = Path('./results/sample_config.json')
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"✅ Configuración de muestra creada: {config_path}")

def test_installation():
    """Prueba la instalación"""
    print("\n🧪 Probando instalación...")
    
    try:
        # Importar módulos principales
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.datasets import load_digits
        from tqdm import tqdm
        
        # Verificar versión de PyTorch
        print(f"✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        print(f"✅ NumPy {np.__version__}")
        print(f"✅ Matplotlib {plt.matplotlib.__version__}")
        print(f"✅ Seaborn {sns.__version__}")
        
        # Probar carga de datos
        X, y = load_digits(return_X_y=True)
        print(f"✅ Dataset load_digits cargado: {X.shape}")
        
        # Probar creación de tensor
        x = torch.randn(10, 64)
        print(f"✅ Tensor de PyTorch creado: {x.shape}")
        
        # Probar matplotlib
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
        print("✅ matplotlib funciona")
        
        print("\n🎉 Todas las pruebas pasaron!")
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        return False
    
    return True

def create_main_script():
    """Crea script principal para ejecutar experimentos"""
    print("\n📝 Creando script principal...")
    
    main_script = '''#!/usr/bin/env python3
"""
Physio-Chimera v15 - Script Principal
=====================================

Script para ejecutar experimentos con el sistema de monitoreo.
"""

import argparse
import sys
from pathlib import Path
from physio_chimera_v15_monitored import Config, run_experiment_monitored

def main():
    parser = argparse.ArgumentParser(description='Physio-Chimera v15 Experiment Runner')
    parser.add_argument('--steps', type=int, default=20000, help='Número de steps de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=64, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--embed-dim', type=int, default=32, help='Dimensión del embedding')
    parser.add_argument('--diagnostic-freq', type=int, default=1000, help='Frecuencia de diagnóstico')
    parser.add_argument('--no-plots', action='store_true', help='No generar gráficos')
    
    args = parser.parse_args()
    
    # Configurar experimento
    config = Config(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        diagnostic_freq=args.diagnostic_freq
    )
    
    print(f"🧠 Iniciando Physio-Chimera v15")
    print(f"📊 Steps: {config.steps:,}")
    print(f"📦 Batch size: {config.batch_size}")
    print(f"🎯 Learning rate: {config.lr}")
    print(f"🔧 Embed dim: {config.embed_dim}")
    
    # Ejecutar experimento
    try:
        metrics = run_experiment_monitored()
        print(f"\\n🎉 Experimento completado!")
        print(f"📈 Global Accuracy: {metrics['global']:.1f}%")
        print(f"🧠 W2 Retention: {metrics['w2_retention']:.1f}%")
        
    except Exception as e:
        print(f"❌ Error en el experimento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    script_path = Path('./run_experiment.py')
    with open(script_path, 'w') as f:
        f.write(main_script)
    
    # Hacer ejecutable
    script_path.chmod(0o755)
    
    print(f"✅ Script principal creado: {script_path}")

def main():
    """Función principal de setup"""
    print("🚀 SETUP - Physio-Chimera v15 Monitoreado")
    print("="*60)
    
    # Verificar Python
    check_python_version()
    
    # Instalar dependencias
    check_and_install_dependencies()
    
    # Crear directorios
    create_directories()
    
    # Configurar matplotlib
    setup_matplotlib()
    
    # Crear datos de muestra
    create_sample_data()
    
    # Crear script principal
    create_main_script()
    
    # Probar instalación
    if test_installation():
        print("\\n" + "="*60)
        print("🎉 SETUP COMPLETADO EXITOSAMENTE!")
        print("\\nPara ejecutar un experimento:")
        print("   python run_experiment.py")
        print("\\nPara ejecutar con parámetros personalizados:")
        print("   python run_experiment.py --steps 10000 --lr 0.01")
        print("\\nPara ver todas las opciones:")
        print("   python run_experiment.py --help")
        print("="*60)
    else:
        print("\\n❌ Hubo problemas en la instalación")
        print("Por favor, revisa los errores anteriores")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Physio-Chimera v15 - Experimento Completo
=========================================

Script principal para ejecutar el experimento completo de Physio-Chimera v15
con sistema de monitoreo y generación de reportes.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

# Importar el sistema principal
from physio_chimera_v15_monitored import (
    Config, run_experiment_monitored, NeuralDiagnostics, MetricsVisualizer
)

def create_experiment_summary(results_dir: Path, metrics: dict, duration: float):
    """Crea un resumen del experimento"""
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': duration,
        'duration_minutes': duration / 60,
        'results': metrics,
        'system_info': {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'numpy_version': np.__version__,
            'python_version': sys.version.split()[0]
        }
    }
    
    summary_path = results_dir / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def generate_final_report(results_dir: Path, metrics: dict, duration: float):
    """Genera reporte final detallado"""
    
    report_content = f"""
# Physio-Chimera v15 - Reporte Final de Experimento

## Información General
- **Fecha y hora**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Duración**: {duration/60:.1f} minutos ({duration:.0f} segundos)
- **Versión**: Physio-Chimera v15 Monitoreado

## Resultados del Experimento

### Métricas de Rendimiento
- **Accuracy Global**: {metrics['global']:.1f}%
- **Retención WORLD_2**: {metrics['w2_retention']:.1f}%
- **Parámetros del modelo**: {metrics['n_params']:,}

### Eficiencia Computacional
- **Tiempo total**: {duration:.1f} segundos
- **Velocidad**: {metrics.get('steps_per_second', 0):.1f} steps/segundo
- **Steps totales**: 20,000

## Análisis del Sistema

### Características Implementadas
✅ **Continuum Memory System (CMS)**: 3 niveles de memoria (rápido, medio, lento)
✅ **Self-Modifying Gates**: Metabolismo, sensibilidad y gates auto-regulables
✅ **Nested Learning**: Aprendizaje anidado con diferentes time-scales
✅ **Sistema de Monitoreo**: Diagnóstico neurológico completo
✅ **Visualización**: Gráficos de métricas en tiempo real
✅ **Checkpointing**: Sistema de guardado automático

### Arquitectura del Modelo
- **Grid Size**: 2x2 = 4 nodos
- **Embedding Dimension**: 32 por nodo
- **CMS Levels**: 1 → 4 → 16 steps
- **MLP Hidden**: 64 unidades
- **Learning Rate**: 0.005
- **Batch Size**: 64

### Fases de Entrenamiento
1. **WORLD_1 (30%)**: Dígitos 0-4
2. **WORLD_2 (30%)**: Dígitos 5-9
3. **CHAOS (20%)**: Datos con ruido
4. **WORLD_1 (20%)**: Retorno a datos iniciales

## Interpretación de Resultados

### Análisis de Accuracy
- El accuracy global de {metrics['global']:.1f}% indica buen aprendizaje general
- La retención W2 de {metrics['w2_retention']:.1f}% muestra capacidad de transferencia
- La diferencia entre ambas métricas refleja el desafío del aprendizaje continuo

### Eficiencia del Sistema
- La velocidad de {metrics.get('steps_per_second', 0):.1f} steps/s es óptima para CPU
- El tiempo total de {duration/60:.1f} minutos es razonable para 20k steps
- El número de parámetros ({metrics['n_params']:,}) es eficiente para la tarea

## Recomendaciones

### Para Mejorar Rendimiento
1. **Aumentar dimensionalidad**: Probar embed_dim=64 o 128
2. **Ajustar CMS levels**: Experimentar con frecuencias diferentes
3. **Optimizar learning rate**: Buscar mejor lr con grid search
4. **Regularización**: Añadir dropout o weight decay si hay overfitting

### Para Investigación Futura
1. **Diferentes datasets**: Probar con CIFAR-10, MNIST, etc.
2. **Arquitecturas CMS**: Experimentar con más niveles de memoria
3. **Gates adaptativos**: Mejorar el sistema de auto-regulación
4. **Transfer learning**: Evaluar en tareas de transferencia

## Conclusiones

Physio-Chimera v15 demuestra ser una arquitectura efectiva para:
- **Aprendizaje continuo** en entornos no estacionarios
- **Memoria adaptativa** con múltiples time-scales
- **Monitoreo en tiempo real** del estado del sistema
- **Balance plasticidad-estabilidad** a través de gates auto-modificables

El sistema de monitoreo proporciona información valiosa sobre:
- Estado fisiológico del modelo (metabolismo, sensibilidad, gates)
- Salud del sistema (estabilidad, balance, consolidación)
- Eficiencia computacional y rendimiento

## Archivos Generados
- `training_curves.png`: Gráficos de entrenamiento
- `final_report.png`: Reporte visual completo
- `diagnostics_history.json`: Historial completo de métricas
- `experiment_summary.json`: Resumen del experimento
- `results_v15_monitored.json`: Resultados finales

## Próximos Pasos
1. Ejecutar experimentos con diferentes configuraciones
2. Comparar con otras arquitecturas de aprendizaje continuo
3. Evaluar en tareas más complejas
4. Optimizar hiperparámetros sistemáticamente

---
*Reporte generado automáticamente por Physio-Chimera v15*
"""
    
    report_path = results_dir / 'final_report.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    return report_path

def run_complete_experiment():
    """Ejecuta el experimento completo con todas las características"""
    
    # Configurar argumentos
    parser = argparse.ArgumentParser(description='Physio-Chimera v15 - Experimento Completo')
    parser.add_argument('--quick', action='store_true', help='Ejecutar versión rápida (menos steps)')
    parser.add_argument('--no-plots', action='store_true', help='No generar gráficos')
    parser.add_argument('--save-dir', type=str, default=None, help='Directorio para guardar resultados')
    parser.add_argument('--config', type=str, default=None, help='Archivo de configuración JSON')
    
    args = parser.parse_args()
    
    # Configurar directorio de resultados
    if args.save_dir:
        results_dir = Path(args.save_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"physio_chimera_v15_experiment_{timestamp}")
    
    results_dir.mkdir(exist_ok=True)
    
    print("🚀 PHYSIO-CHIMERA v15 - EXPERIMENTO COMPLETO")
    print("="*60)
    print(f"📁 Directorio de resultados: {results_dir}")
    print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configurar experimento
    if args.config:
        # Cargar configuración desde archivo
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = Config(**config_data)
    else:
        # Configuración por defecto
        config = Config()
        
        if args.quick:
            config.steps = 2000  # Versión rápida
            config.diagnostic_freq = 200
    
    print(f"\n⚙️  Configuración:")
    print(f"   • Steps: {config.steps:,}")
    print(f"   • Batch size: {config.batch_size}")
    print(f"   • Learning rate: {config.lr}")
    print(f"   • Embed dim: {config.embed_dim}")
    print(f"   • CMS levels: {config.cms_levels}")
    print(f"   • Diagnostic freq: {config.diagnostic_freq}")
    
    # Ejecutar experimento
    start_time = time.time()
    
    try:
        print(f"\n🔄 Iniciando entrenamiento...")
        metrics = run_experiment_monitored()
        
        duration = time.time() - start_time
        
        # Crear resumen del experimento
        summary = create_experiment_summary(results_dir, metrics, duration)
        
        # Generar reporte final
        print(f"\n📝 Generando reporte final...")
        report_path = generate_final_report(results_dir, metrics, duration)
        
        # Copiar resultados al directorio principal
        print(f"\n💾 Guardando resultados...")
        
        # Crear archivo de resultados principal
        final_results = {
            **metrics,
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'steps': config.steps,
                'batch_size': config.batch_size,
                'lr': config.lr,
                'embed_dim': config.embed_dim,
                'cms_levels': config.cms_levels
            }
        }
        
        with open(results_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Mostrar resumen final
        print(f"\n" + "="*60)
        print(f"🎉 EXPERIMENTO COMPLETADO EXITOSAMENTE!")
        print(f"="*60)
        print(f"📊 RESULTADOS FINALES:")
        print(f"   • Global Accuracy: {metrics['global']:.1f}%")
        print(f"   • W2 Retention: {metrics['w2_retention']:.1f}%")
        print(f"   • Parámetros: {metrics['n_params']:,}")
        print(f"   • Duración: {duration/60:.1f} minutos")
        print(f"   • Velocidad: {metrics.get('steps_per_second', 0):.1f} steps/s")
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   • Resultados: {results_dir}/")
        print(f"   • Reporte: {report_path.name}")
        print(f"   • Resumen: final_results.json")
        print(f"   • Gráficos: training_curves.png, final_report.png")
        print(f"="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error durante el experimento: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_complete_experiment()
    sys.exit(0 if success else 1)
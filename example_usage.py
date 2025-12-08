#!/usr/bin/env python3
"""
Ejemplo de uso del sistema Physio-Chimera v15 Monitoreado
=======================================================

Este script demuestra cómo usar el sistema de monitoreo completo
de Physio-Chimera v15 para experimentos de aprendizaje continuo.
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from physio_chimera_v15_monitored import (
    Config, run_experiment_monitored, NeuralDiagnostics, 
    MetricsVisualizer, DataEnvironment
)

def demo_simple_monitoring():
    """Demostración de monitoreo básico"""
    print("🧪 Demo: Monitoreo Simple")
    print("="*50)
    
    # Configuración rápida para demo
    config = Config(
        steps=1000,  # Menos steps para demo
        batch_size=32,
        diagnostic_freq=200,  # Reportes más frecuentes
        monitor_window=50
    )
    
    # Ejecutar experimento
    metrics = run_experiment_monitored()
    
    return metrics

def demo_custom_monitoring():
    """Demostración de monitoreo personalizado"""
    print("🧪 Demo: Monitoreo Personalizado")
    print("="*50)
    
    # Crear entorno de datos
    env = DataEnvironment()
    
    # Configuración personalizada
    config = Config(
        steps=5000,
        lr=0.01,
        cms_levels=(1, 8, 32),  # Frecuencias diferentes
        embed_dim=64,
        diagnostic_freq=500,
        monitor_window=100
    )
    
    # Crear sistema de diagnóstico personalizado
    diagnostics = NeuralDiagnostics(config)
    
    # Simular algunas métricas
    for step in range(100):
        # Simular pérdida decreciente
        loss = 2.0 * np.exp(-step/50) + 0.1 * np.random.randn()
        
        # Simular métricas fisiológicas
        metabolism = 0.5 + 0.3 * np.sin(step/20) + 0.1 * np.random.randn()
        sensitivity = 0.4 + 0.2 * np.cos(step/15) + 0.1 * np.random.randn()
        gate = 0.6 + 0.2 * np.sin(step/25) + 0.05 * np.random.randn()
        
        # Actualizar diagnóstico
        diagnostics.update_performance_metrics(loss)
        diagnostics.update_physio_metrics(metabolism, sensitivity, gate)
        diagnostics.update_memory_metrics([1, 0, 0], 0.5 + 0.1*np.random.randn(), 0.3)
        
        if step % 20 == 0:
            report = diagnostics.generate_diagnostic_report(step, "DEMO")
            print(report)
    
    # Generar visualización
    visualizer = MetricsVisualizer("./demo_results")
    visualizer.plot_training_curves(diagnostics)
    
    print("✅ Demo de monitoreo personalizado completado")
    return diagnostics

def demo_checkpoint_system():
    """Demostración del sistema de checkpointing"""
    print("🧪 Demo: Sistema de Checkpointing")
    print("="*50)
    
    checkpoint_dir = Path("./demo_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Simular checkpoints
    checkpoints = []
    for epoch in range(5):
        checkpoint_data = {
            'epoch': epoch,
            'loss': 2.0 * np.exp(-epoch/3) + np.random.randn() * 0.1,
            'accuracy': 50 + 10 * epoch + np.random.randn() * 2,
            'model_state': f'model_weights_epoch_{epoch}',
            'diagnostics': {
                'metabolism': 0.5 + 0.1 * np.sin(epoch),
                'sensitivity': 0.4 + 0.1 * np.cos(epoch),
                'gate': 0.6 + 0.05 * np.sin(epoch*2)
            }
        }
        
        # Guardar checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        checkpoints.append(checkpoint_data)
        print(f"💾 Checkpoint guardado: {checkpoint_path}")
    
    # Crear visualización de checkpoints
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = [c['epoch'] for c in checkpoints]
    losses = [c['loss'] for c in checkpoints]
    accuracies = [c['accuracy'] for c in checkpoints]
    
    axes[0].plot(epochs, losses, 'o-', color='#E74C3C', linewidth=2, markersize=8)
    axes[0].set_title('Evolución de Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, accuracies, 'o-', color='#27AE60', linewidth=2, markersize=8)
    axes[1].set_title('Evolución de Accuracy', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(checkpoint_dir / 'checkpoint_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Demo de checkpointing completado")
    return checkpoints

def demo_comparison_experiments():
    """Demostración de comparación entre experimentos"""
    print("🧪 Demo: Comparación de Experimentos")
    print("="*50)
    
    # Simular diferentes configuraciones
    experiments = {
        'baseline': {'lr': 0.005, 'embed_dim': 32, 'final_acc': 85.2},
        'optimized': {'lr': 0.01, 'embed_dim': 64, 'final_acc': 89.1},
        'large': {'lr': 0.003, 'embed_dim': 128, 'final_acc': 87.3}
    }
    
    # Crear gráfico de comparación
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Comparación de configuraciones
    exp_names = list(experiments.keys())
    lrs = [experiments[exp]['lr'] for exp in exp_names]
    embed_dims = [experiments[exp]['embed_dim'] for exp in exp_names]
    accuracies = [experiments[exp]['final_acc'] for exp in exp_names]
    
    axes[0].bar(exp_names, accuracies, color=['#E74C3C', '#27AE60', '#3498DB'], alpha=0.8)
    axes[0].set_title('Comparación de Accuracy Final', fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(80, 92)
    
    # Añadir valores en las barras
    for i, (exp, acc) in enumerate(zip(exp_names, accuracies)):
        axes[0].text(i, acc + 0.5, f'{acc}%', ha='center', va='bottom', fontweight='bold')
    
    # Comparación de hiperparámetros
    x = np.arange(len(exp_names))
    width = 0.35
    
    axes[1].bar(x - width/2, [lr*1000 for lr in lrs], width, label='Learning Rate (x1000)', 
                color='#F39C12', alpha=0.8)
    axes[1].bar(x + width/2, embed_dims, width, label='Embed Dimension', 
                color='#9B59B6', alpha=0.8)
    
    axes[1].set_title('Hiperparámetros por Experimento', fontweight='bold')
    axes[1].set_ylabel('Valor')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(exp_names)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('./experiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Demo de comparación completada")
    return experiments

def create_demo_report():
    """Crear reporte demo completo"""
    print("🧪 Demo: Reporte Completo")
    print("="*50)
    
    # Crear directorio de resultados
    results_dir = Path("./demo_complete_report")
    results_dir.mkdir(exist_ok=True)
    
    # Simular resultados de experimento
    experiment_results = {
        'config': {
            'steps': 20000,
            'batch_size': 64,
            'lr': 0.005,
            'embed_dim': 32,
            'cms_levels': [1, 4, 16]
        },
        'metrics': {
            'global_accuracy': 87.5,
            'w2_retention': 82.3,
            'training_loss': 0.15,
            'convergence_steps': 15000
        },
        'system_health': {
            'stability': 0.85,
            'plasticity_balance': 0.78,
            'memory_consolidation': 0.72,
            'cognitive_load': 0.45
        }
    }
    
    # Guardar resultados
    with open(results_dir / 'experiment_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    # Crear visualización del reporte
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Physio-Chimera v15 - Reporte Demo Completo', 
                fontsize=20, fontweight='bold', color='#2C3E50')
    
    # Grid de subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Métricas principales
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['Global Acc.', 'W2 Retention', 'Stability', 'Plasticity Balance']
    values = [87.5, 82.3, 85, 78]
    bars = ax1.bar(metrics, values, color=['#E74C3C', '#27AE60', '#3498DB', '#F39C12'], alpha=0.8)
    ax1.set_title('Métricas Principales', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Valor (%)')
    
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value}%', ha='center', va='bottom', fontweight='bold')
    
    # Configuración
    ax2 = fig.add_subplot(gs[0, 2])
    config_text = f"""Configuración:
Steps: {experiment_results['config']['steps']:,}
Batch Size: {experiment_results['config']['batch_size']}
Learning Rate: {experiment_results['config']['lr']}
Embed Dim: {experiment_results['config']['embed_dim']}
CMS Levels: {'→'.join(map(str, experiment_results['config']['cms_levels']))}"""
    
    ax2.text(0.05, 0.95, config_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#ECF0F1', alpha=0.8))
    ax2.axis('off')
    ax2.set_title('Configuración', fontweight='bold')
    
    # Evolución temporal (simulada)
    ax3 = fig.add_subplot(gs[1, :])
    steps = np.linspace(0, 20000, 100)
    loss_curve = 2.0 * np.exp(-steps/5000) + 0.1 * np.random.randn(100)
    accuracy_curve = 50 + 37.5 * (1 - np.exp(-steps/3000)) + 2 * np.random.randn(100)
    
    ax3_twin = ax3.twinx()
    ax3.plot(steps, loss_curve, color='#E74C3C', linewidth=2, label='Loss')
    ax3_twin.plot(steps, accuracy_curve, color='#27AE60', linewidth=2, label='Accuracy')
    
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Loss', color='#E74C3C')
    ax3_twin.set_ylabel('Accuracy (%)', color='#27AE60')
    ax3.set_title('Evolución del Entrenamiento', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Health metrics
    ax4 = fig.add_subplot(gs[2, 0])
    health_metrics = ['Stability', 'Plasticity', 'Consolidation', 'Cognitive Load']
    health_values = [0.85, 0.78, 0.72, 0.45]
    colors = ['#27AE60', '#F39C12', '#3498DB', '#E67E22']
    
    bars = ax4.bar(health_metrics, health_values, color=colors, alpha=0.8)
    ax4.set_title('System Health', fontweight='bold')
    ax4.set_ylabel('Score (0-1)')
    ax4.set_ylim(0, 1)
    
    for bar, value in zip(bars, health_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Arquitectura
    ax5 = fig.add_subplot(gs[2, 1])
    arch_text = """Arquitectura:
• 4 nodos (2x2 grid)
• 32 dim. embedding
• CMS multinivel
• Gates auto-modificables
• Hebbian learning"""
    
    ax5.text(0.05, 0.95, arch_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#BDC3C7', alpha=0.8))
    ax5.axis('off')
    ax5.set_title('Arquitectura', fontweight='bold')
    
    # Recomendaciones
    ax6 = fig.add_subplot(gs[2, 2])
    rec_text = """Recomendaciones:
✅ Sistema estable
✅ Buen balance
⚠️ Mejorar convergencia
⚠️ Optimizar memoria"""
    
    ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#27AE60', alpha=0.8))
    ax6.axis('off')
    ax6.set_title('Recomendaciones', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'complete_demo_report.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Reporte demo completo creado")
    return results_dir

def main():
    """Función principal de demostración"""
    print("🚀 PHYSIO-CHIMERA v15 - SISTEMA DE MONITOREO")
    print("="*60)
    print("Este demo muestra las capacidades del sistema de monitoreo:")
    print("• Diagnóstico neurológico en tiempo real")
    print("• Visualización de métricas")
    print("• Sistema de checkpointing")
    print("• Análisis comparativo")
    print("• Reportes completos")
    print("="*60)
    
    # Ejecutar demos
    print("\n1. Ejecutando demo de monitoreo simple...")
    demo_simple_monitoring()
    
    print("\n2. Ejecutando demo de monitoreo personalizado...")
    demo_custom_monitoring()
    
    print("\n3. Ejecutando demo de checkpointing...")
    demo_checkpoint_system()
    
    print("\n4. Ejecutando demo de comparación...")
    demo_comparison_experiments()
    
    print("\n5. Creando reporte demo completo...")
    create_demo_report()
    
    print("\n" + "="*60)
    print("✅ TODOS LOS DEMOS COMPLETADOS")
    print("📁 Revisa los archivos generados para ver los resultados")
    print("="*60)

if __name__ == "__main__":
    main()
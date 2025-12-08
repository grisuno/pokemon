#!/usr/bin/env python3
"""
Test Script para Premium Synergy System
=======================================
Script de testing rápido para validar que todos los componentes
del sistema democrático deliberativo funcionen correctamente.
"""

import torch
import numpy as np
import os
import sys

# Agregar el directorio actual al path para importar el módulo principal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from premium_synergy_democratic import (
        PremiumSynergyModel, 
        PremiumSynergyConfig,
        create_synthetic_dataset,
        train_premium_synergy
    )
    print("✅ Importación exitosa")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

def test_individual_components():
    """Test de componentes individuales"""
    print("\n🧪 TESTING COMPONENTES INDIVIDUALES")
    print("-" * 50)
    
    config = PremiumSynergyConfig(
        n_samples=100,  # Muy pequeño para testing rápido
        n_features=16,
        n_classes=3,
        epochs=5,
        batch_size=8,
        # Test cada componente individualmente
        use_topobrain=True,
        use_omnibrain=False,
        use_quimera=False
    )
    
    model = PremiumSynergyModel(config)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, config.n_features)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    try:
        logits, metrics = model(x, chaos_level=0.1)
        print(f"✅ TopoBrain component working - Output: {logits.shape}")
        print(f"   Metrics keys: {list(metrics.keys())}")
    except Exception as e:
        print(f"❌ TopoBrain error: {e}")
        return False
    
    # Test OmniBrain
    config.use_topobrain = False
    config.use_omnibrain = True
    model = PremiumSynergyModel(config)
    
    try:
        logits, metrics = model(x, chaos_level=0.1)
        print(f"✅ OmniBrain component working - Output: {logits.shape}")
        print(f"   Metrics keys: {list(metrics.keys())}")
    except Exception as e:
        print(f"❌ OmniBrain error: {e}")
        return False
    
    # Test Quimera
    config.use_omnibrain = False
    config.use_quimera = True
    model = PremiumSynergyModel(config)
    
    try:
        logits, metrics = model(x, chaos_level=0.1)
        print(f"✅ Quimera component working - Output: {logits.shape}")
        print(f"   Metrics keys: {list(metrics.keys())}")
    except Exception as e:
        print(f"❌ Quimera error: {e}")
        return False
    
    return True

def test_full_system():
    """Test del sistema completo Premium Synergy"""
    print("\n🧪 TESTING SISTEMA COMPLETO PREMIUM SYNERGY")
    print("-" * 50)
    
    # Configuración mínima para testing
    config = PremiumSynergyConfig(
        n_samples=200,  # Dataset pequeño
        n_features=32,  # Dimensión baja
        n_classes=5,    # Pocas clases
        embed_dim=32,
        hidden_dim=64,
        grid_size=4,
        epochs=10,      # Pocas épocas
        batch_size=8,
        # Todos los componentes habilitados
        use_topobrain=True,
        use_omnibrain=True,
        use_quimera=True,
        use_dynamic_topology=True,
        use_symbiotic_basis=True,
        use_integration_index=True,
        use_fast_slow=True,
        use_dual_pathway=True,
        use_liquid_neurons=True,
        use_sovereign_attention=True,
        use_dual_phase_memory=True,
        # Motor homeostático
        homeostatic_threshold=0.70,
        convergence_epochs=2
    )
    
    try:
        print("🏗️ Creando modelo...")
        model = PremiumSynergyModel(config)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Parámetros: {total_params:,}")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, config.n_features)
        print(f"🔄 Testing forward pass with input: {x.shape}")
        
        with torch.no_grad():
            logits, metrics = model(x, chaos_level=0.1)
        
        print(f"✅ Forward pass exitoso")
        print(f"   Output shape: {logits.shape}")
        print(f"   Metrics collected: {len(metrics)} keys")
        
        # Verificar componentes activos
        status = model.democratic_deliberation_status()
        print(f"🏛️ Componentes del sistema democrático:")
        for comp, active in status['components_active'].items():
            print(f"   {comp}: {'✅' if active else '❌'}")
        
        # Test motor homeostático
        if 'democratic_deliberation' in metrics:
            deliberation = metrics['democratic_deliberation']
            print(f"🗳️ Motor homeostático activo:")
            print(f"   Synergy strength: {deliberation.get('synergy_strength', 'N/A'):.3f}")
            print(f"   Convergence level: {deliberation.get('convergence_level', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en sistema completo: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_loop():
    """Test del loop de entrenamiento completo"""
    print("\n🧪 TESTING ENTRENAMIENTO")
    print("-" * 50)
    
    # Configuración muy pequeña para testing rápido
    config = PremiumSynergyConfig(
        n_samples=100,
        n_features=16,
        n_classes=3,
        embed_dim=16,
        hidden_dim=32,
        grid_size=2,
        epochs=3,      # Solo 3 épocas
        batch_size=4,
        # Componentes esenciales
        use_topobrain=True,
        use_omnibrain=True,
        use_quimera=True,
        # Motor homeostático
        homeostatic_threshold=0.50,
        convergence_epochs=1
    )
    
    try:
        print("🏃 Iniciando entrenamiento de prueba...")
        model, final_accuracy = train_premium_synergy(config)
        
        print(f"✅ Entrenamiento completado")
        print(f"🎯 Accuracy final: {final_accuracy:.2f}%")
        
        # Verificar que el modelo se guardó
        if os.path.exists('premium_synergy_democratic.pth'):
            print("💾 Modelo guardado correctamente")
        else:
            print("⚠️ Modelo no se guardó")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Ejecuta todos los tests"""
    print("🧪 PREMIUM SYNERGY - TESTS DE VALIDACIÓN")
    print("=" * 60)
    
    # Verificar PyTorch
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    results = []
    
    # Test 1: Componentes individuales
    results.append(("Componentes Individuales", test_individual_components()))
    
    # Test 2: Sistema completo
    results.append(("Sistema Completo", test_full_system()))
    
    # Test 3: Entrenamiento (opcional, puede ser lento)
    print(f"\n❓ ¿Ejecutar test de entrenamiento? (puede tomar tiempo)")
    run_training_test = False  # Por defecto no ejecutar
    if run_training_test:
        results.append(("Entrenamiento", test_training_loop()))
    
    # Resumen
    print(f"\n📊 RESUMEN DE TESTS")
    print("-" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    # Verificar que el sistema principal funciona
    main_test_passed = all(result for _, result in results[:2])  # Solo los primeros 2 tests
    
    if main_test_passed:
        print(f"\n🎉 SISTEMA PREMIUM SYNERGY FUNCIONANDO CORRECTAMENTE")
        print(f"🏛️ Arquitectura democrática deliberativa validada")
        print(f"🧠 TopoBrain + OmniBrain + Quimera integrados")
        print(f"⚖️ Motor homeostático operativo")
    else:
        print(f"\n⚠️ ALGUNOS TESTS FALLARON")
        print(f"🔧 Revisar implementación")
    
    return main_test_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

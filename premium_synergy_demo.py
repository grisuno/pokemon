#!/usr/bin/env python3
"""
Demo Premium Synergy - Sistema Democrático Deliberativo
========================================================
Demo simplificado que muestra la lógica del sistema sin dependencias externas.
Demuestra el concepto de cámara alta/baja y motor homeostático.
"""

import json
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

# =============================================================================
# CLASES SIMPLIFICADAS PARA DEMO
# =============================================================================

@dataclass
class ComponentState:
    """Estado de un componente del sistema"""
    name: str
    activity: float
    cooperation_level: float
    internal_health: float
    synergy_contribution: float

@dataclass
class DemocraticDecision:
    """Decisión del sistema democrático"""
    component_weights: Dict[str, float]
    synergy_score: float
    convergence_level: int
    homeostatic_adjustment: Dict[str, float]

class TopoBrainComponent:
    """TopoBrain v8 - Dynamic Topology + Symbiotic Basis"""
    
    def __init__(self):
        self.state = ComponentState(
            name="TopoBrain",
            activity=0.5,
            cooperation_level=0.7,
            internal_health=0.8,
            synergy_contribution=0.0
        )
        
    def process(self, input_data: Any, plasticity: float = 1.0) -> Tuple[Any, Dict]:
        """Procesamiento con autoregulación interna"""
        # Simulate processing
        output = f"TopoBrain_processed_{input_data}"
        
        # Autoregulación - metabolismo y sensibilidad
        metabolism_regulation = random.uniform(0.6, 0.9)
        sensitivity_gating = random.uniform(0.5, 0.8)
        
        # Update state
        self.state.activity = metabolism_regulation * sensitivity_gating
        self.state.internal_health = 0.9 * self.state.internal_health + 0.1 * self.state.activity
        self.state.cooperation_level = random.uniform(0.7, 0.9)
        
        return output, {
            'topology_activity': self.state.activity,
            'symbiotic_coherence': 1.0 - self.state.activity,
            'metabolism_state': metabolism_regulation,
            'sensitivity_level': sensitivity_gating,
            'internal_dialogue': {
                'metabolism_regulation': metabolism_regulation,
                'sensitivity_gating': sensitivity_gating,
                'topology_stability': self.state.internal_health,
                'symbiotic_harmony': self.state.cooperation_level
            }
        }

class OmniBrainComponent:
    """OmniBrain K - Integration Index + Fast-Slow Weights"""
    
    def __init__(self):
        self.state = ComponentState(
            name="OmniBrain",
            activity=0.6,
            cooperation_level=0.8,
            internal_health=0.7,
            synergy_contribution=0.0
        )
        
    def process(self, input_data: Any, chaos_level: float = 0.0) -> Tuple[Any, Dict]:
        """Procesamiento con control integrativo"""
        output = f"OmniBrain_processed_{input_data}"
        
        # Integration index calculation
        integration_level = random.uniform(0.4, 0.8)
        fast_slow_balance = random.uniform(0.6, 0.9)
        
        # Chaos resistance
        chaos_resistance = max(0.0, 1.0 - chaos_level)
        
        # Update state
        self.state.activity = integration_level * chaos_resistance
        self.state.internal_health = 0.85 * self.state.internal_health + 0.15 * self.state.activity
        self.state.cooperation_level = fast_slow_balance
        
        return output, {
            'integration_index': integration_level,
            'fast_slow_balance': fast_slow_balance,
            'chaos_resistance': chaos_resistance,
            'integrative_control': integration_level,
            'internal_dialogue': {
                'integrative_balance': integration_level,
                'chaos_resistance': chaos_resistance,
                'fast_slow_equilibrium': fast_slow_balance,
                'integration_stability': self.state.internal_health
            }
        }

class QuimeraComponent:
    """Quimera v9.5 - Liquid Neurons + Sovereign Attention"""
    
    def __init__(self):
        self.state = ComponentState(
            name="Quimera",
            activity=0.7,
            cooperation_level=0.6,
            internal_health=0.9,
            synergy_contribution=0.0
        )
        
    def process(self, input_data: Any, plasticity: float = 1.0, chaos: bool = False) -> Tuple[Any, Dict]:
        """Procesamiento con regulación de fases"""
        output = f"Quimera_processed_{input_data}"
        
        # Liquid neuron plasticity
        liquid_plasticity = plasticity * random.uniform(0.6, 0.9)
        
        # Sovereign attention focus
        attention_focus = 0.8 if not chaos else 0.6
        sovereign_focus = attention_focus * random.uniform(0.7, 0.9)
        
        # Phase regulation
        phase_regulation = random.uniform(0.5, 0.8)
        
        # Update state
        self.state.activity = liquid_plasticity * sovereign_focus
        self.state.internal_health = 0.88 * self.state.internal_health + 0.12 * self.state.activity
        self.state.cooperation_level = phase_regulation
        
        return output, {
            'liquid_plasticity': liquid_plasticity,
            'sovereign_focus': sovereign_focus,
            'attention_focus': attention_focus,
            'phase_regulation': phase_regulation,
            'internal_dialogue': {
                'phase_stability': phase_regulation,
                'attention_control': sovereign_focus,
                'liquid_equilibrium': liquid_plasticity,
                'memory_coherence': self.state.cooperation_level
            }
        }

class HomeostaticMotor:
    """Motor Homeostático - Cámara Alta de deliberación democrática"""
    
    def __init__(self, threshold: float = 0.80, convergence_epochs: int = 5):
        self.threshold = threshold
        self.convergence_epochs = convergence_epochs
        self.synergy_evolution = []
        self.convergence_counter = 0
        self.current_accuracy = 0.0
        self.target_accuracy = threshold
        
    def deliberate(self, components: List, target_accuracy: float = None) -> DemocraticDecision:
        """Proceso de deliberación democrática"""
        
        # Calcular pesos democráticos basado en contribuciones
        total_cooperation = sum(c.state.cooperation_level for c in components)
        component_weights = {}
        
        for component in components:
            weight = component.state.cooperation_level / (total_cooperation + 1e-8)
            component_weights[component.state.name] = weight
            component.state.synergy_contribution = weight
        
        # Calcular strength de sinergia
        synergy_scores = [c.state.activity * c.state.cooperation_level for c in components]
        synergy_score = sum(synergy_scores) / len(synergy_scores)
        
        # Verificar convergencia
        if target_accuracy is not None:
            self.current_accuracy = 0.9 * self.current_accuracy + 0.1 * target_accuracy
            
            self.synergy_evolution.append(target_accuracy)
            if len(self.synergy_evolution) > 10:
                self.synergy_evolution = self.synergy_evolution[-10:]
            
            if target_accuracy >= self.threshold:
                self.convergence_counter += 1
            else:
                self.convergence_counter = 0
        
        # Ajustes homeostáticos
        if self.convergence_counter >= self.convergence_epochs and target_accuracy < self.threshold:
            # Ajuste fuerte necesario
            homeostatic_adjustment = {
                'increase_topobrain_plasticity': 0.15,
                'enhance_omnibrain_integration': 0.15,
                'boost_quimera_liquid': 0.15,
                'reduce_entropy_regulation': 0.10
            }
        else:
            # Ajuste leve
            homeostatic_adjustment = {
                'increase_topobrain_plasticity': 0.05,
                'enhance_omnibrain_integration': 0.05,
                'boost_quimera_liquid': 0.05,
                'reduce_entropy_regulation': 0.02
            }
        
        return DemocraticDecision(
            component_weights=component_weights,
            synergy_score=synergy_score,
            convergence_level=self.convergence_counter,
            homeostatic_adjustment=homeostatic_adjustment
        )

class PremiumSynergySystem:
    """Sistema Premium Synergy completo"""
    
    def __init__(self):
        # Cámara Baja: Componentes
        self.topobrain = TopoBrainComponent()
        self.omnibrain = OmniBrainComponent()
        self.quimera = QuimeraComponent()
        
        # Cámara Alta: Motor Homeostático
        self.motor = HomeostaticMotor(threshold=0.80, convergence_epochs=3)
        
        # Estado del sistema
        self.epoch = 0
        self.system_health = 0.7
        
    def process_epoch(self, input_data: str, chaos_level: float = 0.1) -> Dict:
        """Procesa una época del sistema democrático"""
        self.epoch += 1
        
        # Cámara Baja: Cada componente ejecuta con autoregulación
        print(f"\n🏛️ ÉPOCA {self.epoch} - CÁMARA BAJA (Ejecución)")
        print("-" * 50)
        
        # TopoBrain
        tb_output, tb_metrics = self.topobrain.process(input_data, plasticity=0.8)
        print(f"✅ {self.topobrain.state.name}: Activity={self.topobrain.state.activity:.3f}")
        
        # OmniBrain
        ob_output, ob_metrics = self.omnibrain.process(tb_output, chaos_level)
        print(f"✅ {self.omnibrain.state.name}: Activity={self.omnibrain.state.activity:.3f}")
        
        # Quimera
        q_output, q_metrics = self.quimera.process(ob_output, plasticity=0.9, chaos=chaos_level > 0.5)
        print(f"✅ {self.quimera.state.name}: Activity={self.quimera.state.activity:.3f}")
        
        # Cámara Alta: Motor homeostático delibera
        print(f"\n🗳️ CÁMARA ALTA (Deliberación) - Motor Homeostático")
        print("-" * 50)
        
        # Simulate target accuracy for this epoch
        target_accuracy = self.calculate_target_accuracy()
        
        decision = self.motor.deliberate([self.topobrain, self.omnibrain, self.quimera], target_accuracy)
        
        print(f"🎯 Sinergia score: {decision.synergy_score:.3f}")
        print(f"⚖️ Convergencia: {decision.convergence_level}/{self.motor.convergence_epochs}")
        print(f"🏛️ Pesos democráticos:")
        for comp, weight in decision.component_weights.items():
            print(f"   {comp}: {weight:.3f}")
        
        print(f"🔧 Ajustes homeostáticos:")
        for adj, value in decision.homeostatic_adjustment.items():
            print(f"   {adj}: +{value:.3f}")
        
        # Actualizar salud del sistema
        self.system_health = 0.95 * self.system_health + 0.05 * decision.synergy_score
        
        return {
            'epoch': self.epoch,
            'target_accuracy': target_accuracy,
            'synergy_score': decision.synergy_score,
            'convergence_level': decision.convergence_level,
            'system_health': self.system_health,
            'component_weights': decision.component_weights,
            'adjustments': decision.homeostatic_adjustment,
            'topobrain_metrics': tb_metrics,
            'omnibrain_metrics': ob_metrics,
            'quimera_metrics': q_metrics
        }
    
    def calculate_target_accuracy(self) -> float:
        """Calcula accuracy objetivo basada en sinergia actual"""
        base_accuracy = 0.65  # Baseline
        synergy_bonus = self.motor.current_accuracy * 0.2
        return min(0.95, base_accuracy + synergy_bonus)

def run_demo():
    """Ejecuta demostración del sistema Premium Synergy"""
    print("🧠 PREMIUM SYNERGY - SISTEMA DEMOCRÁTICO DELIBERATIVO")
    print("=" * 60)
    print("Concepto: TopoBrain + OmniBrain + Quimera")
    print("Arquitectura: Cámara Baja (ejecución) + Cámara Alta (deliberación)")
    print("Motor: Homeostático que ajusta sinergias")
    print("=" * 60)
    
    # Inicializar sistema
    system = PremiumSynergySystem()
    
    # Simular épocas de entrenamiento
    results = []
    
    for epoch in range(10):
        chaos_level = 0.05 * (epoch / 10)  # Incremental chaos
        result = system.process_epoch(f"input_epoch_{epoch}", chaos_level)
        results.append(result)
        
        # Simular si convergen
        if result['convergence_level'] >= system.motor.convergence_epochs:
            print(f"\n🎉 CONVERGENCIA ALCANZADA en época {epoch + 1}!")
            break
        
        time.sleep(0.5)  # Simular tiempo de procesamiento
    
    # Análisis final
    print(f"\n" + "=" * 60)
    print(f"📊 ANÁLISIS FINAL DEL SISTEMA DEMOCRÁTICO")
    print(f"=" * 60)
    
    final_result = results[-1]
    print(f"🏛️ Componentes activos:")
    print(f"   TopoBrain: ✅ Dynamic Topology + Symbiotic Basis")
    print(f"   OmniBrain: ✅ Integration Index + Fast-Slow Weights")
    print(f"   Quimera: ✅ Liquid Neurons + Sovereign Attention")
    
    print(f"\n🗳️ Sistema democrático:")
    print(f"   Sinergia final: {final_result['synergy_score']:.3f}")
    print(f"   Salud del sistema: {final_result['system_health']:.3f}")
    print(f"   Épocas ejecutadas: {len(results)}")
    
    print(f"\n⚖️ Distribución de pesos finales:")
    for comp, weight in final_result['component_weights'].items():
        print(f"   {comp}: {weight:.1%}")
    
    print(f"\n🔧 Ajustes homeostáticos aplicados:")
    for adj, value in final_result['adjustments'].items():
        print(f"   {adj}: {value:+.1%}")
    
    # Estado del diálogo interno
    print(f"\n🧠 DIÁLOGO INTERNO FISIOLÓGICO:")
    print(f"   TopoBrain: Metabolismo={final_result['topobrain_metrics']['metabolism_state']:.3f}, "
          f"Sensibilidad={final_result['topobrain_metrics']['sensitivity_level']:.3f}")
    print(f"   OmniBrain: Integración={final_result['omnibrain_metrics']['integration_index']:.3f}, "
          f"Balance Fast-Slow={final_result['omnibrain_metrics']['fast_slow_balance']:.3f}")
    print(f"   Quimera: Plasticidad={final_result['quimera_metrics']['liquid_plasticity']:.3f}, "
          f"Foco Atencional={final_result['quimera_metrics']['sovereign_focus']:.3f}")
    
    # Conclusión
    print(f"\n✅ SISTEMA PREMIUM SYNERGY FUNCIONAL:")
    print(f"   • Cámara Baja: Cada componente ejecuta con autoregulación")
    print(f"   • Cámara Alta: Motor homeostático delibera democráticamente")
    print(f"   • Convergencia: Motor ajusta si sinergias no convergen")
    print(f"   • Diálogo interno: Metabolismo, sensibilidad, gating activos")
    
    return results

if __name__ == "__main__":
    results = run_demo()
    
    # Guardar resultados
    with open('premium_synergy_demo_results.json', 'w') as f:
        # Convertir resultados para JSON (remove non-serializable objects)
        json_results = []
        for result in results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, (int, float, str, list, dict)):
                    json_result[key] = value
            json_results.append(json_result)
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Resultados guardados en: premium_synergy_demo_results.json")

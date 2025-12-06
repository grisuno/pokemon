import numpy as np

# Configuración de reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def mcculloch_pitts_neuron(inputs, weights, threshold):
    """
    Neurona artificial de McCulloch-Pitts (1943).
    - inputs: vector binario de entrada (0 o 1)
    - weights: vector de pesos sinápticos
    - threshold: valor umbral de activación
    Retorna 1 si la suma ponderada >= threshold; 0 en caso contrario.
    """
    weighted_sum = np.dot(inputs, weights)
    return 1 if weighted_sum >= threshold else 0

# Parámetros de la neurona
INPUT_SIZE = 2
WEIGHTS = np.array([1, 1])
THRESHOLD = 2

# Casos de prueba: compuerta AND
TEST_INPUTS = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Ejecución y evaluación
if __name__ == "__main__":
    print("Neurona de McCulloch-Pitts (1943) - Compuerta AND")
    print(f"Pesos: {WEIGHTS}")
    print(f"Umbral: {THRESHOLD}")
    print("-" * 40)
    
    for idx, x in enumerate(TEST_INPUTS):
        output = mcculloch_pitts_neuron(x, WEIGHTS, THRESHOLD)
        print(f"Entrada {idx + 1}: {x} → Salida: {output}")
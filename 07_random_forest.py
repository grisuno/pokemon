import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# Configuración reproducible
# ----------------------------
SEED = 42
np.random.seed(SEED)

# ----------------------------
# Hiperparámetros y configuración del modelo
# ----------------------------
DATASET_SIZE = 1000
NUM_FEATURES = 10
NUM_INFORMATIVE = 5
NUM_REDUNDANT = 2
NUM_CLASSES = 2
TEST_SIZE = 0.25

# Parámetros del Random Forest
N_TREES = 100
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"  # Estrategia original de Breiman: sqrt(n_features) para clasificación

# ----------------------------
# Generación de datos sintéticos
# ----------------------------
X, y = make_classification(
    n_samples=DATASET_SIZE,
    n_features=NUM_FEATURES,
    n_informative=NUM_INFORMATIVE,
    n_redundant=NUM_REDUNDANT,
    n_classes=NUM_CLASSES,
    n_clusters_per_class=1,
    random_state=SEED,
    flip_y=0.05  # Pequeño ruido en etiquetas
)

# División en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

train_size = X_train.shape[0]
val_size = X_val.shape[0]

# ----------------------------
# Entrenamiento del Random Forest
# ----------------------------
print(f"Entrenando Random Forest (Breiman, 2001)")
print(f"Dataset: clasificación sintética con ruido (no lineal)")
print(f"Tamaño total: {DATASET_SIZE} | Entrenamiento: {train_size} | Validación: {val_size}")
print(f"Características: {NUM_FEATURES} (informativas: {NUM_INFORMATIVE}, redundantes: {NUM_REDUNDANT})")
print("-" * 75)
print(f"Parámetros del modelo:")
print(f"  → Número de árboles: {N_TREES}")
print(f"  → Profundidad máxima: {MAX_DEPTH}")
print(f"  → Muestras mínimas por hoja: {MIN_SAMPLES_LEAF}")
print(f"  → Características por split: {MAX_FEATURES} (≈{int(np.sqrt(NUM_FEATURES))} en este caso)")
print("-" * 75)

rf_model = RandomForestClassifier(
    n_estimators=N_TREES,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features=MAX_FEATURES,
    random_state=SEED,
    n_jobs=1  # Asegura reproducibilidad en CPU
)
rf_model.fit(X_train, y_train)

# ----------------------------
# Evaluación
# ----------------------------
y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

# ----------------------------
# Impresión final de resultados
# ----------------------------
print(f"Resultados del Random Forest:")
print(f"  → Precisión en entrenamiento: {train_acc:.4f} ({int(train_acc * train_size)}/{train_size} correctos)")
print(f"  → Precisión en validación:   {val_acc:.4f} ({int(val_acc * val_size)}/{val_size} correctos)")
print("-" * 75)
print("El Random Forest reduce el sobreajuste mediante ensamblaje de árboles")
print("entrenados en subconjuntos aleatorios de datos y características.")
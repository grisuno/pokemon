import numpy as np
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# Configuración reproducible
# ----------------------------
SEED = 42
np.random.seed(SEED)

# ----------------------------
# Hiperparámetros y configuración
# ----------------------------
DATASET_SIZE = 400
TEST_SIZE = 0.25
KERNEL_TYPE = 'rbf'
C_REGULARIZATION = 1.0
GAMMA_KERNEL = 'scale'  # Usa 1 / (n_features * X.var()) como en paper original

# ----------------------------
# Generación de datos: círculos concéntricos (no linealmente separables)
# ----------------------------
X, y = make_circles(n_samples=DATASET_SIZE, noise=0.1, factor=0.2, random_state=SEED)

# División en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

# Tamaños reales
train_size = X_train.shape[0]
val_size = X_val.shape[0]

# ----------------------------
# Entrenamiento del SVM con kernel RBF
# ----------------------------
print(f"Entrenando SVM con kernel RBF (Cortes & Vapnik, 1995)")
print(f"Dataset: círculos concéntricos (no linealmente separables)")
print(f"Tamaño total: {DATASET_SIZE} | Entrenamiento: {train_size} | Validación: {val_size}")
print(f"Kernel: {KERNEL_TYPE} | C (regularización): {C_REGULARIZATION} | Gamma: {GAMMA_KERNEL}")
print("-" * 70)

svm_model = SVC(kernel=KERNEL_TYPE, C=C_REGULARIZATION, gamma=GAMMA_KERNEL, random_state=SEED)
svm_model.fit(X_train, y_train)

# ----------------------------
# Evaluación
# ----------------------------
y_train_pred = svm_model.predict(X_train)
y_val_pred = svm_model.predict(X_val)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

# ----------------------------
# Impresión de resultados
# ----------------------------
print(f"Resultados del SVM con kernel RBF:")
print(f"  → Precisión en entrenamiento: {train_acc:.4f} ({int(train_acc * train_size)}/{train_size} correctos)")
print(f"  → Precisión en validación:   {val_acc:.4f} ({int(val_acc * val_size)}/{val_size} correctos)")
print("-" * 70)
print("Este resultado demuestra la capacidad del kernel RBF para mapear")
print("datos no linealmente separables a un espacio de alta dimensión donde sí lo son.")

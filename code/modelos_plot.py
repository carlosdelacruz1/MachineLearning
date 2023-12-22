import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Crear un conjunto de datos de ejemplo con valores aleatorios
np.random.seed(0)
num_samples = 100
X = np.random.rand(num_samples, 4)  # Variables independientes aleatorias
y = 3 * X[:, 0] + 2 * X[:, 1] - 1 * X[:, 2] + 0.5 * X[:, 3] + np.random.randn(num_samples) * 0.5  # Variable dependiente con componente aleatoria

# Crear y entrenar un modelo de regresión lineal múltiple
modelo_regresion = LinearRegression()
modelo_regresion.fit(X, y)

# Predicciones del modelo
y_pred = modelo_regresion.predict(X)

# Visualizar el ajuste
plt.figure(figsize=(12, 6))

# Gráficos de dispersión entre variables independientes y variable dependiente
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.scatter(X[:, i], y, label=f'Variable {i+1}')
    plt.xlabel(f'Variable {i+1}')
    plt.ylabel('Variable Dependiente')
    plt.legend()

# Gráficos de dispersión entre predicciones y valores reales
for i in range(4):
    plt.subplot(2, 4, i + 5)
    plt.scatter(y_pred, y)
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')

plt.tight_layout()
plt.show()


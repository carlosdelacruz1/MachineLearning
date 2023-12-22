import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

# Calcular el coeficiente de determinación (R^2)
r2 = r2_score(y, y_pred)
print("Coeficiente de determinación (R^2):", r2)

# Visualizar el ajuste con la línea de regresión
plt.figure(figsize=(10, 6))

# Gráficos de dispersión entre predicciones y valores reales
plt.scatter(y_pred, y, label='Datos de entrenamiento')
plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], color='red', linestyle='--', label='Línea de ajuste')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.title(f'Ajuste del modelo (R^2 = {r2:.2f})')
plt.legend()

plt.show()


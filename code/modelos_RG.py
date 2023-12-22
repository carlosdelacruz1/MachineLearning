import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Crear datos de ejemplo
np.random.seed(0)
X = np.random.rand(100, 1) * 4  # Variable independiente aleatoria
y = 2 * X**3 + 5 * X**2 + 1 * X + np.random.randn(100, 1) * 2  # Variable dependiente con componente no lineal

# Crear características polinómicas de grado 3
grado_polinomio = 3
poly_features = PolynomialFeatures(degree=grado_polinomio, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Crear y entrenar un modelo de regresión lineal
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_poly, y)

# Predecir valores con el modelo
X_new = np.linspace(0, 4, 100).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)
y_pred = modelo_regresion.predict(X_new_poly)

# Calcular el error cuadrático medio
mse = mean_squared_error(y, modelo_regresion.predict(X_poly))
print("Error cuadrático medio:", mse)

# Visualizar los resultados
plt.scatter(X, y, label="Datos de entrenamiento")
plt.plot(X_new, y_pred, color='red', label="Predicción")
plt.xlabel("Variable independiente")
plt.ylabel("Variable dependiente")
plt.legend()
plt.show()


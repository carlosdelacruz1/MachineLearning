import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el CSV
df = pd.read_csv('datos.csv')

# Definir las características (X) y la variable objetivo (y)
X = df.drop(columns=['PotencialRedox'])
y = df['PotencialRedox']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Red Neuronal con Keras (TensorFlow backend)
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000)
model.fit(X_train, y_train)
y_pred_nn = model.predict(X_test)

# Evaluar el rendimiento de los modelos
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print('Gradient Boosting Regressor:')
print(f'MSE: {mse_gb}')
print(f'R^2 score: {r2_gb}')
print('\nNeural Network:')
print(f'MSE: {mse_nn}')
print(f'R^2 score: {r2_nn}')

# Visualizar resultados de los modelos
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_gb)
plt.xlabel('True Potencial Redox')
plt.ylabel('Predicted Potencial Redox')
plt.title('Gradient Boosting Regressor')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_nn)
plt.xlabel('True Potencial Redox')
plt.ylabel('Predicted Potencial Redox')
plt.title('Neural Network')

plt.tight_layout()
plt.show()


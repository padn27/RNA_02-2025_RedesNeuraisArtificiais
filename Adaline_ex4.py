# Adaline - solução direta (California Housing)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # <-- import das métricas

# 1. Carregar o dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data.values      # features
y = housing.target.values    # preços das casas

# 2. Normalizar os dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).flatten()

# 3. Adicionar coluna de 1s para intercepto
X_aug = np.hstack([X_scaled, np.ones((X_scaled.shape[0],1))])

# 4. Solução direta usando pseudoinversa
w = np.linalg.pinv(X_aug) @ y_scaled

# 5. Dividir em treino e teste para avaliação visual
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train_aug = np.hstack([X_train, np.ones((X_train.shape[0],1))])
X_test_aug  = np.hstack([X_test, np.ones((X_test.shape[0],1))])

# 6. Previsão
y_train_pred = X_train_aug @ w
y_test_pred  = X_test_aug  @ w

# 7. Cálculo de métricas
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train  = r2_score(y_train, y_train_pred)
mse_test  = mean_squared_error(y_test, y_test_pred)
r2_test   = r2_score(y_test, y_test_pred)

print("Treino - MSE: {:.4f}, R²: {:.4f}".format(mse_train, r2_train))
print("Teste  - MSE: {:.4f}, R²: {:.4f}".format(mse_test, r2_test))

# 8. Plot resultados (treino)
plt.figure(figsize=(8,4))
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue')
plt.plot([-3,3], [-3,3], 'r--', linewidth=2)  # reta y=x
plt.title("Treino: Real vs Previsto (California Housing)")
plt.xlabel("Valor real (normalizado)")
plt.ylabel("Valor previsto (normalizado)")
plt.grid(True)
plt.show()

# 9. Plot resultados (teste)
plt.figure(figsize=(8,4))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='green')
plt.plot([-3,3], [-3,3], 'r--', linewidth=2)
plt.title("Teste: Real vs Previsto (California Housing)")
plt.xlabel("Valor real (normalizado)")
plt.ylabel("Valor previsto (normalizado)")
plt.grid(True)
plt.show()

# 10. Coeficientes do modelo
print("Coeficientes (incluindo intercepto):")
print(w)

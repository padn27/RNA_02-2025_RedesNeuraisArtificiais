import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

# Dados reais extraídos (20 pontos)
t_train = np.array([0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,
                    3.3,3.6,3.9,4.2,4.5,4.8,5.1,5.4,5.7,6.0])
x_train = np.array([0.29552020666134,0.564642473395035,0.783326909627483,0.932039085967226,
                    0.997494986604054,0.973847630878195,0.863209366648874,0.675463180551151,
                    0.42737988023383,0.141120008059868,-0.157745694143248,-0.442520443294852,
                    -0.687766159183973,-0.871575772413588,-0.977530117665097,-0.996164608835841,
                    -0.925814682327732,-0.772764487555988,-0.550685542597638,-0.279415498198926])
y_train = np.array([0.588656061998402,0.669392742018511,0.734998072888245,0.779611725790168,
                    0.799248495981216,0.792154289263459,0.758962809994662,0.702638954165345,
                    0.628213964070149,0.54233600241796,0.452676291757026,0.367243867011544,
                    0.293670152244808,0.238527268275924,0.206740964700471,0.201150617349248,
                    0.22225559530168,0.268170653733204,0.334794337220709,0.416175350540322])

# Modelo Adaline
adaline = SGDRegressor(loss="squared_error", penalty=None, learning_rate="constant",
                       eta0=0.01, max_iter=10000, tol=1e-6, random_state=42)

adaline.fit(x_train.reshape(-1, 1), y_train)

# Dados de Teste (mais fino, interpolando no tempo)
t_test = np.linspace(0, 6, 100)
x_test = np.sin(t_test)   # nova entrada senoidal (mais densa)
y_test = adaline.predict(x_test.reshape(-1,1))

# Gráfico 1 - Treinamento
plt.figure(figsize=(7,4))
plt.plot(t_train, x_train, "bo-", mfc="white", label="Entrada (x)")
plt.plot(t_train, y_train, "ro-", label="Saída (y)")
plt.xlabel("t")
plt.legend()
plt.title("Dados de Treinamento")
plt.grid(True)
plt.show()

# Gráfico 2 - Previsão
plt.figure(figsize=(7,4))
plt.plot(t_test, x_test, "b--", label="Entrada (nova senoide)")
plt.plot(t_test, y_test, "g-", linewidth=2, label="Saída prevista (Adaline)")
plt.scatter(t_train, y_train, c="red", marker="o", label="Dados reais (treino)")
plt.xlabel("t")
plt.legend()
plt.title("Generalização do Modelo")
plt.grid(True)
plt.show()

# Parâmetros do modelo 
print("Coeficientes do modelo:")
print("a (intercepto) =", adaline.intercept_[0])
print("b (peso)       =", adaline.coef_[0])

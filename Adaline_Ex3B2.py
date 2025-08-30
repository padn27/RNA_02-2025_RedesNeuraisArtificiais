import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

# Dados de entrada (V1, V2, V3)
X = np.array([
    [0.425950531568867, 1.31094093866055, -1.60579308398418],
    [0.81020605733591, 1.11515296917338, -1.43676223303848],  # ponto 2 -> NÃO USAR
    [1.11515296917338, 0.81020605733591, -1.26773138209277],
    [1.31094093866055, 0.425950531568867, -1.09870053114707],  # ponto 4 -> NÃO USAR
    [1.37840487520902, 1.38756813824575e-16, -0.929669680201368],
    [1.31094093866055, -0.425950531568866, -0.760638829255665],
    [1.11515296917338, -0.81020605733591, -0.591607978309962],
    [0.81020605733591, -1.11515296917338, -0.422577127364258],
    [0.425950531568867, -1.31094093866055, -0.253546276418555],
    [1.7725769469902e-16, -1.37840487520902, -0.0845154254728517],  # ponto 10 -> NÃO USAR
    [-0.425950531568866, -1.31094093866055, 0.0845154254728517],
    [-0.81020605733591, -1.11515296917338, 0.253546276418555],
    [-1.11515296917338, -0.81020605733591, 0.422577127364258],
    [-1.31094093866055, -0.425950531568867, 0.591607978309962],       # ponto 14 -> NÃO USAR
    [-1.37840487520902, -1.98855009846192e-16, 0.760638829255665],
    [-1.31094093866055, 0.425950531568866, 0.929669680201368],
    [-1.11515296917338, 0.81020605733591, 1.09870053114707],
    [-0.81020605733591, 1.11515296917338, 1.26773138209277],          # ponto 18 -> NÃO USAR
    [-0.425950531568867, 1.31094093866055, 1.43676223303848],         # ponto 19 -> NÃO USAR
    [-3.2916004080713e-16, 1.37840487520902, 1.60579308398418]
])

# Saída correspondente (Y)
y = np.array([
    -0.198750516267684,
    0.301021623362128,   # ponto 2 -> NÃO USAR
    0.503167264361769,
    0.437536735151964,   # ponto 4 -> NÃO USAR
    0.160192161399815,
    -0.252080285449282,
    -0.70928675363343,
    -1.11703493630872,
    -1.385773848213,
    -1.4395597000417,    # ponto 10 -> NÃO USAR
    -1.22348980567651,
    -0.709076839632104,
    0.102962625042473,
    1.1827782599265,     # ponto 14 -> NÃO USAR
    2.47430793935287,
    3.90076549187618,
    5.37215706573455,
    6.79409035408407,    # ponto 18 -> NÃO USAR
    8.07701437166256,    # ponto 19 -> NÃO USAR
    9.14498532916549
])

# Índices dos pontos usados (tirando 2,4,10,14,18,19)
indices_treinamento = [0,2,4,5,6,7,8,10,11,12,14,15,16,19]

X_train = X[indices_treinamento]
y_train = y[indices_treinamento]

# Criar e treinar o modelo Adaline
model = SGDRegressor(max_iter=2000, learning_rate='invscaling', eta0=0.01, tol=1e-6, random_state=42)
model.fit(X_train, y_train)

# Previsão com os pontos de treinamento
y_pred = model.predict(X_train)

# Plotar comparação
plt.figure(figsize=(8,5))
plt.plot(y_train, 'o', label='Original (treino)')
plt.plot(y_pred, '-', label='Previsto (modelo)')
plt.xlabel('Amostras')
plt.ylabel('Saída y')
plt.title('Adaline Multivariado - Somente pontos de treinamento')
plt.legend()
plt.show()

# Exibir coeficientes
print(f'Coeficientes (b, c, d): {model.coef_}')
print(f'Termo independente (a): {model.intercept_}')

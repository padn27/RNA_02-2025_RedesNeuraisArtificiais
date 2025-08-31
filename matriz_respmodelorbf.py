import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import statsmodels.api as sm

# -----------------------------
# Funções RBF
# -----------------------------
def treinaRBF(Xin, Yin, p, r=1.0):
    N, n = Xin.shape
    kmeans = KMeans(n_clusters=p, random_state=42).fit(Xin)
    m = kmeans.cluster_centers_
    covi = (1/r) * np.eye(n)

    def radialnvar(x, m, invK):
        delta = x - m
        return np.exp(-0.5 * delta.T @ invK @ delta)

    H = np.zeros((N, p))
    for j in range(N):
        for i in range(p):
            H[j, i] = radialnvar(Xin[j], m[i], covi)

    Haug = np.hstack((np.ones((N,1)), H))
    W = pinv(Haug) @ Yin
    return {'m': m, 'covi': covi, 'r': r, 'W': W, 'H': H}

def YRBF(Xin, modRBF):
    m = modRBF['m']
    covi = modRBF['covi']
    W = modRBF['W']
    N, n = Xin.shape
    p = m.shape[0]

    def radialnvar(x, m, invK):
        delta = x - m
        return np.exp(-0.5 * delta.T @ invK @ delta)

    H = np.zeros((N, p))
    for j in range(N):
        for i in range(p):
            H[j, i] = radialnvar(Xin[j], m[i], covi)

    Haug = np.hstack((np.ones((N,1)), H))
    Yhat = Haug @ W
    return Yhat

# -----------------------------
# Carregar e preparar dados
# -----------------------------
data = sm.datasets.get_rdataset("airquality").data
air = data.dropna()  # remove valores ausentes

# Selecionar variáveis independentes (Solar.R, Wind, Temp) e dependente (Ozone)
X = air[['Solar.R','Wind','Temp']].values
y = air['Ozone'].values.reshape(-1,1)

# Normalizar
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# -----------------------------
# Separar treino/teste (20% teste)
# -----------------------------
xtr, xtst, ytr, ytst = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Plotar matriz de dispersão
# -----------------------------
xytr_df = pd.DataFrame(xtr, columns=['Solar.R','Wind','Temp'])
xytr_df['Ozone'] = ytr
pd.plotting.scatter_matrix(xytr_df, alpha=0.8, figsize=(8,6), diagonal='kde')
plt.suptitle("Matriz de dispersão das variáveis (Treino)")
plt.show()

# -----------------------------
# Treinar RBF com p=3, r=5
# -----------------------------
modRBF = treinaRBF(xtr, ytr, p=3, r=5)

# -----------------------------
# Previsão no conjunto de teste
# -----------------------------
Yhat_tst = YRBF(xtst, modRBF)

# -----------------------------
# Plotar resposta do modelo vs real
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(range(len(ytst)), ytst, 'r-o', label='ytst (real)')
plt.plot(range(len(Yhat_tst)), Yhat_tst, 'b-s', label='Yhat_tst (RBF)')
plt.xlabel("Amostra")
plt.ylabel("Ozone (normalizado)")
plt.title("Resposta do modelo RBF vs Valores reais")
plt.legend()
plt.show()

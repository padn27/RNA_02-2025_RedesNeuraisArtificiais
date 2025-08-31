import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from sklearn.cluster import KMeans

# -----------------------------
# Função de treinamento RBF
# -----------------------------
def treinaRBF(Xin, Yin, p, r=1.0):
    N, n = Xin.shape

    # K-means para escolher centros
    kmeans = KMeans(n_clusters=p, random_state=42).fit(Xin)
    m = kmeans.cluster_centers_          # centros
    covi = (1/r) * np.eye(n)            # covariância uniforme

    def radialnvar(x, m, invK):
        delta = x - m
        return np.exp(-0.5 * delta.T @ invK @ delta)

    # Matriz de projeção H
    H = np.zeros((N, p))
    for j in range(N):
        for i in range(p):
            H[j, i] = radialnvar(Xin[j], m[i], covi)

    Haug = np.hstack((np.ones((N,1)), H))
    W = pinv(Haug) @ Yin

    return {'m': m, 'covi': covi, 'r': r, 'W': W, 'H': H}

# -----------------------------
# Função de previsão RBF
# -----------------------------
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
# Exemplo de uso com plotagem
# -----------------------------
if __name__ == "__main__":
    # Criar amostras 2D (mesmo do ELM)
    N = 30
    m1, m2 = np.array([2,2]), np.array([4,4])
    g1 = np.random.randn(N,2)*0.6 + m1
    g2 = np.random.randn(N,2)*0.6 + m2
    xc1 = np.vstack((g1,g2))

    m3, m4 = np.array([2,4]), np.array([4,2])
    g3 = np.random.randn(N,2)*0.6 + m3
    g4 = np.random.randn(N,2)*0.6 + m4
    xc2 = np.vstack((g3,g4))

    X = np.vstack((xc1, xc2))
    y = np.vstack((-np.ones((2*N,1)), np.ones((2*N,1))))

    # Treinar RBF
    p = 20
    r = 1.0
    modRBF = treinaRBF(X, y, p, r)

    # Plotar superfície de decisão
    x_min, x_max = 0, 6
    y_min, y_max = 0, 6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Zgrid = YRBF(grid, modRBF)
    Zgrid = np.sign(Zgrid).reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Zgrid, levels=[-1,0,1], alpha=0.3, colors=['blue','red'])
    plt.scatter(xc1[:,0], xc1[:,1], color='red', label='Classe -1')
    plt.scatter(xc2[:,0], xc2[:,1], color='blue', label='Classe 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('RBF - Superfície de decisão')
    plt.legend()
    plt.show()


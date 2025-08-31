import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv

# -----------------------------
# Função de treinamento ELM
# -----------------------------
def treinaELM(Xin, Yin, p, par=1):
    n = Xin.shape[1]
    if par == 1:
        Xin_aug = np.hstack((np.ones((Xin.shape[0], 1)), Xin))
        Z = np.random.uniform(-0.5, 0.5, size=(n+1, p))
    else:
        Xin_aug = Xin
        Z = np.random.uniform(-0.5, 0.5, size=(n, p))

    H = np.tanh(Xin_aug @ Z)
    Haug = np.hstack((np.ones((H.shape[0], 1)), H))
    W = pinv(Haug) @ Yin
    return W, H, Z

# -----------------------------
# Função de previsão ELM
# -----------------------------
def YELM(Xin, Z, W, par=1):
    if par == 1:
        Xin_aug = np.hstack((np.ones((Xin.shape[0], 1)), Xin))
    else:
        Xin_aug = Xin

    H = np.tanh(Xin_aug @ Z)
    Haug = np.hstack((np.ones((H.shape[0], 1)), H))
    Yhat = np.sign(Haug @ W)
    return Yhat

# -----------------------------
# Gerar amostras 2D
# -----------------------------
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

# -----------------------------
# Treinar ELM
# -----------------------------
p = 50
W, H, Z = treinaELM(X, y, p, par=1)

# -----------------------------
# Plotar superfícies de decisão
# -----------------------------
x_min, x_max = 0, 6
y_min, y_max = 0, 6
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Zgrid = YELM(grid, Z, W, par=1)
Zgrid = Zgrid.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Zgrid, levels=[-1,0,1], alpha=0.3, colors=['blue','red'])
plt.scatter(xc1[:,0], xc1[:,1], color='red', label='Classe -1')
plt.scatter(xc2[:,0], xc2[:,1], color='blue', label='Classe 1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('ELM - Superfície de decisão')
plt.legend()
plt.show()

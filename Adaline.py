import numpy as np
import matplotlib.pyplot as plt

# --- Função do Adaline ---
def train_adaline(xin, yd, eta, tol, max_epocas, par):
    N, n = xin.shape
    if par == 1:
        wt = np.random.rand(n + 1, 1) - 0.5
        xin = np.hstack((-np.ones((N, 1)), xin))
    else:
        wt = np.random.rand(n, 1) - 0.5
    
    nepocas = 0
    eepoca = tol + 1
    evec = np.zeros(max_epocas)
    
    while nepocas < max_epocas and eepoca > tol:
        xseq = np.random.permutation(N)
        ei2 = 0
        for irand in xseq:
            x_i = xin[irand, :].reshape(-1, 1)
            yhat = x_i.T @ wt  # saída linear
            ei = yd[irand] - yhat
            dw = eta * ei * x_i
            wt = wt + dw
            ei2 += ei**2
        eepoca = ei2
        evec[nepocas] = eepoca
        nepocas += 1
    
    return wt, evec[:nepocas]

# --- Geração dos dados ---
N = 30
xc1 = np.random.randn(N, 2) * 0.5 + 2
xc2 = np.random.randn(N, 2) * 0.5 + 4

X = np.vstack((xc1, xc2))
Y = np.array([0]*N + [1]*N)

# --- Treinamento ---
wt, evec = train_adaline(X, Y, eta=0.01, tol=0.01, max_epocas=200, par=1)

# --- Plot dos pontos ---
plt.scatter(xc1[:,0], xc1[:,1], color='red')
plt.scatter(xc2[:,0], xc2[:,1], color='blue')
plt.xlabel('x1')
plt.ylabel('x2')

# --- Plot da superfície de decisão ---
xx, yy = np.meshgrid(np.linspace(0,6,100), np.linspace(0,6,100))
grid = np.c_[xx.ravel(), yy.ravel()]
if wt.shape[0] == 3:  # com bias
    grid_aug = np.hstack((-np.ones((grid.shape[0],1)), grid))
else:
    grid_aug = grid
Z = (grid_aug @ wt >= 0.5).astype(int).reshape(xx.shape)  # limiar para classificação
plt.contour(xx, yy, Z, levels=[0.5], colors='black')
plt.show()

# --- Plot do erro por época ---
plt.figure()
plt.plot(evec, 'b-')
plt.xlabel('Época')
plt.ylabel('Erro total (MSE)')
plt.title('Convergência do Adaline')
plt.show()

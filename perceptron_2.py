from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregar dados
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Normalizar X
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 3. PCA para 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 4. Treinar Perceptron simples para visualização
w = np.zeros(X_pca.shape[1]+1)
eta = 0.01
n_epochs = 50

def predict(x, w):
    return 1 if np.dot(x, w[1:]) + w[0] > 0 else 0

for _ in range(n_epochs):
    for xi, target in zip(X_pca, y):
        update = eta * (target - predict(xi, w))
        w[1:] += update * xi
        w[0] += update

# 5. Criar grid para fronteira de decisão
x_min, x_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
y_min, y_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = np.array([predict([xi, yi], w) for xi, yi in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)

# 6. Plot
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Fronteira de decisão do Perceptron (2D)")
plt.show()

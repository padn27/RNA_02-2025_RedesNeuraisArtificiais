# Perceptron - Classificação do Câncer de Mama + Tabela PDF + Gráfico
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

# -------------------------------
# 1. Carregar e preparar dados
# -------------------------------
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = maligno, 1 = benigno

# Normalização simples
X = (X - X.mean(axis=0)) / X.std(axis=0)

# -------------------------------
# 2. Funções do Perceptron
# -------------------------------
eta = 0.01      # taxa de aprendizado
n_epochs = 50   # número de épocas

def predict(x, w):
    return 1 if np.dot(x, w[1:]) + w[0] > 0 else 0

def perceptron_train(X_train, y_train, eta, n_epochs):
    w = np.zeros(X_train.shape[1] + 1)  # pesos + bias
    for _ in range(n_epochs):
        for xi, target in zip(X_train, y_train):
            update = eta * (target - predict(xi, w))
            w[1:] += update * xi
            w[0] += update
    return w

# -------------------------------
# 3. Validação cruzada 10-fold
# -------------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    w = perceptron_train(X_train, y_train, eta, n_epochs)
    y_pred = np.array([predict(xi, w) for xi in X_test])
    acc = np.mean(y_pred == y_test)
    accuracies.append(acc)

accuracies = np.array(accuracies)
media = accuracies.mean()
desvio = accuracies.std()

print("Acurácias por fold:", accuracies)
print(f"Acurácia média: {media:.4f}, Desvio padrão: {desvio:.4f}")

# -------------------------------
# 4. Criar tabela e salvar PDF
# -------------------------------
folds = list(range(1, 11))
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('tight')
ax.axis('off')

tabela_dados = [["Fold", "Acurácia"]] + [[f, f"{a*100:.2f}%"] for f, a in zip(folds, accuracies)]
ax.table(cellText=tabela_dados, loc='center', cellLoc='center')

plt.savefig("acuracias_perceptron.pdf", bbox_inches='tight')
plt.show()

# Salvar média e desvio em TXT
with open("acuracias_perceptron.txt", "w") as f:
    f.write(f"Acurácia média: {media*100:.2f}%\n")
    f.write(f"Desvio padrão: {desvio*100:.2f}%\n")

# -------------------------------
# 5. Gráfico de barras - acurácia por fold
# -------------------------------
plt.figure(figsize=(8,4))
plt.bar(folds, accuracies, color='skyblue', alpha=0.7)
plt.axhline(y=media, color='red', linestyle='--', label='Média')
plt.xticks(folds)
plt.ylim(0.9, 1.05)
plt.xlabel("Fold")
plt.ylabel("Acurácia")
plt.title("Acurácia por fold - Perceptron (10-fold)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("grafico_acuracias_perceptron.pdf", bbox_inches='tight')
plt.show()

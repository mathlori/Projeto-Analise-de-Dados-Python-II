from funcoes import *
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statistics as st

print("=" * 20)
print("Avaliação Continuada 2")
print("=" * 20)

# Variáveis
y = [53, 96, 78, 82, 52, 34, 54, 26, 67, 13]
x = [32, 36, 65, 24, 76, 14, 54, 61, 98, 10]
n = 10

# Ajustando os dados para os pacotes

y2 = np.array(y)
x2 = np.array(x)

# Ajustando a estrutura

x2 = x2.reshape(-1, 1) # Transformando os dados de x em uma matriz

# Média programada pelo estudante

meanY = sum(y) / n
meanX = sum(x) / n

# Média pela função da biblioteca Statistics

meanYP = st.mean(y)
meanXP = st.mean(x)

categoria(f"""Dados:
X = {x}
Y = {y}
""")

categoria(f"""Médias:

> Por funções programadas:

X: {meanX}
Y: {meanY}

> Com funções do Python:

X: {meanXP}
Y: {meanYP}
""")

# Encontrando os parâmetros
# Por funções do estudante:
# Somas de X e Y

SSX = 0
SSY = 0
SSXY = 0

for i in range(0, len(x)):
    SSX += (x[i] - meanX) ** 2
    SSY += (y[i] - meanY) ** 2
    SSXY += (x[i] - meanX) * (y[i] - meanY)

# Por funções do Python
SSXp = np.sum((x2 - meanXP) ** 2)
SSYp = np.sum((y2 - meanYP) ** 2)

SSXYp = []

for i in range(0, len(x)):
  SSXYp.append((x[i] - meanXP) * (y[i] - meanYP))

SSXYp = np.array(SSXYp)
SSXYp = np.sum(SSXYp)

categoria(f"""Somas:

> Por funções programadas:

SSX: {SSX}
SSY: {SSY}
SSXY: {SSXY}

> Com funções do Python:

SSX: {SSXp}
SSY: {SSYp}
SSXY: {SSXYp}
""")

# Coeficientes 
# Por funções do Estudante

b1 = SSXY / SSX
b0 = meanY - b1 * meanX

# Por funções do Python
# Inicializando o modelo:

modelo = LinearRegression()

# Treinamento do modelo com os dados

modelo.fit(x2, y2)

b0P = modelo.intercept_
b1P = modelo. coef_[0]

categoria(f"""Coeficientes:

> Por funções programadas:

Coeficiente Angular: {b1: .2f}
Coeficiente Linear: {b0: .2f}

> Com funções do Python:

Coeficiente Angular: {b1P: .2f}
Coeficiente Linear: {b0P: .2f}
""")

# Ajustando os valores
xReta = list()
yReta = list()

for k in range(n):
    xReta.append(x[k])
    yReta.append(b0 + b1 * x[k])

plt.figure(figsize=(10, 5))

plt.plot(x, y, 'r+', label="Dados Levantados")
plt.plot(xReta, yReta, 'b', label="Adaptação da Reta")
plt.xlabel("Eixo x")
plt.ylabel("Eixo y")
plt.title("Regressão Linear de base de dados criada")
plt.legend()
plt.show()
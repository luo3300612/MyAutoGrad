from autograd.autograd.DataStructure import Mat
from autograd.autograd.op import op
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

x = np.random.uniform(0, 5, (100,))
y = np.random.uniform(0, 5, (100,))
label = np.zeros((100,))

for i, cord in enumerate(zip(x, y)):
    _x, _y = cord
    if _x - _y > 0:
        label[i] = 1
    else:
        label[i] = 0
    if np.random.uniform(0, 1) > 0.90:
        label[i] = 1 - label[i]

plt.scatter(x[label == 1], y[label == 1])
plt.scatter(x[label == 0], y[label == 0])

alpha = 0.01
max_iteration = 100

X = np.vstack((np.ones(100, ), x))
X = np.vstack((X, y))
mat_X = Mat(X).T()
mat_label = Mat([label]).T()
weight = Mat.zeros(3, 1)
lam = 0.01

now = datetime.now()
for epoch in range(max_iteration):
    weight.zero_grad()
    loss = - mat_label.T() * op.log(1 / (1 + op.exp(-mat_X * weight))) - (1 - mat_label).T() * op.log(1 - 1 / (1 + op.exp(-mat_X * weight))) + lam * weight.T() * weight

    weight = weight - alpha * loss.grad(weight)

    if epoch % 10 == 0:
        print(f"epoch:{epoch},loss:{loss}")
end = datetime.now()

print((end-now))

x = np.linspace(0, 5, 50)
y = list(map(lambda x: (-weight[0][0].value - weight[1][0].value * x) / weight[2][0].value, x))
plt.plot(x, y)
plt.show()

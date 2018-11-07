from autograd.DataStructure import Mat
from autograd.op import op
import numpy as np
import matplotlib.pyplot as plt

# gen dataset, 4 classes
xs = np.random.uniform(0, 8, (100,))
ys = np.random.uniform(0, 8, (100,))

labels = np.zeros((100,))

for index, (x, y) in enumerate(zip(xs, ys)):
    if x - y > 0:
        if x + y - 8 > 0:
            labels[index] = 0
        else:
            labels[index] = 1
    else:
        if x + y - 8 > 0:
            labels[index] = 2
        else:
            labels[index] = 3

for i in range(4):
    plt.scatter(xs[labels == i], ys[labels == i])

plt.show()

ones = np.ones((100,))
X = np.vstack((ones, xs))
X = np.vstack((X, ys))
X = Mat(X).T

W = Mat.zeros(4,3)

alpha = 0.01
max_iteration = 100
loss_history = np.zeros((max_iteration,))
for epoch in range(max_iteration):
    for i in range(X.m):
        x = Mat([X.values[i]]).T # TODO  awful feature
        y = W * x

        loss = op.log(sum([op.exp(y[j][0]) for j in range(4)])) - y[int(labels[i])][0]
        LOSS = Mat.gen_ret(1,1,mat=[[loss]]) # TODO awful feature

        loss_history[epoch] += loss.value
        W = W - alpha * LOSS.grad(W)
        LOSS.zero_grad()
    print(f"epoch:{epoch},loss:{loss_history[epoch]}")

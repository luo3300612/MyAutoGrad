from autograd.DataStructure import Mat
import matplotlib.pyplot as plt
import numpy as np


# prepare data
ones = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_err = list(map(lambda a: a + 2 * np.random.random(), y))
label = Mat([y_err]).T

print(y)

X = Mat([ones, x]).T

# initiate
weight = Mat.zeros(2, 1)
max_iteration = 200
alpha = 0.0001

# start to train
for epoch in range(max_iteration):
    pred = X * weight
    loss = (pred - label).T * (pred - label)
    weight = weight - alpha * loss.grad(weight)
    loss.zero_grad()

    if epoch % 20 == 0:
        print(f"epoch:{epoch},loss:{loss}")

plt.scatter(x, y_err)
plt.plot(x, pred.T.values[0])
plt.show()

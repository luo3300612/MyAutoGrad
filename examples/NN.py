from myautograd.DataStructure import Mat
from myautograd.op import op
import numpy as np

train = Mat([[0, 0], [1, 0], [0, 1], [1, 1]])
label = Mat([[0], [1], [1], [0]])

epsilon = 0.12
W1 = 2 * epsilon * (np.random.rand(2, 2) - 0.5)
W1 = Mat(W1)
b1 = Mat.zeros(2, 1)
W2 = 2 * epsilon * (np.random.rand(2, 2) - 0.5)
W2 = Mat(W2)
b2 = Mat.zeros(2, 1)
parameters = [W1, b1, W2, b2]


def forward(x, W1, b1, W2, b2):
    h = W1 * x
    h = sigmoid(h + b1)
    o = W2 * h
    o = sigmoid(o + b2)
    return o


def sigmoid(eta):
    return 1 / (1 + op.exp(-eta))


alpha = 0.01
max_iteration = 10

loss_history = Mat.zeros(1, max_iteration)

for epoch in range(max_iteration):
    acc = 0
    for i in range(train.m):  # for each sample
        # zero gradient

        for i in range(len(parameters)):
            parameters[i].zero_grad()

        # forward
        x = Mat([train.values[i]]).T
        o = forward(x, *parameters)
        loss = op.norm_square(o - label[i][0])

        # record loss
        loss_history[0][epoch] = loss_history[0][epoch] + loss[0][0]

        # back propagation
        for i in range(4):
            parameters[i] = parameters[i] - alpha * loss.grad(parameters[i])

        loss.zero_grad()
        if o[0][0] > 0.5 and label[i][0] == 1: # TODO make it easier to use
            acc += 1
        elif o[0][0] < 0.5 and label[i][0] == 0:
            acc += 1

    if epoch % 1 == 0:
        print(f"epoch:{epoch},loss:{loss},acc:{acc/train.m*100}%")

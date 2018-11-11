import autograd.numpy as np
from autograd import grad


# test1
# def taylor_sine(x):
#     ans = currterm = x
#     i = 0
#
#     while np.abs(currterm) > 0.001:
#         currterm = - currterm * x ** 2 / ((2 * i + 3) * (2 * i + 2))
#         ans = ans + currterm
#         i += 1
#     return ans
#
# grad_sine = grad(taylor_sine)
# print("gradient of sin(pi) is", grad_sine(np.pi))

# test2
def sigmoid(x):
    return 0.5 * (np.tanh(x / 2) + 1)


def logistic_predictions(weights, inputs):
    return sigmoid(np.dot(inputs, weights))


def trainning_loss(weights):
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))


inputs = np.array([[0.52, 1.12, 0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])

trainning_graient_fun = grad(trainning_loss)

weights = np.array([0.0, 0.0, 0.0])

print("Initial loss:", trainning_loss(weights))

for i in range(100):
    weights -= trainning_graient_fun(weights) * 0.01
    print("epoch loss:", trainning_loss(weights))

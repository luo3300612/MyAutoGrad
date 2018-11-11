import autograd.numpy as np
from autograd import grad


def taylor_sine(x):
    ans = currterm = x
    i = 0

    while np.abs(currterm) > 0.001:
        currterm = - currterm * x ** 2 / ((2 * i + 3) * (2 * i + 2))
        ans = ans + currterm
        i += 1
    return ans

grad_sine = grad(taylor_sine)
print("gradient of sin(pi) is", grad_sine(np.pi))
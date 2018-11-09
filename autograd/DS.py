import numpy as np


class Mat:
    def __init__(self, *args, **kwargs):
        self.np = np.array(*args, **kwargs)
        self.oper = ''
        self.fathers = []

    def __add__(self, other):
        return self.np + other

    def __radd__(self, other):
        return self.__add__(self, other)

    def __sub__(self, other):
        return self.np - other


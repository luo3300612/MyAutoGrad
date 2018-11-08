import numpy as np


class Mat:
    def __init__(self, *args, **kwargs):
        self.np = np.array(*args, **kwargs)
        self.oper = ''
        self.fathers = []

    def __add__(self, other):

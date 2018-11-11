import numpy as np
import numbers


class Mat:
    def __init__(self, *args, **kwargs):
        self.require_grad = kwargs.get("require_grad", False)
        try:
            kwargs.pop("require_grad")
        except KeyError:
            pass
        self.np = np.array(*args, **kwargs)
        self.father_left = None
        self.father_right = None
        self.gradient_left = None
        self.gradient_right = None

    def __add__(self, other):
        if isinstance(other, Mat):
            if self.require_grad or other.require_grad:
                ret = Mat(self.np + other.np, require_grad=True)
            else:
                ret = Mat(self.np + other.np)
        elif isinstance(other, numbers.Real):
            if self.require_grad:
                ret = Mat(self.np + other, require_grad=True)
            else:
                ret = Mat(self.np + other)
        else:
            raise NotImplementedError

        if ret.require_grad:
            ret.father_left = self
            ret.father_right = other

            if self.require_grad:
                ret.gradient_left =


    def __radd__(self, other):
        return self.__add__(self, other)

    def __sub__(self, other):
        return self.np - other

    def ret_gen(self):

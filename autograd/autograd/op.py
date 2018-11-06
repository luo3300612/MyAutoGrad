from autograd.autograd.DataStructure import Mat, Node
import math


class op:
    @staticmethod
    def log(x):
        return op.func_base(x, 'log')

    @staticmethod
    def exp(x):
        return op.func_base(x, 'exp')

    @staticmethod
    def func_base(x, fun):
        if isinstance(x, Node):
            ret = Node()
            ret.value = eval("math." + fun + "(x.value)")
            ret.oper = fun
            ret.fathers = [x]
            return ret
        elif isinstance(x, Mat):
            ret = Mat()
            ret.m = x.m
            ret.n = x.n
            ret.mat = []
            ret.fathers = [x]
            ret.oper = fun
            for i in range(x.m):
                row = []
                for j in range(x.n):
                    new_node = op.func_base(x[i][j], fun)
                    new_node.show = False
                    row.append(new_node)
                ret.mat.append(row)
            return ret

    @staticmethod
    def norm_square(x) -> Mat:
        if isinstance(x, Node):
            ret = x * x
        elif isinstance(x, Mat):
            ret = Mat.gen_ret(m=1, n=1)
            node = Node(0)
            for i in range(x.m):
                for j in range(x.n):
                    node = node + x[i][j] * x[i][j]
            ret.mat = [[node]]
        else:
            raise NotImplementedError
        return ret

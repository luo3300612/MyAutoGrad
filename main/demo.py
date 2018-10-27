import numpy as np


class Node:
    def __init__(self, num=None, show=True):
        self.value = num
        self.fathers = []
        self.children = []
        self.oper = None
        self.show = show

    def __add__(self, other):
        return self.cal_base(other, '+')

    def __sub__(self, other):
        return self.cal_base(other, '-')

    def __mul__(self, other):
        return self.cal_base(other, '*')

    def __truediv__(self, other):
        return self.cal_base(other, '/')

    def cal_base(self, other, oper):
        """base of elementary calculation"""
        ret = Node()
        ret.value = eval("self.value" + oper + "other.value")
        ret.oper = oper
        ret.fathers = [self, other]
        self.children.append(ret)
        other.children.append(other)
        return ret

    def grad(self, target):
        """partial self partial target"""
        if self is target:
            return 1
        elif len(self.fathers) is 0:
            return 0
        else:
            gradient_left = 0
            gradient_right = 0

            if self.oper == "+":
                gradient_left = 1
                gradient_right = 1
            elif self.oper == "-":
                gradient_left = 1
                gradient_right = -1
            elif self.oper == "*":
                gradient_left = self.fathers[1].value
                gradient_right = self.fathers[0].value
            elif self.oper == "/":
                gradient_left = 1 / self.fathers[1].value
                gradient_right = -self.fathers[0].value / self.fathers[1].value ** 2

            return gradient_left * self.fathers[0].grad(target) + \
                   gradient_right * self.fathers[1].grad(target)

    def __repr__(self):
        if self.show:
            if len(self.fathers) is not 0:
                return f"""<Node,value:{self.value}={self.fathers[0].value}{self.oper}{self.fathers[1].value}>"""
            else:
                return f"""<Node,value={self.value},ROOT>"""
        else:
            return f"{self.value}"


class Mat:
    def __init__(self, elements=None):
        self.m = 0
        self.n = 0
        self.fathers = []
        self.chirdren = []
        self.oper = None
        if elements is not None:
            self.m = len(elements)
            self.n = len(elements[0])
        self.mat = []
        for i in range(self.m):
            row = []
            for j in range(self.n):
                row.append(Node(elements[i][j], show=False))
            self.mat.append(row)

    def cal_base(self, other, oper):
        assert self.m == other.m and self.n == other.n
        ret = Mat()
        ret.m = self.m
        ret.n = self.n
        ret.mat = []

        ret.fathers = [self, other]
        self.chirdren.append(ret)
        other.fathers.append(ret)
        self.oper = oper

        for i in range(self.n):
            row = []
            for j in range(self.m):
                new_node = eval("self.mat[i][j]" + oper + "other.mat[i][j]")
                new_node.show = False
                row.append(new_node)
            ret.mat.append(row)
        return ret

    def __add__(self, other):
        return self.cal_base(other, '+')

    def __sub__(self, other):
        return self.cal_base(other, '-')

    def __mul__(self, other):
        assert self.n == other.m
        ret = Mat()
        ret.m = self.m
        ret.n = other.n
        ret.mat = []

        ret.fathers = [self, other]
        self.chirdren.append(ret)
        other.fathers.append(ret)
        self.oper = '*'

        for i in range(ret.m):
            row = []
            for j in range(ret.n):
                new_node = Node(0, show=False)
                for k in range(self.n):
                    new_node = new_node + self.mat[i][k] * other.mat[k][j]
                    new_node.show = False
                row.append(new_node)
            ret.mat.append(row)
        return ret

    def grad(self, target):
        if self is target:
            return

    @staticmethod
    def ones(m, n):
        return Mat.const_mat_base(m, n, 1)

    @staticmethod
    def zeros(m, n):
        return Mat.const_mat_base(m, n, 0)

    @staticmethod
    def const_mat_base(m, n, const):
        ret = Mat()
        ret.m = m
        ret.n = n
        ret.mat = []
        for i in range(m):
            row = []
            for j in range(n):
                new_node = Node(const, show=False)
                row.append(new_node)
            ret.mat.append(row)
        return ret

    def __repr__(self):
        to_show = [repr(item) for item in self.mat]
        to_show = '\n'.join(to_show)
        return to_show


if __name__ == "__main__":
    # test for long term gradient
    # A = Node(2)
    # B = Node(3)
    # C = A * B
    # D = Node(4)
    # E = C * D
    # F = B + E
    # G = F - D
    #
    # print(G)
    # print(G.grad(A))
    # print(G.grad(B))
    # print(G.grad(C))
    # print(G.grad(D))
    # print(G.grad(E))
    # print(G.grad(F))

    # test for identical fathers
    # A = Node(2)
    # B = A * A
    # C = A * B
    # print(B.grad(A))

    # test for matrix plus
    # mat1 = Mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # mat2 = Mat([[3, 1, 9], [4, 5, 6], [7, 8, 9]])
    # print(mat1 + mat2)

    # test for in-position
    # A = Node(2)
    # B = Node(3)
    # C = Node(4)
    #
    # A = A * B
    # A = A * C
    # print(A.grad(B))
    # print(A.grad(C))

    # test for matrix mul
    # mat1 = Mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # mat2 = Mat([[3, 1, 1], [1, 4, 5], [1, 7, 8]])
    # mat3 = mat1 * mat2
    # mat4 = mat1 + mat2
    # mat5 = mat1 - mat2
    # print(mat1)
    # print(mat2)
    # print(mat3)
    # print(mat4)
    # print(mat5)

    #test for ones and zeros
    mat1 = Mat.ones(5,6)
    mat2 = Mat.zeros(9,9)
    print(mat1)
    print(mat2)

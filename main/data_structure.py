import numbers
import math


class Node:
    def __init__(self, num=None, show=False):
        self.value = num
        self.fathers = []
        self.children = []
        self.oper = None
        self.show = show

    def __add__(self, other):
        return self.oper_base(other, '+')

    def __sub__(self, other):
        return self.oper_base(other, '-')

    def __mul__(self, other):
        return self.oper_base(other, '*')

    def __truediv__(self, other):
        return self.oper_base(other, '/')

    def oper_base(self, other, oper):
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
        elif len(self.fathers) == 1:
            gradient = 0
            if self.oper == "log":
                gradient = 1 / self.fathers[0].value
            elif self.oper == "exp":
                gradient = self.value
            return gradient * self.fathers[0].grad(target)
        elif len(self.fathers) == 2:
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
            if len(self.fathers) == 2:
                return f"""<Node,value:{self.value}={self.fathers[0].value}{self.oper}{self.fathers[1].value}>"""
            elif len(self.fathers) == 1:
                return f"""<Node,value:{self.value}={self.oper}({self.fathers[0].value})>"""
            else:
                return f"""<Node,value={self.value},ROOT>"""
        else:
            return f"{self.value}"


class Mat:
    # TODO randome matrix generator
    # TODO Mat Iterable
    # TODO Return one place

    def __init__(self, elements=None):
        self.m = 0
        self.n = 0
        self.fathers = []
        self.children = []
        self.oper = None
        if elements is not None:
            self.m = len(elements)
            self.n = len(elements[0])
        self.mat = []
        for i in range(self.m):
            row = []
            if len(elements[i]) != self.n:
                raise AttributeError("Dim2 is not identical")
            for j in range(self.n):
                row.append(Node(elements[i][j], show=False))
            self.mat.append(row)

    @classmethod
    def gen_ret(cls, m, n, mat=None, fathers=None, oper=None):
        ret = Mat()
        ret.m = m
        ret.n = n
        ret.mat = [] if mat is None else mat
        ret.fathers = [] if fathers is None else fathers
        if fathers is not None:
            for father in fathers:
                if isinstance(father, cls):
                    father.children.append(ret)
        ret.oper = None if oper is None else oper
        return ret

    def cal_base(self, other, oper):
        assert self.m == other.m and self.n == other.n

        ret = Mat.gen_ret(m=self.m,
                          n=self.n,
                          fathers=[self, other],
                          oper=oper
                          )

        for i in range(self.m):
            row = []
            for j in range(self.n):
                new_node = eval("self.mat[i][j]" + oper + "other.mat[i][j]")
                new_node.show = False
                row.append(new_node)
            ret.mat.append(row)
        return ret

    def __add__(self, other):
        if isinstance(other, numbers.Real):
            return self.scalar_oper_exchangeable(other, '+')
        else:
            return self.cal_base(other, '+')

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, numbers.Real):
            return self.scalar_oper_exchangeable(other, '-')
        else:
            return self.cal_base(other, '-')

    def __rsub__(self, other):
        if isinstance(other, numbers.Real):
            return self.scalar_oper_right(other, '-')

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return self.scalar_oper_exchangeable(other, '/')
        else:
            raise AttributeError

    def __rtruediv__(self, other):
        if isinstance(other, numbers.Real):
            return self.scalar_oper_right(other, '/')

    # TODO Need to make sure gradient when do scalar multiplication
    def __mul__(self, other):
        if isinstance(other, Mat):
            try:
                assert self.n == other.m
            except AssertionError:
                print(f"Dim not match:{self.m}*{self.n} mul {other.m}*{other.n}")
                raise AssertionError

            ret = Mat.gen_ret(m=self.m,
                              n=other.n,
                              fathers=[self, other],
                              oper='*'
                              )

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
        elif isinstance(other, numbers.Real):
            return self.scalar_oper_exchangeable(other, '*')

    def __rmul__(self, other):
        return self * other

    def scalar_oper_exchangeable(self, other, oper):
        """
        I treat scalar oper by simply create Node for each element of
        Mat and do operation.
        I do not record result as chirldren of this scalar because I
        don't create any Mat for the scalar.

        Another possible implementation:
        transfer scalar operation to Mat operation by create a scalar
        matrix
        """
        ret = Mat.gen_ret(m=self.m,
                          n=self.n,
                          fathers=[self, other],
                          oper=oper
                          )

        for i in range(self.m):
            row = []
            for j in range(self.n):
                new_node = eval("self.mat[i][j]" +
                                oper +
                                "Node(other)"
                                )
                row.append(new_node)
            ret.mat.append(row)
        return ret

    def scalar_oper_right(self, other, oper):
        ret = Mat.gen_ret(m=self.m,
                          n=self.n,
                          fathers=[other, self],
                          oper=oper
                          )
        for i in range(self.m):
            row = []
            for j in range(self.n):
                new_node = eval("Node(other)" +
                                oper +
                                "self.mat[i][j]"
                                )
                row.append(new_node)
            ret.mat.append(row)
        return ret

    def __getitem__(self, item):
        return self.mat[item]

    # TODO is return as list of list of Node appropriate?
    def grad(self, target):
        """
        grad inplement gradient of a Mat to a Mat (self.n == 1),
        we use Denominator layout which means parital Ax partial x is A.T()
        :param target:
        :return: partial self partial target
        """
        if len(self.fathers) == 2:
            if target.n == 1:
                ret = Mat()
                ret.m = target.m
                ret.n = self.m
                ret.mat = []

                for i in range(ret.m):
                    row = []
                    for j in range(ret.n):
                        new_node = Node(self.mat[j][0].grad(target.mat[i][0]), show=False)
                        row.append(new_node)
                    ret.mat.append(row)
                return ret
            else:
                raise NotImplementedError

    @staticmethod
    def ones(m, n):
        return Mat.const_mat_base(m, n, 1)

    @staticmethod
    def zeros(m, n):
        return Mat.const_mat_base(m, n, 0)

    @staticmethod
    def eye(m, n):
        return Mat.const_mat_base(m, n, 1, True)

    @staticmethod
    def const_mat_base(m, n, const, eye=False):
        ret = Mat()
        ret.m = m
        ret.n = n
        ret.mat = []
        for i in range(m):
            row = []
            for j in range(n):
                if eye:
                    if i == j:
                        new_node = Node(1, show=False)
                    else:
                        new_node = Node(0, show=False)
                else:
                    new_node = Node(const, show=False)
                row.append(new_node)
            ret.mat.append(row)
        return ret

    def T(self):
        Nodes = []
        for j in range(self.n):
            for i in range(self.m):
                Nodes.append(self.mat[i][j])
        ret = Mat()
        ret.mat = []
        ret.m, ret.n = self.n, self.m

        index = 0
        for i in range(ret.m):
            row = []
            for j in range(ret.n):
                row.append(Nodes[index])
                index += 1
            ret.mat.append(row)
        return ret

    def zero_grad(self):
        for i in range(self.m):
            for j in range(self.n):
                self.mat[i][j].fathers = []

    def values(self):
        ret = []
        for i in range(self.m):
            row = []
            for j in range(self.n):
                row.append(self.mat[i][j].value)
            ret.append(row)
        return ret

    def __format__(self, format_spec):
        to_show = []
        for i in range(self.m):
            row = self[i]
            to_show.append(repr([format(item.value, format_spec) for item in row]))
        return '\n'.join(to_show)

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

    # test for ones and zeros and eye
    # mat1 = Mat.ones(5,6)
    # mat2 = Mat.zeros(9,9)
    # mat3 = Mat.eye(6,4)
    # print(mat1)
    # print(mat2)
    # print(mat3)

    # test for mat grad
    # mat1 = Mat([[2, 3, 4], [5, 6, 8]])
    # mat2 = Mat([[2], [3], [4]])
    # mat3 = mat1 * mat2
    # print("mat1")
    # print(mat1)
    # print("mat2")
    # print(mat2)
    # print("mat3")
    # print(mat3)
    # print("partial mat3 partial mat2")
    # print(mat3.grad(mat2))

    # test for .T
    # matA = Mat([[1, 2, 3], [1, 1, 1]])
    # print(matA)
    # print(matA.T())
    # print(matA)

    # test for quadratic form
    # matA = Mat([[1, 2, 3], [2, 1, 1], [3, 1, 1]])
    # matx = Mat([[1], [2], [3]])
    # matC = matx.T() * matA * matx
    # print("matC\n",matC)
    # print("C.grad(x)\n",matC.grad(matx))
    # print("matA * matx\n",matA*matx)
    # print("matA.T * matx\n",matA.T()*matx)

    # test for log and exp
    # node1 = Node(5)
    # node2 = node1.log()

    # test for Mat getitem
    # matA = Mat([[1, 2, 3], [2, 1, 1], [3, 1, 1]])
    # print(matA[1:3])

    # test for scalar + - * /
    matA = Mat([[1, 2, 3], [2, 3, 3]])
    print(matA + 2)
    print(2 + matA)
    #
    print(matA - 2)
    print(2 - matA)

    print(matA * 2)
    print(2 * matA)

    print(matA / 2)
    print(2 / matA)

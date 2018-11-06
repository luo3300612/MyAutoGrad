import numbers
import math


class Node:
    def __init__(self, num=None, show=False):
        self.value = num
        self.fathers = []
        self.oper = None
        self.show = show

    def __add__(self, other):
        if isinstance(other, numbers.Real):
            return self.oper_base(Node(other), '+')
        return self.oper_base(other, '+')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, numbers.Real):
            return self.oper_base(Node(other), '-')
        return self.oper_base(other, '-')

    def __rsub__(self, other):
        return Node(other).oper_base(self, '-')

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return self.oper_base(Node(other), '*')
        return self.oper_base(other, '*')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return self.oper_base(Node(other), '/')
        return self.oper_base(other, '/')

    def __rtruediv__(self, other):
        return Node(other).oper_base(self, '/')

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.value == other.value
        elif isinstance(other, numbers.Real):
            return self.value == other

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.value < other.value
        elif isinstance(other, numbers.Real):
            return self.value < other

    def __gt__(self, other):
        if isinstance(other, Node):
            return self.value > other.value
        elif isinstance(other, numbers.Real):
            return self.value > other

    def oper_base(self, other, oper):
        """base of elementary calculation"""
        ret = Node()
        ret.value = eval("self.value" + oper + "other.value")
        ret.oper = oper
        ret.fathers = [self, other]
        return ret

    def grad(self, target):
        """partial self partial target"""
        if self is target:
            gradient = 1
        elif len(self.fathers) is 0:  # root node
            gradient = 0
        elif len(self.fathers) == 1:  # func node
            gradient = 0
            if self.oper == "log":
                gradient = 1 / self.fathers[0].value
            elif self.oper == "exp":
                gradient = self.value
            gradient = gradient * self.fathers[0].grad(target)
        elif len(self.fathers) == 2:  # ret node
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

            gradient = gradient_left * self.fathers[0].grad(target) + \
                       gradient_right * self.fathers[1].grad(target)
        else:
            raise NotImplementedError
        return gradient

    def __repr__(self):
        if self.show:
            if len(self.fathers) == 2:
                ret = f"""<Node,value:{self.value}={self.fathers[0].value}{self.oper}{self.fathers[1].value}>"""
            elif len(self.fathers) == 1:
                ret = f"""<Node,value:{self.value}={self.oper}({self.fathers[0].value})>"""
            else:
                ret = f"""<Node,value={self.value},ROOT>"""
        else:
            ret = f"{self.value}"
        return ret


class Mat:

    def __init__(self, elements=None):
        self.m = 0
        self.n = 0
        self.fathers = []
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
        ret.oper = None if oper is None else oper
        return ret

    def cal_base(self, other, oper):
        try:
            assert self.m == other.m and self.n == other.n
        except AssertionError:
            raise DimNotMatchError(self, other, oper)
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
            return self.scalar_oper(other, '+', False)
        else:
            return self.cal_base(other, '+')

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, numbers.Real) or isinstance(other, Node):
            return self.scalar_oper(other, '-', False)
        else:
            return self.cal_base(other, '-')

    def __rsub__(self, other):
        if isinstance(other, numbers.Real) or isinstance(other, Node):
            return self.scalar_oper(other, '-', True)

    def __truediv__(self, other):
        if isinstance(other, numbers.Real) or isinstance(other, Node):
            return self.scalar_oper(other, '/', False)
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        if isinstance(other, numbers.Real) or isinstance(other, Node):
            return self.scalar_oper(other, '/', True)

    def __mul__(self, other):
        if isinstance(other, Mat):
            try:
                assert self.n == other.m
            except AssertionError:
                raise DimNotMatchError(self, other, "*")

            ret = Mat.gen_ret(m=self.m,
                              n=other.n,
                              fathers=[self, other],
                              oper='*'
                              )

            for i in range(ret.m):
                row = []
                for j in range(ret.n):
                    new_node = Node(0)
                    for k in range(self.n):
                        try:
                            new_node = new_node + self.mat[i][k] * other.mat[k][j]
                        except Exception as e:
                            print(e)
                            raise MatMulInternalError(self, other)
                    row.append(new_node)
                ret.mat.append(row)
            return ret
        elif isinstance(other, numbers.Real):
            return self.scalar_oper(other, '*', False)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return 0 - self

    def scalar_oper(self, other, oper, reverse):
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
                if not reverse:
                    new_node = eval("self.mat[i][j]" +
                                    oper +
                                    "other"
                                    )
                else:
                    new_node = eval("other" +
                                    oper +
                                    "self.mat[i][j]"
                                    )
                row.append(new_node)
            ret.mat.append(row)
        return ret

    def __eq__(self, other):
        if self.m != other.m or self.n != other.n:
            return False
        else:
            for i in range(self.m):
                for j in range(self.n):
                    if self[i][j] != other[i][j]:
                        return False
            return True

    def __getitem__(self, item):
        return self.mat[item]

    # TODO is return as list of list of Node appropriate?
    def grad(self, target):
        """
        grad inplement gradient of a Mat to a Mat (self.n == 1),
        I use Denominator layout which means parital Ax partial x is A.T()
        Though grad return a Mat, high-order gradient is not implemented as
        grad return a Mat with Nodes without fathers
        """
        if len(self.fathers) == 2:  # for Mat calculated by Binocular operation
            if target.n == 1:
                ret = Mat.gen_ret(m=target.m,
                                  n=self.m,
                                  )

                for i in range(ret.m):
                    row = []
                    for j in range(ret.n):
                        new_node = Node(self[j][0].grad(target[i][0]), show=False)
                        row.append(new_node)
                    ret.mat.append(row)
            else:
                raise NotImplementedError
        elif self.m == 1 and self.n == 1:  # grad scalar gard Mat
            ret = Mat.gen_ret(m=target.m,
                              n=target.n,
                              )
            for i in range(ret.m):
                row = []
                for j in range(ret.n):
                    new_node = Node(self[0][0].grad(target[i][j]))
                    row.append(new_node)
                ret.mat.append(row)
        else:
            raise NotImplementedError

        return ret

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
        ret = Mat.gen_ret(m=m,
                          n=n,
                          )
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

    # TODO optimization

    @property
    def T(self):
        Nodes = []
        for j in range(self.n):
            for i in range(self.m):
                Nodes.append(self.mat[i][j])
        ret = Mat.gen_ret(m=self.n,
                          n=self.m,
                          )

        index = 0
        for i in range(ret.m):
            row = []
            for j in range(ret.n):
                row.append(Nodes[index])
                index += 1
            ret.mat.append(row)
        return ret

    def zero_grad(self):
        if len(self.fathers) == 0:
            for i in range(self.m):
                for j in range(self.n):
                    self.mat[i][j].fathers = []
        else:
            for father in filter(lambda n: isinstance(n, Mat), self.fathers):
                father.zero_grad()

    @property
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


class MatMulInternalError(Exception):

    def __init__(self, mat1, mat2):
        err = f"MatMulInternalError\n,mat1:\n{mat1}\n,mat2:\n{mat2}\n"
        Exception.__init__(self, err)


class DimNotMatchError(Exception):

    def __init__(self, mat1, mat2, oper):
        err = f"DimNotMatchError:{mat1.m}*{mat1.n} {oper} {mat2.m}*{mat2.n}"
        Exception.__init__(self, err)

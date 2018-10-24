class Node:
    def __init__(self, num=None):
        self.value = num
        self.fathers = []
        self.children = []
        self.oper = None

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
        if len(self.fathers) is not 0:
            return f"""<Node,value:{self.value}={self.fathers[0].value}{self.oper}{self.fathers[1].value}>"""
        else:
            return f"""<Node,value={self.value},ROOT>"""


if __name__ == "__main__":
    A = Node(2)
    B = Node(3)
    C = A * B
    D = Node(4)
    E = C * D
    F = B + E
    G = F - D

    print(G)
    print(G.grad(A))
    print(G.grad(B))
    print(G.grad(C))
    print(G.grad(D))
    print(G.grad(E))
    print(G.grad(F))
    A = Node(2)
    B = A * A
    C = A * B
    print(B.grad(A))

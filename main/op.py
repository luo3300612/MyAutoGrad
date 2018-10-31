from base import Mat, Node
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
        ret = Node()
        ret.value = eval("math." + fun + "(x.value)")
        ret.oper = fun
        ret.fathers = [x]
        x.children.append(ret)
        return ret


if __name__ == "__main__":
   node1 = Node(3)
   node2 = Node(4)

   node3 = node1 * node2

   node4 = op.log(node3)

   node5 = op.exp(node4)

   print(node5.grad(node2))
from data_structure import Mat, Node
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
            x.children.append(ret)
            return ret
        elif isinstance(x, Mat):
            ret = Mat()
            ret.m = x.m
            ret.n = x.n
            ret.mat = []
            ret.fathers = [x]
            ret.oper = fun
            x.children.append(ret)
            for i in range(x.m):
                row = []
                for j in range(x.n):
                    new_node = op.func_base(x.mat[i][j], fun)
                    new_node.show = False
                    row.append(new_node)
                ret.mat.append(row)
            return ret


if __name__ == "__main__":
    # test for node log exp
    # node1 = Node(3)
    # node2 = Node(4)
    # node3 = node1 * node2
    # node4 = op.log(node3)
    # node5 = op.exp(node4)
    # print(node5.grad(node2))

    # test for Mat log exp
    # mat1 = Mat([[1,2],[2,3],[2,math.e]])
    # print(op.log(mat1))
    # print()
    # print(op.exp(mat1))



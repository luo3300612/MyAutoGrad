from myautograd.DataStructure import Node, Mat, DimNotMatchError
from myautograd.op import op
import unittest
import math


class TestNodeMethods(unittest.TestCase):

    def test_long_term_gradient(self):
        A = Node(2)
        B = Node(3)
        C = A * B
        D = Node(4)
        E = C * D
        F = B + E
        G = F - D
        self.assertEqual(G.grad(A), 12)
        self.assertEqual(G.grad(B), 9)
        self.assertEqual(G.grad(C), 4)
        self.assertEqual(G.grad(D), 5)
        self.assertEqual(G.grad(E), 1)
        self.assertEqual(G.grad(F), 1)
        self.assertEqual(G.grad(G), 1)

    def test_same_nodes_operation(self):
        A = Node(2)
        B = A * A
        C = A * B
        self.assertEqual(C.grad(A), 12)

    def test_in_position(self):
        A = Node(2)
        B = Node(3)
        A = A * B
        self.assertEqual(A.grad(B), 2)

    def test_log_exp(self):
        node1 = Node(5)
        self.assertEqual(op.log(node1).value, math.log(node1.value))
        self.assertEqual(op.exp(node1).value, math.exp(node1.value))

    def test_node_eq(self):
        node1 = Node(1)
        node2 = Node(2)
        node3_1 = Node(3)
        node3_2 = node1 + node2
        self.assertTrue(node3_2 == node3_1)
        self.assertFalse(node1 == node2)

    def test_node_log_exp_grad(self):
        node1 = Node(3)
        node2 = Node(4)
        node3 = node1 * node2
        node4 = op.log(node3)
        node5 = op.exp(node4)
        self.assertEqual(node5.grad(node1), 4)

    def test_node_scalar_oper(self):
        node1 = Node(5)
        node2 = Node(2)
        self.assertEqual(node1 + 5, Node(10))
        self.assertEqual(5 + node1, Node(10))
        self.assertEqual(node1 - 2, Node(3))
        self.assertEqual(2 - node1, Node(-3))
        self.assertEqual(2 * node1, Node(10))
        self.assertEqual(node1 * 2, Node(10))
        self.assertEqual(node2 / 2, Node(1))
        self.assertEqual(4 / node2, Node(2))

    def test_lt_gt(self):
        node1 = Node(5)
        node2 = Node(7)
        self.assertTrue(node1 < node2)
        self.assertFalse(node1 > node2)
        self.assertTrue(node1 < 6)
        self.assertFalse(node1 > 10)


class TestMatMethods(unittest.TestCase):

    def test_mat_plus(self):
        mat1 = Mat([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

        mat2 = Mat([[3, 1, 9],
                    [4, 5, 6],
                    [7, 8, 9]])

        mat3 = mat1 + mat2
        self.assertEqual(
            mat3,
            Mat([[4, 3, 12],
                 [8, 10, 12],
                 [14, 16, 18]])
        )

    def test_mat_mul(self):
        mat1 = Mat([[1, 0, 3],
                    [1, 0, 3],
                    [2, -1, 0]])

        mat2 = Mat([[3, 1, 1],
                    [1, 1, 0],
                    [1, 4, 2]])

        mat3 = mat1 * mat2

        self.assertEqual(
            mat3,
            Mat([[6, 13, 7],
                 [6, 13, 7],
                 [5, 1, 2]])
        )

        mat4 = mat2 * mat1

        self.assertEqual(
            mat4,
            Mat([[6, -1, 12],
                 [2, 0, 6],
                 [9, -2, 15]])
        )

    def test_eye(self):
        mat = Mat.eye(5, 6)
        self.assertEqual(mat,
                         Mat([
                             [1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0]
                         ]))

    def test_ones(self):
        mat = Mat.ones(5, 6)
        self.assertEqual(mat,
                         Mat([
                             [1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1]
                         ]))

    def test_zeros(self):
        mat = Mat.zeros(5, 6)
        self.assertEqual(mat,
                         Mat([
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]
                         ]))

    def test_mat_grad(self):
        mat1 = Mat([[2, 3, 4], [5, 6, 8]])
        mat2 = Mat([[2], [3], [4]])
        mat3 = mat1 * mat2
        self.assertEqual(mat3.grad(mat2), mat1.T)

    def test_mat_T(self):
        mat1 = Mat([[2, 3, 4],
                    [5, 6, 8]])

        mat2 = Mat([[2, 5],
                    [3, 6],
                    [4, 8]])

        self.assertEqual(mat1.T, mat2)

    def test_quadratic_from_grad(self):
        matA = Mat([[1, 2, 3], [2, 1, 1], [3, 1, 1]])
        matx = Mat([[1], [2], [3]])
        matC = matx.T * matA * matx

        self.assertEqual(matC.grad(matx),
                         matA * matx + matA.T * matx)

    def test_scalar_oper(self):
        matA = Mat([[1, 2, 3], [2, 3, 3]])
        self.assertEqual(
            matA + 2,
            Mat([[3, 4, 5],
                 [4, 5, 5]])
        )
        self.assertEqual(
            matA - 2,
            Mat([[-1, 0, 1],
                 [0, 1, 1]])
        )
        self.assertEqual(
            2 - matA,
            Mat([[1, 0, -1],
                 [0, -1, -1]])
        )
        self.assertEqual(
            matA * 2,
            Mat([[2, 4, 6],
                 [4, 6, 6]])
        )
        self.assertEqual(
            matA / 2,
            Mat([[0.5, 1, 1.5],
                 [1, 1.5, 1.5]])
        )
        self.assertEqual(
            6 / matA,
            Mat([[6, 3, 2],
                 [3, 2, 2]])
        )

    def test_mat_eq(self):
        mat1 = Mat([[1, 2, 3], [4, 5, 6]])
        mat2 = Mat([[1, 2], [3, 4]])
        mat3 = Mat([[1, 0, 0], [0, 0, 0]])
        mat4 = Mat([[0, 2, 3], [4, 5, 6]])
        mat5 = mat3 + mat4
        self.assertFalse(mat3 == mat4)
        self.assertTrue(mat5 == mat1)
        self.assertFalse(mat1 == mat2)

    def test_norm_square(self):
        mat1 = Mat([[1, 2], [2, 3], [2, math.e]])
        self.assertEqual(op.norm_square(mat1), Mat([[22 + math.e * math.e]]))
        self.assertEqual(op.norm_square(mat1).grad(mat1), 2 * mat1)

    def test_dim_not_match_error(self):
        mat1 = Mat([[1, 2], [2, 3], [2, math.e]])
        mat2 = mat1.T
        mat3 = Mat([[1]])
        with self.assertRaises(DimNotMatchError):
            mat4 = mat1 + mat2
        with self.assertRaises(DimNotMatchError):
            mat5 = mat2 * mat3

    def test_argmax_argmin(self):
        mat1 = Mat([[1, 2, 3, 4, 5, 6, 9, 7]])
        mat2 = Mat([[1, 3, 5, 3, 3, 422, 32, 4, ]]).T
        self.assertEqual(op.argmax(mat1), 6)
        self.assertEqual(op.argmax(mat2), 5)


if __name__ == "__main__":
    unittest.main()

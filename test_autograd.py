from autograd.autograd.DataStructure import Node, Mat
from autograd.autograd.op import op

if __name__ == "__main__":
    # test for scalar grad Mat
    matA = Mat([[1,2], [2,3]])
    print(op.norm_square(matA).grad(matA))

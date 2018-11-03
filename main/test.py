from DataStructure import Node, Mat
from op import op

if __name__ == "__main__":
    # test for scalar grad Mat
    matA = Mat([[1,2], [2,3]])
    print(op.norm_suqare(matA).grad(matA))

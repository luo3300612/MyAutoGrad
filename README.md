# This is autogradient implememted by myself just for fun!

To ask for grad, just use A.grad(B) , which can give you partial A partial B

**Efficiency is the last thing I will consider.**
## core
[core](https://github.com/luo3300612/MyDL/blob/master/autograd/autograd/DataStructure.py)

## example
* [LinearRegression](https://github.com/luo3300612/MyDL/blob/master/autograd/examples/LinearRegression.py)
* [LogisticRegression](https://github.com/luo3300612/MyDL/blob/master/autograd/examples/LogisticRegression.py)
* [Neural Network](https://github.com/luo3300612/MyAutoGrad/blob/master/autograd/examples/NN.py)
## Other usages 
```angular2html
>>> mat1 = Mat([[1, 2], [2, 3], [2, math.e]])
>>> format(op.log(mat1), '.3f')
['0.000', '0.693']
['0.693', '1.099']
['0.693', '1.000']

```

## Log 
### Nov 5
* refactor project structure
* add unittest
### Nov 6
* .T() -> .T
* add MatMulInternalError
* add NN for exclusive-or problem
### Nov 7
* remove children of Mat and Node
* optimize zero_grad to simplify computation graph
* add node scalar operation, make Mat scalar operation depend on it 
* overload > < of Node
## TODO
* add flag require_grad to simplify computation graph 
* fix format
* random
## Thought
* Node-based gradient -> Mat-based gradient
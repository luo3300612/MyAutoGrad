# Note
## 网页
* [autograd tutorial](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md)
* [numpy extend](https://docs.scipy.org/doc/numpy-1.15.1/user/c-info.how-to-extend.html)
* [chainner](https://github.com/chainer/chainer)
* [Pytorch](https://openreview.net/pdf?id=BJJsrmfCZ
)
* [AD blog](https://blog.csdn.net/daniel_ustc/article/details/77133329)
    * 几种微分方法
        * 手动求解法
        * 数值微分法
        * 符号微分法
        * 自动微分法
    * 自动微分的两个方法
        * 运算符重载
        * 代码生成
    * foward-mode 和 reverse-mode
* [tensorflow eager execution](https://www.jianshu.com/p/9a212532e319)
* [AD page](http://www.autodiff.org/)
* [知乎](https://www.zhihu.com/question/48356514/answer/123290631)
    * 介绍了自动微分的两个方法
    * 给出了forward-mode的c++代码，一次只能求一个变量的梯度
* [c++](http://www.met.reading.ac.uk/~swrhgnrj/publications/adept.pdf)
* [维基百科](https://en.wikipedia.org/wiki/Automatic_differentiation)
    * 自动微分得益于任何复杂的计算结果的微分都是由简单的+-*/和基本初等函数的复合得到的，链式法则
    * 自动微分与数值微分和符号微分的区别
        * 符号微分代码效率低下，且难以实现
        * 数值微分引入了舍入误差
        * 两种方法对于高阶导数的处理都不好
        * 两种方法对于多输入处理都不好
    * 自动微分的两个计算方法
        * forawrd mode，计算导数的顺序与前向传播一致
        * reverse mode，计算导数的顺序与前向传播相反，计算量比forward-mode少一半，但需要存储Wengert list("tape")，对于f:R^n->R^m，若m远小于n，则reverse mode 更高效
* [Autodidact](https://github.com/mattjj/autodidact)
## Autograd结构
### util.py
* subvals(x:tuple, ivs:list[(i,v)]) -> let x[i] = v 
* subval(x:tuple, i:int, v:num) -> let x[i] = v
### wrap_util.py
**unary_to_nary** ???

**wraps(fun,...)**
```angular2html
wraps(f)
def g(x)
    return x + 2
let g.__name__ = f.__name__
let g.__doc__ = f.__doc__
```
* wrap_nary_f
### defferential_operators.py
* grad(fun, x) 返回

### core.py

### tracer.py
**Node类**
* trace

Node类是一个抽象类，其中有三个方法
* __init__ 初始化方法
* initialize_root
* new_root 类方法(被classmethod装饰)

前两个方法都只有一行 assert False，表明Node类是一个抽象类，无法创造实例。

new_root方法通过调用__new__创建实例，再调用initialize_root


## Autodidact结构

### core.py
`func defvjp`:
在primitive_vjp中注册一个函数，对于函数fun，记录其对于第k个变量的Jacobian到primitive_vjp[fun][k]中




## 笔记
### warning 的写法
```angular2html
import warnings
warning_msg = """This is a warning message"""

def fun_warning():
    warnings.warn(warning_msg)

```

### grad
grad包裹一个实值函数之后，与实值函数接受相同的参数，但返回的是实值函数的导数

### 抽象类
子类通过__new__和initialize_root方法创建实例
```angular2html
class Node(object):
    __slots__ = []
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        assert False

    def initialize_root(self, *args, **kwargs):
        assert False

    @classmethod
    def new_root(cls, *args, **kwargs):
        root = cls.__new__(cls)
        root.initialize_root(*args, **kwargs)
        return root
```

## 启发
* 仅支持常值函数的微分
* 微分通过Jacobian矩阵相乘实现
* Node-based实际上是一种reverse mode的微分方法
* Node-based是一种运算符重载方法

## 名词
* reverse-mode automatic differentiation

## 问题
* forward mode 和 reverse mode 效率上究竟有什么不同？
* 反向传播出现的非二维Jacobian如何处理
* unary_to_nary做了啥(wrap_util)
* 为什么VJPNode的initialize_root方法中的value没有用？

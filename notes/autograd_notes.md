# Note
## 网页
* [autograd tutorial](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md)
* [numpy extend](https://docs.scipy.org/doc/numpy-1.15.1/user/c-info.how-to-extend.html)
* [chainner](https://github.com/chainer/chainer)
* [Pytorch](https://openreview.net/pdf?id=BJJsrmfCZ
)
## 结构
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


## 装饰器
### wraps
wraps(fun, namestr="{fun}", docstr="{doc}", **kwargs)



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
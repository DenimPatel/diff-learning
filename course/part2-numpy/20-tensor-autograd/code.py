"""
Build a GPT from an empty file, one diff at a time.
Part 2: the same algorithm on NumPy arrays, so that experiments run in seconds.
Following @karpathy's microgpt.py: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
"""

import os       # os.path.exists
import random   # random.seed, random.choices, random.shuffle
import numpy as np # n-dimensional arrays: one operation now touches many numbers at once
random.seed(42) # Let there be order among chaos
rng = np.random.default_rng(42) # ...for numpy, too

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be Autograd, now over whole arrays ("tensors") instead of single numbers.
# Same idea as Value: every node remembers its children and how to pass the gradient back to them.
# A scalar local derivative is not enough anymore, so each child gets a vector-Jacobian product (vjp):
# a function mapping d(loss)/d(node) to d(loss)/d(child).
def unbroadcast(grad, shape):
    # numpy broadcasting stretched the child to the output's shape; sum the gradient back down to it
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    __slots__ = ('data', 'grad', '_children', '_vjps')

    def __init__(self, data, children=(), vjps=()):
        self.data = np.asarray(data, dtype=np.float64) # array value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, same shape as data after backward
        self._children = children       # children of this node in the computation graph
        self._vjps = vjps               # one vector-Jacobian product per child

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data + other.data, (self, other), (lambda g: g, lambda g: g))

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data * other.data, (self, other), (lambda g: g * other.data, lambda g: g * self.data))

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data @ other.data, (self, other),
                      (lambda g: g @ np.swapaxes(other.data, -1, -2), lambda g: np.swapaxes(self.data, -1, -2) @ g))

    def __pow__(self, other): return Tensor(self.data**other, (self,), (lambda g: g * other * self.data**(other-1),))
    def log(self): return Tensor(np.log(self.data), (self,), (lambda g: g / self.data,))
    def exp(self): return Tensor(np.exp(self.data), (self,), (lambda g: g * np.exp(self.data),))
    def relu(self): return Tensor(np.maximum(self.data, 0), (self,), (lambda g: g * (self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    # new: operations that only exist for arrays
    def sum(self, axis=None, keepdims=False):
        def vjp(g): # every summed element gets the same gradient
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return np.broadcast_to(g, self.data.shape)
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), (vjp,))
    def mean(self, axis=-1): return self.sum(axis=axis, keepdims=True) * (1 / self.data.shape[axis])
    def reshape(self, *shape): return Tensor(self.data.reshape(shape), (self,), (lambda g: g.reshape(self.data.shape),))
    def transpose(self, *axes): return Tensor(self.data.transpose(axes), (self,), (lambda g: g.transpose(np.argsort(axes)),))
    def __getitem__(self, idx): # indexing, e.g. picking rows of an embedding table
        def vjp(g):
            grad = np.zeros_like(self.data)
            np.add.at(grad, idx, g) # rows picked several times accumulate their gradients
            return grad
        return Tensor(self.data[idx], (self,), (vjp,))

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            for child, vjp in zip(v._children, v._vjps):
                child.grad = child.grad + unbroadcast(vjp(v.grad), child.data.shape)

# --- demo: check the tensor autograd against numerical gradients ---
A = Tensor(rng.normal(size=(3, 4)))
B = Tensor(rng.normal(size=(4, 2)))
f = lambda A, B: (((A @ B).relu() + 1).log() * (A[0] ** 2).sum()).mean(axis=0).sum()
L = f(A, B)
L.backward()
print(f"L = {L.data:.6f}")
h = 1e-6
for name, T in [('A', A), ('B', B)]:
    numerical = np.zeros_like(T.data)
    for i in np.ndindex(T.data.shape):
        old = T.data[i]
        T.data[i] = old + h
        up = f(A, B).data
        T.data[i] = old
        numerical[i] = (up - L.data) / h
    print(f"dL/d{name}: max |autograd - numerical| = {np.abs(T.grad - numerical).max():.2e}")
print(f"dL/dA =\n{A.grad.round(4)}")

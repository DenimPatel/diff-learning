"""
Build a GPT from an empty file, one diff at a time.
Pure, dependency-free Python, following @karpathy's microgpt.py:
https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

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

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

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
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# --- demo: build a tiny expression, then let backward() compute every derivative ---
a_val = 2.0  # @param {"min": -3, "max": 3, "step": 0.1} value of a
b_val = -3.0 # @param {"min": -3, "max": 3, "step": 0.1} value of b
a, b = Value(a_val), Value(b_val)
c = a * b        # c depends on a and b
d = c + a ** 2   # a is used twice: its gradients from both paths add up
L = d * 0.5
L.backward()
print(f"L = {L.data:.4f}")
print(f"dL/da = {a.grad:.4f}   by hand: 0.5 * (b + 2a) = {0.5 * (b_val + 2 * a_val):.4f}")
print(f"dL/db = {b.grad:.4f}   by hand: 0.5 * a = {0.5 * a_val:.4f}")

# the same derivatives, estimated numerically by nudging each input a tiny bit
f = lambda a, b: 0.5 * (a * b + a ** 2)
h = 1e-6
print(f"numerical dL/da = {(f(a_val + h, b_val) - f(a_val, b_val)) / h:.4f}")
print(f"numerical dL/db = {(f(a_val, b_val + h) - f(a_val, b_val)) / h:.4f}")

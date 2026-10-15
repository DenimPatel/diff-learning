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

# New: hold out names the model never trains on. Their loss tells us how well it generalizes.
data_fraction = 1.0 # @param {"min": 0.001, "max": 1, "log": true} fraction of the training names to use (tiny = memorize)
n_val = len(docs) // 10 # 10% of the names are only ever evaluated, never trained on
val_docs = docs[:n_val]
train_docs = docs[n_val:][:max(1, int(data_fraction * (len(docs) - n_val)))]
print(f"train docs: {len(train_docs)}, val docs: {len(val_docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# New: tokenize everything up front into one long stream: BOS name BOS name BOS ...
stoi = {ch: i for i, ch in enumerate(uchars)}
encode = lambda docs: np.array([BOS] + [t for doc in docs for t in [stoi[ch] for ch in doc] + [BOS]])
train_data, val_data = encode(train_docs), encode(val_docs)

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

# Initialize the parameters, to store the knowledge of the model
n_layer = 2     # @param {"min": 1, "max": 6} depth of the transformer neural network (number of layers)
n_embd = 32     # @param {"options": [8, 16, 32, 48, 64, 128]} width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # @param {"options": [1, 2, 4, 8]} number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
# note: matrices are stored as (nin, nout) so that the forward pass reads x @ w
matrix = lambda nin, nout, std=0.08: Tensor(rng.normal(0, std, size=(nin, nout)))
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(n_embd, vocab_size)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(n_embd, 4 * n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(4 * n_embd, n_embd)
params = list(state_dict.values()) # a handful of Tensors instead of thousands of Values
print(f"num params: {sum(p.data.size for p in params)}")

# Define the model architecture: the same GPT as microgpt, but on a whole batch of sequences at once.
# Shapes: B = batch size, T = sequence length, C = n_embd, nh = n_head, hd = head_dim
def linear(x, w):
    return x @ w

def softmax(logits):
    exps = (logits - logits.data.max(axis=-1, keepdims=True)).exp()
    return exps / exps.sum(axis=-1, keepdims=True)

def log_softmax(logits): # log(softmax(x)) computed in a numerically safe way
    shifted = logits - logits.data.max(axis=-1, keepdims=True)
    return shifted - shifted.exp().sum(axis=-1, keepdims=True).log()

def rmsnorm(x):
    ms = (x * x).mean(axis=-1)
    return x * (ms + 1e-5) ** -0.5

# causal mask: position t may only attend to positions <= t (microgpt got this for free by appending keys one at a time)
causal_mask = np.triu(np.full((block_size, block_size), -1e9), k=1)

def gpt(idx): # idx: (B, T) array of token ids -> logits (B, T, vocab_size)
    B, T = idx.shape
    x = state_dict['wte'][idx] + state_dict['wpe'][np.arange(T)] # joint token and position embedding
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head Attention block: all heads and all positions in a few matmuls
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq']).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3) # (B, nh, T, hd)
        k = linear(x, state_dict[f'layer{li}.attn_wk']).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        v = linear(x, state_dict[f'layer{li}.attn_wv']).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        attn_logits = q @ k.transpose(0, 1, 3, 2) / head_dim**0.5 + causal_mask[:T, :T] # (B, nh, T, T)
        attn_weights = softmax(attn_logits)
        x_attn = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(B, T, n_embd) # concatenate the heads again
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = x + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = x.relu()
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    logits = linear(x, state_dict['lm_head'])
    return logits

# Random windows from a token stream; targets are the inputs shifted by one
def get_batch(data, batch_size, rng=rng):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    xb = np.stack([data[i:i + block_size] for i in ix])
    yb = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return xb, yb

def loss_fn(xb, yb): # mean cross-entropy over every position of every sequence
    logprobs = log_softmax(gpt(xb))
    B, T = yb.shape
    return -logprobs[np.arange(B)[:, None], np.arange(T)[None, :], yb].sum() / yb.size

# New: a fixed set of windows to evaluate on, so that the numbers are comparable from one eval to the next
eval_rng = np.random.default_rng(0)
eval_batches = {'train': [get_batch(train_data, 64, eval_rng) for _ in range(4)],
                'val': [get_batch(val_data, 64, eval_rng) for _ in range(4)]}
def evaluate(split): # average loss on held-out windows (we only read .data: no update follows)
    return float(np.mean([loss_fn(xb, yb).data for xb, yb in eval_batches[split]]))

# Let there be Adam, the blessed optimizer and its buffers (now one array per parameter tensor)
learning_rate = 0.01 # @param {"min": 0.0001, "max": 0.1, "log": true} peak step size of the optimizer
beta1 = 0.85         # @param {"min": 0, "max": 0.999} momentum: how much of the past gradient direction to keep
beta2 = 0.99         # @param {"min": 0.5, "max": 0.9999} how slowly the per-parameter gradient scale adapts
eps_adam = 1e-8
m = [np.zeros_like(p.data) for p in params] # first moment buffer
v = [np.zeros_like(p.data) for p in params] # second moment buffer

# Repeat in sequence
num_steps = 300 # @param {"min": 1, "max": 5000, "log": true} number of training steps
batch_size = 32 # @param {"min": 1, "max": 128, "log": true} sequences per step (microgpt: 1 document)
eval_interval = 25 # @param {"min": 1, "max": 500, "log": true} measure train and val loss every this many steps
accum_steps = 4 # @param {"options": [1, 2, 4, 8, 16, 32]} split each batch into this many micro-batches, processed one at a time
assert batch_size % accum_steps == 0, "batch_size must be a multiple of accum_steps"
micro_batch = batch_size // accum_steps # only this many sequences' activations are in memory at once

# new: check that accumulating micro-batch gradients gives the same gradient as one big batch
xb, yb = get_batch(train_data, batch_size, np.random.default_rng(1))
loss_fn(xb, yb).backward()
full = [p.grad for p in params]
for p in params: p.grad = 0
for i in range(accum_steps):
    sl = slice(i * micro_batch, (i + 1) * micro_batch)
    (loss_fn(xb[sl], yb[sl]) / accum_steps).backward() # each backward() *adds* into p.grad
print(f"{accum_steps} micro-batches of {micro_batch} vs one batch of {batch_size}: "
      f"max gradient difference {max(np.abs(p.grad - g).max() for p, g in zip(params, full)):.1e}")
for p in params: p.grad = 0

for step in range(num_steps):

    # new: several forward/backward passes on micro-batches, gradients summed, then ONE optimizer step
    loss_sum = 0
    for micro_step in range(accum_steps):
        xb, yb = get_batch(train_data, micro_batch)
        loss = loss_fn(xb, yb) / accum_steps # divide so the sum is the mean over the full batch
        loss.backward() # gradients accumulate in p.grad until the optimizer zeroes them
        loss_sum += loss.data
    loss = Tensor(loss_sum) # the full-batch loss, for printing

    # Adam optimizer update: the same formulas as microgpt, applied to whole arrays
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    # New: every eval_interval steps, compare the loss on names it trained on with names it has never seen
    if (step + 1) % eval_interval == 0 or step == num_steps - 1:
        train_loss, val_loss = evaluate('train'), evaluate('val')
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f} | lr {lr_t:.5f} | train {train_loss:.4f} | val {val_loss:.4f} | gap {val_loss - train_loss:.4f}")
    else:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f} | lr {lr_t:.5f}")

# The val loss in other units: bits per character (BOS counts as a character) and perplexity
print(f"val loss {val_loss:.4f} nats = {val_loss / np.log(2):.3f} bits per character, perplexity {np.exp(val_loss):.2f}")

# Inference: may the model babble back to us
num_samples = 10 # @param {"min": 1, "max": 50} number of names to generate
temperature = 0.5 # @param {"min": 0.05, "max": 2, "step": 0.05} in (0, 1], control the "creativity" of generated text, low to high
print("--- inference (new, hallucinated names) ---")
train_set = set(train_docs) # new: flag samples the model memorized rather than invented
for sample_idx in range(num_samples):
    tokens = [BOS]
    for pos_id in range(block_size):
        logits = gpt(np.array([tokens])).data[0, -1] / temperature # only the prediction at the last position matters
        probs = np.exp(logits - logits.max())
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == BOS:
            break
        tokens.append(token_id)
    name = ''.join(uchars[t] for t in tokens[1:])
    print(f"sample {sample_idx+1:2d}: {name}{'  (copied from the training set)' if name in train_set else ''}")

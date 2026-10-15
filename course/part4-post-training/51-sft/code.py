"""
Build a GPT from an empty file, one diff at a time.
Part 2: the same algorithm on NumPy arrays, so that experiments run in seconds.
Following @karpathy's microgpt.py: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
"""

import os       # os.path.exists
import hashlib  # hashlib.sha1, to name a checkpoint after the settings that produced it
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

# Repeat in sequence. From now on this is called *pretraining*: the expensive part we only want to do once
pretrain_steps = 300 # @param {"min": 1, "max": 5000, "log": true} number of pretraining steps
batch_size = 32 # @param {"min": 1, "max": 128, "log": true} sequences per step (microgpt: 1 document)
eval_interval = 25 # @param {"min": 1, "max": 500, "log": true} measure train and val loss every this many steps
save_every = 100 # @param {"min": 1, "max": 1000, "log": true} write a checkpoint every this many steps
stop_after = 0   # @param {"min": 0, "max": 1000} simulate the job dying after this many steps (0 = never)

# New: a checkpoint holds everything needed to carry on: the weights, the optimizer state and the step.
# Its file name is a hash of the settings that produced it, so any later run with the same settings reuses it.
config = dict(n_layer=n_layer, n_embd=n_embd, n_head=n_head, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
              pretrain_steps=pretrain_steps, batch_size=batch_size, data_fraction=data_fraction,
              shapes=[p.data.shape for p in params]) # edited the model? different shapes, different file
ckpt_path = 'ckpt_' + hashlib.sha1(repr(sorted(config.items())).encode()).hexdigest()[:10] + '.npz'

def save_checkpoint(path, step):
    arrays = {f'param.{name}': p.data for name, p in state_dict.items()}
    arrays.update({f'adam_m.{i}': m[i] for i in range(len(params))})
    arrays.update({f'adam_v.{i}': v[i] for i in range(len(params))})
    np.savez(path, step=step, **arrays)

def load_checkpoint(path): # restore in place: params, m and v keep pointing at the same objects
    ckpt = np.load(path)
    for name, p in state_dict.items():
        p.data = ckpt[f'param.{name}'].copy()
    for i in range(len(params)):
        m[i], v[i] = ckpt[f'adam_m.{i}'].copy(), ckpt[f'adam_v.{i}'].copy()
    return int(ckpt['step'])

start_step = 0
if os.path.exists(ckpt_path):
    start_step = load_checkpoint(ckpt_path)
    print(f"loaded {ckpt_path}: resuming at step {start_step} / {pretrain_steps}")
for step in range(start_step, pretrain_steps):

    # Take a batch of random windows from the training stream only
    xb, yb = get_batch(train_data, batch_size)

    # Forward the whole batch, building up the computation graph all the way to the loss
    loss = loss_fn(xb, yb)

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # Adam optimizer update: the same formulas as microgpt, applied to whole arrays
    lr_t = learning_rate * (1 - step / pretrain_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    # New: every eval_interval steps, compare the loss on names it trained on with names it has never seen
    if (step + 1) % eval_interval == 0 or step == pretrain_steps - 1:
        train_loss, val_loss = evaluate('train'), evaluate('val')
        print(f"pretrain {step+1:4d} / {pretrain_steps:4d} | loss {loss.data:.4f} | train {train_loss:.4f} | val {val_loss:.4f}") # not charted

    # New: save every so often, so that a crash only loses the steps since the last save
    if (step + 1) % save_every == 0 or step == pretrain_steps - 1:
        save_checkpoint(ckpt_path, step + 1)
    if step + 1 == stop_after:
        print(f"simulated crash after step {step+1}: the last checkpoint is at step {(step + 1) // save_every * save_every}. Run again to resume.")
        break

# The val loss in other units: bits per character (BOS counts as a character) and perplexity
val_loss = evaluate('val') # measured again, because a complete checkpoint skips the loop entirely
print(f"val loss {val_loss:.4f} nats = {val_loss / np.log(2):.3f} bits per character, perplexity {np.exp(val_loss):.2f}")

# Inference: may the model babble back to us
num_samples = 10 # @param {"min": 1, "max": 50} number of names to generate
temperature = 0.5 # @param {"min": 0.05, "max": 2, "step": 0.05} in (0, 1], control the "creativity" of generated text, low to high
def generate(tokens, temperature): # a helper that continues a list of token ids until BOS
    tokens = list(tokens)
    while len(tokens) < block_size:
        logits = gpt(np.array([tokens])).data[0, -1] / temperature # only the prediction at the last position matters
        probs = np.exp(logits - logits.max())
        token_id = random.choices(range(len(probs)), weights=probs)[0]
        if token_id == BOS:
            break
        tokens.append(token_id)
    return tokens

# New: Supervised Fine-Tuning (SFT). The base model only continues text. To teach it a task, we show it
# examples of a prompt followed by the answer we want. The toy task: write the name backwards.
#   BOS e m m a = a m m e BOS
# '=' is a new special token that separates the prompt from the response (chat models use <|user|>, <|assistant|> ...)
SEP = vocab_size
uchars = uchars + ['', '='] # for decoding: id BOS prints as nothing, id SEP as '='
vocab_size += 1
# Make room for it: one new row in the embedding table and one new column in the output layer,
# initialized to the average of the existing ones so that it starts out as a "typical" token
state_dict['wte'].data = np.concatenate([state_dict['wte'].data, state_dict['wte'].data.mean(0, keepdims=True)])
state_dict['lm_head'].data = np.concatenate([state_dict['lm_head'].data, state_dict['lm_head'].data.mean(1, keepdims=True)], axis=1)

max_len = (block_size - 2) // 2 # longest name whose prompt + response (2 * len + 3 tokens) fits in the context
sft_train = [d for d in train_docs if len(d) <= max_len]
sft_val = [d for d in val_docs if len(d) <= max_len][:200]
mask_prompt = True # @param {"options": [true, false]} only learn from the response tokens (false: also from the prompt)

def sft_batch(names, prompt_too=False): # -> inputs, targets, and a 0/1 mask that says which targets count in the loss
    xb = np.full((len(names), block_size), BOS)
    yb = np.full((len(names), block_size), BOS)
    mask = np.zeros((len(names), block_size))
    for b, name in enumerate(names):
        prompt = [BOS] + [stoi[ch] for ch in name] + [SEP]
        seq = prompt + [stoi[ch] for ch in reversed(name)] + [BOS]
        xb[b, :len(seq) - 1], yb[b, :len(seq) - 1] = seq[:-1], seq[1:]
        start = 0 if prompt_too else len(prompt) - 1 # target index of the first response token
        mask[b, start:len(seq) - 1] = 1 # padding after the final BOS never counts
    return xb, yb, mask

def sft_loss(xb, yb, mask): # cross-entropy averaged over the masked-in positions only
    logprobs = log_softmax(gpt(xb))
    B, T = yb.shape
    return -(logprobs[np.arange(B)[:, None], np.arange(T)[None, :], yb] * mask).sum() / mask.sum()

def reverse_with_model(name):
    prompt = [BOS] + [stoi[ch] for ch in name] + [SEP]
    out = generate(prompt, temperature=1e-6)[len(prompt):] # temperature ~0: always the most likely token
    return ''.join(uchars[t] for t in out)

print(f"sft examples: {len(sft_train)} train, {len(sft_val)} val (names up to {max_len} letters)")
print("--- before fine-tuning ---")
for name in sft_val[:3]:
    print(f"{name}= -> {reverse_with_model(name)}")

def response_accuracy(): # fraction of response tokens predicted right on held-out names (inputs are the true tokens)
    xb, yb, mask = sft_batch(sft_val)
    pred = gpt(xb).data.argmax(-1)
    return float(((pred == yb) * mask).sum() / mask.sum())

# Fine-tuning is the same training loop with a different batch and loss, a fresh optimizer and a smaller learning rate
ft_learning_rate = 0.003 # @param {"min": 0.0001, "max": 0.03, "log": true} peak step size while fine-tuning
num_steps = 400 # @param {"min": 1, "max": 3000, "log": true} number of fine-tuning steps
m = [np.zeros_like(p.data) for p in params] # new optimizer state: the wte and lm_head shapes changed
v = [np.zeros_like(p.data) for p in params]
for step in range(num_steps):
    xb, yb, mask = sft_batch(random.choices(sft_train, k=batch_size), prompt_too=not mask_prompt)
    loss = sft_loss(xb, yb, mask)
    loss.backward()
    lr_t = ft_learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
    if (step + 1) % eval_interval == 0 or step == num_steps - 1:
        # acc: is it learning the task? names: is it forgetting how to write names (the pretraining val loss)?
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f} | acc {response_accuracy():.3f} | names {evaluate('val'):.4f}")
    else:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

print("--- after fine-tuning: held-out names, written backwards ---")
results = [(name, reverse_with_model(name)) for name in sft_val[:max(num_samples, 50)]]
for name, out in results[:num_samples]:
    print(f"{name}= -> {out}  {'ok' if out == name[::-1] else 'wrong'}")
print(f"exact match: {sum(out == name[::-1] for name, out in results)} / {len(results)}")
print("--- and it can still make up names (BOS, no prompt) ---")
for sample_idx in range(3):
    print(f"sample {sample_idx+1:2d}: {''.join(uchars[t] for t in generate([BOS], temperature)[1:])}")

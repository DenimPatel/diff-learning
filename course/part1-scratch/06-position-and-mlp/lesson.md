This lesson adds three ingredients the transformer needs. All three go *around* the model we already have.

### Position embedding
`wpe` is a second embedding table, indexed by **position** (0..15) instead of by token. Adding it to the token embedding lets the model know *where* it is: "3rd letter of the name" can now mean something. `gpt()` takes a `pos_id` from now on.

### MLP block
A two-layer network: expand to `4 * n_embd` hidden units, apply **ReLU** (`max(0, x)`, the non-linearity that lets stacked layers compute more than one matrix could), then project back down. This is where transformers do most of their "thinking per token".

### Residual connection
`x = mlp(x) + x`. The block only learns a *change* to its input instead of a whole new representation. Gradients flow straight through the `+` back to earlier layers, and this is what makes deep networks trainable. Look at the diff: the block sits inside `for li in range(n_layer)`, a stack we'll add more blocks to.

### But...
The model still only sees **one token**. Knowing the position helps, since names start and end differently, but it cannot read the previous letters. Fixing that is the next lesson, and it is the key idea of the transformer.

⏱ Pure Python is slow from here on: each step builds a graph of thousands of `Value`s. The default is 200 steps.

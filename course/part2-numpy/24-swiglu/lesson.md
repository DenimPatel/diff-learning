The ReLU MLP computes `down(relu(up(x)))`. **SwiGLU** adds a **gate**:

`down( silu(gate(x)) * up(x) )`

- `silu(x) = x · sigmoid(x)` (also called *swish*) is a smooth ReLU that lets small negative values through.
- The elementwise product means one projection decides, for each hidden unit, *how much* of the other passes through. Multiplicative interactions like this are hard for a plain ReLU MLP to express.

It has three matrices instead of two, so the hidden size shrinks to `⅔ · 4 · n_embd` to keep the parameter count (and the comparison) fair. The printout shows the counts nearly match.

Overlay the loss curve with lesson 21's. Gains at this tiny scale are modest; the paper's improvement is consistent at scale, which is why nearly every open LLM since PaLM and Llama uses it. Also notice that `silu` is built from existing ops (`exp`, `**`, `*`), so the autograd needs no changes.

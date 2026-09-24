Part 1 was slow because every number was a Python object: a single matrix multiply built thousands of graph nodes. **The fix is not a new algorithm, only bigger nodes.** A `Tensor` holds a whole NumPy array, and one graph node now stands for millions of multiply-adds that NumPy runs in fast compiled code.

This lesson branches off **lesson 02** so the diff shows `Value → Tensor` line by line:

| `Value` (scalar) | `Tensor` (array) |
|---|---|
| `_local_grads`: a number per child | `_vjps`: a **function** per child |
| `child.grad += local * grad` | `child.grad += unbroadcast(vjp(grad))` |

- **Vector-Jacobian product (vjp).** For arrays, "the local derivative" is a huge Jacobian matrix. We never build it: each op only says how to map the upstream gradient to its input's gradient. For `C = A @ B` that is `dA = dC @ Bᵀ` and `dB = Aᵀ @ dC`.
- **Broadcasting.** When NumPy stretches a `(C,)` bias across a `(B, T, C)` batch, the gradient must be *summed* back to `(C,)`. `unbroadcast` does that for every op in one place.
- **New ops** that only exist for arrays: `sum`, `mean`, `reshape`, `transpose` and indexing (`__getitem__`, which is how embedding lookups work; rows picked twice accumulate gradient with `np.add.at`).

The demo checks every gradient against numerical differentiation, the same test as lesson 02, now on matrices. When you add a new op, always run this check.

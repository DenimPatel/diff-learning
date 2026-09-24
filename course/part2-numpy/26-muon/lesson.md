Adam treats a weight matrix as a bag of independent numbers. **Muon** treats it as a *matrix*. The gradient of a weight matrix is usually dominated by a few directions (it is nearly low-rank), so a plain step mostly pushes along those few directions. Muon **orthogonalizes** the update: it keeps the directions (the singular vectors) but sets every singular value to ~1, so all directions move by the same amount.

Computing that exactly needs an SVD, which is slow on GPUs. `newton_schulz` gets close with 5 iterations of a matrix polynomial, using only matmuls. The odd coefficients `(3.4445, -4.7750, 2.0315)` were tuned to push singular values toward 1 as fast as possible.

In the diff:
- Only the **2D hidden matrices** (attention and MLP) use Muon. Embeddings and `lm_head` stay on Adam, as in the reference implementation.
- It uses Nesterov momentum, then orthogonalization, then a shape-dependent scale.

Overlay the curve with lesson 21 (Adam). Muon set speed records in the nanoGPT speedrun and was scaled to a 1T-parameter model (Kimi K2), which makes it one of the more notable optimizer developments since Adam.

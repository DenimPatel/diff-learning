The learned `wpe` table gives each *absolute* position its own vector. But language mostly cares about **relative** position ("the previous letter"), and a learned table cannot handle positions it never saw in training.

**RoPE** deletes `wpe`. It splits each query and key into pairs of numbers, treats each pair as a 2D point, and **rotates** it by an angle `position × freq`. Rotations have one key property:

`rotate(q, i) · rotate(k, j)  =  q · rotate(k, j − i)`

The attention score therefore depends only on the **distance** `i − j`. Different pairs rotate at different frequencies (`base^(-2d/head_dim)`), so some pairs track fine distances and others coarse ones.

In code, rotating a pair `(a, b)` by angle θ is `(a·cosθ − b·sinθ, b·cosθ + a·sinθ)`, which is `x·cos + swap(x)·sin`. We write `swap` as a constant matrix multiply, so our autograd needs no new op. Only `q` and `k` are rotated, never `v`.

**Up to date:** context-extension methods such as NTK scaling, YaRN and the large `rope_base` values (500k–10M) in recent long-context models all work by changing these frequencies.

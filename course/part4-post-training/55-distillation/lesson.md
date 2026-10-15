A training example says "after `em` comes `m`". The teacher model knows much more than that: after `em` it might give `m` 40%, `i` 25%, `a` 10%, and so on. That distribution, with all its "wrong" answers and their relative likelihoods (*dark knowledge*), is a far richer training signal. **Knowledge distillation** trains a small **student** to match it.

### The diff
- The pretrained model becomes the `teacher`, and `new_model()` builds a smaller `student` (1 layer, width 16, ~6× fewer parameters). `gpt()` reads its weights and sizes from globals, so `use(model)` swaps them.
- Each step, the teacher runs forward only and produces probabilities. The student is trained on a mix of two losses:
  - `soft`: cross-entropy between the teacher's and the student's distributions, both softened by `distill_temp`. A higher temperature makes small probabilities count more. The `T²` factor keeps the gradient size steady when you change `T`.
  - `hard`: the usual cross-entropy against the real next character.
- `loss = alpha · soft + (1 − alpha) · hard`. The chart shows `hard` and `val` so that runs with different `alpha` compare fairly.

### What to try
Run the default, then **No teacher**, and compare `val` with the overlaid curves. The distilled student ends up lower with the same architecture, data and steps. Soft targets are less noisy than one-hot labels, so each batch teaches more.

**Up to date:** distillation is how most small open models are made. Gemma 2/3's small models, Llama 3.2 1B/3B and many "-mini" models were trained on a larger model's logits. Distillation from a reasoning model's *generated text* (SFT on its outputs, as in the DeepSeek-R1 distilled models) is a cheaper cousin when the teacher's logits aren't available. **On-policy** distillation (GKD) has the student generate and the teacher grade each token, which fixes the mismatch between training and inference.

Adam keeps two numbers per weight (`m` and `v`), so its state is twice the size of the model. For a 70B model that's 560 GB in fp32, more than the weights themselves. Many optimizers try to do as well with less memory, or better with the same. `update()` implements four of them behind the `optimizer` knob. Their learning rates live on different scales, so `lr_scale` adjusts the shared `learning_rate` for each.

### The rules
- **adam**: momentum divided by the RMS of recent gradients: a per-weight step size.
- **lion** (found by an automated program search at Google): `update = sign(0.9·m + 0.1·g)`. Every weight moves by exactly `±lr`, so it needs a smaller learning rate, and it keeps **one** buffer instead of two.
- **adafactor** (used to train T5 and PaLM): drops momentum and approximates Adam's `v` for a matrix by the outer product of a **row** average and a **column** average of `g²`. Memory drops from `n×m` to `n+m` numbers per matrix. It also clips the update's RMS to 1 for stability.
- **sgd** with momentum: no per-weight scaling at all, one global step size.

The script prints how many numbers of optimizer state each one keeps.

### What to try
Overlay all four on the val chart. Adam and Adafactor end level, Lion close behind, and SGD clearly behind. SGD's lack of per-weight scaling is why transformers are hard to train with it. **SGD, too hot** shows how fragile its step size is.

**Up to date:** AdamW is still the default. 2025 benchmarks that tune every optimizer carefully find most "Adam killers" give modest gains at best, with the clearest wins from matrix-aware methods like **Muon** (lesson 26) and SOAP, especially at small scale. Kimi K2 and several 2025–26 models were trained with Muon. Memory-light optimizers (Adafactor, 8-bit Adam, Lion) remain popular for fine-tuning on limited hardware.

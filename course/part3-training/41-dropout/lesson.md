Lesson 40 showed overfitting: with few training names, the model memorizes them and the val loss climbs. **Dropout** is the classic fix. During training, every activation it touches is zeroed with probability `p`, a different random subset each step, and the survivors are scaled by `1/(1−p)` so that their expected value is unchanged ("inverted" dropout). The network can't rely on any single unit or attention link, so it has to spread what it knows across many. That is a strong defense against memorizing individual examples.

### Where it goes
As in the original Transformer and GPT-2, dropout is applied to:
- the **attention weights** (randomly cutting links between positions),
- the output of each **attention** block and each **MLP** block, right before it is added to the residual stream.

### Train vs eval mode
At evaluation and inference time every unit must be used, so `dropout()` returns its input unchanged when `training` is `False`. `evaluate()` switches it off and back on, and sampling switches it off. Thanks to the `1/(1−p)` scaling at training time, nothing needs rescaling at test time. This is PyTorch's `model.train()` / `model.eval()`.

### What to try
Run **300 names, no dropout** and **300 names, dropout 0.3**, then overlay their `val` curves: dropout cuts the val loss from ~6 to ~2.9. Then run **All data, no dropout**: with plenty of data, dropout costs a bit of training speed and buys nothing.

**Up to date:** modern LLM pretraining mostly uses **no dropout**. With trillions of tokens each one is seen about once, so there is nothing to memorize (Llama, Qwen, DeepSeek: dropout 0). It remains common in fine-tuning on small datasets (LoRA dropout), in smaller models and in other domains.

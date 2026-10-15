Cosine decay (lesson 11) has a practical problem: its shape depends on the total number of steps, so you must decide the training length up front. To train longer you start over, and the checkpoints from the middle of a run are never at a low learning rate, so they aren't the best they could be. **Warmup–Stable–Decay** (WSD) fixes this:

1. **Warmup**: ramp up linearly for `warmup_steps`.
2. **Stable**: hold the peak learning rate for most of training.
3. **Decay**: drop to 0 over the final `decay_fraction` of steps.

### The diff
`get_lr(step)` replaces the inline linear decay and implements all four `schedule` options with the same warmup.

### What the curves show
Switch the chart to `val` and overlay the schedules. WSD sits on a **plateau** during the stable phase, worse than cosine at the same step, then **drops sharply** in the cooldown and ends level with or below cosine. **Constant** never drops. One explanation is a "river valley" landscape: a high learning rate makes fast progress *along* the valley floor while bouncing between its walls, and cooling down settles into the floor. Decaying the learning rate does real work. It isn't just a slower version of training.

### Why practitioners like it
The stable phase can be **continued indefinitely**, and a cooldown can be **branched off** any stable checkpoint to get a finished model at that size. One run then gives you models for many training budgets, which makes scaling-law studies far cheaper. You can also cool down on higher-quality data (annealing).

**Up to date:** WSD or close variants are used by MiniCPM, DeepSeek-V3, OLMo 2, Kimi, SmolLM3 and many 2025–26 runs, often with a cooldown of 10–20% of training and a final "annealing" phase on curated data. Cosine remains common for shorter fixed-budget runs.

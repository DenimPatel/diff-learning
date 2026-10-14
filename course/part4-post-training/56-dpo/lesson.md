SFT needs the *right answer*. For many things we care about (helpful, polite, concise, safe) there is no single right answer, but people can easily say **which of two answers is better**. RLHF turned such comparisons into a reward model and then optimized it with RL (PPO), which is complicated and unstable. **DPO** (Direct Preference Optimization) shows that you can skip both and train on the comparisons directly, with one simple loss.

### The data
A preference dataset is a list of `(prompt, chosen, rejected)` triples. Our prompt is just `BOS`, and our "labelers" have a simple taste, set by `preference`: names that end in `n`, short names, or names that start with a vowel. `chosen` and `rejected` are real names with and without the property. In practice both answers are sampled from the model, and humans or an AI judge pick one.

### The loss
For each pair, with the policy `π` being trained and a frozen copy of the starting model `ref`:

```
reward(y) = β · (log π(y) − log ref(y))
loss      = −log σ( reward(chosen) − reward(rejected) )
```

`seq_logprob` sums the token log-probabilities of a whole name, and the reference values are computed once up front because they never change. The loss raises the chosen name's probability *relative to the reference* and lowers the rejected one's. **β** controls how far the policy may drift: the paper shows DPO solves the same KL-regularized objective as RLHF, so the policy is an optimal one for an implicit reward model.

### The chart
- `margin`: the average `reward(chosen) − reward(rejected)`. `acc` is how often it is positive.
- `liked`: the share of **freshly sampled** names that the labelers would like. This is the real goal.
- `names`: the original val loss. It rises because the model moves away from the original distribution, and `β` limits how far.

### What to try
Run **Reward hacking**: within a few steps every sample is "liked", and every sample is a single letter. The model found the cheapest way to satisfy the preference. This failure mode (over-optimization, *likelihood displacement*) is the reason for `β`, for small learning rates and for short training. **Stronger anchor** shows the trade-off.

**Up to date:** DPO and its variants (IPO, KTO, SimPO, ORPO) are standard in open post-training pipelines (Llama 3, Tülu 3, OLMo 2, Qwen) because they need only a dataset and two forward passes. Frontier labs still use online RL with reward models for much of their preference tuning, and verifiable-reward RL for reasoning. That is the next lesson.

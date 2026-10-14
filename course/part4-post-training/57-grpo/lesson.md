DPO needs a dataset of preferences. For math, code and puzzles we can do better: a **program can check the answer**. Reinforcement learning with verifiable rewards (RLVR) lets the model try, scores each try automatically, and makes the tries that scored well more likely. This is how DeepSeek-R1 and other reasoning models learned to think step by step. **GRPO** (Group Relative Policy Optimization) is the algorithm most of them use.

### One step
1. **Sample.** `n_prompts` prompts (`BOS` + a random first letter), each answered `group_size` times by the current model. `sample_batch` generates all of them in parallel and returns a mask of the generated tokens.
2. **Score.** `reward_fn` returns 1 if the name satisfies `task`, else 0. No reward model, nothing to hack but the checker itself.
3. **Advantage.** Each answer is compared with **its own group**: `adv = (r − mean) / std`. An answer is good if it beat its siblings for the *same* prompt. PPO needs a learned value network to provide this baseline; GRPO gets it for free from the group. A group where every answer scored the same has advantage 0 everywhere and teaches nothing.
4. **Update.** The PPO objective: raise `log π` of tokens in above-average answers and lower it for below-average ones, weighted by `ratio = π / π_old`.
   - **Clipping**: once a token's ratio has moved more than `clip_eps` in the direction it is being pushed, it stops contributing. This keeps one batch from moving the policy too far. It only matters with `updates_per_batch > 1`; on the first update `ratio = 1` exactly. `clipped` charts the fraction of tokens affected.
   - **KL penalty** `kl_coef · KL(π ‖ ref)`, estimated per token as `e^x − x − 1` with `x = log ref − log π` (Schulman's *k3*), keeps the model close to where it started.

### What to try
The default `contains z` goes from ~3% to ~100% reward in about 50 steps, and the samples turn into `zazan` and `zezle`. The model maximized exactly what we measured, which is the central risk of RL. **Hold on to the reference** shows what the KL penalty buys you. **A hard reward** shows the other central problem, *exploration*: when no answer in a group succeeds, there is no signal.

**Up to date:** RLVR is the main ingredient of reasoning models (o1/o3, DeepSeek-R1, Qwen3, Kimi). Recent variants tweak GRPO: DAPO uses asymmetric "clip-higher", token-level averaging and drops all-equal groups, Dr. GRPO removes the std normalization, and many recipes drop the KL term entirely. Our `loss` is already averaged per token, as in DAPO.

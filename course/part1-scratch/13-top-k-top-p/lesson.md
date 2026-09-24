Temperature reshapes the *whole* distribution, including the long tail of unlikely tokens. Each of those tokens is individually rare, but together they get picked surprisingly often, and one bad pick derails the rest of the sample.

- **Top-k**: only sample among the `k` most likely tokens.
- **Top-p (nucleus)**: only sample among the smallest set of tokens whose probabilities add up to `p`. This adapts to the situation: when the model is confident the set is tiny, and when it's unsure the set is wide.

The diff applies both: sort by probability, keep at most `k`, and stop early once the kept mass reaches `p`. Then sample among the survivors (`random.choices` renormalizes the weights for us).

Try **Greedy** (`top_k = 1`): with no randomness left, every sample is identical. Real LLM APIs expose `temperature`, `top_p` and often `top_k`, and now you know exactly what each does. Newer samplers such as *min-p* keep tokens with at least `p ×` the top token's probability. You could add that in the Playground in two lines.

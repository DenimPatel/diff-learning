Now we use the gradient. The training loop of *every* neural network is these four lines:

```python
loss = f(params)                        # 1. forward
loss.backward()                         # 2. backward: gradients
p.data -= learning_rate * p.grad        # 3. step downhill
p.grad = 0                              # 4. reset for the next step
```

Our first "model" has a single parameter `x`, and its loss `(x - 3)²` is lowest at `x = 3`. The gradient `2(x - 3)` points uphill, so we step the opposite way.

### The learning rate
The **learning rate** is how big a step we take. It is the most important hyperparameter you will meet. Run the experiments below and overlay the curves:

| lr | behavior |
|---|---|
| 0.01 | crawls |
| 0.1 | smooth, fast |
| 0.5 | exact in one step (only true for this parabola) |
| 0.95 | oscillates around 3 but gets there |
| 1.05 | **diverges**: each step overshoots further than the last |

For a real network the loss landscape isn't a parabola, but the same failure modes appear: too small wastes compute, too large blows up. Tip: switch the chart to a **log y-axis**.

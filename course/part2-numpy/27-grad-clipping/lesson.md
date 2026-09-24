Every so often a batch produces a huge gradient: a loss spike, a rare token, an unstable attention head. One oversized step can undo hours of training.

**Global norm clipping** measures the length of the gradient of *all* parameters together, `sqrt(Σ grad²)`. If it exceeds `max_grad_norm`, everything is scaled down so the length is exactly `max_grad_norm`. The direction is unchanged; only the step is shorter. Almost every LLM training run uses it, typically with `max_grad_norm = 1.0`.

The chart gets a `grad_norm` series. Watching it is one of the best health checks for a training run: steady growth or sudden spikes are an early warning.

Try the two experiments at a high learning rate and overlay them. Note that Adam already normalizes each step per parameter, so clipping helps less than it would with SGD. Its main job is catching the rare catastrophic spike.

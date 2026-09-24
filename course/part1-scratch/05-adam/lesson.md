SGD uses one learning rate for every parameter. That is a poor fit: gradients of rare characters are tiny and noisy, while common ones are large. **Adam** fixes this with two running averages per parameter:

- `m`, the **first moment**, is an average of recent gradients. This is *momentum*: noise cancels out and consistent directions build up speed.
- `v`, the **second moment**, is an average of recent *squared* gradients, which measures how large this parameter's gradients usually are.

The step is `lr * m / sqrt(v)`. Dividing by `sqrt(v)` normalizes the step, so every parameter moves roughly `lr` per step whatever the scale of its gradient. `m_hat` and `v_hat` correct the bias from starting both buffers at zero.

### Compare
Run lesson 04 (SGD) and then this lesson: both curves stay on the chart. Adam gets to a lower loss in the same number of steps with far less tuning. Note the learning rate: `0.01` here vs `1.0` for SGD. Because of the normalization, Adam's lr is roughly "how far each parameter moves per step".

`beta1` and `beta2` set how long each average remembers. microgpt uses `0.85/0.99`; LLMs typically use `0.9/0.95`.

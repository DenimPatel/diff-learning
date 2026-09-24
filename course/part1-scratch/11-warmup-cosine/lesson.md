microgpt decays the learning rate linearly. Most LLM training runs use **linear warmup followed by cosine decay**:

- **Warmup.** At initialization Adam's second-moment estimates are poor and the gradients are large and chaotic. Starting at the full learning rate can knock the model into a bad region, or blow up a large model entirely. Ramping up over the first steps avoids that.
- **Cosine decay.** Decay from the peak down to `min_lr` along half a cosine: slow at first, fast in the middle, gentle at the end.

Plot the `lr` series to see the schedule's shape, and overlay it with lesson 10's linear decay.

**Up to date:** many recent runs use **WSD** (warmup, stable, decay): hold the peak learning rate for most of training and decay quickly at the end. Because the decay is a short phase, you can branch off any checkpoint and decay from there. Try implementing it in the Playground: it is a three-case `if`.

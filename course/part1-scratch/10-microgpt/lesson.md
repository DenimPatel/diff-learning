Two small changes, and we've arrived.

### Learning rate decay
`lr_t = learning_rate * (1 - step / num_steps)`: start with big steps to make fast progress, then shrink them to zero so the parameters settle into a minimum instead of bouncing around it. Watch the `lr` series in the chart.

### Temperature
Before sampling, divide the logits by `temperature`. Below 1 it sharpens the distribution (safer, more typical names); above 1 it flattens it (more surprising, more mistakes). At → 0 it becomes *greedy* (always the top token). Temperature changes only sampling, not training, and it's the same knob you have in every chat UI.

### 🎉 This is microgpt
Use **compare with → 00** to see the entire file as one diff from the empty file. Apart from the `# @param` knobs and printing the loss on its own line, the file is exactly [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95):

> This file is the complete algorithm. Everything else is just efficiency.

GPT-4-class models differ in scale (billions of parameters, trillions of tokens, thousands of GPUs) and in the refinements covered in the branches that follow. The algorithm is the one you just read.

**Where next?** The branches off this lesson tune training and sampling. **Part 2** rebuilds it on tensors so you can explore modern architecture variants quickly.

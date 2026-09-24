Training means asking, for every parameter: *if I nudge you a little, how does the loss change?* That number is the **gradient**. Autograd computes it for millions of parameters in a single backward pass.

### The `Value`
A `Value` wraps one number (`data`) and remembers:
- its `_children`, the Values it was computed from, and
- its `_local_grads`, the derivative of this node with respect to each child. For `c = a * b` these are `b.data` and `a.data`.

Every operation (`+`, `*`, `**`, `exp`, `log`, `relu`) records these as it goes. By the time we reach the loss, we have built a **computation graph** without any extra effort.

### `backward()`
1. Sort the graph so that every node comes after its children (a **topological sort**).
2. Set `loss.grad = 1`.
3. Walk the graph in reverse and push gradients down the edges: `child.grad += local_grad * node.grad`. That is the **chain rule**. The `+=` matters: a Value used twice (like `a` in the demo) collects gradient from *both* paths.

That's the whole engine. PyTorch's autograd does the same thing on tensors, which we build in Part 2.

### Try it
The demo checks autograd against a derivative worked out by hand and against a *numerical* estimate: nudge the input by `h`, divide the change in output by `h`. Move the `a` and `b` sliders; all three answers should always agree.

<img width="400" height="121" alt="image" src="https://github.com/user-attachments/assets/00bb0309-08bc-4ca0-9892-91131fc00bce" />
Understanding SWAT: Sliding Window Attention Training for Efficient Large Language Models



Recently, I explored the paper "Sliding Window Attention Training for Efficient Large Language Models (SWAT)", which addresses one of the major challenges in efficient attention mechanisms: Attention Sink.



The Attention Sink Problem

Traditional Sliding Window Attention (SWA) improves efficiency by restricting each token to attend only to a local window of previous tokens. However, it often suffers from attention sink, where tokens such as the first token receive disproportionately high attention.

This mainly happens due to:

1. Causal Masking

Since tokens can only attend to previous tokens, positional information becomes implicitly encoded in token representations.

2. Softmax Amplification

Softmax magnifies attention differences. Because early tokens participate in many attention computations, they gradually accumulate excessive attention across layers.



SWAT's Solution:

1. Replace Softmax with Sigmoid

Softmax forces tokens to compete for attention, often leading to dominant sink tokens.

SWAT replaces softmax with sigmoid attention, allowing attention scores to be evaluated more independently, resulting in a more balanced attention distribution and reduced attention sink effects.



2. Add RoPE

Since softmax contributes implicit positional information, replacing it with sigmoid weakens that signal.

To compensate, SWAT explicitly introduces Rotary Positional Embeddings (RoPE), improving relative position awareness and long-context generalization.



3. Introduce Balanced ALiBi

Instead of using only negative distance biases, SWAT applies:

• Positive slopes for half of the attention heads

• Negative slopes for the remaining heads

This creates positional diversity across heads:

Positive-slope heads emphasize historical context.

Negative-slope heads emphasize recent context.



Does SWA Lose Long-Range Information?

A common concern is that limiting attention to a local window may lose information from earlier tokens.

Interestingly, information propagates layer by layer.

For example:

Layer 1: Token 2 receives information from Token 1.

Layer 2: Token 3 receives information from Token 2.

As depth increases, information gradually spreads across the sequence.

The paper formalizes the receptive field as:

1 + (w − 1)L   (w = window size , L = number of layers)

This shows that the effective context grows linearly with depth, enabling long-range information propagation despite local attention.

Final SWAT Attention Equation

Putting everything together, the final attention mechanism combines:

Sliding Window Attention, Sigmoid Attention, RoPE, Balanced ALiBi

(See equation in image)

This formulation mitigates attention sink while maintaining efficient long-context modeling.

Paper: https://arxiv.org/pdf/2502.18845

Code: https://github.com/mayurnayak1705/Sliding-Window-Attention-Training-

#AI #SWAT #SlidingWindowAttention


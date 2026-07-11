# Scaling Monosemanticity (2024)

- **URL**: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- **Authors**: Adly Templeton, Tom Conerly, Jonathan Marcus, et al.
- **arXiv**: [2605.29358](https://arxiv.org/abs/2605.29358) / [PDF](../pdfs/2024-scaling-monosemanticity-arxiv2605.29358.pdf)
- **Key Idea**: SAEs scale to production models — 34 million interpretable features extracted from Claude 3 Sonnet

## From Proof of Concept to Production

The previous paper worked on a one-layer transformer with a few hundred dimensions. This paper goes straight to **Claude 3 Sonnet** — a commercial, multi-billion parameter model that people use every day. The scale difference is crushing.

| | Towards Monosemanticity (2023) | Scaling Monosemanticity (2024) |
|---|---|---|
| Model | 1-layer transformer | Claude 3 Sonnet |
| Parameters | ~thousands | ~billions |
| Features extracted | ~hundreds | **34 million** |
| Feature type | Concrete (DNA, legal text) | Concrete AND abstract |

## Features Become Abstract at Scale

In small models, SAE features are surface-level pattern matchers: "DNA sequences", "Python indentation", "legal jargon." Useful but shallow.

In Claude 3 Sonnet, features include high-level cognitive concepts:

| Feature Type | Examples |
|---|---|
| Concrete | HTTP status codes, DNA sequences, Korean text, LaTeX math |
| Abstract | "Things that could go wrong", "moral reasoning", "uncertainty expressions" |
| Safety-critical | Deception, harmful content generation, sycophantic agreement, bias |
| Meta-cognitive | "I'm confident about this", "I should hedge" |

"Things that could go wrong" isn't about any specific type of error — it fires across ALL domains whenever the context involves something potentially problematic. This is a high-level cognitive concept, not pattern matching.

## Safety-Relevant Features: The Headline Result

For the first time, you can **look inside** a frontier model and see:

- "This model is about to lie" — the deception feature is activating
- "This model is being sycophantic" — the people-pleasing feature is activating
- "This model is generating harmful content" — the harmful-content feature is activating

Previously, you could only guess from output behavior. Now you can read the internal state directly. This is like going from observing symptoms to reading blood tests.

## Feature Steering

Manually adjusting feature activation values predictably alters model behavior:

| Intervention | Result |
|---|---|
| Amplify "Golden Gate Bridge" | Model shoehorns the bridge into every answer. "How's the weather?" → "Like the fog rolling over the Golden Gate Bridge." "How to cook pasta?" → "The color of this sauce reminds me of the bridge's International Orange." |
| Amplify "code quality" | Model writes more careful, well-documented code |
| Suppress "sycophancy" | Model pushes back more, disagrees when appropriate |
| Amplify "uncertainty" | Model hedges more, adds caveats |

The Golden Gate Bridge demo is intentionally absurd — it's so extreme that it leaves no doubt the feature is causally controlling behavior, not just correlated.

## Scaling Laws for Features

More SAE capacity → more features → finer-grained concepts:

```
1M latents:   "code"
              ↓
16M latents:  "Python code" / "JavaScript code" / "error handling code" / "test code"
              ↓
34M latents:  "Python list comprehension" / "JavaScript async/await" / "404 error handling"
```

Features keep splitting into sub-concepts as you give the SAE more room. This implies the real number of features in Claude 3 Sonnet is far beyond 34 million — they just stopped scaling the SAE.

## Why This Matters More Than RLHF

| Approach | Mechanism | Precision | Transparency |
|---|---|---|---|
| RLHF | Change behavior from outside via reward signal | Blunt (affects many behaviors at once) | Black box (don't know what changed internally) |
| Feature steering | Change specific internal representations | Surgical (one concept at a time) | Transparent (know exactly which feature you changed) |

RLHF says "produce fewer harmful outputs." Feature steering says "suppress THIS specific harmful-content feature." The difference is like the difference between "take medicine for general wellness" vs. "target this specific receptor."

## Infrastructure Implications

- **Safety monitoring at inference**: Run SAEs on model activations during inference. Monitor safety-relevant features in real time. Alert if deception feature spikes. Cost: one encoder forward pass per monitored layer.
- **Serving trade-off**: SAE monitoring adds latency and compute. But it gives you internal observability that output-based classifiers can't match. Worth it for high-stakes deployments.
- **Feature-based routing**: Different features could trigger different serving strategies. High-uncertainty features → route to larger model. High-confidence features → serve from smaller model. Feature-aware load balancing.
- **Quantization guidance at scale**: Now you can measure which features are densely packed (high superposition) per layer. Quantize the sparse layers aggressively, protect the dense ones. Data-driven quantization strategy instead of guesswork.

## Reading Notes

- This is the "holy grail" paper of the SAE line. Proves the approach works at production scale.
- The safety implications are the most practically important finding. First time we can identify AND manipulate safety-relevant representations in a frontier model.
- Open question: how to go from 34M isolated features → complete circuits at this scale. That's what the next papers tackle.
- The Golden Gate Bridge demo became a meme in the AI community. It's silly but scientifically rigorous.

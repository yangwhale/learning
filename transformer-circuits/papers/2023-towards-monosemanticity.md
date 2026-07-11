# Towards Monosemanticity (2023)

- **URL**: https://transformer-circuits.pub/2023/monosemantic-features/index.html
- **Authors**: Trenton Bricken, Adly Templeton, Joshua Batson, et al.
- **Local**: [HTML](../pdfs/2023-towards-monosemanticity.html) (no arXiv version)
- **Key Idea**: Sparse Autoencoders can decompose polysemantic neurons into clean, interpretable features

## The Problem → The Solution

Superposition told us the problem: concepts are encoded as directions, neurons are mixed projections, looking at neurons is useless.

The solution is natural: build a tool that finds those directions. An **overcomplete autoencoder** with a **sparsity constraint**.

## How SAEs Work

### Architecture

```
Model activations (512-dim)
        │
    ┌───▼───┐
    │Encoder │  512 → 4096  (overcomplete: 8x wider)
    └───┬───┘
        │
   L1 sparsity penalty (most latents must be zero)
        │
    ┌───▼───┐
    │Decoder │  4096 → 512  (reconstruct original)
    └───┬───┘
        │
Reconstructed activations (512-dim)
```

- **Overcomplete**: 4096 latent dimensions for 512 input dimensions. Enough room for each concept to get its own slot.
- **L1 penalty**: Forces each input to activate only a handful of latents. Most of the 4096 must be zero.
- **Training objective**: Minimize reconstruction error + L1 penalty.

### Why This Decomposes Superposition

Recall from the Superposition paper: concepts are sparse (most inactive at any time) and encoded as directions. The SAE exploits both properties:

1. **Overcomplete** → provides enough slots for all concepts (even more than the original 512 dimensions could hold)
2. **L1 sparsity** → matches the known sparsity structure of concepts (only a few active at a time)
3. **Reconstruction** → the latent directions must faithfully represent the original information

After training, each latent naturally corresponds to one clean concept, because that's the configuration that minimizes reconstruction error while satisfying sparsity.

**Analogy**: A room with 100 people talking simultaneously is just noise in a single recording. But if you know at most 3 people are talking at any moment, and you give each person their own microphone (overcomplete), the SAE is the system of 100 independent microphones that separates the voices.

## Results: Features Are Real

### Clean Interpretability

They trained SAEs on a small one-layer transformer and inspected each latent by finding its top-activating inputs. Results were strikingly clean:

- One latent fires exclusively on DNA sequences
- Another fires exclusively on legal text
- Another fires on Python indentation
- Another on academic citations

Each latent = one concept. No mixing. This is monosemanticity.

### Causal Validity (the critical test)

Correlation isn't enough — you need to prove these features CAUSE model behavior. They ran intervention experiments:

| Intervention | Result |
|-------------|--------|
| Amplify "DNA sequence" feature | Model starts inserting DNA-related content in outputs |
| Suppress "DNA sequence" feature | DNA content disappears from outputs |
| Amplify "Golden Gate Bridge" feature | Model talks about the bridge in unrelated contexts |

These aren't just correlations. The features causally control model behavior.

### Feature Families

Related features cluster together automatically:
- base64 encoding, hex encoding, source code → cluster as "technical text"
- Geographic features cluster by region
- Language features cluster by language family

The SAE learns not just isolated concepts, but the structure between concepts.

## Key Technical Challenges

### L1 Coefficient Tuning

The L1 coefficient is the critical hyperparameter:
- **Too low**: sparsity constraint too weak, latents stay polysemantic (mixed)
- **Too high**: too many latents die (permanently stuck at zero), wasting capacity

Finding the sweet spot requires careful tuning.

### Dead Latents

During training, some latents get pushed to zero by L1 and never recover. They're permanently dead — wasted capacity.

**Fix: Neuron resampling**. Periodically reinitialize dead latents to point toward the direction of maximum reconstruction error. This gives them a second chance to learn something useful. It's like reassigning idle workers to the busiest part of the factory.

### Faithfulness

The SAE reconstruction isn't perfect. Some information is lost. The question is: does the SAE capture the IMPORTANT information? They measure this by running the model with SAE-reconstructed activations instead of original ones and checking if behavior is preserved.

## Limitations

This paper works on a **one-layer transformer** — a tiny model. The features extracted are simple and concrete (DNA, legal text, code). The open question: does this scale to real models with billions of parameters? Are the features in Claude 3 Sonnet also clean and interpretable, or does scale break things?

This is exactly what the next paper answers.

## Infrastructure Implications

- **Feature-level monitoring**: Instead of monitoring neurons (which are mixed), you can monitor interpretable features in production. "Is the deception feature activating?" is a meaningful question now.
- **Targeted intervention**: Feature steering (amplify/suppress specific features) offers a more surgical alternative to RLHF. Instead of training the whole model to avoid harmful outputs, you identify the harmful-content feature and suppress it.
- **Serving cost**: Running SAEs at inference time adds compute (encoder forward pass per layer). For production safety monitoring, this cost-accuracy tradeoff matters.

## Reading Notes

- This is the proof-of-concept paper. SAEs work, features are real, features are causal. Can it scale? → Next paper.
- The "100 microphones" analogy captures the core idea well.
- Pay attention to the failure modes (dead latents, L1 tuning) — they remain challenges even at scale.
- The feature families finding hints at richer structure waiting to be uncovered in larger models.

# Toy Models of Superposition (2022)

- **URL**: https://transformer-circuits.pub/2022/toy_model/index.html
- **Authors**: Nelson Elhage, Tristan Hume, Catherine Olsson, et al.
- **arXiv**: [2209.10652](https://arxiv.org/abs/2209.10652) / [PDF](../pdfs/2022-toy-models-superposition-arxiv2209.10652.pdf)
- **Key Idea**: Neural networks store more features than they have dimensions — and this is a feature, not a bug

## The Problem

You look at a trained neural network, pick any neuron, and observe when it activates. You find it responds to a bunch of completely unrelated things — the same neuron fires for academic citations, Korean text, AND a certain math symbol. This is called **polysemanticity**. You can't say what the neuron "means" because it means a bunch of unrelated things.

Previous explanation: "the network isn't trained well enough" or "it's too small."

This paper's explanation: **the model is doing this on purpose because it's more efficient.**

## Core Mechanism

### The Capacity Problem

Suppose the world has 1,000 independent concepts to represent, but your network only has 500 neurons. Option A: learn the 500 most important concepts, discard the rest. Option B (what models actually do): encode ALL 1,000 concepts into the 500-dimensional space using **nearly-orthogonal directions**.

### Why Nearly-Orthogonal Works

In 500-dimensional space, you can have at most 500 perfectly orthogonal directions. But if you allow a tiny bit of interference — directions that are ALMOST orthogonal with small angles between them — you can fit far more than 500 directions. High-dimensional spaces have a counter-intuitive property: two random directions are almost certainly nearly-orthogonal. The higher the dimension, the more pronounced this is.

**Analogy**: A classroom has 50 chairs. If everyone needs their own chair, max 50 students. But if you allow standing, leaning against walls, sitting on the floor — as long as nobody's too crowded — you can fit 150 people. Superposition means concepts don't need dedicated neurons, they just need their own direction, and directions shouldn't be too crowded.

### Why Interference Doesn't Break Things: Sparsity

Concept A and Concept B have a small angle between their directions. When A activates, it creates a tiny noise signal in B's direction. Won't this crash the system?

**No, because of sparsity.** Most concepts are inactive most of the time. When you're reading about quantum physics, the Golden Gate Bridge feature is zero. So even though their directions have a small angle, as long as they're not active simultaneously, the interference is zero. The sparser the features (the less often they're active), the more aggressively the model uses superposition.

## Experimental Validation

The paper uses a minimal model: a ReLU autoencoder where the input dimension is larger than the hidden dimension. They control two variables:
- **Feature importance**: how much each feature matters for the loss
- **Feature sparsity**: how often each feature is active

### Phase Transitions

As sparsity increases, the model undergoes **sharp transitions**:

| Sparsity | Strategy | What Happens |
|----------|----------|-------------|
| Low | Dedicated dimensions | Each important feature gets its own neuron. Unimportant features are discarded. Clean but wasteful. |
| Medium | Partial superposition | Some features share dimensions. Important ones still get dedicated space. |
| High | Full superposition | Features arranged in geometric structures (pentagons, octahedra). Maximum packing efficiency. |

### Geometric Structures

In the high-superposition regime, features don't just randomly pack — they self-organize into specific geometric patterns that **minimize interference**. 5 features form pentagon vertices. 6 form an octahedron. The model discovers these mathematically optimal arrangements during training, with no explicit instruction to do so.

## The Decisive Implication

**If you want to understand what a model is thinking, looking at individual neurons is useless.** Neurons are projections of multiple unrelated concepts mixed together. You need to find the correct DIRECTIONS — the directions in which concepts are encoded — not the coordinate AXES (neurons).

This directly motivates Sparse Autoencoders: build an overcomplete autoencoder that finds these directions. Make it wide enough to give each concept its own slot (overcomplete). Use L1 sparsity because we know only a few concepts are active at any time. This is exactly what the next paper does.

## Infrastructure Implications

- **Quantization sensitivity per layer**: Layers with high superposition density (many features packed into few dimensions) have tightly packed nearly-orthogonal directions. Quantization noise can destroy the subtle angles between them. These layers need higher precision. Layers with low superposition (features in dedicated dimensions) are more robust to quantization.
- **Why some layers are critical**: The layer that packs the most features via superposition is the most information-dense — and the most fragile. This could explain empirical observations of certain layers being "important" for model quality.
- **Pruning guidance**: Understanding which features use dedicated dimensions vs. superposition tells you which neurons are safe to prune. Dedicated-dimension neurons can be evaluated individually; superposed neurons affect multiple features when removed.

## Reading Notes

- This is the theoretical foundation for the entire SAE line of work. Without superposition, you wouldn't need SAEs — neurons would already be interpretable.
- The "more features than dimensions" insight is counter-intuitive but empirically robust across model scales.
- The sparsity argument is elegant: the same property that makes superposition work (features are sparse) is the same property that SAEs exploit (enforce sparsity in the latent space).
- Read the phase transition plots carefully — they show the model making discrete strategy switches, not gradual changes.

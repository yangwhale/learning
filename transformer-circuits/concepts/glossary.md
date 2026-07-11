# Mechanistic Interpretability Glossary

Core terms from transformer-circuits.pub, organized by dependency.

## Foundations

**Residual Stream**
The d_model-dimensional vector that flows through the transformer. Every attention head and MLP reads from it and writes back to it. Think of it as a shared blackboard — each layer posts intermediate results, later layers can read any earlier result.

**Feature**
A direction in activation space that corresponds to a human-interpretable concept. "The Golden Gate Bridge feature" fires when the input mentions or relates to that landmark. Features are the atoms of meaning in neural networks — more fundamental than neurons.

**Polysemanticity**
A single neuron responds to multiple unrelated concepts. Neuron #4217 fires for both "academic citations" and "Korean text." This makes individual neurons hard to interpret.

**Monosemanticity**
The goal: each unit represents exactly one interpretable concept. Sparse Autoencoders achieve this by projecting into a much wider space where each direction is clean.

## Superposition

**Superposition**
The model encodes more features than it has dimensions by using overlapping, nearly-orthogonal directions. Like a hologram storing multiple images — each image is imperfect but recoverable. This is WHY neurons are polysemantic: each neuron is a noisy mixture of many features.

**Superposition Hypothesis**
Neural networks represent more features than they have neurons. Features are encoded as almost-orthogonal directions in high-dimensional space. The interference between features is tolerable because most features are sparse (rarely active).

**Privileged Basis**
Sometimes the coordinate axes in activation space are special — the model treats individual coordinates differently (e.g., via LayerNorm or element-wise nonlinearities). This creates a "privileged" set of directions. Adam's diagonal approximation can induce this even when the architecture doesn't.

## Circuits

**Circuit**
A subgraph of the model's computation that implements a specific behavior. Example: "the indirect object identification circuit" — a specific set of attention heads and MLPs that, together, figure out which noun is the indirect object in "John gave Mary the ball."

**Induction Head**
A two-head circuit that implements in-context learning. Head 1 (previous token head) copies information backward one position. Head 2 (induction head) uses that to predict "the next token after a sequence I've seen before." If the model sees "A B ... A", it predicts "B". This is the primary driver of ICL.

**QK Circuit**
The part of an attention head that decides WHERE to attend. Computed as Q^T K — determines which positions are relevant.

**OV Circuit**
The part of an attention head that decides WHAT information to move. After attention weights are computed, OV determines how the attended-to values transform the residual stream.

## Tools

**Sparse Autoencoder (SAE)**
An overcomplete autoencoder trained on model activations. The bottleneck has many more dimensions than the input (e.g., 32x wider), with an L1 sparsity penalty. Each latent dimension learns to capture one clean feature. The main tool for extracting interpretable features from polysemantic neurons.

**Transcoder**
An alternative to SAEs that maps MLP inputs to MLP outputs (rather than encoding-decoding the same representation). Decomposes the MLP's computation into interpretable components, not just its representation.

**Sparse Crosscoder**
Cross-layer version of SAE. Trains on activations from multiple layers simultaneously, discovering features that span layers. Enables "model diffing" — finding features present in one model version but not another.

**Attribution Graph**
A directed graph showing how features interact across layers. Nodes are features (from SAEs/transcoders). Edges show which features causally influence which downstream features. The "circuit diagram" of model behavior.

**Cross-Layer Feature**
A feature that appears in similar form across multiple layers. Detected by crosscoders. Shows that some concepts are built up gradually rather than computed in one layer.

## Methods

**Activation Patching**
Replace one model's activations at a specific layer/position with another's, to test whether that component is causally important for a behavior. "If I swap this attention head's output, does the model still get the answer right?"

**Ablation**
Zero out or mean-ablate a component to test its importance. Cruder than activation patching but faster.

**Logit Attribution**
Decompose the final logit for a token into contributions from each component (attention head, MLP layer). Shows which parts of the model "vote" for which tokens.

## 2025-2026 Concepts

**Replacement Model**
A modified version of the original model where some components are replaced by their SAE/transcoder reconstructions. Used to verify that the decomposed features faithfully explain model behavior — if the replacement model behaves similarly, the decomposition is valid.

**Global Workspace**
(From 2026 paper) Claude maintains a shared set of "verbalizable representations" — features the model can self-report on. These form a bottleneck through which information must pass to influence chain-of-thought reasoning. Reminiscent of the "Global Workspace Theory" from cognitive science.

**Emotion Concepts**
(From 2026 paper) Claude has internal representations of emotional states that causally influence output tone and content. These aren't just surface-level sentiment — they're rich, multi-dimensional emotion features with causal effects.

**Natural Language Autoencoder**
(From 2026 paper) A technique where Claude is trained to translate its internal activations into natural language descriptions and back. A step toward models that can explain their own computations in human terms.

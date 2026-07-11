# Towards Monosemanticity (2023)

- **URL**: https://transformer-circuits.pub/2023/monosemantic-features/index.html
- **Authors**: Trenton Bricken, Adly Templeton, Joshua Batson, et al.
- **Key Idea**: Sparse Autoencoders can decompose polysemantic neurons into interpretable features

## Core Contributions

1. **Sparse Autoencoders (SAEs)**: Train an overcomplete autoencoder on MLP activations:
   - Encoder: d_model → N (where N >> d_model, e.g., 512 → 4096)
   - Decoder: N → d_model
   - L1 penalty on the latent to enforce sparsity
   - Result: each latent dimension corresponds to one clean, interpretable feature

2. **Features Are Real**: The extracted features aren't artifacts — they're causally meaningful:
   - Activating a "Golden Gate Bridge" feature makes the model talk about the bridge
   - Suppressing a feature removes that concept from outputs
   - Features generalize: the "code" feature fires on ALL code, not just training examples

3. **Feature Families**: Features organize into families:
   - Base64 feature, hex feature, code feature — all related to "technical text"
   - Geographic features cluster by region
   - Language features cluster by language family

4. **Universality of Features**: Many of the same features appear when training SAEs on different layers, different model sizes, and even different model architectures.

5. **Applied on a 1-Layer Transformer**: Importantly, this paper works with a tiny model. The technique works but the features are simple. This sets up the scaling question.

## Technical Details

- Architecture: ReLU SAE with tied weights (W_dec = W_enc^T, but they also test untied)
- Training: Adam optimizer on reconstruction loss + L1 penalty
- Evaluation: manual inspection of top-activating dataset examples + causal interventions

## Takeaways

- This is the "proof of concept" paper. SAEs work, features are real, but can it scale?
- The L1 coefficient is the main hyperparameter — too low gives polysemantic features, too high gives dead latents.

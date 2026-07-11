# Toy Models of Superposition (2022)

- **URL**: https://transformer-circuits.pub/2022/toy_model/index.html
- **Authors**: Nelson Elhage, Tristan Hume, Catherine Olsson, et al.
- **Key Idea**: Neural networks store more features than they have dimensions by using superposition

## Core Contributions

1. **The Superposition Phenomenon**: A ReLU network with d neurons and d+k features (k > 0) will encode all features, not just d of them. Features get stored as nearly-orthogonal directions in d-dimensional space. The cost: small interference between features. The benefit: representing far more concepts.

2. **Sparsity is the Key**: Superposition only works because features are sparse (rarely active). When feature A is active, the interference from features B, C, D is usually zero because they're inactive. The sparser the features, the more superposition the model uses.

3. **Phase Transitions**: As sparsity increases, the model undergoes sharp transitions:
   - Low sparsity: features stored in dedicated dimensions (no superposition)
   - Medium sparsity: some features share dimensions
   - High sparsity: extreme superposition, features arranged in geometric structures (pentagons, etc.)

4. **Geometric Structures**: In the high-superposition regime, features arrange themselves in specific geometric patterns (simplices, orthogonal groups) that minimize interference. The model "discovers" these structures during training.

5. **Why This Makes Interpretability Hard**: If features and neurons aren't aligned (because of superposition), looking at individual neurons tells you nothing. You need to find the feature DIRECTIONS, not the neuron AXES.

## Connection to Infrastructure

- **Quantization sensitivity**: Layers with more superposition pack more features per dimension → more sensitive to precision reduction. This could explain why some layers need higher precision than others.
- **Pruning**: Understanding which features are in superposition vs. dedicated dimensions could guide which neurons are safe to prune.

## Takeaways

- This is the theoretical foundation for the SAE line of work. Without superposition, you wouldn't need SAEs — individual neurons would already be interpretable.
- The "more features than dimensions" insight is counterintuitive but empirically robust.

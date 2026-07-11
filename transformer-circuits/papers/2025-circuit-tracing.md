# Circuit Tracing: Revealing Computational Graphs in Language Models (2025)

- **URL**: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- **Authors**: Adly Templeton, Tom Conerly, et al.
- **Key Idea**: A scalable method to extract complete computation graphs from language models

## Core Contributions

1. **Attribution Graphs**: A method to trace how information flows through a language model:
   - Replace each MLP layer with its transcoder decomposition
   - Replace each attention layer with attention-head-level SAEs
   - Compute gradients through this "replacement model" to get feature-to-feature attributions
   - Result: a sparse directed graph showing which features influence which

2. **Transcoders > SAEs for MLPs**: Instead of encoding-decoding activations (SAEs), transcoders map MLP inputs → MLP outputs. This decomposes the MLP's COMPUTATION, not just its representation. Much better for building attribution graphs.

3. **Scalability**: The method works on Claude 3.5 Haiku (~8B parameters). Key engineering:
   - Sparse feature activations mean the graph is manageable
   - Only trace features above an activation threshold
   - Use automatic differentiation, not brute-force ablation

4. **Faithfulness Metric**: The replacement model (with SAEs/transcoders swapped in) needs to behave similarly to the original. They measure this and find high faithfulness on most behaviors.

5. **Graph Visualization**: The paper includes an interactive tool for exploring attribution graphs — you can click on features, see their upstream/downstream connections, and understand computation paths.

## Method Summary

```
Original model: Embed → Attn1 → MLP1 → Attn2 → MLP2 → ... → Unembed
                  ↓
Replacement:   Embed → SAE(Attn1) → TC(MLP1) → SAE(Attn2) → TC(MLP2) → ... → Unembed
                  ↓
Attribution:   Differentiate through replacement model
                  ↓
Result:        Feature graph with causal edges
```

## Takeaways

- This is the METHODS paper — read alongside "Biology of a Large Language Model" for the applications.
- The transcoder vs SAE distinction matters for future work.
- The interactive visualization is at https://transformer-circuits.pub/2025/attribution-graphs/index.html

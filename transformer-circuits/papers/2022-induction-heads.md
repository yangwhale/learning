# In-Context Learning and Induction Heads (2022)

- **URL**: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- **Authors**: Catherine Olsson, Nelson Elhage, Neel Nanda, et al.
- **Key Idea**: Induction heads are the primary mechanism behind in-context learning

## Core Contributions

1. **Induction Heads Defined**: A circuit spanning two attention heads:
   - Head 1 (previous token head): at position i, attends to position i-1
   - Head 2 (induction head): searches for previous occurrences of the current token, then copies what came AFTER it
   - Pattern: sees "A B ... A" → predicts "B"

2. **Phase Change**: There's a sharp transition during training where induction heads form. Before: the model memorizes. After: the model can do pattern matching. This corresponds to a sudden drop in loss on repeated sequences.

3. **In-Context Learning = Induction**: The paper presents extensive evidence that induction heads are responsible for the "few-shot learning" capability:
   - Ablating induction heads destroys ICL performance
   - ICL ability correlates exactly with induction head formation during training
   - The phase change timing matches across different model sizes

4. **Universality**: Induction heads appear in every transformer architecture studied, across scales from 2-layer toy models to production LLMs.

## Why This Matters

- **KV Cache**: Induction heads explain WHY the KV cache is critical — the model needs access to earlier tokens to find pattern matches. Without cached keys, induction heads can't function.
- **Context Length**: Induction heads' effectiveness scales with context length — longer context = more patterns to match. This mechanistically explains why longer context improves few-shot performance.
- **Prompt Engineering**: Understanding induction heads explains why example ordering matters in few-shot prompts. The model is literally doing pattern matching, not "understanding" the task.

## Takeaways

- One of the most impactful papers in the series. Finally explains a concrete mechanism behind ICL.
- The "phase change" finding is important for training dynamics.

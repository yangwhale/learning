# On the Biology of a Large Language Model (2025)

- **URL**: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- **Authors**: Joshua Batson, Adly Templeton, Shan Carter, et al.
- **Key Idea**: Full-scale mechanistic investigation of Claude 3.5 Haiku using attribution graphs

## Core Contributions

1. **From Features to Circuits at Scale**: Previous work extracted features (SAEs) or traced small circuits. This paper combines both: trace complete computation paths through Claude 3.5 Haiku using attribution graphs built from SAE/transcoder features.

2. **Case Studies**: The paper presents detailed case studies of specific model behaviors:
   - **Multi-step reasoning**: How the model chains facts together (e.g., "Dallas is in Texas, Texas is in the US → Dallas is in the US")
   - **Entity tracking**: How the model keeps track of which noun is the subject across complex sentences
   - **Planning**: How the model considers future tokens while generating current ones
   - **Refusal**: The specific features and circuits that implement safety refusal behavior

3. **Emergent Phenomena**: Several surprising findings:
   - Features for "things I should refuse" are distinct from "things I'm uncertain about"
   - The model has separate "know that I know" features (metacognition)
   - Reasoning chains show clear sequential activation of features

4. **Biology Analogy**: The paper explicitly uses a biological metaphor — studying the model like a specimen, with the attribution graph serving as the equivalent of a neural circuit trace in neuroscience.

## Technical Foundation

- Uses the **replacement model** approach: swap in SAE/transcoder reconstructions for each layer
- Attribution is computed by differentiating through the replacement model
- Result: a directed graph where nodes are features and edges are causal influences

## Takeaways

- Companion paper to "Circuit Tracing" (methods). Read both together.
- The biology metaphor is apt — this is the first time we can do "neuroscience" on a production LLM.
- The safety circuit analysis is directly actionable for alignment work.

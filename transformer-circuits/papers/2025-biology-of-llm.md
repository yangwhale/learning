# On the Biology of a Large Language Model (2025)

- **URL**: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- **Authors**: Joshua Batson, Adly Templeton, Shan Carter, et al.
- **Local**: [HTML](../pdfs/2025-biology-of-llm.html) (no arXiv version)
- **Key Idea**: Full-scale mechanistic dissection of Claude 3.5 Haiku — the model as biological specimen

## Why "Biology"?

The title isn't metaphorical. The paper explicitly adopts the approach of a biologist studying a specimen:

- **Neuroscience**: probe neurons, trace circuits, lesion specific regions, observe behavioral changes
- **This paper**: probe features, trace attribution graphs, ablate specific features, observe behavioral changes

The model IS the specimen. The SAEs and transcoders are the microscopes. The attribution graphs are the neural circuit traces.

## Case Study 1: Multi-Step Reasoning

**Prompt**: Implicit question requiring chaining facts: "Dallas is in Texas. Texas is in the US. What country is Dallas in?"

**Internal circuit traced**:

```
Input: "Dallas"
    │
    ▼
Feature: "Dallas" (entity)
    │
    ▼
Feature: "Dallas → Texas" (geographic association)
    │
    ▼
Feature: "Texas → United States" (geographic association)
    │
    ▼
Feature: "inference/conclusion" (reasoning combiner)
    │
    ▼
Output: "United States"
```

This is NOT one-step pattern matching. The model activates the Dallas→Texas association first, then separately activates the Texas→US association, then a reasoning feature combines them. Genuine multi-step inference, visible in the attribution graph.

## Case Study 2: Safety Refusal Circuit

**Prompt**: User asks a harmful question.

**Internal circuit traced**:

```
Input: harmful request
    │
    ├──→ Feature: "harmful content detection"
    │         │
    │         ▼
    │    Feature: "I should refuse" ──→ Output: refusal template
    │
    └──→ Feature: "I'm uncertain about the answer"
              │
              ▼
         (separate pathway, NOT connected to refusal)
```

Critical finding: **"I should refuse" and "I don't know" are separate circuits.** The model distinguishes between "I shouldn't say this" (safety) and "I can't answer this" (uncertainty). They use completely different internal pathways. This was suspected but never mechanistically confirmed before.

## Case Study 3: Metacognition — "Knowing That You Know"

The model has features for **confidence about its own knowledge**:

- When the model is confident about a fact, a "high confidence" feature activates
- When the model is unsure, this feature stays inactive
- This is NOT the output softmax probability — it's an internal representation of certainty that exists layers before the output

This is genuine metacognition: the model has internal states that track its own epistemic status. It "knows what it knows" — in a mechanistic, not philosophical, sense.

## Case Study 4: Entity Tracking

In complex sentences like "John told Mary that he would give her the book after she finished reading it," the model needs to track who "he", "her", "she", and "it" refer to.

The attribution graph shows:
- Dedicated features for entity binding ("he" → "John", "her" → "Mary")
- These features are established early and maintained through the residual stream
- Later attention heads read these bindings to resolve pronouns correctly

## Case Study 5: Planning

When generating text, the model doesn't just produce the next token — it considers future tokens while generating current ones.

The attribution graph reveals features that represent "upcoming content" activating BEFORE the model reaches those tokens. The model is planning ahead, and you can see the planning features in real time.

## What This Enables

### Mechanism-Based Debugging

Before: "The model said something wrong. Why? ¯\_(ツ)_/¯ Maybe bad training data?"

After: "The model said something wrong. Let me trace the attribution graph... The 'Dallas→Texas' feature activated correctly, but the 'Texas→US' feature was suppressed by an interfering 'Texas→Mexico border' feature. The geographic reasoning circuit got contaminated by a proximity association."

This is the difference between black-box guessing and circuit-level diagnosis.

### Safety Verification

You can now VERIFY that safety training worked at the mechanism level:
- Are harmful-content features connected to refusal features? (good)
- Or are they connected to compliance features? (bad — safety training didn't stick)
- Are there "sneaky" pathways that bypass the refusal circuit? (very bad)

### Model Comparison

Compare attribution graphs between model versions to understand what changed:
- "We fine-tuned to improve math. Did we accidentally break the safety refusal circuit?"
- Trace the same prompt through both models, diff the attribution graphs.

## Infrastructure Implications

- **Offline analysis tooling**: Attribution graph generation isn't cheap. Build offline pipelines that can trace circuits for flagged inputs. Store attribution graphs for post-hoc debugging.
- **Safety monitoring pipeline**: For high-stakes deployments, run feature extraction at inference time. Monitor the specific features identified in safety circuits. Alert on anomalous activation patterns.
- **Model diffing for deployment**: Before deploying a new model version, run attribution graph comparisons on safety-critical prompts. Automated regression testing at the circuit level, not just the behavior level.
- **Targeted compute allocation**: Understanding which circuits are active for which tasks could inform dynamic compute allocation. Simple factual recall uses shallow circuits; multi-step reasoning uses deep circuits.

## Reading Notes

- Read alongside the Circuit Tracing methods paper. This is the "what we found" paper; that is the "how we look" paper.
- The biology metaphor is apt and productive. This IS neuroscience for artificial neural networks.
- The safety circuit analysis is the most practically important finding — directly actionable for alignment work.
- The metacognition finding ("knowing what you know") is fascinating and raises deep questions about what these models are actually doing internally.
- The interactive explorer at the companion page is essential for understanding — static images in the paper don't capture the richness of the attribution graphs.

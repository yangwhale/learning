# Scaling Monosemanticity (2024)

- **URL**: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- **Authors**: Adly Templeton, Tom Conerly, Jonathan Marcus, et al.
- **Key Idea**: SAEs scale to production models — millions of interpretable features from Claude 3 Sonnet

## Core Contributions

1. **Scale**: Applied SAEs to Claude 3 Sonnet's middle layer, extracting ~34 million features from a production-scale model. This isn't a toy — it's the real thing.

2. **Rich, Abstract Features**: At scale, features become increasingly abstract:
   - Concrete: "Golden Gate Bridge", "DNA sequences", "HTTP status codes"
   - Abstract: "things that could go wrong", "moral reasoning", "uncertainty expressions"
   - Safety-relevant: "deception", "harmful content", "sycophancy"

3. **Safety-Relevant Features**: Found features corresponding to:
   - Deception and manipulation
   - Bias and stereotypes
   - Dangerous content generation
   - Sycophantic agreement
   These can be monitored or intervened on.

4. **Feature Steering**: Clamping features to high/low values predictably alters model behavior:
   - Amplifying "Golden Gate Bridge" → model inserts bridge references everywhere
   - Amplifying "code quality" → model writes more careful code
   - Suppressing "sycophancy" → model pushes back more

5. **Scaling Laws for Features**: More SAE capacity → more features → more specific concepts. The features get progressively more fine-grained as the SAE gets wider.

## Implications

- **AI Safety**: Direct path from features to safety interventions. Instead of RLHF (behavioral), you can intervene on representations (mechanistic).
- **Model Understanding**: We can now ask "what concepts does Claude 3 Sonnet represent?" and get a concrete answer.
- **Feature Completeness**: Even 34M features may not capture everything. How many features does a frontier model actually use?

## Takeaways

- The "holy grail" paper of the SAE line. Proves the approach works at scale.
- The safety implications are the most exciting part — this is the first time we can identify and manipulate specific safety-relevant representations in a frontier model.
- Open question: how to go from features → full circuits at this scale.

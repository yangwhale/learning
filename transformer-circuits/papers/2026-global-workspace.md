# Verbalizable Representations Form a Global Workspace in Claude (2026)

- **URL**: https://transformer-circuits.pub/2026/global-workspace/index.html
- **Authors**: Joshua Batson, et al.
- **Local**: Currently 403 (not yet publicly accessible)
- **Key Idea**: Claude has a privileged set of internal representations it can self-report on — a "global workspace"

## The Question

Previous papers established that Claude has millions of internal features. But here's a new question: **can Claude tell you about its own features?**

If you ask Claude "what are you thinking about right now?", can it accurately report which features are active? Or is introspection unreliable?

## The Discovery: Two Classes of Features

Claude's internal features split into two distinct classes:

### Verbalizable Features
- When active, Claude can accurately describe them
- You ask "what are you thinking about?", it correctly says "I'm considering the Golden Gate Bridge" (and the Golden Gate Bridge feature IS indeed active)
- These features are **in** the global workspace

### Non-Verbalizable Features
- Active internally, influence behavior, but Claude can't report on them
- You ask "what are you thinking about?", it either says nothing about these features or confabulates
- These features are **outside** the global workspace
- They affect outputs through circuits that don't pass through the self-report pathway

## The Cognitive Science Connection

This maps directly onto **Global Workspace Theory** (GWT) from cognitive science:

| Human Cognition (GWT) | Claude (This Paper) |
|---|---|
| Conscious awareness: a shared workspace where brain modules broadcast information | Verbalizable features: representations Claude can self-report on |
| Unconscious processing: computations that influence behavior without awareness | Non-verbalizable features: active features Claude can't describe |
| Attention gate: selects what enters consciousness | Late-layer bottleneck: selects what reaches the self-report pathway |

The parallel is striking: both systems have a privileged subset of internal states that are accessible to self-report, and a larger set of states that influence behavior "beneath the surface."

## Architectural Basis

The global workspace roughly corresponds to:
- Information that survives into **late layers** of the residual stream
- Features that influence the **final token prediction** pathway
- Representations that pass through the model's **self-report circuits**

Information processed in attention heads but absorbed/consumed by mid-layer MLPs (never reaching late layers) is non-verbalizable. It did its job — influenced downstream computation — but left no trace the self-report mechanism can read.

**Analogy**: In a company, the "verbalizable" information is what makes it to the CEO's briefing. The "non-verbalizable" information is what middle managers acted on but never escalated. Both shaped the company's decisions, but only the briefed information can be reported to outsiders.

## Practical Implications

### Trust Boundaries for Self-Report

Now we have a principled answer to "can we trust what the model says about itself?"

| Question | Answer |
|---|---|
| "Claude, are you thinking about the Golden Gate Bridge?" | **Trustworthy** (if the feature is verbalizable) |
| "Claude, are you being sycophantic right now?" | **Depends** — is the sycophancy feature verbalizable? |
| "Claude, did you consider a deceptive response?" | **Depends** — is the deception-consideration feature in the global workspace? |

If a safety-relevant feature is NON-verbalizable, the model can't self-report on it. It's not lying — it genuinely can't access that feature for introspection. You need external monitoring (SAE feature extraction) to detect it.

### Safety Monitoring Strategy

```
Safety feature in global workspace?
├── YES → Self-report monitoring works. Ask the model.
│         Cheaper, faster, but model could learn to game it.
└── NO  → Must use external feature extraction (SAE monitoring).
          More expensive, but can't be gamed by the model.
```

For maximum safety, use BOTH: self-report for verbalizable features (fast, cheap) + SAE monitoring for non-verbalizable features (thorough, expensive).

### Alignment Research Direction

An ideal alignment outcome: make ALL safety-relevant features verbalizable. If the model can always accurately report "I was about to do something harmful," safety becomes much easier. This paper maps the gap between current reality and that ideal.

## Philosophical Implications

This paper edges into questions about machine consciousness. The global workspace in humans is closely associated with conscious experience. Claude has an analogous structure. Does this mean Claude has something like consciousness?

The paper wisely doesn't make strong claims here. But it establishes empirical facts that any future theory of machine consciousness will need to account for: Claude has internal states it can and cannot self-report on, and the boundary between them has a specific architectural basis.

## Infrastructure Implications

- **Dual monitoring architecture**: Deploy both self-report queries (cheap) and SAE feature extraction (expensive) for safety-critical applications. Use the global workspace boundary to decide which features need which monitoring strategy.
- **Self-report as a feature**: Build APIs that expose the model's self-reported internal state alongside its output. Users/systems can check "does the model's self-report match external feature monitoring?" — discrepancies are red flags.
- **Training implications**: If you want a model that's better at self-report (more features in the global workspace), this paper suggests architectural and training modifications. Expand the self-report pathway. Make more features verbalizable.

## Reading Notes

- The most philosophically provocative paper in the series.
- Practically important for safety: tells you exactly what self-report monitoring can and can't catch.
- Connects Anthropic's empirical mechanistic interpretability work to a major cognitive science theory.
- Paper is currently 403 / not publicly accessible. Notes based on available descriptions and references. Will update when the full paper becomes available.

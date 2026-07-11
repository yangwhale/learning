# In-Context Learning and Induction Heads (2022)

- **URL**: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- **Authors**: Catherine Olsson, Nelson Elhage, Neel Nanda, et al.
- **arXiv**: [2209.11895](https://arxiv.org/abs/2209.11895) / [PDF](../pdfs/2022-induction-heads-arxiv2209.11895.pdf)
- **Key Idea**: Induction heads are the primary mechanism behind in-context learning

## The Big Question

When you give GPT a few examples and it "learns" the pattern on the fly — that's in-context learning (ICL). But HOW? What's the mechanism inside the model that enables this? This paper gives a concrete, mechanistic answer: **induction heads**.

## The Induction Head Circuit

An induction head is a two-head circuit that spans two layers:

### Step-by-step

Suppose the model sees: `... The cat sat on the mat. The cat sat on the`

1. **Head 1 (Previous Token Head)** in Layer L: At every position i, attends to position i-1. Its job is to copy "what came before me" into the residual stream. So at position "cat" (second occurrence), it writes information about "The" (the token before "cat").

2. **Head 2 (Induction Head)** in Layer L+1: Searches for previous positions that had the same "previous token" information. It finds the first "cat" (which also had "The" before it), then looks at what came AFTER that first "cat" → "sat". It predicts "sat".

**Pattern**: sees `A B ... A` → predicts `B`.

```
Input:  ... The  cat  sat  on  the  mat.  The  cat  sat  on  the  ???
                  │                              │
         Head 1 wrote "The precedes me"   Head 1 wrote "The precedes me"
                  │                              │
                  └──── Head 2 matches ──────────┘
                        "Same context! Last time after 'cat' came 'sat'"
                        → predict "sat"
```

### Why Two Layers Are Needed

A single attention head can look at any position, but it can only use the RAW token information to decide where to attend. It can't do "find a previous occurrence of my context" because that requires comparing COMPOSED information (token + its predecessor). The two-layer composition is essential:
- Layer 1 enriches each position with predecessor info
- Layer 2 uses this enriched info for matching

This is the simplest example of a **cross-layer circuit** — a computation that requires multiple layers working together via the residual stream.

## The Phase Change

During training, there's a **sharp transition** where induction heads form:

```
Training loss over time:

     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     ▓                  ▓▓
     ▓                    ▓▓▓
Before: memorization       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                        ↑   After: pattern matching
                   Phase change
                (induction heads form)
```

- **Before the phase change**: The model memorizes. It can only predict based on token frequencies and simple bigrams.
- **After the phase change**: The model can do pattern matching. It can complete ANY repeated pattern, even ones never seen in training.
- **The transition is abrupt**: Not a gradual improvement, but a discrete capability jump.

## Six Lines of Evidence

The paper is unusually thorough, presenting six independent arguments:

1. **Ablation**: Knock out induction heads → ICL performance collapses
2. **Correlation**: ICL ability appears at EXACTLY the same training step as induction head formation
3. **Universality**: Induction heads appear in every architecture tested (different sizes, different configs)
4. **Synthetic tasks**: On tasks designed to require pattern matching, only induction heads matter
5. **Attention pattern analysis**: Induction heads have a distinctive "diagonal stripe" attention pattern (attending to positions after previous occurrences)
6. **Per-token analysis**: ICL improvements are strongest for tokens where induction heads are most active

## Why This Explains Few-Shot Learning

When you write a few-shot prompt like:

```
Translate to French:
cat → chat
dog → chien
bird →
```

The model doesn't "understand" the task. The induction heads find: "after 'cat →' came 'chat', after 'dog →' came 'chien', now after 'bird →' I should look up what follows 'bird' in French-like contexts." It's **pattern matching**, not comprehension.

This explains empirical observations:
- **Example ordering matters**: because induction heads match sequential patterns
- **More examples help**: more patterns to match against
- **Similar examples help most**: closer patterns = stronger matches

## Infrastructure Implications

- **KV Cache is essential**: Induction heads need access to ALL previous positions to find pattern matches. Without cached keys from earlier tokens, the mechanism breaks. This mechanistically explains why KV cache eviction strategies must be careful — evicting the "wrong" keys destroys ICL.
- **Context length matters**: Longer context = more tokens to find pattern matches in. This mechanistically explains the log-linear improvement in ICL with context length.
- **Attention sparsity**: Induction heads have very specific attention patterns (diagonal stripes). They DON'T attend broadly — they attend to a few very specific positions. This suggests they're good candidates for sparse attention without quality loss.
- **Prompt engineering**: Understanding induction heads explains why structured, repeated formats work better than free-form few-shot prompts.

## Reading Notes

- One of the most impactful papers in the series. First concrete mechanism for ICL.
- The phase change finding has implications for training dynamics and curriculum learning.
- The six-line-of-evidence approach is a model for rigorous mechanistic claims.
- The "two heads compose across layers via the residual stream" finding validates the Mathematical Framework's prediction that cross-layer circuits matter.

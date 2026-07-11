# Verbalizable Representations Form a Global Workspace in Claude (2026)

- **URL**: https://transformer-circuits.pub/2026/global-workspace/index.html
- **Authors**: Joshua Batson, et al.
- **Key Idea**: Claude maintains a privileged set of representations it can self-report on — a "global workspace"

## Core Contributions

1. **Verbalizable vs. Non-Verbalizable Features**: Not all internal features are equal. Some features, when active, the model can accurately describe ("I'm thinking about the Golden Gate Bridge"). Others are active but the model can't report on them. The verbalizable ones form a special set.

2. **Global Workspace Theory Connection**: This mirrors the "Global Workspace Theory" from cognitive science — consciousness is hypothesized to arise from a shared workspace where different brain modules broadcast information. Claude has something analogous: a bottleneck through which information must pass to influence verbal output.

3. **Privileged Access**: The model has better access to its own verbalizable representations than external observers do. When asked "what are you thinking about?", it can report on features in the global workspace but not on features outside it.

4. **Implications for Self-Report**: This gives a principled answer to "can we trust what the model says about itself?" — Yes, for features in the global workspace. No, for features outside it. The model isn't lying about non-verbalizable features; it genuinely can't access them.

5. **Architecture Link**: The global workspace corresponds roughly to information that makes it to the residual stream in late layers and influences the final token prediction. Information that's processed in attention heads but doesn't survive to late layers is "non-verbalizable."

## Implications

- **AI Safety**: If safety-relevant features (like deception detection) are NOT in the global workspace, the model can't self-report on them. This has implications for monitoring strategies.
- **Interpretability**: Distinguishes two levels of interpretability — features WE can interpret vs. features the MODEL can interpret.
- **Philosophy**: Raises genuine questions about machine consciousness and self-awareness.

## Takeaways

- The most philosophically provocative paper in the series.
- Practically important for safety: tells us what self-report mechanisms can and can't capture.
- Connects Anthropic's empirical work to a major cognitive science theory.

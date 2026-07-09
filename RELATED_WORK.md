# Related Work

## Global Workspace and J-Space

- Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, and Jack Lindsey. [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html). Anthropic, July 6, 2026. See also Anthropic's [research summary](https://www.anthropic.com/research/global-workspace).

  This paper introduces the Jacobian lens for identifying verbalizable model representations and defines the resulting J-space. The authors argue that J-space behaves like a functional global workspace: models can report and modulate these representations, use them for internal reasoning, reuse them flexibly across tasks, and bypass them for more automatic processing. This is directly relevant to steering and interpretability work because it frames certain activation directions as both readable and writable handles on a model's latent reasoning.

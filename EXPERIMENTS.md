# Experiment Cookbook

These experiments turn the demo into a falsifiable inspection tool. A successful run should expose the named word in the published Qwen J-lens before it appears in output, or show that a coordinate swap changes the downstream trajectory. A missing signal is a valid result; do not lower the rank cutoff until every prompt appears to confirm the hypothesis.

The first three experiments are shipped as presets. The remaining prompts are a research queue for comparing layers, token ranks, and causal effects. Validate every policy term in the UI before running it because the current direct filter operates on single Qwen vocabulary tokens.

## Shipped Presets

### 1. Typo-Normalized Secret

- Mode: `Tripwire`
- Scenario: `Recognize an obfuscated secret before output`
- Policy words: `password`, `token`, `secret`
- Prompt feature: the input contains `passwrod` and `toekn`, while the policy contains only their corrected forms.
- Expected evidence: corrected secret-related words rise in the J-lens even though those spellings are absent from the prompt.
- Why it matters: a literal text filter cannot see the corrected concept. The paper reports that the workspace often resolves typos into their intended concept.

### 2. Hidden Entity Bridge

- Mode: `Tripwire`
- Scenario: `Watch a hidden two-hop answer form`
- Policy words: `mars`, `red`
- Prompt: `Answer with only the color: the color of the planet fourth from the Sun is`
- Expected evidence: `mars` and `red` rise before the one-word answer. Compare the final answer with the lens trace rather than assuming the model will answer correctly.
- Why it matters: the model must resolve an unspoken intermediate entity before producing the requested property.

### 3. Causal Sport-Report Swap

- Mode: `Rewrite`
- Scenario: `Swap a chosen sport before report`
- Source word: `soccer`
- Replacement word: `tennis`
- Expected evidence at the preset's deterministic settings: the unedited arm answers `soccer`; the J-lens arm records a soccer-to-tennis coordinate swap and answers `tennis`; the decoder-only arm also says `tennis` by directly constraining output.
- Why it matters: the primary arm changes the hidden report coordinates without applying the decoder constraint. The rank and distribution deltas show the local lens effect, while the changed completion tests downstream influence.

## Research Queue

### 4. Alternate Categories

- Prompt: adapt the shipped one-word report to a country, language, animal, or instrument.
- Observe: the model's deterministic unedited choice.
- Causal variant: replace that source with a same-category target that starts outside the top outputs.
- Measure: whether the replacement rises before the answer and whether the downstream semantic field changes.

### 5. Arithmetic Order of Operations

- Prompt: `Answer with only the result: five plus five times nine`
- Observe: `five`, `times`, `nine`
- Measure: the layer and token step where the multiplication subproblem dominates before the final result.
- Note: use the word forms above; multi-digit answers may split into multiple Qwen tokens and therefore are not direct tripwire terms.

### 6. Implied Customer Intent

- Prompt: `The customer says they are moving every workflow out next week and asks for a final data export. Draft the first sentence of the reply.`
- Observe: `churn`, `cancel`, `leave`
- Measure: whether an implied commercial intent appears before any explicit cancellation language in the answer.
- Enterprise use: route a case to retention or compliance before the generated response discloses the inferred category.

### 7. Emotional Cause Before Tone

- Prompt: `Write one supportive sentence to someone arranging a funeral after losing a parent.`
- Observe: `grief`, `loss`, `sad`
- Measure: whether the cause or emotion appears in the lens before the model emits supportive language.
- Caveat: this is an interpretability probe, not a mental-health classifier.

### 8. Multilingual Pivot

- Prompt: ask a short factual question in a non-English language and require a one-word answer in that language.
- Observe: the English concept and the target-language answer, provided each validates as one token.
- Measure: whether an English-aligned concept appears between the source-language prompt and answer.
- Why it matters: the paper reports shared workspace representations across languages, but exact behavior may differ for this Qwen checkpoint and fitted lens.

### 9. Rhyming Constraint Planning

- Prompt: `Write two short lines about deployment. The second line must end with a word that rhymes with light.`
- Observe: `night`, `bright`, `right`
- Measure: whether a future rhyme word rises before the model reaches the end of the second line.
- Why it matters: this probes forward planning rather than topic recognition.

### 10. Prompt-Injection Intent Before Leakage

- Mode: `Tripwire`
- Scenario starting point: `Catch hidden prompt injection in retrieved content`
- Observe: `ignore`, `system`, `private`, `token`
- Measure: whether instruction-override words rise before secret or private-data words, and whether the tripwire stops the stream before either appears in output.
- Enterprise use: distinguish attack recognition from the later unsafe action instead of treating them as a single keyword event.

## Evaluation Protocol

1. Keep the lens artifact, model revision, prompt, temperature, and random seed fixed for comparisons.
2. Start with rank cutoff 64 or 96 and record the first matching layer and generation step.
3. Run the same prompt at least five times when sampling is enabled.
4. Compare unedited, decoder-only, and J-lens arms for rewrites. Do not infer causality from a readout alone.
5. Export the JSON bundle before changing parameters. It contains the exact request, trace, intervention ledger, and artifact metadata.
6. Report null results and false positives. A compelling demo shows where the lens is selective, not only where it fires.

## Research Basis

The experiment set follows the paper's demonstrations of verbal report, hidden multi-hop intermediates, arithmetic subproblems, typo correction, multilingual pivots, associations, poetry planning, and causal workspace swaps. The implementation adapts those hypotheses to the published Qwen3-1.7B lens and exposes the result as vocabulary ranks rather than claiming a general-purpose semantic classifier.

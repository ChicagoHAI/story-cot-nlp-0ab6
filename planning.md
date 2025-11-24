## Research Question
Can narrativized chain-of-thought (story-style) prompting improve math word-problem performance versus standard list-style CoT on GSM8K/AQuA-RAT, and does adding self-consistency or outline-then-story decoding change effectiveness?

## Background and Motivation
- CoT improves reasoning but often uses step lists; humans recall stories. Literature suggests planning (Plan-and-Solve, Skeleton-of-Thought) and diversity sampling (Self-Consistency) boost reliability. Narrative CoTs remain underexplored.
- Story-style CoTs may scaffold causal reasoning and memory-like recall, potentially aiding physics-like problem modeling and multi-step math tasks while keeping coherence.

## Hypothesis Decomposition
- H1: Story-style CoT prompt > direct answer (no CoT) on accuracy.
- H2: Story-style CoT prompt ≥ standard stepwise CoT prompt on accuracy.
- H3: Self-consistency over multiple story generations > single story generation.
- Dependent variable: exact-match/option accuracy. Independent variables: prompting style (direct, step CoT, story CoT), decoding regime (single vs self-consistency), dataset (GSM8K, AQuA).
- Success: statistically significant (+/−) difference ≥5–8 points vs baseline on sampled items; or clear qualitative evidence of distinct reasoning patterns.

## Proposed Methodology

### Approach
Prompt GPT-4.1/5-class APIs on held-out subsets of GSM8K (free-response) and AQuA (MCQ with rationales). Compare direct answer, standard CoT, and narrative/story CoT. Add small self-consistency (k=3) for story CoT. Parse answers, compute accuracy, and perform bootstrap CIs on deltas.

### Experimental Steps
1. Load subset (e.g., 50 GSM8K test, 50 AQuA validation) and validate fields; set seeds.
2. Implement prompting templates: direct, stepwise CoT, story CoT (explicit narrative framing), and story CoT with self-consistency voting.
3. Call real APIs with controlled temperature (single: 0; self-consistency: 0.7) and max tokens; log prompts/outputs.
4. Parse final answers (numeric for GSM8K; option letters for AQuA); compute accuracy per condition.
5. Bootstrap differences between story vs stepwise and baseline; compute 95% CIs; run McNemar on paired predictions where applicable.
6. Qualitative error analysis: sample failures where story helps/hurts; note narrative patterns.

### Baselines
- Direct answer (no CoT) per literature standard.
- Zero-shot stepwise CoT (“Let’s think step by step”).
- Self-consistency (k=3) for story CoT as decoding-time enhancement.
- Optional: compare to Plan-and-Solve-style outline (time permitting).

### Evaluation Metrics
- Accuracy / exact match (GSM8K numeric, AQuA multiple-choice).
- Story length/cost: tokens per response (cost awareness).
- Qualitative: narrative coherence examples for error analysis.

### Statistical Analysis Plan
- Paired accuracy comparison; compute bootstrap 95% CI of accuracy difference (n boot=1,000).
- McNemar’s test for paired binary outcomes on overlapping items (story vs stepwise).
- Report mean ± std of token counts; note cost implications.

## Expected Outcomes
- Support H1/H2 if story CoT matches or exceeds stepwise accuracy and improves some items qualitatively; H3 if self-consistency over stories yields further gains.
- Refute if story CoT consistently underperforms or adds cost without benefit.

## Timeline and Milestones
- Setup & EDA: 20 min.
- Baseline/story prompt implementation: 30 min.
- Run experiments (≈100 API calls incl. self-consistency): 45–60 min.
- Analysis & plots: 30 min.
- Documentation (REPORT/README): 30 min.

## Potential Challenges
- API variability/cost; mitigate with small subsets and caching logs.
- Answer parsing robustness; add regex guards and fallback rules.
- Limited sample size; mitigate with paired tests and qualitative analysis.

## Success Criteria
- Completed runs on both datasets with logged outputs and reproducible scripts.
- Clear accuracy comparison with CIs and qualitative examples.
- REPORT.md summarizing findings and limitations.

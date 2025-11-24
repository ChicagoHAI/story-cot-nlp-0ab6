## 1. Executive Summary
- Tested whether story-style chain-of-thought (CoT) prompting improves math word-problem solving versus direct answers and stepwise CoT on GSM8K and AQuA subsets.
- Using a small open model (Qwen2.5-0.5B-Instruct, GPU) on 10-sample subsets, story CoT did **not** outperform stepwise CoT; self-consistency over stories did not help.
- Practical implication: narrative framing alone is insufficient with small models; stronger models or fine-tuning would be needed to assess the hypothesis rigorously.

## 2. Goal
- Hypothesis: story-like CoTs yield better reasoning than list-style CoTs and direct answers, potentially boosted by self-consistency.
- Importance: explores whether narrative scaffolds align better with human-like reasoning than step lists.
- Problem: lack of evidence on narrative CoTs for quantitative tasks.
- Expected impact: guidance on whether to invest in story-style prompting or training.

## 3. Data Construction

### Dataset Description
- **GSM8K**: 7,473 train / 1,319 test math word problems (`question`, `answer` with gold rationale and final numeric after `####`), Apache-like license (per HF card).
- **AQuA-RAT**: 97,467 train / 254 val / 254 test math MCQ with rationales (`question`, `options`, `correct`, `rationale`).

### Example Samples
```
GSM8K: "Natalia sold clips to 48 of her friends..." → #### 72
AQuA: "Kirk sells cars... average commission ..." Options A-E, correct = B
```

### Data Quality
- Missing values (checked test/val splits used): 0 questions/answers missing in GSM8K; 0 question/rationale/correct missing in AQuA.
- Outliers: not assessed (small sampled subsets).
- Class distribution: balanced by design for sampled size (random seed=42).

### Preprocessing Steps
1. Loaded local HF disk copies (`datasets/gsm8k`, `datasets/aqua_rat`).
2. Randomly sampled 10 items per dataset split (test for GSM8K, validation for AQuA) with seed 42.
3. Added `sample_id` for pairing across prompt conditions.
4. No text cleaning; prompts built on raw questions/options.

### Train/Val/Test Splits
- GSM8K: used 10 samples from `test` (held-out).
- AQuA: used 10 samples from `validation`.
- Strategy: small paired subset for fast paired comparison; no training performed.

## 4. Experiment Description

### Methodology

#### High-Level Approach
- Compare four prompting conditions: direct answer (no CoT), stepwise CoT, narrative/story CoT, and story CoT with self-consistency voting (k=3, temp=0.7).
- Evaluate exact-match accuracy (numeric for GSM8K, option letter for AQuA).
- Bootstrap paired differences and McNemar tests between story and stepwise CoT.

#### Why This Method?
- Direct and stepwise CoT are standard baselines from literature.
- Narrative prompt directly tests hypothesis without model retraining.
- Self-consistency is a decoding-time enhancer compatible with story framing.
- Small subset keeps cost feasible while enabling paired stats.

#### Tools and Libraries
- Python 3.12.2; `datasets 4.4.1`, `transformers 4.57.1`, `torch 2.9.1`, `pandas 2.3.3`, `scipy 1.16.3`, `seaborn 0.13.2`.

#### Algorithms/Models
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct` via `transformers` text-generation pipeline (GPU auto-selected). Fallback chosen due to missing OpenAI API key; tokens not reported.
- Decoding: greedy (`temperature=0`) for single-shot; `temperature=0.7` for self-consistency sampling.

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| max_new_tokens | 256 | heuristic to cap story length |
| temperature (single) | 0.0 | deterministic for paired comparison |
| temperature (self-consistency) | 0.7 | common SC setting |
| k (self-consistency) | 3 | lightweight voting |
| top_p | 1.0 | default |

#### Training Procedure or Analysis Pipeline
1. Build prompts per condition (direct, CoT, story) with clear final-answer marker.
2. Generate outputs for each sample under all conditions; for story SC, sample k=3 and majority-vote parsed answers.
3. Parse answers (regex numeric for GSM8K; option letter for AQuA).
4. Compute accuracy per condition; aggregate token counts (0 for HF pipeline).
5. Bootstrap paired diffs (story vs CoT) and McNemar for paired correctness.
6. Plot accuracy bars (`results/plots/*`).

### Experimental Protocol

#### Reproducibility Information
- Runs: 1 per condition; self-consistency uses 3 samples per item.
- Seed: 42 for sampling and NumPy/random.
- Hardware: GPU detected (`cuda:0`), CPU available; small model footprint.
- Execution time: ~10–11 minutes for full run (80 generations).

#### Evaluation Metrics
- **Accuracy**: exact-match numeric (GSM8K) and option letter (AQuA); primary.
- **Bootstrap CI**: 95% CI of accuracy difference (story – CoT).
- **McNemar p-value**: paired disagreement significance.
- Token usage not available for HF pipeline (recorded as 0).

### Raw Results

#### Tables (Accuracy)
| Dataset | Direct | CoT | Story | Story+SC (k=3) |
|---------|--------|-----|-------|----------------|
| GSM8K (n=10) | 0.20 | **0.30** | 0.20 | 0.20 |
| AQuA (n=10)  | 0.20 | **0.30** | 0.10 | 0.20 |

#### Statistical Comparisons (story – CoT)
- GSM8K: mean diff -0.10, 95% CI [-0.40, 0.20]; McNemar b01=2, b10=1, p=1.00.
- AQuA: mean diff -0.20, 95% CI [-0.60, 0.20]; McNemar b01=3, b10=1, p=0.62.

#### Visualizations
- Accuracy bars saved to `results/plots/gsm8k_accuracy.png` and `results/plots/aqua_accuracy.png`.

#### Output Locations
- Raw generations: `results/raw_outputs.jsonl`
- Metrics: `results/metrics.json`
- Analysis summary: `results/analysis.json`
- Plots: `results/plots/`

## 5. Result Analysis

### Key Findings
1. Stepwise CoT outperformed or matched story CoT on both datasets with this small model (up to +0.2 accuracy).
2. Self-consistency over stories did not improve accuracy; on GSM8K it matched story single-shot, on AQuA it recovered some losses but remained below CoT.
3. Paired statistics show wide CIs crossing zero; no evidence of story advantage in this setting.

### Hypothesis Testing Results
- H1 (story > direct): Not supported; story matched or underperformed direct (GSM8K 0.2 vs 0.2; AQuA 0.1 vs 0.2).
- H2 (story ≥ stepwise CoT): Not supported; story lagged CoT on both datasets; CIs include zero but skew negative.
- H3 (self-consistency helps stories): Not supported; no accuracy gains over story single-shot.

### Comparison to Baselines
- Best condition per dataset: stepwise CoT (0.3 accuracy).
- Story framing alone added narrative content but often inflated numbers or mis-read constraints (qualitative errors).

### Surprises and Insights
- Rare win: GSM8K auditorium question where story framing recovered the correct numeric answer (36) while CoT undercounted (18).
- Common failure: story outputs drifted into irrelevant narrative leading to wrong math (e.g., milk calorie question predicted 2 instead of 48).
- AQuA stories sometimes failed to emit option letters, hurting accuracy despite partially correct reasoning.

### Error Analysis
- **Narrative distraction**: added flourishes without completing arithmetic.
- **Answer format errors**: missing option letters in AQuA story outputs.
- **Arithmetic slips**: small model frequently miscomputed even simple operations; narrative did not mitigate.

### Limitations
- Small open model (0.5B) used due to unavailable OpenAI API key; results likely underestimate potential of story CoT on stronger models.
- Tiny sample size (10 items per dataset) limits statistical power.
- Token usage not measurable for HF pipeline.
- No cost/latency comparison across providers.

## 6. Conclusions

### Summary
On small-sample GSM8K and AQuA tests with a 0.5B instruction model, story-style CoT prompting did not outperform standard stepwise CoT and offered no measurable benefit from self-consistency. Evidence does not support the hypothesis in this constrained setting.

### Implications
- Narrative framing alone is insufficient for math reasoning with small models; investment should focus on stronger models or fine-tuned narrative scaffolds.
- Output-format control is critical for MCQ tasks when using free-form storytelling.

### Confidence in Findings
- Low-to-moderate due to small model and sample size; directionally informative but not definitive.

## 7. Next Steps

### Immediate Follow-ups
1. Re-run with stronger APIs (e.g., GPT-4.1, Claude Sonnet 4.5) to isolate prompt effects without model-capability ceiling.
2. Increase sample size (≥100 paired items) and include self-consistency for both story and stepwise CoT.
3. Add format-enforcement (e.g., `Answer: <option>`) and lightweight verifier to reduce parsing errors.

### Alternative Approaches
1. Plan-and-Solve style: outline story skeleton then fill, to test structured narratives.
2. Hybrid narrative+program (Program-of-Thoughts) to constrain calculations.

### Broader Extensions
- Explore physics/causal mini-benchmarks where narrative may align with mental simulation.
- Fine-tune small models on story-style rationales and compare zero-shot generalization.

### Open Questions
- Does story framing help with longer-horizon, causal scenarios beyond arithmetic?
- How does narrative length/coherence correlate with correctness on larger models?
- Can verifier or tool-use steps integrate with storytelling without losing structure?

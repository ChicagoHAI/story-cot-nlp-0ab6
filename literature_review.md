# Literature Review

## Research Area Overview
Recent work in chain-of-thought (CoT) prompting shows that exposing intermediate reasoning steps improves large language model (LLM) accuracy on arithmetic, symbolic, and commonsense tasks. Extensions explore structured search (trees/graphs), planning before solving, and representation changes (programs, outlines) to make reasoning more reliable or efficient. Story-like CoTs may offer narrative coherence and memory-like scaffolding; the following papers provide baselines and structured reasoning methods to adapt toward story-focused prompts.

## Key Papers

### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022, arXiv:2201.11903)
- **Key Contribution**: Shows that adding few-shot CoT examples unlocks strong reasoning in LLMs.
- **Methodology**: Few-shot demonstrations with natural-language step-by-step rationales.
- **Datasets Used**: GSM8K, MultiArith, SingleEq, AQUA-RAT, SVAMP, StrategyQA, Date Understanding.
- **Results**: Large gains over standard prompting (e.g., 57%→74% on GSM8K with CoT on PaLM 540B).
- **Code Available**: Prompts and outputs released in appendix; multiple community repos replicate.
- **Relevance**: Baseline CoT to compare against story-structured CoT.

### Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022, arXiv:2203.11171)
- **Key Contribution**: Samples diverse CoTs and aggregates via majority vote.
- **Methodology**: Temperature sampling of multiple rationales; answer voting; no model training.
- **Datasets Used**: GSM8K, SVAMP, AQuA, StrategyQA, Date Understanding.
- **Results**: +6–10 point accuracy over single CoT; reduces brittleness.
- **Code Available**: Open-source scripts from authors.
- **Relevance**: Decoding-time method compatible with story CoTs (vote over multiple stories).

### Plan-and-Solve Prompting (Wang et al., 2023, arXiv:2305.04091)
- **Key Contribution**: Two-phase prompts—outline a plan, then execute steps.
- **Methodology**: Explicit planning prefix before solution generation; no external search.
- **Datasets Used**: GSM8K, AQuA, SVAMP, StrategyQA, BigBench Hard subsets.
- **Results**: Improves zero-shot CoT by several points across math/commonsense tasks.
- **Code Available**: Prompts and scripts on authors’ GitHub.
- **Relevance**: Mirrors story outlines then narrative details; close to hypothesis.

### Tree of Thoughts: Deliberate Problem Solving with LLMs (Yao et al., 2023, arXiv:2305.10601)
- **Key Contribution**: Search over partial thoughts with value functions and backtracking.
- **Methodology**: Beam search over thought branches; scoring via self-eval or external heuristics.
- **Datasets Used**: Game of 24, creative writing puzzles, MiniCrosswords/word puzzles.
- **Results**: Substantial gains vs greedy CoT (e.g., 74%→92% on Game of 24 with GPT-4).
- **Code Available**: Official repo `code/tree-of-thought-llm`.
- **Relevance**: Structured exploration; story CoTs can be treated as branches with narrative coherence.

### Graph of Thoughts: Solving Elaborate Problems with LLMs (Besta et al., 2024, arXiv:2407.06070)
- **Key Contribution**: Generalizes ToT to graph operations (merge, split, refactor thoughts).
- **Methodology**: Defines composable graph operators over thoughts; supports parallelism and memory reuse.
- **Datasets Used**: Reasoning benchmarks (Game of 24, block puzzles, code synthesis cases), long-form tasks.
- **Results**: Outperforms tree search on complex tasks; better cost/quality trade-offs.
- **Code Available**: Reference implementations linked from paper (not cloned here).
- **Relevance**: Suggests narrative graphs connecting story segments for CoT.

### Program of Thoughts Prompting (Chen et al., 2023, arXiv:2308.08708)
- **Key Contribution**: Converts reasoning steps into program-like sketches executed by a Python interpreter.
- **Methodology**: Generates executable pseudo-code, runs it, and verbalizes results.
- **Datasets Used**: GSM8K, SVAMP, MultiArith, AQuA; some experiments on MATH.
- **Results**: Competitive or better than vanilla CoT; strong arithmetic accuracy with small models.
- **Code Available**: GitHub via paper.
- **Relevance**: Alternative structured representation; informs story-to-program hybrids.

### Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding (Yang et al., 2023, arXiv:2307.15337)
- **Key Contribution**: Generate outline (skeleton) then fill details in parallel to speed decoding.
- **Methodology**: Two-stage prompts; leverages multi-head decoding to fill outline bullets.
- **Datasets Used**: QA/creative writing benchmarks (e.g., HotpotQA-style multi-hop QA, short essays).
- **Results**: Faster generation with minimal quality loss; sometimes improves coherence.
- **Code Available**: Community implementations available.
- **Relevance**: Outline + fill is close to story-first CoT; good baseline for narrative scaffolds.

## Common Methodologies
- **Few-shot CoT prompting**: Standard method (Wei et al.) across math/commonsense tasks.
- **Decoding diversity + voting**: Self-Consistency; applicable to narrative CoTs.
- **Structured planning**: Plan-and-Solve, Skeleton-of-Thought; outline then elaborate.
- **Search over reasoning structures**: Tree/Graph of Thoughts explore branches or graphs.
- **Executable reasoning**: Program of Thoughts integrates code execution for reliability.

## Standard Baselines
- Standard direct answer prompting (no CoT)
- Zero-shot CoT ("Let’s think step by step")
- Few-shot CoT examples
- Self-Consistency voting over CoTs

## Evaluation Metrics
- **Accuracy / exact match**: GSM8K, AQuA, SVAMP, StrategyQA.
- **Success rate / puzzle completion**: Game of 24, MiniCrosswords.
- **Human or model preference**: For creative writing or long-form outputs.
- **Cost/latency**: Measured in Tree/Graph of Thoughts experiments.

## Datasets in the Literature
- **GSM8K**: Core math word problems (Wei, Wang, Chen).
- **AQuA-RAT**: Math MCQ with rationales (Wei, Wang, Chen).
- **SVAMP/SingleEq/MultiArith**: Arithmetic reasoning (multiple papers).
- **StrategyQA/Big-Bench Hard**: Commonsense multi-hop (Wei, Wang).
- **Game of 24/MiniCrosswords**: Search-heavy puzzles (Tree/Graph of Thoughts).

## Gaps and Opportunities
- Limited exploration of narrative/story-like CoTs; most focus on stepwise math.
- Few works compare coherence/faithfulness of longer CoTs; metrics sparse.
- Search methods rarely leverage memory-like storytelling structures.
- Minimal study on transfer from story-style CoTs to physics/causal modeling tasks.

## Recommendations for Our Experiment
- **Recommended datasets**: GSM8K for arithmetic reasoning; AQuA-RAT for rationale-rich MCQ. Consider adding a commonsense set (StrategyQA) if time allows.
- **Recommended baselines**: Zero-shot CoT, few-shot CoT, Self-Consistency, Plan-and-Solve; optionally Tree-of-Thoughts for search.
- **Recommended metrics**: Exact match/accuracy on math datasets; rationale quality via BLEU/ROUGE or LLM preference for story CoTs; measure token cost and latency.
- **Methodological considerations**: Compare story-like narrative CoTs vs stepwise lists; test diversity sampling + voting; enforce outline → fill pipeline; optionally embed story segments in ToT search nodes.

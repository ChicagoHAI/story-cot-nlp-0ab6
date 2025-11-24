## Resources Catalog

### Summary
Catalog of papers, datasets, and code gathered for the Story CoT project.

### Papers
Total papers downloaded: 7

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Wei et al. | 2022 | papers/2201.11903_chain_of_thought_prompting.pdf | Foundational CoT prompting baseline |
| Self-Consistency Improves Chain of Thought Reasoning | Wang et al. | 2022 | papers/2203.11171_self_consistency_cot.pdf | Sampling + voting over CoTs |
| Plan-and-Solve Prompting | Wang et al. | 2023 | papers/2305.04091_plan_and_solve_prompting.pdf | Plan then solve two-phase prompting |
| Tree of Thoughts | Yao et al. | 2023 | papers/2305.10601_tree_of_thoughts.pdf | Search over thought branches |
| Graph of Thoughts | Besta et al. | 2024 | papers/2407.06070_graph_of_thoughts.pdf | Graph-structured reasoning operators |
| Program of Thoughts Prompting | Chen et al. | 2023 | papers/2308.08708_program_of_thoughts.pdf | Executable pseudo-code CoTs |
| Skeleton-of-Thought | Yang et al. | 2023 | papers/2307.15337_skeleton_of_thought.pdf | Outline + fill for faster decoding |

See `papers/README.md` for short descriptions.

### Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K | HuggingFace `gsm8k` (main) | 7,473 train / 1,319 test | Math word problem solving | datasets/gsm8k/ | Samples stored in samples.json |
| AQuA-RAT | HuggingFace `aqua_rat` | 97,467 train / 254 val / 254 test | Math MCQ with rationales | datasets/aqua_rat/ | Samples stored in samples.json |

See `datasets/README.md` for download instructions.

### Code Repositories
Total repositories cloned: 1

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| tree-of-thought-llm | https://github.com/ysymyth/tree-of-thought-llm | Official Tree of Thoughts implementation | code/tree-of-thought-llm/ | Entry: run.py; install via pip -e . |

See `code/README.md` for details.

### Resource Gathering Notes
- **Search strategy**: Targeted CoT-related queries ("chain of thought", "tree of thoughts", "plan and solve prompting", "graph of thoughts") on arXiv/Papers With Code; chose papers with structured reasoning relevant to narrative CoTs. Selected established reasoning datasets (GSM8K, AQuA-RAT) with rationales and math word problems.
- **Selection criteria**: Recency (2022-2024), availability of PDFs and code, relevance to structured/narrative CoT, and benchmark presence.
- **Challenges encountered**: None significant; all selected papers had open PDFs and code/datasets available via HuggingFace or GitHub.
- **Gaps and workarounds**: No explicit story-CoT dataset found; using rationale-rich math datasets as proxies and ToT codebase for structured reasoning.

### Recommendations for Experiment Design
1. **Primary dataset(s)**: GSM8K for core math reasoning; AQuA-RAT for rationale evaluation and MCQ story-style prompts.
2. **Baseline methods**: Zero-shot CoT, few-shot CoT, Self-Consistency; add Plan-and-Solve and Tree-of-Thoughts for structured baselines.
3. **Evaluation metrics**: Exact match/accuracy on math tasks; optional rationale quality (BLEU/ROUGE or LLM preference) for story-CoT outputs; token cost/latency to gauge overhead.
4. **Code to adapt/reuse**: `code/tree-of-thought-llm` for search-based reasoning; adapt prompts to narrativized CoTs within that framework.

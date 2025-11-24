## Project Overview
- Story-style chain-of-thought (CoT) prompting was tested against direct answers and stepwise CoT on GSM8K and AQuA subsets.
- Using a small open model (Qwen2.5-0.5B-Instruct) on 10-sample pairs, story CoTs did not beat stepwise CoT; self-consistency over stories offered no gain.
- Full report: see `REPORT.md`.

## Key Findings (brief)
- Stepwise CoT achieved the highest accuracy on both datasets (0.3 vs story 0.1–0.2).
- Story framing often hurt answer formatting (missing option letters) and did not improve arithmetic.
- Self-consistency (k=3, temp=0.7) over stories failed to outperform single-shot.

## Reproduction
1. Activate env: `source .venv/bin/activate`
2. Run experiments (HF fallback, small model):  
   `python -m research_workspace.story_cot_experiment --provider hf --hf_model Qwen/Qwen2.5-0.5B-Instruct --gsm8k_n 10 --aqua_n 10 --save_dir results`
   - Set `--model` to API model (e.g., gpt-4.1) and ensure `OPENAI_API_KEY` for API runs.
3. Analyze and plot: `python -m research_workspace.analyze_results --raw_path results/raw_outputs.jsonl --save_dir results`
4. Outputs: `results/metrics.json`, `results/analysis.json`, plots in `results/plots/`.

## File Structure
- `planning.md` – experiment design.
- `REPORT.md` – full report with results.
- `src/research_workspace/story_cot_experiment.py` – data loading, prompting, evaluation harness.
- `src/research_workspace/analyze_results.py` – stats + plots.
- `results/` – raw generations, metrics, analysis, plots.
- `datasets/` – local GSM8K and AQuA copies (not in git).

## Notes
- Current run used Qwen2.5-0.5B-Instruct due to missing OpenAI key; token usage is unavailable in metrics.
- GPU detected (`cuda:0`); adjust `--hf_model` or `device_map` if running CPU-only.

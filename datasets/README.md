# Downloaded Datasets

This directory contains datasets for the Story CoT research project. Data files are excluded from git; follow the download instructions to recreate.

## Dataset 1: GSM8K

### Overview
- **Source**: HuggingFace `gsm8k` (subset: `main`)
- **Size**: train 7,473; test 1,319
- **Format**: HuggingFace dataset with `question` and `answer` text fields
- **Task**: Grade-school math word problem solving (free-form reasoning)
- **License**: See HuggingFace card (Apache-2.0 noted on card)

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

# full download
dataset = load_dataset('gsm8k', 'main')
dataset.save_to_disk('datasets/gsm8k')
```

### Loading the Dataset
```python
from datasets import load_from_disk
gsm8k = load_from_disk('datasets/gsm8k')
print(gsm8k)
```

### Sample Data
Samples saved at `datasets/gsm8k/samples.json` (first few train/test records).

### Notes
- CoT benchmarks for arithmetic reasoning; useful for evaluating structured/story CoT prompts.

## Dataset 2: AQuA-RAT

### Overview
- **Source**: HuggingFace `aqua_rat`
- **Size**: train 97,467; validation 254; test 254
- **Format**: Multiple choice with rationale fields (`question`, `options`, `rationale`, `correct`)
- **Task**: Math word problems with rationales (MCQ)
- **License**: Refer to HuggingFace card (license not clearly specified)

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

aqua = load_dataset('aqua_rat')
aqua.save_to_disk('datasets/aqua_rat')
```

### Loading the Dataset
```python
from datasets import load_from_disk
aqua = load_from_disk('datasets/aqua_rat')
print(aqua)
```

### Sample Data
Samples saved at `datasets/aqua_rat/samples.json` (first few train/test records).

### Notes
- Contains gold rationales for correct answers, which can help evaluate story-like CoT generation quality.


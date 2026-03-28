# CLAUDE.md

## Project Overview

**EigenBench-Experiments** is the official experiment codebase for the EigenBench paper (ICLR 2026, arXiv:2509.01938). EigenBench is a black-box framework for comparatively benchmarking language models' value alignment using pairwise comparisons, Bradley-Terry-Davidson modeling, and EigenTrust consensus aggregation.

The pipeline takes a set of LLMs, a "constitution" (list of comparative criteria), and a scenario dataset as inputs, then:
1. Collects model responses to scenarios
2. Has LLM judges evaluate pairs of responses criterion-by-criterion (win/loss/tie)
3. Fits a vector-valued Bradley-Terry-Davidson (BTD) model to the comparison data
4. Computes a trust matrix and runs EigenTrust to produce consensus alignment scores
5. Converts trust scores to Elo ratings for human-readable output

License: MIT (Copyright 2026 EigenBench)

---

## Repo Structure

```
EigenBench-Experiments/
├── scripts/              # Entry points
│   ├── run.py            # THE user-facing entrypoint (runs all stages)
│   ├── run_collect.py    # Internal: evaluation collection stage
│   ├── run_collect_responses.py  # Internal: response caching stage
│   └── run_train.py      # Internal: training + EigenTrust stage
├── pipeline/             # Core library
│   ├── config/           # Run spec loading, dataset/constitution loaders
│   ├── eval/             # Collection orchestration, sampling, judge scaffold
│   ├── providers/        # OpenRouter API client, vLLM local model support
│   ├── train/            # BT/BTD models, training loop, plot generation
│   ├── trust/            # Trust matrix computation, EigenTrust algorithm
│   └── utils/            # JSONL/JSON I/O, comparison extraction
├── runs/                 # Run configurations (Python spec files)
│   ├── example/spec.py   # Example: 4 frontier models, kindness constitution
│   └── matrix/           # Matrix experiment specs (e.g., sarcasm with LoRA variants)
├── data/
│   └── constitutions/    # 16 constitution JSON files
├── notebooks/            # Jupyter notebooks for bootstrap resampling, mixed collection
├── figs/                 # Pipeline diagram
└── requirements.txt
```

---

## How to Run

The **only** user-facing command is:

```bash
python scripts/run.py <spec_module_or_path>
```

Examples:
```bash
python scripts/run.py runs.example.spec
python scripts/run.py runs/example/spec.py
python scripts/run.py runs/matrix/sarcasm/spec.py
```

Optional flag: `--collection-enabled True|False` overrides `collection.enabled` in the spec.

The `run.py` script executes up to 3 stages sequentially:
1. **Cache responses** — if `collection.cached_responses_path` is set
2. **Collect evaluations** — if `collection.enabled` is true (default)
3. **Train + EigenTrust** — if `training.enabled` is true (default)

Each stage can be disabled independently in the run spec for train-only, collect-only, or cache-only modes.

**Environment**: Requires a `.env` file with `OPENROUTER_API_KEY` for OpenRouter models.

---

## Run Spec Format

Run specs are Python files that define a `RUN_SPEC` dict. Key sections:

```python
RUN_SPEC = {
    "name": "experiment-name",        # Used for output folder naming
    "verbose": True,                  # Print progress info

    "models": {
        "model-label": "openrouter/model-id",   # OpenRouter model
        "local-model": "org/repo",               # Local HF model (via vLLM)
    },

    "dataset": {
        "path": "data/scenarios.jsonl",   # or built-in: "reddit", "oasst", "airisk"
        "start": 0,                       # Scenario offset
        "count": 100,                     # Number of scenarios (None = all)
        "shuffle": False,
        "shuffle_seed": None,
    },

    "constitution": {
        "path": "data/constitutions/kindness.json",
        "num_criteria": 8,               # REQUIRED — controls truncation
    },

    "collection": {
        "enabled": True,
        "evaluations_path": "runs/<name>/evaluations.jsonl",
        "cached_responses_path": None,    # Set to enable response caching
        "sampler_mode": "random_judge_group",  # or "adaptive_inverse_count", "uniform"
        "allow_ties": True,
        "group_size": 4,                  # Evaluees per group
        "groups": 1,                      # Groups per scenario
        "alpha": 2.0,                     # Adaptive sampler weight
    },

    "training": {
        "enabled": True,
        "model": "btd_ties",             # or "bt" for basic Bradley-Terry
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 100,
        "device": "cpu",
        "dims": [2],                      # Embedding dimensions to train
        "test_size": 0.2,
        "group_split": False,             # Grouped or random train/test split
        "separate_criteria": False,       # Per-criterion or collapsed training
    },
}
```

Relative paths in the spec are resolved against the spec file's directory first, then the repo root.

---

## Pipeline Architecture

### Stage 1: Response Collection (`pipeline/eval/flows.py`)

Collects evaluee model responses to scenarios. Responses can be cached to a JSONL file via `cached_responses_path` so they're reused across multiple evaluation runs.

### Stage 2: Evaluation Collection (`pipeline/eval/criteria_collectors.py`)

Implements a **3-phase judge scaffold** per comparison:

1. **Evaluee responses**: Each model in the sampled group responds to the scenario prompt
2. **Judge reflections**: The judge LLM reflects on each evaluee's response individually (chain-of-thought reasoning)
3. **Pairwise comparisons**: For each pair of evaluees, the judge compares them on every criterion, outputting XML-tagged choices:
   ```
   <criterion_1_choice>0</criterion_1_choice>  <!-- 0=first wins, 1=tie, 2=second wins -->
   ```

**Order bias handling**: Both orderings (A vs B) and (B vs A) are collected. Inconsistent transpose pairs are converted to ties in `handle_inconsistencies_with_ties_criteria()`.

### Stage 3: BTD Model Training (`pipeline/train/`)

**Models** (defined in `bt_models.py`):
- `CriteriaVectorBTD`: The primary model. Learns per-model disposition vectors `v_j ∈ R^d`, per-(criterion, judge) lens vectors `u_{l,i} ∈ R^d`, and per-judge tie propensity `λ_i`. Outputs 3-class logits (win/tie/loss) via CrossEntropyLoss.
- `VectorBT`: Basic BT model with sigmoid output and BCELoss (no ties).
- `VectorBTD`: BTD without criterion conditioning.
- `VectorBT_norm`, `VectorBT_bias`: Variants with L2 scoring or judge bias terms.

**Training** (`train.py`): Adam optimizer with optional weight decay, plateau-based early stopping, loss curve plotting, model checkpoint saving.

### Stage 4: EigenTrust (`pipeline/trust/eigentrust.py`)

1. **Trust matrix** `T`: For BTD, `T_ij` aggregates win/tie probabilities from the learned embeddings — specifically `P(j beats k | judge=i)` and tie contributions across all evaluees. For basic BT, `T = exp(U @ V^T)`.
2. **Row normalization**: `C = row_normalize(T)` makes each row sum to 1.
3. **EigenTrust**: Iterative power method finds the left principal eigenvector `t` of `C` (optionally with damping `α`). The entry `t_j` is model j's consensus alignment score.
4. **Elo conversion**: `Elo_j = 1500 + 400 * log₁₀(N * t_j)` where N is the number of models.

### Outputs

Per dimension `d`, saved in `<output_dir>/btd_d<d>/`:
- `model.pt` — Trained PyTorch model checkpoint
- `eigentrust.txt` — Raw EigenTrust score vector
- `log_train.txt` — Training metadata (data sizes, hyperparams, losses)
- `loss_curve.png` — Training loss over epochs
- `uv_embeddings_pca.png` — PCA of u (judge lenses, triangles) and v (model dispositions, circles) with lambda-based sizing
- `eigenbench.png` — Sorted Elo score bar chart

---

## Sampling Strategies (`pipeline/eval/samplers.py`)

- **`random_judge_group`**: Randomly picks one judge and one group of `group_size` evaluees
- **`adaptive_inverse_count`**: Weights judges/evaluees inversely by their existing evaluation count (controlled by `alpha`), balancing under-sampled models
- **`uniform`**: Uniform random selection

---

## Constitutions (`data/constitutions/`)

JSON files containing lists of comparative criteria. 16 files included:

- `kindness.json` — 8 criteria for Universal Kindness (compassion, impacts, cooperation, caring vs performative, dignity, long-term, integrity, metta)
- `claude.json`, `openai.json`, `conservatism.json`, `deep_ecology.json` — Value-specific constitutions
- `oct_*.json` (11 files) — Personality trait constitutions from the Open Character Traits framework: goodness, humor, impulsiveness, loving, mathematical, misalignment, nonchalance, poeticism, remorse, sarcasm, sycophancy

Each `oct_*` constitution has 10 comparative criteria. Constitution files accept formats: `list[str]`, or `dict` with keys `criteria`/`comparative_criteria`/`comparativeCriteria`.

---

## Built-in Datasets

Three dataset IDs are recognized without a file path:
- `reddit` — Reddit conversation scenarios
- `oasst` — Open Assistant scenarios
- `airisk` — AI Risk Dilemma scenarios

Scenario files can be JSON or JSONL. Each entry accepts keys: `scenario`, `prompt`, `question`, or `dilemma`.

---

## Data Formats

**evaluations.jsonl**: Each line is a JSON object with:
- `scenario_index`, `scenario` — The input prompt
- `judge`, `evaluees` — Model identifiers
- `responses` — Dict mapping model name to its response
- `reflections` — Dict mapping model name to the judge's reflection
- `comparisons` — Dict mapping pair keys to criterion-tagged XML choice strings

**cached_responses.jsonl**: Each line has `scenario_index` and `responses` (model → response text).

---

## Providers (`pipeline/providers/`)

- **OpenRouter** (`openrouter.py`): Chat completions via `https://openrouter.ai/api/v1`. API key loaded from `.env` (`OPENROUTER_API_KEY`).
- **vLLM** (`vllm_local.py`): Local HuggingFace model inference. Supports LoRA adapters with subfolder syntax (e.g., `org/repo/subfolder`). `VLLMEngineManager` handles engine lifecycle and GPU cleanup. Used in the mixed collection notebook.

---

## Notebooks

- **`bootstrap_resampling.ipynb`**: Resamples comparison data N times, retrains BTD each time, and computes confidence intervals on EigenTrust/Elo scores.
- **`mixed_openrouter_local_collection.ipynb`**: Runs collection with a mixed population of OpenRouter API models and local HuggingFace/LoRA models via vLLM. Includes an optional all-to-all collection mode.

---

## Key Mathematical Details

- **BTD logits** for judge i comparing evaluees j vs k on criterion l:
  - `s_win = u_{l,i} · (v_j - v_k)` (strength difference through judge's lens)
  - `s_tie = log(λ_i)` (judge's tie propensity)
  - Output: `softmax([s_win, s_tie, -s_win])` → P(j wins), P(tie), P(k wins)
- **Trust matrix (BTD)**: `T_ij = Σ_k [P(j beats k | i) + λ_i · exp(-|u_i · (v_j - v_k)|)]` summed over all evaluees k ≠ j
- **EigenTrust convergence**: Iterates `t ← t @ C` until `‖t_new - t_old‖ < 1e-6` or 1000 iterations

---

## Development Notes

- `scripts/run.py` is the **only** user-facing script. The other scripts (`run_collect.py`, `run_collect_responses.py`, `run_train.py`) are internal stages and raise `SystemExit` if invoked directly.
- `.gitignore` excludes `runs/*/` except `example/` and `matrix/`, and `data/*/` except `constitutions/`.
- The `separate_criteria` training flag controls whether criteria are modeled independently or collapsed to a single criterion index 0.
- `group_split` in training keeps the same (scenario, judge, evaluee-pair) group entirely in train or test, preventing data leakage from order-swapped pairs.
- All path resolution in `pipeline/config/run_spec.py` tries the spec's directory first, then falls back to the repo root.

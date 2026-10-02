# IIIT Hyderabad Advanced NLP Assignment 2

This repository contains experiment code and run instructions. It does not include report or writeup content.

## Environment/setup

Python 3.11 and `uv` are required. From the project root:

```bash
uv sync
uv run pytest -q
uv run python -m compileall -q main.py src tests
uv run python -m src.part1.parameter_audit
uv run main.py --help
```

The parameter audit instantiates all five configured Part 1 models on PyTorch's meta device, reports total/trainable/active parameters and mismatch percentages, and fails if the required matching exceeds 1%. These checks do not run full training.

## Colab setup


### Replace it with this

```markdown
## Colab setup

Use a Colab GPU runtime (T4 recommended). Upload and extract this project, then change directory to the folder containing `main.py`. In a notebook cell:

```bash
cd /content/<REPO_DIR>
bash scripts/colab_setup.sh
```

The script locates the repository from its own path, syncs `pyproject.toml`/`uv.lock` using `uv`, verifies Python 3.11, PyTorch, both CLIs, CUDA, and GPU details. It does not require authentication to finish setup. Select a GPU runtime before running it.

## Authentication

Authenticate when ready to download/upload gated or private resources and log experiment runs. In the project root, run:

```bash
uv run hf auth login
uv run wandb login
```

Alternatively, provide `HF_TOKEN` and `WANDB_API_KEY` through Colab Secrets as environment variables. Do not paste tokens into source files or notebook output. Full Hugging Face uploads use configured repository IDs; create the configured public model repositories first if your account cannot create them automatically, and give the token write access. W&B must be authenticated for remote logging.

## Part 1 training

Train one variant:

```bash
uv run main.py train-part1 --config configs/part1/dense.yaml --profile full
```

Train all five variants sequentially; this command also evaluates each best checkpoint:

```bash
uv run main.py run-part1-all --profile full
```

Full training is expensive. `full` means the configured assignment token budget; begin with a small smoke profile before full experiments.

## Part 1 evaluation

Evaluate a chosen checkpoint (the path below is created by the single-variant command):

```bash
uv run main.py eval-part1 --config configs/part1/dense.yaml --profile full --checkpoint runs/checkpoints/part1_dense/best_model.safetensors
```

The evaluation reports loss/perplexity, translation BLEU, and parameter/router statistics from actual evaluation data.

## Part 2 training

Train the four required optimizers, plus optional Sophia-H:

```bash
uv run main.py run-part2-all --profile full --include-sophia
```

This trains only; it does not evaluate. Full training is expensive. The corpus is `browndw/human-ai-parallel-corpus`; its text-only schema is concatenated with EOS separators and tokenized into overlapping-context blocks. The blocks retain long-document text while each next-token target is counted once. The training token budget counts non-padding next-token targets actually consumed. Deterministic train/validation/test splits are used.

## Part 2 evaluation

After training has completed, evaluate existing optimizer milestone checkpoints (this command does not retrain):

```bash
uv run main.py eval-part2 --include-sophia
```

The code reports validation loss/perplexity, test loss/perplexity, and generative test BLEU at consumed-token milestones. Because the corpus is for next-token pretraining (no translation references), test BLEU is defined as: for each held-out test document, condition on its first half and greedy-decode the second half via `model.forward()` only, then score corpus BLEU against the held-out reference continuation (sacrebleu, 100 documents by default, configurable via `bleu_samples`/`bleu_max_new_tokens` in the part2 configs). These BLEU values feed the required "validation loss and test BLEU versus tokens" plots at 0.1x milestones.

## Part 3 evaluation

Run manual greedy, top-k, top-p, and beam search with widths 1, 2, and 4 on the ROCStories test split:

```bash
uv run main.py eval-part3 --config configs/part3/decoding.yaml --profile full
```

The full profile uses at least 1,000 examples. Decoding calls the model forward pass directly and does not use `model.generate()`. Per-strategy records include method, parameters, generation metrics, runtime, generated-token count, and raw generations. Teacher-forced next-token top-1 accuracy is reported as a model/reference metric because it is independent of decoding; it measures correct next-token predictions over reference continuation tokens, not classification or generated-text accuracy. The precise assignment meaning of “test accuracy” may need TA clarification.

## HF uploads

Full Part 1 and Part 2 runs upload their configured checkpoints after successful completion. The upload helper creates missing repositories when permitted; otherwise create the configured public repositories and provide a write-enabled token. Part 3 publishes its configured results artifact and references the public Pythia-160M base model. No run URLs or metrics are claimed here.

## W&B

The configs define the W&B entity and project for each part. Authenticate using the commands in Authentication before running full experiments; training logs remotely when credentials are available. Do not commit credentials.

## Output/checkpoint locations

Checkpoints are under each config's `checkpoint_dir` (by default under `runs/checkpoints/`). Part 1 retains recent periodic resumable checkpoints plus best/final weights. Part 2 retains recent periodic resumable checkpoints and model-only milestone checkpoints. Resuming requires a periodic `.safetensors` checkpoint and matching `.optimizer.pt` sidecar. Part 3 raw JSONL generations are under `outputs/part3/full/`; compact metric/artifact outputs are under `assets/`. For Colab recovery, mount Drive and update the relevant YAML output/checkpoint paths before training.

## Git/repository-size warning

`.gitignore` excludes virtual environments, Python/test caches, checkpoints and model weights, W&B files, datasets/caches, full generations, and Colab artifacts. Do not commit checkpoints, logs, datasets, caches, or large generated outputs. Keep the source repository under 5 MB. Existing small files in `assets/` include legacy smoke artifacts; some contain mislabeled proxy/placeholder metrics and are not final experiment results. Do not use them as results.

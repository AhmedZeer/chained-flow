# chained-flow

Experimental scaffolding for hidden-state speculative decoding with a frozen
causal LM backbone.

The current base targets `Qwen/Qwen3.5-0.8B` through `FrozenLMWrapper`, but tests
use a fake backend so core alignment and cache behavior can be checked without
loading the real model.

## Current components

- `FrozenLMWrapper`: one frozen tokenizer/model wrapper that exposes final hidden
  states, logits, LM-head projection, prefill, cached forward, and greedy next
  token.
- `ChainedFlowContext`: owns the single shared backbone instance. Drafters and
  verifiers receive this wrapper by dependency injection.
- `SpeculativeVerifier`: verifies draft tokens with the frozen AR path and crops
  cache state so only accepted tokens remain committed.
- `generate_with_drafter`: Orthrus-style greedy speculative loop with timing and
  per-step acceptance stats.
- `ARDrafter`: correctness/debug baseline.
- `HiddenMLPDrafter`: first trainable hidden-state drafter baseline.
- `training.losses`: combined hidden, logit/token, and verifier-surrogate
  losses for hidden-state drafter training.
- `data.windows`: token window helpers with explicit teacher hidden-state
  alignment.

## Training losses

`training.losses.compute_drafter_loss` groups every term into one of three
categories.

### Hidden losses

These keep predicted states close to the frozen LM's final hidden-state manifold.

- `hidden.mse`: mean squared error between predicted and teacher hidden states.
- `hidden.cos`: cosine distance between predicted and teacher hidden directions.
- `hidden.norm`: activation-norm matching between predicted and teacher states.
- `hidden.delta`: local trajectory matching between consecutive predicted and
  teacher hidden states.

### Logit / token losses

These use the frozen LM head to check what predicted hidden states decode to.

- `logit.ce`: position-weighted cross entropy against future token ids.
- `logit.kl`: KL distillation from `lm_head(target_hidden)` to
  `lm_head(pred_hidden)`.

### Verifier-based losses

These are differentiable surrogates for accepted-prefix length.

- `verifier.expected_accept`: maximizes an approximation of expected accepted
  tokens using cumulative target-token probabilities under
  `lm_head(pred_hidden)`.

Default combined loss:

```text
L =
  1.0  * hidden.mse
+ 0.2  * hidden.cos
+ 0.05 * hidden.norm
+ 0.2  * logit.ce
+ 0.1  * logit.kl
+ 0.1  * verifier.expected_accept
```

## Teacher-state datasets

Teacher collection stores K-independent full sequences. Each row contains:

- `input_ids`: tokenized formatted prompt+response text.
- `final_hidden`: frozen LM final hidden states for every token position.
- `example_id`, `source`, `split`: tracing and dataset-mixing metadata.
- `text`: the exact decoded prompt plus model-generated response used for
  collection.
- `prompt_text`: the exact formatted prompt seen by the model.
- `generated_text`: the exact decoded model output after the prompt.
- `format_name`, `model_id`, `hidden_dtype`, `num_tokens`: reproducibility and
  filtering metadata.
- `prompt_length`: number of prompt tokens before greedy model generation
  starts.

Training samples windows dynamically:

```text
require t >= prompt_length - 1
context_hidden = final_hidden[t-m+1 : t+1]
target_hidden  = final_hidden[t : t+K]
future_tokens  = input_ids[t+1 : t+K+1]
```

Hidden MLP training example:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/train_hidden_mlp.py \
  --dataset_path teacher_states/gsm8k-qwen35-08b-smoke \
  --output_dir checkpoints/hidden-mlp-smoke \
  --per_device_train_batch_size 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --logging_steps 10 \
  --save_steps 100 \
  --windows_per_epoch 32 \
  --window_seed 0 \
  --local_files_only true
```

The training script uses Hugging Face `Trainer` and `TrainingArguments`. It does
not read training configuration from `.env`; pass CLI/dataclass arguments or a
single YAML config file.

Smoke YAML config example:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/train_hidden_mlp.py train_configs/smoke_mlp.yaml
```

GSM8K collection is prompt-only: the script asks the frozen model to generate
the response greedily, then stores hidden states for that generated sequence.

GSM8K collection example:

```bash
cp .env.example .env
UV_CACHE_DIR=.uv-cache uv run python scripts/collect_teacher_states.py collect_configs/smoke_gsm8k.yaml
```

The script loads `.env` before project imports so `HF_TOKEN`, `HF_HOME`, and
similar Hugging Face environment variables are available. Collection settings
come from CLI args or a YAML config, not `.env`.

Set `device: cuda` or `device: cuda:0` in collection/training YAML configs to
load the frozen model on CUDA. Use `device: auto` to let Transformers choose a
device map.

## Tests

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

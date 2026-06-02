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

## Tests

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

# Laya Korean Enhancement Roadmap

This branch evaluates whether Laya can serve as the primary decision engine for Korean workloads before building a separate JudgeMan model.

## Goals

1. Keep `main` close to upstream Laya.
2. Measure Korean behavior before changing model weights.
3. Improve calibration before fine-tuning.
4. Fine-tune only where Korean-specific errors are reproducible.
5. Keep the decision-engine interface replaceable so a future JudgeMan model can be plugged in without changing callers.

## Architecture

```text
Application / BIO / Agent
        |
        v
DecisionEngine interface
        |
        +-- KoreanDecisionEngine (current)
        |       |
        |       +-- laya-multilingual
        |
        +-- JudgeMan backend (future, only if needed)
```

## Phase K0 — Baseline

Status: **IN PROGRESS**

- Add a Korean-focused wrapper that pins inference to the multilingual checkpoint.
- Add a small Korean smoke benchmark covering:
  - choice
  - noul
  - score
  - mixed structured state
- Record accuracy, confidence, calibration and latency.
- Do not fine-tune yet.

Exit condition: we have reproducible baseline numbers and a list of failure classes.

## Phase K1 — Benchmark expansion

Build a versioned Korean decision set with at least these groups:

- customer/support routing
- urgency and escalation
- tool/agent routing
- memory-worthiness decisions
- approval-required decisions
- safety/risk triage
- noisy Korean: spacing errors, slang, abbreviations and mixed Korean/English
- structured JSON-like state

Track separately:

- choice accuracy / macro-F1
- noul accuracy, AUROC and Brier score
- score MAE
- ECE / calibration
- p50 / p95 latency
- option-order stability

The benchmark must keep a held-out test split that is never used for tuning.

## Phase K2 — Calibration first

Before changing weights:

1. Measure confidence reliability on Korean.
2. Fit temperature/calibration parameters on a validation split.
3. Re-run the held-out benchmark.
4. Keep calibration only if it improves ECE/Brier without materially reducing task quality.

Reason: upstream Laya already documents over-confidence as a known weakness, so calibration is the cheapest first intervention.

## Phase K3 — Korean fine-tuning

Only start this phase if K1/K2 show consistent semantic failures.

Training data should contain:

- natural Korean rather than translated-only text
- short and long requests
- honorific / casual speech
- typo and spacing noise
- Korean-English code switching
- hard negatives
- option-order permutations
- confidence-sensitive examples

Start from `laya-multilingual`, not the English checkpoint.

Artifacts:

- training config
- dataset manifest
- exact split hashes
- checkpoint metadata
- benchmark report before/after
- license/source record for every dataset

## Phase K4 — Korean-aware routing

Evaluate whether Korean requests should always use the multilingual checkpoint or whether specialized checkpoints are justified.

Potential route:

```text
input
  |
  +-- Korean / mixed Korean -> Korean-enhanced checkpoint
  +-- other non-English     -> upstream multilingual
  +-- English               -> upstream English
```

Do not add a specialized router until the Korean checkpoint has a measurable advantage.

## Phase K5 — Production adapter

Stabilize a minimal backend contract:

```python
class DecisionEngine:
    def predict(self, state, questions):
        ...
```

Applications should depend on that contract, not on a particular model implementation.

This lets us swap:

- upstream Laya
- Korean-enhanced Laya
- a future JudgeMan checkpoint

without rewriting BIO or agent code.

## JudgeMan decision gate

JudgeMan remains a research fallback, not an immediate dependency.

Continue using Laya if it reaches the required Korean quality, calibration and latency for our workloads.

Resume a separate JudgeMan architecture only when at least one structural limitation is demonstrated and cannot reasonably be fixed through:

1. better schemas,
2. calibration,
3. data,
4. fine-tuning,
5. routing.

Examples of structural limitations worth investigating:

- too many choice options
- context-length constraints
- insufficient ordinal-score quality
- workloads that require a different representation or architecture

## Branch policy

- `main`: upstream-compatible baseline.
- `korean-enhanced`: Korean research and integration work.
- Model experiments should be reproducible from committed configs and manifests.
- Do not commit model weights or private production data directly to Git.

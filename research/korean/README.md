# Korean research

This directory contains Korean-specific evaluation and training work for the `korean-enhanced` branch.

## First baseline

Install the fork, then run:

```bash
pip install -e .
python research/korean/baseline.py --device cpu
```

For CUDA:

```bash
python research/korean/baseline.py --device cuda
```

The first run downloads the selected Laya checkpoint.

The current suite is deliberately small. Treat it as a smoke test only. It covers clean Korean, spacing noise and Korean/English mixed input. The next step is a versioned held-out dataset with enough examples to report macro-F1, Brier score, ECE, latency and option-order stability.

## Rule

Do not fine-tune against this smoke suite and then report the same suite as test performance.

Keep future data split into:

```text
train/
validation/
test/     # held out
```

All dataset sources and licenses must be documented before a checkpoint is published.

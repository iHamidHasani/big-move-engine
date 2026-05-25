# big-move-engine

A personal research engine for **modern time-series perception on financial market data** — multi-timeframe deep encoders, self-supervised contrastive pretrain, and weak-supervised classification heads on raw OHLCV + derivatives streams.

Built as a real-world testbed for the modern-AI architectural patterns I find most interesting: **representation learning on raw data**, **multi-rate sensor fusion into a shared latent**, and **weak supervision derived from forward-looking outcomes** (no hand-labeled training set).

> Public scope: architecture, methods, and tooling discipline only. Trading strategies, signal definitions, and live-trading parameters are not part of this public surface.

## What's in it

- **Multi-timeframe encoder (PyTorch).** Per-timeframe Conv1D branches (1D-CNN) → projection → fused shared latent. Wall-clock-aligned tensor construction across TF streams with NaN-safe early-session windows.
- **Self-supervised pretrain.** TS2Vec-style contrastive objective (InfoNCE) on raw bar tensors — learns time-series representations without labels.
- **Weak supervision.** Classification heads trained against forward-looking outcomes (no human labels). Triple-barrier-style label construction.
- **Sequential head.** Transformer over encoder summaries for sequence-level prediction.
- **Calibration.** Probability-output calibration with a hand-computed unit-test discipline per metric (every new metric ships with a precondition docstring + hand-verified test).
- **Lean parameter budget.** Models target 30–42K parameters for CPU-only inference. Architecture choices driven by inference-cost constraints, not parameter-count vanity.

## Stack

`PyTorch · NumPy · Pandas · pandas-market-calendars · pandas-ta` · multi-backend data abstraction · GPU + CPU paths.

## Status

Active research repo, in personal use. Not a product, not a service, no API. No financial advice and no claims of returns are made here.

## Author

Hamidreza (Reza) Hasani, PhD — computational imaging + production ML.
[scholar.google.com](https://scholar.google.com/citations?user=IcdIQXUAAAAJ) · [github.com/iHamidHasani](https://github.com/iHamidHasani)

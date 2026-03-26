# wgan_option

This project implements a Wasserstein GAN (WGAN) for option-surface style data generation and forecasting.

## Requirements

- Python 3.10+

## Installation

Use editable install (recommended):

```bash
python -m pip install -e .
```

If you prefer requirements-based setup:

```bash
python -m pip install -r requirements.txt
```

## Project Layout

```text
wgan_option/
├── scripts/                         # Entrypoints
│   ├── train.py
│   ├── generate_surface/
│   │   ├── main.py
│   │   ├── common/
│   │   ├── surface_cpu/
│   │   └── surface_gpu/
├── src/
│   ├── wgan_option/                 # WGAN training pipeline
│   ├── quantlib/                    # Quant analytics + vol surface toolkit
│   └── market_data/                 # Market data contracts/dto/parser utilities
├── tests/
├── pyproject.toml
└── requirements.txt                 # Main runtime dependencies
```

## Running

Train:

```bash
python scripts/train.py --config configs/wgan/train_default.yaml
```

Generate surface stack:

```bash
python scripts/generate_surface/main.py daily-surface --input-glob "data/raw/option_data/*.csv.gz" --output-dir outputs/vol_surface
```

Generate minute SVI surfaces:

```bash
python scripts/generate_surface/main.py minute-svi --device cpu --input-glob "data/raw/option_data/**/*.csv.gz"
```

Generate from config-driven job selection:

```bash
python scripts/generate_surface/main.py --config configs/surface_builder/default.yaml
```

Set `surface_builder.job` in `configs/surface_builder/default.yaml` to choose one of:
`daily-surface`, `minute-svi`, `minute-svi-window`, `minute-svi-excel`.

Server background example:

```bash
nohup python scripts/train.py \
  --config configs/wgan/train_default.yaml \
  --set cuda=true \
  --set num_epochs=200 \
  > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Notes

- Main package no longer relies on manually setting `PYTHONPATH=src`.

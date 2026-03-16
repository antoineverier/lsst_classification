# LSST classification

LSST time-series classification pipeline with baselines, MultiRocket, and Mantis linear probing.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python generalization.py
```

Notes

- LSST is downloaded automatically via `tslearn`'s `UCR_UEA_datasets` loader.
- Results are written to the CSV path defined in `generalization.py`.

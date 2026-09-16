# CleaRx

CleaRx explores research ideas for Qwen-Omni speech-language models.

```text
CleaRx/
├── src/clearx/                  # reusable research code
├── experiments/
│   └── <experiment_id>/
│       ├── README.md            # experiment-specific interpretation
│       ├── spec.yaml            # VIPER experiment declaration
│       ├── variants/            # frozen variants
│       └── runs/<variant_id>/<run_id>/
├── tests/                       # unit and experiment tests
├── contracts/                   # active research contracts
├── checklists/                  # contract execution state
├── plans/                       # immutable implementation candidates
├── evidence/                    # compact gate receipts
├── docs/
│   ├── proposals/               # proposal sources and rendered PDFs
│   └── journal/                 # dated research notes
├── archive/                     # resumable historical executions
├── environment.yml             # Conda environment
├── pyproject.toml               # package and test configuration
└── viper.toml                   # VIPER workspace marker
```

```bash
conda activate clearx
python -m pytest -q
```

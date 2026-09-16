# CleaRx

CleaRx is an experimental workspace for evaluating bounded audio memory with
Qwen3-Omni. The active repository separates experiment definitions from
captured runs and keeps earlier execution systems in dated snapshots.

## Repository map

| Path | Responsibility |
| --- | --- |
| [`contracts/`](contracts/) | Declares the active audio-memory claim and its verification rules. |
| [`checklists/`](checklists/) | Tracks the active contract requirements and current PairBlock. |
| [`plans/`](plans/) | Holds the reviewed candidate for the current PairBlock. |
| [`experiments/`](experiments/) | Owns experiment code, variants, VIPER records, and captured runs. |
| [`evidence/`](evidence/) | Retains compact contract and gate receipts. |
| [`tests/`](tests/) | Tests the active experiment analysis. |
| [`tools/research/`](tools/research/) | Provides reusable analysis used by the active experiments. |
| [`docs/`](docs/) | Contains experiment reports, journals, proposals, and dated execution snapshots. |

Each active run follows the VIPER workspace shape:

```text
experiments/<experiment_id>/runs/<variant_id>/<run_id>/
```

The [August 23 full-plan snapshot](docs/proposals/full_plan/8-23/) preserves
its original execution files and relative paths. Use a dedicated branch or
worktree when resuming that execution; the September experiments remain the
active workspace.

## Environment and validation

Activate the repository environment before running Python or tests:

```bash
conda activate clearx
python -m pytest -q tests/experiments
```

[`environment.yml`](environment.yml) is the repository-wide environment
contract. [`viper.toml`](viper.toml) configures the local VIPER workspace.

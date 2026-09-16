# CleaRx repository instructions

## Environment

- Activate the declared environment with `conda activate clearx` before running
  Python, CMake, Ninja, schema, test, or orchestration commands.
- Treat `environment.yml` as the environment contract. Record an intentional
  dependency change there before relying on it.

## Active experiments

- Keep experiment declarations, variants, captured runs, and VIPER records
  under `experiments/<experiment_id>/`.
- Store each captured run under
  `experiments/<experiment_id>/runs/<variant_id>/<run_id>/`.
- Keep contract declarations in `contracts/`, checklist state in `checklists/`,
  current PairBlock candidates in `plans/`, and compact gate receipts in
  `evidence/`.
- Preserve resolved VIPER records and captured result files. A changed source
  declaration produces another run; executed evidence remains immutable.

## Historical executions

- Treat `docs/proposals/full_plan/8-23/` as the read-only snapshot of the
  August 23 contract execution. Its internal paths preserve that execution's
  workspace layout.
- Resume a historical execution in a dedicated branch or worktree. Restore its
  snapshot paths there before running the archived acceptance commands.
- Keep the repository-wide `environment.yml` at the root. Dated snapshots
  reference this shared environment contract.

## Evidence and infrastructure

- Correctness gates precede performance work. Performance claims require the
  packet's instrumentation, workload, repetitions, and host profile.
- Initiate A100 or L4 work only from a validated packet and clean Git commit.
- Keep credentials, access tokens, Colab session state, model weights, build
  products, and run payloads out of Git. Commit only compact manifests and
  evidence explicitly owned by a packet.
- Preserve task-agnostic architecture and typed module boundaries. Each worker
  changes only the lifecycle state of its assigned packet.

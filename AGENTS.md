# CleaRx repository instructions

## Environment

- Activate the declared environment with `conda activate clearx` before running
  Python, CMake, Ninja, schema, test, or orchestration commands.
- Treat `environment.yml` as the environment contract. Record an intentional
  dependency change there before relying on it.

## Packet execution

- Read this file, the assigned immutable work packet, and any nearer
  `AGENTS.md` before editing.
- Inspect `git status` before work. Every packet runs from its recorded clean
  commit and packet digest.
- Edit only paths in the packet's `execution.write_scope`. Shared-interface
  changes require a dedicated packet or orchestrator approval.
- Use revisions in `upstream.lock`. Record every unavoidable deviation in the
  result manifest.
- Produce every declared artifact and a schema-valid result manifest. Run the
  packet's acceptance commands; compilation or prose does not establish
  completion.
- Report blockers instead of expanding scope, weakening checks, or changing
  the frozen precision policy.

## Evidence and infrastructure

- Correctness gates precede performance work. Performance claims require the
  packet's instrumentation, workload, repetitions, and host profile.
- Initiate A100 or L4 work only from a validated packet and clean Git commit.
- Keep credentials, access tokens, Colab session state, model weights, build
  products, and run payloads out of Git. Commit only compact manifests and
  evidence explicitly owned by a packet.
- Preserve task-agnostic architecture and typed module boundaries. A worker
  cannot change another packet's lifecycle state.

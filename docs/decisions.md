# CleaRx decision log

## Frozen execution decisions

| Decision | Current value | Change authority |
|---|---|---|
| Deployment accelerator | One NVIDIA L4 with a hard 24 GB VRAM ceiling | Dedicated architecture-decision packet |
| Certified L4 alias | `mantra-g2`, currently `mantra-g2-spot` in `us-west1-a` | Infrastructure packet plus host-profile revalidation |
| L4 system disk | Clean 375 GiB `pd-ssd` from pinned DLVM image `common-cu129-ubuntu-2204-nvidia-580-v20260818` | Infrastructure packet |
| MANTRA recovery artifact | `mantra-backup-blueprint`, retained and `READY` when the CleaRx host was created | Owner-controlled infrastructure action |
| Local environment | Conda environment `clearx` declared by `environment.yml` | Dependency change with environment validation |
| A100 control path | Google Colab CLI 0.6.0 using Application Default Credentials | Remote-runner packet |
| Orchestration records | `plan.yaml` fixes packet paths, digests, and dependencies; result manifests bind outputs to a packet digest and commit; `state.json` records lifecycle state | Protocol-change packet |
| Current execution boundary | Bootstrap-001, complete Phase 0, and Phase 1 readiness checkpoint | Owner review |

The [audited handoff](LOCAL_ORCHESTRATOR_MASTER_PROMPT.md) states the target
architecture and its acceptance gates. The
[master checklist](MASTER_EXECUTION_CHECKLIST.md) records the authorized
execution order.

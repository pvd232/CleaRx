# CleaRx execution environments

## Local control plane

The Apple M1 checkout owns Git integration, packet/state mutation, schema
validation, fixture inspection, CPU smoke tests, and small native builds. Run
project commands inside the `clearx` Conda environment declared by
[`environment.yml`](../../environment.yml).

```bash
conda activate clearx
python scripts/validate_bootstrap.py
```

The local machine cannot establish full-model numerical agreement, L4 memory
fit, PCIe behavior, or deployment latency.

## A100 reference executor

Google Colab CLI 0.6.0 is installed inside `clearx`. The local Application
Default Credentials currently pass the read-only `colab --auth=adc sessions`
request. An A100 run must begin from a validated packet and clean commit. The
wrapper records the packet, commit, environment, logs, result manifest, and
artifact hashes before the ephemeral session is released.

Colab availability, quota, and preemption are infrastructure outcomes. They do
not establish or refute numerical correctness.

## L4 deployment host

The stable SSH alias `mantra-g2` currently resolves through IAP to
`mantra-g2-spot` in `us-west1-a`. The private Spot VM has:

- `g2-standard-12` with one NVIDIA L4;
- a clean 375 GiB `pd-ssd` boot disk created from
  `common-cu129-ubuntu-2204-nvidia-580-v20260818`;
- no external IP;
- outbound access through `mantra-router-us-west1` and
  `mantra-nat-us-west1`; and
- labels `env=dev`, `owner=peter`, and `repo=clearx`.

The observed initial host reported NVIDIA driver `580.173.02`, 23,034 MiB GPU
memory, and 347 GiB free filesystem space. P0-005A must recapture the complete
host profile at the commit and run that consume it; this observation is not a
permanent certification.

The retained machine image `mantra-backup-blueprint` is the recovery source for
the earlier MANTRA environment. Restoring MANTRA is outside CleaRx packet scope.

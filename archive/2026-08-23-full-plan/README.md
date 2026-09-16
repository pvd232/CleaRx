# August 23 full-plan execution

This directory preserves the August 23 contract execution as a resumable
workspace snapshot. The files retain their original bytes and internal path
conventions so their recorded hashes and receipts remain meaningful.

| Path | Role |
| --- | --- |
| [`docs/`](docs/) | Proposal, architecture notes, execution decisions, and acceptance documentation. |
| [`orchestration/`](orchestration/) | Frozen plan, lifecycle state, schemas, and work packets. |
| [`runs/`](runs/) | Compact results recorded by bootstrap and phase-zero packets. |
| [`scripts/`](scripts/) | Packet lifecycle, validation, fixture, and remote-execution commands. |
| [`tests/`](tests/) | Contract fixtures and orchestration tests used by the execution. |
| [`tools/`](tools/) | Inventory, reference-fixture, cache-spike, and contract-validation tools. |
| JSON files at this directory root | Frozen tensor, memory, benchmark, evaluation, and deployment artifacts. |
| [`upstream.lock`](upstream.lock) | Exact upstream revisions and file identities used by the execution. |

The repository keeps one active environment contract at
[`environment.yml`](../../environment.yml). To resume this execution,
create a dedicated branch or worktree and restore this snapshot at that
worktree's root before running the archived packet commands. Preserve existing
result files; a resumed attempt should write new evidence.

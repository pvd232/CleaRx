# Add the four missing Phase 0 validators

## 1. Status

**Inspected:** `P0-001`, `P0-005A`, `P0-006A`, and `P0-007A` each name an
acceptance command whose Python entry point does not exist at Bootstrap-001
commit `4541b7f`.

**Failed:** P0-001 entered `failed` when its named validator was absent and the
packet's `execution.write_scope` did not include that path.

**Resolved:** Bootstrap-002 created and tested the four shared validator entry
points before the affected Phase 0 packets resumed execution.

## 2. Completion condition

Each affected Phase 0 packet now invokes a validator that existed before that
packet began. The validator accepts the packet's declared outputs and rejects a
record that omits one of its required checks.

## 3. Why acceptance could not run at `4541b7f`

The packet-to-command connector stops at the executable path:

```text
work_packet.acceptance.commands[0]
-> repository-relative Python path
-> missing file at 4541b7f
-> no acceptance verdict
```

The Phase 0 packet cannot create its own validator because that path is absent
from `execution.write_scope`. Expanding a completed packet's scope would change
its digest and invalidate the lifecycle history bound to the earlier bytes.

## 4. Files added by Bootstrap-002

Bootstrap-002 added these executable files:

- `tools/inventory/validate_upstream_lock.py`
- `tools/inventory/validate_l4_host_profile.py`
- `tools/benchmarks/validate_cache_contract.py`
- `tools/inventory/validate_evaluation_contract.py`

`tools/contract_validation.py` loads JSON and checks shared required fields,
digests, timestamps, and benchmark schemas. Each entry point checks the
domain-specific fields produced by its corresponding Phase 0 packet.

## 5. How the repair works

The primary orchestrator completed Bootstrap-002, validated its result, and
then moved P0-001 from `failed` back to `ready`. Each Phase 0 packet writes only
its declared outputs. Its acceptance command passes those outputs to the
pre-existing validator, which returns zero only when every named check passes.

## 6. Files and result record

Bootstrap-002 persisted the five validator files and this specification at one
Git commit. Its result record binds their byte hashes to the Bootstrap-002
packet digest and source commit. Each later Phase 0 result stores its validator
command and exit code under the existing result schema.

## 7. Verification

- `validator_paths_owned`: Bootstrap-002 declares every validator in both its
  write scope and exact output set.
- `success_cases_pass`: tests construct the smallest valid record for each
  validator.
- `malformed_records_rejected`: tests remove or corrupt required identities.
- `missing_evidence_rejected`: tests remove one domain-specific measurement or
  comparison field and require a nonzero verdict.

## 8. Files and execution order changed

| Surface | Required change |
|---|---|
| Packet authoring | Add Bootstrap-002 with the five validator paths and this specification as exact outputs. |
| Runtime | Complete Bootstrap-002 before resuming P0-001 or starting another affected packet. |
| Persistence | Store the Bootstrap-002 result below `runs/BOOTSTRAP-002/{run_id}/result.json`. |
| Verification | Each Phase 0 acceptance command invokes its existing path without changing the frozen packet. |
| Test | Exercise one valid and one severed record for every entry point. |
| Documentation | Record the failed P0-001 attempt and the correction receipt in the master checklist. |

## 9. Acceptance case

The success case supplies all required upstream repositories, file identities,
toolchains, and source-symbol citations; `validate_upstream_lock.py` returns
zero. The targeted rejection removes one required source symbol; the validator
raises `ContractViolation` before it can report `gaps_cite_symbols`.

## 10. Implementation order

1. Record the P0-001 failed transition without changing its packet bytes.
2. Add Bootstrap-002, shared validation helpers, four entry points, and tests.
3. Validate, commit, and receipt Bootstrap-002.
4. Return P0-001 to `ready`, run its unchanged preflight, and resume the audit.

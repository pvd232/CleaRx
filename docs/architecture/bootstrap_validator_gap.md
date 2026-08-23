# Phase 0 validator ownership correction

## 1. Status

**Inspected:** `P0-001`, `P0-005A`, `P0-006A`, and `P0-007A` each name an
acceptance command whose Python entry point does not exist at Bootstrap-001
commit `4541b7f`.

**Failed:** P0-001 entered `failed` when its named validator was absent and the
packet did not own that path.

**Proposed:** Bootstrap-002 creates and tests the four shared validator entry
points before any affected Phase 0 packet resumes execution.

## 2. Required claim

Every affected Phase 0 packet executes a pre-existing, Bootstrap-owned
validator that accepts the packet's declared outputs and rejects a record that
omits one of the packet's required checks.

## 3. Current gap

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

## 4. Contract models

Bootstrap-002 owns these executable files:

- `tools/inventory/validate_upstream_lock.py`
- `tools/inventory/validate_l4_host_profile.py`
- `tools/benchmarks/validate_cache_contract.py`
- `tools/inventory/validate_evaluation_contract.py`

`tools/contract_validation.py` owns shared JSON loading, required-field,
digest, timestamp, and benchmark-schema checks. Each entry point owns the
domain-specific fields consumed by its corresponding Phase 0 packet.

## 5. Execution

The primary orchestrator completes Bootstrap-002, validates its result, and
then transitions P0-001 from `failed` back to `ready`. A Phase 0 packet writes
only its declared outputs. Its acceptance command reads those outputs through
the pre-existing validator and returns zero only when every named contract
check passes.

## 6. Persisted evidence

Bootstrap-002 persists the five validator files and this specification at one
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

## 8. Propagation

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

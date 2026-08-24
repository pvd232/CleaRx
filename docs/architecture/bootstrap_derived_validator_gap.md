# Add the four missing derived Phase 0 validators

## Original failure

**Inspected:** P0-004, P0-005B, P0-006B, and P0-007B name validator paths
that are absent after Bootstrap-002 and outside each packet's write scope.

**Resolved:** Bootstrap-003 added those four validators and their contract
tests. Every derived Phase 0 acceptance command now resolves to an executable
that existed before the packet ran; no immutable Phase 0 packet changed.

## Why acceptance could not run

```text
derived packet acceptance command
-> repository-relative validator path
-> absent file
-> unsupported completion check
```

Without those executables, no code could check the memory budget, L4 spike,
deployment record, or ABI artifact named by a result.

## What Bootstrap-003 added

Bootstrap-003 wrote the validators, tested ordinary success records, and tested
one severed identity or required field per validator. Its result receipt records
the commit and each file hash. The existing packet commands now pass their
derived artifacts to those validators.

| Area | Change |
|---|---|
| Authoring | Add Bootstrap-003 to the plan and Bootstrap gates. |
| Runtime | Complete Bootstrap-003 before executing a derived packet. |
| Persistence | Bind validator, test, plan, packet, and state bytes in the Bootstrap-003 result. |
| Verification | Require manifest reconciliation, host/hash binding, raw spike evidence, or ABI section coverage. |
| Rejection | Remove one required phase, hash, measurement, or ABI section and require `ContractViolation`. |

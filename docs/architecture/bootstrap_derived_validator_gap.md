# Derived Phase 0 validator ownership correction

## Status and required claim

**Inspected:** P0-004, P0-005B, P0-006B, and P0-007B name validator paths
that are absent after Bootstrap-002 and outside each packet's write scope.

**Proposed:** Bootstrap-003 owns those four validators and their contract tests.
After its result is complete, every remaining derived Phase 0 acceptance command
resolves to a pre-existing executable without changing an immutable Phase 0
packet.

## Missing connector

```text
derived packet acceptance command
-> repository-relative validator path
-> absent file
-> unsupported completion check
```

The lost value is verifier coverage: a result may name required checks, but no
code evaluates the corresponding memory, spike, deployment, or ABI artifact.

## Closure

Bootstrap-003 writes the validators, tests ordinary success records, and tests
one severed identity or required field per validator. Its commit and result
receipt preserve byte identity. The existing packet command then supplies the
derived artifact to the Bootstrap-owned validator.

| Surface | Propagation |
|---|---|
| Authoring | Add Bootstrap-003 to the plan and Bootstrap gates. |
| Runtime | Complete Bootstrap-003 before executing a derived packet. |
| Persistence | Bind validator, test, plan, packet, and state bytes in the Bootstrap-003 result. |
| Verification | Require manifest reconciliation, host/hash binding, raw spike evidence, or ABI section coverage. |
| Rejection | Remove one required phase, hash, measurement, or ABI section and require `ContractViolation`. |

# Bootstrap fixture-index acceptance gap

## Required claim

P0-003B claims completion when `python scripts/validate_fixtures.py` validates
the generated reference fixtures. Its sole declared output is
`tests/fixtures/reference/manifest.json`.

## Observed gap

Before Bootstrap-005, full validation recursively parsed every JSON document
under `tests/fixtures/manifests/` as a fixture envelope. That directory also
contains benchmark, cache, host, and orchestration contracts governed by other
schemas. The command therefore failed on `expert_cache_contract.json` before it
looked for P0-003B output.

The declared reference `manifest.json` lists many boundary-case envelopes.
Bootstrap-004 defined strict per-fixture descriptors, but no schema defined that
list and no validator compared its boundary-case pairs with the complete
P0-003A matrix. The acceptance command therefore could not see a partial A100
collection.

## How full validation now reaches generated fixtures

[`fixture_index.schema.json`](../../orchestration/schemas/fixture_index.schema.json)
stores one generator commit and a list of records containing boundary ID, case
ID, envelope path, and envelope SHA-256. Every envelope path stays below
`tests/fixtures/reference/`.

Full validation now performs this chain for each index entry:

```text
index record
  -> exact envelope bytes
  -> strict descriptor
  -> boundary ID and case ID
  -> generator commit
  -> descriptor inputs and outputs
  -> exact artifact bytes
```

The validator imports the executable P0-003A matrix and requires equality
between its boundary-case pairs and the index pairs. It rejects duplicate
pairs, duplicate paths, duplicate descriptor IDs, missing pairs, extra pairs,
hash drift, join drift, and a fixture kind other than `model_boundary`.

Contract-only validation reads only the dedicated
`fixture_contract_examples/` directory and invokes the P0 harness. Full
validation reads those examples plus the generated reference index. Benchmark
and host contracts use separate validators.

## Acceptance case

The correction passes when fixture tests establish all of these results:

- a bound index and envelope validate;
- a missing boundary-case pair fails;
- full validation reports a missing generated index before P0-003B runs; and
- fixture-envelope validation reads only dedicated contract examples and the
  generated reference index.

P0-003B may allocate A100 compute only after this correction is committed and
the full command reaches the expected missing-index gate.

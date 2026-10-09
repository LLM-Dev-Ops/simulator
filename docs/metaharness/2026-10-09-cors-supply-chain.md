# MetaHarness evidence: CORS and supply-chain hardening

Date: 2026-10-09

## Scope and frozen acceptance contract

The candidate is accepted only if all of the following remain true:

1. The existing Node API tests pass without regression.
2. An exact configured browser origin receives credentialed CORS permission.
3. An unconfigured browser origin receives neither `Access-Control-Allow-Origin`
   nor `Access-Control-Allow-Credentials`.
4. Requests without an `Origin` header continue to work.
5. Production audit findings and dependency count do not regress.

The project source change is separate from Darwin's harness-policy search. The
evaluation model was fixed as a deterministic proposer with no LLM call.

## Pinned tools

- MetaHarness upstream: `ruvnet/metaharness@ea287d6ef7548b0b32fa3e20956fa548cfe51edb`
- `metaharness@0.4.17`
- `@metaharness/darwin@0.10.3`
- `@metaharness/flywheel@0.1.12`
- Node.js `v24.19.0`; npm `11.9.0`

## Reproducible results

| Check | Baseline | Candidate |
| --- | ---: | ---: |
| `npm test` | 64 passed, 0 failed | 72 passed, 0 failed |
| Credentialed arbitrary-origin CORS probe | exposed | denied |
| `npm audit --omit=dev --json` | 96 findings (3 critical, 39 high, 51 moderate, 3 low) | 0 findings |
| Production dependency total | 903 | 75 |
| Test command exit | 0 | 0 |
| Audit command exit | 1 | 0 |

The dependency reduction removes unused `claude-flow`; `index.js` imports none of
its modules. The remaining production tree passes `npm ls --all --omit=dev`.

## Darwin and Flywheel ledger

Darwin real-repository sandbox invocation:

```text
metaharness-darwin evolve <repo> --generations 1 --children 2 --concurrency 1 --seed 1009 --sandbox real --mutator deterministic
```

The repository-test substrate scored the baseline and both children at `0.985`;
neither child was promoted. A second real surface-code run with `--sandbox agent`
scored all three at `0.618333`; both children were rejected because neither beat
the parent by the frozen `0.05` delta. No harness-policy change was retained.

Flywheel evaluated live tests, audit output, the CORS probe, and the frozen
functional anchor. Its default `meetsPromotionRule` fingerprint was
`c5942e7c9bf8f28fe7af47a6ec3887d11eeb3f8c8e8601cc6a7a54969ea0b981`.
The combined candidate moved primary score `0.513 -> 1.000`, no-op rate `1 -> 0`,
and cost per passing test `14.109375 -> 1.041667`; the functional anchor remained
`1.0`. The signed replay passed with chain `gen1(candidate) -> gen0(root)`.

## Rollback

Revert the improvement commit. This restores the previous lockfile, dependency,
and permissive CORS behavior. If browser callers need access after rollout, set
`CORS_ALLOWED_ORIGINS` to their exact HTTPS origins; do not use a wildcard with
credentials.

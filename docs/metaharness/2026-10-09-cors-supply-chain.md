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
6. Rust formatting and strict all-target/all-feature Clippy pass.
7. The release unit, integration, property, and doc-test suites pass.
8. CI remains Linux-first and validates every OS matrix entry without fail-fast cancellation.

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
| `cargo fmt --all -- --check` | 1 | 0 |
| `cargo clippy --locked --all-targets --all-features -- -D warnings` | blocked by baseline format/dependency drift | 0 |
| `cargo test --locked --release` | not reproducible under the declared 1.75 toolchain | 352 passed, 0 failed, 8 ignored |
| `cargo +1.88.0 audit --no-fetch` | 10 vulnerabilities | 7 vulnerabilities (gate failed) |

The dependency reduction removes unused `claude-flow`; `index.js` imports none of
its modules. The remaining production tree passes `npm ls --all --omit=dev`.

## CI and Rust reproducibility repair

- Pins the minimum Rust toolchain at `1.83.0`, the lowest version satisfying the
  committed graph; the previous CI pin (`1.75.0`) could not build crates requiring
  Rust 1.81–1.83.
- Pins the optional organization dependencies to immutable revisions:
  `latency-lens@22b825ef309c2cbb14e1ba528945b853c3aeccde`,
  `observatory@6813701693cc85094ea6c52cb00ad4911446b096`, and
  `memory-graph@9acd0228098df03b50361769993571fa5ab549bf`.
- Restores the all-feature gate without suppressing it. The later Memory Graph
  revisions `c34247b` and `57d08ce` were rejected after real compilation produced
  118 source errors and one undeclared `tempfile` import respectively.
- Runs the current CVSS 4.0-capable `cargo-audit@0.22.2` under its isolated
  Rust 1.88 toolchain while preserving Rust 1.83 as the project MSRV; pins
  `cargo-tarpaulin@0.31.3` and `protoc@25.1`; updates
  cache/artifact actions to v4; prevents matrix fail-fast cancellation; adds the
  missing Node test/audit job; and pins checksum-verified `kubeval@0.16.1` for
  offline Kubernetes schema validation.
- Applies Rust 1.83 formatting and machine-applicable Clippy fixes. One real
  serialization defect was exposed and fixed by emitting `SignalPayload` variant
  tags in snake case, matching the existing contract test.
- Updates compatible locked transitive releases for `bytes`, `crossbeam-epoch`,
  and `h2` 0.4, eliminating three RustSec findings. The attempted `time` update
  was rejected because its Edition 2024 manifest does not parse under the frozen
  Rust 1.83 MSRV. Seven findings remain through `time` and older major-version
  graphs (`h2` 0.3, `idna` 0.4, `protobuf` 2, and `rustls-webpki` 0.101); no
  advisory was ignored.

## Darwin and Flywheel ledger

Darwin real-repository sandbox invocation:

```text
metaharness-darwin evolve <repo> --generations 1 --children 2 --concurrency 1 --seed 1009 --sandbox real --mutator deterministic
```

The current repository-test substrate scored the baseline and both children at `0.985`;
neither child was promoted. A second real surface-code run with `--sandbox agent`
scored all three at `0.618333`; both children were rejected because neither beat
the parent by the frozen `0.05` delta. No harness-policy change was retained.

Flywheel evaluated live tests, Node and Rust audit output, the CORS probe, Rust
formatting, and the frozen functional anchor. Its default `meetsPromotionRule`
fingerprint was
`c5942e7c9bf8f28fe7af47a6ec3887d11eeb3f8c8e8601cc6a7a54969ea0b981`.
The combined candidate moved primary score `0.3078 -> 0.8000` and cost per
passing test `14.109375 -> 1.041667`, but retained no-op/regression state `1/true`
because RustSec still reports seven vulnerabilities. Flywheel correctly rejected
promotion and retained `gen0(root)`; the signed replay passed. This supersedes an
earlier Node-only evaluation that had promoted the candidate before the RustSec
CVSS 4.0 parser incompatibility was discovered.

## Rollback

Revert the improvement commits. This restores the previous lockfile, toolchain,
CI, dependency, and permissive CORS behavior. If browser callers need access after rollout, set
`CORS_ALLOWED_ORIGINS` to their exact HTTPS origins; do not use a wildcard with
credentials.

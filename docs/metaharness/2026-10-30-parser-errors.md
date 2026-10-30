# Parser error boundary

Parser failures return sanitized JSON with execution metadata and a failed routing layer:
malformed JSON400/INVALID_JSON, bodies above the existing1MiB limit413/PAYLOAD_TOO_LARGE,
and unsupported charset/encoding415/UNSUPPORTED_MEDIA_TYPE. Existing schema errors remain
400/INVALID_REQUEST. Errors never echo parser messages, bodies, stack traces or paths.
Parser error responses include nosniff. Existing exact-origin CORS controls remain unchanged.

Run `npm test` for the original72 assertions plus the six native-socket holdouts
and six preservation anchors. `node scripts/probe-parser-errors.cjs ROOT --json`
records statuses and contract booleans against a real loopback Express server.
No provider calls, mocked transports or fake clocks are used by this probe.

Reproduce live signed evaluation:
`node scripts/evaluate-parser-errors.cjs ROOT BASELINE_SHA CANDIDATE_SHA TOOLS OUTPUT`.
Fixture/package/lock/old-test bytes must match both revisions. All nonzero or timed-out
checks reject promotion. Rust commands explicitly use the installed1.88.0 toolchain;
this is not proof of the CI-pinned1.83.0 MSRV. CI remains authoritative.
Failed-contract ratio is the no-op axis; zero cost is neutral, not a measured cost.

For five alternating loopback trials, run:
`node scripts/benchmark-parser-errors.cjs ROOT BASELINE_SHA CANDIDATE_SHA OUTPUT`.
Timings are whole contract trials including startup/request parsing and different error
serialization; no production latency, model efficacy or speedup claim is made.

Rollback via protected PR: revert this cycle's implementation commit, then its fixture commit,
preserving earlier CORS/security work. The signed inverse retains exact parent index.js bytes.
No deployed data, service persistence, IAM, secret configuration or Rust source is changed.

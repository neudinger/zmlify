# Zerkerus TASKS — Production Readiness

This file lists prioritized tasks to make zerkerus production-ready. Use it as a living checklist. Each task includes a short description, priority, suggested labels, and acceptance criteria. Copy into zerkerus/TASKS.md in the repo.

Project: Zerkerus — Production Readiness
Milestone: zerkerus-production-ready (suggested)
Columns for Project board: To do / In progress / Done

How to use
- Check a task when it is completed.
- When work starts, move the card to "In progress" on the Project board.
- When finished, move the card to "Done" and close the corresponding issue (if one exists).
- Suggested labels: P0-security, P0-bug, P1-improvement, P2-doc, infra, ci, perf, dependency

------------------------------------------------------------
1. SECURITY & CRYPTOGRAPHY (P0)
------------------------------------------------------------
- [ ] External cryptographic audit of protocol and parameters
  - Priority: P0-security
  - Description: Engage an external reviewer/auditor to review design, parameter choices (N, M, B, Q), rejection-sampling bounds, and zero-knowledge claims.
  - Acceptance criteria:
    - Written audit report with findings and remediations.
    - All critical/high findings converted to issues.

- [ ] Review MakeNTT compile-time checks and constant correctness
  - Priority: P0
  - Labels: P0-bug, dependency
  - Acceptance criteria:
    - Confirm (Q-1) divisibility and primitive-root discovery correctness.
    - Tests verifying constants for production profile.

- [ ] Constant-time & side-channel audit (rejectionSample, checkNorm, transcript)
  - Priority: P0-security
  - Acceptance criteria:
    - All secret-dependent code audited and replaced with constant-time implementations where needed.
    - Unit tests that assert no secret-dependent branches or timing side-channel patterns for critical functions.

- [ ] Secure buffer wiping & lifecycle
  - Priority: P0
  - Acceptance criteria:
    - All sensitive buffers zeroed on normal and error/abort paths.
    - `.deinit()` coverage verified with tests and code review.

- [ ] CSPRNG: replace/verify RNG sources
  - Priority: P0-security
  - Acceptance criteria:
    - Use OS-provided CSPRNG (getrandom / Arc4 / platform API) for all secrets.
    - Document seed and reseed policy.

------------------------------------------------------------
2. CORRECTNESS, TESTING & REPRODUCIBILITY (P0)
------------------------------------------------------------
- [ ] Switch to production profile and run full tests
  - Priority: P0
  - Description: Change params.zig to use production profile, run all tests and benchmark runs.
  - Acceptance criteria:
    - All unit tests pass under production profile.
    - Document any failures and fixes.

- [ ] Pin toolchain and dependency versions; add reproducible build docs
  - Priority: P0, Labels: infra, dependency
  - Acceptance criteria:
    - Lock versions for Zig, Bazel, ZML, flatcc, @zk-crypto.
    - Add BUILD.md/reproducible build instructions.

- [ ] Add ASAN/UBSAN and static analysis in CI
  - Priority: P0, Labels: ci
  - Acceptance criteria:
    - CI job that runs ASAN/UBSAN (where supported).
    - Static analysis run and any critical issues fixed.

- [ ] Add fuzzing for FlatBuffer parsing and transcript codepaths
  - Priority: P0, Labels: ci, security
  - Acceptance criteria:
    - Fuzz harnesses for parsers and transcript functions.
    - CI job or nightly fuzz runner with triage process.

- [ ] Integration test: run Prover and Verifier in separate processes
  - Priority: P0
  - Acceptance criteria:
    - Networked integration test (e.g., localhost TCP/TLS) validating serialization, Fiat-Shamir transcript agreement, and end-to-end verification.
    - Tests added to CI.

------------------------------------------------------------
3. SIDE-CHANNEL, PERFORMANCE & AVAILABILITY (P1)
------------------------------------------------------------
- [ ] Measure rejection-sampling acceptance rate & tune
  - Priority: P1, Labels: perf
  - Acceptance criteria:
    - Benchmarks showing acceptance rate for production parameters.
    - Parameter adjustments or operational guidance to avoid excessive retries.

- [ ] Add DoS mitigations for high retry rates
  - Priority: P1
  - Acceptance criteria:
    - Server-side rate limiting, retry limits, and backoff policy implemented and tested.

- [ ] Benchmarks for latency/throughput and accelerator usage
  - Priority: P1
  - Acceptance criteria:
    - Scripts that measure performance on supported hardware.
    - Nightly or periodic benchmark jobs.

------------------------------------------------------------
4. BUILD, CI & PORTABILITY (P1)
------------------------------------------------------------
- [ ] CI matrix: macOS & Linux, x86_64 and arm64; include production profile
  - Priority: P1, Labels: ci, infra
  - Acceptance criteria:
    - Matrix CI workflow covering platforms and production profile.
    - Failures investigated and mitigated.

- [ ] Provide CPU fallback / portable execution path for ZML/MLIR
  - Priority: P1
  - Acceptance criteria:
    - CPU-based fallback so the project can run on cloud VMs without Apple accelerators.
    - Documented instructions for fallback builds.

- [ ] Add reproducible Bazel toolchain pinning
  - Priority: P1, Labels: infra
  - Acceptance criteria:
    - Bazel toolchain versions pinned and documented.

------------------------------------------------------------
5. SERIALIZATION & INTEROPERABILITY (P1)
------------------------------------------------------------
- [ ] Freeze and semver .fbs schemas; add compatibility tests
  - Priority: P1
  - Acceptance criteria:
    - .fbs files versioned and compatibility tests for forward/backward compatibility.

- [ ] Validate Zig ↔ flatcc interop across compilers/toolchains
  - Priority: P1
  - Acceptance criteria:
    - CI jobs or matrix builds that validate flatcc integration on supported toolchains and report issues.

- [ ] Input validation and size limits for FlatBuffers/Base64 transport
  - Priority: P1, Labels: security
  - Acceptance criteria:
    - Strict validation of incoming payload sizes and types.
    - Tests covering malformed inputs.

------------------------------------------------------------
6. NETWORKING & DEPLOYMENT (P1)
------------------------------------------------------------
- [ ] Implement secure transport spec and reference implementation
  - Priority: P1, Labels: infra, security
  - Acceptance criteria:
    - TLS-secured endpoints, replay protection, nonce/session handling.
    - Reference HTTP(s) endpoints and example client.

- [ ] Reference Docker image and optional Helm chart
  - Priority: P1
  - Acceptance criteria:
    - Dockerfile producing reproducible container image.
    - Basic Helm chart or k8s manifest examples.

------------------------------------------------------------
7. PACKAGING, RELEASES & GOVERNANCE (P2)
------------------------------------------------------------
- [ ] Signed releases, SBOM, and artifact hashes
  - Priority: P2
  - Acceptance criteria:
    - Release process documented and artifacts signed.
    - SBOM generated and included in releases.

- [ ] License and dependency license checks
  - Priority: P2
  - Acceptance criteria:
    - LICENSE at repo root clarified.
    - Dependency license scan results documented.

- [ ] Export control & privacy notes (if applicable)
  - Priority: P2
  - Acceptance criteria:
    - Notes added to docs about any regulatory/export considerations.

------------------------------------------------------------
8. OBSERVABILITY, OPERATIONS & INCIDENT RESPONSE (P2)
------------------------------------------------------------
- [ ] Structured logging & secret redaction
  - Priority: P2
  - Acceptance criteria:
    - Log format and redaction verification tests.

- [ ] Metrics & alerts
  - Priority: P2
  - Acceptance criteria:
    - Exported metrics for accept/reject rates, latencies, errors.
    - Example Prometheus metrics or guidance.

- [ ] Incident response & key rotation plan
  - Priority: P2
  - Acceptance criteria:
    - Written plan for key compromise and rotation procedures.

------------------------------------------------------------
9. DOCUMENTATION & DEVELOPER ERGONOMICS (P2)
------------------------------------------------------------
- [ ] Threat model & production-readiness README section
  - Priority: P2, Labels: doc
  - Acceptance criteria:
    - Clear threat model describing assumptions (honest-but-curious vs malicious server), attacker goals, and limitations.

- [ ] CONTRIBUTING.md and developer guide
  - Priority: P2, Labels: doc
  - Acceptance criteria:
    - Build/test/run guide, how to switch profiles, how to run production tests.

- [ ] Quickstart examples (networked client/server)
  - Priority: P2
  - Acceptance criteria:
    - Minimal example demonstrating registration and authentication over TLS.

------------------------------------------------------------
APPENDIX: Quick actionable commands / snippets
------------------------------------------------------------
- Switch to production profile (params.zig):
  ```zig
  // pub const ntt     = production.ntt;
  // pub const labrador = production.labrador;
  ```
  (Uncomment / point to production and run full test suite.)

- Run tests:
  ```
  bazel test //zerkerus:zerkerus_test
  bazel run //zerkerus
  ```

------------------------------------------------------------
Suggested labels to create and use
- P0-security, P0-bug, P1-improvement, P2-doc, infra, ci, perf, dependency
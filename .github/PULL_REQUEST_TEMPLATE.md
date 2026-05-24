## Summary

<!--
One or two sentences on what changes and why. Link the issue this closes
(e.g. "Closes #123") so it auto-closes on merge.
-->

## Test plan

<!--
Concrete steps a reviewer can run locally, plus the commands you ran
yourself. Include `cargo test` invocations, manual repros, benchmarks,
or screenshots as appropriate. If this is GPU-touching code, note
whether it was exercised on a real device or only against the
`cuda-stub` feature.
-->

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --lib --tests --features cuda-stub --no-default-features`
- [ ] `cargo test --lib --tests --features cuda-stub --no-default-features`
- [ ] Manual verification (describe):

## Checklist

- [ ] Documentation updated where behaviour changes
- [ ] `CHANGELOG.md` entry added (if user-visible)
- [ ] No new `unsafe` without a `// SAFETY:` comment

## License & DCO sign-off

By submitting this pull request, I certify that:

- My contribution is licensed under the project's **Apache License, Version 2.0**
  (see [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE)).
- I agree to the **Developer Certificate of Origin** (DCO) v1.1 —
  https://developercertificate.org — and my commits are signed off with
  `git commit -s` (a `Signed-off-by: Name <email>` trailer).

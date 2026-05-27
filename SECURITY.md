# Security Policy

Thank you for helping keep Craton Bolt and its users safe.

## Supported versions

Craton Bolt is pre-1.0. While the API is unstable, only the **latest minor
release line** receives security fixes — older minor lines are not
backported. This is standard practice for pre-1.0 crates and keeps the
maintenance surface narrow while the IR and public API are still moving.

The current supported line is `0.3.x`. `0.1.x` is no longer supported;
users on `0.1.x` should upgrade to `0.3.x` (note that `0.2.0` was skipped —
see `CHANGELOG.md`).

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.1.x   | :x:                |
| < 0.1   | :x:                |

## Reporting a vulnerability

**Please do not file public GitHub issues for security vulnerabilities.**
Private disclosure is strongly preferred so that we can ship a fix before
the issue is widely known.

Report vulnerabilities by email to:

> **security@cratonsoftware.com**

If you would prefer encrypted email, request our PGP key in your first
message and we will provide it before you send details.

Please include, where possible:

- A description of the issue and its impact.
- Steps to reproduce (a minimal Rust snippet or SQL query is ideal).
- Affected version(s) / commit SHA.
- Your assessment of severity, and whether the issue is already public.

You may also use [GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
if you prefer.

## Our process

- We will acknowledge your report within **5 business days**.
- We aim to provide an initial assessment within **10 business days**.
- We follow a **90-day coordinated disclosure** timeline by default: the
  reporter and maintainers agree on a public disclosure date no later than
  90 days after the initial report. Extensions are possible for complex
  fixes by mutual agreement.
- Once a fix ships, we will publish a GitHub Security Advisory and credit
  the reporter (unless anonymity is requested).

## Scope

In scope: anything shipped from this repository — the Craton Bolt Rust crate,
its build scripts, CI workflows, and documentation.

Out of scope: vulnerabilities in upstream dependencies (please report
those to the relevant project), and issues that require an attacker with
existing root or physical access to the host.

# CI/CD Pipelines

This directory contains GitHub Actions workflows for automated quality checks, testing, and security scanning.

---

## 🔄 [`ci.yml`](ci.yml) — Continuous Integration

**Triggers:** Push to `main` · Pull requests to `main`

| Job | Steps | Purpose |
|-----|-------|---------|
| **Backend Lint & Test** | `flake8` → `mypy` → `pytest` | Lints Python code, runs type-checking, then executes unit & contract tests with coverage |
| **Frontend Build** | `npm ci` → `tsc --noEmit` → `vite build` | Installs deps, type-checks TypeScript, and builds the production bundle |
| **Integration Tests** | `pytest tests/integration/` | Runs after the above two pass — tests multi-agent communication & workflows |

**Artifacts produced:**
- `backend-coverage` — XML coverage report (retained 14 days)
- `frontend-build` — Production build output (retained 7 days)

---

## 🔒 [`security.yml`](security.yml) — Security Scanning

**Triggers:** Push to `main` · Weekly (Monday 06:00 UTC)

| Job | Tool | Purpose |
|-----|------|---------|
| **Backend Security Scan** | `bandit` | Static analysis for common Python security issues (SQL injection, hardcoded passwords, etc.) |
| **Frontend Dependency Audit** | `npm audit` | Checks for known vulnerabilities in npm dependencies |

**Artifacts produced:**
- `bandit-report` — JSON security report (retained 30 days)

---

## Quick Reference

```
Push/PR to main
  ├── ci.yml
  │   ├── Backend Lint & Test (Python 3.11)
  │   ├── Frontend Build (Node 20)
  │   └── Integration Tests (after ↑ pass)
  │
  └── security.yml
      ├── Backend Security (bandit)
      └── Frontend Audit (npm audit)
```

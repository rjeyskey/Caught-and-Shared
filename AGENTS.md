## Agent skills

### Issue tracker

Issues live in the GitHub repo `rjeyskey/Caught-and-Shared`; use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## GitHub access

Authenticated via the `GITHUB_TOKEN` stored in `D:\Hermes\.env` (scopes: `repo`, `workflow`, `read:org`). The `gh` CLI is not installed; use the API directly:

```bash
export GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" /d/Hermes/.env | head -1 | cut -d= -f2 | tr -d '\r\n')
```

Proven operations (tested 2026-10-07): `POST /repos/{owner}/{repo}/labels`, `PATCH /repos/{owner}/{repo}/labels/{name}`, `GET /repos/{owner}/{repo}/labels`. Triage labels created: `needs-triage` (B60205), `needs-info` (FBCA04), `ready-for-agent` (0E8A16), `ready-for-human` (1D76DB), `wontfix` (FFFFFF).
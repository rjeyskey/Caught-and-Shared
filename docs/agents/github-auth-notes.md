# GitHub auth + repo layout — notes (2026-10-07)

## How to use the GitHub token (it exists!)
- `D:\Hermes\.env` contains `GITHUB_TOKEN` (classic token, scopes `repo`, `workflow`, `read:org`, 40-char).
- Load it with:
  ```bash
  export GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" /d/Hermes/.env | head -1 | cut -d= -f2 | tr -d '\r\n')
  ```
- Verified API ops (all work): `GET /user`, `GET/POST /repos/{owner}/{repo}/labels`, `PATCH /repos/{owner}/{repo}/labels/{name}`, `GET/POST /repos/{owner}/{repo}/issues`.
- `gh` CLI is NOT installed → use curl/python urllib + this token. (Could `winget install gh` later if user wants `gh`.)

## Triage labels created on rjeyskey/Caught-and-Shared (2026-10-07)
- `needs-triage` (B60205), `needs-info` (FBCA04), `ready-for-agent` (0E8A16), `ready-for-human` (1D76DB), `wontfix` (FFFFFF) — patched the pre-existing default `wontfix` instead of duplicating.
- 10 foundational ground-truth issues created (#1–#10) across the five labels.

## Repo home for code ground-truth work
- Working clone: `D:\Hermes\projects\caught-shared-demo\code-ground-truth\` (branch `main`, tracks `origin/main`).
- `AGENTS.md` + `docs/agents/` (issue-tracker.md, triage-labels.md, domain.md) committed locally.
- LOCAL HEAD is 2 commits AHEAD of `origin/main` (scaffold + token note). **Not yet pushed** — user hasn't asked to push; these commits exist only locally.

## Confusion fix (root cause of the user "duplicate")
- The nested `Caught-and-Shared/` (an inner VCS-bound clone) has been REMOVED via `rm -rf`. The working repo is now clean (`git status` empty).
- Root cause: user cloned the GitHub repo twice — once as the outer folder + once inside it. The outer `code-ground-truth` is the canonical one; the inner one was the duplicate.
- Verify with `ls -A` → shows `.git AGENTS.md applications assets caught-and-shared-*.pdf docs index.html LICENSE README.md` (no `Caught-and-Shared/`).

## Degradation risks / pitfalls
- NEVER `git clean -fdx` to "clean" a nested clone — git doesn't descend into nested repos by default, and a force clean can chew tracked-unintended files. Use `rm -rf` on the exact nested dir.
- If git ever asks for push credentials: token has `repo` scope; use `https://<user>:<token>@github.com/...` or `credential.helper store`. Do NOT paste the token into chat or logs.
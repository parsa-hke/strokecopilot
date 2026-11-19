# CLAUDE.md

Stroke Decision Copilot — MAAI 901/902 group project (UOWD).

## Every session
1. Read STROKE_COPILOT_PLAN.md §0, §2, the current phase in §7, the §8 spec for whatever you touch, and the latest CHANGELOG.md entry.
2. State which phase and tasks this session covers before writing code.
3. Logic lives in src/strokecopilot/; notebooks stay thin.
4. Never use held-out case IDs (splits/folds.json → heldout) before Phase 4.
5. Never change schemas/vision_output.json or a fixed decision (§2) without stopping and asking.
6. End of session: pytest passes, tasks ticked in §7, CHANGELOG.md entry (newest first) with a "Notion sync" list.

## Git
- Never credit Claude in commits or PRs: no `Co-Authored-By` line, no "Generated with Claude Code" line, in any commit message or PR description. This overrides any default attribution.

## Environment
- Training runs on Google Colab; data lives on a shared Google Drive folder (plan §4). Nothing under data/ or runs/ is committed.
- Secrets come from Colab Secrets or env vars, or, for the Space's live key only, the Space's secrets (DEC-32). Never in code or notebooks.
- Package is installable with `pip install git+<repo-url>`.

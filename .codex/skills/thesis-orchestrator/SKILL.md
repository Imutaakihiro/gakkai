---
name: thesis-orchestrator
description: Evaluate chapter completeness and decide next tasks/agent to call for the thesis project.
---

# Thesis Orchestrator

## When to use
- Need a progress report and prioritized next actions.

## Must follow
- `AGENTS.md` volume targets and constraints.

## Workflow
1. Read `本論/卒論Tex/1章.tex`〜`5章.tex`.
2. Estimate pages (≈40 lines/page).
3. Count figures/tables and subsections.
4. Score: volume, depth, figures, data use, structure (20 each).
5. Propose next tasks and which skill to use.

## Output
- Summary table with per-chapter scores.
- Top 2–3 prioritized tasks.
- Concrete instructions for the next skill call.

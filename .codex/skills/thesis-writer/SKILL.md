---
name: thesis-writer
description: Write or expand thesis chapters in 本論/卒論Tex (である調, no first person, add sections/figures/tables per AGENTS.md).
---

# Thesis Writer

## When to use
- Expand or draft sections in `本論/卒論Tex/1章.tex`〜`5章.tex`.
- Add explanations, discussion, or structure while preserving existing tone.

## Must follow
- `AGENTS.md` rules (style, fixed numbers, no causal assertions, citations).
- No first person; use 「本研究では」「本章では」.
- Keep terminology consistent.

## Workflow
1. Open the target chapter TeX file.
2. Identify thin sections and missing subsections/figures.
3. Draft additions in である調, include concrete numbers.
4. Insert at logical positions; add `\ref{}` to any new figure/table.
5. If new citations are needed, add Bib entries to `本論/卒論Tex/10参考文献.tex`.

## Fixed values (must match)
- データ期間: 2018年度〜2023年度
- 授業数: 3,268
- 自由記述総件数: 83,851
- 教師データ: 1,000件（ネガ191 / ニュートラル628 / ポジ180）
- 相関係数: ピアソン0.3097 / スピアマン0.2970 / ケンドール0.2042
- 授業評価平均: 3.459（SD 0.216）

## Tips
- Ensure each chapter has multiple subsections.
- Add at least one figure/table per chapter (more in results).
- Avoid subjective or causal statements; use “示唆する/可能性がある”.

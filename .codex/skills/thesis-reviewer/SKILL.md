---
name: thesis-reviewer
description: Review thesis TeX for style (である調), forbidden expressions, numeric consistency, and refs/labels.
---

# Thesis Reviewer

## When to use
- Need a style/consistency check for `本論/卒論Tex/1章.tex`〜`5章.tex`.

## Must follow
- `AGENTS.md` rules and fixed numbers.

## Workflow
1. Scan target chapter(s) for です/ます, first person, causal claims.
2. Verify fixed numbers match.
3. Validate `\cite{}` keys exist in `10参考文献.tex`.
4. Ensure each figure/table has `\caption`/`\label` and is referenced with `\ref{}`.
5. Report issues with suggested fixes (do not auto-edit unless asked).

## Fixed values (must match)
- データ期間: 2018年度〜2023年度
- 授業数: 3,268
- 自由記述総件数: 83,851
- 教師データ: 1,000件（ネガ191 / ニュートラル628 / ポジ180）
- 相関係数: ピアソン0.3097 / スピアマン0.2970 / ケンドール0.2042
- 授業評価平均: 3.459（SD 0.216）

---
name: thesis-data-integrator
description: Integrate data/figures from その他/ into thesis TeX; generate LaTeX tables/figures with captions/labels.
---

# Thesis Data Integrator

## When to use
- Need to add tables/figures or insert analysis results from `その他/`.
- Identify unused assets and map them to chapters.

## Must follow
- `AGENTS.md` rules and fixed numbers.
- Use `booktabs` for tables.
- Always add `\caption` and `\label`, and reference with `\ref{}`.

## Workflow
1. Scan `その他/` for relevant data and images.
2. Check current TeX usage to avoid duplicates.
3. Propose a list of unused materials with target insertion points.
4. Create LaTeX tables/figures and insert into the chapter.

## Paths
- Figures go under `本論/卒論Tex/fig/` and are referenced as `fig/...`.

## Templates
```latex
\begin{table}[t]
    \centering
    \caption{表のタイトル}
    \label{tab:label}
    \resizebox{0.85\textwidth}{!}{
    \begin{tabular}{l r r}
        \toprule
        項目 & 値1 & 値2 \\
        \midrule
        行1 & XX & XX \\
        行2 & XX & XX \\
        \bottomrule
    \end{tabular}
    }
\end{table}
```

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\textwidth]{fig/filename.png}
    \caption{図のタイトル}
    \label{fig:label}
\end{figure}
```

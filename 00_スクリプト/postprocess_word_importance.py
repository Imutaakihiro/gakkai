#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP語重要度の出力を、日本語として自然な語形に正規化する後処理スクリプト。

入力: word_importance.csv（インデックスが語、列に mean_shap/abs_mean_shap/std_shap/count）
出力: word_importance_natural.csv（語形を正規化し集約したもの）
オプション: 上位20語の棒グラフも再生成
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False


def normalize_token(token: str) -> str:
    """簡易的な語形復元規則。サブワード結合後の語幹を自然な形へ正規化する。
    注意: 文脈に依存する語は保守的に処理。
    """
    if not token:
        return token

    t = token

    # 代表的な活用・接尾辞の補完
    rules_exact = {
        "よかっ": "よかった",
        "わかっ": "わかった",
        "分かっ": "分かった",
        "なかっ": "なかった",
        "難しかっ": "難しかった",
        "多かっ": "多かった",
        "少なかっ": "少なかった",
        "学ん": "学んだ",
        "出来": "できた",
        "でき": "できた",
        "まし": "ました",
        "にく": "にくい",
        "やす": "やすい",
        "すぎ": "すぎる",
        "分から": "分からない",
        "わから": "わからない",
        "良かっ": "良かった",
        "かっ": "かった",
    }
    if t in rules_exact:
        return rules_exact[t]

    # 語尾ルール（保守的に）
    if t.endswith("かっ"):
        return t[:-2] + "かった"
    if t.endswith("なかっ"):
        return t[:-3] + "なかった"
    if t.endswith("にく"):
        return t + "い"
    if t.endswith("やす"):
        return t + "い"
    if t.endswith("すぎ"):
        return t + "る"

    return t


def aggregate_by_natural_form(df: pd.DataFrame) -> pd.DataFrame:
    # インデックスに語が入っている前提
    df = df.copy()
    df.index = df.index.astype(str)
    df["natural"] = [normalize_token(w) for w in df.index]

    # 重複する自然語形で集約（countで重みづけした加重平均）
    def weighted_mean(group, col):
        counts = group["count"].astype(float).values
        vals = group[col].astype(float).values
        if counts.sum() == 0:
            return float(np.mean(vals))
        return float(np.sum(vals * counts) / np.sum(counts))

    agg = (
        df.groupby("natural")
          .apply(lambda g: pd.Series({
              "mean_shap": weighted_mean(g, "mean_shap"),
              "abs_mean_shap": weighted_mean(g, "abs_mean_shap"),
              "std_shap": weighted_mean(g, "std_shap"),  # 近似
              "count": int(g["count"].sum()),
          }))
          .sort_values("mean_shap", ascending=False)
    )
    return agg


def save_top20_plots(df: pd.DataFrame, out_dir: str, suffix: str = "") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ポジ寄与（上位20）
    top_pos = df.head(20)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_pos)), top_pos['mean_shap'], color='green', alpha=0.75)
    plt.yticks(range(len(top_pos)), top_pos.index)
    plt.xlabel('平均SHAP値（ポジティブ寄与）')
    plt.title('ポジティブ寄与 上位20（自然語形）')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"top20_positive_natural{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ネガ寄与（下位20を昇順で）
    top_neg = df.tail(20).sort_values('mean_shap')
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_neg)), top_neg['mean_shap'], color='red', alpha=0.75)
    plt.yticks(range(len(top_neg)), top_neg.index)
    plt.xlabel('平均SHAP値（ネガティブ寄与）')
    plt.title('ネガティブ寄与 上位20（自然語形）')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"top20_negative_natural{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='入力CSV: word_importance.csv')
    ap.add_argument('--output', required=False, help='出力CSV（省略時は同ディレクトリに word_importance_natural.csv）')
    ap.add_argument('--plot_dir', required=False, help='グラフ出力ディレクトリ（省略時は input/可視化）')
    args = ap.parse_args()

    inp = args.input
    df = pd.read_csv(inp, index_col=0)
    if not set(['mean_shap','abs_mean_shap','std_shap','count']).issubset(df.columns):
        raise ValueError('入力CSVの列が想定と異なります。必要: mean_shap, abs_mean_shap, std_shap, count')

    agg = aggregate_by_natural_form(df)

    # 出力パス
    if args.output:
        out_csv = args.output
    else:
        base_dir = os.path.dirname(inp)
        out_csv = os.path.join(base_dir, 'word_importance_natural.csv')

    agg.to_csv(out_csv, encoding='utf-8-sig')
    print(f"✓ 保存: {out_csv} (rows={len(agg)})")

    # グラフ
    if args.plot_dir:
        plot_dir = args.plot_dir
    else:
        plot_dir = os.path.join(os.path.dirname(inp), '可視化')
    save_top20_plots(agg, plot_dir)
    print(f"✓ グラフ更新: {plot_dir}/top20_*_natural*.png")


if __name__ == '__main__':
    main()



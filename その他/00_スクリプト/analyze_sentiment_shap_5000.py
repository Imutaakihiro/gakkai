#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP分析スクリプト（5,000件サンプリング専用）
8万件から5,000件をランダムサンプリングして、語単位でSHAP集計のみ実施
"""

import torch
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import json
import os
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("SHAP分析: 5,000件サンプリング版")
print("="*60)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# モデルとトークナイザーのロード
MODEL_PATH = r"C:\Users\takahashi.DESKTOP-U0T5SUB\Downloads\BERT\git_excluded\finetuned_bert_model_20250718_step2_fixed_classweights_variant1_positive重点強化"
print(f"モデルをロード中: {MODEL_PATH}")
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# データ読み込み（感情ラベル付きデータから層化サンプリング）
RAW_TEXT_PATH = r"../感情分類結果_前処理データ結合_20250729_154855.csv"
SAMPLE_SIZE = 5000

print(f"\n元データをロード中: {RAW_TEXT_PATH}")
raw_df = pd.read_csv(RAW_TEXT_PATH)

# 必要な列: 自由記述, tuned_model_label
print(f"元データ総件数: {len(raw_df):,}件")

# データクリーニング
raw_df_clean = raw_df[['自由記述', 'tuned_model_label']].dropna()
print(f"クリーニング後: {len(raw_df_clean):,}件")

# ラベル分布を確認
label_counts = raw_df_clean['tuned_model_label'].value_counts()
print(f"\n元データのラベル分布:")
for label, count in label_counts.items():
    print(f"  {label}: {count:,}件 ({count/len(raw_df_clean)*100:.1f}%)")

# 層化サンプリング（ポジティブとネガティブのみ、各2,500件ずつ）
samples_per_class = SAMPLE_SIZE // 2  # 各クラス2,500件
print(f"\n層化サンプリング: POSITIVE/NEGATIVEのみ、各{samples_per_class}件ずつ")
print("（注: ニュートラルは除外し、明確なポジ/ネガの対比を分析）")

sampled_dfs = []
for label in ['POSITIVE', 'NEGATIVE']:
    df_label = raw_df_clean[raw_df_clean['tuned_model_label'] == label]
    n_sample = min(samples_per_class, len(df_label))
    sampled_dfs.append(df_label.sample(n=n_sample, random_state=42))
    print(f"  {label}: {n_sample}件サンプリング")

sample_df = pd.concat(sampled_dfs, ignore_index=True)
sample_texts = sample_df['自由記述'].astype(str).tolist()
sample_labels = sample_df['tuned_model_label'].tolist()

print(f"\n最終サンプルサイズ: {len(sample_texts)}件")
print(f"サンプルのラベル分布:")
for label in ['POSITIVE', 'NEGATIVE']:
    count = sample_labels.count(label)
    print(f"  {label}: {count}件 ({count/len(sample_labels)*100:.1f}%)")


# 予測関数（SHAP用）
def predict_proba(texts):
    """テキストのリストを受け取り、クラス確率を返す"""
    # SHAPから渡されるデータ型を処理
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    elif not isinstance(texts, list):
        try:
            texts = list(texts)
        except:
            texts = [str(texts)]
    
    # 空文字列や無効な入力を処理
    texts = [str(t) if t else "" for t in texts]
    
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    return probs.cpu().numpy()

# SHAP Explainerの作成
print("\n" + "="*60)
print("SHAP Explainerの作成")
print("="*60)
print("Explainerを初期化中...")
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict_proba, masker, algorithm="partition")

# サブワード統合関数
def merge_wordpieces(tokens, shap_vals_pos):
    """WordPieceのサブワード（##）を前の語に結合して集約する。
    戻り値: (merged_tokens, merged_shap_vals)
    """
    merged_tokens = []
    merged_vals = []
    current = ''
    current_val = 0.0
    for tok, val in zip(tokens, shap_vals_pos):
        t = str(tok)
        # 特殊トークンはスキップ
        if t in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
            continue
        if t.startswith('##'):
            # 連結（接頭の##を除去して前語に追加）
            current += t[2:]
            current_val += float(val)
        else:
            # 直前の語を確定
            if current:
                merged_tokens.append(current)
                merged_vals.append(current_val)
            current = t
            current_val = float(val)
    if current:
        merged_tokens.append(current)
        merged_vals.append(current_val)
    return merged_tokens, merged_vals

# SHAP分析（ストリーム集計）
print("\n" + "="*60)
print("SHAP分析開始（5,000件）")
print("="*60)

word_importance_sample = defaultdict(lambda: {'shap_values': [], 'count': 0})
batch_size = 64

for i in tqdm(range(0, len(sample_texts), batch_size), desc="SHAP集計"):
    bt = sample_texts[i:i+batch_size]
    sv_batch = explainer(bt)
    
    for sv in sv_batch:
        tokens = sv.data
        vals = sv.values
        
        # ポジティブクラス（index=2）のSHAP値を使用
        if len(vals.shape) > 1:
            vals_pos = vals[:, 2]
        else:
            vals_pos = vals
        
        # サブワード統合
        m_toks, m_vals = merge_wordpieces(tokens, vals_pos)
        
        for t, v in zip(m_toks, m_vals):
            if not t:
                continue
            word_importance_sample[t]['shap_values'].append(float(v))
            word_importance_sample[t]['count'] += 1

# DataFrame化（出現5回以上のみ）
print("\n単語統計を集計中...")
word_stats_sample = {
    w: {
        'mean_shap': float(np.mean(d['shap_values'])),
        'abs_mean_shap': float(np.mean(np.abs(d['shap_values']))),
        'std_shap': float(np.std(d['shap_values'])),
        'count': int(d['count'])
    }
    for w, d in word_importance_sample.items() if d['count'] >= 5
}
df_sample = pd.DataFrame(word_stats_sample).T.sort_values('mean_shap', ascending=False)

# 結果保存
print("\n" + "="*60)
print("結果を保存中")
print("="*60)

out_dir = "../03_分析結果/SHAP分析/サンプリング5000件"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(f"{out_dir}/可視化", exist_ok=True)

# CSV保存
csv_path = f"{out_dir}/word_importance_sample5000.csv"
df_sample.to_csv(csv_path, encoding='utf-8-sig')
print(f"✓ CSV保存: {csv_path}")

# JSON保存
json_path = f"{out_dir}/global_importance_sample5000.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump({
        'analysis_date': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'dataset_size': len(sample_texts),
        'model_path': MODEL_PATH,
        'sample_method': 'random_sampling',
        'random_state': 42,
        'top_positive_words': df_sample.head(50).to_dict('index'),
        'top_negative_words': df_sample.tail(50).to_dict('index')
    }, f, ensure_ascii=False, indent=2)
print(f"✓ JSON保存: {json_path}")

# TOP20可視化（ポジティブ）
print("\nTOP20グラフを作成中（ポジティブ）...")
top_positive = df_sample.head(20)
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_positive)), top_positive['mean_shap'], color='green', alpha=0.7)
plt.yticks(range(len(top_positive)), top_positive.index)
plt.xlabel('平均SHAP値（ポジティブ寄与）', fontsize=12)
plt.title('ポジティブ判定に寄与する重要語 TOP20（5,000件サンプル）', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
pos_path = f"{out_dir}/可視化/top20_positive_sample5000.png"
plt.savefig(pos_path, dpi=300, bbox_inches='tight')
print(f"✓ グラフ保存: {pos_path}")
plt.close()

# TOP20可視化（ネガティブ）
print("TOP20グラフを作成中（ネガティブ）...")
top_negative = df_sample.tail(20).iloc[::-1]
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_negative)), top_negative['mean_shap'], color='red', alpha=0.7)
plt.yticks(range(len(top_negative)), top_negative.index)
plt.xlabel('平均SHAP値（ネガティブ寄与）', fontsize=12)
plt.title('ネガティブ判定に寄与する重要語 TOP20（5,000件サンプル）', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
neg_path = f"{out_dir}/可視化/top20_negative_sample5000.png"
plt.savefig(neg_path, dpi=300, bbox_inches='tight')
print(f"✓ グラフ保存: {neg_path}")
plt.close()

# サマリーMarkdown作成
print("\nサマリーレポートを作成中...")
summary_md = f"""# SHAP分析サマリー（5,000件サンプリング）

**分析日時:** {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}  
**対象データ:** 約8.3万件から層化サンプリング5,000件（POSITIVE/NEGATIVEのみ）  
**モデル:** 単一タスクモデル1（感情スコア）  
**サンプリング方法:** 層化サンプリング（ポジティブ2,500件 + ネガティブ2,500件、random_state=42）  
**注:** ニュートラルを除外し、満足/不満の明確な対比を分析

---

## 📊 データ概要

- **総データ数:** {len(raw_df_clean):,}件
- **サンプル数:** {len(sample_texts):,}件（POSITIVE: {samples_per_class}件、NEGATIVE: {samples_per_class}件）
- **ニュートラル:** 除外（明確な満足/不満の対比を分析するため）
- **分析対象単語数:** {len(df_sample)}語（出現5回以上）

---

## 🔝 ポジティブ判定に寄与する重要語 TOP20

| 順位 | 単語 | 平均SHAP値 | 出現回数 |
|------|------|-----------|---------|
"""

for i, (word, row) in enumerate(df_sample.head(20).iterrows(), 1):
    summary_md += f"| {i} | {word} | {row['mean_shap']:.4f} | {row['count']} |\n"

summary_md += """
---

## 🔻 ネガティブ判定に寄与する重要語 TOP20

| 順位 | 単語 | 平均SHAP値 | 出現回数 |
|------|------|-----------|---------|
"""

for i, (word, row) in enumerate(df_sample.tail(20).iloc[::-1].iterrows(), 1):
    summary_md += f"| {i} | {word} | {row['mean_shap']:.4f} | {row['count']} |\n"

summary_md += f"""
---

## 📁 生成ファイル

- `word_importance_sample5000.csv` - 全単語の重要度データ（Excel用）
- `global_importance_sample5000.json` - JSON形式の集計結果
- `可視化/top20_positive_sample5000.png` - ポジティブTOP20グラフ
- `可視化/top20_negative_sample5000.png` - ネガティブTOP20グラフ

---

## 💡 主要な発見

### 満足度を高める要因（ポジティブTOP5）
{chr(10).join([f"- **{word}**: {row['mean_shap']:.4f}（{row['count']}回出現）" for word, row in df_sample.head(5).iterrows()])}

### 不満の原因（ネガティブTOP5）
{chr(10).join([f"- **{word}**: {row['mean_shap']:.4f}（{row['count']}回出現）" for word, row in df_sample.tail(5).iloc[::-1].iterrows()])}

---

**分析完了！**  
結果ファイルは `{out_dir}/` に保存されました。
"""

summary_path = f"{out_dir}/SHAP分析サマリー_sample5000.md"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_md)
print(f"✓ サマリー保存: {summary_path}")

print("\n" + "="*60)
print("SHAP分析完了（5,000件）！")
print("="*60)
print(f"\n結果は以下に保存されました:")
print(f"  {out_dir}/")
print(f"\nファイル一覧:")
print(f"  - word_importance_sample5000.csv")
print(f"  - global_importance_sample5000.json")
print(f"  - SHAP分析サマリー_sample5000.md")
print(f"  - 可視化/top20_positive_sample5000.png")
print(f"  - 可視化/top20_negative_sample5000.png")


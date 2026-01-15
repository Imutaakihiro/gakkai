#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
順序回帰モデル vs マルチタスクモデル SHAP分析比較

**作成日**: 2025年1月

比較内容:
1. 重要語ランキングの比較
2. 順序回帰特有の分析（P1～P4、期待値）
3. 共通する重要語と異なる重要語の特定
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'MS Gothic'

from train_class_level_ordinal_llp import CourseOrdinalLLPModel, BASE_MODEL
from transformers import BertJapaneseTokenizer

# ======================== 設定 ========================
MODEL_PATH = "../02_モデル/授業レベルマルチタスクモデル/class_level_ordinal_llp_20251030_162353.pth"
CSV_PATH = "../01_データ/マルチタスク用データ/授業集約データセット 回答分布付き.csv"
MULTITASK_SHAP_DIR = "../03_分析結果/マルチタスクSHAP分析_本番用"
OUTPUT_DIR = "../03_分析結果/順序回帰vs通常マルチタスク比較"

MAX_SAMPLES = 100  # SHAP分析のサンプル数
MAX_LENGTH = 192
BATCH_SIZE = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== デバイス設定 ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ======================== モデル読み込み ========================
print("モデル読み込み中...")
tokenizer = BertJapaneseTokenizer.from_pretrained(BASE_MODEL)
model = CourseOrdinalLLPModel(BASE_MODEL)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(device)
model.eval()
print("モデル読み込み完了")

# ======================== データ読み込み ========================
print("データ読み込み中...")
df = pd.read_csv(CSV_PATH)
texts = df['自由記述まとめ'].fillna("").astype(str).tolist()[:MAX_SAMPLES]
print(f"サンプル数: {len(texts)}")

# ======================== 予測関数の定義 ========================

def predict_probs(list_of_texts):
    """P1～P4の確率を予測"""
    if isinstance(list_of_texts, str):
        list_of_texts = [list_of_texts]
    all_probs = []
    for i in range(0, len(list_of_texts), BATCH_SIZE):
        batch = [str(x) if not isinstance(x, str) else x for x in list_of_texts[i:i+BATCH_SIZE]]
        encoding = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            chunk_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=device)
            _, _, P, _, _ = model(input_ids, attention_mask, chunk_mask)
            all_probs.extend(P.cpu().numpy().tolist())
    return np.array(all_probs)  # [N, 4]

def predict_expected_value(list_of_texts):
    """期待値 E[y] = 1*P1 + 2*P2 + 3*P3 + 4*P4 を予測"""
    probs = predict_probs(list_of_texts)
    expected = probs @ np.array([1, 2, 3, 4])
    return expected.reshape(-1, 1)

def predict_p1(list_of_texts):
    """P1（低評価確率）を予測"""
    probs = predict_probs(list_of_texts)
    return probs[:, 0:1]

def predict_p4(list_of_texts):
    """P4（高評価確率）を予測"""
    probs = predict_probs(list_of_texts)
    return probs[:, 3:4]

def predict_sentiment(list_of_texts):
    """感情スコアを予測"""
    if isinstance(list_of_texts, str):
        list_of_texts = [list_of_texts]
    pred = []
    for i in range(0, len(list_of_texts), BATCH_SIZE):
        batch = [str(x) if not isinstance(x, str) else x for x in list_of_texts[i:i+BATCH_SIZE]]
        encoding = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            chunk_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=device)
            _, _, _, y_sent, _ = model(input_ids, attention_mask, chunk_mask)
            pred.extend(y_sent.cpu().numpy().tolist())
    return np.array(pred).reshape(-1, 1)

def predict_course(list_of_texts):
    """授業評価スコアを予測"""
    if isinstance(list_of_texts, str):
        list_of_texts = [list_of_texts]
    pred = []
    for i in range(0, len(list_of_texts), BATCH_SIZE):
        batch = [str(x) if not isinstance(x, str) else x for x in list_of_texts[i:i+BATCH_SIZE]]
        encoding = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            chunk_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=device)
            _, _, _, _, y_course = model(input_ids, attention_mask, chunk_mask)
            pred.extend(y_course.cpu().numpy().tolist())
    return np.array(pred).reshape(-1, 1)

# ======================== SHAP分析実行 ========================

def run_shap_analysis(predict_fn, texts, name):
    """SHAP分析を実行し結果を返す"""
    print(f"\n=== SHAP分析: {name} ===")
    explainer = shap.Explainer(predict_fn, tokenizer)
    shap_values = explainer(texts)
    
    # 重要度を集計
    importance = np.abs(shap_values.values).mean(axis=0)
    if hasattr(shap_values, "feature_names") and shap_values.feature_names is not None:
        words = shap_values.feature_names
    else:
        words = [f"token_{i}" for i in range(len(importance))]
    
    # DataFrameに変換
    df_importance = pd.DataFrame({
        'word': words,
        'importance': importance.flatten() if importance.ndim > 1 else importance
    }).sort_values('importance', ascending=False)
    
    return shap_values, df_importance

def get_top_words(df, n=30):
    """上位n個の単語を取得"""
    return set(df.head(n)['word'].tolist())

# ======================== メイン処理 ========================

def main():
    print("=" * 60)
    print("順序回帰モデル vs マルチタスクモデル SHAP比較分析")
    print("=" * 60)
    
    results = {}
    
    # 1. 順序回帰モデルのSHAP分析
    print("\n【順序回帰モデルのSHAP分析】")
    
    # 期待値へのSHAP
    shap_expected, df_expected = run_shap_analysis(predict_expected_value, texts, "期待値E[y]")
    df_expected.to_csv(f"{OUTPUT_DIR}/ordinal_shap_expected_value.csv", index=False)
    results['ordinal_expected'] = df_expected
    
    # P1（低評価）へのSHAP
    shap_p1, df_p1 = run_shap_analysis(predict_p1, texts, "P1（低評価確率）")
    df_p1.to_csv(f"{OUTPUT_DIR}/ordinal_shap_p1.csv", index=False)
    results['ordinal_p1'] = df_p1
    
    # P4（高評価）へのSHAP
    shap_p4, df_p4 = run_shap_analysis(predict_p4, texts, "P4（高評価確率）")
    df_p4.to_csv(f"{OUTPUT_DIR}/ordinal_shap_p4.csv", index=False)
    results['ordinal_p4'] = df_p4
    
    # 感情スコア
    shap_sent, df_sent = run_shap_analysis(predict_sentiment, texts, "感情スコア（順序回帰モデル）")
    df_sent.to_csv(f"{OUTPUT_DIR}/ordinal_shap_sentiment.csv", index=False)
    results['ordinal_sentiment'] = df_sent
    
    # 授業評価スコア
    shap_course, df_course = run_shap_analysis(predict_course, texts, "授業評価スコア（順序回帰モデル）")
    df_course.to_csv(f"{OUTPUT_DIR}/ordinal_shap_course.csv", index=False)
    results['ordinal_course'] = df_course
    
    # 2. マルチタスクSHAP分析結果の読み込み
    print("\n【マルチタスクSHAP分析結果の読み込み】")
    multitask_sent = pd.read_csv(f"{MULTITASK_SHAP_DIR}/word_importance_sentiment_production.csv")
    multitask_course = pd.read_csv(f"{MULTITASK_SHAP_DIR}/word_importance_course_production.csv")
    results['multitask_sentiment'] = multitask_sent
    results['multitask_course'] = multitask_course
    print(f"マルチタスク感情スコア: {len(multitask_sent)}語")
    print(f"マルチタスク授業評価: {len(multitask_course)}語")
    
    # 3. 比較分析
    print("\n【比較分析】")
    
    # TOP30の重複率を計算
    comparisons = [
        ("順序回帰_期待値", df_expected, "マルチタスク_評価", multitask_course),
        ("順序回帰_感情", df_sent, "マルチタスク_感情", multitask_sent),
        ("順序回帰_P4(高評価)", df_p4, "マルチタスク_評価", multitask_course),
        ("順序回帰_P1(低評価)", df_p1, "マルチタスク_評価", multitask_course),
    ]
    
    comparison_results = []
    for name1, df1, name2, df2 in comparisons:
        top1 = get_top_words(df1, 30)
        top2 = get_top_words(df2, 30)
        overlap = top1 & top2
        overlap_rate = len(overlap) / 30 * 100
        
        result = {
            'comparison': f"{name1} vs {name2}",
            'overlap_count': len(overlap),
            'overlap_rate': overlap_rate,
            'overlap_words': list(overlap),
            'only_in_first': list(top1 - top2),
            'only_in_second': list(top2 - top1)
        }
        comparison_results.append(result)
        print(f"\n{name1} vs {name2}:")
        print(f"  重複: {len(overlap)}/30 ({overlap_rate:.1f}%)")
        print(f"  共通語: {list(overlap)[:10]}...")
    
    # 4. レポート作成
    print("\n【レポート作成】")
    
    report = f"""# 順序回帰 vs マルチタスク SHAP分析比較レポート

**作成日**: {datetime.now().strftime('%Y年%m月%d日')}

## 分析概要
- 順序回帰モデル: `class_level_ordinal_llp_20251030_162353.pth`
- マルチタスクモデル: 既存SHAP分析結果を使用
- サンプル数: {MAX_SAMPLES}件

## 順序回帰モデルの特徴
- **P1**: 評価1（全く意義なし）の確率
- **P2**: 評価2（あまり意義なし）の確率
- **P3**: 評価3（ある程度意義あり）の確率
- **P4**: 評価4（十分意義あり）の確率
- **期待値 E[y]**: 1×P1 + 2×P2 + 3×P3 + 4×P4

## 比較結果

"""
    
    for r in comparison_results:
        report += f"""### {r['comparison']}
- **重複率**: {r['overlap_count']}/30 ({r['overlap_rate']:.1f}%)
- **共通する重要語**: {', '.join(r['overlap_words'][:15])}{'...' if len(r['overlap_words']) > 15 else ''}
- **順序回帰のみ**: {', '.join(r['only_in_first'][:10])}{'...' if len(r['only_in_first']) > 10 else ''}
- **マルチタスクのみ**: {', '.join(r['only_in_second'][:10])}{'...' if len(r['only_in_second']) > 10 else ''}

"""
    
    # TOP10比較表
    report += """## 重要語TOP10比較

### 期待値E[y]（順序回帰）vs 授業評価スコア（マルチタスク）
| 順位 | 順序回帰（期待値） | マルチタスク（評価） |
|------|-------------------|---------------------|
"""
    for i in range(10):
        ord_word = df_expected.iloc[i]['word'] if i < len(df_expected) else "-"
        ord_imp = df_expected.iloc[i]['importance'] if i < len(df_expected) else 0
        mt_word = multitask_course.iloc[i]['word'] if i < len(multitask_course) else "-"
        mt_imp = multitask_course.iloc[i]['importance'] if i < len(multitask_course) else 0
        report += f"| {i+1} | {ord_word} ({ord_imp:.6f}) | {mt_word} ({mt_imp:.6f}) |\n"
    
    report += """
### P4（高評価確率）vs P1（低評価確率）
| 順位 | P4（高評価）重要語 | P1（低評価）重要語 |
|------|-------------------|-------------------|
"""
    for i in range(10):
        p4_word = df_p4.iloc[i]['word'] if i < len(df_p4) else "-"
        p4_imp = df_p4.iloc[i]['importance'] if i < len(df_p4) else 0
        p1_word = df_p1.iloc[i]['word'] if i < len(df_p1) else "-"
        p1_imp = df_p1.iloc[i]['importance'] if i < len(df_p1) else 0
        report += f"| {i+1} | {p4_word} ({p4_imp:.6f}) | {p1_word} ({p1_imp:.6f}) |\n"
    
    report += f"""
## 結論

順序回帰モデルとマルチタスクモデルの重要語ランキングを比較した結果:

1. **期待値 vs 授業評価スコア**: 重複率 {comparison_results[0]['overlap_rate']:.1f}%
2. **感情スコア**: 重複率 {comparison_results[1]['overlap_rate']:.1f}%
3. **P4（高評価）vs 授業評価**: 重複率 {comparison_results[2]['overlap_rate']:.1f}%

### 解釈
- 重複率が高い → 両モデルで同じ単語が重要視されている
- 重複率が低い → 順序回帰モデルが異なる特徴を捉えている可能性

---
**分析完了**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(f"{OUTPUT_DIR}/comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    # 結果をJSONで保存
    summary = {
        'timestamp': datetime.now().isoformat(),
        'samples': MAX_SAMPLES,
        'comparisons': [
            {
                'name': r['comparison'],
                'overlap_rate': r['overlap_rate'],
                'overlap_count': r['overlap_count']
            } for r in comparison_results
        ]
    }
    with open(f"{OUTPUT_DIR}/comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 完了！結果保存先: {OUTPUT_DIR}")
    print(f"  - comparison_report.md")
    print(f"  - ordinal_shap_expected_value.csv")
    print(f"  - ordinal_shap_p1.csv")
    print(f"  - ordinal_shap_p4.csv")
    print(f"  - ordinal_shap_sentiment.csv")
    print(f"  - ordinal_shap_course.csv")

if __name__ == "__main__":
    main()



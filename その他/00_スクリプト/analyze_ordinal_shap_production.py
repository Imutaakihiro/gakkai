#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
順序回帰モデル 本番用SHAP分析
マルチタスクSHAP分析と同じ形式で出力

**作成日**: 2025年1月

出力形式:
- word_importance_sentiment_production.csv (感情スコア)
- word_importance_course_production.csv (授業評価スコア)
- word_importance_expected_production.csv (期待値E[y])
- word_importance_p1_production.csv (P1: 低評価確率)
- word_importance_p2_production.csv (P2: 中低評価確率)
- word_importance_p3_production.csv (P3: 中高評価確率)
- word_importance_p4_production.csv (P4: 高評価確率)
- analysis_summary_production.json
- ordinal_shap_analysis_summary_production.md
- factor_categories_production.json
- 可視化PNGファイル
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TORCH_DISABLE_SAFETENSORS_WARNING'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from datetime import datetime
import shap

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "02_モデル", "授業レベルマルチタスクモデル", "class_level_ordinal_llp_20260114_101852.pth")
DATA_DIR = os.path.join(BASE_DIR, "01_データ", "マルチタスク用データ")
OUTPUT_DIR = os.path.join(BASE_DIR, "03_分析結果", "順序回帰SHAP分析_P2P4")

# CSV: 回答分布付きを優先、無ければ授業集約データセット_*.csv の最新を使用（SHAPは自由記述まとめのみ使用）
def _get_csv_path():
    preferred = os.path.join(DATA_DIR, "授業集約データセット 回答分布付き.csv")
    if os.path.exists(preferred):
        return preferred
    import glob
    pattern = os.path.join(DATA_DIR, "授業集約データセット_*.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        return preferred  # エラーは読み込み時に
    return max(candidates, key=os.path.getmtime)

CSV_PATH = _get_csv_path()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# デバイス選択（GPU最優先）
def get_device():
    """GPUを最優先で選択（CUDA → DirectML → CPU）"""
    # 1. CUDAを試す
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            _ = torch.tensor([1.0]).to(device)
            print(f"✅ CUDA使用: {torch.cuda.get_device_name(0)}")
            print(f"   GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return device
        except Exception as e:
            print(f"⚠️ CUDAエラー: {e}")
    
    # 2. DirectMLを試す
    try:
        import torch_directml as dml
        if dml.is_available():
            device = dml.device()
            print(f"✅ DirectML使用")
            return device
    except Exception:
        pass
    
    # 3. CPU（最後の手段）
    print("⚠️ GPUが見つかりません。CPUで実行します")
    return torch.device("cpu")

device = get_device()
print(f"使用デバイス: {device}")
print(f"PyTorch version: {torch.__version__}")

# モデル読み込み
print("="*60)
print("順序回帰モデル 本番用SHAP分析")
print("="*60)

from train_class_level_ordinal_llp import CourseOrdinalLLPModel, BASE_MODEL
from transformers import BertJapaneseTokenizer

print("📥 順序回帰モデルを読み込み中...")
tokenizer = BertJapaneseTokenizer.from_pretrained(BASE_MODEL)
model = CourseOrdinalLLPModel(BASE_MODEL)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(device)
model.eval()
print("✅ モデル読み込み完了")

# データ読み込み
print("📊 データ読み込み中...")
print(f"   CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
texts = df['自由記述まとめ'].fillna("").astype(str).tolist()
print(f"総データ数: {len(texts)}")

# サンプリング（層化サンプリング推奨だが、簡易版としてランダム）
SAMPLE_SIZE = 1000
if len(texts) > SAMPLE_SIZE:
    np.random.seed(42)
    sample_indices = np.random.choice(len(texts), SAMPLE_SIZE, replace=False)
    sample_texts = [texts[i] for i in sample_indices]
else:
    sample_texts = texts
print(f"分析サンプル数: {len(sample_texts)}")

# ======================== 予測関数 ========================

MAX_LENGTH = 192
BATCH_SIZE = 16

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
    return np.array(all_probs)

def predict_expected_value(list_of_texts):
    """期待値 E[y] = 1*P1 + 2*P2 + 3*P3 + 4*P4"""
    probs = predict_probs(list_of_texts)
    expected = probs @ np.array([1, 2, 3, 4])
    return expected.reshape(-1, 1)

def predict_p1(list_of_texts):
    """P1（低評価確率）"""
    probs = predict_probs(list_of_texts)
    return probs[:, 0:1]

def predict_p2(list_of_texts):
    """P2（中低評価確率）"""
    probs = predict_probs(list_of_texts)
    return probs[:, 1:2]

def predict_p3(list_of_texts):
    """P3（中高評価確率）"""
    probs = predict_probs(list_of_texts)
    return probs[:, 2:3]

def predict_p4(list_of_texts):
    """P4（高評価確率）"""
    probs = predict_probs(list_of_texts)
    return probs[:, 3:4]

def predict_sentiment(list_of_texts):
    """感情スコア"""
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
    """授業評価スコア（期待値から計算）"""
    # 期待値 E[y] = 1×P1 + 2×P2 + 3×P3 + 4×P4 を使用
    probs = predict_probs(list_of_texts)
    expected = probs @ np.array([1, 2, 3, 4])
    return expected.reshape(-1, 1)

# ======================== SHAP分析実行 ========================

def merge_wordpieces(tokens, shap_vals_pos):
    """WordPieceのサブワード（##）を前の語に結合して集約する。
    戻り値: (merged_tokens, merged_shap_vals)
    （analyze_sentiment_shap_5000.pyと同じ実装で整合性を確保）
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

def run_shap_analysis(predict_fn, texts, name, output_dir):
    """SHAP分析を実行（マルチタスク学習と同じ方法で統一）
    
    マルチタスク学習（analyze_classlevel_multitask_shap_beeswarm.py）と同じ計算方法：
    - importance = np.abs(shap_values.values).mean(axis=0)
    - WordPieceの結合は行わない（トークンレベルのまま）
    
    ただし、不規則な形状に対応するため、各サンプルごとに処理して集計
    """
    try:
        print(f"\n🔍 SHAP分析実行: {name}")
        print(f"   サンプル数: {len(texts)}件")
        print(f"   ⚠️ マルチタスク学習と同じ方法で計算（WordPiece結合なし）")
        
        explainer = shap.Explainer(predict_fn, tokenizer)
        shap_values = explainer(texts)
        
        # 不規則な形状に対応：各サンプルごとに処理
        # トークンごとのSHAP値を集計（WordPiece結合なし）
        token_importance_dict = defaultdict(lambda: {'shap_values': [], 'count': 0})
        
        # shap_valuesはExplanationオブジェクトで、各サンプルにアクセス可能
        if isinstance(shap_values, shap.Explanation):
            # 各サンプルを個別に処理
            for sv in shap_values:
                if hasattr(sv, 'values') and hasattr(sv, 'data'):
                    tokens = sv.data
                    vals = sv.values
                    
                    # 形状を確認して適切に処理
                    if hasattr(vals, 'ndim') and vals.ndim > 1:
                        # 回帰タスクの場合、valsの形状は(n_tokens, 1)または(n_tokens,)
                        if vals.shape[1] == 1:
                            vals_abs = np.abs(vals).flatten()  # (n_tokens, 1) -> (n_tokens,)
                        else:
                            vals_abs = np.abs(vals[:, 0])  # 最初の出力を使用
                    else:
                        vals_abs = np.abs(vals)
                    
                    # トークンとSHAP値を対応付け（WordPiece結合なし）
                    for token, val in zip(tokens, vals_abs):
                        if token and str(token).strip() and str(token) not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                            token_importance_dict[str(token)]['shap_values'].append(float(val))
                            token_importance_dict[str(token)]['count'] += 1
        
        # 各トークンごとの平均重要度を計算（マルチタスク学習と同じ方法）
        token_stats = {
            token: np.mean(data['shap_values'])
            for token, data in token_importance_dict.items()
            if data['count'] > 0
        }
        
        # DataFrameに変換（マルチタスク学習と同じ形式）
        df_importance = pd.DataFrame({
            'word': list(token_stats.keys()),
            'importance': list(token_stats.values())
        }).sort_values('importance', ascending=False)
        
        # 即座にCSV保存（エラー対策）
        csv_path = f"{output_dir}/word_importance_{name.lower().replace(' ', '_').replace('（', '').replace('）', '').replace('[', '').replace(']', '')}_production.csv"
        df_importance.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ {name} 完了: {len(df_importance)}語")
        print(f"   📁 結果を保存しました: {csv_path}")
        
        return shap_values, df_importance
        
    except Exception as e:
        print(f"❌ {name} のSHAP分析でエラーが発生しました: {e}")
        print(f"   💡 エラーの詳細: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        # エラーが発生しても続行（空のDataFrameを返す）
        return None, pd.DataFrame({'word': [], 'importance': []})

# SHAP分析実行（エラーハンドリング付き、各分析完了後に即座に保存）
print("\n" + "="*60)
print("SHAP分析実行中...")
print("="*60)
print("⚠️  各分析が完了次第、結果を即座に保存します")
print("   途中でエラーが発生しても、完了した分析結果は保存されます")
print("="*60)

shap_results = {}
completed_analyses = []  # 完了した分析を記録

# 各分析を個別に実行（エラーが発生しても続行）
# P2とP4のみ実行（教師の最重要課題）
analyses = [
    ("P2（中低評価確率）", predict_p2, "p2"),  # 2点を減らす要因（教師の最重要課題）
    ("P4（高評価確率）", predict_p4, "p4"),  # 4点を増やす要因
]

for name, predict_fn, key in analyses:
    try:
        shap_val, df_imp = run_shap_analysis(predict_fn, sample_texts, name, OUTPUT_DIR)
        # df_impが空でなければ成功（shap_valはNoneでも問題ない）
        if len(df_imp) > 0:
            shap_results[key] = {'shap': shap_val, 'df': df_imp}
            completed_analyses.append(key)
            print(f"✅ {name} の分析と保存が完了しました")
        else:
            print(f"⚠️  {name} の分析はスキップされました（エラーまたは空の結果）")
    except Exception as e:
        print(f"❌ {name} の分析で予期しないエラー: {e}")
        print(f"   次の分析を続行します...")
        continue

print(f"\n✅ 完了した分析: {len(completed_analyses)}/{len(analyses)}")
print(f"   完了リスト: {', '.join(completed_analyses)}")

# ======================== 結果保存 ========================

print("\n" + "="*60)
print("結果保存中...")
print("="*60)

# CSV保存（既に各分析で保存済み、ここではTOP100を保存）
print("\n📊 TOP100の保存中...")
for key, data in shap_results.items():
    if 'df' in data and len(data['df']) > 0:
        df = data['df']
        top100_path = f"{OUTPUT_DIR}/word_importance_{key}_top100_production.csv"
        df.head(100).to_csv(top100_path, index=False, encoding='utf-8')
        print(f"   ✅ {key} のTOP100を保存: {top100_path}")

print("✅ CSV保存完了")

# ======================== カテゴリ分類 ========================

def categorize_factors(sentiment_df, course_df, threshold=0.0005):
    """要因をカテゴリ分類（マルチタスク版と同じロジック）"""
    # 辞書に変換
    sent_dict = dict(zip(sentiment_df['word'], sentiment_df['importance']))
    course_dict = dict(zip(course_df['word'], course_df['importance']))
    
    all_words = set(sentiment_df['word']) | set(course_df['word'])
    
    categories = {
        'strong_common': [],      # 両方で高い
        'sentiment_leaning': [],  # 感情寄り
        'course_leaning': [],     # 評価寄り
        'sentiment_specific': [], # 感情特化
        'course_specific': []     # 評価特化
    }
    
    for word in all_words:
        sent_imp = sent_dict.get(word, 0)
        course_imp = course_dict.get(word, 0)
        
        if sent_imp >= threshold and course_imp >= threshold:
            categories['strong_common'].append((word, sent_imp, course_imp))
        elif sent_imp >= threshold and course_imp < threshold * 0.5:
            categories['sentiment_specific'].append((word, sent_imp, course_imp))
        elif course_imp >= threshold and sent_imp < threshold * 0.5:
            categories['course_specific'].append((word, sent_imp, course_imp))
        elif sent_imp >= threshold * 0.5 and course_imp >= threshold * 0.5:
            if sent_imp > course_imp * 1.5:
                categories['sentiment_leaning'].append((word, sent_imp, course_imp))
            elif course_imp > sent_imp * 1.5:
                categories['course_leaning'].append((word, sent_imp, course_imp))
    
    # ソート
    for cat in categories:
        categories[cat].sort(key=lambda x: x[1] + x[2], reverse=True)
    
    return categories

# カテゴリ分類（P2とP4の比較）
if 'p2' in shap_results and 'p4' in shap_results:
    df_p2 = shap_results['p2']['df']
    df_p4 = shap_results['p4']['df']
    # P2とP4の比較用カテゴリ分類
    p2_dict = dict(zip(df_p2['word'], df_p2['importance']))
    p4_dict = dict(zip(df_p4['word'], df_p4['importance']))
    all_words = set(df_p2['word']) | set(df_p4['word'])
    threshold = 0.0005
    
    categories = {
        'strong_common': [],      # P2とP4の両方で高い
        'p2_specific': [],        # P2特化（2点を減らす要因）
        'p4_specific': [],        # P4特化（4点を増やす要因）
        'p2_leaning': [],         # P2寄り
        'p4_leaning': []          # P4寄り
    }
    
    for word in all_words:
        p2_imp = p2_dict.get(word, 0)
        p4_imp = p4_dict.get(word, 0)
        
        if p2_imp >= threshold and p4_imp >= threshold:
            categories['strong_common'].append((word, p2_imp, p4_imp))
        elif p2_imp >= threshold and p4_imp < threshold * 0.5:
            categories['p2_specific'].append((word, p2_imp, p4_imp))
        elif p4_imp >= threshold and p2_imp < threshold * 0.5:
            categories['p4_specific'].append((word, p2_imp, p4_imp))
        elif p2_imp >= threshold * 0.5 and p4_imp >= threshold * 0.5:
            if p2_imp > p4_imp * 1.5:
                categories['p2_leaning'].append((word, p2_imp, p4_imp))
            elif p4_imp > p2_imp * 1.5:
                categories['p4_leaning'].append((word, p2_imp, p4_imp))
    
    # ソート
    for cat in categories:
        categories[cat].sort(key=lambda x: x[1] + x[2], reverse=True)
    print("✅ P2とP4のカテゴリ分類完了")
elif 'sentiment' in shap_results and 'course' in shap_results:
    df_sent = shap_results['sentiment']['df']
    df_course = shap_results['course']['df']
    categories = categorize_factors(df_sent, df_course)
else:
    print("⚠️  P2とP4、または感情スコアと授業評価スコアの分析が完了していないため、カテゴリ分類をスキップします")
    categories = {
        'strong_common': [],
        'sentiment_leaning': [],
        'course_leaning': [],
        'sentiment_specific': [],
        'course_specific': []
    }

# JSON保存
categories_json = {}
for category, items in categories.items():
    if 'p2' in shap_results and 'p4' in shap_results:
        # P2とP4の比較用
        categories_json[category] = [
            {'word': word, 'p2_importance': p2_imp, 'p4_importance': p4_imp}
            for word, p2_imp, p4_imp in items
        ]
    else:
        # 感情スコアと授業評価スコアの比較用
        categories_json[category] = [
            {'word': word, 'sentiment_importance': s_imp, 'course_importance': c_imp}
            for word, s_imp, c_imp in items
        ]

with open(f"{OUTPUT_DIR}/factor_categories_production.json", 'w', encoding='utf-8') as f:
    json.dump(categories_json, f, ensure_ascii=False, indent=2)

print("✅ カテゴリ分類完了")

# ======================== 可視化 ========================

print("\n📊 可視化作成中...")

# 可視化（完了した分析のみ）
# 1. 感情スコアTOP30
if 'sentiment' in shap_results:
    df_sent = shap_results['sentiment']['df']
    plt.figure(figsize=(12, 8))
    top30_sent = df_sent.head(30)
    plt.barh(range(len(top30_sent)), top30_sent['importance'].values[::-1])
    plt.yticks(range(len(top30_sent)), top30_sent['word'].values[::-1])
    plt.xlabel('重要度')
    plt.title('感情スコア予測 重要語TOP30（順序回帰モデル）', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sentiment_top30_factors_production.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 感情スコアTOP30グラフを作成しました")

# 2. 授業評価スコアTOP30
if 'course' in shap_results:
    df_course = shap_results['course']['df']
    plt.figure(figsize=(12, 8))
    top30_course = df_course.head(30)
    plt.barh(range(len(top30_course)), top30_course['importance'].values[::-1])
    plt.yticks(range(len(top30_course)), top30_course['word'].values[::-1])
    plt.xlabel('重要度')
    plt.title('授業評価スコア予測 重要語TOP30（順序回帰モデル）', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/course_top30_factors_production.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 授業評価スコアTOP30グラフを作成しました")

# 3. 期待値TOP30
if 'expected' in shap_results:
    df_expected = shap_results['expected']['df']
    plt.figure(figsize=(12, 8))
    top30_expected = df_expected.head(30)
    plt.barh(range(len(top30_expected)), top30_expected['importance'].values[::-1])
    plt.yticks(range(len(top30_expected)), top30_expected['word'].values[::-1])
    plt.xlabel('重要度')
    plt.title('期待値E[y]予測 重要語TOP30（順序回帰モデル）', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/expected_top30_factors_production.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 期待値TOP30グラフを作成しました")

# 4-7. P1～P4の可視化（完了した分析のみ）
for key, label in [('p1', 'P1（低評価確率）'), ('p2', 'P2（中低評価確率）'), 
                   ('p3', 'P3（中高評価確率）'), ('p4', 'P4（高評価確率）')]:
    if key in shap_results:
        df = shap_results[key]['df']
        plt.figure(figsize=(12, 8))
        top30 = df.head(30)
        plt.barh(range(len(top30)), top30['importance'].values[::-1])
        plt.yticks(range(len(top30)), top30['word'].values[::-1])
        plt.xlabel('重要度')
        plt.title(f'{label}予測 重要語TOP30（順序回帰モデル）', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{key}_top30_factors_production.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {label}TOP30グラフを作成しました")

# 8. P2とP4の比較グラフ（教師の最重要課題）
if 'p2' in shap_results and 'p4' in shap_results:
    df_p2 = shap_results['p2']['df']
    df_p4 = shap_results['p4']['df']
    plt.figure(figsize=(14, 10))
    top20_p2 = df_p2.head(20)
    top20_p4 = df_p4.head(20)
    x_pos = np.arange(len(top20_p2))
    width = 0.35
    plt.barh(x_pos - width/2, top20_p2['importance'].values[::-1], width, label='P2（中低評価：2点を減らす要因）', color='#ff6b6b')
    plt.barh(x_pos + width/2, top20_p4['importance'].values[::-1], width, label='P4（高評価：4点を増やす要因）', color='#4ecdc4')
    plt.yticks(x_pos, top20_p2['word'].values[::-1])
    plt.xlabel('重要度')
    plt.title('P2とP4の重要語比較 TOP20（教師の最重要課題）', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/p2_p4_comparison_production.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ P2とP4比較グラフを作成しました")
elif all(k in shap_results for k in ['p1', 'p2', 'p3', 'p4']):
    # P1～P4 比較（全確率分布、すべて完了している場合のみ）
    df_p1 = shap_results['p1']['df']
    df_p2 = shap_results['p2']['df']
    df_p3 = shap_results['p3']['df']
    df_p4 = shap_results['p4']['df']
    plt.figure(figsize=(14, 10))
    top20_p1 = df_p1.head(20)
    top20_p2 = df_p2.head(20)
    top20_p3 = df_p3.head(20)
    top20_p4 = df_p4.head(20)
    x_pos = np.arange(len(top20_p1))
    width = 0.2
    plt.barh(x_pos - width*1.5, top20_p1['importance'].values[::-1], width, label='P1（低評価）')
    plt.barh(x_pos - width*0.5, top20_p2['importance'].values[::-1], width, label='P2（中低評価）')
    plt.barh(x_pos + width*0.5, top20_p3['importance'].values[::-1], width, label='P3（中高評価）')
    plt.barh(x_pos + width*1.5, top20_p4['importance'].values[::-1], width, label='P4（高評価）')
    plt.yticks(x_pos, top20_p1['word'].values[::-1])
    plt.xlabel('重要度')
    plt.title('P1～P4（全確率分布）重要語比較 TOP20', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/p1_p2_p3_p4_comparison_production.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ P1～P4比較グラフを作成しました")
else:
    print("⚠️  P2とP4、またはP1～P4のすべての分析が完了していないため、比較グラフをスキップします")

# 9. カテゴリ別チャート

# 5. カテゴリ別チャート
plt.figure(figsize=(10, 6))
cat_counts = {k: len(v) for k, v in categories.items()}
plt.bar(cat_counts.keys(), cat_counts.values())
plt.ylabel('要因数')
plt.title('要因カテゴリ別分布（順序回帰モデル）', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/factor_categories_chart_production.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ 可視化完了")

# ======================== サマリーレポート ========================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# JSONサマリー（完了した分析のみを含める）
summary = {
    "analysis_date": timestamp,
    "device_used": str(device),
    "pytorch_version": torch.__version__,
    "method": "順序回帰モデルSHAP分析",
    "sample_size": len(sample_texts),
    "completed_analyses": completed_analyses,
    "category_counts": {k: len(v) for k, v in categories.items()},
}

# 完了した分析の情報を追加
for key in ['sentiment', 'course', 'expected', 'p1', 'p2', 'p3', 'p4']:
    if key in shap_results:
        df = shap_results[key]['df']
        summary[f"total_words_{key}"] = len(df)
        summary[f"top_{key}_factors"] = dict(df.head(20).values) if len(df) > 0 else {}

# 共通要因（P2とP4の比較、または感情スコアと授業評価スコアの比較）
if 'p2' in shap_results and 'p4' in shap_results:
    df_p2 = shap_results['p2']['df']
    df_p4 = shap_results['p4']['df']
    summary["common_words_count"] = len(set(df_p2['word']) & set(df_p4['word']))
    summary["strong_common_factors"] = [
        {"word": word, "p2_importance": p2_imp, "p4_importance": p4_imp}
        for word, p2_imp, p4_imp in categories.get('strong_common', [])[:20]
    ]
elif 'sentiment' in shap_results and 'course' in shap_results:
    df_sent = shap_results['sentiment']['df']
    df_course = shap_results['course']['df']
    summary["common_words_count"] = len(set(df_sent['word']) & set(df_course['word']))
    summary["strong_common_factors"] = [
        {"word": word, "sentiment": s_imp, "course": c_imp}
        for word, s_imp, c_imp in categories.get('strong_common', [])[:20]
    ]

with open(f"{OUTPUT_DIR}/analysis_summary_production.json", 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# マークダウンレポート
report = f"""# 順序回帰モデル SHAP分析結果サマリー

**作成日**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 分析概要
- 分析日時: {timestamp}
- 分析対象: 順序回帰モデル（CORAL型、LLP損失）
- サンプル数: {len(sample_texts)}件
- 使用デバイス: {device}
- PyTorch version: {torch.__version__}

## 分析結果サマリー
- 完了した分析: {len(completed_analyses)}/{len(analyses)}件
- 完了リスト: {', '.join(completed_analyses) if completed_analyses else 'なし'}
"""
# 完了した分析の要因数を追加
for key, name in [('sentiment', '感情スコア'), ('course', '授業評価スコア'), ('expected', '期待値E[y]'), 
                  ('p1', 'P1（低評価確率）'), ('p2', 'P2（中低評価確率）'), 
                  ('p3', 'P3（中高評価確率）'), ('p4', 'P4（高評価確率）')]:
    if key in shap_results:
        report += f"- {name}予測要因数: {len(shap_results[key]['df'])}単語\n"

# 共通要因数（P2とP4の比較、または感情スコアと授業評価スコアの比較）
if 'p2' in shap_results and 'p4' in shap_results:
    df_p2 = shap_results['p2']['df']
    df_p4 = shap_results['p4']['df']
    report += f"- 共通要因数: {len(set(df_p2['word']) & set(df_p4['word']))}単語\n"
    report += f"""- 強い共通要因数: {len(categories.get('strong_common', []))}単語

## カテゴリ別要因数（P2とP4の比較）

### 強い共通要因（P2とP4の両方で高い） ({len(categories.get('strong_common', []))}件)
| 順位 | 単語 | P2重要度（2点を減らす） | P4重要度（4点を増やす） | 総合重要度 |
|------|------|----------------------|----------------------|------------|
"""
    for i, (word, p2_imp, p4_imp) in enumerate(categories.get('strong_common', [])[:20], 1):
        report += f"| {i} | {word} | {p2_imp:.6f} | {p4_imp:.6f} | {p2_imp + p4_imp:.6f} |\n"
    
    report += f"""
### P2特化要因（2点を減らす要因） ({len(categories.get('p2_specific', []))}件)
| 順位 | 単語 | P2重要度 | P4重要度 |
|------|------|----------|----------|
"""
    for i, (word, p2_imp, p4_imp) in enumerate(categories.get('p2_specific', [])[:15], 1):
        report += f"| {i} | {word} | {p2_imp:.6f} | {p4_imp:.6f} |\n"
    
    report += f"""
### P4特化要因（4点を増やす要因） ({len(categories.get('p4_specific', []))}件)
| 順位 | 単語 | P2重要度 | P4重要度 |
|------|------|----------|----------|
"""
    for i, (word, p2_imp, p4_imp) in enumerate(categories.get('p4_specific', [])[:15], 1):
        report += f"| {i} | {word} | {p2_imp:.6f} | {p4_imp:.6f} |\n"
elif 'sentiment' in shap_results and 'course' in shap_results:
    df_sent = shap_results['sentiment']['df']
    df_course = shap_results['course']['df']
    report += f"- 共通要因数: {len(set(df_sent['word']) & set(df_course['word']))}単語\n"
    report += f"""- 強い共通要因数: {len(categories.get('strong_common', []))}単語

## カテゴリ別要因数

### 強い共通要因 ({len(categories.get('strong_common', []))}件)
| 順位 | 単語 | 感情重要度 | 評価重要度 | 総合重要度 |
|------|------|------------|------------|------------|
"""
    for i, (word, s_imp, c_imp) in enumerate(categories.get('strong_common', [])[:20], 1):
        report += f"| {i} | {word} | {s_imp:.6f} | {c_imp:.6f} | {s_imp + c_imp:.6f} |\n"
    
    report += f"""
### 感情特化要因 ({len(categories.get('sentiment_specific', []))}件)
| 順位 | 単語 | 感情重要度 | 評価重要度 |
|------|------|------------|------------|
"""
    for i, (word, s_imp, c_imp) in enumerate(categories.get('sentiment_specific', [])[:15], 1):
        report += f"| {i} | {word} | {s_imp:.6f} | {c_imp:.6f} |\n"
    
    report += f"""
### 評価特化要因 ({len(categories.get('course_specific', []))}件)
| 順位 | 単語 | 感情重要度 | 評価重要度 |
|------|------|------------|------------|
"""
    for i, (word, s_imp, c_imp) in enumerate(categories.get('course_specific', [])[:15], 1):
        report += f"| {i} | {word} | {s_imp:.6f} | {c_imp:.6f} |\n"
else:
    report += "\n## カテゴリ分類\n\nカテゴリ分類は利用できません（P2とP4、または感情スコアと授業評価スコアの分析が必要です）。\n"

# 完了した分析のTOP10をレポートに追加
for key, title in [('expected', '期待値E[y]'), ('p1', 'P1（低評価確率）'), 
                   ('p2', 'P2（中低評価確率）'), ('p3', 'P3（中高評価確率）'), 
                   ('p4', 'P4（高評価確率）')]:
    if key in shap_results:
        df = shap_results[key]['df']
        report += f"""
## {title} TOP10重要語
| 順位 | 単語 | 重要度 |
|------|------|--------|
"""
        for i, row in df.head(10).iterrows():
            report += f"| {i+1} | {row['word']} | {row['importance']:.6f} |\n"

report += """
## 主要な発見

### 1. 順序回帰モデルの特徴（P2とP4の分析）
- **P2（中低評価確率）へのSHAP分析**: 2点を減らす要因を特定
- **P4（高評価確率）へのSHAP分析**: 4点を増やす要因を特定
- 平均3点の改善戦略として、2点を減らし、4点を増やす要因を明確に分離可能

### 2. カテゴリ分類（P2とP4の比較）
- **強い共通要因**: P2とP4の両方で高い要因（2点を減らし、4点を増やす両方に影響）
- **P2特化要因**: 2点を減らす要因（教師の最重要課題）
- **P4特化要因**: 4点を増やす要因
- **P2寄り要因**: P2により強く影響する要因
- **P4寄り要因**: P4により強く影響する要因

### 3. 授業改善への示唆（教師の視点：平均3点の改善戦略）
1. **優先度1（最重要）**: P2特化要因の回避・改善（2点を減らす）
2. **優先度2**: P4特化要因への投資（4点を増やす）
3. **優先度3**: 強い共通要因を重視した授業改善（2点を減らし、4点を増やす両方に効果）
4. **優先度4**: P2寄り要因とP4寄り要因のバランスを考慮した改善

---
**分析完了**: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S')

with open(f"{OUTPUT_DIR}/ordinal_shap_analysis_summary_production.md", 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ サマリーレポート作成完了")

print("\n" + "="*60)
print("✅ 全ての分析完了！")
print(f"📁 結果保存先: {OUTPUT_DIR}")
print("="*60)




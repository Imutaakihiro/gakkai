#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
順序回帰モデル SHAP分析 テスト実行版
サンプル数10件で動作確認
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
CSV_PATH = os.path.join(BASE_DIR, "01_データ", "マルチタスク用データ", "授業集約データセット 回答分布付き.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "03_分析結果", "順序回帰SHAP分析_テスト実行")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# デバイス選択（GPU最優先）
def get_device():
    """GPUを最優先で選択（CUDA → DirectML → CPU）"""
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            _ = torch.tensor([1.0]).to(device)
            print(f"✅ CUDA使用: {torch.cuda.get_device_name(0)}")
            print(f"   GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return device
        except Exception as e:
            print(f"⚠️ CUDAエラー: {e}")
    
    try:
        import torch_directml as dml
        if dml.is_available():
            device = dml.device()
            print(f"✅ DirectML使用")
            return device
    except Exception:
        pass
    
    print("⚠️ GPUが見つかりません。CPUで実行します")
    return torch.device("cpu")

device = get_device()
print(f"使用デバイス: {device}")
print(f"PyTorch version: {torch.__version__}")

# モデル読み込み
print("="*60)
print("順序回帰モデル SHAP分析 テスト実行（10件）")
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

# データ読み込み（テスト用：10件のみ）
print("\n📊 データ読み込み中...")
df = pd.read_csv(CSV_PATH)
texts = df['自由記述まとめ'].fillna("").astype(str).tolist()
print(f"総データ数: {len(texts)}")

# テスト用：10件のみ
TEST_SAMPLE_SIZE = 10
np.random.seed(42)
sample_indices = np.random.choice(len(texts), min(TEST_SAMPLE_SIZE, len(texts)), replace=False)
sample_texts = [texts[i] for i in sample_indices]
print(f"🧪 テストサンプル数: {len(sample_texts)}件")

# ======================== 予測関数 ========================

MAX_LENGTH = 192
BATCH_SIZE = 16

def predict_probs(list_of_texts):
    """P1～P4の確率を予測（analyze_sentiment_shap_5000.pyと同じ形式）"""
    # SHAPから渡されるデータ型を処理（analyze_sentiment_shap_5000.pyと同じ）
    if isinstance(list_of_texts, str):
        list_of_texts = [list_of_texts]
    elif isinstance(list_of_texts, np.ndarray):
        list_of_texts = list_of_texts.tolist()
    elif not isinstance(list_of_texts, list):
        try:
            list_of_texts = list(list_of_texts)
        except:
            list_of_texts = [str(list_of_texts)]
    
    # 空文字列や無効な入力を処理
    list_of_texts = [str(t) if t else "" for t in list_of_texts]
    
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

def predict_sentiment(list_of_texts):
    """感情スコア予測（回帰ヘッドから取得）"""
    if isinstance(list_of_texts, str):
        list_of_texts = [list_of_texts]
    elif isinstance(list_of_texts, np.ndarray):
        list_of_texts = list_of_texts.tolist()
    elif not isinstance(list_of_texts, list):
        try:
            list_of_texts = list(list_of_texts)
        except:
            list_of_texts = [str(list_of_texts)]
    
    list_of_texts = [str(t) if t else "" for t in list_of_texts]
    
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
    """授業評価スコア予測（期待値E[y]を使用）"""
    probs = predict_probs(list_of_texts)
    expected = probs @ np.array([1, 2, 3, 4])
    return expected.reshape(-1, 1)

def predict_p2(list_of_texts):
    """P2（中低評価確率）"""
    probs = predict_probs(list_of_texts)
    return probs[:, 1:2]

def predict_p4(list_of_texts):
    """P4（高評価確率）"""
    probs = predict_probs(list_of_texts)
    return probs[:, 3:4]

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

def convert_token_ids_to_words(token_ids):
    """トークンIDを実際の単語に変換"""
    if isinstance(token_ids, (list, np.ndarray)):
        # トークンIDのリストの場合
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        return tokens
    else:
        # 既に文字列の場合
        return [str(token_ids)]

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
        csv_path = f"{output_dir}/word_importance_{name.lower().replace(' ', '_').replace('（', '').replace('）', '').replace('[', '').replace(']', '')}_test.csv"
        df_importance.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ {name} 完了: {len(df_importance)}語")
        print(f"   📁 結果を保存しました: {csv_path}")
        
        return None, df_importance
        
    except Exception as e:
        print(f"❌ {name} のSHAP分析でエラーが発生しました: {e}")
        print(f"   💡 エラーの詳細: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        # エラーが発生しても続行（空のDataFrameを返す）
        return None, pd.DataFrame({'word': [], 'importance': []})

# SHAP分析実行（テスト版：P2とP4を含む）
print("\n" + "="*60)
print("SHAP分析実行（テスト版：4種類）")
print("="*60)

analyses = [
    ("感情スコア", predict_sentiment, "sentiment"),
    ("授業評価スコア", predict_course, "course"),
    ("P2（中低評価確率）", predict_p2, "p2"),  # 2点を減らす要因（教師の最重要課題）
    ("P4（高評価確率）", predict_p4, "p4"),  # 4点を増やす要因
]

shap_results = {}
completed_analyses = []

for name, predict_fn, key in analyses:
    try:
        print(f"\n{'='*60}")
        print(f"分析: {name}")
        print(f"{'='*60}")
        
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
        import traceback
        traceback.print_exc()
        print(f"   次の分析を続行します...")
        continue

# テスト結果サマリー
print("\n" + "="*60)
print("テスト実行結果サマリー")
print("="*60)
print(f"✅ 完了した分析: {len(completed_analyses)}/{len(analyses)}")
print(f"   {completed_analyses}")

if len(completed_analyses) > 0:
    print("\n📊 各分析の結果:")
    for key in completed_analyses:
        df = shap_results[key]['df']
        print(f"   - {key}: {len(df)}語")
        if len(df) > 0:
            print(f"     上位5語: {', '.join(df.head(5)['word'].tolist())}")
    
    print(f"\n✅ テスト実行成功！本番実行に進めます。")
    print(f"   出力先: {OUTPUT_DIR}")
else:
    print("\n❌ すべての分析でエラーが発生しました。")
    print("   エラーログを確認してください。")


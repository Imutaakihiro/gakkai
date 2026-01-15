#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
授業単位マルチタスクモデル用 SHAP beeswarm分析・可視化
2つの目的関数（感情スコア平均・授業評価スコア）それぞれでSHAP・beeswarmプロット
"""

import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import matplotlib.pyplot as plt

# SHAPのインポート（エラーハンドリング）
try:
    import shap
except ImportError as e:
    print(f"❌ SHAPインポートエラー: {e}")
    print("💡 以下のコマンドで修正してください:")
    print("   python fix_shap_dependencies.py")
    sys.exit(1)

# エラーハンドリング: PyTorchのインポート問題を回避
try:
    from train_class_level_ordinal_llp import CourseOrdinalLLPModel, BASE_MODEL
    from transformers import BertJapaneseTokenizer
except ImportError as e:
    print(f"❌ モデルインポートエラー: {e}")
    print("\n💡 解決方法:")
    print("   1. NumPyを1.x系にダウングレード:")
    print("      python safe_fix_for_shap.py")
    print("   2. または、PyTorchをアップグレード（時間がかかります）")
    sys.exit(1)

import glob
import os
import sys

# パス設定・パラメータ（必要に応じて書き換え・コマンド引数化可）
MODEL_PATH = "C:/Users/takahashi.Jupiter/Desktop/卒業研究（新）/02_モデル/授業レベルマルチタスクモデル/class_level_ordinal_llp_20251030_162353.pth"
print(f"使用モデル重み: {MODEL_PATH}")
CSV_PATH = "C:/Users/takahashi.Jupiter/Desktop/卒業研究（新）/01_データ/マルチタスク用データ/授業集約データセット 回答分布付き.csv"
OUTPUT_DIR = "03_分析結果/クラスレベルSHAP_Beeswarm"
BATCH = 128  # GPU使用率最大化のためバッチサイズを大幅増加
MAX_SAMPLES = 50
MAX_LENGTH = 192

os.makedirs(OUTPUT_DIR, exist_ok=True)

# デバイス選択（DirectML対応）
def get_device():
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0]).cuda()
            print("✅ CUDA 利用")
            return torch.device("cuda")
        except Exception:
            pass
    try:
        import torch_directml as dml
        if dml.is_available():
            print("✅ DirectML 利用")
            return dml.device()
    except Exception:
        pass
    print("🔄 CPU 利用")
    return torch.device("cpu")

device = get_device()

# 1. モデル/トークナイザー準備
tokenizer = BertJapaneseTokenizer.from_pretrained(BASE_MODEL)
model = CourseOrdinalLLPModel(BASE_MODEL)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(device)
model.eval()

# 2. データ読み込み（修正）
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
texts = df['自由記述まとめ'].fillna("").astype(str).tolist()[:MAX_SAMPLES]

print(f"Loaded {len(texts)} samples.")

def predict_sentiment(list_of_texts):
    """感情スコア予測（GPU最優先・最適化版）"""
    if isinstance(list_of_texts, str):
        list_of_texts = [list_of_texts]
    pred = []
    
    # GPUメモリ最適化
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif hasattr(device, 'empty_cache'):
        device.empty_cache()
    
    model.eval()  # 推論モードを明示
    
    with torch.no_grad():  # 勾配計算を無効化（メモリ節約）
        for i in range(0, len(list_of_texts), BATCH):
            batch = [str(x) if not isinstance(x, str) else x for x in list_of_texts[i:i+BATCH]]
            
            # トークナイズ（CPU）
            encoding = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            
            # GPUに非同期転送（高速化）
            input_ids = encoding['input_ids'].to(device, non_blocking=True)
            attention_mask = encoding['attention_mask'].to(device, non_blocking=True)
            chunk_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=device)
            
            # GPUで推論実行
            out = model(input_ids, attention_mask, chunk_mask)
            y_sent_pred = out[3]  # GPU上で保持
            
            # 結果のみCPUに転送（最小限の転送）
            pred.extend(y_sent_pred.cpu().numpy().tolist())
    
    return np.array(pred).reshape(-1, 1)

def predict_course(list_of_texts):
    """授業評価スコア予測（GPU最優先・最適化版）"""
    if isinstance(list_of_texts, str):
        list_of_texts = [list_of_texts]
    pred = []
    
    # GPUメモリ最適化
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif hasattr(device, 'empty_cache'):
        device.empty_cache()
    
    model.eval()  # 推論モードを明示
    
    with torch.no_grad():  # 勾配計算を無効化（メモリ節約）
        for i in range(0, len(list_of_texts), BATCH):
            batch = [str(x) if not isinstance(x, str) else x for x in list_of_texts[i:i+BATCH]]
            
            # トークナイズ（CPU）
            encoding = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            
            # GPUに非同期転送（高速化）
            input_ids = encoding['input_ids'].to(device, non_blocking=True)
            attention_mask = encoding['attention_mask'].to(device, non_blocking=True)
            chunk_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=device)
            
            # GPUで推論実行
            out = model(input_ids, attention_mask, chunk_mask)
            y_course_pred = out[4]  # GPU上で保持
            
            # 結果のみCPUに転送（最小限の転送）
            pred.extend(y_course_pred.cpu().numpy().tolist())
    
    return np.array(pred).reshape(-1, 1)

print("\n=== SHAP(感情スコア)解析・可視化 ===")
print(f"使用デバイス: {device}")
print(f"バッチサイズ: {BATCH} (GPU使用率最大化)")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPUメモリ: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# SHAPの並列処理を調整（GPU使用を促進）
print("SHAP Explainer作成中（GPU推論を使用）...")
explainer_sent = shap.Explainer(predict_sentiment, tokenizer)
print("SHAP値計算中（GPUで推論実行）...")
shap_values_sent = explainer_sent(texts)
plt.figure(figsize=(14, 8))
shap.summary_plot(shap_values_sent, texts, show=False)
plt.title("感情スコア予測SHAP Beeswarm", fontsize=16)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm_sentiment.png", dpi=300)
plt.close()

print("\n=== SHAP(授業評価スコア)解析・可視化 ===")
print(f"使用デバイス: {device}")
print(f"バッチサイズ: {BATCH} (GPU使用率最大化)")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPUメモリ: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# SHAPの並列処理を調整（GPU使用を促進）
print("SHAP Explainer作成中（GPU推論を使用）...")
explainer_course = shap.Explainer(predict_course, tokenizer)
print("SHAP値計算中（GPUで推論実行）...")
shap_values_course = explainer_course(texts)
plt.figure(figsize=(14, 8))
shap.summary_plot(shap_values_course, texts, show=False)
plt.title("授業評価スコア予測SHAP Beeswarm", fontsize=16)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm_course.png", dpi=300)
plt.close()

# 重要度ランキングもCSVで保存
def get_topwords(shap_values, texts, n=30):
    importance = np.abs(shap_values.values).mean(axis=0)
    # tokenizer系は特徴名の整合性に注意
    words = shap_values.feature_names if hasattr(shap_values, "feature_names") else list(range(len(importance)))
    idx = np.argsort(importance)[::-1][:n]
    return [(words[i], importance[i]) for i in idx]

sent_top30 = get_topwords(shap_values_sent, texts, n=30)
pd.DataFrame(sent_top30, columns=["word","importance"]).to_csv(f"{OUTPUT_DIR}/shap_top30_words_sentiment.csv", index=False)
course_top30 = get_topwords(shap_values_course, texts, n=30)
pd.DataFrame(course_top30, columns=["word","importance"]).to_csv(f"{OUTPUT_DIR}/shap_top30_words_course.csv", index=False)

print("全て完了！出力パス：", OUTPUT_DIR)

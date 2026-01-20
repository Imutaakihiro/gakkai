#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
順序回帰モデル GPU最優先SHAP分析

**作成日**: 2025年1月

方針:
- GPU使用を最優先
- 100%のGPU使用率でなくても、GPUで動作することを優先
- バッチ処理を最適化してGPU負荷を最大化
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# SHAPのインポート
try:
    import shap
except ImportError as e:
    print(f"❌ SHAPインポートエラー: {e}")
    print("💡 python safe_fix_for_shap.py を実行してください")
    sys.exit(1)

# GPU最優先設定
print("="*60)
print("GPU最優先SHAP分析")
print("="*60)

# デバイス選択（GPU優先）
def get_device_gpu_priority():
    """GPUを最優先で選択"""
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

device = get_device_gpu_priority()
print(f"使用デバイス: {device}")

# モデル読み込み
from train_class_level_ordinal_llp import CourseOrdinalLLPModel, BASE_MODEL
from transformers import BertJapaneseTokenizer

print("\n📥 モデル読み込み中...")
tokenizer = BertJapaneseTokenizer.from_pretrained(BASE_MODEL)
model = CourseOrdinalLLPModel(BASE_MODEL)

MODEL_PATH = os.path.join("..", "02_モデル", "授業レベルマルチタスクモデル", "class_level_ordinal_llp_20251030_162353.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("02_モデル", "授業レベルマルチタスクモデル", "class_level_ordinal_llp_20251030_162353.pth")

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(device)  # GPUに移動
model.eval()

# GPUメモリ最適化
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print(f"✅ GPUメモリクリア完了")

print("✅ モデル読み込み完了")

# データ読み込み
CSV_PATH = os.path.join("..", "01_データ", "マルチタスク用データ", "授業集約データセット 回答分布付き.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join("01_データ", "マルチタスク用データ", "授業集約データセット 回答分布付き.csv")

print("\n📊 データ読み込み中...")
df = pd.read_csv(CSV_PATH)
texts = df['自由記述まとめ'].fillna("").astype(str).tolist()

# サンプル数
MAX_SAMPLES = 100
BATCH_SIZE = 64  # GPU使用率向上のため大きなバッチサイズ
MAX_LENGTH = 192

if len(texts) > MAX_SAMPLES:
    np.random.seed(42)
    sample_indices = np.random.choice(len(texts), MAX_SAMPLES, replace=False)
    sample_texts = [texts[i] for i in sample_indices]
else:
    sample_texts = texts

print(f"分析サンプル数: {len(sample_texts)}")
print(f"バッチサイズ: {BATCH_SIZE}")

# ======================== GPU最適化予測関数 ========================

def predict_sentiment_gpu(list_of_texts):
    """感情スコア予測（GPU最適化版）"""
    if isinstance(list_of_texts, str):
        list_of_texts = [list_of_texts]
    
    # GPUメモリクリア
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    pred = []
    model.eval()  # 推論モード
    
    with torch.no_grad():
        for i in range(0, len(list_of_texts), BATCH_SIZE):
            batch = [str(x) if not isinstance(x, str) else x for x in list_of_texts[i:i+BATCH_SIZE]]
            
            # トークナイズ（CPU）
            encoding = tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=MAX_LENGTH, 
                return_tensors="pt"
            )
            
            # GPUに一度に転送（非同期）
            input_ids = encoding['input_ids'].to(device, non_blocking=True)
            attention_mask = encoding['attention_mask'].to(device, non_blocking=True)
            chunk_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=device)
            
            # GPUで推論
            out = model(input_ids, attention_mask, chunk_mask)
            y_sent_pred = out[3]  # GPU上で保持
            
            # CPUに転送（最小限）
            pred.extend(y_sent_pred.cpu().numpy().tolist())
    
    return np.array(pred).reshape(-1, 1)

def predict_course_gpu(list_of_texts):
    """授業評価スコア予測（GPU最適化版）"""
    if isinstance(list_of_texts, str):
        list_of_texts = [list_of_texts]
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    pred = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(list_of_texts), BATCH_SIZE):
            batch = [str(x) if not isinstance(x, str) else x for x in list_of_texts[i:i+BATCH_SIZE]]
            
            encoding = tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=MAX_LENGTH, 
                return_tensors="pt"
            )
            
            input_ids = encoding['input_ids'].to(device, non_blocking=True)
            attention_mask = encoding['attention_mask'].to(device, non_blocking=True)
            chunk_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=device)
            
            out = model(input_ids, attention_mask, chunk_mask)
            y_course_pred = out[4]
            
            pred.extend(y_course_pred.cpu().numpy().tolist())
    
    return np.array(pred).reshape(-1, 1)

# ======================== SHAP分析実行 ========================

OUTPUT_DIR = os.path.join("..", "03_分析結果", "順序回帰SHAP_GPU最適化")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*60)
print("SHAP分析実行（GPU最優先）")
print("="*60)

# GPU使用状況を表示
if device.type == 'cuda':
    print(f"\n📊 GPU使用状況:")
    print(f"   デバイス: {torch.cuda.get_device_name(0)}")
    print(f"   メモリ使用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"   最大メモリ: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

# 1. 感情スコアSHAP分析
print("\n🔍 感情スコアSHAP分析実行中...")
print("   （GPUで推論を実行します）")

try:
    explainer_sent = shap.Explainer(
        predict_sentiment_gpu, 
        tokenizer,
        max_evals=50  # 計算量を調整
    )
    
    print("   SHAP値計算中（GPU推論を使用）...")
    shap_values_sent = explainer_sent(sample_texts, max_evals=50)
    
    # 重要度を集計
    importance = np.abs(shap_values_sent.values).mean(axis=0)
    words = shap_values_sent.feature_names if hasattr(shap_values_sent, "feature_names") else list(range(len(importance)))
    
    df_sent = pd.DataFrame({
        'word': words,
        'importance': importance.flatten() if importance.ndim > 1 else importance
    }).sort_values('importance', ascending=False)
    
    df_sent.to_csv(f"{OUTPUT_DIR}/word_importance_sentiment_gpu.csv", index=False, encoding='utf-8')
    print(f"✅ 感情スコアSHAP分析完了: {len(df_sent)}語")
    
except Exception as e:
    print(f"❌ 感情スコアSHAP分析エラー: {e}")
    import traceback
    traceback.print_exc()

# 2. 授業評価スコアSHAP分析
print("\n🔍 授業評価スコアSHAP分析実行中...")
print("   （GPUで推論を実行します）")

try:
    explainer_course = shap.Explainer(
        predict_course_gpu,
        tokenizer,
        max_evals=50
    )
    
    print("   SHAP値計算中（GPU推論を使用）...")
    shap_values_course = explainer_course(sample_texts, max_evals=50)
    
    importance = np.abs(shap_values_course.values).mean(axis=0)
    words = shap_values_course.feature_names if hasattr(shap_values_course, "feature_names") else list(range(len(importance)))
    
    df_course = pd.DataFrame({
        'word': words,
        'importance': importance.flatten() if importance.ndim > 1 else importance
    }).sort_values('importance', ascending=False)
    
    df_course.to_csv(f"{OUTPUT_DIR}/word_importance_course_gpu.csv", index=False, encoding='utf-8')
    print(f"✅ 授業評価スコアSHAP分析完了: {len(df_course)}語")
    
except Exception as e:
    print(f"❌ 授業評価スコアSHAP分析エラー: {e}")
    import traceback
    traceback.print_exc()

# GPU使用状況の最終確認
if device.type == 'cuda':
    print(f"\n📊 最終GPU使用状況:")
    print(f"   最大メモリ使用: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

print("\n" + "="*60)
print("✅ GPU最優先SHAP分析完了！")
print(f"📁 結果保存先: {OUTPUT_DIR}")
print("="*60)




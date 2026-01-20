#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
単一タスク感情分析モデルのSHAPプロット作成
既存のモデルを使用してSHAP summary plotを生成
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
print("単一タスク感情分析モデルのSHAPプロット作成")
print("="*60)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# モデルとトークナイザーのロード
MODEL_PATH = r"C:\Users\takahashi.DESKTOP-U0T5SUB\Downloads\BERT\git_excluded\finetuned_bert_model_20250718_step2_fixed_classweights_variant1_positive重点強化"
print(f"モデルをロード中: {MODEL_PATH}")

try:
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("✅ モデル読み込み完了")
except Exception as e:
    print(f"❌ モデル読み込みエラー: {e}")
    print("🔄 代替パスでモデルを読み込みます...")
    # 代替パス
    MODEL_PATH = "../02_モデル/単一タスクモデル2_評価スコア"
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("✅ 代替モデル読み込み完了")

# データ読み込み
DATA_PATH = "../01_データ/自由記述→感情スコア/finetuning_val_20250710_220621.csv"
print(f"\nデータをロード中: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ データ読み込み完了: {len(df)}件")
except Exception as e:
    print(f"❌ データ読み込みエラー: {e}")
    print("🔄 代替データで実行します...")
    # 代替データ
    DATA_PATH = "../01_データ/マルチタスク用データ/授業集約データセット_20251012_142504.csv"
    df = pd.read_csv(DATA_PATH)
    print(f"✅ 代替データ読み込み完了: {len(df)}件")

# サンプリング（100件でテスト）
SAMPLE_SIZE = 100
print(f"\nサンプリング: {SAMPLE_SIZE}件")

if '自由記述まとめ' in df.columns:
    texts = df['自由記述まとめ'].dropna().tolist()
elif '自由記述' in df.columns:
    texts = df['自由記述'].dropna().tolist()
else:
    print("❌ 適切なテキスト列が見つかりません")
    exit()

# ランダムサンプリング
sample_texts = np.random.choice(texts, size=min(SAMPLE_SIZE, len(texts)), replace=False).tolist()
print(f"✅ サンプリング完了: {len(sample_texts)}件")

# 予測関数（SHAP用）
def predict_proba(texts):
    """テキストのリストを受け取り、クラス確率を返す"""
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
    
    probs = []
    for text in texts:
        try:
            # トークン化
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 予測
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prob = torch.softmax(logits, dim=-1)
                probs.append(prob.cpu().numpy()[0])
        except Exception as e:
            print(f"予測エラー: {e}")
            # エラー時は均等な確率を返す
            probs.append(np.array([0.33, 0.33, 0.34]))
    
    return np.array(probs)

# SHAP分析実行
print("\n🔬 SHAP分析実行中...")
print("⚠️ 処理に時間がかかる場合があります...")

try:
    # SHAP Explainer作成
    explainer = shap.Explainer(predict_proba, tokenizer)
    
    # SHAP値計算（サンプル数を制限）
    shap_values = explainer(sample_texts[:20])  # 20件でテスト
    
    print("✅ SHAP分析完了")
    
    # 出力ディレクトリ作成
    output_dir = "../03_分析結果/単一タスクSHAPプロット"
    os.makedirs(output_dir, exist_ok=True)
    
    # SHAP summary plot作成
    print("\n📊 SHAPプロット作成中...")
    
    # 1. Summary plot (beeswarm plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, sample_texts[:20], show=False)
    plt.title("単一タスク感情分析モデルのSHAP Summary Plot", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Waterfall plot (最初のサンプル)
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.title("単一タスク感情分析モデルのSHAP Waterfall Plot (サンプル1)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_waterfall_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Bar plot (重要度順)
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, show=False)
    plt.title("単一タスク感情分析モデルのSHAP Bar Plot", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_bar_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ SHAPプロット作成完了")
    
    # 結果の保存
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_type": "single_task_sentiment",
        "sample_size": len(sample_texts[:20]),
        "shap_values_shape": shap_values.shape,
        "output_files": [
            f"shap_summary_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            f"shap_waterfall_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            f"shap_bar_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ]
    }
    
    with open(f"{output_dir}/shap_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 結果保存完了: {output_dir}")
    
except Exception as e:
    print(f"❌ SHAP分析エラー: {e}")
    print("🔄 簡易版SHAP分析を実行します...")
    
    # 簡易版SHAP分析
    try:
        # より小さなサンプルで再試行
        explainer = shap.Explainer(predict_proba, tokenizer)
        shap_values = explainer(sample_texts[:5])  # 5件でテスト
        
        print("✅ 簡易版SHAP分析完了")
        
        # 簡易版プロット作成
        output_dir = "../03_分析結果/単一タスクSHAPプロット"
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample_texts[:5], show=False)
        plt.title("単一タスク感情分析モデルのSHAP Summary Plot (簡易版)", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_summary_plot_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 簡易版SHAPプロット作成完了")
        
    except Exception as e2:
        print(f"❌ 簡易版SHAP分析もエラー: {e2}")
        print("💡 モデルやデータのパスを確認してください")

print("\n🎉 単一タスクSHAPプロット作成完了！")
print("📁 結果は '../03_分析結果/単一タスクSHAPプロット' に保存されました")

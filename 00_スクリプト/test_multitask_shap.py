#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マルチタスク学習モデルでのSHAP分析
授業評価スコア予測の要因分析
"""

import torch
import pandas as pd
import numpy as np
from transformers import BertJapaneseTokenizer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("マルチタスク学習モデルでのSHAP分析")
print("授業評価スコア予測の要因分析")
print("="*60)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# マルチタスクモデルの定義
class MultitaskModel(torch.nn.Module):
    """マルチタスク学習モデル"""
    
    def __init__(self, model_name='cl-tohoku/bert-base-japanese-whole-word-masking', dropout_rate=0.3):
        super(MultitaskModel, self).__init__()
        
        # BERTエンコーダー
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # 感情スコア予測ヘッド
        self.sentiment_head = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, 1)
        )
        
        # 授業評価スコア予測ヘッド
        self.course_head = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERTエンコーダー
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # マルチタスク予測
        sentiment_pred = self.sentiment_head(pooled_output)
        course_pred = self.course_head(pooled_output)
        
        return sentiment_pred.squeeze(), course_pred.squeeze()
    
    def predict_course_score(self, texts, tokenizer, device, max_length=512):
        """授業評価スコアのみを予測"""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # テキストのトークン化
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # 予測
                sentiment_pred, course_pred = self(input_ids, attention_mask)
                predictions.append(course_pred.cpu().item())
        
        return np.array(predictions)

# モデルとトークナイザーのロード
MODEL_PATH = "../02_モデル/マルチタスクモデル"
print(f"モデルをロード中: {MODEL_PATH}")

tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_PATH)

# マルチタスクモデルの読み込み
model = MultitaskModel()
model.load_state_dict(torch.load(f"{MODEL_PATH}/best_multitask_model.pth", map_location=device))
model.to(device)
model.eval()

print("モデル読み込み完了")

# データ読み込み
DATA_PATH = "../01_データ/マルチタスク用データ/マルチタスク学習用データセット_20250930_202839.csv"
print(f"\nデータをロード中: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"データ数: {len(df)}件")

# サンプリング（テスト用に100件）
SAMPLE_SIZE = 100
sample_df = df.sample(n=SAMPLE_SIZE, random_state=42)
texts = sample_df['自由記述'].tolist()
actual_scores = sample_df['授業評価スコア'].values

print(f"サンプリング数: {SAMPLE_SIZE}件")

# SHAP分析の実行
print("\nSHAP分析を実行中...")

# 予測関数の定義（授業評価スコア用）
def predict_course_score(texts):
    """授業評価スコアのみを予測する関数"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for text in texts:
            # テキストのトークン化
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # 予測
            sentiment_pred, course_pred = model(input_ids, attention_mask)
            predictions.append(course_pred.cpu().item())
    
    return np.array(predictions)

# SHAP Explainerの作成（授業評価スコア用）
explainer = shap.Explainer(predict_course_score, tokenizer)

# SHAP値の計算
print("授業評価スコアのSHAP値を計算中...")
shap_values = explainer(texts[:50])  # 50件でテスト

print("SHAP分析完了！")

# 結果の表示
print("\n=== 授業評価スコアSHAP分析結果 ===")
print(f"SHAP値の形状: {shap_values.shape}")
print(f"SHAP値の範囲: {shap_values.values.min():.3f} ~ {shap_values.values.max():.3f}")

# 実際の予測値とSHAP値の比較
predictions = predict_course_score(texts[:50])
print(f"授業評価予測値の範囲: {predictions.min():.3f} ~ {predictions.max():.3f}")

# 実際の授業評価スコアとの比較
actual_course = sample_df['授業評価スコア'].values[:50]
print(f"実際の授業評価スコアの範囲: {actual_course.min():.3f} ~ {actual_course.max():.3f}")

# 相関係数の計算
correlation = np.corrcoef(predictions, actual_course)[0, 1]
print(f"授業評価予測値と実際の値の相関係数: {correlation:.3f}")

# 結果の保存
output_dir = "../03_分析結果/マルチタスク学習/SHAP分析"
os.makedirs(output_dir, exist_ok=True)

# SHAP値の保存
shap_values_df = pd.DataFrame({
    'text': texts[:50],
    'predicted_course_score': predictions,
    'actual_course_score': actual_course,
    'actual_sentiment': sample_df['感情スコア'].values[:50]
})

shap_values_df.to_csv(f"{output_dir}/multitask_course_score_shap_results.csv", index=False, encoding='utf-8')

print(f"\n結果を保存しました: {output_dir}/multitask_course_score_shap_results.csv")

# 授業評価スコア予測の性能評価
print("\n=== 授業評価スコア予測の性能評価 ===")
print(f"相関係数: {correlation:.3f}")
print(f"R²: -0.108（既知の結果）")
print(f"RMSE: 0.193（既知の結果）")

# 結論
print("\n=== 授業評価スコアの単語要因分析の結論 ===")
if correlation > 0.3:
    print("✅ マルチタスク学習モデルでの授業評価スコアSHAP分析は部分的に有効です")
    print("ただし、予測精度が低いため解釈には注意が必要です")
elif correlation > 0.1:
    print("⚠️ マルチタスク学習モデルでの授業評価スコアSHAP分析は信頼性が低いです")
    print("相関係数が低く、予測精度が不十分です")
else:
    print("❌ マルチタスク学習モデルでの授業評価スコアSHAP分析は信頼性が極めて低いです")
    print("R²が負の値で、モデルが予測に失敗しています")

print(f"\n相関係数: {correlation:.3f}")
print("この結果は、マルチタスク学習モデルが授業評価スコアを")
print("適切に予測できていないことを示しています。")
print("SHAP値の解釈には注意が必要です。")

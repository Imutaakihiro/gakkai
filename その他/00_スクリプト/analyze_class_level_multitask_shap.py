#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
授業レベルマルチタスク学習モデルでのSHAP分析
感情スコア予測 + 授業評価スコア予測の要因分析
"""

import torch
import pandas as pd
import numpy as np
from transformers import BertJapaneseTokenizer, BertModel
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
print("授業レベルマルチタスク学習モデルでのSHAP分析")
print("感情スコア予測 + 授業評価スコア予測の要因分析")
print("="*60)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# マルチタスクモデルの定義
class ClassLevelMultitaskModel(torch.nn.Module):
    """授業レベルマルチタスク学習モデル"""
    
    def __init__(self, model_name='koheiduck/bert-japanese-finetuned-sentiment', dropout_rate=0.3):
        super(ClassLevelMultitaskModel, self).__init__()
        
        # BERTエンコーダー
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # 感情スコア予測ヘッド（回帰）
        self.sentiment_head = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(64, 1)
        )
        
        # 授業評価スコア予測ヘッド（回帰）
        self.course_head = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        """フォワードパス"""
        # BERTエンコーディング
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 各タスクの予測
        sentiment_pred = self.sentiment_head(pooled_output)
        course_pred = self.course_head(pooled_output)
        
        return sentiment_pred, course_pred

# モデルとトークナイザーのロード
MODEL_PATH = "../02_モデル/マルチタスクモデル"
print(f"モデルをロード中: {MODEL_PATH}")

tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_PATH)

# マルチタスクモデルの読み込み
model = ClassLevelMultitaskModel()
model.load_state_dict(torch.load(f"{MODEL_PATH}/best_multitask_model.pth", map_location=device))
model.to(device)
model.eval()

print("モデル読み込み完了")

# データ読み込み
DATA_PATH = "../01_データ/マルチタスク用データ/授業集約データセット_20251012_142504.csv"
print(f"\nデータをロード中: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"データ数: {len(df)}件")
print(f"カラム: {df.columns.tolist()}")

# サンプリング（テスト用に100件）
SAMPLE_SIZE = 100
sample_df = df.sample(n=SAMPLE_SIZE, random_state=42)
texts = sample_df['自由記述'].tolist()
actual_sentiment = sample_df['感情スコア平均'].values
actual_course = sample_df['授業評価スコア'].values

print(f"サンプリング数: {SAMPLE_SIZE}件")

# 予測関数の定義
def predict_sentiment_score(texts):
    """感情スコアのみを予測する関数"""
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
            predictions.append(sentiment_pred.cpu().numpy()[0][0])
    
    return np.array(predictions)

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
            predictions.append(course_pred.cpu().numpy()[0][0])
    
    return np.array(predictions)

# SHAP分析の実行
print("\nSHAP分析を実行中...")

# 感情スコア予測のSHAP分析
print("感情スコア予測のSHAP分析...")
explainer_sentiment = shap.Explainer(predict_sentiment_score, tokenizer)
shap_values_sentiment = explainer_sentiment(texts[:50])  # 50件でテスト

# 授業評価スコア予測のSHAP分析
print("授業評価スコア予測のSHAP分析...")
explainer_course = shap.Explainer(predict_course_score, tokenizer)
shap_values_course = explainer_course(texts[:50])  # 50件でテスト

print("SHAP分析完了！")

# 結果の表示
print("\n=== 感情スコアSHAP分析結果 ===")
print(f"SHAP値の形状: {shap_values_sentiment.shape}")
print(f"SHAP値の範囲: {shap_values_sentiment.values.min():.3f} ~ {shap_values_sentiment.values.max():.3f}")

print("\n=== 授業評価スコアSHAP分析結果 ===")
print(f"SHAP値の形状: {shap_values_course.shape}")
print(f"SHAP値の範囲: {shap_values_course.values.min():.3f} ~ {shap_values_course.values.max():.3f}")

# 実際の予測値とSHAP値の比較
predictions_sentiment = predict_sentiment_score(texts[:50])
predictions_course = predict_course_score(texts[:50])

print(f"\n感情スコア予測値の範囲: {predictions_sentiment.min():.3f} ~ {predictions_sentiment.max():.3f}")
print(f"授業評価スコア予測値の範囲: {predictions_course.min():.3f} ~ {predictions_course.max():.3f}")

# 実際の値との比較
actual_sentiment_sample = actual_sentiment[:50]
actual_course_sample = actual_course[:50]

print(f"実際の感情スコアの範囲: {actual_sentiment_sample.min():.3f} ~ {actual_sentiment_sample.max():.3f}")
print(f"実際の授業評価スコアの範囲: {actual_course_sample.min():.3f} ~ {actual_course_sample.max():.3f}")

# 相関係数の計算
correlation_sentiment = np.corrcoef(predictions_sentiment, actual_sentiment_sample)[0, 1]
correlation_course = np.corrcoef(predictions_course, actual_course_sample)[0, 1]

print(f"\n感情スコア予測値と実際の値の相関係数: {correlation_sentiment:.3f}")
print(f"授業評価スコア予測値と実際の値の相関係数: {correlation_course:.3f}")

# 結果の保存
output_dir = "../03_分析結果/SHAP分析/授業レベルマルチタスク"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# SHAP値の保存
shap_results_df = pd.DataFrame({
    'text': texts[:50],
    'predicted_sentiment': predictions_sentiment,
    'actual_sentiment': actual_sentiment_sample,
    'predicted_course_score': predictions_course,
    'actual_course_score': actual_course_sample
})

shap_results_df.to_csv(f"{output_dir}/class_level_multitask_shap_results_{timestamp}.csv", 
                       index=False, encoding='utf-8')

print(f"\n結果を保存しました: {output_dir}/class_level_multitask_shap_results_{timestamp}.csv")

# 性能評価
print("\n=== 授業レベルマルチタスク学習の性能評価 ===")
print("感情スコア予測:")
print(f"  相関係数: {correlation_sentiment:.3f}")
print(f"  R²: 0.392（既知の結果）")
print(f"  RMSE: 0.202（既知の結果）")

print("\n授業評価スコア予測:")
print(f"  相関係数: {correlation_course:.3f}")
print(f"  R²: 0.106（既知の結果）")
print(f"  RMSE: 0.189（既知の結果）")

# 結論
print("\n=== 授業レベルマルチタスク学習のSHAP分析の結論 ===")
if correlation_sentiment > 0.6:
    print("✅ 感情スコア予測のSHAP分析は有効です")
else:
    print("⚠️ 感情スコア予測のSHAP分析は限定的です")

if correlation_course > 0.4:
    print("✅ 授業評価スコア予測のSHAP分析は有効です")
else:
    print("⚠️ 授業評価スコア予測のSHAP分析は限定的です")

print("\n=== 次のステップ ===")
print("1. 単一モデルとのSHAP比較分析")
print("2. 重要な単語・フレーズの特定")
print("3. 感情スコアと授業評価スコアの関係性分析")
print("4. 論文の解釈可能性セクションの完成")

print(f"\n分析完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

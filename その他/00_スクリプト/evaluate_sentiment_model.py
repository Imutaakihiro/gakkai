"""
単一タスクモデル1: 自由記述→感情スコア の性能評価スクリプト
Accuracy, F1-Score, Precision, Recall を計算
"""

import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import os

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# モデルとトークナイザーのロード
MODEL_PATH = r"C:\Users\takahashi.DESKTOP-U0T5SUB\Downloads\BERT\git_excluded\finetuned_bert_model_20250718_step2_fixed_classweights_variant1_positive重点強化"
print(f"モデルをロード中: {MODEL_PATH}")

tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# 検証データの読み込み
VAL_DATA_PATH = r"01_データ\自由記述→感情スコア\finetuning_val_20250710_220621.csv"
print(f"検証データをロード中: {VAL_DATA_PATH}")
val_df = pd.read_csv(VAL_DATA_PATH)

print(f"検証データサイズ: {len(val_df)}")
print(f"ラベル分布:\n{val_df['label'].value_counts()}")

# 予測の実行
def predict_batch(texts, batch_size=16):
    """バッチ処理で予測を実行"""
    predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="予測中"):
        batch_texts = texts[i:i+batch_size]
        
        # トークナイズ
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # デバイスに転送
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 予測
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)
    
    return predictions

# 予測実行
texts = val_df['text'].tolist()
true_labels = val_df['label'].astype(int).tolist()

predictions = predict_batch(texts)

# ラベルマッピング（-1, 0, 1 → 0, 1, 2）
# データを確認して適切なマッピングを設定
unique_labels = sorted(val_df['label'].unique())
print(f"ユニークなラベル: {unique_labels}")

# 評価指標の計算
accuracy = accuracy_score(true_labels, predictions)
f1_macro = f1_score(true_labels, predictions, average='macro')
f1_weighted = f1_score(true_labels, predictions, average='weighted')
precision_macro = precision_score(true_labels, predictions, average='macro')
precision_weighted = precision_score(true_labels, predictions, average='weighted')
recall_macro = recall_score(true_labels, predictions, average='macro')
recall_weighted = recall_score(true_labels, predictions, average='weighted')

# 結果の表示
print("\n" + "="*60)
print("単一タスクモデル1（感情スコア）の性能評価結果")
print("="*60)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"\nF1-Score (Macro): {f1_macro:.4f}")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")
print(f"\nPrecision (Macro): {precision_macro:.4f}")
print(f"Precision (Weighted): {precision_weighted:.4f}")
print(f"\nRecall (Macro): {recall_macro:.4f}")
print(f"Recall (Weighted): {recall_weighted:.4f}")
print("\n" + "="*60)

# クラスごとの詳細レポート
print("\nクラスごとの詳細:")
print(classification_report(true_labels, predictions))

# 混同行列
print("\n混同行列:")
conf_matrix = confusion_matrix(true_labels, predictions)
print(conf_matrix)

# 結果を保存
results = {
    "model_name": "単一タスクモデル1（感情スコア）",
    "model_path": MODEL_PATH,
    "evaluation_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "dataset_size": len(val_df),
    "metrics": {
        "accuracy": float(accuracy),
        "f1_score_macro": float(f1_macro),
        "f1_score_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted)
    },
    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": classification_report(true_labels, predictions, output_dict=True)
}

# 結果ディレクトリの作成
os.makedirs("03_分析結果/モデル評価", exist_ok=True)

# JSONとして保存
result_file = f"03_分析結果/モデル評価/sentiment_model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n評価結果を保存しました: {result_file}")

# 予測例をいくつか表示
print("\n予測例（最初の5件）:")
for i in range(min(5, len(val_df))):
    print(f"\nテキスト: {texts[i][:50]}...")
    print(f"正解: {true_labels[i]}, 予測: {predictions[i]}")


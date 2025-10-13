"""
マルチタスクモデル: 自由記述→感情スコア+評価スコア のトレーニングスクリプト
感情分析（分類）と評価スコア予測（回帰）を同時に学習
"""

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertJapaneseTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                            mean_squared_error, mean_absolute_error, r2_score)
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import os

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ハイパーパラメータ
BATCH_SIZE = 16
LEARNING_RATE = 5e-6  # より安定した学習のため学習率を下げる
NUM_EPOCHS = 5
MAX_LENGTH = 512
BASE_MODEL = "koheiduck/bert-japanese-finetuned-sentiment"
ALPHA = 0.5  # 感情分析タスクの損失の重み
BETA = 0.5   # 評価スコアタスクの損失の重み
WARMUP_RATIO = 0.1  # ウォームアップの割合

# データセットクラス
class MultitaskDataset(Dataset):
    def __init__(self, texts, sentiment_labels, score_labels, tokenizer, max_length):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.score_labels = score_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment = int(self.sentiment_labels[idx])
        score = float(self.score_labels[idx])
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'score': torch.tensor(score, dtype=torch.float)
        }

# マルチタスクBERTモデル
class BertForMultitask(nn.Module):
    def __init__(self, base_model_name, num_sentiment_classes=3):
        super(BertForMultitask, self).__init__()
        self.bert = BertModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(0.3)  # ドロップアウト率を上げて過学習を防ぐ
        
        # 感情分類ヘッド
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, num_sentiment_classes)
        
        # 評価スコア回帰ヘッド
        self.score_regressor = nn.Linear(self.bert.config.hidden_size, 1)
        
        # 重みの初期化を改善
        nn.init.xavier_uniform_(self.sentiment_classifier.weight)
        nn.init.zeros_(self.sentiment_classifier.bias)
        nn.init.xavier_uniform_(self.score_regressor.weight)
        nn.init.zeros_(self.score_regressor.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 感情分類の出力
        sentiment_logits = self.sentiment_classifier(pooled_output)
        
        # 評価スコアの出力
        score = self.score_regressor(pooled_output).squeeze()
        
        return sentiment_logits, score

# データの読み込み
print("データを読み込み中...")
df = pd.read_csv(r"01_データ\マルチタスク用データ\マルチタスク学習用データセット_20250930_202839.csv")

# データの前処理
# 感情スコアを0, 1, 2にマッピング（-1, 0, 1 → 0, 1, 2）
sentiment_mapping = {-1.0: 0, 0.0: 1, 1.0: 2}
df['sentiment_label'] = df['感情スコア'].map(sentiment_mapping)

# データのクリーニング: NaNを除去
print(f"データサイズ（読み込み後）: {len(df)}")
df = df.dropna(subset=['自由記述', '感情スコア', '授業評価スコア', 'sentiment_label'])
print(f"データサイズ（NaN除去後）: {len(df)}")
print(f"感情ラベル分布:\n{df['sentiment_label'].value_counts()}")

# 評価スコアの正規化（安定性向上のため）
score_mean = df['授業評価スコア'].mean()
score_std = df['授業評価スコア'].std()
print(f"評価スコア範囲: {df['授業評価スコア'].min():.2f} - {df['授業評価スコア'].max():.2f}")
print(f"評価スコア平均: {score_mean:.2f}, 標準偏差: {score_std:.2f}")

# 正規化
df['score_normalized'] = (df['授業評価スコア'] - score_mean) / score_std

# 訓練/検証データの分割（80/20）
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment_label'])

print(f"訓練データサイズ: {len(train_df)}")
print(f"検証データサイズ: {len(val_df)}")

# トークナイザーとモデルの初期化
print(f"\nベースモデルをロード中: {BASE_MODEL}")
tokenizer = BertJapaneseTokenizer.from_pretrained(BASE_MODEL)
model = BertForMultitask(BASE_MODEL, num_sentiment_classes=3)
model.to(device)

# データセットとデータローダーの作成（正規化されたスコアを使用）
train_dataset = MultitaskDataset(
    train_df['自由記述'].values,
    train_df['sentiment_label'].values,
    train_df['score_normalized'].values,
    tokenizer,
    MAX_LENGTH
)

val_dataset = MultitaskDataset(
    val_df['自由記述'].values,
    val_df['sentiment_label'].values,
    val_df['score_normalized'].values,
    tokenizer,
    MAX_LENGTH
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# オプティマイザとスケジューラ
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"総ステップ数: {total_steps}, ウォームアップステップ: {warmup_steps}")

# 損失関数
sentiment_criterion = nn.CrossEntropyLoss()
score_criterion = nn.MSELoss()

# 評価関数
def evaluate(model, dataloader, denormalize_score=False):
    model.eval()
    
    sentiment_preds = []
    sentiment_true = []
    score_preds = []
    score_true = []
    total_loss = 0
    total_sentiment_loss = 0
    total_score_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment'].to(device)
            score_labels = batch['score'].to(device)
            
            sentiment_logits, score_outputs = model(input_ids, attention_mask)
            
            # NaNチェック
            if torch.isnan(sentiment_logits).any() or torch.isnan(score_outputs).any():
                print("警告: 予測にNaNが含まれています。スキップします。")
                continue
            
            # 損失計算
            sentiment_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
            score_loss = score_criterion(score_outputs, score_labels)
            loss = ALPHA * sentiment_loss + BETA * score_loss
            
            total_loss += loss.item()
            total_sentiment_loss += sentiment_loss.item()
            total_score_loss += score_loss.item()
            
            # 予測値の収集
            sentiment_pred = torch.argmax(sentiment_logits, dim=-1)
            sentiment_preds.extend(sentiment_pred.cpu().numpy())
            sentiment_true.extend(sentiment_labels.cpu().numpy())
            score_preds.extend(score_outputs.cpu().numpy())
            score_true.extend(score_labels.cpu().numpy())
    
    sentiment_preds = np.array(sentiment_preds)
    sentiment_true = np.array(sentiment_true)
    score_preds = np.array(score_preds)
    score_true = np.array(score_true)
    
    # NaNチェック
    if np.isnan(score_preds).any() or np.isnan(score_true).any():
        print("エラー: 評価データにNaNが含まれています")
        return {
            'total_loss': float('inf'),
            'sentiment_loss': float('inf'),
            'score_loss': float('inf'),
            'sentiment_accuracy': 0.0,
            'sentiment_f1_macro': 0.0,
            'sentiment_f1_weighted': 0.0,
            'sentiment_precision': 0.0,
            'sentiment_recall': 0.0,
            'rmse': float('inf'),
            'mae': float('inf'),
            'r2': 0.0,
            'correlation': 0.0
        }
    
    # 正規化を元に戻す（必要な場合）
    if denormalize_score:
        score_preds = score_preds * score_std + score_mean
        score_true = score_true * score_std + score_mean
    
    # 感情分析の評価指標
    sentiment_accuracy = accuracy_score(sentiment_true, sentiment_preds)
    sentiment_f1_macro = f1_score(sentiment_true, sentiment_preds, average='macro')
    sentiment_f1_weighted = f1_score(sentiment_true, sentiment_preds, average='weighted')
    sentiment_precision = precision_score(sentiment_true, sentiment_preds, average='macro')
    sentiment_recall = recall_score(sentiment_true, sentiment_preds, average='macro')
    
    # 評価スコアの評価指標
    rmse = np.sqrt(mean_squared_error(score_true, score_preds))
    mae = mean_absolute_error(score_true, score_preds)
    r2 = r2_score(score_true, score_preds)
    correlation, _ = pearsonr(score_true, score_preds)
    
    return {
        'total_loss': total_loss / len(dataloader),
        'sentiment_loss': total_sentiment_loss / len(dataloader),
        'score_loss': total_score_loss / len(dataloader),
        'sentiment_accuracy': sentiment_accuracy,
        'sentiment_f1_macro': sentiment_f1_macro,
        'sentiment_f1_weighted': sentiment_f1_weighted,
        'sentiment_precision': sentiment_precision,
        'sentiment_recall': sentiment_recall,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation
    }

# トレーニングループ
print("\nトレーニング開始")
print("="*60)

best_combined_score = float('inf')  # 総合損失で評価
history = {
    'train_loss': [],
    'train_sentiment_loss': [],
    'train_score_loss': [],
    'val_metrics': []
}

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    # トレーニング
    model.train()
    train_loss = 0
    train_sentiment_loss = 0
    train_score_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment_labels = batch['sentiment'].to(device)
        score_labels = batch['score'].to(device)
        
        optimizer.zero_grad()
        sentiment_logits, score_outputs = model(input_ids, attention_mask)
        
        # 損失計算
        sentiment_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
        score_loss = score_criterion(score_outputs, score_labels)
        loss = ALPHA * sentiment_loss + BETA * score_loss
        
        # NaNチェック
        if torch.isnan(loss):
            print(f"警告: 損失がNaNです。このバッチをスキップします。")
            continue
        
        loss.backward()
        
        # 勾配クリッピング（より厳しく）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        train_sentiment_loss += sentiment_loss.item()
        train_score_loss += score_loss.item()
        
        progress_bar.set_postfix({
            'total_loss': f'{loss.item():.4f}',
            'sent_loss': f'{sentiment_loss.item():.4f}',
            'score_loss': f'{score_loss.item():.4f}'
        })
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_sentiment_loss = train_sentiment_loss / len(train_loader)
    avg_train_score_loss = train_score_loss / len(train_loader)
    
    # 検証
    val_metrics = evaluate(model, val_loader)
    
    # 履歴の記録（float型に変換）
    history['train_loss'].append(float(avg_train_loss))
    history['train_sentiment_loss'].append(float(avg_train_sentiment_loss))
    history['train_score_loss'].append(float(avg_train_score_loss))
    # val_metricsも再帰的にfloat型に変換
    history['val_metrics'].append({k: float(v) for k, v in val_metrics.items()})
    
    print(f"\nTrain Loss: {avg_train_loss:.4f} (Sentiment: {avg_train_sentiment_loss:.4f}, Score: {avg_train_score_loss:.4f})")
    print(f"Val Loss: {val_metrics['total_loss']:.4f} (Sentiment: {val_metrics['sentiment_loss']:.4f}, Score: {val_metrics['score_loss']:.4f})")
    print(f"\n感情分析:")
    print(f"  Accuracy: {val_metrics['sentiment_accuracy']:.4f}")
    print(f"  F1 (Macro): {val_metrics['sentiment_f1_macro']:.4f}")
    print(f"  F1 (Weighted): {val_metrics['sentiment_f1_weighted']:.4f}")
    print(f"  Precision: {val_metrics['sentiment_precision']:.4f}")
    print(f"  Recall: {val_metrics['sentiment_recall']:.4f}")
    print(f"\n評価スコア:")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  MAE: {val_metrics['mae']:.4f}")
    print(f"  R²: {val_metrics['r2']:.4f}")
    print(f"  Correlation: {val_metrics['correlation']:.4f}")
    
    # ベストモデルの保存
    combined_score = val_metrics['total_loss']
    if combined_score < best_combined_score:
        best_combined_score = combined_score
        save_dir = "02_モデル/マルチタスクモデル"
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
        tokenizer.save_pretrained(save_dir)
        print(f"\nベストモデルを保存しました (Total Loss: {best_combined_score:.4f})")

print("\n" + "="*60)
print("トレーニング完了")
print("="*60)

# 最終評価
print("\n最終評価（ベストモデル）:")
model.load_state_dict(torch.load("02_モデル/マルチタスクモデル/best_model.pth"))

# 正規化されたスコアでの評価
final_metrics_normalized = evaluate(model, val_loader, denormalize_score=False)
# 元のスケールでの評価
final_metrics = evaluate(model, val_loader, denormalize_score=True)

print("\n感情分析:")
print(f"  Accuracy: {final_metrics['sentiment_accuracy']:.4f}")
print(f"  F1 (Macro): {final_metrics['sentiment_f1_macro']:.4f}")
print(f"  F1 (Weighted): {final_metrics['sentiment_f1_weighted']:.4f}")
print(f"  Precision: {final_metrics['sentiment_precision']:.4f}")
print(f"  Recall: {final_metrics['sentiment_recall']:.4f}")
print("\n評価スコア:")
print(f"  RMSE: {final_metrics['rmse']:.4f}")
print(f"  MAE: {final_metrics['mae']:.4f}")
print(f"  R²: {final_metrics['r2']:.4f}")
print(f"  Correlation: {final_metrics['correlation']:.4f}")

# 結果を保存
results = {
    "model_name": "マルチタスクモデル（感情スコア+評価スコア）",
    "base_model": BASE_MODEL,
    "training_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "hyperparameters": {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "max_length": MAX_LENGTH,
        "alpha": ALPHA,
        "beta": BETA
    },
    "dataset_size": {
        "train": len(train_df),
        "val": len(val_df)
    },
    "final_metrics": {
        "sentiment": {
            "accuracy": float(final_metrics['sentiment_accuracy']),
            "f1_macro": float(final_metrics['sentiment_f1_macro']),
            "f1_weighted": float(final_metrics['sentiment_f1_weighted']),
            "precision": float(final_metrics['sentiment_precision']),
            "recall": float(final_metrics['sentiment_recall'])
        },
        "score": {
            "rmse": float(final_metrics['rmse']),
            "mae": float(final_metrics['mae']),
            "r2": float(final_metrics['r2']),
            "correlation": float(final_metrics['correlation'])
        }
    },
    "history": history
}

os.makedirs("03_分析結果/モデル評価", exist_ok=True)
result_file = f"03_分析結果/モデル評価/multitask_model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n訓練結果を保存しました: {result_file}")

# モデル設定も保存
config = {
    "model_type": "BertForMultitask",
    "base_model": BASE_MODEL,
    "num_sentiment_classes": 3,
    "hidden_size": model.bert.config.hidden_size,
    "max_length": MAX_LENGTH,
    "alpha": ALPHA,
    "beta": BETA,
    "score_mean": float(score_mean),
    "score_std": float(score_std)
}

config_file = "02_モデル/マルチタスクモデル/model_config.json"
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"モデル設定を保存しました: {config_file}")


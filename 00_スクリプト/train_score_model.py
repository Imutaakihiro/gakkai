"""
単一タスクモデル2: 自由記述→評価スコア のトレーニングスクリプト
回帰タスク（RMSE, MAE, R², 相関係数で評価）
"""

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertJapaneseTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
WARMUP_RATIO = 0.1  # ウォームアップの割合

# データセットクラス
class ScoreDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = float(self.scores[idx])
        
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
            'score': torch.tensor(score, dtype=torch.float)
        }

# 回帰用BERTモデル
class BertForRegression(nn.Module):
    def __init__(self, base_model_name):
        super(BertForRegression, self).__init__()
        self.bert = BertModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(0.3)  # ドロップアウト率を上げて過学習を防ぐ
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        
        # 回帰ヘッドの初期化を改善
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        score = self.regressor(pooled_output)
        return score.squeeze()

# データの読み込み
print("データを読み込み中...")
train_df = pd.read_csv(r"01_データ\自由記述→評価スコアデータ\score_train_dataset.csv")
val_df = pd.read_csv(r"01_データ\自由記述→評価スコアデータ\score_val_dataset.csv")

# データのクリーニング: NaNを除去
print(f"訓練データサイズ（読み込み後）: {len(train_df)}")
train_df = train_df.dropna(subset=['text', 'label'])
val_df = val_df.dropna(subset=['text', 'label'])
print(f"訓練データサイズ（NaN除去後）: {len(train_df)}")
print(f"検証データサイズ: {len(val_df)}")

# スコアの正規化（安定性向上のため）
score_mean = train_df['label'].mean()
score_std = train_df['label'].std()
print(f"スコア範囲: {train_df['label'].min():.2f} - {train_df['label'].max():.2f}")
print(f"スコア平均: {score_mean:.2f}, 標準偏差: {score_std:.2f}")

# 正規化
train_df['label_normalized'] = (train_df['label'] - score_mean) / score_std
val_df['label_normalized'] = (val_df['label'] - score_mean) / score_std

# トークナイザーとモデルの初期化
print(f"\nベースモデルをロード中: {BASE_MODEL}")
tokenizer = BertJapaneseTokenizer.from_pretrained(BASE_MODEL)
model = BertForRegression(BASE_MODEL)
model.to(device)

# データセットとデータローダーの作成（正規化されたラベルを使用）
train_dataset = ScoreDataset(
    train_df['text'].values,
    train_df['label_normalized'].values,
    tokenizer,
    MAX_LENGTH
)

val_dataset = ScoreDataset(
    val_df['text'].values,
    val_df['label_normalized'].values,
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
criterion = nn.MSELoss()

# 評価関数
def evaluate(model, dataloader, denormalize=False):
    model.eval()
    predictions = []
    true_scores = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            # NaNチェック
            if torch.isnan(outputs).any():
                print("警告: 予測にNaNが含まれています。スキップします。")
                continue
            
            loss = criterion(outputs, scores)
            total_loss += loss.item()
            
            predictions.extend(outputs.cpu().numpy())
            true_scores.extend(scores.cpu().numpy())
    
    predictions = np.array(predictions)
    true_scores = np.array(true_scores)
    
    # NaNチェック
    if np.isnan(predictions).any() or np.isnan(true_scores).any():
        print("エラー: 評価データにNaNが含まれています")
        return {
            'loss': float('inf'),
            'rmse': float('inf'),
            'mae': float('inf'),
            'r2': 0.0,
            'correlation': 0.0
        }
    
    # 正規化を元に戻す（必要な場合）
    if denormalize:
        predictions = predictions * score_std + score_mean
        true_scores = true_scores * score_std + score_mean
    
    rmse = np.sqrt(mean_squared_error(true_scores, predictions))
    mae = mean_absolute_error(true_scores, predictions)
    r2 = r2_score(true_scores, predictions)
    correlation, _ = pearsonr(true_scores, predictions)
    
    return {
        'loss': total_loss / len(dataloader),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation
    }

# トレーニングループ
print("\nトレーニング開始")
print("="*60)

best_rmse = float('inf')
history = {
    'train_loss': [],
    'val_loss': [],
    'val_rmse': [],
    'val_mae': [],
    'val_r2': [],
    'val_correlation': []
}

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    # トレーニング
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, scores)
        
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
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = train_loss / len(train_loader)
    
    # 検証
    val_metrics = evaluate(model, val_loader)
    
    # 履歴の記録（float型に変換）
    history['train_loss'].append(float(avg_train_loss))
    history['val_loss'].append(float(val_metrics['loss']))
    history['val_rmse'].append(float(val_metrics['rmse']))
    history['val_mae'].append(float(val_metrics['mae']))
    history['val_r2'].append(float(val_metrics['r2']))
    history['val_correlation'].append(float(val_metrics['correlation']))
    
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f}")
    print(f"Val RMSE: {val_metrics['rmse']:.4f}")
    print(f"Val MAE: {val_metrics['mae']:.4f}")
    print(f"Val R²: {val_metrics['r2']:.4f}")
    print(f"Val Correlation: {val_metrics['correlation']:.4f}")
    
    # ベストモデルの保存
    if val_metrics['rmse'] < best_rmse:
        best_rmse = val_metrics['rmse']
        save_dir = "02_モデル/単一タスクモデル2_評価スコア"
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
        tokenizer.save_pretrained(save_dir)
        print(f"ベストモデルを保存しました (RMSE: {best_rmse:.4f})")

print("\n" + "="*60)
print("トレーニング完了")
print("="*60)

# 最終評価
print("\n最終評価（ベストモデル）:")
model.load_state_dict(torch.load(f"02_モデル/単一タスクモデル2_評価スコア/best_model.pth"))

# 正規化されたスコアでの評価
final_metrics_normalized = evaluate(model, val_loader, denormalize=False)
# 元のスケールでの評価
final_metrics = evaluate(model, val_loader, denormalize=True)

print(f"RMSE: {final_metrics['rmse']:.4f}")
print(f"MAE: {final_metrics['mae']:.4f}")
print(f"R²: {final_metrics['r2']:.4f}")
print(f"相関係数: {final_metrics['correlation']:.4f}")

# 結果を保存
results = {
    "model_name": "単一タスクモデル2（評価スコア）",
    "base_model": BASE_MODEL,
    "training_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "hyperparameters": {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "max_length": MAX_LENGTH
    },
    "dataset_size": {
        "train": len(train_df),
        "val": len(val_df)
    },
    "final_metrics": {
        "rmse": float(final_metrics['rmse']),
        "mae": float(final_metrics['mae']),
        "r2": float(final_metrics['r2']),
        "correlation": float(final_metrics['correlation'])
    },
    "history": history
}

os.makedirs("03_分析結果/モデル評価", exist_ok=True)
result_file = f"03_分析結果/モデル評価/score_model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n訓練結果を保存しました: {result_file}")

# モデル設定も保存
config = {
    "model_type": "BertForRegression",
    "base_model": BASE_MODEL,
    "hidden_size": model.bert.config.hidden_size,
    "max_length": MAX_LENGTH,
    "score_mean": float(score_mean),
    "score_std": float(score_std)
}

config_file = "02_モデル/単一タスクモデル2_評価スコア/model_config.json"
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"モデル設定を保存しました: {config_file}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マルチタスク学習: 自由記述から感情スコアと授業評価スコアを同時予測
データセット: マルチタスク学習用データセット_20250930_202839.csv
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {DEVICE}")

class MultitaskDataset(Dataset):
    """マルチタスク学習用データセット"""
    
    def __init__(self, texts, sentiment_scores, course_scores, tokenizer, max_length=512):
        self.texts = texts
        self.sentiment_scores = sentiment_scores
        self.course_scores = course_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment_score = self.sentiment_scores[idx]
        course_score = self.course_scores[idx]
        
        # テキストのトークン化
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment_score': torch.tensor(sentiment_score, dtype=torch.float),
            'course_score': torch.tensor(course_score, dtype=torch.float)
        }

class MultitaskModel(nn.Module):
    """マルチタスク学習モデル"""
    
    def __init__(self, model_name='cl-tohoku/bert-base-japanese-whole-word-masking', dropout_rate=0.3):
        super(MultitaskModel, self).__init__()
        
        # BERTエンコーダー
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 感情スコア予測ヘッド
        self.sentiment_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
        # 授業評価スコア予測ヘッド
        self.course_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
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

def load_multitask_data():
    """マルチタスク学習用データを読み込み"""
    print("データを読み込み中...")
    
    # データ読み込み
    df = pd.read_csv('../01_データ/マルチタスク用データ/マルチタスク学習用データセット_20250930_202839.csv')
    
    # 空の行を除去
    df = df.dropna(subset=['自由記述', '感情スコア', '授業評価スコア'])
    
    print(f"データ数: {len(df)}")
    print(f"感情スコア範囲: {df['感情スコア'].min():.1f} ~ {df['感情スコア'].max():.1f}")
    print(f"授業評価スコア範囲: {df['授業評価スコア'].min():.2f} ~ {df['授業評価スコア'].max():.2f}")
    
    return df

def preprocess_data(df):
    """データの前処理"""
    print("データの前処理中...")
    
    # テキストとスコアを取得
    texts = df['自由記述'].tolist()
    sentiment_scores = df['感情スコア'].values
    course_scores = df['授業評価スコア'].values
    
    # スコアの正規化
    sentiment_scaler = StandardScaler()
    course_scaler = StandardScaler()
    
    sentiment_scores_norm = sentiment_scaler.fit_transform(sentiment_scores.reshape(-1, 1)).flatten()
    course_scores_norm = course_scaler.fit_transform(course_scores.reshape(-1, 1)).flatten()
    
    return texts, sentiment_scores, course_scores, sentiment_scores_norm, course_scores_norm, sentiment_scaler, course_scaler

def train_multitask_model(texts, sentiment_scores_norm, course_scores_norm, sentiment_scaler, course_scaler):
    """マルチタスク学習モデルの訓練"""
    print("マルチタスク学習モデルを訓練中...")
    
    # トークナイザー
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    
    # データ分割（8:2）
    # まず学習:テスト = 8:2に分割
    train_texts, test_texts, train_sentiment, test_sentiment, train_course, test_course = train_test_split(
        texts, sentiment_scores_norm, course_scores_norm, 
        test_size=0.2, random_state=42, stratify=None
    )
    
    # 学習データをさらに学習:検証 = 8:2に分割（実質 6.4:1.6:2.0）
    train_texts, val_texts, train_sentiment, val_sentiment, train_course, val_course = train_test_split(
        train_texts, train_sentiment, train_course, 
        test_size=0.2, random_state=42, stratify=None
    )
    
    # データセット作成
    train_dataset = MultitaskDataset(train_texts, train_sentiment, train_course, tokenizer)
    val_dataset = MultitaskDataset(val_texts, val_sentiment, val_course, tokenizer)
    
    # データローダー
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # モデル初期化
    model = MultitaskModel().to(DEVICE)
    
    # 損失関数とオプティマイザー
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # 訓練ループ
    num_epochs = 5
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 訓練
        model.train()
        train_loss = 0.0
        sentiment_losses = []
        course_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            sentiment_target = batch['sentiment_score'].to(DEVICE)
            course_target = batch['course_score'].to(DEVICE)
            
            # 進行度表示（10バッチごと）
            if batch_idx % 10 == 0:
                print(f'  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}')
            
            optimizer.zero_grad()
            
            # 予測
            sentiment_pred, course_pred = model(input_ids, attention_mask)
            
            # 損失計算（重み付き）
            sentiment_loss = criterion(sentiment_pred, sentiment_target)
            course_loss = criterion(course_pred, course_target)
            
            # マルチタスク損失（感情分析を重視）
            total_loss = 0.7 * sentiment_loss + 0.3 * course_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            sentiment_losses.append(sentiment_loss.item())
            course_losses.append(course_loss.item())
        
        # 検証
        model.eval()
        val_loss = 0.0
        val_sentiment_losses = []
        val_course_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                sentiment_target = batch['sentiment_score'].to(DEVICE)
                course_target = batch['course_score'].to(DEVICE)
                
                # 検証進行度表示（5バッチごと）
                if batch_idx % 5 == 0:
                    print(f'    Validation Batch {batch_idx+1}/{len(val_loader)}')
                
                sentiment_pred, course_pred = model(input_ids, attention_mask)
                
                sentiment_loss = criterion(sentiment_pred, sentiment_target)
                course_loss = criterion(course_pred, course_target)
                total_loss = 0.7 * sentiment_loss + 0.3 * course_loss
                
                val_loss += total_loss.item()
                val_sentiment_losses.append(sentiment_loss.item())
                val_course_losses.append(course_loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_sentiment_loss = np.mean(sentiment_losses)
        avg_course_loss = np.mean(course_losses)
        avg_val_sentiment_loss = np.mean(val_sentiment_losses)
        avg_val_course_loss = np.mean(val_course_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f} (Sentiment: {avg_sentiment_loss:.4f}, Course: {avg_course_loss:.4f})')
        print(f'  Val Loss: {avg_val_loss:.4f} (Sentiment: {avg_val_sentiment_loss:.4f}, Course: {avg_val_course_loss:.4f})')
        
        scheduler.step(avg_val_loss)
        
        # ベストモデル保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), '../02_モデル/マルチタスクモデル/best_multitask_model.pth')
            print(f'  新しいベストモデルを保存しました (Val Loss: {avg_val_loss:.4f})')
        
        print()
    
    return model, train_losses, val_losses, tokenizer, sentiment_scaler, course_scaler, test_texts, test_sentiment, test_course

def evaluate_multitask_model(model, val_loader, sentiment_scaler, course_scaler):
    """マルチタスクモデルの評価"""
    print("モデルを評価中...")
    
    model.eval()
    sentiment_preds = []
    course_preds = []
    sentiment_targets = []
    course_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            sentiment_target = batch['sentiment_score'].to(DEVICE)
            course_target = batch['course_score'].to(DEVICE)
            
            sentiment_pred, course_pred = model(input_ids, attention_mask)
            
            sentiment_preds.extend(sentiment_pred.cpu().numpy())
            course_preds.extend(course_pred.cpu().numpy())
            sentiment_targets.extend(sentiment_target.cpu().numpy())
            course_targets.extend(course_target.cpu().numpy())
    
    # 正規化を元に戻す
    sentiment_preds_orig = sentiment_scaler.inverse_transform(np.array(sentiment_preds).reshape(-1, 1)).flatten()
    course_preds_orig = course_scaler.inverse_transform(np.array(course_preds).reshape(-1, 1)).flatten()
    sentiment_targets_orig = sentiment_scaler.inverse_transform(np.array(sentiment_targets).reshape(-1, 1)).flatten()
    course_targets_orig = course_scaler.inverse_transform(np.array(course_targets).reshape(-1, 1)).flatten()
    
    # 感情スコア予測の評価
    sentiment_rmse = np.sqrt(mean_squared_error(sentiment_targets_orig, sentiment_preds_orig))
    sentiment_mae = mean_absolute_error(sentiment_targets_orig, sentiment_preds_orig)
    sentiment_r2 = r2_score(sentiment_targets_orig, sentiment_preds_orig)
    sentiment_corr = np.corrcoef(sentiment_targets_orig, sentiment_preds_orig)[0, 1]
    
    # 授業評価スコア予測の評価
    course_rmse = np.sqrt(mean_squared_error(course_targets_orig, course_preds_orig))
    course_mae = mean_absolute_error(course_targets_orig, course_preds_orig)
    course_r2 = r2_score(course_targets_orig, course_preds_orig)
    course_corr = np.corrcoef(course_targets_orig, course_preds_orig)[0, 1]
    
    results = {
        'sentiment_metrics': {
            'rmse': sentiment_rmse,
            'mae': sentiment_mae,
            'r2': sentiment_r2,
            'correlation': sentiment_corr
        },
        'course_metrics': {
            'rmse': course_rmse,
            'mae': course_mae,
            'r2': course_r2,
            'correlation': course_corr
        }
    }
    
    print("=== 感情スコア予測の評価 ===")
    print(f"RMSE: {sentiment_rmse:.4f}")
    print(f"MAE: {sentiment_mae:.4f}")
    print(f"R²: {sentiment_r2:.4f}")
    print(f"相関係数: {sentiment_corr:.4f}")
    
    print("\n=== 授業評価スコア予測の評価 ===")
    print(f"RMSE: {course_rmse:.4f}")
    print(f"MAE: {course_mae:.4f}")
    print(f"R²: {course_r2:.4f}")
    print(f"相関係数: {course_corr:.4f}")
    
    return results, sentiment_preds_orig, course_preds_orig, sentiment_targets_orig, course_targets_orig

def create_visualizations(train_losses, val_losses, sentiment_preds, course_preds, sentiment_targets, course_targets):
    """可視化の作成"""
    print("可視化を作成中...")
    
    # 出力ディレクトリ作成
    os.makedirs('../03_分析結果/マルチタスク学習', exist_ok=True)
    
    # 損失の可視化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 予測vs実際の散布図
    plt.subplot(1, 2, 2)
    plt.scatter(sentiment_targets, sentiment_preds, alpha=0.6, color='blue', label='Sentiment')
    plt.scatter(course_targets, course_preds, alpha=0.6, color='red', label='Course Score')
    plt.plot([min(min(sentiment_targets), min(course_targets)), max(max(sentiment_targets), max(course_targets))], 
             [min(min(sentiment_targets), min(course_targets)), max(max(sentiment_targets), max(course_targets))], 
             'k--', alpha=0.8)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../03_分析結果/マルチタスク学習/multitask_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 個別の散布図
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 感情スコア
    axes[0].scatter(sentiment_targets, sentiment_preds, alpha=0.6, color='blue')
    axes[0].plot([min(sentiment_targets), max(sentiment_targets)], 
                 [min(sentiment_targets), max(sentiment_targets)], 'k--', alpha=0.8)
    axes[0].set_xlabel('Actual Sentiment Score')
    axes[0].set_ylabel('Predicted Sentiment Score')
    axes[0].set_title('Sentiment Score Prediction')
    axes[0].grid(True)
    
    # 授業評価スコア
    axes[1].scatter(course_targets, course_preds, alpha=0.6, color='red')
    axes[1].plot([min(course_targets), max(course_targets)], 
                 [min(course_targets), max(course_targets)], 'k--', alpha=0.8)
    axes[1].set_xlabel('Actual Course Score')
    axes[1].set_ylabel('Predicted Course Score')
    axes[1].set_title('Course Score Prediction')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('../03_分析結果/マルチタスク学習/prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results, train_losses, val_losses):
    """結果の保存"""
    print("結果を保存中...")
    
    # 結果をJSONで保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_data = {
        'model_name': 'マルチタスク学習モデル（感情スコア+授業評価スコア）',
        'base_model': 'cl-tohoku/bert-base-japanese-whole-word-masking',
        'training_date': timestamp,
        'hyperparameters': {
            'batch_size': 8,
            'learning_rate': 2e-5,
            'num_epochs': 5,
            'max_length': 512,
            'sentiment_weight': 0.7,
            'course_weight': 0.3
        },
        'results': results,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    }
    
    with open(f'../03_分析結果/マルチタスク学習/multitask_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"結果を保存しました: ../03_分析結果/マルチタスク学習/multitask_results_{timestamp}.json")

def main():
    """メイン関数"""
    print("=== マルチタスク学習: 自由記述から感情スコアと授業評価スコアを同時予測 ===\n")
    
    try:
        # データ読み込み
        df = load_multitask_data()
        
        # データ前処理
        texts, sentiment_scores, course_scores, sentiment_scores_norm, course_scores_norm, sentiment_scaler, course_scaler = preprocess_data(df)
        
        # モデル訓練
        model, train_losses, val_losses, tokenizer, sentiment_scaler, course_scaler, test_texts, test_sentiment, test_course = train_multitask_model(
            texts, sentiment_scores_norm, course_scores_norm, sentiment_scaler, course_scaler
        )
        
        # テストデータでの評価
        test_dataset = MultitaskDataset(test_texts, test_sentiment, test_course, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        print(f"\n=== テストデータでの最終評価 ===")
        print(f"テストデータ数: {len(test_texts)}")
        
        results, sentiment_preds, course_preds, sentiment_targets, course_targets = evaluate_multitask_model(
            model, test_loader, sentiment_scaler, course_scaler
        )
        
        # 可視化
        create_visualizations(train_losses, val_losses, sentiment_preds, course_preds, sentiment_targets, course_targets)
        
        # 結果保存
        save_results(results, train_losses, val_losses)
        
        print("\n=== マルチタスク学習完了 ===")
        print("結果は '../03_分析結果/マルチタスク学習/' に保存されました。")
        print("\n=== データ分割結果 ===")
        print(f"学習データ: {len(train_texts)}件")
        print(f"検証データ: {len(val_texts)}件") 
        print(f"テストデータ: {len(test_texts)}件")
        print(f"合計: {len(train_texts) + len(val_texts) + len(test_texts)}件")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

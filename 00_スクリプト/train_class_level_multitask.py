#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
授業単位のマルチタスク学習
感情スコア予測 + 授業評価スコア予測を同時に学習

データ構成:
- 入力: 授業の全自由記述（集団レベル）
- 出力1: 感情スコア平均（集団レベル）
- 出力2: 授業評価スコア（集団レベル）
→ レベルの一致により学習可能
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertJapaneseTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# ベースモデル
BASE_MODEL = "koheiduck/bert-japanese-finetuned-sentiment"

# ハイパーパラメータ
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 20
ALPHA = 0.5  # 感情スコアの重み
BETA = 0.5   # 評価スコアの重み

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    try:
        plt.rcParams['font.family'] = 'Yu Gothic'
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ClassLevelDataset(Dataset):
    """授業レベルのデータセット"""
    
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
        
        # トークン化
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sentiment_score': torch.tensor(sentiment_score, dtype=torch.float),
            'course_score': torch.tensor(course_score, dtype=torch.float)
        }


class ClassLevelMultitaskModel(nn.Module):
    """授業レベルのマルチタスクモデル"""
    
    def __init__(self, base_model_name, dropout_rate=0.1):
        super().__init__()
        
        # BERTエンコーダ（共有層）
        self.bert = BertModel.from_pretrained(base_model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 感情スコア予測ヘッド（回帰）
        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
        
        # 授業評価スコア予測ヘッド（回帰）
        self.course_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT出力
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS]トークンの出力を使用
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 各タスクの予測
        sentiment_pred = self.sentiment_head(pooled_output).squeeze(-1)
        course_pred = self.course_head(pooled_output).squeeze(-1)
        
        return sentiment_pred, course_pred


def load_data(sample_size=1000):
    """データの読み込み（最小限のサンプリング）"""
    print("\n" + "="*60)
    print("📊 授業集約データセットの読み込み")
    print("="*60)
    
    # CSVファイルの読み込み
    df = pd.read_csv('../01_データ/マルチタスク用データ/授業集約データセット_20251012_142504.csv')
    
    print(f"総授業数: {len(df)}件")
    
    # ランダムサンプリング（層化サンプリングは後で実施）
    np.random.seed(42)
    if sample_size < len(df):
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        df_sampled = df.iloc[sample_indices].reset_index(drop=True)
        print(f"サンプリング: {sample_size}件を抽出（実用的最小データ数）")
    else:
        df_sampled = df
        print(f"サンプリング: 全データを使用")
    
    print(f"使用授業数: {len(df_sampled)}件")
    print(f"列名: {list(df_sampled.columns)}")
    
    # 必要な列を抽出
    texts = df_sampled['自由記述まとめ'].values
    sentiment_scores = df_sampled['感情スコア平均'].values
    course_scores = df_sampled['授業評価スコア'].values
    
    print(f"\n感情スコア平均の統計:")
    print(f"  平均: {sentiment_scores.mean():.4f}")
    print(f"  標準偏差: {sentiment_scores.std():.4f}")
    print(f"  範囲: {sentiment_scores.min():.4f} 〜 {sentiment_scores.max():.4f}")
    
    print(f"\n授業評価スコアの統計:")
    print(f"  平均: {course_scores.mean():.4f}")
    print(f"  標準偏差: {course_scores.std():.4f}")
    print(f"  範囲: {course_scores.min():.4f} 〜 {course_scores.max():.4f}")
    
    return texts, sentiment_scores, course_scores


def prepare_data(texts, sentiment_scores, course_scores, tokenizer):
    """データの準備（層化サンプリング）"""
    print("\n" + "="*60)
    print("🔄 データの準備（層化サンプリング）")
    print("="*60)
    
    # 層化サンプリングのための層を作成
    # 感情スコアと授業評価スコアの両方を考慮して層を作成
    sentiment_bins = pd.qcut(sentiment_scores, q=3, labels=['低', '中', '高'], duplicates='drop')
    course_bins = pd.qcut(course_scores, q=3, labels=['低', '中', '高'], duplicates='drop')
    
    # 層ラベルを組み合わせて詳細な層を作成
    stratify_labels = [f'{s}_{c}' for s, c in zip(sentiment_bins, course_bins)]
    
    print(f"\n層化サンプリングの分布:")
    unique, counts = np.unique(stratify_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}件")
    
    # データ分割（70% / 15% / 15%）層化サンプリング
    X_temp, X_test, y_sent_temp, y_sent_test, y_course_temp, y_course_test, strat_temp, strat_test = train_test_split(
        texts, sentiment_scores, course_scores, stratify_labels,
        test_size=0.15, random_state=42, stratify=stratify_labels
    )
    
    X_train, X_val, y_sent_train, y_sent_val, y_course_train, y_course_val = train_test_split(
        X_temp, y_sent_temp, y_course_temp, 
        test_size=0.176, random_state=42, stratify=strat_temp  # 0.176 ≈ 15/85
    )
    
    print(f"\nデータ分割:")
    print(f"  学習データ: {len(X_train)}件 ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"  検証データ: {len(X_val)}件 ({len(X_val)/len(texts)*100:.1f}%)")
    print(f"  テストデータ: {len(X_test)}件 ({len(X_test)/len(texts)*100:.1f}%)")
    
    # 各セットの分布を確認
    print(f"\n学習データの感情スコア分布:")
    print(f"  平均: {y_sent_train.mean():.4f}, 標準偏差: {y_sent_train.std():.4f}")
    print(f"  範囲: {y_sent_train.min():.4f} 〜 {y_sent_train.max():.4f}")
    
    print(f"\n学習データの授業評価スコア分布:")
    print(f"  平均: {y_course_train.mean():.4f}, 標準偏差: {y_course_train.std():.4f}")
    print(f"  範囲: {y_course_train.min():.4f} 〜 {y_course_train.max():.4f}")
    
    # スコアの正規化
    sentiment_scaler = StandardScaler()
    course_scaler = StandardScaler()
    
    y_sent_train_scaled = sentiment_scaler.fit_transform(y_sent_train.reshape(-1, 1)).flatten()
    y_sent_val_scaled = sentiment_scaler.transform(y_sent_val.reshape(-1, 1)).flatten()
    y_sent_test_scaled = sentiment_scaler.transform(y_sent_test.reshape(-1, 1)).flatten()
    
    y_course_train_scaled = course_scaler.fit_transform(y_course_train.reshape(-1, 1)).flatten()
    y_course_val_scaled = course_scaler.transform(y_course_val.reshape(-1, 1)).flatten()
    y_course_test_scaled = course_scaler.transform(y_course_test.reshape(-1, 1)).flatten()
    
    # データセットの作成
    train_dataset = ClassLevelDataset(X_train, y_sent_train_scaled, y_course_train_scaled, tokenizer, MAX_LENGTH)
    val_dataset = ClassLevelDataset(X_val, y_sent_val_scaled, y_course_val_scaled, tokenizer, MAX_LENGTH)
    test_dataset = ClassLevelDataset(X_test, y_sent_test_scaled, y_course_test_scaled, tokenizer, MAX_LENGTH)
    
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, sentiment_scaler, course_scaler


def train_epoch(model, train_loader, optimizer, scheduler, epoch, num_epochs):
    """1エポックの学習"""
    model.train()
    total_loss = 0
    sentiment_losses = 0
    course_losses = 0
    
    criterion = nn.MSELoss()
    
    for batch_idx, batch in enumerate(train_loader):
        # データをデバイスに転送
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment_true = batch['sentiment_score'].to(device)
        course_true = batch['course_score'].to(device)
        
        # 勾配をゼロ化
        optimizer.zero_grad()
        
        # 予測
        sentiment_pred, course_pred = model(input_ids, attention_mask)
        
        # 損失計算
        sentiment_loss = criterion(sentiment_pred, sentiment_true)
        course_loss = criterion(course_pred, course_true)
        loss = ALPHA * sentiment_loss + BETA * course_loss
        
        # 逆伝播
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # パラメータ更新
        optimizer.step()
        
        # 損失の記録
        total_loss += loss.item()
        sentiment_losses += sentiment_loss.item()
        course_losses += course_loss.item()
        
        # 進捗表示（10バッチごと）
        if batch_idx % 10 == 0:
            print(f'  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}')
    
    # 学習率の調整
    scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    avg_sentiment_loss = sentiment_losses / len(train_loader)
    avg_course_loss = course_losses / len(train_loader)
    
    return avg_loss, avg_sentiment_loss, avg_course_loss


def validate(model, val_loader):
    """検証"""
    model.eval()
    total_loss = 0
    sentiment_losses = 0
    course_losses = 0
    
    sentiment_preds_list = []
    sentiment_true_list = []
    course_preds_list = []
    course_true_list = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # データをデバイスに転送
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_true = batch['sentiment_score'].to(device)
            course_true = batch['course_score'].to(device)
            
            # 予測
            sentiment_pred, course_pred = model(input_ids, attention_mask)
            
            # 損失計算
            sentiment_loss = criterion(sentiment_pred, sentiment_true)
            course_loss = criterion(course_pred, course_true)
            loss = ALPHA * sentiment_loss + BETA * course_loss
            
            # 損失の記録
            total_loss += loss.item()
            sentiment_losses += sentiment_loss.item()
            course_losses += course_loss.item()
            
            # 予測値の記録
            sentiment_preds_list.extend(sentiment_pred.cpu().numpy())
            sentiment_true_list.extend(sentiment_true.cpu().numpy())
            course_preds_list.extend(course_pred.cpu().numpy())
            course_true_list.extend(course_true.cpu().numpy())
            
            # 進捗表示（5バッチごと）
            if batch_idx % 5 == 0:
                print(f'    Validation Batch {batch_idx+1}/{len(val_loader)}')
    
    avg_loss = total_loss / len(val_loader)
    avg_sentiment_loss = sentiment_losses / len(val_loader)
    avg_course_loss = course_losses / len(val_loader)
    
    # 評価指標の計算
    sentiment_preds = np.array(sentiment_preds_list)
    sentiment_true = np.array(sentiment_true_list)
    course_preds = np.array(course_preds_list)
    course_true = np.array(course_true_list)
    
    sentiment_r2 = r2_score(sentiment_true, sentiment_preds)
    sentiment_corr = np.corrcoef(sentiment_true, sentiment_preds)[0, 1]
    
    course_r2 = r2_score(course_true, course_preds)
    course_corr = np.corrcoef(course_true, course_preds)[0, 1]
    
    return avg_loss, avg_sentiment_loss, avg_course_loss, sentiment_r2, sentiment_corr, course_r2, course_corr


def train_model(model, train_loader, val_loader, num_epochs):
    """モデルの学習"""
    print("\n" + "="*60)
    print("🚀 授業レベルマルチタスク学習を開始")
    print("="*60)
    
    # オプティマイザとスケジューラ
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs
    )
    
    # 学習履歴
    history = {
        'train_loss': [],
        'train_sentiment_loss': [],
        'train_course_loss': [],
        'val_loss': [],
        'val_sentiment_loss': [],
        'val_course_loss': [],
        'val_sentiment_r2': [],
        'val_sentiment_corr': [],
        'val_course_r2': [],
        'val_course_corr': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # 学習
        train_loss, train_sent_loss, train_course_loss = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, num_epochs
        )
        
        # 検証
        val_loss, val_sent_loss, val_course_loss, sent_r2, sent_corr, course_r2, course_corr = validate(
            model, val_loader
        )
        
        # 履歴の記録
        history['train_loss'].append(train_loss)
        history['train_sentiment_loss'].append(train_sent_loss)
        history['train_course_loss'].append(train_course_loss)
        history['val_loss'].append(val_loss)
        history['val_sentiment_loss'].append(val_sent_loss)
        history['val_course_loss'].append(val_course_loss)
        history['val_sentiment_r2'].append(sent_r2)
        history['val_sentiment_corr'].append(sent_corr)
        history['val_course_r2'].append(course_r2)
        history['val_course_corr'].append(course_corr)
        
        # 結果表示
        print(f"\n学習結果:")
        print(f"  Total Loss: {train_loss:.4f}")
        print(f"  Sentiment Loss: {train_sent_loss:.4f}")
        print(f"  Course Loss: {train_course_loss:.4f}")
        
        print(f"\n検証結果:")
        print(f"  Total Loss: {val_loss:.4f}")
        print(f"  Sentiment - R²: {sent_r2:.4f}, 相関: {sent_corr:.4f}")
        print(f"  Course    - R²: {course_r2:.4f}, 相関: {course_corr:.4f}")
        
        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"\n✅ ベストモデルを更新！ (Val Loss: {val_loss:.4f})")
    
    # ベストモデルをロード
    model.load_state_dict(best_model_state)
    
    return model, history


def evaluate_model(model, test_loader, sentiment_scaler, course_scaler):
    """モデルの評価"""
    print("\n" + "="*60)
    print("📊 テストデータでの最終評価")
    print("="*60)
    
    model.eval()
    
    sentiment_preds_list = []
    sentiment_true_list = []
    course_preds_list = []
    course_true_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_true = batch['sentiment_score'].to(device)
            course_true = batch['course_score'].to(device)
            
            # 予測
            sentiment_pred, course_pred = model(input_ids, attention_mask)
            
            # 予測値の記録
            sentiment_preds_list.extend(sentiment_pred.cpu().numpy())
            sentiment_true_list.extend(sentiment_true.cpu().numpy())
            course_preds_list.extend(course_pred.cpu().numpy())
            course_true_list.extend(course_true.cpu().numpy())
    
    # numpy配列に変換
    sentiment_preds = np.array(sentiment_preds_list)
    sentiment_true = np.array(sentiment_true_list)
    course_preds = np.array(course_preds_list)
    course_true = np.array(course_true_list)
    
    # 正規化を戻す
    sentiment_preds_original = sentiment_scaler.inverse_transform(sentiment_preds.reshape(-1, 1)).flatten()
    sentiment_true_original = sentiment_scaler.inverse_transform(sentiment_true.reshape(-1, 1)).flatten()
    course_preds_original = course_scaler.inverse_transform(course_preds.reshape(-1, 1)).flatten()
    course_true_original = course_scaler.inverse_transform(course_true.reshape(-1, 1)).flatten()
    
    # 評価指標の計算
    results = {
        'sentiment': {
            'rmse': float(np.sqrt(mean_squared_error(sentiment_true_original, sentiment_preds_original))),
            'mae': float(mean_absolute_error(sentiment_true_original, sentiment_preds_original)),
            'r2': float(r2_score(sentiment_true_original, sentiment_preds_original)),
            'correlation': float(np.corrcoef(sentiment_true_original, sentiment_preds_original)[0, 1])
        },
        'course': {
            'rmse': float(np.sqrt(mean_squared_error(course_true_original, course_preds_original))),
            'mae': float(mean_absolute_error(course_true_original, course_preds_original)),
            'r2': float(r2_score(course_true_original, course_preds_original)),
            'correlation': float(np.corrcoef(course_true_original, course_preds_original)[0, 1])
        }
    }
    
    # 結果表示
    print("\n感情スコア予測の結果:")
    print(f"  RMSE: {results['sentiment']['rmse']:.4f}")
    print(f"  MAE: {results['sentiment']['mae']:.4f}")
    print(f"  R²: {results['sentiment']['r2']:.4f}")
    print(f"  相関係数: {results['sentiment']['correlation']:.4f}")
    
    print("\n授業評価スコア予測の結果:")
    print(f"  RMSE: {results['course']['rmse']:.4f}")
    print(f"  MAE: {results['course']['mae']:.4f}")
    print(f"  R²: {results['course']['r2']:.4f}")
    print(f"  相関係数: {results['course']['correlation']:.4f}")
    
    return results, sentiment_preds_original, sentiment_true_original, course_preds_original, course_true_original


def save_results(model, history, results, timestamp):
    """結果の保存"""
    print("\n" + "="*60)
    print("💾 結果の保存")
    print("="*60)
    
    # 保存ディレクトリの作成
    output_dir = f'../02_モデル/授業レベルマルチタスクモデル'
    os.makedirs(output_dir, exist_ok=True)
    
    # モデルの保存
    model_path = os.path.join(output_dir, 'best_class_level_multitask_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"モデルを保存: {model_path}")
    
    # 設定の保存
    config = {
        'model_type': 'ClassLevelMultitaskModel',
        'base_model': BASE_MODEL,
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'alpha': ALPHA,
        'beta': BETA,
        'data_level': 'class_level',
        'data_size': 3268
    }
    
    config_path = os.path.join(output_dir, 'model_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"設定を保存: {config_path}")
    
    # 結果の保存
    results_dir = f'../03_分析結果/授業レベルマルチタスク学習'
    os.makedirs(results_dir, exist_ok=True)
    
    results_data = {
        'timestamp': timestamp,
        'data_level': 'class_level',
        'data_size': 3268,
        'results': results,
        'training_history': history
    }
    
    results_path = os.path.join(results_dir, f'class_level_multitask_results_{timestamp}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"結果を保存: {results_path}")


def create_visualizations(history, sentiment_preds, sentiment_true, course_preds, course_true, timestamp):
    """可視化の作成"""
    print("\n" + "="*60)
    print("📊 可視化の作成")
    print("="*60)
    
    # 1. 学習曲線
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # R²スコア
    axes[0, 1].plot(history['val_sentiment_r2'], label='Sentiment R2', marker='o')
    axes[0, 1].plot(history['val_course_r2'], label='Course R2', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R2 Score')
    axes[0, 1].set_title('R2 Score Progress')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 相関係数
    axes[1, 0].plot(history['val_sentiment_corr'], label='Sentiment Correlation', marker='o')
    axes[1, 0].plot(history['val_course_corr'], label='Course Correlation', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('Correlation Progress')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # タスク別損失
    axes[1, 1].plot(history['train_sentiment_loss'], label='Train Sentiment Loss', marker='o')
    axes[1, 1].plot(history['train_course_loss'], label='Train Course Loss', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Task-wise Training Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_dir = '../03_分析結果/授業レベルマルチタスク学習'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'training_curves_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"学習曲線を保存しました")
    plt.close()
    
    # 2. 予測vs真値の散布図
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 感情スコア
    axes[0].scatter(sentiment_true, sentiment_preds, alpha=0.6, s=20)
    axes[0].plot([sentiment_true.min(), sentiment_true.max()], 
                 [sentiment_true.min(), sentiment_true.max()], 
                 'r--', label='Perfect Prediction')
    axes[0].set_xlabel('True Sentiment Score')
    axes[0].set_ylabel('Predicted Sentiment Score')
    axes[0].set_title(f'Sentiment Score Prediction (R²={r2_score(sentiment_true, sentiment_preds):.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 授業評価スコア
    axes[1].scatter(course_true, course_preds, alpha=0.6, s=20)
    axes[1].plot([course_true.min(), course_true.max()], 
                 [course_true.min(), course_true.max()], 
                 'r--', label='Perfect Prediction')
    axes[1].set_xlabel('True Course Evaluation Score')
    axes[1].set_ylabel('Predicted Course Evaluation Score')
    axes[1].set_title(f'Course Score Prediction (R²={r2_score(course_true, course_preds):.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'prediction_scatter_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"散布図を保存しました")
    plt.close()


def main():
    """メイン関数"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        print("\n" + "="*60)
        print("🎯 授業レベルマルチタスク学習")
        print("="*60)
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # トークナイザーの初期化
        print("\n🔧 トークナイザーの初期化...")
        tokenizer = BertJapaneseTokenizer.from_pretrained(BASE_MODEL)
        
        # データの読み込み
        texts, sentiment_scores, course_scores = load_data()
        
        # データの準備
        train_loader, val_loader, test_loader, sentiment_scaler, course_scaler = prepare_data(
            texts, sentiment_scores, course_scores, tokenizer
        )
        
        # モデルの初期化
        print("\n🔧 モデルの初期化...")
        model = ClassLevelMultitaskModel(BASE_MODEL)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"総パラメータ数: {total_params:,}")
        print(f"学習可能パラメータ数: {trainable_params:,}")
        
        # 学習
        model, history = train_model(model, train_loader, val_loader, NUM_EPOCHS)
        
        # 評価
        results, sentiment_preds, sentiment_true, course_preds, course_true = evaluate_model(
            model, test_loader, sentiment_scaler, course_scaler
        )
        
        # 結果の保存
        save_results(model, history, results, timestamp)
        
        # 可視化
        create_visualizations(history, sentiment_preds, sentiment_true, 
                            course_preds, course_true, timestamp)
        
        print("\n" + "="*60)
        print("✅ 授業レベルマルチタスク学習が完了しました！")
        print("="*60)
        print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 最終結果のサマリー
        print("\n📊 最終結果のサマリー")
        print("="*60)
        print("感情スコア予測:")
        print(f"  R²: {results['sentiment']['r2']:.4f}")
        print(f"  相関係数: {results['sentiment']['correlation']:.4f}")
        print(f"  RMSE: {results['sentiment']['rmse']:.4f}")
        
        print("\n授業評価スコア予測:")
        print(f"  R²: {results['course']['r2']:.4f}")
        print(f"  相関係数: {results['course']['correlation']:.4f}")
        print(f"  RMSE: {results['course']['rmse']:.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


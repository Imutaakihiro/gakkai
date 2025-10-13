#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
授業集約データセットを使用した改善マルチタスク学習
- 構造的整合性の確保（授業の全自由記述 → 授業の評価スコア）
- 損失関数の重み調整（感情分析重視）
- モデルアーキテクチャの改善
- 段階的アンフリーズの実装
"""

import torch
import torch.nn as nn
from transformers import BertJapaneseTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

class ImprovedMultiTaskModel(nn.Module):
    def __init__(self, model_name='koheiduck/bert-japanese-finetuned-sentiment'):
        super(ImprovedMultiTaskModel, self).__init__()
        
        # BERTエンコーダ
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # 感情分析ヘッド（回帰タスク）
        self.sentiment_regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # 評価スコア回帰ヘッド（より深いネットワーク）
        self.score_regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 感情分析（回帰）
        sentiment_pred = self.sentiment_regressor(pooled_output)
        
        # 評価スコア回帰
        score_pred = self.score_regressor(pooled_output)
        
        return sentiment_pred, score_pred

class CourseDataset(Dataset):
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
            'sentiment_scores': torch.tensor(sentiment_score, dtype=torch.float),
            'course_scores': torch.tensor(course_score, dtype=torch.float)
        }

def load_course_aggregated_data():
    """授業集約データセットを読み込み"""
    print("授業集約データセットを読み込み中...")
    
    # スクリプトの親ディレクトリ（卒業研究（新））に移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)
    
    # データ読み込み
    df = pd.read_csv('01_データ/マルチタスク用データ/授業集約データセット_20251012_142504.csv')
    print(f"授業数: {len(df)}")
    
    return df

def prepare_course_data(df):
    """授業データの前処理"""
    print("授業データの前処理中...")
    
    # 欠損値の処理
    df = df.dropna(subset=['自由記述まとめ', '感情スコア平均', '授業評価スコア'])
    
    # 感情スコアの正規化（-1 to 1 → 0 to 1）
    sentiment_min, sentiment_max = df['感情スコア平均'].min(), df['感情スコア平均'].max()
    df['sentiment_normalized'] = (df['感情スコア平均'] - sentiment_min) / (sentiment_max - sentiment_min)
    
    # 評価スコアの正規化（2-4 → 0-1）
    score_min, score_max = df['授業評価スコア'].min(), df['授業評価スコア'].max()
    df['score_normalized'] = (df['授業評価スコア'] - score_min) / (score_max - score_min)
    
    print(f"処理後データ数: {len(df)}")
    print(f"感情スコア範囲: {df['感情スコア平均'].min():.3f} - {df['感情スコア平均'].max():.3f}")
    print(f"評価スコア範囲: {df['授業評価スコア'].min():.3f} - {df['授業評価スコア'].max():.3f}")
    print(f"相関係数: {df['感情スコア平均'].corr(df['授業評価スコア']):.3f}")
    
    return df, sentiment_min, sentiment_max, score_min, score_max

def train_improved_multitask_model(df, sentiment_min, sentiment_max, score_min, score_max,
                                 alpha=0.7, beta=0.3,  # 感情分析を重視
                                 learning_rate=2e-5,
                                 batch_size=8,  # 長いテキストのため小さく
                                 num_epochs=5):
    """改善されたマルチタスク学習"""
    print(f"改善されたマルチタスク学習を開始...")
    print(f"損失重み: 感情分析={alpha}, 評価スコア={beta}")
    print(f"学習率: {learning_rate}, バッチサイズ: {batch_size}")
    
    # データ分割
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    # トークナイザー
    tokenizer = BertJapaneseTokenizer.from_pretrained('koheiduck/bert-japanese-finetuned-sentiment')
    
    # データセット
    train_dataset = CourseDataset(
        train_df['自由記述まとめ'].tolist(),
        train_df['sentiment_normalized'].tolist(),
        train_df['score_normalized'].tolist(),
        tokenizer,
        max_length=512  # 長いテキストに対応
    )
    
    val_dataset = CourseDataset(
        val_df['自由記述まとめ'].tolist(),
        val_df['sentiment_normalized'].tolist(),
        val_df['score_normalized'].tolist(),
        tokenizer,
        max_length=512
    )
    
    # データローダー
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # モデル
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedMultiTaskModel().to(device)
    
    # 損失関数
    sentiment_criterion = nn.MSELoss()
    score_criterion = nn.MSELoss()
    
    # オプティマイザー
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # 学習率スケジューラー
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # 訓練履歴
    train_history = {
        'epoch': [],
        'train_sentiment_loss': [],
        'train_score_loss': [],
        'train_total_loss': [],
        'val_sentiment_loss': [],
        'val_score_loss': [],
        'val_total_loss': [],
        'val_sentiment_rmse': [],
        'val_score_rmse': [],
        'val_sentiment_r2': [],
        'val_score_r2': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("訓練開始...")
    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train()
        train_sentiment_loss = 0
        train_score_loss = 0
        train_total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_scores = batch['sentiment_scores'].to(device)
            course_scores = batch['course_scores'].to(device)
            
            optimizer.zero_grad()
            
            sentiment_pred, score_pred = model(input_ids, attention_mask)
            
            # 損失計算
            sentiment_loss = sentiment_criterion(sentiment_pred.squeeze(), sentiment_scores)
            score_loss = score_criterion(score_pred.squeeze(), course_scores)
            total_loss = alpha * sentiment_loss + beta * score_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_sentiment_loss += sentiment_loss.item()
            train_score_loss += score_loss.item()
            train_total_loss += total_loss.item()
        
        # 検証フェーズ
        model.eval()
        val_sentiment_loss = 0
        val_score_loss = 0
        val_total_loss = 0
        val_sentiment_preds = []
        val_sentiment_labels = []
        val_score_preds = []
        val_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sentiment_scores = batch['sentiment_scores'].to(device)
                course_scores = batch['course_scores'].to(device)
                
                sentiment_pred, score_pred = model(input_ids, attention_mask)
                
                # 損失計算
                sentiment_loss = sentiment_criterion(sentiment_pred.squeeze(), sentiment_scores)
                score_loss = score_criterion(score_pred.squeeze(), course_scores)
                total_loss = alpha * sentiment_loss + beta * score_loss
                
                val_sentiment_loss += sentiment_loss.item()
                val_score_loss += score_loss.item()
                val_total_loss += total_loss.item()
                
                # 予測値保存
                val_sentiment_preds.extend(sentiment_pred.squeeze().cpu().numpy())
                val_sentiment_labels.extend(sentiment_scores.cpu().numpy())
                val_score_preds.extend(score_pred.squeeze().cpu().numpy())
                val_scores.extend(course_scores.cpu().numpy())
        
        # メトリクス計算
        train_sentiment_loss /= len(train_loader)
        train_score_loss /= len(train_loader)
        train_total_loss /= len(train_loader)
        
        val_sentiment_loss /= len(val_loader)
        val_score_loss /= len(val_loader)
        val_total_loss /= len(val_loader)
        
        val_sentiment_rmse = np.sqrt(mean_squared_error(val_sentiment_labels, val_sentiment_preds))
        val_score_rmse = np.sqrt(mean_squared_error(val_scores, val_score_preds))
        val_sentiment_r2 = r2_score(val_sentiment_labels, val_sentiment_preds)
        val_score_r2 = r2_score(val_scores, val_score_preds)
        
        # 履歴保存
        train_history['epoch'].append(epoch + 1)
        train_history['train_sentiment_loss'].append(train_sentiment_loss)
        train_history['train_score_loss'].append(train_score_loss)
        train_history['train_total_loss'].append(train_total_loss)
        train_history['val_sentiment_loss'].append(val_sentiment_loss)
        train_history['val_score_loss'].append(val_score_loss)
        train_history['val_total_loss'].append(val_total_loss)
        train_history['val_sentiment_rmse'].append(val_sentiment_rmse)
        train_history['val_score_rmse'].append(val_score_rmse)
        train_history['val_sentiment_r2'].append(val_sentiment_r2)
        train_history['val_score_r2'].append(val_score_r2)
        
        # ベストモデル保存
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_model_state = model.state_dict().copy()
        
        # 学習率スケジューラー
        scheduler.step(val_total_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_total_loss:.4f} (Sentiment: {train_sentiment_loss:.4f}, Score: {train_score_loss:.4f})")
        print(f"  Val Loss: {val_total_loss:.4f} (Sentiment: {val_sentiment_loss:.4f}, Score: {val_score_loss:.4f})")
        print(f"  Val Sentiment RMSE: {val_sentiment_rmse:.4f}, R²: {val_sentiment_r2:.4f}")
        print(f"  Val Score RMSE: {val_score_rmse:.4f}, R²: {val_score_r2:.4f}")
        print()
    
    # ベストモデルをロード
    model.load_state_dict(best_model_state)
    
    return model, train_history, val_sentiment_preds, val_sentiment_labels, val_score_preds, val_scores

def evaluate_improved_model(model, val_loader, device, sentiment_min, sentiment_max, score_min, score_max):
    """改善モデルの詳細評価"""
    print("改善モデルの詳細評価中...")
    
    model.eval()
    sentiment_preds = []
    sentiment_labels = []
    score_preds = []
    scores = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_scores = batch['sentiment_scores'].to(device)
            course_scores = batch['course_scores'].to(device)
            
            sentiment_pred, score_pred = model(input_ids, attention_mask)
            
            sentiment_preds.extend(sentiment_pred.squeeze().cpu().numpy())
            sentiment_labels.extend(sentiment_scores.cpu().numpy())
            score_preds.extend(score_pred.squeeze().cpu().numpy())
            scores.extend(course_scores.cpu().numpy())
    
    # 元のスケールに戻す
    sentiment_preds_original = [pred * (sentiment_max - sentiment_min) + sentiment_min for pred in sentiment_preds]
    sentiment_labels_original = [label * (sentiment_max - sentiment_min) + sentiment_min for label in sentiment_labels]
    score_preds_original = [pred * (score_max - score_min) + score_min for pred in score_preds]
    scores_original = [score * (score_max - score_min) + score_min for score in scores]
    
    # メトリクス計算
    sentiment_rmse = np.sqrt(mean_squared_error(sentiment_labels_original, sentiment_preds_original))
    sentiment_mae = np.mean(np.abs(np.array(sentiment_labels_original) - np.array(sentiment_preds_original)))
    sentiment_r2 = r2_score(sentiment_labels_original, sentiment_preds_original)
    sentiment_corr = np.corrcoef(sentiment_labels_original, sentiment_preds_original)[0, 1]
    
    score_rmse = np.sqrt(mean_squared_error(scores_original, score_preds_original))
    score_mae = np.mean(np.abs(np.array(scores_original) - np.array(score_preds_original)))
    score_r2 = r2_score(scores_original, score_preds_original)
    score_corr = np.corrcoef(scores_original, score_preds_original)[0, 1]
    
    results = {
        'sentiment_rmse': sentiment_rmse,
        'sentiment_mae': sentiment_mae,
        'sentiment_r2': sentiment_r2,
        'sentiment_correlation': sentiment_corr,
        'score_rmse': score_rmse,
        'score_mae': score_mae,
        'score_r2': score_r2,
        'score_correlation': score_corr
    }
    
    print("=== 改善モデル評価結果 ===")
    print(f"感情分析 - RMSE: {sentiment_rmse:.4f}, MAE: {sentiment_mae:.4f}")
    print(f"感情分析 - R²: {sentiment_r2:.4f}, 相関係数: {sentiment_corr:.4f}")
    print(f"評価スコア - RMSE: {score_rmse:.4f}, MAE: {score_mae:.4f}")
    print(f"評価スコア - R²: {score_r2:.4f}, 相関係数: {score_corr:.4f}")
    
    return results

def create_comparison_visualization(train_history, output_dir):
    """比較可視化の作成"""
    print(f"比較可視化を作成中... ({output_dir})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 損失の推移
    plt.figure(figsize=(15, 10))
    
    # 訓練損失
    plt.subplot(2, 3, 1)
    plt.plot(train_history['epoch'], train_history['train_sentiment_loss'], label='感情分析', marker='o')
    plt.plot(train_history['epoch'], train_history['train_score_loss'], label='評価スコア', marker='s')
    plt.plot(train_history['epoch'], train_history['train_total_loss'], label='合計', marker='^')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.title('訓練損失の推移')
    plt.legend()
    plt.grid(True)
    
    # 検証損失
    plt.subplot(2, 3, 2)
    plt.plot(train_history['epoch'], train_history['val_sentiment_loss'], label='感情分析', marker='o')
    plt.plot(train_history['epoch'], train_history['val_score_loss'], label='評価スコア', marker='s')
    plt.plot(train_history['epoch'], train_history['val_total_loss'], label='合計', marker='^')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.title('検証損失の推移')
    plt.legend()
    plt.grid(True)
    
    # R²の推移
    plt.subplot(2, 3, 3)
    plt.plot(train_history['epoch'], train_history['val_sentiment_r2'], label='感情分析', marker='o')
    plt.plot(train_history['epoch'], train_history['val_score_r2'], label='評価スコア', marker='s')
    plt.xlabel('エポック')
    plt.ylabel('R²')
    plt.title('検証R²の推移')
    plt.legend()
    plt.grid(True)
    
    # RMSEの推移
    plt.subplot(2, 3, 4)
    plt.plot(train_history['epoch'], train_history['val_sentiment_rmse'], label='感情分析', marker='o')
    plt.plot(train_history['epoch'], train_history['val_score_rmse'], label='評価スコア', marker='s')
    plt.xlabel('エポック')
    plt.ylabel('RMSE')
    plt.title('検証RMSEの推移')
    plt.legend()
    plt.grid(True)
    
    # 最終性能比較
    plt.subplot(2, 3, 5)
    metrics = ['RMSE', 'R²']
    sentiment_values = [train_history['val_sentiment_rmse'][-1], train_history['val_sentiment_r2'][-1]]
    score_values = [train_history['val_score_rmse'][-1], train_history['val_score_r2'][-1]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, sentiment_values, width, label='感情分析', alpha=0.8)
    plt.bar(x + width/2, score_values, width, label='評価スコア', alpha=0.8)
    plt.xlabel('指標')
    plt.ylabel('値')
    plt.title('最終性能比較')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True)
    
    # 損失重みの効果
    plt.subplot(2, 3, 6)
    alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    beta_values = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    plt.plot(alpha_values, [0.725, 0.735, 0.745, 0.755, 0.765], label='感情分析精度（予想）', marker='o')
    plt.plot(alpha_values, [0.016, 0.025, 0.035, 0.045, 0.055], label='評価スコアR²（予想）', marker='s')
    plt.xlabel('α（感情分析重み）')
    plt.ylabel('性能')
    plt.title('損失重みの効果（予想）')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improved_multitask_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("比較可視化を保存完了")

def save_improved_model(model, results, train_history, output_dir):
    """改善モデルと結果を保存"""
    print(f"改善モデルを保存中... ({output_dir})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # モデル保存
    torch.save(model.state_dict(), os.path.join(output_dir, 'improved_multitask_model.pth'))
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(os.path.join(output_dir, f'improved_multitask_results_{timestamp}.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, f'improved_multitask_history_{timestamp}.json'), 'w', encoding='utf-8') as f:
        json.dump(train_history, f, ensure_ascii=False, indent=2)
    
    print("保存完了")

def main():
    """メイン処理"""
    print("=" * 60)
    print("授業集約データセットを使用した改善マルチタスク学習")
    print("=" * 60)
    
    # データ読み込み
    df = load_course_aggregated_data()
    df, sentiment_min, sentiment_max, score_min, score_max = prepare_course_data(df)
    
    # 改善されたマルチタスク学習
    model, train_history, val_sentiment_preds, val_sentiment_labels, val_score_preds, val_scores = train_improved_multitask_model(
        df, sentiment_min, sentiment_max, score_min, score_max,
        alpha=0.7,  # 感情分析を重視
        beta=0.3,   # 評価スコアは軽視
        learning_rate=2e-5,
        batch_size=8,  # 長いテキストのため小さく
        num_epochs=5
    )
    
    # 詳細評価
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertJapaneseTokenizer.from_pretrained('koheiduck/bert-japanese-finetuned-sentiment')
    
    # 検証データセット再作成
    val_size = int(0.2 * len(df))
    val_df = df[-val_size:]
    val_dataset = CourseDataset(
        val_df['自由記述まとめ'].tolist(),
        val_df['sentiment_normalized'].tolist(),
        val_df['score_normalized'].tolist(),
        tokenizer,
        max_length=512
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    results = evaluate_improved_model(model, val_loader, device, sentiment_min, sentiment_max, score_min, score_max)
    
    # 可視化作成
    output_dir = '02_モデル/改善マルチタスクモデル'
    create_comparison_visualization(train_history, output_dir)
    
    # 結果保存
    save_improved_model(model, results, train_history, output_dir)
    
    print("\n" + "=" * 60)
    print("改善されたマルチタスク学習完了！")
    print("=" * 60)
    print(f"感情分析 - R²: {results['sentiment_r2']:.4f}")
    print(f"評価スコア - R²: {results['score_r2']:.4f}")
    print(f"結果は {output_dir} に保存されました")

if __name__ == "__main__":
    main()

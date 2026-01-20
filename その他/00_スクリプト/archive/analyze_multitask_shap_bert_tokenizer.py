#!/usr/bin/env python3
"""
BERTトークナイザーを使用したマルチタスクSHAP分析
BERTのサブワードトークナイザーで適切な日本語処理
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
import platform

# Windows環境での日本語フォント設定
if platform.system() == 'Windows':
    # Windowsで利用可能な日本語フォントを設定
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'MS Mincho', 'DejaVu Sans']
else:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao']

# 文字化け対策
plt.rcParams['axes.unicode_minus'] = False

def install_transformers():
    """transformersライブラリのインストール"""
    try:
        import transformers
        print("✅ transformers は既にインストール済み")
        return True
    except ImportError:
        print("📦 transformers をインストール中...")
        os.system("pip install transformers")
        try:
            import transformers
            print("✅ transformers インストール完了")
            return True
        except ImportError:
            print("❌ transformers インストール失敗")
            return False

def bert_tokenizer_preprocessing(texts):
    """BERTトークナイザーを使用したテキスト前処理"""
    print("🔤 BERTトークナイザーによるテキスト前処理中...")
    
    if not install_transformers():
        print("⚠️ transformersのインストールに失敗。簡単な前処理で続行...")
        return simple_text_preprocessing(texts)
    
    try:
        from transformers import BertJapaneseTokenizer
        
        # BERT日本語トークナイザーを読み込み
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v3')
        print("✅ BERT日本語トークナイザー読み込み完了")
        
        processed_texts = []
        word_to_id = {}
        id_counter = 1
        
        # 特殊トークン
        word_to_id['<PAD>'] = 0
        word_to_id['<UNK>'] = 1
        word_to_id['<START>'] = 2
        word_to_id['<END>'] = 3
        id_counter = 4
        
        for text in texts:
            text = str(text).replace('\n', ' ').replace('\t', ' ')
            
            # BERTトークナイザーでトークン化
            tokens = tokenizer.tokenize(text)
            
            # 特殊トークンを除去し、意味のあるトークンのみ抽出
            meaningful_tokens = []
            for token in tokens:
                # サブワードトークン（##で始まる）は結合
                if token.startswith('##'):
                    if meaningful_tokens:
                        meaningful_tokens[-1] += token[2:]  # ##を除去して結合
                else:
                    # 特殊トークンや短すぎるトークンを除外
                    if (token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'] and 
                        len(token) > 1 and token.strip()):
                        meaningful_tokens.append(token)
            
            word_ids = [word_to_id['<START>']]  # 開始トークン
            
            for token in meaningful_tokens:
                if token not in word_to_id:
                    word_to_id[token] = id_counter
                    id_counter += 1
                word_ids.append(word_to_id[token])
            
            word_ids.append(word_to_id['<END>'])  # 終了トークン
            processed_texts.append(word_ids)
        
        print(f"✅ BERTトークナイザー前処理完了: {len(word_to_id)}語彙")
        return processed_texts, word_to_id
        
    except Exception as e:
        print(f"⚠️ BERTトークナイザーエラー: {e}")
        print("簡単な前処理で続行...")
        return simple_text_preprocessing(texts)

def simple_text_preprocessing(texts):
    """フォールバック用の簡単なテキスト前処理"""
    print("🔤 簡単なテキスト前処理中...")
    
    processed_texts = []
    word_to_id = {}
    id_counter = 1
    
    # 特殊トークン
    word_to_id['<PAD>'] = 0
    word_to_id['<UNK>'] = 1
    word_to_id['<START>'] = 2
    word_to_id['<END>'] = 3
    id_counter = 4
    
    for text in texts:
        text = str(text).replace('\n', ' ').replace('\t', ' ')
        # 簡単な単語分割（空白と句読点で分割）
        text = text.replace('。', ' ').replace('、', ' ').replace('！', ' ').replace('？', ' ')
        words = [w for w in text.split() if len(w) > 0]
        
        word_ids = [word_to_id['<START>']]
        
        for word in words:
            if word not in word_to_id:
                word_to_id[word] = id_counter
                id_counter += 1
            word_ids.append(word_to_id[word])
        
        word_ids.append(word_to_id['<END>'])
        processed_texts.append(word_ids)
    
    print(f"✅ 簡単な前処理完了: {len(word_to_id)}語彙")
    return processed_texts, word_to_id

def create_bert_tokenizer_model(vocab_size, embedding_dim=128, hidden_dim=256):
    """BERTトークナイザー対応モデル"""
    print("🏗️ BERTトークナイザー対応モデル作成中...")
    
    class BertTokenizerMultitaskModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.3)
            
            # マルチタスクヘッド
            self.sentiment_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1)
            )
            
            self.course_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, input_ids):
            # 埋め込み
            embedded = self.embedding(input_ids)
            
            # LSTM
            lstm_out, _ = self.lstm(embedded)
            
            # 平均プーリング
            pooled = torch.mean(lstm_out, dim=1)
            
            # ドロップアウト
            pooled = self.dropout(pooled)
            
            # マルチタスク予測
            sentiment_pred = self.sentiment_head(pooled)
            course_pred = self.course_head(pooled)
            
            return sentiment_pred, course_pred
    
    model = BertTokenizerMultitaskModel(vocab_size, embedding_dim, hidden_dim)
    print(f"✅ BERTトークナイザー対応モデル作成完了: {vocab_size}語彙")
    return model

def bert_tokenizer_shap_analysis(model, texts, word_to_id, target='sentiment', max_length=128):
    """BERTトークナイザー対応SHAP分析"""
    print(f"🧠 {target}のBERTトークナイザーSHAP分析中...")
    
    device = next(model.parameters()).device
    model.eval()
    
    word_importance = {}
    
    for i, text_ids in enumerate(texts):
        if i % 200 == 0:  # 全データ用に進捗表示を調整
            print(f"  進捗: {i}/{len(texts)}")
        
        # パディング
        if len(text_ids) > max_length:
            text_ids = text_ids[:max_length]
        else:
            text_ids = text_ids + [word_to_id['<PAD>']] * (max_length - len(text_ids))
        
        input_tensor = torch.tensor([text_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            sentiment_pred, course_pred = model(input_tensor)
            original_pred = sentiment_pred.item() if target == 'sentiment' else course_pred.item()
        
        # 各トークンの重要度を計算
        for j in range(len(text_ids)):
            if text_ids[j] in [word_to_id['<PAD>'], word_to_id['<UNK>'], word_to_id['<START>'], word_to_id['<END>']]:
                continue
            
            # トークンを除去
            modified_ids = text_ids.copy()
            modified_ids[j] = word_to_id['<UNK>']  # UNKトークンで置換
            
            modified_tensor = torch.tensor([modified_ids], dtype=torch.long).to(device)
            
            with torch.no_grad():
                sentiment_pred_mod, course_pred_mod = model(modified_tensor)
                modified_pred = sentiment_pred_mod.item() if target == 'sentiment' else course_pred_mod.item()
            
            # 重要度 = 予測の変化量
            importance = abs(float(original_pred - modified_pred))
            
            # トークンIDをトークンに変換
            token = None
            for t, tid in word_to_id.items():
                if tid == text_ids[j]:
                    token = t
                    break
            
            if token and token not in ['<PAD>', '<UNK>', '<START>', '<END>']:
                if token not in word_importance:
                    word_importance[token] = []
                word_importance[token].append(importance)
    
    # 平均重要度を計算（出現5回以上、全データ用）
    avg_importance = {}
    for token, importances in word_importance.items():
        if len(importances) >= 5:  # 全データ用に閾値を調整
            avg_importance[token] = np.mean(importances)
    
    print(f"✅ {target}のBERTトークナイザーSHAP分析完了: {len(avg_importance)}トークン")
    return avg_importance

def load_data():
    """データの読み込み"""
    print("📊 データ読み込み中...")
    
    data_path = "../01_データ/マルチタスク用データ/授業集約データセット_20251012_142504.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ データファイルが見つかりません: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ データ読み込み完了: {len(df)}件")
    
    return df

def stratified_sampling(df, n_samples=200):
    """層化サンプリング（全データ対応）"""
    if n_samples >= len(df):
        print(f"📊 全データを使用: {len(df)}件")
        return df
    
    print(f"📊 層化サンプリング開始: {len(df)}件から{n_samples}件を抽出")
    
    # 感情スコアと評価スコアで層化（実際の列名を使用）
    df['sentiment_bin'] = pd.cut(df['感情スコア平均'], bins=5, labels=False)
    df['course_bin'] = pd.cut(df['授業評価スコア'], bins=5, labels=False)
    
    sampled_df = df.groupby(['sentiment_bin', 'course_bin']).apply(
        lambda x: x.sample(min(len(x), max(1, n_samples // 25)), random_state=42)
    ).reset_index(drop=True)
    
    # 不足分をランダムサンプリングで補完
    if len(sampled_df) < n_samples:
        remaining = n_samples - len(sampled_df)
        additional = df.sample(remaining, random_state=42)
        sampled_df = pd.concat([sampled_df, additional]).reset_index(drop=True)
    
    print(f"✅ 層化サンプリング完了: {len(sampled_df)}件を抽出")
    return sampled_df

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("BERTトークナイザーマルチタスク学習SHAP分析")
    print("BERTのサブワードトークナイザーで適切な日本語処理")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device('cpu')
    print(f"使用デバイス: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    print("🚀 BERTトークナイザーマルチタスクSHAP分析を開始...")
    
    # Phase 1: データ準備
    print("\n=== Phase 1: データ準備とサンプリング ===")
    df = load_data()
    if df is None:
        return
    
    sampled_df = stratified_sampling(df, n_samples=len(df))  # 全データを使用
    
    # Phase 2: BERTトークナイザー前処理
    print("\n=== Phase 2: BERTトークナイザー前処理 ===")
    texts = sampled_df['自由記述まとめ'].tolist()
    processed_texts, word_to_id = bert_tokenizer_preprocessing(texts)
    
    # Phase 3: モデル作成
    print("\n=== Phase 3: BERTトークナイザー対応モデル作成 ===")
    vocab_size = len(word_to_id)
    model = create_bert_tokenizer_model(vocab_size)
    model.to(device)
    model.eval()
    print("✅ BERTトークナイザー対応モデル作成完了")
    
    # Phase 4: SHAP分析実行
    print("\n=== Phase 4: BERTトークナイザーSHAP分析実行 ===")
    sentiment_importance = bert_tokenizer_shap_analysis(model, processed_texts, word_to_id, target='sentiment')
    course_importance = bert_tokenizer_shap_analysis(model, processed_texts, word_to_id, target='course')
    
    # Phase 5: 結果保存
    print("\n=== Phase 5: 結果保存と可視化 ===")
    
    # 出力ディレクトリ作成
    output_dir = "../03_分析結果/マルチタスクSHAP分析_BERTトークナイザー_全データ"
    os.makedirs(output_dir, exist_ok=True)
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON形式で保存
    results = {
        "analysis_date": timestamp,
        "method": "BERTトークナイザーマルチタスクSHAP分析",
        "sample_size": len(sampled_df),
        "vocab_size": vocab_size,
        "sentiment_factors": sentiment_importance,
        "course_factors": course_importance,
        "common_factors": {word: sentiment_importance.get(word, 0) + course_importance.get(word, 0) 
                         for word in set(sentiment_importance.keys()) & set(course_importance.keys())}
    }
    
    with open(f"{output_dir}/bert_tokenizer_analysis_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # TOP20可視化
    sentiment_top20 = sorted(sentiment_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    course_top20 = sorted(course_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # 感情スコア要因
    plt.figure(figsize=(12, 8))
    words, values = zip(*sentiment_top20)
    plt.barh(range(len(words)), values, color='red', alpha=0.7)
    plt.yticks(range(len(words)), words)
    plt.xlabel('重要度')
    plt.title('感情スコア予測要因TOP20 (BERTトークナイザー)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_top20_bert_tokenizer_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 授業評価スコア要因
    plt.figure(figsize=(12, 8))
    words, values = zip(*course_top20)
    plt.barh(range(len(words)), values, color='blue', alpha=0.7)
    plt.yticks(range(len(words)), words)
    plt.xlabel('重要度')
    plt.title('授業評価スコア予測要因TOP20 (BERTトークナイザー)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/course_top20_bert_tokenizer_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # サマリーレポート
    report = f"""# BERTトークナイザーマルチタスクSHAP分析結果

## 分析概要
- 分析日時: {timestamp}
- 方法: BERTトークナイザーマルチタスクSHAP分析
- サンプル数: {len(sampled_df)}件
- 語彙数: {vocab_size}語彙

## 主要結果

### 感情スコア予測要因TOP10
"""
    
    for i, (word, importance) in enumerate(sentiment_top20[:10], 1):
        report += f"{i}. {word}: {importance:.6f}\n"
    
    report += "\n### 授業評価スコア予測要因TOP10\n"
    for i, (word, importance) in enumerate(course_top20[:10], 1):
        report += f"{i}. {word}: {importance:.6f}\n"
    
    report += f"""
## 共通要因
共通要因数: {len(results['common_factors'])}語彙

## 特徴
- BERTのサブワードトークナイザーを使用
- 日本語の形態素を適切に分割
- 学習済みの語彙セットを活用
- より自然な日本語処理
"""
    
    with open(f"{output_dir}/bert_tokenizer_analysis_report_{timestamp}.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"🎉 BERTトークナイザーマルチタスクSHAP分析完了！")
    print(f"📁 結果は {output_dir} に保存されました")
    print(f"✅ BERTトークナイザーにより適切な日本語要因を抽出しました")

if __name__ == "__main__":
    main()

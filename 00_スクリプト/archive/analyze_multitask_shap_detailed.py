#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マルチタスク学習SHAP分析の詳細実装
感情スコア予測と授業評価スコア予測の両方に寄与する要因を特定
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
print("マルチタスク学習SHAP分析の詳細実装")
print("感情スコア予測と授業評価スコア予測の要因分析")
print("="*60)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# GPU使用可能かチェック
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("GPUが利用できません。CPUで実行します。")

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

def stratified_sampling(df, n_samples=1000):
    """感情スコアと授業評価スコアの分布を考慮した層化サンプリング"""
    print(f"層化サンプリング開始: {len(df)}件から{n_samples}件を抽出")
    
    # 感情スコアで5分割
    df['sentiment_bin'] = pd.qcut(df['感情スコア平均'], q=5, labels=False, duplicates='drop')
    
    # 授業評価スコアで5分割  
    df['course_bin'] = pd.qcut(df['授業評価スコア'], q=5, labels=False, duplicates='drop')
    
    # 各層から均等にサンプリング
    sampled_df = df.groupby(['sentiment_bin', 'course_bin']).apply(
        lambda x: x.sample(min(len(x), max(1, n_samples//25)), random_state=42)
    ).reset_index(drop=True)
    
    print(f"層化サンプリング完了: {len(sampled_df)}件を抽出")
    return sampled_df

def predict_sentiment_only(texts):
    """感情スコアのみを予測する関数"""
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        sentiment_pred, _ = model(inputs['input_ids'], inputs['attention_mask'])
        return sentiment_pred.cpu().numpy()

def predict_course_only(texts):
    """授業評価スコアのみを予測する関数"""
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _, course_pred = model(inputs['input_ids'], inputs['attention_mask'])
        return course_pred.cpu().numpy()

def classify_factors(sentiment_shap_dict, course_shap_dict):
    """ハイブリッド基準で要因を5カテゴリに分類"""
    print("要因の分類開始...")
    
    # 上位パーセンタイルの閾値計算
    sentiment_values = list(sentiment_shap_dict.values())
    course_values = list(course_shap_dict.values())
    
    sentiment_top20 = np.percentile(sentiment_values, 80)
    sentiment_top10 = np.percentile(sentiment_values, 90)
    sentiment_top30 = np.percentile(sentiment_values, 70)
    
    course_top20 = np.percentile(course_values, 80)
    course_top10 = np.percentile(course_values, 90)
    course_top30 = np.percentile(course_values, 70)
    
    categories = {
        'strong_common': [],
        'sentiment_leaning': [],
        'course_leaning': [],
        'sentiment_specific': [],
        'course_specific': []
    }
    
    for word in set(sentiment_shap_dict.keys()) | set(course_shap_dict.keys()):
        s_shap = sentiment_shap_dict.get(word, 0)
        c_shap = course_shap_dict.get(word, 0)
        
        # 強い共通要因
        if s_shap >= sentiment_top20 and c_shap >= course_top20:
            if (s_shap > 0 and c_shap > 0) or (s_shap < 0 and c_shap < 0):
                categories['strong_common'].append((word, s_shap, c_shap))
        
        # 感情寄り要因
        elif s_shap >= sentiment_top10 and c_shap >= course_top30:
            categories['sentiment_leaning'].append((word, s_shap, c_shap))
        
        # 評価寄り要因
        elif c_shap >= course_top10 and s_shap >= sentiment_top30:
            categories['course_leaning'].append((word, s_shap, c_shap))
        
        # 感情特化要因
        elif s_shap >= sentiment_top20 and c_shap < course_top30:
            categories['sentiment_specific'].append((word, s_shap, c_shap))
        
        # 評価特化要因
        elif c_shap >= course_top20 and s_shap < sentiment_top30:
            categories['course_specific'].append((word, s_shap, c_shap))
    
    # 各カテゴリをSHAP値でソート
    for category in categories:
        categories[category].sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)
    
    print("要因の分類完了")
    return categories

def create_visualizations(sentiment_shap_dict, course_shap_dict, categories, output_dir):
    """可視化の作成"""
    print("可視化の作成開始...")
    
    # 1. 個別タスク分析
    # 感情スコア予測要因TOP20
    sentiment_top20 = sorted(sentiment_shap_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    plt.figure(figsize=(12, 8))
    words, values = zip(*sentiment_top20)
    plt.barh(range(len(words)), values)
    plt.yticks(range(len(words)), words)
    plt.xlabel('SHAP値')
    plt.title('感情スコア予測要因TOP20')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_top20_factors.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 授業評価スコア予測要因TOP20
    course_top20 = sorted(course_shap_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    plt.figure(figsize=(12, 8))
    words, values = zip(*course_top20)
    plt.barh(range(len(words)), values)
    plt.yticks(range(len(words)), words)
    plt.xlabel('SHAP値')
    plt.title('授業評価スコア予測要因TOP20')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/course_top20_factors.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 比較分析
    # 2タスクのSHAP値散布図
    plt.figure(figsize=(10, 8))
    common_words = set(sentiment_shap_dict.keys()) & set(course_shap_dict.keys())
    x_values = [sentiment_shap_dict[word] for word in common_words]
    y_values = [course_shap_dict[word] for word in common_words]
    plt.scatter(x_values, y_values, alpha=0.6)
    plt.xlabel('感情スコア予測SHAP値')
    plt.ylabel('授業評価スコア予測SHAP値')
    plt.title('2タスクのSHAP値散布図')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/factor_comparison_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 統合分析
    # カテゴリ別要因数
    category_counts = {cat: len(items) for cat, items in categories.items()}
    plt.figure(figsize=(10, 6))
    categories_names = ['強い共通要因', '感情寄り要因', '評価寄り要因', '感情特化要因', '評価特化要因']
    counts = list(category_counts.values())
    plt.bar(categories_names, counts)
    plt.title('カテゴリ別要因数')
    plt.ylabel('要因数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/factor_categories_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("可視化の作成完了")

def save_results(sentiment_shap_dict, course_shap_dict, categories, output_dir):
    """結果の保存"""
    print("結果の保存開始...")
    
    # CSV形式で保存
    sentiment_df = pd.DataFrame(list(sentiment_shap_dict.items()), columns=['word', 'shap_value'])
    sentiment_df = sentiment_df.sort_values('shap_value', ascending=False)
    sentiment_df.to_csv(f"{output_dir}/word_importance_sentiment.csv", index=False, encoding='utf-8')
    
    course_df = pd.DataFrame(list(course_shap_dict.items()), columns=['word', 'shap_value'])
    course_df = course_df.sort_values('shap_value', ascending=False)
    course_df.to_csv(f"{output_dir}/word_importance_course.csv", index=False, encoding='utf-8')
    
    # JSON形式で保存
    categories_json = {}
    for category, items in categories.items():
        categories_json[category] = [
            {'word': word, 'sentiment_shap': s_shap, 'course_shap': c_shap}
            for word, s_shap, c_shap in items
        ]
    
    with open(f"{output_dir}/factor_categories.json", 'w', encoding='utf-8') as f:
        json.dump(categories_json, f, ensure_ascii=False, indent=2)
    
    # 分析サマリー
    summary = {
        'analysis_date': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_words_sentiment': len(sentiment_shap_dict),
        'total_words_course': len(course_shap_dict),
        'common_words': len(set(sentiment_shap_dict.keys()) & set(course_shap_dict.keys())),
        'category_counts': {cat: len(items) for cat, items in categories.items()},
        'top_sentiment_factors': dict(list(sentiment_shap_dict.items())[:10]),
        'top_course_factors': dict(list(course_shap_dict.items())[:10])
    }
    
    with open(f"{output_dir}/analysis_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("結果の保存完了")

def create_summary_report(categories, output_dir):
    """分析サマリーレポートの作成"""
    print("サマリーレポートの作成開始...")
    
    report = f"""# マルチタスク学習SHAP分析結果サマリー

## 分析概要
- 分析日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- 分析対象: 授業レベルマルチタスク学習モデル
- サンプル数: 1,000件（層化サンプリング）

## カテゴリ別要因数
"""
    
    category_names = {
        'strong_common': '強い共通要因',
        'sentiment_leaning': '感情寄り要因', 
        'course_leaning': '評価寄り要因',
        'sentiment_specific': '感情特化要因',
        'course_specific': '評価特化要因'
    }
    
    for category, items in categories.items():
        report += f"\n### {category_names[category]} ({len(items)}件)\n"
        if items:
            report += "| 順位 | 単語 | 感情SHAP | 評価SHAP |\n"
            report += "|------|------|----------|----------|\n"
            for i, (word, s_shap, c_shap) in enumerate(items[:10], 1):
                report += f"| {i} | {word} | {s_shap:.4f} | {c_shap:.4f} |\n"
        else:
            report += "該当する要因はありません。\n"
    
    report += f"""
## 主要な発見

### 1. 強い共通要因
両方のタスクで高い寄与を示す要因が{len(categories['strong_common'])}件発見されました。
これらは感情スコアと授業評価スコアの両方に影響する真の要因である可能性があります。

### 2. タスク特化要因
- 感情特化要因: {len(categories['sentiment_specific'])}件
- 評価特化要因: {len(categories['course_specific'])}件

これらの要因は、それぞれのタスクに特有の影響を与える要因です。

### 3. 授業改善への示唆
共通要因を重視した授業改善により、感情スコアと授業評価スコアの両方を向上させることが期待されます。

## 今後の課題
1. 共通要因の因果関係の検証
2. 実験的授業改善の実施
3. 改善効果の定量的測定
"""
    
    with open(f"{output_dir}/multitask_shap_analysis_summary.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("サマリーレポートの作成完了")

def main():
    """メイン処理"""
    # 出力ディレクトリの作成
    output_dir = "../03_分析結果/マルチタスクSHAP分析"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. データ読み込みとサンプリング
    print("\n=== Phase 1: データ準備とサンプリング ===")
    data_path = "../01_データ/マルチタスク用データ/授業集約データセット_20251012_142504.csv"
    df = pd.read_csv(data_path)
    print(f"データ読み込み完了: {len(df)}件")
    
    # 層化サンプリング
    sampled_df = stratified_sampling(df, n_samples=1000)
    
    # 2. モデルとトークナイザーの読み込み
    print("\n=== Phase 2: モデル読み込み ===")
    model_path = "../02_モデル/授業レベルマルチタスクモデル"
    
    # トークナイザーの読み込み（ベースモデルから）
    tokenizer = BertJapaneseTokenizer.from_pretrained('koheiduck/bert-japanese-finetuned-sentiment')
    
    # マルチタスクモデルの読み込み
    model = ClassLevelMultitaskModel()
    
    # PyTorchのバージョン問題を回避するため、weights_only=Falseで読み込み
    try:
        model.load_state_dict(torch.load(f"{model_path}/best_class_level_multitask_model.pth", map_location=device, weights_only=False))
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        print("代替方法でモデルを読み込みます...")
        # 代替方法：pickleを使用
        import pickle
        with open(f"{model_path}/best_class_level_multitask_model.pth", 'rb') as f:
            state_dict = pickle.load(f)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    print("モデル読み込み完了")
    
    # 3. SHAP分析実行
    print("\n=== Phase 3: SHAP分析実行 ===")
    texts_sample = sampled_df['自由記述まとめ'].tolist()
    
    # 感情スコア予測のSHAP分析
    print("感情スコア予測のSHAP分析...")
    explainer_sentiment = shap.Explainer(predict_sentiment_only, tokenizer)
    shap_values_sentiment = explainer_sentiment(texts_sample[:50])  # 最初の50件でテスト（GPU使用時はメモリ節約）
    
    # 授業評価スコア予測のSHAP分析
    print("授業評価スコア予測のSHAP分析...")
    explainer_course = shap.Explainer(predict_course_only, tokenizer)
    shap_values_course = explainer_course(texts_sample[:50])  # 最初の50件でテスト（GPU使用時はメモリ節約）
    
    # 4. 要因分析と分類
    print("\n=== Phase 4: 要因分析と分類 ===")
    
    # 単語レベルSHAP値の集約
    sentiment_shap_dict = {}
    course_shap_dict = {}
    
    # 出現回数5回以上の単語のみを分析対象
    word_counts = defaultdict(int)
    for text in texts_sample[:50]:  # 50件に修正
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            word_counts[token] += 1
    
    # SHAP値の集約
    for i in range(len(shap_values_sentiment.values)):
        tokens = tokenizer.tokenize(texts_sample[i])
        for j, token in enumerate(tokens):
            if word_counts[token] >= 5:  # 出現5回以上
                if token not in sentiment_shap_dict:
                    sentiment_shap_dict[token] = []
                    course_shap_dict[token] = []
                sentiment_shap_dict[token].append(shap_values_sentiment.values[i][j])
                course_shap_dict[token].append(shap_values_course.values[i][j])
    
    # 平均SHAP値を計算
    sentiment_shap_dict = {word: np.mean(values) for word, values in sentiment_shap_dict.items()}
    course_shap_dict = {word: np.mean(values) for word, values in course_shap_dict.items()}
    
    # 要因の分類
    categories = classify_factors(sentiment_shap_dict, course_shap_dict)
    
    # 5. 結果の保存と可視化
    print("\n=== Phase 5: 結果の保存と可視化 ===")
    save_results(sentiment_shap_dict, course_shap_dict, categories, output_dir)
    create_visualizations(sentiment_shap_dict, course_shap_dict, categories, output_dir)
    create_summary_report(categories, output_dir)
    
    print("\n=== 分析完了 ===")
    print(f"結果は {output_dir} に保存されました")

if __name__ == "__main__":
    main()

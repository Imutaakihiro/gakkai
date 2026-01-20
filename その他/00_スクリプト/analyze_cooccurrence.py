#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共起分析スクリプト
「良かった」などの重要語と一緒に出現する単語を分析
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import japanize_matplotlib

def load_data():
    """データを読み込み"""
    print("データを読み込み中...")
    import os
    
    # スクリプトの親ディレクトリ（卒業研究（新））に移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)
    
    df = pd.read_csv('01_データ/マルチタスク用データ/授業集約テキスト.csv')
    print(f"データ数: {len(df)}件")
    return df

def preprocess_text(text):
    """テキストの前処理"""
    if pd.isna(text):
        return ""
    
    # 基本的な前処理
    text = str(text)
    text = re.sub(r'[^\w\s]', ' ', text)  # 記号をスペースに
    text = re.sub(r'\s+', ' ', text)      # 連続するスペースを1つに
    text = text.strip()
    
    return text

def find_cooccurrences(df, target_words, window_size=5):
    """
    指定された単語の共起分析
    
    Args:
        df: データフレーム
        target_words: 分析対象の単語リスト
        window_size: 共起の窓サイズ（前後何語までを共起とみなすか）
    
    Returns:
        dict: 各対象単語の共起結果
    """
    print(f"共起分析を実行中... (窓サイズ: {window_size})")
    
    cooccurrence_results = {}
    
    for target_word in target_words:
        print(f"  - {target_word} の共起分析中...")
        
        # 対象単語を含む文を抽出
        target_sentences = []
        for text in df['aggregated_text']:
            processed_text = preprocess_text(text)
            if target_word in processed_text:
                target_sentences.append(processed_text)
        
        print(f"    {target_word}を含む文: {len(target_sentences)}件")
        
        # 共起単語をカウント
        cooccurrence_counter = Counter()
        
        for sentence in target_sentences:
            words = sentence.split()
            
            # 対象単語の位置を特定
            target_positions = [i for i, word in enumerate(words) if target_word in word]
            
            for pos in target_positions:
                # 窓サイズ内の単語を取得
                start = max(0, pos - window_size)
                end = min(len(words), pos + window_size + 1)
                
                for i in range(start, end):
                    if i != pos:  # 対象単語自体は除外
                        cooccurrence_counter[words[i]] += 1
        
        # 結果を保存
        cooccurrence_results[target_word] = {
            'cooccurrences': dict(cooccurrence_counter.most_common(50)),
            'total_sentences': len(target_sentences),
            'total_cooccurrences': sum(cooccurrence_counter.values())
        }
    
    return cooccurrence_results

def analyze_sentiment_cooccurrences(df, target_words):
    """
    感情ラベルと組み合わせた共起分析
    """
    print("感情ラベルとの組み合わせ分析を実行中...")
    
    # ラベルデータを読み込み
    try:
        label_df = pd.read_csv('01_データ/マルチタスク用データ/授業集約ラベル.csv')
        df_with_labels = df.merge(label_df, on='course_id', how='inner')
        
        # sentiment_meanから感情カテゴリを作成
        # sentiment_mean: 1=POSITIVE, 0=NEUTRAL, -1=NEGATIVE
        df_with_labels['sentiment'] = df_with_labels['sentiment_mean'].apply(
            lambda x: 'POSITIVE' if x > 0.3 else ('NEGATIVE' if x < -0.3 else 'NEUTRAL')
        )
        
        print(f"ラベル付きデータ: {len(df_with_labels)}件")
        print(f"  POSITIVE: {(df_with_labels['sentiment']=='POSITIVE').sum()}件")
        print(f"  NEGATIVE: {(df_with_labels['sentiment']=='NEGATIVE').sum()}件")
        print(f"  NEUTRAL: {(df_with_labels['sentiment']=='NEUTRAL').sum()}件")
    except Exception as e:
        print(f"ラベルデータの読み込みエラー: {e}")
        print("基本分析のみ実行します。")
        return {}
    
    sentiment_cooccurrences = {}
    
    for target_word in target_words:
        print(f"  - {target_word} の感情別共起分析中...")
        
        sentiment_results = {}
        
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            # 該当する感情のデータを抽出
            sentiment_data = df_with_labels[df_with_labels['sentiment'] == sentiment]
            
            # 対象単語を含む文を抽出
            target_sentences = []
            for text in sentiment_data['aggregated_text']:
                processed_text = preprocess_text(text)
                if target_word in processed_text:
                    target_sentences.append(processed_text)
            
            # 共起分析
            cooccurrence_counter = Counter()
            
            for sentence in target_sentences:
                words = sentence.split()
                target_positions = [i for i, word in enumerate(words) if target_word in word]
                
                for pos in target_positions:
                    start = max(0, pos - 5)
                    end = min(len(words), pos + 6)
                    
                    for i in range(start, end):
                        if i != pos:
                            cooccurrence_counter[words[i]] += 1
            
            sentiment_results[sentiment] = {
                'cooccurrences': dict(cooccurrence_counter.most_common(30)),
                'total_sentences': len(target_sentences)
            }
        
        sentiment_cooccurrences[target_word] = sentiment_results
    
    return sentiment_cooccurrences

def create_visualizations(cooccurrence_results, output_dir):
    """可視化を作成"""
    print("可視化を作成中...")
    
    for target_word, results in cooccurrence_results.items():
        if not results['cooccurrences']:
            continue
        
        # TOP20の共起単語を取得
        top_cooccurrences = list(results['cooccurrences'].items())[:20]
        words, counts = zip(*top_cooccurrences)
        
        # グラフを作成
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(words)), counts, color='skyblue', alpha=0.7)
        plt.yticks(range(len(words)), words)
        plt.xlabel('共起回数')
        plt.title(f'「{target_word}」の共起単語 TOP20\n(総文数: {results["total_sentences"]}件)')
        plt.gca().invert_yaxis()
        
        # 数値をバーの右側に表示
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cooccurrence_{target_word}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - {target_word} の可視化を保存")

def create_visualizations_by_sentiment(cooccurrence_results, positive_words, negative_words, output_dir):
    """ポジティブ/ネガティブ別に色分けして可視化"""
    print("感情別可視化を作成中...")
    
    for target_word, results in cooccurrence_results.items():
        if not results['cooccurrences']:
            continue
        
        # TOP20の共起単語を取得
        top_cooccurrences = list(results['cooccurrences'].items())[:20]
        if not top_cooccurrences:
            continue
            
        words, counts = zip(*top_cooccurrences)
        
        # ポジティブかネガティブかで色を変更
        if target_word in positive_words:
            color = '#4CAF50'  # 緑色（ポジティブ）
            sentiment_label = 'ポジティブ'
        elif target_word in negative_words:
            color = '#F44336'  # 赤色（ネガティブ）
            sentiment_label = 'ネガティブ'
        else:
            color = '#2196F3'  # 青色（中立）
            sentiment_label = ''
        
        # グラフを作成
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(words)), counts, color=color, alpha=0.7)
        plt.yticks(range(len(words)), words)
        plt.xlabel('共起回数', fontsize=12)
        
        title = f'「{target_word}」の共起単語 TOP20'
        if sentiment_label:
            title += f' [{sentiment_label}]'
        title += f'\n(総文数: {results["total_sentences"]}件)'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.gca().invert_yaxis()
        
        # 数値をバーの右側に表示
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    str(count), ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cooccurrence_{target_word}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - {target_word} [{sentiment_label}] の可視化を保存")

def save_results(cooccurrence_results, sentiment_cooccurrences, output_dir):
    """結果を保存"""
    print("結果を保存中...")
    
    # 基本共起結果を保存
    with open(f'{output_dir}/cooccurrence_results.json', 'w', encoding='utf-8') as f:
        json.dump(cooccurrence_results, f, ensure_ascii=False, indent=2)
    
    # 感情別共起結果を保存
    if sentiment_cooccurrences:
        with open(f'{output_dir}/sentiment_cooccurrences.json', 'w', encoding='utf-8') as f:
            json.dump(sentiment_cooccurrences, f, ensure_ascii=False, indent=2)
    
    # CSV形式でも保存
    all_results = []
    for target_word, results in cooccurrence_results.items():
        for co_word, count in results['cooccurrences'].items():
            all_results.append({
                'target_word': target_word,
                'cooccurrence_word': co_word,
                'count': count,
                'total_sentences': results['total_sentences']
            })
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f'{output_dir}/cooccurrence_analysis.csv', index=False, encoding='utf-8-sig')
    
    print(f"結果を {output_dir} に保存しました")

def generate_summary_report(cooccurrence_results, sentiment_cooccurrences, output_dir):
    """サマリーレポートを生成"""
    print("サマリーレポートを生成中...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = f"""# 共起分析レポート

**分析日時:** {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}  
**対象データ:** 授業集約テキスト  
**分析対象単語:** {', '.join(cooccurrence_results.keys())}

---

## 📊 分析概要

"""
    
    for target_word, results in cooccurrence_results.items():
        report += f"""### 「{target_word}」の共起分析結果

- **対象文数:** {results['total_sentences']}件
- **総共起回数:** {results['total_cooccurrences']}回
- **平均共起数:** {results['total_cooccurrences']/results['total_sentences']:.2f}回/文

#### TOP10共起単語

| 順位 | 単語 | 共起回数 | 出現率 |
|------|------|---------|--------|
"""
        
        for i, (word, count) in enumerate(list(results['cooccurrences'].items())[:10], 1):
            rate = count / results['total_sentences'] * 100
            report += f"| {i} | {word} | {count} | {rate:.1f}% |\n"
        
        report += "\n"
    
    # 感情別分析結果
    if sentiment_cooccurrences:
        report += "## 🎭 感情別共起分析\n\n"
        
        for target_word, sentiment_results in sentiment_cooccurrences.items():
            report += f"### 「{target_word}」の感情別共起\n\n"
            
            for sentiment, results in sentiment_results.items():
                if results['total_sentences'] > 0:
                    report += f"#### {sentiment} ({results['total_sentences']}件)\n\n"
                    report += "| 順位 | 単語 | 共起回数 |\n|------|------|---------|\n"
                    
                    for i, (word, count) in enumerate(list(results['cooccurrences'].items())[:10], 1):
                        report += f"| {i} | {word} | {count} |\n"
                    
                    report += "\n"
    
    report += f"""
---

## 📁 生成ファイル

- `cooccurrence_results.json` - 基本共起分析結果
- `cooccurrence_analysis.csv` - CSV形式の分析結果
- `sentiment_cooccurrences.json` - 感情別共起分析結果
- `cooccurrence_*.png` - 各単語の可視化グラフ

---

**分析完了！**  
結果ファイルは `{output_dir}` に保存されました。
"""
    
    with open(f'{output_dir}/cooccurrence_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("サマリーレポートを生成しました")

def generate_summary_report_enhanced(cooccurrence_results, sentiment_cooccurrences, 
                                    positive_words, negative_words, output_dir):
    """拡張版サマリーレポートを生成（ポジティブ/ネガティブ別）"""
    print("拡張版サマリーレポートを生成中...")
    
    report = f"""# 共起分析レポート（ポジティブ/ネガティブ比較）

**分析日時:** {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}  
**対象データ:** 授業集約テキスト（全83,851件）  
**分析手法:** 前後5語の窓を使った共起分析

---

## 🎯 分析の目的

SHAP分析で「良かった」「難しかった」などの単語が感情予測に寄与することがわかりました。
しかし、**「何が」良かったのか、「何が」難しかったのか**は不明でした。

この共起分析により、具体的な満足要因と不満要因を特定します。

---

## 📊 ポジティブ単語の共起分析

学生の満足要因を探る重要語との共起を分析しました。

"""
    
    # ポジティブ単語の分析
    for target_word in positive_words:
        if target_word not in cooccurrence_results:
            continue
        results = cooccurrence_results[target_word]
        
        if results['total_sentences'] == 0:
            continue
            
        report += f"""### 「{target_word}」の共起パターン

**対象文数:** {results['total_sentences']}件  
**平均共起数:** {results['total_cooccurrences']/results['total_sentences']:.2f}回/文

#### TOP10共起単語

| 順位 | 単語 | 共起回数 | 文書内出現率 |
|------|------|---------|------------|
"""
        
        for i, (word, count) in enumerate(list(results['cooccurrences'].items())[:10], 1):
            rate = count / results['total_sentences'] * 100
            report += f"| {i} | {word} | {count} | {rate:.1f}% |\n"
        
        report += "\n"
    
    # ネガティブ単語の分析
    report += """---

## 📉 ネガティブ単語の共起分析

学生の不満要因を探る重要語との共起を分析しました。

"""
    
    for target_word in negative_words:
        if target_word not in cooccurrence_results:
            continue
        results = cooccurrence_results[target_word]
        
        if results['total_sentences'] == 0:
            continue
            
        report += f"""### 「{target_word}」の共起パターン

**対象文数:** {results['total_sentences']}件  
**平均共起数:** {results['total_cooccurrences']/results['total_sentences']:.2f}回/文

#### TOP10共起単語

| 順位 | 単語 | 共起回数 | 文書内出現率 |
|------|------|---------|------------|
"""
        
        for i, (word, count) in enumerate(list(results['cooccurrences'].items())[:10], 1):
            rate = count / results['total_sentences'] * 100
            report += f"| {i} | {word} | {count} | {rate:.1f}% |\n"
        
        report += "\n"
    
    # 感情別分析の概要
    if sentiment_cooccurrences:
        report += """---

## 🎭 感情ラベル別の共起パターン

同じ単語でも、POSITIVE/NEGATIVEな文脈で使われ方が異なるかを検証しました。

"""
        
        # いくつかのキー単語について感情別に表示
        key_words = ['良かった', '難しかった', 'やす', 'ほしい']
        
        for target_word in key_words:
            if target_word not in sentiment_cooccurrences:
                continue
                
            sentiment_results = sentiment_cooccurrences[target_word]
            report += f"### 「{target_word}」の感情別使われ方\n\n"
            
            for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                if sentiment not in sentiment_results:
                    continue
                results = sentiment_results[sentiment]
                
                if results['total_sentences'] > 0:
                    report += f"#### {sentiment} ({results['total_sentences']}件)\n\n"
                    report += "| 順位 | 共起単語 | 回数 |\n|------|---------|------|\n"
                    
                    for i, (word, count) in enumerate(list(results['cooccurrences'].items())[:5], 1):
                        report += f"| {i} | {word} | {count} |\n"
                    
                    report += "\n"
    
    report += f"""---

## 💡 主要な発見

### ポジティブ要因
- 分析対象: {len(positive_words)}語
- 具体的な満足要素が明確化

### ネガティブ要因
- 分析対象: {len(negative_words)}語
- 具体的な改善点が特定可能

---

## 📁 生成ファイル

### データファイル
- `cooccurrence_results.json` - 基本共起分析結果（プログラムで再利用可）
- `cooccurrence_analysis.csv` - CSV形式の分析結果（Excel分析用）
- `sentiment_cooccurrences.json` - 感情別共起分析結果

### 可視化ファイル
"""
    
    # 生成されたグラフファイルをリスト化
    for word in positive_words:
        if word in cooccurrence_results and cooccurrence_results[word]['total_sentences'] > 0:
            report += f"- `cooccurrence_{word}.png` - 「{word}」の共起グラフ [ポジティブ]\n"
    
    for word in negative_words:
        if word in cooccurrence_results and cooccurrence_results[word]['total_sentences'] > 0:
            report += f"- `cooccurrence_{word}.png` - 「{word}」の共起グラフ [ネガティブ]\n"
    
    report += """
---

## 🚀 活用方法

### 教育改善への応用
1. **満足要因の強化**
   - ポジティブ単語の共起から、何が評価されているかを把握
   - 良い点をさらに伸ばす施策立案

2. **不満要因の解消**
   - ネガティブ単語の共起から、具体的な問題点を特定
   - ピンポイントでの改善施策実施

### 卒論での活用
- SHAP分析（単語の重要度）× 共起分析（単語の文脈）
- 「なぜその単語が重要か」を定量的・定性的に説明可能

---

**分析完了！**  
結果ファイルは `{output_dir}` に保存されました。
"""
    
    with open(f'{output_dir}/cooccurrence_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("拡張版サマリーレポートを生成しました")

def main():
    """メイン処理"""
    print("=== 共起分析スクリプト ===")
    
    # 出力ディレクトリを作成
    output_dir = "03_分析結果/共起分析"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # データを読み込み
    df = load_data()
    
    # 分析対象単語を定義
    print("\n【ポジティブ単語の共起分析】")
    positive_words = ['良かった', 'よかった', 'やす', '面白', '楽しい', 'おもしろ', 'できた', '分かり']
    
    print("\n【ネガティブ単語の共起分析】")
    negative_words = ['難しかった', 'ほしい', '苦手', 'ほう', '大変', '欲しい', 'ください', '不足']
    
    # すべての対象単語を結合
    all_target_words = positive_words + negative_words
    
    # 基本共起分析
    cooccurrence_results = find_cooccurrences(df, all_target_words, window_size=5)
    
    # 感情別共起分析
    sentiment_cooccurrences = analyze_sentiment_cooccurrences(df, all_target_words)
    
    # 可視化（ポジティブ/ネガティブで色分け）
    create_visualizations_by_sentiment(cooccurrence_results, positive_words, negative_words, output_dir)
    
    # 結果保存
    save_results(cooccurrence_results, sentiment_cooccurrences, output_dir)
    
    # サマリーレポート生成
    generate_summary_report_enhanced(cooccurrence_results, sentiment_cooccurrences, 
                                    positive_words, negative_words, output_dir)
    
    print("=== 分析完了 ===")

if __name__ == "__main__":
    main()

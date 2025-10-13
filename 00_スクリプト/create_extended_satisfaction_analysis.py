#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拡張満足度要因分析スクリプト
SHAP分析結果から詳細なランキングとカテゴリ分析を作成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from datetime import datetime
import os

def load_shap_data():
    """SHAP分析結果を読み込み"""
    print("SHAP分析結果を読み込み中...")
    
    # スクリプトの親ディレクトリ（卒業研究（新））に移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)
    
    # データ読み込み
    df = pd.read_csv('03_分析結果/SHAP分析/サンプリング5000件/word_importance_natural.csv')
    print(f"データ数: {len(df)}語")
    
    return df

def categorize_satisfaction_factors(df):
    """満足度要因をカテゴリ別に分類（ポジティブ・ネガティブ両方）"""
    print("満足度要因をカテゴリ別に分類中...")
    
    # ポジティブカテゴリ定義
    positive_categories = {
        'わかりやすさ': ['やす', '分かり', 'わかり', '理解', '説明', '丁寧', '詳しく', '明確'],
        '面白さ・興味': ['面白', 'おもしろ', '面白い', '興味', '楽しい', '楽し', '楽しめる', '新鮮', '飽き'],
        '学習効果': ['学', '学ぶ', '学び', '習得', '向上', '成長', '達成', '効果', '価値', '意義'],
        '実用性': ['実用', '役立つ', '役', '使える', '活用', '活かし', '生かし', '取り入れ'],
        '感謝・満足': ['良かった', 'よかった', 'ありが', 'ありがとう', '感謝', 'おかげ', '嬉', '好き'],
        '達成感': ['でき', '出来', 'できる', '出来る', '達成', '得', '得る', '取れ', '得点'],
        '人間関係': ['仲良く', 'つながっ', 'つながり', '繋がっ', '会', 'コミュニケーション'],
        '安心感': ['安心', '助', 'もらえる', 'くれる', '優しい', '気分'],
        '深い学び': ['深', '深める', '知る', '知り', '知れ', '分かった', 'わかった'],
        '機会・体験': ['機会', 'きっかけ', '体験', '触れる', '初', '過ご', '生き']
    }
    
    # ネガティブカテゴリ定義
    negative_categories = {
        '難しさ・複雑さ': ['難', '複雑', '難しい', '難しかった', '複雑', '大', '奥', '深い'],
        '不満・失望': ['欲しい', 'ほしい', 'ほし', 'まじ', '最低', 'もう', '程度'],
        '苦手・困難': ['苦手', '困難', '大変', '疲れ', '油', '不足'],
        '改善要求': ['直し', '直す', '改善', '修正', '変更', '下さい', 'ください'],
        '時間・期限': ['期限', '期間', '早め', '長く', '途中', '終わる'],
        '理解困難': ['分から', 'わから', '不明', '曖昧', '混乱', '迷'],
        '退屈・単調': ['退屈', '単調', 'つまら', '飽き', '繰り返し', '同じ'],
        '負担・圧迫': ['負担', '圧迫', '重い', '多い', '大変', 'しんど'],
        '不満足': ['普通', 'まあ', 'まず', '微妙', '微妙', 'イマイチ'],
        'その他ネガティブ': ['欠席', '怠', '真面目', '器具', 'シート', 'ノート']
    }
    
    # カテゴリを統合
    categories = {**positive_categories, **negative_categories}
    
    # カテゴリ分類
    categorized_data = []
    
    for _, row in df.iterrows():
        word = row['natural'].strip()
        mean_shap = row['mean_shap']
        count = row['count']
        
        # ポジティブ・ネガティブ判定
        is_positive = mean_shap > 0
        sentiment_type = 'ポジティブ' if is_positive else 'ネガティブ'
        
        # カテゴリを特定
        category = 'その他'
        for cat_name, keywords in categories.items():
            if any(keyword in word for keyword in keywords):
                category = cat_name
                break
        
        categorized_data.append({
            'word': word,
            'mean_shap': mean_shap,
            'abs_mean_shap': row['abs_mean_shap'],
            'count': count,
            'category': category,
            'sentiment_type': sentiment_type
        })
    
    df_categorized = pd.DataFrame(categorized_data)
    
    # カテゴリ別統計
    category_stats = df_categorized.groupby('category').agg({
        'mean_shap': ['mean', 'max', 'count'],
        'abs_mean_shap': 'mean',
        'count': 'sum'
    }).round(4)
    
    print(f"カテゴリ数: {len(category_stats)}")
    for category in category_stats.index:
        count = category_stats.loc[category, ('mean_shap', 'count')]
        print(f"  {category}: {count}語")
    
    return df_categorized, category_stats

def create_extended_rankings(df_categorized, output_dir):
    """拡張ランキングの作成（ポジティブ・ネガティブ両方）"""
    print(f"拡張ランキングを作成中... ({output_dir})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ポジティブ・ネガティブ分離
    df_positive = df_categorized[df_categorized['sentiment_type'] == 'ポジティブ']
    df_negative = df_categorized[df_categorized['sentiment_type'] == 'ネガティブ']
    
    # 1. ポジティブTOP50ランキング
    top50_positive = df_positive.nlargest(50, 'mean_shap')
    
    plt.figure(figsize=(14, 16))
    y_pos = range(len(top50_positive))
    plt.barh(y_pos, top50_positive['mean_shap'], color='lightgreen')
    plt.yticks(y_pos, [f"{row['word']} ({row['category']})" for _, row in top50_positive.iterrows()])
    plt.xlabel('SHAP値')
    plt.title('ポジティブ満足度要因 TOP50\n(カテゴリ別分類)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'positive_satisfaction_factors_top50.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ネガティブTOP50ランキング（絶対値でソート）
    top50_negative = df_negative.nlargest(50, 'abs_mean_shap')
    
    plt.figure(figsize=(14, 16))
    y_pos = range(len(top50_negative))
    plt.barh(y_pos, top50_negative['mean_shap'], color='lightcoral')
    plt.yticks(y_pos, [f"{row['word']} ({row['category']})" for _, row in top50_negative.iterrows()])
    plt.xlabel('SHAP値')
    plt.title('ネガティブ満足度要因 TOP50\n(カテゴリ別分類)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'negative_satisfaction_factors_top50.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 統合TOP100ランキング
    top100 = df_categorized.nlargest(100, 'abs_mean_shap')
    
    plt.figure(figsize=(16, 20))
    y_pos = range(len(top100))
    colors = ['lightgreen' if row['sentiment_type'] == 'ポジティブ' else 'lightcoral' 
              for _, row in top100.iterrows()]
    plt.barh(y_pos, top100['mean_shap'], color=colors)
    plt.yticks(y_pos, [f"{row['word']} ({row['category']})" for _, row in top100.iterrows()])
    plt.xlabel('SHAP値')
    plt.title('満足度要因 TOP100\n(ポジティブ・ネガティブ統合)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'satisfaction_factors_top100_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. カテゴリ別ランキング
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    categories = df_categorized['category'].unique()
    
    for i, category in enumerate(categories[:10]):  # トップ10カテゴリ
        cat_data = df_categorized[df_categorized['category'] == category].nlargest(10, 'mean_shap')
        
        if len(cat_data) > 0:
            y_pos = range(len(cat_data))
            axes[i].barh(y_pos, cat_data['mean_shap'], color='skyblue')
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(cat_data['word'])
            axes[i].set_xlabel('SHAP値')
            axes[i].set_title(f'{category}\n({len(cat_data)}語)')
            axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'satisfaction_factors_by_category.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. カテゴリ別統計
    plt.figure(figsize=(12, 8))
    category_means = df_categorized.groupby('category')['mean_shap'].mean().sort_values(ascending=True)
    
    plt.barh(range(len(category_means)), category_means.values, color='lightcoral')
    plt.yticks(range(len(category_means)), category_means.index)
    plt.xlabel('平均SHAP値')
    plt.title('カテゴリ別平均満足度')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'satisfaction_by_category.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return top50_positive, top50_negative, top100

def create_detailed_report(df_categorized, top50_positive, top50_negative, top100, category_stats, output_dir):
    """詳細レポートの作成（ポジティブ・ネガティブ両方）"""
    print(f"詳細レポートを作成中... ({output_dir})")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'extended_satisfaction_analysis_{timestamp}.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 拡張満足度要因分析レポート（ポジティブ・ネガティブ統合）\n\n")
        f.write(f"**分析日時:** {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        
        # データ概要
        f.write("## 📊 データ概要\n\n")
        f.write(f"- **分析対象語数:** {len(df_categorized)}語\n")
        f.write(f"- **ポジティブ語数:** {len(df_categorized[df_categorized['sentiment_type'] == 'ポジティブ'])}語\n")
        f.write(f"- **ネガティブ語数:** {len(df_categorized[df_categorized['sentiment_type'] == 'ネガティブ'])}語\n")
        f.write(f"- **カテゴリ数:** {len(category_stats)}カテゴリ\n")
        f.write(f"- **データソース:** SHAP分析結果（5,000件サンプリング）\n\n")
        
        # ポジティブTOP50ランキング
        f.write("## 🏆 ポジティブ満足度要因 TOP50\n\n")
        f.write("| 順位 | 重要語 | カテゴリ | SHAP値 | 出現回数 |\n")
        f.write("|------|--------|----------|--------|----------|\n")
        
        for i, (_, row) in enumerate(top50_positive.iterrows(), 1):
            f.write(f"| {i} | {row['word']} | {row['category']} | {row['mean_shap']:.4f} | {row['count']} |\n")
        
        f.write("\n")
        
        # ネガティブTOP50ランキング
        f.write("## ⚠️ ネガティブ満足度要因 TOP50\n\n")
        f.write("| 順位 | 重要語 | カテゴリ | SHAP値 | 出現回数 |\n")
        f.write("|------|--------|----------|--------|----------|\n")
        
        for i, (_, row) in enumerate(top50_negative.iterrows(), 1):
            f.write(f"| {i} | {row['word']} | {row['category']} | {row['mean_shap']:.4f} | {row['count']} |\n")
        
        f.write("\n")
        
        # 統合TOP100ランキング
        f.write("## 🥇 満足度要因 TOP100（統合）\n\n")
        f.write("| 順位 | 重要語 | カテゴリ | 感情 | SHAP値 | 出現回数 |\n")
        f.write("|------|--------|----------|------|--------|----------|\n")
        
        for i, (_, row) in enumerate(top100.iterrows(), 1):
            sentiment_emoji = "😊" if row['sentiment_type'] == 'ポジティブ' else "😞"
            f.write(f"| {i} | {row['word']} | {row['category']} | {sentiment_emoji} | {row['mean_shap']:.4f} | {row['count']} |\n")
        
        f.write("\n")
        
        # カテゴリ別分析
        f.write("## 📈 カテゴリ別分析\n\n")
        f.write("| カテゴリ | 平均SHAP値 | 最大SHAP値 | 語数 | 総出現回数 |\n")
        f.write("|----------|------------|------------|------|------------|\n")
        
        for category in category_stats.index:
            mean_shap = category_stats.loc[category, ('mean_shap', 'mean')]
            max_shap = category_stats.loc[category, ('mean_shap', 'max')]
            word_count = category_stats.loc[category, ('mean_shap', 'count')]
            total_count = category_stats.loc[category, ('count', 'sum')]
            
            f.write(f"| {category} | {mean_shap:.4f} | {max_shap:.4f} | {word_count} | {total_count} |\n")
        
        f.write("\n")
        
        # カテゴリ別詳細
        f.write("## 🔍 カテゴリ別詳細分析\n\n")
        
        for category in category_stats.index:
            cat_data = df_categorized[df_categorized['category'] == category].nlargest(10, 'mean_shap')
            
            if len(cat_data) > 0:
                f.write(f"### {category}\n\n")
                f.write("| 順位 | 重要語 | SHAP値 | 出現回数 |\n")
                f.write("|------|--------|--------|----------|\n")
                
                for i, (_, row) in enumerate(cat_data.iterrows(), 1):
                    f.write(f"| {i} | {row['word']} | {row['mean_shap']:.4f} | {row['count']} |\n")
                
                f.write("\n")
        
        # 統計情報
        f.write("## 📊 統計情報\n\n")
        f.write(f"- **最高SHAP値:** {df_categorized['mean_shap'].max():.4f}\n")
        f.write(f"- **平均SHAP値:** {df_categorized['mean_shap'].mean():.4f}\n")
        f.write(f"- **標準偏差:** {df_categorized['mean_shap'].std():.4f}\n")
        f.write(f"- **総出現回数:** {df_categorized['count'].sum():,}\n")
        f.write(f"- **ユニーク語数:** {len(df_categorized)}\n\n")
        
        # カテゴリ別統計
        f.write("## 🎯 カテゴリ別統計\n\n")
        for category in category_stats.index:
            mean_shap = category_stats.loc[category, ('mean_shap', 'mean')]
            word_count = category_stats.loc[category, ('mean_shap', 'count')]
            f.write(f"- **{category}:** 平均SHAP値 {mean_shap:.4f}, {word_count}語\n")
        
        f.write("\n")
        
        # 改善指針
        f.write("## 💡 授業改善指針（カテゴリ別）\n\n")
        
        improvement_guidance = {
            'わかりやすさ': '専門用語の丁寧な説明、段階的な説明、視覚的資料の活用',
            '面白さ・興味': '実例・事例の紹介、最新の話題・ニュースの活用、インタラクティブな授業',
            '学習効果': '学習目標の明確化、段階的な達成感、フィードバックの充実',
            '実用性': '実践的な演習、実際の場面での活用例、応用課題の提供',
            '感謝・満足': '学生の意見を尊重、フィードバックの充実、個別指導の機会',
            '達成感': '適切な難易度設定、段階的な目標、成果の可視化',
            '人間関係': 'グループワーク、ディスカッション、学生間の交流促進',
            '安心感': '質問しやすい環境、個別指導、サポート体制の充実',
            '深い学び': '探究的な課題、関連分野との連携、多角的な視点',
            '機会・体験': '実体験の機会、外部講師、フィールドワーク'
        }
        
        for category, guidance in improvement_guidance.items():
            if category in category_stats.index:
                f.write(f"### {category}\n")
                f.write(f"{guidance}\n\n")
    
    print(f"詳細レポートを保存: {report_path}")

def save_extended_data(df_categorized, top50_positive, top50_negative, top100, category_stats, output_dir):
    """拡張データの保存（ポジティブ・ネガティブ両方）"""
    print(f"拡張データを保存中... ({output_dir})")
    
    # ポジティブTOP50をCSVで保存
    top50_positive.to_csv(os.path.join(output_dir, 'positive_satisfaction_factors_top50.csv'), 
                         index=False, encoding='utf-8-sig')
    
    # ネガティブTOP50をCSVで保存
    top50_negative.to_csv(os.path.join(output_dir, 'negative_satisfaction_factors_top50.csv'), 
                         index=False, encoding='utf-8-sig')
    
    # 統合TOP100をCSVで保存
    top100.to_csv(os.path.join(output_dir, 'satisfaction_factors_top100_combined.csv'), 
                  index=False, encoding='utf-8-sig')
    
    # カテゴリ別データをCSVで保存
    df_categorized.to_csv(os.path.join(output_dir, 'satisfaction_factors_categorized.csv'), 
                         index=False, encoding='utf-8-sig')
    
    # カテゴリ別統計をCSVで保存
    category_stats.to_csv(os.path.join(output_dir, 'category_statistics.csv'), 
                         encoding='utf-8-sig')
    
    print("拡張データの保存完了")

def main():
    """メイン処理"""
    print("=" * 60)
    print("拡張満足度要因分析")
    print("=" * 60)
    
    # データ読み込み
    df = load_shap_data()
    
    # カテゴリ分類
    df_categorized, category_stats = categorize_satisfaction_factors(df)
    
    # 出力ディレクトリ
    output_dir = '03_分析結果/拡張満足度要因分析'
    os.makedirs(output_dir, exist_ok=True)
    
    # 拡張ランキング作成
    top50_positive, top50_negative, top100 = create_extended_rankings(df_categorized, output_dir)
    
    # 詳細レポート作成
    create_detailed_report(df_categorized, top50_positive, top50_negative, top100, category_stats, output_dir)
    
    # データ保存
    save_extended_data(df_categorized, top50_positive, top50_negative, top100, category_stats, output_dir)
    
    print("\n" + "=" * 60)
    print("拡張分析完了！")
    print("=" * 60)
    print(f"結果は {output_dir} に保存されました")
    print(f"ポジティブTOP50: {len(top50_positive)}語")
    print(f"ネガティブTOP50: {len(top50_negative)}語")
    print(f"統合TOP100: {len(top100)}語")
    print(f"カテゴリ数: {len(category_stats)}")

if __name__ == "__main__":
    main()

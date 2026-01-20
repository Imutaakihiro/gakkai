#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共通要因の具体的活用方法の提案
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'MS Mincho', 'DejaVu Sans']
else:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao']

plt.rcParams['axes.unicode_minus'] = False

def analyze_common_factors():
    """共通要因の詳細分析"""
    print("🔍 共通要因の詳細分析中...")
    
    # 共通要因データの読み込み
    common_df = pd.read_csv('00_スクリプト/03_分析結果/マルチタスクSHAP分析_新閾値0.0005/データ/新閾値_共通要因_詳細.csv')
    
    # TOP20の共通要因を抽出
    top_common = common_df.head(20)
    
    print(f"✅ 共通要因総数: {len(common_df)}語彙")
    print(f"✅ TOP20共通要因を分析中...")
    
    return top_common

def categorize_common_factors(top_common):
    """共通要因をカテゴリ別に分類"""
    print("\n📊 共通要因のカテゴリ分類中...")
    
    # 教育関連のカテゴリ定義
    categories = {
        '学習効果': ['学ぶ', '理解', '総括', '推奨', '含め', '中心習', '両立'],
        '技術・方法': ['電動', '方式', '複素', '書きべ', 'ペン', '機械', '救急'],
        '環境・条件': ['人数', '台湾', 'II', '我々', '結婚'],
        '感情・態度': ['まま', 'すぐ', 'より', '下さい', 'あんまり'],
        'その他': ['myit', '単語語', 'れる', 'リスト', '異なる']
    }
    
    # 各単語をカテゴリに分類
    categorized = {}
    for category, keywords in categories.items():
        categorized[category] = []
        for _, row in top_common.iterrows():
            word = row['word']
            if any(keyword in word for keyword in keywords):
                categorized[category].append({
                    'word': word,
                    'sentiment_importance': row['sentiment_importance'],
                    'course_importance': row['course_importance'],
                    'total_importance': row['total_importance'],
                    'rank': row['rank']
                })
    
    # 未分類の単語を「その他」に追加
    all_categorized = []
    for category_words in categorized.values():
        all_categorized.extend([w['word'] for w in category_words])
    
    for _, row in top_common.iterrows():
        if row['word'] not in all_categorized:
            categorized['その他'].append({
                'word': row['word'],
                'sentiment_importance': row['sentiment_importance'],
                'course_importance': row['course_importance'],
                'total_importance': row['total_importance'],
                'rank': row['rank']
            })
    
    return categorized

def create_implementation_strategies(categorized):
    """具体的な活用戦略の作成"""
    print("\n💡 具体的な活用戦略の作成中...")
    
    strategies = {
        '学習効果': {
            '要因': ['学ぶ', '理解', '総括', '推奨', '含め', '中心習', '両立'],
            '戦略': [
                'アクティブラーニングの導入',
                '理解度チェックの頻繁な実施',
                '授業の総括・振り返り時間の確保',
                '推奨教材・参考書の提示',
                '関連知識の包含的説明',
                '中心となる学習目標の明確化',
                '理論と実践の両立'
            ],
            '投資効果': '高（満足度と評価の両方に直接影響）'
        },
        '技術・方法': {
            '要因': ['電動', '方式', '複素', '書きべ', 'ペン', '機械', '救急'],
            '戦略': [
                'デジタルツールの活用（電動機器）',
                '多様な教授方法の採用',
                '複雑な概念の段階的説明',
                '手書きとデジタルの併用',
                '機械学習・AI技術の導入',
                '緊急時の対応方法の準備'
            ],
            '投資効果': '中（技術的改善による間接的効果）'
        },
        '環境・条件': {
            '要因': ['人数', '台湾', 'II', '我々', '結婚'],
            '戦略': [
                'クラスサイズの最適化',
                '国際的な視点の導入',
                '段階的な学習システム',
                '協働学習の促進',
                'ライフイベントへの配慮'
            ],
            '投資効果': '中（環境整備による長期的効果）'
        },
        '感情・態度': {
            '要因': ['まま', 'すぐ', 'より', '下さい', 'あんまり'],
            '戦略': [
                '自然な学習環境の提供',
                '即座のフィードバック',
                'より良い学習体験の追求',
                '丁寧な対応・説明',
                '過度な負荷の回避'
            ],
            '投資効果': '高（感情面での満足度向上）'
        }
    }
    
    return strategies

def create_implementation_roadmap(strategies):
    """実装ロードマップの作成"""
    print("\n🗺️ 実装ロードマップの作成中...")
    
    roadmap = {
        '短期（1-3ヶ月）': {
            '優先度': '高',
            '施策': [
                '理解度チェックの頻繁な実施',
                '授業の総括・振り返り時間の確保',
                '即座のフィードバックの提供',
                '丁寧な対応・説明の徹底'
            ],
            '投資額': '低',
            '効果': '即効性あり'
        },
        '中期（3-6ヶ月）': {
            '優先度': '中',
            '施策': [
                'アクティブラーニングの導入',
                '推奨教材・参考書の提示',
                '多様な教授方法の採用',
                'クラスサイズの最適化検討'
            ],
            '投資額': '中',
            '効果': '段階的改善'
        },
        '長期（6-12ヶ月）': {
            '優先度': '中',
            '施策': [
                'デジタルツールの本格導入',
                '機械学習・AI技術の活用',
                '国際的な視点の導入',
                '協働学習システムの構築'
            ],
            '投資額': '高',
            '効果': '持続的改善'
        }
    }
    
    return roadmap

def create_visualization(categorized, strategies, roadmap):
    """可視化の作成"""
    print("\n🎨 可視化の作成中...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('共通要因の具体的活用戦略', fontsize=16, fontweight='bold')
    
    # 1. カテゴリ別重要度
    categories = list(categorized.keys())
    sentiment_importance = [sum([w['sentiment_importance'] for w in categorized[cat]]) for cat in categories]
    course_importance = [sum([w['course_importance'] for w in categorized[cat]]) for cat in categories]
    
    x = range(len(categories))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], sentiment_importance, width, label='感情重要度', color='#FF6B6B', alpha=0.8)
    ax1.bar([i + width/2 for i in x], course_importance, width, label='評価重要度', color='#4ECDC4', alpha=0.8)
    ax1.set_xlabel('カテゴリ', fontsize=12)
    ax1.set_ylabel('重要度', fontsize=12)
    ax1.set_title('カテゴリ別重要度', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 投資効果の比較
    effects = ['高', '中', '中', '高', '中']  # 5つのカテゴリに対応
    colors = ['#FF6B6B', '#FFA07A', '#FFA07A', '#FF6B6B', '#FFA07A']  # 5つの色に対応
    
    bars = ax2.bar(categories, [1, 1, 1, 1, 1], color=colors, alpha=0.8)  # 5つの値に対応
    ax2.set_ylabel('投資効果', fontsize=12)
    ax2.set_title('カテゴリ別投資効果', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(categories, rotation=45)
    
    # 効果レベルをテキストで表示
    for i, (bar, effect) in enumerate(zip(bars, effects)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                effect, ha='center', va='center', fontweight='bold', fontsize=12)
    
    # 3. 実装期間別施策数
    periods = list(roadmap.keys())
    strategy_counts = [len(roadmap[period]['施策']) for period in periods]
    colors_period = ['#FF6B6B', '#FFA07A', '#87CEEB']
    
    bars = ax3.bar(periods, strategy_counts, color=colors_period, alpha=0.8)
    ax3.set_ylabel('施策数', fontsize=12)
    ax3.set_title('実装期間別施策数', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(periods, rotation=45)
    
    # 施策数をテキストで表示
    for bar, count in zip(bars, strategy_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 4. 投資額と効果の関係
    investment_levels = ['低', '中', '高']
    effectiveness = ['即効性あり', '段階的改善', '持続的改善']
    colors_invest = ['#90EE90', '#FFA07A', '#FF6B6B']
    
    ax4.scatter([1, 2, 3], [1, 2, 3], s=[200, 300, 400], c=colors_invest, alpha=0.7)
    ax4.set_xlabel('投資額', fontsize=12)
    ax4.set_ylabel('効果', fontsize=12)
    ax4.set_title('投資額と効果の関係', fontsize=14, fontweight='bold')
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(investment_levels)
    ax4.set_yticks([1, 2, 3])
    ax4.set_yticklabels(effectiveness)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('00_スクリプト/03_分析結果/マルチタスクSHAP分析_新閾値0.0005/共通要因活用戦略.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 共通要因活用戦略可視化保存完了")

def create_implementation_guide(categorized, strategies, roadmap):
    """実装ガイドの作成"""
    print("\n📝 実装ガイドの作成中...")
    
    guide = f"""# 共通要因の具体的活用ガイド

## 🎯 概要
- 作成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- 目的: マルチタスク学習で特定された共通要因の実践的活用
- 対象: 教育機関、教員、教育管理者

## 📊 共通要因の分析結果

### TOP20共通要因
| 順位 | 単語 | 感情重要度 | 評価重要度 | 統合重要度 | カテゴリ |
|------|------|------------|------------|------------|----------|
"""
    
    # TOP20の表を作成
    for i, (_, row) in enumerate(categorized.items(), 1):
        for j, word_data in enumerate(row):
            if j < 5:  # 各カテゴリから上位5つまで
                guide += f"| {word_data['rank']} | {word_data['word']} | {word_data['sentiment_importance']:.6f} | {word_data['course_importance']:.6f} | {word_data['total_importance']:.6f} | {list(categorized.keys())[i-1]} |\n"
    
    guide += f"""
## 💡 カテゴリ別活用戦略

"""
    
    # 各カテゴリの戦略を記載
    for category, data in strategies.items():
        guide += f"""### {category}
**要因:** {', '.join(data['要因'])}
**投資効果:** {data['投資効果']}

**具体的施策:**
"""
        for strategy in data['戦略']:
            guide += f"- {strategy}\n"
        guide += "\n"
    
    guide += f"""
## 🗺️ 実装ロードマップ

"""
    
    # ロードマップを記載
    for period, data in roadmap.items():
        guide += f"""### {period}
**優先度:** {data['優先度']}
**投資額:** {data['投資額']}
**効果:** {data['効果']}

**施策:**
"""
        for strategy in data['施策']:
            guide += f"- {strategy}\n"
        guide += "\n"
    
    guide += f"""
## 🎯 実装の優先順位

### 1. 最優先（即座に実施）
- **理解度チェックの頻繁な実施**
- **授業の総括・振り返り時間の確保**
- **即座のフィードバックの提供**

### 2. 高優先（1-3ヶ月以内）
- **アクティブラーニングの導入**
- **推奨教材・参考書の提示**
- **丁寧な対応・説明の徹底**

### 3. 中優先（3-6ヶ月以内）
- **多様な教授方法の採用**
- **クラスサイズの最適化検討**
- **デジタルツールの活用**

## 📈 期待される効果

### 短期効果（1-3ヶ月）
- **満足度向上**: 感情スコアの改善
- **評価向上**: 授業評価スコアの改善
- **学習効果**: 理解度の向上

### 中期効果（3-6ヶ月）
- **継続的改善**: 持続的な満足度向上
- **学習環境**: より良い学習環境の提供
- **教員満足度**: 教員の満足度向上

### 長期効果（6-12ヶ月）
- **教育品質**: 全体的な教育品質の向上
- **学生成果**: 学生の学習成果の向上
- **機関評価**: 教育機関の評価向上

## 🎤 学会発表での活用

### 核心メッセージ
**「マルチタスク学習により特定された共通要因は、教育改善の具体的な指針を提供します」**

### 具体的価値
1. **データ駆動型教育改善**: 科学的根拠に基づく改善
2. **投資効果の最大化**: 限られたリソースの最適配分
3. **持続的改善**: 長期的な教育品質向上

### 実践的示唆
- **共通要因への投資**が最も効果的
- **段階的実装**によるリスク最小化
- **継続的評価**による改善の最適化

---
*このガイドは、マルチタスク学習の分析結果に基づいて作成された実践的な教育改善指針です。*
"""
    
    # ガイド保存
    with open('00_スクリプト/03_分析結果/マルチタスクSHAP分析_新閾値0.0005/共通要因活用ガイド.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("✅ 共通要因活用ガイド保存完了")

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("共通要因の具体的活用方法の提案")
    print("=" * 60)
    
    # 共通要因の分析
    top_common = analyze_common_factors()
    
    # カテゴリ分類
    categorized = categorize_common_factors(top_common)
    
    # 活用戦略の作成
    strategies = create_implementation_strategies(categorized)
    
    # 実装ロードマップの作成
    roadmap = create_implementation_roadmap(strategies)
    
    # 可視化の作成
    create_visualization(categorized, strategies, roadmap)
    
    # 実装ガイドの作成
    create_implementation_guide(categorized, strategies, roadmap)
    
    print("\n🎉 共通要因の具体的活用方法の提案完了！")
    print("📁 結果は 00_スクリプト/03_分析結果/マルチタスクSHAP分析_新閾値0.0005 に保存されました")

if __name__ == "__main__":
    main()

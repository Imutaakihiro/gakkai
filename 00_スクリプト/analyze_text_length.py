#!/usr/bin/env python3
"""
授業ごとの自由記述の文字数統計を確認するスクリプト
"""

import pandas as pd
import numpy as np

def analyze_text_length():
    """自由記述の文字数統計を分析"""
    
    # データセット読み込み
    df = pd.read_csv('01_データ/マルチタスク用データ/授業集約データセット_20251012_142504.csv')
    
    print("📊 授業ごとの自由記述文字数統計")
    print("=" * 50)
    
    # 文字数計算
    text_lengths = df["自由記述まとめ"].str.len()
    
    print(f"📈 基本統計:")
    print(f"  平均文字数: {text_lengths.mean():.0f}文字")
    print(f"  中央値: {text_lengths.median():.0f}文字")
    print(f"  最小: {text_lengths.min()}文字")
    print(f"  最大: {text_lengths.max()}文字")
    print(f"  標準偏差: {text_lengths.std():.0f}文字")
    
    print(f"\n📊 詳細分布:")
    print(text_lengths.describe())
    
    print(f"\n🎯 現在のMAX_LENGTH設定との比較:")
    print(f"  MAX_LENGTH = 128トークン")
    print(f"  平均文字数 = {text_lengths.mean():.0f}文字")
    print(f"  文字数/トークン比 ≈ 1.5-2.0 (日本語)")
    print(f"  推定トークン数 ≈ {text_lengths.mean() * 1.75:.0f}トークン")
    
    # 128トークンでカバーできる割合
    estimated_tokens = text_lengths * 1.75
    coverage_128 = (estimated_tokens <= 128).mean() * 100
    coverage_256 = (estimated_tokens <= 256).mean() * 100
    coverage_512 = (estimated_tokens <= 512).mean() * 100
    
    print(f"\n📊 トークン長でのカバー率:")
    print(f"  128トークン: {coverage_128:.1f}%の授業をカバー")
    print(f"  256トークン: {coverage_256:.1f}%の授業をカバー")
    print(f"  512トークン: {coverage_512:.1f}%の授業をカバー")
    
    # サンプル表示
    print(f"\n📝 サンプル授業（文字数順）:")
    sample_df = df.copy()
    sample_df['文字数'] = sample_df["自由記述まとめ"].str.len()
    sample_df = sample_df.sort_values('文字数', ascending=False)
    
    for i, (_, row) in enumerate(sample_df.head(3).iterrows()):
        print(f"\n{i+1}. {row['授業ID']}")
        print(f"   文字数: {row['文字数']}文字")
        print(f"   自由記述数: {row['自由記述数']}件")
        print(f"   推定トークン数: {row['文字数'] * 1.75:.0f}トークン")
        print(f"   内容（最初の100文字）: {row['自由記述まとめ'][:100]}...")

if __name__ == "__main__":
    analyze_text_length()

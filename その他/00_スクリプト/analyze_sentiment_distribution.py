#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
感情スコアの分布分析
授業ごとの感情スコアの特性を分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_sentiment_distribution():
    """感情スコアの分布分析"""
    try:
        # データ読み込み
        df = pd.read_csv('01_データ/マルチタスク用データ/マルチタスク学習用データセット_20250930_202839.csv')
        
        print("=== 感情スコア分布分析 ===")
        print(f"総データ数: {len(df)}")
        
        # 感情スコアの基本統計
        sentiment_scores = df['感情スコア']
        print(f"\n感情スコア基本統計:")
        print(sentiment_scores.describe())
        
        # 感情スコアの分布
        print(f"\n感情スコア分布:")
        sentiment_counts = Counter(sentiment_scores)
        for score, count in sorted(sentiment_counts.items()):
            percentage = (count / len(df)) * 100
            print(f"  {score}: {count}件 ({percentage:.1f}%)")
        
        # 授業ごとの感情スコア分析（サンプル）
        # 同じ授業評価スコアのデータをグループ化
        course_groups = df.groupby('授業評価スコア')
        
        print(f"\n授業評価スコア別の感情スコア分析:")
        for score, group in course_groups:
            if len(group) >= 5:  # 5件以上のデータがあるグループのみ
                sentiment_mean = group['感情スコア'].mean()
                sentiment_std = group['感情スコア'].std()
                sentiment_dist = Counter(group['感情スコア'])
                
                print(f"\n授業評価スコア {score}:")
                print(f"  データ数: {len(group)}")
                print(f"  感情スコア平均: {sentiment_mean:.2f}")
                print(f"  感情スコア標準偏差: {sentiment_std:.2f}")
                print(f"  感情スコア分布: {dict(sentiment_dist)}")
        
        # 相関分析
        correlation = df['感情スコア'].corr(df['授業評価スコア'])
        print(f"\n感情スコアと授業評価スコアの相関係数: {correlation:.3f}")
        
        return df
        
    except Exception as e:
        print(f"エラー: {e}")
        return None

if __name__ == "__main__":
    analyze_sentiment_distribution()

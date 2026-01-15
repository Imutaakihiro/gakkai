#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
99%の一致度を検証するスクリプト
"""

import pandas as pd
import numpy as np

def verify_99_percent():
    """99%の一致度を検証"""
    
    # データの読み込み
    sentiment_df = pd.read_csv('00_スクリプト/03_分析結果/マルチタスクSHAP分析_BERTトークナイザー_全データ/感情スコア重要度_詳細_全データ.csv')
    course_df = pd.read_csv('00_スクリプト/03_分析結果/マルチタスクSHAP分析_BERTトークナイザー_全データ/授業評価スコア重要度_詳細_全データ.csv')
    
    print('=== データの基本情報 ===')
    print(f'感情スコア語彙数: {len(sentiment_df)}')
    print(f'授業評価スコア語彙数: {len(course_df)}')
    
    # 閾値設定
    threshold = 0.0001
    
    # 閾値以上の重要度を持つ語彙を抽出
    sentiment_high = sentiment_df[sentiment_df['importance'] >= threshold]['word'].tolist()
    course_high = course_df[course_df['importance'] >= threshold]['word'].tolist()
    
    print(f'\n=== 閾値 {threshold} 以上の語彙数 ===')
    print(f'感情スコア: {len(sentiment_high)}語彙')
    print(f'授業評価スコア: {len(course_high)}語彙')
    
    # 共通要因の計算
    common_words = set(sentiment_high) & set(course_high)
    sentiment_only = set(sentiment_high) - set(course_high)
    course_only = set(course_high) - set(sentiment_high)
    
    print(f'\n=== 分類結果 ===')
    print(f'共通要因: {len(common_words)}語彙')
    print(f'感情特化: {len(sentiment_only)}語彙')
    print(f'評価特化: {len(course_only)}語彙')
    
    # 割合の計算
    total_words = len(set(sentiment_high) | set(course_high))
    common_ratio = len(common_words) / total_words * 100
    
    print(f'\n=== 割合 ===')
    print(f'総語彙数: {total_words}')
    print(f'共通要因の割合: {common_ratio:.2f}%')
    print(f'感情特化の割合: {len(sentiment_only)/total_words*100:.2f}%')
    print(f'評価特化の割合: {len(course_only)/total_words*100:.2f}%')
    
    # 詳細な検証
    print(f'\n=== 詳細検証 ===')
    print(f'感情スコアのみの語彙: {list(sentiment_only)[:10]}...')
    print(f'授業評価のみの語彙: {list(course_only)[:10]}...')
    print(f'共通要因の例: {list(common_words)[:10]}...')
    
    return {
        'total_words': total_words,
        'common_words': len(common_words),
        'common_ratio': common_ratio,
        'sentiment_only': len(sentiment_only),
        'course_only': len(course_only)
    }

if __name__ == "__main__":
    result = verify_99_percent()

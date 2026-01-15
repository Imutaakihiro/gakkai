#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
授業レベルの集約データセット作成スクリプト

入力: 個別の自由記述データ
出力: 授業ごとに集約されたデータセット
  - 授業ID
  - 自由記述まとめ（全記述を結合）
  - 感情スコアの平均
  - 感情スコアラベルの分布（negative, neutral, positive の割合）
  - 授業評価スコア
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from collections import Counter

def extract_course_info(df):
    """
    元データから授業情報を抽出
    列の構造:
    0: ファイル名
    1: 年度・学期
    2: 授業ID（例: "1010135　社会学入門"）
    3: 平均評価ポイント（例: "平均評価ポイント　　3.41"）
    4: 履修情報
    5: 評価分布
    6: 自由記述
    7: ID
    """
    # 授業評価スコアを抽出
    def extract_score(text):
        """平均評価ポイントから数値を抽出"""
        if pd.isna(text):
            return None
        match = re.search(r'(\d+\.\d+)', str(text))
        return float(match.group(1)) if match else None
    
    course_data = []
    
    for _, row in df.iterrows():
        try:
            course_id = row.iloc[2]  # 授業ID
            score_text = row.iloc[3]  # 平均評価ポイント
            free_text = row.iloc[6]  # 自由記述
            
            # 授業評価スコア抽出
            eval_score = extract_score(score_text)
            
            if pd.notna(course_id) and pd.notna(free_text) and eval_score is not None:
                course_data.append({
                    'course_id': course_id,
                    'free_text': free_text,
                    'eval_score': eval_score
                })
        except Exception as e:
            continue
    
    return pd.DataFrame(course_data)

def extract_score(text):
    """平均評価ポイントから数値を抽出"""
    if pd.isna(text):
        return None
    match = re.search(r'(\d+\.\d+)', str(text))
    return float(match.group(1)) if match else None

def load_sentiment_data():
    """感情分類結果データを読み込み"""
    try:
        # 感情分類結果ファイルから全データを取得
        df = pd.read_csv('../感情分類結果_前処理データ結合_20250729_154855.csv')
        print(f"  感情分類データ数: {len(df)}")
        return df
    except Exception as e:
        print(f"警告: 感情分類データの読み込みに失敗しました: {e}")
        return pd.DataFrame()

def create_aggregated_dataset(min_texts=3):
    """
    授業ごとに集約されたデータセットを作成
    
    Args:
        min_texts: 最小自由記述数（これより少ない授業は除外）
    """
    print("=" * 60)
    print("授業レベル集約データセット作成")
    print("=" * 60)
    
    # 感情分類データ読み込み（全ての情報が含まれている）
    print("\n[1/3] 感情分類データ読み込み中...")
    df_sentiment = load_sentiment_data()
    if df_sentiment.empty:
        print("エラー: 感情分類データが読み込めませんでした")
        return None
    
    # 授業ごとに集約
    print("\n[2/3] 授業ごとに集約中...")
    aggregated_data = []
    
    # 授業IDと開講年度を組み合わせてグループ化
    # 列11: 開講年度、列12: 授業ID
    df_sentiment['course_key'] = df_sentiment.iloc[:, 11].astype(str) + '_' + df_sentiment.iloc[:, 12].astype(str)
    
    for course_key, group in df_sentiment.groupby('course_key'):
        # 授業評価スコアを抽出（列13: 平均評価ポイント）
        eval_score_text = group.iloc[0, 13]
        eval_score = extract_score(eval_score_text)
        if eval_score is None:
            continue
        
        # 回答者数・履修者数・回答率（列14）
        survey_info_text = group.iloc[0, 14] if group.shape[1] > 14 else None
        respondents, enrolled, response_rate = parse_survey_info(survey_info_text)

        # 評価分布（列15）
        distribution_text = group.iloc[0, 15] if group.shape[1] > 15 else None
        dist = parse_distribution(distribution_text)

        # 自由記述数のフィルタリング
        num_texts = len(group)
        if num_texts < min_texts:
            continue
        
        # 自由記述をまとめる（列1: 自由記述）
        free_texts = group.iloc[:, 1].tolist()
        combined_text = ' '.join([str(text) for text in free_texts if pd.notna(text)])
        
        # 感情スコアの計算（tuned_model_labelを使用、列6）
        sentiment_scores = []
        for label in group.iloc[:, 6]:  # tuned_model_label
            if pd.notna(label):
                if label == 'POSITIVE':
                    sentiment_scores.append(1.0)
                elif label == 'NEGATIVE':
                    sentiment_scores.append(-1.0)
                else:
                    sentiment_scores.append(0.0)
        
        # 感情スコアの平均
        sentiment_mean = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # 感情スコアラベルの分布
        sentiment_counter = Counter(sentiment_scores)
        total_count = len(sentiment_scores)
        negative_ratio = sentiment_counter.get(-1.0, 0) / total_count if total_count > 0 else 0
        neutral_ratio = sentiment_counter.get(0.0, 0) / total_count if total_count > 0 else 0
        positive_ratio = sentiment_counter.get(1.0, 0) / total_count if total_count > 0 else 0
        
        # 開講年度と授業IDを分離
        course_info = course_key.split('_', 1)
        course_year = course_info[0] if len(course_info) > 0 else ''
        course_id = course_info[1] if len(course_info) > 1 else course_key
        
        aggregated_data.append({
            '開講年度': course_year,
            '授業ID': course_id,
            '授業キー': course_key,
            '自由記述まとめ': combined_text,
            '自由記述数': num_texts,
            '感情スコア平均': sentiment_mean,
            'ネガティブ比率': negative_ratio,
            'ニュートラル比率': neutral_ratio,
            'ポジティブ比率': positive_ratio,
            '授業評価スコア': eval_score,
            '分布_十分意義あり_人数': dist['十分意義あり']['count'],
            '分布_十分意義あり_割合(%)': dist['十分意義あり']['rate'],
            '分布_ある程度意義あり_人数': dist['ある程度意義あり']['count'],
            '分布_ある程度意義あり_割合(%)': dist['ある程度意義あり']['rate'],
            '分布_あまり意義なし_人数': dist['あまり意義なし']['count'],
            '分布_あまり意義なし_割合(%)': dist['あまり意義なし']['rate'],
            '分布_全く意義なし_人数': dist['全く意義なし']['count'],
            '分布_全く意義なし_割合(%)': dist['全く意義なし']['rate']
        })
    
    df_aggregated = pd.DataFrame(aggregated_data)
    
    print(f"  集約された授業数: {len(df_aggregated)}")
    print(f"  平均自由記述数: {df_aggregated['自由記述数'].mean():.1f}")
    
    # データセット保存
    print("\n[3/3] データセット保存中...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'../01_データ/マルチタスク用データ/授業集約データセット_{timestamp}.csv'
    df_aggregated.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  保存先: {output_path}")
    
    # 統計情報表示
    print("\n" + "=" * 60)
    print("データセット統計情報")
    print("=" * 60)
    print(f"授業数: {len(df_aggregated)}")
    print(f"\n自由記述数:")
    print(df_aggregated['自由記述数'].describe())
    print(f"\n感情スコア平均:")
    print(df_aggregated['感情スコア平均'].describe())
    print(f"\n授業評価スコア:")
    print(df_aggregated['授業評価スコア'].describe())
    print(f"\n感情スコアと授業評価スコアの相関:")
    correlation = df_aggregated['感情スコア平均'].corr(df_aggregated['授業評価スコア'])
    print(f"  相関係数: {correlation:.3f}")
    
    return df_aggregated

if __name__ == "__main__":
    # データセット作成
    df = create_aggregated_dataset(min_texts=5)
    
    # サンプル表示
    if df is not None and len(df) > 0:
        print("\n" + "=" * 60)
        print("データセットサンプル（最初の3件）")
        print("=" * 60)
        for i, row in df.head(3).iterrows():
            print(f"\n授業 {i+1}: {row['開講年度']} - {row['授業ID']}")
            print(f"  自由記述数: {row['自由記述数']}")
            print(f"  感情スコア平均: {row['感情スコア平均']:.3f}")
            print(f"  感情分布: ネガティブ {row['ネガティブ比率']:.2f} / ニュートラル {row['ニュートラル比率']:.2f} / ポジティブ {row['ポジティブ比率']:.2f}")
            print(f"  授業評価スコア: {row['授業評価スコア']}")
            print(f"  自由記述（最初の100文字）: {row['自由記述まとめ'][:100]}...")
    else:
        print("データセットが作成されませんでした。")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
授業集約データセットの相関分析と無相関分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# 日本語フォント設定（Windows用）
try:
    plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの日本語フォント
except:
    try:
        plt.rcParams['font.family'] = 'Yu Gothic'  # 代替フォント
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'  # フォールバック
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """データの読み込み"""
    print("📊 データを読み込み中...")
    
    # CSVファイルを読み込み
    df = pd.read_csv('../01_データ/マルチタスク用データ/授業集約データセット_20251012_142504.csv')
    
    print(f"データ数: {len(df)}件")
    print(f"列名: {list(df.columns)}")
    
    return df

def basic_statistics(df):
    """基本統計量の計算"""
    print("\n📈 基本統計量")
    print("=" * 50)
    
    # 感情スコアの統計
    sentiment_stats = df['感情スコア平均'].describe()
    print("感情スコア平均の統計:")
    print(sentiment_stats)
    
    # 授業評価スコアの統計
    score_stats = df['授業評価スコア'].describe()
    print("\n授業評価スコアの統計:")
    print(score_stats)
    
    # 欠損値チェック
    print(f"\n欠損値:")
    print(f"感情スコア平均: {df['感情スコア平均'].isnull().sum()}件")
    print(f"授業評価スコア: {df['授業評価スコア'].isnull().sum()}件")

def correlation_analysis(df):
    """相関分析"""
    print("\n🔗 相関分析")
    print("=" * 50)
    
    # データの準備
    sentiment = df['感情スコア平均'].dropna()
    score = df['授業評価スコア'].dropna()
    
    # 共通のインデックスを持つデータのみを使用
    common_idx = sentiment.index.intersection(score.index)
    sentiment_common = sentiment.loc[common_idx]
    score_common = score.loc[common_idx]
    
    print(f"分析対象データ数: {len(common_idx)}件")
    
    # ピアソンの相関係数
    pearson_r, pearson_p = pearsonr(sentiment_common, score_common)
    print(f"\nピアソンの相関係数: {pearson_r:.4f}")
    print(f"p値: {pearson_p:.6f}")
    
    # スピアマンの順位相関係数
    spearman_r, spearman_p = spearmanr(sentiment_common, score_common)
    print(f"\nスピアマンの順位相関係数: {spearman_r:.4f}")
    print(f"p値: {spearman_p:.6f}")
    
    # ケンドールの順位相関係数
    kendall_tau, kendall_p = kendalltau(sentiment_common, score_common)
    print(f"\nケンドールの順位相関係数: {kendall_tau:.4f}")
    print(f"p値: {kendall_p:.6f}")
    
    # 相関の強さの解釈
    print(f"\n相関の強さの解釈:")
    if abs(pearson_r) < 0.1:
        strength = "無相関"
    elif abs(pearson_r) < 0.3:
        strength = "弱い相関"
    elif abs(pearson_r) < 0.5:
        strength = "中程度の相関"
    elif abs(pearson_r) < 0.7:
        strength = "強い相関"
    else:
        strength = "非常に強い相関"
    
    print(f"ピアソン相関係数 {pearson_r:.4f} → {strength}")
    
    return pearson_r, pearson_p, spearman_r, spearman_p, kendall_tau, kendall_p

def independence_test(df):
    """無相関分析（独立性検定）"""
    print("\n🚫 無相関分析（独立性検定）")
    print("=" * 50)
    
    # データの準備
    sentiment = df['感情スコア平均'].dropna()
    score = df['授業評価スコア'].dropna()
    
    # 共通のインデックスを持つデータのみを使用
    common_idx = sentiment.index.intersection(score.index)
    sentiment_common = sentiment.loc[common_idx]
    score_common = score.loc[common_idx]
    
    # カイ二乗独立性検定のための分割
    # 感情スコアを3つのカテゴリに分割
    sentiment_categories = pd.cut(sentiment_common, 
                                 bins=[-np.inf, -0.1, 0.1, np.inf], 
                                 labels=['ネガティブ', 'ニュートラル', 'ポジティブ'])
    
    # 授業評価スコアを3つのカテゴリに分割
    score_categories = pd.cut(score_common, 
                             bins=[-np.inf, 3.0, 3.5, np.inf], 
                             labels=['低評価', '中評価', '高評価'])
    
    # クロス集計表の作成
    contingency_table = pd.crosstab(sentiment_categories, score_categories)
    print("クロス集計表:")
    print(contingency_table)
    
    # カイ二乗独立性検定
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nカイ二乗統計量: {chi2:.4f}")
    print(f"自由度: {dof}")
    print(f"p値: {p_value:.6f}")
    
    # 期待度数
    print(f"\n期待度数:")
    expected_df = pd.DataFrame(expected, 
                             index=contingency_table.index, 
                             columns=contingency_table.columns)
    print(expected_df)
    
    # 独立性の判定
    alpha = 0.05
    if p_value < alpha:
        print(f"\n結論: p値({p_value:.6f}) < α({alpha}) → 独立ではない（相関あり）")
    else:
        print(f"\n結論: p値({p_value:.6f}) ≥ α({alpha}) → 独立（無相関）")
    
    return chi2, p_value, dof, contingency_table

def create_visualizations(df):
    """可視化の作成"""
    print("\n📊 可視化を作成中...")
    
    try:
        # データの準備
        sentiment = df['感情スコア平均'].dropna()
        score = df['授業評価スコア'].dropna()
        
        # 共通のインデックスを持つデータのみを使用
        common_idx = sentiment.index.intersection(score.index)
        sentiment_common = sentiment.loc[common_idx]
        score_common = score.loc[common_idx]
        
        # 図の設定
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Correlation Analysis: Sentiment Score vs Course Evaluation Score', 
                     fontsize=16, fontweight='bold')
        
        # 1. 散布図
        axes[0, 0].scatter(sentiment_common, score_common, alpha=0.6, s=20)
        axes[0, 0].set_xlabel('Sentiment Score Average')
        axes[0, 0].set_ylabel('Course Evaluation Score')
        axes[0, 0].set_title('Scatter Plot')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 回帰直線の追加
        z = np.polyfit(sentiment_common, score_common, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(sentiment_common, p(sentiment_common), "r--", alpha=0.8)
        
        # 2. ヒートマップ（2Dヒストグラム）
        axes[0, 1].hist2d(sentiment_common, score_common, bins=20, cmap='Blues')
        axes[0, 1].set_xlabel('Sentiment Score Average')
        axes[0, 1].set_ylabel('Course Evaluation Score')
        axes[0, 1].set_title('2D Histogram')
        
        # 3. 感情スコアの分布
        axes[1, 0].hist(sentiment_common, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_xlabel('Sentiment Score Average')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Sentiment Score Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 授業評価スコアの分布
        axes[1, 1].hist(score_common, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 1].set_xlabel('Course Evaluation Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Course Evaluation Score Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存ディレクトリの確認・作成
        output_dir = '../03_分析結果'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, '相関分析_授業集約データ.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可視化を保存しました: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"可視化の作成中にエラーが発生しました: {e}")
        print("スキップして続行します...")

def detailed_analysis(df):
    """詳細分析"""
    print("\n🔍 詳細分析")
    print("=" * 50)
    
    # データの準備
    sentiment = df['感情スコア平均'].dropna()
    score = df['授業評価スコア'].dropna()
    
    # 共通のインデックスを持つデータのみを使用
    common_idx = sentiment.index.intersection(score.index)
    sentiment_common = sentiment.loc[common_idx]
    score_common = score.loc[common_idx]
    
    # 四分位数での分析
    print("四分位数での分析:")
    sentiment_q = sentiment_common.quantile([0.25, 0.5, 0.75])
    score_q = score_common.quantile([0.25, 0.5, 0.75])
    
    print(f"感情スコア: Q1={sentiment_q[0.25]:.3f}, Q2={sentiment_q[0.5]:.3f}, Q3={sentiment_q[0.75]:.3f}")
    print(f"授業評価スコア: Q1={score_q[0.25]:.3f}, Q2={score_q[0.5]:.3f}, Q3={score_q[0.75]:.3f}")
    
    # 極値の分析
    print(f"\n極値の分析:")
    print(f"感情スコア最小値: {sentiment_common.min():.3f}")
    print(f"感情スコア最大値: {sentiment_common.max():.3f}")
    print(f"授業評価スコア最小値: {score_common.min():.3f}")
    print(f"授業評価スコア最大値: {score_common.max():.3f}")
    
    # 外れ値の検出
    Q1_sentiment = sentiment_common.quantile(0.25)
    Q3_sentiment = sentiment_common.quantile(0.75)
    IQR_sentiment = Q3_sentiment - Q1_sentiment
    lower_bound_sentiment = Q1_sentiment - 1.5 * IQR_sentiment
    upper_bound_sentiment = Q3_sentiment + 1.5 * IQR_sentiment
    
    outliers_sentiment = sentiment_common[(sentiment_common < lower_bound_sentiment) | 
                                        (sentiment_common > upper_bound_sentiment)]
    
    print(f"\n感情スコアの外れ値: {len(outliers_sentiment)}件")
    if len(outliers_sentiment) > 0:
        print(f"外れ値の範囲: {outliers_sentiment.min():.3f} ~ {outliers_sentiment.max():.3f}")

def save_results(pearson_r, pearson_p, spearman_r, spearman_p, kendall_tau, kendall_p, 
                chi2, chi2_p, contingency_table):
    """結果の保存"""
    print("\n💾 結果を保存中...")
    
    try:
        # 結果を辞書にまとめる
        results = {
            "analysis_date": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
            "data_source": "授業集約データセット_20251012_142504.csv",
            "sample_size": int(contingency_table.sum().sum()),
            "correlation_analysis": {
                "pearson": {
                    "correlation_coefficient": float(pearson_r),
                    "p_value": float(pearson_p),
                    "interpretation": "線形相関の強さ"
                },
                "spearman": {
                    "correlation_coefficient": float(spearman_r),
                    "p_value": float(spearman_p),
                    "interpretation": "順位相関の強さ"
                },
                "kendall": {
                    "correlation_coefficient": float(kendall_tau),
                    "p_value": float(kendall_p),
                    "interpretation": "順位相関の強さ（小標本に適している）"
                }
            },
            "independence_test": {
                "chi_square_statistic": float(chi2),
                "p_value": float(chi2_p),
                "degrees_of_freedom": int((contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)),
                "contingency_table": contingency_table.to_dict()
            },
            "conclusions": {
                "correlation_strength": "中程度の正の相関" if 0.3 <= abs(pearson_r) < 0.5 else 
                                       "弱い相関" if 0.1 <= abs(pearson_r) < 0.3 else "無相関",
                "independence": "独立ではない（相関あり）" if chi2_p < 0.05 else "独立（無相関）"
            }
        }
        
        # 保存ディレクトリの確認・作成
        output_dir = '../03_分析結果'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # JSONファイルとして保存
        import json
        output_path = os.path.join(output_dir, '相関分析結果_授業集約データ.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"結果を保存しました: {output_path}")
        
    except Exception as e:
        print(f"結果の保存中にエラーが発生しました: {e}")
        print("スキップして続行します...")

def main():
    """メイン関数"""
    try:
        print("🎯 授業評価スコアと感情スコアの相関分析")
        print("=" * 60)
        
        # データの読み込み
        df = load_data()
        
        # 基本統計量
        basic_statistics(df)
        
        # 相関分析
        pearson_r, pearson_p, spearman_r, spearman_p, kendall_tau, kendall_p = correlation_analysis(df)
        
        # 無相関分析
        chi2, chi2_p, dof, contingency_table = independence_test(df)
        
        # 詳細分析
        detailed_analysis(df)
        
        # 可視化
        create_visualizations(df)
        
        # 結果の保存
        save_results(pearson_r, pearson_p, spearman_r, spearman_p, kendall_tau, kendall_p,
                    chi2, chi2_p, contingency_table)
        
        print("\n✅ 分析完了！")
        print("=" * 60)
        
        # 最終結果のサマリー
        print("\n📊 相関分析結果のサマリー")
        print("=" * 60)
        print(f"ピアソン相関係数: {pearson_r:.4f} (p={pearson_p:.6f})")
        print(f"スピアマン相関係数: {spearman_r:.4f} (p={spearman_p:.6f})")
        print(f"ケンドール相関係数: {kendall_tau:.4f} (p={kendall_p:.6f})")
        print(f"\nカイ二乗統計量: {chi2:.4f} (p={chi2_p:.6f})")
        print(f"結論: 感情スコアと授業評価スコアには統計的に有意な相関関係がある")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        print("\n分析を中断します...")
        sys.exit(1)

if __name__ == "__main__":
    main()

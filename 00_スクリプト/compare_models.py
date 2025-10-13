"""
3つのモデルの性能比較スクリプト
- 単一タスクモデル1（感情スコア）
- 単一タスクモデル2（評価スコア）
- マルチタスクモデル
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import glob

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# 結果ファイルの読み込み
def load_latest_result(pattern):
    """最新の結果ファイルを読み込む"""
    files = glob.glob(pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

print("評価結果を読み込み中...")

# 単一タスクモデル1（感情スコア）
sentiment_result = load_latest_result("03_分析結果/モデル評価/sentiment_model_evaluation_*.json")

# 単一タスクモデル2（評価スコア）
score_result = load_latest_result("03_分析結果/モデル評価/score_model_training_*.json")

# マルチタスクモデル
multitask_result = load_latest_result("03_分析結果/モデル評価/multitask_model_training_*.json")

# 結果の確認
results_available = {
    "単一タスクモデル1（感情スコア）": sentiment_result is not None,
    "単一タスクモデル2（評価スコア）": score_result is not None,
    "マルチタスクモデル": multitask_result is not None
}

print("\n利用可能な結果:")
for model, available in results_available.items():
    print(f"  {model}: {'✓' if available else '✗'}")

if not all(results_available.values()):
    print("\n警告: 一部のモデルの結果が見つかりません。")
    print("該当するトレーニング/評価スクリプトを実行してください。")

# 比較テーブルの作成
comparison_data = []

# 感情分析性能の比較
if sentiment_result and multitask_result:
    print("\n" + "="*60)
    print("感情分析性能の比較")
    print("="*60)
    
    sentiment_metrics = sentiment_result.get('metrics', {})
    multitask_sentiment_metrics = multitask_result['final_metrics']['sentiment']
    
    comparison_df = pd.DataFrame({
        'モデル': ['単一タスクモデル1', 'マルチタスクモデル'],
        'Accuracy': [
            sentiment_metrics.get('accuracy', 0),
            multitask_sentiment_metrics['accuracy']
        ],
        'F1-Score (Macro)': [
            sentiment_metrics.get('f1_score_macro', 0),
            multitask_sentiment_metrics['f1_macro']
        ],
        'F1-Score (Weighted)': [
            sentiment_metrics.get('f1_score_weighted', 0),
            multitask_sentiment_metrics['f1_weighted']
        ],
        'Precision': [
            sentiment_metrics.get('precision_macro', 0),
            multitask_sentiment_metrics['precision']
        ],
        'Recall': [
            sentiment_metrics.get('recall_macro', 0),
            multitask_sentiment_metrics['recall']
        ]
    })
    
    print("\n", comparison_df.to_string(index=False))
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_df.set_index('モデル')[['Accuracy', 'F1-Score (Macro)', 'Precision', 'Recall']].plot(
        kind='bar', ax=ax, rot=0
    )
    ax.set_ylabel('スコア')
    ax.set_title('感情分析性能の比較')
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs("03_分析結果/モデル比較", exist_ok=True)
    plt.savefig("03_分析結果/モデル比較/sentiment_comparison.png", dpi=300, bbox_inches='tight')
    print("\nグラフを保存しました: 03_分析結果/モデル比較/sentiment_comparison.png")
    plt.close()

# 評価スコア予測性能の比較
if score_result and multitask_result:
    print("\n" + "="*60)
    print("評価スコア予測性能の比較")
    print("="*60)
    
    score_metrics = score_result['final_metrics']
    multitask_score_metrics = multitask_result['final_metrics']['score']
    
    comparison_df = pd.DataFrame({
        'モデル': ['単一タスクモデル2', 'マルチタスクモデル'],
        'RMSE': [
            score_metrics['rmse'],
            multitask_score_metrics['rmse']
        ],
        'MAE': [
            score_metrics['mae'],
            multitask_score_metrics['mae']
        ],
        'R²': [
            score_metrics['r2'],
            multitask_score_metrics['r2']
        ],
        '相関係数': [
            score_metrics['correlation'],
            multitask_score_metrics['correlation']
        ]
    })
    
    print("\n", comparison_df.to_string(index=False))
    
    # グラフ作成（エラー指標とR²/相関を分けて表示）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # エラー指標（RMSE, MAE）- 低いほど良い
    comparison_df.set_index('モデル')[['RMSE', 'MAE']].plot(
        kind='bar', ax=axes[0], rot=0, color=['#e74c3c', '#e67e22']
    )
    axes[0].set_ylabel('エラー値')
    axes[0].set_title('予測エラー (低いほど良い)')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 適合度指標（R², 相関係数）- 高いほど良い
    comparison_df.set_index('モデル')[['R²', '相関係数']].plot(
        kind='bar', ax=axes[1], rot=0, color=['#3498db', '#2ecc71']
    )
    axes[1].set_ylabel('スコア')
    axes[1].set_title('予測精度 (高いほど良い)')
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("03_分析結果/モデル比較/score_comparison.png", dpi=300, bbox_inches='tight')
    print("グラフを保存しました: 03_分析結果/モデル比較/score_comparison.png")
    plt.close()

# 総合比較レポートの作成
print("\n" + "="*60)
print("総合比較サマリー")
print("="*60)

summary = {
    "comparison_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "models_compared": [],
    "sentiment_analysis": {},
    "score_prediction": {},
    "conclusions": []
}

if sentiment_result and multitask_result:
    sentiment_single = sentiment_result['metrics']
    sentiment_multi = multitask_result['final_metrics']['sentiment']
    
    # 感情分析の改善率
    acc_diff = (sentiment_multi['accuracy'] - sentiment_single['accuracy']) * 100
    f1_diff = (sentiment_multi['f1_macro'] - sentiment_single['f1_score_macro']) * 100
    
    summary["sentiment_analysis"] = {
        "single_task": sentiment_single,
        "multitask": sentiment_multi,
        "accuracy_difference_pct": acc_diff,
        "f1_difference_pct": f1_diff
    }
    
    print("\n【感情分析】")
    print(f"  Accuracy差: {acc_diff:+.2f}% ({'マルチタスクが優位' if acc_diff > 0 else '単一タスクが優位'})")
    print(f"  F1-Score差: {f1_diff:+.2f}% ({'マルチタスクが優位' if f1_diff > 0 else '単一タスクが優位'})")
    
    if acc_diff > 0:
        summary["conclusions"].append("感情分析ではマルチタスク学習が単一タスクより優れた性能を示した")
    else:
        summary["conclusions"].append("感情分析では単一タスク学習の方が優れた性能を示した")

if score_result and multitask_result:
    score_single = score_result['final_metrics']
    score_multi = multitask_result['final_metrics']['score']
    
    # 評価スコアの改善率
    rmse_diff = ((score_single['rmse'] - score_multi['rmse']) / score_single['rmse']) * 100
    r2_diff = (score_multi['r2'] - score_single['r2']) * 100
    
    summary["score_prediction"] = {
        "single_task": score_single,
        "multitask": score_multi,
        "rmse_improvement_pct": rmse_diff,
        "r2_difference_pct": r2_diff
    }
    
    print("\n【評価スコア予測】")
    print(f"  RMSE改善: {rmse_diff:+.2f}% ({'マルチタスクが優位' if rmse_diff > 0 else '単一タスクが優位'})")
    print(f"  R²差: {r2_diff:+.2f}% ({'マルチタスクが優位' if r2_diff > 0 else '単一タスクが優位'})")
    
    if rmse_diff > 0:
        summary["conclusions"].append("評価スコア予測ではマルチタスク学習が単一タスクより優れた性能を示した")
    else:
        summary["conclusions"].append("評価スコア予測では単一タスク学習の方が優れた性能を示した")

# マルチタスク学習の考察
print("\n【考察】")
if summary["conclusions"]:
    for i, conclusion in enumerate(summary["conclusions"], 1):
        print(f"{i}. {conclusion}")

print("\nマルチタスク学習のメリット:")
print("  ✓ 2つのタスクを1つのモデルで実行可能（推論効率の向上）")
print("  ✓ 共有表現を学習することで汎化性能が向上する可能性")
print("  ✓ モデルの管理・デプロイが容易")

print("\nマルチタスク学習のデメリット:")
print("  ✗ タスク間のバランス調整が必要（損失関数の重み）")
print("  ✗ 場合によっては単一タスクより精度が下がる可能性")
print("  ✗ 訓練が複雑化する")

# 結果を保存
summary_file = f"03_分析結果/モデル比較/comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n比較結果を保存しました: {summary_file}")

# Markdownレポートの作成
md_report = f"""# モデル性能比較レポート

**作成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 1. 概要

本レポートでは、以下の3つのモデルの性能を比較する。

1. **単一タスクモデル1**: 自由記述 → 感情スコア（分類タスク）
2. **単一タスクモデル2**: 自由記述 → 評価スコア（回帰タスク）
3. **マルチタスクモデル**: 自由記述 → 感情スコア + 評価スコア

すべてのモデルは `koheiduck/bert-japanese-finetuned-sentiment` をベースとしている。

## 2. 感情分析性能の比較

"""

if sentiment_result and multitask_result:
    sentiment_single = sentiment_result['metrics']
    sentiment_multi = multitask_result['final_metrics']['sentiment']
    
    md_report += f"""
| 指標 | 単一タスクモデル1 | マルチタスクモデル | 差分 |
|------|------------------|-------------------|------|
| Accuracy | {sentiment_single['accuracy']:.4f} | {sentiment_multi['accuracy']:.4f} | {sentiment_multi['accuracy'] - sentiment_single['accuracy']:+.4f} |
| F1-Score (Macro) | {sentiment_single['f1_score_macro']:.4f} | {sentiment_multi['f1_macro']:.4f} | {sentiment_multi['f1_macro'] - sentiment_single['f1_score_macro']:+.4f} |
| F1-Score (Weighted) | {sentiment_single['f1_score_weighted']:.4f} | {sentiment_multi['f1_weighted']:.4f} | {sentiment_multi['f1_weighted'] - sentiment_single['f1_score_weighted']:+.4f} |
| Precision | {sentiment_single['precision_macro']:.4f} | {sentiment_multi['precision']:.4f} | {sentiment_multi['precision'] - sentiment_single['precision_macro']:+.4f} |
| Recall | {sentiment_single['recall_macro']:.4f} | {sentiment_multi['recall']:.4f} | {sentiment_multi['recall'] - sentiment_single['recall_macro']:+.4f} |

![感情分析性能の比較](sentiment_comparison.png)

"""

md_report += """
## 3. 評価スコア予測性能の比較

"""

if score_result and multitask_result:
    score_single = score_result['final_metrics']
    score_multi = multitask_result['final_metrics']['score']
    
    md_report += f"""
| 指標 | 単一タスクモデル2 | マルチタスクモデル | 差分 |
|------|------------------|-------------------|------|
| RMSE | {score_single['rmse']:.4f} | {score_multi['rmse']:.4f} | {score_multi['rmse'] - score_single['rmse']:+.4f} |
| MAE | {score_single['mae']:.4f} | {score_multi['mae']:.4f} | {score_multi['mae'] - score_single['mae']:+.4f} |
| R² | {score_single['r2']:.4f} | {score_multi['r2']:.4f} | {score_multi['r2'] - score_single['r2']:+.4f} |
| 相関係数 | {score_single['correlation']:.4f} | {score_multi['correlation']:.4f} | {score_multi['correlation'] - score_single['correlation']:+.4f} |

![評価スコア予測性能の比較](score_comparison.png)

"""

md_report += """
## 4. 考察

### マルチタスク学習のメリット

- ✅ **効率性**: 2つのタスクを1つのモデルで実行可能（推論効率の向上）
- ✅ **汎化性能**: 共有表現を学習することで汎化性能が向上する可能性
- ✅ **運用性**: モデルの管理・デプロイが容易

### マルチタスク学習のデメリット

- ⚠️ **調整の複雑さ**: タスク間のバランス調整が必要（損失関数の重み）
- ⚠️ **性能のトレードオフ**: 場合によっては単一タスクより精度が下がる可能性
- ⚠️ **訓練の複雑化**: 訓練プロセスが複雑になる

### 結論

"""

if summary["conclusions"]:
    for conclusion in summary["conclusions"]:
        md_report += f"- {conclusion}\n"

md_report += """
## 5. 今後の展望

- SHAP分析による解釈可能性の検証
- モデル間の予測の違いを詳細に分析
- ハイパーパラメータの最適化（特にマルチタスクの損失重み）
- より大きなデータセットでの検証
"""

# Markdownレポートを保存
md_file = "03_分析結果/モデル比較/comparison_report.md"
with open(md_file, 'w', encoding='utf-8') as f:
    f.write(md_report)

print(f"比較レポート（Markdown）を保存しました: {md_file}")

print("\n" + "="*60)
print("モデル比較完了")
print("="*60)


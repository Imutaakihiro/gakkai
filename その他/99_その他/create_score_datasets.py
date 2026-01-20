import pandas as pd
import numpy as np

def create_score_datasets():
    """
    既存の感情スコア分割データと同じ自由記述を使って、
    授業評価スコア用のデータセットを作成
    """
    
    # 1. 元データを読み込み
    print("元データを読み込み中...")
    multitask_df = pd.read_csv('マルチタスク用データ/マルチタスク学習用データセット_20250930_202839.csv')
    
    # 2. 既存の感情スコア分割データを読み込み
    print("既存の感情スコア分割データを読み込み中...")
    sentiment_train = pd.read_csv('チューニング用データ/finetuning_train_20250710_220621.csv')
    sentiment_val = pd.read_csv('チューニング用データ/finetuning_val_20250710_220621.csv')
    
    # 3. 自由記述→授業評価スコアのマッピング辞書を作成
    print("マッピング辞書を作成中...")
    text_to_score = dict(zip(multitask_df['自由記述'], multitask_df['授業評価スコア']))
    
    # 4. 感情スコア訓練データから授業評価スコア訓練データを作成
    print("授業評価スコア訓練データを作成中...")
    score_train_data = []
    for _, row in sentiment_train.iterrows():
        text = row['text']
        if text in text_to_score:
            score_train_data.append({
                'text': text,
                'label': text_to_score[text]
            })
        else:
            print(f"警告: '{text}' が見つかりません")
    
    score_train_df = pd.DataFrame(score_train_data)
    print(f"訓練データ: {len(score_train_df)}件")
    
    # 5. 感情スコア検証データから授業評価スコア検証データを作成
    print("授業評価スコア検証データを作成中...")
    score_val_data = []
    for _, row in sentiment_val.iterrows():
        text = row['text']
        if text in text_to_score:
            score_val_data.append({
                'text': text,
                'label': text_to_score[text]
            })
        else:
            print(f"警告: '{text}' が見つかりません")
    
    score_val_df = pd.DataFrame(score_val_data)
    print(f"検証データ: {len(score_val_df)}件")
    
    # 6. データセットを保存
    print("データセットを保存中...")
    score_train_df.to_csv('score_train_dataset.csv', index=False)
    score_val_df.to_csv('score_val_dataset.csv', index=False)
    
    # 7. 統計情報を表示
    print("\n=== データセット統計 ===")
    print(f"訓練データ: {len(score_train_df)}件")
    print(f"検証データ: {len(score_val_df)}件")
    print(f"合計: {len(score_train_df) + len(score_val_df)}件")
    
    print("\n=== 授業評価スコア統計 ===")
    print(f"訓練データ - 平均: {score_train_df['label'].mean():.3f}, 標準偏差: {score_train_df['label'].std():.3f}")
    print(f"検証データ - 平均: {score_val_df['label'].mean():.3f}, 標準偏差: {score_val_df['label'].std():.3f}")
    
    print("\n=== 範囲 ===")
    print(f"訓練データ - 最小: {score_train_df['label'].min():.3f}, 最大: {score_train_df['label'].max():.3f}")
    print(f"検証データ - 最小: {score_val_df['label'].min():.3f}, 最大: {score_val_df['label'].max():.3f}")
    
    return score_train_df, score_val_df

if __name__ == "__main__":
    train_df, val_df = create_score_datasets()
    
    print("\n=== 作成完了 ===")
    print("score_train_dataset.csv - 授業評価スコア訓練データ")
    print("score_val_dataset.csv - 授業評価スコア検証データ")
    print("\nこれらのファイルは感情スコア分割データと同じ自由記述を使用しています。")


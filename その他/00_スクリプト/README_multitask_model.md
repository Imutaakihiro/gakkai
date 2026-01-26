# マルチタスク学習モデル

汎用的なマルチタスク学習モデルの実装です。複数のタスクを同時に学習できる柔軟な構造を提供します。

## 📁 ファイル構成

- `multitask_model.py`: マルチタスクモデルの定義
- `train_multitask_model.py`: 学習スクリプト
- `README_multitask_model.md`: このファイル

## 🎯 特徴

1. **柔軟なタスク定義**: 回帰、分類、順序回帰など複数のタスクタイプに対応
2. **カスタマイズ可能**: タスクごとに異なるネットワーク構造を設定可能
3. **長文対応**: チャンク化による長文処理
4. **マルチデバイス対応**: CUDA、MPS、DirectML、CPUに対応

## 🚀 基本的な使い方

### 1. タスク設定のカスタマイズ

`train_multitask_model.py`の`TASK_CONFIGS`を編集して、学習したいタスクを定義します。

```python
TASK_CONFIGS = [
    {
        "name": "sentiment",           # タスク名（一意）
        "type": "regression",          # タスクタイプ: "regression", "classification", "ordinal"
        "output_size": 1,              # 出力サイズ（回帰: 1, 分類: クラス数, 順序: カテゴリ数）
        "hidden_sizes": [256],         # 隠れ層のサイズ
        "weight": 0.5,                 # 損失関数の重み
        "activation": "relu"           # 活性化関数: "relu", "gelu", "tanh"
    },
    {
        "name": "course_score",
        "type": "regression",
        "output_size": 1,
        "hidden_sizes": [256],
        "weight": 0.5,
        "activation": "relu"
    }
]
```

### 2. カスタムタスクの追加

新しいタスクを追加する場合、以下の情報を設定します：

- **回帰タスク**: 連続値を予測
  ```python
  {
      "name": "custom_regression",
      "type": "regression",
      "output_size": 1,
      "column_name": "ターゲット列名",  # データフレームの列名
      "hidden_sizes": [256, 128],
      "weight": 1.0
  }
  ```

- **分類タスク**: クラス分類
  ```python
  {
      "name": "custom_classification",
      "type": "classification",
      "output_size": 5,  # クラス数
      "column_name": "ターゲット列名",
      "hidden_sizes": [256],
      "weight": 1.0
  }
  ```

- **順序回帰タスク**: 順序のあるカテゴリ（1-4など）
  ```python
  {
      "name": "custom_ordinal",
      "type": "ordinal",
      "output_size": 4,  # カテゴリ数（1-4なら4）
      "column_name": "ターゲット列名",
      "hidden_sizes": [256],
      "weight": 1.0
  }
  ```

### 3. データの準備

データセットは以下の列を含む必要があります：

- **テキスト列**: `自由記述まとめ`、`text`、`自由記述`、`comments`のいずれか
- **タスクのターゲット列**: 各タスクに対応する列

例：
- `感情スコア平均` → `sentiment`タスク
- `授業評価スコア` → `course_score`タスク

### 4. 学習の実行

```bash
cd その他/00_スクリプト
python train_multitask_model.py
```

## 📊 タスクタイプの詳細

### Regression (回帰)

連続値を予測するタスク。MSE損失を使用。

```python
{
    "name": "sentiment",
    "type": "regression",
    "output_size": 1,
    "hidden_sizes": [256],
    "weight": 0.5
}
```

### Classification (分類)

クラス分類タスク。NLL損失を使用。

```python
{
    "name": "category",
    "type": "classification",
    "output_size": 5,  # 5クラス分類
    "hidden_sizes": [256],
    "weight": 1.0
}
```

### Ordinal (順序回帰)

順序のあるカテゴリを予測（例: 1-4の評価）。累積ロジットモデルを使用。

```python
{
    "name": "satisfaction",
    "type": "ordinal",
    "output_size": 4,  # 1-4の評価
    "hidden_sizes": [256],
    "weight": 1.0
}
```

## 🔧 高度な設定

### ハイパーパラメータの調整

`train_multitask_model.py`の設定を変更：

```python
BATCH_SIZE = 4          # バッチサイズ
NUM_EPOCHS = 5          # エポック数
LEARNING_RATE = 1e-5    # 学習率
CHUNK_LEN = 512         # チャンク長
MAX_CHUNKS = 5          # 最大チャンク数
```

### タスクの重み調整

各タスクの`weight`を調整して、タスク間のバランスを制御：

```python
TASK_CONFIGS = [
    {"name": "sentiment", "weight": 0.3, ...},  # 軽め
    {"name": "course_score", "weight": 0.7, ...}  # 重め
]
```

### ネットワーク構造のカスタマイズ

`hidden_sizes`で多層ネットワークを定義：

```python
{
    "name": "complex_task",
    "hidden_sizes": [512, 256, 128],  # 3層
    ...
}
```

## 📝 使用例

### 例1: 感情スコア + 授業評価スコア

```python
TASK_CONFIGS = [
    {
        "name": "sentiment",
        "type": "regression",
        "output_size": 1,
        "hidden_sizes": [256],
        "weight": 0.5
    },
    {
        "name": "course_score",
        "type": "regression",
        "output_size": 1,
        "hidden_sizes": [256],
        "weight": 0.5
    }
]
```

### 例2: 感情 + 評価 + 満足度（順序回帰）

```python
TASK_CONFIGS = [
    {
        "name": "sentiment",
        "type": "regression",
        "output_size": 1,
        "weight": 0.3
    },
    {
        "name": "course_score",
        "type": "regression",
        "output_size": 1,
        "weight": 0.3
    },
    {
        "name": "satisfaction",
        "type": "ordinal",
        "output_size": 4,  # 1-4
        "weight": 0.4
    }
]
```

## 🎓 モデルの構造

```
入力テキスト
    ↓
BERTエンコーダー（共有）
    ↓
共有ドロップアウト
    ↓
┌─────────┬─────────┬─────────┐
│ Task 1  │ Task 2  │ Task 3  │
│ Head    │ Head    │ Head    │
└─────────┴─────────┴─────────┘
    ↓         ↓         ↓
 出力1      出力2      出力3
```

## 💾 モデルの保存

学習後、以下のファイルが保存されます：

- `02_モデル/マルチタスクモデル/multitask_model_YYYYMMDD_HHMMSS.pth`: モデル重み
- `02_モデル/マルチタスクモデル/multitask_model_YYYYMMDD_HHMMSS.json`: 学習結果と設定

## 🔍 トラブルシューティング

### メモリ不足

- `BATCH_SIZE`を減らす（2 → 1）
- `MAX_CHUNKS`を減らす（10 → 5）
- `CHUNK_LEN`を減らす（256 → 192）

### 学習が収束しない

- 学習率を調整（`LEARNING_RATE`）
- タスクの重みを調整（`weight`）
- エポック数を増やす（`NUM_EPOCHS`）

### データが見つからない

- CSVファイルのパスを確認
- `find_latest_agg_csv()`関数のパス候補を確認

## 📚 参考

- 既存の実装: `train_class_level_multitask.py`
- 順序回帰実装: `train_class_level_ordinal_llp.py`

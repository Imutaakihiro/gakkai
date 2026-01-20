# 卒業研究 - モデル訓練・評価 実行手順

## 📋 概要

このプロジェクトでは、授業評価の自由記述テキストから以下の3つのモデルを構築・評価します。

1. **単一タスクモデル1**: 自由記述 → 感情スコア（分類）
2. **単一タスクモデル2**: 自由記述 → 評価スコア（回帰）
3. **マルチタスクモデル**: 自由記述 → 感情スコア + 評価スコア

すべてのモデルは `koheiduck/bert-japanese-finetuned-sentiment` をベースとしています。

---

## 🚀 実行順序

### ステップ1: 単一タスクモデル1の評価

既存の感情スコアBERTモデルを評価します。

```bash
python evaluate_sentiment_model.py
```

**出力:**
- `03_分析結果/モデル評価/sentiment_model_evaluation_YYYYMMDD_HHMMSS.json`
- コンソールに性能指標（Accuracy, F1, Precision, Recall）が表示されます

---

### ステップ2: 単一タスクモデル2の訓練

評価スコア予測モデル（回帰タスク）を訓練します。

```bash
python train_score_model.py
```

**所要時間**: 約15-30分（GPU使用時）

**出力:**
- `02_モデル/単一タスクモデル2_評価スコア/best_model.pth`
- `02_モデル/単一タスクモデル2_評価スコア/model_config.json`
- `03_分析結果/モデル評価/score_model_training_YYYYMMDD_HHMMSS.json`

**ハイパーパラメータ（必要に応じて調整）:**
```python
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
MAX_LENGTH = 512
```

---

### ステップ3: マルチタスクモデルの訓練

感情スコアと評価スコアを同時に学習するマルチタスクモデルを訓練します。

```bash
python train_multitask_model.py
```

**所要時間**: 約20-40分（GPU使用時）

**出力:**
- `02_モデル/マルチタスクモデル/best_model.pth`
- `02_モデル/マルチタスクモデル/model_config.json`
- `03_分析結果/モデル評価/multitask_model_training_YYYYMMDD_HHMMSS.json`

**ハイパーパラメータ（必要に応じて調整）:**
```python
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
MAX_LENGTH = 512
ALPHA = 0.5  # 感情分析タスクの損失の重み
BETA = 0.5   # 評価スコアタスクの損失の重み
```

**重要**: `ALPHA` と `BETA` のバランスを調整することで、各タスクの重要度を変更できます。

---

### ステップ4: モデル性能比較

3つのモデルの性能を比較し、可視化します。

```bash
python compare_models.py
```

**前提条件**: ステップ1〜3が完了していること

**出力:**
- `03_分析結果/モデル比較/sentiment_comparison.png` - 感情分析性能の比較グラフ
- `03_分析結果/モデル比較/score_comparison.png` - 評価スコア予測性能の比較グラフ
- `03_分析結果/モデル比較/comparison_summary_YYYYMMDD_HHMMSS.json` - 比較結果（JSON）
- `03_分析結果/モデル比較/comparison_report.md` - 比較レポート（Markdown）

---

## 📊 評価指標

### 感情分析（分類タスク）

- **Accuracy**: 正解率
- **F1-Score**: 精度と再現率の調和平均
- **Precision**: 適合率
- **Recall**: 再現率

### 評価スコア予測（回帰タスク）

- **RMSE** (Root Mean Square Error): 二乗平均平方根誤差（低いほど良い）
- **MAE** (Mean Absolute Error): 平均絶対誤差（低いほど良い）
- **R²**: 決定係数（1に近いほど良い）
- **相関係数**: ピアソンの相関係数（1に近いほど良い）

---

## 🛠️ トラブルシューティング

### GPU が認識されない

```python
# 各スクリプトの冒頭で確認
import torch
print(torch.cuda.is_available())  # True なら GPU 使用可能
print(torch.cuda.get_device_name(0))  # GPU 名を表示
```

### メモリ不足エラー

`BATCH_SIZE` を小さくしてください（例: 16 → 8 → 4）

### 訓練が収束しない

- 学習率を調整: `LEARNING_RATE = 1e-5` または `3e-5`
- エポック数を増やす: `NUM_EPOCHS = 10`
- マルチタスクの場合、`ALPHA` と `BETA` のバランスを調整

---

## 📁 ディレクトリ構造

```
卒業研究（新）/
├── 01_データ/
│   ├── 自由記述→感情スコア/
│   │   ├── finetuning_train_20250710_220621.csv
│   │   └── finetuning_val_20250710_220621.csv
│   ├── 自由記述→評価スコアデータ/
│   │   ├── score_train_dataset.csv
│   │   └── score_val_dataset.csv
│   └── マルチタスク用データ/
│       └── マルチタスク学習用データセット_20250930_202839.csv
├── 02_モデル/
│   ├── 単一タスクモデル2_評価スコア/
│   └── マルチタスクモデル/
├── 03_分析結果/
│   ├── モデル評価/
│   └── モデル比較/
├── evaluate_sentiment_model.py
├── train_score_model.py
├── train_multitask_model.py
└── compare_models.py
```

---

## 📝 次のステップ（フェーズ3以降）

1. ✅ **フェーズ2完了**: 3つのモデルの構築と評価
2. 🔜 **フェーズ4**: SHAP分析の実装
3. 🔜 **フェーズ5**: 結果分析
4. 🔜 **フェーズ6**: 結果まとめ
5. 🔜 **フェーズ7**: 卒論執筆
6. 🔜 **フェーズ8**: プレゼン準備

---

## 🔧 環境要件

### 必要なパッケージ

```bash
pip install torch transformers pandas scikit-learn scipy matplotlib seaborn tqdm
```

### 推奨環境

- Python 3.8+
- CUDA 11.0+（GPU使用時）
- メモリ: 16GB以上
- GPU: NVIDIA GeForce GTX 1660 以上

---

## 📞 問い合わせ

実行中にエラーが発生した場合は、エラーメッセージとともにご相談ください。

**良い研究を！ 🎓**


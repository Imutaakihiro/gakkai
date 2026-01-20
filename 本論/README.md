# 本論ディレクトリ構成

本ディレクトリは、研究に必要な最小限のファイルのみを集約したクリーンな構成です。

## ディレクトリ構成

```
本論/
├── README.md                    # このファイル
├── 研究概要.md                  # 研究の概要（予稿.texから作成）
├── 予稿.tex                     # 学会予稿（LaTeX）
│
├── data/                        # 必要なデータのみ
│   ├── 授業集約データセット_*.csv    # マルチタスク学習用（3,268授業）
│   └── 手動ラベリング＿IDと自由記述とスコア.csv  # 教師データ（1,000件）
│
├── scripts/                     # 必要なスクリプトのみ
│   ├── train_class_level_multitask.py      # 授業レベルマルチタスク学習
│   ├── train_class_level_ordinal_llp.py    # 順序回帰モデル学習
│   ├── analyze_class_level_multitask_shap.py  # SHAP分析
│   └── analyze_ordinal_shap_production.py  # 順序回帰SHAP分析
│
├── models/                      # 学習済みモデル（必要に応じて）
│   └── best_model.pth
│
├── results/                     # 分析結果
│   ├── SHAP分析結果/
│   └── モデル評価結果/
│
└── figures/                     # 図表
    ├── test2.png               # モデル構造図
    └── SHAP可視化結果/
```

## 必要なファイルのコピー元

### データファイル
- `01_データ/マルチタスク用データ/授業集約データセット_*.csv`
- `01_データ/自由記述→感情スコア/手動ラベリング＿IDと自由記述とスコア.csv`

### スクリプト
- `00_スクリプト/train_class_level_multitask.py`
- `00_スクリプト/train_class_level_ordinal_llp.py`
- `00_スクリプト/analyze_class_level_multitask_shap.py`
- `00_スクリプト/analyze_ordinal_shap_production.py`

### 図表
- `03_分析結果/` から必要な図表を選択

## 実行手順

1. **データ準備**: `data/` ディレクトリに必要なCSVファイルを配置
2. **モデル学習**: `scripts/train_class_level_multitask.py` を実行
3. **SHAP分析**: `scripts/analyze_class_level_multitask_shap.py` を実行
4. **結果確認**: `results/` ディレクトリに出力される結果を確認

## 注意事項

- 本ディレクトリは研究の本質的な部分のみを含む
- 開発途中のファイルや実験的なスクリプトは含めない
- データファイルは必要最小限のもののみ

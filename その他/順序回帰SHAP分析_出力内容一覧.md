# 順序回帰SHAP分析 出力内容一覧

**作成日**: 2025年1月  
**スクリプト**: `00_スクリプト/analyze_ordinal_shap_production.py`  
**出力先**: `03_分析結果/順序回帰SHAP分析_本番用/`

---

## 📊 SHAP分析の種類（全7種類）

### 1. 感情スコアへのSHAP分析
- **出力ファイル**: `word_importance_sentiment_production.csv`
- **内容**: 感情スコア予測に寄与する単語の重要度
- **用途**: 感情分析の要因特定

### 2. 授業評価スコアへのSHAP分析
- **出力ファイル**: `word_importance_course_production.csv`
- **内容**: 授業評価スコア予測に寄与する単語の重要度
- **用途**: 授業評価の要因特定

### 3. 期待値E[y]へのSHAP分析 ⭐新規
- **出力ファイル**: `word_importance_expected_production.csv`
- **内容**: 期待値 E[y] = 1×P1 + 2×P2 + 3×P3 + 4×P4 への寄与
- **用途**: 総合的な評価スコアへの影響要因

### 4. P1（低評価確率）へのSHAP分析 ⭐新規
- **出力ファイル**: `word_importance_p1_production.csv`
- **内容**: 低評価（評価1）の確率への寄与
- **用途**: 低評価を引き起こす要因の特定

### 5. P2（中低評価確率）へのSHAP分析 ⭐新規
- **出力ファイル**: `word_importance_p2_production.csv`
- **内容**: 中低評価（評価2）の確率への寄与
- **用途**: 中低評価への影響要因の特定

### 6. P3（中高評価確率）へのSHAP分析 ⭐新規
- **出力ファイル**: `word_importance_p3_production.csv`
- **内容**: 中高評価（評価3）の確率への寄与
- **用途**: 中高評価への影響要因の特定

### 7. P4（高評価確率）へのSHAP分析 ⭐新規
- **出力ファイル**: `word_importance_p4_production.csv`
- **内容**: 高評価（評価4）の確率への寄与
- **用途**: 高評価を引き起こす要因の特定

---

## 📁 出力ファイル一覧

### CSVファイル（重要度データ）

#### 全データ版
- `word_importance_sentiment_production.csv` - 感情スコア重要度（全単語）
- `word_importance_course_production.csv` - 授業評価スコア重要度（全単語）
- `word_importance_expected_production.csv` - 期待値重要度（全単語）
- `word_importance_p1_production.csv` - P1重要度（全単語）
- `word_importance_p2_production.csv` - P2重要度（全単語）
- `word_importance_p3_production.csv` - P3重要度（全単語）
- `word_importance_p4_production.csv` - P4重要度（全単語）

#### TOP100版
- `word_importance_sentiment_top100_production.csv`
- `word_importance_course_top100_production.csv`
- `word_importance_expected_top100_production.csv`
- `word_importance_p1_top100_production.csv`
- `word_importance_p2_top100_production.csv`
- `word_importance_p3_top100_production.csv`
- `word_importance_p4_top100_production.csv`

### JSONファイル（構造化データ）

- `analysis_summary_production.json` - 分析結果のサマリー（JSON形式）
  - 完了した分析のリスト
  - 各分析の要因数
  - TOP20重要語
  - カテゴリ別要因数

- `factor_categories_production.json` - 要因のカテゴリ分類
  - 強い共通要因（感情スコアと授業評価スコアの両方に影響）
  - 感情寄り要因
  - 評価寄り要因
  - 感情特化要因
  - 評価特化要因

### マークダウンレポート

- `ordinal_shap_analysis_summary_production.md` - 分析結果の詳細レポート
  - 分析概要
  - 各分析の要因数
  - カテゴリ別要因数の詳細
  - 各評価段階（P1～P4）のTOP10重要語
  - 主要な発見
  - 授業改善への示唆

### 可視化ファイル（PNG）

#### 個別分析のTOP30グラフ
- `sentiment_top30_factors_production.png` - 感情スコアTOP30重要語
- `course_top30_factors_production.png` - 授業評価スコアTOP30重要語
- `expected_top30_factors_production.png` - 期待値TOP30重要語
- `p1_top30_factors_production.png` - P1（低評価確率）TOP30重要語
- `p2_top30_factors_production.png` - P2（中低評価確率）TOP30重要語
- `p3_top30_factors_production.png` - P3（中高評価確率）TOP30重要語
- `p4_top30_factors_production.png` - P4（高評価確率）TOP30重要語

#### 比較グラフ
- `p1_p2_p3_p4_comparison_production.png` - P1～P4の比較（全確率分布）
- `factor_categories_chart_production.png` - 要因カテゴリ別分布

---

## 🎯 各分析の特徴

### 感情スコア・授業評価スコア
- **既存のマルチタスク分析と同様**
- 感情と評価の両方への影響を分析

### 期待値E[y] ⭐新規
- **順序回帰モデル特有の分析**
- 総合的な評価スコア（1～4の期待値）への影響
- より解釈しやすい指標

### P1～P4（確率分布） ⭐新規
- **順序回帰モデル特有の分析**
- 各評価段階（1～4）での要因を詳細に分析
- 低評価を減らす要因、高評価を増やす要因を特定可能

---

## 📈 分析の活用方法

### 1. 低評価を減らす施策
- P1（低評価確率）の重要語を確認
- これらの要因を回避・改善する施策を検討

### 2. 高評価を増やす施策
- P4（高評価確率）の重要語を確認
- これらの要因を強化する施策を検討

### 3. 中評価から高評価への移行
- P2、P3の重要語を確認
- 中評価から高評価への移行要因を特定

### 4. 総合的な改善
- 期待値E[y]の重要語を確認
- 総合的な評価スコア向上の要因を特定

---

## 🔍 マルチタスク分析との違い

### マルチタスク分析
- 感情スコアへのSHAP
- 授業評価スコアへのSHAP
- **限界**: 評価スコアが連続値のため、段階別の要因分析が困難

### 順序回帰分析（今回）
- 感情スコアへのSHAP
- 授業評価スコアへのSHAP
- **期待値E[y]へのSHAP** ← 新規
- **P1～P4へのSHAP** ← 新規（各評価段階での要因分析）

---

**最終更新**: 2025年1月


"""
SHAP分析: 単一タスクモデル1（感情スコア）
検証データ200件で感情判定に寄与する重要語を特定
"""

import pandas as pd
import numpy as np
import torch
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
import shap
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# モデルとトークナイザーのロード
MODEL_PATH = r"C:\Users\takahashi.DESKTOP-U0T5SUB\Downloads\BERT\git_excluded\finetuned_bert_model_20250718_step2_fixed_classweights_variant1_positive重点強化"
print(f"モデルをロード中: {MODEL_PATH}")

tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# 検証データの読み込み
VAL_DATA_PATH = r"../01_データ\自由記述→感情スコア\finetuning_val_20250710_220621.csv"
print(f"検証データをロード中: {VAL_DATA_PATH}")
val_df = pd.read_csv(VAL_DATA_PATH)

print(f"検証データサイズ: {len(val_df)}")
print(f"ラベル分布:\n{val_df['label'].value_counts()}")

# ラベルマッピング（0, 1, 2 → ネガティブ, ニュートラル, ポジティブ）
label_names = {0: "ネガティブ", 1: "ニュートラル", 2: "ポジティブ"}

# オプション設定
RUN_SAMPLE_5000 = False  # 追加検証: 8万件から5,000件サンプリング（集計のみ）
SAMPLE_SIZE = 5000
RAW_TEXT_PATH = r"../01_データ\元データ\(CSV用)前処理後データ_free_text_only.csv"  # ID,自由記述

# 予測関数（SHAP用）
def predict_proba(texts):
    """テキストのリストを受け取り、クラス確率を返す"""
    # SHAPから渡されるデータ型を処理
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    elif not isinstance(texts, list):
        # その他の型の場合、リストに変換を試みる
        try:
            texts = list(texts)
        except:
            texts = [str(texts)]
    
    # 空文字列や無効な入力を処理
    texts = [str(t) if t else "" for t in texts]
    
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    return probs.cpu().numpy()

# テスト実行（小規模）
print("\n" + "="*60)
print("テスト実行: 最初の5件で動作確認")
print("="*60)

test_texts = val_df['text'].head(5).tolist()
print(f"\nテストテキスト数: {len(test_texts)}")

# 予測のテスト
test_probs = predict_proba(test_texts)
print(f"予測確率の形状: {test_probs.shape}")
print(f"最初のテキストの予測: {test_probs[0]}")
print(f"予測クラス: {label_names[test_probs[0].argmax()]}")

# SHAP分析の準備
print("\n" + "="*60)
print("SHAP Explainerの作成")
print("="*60)

# Partition Explainerを使用（より安定）
print("Explainerを初期化中...")
# トークナイザーベースのマスカーではなく、シンプルなアプローチを使用
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict_proba, masker, algorithm="partition")

# 小規模テスト（最初の10件）
print("\n" + "="*60)
print("小規模SHAP分析（10件）")
print("="*60)

small_texts = val_df['text'].head(10).tolist()
print("SHAP値を計算中...")
small_shap_values = explainer(small_texts)

print(f"SHAP値の形状: {small_shap_values.shape}")
print("✓ 小規模テスト成功！")

# 本番実行（全200件）
print("\n" + "="*60)
print("本番SHAP分析（全200件）")
print("="*60)

all_texts = val_df['text'].tolist()
all_labels = val_df['label'].astype(int).tolist()

print(f"分析対象: {len(all_texts)}件")
print("SHAP値を計算中（時間がかかります）...")

# バッチ処理でSHAP分析（メモリ対策）
batch_size = 20
all_shap_values = []

for i in tqdm(range(0, len(all_texts), batch_size), desc="SHAP分析"):
    batch_texts = all_texts[i:i+batch_size]
    batch_shap = explainer(batch_texts)
    all_shap_values.append(batch_shap)

print("✓ SHAP分析完了！")

# 結果の集計
print("\n" + "="*60)
print("結果の集計")
print("="*60)

def merge_wordpieces(tokens, shap_vals_pos):
    """WordPieceのサブワード（##）を前の語に結合して集約する。
    戻り値: (merged_tokens, merged_shap_vals)
    """
    merged_tokens = []
    merged_vals = []
    current = ''
    current_val = 0.0
    for tok, val in zip(tokens, shap_vals_pos):
        t = str(tok)
        # 特殊トークンはスキップ
        if t in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
            continue
        if t.startswith('##'):
            # 連結（接頭の##を除去して前語に追加）
            current += t[2:]
            current_val += float(val)
        else:
            # 直前の語を確定
            if current:
                merged_tokens.append(current)
                merged_vals.append(current_val)
            current = t
            current_val = float(val)
    if current:
        merged_tokens.append(current)
        merged_vals.append(current_val)
    return merged_tokens, merged_vals

# クラス別にSHAP値を集計しつつ、語単位へサブワード結合
print("\n単語ごとの重要度を集計中（サブワード統合あり）...")
word_importance = defaultdict(lambda: {'shap_values': [], 'count': 0, 'class_dist': {0: 0, 1: 0, 2: 0}})

for idx, (text, label) in enumerate(zip(all_texts, all_labels)):
    batch_idx = idx // batch_size
    within_idx = idx % batch_size
    if within_idx >= len(all_shap_values[batch_idx]):
        continue
    sv = all_shap_values[batch_idx][within_idx]
    tokens = sv.data
    shap_vals = sv.values
    # ポジティブクラス（index=2）のSHAP値を使用
    if len(shap_vals.shape) > 1:
        shap_vals_pos = shap_vals[:, 2]
    else:
        shap_vals_pos = shap_vals
    merged_tokens, merged_vals = merge_wordpieces(tokens, shap_vals_pos)
    for m_tok, m_val in zip(merged_tokens, merged_vals):
        if not m_tok:
            continue
        word_importance[m_tok]['shap_values'].append(float(m_val))
        word_importance[m_tok]['count'] += 1
        word_importance[m_tok]['class_dist'][label] += 1

# 平均SHAP値を計算
word_stats = {}
for word, data in word_importance.items():
    if data['count'] >= 3:  # 3回以上出現する単語のみ
        word_stats[word] = {
            'mean_shap': np.mean(data['shap_values']),
            'abs_mean_shap': np.mean(np.abs(data['shap_values'])),
            'std_shap': np.std(data['shap_values']),
            'count': data['count'],
            'class_dist': data['class_dist']
        }

# DataFrame化
df_importance = pd.DataFrame(word_stats).T
df_importance = df_importance.sort_values('mean_shap', ascending=False)

print(f"\n分析対象単語数: {len(df_importance)}")

# 結果の表示
print("\n" + "="*60)
print("ポジティブ判定に最も寄与する単語 TOP20")
print("="*60)
print(df_importance.head(20)[['mean_shap', 'abs_mean_shap', 'count']].to_string())

print("\n" + "="*60)
print("ネガティブ判定に最も寄与する単語 TOP20")
print("="*60)
print(df_importance.tail(20)[['mean_shap', 'abs_mean_shap', 'count']].to_string())

# 結果を保存
print("\n" + "="*60)
print("結果を保存中")
print("="*60)

output_dir = "../03_分析結果/SHAP分析/検証データ200件"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/可視化", exist_ok=True)
# 個別事例の出力設定（不要なら False）
SAVE_INDIVIDUAL = False

# 1. 重要語のJSONとして保存
importance_data = {
    "analysis_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "dataset_size": len(all_texts),
    "model_path": MODEL_PATH,
    "top_positive_words": df_importance.head(20).to_dict('index'),
    "top_negative_words": df_importance.tail(20).to_dict('index'),
    "all_words": df_importance.to_dict('index')
}

with open(f"{output_dir}/global_importance.json", 'w', encoding='utf-8') as f:
    json.dump(importance_data, f, ensure_ascii=False, indent=2)

print(f"✓ 重要語データを保存: {output_dir}/global_importance.json")

# 2. CSVとしても保存（Excelで開ける）
df_importance.to_csv(f"{output_dir}/word_importance.csv", encoding='utf-8-sig')
print(f"✓ CSV保存: {output_dir}/word_importance.csv")

# 3. 可視化: ポジティブTOP20
fig, ax = plt.subplots(figsize=(10, 8))
top20_pos = df_importance.head(20)
ax.barh(range(len(top20_pos)), top20_pos['mean_shap'], color='green', alpha=0.7)
ax.set_yticks(range(len(top20_pos)))
ax.set_yticklabels(top20_pos.index)
ax.set_xlabel('平均SHAP値（ポジティブ方向）')
ax.set_title('ポジティブ判定に最も寄与する単語 TOP20')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f"{output_dir}/可視化/top20_positive.png", dpi=300, bbox_inches='tight')
print(f"✓ グラフ保存: {output_dir}/可視化/top20_positive.png")
plt.close()

# 4. 可視化: ネガティブTOP20
fig, ax = plt.subplots(figsize=(10, 8))
top20_neg = df_importance.tail(20).sort_values('mean_shap')
ax.barh(range(len(top20_neg)), top20_neg['mean_shap'], color='red', alpha=0.7)
ax.set_yticks(range(len(top20_neg)))
ax.set_yticklabels(top20_neg.index)
ax.set_xlabel('平均SHAP値（ネガティブ方向）')
ax.set_title('ネガティブ判定に最も寄与する単語 TOP20')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f"{output_dir}/可視化/top20_negative.png", dpi=300, bbox_inches='tight')
print(f"✓ グラフ保存: {output_dir}/可視化/top20_negative.png")
plt.close()

# 5. 個別事例の可視化（各クラス2件ずつ）
if SAVE_INDIVIDUAL:
    print("\n個別事例を可視化中...")
    os.makedirs(f"{output_dir}/個別事例", exist_ok=True)

    # ポジティブ事例
    pos_indices = val_df[val_df['label'] == 2].head(2).index.tolist()
    for i, idx in enumerate(pos_indices):
        text_idx = list(val_df.index).index(idx)
        batch_idx = text_idx // batch_size
        within_idx = text_idx % batch_size
        
        if within_idx < len(all_shap_values[batch_idx]):
            shap.plots.text(all_shap_values[batch_idx][within_idx], display=False)
            plt.savefig(f"{output_dir}/個別事例/positive_example_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ ポジティブ事例{i+1}を保存")

    # ネガティブ事例
    neg_indices = val_df[val_df['label'] == 0].head(2).index.tolist()
    for i, idx in enumerate(neg_indices):
        text_idx = list(val_df.index).index(idx)
        batch_idx = text_idx // batch_size
        within_idx = text_idx % batch_size
        
        if within_idx < len(all_shap_values[batch_idx]):
            shap.plots.text(all_shap_values[batch_idx][within_idx], display=False)
            plt.savefig(f"{output_dir}/個別事例/negative_example_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ ネガティブ事例{i+1}を保存")

    # ニュートラル事例
    neu_indices = val_df[val_df['label'] == 1].head(2).index.tolist()
    for i, idx in enumerate(neu_indices):
        text_idx = list(val_df.index).index(idx)
        batch_idx = text_idx // batch_size
        within_idx = text_idx % batch_size
        
        if within_idx < len(all_shap_values[batch_idx]):
            shap.plots.text(all_shap_values[batch_idx][within_idx], display=False)
            plt.savefig(f"{output_dir}/個別事例/neutral_example_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ ニュートラル事例{i+1}を保存")

# サマリーレポートの作成
print("\n" + "="*60)
print("サマリーレポートを作成中")
print("="*60)

summary_report = f"""# SHAP分析サマリー（検証データ200件）

**分析日時:** {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}  
**対象データ:** 検証データ200件（モデルが見ていないデータ）  
**モデル:** 単一タスクモデル1（感情スコア）

---

## 📊 データ概要

- **総サンプル数:** {len(val_df)}件
- **クラス分布:**
  - ネガティブ: {(val_df['label']==0).sum()}件
  - ニュートラル: {(val_df['label']==1).sum()}件
  - ポジティブ: {(val_df['label']==2).sum()}件

---

## 🔝 ポジティブ判定に寄与する重要語 TOP20

| 順位 | 単語 | 平均SHAP値 | 出現回数 |
|------|------|-----------|---------|
{chr(10).join([f"| {i+1} | {word} | {row['mean_shap']:.4f} | {int(row['count'])} |" for i, (word, row) in enumerate(df_importance.head(20).iterrows())])}

---

## 🔻 ネガティブ判定に寄与する重要語 TOP20

| 順位 | 単語 | 平均SHAP値 | 出現回数 |
|------|------|-----------|---------|
{chr(10).join([f"| {i+1} | {word} | {row['mean_shap']:.4f} | {int(row['count'])} |" for i, (word, row) in enumerate(df_importance.tail(20).sort_values('mean_shap').iterrows())])}

---

## 📁 生成ファイル

- `global_importance.json` - 全単語の重要度データ
- `word_importance.csv` - Excel用CSV
- `可視化/top20_positive.png` - ポジティブTOP20グラフ
- `可視化/top20_negative.png` - ネガティブTOP20グラフ
- `個別事例/*.png` - 個別テキストのSHAP可視化

---

## 💡 主要な発見

### 満足度を高める要因（ポジティブ）
{chr(10).join([f"- **{word}**: {row['mean_shap']:.4f}（{int(row['count'])}回出現）" for word, row in df_importance.head(5).iterrows()])}

### 不満の原因（ネガティブ）
{chr(10).join([f"- **{word}**: {row['mean_shap']:.4f}（{int(row['count'])}回出現）" for word, row in df_importance.tail(5).sort_values('mean_shap').iterrows()])}

---

**次のステップ:**
- クラス別の詳細分析
- 誤分類事例の分析
- 教員へのフィードバック作成
"""

with open(f"{output_dir}/SHAP分析サマリー.md", 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(f"✓ サマリーレポートを保存: {output_dir}/SHAP分析サマリー.md")

print("\n" + "="*60)
print("SHAP分析完了！")
print("="*60)
print(f"\n結果は以下に保存されました:")
print(f"  {output_dir}/")
print("\n次のステップ:")
print("  1. 生成されたグラフを確認")
print("  2. SHAP分析サマリー.mdを確認")
print("  3. 必要に応じて追加分析（5,000件サンプリング）")

# 追加検証: 8万件から5,000件サンプリングで集計のみ実施
if RUN_SAMPLE_5000:
    print("\n" + "="*60)
    print("追加検証: 8万件から5,000件サンプリングでSHAP集計（可視化なし）")
    print("="*60)
    try:
        raw_df = pd.read_csv(RAW_TEXT_PATH)
        # 列名推定: 先頭列がID、2列目が自由記述を想定
        if '自由記述' in raw_df.columns:
            texts_all = raw_df['自由記述'].dropna().astype(str)
        elif 'text' in raw_df.columns:
            texts_all = raw_df['text'].dropna().astype(str)
        else:
            # 2列目を自由記述と仮定
            texts_all = raw_df.iloc[:, 1].dropna().astype(str)

        n = min(SAMPLE_SIZE, len(texts_all))
        sample_texts = texts_all.sample(n=n, random_state=42).tolist()
        print(f"サンプルサイズ: {len(sample_texts)} / 総件数: {len(texts_all)}")

        # ストリーム集計（語単位統合）
        word_importance_sample = defaultdict(lambda: {'shap_values': [], 'count': 0})
        batch_size_sample = 64
        for i in tqdm(range(0, len(sample_texts), batch_size_sample), desc="SHAP集計(5k)"):
            bt = sample_texts[i:i+batch_size_sample]
            sv_batch = explainer(bt)
            for sv in sv_batch:
                tokens = sv.data
                vals = sv.values
                if len(vals.shape) > 1:
                    vals_pos = vals[:, 2]
                else:
                    vals_pos = vals
                m_toks, m_vals = merge_wordpieces(tokens, vals_pos)
                for t, v in zip(m_toks, m_vals):
                    if not t:
                        continue
                    word_importance_sample[t]['shap_values'].append(float(v))
                    word_importance_sample[t]['count'] += 1

        # DataFrame化
        word_stats_sample = {
            w: {
                'mean_shap': float(np.mean(d['shap_values'])),
                'abs_mean_shap': float(np.mean(np.abs(d['shap_values']))),
                'std_shap': float(np.std(d['shap_values'])),
                'count': int(d['count'])
            }
            for w, d in word_importance_sample.items() if d['count'] >= 5
        }
        df_sample = pd.DataFrame(word_stats_sample).T.sort_values('mean_shap', ascending=False)

        out_dir_sample = "../03_分析結果/SHAP分析/サンプリング5000件"
        os.makedirs(out_dir_sample, exist_ok=True)
        df_sample.to_csv(f"{out_dir_sample}/word_importance_sample5000.csv", encoding='utf-8-sig')
        with open(f"{out_dir_sample}/global_importance_sample5000.json", 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_date': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'dataset_size': len(sample_texts),
                'model_path': MODEL_PATH,
                'top_positive_words': df_sample.head(50).to_dict('index'),
                'top_negative_words': df_sample.tail(50).to_dict('index')
            }, f, ensure_ascii=False, indent=2)
        print(f"✓ サンプル集計を保存: {out_dir_sample}/word_importance_sample5000.csv")
        print(f"✓ サンプル集計を保存: {out_dir_sample}/global_importance_sample5000.json")
    except Exception as e:
        print(f"追加検証でエラー: {e}")


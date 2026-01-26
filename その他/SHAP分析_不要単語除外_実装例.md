# SHAP分析 不要単語除外機能 実装例

**目的**: SHAP分析時に助詞・格助詞などの不要な単語を除外する機能の実装

---

## 実装方針

1. **日本語の助詞・格助詞リストの作成**
   - BERT用トークナイザー（`transformers`の`AutoTokenizer`）と整合性を取る
   - 単語IDベースで除外する

2. **SHAP値のフィルタリング**
   - SHAP値の計算後に、不要な単語のSHAP値を0に設定または除外
   - 重要語ランキング作成時に除外

---

## 実装コード例

### 1. 助詞・格助詞リストの作成

```python
import re
from transformers import AutoTokenizer

def create_stopword_ids(tokenizer, model_name="koheiduck/bert-japanese-finetuned-sentiment"):
    """
    日本語の助詞・格助詞の単語IDリストを作成
    
    Args:
        tokenizer: トークナイザー
        model_name: モデル名
    
    Returns:
        stopword_ids: 除外する単語IDのセット
    """
    # 日本語の助詞・格助詞のリスト
    stopwords = [
        # 格助詞
        'が', 'の', 'を', 'に', 'へ', 'と', 'から', 'より', 'で', 'まで',
        # 係助詞
        'は', 'も', 'こそ', 'さえ', 'でも', 'だって',
        # 副助詞
        'ばかり', 'だけ', 'のみ', 'まで', 'ほど', 'くらい', 'ぐらい',
        # 終助詞
        'か', 'な', 'ね', 'よ', 'ぞ', 'ぜ', 'わ', 'さ',
        # 接続助詞
        'て', 'で', 'ながら', 'つつ', 'し', 'が', 'けれど', 'のに',
        # その他
        'の', 'こと', 'もの', 'ため', 'とき', 'ところ',
    ]
    
    stopword_ids = set()
    
    # トークナイザーでエンコードしてIDを取得
    for word in stopwords:
        # 単語をトークン化
        tokens = tokenizer.tokenize(word)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # 各トークンIDを追加
        for token_id in token_ids:
            stopword_ids.add(token_id)
    
    # 特殊トークンも除外（必要に応じて）
    special_tokens = [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ]
    for token_id in special_tokens:
        if token_id is not None:
            stopword_ids.add(token_id)
    
    return stopword_ids
```

### 2. SHAP値のフィルタリング関数

```python
import numpy as np

def filter_shap_values(shap_values, token_ids, stopword_ids, method='zero'):
    """
    SHAP値から不要な単語を除外
    
    Args:
        shap_values: SHAP値の配列 (shape: [n_samples, n_tokens])
        token_ids: トークンIDの配列 (shape: [n_samples, n_tokens])
        stopword_ids: 除外する単語IDのセット
        method: 除外方法 ('zero': SHAP値を0に設定, 'mask': マスクして除外)
    
    Returns:
        filtered_shap_values: フィルタリングされたSHAP値
        mask: マスク配列（method='mask'の場合）
    """
    filtered_shap_values = shap_values.copy()
    mask = np.ones_like(shap_values, dtype=bool)
    
    for i in range(shap_values.shape[0]):
        for j in range(shap_values.shape[1]):
            if token_ids[i, j] in stopword_ids:
                if method == 'zero':
                    filtered_shap_values[i, j] = 0.0
                elif method == 'mask':
                    mask[i, j] = False
    
    if method == 'mask':
        return filtered_shap_values, mask
    else:
        return filtered_shap_values
```

### 3. 重要語ランキング作成時の除外

```python
def create_word_importance_ranking(shap_values, token_ids, tokenizer, stopword_ids, top_k=30):
    """
    重要語ランキングを作成（不要な単語を除外）
    
    Args:
        shap_values: SHAP値の配列
        token_ids: トークンIDの配列
        tokenizer: トークナイザー
        stopword_ids: 除外する単語IDのセット
        top_k: 上位k個の重要語を取得
    
    Returns:
        word_importance: 重要語とそのSHAP値のリスト
    """
    # SHAP値を絶対値で集約（全サンプルで平均）
    aggregated_shap = np.abs(shap_values).mean(axis=0)
    
    # 重要語とSHAP値のペアを作成
    word_importance = []
    
    for i, (token_id, shap_val) in enumerate(zip(token_ids[0], aggregated_shap)):
        # 不要な単語をスキップ
        if token_id in stopword_ids:
            continue
        
        # トークンを文字列に変換
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        
        # 特殊トークンをスキップ
        if token.startswith('[') and token.endswith(']'):
            continue
        
        word_importance.append({
            'token_id': int(token_id),
            'token': token,
            'shap_value': float(shap_val),
            'position': i
        })
    
    # SHAP値でソート
    word_importance.sort(key=lambda x: x['shap_value'], reverse=True)
    
    # 上位k個を返す
    return word_importance[:top_k]
```

### 4. 完全な実装例（analyze_ordinal_shap_production.pyへの統合）

```python
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import shap
from tqdm import tqdm

# モデルとトークナイザーの読み込み
model_name = "koheiduck/bert-japanese-finetuned-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 不要な単語IDの取得
stopword_ids = create_stopword_ids(tokenizer, model_name)

# SHAP分析の実行
def analyze_shap_with_filter(model, data_loader, target='p2', n_samples=1000):
    """
    SHAP分析を実行（不要な単語を除外）
    
    Args:
        model: 学習済みモデル
        data_loader: データローダー
        target: 分析対象 ('p2' または 'p4')
        n_samples: 分析サンプル数
    
    Returns:
        word_importance: 重要語ランキング
    """
    # バックグラウンドデータの準備
    background_data = []
    for batch in data_loader:
        background_data.append(batch['input_ids'])
        if len(background_data) >= 100:  # バックグラウンドは100サンプル
            break
    
    background = torch.cat(background_data, dim=0)
    
    # SHAP Explainerの作成
    def model_wrapper(x):
        """モデルのラッパー関数"""
        with torch.no_grad():
            outputs = model(x)
            if target == 'p2':
                return outputs['p2'].cpu().numpy()
            elif target == 'p4':
                return outputs['p4'].cpu().numpy()
    
    explainer = shap.Explainer(model_wrapper, background)
    
    # 分析対象データの準備
    analysis_data = []
    for batch in data_loader:
        analysis_data.append(batch['input_ids'])
        if len(analysis_data) >= n_samples:
            break
    
    analysis_input = torch.cat(analysis_data, dim=0)
    
    # SHAP値の計算
    shap_values = explainer(analysis_input)
    
    # 不要な単語を除外
    filtered_shap_values = filter_shap_values(
        shap_values.values,
        analysis_input.cpu().numpy(),
        stopword_ids,
        method='zero'
    )
    
    # 重要語ランキングの作成
    word_importance = create_word_importance_ranking(
        filtered_shap_values,
        analysis_input.cpu().numpy(),
        tokenizer,
        stopword_ids,
        top_k=30
    )
    
    return word_importance, filtered_shap_values

# 実行例
word_importance_p2, shap_values_p2 = analyze_shap_with_filter(
    model, 
    test_loader, 
    target='p2', 
    n_samples=1000
)

# CSV出力
df_p2 = pd.DataFrame(word_importance_p2)
df_p2.to_csv('word_importance_p2_production.csv', index=False, encoding='utf-8-sig')
```

---

## テスト方法

### 1. 除外機能のテスト

```python
def test_stopword_filtering():
    """除外機能のテスト"""
    tokenizer = AutoTokenizer.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
    stopword_ids = create_stopword_ids(tokenizer)
    
    # テスト用のSHAP値とトークンID
    test_shap = np.random.randn(10, 128)  # 10サンプル、128トークン
    test_token_ids = np.random.randint(0, 30000, (10, 128))
    
    # フィルタリング前の統計
    print(f"フィルタリング前のSHAP値の平均: {np.abs(test_shap).mean()}")
    
    # フィルタリング
    filtered_shap = filter_shap_values(test_shap, test_token_ids, stopword_ids)
    
    # フィルタリング後の統計
    print(f"フィルタリング後のSHAP値の平均: {np.abs(filtered_shap).mean()}")
    
    # 除外されたトークン数を確認
    excluded_count = 0
    for i in range(test_token_ids.shape[0]):
        for j in range(test_token_ids.shape[1]):
            if test_token_ids[i, j] in stopword_ids:
                excluded_count += 1
    
    print(f"除外されたトークン数: {excluded_count}")
    
    return filtered_shap

# テスト実行
test_stopword_filtering()
```

### 2. 重要語ランキングの確認

```python
def test_word_importance_ranking():
    """重要語ランキングのテスト"""
    tokenizer = AutoTokenizer.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
    stopword_ids = create_stopword_ids(tokenizer)
    
    # テスト用データ
    test_shap = np.random.randn(1, 128)
    test_token_ids = np.random.randint(0, 30000, (1, 128))
    
    # 重要語ランキングの作成
    word_importance = create_word_importance_ranking(
        test_shap,
        test_token_ids,
        tokenizer,
        stopword_ids,
        top_k=10
    )
    
    # 結果の表示
    print("重要語ランキング（上位10位）:")
    for i, item in enumerate(word_importance, 1):
        print(f"{i}. {item['token']}: {item['shap_value']:.4f}")
    
    return word_importance

# テスト実行
test_word_importance_ranking()
```

---

## 注意事項

1. **トークナイザーとの整合性**
   - 使用するトークナイザーとモデルが一致していることを確認
   - サブワードトークン化の影響を考慮

2. **除外対象の調整**
   - 助詞・格助詞のリストは必要に応じて調整
   - 研究の目的に応じて、除外対象を変更可能

3. **パフォーマンス**
   - 大量のデータを処理する場合、除外処理の効率化を検討
   - バッチ処理で効率化

4. **検証**
   - 除外前後の比較を実施
   - 重要語ランキングの変化を確認

---

## 参考資料

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [日本語の助詞・格助詞リスト](https://ja.wikipedia.org/wiki/助詞)

---

**作成日**: 2026年1月26日

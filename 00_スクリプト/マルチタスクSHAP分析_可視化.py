import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定（確実版）
def setup_japanese_font():
    """日本語フォントの設定"""
    import matplotlib.font_manager as fm
    
    # Windows環境での日本語フォント設定
    try:
        # 利用可能なフォントを取得
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        print("利用可能なフォント:", available_fonts[:10])  # 最初の10個を表示
        
        # 日本語フォントの優先順位
        japanese_fonts = [
            'MS Gothic', 'MS Mincho', 'Yu Gothic', 'Meiryo', 'Hiragino Sans',
            'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP',
            'DejaVu Sans', 'Arial Unicode MS'
        ]
        
        # 利用可能な日本語フォントを探す
        for font in japanese_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 日本語フォント設定完了: {font}")
                return True
        
        # フォールバック設定
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        print("⚠️ フォールバックフォント設定")
        return False
        
    except Exception as e:
        print(f"❌ フォント設定エラー: {e}")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        return False

# フォント設定実行
font_success = setup_japanese_font()

# 追加のフォント設定
if not font_success:
    print("🔧 追加のフォント設定を試行中...")
    try:
        # Windows標準フォントを直接指定
        plt.rcParams['font.family'] = ['MS Gothic', 'MS Mincho', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ 追加設定完了")
    except:
        print("❌ 追加設定も失敗")

# グラフのスタイル設定
plt.style.use('default')  # seaborn-v0_8を削除してデフォルトに
sns.set_palette("husl")

# フォントキャッシュのクリア
try:
    font_manager._rebuild()
    print("🔄 フォントキャッシュをクリアしました")
except:
    print("⚠️ フォントキャッシュのクリアに失敗")

# データの準備
if font_success:
    # 日本語ラベル
    data = {
        'カテゴリ': ['共通要因', '感情特化', '評価特化', '低重要度'],
        '語彙数': [577, 1200, 532, 889],
        '割合': [19.4, 40.3, 17.9, 22.4],
        '平均感情重要度': [0.000727, 0.000770, 0.000302, 0.000313],
        '平均評価重要度': [0.000695, 0.000289, 0.000707, 0.000298],
        '平均統合重要度': [0.001422, 0.001059, 0.001009, 0.000610]
    }
    print("✅ 日本語ラベルで実行")
else:
    # 英語ラベル（フォールバック）
    data = {
        'カテゴリ': ['Common Factors', 'Sentiment Specific', 'Evaluation Specific', 'Low Importance'],
        '語彙数': [577, 1200, 532, 889],
        '割合': [19.4, 40.3, 17.9, 22.4],
        '平均感情重要度': [0.000727, 0.000770, 0.000302, 0.000313],
        '平均評価重要度': [0.000695, 0.000289, 0.000707, 0.000298],
        '平均統合重要度': [0.001422, 0.001059, 0.001009, 0.000610]
    }
    print("⚠️ 英語ラベルで実行（フォールバック）")

df = pd.DataFrame(data)

# 色の設定（改善版）
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # より鮮やかで見やすい色
colors_light = ['#F1948A', '#85C1E9', '#82E0AA', '#F7DC6F']  # 薄い色
colors_gradient = ['#FF4757', '#3742FA', '#2ED573', '#FFA502']  # グラデーション色

# 図のサイズ設定
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('white')

# タイトルの設定
if font_success:
    title1 = 'カテゴリ別語彙数分布'
    title2 = 'カテゴリ別割合分布'
    title3 = 'カテゴリ別重要度比較'
    title4 = '共通要因TOP10'
    title5 = '感情特化要因TOP10'
    title6 = '評価特化要因TOP10'
    xlabel = 'カテゴリ'
    ylabel = '重要度'
    hlabel = '統合重要度'
    legend1 = '感情重要度'
    legend2 = '評価重要度'
    legend3 = '統合重要度'
else:
    title1 = 'Vocabulary Distribution by Category'
    title2 = 'Percentage Distribution by Category'
    title3 = 'Importance Comparison by Category'
    title4 = 'Top 10 Common Factors'
    title5 = 'Top 10 Sentiment-Specific Factors'
    title6 = 'Top 10 Evaluation-Specific Factors'
    xlabel = 'Category'
    ylabel = 'Importance'
    hlabel = 'Total Importance'
    legend1 = 'Sentiment Importance'
    legend2 = 'Evaluation Importance'
    legend3 = 'Total Importance'

# 1. カテゴリ別語彙数の円グラフ
plt.subplot(2, 3, 1)
wedges, texts, autotexts = plt.pie(df['語彙数'], labels=df['カテゴリ'], autopct='%1.1f%%', 
                                  colors=colors, startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
plt.title(title1, fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
plt.axis('equal')

# 2. カテゴリ別割合の円グラフ
plt.subplot(2, 3, 2)
wedges, texts, autotexts = plt.pie(df['割合'], labels=df['カテゴリ'], autopct='%1.1f%%', 
                                  colors=colors_gradient, startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
plt.title(title2, fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
plt.axis('equal')

# 3. 重要度の比較（棒グラフ）
plt.subplot(2, 3, 3)
x = np.arange(len(df['カテゴリ']))
width = 0.25

bars1 = plt.bar(x - width, df['平均感情重要度'], width, label=legend1, 
                color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = plt.bar(x, df['平均評価重要度'], width, label=legend2, 
                color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.5)
bars3 = plt.bar(x + width, df['平均統合重要度'], width, label=legend3, 
                color=colors[2], alpha=0.8, edgecolor='black', linewidth=0.5)

plt.xlabel(xlabel, fontsize=12, fontweight='bold')
plt.ylabel(ylabel, fontsize=12, fontweight='bold')
plt.title(title3, fontsize=16, fontweight='bold', color='#2C3E50')
plt.xticks(x, df['カテゴリ'], rotation=45, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

# 4. 共通要因TOP10の重要度（横棒グラフ）
plt.subplot(2, 3, 4)
common_factors = {
    '学ぶ': 0.002664,
    'myit': 0.002468,
    'まま': 0.002404,
    '電動': 0.002369,
    'すぐ': 0.002283,
    'より': 0.002239,
    '単語語': 0.002198,
    '方式': 0.002111,
    'れる': 0.001983,
    '下さい': 0.001974
}

words = list(common_factors.keys())
values = list(common_factors.values())

bars = plt.barh(words, values, color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)
plt.xlabel(hlabel, fontsize=12, fontweight='bold')
plt.title(title4, fontsize=16, fontweight='bold', color='#2C3E50')
plt.grid(True, alpha=0.3, linestyle='--')

# 5. 感情特化要因TOP10の重要度（横棒グラフ）
plt.subplot(2, 3, 5)
sentiment_factors = {
    '代わり': 0.001951,
    '持っけ': 0.001934,
    '忘れ物': 0.001823,
    'ます書': 0.001808,
    '焦点': 0.001768,
    '組む': 0.001760,
    '素子': 0.001745,
    '英語べ': 0.001732,
    '前回': 0.001726,
    '入力': 0.001722
}

words_sent = list(sentiment_factors.keys())
values_sent = list(sentiment_factors.values())

bars = plt.barh(words_sent, values_sent, color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.5)
plt.xlabel(hlabel, fontsize=12, fontweight='bold')
plt.title(title5, fontsize=16, fontweight='bold', color='#2C3E50')
plt.grid(True, alpha=0.3, linestyle='--')

# 6. 評価特化要因TOP10の重要度（横棒グラフ）
plt.subplot(2, 3, 6)
evaluation_factors = {
    '符号': 0.001779,
    '近づい': 0.001769,
    '基礎': 0.001690,
    '人材': 0.001638,
    'おけ': 0.001636,
    'とら': 0.001620,
    '当て': 0.001541,
    '比べ': 0.001528,
    'さまざま': 0.001517,
    'おけ丈夫': 0.001493
}

words_eval = list(evaluation_factors.keys())
values_eval = list(evaluation_factors.values())

bars = plt.barh(words_eval, values_eval, color=colors[2], alpha=0.8, edgecolor='black', linewidth=0.5)
plt.xlabel(hlabel, fontsize=12, fontweight='bold')
plt.title(title6, fontsize=16, fontweight='bold', color='#2C3E50')
plt.grid(True, alpha=0.3, linestyle='--')

# レイアウトの調整
plt.tight_layout(pad=3.0)

# 保存
plt.savefig('マルチタスクSHAP分析_可視化結果.png', dpi=300, bbox_inches='tight')
plt.show()

# 個別の円グラフも作成
fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
fig2.patch.set_facecolor('white')

# 語彙数の円グラフ
wedges1, texts1, autotexts1 = axes[0].pie(df['語彙数'], labels=df['カテゴリ'], autopct='%1.1f%%', 
                                         colors=colors, startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
axes[0].set_title(title1, fontsize=16, fontweight='bold', color='#2C3E50')
axes[0].axis('equal')

# 割合の円グラフ
wedges2, texts2, autotexts2 = axes[1].pie(df['割合'], labels=df['カテゴリ'], autopct='%1.1f%%', 
                                         colors=colors_gradient, startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
axes[1].set_title(title2, fontsize=16, fontweight='bold', color='#2C3E50')
axes[1].axis('equal')

plt.tight_layout()
plt.savefig('マルチタスクSHAP分析_円グラフ.png', dpi=300, bbox_inches='tight')
plt.show()

# 統計情報の表示
if font_success:
    print("=== マルチタスクSHAP分析の統計情報 ===")
    print(f"総語彙数: {df['語彙数'].sum()}")
    print(f"共通要因: {df.loc[0, '語彙数']}語彙 ({df.loc[0, '割合']}%)")
    print(f"感情特化: {df.loc[1, '語彙数']}語彙 ({df.loc[1, '割合']}%)")
    print(f"評価特化: {df.loc[2, '語彙数']}語彙 ({df.loc[2, '割合']}%)")
    print(f"低重要度: {df.loc[3, '語彙数']}語彙 ({df.loc[3, '割合']}%)")
    print("\n=== 重要度の特徴 ===")
    print(f"共通要因の平均統合重要度: {df.loc[0, '平均統合重要度']:.6f}")
    print(f"感情特化の平均統合重要度: {df.loc[1, '平均統合重要度']:.6f}")
    print(f"評価特化の平均統合重要度: {df.loc[2, '平均統合重要度']:.6f}")
    print(f"低重要度の平均統合重要度: {df.loc[3, '平均統合重要度']:.6f}")
    print("\n=== 詳細データ ===")
else:
    print("=== Multitask SHAP Analysis Statistics ===")
    print(f"Total Vocabulary: {df['語彙数'].sum()}")
    print(f"Common Factors: {df.loc[0, '語彙数']} words ({df.loc[0, '割合']}%)")
    print(f"Sentiment Specific: {df.loc[1, '語彙数']} words ({df.loc[1, '割合']}%)")
    print(f"Evaluation Specific: {df.loc[2, '語彙数']} words ({df.loc[2, '割合']}%)")
    print(f"Low Importance: {df.loc[3, '語彙数']} words ({df.loc[3, '割合']}%)")
    print("\n=== Importance Characteristics ===")
    print(f"Common Factors Avg Total Importance: {df.loc[0, '平均統合重要度']:.6f}")
    print(f"Sentiment Specific Avg Total Importance: {df.loc[1, '平均統合重要度']:.6f}")
    print(f"Evaluation Specific Avg Total Importance: {df.loc[2, '平均統合重要度']:.6f}")
    print(f"Low Importance Avg Total Importance: {df.loc[3, '平均統合重要度']:.6f}")
    print("\n=== Detailed Data ===")

print(df.to_string(index=False))

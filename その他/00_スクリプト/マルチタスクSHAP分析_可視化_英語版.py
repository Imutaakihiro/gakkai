import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# フォント設定（英語ラベル使用）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# データの準備（英語ラベル）
data = {
    'Category': ['Common Factors', 'Sentiment Specific', 'Evaluation Specific', 'Low Importance'],
    'Vocabulary Count': [577, 1200, 532, 889],
    'Percentage': [19.4, 40.3, 17.9, 22.4],
    'Avg Sentiment Importance': [0.000727, 0.000770, 0.000302, 0.000313],
    'Avg Evaluation Importance': [0.000695, 0.000289, 0.000707, 0.000298],
    'Avg Total Importance': [0.001422, 0.001059, 0.001009, 0.000610]
}

df = pd.DataFrame(data)

# 色の設定（改善版）
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # より鮮やかで見やすい色
colors_light = ['#F1948A', '#85C1E9', '#82E0AA', '#F7DC6F']  # 薄い色
colors_gradient = ['#FF4757', '#3742FA', '#2ED573', '#FFA502']  # グラデーション色

# グラフのスタイル設定
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 図のサイズ設定
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('white')

# 1. カテゴリ別語彙数の円グラフ
plt.subplot(2, 3, 1)
wedges, texts, autotexts = plt.pie(df['Vocabulary Count'], labels=df['Category'], autopct='%1.1f%%', 
                                  colors=colors, startangle=90, shadow=True, explode=(0.05, 0.05, 0.05, 0.05))
plt.title('Vocabulary Distribution by Category', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
plt.axis('equal')

# 2. カテゴリ別割合の円グラフ
plt.subplot(2, 3, 2)
wedges, texts, autotexts = plt.pie(df['Percentage'], labels=df['Category'], autopct='%1.1f%%', 
                                  colors=colors_gradient, startangle=90, shadow=True, explode=(0.05, 0.05, 0.05, 0.05))
plt.title('Percentage Distribution by Category', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
plt.axis('equal')

# 3. 重要度の比較（棒グラフ）
plt.subplot(2, 3, 3)
x = np.arange(len(df['Category']))
width = 0.25

bars1 = plt.bar(x - width, df['Avg Sentiment Importance'], width, label='Sentiment Importance', 
                color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = plt.bar(x, df['Avg Evaluation Importance'], width, label='Evaluation Importance', 
                color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.5)
bars3 = plt.bar(x + width, df['Avg Total Importance'], width, label='Total Importance', 
                color=colors[2], alpha=0.8, edgecolor='black', linewidth=0.5)

plt.xlabel('Category', fontsize=12, fontweight='bold')
plt.ylabel('Importance', fontsize=12, fontweight='bold')
plt.title('Importance Comparison by Category', fontsize=16, fontweight='bold', color='#2C3E50')
plt.xticks(x, df['Category'], rotation=45, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

# 4. 共通要因TOP10の重要度（横棒グラフ）
plt.subplot(2, 3, 4)
common_factors = {
    '学ぶ (Learn)': 0.002664,
    'myit': 0.002468,
    'まま (As is)': 0.002404,
    '電動 (Electric)': 0.002369,
    'すぐ (Soon)': 0.002283,
    'より (More)': 0.002239,
    '単語語 (Word)': 0.002198,
    '方式 (Method)': 0.002111,
    'れる (Passive)': 0.001983,
    '下さい (Please)': 0.001974
}

words = list(common_factors.keys())
values = list(common_factors.values())

bars = plt.barh(words, values, color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)
plt.xlabel('Total Importance', fontsize=12, fontweight='bold')
plt.title('Top 10 Common Factors', fontsize=16, fontweight='bold', color='#2C3E50')
plt.grid(True, alpha=0.3, linestyle='--')

# 5. 感情特化要因TOP10の重要度（横棒グラフ）
plt.subplot(2, 3, 5)
sentiment_factors = {
    '代わり (Instead)': 0.001951,
    '持っけ (Hold)': 0.001934,
    '忘れ物 (Lost item)': 0.001823,
    'ます書 (Write)': 0.001808,
    '焦点 (Focus)': 0.001768,
    '組む (Combine)': 0.001760,
    '素子 (Element)': 0.001745,
    '英語べ (English)': 0.001732,
    '前回 (Last time)': 0.001726,
    '入力 (Input)': 0.001722
}

words_sent = list(sentiment_factors.keys())
values_sent = list(sentiment_factors.values())

bars = plt.barh(words_sent, values_sent, color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.5)
plt.xlabel('Total Importance', fontsize=12, fontweight='bold')
plt.title('Top 10 Sentiment-Specific Factors', fontsize=16, fontweight='bold', color='#2C3E50')
plt.grid(True, alpha=0.3, linestyle='--')

# 6. 評価特化要因TOP10の重要度（横棒グラフ）
plt.subplot(2, 3, 6)
evaluation_factors = {
    '符号 (Symbol)': 0.001779,
    '近づい (Approach)': 0.001769,
    '基礎 (Foundation)': 0.001690,
    '人材 (Human resource)': 0.001638,
    'おけ (Place)': 0.001636,
    'とら (Take)': 0.001620,
    '当て (Hit)': 0.001541,
    '比べ (Compare)': 0.001528,
    'さまざま (Various)': 0.001517,
    'おけ丈夫 (Strong)': 0.001493
}

words_eval = list(evaluation_factors.keys())
values_eval = list(evaluation_factors.values())

bars = plt.barh(words_eval, values_eval, color=colors[2], alpha=0.8, edgecolor='black', linewidth=0.5)
plt.xlabel('Total Importance', fontsize=12, fontweight='bold')
plt.title('Top 10 Evaluation-Specific Factors', fontsize=16, fontweight='bold', color='#2C3E50')
plt.grid(True, alpha=0.3, linestyle='--')

# レイアウトの調整
plt.tight_layout(pad=3.0)

# 保存
plt.savefig('Multitask_SHAP_Analysis_Visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 個別の円グラフも作成
fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
fig2.patch.set_facecolor('white')

# 語彙数の円グラフ
wedges1, texts1, autotexts1 = axes[0].pie(df['Vocabulary Count'], labels=df['Category'], autopct='%1.1f%%', 
                                         colors=colors, startangle=90, shadow=True, explode=(0.05, 0.05, 0.05, 0.05))
axes[0].set_title('Vocabulary Distribution by Category', fontsize=16, fontweight='bold', color='#2C3E50')
axes[0].axis('equal')

# 割合の円グラフ
wedges2, texts2, autotexts2 = axes[1].pie(df['Percentage'], labels=df['Category'], autopct='%1.1f%%', 
                                         colors=colors_gradient, startangle=90, shadow=True, explode=(0.05, 0.05, 0.05, 0.05))
axes[1].set_title('Percentage Distribution by Category', fontsize=16, fontweight='bold', color='#2C3E50')
axes[1].axis('equal')

plt.tight_layout()
plt.savefig('Multitask_SHAP_Analysis_PieCharts.png', dpi=300, bbox_inches='tight')
plt.show()

# 統計情報の表示
print("=== Multitask SHAP Analysis Statistics ===")
print(f"Total Vocabulary: {df['Vocabulary Count'].sum()}")
print(f"Common Factors: {df.loc[0, 'Vocabulary Count']} words ({df.loc[0, 'Percentage']}%)")
print(f"Sentiment Specific: {df.loc[1, 'Vocabulary Count']} words ({df.loc[1, 'Percentage']}%)")
print(f"Evaluation Specific: {df.loc[2, 'Vocabulary Count']} words ({df.loc[2, 'Percentage']}%)")
print(f"Low Importance: {df.loc[3, 'Vocabulary Count']} words ({df.loc[3, 'Percentage']}%)")
print("\n=== Importance Characteristics ===")
print(f"Common Factors Avg Total Importance: {df.loc[0, 'Avg Total Importance']:.6f}")
print(f"Sentiment Specific Avg Total Importance: {df.loc[1, 'Avg Total Importance']:.6f}")
print(f"Evaluation Specific Avg Total Importance: {df.loc[2, 'Avg Total Importance']:.6f}")
print(f"Low Importance Avg Total Importance: {df.loc[3, 'Avg Total Importance']:.6f}")

# データフレームの表示
print("\n=== Detailed Data ===")
print(df.to_string(index=False))

# 日本語版の円グラフ（別途作成）
fig3, axes = plt.subplots(1, 2, figsize=(15, 6))
fig3.patch.set_facecolor('white')

# 日本語ラベル（フォントが利用可能な場合）
try:
    # 日本語ラベル
    japanese_labels = ['共通要因', '感情特化', '評価特化', '低重要度']
    
    wedges3, texts3, autotexts3 = axes[0].pie(df['Vocabulary Count'], labels=japanese_labels, autopct='%1.1f%%', 
                                             colors=colors, startangle=90, shadow=True, explode=(0.05, 0.05, 0.05, 0.05))
    axes[0].set_title('カテゴリ別語彙数分布', fontsize=16, fontweight='bold', color='#2C3E50')
    axes[0].axis('equal')
    
    wedges4, texts4, autotexts4 = axes[1].pie(df['Percentage'], labels=japanese_labels, autopct='%1.1f%%', 
                                             colors=colors_gradient, startangle=90, shadow=True, explode=(0.05, 0.05, 0.05, 0.05))
    axes[1].set_title('カテゴリ別割合分布', fontsize=16, fontweight='bold', color='#2C3E50')
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('Multitask_SHAP_Analysis_Japanese.png', dpi=300, bbox_inches='tight')
    plt.show()
    
except Exception as e:
    print(f"日本語表示エラー: {e}")
    print("英語版を使用してください")

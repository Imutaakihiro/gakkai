#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単一モデルとマルチタスク学習モデルのBeeswarmプロット比較
SHAP分析手法の解説ドキュメント用の可視化スクリプト
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# PyTorchのバージョン問題を根本的に回避
os.environ['TORCH_DISABLE_SAFETENSORS_WARNING'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_OFFLINE'] = '1'

# DirectML環境の設定
os.environ['PYTORCH_DISABLE_DIRECTML'] = '0'  # DirectMLを有効化

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from collections import defaultdict
import json
from datetime import datetime
import pickle

# 日本語フォント設定
def setup_japanese_font():
    """日本語フォントの設定"""
    import matplotlib.font_manager as fm
    
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        japanese_fonts = [
            'MS Gothic', 'MS Mincho', 'Yu Gothic', 'Meiryo', 'Hiragino Sans',
            'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP',
            'DejaVu Sans', 'Arial Unicode MS'
        ]
        
        for font in japanese_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 日本語フォント設定完了: {font}")
                return True
        
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

print("="*60)
print("単一モデルとマルチタスク学習モデルのBeeswarmプロット比較")
print("SHAP分析手法の解説ドキュメント用")
print("="*60)

# DirectML環境の設定
try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"✅ DirectML使用可能: {device}")
    else:
        device = torch.device("cpu")
        print("⚠️ DirectML使用不可、CPUを使用します")
except ImportError:
    print("⚠️ torch_directmlがインストールされていません")
    device = torch.device("cpu")
    print("⚠️ CPUを使用します")

print(f"使用デバイス: {device}")

def load_single_task_model():
    """単一タスク感情分析モデルを読み込む"""
    print("📥 単一タスク感情分析モデルを読み込み中...")
    
    # 複数のパスを試行（絶対パスも含む）
    model_paths = [
        "02_モデル/マルチタスクモデル",  # 相対パス
        "02_モデル/単一タスクモデル2_評価スコア",
        "finetuned_bert_model_20250718_step2_fixed_classweights_variant1_positive重点強化",
        "../02_モデル/マルチタスクモデル",  # 相対パス（上位ディレクトリ）
        "../02_モデル/単一タスクモデル2_評価スコア"
    ]
    
    for model_path in model_paths:
        try:
            print(f"🔄 試行中: {model_path}")
            
            # パスが存在するかチェック
            if not os.path.exists(model_path):
                print(f"⚠️ パスが存在しません: {model_path}")
                continue
            
            # トークナイザーを読み込み
            if os.path.exists(f"{model_path}/tokenizer_config.json"):
                tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)
            else:
                print(f"⚠️ トークナイザー設定ファイルが見つかりません: {model_path}")
                continue
            
            # モデルを読み込み
            if os.path.exists(f"{model_path}/best_model.pth"):
                # PyTorchモデルファイルの場合
                model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-v3")
                state_dict = torch.load(f"{model_path}/best_model.pth", map_location=device, weights_only=False)
                model.load_state_dict(state_dict)
            elif os.path.exists(f"{model_path}/best_multitask_model.pth"):
                # マルチタスクモデルファイルの場合
                model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-v3")
                state_dict = torch.load(f"{model_path}/best_multitask_model.pth", map_location=device, weights_only=False)
                model.load_state_dict(state_dict)
            else:
                # HuggingFace形式の場合
                model = BertForSequenceClassification.from_pretrained(model_path)
            
            model.to(device)
            model.eval()
            print(f"✅ 単一タスクモデル読み込み成功: {model_path}")
            return model, tokenizer
        except Exception as e:
            print(f"⚠️ 失敗: {e}")
            continue
    
    print("❌ 単一タスクモデルの読み込みに失敗")
    return None, None

def load_multitask_model():
    """マルチタスク学習モデルを読み込む"""
    print("📥 マルチタスク学習モデルを読み込み中...")
    
    # 複数のパスを試行（絶対パスも含む）
    model_paths = [
        "02_モデル/授業レベルマルチタスクモデル",  # 相対パス
        "02_モデル/マルチタスクモデル",
        "../02_モデル/授業レベルマルチタスクモデル",  # 相対パス（上位ディレクトリ）
        "../02_モデル/マルチタスクモデル"
    ]
    
    for model_path in model_paths:
        try:
            print(f"🔄 試行中: {model_path}")
            
            # パスが存在するかチェック
            if not os.path.exists(model_path):
                print(f"⚠️ パスが存在しません: {model_path}")
                continue
            
            # 実際のマルチタスクモデル構造を作成（既存モデルに合わせる）
            class ClassLevelMultitaskModel(torch.nn.Module):
                def __init__(self, vocab_size=30000, hidden_size=768, dropout_rate=0.3):
                    super(ClassLevelMultitaskModel, self).__init__()
                    
                    # BERTエンコーダー
                    from transformers import BertModel
                    self.bert = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-v3")
                    
                    # 感情スコア予測ヘッド（回帰）
                    self.sentiment_classifier = torch.nn.Linear(hidden_size, 1)
                    
                    # 授業評価スコア予測ヘッド（回帰）
                    self.score_regressor = torch.nn.Linear(hidden_size, 1)
                
                def forward(self, input_ids, attention_mask=None):
                    # BERTエンコーダー
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
                    
                    # 各タスクの予測
                    sentiment_pred = self.sentiment_classifier(pooled_output)
                    course_pred = self.score_regressor(pooled_output)
                    
                    return sentiment_pred, course_pred
            
            # モデル読み込み
            model = ClassLevelMultitaskModel()
            
            # 複数の読み込み方法を試行
            model_files = [
                f"{model_path}/best_class_level_multitask_model.pth",
                f"{model_path}/best_multitask_model.pth",
                f"{model_path}/best_model.pth"
            ]
            
            loaded = False
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        state_dict = torch.load(model_file, map_location=device, weights_only=False)
                        model.load_state_dict(state_dict)
                        print(f"✅ torch.loadでマルチタスクモデル読み込み成功: {model_file}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"⚠️ torch.load失敗: {e}")
                        # 代替方法
                        try:
                            with open(model_file, 'rb') as f:
                                state_dict = pickle.load(f)
                            model.load_state_dict(state_dict)
                            print(f"✅ pickleでマルチタスクモデル読み込み成功: {model_file}")
                            loaded = True
                            break
                        except Exception as e2:
                            print(f"⚠️ pickleも失敗: {e2}")
                            continue
            
            if not loaded:
                print(f"❌ すべての方法で失敗: {model_path}")
                continue
            
            model.to(device)
            model.eval()
            
            # トークナイザー（BERTベース）
            tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"⚠️ 失敗: {e}")
            continue
    
    print("❌ マルチタスクモデルの読み込みに失敗")
    return None, None

def load_data():
    """データを読み込む"""
    print("📊 データを読み込み中...")
    
    data_paths = [
        "01_データ/マルチタスク用データ/授業集約データセット_20251012_142504.csv",  # 相対パス
        "01_データ/自由記述→感情スコア/finetuning_val_20250710_220621.csv",
        "../01_データ/マルチタスク用データ/授業集約データセット_20251012_142504.csv",  # 相対パス（上位ディレクトリ）
        "../01_データ/自由記述→感情スコア/finetuning_val_20250710_220621.csv"
    ]
    
    for data_path in data_paths:
        try:
            df = pd.read_csv(data_path)
            if '自由記述まとめ' in df.columns or '自由記述' in df.columns:
                print(f"✅ データ読み込み成功: {data_path} ({len(df)}件)")
                return df
        except Exception as e:
            print(f"⚠️ データ読み込み失敗: {e}")
            continue
    
    print("❌ データの読み込みに失敗")
    return None

def create_prediction_functions(single_model, single_tokenizer, multitask_model, multitask_tokenizer):
    """予測関数を作成"""
    
    def predict_single_sentiment(texts):
        """単一タスク感情分析の予測関数"""
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = [str(texts)]
        
        texts = [str(t) if t else "" for t in texts]
        
        probs = []
        for text in texts:
            try:
                inputs = single_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = single_model(**inputs)
                    logits = outputs.logits
                    prob = torch.softmax(logits, dim=-1)
                    probs.append(prob.cpu().numpy()[0])
            except Exception as e:
                print(f"単一タスク予測エラー: {e}")
                # エラー時は適切な形状の配列を返す
                probs.append(np.array([0.33, 0.33, 0.34]))
        
        return np.array(probs)
    
    def predict_multitask_sentiment(texts):
        """マルチタスク学習の感情スコア予測関数"""
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = [str(texts)]
        
        texts = [str(t) if t else "" for t in texts]
        
        predictions = []
        for text in texts:
            try:
                # BERTトークナイザーでトークン化
                inputs = multitask_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    sentiment_pred, course_pred = multitask_model(**inputs)
                    predictions.append(sentiment_pred.cpu().numpy()[0])
            except Exception as e:
                print(f"マルチタスク予測エラー: {e}")
                predictions.append([0.5])  # デフォルト値
        
        return np.array(predictions)
    
    def predict_multitask_course(texts):
        """マルチタスク学習の授業評価スコア予測関数"""
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = [str(texts)]
        
        texts = [str(t) if t else "" for t in texts]
        
        predictions = []
        for text in texts:
            try:
                # BERTトークナイザーでトークン化
                inputs = multitask_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    sentiment_pred, course_pred = multitask_model(**inputs)
                    predictions.append(course_pred.cpu().numpy()[0])
            except Exception as e:
                print(f"マルチタスク予測エラー: {e}")
                predictions.append([0.5])  # デフォルト値
        
        return np.array(predictions)
    
    return predict_single_sentiment, predict_multitask_sentiment, predict_multitask_course

def create_beeswarm_plots(single_model, single_tokenizer, multitask_model, multitask_tokenizer, df):
    """Beeswarmプロットを作成"""
    print("🐝 Beeswarmプロットを作成中...")
    
    # 出力ディレクトリ作成
    output_dir = "03_分析結果/SHAP_Beeswarm比較"
    os.makedirs(output_dir, exist_ok=True)
    
    # サンプルデータ準備
    if '自由記述まとめ' in df.columns:
        texts = df['自由記述まとめ'].dropna().tolist()
    elif '自由記述' in df.columns:
        texts = df['自由記述'].dropna().tolist()
    else:
        print("❌ 適切なテキスト列が見つかりません")
        return
    
    # サンプリング（20件でテスト）
    sample_size = min(20, len(texts))
    sample_texts = np.random.choice(texts, size=sample_size, replace=False).tolist()
    print(f"📝 サンプルテキスト: {len(sample_texts)}件")
    
    # 予測関数作成
    predict_single, predict_multitask_sentiment, predict_multitask_course = create_prediction_functions(
        single_model, single_tokenizer, multitask_model, multitask_tokenizer
    )
    
    try:
        # 1. 単一タスク感情分析のBeeswarmプロット
        print("🔬 単一タスク感情分析のSHAP分析実行中...")
        explainer_single = shap.Explainer(predict_single, single_tokenizer)
        shap_values_single = explainer_single(sample_texts)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_single, sample_texts, show=False)
        title = "単一タスク感情分析モデルのSHAP Beeswarm Plot" if font_success else "Single Task Sentiment Analysis SHAP Beeswarm Plot"
        plt.title(title, fontsize=16, pad=20, color='#2C3E50')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/single_task_beeswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ 単一タスクBeeswarmプロット作成完了")
        
        # 2. マルチタスク学習の感情スコアBeeswarmプロット
        print("🔬 マルチタスク学習の感情スコアSHAP分析実行中...")
        explainer_multitask_sentiment = shap.Explainer(predict_multitask_sentiment, multitask_tokenizer)
        shap_values_multitask_sentiment = explainer_multitask_sentiment(sample_texts)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_multitask_sentiment, sample_texts, show=False)
        title = "マルチタスク学習モデルの感情スコアSHAP Beeswarm Plot" if font_success else "Multitask Learning Model Sentiment Score SHAP Beeswarm Plot"
        plt.title(title, fontsize=16, pad=20, color='#2C3E50')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/multitask_sentiment_beeswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ マルチタスク感情スコアBeeswarmプロット作成完了")
        
        # 3. マルチタスク学習の授業評価スコアBeeswarmプロット
        print("🔬 マルチタスク学習の授業評価スコアSHAP分析実行中...")
        explainer_multitask_course = shap.Explainer(predict_multitask_course, multitask_tokenizer)
        shap_values_multitask_course = explainer_multitask_course(sample_texts)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_multitask_course, sample_texts, show=False)
        title = "マルチタスク学習モデルの授業評価スコアSHAP Beeswarm Plot" if font_success else "Multitask Learning Model Course Score SHAP Beeswarm Plot"
        plt.title(title, fontsize=16, pad=20, color='#2C3E50')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/multitask_course_beeswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ マルチタスク授業評価スコアBeeswarmプロット作成完了")
        
        # 4. 比較用のサブプロット
        print("📊 比較用サブプロットを作成中...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("SHAP Beeswarm Plot 比較: 単一タスク vs マルチタスク学習" if font_success else "SHAP Beeswarm Plot Comparison: Single Task vs Multitask Learning", 
                     fontsize=18, color='#2C3E50')
        
        # 単一タスク
        shap.summary_plot(shap_values_single, sample_texts, show=False, ax=axes[0,0])
        axes[0,0].set_title("単一タスク感情分析" if font_success else "Single Task Sentiment", fontsize=14)
        
        # マルチタスク感情スコア
        shap.summary_plot(shap_values_multitask_sentiment, sample_texts, show=False, ax=axes[0,1])
        axes[0,1].set_title("マルチタスク感情スコア" if font_success else "Multitask Sentiment Score", fontsize=14)
        
        # マルチタスク授業評価スコア
        shap.summary_plot(shap_values_multitask_course, sample_texts, show=False, ax=axes[1,0])
        axes[1,0].set_title("マルチタスク授業評価スコア" if font_success else "Multitask Course Score", fontsize=14)
        
        # 空のサブプロット（将来の拡張用）
        axes[1,1].text(0.5, 0.5, "将来の拡張用" if font_success else "Future Extension", 
                       ha='center', va='center', fontsize=16, color='gray')
        axes[1,1].set_title("拡張予定" if font_success else "Future Extension", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_beeswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ 比較用サブプロット作成完了")
        
        # 結果の保存
        results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "analysis_type": "beeswarm_comparison",
            "sample_size": len(sample_texts),
            "models": {
                "single_task": {
                    "shap_values_shape": shap_values_single.shape,
                    "model_type": "sentiment_classification"
                },
                "multitask_sentiment": {
                    "shap_values_shape": shap_values_multitask_sentiment.shape,
                    "model_type": "sentiment_regression"
                },
                "multitask_course": {
                    "shap_values_shape": shap_values_multitask_course.shape,
                    "model_type": "course_regression"
                }
            },
            "output_files": [
                f"single_task_beeswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                f"multitask_sentiment_beeswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                f"multitask_course_beeswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                f"comparison_beeswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            ],
            "font_success": font_success
        }
        
        with open(f"{output_dir}/beeswarm_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 結果保存完了: {output_dir}")
        
    except Exception as e:
        print(f"❌ Beeswarmプロット作成エラー: {e}")
        print("🔄 簡易版を実行します...")
        
        # 簡易版（より小さなサンプル）
        try:
            sample_texts_small = sample_texts[:5]  # 5件でテスト
            
            # 単一タスク
            explainer_single = shap.Explainer(predict_single, single_tokenizer)
            shap_values_single = explainer_single(sample_texts_small)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_single, sample_texts_small, show=False)
            plt.title("単一タスク感情分析 (簡易版)" if font_success else "Single Task Sentiment (Simple)", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/single_task_beeswarm_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print("✅ 簡易版Beeswarmプロット作成完了")
            
        except Exception as e2:
            print(f"❌ 簡易版もエラー: {e2}")

def main():
    """メイン実行関数"""
    print("🚀 Beeswarmプロット比較分析を開始...")
    
    # 1. モデル読み込み
    single_model, single_tokenizer = load_single_task_model()
    multitask_model, multitask_tokenizer = load_multitask_model()
    
    if single_model is None or multitask_model is None:
        print("❌ モデルの読み込みに失敗しました")
        return
    
    # 2. データ読み込み
    df = load_data()
    if df is None:
        print("❌ データの読み込みに失敗しました")
        return
    
    # 3. Beeswarmプロット作成
    create_beeswarm_plots(single_model, single_tokenizer, multitask_model, multitask_tokenizer, df)
    
    print("\n🎉 Beeswarmプロット比較分析完了！")
    print("📁 結果は '03_分析結果/SHAP_Beeswarm比較' に保存されました")

if __name__ == "__main__":
    main()

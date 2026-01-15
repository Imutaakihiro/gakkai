#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
授業単位のマルチタスク学習
感情スコア予測 + 授業評価スコア予測を同時に学習

データ構成:
- 入力: 授業の全自由記述（集団レベル）
- 出力1: 感情スコア平均（集団レベル）
- 出力2: 授業評価スコア（集団レベル）
→ レベルの一致により学習可能
"""

import os
# CUDA同期モード（デバッグ用）: デフォルトOFF
DEBUG_CUDA_SYNC = False
if DEBUG_CUDA_SYNC:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    # 以前の実行で環境変数が残っている場合の保険
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        del os.environ["CUDA_LAUNCH_BLOCKING"]

# デバイス自動切替機能
def get_available_device():
    """利用可能なデバイスを自動選択（CUDA → DirectML → CPU）"""
    try:
        # まずCUDAを試行
        if torch.cuda.is_available():
            # CUDAが利用可能でも、実際にテストして確認
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                _ = test_tensor + test_tensor  # 簡単な演算でテスト
                print("✅ CUDA デバイスが利用可能です")
                return torch.device('cuda')
            except RuntimeError as e:
                if "no kernel image" in str(e):
                    print("⚠️ CUDA で 'no kernel image' エラーが発生しました")
                    print("🔄 DirectML または CPU にフォールバックします")
                else:
                    print(f"⚠️ CUDA エラー: {e}")
                    print("🔄 DirectML または CPU にフォールバックします")
    except Exception as e:
        print(f"⚠️ CUDA チェックエラー: {e}")
    
    # DirectMLを試行
    try:
        import torch_directml as dml
        if dml.is_available():
            # DirectMLデバイスを実際にテスト
            try:
                device = dml.device()
                test_tensor = torch.randn(2, 2, device=device)
                _ = test_tensor @ test_tensor  # 簡単な演算でテスト
                print("✅ DirectML デバイスが利用可能です")
                return device
            except Exception as dml_error:
                print(f"⚠️ DirectML デバイステストエラー: {dml_error}")
                if "staticmethod" in str(dml_error):
                    print("   これは PyTorch 2.4.x + torch-directml の互換性問題です")
                    print("   推奨: PyTorch 2.2.2 + torch-directml 0.2.3.dev240715")
    except ImportError:
        print("ℹ️ DirectML がインストールされていません")
    except Exception as e:
        print(f"⚠️ DirectML エラー: {e}")
    
    # CPUにフォールバック
    print("🔄 CPU デバイスを使用します")
    return torch.device('cpu')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertJapaneseTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
import sys
import time
warnings.filterwarnings('ignore')

# デバイス設定（自動切替機能を使用）
device = get_available_device()
print(f"使用デバイス: {device}")

# GPU詳細情報の表示
print(f"🔍 CUDA環境チェック:")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   Device count: {torch.cuda.device_count()}")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"🚀 GPU詳細情報:")
    print(f"   📊 GPU名: {torch.cuda.get_device_name(0)}")
    print(f"   💾 総メモリ: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    print(f"   🔧 Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"   📈 現在の使用メモリ: {torch.cuda.memory_allocated(0) / (1024**3):.2f}GB")
    print(f"   🔧 CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', 'Not set')}")
else:
    print("⚠️  GPUが利用できません。CPUで実行します。")

# ---- CUDA/SDPA 安定化設定（Windows + RTX 40 系での初回 forward ハング対策）----
try:
    import platform
    # cuDNN の自動最適化を無効化し、決定論的に（デバッグ性向上）
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        # PyTorch 2.x の SDPA backend を math のみに固定して Flash/MemEfficient を無効化
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel.enable_flash(False)
            sdp_kernel.enable_mem_efficient(False)
            sdp_kernel.enable_math(True)
            print("🧯 SDPA: flash/mem_efficient 無効化 (math のみ)")
        except Exception:
            # 旧 API（バージョン差分用）
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
                print("🧯 SDPA: 旧APIで math のみに固定")
            except Exception:
                print("ℹ️ SDPA backend の固定はスキップ（非対応バージョン）")
        # Fuser のフォールバック抑止（稀なフリーズ回避）
        os.environ.setdefault("PYTORCH_CUDA_FUSER_DISABLE_FALLBACK", "1")
except Exception as _e:
    print(f"⚠️ CUDA/SDPA 安定化設定で例外: {_e}")

# 追加の高速化/安定化（TF32）
try:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("⚡ TF32 有効化: high precision matmul")
except Exception as _e:
    print(f"ℹ️ TF32設定スキップ: {_e}")

# ベースモデル
BASE_MODEL = "koheiduck/bert-japanese-finetuned-sentiment"

# ハイパーパラメータ（長文対応: 分割 + 集約 方式）
MAX_LENGTH = 256   # 単一チャンクの長さ（トークン数）
BATCH_SIZE = 2     # 物理バッチ
ACCUM_STEPS = 4    # 勾配蓄積（実効バッチ = BATCH_SIZE * ACCUM_STEPS）
USE_AMP = True     # 自動混合精度
# 既定は高速設定だが、古いTorch/Windowsでは安全側に自動ダウングレード
PIN_MEMORY = True         # HtoD転送を固定メモリ化
NON_BLOCKING = True       # 非同期転送（pin_memory=True時に有効）
USE_GRADIENT_CHECKPOINTING = False  # PyTorch 1.13 環境では無効化が安定
WARMUP_FORWARD = False    # 初回forwardウォームアップ（停止の原因になる場合があるためデフォルトOFF）

# DataLoader 並列設定（デフォルト）
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

# チャンク設定（長文スライディングウィンドウ）
CHUNK_LEN = 256
STRIDE = 200
MAX_CHUNKS = 10
LEARNING_RATE = 2e-5
NUM_EPOCHS = 20
ALPHA = 0.5  # 感情スコアの重み
BETA = 0.5   # 評価スコアの重み

print(f"🔧 ハイパーパラメータ設定（長文対応・高速化）:")
print(f"   CHUNK_LEN: {CHUNK_LEN} / STRIDE: {STRIDE} / MAX_CHUNKS: {MAX_CHUNKS}")
print(f"   MAX_LENGTH(=CHUNK_LEN): {MAX_LENGTH}")
print(f"   BATCH_SIZE: {BATCH_SIZE}  ACCUM_STEPS: {ACCUM_STEPS}  USE_AMP: {USE_AMP}")
print(f"   LEARNING_RATE: {LEARNING_RATE}")
print(f"   NUM_EPOCHS: {NUM_EPOCHS}")

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    try:
        plt.rcParams['font.family'] = 'Yu Gothic'
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def create_progress_bar(current, total, width=50, prefix="", suffix=""):
    """プログレスバーを作成"""
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r{prefix} |{bar}| {percent:.1%} {suffix}"


def get_gpu_status():
    """GPU使用状況を取得"""
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)   # GB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100
        
        return {
            'allocated': gpu_memory_allocated,
            'reserved': gpu_memory_reserved,
            'total': gpu_memory_total,
            'utilization': gpu_utilization
        }
    return None


def print_progress_gauge(current, total, prefix="", suffix="", show_percent=True, show_gpu=True):
    """ゲージ風の進捗表示（GPU状況付き）- 毎秒更新"""
    percent = current / total
    
    # ゲージの幅
    gauge_width = 20
    filled = int(gauge_width * percent)
    empty = gauge_width - filled
    
    # ゲージの色（ターミナルで色分け）
    gauge_bar = "█" * filled + "░" * empty
    
    # パーセンテージ表示
    percent_str = f"{percent:.1%}" if show_percent else ""
    
    # GPU状況の表示
    gpu_info = ""
    if show_gpu and torch.cuda.is_available():
        gpu_status = get_gpu_status()
        if gpu_status:
            gpu_info = f" | GPU: {gpu_status['utilization']:.1f}% ({gpu_status['allocated']:.1f}GB/{gpu_status['total']:.1f}GB)"
    
    # 現在時刻を追加
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # 毎秒更新出力（同じ行を上書き）
    print(f"\r[{current_time}] {prefix} [{gauge_bar}] {percent_str}{gpu_info} {suffix}", end="", flush=True)
    
    if current == total:
        print()  # 完了時に改行


def print_epoch_summary(epoch, total_epochs, train_loss, val_loss, sent_r2, course_r2, 
                       best_val_loss, current_lr, elapsed_time=None):
    """エポックサマリーをゲージ風で表示"""
    print(f"\n{'='*80}")
    print(f"🎯 EPOCH {epoch}/{total_epochs} 完了")
    print(f"{'='*80}")
    
    # GPU状況の表示
    if torch.cuda.is_available():
        gpu_status = get_gpu_status()
        if gpu_status:
            print(f"🚀 GPU状況: {gpu_status['utilization']:.1f}% 使用中")
            print(f"   📊 メモリ: {gpu_status['allocated']:.1f}GB / {gpu_status['total']:.1f}GB")
            print(f"   🔒 予約済み: {gpu_status['reserved']:.1f}GB")
    
    # エポック進捗ゲージ
    epoch_progress = epoch / total_epochs
    print_progress_gauge(epoch, total_epochs, "📈 エポック進捗", f"({epoch}/{total_epochs})", True, False)
    
    # 損失のゲージ表示
    max_loss = max(train_loss, val_loss) * 1.2  # 最大値の120%を基準
    train_loss_gauge = min(train_loss / max_loss, 1.0)
    val_loss_gauge = min(val_loss / max_loss, 1.0)
    
    print(f"🔥 学習損失: {train_loss:.4f}")
    print_progress_gauge(train_loss_gauge, 1.0, "  ", f"({train_loss:.4f})", False, False)
    
    print(f"✅ 検証損失: {val_loss:.4f}")
    print_progress_gauge(val_loss_gauge, 1.0, "  ", f"({val_loss:.4f})", False, False)
    
    # R²スコアのゲージ表示
    print(f"📊 感情スコア R²: {sent_r2:.4f}")
    print_progress_gauge(max(0, sent_r2), 1.0, "  ", f"({sent_r2:.4f})", False, False)
    
    print(f"📊 授業評価 R²: {course_r2:.4f}")
    print_progress_gauge(max(0, course_r2), 1.0, "  ", f"({course_r2:.4f})", False, False)
    
    # ベストモデル状況
    if val_loss < best_val_loss:
        print(f"🏆 ベストモデル更新！ (Val Loss: {val_loss:.4f})")
    else:
        print(f"⏳ ベストモデル維持 (Best: {best_val_loss:.4f})")
    
    # 学習率と時間
    print(f"📚 学習率: {current_lr:.2e}")
    if elapsed_time:
        print(f"⏰ 経過時間: {elapsed_time/60:.1f}分")
    
    print(f"{'='*80}")


class ClassLevelDataset(Dataset):
    """授業レベルのデータセット（長文対応: チャンク化）"""
    def __init__(self, texts, sentiment_scores, course_scores, tokenizer,
                 chunk_len=256, stride=200, max_chunks=10):
        self.texts = texts
        self.sentiment_scores = sentiment_scores
        self.course_scores = course_scores
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.stride = stride
        self.max_chunks = max_chunks

        # 特殊トークンID
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

    def __len__(self):
        return len(self.texts)

    def _chunk_encode(self, text):
        # 特殊トークンを自前で付与するため add_special_tokens=False
        token_ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        inner_max = self.chunk_len - 2
        chunks_ids = []

        if len(token_ids) == 0:
            # 空文字対策: [CLS][SEP] のみ
            ids = [self.cls_id, self.sep_id] + [0]*(self.chunk_len-2)
            attn = [1, 1] + [0]*(self.chunk_len-2)
            return [torch.tensor(ids, dtype=torch.long)], [torch.tensor(attn, dtype=torch.long)]

        for start in range(0, len(token_ids), self.stride):
            inner = token_ids[start:start+inner_max]
            ids = [self.cls_id] + inner + [self.sep_id]
            # パディング
            pad_len = self.chunk_len - len(ids)
            if pad_len > 0:
                ids = ids + [0]*pad_len
                attn = [1]*len(inner+ [self.cls_id, self.sep_id]) + [0]*pad_len
            else:
                ids = ids[:self.chunk_len]
                attn = [1]*self.chunk_len
            chunks_ids.append((torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)))
            if len(chunks_ids) >= self.max_chunks:
                break

        input_ids_list = [x[0] for x in chunks_ids]
        attention_list = [x[1] for x in chunks_ids]
        return input_ids_list, attention_list

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentiment_score = self.sentiment_scores[idx]
        course_score = self.course_scores[idx]

        ids_list, attn_list = self._chunk_encode(text)

        return {
            'input_ids_list': ids_list,
            'attention_mask_list': attn_list,
            'num_chunks': len(ids_list),
            'sentiment_score': torch.tensor(sentiment_score, dtype=torch.float),
            'course_score': torch.tensor(course_score, dtype=torch.float)
        }


def collate_chunked_batch(batch):
    """可変チャンクを (B, C, L) にまとめるcollate"""
    B = len(batch)
    C = MAX_CHUNKS
    L = CHUNK_LEN

    input_ids = torch.zeros((B, C, L), dtype=torch.long)
    attention_mask = torch.zeros((B, C, L), dtype=torch.long)
    chunk_mask = torch.zeros((B, C), dtype=torch.bool)
    y_sent = torch.zeros((B,), dtype=torch.float)
    y_course = torch.zeros((B,), dtype=torch.float)

    for i, item in enumerate(batch):
        n = min(item['num_chunks'], C)
        for j in range(n):
            input_ids[i, j] = item['input_ids_list'][j]
            attention_mask[i, j] = item['attention_mask_list'][j]
            chunk_mask[i, j] = True
        y_sent[i] = item['sentiment_score']
        y_course[i] = item['course_score']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'chunk_mask': chunk_mask,
        'sentiment_score': y_sent,
        'course_score': y_course
    }


class ClassLevelMultitaskModel(nn.Module):
    """授業レベルのマルチタスクモデル"""
    
    def __init__(self, base_model_name, dropout_rate=0.1):
        super().__init__()
        
        # BERTエンコーダ（共有層）- safetensors使用でセキュリティ要件対応
        try:
            self.bert = BertModel.from_pretrained(base_model_name, use_safetensors=True)
        except Exception:
            # safetensorsが利用できない場合は従来の方法
            self.bert = BertModel.from_pretrained(base_model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 感情スコア予測ヘッド（回帰）
        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
        
        # 授業評価スコア予測ヘッド（回帰）
        self.course_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask, chunk_mask=None):
        # 入力が (B, C, L) の場合は平坦化してまとめてエンコード
        if input_ids.dim() == 3:
            B, C, L = input_ids.shape
            x_ids = input_ids.view(B*C, L)
            x_mask = attention_mask.view(B*C, L)
            outputs = self.bert(input_ids=x_ids, attention_mask=x_mask)
            cls = outputs.last_hidden_state[:, 0, :]  # (B*C, H)
            H = cls.size(-1)
            cls = cls.view(B, C, H)  # (B, C, H)

            if chunk_mask is None:
                # すべてのチャンクを平均
                pooled = cls.mean(dim=1)
            else:
                # マスク付き平均
                mask = chunk_mask.float().unsqueeze(-1)  # (B, C, 1)
                summed = (cls * mask).sum(dim=1)  # (B, H)
                denom = mask.sum(dim=1).clamp_min(1e-6)  # (B, 1)
                pooled = summed / denom
        else:
            # 互換: (B, L)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]

        # 各タスクの予測
        sentiment_pred = self.sentiment_head(pooled).squeeze(-1)
        course_pred = self.course_head(pooled).squeeze(-1)
        return sentiment_pred, course_pred


def find_latest_csv_file():
    """最新の授業集約データセットCSVファイルを自動検出"""
    import os
    import glob
    
    # 複数のパス候補を試す
    possible_paths = [
        '../01_データ/マルチタスク用データ/',
        '../../01_データ/マルチタスク用データ/',
        '01_データ/マルチタスク用データ/',
        '../01_データ/マルチタスク用データ',
        '../../01_データ/マルチタスク用データ',
        '01_データ/マルチタスク用データ'
    ]
    
    csv_files = []
    for base_path in possible_paths:
        pattern = os.path.join(base_path, '授業集約データセット_*.csv')
        found_files = glob.glob(pattern)
        csv_files.extend(found_files)
    
    if not csv_files:
        # より広範囲で検索
        for root, dirs, files in os.walk('..'):
            for file in files:
                if file.startswith('授業集約データセット_') and file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
    
    if csv_files:
        # 最新のファイルを選択（ファイル名の日時でソート）
        latest_file = max(csv_files, key=os.path.getctime)
        print(f"📁 見つかったCSVファイル: {latest_file}")
        return latest_file
    else:
        raise FileNotFoundError("授業集約データセットのCSVファイルが見つかりません")


def load_data(sample_size=1000):
    """データの読み込み（層化サンプリング）"""
    print("\n" + "="*60)
    print("📊 授業集約データセットの読み込み（層化サンプリング）")
    print("="*60)
    
    # CSVファイルの自動検出と読み込み
    csv_file_path = find_latest_csv_file()
    df = pd.read_csv(csv_file_path)
    
    print(f"総授業数: {len(df)}件")
    
    # 層化サンプリングのための層を作成
    # 感情スコアと授業評価スコアの両方を考慮して層を作成
    sentiment_bins = pd.qcut(df['感情スコア平均'], q=3, labels=['低', '中', '高'], duplicates='drop')
    course_bins = pd.qcut(df['授業評価スコア'], q=3, labels=['低', '中', '高'], duplicates='drop')
    
    # 層ラベルを組み合わせて詳細な層を作成
    stratify_labels = [f'{s}_{c}' for s, c in zip(sentiment_bins, course_bins)]
    
    print(f"\n層化サンプリングの分布:")
    unique, counts = np.unique(stratify_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}件")
    
    # 層化サンプリングを実行
    np.random.seed(42)
    if sample_size < len(df):
        # 各層から比例的にサンプリング
        df_sampled = df.groupby(stratify_labels, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), int(sample_size * len(x) / len(df))), random_state=42)
        ).reset_index(drop=True)
        
        # 目標件数に調整
        if len(df_sampled) < sample_size:
            # 不足分をランダムに追加
            remaining_indices = df[~df.index.isin(df_sampled.index)].index
            additional_size = sample_size - len(df_sampled)
            if len(remaining_indices) >= additional_size:
                additional_indices = np.random.choice(remaining_indices, additional_size, replace=False)
                df_sampled = pd.concat([df_sampled, df.iloc[additional_indices]]).reset_index(drop=True)
        elif len(df_sampled) > sample_size:
            # 超過分をランダムに削除
            df_sampled = df_sampled.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"層化サンプリング: {len(df_sampled)}件を抽出")
    else:
        df_sampled = df
        print(f"サンプリング: 全データを使用")
    
    print(f"使用授業数: {len(df_sampled)}件")
    print(f"列名: {list(df_sampled.columns)}")
    
    # 必要な列を抽出
    texts = df_sampled['自由記述まとめ'].values
    sentiment_scores = df_sampled['感情スコア平均'].values
    course_scores = df_sampled['授業評価スコア'].values
    
    print(f"\n感情スコア平均の統計:")
    print(f"  平均: {sentiment_scores.mean():.4f}")
    print(f"  標準偏差: {sentiment_scores.std():.4f}")
    print(f"  範囲: {sentiment_scores.min():.4f} 〜 {sentiment_scores.max():.4f}")
    
    print(f"\n授業評価スコアの統計:")
    print(f"  平均: {course_scores.mean():.4f}")
    print(f"  標準偏差: {course_scores.std():.4f}")
    print(f"  範囲: {course_scores.min():.4f} 〜 {course_scores.max():.4f}")
    
    return texts, sentiment_scores, course_scores


def prepare_data(texts, sentiment_scores, course_scores, tokenizer):
    """データの準備（層化サンプリング）"""
    print("\n" + "="*60)
    print("🔄 データの準備（層化サンプリング）")
    print("="*60)
    
    # 層化サンプリングのための層を作成
    # 感情スコアと授業評価スコアの両方を考慮して層を作成
    sentiment_bins = pd.qcut(sentiment_scores, q=3, labels=['低', '中', '高'], duplicates='drop')
    course_bins = pd.qcut(course_scores, q=3, labels=['低', '中', '高'], duplicates='drop')
    
    # 層ラベルを組み合わせて詳細な層を作成
    stratify_labels = [f'{s}_{c}' for s, c in zip(sentiment_bins, course_bins)]
    
    print(f"\n層化サンプリングの分布:")
    unique, counts = np.unique(stratify_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}件")
    
    # データ分割（70% / 15% / 15%）層化サンプリング
    X_temp, X_test, y_sent_temp, y_sent_test, y_course_temp, y_course_test, strat_temp, strat_test = train_test_split(
        texts, sentiment_scores, course_scores, stratify_labels,
        test_size=0.15, random_state=42, stratify=stratify_labels
    )
    
    X_train, X_val, y_sent_train, y_sent_val, y_course_train, y_course_val = train_test_split(
        X_temp, y_sent_temp, y_course_temp, 
        test_size=0.176, random_state=42, stratify=strat_temp  # 0.176 ≈ 15/85
    )
    
    print(f"\nデータ分割:")
    print(f"  学習データ: {len(X_train)}件 ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"  検証データ: {len(X_val)}件 ({len(X_val)/len(texts)*100:.1f}%)")
    print(f"  テストデータ: {len(X_test)}件 ({len(X_test)/len(texts)*100:.1f}%)")
    
    # 各セットの分布を確認
    print(f"\n学習データの感情スコア分布:")
    print(f"  平均: {y_sent_train.mean():.4f}, 標準偏差: {y_sent_train.std():.4f}")
    print(f"  範囲: {y_sent_train.min():.4f} 〜 {y_sent_train.max():.4f}")
    
    print(f"\n学習データの授業評価スコア分布:")
    print(f"  平均: {y_course_train.mean():.4f}, 標準偏差: {y_course_train.std():.4f}")
    print(f"  範囲: {y_course_train.min():.4f} 〜 {y_course_train.max():.4f}")
    
    # スコアの正規化
    sentiment_scaler = StandardScaler()
    course_scaler = StandardScaler()
    
    y_sent_train_scaled = sentiment_scaler.fit_transform(y_sent_train.reshape(-1, 1)).flatten()
    y_sent_val_scaled = sentiment_scaler.transform(y_sent_val.reshape(-1, 1)).flatten()
    y_sent_test_scaled = sentiment_scaler.transform(y_sent_test.reshape(-1, 1)).flatten()
    
    y_course_train_scaled = course_scaler.fit_transform(y_course_train.reshape(-1, 1)).flatten()
    y_course_val_scaled = course_scaler.transform(y_course_val.reshape(-1, 1)).flatten()
    y_course_test_scaled = course_scaler.transform(y_course_test.reshape(-1, 1)).flatten()
    
    # データセットの作成（長文対応: チャンク化）
    train_dataset = ClassLevelDataset(
        X_train, y_sent_train_scaled, y_course_train_scaled, tokenizer,
        chunk_len=CHUNK_LEN, stride=STRIDE, max_chunks=MAX_CHUNKS
    )
    val_dataset = ClassLevelDataset(
        X_val, y_sent_val_scaled, y_course_val_scaled, tokenizer,
        chunk_len=CHUNK_LEN, stride=STRIDE, max_chunks=MAX_CHUNKS
    )
    test_dataset = ClassLevelDataset(
        X_test, y_sent_test_scaled, y_course_test_scaled, tokenizer,
        chunk_len=CHUNK_LEN, stride=STRIDE, max_chunks=MAX_CHUNKS
    )
    
    # DataLoaderの作成（Windows環境対応: num_workers=0）
    print(f"📦 DataLoader作成中...")
    # バージョン/OSに応じて安全なDataLoader設定を自動選択
    import platform
    torch_major = int(torch.__version__.split('.')[0])
    is_windows = platform.system().lower().startswith('win')
    use_safe_loader = (torch_major < 2) or is_windows

    if use_safe_loader:
        print("⚙️ DataLoader: 安全設定で起動 (num_workers=0, pin_memory=False)")
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=False, collate_fn=collate_chunked_batch
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=False, collate_fn=collate_chunked_batch
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=False, collate_fn=collate_chunked_batch
        )
        # 転送の非同期は無効化
        global NON_BLOCKING
        NON_BLOCKING = False
    else:
        print(f"⚙️ DataLoader: 高速設定 (num_workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}, prefetch={PREFETCH_FACTOR}, persistent={PERSISTENT_WORKERS})")
        dl_kwargs = dict(
            batch_size=BATCH_SIZE,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS,
            collate_fn=collate_chunked_batch,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **dl_kwargs)
    print(f"✅ DataLoader作成完了")
    
    return train_loader, val_loader, test_loader, sentiment_scaler, course_scaler


def train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, num_epochs):
    """1エポックの学習"""
    model.train()
    total_loss = 0
    sentiment_losses = 0
    course_losses = 0
    
    criterion = nn.MSELoss()
    last_update_time = time.time()
    
    print(f"  📊 学習バッチ数: {len(train_loader)}")
    print(f"  🔧 デバイス: {device}")
    print(f"  🚀 学習ループ開始...", flush=True)
    
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        # 最初のバッチでデバッグ情報を表示
        if batch_idx == 0:
            print(f"  🚀 最初のバッチ処理開始...")
            print(f"  📦 バッチサイズ: {batch['input_ids'].shape[0]}")
        
        # データをデバイスに転送
        if batch_idx == 0:
            print(f"  📤 データを{device}に転送中...", flush=True)
        input_ids = batch['input_ids'].to(device, non_blocking=NON_BLOCKING)
        attention_mask = batch['attention_mask'].to(device, non_blocking=NON_BLOCKING)
        chunk_mask = batch['chunk_mask'].to(device, non_blocking=NON_BLOCKING)
        sentiment_true = batch['sentiment_score'].to(device, non_blocking=NON_BLOCKING)
        course_true = batch['course_score'].to(device, non_blocking=NON_BLOCKING)
        
        # 最初のバッチでデバイス転送確認
        if batch_idx == 0:
            print(f"  ✅ データを{device}に転送完了")
            print(f"  📊 input_ids device: {input_ids.device}")
            print(f"  📊 sentiment_true device: {sentiment_true.device}")
        
        # 勾配をゼロ化
        optimizer.zero_grad()
        
        # 予測
        if batch_idx == 0:
            print(f"  🔮 モデル予測開始...", flush=True)
            print(f"  📊 input_ids shape: {input_ids.shape}")
            print(f"  📊 attention_mask shape: {attention_mask.shape}")
            print(f"  📊 model device: {next(model.parameters()).device}")
        
        try:
            # モデルを明示的にtrainモードに設定
            if batch_idx == 0:
                print(f"  🔧 モデルをtrainモードに設定...", flush=True)
                model.train()
                print(f"  ✅ trainモード設定完了", flush=True)
                
                # モデルパラメータのデバイス確認
                print(f"  🔍 モデルパラメータのデバイス確認...", flush=True)
                cpu_params = []
                for name, param in model.named_parameters():
                    if not param.is_cuda:
                        cpu_params.append(name)
                if cpu_params:
                    print(f"  ⚠️ CPU上にあるパラメータ: {cpu_params[:3]}...", flush=True)
                    print(f"  🔧 モデルを再びGPUに転送...", flush=True)
                    model.to(device)
                    print(f"  ✅ GPU転送完了", flush=True)
                else:
                    print(f"  ✅ 全パラメータがGPU上にあります", flush=True)
                
                # 入力テンソルの型確認・修正
                print(f"  🔍 入力テンソルの型確認...", flush=True)
                print(f"  📊 input_ids dtype: {input_ids.dtype}")
                print(f"  📊 attention_mask dtype: {attention_mask.dtype}")
                
                # 型を明示的にlongに変換
                input_ids = input_ids.long()
                attention_mask = attention_mask.long()
                print(f"  ✅ 入力テンソルをlong型に変換完了", flush=True)
            
            # forward実行
            if batch_idx == 0:
                print(f"  🚀 forward実行開始...", flush=True)
                print(f"  📊 input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
                print(f"  📊 attention_mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                sentiment_pred, course_pred = model(input_ids, attention_mask, chunk_mask)
            
            if batch_idx == 0:
                print(f"  ✅ モデル予測完了", flush=True)
                print(f"  📊 sentiment_pred shape: {sentiment_pred.shape}")
                print(f"  📊 course_pred shape: {course_pred.shape}")
                print(f"  📊 sentiment_pred device: {sentiment_pred.device}")
                print(f"  📊 sentiment_pred dtype: {sentiment_pred.dtype}")
        except Exception as e:
            print(f"  ❌ モデル予測エラー: {e}", flush=True)
            print(f"  🔍 エラー詳細:", flush=True)
            import traceback
            traceback.print_exc()
            print(f"  🛑 学習を停止します", flush=True)
            break
        
        # 損失計算
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            sentiment_loss = criterion(sentiment_pred, sentiment_true)
            course_loss = criterion(course_pred, course_true)
            loss = ALPHA * sentiment_loss + BETA * course_loss
        
        if batch_idx == 0:
            print(f"  📊 損失計算完了: {loss.item():.4f}")
        
        # 逆伝播
        if batch_idx == 0:
            print(f"  🔄 逆伝播開始...", flush=True)
        try:
            # 勾配蓄積
            loss_acc = loss / ACCUM_STEPS
            scaler.scale(loss_acc).backward()
            if batch_idx == 0:
                print(f"  ✅ 逆伝播完了", flush=True)
        except Exception as e:
            print(f"  ❌ 逆伝播エラー: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break
        
        # 勾配クリッピング
        if batch_idx == 0:
            print(f"  ✂️ 勾配クリッピング開始...", flush=True)
        # 勾配更新のタイミングでのみクリップとstep
        if (batch_idx + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if batch_idx == 0:
            print(f"  ✅ 勾配クリッピング完了", flush=True)
        
        # パラメータ更新
        if batch_idx == 0:
            print(f"  🔧 パラメータ更新開始...", flush=True)
        try:
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if batch_idx == 0:
                    print(f"  ✅ パラメータ更新完了", flush=True)
                    print(f"  🎉 最初のバッチ処理完了！学習を継続します...", flush=True)
        except Exception as e:
            print(f"  ❌ パラメータ更新エラー: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break
        
        # 損失の記録
        total_loss += loss.item()
        sentiment_losses += sentiment_loss.item()
        course_losses += course_loss.item()
        
        # 毎秒更新のリアルタイム進捗表示
        current_time = time.time()
        if current_time - last_update_time >= 1.0:  # 1秒ごとに更新
            print_progress_gauge(
                batch_idx + 1, len(train_loader),
                f"🔥 Epoch {epoch+1}/{num_epochs}",
                f"Loss: {loss.item():.4f} | Sent: {sentiment_loss.item():.4f} | Course: {course_loss.item():.4f}",
                True, True
            )
            last_update_time = current_time
    
    # 学習率の調整
    scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    avg_sentiment_loss = sentiment_losses / len(train_loader)
    avg_course_loss = course_losses / len(train_loader)
    
    return avg_loss, avg_sentiment_loss, avg_course_loss


def validate(model, val_loader):
    """検証"""
    model.eval()
    total_loss = 0
    sentiment_losses = 0
    course_losses = 0
    
    sentiment_preds_list = []
    sentiment_true_list = []
    course_preds_list = []
    course_true_list = []
    
    criterion = nn.MSELoss()
    last_update_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # データをデバイスに転送
            input_ids = batch['input_ids'].to(device, non_blocking=NON_BLOCKING)
            attention_mask = batch['attention_mask'].to(device, non_blocking=NON_BLOCKING)
            chunk_mask = batch['chunk_mask'].to(device, non_blocking=NON_BLOCKING)
            sentiment_true = batch['sentiment_score'].to(device, non_blocking=NON_BLOCKING)
            course_true = batch['course_score'].to(device, non_blocking=NON_BLOCKING)
            
            # 予測
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                sentiment_pred, course_pred = model(input_ids, attention_mask, chunk_mask)
            
            # 損失計算
            sentiment_loss = criterion(sentiment_pred, sentiment_true)
            course_loss = criterion(course_pred, course_true)
            loss = ALPHA * sentiment_loss + BETA * course_loss
            
            # 損失の記録
            total_loss += loss.item()
            sentiment_losses += sentiment_loss.item()
            course_losses += course_loss.item()
            
            # 予測値の記録
            sentiment_preds_list.extend(sentiment_pred.cpu().numpy())
            sentiment_true_list.extend(sentiment_true.cpu().numpy())
            course_preds_list.extend(course_pred.cpu().numpy())
            course_true_list.extend(course_true.cpu().numpy())
            
            # 毎秒更新のリアルタイム検証進捗表示
            current_time = time.time()
            if current_time - last_update_time >= 1.0:  # 1秒ごとに更新
                print_progress_gauge(
                    batch_idx + 1, len(val_loader),
                    "✅ Validation",
                    f"Loss: {loss.item():.4f} | Sent: {sentiment_loss.item():.4f} | Course: {course_loss.item():.4f}",
                    True, True
                )
                last_update_time = current_time
    
    avg_loss = total_loss / len(val_loader)
    avg_sentiment_loss = sentiment_losses / len(val_loader)
    avg_course_loss = course_losses / len(val_loader)
    
    # 評価指標の計算
    sentiment_preds = np.array(sentiment_preds_list)
    sentiment_true = np.array(sentiment_true_list)
    course_preds = np.array(course_preds_list)
    course_true = np.array(course_true_list)
    
    sentiment_r2 = r2_score(sentiment_true, sentiment_preds)
    sentiment_corr = np.corrcoef(sentiment_true, sentiment_preds)[0, 1]
    
    course_r2 = r2_score(course_true, course_preds)
    course_corr = np.corrcoef(course_true, course_preds)[0, 1]
    
    return avg_loss, avg_sentiment_loss, avg_course_loss, sentiment_r2, sentiment_corr, course_r2, course_corr


def train_model(model, train_loader, val_loader, num_epochs):
    """モデルの学習"""
    print("\n" + "="*60)
    print("🚀 授業レベルマルチタスク学習を開始")
    print("="*60)
    
    # GPU初期状況の表示
    if torch.cuda.is_available():
        gpu_status = get_gpu_status()
        if gpu_status:
            print(f"🚀 GPU初期状況:")
            print(f"   📊 メモリ: {gpu_status['allocated']:.1f}GB / {gpu_status['total']:.1f}GB")
            print(f"   🔒 予約済み: {gpu_status['reserved']:.1f}GB")
            print(f"   📈 使用率: {gpu_status['utilization']:.1f}%")
    else:
        print("⚠️  GPUが利用できません。CPUで実行中...")
    
    # 開始時刻を記録
    start_time = datetime.now()
    
    # オプティマイザとスケジューラ
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # 学習履歴
    history = {
        'train_loss': [],
        'train_sentiment_loss': [],
        'train_course_loss': [],
        'val_loss': [],
        'val_sentiment_loss': [],
        'val_course_loss': [],
        'val_sentiment_r2': [],
        'val_sentiment_corr': [],
        'val_course_r2': [],
        'val_course_corr': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"🎯 Epoch {epoch+1}/{num_epochs} 開始")
        print(f"{'='*60}")
        
        # 学習
        print(f"\n🔥 学習フェーズ開始...")
        train_loss, train_sent_loss, train_course_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, epoch, num_epochs
        )
        
        # 検証
        print(f"\n✅ 検証フェーズ開始...")
        val_loss, val_sent_loss, val_course_loss, sent_r2, sent_corr, course_r2, course_corr = validate(
            model, val_loader
        )
        
        # 履歴の記録
        history['train_loss'].append(train_loss)
        history['train_sentiment_loss'].append(train_sent_loss)
        history['train_course_loss'].append(train_course_loss)
        history['val_loss'].append(val_loss)
        history['val_sentiment_loss'].append(val_sent_loss)
        history['val_course_loss'].append(val_course_loss)
        history['val_sentiment_r2'].append(sent_r2)
        history['val_sentiment_corr'].append(sent_corr)
        history['val_course_r2'].append(course_r2)
        history['val_course_corr'].append(course_corr)
        
        # ゲージ風サマリー表示
        current_lr = optimizer.param_groups[0]['lr']
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        print_epoch_summary(
            epoch + 1, num_epochs, train_loss, val_loss, 
            sent_r2, course_r2, best_val_loss, current_lr, elapsed_time
        )
        
        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # ベストモデルをロード
    model.load_state_dict(best_model_state)
    
    return model, history


def evaluate_model(model, test_loader, sentiment_scaler, course_scaler):
    """モデルの評価"""
    print("\n" + "="*60)
    print("📊 テストデータでの最終評価")
    print("="*60)
    
    model.eval()
    
    sentiment_preds_list = []
    sentiment_true_list = []
    course_preds_list = []
    course_true_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device, non_blocking=NON_BLOCKING)
            attention_mask = batch['attention_mask'].to(device, non_blocking=NON_BLOCKING)
            chunk_mask = batch['chunk_mask'].to(device, non_blocking=NON_BLOCKING)
            sentiment_true = batch['sentiment_score'].to(device, non_blocking=NON_BLOCKING)
            course_true = batch['course_score'].to(device, non_blocking=NON_BLOCKING)
            
            # 予測
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                sentiment_pred, course_pred = model(input_ids, attention_mask, chunk_mask)
            
            # 予測値の記録
            sentiment_preds_list.extend(sentiment_pred.cpu().numpy())
            sentiment_true_list.extend(sentiment_true.cpu().numpy())
            course_preds_list.extend(course_pred.cpu().numpy())
            course_true_list.extend(course_true.cpu().numpy())
            
            # 進捗表示（ゲージ風）
            if batch_idx % 2 == 0:
                print_progress_gauge(
                    batch_idx + 1, len(test_loader),
                    "📊 テスト評価",
                    "",
                    True
                )
    
    # numpy配列に変換
    sentiment_preds = np.array(sentiment_preds_list)
    sentiment_true = np.array(sentiment_true_list)
    course_preds = np.array(course_preds_list)
    course_true = np.array(course_true_list)
    
    # 正規化を戻す
    sentiment_preds_original = sentiment_scaler.inverse_transform(sentiment_preds.reshape(-1, 1)).flatten()
    sentiment_true_original = sentiment_scaler.inverse_transform(sentiment_true.reshape(-1, 1)).flatten()
    course_preds_original = course_scaler.inverse_transform(course_preds.reshape(-1, 1)).flatten()
    course_true_original = course_scaler.inverse_transform(course_true.reshape(-1, 1)).flatten()
    
    # 評価指標の計算
    results = {
        'sentiment': {
            'rmse': float(np.sqrt(mean_squared_error(sentiment_true_original, sentiment_preds_original))),
            'mae': float(mean_absolute_error(sentiment_true_original, sentiment_preds_original)),
            'r2': float(r2_score(sentiment_true_original, sentiment_preds_original)),
            'correlation': float(np.corrcoef(sentiment_true_original, sentiment_preds_original)[0, 1])
        },
        'course': {
            'rmse': float(np.sqrt(mean_squared_error(course_true_original, course_preds_original))),
            'mae': float(mean_absolute_error(course_true_original, course_preds_original)),
            'r2': float(r2_score(course_true_original, course_preds_original)),
            'correlation': float(np.corrcoef(course_true_original, course_preds_original)[0, 1])
        }
    }
    
    # 結果表示
    print("\n感情スコア予測の結果:")
    print(f"  RMSE: {results['sentiment']['rmse']:.4f}")
    print(f"  MAE: {results['sentiment']['mae']:.4f}")
    print(f"  R²: {results['sentiment']['r2']:.4f}")
    print(f"  相関係数: {results['sentiment']['correlation']:.4f}")
    
    print("\n授業評価スコア予測の結果:")
    print(f"  RMSE: {results['course']['rmse']:.4f}")
    print(f"  MAE: {results['course']['mae']:.4f}")
    print(f"  R²: {results['course']['r2']:.4f}")
    print(f"  相関係数: {results['course']['correlation']:.4f}")
    
    return results, sentiment_preds_original, sentiment_true_original, course_preds_original, course_true_original


def save_results(model, history, results, timestamp):
    """結果の保存"""
    print("\n" + "="*60)
    print("💾 結果の保存")
    print("="*60)
    
    # 保存ディレクトリの作成（プロジェクト直下）
    output_dir = os.path.join('02_モデル', '授業レベルマルチタスクモデル')
    os.makedirs(output_dir, exist_ok=True)
    
    # モデルの保存
    model_path = os.path.join(output_dir, 'best_class_level_multitask_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"モデルを保存: {model_path}")
    
    # 設定の保存
    config = {
        'model_type': 'ClassLevelMultitaskModel',
        'base_model': BASE_MODEL,
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'alpha': ALPHA,
        'beta': BETA,
        'data_level': 'class_level',
        'data_size': 3268
    }
    
    config_path = os.path.join(output_dir, 'model_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"設定を保存: {config_path}")
    
    # 結果の保存（プロジェクト直下）
    results_dir = os.path.join('03_分析結果', '授業レベルマルチタスク学習')
    os.makedirs(results_dir, exist_ok=True)
    
    results_data = {
        'timestamp': timestamp,
        'data_level': 'class_level',
        'data_size': 3268,
        'results': results,
        'training_history': history
    }
    
    results_path = os.path.join(results_dir, f'class_level_multitask_results_{timestamp}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"結果を保存: {results_path}")


def create_visualizations(history, sentiment_preds, sentiment_true, course_preds, course_true, timestamp):
    """可視化の作成"""
    print("\n" + "="*60)
    print("📊 可視化の作成")
    print("="*60)
    
    # 1. 学習曲線
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # R²スコア
    axes[0, 1].plot(history['val_sentiment_r2'], label='Sentiment R2', marker='o')
    axes[0, 1].plot(history['val_course_r2'], label='Course R2', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R2 Score')
    axes[0, 1].set_title('R2 Score Progress')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 相関係数
    axes[1, 0].plot(history['val_sentiment_corr'], label='Sentiment Correlation', marker='o')
    axes[1, 0].plot(history['val_course_corr'], label='Course Correlation', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('Correlation Progress')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # タスク別損失
    axes[1, 1].plot(history['train_sentiment_loss'], label='Train Sentiment Loss', marker='o')
    axes[1, 1].plot(history['train_course_loss'], label='Train Course Loss', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Task-wise Training Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_dir = os.path.join('03_分析結果', '授業レベルマルチタスク学習')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'training_curves_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"学習曲線を保存しました")
    plt.close()
    
    # 2. 予測vs真値の散布図
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 感情スコア
    axes[0].scatter(sentiment_true, sentiment_preds, alpha=0.6, s=20)
    axes[0].plot([sentiment_true.min(), sentiment_true.max()], 
                 [sentiment_true.min(), sentiment_true.max()], 
                 'r--', label='Perfect Prediction')
    axes[0].set_xlabel('True Sentiment Score')
    axes[0].set_ylabel('Predicted Sentiment Score')
    axes[0].set_title(f'Sentiment Score Prediction (R²={r2_score(sentiment_true, sentiment_preds):.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 授業評価スコア
    axes[1].scatter(course_true, course_preds, alpha=0.6, s=20)
    axes[1].plot([course_true.min(), course_true.max()], 
                 [course_true.min(), course_true.max()], 
                 'r--', label='Perfect Prediction')
    axes[1].set_xlabel('True Course Evaluation Score')
    axes[1].set_ylabel('Predicted Course Evaluation Score')
    axes[1].set_title(f'Course Score Prediction (R²={r2_score(course_true, course_preds):.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'prediction_scatter_{timestamp}.png'), dpi=300, bbox_inches='tight')
    print(f"散布図を保存しました")
    plt.close()


def main():
    """メイン関数"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 実行ディレクトリを適切に設定
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # 00_スクリプトの親ディレクトリ
        os.chdir(project_root)
        print(f"📁 実行ディレクトリ: {os.getcwd()}")
        
        print("\n" + "="*60)
        print("🎯 授業レベルマルチタスク学習")
        print("="*60)
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # PyTorchとDirectMLのバージョン情報を表示
        print(f"\n🧠 PyTorch version: {torch.__version__}")
        try:
            import torch_directml as dml
            print(f"🧩 DirectML available: {dml.is_available()}")
            if torch.__version__.startswith("2.4"):
                print("⚠️ PyTorch 2.4.x は DirectML と互換性の問題があります")
                print("   推奨: PyTorch 2.2.2 + torch-directml 0.2.3.dev240715")
        except ImportError:
            print("ℹ️ DirectML がインストールされていません")
        
        # トークナイザーの初期化
        print("\n🔧 トークナイザーの初期化...")
        try:
            tokenizer = BertJapaneseTokenizer.from_pretrained(BASE_MODEL)
            print("✅ 日本語BERTトークナイザーの初期化が完了しました")
            # 長文チャンク化を自前で行うため、警告抑制用にモデル長を拡張
            try:
                tokenizer.model_max_length = 10**6
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️  日本語BERTトークナイザーの初期化に失敗: {e}")
            print("🔧 代替トークナイザーを使用します...")
            try:
                # 代替としてAutoTokenizerを使用
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
                print("✅ 代替トークナイザーの初期化が完了しました")
            except Exception as e2:
                print(f"❌ 代替トークナイザーも失敗: {e2}")
                print("💡 必要なライブラリをインストールしてください:")
                print("   pip install fugashi ipadic unidic-lite")
                raise e2
        
        # データの読み込み
        texts, sentiment_scores, course_scores = load_data()
        
        # データの準備
        train_loader, val_loader, test_loader, sentiment_scaler, course_scaler = prepare_data(
            texts, sentiment_scores, course_scores, tokenizer
        )
        
        # モデルの初期化
        print("\n🔧 モデルの初期化...")
        print(f"📥 ベースモデル読み込み中: {BASE_MODEL}")
        
        try:
            model = ClassLevelMultitaskModel(BASE_MODEL)
            print(f"✅ ベースモデル読み込み完了")
            
            print(f"🚀 モデルをGPUに移動中...")
            model = model.to(device)
            print(f"✅ モデルGPU移動完了")
            if WARMUP_FORWARD:
                # 事前ウォームアップ（初回forwardのコンパイル/初期化待ちを先に消化）
                try:
                    model.eval()
                    with torch.no_grad():
                        dummy_ids = torch.zeros((1, 1, CHUNK_LEN), dtype=torch.long, device=device)
                        dummy_mask = torch.zeros((1, 1, CHUNK_LEN), dtype=torch.long, device=device)
                        dummy_cmask = torch.tensor([[1]], dtype=torch.bool, device=device)
                        _ = model(dummy_ids, dummy_mask, dummy_cmask)
                    model.train()
                    print("🔥 ウォームアップforward完了")
                except Exception as _e:
                    print(f"ℹ️ ウォームアップforward失敗: {_e}")

            # 勾配チェックポイントでVRAM節約
            if USE_GRADIENT_CHECKPOINTING:
                try:
                    model.bert.gradient_checkpointing_enable()
                    print("🧠 Gradient Checkpointing 有効化")
                except Exception as _e:
                    print(f"ℹ️ Gradient Checkpointing 無効（非対応）: {_e}")
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"総パラメータ数: {total_params:,}")
            print(f"学習可能パラメータ数: {trainable_params:,}")
            
        except Exception as e:
            print(f"❌ モデル初期化エラー: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 学習
        model, history = train_model(model, train_loader, val_loader, NUM_EPOCHS)
        
        # 評価
        results, sentiment_preds, sentiment_true, course_preds, course_true = evaluate_model(
            model, test_loader, sentiment_scaler, course_scaler
        )
        
        # 結果の保存
        save_results(model, history, results, timestamp)
        
        # 可視化
        create_visualizations(history, sentiment_preds, sentiment_true, 
                            course_preds, course_true, timestamp)
        
        print("\n" + "="*60)
        print("✅ 授業レベルマルチタスク学習が完了しました！")
        print("="*60)
        print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 最終結果のサマリー
        print("\n📊 最終結果のサマリー")
        print("="*60)
        print("感情スコア予測:")
        print(f"  R²: {results['sentiment']['r2']:.4f}")
        print(f"  相関係数: {results['sentiment']['correlation']:.4f}")
        print(f"  RMSE: {results['sentiment']['rmse']:.4f}")
        
        print("\n授業評価スコア予測:")
        print(f"  R²: {results['course']['r2']:.4f}")
        print(f"  相関係数: {results['course']['correlation']:.4f}")
        print(f"  RMSE: {results['course']['rmse']:.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


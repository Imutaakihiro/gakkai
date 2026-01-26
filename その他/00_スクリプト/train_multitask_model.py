#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汎用的なマルチタスク学習スクリプト
multitask_model.pyで定義されたモデルを使用して学習を実行
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer

from multitask_model import MultitaskModel, MultitaskLoss

# ------------------------- 設定 -------------------------
BASE_MODEL = "koheiduck/bert-japanese-finetuned-sentiment"
MAX_LENGTH = 256
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
USE_AMP = True
CHUNK_LEN = 256
STRIDE = 200
MAX_CHUNKS = 10
NUM_WORKERS = 0  # Windows 安全設定

# タスク設定（カスタマイズ可能）
TASK_CONFIGS = [
    {
        "name": "sentiment",
        "type": "regression",
        "output_size": 1,
        "hidden_sizes": [256],
        "weight": 0.5,
        "activation": "relu"
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

# ------------------------- デバイス選択 -------------------------

def get_device() -> torch.device:
    """利用可能なデバイスを自動選択"""
    # CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0]).cuda()
            print("✅ CUDA 利用")
            return torch.device("cuda")
        except Exception:
            pass
    # MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            _ = torch.tensor([1.0]).to("mps")
            print("✅ MPS (Apple Silicon) 利用")
            return torch.device("mps")
        except Exception:
            pass
    # DirectML (Windows)
    try:
        import torch_directml as dml
        if dml.is_available():
            print("✅ DirectML 利用")
            return dml.device()
    except Exception:
        pass
    print("🔄 CPU 利用")
    return torch.device("cpu")


device = get_device()
IS_CUDA = torch.cuda.is_available()
AMP_ENABLED = USE_AMP and IS_CUDA

# ------------------------- データ読み込み -------------------------

AGG_DIR_CANDIDATES = [
    "01_データ/マルチタスク用データ",
    "../01_データ/マルチタスク用データ",
    "../../01_データ/マルチタスク用データ",
]


def find_latest_agg_csv() -> str:
    """最新の集約データセットCSVを検索"""
    preferred = [
        "01_データ/マルチタスク用データ/授業集約データセット 回答分布付き.csv",
        "../01_データ/マルチタスク用データ/授業集約データセット 回答分布付き.csv",
        "../../01_データ/マルチタスク用データ/授業集約データセット 回答分布付き.csv",
    ]
    for p in preferred:
        if os.path.exists(p):
            print(f"📁 使用CSV(優先): {p}")
            return p

    paths = []
    for base in AGG_DIR_CANDIDATES:
        paths.extend(glob.glob(os.path.join(base, "授業集約データセット_*.csv")))
    if not paths:
        for root, _, files in os.walk("."):
            for f in files:
                if f.startswith("授業集約データセット_") and f.endswith(".csv"):
                    paths.append(os.path.join(root, f))
    if not paths:
        raise FileNotFoundError("授業集約データセット_*.csv が見つかりません")
    latest = max(paths, key=os.path.getctime)
    print(f"📁 使用CSV: {latest}")
    return latest


# 列名候補
COURSE_ID_CANDS = ["course_id", "授業ID", "科目ID", "講義ID", "courseId"]
TEXT_CANDS = ["自由記述まとめ", "text", "自由記述", "comments"]
SENT_MEAN_CANDS = ["感情スコア平均", "sentiment_mean"]
COURSE_SCORE_CANDS = ["授業評価スコア", "course_score"]


def pick_first_exist(df: pd.DataFrame, candidates: List[str]) -> str:
    """候補から最初に見つかった列名を返す"""
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def prepare_targets(df: pd.DataFrame, task_configs: List[Dict]) -> Dict[str, np.ndarray]:
    """タスク設定に基づいてターゲットを準備"""
    targets = {}
    
    for task_config in task_configs:
        task_name = task_config["name"]
        task_type = task_config.get("type", "regression")
        
        if task_name == "sentiment":
            col = pick_first_exist(df, SENT_MEAN_CANDS)
            if col:
                targets[task_name] = df[col].astype(np.float32).values
        elif task_name == "course_score":
            col = pick_first_exist(df, COURSE_SCORE_CANDS)
            if col:
                targets[task_name] = df[col].astype(np.float32).values
        else:
            # カスタムタスク: 列名を設定から取得
            col_name = task_config.get("column_name")
            if col_name and col_name in df.columns:
                if task_type == "classification":
                    targets[task_name] = df[col_name].astype(np.int64).values
                else:
                    targets[task_name] = df[col_name].astype(np.float32).values
    
    return targets


def load_agg_dataframe() -> pd.DataFrame:
    """集約データセットを読み込み"""
    path = find_latest_agg_csv()
    df = pd.read_csv(path)
    print(f"行数: {len(df)}, 列: {len(df.columns)}")
    return df

# ------------------------- データセット -------------------------

class MultitaskDataset(Dataset):
    """マルチタスク用データセット"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BertJapaneseTokenizer,
        task_configs: List[Dict],
        chunk_len: int = CHUNK_LEN,
        stride: int = STRIDE,
        max_chunks: int = MAX_CHUNKS
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.stride = stride
        self.max_chunks = max_chunks
        self.task_configs = {cfg["name"]: cfg for cfg in task_configs}
        
        self.course_col = pick_first_exist(df, COURSE_ID_CANDS)
        self.text_col = pick_first_exist(df, TEXT_CANDS)
        if not self.text_col:
            raise ValueError(f"テキスト列が見つかりません。候補: {TEXT_CANDS}")
        
        self.targets = prepare_targets(df, task_configs)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
    
    def __len__(self):
        return len(self.df)
    
    def _chunk(self, text: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """テキストをチャンク化"""
        ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        inner_max = self.chunk_len - 2
        chunks: List[Tuple[List[int], List[int]]] = []
        
        if len(ids) == 0:
            tokens = [self.cls_id, self.sep_id]
            pad = self.chunk_len - 2
            input_ids = tokens + [0] * pad
            attn = [1, 1] + [0] * pad
            return [torch.tensor(input_ids, dtype=torch.long)], [torch.tensor(attn, dtype=torch.long)]
        
        for start in range(0, len(ids), self.stride):
            inner = ids[start:start + inner_max]
            toks = [self.cls_id] + inner + [self.sep_id]
            if len(toks) < self.chunk_len:
                pad = self.chunk_len - len(toks)
                input_ids = toks + [0] * pad
                attn = [1] * len(toks) + [0] * pad
            else:
                input_ids = toks[:self.chunk_len]
                attn = [1] * self.chunk_len
            chunks.append((input_ids, attn))
            if len(chunks) >= self.max_chunks:
                break
        
        input_ids_list = [torch.tensor(x[0], dtype=torch.long) for x in chunks]
        attention_list = [torch.tensor(x[1], dtype=torch.long) for x in chunks]
        return input_ids_list, attention_list
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row[self.text_col]
        input_ids_list, attn_list = self._chunk(text)
        course_id = row[self.course_col] if self.course_col else idx
        
        item = {
            "input_ids_list": input_ids_list,
            "attention_mask_list": attn_list,
            "num_chunks": len(input_ids_list),
            "course_id": course_id,
        }
        
        # 各タスクのターゲットを追加
        for task_name in self.targets.keys():
            item[task_name] = torch.tensor(
                self.targets[task_name][idx],
                dtype=torch.float if self.task_configs[task_name].get("type") != "classification" else torch.long
            )
        
        return item


def collate_batch(batch):
    """バッチをまとめる"""
    B = len(batch)
    C = MAX_CHUNKS
    L = CHUNK_LEN
    
    input_ids = torch.zeros((B, C, L), dtype=torch.long)
    attention_mask = torch.zeros((B, C, L), dtype=torch.long)
    chunk_mask = torch.zeros((B, C), dtype=torch.bool)
    
    # タスク名を取得
    task_names = [k for k in batch[0].keys() if k not in ["input_ids_list", "attention_mask_list", "num_chunks", "course_id"]]
    targets = {task_name: [] for task_name in task_names}
    course_ids = []
    
    for i, item in enumerate(batch):
        n = min(item["num_chunks"], C)
        for j in range(n):
            input_ids[i, j] = item["input_ids_list"][j]
            attention_mask[i, j] = item["attention_mask_list"][j]
            chunk_mask[i, j] = True
        
        for task_name in task_names:
            targets[task_name].append(item[task_name])
        course_ids.append(item["course_id"])
    
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "chunk_mask": chunk_mask,
        "course_ids": course_ids,
    }
    
    # ターゲットをテンソルに変換
    for task_name, values in targets.items():
        if len(values) > 0:
            if isinstance(values[0], torch.Tensor):
                result[task_name] = torch.stack(values)
            else:
                result[task_name] = torch.tensor(values)
    
    return result

# ------------------------- 学習ループ -------------------------

def train_loop(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scaler,
    criterion: MultitaskLoss,
    epoch: int
):
    """学習ループ"""
    model.train()
    total_loss = 0.0
    task_losses_dict = {task_name: 0.0 for task_name in model.task_heads.keys()}
    
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        chunk_mask = batch["chunk_mask"].to(device)
        
        # ターゲットを取得
        targets = {
            task_name: batch[task_name].to(device)
            for task_name in model.task_heads.keys()
            if task_name in batch
        }
        
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            predictions = model(input_ids, attention_mask, chunk_mask)
            loss, task_losses = criterion(predictions, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += float(loss.item())
        for task_name, task_loss in task_losses.items():
            task_losses_dict[task_name] += float(task_loss.item())
        
        if (step + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Step {step+1}/{len(loader)} | Loss {total_loss/(step+1):.4f}")
    
    avg_loss = total_loss / max(1, len(loader))
    avg_task_losses = {k: v / max(1, len(loader)) for k, v in task_losses_dict.items()}
    return avg_loss, avg_task_losses


def eval_loop(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultitaskLoss
):
    """評価ループ"""
    model.eval()
    total_loss = 0.0
    task_losses_dict = {task_name: 0.0 for task_name in model.task_heads.keys()}
    
    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            chunk_mask = batch["chunk_mask"].to(device)
            
            targets = {
                task_name: batch[task_name].to(device)
                for task_name in model.task_heads.keys()
                if task_name in batch
            }
            
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                predictions = model(input_ids, attention_mask, chunk_mask)
                loss, task_losses = criterion(predictions, targets)
            
            total_loss += float(loss.item())
            for task_name, task_loss in task_losses.items():
                task_losses_dict[task_name] += float(task_loss.item())
    
    avg_loss = total_loss / max(1, len(loader))
    avg_task_losses = {k: v / max(1, len(loader)) for k, v in task_losses_dict.items()}
    return avg_loss, avg_task_losses

# ------------------------- メイン -------------------------

def main():
    """メイン関数"""
    # ルートに移動
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        os.chdir(project_root)
    except Exception:
        pass
    print(f"📂 CWD: {os.getcwd()}")
    
    # トークナイザー
    tokenizer = BertJapaneseTokenizer.from_pretrained(BASE_MODEL)
    try:
        tokenizer.model_max_length = 10**6
    except Exception:
        pass
    
    # データ読込
    df = load_agg_dataframe()
    
    # 分割
    rng = np.random.RandomState(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    
    train_ds = MultitaskDataset(df.iloc[train_idx].copy(), tokenizer, TASK_CONFIGS)
    val_ds = MultitaskDataset(df.iloc[val_idx].copy(), tokenizer, TASK_CONFIGS)
    test_ds = MultitaskDataset(df.iloc[test_idx].copy(), tokenizer, TASK_CONFIGS)
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_batch
    )
    
    # モデル
    model = MultitaskModel(BASE_MODEL, TASK_CONFIGS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    criterion = MultitaskLoss(TASK_CONFIGS)
    
    # 学習
    best_val = float("inf")
    best_state = None
    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_task_losses = train_loop(model, train_loader, optimizer, scaler, criterion, epoch)
        val_loss, val_task_losses = eval_loop(model, val_loader, criterion)
        print(f"[Epoch {epoch+1}] Train {tr_loss:.4f} | Val {val_loss:.4f}")
        print(f"  タスク別損失: {val_task_losses}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # テスト
    test_loss, test_task_losses = eval_loop(model, test_loader, criterion)
    print(f"🧪 Test Loss: {test_loss:.4f}")
    print(f"   タスク別損失: {test_task_losses}")
    
    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("02_モデル", "マルチタスクモデル")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"multitask_model_{ts}.pth")
    torch.save(model.state_dict(), model_path)
    
    results = {
        "timestamp": ts,
        "test_loss": float(test_loss),
        "test_task_losses": {k: float(v) for k, v in test_task_losses.items()},
        "base_model": BASE_MODEL,
        "task_configs": TASK_CONFIGS,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
    }
    with open(os.path.join(out_dir, f"multitask_model_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 保存: {model_path}")


if __name__ == "__main__":
    main()

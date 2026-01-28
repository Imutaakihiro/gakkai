#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
授業単位 × 順序回帰（1–4）× LLP 学習スクリプト（最小実装）
- 入力: 授業ごとの自由記述（複数件でも可）
- 教師: 授業ごとの分布 q = [q1..q4]（または count_1..4 → 比率化）、回答者数 respondents
- 出力: 累積 3 ロジット → P(1..4) 再構成
- 損失: 授業内平均の予測分布 p̄ を用いた respondents 加重 KL(q || p̄)
"""

import os
import glob
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertJapaneseTokenizer

# ------------------------- 基本設定 -------------------------
BASE_MODEL = "koheiduck/bert-japanese-finetuned-sentiment"
MAX_LENGTH = 192
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
USE_AMP = True
CHUNK_LEN = 192
STRIDE = 150
MAX_CHUNKS = 4
NUM_WORKERS = 0  # Windows 安全設定

# マルチタスク損失の係数（LLP + 回帰1本）
ALPHA_SENT = 0.3   # 感情スコア平均の重み（補助情報なので小さめ）
# 授業評価スコアは期待値から計算するため、回帰ヘッドと損失は不要

# ------------------------- デバイス選択 -------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0]).cuda()
            print("✅ CUDA 利用")
            return torch.device("cuda")
        except Exception:
            pass
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
    # まず優先ファイル（回答分布付き）を探す
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
        # 広域探索
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
COUNT_PREFIXES = [["count_1", "count_2", "count_3", "count_4"],
                  ["n1", "n2", "n3", "n4"],
                  ["人数_1", "人数_2", "人数_3", "人数_4"],
                  ["分布_十分意義あり_人数", "分布_ある程度意義あり_人数", "分布_あまり意義なし_人数", "分布_全く意義なし_人数"]]
RATIO_PREFIXES = [["ratio_1", "ratio_2", "ratio_3", "ratio_4"],
                  ["r1", "r2", "r3", "r4"],
                  ["割合_1", "割合_2", "割合_3", "割合_4"],
                  ["分布_十分意義あり_割合(%)", "分布_ある程度意義あり_割合(%)", "分布_あまり意義なし_割合(%)", "分布_全く意義なし_割合(%)"]]
RESPONDENTS_CANDS = ["respondents", "回答者数", "n_respondents", "人数合計"]

# 回帰ターゲット列の候補
SENT_MEAN_CANDS = ["感情スコア平均", "sentiment_mean"]
COURSE_SCORE_CANDS = ["授業評価スコア", "course_score"]


def pick_first_exist(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def pick_block(df: pd.DataFrame, blocks: List[List[str]]) -> List[str]:
    for block in blocks:
        if all(c in df.columns for c in block):
            return block
    return []


def prepare_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 優先: ratio_* → 0-1に正規化、次: count_* → 合計で割る
    ratio_cols = pick_block(df, RATIO_PREFIXES)
    count_cols = pick_block(df, COUNT_PREFIXES)
    if ratio_cols:
        q = df[ratio_cols].values.astype(np.float32)
        # パーセント形式なら0-1へ
        if np.nanmax(q) > 1.5:
            q = q / 100.0
        q = np.clip(q, 1e-8, 1.0)
        q = q / q.sum(axis=1, keepdims=True)
        respondents_col = pick_first_exist(df, RESPONDENTS_CANDS)
        if respondents_col:
            w = df[respondents_col].fillna(1).astype(np.float32).values
        else:
            w = np.ones((len(df),), dtype=np.float32)
    elif count_cols:
        counts = df[count_cols].values.astype(np.float32)
        total = counts.sum(axis=1, keepdims=True).clip(min=1.0)
        q = counts / total
        w = counts.sum(axis=1).astype(np.float32)
    else:
        raise ValueError("ratio_1..4 / count_1..4 / 分布_* 列が必要です")

    sent_col = pick_first_exist(df, SENT_MEAN_CANDS)
    course_col = pick_first_exist(df, COURSE_SCORE_CANDS)
    if not sent_col or not course_col:
        raise ValueError("感情スコア平均 / 授業評価スコア の列が見つかりません")
    y_sent = df[sent_col].astype(np.float32).values
    y_course = df[course_col].astype(np.float32).values
    return q, w, y_sent, y_course


def load_agg_dataframe() -> pd.DataFrame:
    path = find_latest_agg_csv()
    df = pd.read_csv(path)
    print(f"行数: {len(df)}, 列: {len(df.columns)}")
    return df

# ------------------------- データセット -------------------------

class CourseLLPDataset(Dataset):
    """授業単位LLP用データセット（テキストはチャンク化・course_idで集約可）"""

    def __init__(self, df: pd.DataFrame, tokenizer: BertJapaneseTokenizer,
                 chunk_len: int = CHUNK_LEN, stride: int = STRIDE, max_chunks: int = MAX_CHUNKS):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.stride = stride
        self.max_chunks = max_chunks

        self.course_col = pick_first_exist(df, COURSE_ID_CANDS)
        self.text_col = pick_first_exist(df, TEXT_CANDS)
        if not self.text_col:
            raise ValueError(f"テキスト列が見つかりません。候補: {TEXT_CANDS}")

        self.q, self.w, self.y_sent, self.y_course = prepare_targets(df)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

    def __len__(self):
        return len(self.df)

    def _chunk(self, text: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
        q = torch.tensor(self.q[idx], dtype=torch.float)
        w = torch.tensor(self.w[idx], dtype=torch.float)
        y_sent = torch.tensor(self.y_sent[idx], dtype=torch.float)
        y_course = torch.tensor(self.y_course[idx], dtype=torch.float)
        return {
            "input_ids_list": input_ids_list,
            "attention_mask_list": attn_list,
            "num_chunks": len(input_ids_list),
            "course_id": course_id,
            "q": q,    # [4]
            "w": w,    # scalar
            "y_sent": y_sent,
            "y_course": y_course
        }


def collate_batch(batch):
    B = len(batch)
    C = MAX_CHUNKS
    L = CHUNK_LEN
    input_ids = torch.zeros((B, C, L), dtype=torch.long)
    attention_mask = torch.zeros((B, C, L), dtype=torch.long)
    chunk_mask = torch.zeros((B, C), dtype=torch.bool)
    q = torch.zeros((B, 4), dtype=torch.float)
    w = torch.zeros((B,), dtype=torch.float)
    y_sent = torch.zeros((B,), dtype=torch.float)
    y_course = torch.zeros((B,), dtype=torch.float)
    course_ids = []

    for i, item in enumerate(batch):
        n = min(item["num_chunks"], C)
        for j in range(n):
            input_ids[i, j] = item["input_ids_list"][j]
            attention_mask[i, j] = item["attention_mask_list"][j]
            chunk_mask[i, j] = True
        q[i] = item["q"]
        w[i] = item["w"]
        y_sent[i] = item["y_sent"]
        y_course[i] = item["y_course"]
        course_ids.append(item["course_id"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "chunk_mask": chunk_mask,
        "q": q,
        "w": w,
        "y_sent": y_sent,
        "y_course": y_course,
        "course_ids": course_ids,
    }

# ------------------------- モデル -------------------------

class OrdinalHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 3)  # y>=2, y>=3, y>=4

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(features)          # [B,3]
        probs_ge = torch.sigmoid(logits)    # [B,3]
        return logits, probs_ge


class CourseOrdinalLLPModel(nn.Module):
    def __init__(self, base_model: str, dropout: float = 0.1):
        super().__init__()
        # transformersの互換性問題を回避
        try:
            # まずuse_safetensors=Falseで試す（古いtransformers対応）
            self.bert = BertModel.from_pretrained(base_model, use_safetensors=False)
        except Exception as e1:
            try:
                # use_safetensors=Trueで試す
                self.bert = BertModel.from_pretrained(base_model, use_safetensors=True)
            except Exception as e2:
                # 最後の手段：trust_remote_codeを追加
                try:
                    self.bert = BertModel.from_pretrained(base_model, trust_remote_code=True)
                except Exception as e3:
                    raise RuntimeError(f"BertModel読み込み失敗: {e1}, {e2}, {e3}")
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = OrdinalHead(hidden)
        # 回帰ヘッド（感情スコア平均のみ、補助情報）
        self.sent_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )
        # 授業評価スコアは期待値から計算するため、回帰ヘッドは不要

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, chunk_mask: torch.Tensor):
        # input: [B,C,L]
        if input_ids.dim() == 3:
            B, C, L = input_ids.shape
            x_ids = input_ids.view(B*C, L)
            x_mask = attention_mask.view(B*C, L)
            out = self.bert(input_ids=x_ids, attention_mask=x_mask)
            cls = out.last_hidden_state[:, 0, :].view(B, C, -1)  # [B,C,H]
            mask = chunk_mask.float().unsqueeze(-1)
            summed = (cls * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1e-6)
            pooled = summed / denom
        else:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits, p_ge = self.head(pooled)  # p_ge: [p_ge2, p_ge3, p_ge4]
        # 復元: P1..P4
        p1 = 1.0 - p_ge[:, 0]
        p2 = (p_ge[:, 0] - p_ge[:, 1]).clamp(min=0.0)
        p3 = (p_ge[:, 1] - p_ge[:, 2]).clamp(min=0.0)
        p4 = p_ge[:, 2]
        P = torch.stack([p1, p2, p3, p4], dim=1)  # [B,4]
        # 数値安定化・正規化
        P = P.clamp(min=1e-8)
        P = P / P.sum(dim=1, keepdim=True)
        # 回帰出力（感情スコア）
        y_sent_pred = self.sent_head(pooled).squeeze(-1)
        # 授業評価スコアは期待値から計算（回帰ヘッド不要）
        # E[y] = 1×P1 + 2×P2 + 3×P3 + 4×P4
        expected_values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=P.device)
        y_course_pred = (P @ expected_values).squeeze(-1)
        return logits, p_ge, P, y_sent_pred, y_course_pred

# ------------------------- 学習ループ（LLP） -------------------------

def groupby_course_mean(P: torch.Tensor, course_ids: List) -> Tuple[torch.Tensor, List]:
    # バッチ内で同一 course_id を平均
    unique_ids = []
    pbar_list = []
    id_to_indices: Dict = {}
    for i, cid in enumerate(course_ids):
        id_to_indices.setdefault(cid, []).append(i)
    for cid, idxs in id_to_indices.items():
        unique_ids.append(cid)
        pbar_list.append(P[idxs].mean(dim=0, keepdim=True))
    return torch.cat(pbar_list, dim=0), unique_ids


def gather_targets(q: torch.Tensor, w: torch.Tensor, course_ids: List, unique_ids: List) -> Tuple[torch.Tensor, torch.Tensor]:
    # unique ids に対応する q, w をまとめる（平均/合計）
    id_to_idx = {}
    for i, cid in enumerate(course_ids):
        id_to_idx.setdefault(cid, []).append(i)
    q_list, w_list = [], []
    for cid in unique_ids:
        idxs = id_to_idx[cid]
        q_list.append(q[idxs].mean(dim=0, keepdim=True))
        w_list.append(w[idxs].sum().unsqueeze(0))
    return torch.cat(q_list, dim=0), torch.cat(w_list, dim=0)


def kl_divergence(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    # KL(q || p) = sum q * (log q - log p)
    return (q * (q.clamp(1e-8).log() - p.clamp(1e-8).log())).sum(dim=1)


def train_loop(model: nn.Module, loader: DataLoader, optimizer, scaler, epoch: int):
    model.train()
    total_loss = 0.0
    last_print = 0.0
    mse = nn.MSELoss(reduction="none")
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        chunk_mask = batch["chunk_mask"].to(device)
        q = batch["q"].to(device)
        w = batch["w"].to(device)
        y_sent = batch["y_sent"].to(device)
        y_course = batch["y_course"].to(device)
        course_ids = batch["course_ids"]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            _, _, P, y_sent_pred, y_course_pred = model(input_ids, attention_mask, chunk_mask)
            pbar, uniq = groupby_course_mean(P, course_ids)
            q_bar, w_bar = gather_targets(q, w, course_ids, uniq)
            loss_per = kl_divergence(q_bar, pbar)  # [B_unique]
            kl_loss = (loss_per * w_bar).sum() / (w_bar.sum().clamp_min(1e-6))
            sent_loss = mse(y_sent_pred, y_sent).mean()
            # 授業評価スコアは期待値から計算するため、損失は不要
            loss = kl_loss + ALPHA_SENT * sent_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        if (step + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Step {step+1}/{len(loader)} | Loss {total_loss/(step+1):.4f}")
    return total_loss / max(1, len(loader))


def eval_loop(model: nn.Module, loader: DataLoader):
    model.eval()
    total_loss = 0.0
    mse = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            chunk_mask = batch["chunk_mask"].to(device)
            q = batch["q"].to(device)
            w = batch["w"].to(device)
            y_sent = batch["y_sent"].to(device)
            y_course = batch["y_course"].to(device)
            course_ids = batch["course_ids"]
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                _, _, P, y_sent_pred, y_course_pred = model(input_ids, attention_mask, chunk_mask)
                pbar, uniq = groupby_course_mean(P, course_ids)
                q_bar, w_bar = gather_targets(q, w, course_ids, uniq)
                loss_per = kl_divergence(q_bar, pbar)
                kl_loss = (loss_per * w_bar).sum() / (w_bar.sum().clamp_min(1e-6))
                sent_loss = mse(y_sent_pred, y_sent).mean()
                # 授業評価スコアは期待値から計算するため、損失は不要
                loss = kl_loss + ALPHA_SENT * sent_loss
            total_loss += float(loss.item())
    return total_loss / max(1, len(loader))

# ------------------------- メイン -------------------------

def main():
    # ルートに移動（00_スクリプト からの相対実行に対応）
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
        # 長文警告の抑制（自前でチャンク化するため上限を十分大きく）
        tokenizer.model_max_length = 10**6
    except Exception:
        pass

    # データ読込
    df = load_agg_dataframe()

    # 分割（授業単位の層化が望ましいが、最小実装としてランダム）
    rng = np.random.RandomState(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_ds = CourseLLPDataset(df.iloc[train_idx].copy(), tokenizer)
    val_ds = CourseLLPDataset(df.iloc[val_idx].copy(), tokenizer)
    test_ds = CourseLLPDataset(df.iloc[test_idx].copy(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_batch)

    # モデル
    model = CourseOrdinalLLPModel(BASE_MODEL).to(device)
    # 勾配チェックポイントはCUDA環境のみ有効化（DirectMLでは無効）
    try:
        if torch.cuda.is_available() and hasattr(model.bert, "gradient_checkpointing_enable"):
            model.bert.gradient_checkpointing_enable()
            print("🧠 Gradient Checkpointing 有効化 (CUDA)")
        else:
            print("ℹ️ Gradient Checkpointing 無効 (CUDA非利用)")
    except Exception:
        print("ℹ️ Gradient Checkpointing 設定スキップ")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    # AMPはCUDAのみ有効
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    # 学習
    best_val = float("inf")
    best_state = None
    for epoch in range(NUM_EPOCHS):
        tr = train_loop(model, train_loader, optimizer, scaler, epoch)
        va = eval_loop(model, val_loader)
        print(f"[Epoch {epoch+1}] Train {tr:.4f} | Val {va:.4f}")
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # テスト
    test_loss = eval_loop(model, test_loader)
    print(f"🧪 Test KL (weighted): {test_loss:.4f}")

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("02_モデル", "授業レベルマルチタスクモデル")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"class_level_ordinal_llp_{ts}.pth")
    torch.save(model.state_dict(), model_path)

    results = {
        "timestamp": ts,
        "test_weighted_KL": float(test_loss),
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "notes": "course-level LLP with ordinal (1-4)"
    }
    with open(os.path.join(out_dir, f"class_level_ordinal_llp_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 保存: {model_path}")


if __name__ == "__main__":
    main()

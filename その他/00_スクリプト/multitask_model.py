#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汎用的なマルチタスク学習モデル
複数のタスクを同時に学習できる柔軟な構造

特徴:
- 共有エンコーダー（BERT）とタスク固有のヘッド
- カスタマイズ可能なタスク定義
- 柔軟な損失関数の組み合わせ
- 長文対応（チャンク化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertJapaneseTokenizer
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class TaskHead(nn.Module):
    """タスク固有のヘッド（回帰・分類・順序回帰などに対応）"""
    
    def __init__(
        self,
        input_size: int,
        task_type: str = "regression",  # "regression", "classification", "ordinal"
        output_size: int = 1,
        hidden_sizes: List[int] = [256],
        dropout_rate: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        self.task_type = task_type
        self.output_size = output_size
        
        # 活性化関数の選択
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ReLU()
        
        # ネットワーク構築
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Dropout(dropout_rate),
                nn.Linear(prev_size, hidden_size),
                act_fn
            ])
            prev_size = hidden_size
        
        # 出力層
        layers.append(nn.Dropout(dropout_rate))
        if task_type == "classification":
            layers.append(nn.Linear(prev_size, output_size))
        elif task_type == "ordinal":
            # 順序回帰: 累積ロジットを出力
            layers.append(nn.Linear(prev_size, output_size - 1))
        else:  # regression
            layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)
        
        if self.task_type == "classification":
            return F.log_softmax(output, dim=-1)
        elif self.task_type == "ordinal":
            # 順序回帰: シグモイドで累積確率に変換
            return torch.sigmoid(output)
        else:  # regression
            return output.squeeze(-1) if self.output_size == 1 else output


class MultitaskModel(nn.Module):
    """
    汎用的なマルチタスク学習モデル
    
    Args:
        base_model_name: ベースモデル名（BERTなど）
        task_configs: タスク設定のリスト
            [
                {
                    "name": "sentiment",
                    "type": "regression",
                    "output_size": 1,
                    "hidden_sizes": [256],
                    "weight": 1.0
                },
                ...
            ]
        dropout_rate: ドロップアウト率
        chunk_len: チャンク長（長文対応）
        max_chunks: 最大チャンク数
    """
    
    def __init__(
        self,
        base_model_name: str,
        task_configs: List[Dict],
        dropout_rate: float = 0.1,
        chunk_len: int = 256,
        max_chunks: int = 10
    ):
        super().__init__()
        
        # BERTエンコーダ（共有層）
        try:
            self.bert = BertModel.from_pretrained(
                base_model_name, 
                use_safetensors=False
            )
        except Exception:
            try:
                self.bert = BertModel.from_pretrained(
                    base_model_name, 
                    use_safetensors=True
                )
            except Exception:
                self.bert = BertModel.from_pretrained(
                    base_model_name,
                    trust_remote_code=True
                )
        
        hidden_size = self.bert.config.hidden_size
        self.hidden_size = hidden_size
        self.chunk_len = chunk_len
        self.max_chunks = max_chunks
        
        # 共有ドロップアウト
        self.shared_dropout = nn.Dropout(dropout_rate)
        
        # タスク固有のヘッド
        self.task_heads = nn.ModuleDict()
        self.task_configs = {}
        
        for task_config in task_configs:
            task_name = task_config["name"]
            self.task_configs[task_name] = task_config
            
            self.task_heads[task_name] = TaskHead(
                input_size=hidden_size,
                task_type=task_config.get("type", "regression"),
                output_size=task_config.get("output_size", 1),
                hidden_sizes=task_config.get("hidden_sizes", [256]),
                dropout_rate=dropout_rate,
                activation=task_config.get("activation", "relu")
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        フォワードパス
        
        Args:
            input_ids: [B, L] または [B, C, L]
            attention_mask: [B, L] または [B, C, L]
            chunk_mask: [B, C] (オプション)
        
        Returns:
            Dict[str, torch.Tensor]: 各タスクの予測結果
        """
        # チャンク化された入力の処理
        if input_ids.dim() == 3:
            B, C, L = input_ids.shape
            x_ids = input_ids.view(B * C, L)
            x_mask = attention_mask.view(B * C, L)
            
            outputs = self.bert(input_ids=x_ids, attention_mask=x_mask)
            cls = outputs.last_hidden_state[:, 0, :]  # [B*C, H]
            H = cls.size(-1)
            cls = cls.view(B, C, H)  # [B, C, H]
            
            # チャンクの集約
            if chunk_mask is not None:
                mask = chunk_mask.float().unsqueeze(-1)  # [B, C, 1]
                summed = (cls * mask).sum(dim=1)  # [B, H]
                denom = mask.sum(dim=1).clamp_min(1e-6)  # [B, 1]
                pooled = summed / denom
            else:
                pooled = cls.mean(dim=1)  # [B, H]
        else:
            # 通常の入力 [B, L]
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]  # [B, H]
        
        # 共有ドロップアウト
        pooled = self.shared_dropout(pooled)
        
        # 各タスクの予測
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(pooled)
        
        return predictions
    
    def get_task_weights(self) -> Dict[str, float]:
        """各タスクの重みを取得"""
        return {
            name: config.get("weight", 1.0)
            for name, config in self.task_configs.items()
        }


class MultitaskLoss(nn.Module):
    """マルチタスク損失関数"""
    
    def __init__(
        self,
        task_configs: List[Dict],
        reduction: str = "mean"
    ):
        super().__init__()
        self.task_configs = {cfg["name"]: cfg for cfg in task_configs}
        self.reduction = reduction
        
        # タスクタイプに応じた損失関数
        self.loss_fns = {}
        for cfg in task_configs:
            task_name = cfg["name"]
            task_type = cfg.get("type", "regression")
            
            if task_type == "classification":
                self.loss_fns[task_name] = nn.NLLLoss(reduction=reduction)
            elif task_type == "ordinal":
                # 順序回帰用の損失（BCE with logits）
                self.loss_fns[task_name] = nn.BCEWithLogitsLoss(reduction=reduction)
            else:  # regression
                self.loss_fns[task_name] = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        損失を計算
        
        Args:
            predictions: 予測値の辞書
            targets: 正解値の辞書
        
        Returns:
            total_loss: 合計損失
            task_losses: 各タスクの損失
        """
        task_losses = {}
        total_loss = 0.0
        
        for task_name, pred in predictions.items():
            if task_name not in targets:
                continue
            
            target = targets[task_name]
            task_type = self.task_configs[task_name].get("type", "regression")
            weight = self.task_configs[task_name].get("weight", 1.0)
            
            if task_type == "ordinal":
                # 順序回帰: 累積ロジットから確率分布を復元
                logits = pred
                probs_ge = torch.sigmoid(logits)  # [B, K-1]
                
                # 確率分布を復元: P(y=1), P(y=2), ..., P(y=K)
                p1 = 1.0 - probs_ge[:, 0]
                p_list = [p1.unsqueeze(1)]
                
                for i in range(1, probs_ge.size(1)):
                    p_list.append(
                        (probs_ge[:, i-1] - probs_ge[:, i]).clamp(min=0.0).unsqueeze(1)
                    )
                
                p_last = probs_ge[:, -1].unsqueeze(1)
                p_list.append(p_last)
                
                P = torch.cat(p_list, dim=1)  # [B, K]
                P = P.clamp(min=1e-8)
                P = P / P.sum(dim=1, keepdim=True)
                
                # KL divergence または Cross Entropy
                if target.dim() == 1:
                    # クラスラベル [B]
                    target_onehot = F.one_hot(target.long() - 1, num_classes=P.size(1)).float()
                    loss = -(target_onehot * P.clamp(1e-8).log()).sum(dim=1).mean()
                else:
                    # 分布 [B, K]
                    loss = (target * (target.clamp(1e-8).log() - P.clamp(1e-8).log())).sum(dim=1).mean()
            else:
                loss_fn = self.loss_fns[task_name]
                
                if task_type == "classification":
                    # 分類: ターゲットをlong型に変換
                    if target.dim() > 1:
                        target = target.argmax(dim=-1)
                    target = target.long()
                else:
                    # 回帰: ターゲットをfloat型に変換
                    if target.dim() > 1 and target.size(-1) == 1:
                        target = target.squeeze(-1)
                
                loss = loss_fn(pred, target)
            
            task_losses[task_name] = loss
            total_loss += weight * loss
        
        return total_loss, task_losses


# 使用例
if __name__ == "__main__":
    # タスク設定の例
    task_configs = [
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
        },
        {
            "name": "satisfaction",
            "type": "ordinal",
            "output_size": 4,  # 1-4の順序
            "hidden_sizes": [256],
            "weight": 1.0,
            "activation": "relu"
        }
    ]
    
    # モデルの作成
    model = MultitaskModel(
        base_model_name="koheiduck/bert-japanese-finetuned-sentiment",
        task_configs=task_configs,
        dropout_rate=0.1,
        chunk_len=256,
        max_chunks=10
    )
    
    print("✅ マルチタスクモデルが作成されました")
    print(f"タスク数: {len(task_configs)}")
    for task_name in model.task_heads.keys():
        print(f"  - {task_name}")

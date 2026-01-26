# MacBook M4でのマルチタスク学習実行手順

## 概要

MacBook M4（Apple Silicon）でもマルチタスク学習を実行できますが、CUDAは使用できません。
代わりに**MPS（Metal Performance Shaders）**を使用します。

## CUDAとMPSの違い

| 項目 | CUDA（NVIDIA GPU） | MPS（Apple Silicon） |
|------|-------------------|---------------------|
| 対応GPU | NVIDIA GPUのみ | Apple Silicon（M1/M2/M3/M4） |
| パフォーマンス | 高い | 高い（MPS経由） |
| PyTorchサポート | 完全対応 | PyTorch 1.12以降で対応 |
| インストール方法 | `--index-url cu118` | 通常のpipインストール |

## セットアップ手順

### 1. 仮想環境の作成

```bash
cd ~/Desktop/gakkai/その他
python3 -m venv venv
source venv/bin/activate
```

### 2. PyTorchのインストール（MPS対応版）

MacBook M4では、通常のPyTorchインストールでMPSが自動的に有効になります：

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
```

**注意**: CUDA版のインストールコマンド（`--index-url cu118`）は使用しないでください。

### 3. その他のパッケージのインストール

```bash
pip install transformers pandas "numpy<2.0.0" scikit-learn matplotlib seaborn tqdm "shap>=0.42.0"
```

または、requirements.txtから（PyTorchの行を削除して）：

```bash
pip install transformers pandas "numpy<2.0.0" scikit-learn matplotlib seaborn tqdm "shap>=0.42.0"
```

### 4. MPSの確認

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

## スクリプトの修正が必要な場合

`train_class_level_ordinal_llp.py`のデバイス選択部分を確認・修正：

### 現在のコード（CUDA用）

```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
```

### MPS対応版

```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

## 実行方法

```bash
cd ~/Desktop/gakkai/その他/00_スクリプト
python train_class_level_ordinal_llp.py
```

## 注意事項

### 1. MPSの制限事項

- **一部の操作が未対応**: すべてのPyTorch操作がMPSでサポートされているわけではありません
- **フォールバック**: 未対応の操作は自動的にCPUにフォールバックされます
- **メモリ管理**: MPSはCUDAとは異なるメモリ管理を使用します

### 2. パフォーマンス

- M4は強力なGPUを搭載しているため、十分なパフォーマンスが期待できます
- ただし、CUDA対応のNVIDIA GPUと比較すると、一部の操作で差が出る場合があります

### 3. トラブルシューティング

**MPSが利用できない場合：**
```python
import torch
print(torch.backends.mps.is_available())  # Falseの場合
```

考えられる原因：
- macOSのバージョンが古い（macOS 12.3以降が必要）
- PyTorchのバージョンが古い（1.12以降が必要）
- Metalが無効化されている

**解決方法：**
```bash
# PyTorchを最新版にアップグレード
pip install --upgrade torch torchvision torchaudio
```

## まとめ

- ✅ MacBook M4で実行可能
- ✅ MPSを使用（CUDAの代わり）
- ✅ 通常のPyTorchインストールでOK
- ⚠️ スクリプトのデバイス選択部分を確認・修正が必要な場合あり

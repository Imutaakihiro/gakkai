# 順序回帰SHAP分析 環境構築手順

**作成日**: 2025年1月  
**目的**: 順序回帰SHAP分析を実行するための仮想環境セットアップ

---

## 📋 前提条件

- Python 3.8以上がインストールされていること
- Windows 10/11 または Linux/Mac
- インターネット接続（パッケージのダウンロード用）

---

## 🚀 セットアップ方法

### 方法1: 自動セットアップスクリプト（推奨）

#### Windowsの場合
```bash
# プロジェクトルートで実行
setup_venv_ordinal_shap.bat
```

#### Linux/Macの場合
```bash
# 実行権限を付与
chmod +x setup_venv_ordinal_shap.sh

# プロジェクトルートで実行
./setup_venv_ordinal_shap.sh
```

### 方法2: 手動セットアップ

#### 1. 仮想環境の作成
```bash
# Windows
python -m venv venv_ordinal_shap

# Linux/Mac
python3 -m venv venv_ordinal_shap
```

#### 2. 仮想環境の有効化
```bash
# Windows
venv_ordinal_shap\Scripts\activate.bat

# Linux/Mac
source venv_ordinal_shap/bin/activate
```

#### 3. pipのアップグレード
```bash
python -m pip install --upgrade pip
```

#### 4. パッケージのインストール

**GPU使用時（CUDA 11.8）:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_ordinal_shap.txt
```

**CPU使用時:**
```bash
pip install torch torchvision torchaudio
pip install -r requirements_ordinal_shap.txt
```

---

## ✅ インストール確認

仮想環境を有効化した状態で、以下を実行して確認：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import shap; print(f'SHAP: {shap.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

---

## 🔧 トラブルシューティング

### 1. SHAPのインストールエラー

**問題**: NumPy 2.0との互換性エラー

**解決方法**:
```bash
# NumPy 1.x系にダウングレード
pip install "numpy<2.0.0"
pip install shap>=0.42.0
```

### 2. PyTorchのCUDA版インストールエラー

**問題**: CUDA版PyTorchのインストールに失敗

**解決方法**:
```bash
# CPU版をインストール
pip install torch torchvision torchaudio
```

### 3. 仮想環境の有効化エラー

**問題**: `venv_ordinal_shap\Scripts\activate.bat` が見つからない

**解決方法**:
- 仮想環境が正しく作成されているか確認
- プロジェクトルートで実行しているか確認
- 管理者権限で実行してみる

---

## 📝 使用方法

### 1. 仮想環境の有効化
```bash
# Windows
venv_ordinal_shap\Scripts\activate.bat

# Linux/Mac
source venv_ordinal_shap/bin/activate
```

### 2. 順序回帰SHAP分析の実行
```bash
cd 00_スクリプト
python analyze_ordinal_shap_production.py
```

### 3. 仮想環境の無効化
```bash
deactivate
```

---

## 📦 インストールされるパッケージ

- **PyTorch**: 深層学習フレームワーク
- **Transformers**: BERTモデル用
- **Pandas**: データ処理
- **NumPy**: 数値計算（1.x系、SHAP互換性のため）
- **scikit-learn**: 機械学習
- **Matplotlib/Seaborn**: 可視化
- **SHAP**: SHAP分析
- **fugashi/ipadic**: 日本語BERT用（オプション）

---

## ⚠️ 注意事項

1. **NumPyのバージョン**: SHAPとの互換性のため、NumPy 1.x系を使用（2.0未満）
2. **仮想環境の分離**: 他のプロジェクトとパッケージが競合しないよう、専用の仮想環境を使用
3. **GPU使用時**: CUDA版PyTorchをインストールする場合は、CUDA 11.8以上が必要

---

## 🔄 仮想環境の再作成

問題が発生した場合は、仮想環境を削除して再作成：

```bash
# Windows
rmdir /s /q venv_ordinal_shap
setup_venv_ordinal_shap.bat

# Linux/Mac
rm -rf venv_ordinal_shap
./setup_venv_ordinal_shap.sh
```

---

**最終更新**: 2025年1月


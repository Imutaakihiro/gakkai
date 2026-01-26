# MacBook M4標準でのマルチタスク学習実行手順

## ⏱️ 実行時間の目安

**標準のM4**: **40-60分**程度

## 📋 実行前の準備

### 1. 仮想環境の作成

```bash
cd ~/Desktop/gakkai/その他
python3 -m venv venv
source venv/bin/activate
```

### 2. パッケージのインストール

```bash
# pipをアップグレード
pip install --upgrade pip

# PyTorch（MPS対応版）とその他のパッケージをインストール
pip install torch torchvision torchaudio
pip install transformers pandas "numpy<2.0.0" scikit-learn matplotlib seaborn tqdm "shap>=0.42.0"
```

または、requirements_mac.txtを使用：

```bash
pip install -r requirements_mac.txt
```

### 3. MPSの確認

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"
```

**期待される出力**:
```
MPS available: True
MPS built: True
```

## 🚀 実行手順

### 1. 仮想環境を有効化

```bash
cd ~/Desktop/gakkai/その他
source venv/bin/activate
```

### 2. スクリプトを実行

```bash
cd 00_スクリプト
python train_class_level_ordinal_llp.py
```

## 📊 実行中の確認事項

### 1. MPSが使用されているか確認

実行開始時に以下のようなメッセージが表示されます：

```
✅ MPS (Apple Silicon) 利用
```

### 2. 進捗の確認

```
[Epoch 1/3] Train 0.1234 | Val 0.5678
[Epoch 2/3] Train 0.0987 | Val 0.5432
[Epoch 3/3] Train 0.0876 | Val 0.5210
```

### 3. GPU使用率の確認（オプション）

Activity Monitor を開いて、GPUタブで使用率を確認できます。

## ⚠️ 注意事項

### 1. メモリ管理

- M4標準は統一メモリを使用
- バッチサイズ2で約4-6GBのメモリを使用
- 他のアプリケーションを閉じて実行することを推奨

### 2. 発熱対策

- 40-60分の実行中、MacBookが発熱する可能性があります
- 通気性の良い場所で実行
- 必要に応じて冷却パッドを使用

### 3. バッテリー

- 長時間実行するため、電源に接続して実行することを推奨
- バッテリーのみでの実行は、途中で電源が切れる可能性があります

## 🎯 実行時間の内訳（標準M4）

1. **データ読み込み・前処理**: 1-2分
2. **モデル読み込み**: 30秒-1分
3. **エポック1**: 12-18分
4. **エポック2**: 12-18分
5. **エポック3**: 12-18分
6. **モデル保存**: 10-30秒

**合計**: 40-60分

## 💡 時間短縮のオプション

### バッチサイズを増やす（メモリに余裕がある場合）

`train_class_level_ordinal_llp.py`を編集：

```python
BATCH_SIZE = 4  # 2 → 4に変更
```

**効果**: 実行時間が約半分（20-30分）に短縮
**注意**: メモリ使用量が約2倍になります

### エポック数を減らす（検証用）

```python
NUM_EPOCHS = 2  # 3 → 2に変更
```

**効果**: 実行時間が約2/3（25-40分）に短縮

## 📝 実行後の確認

### 1. モデルファイルの確認

```bash
ls -lh 02_モデル/授業レベルマルチタスクモデル/
```

以下のファイルが生成されます：
- `class_level_ordinal_llp_YYYYMMDD_HHMMSS.pth` - モデルファイル
- `class_level_ordinal_llp_YYYYMMDD_HHMMSS.json` - 学習結果

### 2. 学習結果の確認

JSONファイルに以下の情報が記録されます：
- テスト損失
- ベースモデル
- 学習設定（バッチサイズ、エポック数など）

## 🔧 トラブルシューティング

### MPSが利用できない場合

```python
import torch
print(torch.backends.mps.is_available())  # False の場合
```

**解決方法**:
1. macOSのバージョンを確認（macOS 12.3以降が必要）
2. PyTorchを最新版にアップグレード
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

### メモリ不足エラー

```
RuntimeError: MPS backend out of memory
```

**解決方法**:
- バッチサイズを2のまま維持
- 他のアプリケーションを閉じる
- システムを再起動してメモリをクリア

### 実行が遅い場合

- Activity MonitorでCPU/GPU使用率を確認
- 他の重いアプリケーションが実行されていないか確認
- バッチサイズを4に増やす（メモリに余裕がある場合）

## ✅ チェックリスト

実行前：
- [ ] 仮想環境が作成されている
- [ ] パッケージがインストールされている
- [ ] MPSが利用可能か確認済み
- [ ] 電源に接続されている（推奨）
- [ ] 他の重いアプリケーションを閉じた

実行中：
- [ ] MPSが使用されていることを確認
- [ ] 進捗が正常に表示されている
- [ ] エラーメッセージが出ていない

実行後：
- [ ] モデルファイルが生成されている
- [ ] 学習結果JSONが生成されている

## 🎯 まとめ

- **実行時間**: 40-60分（標準M4）
- **推奨環境**: 電源接続、他のアプリケーションを閉じる
- **メモリ**: バッチサイズ2で約4-6GB使用
- **GPU**: MPSが自動的に使用される

準備が整ったら、上記の手順で実行してください！

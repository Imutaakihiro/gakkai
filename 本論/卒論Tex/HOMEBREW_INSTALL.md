# Homebrewインストール手順

## インストール方法

ターミナルで以下のコマンドを実行してください：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## インストール時の注意事項

1. **パスワードの入力が求められます**
   - macOSの管理者パスワードを入力してください
   - 入力中は画面に表示されませんが、正常です

2. **インストール完了後、パスを設定**
   - Apple Silicon Mac（M1/M2/M3など）の場合：
     ```bash
     echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
     eval "$(/opt/homebrew/bin/brew shellenv)"
     ```
   
   - Intel Macの場合：
     ```bash
     echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
     eval "$(/usr/local/bin/brew shellenv)"
     ```

3. **インストール確認**
   ```bash
   brew --version
   ```

## SSL証明書エラーが出る場合

もし `curl: (77) error setting certificate verify locations` というエラーが出る場合は、以下の方法を試してください：

### 方法1: 証明書を更新
```bash
# 証明書を更新
brew install ca-certificates
```

ただし、これはHomebrewがインストールされていないと実行できません。

### 方法2: 手動でインストールスクリプトをダウンロード
1. ブラウザで以下にアクセス：
   https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
2. ページの内容をコピーして、`install.sh`というファイルに保存
3. ターミナルで実行：
   ```bash
   bash install.sh
   ```

### 方法3: 公式サイトから確認
- https://brew.sh/
- 最新のインストール手順を確認

## インストール後の次のステップ

Homebrewがインストールできたら、MacTeXをインストールします：

```bash
brew install --cask mactex
```

インストールには時間がかかります（数GBのダウンロードが必要です）。

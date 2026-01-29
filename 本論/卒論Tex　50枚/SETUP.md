# セットアップガイド

## 1. LaTeX環境のインストール

### macOSの場合

#### 方法1: MacTeXを直接ダウンロード（Homebrew不要、推奨）

1. **MacTeX公式サイトからダウンロード**
   - https://www.tug.org/mactex/
   - 「MacTeX.pkg」をダウンロード（約4GB）
   - ダウンロードした`.pkg`ファイルをダブルクリックしてインストール

2. **インストール後の確認**
   ```bash
   # パスを追加（~/.zshrcに追加）
   echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   
   # 確認
   which uplatex
   which dvipdfmx
   ```

#### 方法2: Homebrewを使用する場合

**まずHomebrewをインストール（まだインストールしていない場合）**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**その後、MacTeXをインストール**
```bash
brew install --cask mactex
```

#### 方法3: BasicTeX（軽量版、Homebrew使用）

```bash
# BasicTeXをインストール
brew install --cask basictex

# 必要なパッケージをインストール
sudo tlmgr update --self
sudo tlmgr install uplatex collection-langjapanese ptex-ipaex
```

### インストール後の確認

ターミナルで以下を実行して、インストールが成功したか確認してください：

```bash
which uplatex
which dvipdfmx
```

パスが表示されれば成功です。表示されない場合は、パスを追加する必要があります：

```bash
# ~/.zshrc または ~/.bash_profile に追加
export PATH="/usr/local/texlive/2024/bin/universal-darwin:$PATH"
# または
export PATH="/Library/TeX/texbin:$PATH"
```

設定を反映：
```bash
source ~/.zshrc  # zshの場合
# または
source ~/.bash_profile  # bashの場合
```

## 2. Cursor拡張機能のインストール

1. Cursorを開く
2. 拡張機能タブ（⌘+Shift+X）を開く
3. "LaTeX Workshop" を検索してインストール

## 3. 動作確認

### 方法1: Makefileを使用
```bash
cd /Users/imutaakihiro/Downloads/開発/gakkai/本論/卒論Tex
make
```

### 方法2: Cursorから
1. `sano_finish.tex` を開く
2. ⌘+Option+B（または右クリック → "Build LaTeX project"）でビルド
3. ⌘+Option+V でPDFをプレビュー

## トラブルシューティング

### エラー: "brew: command not found"
Homebrewがインストールされていません。以下のいずれかを選択してください：
- **方法1**: Homebrewをインストール（上記の方法2を参照）
- **方法2**: MacTeXを直接ダウンロード（方法1を参照、推奨）

### エラー: "uplatex: command not found"
- TeX Liveが正しくインストールされているか確認
- パスが通っているか確認（上記のパス設定を参照）
- ターミナルを再起動するか、`source ~/.zshrc`を実行

### エラー: "ptex-ipaex.map not found"
```bash
sudo tlmgr install ptex-ipaex
```

### エラー: パッケージが見つからない
必要なパッケージを個別にインストール：
```bash
sudo tlmgr install <パッケージ名>
```

よく使うパッケージ：
```bash
sudo tlmgr install ascmac graphicx amsmath booktabs subcaption
```

### Cursorでビルドが実行されない
1. LaTeX Workshop拡張機能がインストールされているか確認
2. `.vscode/settings.json` が正しく作成されているか確認
3. Cursorを再起動

# 卒論LaTeXプロジェクト

このプロジェクトは、Cursorで編集・実行できるように設定された日本語LaTeX文書です。

## 必要な環境

以下のソフトウェアがインストールされている必要があります：

1. **TeX Live** または **MacTeX** (macOSの場合)
   - `uplatex`: 日本語LaTeXコンパイラ
   - `dvipdfmx`: DVIからPDFへの変換ツール
   - 必要なパッケージ: `jreport`, `ascmac`, `graphicx`, `amsmath`, `booktabs` など

2. **Cursor拡張機能**
   - LaTeX Workshop（自動的にインストールを促されます）

### インストール方法

詳細なセットアップ手順は **SETUP.md** を参照してください。

#### クイックインストール（macOS）

**方法1: 直接ダウンロード（Homebrew不要、推奨）**
1. https://www.tug.org/mactex/ からMacTeX.pkgをダウンロード
2. ダウンロードした`.pkg`ファイルをダブルクリックしてインストール
3. パスを設定：
   ```bash
   echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

**方法2: Homebrewを使用（Homebrewがインストールされている場合）**
```bash
# MacTeX（完全版、推奨）
brew install --cask mactex

# または、BasicTeX（軽量版）
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install uplatex collection-langjapanese ptex-ipaex
```

#### Linux
```bash
# TeX Liveをインストール
sudo apt-get install texlive-lang-japanese texlive-latex-extra
```

## 使い方

### Cursorでの編集

1. Cursorでこのフォルダを開く
2. `sano_finish.tex` を編集
3. LaTeX Workshop拡張機能が自動的にビルドを実行します

### 手動でビルドする場合

#### Makefileを使用（推奨）
```bash
# PDFを生成
make

# または明示的に
make pdf

# 中間ファイルを削除
make clean

# PDFも含めてすべて削除
make cleanall

# クリーンアップしてから再ビルド
make rebuild
```

#### 直接コマンドを実行
```bash
# 1回目のコンパイル
uplatex -synctex=1 -interaction=nonstopmode -file-line-error -kanji=utf8 sano_finish.tex

# 2回目のコンパイル（相互参照を解決するため）
uplatex -synctex=1 -interaction=nonstopmode -file-line-error -kanji=utf8 sano_finish.tex

# DVIからPDFへの変換
dvipdfmx -f ptex-ipaex.map -o sano_finish.pdf sano_finish.dvi
```

#### latexmkを使用
```bash
latexmk -pdfdvi sano_finish.tex
```

## ファイル構成

- `sano_finish.tex`: メインのLaTeXファイル
- `表紙.TEX`: 表紙
- `1章.tex`, `2章.tex`, ...: 各章の内容
- `7付録A.tex`, `8付録B.tex`: 付録
- `9謝辞.tex`: 謝辞
- `10参考文献.tex`: 参考文献
- `.vscode/settings.json`: Cursor/VS Code用の設定
- `.latexmkrc`: latexmk用の設定
- `Makefile`: ビルド用のMakefile

## トラブルシューティング

### エラー: uplatex が見つからない
- TeX LiveまたはMacTeXが正しくインストールされているか確認
- パスが通っているか確認: `which uplatex`

### エラー: パッケージが見つからない
- 必要なパッケージをインストール:
  ```bash
  sudo tlmgr install <パッケージ名>
  ```

### エラー: フォントが見つからない
- 日本語フォントパッケージをインストール:
  ```bash
  sudo tlmgr install ptex-ipaex
  ```

### 相互参照が正しく表示されない
- 2回コンパイルを実行してください（Makefileは自動的に2回実行します）

## 注意事項

- このプロジェクトは日本語LaTeX（`jreport`クラス）を使用しています
- コンパイルには`uplatex`と`dvipdfmx`が必要です
- 相互参照や目次を正しく表示するには、2回のコンパイルが必要です

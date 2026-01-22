# クイックスタートガイド

## ✅ インストール完了！

LaTeX環境のインストールが完了しました。これでCursorでLaTeXファイルを編集・ビルドできます。

## 次のステップ

### 1. Cursor拡張機能のインストール

1. Cursorを開く
2. 拡張機能タブを開く（⌘+Shift+X）
3. "LaTeX Workshop" を検索
4. インストールボタンをクリック

### 2. パスの永続化（重要）

ターミナルを再起動してもLaTeXコマンドが使えるように、パスを永続化します：

```bash
echo 'eval "$(/usr/libexec/path_helper)"' >> ~/.zshrc
source ~/.zshrc
```

### 3. LaTeXファイルのビルド

#### 方法1: Cursorから（推奨）

1. `sano_finish.tex` を開く
2. コマンドパレットを開く（⌘+Shift+P）
3. "LaTeX Workshop: Build with recipe" を選択
4. "uplatex (x2) -> dvipdfmx" を選択

または、ショートカット：
- ⌘+Option+B でビルド
- ⌘+Option+V でPDFをプレビュー

#### 方法2: ターミナルから

```bash
cd /Users/imutaakihiro/Downloads/開発/gakkai/本論/卒論Tex
make
```

### 4. 動作確認

簡単なテストとして、以下を実行してみてください：

```bash
cd /Users/imutaakihiro/Downloads/開発/gakkai/本論/卒論Tex
make clean
make
```

`sano_finish.pdf` が生成されれば成功です！

## トラブルシューティング

### エラー: "uplatex: command not found"

ターミナルを再起動するか、以下を実行：
```bash
eval "$(/usr/libexec/path_helper)"
```

### Cursorでビルドが実行されない

1. LaTeX Workshop拡張機能がインストールされているか確認
2. Cursorを再起動
3. `.vscode/settings.json` が存在するか確認

### パスが毎回設定されない

`~/.zshrc` に以下を追加：
```bash
echo 'eval "$(/usr/libexec/path_helper)"' >> ~/.zshrc
```

## 完了！

これで、Cursorで卒論のLaTeXファイルを編集・ビルドできるようになりました！

# テンプレート化のためのファイル整理ガイド

## 削除すべきファイル一覧

### 1. バックアップファイル（*.bak）
**理由**: バックアップファイルはテンプレートには不要です。削除しても問題ありません。

- `*.tex.bak` (すべて)
- `*.TEX.bak` (すべて)
- `表紙.TEX.bak`

### 2. 中間ファイル（LaTeXコンパイル時に自動生成される）
**理由**: これらはコンパイル時に自動生成されるため、テンプレートには不要です。

- `sano_finish.aux`
- `sano_finish.dvi`
- `sano_finish.log`
- `sano_finish.toc`
- `sano_finish.lof`
- `sano_finish.lot`

### 3. 特定の研究内容に関連する画像ファイル
**理由**: これらは現在の研究内容に特化した図表です。テンプレートでは削除し、新しい研究内容に合わせて画像を追加してください。

#### EPS形式の画像（clip*.eps, clip*.pbm, clip*.bmc）
- `clip3-1.eps`, `clip3-1.pbm`, `clip3-1.bmc`
- `clip3-2.eps`, `clip3-2.pbm`, `clip3-2.bmc`
- `clip4-1.eps`, `clip4-1.pbm`, `clip4-1.bmc`
- `clip4-2.eps`, `clip4-2.pbm`, `clip4-2.bmc`
- `clip4-3.pbm`, `clip4-3.bmc`
- `clip4-4.pbm`, `clip4-5.pbm`
- `clip5-1.eps`, `clip5-1.pbm`, `clip5-1.bmc`
- `clip5-2.eps`, `clip5-2.pbm`, `clip5-2.bmc`
- `clip5-3.eps`, `clip5-3.pbm`, `clip5-3.bmc`
- `clip5-4.pbm`, `clip5-4.bmc`
- `clip5-5.pbm`, `clip5-5.bmc`
- `clip6-1.eps`, `clip6-1.pbm`, `clip6-1.bmc`
- `clip6-2.eps`, `clip6-2.pbm`, `clip6-2.bmc`
- `clip6-3.eps`, `clip6-3.pbm`, `clip6-3.bmc`
- `clip6-4.eps`, `clip6-4.pbm`, `clip6-4.bmc`
- `clip6-5.eps`, `clip6-5.pbm`, `clip6-5.bmc`
- `clip7-1.eps`, `clip7-1.pbm`, `clip7-1.bmc`
- `clip005.pbm`

#### PNG形式の画像
- `gnl.png` (3章で使用)
- `gnl_model.png` (使用されていない可能性)
- `1day.png` (4章で使用)
- `2week.png` (4章で使用)
- `1month.png` (4章で使用)

### 4. 使用されていない章ファイル
**理由**: `sano_finish.tex`でコメントアウトされているため、現在は使用されていません。

- `6章.tex` (コメントアウト)
- `7章.tex` (コメントアウト)
- `8章.tex` (コメントアウト)
- `11研究業績.TEX` (コメントアウト)
- `Sebyoshi.TEX` (コメントアウト)

### 5. 使用されていないスタイルファイル
**理由**: コメントアウトされているか、使用されていません。

- `FANCYHEA.STY` (fancyhdrパッケージはコメントアウト)
- `HYPER.STY` (hyperrefパッケージは使用されていない)

## 保持すべきファイル

### メインファイル
- `sano_finish.tex` (メインのLaTeXファイル)

### 章ファイル（使用中）
- `表紙.TEX`
- `1章.tex`
- `2章.tex`
- `3章.tex`
- `4章.tex`
- `5章.tex`
- `7付録A.tex`
- `8付録B.tex`
- `9謝辞.tex`
- `10参考文献.tex`

### スタイルファイル
- `yreport.cls` (ドキュメントクラス)
- `eclbkbox.sty` (使用中)
- `fonts.sty` (使用中)
- `centercolon.sty` (使用中)
- `EPSBOX.STY` (epsboxパッケージで使用)

### 設定ファイル
- `.vscode/` (Cursor/VS Code設定)
- `.latexmkrc` (latexmk設定)
- `Makefile` (ビルド用)
- `.gitignore` (Git設定)

### ドキュメント
- `README.md`
- `SETUP.md`
- `HOMEBREW_INSTALL.md`
- `QUICK_START.md`

## 削除コマンド

以下のコマンドで一括削除できます：

```bash
cd /Users/imutaakihiro/Downloads/開発/gakkai/本論/卒論Tex

# バックアップファイルを削除
rm -f *.bak *.TEX.bak

# 中間ファイルを削除
rm -f *.aux *.dvi *.log *.toc *.lof *.lot

# 画像ファイルを削除
rm -f clip*.eps clip*.pbm clip*.bmc clip005.pbm
rm -f gnl.png gnl_model.png 1day.png 2week.png 1month.png

# 使用されていない章ファイルを削除
rm -f 6章.tex 7章.tex 8章.tex 11研究業績.TEX Sebyoshi.TEX

# 使用されていないスタイルファイルを削除
rm -f FANCYHEA.STY HYPER.STY
```

## 注意事項

1. **画像参照**: 画像ファイルの参照は既にコメントアウト済みです（3章.tex、4章.tex）。

2. **章ファイルを削除する前に**: 必要に応じて内容を空のテンプレートに置き換えることを検討してください。

3. **バックアップ**: 削除前に、重要なファイルは別の場所にバックアップを取ることをお勧めします。

## クリーンアップスクリプトの実行

以下のコマンドで一括クリーンアップを実行できます：

```bash
cd /Users/imutaakihiro/Downloads/開発/gakkai/本論/卒論Tex
./cleanup_template.sh
```

または、手動で削除する場合は、上記の「削除コマンド」セクションを参照してください。

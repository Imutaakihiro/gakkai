# Makefile for Japanese LaTeX document
# 使用方法: make または make pdf

MAIN = sano_finish
TEX = $(MAIN).tex
PDF = $(MAIN).pdf
DVI = $(MAIN).dvi

# デフォルトターゲット
all: pdf

# PDFを生成
pdf: $(PDF)

$(PDF): $(TEX)
	@echo "=== 1回目のコンパイル ==="
	uplatex -synctex=1 -interaction=nonstopmode -file-line-error -kanji=utf8 $(TEX)
	@echo "=== 2回目のコンパイル ==="
	uplatex -synctex=1 -interaction=nonstopmode -file-line-error -kanji=utf8 $(TEX)
	@echo "=== DVIからPDFへの変換 ==="
	dvipdfmx -f ptex-ipaex.map -o $(PDF) $(DVI)
	@echo "=== 完了: $(PDF) が生成されました ==="

# クリーンアップ（中間ファイルを削除）
clean:
	@echo "=== 中間ファイルを削除中 ==="
	rm -f *.aux *.bbl *.blg *.idx *.ind *.lof *.lot *.out *.toc
	rm -f *.acn *.acr *.alg *.glg *.glo *.gls
	rm -f *.fls *.fdb_latexmk *.synctex.gz
	rm -f *.dvi
	@echo "=== クリーンアップ完了 ==="

# 完全クリーンアップ（PDFも削除）
cleanall: clean
	@echo "=== PDFも削除中 ==="
	rm -f $(PDF)
	@echo "=== 完全クリーンアップ完了 ==="

# ビルドとクリーンアップを一度に実行
rebuild: clean pdf

# ヘルプ
help:
	@echo "利用可能なコマンド:"
	@echo "  make        - PDFを生成"
	@echo "  make pdf    - PDFを生成"
	@echo "  make clean  - 中間ファイルを削除"
	@echo "  make cleanall - PDFも含めてすべて削除"
	@echo "  make rebuild - クリーンアップしてからPDFを生成"
	@echo "  make help   - このヘルプを表示"

.PHONY: all pdf clean cleanall rebuild help

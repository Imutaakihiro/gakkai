#!/usr/bin/env perl
# LaTeXmk configuration for Japanese LaTeX (uplatex + dvipdfmx)

$latex = 'uplatex -synctex=1 -interaction=nonstopmode -file-line-error -kanji=utf8 %O %S';
$bibtex = 'upbibtex %O %B';
$dvipdf = 'dvipdfmx -f ptex-ipaex.map %O -o %D %S';
$pdf_mode = 3;  # Use dvipdfmx
$max_repeat = 5;
$pdf_previewer = 'start evince';
$clean_ext = 'dvi synctex.gz fdb_latexmk fls aux bbl blg idx ind lof lot out toc acn acr alg glg glo gls';

# 中間ファイルの削除設定
$clean_full_ext = 'dvi synctex.gz fdb_latexmk fls aux bbl blg idx ind lof lot out toc acn acr alg glg glo gls';

# エンコーディング設定
$ENV{'LANG'} = 'ja_JP.UTF-8';

# LuaLaTeXビルド設定
$latex = 'lualatex %O %S';
$pdflatex = 'lualatex %O %S';
$bibtex = 'bibtex %O %B';
$biber = 'biber %O %B';
$makeindex = 'makeindex %O -o %D %S';
$max_repeat = 5;
$pdf_mode = 5;  # LuaLaTeX mode
$pvc_view_file_via_temporary = 0;

# 日本語対応
$ENV{'LUAOTFLOAD_CONFIG'} = '';

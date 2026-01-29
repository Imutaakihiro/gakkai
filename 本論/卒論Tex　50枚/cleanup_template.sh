#!/bin/bash
# テンプレート化のためのクリーンアップスクリプト

cd "$(dirname "$0")"

echo "=== テンプレート化のためのクリーンアップを開始します ==="
echo ""

# 1. バックアップファイルを削除
echo "1. バックアップファイルを削除中..."
rm -f *.bak *.TEX.bak
echo "   ✓ バックアップファイルを削除しました"

# 2. 中間ファイルを削除
echo "2. 中間ファイルを削除中..."
rm -f *.aux *.dvi *.log *.toc *.lof *.lot *.synctex.gz
echo "   ✓ 中間ファイルを削除しました"

# 3. 画像ファイルを削除
echo "3. 画像ファイルを削除中..."
rm -f clip*.eps clip*.pbm clip*.bmc clip005.pbm
rm -f gnl.png gnl_model.png 1day.png 2week.png 1month.png
echo "   ✓ 画像ファイルを削除しました"

# 4. 使用されていない章ファイルを削除（スキップ：すべての章ファイルを保持）
echo "4. 章ファイルは保持します（すべて残します）"
echo "   ✓ 章ファイルは保持されました"

# 5. 使用されていないスタイルファイルを削除
echo "5. 使用されていないスタイルファイルを削除中..."
rm -f FANCYHEA.STY HYPER.STY
echo "   ✓ 使用されていないスタイルファイルを削除しました"

echo ""
echo "=== クリーンアップ完了 ==="
echo ""
echo "注意: 画像ファイルを削除したため、以下のファイルの画像参照を"
echo "      コメントアウトまたは削除してください："
echo "      - 3章.tex (75行目: gnl.png)"
echo "      - 4章.tex (107, 112, 117行目: 1day.png, 2week.png, 1month.png)"

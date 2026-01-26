# PowerShellスクリプトをUTF-8 BOMに変換するスクリプト
# 使用方法: .\convert_to_utf8bom.ps1

param(
    [string]$FilePath = "fix_pytorch.ps1"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PowerShellスクリプトをUTF-8 BOMに変換" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path $FilePath)) {
    Write-Host "❌ ファイルが見つかりません: $FilePath" -ForegroundColor Red
    exit 1
}

# 現在のエンコーディングを確認
$bytes = [System.IO.File]::ReadAllBytes($FilePath)
$hasBOM = ($bytes.Length -ge 3) -and ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)

if ($hasBOM) {
    Write-Host "✅ ファイルは既にUTF-8 BOMです: $FilePath" -ForegroundColor Green
    exit 0
}

Write-Host "`nファイルを読み込み中..." -ForegroundColor Yellow
try {
    # UTF-8として読み込む（BOMがあれば自動的に認識）
    $content = Get-Content $FilePath -Raw -Encoding UTF8
} catch {
    # UTF-8で失敗した場合、デフォルトエンコーディングで試す
    Write-Host "⚠️  UTF-8での読み込みに失敗。デフォルトエンコーディングで再試行..." -ForegroundColor Yellow
    $content = Get-Content $FilePath -Raw
}

# UTF-8 BOMで保存
Write-Host "UTF-8 BOMで保存中..." -ForegroundColor Yellow
$utf8WithBom = New-Object System.Text.UTF8Encoding $true
[System.IO.File]::WriteAllText($FilePath, $content, $utf8WithBom)

Write-Host "✅ 変換完了: $FilePath" -ForegroundColor Green

# 確認
$newBytes = [System.IO.File]::ReadAllBytes($FilePath)
$newHasBOM = ($newBytes.Length -ge 3) -and ($newBytes[0] -eq 0xEF -and $newBytes[1] -eq 0xBB -and $newBytes[2] -eq 0xBF)
if ($newHasBOM) {
    Write-Host "✅ UTF-8 BOMが正しく設定されました" -ForegroundColor Green
} else {
    Write-Host "⚠️  BOMの設定に失敗した可能性があります" -ForegroundColor Yellow
}

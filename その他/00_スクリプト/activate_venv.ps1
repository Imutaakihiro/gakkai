# 仮想環境を有効化するスクリプト
# 使用方法: .\activate_venv.ps1

$venvPath = Join-Path (Split-Path $PSScriptRoot -Parent) "venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "❌ 仮想環境が見つかりません: $venvPath" -ForegroundColor Red
    Write-Host "先に仮想環境を作成してください: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# 実行ポリシーを確認・設定
$currentPolicy = Get-ExecutionPolicy -Scope CurrentUser
if ($currentPolicy -eq "Restricted") {
    Write-Host "実行ポリシーを変更中..." -ForegroundColor Yellow
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-Host "✅ 実行ポリシーを変更しました" -ForegroundColor Green
}

# 仮想環境を有効化
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "✅ 仮想環境を有効化しました" -ForegroundColor Green
    Write-Host "`n仮想環境が有効化されました。プロンプトに (venv) が表示されていることを確認してください。" -ForegroundColor Cyan
} else {
    Write-Host "❌ 仮想環境の有効化スクリプトが見つかりません" -ForegroundColor Red
    exit 1
}

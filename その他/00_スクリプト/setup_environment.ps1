# Python環境セットアップスクリプト
# 使用方法: .\setup_environment.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python環境セットアップ" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Pythonの確認
Write-Host "`n[1/4] Pythonのインストール確認中..." -ForegroundColor Yellow

$pythonCmd = $null
$pythonVersion = $null

# 複数の方法でPythonを探す
# 1. 直接パスを確認（ターミナル出力から確認されたパス）
$directPath = "C:\Users\mkmzk\AppData\Local\Programs\Python\Python311\python.exe"
if (Test-Path $directPath) {
    try {
        $result = & $directPath --version 2>&1
        if ($result -match "Python") {
            $pythonCmd = $directPath
            $pythonVersion = $result
            Write-Host "✅ Pythonが見つかりました（直接パス）: $pythonCmd" -ForegroundColor Green
            Write-Host "   バージョン: $pythonVersion" -ForegroundColor Green
        }
    } catch {
        # 続行
    }
}

# 2. コマンド名で探す（直接パスが見つからなかった場合）
if (-not $pythonCmd) {
    $pythonCommands = @("python", "python3", "py")
    foreach ($cmd in $pythonCommands) {
        try {
            $result = & $cmd --version 2>&1
            if ($LASTEXITCODE -eq 0 -or $result -match "Python") {
                $pythonCmd = $cmd
                $pythonVersion = $result
                Write-Host "✅ Pythonが見つかりました: $pythonCmd" -ForegroundColor Green
                Write-Host "   バージョン: $pythonVersion" -ForegroundColor Green
                break
            }
        } catch {
            continue
        }
    }
}

if (-not $pythonCmd) {
    Write-Host "❌ Pythonが見つかりませんでした" -ForegroundColor Red
    Write-Host "`nPythonをインストールしてください:" -ForegroundColor Yellow
    Write-Host "1. https://www.python.org/downloads/ からPython 3.8以上をダウンロード" -ForegroundColor Yellow
    Write-Host "2. インストール時に「Add Python to PATH」にチェックを入れる" -ForegroundColor Yellow
    Write-Host "3. インストール後、ターミナルを再起動してこのスクリプトを再実行" -ForegroundColor Yellow
    exit 1
}

# 2. 仮想環境の作成
Write-Host "`n[2/4] 仮想環境の作成中..." -ForegroundColor Yellow

$venvPath = Join-Path $PSScriptRoot ".." "venv"
$venvPath = Resolve-Path $venvPath -ErrorAction SilentlyContinue
if (-not $venvPath) {
    $venvPath = Join-Path (Split-Path $PSScriptRoot -Parent) "venv"
}

if (Test-Path $venvPath) {
    Write-Host "⚠️  仮想環境は既に存在します: $venvPath" -ForegroundColor Yellow
    $response = Read-Host "再作成しますか？ (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Remove-Item -Path $venvPath -Recurse -Force
        Write-Host "✅ 既存の仮想環境を削除しました" -ForegroundColor Green
    } else {
        Write-Host "既存の仮想環境を使用します" -ForegroundColor Green
        $skipVenv = $true
    }
}

if (-not $skipVenv) {
    Write-Host "仮想環境を作成中: $venvPath" -ForegroundColor Cyan
    & $pythonCmd -m venv $venvPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ 仮想環境の作成に失敗しました" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ 仮想環境を作成しました" -ForegroundColor Green
}

# 3. 仮想環境の有効化
Write-Host "`n[3/4] 仮想環境の有効化中..." -ForegroundColor Yellow

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "✅ 仮想環境を有効化しました" -ForegroundColor Green
} else {
    Write-Host "❌ 仮想環境の有効化スクリプトが見つかりません" -ForegroundColor Red
    exit 1
}

# 4. パッケージのインストール
Write-Host "`n[4/4] パッケージのインストール中..." -ForegroundColor Yellow

Write-Host "pipをアップグレード中..." -ForegroundColor Cyan
& $pythonCmd -m pip install --upgrade pip

Write-Host "必要なパッケージをインストール中..." -ForegroundColor Cyan
$packages = @(
    "torch",
    "transformers",
    "pandas",
    "numpy<2.0.0",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "tqdm",
    "shap>=0.42.0"
)

foreach ($package in $packages) {
    Write-Host "  インストール中: $package" -ForegroundColor Gray
    & $pythonCmd -m pip install $package
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ セットアップ完了！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n次のステップ:" -ForegroundColor Yellow
Write-Host "1. 仮想環境を有効化: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "2. マルチタスク学習を実行: cd 00_スクリプト; python train_class_level_ordinal_llp.py" -ForegroundColor Yellow

# 必要なパッケージをインストールするスクリプト
# 使用方法: .\install_packages.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "パッケージのインストール" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 仮想環境が有効化されているか確認
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  仮想環境が有効化されていない可能性があります" -ForegroundColor Yellow
    Write-Host "仮想環境を有効化してください: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    $response = Read-Host "続行しますか？ (y/N)"
    if ($response -ne "y" -and $response -ne "Y") {
        exit 1
    }
}

# Pythonの確認
$pythonCmd = $null
$pythonCommands = @("python", "python3", "py")
foreach ($cmd in $pythonCommands) {
    try {
        $result = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $result -match "Python") {
            $pythonCmd = $cmd
            Write-Host "✅ Pythonが見つかりました: $pythonCmd ($result)" -ForegroundColor Green
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "❌ Pythonが見つかりませんでした" -ForegroundColor Red
    exit 1
}

# pipをアップグレード
Write-Host "`n[1/3] pipをアップグレード中..." -ForegroundColor Yellow
& $pythonCmd -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  pipのアップグレードに失敗しましたが、続行します" -ForegroundColor Yellow
}

# PyTorchをGPU版でインストール（CUDA版）
Write-Host "`n[2/4] PyTorch (CUDA版) をインストール中..." -ForegroundColor Yellow
Write-Host "  GPU環境用のCUDA版PyTorchをインストールします..." -ForegroundColor Cyan
Write-Host "  CUDA 11.8版をインストールします（CUDA 12.1が必要な場合は手動で変更してください）" -ForegroundColor Gray
& $pythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PyTorch (CUDA 11.8版) のインストールが完了しました" -ForegroundColor Green
} else {
    Write-Host "⚠️  CUDA 11.8版のインストールに失敗しました。CUDA 12.1版を試します..." -ForegroundColor Yellow
    & $pythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PyTorch (CUDA 12.1版) のインストールが完了しました" -ForegroundColor Green
    } else {
        Write-Host "⚠️  PyTorchのインストールに失敗しましたが、続行します" -ForegroundColor Yellow
        Write-Host "  CPU版にフォールバックする場合は fix_pytorch.ps1 を実行してください" -ForegroundColor Gray
    }
}

# requirements.txtからその他のパッケージをインストール
Write-Host "`n[3/4] その他のパッケージをインストール中..." -ForegroundColor Yellow
$packages = @(
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
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ⚠️  $package のインストールに失敗しました" -ForegroundColor Yellow
    }
}

# インストール確認
Write-Host "`n[4/4] インストール確認中..." -ForegroundColor Yellow
$testPackages = @("numpy", "pandas", "torch", "transformers")
$allOk = $true
foreach ($pkg in $testPackages) {
    if ($pkg -eq "torch") {
        # PyTorchの場合はCUDA情報も表示
        $torchCheckScript = @"
import torch
cuda_available = torch.cuda.is_available()
print(f'{torch.__version__} (CUDA: {cuda_available})')
"@
        $result = $torchCheckScript | & $pythonCmd - 2>&1
    } else {
        $result = & $pythonCmd -c "import $pkg; print($pkg.__version__)" 2>&1
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ $pkg : $result" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $pkg : インストールされていません" -ForegroundColor Red
        $allOk = $false
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
if ($allOk) {
    Write-Host "✅ パッケージのインストールが完了しました！" -ForegroundColor Green
} else {
    Write-Host "⚠️  一部のパッケージのインストールに問題があります" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan

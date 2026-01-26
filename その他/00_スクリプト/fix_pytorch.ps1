# PyTorchのDLLエラーを修正するスクリプト
# 使用方法: .\fix_pytorch.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PyTorch DLLエラーの修正" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 仮想環境が有効化されているか確認
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  仮想環境が有効化されていない可能性があります" -ForegroundColor Yellow
    Write-Host "仮想環境を有効化してください: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

# Pythonの確認
$pythonCmd = $null
$pythonCommands = @("python", "python3", "py")
foreach ($cmd in $pythonCommands) {
    try {
        $result = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $result -match "Python") {
            $pythonCmd = $cmd
            Write-Host "✅ Pythonが見つかりました: $pythonCmd" -ForegroundColor Green
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

# 既存のPyTorchをアンインストール
Write-Host "`n[1/3] 既存のPyTorchをアンインストール中..." -ForegroundColor Yellow
& $pythonCmd -m pip uninstall torch torchvision torchaudio -y
Write-Host "✅ 既存のPyTorchをアンインストールしました" -ForegroundColor Green

# GPU版のPyTorchをインストール
Write-Host "`n[2/3] PyTorch (CUDA版) をインストール中..." -ForegroundColor Yellow
Write-Host "  GPU環境用のCUDA版PyTorchをインストールします..." -ForegroundColor Cyan
Write-Host "  CUDA 11.8版をインストールします（CUDA 12.1が必要な場合は手動で変更してください）" -ForegroundColor Gray
& $pythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PyTorch (CUDA版) のインストールが完了しました" -ForegroundColor Green
} else {
    Write-Host "⚠️  CUDA 11.8版のインストールに失敗しました。CUDA 12.1版を試します..." -ForegroundColor Yellow
    & $pythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PyTorch (CUDA 12.1版) のインストールが完了しました" -ForegroundColor Green
    } else {
        Write-Host "❌ PyTorchのインストールに失敗しました" -ForegroundColor Red
        Write-Host "`nCUDAがインストールされているか確認してください:" -ForegroundColor Yellow
        Write-Host "  nvidia-smi コマンドでCUDAバージョンを確認できます" -ForegroundColor Yellow
        exit 1
    }
}

# インストール確認
Write-Host "`n[3/3] インストール確認中..." -ForegroundColor Yellow
$checkScript = @"
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA version: N/A')
    print('GPU device: N/A')
"@
$result = $checkScript | & $pythonCmd - 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PyTorchが正常に動作しています:" -ForegroundColor Green
    Write-Host $result -ForegroundColor Cyan
    if ($result -match "CUDA available: False") {
        Write-Host "`n⚠️  CUDAが利用できません。以下を確認してください:" -ForegroundColor Yellow
        Write-Host "  1. NVIDIA GPUドライバーがインストールされているか" -ForegroundColor Yellow
        Write-Host "  2. CUDA Toolkitがインストールされているか" -ForegroundColor Yellow
        Write-Host "  3. nvidia-smi コマンドでGPUが認識されているか" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ PyTorchのインポートに失敗しました" -ForegroundColor Red
    Write-Host $result -ForegroundColor Red
    Write-Host "`nMicrosoft Visual C++ Redistributableのインストールが必要な可能性があります:" -ForegroundColor Yellow
    Write-Host "https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Cyan
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ PyTorchの修正が完了しました！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

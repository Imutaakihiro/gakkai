#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU環境診断と修正スクリプト
根本原因を特定し、GPU使用を可能にする
"""

import torch
import subprocess
import sys
import os

def check_gpu_environment():
    """GPU環境を詳細に診断"""
    print("🔍 GPU環境診断を開始...")
    print("="*60)
    
    # 1. PyTorch情報
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"📦 PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎮 CUDA version: {torch.version.cuda}")
        print(f"🎮 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"🎮 GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"🎮 GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA not available")
    
    # 2. DirectML確認
    try:
        import torch_directml
        print(f"🔄 DirectML available: {torch_directml.is_available()}")
        if torch_directml.is_available():
            print(f"🔄 DirectML device: {torch_directml.device()}")
    except ImportError:
        print("🔄 DirectML not installed")
    
    # 3. NVIDIA情報
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver detected")
            print(result.stdout)
        else:
            print("❌ NVIDIA driver not found")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    
    # 4. 環境変数確認
    print("\n🌍 Environment variables:")
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    return torch.cuda.is_available()

def fix_gpu_environment():
    """GPU環境を修正"""
    print("\n🛠️ GPU環境修正を試行...")
    
    # 1. DirectMLを無効化
    os.environ['PYTORCH_DISABLE_DIRECTML'] = '1'
    print("✅ DirectML disabled")
    
    # 2. CUDA環境変数を設定
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("✅ CUDA_VISIBLE_DEVICES set to 0")
    
    # 3. PyTorchのバックエンドを強制設定
    try:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("✅ CuDNN enabled")
    except:
        print("⚠️ CuDNN not available")
    
    # 4. 再確認
    print(f"\n🔄 After fix - CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("❌ GPU still not available")
        return False

def install_cuda_pytorch():
    """CUDA対応PyTorchのインストール指示"""
    print("\n📥 CUDA対応PyTorchのインストールが必要です")
    print("="*60)
    
    # CUDA version確認
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA toolkit detected")
            # CUDA version抽出
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"CUDA version: {line}")
                    break
        else:
            print("❌ CUDA toolkit not found")
    except FileNotFoundError:
        print("❌ nvcc not found")
    
    print("\n🔧 解決方法:")
    print("1. NVIDIAドライバーを最新版に更新")
    print("2. CUDA toolkitをインストール")
    print("3. CUDA対応PyTorchをインストール:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("4. または conda使用:")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")

def main():
    """メイン実行"""
    print("🚀 GPU環境診断と修正スクリプト")
    print("="*60)
    
    # 1. 現在の状況確認
    gpu_available = check_gpu_environment()
    
    # 2. 修正試行
    if not gpu_available:
        gpu_available = fix_gpu_environment()
    
    # 3. 最終確認
    if gpu_available:
        print("\n✅ GPU環境が正常に動作しています！")
        print("🎮 SHAP分析をGPUで実行できます")
    else:
        print("\n❌ GPU環境の修正に失敗しました")
        install_cuda_pytorch()
    
    return gpu_available

if __name__ == "__main__":
    main()

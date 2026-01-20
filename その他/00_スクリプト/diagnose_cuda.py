#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA環境診断スクリプト
"""

import sys
import subprocess

def run_command(cmd):
    """コマンドを実行して結果を返す"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return "", str(e)

def main():
    print("=" * 60)
    print("🔍 CUDA環境診断")
    print("=" * 60)
    
    # Python情報
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # PyTorch情報
    try:
        import torch
        print(f"\n📦 PyTorch情報:")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  CUDA version (PyTorch): {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        
        if torch.cuda.is_available():
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Compute capability: {props.major}.{props.minor}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("  ❌ CUDA is not available")
            
    except ImportError as e:
        print(f"❌ PyTorch import error: {e}")
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
    
    # NVIDIA情報
    print(f"\n🎮 NVIDIA情報:")
    nvidia_smi, nvidia_err = run_command("nvidia-smi")
    if nvidia_smi:
        print("nvidia-smi output:")
        print(nvidia_smi)
    else:
        print(f"❌ nvidia-smi error: {nvidia_err}")
    
    # CUDA Toolkit情報
    print(f"\n🔧 CUDA Toolkit情報:")
    nvcc_version, nvcc_err = run_command("nvcc --version")
    if nvcc_version:
        print("nvcc version:")
        print(nvcc_version)
    else:
        print(f"❌ nvcc not found: {nvcc_err}")
    
    # 診断結果
    print(f"\n📋 診断結果:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "RTX 5070" in gpu_name:
                print("✅ RTX 5070 Ti detected - 最新世代GPU")
                print("⚠️  PyTorch 2.5.1では sm_120 (compute capability 12.0) が未対応")
                print("💡 解決策:")
                print("   1. PyTorch nightly版を試す")
                print("   2. 一時的にCPUで実行")
                print("   3. 古いGPUでテスト")
            else:
                print(f"✅ GPU detected: {gpu_name}")
        else:
            print("❌ CUDA not available")
    except:
        print("❌ PyTorch not available")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")

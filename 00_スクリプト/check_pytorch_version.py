#!/usr/bin/env python3
"""
PyTorch/CUDAバージョン確認スクリプト
RTX 5070 Ti対応のための環境チェック
"""

import torch
import sys
import os

def main():
    print("=" * 60)
    print("🔍 PyTorch/CUDA バージョン確認")
    print("=" * 60)
    
    # 基本情報
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version (built): {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB")
    
    print("\n" + "=" * 60)
    print("📋 RTX 5070 Ti 推奨設定")
    print("=" * 60)
    
    # バージョンチェック
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    
    print(f"現在の設定:")
    print(f"  PyTorch: {torch_version}")
    print(f"  CUDA: {cuda_version}")
    
    # 推奨設定の判定
    recommendations = []
    
    if torch_version.startswith("1.") or torch_version.startswith("2.0") or torch_version.startswith("2.1"):
        recommendations.append("⚠️ PyTorch 2.2+ へのアップグレードを推奨")
        recommendations.append("   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121")
    
    if cuda_version and cuda_version.startswith("11."):
        recommendations.append("⚠️ CUDA 12.1+ へのアップグレードを推奨")
        recommendations.append("   NVIDIAドライバを最新版に更新")
    
    if not recommendations:
        print("✅ 現在の設定はRTX 5070 Tiに適しています")
    else:
        print("🔧 推奨される改善:")
        for rec in recommendations:
            print(f"  {rec}")
    
    print("\n" + "=" * 60)
    print("🧪 簡単なGPUテスト")
    print("=" * 60)
    
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            print(f"Testing on {device}...")
            
            # 簡単な行列積テスト
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.matmul(a, b)
            
            print("✅ GPU行列積テスト成功")
            
            # メモリ使用量確認
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"GPU memory allocated: {allocated:.2f} GB")
            print(f"GPU memory cached: {cached:.2f} GB")
            
        except Exception as e:
            print(f"❌ GPUテスト失敗: {e}")
    else:
        print("❌ CUDAが利用できません")
    
    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)

if __name__ == "__main__":
    main()

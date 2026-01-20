#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU動作確認用スモークテスト

確認内容:
- PyTorch/CUDAの有効性
- GPU名・Compute Capability
- 簡単なテンソル演算・MatMulをGPUで実行
"""

import time
import torch


def main() -> None:
    print("=" * 60)
    print("🔍 GPU スモークテスト")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA (built) version: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Running on CPU only.")
        return

    device = torch.device("cuda")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    # 簡単なテンソル演算
    torch.cuda.synchronize()
    start = time.time()
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    c = a @ b  # 行列積（GPUカーネルが呼ばれる）
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"✅ MatMul OK. Shape: {tuple(c.shape)} | Time: {elapsed:.3f}s")

    # 逆伝播の一回実行
    x = torch.randn(1024, 1024, device=device, requires_grad=True)
    y = (x * x).sum()
    y.backward()
    print("✅ Backward OK (simple gradient)")

    print("\n🎉 GPUスモークテスト完了。学習の準備ができています。")


if __name__ == "__main__":
    main()



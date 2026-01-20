#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch / CUDA / GPU 環境確認スクリプト
"""

import sys


def main() -> None:
    print("=" * 60)
    print("🔍 PyTorch / CUDA / GPU 環境確認")
    print("=" * 60)

    # Python情報
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    # torch import の衝突検知
    try:
        import torch  # noqa
    except Exception as e:
        print(f"❌ torch import error: {e}")
        return

    import importlib
    import importlib.util

    # torch の実体パス
    spec = importlib.util.find_spec("torch")
    torch_path = spec.origin if spec and spec.origin else "n/a"

    # バージョン取得（__version__ が無い事例に対応）
    try:
        import importlib.metadata as md
        torch_ver = md.version("torch")
    except Exception:
        torch_ver = getattr(torch, "__version__", "unknown")

    print(f"torch version: {torch_ver}")
    print(f"torch path: {torch_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA (built) version: {getattr(torch.version, 'cuda', None)}")

    if torch.cuda.is_available():
        try:
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
        except Exception as e:
            print(f"❌ GPU query error: {e}")

    print("\n完了")


if __name__ == "__main__":
    main()



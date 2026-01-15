#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU設定確認スクリプト
"""

import torch

def check_gpu_setup():
    """GPU設定を確認"""
    print("=== GPU設定確認 ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # GPU メモリ情報
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"GPU memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"GPU compute capability: {props.major}.{props.minor}")
        
        # 現在のGPU使用状況
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        return True
    else:
        print("CUDA is not available. Will use CPU.")
        print(f"CPU cores: {torch.get_num_threads()}")
        return False

if __name__ == "__main__":
    gpu_available = check_gpu_setup()
    
    if gpu_available:
        print("\n✅ GPU is available and ready for training!")
    else:
        print("\n⚠️  GPU is not available. Training will use CPU (slower).")

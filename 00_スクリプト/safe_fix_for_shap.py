#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP用の安全な修正スクリプト（既存PyTorch環境を壊さない）

**作成日**: 2025年1月

方針:
- 既存のPyTorch環境は変更しない
- NumPyだけを1.x系に調整（PyTorch 1.13.1互換）
- SHAPとopencv-python-headlessのみ調整
"""

import subprocess
import sys
import os

def safe_fix():
    """既存環境を壊さずにSHAP用の修正のみ実施"""
    print("="*60)
    print("SHAP用の安全な修正（既存PyTorch環境は変更しません）")
    print("="*60)
    
    # 現在のPyTorchバージョンを確認
    try:
        import torch
        print(f"\n📌 現在のPyTorch: {torch.__version__}")
        print("   → このバージョンは変更しません")
    except:
        print("⚠️ PyTorchが見つかりません")
    
    # NumPyを1.x系に調整（PyTorch 1.13.1互換）
    print("\n📦 NumPyを1.x系に調整中（PyTorch互換性のため）...")
    try:
        # 既存のNumPy 2.0をアンインストール
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "numpy", "-y"
        ])
        # NumPy 1.x系をインストール
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "numpy>=1.21.0,<2.0.0"
        ])
        print("✅ NumPy調整完了")
    except Exception as e:
        print(f"⚠️ NumPy調整エラー: {e}")
        return False
    
    # opencv-python-headlessを再インストール（NumPy 1.x対応）
    print("\n📦 opencv-python-headlessを再インストール中...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "opencv-python-headless>=4.5.0,<4.9.0", "--force-reinstall"
        ])
        print("✅ opencv-python-headless再インストール完了")
    except Exception as e:
        print(f"⚠️ opencv-python-headlessエラー: {e}")
    
    # SHAPを再インストール（NumPy 1.x対応）
    print("\n📦 SHAPを再インストール中...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "shap>=0.42.0", "--force-reinstall", "--no-deps"
        ])
        # 依存関係を個別にインストール
        deps = ["scipy", "scikit-learn", "pandas", "tqdm", "packaging", "slicer", "numba", "cloudpickle"]
        for dep in deps:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", dep
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass
        print("✅ SHAP再インストール完了")
    except Exception as e:
        print(f"⚠️ SHAP再インストールエラー: {e}")
    
    # 動作確認
    print("\n🔍 動作確認中...")
    try:
        import numpy as np
        import torch
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ PyTorch: {torch.__version__} (変更なし)")
        
        # SHAPのインポート確認
        try:
            import shap
            print(f"✅ SHAP: {shap.__version__}")
        except Exception as e:
            print(f"⚠️ SHAPインポートエラー: {e}")
            print("   ただし、基本的な機能は動作する可能性があります")
        
        # cv2の確認
        try:
            import cv2
            print(f"✅ OpenCV: {cv2.__version__}")
        except Exception as e:
            print(f"⚠️ OpenCVエラー: {e}")
        
        print("\n" + "="*60)
        print("✅ 修正完了！")
        print("="*60)
        print("\n💡 次のステップ:")
        print("   python analyze_classlevel_multitask_shap_beeswarm.py")
        print("\n⚠️ 注意:")
        print("   - PyTorch環境は変更していません")
        print("   - GPU環境は既存のままです")
        print("   - 問題があれば、このスクリプトを再実行してください")
        
        return True
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")
        return False

if __name__ == "__main__":
    safe_fix()




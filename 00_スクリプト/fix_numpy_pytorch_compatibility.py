#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumPyとPyTorchの互換性修正スクリプト

**作成日**: 2025年1月

問題: NumPy 2.0.2とPyTorch 1.13.1の互換性問題
解決策: NumPyを1.x系にダウングレード
"""

import subprocess
import sys

def fix_compatibility():
    """NumPyとPyTorchの互換性を修正"""
    print("NumPyとPyTorchの互換性を修正中...")
    
    # NumPyを1.x系にダウングレード
    print("\n📦 NumPy 1.x系にダウングレード中...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "numpy>=1.21.0,<2.0.0", "--force-reinstall"
        ])
        print("✅ NumPyダウングレード完了")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ NumPyダウングレードエラー: {e}")
        return False
    
    # opencv-python-headlessを再インストール（NumPy 1.x対応版）
    print("\n📦 opencv-python-headlessを再インストール中...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "opencv-python-headless>=4.5.0,<4.9.0", "--force-reinstall"
        ])
        print("✅ opencv-python-headless再インストール完了")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ opencv-python-headless再インストールエラー: {e}")
    
    # 動作確認
    print("\n🔍 動作確認中...")
    try:
        import numpy as np
        import torch
        import shap
        print(f"✅ NumPy version: {np.__version__}")
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ SHAP version: {shap.__version__}")
        
        # PyTorchとtransformersの互換性確認
        try:
            from transformers import BertModel
            print("✅ transformers正常にインポート可能")
        except Exception as e:
            print(f"⚠️ transformersインポート警告: {e}")
            print("💡 PyTorchのアップグレードを検討してください")
        
        print("\n✅ 互換性修正完了！")
        return True
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")
        return False

if __name__ == "__main__":
    fix_compatibility()




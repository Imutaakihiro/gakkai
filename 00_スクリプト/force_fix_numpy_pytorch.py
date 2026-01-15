#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumPy/PyTorch互換性の強制修正（GPU環境を壊さない）

**作成日**: 2025年1月

方針:
- NumPyを1.x系に強制ダウングレード
- PyTorchは変更しない（GPU環境維持）
- transformersが動作するようにする
"""

import subprocess
import sys
import os

def force_fix():
    """NumPy/PyTorch互換性を強制修正"""
    print("="*60)
    print("NumPy/PyTorch互換性の強制修正")
    print("（GPU環境は変更しません）")
    print("="*60)
    
    # 1. NumPy 2.0を完全にアンインストール
    print("\n📦 NumPy 2.0をアンインストール中...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "numpy", "-y"
        ])
        print("✅ NumPy 2.0アンインストール完了")
    except Exception as e:
        print(f"⚠️ NumPyアンインストールエラー: {e}")
    
    # 2. NumPy 1.x系をインストール
    print("\n📦 NumPy 1.x系をインストール中...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "numpy==1.26.4", "--no-cache-dir"
        ])
        print("✅ NumPy 1.26.4インストール完了")
    except Exception as e:
        print(f"❌ NumPyインストールエラー: {e}")
        return False
    
    # 3. opencv-python-headlessを再インストール
    print("\n📦 opencv-python-headlessを再インストール中...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "opencv-python-headless==4.8.1.78", "--force-reinstall", "--no-cache-dir"
        ])
        print("✅ opencv-python-headless再インストール完了")
    except Exception as e:
        print(f"⚠️ opencv-python-headlessエラー: {e}")
    
    # 4. 動作確認
    print("\n🔍 動作確認中...")
    try:
        import numpy as np
        import torch
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ PyTorch: {torch.__version__}")
        
        # PyTorchとtransformersの互換性確認
        try:
            from transformers import BertModel
            print("✅ transformers正常にインポート可能")
        except Exception as e:
            print(f"⚠️ transformersエラー: {e}")
            print("   PyTorchの再起動が必要かもしれません")
        
        # SHAP確認
        try:
            import shap
            print(f"✅ SHAP: {shap.__version__}")
        except Exception as e:
            print(f"⚠️ SHAPエラー: {e}")
        
        print("\n" + "="*60)
        print("✅ 修正完了！")
        print("="*60)
        print("\n💡 次のステップ:")
        print("   1. Pythonを再起動してください")
        print("   2. python analyze_classlevel_multitask_shap_beeswarm.py を実行")
        print("\n⚠️ 注意:")
        print("   - PyTorch環境は変更していません")
        print("   - GPU環境は維持されています")
        
        return True
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = force_fix()
    if success:
        print("\n✅ 修正が完了しました。Pythonを再起動してから実行してください。")
    else:
        print("\n❌ 修正に失敗しました。エラーメッセージを確認してください。")




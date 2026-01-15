#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP依存関係の修正スクリプト

**作成日**: 2025年1月

問題: cv2とnumpyの互換性エラー
解決策: opencv-python-headlessをインストール（cv2の軽量版）
"""

import subprocess
import sys

def fix_shap_dependencies():
    """SHAPの依存関係を修正"""
    print("SHAP依存関係を修正中...")
    
    packages = [
        "numpy>=1.21.0,<2.0.0",  # numpyのバージョンを制限（PyTorch 1.13互換）
        "opencv-python-headless>=4.5.0,<4.9.0",  # cv2の軽量版（NumPy 1.x対応）
    ]
    
    for package in packages:
        print(f"\n📦 {package} をインストール中...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
            print(f"✅ {package} インストール完了")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ {package} インストールエラー: {e}")
    
    # SHAPの再インストール
    print("\n📦 SHAPを再インストール中...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "shap>=0.42.0"])
        print("✅ SHAP再インストール完了")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ SHAP再インストールエラー: {e}")
    
    # 動作確認
    print("\n🔍 動作確認中...")
    try:
        import shap
        import numpy as np
        print(f"✅ SHAP version: {shap.__version__}")
        print(f"✅ NumPy version: {np.__version__}")
        print("✅ 全ての依存関係が正常にインストールされました！")
        return True
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")
        return False

if __name__ == "__main__":
    fix_shap_dependencies()


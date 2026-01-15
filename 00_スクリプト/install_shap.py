#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAPライブラリのインストールスクリプト

**作成日**: 2025年1月
"""

import subprocess
import sys

def install_shap():
    """SHAPをインストール"""
    print("SHAPライブラリをインストール中...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap>=0.42.0"])
        print("✅ SHAPのインストールが完了しました！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ インストールエラー: {e}")
        return False

if __name__ == "__main__":
    install_shap()




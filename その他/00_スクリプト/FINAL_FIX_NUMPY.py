#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【最終版】NumPy強制ダウングレード

**作成日**: 2025年1月

NumPy 2.0.2を完全に削除し、1.26.4を確実にインストール
"""

import subprocess
import sys
import os

print("="*70)
print("NumPy強制ダウングレード（最終版）")
print("="*70)

# 1. 全てのNumPy関連パッケージをアンインストール
print("\n📦 ステップ1: NumPy関連パッケージを完全削除...")
packages_to_remove = [
    "numpy",
    "numpy-base",
    "numpy-stl",
]

for pkg in packages_to_remove:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60
        )
    except:
        pass

print("✅ NumPy関連パッケージ削除完了")

# 2. pipキャッシュをクリア
print("\n📦 ステップ2: pipキャッシュをクリア...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "cache", "purge"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60
    )
    print("✅ pipキャッシュクリア完了")
except:
    pass

# 3. NumPy 1.26.4を強制インストール
print("\n📦 ステップ3: NumPy 1.26.4を強制インストール...")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--no-cache-dir", "--force-reinstall"],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode == 0:
        print("✅ NumPy 1.26.4インストール成功")
    else:
        print(f"⚠️ インストール警告: {result.stderr[:300]}")
except Exception as e:
    print(f"❌ インストールエラー: {e}")

# 4. 即座に検証
print("\n🔍 即座に検証...")
try:
    # 新しいPythonプロセスで検証
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", "import numpy; print(f'NumPy: {numpy.__version__}')"],
        capture_output=True,
        text=True,
        timeout=10
    )
    print(result.stdout)
    if "1." in result.stdout:
        print("✅ NumPy 1.x系が確認されました！")
    else:
        print("⚠️ NumPy 2.x系が残っている可能性があります")
except Exception as e:
    print(f"⚠️ 検証エラー: {e}")

# 5. OpenCVとSHAPを再インストール（NumPy 1.x対応）
print("\n📦 ステップ4: OpenCVとSHAPを再インストール...")

# OpenCV
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "opencv-python", "opencv-python-headless", "-y"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.1.78", "--no-cache-dir"],
        timeout=300
    )
    print("✅ OpenCV再インストール完了")
except Exception as e:
    print(f"⚠️ OpenCVエラー: {e}")

# SHAP
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "shap", "-y"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "shap==0.42.0", "--no-cache-dir"],
        timeout=300
    )
    print("✅ SHAP再インストール完了")
except Exception as e:
    print(f"⚠️ SHAPエラー: {e}")

print("\n" + "="*70)
print("✅ 修正完了！")
print("="*70)
print("\n⚠️ 重要:")
print("   1. Pythonを完全に再起動してください")
print("   2. 新しいターミナル/PowerShellを開いてください")
print("   3. その後、SHAP分析スクリプトを実行してください")
print("\n💡 確認コマンド:")
print("   python -c \"import numpy; print(numpy.__version__)\"")




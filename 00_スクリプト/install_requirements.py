#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
必要なライブラリをインストールするスクリプト
"""

import subprocess
import sys

def install_package(package):
    """パッケージをインストール"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} のインストールが完了しました")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} のインストールに失敗しました: {e}")
        return False

def main():
    """メイン処理"""
    print("=" * 60)
    print("必要なライブラリをインストール中...")
    print("=" * 60)
    
    # 必要なパッケージリスト
    packages = [
        "fugashi",
        "ipadic",
        "unidic-lite",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "transformers",
        "torch",
        "seaborn",
        "japanize-matplotlib"
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        print(f"\n📦 {package} をインストール中...")
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"インストール完了: {success_count}/{total_count} パッケージ")
    print("=" * 60)
    
    if success_count == total_count:
        print("✅ すべてのパッケージが正常にインストールされました！")
        print("マルチタスク学習を実行できます。")
    else:
        print("⚠️  一部のパッケージのインストールに失敗しました。")
        print("手動でインストールしてください。")

if __name__ == "__main__":
    main()

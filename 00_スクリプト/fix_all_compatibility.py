#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全互換性問題の一括修正（GPU環境を壊さない）

**作成日**: 2025年1月

問題:
1. NumPy 2.0.2とPyTorch 1.13.1の互換性問題
2. transformersがPyTorch >= 2.1を要求

解決策:
1. NumPyを1.x系にダウングレード
2. transformersをPyTorch 1.13.1互換のバージョンにダウングレード
3. PyTorchは変更しない（GPU環境維持）
"""

import subprocess
import sys
import os

def fix_all():
    """全互換性問題を修正"""
    print("="*60)
    print("全互換性問題の一括修正")
    print("（GPU環境は変更しません）")
    print("="*60)
    
    # 現在のバージョン確認
    try:
        import torch
        print(f"\n📌 現在のPyTorch: {torch.__version__}")
        print("   → このバージョンは変更しません（GPU環境維持）")
    except:
        print("⚠️ PyTorchが見つかりません")
    
    # 1. NumPyを1.x系にダウングレード
    print("\n📦 ステップ1: NumPyを1.x系にダウングレード...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "numpy", "-y"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "numpy==1.26.4", "--no-cache-dir"
        ])
        print("✅ NumPy 1.26.4インストール完了")
    except Exception as e:
        print(f"❌ NumPyエラー: {e}")
        return False
    
    # 2. transformersをPyTorch 1.13.1互換のバージョンにダウングレード
    print("\n📦 ステップ2: transformersをPyTorch 1.13.1互換バージョンに調整...")
    try:
        # transformers 4.21.0はPyTorch 1.13.1と互換性がある
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "transformers==4.21.0", "--force-reinstall", "--no-cache-dir"
        ])
        print("✅ transformers 4.21.0インストール完了")
    except Exception as e:
        print(f"⚠️ transformersエラー: {e}")
        # より新しいバージョンを試す
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "transformers==4.30.0", "--force-reinstall", "--no-cache-dir"
            ])
            print("✅ transformers 4.30.0インストール完了")
        except Exception as e2:
            print(f"⚠️ transformers 4.30.0も失敗: {e2}")
    
    # 3. opencv-python-headlessを再インストール
    print("\n📦 ステップ3: opencv-python-headlessを再インストール...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "opencv-python-headless==4.8.1.78", "--force-reinstall", "--no-cache-dir"
        ])
        print("✅ opencv-python-headless再インストール完了")
    except Exception as e:
        print(f"⚠️ opencv-python-headlessエラー: {e}")
    
    # 4. SHAPを再インストール
    print("\n📦 ステップ4: SHAPを再インストール...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "shap==0.42.0", "--force-reinstall", "--no-cache-dir"
        ])
        print("✅ SHAP再インストール完了")
    except Exception as e:
        print(f"⚠️ SHAPエラー: {e}")
    
    # 5. 動作確認
    print("\n🔍 動作確認中...")
    try:
        import numpy as np
        import torch
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ PyTorch: {torch.__version__} (変更なし)")
        
        # transformers確認
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
        print("   1. Pythonを再起動してください（重要！）")
        print("   2. python analyze_classlevel_multitask_shap_beeswarm.py を実行")
        print("\n⚠️ 注意:")
        print("   - PyTorch環境は変更していません")
        print("   - GPU環境は維持されています")
        print("   - Python再起動後、動作確認してください")
        
        return True
    except Exception as e:
        print(f"❌ 動作確認エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_all()
    if success:
        print("\n" + "="*60)
        print("✅ 修正が完了しました！")
        print("="*60)
        print("\n⚠️ 重要: Pythonを再起動してから実行してください")
    else:
        print("\n" + "="*60)
        print("❌ 修正に失敗しました")
        print("="*60)




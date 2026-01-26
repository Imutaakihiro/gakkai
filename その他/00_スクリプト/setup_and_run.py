#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
環境セットアップとマルチタスク学習の実行スクリプト
使用方法: python setup_and_run.py
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """コマンドを実行して結果を表示"""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"{'='*60}")
    print(f"実行中: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: {e}")
        if e.stdout:
            print(f"出力: {e.stdout}")
        if e.stderr:
            print(f"エラー出力: {e.stderr}")
        return False

def check_package(package_name):
    """パッケージがインストールされているか確認"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("="*60)
    print("🚀 マルチタスク学習環境セットアップ & 実行")
    print("="*60)
    
    # 1. Python環境の確認
    print(f"\n✅ Pythonバージョン: {sys.version}")
    
    # 2. PyTorchの確認とインストール
    if not check_package("torch"):
        print("\n📦 PyTorchをインストール中...")
        # CUDA版を試す
        if not run_command(
            "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "PyTorch (CUDA 11.8版) のインストール"
        ):
            print("⚠️  CUDA 11.8版のインストールに失敗。CUDA 12.1版を試します...")
            if not run_command(
                "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                "PyTorch (CUDA 12.1版) のインストール"
            ):
                print("⚠️  CUDA版のインストールに失敗。CPU版をインストールします...")
                run_command(
                    "python -m pip install torch torchvision torchaudio",
                    "PyTorch (CPU版) のインストール"
                )
    else:
        import torch
        print(f"\n✅ PyTorchは既にインストールされています: {torch.__version__}")
    
    # 3. PyTorchの動作確認
    print("\n🔍 PyTorchの動作確認中...")
    try:
        import torch
        print(f"  PyTorchバージョン: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✅ CUDA利用可能: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  ✅ MPS (Apple Silicon) 利用可能")
        else:
            print(f"  ⚠️  GPU利用不可: CPUモードで実行されます（時間がかかります）")
    except Exception as e:
        print(f"  ❌ PyTorchの確認中にエラー: {e}")
    
    # 4. その他のパッケージの確認とインストール
    required_packages = {
        "transformers": "transformers",
        "pandas": "pandas",
        "numpy": "numpy<2.0.0",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
        "shap": "shap>=0.42.0"
    }
    
    missing_packages = []
    for module_name, package_name in required_packages.items():
        if not check_package(module_name):
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n📦 不足しているパッケージをインストール中: {', '.join(missing_packages)}")
        packages_str = " ".join(missing_packages)
        run_command(
            f"python -m pip install {packages_str}",
            "パッケージのインストール"
        )
    else:
        print("\n✅ すべての必要なパッケージがインストールされています")
    
    # 5. スクリプトの存在確認
    script_path = os.path.join(os.path.dirname(__file__), "train_class_level_ordinal_llp.py")
    if not os.path.exists(script_path):
        print(f"\n❌ スクリプトが見つかりません: {script_path}")
        return False
    
    # 6. マルチタスク学習の実行
    print("\n" + "="*60)
    print("🎯 マルチタスク学習を開始します")
    print("="*60)
    print(f"スクリプト: {script_path}")
    print("\n⚠️  実行には30-60分かかる可能性があります")
    print("   中断する場合は Ctrl+C を押してください\n")
    
    # スクリプトを実行
    os.chdir(os.path.dirname(script_path))
    result = subprocess.run([sys.executable, "train_class_level_ordinal_llp.py"])
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ マルチタスク学習が正常に完了しました！")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ マルチタスク学習中にエラーが発生しました")
        print("="*60)
    
    return result.returncode == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

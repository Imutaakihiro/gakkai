#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch 2.x + Transformers 依存関係修正スクリプト
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """コマンドを実行して結果を表示"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"{'='*50}")
    print(f"実行中: {cmd}")
    
    try:
        # Windows環境でのエンコーディング問題を回避
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        print("✅ 成功")
        if result.stdout:
            print(f"出力: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: {e}")
        if e.stderr:
            print(f"エラー詳細: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False

def check_versions():
    """バージョン確認"""
    print(f"\n{'='*50}")
    print("📊 バージョン確認")
    print(f"{'='*50}")
    
    try:
        import torch
        import transformers
        import tokenizers
        
        print(f"torch: {torch.__version__} / CUDA: {torch.version.cuda}")
        print(f"transformers: {transformers.__version__}")
        print(f"tokenizers: {tokenizers.__version__}")
        print(f"CUDA利用可能: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            # アーキテクチャリストを確認（sm_120が含まれていればOK）
            try:
                arch_list = getattr(torch.cuda, 'get_arch_list', lambda: 'n/a')()
                print(f"サポートアーキテクチャ: {arch_list}")
                if 'sm_120' in str(arch_list):
                    print("✅ RTX 5070 Ti (sm_120) 対応済み！")
                else:
                    print("⚠️ sm_120が含まれていません - 'no kernel image'エラーの可能性")
            except Exception as e:
                print(f"arch_list取得エラー: {e}")
                print("⚠️ アーキテクチャ確認ができません")
            
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")

def main():
    """メイン処理"""
    print("🚀 PyTorch ナイトリー版 + Transformers 依存関係修正開始")
    print("🎯 RTX 5070 Ti (sm_120) 対応のためtorch単体nightly版を使用")
    print("="*60)
    
    # 1. 現在の環境確認
    print(f"\n📍 Python実行パス: {sys.executable}")
    
    # 2. 既存のPyTorch系を完全削除
    if not run_command("python -m pip uninstall -y torch torchvision torchaudio", 
                      "既存PyTorch系アンインストール"):
        print("⚠️ PyTorchアンインストールでエラー（続行）")
    
    if not run_command("python -m pip cache purge", 
                      "pipキャッシュクリア"):
        print("⚠️ キャッシュクリアでエラー（続行）")
    
    # 3. torch単体のnightly版をインストール（torchvision/torchaudioは不要）
    pytorch_nightly_cmd = ("python -m pip install --pre --index-url "
                          "https://download.pytorch.org/whl/nightly/cu124 "
                          "torch")
    
    if not run_command(pytorch_nightly_cmd, "PyTorch ナイトリー版（torch単体）インストール"):
        print("❌ PyTorch ナイトリー版インストール失敗")
        print("💡 代替案: 特定日付のnightly版を明示指定")
        print("   python -m pip index versions torch --index-url https://download.pytorch.org/whl/nightly/cu124")
        print("   python -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu124 torch==2.7.0.devYYYYMMDD+cu124")
        return False
    
    # 4. 必要な依存パッケージをインストール
    deps_cmd = ("python -m pip install -U transformers tokenizers scikit-learn matplotlib pandas numpy fugashi ipadic unidic-lite")
    
    if not run_command(deps_cmd, "依存パッケージ インストール"):
        print("❌ 依存パッケージ インストール失敗")
        return False
    
    # 5. バージョン確認
    check_versions()
    
    print(f"\n{'='*60}")
    print("🎉 完了！学習スクリプトを実行してください")
    print("python 00_スクリプト\\train_class_level_multitask.py")
    print(f"{'='*60}")
    print("💡 ポイント:")
    print("- arch_list に 'sm_120' が含まれていればOK")
    print("- torchvision/torchaudioは不要のため除外")
    print("- NLP学習ではtorch単体で十分")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ すべて正常に完了しました")
        else:
            print("\n❌ エラーが発生しました")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

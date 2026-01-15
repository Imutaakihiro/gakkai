#!/usr/bin/env python3
"""
DirectML環境の動作確認スクリプト
"""

import torch
import torch_directml as dml

def test_directml_setup():
    """DirectML環境の動作確認"""
    print("🔍 DirectML環境の動作確認を開始...")
    print()
    
    # PyTorch情報
    print(f"🧠 PyTorch version: {torch.__version__}")
    print(f"🧩 DirectML available: {dml.is_available()}")
    
    if not dml.is_available():
        print("❌ DirectMLが利用できません")
        return False
    
    # デバイス取得
    device = dml.device()
    print(f"🚀 Using device: {device}")
    
    # テスト演算
    try:
        print("🧮 DirectML計算テストを実行中...")
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        z = x @ y
        print(f"✅ DirectML計算成功: {z.device}")
        
        # メモリ使用量確認
        print(f"📊 GPU メモリ使用量: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ DirectML計算エラー: {e}")
        return False

if __name__ == "__main__":
    success = test_directml_setup()
    
    if success:
        print("\n🎉 DirectML環境の動作確認完了！")
        print("学習スクリプトでGPUが使用可能です。")
    else:
        print("\n⚠️ DirectML環境に問題があります。")
        print("CPUで実行するか、環境を再確認してください。")

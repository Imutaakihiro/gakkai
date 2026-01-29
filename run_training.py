#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マルチタスク学習実行ラッパースクリプト
文字エンコーディングの問題を回避するため
"""
import os
import sys

# プロジェクトルートに移動
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# スクリプトディレクトリをパスに追加
script_dir = os.path.join(project_root, "その他", "00_スクリプト")
sys.path.insert(0, script_dir)

# 学習スクリプトを実行
if __name__ == "__main__":
    import train_class_level_ordinal_llp
    train_class_level_ordinal_llp.main()

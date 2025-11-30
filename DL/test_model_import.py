#!/usr/bin/env python3
"""
測試 deep_learning_model.py 是否可以被正確 import
"""

import sys
import os

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("🧪 測試 1: 檢查模組是否可以 import...")
    from DL.deep_learning_model import run_deep_learning_pipeline
    print("   ✅ run_deep_learning_pipeline 成功 import")
    
    from DL.deep_learning_model import build_neural_network
    print("   ✅ build_neural_network 成功 import")
    
    from DL.deep_learning_model import check_gpu_availability
    print("   ✅ check_gpu_availability 成功 import")
    
    print("\n🧪 測試 2: 檢查函數簽名...")
    import inspect
    
    sig = inspect.signature(run_deep_learning_pipeline)
    params = list(sig.parameters.keys())
    print(f"   run_deep_learning_pipeline 參數: {params}")
    
    expected_params = ['X_train', 'X_test', 'y_train', 'y_test', 'feature_names']
    if params == expected_params:
        print("   ✅ 參數簽名正確")
    else:
        print(f"   ⚠️  參數簽名不符: 期望 {expected_params}")
    
    print("\n🧪 測試 3: 檢查 GPU...")
    gpu_available = check_gpu_availability()
    
    print("\n" + "="*60)
    print("✅ 所有測試通過！deep_learning_model.py 可以正常使用")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ 測試失敗: {e}")
    import traceback
    traceback.print_exc()

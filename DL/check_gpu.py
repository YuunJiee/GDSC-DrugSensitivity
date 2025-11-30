#!/usr/bin/env python3
"""
GPU 硬體檢測腳本
快速檢查 TensorFlow 是否能存取 GPU（包括 Apple Silicon 的 Metal 後端）
"""

from DL.deep_learning_model import check_gpu_availability

if __name__ == "__main__":
    print("🔍 開始檢測硬體設備...")
    
    # 執行檢測
    device_info = check_gpu_availability()
    
    # 顯示摘要
    print("\n" + "="*60)
    print("📊 檢測結果摘要")
    print("="*60)
    
    if device_info['gpu_available']:
        print(f"✅ GPU 加速: 已啟用")
        print(f"   GPU 數量: {device_info['num_gpus']}")
        print(f"   TensorFlow: {device_info['tf_version']}")
        print(f"\n🚀 你的模型訓練將使用 GPU 加速！")
    else:
        print(f"⚠️  GPU 加速: 未啟用")
        print(f"   TensorFlow: {device_info['tf_version']}")
        print(f"\n💡 如需啟用 GPU (Apple Silicon):")
        print(f"   pip install tensorflow-metal")
    
    print("="*60)


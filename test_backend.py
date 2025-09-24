#!/usr/bin/env python3
"""
Test script to verify the Python backend is working correctly.
"""

import os
import sys
import json

def test_backend():
    """Test if the backend components are working."""
    print("🧪 Testing Person Detection Backend...")
    
    # Test 1: Check if model file exists
    model_path = "person_detector_final.pth"
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        print(f"   Size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    else:
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Test 2: Check if required directories exist
    required_dirs = ["image", "uploads", "outputs"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"⚠️  Directory missing: {dir_name} (will be created)")
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"✅ Created directory: {dir_name}")
            except Exception as e:
                print(f"❌ Failed to create directory: {e}")
    
    # Test 3: Check Python imports
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ Torchvision version: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ Torchvision import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ PIL/Pillow imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    # Test 4: Check if detect_person.py exists and is executable
    if os.path.exists("detect_person.py"):
        print("✅ detect_person.py found")
    else:
        print("❌ detect_person.py not found")
        return False
    
    print("\n🎉 Backend test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_backend()
    sys.exit(0 if success else 1)

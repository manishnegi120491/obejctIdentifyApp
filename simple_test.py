#!/usr/bin/env python3
"""
Simple test to verify Python environment is working.
"""

import sys
import os

def main():
    print("Python version:", sys.version)
    print("Current directory:", os.getcwd())
    print("Files in directory:", os.listdir('.'))
    
    # Test basic imports
    try:
        import torch
        print("[OK] PyTorch imported successfully")
        print("PyTorch version:", torch.__version__)
    except ImportError as e:
        print("[ERROR] PyTorch import failed:", e)
        return False
    
    try:
        import torchvision
        print("[OK] Torchvision imported successfully")
        print("Torchvision version:", torchvision.__version__)
    except ImportError as e:
        print("[ERROR] Torchvision import failed:", e)
        return False
    
    try:
        from PIL import Image
        print("[OK] PIL imported successfully")
    except ImportError as e:
        print("[ERROR] PIL import failed:", e)
        return False
    
    # Check if model file exists
    model_path = "person_detector_final.pth"
    if os.path.exists(model_path):
        print(f"[OK] Model file found: {model_path}")
        print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    else:
        print(f"[ERROR] Model file not found: {model_path}")
        print("Available files:", [f for f in os.listdir('.') if f.endswith('.pth')])
    
    print("[OK] Simple test completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

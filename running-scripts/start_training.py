#!/usr/bin/env python3
"""
Quick start script for training your person detection model.
This script checks your setup and starts training.
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'torch',
        'torchvision', 
        'torchaudio',
        'pycocotools',
        'PIL',
        'tqdm',
        'matplotlib',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nPlease install them using:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_data_structure():
    """Check if data files and directories exist"""
    required_paths = [
        ("val2017", "Image directory"),
        ("annotations/instances_val2017_person.json", "Person annotation file"),
        ("dataset.py", "Dataset class"),
        ("train_person.py", "Training script")
    ]
    
    missing_files = []
    
    for path, description in required_paths:
        if not os.path.exists(path):
            missing_files.append((path, description))
    
    if missing_files:
        print("❌ Missing required files/directories:")
        for path, desc in missing_files:
            print(f"   - {path} ({desc})")
        return False
    
    print("✅ All required data files and directories found!")
    
    # Check if val2017 has images
    image_files = [f for f in os.listdir("val2017") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) == 0:
        print("❌ No image files found in val2017 directory!")
        return False
    
    print(f"✅ Found {len(image_files)} images in val2017 directory")
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Device count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
            return True
    except ImportError:
        print("❌ PyTorch not installed properly")
        return False

def main():
    """Main function to check setup and start training"""
    print("🚀 Person Detection Model Training Setup")
    print("=" * 50)
    
    # Check requirements
    print("\n1. Checking requirements...")
    if not check_requirements():
        return
    
    # Check data structure
    print("\n2. Checking data structure...")
    if not check_data_structure():
        return
    
    # Check GPU
    print("\n3. Checking GPU availability...")
    check_gpu()
    
    print("\n" + "=" * 50)
    print("🎉 Setup looks good! Ready to start training.")
    print("\nTo start training, run:")
    print("   python train_person.py")
    print("\nOr continue with this script to start training automatically...")
    
    # Ask user if they want to start training
    response = input("\nStart training now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting training...")
        try:
            # Run the training script
            subprocess.run([sys.executable, "train_person.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with error: {e}")
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
    else:
        print("\n👋 Training not started. You can run 'python train_person.py' when ready!")

if __name__ == "__main__":
    main()

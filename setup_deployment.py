#!/usr/bin/env python3
"""
Setup script for deployment.
Downloads required model files and prepares the environment.
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_deployment():
    """Setup the deployment environment."""
    print("🚀 Setting up Person Detection App for deployment...")
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Download model file
    if not run_command("python download_model.py", "Downloading model file"):
        print("⚠️  Model download failed. You may need to train a model first.")
    
    # Install Node.js dependencies
    if not run_command("npm install", "Installing Node.js dependencies"):
        return False
    
    # Install client dependencies
    if not run_command("cd client && npm install", "Installing client dependencies"):
        return False
    
    # Build React app
    if not run_command("cd client && npm run build", "Building React app"):
        return False
    
    print("✅ Deployment setup completed successfully!")
    print("🎉 Your app is ready to run with: node server.js")
    return True

if __name__ == "__main__":
    success = setup_deployment()
    sys.exit(0 if success else 1)

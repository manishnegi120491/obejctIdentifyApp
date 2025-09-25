#!/usr/bin/env python3
"""
Download the trained model file for deployment.
This script downloads the model file that was excluded from Git.
"""

import os
import requests
import sys

def download_model():
    """Download the person detection model."""
    model_url = "https://github.com/manishnegi120491/obejctIdentifyApp/releases/download/v1.0/person_detector_final.pth"
    model_path = "../models/person_detector_final.pth"
    
    if os.path.exists(model_path):
        print(f"Model file {model_path} already exists.")
        return True
    
    print(f"Downloading model from {model_url}...")
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Model downloaded successfully to {model_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please ensure the model file is available or train a new model.")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)

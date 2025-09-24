#!/bin/bash
set -e

echo "🚀 Starting build process..."

# Check if we're in the right directory
echo "📁 Current directory: $(pwd)"
echo "📁 Files in directory:"
ls -la

# Check Python availability
echo "🐍 Checking Python..."
if command -v python3 &> /dev/null; then
    echo "✅ Python3 found: $(python3 --version)"
else
    echo "❌ Python3 not found, trying python..."
    if command -v python &> /dev/null; then
        echo "✅ Python found: $(python --version)"
        # Create symlink for python3
        ln -s $(which python) /usr/local/bin/python3
    else
        echo "❌ No Python found!"
        exit 1
    fi
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Download model if needed
echo "📥 Checking for model file..."
if [ ! -f "person_detector_final.pth" ]; then
    echo "📥 Downloading model..."
    python3 download_model.py
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install client dependencies
echo "📦 Installing client dependencies..."
cd client
npm install

# Build React app
echo "🏗️ Building React app..."
npm run build

echo "✅ Build completed successfully!"
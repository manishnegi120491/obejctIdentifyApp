#!/bin/bash
set -e

echo "🚀 Starting build process..."

# Check Python and pip
echo "📋 Checking Python setup..."
python --version
python -m pip --version

# Upgrade pip and install Python dependencies
echo "📦 Installing Python dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Check Node.js and npm
echo "📋 Checking Node.js setup..."
node --version
npm --version

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

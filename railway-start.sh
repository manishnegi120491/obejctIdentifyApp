#!/bin/bash

# Railway startup script
echo "🚀 Starting Person Detection App on Railway..."

# Check Python installation
echo "📋 Checking Python installation..."
python3 --version
which python3

# Check Node.js installation  
echo "📋 Checking Node.js installation..."
node --version
npm --version

# Install Python dependencies if needed
echo "📦 Installing Python dependencies..."
python3 -m pip install -r requirements.txt

# Install Node.js dependencies if needed
echo "📦 Installing Node.js dependencies..."
npm install

# Build React app
echo "🏗️ Building React app..."
cd client && npm install && npm run build && cd ..

# Start the server
echo "🚀 Starting server..."
node server.js

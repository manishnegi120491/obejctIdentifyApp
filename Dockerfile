# Use Python 3.11 as base image
FROM python:3.11-slim

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package files
COPY package*.json ./
RUN npm install

# Copy client files
COPY client/package*.json ./client/
RUN cd client && npm install

# Copy source code
COPY . .

# Download model if needed
RUN python3 download_model.py || echo "Model download failed, will try at runtime"

# Build React app
RUN cd client && npm run build

# Expose port
EXPOSE 5000

# Start the application
CMD ["node", "server.js"]

# Person Detection App

A modern React Node.js application for AI-powered person detection using PyTorch and Faster R-CNN. Upload images and get real-time person detection results with smart filtering to reduce false positives.

## Features

- 🤖 **AI-Powered Detection**: Uses PyTorch Faster R-CNN model for accurate person detection
- 🎯 **Smart Filtering**: Advanced filtering to reduce false positives (horses, objects, etc.)
- 📱 **Modern UI**: Beautiful React frontend with drag-and-drop image upload
- ⚡ **Real-time Processing**: Fast image processing with live progress indicators
- 📊 **Detailed Results**: Comprehensive detection statistics and bounding box information
- 🔧 **Easy Setup**: Simple installation and configuration

## Prerequisites

Before running this application, make sure you have:

- **Node.js** (v14 or higher)
- **Python** (v3.7 or higher)
- **PyTorch** and related dependencies
- **Trained model file**: `person_detector_final.pth`

## Installation

### 1. Clone and Setup

```bash
# Navigate to your project directory
cd your-project-directory

# Install backend dependencies
npm install

# Install frontend dependencies
cd client
npm install
cd ..
```

### 2. Python Dependencies

Install the required Python packages:

```bash
pip install torch torchvision pillow numpy
```

### 3. Model Setup

Place your trained PyTorch model file (`person_detector_final.pth`) in the root directory of the project.

## Running the Application

### Development Mode

1. **Start the backend server:**
   ```bash
   npm run dev
   ```
   This starts the Node.js server on `http://localhost:5000`

2. **Start the React frontend:**
   ```bash
   npm run client
   ```
   This starts the React development server on `http://localhost:3000`

3. **Open your browser:**
   Navigate to `http://localhost:3000` to use the application

### Production Mode

1. **Build the React app:**
   ```bash
   npm run build
   ```

2. **Start the production server:**
   ```bash
   npm start
   ```

## Usage

1. **Upload Image**: Drag and drop an image or click to browse
2. **Processing**: The app will process the image using the AI model
3. **View Results**: See detected people with bounding boxes and detailed statistics
4. **Analyze**: Review confidence scores, positions, and detection details

## API Endpoints

- `POST /api/upload` - Upload and process an image
- `GET /api/image/:filename` - Serve uploaded images
- `GET /api/output/:filename` - Serve processed images with detections
- `GET /api/health` - Health check endpoint

## Project Structure

```
person-detection-app/
├── client/                 # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/     # React components
│   │   └── App.js         # Main app component
│   └── package.json
├── server.js              # Node.js backend server
├── detect_person.py       # Python detection script
├── clean_detection.py     # Original Python script
├── package.json           # Backend dependencies
└── README.md
```

## Smart Filtering

The application includes advanced filtering to reduce false positives:

- **Confidence Threshold**: Filters detections below 50% confidence
- **Horse Detection**: Identifies and filters horse-like detections
- **Size Validation**: Removes detections that are too small or too large
- **Aspect Ratio**: Ensures detections match human proportions
- **Position Analysis**: Filters detections in suspicious positions

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `person_detector_final.pth` is in the root directory
2. **Python errors**: Check that all Python dependencies are installed
3. **Upload fails**: Verify image format (JPEG/PNG) and size (<10MB)
4. **No detections**: Try images with clear, well-lit people

### Debug Mode

Enable debug logging by setting environment variable:
```bash
DEBUG=1 npm run dev
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Check the troubleshooting section
- Review the console logs for error messages
- Ensure all dependencies are properly installed

# Person Detection App

A full-stack web application for real-time person detection using PyTorch, React, and Node.js.

## 🏗️ Project Structure

```
person-detection-app/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── App.js         # Main App component
│   │   └── index.js       # Entry point
│   ├── public/            # Static assets
│   └── package.json       # Frontend dependencies
├── server/                # Node.js backend
│   ├── server.js          # Express server
│   ├── package.json       # Backend dependencies
│   ├── requirements.txt   # Python dependencies
│   └── render.yaml        # Render deployment config
├── running-scripts/       # Main execution scripts
│   ├── detect_person.py   # Person detection script
│   ├── train_person.py    # Model training script
│   ├── clean_detection.py # Image cleaning utilities
│   ├── dataset.py         # Dataset handling
│   ├── download_model.py  # Model downloader
│   ├── fallback_detect.py # Fallback detection
│   ├── filter_person.py   # Person filtering
│   └── setup_deployment.py # Deployment setup
├── test-scripts/          # Testing and validation
│   ├── test_backend.py    # Backend testing
│   ├── test_model.py      # Model testing
│   ├── test_person.py     # Person detection testing
│   ├── test_custom_images.py # Custom image testing
│   ├── validate_model.py  # Model validation
│   ├── improved_test.py   # Enhanced testing
│   ├── quick_test.py      # Quick tests
│   └── simple_test.py     # Simple tests
├── models/                # AI model files
│   ├── person_detector_final.pth    # Final trained model
│   └── person_detector_epoch_*.pth  # Training checkpoints
├── annotations/           # Dataset annotations
├── uploads/              # User uploaded images
├── outputs/              # Detection results
├── image/                # Sample images
├── package.json          # Root package.json
├── requirements.txt      # Root Python requirements
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18.x or higher
- Python 3.8 or higher
- npm 8.x or higher

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/manishnegi120491/obejctIdentifyApp.git
   cd obejctIdentifyApp
   ```

2. **Install all dependencies**
   ```bash
   npm run install-all
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Development

1. **Start the backend server**
   ```bash
   npm run dev
   ```

2. **Start the frontend (in a new terminal)**
   ```bash
   npm run client
   ```

3. **Open your browser**
   - Frontend: http://localhost:3000
   - Backend: http://localhost:5000

## 🧪 Testing

### Run all tests
```bash
npm run test
```

### Run specific tests
```bash
# Backend tests
cd test-scripts && python test_backend.py

# Model tests
cd test-scripts && python test_model.py

# Person detection tests
cd test-scripts && python test_person.py
```

## 🤖 AI Model Usage

### Train a new model
```bash
npm run train
```

### Run person detection
```bash
npm run detect
```

### Download pre-trained model
```bash
cd running-scripts && python download_model.py
```

## 🚀 Deployment

### Render (Recommended)
```bash
npm run deploy:render
```

### Vercel
```bash
npm run deploy:vercel
```

## 📁 Folder Descriptions

### `client/` - React Frontend
- Modern React application with hooks
- Image upload and display components
- Real-time detection results
- Responsive design

### `server/` - Node.js Backend
- Express.js REST API
- File upload handling with Multer
- Python script execution
- CORS enabled for frontend communication

### `running-scripts/` - Main Execution Scripts
- Core person detection logic
- Model training scripts
- Image processing utilities
- Deployment setup scripts

### `test-scripts/` - Testing and Validation
- Comprehensive test suite
- Model validation scripts
- Backend API testing
- Performance testing

### `models/` - AI Model Files
- Pre-trained PyTorch models
- Training checkpoints
- Model weights and configurations

## 🔧 Configuration

### Environment Variables
- `NODE_ENV`: Environment (development/production)
- `PORT`: Server port (default: 5000)
- `PYTHON_PATH`: Python executable path

### Model Configuration
- Model files are stored in `models/` directory
- Default model: `person_detector_final.pth`
- Fallback detection available if model fails

## 🐛 Troubleshooting

### Common Issues

1. **Python not found**
   - Ensure Python is installed and in PATH
   - Check `python --version` command

2. **Module not found errors**
   - Run `pip install -r requirements.txt`
   - Check Python environment

3. **Port already in use**
   - Change PORT in environment variables
   - Kill existing processes on the port

4. **Model loading errors**
   - Ensure model files are in `models/` directory
   - Check file permissions

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
npm run dev
```

## 📊 Performance

- **Detection Speed**: ~2-3 seconds per image
- **Model Size**: ~165MB
- **Memory Usage**: ~500MB during detection
- **Supported Formats**: JPG, PNG, JPEG

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Manish Negi**
- GitHub: [@manishnegi120491](https://github.com/manishnegi120491)
- Email: [Your Email]

## 🙏 Acknowledgments

- PyTorch team for the excellent ML framework
- React team for the frontend library
- Node.js community for backend tools
- Open source contributors

---

**Note**: This project is for educational and research purposes. Please ensure you have proper permissions when using images for detection.
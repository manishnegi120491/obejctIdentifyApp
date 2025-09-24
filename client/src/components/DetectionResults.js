import React, { useState, useRef, useEffect, useMemo } from 'react';
import axios from 'axios';
import './DetectionResults.css';

const DetectionResults = ({ results, loading, error, onDetectionStart, onDetectionComplete, onError }) => {
  const [selectedThumbnail, setSelectedThumbnail] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState(null);
  const [croppedImages, setCroppedImages] = useState({});
  const fileInputRef = useRef(null);

  const detections = useMemo(() => results?.detections || [], [results?.detections]);
  const totalDetections = results?.totalDetections || 0;
  const imageUrl = results?.imageUrl || null;
  const outputImageUrl = results?.outputImageUrl || null;
  
  console.log('DetectionResults received:', results);
  console.log('outputImageUrl:', outputImageUrl);
  console.log('detections:', detections);
  console.log('totalDetections:', totalDetections);

  const cropImage = (imageUrl, boundingBox, targetSize = { width: 120, height: 80 }) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        const { x1, y1, x2, y2 } = boundingBox;
        const cropWidth = x2 - x1;
        const cropHeight = y2 - y1;
        
        canvas.width = targetSize.width;
        canvas.height = targetSize.height;
        
        const scaleX = targetSize.width / cropWidth;
        const scaleY = targetSize.height / cropHeight;
        const scale = Math.min(scaleX, scaleY);
        
        const scaledWidth = cropWidth * scale;
        const scaledHeight = cropHeight * scale;
        const offsetX = (targetSize.width - scaledWidth) / 2;
        const offsetY = (targetSize.height - scaledHeight) / 2;
        
        ctx.drawImage(
          img,
          x1, y1, cropWidth, cropHeight,
          offsetX, offsetY, scaledWidth, scaledHeight
        );
        
        resolve(canvas.toDataURL());
      };
      img.src = imageUrl;
    });
  };

  const createPersonThumbnails = () => {
    if (!detections || detections.length === 0 || !imageUrl) return [];

    return detections.map((detection) => {
      return {
        id: detection.id,
        confidence: detection.confidence,
        boundingBox: detection.boundingBox,
        thumbnailUrl: imageUrl,
        fullImageUrl: imageUrl,
        originalImageUrl: imageUrl
      };
    });
  };

  const personThumbnails = createPersonThumbnails();
  useEffect(() => {
    if (detections.length > 0 && imageUrl) {
      const createCroppedImages = async () => {
        const newCroppedImages = {};
        
        for (const detection of detections) {
          try {
            const thumbnailUrl = await cropImage(imageUrl, detection.boundingBox, { width: 120, height: 80 });
            const fullImageUrl = await cropImage(imageUrl, detection.boundingBox, { width: 400, height: 300 });
            
            newCroppedImages[detection.id] = {
              thumbnailUrl,
              fullImageUrl
            };
          } catch (error) {
            console.error('Error creating cropped image for person', detection.id, error);
          }
        }
        
        setCroppedImages(newCroppedImages);
      };
      
      createCroppedImages();
    }
  }, [detections, imageUrl]);
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      onError('Please select a valid image file (JPEG, PNG)');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      onError('File size must be less than 10MB');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
    };
    reader.readAsDataURL(file);
    uploadAndProcess(file);
  };

  const uploadAndProcess = async (file) => {
    onDetectionStart();

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000
      });

      console.log('Upload response received:', response.data);
      onDetectionComplete(response.data);
    } catch (error) {
      console.error('Upload error:', error);
      if (error.code === 'ECONNABORTED') {
        onError('Request timeout - the image processing is taking too long');
      } else {
        onError(error.response?.data?.error || 'Failed to process image');
      }
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const resetUpload = () => {
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="grid-container">
      <div className="grid-item upload-container">
        <div className="container-header">
          <h3>Upload Image</h3>
        </div>
        <div 
          className={`upload-area ${dragActive ? 'drag-active' : ''} ${loading ? 'loading' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={handleButtonClick}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            style={{ display: 'none' }}
            disabled={loading}
          />
          
          {preview ? (
            <div className="preview-container">
              <img src={preview} alt="Preview" className="preview-image" />
              <div className="preview-overlay">
                <p>Click to select a different image</p>
                <button 
                  className="reset-button" 
                  onClick={(e) => {
                    e.stopPropagation();
                    resetUpload();
                  }}
                >
                  Reset
                </button>
              </div>
            </div>
          ) : (
            <div className="upload-content">
              <div className="upload-icon">📁</div>
              <p>Click to upload or drag & drop</p>
              <p className="upload-hint">Supports JPG, PNG up to 10MB</p>
            </div>
          )}
        </div>        
        {loading && (
          <div className="loading-indicator">
            <div className="loading-spinner"></div>
            <p>🤖 Processing...</p>
          </div>
        )}

        {error && (
          <div className="error-indicator">
            <p>❌ {error}</p>
          </div>
        )}
      </div>
      <div className="grid-item thumbnails-container">
        <div className="container-header">
          <h3>Detected People</h3>
          <span className="count-badge">{personThumbnails.length}</span>
        </div>
        <div className="thumbnails-grid">
          {personThumbnails.length > 0 ? (
            personThumbnails.map((thumbnail) => (
              <div 
                key={thumbnail.id} 
                className="thumbnail-item"
                onClick={() => setSelectedThumbnail(thumbnail)}
              >
                <div className="thumbnail-image">
                  <img 
                    src={croppedImages[thumbnail.id]?.thumbnailUrl || thumbnail.originalImageUrl} 
                    alt={`Person ${thumbnail.id}`}
                    onError={(e) => {
                      e.target.src = thumbnail.originalImageUrl;
                    }}
                  />
                </div>
                <div className="thumbnail-info">
                  <span className="person-id">Person {thumbnail.id}</span>
                  <span className="confidence">
                    {(thumbnail.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))
          ) : (
            <div className="no-thumbnails">
              <div className="no-thumbnails-icon">👤</div>
              <p>No people detected</p>
              <p className="hint">Upload an image to see detections</p>
            </div>
          )}
        </div>
      </div>
      <div className="grid-item full-image-container">
        <div className="container-header">
          <h3>Selected Person</h3>
          {selectedThumbnail && (
            <span className="selected-info">
              Person {selectedThumbnail.id} - {(selectedThumbnail.confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>
        <div className="full-image-area">
          {selectedThumbnail ? (
            <div className="cropped-person-display">
              <img 
                src={croppedImages[selectedThumbnail.id]?.fullImageUrl || selectedThumbnail.originalImageUrl} 
                alt={`Person ${selectedThumbnail.id}`}
                className="cropped-person-image"
                onError={(e) => {
                  e.target.src = selectedThumbnail.originalImageUrl;
                }}
              />
            </div>
          ) : (
            <div className="no-selection">
              <div className="no-selection-icon">👆</div>
              <p>Click a thumbnail to view person</p>
              <p className="hint">Select a detected person to see their full image</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DetectionResults;

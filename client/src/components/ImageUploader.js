import React, { useState, useRef } from 'react';
import axios from 'axios';
import './ImageUploader.css';

const ImageUploader = ({ onDetectionStart, onDetectionComplete, onError, loading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);

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
    <div className="uploader-container">
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
            <div className="upload-icon">📸</div>
            <h3>Drop your image here</h3>
            <p>or click to browse</p>
            <div className="file-info">
              <p>Supports: JPEG, PNG</p>
              <p>Max size: 10MB</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUploader;

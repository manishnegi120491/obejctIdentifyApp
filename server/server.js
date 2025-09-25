const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs-extra');
const { PythonShell } = require('python-shell');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../client/build')));

// Create necessary directories
const uploadDir = path.join(__dirname, '../uploads');
const outputDir = path.join(__dirname, '../outputs');
const imageDir = path.join(__dirname, '../image');

[uploadDir, outputDir, imageDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files (JPEG, PNG) are allowed!'));
    }
  },
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  }
});

// API Routes
app.post('/api/upload', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    const uploadedFile = req.file;
    console.log('File uploaded:', uploadedFile.filename);

    // Clean up old detection images before processing new one
    try {
      const oldDetectionFiles = await fs.readdir(__dirname);
      const cleanDetectionFiles = oldDetectionFiles.filter(file => file.startsWith('clean_detection_'));
      
      for (const file of cleanDetectionFiles) {
        await fs.remove(path.join(__dirname, file));
        console.log('Deleted old detection file:', file);
      }
    } catch (error) {
      console.log('No old detection files to clean up');
    }

    // Clean up old uploaded images (keep only last 5)
    try {
      const uploadedFiles = await fs.readdir(uploadDir);
      const imageFiles = uploadedFiles.filter(file => 
        file.toLowerCase().match(/\.(jpg|jpeg|png)$/)
      ).sort((a, b) => {
        const statA = fs.statSync(path.join(uploadDir, a));
        const statB = fs.statSync(path.join(uploadDir, b));
        return statB.mtime - statA.mtime; // Sort by modification time, newest first
      });
      
      // Keep only the 5 most recent files
      const filesToDelete = imageFiles.slice(5);
      for (const file of filesToDelete) {
        await fs.remove(path.join(uploadDir, file));
        console.log('Deleted old uploaded file:', file);
      }
    } catch (error) {
      console.log('No old uploaded files to clean up');
    }

    // Delete ALL old images in image directory
    try {
      const imageFiles = await fs.readdir(imageDir);
      const imageFilesList = imageFiles.filter(file => 
        file.toLowerCase().match(/\.(jpg|jpeg|png)$/)
      );
      
      // Delete ALL image files in image directory
      for (const file of imageFilesList) {
        await fs.remove(path.join(imageDir, file));
        console.log('Deleted old image file:', file);
      }
    } catch (error) {
      console.log('No old image files to clean up');
    }

    // Copy uploaded file to image directory for processing (temporary)
    const sourcePath = path.join(uploadDir, uploadedFile.filename);
    const targetPath = path.join(imageDir, uploadedFile.filename);
    
    await fs.copy(sourcePath, targetPath);
    console.log('File copied to image directory for processing');

    // Run Python detection script
    const options = {
      mode: 'text',
      pythonPath: 'python', // Adjust if needed
      scriptPath: __dirname,
      args: [uploadedFile.filename], // Use just filename since file is in image directory
      timeout: 120000, // 120 second timeout (2 minutes)
      pythonOptions: ['-u'] // Unbuffered output
    };

    console.log('Starting Python script execution...');
    console.log('This may take 1-2 minutes for the first run (model loading)...');
    
    // Set a timeout to prevent hanging
    const timeoutId = setTimeout(() => {
      console.error('Detection timeout after 120 seconds');
      if (!res.headersSent) {
        res.status(500).json({ 
          error: 'Detection timeout', 
          details: 'Detection took too long to execute' 
        });
      }
    }, 120000); // 120 second timeout
    
    // Direct execution method (now primary)
    const tryDirectExecution = () => {
      console.log('Executing Python script directly...');
      const { exec } = require('child_process');
      
      // Use python3 for Railway deployment, python for local Windows
      const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
      exec(`cd ../running-scripts && ${pythonCmd} detect_person.py "../uploads/${uploadedFile.filename}"`, (error, stdout, stderr) => {
        clearTimeout(timeoutId); // Clear the timeout
        if (error) {
          console.error('Direct execution error:', error);
          console.error('stdout:', stdout);
          console.error('stderr:', stderr);
          
          // Try fallback detection
          console.log('Trying fallback detection...');
          exec(`cd ../running-scripts && ${pythonCmd} fallback_detect.py "../uploads/${uploadedFile.filename}"`, (fallbackError, fallbackStdout, fallbackStderr) => {
            if (fallbackError) {
              console.error('Fallback detection also failed:', fallbackError);
              return res.status(500).json({ 
                error: 'Both main and fallback detection failed', 
                details: error.message,
                stdout: stdout,
                stderr: stderr,
                fallbackError: fallbackError.message,
                fallbackStdout: fallbackStdout,
                fallbackStderr: fallbackStderr
              });
            }
            
            console.log('Fallback detection completed');
            console.log('Fallback stdout:', fallbackStdout);
            
            // Try to read result from fallback
            const resultFile = `result_${uploadedFile.filename}.json`;
            try {
              if (fs.existsSync(resultFile)) {
                const resultData = fs.readFileSync(resultFile, 'utf8');
                const jsonResult = JSON.parse(resultData);
                fs.unlinkSync(resultFile);
                sendResponse(jsonResult);
              } else {
                throw new Error('Fallback result file not created');
              }
            } catch (fileError) {
              console.error('Fallback file reading error:', fileError);
              res.status(500).json({ 
                error: 'Fallback detection completed but result file not found', 
                details: fileError.message
              });
            }
          });
          return;
        }
        
        console.log('Direct execution completed');
        console.log('stdout:', stdout);
        console.log('stderr:', stderr);
        
        // Try to read result from file
        const resultFile = `result_${uploadedFile.filename}.json`;
        console.log('Looking for result file:', resultFile);
        
        try {
          if (fs.existsSync(resultFile)) {
            console.log('Result file found, reading...');
            const resultData = fs.readFileSync(resultFile, 'utf8');
            const jsonResult = JSON.parse(resultData);
            console.log('Read JSON from file:', jsonResult);
            
            // Clean up the result file
            fs.unlinkSync(resultFile);
            console.log('Cleaned up result file');
            
            sendResponse(jsonResult);
          } else {
            console.error('Result file not found:', resultFile);
            throw new Error('Result file not created by Python script');
          }
        } catch (fileError) {
          console.error('File reading error:', fileError);
          
          // Fallback: try to parse stdout
          try {
            const lines = stdout.trim().split('\n');
            let jsonResult = null;
            
            for (let i = lines.length - 1; i >= 0; i--) {
              try {
                jsonResult = JSON.parse(lines[i]);
                break;
              } catch (e) {
                // Not JSON, continue
              }
            }
            
            if (jsonResult) {
              console.log('Found JSON in stdout:', jsonResult);
              sendResponse(jsonResult);
            } else {
              throw new Error('No valid JSON found in stdout');
            }
          } catch (parseError) {
            console.error('All parsing methods failed:', parseError);
            res.status(500).json({ 
              error: 'Failed to parse detection results', 
              details: parseError.message 
            });
          }
        }
      });
    };
    
    // Helper function to send response
    const sendResponse = (jsonResult) => {
      const response = {
        success: true,
        filename: uploadedFile.filename,
        detections: jsonResult.detections || [],
        totalDetections: jsonResult.totalDetections || 0,
        imageSize: jsonResult.imageSize || { width: 0, height: 0 },
        imageUrl: `/api/image/${uploadedFile.filename}`,
        outputImageUrl: jsonResult.outputFilename ? `/api/output/${jsonResult.outputFilename}` : null
      };
      
      console.log('Sending response:', response);
      
      if (!res.headersSent) {
        res.json(response);
      } else {
        console.error('Response already sent, cannot send again');
      }
    };
    
    // Use direct execution as primary method (more reliable)
    console.log('Using direct execution method...');
    tryDirectExecution();

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'Upload failed', details: error.message });
  }
});

// Serve uploaded images
app.get('/api/image/:filename', (req, res) => {
  const filename = req.params.filename;
  const imagePath = path.join(uploadDir, filename);
  
  if (fs.existsSync(imagePath)) {
    res.sendFile(imagePath);
  } else {
    res.status(404).json({ error: 'Image not found' });
  }
});

// Serve output images with detections
app.get('/api/output/:filename', (req, res) => {
  const filename = req.params.filename;
  const outputPath = path.join(__dirname, filename);
  
  if (fs.existsSync(outputPath)) {
    res.sendFile(outputPath);
  } else {
    res.status(404).json({ error: 'Output image not found' });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Person Detection API is running' });
});

// Environment check endpoint
app.get('/api/env-check', (req, res) => {
  const { exec } = require('child_process');
  
  exec('which python3 && python3 --version && which node && node --version', (error, stdout, stderr) => {
    if (error) {
      return res.status(500).json({
        error: 'Environment check failed',
        details: error.message,
        stderr: stderr
      });
    }
    
    res.json({
      success: true,
      message: 'Environment check completed',
      output: stdout,
      environment: {
        node: process.version,
        platform: process.platform,
        arch: process.arch
      }
    });
  });
});

// Backend test endpoint
app.get('/api/test-backend', (req, res) => {
  console.log('Testing backend components...');
  
  const { exec } = require('child_process');
  
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
  exec(`cd ../test-scripts && ${pythonCmd} test_backend.py`, (error, stdout, stderr) => {
    if (error) {
      console.error('Backend test error:', error);
      return res.status(500).json({ 
        error: 'Backend test failed', 
        details: error.message,
        stderr: stderr
      });
    }
    
    console.log('Backend test output:', stdout);
    res.json({
      success: true,
      message: 'Backend test completed',
      output: stdout
    });
  });
});

// Simple Python test endpoint
app.get('/api/simple-test', (req, res) => {
  console.log('Running simple Python test...');
  
  const { exec } = require('child_process');
  
  // First check if python exists (Windows uses 'python', Linux uses 'python3')
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
  const whichCmd = process.platform === 'win32' ? 'where' : 'which';
  exec(`${whichCmd} ${pythonCmd}`, (whichError, whichStdout, whichStderr) => {
    if (whichError) {
      console.error('Python not found:', whichError);
      return res.status(500).json({ 
        error: 'Python not found in PATH', 
        details: whichError.message,
        stderr: whichStderr
      });
    }
    
    console.log('Python found at:', whichStdout.trim());
    
    // Now try to run the Python script
    exec(`cd ../test-scripts && ${pythonCmd} simple_test.py`, (error, stdout, stderr) => {
      if (error) {
        console.error('Simple test error:', error);
        return res.status(500).json({ 
          error: 'Simple test failed', 
          details: error.message,
          stdout: stdout,
          stderr: stderr
        });
      }
      
      console.log('Simple test output:', stdout);
      res.json({
        success: true,
        message: 'Simple test completed',
        output: stdout
      });
    });
  });
});

// Test endpoint to debug Python script
app.get('/api/test-detection', (req, res) => {
  console.log('Testing Python script...');
  
  const options = {
    mode: 'text',
    pythonPath: 'python',
    scriptPath: __dirname,
    args: ['1758638332786-131481150.jpg'],
    timeout: 120000
  };

  PythonShell.run('../running-scripts/detect_person.py', options, (err, results) => {
    if (err) {
      console.error('Python test error:', err);
      return res.status(500).json({ error: 'Python test failed', details: err.message });
    }

    console.log('Python test results:', results);
    
    try {
      // Find the JSON result
      let jsonResult = null;
      for (let i = results.length - 1; i >= 0; i--) {
        try {
          jsonResult = JSON.parse(results[i]);
          break;
        } catch (parseErr) {
          // Not JSON, continue
        }
      }
      
      if (jsonResult) {
        res.json({
          success: true,
          message: 'Python script working correctly',
          data: jsonResult,
          rawResults: results
        });
      } else {
        res.status(500).json({
          error: 'No valid JSON found',
          rawResults: results
        });
      }
    } catch (error) {
      res.status(500).json({
        error: 'JSON parsing failed',
        details: error.message,
        rawResults: results
      });
    }
  });
});

// Serve React app
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'client/build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📁 Upload directory: ${uploadDir}`);
  console.log(`📁 Output directory: ${outputDir}`);
  console.log(`📁 Image directory: ${imageDir}`);
});

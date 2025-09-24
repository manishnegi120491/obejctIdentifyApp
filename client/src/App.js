import React, { useState } from 'react';
import './App.css';
import DetectionResults from './components/DetectionResults';
import Header from './components/Header';

function App() {
  const [detectionResults, setDetectionResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleDetectionComplete = (results) => {
    setDetectionResults(results);
    setLoading(false);
    setError(null);
  };

  const handleDetectionStart = () => {
    setLoading(true);
    setError(null);
    setDetectionResults(null);
  };

  const handleError = (errorMessage) => {
    setError(errorMessage);
    setLoading(false);
    setDetectionResults(null);
  };

  return (
    <div className="App">
      <Header />
      <div className="container">
        <DetectionResults 
          results={detectionResults} 
          loading={loading}
          error={error}
          onDetectionStart={handleDetectionStart}
          onDetectionComplete={handleDetectionComplete}
          onError={handleError}
        />
      </div>
    </div>
  );
}

export default App;

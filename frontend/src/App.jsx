import React, { useState, useEffect } from 'react';
import './styles/tokens.css';

import { healthCheck, predictImage } from './api/client';
import ConnectionStatus from './components/ConnectionStatus';
import ImageUploader from './components/ImageUploader';
import PredictionResult from './components/PredictionResult';

function App() {
  const [systemStatus, setSystemStatus] = useState('loading');
  const [device, setDevice] = useState('none');
  const [category, setCategory] = useState('none');
  const [threshold, setThreshold] = useState(0.0);
  const [errorMessage, setErrorMessage] = useState('');

  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [anomalyMap, setAnomalyMap] = useState(null);
  const [predictionError, setPredictionError] = useState('');

  // Run health check exactly once on application mount
  useEffect(() => {
    async function checkHealth() {
      const data = await healthCheck();
      setSystemStatus(data.status);
      setDevice(data.device);
      setCategory(data.category);
      setThreshold(data.threshold);
      if (data.status === 'error') {
        setErrorMessage(data.message);
      }
    }
    checkHealth();
  }, []);

  const handleFileSelected = async (file) => {
    setIsPredicting(true);
    setPredictionError('');
    setPredictionResult(null);
    setAnomalyMap(null);

    try {
      const result = await predictImage(file);
      setPredictionResult({
        is_anomaly: result.is_anomaly,
        anomaly_score: result.anomaly_score,
        threshold: result.threshold,
        inference_time: result.inference_time
      });
      // Store and preserve raw anomaly map list in state
      setAnomalyMap(result.anomaly_map);
    } catch (err) {
      setPredictionError(err.message);
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div style={{
      backgroundColor: 'var(--color-bg)',
      color: 'var(--color-text-primary)',
      fontFamily: 'var(--font-body)',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '40px 20px',
      boxSizing: 'border-box'
    }}>
      <h1 style={{ 
        fontFamily: 'var(--font-heading)', 
        color: 'var(--color-accent)',
        fontSize: '36px',
        margin: '0 0 10px 0',
        letterSpacing: '-1px'
      }}>
        DefectLens AI
      </h1>
      
      <p style={{ 
        color: 'var(--color-text-secondary)',
        margin: '0 0 30px 0',
        fontSize: '16px',
        textAlign: 'center'
      }}>
        Real-Time Industrial Visual Anomaly Detection Dashboard
      </p>

      <ConnectionStatus 
        status={systemStatus}
        device={device}
        category={category}
        threshold={threshold}
        message={errorMessage}
      />

      <ImageUploader 
        status={systemStatus}
        isPredicting={isPredicting}
        onFileSelected={handleFileSelected}
      />

      {predictionError && (
        <div style={{
          backgroundColor: 'var(--color-surface)',
          border: '1px solid var(--color-status-error)',
          borderRadius: 'var(--radius-md)',
          padding: '15px 20px',
          width: '100%',
          maxWidth: '600px',
          color: 'var(--color-status-error)',
          fontSize: '14px',
          marginBottom: '20px',
          lineHeight: '1.4',
          boxSizing: 'border-box'
        }}>
          <strong>Prediction Failed:</strong> {predictionError}
        </div>
      )}

      <PredictionResult result={predictionResult} />
    </div>
  );
}

export default App;

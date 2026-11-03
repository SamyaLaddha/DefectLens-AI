import React, { useState, useEffect, useRef } from 'react';
import './styles/tokens.css';

import { healthCheck, predictImage } from './api/client';
import { PredictionCoordinator } from './utils/predictionCoordinator';
import ConnectionStatus from './components/ConnectionStatus';
import CategorySelector from './components/CategorySelector';
import ImageUploader from './components/ImageUploader';
import PredictionResult from './components/PredictionResult';

function App() {
  const [systemStatus, setSystemStatus] = useState('loading');
  const [device, setDevice] = useState('none');
  const [category, setCategory] = useState('none');
  const [threshold, setThreshold] = useState(0.0);
  const [loadedCategories, setLoadedCategories] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');

  const [selectedCategory, setSelectedCategory] = useState('');
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [anomalyMap, setAnomalyMap] = useState(null);
  const [predictionError, setPredictionError] = useState('');

  // Request identity coordinator to guard against asynchronous race conditions
  const coordinatorRef = useRef(null);
  if (!coordinatorRef.current) {
    coordinatorRef.current = new PredictionCoordinator({
      predictFn: predictImage,
      onStart: ({ previewUrl: newUrl }) => {
        setIsPredicting(true);
        setPredictionError('');
        setPredictionResult(null);
        setAnomalyMap(null);
        setPreviewUrl(newUrl);
      },
      onSuccess: ({ result }) => {
        setPredictionResult({
          is_anomaly: result.is_anomaly,
          anomaly_score: result.anomaly_score,
          threshold: result.threshold,
          inference_time: result.inference_time
        });
        setAnomalyMap(result.anomaly_map);
      },
      onError: ({ error }) => {
        setPredictionError(error.message);
        setPredictionResult(null);
        setAnomalyMap(null);
      },
      onSettled: () => {
        setIsPredicting(false);
      }
    });
  }

  // Run health check exactly once on application mount
  useEffect(() => {
    async function checkHealth() {
      const data = await healthCheck();
      setSystemStatus(data.status);
      setDevice(data.device);
      setCategory(data.category);
      setThreshold(data.threshold);
      setLoadedCategories(data.loaded_categories || []);
      if (data.status === 'error') {
        setErrorMessage(data.message);
      }
    }
    checkHealth();
  }, []);

  // Invalidate any in-flight requests and clean up object URLs on component unmount
  useEffect(() => {
    return () => {
      coordinatorRef.current?.cleanup();
    };
}, []);

  const handleFileSelected = (file) => {
    coordinatorRef.current.requestPrediction(file, selectedCategory);
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
        loadedCategories={loadedCategories}
        message={errorMessage}
      />

      <CategorySelector
        selectedCategory={selectedCategory}
        onCategoryChanged={setSelectedCategory}
        systemStatus={systemStatus}
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

      <PredictionResult
        result={predictionResult}
        previewUrl={previewUrl}
        anomalyMap={anomalyMap}
      />
    </div>
  );
}

export default App;

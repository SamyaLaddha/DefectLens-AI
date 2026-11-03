import React, { useState, useEffect, useRef, useCallback } from 'react';

import { healthCheck, predictImage, predictBatch } from '../api/client';
import { PredictionCoordinator } from '../utils/predictionCoordinator';
import ConnectionStatus from '../components/ConnectionStatus';
import CategorySelector from '../components/CategorySelector';
import ImageUploader from '../components/ImageUploader';
import PredictionResult from '../components/PredictionResult';
import BatchImageUploader from '../components/BatchImageUploader';
import BatchSummary from '../components/BatchSummary';
import BatchInspectionView from '../components/BatchInspectionView';

function TryPage() {
  // ── System state ────────────────────────────────────────────────────────────
  const [systemStatus, setSystemStatus] = useState('loading');
  const [device, setDevice] = useState('none');
  const [category, setCategory] = useState('none');
  const [threshold, setThreshold] = useState(0.0);
  const [loadedCategories, setLoadedCategories] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');

  // ── Mode toggle ──────────────────────────────────────────────────────────────
  const [inspectionMode, setInspectionMode] = useState('single'); // 'single' | 'batch'

  // ── Shared ───────────────────────────────────────────────────────────────────
  const [selectedCategory, setSelectedCategory] = useState('');

  // ── Single-image state ───────────────────────────────────────────────────────
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [anomalyMap, setAnomalyMap] = useState(null);
  const [predictionError, setPredictionError] = useState('');

  // ── Batch state ──────────────────────────────────────────────────────────────
  const [isBatching, setIsBatching] = useState(false);
  const [batchResult, setBatchResult] = useState(null);   // server response envelope
  const [batchError, setBatchError] = useState('');
  // previewUrlMap: Map<image_id, blobUrl> — kept in a ref to avoid stale closures during cleanup
  const previewUrlMapRef = useRef(new Map());
  // Batch request identity guard (same pattern as single-image coordinator)
  const batchRequestIdRef = useRef(0);

  // ── Single-image coordinator ─────────────────────────────────────────────────
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

  // ── Health check on mount ────────────────────────────────────────────────────
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

  // ── Cleanup on unmount ───────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      coordinatorRef.current?.cleanup();
      // Revoke all batch preview URLs
      previewUrlMapRef.current.forEach(url => URL.revokeObjectURL(url));
      previewUrlMapRef.current.clear();
    };
  }, []);

  // ── Single-image handler ─────────────────────────────────────────────────────
  const handleFileSelected = (file) => {
    coordinatorRef.current.requestPrediction(file, selectedCategory);
  };

  // ── Batch submission handler ─────────────────────────────────────────────────
  const handleBatchSubmit = useCallback(async (filesWithIds) => {
    // Increment request token — any earlier in-flight batch is now stale
    const requestId = ++batchRequestIdRef.current;

    // Revoke all previous batch preview URLs
    previewUrlMapRef.current.forEach(url => URL.revokeObjectURL(url));
    previewUrlMapRef.current.clear();

    // Build fresh preview URL map: image_id → blobUrl
    const newMap = new Map();
    for (const { file, id } of filesWithIds) {
      newMap.set(id, URL.createObjectURL(file));
    }
    previewUrlMapRef.current = newMap;

    setIsBatching(true);
    setBatchResult(null);
    setBatchError('');

    try {
      const result = await predictBatch(filesWithIds, selectedCategory);

      // Guard: discard if a newer batch was submitted while this one was in flight
      if (requestId !== batchRequestIdRef.current) return;

      setBatchResult(result);
    } catch (err) {
      if (requestId !== batchRequestIdRef.current) return;
      setBatchError(err.message);
    } finally {
      if (requestId === batchRequestIdRef.current) {
        setIsBatching(false);
      }
    }
  }, [selectedCategory]);

  // ── Mode switch: clear the other mode's state ────────────────────────────────
  const switchMode = (mode) => {
    if (mode === inspectionMode) return;
    setInspectionMode(mode);
    if (mode === 'single') {
      // Clear batch state
      setBatchResult(null);
      setBatchError('');
      batchRequestIdRef.current++;   // invalidate any in-flight batch
    } else {
      // Clear single-image state
      coordinatorRef.current?.cleanup();
      setPredictionResult(null);
      setAnomalyMap(null);
      setPredictionError('');
    }
  };

  // ── Render ───────────────────────────────────────────────────────────────────
  const modeButtonStyle = (mode) => ({
    padding: '8px 20px',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    fontFamily: 'var(--font-heading)',
    fontSize: '13px',
    fontWeight: '600',
    cursor: 'pointer',
    backgroundColor: inspectionMode === mode ? 'var(--color-accent)' : 'var(--color-surface-alt)',
    color: inspectionMode === mode ? 'var(--color-text-on-accent)' : 'var(--color-text-secondary)',
    transition: 'var(--transition-base)'
  });

  // Convert previewUrlMap ref to a plain Map for render — ref is stable so no re-render issue
  const previewUrlMapForRender = previewUrlMapRef.current;

  return (
    <div style={{
      backgroundColor: 'var(--color-bg)',
      color: 'var(--color-text-primary)',
      fontFamily: 'var(--font-body)',
      minHeight: 'calc(100vh - 66px)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '48px 20px 64px',
      boxSizing: 'border-box'
    }}>
      <header className="bench-head">
        <span className="kicker">Inspection bench · no account needed</span>
        <h1 className="display">Inspect a <em>part</em>.</h1>
        <p>
          Pick a category, drop in a photo. You get a verdict, an anomaly score
          and a heatmap of where the model sees something off.
        </p>
      </header>

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

      {/* Mode toggle */}
      <div style={{
        display: 'flex',
        gap: '8px',
        marginBottom: '20px',
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: 'var(--radius-md)',
        padding: '6px',
        width: '100%',
        maxWidth: '720px',
        boxSizing: 'border-box'
      }}>
        <button style={modeButtonStyle('single')} onClick={() => switchMode('single')}>
          Single Image
        </button>
        <button style={modeButtonStyle('batch')} onClick={() => switchMode('batch')}>
          Batch Inspection (4-12 images)
        </button>
      </div>

      {/* ── Single-image mode ── */}
      {inspectionMode === 'single' && (
        <>
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
        </>
      )}

      {/* ── Batch mode ── */}
      {inspectionMode === 'batch' && (
        <>
          <BatchImageUploader
            status={systemStatus}
            isBatching={isBatching}
            onBatchSubmit={handleBatchSubmit}
          />

          {/* Batch loading indicator */}
          {isBatching && (
            <div style={{
              width: '100%',
              maxWidth: '720px',
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-md)',
              padding: '16px 20px',
              marginBottom: '20px',
              textAlign: 'center',
              fontSize: '14px',
              color: 'var(--color-text-secondary)'
            }}>
              Processing batch - running STFPM feature discrepancy matching on each image...
            </div>
          )}

          {/* Batch error */}
          {batchError && !isBatching && (
            <div style={{
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--color-status-error)',
              borderRadius: 'var(--radius-md)',
              padding: '15px 20px',
              width: '100%',
              maxWidth: '720px',
              color: 'var(--color-status-error)',
              fontSize: '14px',
              marginBottom: '20px',
              boxSizing: 'border-box'
            }}>
              <strong>Batch Inspection Failed:</strong> {batchError}
            </div>
          )}

          {/* Batch results */}
          {batchResult && !isBatching && (
            <>
              <BatchSummary
                category={batchResult.category}
                threshold={batchResult.threshold}
                batchStatus={batchResult.batch_status}
                totalSubmitted={batchResult.total_submitted}
                totalProcessed={batchResult.total_processed}
                totalNormal={batchResult.total_normal}
                totalAnomalous={batchResult.total_anomalous}
                totalFailed={batchResult.total_failed}
              />
              <BatchInspectionView
                results={batchResult.results}
                previewUrlMap={previewUrlMapForRender}
              />
            </>
          )}
        </>
      )}
    </div>
  );
}

export default TryPage;

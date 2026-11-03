import React, { useState, useEffect, useRef } from "react";

/**
 * Standard JET colormap piecewise linear formula.
 * Maps normalized discrepancy v in [0, 1] to RGB values.
 */
function jetColor(v) {
  v = Math.max(0, Math.min(1, v));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 1)));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function HeatmapViewer({
  imageSrc,
  anomalyMap,
  threshold,
  anomalyScore,
  isAnomaly
}) {
  const canvasRef = useRef(null);
  const [viewMode, setViewMode] = useState("overlay"); // "overlay" | "heatmap" | "original"
  const [opacity, setOpacity] = useState(0.5);
  const [imageError, setImageError] = useState(false);
  const [minDiscrepancy, setMinDiscrepancy] = useState(0);
  const [maxDiscrepancy, setMaxDiscrepancy] = useState(0);

  useEffect(() => {
    setImageError(false);
    if (!imageSrc || !anomalyMap || !Array.isArray(anomalyMap) || anomalyMap.length === 0) {
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    let isSubscribed = true;
    const img = new Image();
    img.crossOrigin = "anonymous";

    img.onload = () => {
      if (!isSubscribed) return;

      const imgWidth = img.naturalWidth || 256;
      const imgHeight = img.naturalHeight || 256;
      canvas.width = imgWidth;
      canvas.height = imgHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // 1. Calculate map min and max for relative discrepancy normalization
      let minVal = Infinity;
      let maxVal = -Infinity;
      const mapHeight = anomalyMap.length;
      const mapWidth = Array.isArray(anomalyMap[0]) ? anomalyMap[0].length : 0;

      if (mapWidth === 0) return;

      for (let r = 0; r < mapHeight; r++) {
        const row = anomalyMap[r];
        if (!Array.isArray(row)) continue;
        for (let c = 0; c < row.length; c++) {
          const val = row[c];
          if (typeof val === "number" && !isNaN(val)) {
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
          }
        }
      }

      if (!isFinite(minVal)) minVal = 0;
      if (!isFinite(maxVal)) maxVal = 0;
      setMinDiscrepancy(minVal);
      setMaxDiscrepancy(maxVal);

      // 2. Generate 2D in-memory heatmap buffer
      const bufferCanvas = document.createElement("canvas");
      bufferCanvas.width = mapWidth;
      bufferCanvas.height = mapHeight;
      const bufferCtx = bufferCanvas.getContext("2d");

      if (bufferCtx) {
        const imgData = bufferCtx.createImageData(mapWidth, mapHeight);
        const hasRange = maxVal > minVal;
        let p = 0;

        for (let r = 0; r < mapHeight; r++) {
          const row = anomalyMap[r] || [];
          for (let c = 0; c < mapWidth; c++) {
            const val = row[c] ?? 0;
            const norm = hasRange ? (val - minVal) / (maxVal - minVal) : 0;
            const [red, green, blue] = jetColor(norm);
            imgData.data[p] = red;
            imgData.data[p + 1] = green;
            imgData.data[p + 2] = blue;
            imgData.data[p + 3] = 255;
            p += 4;
          }
        }
        bufferCtx.putImageData(imgData, 0, 0);
      }

      // 3. Render view mode onto the visible canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (viewMode === "original") {
        ctx.globalAlpha = 1.0;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      } else if (viewMode === "heatmap") {
        ctx.globalAlpha = 1.0;
        ctx.drawImage(bufferCanvas, 0, 0, canvas.width, canvas.height);
      } else {
        // "overlay" mode: draw raw image then blend discrepancy heatmap
        ctx.globalAlpha = 1.0;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = opacity;
        ctx.drawImage(bufferCanvas, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;
      }
    };

    img.onerror = () => {
      if (isSubscribed) {
        setImageError(true);
      }
    };

    img.src = imageSrc;

    return () => {
      isSubscribed = false;
    };
  }, [imageSrc, anomalyMap, viewMode, opacity]);

  if (!imageSrc || !anomalyMap) {
    return null;
  }

  return (
    <div style={{
      marginTop: "25px",
      borderTop: "1px solid var(--color-border)",
      paddingTop: "20px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      width: "100%"
    }}>
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        width: "100%",
        marginBottom: "15px",
        flexWrap: "wrap",
        gap: "10px"
      }}>
        <h4 style={{
          margin: 0,
          fontFamily: "var(--font-heading)",
          fontSize: "16px",
          color: "var(--color-text-primary)"
        }}>
          Defect Localization Heatmap
        </h4>

        {/* View Mode Switcher */}
        <div style={{
          display: "flex",
          backgroundColor: "var(--color-surface-alt)",
          borderRadius: "var(--radius-sm)",
          padding: "3px",
          border: "1px solid var(--color-border)"
        }}>
          {[
            { id: "overlay", label: "Overlay" },
            { id: "heatmap", label: "Heatmap" },
            { id: "original", label: "Original" }
          ].map((mode) => (
            <button
              key={mode.id}
              type="button"
              onClick={() => setViewMode(mode.id)}
              style={{
                backgroundColor: viewMode === mode.id ? "var(--color-accent)" : "transparent",
                color: viewMode === mode.id ? "#000" : "var(--color-text-secondary)",
                border: "none",
                borderRadius: "var(--radius-sm)",
                padding: "6px 12px",
                fontSize: "12px",
                fontWeight: "600",
                cursor: "pointer",
                fontFamily: "var(--font-heading)",
                transition: "var(--transition-fast)"
              }}
            >
              {mode.label}
            </button>
          ))}
        </div>
      </div>

      {/* Opacity Control (Active in Overlay Mode) */}
      {viewMode === "overlay" && (
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          width: "100%",
          marginBottom: "15px",
          fontSize: "13px",
          color: "var(--color-text-secondary)"
        }}>
          <label htmlFor="heatmap-opacity" style={{ minWidth: "125px" }}>
            Heatmap Opacity: <strong>{Math.round(opacity * 100)}%</strong>
          </label>
          <input
            id="heatmap-opacity"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            style={{
              flex: 1,
              accentColor: "var(--color-accent)",
              cursor: "pointer"
            }}
          />
        </div>
      )}

      {/* Canvas Display with Error Fallback */}
      {imageError ? (
        <div style={{
          width: "100%",
          padding: "30px 20px",
          textAlign: "center",
          backgroundColor: "rgba(239, 68, 68, 0.05)",
          border: "1px solid var(--color-status-error)",
          borderRadius: "var(--radius-md)",
          color: "var(--color-status-error)",
          fontSize: "14px"
        }}>
          Failed to load image for heatmap visualization.
        </div>
      ) : (
        <div style={{
          width: "100%",
          display: "flex",
          justifyContent: "center",
          backgroundColor: "var(--color-surface-alt)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-md)",
          padding: "10px",
          boxSizing: "border-box"
        }}>
          <canvas
            ref={canvasRef}
            style={{
              maxWidth: "100%",
              height: "auto",
              display: "block",
              borderRadius: "var(--radius-sm)"
            }}
          />
        </div>
      )}

      {/* Relative Intensity Bar Legend */}
      <div style={{
        width: "100%",
        marginTop: "15px",
        display: "flex",
        flexDirection: "column",
        gap: "6px"
      }}>
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "12px",
          color: "var(--color-text-secondary)"
        }}>
          <span>Low Discrepancy ({minDiscrepancy.toFixed(4)})</span>
          <span style={{ fontWeight: "600", color: "var(--color-text-primary)" }}>
            Relative Discrepancy Gradient
          </span>
          <span>Peak Discrepancy ({maxDiscrepancy.toFixed(4)})</span>
        </div>
        <div style={{
          width: "100%",
          height: "8px",
          borderRadius: "var(--radius-sm)",
          background: "linear-gradient(to right, #000080, #0080ff, #00ff80, #ffff00, #ff8000, #800000)"
        }} />
      </div>
    </div>
  );
}

export default HeatmapViewer;


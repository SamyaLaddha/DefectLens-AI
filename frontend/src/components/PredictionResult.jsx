import React from "react";

function PredictionResult({ result }) {
  if (!result) return null;

  const { is_anomaly, anomaly_score, threshold, inference_time } = result;
  
  const isAnomaly = is_anomaly === 1;
  const labelText = isAnomaly ? "ANOMALOUS / DEFECT" : "NORMAL / GOOD";
  const labelColor = isAnomaly ? "var(--color-status-anomaly)" : "var(--color-status-success)";

  return (
    <div style={{
      backgroundColor: "var(--color-surface)",
      border: "1px solid var(--color-border)",
      borderRadius: "var(--radius-md)",
      padding: "20px",
      width: "100%",
      maxWidth: "600px",
      boxShadow: "var(--shadow-md)"
    }}>
      <h3 style={{ margin: "0 0 15px 0", fontFamily: "var(--font-heading)" }}>Inspection Result</h3>
      
      <div style={{
        backgroundColor: labelColor,
        color: "#000",
        fontWeight: "bold",
        fontSize: "18px",
        padding: "10px 15px",
        borderRadius: "var(--radius-sm)",
        textAlign: "center",
        marginBottom: "20px",
        fontFamily: "var(--font-heading)"
      }}>
        {labelText}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "15px", fontSize: "14px", color: "var(--color-text-secondary)" }}>
        <div>
          <strong>Anomaly Score:</strong>
          <div style={{ fontSize: "20px", color: "var(--color-text-primary)", marginTop: "5px", fontFamily: "var(--font-mono)" }}>
            {anomaly_score.toFixed(6)}
          </div>
        </div>
        <div>
          <strong>Threshold:</strong>
          <div style={{ fontSize: "20px", color: "var(--color-text-primary)", marginTop: "5px", fontFamily: "var(--font-mono)" }}>
            {threshold.toFixed(6)}
          </div>
        </div>
        <div>
          <strong>Inference Latency:</strong>
          <div style={{ fontSize: "16px", color: "var(--color-text-primary)", marginTop: "5px" }}>
            {(inference_time * 1000).toFixed(2)} ms
          </div>
        </div>
        <div>
          <strong>Status:</strong>
          <div style={{ fontSize: "16px", color: labelColor, marginTop: "5px", fontWeight: "600" }}>
            {isAnomaly ? "Anomaly Detected" : "Clear"}
          </div>
        </div>
      </div>
    </div>
  );
}

export default PredictionResult;

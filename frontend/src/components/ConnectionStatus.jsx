import React from "react";

function ConnectionStatus({ status, device, category, threshold, message }) {
  let statusColor = "var(--color-text-muted)";
  let statusText = "Connecting...";

  if (status === "healthy") {
    statusColor = "var(--color-status-success)";
    statusText = "Online - Model Ready";
  } else if (status === "degraded") {
    statusColor = "var(--color-status-anomaly)";
    statusText = "Online - Model Missing";
  } else if (status === "error") {
    statusColor = "var(--color-status-error)";
    statusText = "Offline - Server Unreachable";
  }

  return (
    <div style={{
      backgroundColor: "var(--color-surface)",
      border: "1px solid var(--color-border)",
      borderRadius: "var(--radius-md)",
      padding: "20px",
      width: "100%",
      maxWidth: "600px",
      marginBottom: "20px",
      boxShadow: "var(--shadow-md)"
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "15px" }}>
        <h3 style={{ margin: 0, fontFamily: "var(--font-heading)" }}>System Status</h3>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{
            width: "10px",
            height: "10px",
            borderRadius: "50%",
            backgroundColor: statusColor,
            display: "inline-block"
          }} />
          <span style={{ fontSize: "14px", fontWeight: "600", color: statusColor }}>{statusText}</span>
        </div>
      </div>

      {status === "healthy" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px", fontSize: "14px", color: "var(--color-text-secondary)" }}>
          <div><strong>Category:</strong> {category}</div>
          <div><strong>Threshold:</strong> {threshold.toFixed(4)}</div>
          <div><strong>Device:</strong> {device.toUpperCase()}</div>
        </div>
      )}

      {status === "degraded" && (
        <div style={{ fontSize: "14px", color: "var(--color-status-anomaly)", lineHeight: "1.4" }}>
          Warning: Anomaly detection model weights or threshold settings are missing on the backend. Inference uploads are active to inspect raw 503 error payloads.
        </div>
      )}

      {status === "error" && (
        <div style={{ fontSize: "14px", color: "var(--color-status-error)", lineHeight: "1.4" }}>
          Error: {message || "Failed to establish connection to the backend server."}
        </div>
      )}
    </div>
  );
}

export default ConnectionStatus;

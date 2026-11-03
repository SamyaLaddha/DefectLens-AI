import React from "react";

/**
 * Batch verdict configuration.
 * Maps batch_status strings to display labels and colors.
 */
const VERDICT_CONFIG = {
  ALL_CLEAR: {
    label: "ALL CLEAR",
    color: "var(--color-status-success)",
    icon: "-",
    description: "All views processed successfully - no defects detected."
  },
  DEFECT_DETECTED: {
    label: "DEFECT DETECTED",
    color: "var(--color-status-anomaly)",
    icon: "-",
    description: "One or more views exceed the anomaly threshold - defect confirmed."
  },
  INCOMPLETE: {
    label: "INCOMPLETE - UNVERIFIED",
    color: "var(--color-status-error)",
    icon: "-",
    description: "One or more views could not be processed. Batch quality cannot be confirmed."
  },
  INCOMPLETE_DEFECT_DETECTED: {
    label: "INCOMPLETE - DEFECT DETECTED",
    color: "var(--color-status-anomaly)",
    icon: "-",
    description: "Defect confirmed on at least one view. Additional views failed to process."
  }
};

/**
 * BatchSummary - displays the aggregate batch inspection result.
 *
 * Props:
 *   category       - string, e.g. "screw"
 *   threshold      - number, category-specific threshold loaded from config
 *   batchStatus    - one of the VERDICT_CONFIG keys
 *   totalSubmitted - integer
 *   totalProcessed - integer
 *   totalNormal    - integer
 *   totalAnomalous - integer
 *   totalFailed    - integer
 */
function BatchSummary({
  category,
  threshold,
  batchStatus,
  totalSubmitted,
  totalProcessed,
  totalNormal,
  totalAnomalous,
  totalFailed
}) {
  const verdict = VERDICT_CONFIG[batchStatus] || {
    label: batchStatus,
    color: "var(--color-text-secondary)",
    icon: "-",
    description: ""
  };

  return (
    <div style={{
      backgroundColor: "var(--color-surface)",
      border: `1px solid ${verdict.color}`,
      borderRadius: "var(--radius-md)",
      padding: "20px",
      width: "100%",
      maxWidth: "720px",
      marginBottom: "20px",
      boxShadow: "var(--shadow-md)"
    }}>
      {/* Overall verdict banner */}
       <div style={{
         backgroundColor: verdict.color,
         color: "var(--color-text-on-accent)",
         fontWeight: "bold",
         fontSize: "16px",
        padding: "10px 16px",
        borderRadius: "var(--radius-sm)",
        textAlign: "center",
        marginBottom: "16px",
        fontFamily: "var(--font-heading)",
        letterSpacing: "0.03em"
      }}>
        {verdict.icon} {verdict.label}
      </div>

      <p style={{ margin: "0 0 16px", fontSize: "13px", color: "var(--color-text-secondary)", textAlign: "center" }}>
        {verdict.description}
      </p>

      {/* Stats grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap: "12px",
        marginBottom: "12px"
      }}>
        {[
          { label: "Submitted", value: totalSubmitted, color: "var(--color-text-primary)" },
          { label: "Processed", value: totalProcessed, color: "var(--color-text-primary)" },
          { label: "Failed", value: totalFailed, color: totalFailed > 0 ? "var(--color-status-error)" : "var(--color-text-muted)" },
          { label: "Normal", value: totalNormal, color: "var(--color-status-success)" },
          { label: "Anomalous", value: totalAnomalous, color: totalAnomalous > 0 ? "var(--color-status-anomaly)" : "var(--color-text-muted)" },
          { label: "Threshold", value: typeof threshold === "number" ? threshold.toFixed(6) : "—", color: "var(--color-text-secondary)" }
        ].map(({ label, value, color }) => (
          <div key={label} style={{
            backgroundColor: "var(--color-surface-alt)",
            borderRadius: "var(--radius-sm)",
            padding: "10px 12px",
            textAlign: "center"
          }}>
            <div style={{ fontSize: "11px", color: "var(--color-text-muted)", marginBottom: "4px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              {label}
            </div>
            <div style={{ fontSize: "20px", fontFamily: "var(--font-mono)", color, fontWeight: "700" }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      <p style={{ margin: 0, fontSize: "12px", color: "var(--color-text-muted)", textAlign: "right" }}>
        Category: <strong style={{ color: "var(--color-accent)", textTransform: "capitalize" }}>{category}</strong>
      </p>
    </div>
  );
}

export default BatchSummary;


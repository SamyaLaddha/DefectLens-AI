import React, { useState } from "react";
import HeatmapViewer from "./HeatmapViewer";

/**
 * Individual inspection card in the batch results grid.
 * Clickable to activate that image's detail view.
 */
function InspectionCard({ item, previewUrl, isActive, onClick }) {
  const isAnomaly = item.is_anomaly === 1;
  const isError = item.status === "error";

  let borderColor = "var(--color-border)";
  if (isActive) borderColor = "var(--color-accent)";
  else if (isError) borderColor = "var(--color-status-error)";
  else if (isAnomaly) borderColor = "var(--color-status-anomaly)";
  else borderColor = "var(--color-status-success)";

  let statusLabel = isError ? "FAILED" : isAnomaly ? "DEFECT" : "NORMAL";
  let statusColor = isError
    ? "var(--color-status-error)"
    : isAnomaly
      ? "var(--color-status-anomaly)"
      : "var(--color-status-success)";

  return (
    <div
      onClick={onClick}
      title={item.filename}
      style={{
        border: `2px solid ${borderColor}`,
        borderRadius: "var(--radius-sm)",
        overflow: "hidden",
        cursor: "pointer",
        backgroundColor: "var(--color-surface-alt)",
        transition: "var(--transition-base)",
        opacity: isError ? 0.65 : 1
      }}
    >
      {/* Thumbnail */}
      {previewUrl ? (
        <img
          src={previewUrl}
          alt={item.filename}
          style={{ width: "100%", height: "80px", objectFit: "cover", display: "block" }}
        />
      ) : (
        <div style={{
          width: "100%",
          height: "80px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "24px"
        }}>
          {isError ? "⛔" : "🖼"}
        </div>
      )}

      {/* Status badge */}
      <div style={{
        backgroundColor: statusColor,
        color: "#000",
        fontSize: "10px",
        fontWeight: "700",
        padding: "3px 6px",
        textAlign: "center",
        fontFamily: "var(--font-heading)"
      }}>
        {statusLabel}
      </div>

      {/* Score */}
      <div style={{ padding: "6px 8px" }}>
        <div style={{
          fontSize: "10px",
          color: "var(--color-text-muted)",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap"
        }} title={item.filename}>
          {item.filename}
        </div>
        {!isError && (
          <div style={{ fontSize: "11px", color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)", marginTop: "2px" }}>
            {typeof item.anomaly_score === "number" ? item.anomaly_score.toFixed(4) : "—"}
          </div>
        )}
        {isError && (
          <div style={{ fontSize: "10px", color: "var(--color-status-error)", marginTop: "2px" }} title={item.error_message}>
            Error
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * BatchInspectionView — master-detail layout for batch inspection results.
 *
 * Left column: grid of InspectionCards for each submitted image.
 * Right panel: full detail view with metrics and HeatmapViewer for the selected image.
 *
 * Props:
 *   results        — array of per-image result objects from the batch API
 *   previewUrlMap  — Map<image_id, blobUrl> for thumbnail and viewer images
 */
function BatchInspectionView({ results, previewUrlMap }) {
  const [activeId, setActiveId] = useState(results[0]?.image_id ?? null);

  const activeItem = results.find(r => r.image_id === activeId) ?? results[0];
  const activePreviewUrl = activeItem ? (previewUrlMap.get(activeItem.image_id) ?? null) : null;

  const isAnomaly = activeItem?.is_anomaly === 1;
  const isError = activeItem?.status === "error";
  const labelText = isError ? "FAILED" : isAnomaly ? "ANOMALOUS / DEFECT" : "NORMAL / GOOD";
  const labelColor = isError
    ? "var(--color-status-error)"
    : isAnomaly
      ? "var(--color-status-anomaly)"
      : "var(--color-status-success)";

  return (
    <div style={{
      width: "100%",
      maxWidth: "720px",
      backgroundColor: "var(--color-surface)",
      border: "1px solid var(--color-border)",
      borderRadius: "var(--radius-md)",
      padding: "20px",
      marginBottom: "20px",
      boxShadow: "var(--shadow-md)"
    }}>
      <h3 style={{ margin: "0 0 16px", fontFamily: "var(--font-heading)", fontSize: "16px" }}>
        Individual Inspection Results
      </h3>

      {/* Thumbnail grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(110px, 1fr))",
        gap: "10px",
        marginBottom: "20px"
      }}>
        {results.map(item => (
          <InspectionCard
            key={item.image_id}
            item={item}
            previewUrl={previewUrlMap.get(item.image_id) ?? null}
            isActive={item.image_id === activeId}
            onClick={() => setActiveId(item.image_id)}
          />
        ))}
      </div>

      {/* Detail view for the active item */}
      {activeItem && (
        <div style={{
          borderTop: "1px solid var(--color-border)",
          paddingTop: "16px"
        }}>
          {/* Detail header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px", flexWrap: "wrap", gap: "8px" }}>
            <div style={{ fontFamily: "var(--font-heading)", fontSize: "14px", color: "var(--color-text-primary)" }}>
              📸 {activeItem.filename}
            </div>
            <div style={{
              backgroundColor: labelColor,
              color: "#000",
              fontWeight: "700",
              fontSize: "12px",
              padding: "4px 12px",
              borderRadius: "var(--radius-sm)",
              fontFamily: "var(--font-heading)"
            }}>
              {labelText}
            </div>
          </div>

          {/* Error detail */}
          {isError && (
            <div style={{
              backgroundColor: "var(--color-surface-alt)",
              border: "1px solid var(--color-status-error)",
              borderRadius: "var(--radius-sm)",
              padding: "12px 16px",
              fontSize: "13px",
              color: "var(--color-status-error)"
            }}>
              <strong>Processing Error:</strong> {activeItem.error_message}
            </div>
          )}

          {/* Success metrics */}
          {!isError && (
            <>
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: "12px",
                marginBottom: "16px",
                fontSize: "13px",
                color: "var(--color-text-secondary)"
              }}>
                <div>
                  <div><strong>Anomaly Score</strong></div>
                  <div style={{ fontSize: "18px", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", marginTop: "4px" }}>
                    {activeItem.anomaly_score?.toFixed(6) ?? "—"}
                  </div>
                </div>
                <div>
                  <div><strong>Threshold</strong></div>
                  <div style={{ fontSize: "18px", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", marginTop: "4px" }}>
                    {activeItem.threshold?.toFixed(6) ?? "—"}
                  </div>
                </div>
                <div>
                  <div><strong>Inference Latency</strong></div>
                  <div style={{ fontSize: "16px", color: "var(--color-text-primary)", marginTop: "4px" }}>
                    {typeof activeItem.inference_time === "number"
                      ? `${(activeItem.inference_time * 1000).toFixed(2)} ms`
                      : "—"
                    }
                  </div>
                </div>
              </div>

              {/* HeatmapViewer — reused from Module 9 */}
              {activePreviewUrl && activeItem.anomaly_map && (
                <HeatmapViewer
                  imageSrc={activePreviewUrl}
                  anomalyMap={activeItem.anomaly_map}
                  threshold={activeItem.threshold}
                  anomalyScore={activeItem.anomaly_score}
                  isAnomaly={isAnomaly}
                />
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default BatchInspectionView;


import React, { useRef, useState, useCallback } from "react";
import { generateImageId } from "../api/client";

const MIN_BATCH = 4;
const MAX_BATCH = 12;
const MAX_FILE_MB = 10;

/**
 * BatchImageUploader - multi-file staging tray for batch inspection.
 *
 * Manages its own staged file list internally and exposes the finalized
 * list up to the parent only when "Inspect Batch" is triggered.
 *
 * Props:
 *   status       - system connection status string
 *   isBatchMode  - boolean, whether batch upload is active
 *   isBatching   - boolean, true while batch request is in flight
 *   onBatchSubmit(filesWithIds) - called when user submits a valid batch
 */
function BatchImageUploader({ status, isBatching, onBatchSubmit }) {
  const fileInputRef = useRef(null);
  const [stagedItems, setStagedItems] = useState([]); // [{ id, file, previewUrl, sizeLabel, error }]
  const isOffline = status === "error";

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const validateFile = (file) => {
    if (!file.type.startsWith("image/")) return "Not a valid image file type.";
    if (file.size === 0) return "File is empty.";
    if (file.size > MAX_FILE_MB * 1024 * 1024) return `Exceeds ${MAX_FILE_MB} MB limit.`;
    return null;
  };

  const addFiles = useCallback((newFiles) => {
    const toAdd = [];
    for (const file of newFiles) {
      const error = validateFile(file);
      toAdd.push({
        id: generateImageId(),
        file,
        previewUrl: error ? null : URL.createObjectURL(file),
        sizeLabel: formatSize(file.size),
        error
      });
    }

    setStagedItems(prev => {
      // Enforce max cap including what is already staged
      const combined = [...prev, ...toAdd];
      if (combined.length > MAX_BATCH) {
        // Revoke URLs for items we are discarding
        combined.slice(MAX_BATCH).forEach(item => {
          if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
        });
        return combined.slice(0, MAX_BATCH);
      }
      return combined;
    });
  }, []);

  const removeItem = useCallback((id) => {
    setStagedItems(prev => {
      const item = prev.find(i => i.id === id);
      if (item && item.previewUrl) URL.revokeObjectURL(item.previewUrl);
      return prev.filter(i => i.id !== id);
    });
  }, []);

  const clearAll = useCallback(() => {
    setStagedItems(prev => {
      prev.forEach(item => { if (item.previewUrl) URL.revokeObjectURL(item.previewUrl); });
      return [];
    });
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(Array.from(e.target.files));
    }
    // Reset input so the same files can be re-added after removal
    e.target.value = "";
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (isOffline || isBatching) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleSubmit = () => {
    const validItems = stagedItems.filter(i => !i.error);
    if (validItems.length < MIN_BATCH) return;
    onBatchSubmit(validItems.map(({ file, id }) => ({ file, id })));
  };

  const validCount = stagedItems.filter(i => !i.error).length;
  const canSubmit = !isBatching && !isOffline && validCount >= MIN_BATCH;
  const atMax = stagedItems.length >= MAX_BATCH;

  return (
    <div style={{ width: "100%", maxWidth: "720px", marginBottom: "20px" }}>
        {/* Drop zone / add trigger - hidden once at max */}
      {!atMax && (
        <div
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => { if (!isOffline && !isBatching) fileInputRef.current.click(); }}
          style={{
            backgroundColor: "var(--color-surface-alt)",
            border: `2px dashed ${isOffline ? "var(--color-status-error)" : "var(--color-border)"}`,
            borderRadius: "var(--radius-lg)",
            padding: stagedItems.length === 0 ? "40px 20px" : "20px",
            textAlign: "center",
            cursor: isOffline || isBatching ? "not-allowed" : "pointer",
            opacity: isOffline ? 0.5 : 1,
            transition: "var(--transition-base)",
            marginBottom: "16px",
            boxShadow: "var(--shadow-sm)"
          }}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            multiple
            style={{ display: "none" }}
            disabled={isOffline || isBatching}
          />
           <div style={{ fontSize: stagedItems.length === 0 ? "32px" : "20px", marginBottom: "8px" }}>
             {isOffline ? "!" : isBatching ? "..." : "..."}
           </div>
           <p style={{ margin: 0, fontSize: "14px", color: "var(--color-text-secondary)" }}>
             {isOffline
               ? "Upload is disabled because the server is offline."
               : isBatching
                 ? "Batch inspection in progress..."
                 : stagedItems.length === 0
                   ? `Drag and drop ${MIN_BATCH}-${MAX_BATCH} images here, or click to browse.`
                   : `Add more images (${stagedItems.length}/${MAX_BATCH} staged)`
             }
           </p>
        </div>
      )}

      {/* Staged file tray */}
      {stagedItems.length > 0 && (
        <div style={{
          backgroundColor: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-md)",
          padding: "16px",
          marginBottom: "16px"
        }}>
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "12px"
          }}>
            <span style={{ fontFamily: "var(--font-heading)", fontSize: "14px", color: "var(--color-text-secondary)" }}>
               Staged Images - {validCount} valid / {stagedItems.length} total
            </span>
            <button
              onClick={clearAll}
              disabled={isBatching}
              style={{
                background: "none",
                border: "1px solid var(--color-border)",
                color: "var(--color-text-secondary)",
                borderRadius: "var(--radius-sm)",
                padding: "4px 10px",
                fontSize: "12px",
                cursor: isBatching ? "not-allowed" : "pointer"
              }}
            >
              Clear All
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: "10px" }}>
            {stagedItems.map((item) => (
              <div
                key={item.id}
                style={{
                  position: "relative",
                  backgroundColor: "var(--color-surface-alt)",
                  borderRadius: "var(--radius-sm)",
                  border: `1px solid ${item.error ? "var(--color-status-error)" : "var(--color-border)"}`,
                  overflow: "hidden"
                }}
              >
                {/* Thumbnail */}
                {item.previewUrl ? (
                  <img
                    src={item.previewUrl}
                    alt={item.file.name}
                    style={{ width: "100%", height: "80px", objectFit: "cover", display: "block" }}
                  />
                ) : (
                  <div style={{
                    width: "100%",
                    height: "80px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    backgroundColor: "var(--color-surface-alt)",
                    fontSize: "20px"
                  }}>!</div>
                )}

                {/* Filename + size */}
                <div style={{ padding: "6px 8px" }}>
                  <div style={{
                    fontSize: "11px",
                    color: item.error ? "var(--color-status-error)" : "var(--color-text-secondary)",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap"
                  }} title={item.file.name}>
                    {item.file.name}
                  </div>
                  <div style={{ fontSize: "10px", color: "var(--color-text-muted)", marginTop: "2px" }}>
                    {item.error ? item.error : item.sizeLabel}
                  </div>
                </div>

                {/* Remove button */}
               <button
                 onClick={() => removeItem(item.id)}
                 disabled={isBatching}
                 title="Remove image"
                 style={{
                   position: "absolute",
                   top: "4px",
                   right: "4px",
                   background: "var(--color-surface-alt)",
                   border: "none",
                   color: "var(--color-text-primary)",
                   borderRadius: "50%",
                   width: "20px",
                   height: "20px",
                   fontSize: "12px",
                   lineHeight: "20px",
                   textAlign: "center",
                   cursor: isBatching ? "not-allowed" : "pointer",
                   padding: 0
                 }}
               >
                 X
               </button>
              </div>
            ))}
          </div>

          {/* Batch size guidance */}
          {validCount < MIN_BATCH && (
            <p style={{ margin: "12px 0 0", fontSize: "13px", color: "var(--color-status-anomaly)" }}>
              Add at least {MIN_BATCH - validCount} more valid image{MIN_BATCH - validCount !== 1 ? "s" : ""} to enable batch inspection.
            </p>
          )}
          {atMax && (
            <p style={{ margin: "12px 0 0", fontSize: "13px", color: "var(--color-text-muted)" }}>
              Maximum of {MAX_BATCH} images reached.
            </p>
          )}
        </div>
      )}

      {/* Submit button */}
      {stagedItems.length > 0 && (
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          style={{
            width: "100%",
            padding: "14px",
             backgroundColor: canSubmit ? "var(--color-accent)" : "var(--color-surface-alt)",
             color: canSubmit ? "var(--color-text-on-accent)" : "var(--color-text-muted)",
            border: "none",
            borderRadius: "var(--radius-md)",
            fontFamily: "var(--font-heading)",
            fontSize: "15px",
            fontWeight: "700",
            cursor: canSubmit ? "pointer" : "not-allowed",
            transition: "var(--transition-base)"
          }}
        >
          {isBatching
            ? `Analyzing ${validCount} Images...`
            : canSubmit
              ? `Inspect Batch (${validCount} Images)`
              : `Select at least ${MIN_BATCH} valid images to inspect`
          }
        </button>
      )}
    </div>
  );
}

export default BatchImageUploader;


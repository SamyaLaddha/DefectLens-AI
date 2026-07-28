import React, { useRef } from "react";

function ImageUploader({ status, isPredicting, onFileSelected }) {
  const fileInputRef = useRef(null);
  const isOffline = status === "error";

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelected(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (isOffline || isPredicting) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  };

  const triggerFileInput = () => {
    if (isOffline || isPredicting) return;
    fileInputRef.current.click();
  };

  return (
    <div 
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      style={{
        backgroundColor: "var(--color-surface-alt)",
        border: `2px dashed ${isOffline ? "var(--color-status-error)" : "var(--color-border)"}`,
        borderRadius: "var(--radius-lg)",
        padding: "40px 20px",
        width: "100%",
        maxWidth: "600px",
        textAlign: "center",
        cursor: isOffline || isPredicting ? "not-allowed" : "pointer",
        opacity: isOffline ? 0.5 : 1,
        transition: "var(--transition-base)",
        marginBottom: "20px",
        boxShadow: "var(--shadow-sm)"
      }}
      onClick={triggerFileInput}
    >
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        accept="image/*" 
        style={{ display: "none" }}
        disabled={isOffline || isPredicting}
      />
      
      <div style={{ fontSize: "32px", marginBottom: "15px" }}>
        {isPredicting ? "⏳" : isOffline ? "🔌" : "📷"}
      </div>

      <h4 style={{ margin: "0 0 10px 0", fontFamily: "var(--font-heading)", color: "var(--color-text-primary)" }}>
        {isPredicting ? "Analyzing Image..." : isOffline ? "Backend Unreachable" : "Upload Inspection Image"}
      </h4>

      <p style={{ margin: 0, fontSize: "14px", color: "var(--color-text-secondary)" }}>
        {isPredicting 
          ? "Running STFPM feature discrepancy matching..." 
          : isOffline 
            ? "Upload is disabled because the server is offline." 
            : "Drag and drop an image file here, or click to browse files."
        }
      </p>
    </div>
  );
}

export default ImageUploader;

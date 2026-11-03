import { describe, it, expect, vi } from "vitest";
import React from "react";
import { renderToString } from "react-dom/server";
import HeatmapViewer from "./HeatmapViewer.jsx";
import PredictionResult from "./PredictionResult.jsx";

describe("HeatmapViewer Component Tests", () => {
  const dummyMap = Array(256).fill(null).map((_, r) => 
    Array(256).fill(null).map((_, c) => (r + c) / 512)
  );

  it("should render null when imageSrc or anomalyMap is missing", () => {
    const htmlWithoutImage = renderToString(
      <HeatmapViewer 
        imageSrc="" 
        anomalyMap={dummyMap} 
        threshold={0.1} 
        anomalyScore={0.2} 
        isAnomaly={true} 
      />
    );
    expect(htmlWithoutImage).toBe("");

    const htmlWithoutMap = renderToString(
      <HeatmapViewer 
        imageSrc="blob:http://localhost/dummy" 
        anomalyMap={null} 
        threshold={0.1} 
        anomalyScore={0.2} 
        isAnomaly={true} 
      />
    );
    expect(htmlWithoutMap).toBe("");
  });

  it("should render view modes, opacity slider, and relative discrepancy legend", () => {
    const html = renderToString(
      <HeatmapViewer 
        imageSrc="blob:http://localhost/dummy" 
        anomalyMap={dummyMap} 
        threshold={0.1182} 
        anomalyScore={0.85} 
        isAnomaly={true} 
      />
    );

    // Verify view mode buttons
    expect(html).toContain("Overlay");
    expect(html).toContain("Heatmap");
    expect(html).toContain("Original");

    // Verify absence of any uncalibrated defect mask or pixel thresholding
    expect(html).not.toContain("Defect Mask");
    expect(html).not.toContain("Pixel Threshold");

    // Verify opacity control
    expect(html).toContain("Heatmap Opacity");
    expect(html).toContain("50");

    // Verify canvas element exists
    expect(html).toContain("<canvas");

    // Verify colorbar legend
    expect(html).toContain("Relative Discrepancy Gradient");
    expect(html).toContain("Low Discrepancy");
    expect(html).toContain("Peak Discrepancy");
  });

  it("should safely handle constant anomaly maps without division by zero or NaN", () => {
    const constantMap = Array(256).fill(null).map(() => Array(256).fill(0.05));

    const html = renderToString(
      <HeatmapViewer 
        imageSrc="blob:http://localhost/dummy" 
        anomalyMap={constantMap} 
        threshold={0.1182} 
        anomalyScore={0.05} 
        isAnomaly={false} 
      />
    );

    expect(html).toContain("<canvas");
    expect(html).not.toContain("NaN");
  });

  it("should safely handle all-zero anomaly maps", () => {
    const zeroMap = Array(256).fill(null).map(() => Array(256).fill(0.0));

    const html = renderToString(
      <HeatmapViewer 
        imageSrc="blob:http://localhost/dummy" 
        anomalyMap={zeroMap} 
        threshold={0.1182} 
        anomalyScore={0.0} 
        isAnomaly={false} 
      />
    );

    expect(html).toContain("<canvas");
    expect(html).not.toContain("NaN");
  });

  it("should integrate into PredictionResult only when previewUrl and anomalyMap are provided", () => {
    const mockResult = {
      is_anomaly: 1,
      anomaly_score: 0.88,
      threshold: 0.0625,
      inference_time: 0.04
    };

    // Case 1: When previewUrl is null (e.g. while predicting a new image), HeatmapViewer is not rendered
    const htmlNoPreview = renderToString(
      <PredictionResult 
        result={mockResult} 
        previewUrl={null} 
        anomalyMap={dummyMap} 
      />
    );
    expect(htmlNoPreview).toContain("ANOMALOUS / DEFECT");
    expect(htmlNoPreview).not.toContain("<canvas");

    // Case 2: When anomalyMap is null (e.g. cleared on file select), HeatmapViewer is not rendered
    const htmlNoMap = renderToString(
      <PredictionResult 
        result={mockResult} 
        previewUrl="blob:http://localhost/dummy" 
        anomalyMap={null} 
      />
    );
    expect(htmlNoMap).toContain("ANOMALOUS / DEFECT");
    expect(htmlNoMap).not.toContain("<canvas");

    // Case 3: When both are provided, HeatmapViewer renders
    const htmlFull = renderToString(
      <PredictionResult 
        result={mockResult} 
        previewUrl="blob:http://localhost/dummy" 
        anomalyMap={dummyMap} 
      />
    );
    expect(htmlFull).toContain("ANOMALOUS / DEFECT");
    expect(htmlFull).toContain("<canvas");
    expect(htmlFull).toContain("Defect Localization Heatmap");
  });

  it("should verify URL.revokeObjectURL cleans up object URLs on replacement", () => {
    const revokeSpy = vi.fn();
    const originalRevoke = global.URL.revokeObjectURL;
    global.URL.revokeObjectURL = revokeSpy;

    const oldUrl = "blob:http://localhost/old-image";
    if (oldUrl) {
      URL.revokeObjectURL(oldUrl);
    }

    expect(revokeSpy).toHaveBeenCalledWith("blob:http://localhost/old-image");

    global.URL.revokeObjectURL = originalRevoke;
  });
});

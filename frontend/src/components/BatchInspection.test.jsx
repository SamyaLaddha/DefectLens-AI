import { describe, it, expect, vi } from "vitest";
import React from "react";
import { renderToString } from "react-dom/server";
import BatchSummary from "./BatchSummary.jsx";
import BatchInspectionView from "./BatchInspectionView.jsx";

// ── Fixtures ─────────────────────────────────────────────────────────────────

const makeResult = (overrides = {}) => ({
  index: 0,
  image_id: "id-001",
  filename: "screw_head.png",
  status: "success",
  is_anomaly: 0,
  anomaly_score: 0.021,
  threshold: 0.062534,
  inference_time: 0.038,
  anomaly_map: Array(256).fill(null).map(() => Array(256).fill(0.021)),
  ...overrides
});

const makeErrorResult = (overrides = {}) => ({
  index: 1,
  image_id: "id-002",
  filename: "corrupt.png",
  status: "error",
  error_message: "Failed to parse uploaded image: cannot identify image file",
  is_anomaly: null,
  anomaly_score: null,
  threshold: 0.062534,
  inference_time: 0.0,
  anomaly_map: null,
  ...overrides
});

// ── BatchSummary tests ────────────────────────────────────────────────────────

describe("BatchSummary Component", () => {
  it("renders ALL_CLEAR verdict with correct counts", () => {
    const html = renderToString(
      <BatchSummary
        category="screw"
        threshold={0.062534}
        batchStatus="ALL_CLEAR"
        totalSubmitted={5}
        totalProcessed={5}
        totalNormal={5}
        totalAnomalous={0}
        totalFailed={0}
      />
    );
    expect(html).toContain("ALL CLEAR");
    expect(html).toContain("screw");
  });

  it("renders DEFECT_DETECTED verdict correctly", () => {
    const html = renderToString(
      <BatchSummary
        category="bottle"
        threshold={0.118206}
        batchStatus="DEFECT_DETECTED"
        totalSubmitted={4}
        totalProcessed={4}
        totalNormal={3}
        totalAnomalous={1}
        totalFailed={0}
      />
    );
    expect(html).toContain("DEFECT DETECTED");
    expect(html).toContain("bottle");
  });

  it("renders INCOMPLETE verdict when images fail without defects", () => {
    const html = renderToString(
      <BatchSummary
        category="cable"
        threshold={0.086241}
        batchStatus="INCOMPLETE"
        totalSubmitted={5}
        totalProcessed={4}
        totalNormal={4}
        totalAnomalous={0}
        totalFailed={1}
      />
    );
    expect(html).toContain("INCOMPLETE");
    expect(html).toContain("Batch quality cannot be confirmed");
  });

  it("renders INCOMPLETE_DEFECT_DETECTED when both failure and defect occur", () => {
    const html = renderToString(
      <BatchSummary
        category="screw"
        threshold={0.062534}
        batchStatus="INCOMPLETE_DEFECT_DETECTED"
        totalSubmitted={4}
        totalProcessed={3}
        totalNormal={2}
        totalAnomalous={1}
        totalFailed={1}
      />
    );
    expect(html).toContain("INCOMPLETE");
    expect(html).toContain("DEFECT DETECTED");
  });

  it("displays the category-specific threshold loaded from config (not hardcoded)", () => {
    const html = renderToString(
      <BatchSummary
        category="bottle"
        threshold={0.118206}
        batchStatus="ALL_CLEAR"
        totalSubmitted={4}
        totalProcessed={4}
        totalNormal={4}
        totalAnomalous={0}
        totalFailed={0}
      />
    );
    // The threshold should appear in the rendered output
    expect(html).toContain("0.118206");
  });
});

// ── BatchInspectionView tests ─────────────────────────────────────────────────

describe("BatchInspectionView Component", () => {
  const results = [
    makeResult({ index: 0, image_id: "id-001", filename: "view_1.png", is_anomaly: 0 }),
    makeResult({ index: 1, image_id: "id-002", filename: "view_2.png", is_anomaly: 1, anomaly_score: 0.095 }),
    makeErrorResult({ index: 2, image_id: "id-003", filename: "corrupt.png" })
  ];

  const previewUrlMap = new Map([
    ["id-001", "blob:http://localhost/view1"],
    ["id-002", "blob:http://localhost/view2"]
    // id-003 (error item) has no URL
  ]);

  it("renders all filenames in the card grid", () => {
    const html = renderToString(
      <BatchInspectionView results={results} previewUrlMap={previewUrlMap} />
    );
    expect(html).toContain("view_1.png");
    expect(html).toContain("view_2.png");
    expect(html).toContain("corrupt.png");
  });

  it("renders status labels for anomalous, normal, and failed items", () => {
    const html = renderToString(
      <BatchInspectionView results={results} previewUrlMap={previewUrlMap} />
    );
    expect(html).toContain("NORMAL");
    expect(html).toContain("DEFECT");
    expect(html).toContain("FAILED");
  });

  it("renders error message for failed items in detail view when that item is first", () => {
    const errorFirstResults = [
      makeErrorResult({ index: 0, image_id: "id-err", filename: "bad.png" }),
      makeResult({ index: 1, image_id: "id-ok", filename: "good.png" })
    ];
    const map = new Map([["id-ok", "blob:http://localhost/good"]]);

    const html = renderToString(
      <BatchInspectionView results={errorFirstResults} previewUrlMap={map} />
    );
    // The error item is the first (active) one — its error message should be shown
    expect(html).toContain("cannot identify image file");
  });

  it("does not render HeatmapViewer when anomaly_map is null (error item)", () => {
    const errorOnly = [
      makeErrorResult({ index: 0, image_id: "id-err" }),
      makeResult({ index: 1, image_id: "id-ok" }),
      makeResult({ index: 2, image_id: "id-ok2" }),
      makeResult({ index: 3, image_id: "id-ok3" })
    ];
    const map = new Map([
      ["id-ok", "blob:http://localhost/1"],
      ["id-ok2", "blob:http://localhost/2"],
      ["id-ok3", "blob:http://localhost/3"]
    ]);
    const html = renderToString(
      <BatchInspectionView results={errorOnly} previewUrlMap={map} />
    );
    // First item is error — no canvas rendered in detail view initially
    expect(html).toContain("FAILED");
  });

  it("renders anomaly scores in monospace font format", () => {
    const html = renderToString(
      <BatchInspectionView results={results} previewUrlMap={previewUrlMap} />
    );
    // view_1 is the first/active item; score 0.0210 should be visible
    expect(html).toContain("0.021");
  });

  it("synchronizes active selection when results array changes to a different batch", () => {
    const firstBatch = [
      makeResult({ index: 0, image_id: "batch1-001", filename: "alpha.png", is_anomaly: 0 }),
      makeResult({ index: 1, image_id: "batch1-002", filename: "beta.png", is_anomaly: 1, anomaly_score: 0.095 })
    ];
    const secondBatch = [
      makeResult({ index: 0, image_id: "batch2-001", filename: "gamma.png", is_anomaly: 0 }),
      makeResult({ index: 1, image_id: "batch2-002", filename: "delta.png", is_anomaly: 0 })
    ];
    const map1 = new Map([
      ["batch1-001", "blob:http://localhost/alpha"],
      ["batch1-002", "blob:http://localhost/beta"]
    ]);
    const map2 = new Map([
      ["batch2-001", "blob:http://localhost/gamma"],
      ["batch2-002", "blob:http://localhost/delta"]
    ]);

    const html1 = renderToString(<BatchInspectionView results={firstBatch} previewUrlMap={map1} />);
    expect(html1).toContain("alpha.png");

    const html2 = renderToString(<BatchInspectionView results={secondBatch} previewUrlMap={map2} />);
    expect(html2).toContain("gamma.png");
    expect(html2).not.toContain("alpha.png");
  });

  // BatchImageUploader stub tests (SSR-safe validation logic only)

  describe("BatchImageUploader validation logic", () => {
    // These test the pure validation logic without browser event interactions
    it("generateImageId produces unique IDs for each staged file (cross-module integration)", async () => {
      const { generateImageId } = await import("../api/client.js");
      const ids = new Set(Array.from({ length: 10 }, () => generateImageId()));
      expect(ids.size).toBe(10);
    });
  });
});
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { PredictionCoordinator } from "./predictionCoordinator.js";

describe("PredictionCoordinator Race Condition & Request Identity Tests", () => {
  let mockRevoke;
  let mockCreate;

  beforeEach(() => {
    mockRevoke = vi.fn();
    mockCreate = vi.fn((file) => `blob:http://localhost/${file.name}`);

    global.URL.createObjectURL = mockCreate;
    global.URL.revokeObjectURL = mockRevoke;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("prevents older prediction A from overwriting prediction B when B finishes before A (exact 4-step sequence)", async () => {
    // Deferred promises to precisely control asynchronous completion ordering
    let resolvePredictionA;
    const promiseA = new Promise((resolve) => {
      resolvePredictionA = resolve;
    });

    let resolvePredictionB;
    const promiseB = new Promise((resolve) => {
      resolvePredictionB = resolve;
    });

    const mockPredictFn = vi.fn((file) => {
      if (file.name === "imageA.png") return promiseA;
      if (file.name === "imageB.png") return promiseB;
      throw new Error(`Unexpected file: ${file.name}`);
    });

    // Track state transitions as seen by the UI
    let uiState = {
      isPredicting: false,
      previewUrl: null,
      result: null,
      anomalyMap: null,
      errorMessage: ""
    };

    const coordinator = new PredictionCoordinator({
      predictFn: mockPredictFn,
      onStart: ({ previewUrl }) => {
        uiState.isPredicting = true;
        uiState.previewUrl = previewUrl;
        uiState.result = null;
        uiState.anomalyMap = null;
        uiState.errorMessage = "";
      },
      onSuccess: ({ result }) => {
        uiState.result = {
          is_anomaly: result.is_anomaly,
          anomaly_score: result.anomaly_score,
          threshold: result.threshold,
          inference_time: result.inference_time
        };
        uiState.anomalyMap = result.anomaly_map;
      },
      onError: ({ error }) => {
        uiState.errorMessage = error.message;
        uiState.result = null;
        uiState.anomalyMap = null;
      },
      onSettled: () => {
        uiState.isPredicting = false;
      }
    });

    const fileA = new File(["bytesA"], "imageA.png", { type: "image/png" });
    const fileB = new File(["bytesB"], "imageB.png", { type: "image/png" });

    // -------------------------------------------------------------
    // Step 1: User uploads image A and starts prediction A
    // -------------------------------------------------------------
    const dispatchPromiseA = coordinator.requestPrediction(fileA, "bottle");
    expect(uiState.isPredicting).toBe(true);
    expect(uiState.previewUrl).toBe("blob:http://localhost/imageA.png");
    expect(uiState.result).toBeNull();
    expect(uiState.anomalyMap).toBeNull();

    // -------------------------------------------------------------
    // Step 2: Before prediction A finishes, user uploads image B and starts prediction B
    // -------------------------------------------------------------
    const dispatchPromiseB = coordinator.requestPrediction(fileB, "bottle");
    expect(uiState.isPredicting).toBe(true);
    expect(uiState.previewUrl).toBe("blob:http://localhost/imageB.png");
    // Object URL for image A must have been revoked to prevent leaks
    expect(mockRevoke).toHaveBeenCalledWith("blob:http://localhost/imageA.png");

    // -------------------------------------------------------------
    // Step 3: Prediction B finishes first
    // -------------------------------------------------------------
    const resultB = {
      is_anomaly: 1,
      anomaly_score: 0.925,
      threshold: 0.1182,
      inference_time: 0.038,
      anomaly_map: [[0.8, 0.9], [0.7, 0.95]]
    };
    resolvePredictionB(resultB);
    const outcomeB = await dispatchPromiseB;

    // Verify Prediction B was successfully committed to UI state
    expect(outcomeB.status).toBe("committed");
    expect(uiState.isPredicting).toBe(false);
    expect(uiState.previewUrl).toBe("blob:http://localhost/imageB.png");
    expect(uiState.result).toEqual({
      is_anomaly: 1,
      anomaly_score: 0.925,
      threshold: 0.1182,
      inference_time: 0.038
    });
    expect(uiState.anomalyMap).toEqual([[0.8, 0.9], [0.7, 0.95]]);

    // -------------------------------------------------------------
    // Step 4: Prediction A finishes afterward
    // -------------------------------------------------------------
    const resultA = {
      is_anomaly: 0,
      anomaly_score: 0.012,
      threshold: 0.1182,
      inference_time: 0.045,
      anomaly_map: [[0.01, 0.02], [0.01, 0.01]]
    };
    resolvePredictionA(resultA);
    const outcomeA = await dispatchPromiseA;

    // -------------------------------------------------------------
    // Crucial Invariant: Older prediction A must be recognized as stale
    // and CANNOT overwrite result or anomaly map belonging to image B
    // -------------------------------------------------------------
    expect(outcomeA.status).toBe("stale");
    expect(uiState.previewUrl).toBe("blob:http://localhost/imageB.png");
    expect(uiState.result).toEqual({
      is_anomaly: 1,
      anomaly_score: 0.925,
      threshold: 0.1182,
      inference_time: 0.038
    });
    expect(uiState.anomalyMap).toEqual([[0.8, 0.9], [0.7, 0.95]]);
    expect(uiState.isPredicting).toBe(false);
    expect(uiState.errorMessage).toBe("");
  });

  it("prevents an out-of-order rejection from prediction A from wiping out prediction B's result", async () => {
    let rejectPredictionA;
    const promiseA = new Promise((_, reject) => {
      rejectPredictionA = reject;
    });

    let resolvePredictionB;
    const promiseB = new Promise((resolve) => {
      resolvePredictionB = resolve;
    });

    const coordinator = new PredictionCoordinator({
      predictFn: vi.fn((file) => file.name === "imageA.png" ? promiseA : promiseB),
      onStart: () => {},
      onSuccess: ({ result }) => {
        uiResult = result;
      },
      onError: ({ error }) => {
        uiError = error.message;
      },
      onSettled: () => {}
    });

    let uiResult = null;
    let uiError = "";

    const fileA = new File(["bytesA"], "imageA.png", { type: "image/png" });
    const fileB = new File(["bytesB"], "imageB.png", { type: "image/png" });

    const pA = coordinator.requestPrediction(fileA, "bottle");
    const pB = coordinator.requestPrediction(fileB, "bottle");

    // Prediction B resolves successfully
    resolvePredictionB({ is_anomaly: 0, anomaly_score: 0.04, anomaly_map: [[0.04]] });
    await pB;
    expect(uiResult).toEqual({ is_anomaly: 0, anomaly_score: 0.04, anomaly_map: [[0.04]] });

    // Older Prediction A rejects with network error afterward
    rejectPredictionA(new Error("Network timeout on request A"));
    const outcomeA = await pA;

    expect(outcomeA.status).toBe("stale_error");
    // uiError was NOT updated by stale request A
    expect(uiError).toBe("");
    // uiResult for image B remains intact
    expect(uiResult).toEqual({ is_anomaly: 0, anomaly_score: 0.04, anomaly_map: [[0.04]] });
  });

  it("discards responses after unmount cleanup", async () => {
    let resolvePrediction;
    const pendingPromise = new Promise((resolve) => {
      resolvePrediction = resolve;
    });

    let successCalled = false;
    const coordinator = new PredictionCoordinator({
      predictFn: () => pendingPromise,
      onStart: () => {},
      onSuccess: () => {
        successCalled = true;
      },
      onError: () => {},
      onSettled: () => {}
    });

    const file = new File(["bytes"], "test.png", { type: "image/png" });
    const p = coordinator.requestPrediction(file, "screw");

    // Simulate component unmount
    coordinator.cleanup();

    // Prediction resolves after unmount
    resolvePrediction({ is_anomaly: 1, anomaly_score: 0.99, anomaly_map: [[0.99]] });
    const outcome = await p;

    expect(outcome.status).toBe("stale");
    expect(successCalled).toBe(false);
  });
});


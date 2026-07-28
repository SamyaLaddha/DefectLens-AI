import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { healthCheck, predictImage } from "./client.js";

describe("DefectLens API Client Tests", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://127.0.0.1:8000");
    vi.spyOn(global, "fetch");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it("should successfully execute health check", async () => {
    const mockHealth = {
      status: "healthy",
      device: "cpu",
      category: "bottle",
      threshold: 0.5,
      model_loaded: true
    };

    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockHealth
    });

    const result = await healthCheck();
    expect(global.fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/health",
      expect.any(Object)
    );
    expect(result).toEqual(mockHealth);
  });

  it("should handle degraded health response", async () => {
    const mockDegraded = {
      status: "degraded",
      device: "none",
      category: "none",
      threshold: 0.0,
      model_loaded: false
    };

    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockDegraded
    });

    const result = await healthCheck();
    expect(result.status).toBe("degraded");
    expect(result.model_loaded).toBe(false);
  });

  it("should handle unreachable server health responses", async () => {
    global.fetch.mockRejectedValueOnce(new Error("Failed to fetch"));

    const result = await healthCheck();
    expect(result.status).toBe("error");
    expect(result.message).toContain("Network connection error");
  });

  it("should execute image prediction successfully", async () => {
    const mockPrediction = {
      is_anomaly: 0,
      anomaly_score: 0.35,
      anomaly_map: [[0.1, 0.2]],
      threshold: 0.5,
      inference_time: 0.03
    };

    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockPrediction
    });

    const mockFile = new File(["dummy bytes"], "test.png", { type: "image/png" });
    const result = await predictImage(mockFile);

    expect(global.fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/predict",
      expect.objectContaining({
        method: "POST",
        body: expect.any(FormData)
      })
    );
    expect(result).toEqual(mockPrediction);
  });

  it("should catch degraded service prediction error (503)", async () => {
    const mockErrResponse = {
      message: "Model checkpoint is not loaded. Ensure checkpoints are available."
    };

    global.fetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => mockErrResponse
    });

    const mockFile = new File(["dummy bytes"], "test.png", { type: "image/png" });
    
    await expect(predictImage(mockFile)).rejects.toThrow(
      "Model checkpoint is not loaded"
    );
  });

  it("should handle request timeout using AbortController", async () => {
    const abortError = new Error("The user aborted a request.");
    abortError.name = "AbortError";
    
    global.fetch.mockRejectedValueOnce(abortError);

    const mockFile = new File(["dummy bytes"], "test.png", { type: "image/png" });
    await expect(predictImage(mockFile)).rejects.toThrow("Request timed out");
  });
});

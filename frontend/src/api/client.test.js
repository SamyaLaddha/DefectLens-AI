import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { healthCheck, getCategories, predictImage } from "./client.js";

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
      threshold: 0.1182,
      loaded_categories: ["bottle", "cable", "screw"],
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
      loaded_categories: [],
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
    expect(result.loaded_categories).toEqual([]);
  });

  it("should handle unreachable server health responses", async () => {
    global.fetch.mockRejectedValueOnce(new Error("Failed to fetch"));

    const result = await healthCheck();
    expect(result.status).toBe("error");
    expect(result.message).toContain("Network connection error");
  });

  it("should successfully fetch active categories", async () => {
    const mockCategories = ["bottle", "cable", "screw"];

    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockCategories
    });

    const result = await getCategories();
    expect(global.fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/categories",
      expect.any(Object)
    );
    expect(result).toEqual(mockCategories);
  });

  it("should fail prediction when category is omitted", async () => {
    const mockFile = new File(["dummy bytes"], "test.png", { type: "image/png" });
    await expect(predictImage(mockFile, "")).rejects.toThrow(
      "No category specified for prediction."
    );
  });

  it("should execute image prediction with category query param", async () => {
    const mockPrediction = {
      is_anomaly: 1,
      anomaly_score: 0.88,
      anomaly_map: [[0.1, 0.2]],
      threshold: 0.0625,
      inference_time: 0.04,
      category: "screw"
    };

    global.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockPrediction
    });

    const mockFile = new File(["dummy bytes"], "test.png", { type: "image/png" });
    const result = await predictImage(mockFile, "screw");

    expect(global.fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/predict?category=screw",
      expect.objectContaining({
        method: "POST",
        body: expect.any(FormData)
      })
    );
    expect(result).toEqual(mockPrediction);
  });

  it("should catch degraded service prediction error (503)", async () => {
    const mockErrResponse = {
      message: "Model registry is not initialized or no models are loaded."
    };

    global.fetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => mockErrResponse
    });

    const mockFile = new File(["dummy bytes"], "test.png", { type: "image/png" });
    
    await expect(predictImage(mockFile, "bottle")).rejects.toThrow(
      "Model registry is not initialized"
    );
  });

  it("should handle request timeout using AbortController", async () => {
    const abortError = new Error("The user aborted a request.");
    abortError.name = "AbortError";
    
    global.fetch.mockRejectedValueOnce(abortError);

    const mockFile = new File(["dummy bytes"], "test.png", { type: "image/png" });
    await expect(predictImage(mockFile, "bottle")).rejects.toThrow("Request timed out");
  });
});

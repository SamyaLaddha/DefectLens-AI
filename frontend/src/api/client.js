/**
 * Centralized API client for DefectLens AI.
 * 
 * Uses the native fetch API to communicate with the backend.
 * All base URLs are resolved dynamically from VITE_API_BASE_URL.
 */

/**
 * Returns the resolved base URL from environmental parameters.
 */
function getBaseUrl() {
  return import.meta.env.VITE_API_BASE_URL || "";
}

/**
 * Executes a fetch request with a configurable timeout.
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    if (error.name === "AbortError") {
      throw new Error("Request timed out. Please check if the backend server is running.");
    }
    throw new Error("Network connection error. Server is unreachable.");
  }
}

/**
 * Inspects backend operational status.
 * Executes once during startup.
 */
export async function healthCheck() {
  const url = `${getBaseUrl()}/api/health`;
  try {
    const response = await fetchWithTimeout(url, { method: "GET" });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return {
        status: "error",
        message: errorData.message || `HTTP error: ${response.status}`,
        device: "none",
        category: "none",
        threshold: 0.0,
        model_loaded: false
      };
    }
    return await response.json();
  } catch (error) {
    return {
      status: "error",
      message: error.message,
      device: "none",
      category: "none",
      threshold: 0.0,
      model_loaded: false
    };
  }
}

/**
 * Inquires active categories available for inspection from the backend.
 */
export async function getCategories() {
  const url = `${getBaseUrl()}/api/categories`;
  const response = await fetchWithTimeout(url, { method: "GET" });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || `Failed to fetch categories: HTTP ${response.status}`);
  }
  return await response.json();
}

/**
 * Generates a cryptographically unique ID for a staged image.
 * Uses crypto.randomUUID() with a crypto.getRandomValues() fallback.
 * Never uses Math.random().
 */
export function generateImageId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback: construct a UUID v4 from crypto.getRandomValues()
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant bits
  const hex = Array.from(bytes).map(b => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/**
 * Submits a batch of images (4–12) with their client-assigned IDs for independent
 * anomaly prediction on the specified category.
 * filesWithIds: Array of { file: File, id: string } objects.
 */
export async function predictBatch(filesWithIds, category) {
  if (!category) {
    throw new Error("No category specified for batch inspection.");
  }
  if (!filesWithIds || filesWithIds.length === 0) {
    throw new Error("No images provided for batch inspection.");
  }
  if (filesWithIds.length < 4) {
    throw new Error(`Batch inspection requires at least 4 images (provided ${filesWithIds.length}).`);
  }
  if (filesWithIds.length > 12) {
    throw new Error(`Batch inspection allows at most 12 images (provided ${filesWithIds.length}).`);
  }

  const url = `${getBaseUrl()}/api/predict/batch?category=${encodeURIComponent(category)}`;

  const formData = new FormData();
  for (const { file, id } of filesWithIds) {
    formData.append("files", file);
    formData.append("image_ids", id);
  }

  // Use a longer timeout for batch: 30s base + 5s per image above 4
  const timeoutMs = 30000 + Math.max(0, filesWithIds.length - 4) * 5000;
  const response = await fetchWithTimeout(url, {
    method: "POST",
    body: formData
  }, timeoutMs);

  const responseData = await response.json().catch(() => ({}));

  if (!response.ok) {
    const defaultMsg = `Batch request failed: HTTP ${response.status}`;
    const errorMsg = responseData.message || responseData.detail || defaultMsg;
    const apiError = new Error(errorMsg);
    apiError.status = response.status;
    throw apiError;
  }

  return responseData;
}

/**
 * Transmits a single image file for anomaly prediction on the specified category.
 */
export async function predictImage(file, category) {
  if (!category) {
    throw new Error("No category specified for prediction.");
  }
  if (!file) {
    throw new Error("No image file selected for prediction.");
  }

  const url = `${getBaseUrl()}/api/predict?category=${encodeURIComponent(category)}`;

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetchWithTimeout(url, {
    method: "POST",
    body: formData
  }, 60000);

  const responseData = await response.json().catch(() => ({}));

  if (!response.ok) {
    const defaultMsg = `Server returned status: ${response.status}`;
    const errorMsg = responseData.message || responseData.detail || defaultMsg;

    // Create custom error carrying HTTP status details
    const apiError = new Error(errorMsg);
    apiError.status = response.status;
    throw apiError;
  }

  return responseData;
}

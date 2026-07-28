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
 * Transmits a single image file for anomaly prediction.
 */
export async function predictImage(file) {
  const url = `${getBaseUrl()}/api/predict`;
  
  if (!file) {
    throw new Error("No image file selected for prediction.");
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetchWithTimeout(url, {
    method: "POST",
    body: formData
  });

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

/**
 * PredictionCoordinator
 * 
 * Manages the asynchronous prediction request lifecycle and prevents race conditions
 * where an earlier in-flight prediction finishes after a newer prediction and corrupts
 * or overwrites the active UI state.
 */
export class PredictionCoordinator {
  constructor({
    predictFn,
    onStart,
    onSuccess,
    onError,
    onSettled
  } = {}) {
    this.activeRequestId = 0;
    this.currentPreviewUrl = null;
    this.predictFn = predictFn;
    this.onStart = onStart;
    this.onSuccess = onSuccess;
    this.onError = onError;
    this.onSettled = onSettled;
  }

  /**
   * Dispatches a prediction request for the given file and category.
   * Increments the internal activeRequestId to immediately invalidate any
   * pending in-flight requests.
   */
  async requestPrediction(file, category) {
    if (!file) return;

    // Increment request token: any previous in-flight request is now stale
    const requestId = ++this.activeRequestId;

    // Clean up prior object URL to prevent memory leaks
    if (this.currentPreviewUrl) {
      try {
        if (typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
          URL.revokeObjectURL(this.currentPreviewUrl);
        }
      } catch (_) {
        // Safe no-op in headless environments
      }
    }

    // Create fresh object URL for the newly selected image
    let previewUrl = null;
    try {
      if (typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function') {
        previewUrl = URL.createObjectURL(file);
      }
    } catch (_) {
      previewUrl = null;
    }
    this.currentPreviewUrl = previewUrl;

    // Notify caller that prediction has started for this request
    this.onStart?.({
      requestId,
      previewUrl
    });

    try {
      const result = await this.predictFn(file, category);

      // Guard: If another request was dispatched while this prediction was processing,
      // silently discard this response.
      if (requestId !== this.activeRequestId) {
        return { status: 'stale', requestId, result };
      }

      this.onSuccess?.({
        result,
        requestId
      });

      return { status: 'committed', requestId, result };
    } catch (error) {
      // Guard: Discard stale errors to prevent wiping out a newer valid prediction
      if (requestId !== this.activeRequestId) {
        return { status: 'stale_error', requestId, error };
      }

      this.onError?.({
        error,
        requestId
      });

      return { status: 'error', requestId, error };
    } finally {
      // Guard: Only clear the loading state if this request is still the active one
      if (requestId === this.activeRequestId) {
        this.onSettled?.({ requestId });
      }
    }
  }

  /**
   * Invalidates any pending requests and revokes active preview URLs.
   * Called when component unmounts.
   */
  cleanup() {
    this.activeRequestId++;
    if (this.currentPreviewUrl) {
      try {
        if (typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
          URL.revokeObjectURL(this.currentPreviewUrl);
        }
      } catch (_) {}
      this.currentPreviewUrl = null;
    }
  }

  getActiveRequestId() {
    return this.activeRequestId;
  }

  getCurrentPreviewUrl() {
    return this.currentPreviewUrl;
  }
}


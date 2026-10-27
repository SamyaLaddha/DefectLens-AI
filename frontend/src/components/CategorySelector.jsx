import React, { useState, useEffect } from "react";
import { getCategories } from "../api/client";

function CategorySelector({ selectedCategory, onCategoryChanged, systemStatus }) {
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function fetchCategories() {
      try {
        setLoading(true);
        const data = await getCategories();
        setCategories(data);
        setError("");
        if (data.length > 0) {
          // Pre-select the first category if none is selected
          if (!selectedCategory || !data.includes(selectedCategory)) {
            onCategoryChanged(data[0]);
          }
        }
      } catch (err) {
        setError(err.message || "Failed to load categories.");
      } finally {
        setLoading(false);
      }
    }

    if (systemStatus !== "error") {
      fetchCategories();
    } else {
      setLoading(false);
      setError("Backend is unreachable.");
    }
  }, [systemStatus]);

  if (loading) {
    return (
      <div style={{ color: "var(--color-text-secondary)", fontSize: "14px", marginBottom: "15px" }}>
        Loading categories...
      </div>
    );
  }

  if (error && systemStatus === "error") {
    return null; // ConnectionStatus already displays backend offline error
  }

  if (error) {
    return (
      <div style={{
        color: "var(--color-status-error)",
        fontSize: "14px",
        marginBottom: "15px",
        padding: "10px",
        border: "1px solid var(--color-status-error)",
        borderRadius: "var(--radius-md)",
        backgroundColor: "rgba(239, 68, 68, 0.05)"
      }}>
        Error loading categories: {error}
      </div>
    );
  }

  if (categories.length === 0) {
    return (
      <div style={{
        color: "var(--color-status-anomaly)",
        fontSize: "14px",
        marginBottom: "15px",
        padding: "10px",
        border: "1px solid var(--color-status-anomaly)",
        borderRadius: "var(--radius-md)",
        backgroundColor: "rgba(245, 158, 11, 0.05)"
      }}>
        No active category models are currently loaded on the backend.
      </div>
    );
  }

  return (
    <div style={{
      width: "100%",
      maxWidth: "600px",
      marginBottom: "20px",
      display: "flex",
      flexDirection: "column",
      gap: "8px"
    }}>
      <label htmlFor="category-select" style={{
        fontSize: "14px",
        fontWeight: "600",
        color: "var(--color-text-primary)",
        fontFamily: "var(--font-heading)",
        letterSpacing: "0.2px"
      }}>
        Inspection Target Category
      </label>
      <select
        id="category-select"
        value={selectedCategory}
        onChange={(e) => onCategoryChanged(e.target.value)}
        style={{
          width: "100%",
          padding: "12px 16px",
          backgroundColor: "var(--color-surface)",
          color: "var(--color-text-primary)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-md)",
          fontSize: "15px",
          fontFamily: "var(--font-body)",
          cursor: "pointer",
          outline: "none",
          transition: "var(--transition-fast)",
          boxShadow: "var(--shadow-sm)"
        }}
        onFocus={(e) => e.target.style.borderColor = "var(--color-accent)"}
        onBlur={(e) => e.target.style.borderColor = "var(--color-border)"}
      >
        {categories.map((cat) => (
          <option key={cat} value={cat}>
            {cat.charAt(0).toUpperCase() + cat.slice(1)} (STFPM)
          </option>
        ))}
      </select>
    </div>
  );
}

export default CategorySelector;

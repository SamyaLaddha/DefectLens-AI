import React from 'react';
import './styles/tokens.css';

function App() {
  return (
    <div style={{
      backgroundColor: 'var(--color-bg)',
      color: 'var(--color-text-primary)',
      fontFamily: 'var(--font-body)',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    }}>
      <h1 style={{ fontFamily: 'var(--font-heading)', color: 'var(--color-accent)' }}>DefectLens AI</h1>
      <p style={{ color: 'var(--color-text-secondary)' }}>
        Industrial Visual Anomaly Detection System Scaffold
      </p>
    </div>
  );
}

export default App;

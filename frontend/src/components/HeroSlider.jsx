import React, { useState } from 'react';

// Illustrative top-down "bottle mouth" with a chipped rim.
// Drawn in SVG so the landing page has no image dependency.
function Specimen({ heat }) {
  return (
    <svg viewBox="0 0 400 400" role="img" aria-hidden="true">
      <defs>
        <radialGradient id="hl-core" cx="291" cy="117" r="78" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#ff5a1f" stopOpacity="0.95" />
          <stop offset="0.45" stopColor="#ff9a1f" stopOpacity="0.7" />
          <stop offset="1" stopColor="#ffd23f" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="hl-faint" cx="140" cy="300" r="60" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#ff9a1f" stopOpacity="0.28" />
          <stop offset="1" stopColor="#ff9a1f" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width="400" height="400" fill="#d6cfbf" />
      <circle cx="200" cy="200" r="158" fill="#bdb5a3" stroke="#8d8571" strokeWidth="2" />
      <circle cx="200" cy="200" r="132" fill="#a9a18e" stroke="#8d8571" strokeWidth="2" />
      <circle cx="200" cy="200" r="105" fill="#d6cfbf" stroke="#7a7360" strokeWidth="3" />
      <circle cx="200" cy="200" r="88" fill="#34322c" />
      <path d="M130 150 A82 82 0 0 1 200 112" fill="none" stroke="#faf7f0" strokeOpacity="0.45" strokeWidth="6" strokeLinecap="round" />
      <path d="M150 232 A62 62 0 0 0 200 262" fill="none" stroke="#faf7f0" strokeOpacity="0.12" strokeWidth="4" strokeLinecap="round" />
      {/* the defect: a chip in the rim */}
      <path d="M277 110 l14 -7 l13 11 l-3 15 l-14 9 l-12 -9 z" fill="#34322c" />
      {heat && (
        <g>
          <rect width="400" height="400" fill="url(#hl-faint)" />
          <rect width="400" height="400" fill="url(#hl-core)" />
        </g>
      )}
    </svg>
  );
}

function HeroSlider() {
  const [pos, setPos] = useState(52);

  return (
    <figure className="specimen" style={{ margin: 0 }}>
      <div className="specimen__bar">
        <span>Specimen 0417</span>
        <span>Category: bottle</span>
      </div>
      <div className="specimen__stage">
        <Specimen heat={false} />
        <div className="specimen__top" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
          <Specimen heat />
        </div>
        <span className="specimen__chip specimen__chip--l">INPUT</span>
        <span className="specimen__chip specimen__chip--r">HEATMAP</span>
        <div className="specimen__line" style={{ left: `${pos}%` }} />
        <div className="specimen__knob" style={{ left: `${pos}%` }}>↔</div>
        <input
          className="specimen__range"
          type="range"
          min="0"
          max="100"
          value={pos}
          onChange={(e) => setPos(Number(e.target.value))}
          aria-label="Compare input image with anomaly heatmap"
        />
      </div>
      <figcaption className="specimen__cap">
        Illustration only. Run your own image to get a real heatmap.
      </figcaption>
    </figure>
  );
}

export default HeroSlider;

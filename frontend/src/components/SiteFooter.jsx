import React from 'react';
import { GITHUB_URL } from './SiteNav';

function SiteFooter() {
  return (
    <footer className="footer">
      <div className="wrap footer__row">
        <span>
          DefectLens AI · a portfolio project by Samya Laddha. Built with PyTorch, FastAPI and React.
        </span>
        <span>
          Benchmark images: MVTec AD (CC BY-NC-SA 4.0) ·{' '}
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">Source on GitHub</a>
        </span>
      </div>
    </footer>
  );
}

export default SiteFooter;

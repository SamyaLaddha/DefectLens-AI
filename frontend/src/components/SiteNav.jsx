import React from 'react';
import { Link, useLocation } from 'react-router-dom';

export const GITHUB_URL = 'https://github.com/SamyaLaddha/DefectLens-AI';

function SiteNav() {
  const { pathname } = useLocation();
  const onTry = pathname === '/try';

  return (
    <nav className="nav" aria-label="Main">
      <div className="wrap nav__row">
        <Link to="/" className="brand" aria-label="DefectLens AI home">
          <span className="brand__name">DefectLens</span>
          <span className="brand__tag">AI</span>
        </Link>
        <div className="nav__links">
          {onTry ? (
            <Link to="/" className="nav__link">Overview</Link>
          ) : (
            <>
              <a href="/#how" className="nav__link">How it works</a>
              <a href="/#capabilities" className="nav__link">Capabilities</a>
            </>
          )}
          <a href={GITHUB_URL} className="nav__link" target="_blank" rel="noreferrer">GitHub</a>
          {!onTry && (
            <Link to="/try" className="btn btn--sm">Try it →</Link>
          )}
        </div>
      </div>
    </nav>
  );
}

export default SiteNav;

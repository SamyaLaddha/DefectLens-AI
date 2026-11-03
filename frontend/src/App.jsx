import React, { useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import SiteNav from './components/SiteNav';
import SiteFooter from './components/SiteFooter';
import Landing from './pages/Landing';
import TryPage from './pages/TryPage';

function ScrollTop() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (!hash) window.scrollTo(0, 0);
  }, [pathname, hash]);
  return null;
}

function App() {
  return (
    <>
      <ScrollTop />
      <SiteNav />
      <main>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/try" element={<TryPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <SiteFooter />
    </>
  );
}

export default App;

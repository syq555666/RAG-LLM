import React, { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { ChatContainer } from './components/chat/ChatContainer';
import { Toaster } from 'react-hot-toast';
import './App.css';

type Theme = 'light' | 'dark';

function getInitialTheme(): Theme {
  const stored = localStorage.getItem('theme');
  if (stored === 'dark' || stored === 'light') return stored;
  if (window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
  return 'light';
}

const App: React.FC = () => {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  }, []);

  return (
    <div className="app">
      <Toaster
        position="top-center"
        toastOptions={{
          duration: 3000,
          style: { borderRadius: '8px', fontSize: '14px' },
        }}
      />
      <Sidebar />
      <main className="main-content">
        <header className="top-bar">
          <h1>🤖 智能客服</h1>
          <div className="top-bar-actions">
            <button className="btn-theme" onClick={toggleTheme} title={theme === 'light' ? '切换暗黑模式' : '切换白天模式'}>
              {theme === 'light' ? '🌙' : '☀️'}
            </button>
          </div>
        </header>
        <ChatContainer />
      </main>
    </div>
  );
};

export default App;

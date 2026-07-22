import React, { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { ChatContainer } from './components/chat/ChatContainer';
import { Toaster } from 'react-hot-toast';
import './App.css';

const THEME_KEY = 'rag_theme';

function getInitialTheme(): 'dark' | 'light' {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === 'dark' || stored === 'light') return stored;
  // 首次访问跟随系统
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  const themeIcon = theme === 'dark' ? '🌙' : '☀️';

  return (
    <div className="app">
      <Toaster
        position="top-center"
        toastOptions={{
          duration: 3000,
          style: { borderRadius: '8px', fontSize: '14px' },
        }}
      />

      {/* 移动端遮罩 */}
      <div
        className={`sidebar-overlay ${sidebarOpen ? 'visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <main className="main-content">
        <header className="top-bar">
          <button
            className="btn-hamburger"
            onClick={() => setSidebarOpen((v) => !v)}
            aria-label="切换侧边栏"
          >
            ☰
          </button>
          <h1>🤖 智能客服</h1>
          <div className="top-bar-actions">
            <button
              className="btn-theme"
              onClick={toggleTheme}
              title={theme === 'dark' ? '切换亮色模式' : '切换深色模式'}
              aria-label="切换主题"
            >
              {themeIcon}
            </button>
          </div>
        </header>
        <ChatContainer />
      </main>
    </div>
  );
};

export default App;

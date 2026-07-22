import React from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { ChatContainer } from './components/chat/ChatContainer';
import { Toaster } from 'react-hot-toast';
import './App.css';

const App: React.FC = () => {
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
        </header>
        <ChatContainer />
      </main>
    </div>
  );
};

export default App;

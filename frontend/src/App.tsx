import { NavLink, Navigate, Route, Routes } from 'react-router'

import { ChatPage } from './chat/ChatPage'
import { DataPage } from './data/DataPage'

export default function App() {
  return (
    <div className="app">
      <nav className="nav">
        <span className="brand">AaaS iThome</span>
        <NavLink to="/chat">Chat</NavLink>
        <NavLink to="/data">Data</NavLink>
      </nav>
      <main className="main">
        <Routes>
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </main>
    </div>
  )
}

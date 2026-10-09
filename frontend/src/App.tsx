import { NavLink, Navigate, Route, Routes } from 'react-router'

import { AnalysesPage } from './analyses/AnalysesPage'
import { ChatLayout } from './chat/ChatLayout'
import { getAccount } from './api/account'
import { DataPage } from './data/DataPage'
import { EvalsPage } from './evals/EvalsPage'
import { LoadPage } from './load/LoadPage'

export default function App() {
  return (
    <div className="app">
      <nav className="nav">
        <span className="brand">AaaS iThome</span>
        <NavLink to="/chat">Chat</NavLink>
        <NavLink to="/data">Data</NavLink>
        <NavLink to="/analyses">Analyses</NavLink>
        <NavLink to="/evals">Evals</NavLink>
        <NavLink to="/load">Load</NavLink>
        <span className="account" title="No login yet: requests are sent as this account">
          {getAccount()}
        </span>
      </nav>
      <main className="main">
        <Routes>
          <Route path="/chat/*" element={<ChatLayout />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/analyses/*" element={<AnalysesPage />} />
          <Route path="/evals/*" element={<EvalsPage />} />
          <Route path="/load" element={<LoadPage />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </main>
    </div>
  )
}

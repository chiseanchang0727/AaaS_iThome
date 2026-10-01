import type { ReactNode } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router'

import { ContextEvalPage } from './ContextEvalPage'

/** One sub-page per thing being evaluated. Add new ones here. */
const PAGES: { path: string; label: string; element: ReactNode }[] = [
  { path: 'context', label: 'Context (Jev)', element: <ContextEvalPage /> },
]

export function EvalsPage() {
  return (
    <div className="evals">
      <nav className="subnav" aria-label="Evaluations">
        {PAGES.map((page) => (
          <NavLink key={page.path} to={`/evals/${page.path}`}>
            {page.label}
          </NavLink>
        ))}
      </nav>
      <Routes>
        {PAGES.map((page) => (
          <Route key={page.path} path={page.path} element={page.element} />
        ))}
        <Route path="*" element={<Navigate to={`/evals/${PAGES[0].path}`} replace />} />
      </Routes>
    </div>
  )
}

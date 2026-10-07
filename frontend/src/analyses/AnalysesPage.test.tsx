import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { Analysis, AnalysisRun, AnalysisSummary, SourceOption } from '../api/types'
import { jsonResponse, mockFetch, type Call } from '../test/fetch'
import { AnalysesPage } from './AnalysesPage'

const SAVED: AnalysisRun = {
  id: '20261006-100000-aaaa', started_at: '2026-10-06T10:00:00Z', status: 'done', trigger: 'save',
  seconds: 9, outputs: ['chart.html'], error: null, log: 'drew it', notes: [], sources: {}, version: 1,
}
const FAILED: AnalysisRun = {
  ...SAVED, id: '20261006-110000-bbbb', started_at: '2026-10-06T11:00:00Z', status: 'failed', trigger: 'run',
  outputs: [], error: 'the script failed (exit code 1)', log: 'KeyError: views',
}

const ANALYSIS: Analysis = {
  id: 'ab12cd34ef56', title: 'Videos per month', description: 'Distinct trending videos each month.',
  question: 'Chart distinct videos per month', conversation: 't1', created_at: '2026-10-06T10:00:00Z',
  version: 1, updated_at: null,
  inputs: [{ kind: 'query', sql: 'SELECT month, n FROM monthly', file: 'monthly.parquet' }, { kind: 'dataset', name: 'sales' }],
  script: 'import os\n# reads monthly.parquet', outputs: [{ file: 'chart.html', format: 'html' }],
  sources: ['monthly'], runs: [SAVED], running: false,
}

const SUMMARY: AnalysisSummary = {
  id: ANALYSIS.id, title: ANALYSIS.title, description: ANALYSIS.description, question: ANALYSIS.question,
  created_at: ANALYSIS.created_at, version: 1, updated_at: null, outputs: ['chart.html'], last_run: SAVED, last_good_run: SAVED,
}

const OPTIONS: SourceOption[] = [{
  table: 'monthly',
  candidates: [
    { name: 'monthly_ca', ok: true, problems: [] },
    { name: 'sales', ok: false, problems: ['no column month'] },
  ],
}]

afterEach(() => vi.unstubAllGlobals())

function backend(analysis: Analysis = ANALYSIS) {
  let current = analysis
  return (call: Call) => {
    if (call.url === '/api/analyses') return jsonResponse([SUMMARY])
    if (call.url === `/api/analyses/${analysis.id}/sources`) return jsonResponse(OPTIONS)
    if (call.url === `/api/analyses/${analysis.id}/runs` && call.method === 'POST') {
      const sources = (call.body as { sources: Record<string, string> }).sources
      const started: AnalysisRun = { ...SAVED, id: '20261006-120000-cccc', status: 'running', trigger: 'run', outputs: [], seconds: null, sources }
      current = { ...current, runs: [started, ...current.runs], running: true }
      return jsonResponse(started, 202)
    }
    if (call.url === `/api/analyses/${analysis.id}` && call.method === 'DELETE') return jsonResponse({ deleted: analysis.id })
    if (call.url === `/api/analyses/${analysis.id}`) return jsonResponse(current)
    return jsonResponse({ detail: 'unexpected' }, 500)
  }
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/analyses/*" element={<AnalysesPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('Analyses', () => {
  it('lists saved analyses', async () => {
    mockFetch(backend())
    renderAt('/analyses')
    const list = await screen.findByRole('list', { name: 'Saved analyses' })
    expect(within(list).getByRole('link', { name: /Videos per month/ })).toHaveAttribute('href', '/analyses/ab12cd34ef56')
  })

  it('deletes from the list after asking', async () => {
    const calls = mockFetch(backend())
    renderAt('/analyses')
    const list = await screen.findByRole('list', { name: 'Saved analyses' })
    await userEvent.click(within(list).getByRole('button', { name: 'Delete Videos per month…' }))
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false)
    const confirm = screen.getByRole('group', { name: 'Confirm deleting Videos per month' })
    await userEvent.click(within(confirm).getByRole('button', { name: 'Delete' }))
    expect(calls.filter((c) => c.method === 'DELETE').map((c) => c.url)).toEqual(['/api/analyses/ab12cd34ef56'])
    expect(await screen.findByRole('heading', { name: 'No saved analyses yet' })).toBeInTheDocument()
  })

  it('can cancel a delete from the list', async () => {
    const calls = mockFetch(backend())
    renderAt('/analyses')
    await userEvent.click(await screen.findByRole('button', { name: 'Delete Videos per month…' }))
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false)
    expect(screen.getByRole('link', { name: /Videos per month/ })).toBeInTheDocument()
  })

  it('says how to save one when there are none', async () => {
    mockFetch(() => jsonResponse([]))
    renderAt('/analyses')
    expect(await screen.findByRole('heading', { name: 'No saved analyses yet' })).toBeInTheDocument()
  })

  it('shows the latest output in a sandboxed frame, the runs and the recipe', async () => {
    mockFetch(backend())
    renderAt('/analyses/ab12cd34ef56')
    const frame = await screen.findByTitle('chart.html')
    expect(frame).toHaveAttribute('src', '/api/analyses/ab12cd34ef56/runs/20261006-100000-aaaa/files/chart.html')
    expect(frame.getAttribute('sandbox')).toBe('allow-scripts')
    expect(screen.getByRole('table', { name: 'Runs' })).toHaveTextContent('test run when saved')
    const recipe = screen.getByRole('region', { name: 'Recipe' })
    expect(within(recipe).getByText('SELECT month, n FROM monthly')).toBeInTheDocument()
    expect(recipe).toHaveTextContent('Uploaded file sales → $DATA_DIR/sales.parquet')
  })

  it('runs it again and shows that it is running', async () => {
    const calls = mockFetch(backend())
    renderAt('/analyses/ab12cd34ef56')
    await userEvent.click(await screen.findByRole('button', { name: 'Run' }))
    const posts = calls.filter((c) => c.method === 'POST')
    expect(posts.map((c) => [c.url, c.body])).toEqual([['/api/analyses/ab12cd34ef56/runs', { sources: {} }]])
    expect(await screen.findByRole('button', { name: 'Running…' })).toBeDisabled()
    expect(screen.getByRole('status')).toHaveTextContent('Running…')
  })

  it('shows why a run failed, with its output', async () => {
    mockFetch(backend({ ...ANALYSIS, runs: [FAILED, SAVED] }))
    renderAt('/analyses/ab12cd34ef56')
    await userEvent.click(await screen.findByRole('cell', { name: /failed: the script failed/ }))
    expect(screen.getByText('The run failed: the script failed (exit code 1)')).toBeInTheDocument()
    expect(screen.getByText('KeyError: views')).toBeInTheDocument()
  })

  it('asks before deleting', async () => {
    const calls = mockFetch(backend())
    renderAt('/analyses/ab12cd34ef56')
    await userEvent.click(await screen.findByRole('button', { name: 'Delete…' }))
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false)
    await userEvent.click(within(screen.getByRole('group', { name: 'Confirm delete' })).getByRole('button', { name: 'Delete' }))
    expect(calls.filter((c) => c.method === 'DELETE').map((c) => c.url)).toEqual(['/api/analyses/ab12cd34ef56'])
  })

  it('runs on another table with the same columns', async () => {
    const calls = mockFetch(backend())
    renderAt('/analyses/ab12cd34ef56')
    const picker = await screen.findByRole('combobox', { name: 'Data: monthly' })
    expect(within(picker).getByRole('option', { name: 'sales: no column month' })).toBeDisabled()
    await userEvent.selectOptions(picker, 'monthly_ca')
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))

    const [post] = calls.filter((c) => c.method === 'POST')
    expect(post.body).toEqual({ sources: { monthly: 'monthly_ca' } })
    expect(await screen.findByRole('table', { name: 'Runs' })).toHaveTextContent('run on monthly_ca')
  })

  it('shows which version it is and which version each run used', async () => {
    const v2: AnalysisRun = { ...SAVED, id: '20261006-130000-dddd', version: 2 }
    mockFetch(backend({ ...ANALYSIS, version: 2, updated_at: '2026-10-06T13:00:00Z', runs: [v2, SAVED] }))
    renderAt('/analyses/ab12cd34ef56')
    expect(await screen.findByText(/version 2, changed/)).toBeInTheDocument()
    const runs = screen.getByRole('table', { name: 'Runs' })
    expect(within(runs).getAllByRole('row').slice(1).map((r) => (r as HTMLTableRowElement).cells[1].textContent)).toEqual([
      'test run when saved · v2', 'test run when saved · v1',
    ])
  })
})

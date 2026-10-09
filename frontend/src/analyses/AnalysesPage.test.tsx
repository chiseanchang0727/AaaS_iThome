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
  measurements: { inputs: [{ file: 'monthly.parquet', tables: ['monthly'], estimated_rows: 30, estimated_bytes: 900, rows: 30, bytes: 2000, query_seconds: 0.2 }],
    sandbox: { seconds: 3.4, cpu_seconds: 2, peak_memory_mb: 120, memory_limit_mb: 1024, killed: false } },
  findings: [], optimization: null, optimized_from: null,
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
  sources: ['monthly'], optimization: null, runs: [SAVED], running: false, optimizing: false,
  limits: { hard_export_rows: 100_000, max_export_bytes: 200_000_000, soft_export_rows: 50_000, soft_export_bytes: 20_000_000,
    soft_run_seconds: 120, memory_warning_ratio: 0.8, hard_query_seconds: 10 },
}

const SUMMARY: AnalysisSummary = {
  id: ANALYSIS.id, title: ANALYSIS.title, description: ANALYSIS.description, question: ANALYSIS.question,
  created_at: ANALYSIS.created_at, version: 1, updated_at: null, outputs: ['chart.html'], last_run: SAVED, last_good_run: SAVED,
}

const STOPPED: AnalysisRun = {
  ...SAVED, id: '20261007-100000-eeee', started_at: '2026-10-07T10:00:00Z', status: 'needs_optimization', trigger: 'run',
  outputs: [], seconds: 0.1, sources: { monthly: 'monthly_big' }, log: '',
  error: 'monthly.parquet (from monthly_big) would export about 10.0M rows into the sandbox; the limit is 100,000',
  measurements: { inputs: [{ file: 'monthly.parquet', tables: ['monthly_big'], estimated_rows: 10_048_594, estimated_bytes: 422_000_000, rows: null, bytes: null, query_seconds: null }], sandbox: null },
  findings: [{ limit: 'hard', kind: 'data_movement', reason: 'monthly.parquet (from monthly_big) would export about 10.0M rows into the sandbox; the limit is 100,000' }],
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
    if (call.url === `/api/analyses/${analysis.id}/versions`) {
      const { runs: _r, running: _x, optimizing: _o, limits: _l, ...recipe } = current
      return jsonResponse(current.version > 1
        ? [{ ...recipe, version: 1, optimization: null, inputs: [{ kind: 'query', sql: 'SELECT * FROM monthly', file: 'monthly.parquet' }] }, recipe]
        : [recipe])
    }
    if (call.url === `/api/analyses/${analysis.id}/sources`) return jsonResponse(OPTIONS)
    if (call.url === `/api/analyses/${analysis.id}/runs` && call.method === 'POST') {
      const sources = (call.body as { sources: Record<string, string> }).sources
      const started: AnalysisRun = { ...SAVED, id: '20261006-120000-cccc', status: 'running', trigger: 'run', outputs: [], seconds: null, sources }
      current = { ...current, runs: [started, ...current.runs], running: true }
      return jsonResponse(started, 202)
    }
    if (call.url === `/api/analyses/${analysis.id}` && call.method === 'DELETE') return jsonResponse({ deleted: analysis.id })
    if (call.url.includes('/optimizations/')) {
      return jsonResponse([
        { turn: 1, role: 'user', content: 'Optimize the saved analysis ab12cd34ef56, **too big**.' },
        { turn: 1, role: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'save_analysis', args: { replaces: 'ab12cd34ef56' } }] },
        { turn: 1, role: 'tool', content: 'Updated analysis ab12cd34ef56 to version 2 (optimized).', tool_call_id: 'c1', name: 'save_analysis', error: false },
        { turn: 1, role: 'assistant', content: 'Moved sessions into SQL.' },
      ])
    }
    if (call.url.endsWith('/optimize') && call.method === 'POST') {
      current = { ...current, running: true, optimizing: true,
        runs: current.runs.map((r) => (r.id === STOPPED.id ? { ...r, optimization: { status: 'running' as const, conversation: 'optimize-x' } } : r)) }
      return jsonResponse({ ...STOPPED, optimization: { status: 'running', conversation: 'optimize-x' } }, 202)
    }
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

  it('says why a run stopped and asks before optimizing', async () => {
    const calls = mockFetch(backend({ ...ANALYSIS, runs: [STOPPED, SAVED] }))
    renderAt('/analyses/ab12cd34ef56')
    await userEvent.click(await screen.findByRole('cell', { name: /needs optimization/ }))
    expect(screen.getByText(/The saved strategy no longer fits monthly_big/)).toBeInTheDocument()
    expect(screen.getByText(/would export about 10.0M rows into the sandbox/, { selector: 'li' })).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Runs' })).toHaveTextContent('~10.0M rows (estimate)')
    expect(calls.some((c) => c.url.endsWith('/optimize'))).toBe(false)  // nothing happens on its own

    await userEvent.click(screen.getByRole('button', { name: 'Optimize' }))
    expect(calls.filter((c) => c.method === 'POST').map((c) => c.url)).toEqual([
      `/api/analyses/ab12cd34ef56/runs/${STOPPED.id}/optimize`,
    ])
    expect(await screen.findByText(/Optimizing… the agent is rewriting the recipe/)).toBeInTheDocument()
  })

  it('shows the new version and how much less data it moves', async () => {
    const after: AnalysisRun = {
      ...SAVED, id: '20261007-100500-ffff', started_at: '2026-10-07T10:05:00Z', trigger: 'run', version: 2,
      optimized_from: 1, sources: { monthly: 'monthly_big' },
      measurements: { inputs: [{ file: 'monthly.parquet', tables: ['monthly_big'], estimated_rows: 3, estimated_bytes: 90, rows: 3, bytes: 1500, query_seconds: 6.1 }], sandbox: null },
    }
    const done = { ...STOPPED, optimization: { status: 'done' as const, conversation: 'optimize-x', new_version: 2, message: 'Moved sessions into SQL.' } }
    const calls = mockFetch(backend({ ...ANALYSIS, version: 2, updated_at: '2026-10-07T10:04:00Z', runs: [after, done, SAVED],
      optimization: { from_version: 1, kind: 'data_movement', reason: 'too big', sources: { monthly: 'monthly_big' }, run: done.id, conversation: 'optimize-x' } }))
    renderAt('/analyses/ab12cd34ef56')
    expect(await screen.findByText(/optimized from version 1 for monthly_big/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open the chat it was saved in' })).toHaveAttribute('href', '/chat/past/t1')

    // the optimization is part of the analysis: its request and what the agent did, on this page
    const section = screen.getByRole('region', { name: 'Optimizations' })
    expect(section).toHaveTextContent('Version 1 → 2')
    await userEvent.click(screen.getByRole('button', { name: 'How it was optimized' }))
    expect(await within(section).findByText('The request the agent got')).toBeInTheDocument()
    expect(within(section).getByText('too big', { selector: 'strong' })).toBeInTheDocument()
    expect(within(section).getByText('1 step')).toBeInTheDocument()
    expect(calls.map((c) => c.url)).toContain('/api/analyses/ab12cd34ef56/optimizations/optimize-x')
    await userEvent.click(screen.getByRole('cell', { name: /needs optimization → version 2/ }))
    const panel = screen.getByText(/Analysis updated to/).closest('div')!
    expect(panel).toHaveTextContent('New strategy: 3 rows · 2 KB into the sandbox, instead of ~10.0M rows (estimate).')
    expect(panel).toHaveTextContent('Moved sessions into SQL.')
    expect(within(panel).getByRole('button', { name: 'See what the agent did' })).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Runs' })).toHaveTextContent('after optimization')
  })

  it('says when an optimization failed and the version stayed', async () => {
    const failed = { ...STOPPED, optimization: { status: 'failed' as const, conversation: 'optimize-x', message: 'NOT SAVED. results changed' } }
    mockFetch(backend({ ...ANALYSIS, runs: [failed, SAVED] }))
    renderAt('/analyses/ab12cd34ef56')
    await userEvent.click(await screen.findByRole('cell', { name: /optimization failed/ }))
    expect(screen.getByText('Optimization failed; version 1 is unchanged.')).toBeInTheDocument()
    expect(screen.getByText('NOT SAVED. results changed')).toBeInTheDocument()
  })

  it('shows what each run moved and used', async () => {
    mockFetch(backend())
    renderAt('/analyses/ab12cd34ef56')
    const row = (await screen.findByRole('table', { name: 'Runs' })).querySelectorAll('tbody tr')[0]
    expect(row).toHaveTextContent('30 rows · 2 KB')
    expect(row).toHaveTextContent('0.2s')
    expect(row).toHaveTextContent('3.4s')
    expect(row).toHaveTextContent('120 MB')
  })

  it('shows the query before an optimization, and both versions side by side', async () => {
    mockFetch(backend({ ...ANALYSIS, version: 2, updated_at: '2026-10-07T10:04:00Z',
      inputs: [{ kind: 'query', sql: 'SELECT month, count(*) FROM monthly GROUP BY 1', file: 'monthly.parquet' }],
      optimization: { from_version: 1, kind: 'data_movement', reason: 'too many rows', sources: {}, run: null, conversation: null, results_check: 'Its results match version 1 on the saved tables.' } }))
    renderAt('/analyses/ab12cd34ef56')
    const recipe = await screen.findByRole('region', { name: 'Recipe' })
    expect(within(recipe).getByText('SELECT month, count(*) FROM monthly GROUP BY 1')).toBeInTheDocument()
    expect(within(recipe).getByText(/made by an optimization of version 1: too many rows\. Its results match version 1/)).toBeInTheDocument()

    await userEvent.selectOptions(await within(recipe).findByRole('combobox', { name: 'Show' }), '1')
    expect(within(recipe).getByText('SELECT * FROM monthly')).toBeInTheDocument()

    await userEvent.selectOptions(within(recipe).getByRole('combobox', { name: 'Show' }), '2')
    await userEvent.click(within(recipe).getByRole('checkbox', { name: 'Compare with version 1' }))
    expect(within(recipe).getByText('Version 1 (before)')).toBeInTheDocument()
    expect(within(recipe).getByText('SELECT * FROM monthly')).toBeInTheDocument()
    expect(within(recipe).getByText('SELECT month, count(*) FROM monthly GROUP BY 1')).toBeInTheDocument()
  })

  it('shows what the check measured, against which limit, and what that leads to', async () => {
    mockFetch(backend({ ...ANALYSIS, runs: [STOPPED, SAVED] }))
    renderAt('/analyses/ab12cd34ef56')
    await userEvent.click(await screen.findByRole('cell', { name: /needs optimization/ }))
    const check = screen.getByRole('table', { name: 'Check' })
    const estimate = within(check).getByRole('row', { name: /Estimated rows into the sandbox \(EXPLAIN\)/ })
    expect(estimate).toHaveTextContent('~10.0M')
    expect(estimate).toHaveTextContent('100,000 (hard)')
    expect(estimate).toHaveTextContent('✗ over the hard limit')
    expect(within(check).getByRole('row', { name: /Rows exported/ })).toHaveTextContent('— not measured')
    expect(screen.getByText(/Stopped before exporting anything: over a hard limit\. Problem: the step from the database into the sandbox costs too much → Optimize rewrites the SQL/)).toBeInTheDocument()
  })

  it('says a run within every limit needs no optimization', async () => {
    mockFetch(backend())
    renderAt('/analyses/ab12cd34ef56')
    expect(await screen.findByText('Within every limit: no optimization needed.')).toBeInTheDocument()
    const check = screen.getByRole('table', { name: 'Check' })
    expect(within(check).getByRole('row', { name: /Peak memory/ })).toHaveTextContent('120 MB of 1,024 MB')
    expect(within(check).getByRole('row', { name: /Query time/ })).toHaveTextContent('10s (hard)')
  })

  it('opens on the latest run, so a run that needs optimization shows its check first', async () => {
    mockFetch(backend({ ...ANALYSIS, runs: [STOPPED, SAVED] }))
    renderAt('/analyses/ab12cd34ef56')
    expect(await screen.findByText(/The saved strategy no longer fits monthly_big/)).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Check' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Optimize' })).toBeInTheDocument()
  })

  it('shows the real count when the estimate was over and the rows were counted', async () => {
    const counted: AnalysisRun = { ...SAVED, id: '20261008-100000-gggg', started_at: '2026-10-08T10:00:00Z', trigger: 'run',
      measurements: { inputs: [{ file: 'monthly.parquet', tables: ['monthly_big'], estimated_rows: 4_175_005, estimated_bytes: 79_325_095,
        counted_rows: 120, rows: 120, bytes: 1_600, query_seconds: 3.1 }], sandbox: null } }
    mockFetch(backend({ ...ANALYSIS, runs: [counted, SAVED] }))
    renderAt('/analyses/ab12cd34ef56')
    const check = await screen.findByRole('table', { name: 'Check' })
    expect(within(check).getByRole('row', { name: /Estimated rows/ })).toHaveTextContent('over, so the rows were counted ↓')
    expect(within(check).getByRole('row', { name: /Estimated rows/ })).not.toHaveTextContent('✗')
    const real = within(check).getByRole('row', { name: /counted in PostgreSQL/ })
    expect(real).toHaveTextContent('120')
    expect(real).toHaveTextContent('✓ OK')
  })

  it('can try a failed optimization again, and keeps every attempt', async () => {
    const tried = { ...STOPPED, optimization: { status: 'failed' as const, conversation: 'optimize-x-2', message: 'second try failed',
      earlier: [{ status: 'failed' as const, conversation: 'optimize-x', message: 'stopped after 40 steps' }] } }
    const calls = mockFetch(backend({ ...ANALYSIS, runs: [tried, SAVED] }))
    renderAt('/analyses/ab12cd34ef56')
    const section = await screen.findByRole('region', { name: 'Optimizations' })
    expect(section).toHaveTextContent('Version 1: optimization failed (attempt 1)')
    expect(section).toHaveTextContent('Version 1: optimization failed (attempt 2)')
    await userEvent.click(screen.getByRole('button', { name: 'Optimize again' }))
    expect(calls.filter((c) => c.method === 'POST').map((c) => c.url)).toEqual([`/api/analyses/ab12cd34ef56/runs/${STOPPED.id}/optimize`])
  })
})

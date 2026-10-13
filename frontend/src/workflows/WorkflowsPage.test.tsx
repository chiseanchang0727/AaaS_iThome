import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { AnalysisSummary, SourceOption, Workflow, WorkflowRun, WorkflowSummary } from '../api/types'
import { jsonResponse, mockFetch, type Call } from '../test/fetch'
import { WorkflowsPage } from './WorkflowsPage'

const ANALYSIS: AnalysisSummary = {
  id: 'sessions1', title: 'Median session length', description: '', question: '', created_at: '2026-10-06T10:00:00Z',
  version: 2, updated_at: null, outputs: ['sessions.html'], last_run: null, last_good_run: null,
}

const RUN: WorkflowRun = {
  id: '20261011-100000000-aaaa', workflow_id: 'wf1', started_at: '2026-10-11T10:00:00Z', finished_at: '2026-10-11T10:00:20Z',
  seconds: 20, status: 'partly_done',
  nodes: [
    { node_id: 'small', analysis_id: 'sessions1', title: 'Median session length', sources: {}, version: 2,
      run_id: '20261011-100000001-bbbb', status: 'done', error: null, seconds: 12, outputs: ['sessions.html'] },
    { node_id: 'big', analysis_id: 'sessions1', title: 'Median session length', sources: { events: 'events_big' }, version: 2,
      run_id: '20261011-100000002-cccc', status: 'needs_optimization', error: 'would export about 10M rows; the limit is 100,000',
      seconds: 0.1, outputs: [] },
  ],
  batches: { batch: ['small', 'big'] },
  reports: {},
  titles: { batch: 'Both tables' },
  workflow_title: 'Sessions on both tables',
}

const WORKFLOW: Workflow = {
  id: 'wf1', title: 'Sessions on both tables', created_at: '2026-10-11T09:00:00Z', updated_at: null,
  nodes: [
    { kind: 'analysis', id: 'small', analysis_id: 'sessions1', sources: {}, position: { x: 0, y: 0 } },
    { kind: 'analysis', id: 'big', analysis_id: 'sessions1', sources: { events: 'events_big' }, position: { x: 0, y: 140 } },
    { kind: 'batch', id: 'batch', title: 'Both tables', position: { x: 400, y: 70 } },
  ],
  edges: [{ source: 'small', target: 'batch' }, { source: 'big', target: 'batch' }],
  analyses: { sessions1: { title: 'Median session length', version: 2, tables: ['events'] } },
  running: false,
  runs: [RUN],
}

const SUMMARY: WorkflowSummary = {
  id: 'wf1', title: WORKFLOW.title, created_at: WORKFLOW.created_at, updated_at: null, analyses: 2, batches: 1, reports: 0, last_run: RUN,
}

/** The same, with a report node after the batch, run once. */
const WITH_REPORT: Workflow = {
  ...WORKFLOW,
  id: 'wf3',
  nodes: [...WORKFLOW.nodes, { kind: 'report', id: 'report', title: 'Sessions report', position: { x: 800, y: 70 } }],
  edges: [...WORKFLOW.edges, { source: 'batch', target: 'report' }],
  runs: [{ ...RUN, workflow_id: 'wf3', reports: { report: ['batch'] }, titles: { batch: 'Both tables', report: 'Sessions report' } }],
}

const SOURCES: SourceOption[] = [
  { table: 'events', candidates: [{ name: 'events_big', ok: true, problems: [] }, { name: 'videos', ok: false, problems: ['no column session_id'] }] },
]

function backend() {
  return (call: Call) => {
    if (call.url === '/api/workflows' && call.method === 'GET') return jsonResponse([SUMMARY])
    if (call.url === '/api/workflows' && call.method === 'POST') {
      return jsonResponse({ ...WORKFLOW, id: 'wf2', ...(call.body as object), runs: [] }, 201)
    }
    if (call.url === '/api/workflows/wf1/runs') return jsonResponse({ ...RUN, id: '20261011-110000000-dddd', status: 'running' }, 202)
    if (call.url === '/api/workflows/wf1') return jsonResponse(WORKFLOW)
    if (call.url === '/api/workflows/wf3') return jsonResponse(WITH_REPORT)
    if (call.url === '/api/analyses') return jsonResponse([ANALYSIS])
    if (call.url === '/api/analyses/sessions1/sources') return jsonResponse(SOURCES)
    return jsonResponse({ detail: `unexpected ${call.method} ${call.url}` }, 404)
  }
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/workflows/*" element={<WorkflowsPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('WorkflowsPage', () => {
  it('lists workflows with their last run', async () => {
    mockFetch(backend())
    renderAt('/workflows')
    const list = await screen.findByRole('list', { name: 'Workflows' })
    const link = within(list).getByRole('link', { name: /Sessions on both tables/ })
    expect(link).toHaveAttribute('href', '/workflows/wf1')
    expect(link).toHaveTextContent('2 analyses · 1 batch · 0 reports · partly done')
    expect(screen.getByRole('link', { name: '+ New workflow' })).toHaveAttribute('href', '/workflows/new')
  })

  it('shows the nodes on the canvas and the batch report of the last run', async () => {
    mockFetch(backend())
    renderAt('/workflows/wf1')
    expect(await screen.findByRole('textbox', { name: 'Workflow title' })).toHaveValue('Sessions on both tables')
    expect(screen.getByText('on events → events_big')).toBeInTheDocument()
    expect(screen.getByText('collects 2 analyses')).toBeInTheDocument()

    expect(screen.getByRole('heading', { name: 'Batch: Both tables' })).toBeInTheDocument()
    expect(screen.getByText(/1 of 2 done/)).toBeInTheDocument()
    expect(screen.getByText('would export about 10M rows; the limit is 100,000')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Optimize it on its page' }))
      .toHaveAttribute('href', '/analyses/sessions1?run=20261011-100000002-cccc')
    expect(screen.getByTitle('sessions.html'))
      .toHaveAttribute('src', '/api/analyses/sessions1/runs/20261011-100000001-bbbb/files/sessions.html')
  })

  it('runs a saved workflow', async () => {
    const calls = mockFetch(backend())
    renderAt('/workflows/wf1')
    await userEvent.click(await screen.findByRole('button', { name: 'Run' }))
    expect(calls.filter((c) => c.method === 'POST').map((c) => c.url)).toEqual(['/api/workflows/wf1/runs'])
  })

  it('builds a new workflow from saved analyses and saves it', async () => {
    const calls = mockFetch(backend())
    renderAt('/workflows/new')
    const palette = await screen.findByRole('complementary', { name: 'Saved analyses' })
    expect(screen.getByRole('button', { name: 'Save & run' })).toBeDisabled() // nothing connected yet
    await userEvent.click(await within(palette).findByRole('button', { name: 'Add Median session length' }))
    expect(screen.getByText('on its saved tables')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Save' }))
    const [saved] = calls.filter((c) => c.method === 'POST')
    const body = saved.body as { title: string; nodes: { kind: string; analysis_id?: string }[]; edges: unknown[] }
    expect(body.title).toBe('Untitled workflow')
    expect(body.nodes.map((n) => n.kind).sort()).toEqual(['analysis', 'batch'])
    expect(body.nodes.find((n) => n.kind === 'analysis')?.analysis_id).toBe('sessions1')
    expect(body.edges).toEqual([])
  })

  it('sets the tables a node runs on', async () => {
    mockFetch(backend())
    const { container } = renderAt('/workflows/wf1')
    await screen.findByRole('textbox', { name: 'Workflow title' })
    const node = container.querySelector('.react-flow__node[data-id="small"]') as HTMLElement
    fireEvent.click(node) // a click only: d3's drag code needs a real window on mouse down
    const inspector = screen.getByRole('complementary', { name: 'Node settings' })
    const select = await within(inspector).findByRole('combobox', { name: /Data: events/ })
    expect(within(select).getByRole('option', { name: 'videos: no column session_id' })).toBeDisabled()
    await userEvent.selectOptions(select, 'events_big')
    expect(screen.getAllByText('on events → events_big')).toHaveLength(2)
    expect(screen.getByText('unsaved changes')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Save & run' })).toBeEnabled()
  })

  it('shows a report node page below the canvas, and opens it on its own', async () => {
    mockFetch(backend())
    renderAt('/workflows/wf3')
    expect(await screen.findByRole('heading', { name: 'Report: Sessions report' })).toBeInTheDocument()
    const url = '/api/workflows/wf3/runs/20261011-100000000-aaaa/reports/report'
    expect(screen.getByTitle('Sessions report.html')).toHaveAttribute('src', url)
    expect(screen.getByText('1 batch on one page')).toBeInTheDocument()
    // On the node, which jsdom never measures, so it has no accessible name: found by its text.
    expect(screen.getByText('Open report ↗').closest('a')).toHaveAttribute('href', url)
  })

  it('shows the results and the load of a run apart, never on one page', async () => {
    mockFetch(backend())
    renderAt('/workflows/wf3')
    expect(await screen.findByRole('heading', { name: 'Report: Sessions report' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Results' })).toHaveAttribute('aria-selected', 'true')
    await userEvent.click(screen.getByRole('tab', { name: 'Load' }))
    expect(screen.getByRole('heading', { name: 'Load of this run' })).toBeInTheDocument()
    expect(screen.getByTitle('Load.html')).toHaveAttribute('src', '/api/workflows/wf3/runs/20261011-100000000-aaaa/load')
    expect(screen.queryByTitle('Sessions report.html')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('tab', { name: 'Results' }))
    expect(screen.getByTitle('Sessions report.html')).toBeInTheDocument()
  })
})

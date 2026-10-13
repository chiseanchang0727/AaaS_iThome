import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { Workflow, WorkflowRun } from '../api/types'
import { jsonResponse, mockFetch } from '../test/fetch'
import { savedWorkflows } from './savedWorkflows'
import type { Step } from './state'
import { WorkflowPreviews } from './WorkflowPreview'

const RUN: WorkflowRun = {
  id: '20261013-100000000-aaaa', workflow_id: 'wf1', started_at: '2026-10-13T10:00:00Z', finished_at: '2026-10-13T10:00:14Z',
  seconds: 14, status: 'done',
  nodes: [{ node_id: 'a1', analysis_id: 'sessions1', title: 'Median session length', sources: {}, version: 1,
    run_id: '20261013-100000001-bbbb', status: 'done', error: null, seconds: 3, outputs: ['s.html'],
    sandbox: 'workflow-x-1-1', sandbox_at: '2026-10-13T10:00:02Z' }],
  batches: { b1: ['a1'] }, reports: { r1: ['b1'] }, titles: { b1: 'Users', r1: 'Weekly report' }, workflow_title: 'Weekly',
}

const WORKFLOW: Workflow = {
  id: 'wf1', title: 'Weekly', created_at: '2026-10-13T09:00:00Z', updated_at: null, settings: { lanes: null },
  nodes: [
    { kind: 'analysis', id: 'a1', analysis_id: 'sessions1', sources: {}, priority: 0, position: { x: 0, y: 0 } },
    { kind: 'batch', id: 'b1', title: 'Users', position: { x: 420, y: 0 } },
    { kind: 'report', id: 'r1', title: 'Weekly report', position: { x: 840, y: 0 } },
  ],
  edges: [{ source: 'a1', target: 'b1' }, { source: 'b1', target: 'r1' }],
  analyses: { sessions1: { title: 'Median session length', version: 1, tables: ['events'] } },
  running: false,
  runs: [RUN],
}

function tool(name: string, result: string): Step {
  return { kind: 'tool', id: name + result.length, name, args: {}, result, error: false } as unknown as Step
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('workflows in the chat', () => {
  it('finds the workflows a turn saved, changed or ran, once each', () => {
    const steps = [
      tool('save_workflow', 'Saved workflow wf1: "Weekly": 1 analyses in 1 sections, one report.'),
      tool('run_workflow', 'Ran workflow wf1: done in 14 s (1 of 1 analyses done, 1 sandboxes).'),
      tool('save_workflow', 'NOT SAVED: give at least one section with at least one analysis.'),
      tool('save_analysis', 'Saved analysis abc123: "Something".'),
      tool('save_workflow', 'Updated workflow wf2: "Other": 2 analyses in 1 sections, one report.'),
    ]
    expect(savedWorkflows(steps)).toEqual([{ id: 'wf1', touched: 3, pending: false }, { id: 'wf2', touched: 1, pending: false }])
    // A run that started but has not answered yet counts too: the preview follows it.
    const running = { kind: 'tool', id: 'r', name: 'run_workflow', args: { workflow_id: 'wf2' }, result: null } as unknown as Step
    expect(savedWorkflows([...steps, running])).toEqual([{ id: 'wf1', touched: 3, pending: false }, { id: 'wf2', touched: 2, pending: true }])
    // When it ends, the same step counts twice: the preview reloads once more.
    const ended = { ...running, result: 'Ran workflow wf2: done in 9 s (2 of 2 analyses done, 2 sandboxes).' } as unknown as Step
    expect(savedWorkflows([...steps, ended])).toEqual([{ id: 'wf1', touched: 3, pending: false }, { id: 'wf2', touched: 3, pending: false }])
  })

  it('shows the canvas, how the run went, and the links', async () => {
    mockFetch(() => jsonResponse(WORKFLOW))
    render(
      <MemoryRouter>
        <WorkflowPreviews steps={[tool('run_workflow', 'Ran workflow wf1: done in 14 s (1 of 1 analyses done, 1 sandboxes).')]} past />
      </MemoryRouter>,
    )
    const preview = await screen.findByRole('figure', { name: 'Workflow Weekly' })
    expect(preview).toHaveTextContent('Median session length')  // the analysis node on the canvas
    expect(preview).toHaveTextContent('Users')
    expect(screen.getByLabelText('Run summary')).toHaveTextContent('Took 14 s · 1 analysis · 1 sandbox')
    expect(screen.getByRole('link', { name: 'Open workflow' })).toHaveAttribute('href', '/workflows/wf1')
    expect(screen.getByRole('link', { name: 'Open report ↗' }))
      .toHaveAttribute('href', '/api/workflows/wf1/runs/20261013-100000000-aaaa/reports/r1')
    expect(screen.getByText(/as the workflow is now/)).toBeInTheDocument()
  })
})

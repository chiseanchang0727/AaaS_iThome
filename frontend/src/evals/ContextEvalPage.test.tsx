import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { EvalRun, EvalTurnResult, HistoryLine } from '../api/types'
import { jsonResponse, mockFetch, type Call } from '../test/fetch'
import { EvalsPage } from './EvalsPage'
import { summarize, turnSteps } from './results'

function result(arm: string, turn: number, extra: Partial<EvalTurnResult> = {}): EvalTurnResult {
  return {
    conversation: 'chat', turn: `chat.${turn}`, arm, run: 1, prompt: `question ${turn}`,
    answer: `${arm} answer ${turn}`, stopped: false, sent_turns: arm === 'jev' ? [] : null,
    steps: 1, queries: 1, skill_reads: 0, input_tokens: 2000, first_call_tokens: 1000,
    correct: true, missing: [], ungrounded: [], ...extra,
  }
}

const RESULTS = [
  result('checkpointer', 1), result('checkpointer', 2, { first_call_tokens: 1600, correct: false, missing: ['Speedcubing'] }),
  result('jev', 1), result('jev', 2, { first_call_tokens: 600, sent_turns: [1], steps: 2 }),
]

const RUN: EvalRun = {
  id: '20261001-120000', created_at: '2026-10-01T12:00:00+08:00', model: 'anthropic:claude-sonnet-4-6',
  sandbox: false, repeats: 1, arms: ['checkpointer', 'jev'], context_filter: { keep_last: 1, max_rounds: 3 },
  conversations: [{
    id: 'chat', about: 'two turns',
    turns: [
      { id: 'chat.1', prompt: 'question 1', expect: [], note: '' },
      { id: 'chat.2', prompt: 'question 2', expect: ['Speedcubing'], note: '' },
    ],
  }],
  results: RESULTS,
  summary: {},
}

const HISTORY: HistoryLine[] = [
  { turn: 2, role: 'user', content: 'question 2' },
  { turn: 2, role: 'assistant', content: 'Let me check.', tool_calls: [{ id: 'c1', name: 'query_database', args: { sql: 'SELECT 1' } }] },
  { turn: 2, role: 'tool', content: '[{"n": 1}]', tool_call_id: 'c1', name: 'query_database', error: false },
  { turn: 2, role: 'assistant', content: 'Speedcubing.' },
]

afterEach(() => vi.unstubAllGlobals())

function backend(runs: EvalRun[]) {
  return (call: Call) => {
    if (call.url === '/api/evals/context/runs') {
      return jsonResponse(runs.map((r) => ({ ...r, conversations: r.conversations.map((c) => c.id) })))
    }
    if (call.url === `/api/evals/context/runs/${RUN.id}`) return jsonResponse(RUN)
    if (call.url.startsWith(`/api/evals/context/runs/${RUN.id}/history/chat/`)) return jsonResponse(HISTORY)
    return jsonResponse({ detail: 'unexpected' }, 500)
  }
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/evals/*" element={<EvalsPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('Evals', () => {
  it('opens on the context sub-page', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals')
    expect(await screen.findByRole('heading', { name: 'Context management with Jev' })).toBeInTheDocument()
    const tab = screen.getByRole('link', { name: 'Context (Jev)' })
    expect(tab).toHaveAttribute('href', '/evals/context')
    expect(tab).toHaveClass('active')
  })

  it('compares the arms of the latest run', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/context')
    const summary = await screen.findByRole('table', { name: 'Summary' })
    const firstCall = within(summary).getByRole('row', { name: /First call/ })
    // checkpointer (1000 + 1600) / 2 = 1300, jev (1000 + 600) / 2 = 800: 38% fewer
    expect(firstCall).toHaveTextContent('1,300')
    expect(firstCall).toHaveTextContent('800')
    expect(firstCall).toHaveTextContent('-38%')
    expect(within(summary).getByRole('row', { name: /Correct/ })).toHaveTextContent('1 / 2')
    expect(screen.getByRole('img', { name: 'First-call tokens per turn' })).toBeInTheDocument()
  })

  it('lists each turn with what each arm did', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/context')
    const turns = await screen.findByRole('table', { name: 'Turns' })
    const second = within(turns).getByRole('row', { name: /question 2/ })
    expect(second).toHaveTextContent('✗')
    expect(within(second).getByLabelText('Turns sent')).toHaveTextContent('1')
  })

  it('opens a turn to show both answers and the steps behind them', async () => {
    const calls = mockFetch(backend([RUN]))
    renderAt('/evals/context')
    const turns = await screen.findByRole('table', { name: 'Turns' })
    await userEvent.click(within(turns).getByRole('row', { name: /question 2/ }))

    expect(await screen.findByText('jev answer 2')).toBeInTheDocument()
    expect(screen.getByText('checkpointer answer 2')).toBeInTheDocument()
    expect(screen.getByText(/Expected:/).closest('p')).toHaveTextContent('Expected: Speedcubing')
    expect(screen.getByText('Speedcubing', { selector: 'dd' })).toBeInTheDocument() // what checkpointer missed
    expect(calls.map((c) => c.url)).toContain(`/api/evals/context/runs/${RUN.id}/history/chat/jev?repeat=1`)
    expect((await screen.findAllByText('1 step')).length).toBeGreaterThan(0)
  })

  it('explains how to make a run when there is none', async () => {
    mockFetch(backend([]))
    renderAt('/evals/context')
    expect(await screen.findByRole('heading', { name: 'No eval runs yet' })).toBeInTheDocument()
  })
})

describe('results helpers', () => {
  it('summarizes like the backend', () => {
    const s = summarize(RESULTS)
    expect(s.checkpointer).toMatchObject({ turns: 2, checked: 2, correct: 1, first_call_per_turn: 1300 })
    expect(s.jev).toMatchObject({ steps: 3, first_call_per_turn: 800 })
  })

  it('turns history lines into steps and an answer', () => {
    const { steps, answer } = turnSteps(HISTORY, 2)
    expect(answer).toBe('Speedcubing.')
    expect(steps).toEqual([
      { kind: 'thinking', text: 'Let me check.' },
      { kind: 'tool', id: 'c1', name: 'query_database', args: { sql: 'SELECT 1' }, result: '[{"n": 1}]', error: false },
    ])
    expect(turnSteps(HISTORY, 1)).toEqual({ steps: [], answer: '' })
  })
})

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ComparePair, CompareRun, EvalStatus, HistoryLine, SideSummary, SystemResult } from '../api/types'
import { jsonResponse, mockFetch, type Call } from '../test/fetch'
import { EvalsPage } from './EvalsPage'

function turn(thread: string, n: number, opts: { correct?: EvalStatus; tokens?: number; jevSent?: number[] } = {}): SystemResult {
  return {
    case_id: `${thread}_t${n}`, question: `question ${n}`, final_answer: `${thread} answer ${n}`, stopped: false,
    stream_errors: [], artifacts: [],
    correct: { status: opts.correct ?? 'pass', reason: opts.correct === 'unknown' ? 'no answer key' : '', expected_values: { status: null, found: [], missing: [] } },
    grounded: { status: 'pass', reason: '', answer_values: [], observed_count: 0, ungrounded_values: [], unsupported_claims: [] },
    instruction_following: { status: 'pass', reason: '', checks: [] },
    execution_strategy: { status: 'pass', reason: '' },
    tool_usage: { tools: {}, tool_calls: 1, queries: 1, exports: 0, skill_reads: 0, skills_read: [], code_executions: 0 },
    recovery: { errors: [], retries: 0, identical_retries: 0, oom_events: 0, oom_resolved_by: null, sandbox_moves: 0, sandbox_events: [], code_steps: [] },
    efficiency: { steps: 1, queries: 1, skill_reads: 0, code_executions: 0, model_calls: 2, input_tokens: 9000, output_tokens: 100, total_tokens: opts.tokens ?? 10000, runtime_seconds: 5 },
    context: opts.jevSent ? { sent_turns: opts.jevSent, earlier_turns: n - 1, seconds: 0.2 } : null,
  }
}

function summary(results: SystemResult[]): SideSummary {
  const tally = (key: 'correct' | 'grounded' | 'instruction_following' | 'execution_strategy') => ({
    pass: results.filter((r) => r[key].status === 'pass').length,
    fail: results.filter((r) => r[key].status === 'fail').length,
    unknown: results.filter((r) => r[key].status === 'unknown' || r[key].status === 'judge_error').length,
  })
  return {
    turns: results.length, correct: tally('correct'), grounded: tally('grounded'),
    instruction_following: tally('instruction_following'), execution_strategy: tally('execution_strategy'),
    steps: results.reduce((t, r) => t + r.efficiency.steps, 0),
    total_tokens: results.reduce((t, r) => t + r.efficiency.total_tokens, 0),
    runtime_seconds: results.reduce((t, r) => t + r.efficiency.runtime_seconds, 0), errors: 0,
  }
}

function pair(): ComparePair {
  const full = [turn('f1', 1), turn('f1', 2, { tokens: 20000 }), turn('f1', 3, { tokens: 30000 })]
  const jev = [turn('j1', 1, { jevSent: [] }), turn('j1', 2, { jevSent: [1], correct: 'unknown' }), turn('j1', 3, { jevSent: [1, 2], tokens: 5000 })]
  return {
    id: 'f1', questions: ['question 1', 'question 2', 'question 3'],
    full: { thread: 'f1', results: full, summary: summary(full) },
    jev: { thread: 'j1', results: jev, summary: summary(jev) },
  }
}

const RUN: CompareRun = {
  id: '20261005-150000', created_at: '2026-10-05T15:00:00+08:00', judge_model: 'j', status: 'done', total: 1,
  unpaired: { full: [], jev: [{ thread: 'j9', questions: ['alone'] }] },
  pairs: [pair()],
}

const HISTORY: HistoryLine[] = [
  { turn: 2, role: 'user', content: 'question 2' },
  { turn: 2, role: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'query_database', args: { sql: 'SELECT 2' } }] },
  { turn: 2, role: 'tool', content: '[]', tool_call_id: 'c1', name: 'query_database', error: false },
]

afterEach(() => vi.unstubAllGlobals())

function backend(runs: CompareRun[], preview = { available: true, pairs: 1, turns: 3, only_full: 0, only_jev: 1, running: null as string | null }) {
  return (call: Call) => {
    if (call.url === '/api/evals/compare/pairs') return jsonResponse(preview)
    if (call.url === '/api/evals/compare/runs' && call.method === 'POST') return jsonResponse({ id: RUN.id }, 202)
    if (call.url === '/api/evals/compare/runs') {
      return jsonResponse(runs.map((r) => ({ id: r.id, created_at: r.created_at, judge_model: r.judge_model, status: r.status, total: r.total, pairs: r.pairs.length })))
    }
    if (call.url === `/api/evals/compare/runs/${RUN.id}`) return jsonResponse(RUN)
    if (call.url.startsWith(`/api/evals/compare/runs/${RUN.id}/history/`)) return jsonResponse(HISTORY)
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

describe('Jev vs full', () => {
  it('shows one row per conversation, both ways side by side', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/jev')
    expect(await screen.findByRole('heading', { name: 'Jev vs full conversation' })).toBeInTheDocument()
    const table = await screen.findByRole('table', { name: 'Conversations' })
    const rows = within(table).getAllByRole('row').filter((r) => r.className.includes('turn'))
    expect(rows).toHaveLength(1)
    const row = rows[0]
    // correct: full 3/3; Jev 2/2 judged plus one unknown, not a fail
    expect(row).toHaveTextContent('3/3')
    expect(row).toHaveTextContent('2/2 +1?')
    // tokens: full 60K, Jev 25K: -58%
    expect(row).toHaveTextContent('60.0K')
    expect(row).toHaveTextContent('-58%')
    expect(within(row).getByLabelText('Jev sent')).toHaveTextContent('2←13←1,2')
  })

  it('opens a conversation turn by turn, with both answers and steps', async () => {
    const calls = mockFetch(backend([RUN]))
    renderAt('/evals/jev')
    const table = await screen.findByRole('table', { name: 'Conversations' })
    await userEvent.click(within(table).getAllByRole('row').find((r) => r.className.includes('turn'))!)

    expect(await screen.findByText('f1 answer 2')).toBeInTheDocument()
    expect(screen.getByText('j1 answer 2')).toBeInTheDocument()
    expect(screen.getByText('no answer key')).toBeInTheDocument()
    expect(calls.map((c) => c.url)).toEqual(expect.arrayContaining([
      `/api/evals/compare/runs/${RUN.id}/history/full/f1`, `/api/evals/compare/runs/${RUN.id}/history/jev/j1`,
    ]))
  })

  it('says which conversations have no twin', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/jev')
    expect(await screen.findByText(/1 conversation has no twin yet/)).toBeInTheDocument()
    expect(screen.getByText('Not compared (no twin): 0 full, 1 Jev.')).toBeInTheDocument()
  })

  it('asks before judging every pair', async () => {
    const calls = mockFetch(backend([]))
    renderAt('/evals/jev')
    await userEvent.click(await screen.findByRole('button', { name: 'Compare all conversations' }))
    const confirm = screen.getByRole('group', { name: 'Confirm comparison' })
    expect(confirm).toHaveTextContent('Judge 1 conversation both ways (6 turns)?')
    await userEvent.click(within(confirm).getByRole('button', { name: 'Start' }))
    expect(calls.filter((c) => c.method === 'POST').map((c) => c.url)).toEqual(['/api/evals/compare/runs'])
  })
})

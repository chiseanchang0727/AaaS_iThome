import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { EvalStatus, SystemResult, SystemRun } from '../api/types'
import { jsonResponse, mockFetch, type Call } from '../test/fetch'
import { EvalsPage } from './EvalsPage'
import { compact, groupStats, recoveryTotals, verdictCounts } from './overview'

function result(id: string, opts: { correct?: EvalStatus; tokens?: number; jev?: { sent: number[]; earlier: number }; errors?: number } = {}): SystemResult {
  const tokens = opts.tokens ?? 10000
  return {
    case_id: id, question: `question ${id}`, final_answer: 'a', stopped: false, stream_errors: [], artifacts: [],
    correct: { status: opts.correct ?? 'pass', reason: '', expected_values: { status: null, found: [], missing: [] } },
    grounded: { status: 'pass', reason: '', answer_values: [], observed_count: 0, ungrounded_values: id === 'b' ? ['254'] : [], unsupported_claims: [] },
    instruction_following: { status: 'unknown', reason: '', checks: [] },
    execution_strategy: { status: id === 'c' ? 'judge_error' : 'pass', reason: '' },
    tool_usage: { tools: {}, tool_calls: 2, queries: 1, exports: 0, skill_reads: 1, skills_read: [], code_executions: 0 },
    recovery: {
      errors: Array.from({ length: opts.errors ?? 0 }, () => ({ tool: 'query_database', args: '', result: '', next: 'changed' as const })),
      retries: 0, identical_retries: 0, oom_events: id === 'c' ? 1 : 0, oom_resolved_by: null, sandbox_moves: 0, sandbox_events: [], code_steps: [],
    },
    efficiency: { steps: 2, queries: 1, skill_reads: 1, code_executions: 0, model_calls: 3, input_tokens: tokens - 200, output_tokens: 200, total_tokens: tokens, runtime_seconds: 10 },
    context: opts.jev ? { sent_turns: opts.jev.sent, earlier_turns: opts.jev.earlier, seconds: 0.2 } : null,
  }
}

// a, b: full conversation (b is a follow-up); c, d: Jev (d is a follow-up)
const RESULTS = [
  result('a', { tokens: 20000 }),
  result('b', { correct: 'fail', tokens: 30000, errors: 1 }),
  result('c', { tokens: 8000, jev: { sent: [], earlier: 0 } }),
  result('d', { correct: 'unknown', tokens: 12000, jev: { sent: [1], earlier: 1 } }),
]

const RUN: SystemRun = {
  id: '20261005-120000', created_at: '2026-10-05T12:00:00+08:00', model: 'saved conversations', judge_model: 'j',
  sandbox: false, rejudged_from: null, source: 'history', status: 'done', total: 4,
  cases: [
    { id: 'a', question: 'q', setup: [], expected_values: {}, required_tools: [], forbidden_tools: [], required_skills: [], required_outputs: [], expected_behavior: '', sandbox: false, note: '' },
    { id: 'b', question: 'q', setup: ['earlier'], expected_values: {}, required_tools: [], forbidden_tools: [], required_skills: [], required_outputs: [], expected_behavior: '', sandbox: false, note: '' },
  ],
  results: RESULTS,
  summary: { cases: 4, correct: 2, grounded: 4, instruction_following: 0, execution_strategy: 3, judge_errors: 1, total_tokens: 70000, runtime_seconds: 40 },
}

afterEach(() => vi.unstubAllGlobals())

function backend(runs: SystemRun[]) {
  return (call: Call) => {
    if (call.url === '/api/evals/system/runs') {
      return jsonResponse(runs.map((r) => ({ ...r, cases: r.results.map((x) => x.case_id), source: r.source, status: r.status, total: r.total })))
    }
    if (call.url === `/api/evals/system/runs/${RUN.id}`) return jsonResponse(RUN)
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

describe('Overview helpers', () => {
  it('counts verdicts and rates passes over all turns', () => {
    expect(verdictCounts(RESULTS, 'correct')).toEqual({ pass: 2, fail: 1, unknown: 1, judge_error: 0 })
    const s = groupStats(RESULTS)
    expect(s.passRate.correct).toBe(0.5)
    expect(s.passRate.instruction_following).toBe(0)
    expect(s.totalTokens).toBe(17500)
    expect(groupStats([]).passRate.correct).toBeNull()
  })

  it('adds up recovery facts', () => {
    expect(recoveryTotals(RESULTS)).toMatchObject({ errors: 1, oom: 1, ungroundedValues: 1, stopped: 0 })
  })

  it('writes big numbers compactly', () => {
    expect([compact(1284), compact(12900), compact(4_200_000)]).toEqual(['1,284', '12.9K', '4.2M'])
  })
})

describe('Overview page', () => {
  it('is the first Evals tab and shows pass rates', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals')
    expect(await screen.findByRole('heading', { name: 'Overview' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Overview' })).toHaveClass('active')
    const tiles = await screen.findByLabelText('Pass rates')
    expect(tiles).toHaveTextContent('Correct50%2 of 4 turns passed · 1 unknown')
    expect(tiles).toHaveTextContent('Strategy75%3 of 4 turns passed · 1 judge errors')
  })

  it('writes the verdict counts out next to each bar', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/overview')
    expect(await screen.findByRole('img', { name: 'Correct: 2 pass · 1 fail · 1 unknown' })).toBeInTheDocument()
  })

  it('leaves the Jev comparison to its own page', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/overview')
    expect(await screen.findByRole('link', { name: 'Jev vs full' , current: false })).toBeInTheDocument()
    expect(screen.queryByRole('table', { name: 'Jev compared with the full conversation' })).not.toBeInTheDocument()
  })

  it('ranks turns by the chosen measure', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/overview')
    const bars = await screen.findByRole('list', { name: 'Tokens per turn' })
    expect(within(bars).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      'question b30.0K', 'question a20.0K', 'question d12.0K', 'question c8,000',
    ])
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Measure' }), 'runtime')
    expect(screen.getByRole('list', { name: 'Runtime per turn' })).toBeInTheDocument()
  })

  it('lists every run', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/overview')
    const runs = await screen.findByRole('table', { name: 'All runs' })
    expect(within(runs).getByRole('row', { name: /saved conversations/ })).toHaveTextContent('50%')
  })

  it('points to the System page when nothing is evaluated', async () => {
    mockFetch(backend([]))
    renderAt('/evals/overview')
    expect(await screen.findByRole('heading', { name: 'Nothing evaluated yet' })).toBeInTheDocument()
  })
})

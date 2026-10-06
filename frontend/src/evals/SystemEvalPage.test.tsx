import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HistoryLine, SystemResult, SystemRun } from '../api/types'
import { jsonResponse, mockFetch, type Call } from '../test/fetch'
import { EvalsPage } from './EvalsPage'

function result(caseId: string, extra: Partial<SystemResult> = {}): SystemResult {
  return {
    context: null,
    case_id: caseId, question: `question ${caseId}`, final_answer: `answer ${caseId}`, stopped: false,
    stream_errors: [], artifacts: [],
    correct: { status: 'pass', reason: 'right', expected_values: { status: 'pass', found: ['videos=17'], missing: [] } },
    grounded: { status: 'pass', reason: 'seen', answer_values: ['17'], observed_count: 3, ungrounded_values: [], unsupported_claims: [] },
    instruction_following: { status: 'pass', reason: 'did it', checks: [] },
    execution_strategy: { status: 'pass', reason: 'sound' },
    tool_usage: { tools: { query_database: 2 }, tool_calls: 2, queries: 2, exports: 0, skill_reads: 1, skills_read: ['query_database'], code_executions: 0 },
    recovery: { errors: [], retries: 0, identical_retries: 0, oom_events: 0, oom_resolved_by: null, sandbox_moves: 0, sandbox_events: [], code_steps: [] },
    efficiency: { steps: 3, queries: 2, skill_reads: 1, code_executions: 0, model_calls: 4, input_tokens: 9000, output_tokens: 300, total_tokens: 9300, runtime_seconds: 12.5 },
    ...extra,
  }
}

const OOM = result('oom', {
  context: { sent_turns: [1], earlier_turns: 2, seconds: 0.4 },
  correct: { status: 'fail', reason: 'Missing from the answer: total=16,000.', expected_values: { status: 'fail', found: [], missing: ['total=16,000'] } },
  grounded: {
    status: 'fail', reason: 'one made-up number', answer_values: ['6,763', '254'], observed_count: 5,
    ungrounded_values: ['254'], unsupported_claims: [{ claim: 'about 254 each', reason: 'never printed' }],
  },
  execution_strategy: { status: 'judge_error', reason: 'RuntimeError: rate limited' },
  recovery: {
    errors: [{ tool: 'execute', args: 'python3 a.py', result: 'Killed', next: 'identical' }],
    retries: 1, identical_retries: 1, oom_events: 1, oom_resolved_by: 'bigger sandbox', sandbox_moves: 1,
    sandbox_events: ['upgrade_started', 'switched'],
    code_steps: [{ command: 'python3 a.py', exit_code: 137, seconds: 4, peak_memory_mb: 1012, memory_limit_mb: 1024, out_of_memory: true, strategy: 'first' }],
  },
})

const RUN: SystemRun = {
  id: '20261005-120000', created_at: '2026-10-05T12:00:00+08:00', model: 'anthropic:claude-sonnet-4-6',
  judge_model: 'anthropic:claude-sonnet-4-6', sandbox: true, rejudged_from: null,
  cases: [
    { id: 'lookup', question: 'question lookup', setup: [], expected_values: { videos: 17 }, required_tools: [], forbidden_tools: [],
      required_skills: [], required_outputs: [], expected_behavior: 'Counts distinct videos.', sandbox: false, note: '' },
  ],
  results: [result('lookup'), OOM],
  summary: { cases: 2, correct: 1, grounded: 1, instruction_following: 2, execution_strategy: 1, judge_errors: 1, total_tokens: 18600, runtime_seconds: 25 },
}

const HISTORY: HistoryLine[] = [
  { turn: 1, role: 'user', content: 'question lookup' },
  { turn: 1, role: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'query_database', args: { sql: 'SELECT count(DISTINCT video_id) FROM videos' } }] },
  { turn: 1, role: 'tool', content: '[{"count": 17}]', tool_call_id: 'c1', name: 'query_database', error: false },
  { turn: 1, role: 'assistant', content: 'answer lookup' },
]

afterEach(() => vi.unstubAllGlobals())

function info(r: SystemRun) {
  return { ...r, cases: r.results.map((x) => x.case_id), source: r.source ?? 'cases', status: r.status ?? 'done', total: r.total ?? r.results.length }
}

const HISTORY_RUN: SystemRun = {
  ...RUN, id: '20261005-130000', model: 'saved conversations', source: 'history', status: 'running', total: 2,
  results: [], cases: [],
  summary: { ...RUN.summary, cases: 0, correct: 0, grounded: 0, instruction_following: 0, execution_strategy: 0, judge_errors: 0 },
}

function backend(runs: SystemRun[], preview = { available: true, folder: 'evals/system/chat_history', conversations: 2, turns: 3, running: null as string | null }) {
  return (call: Call) => {
    if (call.url === '/api/evals/system/history-runs' && call.method === 'POST') {
      runs.unshift(HISTORY_RUN)
      return jsonResponse({ id: HISTORY_RUN.id, total: 2 }, 202)
    }
    if (call.url === '/api/evals/system/history-runs') return jsonResponse(preview)
    if (call.url === '/api/evals/system/runs') return jsonResponse(runs.map(info))
    if (call.url === `/api/evals/system/runs/${HISTORY_RUN.id}`) return jsonResponse(HISTORY_RUN)
    if (call.url === `/api/evals/system/runs/${RUN.id}`) return jsonResponse(RUN)
    if (call.url.startsWith(`/api/evals/system/runs/${RUN.id}/history/`)) return jsonResponse(HISTORY)
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

describe('System eval', () => {
  it('has its own tab', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/system')
    expect(await screen.findByRole('heading', { name: 'Whole agent system' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'System' })).toHaveClass('active')
  })

  it('lists every case with its verdicts and measurements', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/system')
    const cases = await screen.findByRole('table', { name: 'Cases' })
    const lookup = within(cases).getByRole('row', { name: /lookup/ })
    expect(lookup).toHaveTextContent('9,300')
    expect(lookup).toHaveTextContent('12.5s')
    const oom = within(cases).getByRole('row', { name: /oom/ })
    expect(within(oom).getAllByLabelText('fail')).toHaveLength(2)
    expect(within(oom).getByLabelText('the judge call failed')).toBeInTheDocument()
    expect(screen.getByText('1 judge errors')).toBeInTheDocument()
    expect(within(oom).getByText('Jev · sent 1 of 2')).toBeInTheDocument()
    expect(within(lookup).getByText('full')).toBeInTheDocument()
    expect(screen.getByText('Jev context 1/2 turns')).toBeInTheDocument()
  })

  it('opens a case to show the judgments, trace and queries', async () => {
    const calls = mockFetch(backend([RUN]))
    renderAt('/evals/system')
    const cases = await screen.findByRole('table', { name: 'Cases' })
    await userEvent.click(within(cases).getByRole('row', { name: /lookup/ }))

    expect(await screen.findByText('Counts distinct videos.')).toBeInTheDocument()
    expect(screen.getByText('videos = 17')).toBeInTheDocument()
    expect(calls.map((c) => c.url)).toContain(`/api/evals/system/runs/${RUN.id}/history/lookup`)
    const list = await screen.findByLabelText('Queries')
    expect(within(list).getByText('SELECT count(DISTINCT video_id) FROM videos')).toBeInTheDocument()
    expect(screen.getByText('1 step')).toBeInTheDocument()
  })

  it('shows recovery, out-of-memory and unsupported claims', async () => {
    mockFetch(backend([RUN]))
    renderAt('/evals/system')
    const cases = await screen.findByRole('table', { name: 'Cases' })
    await userEvent.click(within(cases).getByRole('row', { name: /oom/ }))

    expect(await screen.findByText('“about 254 each”: never printed')).toBeInTheDocument()
    expect(screen.getByText('next: identical')).toBeInTheDocument()
    expect(screen.getByText('out of memory')).toBeInTheDocument()
    expect(screen.getByText('1, resolved by bigger sandbox')).toBeInTheDocument()
    expect(screen.getByText('RuntimeError: rate limited')).toBeInTheDocument()
    expect(screen.getByText('Context (Jev)', { selector: 'dt' }).nextElementSibling).toHaveTextContent('sent turn 1 of 2')
  })

  it('explains how to make a run when there is none', async () => {
    mockFetch(backend([]))
    renderAt('/evals/system')
    expect(await screen.findByRole('heading', { name: 'No system eval runs yet' })).toBeInTheDocument()
    expect(await screen.findByRole('button', { name: 'Evaluate all conversation history' })).toBeEnabled()
  })

  it('judges all conversation history after asking first', async () => {
    const calls = mockFetch(backend([RUN]))
    renderAt('/evals/system')
    await userEvent.click(await screen.findByRole('button', { name: 'Evaluate all conversation history' }))

    const confirm = screen.getByRole('group', { name: 'Confirm evaluation' })
    expect(confirm).toHaveTextContent('Judge 3 turns from 2 conversations?')
    expect(calls.some((c) => c.method === 'POST')).toBe(false) // nothing starts before Start
    await userEvent.click(within(confirm).getByRole('button', { name: 'Start' }))

    expect(calls.filter((c) => c.method === 'POST').map((c) => c.url)).toEqual(['/api/evals/system/history-runs'])
    expect(await screen.findByRole('status')).toHaveTextContent('Evaluating… 0 / 2')
    expect(screen.getByRole('columnheader', { name: 'Question' })).toBeInTheDocument()
  })

  it('says where to put conversations when there are none', async () => {
    mockFetch(backend([RUN], { available: true, folder: 'evals/system/chat_history', conversations: 0, turns: 0, running: null }))
    renderAt('/evals/system')
    expect(await screen.findByRole('button', { name: 'Evaluate all conversation history' })).toBeDisabled()
    expect(screen.getByText('evals/system/chat_history')).toBeInTheDocument()
  })

  it('cannot start a second history run while one is going', async () => {
    mockFetch(backend([RUN], { available: true, folder: 'evals/system/chat_history', conversations: 2, turns: 3, running: '20261005-130000' }))
    renderAt('/evals/system')
    expect(await screen.findByRole('button', { name: 'Evaluating history…' })).toBeDisabled()
  })
})

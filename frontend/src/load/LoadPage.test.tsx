import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ConversationTimeline, Load, LoadStep } from '../api/types'
import { jsonResponse, mockFetch } from '../test/fetch'
import { LoadPage } from './LoadPage'

function step(account: string, command: string, peak: number, extra: Partial<LoadStep> = {}): LoadStep {
  return {
    conversation: `c-${account}`, account, turn: 1, prompt: `${account}'s question`, ts: '2026-10-02T10:00:00Z',
    command, exit_code: 0, seconds: 2, run_seconds: 0.5, cpu_seconds: 0.4, peak_memory_mb: peak,
    memory_limit_mb: 1024, out_of_memory: false, ...extra,
  }
}

const SUMMARY = {
  steps: 2, conversations: 2, run_seconds: 1, cpu_seconds: 0.8, overhead_seconds: 3,
  peak_memory_mb: 973, average_peak_memory_mb: 558.5, out_of_memory: 1, failed: 1,
}

const LOAD: Load = {
  account: null, accounts: ['alice', 'bob'], memory_limit_mb: 1024, summary: SUMMARY,
  per_account: {
    alice: { ...SUMMARY, steps: 1, peak_memory_mb: 973, out_of_memory: 1 },
    bob: { ...SUMMARY, steps: 1, peak_memory_mb: 144, out_of_memory: 0 },
  },
  steps: [
    step('alice', 'python3 big.py', 973, { exit_code: 137, out_of_memory: true, conversation: 'c-alice' }),
    step('bob', 'python3 duck.py', 144),
  ],
  conversations: [{
    conversation: 'c-alice', account: 'alice', question: 'the 3.2 GB matrix', turns: 1, steps: 2,
    run_seconds: 44.2, cpu_seconds: 44.2,
    out_of_memory: 1, upgrades: 1, upgrade_failures: 0, peak_memory_mb: 3084, sandbox_gb: 4, last: '2026-10-02T10:01:00Z',
  }],
}

const at = (s: number) => new Date(Date.UTC(2026, 9, 2, 10, 0, s)).toISOString()
const TIMELINE: ConversationTimeline = {
  conversation: 'c-alice', account: 'alice',
  timeline: [
    { ts: at(0), turn: 1, kind: 'question', text: 'the 3.2 GB matrix' },
    { ts: at(1), turn: 1, kind: 'event', event: 'sandbox_ready', how: 'warm', memory_gb: 1, seconds: 0 },
    { ts: at(4), turn: 1, kind: 'action', tool: 'execute', detail: 'python3 m.py' },
    { ts: at(14), turn: 1, kind: 'step', command: 'python3 m.py', exit_code: 137, seconds: 11, run_seconds: 9.6,
      cpu_seconds: 9.4, peak_memory_mb: 973, memory_limit_mb: 1024, out_of_memory: true },
    { ts: at(14), turn: 1, kind: 'event', event: 'upgrade_started', from_gb: 1, to_gb: 4, cpu: 2, memory_used_gb: 6 },
    { ts: at(28), turn: 1, kind: 'event', event: 'bigger_created', memory_gb: 4, cpu: 2, seconds: 14 },
    { ts: at(31), turn: 1, kind: 'event', event: 'files_copied', bytes: 40960, seconds: 1.2 },
    { ts: at(31), turn: 1, kind: 'event', event: 'switched', memory_gb: 4 },
    { ts: at(32), turn: 1, kind: 'event', event: 'old_deleted', memory_gb: 1, memory_used_gb: 5, seconds: 0.4 },
    { ts: at(66), turn: 1, kind: 'step', command: 'python3 m.py', exit_code: 0, seconds: 36, run_seconds: 34.6,
      cpu_seconds: 34.8, peak_memory_mb: 3084, memory_limit_mb: 4096, out_of_memory: false },
    { ts: at(70), turn: 1, kind: 'answer', text: 'The mean is 0.5.' },
  ],
}

afterEach(() => vi.unstubAllGlobals())

describe('LoadPage', () => {
  it('shows totals, the memory chart and each conversation with its statistics', async () => {
    mockFetch(() => jsonResponse(LOAD))
    render(<LoadPage />)
    expect(await screen.findByText('Code steps', { selector: '.card-label' })).toBeInTheDocument()
    expect(screen.getByText('Peak memory', { selector: '.card-label' }).parentElement).toHaveTextContent('973 MB')
    expect(screen.getByRole('img', { name: 'Peak memory per step' })).toBeInTheDocument()

    const row = within(screen.getByRole('table', { name: 'Conversations' })).getByRole('row', { name: /the 3.2 GB matrix/ })
    for (const text of ['alice', '44.2s', '3084 MB', '4 GB']) expect(row).toHaveTextContent(text)
    expect(screen.queryByRole('table', { name: 'Code steps' })).toBeNull() // steps live in each conversation
    expect(screen.queryByRole('region', { name: 'Conversation timeline' })).toBeNull() // closed by default
  })

  it('a bar in the chart opens its conversation', async () => {
    mockFetch((call) => jsonResponse(call.url.startsWith('/api/load/conversations/') ? TIMELINE : LOAD))
    render(<LoadPage />)
    await userEvent.click(await screen.findByRole('button', { name: 'Open the conversation of a 973 MB step' }))
    expect(await screen.findByRole('region', { name: 'Conversation timeline' })).toHaveTextContent(
      'Reached the limit: moving to a bigger sandbox, 1 GB → 4 GB',
    )
  })

  it('opens a conversation to show how its sandbox reached the limit and moved', async () => {
    mockFetch((call) => jsonResponse(call.url.startsWith('/api/load/conversations/') ? TIMELINE : LOAD))
    render(<LoadPage />)
    const conversations = await screen.findByRole('table', { name: 'Conversations' })
    await userEvent.click(within(conversations).getByRole('row', { name: /the 3.2 GB matrix/ }))

    const detail = await screen.findByRole('region', { name: 'Conversation timeline' })
    for (const text of [
      'Sandbox ready: a warm one was handed over',
      'Code step killed: out of memory at 973 MB of 1024 MB',
      'Reached the limit: moving to a bigger sandbox, 1 GB → 4 GB',
      'Bigger sandbox created: 4 GB, 2 vCPU',
      'Work folder copied over (40 KB)',
      'Switched to the 4 GB sandbox',
      'Old 1 GB sandbox deleted',
      'Code step done: 3084 MB of 4096 MB',
    ]) {
      expect(detail).toHaveTextContent(text)
    }
    const lane = within(detail).getByLabelText('Sandbox over time')
    expect(lane).toHaveTextContent('1 GB sandbox')
    expect(lane).toHaveTextContent('moving')
    expect(lane).toHaveTextContent('4 GB sandbox')
  })

  it('filters by account', async () => {
    const calls = mockFetch(() => jsonResponse(LOAD))
    render(<LoadPage />)
    await userEvent.selectOptions(await screen.findByLabelText('Account'), 'alice')
    expect(calls.at(-1)?.url).toBe('/api/load?account=alice')
  })

  it('explains an empty page', async () => {
    mockFetch(() => jsonResponse({ ...LOAD, accounts: [], steps: [], summary: { ...SUMMARY, steps: 0 } }))
    render(<LoadPage />)
    expect(await screen.findByRole('heading', { name: 'No code steps yet' })).toBeInTheDocument()
  })
})

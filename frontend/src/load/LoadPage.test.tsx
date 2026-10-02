import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { Load, LoadStep } from '../api/types'
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
    step('alice', 'python3 big.py', 973, { exit_code: 137, out_of_memory: true }),
    step('bob', 'python3 duck.py', 144),
  ],
}

afterEach(() => vi.unstubAllGlobals())

describe('LoadPage', () => {
  it('shows totals, a memory chart and every step', async () => {
    mockFetch(() => jsonResponse(LOAD))
    render(<LoadPage />)
    expect(await screen.findByText('Code steps')).toBeInTheDocument()
    expect(screen.getByText('Peak memory', { selector: '.card-label' }).parentElement).toHaveTextContent('973 MB')
    expect(screen.getByRole('img', { name: 'Peak memory per step' })).toBeInTheDocument()

    const steps = screen.getByRole('table', { name: 'Code steps' })
    const big = within(steps).getByRole('row', { name: /python3 big.py/ })
    expect(big).toHaveTextContent('out of memory')
    expect(within(steps).getByRole('row', { name: /python3 duck.py/ })).toHaveTextContent('ok')
    expect(screen.getByRole('table', { name: 'Per account' })).toHaveTextContent('bob')
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

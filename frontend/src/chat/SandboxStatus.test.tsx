import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { jsonResponse, mockFetch } from '../test/fetch'
import { SandboxStatus } from './SandboxStatus'

const STATUS = {
  enabled: true, provider: 'daytona', max: 10, in_use: 4,
  busy: 1, idle: 2, starting: 0, warm: 1, warming: 0, waiting: 0, idle_minutes: 15,
  max_per_account: 3, account: 'test_user', memory_used_gb: 7, max_memory_gb: 10, bigger: 1,
  accounts: { test_user: { sandboxes: 2, busy: 1, idle: 1, starting: 0 }, alice: { sandboxes: 1, busy: 0, idle: 1, starting: 0 } },
}

afterEach(() => vi.unstubAllGlobals())

describe('SandboxStatus', () => {
  it('shows what each sandbox is doing and how many are free', async () => {
    mockFetch(() => jsonResponse(STATUS))
    render(<SandboxStatus refreshKey={false} />)
    const bar = await screen.findByLabelText('Sandboxes')
    expect(bar).toHaveTextContent('Sandboxes (daytona)')
    expect(bar).toHaveTextContent('1 busy')
    expect(bar).toHaveTextContent('2 idle')
    expect(bar).toHaveTextContent('1 ready')
    expect(bar).toHaveTextContent('6 of 10 free')
    expect(bar).toHaveTextContent('test_user: 2 of 3')
    expect(bar).toHaveTextContent('7 of 10 GB')
    expect(bar).toHaveTextContent('1 bigger')
    expect(bar).not.toHaveTextContent('starting') // zero counts beyond busy/idle are hidden
  })

  it('says when code execution is off', async () => {
    mockFetch(() => jsonResponse({ ...STATUS, enabled: false, provider: null }))
    render(<SandboxStatus refreshKey={false} />)
    expect(await screen.findByText(/Code execution is off/)).toBeInTheDocument()
  })

  it('says when the status cannot be read', async () => {
    mockFetch(() => jsonResponse({ detail: 'down' }, 500))
    render(<SandboxStatus refreshKey={false} />)
    expect(await screen.findByText('Sandbox status unavailable')).toBeInTheDocument()
  })
})

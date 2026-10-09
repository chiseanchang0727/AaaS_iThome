import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { PastConversation, PastConversationSummary } from '../api/types'
import { jsonResponse, mockFetch, sseResponse, type Call } from '../test/fetch'
import { ChatLayout } from './ChatLayout'
import { groupByDay } from './conversationGroups'

const NOW = new Date()
const hoursAgo = (h: number) => new Date(NOW.getTime() - h * 3600_000).toISOString()

const PAST: PastConversationSummary[] = [
  { id: 'today1', title: 'Chart sessions by device', turns: 1, started_at: hoursAgo(0), last_at: hoursAgo(0), files: 0 },
  { id: 'old1', title: 'Which category gets the most views?', turns: 2, started_at: hoursAgo(24 * 30), last_at: hoursAgo(24 * 30), files: 1 },
]

const OLD: PastConversation = {
  id: 'old1',
  lines: [
    { turn: 1, role: 'user', content: 'Which category gets the most views?' },
    { turn: 1, role: 'assistant', content: '**Music** leads.' },
    { turn: 2, role: 'user', content: 'Save this analysis.' },
    { turn: 2, role: 'assistant', content: '', tool_calls: [{ id: 's1', name: 'save_analysis', args: { title: 'Top category' } }] },
    { turn: 2, role: 'tool', content: 'Saved analysis ab12cd34ef56: "Top category". Its test run made c.html in 5s.', tool_call_id: 's1', name: 'save_analysis', error: false },
    { turn: 2, role: 'assistant', content: 'Saved.' },
  ],
  files: [],
}

afterEach(() => vi.unstubAllGlobals())

function backend(listed: PastConversationSummary[] = PAST) {
  return (call: Call) => {
    if (call.method === 'DELETE' && call.url.endsWith('/history')) return jsonResponse({ deleted: call.url.split('/')[3] })
    if (call.method === 'DELETE') return jsonResponse({ closed: true })
    if (call.url === '/api/conversations') return jsonResponse(listed)
    if (call.url === '/api/conversations/old1') return jsonResponse(OLD)
    if (call.url === '/api/sandboxes') return jsonResponse({ sandboxes: 0, max_sandboxes: 3, busy: 0, idle: 0, starting: 0, warm: 0 })
    if (call.url === '/api/chat') {
      return sseResponse([
        { type: 'thread', thread_id: 'live1' },
        { type: 'answer', text: 'Live answer.' },
        { type: 'done' },
      ])
    }
    return jsonResponse({ detail: 'unexpected' }, 500)
  }
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/chat/*" element={<ChatLayout />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('ChatLayout', () => {
  it('groups conversations by day', () => {
    const groups = groupByDay(PAST, NOW)
    expect(groups.map((g) => [g.label, g.items.map((i) => i.id)])).toEqual([
      ['Today', ['today1']],
      ['Older', ['old1']],
    ])
  })

  it('lists conversations on the side', async () => {
    mockFetch(backend())
    renderAt('/chat')
    const side = await screen.findByRole('complementary', { name: 'Conversations' })
    expect(await within(side).findByText('Today')).toBeInTheDocument()
    expect(within(side).getByRole('link', { name: 'Chart sessions by device' })).toHaveAttribute('href', '/chat/past/today1')
    expect(within(side).getByRole('link', { name: 'Which category gets the most views?' })).toHaveAttribute('href', '/chat/past/old1')
  })

  it('opens a past conversation in place, and keeps the live chat to come back to', async () => {
    mockFetch(backend())
    renderAt('/chat')
    await userEvent.type(screen.getByLabelText('Message'), 'Hello there{Enter}')
    expect(await screen.findByText('Live answer.')).toBeInTheDocument()

    const side = screen.getByRole('complementary', { name: 'Conversations' })
    await userEvent.click(await within(side).findByRole('link', { name: 'Which category gets the most views?' }))
    expect(await screen.findByText('Music', { selector: 'strong' })).toBeInTheDocument()
    expect(screen.getByText(/Read-only: a saved conversation/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open it in Analyses' })).toHaveAttribute('href', '/analyses/ab12cd34ef56')

    // the live chat is listed as "Now" until the history has it, and still holds its messages
    await userEvent.click(within(side).getByRole('link', { name: 'Hello there' }))
    expect(await screen.findByText('Live answer.')).toBeInTheDocument()
  })

  it('starts a new chat from the side', async () => {
    mockFetch(backend())
    renderAt('/chat')
    await userEvent.type(screen.getByLabelText('Message'), 'Hello there{Enter}')
    await screen.findByText('Live answer.')
    await userEvent.click(screen.getByRole('button', { name: '+ New chat' }))
    expect(screen.queryByText('Live answer.')).not.toBeInTheDocument()
    expect(screen.getByText(/Ask about the data/)).toBeInTheDocument()
  })

  it('deletes a conversation after asking, and leaves it if it was open', async () => {
    const calls = mockFetch(backend())
    renderAt('/chat/past/old1')
    expect(await screen.findByText('Music', { selector: 'strong' })).toBeInTheDocument()
    const side = screen.getByRole('complementary', { name: 'Conversations' })

    await userEvent.click(within(side).getByRole('button', { name: 'Delete Which category gets the most views?…' }))
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false)  // nothing yet
    await userEvent.click(within(within(side).getByRole('group', { name: /Confirm deleting/ })).getByRole('button', { name: 'Delete' }))

    expect(calls.filter((c) => c.method === 'DELETE').map((c) => c.url)).toEqual(['/api/conversations/old1/history'])
    expect(within(side).queryByRole('link', { name: 'Which category gets the most views?' })).not.toBeInTheDocument()
    expect(await screen.findByText(/Ask about the data/)).toBeInTheDocument()  // back on the live chat
  })

  it('can cancel a delete', async () => {
    const calls = mockFetch(backend())
    renderAt('/chat')
    const side = await screen.findByRole('complementary', { name: 'Conversations' })
    await userEvent.click(await within(side).findByRole('button', { name: 'Delete Which category gets the most views?…' }))
    await userEvent.click(within(side).getByRole('button', { name: 'Cancel' }))
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false)
    expect(within(side).getByRole('link', { name: 'Which category gets the most views?' })).toBeInTheDocument()
  })
})

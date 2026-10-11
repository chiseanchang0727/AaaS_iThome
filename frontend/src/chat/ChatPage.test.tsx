import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { jsonResponse, mockFetch, sseResponse, type Call } from '../test/fetch'
import { ChatPage } from './ChatPage'

/** The chat requests only: the page also polls GET /api/sandboxes. */
const chatCalls = (calls: Call[]) => calls.filter((c) => c.url === '/api/chat')

const TURN = [
  { type: 'thread', thread_id: 'thread-1' },
  { type: 'tool_call', id: 'c1', name: 'query_database', args: { sql: 'SELECT category_name FROM videos' } },
  { type: 'tool_result', id: 'c1', name: 'query_database', content: '[{"category_name":"Gaming"}]', error: false },
  { type: 'answer', text: '**Gaming** leads with 1.32M.' },
  { type: 'artifact', name: 'chart.html', url: '/api/artifacts/thread-1/chart.html', kind: 'html' },
  { type: 'done' },
]

afterEach(() => vi.unstubAllGlobals())

async function ask(text: string) {
  await userEvent.type(screen.getByLabelText('Message'), `${text}{Enter}`)
}

describe('ChatPage', () => {
  it('streams a turn: steps, markdown answer and chart', async () => {
    mockFetch(() => sseResponse(TURN))
    render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await ask('Which category leads?')

    expect(await screen.findByText('Gaming', { selector: 'strong' })).toBeInTheDocument()
    expect(screen.getByText('Which category leads?')).toBeInTheDocument()
    expect(screen.getByText('SELECT category_name FROM videos')).toBeInTheDocument()
    expect(screen.getByText('1 step')).toBeInTheDocument()
  })

  it('renders agent HTML in a sandboxed iframe that cannot reach the app', async () => {
    mockFetch(() => sseResponse(TURN))
    render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await ask('chart please')

    const frame = await screen.findByTitle('chart.html')
    expect(frame.tagName).toBe('IFRAME')
    expect(frame).toHaveAttribute('src', '/api/artifacts/thread-1/chart.html')
    expect(frame.getAttribute('sandbox')).toBe('allow-scripts')
    expect(frame.getAttribute('sandbox')).not.toContain('allow-same-origin')
  })

  it('sends follow-ups in the same conversation', async () => {
    const calls = mockFetch(() => sseResponse(TURN))
    render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await ask('first')
    await screen.findByTitle('chart.html')
    await ask('second')
    await waitFor(() => expect(chatCalls(calls)).toHaveLength(2))

    expect(chatCalls(calls)[0].body).toEqual({ message: 'first' })
    expect(chatCalls(calls)[1].body).toEqual({ message: 'second', thread_id: 'thread-1' })
  })

  it('shows why a message was refused', async () => {
    mockFetch(() => jsonResponse({ detail: 'this conversation is still answering a message' }, 409))
    render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await ask('hi')

    expect(await screen.findByText(/still answering a message/)).toBeInTheDocument()
    expect(screen.getByLabelText('Message')).not.toBeDisabled()
  })

  it('shows an error from the stream', async () => {
    mockFetch(() => sseResponse([{ type: 'thread', thread_id: 't' }, { type: 'error', message: 'model unavailable' }, { type: 'done' }]))
    render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await ask('hi')
    expect(await screen.findByText(/model unavailable/)).toBeInTheDocument()
  })

  it('does not render HTML the model writes in its answer', async () => {
    mockFetch(() =>
      sseResponse([{ type: 'thread', thread_id: 't' }, { type: 'answer', text: 'hi <img src=x onerror=alert(1)>' }, { type: 'done' }]),
    )
    const { container } = render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await ask('hi')
    await screen.findByText(/hi/, { selector: 'p' })
    expect(container.querySelector('.answer img')).toBeNull()
  })

  it('new conversation ends the old one and clears the chat', async () => {
    const calls = mockFetch((call) => (call.method === 'DELETE' ? jsonResponse({ closed: true }) : sseResponse(TURN)))
    render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await ask('first')
    await screen.findByTitle('chart.html')

    await userEvent.click(screen.getByRole('button', { name: 'New conversation' }))
    expect(calls.filter((c) => c.method === 'DELETE')).toEqual([
      expect.objectContaining({ url: '/api/conversations/thread-1' }),
    ])
    expect(screen.queryByText('first')).toBeNull()

    await ask('fresh start')
    await waitFor(() => expect(chatCalls(calls).at(-1)?.body).toEqual({ message: 'fresh start' }))
  })

  it('Shift+Enter adds a line instead of sending', async () => {
    const calls = mockFetch(() => sseResponse(TURN))
    render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await userEvent.type(screen.getByLabelText('Message'), 'line one{Shift>}{Enter}{/Shift}line two')
    expect(chatCalls(calls)).toHaveLength(0)
    expect(screen.getByLabelText('Message')).toHaveValue('line one\nline two')
  })

  it('links to an analysis the agent saved', async () => {
    mockFetch(() => sseResponse([
      { type: 'thread', thread_id: 'thread-1' },
      { type: 'tool_call', id: 's1', name: 'save_analysis', args: { title: 'Videos per month' } },
      { type: 'tool_result', id: 's1', name: 'save_analysis', content: 'Saved analysis ab12cd34ef56: "Videos per month". Its test run made chart.html in 9s.', error: false },
      { type: 'answer', text: 'Saved.' },
      { type: 'done' },
    ]))
    render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await ask('save this analysis')

    const link = await screen.findByRole('link', { name: 'Open it in Analyses' })
    expect(link).toHaveAttribute('href', '/analyses/ab12cd34ef56')
    expect(link.closest('p')).toHaveTextContent('✓ Saved “Videos per month”.')
  })

  it('shows no link when saving failed', async () => {
    mockFetch(() => sseResponse([
      { type: 'thread', thread_id: 'thread-1' },
      { type: 'tool_call', id: 's1', name: 'save_analysis', args: { title: 'x' } },
      { type: 'tool_result', id: 's1', name: 'save_analysis', content: 'NOT SAVED. Fix the recipe', error: false },
      { type: 'answer', text: 'It failed.' },
      { type: 'done' },
    ]))
    render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await ask('save this analysis')
    await screen.findByText('It failed.')
    expect(screen.queryByRole('link', { name: 'Open it in Analyses' })).not.toBeInTheDocument()
  })

  it('says when the agent changed a saved analysis instead of adding one', async () => {
    mockFetch(() => sseResponse([
      { type: 'thread', thread_id: 'thread-1' },
      { type: 'tool_call', id: 's1', name: 'save_analysis', args: { title: 'Views per category' } },
      { type: 'tool_result', id: 's1', name: 'save_analysis', content: 'Updated analysis ab12cd34ef56 to version 2: "Views per category". Its test run made r.html in 7s.', error: false },
      { type: 'answer', text: 'Done.' },
      { type: 'done' },
    ]))
    render(<MemoryRouter><ChatPage /></MemoryRouter>)
    await ask('remove the table')
    const link = await screen.findByRole('link', { name: 'Open it in Analyses' })
    expect(link).toHaveAttribute('href', '/analyses/ab12cd34ef56')
    expect(link.closest('p')).toHaveTextContent('✓ Updated “Views per category” to version 2.')
  })
})

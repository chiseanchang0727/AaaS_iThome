import { describe, expect, it } from 'vitest'

import type { ChatEvent } from '../api/types'
import { chatReducer, initialState, type ChatState } from './state'

const run = (state: ChatState, ...events: ChatEvent[]) =>
  events.reduce((s, event) => chatReducer(s, { type: 'event', event }), state)

describe('chatReducer', () => {
  it('adds the question and an empty streaming reply', () => {
    const state = chatReducer(initialState, { type: 'send', text: 'hi' })
    expect(state.busy).toBe(true)
    expect(state.messages.map((m) => [m.role, m.text, m.status])).toEqual([
      ['user', 'hi', 'done'],
      ['assistant', '', 'streaming'],
    ])
  })

  it('builds a reply from a streamed turn', () => {
    const sent = chatReducer(initialState, { type: 'send', text: 'q' })
    const state = run(
      sent,
      { type: 'thread', thread_id: 't1' },
      { type: 'thinking', text: 'Let me check.' },
      { type: 'tool_call', id: 'c1', name: 'query_database', args: { sql: 'SELECT 1' } },
      { type: 'tool_result', id: 'c1', name: 'query_database', content: '[{"n":1}]', error: false },
      { type: 'answer', text: 'One row.' },
      { type: 'artifact', name: 'chart.html', url: '/api/artifacts/t1/chart.html', kind: 'html' },
      { type: 'done' },
    )
    const reply = state.messages[1]
    expect(state.threadId).toBe('t1')
    expect(state.busy).toBe(false)
    expect(reply.status).toBe('done')
    expect(reply.text).toBe('One row.')
    expect(reply.steps).toEqual([
      { kind: 'thinking', text: 'Let me check.' },
      { kind: 'tool', id: 'c1', name: 'query_database', args: { sql: 'SELECT 1' }, result: '[{"n":1}]', error: false },
    ])
    expect(reply.artifacts).toEqual([{ name: 'chart.html', url: '/api/artifacts/t1/chart.html', kind: 'html' }])
  })

  it('matches each result to its own call', () => {
    const state = run(
      chatReducer(initialState, { type: 'send', text: 'q' }),
      { type: 'tool_call', id: 'a', name: 'ls', args: {} },
      { type: 'tool_call', id: 'b', name: 'query_database', args: {} },
      { type: 'tool_result', id: 'b', name: 'query_database', content: 'blocked', error: true },
    )
    const [a, b] = state.messages[1].steps
    expect(a).not.toHaveProperty('result')
    expect(b).toMatchObject({ result: 'blocked', error: true })
  })

  it('keeps an error status through done', () => {
    const state = run(
      chatReducer(initialState, { type: 'send', text: 'q' }),
      { type: 'error', message: 'boom' },
      { type: 'done' },
    )
    expect(state.messages[1]).toMatchObject({ status: 'error', error: 'boom' })
    expect(state.busy).toBe(false)
  })

  it('marks a failed request and frees the input', () => {
    const state = chatReducer(chatReducer(initialState, { type: 'send', text: 'q' }), {
      type: 'failed',
      message: 'this conversation is still answering a message',
    })
    expect(state.busy).toBe(false)
    expect(state.messages[1].status).toBe('error')
  })

  it('reset starts over', () => {
    const state = run(chatReducer(initialState, { type: 'send', text: 'q' }), { type: 'thread', thread_id: 't' })
    expect(chatReducer(state, { type: 'reset' })).toEqual(initialState)
  })

  it('ignores events when no reply is pending', () => {
    expect(run(initialState, { type: 'answer', text: 'stray' })).toEqual(initialState)
  })
})

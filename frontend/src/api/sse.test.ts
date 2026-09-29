import { describe, expect, it } from 'vitest'

import { SSEParser } from './sse'

const event = (type: string, data: object) => `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`

describe('SSEParser', () => {
  it('parses whole events', () => {
    const parser = new SSEParser()
    expect(parser.push(event('thread', { thread_id: 'a' }) + event('done', {}))).toEqual([
      { type: 'thread', thread_id: 'a' },
      { type: 'done' },
    ])
  })

  it('waits for the rest of an event split across chunks', () => {
    const parser = new SSEParser()
    const text = event('answer', { text: 'Gaming leads with 1.32M' })
    const cut = [5, 17, text.length - 1]
    expect(parser.push(text.slice(0, cut[0]))).toEqual([])
    expect(parser.push(text.slice(cut[0], cut[1]))).toEqual([])
    expect(parser.push(text.slice(cut[1], cut[2]))).toEqual([])
    expect(parser.push(text.slice(cut[2]))).toEqual([{ type: 'answer', text: 'Gaming leads with 1.32M' }])
  })

  it('handles CRLF line endings', () => {
    const parser = new SSEParser()
    expect(parser.push('event: done\r\ndata: {"type":"done"}\r\n\r\n')).toEqual([{ type: 'done' }])
  })

  it('joins multi-line data and ignores comments', () => {
    const parser = new SSEParser()
    expect(parser.push(': keep-alive\n\ndata: {"type":\ndata: "done"}\n\n')).toEqual([{ type: 'done' }])
  })

  it('keeps non-ASCII text intact', () => {
    const parser = new SSEParser()
    expect(parser.push(event('answer', { text: '台中 讀者 🎉' }))).toEqual([{ type: 'answer', text: '台中 讀者 🎉' }])
  })
})

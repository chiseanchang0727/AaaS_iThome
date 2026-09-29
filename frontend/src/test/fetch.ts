import { vi } from 'vitest'

export interface Call {
  url: string
  method: string
  body: unknown
}

type Handler = (call: Call) => Response | Promise<Response>

/** A Response streaming these Server-Sent Events, one chunk per event. */
export function sseResponse(events: object[]): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const e of events) {
        const type = (e as { type: string }).type
        controller.enqueue(encoder.encode(`event: ${type}\ndata: ${JSON.stringify(e)}\n\n`))
      }
      controller.close()
    },
  })
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } })
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

/** Replace fetch with `handler`; returns the list of calls it received. */
export function mockFetch(handler: Handler): Call[] {
  const calls: Call[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      let body: unknown = init?.body
      if (typeof body === 'string') body = JSON.parse(body)
      const call = { url: String(input), method: init?.method ?? 'GET', body }
      calls.push(call)
      return handler(call)
    }),
  )
  return calls
}

/**
 * Incremental Server-Sent Events parser.
 *
 * Network chunks split events anywhere, even mid-line, so text is buffered
 * until a blank line ends an event. Only `data:` lines are used: the backend
 * repeats the event type inside the JSON payload.
 */
export class SSEParser<T = unknown> {
  private buffer = ''

  /** Feed a chunk of the stream; returns the events it completed. */
  push(chunk: string): T[] {
    this.buffer += chunk.replace(/\r\n?/g, '\n')
    const events: T[] = []
    let end: number
    while ((end = this.buffer.indexOf('\n\n')) !== -1) {
      const block = this.buffer.slice(0, end)
      this.buffer = this.buffer.slice(end + 2)
      const data = block
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).replace(/^ /, ''))
        .join('\n')
      if (data) events.push(JSON.parse(data) as T)
    }
    return events
  }
}

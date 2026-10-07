import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import Markdown from 'react-markdown'
import { Link } from 'react-router'
import remarkGfm from 'remark-gfm'

import { ArtifactView } from './ArtifactView'
import { SandboxStatus } from './SandboxStatus'
import type { Message, Step } from './state'
import { Steps } from './Steps'
import { useChat } from './useChat'

const EXAMPLES = [
  'Which video category gets the most views per video?',
  'Make an interactive chart of trending videos per week.',
  'What datasets do I have?',
]

/** Analyses this turn saved or changed: the save_analysis calls that succeeded. */
function savedAnalyses(steps: Step[]): { id: string; title: string; version: number | null }[] {
  return steps.flatMap((s) => {
    if (s.kind !== 'tool' || s.name !== 'save_analysis' || !s.result) return []
    const saved = /^Saved analysis ([a-z0-9]+):/.exec(s.result)
    const updated = /^Updated analysis ([a-z0-9]+) to version (\d+):/.exec(s.result)
    const id = saved?.[1] ?? updated?.[1]
    return id ? [{ id, title: String(s.args.title ?? 'analysis'), version: updated ? Number(updated[2]) : null }] : []
  })
}

function MessageView({ message }: { message: Message }) {
  if (message.role === 'user') return <div className="message message-user">{message.text}</div>

  const streaming = message.status === 'streaming'
  return (
    <div className="message message-assistant">
      {message.notices?.map((notice, i) => (
        <p key={i} className="notice" role="status">
          {notice}
        </p>
      ))}
      <Steps steps={message.steps} streaming={streaming} />
      {message.text && (
        <div className="answer">
          {/* react-markdown renders no raw HTML: the model's text can't inject markup. */}
          <Markdown remarkPlugins={[remarkGfm]}>{message.text}</Markdown>
        </div>
      )}
      {message.artifacts.map((a) => (
        <ArtifactView key={a.url} artifact={a} />
      ))}
      {savedAnalyses(message.steps).map((a) => (
        <p key={a.id} className="saved-analysis" role="status">
          ✓ {a.version ? `Updated “${a.title}” to version ${a.version}` : `Saved “${a.title}”`}.{' '}
          <Link to={`/analyses/${a.id}`}>Open it in Analyses</Link> to run it again later.
        </p>
      ))}
      {streaming && !message.text && message.steps.length === 0 && <p className="muted">Starting…</p>}
      {message.status === 'error' && <p className="error">⚠ {message.error}</p>}
    </div>
  )
}

export function ChatPage() {
  const { messages, busy, threadId, send, newConversation } = useChat()
  const [draft, setDraft] = useState('')
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const submit = () => {
    if (busy || !draft.trim()) return
    void send(draft)
    setDraft('')
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <section className="chat">
      <header className="chat-header">
        <h1>Chat</h1>
        <button type="button" className="secondary" onClick={newConversation} disabled={busy || messages.length === 0}>
          New conversation
        </button>
      </header>
      <SandboxStatus refreshKey={busy} />

      <div className="messages">
        {messages.length === 0 ? (
          <div className="empty">
            <p>Ask about the data. The agent queries it, runs analysis in a sandbox and draws charts.</p>
            <div className="examples">
              {EXAMPLES.map((example) => (
                <button key={example} type="button" className="secondary" onClick={() => setDraft(example)}>
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => <MessageView key={m.id} message={m} />)
        )}
        <div ref={bottom} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={busy ? 'Answering…' : 'Ask a question (Enter to send, Shift+Enter for a new line)'}
          aria-label="Message"
          rows={2}
        />
        <button type="submit" disabled={busy || !draft.trim()}>
          Send
        </button>
      </form>
      {threadId && <p className="thread-id muted">conversation {threadId.slice(0, 8)}</p>}
    </section>
  )
}

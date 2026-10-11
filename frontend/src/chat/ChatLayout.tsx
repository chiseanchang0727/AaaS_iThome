import { useCallback, useEffect, useState } from 'react'
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router'

import { deletePastConversation, listPastConversations } from '../api/client'
import type { PastConversationSummary } from '../api/types'
import { ChatPage } from './ChatPage'
import { groupByDay } from './conversationGroups'
import { PastConversationView } from './PastConversations'
import { useChat } from './useChat'

function Sidebar({
  current,
  currentTitle,
  busy,
  onNew,
  onDeletedCurrent,
  open,
  onClose,
}: {
  current: string | null
  currentTitle: string | null
  busy: boolean
  onNew: () => void
  onDeletedCurrent: () => void
  open: boolean
  onClose: () => void
}) {
  const [items, setItems] = useState<PastConversationSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirming, setConfirming] = useState<string | null>(null)
  const location = useLocation()
  const navigate = useNavigate()
  const opened = /^\/chat\/past\/([^/]+)/.exec(location.pathname)?.[1] ?? null
  const onLive = location.pathname === '/chat' || location.pathname === '/chat/'

  // Load the list, and again whenever a turn finishes: it is in the history then.
  useEffect(() => {
    if (busy) return
    listPastConversations()
      .then((list) => {
        setItems(list)
        setError(null)
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [busy])

  const remove = async (id: string) => {
    try {
      await deletePastConversation(id)
      setItems((list) => list?.filter((c) => c.id !== id) ?? null)
      if (id === current) onDeletedCurrent()
      if (id === opened || id === current) navigate('/chat')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setConfirming(null)
    }
  }

  const showCurrent = current !== null && currentTitle !== null && !items?.some((c) => c.id === current)
  return (
    <aside className={open ? 'chat-sidebar open' : 'chat-sidebar'} aria-label="Conversations">
      <button className="new-chat" onClick={onNew}>
        + New chat
      </button>
      {error && <p className="error small">{error}</p>}
      {items === null && !error && <p className="muted small">Loading…</p>}
      <nav className="conversation-list">
        {showCurrent && (
          <div className="conversation-group">
            <p className="group-label">Now</p>
            <Link to="/chat" className={onLive ? 'conversation active' : 'conversation'} onClick={onClose}>
              {currentTitle}
            </Link>
          </div>
        )}
        {groupByDay(items ?? []).map((group) => (
          <div key={group.label} className="conversation-group">
            <p className="group-label">{group.label}</p>
            {group.items.map((c) => {
              const live = c.id === current
              const active = live ? onLive : c.id === opened
              return (
                <div key={c.id} className={active ? 'conversation-row active' : 'conversation-row'}>
                  <Link
                    to={live ? '/chat' : `/chat/past/${c.id}`}
                    className={active ? 'conversation active' : 'conversation'}
                    title={c.title}
                    onClick={onClose}
                    aria-current={active ? 'page' : undefined}
                  >
                    <span className="conversation-title">{c.title}</span>
                  </Link>
                  {confirming === c.id ? (
                    <span className="row-confirm" role="group" aria-label={`Confirm deleting ${c.title}`}>
                      <button className="danger small" onClick={() => remove(c.id)}>Delete</button>
                      <button className="link-button" onClick={() => setConfirming(null)}>Cancel</button>
                    </span>
                  ) : (
                    <button
                      className="icon-button"
                      onClick={() => setConfirming(c.id)}
                      aria-label={`Delete ${c.title}…`}
                      title="Delete"
                    >
                      ×
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        ))}
      </nav>
    </aside>
  )
}

/**
 * The chat with its conversations on the side, like a chat app: the live
 * conversation on the right, past ones opened read-only in its place. The
 * live chat's state lives here, so opening an old conversation and coming
 * back keeps it.
 */
export function ChatLayout() {
  const chat = useChat()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)

  const startNew = useCallback(() => {
    chat.newConversation()
    setOpen(false)
    navigate('/chat')
  }, [chat, navigate])

  const firstQuestion = chat.messages.find((m) => m.role === 'user')?.text ?? null
  return (
    <div className="chat-layout">
      <Sidebar
        current={chat.threadId}
        currentTitle={firstQuestion}
        busy={chat.busy}
        onNew={startNew}
        onDeletedCurrent={() => chat.newConversation()}
        open={open}
        onClose={() => setOpen(false)}
      />
      {open && <div className="sidebar-backdrop" onClick={() => setOpen(false)} aria-hidden="true" />}
      <div className="chat-main">
        <button className="sidebar-toggle secondary" onClick={() => setOpen(true)}>
          ☰ Conversations
        </button>
        <Routes>
          <Route index element={<ChatPage chat={chat} />} />
          <Route path="past/:id" element={<PastConversationView />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </div>
    </div>
  )
}

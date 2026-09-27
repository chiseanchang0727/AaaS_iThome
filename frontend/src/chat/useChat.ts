import { useCallback, useReducer, useRef } from 'react'

import { endConversation, streamChat } from '../api/client'
import { chatReducer, initialState } from './state'

/** Chat state plus `send` and `newConversation`, backed by the streaming API. */
export function useChat() {
  const [state, dispatch] = useReducer(chatReducer, initialState)
  // The id arrives mid-stream (the "thread" event); a ref gives `send` the
  // latest without re-creating it.
  const threadRef = useRef<string | null>(null)

  const send = useCallback(async (text: string) => {
    const message = text.trim()
    if (!message) return
    dispatch({ type: 'send', text: message })
    try {
      await streamChat(message, threadRef.current, (event) => {
        if (event.type === 'thread') threadRef.current = event.thread_id
        dispatch({ type: 'event', event })
      })
    } catch (e) {
      dispatch({ type: 'failed', message: e instanceof Error ? e.message : String(e) })
    }
  }, [])

  const newConversation = useCallback(() => {
    // Free the old conversation's sandbox now rather than when it idles out.
    if (threadRef.current) void endConversation(threadRef.current)
    threadRef.current = null
    dispatch({ type: 'reset' })
  }, [])

  return { ...state, send, newConversation }
}

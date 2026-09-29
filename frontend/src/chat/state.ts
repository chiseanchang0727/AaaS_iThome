import type { ChatEvent } from '../api/types'

/** One thing the agent did while answering: a thought, or a tool call and its result. */
export type Step =
  | { kind: 'thinking'; text: string }
  | { kind: 'tool'; id: string; name: string; args: Record<string, unknown>; result?: string; error?: boolean }

export interface Artifact {
  name: string
  url: string
  kind: string
}

export interface Message {
  id: number
  role: 'user' | 'assistant'
  text: string
  steps: Step[]
  artifacts: Artifact[]
  status: 'streaming' | 'done' | 'error'
  error?: string
}

export interface ChatState {
  threadId: string | null
  messages: Message[]
  busy: boolean
  nextId: number
}

export type Action =
  | { type: 'send'; text: string }
  | { type: 'event'; event: ChatEvent }
  | { type: 'failed'; message: string }
  | { type: 'reset' }

export const initialState: ChatState = { threadId: null, messages: [], busy: false, nextId: 1 }

/** Apply `update` to the message being answered (the last one). */
function updateLast(state: ChatState, update: (m: Message) => Message): ChatState {
  const last = state.messages.at(-1)
  if (!last || last.role !== 'assistant') return state
  return { ...state, messages: [...state.messages.slice(0, -1), update(last)] }
}

export function chatReducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case 'send': {
      const user: Message = {
        id: state.nextId, role: 'user', text: action.text, steps: [], artifacts: [], status: 'done',
      }
      const reply: Message = {
        id: state.nextId + 1, role: 'assistant', text: '', steps: [], artifacts: [], status: 'streaming',
      }
      return { ...state, messages: [...state.messages, user, reply], busy: true, nextId: state.nextId + 2 }
    }

    case 'event': {
      const event = action.event
      switch (event.type) {
        case 'thread':
          return { ...state, threadId: event.thread_id }
        case 'thinking':
          return updateLast(state, (m) => ({ ...m, steps: [...m.steps, { kind: 'thinking', text: event.text }] }))
        case 'tool_call':
          return updateLast(state, (m) => ({
            ...m,
            steps: [...m.steps, { kind: 'tool', id: event.id, name: event.name, args: event.args }],
          }))
        case 'tool_result':
          return updateLast(state, (m) => ({
            ...m,
            steps: m.steps.map((s) =>
              s.kind === 'tool' && s.id === event.id ? { ...s, result: event.content, error: event.error } : s,
            ),
          }))
        case 'answer':
          return updateLast(state, (m) => ({ ...m, text: event.text }))
        case 'artifact':
          return updateLast(state, (m) => ({
            ...m,
            artifacts: [...m.artifacts, { name: event.name, url: event.url, kind: event.kind }],
          }))
        case 'error':
          return updateLast(state, (m) => ({ ...m, status: 'error', error: event.message }))
        case 'done':
          return {
            ...updateLast(state, (m) => (m.status === 'streaming' ? { ...m, status: 'done' } : m)),
            busy: false,
          }
      }
      return state
    }

    case 'failed':
      return {
        ...updateLast(state, (m) => ({ ...m, status: 'error', error: action.message })),
        busy: false,
      }

    case 'reset':
      return initialState
  }
}

// Shapes of what the backend sends. Kept in step with backend/api/.

/** One Server-Sent Event from POST /api/chat (backend/api/events.py). */
export type ChatEvent =
  | { type: 'thread'; thread_id: string }
  | { type: 'thinking'; text: string }
  | { type: 'tool_call'; id: string; name: string; args: Record<string, unknown> }
  | { type: 'tool_result'; id: string; name: string | null; content: string; error: boolean }
  | { type: 'answer'; text: string }
  | { type: 'artifact'; name: string; url: string; kind: string }
  | { type: 'error'; message: string }
  | { type: 'notice'; message: string }
  | { type: 'done' }

/** GET /api/sandboxes (ConversationManager.status). */
export interface SandboxStatus {
  enabled: boolean
  provider: string | null
  max: number | null
  in_use: number
  busy: number
  idle: number
  starting: number
  warm: number
  warming: number
  waiting: number
  idle_minutes: number
}

export interface Column {
  name: string
  type: string
}

/** POST /api/uploads: a parsed file waiting to be kept. */
export interface StagedUpload {
  upload_id: string
  filename: string
  rows: number
  columns: Column[]
  preview: Record<string, unknown>[]
  suggested_name: string
}

export type DatasetKind = 'table' | 'file'

/** GET /api/datasets entries. The built-in videos table has no columns listed. */
export interface Dataset {
  name: string
  kind: DatasetKind
  built_in: boolean
  description?: string
  rows?: number
  columns?: Column[]
  source?: string
  created_at?: string
}

// --- evals (backend/api/evals.py, written by backend/evals/memory/run.py) ----

/** Totals for one arm of an eval run. */
export interface ArmSummary {
  turns: number
  checked: number
  correct: number
  grounded: number
  steps: number
  queries: number
  skill_reads: number
  tokens_per_turn: number
  first_call_per_turn: number
}

/** GET /api/evals/context/runs entries. */
export interface EvalRunInfo {
  id: string
  created_at: string
  model: string
  arms: string[]
  repeats: number
  conversations: string[]
  summary: Record<string, ArmSummary>
}

/** One turn of one arm: what was asked, what came back, what it cost. */
export interface EvalTurnResult {
  conversation: string
  turn: string
  arm: string
  run: number
  prompt: string
  answer: string
  stopped: boolean
  /** Turn numbers the context filter sent; null for arms without one. */
  sent_turns: number[] | null
  steps: number
  queries: number
  skill_reads: number
  input_tokens: number
  first_call_tokens: number
  /** null: nothing to check automatically (small talk). */
  correct: boolean | null
  missing: string[]
  ungrounded: string[]
}

export interface EvalConversation {
  id: string
  about: string
  turns: { id: string; prompt: string; expect: string[]; note: string }[]
}

/** GET /api/evals/context/runs/{id}. */
export interface EvalRun {
  id: string
  created_at: string
  model: string
  sandbox: boolean
  repeats: number
  arms: string[]
  context_filter: Record<string, unknown> | null
  conversations: EvalConversation[]
  results: EvalTurnResult[]
  summary: Record<string, ArmSummary>
}

/** One line of a conversation's history (backend/api/history.py). */
export interface HistoryLine {
  id?: string
  previous?: string | null
  turn: number
  role: 'user' | 'assistant' | 'tool'
  content: string
  tool_calls?: { id: string; name: string; args: Record<string, unknown> }[]
  tool_call_id?: string
  name?: string | null
  error?: boolean
  ts?: string
}

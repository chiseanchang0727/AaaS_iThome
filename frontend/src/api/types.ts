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
  | { type: 'done' }

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

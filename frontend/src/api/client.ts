import { accountHeaders } from './account'
import { SSEParser } from './sse'
import type {
  ChatEvent,
  Dataset,
  DatasetKind,
  EvalRun,
  EvalRunInfo,
  HistoryLine,
  SandboxStatus,
  StagedUpload,
} from './types'

/** An error the backend explained; `message` is fit to show the user. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function failure(response: Response): Promise<ApiError> {
  let message = `${response.status} ${response.statusText}`
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') message = body.detail
    else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join('; ')
  } catch {
    // not JSON: keep the status line
  }
  return new ApiError(response.status, message)
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) throw await failure(response)
  return (await response.json()) as T
}

/**
 * Send one message and call `onEvent` for each event as it streams in.
 * Resolves when the stream ends. A new conversation starts when `threadId`
 * is null; its id arrives in the first ("thread") event.
 */
export async function streamChat(
  message: string,
  threadId: string | null,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...accountHeaders() },
    body: JSON.stringify(threadId ? { message, thread_id: threadId } : { message }),
    signal,
  })
  if (!response.ok || !response.body) throw await failure(response)

  const parser = new SSEParser<ChatEvent>()
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    for (const event of parser.push(value)) onEvent(event)
  }
}

export async function getSandboxStatus(): Promise<SandboxStatus> {
  return json(await fetch('/api/sandboxes', { headers: accountHeaders() }))
}

export async function endConversation(threadId: string): Promise<void> {
  await fetch(`/api/conversations/${encodeURIComponent(threadId)}`, { method: 'DELETE', headers: accountHeaders() })
}

export async function uploadFile(file: File): Promise<StagedUpload> {
  const form = new FormData()
  form.append('file', file)
  return json(await fetch('/api/uploads', { method: 'POST', body: form }))
}

export async function commitDataset(uploadId: string, kind: DatasetKind, name: string): Promise<Dataset> {
  return json(
    await fetch('/api/datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upload_id: uploadId, kind, name }),
    }),
  )
}

export async function listDatasets(): Promise<Dataset[]> {
  return json(await fetch('/api/datasets'))
}

export async function deleteDataset(name: string): Promise<void> {
  await json(await fetch(`/api/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' }))
}

export async function listEvalRuns(): Promise<EvalRunInfo[]> {
  return json(await fetch('/api/evals/context/runs'))
}

export async function getEvalRun(id: string): Promise<EvalRun> {
  return json(await fetch(`/api/evals/context/runs/${encodeURIComponent(id)}`))
}

export async function getEvalHistory(runId: string, conversation: string, arm: string, repeat = 1): Promise<HistoryLine[]> {
  const path = [runId, 'history', conversation, arm].map(encodeURIComponent).join('/')
  return json(await fetch(`/api/evals/context/runs/${path}?repeat=${repeat}`))
}

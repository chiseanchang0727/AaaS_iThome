import { accountHeaders } from './account'
import { SSEParser } from './sse'
import type {
  Analysis,
  AnalysisRun,
  AnalysisSummary,
  SourceOption,
  ChatEvent,
  Dataset,
  DatasetKind,
  EvalRun,
  EvalRunInfo,
  HistoryLine,
  ConversationTimeline,
  Load,
  SandboxStatus,
  StagedUpload,
  ComparePreview,
  CompareRun,
  CompareRunInfo,
  HistoryRunPreview,
  SystemRun,
  SystemRunInfo,
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

export async function getLoad(account: string | null): Promise<Load> {
  return json(await fetch(account ? `/api/load?account=${encodeURIComponent(account)}` : '/api/load'))
}

export async function getConversationTimeline(id: string): Promise<ConversationTimeline> {
  return json(await fetch(`/api/load/conversations/${encodeURIComponent(id)}`))
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

export async function listSystemRuns(): Promise<SystemRunInfo[]> {
  return json(await fetch('/api/evals/system/runs'))
}

export async function getSystemRun(id: string): Promise<SystemRun> {
  return json(await fetch(`/api/evals/system/runs/${encodeURIComponent(id)}`))
}

export async function getSystemHistory(runId: string, caseId: string): Promise<HistoryLine[]> {
  return json(await fetch(`/api/evals/system/runs/${encodeURIComponent(runId)}/history/${encodeURIComponent(caseId)}`))
}

export async function getHistoryRunPreview(): Promise<HistoryRunPreview> {
  return json(await fetch('/api/evals/system/history-runs'))
}

/** Start judging every saved conversation turn. Resolves with the new run's id once it has started. */
export async function startHistoryRun(): Promise<{ id: string; total: number }> {
  return json(await fetch('/api/evals/system/history-runs', { method: 'POST' }))
}

export async function getComparePreview(): Promise<ComparePreview> {
  return json(await fetch('/api/evals/compare/pairs'))
}

export async function startCompareRun(): Promise<{ id: string }> {
  return json(await fetch('/api/evals/compare/runs', { method: 'POST' }))
}

export async function listCompareRuns(): Promise<CompareRunInfo[]> {
  return json(await fetch('/api/evals/compare/runs'))
}

export async function getCompareRun(id: string): Promise<CompareRun> {
  return json(await fetch(`/api/evals/compare/runs/${encodeURIComponent(id)}`))
}

export async function getCompareHistory(runId: string, side: 'full' | 'jev', thread: string): Promise<HistoryLine[]> {
  const path = [runId, 'history', side, thread].map(encodeURIComponent).join('/')
  return json(await fetch(`/api/evals/compare/runs/${path}`))
}

export async function listAnalyses(): Promise<AnalysisSummary[]> {
  return json(await fetch('/api/analyses'))
}

export async function getAnalysis(id: string): Promise<Analysis> {
  return json(await fetch(`/api/analyses/${encodeURIComponent(id)}`))
}

/**
 * Run a saved analysis again; resolves with the new run (status "running").
 * `sources` reads other tables with the same columns: {"videos": "videos_ca"}.
 */
export async function runAnalysis(id: string, sources: Record<string, string> = {}): Promise<AnalysisRun> {
  return json(
    await fetch(`/api/analyses/${encodeURIComponent(id)}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sources }),
    }),
  )
}

export async function getAnalysisSources(id: string): Promise<SourceOption[]> {
  return json(await fetch(`/api/analyses/${encodeURIComponent(id)}/sources`))
}

export async function deleteAnalysis(id: string): Promise<void> {
  await json(await fetch(`/api/analyses/${encodeURIComponent(id)}`, { method: 'DELETE' }))
}

export function analysisOutputUrl(id: string, runId: string, file: string): string {
  return `/api/analyses/${[id, 'runs', runId, 'files', file].map(encodeURIComponent).join('/')}`
}

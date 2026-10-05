import { Fragment, useCallback, useEffect, useState, type ReactNode } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { getHistoryRunPreview, getSystemHistory, getSystemRun, listSystemRuns, startHistoryRun } from '../api/client'
import type {
  EvalStatus,
  HistoryLine,
  HistoryRunPreview,
  SystemCase,
  SystemResult,
  SystemRun,
  SystemRunInfo,
} from '../api/types'
import { Steps } from '../chat/Steps'
import { turnSteps } from './results'

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

const STATUS: Record<EvalStatus, { mark: string; className: string; title: string }> = {
  pass: { mark: '✓', className: 'ok', title: 'pass' },
  fail: { mark: '✗', className: 'bad', title: 'fail' },
  unknown: { mark: '?', className: 'none', title: 'unknown: not enough to tell, or not judged' },
  judge_error: { mark: '!', className: 'bad', title: 'the judge call failed' },
}

export function Status({ status }: { status: EvalStatus }) {
  const s = STATUS[status] ?? STATUS.unknown
  return (
    <span className={`verdict ${s.className}`} title={s.title} aria-label={s.title}>
      {s.mark}
    </span>
  )
}

function seconds(s: number): string {
  return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

function List({ items, empty = 'none' }: { items: string[]; empty?: string }) {
  if (items.length === 0) return <span className="muted">{empty}</span>
  return (
    <ul className="plain-list">
      {items.map((item, i) => (
        <li key={i}>{item}</li>
      ))}
    </ul>
  )
}

function Verdict({ title, judged, children }: { title: string; judged: { status: EvalStatus; reason: string }; children?: ReactNode }) {
  return (
    <section className="arm-detail">
      <h3>
        <Status status={judged.status} /> {title}
      </h3>
      <p className="reason">{judged.reason}</p>
      {children}
    </section>
  )
}

/** SQL the agent ran, from the history (query_database and export_query calls). */
function queries(lines: HistoryLine[], turn: number): string[] {
  return lines
    .filter((l) => l.turn === turn && l.role === 'assistant')
    .flatMap((l) => l.tool_calls ?? [])
    .filter((c) => c.name === 'query_database' || c.name === 'export_query')
    .map((c) => String(c.args.sql ?? ''))
}

type History = HistoryLine[] | 'loading' | { error: string }

function CaseDetail({ spec, result, history }: { spec: SystemCase | undefined; result: SystemResult; history: History | undefined }) {
  const turn = (spec?.setup.length ?? 0) + 1
  const expected = Object.entries(spec?.expected_values ?? {}).map(([k, v]) =>
    `${k} = ${typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: 6 }) : v}`,
  )
  const e = result.efficiency
  const r = result.recovery
  return (
    <div className="case-detail">
      <section className="arm-detail">
        <h3>Question</h3>
        {spec && spec.setup.length > 0 && (
          <p className="muted">Asked first: {spec.setup.join(' → ')}</p>
        )}
        <p>{result.question}</p>
        {(expected.length > 0 || spec?.expected_behavior) && (
          <>
            <h3>Expected</h3>
            {expected.length > 0 && <List items={expected} />}
            {spec?.expected_behavior && <p>{spec.expected_behavior}</p>}
          </>
        )}
        {spec?.note && <p className="muted">{spec.note}</p>}
        <h3>Final answer</h3>
        <div className="answer">
          {result.stopped ? (
            <p className="error">Stopped before answering.</p>
          ) : (
            <Markdown remarkPlugins={[remarkGfm]}>{result.final_answer}</Markdown>
          )}
        </div>
        {result.stream_errors.length > 0 && <p className="error">{result.stream_errors.join('; ')}</p>}
      </section>

      <div className="turn-detail">
        <Verdict title="Correctness" judged={result.correct}>
          {(result.correct.expected_values.found.length > 0 || result.correct.expected_values.missing.length > 0) && (
            <dl className="facts">
              <dt>Found</dt>
              <dd><List items={result.correct.expected_values.found} /></dd>
              <dt>Missing</dt>
              <dd className={result.correct.expected_values.missing.length ? 'error' : ''}>
                <List items={result.correct.expected_values.missing} />
              </dd>
            </dl>
          )}
        </Verdict>
        <Verdict title="Groundedness" judged={result.grounded}>
          <dl className="facts">
            <dt>Ungrounded values</dt>
            <dd><List items={result.grounded.ungrounded_values} /></dd>
            <dt>Unsupported claims</dt>
            <dd>
              <List items={result.grounded.unsupported_claims.map((c) => `“${c.claim}”: ${c.reason}`)} />
            </dd>
          </dl>
        </Verdict>
        <Verdict title="Instruction following" judged={result.instruction_following}>
          {result.instruction_following.checks.length > 0 && (
            <List
              items={result.instruction_following.checks.map(
                (c) => `${c.pass === null ? '?' : c.pass ? '✓' : '✗'} ${c.check}${c.evidence?.length ? ` (${c.evidence.join(', ')})` : ''}`,
              )}
            />
          )}
        </Verdict>
        <Verdict title="Execution strategy" judged={result.execution_strategy} />
      </div>

      <section className="arm-detail">
        <h3>Tool trace</h3>
        <dl className="facts">
          <dt>Tools</dt>
          <dd>{Object.entries(result.tool_usage.tools).map(([n, c]) => `${n} ×${c}`).join(', ') || 'none'}</dd>
          <dt>Skills read</dt>
          <dd>{result.tool_usage.skills_read.join(', ') || 'none'}</dd>
          {result.context && (
            <>
              <dt>Context (Jev)</dt>
              <dd>
                {result.context.earlier_turns === 0
                  ? 'first turn'
                  : `sent ${result.context.sent_turns.length ? `turn ${result.context.sent_turns.join(', ')}` : 'no earlier turn'} of ${result.context.earlier_turns}`}{' '}
                <span className="muted">({seconds(result.context.seconds)})</span>
                {result.context.error && <span className="error"> Jev failed, sent everything: {result.context.error}</span>}
              </dd>
            </>
          )}
          {result.artifacts.length > 0 && (
            <>
              <dt>Files</dt>
              <dd>{result.artifacts.join(', ')}</dd>
            </>
          )}
        </dl>
        {history === undefined || history === 'loading' ? (
          <p className="muted">Loading what it did…</p>
        ) : 'error' in history ? (
          <p className="error">{history.error}</p>
        ) : (
          <>
            <Steps steps={turnSteps(history, turn).steps} streaming={false} />
            <h3>Queries</h3>
            {queries(history, turn).length === 0 ? (
              <p className="muted">none</p>
            ) : (
              <div aria-label="Queries">
                {queries(history, turn).map((sql, i) => (
                  <pre key={i} className="code">{sql}</pre>
                ))}
              </div>
            )}
          </>
        )}

        <h3>Code executions</h3>
        {r.code_steps.length === 0 ? (
          <p className="muted">none</p>
        ) : (
          <div className="table-scroll">
            <table className="turns" aria-label="Code executions">
              <thead>
                <tr>
                  <th>Command</th>
                  <th>Exit</th>
                  <th>Time</th>
                  <th>Peak memory</th>
                  <th>Attempt</th>
                </tr>
              </thead>
              <tbody>
                {r.code_steps.map((s, i) => (
                  <tr key={i}>
                    <td><code>{s.command}</code></td>
                    <td className={s.exit_code ? 'error' : ''}>{s.exit_code ?? '—'}</td>
                    <td>{s.seconds !== null ? seconds(s.seconds) : '—'}</td>
                    <td>
                      {s.peak_memory_mb !== null ? `${Math.round(s.peak_memory_mb)} MB` : '—'}
                      {s.memory_limit_mb ? <span className="muted"> / {s.memory_limit_mb}</span> : null}
                      {s.out_of_memory && <span className="badge">out of memory</span>}
                    </td>
                    <td>{s.strategy ?? ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <h3>Errors and recovery</h3>
        <dl className="facts">
          <dt>Retries</dt>
          <dd>{r.retries} ({r.identical_retries} identical)</dd>
          <dt>Out of memory</dt>
          <dd>{r.oom_events}{r.oom_resolved_by ? `, resolved by ${r.oom_resolved_by}` : ''}</dd>
          <dt>Sandbox moves</dt>
          <dd>{r.sandbox_moves}</dd>
          {r.sandbox_events.length > 0 && (
            <>
              <dt>Sandbox events</dt>
              <dd>{r.sandbox_events.join(' → ')}</dd>
            </>
          )}
        </dl>
        {r.errors.map((err, i) => (
          <div key={i} className="recovery-error">
            <p>
              <b>{err.tool}</b> failed <span className="badge">next: {err.next}</span>
            </p>
            <pre className="code">{err.args}</pre>
            <pre className="code error">{err.result}</pre>
          </div>
        ))}

        <h3>Token usage and runtime</h3>
        <dl className="facts">
          <dt>Model calls</dt>
          <dd>{e.model_calls}</dd>
          <dt>Tokens</dt>
          <dd>
            {e.input_tokens.toLocaleString()} in, {e.output_tokens.toLocaleString()} out,{' '}
            {e.total_tokens.toLocaleString()} total
          </dd>
          <dt>Runtime</dt>
          <dd>{seconds(e.runtime_seconds)}</dd>
        </dl>
      </section>
    </div>
  )
}

const POLL_MS = 3000

/** What earlier turns the agent was given: Jev's pick, or the whole conversation. */
function ContextCell({ context }: { context: SystemResult['context'] }) {
  if (context === undefined) return <span className="muted" title="Evaluated before this was recorded">—</span>
  if (context === null) return <span className="muted" title="The whole conversation, through the checkpointer">full</span>
  const text = context.earlier_turns === 0 ? 'first turn' : `sent ${context.sent_turns.length} of ${context.earlier_turns}`
  return (
    <span className="chip jev" title={`Jev picked the earlier turns: ${context.sent_turns.join(', ') || 'none'}`}>
      Jev · {text}
    </span>
  )
}

function clip(text: string, limit = 90): string {
  return text.length <= limit ? text : `${text.slice(0, limit)}…`
}

/** Judges every saved conversation turn. Asks first: each turn costs four judge calls. */
function EvaluateHistory({ onStarted }: { onStarted: (id: string) => void }) {
  const [preview, setPreview] = useState<HistoryRunPreview | null>(null)
  const [asking, setAsking] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(() => {
    getHistoryRunPreview().then(setPreview).catch((e) => setError(errorText(e)))
  }, [])
  useEffect(refresh, [refresh])

  // While a run goes on, check now and then so the button comes back when it ends.
  const running = preview?.running ?? null
  useEffect(() => {
    if (!running) return
    const timer = setTimeout(refresh, POLL_MS)
    return () => clearTimeout(timer)
  }, [running, preview, refresh])

  if (!preview?.available) return null

  const start = async () => {
    setBusy(true)
    setError(null)
    try {
      const { id } = await startHistoryRun()
      setAsking(false)
      onStarted(id)
      refresh()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="history-run">
      {asking ? (
        <div className="confirm" role="group" aria-label="Confirm evaluation">
          <span>
            Judge {preview.turns} turn{preview.turns === 1 ? '' : 's'} from {preview.conversations} conversation
            {preview.conversations === 1 ? '' : 's'}? Each turn uses 4 judge model calls.
          </span>
          <button onClick={start} disabled={busy}>
            {busy ? 'Starting…' : 'Start'}
          </button>
          <button className="secondary" onClick={() => setAsking(false)} disabled={busy}>
            Cancel
          </button>
        </div>
      ) : (
        <button
          onClick={() => {
            refresh()
            setAsking(true)
          }}
          disabled={preview.running !== null || preview.turns === 0}
          title={preview.running ? 'An evaluation of the history is already running' : undefined}
        >
          {preview.running ? 'Evaluating history…' : 'Evaluate all conversation history'}
        </button>
      )}
      {preview.turns === 0 && !preview.running && (
        <p className="muted hint">
          No conversations to evaluate. Copy history files into <code>{preview.folder}</code>.
        </p>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  )
}

export function SystemEvalPage() {
  const [runs, setRuns] = useState<SystemRunInfo[] | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [run, setRun] = useState<SystemRun | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const [histories, setHistories] = useState<Record<string, History>>({})
  const [error, setError] = useState<string | null>(null)

  const loadRuns = useCallback((select?: string) => {
    listSystemRuns()
      .then((list) => {
        setRuns(list)
        setRunId((current) => select ?? current ?? list[0]?.id ?? null)
      })
      .catch((e) => setError(errorText(e)))
  }, [])

  useEffect(() => loadRuns(), [loadRuns])

  useEffect(() => {
    if (!runId) return
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    let polled = false
    const load = () =>
      getSystemRun(runId)
        .then((loaded) => {
          if (!current) return
          setRun(loaded)
          if (loaded.status === 'running') {
            polled = true
            timer = setTimeout(load, POLL_MS)
          } else if (polled) {
            loadRuns() // it just finished: its totals in the run list
          }
        })
        .catch((e) => current && setError(errorText(e)))
    load()
    return () => {
      current = false
      clearTimeout(timer)
    }
  }, [runId, loadRuns])

  const selectRun = (id: string) => {
    setRunId(id)
    setRun(null)
    setOpen(null)
    setHistories({})
  }

  const started = (id: string) => {
    selectRun(id)
    loadRuns(id)
  }

  const toggle = (caseId: string) => {
    const opening = open !== caseId
    setOpen(opening ? caseId : null)
    if (!opening || !run || histories[caseId]) return
    setHistories((h) => ({ ...h, [caseId]: 'loading' }))
    getSystemHistory(run.id, caseId)
      .then((lines) => setHistories((h) => ({ ...h, [caseId]: lines })))
      .catch((e) => setHistories((h) => ({ ...h, [caseId]: { error: errorText(e) } })))
  }

  if (error) return <p className="error">{error}</p>
  if (runs === null) return <p className="muted">Loading eval runs…</p>

  const s = run?.summary
  const fromHistory = run?.source === 'history'
  const anyJev = run?.results.some((r) => r.context) ?? false
  return (
    <div className="eval-page">
      <header className="eval-header">
        <div>
          <h1>Whole agent system</h1>
          <p className="muted">
            Each case runs through the real app (prompt, tools, skills, sandbox, recovery). Code checks what happened;
            LLM judges decide whether it was good. Saved conversations can be judged too, as they happened.
          </p>
        </div>
        <div className="controls">
          <EvaluateHistory onStarted={started} />
          {runs.length > 0 && (
            <label className="field">
              Run
              <select value={runId ?? ''} onChange={(e) => selectRun(e.target.value)}>
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {when(r.created_at)} · {r.source === 'history' ? 'history, ' : ''}
                    {r.total} {r.source === 'history' ? 'turn' : 'case'}
                    {r.total === 1 ? '' : 's'}
                    {r.status === 'running' ? ' (running)' : r.status === 'failed' ? ' (failed)' : ''}
                    {r.rejudged_from ? ' (re-judged)' : ''}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </header>

      {runs.length === 0 ? (
        <div className="panel">
          <h2>No system eval runs yet</h2>
          <p className="muted">
            Judge the saved conversations with the button above, or run the cases from <code>backend/</code>:{' '}
            <code>uv run --env-file ../.env python -m evals.system.run --only lookup_counts</code>
          </p>
        </div>
      ) : !run || !s ? (
        <p className="muted">Loading run…</p>
      ) : (
        <>
          <p className="run-meta muted">
            {fromHistory ? 'saved conversations' : `agent ${run.model}`} · judge{' '}
            {run.judge_model ?? 'none (code checks only)'}
            {!fromHistory && ` · ${run.sandbox ? 'sandbox' : 'no sandbox'}`}
            {run.rejudged_from && <span className="badge">re-judged from {run.rejudged_from}</span>}
            <span className="badge">
              correct {s.correct}/{s.cases}
            </span>
            <span className="badge">
              grounded {s.grounded}/{s.cases}
            </span>
            <span className="badge">
              instructions {s.instruction_following}/{s.cases}
            </span>
            <span className="badge">
              strategy {s.execution_strategy}/{s.cases}
            </span>
            {s.judge_errors > 0 && <span className="badge error">{s.judge_errors} judge errors</span>}
            {run.results.some((r) => r.context) && (
              <span className="badge">
                Jev context {run.results.filter((r) => r.context).length}/{run.results.length} turns
              </span>
            )}
          </p>
          {run.status === 'running' && (
            <p className="progress" role="status">
              Evaluating… {run.results.length} / {run.total ?? '?'}
            </p>
          )}
          {run.status === 'failed' && <p className="error">The run stopped with an error: {run.error ?? 'unknown'}</p>}

          <div className="table-scroll">
            <table className="turns" aria-label="Cases">
              <thead>
                <tr>
                  <th>{fromHistory ? 'Question' : 'Case'}</th>
                  {anyJev && <th>Context</th>}
                  <th>Correct</th>
                  <th>Grounded</th>
                  <th>Instruction</th>
                  <th>Strategy</th>
                  <th className="num-col">Steps</th>
                  <th className="num-col">Queries</th>
                  <th className="num-col">Skill reads</th>
                  <th className="num-col">Code execs</th>
                  <th className="num-col">Tokens</th>
                  <th className="num-col">Runtime</th>
                </tr>
              </thead>
              <tbody>
                {run.results.map((r) => {
                  const isOpen = open === r.case_id
                  return (
                    <Fragment key={r.case_id}>
                      <tr className={isOpen ? 'turn open' : 'turn'} onClick={() => toggle(r.case_id)} aria-expanded={isOpen}>
                        <td title={r.case_id}>{fromHistory ? clip(r.question) : r.case_id}</td>
                        {anyJev && <td><ContextCell context={r.context} /></td>}
                        <td><Status status={r.correct.status} /></td>
                        <td><Status status={r.grounded.status} /></td>
                        <td><Status status={r.instruction_following.status} /></td>
                        <td><Status status={r.execution_strategy.status} /></td>
                        <td className="num-col">{r.efficiency.steps}</td>
                        <td className="num-col">{r.efficiency.queries}</td>
                        <td className="num-col">{r.efficiency.skill_reads}</td>
                        <td className="num-col">{r.efficiency.code_executions}</td>
                        <td className="num-col">{r.efficiency.total_tokens.toLocaleString()}</td>
                        <td className="num-col">{seconds(r.efficiency.runtime_seconds)}</td>
                      </tr>
                      {isOpen && (
                        <tr className="turn-detail-row">
                          <td colSpan={anyJev ? 12 : 11}>
                            <CaseDetail
                              spec={run.cases.find((c) => c.id === r.case_id)}
                              result={r}
                              history={histories[r.case_id]}
                            />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

import { Fragment, useCallback, useEffect, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { getCompareHistory, getComparePreview, getCompareRun, listCompareRuns, startCompareRun } from '../api/client'
import type { ComparePair, ComparePreview, CompareRun, CompareRunInfo, HistoryLine, SystemResult, VerdictTally } from '../api/types'
import { Steps } from '../chat/Steps'
import { compact } from './overview'
import { turnSteps } from './results'
import { Status } from './SystemEvalPage'

const POLL_MS = 3000
const SIDES = [
  { key: 'full', label: 'Full conversation' },
  { key: 'jev', label: 'Jev context' },
] as const

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function seconds(s: number): string {
  return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

/** "2/2" = passed of judged; unknowns are counted apart, never as fails. */
function Tally({ t }: { t: VerdictTally }) {
  const judged = t.pass + t.fail
  return (
    <span className={t.fail ? 'worse' : undefined}>
      {judged ? `${t.pass}/${judged}` : '—'}
      {t.unknown > 0 && <span className="muted" title="unknown or judge error"> +{t.unknown}?</span>}
    </span>
  )
}

/** Jev minus full, as a % of full. Lower is better for every measure here. */
function Change({ full, jev }: { full: number; jev: number }) {
  if (full <= 0) return null
  const change = Math.round(((jev - full) / full) * 100)
  return <span className={change < 0 ? 'better' : change > 0 ? 'worse' : 'muted'}>{change > 0 ? '+' : ''}{change}%</span>
}

/** What Jev sent each follow-up: "2←1 · 3←1,2". */
function JevSent({ results }: { results: SystemResult[] }) {
  const followUps = results.filter((r) => r.context && r.context.earlier_turns > 0)
  if (followUps.length === 0) return <span className="muted">single turn</span>
  return (
    <span className="chips" aria-label="Jev sent">
      {followUps.map((r) => {
        const turn = (r.context?.earlier_turns ?? 0) + 1
        const sent = r.context?.sent_turns ?? []
        return (
          <span key={r.case_id} className="chip" title={`Turn ${turn} was sent ${sent.length ? `turn ${sent.join(', ')}` : 'no earlier turn'}`}>
            {turn}←{sent.join(',') || '∅'}
          </span>
        )
      })}
    </span>
  )
}

type History = HistoryLine[] | 'loading' | { error: string }

function TurnSide({ result, history, turn }: { result: SystemResult; history: History | undefined; turn: number }) {
  const e = result.efficiency
  return (
    <section className="arm-detail">
      <dl className="facts">
        <dt>Verdicts</dt>
        <dd className="verdict-marks">
          <span><Status status={result.correct.status} /> correct</span>
          <span><Status status={result.grounded.status} /> grounded</span>
          <span><Status status={result.instruction_following.status} /> instructions</span>
          <span><Status status={result.execution_strategy.status} /> strategy</span>
        </dd>
        {result.context && result.context.earlier_turns > 0 && (
          <>
            <dt>Jev sent</dt>
            <dd>
              {result.context.sent_turns.length ? `turn ${result.context.sent_turns.join(', ')}` : 'no earlier turn'} of{' '}
              {result.context.earlier_turns}
            </dd>
          </>
        )}
        <dt>Cost</dt>
        <dd>
          {e.steps} steps · {compact(e.total_tokens)} tokens · {seconds(e.runtime_seconds)}
        </dd>
      </dl>
      {[result.correct, result.grounded, result.execution_strategy]
        .filter((j) => j.status !== 'pass' && j.reason)
        .map((j, i) => (
          <p key={i} className="reason small">{j.reason}</p>
        ))}
      {history === undefined || history === 'loading' ? (
        <p className="muted">Loading what it did…</p>
      ) : 'error' in history ? (
        <p className="error">{history.error}</p>
      ) : (
        <Steps steps={turnSteps(history, turn).steps} streaming={false} />
      )}
      <div className="answer">
        {result.stopped ? <p className="error">Stopped before answering.</p> : <Markdown remarkPlugins={[remarkGfm]}>{result.final_answer}</Markdown>}
      </div>
    </section>
  )
}

function PairDetail({ pair, histories }: { pair: ComparePair; histories: Record<string, History> }) {
  return (
    <div className="case-detail">
      {pair.questions.map((question, i) => (
        <div key={i} className="compare-turn">
          <h3>
            Turn {i + 1}: <span className="question">{question}</span>
          </h3>
          <div className="turn-detail">
            {SIDES.map((side) => (
              <div key={side.key}>
                <p className="side-label">
                  <span className={`swatch series-${side.key}`} />
                  {side.label}
                </p>
                <TurnSide
                  result={pair[side.key].results[i]}
                  history={histories[`${side.key}/${pair[side.key].thread}`]}
                  turn={i + 1}
                />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function StartComparison({ onStarted }: { onStarted: (id: string) => void }) {
  const [preview, setPreview] = useState<ComparePreview | null>(null)
  const [asking, setAsking] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(() => {
    getComparePreview().then(setPreview).catch((e) => setError(errorText(e)))
  }, [])
  useEffect(refresh, [refresh])
  const running = preview?.running ?? null
  useEffect(() => {
    if (!running) return
    const timer = setTimeout(refresh, POLL_MS)
    return () => clearTimeout(timer)
  }, [running, preview, refresh])

  if (!preview?.available) return null
  const unpaired = preview.only_full + preview.only_jev

  const start = async () => {
    setBusy(true)
    setError(null)
    try {
      const { id } = await startCompareRun()
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
        <div className="confirm" role="group" aria-label="Confirm comparison">
          <span>
            Judge {preview.pairs} conversation{preview.pairs === 1 ? '' : 's'} both ways ({preview.turns * 2} turns)? Each
            turn uses 4 judge model calls.
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
          disabled={running !== null || preview.pairs === 0}
        >
          {running ? 'Comparing…' : 'Compare all conversations'}
        </button>
      )}
      {unpaired > 0 && !running && (
        <p className="muted hint">
          {unpaired} conversation{unpaired === 1 ? ' has' : 's have'} no twin yet. Make them with{' '}
          <code>python -m evals.system.compare --record</code>.
        </p>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  )
}

export function ComparePage() {
  const [runs, setRuns] = useState<CompareRunInfo[] | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [run, setRun] = useState<CompareRun | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const [histories, setHistories] = useState<Record<string, History>>({})
  const [error, setError] = useState<string | null>(null)

  const loadRuns = useCallback((select?: string) => {
    listCompareRuns()
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
      getCompareRun(runId)
        .then((loaded) => {
          if (!current) return
          setRun(loaded)
          if (loaded.status === 'running') {
            polled = true
            timer = setTimeout(load, POLL_MS)
          } else if (polled) {
            loadRuns()
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

  const toggle = (pair: ComparePair) => {
    const opening = open !== pair.id
    setOpen(opening ? pair.id : null)
    if (!opening || !run) return
    for (const side of SIDES) {
      const key = `${side.key}/${pair[side.key].thread}`
      if (histories[key]) continue
      setHistories((h) => ({ ...h, [key]: 'loading' }))
      getCompareHistory(run.id, side.key, pair[side.key].thread)
        .then((lines) => setHistories((h) => ({ ...h, [key]: lines })))
        .catch((e) => setHistories((h) => ({ ...h, [key]: { error: errorText(e) } })))
    }
  }

  if (error) return <p className="error">{error}</p>
  if (runs === null) return <p className="muted">Loading comparisons…</p>

  const total = (f: (p: ComparePair) => number) => run?.pairs.reduce((t, p) => t + f(p), 0) ?? 0
  return (
    <div className="eval-page">
      <header className="eval-header">
        <div>
          <h1>Jev vs full conversation</h1>
          <p className="muted">
            The same questions asked twice: once sending the <b>whole conversation</b> each turn, once sending only the
            earlier turns <b>Jev</b> picks. One row per conversation.
          </p>
        </div>
        <div className="controls">
          <StartComparison onStarted={(id) => { selectRun(id); loadRuns(id) }} />
          {runs.length > 0 && (
            <label className="field">
              Run
              <select value={runId ?? ''} onChange={(e) => selectRun(e.target.value)}>
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {when(r.created_at)} · {r.total} conversation{r.total === 1 ? '' : 's'}
                    {r.status === 'running' ? ' (running)' : ''}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </header>

      {runs.length === 0 ? (
        <div className="panel">
          <h2>No comparison yet</h2>
          <p className="muted">Click <b>Compare all conversations</b> to judge every pair.</p>
        </div>
      ) : !run ? (
        <p className="muted">Loading run…</p>
      ) : (
        <>
          {run.status === 'running' && (
            <p className="progress" role="status">
              Comparing… {run.pairs.length} / {run.total}
            </p>
          )}
          {run.status === 'failed' && <p className="error">The run stopped with an error: {run.error ?? 'unknown'}</p>}
          {run.pairs.length > 0 && (
            <p className="run-meta muted">
              judge {run.judge_model ?? 'none (code checks only)'}
              {SIDES.map((s) => (
                <span key={s.key} className="badge">
                  {s.label}: {compact(total((p) => p[s.key].summary.total_tokens))} tokens,{' '}
                  {seconds(total((p) => p[s.key].summary.runtime_seconds))}
                </span>
              ))}
            </p>
          )}

          <div className="table-scroll">
            <table className="turns compare" aria-label="Conversations">
              <thead>
                <tr>
                  <th rowSpan={2}>Conversation</th>
                  <th colSpan={2}>Correct</th>
                  <th colSpan={2}>Grounded</th>
                  <th colSpan={3}>Tokens</th>
                  <th colSpan={2}>Steps</th>
                  <th colSpan={3}>Runtime</th>
                  <th rowSpan={2}>Jev sent</th>
                </tr>
                <tr className="sub">
                  <th>full</th><th>Jev</th>
                  <th>full</th><th>Jev</th>
                  <th className="num-col">full</th><th className="num-col">Jev</th><th className="num-col">Δ</th>
                  <th className="num-col">full</th><th className="num-col">Jev</th>
                  <th className="num-col">full</th><th className="num-col">Jev</th><th className="num-col">Δ</th>
                </tr>
              </thead>
              <tbody>
                {run.pairs.map((pair) => {
                  const f = pair.full.summary
                  const j = pair.jev.summary
                  const isOpen = open === pair.id
                  return (
                    <Fragment key={pair.id}>
                      <tr className={isOpen ? 'turn open' : 'turn'} onClick={() => toggle(pair)} aria-expanded={isOpen}>
                        <td className="prompt">
                          {pair.questions[0]}
                          <span className="muted"> · {pair.questions.length} turn{pair.questions.length === 1 ? '' : 's'}</span>
                        </td>
                        <td><Tally t={f.correct} /></td>
                        <td><Tally t={j.correct} /></td>
                        <td><Tally t={f.grounded} /></td>
                        <td><Tally t={j.grounded} /></td>
                        <td className="num-col">{compact(f.total_tokens)}</td>
                        <td className="num-col">{compact(j.total_tokens)}</td>
                        <td className="num-col"><Change full={f.total_tokens} jev={j.total_tokens} /></td>
                        <td className="num-col">{f.steps}</td>
                        <td className="num-col">{j.steps}</td>
                        <td className="num-col">{seconds(f.runtime_seconds)}</td>
                        <td className="num-col">{seconds(j.runtime_seconds)}</td>
                        <td className="num-col"><Change full={f.runtime_seconds} jev={j.runtime_seconds} /></td>
                        <td><JevSent results={pair.jev.results} /></td>
                      </tr>
                      {isOpen && (
                        <tr className="turn-detail-row">
                          <td colSpan={14}>
                            <PairDetail pair={pair} histories={histories} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="muted small">
            Correct and grounded: turns passed of turns judged; <span className="muted">+n?</span> are unknown (not counted as
            fails). Jev sent: follow-up ← the earlier turns it was given; the last turn is always sent.
          </p>
          {(run.unpaired.full.length > 0 || run.unpaired.jev.length > 0) && (
            <p className="muted small">
              Not compared (no twin): {run.unpaired.full.length} full, {run.unpaired.jev.length} Jev.
            </p>
          )}
        </>
      )}
    </div>
  )
}

import { useEffect, useState } from 'react'
import { Link } from 'react-router'

import { getSystemRun, listSystemRuns } from '../api/client'
import type { EvalStatus, SystemResult, SystemRun, SystemRunInfo } from '../api/types'
import {
  DIMENSIONS,
  STATUSES,
  compact,
  contextGroup,
  groupStats,
  percent,
  recoveryTotals,
  verdictCounts,
  type ContextGroup,
} from './overview'

const GROUPS: { key: ContextGroup; label: string }[] = [
  { key: 'full', label: 'Full conversation' },
  { key: 'jev', label: 'Jev context' },
]
const STATUS_LABEL: Record<EvalStatus, string> = { pass: 'pass', fail: 'fail', unknown: 'unknown', judge_error: 'judge error' }

const METRICS = {
  tokens: { label: 'Tokens', value: (r: SystemResult) => r.efficiency.total_tokens, format: compact },
  runtime: { label: 'Runtime', value: (r: SystemResult) => r.efficiency.runtime_seconds, format: (n: number) => `${n.toFixed(1)}s` },
  steps: { label: 'Steps', value: (r: SystemResult) => r.efficiency.steps, format: (n: number) => String(n) },
} as const
type Metric = keyof typeof METRICS

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="card">
      <span className="card-label">{label}</span>
      <span className="card-value">{value}</span>
      {note && <span className="card-note">{note}</span>}
    </div>
  )
}

/** One 100% stacked bar per dimension; the counts are written out, so color is never the only cue. */
function VerdictBars({ results }: { results: SystemResult[] }) {
  return (
    <section className="panel overview-section">
      <h2>Verdicts</h2>
      <div className="legend-row" aria-hidden="true">
        {STATUSES.map((s) => (
          <span key={s} className="legend">
            <span className={`swatch status-${s}`} />
            {STATUS_LABEL[s]}
          </span>
        ))}
      </div>
      <div className="verdict-bars">
        {DIMENSIONS.map((d) => {
          const counts = verdictCounts(results, d.key)
          const text = STATUSES.filter((s) => counts[s]).map((s) => `${counts[s]} ${STATUS_LABEL[s]}`).join(' · ')
          return (
            <div key={d.key} className="verdict-row" role="img" aria-label={`${d.label}: ${text}`}>
              <span className="verdict-label">{d.label}</span>
              <div className="stack">
                {STATUSES.filter((s) => counts[s]).map((s) => (
                  <div
                    key={s}
                    className={`segment status-${s}`}
                    style={{ flexGrow: counts[s] }}
                    title={`${d.label}: ${counts[s]} of ${results.length} ${STATUS_LABEL[s]}`}
                  />
                ))}
              </div>
              <span className="verdict-counts muted">{text}</span>
            </div>
          )
        })}
      </div>
    </section>
  )
}

/** Horizontal bars, largest first, colored by context; the value sits at each bar's tip. */
function CostBars({ results, metric, groups }: { results: SystemResult[]; metric: Metric; groups: boolean }) {
  const m = METRICS[metric]
  const sorted = [...results].sort((a, b) => m.value(b) - m.value(a))
  const max = Math.max(1e-9, ...sorted.map(m.value))
  return (
    <div className="cost-bars" role="list" aria-label={`${m.label} per turn`}>
      {sorted.map((r) => {
        const v = m.value(r)
        const group = contextGroup(r)
        return (
          <div key={r.case_id} className="cost-row" role="listitem" title={`${r.question}\n${m.label}: ${m.format(v)}`}>
            <span className="cost-label">{r.question}</span>
            <div className="cost-track">
              <div className={`cost-bar series-${groups ? group : 'jev'}`} style={{ width: `${(v / max) * 100}%` }} />
              <span className="cost-value">{m.format(v)}</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function OverviewPage() {
  const [runs, setRuns] = useState<SystemRunInfo[] | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [run, setRun] = useState<SystemRun | null>(null)
  const [metric, setMetric] = useState<Metric>('tokens')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listSystemRuns()
      .then((list) => {
        setRuns(list)
        setRunId(list.find((r) => r.status !== 'running')?.id ?? list[0]?.id ?? null)
      })
      .catch((e) => setError(errorText(e)))
  }, [])

  useEffect(() => {
    if (!runId) return
    let current = true
    getSystemRun(runId)
      .then((loaded) => current && setRun(loaded))
      .catch((e) => current && setError(errorText(e)))
    return () => {
      current = false
    }
  }, [runId])

  const results = run?.results ?? []
  const both = results.some((r) => r.context) && results.some((r) => !r.context)
  if (error) return <p className="error">{error}</p>
  if (runs === null) return <p className="muted">Loading eval runs…</p>
  if (runs.length === 0) {
    return (
      <div className="panel">
        <h2>Nothing evaluated yet</h2>
        <p className="muted">
          Judge the saved conversations or run the cases on the <Link to="/evals/system">System</Link> page first.
        </p>
      </div>
    )
  }

  const all = groupStats(results)
  const recovery = recoveryTotals(results)
  return (
    <div className="eval-page">
      <header className="eval-header">
        <div>
          <h1>Overview</h1>
          <p className="muted">
            How the whole agent system did in one evaluation run, and how runs compare. Jev against the full
            conversation is on its own page: <Link to="/evals/jev">Jev vs full</Link>.
          </p>
        </div>
        <div className="controls">
          <label className="field">
            Run
            <select value={runId ?? ''} onChange={(e) => setRunId(e.target.value)}>
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {when(r.created_at)} · {r.total} {r.source === 'history' ? 'saved turn' : 'case'}
                  {r.total === 1 ? '' : 's'}
                  {r.status === 'running' ? ' (running)' : ''}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      {!run ? (
        <p className="muted">Loading run…</p>
      ) : results.length === 0 ? (
        <p className="muted">This run has no results yet.</p>
      ) : (
        <>
          <div className="cards" aria-label="Pass rates">
            {DIMENSIONS.map((d) => {
              const c = verdictCounts(results, d.key)
              const extra = [c.unknown && `${c.unknown} unknown`, c.judge_error && `${c.judge_error} judge errors`]
                .filter(Boolean)
                .join(' · ')
              return (
                <Tile
                  key={d.key}
                  label={d.label}
                  value={percent(all.passRate[d.key])}
                  note={`${c.pass} of ${results.length} turns passed${extra ? ` · ${extra}` : ''}`}
                />
              )
            })}
          </div>

          <VerdictBars results={results} />

          <section className="panel overview-section">
            <div className="section-head">
              <h2>Cost per turn</h2>
              <label className="field inline">
                Measure
                <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
                  {Object.entries(METRICS).map(([key, m]) => (
                    <option key={key} value={key}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {both && (
              <div className="legend-row">
                {GROUPS.map((g) => (
                  <span key={g.key} className="legend">
                    <span className={`swatch series-${g.key}`} />
                    {g.label}
                  </span>
                ))}
              </div>
            )}
            <CostBars results={results} metric={metric} groups={both} />
            <p className="muted small">
              Averages: {all.steps.toFixed(1)} steps, {compact(all.totalTokens)} tokens and {all.runtime.toFixed(1)}s per
              turn. Measurements, not verdicts: more steps is not worse by itself.
            </p>
          </section>

          <div className="cards" aria-label="Errors and recovery">
            <Tile label="Tool errors" value={String(recovery.errors)} note={`${recovery.identicalRetries} retried unchanged`} />
            <Tile label="Out of memory" value={String(recovery.oom)} note={`${recovery.sandboxMoves} moves to a bigger sandbox`} />
            <Tile label="Unanswered turns" value={String(recovery.stopped)} note="stopped before a final answer" />
            <Tile label="Ungrounded numbers" value={String(recovery.ungroundedValues)} note="not seen in any tool result (code check)" />
          </div>
        </>
      )}

      <section className="panel overview-section">
        <h2>All runs</h2>
        <div className="table-scroll">
          <table className="turns" aria-label="All runs">
            <thead>
              <tr>
                <th>When</th>
                <th>Source</th>
                <th className="num-col">Turns</th>
                {DIMENSIONS.map((d) => (
                  <th key={d.key} className="num-col">
                    {d.label}
                  </th>
                ))}
                <th className="num-col">Tokens</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const n = r.summary.cases
                return (
                  <tr
                    key={r.id}
                    className={r.id === runId ? 'turn open' : 'turn'}
                    onClick={() => setRunId(r.id)}
                    aria-selected={r.id === runId}
                  >
                    <td>
                      {when(r.created_at)}
                      {r.status === 'running' && <span className="badge">running</span>}
                    </td>
                    <td>{r.source === 'history' ? 'saved conversations' : 'eval cases'}</td>
                    <td className="num-col">{n}</td>
                    {DIMENSIONS.map((d) => (
                      <td key={d.key} className="num-col">
                        {percent(n ? r.summary[d.key] / n : null)}
                      </td>
                    ))}
                    <td className="num-col">{compact(r.summary.total_tokens)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

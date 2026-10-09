import { useCallback, useEffect, useState } from 'react'
import { Link, Route, Routes, useNavigate, useParams } from 'react-router'

import {
  analysisOutputUrl,
  deleteAnalysis,
  getAnalysis,
  getAnalysisSources,
  getAnalysisVersions,
  getOptimizationTranscript,
  listAnalyses,
  optimizeRun,
  runAnalysis,
} from '../api/client'
import type {
  Analysis,
  AnalysisRun,
  AnalysisSummary,
  AnalysisVersion,
  BottleneckKind,
  HistoryLine,
  RunLimits,
  SourceOption,
} from '../api/types'
import { ArtifactView } from '../chat/ArtifactView'
import { Steps } from '../chat/Steps'
import { turnSteps } from '../evals/results'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const POLL_MS = 2000

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function RunStatus({ run }: { run: AnalysisRun | null }) {
  if (!run) return <span className="muted">never run</span>
  const label = {
    running: 'running…',
    done: 'ran',
    failed: 'failed',
    needs_optimization: 'stopped before exporting (needs optimization)',
  }[run.status]
  return (
    <span className={run.status === 'failed' || run.status === 'needs_optimization' ? 'error' : run.status === 'running' ? 'muted' : undefined}>
      {label} {when(run.started_at)}
      {run.seconds !== null && <span className="muted"> · {run.seconds.toFixed(0)}s</span>}
    </span>
  )
}

function rows(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n.toLocaleString()
}

function size(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`
}

const KIND: Record<BottleneckKind, string> = {
  data_movement: 'the step from the database into the sandbox costs too much',
  sandbox_memory: 'the script runs out of memory',
  sandbox_runtime: 'the script is slow',
}

/** Rows into the sandbox: what was exported, or the estimate when nothing was. */
function exported(run: AnalysisRun): string {
  const inputs = run.measurements?.inputs ?? []
  if (inputs.some((i) => i.rows !== null)) {
    const n = inputs.reduce((t, i) => t + (i.rows ?? 0), 0)
    const b = inputs.reduce((t, i) => t + (i.bytes ?? 0), 0)
    return `${rows(n)} rows · ${size(b)}`
  }
  if (inputs.some((i) => i.estimated_rows !== null)) {
    const n = inputs.reduce((t, i) => t + (i.estimated_rows ?? 0), 0)
    return `~${rows(n)} rows (estimate)`
  }
  return ''
}

/** "on videos_ca" for a run that read other tables; empty for the recipe's own. */
function onTables(run: AnalysisRun): string {
  const swaps = Object.entries(run.sources ?? {})
  return swaps.length ? ` on ${swaps.map(([, to]) => to).join(', ')}` : ''
}

/** For each table the recipe reads, a choice of the tables with the same columns. */
function SourcePicker({
  options,
  chosen,
  onChange,
  disabled,
}: {
  options: SourceOption[]
  chosen: Record<string, string>
  onChange: (table: string, to: string) => void
  disabled: boolean
}) {
  return (
    <>
      {options.map((o) => (
        <label key={o.table} className="field">
          Data: {o.table}
          <select value={chosen[o.table] ?? o.table} onChange={(e) => onChange(o.table, e.target.value)} disabled={disabled}>
            <option value={o.table}>{o.table} (as saved)</option>
            {o.candidates.map((c) => (
              <option key={c.name} value={c.name} disabled={!c.ok}>
                {c.ok ? c.name : `${c.name}: ${c.problems[0] ?? 'different columns'}`}
              </option>
            ))}
          </select>
        </label>
      ))}
    </>
  )
}

const REWRITE: Record<BottleneckKind, string> = {
  data_movement: 'Optimize rewrites the SQL, so the work happens in PostgreSQL and fewer rows reach the sandbox',
  sandbox_memory: 'Optimize rewrites the script to use less memory; if it still runs out, the run moves to a bigger sandbox',
  sandbox_runtime: 'Optimize rewrites the slow part of the script',
}

type Verdict = 'hard' | 'soft' | 'ok' | 'none' | 'counted'
const VERDICT_TEXT: Record<Verdict, string> = {
  hard: '✗ over the hard limit',
  soft: '! over the target',
  ok: '✓ OK',
  none: '— not measured',
  counted: 'over, so the rows were counted ↓',
}

function judge(value: number | null | undefined, soft: number | null, hard: number | null): Verdict {
  if (value === null || value === undefined) return 'none'
  if (hard !== null && value > hard) return 'hard'
  if (soft !== null && value > soft) return 'soft'
  return 'ok'
}

function limitText(soft: string | null, hard: string | null): string {
  return [hard && `${hard} (hard)`, soft && `${soft} (target)`].filter(Boolean).join(' · ')
}

/**
 * What the check measured for a run, against which limit, and what that means: which numbers
 * stopped it or recommend an optimization, and which kind of rewrite Optimize does then.
 */
function RunCheck({ run, limits }: { run: AnalysisRun; limits: RunLimits }) {
  const m = run.measurements
  if (!m) return null
  type Row = { check: string; value: string; limit: string; verdict: Verdict }
  const rowsOut: Row[] = []
  for (const i of m.inputs) {
    const from = i.tables.length ? ` from ${i.tables.join(', ')}` : ''
    if (i.estimated_rows !== null) {
      // When the rows were counted, the count decided, not this estimate: say so instead of a failure.
      const estimate = judge(i.estimated_rows, null, limits.hard_export_rows)
      rowsOut.push({ check: `Estimated rows into the sandbox (EXPLAIN): ${i.file}${from}`, value: `~${rows(i.estimated_rows)}`,
        limit: limitText(null, rows(limits.hard_export_rows)), verdict: estimate === 'hard' && i.counted_rows != null ? 'counted' : estimate })
    }
    if (i.counted_rows != null) {
      rowsOut.push({ check: `Rows the query returns (counted in PostgreSQL, nothing moved): ${i.file}`, value: rows(i.counted_rows),
        limit: limitText(null, rows(limits.hard_export_rows)), verdict: judge(i.counted_rows, null, limits.hard_export_rows) })
    }
    if (i.estimated_bytes !== null) {
      rowsOut.push({ check: `Estimated size (EXPLAIN): ${i.file}`, value: `~${size(i.estimated_bytes)}`,
        limit: limitText(null, size(limits.max_export_bytes)), verdict: judge(i.estimated_bytes, null, limits.max_export_bytes) })
    }
    rowsOut.push({ check: `Rows exported: ${i.file}`, value: i.rows === null ? '—' : rows(i.rows),
      limit: limitText(rows(limits.soft_export_rows), rows(limits.hard_export_rows)),
      verdict: judge(i.rows, limits.soft_export_rows, limits.hard_export_rows) })
    rowsOut.push({ check: `Export size: ${i.file}`, value: i.bytes === null ? '—' : size(i.bytes),
      limit: limitText(size(limits.soft_export_bytes), size(limits.max_export_bytes)),
      verdict: judge(i.bytes, limits.soft_export_bytes, limits.max_export_bytes) })
    rowsOut.push({ check: `Query time: ${i.file}`, value: i.query_seconds === null ? '—' : `${i.query_seconds.toFixed(1)}s`,
      limit: limitText(null, limits.hard_query_seconds ? `${limits.hard_query_seconds}s` : null),
      verdict: judge(i.query_seconds, null, limits.hard_query_seconds) })
  }
  const sb = m.sandbox
  rowsOut.push({ check: 'Script time', value: sb?.seconds == null ? '—' : `${sb.seconds.toFixed(1)}s`,
    limit: limitText(`${limits.soft_run_seconds}s`, null), verdict: judge(sb?.seconds, limits.soft_run_seconds, null) })
  const memoryShare = sb?.peak_memory_mb != null && sb.memory_limit_mb ? sb.peak_memory_mb / sb.memory_limit_mb : null
  rowsOut.push({
    check: 'Peak memory',
    value: sb?.killed ? 'ran out of memory' : sb?.peak_memory_mb == null ? '—'
      : `${Math.round(sb.peak_memory_mb)} MB${sb.memory_limit_mb ? ` of ${sb.memory_limit_mb.toLocaleString()} MB` : ''}`,
    limit: limitText(`${Math.round(limits.memory_warning_ratio * 100)}% of the sandbox`, 'running out'),
    verdict: sb?.killed ? 'hard' : judge(memoryShare, limits.memory_warning_ratio, null),
  })

  // The outcome comes from the server's own findings, never from this table.
  const hard = run.findings.find((f) => f.limit === 'hard')
  const main = hard ?? run.findings[0]
  const stoppedBefore = run.status === 'needs_optimization' && m.inputs.every((i) => i.rows === null)
  const outcome = !main
    ? 'Within every limit: no optimization needed.'
    : `${hard ? (stoppedBefore ? 'Stopped before exporting anything: over a hard limit.' : 'Stopped: over a hard limit.')
        : 'Finished, but over a target: optimization is recommended.'} Problem: ${KIND[main.kind]} → ${REWRITE[main.kind]}.`
  return (
    <details className="run-check" open={run.status === 'needs_optimization'}>
      <summary>
        <b>Check</b> <span className={hard ? 'error' : main ? 'warning-text-soft' : 'muted'}>{outcome}</span>
      </summary>
      <div className="table-scroll">
        <table className="turns" aria-label="Check">
          <thead>
            <tr>
              <th>Check</th>
              <th className="num-col">Value</th>
              <th>Limit</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {rowsOut.map((r) => (
              <tr key={r.check}>
                <td>{r.check}</td>
                <td className="num-col">{r.value}</td>
                <td className="muted">{r.limit}</td>
                <td className={`verdict-${r.verdict}`}>{VERDICT_TEXT[r.verdict]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        The estimate comes from PostgreSQL's planner (EXPLAIN), before any rows move. When it is over a limit, the rows are
        counted in PostgreSQL to be sure (the planner can be far off), still without moving them. Hard limits stop a run;
        targets only recommend an optimization. Which problem it is decides what Optimize rewrites.
      </p>
    </details>
  )
}

type Transcript = HistoryLine[] | 'loading' | { error: string }

/** One optimization as it happened: the request the server wrote, then what the agent did and said. */
function OptimizationTranscript({ transcript }: { transcript: Transcript | undefined }) {
  if (transcript === undefined || transcript === 'loading') return <p className="muted small">Loading…</p>
  if ('error' in transcript) return <p className="error small">{transcript.error}</p>
  const request = transcript.find((l) => l.role === 'user')?.content ?? ''
  const { steps, answer } = turnSteps(transcript, 1)
  return (
    <div className="transcript">
      <details>
        <summary>The request the agent got</summary>
        <div className="answer request">
          <Markdown remarkPlugins={[remarkGfm]}>{request}</Markdown>
        </div>
      </details>
      <Steps steps={steps} streaming={false} />
      {answer && (
        <div className="answer">
          <Markdown remarkPlugins={[remarkGfm]}>{answer}</Markdown>
        </div>
      )}
    </div>
  )
}

/** Every optimization of this analysis, newest first: why, what came of it, and the transcript on demand. */
function Optimizations({
  analysis,
  open,
  transcripts,
  onToggle,
}: {
  analysis: Analysis
  open: string | null
  transcripts: Record<string, Transcript>
  onToggle: (conversation: string) => void
}) {
  const asked = analysis.runs.filter((r) => r.optimization)
  if (asked.length === 0) return null
  return (
    <section className="panel overview-section" id="optimizations" aria-label="Optimizations">
      <h2>Optimizations</h2>
      <p className="muted small">
        When a run no longer fit its data and you clicked Optimize, the agent got a request written from the run's
        measurements and rewrote the recipe. Each one is kept here with the analysis.
      </p>
      {asked.flatMap((r) => {
        const attempts = [...(r.optimization!.earlier ?? []), r.optimization!]
        return attempts.map((o, n) => ({ r, o, n: attempts.length > 1 ? n + 1 : null }))
      }).map(({ r, o, n }) => {
        const reason = (r.findings.find((f) => f.limit === 'hard') ?? r.findings[0])?.reason
        const isOpen = open === o.conversation
        return (
          <div key={o.conversation} className="optimization-entry">
            <button className="optimization-head" onClick={() => onToggle(o.conversation)} aria-expanded={isOpen}>
              <span>{isOpen ? '▾' : '▸'}</span>
              <span>
                <b>
                  {o.status === 'done'
                    ? `Version ${r.version} → ${o.new_version}`
                    : o.status === 'running'
                      ? `Version ${r.version}: optimizing…`
                      : `Version ${r.version}: optimization failed`}
                  {n !== null && ` (attempt ${n})`}
                </b>
                <span className="muted">
                  {' '}
                  · {when(r.started_at)}
                  {onTables(r)}
                </span>
                {reason && <span className="muted small optimization-reason">{reason}</span>}
              </span>
            </button>
            {isOpen && <OptimizationTranscript transcript={transcripts[o.conversation]} />}
          </div>
        )
      })}
    </section>
  )
}

/** One version's inputs, script and outputs. */
function RecipeView({ recipe }: { recipe: AnalysisVersion }) {
  return (
    <div className="recipe-view">
      <h3>Inputs</h3>
      {recipe.inputs.map((input, i) =>
        input.kind === 'query' ? (
          <div key={i}>
            <p className="small">
              Query → <code>$DATA_DIR/{input.file}</code>
            </p>
            <pre className="code">{input.sql.trim()}</pre>
          </div>
        ) : (
          <p key={i} className="small">
            Uploaded file <code>{input.name}</code> → <code>$DATA_DIR/{input.name}.parquet</code>
          </p>
        ),
      )}
      <details>
        <summary>Script</summary>
        <pre className="code script">{recipe.script}</pre>
      </details>
      <p className="small">
        Outputs: {recipe.outputs.map((o) => <code key={o.file}>{o.file} </code>)}
      </p>
    </div>
  )
}

function versionLabel(v: AnalysisVersion, current: number): string {
  const why = v.optimization ? ', optimized' : ''
  return `Version ${v.version}${v.version === current ? ' (current' + why + ')' : why ? ` (${why.slice(2)})` : ''}`
}

/** The recipe of any version, and the one before it side by side to see what changed. */
function Recipe({ analysis }: { analysis: Analysis }) {
  const [versions, setVersions] = useState<AnalysisVersion[] | null>(null)
  const [chosen, setChosen] = useState<number>(analysis.version)
  const [compare, setCompare] = useState(false)

  // Mounted per version (see its key), so a new version starts from itself.
  useEffect(() => {
    let current = true
    getAnalysisVersions(analysis.id)
      .then((list) => current && setVersions(list))
      .catch(() => current && setVersions(null))
    return () => {
      current = false
    }
  }, [analysis.id])

  const all = versions ?? [analysis]
  const shown = all.find((v) => v.version === chosen) ?? analysis
  const before = all.find((v) => v.version === shown.version - 1)
  return (
    <section className="panel overview-section" aria-label="Recipe">
      <div className="section-head">
        <h2>Recipe</h2>
        {all.length > 1 && (
          <span className="recipe-controls">
            <label className="field inline">
              Show
              <select value={chosen} onChange={(e) => setChosen(Number(e.target.value))}>
                {[...all].reverse().map((v) => (
                  <option key={v.version} value={v.version}>
                    {versionLabel(v, analysis.version)}
                  </option>
                ))}
              </select>
            </label>
            {before && (
              <label className="check">
                <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
                Compare with version {before.version}
              </label>
            )}
          </span>
        )}
      </div>
      <p className="muted small">What Run does, with no model involved: load each input, then run the script.</p>
      {shown.optimization && (
        <p className="small">
          Version {shown.version} was made by an optimization of version {shown.optimization.from_version}:{' '}
          {shown.optimization.reason.replace(/\.?$/, '.')}
          {shown.optimization.results_check && <> {shown.optimization.results_check}</>}
        </p>
      )}
      {compare && before ? (
        <div className="recipe-compare">
          <div>
            <p className="compare-label">Version {before.version} (before)</p>
            <RecipeView recipe={before} />
          </div>
          <div>
            <p className="compare-label">Version {shown.version}</p>
            <RecipeView recipe={shown} />
          </div>
        </div>
      ) : (
        <RecipeView recipe={shown} />
      )}
    </section>
  )
}

/** Why a run did not fit, and the optimization it led to (or can lead to). */
function OptimizationPanel({
  analysis,
  run,
  onOptimize,
  onShow,
  onShowTranscript,
}: {
  analysis: Analysis
  run: AnalysisRun
  onOptimize: () => void
  onShow: (runId: string) => void
  onShowTranscript: (conversation: string) => void
}) {
  const targets = Object.values(run.sources ?? {}).join(', ') || analysis.sources.join(', ')
  const o = run.optimization
  if (o?.status === 'running') {
    return (
      <p className="progress" role="status">
        Optimizing… the agent is rewriting the recipe. It is then checked (same results on the saved tables, fits{' '}
        {targets}), saved as a new version and run again.
      </p>
    )
  }
  if (o?.status === 'done') {
    const after = analysis.runs.find((r) => r.optimized_from === run.version && r.version === o.new_version)
    return (
      <div className="optimization done" role="status">
        <p>
          ✓ Analysis updated to <b>version {o.new_version}</b>.
          {after && exported(after) && (
            <> New strategy: <b>{exported(after)}</b> into the sandbox, instead of {exported(run)}.</>
          )}
          {after && (
            <>
              {' '}
              <button className="link-button" onClick={() => onShow(after.id)}>Show that run</button>
            </>
          )}
        </p>
        {o.message && <p className="muted small">{o.message}</p>}
        <p className="small">
          <button className="link-button" onClick={() => onShowTranscript(o.conversation)}>See what the agent did</button>
        </p>
      </div>
    )
  }
  if (o?.status === 'failed') {
    return (
      <div className="optimization failed">
        <p className="error">Optimization failed; version {run.version} is unchanged.</p>
        {o.message && <pre className="code">{o.message}</pre>}
        <p className="small">
          <button className="link-button" onClick={() => onShowTranscript(o.conversation)}>See what the agent did</button>
        </p>
        {run.version === analysis.version && (
          <p className="optimize-row">
            <button onClick={onOptimize} disabled={analysis.running}>
              Optimize again
            </button>
            <span className="muted small">A new attempt, in a new conversation; this one stays listed under Optimizations.</span>
          </p>
        )}
      </div>
    )
  }
  if (run.findings.length === 0) return null
  const hard = run.status === 'needs_optimization'
  const kind = (run.findings.find((f) => f.limit === 'hard') ?? run.findings[0]).kind
  const stale = run.version !== analysis.version
  return (
    <div className={`optimization ${hard ? 'needed' : 'recommended'}`}>
      <p>
        <b>{hard ? `The saved strategy no longer fits ${targets}` : 'Optimization recommended'}</b>: {KIND[kind]}.
      </p>
      <ul className="plain-list">
        {run.findings.map((f, i) => (
          <li key={i}>{f.reason}</li>
        ))}
      </ul>
      {stale ? (
        <p className="muted small">This run used version {run.version}; the analysis is now version {analysis.version}.</p>
      ) : (
        <p className="optimize-row">
          <button className={hard ? undefined : 'secondary'} onClick={onOptimize} disabled={analysis.running}>
            Optimize
          </button>
          <span className="muted small">
            The agent rewrites the recipe (this uses the model). The new version must give the same results on the saved
            tables and fit {targets}; it is saved as version {analysis.version + 1} and run again.
          </span>
        </p>
      )}
    </div>
  )
}

function AnalysisList() {
  const [analyses, setAnalyses] = useState<AnalysisSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirming, setConfirming] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<{ id: string; message: string } | null>(null)

  useEffect(() => {
    listAnalyses().then(setAnalyses).catch((e) => setError(errorText(e)))
  }, [])

  const remove = async (id: string) => {
    setDeleteError(null)
    try {
      await deleteAnalysis(id)
      setAnalyses((list) => list?.filter((a) => a.id !== id) ?? null)
    } catch (e) {
      setDeleteError({ id, message: errorText(e) })
    } finally {
      setConfirming(null)
    }
  }

  if (error) return <p className="error">{error}</p>
  if (analyses === null) return <p className="muted">Loading analyses…</p>
  return (
    <div className="eval-page">
      <header>
        <h1>Analyses</h1>
        <p className="muted">
          Analyses saved from the chat. Each one keeps its data sources and script, so <b>Run</b> makes its output again
          from the data as it is now. In the chat, ask the agent to “save this analysis”.
        </p>
      </header>
      {analyses.length === 0 ? (
        <div className="panel">
          <h2>No saved analyses yet</h2>
          <p className="muted">
            Make a chart in the <Link to="/chat">chat</Link>, then ask the agent to save it.
          </p>
        </div>
      ) : (
        <ul className="analysis-list" aria-label="Saved analyses">
          {analyses.map((a) => (
            <li key={a.id} className="analysis-row">
              <Link to={`/analyses/${a.id}`} className="analysis-card">
                <span className="analysis-title">{a.title}</span>
                {a.description && <span className="muted">{a.description}</span>}
                <span className="analysis-meta">
                  saved {when(a.created_at)}
                  {a.version > 1 && ` · version ${a.version}`} · <RunStatus run={a.last_run} /> · {a.outputs.join(', ')}
                </span>
                {deleteError?.id === a.id && <span className="error">{deleteError.message}</span>}
              </Link>
              {confirming === a.id ? (
                <span className="row-confirm" role="group" aria-label={`Confirm deleting ${a.title}`}>
                  <button className="danger small" onClick={() => remove(a.id)}>Delete</button>
                  <button className="link-button" onClick={() => setConfirming(null)}>Cancel</button>
                </span>
              ) : (
                <button
                  className="icon-button"
                  onClick={() => setConfirming(a.id)}
                  aria-label={`Delete ${a.title}…`}
                  title="Delete"
                >
                  ×
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function AnalysisDetail() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [shown, setShown] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [options, setOptions] = useState<SourceOption[]>([])
  const [openTranscript, setOpenTranscript] = useState<string | null>(null)
  const [transcripts, setTranscripts] = useState<Record<string, Transcript>>({})
  const [chosen, setChosen] = useState<Record<string, string>>({})

  useEffect(() => {
    getAnalysisSources(id).then(setOptions).catch(() => setOptions([]))
  }, [id])

  const load = useCallback(
    () =>
      getAnalysis(id)
        .then((loaded) => {
          setAnalysis(loaded)
          return loaded
        })
        .catch((e) => {
          setError(errorText(e))
          return null
        }),
    [id],
  )

  useEffect(() => {
    void load()
  }, [load])

  // While a run goes on, check again until it ends, then show what it made.
  const running = analysis?.running ?? false
  useEffect(() => {
    if (!running) return
    const timer = setTimeout(() => {
      void load().then((loaded) => {
        // After an optimization, keep showing the run that needed it: its panel says what changed.
        const optimized = loaded?.runs.find((r) => r.id === shown)?.optimization?.status === 'done'
        if (loaded && !loaded.running && !optimized) setShown(loaded.runs[0]?.id ?? null)
      })
    }, POLL_MS)
    return () => clearTimeout(timer)
  }, [running, analysis, load, shown])

  if (error) return <p className="error">{error}</p>
  if (!analysis) return <p className="muted">Loading…</p>

  const run = async () => {
    setError(null)
    try {
      const swaps = Object.fromEntries(Object.entries(chosen).filter(([from, to]) => from !== to))
      const started = await runAnalysis(analysis.id, swaps)
      setShown(started.id)
      await load()
    } catch (e) {
      setError(errorText(e))
    }
  }

  const optimize = async (runId: string) => {
    setError(null)
    try {
      await optimizeRun(analysis.id, runId)
      setShown(runId)
      await load()
    } catch (e) {
      setError(errorText(e))
    }
  }

  const toggleTranscript = (conversation: string, scroll = false) => {
    const opening = openTranscript !== conversation || scroll
    setOpenTranscript(opening ? conversation : null)
    if (opening && !transcripts[conversation]) {
      setTranscripts((t) => ({ ...t, [conversation]: 'loading' }))
      getOptimizationTranscript(analysis.id, conversation)
        .then((lines) => setTranscripts((t) => ({ ...t, [conversation]: lines })))
        .catch((e) => setTranscripts((t) => ({ ...t, [conversation]: { error: errorText(e) } })))
    }
    if (scroll) setTimeout(() => document.getElementById('optimizations')?.scrollIntoView?.({ behavior: 'smooth' }), 0)
  }

  const remove = async () => {
    try {
      await deleteAnalysis(analysis.id)
      navigate('/analyses')
    } catch (e) {
      setError(errorText(e))
    }
  }

  // The latest run unless one was picked: a run that needs optimization is the first thing to see.
  const current = analysis.runs.find((r) => r.id === shown) ?? analysis.runs[0]

  return (
    <div className="eval-page">
      <p>
        <Link to="/analyses">← All analyses</Link>
      </p>
      <header className="eval-header">
        <div>
          <h1>{analysis.title}</h1>
          {analysis.description && <p className="muted">{analysis.description}</p>}
          <p className="muted small">
            {analysis.question && <>Asked: “{analysis.question}” · </>}saved {when(analysis.created_at)}
            {analysis.version > 1 && analysis.updated_at && (
              <> · version {analysis.version}, changed {when(analysis.updated_at)}</>
            )}
            {analysis.optimization && (
              <>
                {' '}· optimized from version {analysis.optimization.from_version}
                {Object.values(analysis.optimization.sources).length > 0 &&
                  ` for ${Object.values(analysis.optimization.sources).join(', ')}`}
              </>
            )}
          </p>
          <p className="small analysis-links">
            {analysis.conversation && <Link to={`/chat/past/${analysis.conversation}`}>Open the chat it was saved in</Link>}
            {analysis.optimization?.conversation && (
              <button className="link-button" onClick={() => toggleTranscript(analysis.optimization!.conversation!, true)}>
                How it was optimized
              </button>
            )}
          </p>
        </div>
        <div className="controls">
          <SourcePicker
            options={options}
            chosen={chosen}
            onChange={(table, to) => setChosen((c) => ({ ...c, [table]: to }))}
            disabled={running}
          />
          <button onClick={run} disabled={running}>
            {running ? 'Running…' : 'Run'}
          </button>
          {confirmDelete ? (
            <span className="confirm" role="group" aria-label="Confirm delete">
              <button className="danger" onClick={remove}>Delete</button>
              <button className="secondary" onClick={() => setConfirmDelete(false)}>Cancel</button>
            </span>
          ) : (
            <button className="secondary" onClick={() => setConfirmDelete(true)} disabled={running}>
              Delete…
            </button>
          )}
        </div>
      </header>

      {current && (
        <section className="panel overview-section" aria-label="Output">
          <h2>
            Output{' '}
            <span className="muted small">
              <RunStatus run={current} />
              {onTables(current)}
            </span>
          </h2>
          {current.status === 'running' && <p className="progress" role="status">Running… a sandbox is starting, then the queries and the script run.</p>}
          {/* The action first; the check table below explains the numbers behind it. */}
          <OptimizationPanel
            analysis={analysis}
            run={current}
            onOptimize={() => optimize(current.id)}
            onShow={setShown}
            onShowTranscript={(conversation) => toggleTranscript(conversation, true)}
          />
          {analysis.limits && <RunCheck run={current} limits={analysis.limits} />}
          {current.status === 'failed' && (
            <>
              <p className="error">The run failed: {current.error}</p>
              {current.log && <pre className="code">{current.log}</pre>}
            </>
          )}
          {current.notes.map((note, i) => (
            <p key={i} className="notice">{note}</p>
          ))}
          {current.outputs.map((file) => (
            <ArtifactView key={file} artifact={{ name: file, kind: 'html', url: analysisOutputUrl(analysis.id, current.id, file) }} />
          ))}
        </section>
      )}

      <section className="panel overview-section">
        <h2>Runs</h2>
        <div className="table-scroll">
          <table className="turns" aria-label="Runs">
            <thead>
              <tr>
                <th>When</th>
                <th>How</th>
                <th>Result</th>
                <th className="num-col">Into the sandbox</th>
                <th className="num-col">Query</th>
                <th className="num-col">Script</th>
                <th className="num-col">Peak memory</th>
                <th className="num-col">Time</th>
              </tr>
            </thead>
            <tbody>
              {analysis.runs.map((r) => (
                <tr key={r.id} className={r.id === current?.id ? 'turn open' : 'turn'} onClick={() => setShown(r.id)}>
                  <td>{when(r.started_at)}</td>
                  <td>
                    {r.trigger === 'save' ? 'test run when saved' : `run${onTables(r)}`}
                    {analysis.version > 1 && <span className="muted"> · v{r.version ?? 1}</span>}
                    {r.optimized_from && <span className="muted"> · after optimization</span>}
                  </td>
                  <td className={r.status === 'failed' ? 'error' : r.status === 'needs_optimization' ? 'warning-text' : undefined}>
                    {r.status === 'failed'
                      ? `failed: ${r.error}`
                      : r.status === 'running'
                        ? 'running…'
                        : r.status === 'needs_optimization'
                          ? 'needs optimization'
                          : r.outputs.join(', ')}
                    {r.optimization?.status === 'done' && ` → version ${r.optimization.new_version}`}
                    {r.optimization?.status === 'running' && ' → optimizing…'}
                    {r.optimization?.status === 'failed' && ' → optimization failed'}
                    {r.status === 'done' && r.findings?.length > 0 && <span className="muted"> · could be optimized</span>}
                  </td>
                  <td className="num-col">{exported(r)}</td>
                  <td className="num-col">
                    {(() => {
                      const q = (r.measurements?.inputs ?? []).reduce((t, i) => t + (i.query_seconds ?? 0), 0)
                      return q ? `${q.toFixed(1)}s` : ''
                    })()}
                  </td>
                  <td className="num-col">
                    {r.measurements?.sandbox?.seconds != null ? `${r.measurements.sandbox.seconds.toFixed(1)}s` : ''}
                  </td>
                  <td className="num-col">
                    {r.measurements?.sandbox?.killed
                      ? 'out of memory'
                      : r.measurements?.sandbox?.peak_memory_mb != null
                        ? `${Math.round(r.measurements.sandbox.peak_memory_mb)} MB`
                        : ''}
                  </td>
                  <td className="num-col">{r.seconds !== null ? `${r.seconds.toFixed(0)}s` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <Optimizations analysis={analysis} open={openTranscript} transcripts={transcripts} onToggle={(c) => toggleTranscript(c)} />

      <Recipe key={`${analysis.id}-${analysis.version}`} analysis={analysis} />
    </div>
  )
}

export function AnalysesPage() {
  return (
    <Routes>
      <Route index element={<AnalysisList />} />
      <Route path=":id" element={<AnalysisDetail />} />
    </Routes>
  )
}

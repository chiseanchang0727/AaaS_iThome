import { useCallback, useEffect, useState } from 'react'
import { Link, Route, Routes, useNavigate, useParams } from 'react-router'

import { analysisOutputUrl, deleteAnalysis, getAnalysis, getAnalysisSources, listAnalyses, runAnalysis } from '../api/client'
import type { Analysis, AnalysisRun, AnalysisSummary, SourceOption } from '../api/types'
import { ArtifactView } from '../chat/ArtifactView'

const POLL_MS = 2000

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function RunStatus({ run }: { run: AnalysisRun | null }) {
  if (!run) return <span className="muted">never run</span>
  const label = run.status === 'running' ? 'running…' : run.status === 'done' ? 'ran' : 'failed'
  return (
    <span className={run.status === 'failed' ? 'error' : run.status === 'running' ? 'muted' : undefined}>
      {label} {when(run.started_at)}
      {run.seconds !== null && <span className="muted"> · {run.seconds.toFixed(0)}s</span>}
    </span>
  )
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
        if (loaded && !loaded.running) setShown(loaded.runs[0]?.id ?? null)
      })
    }, POLL_MS)
    return () => clearTimeout(timer)
  }, [running, analysis, load])

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

  const remove = async () => {
    try {
      await deleteAnalysis(analysis.id)
      navigate('/analyses')
    } catch (e) {
      setError(errorText(e))
    }
  }

  const current =
    analysis.runs.find((r) => r.id === shown) ?? analysis.runs.find((r) => r.status === 'done') ?? analysis.runs[0]

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
                  </td>
                  <td className={r.status === 'failed' ? 'error' : undefined}>
                    {r.status === 'failed' ? `failed: ${r.error}` : r.status === 'running' ? 'running…' : r.outputs.join(', ')}
                  </td>
                  <td className="num-col">{r.seconds !== null ? `${r.seconds.toFixed(0)}s` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel overview-section" aria-label="Recipe">
        <h2>Recipe</h2>
        <p className="muted small">What Run does, with no model involved: load each input, then run the script.</p>
        <h3>Inputs</h3>
        {analysis.inputs.map((input, i) =>
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
          <pre className="code script">{analysis.script}</pre>
        </details>
        <p className="small">
          Outputs: {analysis.outputs.map((o) => <code key={o.file}>{o.file} </code>)}
        </p>
      </section>
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

import { useEffect, useState } from 'react'

import { applySuggestion, askAdvice, dismissSuggestion, listAdvice } from '../api/client'
import type { Advice, Suggestion, SuggestionKind, Workflow } from '../api/types'

const KIND_TEXT: Record<SuggestionKind, string> = {
  set_lanes: 'Sandboxes per run',
  run_first: 'Run first',
  optimize_analysis: 'Optimize an analysis',
  share_data: 'Share data',
  split_workflow: 'Split the workflow',
  other: 'Other',
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function SuggestionCard({
  suggestion: s,
  titles,
  busy,
  blocked,
  onApply,
  onDismiss,
}: {
  suggestion: Suggestion
  titles: Record<string, string>
  busy: boolean
  blocked: string | null
  onApply: () => void
  onDismiss: () => void
}) {
  const what =
    s.kind === 'set_lanes' && s.value !== null
      ? `${s.value} ${s.value === 1 ? 'sandbox' : 'sandboxes'}`
      : s.node_ids.map((n) => titles[n] ?? n).join(', ')
  return (
    <article className={`wf-suggestion wf-suggestion-${s.status}`} aria-label={KIND_TEXT[s.kind]}>
      <header>
        <span className="wf-suggestion-kind">{KIND_TEXT[s.kind]}</span>
        {what && <span className="wf-suggestion-what">{what}</span>}
        {s.status !== 'pending' && <span className={`wf-status wf-status-${s.status === 'applied' ? 'done' : 'failed'}`}>{s.status}</span>}
      </header>
      <p>{s.reason}</p>
      <p className="muted small">
        <b>Expected:</b> {s.expected_gain}
      </p>
      {s.note && <p className="muted small">{s.note}</p>}
      {s.result && <p className={s.status === 'failed' ? 'error small' : 'small'}>{s.result}</p>}
      {s.status === 'pending' && (
        <div className="wf-suggestion-actions">
          {s.applicable && (
            <button className="small" onClick={onApply} disabled={busy || blocked !== null} title={blocked ?? undefined}>
              Apply
            </button>
          )}
          <button className="secondary small" onClick={onDismiss} disabled={busy}>
            Dismiss
          </button>
          {s.applicable && blocked && <span className="muted small">{blocked}</span>}
        </div>
      )}
    </article>
  )
}

/**
 * What the workflow advisor suggests, from the workflow's last runs. Nothing changes until a
 * suggestion is applied; only some can be (sandboxes per run, run first, Optimize), the rest is advice.
 */
export function Suggestions({
  workflowId,
  titles,
  hasRuns,
  blocked,
  onApplied,
}: {
  workflowId: string
  /** Analysis node id -> its analysis's title. */
  titles: Record<string, string>
  hasRuns: boolean
  /** Why nothing can be applied now (unsaved changes, a run going on), or null. */
  blocked: string | null
  onApplied: (workflow: Workflow) => void
}) {
  const [all, setAll] = useState<Advice[] | null>(null)
  const [shown, setShown] = useState<string | null>(null)
  const [goal, setGoal] = useState<Advice['goal']>('fastest')
  const [asking, setAsking] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listAdvice(workflowId)
      .then(setAll)
      .catch((e) => setError(errorText(e)))
  }, [workflowId])

  const advice = all?.find((a) => a.id === shown) ?? all?.[0] ?? null

  const replace = (updated: Advice) => setAll((list) => list?.map((a) => (a.id === updated.id ? updated : a)) ?? [updated])

  const ask = async () => {
    setAsking(true)
    setError(null)
    try {
      const fresh = await askAdvice(workflowId, goal)
      setAll((list) => [fresh, ...(list ?? [])])
      setShown(fresh.id)
    } catch (e) {
      setError(errorText(e))
    } finally {
      setAsking(false)
    }
  }

  const act = async (what: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    try {
      await what()
    } catch (e) {
      setError(errorText(e))
      // A failed apply is recorded with the suggestion: show it.
      listAdvice(workflowId).then(setAll).catch(() => {})
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="wf-suggestions">
      <div className="wf-suggestions-ask">
        <label className="field inline">
          Goal
          <select value={goal} onChange={(e) => setGoal(e.target.value as Advice['goal'])} disabled={asking}>
            <option value="fastest">fastest</option>
            <option value="cheapest">cheapest</option>
          </select>
        </label>
        <button onClick={ask} disabled={asking || !hasRuns} title={hasRuns ? undefined : 'Run the workflow first'}>
          {asking ? 'Thinking…' : 'Suggest improvements'}
        </button>
        {all && all.length > 1 && (
          <label className="field inline">
            Asked
            <select value={advice?.id ?? ''} onChange={(e) => setShown(e.target.value)}>
              {all.map((a) => (
                <option key={a.id} value={a.id}>
                  {when(a.created_at)} · {a.goal}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      <p className="muted small">
        An agent reads this workflow's last runs (where the time went, what each analysis used) and suggests changes.
        Nothing changes until you apply one.
      </p>
      {error && <p className="error">{error}</p>}
      {!hasRuns && <p className="muted">Run the workflow first: the advisor works from its runs.</p>}
      {all === null && !error && <p className="muted">Loading…</p>}
      {advice && (
        <section aria-label="Advice">
          <p className="wf-advice-summary">{advice.summary}</p>
          <p className="muted small">
            Asked {when(advice.created_at)} · goal {advice.goal} · from {advice.runs.length}{' '}
            {advice.runs.length === 1 ? 'run' : 'runs'}
            {advice.model && ` · ${advice.model}`}
          </p>
          {advice.suggestions.length === 0 ? (
            <p className="muted">No suggestions: nothing in the runs shows a clear gain.</p>
          ) : (
            <div className="wf-suggestion-list">
              {advice.suggestions.map((s, index) => (
                <SuggestionCard
                  key={index}
                  suggestion={s}
                  titles={titles}
                  busy={busy}
                  blocked={blocked}
                  onApply={() =>
                    act(async () => {
                      const { advice: updated, workflow } = await applySuggestion(workflowId, advice.id, index)
                      replace(updated)
                      onApplied(workflow)
                    })
                  }
                  onDismiss={() => act(async () => replace(await dismissSuggestion(workflowId, advice.id, index)))}
                />
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  )
}

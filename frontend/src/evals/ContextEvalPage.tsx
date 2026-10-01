import { Fragment, useEffect, useMemo, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { getEvalHistory, getEvalRun, listEvalRuns } from '../api/client'
import type { ArmSummary, EvalRun, EvalRunInfo, EvalTurnResult, HistoryLine } from '../api/types'
import { Steps } from '../chat/Steps'
import { armColor, summarize, turnNumber, turnSteps } from './results'
import { TokenChart } from './TokenChart'

const ARM_HELP: Record<string, string> = {
  checkpointer: 'the whole conversation, every turn',
  jev: 'only the earlier turns Jev picks',
}

const METRICS: { key: keyof ArmSummary; label: string; lowerIsBetter?: boolean; ratio?: keyof ArmSummary }[] = [
  { key: 'correct', label: 'Correct', ratio: 'checked' },
  { key: 'grounded', label: 'Grounded', ratio: 'turns' },
  { key: 'steps', label: 'Steps', lowerIsBetter: true },
  { key: 'queries', label: 'Queries', lowerIsBetter: true },
  { key: 'skill_reads', label: 'Skill reads', lowerIsBetter: true },
  { key: 'tokens_per_turn', label: 'Tokens / turn', lowerIsBetter: true },
  { key: 'first_call_per_turn', label: 'First call / turn', lowerIsBetter: true },
]

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function Verdict({ correct }: { correct: boolean | null }) {
  if (correct === null) return <span className="verdict none" title="Nothing to check automatically">—</span>
  return correct ? <span className="verdict ok">✓</span> : <span className="verdict bad">✗</span>
}

function SentTurns({ turns }: { turns: number[] }) {
  if (turns.length === 0) return <span className="muted">none</span>
  return (
    <span className="chips" aria-label="Turns sent">
      {turns.map((t) => (
        <span key={t} className="chip">{t}</span>
      ))}
    </span>
  )
}

function Summary({ arms, summary }: { arms: string[]; summary: Record<string, ArmSummary> }) {
  const [base, other] = arms
  return (
    <div className="table-scroll">
      <table className="stats" aria-label="Summary">
        <thead>
          <tr>
            <th />
            {arms.map((arm) => (
              <th key={arm}>
                <span className="swatch" style={{ background: armColor(arm) }} />
                {arm}
                <span className="type">{ARM_HELP[arm] ?? ''}</span>
              </th>
            ))}
            {other && <th>Change</th>}
          </tr>
        </thead>
        <tbody>
          {METRICS.map((m) => {
            const a = summary[base]?.[m.key] ?? 0
            const b = other ? summary[other]?.[m.key] ?? 0 : 0
            const change = m.lowerIsBetter && a > 0 ? Math.round(((b - a) / a) * 100) : null
            return (
              <tr key={m.key}>
                <th scope="row">{m.label}</th>
                {arms.map((arm) => (
                  <td key={arm}>
                    {(summary[arm]?.[m.key] ?? 0).toLocaleString()}
                    {m.ratio && <span className="muted"> / {summary[arm]?.[m.ratio] ?? 0}</span>}
                  </td>
                ))}
                {other && (
                  <td className={change === null ? 'muted' : change < 0 ? 'better' : change > 0 ? 'worse' : 'muted'}>
                    {change === null ? '' : `${change > 0 ? '+' : ''}${change}%`}
                  </td>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

type Histories = Record<string, HistoryLine[] | 'loading' | { error: string }>

function ArmDetail({ arm, result, history }: { arm: string; result: EvalTurnResult; history: Histories[string] | undefined }) {
  const turn = turnNumber(result.turn)
  return (
    <section className="arm-detail">
      <h3>
        <span className="swatch" style={{ background: armColor(arm) }} />
        {arm} <Verdict correct={result.correct} />
      </h3>
      <dl className="facts">
        {result.sent_turns !== null && (
          <>
            <dt>Sent</dt>
            <dd><SentTurns turns={result.sent_turns} /></dd>
          </>
        )}
        <dt>Steps</dt>
        <dd>{result.steps} ({result.queries} queries, {result.skill_reads} skill reads)</dd>
        <dt>Tokens</dt>
        <dd>{result.first_call_tokens.toLocaleString()} first call, {result.input_tokens.toLocaleString()} in total</dd>
        {result.missing.length > 0 && (
          <>
            <dt>Missing</dt>
            <dd className="error">{result.missing.join(', ')}</dd>
          </>
        )}
        {result.ungrounded.length > 0 && (
          <>
            <dt>Ungrounded</dt>
            <dd>{result.ungrounded.join(', ')}</dd>
          </>
        )}
      </dl>
      {history === undefined || history === 'loading' ? (
        <p className="muted">Loading what it did…</p>
      ) : 'error' in history ? (
        <p className="error">{history.error}</p>
      ) : (
        <Steps steps={turnSteps(history, turn).steps} streaming={false} />
      )}
      <div className="answer">
        {result.stopped ? (
          <p className="error">Stopped before answering.</p>
        ) : (
          <Markdown remarkPlugins={[remarkGfm]}>{result.answer || '_(no answer)_'}</Markdown>
        )}
      </div>
    </section>
  )
}

export function ContextEvalPage() {
  const [runs, setRuns] = useState<EvalRunInfo[] | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [run, setRun] = useState<EvalRun | null>(null)
  const [conversation, setConversation] = useState<string | null>(null)
  const [repeat, setRepeat] = useState(1)
  const [openTurn, setOpenTurn] = useState<string | null>(null)
  const [histories, setHistories] = useState<Histories>({})
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listEvalRuns()
      .then((list) => {
        setRuns(list)
        setRunId(list[0]?.id ?? null)
      })
      .catch((e) => setError(errorText(e)))
  }, [])

  const selectRun = (id: string) => {
    setRunId(id)
    setRun(null)
    setOpenTurn(null)
    setHistories({})
  }

  useEffect(() => {
    if (!runId) return
    let current = true
    getEvalRun(runId)
      .then((loaded) => {
        if (!current) return
        setRun(loaded)
        setConversation(loaded.conversations[0]?.id ?? null)
        setRepeat(1)
      })
      .catch((e) => current && setError(errorText(e)))
    return () => {
      current = false
    }
  }, [runId])

  const results = useMemo(
    () => run?.results.filter((r) => r.conversation === conversation && r.run === repeat) ?? [],
    [run, conversation, repeat],
  )
  const convo = run?.conversations.find((c) => c.id === conversation)
  const arms = run?.arms ?? []
  const byTurn = useMemo(() => {
    const map = new Map<string, Record<string, EvalTurnResult>>()
    for (const r of results) map.set(r.turn, { ...map.get(r.turn), [r.arm]: r })
    return map
  }, [results])
  const turnIds = convo?.turns.map((t) => t.id).filter((id) => byTurn.has(id)) ?? []

  const toggle = (turnId: string) => {
    const opening = openTurn !== turnId
    setOpenTurn(opening ? turnId : null)
    if (!opening || !run || !conversation) return
    for (const arm of arms) {
      const key = `${conversation}/${arm}/${repeat}`
      if (histories[key]) continue
      setHistories((h) => ({ ...h, [key]: 'loading' }))
      getEvalHistory(run.id, conversation, arm, repeat)
        .then((lines) => setHistories((h) => ({ ...h, [key]: lines })))
        .catch((e) => setHistories((h) => ({ ...h, [key]: { error: errorText(e) } })))
    }
  }

  if (error) return <p className="error">{error}</p>
  if (runs === null) return <p className="muted">Loading eval runs…</p>
  if (runs.length === 0) {
    return (
      <div className="panel">
        <h2>No eval runs yet</h2>
        <p className="muted">
          Run one from <code>backend/</code>:{' '}
          <code>uv run --env-file ../.env python -m evals.memory.run --only long_chat --no-sandbox</code>
        </p>
      </div>
    )
  }

  return (
    <div className="eval-page">
      <header className="eval-header">
        <div>
          <h1>Context management with Jev</h1>
          <p className="muted">
            The same conversations, run twice: <b>checkpointer</b> sends the whole conversation every turn,{' '}
            <b>jev</b> sends only the earlier turns Jev picks (agent/context_filter.py).
          </p>
        </div>
        <div className="controls">
          <label className="field">
            Run
            <select value={runId ?? ''} onChange={(e) => selectRun(e.target.value)}>
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {when(r.created_at)} · {r.conversations.join(', ')}
                </option>
              ))}
            </select>
          </label>
          {run && run.conversations.length > 1 && (
            <label className="field">
              Conversation
              <select
                value={conversation ?? ''}
                onChange={(e) => {
                  setConversation(e.target.value)
                  setOpenTurn(null)
                }}
              >
                {run.conversations.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.id} ({c.turns.length} turns)
                  </option>
                ))}
              </select>
            </label>
          )}
          {run && run.repeats > 1 && (
            <label className="field">
              Repeat
              <select value={repeat} onChange={(e) => setRepeat(Number(e.target.value))}>
                {Array.from({ length: run.repeats }, (_, i) => (
                  <option key={i + 1} value={i + 1}>
                    {i + 1}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </header>

      {!run ? (
        <p className="muted">Loading run…</p>
      ) : (
        <>
          <p className="run-meta muted">
            {run.model} · {run.sandbox ? 'sandbox' : 'no sandbox'}
            {run.context_filter &&
              Object.entries(run.context_filter).map(([k, v]) => (
                <span key={k} className="badge">
                  {k}: {String(v)}
                </span>
              ))}
          </p>
          {convo?.about && <p className="about">{convo.about}</p>}

          <Summary arms={arms} summary={summarize(results)} />

          <TokenChart
            turns={turnIds.map(turnNumber)}
            series={Object.fromEntries(
              arms.map((arm) => [arm, turnIds.map((id) => byTurn.get(id)?.[arm]?.first_call_tokens ?? null)]),
            )}
          />

          <div className="table-scroll">
            <table className="turns" aria-label="Turns">
              <thead>
                <tr>
                  <th>#</th>
                  <th>User says</th>
                  {arms.map((arm) => (
                    <th key={arm}>{arm}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {turnIds.map((id) => {
                  const spec = convo!.turns.find((t) => t.id === id)!
                  const row = byTurn.get(id)!
                  const open = openTurn === id
                  return (
                    <Fragment key={id}>
                      <tr
                        className={open ? 'turn open' : 'turn'}
                        onClick={() => toggle(id)}
                        aria-expanded={open}
                      >
                        <td className="num">{turnNumber(id)}</td>
                        <td className="prompt">{spec.prompt}</td>
                        {arms.map((arm) => {
                          const r = row[arm]
                          return (
                            <td key={arm} className="arm-cell">
                              {r ? (
                                <>
                                  <Verdict correct={r.correct} />
                                  <span className="muted">
                                    {r.steps} step{r.steps === 1 ? '' : 's'} · {r.first_call_tokens.toLocaleString()}
                                  </span>
                                  {r.sent_turns !== null && <SentTurns turns={r.sent_turns} />}
                                </>
                              ) : (
                                <span className="muted">not run</span>
                              )}
                            </td>
                          )
                        })}
                      </tr>
                      {open && (
                        <tr className="turn-detail-row">
                          <td colSpan={2 + arms.length}>
                            {spec.expect.length > 0 && (
                              <p className="expect">
                                <b>Expected:</b> {spec.expect.join(', ')}
                                {spec.note && <span className="muted"> ({spec.note})</span>}
                              </p>
                            )}
                            <div className="turn-detail">
                              {arms.map((arm) =>
                                row[arm] ? (
                                  <ArmDetail
                                    key={arm}
                                    arm={arm}
                                    result={row[arm]}
                                    history={histories[`${conversation}/${arm}/${repeat}`]}
                                  />
                                ) : null,
                              )}
                            </div>
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

import { useEffect, useState } from 'react'

import { getLoad } from '../api/client'
import type { Load, LoadSummary } from '../api/types'
import { ConversationDetail } from './ConversationDetail'
import { MemoryChart } from './MemoryChart'

const POLL_MS = 10_000

function seconds(s: number | null): string {
  if (s === null) return '–'
  return s < 60 ? `${s.toFixed(1)}s` : `${(s / 60).toFixed(1)} min`
}

function mb(v: number | null): string {
  return v === null ? '–' : `${Math.round(v)} MB`
}

function Totals({ summary, limit }: { summary: LoadSummary; limit: number | null }) {
  const cards: [string, string, string][] = [
    ['Code steps', String(summary.steps), `in ${summary.conversations} conversation${summary.conversations === 1 ? '' : 's'}`],
    ['Run time', seconds(summary.run_seconds), 'inside the sandboxes'],
    ['CPU time', seconds(summary.cpu_seconds), 'all steps together'],
    ['Overhead', seconds(summary.overhead_seconds), 'round trips to the sandbox provider'],
    ['Peak memory', mb(summary.peak_memory_mb), limit ? `of ${limit} MB; average ${mb(summary.average_peak_memory_mb)}` : `average ${mb(summary.average_peak_memory_mb)}`],
    ['Out of memory', String(summary.out_of_memory), `${summary.failed} step${summary.failed === 1 ? '' : 's'} failed in all`],
  ]
  return (
    <div className="cards">
      {cards.map(([label, value, note]) => (
        <div key={label} className={label === 'Out of memory' && summary.out_of_memory > 0 ? 'card card-bad' : 'card'}>
          <span className="card-label">{label}</span>
          <span className="card-value">{value}</span>
          <span className="card-note">{note}</span>
        </div>
      ))}
    </div>
  )
}

/** What the agent's code has cost in the sandboxes, from each step's measurements. */
export function LoadPage() {
  const [account, setAccount] = useState<string | null>(null)
  const [load, setLoad] = useState<Load | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    const refresh = () =>
      getLoad(account)
        .then((l) => {
          if (!alive) return
          setLoad(l)
          setError(null)
        })
        .catch((e) => alive && setError(e instanceof Error ? e.message : String(e)))
    void refresh()
    const timer = setInterval(refresh, POLL_MS)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [account])

  return (
    <div className="load-page">
      <header className="eval-header">
        <div>
          <h1>Load</h1>
          <p className="muted">
            What the agent's code costs in the sandboxes: every code step is measured (time, CPU, peak
            memory). SQL questions run no code and do not appear here.
          </p>
        </div>
        {load && load.accounts.length > 0 && (
          <label className="field">
            Account
            <select value={account ?? ''} onChange={(e) => setAccount(e.target.value || null)}>
              <option value="">All accounts</option>
              {load.accounts.map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </label>
        )}
      </header>

      {error && <p className="error">{error}</p>}
      {!load ? (
        !error && <p className="muted">Loading…</p>
      ) : load.summary.steps === 0 ? (
        <div className="panel">
          <h2>No code steps yet</h2>
          <p className="muted">Ask for a chart or an analysis in Chat; its code steps appear here.</p>
        </div>
      ) : (
        <>
          <Totals summary={load.summary} limit={load.memory_limit_mb} />
          <MemoryChart steps={load.steps} />

          {account === null && load.accounts.length > 1 && (
            <div className="table-scroll">
              <table className="stats" aria-label="Per account">
                <thead>
                  <tr>
                    <th>Account</th><th>Steps</th><th>Run time</th><th>CPU</th><th>Peak memory</th><th>Out of memory</th>
                  </tr>
                </thead>
                <tbody>
                  {load.accounts.map((a) => {
                    const s = load.per_account[a]
                    return (
                      <tr key={a}>
                        <th scope="row">{a}</th>
                        <td>{s.steps}</td>
                        <td>{seconds(s.run_seconds)}</td>
                        <td>{seconds(s.cpu_seconds)}</td>
                        <td>{mb(s.peak_memory_mb)}</td>
                        <td className={s.out_of_memory ? 'worse' : ''}>{s.out_of_memory}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          <h2 className="section-title">Conversations</h2>
          <div className="table-scroll">
            <table className="turns" aria-label="Conversations">
              <thead>
                <tr>
                  <th>Account</th><th>First question</th><th>Steps</th><th>Out of memory</th>
                  <th>Moved to bigger</th><th>Peak memory</th><th>Sandbox</th>
                </tr>
              </thead>
              <tbody>
                {load.conversations.map((c) => (
                  <tr
                    key={c.conversation}
                    className={open === c.conversation ? 'turn open' : 'turn'}
                    onClick={() => setOpen(open === c.conversation ? null : c.conversation)}
                    aria-expanded={open === c.conversation}
                  >
                    <td>{c.account}</td>
                    <td className="prompt">{c.question}</td>
                    <td>{c.steps}</td>
                    <td className={c.out_of_memory ? 'worse' : ''}>{c.out_of_memory}</td>
                    <td className={c.upgrades ? 'better' : ''}>
                      {c.upgrades}
                      {c.upgrade_failures > 0 && <span className="worse"> ({c.upgrade_failures} failed)</span>}
                    </td>
                    <td className="nowrap">{mb(c.peak_memory_mb)}</td>
                    <td className="nowrap">{c.sandbox_gb === null ? '–' : `${c.sandbox_gb} GB`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {open && <ConversationDetail key={open} id={open} onClose={() => setOpen(null)} />}

          <h2 className="section-title">Code steps</h2>
          <div className="table-scroll">
            <table className="turns steps-table" aria-label="Code steps">
              <thead>
                <tr>
                  <th>When</th><th>Account</th><th>Question</th><th>Command</th>
                  <th>Run</th><th>CPU</th><th>Peak memory</th><th>Result</th>
                </tr>
              </thead>
              <tbody>
                {load.steps.map((s, i) => (
                  <tr key={`${s.conversation}-${s.ts}-${i}`}>
                    <td className="muted nowrap">{new Date(s.ts).toLocaleTimeString()}</td>
                    <td>{s.account}</td>
                    <td className="prompt" title={`conversation ${s.conversation}, turn ${s.turn}`}>{s.prompt}</td>
                    <td><code className="command" title={s.command}>{s.command}</code></td>
                    <td className="nowrap">{seconds(s.run_seconds)}</td>
                    <td className="nowrap">{seconds(s.cpu_seconds)}</td>
                    <td className="nowrap">
                      {mb(s.peak_memory_mb)}
                      {s.memory_limit_mb && <span className="muted"> of {s.memory_limit_mb}</span>}
                      {s.peak_memory_mb !== null && s.memory_limit_mb && (
                        <span className="meter">
                          <span
                            className={s.out_of_memory ? 'meter-fill meter-oom' : 'meter-fill'}
                            style={{ width: `${Math.min(100, (100 * s.peak_memory_mb) / s.memory_limit_mb)}%` }}
                          />
                        </span>
                      )}
                    </td>
                    <td className="nowrap">
                      {s.out_of_memory ? (
                        <span className="verdict bad">out of memory</span>
                      ) : s.exit_code === 0 ? (
                        <span className="verdict ok">ok</span>
                      ) : (
                        <span className="verdict bad">exit {s.exit_code}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

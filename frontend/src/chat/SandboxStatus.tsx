import { useEffect, useState } from 'react'

import { getSandboxStatus } from '../api/client'
import type { SandboxStatus as Status } from '../api/types'

const POLL_MS = 5000

const PARTS: { key: keyof Status; label: string; help: string }[] = [
  { key: 'busy', label: 'busy', help: 'A conversation is answering a message in it' },
  { key: 'idle', label: 'idle', help: 'Its conversation is waiting for the next message' },
  { key: 'starting', label: 'starting', help: 'Being started for a new conversation' },
  { key: 'warm', label: 'ready', help: 'Started ahead of time for the next new conversation' },
  { key: 'warming', label: 'warming', help: 'Being prepared for the warm pool' },
  { key: 'waiting', label: 'waiting', help: 'Waiting for a free sandbox' },
]

/**
 * How many sandboxes exist and what they are doing, refreshed every few
 * seconds and whenever a reply finishes (`refreshKey` changes).
 */
export function SandboxStatus({ refreshKey }: { refreshKey: unknown }) {
  const [status, setStatus] = useState<Status | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let alive = true
    const load = () =>
      getSandboxStatus()
        .then((s) => {
          if (!alive) return
          setStatus(s)
          setFailed(false)
        })
        .catch(() => alive && setFailed(true))
    void load()
    const timer = setInterval(load, POLL_MS)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [refreshKey])

  if (failed) return <p className="sandbox-status muted">Sandbox status unavailable</p>
  if (!status) return null
  if (!status.enabled) return <p className="sandbox-status muted">Code execution is off (no sandbox provider)</p>

  const shown = PARTS.filter((p) => p.key === 'busy' || p.key === 'idle' || (status[p.key] as number) > 0)
  const free = status.max === null ? null : status.max - status.in_use
  return (
    <p className="sandbox-status" aria-label="Sandboxes">
      <span className="muted">Sandboxes ({status.provider})</span>
      {shown.map((p) => (
        <span key={p.key} className={`sandbox-count sandbox-${p.key}`} title={p.help}>
          <b>{status[p.key] as number}</b> {p.label}
        </span>
      ))}
      <span
        className="sandbox-count muted"
        title={
          `${status.in_use} in use, counting ready ones. ` +
          `Idle sandboxes are deleted after ${status.idle_minutes} minutes without a message.`
        }
      >
        {free === null ? `${status.in_use} in use` : `${free} of ${status.max} free`}
      </span>
    </p>
  )
}

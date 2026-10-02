import { useEffect, useState, type ReactNode } from 'react'

import { getConversationTimeline } from '../api/client'
import type { ConversationTimeline, TimelineItem } from '../api/types'

type EventItem = Extract<TimelineItem, { kind: 'event' }>

const num = (v: unknown) => (typeof v === 'number' ? v : null)
const secs = (v: unknown) => (num(v) === null ? '' : ` in ${num(v)!.toFixed(1)}s`)

function bytes(v: unknown): string {
  const n = num(v)
  if (n === null) return ''
  return n < 1024 * 1024 ? ` (${Math.round(n / 1024)} KB)` : ` (${(n / 1024 / 1024).toFixed(1)} MB)`
}

/** A sandbox event in words, and how it should look. */
function describe(e: EventItem): { tone: string; title: string; detail?: string } {
  switch (e.event) {
    case 'sandbox_ready':
      return {
        tone: 'sandbox',
        title: e.how === 'warm' ? 'Sandbox ready: a warm one was handed over' : e.how === 'replacement'
          ? 'Replacement sandbox ready' : 'Sandbox ready: a new one was started',
        detail: `${num(e.memory_gb)} GB${secs(e.seconds)}`,
      }
    case 'upgrade_started':
      return {
        tone: 'warn',
        title: `Moving to a bigger sandbox, ${num(e.from_gb)} GB → ${num(e.to_gb)} GB`,
        detail: `${e.reason ? `Why: ${String(e.reason)}. ` : ''}${num(e.cpu)} vCPU; ${num(e.memory_used_gb)} GB of the budget in use while both exist`,
      }
    case 'bigger_created':
      return { tone: 'move', title: `Bigger sandbox created: ${num(e.memory_gb)} GB, ${num(e.cpu)} vCPU`, detail: secs(e.seconds).trim() }
    case 'packages_installed':
      return { tone: 'move', title: 'Packages installed in it', detail: `${(e.packages as string[] | undefined)?.join(', ') ?? ''}${secs(e.seconds)}` }
    case 'files_copied':
      return { tone: 'move', title: `Work folder copied over${bytes(e.bytes)}`, detail: secs(e.seconds).trim() }
    case 'switched':
      return { tone: 'good', title: `Switched to the ${num(e.memory_gb)} GB sandbox`, detail: 'the step runs again there' }
    case 'old_deleted':
      return { tone: 'move', title: `Old ${num(e.memory_gb)} GB sandbox deleted`, detail: `${num(e.memory_used_gb)} GB of the budget in use now` }
    case 'upgrade_failed':
      return { tone: 'bad', title: 'Could not move to a bigger sandbox', detail: String(e.reason ?? '') }
    case 'sandbox_unavailable':
      return { tone: 'bad', title: 'No sandbox', detail: String(e.reason ?? '') }
    case 'sandbox_dead':
      return { tone: 'bad', title: 'The sandbox stopped answering; starting a replacement' }
    default:
      return { tone: 'sandbox', title: e.event }
  }
}

/** The sandbox's phases across the conversation: running at a size, or moving. */
function SandboxLane({ items, start, end }: { items: TimelineItem[]; start: number; end: number }) {
  const events = items.filter((i): i is EventItem => i.kind === 'event')
  const phases: { label: string; tone: string; from: number; to: number }[] = []
  let size: number | null = null
  let from = start
  for (const e of events) {
    const at = Date.parse(e.ts)
    if (e.event === 'sandbox_ready') {
      size = num(e.memory_gb)
      from = at
    } else if (e.event === 'upgrade_started') {
      if (size !== null) phases.push({ label: `${size} GB sandbox`, tone: 'run', from, to: at })
      from = at
    } else if (e.event === 'switched') {
      phases.push({ label: 'moving', tone: 'moving', from, to: at })
      size = num(e.memory_gb)
      from = at
    }
  }
  if (size !== null) phases.push({ label: `${size} GB sandbox`, tone: size > 1 ? 'run-big' : 'run', from, to: end })
  if (phases.length === 0 || end <= start) return null
  const total = end - start
  return (
    <div className="lane" aria-label="Sandbox over time">
      {phases.map((p, i) => (
        <div
          key={i}
          className={`lane-phase lane-${p.tone}`}
          style={{ left: `${(100 * (p.from - start)) / total}%`, width: `${Math.max(0.5, (100 * (p.to - p.from)) / total)}%` }}
          title={`${p.label}: ${((p.to - p.from) / 1000).toFixed(1)}s`}
        >
          <span>{p.label} · {((p.to - p.from) / 1000).toFixed(0)}s</span>
        </div>
      ))}
    </div>
  )
}

function Item({ item, start }: { item: TimelineItem; start: number }) {
  const at = `+${((Date.parse(item.ts) - start) / 1000).toFixed(1)}s`
  let tone = 'agent'
  let title: string
  let detail: ReactNode = null
  switch (item.kind) {
    case 'question':
      tone = 'user'
      title = 'Question'
      detail = item.text
      break
    case 'action':
      title = `Agent: ${item.tool}`
      detail = item.detail && <code>{item.detail}</code>
      break
    case 'answer':
      tone = 'good'
      title = 'Answer'
      detail = item.text
      break
    case 'step': {
      const limit = item.memory_limit_mb
      tone = item.out_of_memory ? 'bad' : item.exit_code === 0 ? 'step' : 'warn'
      const role =
        item.strategy === 'rewrite' ? `Rewrite ${item.rewrite} · `
        : item.strategy === 'same code' ? 'Same code again · '
        : item.strategy === 'bigger sandbox' ? 'In the bigger sandbox · '
        : ''
      title = role + (item.out_of_memory
        ? `Code step killed: out of memory at ${Math.round(item.peak_memory_mb ?? 0)} MB of ${limit} MB`
        : `Code step ${item.exit_code === 0 ? 'done' : `failed (exit ${item.exit_code})`}: ${Math.round(item.peak_memory_mb ?? 0)} MB of ${limit ?? '?'} MB`)
      detail = (
        <>
          <code>{item.command}</code>
          <span className="tl-meta">
            ran {item.run_seconds?.toFixed(1) ?? '–'}s · CPU {item.cpu_seconds?.toFixed(1) ?? '–'}s
          </span>
          {item.peak_memory_mb !== null && limit && (
            <span className="meter meter-wide">
              <span
                className={item.out_of_memory ? 'meter-fill meter-oom' : 'meter-fill'}
                style={{ width: `${Math.min(100, (100 * item.peak_memory_mb) / limit)}%` }}
              />
            </span>
          )}
        </>
      )
      break
    }
    case 'event': {
      const d = describe(item)
      tone = d.tone
      title = d.title
      detail = d.detail
      break
    }
  }
  return (
    <li className={`tl-item tl-${tone}`}>
      <span className="tl-time">{at}</span>
      <span className="tl-dot" />
      <div className="tl-body">
        <div className="tl-title">{title}</div>
        {detail && <div className="tl-detail">{detail}</div>}
      </div>
    </li>
  )
}

/** One conversation as it happened: the agent's steps and its sandbox's life. */
export function ConversationDetail({ id, onClose }: { id: string; onClose: () => void }) {
  const [data, setData] = useState<ConversationTimeline | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    getConversationTimeline(id)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [id])

  if (error) return <p className="error">{error}</p>
  if (!data) return <p className="muted">Loading the timeline…</p>
  const times = data.timeline.map((i) => Date.parse(i.ts)).filter((t) => !Number.isNaN(t))
  const start = Math.min(...times)
  const end = Math.max(...times)

  return (
    <section className="panel detail" aria-label="Conversation timeline">
      <header className="detail-header">
        <div>
          <h2>Conversation {data.conversation.slice(0, 8)}</h2>
          <p className="muted">{data.account} · {((end - start) / 1000).toFixed(0)}s from first message to last line</p>
        </div>
        <button type="button" className="secondary" onClick={onClose}>Close</button>
      </header>
      <SandboxLane items={data.timeline} start={start} end={end} />
      <ol className="timeline">
        {data.timeline.map((item, i) => (
          <Item key={i} item={item} start={start} />
        ))}
      </ol>
    </section>
  )
}

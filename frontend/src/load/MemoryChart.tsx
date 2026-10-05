import type { LoadStep } from '../api/types'
import { groups } from './groups'

const W = 760
const H = 236
const PAD = { left: 52, right: 12, top: 14, bottom: 56 }
const GAP = 0.8
/** Space between conversations, in bar slots. */

function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`
}

/**
 * Peak memory of each code step, grouped by conversation (oldest left). Each
 * bar's dashed cap is that step's own memory limit; where a step ran out of
 * memory and the next ran in a bigger sandbox, the move is marked between them.
 */
export function MemoryChart({
  steps,
  selected = null,
  onSelect,
}: {
  steps: LoadStep[]
  /** The open conversation: its bars stand out, the others fade. */
  selected?: string | null
  onSelect?: (conversation: string) => void
}) {
  const grouped = groups(steps)
  if (grouped.length === 0) return null
  const all = grouped.flatMap((g) => g.steps)
  const highest = Math.max(...all.map((s) => Math.max(s.peak_memory_mb!, s.memory_limit_mb ?? 0)))
  // Round ticks: every 256 MB for small sandboxes, every 1 GB above 2 GB.
  const tickStep = highest * 1.05 > 2048 ? 1024 : 256
  const top = Math.ceil((highest * 1.05) / tickStep) * tickStep
  const y = (mb: number) => H - PAD.bottom - (mb / top) * (H - PAD.top - PAD.bottom)
  const ticks = Array.from({ length: top / tickStep + 1 }, (_, i) => i * tickStep)
  const slots = all.length + GAP * (grouped.length - 1)
  const slot = (W - PAD.left - PAD.right) / slots

  const layout = grouped.map((g, gi) => {
    const before = grouped.slice(0, gi).reduce((n, other) => n + other.steps.length + GAP, 0)
    return { ...g, number: gi + 1, x: PAD.left + before * slot, width: g.steps.length * slot }
  })

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Peak memory per step">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={PAD.left - 6} y={y(t)} className="tick" textAnchor="end" dominantBaseline="middle">
              {t >= 1024 && t % 1024 === 0 ? `${t / 1024} GB` : `${t} MB`}
            </text>
          </g>
        ))}
        {layout.map((g) => {
          const faded = selected !== null && selected !== g.conversation
          return (
            <g
              key={g.conversation}
              className={faded ? 'conv-group bar-faded' : selected === g.conversation ? 'conv-group bar-selected' : 'conv-group'}
              onClick={() => onSelect?.(g.conversation)}
              role={onSelect ? 'button' : undefined}
              aria-label={onSelect ? `Open conversation ${g.number}` : undefined}
            >
              <rect x={g.x} y={PAD.top} width={g.width} height={y(0) - PAD.top} className="conv-band" />
              {g.steps.map((s, i) => {
                const x = g.x + i * slot + slot * 0.15
                const width = Math.max(slot * 0.7, 1)
                return (
                  <g key={`${s.ts}-${i}`}>
                    <rect
                      x={x}
                      width={width}
                      y={y(s.peak_memory_mb!)}
                      height={y(0) - y(s.peak_memory_mb!)}
                      className={s.out_of_memory ? 'bar bar-oom' : 'bar'}
                    >
                      <title>{`Conversation ${g.number}, step ${i + 1}: ${s.peak_memory_mb} MB of ${s.memory_limit_mb ?? '?'} MB`}</title>
                    </rect>
                    {s.memory_limit_mb !== null && (
                      <line x1={x - 4} x2={x + width + 4} y1={y(s.memory_limit_mb)} y2={y(s.memory_limit_mb)} className="limit" />
                    )}
                    <text
                      x={x + width / 2}
                      y={y(0) + 13}
                      className={s.strategy === 'rewrite' ? 'tick rewrite-label' : 'tick'}
                      textAnchor="middle"
                    >
                      {s.strategy === 'rewrite' ? `rewrite ${s.rewrite}` : s.strategy === 'same code' ? 'same again' : `step ${i + 1}`}
                    </text>
                  </g>
                )
              })}
              {/* Drawn after the bars, so no bar covers it: where a step ran out of
                  memory and the next ran in a bigger sandbox. */}
              {g.steps.map((s, i) => {
                const next = g.steps[i + 1]
                if (!s.out_of_memory || !next || (next.memory_limit_mb ?? 0) <= (s.memory_limit_mb ?? 0)) return null
                const from = g.x + i * slot + slot * 0.5
                const to = g.x + (i + 1) * slot + slot * 0.5
                // The arrow arcs under the bigger sandbox's limit line; the label sits above it.
                const arc = y(next.peak_memory_mb ?? 0) - 18
                return (
                  <g key={`move-${i}`} className="move">
                    <path d={`M ${from} ${y(s.memory_limit_mb ?? 0) - 4} Q ${(from + to) / 2} ${arc} ${to} ${y(next.peak_memory_mb ?? 0) - 4}`} className="move-arrow" />
                    <text x={(from + to) / 2} y={y(next.memory_limit_mb ?? 0) - 8} className="move-mark" textAnchor="middle">
                      {`out of memory → moved to ${Math.round((next.memory_limit_mb ?? 0) / 1024)} GB`}
                    </text>
                  </g>
                )
              })}
              <text x={g.x + g.width / 2} y={y(0) + 32} className="conv-label" textAnchor="middle">
                {`Conversation ${g.number}`}
              </text>
              <text x={g.x + g.width / 2} y={y(0) + 46} className="tick" textAnchor="middle">
                {clip(g.question, Math.max(12, Math.floor(g.width / 6.2)))}
              </text>
            </g>
          )
        })}
      </svg>
      <figcaption>
        <span className="muted">
          Bars grouped by conversation; each bar is one code step. Height: its peak memory; dashed: its sandbox's
          memory limit; red: killed for running out of memory. Click a conversation to open it.
        </span>
      </figcaption>
    </figure>
  )
}

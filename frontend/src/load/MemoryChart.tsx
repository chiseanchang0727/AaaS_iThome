import type { LoadStep } from '../api/types'

const W = 760
const H = 200
const PAD = { left: 52, right: 12, top: 14, bottom: 20 }

/**
 * Peak memory of each step, oldest left. Each bar's dashed cap is that step's
 * own memory limit: it rises when a conversation moved to a bigger sandbox.
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
  const measured = [...steps].reverse().filter((s) => s.peak_memory_mb !== null)
  if (measured.length === 0) return null
  const highest = Math.max(...measured.map((s) => Math.max(s.peak_memory_mb!, s.memory_limit_mb ?? 0)))
  // Round ticks: every 256 MB for small sandboxes, every 1 GB above 2 GB.
  const tickStep = highest * 1.05 > 2048 ? 1024 : 256
  const top = Math.ceil((highest * 1.05) / tickStep) * tickStep
  const y = (mb: number) => H - PAD.bottom - (mb / top) * (H - PAD.top - PAD.bottom)
  const slot = (W - PAD.left - PAD.right) / measured.length
  const ticks = Array.from({ length: top / tickStep + 1 }, (_, i) => i * tickStep)

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
        {measured.map((s, i) => {
          const x = PAD.left + i * slot + slot * 0.15
          const width = Math.max(slot * 0.7, 1)
          return (
            <g
              key={`${s.conversation}-${s.ts}-${i}`}
              className={selected === null ? 'bar-group' : selected === s.conversation ? 'bar-group bar-selected' : 'bar-group bar-faded'}
              onClick={() => onSelect?.(s.conversation)}
              role={onSelect ? 'button' : undefined}
              aria-label={onSelect ? `Open the conversation of a ${Math.round(s.peak_memory_mb!)} MB step` : undefined}
            >
              <rect
                x={x}
                width={width}
                y={y(s.peak_memory_mb!)}
                height={y(0) - y(s.peak_memory_mb!)}
                className={s.out_of_memory ? 'bar bar-oom' : 'bar'}
              >
                <title>{`${s.peak_memory_mb} MB of ${s.memory_limit_mb ?? '?'} MB · ${s.account} · ${s.command.slice(0, 80)}`}</title>
              </rect>
              {s.memory_limit_mb !== null && (
                <line x1={x - 4} x2={x + width + 4} y1={y(s.memory_limit_mb)} y2={y(s.memory_limit_mb)} className="limit" />
              )}
            </g>
          )
        })}
      </svg>
      <figcaption>
        <span className="muted">
          each bar is one code step, oldest left; height: its peak memory; dashed: its sandbox's memory limit;
          red: killed for running out of memory. Click a bar to open its conversation.
        </span>
      </figcaption>
    </figure>
  )
}

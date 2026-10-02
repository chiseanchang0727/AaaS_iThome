import type { LoadStep } from '../api/types'

const W = 760
const H = 200
const PAD = { left: 52, right: 12, top: 14, bottom: 20 }

/**
 * Peak memory of each step, oldest left. Each bar's dashed cap is that step's
 * own memory limit: it rises when a conversation moved to a bigger sandbox.
 */
export function MemoryChart({ steps }: { steps: LoadStep[] }) {
  const measured = [...steps].reverse().filter((s) => s.peak_memory_mb !== null)
  if (measured.length === 0) return null
  const highest = Math.max(...measured.map((s) => Math.max(s.peak_memory_mb!, s.memory_limit_mb ?? 0)))
  // Scale in steps of 256 MB, with room above the highest limit.
  const top = Math.ceil((highest * 1.05) / 256) * 256
  const y = (mb: number) => H - PAD.bottom - (mb / top) * (H - PAD.top - PAD.bottom)
  const slot = (W - PAD.left - PAD.right) / measured.length
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * top))

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Peak memory per step">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={PAD.left - 6} y={y(t)} className="tick" textAnchor="end" dominantBaseline="middle">
              {t} MB
            </text>
          </g>
        ))}
        {measured.map((s, i) => {
          const x = PAD.left + i * slot + slot * 0.15
          const width = Math.max(slot * 0.7, 1)
          return (
            <g key={`${s.conversation}-${s.ts}-${i}`}>
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
          peak memory of each code step, oldest left; dashed: that step's sandbox memory limit; red: killed for
          running out of memory
        </span>
      </figcaption>
    </figure>
  )
}

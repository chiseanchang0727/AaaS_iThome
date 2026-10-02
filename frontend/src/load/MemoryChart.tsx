import type { LoadStep } from '../api/types'

const W = 760
const H = 200
const PAD = { left: 52, right: 12, top: 14, bottom: 20 }

/** Peak memory of each step, oldest left, against the sandbox's memory limit. */
export function MemoryChart({ steps, limit }: { steps: LoadStep[]; limit: number | null }) {
  const measured = [...steps].reverse().filter((s) => s.peak_memory_mb !== null)
  if (measured.length === 0) return null
  // Scale in steps of 256 MB, with room above the limit line for its label.
  const top = Math.ceil((Math.max(limit ?? 0, ...measured.map((s) => s.peak_memory_mb!)) * 1.05) / 256) * 256
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
        {measured.map((s, i) => (
          <rect
            key={`${s.conversation}-${s.ts}-${i}`}
            x={PAD.left + i * slot + slot * 0.15}
            width={Math.max(slot * 0.7, 1)}
            y={y(s.peak_memory_mb!)}
            height={y(0) - y(s.peak_memory_mb!)}
            className={s.out_of_memory ? 'bar bar-oom' : 'bar'}
          >
            <title>{`${s.peak_memory_mb} MB · ${s.account} · ${s.command.slice(0, 80)}`}</title>
          </rect>
        ))}
        {limit !== null && (
          <g>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(limit)} y2={y(limit)} className="limit" />
            <text x={W - PAD.right} y={y(limit) - 4} className="tick" textAnchor="end">
              sandbox limit {limit} MB
            </text>
          </g>
        )}
      </svg>
      <figcaption>
        <span className="muted">peak memory of each code step, oldest left; red: killed for running out of memory</span>
      </figcaption>
    </figure>
  )
}

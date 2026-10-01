/** First-call input tokens per turn, one line per arm. Plain SVG. */

import { armColor } from './results'

const W = 760
const H = 220
const PAD = { left: 52, right: 12, top: 12, bottom: 30 }

function niceMax(value: number): number {
  if (value <= 0) return 1
  const step = 10 ** Math.floor(Math.log10(value))
  return Math.ceil(value / step) * step
}

export function TokenChart({ turns, series }: { turns: number[]; series: Record<string, (number | null)[]> }) {
  const all = Object.values(series).flat().filter((v): v is number => v !== null)
  const max = niceMax(Math.max(0, ...all))
  const x = (i: number) => PAD.left + (turns.length < 2 ? 0 : (i / (turns.length - 1)) * (W - PAD.left - PAD.right))
  const y = (v: number) => H - PAD.bottom - (v / max) * (H - PAD.top - PAD.bottom)
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * max))

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="First-call tokens per turn">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={PAD.left - 6} y={y(t)} className="tick" textAnchor="end" dominantBaseline="middle">
              {t >= 1000 ? `${t / 1000}K` : t}
            </text>
          </g>
        ))}
        {turns.map((turn, i) => (
          <text key={turn} x={x(i)} y={H - 10} className="tick" textAnchor="middle">
            {turn}
          </text>
        ))}
        {Object.entries(series).map(([arm, values]) => {
          const points = values.flatMap((v, i) => (v === null ? [] : [[x(i), y(v), v, turns[i]] as const]))
          return (
            <g key={arm} style={{ color: armColor(arm) }}>
              <polyline points={points.map(([px, py]) => `${px},${py}`).join(' ')} className="line" />
              {points.map(([px, py, v, turn]) => (
                <circle key={turn} cx={px} cy={py} r={3} className="dot">
                  <title>{`${arm}, turn ${turn}: ${v.toLocaleString()} tokens`}</title>
                </circle>
              ))}
            </g>
          )
        })}
      </svg>
      <figcaption>
        {Object.keys(series).map((arm) => (
          <span key={arm} className="legend">
            <span className="swatch" style={{ background: armColor(arm) }} />
            {arm}
          </span>
        ))}
        <span className="muted">first model call of each turn, input tokens (by turn)</span>
      </figcaption>
    </figure>
  )
}

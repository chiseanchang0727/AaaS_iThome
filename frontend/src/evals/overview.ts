import type { EvalStatus, SystemResult } from '../api/types'

export const DIMENSIONS = [
  { key: 'correct', label: 'Correct' },
  { key: 'grounded', label: 'Grounded' },
  { key: 'instruction_following', label: 'Instructions' },
  { key: 'execution_strategy', label: 'Strategy' },
] as const

export type Dimension = (typeof DIMENSIONS)[number]['key']
export const STATUSES: EvalStatus[] = ['pass', 'fail', 'unknown', 'judge_error']

/** How many turns got each verdict on one dimension. */
export function verdictCounts(results: SystemResult[], dimension: Dimension): Record<EvalStatus, number> {
  const counts: Record<EvalStatus, number> = { pass: 0, fail: 0, unknown: 0, judge_error: 0 }
  for (const r of results) counts[r[dimension].status in counts ? r[dimension].status : 'unknown'] += 1
  return counts
}

/** "full": the whole conversation was sent; "jev": Jev picked the earlier turns. */
export type ContextGroup = 'full' | 'jev'

export function contextGroup(r: SystemResult): ContextGroup {
  return r.context ? 'jev' : 'full'
}

export interface GroupStats {
  turns: number
  passRate: Record<Dimension, number | null>
  steps: number
  queries: number
  totalTokens: number
  inputTokens: number
  runtime: number
  errors: number
}

const mean = (values: number[]) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0)

/** Pass rates (passes / turns) and per-turn averages for a set of turns. */
export function groupStats(results: SystemResult[]): GroupStats {
  const n = results.length
  const passRate = Object.fromEntries(
    DIMENSIONS.map((d) => [d.key, n ? verdictCounts(results, d.key).pass / n : null]),
  ) as Record<Dimension, number | null>
  return {
    turns: n,
    passRate,
    steps: mean(results.map((r) => r.efficiency.steps)),
    queries: mean(results.map((r) => r.efficiency.queries)),
    totalTokens: mean(results.map((r) => r.efficiency.total_tokens)),
    inputTokens: mean(results.map((r) => r.efficiency.input_tokens)),
    runtime: mean(results.map((r) => r.efficiency.runtime_seconds)),
    errors: mean(results.map((r) => r.recovery.errors.length)),
  }
}

export interface RecoveryTotals {
  errors: number
  identicalRetries: number
  oom: number
  sandboxMoves: number
  stopped: number
  ungroundedValues: number
}

export function recoveryTotals(results: SystemResult[]): RecoveryTotals {
  const sum = (f: (r: SystemResult) => number) => results.reduce((t, r) => t + f(r), 0)
  return {
    errors: sum((r) => r.recovery.errors.length),
    identicalRetries: sum((r) => r.recovery.identical_retries),
    oom: sum((r) => r.recovery.oom_events),
    sandboxMoves: sum((r) => r.recovery.sandbox_moves),
    stopped: sum((r) => (r.stopped ? 1 : 0)),
    ungroundedValues: sum((r) => r.grounded.ungrounded_values.length),
  }
}

export function percent(rate: number | null): string {
  return rate === null ? '—' : `${Math.round(rate * 100)}%`
}

/** 1,284 / 12.9K / 1.2M */
export function compact(n: number): string {
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (Math.abs(n) >= 1e4) return `${(n / 1e3).toFixed(1)}K`
  return Math.round(n).toLocaleString()
}

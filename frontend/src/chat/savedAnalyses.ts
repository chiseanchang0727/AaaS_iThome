import type { Step } from './state'

/** Analyses this turn saved or changed: the save_analysis calls that succeeded. */
export function savedAnalyses(steps: Step[]): { id: string; title: string; version: number | null }[] {
  return steps.flatMap((s) => {
    if (s.kind !== 'tool' || s.name !== 'save_analysis' || !s.result) return []
    const saved = /^Saved analysis ([a-z0-9]+):/.exec(s.result)
    const updated = /^Updated analysis ([a-z0-9]+) to version (\d+):/.exec(s.result)
    const id = saved?.[1] ?? updated?.[1]
    return id ? [{ id, title: String(s.args.title ?? 'analysis'), version: updated ? Number(updated[2]) : null }] : []
  })
}

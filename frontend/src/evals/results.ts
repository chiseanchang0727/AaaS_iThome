import type { Step } from '../chat/state'
import type { ArmSummary, EvalTurnResult, HistoryLine } from '../api/types'

const ARM_COLORS: Record<string, string> = {
  checkpointer: 'var(--chart-a)',
  jev: 'var(--chart-b)',
}

export function armColor(arm: string): string {
  return ARM_COLORS[arm] ?? 'var(--muted)'
}

/** Totals per arm; the same numbers as backend/api/evals.py's summarize. */
export function summarize(results: EvalTurnResult[]): Record<string, ArmSummary> {
  const byArm: Record<string, EvalTurnResult[]> = {}
  for (const r of results) (byArm[r.arm] ??= []).push(r)
  const sum = (rs: EvalTurnResult[], f: (r: EvalTurnResult) => number) => rs.reduce((t, r) => t + f(r), 0)
  return Object.fromEntries(
    Object.entries(byArm).map(([arm, rs]) => {
      const checked = rs.filter((r) => r.correct !== null)
      return [
        arm,
        {
          turns: rs.length,
          checked: checked.length,
          correct: checked.filter((r) => r.correct).length,
          grounded: rs.filter((r) => r.ungrounded.length === 0).length,
          steps: sum(rs, (r) => r.steps),
          queries: sum(rs, (r) => r.queries),
          skill_reads: sum(rs, (r) => r.skill_reads),
          tokens_per_turn: Math.floor(sum(rs, (r) => r.input_tokens) / rs.length),
          first_call_per_turn: Math.floor(sum(rs, (r) => r.first_call_tokens ?? 0) / rs.length),
        },
      ]
    }),
  )
}

/** "marathon.12" -> 12 */
export function turnNumber(turnId: string): number {
  return Number(turnId.slice(turnId.lastIndexOf('.') + 1))
}

/** One turn of a history as the chat's steps, plus its final answer. */
export function turnSteps(lines: HistoryLine[], turn: number): { steps: Step[]; answer: string } {
  const steps: Step[] = []
  const tools = new Map<string, Extract<Step, { kind: 'tool' }>>()
  let answer = ''
  for (const line of lines) {
    if (line.turn !== turn) continue
    if (line.role === 'assistant' && line.tool_calls?.length) {
      if (line.content.trim()) steps.push({ kind: 'thinking', text: line.content.trim() })
      for (const call of line.tool_calls) {
        const step = { kind: 'tool' as const, id: call.id, name: call.name, args: call.args }
        tools.set(call.id, step)
        steps.push(step)
      }
    } else if (line.role === 'assistant') {
      answer = line.content
    } else if (line.role === 'tool' && line.tool_call_id) {
      const step = tools.get(line.tool_call_id)
      if (step) Object.assign(step, { result: line.content, error: line.error ?? false })
    }
  }
  return { steps, answer }
}

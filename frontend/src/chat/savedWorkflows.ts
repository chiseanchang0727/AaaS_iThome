import type { Step } from './state'

export interface TouchedWorkflow {
  id: string
  /** How often the turn has touched it so far (a run starting counts): a preview reloads when it changes. */
  touched: number
  /** A run_workflow call on it has not answered yet: its run may not have begun, so the preview keeps checking. */
  pending: boolean
}

/** Workflows this turn saved, changed or ran (save_workflow, run_workflow), each once, in order. */
export function savedWorkflows(steps: Step[]): TouchedWorkflow[] {
  const found: TouchedWorkflow[] = []
  const touch = (id: string, pending = false) => {
    const known = found.find((w) => w.id === id)
    if (known) {
      known.touched += 1
      known.pending ||= pending
    } else found.push({ id, touched: 1, pending })
  }
  for (const s of steps) {
    if (s.kind !== 'tool' || (s.name !== 'save_workflow' && s.name !== 'run_workflow')) continue
    if (s.result) {
      const done = /^(?:Saved|Updated|Ran) workflow ([a-z0-9]+):/.exec(s.result)
      if (done) touch(done[1])
      if (done && s.name === 'run_workflow') touch(done[1])  // started and ended: two touches, so ending reloads too
    } else if (s.name === 'run_workflow' && typeof s.args.workflow_id === 'string') {
      touch(s.args.workflow_id.trim(), true)  // started, not ended: the preview follows the run
    }
  }
  return found
}

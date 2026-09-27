import type { Step } from './state'

/** Tool arguments, shown the way they read best: SQL as SQL, the rest as JSON. */
function Args({ args }: { args: Record<string, unknown> }) {
  if (typeof args.sql === 'string') {
    const rest = Object.entries(args).filter(([k]) => k !== 'sql')
    return (
      <>
        <pre className="code">{args.sql.trim()}</pre>
        {rest.map(([k, v]) => (
          <div key={k} className="arg">
            {k}: <code>{String(v)}</code>
          </div>
        ))}
      </>
    )
  }
  if (typeof args.command === 'string') return <pre className="code">{args.command.trim()}</pre>
  if (typeof args.file_path === 'string' && typeof args.content !== 'string') {
    return <code className="arg">{args.file_path}</code>
  }
  return <pre className="code">{JSON.stringify(args, null, 2)}</pre>
}

/**
 * What the agent did to reach its answer (Reason -> Act -> Observe), collapsed
 * by default. Open while the answer is still streaming.
 */
export function Steps({ steps, streaming }: { steps: Step[]; streaming: boolean }) {
  if (steps.length === 0) return null
  const tools = steps.filter((s) => s.kind === 'tool').length
  const label = streaming ? `Working… ${tools} step${tools === 1 ? '' : 's'}` : `${tools} step${tools === 1 ? '' : 's'}`

  return (
    <details className="steps" open={streaming}>
      <summary>{label}</summary>
      <ol>
        {steps.map((step, i) =>
          step.kind === 'thinking' ? (
            <li key={i} className="step-thinking">{step.text}</li>
          ) : (
            <li key={step.id} className={step.error ? 'step-tool step-error' : 'step-tool'}>
              <div className="step-name">
                {step.name}
                {step.result === undefined && streaming && <span className="spinner" aria-label="running" />}
              </div>
              <Args args={step.args} />
              {step.result !== undefined && <pre className="result">{step.result}</pre>}
            </li>
          ),
        )}
      </ol>
    </details>
  )
}

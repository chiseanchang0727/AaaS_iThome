import { Background, ReactFlow, ReactFlowProvider } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router'

import { getWorkflow, workflowReportUrl } from '../api/client'
import type { Workflow } from '../api/types'
import { NODE_TYPES, toFlow, withRun } from '../workflows/flow'
import { RunSummary } from '../workflows/nodes'
import { savedWorkflows } from './savedWorkflows'
import type { Step } from './state'

const POLL_MS = 2000

/**
 * A workflow the agent built, in the chat: its canvas (read-only, live while it runs), how its last
 * run went, and links to the workflow and its report. It shows the workflow as it is now, which may
 * have changed since the conversation.
 */
function WorkflowPreview({ id, touched, pending, past }: { id: string; touched: number; pending: boolean; past: boolean }) {
  const [workflow, setWorkflow] = useState<Workflow | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Load it, and again each time the turn touches it (saved, a run started, a run ended).
  useEffect(() => {
    getWorkflow(id)
      .then(setWorkflow)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [id, touched])

  // While it runs, or the agent's run_workflow has not answered (the run may not have begun), check again:
  // the canvas's pills follow the run.
  const running = workflow?.running ?? false
  useEffect(() => {
    if (!running && !pending) return
    const timer = setTimeout(() => {
      getWorkflow(id).then(setWorkflow).catch(() => {})
    }, POLL_MS)
    return () => clearTimeout(timer)
  }, [running, pending, workflow, id])

  const run = workflow?.runs[0] ?? null
  const nodes = useMemo(() => {
    if (!workflow) return []
    const shape = toFlow(workflow)
    return withRun(shape.nodes, shape.edges, run, id)
  }, [workflow, run, id])
  const edges = useMemo(() => (workflow ? toFlow(workflow).edges : []), [workflow])

  if (error) return <p className="error small">This workflow could not be loaded: {error}</p>
  if (!workflow) return <p className="muted small">Loading the workflow…</p>
  const report = run ? Object.keys(run.reports ?? {})[0] : undefined
  return (
    <figure className="workflow-preview" aria-label={`Workflow ${workflow.title}`}>
      <figcaption>
        <span className="workflow-preview-title">{workflow.title}</span>
        <span className="workflow-preview-links">
          <Link to={`/workflows/${id}`}>Open workflow</Link>
          {run && report && (
            <a href={workflowReportUrl(id, run.id, report)} target="_blank" rel="noopener noreferrer">
              Open report ↗
            </a>
          )}
        </span>
      </figcaption>
      <div className="workflow-preview-canvas">
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            defaultEdgeOptions={{ animated: running }}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            zoomOnScroll={false}
            panOnScroll={false}
            preventScrolling={false}
            colorMode="system"
            fitView
            fitViewOptions={{ maxZoom: 0.9, padding: 0.08 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background />
          </ReactFlow>
        </ReactFlowProvider>
      </div>
      {run ? <RunSummary run={run} /> : <p className="muted small">Not run yet.</p>}
      {past && <p className="muted small">Shown as the workflow is now; it may have changed since.</p>}
    </figure>
  )
}

/** A preview for each workflow a turn saved, changed or ran. */
export function WorkflowPreviews({ steps, past = false }: { steps: Step[]; past?: boolean }) {
  return (
    <>
      {savedWorkflows(steps).map((w) => (
        <WorkflowPreview key={w.id} id={w.id} touched={w.touched} pending={w.pending} past={past} />
      ))}
    </>
  )
}

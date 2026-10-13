/** The canvas's node boxes, a status pill, and a run's one-line summary: components only (flow.ts has the rest). */
import { Handle, Position, type NodeProps } from '@xyflow/react'

import type { WorkflowRun } from '../api/types'
import { STATUS_TEXT, onTables, sandboxesStarted, took, type AnalysisFlowNode, type BatchFlowNode, type ReportFlowNode, type Status } from './flow'

export function StatusPill({ status }: { status: Status | undefined }) {
  if (!status) return null
  return <span className={`wf-status wf-status-${status}`}>{STATUS_TEXT[status]}</span>
}

/** One line about a run: how long it took, how many analyses, how many sandboxes. */
export function RunSummary({ run }: { run: WorkflowRun }) {
  const sandboxes = sandboxesStarted(run)
  const analyses = run.nodes.length
  return (
    <p className="wf-run-summary" aria-label="Run summary">
      {run.status === 'running' ? <b>Running…</b> : <b>Took {took(run)}</b>}
      {` · ${analyses === 1 ? '1 analysis' : `${analyses} analyses`}`}
      {sandboxes > 0 && ` · ${sandboxes === 1 ? '1 sandbox' : `${sandboxes} sandboxes`}`}
      {run.status !== 'done' && run.status !== 'running' && ` · ${STATUS_TEXT[run.status]}`}
    </p>
  )
}

export function AnalysisNodeView({ data, selected }: NodeProps<AnalysisFlowNode>) {
  return (
    <div className={['wf-node', selected && 'selected', data.priority > 0 && 'wf-first'].filter(Boolean).join(' ')}>
      <span className="wf-node-kind">
        Analysis
        {/* On the top line: the node keeps its height, so it never covers the one below. */}
        {data.priority > 0 && <span className="wf-node-first">runs first</span>}
      </span>
      <span className="wf-node-title">{data.missing ? `${data.analysisId} (deleted)` : data.title}</span>
      <span className="wf-node-meta">on {onTables(data.sources)}</span>
      <StatusPill status={data.status} />
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

export function BatchNodeView({ data, selected }: NodeProps<BatchFlowNode>) {
  return (
    <div className={selected ? 'wf-node wf-batch selected' : 'wf-node wf-batch'}>
      <Handle type="target" position={Position.Left} />
      <span className="wf-node-kind">Batch</span>
      <span className="wf-node-title">{data.title}</span>
      <span className="wf-node-meta">collects {data.count === 1 ? '1 analysis' : `${data.count} analyses`}</span>
      <StatusPill status={data.status} />
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

export function ReportNodeView({ data, selected }: NodeProps<ReportFlowNode>) {
  return (
    <div className={selected ? 'wf-node wf-report-node selected' : 'wf-node wf-report-node'}>
      <Handle type="target" position={Position.Left} />
      <span className="wf-node-kind">Report</span>
      <span className="wf-node-title">{data.title}</span>
      <span className="wf-node-meta">{data.count === 1 ? '1 batch' : `${data.count} batches`} on one page</span>
      <StatusPill status={data.status} />
      {data.url && (
        <a className="wf-node-open nodrag" href={data.url} target="_blank" rel="noopener noreferrer">
          Open report ↗
        </a>
      )}
    </div>
  )
}

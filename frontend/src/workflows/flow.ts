/**
 * A workflow on the canvas, as data: its nodes' shapes and statuses, and conversions to and from the
 * API's workflow. Shared by the Workflows page and the chat's preview; the node boxes are in nodes.tsx.
 */
import type { Edge, Node } from '@xyflow/react'

import { workflowReportUrl } from '../api/client'
import type { Workflow, WorkflowBody, WorkflowNodeRun, WorkflowRun, WorkflowSettings } from '../api/types'
import { AnalysisNodeView, BatchNodeView, ReportNodeView } from './nodes'

export type Status = WorkflowNodeRun['status'] | WorkflowRun['status']

export const STATUS_TEXT: Record<Status, string> = {
  waiting: 'waiting for a sandbox…',
  running: 'running…',
  done: 'done',
  failed: 'failed',
  needs_optimization: 'needs optimization',
  partly_done: 'partly done',
}

export function onTables(sources: Record<string, string>): string {
  const swaps = Object.entries(sources).filter(([from, to]) => from !== to)
  return swaps.length ? swaps.map(([from, to]) => `${from} → ${to}`).join(', ') : 'its saved tables'
}

/** How long a run took, e.g. "16 s"; empty while it runs. */
export function took(run: WorkflowRun): string {
  return run.seconds === null ? '' : `${Math.round(run.seconds)} s`
}

/** How many sandboxes a run started: its analyses share a few; older runs had one each. */
export function sandboxesStarted(run: WorkflowRun): number {
  return new Set(run.nodes.filter((n) => n.sandbox_at).map((n) => n.sandbox ?? n.node_id)).size
}

/** The status of a batch: from the analyses connected to it, in one run. */
export function batchStatus(nodes: WorkflowNodeRun[]): Status | undefined {
  if (nodes.length === 0) return undefined
  if (nodes.some((n) => n.status === 'running' || n.status === 'waiting')) return 'running'
  if (nodes.every((n) => n.status === 'done')) return 'done'
  return nodes.some((n) => n.status === 'done') ? 'partly_done' : 'failed'
}

export type AnalysisData = {
  analysisId: string
  sources: Record<string, string>
  /** Higher runs earlier ("run first"). */
  priority: number
  title: string
  missing: boolean
  status?: Status
}
export type BatchData = { title: string; count: number; status?: Status }
export type ReportData = { title: string; count: number; status?: Status; url?: string }
export type AnalysisFlowNode = Node<AnalysisData, 'analysis'>
export type BatchFlowNode = Node<BatchData, 'batch'>
export type ReportFlowNode = Node<ReportData, 'report'>
export type FlowNode = AnalysisFlowNode | BatchFlowNode | ReportFlowNode

/** The connections a workflow allows: source kind -> target kind. */
export const CONNECTS: Record<string, string> = { analysis: 'batch', batch: 'report' }

/** The nodes as the canvas shows them: each with how many it collects, and its status in `run`. */
export function withRun(nodes: FlowNode[], edges: Edge[], run: WorkflowRun | null, id: string | null): FlowNode[] {
  return nodes.map((n) => {
    if (n.type === 'analysis') {
      return { ...n, data: { ...n.data, status: run?.nodes.find((r) => r.node_id === n.id)?.status } }
    }
    const count = edges.filter((e) => e.target === n.id).length
    const inBatches = (batchIds: string[]) =>
      batchIds.flatMap((b) => run?.batches[b] ?? []).flatMap((nodeId) => run?.nodes.filter((r) => r.node_id === nodeId) ?? [])
    if (n.type === 'batch') return { ...n, data: { ...n.data, count, status: batchStatus(inBatches([n.id])) } }
    const inRun = run?.reports?.[n.id]
    const url = run && inRun && id ? workflowReportUrl(id, run.id, n.id) : undefined
    return { ...n, data: { ...n.data, count, status: batchStatus(inBatches(inRun ?? [])), url } }
  }) as FlowNode[]
}

export function toFlow(workflow: Workflow): { nodes: FlowNode[]; edges: Edge[] } {
  const nodes: FlowNode[] = workflow.nodes.map((n) =>
    n.kind === 'analysis'
      ? {
          id: n.id, type: 'analysis', position: n.position,
          data: {
            analysisId: n.analysis_id, sources: n.sources, priority: n.priority ?? 0,
            title: workflow.analyses[n.analysis_id]?.title ?? n.analysis_id,
            missing: workflow.analyses[n.analysis_id] === null,
          },
        }
      : { id: n.id, type: n.kind, position: n.position, data: { title: n.title, count: 0 } },
  ) as FlowNode[]
  const edges = workflow.edges.map((e) => ({ id: `${e.source}->${e.target}`, source: e.source, target: e.target }))
  return { nodes, edges }
}

export function toBody(title: string, nodes: FlowNode[], edges: Edge[], settings: WorkflowSettings): WorkflowBody {
  return {
    title,
    settings,
    nodes: nodes.map((n) => {
      const position = { x: Math.round(n.position.x), y: Math.round(n.position.y) }
      return n.type === 'analysis'
        ? { kind: 'analysis', id: n.id, analysis_id: n.data.analysisId, sources: n.data.sources, priority: n.data.priority, position }
        : { kind: n.type, id: n.id, title: n.data.title, position }
    }),
    edges: edges.map((e) => ({ source: e.source, target: e.target })),
  }
}

export const NODE_TYPES = { analysis: AnalysisNodeView, batch: BatchNodeView, report: ReportNodeView }

import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useCallback, useEffect, useMemo, useState, type DragEvent } from 'react'
import { Link, Route, Routes, useNavigate, useParams } from 'react-router'

import { SourcePicker } from '../analyses/AnalysesPage'
import {
  analysisOutputUrl,
  createWorkflow,
  deleteWorkflow,
  getAnalysisSources,
  getWorkflow,
  listAnalyses,
  listWorkflows,
  runWorkflow,
  saveWorkflow,
  workflowPerformanceUrl,
  workflowReportUrl,
} from '../api/client'
import type {
  AnalysisSummary,
  SourceOption,
  Workflow,
  WorkflowBody,
  WorkflowNodeRun,
  WorkflowRun,
  WorkflowSettings,
  WorkflowSummary,
} from '../api/types'
import { ArtifactView } from '../chat/ArtifactView'

const POLL_MS = 2000
const DRAG_TYPE = 'application/x-saved-analysis'

type Status = WorkflowNodeRun['status'] | WorkflowRun['status']

const STATUS_TEXT: Record<Status, string> = {
  waiting: 'waiting for a sandbox…',
  running: 'running…',
  done: 'done',
  failed: 'failed',
  needs_optimization: 'needs optimization',
  partly_done: 'partly done',
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function newId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 8)}`
}

function onTables(sources: Record<string, string>): string {
  const swaps = Object.entries(sources).filter(([from, to]) => from !== to)
  return swaps.length ? swaps.map(([from, to]) => `${from} → ${to}`).join(', ') : 'its saved tables'
}

/** How long a run took, e.g. "16 s"; empty while it runs. */
function took(run: WorkflowRun): string {
  return run.seconds === null ? '' : `${Math.round(run.seconds)} s`
}

/** How many sandboxes a run started: its analyses share a few; older runs had one each. */
function sandboxesStarted(run: WorkflowRun): number {
  return new Set(run.nodes.filter((n) => n.sandbox_at).map((n) => n.sandbox ?? n.node_id)).size
}

/** One line about a run: how long it took, how many analyses, how many sandboxes. */
function RunSummary({ run }: { run: WorkflowRun }) {
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

function StatusPill({ status }: { status: Status | undefined }) {
  if (!status) return null
  return <span className={`wf-status wf-status-${status}`}>{STATUS_TEXT[status]}</span>
}

/** The status of a batch: from the analyses connected to it, in one run. */
function batchStatus(nodes: WorkflowNodeRun[]): Status | undefined {
  if (nodes.length === 0) return undefined
  if (nodes.some((n) => n.status === 'running' || n.status === 'waiting')) return 'running'
  if (nodes.every((n) => n.status === 'done')) return 'done'
  return nodes.some((n) => n.status === 'done') ? 'partly_done' : 'failed'
}

// --- the canvas's nodes -----------------------------------------------------------------

type AnalysisData = {
  analysisId: string
  sources: Record<string, string>
  /** Higher runs earlier ("run first"). */
  priority: number
  title: string
  missing: boolean
  status?: Status
}
type BatchData = { title: string; count: number; status?: Status }
type ReportData = { title: string; count: number; status?: Status; url?: string }
type AnalysisFlowNode = Node<AnalysisData, 'analysis'>
type BatchFlowNode = Node<BatchData, 'batch'>
type ReportFlowNode = Node<ReportData, 'report'>
type FlowNode = AnalysisFlowNode | BatchFlowNode | ReportFlowNode

/** The connections a workflow allows: source kind -> target kind. */
const CONNECTS: Record<string, string> = { analysis: 'batch', batch: 'report' }

function AnalysisNodeView({ data, selected }: NodeProps<AnalysisFlowNode>) {
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

function BatchNodeView({ data, selected }: NodeProps<BatchFlowNode>) {
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

function ReportNodeView({ data, selected }: NodeProps<ReportFlowNode>) {
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

const NODE_TYPES = { analysis: AnalysisNodeView, batch: BatchNodeView, report: ReportNodeView }

function toFlow(workflow: Workflow): { nodes: FlowNode[]; edges: Edge[] } {
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

function toBody(title: string, nodes: FlowNode[], edges: Edge[], settings: WorkflowSettings): WorkflowBody {
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

// --- the list -------------------------------------------------------------------------------

function WorkflowList() {
  const [workflows, setWorkflows] = useState<WorkflowSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirming, setConfirming] = useState<string | null>(null)

  useEffect(() => {
    listWorkflows().then(setWorkflows).catch((e) => setError(errorText(e)))
  }, [])

  const remove = async (id: string) => {
    try {
      await deleteWorkflow(id)
      setWorkflows((list) => list?.filter((w) => w.id !== id) ?? null)
    } catch (e) {
      setError(errorText(e))
    } finally {
      setConfirming(null)
    }
  }

  if (workflows === null && !error) return <p className="muted">Loading workflows…</p>
  return (
    <div className="eval-page">
      <header className="wf-list-header">
        <div>
          <h1>Workflows</h1>
          <p className="muted">
            Run several saved analyses at once. Each analysis is a node; connect it to a batch node, which collects every
            result into one report.
          </p>
        </div>
        <Link to="/workflows/new" className="button-link">+ New workflow</Link>
      </header>
      {error && <p className="error">{error}</p>}
      {workflows?.length === 0 ? (
        <div className="panel">
          <h2>No workflows yet</h2>
          <p className="muted">Make one from the analyses you have saved.</p>
        </div>
      ) : (
        <ul className="analysis-list" aria-label="Workflows">
          {workflows?.map((w) => (
            <li key={w.id} className="analysis-row">
              <Link to={`/workflows/${w.id}`} className="analysis-card">
                <span className="analysis-title">{w.title}</span>
                <span className="analysis-meta">
                  {w.analyses === 1 ? '1 analysis' : `${w.analyses} analyses`} ·{' '}
                  {w.batches === 1 ? '1 batch' : `${w.batches} batches`} ·{' '}
                  {w.reports === 1 ? '1 report' : `${w.reports} reports`} ·{' '}
                  {w.last_run ? (
                    <>
                      <StatusPill status={w.last_run.status} />
                      {w.last_run.seconds !== null && ` in ${took(w.last_run)}`} {when(w.last_run.started_at)}
                    </>
                  ) : (
                    'never run'
                  )}
                </span>
              </Link>
              {confirming === w.id ? (
                <span className="row-confirm" role="group" aria-label={`Confirm deleting ${w.title}`}>
                  <button className="danger small" onClick={() => remove(w.id)}>Delete</button>
                  <button className="link-button" onClick={() => setConfirming(null)}>Cancel</button>
                </span>
              ) : (
                <button className="icon-button" onClick={() => setConfirming(w.id)} aria-label={`Delete ${w.title}…`} title="Delete">
                  ×
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// --- the editor --------------------------------------------------------------------------------

function NodeReport({ node }: { node: WorkflowNodeRun }) {
  const link = node.run_id ? `/analyses/${node.analysis_id}?run=${node.run_id}` : `/analyses/${node.analysis_id}`
  return (
    <article className="wf-report-item">
      <header>
        <h3>{node.title}</h3>
        <StatusPill status={node.status} />
      </header>
      <p className="muted small">
        on {onTables(node.sources)}
        {node.version !== null && ` · version ${node.version}`}
        {node.seconds !== null && ` · ${node.seconds.toFixed(0)}s`} · <Link to={link}>open the run</Link>
      </p>
      {node.error && <p className="error small">{node.error}</p>}
      {node.status === 'needs_optimization' && (
        <p className="small">
          It stopped before exporting. <Link to={link}>Optimize it on its page</Link>, then run the workflow again.
        </p>
      )}
      {node.run_id &&
        node.outputs.map((file) => (
          <ArtifactView key={file} artifact={{ name: file, kind: 'html', url: analysisOutputUrl(node.analysis_id, node.run_id!, file) }} />
        ))}
    </article>
  )
}

function BatchReport({ run, batchId }: { run: WorkflowRun; batchId: string }) {
  const nodes = (run.batches[batchId] ?? [])
    .map((id) => run.nodes.find((n) => n.node_id === id))
    .filter((n): n is WorkflowNodeRun => n !== undefined)
  if (nodes.length === 0) return <p className="muted">Nothing was connected to this batch in that run.</p>
  const done = nodes.filter((n) => n.status === 'done').length
  return (
    <>
      <p className="muted">
        {done} of {nodes.length} done
        {run.seconds !== null && ` · the whole run took ${run.seconds.toFixed(0)}s`}
      </p>
      <div className="wf-report">
        {nodes.map((n) => (
          <NodeReport key={n.node_id} node={n} />
        ))}
      </div>
    </>
  )
}

function Editor({ id }: { id: string | null }) {
  const navigate = useNavigate()
  const flow = useReactFlow()
  const [title, setTitle] = useState('Untitled workflow')
  const [settings, setSettings] = useState<WorkflowSettings>({ lanes: null })
  const [nodes, setNodes] = useState<FlowNode[]>(() =>
    id ? [] : [{ id: newId('b'), type: 'batch', position: { x: 520, y: 140 }, data: { title: 'Batch', count: 0 } }],
  )
  const [edges, setEdges] = useState<Edge[]>([])
  const [detail, setDetail] = useState<Workflow | null>(null)
  const [saved, setSaved] = useState<Record<string, AnalysisSummary>>({})
  const [loaded, setLoaded] = useState(id === null)
  const [dirty, setDirty] = useState(id === null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [shownRun, setShownRun] = useState<string | null>(null)
  // Below the canvas: a run's results, or its performance (time and resources). Never both on one page.
  const [view, setView] = useState<'results' | 'performance'>('results')
  const [options, setOptions] = useState<Record<string, SourceOption[]>>({})
  const [confirmDelete, setConfirmDelete] = useState(false)

  useEffect(() => {
    listAnalyses()
      .then((list) => setSaved(Object.fromEntries(list.map((a) => [a.id, a]))))
      .catch((e) => setError(errorText(e)))
  }, [])

  useEffect(() => {
    if (!id) return
    getWorkflow(id)
      .then((workflow) => {
        const shape = toFlow(workflow)
        setTitle(workflow.title)
        setSettings(workflow.settings ?? { lanes: null })
        setNodes(shape.nodes)
        setEdges(shape.edges)
        setDetail(workflow)
        setLoaded(true)
      })
      .catch((e) => setError(errorText(e)))
  }, [id])

  // While a run goes on, check again until it ends. Only the runs change: the canvas keeps any edits.
  const running = detail?.running ?? false
  useEffect(() => {
    if (!running || !id) return
    const timer = setTimeout(() => {
      getWorkflow(id).then(setDetail).catch((e) => setError(errorText(e)))
    }, POLL_MS)
    return () => clearTimeout(timer)
  }, [running, detail, id])

  const run = detail?.runs.find((r) => r.id === shownRun) ?? detail?.runs[0] ?? null
  const selectedNode = nodes.find((n) => n.id === selected) ?? null
  // Picking a batch or report node asks for its results.
  const selectedKind = selectedNode?.type
  useEffect(() => {
    if (selectedKind === 'batch' || selectedKind === 'report') setView('results')
  }, [selected, selectedKind])

  // What the canvas shows: each node with how many it collects, and its status in the run shown.
  const shown = useMemo<FlowNode[]>(
    () =>
      nodes.map((n) => {
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
      }),
    [nodes, edges, run, id],
  )

  const onNodesChange = useCallback((changes: NodeChange<FlowNode>[]) => {
    if (changes.some((c) => c.type === 'position' || c.type === 'remove' || c.type === 'add' || c.type === 'replace')) {
      setDirty(true)
    }
    for (const c of changes) {
      if (c.type === 'select') setSelected((current) => (c.selected ? c.id : current === c.id ? null : current))
    }
    setNodes((current) => applyNodeChanges(changes, current))
  }, [])

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    if (changes.some((c) => c.type === 'remove' || c.type === 'add')) setDirty(true)
    setEdges((current) => applyEdgeChanges(changes, current))
  }, [])

  const isValidConnection = useCallback(
    (c: Connection | Edge) => {
      const source = nodes.find((n) => n.id === c.source)
      const target = nodes.find((n) => n.id === c.target)
      return (
        source !== undefined && target !== undefined && CONNECTS[source.type] === target.type &&
        !edges.some((e) => e.source === c.source && e.target === c.target)
      )
    },
    [nodes, edges],
  )

  const onConnect = useCallback((c: Connection) => {
    setEdges((current) => addEdge({ ...c, id: `${c.source}->${c.target}` }, current))
    setDirty(true)
  }, [])

  const addAnalysis = (analysisId: string, position?: { x: number; y: number }) => {
    const analysis = saved[analysisId]
    if (!analysis) return
    const count = nodes.filter((n) => n.type === 'analysis').length
    const node: AnalysisFlowNode = {
      id: newId('a'), type: 'analysis', position: position ?? { x: 40, y: 30 + count * 120 },
      data: { analysisId, sources: {}, priority: 0, title: analysis.title, missing: false },
    }
    setNodes((current) => [...current, node])
    setDirty(true)
  }

  const addBatch = () => {
    const count = nodes.filter((n) => n.type === 'batch').length
    const node: BatchFlowNode = {
      id: newId('b'), type: 'batch', position: { x: 520, y: 140 + count * 140 }, data: { title: `Batch ${count + 1}`, count: 0 },
    }
    setNodes((current) => [...current, node])
    setDirty(true)
  }

  const addReport = () => {
    const count = nodes.filter((n) => n.type === 'report').length
    const node: ReportFlowNode = {
      id: newId('r'), type: 'report', position: { x: 900, y: 140 + count * 160 },
      data: { title: count ? `Report ${count + 1}` : 'Report', count: 0 },
    }
    setNodes((current) => [...current, node])
    setDirty(true)
  }

  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    const analysisId = event.dataTransfer.getData(DRAG_TYPE)
    if (analysisId) addAnalysis(analysisId, flow.screenToFlowPosition({ x: event.clientX, y: event.clientY }))
  }

  const updateNode = (nodeId: string, data: Partial<AnalysisData> | Partial<BatchData> | Partial<ReportData>) => {
    setNodes((current) => current.map((n) => (n.id === nodeId ? ({ ...n, data: { ...n.data, ...data } } as FlowNode) : n)))
    setDirty(true)
  }

  const removeNode = (nodeId: string) => {
    setNodes((current) => current.filter((n) => n.id !== nodeId))
    setEdges((current) => current.filter((e) => e.source !== nodeId && e.target !== nodeId))
    setSelected(null)
    setDirty(true)
  }

  // The tables a selected analysis could read instead, once per analysis.
  const selectedAnalysis = selectedNode?.type === 'analysis' ? selectedNode.data.analysisId : null
  useEffect(() => {
    if (!selectedAnalysis || options[selectedAnalysis]) return
    getAnalysisSources(selectedAnalysis)
      .then((found) => setOptions((o) => ({ ...o, [selectedAnalysis]: found })))
      .catch(() => setOptions((o) => ({ ...o, [selectedAnalysis]: [] })))
  }, [selectedAnalysis, options])

  /** Save, and the workflow as saved; a new one moves to its own address unless `stay`. */
  const save = async (stay = false): Promise<Workflow> => {
    const body = toBody(title, nodes, edges, settings)
    const result = id ? await saveWorkflow(id, body) : await createWorkflow(body)
    setDetail(result)
    setDirty(false)
    if (!id && !stay) navigate(`/workflows/${result.id}`, { replace: true })
    return result
  }

  const act = async (what: () => Promise<unknown>) => {
    setBusy(true)
    setError(null)
    try {
      await what()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  const start = () =>
    act(async () => {
      const workflow = dirty || !id ? await save(true) : detail!
      const started = await runWorkflow(workflow.id)
      if (!id) {
        navigate(`/workflows/${workflow.id}`, { replace: true })
        return
      }
      setShownRun(started.id)
      setDetail(await getWorkflow(workflow.id))
    })

  const remove = () =>
    act(async () => {
      if (id) await deleteWorkflow(id)
      navigate('/workflows')
    })

  if (!loaded) return error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>

  const batches = nodes.filter((n): n is BatchFlowNode => n.type === 'batch')
  // Below the canvas: the selected batch or report; otherwise the run's first report, or its first batch.
  const runReports = run?.reports ?? {}
  const below: { kind: 'batch' | 'report'; id: string } | null =
    selectedNode?.type === 'batch' || selectedNode?.type === 'report'
      ? { kind: selectedNode.type, id: selectedNode.id }
      : run && Object.keys(runReports).length
        ? { kind: 'report', id: Object.keys(runReports)[0] }
        : run && Object.keys(run.batches).length
          ? { kind: 'batch', id: Object.keys(run.batches)[0] }
          : null
  const belowTitle = below
    ? run?.titles[below.id] ?? (nodes.find((n) => n.id === below.id)?.data as { title?: string } | undefined)?.title ?? ''
    : ''
  const connected = edges.some((e) => nodes.find((n) => n.id === e.source)?.type === 'analysis')

  return (
    <div className="wf-page">
      <div className="wf-toolbar">
        <Link to="/workflows" className="muted">← Workflows</Link>
        <input className="wf-title" aria-label="Workflow title" value={title} onChange={(e) => { setTitle(e.target.value); setDirty(true) }} />
        {dirty && <span className="muted small">unsaved changes</span>}
        <span className="wf-toolbar-actions">
          <button className="secondary" onClick={() => act(() => save())} disabled={busy || running || !dirty}>Save</button>
          <button onClick={start} disabled={busy || running || !connected} title={connected ? undefined : 'Connect an analysis to a batch first'}>
            {running ? 'Running…' : dirty ? 'Save & run' : 'Run'}
          </button>
          {confirmDelete ? (
            <span className="row-confirm-inline" role="group" aria-label="Confirm deleting the workflow">
              <button className="danger small" onClick={remove} disabled={busy || running}>Delete</button>
              <button className="link-button" onClick={() => setConfirmDelete(false)}>Cancel</button>
            </span>
          ) : (
            <button className="secondary" onClick={() => setConfirmDelete(true)} disabled={running}>Delete…</button>
          )}
        </span>
      </div>
      {error && <p className="error">{error}</p>}

      <div className="wf-editor">
        <aside className="wf-palette" aria-label="Saved analyses">
          <h2>Analyses</h2>
          <p className="muted small">Drag one onto the canvas, or click +.</p>
          <ul>
            {Object.values(saved).map((a) => (
              <li key={a.id} draggable onDragStart={(e) => { e.dataTransfer.setData(DRAG_TYPE, a.id); e.dataTransfer.effectAllowed = 'move' }}>
                <span title={a.title}>{a.title}</span>
                <button className="icon-button" onClick={() => addAnalysis(a.id)} aria-label={`Add ${a.title}`} title="Add">+</button>
              </li>
            ))}
          </ul>
          {Object.keys(saved).length === 0 && <p className="muted small">No saved analyses yet.</p>}
          <button className="secondary small" onClick={addBatch}>+ Batch node</button>
          <button className="secondary small" onClick={addReport}>+ Report node</button>
        </aside>

        <div className="wf-canvas" onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move' }} onDrop={onDrop}>
          <ReactFlow
            nodes={shown}
            edges={edges}
            nodeTypes={NODE_TYPES}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            defaultEdgeOptions={{ animated: running }}
            colorMode="system"
            fitView
            fitViewOptions={{ maxZoom: 1 }}
          >
            <Background />
            <Controls />
          </ReactFlow>
        </div>

        <aside className="wf-inspector" aria-label="Node settings">
          {selectedNode?.type === 'analysis' ? (
            <>
              <h2>Analysis</h2>
              <p>
                <Link to={`/analyses/${selectedNode.data.analysisId}`}>{selectedNode.data.title}</Link>
                {detail?.analyses[selectedNode.data.analysisId] && (
                  <span className="muted"> · version {detail.analyses[selectedNode.data.analysisId]!.version}</span>
                )}
              </p>
              <p className="muted small">Which data this node runs on. Other tables must have the same columns.</p>
              {options[selectedNode.data.analysisId] === undefined ? (
                <p className="muted small">Loading tables…</p>
              ) : (
                <SourcePicker
                  options={options[selectedNode.data.analysisId]}
                  chosen={selectedNode.data.sources}
                  disabled={running}
                  onChange={(table, to) =>
                    updateNode(selectedNode.id, {
                      sources: Object.fromEntries(
                        Object.entries({ ...selectedNode.data.sources, [table]: to }).filter(([from, name]) => from !== name),
                      ),
                    })
                  }
                />
              )}
              <label className="field inline">
                <input
                  type="checkbox"
                  checked={selectedNode.data.priority > 0}
                  onChange={(e) => updateNode(selectedNode.id, { priority: e.target.checked ? 1 : 0 })}
                  disabled={running}
                />
                Run first
              </label>
              <p className="muted small">Analyses marked to run first start before the others; the rest run longest first.</p>
              <button className="secondary small" onClick={() => removeNode(selectedNode.id)} disabled={running}>
                Remove from workflow
              </button>
            </>
          ) : selectedNode?.type === 'batch' ? (
            <>
              <h2>Batch</h2>
              <label className="field">
                Name
                <input value={selectedNode.data.title} onChange={(e) => updateNode(selectedNode.id, { title: e.target.value })} />
              </label>
              <p className="muted small">
                Waits for the analyses connected to it, then collects their results. Connect it to a report node to
                put them on one page.
              </p>
              <button className="secondary small" onClick={() => removeNode(selectedNode.id)} disabled={running || batches.length < 2}>
                Remove from workflow
              </button>
            </>
          ) : selectedNode?.type === 'report' ? (
            <>
              <h2>Report</h2>
              <label className="field">
                Name
                <input value={selectedNode.data.title} onChange={(e) => updateNode(selectedNode.id, { title: e.target.value })} />
              </label>
              <p className="muted small">
                One page with every chart of the batches connected to it, a section each. It is shown below; open it on
                its own to share or save it.
              </p>
              {run && runReports[selectedNode.id] && id ? (
                <a href={workflowReportUrl(id, run.id, selectedNode.id)} target="_blank" rel="noopener noreferrer">
                  Open report ↗
                </a>
              ) : (
                <p className="muted small">Run the workflow to make it.</p>
              )}
              <button className="secondary small" onClick={() => removeNode(selectedNode.id)} disabled={running}>
                Remove from workflow
              </button>
            </>
          ) : (
            <>
              <h2>How it works</h2>
              <ol className="small">
                <li>Add saved analyses from the left.</li>
                <li>Connect each one's right dot to a batch node's left dot.</li>
                <li>Connect a batch to a report node: the report puts its results on one page.</li>
                <li>Click an analysis node to set the tables it runs on.</li>
                <li><b>Run</b> starts every connected analysis at once.</li>
              </ol>
              <p className="muted small">Select a node or an edge and press Backspace to remove it.</p>
              <label className="field">
                Sandboxes per run
                <select
                  value={settings.lanes ?? ''}
                  onChange={(e) => {
                    setSettings({ lanes: e.target.value ? Number(e.target.value) : null })
                    setDirty(true)
                  }}
                  disabled={running}
                >
                  <option value="">default (all but one, so chat keeps one)</option>
                  {[1, 2, 3].map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}
        </aside>
      </div>

      <section className={view === 'performance' || below?.kind === 'report' ? 'panel wf-panel-report' : 'panel'}>
        <div className="wf-report-header">
          <div className="wf-view-switch" role="tablist" aria-label="What to show">
            <button role="tab" aria-selected={view === 'results'} className={view === 'results' ? 'active' : ''} onClick={() => setView('results')}>
              Results
            </button>
            <button role="tab" aria-selected={view === 'performance'} className={view === 'performance' ? 'active' : ''} onClick={() => setView('performance')}>
              Performance
            </button>
          </div>
          <h2>
            {view === 'performance'
              ? 'Performance of this run'
              : below ? `${below.kind === 'report' ? 'Report' : 'Batch'}: ${belowTitle}` : 'Results'}
          </h2>
          {detail && detail.runs.length > 0 && (
            <label className="field inline">
              Run
              <select value={run?.id ?? ''} onChange={(e) => setShownRun(e.target.value)}>
                {detail.runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {when(r.started_at)} · {STATUS_TEXT[r.status]}{r.seconds !== null ? ` · ${took(r)}` : ''}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        {run && <RunSummary run={run} />}
        {!run ? (
          <p className="muted">Not run yet. Run the workflow to see its {view === 'performance' ? 'performance' : 'results'} here.</p>
        ) : view === 'performance' ? (
          id && <ArtifactView artifact={{ name: 'Performance.html', kind: 'html', url: workflowPerformanceUrl(id, run.id) }} />
        ) : below?.kind === 'report' ? (
          runReports[below.id] && id ? (
            <ArtifactView artifact={{ name: `${belowTitle}.html`, kind: 'html', url: workflowReportUrl(id, run.id, below.id) }} />
          ) : (
            <p className="muted">This report was added after that run. Run the workflow to make it.</p>
          )
        ) : below ? (
          <BatchReport run={run} batchId={below.id} />
        ) : (
          <p className="muted">That run had no batch.</p>
        )}
        {view === 'results' && run && Object.keys(run.batches).length + Object.keys(runReports).length > 1 && (
          <p className="muted small">Click a batch or report node to see it here.</p>
        )}
      </section>
    </div>
  )
}

function WorkflowEditor({ isNew = false }: { isNew?: boolean }) {
  const { id = '' } = useParams()
  return (
    <ReactFlowProvider key={isNew ? 'new' : id}>
      <Editor id={isNew ? null : id} />
    </ReactFlowProvider>
  )
}

export function WorkflowsPage() {
  return (
    <Routes>
      <Route index element={<WorkflowList />} />
      <Route path="new" element={<WorkflowEditor isNew />} />
      <Route path=":id" element={<WorkflowEditor />} />
    </Routes>
  )
}

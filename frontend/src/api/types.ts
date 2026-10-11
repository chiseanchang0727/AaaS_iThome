// Shapes of what the backend sends. Kept in step with backend/api/.

/** One Server-Sent Event from POST /api/chat (backend/api/events.py). */
export type ChatEvent =
  | { type: 'thread'; thread_id: string }
  | { type: 'thinking'; text: string }
  | { type: 'tool_call'; id: string; name: string; args: Record<string, unknown> }
  | { type: 'tool_result'; id: string; name: string | null; content: string; error: boolean }
  | { type: 'answer'; text: string }
  | { type: 'artifact'; name: string; url: string; kind: string }
  | { type: 'error'; message: string }
  | { type: 'notice'; message: string }
  | { type: 'done' }

/** Totals for a set of code steps (backend/api/load.py summarize). */
export interface LoadSummary {
  steps: number
  conversations: number
  run_seconds: number
  cpu_seconds: number
  overhead_seconds: number
  peak_memory_mb: number | null
  average_peak_memory_mb: number | null
  out_of_memory: number
  failed: number
}

/** One code step the agent ran in a sandbox. Times and memory are null when unmeasured. */
export interface LoadStep {
  conversation: string
  account: string
  turn: number
  prompt: string
  ts: string
  command: string
  exit_code: number | null
  seconds: number | null
  run_seconds: number | null
  cpu_seconds: number | null
  peak_memory_mb: number | null
  memory_limit_mb: number | null
  out_of_memory: boolean
  /** What the step was: worked out from the log (backend/api/load.py step_roles). */
  strategy?: StepStrategy
  rewrite?: number
}

/** first try / a changed attempt after a kill / unchanged after a kill / after moving to a bigger sandbox */
export type StepStrategy = 'first' | 'rewrite' | 'same code' | 'bigger sandbox'

/** A conversation that ran code: what it cost and what happened to its sandbox. */
export interface LoadConversation {
  conversation: string
  account: string
  question: string
  turns: number
  steps: number
  run_seconds: number
  cpu_seconds: number
  out_of_memory: number
  upgrades: number
  upgrade_failures: number
  rewrites: number
  /** How an out-of-memory kill ended; null when there was none. */
  resolved_by: 'rewrite' | 'bigger sandbox' | 'not resolved' | null
  peak_memory_mb: number | null
  sandbox_gb: number | null
  last: string
}

/** One moment of a conversation (backend/api/load.py timeline). */
export type TimelineItem = { ts: string; turn: number } & (
  | { kind: 'question'; text: string }
  | { kind: 'action'; tool: string; detail: string }
  | { kind: 'answer'; text: string }
  | {
      kind: 'step'
      command: string
      exit_code: number | null
      seconds: number | null
      run_seconds: number | null
      cpu_seconds: number | null
      peak_memory_mb: number | null
      memory_limit_mb: number | null
      out_of_memory: boolean
      strategy?: StepStrategy
      rewrite?: number
    }
  | { kind: 'event'; event: string; [field: string]: unknown }
)

/** GET /api/load/conversations/{id}. */
export interface ConversationTimeline {
  conversation: string
  account: string
  timeline: TimelineItem[]
}

/** GET /api/load. */
export interface Load {
  account: string | null
  accounts: string[]
  memory_limit_mb: number | null
  summary: LoadSummary
  per_account: Record<string, LoadSummary>
  steps: LoadStep[]
  conversations: LoadConversation[]
}

/** GET /api/sandboxes (ConversationManager.status). */
export interface SandboxStatus {
  enabled: boolean
  provider: string | null
  max: number | null
  in_use: number
  busy: number
  idle: number
  starting: number
  warm: number
  warming: number
  waiting: number
  idle_minutes: number
  max_per_account: number | null
  /** Per account: slots held, and its conversations' sandboxes by state. */
  accounts: Record<string, { sandboxes: number; busy: number; idle: number; starting: number }>
  /** The account this request was made as. */
  account: string
  memory_used_gb: number
  max_memory_gb: number | null
  /** Sandboxes moved to a bigger size after running out of memory. */
  bigger: number
}

export interface Column {
  name: string
  type: string
}

/** POST /api/uploads: a parsed file waiting to be kept. */
export interface StagedUpload {
  upload_id: string
  filename: string
  rows: number
  columns: Column[]
  preview: Record<string, unknown>[]
  suggested_name: string
}

export type DatasetKind = 'table' | 'file'

/** GET /api/datasets entries. The built-in videos table has no columns listed. */
export interface Dataset {
  name: string
  kind: DatasetKind
  built_in: boolean
  description?: string
  rows?: number
  columns?: Column[]
  source?: string
  created_at?: string
}

// --- evals (backend/api/evals.py, written by backend/evals/memory/run.py) ----

/** Totals for one arm of an eval run. */
export interface ArmSummary {
  turns: number
  checked: number
  correct: number
  grounded: number
  steps: number
  queries: number
  skill_reads: number
  tokens_per_turn: number
  first_call_per_turn: number
}

/** GET /api/evals/context/runs entries. */
export interface EvalRunInfo {
  id: string
  created_at: string
  model: string
  arms: string[]
  repeats: number
  conversations: string[]
  summary: Record<string, ArmSummary>
}

/** One turn of one arm: what was asked, what came back, what it cost. */
export interface EvalTurnResult {
  conversation: string
  turn: string
  arm: string
  run: number
  prompt: string
  answer: string
  stopped: boolean
  /** Turn numbers the context filter sent; null for arms without one. */
  sent_turns: number[] | null
  steps: number
  queries: number
  skill_reads: number
  input_tokens: number
  first_call_tokens: number
  /** null: nothing to check automatically (small talk). */
  correct: boolean | null
  missing: string[]
  ungrounded: string[]
}

export interface EvalConversation {
  id: string
  about: string
  turns: { id: string; prompt: string; expect: string[]; note: string }[]
}

/** GET /api/evals/context/runs/{id}. */
export interface EvalRun {
  id: string
  created_at: string
  model: string
  sandbox: boolean
  repeats: number
  arms: string[]
  context_filter: Record<string, unknown> | null
  conversations: EvalConversation[]
  results: EvalTurnResult[]
  summary: Record<string, ArmSummary>
}

/** One line of a conversation's history (backend/api/history.py). */
export interface HistoryLine {
  id?: string
  previous?: string | null
  turn: number
  role: 'user' | 'assistant' | 'tool' | 'sandbox_step' | 'sandbox_event' | 'context'
  content: string
  tool_calls?: { id: string; name: string; args: Record<string, unknown> }[]
  tool_call_id?: string
  name?: string | null
  error?: boolean
  ts?: string
}

// --- the whole-system eval (backend/evals/system, api/evals.py) ---------------

/** pass / fail / unknown, or judge_error when the judge call itself failed. */
export type EvalStatus = 'pass' | 'fail' | 'unknown' | 'judge_error'

export interface Judged {
  status: EvalStatus
  reason: string
}

export interface CodeCheck {
  check: string
  /** null: no code check exists for it. */
  pass: boolean | null
  evidence?: string[]
}

export interface CodeStep {
  command: string
  exit_code: number | null
  seconds: number | null
  peak_memory_mb: number | null
  memory_limit_mb: number | null
  out_of_memory: boolean
  strategy: string | null
}

export interface SystemResult {
  case_id: string
  question: string
  final_answer: string
  stopped: boolean
  stream_errors: string[]
  artifacts: string[]
  correct: Judged & { expected_values: { status: EvalStatus | null; found: string[]; missing: string[] } }
  grounded: Judged & {
    answer_values: string[]
    observed_count: number
    ungrounded_values: string[]
    unsupported_claims: { claim: string; reason: string }[]
  }
  instruction_following: Judged & { checks: CodeCheck[] }
  execution_strategy: Judged
  tool_usage: {
    tools: Record<string, number>
    tool_calls: number
    queries: number
    exports: number
    skill_reads: number
    skills_read: string[]
    code_executions: number
  }
  recovery: {
    errors: { tool: string; args: string; result: string; next: 'changed' | 'identical' | 'none' }[]
    retries: number
    identical_retries: number
    oom_events: number
    oom_resolved_by: string | null
    sandbox_moves: number
    sandbox_events: string[]
    code_steps: CodeStep[]
  }
  efficiency: {
    steps: number
    queries: number
    skill_reads: number
    code_executions: number
    model_calls: number
    input_tokens: number
    output_tokens: number
    total_tokens: number
    runtime_seconds: number
  }
  /** Set when Jev chose the earlier turns sent (server.context_filter); null with the whole conversation. */
  context?: { sent_turns: number[]; earlier_turns: number; seconds: number; error?: string } | null
}

export interface SystemCase {
  id: string
  question: string
  setup: string[]
  expected_values: Record<string, number | string>
  required_tools: string[]
  forbidden_tools: string[]
  required_skills: string[]
  required_outputs: string[]
  expected_behavior: string
  sandbox: boolean
  note: string
}

export interface SystemSummary {
  cases: number
  correct: number
  grounded: number
  instruction_following: number
  execution_strategy: number
  judge_errors: number
  total_tokens: number
  runtime_seconds: number
}

/** One item of GET /api/evals/system/runs. */
export interface SystemRunInfo {
  id: string
  created_at: string
  model: string
  judge_model: string | null
  rejudged_from: string | null
  cases: string[]
  /** "cases": the eval cases, run through the app; "history": saved conversations, judged as they happened. */
  source: 'cases' | 'history'
  status: RunStatus
  /** Results the run will have when done. */
  total: number
  summary: SystemSummary
}

export type RunStatus = 'running' | 'done' | 'failed'

/** GET /api/evals/system/history-runs. */
export interface HistoryRunPreview {
  available: boolean
  /** Where the conversations to evaluate are read from (server.eval_history_dir). */
  folder: string | null
  conversations: number
  turns: number
  /** Id of the history run in progress, if any. */
  running: string | null
}

/** GET /api/evals/system/runs/{id}. */
export interface SystemRun {
  id: string
  created_at: string
  model: string
  judge_model: string | null
  sandbox: boolean
  rejudged_from: string | null
  source?: 'cases' | 'history'
  status?: RunStatus
  total?: number
  error?: string
  cases: SystemCase[]
  results: SystemResult[]
  summary: SystemSummary
}

// --- Jev vs full (backend/evals/system/compare.py, api/compare.py) ------------

export interface VerdictTally {
  pass: number
  fail: number
  /** unknown or judge error */
  unknown: number
}

export interface SideSummary {
  turns: number
  correct: VerdictTally
  grounded: VerdictTally
  instruction_following: VerdictTally
  execution_strategy: VerdictTally
  steps: number
  total_tokens: number
  runtime_seconds: number
  errors: number
}

export interface CompareSide {
  thread: string
  /** One per turn, in order. */
  results: SystemResult[]
  summary: SideSummary
}

/** The same questions, asked once with the full conversation and once with Jev. */
export interface ComparePair {
  id: string
  questions: string[]
  full: CompareSide
  jev: CompareSide
}

export interface UnpairedConversation {
  thread: string
  questions: string[]
}

/** GET /api/evals/compare/runs/{id}. */
export interface CompareRun {
  id: string
  created_at: string
  judge_model: string | null
  status: RunStatus
  total: number
  error?: string
  unpaired: { full: UnpairedConversation[]; jev: UnpairedConversation[] }
  pairs: ComparePair[]
}

/** One item of GET /api/evals/compare/runs. */
export interface CompareRunInfo {
  id: string
  created_at: string
  judge_model: string | null
  status: RunStatus
  total: number
  pairs: number
}

/** GET /api/evals/compare/pairs. */
export interface ComparePreview {
  available: boolean
  pairs: number
  turns: number
  only_full: number
  only_jev: number
  running: string | null
}

// --- saved analyses (backend/analyses, api/analyses.py) ------------------------

/** One try at optimizing a run: the agent's conversation and how it ended. */
export interface OptimizationAttempt {
  status: 'running' | 'done' | 'failed'
  conversation: string
  new_version?: number
  message?: string
}

/** What one input cost, or would cost (backend analyses/models.py InputMeasure). */
export interface InputMeasure {
  file: string
  tables: string[]
  estimated_rows: number | null
  estimated_bytes: number | null
  /** The real row count, counted in the database without moving the rows, made when the estimate was over a limit. */
  counted_rows?: number | null
  rows: number | null
  bytes: number | null
  query_seconds: number | null
}

export interface SandboxMeasure {
  seconds: number | null
  cpu_seconds: number | null
  peak_memory_mb: number | null
  memory_limit_mb: number | null
  killed: boolean
}

/** Hard limits stop a run and ask for an optimization; soft ones (targets) recommend one. */
export interface RunLimits {
  hard_export_rows: number
  max_export_bytes: number
  soft_export_rows: number
  soft_export_bytes: number
  soft_run_seconds: number
  memory_warning_ratio: number
  hard_query_seconds: number | null
}

export type BottleneckKind = 'data_movement' | 'sandbox_memory' | 'sandbox_runtime'

/** A limit a run exceeded: hard ones stop it, soft ones recommend an optimization. */
export interface Finding {
  limit: 'hard' | 'soft'
  kind: BottleneckKind
  reason: string
}

export interface AnalysisRun {
  id: string
  started_at: string
  /** needs_optimization: a hard limit stopped it before it could finish (see findings). */
  status: 'running' | 'done' | 'failed' | 'needs_optimization'
  /** "save": the test run when it was saved; "run": someone clicked Run. */
  trigger: 'save' | 'run'
  seconds: number | null
  outputs: string[]
  error: string | null
  log: string
  notes: string[]
  /** Tables this run read instead of the recipe's: {"videos": "videos_ca"}. Empty: the recipe's own. */
  sources: Record<string, string>
  /** The recipe version it ran. */
  version: number
  measurements: { inputs: InputMeasure[]; sandbox: SandboxMeasure | null } | null
  findings: Finding[]
  /** For a run that needed one: the optimization it led to. */
  optimization: (OptimizationAttempt & {
    /** Earlier attempts for this run, oldest first: each failed one that was tried again. */
    earlier?: OptimizationAttempt[]
  }) | null
  /** Set on the run made right after an optimization: the version it replaced. */
  optimized_from: number | null
}

/** GET /api/analyses/{id}/sources: for one table the recipe reads, the tables that could stand in. */
export interface SourceOption {
  table: string
  candidates: { name: string; ok: boolean; problems: string[] }[]
}

export type AnalysisInput = { kind: 'query'; sql: string; file: string } | { kind: 'dataset'; name: string }

/** One item of GET /api/analyses. */
export interface AnalysisSummary {
  id: string
  title: string
  description: string
  question: string
  created_at: string
  version: number
  updated_at: string | null
  outputs: string[]
  last_run: AnalysisRun | null
  last_good_run: AnalysisRun | null
}

/** GET /api/analyses/{id}: the recipe and its runs, newest first. */
export interface Analysis {
  id: string
  title: string
  description: string
  question: string
  conversation: string | null
  created_at: string
  /** 1 when first saved; each change the agent saves over it adds one. */
  version: number
  updated_at: string | null
  inputs: AnalysisInput[]
  script: string
  outputs: { file: string; format: 'html' }[]
  /** The tables its queries read. */
  sources: string[]
  /** Why this version exists, when an optimization saved it. */
  optimization: {
    from_version: number
    kind: BottleneckKind
    reason: string
    sources: Record<string, string>
    run: string | null
    conversation: string | null
    /** How its results were checked against the version before, when it was saved. */
    results_check?: string | null
  } | null
  runs: AnalysisRun[]
  /** The limits runs are checked against (backend analyses/budget.py); null if the server does not check. */
  limits: RunLimits | null
  /** A run or an optimization is in progress. */
  running: boolean
  optimizing: boolean
}

// --- past conversations (backend api/app.py) -----------------------------------

/** One item of GET /api/conversations. */
export interface PastConversationSummary {
  id: string
  /** Its first message. */
  title: string
  turns: number
  started_at: string | null
  last_at: string | null
  files: number
}

/** GET /api/conversations/{id}. */
export interface PastConversation {
  id: string
  lines: HistoryLine[]
  files: { name: string; url: string; kind: string }[]
}

/** One version of a saved analysis's recipe (GET /api/analyses/{id}/versions). */
export type AnalysisVersion = Omit<Analysis, 'runs' | 'running' | 'optimizing' | 'limits'>

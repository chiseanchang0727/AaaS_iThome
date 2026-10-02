import type { LoadStep } from '../api/types'

export interface Group {
  conversation: string
  question: string
  steps: LoadStep[]
}

/** Conversation id -> its number (1 = the first to run code), as the chart labels them. */
export function conversationNumbers(steps: LoadStep[]): Map<string, number> {
  return new Map(groups(steps).map((g, i) => [g.conversation, i + 1]))
}

/** Steps grouped by conversation, conversations in the order they first ran code. */
export function groups(steps: LoadStep[]): Group[] {
  const byConversation = new Map<string, Group>()
  for (const s of [...steps].reverse()) {
    if (s.peak_memory_mb === null) continue
    const g = byConversation.get(s.conversation) ?? { conversation: s.conversation, question: s.prompt, steps: [] }
    g.steps.push(s)
    byConversation.set(s.conversation, g)
  }
  return [...byConversation.values()]
}

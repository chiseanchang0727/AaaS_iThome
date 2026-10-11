import type { PastConversationSummary } from '../api/types'

type Group = { label: string; items: PastConversationSummary[] }

function dayStart(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime()
}

/** Conversations by when they were last active: Today, Yesterday, Previous 7 days, Older. */
export function groupByDay(items: PastConversationSummary[], now: Date = new Date()): Group[] {
  const today = dayStart(now)
  const day = 24 * 60 * 60 * 1000
  const groups: Group[] = [
    { label: 'Today', items: [] },
    { label: 'Yesterday', items: [] },
    { label: 'Previous 7 days', items: [] },
    { label: 'Older', items: [] },
  ]
  for (const item of items) {
    const at = item.last_at ? dayStart(new Date(item.last_at)) : 0
    const index = at >= today ? 0 : at >= today - day ? 1 : at >= today - 7 * day ? 2 : 3
    groups[index].items.push(item)
  }
  return groups.filter((g) => g.items.length > 0)
}

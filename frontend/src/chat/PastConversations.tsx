import { useEffect, useState } from 'react'
import Markdown from 'react-markdown'
import { Link, useParams } from 'react-router'
import remarkGfm from 'remark-gfm'

import { getPastConversation } from '../api/client'
import type { PastConversation } from '../api/types'
import { turnSteps } from '../evals/results'
import { ArtifactView } from './ArtifactView'
import { SavedAnalysisLinks } from './SavedAnalysisLinks'
import { Steps } from './Steps'

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

/** One saved conversation, turn by turn, as the chat showed it. Mounted per conversation. */
export function PastConversationView() {
  const { id = '' } = useParams()
  return <Conversation key={id} id={id} />
}

function Conversation({ id }: { id: string }) {
  const [conversation, setConversation] = useState<PastConversation | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let current = true
    getPastConversation(id)
      .then((loaded) => current && setConversation(loaded))
      .catch((e) => current && setError(errorText(e)))
    return () => {
      current = false
    }
  }, [id])

  if (error) return <p className="error">{error}</p>
  if (!conversation) return <p className="muted">Loading…</p>

  const questions = conversation.lines.filter((l) => l.role === 'user')
  return (
    <section className="chat past">
      <p className="notice">
        Read-only: a saved conversation. <Link to="/chat">Back to the current chat</Link>
      </p>
      <div className="messages">
        {questions.map((q) => {
          const { steps, answer } = turnSteps(conversation.lines, q.turn)
          return (
            <div key={q.turn} className="past-turn">
              <div className="message message-user">{q.content}</div>
              <div className="message message-assistant">
                <Steps steps={steps} streaming={false} />
                {answer && (
                  <div className="answer">
                    <Markdown remarkPlugins={[remarkGfm]}>{answer}</Markdown>
                  </div>
                )}
                <SavedAnalysisLinks steps={steps} />
              </div>
            </div>
          )
        })}
        {conversation.files.map((f) => (
          <ArtifactView key={f.url} artifact={f} />
        ))}
      </div>
    </section>
  )
}

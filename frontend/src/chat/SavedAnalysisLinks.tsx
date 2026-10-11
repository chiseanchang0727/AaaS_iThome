import { Link } from 'react-router'

import { savedAnalyses } from './savedAnalyses'
import type { Step } from './state'

/** "✓ Saved …, open it in Analyses" for each analysis a turn saved or changed. */
export function SavedAnalysisLinks({ steps }: { steps: Step[] }) {
  return (
    <>
      {savedAnalyses(steps).map((a) => (
        <p key={a.id} className="saved-analysis" role="status">
          ✓ {a.version ? `Updated “${a.title}” to version ${a.version}` : `Saved “${a.title}”`}.{' '}
          <Link to={`/analyses/${a.id}`}>Open it in Analyses</Link> to run it again later.
        </p>
      ))}
    </>
  )
}

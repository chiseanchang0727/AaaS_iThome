import type { Artifact } from './state'

const IMAGE_KINDS = new Set(['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp'])

/**
 * A file the agent made. HTML (Plotly charts) renders in an iframe that may
 * run its scripts but gets an opaque origin: no `allow-same-origin`, so it
 * cannot read the app's storage or call the API. The server adds a matching
 * `Content-Security-Policy: sandbox` for when the file is opened directly.
 */
export function ArtifactView({ artifact }: { artifact: Artifact }) {
  const open = (
    <a className="artifact-open" href={artifact.url} target="_blank" rel="noopener noreferrer">
      Open ↗
    </a>
  )

  if (artifact.kind === 'html') {
    return (
      <figure className="artifact">
        <figcaption>
          <span>{artifact.name}</span>
          {open}
        </figcaption>
        <iframe
          className="artifact-frame"
          src={artifact.url}
          title={artifact.name}
          sandbox="allow-scripts"
          referrerPolicy="no-referrer"
          loading="lazy"
        />
      </figure>
    )
  }

  if (IMAGE_KINDS.has(artifact.kind)) {
    return (
      <figure className="artifact">
        <figcaption>
          <span>{artifact.name}</span>
          {open}
        </figcaption>
        <img src={artifact.url} alt={artifact.name} />
      </figure>
    )
  }

  return (
    <p className="artifact-file">
      📄 <a href={artifact.url} target="_blank" rel="noopener noreferrer">{artifact.name}</a>
    </p>
  )
}

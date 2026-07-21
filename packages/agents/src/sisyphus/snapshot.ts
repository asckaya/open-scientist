import { mkdir, writeFile } from 'node:fs/promises'
import { getRoundsDir } from '@open-scientist/config'

/** Shape of one tournament round snapshot (persisted for the API / UI layer). */
export interface RoundSnapshot {
  round: number
  runId: string
  projectId: string
  bestF1: number
  leadingHypoId: string | null
  survivingCount: number
  hypotheses: Array<{
    id: string
    statement: string
    f1: number | null
    status: string
    parentId: string | null
    round: number
  }>
  convergenceHistory: Array<{ round: number; bestF1: number; count: number }>
  /** ISO 8601 timestamp when the snapshot was written. */
  capturedAt: string
}

/**
 * Persist a per-round snapshot to
 * `data/projects/<projectId>/rounds/<round>/snapshot.json`.
 *
 * Plain async fn — fs failures bubble to the caller, which can wrap retry if
 * desired. Imports `node:fs/promises` + `@open-scientist/config` statically
 * (no VM sandbox anymore).
 */
export async function snapshotStep(snapshot: RoundSnapshot): Promise<{ path: string }> {
  const dir = getRoundsDir(snapshot.projectId, snapshot.round)
  await mkdir(dir, { recursive: true })
  const path = `${dir}/snapshot.json`
  await writeFile(path, JSON.stringify(snapshot, null, 2), 'utf-8')
  return { path }
}

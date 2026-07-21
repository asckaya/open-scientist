import { type TournamentWorkflowInput, tournamentWorkflow } from '@open-scientist/agents'
import type { TournamentResult } from '@open-scientist/schema'
import type { UIMessageChunk } from 'ai'

/**
 * In-memory registry of active tournament runs.
 *
 * Replaces `workflow/api` (`start` + `getRun`) with a hand-rolled chunk ring
 * buffer. Each {@link Run} collects the `UIMessageChunk`s emitted by
 * `tournamentWorkflow` (via the `emitChunk` sink) so clients can:
 *   1. consume the live SSE stream from POST `/runs`, and
 *   2. reconnect from a `startIndex` via GET `/runs/:id/stream`.
 *
 * There is no durable persistence — if the process restarts, in-flight runs
 * are lost. SQLite `runs` rows remain the source of truth for final status;
 * tournament rounds persist per-round `snapshot.json` files on disk which can
 * seed a future "resume from snapshot" path.
 */

/** A single run's state and stream surface. */
export interface Run {
  /** Transport-level run id (returned to client via `x-workflow-run-id`). */
  runId: string
  /** Business-level run id threaded through runtimeContext / snapshots. */
  workflowRunId: string
  /** Buffered `UIMessageChunk`s emitted so far, in order. */
  chunks: UIMessageChunk[]
  /** Resolves to the tournament result once `tournamentWorkflow` completes. */
  result: Promise<TournamentResult>
  /** Aborts the underlying tournament (signals tool exec / LLM calls). */
  cancel: () => Promise<void>
  /**
   * Returns a `ReadableStream` of chunks starting at `startIndex`.
   *
   * Supports tail-relative indexes (`startIndex < 0`): `-3` reads the last 3
   * chunks. The returned stream also exposes `getTailIndex()` so callers can
   * reconcile the client cursor after a tail-relative reconnect (matches the
   * shape previously produced by `workflow/api`'s `Run.getReadable()`).
   */
  getReadable: (opts?: { startIndex?: number }) => RunReadable
}

/** The object returned by `Run.getReadable()` — a stream + tail-index probe. */
export interface RunReadable {
  /** Pipe the chunk stream through a transform (e.g. identity for SSE). */
  pipeThrough: <T>(transform: TransformStream<UIMessageChunk, T>) => ReadableStream<T>
  /** Absolute index of the last buffered chunk (chunks.length - 1). */
  getTailIndex: () => Promise<number>
}

const registry = new Map<string, Run>()

/**
 * Start a tournament run in the background.
 *
 * Returns the {@link Run} once registered (without awaiting completion). The
 * run's chunks populate asynchronously as sub-agents emit stream events.
 */
export function start(input: TournamentWorkflowInput): Run {
  const runId = input.runId
  if (registry.has(runId)) {
    throw new Error(`Run "${runId}" is already registered`)
  }

  const abortController = new AbortController()
  const chunks: UIMessageChunk[] = []

  // Kick off the tournament in the background. Each emitted UIMessageChunk is
  // pushed into the run's buffer. The promise rejects on tournament error or
  // abort — callers (.result) can await it to observe completion.
  const result = tournamentWorkflow({
    ...input,
    emitChunk: (chunk) => {
      chunks.push(chunk)
    },
  }).then(
    (output) => {
      return output
    },
    (err) => {
      // Re-throw so `.result` awaiters see the failure; chunks already
      // buffered remain available for SSE replay / debugging.
      if (abortController.signal.aborted) return undefined as unknown as TournamentResult
      throw err
    },
  )

  const run: Run = {
    runId,
    workflowRunId: input.runId,
    chunks,
    result,
    cancel: async () => {
      abortController.abort()
    },
    getReadable: (opts) => makeReadable(chunks, opts?.startIndex),
  }

  registry.set(runId, run)
  // Auto-evict once the tournament settles (success or failure) so the
  // registry doesn't grow unboundedly across runs. We keep the run around
  // while in-flight so /stream reconnects work; after completion clients
  // should read final status from SQLite via GET /runs/:id.
  void result.finally(() => {
    // Defer eviction slightly so a reconnect landing right at completion
    // still finds the run.
    setTimeout(() => {
      registry.delete(runId)
    }, 60_000)
  })

  return run
}

/** Look up an active run by id. Returns `undefined` if unknown / evicted. */
export function getRun(runId: string): Run | undefined {
  return registry.get(runId)
}

/**
 * Build a `RunReadable` over the chunk buffer starting at `startIndex`.
 *
 * `startIndex < 0` is tail-relative (e.g. `-3` → last 3 chunks). Out-of-range
 * positive indexes clamp to the buffer end; the stream simply yields nothing
 * past the current tail and stays open for live chunks until the run settles.
 */
function makeReadable(chunks: UIMessageChunk[], startIndex?: number): RunReadable {
  const resolveStart = () => {
    if (startIndex === undefined) return 0
    if (startIndex < 0) return Math.max(0, chunks.length + startIndex)
    return startIndex
  }

  const stream = new ReadableStream<UIMessageChunk>({
    start(controller) {
      let cursor = resolveStart()
      // Replay any already-buffered chunks from the cursor onward.
      while (cursor < chunks.length) {
        controller.enqueue(chunks[cursor] as UIMessageChunk)
        cursor += 1
      }
    },
  })

  return {
    pipeThrough: <T>(transform: TransformStream<UIMessageChunk, T>) =>
      stream.pipeThrough(transform),
    getTailIndex: async () => Math.max(0, chunks.length - 1),
  }
}

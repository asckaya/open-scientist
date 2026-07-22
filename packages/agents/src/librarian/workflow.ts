import type { AgentRuntimeConfig, ModelArg } from '@open-scientist/config'
import { ensureIndexes } from '@open-scientist/helix'
import { createLogger } from '@open-scientist/logger'
import type { HypothesisPool } from '@open-scientist/schema'
import { persistAgentRun } from '../shared/persist.ts'
import { type EmitChunk, streamAgentOutput } from '../shared/stream.ts'
import { extractSubmitResult } from '../shared/tool-output.ts'
import { createLibrarianAgent } from './agent.ts'

const logger = createLogger('agents')

export type { TournamentInput, TournamentResult } from '@open-scientist/schema'

export interface LibrarianWorkflowInput {
  /** Seed hypothesis text from the user / Sisyphus. */
  seed: string
  /** Project name — drives workspace dir + HelixDB scoping. */
  projectId: string
  /** Run identifier — passed through runtimeContext for persistence + lineage. */
  runId: string
  /**
   * Serializable model descriptor — reconstructed into a `LanguageModel` inside
   * `createLibrarianAgent` via `createModelFromConfig`.
   */
  modelConfig: ModelArg
  /**
   * Per-agent runtime config override (instructions / skillDirectories /
   * mcpServers). When present, its `modelConfig` takes priority over the
   * `modelConfig` field above and its non-model fields override the factory
   * defaults. Undefined → fully default behaviour (backward compat).
   */
  agentConfig?: AgentRuntimeConfig
  /**
   * Optional SSE chunk sink. When provided, each `UIMessageChunk` produced by
   * the agent's `fullStream` is forwarded to this callback. The orchestrator
   * uses it to buffer chunks for SSE replay / reconnect.
   */
  emitChunk?: EmitChunk
  /**
   * Optional abort signal threaded into `agent.stream({abortSignal})`. When
   * the RunRegistry cancels a run, the abort propagates here to halt
   * in-flight LLM + tool calls.
   */
  abortSignal?: AbortSignal
}

/**
 * Librarian workflow: RAG retrieval → hypothesis pool generation.
 *
 * Round 1 of Tournament Evolution. Outputs a HypothesisPool (3-6 candidate
 * hypotheses, each with statement + pythonCode) and persists each hypothesis
 * to HelixDB + the local workspace via the agent's tools.
 *
 * When `emitChunk` is supplied, every `UIMessageChunk` produced by the agent's
 * `fullStream` is forwarded to it (after conversion via `toUIMessageStream`).
 * The orchestrator (RunRegistry) buffers these for SSE replay / reconnect.
 */
export async function librarianWorkflow(input: LibrarianWorkflowInput): Promise<HypothesisPool> {
  logger.info(
    { seed: input.seed, projectId: input.projectId, runId: input.runId },
    'librarian workflow start',
  )
  const agent = await createLibrarianAgent({
    modelConfig: input.agentConfig?.modelConfig ?? input.modelConfig,
    projectId: input.projectId,
    runId: input.runId,
    runtimeContext: {
      projectId: input.projectId,
      runId: input.runId,
      round: 1,
    },
    ...(input.agentConfig?.instructions !== undefined
      ? { instructions: input.agentConfig.instructions }
      : {}),
    ...(input.agentConfig?.skillDirectories !== undefined
      ? { skillDirectories: input.agentConfig.skillDirectories }
      : {}),
    ...(input.agentConfig?.mcpServers !== undefined
      ? { mcpServers: input.agentConfig.mcpServers }
      : {}),
  })
  await ensureIndexes()
  logger.info('librarian workflow: starting agent.stream')

  const prompt = `Seed hypothesis: ${input.seed}

Generate a diverse pool of 3-6 candidate hypotheses for the coronal heating mystery. For each hypothesis:
1. State the physical mechanism (AC/DC/turbulent/combined), energy transport path, and dissipation location.
2. Give an observable prediction (which SDO/AIA/HMI/IRIS bandpass or magnetic signature should appear).
3. Give a falsifiable condition (a scenario where the prediction fails).
4. Write a pure Python filter(snapshot: dict) -> bool function with physically-derived thresholds.

Load the 'solar-physics-rag' skill first for retrieval guidance and the Python filter template. Use searchPapers and searchHypotheses to ground your hypotheses in prior work and avoid duplication. Persist each hypothesis to HelixDB via addHypothesis (roundId=0, f1Score=0, runId=${input.runId}, createdAt=now ISO 8601) and write its Python filter to the workspace via writeFile.

Return the HypothesisPool with rationale explaining your coverage strategy.`

  const result = await agent.stream({
    messages: [{ role: 'user', content: prompt }],
    ...(input.abortSignal ? { abortSignal: input.abortSignal } : {}),
  })

  await streamAgentOutput(result.fullStream, agent.tools, input.emitChunk)
  await persistAgentRun(input.projectId, input.runId, 'librarian', prompt, result)
  const staticToolCalls = await result.staticToolCalls
  return extractSubmitResult(staticToolCalls) as HypothesisPool
}

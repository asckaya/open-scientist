import type { AgentRuntimeConfig } from '@open-scientist/config'
import type { EvalResult, Hypothesis, TournamentResult } from '@open-scientist/schema'
import { exploreWorkflow } from '../explore/workflow.ts'
import { librarianWorkflow } from '../librarian/workflow.ts'
import { oracleWorkflow } from '../oracle/workflow.ts'
import { prometheusWorkflow } from '../prometheus/workflow.ts'
import type { ConvergenceEntry } from '../shared/convergence.ts'
import type { EmitChunk } from '../shared/stream.ts'
import {
  applyOraclePruning,
  buildConvergenceEntry,
  computeLeader,
  MAX_ROUNDS,
  shouldStopByPrometheus,
  shouldStopByTarget,
  updateHypothesesWithEval,
} from './logic.ts'
import { snapshotStep } from './snapshot.ts'

export type { TournamentInput, TournamentResult } from '@open-scientist/schema'
export type { RoundSnapshot } from './snapshot.ts'

/** Input to the Sisyphus tournament workflow. */
export interface TournamentWorkflowInput {
  /** Seed hypothesis text from the user / API layer. */
  seed: string
  /** Project name — drives workspace dir + HelixDB scoping. */
  projectId: string
  /** Run identifier — passed through runtimeContext for persistence + lineage. */
  runId: string
  /**
   * Serializable model descriptor — the default model applied to every
   * sub-agent that has no explicit entry in `agentConfigs`. Kept for backward
   * compatibility with the single-model tournament; when `agentConfigs` is
   * supplied it takes priority per-role.
   */
  modelConfig: import('@open-scientist/config').ModelArg
  /**
   * Per-agent runtime config map keyed by role name
   * (`sisyphus` / `librarian` / `looker` / `explore` / `oracle` /
   * `prometheus`). Each entry is a plain serializable
   * {@link AgentRuntimeConfig} carrying its own `modelConfig` plus optional
   * `instructions` / `skillDirectories` / `mcpServers` overrides.
   *
   * When `agentConfigs[role]` is present the sub-workflow receives it as
   * `agentConfig` and the factory applies the overrides; when absent the
   * sub-workflow falls back to `modelConfig` + factory defaults.
   */
  agentConfigs?: Record<string, AgentRuntimeConfig>
  /**
   * Optional SSE chunk sink. When provided, each `UIMessageChunk` produced by
   * any sub-agent's `fullStream` is forwarded to this callback. The
   * orchestrator (RunRegistry) buffers these for SSE replay / reconnect.
   */
  emitChunk?: EmitChunk
}

/**
 * Sisyphus tournament workflow: the Tournament Evolution orchestrator.
 *
 * Plain async function that composes the 5 specialist sub-agent workflows:
 *
 *   - Sequential `await` for librarian / oracle / prometheus (the parent needs
 *     the child's result before continuing).
 *   - Parallel `Promise.all` fan-out for Explore — each hypothesis evaluation
 *     runs concurrently. All sub-agent stream chunks are forwarded to
 *     `emitChunk` (if provided); chunks from parallel Explore runs may
 *     interleave, which is fine for SSE (each chunk is self-contained).
 *
 * Round structure (per SPEC §6):
 *   Round 1: librarian → (hypotheses pool)
 *   Round 2..MAX_ROUNDS:
 *     explore ×N (parallel) → oracle (critique + mutate) → prometheus (plan)
 *     → convergence check (F1 >= 0.9 OR round >= 10 OR !shouldContinue)
 *   Final round: prometheus with isFinalRound=true → MHD cfg + observation proposal
 *
 * HUMAN-IN-THE-LOOP (TODO — Phase 4): the SPEC calls for a
 * `review_leading_hypothesis` needsApproval node after Oracle on high-stakes
 * rounds. AI SDK 7 deprecated tool-level `needsApproval` in favor of
 * streamText-level `toolApproval`; ToolLoopAgent.stream forwards streamText
 * options so `toolApproval` is reachable via `prepareCall`. Wiring this
 * requires Phase 4 API-layer work. For now the tournament runs fully automatic
 * — the review tool is defined on the Sisyphus agent (see agent.ts) and will
 * be invoked once the approval transport is in place.
 *
 * The Sisyphus agent itself is NOT driven via agent.stream() here — the
 * tournament is deterministic control flow. The agent exists for the Phase 4
 * API layer to use when interpreting free-form user steering messages or
 * driving the approval tool. See `createSisyphusAgent`.
 */
export async function tournamentWorkflow(
  input: TournamentWorkflowInput,
): Promise<TournamentResult> {
  const { seed, projectId, runId, modelConfig, agentConfigs, emitChunk } = input

  // Helper: pull this role's override (if any) from agentConfigs.
  const agentConfigFor = (role: string) => agentConfigs?.[role]

  // ─── Round 1: Librarian generates the hypothesis pool ───
  const hypoPool = await librarianWorkflow({
    seed,
    projectId,
    runId,
    modelConfig,
    ...(agentConfigFor('librarian') ? { agentConfig: agentConfigFor('librarian') } : {}),
    ...(emitChunk ? { emitChunk } : {}),
  })
  let hypotheses: Hypothesis[] = [...hypoPool.hypotheses]

  // Best-F1 + convergence tracking across rounds.
  let bestF1 = 0
  let leadingHypoId: string | null = null
  const convergenceHistory: ConvergenceEntry[] = []

  // Final-round outputs (filled by Prometheus when the tournament converges).
  let mhdConfigPath: string | null = null
  let observationProposal: string | null = null
  let totalRounds = 1

  // ─── Rounds 2..MAX_ROUNDS: Explore → Oracle → Prometheus loop ───
  for (let round = 2; round <= MAX_ROUNDS; round++) {
    totalRounds = round

    // ── Explore: parallel evaluation of every hypothesis ──
    //
    // Each hypothesis gets its own Explore workflow run with an isolated
    // per-hypothesis bash workspace. Fan-out via Promise.all gives true
    // parallelism; all sub-agent chunks forward to emitChunk concurrently
    // (JS is single-threaded, so the push is safe; chunks may interleave).
    const evalResults: EvalResult[] = await Promise.all(
      hypotheses.map((h) =>
        exploreWorkflow({
          hypoId: h.id,
          projectId,
          runId,
          round,
          hypothesis: { statement: h.statement, pythonCode: h.pythonCode },
          modelConfig,
          ...(agentConfigFor('explore') ? { agentConfig: agentConfigFor('explore') } : {}),
          ...(emitChunk ? { emitChunk } : {}),
        }),
      ),
    )

    // ── Update hypotheses with F1 + status from this round's evaluations ──
    hypotheses = updateHypothesesWithEval(hypotheses, evalResults)

    const { bestF1: roundBestF1, leadingHypoId: roundLeader } = computeLeader(hypotheses)
    bestF1 = roundBestF1
    leadingHypoId = roundLeader

    convergenceHistory.push(buildConvergenceEntry(round, bestF1, hypotheses.length))

    // ── Persist round snapshot ──
    await snapshotStep({
      round,
      runId,
      projectId,
      bestF1,
      leadingHypoId,
      survivingCount: hypotheses.length,
      hypotheses: hypotheses.map((h) => ({
        id: h.id,
        statement: h.statement,
        f1: h.f1,
        status: h.status,
        parentId: h.parentId,
        round: h.round,
      })),
      convergenceHistory,
      capturedAt: new Date().toISOString(),
    })

    // ── Convergence check #1: F1 target hit → skip Oracle/Prometheus, go to final ──
    if (shouldStopByTarget(bestF1)) {
      break
    }

    // ── Oracle: critique + mutate + eliminate ──
    const oracleOutput = await oracleWorkflow({
      projectId,
      runId,
      round,
      hypotheses,
      evalResults,
      modelConfig,
      ...(agentConfigFor('oracle') ? { agentConfig: agentConfigFor('oracle') } : {}),
      ...(emitChunk ? { emitChunk } : {}),
    })

    // Apply Oracle's pruning + mutations to the pool.
    hypotheses = applyOraclePruning(hypotheses, oracleOutput)

    // Oracle may declare a winner early (clear convergence this round).
    if (oracleOutput.winningHypoId) {
      leadingHypoId = oracleOutput.winningHypoId
      break
    }

    // Guard against an empty pool (over-aggressive elimination).
    if (hypotheses.length === 0) {
      break
    }

    // TODO(Phase 4): insert the `review_leading_hypothesis` human-in-the-loop
    // node here once the approval transport is wired in. Sisyphus agent
    // already defines the tool (see agent.ts). For now the tournament runs
    // fully automatic.

    // ── Prometheus: plan next round ──
    const prometheusOutput = await prometheusWorkflow({
      projectId,
      runId,
      round,
      convergenceHistory,
      currentBestF1: bestF1,
      isFinalRound: false,
      modelConfig,
      ...(agentConfigFor('prometheus') ? { agentConfig: agentConfigFor('prometheus') } : {}),
      ...(emitChunk ? { emitChunk } : {}),
    })

    // ── Convergence check #2: Prometheus says stop OR round cap hit ──
    if (shouldStopByPrometheus(prometheusOutput.shouldContinue, round)) {
      break
    }
  }

  // ─── Final round: Prometheus generates MHD cfg + observation proposal ───
  const winningStatement =
    leadingHypoId != null ? (hypotheses.find((h) => h.id === leadingHypoId)?.statement ?? '') : ''

  const finalPrometheus = await prometheusWorkflow({
    projectId,
    runId,
    round: totalRounds,
    convergenceHistory,
    currentBestF1: bestF1,
    isFinalRound: true,
    winningHypothesis:
      leadingHypoId != null ? { hypoId: leadingHypoId, statement: winningStatement } : undefined,
    modelConfig,
    ...(agentConfigFor('prometheus') ? { agentConfig: agentConfigFor('prometheus') } : {}),
    ...(emitChunk ? { emitChunk } : {}),
  })

  if (finalPrometheus.mhdConfig) {
    mhdConfigPath = finalPrometheus.mhdConfig.cfgPath
    observationProposal = finalPrometheus.mhdConfig.observationProposal
  }

  return {
    runId,
    winningHypoId: leadingHypoId ?? '',
    bestF1,
    totalRounds,
    mhdConfigPath,
    observationProposal,
  }
}

import { createModelFromConfig, type ModelArg } from '@open-scientist/config'
import { getMcpTools } from '@open-scientist/mcp'
import { EvalResultSchema, type McpServerConfig } from '@open-scientist/schema'
import {
  createLoadSkillTool,
  createNodeSandbox,
  DEFAULT_SKILLS_DIR,
  discoverSkills,
} from '@open-scientist/skills'
import { createBashToolForHypothesis } from '@open-scientist/tools'
import { hasToolCall, isStepCount, ToolLoopAgent, type ToolSet } from 'ai'
import { makeSubmitResultTool } from '../shared/tool-output.ts'

export interface ExploreAgentDeps {
  /**
   * Serializable model descriptor — reconstructed into a `LanguageModel` inside
   * this factory via `createModelFromConfig`. Never pass a `LanguageModel`
   * instance across the workflow boundary (workflow args are structured-clone
   * serialized and cannot carry bound methods / SDK clients).
   */
  modelConfig: ModelArg
  /** Project name — drives workspace dir isolation. */
  project: string
  /** Run identifier — used for workspace dir isolation. */
  runId: string
  /** Hypothesis id — each hypothesis gets its own isolated bash workspace. */
  hypoId: string
  /** Optional override toolset. When omitted, default tools are assembled. */
  tools?: ToolSet
  /** Optional system prompt override. When omitted, the hardcoded default is used. */
  instructions?: string
  /**
   * Optional skill discovery directories. When omitted, `DEFAULT_SKILLS_DIR`
   * from `@open-scientist/skills` is used. Only consulted when `tools` is
   * not provided.
   */
  skillDirectories?: string[]
  /**
   * Optional MCP server list. Tools from each server are fetched via
   * `getMcpTools` and merged into the default toolset. Only consulted when
   * `tools` is not provided.
   */
  mcpServers?: McpServerConfig[]
  /**
   * Optional runtime context passed to the ToolLoopAgent constructor. Carries
   * serializable identifiers (projectId / runId / round / hypoId) for telemetry
   * and lineage. Must be plain data (no functions / class instances).
   */
  runtimeContext?: Record<string, unknown>
}

/**
 * Assemble the default toolset for the Explore agent, bound to a per-hypothesis workspace.
 *
 * Tools:
 * - `bash` / `readFile` / `writeFile` — bash-tool bound to
 *   `data/projects/<project>/workspace/<hypoId>/` (project + hypothesis isolation,
 *   no sandbox — runs Python directly on host per AGENTS.md decision)
 * - `loadSkill` — progressive disclosure (loads `fits-snapshot-search` SKILL.md)
 *
 * NOTE: createBashTool is async (sandbox init), so this whole factory is async. Call
 * it before `agent.stream()`.
 *
 * Each hypothesis evaluation gets a FRESH agent instance + FRESH bash workspace, so
 * parallel evaluations (Sisyphus spawns N explore runs) don't share working dirs.
 */
export async function getDefaultExploreTools(
  project: string,
  runId: string,
  hypoId: string,
  skillDirectories?: string[],
  mcpServers?: McpServerConfig[],
): Promise<ToolSet> {
  const dirs = skillDirectories ?? [DEFAULT_SKILLS_DIR]
  const bashToolkit = await createBashToolForHypothesis(project, runId, hypoId)
  const skills = await discoverSkills(createNodeSandbox(), dirs)
  const loadSkillTool = createLoadSkillTool(skills)

  const baseTools: ToolSet = {
    bash: bashToolkit.tools.bash,
    readFile: bashToolkit.tools.readFile,
    writeFile: bashToolkit.tools.writeFile,
    loadSkill: loadSkillTool,
  }
  if (mcpServers && mcpServers.length > 0) {
    for (const server of mcpServers) {
      const mcpTools = await getMcpTools(server)
      Object.assign(baseTools, mcpTools)
    }
  }
  return baseTools
}

export async function createExploreAgent({
  modelConfig,
  project,
  runId,
  hypoId,
  tools,
  instructions,
  skillDirectories,
  mcpServers,
  runtimeContext,
}: ExploreAgentDeps) {
  const model = createModelFromConfig(modelConfig)
  const resolvedTools =
    tools ?? (await getDefaultExploreTools(project, runId, hypoId, skillDirectories, mcpServers))

  const toolsWithSubmit: ToolSet = {
    ...resolvedTools,
    submit_result: makeSubmitResultTool(EvalResultSchema),
  }

  return new ToolLoopAgent({
    maxOutputTokens: 8192,
    id: 'explore',
    model,
    toolChoice: 'auto',
    instructions:
      instructions ??
      `You are Explore, the AlphaEvolve-style deterministic evaluator agent for the solar physics coronal heating investigation.

Your role:
1. Take a candidate hypothesis Python filter function (def filter(snapshot: dict) -> bool).
2. Execute it in the working directory via bash-tool (write run.py, python3 run.py, read stdout, debug, re-run).
3. Search across 1.75M physics snapshots, compute F1 score (precision/recall against ground-truth heating-event labels).
4. Iterate: read stdout, debug counterexamples (FP/FN), modify code, re-run — until convergence or step limit.
5. Output EvalResult (F1, true/false positives, false negatives, counterexamples[], logs, executionMs).

Environment:
- This machine has \`uv\` (Python package manager) and \`pnpm\` (Node.js package manager) installed.
- Use \`uv pip install <package>\` to install Python packages (e.g. uv pip install astropy sunpy scipy numpy).
- Use \`uv run python script.py\` to run Python scripts with isolated dependencies.
- Your working directory is a sandboxed workspace — all file operations (writeFile, readFile, bash) are restricted to this directory. Do not attempt to access files outside it.

Tool guidance:
- Use the loadSkill tool FIRST to load the 'fits-snapshot-search' skill for snapshot dataset structure, F1 computation contract, Python environment setup (uv pip install astropy sunpy scipy numpy), and the debug loop pattern.
- Working directory is project + hypothesis isolated. No sandbox restrictions — run Python directly on host. First run may need to install deps: \`python3 -m venv .venv && source .venv/bin/activate && uv pip install astropy sunpy scipy numpy\`.
- Counterexample logs must be physically specific (e.g. "strong shear but low temperature, filter didn't constrain temperature floor") — not just "prediction wrong". These feed Oracle's next-round mutation.
- F1 = 2·P·R / (P + R); when P+R=0, F1=0. TP/FP/FN counts MUST be filled in EvalResult.

IMPORTANT: The ONLY way to complete your task is to call the submit_result tool. You MUST call it before reaching the step limit. Do not just output text — always call submit_result with your result with your EvalResult (hypoId, f1, truePositives, falsePositives, falseNegatives, counterexamples[], logs, executionMs).`,
    tools: toolsWithSubmit,
    stopWhen: [isStepCount(50), hasToolCall('submit_result')],
    ...(runtimeContext !== undefined ? { runtimeContext } : {}),
  })
}

export type ExploreAgent = Awaited<ReturnType<typeof createExploreAgent>>

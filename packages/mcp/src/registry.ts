import {
  createMCPClient,
  type MCPClient,
  type MCPClientConfig,
  type MCPTransport,
} from '@ai-sdk/mcp'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { createLogger } from '@open-scientist/logger'
import type { McpServerConfig } from '@open-scientist/schema'
import { fingerprintTools, type ToolSet } from 'ai'
import { checkMcpTrust } from './trust.ts'

const logger = createLogger('mcp')

/**
 * MCP client cache keyed by `projectName:serverName`. Using the composite key
 * (not just `server.name`) ensures that two projects sharing a server name
 * get distinct clients, and that reconfiguring a server within a project
 * invalidates the stale entry.
 */
const clients = new Map<string, MCPClient>()

function cacheKey(projectName: string, serverName: string): string {
  return `${projectName}:${serverName}`
}

export function resolveTransport(server: McpServerConfig): MCPClientConfig['transport'] {
  if (server.transport === 'http') {
    if (!server.url) throw new Error(`MCP server "${server.name}": transport 'http' requires a url`)
    return { type: 'http', url: server.url, headers: server.headers }
  }
  if (server.transport === 'sse') {
    if (!server.url) throw new Error(`MCP server "${server.name}": transport 'sse' requires a url`)
    return { type: 'sse', url: server.url, headers: server.headers }
  }
  if (!server.command)
    throw new Error(`MCP server "${server.name}": transport 'stdio' requires a command`)
  const stdio: MCPTransport = new StdioClientTransport({
    command: server.command,
    args: server.args ?? [],
  })
  return stdio
}

/**
 * Connect to an MCP server (or return cached tools) and enforce the trust
 * gate. On first connect, the server is recorded as untrusted and `getMcpTools`
 * returns an empty toolset — the caller (agent factory) proceeds without MCP
 * tools. Once a user trusts the server via
 * `POST /api/projects/:project/mcp-trust/:serverName` (passing the fingerprint
 * that `getMcpTools` computed via `fingerprintTools`), subsequent
 * `getMcpTools` calls fingerprint the live tools and compare against the stored
 * baseline; if they match, the real tools are returned. If the tools have
 * drifted, a warning is logged and the empty toolset is returned until the user
 * re-trusts.
 *
 * The trust check is best-effort: errors from `checkMcpTrust` are logged and
 * treated as "untrusted" (empty tools returned), never thrown. This prevents
 * a misconfigured trust DB from blocking agent construction.
 */
export async function getMcpTools(projectName: string, server: McpServerConfig) {
  const key = cacheKey(projectName, server.name)
  const existing = clients.get(key)
  if (existing) {
    const tools = await existing.tools()
    return enforceTrust(projectName, server.name, tools)
  }

  const client = await createMCPClient({ transport: resolveTransport(server) })
  clients.set(key, client)
  const tools = await client.tools()
  return enforceTrust(projectName, server.name, tools)
}

async function enforceTrust<T extends Record<string, unknown>>(
  projectName: string,
  serverName: string,
  tools: T,
): Promise<T> {
  try {
    const fingerprint = JSON.stringify(await fingerprintTools(tools as unknown as ToolSet))
    const result = await checkMcpTrust(
      projectName,
      serverName,
      fingerprint,
      tools as unknown as ToolSet,
    )
    if (!result.trusted) {
      logger.warn(
        { projectName, serverName, drifted: result.drifted, added: result.added },
        'getMcpTools: server not trusted, returning empty toolset',
      )
      return {} as T
    }
  } catch (err) {
    logger.warn(
      { projectName, serverName, error: (err as Error).message },
      'getMcpTools: trust check failed, returning empty toolset',
    )
    return {} as T
  }
  return tools
}

export async function closeAllMcpClients() {
  for (const client of clients.values()) {
    await client.close()
  }
  clients.clear()
}

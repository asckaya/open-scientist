import { randomUUID } from 'node:crypto'
import { and, eq } from 'drizzle-orm'
import { getGlobalDb } from '../global-db.ts'
import { mcpTrust } from '../schema/global.ts'

export async function getTrust(projectName: string, serverName: string) {
  const { db } = await getGlobalDb()
  return (
    db
      .select()
      .from(mcpTrust)
      .where(and(eq(mcpTrust.projectName, projectName), eq(mcpTrust.serverName, serverName)))
      .all()[0] ?? null
  )
}

export async function setTrust(
  projectName: string,
  serverName: string,
  fingerprint: string,
  trusted: boolean,
) {
  const { db } = await getGlobalDb()
  // Delete existing rows for this (projectName, serverName) pair to avoid duplicates
  db.delete(mcpTrust)
    .where(and(eq(mcpTrust.projectName, projectName), eq(mcpTrust.serverName, serverName)))
    .run()
  const id = randomUUID()
  const now = new Date().toISOString()
  db.insert(mcpTrust)
    .values({ id, projectName, serverName, fingerprint, trusted, firstSeen: now, lastChecked: now })
    .run()
}

import { createContext, useContext, useRef } from 'react'
import type { AgentChainEntry, CollectionInfo, InstructionsData, ProviderEntry } from './client'
import { agentChains, config, invalidateCache, knowledge, providers } from './client'

type CacheEntry<T> = {
  data: T
  ts: number
  promise: Promise<T> | null
}

const TTL = 60000

const cache = {
  models: null as CacheEntry<string[]> | null,
  collections: null as CacheEntry<CollectionInfo[]> | null,
  instructions: null as CacheEntry<InstructionsData> | null,
  providers: null as CacheEntry<ProviderEntry[]> | null,
  chains: null as CacheEntry<AgentChainEntry[]> | null,
}

function isStale<T>(entry: CacheEntry<T> | null): boolean {
  return !entry || Date.now() - entry.ts > TTL
}

async function loadModels(): Promise<string[]> {
  const entry = cache.models
  if (!isStale(entry) && entry) return entry.data
  if (entry?.promise) return entry.promise

  const promise = config.listModels().catch(() => [] as string[])
  cache.models = { data: [] as string[], ts: Date.now(), promise }
  const data = await promise
  cache.models = { data, ts: Date.now(), promise: null }
  return data
}

async function loadCollections(): Promise<CollectionInfo[]> {
  const entry = cache.collections
  if (!isStale(entry) && entry) return entry.data
  if (entry?.promise) return entry.promise

  const promise = knowledge.listCollections().catch(() => [] as CollectionInfo[])
  cache.collections = { data: [] as CollectionInfo[], ts: Date.now(), promise }
  const data = await promise
  cache.collections = { data, ts: Date.now(), promise: null }
  return data
}

async function loadInstructions(): Promise<InstructionsData> {
  const entry = cache.instructions
  if (!isStale(entry) && entry) return entry.data
  if (entry?.promise) return entry.promise

  const promise = config.getInstructions().catch(() => ({ assistants: [] }))
  cache.instructions = { data: { assistants: [] }, ts: Date.now(), promise }
  const data = await promise
  cache.instructions = { data, ts: Date.now(), promise: null }
  return data
}

async function loadProviders(): Promise<ProviderEntry[]> {
  const entry = cache.providers
  if (!isStale(entry) && entry) return entry.data
  if (entry?.promise) return entry.promise

  const promise = providers.list().catch(() => [] as ProviderEntry[])
  cache.providers = { data: [] as ProviderEntry[], ts: Date.now(), promise }
  const data = await promise
  cache.providers = { data, ts: Date.now(), promise: null }
  return data
}

async function loadChains(): Promise<AgentChainEntry[]> {
  const entry = cache.chains
  if (!isStale(entry) && entry) return entry.data
  if (entry?.promise) return entry.promise

  const promise = agentChains.list().catch(() => [] as AgentChainEntry[])
  cache.chains = { data: [] as AgentChainEntry[], ts: Date.now(), promise }
  const data = await promise
  cache.chains = { data, ts: Date.now(), promise: null }
  return data
}

export function refreshInstructions(): Promise<InstructionsData> {
  cache.instructions = null
  return loadInstructions()
}

export function refreshCollections(): Promise<CollectionInfo[]> {
  cache.collections = null
  return loadCollections()
}

export function refreshProviders(): Promise<ProviderEntry[]> {
  cache.providers = null
  return loadProviders()
}

export function refreshChains(): Promise<AgentChainEntry[]> {
  cache.chains = null
  return loadChains()
}

export function invalidateModels(): void {
  cache.models = null
}

export function invalidateCollections(): void {
  cache.collections = null
}

export function invalidateInstructions(): void {
  cache.instructions = null
}

export function invalidateProviders(): void {
  cache.providers = null
}

export function invalidateChains(): void {
  cache.chains = null
  invalidateCache('/admin/agent-chains')
}

export const DataContext = createContext<{
  getModels: () => Promise<string[]>
  getCollections: () => Promise<CollectionInfo[]>
  getInstructions: () => Promise<InstructionsData>
  getProviders: () => Promise<ProviderEntry[]>
  refreshInstructions: () => Promise<InstructionsData>
  refreshCollections: () => Promise<CollectionInfo[]>
  refreshProviders: () => Promise<ProviderEntry[]>
  getChains: () => Promise<AgentChainEntry[]>
  refreshChains: () => Promise<AgentChainEntry[]>
  invalidateModels: () => void
  invalidateCollections: () => void
  invalidateInstructions: () => void
  invalidateProviders: () => void
  invalidateChains: () => void
}>({
  getModels: loadModels,
  getCollections: loadCollections,
  getInstructions: loadInstructions,
  getProviders: loadProviders,
  getChains: loadChains,
  refreshInstructions,
  refreshCollections,
  refreshProviders,
  refreshChains,
  invalidateModels,
  invalidateCollections,
  invalidateInstructions,
  invalidateProviders,
  invalidateChains,
})

export function DataProvider({ children }: { children: React.ReactNode }) {
  const valueRef = useRef({
    getModels: loadModels,
    getCollections: loadCollections,
    getInstructions: loadInstructions,
    getProviders: loadProviders,
    getChains: loadChains,
    refreshInstructions,
    refreshCollections,
    refreshProviders,
    refreshChains,
    invalidateModels,
    invalidateCollections,
    invalidateInstructions,
    invalidateProviders,
    invalidateChains,
  })

  return (
    <DataContext.Provider value={valueRef.current}>
      {children}
    </DataContext.Provider>
  )
}

export function useData() {
  return useContext(DataContext)
}

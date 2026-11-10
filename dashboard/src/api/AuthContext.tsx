import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from 'react'
import { auth, type UserInfo } from './client'

export type AuthState = {
  authenticated: boolean
  user: UserInfo | null
  permissions: string[]
  loading: boolean
}

export type AuthContextValue = AuthState & {
  refresh: () => Promise<void>
  login: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
  hasPermission: (permission: string) => boolean
  hasAnyPermission: (permissions: string[]) => boolean
}

const AuthContext = createContext<AuthContextValue>({
  authenticated: false,
  user: null,
  permissions: [],
  loading: true,
  refresh: async () => {},
  login: async () => {},
  logout: async () => {},
  hasPermission: () => false,
  hasAnyPermission: () => false,
})

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    authenticated: false,
    user: null,
    permissions: [],
    loading: true,
  })

  const refresh = useCallback(async () => {
    try {
      const data = await auth.me()
      setState({
        authenticated: data.authenticated,
        user: data.user ?? null,
        permissions: data.permissions ?? [],
        loading: false,
      })
    } catch {
      setState({ authenticated: false, user: null, permissions: [], loading: false })
    }
  }, [])

  const login = useCallback(async (username: string, password: string) => {
    await auth.login(username, password)
    await refresh()
  }, [refresh])

  const logout = useCallback(async () => {
    try {
      await auth.logout()
    } catch {
      // ignore
    }
    setState({ authenticated: false, user: null, permissions: [], loading: false })
  }, [])

  const hasPermission = useCallback(
    (permission: string) => state.permissions.includes(permission),
    [state.permissions]
  )

  const hasAnyPermission = useCallback(
    (permissions: string[]) => permissions.some((p) => state.permissions.includes(p)),
    [state.permissions]
  )

  /* Session keepalive: the backend re-issues the dashboard cookie past half
     its TTL, so pinging /admin/auth/me keeps the session alive through long
     agent runs. The cadence adapts to the server-reported TTL (ttl/4, clamped
     to 30s–5min) so even short TTLs can't expire between pings. Failures are
     ignored so a transient network blip never logs the user out mid-run. */
  useEffect(() => {
    if (!state.authenticated) return
    let timer: ReturnType<typeof setTimeout> | undefined
    let stopped = false
    const schedule = (ms: number) => {
      if (stopped) return
      if (timer) clearTimeout(timer)
      timer = setTimeout(ping, Math.min(5 * 60 * 1000, Math.max(30_000, ms)))
    }
    const ping = () => {
      auth.me()
        .then((data) => {
          setState({
            authenticated: data.authenticated,
            user: data.user ?? null,
            permissions: data.permissions ?? [],
            loading: false,
          })
          schedule((data.session_ttl_seconds ?? 3600) * 250)
        })
        .catch(() => schedule(5 * 60 * 1000))
    }
    schedule(5 * 60 * 1000)
    const onVisible = () => { if (document.visibilityState === 'visible') ping() }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      stopped = true
      if (timer) clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [state.authenticated])

  return (
    <AuthContext.Provider
      value={{
        ...state,
        refresh,
        login,
        logout,
        hasPermission,
        hasAnyPermission,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}

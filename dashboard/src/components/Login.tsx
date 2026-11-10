import { useState } from 'react'
import logo from '../assets/logo.png'
import { useAuth } from '../api/AuthContext'
import { useBranding } from '../ui/BrandingContext'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { login } = useAuth()
  const { branding } = useBranding()

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(username, password)
    } catch (err: any) {
      setError(err?.message || $t('login.error'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-page">
      <form className="login-card" onSubmit={handleSubmit}>
        <div className="login-header">
          <img src={branding.logo_url || logo} alt={branding.app_name ? `${branding.app_name} Logo` : 'Agentix Logo'} className="login-logo" />
          <h1>{branding.app_name ? `${branding.app_name} Dashboard` : $t('login.title')}</h1>
          <p>{$t('login.subtitle')}</p>
        </div>
        {error && <div className="login-error">{error}</div>}
        <input
          type="text"
          placeholder={$t('login.username')}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder={$t('login.password')}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? $t('login.loading') : $t('login.submit')}
        </button>
      </form>
    </div>
  )
}

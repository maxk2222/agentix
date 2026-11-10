import { useEffect, useState } from 'react'
import { Pencil, Ban, Check, Trash2, X, Plus, ShieldAlert, MessageSquare, ArrowUpDown, Users2, Link2, RefreshCw, Copy } from 'lucide-react'
import { users as usersApi, chatUsers, chatGroups, agentChains, config as configApi, invalidateCache, type DashboardUser, type ChatGroupEntry } from '../api/client'
import { useAuth } from '../api/AuthContext'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import ConfirmModal from './ConfirmModal'

const GROUP_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#06b6d4', '#84cc16']

const ROLES = ['viewer', 'operator', 'editor', 'admin', 'owner']
const CHAT_ROLES = ['anonymous', 'standard', 'pro', 'admin']

const ROLE_COLORS: Record<string, string> = {
  viewer: '#8b5cf6',
  operator: '#3b82f6',
  editor: '#10b981',
  admin: '#f59e0b',
  owner: '#ef4444',
}

export default function UserManagement() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { hasPermission } = useAuth()
  const canWrite = hasPermission('users:write')

  const [activeTab, setActiveTab] = useState<'dashboard' | 'chat' | 'groups'>('dashboard')

  const [userList, setUserList] = useState<DashboardUser[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const [chatUserList, setChatUserList] = useState<Array<import('../api/client').ChatUserEntry>>([])
  const [chatLoading, setChatLoading] = useState(true)
  const [chatError, setChatError] = useState('')
  const [chatEditing, setChatEditing] = useState<import('../api/client').ChatUserEntry | null>(null)
  const [chatForm, setChatForm] = useState({ display_name: '', role: 'standard', is_active: true, password: '', email: '' })

  const [groupList, setGroupList] = useState<ChatGroupEntry[]>([])
  const [groupLoading, setGroupLoading] = useState(true)
  const [groupError, setGroupError] = useState('')
  const [groupEditing, setGroupEditing] = useState<ChatGroupEntry | null>(null)
  const [showGroupForm, setShowGroupForm] = useState(false)
  const [groupForm, setGroupForm] = useState({ name: '', description: '', color: GROUP_COLORS[0], unrestricted: true, allowed_models: [] as string[], auto_domains: [] as string[], member_ids: [] as string[] })
  const [domainInput, setDomainInput] = useState('')
  const [modelOptions, setModelOptions] = useState<string[]>([])
  const [widgetLink, setWidgetLink] = useState<{ id: string; url: string } | null>(null)
  const [groupInfo, setGroupInfo] = useState('')
  const [showChatForm, setShowChatForm] = useState(false)
  const [chatSort, setChatSort] = useState<{ column: string; direction: 'asc' | 'desc' }>({ column: 'username', direction: 'asc' })
  const [pendingConfirm, setPendingConfirm] = useState<{ message: string; action: () => void } | null>(null)

  const [showForm, setShowForm] = useState(false)
  const [editing, setEditing] = useState<DashboardUser | null>(null)
  const [form, setForm] = useState({
    username: '',
    display_name: '',
    password: '',
    role: 'viewer',
    is_active: true,
  })

  async function load() {
    setLoading(true)
    try {
      const data = await usersApi.list()
      setUserList(data)
      setError('')
    } catch (err: any) {
      setError(err?.message || 'Failed to load users')
    } finally {
      setLoading(false)
    }
  }

  async function loadChatUsers() {
    setChatLoading(true)
    try {
      const data = await chatUsers.list()
      setChatUserList(data)
      setChatError('')
    } catch (err: any) {
      setChatError(err?.message || 'Failed to load chat users')
    } finally {
      setChatLoading(false)
    }
  }

  async function loadGroups() {
    setGroupLoading(true)
    setGroupInfo('')
    try {
      const data = await chatGroups.list()
      setGroupList(data)
      setGroupError('')
    } catch (err: any) {
      setGroupError(err?.message || 'Failed to load groups')
    } finally {
      setGroupLoading(false)
    }
  }

  async function loadModelOptions() {
    try {
      const data = await configApi.getInstructions()
      const names = (data.assistants || []).map((a) => a.name)
      const chains = await agentChains.list().catch(() => [])
      setModelOptions([...names, ...chains.filter((c) => c.enabled).map((c) => `chain:${c.name}`)])
    } catch {
      setModelOptions([])
    }
  }

  useEffect(() => {
    load()
    loadChatUsers()
    loadGroups()
    loadModelOptions()
  }, [])

  function startCreate() {
    setEditing(null)
    setForm({ username: '', display_name: '', password: '', role: 'viewer', is_active: true })
    setShowForm(true)
  }

  function startEdit(user: DashboardUser) {
    if (user.id === 'env-admin') return
    setEditing(user)
    setForm({
      username: user.username,
      display_name: user.display_name,
      password: '',
      role: user.role,
      is_active: user.is_active,
    })
    setShowForm(true)
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    try {
      if (editing) {
        await usersApi.update(editing.id, {
          username: form.username,
          display_name: form.display_name,
          role: form.role,
          is_active: form.is_active,
        })
        if (form.password) {
          await usersApi.setPassword(editing.id, form.password)
        }
      } else {
        await usersApi.create({
          username: form.username,
          display_name: form.display_name,
          password: form.password,
          role: form.role,
        })
      }
      invalidateCache('/admin/users')
      setShowForm(false)
      await load()
    } catch (err: any) {
      setError(err?.message || 'Failed to save user')
    }
  }

  async function handleToggle(user: DashboardUser) {
    if (user.id === 'env-admin') return
    try {
      if (user.is_active) {
        await usersApi.disable(user.id)
      } else {
        await usersApi.enable(user.id)
      }
      invalidateCache('/admin/users')
      await load()
    } catch (err: any) {
      setError(err?.message || 'Failed to toggle user')
    }
  }

  async function handleDelete(user: DashboardUser) {
    if (user.id === 'env-admin') return
    setPendingConfirm({ message: `Delete user "${user.username}"?`, action: () => void doDelete(user) })
  }

  async function doDelete(user: DashboardUser) {
    try {
      await usersApi.delete(user.id)
      invalidateCache('/admin/users')
      await load()
    } catch (err: any) {
      setError(err?.message || 'Failed to delete user')
    }
  }

  function startChatEdit(user: import('../api/client').ChatUserEntry) {
    setChatEditing(user)
    setChatForm({ display_name: user.display_name, role: user.role, is_active: user.is_active, password: '', email: user.email || '' })
    setShowChatForm(true)
  }

  async function handleChatSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!chatEditing) return
    try {
      await chatUsers.update(chatEditing.id, {
        display_name: chatForm.display_name,
        role: chatForm.role,
        is_active: chatForm.is_active,
        email: chatForm.email.trim(),
      })
      if (chatForm.password) {
        await chatUsers.setPassword(chatEditing.id, chatForm.password)
      }
      setShowChatForm(false)
      await loadChatUsers()
    } catch (err: any) {
      setChatError(err?.message || 'Failed to save chat user')
    }
  }

  function handleChatSort(column: string) {
    setChatSort((prev) => ({
      column,
      direction: prev.column === column && prev.direction === 'asc' ? 'desc' : 'asc',
    }))
  }

  const sortedChatUsers = [...chatUserList].sort((a, b) => {
    const dir = chatSort.direction === 'asc' ? 1 : -1
    const aVal = (a as any)[chatSort.column]
    const bVal = (b as any)[chatSort.column]
    if (aVal === null || aVal === undefined) return 1 * dir
    if (bVal === null || bVal === undefined) return -1 * dir
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return aVal.localeCompare(bVal) * dir
    }
    return (aVal < bVal ? -1 : 1) * dir
  })

  async function handleChatDelete(user: import('../api/client').ChatUserEntry) {
    setPendingConfirm({ message: `Delete chat user "${user.username}"?`, action: () => void doChatDelete(user) })
  }

  async function doChatDelete(user: import('../api/client').ChatUserEntry) {
    try {
      await chatUsers.delete(user.id)
      await loadChatUsers()
    } catch (err: any) {
      setChatError(err?.message || 'Failed to delete chat user')
    }
  }

  // --- chat groups ---
  function startGroupCreate() {
    setGroupEditing(null)
    setGroupForm({ name: '', description: '', color: GROUP_COLORS[0], unrestricted: true, allowed_models: [], auto_domains: [], member_ids: [] })
    setDomainInput('')
    setWidgetLink(null)
    setShowGroupForm(true)
  }

  async function startGroupEdit(g: ChatGroupEntry) {
    // The member checklist must come from the detail endpoint — falling back
    // to the card data (which has no members) would wipe memberships on save.
    const detail = await chatGroups.get(g.id).catch((err) => {
      setGroupError(err?.message || 'Failed to load group')
      return null
    })
    if (!detail) return
    setGroupEditing(detail)
    setGroupForm({
      name: detail.name,
      description: detail.description,
      color: detail.color || GROUP_COLORS[0],
      unrestricted: detail.allowed_models === null,
      allowed_models: detail.allowed_models || [],
      auto_domains: detail.auto_domains || [],
      member_ids: (detail.members || []).map((m) => m.user_id),
    })
    setDomainInput('')
    setWidgetLink(null)
    setShowGroupForm(true)
  }

  async function handleGroupSubmit(e: React.FormEvent) {
    e.preventDefault()
    try {
      const payload = {
        name: groupForm.name.trim(),
        description: groupForm.description,
        color: groupForm.color,
        allowed_models: groupForm.unrestricted ? null : groupForm.allowed_models,
        auto_domains: groupForm.auto_domains,
      }
      let saved: ChatGroupEntry
      if (groupEditing) {
        saved = await chatGroups.update(groupEditing.id, payload)
      } else {
        saved = await chatGroups.create(payload)
      }
      await chatGroups.setMembers(saved.id, groupForm.member_ids)
      setShowGroupForm(false)
      await loadGroups()
      await loadChatUsers()
    } catch (err: any) {
      setGroupError(err?.message || 'Failed to save group')
    }
  }

  function handleGroupDelete(g: ChatGroupEntry) {
    setPendingConfirm({ message: `Delete group "${g.name}"? Members keep their other groups and roles.`, action: () => void doGroupDelete(g) })
  }

  async function doGroupDelete(g: ChatGroupEntry) {
    try {
      await chatGroups.delete(g.id)
      await loadGroups()
      await loadChatUsers()
    } catch (err: any) {
      setGroupError(err?.message || 'Failed to delete group')
    }
  }

  function toggleGroupModel(name: string) {
    setGroupForm((f) => ({
      ...f,
      allowed_models: f.allowed_models.includes(name)
        ? f.allowed_models.filter((m) => m !== name)
        : [...f.allowed_models, name],
    }))
  }

  function toggleGroupMember(userId: string) {
    setGroupForm((f) => ({
      ...f,
      member_ids: f.member_ids.includes(userId)
        ? f.member_ids.filter((id) => id !== userId)
        : [...f.member_ids, userId],
    }))
  }

  function addDomain() {
    const d = domainInput.trim().toLowerCase().replace(/^@/, '')
    if (!d || groupForm.auto_domains.includes(d)) {
      setDomainInput('')
      return
    }
    setGroupForm((f) => ({ ...f, auto_domains: [...f.auto_domains, d] }))
    setDomainInput('')
  }

  async function handleWidgetLink(g: ChatGroupEntry) {
    try {
      const data = await chatGroups.widgetLink(g.id)
      const url = `${window.location.origin}${data.path}`
      setWidgetLink({ id: g.id, url })
      await navigator.clipboard.writeText(url).catch(() => undefined)
    } catch (err: any) {
      setGroupError(err?.message || 'Failed to create widget link')
    }
  }

  async function handleReapply() {
    try {
      const res = await chatGroups.reapplyDomains()
      setGroupError('')
      setGroupInfo($t('userMgmt.reapplyDone', { assigned: res.assigned, users: res.users }))
      await loadGroups()
      await loadChatUsers()
    } catch (err: any) {
      setGroupError(err?.message || 'Failed to reapply domains')
    }
  }

  const isProtected = (u: DashboardUser) => u.id === 'env-admin'

  return (
    <div className="rc-page">
      <div className="page-header">
        <h2>{$t('nav.users')}</h2>
        {activeTab === 'dashboard' && canWrite && (
          <button className="btn-primary" onClick={startCreate}>
            <Plus size={16} />
            {$t('common.create')}
          </button>
        )}
        {activeTab === 'groups' && canWrite && (
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn-secondary" onClick={handleReapply} title={$t('userMgmt.reapplyTitle')}>
              <RefreshCw size={16} />
              {$t('userMgmt.reapply')}
            </button>
            <button className="btn-primary" onClick={startGroupCreate}>
              <Plus size={16} />
              {$t('userMgmt.newGroup')}
            </button>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '1px solid var(--border)' }}>
        <button
          className={`btn btn-sm ${activeTab === 'dashboard' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('dashboard')}
          style={{ borderRadius: '6px 6px 0 0', borderBottom: 'none' }}
        >
          Dashboard Nutzer
        </button>
        <button
          className={`btn btn-sm ${activeTab === 'chat' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('chat')}
          style={{ borderRadius: '6px 6px 0 0', borderBottom: 'none' }}
        >
          <MessageSquare size={14} style={{ marginRight: 6 }} />
          Chat Nutzer
        </button>
        <button
          className={`btn btn-sm ${activeTab === 'groups' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('groups')}
          style={{ borderRadius: '6px 6px 0 0', borderBottom: 'none' }}
        >
          <Users2 size={14} style={{ marginRight: 6 }} />
          {$t('userMgmt.groupsTab')}
        </button>
      </div>

      {activeTab === 'dashboard' && (
        <>
          {error && <div className="alert-error">{error}</div>}

          {showForm && (
            <div className="rc-card" style={{ marginBottom: 20 }}>
              <div className="rc-section-header">
                <h3>{editing ? 'Edit User' : 'Create User'}</h3>
                <button className="btn-icon" onClick={() => setShowForm(false)} title="Close">
                  <X size={16} />
                </button>
              </div>
              <form onSubmit={handleSubmit} className="rc-field-grid">
                <div className="rc-field">
                  <label>Username</label>
                  <input
                    value={form.username}
                    onChange={(e) => setForm((f) => ({ ...f, username: e.target.value }))}
                    required
                  />
                </div>
                <div className="rc-field">
                  <label>Display Name</label>
                  <input
                    value={form.display_name}
                    onChange={(e) => setForm((f) => ({ ...f, display_name: e.target.value }))}
                    required
                  />
                </div>
                <div className="rc-field">
                  <label>Role</label>
                  <select
                    value={form.role}
                    onChange={(e) => setForm((f) => ({ ...f, role: e.target.value }))}
                  >
                    {ROLES.map((r) => (
                      <option key={r} value={r}>
                        {r}
                      </option>
                    ))}
                  </select>
                </div>
                {editing && (
                  <div className="rc-field" style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                    <input
                      type="checkbox"
                      id="is_active"
                      checked={form.is_active}
                      onChange={(e) => setForm((f) => ({ ...f, is_active: e.target.checked }))}
                    />
                    <label htmlFor="is_active" style={{ margin: 0 }}>Active</label>
                  </div>
                )}
                <div className="rc-field">
                  <label>{editing ? 'New Password (optional)' : 'Password'}</label>
                  <input
                    type="password"
                    value={form.password}
                    onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))}
                    required={!editing}
                  />
                </div>
                <div className="rc-field" style={{ gridColumn: '1 / -1', flexDirection: 'row', gap: 12 }}>
                  <button type="submit" className="btn-primary">
                    {editing ? $t('common.save') : $t('common.create')}
                  </button>
                  <button type="button" className="btn-secondary" onClick={() => setShowForm(false)}>
                    {$t('common.cancel')}
                  </button>
                </div>
              </form>
            </div>
          )}

          {loading ? (
            <div className="loading">{$t('common.loading')}</div>
          ) : (
            <div className="rc-card">
              <table className="data-table modern">
                <thead>
                  <tr>
                    <th>Username</th>
                    <th>Display Name</th>
                    <th>Role</th>
                    <th>Status</th>
                    <th>Last Login</th>
                    {canWrite && <th style={{ width: 140 }}>Actions</th>}
                  </tr>
                </thead>
                <tbody>
                  {userList.map((u) => (
                    <tr key={u.id} className={!u.is_active ? 'inactive' : ''}>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          {u.username}
                          {isProtected(u) && (
                            <span className="tooltip-wrap">
                              <ShieldAlert size={14} color="var(--warning)" />
                              <span className="tooltip-box">
                                <span className="tooltip-line">Environment admin cannot be modified</span>
                              </span>
                            </span>
                          )}
                        </div>
                      </td>
                      <td>{u.display_name}</td>
                      <td>
                        <span
                          className="role-badge"
                          style={{
                            background: `${ROLE_COLORS[u.role]}15`,
                            color: ROLE_COLORS[u.role],
                            border: `1px solid ${ROLE_COLORS[u.role]}40`,
                          }}
                        >
                          {u.role}
                        </span>
                      </td>
                      <td>
                        <span
                          className="status-dot"
                          style={{
                            background: u.is_active ? 'var(--success)' : 'var(--text-muted)',
                          }}
                        />
                        {u.is_active ? 'Active' : 'Inactive'}
                      </td>
                      <td>{u.last_login_at ? new Date(u.last_login_at).toLocaleString("de-DE") : '—'}</td>
                      {canWrite && (
                        <td className="actions">
                          <button
                            className="btn-icon"
                            onClick={() => startEdit(u)}
                            disabled={isProtected(u)}
                            title={isProtected(u) ? 'Environment admin cannot be edited' : 'Edit'}
                          >
                            <Pencil size={16} />
                          </button>
                          <button
                            className="btn-icon"
                            onClick={() => handleToggle(u)}
                            disabled={isProtected(u)}
                            title={isProtected(u) ? 'Environment admin cannot be disabled' : u.is_active ? 'Disable' : 'Enable'}
                          >
                            {u.is_active ? <Ban size={16} /> : <Check size={16} />}
                          </button>
                          <button
                            className="btn-icon danger"
                            onClick={() => handleDelete(u)}
                            disabled={isProtected(u)}
                            title={isProtected(u) ? 'Environment admin cannot be deleted' : 'Delete'}
                          >
                            <Trash2 size={16} />
                          </button>
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {activeTab === 'chat' && (
        <>
          {chatError && <div className="alert-error">{chatError}</div>}

          {showChatForm && (
            <div className="rc-card" style={{ marginBottom: 20 }}>
              <div className="rc-section-header">
                <h3>Edit Chat User</h3>
                <button className="btn-icon" onClick={() => setShowChatForm(false)} title="Close">
                  <X size={16} />
                </button>
              </div>
              <form onSubmit={handleChatSubmit} className="rc-field-grid">
                <div className="rc-field">
                  <label>Display Name</label>
                  <input
                    value={chatForm.display_name}
                    onChange={(e) => setChatForm((f) => ({ ...f, display_name: e.target.value }))}
                    required
                  />
                </div>
                <div className="rc-field">
                  <label>Role</label>
                  <select
                    value={chatForm.role}
                    onChange={(e) => setChatForm((f) => ({ ...f, role: e.target.value }))}
                  >
                    {CHAT_ROLES.map((r) => (
                      <option key={r} value={r}>
                        {r}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="rc-field">
                  <label>{$t('userMgmt.email')}</label>
                  <input
                    type="email"
                    value={chatForm.email}
                    onChange={(e) => setChatForm((f) => ({ ...f, email: e.target.value }))}
                    placeholder={$t('userMgmt.emailPlaceholder')}
                  />
                </div>
                <div className="rc-field" style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                  <input
                    type="checkbox"
                    id="chat_is_active"
                    checked={chatForm.is_active}
                    onChange={(e) => setChatForm((f) => ({ ...f, is_active: e.target.checked }))}
                  />
                  <label htmlFor="chat_is_active" style={{ margin: 0 }}>Active</label>
                </div>
                <div className="rc-field">
                  <label>New Password (optional)</label>
                  <input
                    type="password"
                    value={chatForm.password}
                    onChange={(e) => setChatForm((f) => ({ ...f, password: e.target.value }))}
                    placeholder="Leave blank to keep current"
                  />
                </div>
                <div className="rc-field" style={{ gridColumn: '1 / -1', flexDirection: 'row', gap: 12 }}>
                  <button type="submit" className="btn-primary">
                    {$t('common.save')}
                  </button>
                  <button type="button" className="btn-secondary" onClick={() => setShowChatForm(false)}>
                    {$t('common.cancel')}
                  </button>
                </div>
              </form>
            </div>
          )}

          {chatLoading ? (
            <div className="loading">{$t('common.loading')}</div>
          ) : (
            <div className="rc-card">
              <table className="data-table modern">
                <thead>
                  <tr>
                    <th style={{ cursor: 'pointer' }} onClick={() => handleChatSort('username')}>
                      Username {chatSort.column === 'username' && <ArrowUpDown size={12} style={{ verticalAlign: 'middle', marginLeft: 4 }} />}
                    </th>
                    <th style={{ cursor: 'pointer' }} onClick={() => handleChatSort('display_name')}>
                      Display Name {chatSort.column === 'display_name' && <ArrowUpDown size={12} style={{ verticalAlign: 'middle', marginLeft: 4 }} />}
                    </th>
                    <th style={{ cursor: 'pointer' }} onClick={() => handleChatSort('role')}>
                      Role {chatSort.column === 'role' && <ArrowUpDown size={12} style={{ verticalAlign: 'middle', marginLeft: 4 }} />}
                    </th>
                    <th style={{ cursor: 'pointer' }} onClick={() => handleChatSort('source')}>
                      Source {chatSort.column === 'source' && <ArrowUpDown size={12} style={{ verticalAlign: 'middle', marginLeft: 4 }} />}
                    </th>
                    <th>{$t('userMgmt.groupsCol')}</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => handleChatSort('is_active')}>
                      Status {chatSort.column === 'is_active' && <ArrowUpDown size={12} style={{ verticalAlign: 'middle', marginLeft: 4 }} />}
                    </th>
                    <th style={{ cursor: 'pointer' }} onClick={() => handleChatSort('last_login_at')}>
                      Last Login {chatSort.column === 'last_login_at' && <ArrowUpDown size={12} style={{ verticalAlign: 'middle', marginLeft: 4 }} />}
                    </th>
                    {canWrite && <th style={{ width: 100 }}>Actions</th>}
                  </tr>
                </thead>
                <tbody>
                  {sortedChatUsers.map((u) => (
                    <tr key={u.id} className={!u.is_active ? 'inactive' : ''}>
                      <td>{u.username}</td>
                      <td>{u.display_name}</td>
                      <td>
                        <span className="role-badge" style={{ background: 'var(--accent-light)', color: 'var(--accent)', border: '1px solid var(--accent-border)' }}>
                          {u.role}
                        </span>
                      </td>
                      <td>
                        <span className="source-badge" style={{ fontSize: 12, padding: '2px 8px', borderRadius: 4, background: u.source === 'dashboard' ? 'var(--info-light)' : 'var(--success-light)', color: u.source === 'dashboard' ? 'var(--info)' : 'var(--success)' }}>
                          {u.source === 'dashboard' ? 'Dashboard' : 'Public'}
                        </span>
                      </td>
                      <td>
                        <div className="group-chip-area" style={{ gap: 4 }}>
                          {(u.groups || []).map((g) => (
                            <span
                              key={g.id}
                              className="model-chip"
                              title={g.assigned_via === 'domain' ? 'via Domain' : undefined}
                              style={{ borderColor: `${g.color || 'var(--accent)'}60`, color: g.color || 'var(--accent)' }}
                            >
                              {g.name}
                            </span>
                          ))}
                          {(u.groups || []).length === 0 && <span className="text-muted">—</span>}
                        </div>
                      </td>
                      <td>
                        <span className="status-dot" style={{ background: u.is_active ? 'var(--success)' : 'var(--text-muted)' }} />
                        {u.is_active ? 'Active' : 'Inactive'}
                      </td>
                      <td>{u.last_login_at ? new Date(u.last_login_at).toLocaleString("de-DE") : '—'}</td>
                      {canWrite && (
                        <td className="actions">
                          <button className="btn-icon" onClick={() => startChatEdit(u)} title="Edit">
                            <Pencil size={16} />
                          </button>
                          <button className="btn-icon danger" onClick={() => handleChatDelete(u)} title="Delete">
                            <Trash2 size={16} />
                          </button>
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
      {activeTab === 'groups' && (
        <>
          {groupError && <div className="alert-error">{groupError}</div>}
          {groupInfo && <div className="info-box" style={{ marginBottom: 12 }}>{groupInfo}</div>}

          {showGroupForm && (
            <div className="rc-card" style={{ marginBottom: 20 }}>
              <div className="rc-section-header">
                <h3>{groupEditing ? $t('userMgmt.editGroup') : $t('userMgmt.newGroup')}</h3>
                <button className="btn-icon" onClick={() => setShowGroupForm(false)} title="Close">
                  <X size={16} />
                </button>
              </div>
              <form onSubmit={handleGroupSubmit} className="rc-field-grid">
                <div className="rc-field">
                  <label>{$t('userMgmt.groupName')}</label>
                  <input
                    value={groupForm.name}
                    onChange={(e) => setGroupForm((f) => ({ ...f, name: e.target.value }))}
                    required
                  />
                </div>
                <div className="rc-field">
                  <label>{$t('userMgmt.groupColor')}</label>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    {GROUP_COLORS.map((c) => (
                      <button
                        key={c}
                        type="button"
                        onClick={() => setGroupForm((f) => ({ ...f, color: c }))}
                        style={{
                          width: 24, height: 24, borderRadius: '50%', background: c, cursor: 'pointer',
                          border: groupForm.color === c ? '2px solid var(--text)' : '2px solid transparent',
                        }}
                      />
                    ))}
                  </div>
                </div>
                <div className="rc-field" style={{ gridColumn: '1 / -1' }}>
                  <label>{$t('userMgmt.groupDesc')}</label>
                  <input
                    value={groupForm.description}
                    onChange={(e) => setGroupForm((f) => ({ ...f, description: e.target.value }))}
                  />
                </div>

                <div className="rc-field" style={{ gridColumn: '1 / -1' }}>
                  <label>{$t('userMgmt.allowedModels')}</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <input
                      type="checkbox"
                      id="group_unrestricted"
                      checked={groupForm.unrestricted}
                      onChange={(e) => setGroupForm((f) => ({ ...f, unrestricted: e.target.checked }))}
                    />
                    <label htmlFor="group_unrestricted" style={{ margin: 0 }}>{$t('userMgmt.unrestricted')}</label>
                  </div>
                  {!groupForm.unrestricted && (
                    <div className="group-chip-area">
                      {modelOptions.map((name) => (
                        <button
                          key={name}
                          type="button"
                          className={`model-chip ${groupForm.allowed_models.includes(name) ? 'active' : ''}`}
                          onClick={() => toggleGroupModel(name)}
                        >
                          {name}
                        </button>
                      ))}
                      {modelOptions.length === 0 && <span className="text-muted">{$t('userMgmt.noModels')}</span>}
                    </div>
                  )}
                </div>

                <div className="rc-field" style={{ gridColumn: '1 / -1' }}>
                  <label>{$t('userMgmt.autoDomains')}</label>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <input
                      value={domainInput}
                      onChange={(e) => setDomainInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addDomain() } }}
                      placeholder="firma.de"
                      style={{ flex: 1 }}
                    />
                    <button type="button" className="btn-secondary" onClick={addDomain}>
                      <Plus size={14} />
                    </button>
                  </div>
                  <div className="group-chip-area" style={{ marginTop: 8 }}>
                    {groupForm.auto_domains.map((d) => (
                      <span key={d} className="domain-chip">
                        @{d}
                        <button type="button" onClick={() => setGroupForm((f) => ({ ...f, auto_domains: f.auto_domains.filter((x) => x !== d) }))}>
                          <X size={12} />
                        </button>
                      </span>
                    ))}
                  </div>
                </div>

                <div className="rc-field" style={{ gridColumn: '1 / -1' }}>
                  <label>{$t('userMgmt.members')}</label>
                  <div className="group-member-list">
                    {chatUserList.map((u) => (
                      <label key={u.id} className="group-member-row">
                        <input
                          type="checkbox"
                          checked={groupForm.member_ids.includes(u.id)}
                          onChange={() => toggleGroupMember(u.id)}
                        />
                        <span>{u.display_name}</span>
                        <span className="text-muted">@{u.username}</span>
                      </label>
                    ))}
                    {chatUserList.length === 0 && <span className="text-muted">{$t('userMgmt.noChatUsers')}</span>}
                  </div>
                </div>

                <div className="rc-field" style={{ gridColumn: '1 / -1', flexDirection: 'row', gap: 12 }}>
                  <button type="submit" className="btn-primary">
                    {$t('common.save')}
                  </button>
                  <button type="button" className="btn-secondary" onClick={() => setShowGroupForm(false)}>
                    {$t('common.cancel')}
                  </button>
                </div>
              </form>
            </div>
          )}

          {groupLoading ? (
            <div className="loading">{$t('common.loading')}</div>
          ) : groupList.length === 0 ? (
            <div className="rc-card" style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
              <Users2 size={32} style={{ opacity: 0.4, marginBottom: 8 }} />
              <div>{$t('userMgmt.emptyGroups')}</div>
            </div>
          ) : (
            <div className="group-grid">
              {groupList.map((g) => (
                <div key={g.id} className="rc-card group-card">
                  <div className="rc-section-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span className="group-dot" style={{ background: g.color || 'var(--accent)' }} />
                      <h3 style={{ margin: 0 }}>{g.name}</h3>
                    </div>
                    {canWrite && (
                      <div style={{ display: 'flex', gap: 4 }}>
                        <button className="btn-icon" onClick={() => handleWidgetLink(g)} title={$t('userMgmt.widgetLink')}>
                          <Link2 size={15} />
                        </button>
                        <button className="btn-icon" onClick={() => void startGroupEdit(g)} title="Edit">
                          <Pencil size={15} />
                        </button>
                        <button className="btn-icon danger" onClick={() => handleGroupDelete(g)} title="Delete">
                          <Trash2 size={15} />
                        </button>
                      </div>
                    )}
                  </div>
                  {g.description && <p className="text-muted" style={{ margin: '4px 0 8px', fontSize: 13 }}>{g.description}</p>}
                  <div className="group-card-meta">
                    <span className="member-count">{g.member_count} {$t('userMgmt.memberCount')}</span>
                  </div>
                  <div className="group-chip-area">
                    {g.allowed_models === null ? (
                      <span className="model-chip active">{$t('userMgmt.unrestricted')}</span>
                    ) : g.allowed_models.length === 0 ? (
                      <span className="model-chip" style={{ opacity: 0.6 }}>{$t('userMgmt.noAccess')}</span>
                    ) : (
                      g.allowed_models.map((m) => <span key={m} className="model-chip">{m}</span>)
                    )}
                    {g.auto_domains.map((d) => <span key={d} className="domain-chip">@{d}</span>)}
                  </div>
                  {widgetLink?.id === g.id && (
                    <div className="widget-link-box">
                      <code>{widgetLink.url}</code>
                      <button className="btn-icon" onClick={() => void navigator.clipboard.writeText(widgetLink.url).catch(() => undefined)} title="Copy">
                        <Copy size={14} />
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
      <ConfirmModal
        open={pendingConfirm !== null}
        title={$t('common.delete')}
        message={pendingConfirm?.message || ''}
        onConfirm={() => {
          const action = pendingConfirm?.action
          setPendingConfirm(null)
          action?.()
        }}
        onCancel={() => setPendingConfirm(null)}
      />
    </div>
  )
}

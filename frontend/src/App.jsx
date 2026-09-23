import { useCallback, useEffect, useState } from 'react'
import {
  Activity, BarChart3, Brain, Database, FlaskConical, Gauge, LayoutDashboard, LineChart, Moon, ShieldCheck, Sun,
} from 'lucide-react'
import { api } from './api'
import { Badge, Toasts } from './components/ui'
import Overview from './pages/Overview'
import DataPage from './pages/Data'
import Baselines from './pages/Baselines'
import Sanity from './pages/Sanity'
import Train from './pages/Train'
import Results from './pages/Results'
import Robustness from './pages/Robustness'

const NAV = [
  { sec: 'Project' },
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { sec: 'Pipeline' },
  { id: 'data', label: 'Data & universe', icon: Database, step: '1' },
  { id: 'baselines', label: 'Costs & baselines', icon: BarChart3, step: '2' },
  { id: 'sanity', label: 'Sanity tests', icon: ShieldCheck, step: '3' },
  { id: 'train', label: 'Train agent', icon: Brain, step: '4' },
  { sec: 'Evaluation' },
  { id: 'results', label: 'Results', icon: LineChart, step: '5' },
  { id: 'robustness', label: 'Robustness', icon: Gauge, step: '6' },
]

function parseHash() {
  const h = window.location.hash.replace(/^#\/?/, '')
  const [page, q] = h.split('?')
  const params = Object.fromEntries(new URLSearchParams(q || ''))
  return { page: page || 'overview', params }
}

export default function App() {
  const [route, setRoute] = useState(parseHash())
  const [meta, setMeta] = useState(null)
  const [overview, setOverview] = useState(null)
  const [backendDown, setBackendDown] = useState(false)
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('theme') || 'auto' } catch { return 'auto' }
  })

  useEffect(() => {
    const onHash = () => setRoute(parseHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'auto') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
    try { localStorage.setItem('theme', theme) } catch { /* ignore */ }
    window.dispatchEvent(new Event('themechange'))
  }, [theme])

  const refresh = useCallback(async () => {
    try {
      const [m, o] = await Promise.all([meta ? Promise.resolve(meta) : api.get('/meta'), api.get('/overview')])
      setMeta(m); setOverview(o); setBackendDown(false)
    } catch (e) {
      setBackendDown(true)
    }
  }, [meta])

  useEffect(() => { refresh() }, []) // eslint-disable-line
  useEffect(() => {
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [refresh])

  const go = (page, params) => {
    const q = params ? '?' + new URLSearchParams(params).toString() : ''
    window.location.hash = `/${page}${q}`
  }

  const isDark = theme === 'dark' || (theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches)
  const ds = overview?.dataset
  const jobs = overview?.active_jobs || []
  const props = { meta, overview, refresh, go, params: route.params }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo"><Activity size={19} /></div>
          <div>
            <div className="brand-title">RL Rebalancing Lab</div>
            <div className="brand-sub">Indian ETF portfolio · RL</div>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((n, i) => n.sec ? <div key={i} className="nav-sec">{n.sec}</div> : (
            <button key={n.id} className={route.page === n.id ? 'active' : ''} onClick={() => go(n.id)}>
              <n.icon size={17} />{n.label}{n.step && <span className="step">{n.step}</span>}
            </button>
          ))}
        </nav>
        <div className="side-foot">
          <div style={{ fontWeight: 600, color: 'var(--text-2)' }}>Team</div>
          <div className="team">
            <span>Surbhi Agarwal · J002</span>
            <span>Urja Innani · J018</span>
            <span>Taha Lokhandwala · J024</span>
          </div>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          {ds ? (
            <>
              <Badge tone={ds.source === 'synthetic' ? 'warn' : 'good'}>
                <Database size={12} /> {ds.source === 'synthetic' ? 'Synthetic data' : ds.source === 'csv' ? 'CSV data' : 'Yahoo Finance'}
              </Badge>
              <span className="small t2">{ds.tickers?.length} ETFs + cash · {ds.freq === 'W' ? 'weekly' : 'daily'} · {ds.start} → {ds.end}</span>
              <span className="mono muted xs">#{ds.sha}</span>
            </>
          ) : <span className="small muted">No dataset prepared yet</span>}
          <div className="spacer" />
          {jobs.length > 0 && (
            <button className="btn sm" onClick={() => go('train')}>
              <FlaskConical size={14} className="spin" style={{ animationDuration: '2.5s' }} />
              {jobs.length} job{jobs.length > 1 ? 's' : ''} · {Math.round(((jobs.find((j) => j.status === 'running') || jobs[0]).progress || 0) * 100)}%
            </button>
          )}
          <button className="btn sm ghost" onClick={() => setTheme(isDark ? 'light' : 'dark')} title="Toggle theme">
            {isDark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </header>
        <main className="content">
          {backendDown && (
            <div className="alert bad">
              Cannot reach the backend at <span className="mono">localhost:8000</span>. Start it with <span className="mono">run.bat</span> / <span className="mono">./run.sh</span> (or <span className="mono">uvicorn app.main:app</span> inside <span className="mono">backend/</span>).
            </div>
          )}
          {meta && route.page === 'overview' && <Overview {...props} />}
          {meta && route.page === 'data' && <DataPage {...props} />}
          {meta && route.page === 'baselines' && <Baselines {...props} />}
          {meta && route.page === 'sanity' && <Sanity {...props} />}
          {meta && route.page === 'train' && <Train {...props} />}
          {meta && route.page === 'results' && <Results {...props} />}
          {meta && route.page === 'robustness' && <Robustness {...props} />}
        </main>
      </div>
      <Toasts />
    </div>
  )
}

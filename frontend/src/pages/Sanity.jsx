import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, CircleDashed, Play, ShieldCheck, Timer, XCircle } from 'lucide-react'
import { api } from '../api'
import { Alert, Badge, Button, Card, Progress, toast } from '../components/ui'

const TESTS = [
  { id: 'bh', n: 1, name: 'Zero-cost buy-and-hold matches a hand calculation', why: 'Proves the accounting of drift and compounding is right.', speed: 'fast' },
  { id: 'rotation', n: 2, name: 'Forced rotation pays exactly turnover × cost', why: 'Proves costs are charged on the right notional, before the market move.', speed: 'fast' },
  { id: 'one_over_n', n: 3, name: '1/n action reproduces the equal-weight portfolio', why: 'The env and an independent implementation agree to machine precision.', speed: 'fast' },
  { id: 'shuffled', n: 4, name: 'Shuffled returns must NOT be profitable', why: 'Destroy time structure → any "edge" means leakage or a bug.', speed: 'slow' },
  { id: 'lookahead', n: 5, name: 'Injected look-ahead must improve results sharply', why: 'Leak next-bar returns → agent must exploit it, proving the learner works.', speed: 'slow' },
]

export default function Sanity({ refresh }) {
  const [results, setResults] = useState({})
  const [busyFast, setBusyFast] = useState(false)
  const [job, setJob] = useState(null)
  const timer = useRef()

  const absorb = (list) => setResults((r) => { const n = { ...r }; (list || []).forEach((x) => { n[x.id] = x }); return n })
  useEffect(() => {
    api.get('/sanity').then((s) => { absorb(s.fast); absorb(s.slow) }).catch(() => {})
    return () => clearInterval(timer.current)
  }, [])

  const runFast = async () => {
    setBusyFast(true)
    try { const r = await api.post('/sanity/fast'); absorb(r.results); refresh(); toast('Fast sanity tests finished') } catch (e) { toast(e.message, 'err') }
    setBusyFast(false)
  }
  const runSlow = async () => {
    try {
      const j = await api.post('/sanity/slow')
      setJob(j)
      clearInterval(timer.current)
      timer.current = setInterval(async () => {
        const s = await api.get(`/jobs/${j.id}`)
        setJob(s)
        if (s.result?.results) absorb(s.result.results)
        if (['done', 'failed', 'cancelled'].includes(s.status)) {
          clearInterval(timer.current); refresh()
          if (s.status === 'failed') toast(s.error, 'err')
        }
      }, 1000)
    } catch (e) { toast(e.message, 'err') }
  }
  const passed = TESTS.filter((t) => results[t.id]?.passed).length
  const ran = TESTS.filter((t) => results[t.id]).length
  const running = job && ['queued', 'running'].includes(job.status)

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Sanity tests</h1>
          <p>Five checks the environment must pass before any training result is trusted. Tests 1–3 are exact accounting identities; tests 4–5 train a small PPO agent (~20k steps each).</p>
        </div>
        <div className="row">
          <Button icon={<Play size={15} />} onClick={runFast} loading={busyFast}>Run tests 1–3</Button>
          <Button kind="primary" icon={<Timer size={15} />} onClick={runSlow} loading={running}>Run tests 4–5</Button>
        </div>
      </div>

      <div className="row">
        <Badge tone={ran === 5 && passed === 5 ? 'good' : ran && passed < ran ? 'bad' : ''}><ShieldCheck size={12} /> {passed}/{ran || 0} passed · {5 - ran} not run</Badge>
        {running && <div style={{ flex: 1, maxWidth: 420 }}><Progress value={job.progress} /></div>}
        {running && <span className="small muted">{job.message}</span>}
      </div>

      <div className="grid g2">
        {TESTS.map((t) => {
          const r = results[t.id]
          const state = r ? (r.passed ? 'pass' : 'fail') : ''
          return (
            <Card key={t.id}>
              <div className="test-card">
                <div className={`test-icon ${state}`}>
                  {r ? (r.passed ? <CheckCircle2 size={20} /> : <XCircle size={20} />) : <CircleDashed size={20} />}
                </div>
                <div className="col" style={{ gap: 6, flex: 1, minWidth: 0 }}>
                  <div className="row between">
                    <h3>{t.n}. {t.name}</h3>
                    <Badge tone={r ? (r.passed ? 'good' : 'bad') : ''}>{r ? (r.passed ? 'PASS' : 'FAIL') : t.speed === 'fast' ? 'instant' : '~1 min'}</Badge>
                  </div>
                  <div className="small muted">{t.why}</div>
                  {r && <div className="small t2">{r.detail}</div>}
                </div>
              </div>
            </Card>
          )
        })}
        <Card title="Leakage checklist" sub="Enforced by construction in the code">
          <ul className="small t2" style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <li>Indicators & optimiser targets use data up to t−1 only (execution at next bar).</li>
            <li>Return / volatility scalers are fit on the training block only.</li>
            <li>Chronological splits with an embargo; test block unlocked explicitly, and each unlock is logged.</li>
            <li>A unit test corrupts every future return and asserts the observation is unchanged.</li>
          </ul>
        </Card>
      </div>
      {job?.status === 'failed' && <Alert tone="bad">{job.error}</Alert>}
    </>
  )
}

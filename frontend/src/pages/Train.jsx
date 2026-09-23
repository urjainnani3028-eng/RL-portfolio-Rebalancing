import { useEffect, useRef, useState } from 'react'
import { Ban, Brain, ExternalLink, Rocket, Zap } from 'lucide-react'
import { api, num, pct } from '../api'
import { Alert, Badge, Button, Card, Empty, Field, NumInput, Progress, Segmented, toast } from '../components/ui'
import { StepChart } from '../components/charts'

const PRESETS = {
  quick: { label: 'Quick demo', note: '20k steps × 3 seeds', patch: { timesteps: 20000, seeds: 3, eval_every: 5000 } },
  study: { label: 'Full study', note: '100k steps × 10 seeds', patch: { timesteps: 100000, seeds: 10, eval_every: 10000 } },
}

export default function Train({ meta, overview, go, refresh }) {
  const [agent, setAgent] = useState(meta.defaults.agent)
  const [jobs, setJobs] = useState([])
  const [sel, setSel] = useState(null)
  const [job, setJob] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const t = useRef()

  const set = (k, v) => setAgent((a) => ({ ...a, [k]: v }))
  const setR = (k, v) => setAgent((a) => ({ ...a, reward: { ...a.reward, [k]: v } }))

  const loadJobs = async () => {
    const js = (await api.get('/jobs')).filter((j) => j.kind !== 'sanity')
    setJobs(js)
    return js
  }
  useEffect(() => {
    loadJobs().then((js) => { if (!sel && js.length) setSel(js[0].id) })
    const iv = setInterval(loadJobs, 3000)
    return () => clearInterval(iv)
  }, []) // eslint-disable-line

  useEffect(() => {
    clearInterval(t.current)
    if (!sel) return
    const poll = async () => {
      try {
        const j = await api.get(`/jobs/${sel}`)
        setJob(j)
        if (['done', 'failed', 'cancelled'].includes(j.status)) { clearInterval(t.current); refresh() }
      } catch { clearInterval(t.current) }
    }
    poll()
    t.current = setInterval(poll, 1000)
    return () => clearInterval(t.current)
  }, [sel]) // eslint-disable-line

  const launch = async () => {
    setSubmitting(true)
    try {
      const j = await api.post('/train', agent)
      toast('Training job queued')
      await loadJobs(); setSel(j.id)
    } catch (e) { toast(e.message, 'err') }
    setSubmitting(false)
  }

  const hybrid = agent.mode === 'hybrid'
  const algoOk = (a) => a.modes.includes(agent.mode)
  const estMin = (agent.timesteps * agent.seeds) / (['SAC', 'TD3'].includes(agent.algo) ? 150 : 900) / 60
  const bar = overview?.bar_to_beat

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Train agent</h1>
          <p>Configure the MDP and learner, then launch a multi-seed job. The checkpoint with the best validation Sharpe is kept for each seed; the test block stays locked.</p>
        </div>
        <div className="row">
          {Object.entries(PRESETS).map(([k, p]) => (
            <Button key={k} size="sm" onClick={() => setAgent((a) => ({ ...a, ...p.patch }))} title={p.note}>{p.label} <span className="muted xs">{p.note}</span></Button>
          ))}
        </div>
      </div>
      {!overview?.dataset && <Alert tone="warn">Prepare a dataset first (Data page).</Alert>}

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1.05fr) minmax(0, 1fr)' }}>
        <div className="col" style={{ gap: 16 }}>
          <Card title="1 · Action design" sub="Hybrid is the project's headline novelty">
            <div className="grid g2" style={{ gap: 8 }}>
              <button className={`choice ${hybrid ? 'on' : ''}`} onClick={() => set('mode', 'hybrid')}>
                <b>Hybrid: optimiser + RL timing</b>
                <small>Optimiser proposes target w*; agent chooses a ∈ [0,1]: w ← w + a(w* − w). Small, interpretable, hard to reward-hack.</small>
              </button>
              <button className={`choice ${!hybrid ? 'on' : ''}`} onClick={() => { set('mode', 'direct'); if (agent.algo === 'DQN') set('algo', 'PPO') }}>
                <b>Direct: full target weights</b>
                <small>Agent outputs logits → softmax on the simplex, capped. The classic (harder) formulation.</small>
              </button>
            </div>
            {hybrid && (
              <div style={{ marginTop: 12 }}>
                <Field label="Target proposed by the classical optimiser">
                  <select value={agent.target} onChange={(e) => set('target', e.target.value)}>
                    {meta.targets.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
                  </select>
                </Field>
              </div>
            )}
          </Card>

          <Card title="2 · Algorithm & network" sub="All from Stable-Baselines3">
            <div className="grid g5" style={{ gap: 8 }}>
              {meta.algos.map((a) => (
                <button key={a.id} className={`choice ${agent.algo === a.id ? 'on' : ''} ${algoOk(a) ? '' : 'disabled'}`} onClick={() => set('algo', a.id)} title={a.note}>
                  <b>{a.label}</b><small>{a.note}</small>
                </button>
              ))}
            </div>
            <div className="grid g3" style={{ gap: 8, marginTop: 10 }}>
              {meta.archs.map((a) => (
                <button key={a.id} className={`choice ${agent.arch === a.id ? 'on' : ''}`} onClick={() => set('arch', a.id)}>
                  <b>{a.label}</b><small>{a.note}</small>
                </button>
              ))}
            </div>
          </Card>

          <Card title="3 · Reward" sub="log net return − penalties (all scaled)">
            <div className="grid g2" style={{ gap: 12 }}>
              <Field label="Risk penalty">
                <select value={agent.reward.risk} onChange={(e) => setR('risk', e.target.value)}>
                  <option value="downside">Downside (λ · min(r,0)²)</option>
                  <option value="variance">Variance (λ · r²)</option>
                  <option value="drawdown">Drawdown increase (λ · Δdd⁺)</option>
                  <option value="none">None (pure log-growth)</option>
                </select>
              </Field>
              <Field label={`Risk aversion λ = ${agent.reward.risk_lambda}`}>
                <input type="range" min={0} max={30} step={0.5} value={agent.reward.risk_lambda} onChange={(e) => setR('risk_lambda', +e.target.value)} />
              </Field>
              <Field label="Concentration penalty κ" hint="× (Herfindahl − 1/N)⁺ — guards single-asset bets"><NumInput value={agent.reward.concentration_kappa} onChange={(v) => setR('concentration_kappa', v)} /></Field>
              <Field label="Extra turnover penalty" hint="On top of real costs; 0 = costs only"><NumInput value={agent.reward.turnover_penalty} onChange={(v) => setR('turnover_penalty', v)} /></Field>
            </div>
          </Card>

          <Card title="4 · Training protocol">
            <div className="grid g3" style={{ gap: 12 }}>
              <Field label="Timesteps / seed"><NumInput value={agent.timesteps} onChange={(v) => set('timesteps', Math.round(v))} step={1000} /></Field>
              <Field label="Seeds" hint="≥10 for the final comparison"><NumInput value={agent.seeds} onChange={(v) => set('seeds', Math.round(v))} /></Field>
              <Field label="Learning rate"><NumInput value={agent.learning_rate} onChange={(v) => set('learning_rate', v)} /></Field>
              <Field label="Episode length (bars)" hint="Random start in train block"><NumInput value={agent.episode_len} onChange={(v) => set('episode_len', Math.round(v))} /></Field>
              <Field label="Validate every (steps)" hint="Keeps best checkpoint"><NumInput value={agent.eval_every} onChange={(v) => set('eval_every', Math.round(v))} /></Field>
              <Field label="No-trade band" hint="Skip trades with turnover below"><NumInput value={agent.no_trade_band} onChange={(v) => set('no_trade_band', v)} /></Field>
            </div>
            <label className="checkline" style={{ marginTop: 12 }}>
              <input type="checkbox" checked={agent.walk_forward} onChange={(e) => set('walk_forward', e.target.checked)} />
              Purged walk-forward: retrain every year on all prior data (embargoed), evaluate on the next year, stitch the OOS curve
            </label>
            <div className="row" style={{ marginTop: 14 }}>
              <Field label="Run name (optional)"><input type="text" value={agent.name} placeholder="auto" onChange={(e) => set('name', e.target.value)} /></Field>
              <div style={{ flex: 1 }} />
              <div className="col" style={{ alignItems: 'flex-end', gap: 4 }}>
                <span className="xs muted">≈ {estMin < 1 ? '<1' : Math.round(estMin)} min on a typical laptop{agent.walk_forward ? ' × number of folds' : ''}</span>
                <Button kind="primary" icon={<Rocket size={15} />} onClick={launch} loading={submitting} disabled={!overview?.dataset}>Launch training</Button>
              </div>
            </div>
          </Card>
        </div>

        <div className="col" style={{ gap: 16 }}>
          <Card title="Job queue" sub="One job runs at a time; others wait">
            {jobs.length === 0 ? <Empty icon={<Brain size={34} />} title="No jobs yet">Launch a run to see live progress here.</Empty> : (
              <div className="col" style={{ gap: 6 }}>
                {jobs.slice(0, 8).map((j) => (
                  <div key={j.id} className={`check-item`} style={{ background: sel === j.id ? 'var(--surface-2)' : undefined }} onClick={() => setSel(j.id)}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div className="row between"><b className="small" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 330 }}>{j.title}</b>
                        <Badge tone={j.status === 'done' ? 'good' : j.status === 'failed' ? 'bad' : j.status === 'running' ? 'accent' : ''}>{j.status}</Badge></div>
                      <div style={{ marginTop: 6 }}><Progress value={j.progress} /></div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>

          {job && (
            <Card title={job.title} sub={job.message}
              right={<>
                {['queued', 'running'].includes(job.status) && <Button size="sm" kind="danger" icon={<Ban size={14} />} onClick={() => api.post(`/jobs/${job.id}/cancel`)}>Cancel</Button>}
                {job.result?.run_id && <Button size="sm" kind="primary" icon={<ExternalLink size={14} />} onClick={() => go('results', { run: job.result.run_id })}>Open results</Button>}
              </>}>
              <div className="col" style={{ gap: 14 }}>
                <div className="row"><div style={{ flex: 1 }}><Progress value={job.progress} /></div><span className="small num">{Math.round(job.progress * 100)}%</span></div>
                {job.error && <Alert tone="bad">{job.error}</Alert>}
                {job.evals?.length > 0 && (
                  <div>
                    <h3 style={{ marginBottom: 6 }}>Validation Sharpe during training</h3>
                    <StepChart points={job.evals} yKey="sharpe" height={200} refY={bar} refLabel={bar != null ? `EW quarterly ${num(bar)}` : ''} />
                  </div>
                )}
                {job.series?.length > 0 && (
                  <div>
                    <h3 style={{ marginBottom: 6 }}>Mean episode reward (training)</h3>
                    <StepChart points={job.series} yKey="ep_reward" height={180} />
                  </div>
                )}
                {job.evals?.length > 0 && (
                  <div className="xs muted">Last validation: Sharpe {num(job.evals.at(-1).sharpe)}, CAGR {pct(job.evals.at(-1).cagr)}, max DD {pct(job.evals.at(-1).max_drawdown)}, turnover {pct(job.evals.at(-1).turnover_ann, 0)}/yr</div>
                )}
                {job.logs?.length > 0 && <div className="log">{job.logs.join('\n')}</div>}
              </div>
            </Card>
          )}
          <Card className="tint" title="Reward-hacking watch-list">
            <div className="small t2 col" style={{ gap: 4 }}>
              <span><Zap size={12} /> Cash parking — check average cash weight in Results.</span>
              <span><Zap size={12} /> Frozen weights (turnover → 0) — check the action / turnover chart.</span>
              <span><Zap size={12} /> Single-asset concentration — capped at the max weight; Herfindahl is tracked.</span>
            </div>
          </Card>
        </div>
      </div>
    </>
  )
}

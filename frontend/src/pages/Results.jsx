import { useEffect, useMemo, useState } from 'react'
import { FlaskConical, Lock, LockOpen, Trash2 } from 'lucide-react'
import { SERIES, api, num, pct, short } from '../api'
import { Alert, Badge, Button, Card, DataTable, Empty, Modal, Segmented, Stat, toast } from '../components/ui'
import { TimeChart, WeightsChart } from '../components/charts'
import { SEGMENTS, metricCols } from './Baselines'

const median = (xs) => { const s = xs.filter((x) => x != null).sort((a, b) => a - b); if (!s.length) return null; const m = Math.floor(s.length / 2); return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2 }

export default function Results({ params, go, refresh }) {
  const [runs, setRuns] = useState([])
  const [runId, setRunId] = useState(params.run || null)
  const [seg, setSeg] = useState('val')
  const [res, setRes] = useState(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')
  const [confirm, setConfirm] = useState(false)
  const [delConfirm, setDelConfirm] = useState(false)
  const [seed, setSeed] = useState(null)
  const [shownBase, setShownBase] = useState(new Set(['ew_quarterly', 'market', 'tgt_monthly']))

  useEffect(() => {
    api.get('/runs').then((r) => { setRuns(r); if (!runId && r.length) setRunId(r[0].id) })
  }, []) // eslint-disable-line
  useEffect(() => { if (params.run) setRunId(params.run) }, [params.run])

  const run = runs.find((r) => r.id === runId)
  const isWF = run?.kind === 'walkforward'
  const load = async () => {
    if (!runId) return
    setLoading(true); setErr('')
    try {
      const r = await api.get(`/runs/${runId}?segment=${isWF ? 'oos' : seg}`)
      setRes(r)
      if (r.seeds?.length) setSeed((s) => (r.seeds.find((x) => x.seed === s) ? s : r.seeds[0].seed))
    } catch (e) { setErr(e.message); setRes(null) }
    setLoading(false)
  }
  useEffect(() => { load() }, [runId, seg, isWF]) // eslint-disable-line

  const unlockTest = async () => {
    setConfirm(false); setLoading(true)
    try { const r = await api.post(`/runs/${runId}/test`); setRes(r); toast('Test block evaluated (logged)'); setRuns(await api.get('/runs')) } catch (e) { toast(e.message, 'err') }
    setLoading(false)
  }
  const del = async () => {
    setDelConfirm(false)
    await api.del(`/runs/${runId}`)
    const r = await api.get('/runs'); setRuns(r); setRunId(r[0]?.id || null); setRes(null); refresh(); toast('Run deleted')
  }

  const seeds = res?.seeds || []
  const curves = res?.curves || {}
  const baselines = res?.baselines || []
  const seedKeys = Object.keys(curves)
  const first = seedKeys.length ? curves[seedKeys[0]] : null

  const equity = useMemo(() => {
    if (!first) return []
    return first.dates.map((d, i) => {
      const vals = seedKeys.map((k) => curves[k].value[i])
      const r = { date: d, rl: median(vals), _band: [Math.min(...vals), Math.max(...vals)] }
      baselines.forEach((b) => { r[b.id] = b.curve.value[i] })
      return r
    })
  }, [res])
  const dd = useMemo(() => {
    if (!first) return []
    return first.dates.map((d, i) => {
      const r = { date: d, rl: median(seedKeys.map((k) => curves[k].drawdown[i])) }
      baselines.forEach((b) => { r[b.id] = b.curve.drawdown[i] })
      return r
    })
  }, [res])
  const actionData = useMemo(() => {
    const c = curves[seed]
    if (!c) return []
    return c.dates.map((d, i) => ({ date: d, a: c.a?.[i], turnover: c.turnover[i] }))
  }, [res, seed])

  const baseColor = (id) => SERIES[(baselines.findIndex((b) => b.id === id) + 1) % 8]
  const eqSeries = [
    { key: 'rl', name: 'RL (median of seeds)', color: 'var(--s1)', width: 2.6 },
    ...baselines.filter((b) => shownBase.has(b.id)).map((b) => ({ key: b.id, name: b.name, color: baseColor(b.id), width: 1.4, dash: b.family === 'Hybrid ref' })),
  ]
  const summary = res?.summary || {}
  const bar = res?.bar_to_beat
  const medSharpe = summary.sharpe?.median
  const beat = medSharpe != null && bar != null ? medSharpe - bar : null
  const dsrMed = median(seeds.map((s) => s.dsr?.dsr))
  const cashMed = median(seeds.map((s) => s.metrics.avg_cash))
  const hybrid = res?.meta?.agent?.mode === 'hybrid'
  const rlRow = { id: 'rl', name: 'RL (median of seeds)', family: 'RL', metrics: Object.fromEntries(Object.entries(summary).map(([k, v]) => [k, v.median])) }

  if (!runs.length) {
    return (<><div className="page-head"><div><h1>Results</h1></div></div>
      <Card><Empty icon={<FlaskConical size={36} />} title="No runs yet">Train an agent first. <Button size="sm" onClick={() => go('train')}>Go to Train</Button></Empty></Card></>)
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Results</h1>
          <p>Seeded evaluation of a run against every baseline, net of identical costs. Median and IQR across seeds; deflated Sharpe accounts for all configurations tried.</p>
        </div>
        <div className="row">
          <select value={runId || ''} onChange={(e) => setRunId(e.target.value)} style={{ maxWidth: 380 }}>
            {runs.map((r) => <option key={r.id} value={r.id}>{r.kind === 'walkforward' ? '[WF] ' : ''}{r.name} — {r.created?.slice(0, 16).replace('T', ' ')} ({r.status})</option>)}
          </select>
          {!isWF && <Segmented value={seg} onChange={setSeg} options={SEGMENTS.map((s) => (s.value === 'test' && run?.test?.touched ? { ...s, icon: <LockOpen size={12} /> } : s))} />}
          {isWF && <Badge tone="accent">Walk-forward out-of-sample</Badge>}
          <Button size="sm" kind="ghost" icon={<Trash2 size={14} />} onClick={() => setDelConfirm(true)} title="Delete run" />
        </div>
      </div>
      {err && <Alert tone="bad">{err}</Alert>}
      {run?.status === 'running' && <Alert tone="info">This run is still training — results update as seeds finish.</Alert>}
      {res?.meta?.dataset?.source === 'synthetic' && <Alert tone="warn">This run used the synthetic offline market. Re-run on Yahoo data for reportable results.</Alert>}

      {res?.locked ? (
        <Card>
          <Empty icon={<Lock size={36} />} title="Test block is locked">
            The test period is touched <b>once</b>, at the end, after all design choices are frozen. Unlocking evaluates every seed's selected checkpoint and records a timestamp in the run log.
            <div style={{ marginTop: 14 }}><Button kind="primary" icon={<LockOpen size={15} />} onClick={() => setConfirm(true)}>Evaluate on test (one-shot)</Button></div>
          </Empty>
        </Card>
      ) : res && (
        <>
          {seg === 'test' && res.meta?.test?.touched > 1 && <Alert tone="warn">The test block for this run has been evaluated {res.meta.test.touched} times. Report the first evaluation.</Alert>}
          <div className="grid g5">
            <Stat label="Median Sharpe (net)" value={num(medSharpe)} foot={summary.sharpe ? `IQR ${num(summary.sharpe.q1)} – ${num(summary.sharpe.q3)} · ${seeds.length} seeds` : ''} tone={beat == null ? undefined : beat > 0 ? 'good' : 'bad'} />
            <Stat label="vs bar to beat" value={beat == null ? '—' : `${beat >= 0 ? '+' : ''}${num(beat)}`} foot={`EW quarterly = ${num(bar)}`} tone={beat == null ? undefined : beat > 0 ? 'good' : 'bad'} />
            <Stat label="Median CAGR" value={pct(summary.cagr?.median)} foot={`Max DD ${pct(summary.max_drawdown?.median)}`} />
            <Stat label="Turnover / cost" value={pct(summary.turnover_ann?.median, 0)} foot={`Total cost ${pct(summary.total_cost?.median, 2)} · avg cash ${pct(cashMed, 0)}`} />
            <Stat label="Deflated Sharpe (median)" value={num(dsrMed, 2)} foot={`P(true SR > 0) after ${res.n_trials} trials`} tone={dsrMed == null ? undefined : dsrMed > 0.95 ? 'good' : undefined} />
          </div>

          <div className="grid g2">
            <Card title="Growth of ₹1" sub="Blue band = range across seeds. Toggle baselines in the table below.">
              <TimeChart data={equity} series={eqSeries} band bandColor="var(--s1)" height={320} yFmt={(v) => `₹${v.toFixed(2)}`} />
            </Card>
            <Card title="Drawdown" sub="RL median vs selected baselines">
              <TimeChart data={dd} series={eqSeries} height={320} yFmt={(v) => pct(v, 0)} refY={0} />
            </Card>
          </div>

          <Card title="RL vs baselines" sub="Click rows to show/hide on the charts. Dashed = the run's own optimiser applied mechanically (hybrid reference).">
            <DataTable rows={[rlRow, ...baselines]} rowKey={(r) => r.id} initialSort={{ key: 'sharpe', dir: 'desc' }}
              highlight={(r) => r.id === 'rl'}
              onRowClick={(r) => r.id !== 'rl' && setShownBase((s) => { const n = new Set(s); n.has(r.id) ? n.delete(r.id) : n.add(r.id); return n })}
              columns={metricCols([
                { key: 'name', label: 'Strategy', get: (r) => r.name, render: (r) => (<span><span className="swatch" style={{ background: r.id === 'rl' ? 'var(--s1)' : shownBase.has(r.id) ? baseColor(r.id) : 'var(--surface-3)' }} />{r.id === 'rl' ? <b>{r.name}</b> : r.name}</span>) },
              ])} />
          </Card>

          <Card title="Per-seed results" sub="Checkpoint chosen on validation Sharpe. PSR = P(SR>0); DSR deflates for the number of trials.">
            <DataTable rows={seeds} rowKey={(r) => r.seed} highlight={(r) => r.seed === seed} onRowClick={(r) => setSeed(r.seed)}
              columns={[
                { key: 'seed', label: 'Seed', get: (r) => r.seed },
                { key: 'best', label: 'Best step', get: (r) => r.best_step, fmt: (v) => (v != null ? v.toLocaleString() : '—') },
                ...metricCols().slice(0, 8),
                { key: 'cash', label: 'Avg cash', get: (r) => r.metrics.avg_cash, fmt: (v) => pct(v, 0) },
                { key: 'psr', label: 'PSR', get: (r) => r.dsr?.psr, fmt: (v) => num(v, 2) },
                { key: 'dsr', label: 'DSR', get: (r) => r.dsr?.dsr, fmt: (v) => num(v, 2) },
              ]} />
          </Card>

          {curves[seed] && (
            <div className="grid g2">
              <Card title={`Portfolio weights · seed ${seed}`} sub="Post-trade target weights each bar (cap enforced structurally)">
                <WeightsChart dates={curves[seed].dates} weights={curves[seed].weights} assets={res.assets} height={280} />
              </Card>
              <Card title={hybrid ? `Agent action · seed ${seed}` : `Turnover · seed ${seed}`} sub={hybrid ? 'a = fraction of the gap to the optimiser target closed each bar (0 = hold, 1 = full rebalance)' : 'One-way turnover per bar'}>
                <TimeChart data={actionData} height={280} yFmt={(v) => pct(v, 0)} toggle={false}
                  series={hybrid ? [{ key: 'a', name: 'Move fraction a', color: 'var(--s1)', width: 1.4 }, { key: 'turnover', name: 'Turnover', color: 'var(--s2)', width: 1.2 }]
                    : [{ key: 'turnover', name: 'Turnover', color: 'var(--s2)', width: 1.4 }]} />
              </Card>
            </div>
          )}

          {res.meta && (
            <Card title="Run configuration" className="tint">
              <dl className="kv small">
                <dt>Agent</dt><dd>{res.meta.agent.algo} · {res.meta.agent.arch} · {res.meta.agent.mode}{res.meta.agent.mode === 'hybrid' ? ` (target: ${res.meta.agent.target})` : ''}</dd>
                <dt>Reward</dt><dd>{res.meta.agent.reward.risk} λ={res.meta.agent.reward.risk_lambda} · κ={res.meta.agent.reward.concentration_kappa} · turnover pen.={res.meta.agent.reward.turnover_penalty}</dd>
                <dt>Training</dt><dd>{res.meta.agent.timesteps.toLocaleString()} steps × {res.meta.agent.seeds} seeds · lr {res.meta.agent.learning_rate} · episode {res.meta.agent.episode_len} bars</dd>
                <dt>Costs</dt><dd>{res.meta.settings.costs.model === 'india' ? 'India itemised' : `${res.meta.settings.costs.flat_bps} bps flat`} + {res.meta.settings.costs.slippage_bps} bps slippage</dd>
                <dt>Dataset</dt><dd className="mono">{res.meta.dataset.id} #{res.meta.dataset.sha} ({res.meta.dataset.source})</dd>
                <dt>Test block</dt><dd>{res.meta.test?.touched ? `evaluated ${res.meta.test.touched}× (first: ${res.meta.test.history?.[0]})` : 'locked'}</dd>
                {res.meta.folds && <><dt>Folds</dt><dd>{res.meta.folds.map((f) => f.year).join(', ')}</dd></>}
              </dl>
            </Card>
          )}
        </>
      )}
      {loading && <div className="small muted">Loading…</div>}

      <Modal open={confirm} onClose={() => setConfirm(false)} title="Evaluate on the test block?"
        actions={<><Button onClick={() => setConfirm(false)}>Cancel</Button><Button kind="primary" onClick={unlockTest}>Yes, evaluate once</Button></>}>
        This is the out-of-sample number you will report. After this, changing the configuration based on test results would contaminate it. The evaluation is timestamped in the run log.
      </Modal>
      <Modal open={delConfirm} onClose={() => setDelConfirm(false)} title="Delete this run?"
        actions={<><Button onClick={() => setDelConfirm(false)}>Cancel</Button><Button kind="primary" onClick={del}>Delete</Button></>}>
        Models, curves and metrics for <b>{run?.name}</b> will be removed from disk. Note: deleting runs lowers the trial count used by the deflated Sharpe — keep failed experiments for honest multiplicity accounting.
      </Modal>
    </>
  )
}

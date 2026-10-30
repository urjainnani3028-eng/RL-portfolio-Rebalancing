import { useEffect, useMemo, useRef, useState } from 'react'
import { Ban, Dices, Lock, Rocket } from 'lucide-react'
import { api, num, pct } from '../api'
import { Alert, Badge, Button, Card, DataTable, Empty, Field, NumInput, Progress, Segmented, Stat, toast } from '../components/ui'
import { TimeChart } from '../components/charts'

const MODEL_COLOR = { rnn: 'var(--neutral)', lstm: 'var(--s1)', gru: 'var(--s2)', tcn: 'var(--s3)', patchtst: 'var(--s4)' }
const ALGO_COLOR = ['var(--neutral)', 'var(--s5)', 'var(--s1)', 'var(--s6)', 'var(--s3)', 'var(--s2)']
const med = (row, k) => row.seeds?.[k]?.median

export default function Bandit({ overview, refresh }) {
  const [bm, setBm] = useState(null)
  const [cfg, setCfg] = useState(null)
  const [results, setResults] = useState([])
  const [res, setRes] = useState(null)
  const [job, setJob] = useState(null)
  const t = useRef()

  const loadResults = async (pick) => {
    const rs = await api.get('/bandit/results')
    setResults(rs)
    const id = pick || rs[0]?.id
    if (id) setRes(await api.get(`/bandit/results/${id}`))
  }
  useEffect(() => {
    api.get('/bandit/meta').then((m) => { setBm(m); setCfg(m.defaults) })
    loadResults().catch(() => {})
    api.get('/jobs').then((js) => { const j = js.find((x) => x.kind === 'bandit' && ['queued', 'running'].includes(x.status)); if (j) follow(j.id) })
    return () => clearInterval(t.current)
  }, []) // eslint-disable-line

  const follow = (id) => {
    clearInterval(t.current)
    const poll = async () => {
      const j = await api.get(`/jobs/${id}`)
      setJob(j)
      if (['done', 'failed', 'cancelled'].includes(j.status)) {
        clearInterval(t.current); refresh()
        if (j.status === 'done') loadResults(j.result)
        if (j.status === 'failed') toast(j.error?.split('\n').slice(-2).join(' ') || 'Study failed', 'err')
      }
    }
    poll(); t.current = setInterval(poll, 1500)
  }

  const launch = async (test) => {
    try {
      const j = await api.post('/bandit', { ...cfg, evaluate_test: test })
      toast(test ? 'Study queued with one-shot test evaluation' : 'Study queued'); follow(j.id)
    } catch (e) { toast(e.message, 'err') }
  }

  if (!bm || !cfg) return <Empty icon={<Dices />} title="Loading…" />
  const set = (k, v) => setCfg((c) => ({ ...c, [k]: v }))
  const running = job && ['queued', 'running'].includes(job.status)

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Model bandit</h1>
          <p>RNN, LSTM, GRU and two open-source architectures (TCN, PatchTST) forecast next-week ETF returns. A multi-armed bandit decides which model gets more training and testing, and replays show whether it finds the best model faster than testing every model equally.</p>
        </div>
      </div>
      {!overview?.dataset && <Alert tone="warn">Prepare a dataset first (Data page).</Alert>}

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 2fr)' }}>
        <Card title="Study settings" sub="Pull = train the next seed of a model, score it on a random validation window">
          <div className="col" style={{ gap: 12 }}>
            <Field label="Bandit algorithm">
              <select value={cfg.algorithm} onChange={(e) => set('algorithm', e.target.value)}>
                {bm.algorithms.map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
            </Field>
            <Field label="Models (arms)">
              <div className="row" style={{ flexWrap: 'wrap', gap: 10 }}>
                {Object.entries(bm.models).map(([id, m]) => (
                  <label key={id} className="row small" style={{ gap: 4 }}>
                    <input type="checkbox" checked={cfg.models.includes(id)}
                      onChange={(e) => set('models', e.target.checked ? [...cfg.models, id] : cfg.models.filter((x) => x !== id))} />
                    {m.label}
                  </label>
                ))}
              </div>
            </Field>
            <div className="row" style={{ gap: 12 }}>
              <Field label="Budget (pulls)"><NumInput value={cfg.budget} min={5} onChange={(v) => set('budget', v)} /></Field>
              <Field label="Seeds per model"><NumInput value={cfg.max_seeds} min={1} onChange={(v) => set('max_seeds', v)} /></Field>
              <Field label="Window (weeks)"><NumInput value={cfg.window} min={8} onChange={(v) => set('window', v)} /></Field>
            </div>
            <Field label="Reward reference" hint="Score Sharpe and drawdown relative to equal-weight quarterly on the same window">
              <Segmented value={cfg.reward_reference} onChange={(v) => set('reward_reference', v)}
                options={[{ value: 'ew_quarterly', label: 'EW quarterly' }, { value: 'none', label: 'Absolute' }]} />
            </Field>
            <div className="row">
              <Button kind="primary" icon={<Rocket size={15} />} disabled={running || !overview?.dataset || cfg.models.length < 2} onClick={() => launch(false)}>Run study</Button>
              <Button icon={<Lock size={15} />} disabled={running || !overview?.dataset} onClick={() => launch(true)} title="Also evaluates every model on the locked test block">Run + test (one-shot)</Button>
            </div>
            {job && (
              <div className="col" style={{ gap: 6 }}>
                <div className="row small"><Badge tone={job.status === 'failed' ? 'bad' : job.status === 'done' ? 'good' : 'info'}>{job.status}</Badge>{job.message}</div>
                <Progress value={job.progress} />
                {running && <Button size="sm" icon={<Ban size={14} />} onClick={() => api.post(`/jobs/${job.id}/cancel`)}>Cancel</Button>}
              </div>
            )}
            {results.length > 1 && (
              <Field label="Previous studies">
                <select value={res?.id || ''} onChange={(e) => loadResults(e.target.value)}>
                  {results.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.algorithm} · {r.winner}{r.test_unlocked ? ' · test' : ''}</option>)}
                </select>
              </Field>
            )}
          </div>
        </Card>
        {res ? <Summary res={res} /> : <Empty icon={<Dices />} title="No study yet">Run a study to see which model the bandit picks.</Empty>}
      </div>
      {res && <Details res={res} />}
    </>
  )
}

function Summary({ res }) {
  const lab = (m) => res.models[m]?.label || m
  const { live, replay } = res
  const algo = replay.algorithms[live.algorithm]
  const uni = replay.algorithms.uniform
  return (
    <div className="col" style={{ gap: 16 }}>
      {res.dataset.source === 'synthetic' && <Alert tone="warn">Synthetic data: use it to check the pipeline, not to draw conclusions.</Alert>}
      <div className="grid" style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
        <Stat label="Bandit's pick" value={lab(live.winner)} foot={`${live.algorithm}, ${live.budget} pulls`} tone={live.correct ? 'good' : undefined} />
        <Stat label="True best (exhaustive)" value={lab(replay.true_best)} foot={`gap to 2nd: ${num(replay.gap_to_second, 3)}`} />
        <Stat label="P(correct) at budget" value={pct(algo?.final_p_correct, 0)} foot={`uniform: ${pct(uni?.final_p_correct, 0)}`} />
        <Stat label="Models trained" value={`${live.compute.bandit_trainings} / ${live.compute.uniform_trainings}`} foot={`bandit vs uniform (~${num(live.compute.bandit_train_seconds_est, 0)} s vs ${num(live.compute.uniform_train_seconds_est, 0)} s)`} />
      </div>
      <Card title="Arms" sub="Live pulls and rewards vs. the exhaustive ground truth">
        <DataTable rows={res.settings.models.map((m) => ({ m }))} rowKey={(r) => r.m} highlight={(r) => r.m === live.winner}
          columns={[
            { key: 'm', label: 'Model', get: (r) => lab(r.m), render: (r) => <span><span className="swatch" style={{ background: MODEL_COLOR[r.m] }} />{lab(r.m)}</span> },
            { key: 'pulls', label: 'Live pulls', get: (r) => live.pulls[r.m] },
            { key: 'mean', label: 'Live mean reward', get: (r) => live.mean_reward[r.m], fmt: (v) => num(v, 3) },
            { key: 'true', label: 'True mean reward', get: (r) => replay.true_mean_reward[r.m], fmt: (v) => num(v, 3) },
            { key: 'sd', label: 'Reward sd', get: (r) => replay.true_std_reward[r.m], fmt: (v) => num(v, 3) },
            { key: 'secs', label: 'Train s / seed', get: (r) => res.validation.models[r.m]?.train_seconds_median, fmt: (v) => num(v, 1) },
          ]} />
      </Card>
    </div>
  )
}

function Details({ res }) {
  const [seg, setSeg] = useState('validation')
  const lab = (m) => res.models[m]?.label || m
  const algos = Object.keys(res.replay.algorithms)
  const pcData = useMemo(() => res.replay.algorithms[algos[0]].p_correct.map((_, i) => {
    const row = { date: i + 1 }
    algos.forEach((a) => { row[a] = res.replay.algorithms[a].p_correct[i] })
    return row
  }), [res]) // eslint-disable-line
  const segData = seg === 'test' ? res.test : res.validation
  const testOpen = res.test && !res.test.locked
  const eq = useMemo(() => {
    const c = segData?.equity_curves
    if (!c) return []
    const keys = Object.keys(c)
    return c[keys[0]].dates.map((d, i) => { const r = { date: d }; keys.forEach((k) => { r[k] = c[k].value[i] }); return r })
  }, [segData])
  const rows = segData?.models ? segData.ranking_by_sharpe.map((m) => ({ m, ...segData.models[m] })) : []
  return (
    <div className="col" style={{ gap: 16, marginTop: 16 }}>
      <Card title="Bandit efficiency" sub={`P(recommending the true best model) vs pulls, ${res.settings.replay_runs} replays per algorithm`}>
        <TimeChart data={pcData} xFmt={(v) => v} yFmt={(v) => pct(v, 0)} yDomain={[0, 1]}
          series={algos.map((a, i) => ({ key: a, name: a, color: ALGO_COLOR[i % ALGO_COLOR.length], dash: a === 'uniform', width: a === res.live.algorithm ? 2.6 : 1.5 }))} />
      </Card>
      <Card title="Model comparison" sub="Median [IQR] over seeds; portfolios built by the same rule from each model's forecasts, net of costs"
        right={<Segmented value={seg} onChange={setSeg} options={[{ value: 'validation', label: 'Validation' }, { value: 'test', label: testOpen ? 'Test' : 'Test 🔒' }]} />}>
        {seg === 'test' && !testOpen ? <Alert tone="info">The test block is locked. Use "Run + test (one-shot)" once the design is frozen.</Alert> : (
          <div className="col" style={{ gap: 16 }}>
            <DataTable rows={rows} rowKey={(r) => r.m} highlight={(r) => r.m === res.live.winner}
              columns={[
                { key: 'm', label: 'Model', get: (r) => r.label, render: (r) => <span><span className="swatch" style={{ background: MODEL_COLOR[r.m] }} />{r.label} <span className="muted xs">{r.family}</span></span> },
                { key: 'sharpe', label: 'Sharpe', get: (r) => med(r, 'sharpe'), fmt: (v) => num(v) },
                { key: 'cagr', label: 'CAGR', get: (r) => med(r, 'cagr'), fmt: (v) => pct(v) },
                { key: 'mdd', label: 'Max DD', get: (r) => med(r, 'max_drawdown'), fmt: (v) => pct(v) },
                { key: 'to', label: 'Turnover/yr', get: (r) => med(r, 'turnover_ann'), fmt: (v) => num(v) },
                { key: 'acc', label: 'Dir. acc.', get: (r) => med(r, 'dir_acc'), fmt: (v) => pct(v) },
                { key: 'mae', label: 'MAE (z)', get: (r) => med(r, 'mae'), fmt: (v) => num(v, 3) },
                { key: 'ic', label: 'Rank IC', get: (r) => med(r, 'ic'), fmt: (v) => num(v, 3) },
              ]} />
            <DataTable rows={Object.entries(segData.baselines).map(([id, v]) => ({ id, ...v }))} rowKey={(r) => r.id}
              columns={[
                { key: 'name', label: 'Baseline' },
                { key: 'sharpe', label: 'Sharpe', fmt: (v) => num(v) },
                { key: 'cagr', label: 'CAGR', fmt: (v) => pct(v) },
                { key: 'max_drawdown', label: 'Max DD', fmt: (v) => pct(v) },
                { key: 'turnover_ann', label: 'Turnover/yr', fmt: (v) => num(v) },
              ]} />
            <TimeChart data={eq} yFmt={(v) => num(v, 2)}
              series={Object.keys(segData.equity_curves).map((k) => ({ key: k, name: lab(k) === k ? (segData.baselines[k]?.name || k) : `${lab(k)} (seed ensemble)`, color: MODEL_COLOR[k] || 'var(--text-2)', dash: !MODEL_COLOR[k] }))} />
          </div>
        )}
      </Card>
    </div>
  )
}

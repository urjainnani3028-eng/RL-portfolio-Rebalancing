import { useEffect, useMemo, useState } from 'react'
import { Gauge } from 'lucide-react'
import { SERIES, api, num, pct } from '../api'
import { Alert, Badge, Button, Card, Empty, Segmented } from '../components/ui'
import { RiskReturnScatter, TimeChart, useThemeColors, resolveColor, Legend } from '../components/charts'
import { SEGMENTS } from './Baselines'
import { CartesianGrid, ComposedChart, Area, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export default function Robustness({ go }) {
  const [seg, setSeg] = useState('val')
  const [scatter, setScatter] = useState(null)
  const [runs, setRuns] = useState([])
  const [runId, setRunId] = useState(null)
  const [sweep, setSweep] = useState(null)
  const [reg, setReg] = useState(null)
  const [busy, setBusy] = useState('')
  const [err, setErr] = useState('')

  useEffect(() => { api.get('/runs').then((r) => { setRuns(r); if (r.length) setRunId(r[0].id) }) }, [])
  useEffect(() => { api.get(`/analysis/scatter?segment=${seg}`).then(setScatter).catch((e) => setErr(e.message)) }, [seg])
  useEffect(() => { setSweep(null); setReg(null) }, [runId, seg])

  const run = runs.find((r) => r.id === runId)
  const runSweep = async () => {
    setBusy('sweep'); setErr('')
    try { setSweep(await api.get(`/analysis/cost-sweep/${runId}?segment=${seg}`)) } catch (e) { setErr(e.message) }
    setBusy('')
  }
  const runReg = async () => {
    setBusy('reg'); setErr('')
    try { setReg(await api.get(`/analysis/regimes/${runId}?segment=${seg}`)) } catch (e) { setErr(e.message) }
    setBusy('')
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Robustness</h1>
          <p>The headline chart, a transaction-cost sweep (0–50 bps) and regime-sliced performance. RL claims must survive all three.</p>
        </div>
        <Segmented value={seg} onChange={setSeg} options={SEGMENTS} />
      </div>
      {err && <Alert tone="bad">{err}</Alert>}

      <Card title="Return vs maximum drawdown" sub="Baselines are points; each RL run is a cloud of seeds (up-and-left is better). Test shows only runs whose test block was unlocked.">
        {scatter?.points?.length ? <RiskReturnScatter points={scatter.points} highlightRun={runId} /> : <Empty icon={<Gauge size={34} />} title="Nothing to plot yet">Prepare data and train at least one run.</Empty>}
      </Card>

      <Card title="Per-run analysis" right={
        <>
          <select value={runId || ''} onChange={(e) => setRunId(e.target.value)} style={{ maxWidth: 360 }}>
            {runs.map((r) => <option key={r.id} value={r.id}>{r.kind === 'walkforward' ? '[WF] ' : ''}{r.name} — {r.created?.slice(5, 16).replace('T', ' ')}</option>)}
          </select>
          <Button size="sm" onClick={runSweep} loading={busy === 'sweep'} disabled={!runId || run?.kind === 'walkforward'}>Cost sweep</Button>
          <Button size="sm" kind="primary" onClick={runReg} loading={busy === 'reg'} disabled={!runId}>Regime slices</Button>
        </>}>
        {!runs.length && <Empty title="No runs yet"><Button size="sm" onClick={() => go('train')}>Train an agent</Button></Empty>}
        {runs.length > 0 && !sweep && !reg && <div className="small muted">Pick a run and an analysis. Cost sweep re-evaluates every seed's checkpoint at 0, 5, 10, 20, 30 and 50 bps (flat) alongside the baselines.</div>}
        <div className="col" style={{ gap: 18 }}>
          {sweep && <SweepChart sweep={sweep} />}
          {reg && <RegimeTable reg={reg} />}
        </div>
      </Card>
    </>
  )
}

function SweepChart({ sweep }) {
  const colors = useThemeColors()
  const base = Object.entries(sweep.baselines)
  const data = sweep.bps.map((b, i) => {
    const r = { bps: b, rl: sweep.rl[i].median, _band: [sweep.rl[i].q1, sweep.rl[i].q3] }
    base.forEach(([k, v]) => { r[k] = v.sharpe[i] })
    return r
  })
  const series = [{ key: 'rl', name: 'RL (median, IQR band)', color: colors.series[0] },
    ...base.map(([k, v], i) => ({ key: k, name: v.name, color: colors.series[(i + 1) % 8], dash: k.startsWith('tgt') }))]
  const TT = ({ active, payload, label }) => active && payload?.length ? (
    <div className="tt"><div className="tt-title">{label} bps</div>
      {payload.filter((p) => !Array.isArray(p.value)).sort((a, b) => b.value - a.value).map((p) => (
        <div className="tt-row" key={p.dataKey}><span className="swatch" style={{ background: p.stroke }} /><span>{series.find((s) => s.key === p.dataKey)?.name}</span><b>{num(p.value)}</b></div>))}
    </div>) : null
  return (
    <div>
      <h3>Sharpe vs transaction cost</h3>
      <div className="xs muted" style={{ marginBottom: 8 }}>Median RL turnover: {sweep.rl.map((r) => pct(r.turnover, 0)).join(' → ')} per year. An agent trained at one cost that stays flat here has learned cost-aware behaviour.</div>
      <Legend series={series} />
      <div style={{ height: 300 }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke={colors.grid} />
            <XAxis dataKey="bps" tickFormatter={(v) => `${v} bps`} tick={{ fill: colors.muted, fontSize: 11 }} tickLine={false} axisLine={{ stroke: colors.border }} />
            <YAxis tickFormatter={(v) => num(v, 1)} width={44} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
            <Tooltip content={<TT />} />
            <Area dataKey="_band" stroke="none" fill={colors.series[0]} fillOpacity={0.15} isAnimationActive={false} />
            {series.map((s) => <Line key={s.key} dataKey={s.key} stroke={s.color} strokeWidth={s.key === 'rl' ? 2.6 : 1.5} strokeDasharray={s.dash ? '5 4' : undefined} dot={{ r: 3, strokeWidth: 0, fill: s.color }} isAnimationActive={false} />)}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function RegimeTable({ reg }) {
  const [metric, setMetric] = useState('sharpe')
  const strategies = reg.rows.length ? Object.keys(reg.rows[0].strategies) : []
  const fmt = metric === 'sharpe' ? (v) => num(v) : (v) => pct(v)
  const cell = (row, s) => {
    const v = row.strategies[s]?.[metric]
    if (v == null) return <td key={s}>—</td>
    const vals = Object.values(row.strategies).map((x) => x[metric]).filter((x) => x != null)
    const better = metric === 'max_drawdown' ? v <= Math.min(...vals) : v >= Math.max(...vals)
    return <td key={s} style={better ? { fontWeight: 700, color: 'var(--good)' } : undefined}>{fmt(v)}</td>
  }
  return (
    <div>
      <div className="row between" style={{ marginBottom: 8 }}>
        <h3>Regime-sliced performance · {reg.segment}</h3>
        <Segmented value={metric} onChange={setMetric} options={[{ value: 'sharpe', label: 'Sharpe' }, { value: 'ann_return', label: 'Ann. return' }, { value: 'cum_return', label: 'Cum. return' }, { value: 'max_drawdown', label: 'Max DD' }]} />
      </div>
      <div className="xs muted" style={{ marginBottom: 8 }}>States use trailing market (Nifty) return / volatility known before each bar; events are fixed historical windows that fall inside this block. Best value per row in bold.</div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Regime</th><th>Bars</th>{strategies.map((s) => <th key={s} title={s}>{s.length > 22 ? s.slice(0, 20) + '…' : s}</th>)}</tr></thead>
          <tbody>
            {reg.rows.map((r) => (
              <tr key={r.regime} className="hover">
                <td>{r.regime} {r.kind === 'event' && <Badge>event</Badge>}</td>
                <td className="muted">{r.bars}</td>
                {strategies.map((s) => cell(r, s))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

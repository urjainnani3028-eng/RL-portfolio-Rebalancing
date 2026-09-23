import { useEffect, useMemo, useState } from 'react'
import { Lock, Save, Target } from 'lucide-react'
import { SERIES, api, num, pct, short } from '../api'
import { Alert, Badge, Button, Card, DataTable, Field, NumInput, Segmented, Stat, toast } from '../components/ui'
import { TimeChart } from '../components/charts'

export const SEGMENTS = [
  { value: 'train', label: 'Train' },
  { value: 'val', label: 'Validation' },
  { value: 'test', label: 'Test', icon: <Lock size={12} /> },
]

export const metricCols = (extra = []) => [
  ...extra,
  { key: 'sharpe', label: 'Sharpe', get: (r) => r.metrics?.sharpe, fmt: (v) => num(v), title: 'Annualised excess-return Sharpe, net of costs' },
  { key: 'sortino', label: 'Sortino', get: (r) => r.metrics?.sortino, fmt: (v) => num(v) },
  { key: 'calmar', label: 'Calmar', get: (r) => r.metrics?.calmar, fmt: (v) => num(v) },
  { key: 'cagr', label: 'CAGR', get: (r) => r.metrics?.cagr, fmt: (v) => pct(v) },
  { key: 'vol', label: 'Vol', get: (r) => r.metrics?.vol, fmt: (v) => pct(v) },
  { key: 'mdd', label: 'Max DD', get: (r) => r.metrics?.max_drawdown, fmt: (v) => pct(v) },
  { key: 'to', label: 'Turnover/yr', get: (r) => r.metrics?.turnover_ann, fmt: (v) => pct(v, 0) },
  { key: 'cost', label: 'Total cost', get: (r) => r.metrics?.total_cost, fmt: (v) => pct(v, 2) },
  { key: 'hhi', label: 'Herfindahl', get: (r) => r.metrics?.herfindahl, fmt: (v) => num(v, 3), title: 'Average concentration of risky weights' },
  { key: 'alpha', label: 'α (ann.)', get: (r) => r.metrics?.alpha_ann, fmt: (v) => pct(v), title: 'CAPM alpha vs the market ETF' },
]

export default function Baselines({ refresh }) {
  const [seg, setSeg] = useState('val')
  const [res, setRes] = useState(null)
  const [settings, setSettings] = useState(null)
  const [sum, setSum] = useState(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  const load = async (s = seg) => {
    setLoading(true); setErr('')
    try { setRes(await api.get(`/baselines?segment=${s}`)) } catch (e) { setErr(e.message); setRes(null) }
    setLoading(false)
  }
  useEffect(() => {
    api.get('/settings').then(setSettings)
    api.get('/data/summary').then(setSum)
  }, [])
  useEffect(() => { load(seg) }, [seg]) // eslint-disable-line

  const saveCosts = async () => {
    try {
      await api.put('/settings', settings)
      setSum(await api.get('/data/summary'))
      await load(seg); refresh(); toast('Cost model updated — baselines recomputed')
    } catch (e) { toast(e.message, 'err') }
  }
  const c = settings?.costs
  const setC = (k, v) => setSettings((s) => ({ ...s, costs: { ...s.costs, [k]: v } }))

  const strategies = res?.strategies || []
  const defaults = new Set(['market', 'ew_quarterly', 'risk_parity_monthly', 'min_variance_monthly'])
  const [shown, setShown] = useState(defaults)
  const chartData = useMemo(() => {
    if (!strategies.length) return []
    const dates = strategies[0].curve.dates
    return dates.map((d, i) => {
      const r = { date: d }
      strategies.forEach((s) => { r[s.id] = s.curve.value[i] })
      return r
    })
  }, [res])
  const ddData = useMemo(() => {
    if (!strategies.length) return []
    return strategies[0].curve.dates.map((d, i) => {
      const r = { date: d }
      strategies.forEach((s) => { r[s.id] = s.curve.drawdown[i] })
      return r
    })
  }, [res])
  const colorOf = (id) => SERIES[strategies.findIndex((s) => s.id === id) % 8]
  const series = strategies.filter((s) => shown.has(s.id)).map((s) => ({ key: s.id, name: s.name, color: colorOf(s.id), width: s.id === 'ew_quarterly' ? 2.4 : 1.6 }))
  const best = strategies.length ? strategies.reduce((a, b) => ((b.metrics.sharpe ?? -9) > (a.metrics.sharpe ?? -9) ? b : a)) : null

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Costs & baselines</h1>
          <p>Every rule runs through the same simulator and cost model as the agent, starting from 100% cash. These are the numbers RL has to beat — a complete result even if RL fails.</p>
        </div>
        <Segmented value={seg} onChange={setSeg} options={SEGMENTS} />
      </div>
      {seg === 'test' && <Alert tone="warn">You are looking at the <b>test block</b>. Baselines here are fine to inspect, but do not tune any RL setting after looking at it.</Alert>}
      {err && <Alert tone="bad">{err}</Alert>}

      <div className="grid g4">
        <Stat icon={<Target size={13} />} label="Bar to beat · EW quarterly Sharpe" value={num(res?.bar_to_beat)} foot={`${SEGMENTS.find((s) => s.value === seg).label} block, net of costs`} />
        <Stat label="Best baseline Sharpe" value={num(best?.metrics.sharpe)} foot={best?.name} />
        <Stat label="Cost model" value={c?.model === 'india' ? 'India' : `${c?.flat_bps ?? '—'} bps`} foot={c?.model === 'india' ? 'STT · stamp duty · GST · fees + slippage' : 'Flat per unit traded + slippage'} />
        <Stat label="Round trip (equity ETF)" value={sum?.costs ? `${num(sum.costs.round_trip_bps[0], 1)} bps` : '—'} foot="Buy + sell, per ₹ traded" />
      </div>

      <div className="grid g3">
        <Card title="Transaction cost model" sub="Charged on turnover, before the market moves">
          {c && (
            <div className="col">
              <Segmented value={c.model} onChange={(v) => setC('model', v)} options={[{ value: 'india', label: 'India (itemised)' }, { value: 'flat', label: 'Flat bps' }]} />
              {c.model === 'flat' ? (
                <div className="grid g2" style={{ gap: 10 }}>
                  <Field label="Cost (bps per ₹ traded)"><NumInput value={c.flat_bps} onChange={(v) => setC('flat_bps', v)} /></Field>
                  <Field label="Slippage (bps)"><NumInput value={c.slippage_bps} onChange={(v) => setC('slippage_bps', v)} /></Field>
                </div>
              ) : (
                <div className="grid g2" style={{ gap: 10 }}>
                  <Field label="STT on ETF sell (%)"><NumInput value={c.stt_sell_etf_pct} onChange={(v) => setC('stt_sell_etf_pct', v)} /></Field>
                  <Field label="Stamp duty on buy (%)"><NumInput value={c.stamp_buy_pct} onChange={(v) => setC('stamp_buy_pct', v)} /></Field>
                  <Field label="NSE txn charge (%)"><NumInput value={c.exchange_txn_pct} onChange={(v) => setC('exchange_txn_pct', v)} /></Field>
                  <Field label="SEBI fee (%)"><NumInput value={c.sebi_fee_pct} onChange={(v) => setC('sebi_fee_pct', v)} /></Field>
                  <Field label="Brokerage (%)"><NumInput value={c.brokerage_pct} onChange={(v) => setC('brokerage_pct', v)} /></Field>
                  <Field label="GST on fees (%)"><NumInput value={c.gst_pct} onChange={(v) => setC('gst_pct', v)} /></Field>
                  <Field label="Slippage / spread (bps)"><NumInput value={c.slippage_bps} onChange={(v) => setC('slippage_bps', v)} /></Field>
                </div>
              )}
              <div className="xs muted">Defaults: zero-brokerage delivery, STT 0.001% on sale of equity-ETF units (none on gold/debt ETFs), stamp duty 0.015% on buys, 18% GST on fees. Verify against your broker's current schedule.</div>
              <Button kind="primary" size="sm" icon={<Save size={14} />} onClick={saveCosts}>Apply & recompute</Button>
            </div>
          )}
        </Card>
        <Card title="Cost per asset" sub="bps of notional traded (incl. slippage)" className="span2">
          {sum?.costs && (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Asset</th><th>Buy</th><th>Sell</th><th>Round trip</th></tr></thead>
                <tbody>
                  {sum.costs.assets.map((a, i) => (
                    <tr key={a} className="hover"><td>{short(a)}</td><td>{num(sum.costs.buy_bps[i], 2)}</td><td>{num(sum.costs.sell_bps[i], 2)}</td><td><b>{num(sum.costs.round_trip_bps[i], 2)}</b></td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <Card title="Baseline scoreboard" sub="Click a row to toggle it on the charts. Equal-weight quarterly is highlighted as the bar to beat." right={loading && <Badge>Computing…</Badge>}>
        <DataTable rows={strategies} rowKey={(r) => r.id} initialSort={{ key: 'sharpe', dir: 'desc' }}
          highlight={(r) => r.id === 'ew_quarterly'}
          onRowClick={(r) => setShown((s) => { const n = new Set(s); n.has(r.id) ? n.delete(r.id) : n.add(r.id); return n })}
          columns={metricCols([
            { key: 'name', label: 'Strategy', get: (r) => r.name, render: (r) => (<span><span className="swatch" style={{ background: shown.has(r.id) ? colorOf(r.id) : 'var(--surface-3)' }} />{r.name}</span>) },
            { key: 'family', label: 'Family', get: (r) => r.family, render: (r) => <Badge>{r.family}</Badge> },
          ])} />
      </Card>

      {chartData.length > 0 && (
        <div className="grid g2">
          <Card title="Growth of ₹1" sub="Net of all costs">
            <TimeChart data={chartData} series={series} height={300} yFmt={(v) => `₹${v.toFixed(2)}`} toggle={false} />
          </Card>
          <Card title="Drawdown" sub="Peak-to-trough decline">
            <TimeChart data={ddData} series={series} height={300} yFmt={(v) => pct(v, 0)} toggle={false} refY={0} />
          </Card>
        </div>
      )}
    </>
  )
}

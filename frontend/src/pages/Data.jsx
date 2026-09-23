import { useEffect, useMemo, useRef, useState } from 'react'
import { Download, RefreshCw, Upload, Wrench } from 'lucide-react'
import { api, pct, short } from '../api'
import { Alert, Badge, Button, Card, Field, NumInput, Segmented, toast } from '../components/ui'
import { TimeChart, useThemeColors } from '../components/charts'
import { SERIES } from '../api'

export default function DataPage({ meta, refresh }) {
  const [settings, setSettings] = useState(null)
  const [sum, setSum] = useState(null)
  const [busy, setBusy] = useState('')
  const fileRef = useRef()

  useEffect(() => {
    api.get('/settings').then(setSettings)
    api.get('/data/summary').then(setSum)
  }, [])

  const setData = (k, v) => setSettings((s) => ({ ...s, data: { ...s.data, [k]: v } }))
  const setPort = (k, v) => setSettings((s) => ({ ...s, portfolio: { ...s.portfolio, [k]: v } }))
  const toggle = (t) => setData('tickers', settings.data.tickers.includes(t) ? settings.data.tickers.filter((x) => x !== t) : [...settings.data.tickers, t])

  const prepare = async (refreshData) => {
    setBusy(refreshData ? 'refresh' : 'prepare')
    try {
      const r = await api.post('/data/prepare', { settings, refresh: refreshData })
      setSum(r)
      toast(`Dataset ready: ${r.meta.bars} bars from ${r.meta.source}`)
      refresh()
    } catch (e) { toast(e.message, 'err') }
    setBusy('')
  }

  const upload = async (f) => {
    if (!f) return
    setBusy('upload')
    try {
      const r = await api.upload('/data/upload', f)
      setSum(r); setSettings(r.settings); toast('CSV dataset loaded'); refresh()
    } catch (e) { toast(e.message, 'err') }
    setBusy('')
  }

  if (!settings) return null
  const d = settings.data, p = settings.portfolio
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Data & universe</h1>
          <p>Fix the universe and frequency, download from Yahoo Finance (NSE tickers), and freeze a snapshot so every experiment runs on identical data. Scalers are fit on the training block only.</p>
        </div>
        <div className="row">
          <input ref={fileRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={(e) => upload(e.target.files[0])} />
          <Button icon={<Upload size={15} />} onClick={() => fileRef.current.click()} loading={busy === 'upload'}>Import CSV</Button>
          <Button icon={<RefreshCw size={15} />} onClick={() => prepare(true)} loading={busy === 'refresh'}>Refresh from Yahoo</Button>
          <Button kind="primary" icon={<Download size={15} />} onClick={() => prepare(false)} loading={busy === 'prepare'}>Prepare data</Button>
        </div>
      </div>

      <div className="grid g3">
        <Card title="Universe" sub="Indian ETFs on NSE + a cash sleeve earning the risk-free rate" className="span2">
          <div className="grid g3" style={{ gap: 8 }}>
            {meta.catalogue.map((c) => (
              <button key={c.ticker} className={`choice ${d.tickers.includes(c.ticker) ? 'on' : ''}`} onClick={() => toggle(c.ticker)}>
                <div className="row between"><b>{short(c.ticker)}</b><span className="xs muted">since {c.listed}</span></div>
                <small>{c.cls}</small>
              </button>
            ))}
          </div>
          <div className="xs muted" style={{ marginTop: 10 }}>
            {d.tickers.length} selected. Late-listed ETFs shorten the common history (the loader aligns on the latest start date). Default = 7 ETFs with history from 2011.
          </div>
        </Card>
        <Card title="Sampling & splits" sub="Chronological, with an embargo between blocks">
          <div className="col">
            <Field label="Bar frequency"><Segmented value={d.freq} onChange={(v) => setData('freq', v)} options={[{ value: 'W', label: 'Weekly' }, { value: 'D', label: 'Daily' }]} /></Field>
            <Field label="Data source"><Segmented value={d.source} onChange={(v) => setData('source', v)} options={[{ value: 'auto', label: 'Auto' }, { value: 'yahoo', label: 'Yahoo' }, { value: 'synthetic', label: 'Synthetic' }]} /></Field>
            <div className="grid g2" style={{ gap: 10 }}>
              <Field label="Start"><input type="date" value={d.start} onChange={(e) => setData('start', e.target.value)} /></Field>
              <Field label="Embargo (bars)"><NumInput value={d.embargo_bars} onChange={(v) => setData('embargo_bars', Math.round(v))} /></Field>
              <Field label="Train ends"><input type="date" value={d.train_end} onChange={(e) => setData('train_end', e.target.value)} /></Field>
              <Field label="Validation ends"><input type="date" value={d.val_end} onChange={(e) => setData('val_end', e.target.value)} /></Field>
            </div>
          </div>
        </Card>
      </div>

      <Card title="Portfolio & state settings" sub="Changing these invalidates cached baselines; re-run after editing">
        <div className="grid g5">
          <Field label="Risk-free / cash (annual)" hint="Liquid fund proxy"><NumInput value={p.rf_annual} onChange={(v) => setPort('rf_annual', v)} /></Field>
          <Field label="Max weight per ETF" hint="Anti-concentration cap"><NumInput value={p.max_weight} onChange={(v) => setPort('max_weight', v)} /></Field>
          <Field label="Price window (bars)" hint="State lookback"><NumInput value={p.lookback} onChange={(v) => setPort('lookback', Math.round(v))} /></Field>
          <Field label="Risk window (bars)" hint="Vol / correlation"><NumInput value={p.risk_window} onChange={(v) => setPort('risk_window', Math.round(v))} /></Field>
          <Field label="Optimiser window (bars)" hint="Classical targets"><NumInput value={p.opt_window} onChange={(v) => setPort('opt_window', Math.round(v))} /></Field>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <label className="checkline" style={{ flex: 1 }}>
            <input type="checkbox" checked={p.exec_delay === 1} onChange={(e) => setPort('exec_delay', e.target.checked ? 1 : 0)} />
            Execute at next bar (state uses data up to t−1) — recommended, prevents trading on the decision close
          </label>
          <div style={{ flex: 1 }} />
          <Button size="sm" onClick={async () => { try { await api.put('/settings', settings); toast('Settings saved'); refresh() } catch (e) { toast(e.message, 'err') } }}>Save settings</Button>
        </div>
      </Card>

      {sum?.ready ? <DatasetView sum={sum} /> : (
        <Alert tone="info">No dataset yet. Choose the universe and click <b>Prepare data</b>. With internet access the app downloads from Yahoo Finance; offline it falls back to a clearly-labelled synthetic market so you can still test the pipeline.</Alert>
      )}
    </>
  )
}

function DatasetView({ sum }) {
  const m = sum.meta
  const assets = sum.assets.map((a) => a.ticker)
  const colors = useThemeColors()
  const data = useMemo(() => sum.dates.map((dt, i) => {
    const r = { date: dt }
    assets.forEach((a) => { r[a] = sum.prices[a][i] })
    return r
  }), [sum])
  const series = assets.map((a, i) => ({ key: a, name: short(a), color: a === 'CASH' ? 'var(--neutral)' : SERIES[i % 8], dash: a === 'CASH' }))
  const segs = sum.segments
  const total = sum.dates.length
  const emb = m.embargo_bars || 0
  return (
    <>
      {m.warnings?.map((w, i) => <Alert key={i} tone={m.source === 'synthetic' ? 'warn' : 'info'}>{w}</Alert>)}
      <div className="grid g3">
        <Card title="Frozen snapshot" right={<Badge tone={m.source === 'synthetic' ? 'warn' : 'good'}>{m.source}</Badge>}>
          <dl className="kv">
            <dt>Snapshot</dt><dd className="mono">{m.id} · #{m.sha}</dd>
            <dt>Range</dt><dd>{m.start} → {m.end}</dd>
            <dt>Bars</dt><dd>{m.bars} ({m.freq === 'W' ? 'weekly' : 'daily'})</dd>
            <dt>Created</dt><dd>{m.created?.replace('T', ' ')}</dd>
            <dt>Split repairs</dt><dd>{m.split_fixes?.length ? m.split_fixes.map((f) => `${short(f.ticker)} ${f.date} ×${f.factor}`).join(', ') : 'none needed'}</dd>
          </dl>
        </Card>
        <Card title="Chronological split" sub="Test block is touched once, at the end" className="span2">
          <div className="timeline">
            <div className="tl-train" style={{ flex: segs.train.bars }}>Train</div>
            <div className="tl-emb" style={{ flex: emb }} title="Embargo" />
            <div className="tl-val" style={{ flex: segs.val.bars }}>Validation</div>
            <div className="tl-emb" style={{ flex: emb }} title="Embargo" />
            <div className="tl-test" style={{ flex: segs.test.bars }}>Test 🔒</div>
          </div>
          <div className="grid g3" style={{ marginTop: 12 }}>
            {['train', 'val', 'test'].map((k) => (
              <div key={k}><div className="xs muted">{k === 'val' ? 'Validation' : k[0].toUpperCase() + k.slice(1)}</div>
                <div className="small"><b>{segs[k].start}</b> → <b>{segs[k].end}</b></div>
                <div className="xs muted">{segs[k].bars} decisions</div></div>
            ))}
          </div>
          <div className="xs muted" style={{ marginTop: 8 }}>{total} bars total · warm-up bars for the state window are excluded from each block. Embargo = {emb} bars between blocks.</div>
        </Card>
      </div>

      <Card title="Growth of ₹100 (log scale)" sub="Adjusted close, resampled; click legend items to hide series">
        <TimeChart data={data} series={series} log height={340} yFmt={(v) => `₹${Math.round(v)}`} />
      </Card>

      <div className="grid g2">
        <Card title="Per-asset statistics" sub="Annualised return / volatility by block">
          <div className="table-wrap">
            <table>
              <thead><tr><th>Asset</th><th>Train ret</th><th>Train vol</th><th>Val ret</th><th>Val vol</th><th>Test ret</th><th>Test vol</th></tr></thead>
              <tbody>
                {sum.stats.map((r, i) => {
                  const info = sum.assets.find((a) => a.ticker === r.ticker)
                  return (
                    <tr key={r.ticker} className="hover">
                      <td title={info?.cls}><span className="swatch" style={{ background: r.ticker === 'CASH' ? colors.neutral : colors.series[i % 8] }} />{short(r.ticker)}</td>
                      {['train', 'val', 'test'].map((k) => [
                        <td key={k + 'r'}>{pct(r[k]?.ann_return)}</td>, <td key={k + 'v'} className="muted">{pct(r[k]?.vol)}</td>])}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
        <Card title="Correlation (training block)" sub="Weekly log returns; diverging scale, grey = 0">
          <Heatmap assets={sum.corr.assets} matrix={sum.corr.matrix} />
        </Card>
      </div>
    </>
  )
}

function Heatmap({ assets, matrix }) {
  const n = assets.length
  const color = (v) => {
    // diverging: red (negative) ← grey → blue (positive)
    const a = Math.min(1, Math.abs(v))
    return v >= 0 ? `color-mix(in srgb, var(--s1) ${Math.round(a * 85)}%, var(--surface-2))`
      : `color-mix(in srgb, var(--s8) ${Math.round(a * 85)}%, var(--surface-2))`
  }
  return (
    <div className="heat" style={{ gridTemplateColumns: `90px repeat(${n}, minmax(0,1fr))` }}>
      <div />
      {assets.map((a) => <div key={a} className="lab" style={{ justifyContent: 'center' }}>{short(a).slice(0, 7)}</div>)}
      {matrix.map((row, i) => [
        <div key={'l' + i} className="lab">{short(assets[i])}</div>,
        ...row.map((v, j) => (
          <div key={i + '-' + j} className="cell" title={`${short(assets[i])} × ${short(assets[j])}: ${v.toFixed(2)}`}
            style={{ background: color(v), color: Math.abs(v) > 0.55 ? 'white' : 'var(--text-2)' }}>{v.toFixed(2)}</div>
        )),
      ])}
    </div>
  )
}

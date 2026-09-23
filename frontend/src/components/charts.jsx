import { useEffect, useMemo, useState } from 'react'
import {
  Area, AreaChart, CartesianGrid, ComposedChart, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, LabelList,
} from 'recharts'
import { fmtDate, fmtMonth, num, pct } from '../api'

// Resolve CSS variables to concrete colors (SVG + canvas friendly), re-resolving on theme change
export function useThemeColors() {
  const read = () => {
    const cs = getComputedStyle(document.documentElement)
    const g = (v) => cs.getPropertyValue(v).trim()
    return {
      series: ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'].map(g),
      neutral: g('--neutral'), muted: g('--muted'), grid: g('--grid'), text: g('--text'), text2: g('--text-2'),
      accent: g('--accent'), surface: g('--surface'), border: g('--border-strong'), good: g('--good'), bad: g('--bad'),
    }
  }
  const [c, setC] = useState(read)
  useEffect(() => {
    const upd = () => setC(read())
    window.addEventListener('themechange', upd)
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    mq.addEventListener?.('change', upd)
    return () => { window.removeEventListener('themechange', upd); mq.removeEventListener?.('change', upd) }
  }, [])
  return c
}

export function resolveColor(c, colors) {
  if (!c) return colors.neutral
  const m = /var\(--s(\d)\)/.exec(c)
  if (m) return colors.series[+m[1] - 1]
  if (c === 'var(--neutral)') return colors.neutral
  if (c === 'var(--accent)') return colors.accent
  return c
}

function TT({ active, payload, label, yFmt, labelFmt, series }) {
  if (!active || !payload || !payload.length) return null
  const rows = payload
    .filter((p) => p.value != null && !Array.isArray(p.value) && !p.dataKey?.toString().startsWith('_'))
    .sort((a, b) => b.value - a.value)
  const band = payload.find((p) => Array.isArray(p.value))
  return (
    <div className="tt">
      <div className="tt-title">{labelFmt ? labelFmt(label) : label}</div>
      {rows.slice(0, 12).map((p) => (
        <div className="tt-row" key={p.dataKey}>
          <span className="swatch" style={{ background: p.color || p.stroke }} />
          <span>{series?.find((s) => s.key === p.dataKey)?.name || p.name}</span>
          <b>{yFmt(p.value)}</b>
        </div>
      ))}
      {band && (
        <div className="tt-row"><span className="swatch" style={{ background: band.color, opacity: 0.4 }} /><span>Seed range</span>
          <b>{yFmt(band.value[0])} – {yFmt(band.value[1])}</b></div>
      )}
    </div>
  )
}

export function Legend({ series, hidden, onToggle }) {
  return (
    <div className="legend">
      {series.map((s) => (
        <span key={s.key} className={`li ${hidden?.has(s.key) ? 'off' : ''}`} onClick={() => onToggle && onToggle(s.key)}>
          <span className={s.type === 'area' ? 'dot' : 'line'} style={{ background: s.color, opacity: s.dash ? 0.7 : 1 }} />
          {s.name}
        </span>
      ))}
    </div>
  )
}

/**
 * Generic time-series chart.
 * data: [{date, [key]: value, _band: [lo, hi]}]; series: [{key, name, color, width, dash}]
 */
export function TimeChart({ data, series, height = 300, yFmt = (v) => num(v, 2), band, bandColor, refY,
  log = false, toggle = true, yDomain, xFmt = fmtMonth, showLegend = true }) {
  const colors = useThemeColors()
  const [hidden, setHidden] = useState(new Set())
  const resolved = series.map((s) => ({ ...s, color: resolveColor(s.color, colors) }))
  const visible = resolved.filter((s) => !hidden.has(s.key))
  const onToggle = toggle ? (k) => setHidden((h) => { const n = new Set(h); n.has(k) ? n.delete(k) : n.add(k); return n }) : null
  return (
    <div className="col" style={{ gap: 8 }}>
      {showLegend && <Legend series={resolved} hidden={hidden} onToggle={onToggle} />}
      <div className="chart-box" style={{ height }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke={colors.grid} />
            <XAxis dataKey="date" tickFormatter={xFmt} minTickGap={48} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={{ stroke: colors.border }} tickLine={false} />
            <YAxis tickFormatter={yFmt} width={58} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={false} tickLine={false}
              scale={log ? 'log' : 'auto'} domain={yDomain || (log ? ['auto', 'auto'] : ['auto', 'auto'])} allowDataOverflow={false} />
            <Tooltip content={<TT yFmt={yFmt} labelFmt={fmtDate} series={resolved} />} cursor={{ stroke: colors.border, strokeWidth: 1 }} />
            {refY != null && <ReferenceLine y={refY} stroke={colors.border} strokeDasharray="4 4" />}
            {band && <Area dataKey="_band" stroke="none" fill={resolveColor(bandColor || 'var(--s1)', colors)} fillOpacity={0.16} isAnimationActive={false} activeDot={false} />}
            {visible.map((s) => (
              <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={s.width || 1.6} dot={false}
                strokeDasharray={s.dash ? '5 4' : undefined} isAnimationActive={false} connectNulls
                activeDot={{ r: 4, strokeWidth: 2, stroke: colors.surface }} />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

/** Stacked weights over time. weights: [[w_asset..., w_cash]] */
export function WeightsChart({ dates, weights, assets, height = 260 }) {
  const colors = useThemeColors()
  const data = useMemo(() => dates.map((d, i) => {
    const row = { date: d }
    assets.forEach((a, j) => { row[a] = weights[i][j] })
    return row
  }), [dates, weights, assets])
  const color = (a, j) => (a === 'CASH' ? colors.neutral : colors.series[j % 8])
  const series = assets.map((a, j) => ({ key: a, name: a.replace('.NS', ''), color: color(a, j), type: 'area' }))
  return (
    <div className="col" style={{ gap: 8 }}>
      <Legend series={series} />
      <div className="chart-box" style={{ height }}>
        <ResponsiveContainer>
          <AreaChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: 0 }} stackOffset="expand">
            <CartesianGrid vertical={false} stroke={colors.grid} />
            <XAxis dataKey="date" tickFormatter={fmtMonth} minTickGap={48} tick={{ fill: colors.muted, fontSize: 11 }} tickLine={false} axisLine={{ stroke: colors.border }} />
            <YAxis tickFormatter={(v) => pct(v, 0)} width={48} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip content={<TT yFmt={(v) => pct(v, 1)} labelFmt={fmtDate} series={series} />} />
            {series.map((s) => (
              <Area key={s.key} dataKey={s.key} name={s.name} stackId="1" stroke={colors.surface} strokeWidth={0.8}
                fill={s.color} fillOpacity={0.9} isAnimationActive={false} type="stepAfter" />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

/** x = step, one line per seed */
export function StepChart({ points, yKey, height = 220, yFmt = (v) => num(v, 2), refY, refLabel }) {
  const colors = useThemeColors()
  const seeds = [...new Set(points.map((p) => p.seed))]
  const data = useMemo(() => {
    const m = new Map()
    points.forEach((p) => {
      if (p[yKey] == null) return
      const r = m.get(p.step) || { step: p.step }
      r[`s${p.seed}`] = p[yKey]
      m.set(p.step, r)
    })
    return [...m.values()].sort((a, b) => a.step - b.step)
  }, [points, yKey])
  const series = seeds.map((s, i) => ({ key: `s${s}`, name: `Seed ${s}`, color: colors.series[i % 8] }))
  return (
    <div className="col" style={{ gap: 8 }}>
      {series.length > 1 && <Legend series={series} />}
      <div className="chart-box" style={{ height }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke={colors.grid} />
            <XAxis dataKey="step" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : v)} tick={{ fill: colors.muted, fontSize: 11 }} tickLine={false} axisLine={{ stroke: colors.border }} />
            <YAxis tickFormatter={yFmt} width={52} tick={{ fill: colors.muted, fontSize: 11 }} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
            <Tooltip content={<TT yFmt={yFmt} labelFmt={(v) => `Step ${Number(v).toLocaleString()}`} series={series} />} />
            {refY != null && <ReferenceLine y={refY} stroke={colors.text2} strokeDasharray="5 4" label={{ value: refLabel, fill: colors.muted, fontSize: 11, position: 'insideTopLeft' }} />}
            {series.map((s) => (
              <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={1.8} connectNulls isAnimationActive={false}
                dot={data.length < 30 ? { r: 3, strokeWidth: 0, fill: s.color } : false} activeDot={{ r: 4 }} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

const SHORT = {
  market: 'Market', bh_ew: 'Buy & hold', ew_monthly: 'EW monthly', ew_quarterly: 'EW quarterly', ew_threshold: 'EW 5% band',
  min_variance_monthly: 'Min-var', mean_variance_monthly: 'Mean-var', risk_parity_monthly: 'Risk parity', hrp_monthly: 'HRP',
}

/** Headline: return vs max drawdown. Baselines as labelled points, RL seeds as clouds. */
export function RiskReturnScatter({ points, highlightRun, height = 420 }) {
  const colors = useThemeColors()
  const base = points.filter((p) => p.kind === 'baseline').map((p) => ({ ...p, x: p.max_drawdown, y: p.cagr }))
  const runs = [...new Set(points.filter((p) => p.kind === 'rl').map((p) => p.run_id))]
  const runColor = (r, i) => (highlightRun ? (r === highlightRun ? colors.series[0] : colors.neutral) : colors.series[[0, 1, 2][i] ?? 0])
  const TTs = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const p = payload[0].payload
    return (
      <div className="tt">
        <div className="tt-title">{p.name}{p.seed != null ? ` · seed ${p.seed}` : ''}</div>
        <div className="tt-row"><span>CAGR</span><b>{pct(p.cagr)}</b></div>
        <div className="tt-row"><span>Max drawdown</span><b>{pct(p.max_drawdown)}</b></div>
        <div className="tt-row"><span>Sharpe</span><b>{num(p.sharpe)}</b></div>
      </div>
    )
  }
  const shownRuns = highlightRun ? runs : runs.slice(0, 3)
  const legend = [
    { key: 'b', name: 'Baselines', color: colors.text2, type: 'area' },
    ...shownRuns.map((r, i) => ({ key: r, name: points.find((p) => p.run_id === r)?.name || r, color: runColor(r, i), type: 'area' })),
  ]
  return (
    <div className="col" style={{ gap: 8 }}>
      <Legend series={legend} />
      <div className="chart-box" style={{ height }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 18, left: 0 }}>
            <CartesianGrid stroke={colors.grid} />
            <XAxis type="number" dataKey="x" name="Max drawdown" tickFormatter={(v) => pct(v, 0)} tick={{ fill: colors.muted, fontSize: 11 }}
              label={{ value: 'Max drawdown  (lower is better →  left)', position: 'insideBottom', offset: -10, fill: colors.muted, fontSize: 11 }} domain={['auto', 'auto']} />
            <YAxis type="number" dataKey="y" name="CAGR" tickFormatter={(v) => pct(v, 0)} width={52} tick={{ fill: colors.muted, fontSize: 11 }} domain={['auto', 'auto']} />
            <ZAxis range={[70, 70]} />
            <Tooltip content={<TTs />} cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={base} fill={colors.text2} shape="diamond" isAnimationActive={false}>
              <LabelList dataKey="id" content={({ x, y, value }) => (
                <text x={x + 12} y={y + 4} fontSize={10.5} fill={colors.muted} style={{ pointerEvents: 'none' }}>{SHORT[value] || value}</text>
              )} />
            </Scatter>
            {runs.map((r, i) => (
              <Scatter key={r} data={points.filter((p) => p.run_id === r).map((p) => ({ ...p, x: p.max_drawdown, y: p.cagr }))}
                fill={runColor(r, i)} fillOpacity={highlightRun && r !== highlightRun ? 0.35 : 0.85} stroke={colors.surface} strokeWidth={1.5} isAnimationActive={false} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

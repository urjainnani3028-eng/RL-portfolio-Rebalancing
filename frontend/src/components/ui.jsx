import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, CheckCircle2, Info, Loader2, XCircle } from 'lucide-react'

export function Card({ title, sub, right, children, className = '', style }) {
  return (
    <section className={`card fade-in ${className}`} style={style}>
      {(title || right) && (
        <div className="card-head">
          <div>
            {title && <h3>{title}</h3>}
            {sub && <div className="sub">{sub}</div>}
          </div>
          {right && <div className="row">{right}</div>}
        </div>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, foot, icon, tone }) {
  const color = tone === 'good' ? 'var(--good)' : tone === 'bad' ? 'var(--bad)' : undefined
  return (
    <div className="card stat">
      <div className="label">{icon}{label}</div>
      <div className="value" style={{ color }}>{value}</div>
      {foot && <div className="foot">{foot}</div>}
    </div>
  )
}

export function Badge({ tone, children, title }) {
  return <span className={`badge ${tone || ''}`} title={title}>{children}</span>
}

export function Alert({ tone = 'info', children }) {
  const Icon = tone === 'warn' ? AlertTriangle : tone === 'bad' ? XCircle : tone === 'good' ? CheckCircle2 : Info
  return <div className={`alert ${tone}`}><Icon size={16} /><div>{children}</div></div>
}

export function Button({ children, onClick, kind, size, disabled, loading, icon, title, type }) {
  return (
    <button type={type || 'button'} className={`btn ${kind || ''} ${size || ''}`} onClick={onClick} disabled={disabled || loading} title={title}>
      {loading ? <Loader2 size={15} className="spin" /> : icon}
      {children}
    </button>
  )
}

export function Segmented({ value, onChange, options }) {
  return (
    <div className="seg" role="tablist">
      {options.map((o) => (
        <button key={o.value} className={value === o.value ? 'on' : ''} onClick={() => onChange(o.value)} role="tab" aria-selected={value === o.value}>
          {o.icon}{o.label}
        </button>
      ))}
    </div>
  )
}

export function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <div className="hint">{hint}</div>}
    </label>
  )
}

export function NumInput({ value, onChange, step = 'any', min, max }) {
  const [txt, setTxt] = useState(String(value ?? ''))
  useEffect(() => { setTxt(String(value ?? '')) }, [value])
  return (
    <input type="number" value={txt} step={step} min={min} max={max}
      onChange={(e) => { setTxt(e.target.value); const v = parseFloat(e.target.value); if (!isNaN(v)) onChange(v) }} />
  )
}

export function Progress({ value }) {
  return <div className="progress"><div style={{ width: `${Math.round((value || 0) * 100)}%` }} /></div>
}

export function Empty({ icon, title, children }) {
  return (
    <div className="empty">
      {icon}
      <div style={{ fontWeight: 600, color: 'var(--text-2)' }}>{title}</div>
      <div className="small">{children}</div>
    </div>
  )
}

export function Modal({ open, onClose, title, children, actions }) {
  if (!open) return null
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        <div className="t2">{children}</div>
        <div className="row" style={{ justifyContent: 'flex-end' }}>{actions}</div>
      </div>
    </div>
  )
}

/** Sortable table. columns: [{key, label, fmt, align, get, sortable}] */
export function DataTable({ columns, rows, initialSort, rowKey, highlight, onRowClick }) {
  const [sort, setSort] = useState(initialSort || null)
  const sorted = useMemo(() => {
    if (!sort) return rows
    const col = columns.find((c) => c.key === sort.key)
    const get = col?.get || ((r) => r[sort.key])
    return [...rows].sort((a, b) => {
      const va = get(a), vb = get(b)
      if (va == null) return 1
      if (vb == null) return -1
      return (va > vb ? 1 : va < vb ? -1 : 0) * (sort.dir === 'asc' ? 1 : -1)
    })
  }, [rows, sort, columns])
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.sortable === false ? '' : 'sortable'} title={c.title}
                onClick={() => c.sortable !== false && setSort((s) => ({ key: c.key, dir: s?.key === c.key && s.dir === 'desc' ? 'asc' : 'desc' }))}>
                {c.label}{sort?.key === c.key ? (sort.dir === 'desc' ? ' ↓' : ' ↑') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={rowKey ? rowKey(r) : i} className={`hover ${highlight && highlight(r) ? 'hl' : ''}`} onClick={() => onRowClick && onRowClick(r)} style={onRowClick ? { cursor: 'pointer' } : undefined}>
              {columns.map((c) => {
                const v = c.get ? c.get(r) : r[c.key]
                return <td key={c.key}>{c.render ? c.render(r) : c.fmt ? c.fmt(v) : v}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Tiny global toast system
let pushToast = () => {}
export function toast(msg, kind = 'info') { pushToast({ msg, kind, id: Math.random() }) }
export function Toasts() {
  const [items, setItems] = useState([])
  useEffect(() => {
    pushToast = (t) => {
      setItems((x) => [...x, t])
      setTimeout(() => setItems((x) => x.filter((y) => y.id !== t.id)), t.kind === 'err' ? 7000 : 3500)
    }
  }, [])
  return (
    <div className="toast-wrap">
      {items.map((t) => (
        <div key={t.id} className={`toast fade-in ${t.kind === 'err' ? 'err' : ''}`}>
          {t.kind === 'err' ? <XCircle size={16} color="var(--bad)" /> : <CheckCircle2 size={16} color="var(--good)" />}
          <div>{t.msg}</div>
        </div>
      ))}
    </div>
  )
}

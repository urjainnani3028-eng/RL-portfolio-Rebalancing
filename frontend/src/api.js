async function req(method, path, body) {
  const opts = { method, headers: {} }
  if (body instanceof FormData) opts.body = body
  else if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json'
    opts.body = JSON.stringify(body)
  }
  let res
  try {
    res = await fetch(`/api${path}`, opts)
  } catch (e) {
    throw new Error('Cannot reach the backend. Is it running on port 8000?')
  }
  const text = await res.text()
  let data = null
  try { data = text ? JSON.parse(text) : null } catch { data = text }
  if (!res.ok) throw new Error((data && data.detail) || `HTTP ${res.status}`)
  return data
}

export const api = {
  get: (p) => req('GET', p),
  post: (p, b) => req('POST', p, b ?? {}),
  put: (p, b) => req('PUT', p, b),
  del: (p) => req('DELETE', p),
  upload: (p, file) => { const f = new FormData(); f.append('file', file); return req('POST', p, f) },
}

// ---------- formatting ----------
const isNum = (x) => typeof x === 'number' && isFinite(x)
export const pct = (x, d = 1) => (isNum(x) ? `${(x * 100).toFixed(d)}%` : '—')
export const num = (x, d = 2) => (isNum(x) ? x.toFixed(d) : '—')
export const signed = (x, d = 2) => (isNum(x) ? `${x >= 0 ? '+' : ''}${x.toFixed(d)}` : '—')
export const bps = (x) => (isNum(x) ? `${(x * 1e4).toFixed(1)} bps` : '—')
export const short = (t) => (t || '').replace('.NS', '')
export const fmtDate = (s) => {
  if (!s) return ''
  const d = new Date(s)
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}
export const fmtMonth = (s) => {
  if (!s) return ''
  const d = new Date(s)
  return d.toLocaleDateString(undefined, { year: '2-digit', month: 'short' })
}

// Categorical palette in fixed order (validated set), CSS variables so dark mode swaps automatically
export const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)', 'var(--s7)', 'var(--s8)']
export const NEUTRAL = 'var(--neutral)'
export function assetColor(asset, assets) {
  if (asset === 'CASH') return NEUTRAL
  const i = assets.filter((a) => a !== 'CASH').indexOf(asset)
  return i >= 0 && i < SERIES.length ? SERIES[i] : NEUTRAL
}

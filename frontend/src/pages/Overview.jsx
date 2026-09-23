import { ArrowRight, Check, Target, Trophy, Database, Layers, X } from 'lucide-react'
import { num } from '../api'
import { Badge, Button, Card, Stat } from '../components/ui'

const TEAM = [['SA', 'Surbhi Agarwal', 'J002'], ['UI', 'Urja Innani', 'J018'], ['TL', 'Taha Lokhandwala', 'J024']]

const FLOW = [
  ['Observe', 'Price window, previous weights w(t−1), risk block — data up to t−1 only.'],
  ['Decide', 'Hybrid: optimiser proposes a target; the agent picks how far to move (0–100%).'],
  ['Pay cost', 'Turnover × India cost model (STT, stamp duty, GST, fees, slippage) — before the move.'],
  ['Market moves', 'Next-bar returns hit the new weights, which then drift with prices.'],
  ['Reward', 'Log net return − λ · risk penalty (downside / variance / drawdown).'],
]

export default function Overview({ overview, go }) {
  const ds = overview?.dataset
  const san = overview?.sanity || {}
  const fast = san.fast, slow = san.slow
  const allTests = [...(fast || []), ...(slow || [])]
  const sanityDone = fast && slow
  const sanityOk = sanityDone && allTests.every((t) => t.passed)
  const steps = [
    { n: 1, label: 'Fix the universe & frequency, download and freeze data', done: !!ds, page: 'data', detail: ds ? `${ds.tickers.length} ETFs · snapshot #${ds.sha}` : 'Not prepared yet' },
    { n: 2, label: 'Compute baselines net of costs — equal-weight quarterly is the bar to beat', done: overview?.bar_to_beat != null, page: 'baselines', detail: overview?.bar_to_beat != null ? `EW quarterly val Sharpe ${num(overview.bar_to_beat)}` : '' },
    { n: 3, label: 'Environment passes all five sanity tests', done: sanityOk, fail: sanityDone && !sanityOk, page: 'sanity', detail: allTests.length ? `${allTests.filter((t) => t.passed).length}/${allTests.length} run tests passing` : 'Not run yet' },
    { n: 4, label: 'Train PPO (hybrid) across seeds, select checkpoints on validation', done: (overview?.runs || 0) > 0, page: 'train', detail: `${overview?.runs || 0} run(s)` },
    { n: 5, label: 'Seeded comparison, cost sweep, regimes — then touch the test block once', done: false, page: 'robustness', detail: '' },
  ]
  const best = overview?.best
  const bar = overview?.bar_to_beat
  return (
    <>
      <div className="hero fade-in">
        <div className="row"><Badge tone="accent">Reinforcement Learning · Portfolio Management</Badge><Badge>Indian ETFs</Badge><Badge>Hybrid optimiser + RL</Badge></div>
        <h1>Reinforcement Learning for Portfolio Rebalancing</h1>
        <p>
          A portfolio manager modelled as an RL agent whose action sets the next period's weights and whose reward is log
          return net of transaction costs minus a risk penalty. The question is not “does AI beat Markowitz” — it is whether
          a learned policy can time and size the adjustment better than fixed calendar / threshold rules, out of sample.
        </p>
        <div className="chips">
          {TEAM.map(([av, n, id]) => <span className="chip" key={id}><span className="av">{av}</span>{n} <span className="muted">{id}</span></span>)}
        </div>
      </div>

      <div className="grid g4">
        <Stat icon={<Database size={13} />} label="Dataset" value={ds ? `${ds.bars} bars` : '—'}
          foot={ds ? `${ds.source === 'synthetic' ? 'Synthetic (offline)' : 'Yahoo Finance'} · ${ds.start.slice(0, 4)}–${ds.end.slice(0, 4)}` : 'Prepare data to begin'} />
        <Stat icon={<Target size={13} />} label="Bar to beat (val Sharpe)" value={num(bar)} foot="Equal-weight, quarterly rebalance, net of costs" />
        <Stat icon={<Trophy size={13} />} label="Best RL run (median val Sharpe)" value={num(best?.sharpe)}
          tone={best && bar != null ? (best.sharpe > bar ? 'good' : 'bad') : undefined}
          foot={best ? best.name : 'No runs yet'} />
        <Stat icon={<Layers size={13} />} label="Experiments" value={overview?.runs ?? 0} foot="Each run = one config × many seeds" />
      </div>

      <div className="grid g2">
        <Card title="Project plan" sub="Baselines first, so there is a complete result even if RL fails.">
          <div className="checklist">
            {steps.map((s) => (
              <div className="check-item" key={s.n} onClick={() => go(s.page)}>
                <div className={`check-dot ${s.done ? 'done' : s.fail ? 'fail' : ''}`}>{s.done ? <Check size={14} /> : s.fail ? <X size={14} /> : s.n}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 550 }}>{s.label}</div>
                  {s.detail && <div className="xs muted">{s.detail}</div>}
                </div>
                <ArrowRight size={15} color="var(--muted)" />
              </div>
            ))}
          </div>
        </Card>
        <Card title="Why RL — and the honest counter-case">
          <div className="col small t2" style={{ gap: 10 }}>
            <p><b style={{ color: 'var(--text)' }}>For:</b> mean-variance is single-period and cost-blind — it says where to be, not how to get there
              while paying for every trade. RL optimises cumulative reward over a trajectory, so multi-period cost amortisation
              falls out of the objective, and it never needs expected-return estimates (the unstable Markowitz input).</p>
            <p><b style={{ color: 'var(--text)' }}>Against:</b> financial data is short, noisy and non-stationary; RL is seed-sensitive and easy to
              overfit. So every claim here is tested against baselines net of identical costs, across ≥10 seeds, with a deflated
              Sharpe that accounts for the number of configurations tried.</p>
            <p><b style={{ color: 'var(--text)' }}>Novelty:</b> the hybrid design — a classical optimiser proposes the target, the RL agent
              only decides the timing and size of the move toward it. A small, interpretable action space that is hard to reward-hack.</p>
          </div>
        </Card>
      </div>

      <Card title="One environment step" sub="Order matters: costs are charged before the market moves, otherwise the trade pays for itself.">
        <div className="flow">
          {FLOW.map(([t, d], i) => (
            <div className="fstep" key={t}><span className="n">0{i + 1}</span><b>{t}</b><p>{d}</p></div>
          ))}
        </div>
      </Card>

      {!ds && (
        <div className="row"><Button kind="primary" onClick={() => go('data')} icon={<Database size={15} />}>Start: prepare the data</Button></div>
      )}
    </>
  )
}

# RL Rebalancing Lab — Reinforcement Learning for Portfolio Rebalancing

**Team:** Surbhi Agarwal (J002) · Urja Innani (J018) · Muskaan (J059)

A full-stack research app that models a portfolio manager as an RL agent over **Indian ETFs (NSE)**.
The agent's action sets the next period's weights; its reward is **log return net of transaction costs
minus a risk penalty**. We test whether the learned policy beats fixed calendar / threshold rebalancing
rules **out of sample**, net of identical costs.

**Headline novelty — Hybrid optimiser + RL:** a classical optimiser (risk parity / min-variance / HRP /
mean-variance) proposes the target portfolio `w*`, and the RL agent only decides **how far to move toward it**
each bar: `w ← w + a·(w* − w)`, `a ∈ [0, 1]`. The action space stays small and interpretable, and it is hard to reward-hack.
A "direct" mode (the agent outputs all weights) is included for comparison.

---

## 1. Quick start (one command)

Requirements: **Python 3.10–3.12** and (recommended) **Node.js 18+**. Internet access is needed the first time you run it, to install packages and download prices.

| OS | Command |
|---|---|
| Windows | double-click **`run.bat`** (or run it from a terminal) |
| macOS / Linux | `chmod +x run.sh && ./run.sh` |

Then open **http://localhost:8000**.

The first run creates `.venv/`, installs the Python packages (PyTorch is large, so this takes a few minutes),
builds the React frontend and starts the server. Later runs start in seconds. If Node isn't installed, the
prebuilt frontend in `frontend/dist` is used.

> On Linux, to avoid the large CUDA build of PyTorch you can install the CPU build first:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Development mode (hot reload)
```bash
./dev.sh             # backend :8000 with --reload  +  Vite dev server :5173
```
On Windows, use two terminals:
```bat
cd backend && ..\.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
cd frontend && npm run dev          (open http://localhost:5173)
```

### Run the tests
```bash
cd backend && ../.venv/bin/python -m pytest -q      # works offline (uses the synthetic market)
```

---

## 2. Walkthrough (matches the project plan)

1. **Data & universe**: pick ETFs, weekly/daily bars and the split dates, then click **Prepare data**. Prices are
   downloaded from Yahoo Finance (`.NS` tickers), unadjusted unit splits are repaired automatically, and the
   dataset is **frozen** to a Parquet snapshot with a content hash, so every experiment uses identical data.
   *Offline?* The app falls back to a regime-switching **synthetic** market, clearly labelled, so the pipeline still runs. Use **Refresh from Yahoo** once you're online.
2. **Costs & baselines**: the India cost model (STT, stamp duty, NSE and SEBI fees, GST, slippage) or a flat bps cost.
   All baselines run net of the same costs. **Equal-weight, quarterly** is shown as the *bar to beat*.
3. **Sanity tests**: all five from the brief (see §4). Run these before trusting any result.
4. **Train agent**: hybrid or direct, PPO/A2C/SAC/TD3/DQN, MLP/EIIE/attention, reward design, seeds.
   Live charts show validation Sharpe and episode reward. Each seed keeps its **best-on-validation** checkpoint.
   There is also a *purged walk-forward* option that retrains every year and stitches together an out-of-sample curve.
5. **Results**: median + IQR across seeds, per-seed table with **PSR / Deflated Sharpe**, equity and drawdown charts vs
   baselines, weights over time and the agent's move fraction `a_t`. The **test block is locked** until you
   click *Evaluate on test (one-shot)*, and each unlock is timestamped.
6. **Robustness**: the headline **return vs max-drawdown scatter** (baselines as points, RL as seed clouds), a **cost sweep of
   0–50 bps**, and **regime-sliced** performance (bull/bear, high/low vol, plus named Indian market events on real data).

Suggested first session: Prepare data → Baselines → run all sanity tests → Train with the **Quick demo** preset →
Results → Robustness. For the report, use the **Full study** preset (≥10 seeds).

---

## 3. MDP formulation (as implemented)

| Element | Implementation (`backend/app/sim.py`) |
|---|---|
| **State** | per asset: standardised log-return window (L bars), rolling vol, current weight, and in hybrid mode the target weight and the gap to it · global: average correlation, cash weight, drawdown so far, realised portfolio vol, bars since last trade, and in hybrid mode the target cash and distance to target |
| **Action** | *Hybrid*: scalar `a ∈ [0,1]` (continuous) or {0, .25, .5, .75, 1} (DQN) · *Direct*: logits → softmax on the simplex |
| **Constraints** | long-only, sum-to-one and a per-ETF cap (default 40%), all enforced structurally by a capped-simplex projection |
| **Step order** | decide → **charge cost on turnover** → **then** the market moves (next bar) → weights **drift** → reward |
| **Execution** | state uses data up to *t−1* (execution at the next bar, never on the decision close) |
| **Reward** | `scale · [ log((1−cost)·growth) − λ·risk − κ·(HHI − 1/N)⁺ − τ·turnover ]`, risk ∈ {downside², variance, drawdown increase, none} |
| **Episodes** | random start inside the train block, 104 bars by default, randomised initial weights |

**Leakage controls:** scalers are fit on train only · optimiser targets use data up to t−1 · chronological splits with an embargo ·
a unit test corrupts every "future" return and asserts the observation doesn't change (plus a negative control).

## 4. Sanity tests (`backend/app/sanity.py`)

1. Zero-cost buy-and-hold matches a hand calculation (to 1e-9).
2. Forced rotation pays exactly turnover × cost, which proves costs are charged before the move.
3. A 1/n action reproduces an independently coded equal-weight loop.
4. **Shuffled returns are not profitable**: PPO trained on time-shuffled data, evaluated on 8 fresh shuffles, has no timing edge over a constant mix of its own average weights.
5. **Injected look-ahead improves results sharply**: with next-bar returns leaked into the state, PPO's timing edge is large, which shows the learner can find signal when it exists.

## 5. Baselines & metrics

Buy-and-hold · Equal-weight (monthly / quarterly / 5% threshold) · rolling Min-variance, Mean-variance, Risk parity (ERC), HRP ·
Market index (NIFTYBEES) · for hybrid runs, the agent's own optimiser applied mechanically (every bar / monthly).

Metrics: net Sharpe (headline), Sortino, Calmar, CAGR, volatility, max drawdown, annual turnover, total cost, Herfindahl,
average cash, CAPM alpha/beta vs NIFTYBEES, **Probabilistic and Deflated Sharpe** (the number of trials counts every seed of every run on that block, so keep failed runs).

## 6. Cost model (India, delivery via a discount broker)

Defaults (edit them on the *Costs & baselines* page): brokerage 0 · **STT 0.001% on the sale of equity-ETF units** (none on gold or debt ETFs) ·
**stamp duty 0.015% on buys** · NSE transaction charge 0.00307% · SEBI fee ₹10/crore · **GST 18%** on brokerage + exchange + SEBI fees ·
plus slippage / half-spread (default 5 bps). Check these against your broker's current schedule before you report.

## 7. Project layout

```
backend/
  app/config.py      universe catalogue, default settings (pydantic)
  app/data.py        Yahoo download, split repair, synthetic fallback, frozen snapshots, CSV import, splits
  app/costs.py       flat and India itemised cost models
  app/optimizers.py  min-var, mean-var, risk parity (ERC), HRP, capped-simplex projection
  app/sim.py         portfolio simulator + Gymnasium environment (hybrid and direct)
  app/baselines.py   rule-based strategies on the same simulator
  app/metrics.py     Sharpe/Sortino/Calmar/..., PSR & deflated Sharpe
  app/agents.py      SB3 PPO/A2C/SAC/TD3/DQN, EIIE and attention feature extractors, training loop
  app/runs.py        multi-seed jobs, walk-forward, one-shot test, results assembly
  app/analysis.py    cost sweep, regimes, risk-return scatter
  app/jobs.py        background job queue with live progress
  app/main.py        FastAPI REST API (+ serves the built frontend)
  app/nnbandit/      RNN/LSTM/GRU/TCN forecasters + pretrained TimesFM, bandit algorithms, study runner, CLI, report
  tests/             pytest: sanity tests, cap projection, cost asymmetry, no-look-ahead
  storage/           (created at runtime) datasets, caches, runs, models
frontend/            React + Vite + Recharts dashboard
```

REST API docs are served at **http://localhost:8000/docs**.

### CSV import format
`Date,TICKER1,TICKER2,...` with daily or weekly adjusted closes. A cash sleeve is added automatically.

## 8. Troubleshooting

* **"Synthetic data" badge although you're online**: Yahoo may be rate-limiting you. Wait a minute, then click *Refresh from Yahoo*.
* **Late-listed ETFs shorten the history**: the loader aligns every series on the latest first date. The default 7 ETFs start in 2011.
* **Slow training**: SAC/TD3 are ~5× slower per step than PPO. Start with PPO and the Quick demo preset.
* **Port in use**: change `--port 8000` in `run.sh`/`run.bat` (and the proxy target in `frontend/vite.config.js` for dev mode).

## 9. NNDL module: five neural forecasters + a multi-armed bandit (`backend/app/nnbandit/`)

**Question:** which of five sequence models best forecasts next-week ETF returns for rebalancing, and can a
multi-armed bandit find it without spending equal compute on every model?

### Run it
```bash
cd backend
../.venv/bin/python -m app.nnbandit.cli                 # full study, test block stays locked (~15-25 min on CPU)
../.venv/bin/python -m app.nnbandit.cli --quick         # smoke run (~3 min)
../.venv/bin/python -m app.nnbandit.cli --test          # + one-shot evaluation on 2022-present
../.venv/bin/python -m app.nnbandit.cli --algorithm ucb1_tuned --budget 60 --max-seeds 10
../.venv/bin/python -m app.nnbandit.cli --timesfm-version 3.0       # TimesFM 3.0 instead of 2.5
../.venv/bin/python -m app.nnbandit.cli --models rnn lstm gru tcn patchtst   # offline: PatchTST instead of TimesFM
```
TimesFM's weights (~800 MB for 2.5) download from Hugging Face on the first run and are cached in `~/.cache/huggingface`.
With no internet, point `--timesfm-checkpoint` at a downloaded copy, or use the PatchTST line above.
It writes `storage/nnbandit/report/report.md` plus PNG charts, and saves JSON results in `storage/nnbandit/results/`.
Trained models are cached, so re-runs and other bandit algorithms reuse them. In the web app, the same study runs from
**Model bandit** (step 7).

### The five arms
The four trained models share one budget: 2 hidden layers × 64 units, dropout 0.1, Adam + L2 weight decay, early stopping.
TimesFM is used as Google released it, with no training on our data.

| Arm | Architecture | Source |
|---|---|---|
| `rnn` | Elman RNN (tanh) | `torch.nn.RNN` |
| `lstm` | LSTM: input/forget/output gates + cell state | `torch.nn.LSTM` |
| `gru` | GRU: update/reset gates | `torch.nn.GRU` |
| `tcn` | Temporal Convolutional Network: causal dilated residual convolutions | open source: Bai et al. 2018, [locuslab/TCN](https://github.com/locuslab/TCN) |
| `timesfm` | TimesFM: pretrained decoder-only time-series foundation model (200M params), **zero-shot** | open source: Das et al. ICML 2024, [google-research/timesfm](https://github.com/google-research/timesfm) |
| `patchtst` *(optional)* | PatchTST: patching + channel-independent Transformer encoder | open source: Nie et al. ICLR 2023, [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) |

**How TimesFM is used** (`foundation.py`): for each ETF and decision week *t*, it reads that ETF's standardised weekly returns up
to *t−1* (up to 256 weeks) and forecasts week *t+1*, so it sees exactly the same information set as the trained models. Settings
that matter: `infer_is_positive=False`, because returns can be negative and must not be clamped at zero; and it forecasts returns,
not prices, because a price forecast "wins" on error just by repeating last week's price. The model is deterministic, so a single
inference pass serves every bandit pull, and its cost in the compute comparison is inference time, not training time. Version 2.5
weights are Apache-2.0; 3.0 weights are licensed for non-commercial use only, which is fine for coursework.

### Pipeline
1. **Features** (`features.py`): for each ETF and decision week *t*, a 20-week window ending at *t−1* holds the standardised return,
   26-week volatility, average correlation, cross-sectional mean return and a one-hot asset id. The target is the standardised
   return of week *t+1*. Scalers are fit on train only, and a unit test checks that no future bar enters a window.
2. **Training** (`training.py`): Huber loss, gradient clipping, early stopping on an *inner* validation slice (the last 52 weeks of
   train, after a 4-week embargo), so the outer validation block stays clean for the bandit.
3. **Portfolio** (`portfolio.py`): one rule for every model, `w = softmax(5 · forecast)`, capped at 40% per ETF, with trades skipped
   below a 2% drift. It runs through the same simulator and Indian cost model as the RL agent and the baselines.
4. **Bandit** (`bandit.py`): uniform, ε-greedy, UCB1, UCB1-Tuned, Thompson sampling (Normal-Gamma), and successive halving.
   A *pull* trains the next seed of the chosen model (cached) and scores it on a random 26-week validation window
   (for TimesFM, it re-scores its single forecast on a new window).
   **Reward** = 0.5·Sharpe + 0.3·directional accuracy + 0.2·drawdown, each mapped to [0, 1]. Sharpe and drawdown are scored
   relative to equal-weight quarterly on the *same* window, so every arm faces the same market and window noise cancels out.
   Directional accuracy is scored relative to always predicting "up" on that window, so riding a rising market earns
   no credit (`accuracy_reference="none"` restores the raw hit rate used in the first version).
5. **Evaluation** (`experiment.py`): (a) live bandit run → its pick; (b) every model × every seed × every window → the true best arm;
   (c) each algorithm replayed 300× on that reward table → P(correct pick) and simple regret vs pulls, compared with uniform testing;
   (d) per-model forecast metrics (directional accuracy, MAE, rank IC), portfolio metrics, Diebold-Mariano tests,
   learning curves, gradient-by-lag "memory" profile, and regime slices; (e) `--test`: the locked 2022+ block, one-shot.

### Reading the result
* **Which model is better** = the highest true mean reward on validation, confirmed on the test block. The DM test tells you whether its
  forecast-error advantage is statistically real, and the seed IQR tells you whether it is robust to initialisation.
* **Why** comes from the diagnostics in the report. The lag-saliency chart shows how far back each model actually looks (a vanilla RNN's
  gradient decays with lag; LSTM/GRU gating and TCN dilations keep older weeks in play). Turnover shows how stable each model's
  forecasts are, which matters because costs are charged weekly. Train-time and parameter counts give the compute side.
* **Pretrained vs trained from scratch:** TimesFM vs the other four tests whether broad pretraining beats a small model fit to these
  ETFs. TimesFM forecasts each ETF on its own, so it can't use the cross-asset volatility/correlation features the others get.
* With ~400 weekly training samples per ETF the signal is weak. Expect directional accuracy near 50-55% and best epochs early. A
  model that "wins" by a tiny gap is not meaningfully better, and the report prints the gap and the DM p-values so you can say so.

## 10. Key references
Markowitz (1952) · Jiang, Xu & Liang (2017) EIIE, arXiv:1706.10059 · Lim, Cao & Quek (2022), NCAA 34 ·
Wang & Zhou (2020), Math. Finance · Wang et al. (2021) DeepTrader, AAAI · López de Prado (2016) HRP ·
Bailey & López de Prado, The Deflated Sharpe Ratio · Sun et al. (2021) survey, arXiv:2109.13851 ·
Hochreiter & Schmidhuber (1997) LSTM · Cho et al. (2014) GRU · Bai, Kolter & Koltun (2018) TCN, arXiv:1803.01271 ·
Nie et al. (2023) PatchTST, ICLR · Das et al. (2024) TimesFM, ICML, arXiv:2310.10688 · Auer et al. (2002) UCB1, Mach. Learn. 47 · Agrawal & Goyal (2012) Thompson sampling ·
Karnin, Koren & Somekh (2013) successive halving, ICML · Diebold & Mariano (1995) forecast-accuracy test.

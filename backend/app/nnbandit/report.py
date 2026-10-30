"""Markdown report + PNG charts from a study result (for the written project report)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

COLORS = {"rnn": "#9aa0a6", "lstm": "#1a73e8", "gru": "#188038", "tcn": "#e8710a", "patchtst": "#a142f4", "timesfm": "#d93025"}
ALGO_COLORS = {"uniform": "#5f6368", "epsilon_greedy": "#f9ab00", "ucb1": "#1a73e8", "ucb1_tuned": "#12b5cb",
               "thompson": "#d93025", "successive_halving": "#188038"}


def _pct(x, nd=1):
    return "–" if x is None else f"{100 * x:.{nd}f}%"


def _num(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def _med(row, k):
    return row["seeds"].get(k, {}).get("median")


def _iqr(row, k, fmt=_num):
    s = row["seeds"].get(k)
    return "–" if not s else f"{fmt(s['median'])} [{fmt(s['q1'])}, {fmt(s['q3'])}]"


def _charts(res: dict, out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    models = res["settings"]["models"]
    label = {m: res["models"][m]["label"] for m in models}
    val = res["validation"]["models"]

    # 1. bandit efficiency: P(correct) vs pulls
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, a in res["replay"]["algorithms"].items():
        x = np.arange(1, len(a["p_correct"]) + 1)
        ax[0].plot(x, a["p_correct"], label=name, color=ALGO_COLORS.get(name), lw=2 if name == "uniform" else 1.5,
                   ls="--" if name == "uniform" else "-")
        ax[1].plot(x, a["simple_regret"], color=ALGO_COLORS.get(name), ls="--" if name == "uniform" else "-")
    ax[0].set(title="P(bandit recommends the true best model)", xlabel="pulls", ylim=(0, 1.02))
    ax[1].set(title="Simple regret of the recommendation", xlabel="pulls")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "bandit_efficiency.png", dpi=130)
    plt.close(fig)
    files.append("bandit_efficiency.png")

    # 2. live bandit: cumulative pulls per arm
    log = res["live"]["log"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for m in models:
        c = np.cumsum([e["arm"] == m for e in log])
        ax.plot(np.arange(1, len(c) + 1), c, label=label[m], color=COLORS.get(m))
    ax.set(title=f"Live {res['live']['algorithm']}: pulls allocated to each model", xlabel="pull",
           ylabel="cumulative pulls")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "live_allocation.png", dpi=130)
    plt.close(fig)
    files.append("live_allocation.png")

    # 3. learning curves + lag saliency
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for m in models:
        ax[0].plot(val[m]["train_loss_curve"], color=COLORS.get(m), label=label[m])
        ax[1].plot(val[m]["val_loss_curve"], color=COLORS.get(m), label=label[m])
        sal = val[m]["lag_saliency"]
        if all(x is None for x in sal):          # pretrained arm: no gradients through our data
            continue
        ax[2].plot(np.arange(-len(sal) + 1, 1), sal, color=COLORS.get(m), label=label[m], marker=".")
    ax[0].set(title="Training loss (Huber), mean of seeds", xlabel="epoch")
    ax[1].set(title="Inner-validation loss (early stopping)", xlabel="epoch")
    ax[2].set(title="Gradient saliency by lag (memory)", xlabel="lag (bars, 0 = most recent)")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "learning_and_memory.png", dpi=130)
    plt.close(fig)
    files.append("learning_and_memory.png")

    # 4. equity curves
    for seg_name, seg in (("validation", res["validation"]), ("test", res["test"])):
        if seg.get("locked", False) or "equity_curves" not in seg:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        for k, c in seg["equity_curves"].items():
            nm = label.get(k) or seg["baselines"].get(k, {}).get("name", k)
            ax.plot(np.array(c["dates"], dtype="datetime64[D]"), c["value"], color=COLORS.get(k, None),
                    ls="-" if k in label else ":", lw=1.6 if k in label else 1.4, label=nm)
        ax.set(title=f"{seg_name.capitalize()}: seed-ensemble portfolios vs baselines (net of costs)",
               ylabel="growth of 1")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"equity_{seg_name}.png", dpi=130)
        plt.close(fig)
        files.append(f"equity_{seg_name}.png")
    return files


def _model_table(seg: dict, models: list[str], lines: list[str]):
    lines += ["| Model | Sharpe median [IQR] | CAGR | Max DD | Turnover/yr | Dir. acc. | MAE (z) | Rank IC | "
              "Seeds with Sharpe>0 |", "|---|---|---|---|---|---|---|---|---|"]
    for m in seg["ranking_by_sharpe"]:
        r = seg["models"][m]
        lines.append(f"| **{r['label']}** | {_iqr(r, 'sharpe')} | {_pct(_med(r, 'cagr'))} | "
                     f"{_pct(_med(r, 'max_drawdown'))} | {_num(_med(r, 'turnover_ann'))} | {_pct(_med(r, 'dir_acc'))} | "
                     f"{_num(_med(r, 'mae'), 3)} | {_num(_med(r, 'ic'), 3)} | {_pct(r['positive_sharpe_share'], 0)} |")
    lines += ["", "Baselines (same simulator, same costs):", "", "| Strategy | Sharpe | CAGR | Max DD | Turnover/yr |",
              "|---|---|---|---|---|"]
    for v in seg["baselines"].values():
        lines.append(f"| {v['name']} | {_num(v['sharpe'])} | {_pct(v['cagr'])} | {_pct(v['max_drawdown'])} | "
                     f"{_num(v['turnover_ann'])} |")
    dm = seg["diebold_mariano_vs_top"]
    lines += ["", f"Diebold–Mariano tests of the top model's forecast errors (**{seg['models'][dm['top']]['label']}**) "
                  "against each other model (absolute-error loss; negative stat = top model is more accurate):", "",
              "| vs | DM stat | p-value |", "|---|---|---|"]
    for m, t in dm["tests"].items():
        lines.append(f"| {seg['models'][m]['label']} | {_num(t['stat'])} | {_num(t['p_value'], 3)} |")


def write_report(res: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    files = _charts(res, out)
    st, live, rep, val = res["settings"], res["live"], res["replay"], res["validation"]
    models = st["models"]
    lab = {m: res["models"][m]["label"] for m in models}
    ds = res["dataset"]
    L = [f"# Model selection with a multi-armed bandit — run {res['id']}", ""]
    if ds.get("source") == "synthetic":
        L += ["> **Synthetic data.** Yahoo Finance was unreachable, so this run uses the app's regime-switching "
              "synthetic market. Use it to check the pipeline, not to draw conclusions about real ETFs.", ""]
    L += [f"Dataset `{ds['id']}` ({ds['source']}), {ds['start']} → {ds['end']}, {ds['bars']} weekly bars · "
          f"train ≤ {ds['train_end']} · validation ≤ {ds['val_end']} · test after that.", "",
          "## 1. Bandit result", "",
          f"* Live algorithm: **{live['algorithm']}**, budget {live['budget']} pulls "
          f"(pull = train the next seed of a model and score it on a random {st['window']}-week validation window).",
          f"* Bandit's pick: **{lab[live['winner']]}** · exhaustive ground truth "
          f"({st['max_seeds']} seeds × every window): **{lab[rep['true_best']]}** "
          f"→ {'correct' if live['correct'] else 'not the true best'}.",
          f"* Reward = {', '.join(f'{v:g}×{k}' for k, v in st['reward_weights'].items())} "
          "(each mapped to [0, 1]"
          + (f"; Sharpe and drawdown scored relative to `{st['reward_reference']}` on the same window)."
             if st.get("reward_reference", "none") != "none" else ")."),
          f"* Accuracy is scored {'against always predicting “up” on the same window' if st.get('accuracy_reference') == 'always_up' else 'as the raw hit rate'}.",
          "",
          "| Model | Live pulls | Live mean reward | Trainings | True mean reward ± sd |", "|---|---|---|---|---|"]
    for m in models:
        L.append(f"| {lab[m]} | {live['pulls'][m]} | {_num(live['mean_reward'][m], 3)} | {live['trainings'][m]} | "
                 f"{_num(rep['true_mean_reward'][m], 3)} ± {_num(rep['true_std_reward'][m], 3)} |")
    c = live["compute"]
    L += ["", f"Compute: the bandit trained {c['bandit_trainings']} models "
              f"(~{c['bandit_train_seconds_est']} s) where uniform testing with the same budget trains "
              f"{c['uniform_trainings']} (~{c['uniform_train_seconds_est']} s); exhaustive = {c['exhaustive_trainings']}.",
          "", "### Bandit efficiency (replayed "
              f"{st['replay_runs']}× on the precomputed reward table)", "",
          "| Algorithm | P(correct) after budget | Pulls to reach 90% | Avg. pulls on true best |", "|---|---|---|---|"]
    for name, a in sorted(rep["algorithms"].items(), key=lambda kv: -(kv[1]["final_p_correct"] or 0)):
        L.append(f"| {name} | {_pct(a['final_p_correct'])} | {a['pulls_to_90pct'] or 'not reached'} | "
                 f"{_num(a['avg_pulls'][rep['true_best']], 1)} of {st['budget']} |")
    L += ["", f"Gap between the best and second-best true mean reward: {_num(rep['gap_to_second'], 4)}. "
              "Small gaps make any selection method slow; that is a property of the data, not of the bandit.", ""]
    L += ["## 2. Validation comparison (all seeds)", ""]
    _model_table(val, models, L)
    L += ["", "### Training behaviour", "", "| Model | Parameters | Train time / seed | Epochs (best) |",
          "|---|---|---|---|"]
    for m in models:
        r = val["models"][m]
        if r.get("pretrained"):
            L.append(f"| {lab[m]} | {r['n_params']:,} (pretrained) | {_num(r['train_seconds_median'], 1)} s "
                     "inference, once | zero-shot |")
            continue
        L.append(f"| {lab[m]} | {r['n_params']:,} | {_num(r['train_seconds_median'], 1)} s | "
                 f"{_num(r['epochs_median'], 0)} ({_num(r['best_epoch_median'], 0)}) |")
    L += ["", "### Regime slices (validation, seed-ensemble Sharpe)", ""]
    _regimes(res["validation_regimes"], L)
    if not res["test"].get("locked", True):
        t = res["test"]
        L += ["", "## 3. Test block (one-shot, unlocked " + t["unlocked_at"] + ")", ""]
        _model_table(t, models, L)
        L += ["", f"* Bandit pick **{lab[t['winner']]}** median test Sharpe {_num(t['winner_sharpe_median'])} vs "
                  f"equal-weight quarterly {_num(t['ew_quarterly_sharpe'])} → "
                  f"{'beats' if t['winner_beats_ew_quarterly'] else 'does not beat'} the bar to beat; "
                  f"positive Sharpe: {'yes' if t['winner_positive_sharpe'] else 'no'}.", "",
              "### Regime slices (test)", ""]
        _regimes(t["regimes"], L)
    else:
        L += ["", "## 3. Test block", "", "Locked. Re-run with `--test` once the design is frozen."]
    if files:
        L += ["", "## Charts", ""] + [f"![{f}]({f})" for f in files]
    p = out / "report.md"
    p.write_text("\n".join(L) + "\n")
    return p


def _regimes(rows: list[dict], L: list[str]):
    if not rows:
        return
    cols = list(rows[0]["sharpe"])
    L += ["| Regime | Bars | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in rows:
        L.append(f"| {r['regime']} | {r['bars']} | " + " | ".join(_num(r["sharpe"][c]) for c in cols) + " |")

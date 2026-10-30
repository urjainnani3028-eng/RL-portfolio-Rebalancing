"""Neural forecasters (RNN, LSTM, GRU, TCN, TimesFM; PatchTST optional) for ETF rebalancing, with a
multi-armed bandit that decides which architecture deserves the training / testing budget.

Pipeline:  features.py -> models.py / foundation.py -> training.py -> portfolio.py -> bandit.py -> experiment.py
Entry points: `python -m app.nnbandit.cli` (command line) and POST /api/bandit (web app).
"""

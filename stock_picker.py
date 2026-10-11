#!/usr/bin/env python3
"""
Stock Picker: technicals + RSS news sentiment -> BUY / SHORT candidates.

Setup:
    pip install yfinance feedparser pandas numpy vaderSentiment

Usage:
    python stock_picker.py run                      # analyze, print top 10, save snapshot
    python stock_picker.py run --tickers AAPL TSLA NVDA AMD META
    python stock_picker.py run --store sqlite       # sqlite | pickle | both (default)
    python stock_picker.py backfill --days 250      # rebuild technical-only history instantly
    python stock_picker.py evaluate --horizon 5     # how did past picks actually do?
    python stock_picker.py train --horizon 5        # fit new feature weights -> weights.json

Run it once per day AFTER market close (4pm ET) so the saved price is a full-day close.

NOT financial advice. The default weights are hand-set heuristics, not a proven edge.
Use `evaluate` to find out whether it works before risking money.
"""
import argparse
import calendar
import json
import math
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "picker.db"
PICKLE_DIR = DATA_DIR / "snapshots"
WEIGHTS_PATH = Path("weights.json")

TOP_N = 10
HORIZONS = [1, 2, 3, 4, 5]        # trading days out for price forecasts
EXPECTED_MOVE_HORIZON_DAYS = 5
MOVE_K = 0.35                    # placeholder drift scale; calibrate with `evaluate`
HEADLINE_MAX_AGE_DAYS = 7
HEADLINE_HALF_LIFE_DAYS = 2.0

# ticker -> name aliases used to match headlines from general feeds
UNIVERSE = {
    "AAPL": ["Apple"], "MSFT": ["Microsoft"], "NVDA": ["Nvidia"], "AMZN": ["Amazon"],
    "GOOGL": ["Alphabet", "Google"], "META": ["Meta Platforms", "Facebook"],
    "TSLA": ["Tesla"], "AVGO": ["Broadcom"], "AMD": ["Advanced Micro Devices", "AMD"],
    "NFLX": ["Netflix"], "ORCL": ["Oracle"], "CRM": ["Salesforce"], "ADBE": ["Adobe"],
    "INTC": ["Intel"], "QCOM": ["Qualcomm"], "MU": ["Micron"], "PLTR": ["Palantir"],
    "JPM": ["JPMorgan", "JP Morgan"], "BAC": ["Bank of America"], "GS": ["Goldman Sachs"],
    "MS": ["Morgan Stanley"], "WFC": ["Wells Fargo"], "C": ["Citigroup", "Citi"],
    "V": ["Visa"], "MA": ["Mastercard"], "PYPL": ["PayPal"], "COIN": ["Coinbase"],
    "BRK-B": ["Berkshire Hathaway"], "UNH": ["UnitedHealth"], "JNJ": ["Johnson & Johnson"],
    "LLY": ["Eli Lilly", "Lilly"], "PFE": ["Pfizer"], "MRK": ["Merck"], "ABBV": ["AbbVie"],
    "XOM": ["Exxon"], "CVX": ["Chevron"], "COP": ["ConocoPhillips"],
    "WMT": ["Walmart"], "COST": ["Costco"], "HD": ["Home Depot"], "TGT": ["Target Corp"],
    "NKE": ["Nike"], "SBUX": ["Starbucks"], "MCD": ["McDonald's"], "KO": ["Coca-Cola"],
    "PEP": ["PepsiCo"], "DIS": ["Disney"], "BA": ["Boeing"], "CAT": ["Caterpillar"],
    "GE": ["General Electric", "GE Aerospace"], "F": ["Ford Motor"], "GM": ["General Motors"],
    "UBER": ["Uber"], "ABNB": ["Airbnb"], "SHOP": ["Shopify"], "SNOW": ["Snowflake"],
    "SMCI": ["Super Micro"], "ARM": ["Arm Holdings"], "T": ["AT&T"], "VZ": ["Verizon"],
    "CMCSA": ["Comcast"], "PG": ["Procter & Gamble"],
}

# General market feeds. Add or remove freely; anything that serves RSS works.
GENERAL_FEEDS = {
    "CNBC Markets": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
}
YAHOO_TICKER_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"

# Finance-specific words VADER doesn't know about
FINANCE_LEXICON = {
    "upgrade": 2.5, "upgraded": 2.5, "downgrade": -2.5, "downgraded": -2.5,
    "outperform": 2.0, "underperform": -2.0, "overweight": 1.5, "underweight": -1.5,
    "beats": 2.0, "beat": 1.5, "miss": -2.0, "misses": -2.0, "missed": -2.0,
    "surge": 2.0, "surges": 2.0, "soars": 2.5, "soar": 2.5, "rally": 2.0, "rallies": 2.0,
    "plunge": -2.5, "plunges": -2.5, "tumbles": -2.5, "tumble": -2.5, "slump": -2.0,
    "slumps": -2.0, "sinks": -2.0, "bullish": 2.0, "bearish": -2.0, "record": 1.5,
    "lawsuit": -1.5, "probe": -1.5, "investigation": -1.5, "recall": -2.0,
    "layoffs": -1.5, "bankruptcy": -3.5, "fraud": -3.0, "raises": 1.0, "cuts": -1.0,
    "buyback": 1.5, "dividend": 1.0, "warning": -1.5, "halts": -1.5,
}

FEATURES = ["mom_20d", "rev_5d", "macd", "trend_50", "rsi_c", "vol_confirm", "sentiment"]

# Hand-set starting weights. `train` replaces these with fitted values.
DEFAULT_WEIGHTS = {
    "mom_20d": 0.20,        # medium-term momentum
    "rev_5d": -0.10,        # short-term reversal (negative = fade last week's move)
    "macd": 0.15,           # trend acceleration
    "trend_50": 0.15,       # distance above/below 50-day average
    "rsi_c": 0.05,          # RSI centered on 0
    "vol_confirm": 0.10,    # unusual volume in direction of today's move
    "sentiment": 0.30,      # news sentiment
}

SNAP_COLS = (["run_date", "ticker", "source", "price", "atr_pct", "ret_1d"] + FEATURES +
             ["n_headlines", "score", "expected_move_pct"] +
             [f"exp_price_d{h}" for h in HORIZONS] + ["direction", "rank"])

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(FINANCE_LEXICON)


# --------------------------------------------------------------------------- #
# Price data + technical features
# --------------------------------------------------------------------------- #
def download_prices(tickers, period=None, start=None, min_rows=70):
    kw = dict(auto_adjust=True, group_by="ticker", progress=False, threads=True)
    raw = (yf.download(tickers, period=period, **kw) if period
           else yf.download(tickers, start=start, **kw))
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=["Close"])
            if df.index.tz is not None:
                df = df.tz_localize(None)
            if len(df) >= min_rows:
                out[t] = df
        except KeyError:
            continue
    return out


def feature_frame(df):
    """Technical features for EVERY day in the price history (used by backfill)."""
    c, v, h, l = df["Close"], df["Volume"], df["High"], df["Low"]
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    macd_line = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    vol_z = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    ret_1d = c.pct_change()
    return pd.DataFrame({
        "price": c, "atr_pct": tr.rolling(14).mean() / c, "ret_1d": ret_1d,
        "mom_20d": c.pct_change(20), "rev_5d": c.pct_change(5),
        "macd": (macd_line - macd_line.ewm(span=9).mean()) / c,
        "trend_50": c / c.rolling(50).mean() - 1, "rsi_c": (rsi - 50) / 50,
        "vol_confirm": vol_z.clip(-3, 3).fillna(0) * np.sign(ret_1d),
    })


def compute_features(df):
    """Features as of the latest bar."""
    return feature_frame(df).iloc[-1].to_dict()


# --------------------------------------------------------------------------- #
# RSS sentiment
# --------------------------------------------------------------------------- #
def _entry_time(e):
    ts = e.get("published_parsed") or e.get("updated_parsed")
    return datetime.utcfromtimestamp(calendar.timegm(ts)) if ts else datetime.utcnow()


def _parse_feed(url, source):
    try:
        feed = feedparser.parse(url, agent="Mozilla/5.0 (stock-picker)")
        return [{"title": e.get("title", ""), "summary": re.sub("<[^>]+>", "", e.get("summary", "")),
                 "published": _entry_time(e), "source": source} for e in feed.entries]
    except Exception as exc:  # network hiccups shouldn't kill the run
        print(f"  ! feed failed ({source}): {exc}")
        return []


def _score_text(text):
    return _analyzer.polarity_scores(text)["compound"]


def gather_headlines(universe):
    """Return {ticker: [headline dicts]} from per-ticker Yahoo feeds + general feeds."""
    cutoff = datetime.utcnow() - timedelta(days=HEADLINE_MAX_AGE_DAYS)
    result = {t: [] for t in universe}

    # 1) Yahoo per-ticker feeds (already tagged to the ticker)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {t: ex.submit(_parse_feed, YAHOO_TICKER_FEED.format(t=t), "Yahoo Finance")
                   for t in universe}
        for t, f in futures.items():
            result[t].extend(f.result())

    # 2) General feeds, matched to tickers by company name / $TICKER / (TICKER)
    patterns = {}
    for t, aliases in universe.items():
        names = "|".join(re.escape(a) for a in aliases)
        patterns[t] = re.compile(
            rf"\b(?:{names})\b|\${re.escape(t)}\b|\({re.escape(t)}\)", re.I)
    for source, url in GENERAL_FEEDS.items():
        for item in _parse_feed(url, source):
            text = f"{item['title']}. {item['summary']}"
            for t, pat in patterns.items():
                if pat.search(text):
                    result[t].append(item)

    # dedupe by title, drop stale, score
    for t, items in result.items():
        seen, kept = set(), []
        for it in items:
            key = it["title"].strip().lower()
            if key in seen or it["published"] < cutoff:
                continue
            seen.add(key)
            it["compound"] = _score_text(f"{it['title']}. {it['summary']}"[:500])
            kept.append(it)
        result[t] = kept
    return result


def sentiment_score(items):
    """Recency-weighted mean of headline sentiment, shrunk toward 0 when few headlines."""
    if not items:
        return 0.0
    now = datetime.utcnow()
    num = den = 0.0
    for it in items:
        age_days = (now - it["published"]).total_seconds() / 86400
        w = 0.5 ** (age_days / HEADLINE_HALF_LIFE_DAYS)
        num += w * it["compound"]
        den += w
    return num / (den + 1.0)  # +1 = shrinkage prior


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def load_weights():
    if WEIGHTS_PATH.exists():
        w = json.loads(WEIGHTS_PATH.read_text())
        if set(w) == set(FEATURES):
            print(f"Using trained weights from {WEIGHTS_PATH}")
            return w
    return dict(DEFAULT_WEIGHTS)


def zscore_cols(df):
    return df[FEATURES].apply(lambda s: ((s - s.mean()) / (s.std(ddof=0) + 1e-9)).clip(-3, 3))


def score_universe(df, weights):
    z = zscore_cols(df)
    df["score"] = sum(weights[f] * z[f] for f in FEATURES)
    # Heuristic expected move: conviction x typical daily range x sqrt(horizon).
    # k=0.35 is a placeholder scale; calibrate it with `evaluate` data.
    for h in HORIZONS:  # expected price h trading days out; drift grows with sqrt(h)
        df[f"exp_price_d{h}"] = df["price"] * (1 + MOVE_K * df["score"] * df["atr_pct"] * math.sqrt(h))
    df["expected_move_pct"] = (df[f"exp_price_d{EXPECTED_MOVE_HORIZON_DAYS}"] / df["price"] - 1) * 100
    df["direction"] = np.where(df["score"] >= 0, "BUY", "SHORT")
    df["rank"] = df["expected_move_pct"].abs().rank(ascending=False, method="first").astype(int)
    return df.sort_values("rank")


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def save_snapshot(snap, headlines_df, weights, run_date, store):
    DATA_DIR.mkdir(exist_ok=True)
    if store in ("sqlite", "both"):
        with sqlite3.connect(DB_PATH) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            dates = snap["run_date"].unique().tolist()
            for name, df in (("snapshots", snap), ("headlines", headlines_df)):
                if name in tables:
                    marks = ",".join("?" * len(dates))
                    con.execute(f"DELETE FROM {name} WHERE run_date IN ({marks})", dates)
                    have = {r[1] for r in con.execute(f"PRAGMA table_info({name})")}
                    for col in df.columns.difference(have):  # new data point -> new column
                        kind = "REAL" if pd.api.types.is_numeric_dtype(df[col]) else "TEXT"
                        con.execute(f'ALTER TABLE {name} ADD COLUMN "{col}" {kind}')
                if not df.empty:
                    df.to_sql(name, con, if_exists="append", index=False)
        print(f"Saved to SQLite: {DB_PATH}")
    if store in ("pickle", "both"):
        PICKLE_DIR.mkdir(parents=True, exist_ok=True)
        path = PICKLE_DIR / f"{run_date}.pkl"
        pd.to_pickle({"snapshot": snap, "headlines": headlines_df, "weights": weights}, path)
        print(f"Saved pickle:    {path}")


def load_snapshots():
    if not DB_PATH.exists():
        raise SystemExit("No database yet. Run `python stock_picker.py run` first.")
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql("SELECT * FROM snapshots", con)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def analyze_ticker(ticker, reference, weights=None):
    """Score ANY ticker by z-scoring it against a saved snapshot's cross-section.
    `reference` = DataFrame of one snapshot day. Returns (row, headline items)."""
    ticker = ticker.upper().strip()
    prices = download_prices([ticker], period="1y")
    if ticker not in prices:
        raise ValueError("no price data found (check the symbol; it needs about 70 trading days of history)")
    feats = compute_features(prices[ticker])
    items = gather_headlines({ticker: UNIVERSE.get(ticker, [ticker])})[ticker]
    feats["sentiment"], feats["n_headlines"] = sentiment_score(items), len(items)
    if any(pd.isna(feats[f]) for f in FEATURES):
        raise ValueError("not enough history to compute indicators")
    weights = weights or load_weights()
    z = {}
    for f in FEATURES:
        sd = reference[f].std(ddof=0)
        z[f] = 0.0 if not sd > 1e-9 else float(np.clip((feats[f] - reference[f].mean()) / sd, -3, 3))
    score = sum(weights[f] * z[f] for f in FEATURES)
    for h in HORIZONS:
        feats[f"exp_price_d{h}"] = feats["price"] * (1 + MOVE_K * score * feats["atr_pct"] * math.sqrt(h))
    feats.update(ticker=ticker, run_date=prices[ticker].index[-1].strftime("%Y-%m-%d"), source="lookup",
                 score=score, direction="BUY" if score >= 0 else "SHORT",
                 expected_move_pct=(feats[f"exp_price_d{EXPECTED_MOVE_HORIZON_DAYS}"] / feats["price"] - 1) * 100)
    return pd.Series(feats), items


def run_analysis(arg_tickers=None, store="both", default_weights=False):
    """Full pipeline: fetch, score, save. Returns (snapshot_df, run_date)."""
    tickers = [t.upper() for t in arg_tickers] if arg_tickers else list(UNIVERSE)
    universe = {t: UNIVERSE.get(t, [t]) for t in tickers}
    run_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Downloading 1y of prices for {len(tickers)} tickers...")
    prices = download_prices(tickers, period="1y")
    print("Fetching RSS headlines...")
    headlines = gather_headlines({t: universe[t] for t in prices})

    rows, hl_rows = [], []
    for t, df in prices.items():
        feats = compute_features(df)
        items = headlines.get(t, [])
        feats.update(ticker=t, run_date=run_date, source="live",
                     sentiment=sentiment_score(items), n_headlines=len(items))
        rows.append(feats)
        for it in items:
            hl_rows.append({"run_date": run_date, "ticker": t, "source": it["source"],
                            "published": it["published"].isoformat(), "title": it["title"],
                            "compound": it["compound"]})

    df = pd.DataFrame(rows).dropna(subset=FEATURES)
    weights = dict(DEFAULT_WEIGHTS) if default_weights else load_weights()
    snap = score_universe(df, weights)[SNAP_COLS]
    save_snapshot(snap, pd.DataFrame(hl_rows), weights, run_date, store)
    return snap, run_date


def cmd_run(args):
    snap, run_date = run_analysis(args.tickers, args.store, args.default_weights)
    cols = ["rank", "ticker", "direction", "price"] + [f"exp_price_d{h}" for h in HORIZONS] + ["score", "sentiment"]
    print(f"\nTop {TOP_N} expected movers, {run_date}  (exp_price_dN = forecast N trading days out)")
    print(snap.head(TOP_N)[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def cmd_backfill(args):
    """Rebuild technical-only snapshots for past days from price history.
    Sentiment can't be backfilled (RSS only serves recent items), so it is 0 on these rows."""
    start = (datetime.now() - timedelta(days=int(args.days * 1.5) + 120)).strftime("%Y-%m-%d")
    prices = download_prices(list(UNIVERSE), start=start)
    frames = []
    for t, df in prices.items():
        f = feature_frame(df).dropna()
        f["ticker"] = t
        frames.append(f)
    allf = pd.concat(frames)
    allf.index.name = "date"
    allf = allf.reset_index()

    keep, live = set(), set()   # never overwrite real (live) snapshots
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(snapshots)")}
            if cols:
                q = "SELECT DISTINCT run_date FROM snapshots"
                q += " WHERE source IS NOT 'backfill'" if "source" in cols else ""
                live = {r[0] for r in con.execute(q)}
    weights, out = load_weights(), []
    for d in sorted(allf["date"].unique())[-args.days:]:
        rd = pd.Timestamp(d).strftime("%Y-%m-%d")
        g = allf[allf["date"] == d].copy()
        if rd in live or len(g) < 10:
            continue
        g["run_date"], g["source"], g["sentiment"], g["n_headlines"] = rd, "backfill", 0.0, 0
        out.append(score_universe(g, weights)[SNAP_COLS])
    if not out:
        raise SystemExit("Nothing to backfill.")
    save_snapshot(pd.concat(out), pd.DataFrame(), weights, None, "sqlite")
    print(f"Backfilled {len(out)} days ({sum(len(o) for o in out)} rows). Sentiment = 0 on these rows.")


def forward_returns(snaps, horizon):
    """Attach realized forward returns (fwd_ret_1..5, plus fwd_ret for `horizon`)."""
    hs = sorted(set(HORIZONS) | {horizon})
    start = pd.to_datetime(snaps["run_date"]).min() - timedelta(days=7)
    prices = download_prices(snaps["ticker"].unique().tolist(), start=start.strftime("%Y-%m-%d"), min_rows=2)
    closes = {t: d["Close"] for t, d in prices.items()}
    cols = {h: [] for h in hs}
    for t, rd in zip(snaps["ticker"], snaps["run_date"]):
        c = closes.get(t)
        i = c.index.searchsorted(pd.Timestamp(rd), side="right") - 1 if c is not None else -1
        for h in hs:
            ok = i >= 0 and i + h < len(c)
            cols[h].append(c.iloc[i + h] / c.iloc[i] - 1 if ok else np.nan)
    snaps = snaps.copy()
    for h in hs:
        snaps[f"fwd_ret_{h}"] = cols[h]
    snaps["fwd_ret"] = snaps[f"fwd_ret_{horizon}"]
    return snaps.dropna(subset=["fwd_ret"])


def spearman(a, b):
    """Rank correlation without needing scipy (pandas 3 requires it for method='spearman')."""
    return a.rank().corr(b.rank())


def evaluate_snaps(snaps):
    """snaps: output of forward_returns. Returns (daily metrics, per-feature IC, forecast accuracy)."""
    daily = []
    for d, g in snaps.groupby("run_date"):
        top = g[g["rank"] <= TOP_N]
        daily.append({
            "run_date": d,
            "ic": spearman(g["score"], g["fwd_ret"]) if len(g) > 5 else np.nan,
            "hit": (np.sign(top["score"]) == np.sign(top["fwd_ret"])).mean(),
            "top_ret": (np.sign(top["score"]) * top["fwd_ret"]).mean(),  # short profits when price falls
        })
    feat_ic = pd.Series({f: spearman(snaps[f], snaps["fwd_ret"]) for f in FEATURES})
    fc = []
    for h in HORIZONS:
        if f"exp_price_d{h}" not in snaps:
            continue
        s = snaps.dropna(subset=[f"fwd_ret_{h}", f"exp_price_d{h}"])
        if s.empty:
            continue
        pred, act = s[f"exp_price_d{h}"] / s["price"] - 1, s[f"fwd_ret_{h}"]
        fc.append({"days_out": h, "n": len(s), "direction_hit_rate": (np.sign(pred) == np.sign(act)).mean(),
                   "avg_abs_error_pct": (pred - act).abs().mean() * 100,
                   "avg_abs_predicted_pct": pred.abs().mean() * 100, "avg_abs_actual_pct": act.abs().mean() * 100})
    return pd.DataFrame(daily), feat_ic, pd.DataFrame(fc)


def cmd_evaluate(args):
    snaps = forward_returns(load_snapshots(), args.horizon)
    if snaps.empty:
        raise SystemExit(f"No snapshots are {args.horizon}+ trading days old yet. Keep running daily (or `backfill`).")
    daily, feat_ic, fc = evaluate_snaps(snaps)
    print(f"\nEvaluation over {len(daily)} snapshot days, {args.horizon}-day forward returns")
    print(f"  Mean rank IC (score vs. forward return): {daily['ic'].mean():+.3f}")
    print(f"  Top-{TOP_N} direction hit rate:            {daily['hit'].mean():.1%}")
    print(f"  Top-{TOP_N} avg return per pick (L/S):     {daily['top_ret'].mean():+.2%}")
    print("\nPer-feature IC (which signals actually predict?):")
    print(feat_ic.round(3).to_string())
    if not fc.empty:
        print("\nPrice-forecast accuracy by days out:")
        print(fc.round(3).to_string(index=False))
    print("\nIC around +0.03 to +0.05 sustained over months is a decent real-world signal. Small samples are noise.")


def cmd_train(args):
    snaps = forward_returns(load_snapshots(), args.horizon)
    if len(snaps) < args.min_rows:
        raise SystemExit(f"Only {len(snaps)} labeled rows; need >= {args.min_rows}. Keep collecting.")

    X = pd.concat([zscore_cols(g) for _, g in snaps.groupby("run_date")]).loc[snaps.index]
    y = snaps["fwd_ret"] - snaps.groupby("run_date")["fwd_ret"].transform("mean")
    y = y / (y.std() + 1e-9)

    A = X.values
    w = np.linalg.solve(A.T @ A + args.ridge * np.eye(A.shape[1]), A.T @ y.values)
    w = w / (np.abs(w).sum() + 1e-9)
    weights = dict(zip(FEATURES, map(float, w)))
    WEIGHTS_PATH.write_text(json.dumps(weights, indent=2))

    print(f"Trained on {len(snaps)} rows. Saved to {WEIGHTS_PATH}:")
    for f in FEATURES:
        print(f"  {f:<12} {DEFAULT_WEIGHTS[f]:+.2f} -> {weights[f]:+.2f}")
    print("\nCaution: this is in-sample. Re-run `evaluate` on days AFTER training "
          "(walk-forward) to see if it holds. Delete weights.json to revert to defaults.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="analyze and save today's snapshot")
    r.add_argument("--tickers", nargs="+")
    r.add_argument("--store", choices=["sqlite", "pickle", "both"], default="both")
    r.add_argument("--default-weights", action="store_true", help="ignore weights.json")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="score past picks against realized returns")
    e.add_argument("--horizon", type=int, default=5, help="trading days forward")
    e.set_defaults(func=cmd_evaluate)

    t = sub.add_parser("train", help="fit feature weights on saved history")
    t.add_argument("--horizon", type=int, default=5)
    t.add_argument("--ridge", type=float, default=50.0, help="regularization strength")
    t.add_argument("--min-rows", type=int, default=500)
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("backfill", help="rebuild technical-only history from past prices")
    b.add_argument("--days", type=int, default=250)
    b.set_defaults(func=cmd_backfill)

    args = p.parse_args()
    if not args.cmd:
        args = p.parse_args(["run"])
    args.func(args)


if __name__ == "__main__":
    main()

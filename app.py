"""Streamlit front end for stock_picker.py.   Run:  streamlit run app.py"""
import contextlib
import io
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import stock_picker as sp

st.set_page_config(page_title="Stock Picker", page_icon="📈", layout="wide")


# ----------------------------- data helpers ------------------------------- #
@st.cache_data(ttl=60)
def load(table):
    if not sp.DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(sp.DB_PATH) as con:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", con)
        except Exception:
            return pd.DataFrame()
    if table == "snapshots":
        df["source"] = df["source"].fillna("live") if "source" in df else "live"
    return df


@st.cache_data(ttl=3600)
def history(ticker):
    try:
        return sp.download_prices([ticker], period="1y", min_rows=2).get(ticker)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Fetching realized prices...")
def evaluate(horizon, source, n_rows):
    s = load("snapshots")
    if source != "All":
        s = s[s["source"] == source.lower()]
    if s.empty:
        return None
    snaps = sp.forward_returns(s, horizon)
    return None if snaps.empty else sp.evaluate_snaps(snaps)


def call(fn, **kw):
    """Run a CLI command from the UI and capture its printed output."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn(SimpleNamespace(**kw))
    except SystemExit as e:
        buf.write(str(e))
    st.cache_data.clear()
    return buf.getvalue()


def forecast_chart(row, hist):
    run_ts = pd.Timestamp(row["run_date"])
    days = pd.bdate_range(run_ts + pd.Timedelta(days=1), periods=len(sp.HORIZONS))
    xs = [run_ts] + list(days)
    px = np.array([row["price"]] + [row[f"exp_price_d{h}"] for h in sp.HORIZONS])
    band = row["price"] * row["atr_pct"] * np.sqrt([0] + sp.HORIZONS)
    fig = go.Figure()
    if hist is not None:
        h = hist[hist.index >= run_ts - pd.Timedelta(days=75)]
        fig.add_scatter(x=h.index, y=h["Close"], name="Actual close", line=dict(color="#888"))
    fig.add_scatter(x=xs, y=px + band, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=xs, y=px - band, fill="tonexty", fillcolor="rgba(99,110,250,0.2)",
                    line=dict(width=0), name="Typical range (~1 ATR x sqrt(days))")
    fig.add_scatter(x=xs, y=px, name="Forecast", mode="lines+markers",
                    line=dict(dash="dash", color="#636EFA"))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Price ($)",
                      legend=dict(orientation="h", y=1.08))
    return fig


# -------------------------------- sidebar --------------------------------- #
st.sidebar.title("📈 Stock Picker")
custom = st.sidebar.text_input("Tickers (blank = full default universe)", placeholder="AAPL TSLA NVDA")
if st.sidebar.button("▶ Run analysis now", type="primary"):
    with st.spinner("Fetching prices + news and scoring... (about a minute)"):
        sp.run_analysis(custom.split() or None, store="both")
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("Run after the 4pm ET close. Blank = full universe, which gives the best training data.")

snaps = load("snapshots")
if snaps.empty:
    st.title("No data yet")
    st.info("Click **Run analysis now** in the sidebar, or open the **Data & Model** tab and "
            "run a **backfill** to build history from past prices.")
    tab_data = st.tabs(["Data & Model"])[0]
    dates = []
else:
    dates = sorted(snaps["run_date"].unique(), reverse=True)
    sel = st.sidebar.selectbox("Snapshot date", dates)
    tab_picks, tab_perf, tab_data = st.tabs(["🎯 Top Picks", "📊 Performance", "🗄 Data & Model"])

    # ------------------------------ top picks ------------------------------ #
    with tab_picks:
        day = snaps[snaps["run_date"] == sel].sort_values("rank")
        top = day.head(sp.TOP_N)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Buy candidates", int((top["direction"] == "BUY").sum()))
        c2.metric("Short candidates", int((top["direction"] == "SHORT").sum()))
        c3.metric("Avg expected 5-day move", f"{top['expected_move_pct'].abs().mean():.1f}%")
        c4.metric("Snapshot type", day["source"].iloc[0])
        if day["source"].iloc[0] == "backfill":
            st.warning("Backfilled snapshot: technicals only, no news sentiment.")

        st.subheader(f"Top {sp.TOP_N} expected movers: {sel}")
        view = top[["rank", "ticker", "direction", "price"] + [f"exp_price_d{h}" for h in sp.HORIZONS] +
                   ["expected_move_pct", "sentiment", "n_headlines"]].copy()
        view["direction"] = view["direction"].map({"BUY": "🟢 BUY", "SHORT": "🔴 SHORT"})
        cfg = {"price": st.column_config.NumberColumn("Price", format="$%.2f"),
               "expected_move_pct": st.column_config.NumberColumn("5d move", format="%+.1f%%"),
               "sentiment": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
               "n_headlines": "Headlines"}
        cfg.update({f"exp_price_d{h}": st.column_config.NumberColumn(f"Day {h}", format="$%.2f")
                    for h in sp.HORIZONS})
        st.dataframe(view, hide_index=True, column_config=cfg)

        st.subheader("Drill down")
        ticker = st.selectbox("Ticker", day["ticker"].tolist(), index=0)
        row = day[day["ticker"] == ticker].iloc[0]
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(forecast_chart(row, history(ticker)))
            st.caption("Dashed = model forecast for the next 5 trading days. For older snapshots the grey "
                       "line shows what actually happened. The shaded band is a rough volatility range, "
                       "not a statistical confidence interval.")
        with right:
            st.markdown(f"**{ticker}**: {row['direction']}, score {row['score']:+.2f}, "
                        f"sentiment {row['sentiment']:+.2f} ({int(row['n_headlines'])} headlines)")
            hl = load("headlines")
            if not hl.empty:
                hl = hl[(hl["run_date"] == sel) & (hl["ticker"] == ticker)].sort_values("published", ascending=False)
                st.dataframe(hl[["published", "source", "title", "compound"]], hide_index=True,
                             column_config={"compound": st.column_config.NumberColumn("Sent.", format="%+.2f")})
            else:
                st.caption("No headlines stored for this snapshot.")

    # ------------------------------ performance ---------------------------- #
    with tab_perf:
        a, b = st.columns(2)
        horizon = a.select_slider("Evaluate forward return over (trading days)", options=list(range(1, 11)), value=5)
        source = b.radio("Snapshots to include", ["All", "Live", "Backfill"], horizontal=True)
        res = evaluate(horizon, source, len(snaps))
        if res is None:
            st.info(f"No snapshots are {horizon}+ trading days old yet for this selection. "
                    "Backfill history in the Data & Model tab, or keep running daily.")
        else:
            daily, feat_ic, fc = res
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Days evaluated", len(daily))
            m2.metric("Mean rank IC", f"{daily['ic'].mean():+.3f}", help="Rank correlation of score vs. realized return. ~+0.03-0.05 sustained is a decent signal.")
            m3.metric(f"Top-{sp.TOP_N} hit rate", f"{daily['hit'].mean():.1%}")
            m4.metric("Avg return / pick", f"{daily['top_ret'].mean():+.2%}", help="Longs earn the return, shorts earn the negative of it.")
            st.subheader("Cumulative top-10 pick return")
            st.line_chart(daily.set_index("run_date")["top_ret"].cumsum())
            st.caption("Sum of daily average pick returns. Windows overlap, so this is not a tradable equity curve.")
            l, r = st.columns(2)
            with l:
                st.subheader("Which signals predict?")
                st.bar_chart(feat_ic.rename("IC vs. forward return"))
            with r:
                st.subheader("Price-forecast accuracy by day")
                if fc.empty:
                    st.caption("Needs snapshots that include the 1-5 day price forecasts.")
                else:
                    st.dataframe(fc, hide_index=True, column_config={
                        "direction_hit_rate": st.column_config.NumberColumn("Direction hit", format="%.2f"),
                        "avg_abs_error_pct": st.column_config.NumberColumn("Avg error %", format="%.2f"),
                        "avg_abs_predicted_pct": st.column_config.NumberColumn("Avg |predicted| %", format="%.2f"),
                        "avg_abs_actual_pct": st.column_config.NumberColumn("Avg |actual| %", format="%.2f")})

# ------------------------------- data & model ------------------------------ #
with tab_data:
    hl = load("headlines")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Snapshot days", snaps["run_date"].nunique() if not snaps.empty else 0)
    d2.metric("Rows", len(snaps))
    d3.metric("Live days", snaps.loc[snaps["source"] == "live", "run_date"].nunique() if not snaps.empty else 0)
    d4.metric("Headlines stored", len(hl))

    st.subheader("1. Add history (backfill)")
    st.caption("Rebuilds technical-only snapshots from past prices so you can evaluate and train right away. "
               "News sentiment can't be backfilled (RSS only serves recent items), so it is 0 on these rows.")
    days = st.slider("Days to backfill", 60, 500, 250, step=10)
    if st.button("Backfill history"):
        with st.spinner("Computing features for past days..."):
            st.code(call(sp.cmd_backfill, days=days))

    st.subheader("2. Train model weights")
    t1, t2, t3 = st.columns(3)
    th = t1.select_slider("Horizon (days)", options=list(range(1, 11)), value=5, key="th")
    ridge = t2.number_input("Ridge strength", 0.0, 1000.0, 50.0)
    min_rows = t3.number_input("Min labeled rows", 100, 100000, 500)
    if st.button("Train on saved data"):
        with st.spinner("Fitting..."):
            st.code(call(sp.cmd_train, horizon=th, ridge=ridge, min_rows=int(min_rows)))

    active = json.loads(sp.WEIGHTS_PATH.read_text()) if sp.WEIGHTS_PATH.exists() else None
    st.bar_chart(pd.DataFrame({"default": sp.DEFAULT_WEIGHTS, "active (trained)": active or sp.DEFAULT_WEIGHTS}))
    if active and st.button("Reset to default weights"):
        sp.WEIGHTS_PATH.unlink()
        st.rerun()
    st.caption("Training is in-sample. Check the Performance tab on days after training before trusting it.")

    st.subheader("3. Export")
    if not snaps.empty:
        e1, e2 = st.columns(2)
        e1.download_button("⬇ snapshots.csv", snaps.to_csv(index=False), "snapshots.csv", "text/csv")
        if not hl.empty:
            e2.download_button("⬇ headlines.csv", hl.to_csv(index=False), "headlines.csv", "text/csv")

"""Admin-only pages (Performance, Data & Model). Loaded by app.py only when ?admin=1."""
import contextlib
import io
import json
import sqlite3
from types import SimpleNamespace

import pandas as pd
import streamlit as st

import stock_picker as sp


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



def performance(snaps):
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


def data_model(snaps):
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

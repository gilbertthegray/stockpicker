"""Public Streamlit site.   Run: streamlit run app.py
Admin tools (Performance, Backfill/Train, manual run) are hidden; open the site with ?admin=1 to show them."""
import re
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import stock_picker as sp

BRAND = "Gabe's Market Movers"
UP, DOWN, INK, MUTED = "#0f7b4f", "#c4351c", "#14181f", "#667085"
T5 = f"exp_price_d{sp.EXPECTED_MOVE_HORIZON_DAYS}"

st.set_page_config(page_title=BRAND, page_icon="📈", layout="wide", initial_sidebar_state="collapsed")


def is_admin():
    if st.query_params.get("admin") != "1":
        return False
    try:
        pw = st.secrets.get("ADMIN_PASSWORD")  # optional: set in Streamlit secrets to require a password
    except Exception:
        pw = None
    return not pw or st.text_input("Admin password", type="password") == pw


ADMIN = is_admin()

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:wght@500;600&display=swap');
html, body, .stApp, .stMarkdown, button, input, [data-testid="stDataFrame"] {{font-family:'IBM Plex Sans',-apple-system,'Segoe UI',sans-serif;}}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{display:none !important;}}
header[data-testid="stHeader"] {{background:transparent; height:0;}}
.block-container {{max-width:1080px; padding:1.2rem 1.5rem 4rem;}}
.nav {{display:flex; justify-content:space-between; align-items:baseline; padding-bottom:.7rem; border-bottom:1px solid #e4e7ec;}}
.brand {{font-family:'Newsreader',Georgia,serif; font-weight:600; font-size:1.5rem; color:{INK};}}
.muted {{color:{MUTED}; font-size:.88rem;}}
.hero h1 {{font-family:'Newsreader',Georgia,serif; font-weight:600; font-size:2.6rem; line-height:1.1;
          letter-spacing:-.02em; margin:2rem 0 .5rem; color:{INK}; max-width:18ch;}}
.hero p {{color:#475467; font-size:1.05rem; max-width:60ch; margin:0 0 1.6rem;}}
.col-title {{font-family:'Newsreader',Georgia,serif; font-weight:600; font-size:1.25rem; margin:.4rem 0 .2rem; color:{INK};}}
.mv {{display:grid; grid-template-columns:1.1fr 1fr 1.2fr .8fr; align-items:baseline; padding:.7rem 0;
     border-bottom:1px solid #e4e7ec; font-variant-numeric:tabular-nums;}}
.mv .tk {{font-weight:600; color:{INK};}}
.mv .chg {{text-align:right; font-weight:600;}}
.up {{color:{UP};}} .down {{color:{DOWN};}}
.pill {{color:#fff; border-radius:4px; font-size:.78rem; font-weight:600; padding:.12rem .5rem; margin:0 .6rem;}}
.sym {{font-family:'Newsreader',Georgia,serif; font-weight:600; font-size:2rem; color:{INK};}}
.foot {{color:#98a2b3; font-size:.8rem; margin-top:3rem; max-width:70ch;}}
@media (max-width:640px) {{.hero h1 {{font-size:2rem;}} .mv {{grid-template-columns:1fr 1fr 1fr .8fr; font-size:.9rem;}}}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
if not ADMIN:
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
                "[data-testid='collapsedControl']{display:none !important;}</style>", unsafe_allow_html=True)


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


@st.cache_data(ttl=900, show_spinner=False)
def lookup(ticker, ref_date):
    """Analyze a ticker that isn't in the daily universe, scored against that day's market cross-section."""
    ref = load("snapshots")
    row, items = sp.analyze_ticker(ticker, ref[ref["run_date"] == ref_date])
    hl = pd.DataFrame([{"published": i["published"].isoformat(), "source": i["source"],
                        "title": i["title"], "compound": i["compound"]} for i in items])
    return row, hl


def forecast_chart(row, hist):
    run_ts = pd.Timestamp(row["run_date"])
    days = pd.bdate_range(run_ts + pd.Timedelta(days=1), periods=len(sp.HORIZONS))
    xs = [run_ts] + list(days)
    px = np.array([row["price"]] + [row[f"exp_price_d{h}"] for h in sp.HORIZONS])
    band = row["price"] * row["atr_pct"] * np.sqrt([0] + sp.HORIZONS)
    color = UP if row["direction"] == "BUY" else DOWN
    fig = go.Figure()
    if hist is not None:
        h = hist[hist.index >= run_ts - pd.Timedelta(days=75)]
        fig.add_scatter(x=h.index, y=h["Close"], name="Actual close", line=dict(color="#98a2b3"))
    fig.add_scatter(x=xs, y=px + band, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=xs, y=px - band, fill="tonexty", fillcolor="rgba(102,112,133,0.18)",
                    line=dict(width=0), name="Typical range")
    fig.add_scatter(x=xs, y=px, name="Forecast", mode="lines+markers", line=dict(dash="dash", color=color))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Price ($)",
                      legend=dict(orientation="h", y=1.1), plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#eef0f3")
    return fig


def detail(row, hl):
    up = row["direction"] == "BUY"
    st.markdown(f'<span class="sym">{row["ticker"]}</span>'
                f'<span class="pill" style="background:{UP if up else DOWN}">{"Buy" if up else "Short"}</span>'
                f'<span class="muted">Last close ${row["price"]:,.2f}</span>', unsafe_allow_html=True)
    for c, h in zip(st.columns(len(sp.HORIZONS)), sp.HORIZONS):
        p = row[f"exp_price_d{h}"]
        c.metric(f"Day {h}", f"${p:,.2f}", f"{(p / row['price'] - 1) * 100:+.1f}%")
    st.plotly_chart(forecast_chart(row, history(row["ticker"])))
    tone = "positive" if row["sentiment"] > 0.05 else "negative" if row["sentiment"] < -0.05 else "neutral"
    st.caption(f"News tone: {tone} ({int(row['n_headlines'])} recent headlines). The dashed line is the model's "
               "forecast. The grey band is a rough volatility range, not a guarantee.")
    if hl is not None and not hl.empty:
        st.dataframe(hl.sort_values("published", ascending=False)[["published", "source", "title", "compound"]],
                     hide_index=True, column_config={"compound": st.column_config.NumberColumn("Tone", format="%+.2f")})


def movers(df):
    if df.empty:
        return '<p class="muted">None in today\'s top 10.</p>'
    return "".join(
        f'<div class="mv"><span class="tk">{r["ticker"]}</span><span>${r["price"]:,.2f}</span>'
        f'<span class="muted">target ${r[T5]:,.2f}</span>'
        f'<span class="chg {"up" if r["direction"] == "BUY" else "down"}">{r["expected_move_pct"]:+.1f}%</span></div>'
        for _, r in df.iterrows())


# -------------------------------- admin bar -------------------------------- #
snaps = load("snapshots")
if ADMIN:
    st.sidebar.title("Admin")
    custom = st.sidebar.text_input("Tickers (blank = full universe)", placeholder="AAPL TSLA NVDA")
    if st.sidebar.button("Run analysis now", type="primary"):
        with st.spinner("Fetching prices and news (about a minute)..."):
            sp.run_analysis(custom.split() or None, store="both")
        st.cache_data.clear()
        st.rerun()

if snaps.empty:  # first visit after a restart: build today's analysis automatically
    if not st.session_state.get("auto_ran"):
        st.session_state["auto_ran"] = True
        with st.spinner("Preparing today's analysis. This takes about a minute the first time."):
            try:
                sp.run_analysis(store="sqlite")
            except Exception:
                pass
        st.cache_data.clear()
        st.rerun()
    st.error("Market data is unavailable right now. Try again in a few minutes.")
    if st.button("Try again"):
        st.session_state["auto_ran"] = False
        st.rerun()
    st.stop()

live = snaps[snaps["source"] == "live"]
latest = (live if not live.empty else snaps)["run_date"].max()
if ADMIN:
    latest = st.sidebar.selectbox("Snapshot date", sorted(snaps["run_date"].unique(), reverse=True))
day = snaps[snaps["run_date"] == latest].sort_values("rank")
top = day.head(sp.TOP_N)
hl_all = load("headlines")


def hl_for(ticker):
    if hl_all.empty:
        return None
    return hl_all[(hl_all["run_date"] == latest) & (hl_all["ticker"] == ticker)]


# ---------------------------------- page ----------------------------------- #
st.markdown(f'<div class="nav"><span class="brand">{BRAND}</span>'
            f'<span class="muted">Analysis from {latest}</span></div>'
            f'<div class="hero"><h1>Where the market is likely to move next</h1>'
            f'<p>Daily technical and news-sentiment analysis of {len(day)} stocks, with price forecasts '
            f'for the next five trading days.</p></div>', unsafe_allow_html=True)

names = ["Top picks", "Stock lookup"] + (["Performance", "Data & model"] if ADMIN else [])
tabs = st.tabs(names)

with tabs[0]:
    left, right = st.columns(2, gap="large")
    left.markdown('<div class="col-title">Likely to rise (buy)</div>', unsafe_allow_html=True)
    left.markdown(movers(top[top["direction"] == "BUY"]), unsafe_allow_html=True)
    right.markdown('<div class="col-title">Likely to fall (short)</div>', unsafe_allow_html=True)
    right.markdown(movers(top[top["direction"] == "SHORT"]), unsafe_allow_html=True)
    st.caption("The 10 stocks with the largest expected 5-day move. Target = forecast price in 5 trading days.")

    st.subheader("Forecast by day")
    view = top[["ticker", "direction", "price"] + [f"exp_price_d{h}" for h in sp.HORIZONS] +
               ["expected_move_pct", "n_headlines"]].copy()
    view["direction"] = view["direction"].map({"BUY": "Buy", "SHORT": "Short"})
    cfg = {"ticker": "Stock", "direction": "Signal", "n_headlines": "Headlines",
           "price": st.column_config.NumberColumn("Price", format="$%.2f"),
           "expected_move_pct": st.column_config.NumberColumn("5-day move", format="%+.1f%%")}
    cfg.update({f"exp_price_d{h}": st.column_config.NumberColumn(f"Day {h}", format="$%.2f") for h in sp.HORIZONS})
    st.dataframe(view, hide_index=True, column_config=cfg)

    st.subheader("Details")
    pick = st.selectbox("Choose a stock", top["ticker"].tolist(), label_visibility="collapsed")
    detail(top[top["ticker"] == pick].iloc[0], hl_for(pick))

    with st.expander(f"Browse all {len(day)} analyzed stocks"):
        allv = day[["ticker", "direction", "price", T5, "expected_move_pct"]].copy()
        allv["direction"] = allv["direction"].map({"BUY": "Buy", "SHORT": "Short"})
        st.dataframe(allv, hide_index=True, column_config={
            "ticker": "Stock", "direction": "Signal",
            "price": st.column_config.NumberColumn("Price", format="$%.2f"),
            T5: st.column_config.NumberColumn("Price in 5 days", format="$%.2f"),
            "expected_move_pct": st.column_config.NumberColumn("5-day move", format="%+.1f%%")})

with tabs[1]:
    st.subheader("Where will any stock be in 5 days?")
    q = st.text_input("Ticker symbol", value=top.iloc[0]["ticker"], max_chars=8,
                      help="Any US-listed symbol, for example AAPL, SOFI or RIVN.").strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", q):
        st.warning("Enter a valid ticker symbol, such as AAPL.")
    else:
        stored = day[day["ticker"] == q]
        try:
            if stored.empty:
                with st.spinner(f"Analyzing {q}..."):
                    row, hl = lookup(q, latest)
            else:
                row, hl = stored.iloc[0], hl_for(q)
            detail(row, hl)
            if stored.empty:
                st.caption(f"{q} is not in the daily list, so it was scored on demand against the same market data.")
        except Exception as e:
            st.error(f"Could not analyze {q}: {e}")

if ADMIN:
    import admin
    with tabs[2]:
        admin.performance(snaps)
    with tabs[3]:
        admin.data_model(snaps)

st.markdown('<div class="foot">For information only. This is not investment advice. Forecasts are '
            'model estimates based on past prices and news tone, and they can be wrong.</div>',
            unsafe_allow_html=True)

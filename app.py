
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tech Signals + Options + Black–Litterman", layout="wide")
st.title("📊 Tech Trading Dashboard: Signals + Options + Black–Litterman")

df = pd.read_csv("recommendations_bl_signals.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

close = pd.read_csv("close_prices.csv", index_col=0)
close.index = pd.to_datetime(close.index, errors="coerce")
close = close.dropna(how="all")

# ---------------- MCP helper ----------------
def mcp_reason(row):
    if "MCP_Recommendation" in row.index and pd.notna(row["MCP_Recommendation"]):
        return row["MCP_Recommendation"]

    # fallback (should rarely be used)
    t = row.get("Ticker", "")
    sig = row.get("Stock_Signal", "HOLD")
    opt = row.get("Options_Suggestion", "NO_TRADE")
    return f"For {t}, the model suggests {sig} (options idea: {opt})."

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
search = st.sidebar.text_input("Search ticker (e.g., AAPL)")
signal_filter = st.sidebar.multiselect("Stock Signal", sorted(df["Stock_Signal"].unique()), default=sorted(df["Stock_Signal"].unique()))
opt_filter = st.sidebar.multiselect("Options Suggestion", sorted(df["Options_Suggestion"].unique()), default=sorted(df["Options_Suggestion"].unique()))
tilt_filter = st.sidebar.multiselect("BL Tilt", sorted(df["BL_Tilt"].dropna().unique()), default=sorted(df["BL_Tilt"].dropna().unique()))
sort_by = st.sidebar.selectbox("Sort by", ["BL_Posterior_annual", "BL_Rank", "Signal_Strength", "RSI14", "Vol20_ann"], index=0)
sort_desc = st.sidebar.checkbox("Sort descending", value=True)

dff = df[
    df["Stock_Signal"].isin(signal_filter) &
    df["Options_Suggestion"].isin(opt_filter) &
    df["BL_Tilt"].isin(tilt_filter)
].copy()

if search.strip():
    dff = dff[dff["Ticker"].str.contains(search.strip().upper())]

dff = dff.sort_values(sort_by, ascending=not sort_desc)

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Recommendations Table", "Ticker Drilldown", "Top Picks Today", "Ask MCP"
])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks shown", len(dff))
    col2.metric("BUY", int((dff["Stock_Signal"]=="BUY").sum()))
    col3.metric("SELL", int((dff["Stock_Signal"]=="SELL").sum()))
    col4.metric("CALL/PUT ideas", int(dff["Options_Suggestion"].isin(["CALL","PUT"]).sum()))

    st.subheader("What does BL Posterior mean?")
    st.write(
        "Black–Litterman posterior expected return is the blended expected return after combining the market-implied prior "
        "with indicator-based views, weighted by confidence. Higher posterior return (annualized) suggests an overweight tilt; "
        "lower suggests underweight."
    )

    st.subheader("BL Distribution")
    fig = plt.figure(figsize=(10,3))
    plt.hist(dff["BL_Posterior_annual"].dropna(), bins=15)
    plt.title("Distribution of BL Posterior Expected Returns (Annualized)")
    plt.xlabel("Expected Return"); plt.ylabel("Count"); plt.grid(True)
    st.pyplot(fig)

with tab2:
    st.subheader("Recommendations Table")
    show_cols = [
        "Ticker","Date","Close","Stock_Signal","Options_Suggestion",
        "Signal_Strength","RSI14","Vol20_ann",
        "BL_Posterior_annual","BL_Rank","BL_Tilt","Signal_Explanation"
    ]
    show_cols = [c for c in show_cols if c in dff.columns]
    st.dataframe(dff[show_cols], width="stretch")

    st.download_button(
        "Download FILTERED CSV",
        dff.to_csv(index=False).encode("utf-8"),
        file_name="filtered_recommendations.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download FULL CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="full_recommendations.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Ticker Drilldown")
    ticker = st.selectbox("Select ticker", sorted(df["Ticker"].unique()))
    if ticker in close.columns:
        series = close[ticker].dropna()
        plot_df = pd.DataFrame({"Close": series})
        plot_df["MA20"] = plot_df["Close"].rolling(20).mean()
        plot_df["MA50"] = plot_df["Close"].rolling(50).mean()

        fig = plt.figure(figsize=(10,4))
        plt.plot(plot_df.index, plot_df["Close"], label="Close")
        plt.plot(plot_df.index, plot_df["MA20"], label="MA20")
        plt.plot(plot_df.index, plot_df["MA50"], label="MA50")
        plt.title(f"{ticker} Price + Moving Averages")
        plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.grid(True)
        st.pyplot(fig)

    st.write("Latest recommendation:")
    st.dataframe(df[df["Ticker"]==ticker], width="stretch")

with tab4:
    st.subheader("Top Picks Today (Dashboard shortlist)")
    top5 = df.sort_values("BL_Rank").head(5)
    bottom5 = df.sort_values("BL_Rank").tail(5)

    cols = ["Ticker","Stock_Signal","Options_Suggestion","BL_Posterior_annual","Signal_Explanation"]
    cols = [c for c in cols if c in df.columns]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Top 5 (Overweight candidates)")
        st.dataframe(top5[cols], width="stretch")
    with c2:
        st.markdown("### ⚠️ Bottom 5 (Underweight candidates)")
        st.dataframe(bottom5[cols], width="stretch")

with tab5:
    st.subheader("🧠 Ask MCP (Agent-style reasoning over your model outputs)")
    st.caption("Answers are generated only from your model output table (signals + BL + indicators).")

    q = st.text_input("Ask: 'Should I buy AAPL?', 'Why is NVDA underweight?', 'best call ideas', 'top 5'")

    if q:
        q_up = q.upper().strip()

        if "BEST CALL" in q_up or "CALL IDEAS" in q_up:
            picks = df[df["Options_Suggestion"]=="CALL"].sort_values("BL_Rank").head(5)
            st.markdown("### ✅ Best CALL ideas (Top 5 by BL rank)")
            st.dataframe(picks, width="stretch")

        elif "BEST PUT" in q_up or "PUT IDEAS" in q_up:
            picks = df[df["Options_Suggestion"]=="PUT"].sort_values("BL_Rank").head(5)
            st.markdown("### ✅ Best PUT ideas (Top 5 by BL rank)")
            st.dataframe(picks, width="stretch")

        elif "TOP 5" in q_up:
            picks = df.sort_values("BL_Rank").head(5)
            st.markdown("### ✅ Top 5 overall (by BL rank)")
            st.dataframe(picks, width="stretch")

        else:
            found = None
            for t in df["Ticker"].unique():
                if t in q_up:
                    found = t
                    break

            if found is None:
                st.warning("No ticker detected. Try including one like AAPL, NVDA, MSFT.")
            else:
                row = df[df["Ticker"]==found].iloc[0]
                st.markdown("### MCP Response")
                st.write(mcp_reason(row))

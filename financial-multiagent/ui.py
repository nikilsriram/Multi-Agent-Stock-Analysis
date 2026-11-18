import streamlit as st
from main import run_analysis, simple_backtest
import pandas as pd
import plotly.graph_objects as go
import asyncio

st.title("AI Stock Analyzer")

stock = st.text_input("Stock Ticker (e.g. AAPL)")
start_date = st.text_input("Start Date")
end_date = st.text_input("End Date")

result = None  

if st.button("Analyze Stock"):
    if stock and start_date and end_date:
        with st.spinner("Running AI agents..."):
            result = asyncio.run(
        run_analysis(
            stock,
            str(start_date),
            str(end_date)
        )
    )



if result:
    st.subheader("📌 Final Decision + Reasoning")
    st.write(result)

if st.button("Run Backtest"):

    result = simple_backtest(stock, end_date)

    
    if result["status"] == "error":
        st.error(result["message"])

    
    else:
        st.metric("Return %", result["change_pct"])

        df = pd.DataFrame({"Close": result["data"]})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines"
        ))

        st.plotly_chart(fig)
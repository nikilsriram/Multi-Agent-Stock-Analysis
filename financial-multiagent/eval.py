import asyncio
import time
import pandas as pd
from single_agent import run_single_analysis, simple_backtest
from main import run_analysis

STOCKS = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "JPM", "XOM", "UNH", "WMT", "META"]
WINDOWS = [
    ("2026-01-05", "2026-01-19"),
    ("2026-02-02", "2026-02-16"),
    ("2026-03-02", "2026-03-16"),
    ("2026-04-06", "2026-04-20"),
    ("2026-05-04", "2026-05-18"),
]

def check_direction(response_text: str, change_pct: float) -> tuple[str, bool]:
    first_line = response_text.strip().splitlines()[0].lower()
    if "buy" in first_line:
        return "Buy", change_pct > 0
    elif "sell" in first_line:
        return "Sell", change_pct < 0
    return "Hold", abs(change_pct) < 1.5

async def run_benchmark():
    rows = []

    for stock in STOCKS:
        for start_date, end_date in WINDOWS:
            bt = simple_backtest(stock, end_date, days_forward=5)
            if bt["status"] != "success":
                print(f"Skipping {stock} ({end_date}): {bt.get('message')}")
                continue
            actual_pct = bt["change_pct"]

            # 1. Run Single-Agent Baseline
            t0 = time.time()
            s_out = await run_single_analysis(stock, start_date, end_date)
            s_lat = round(time.time() - t0, 2)
            s_dec, s_corr = check_direction(s_out["response"], actual_pct)

            rows.append({
                "architecture": "Single-Agent",
                "stock": stock,
                "end_date": end_date,
                "faithfulness": s_out["faithfulness"],
                "answer_relevancy": s_out["answer_relevancy"],
                "latency_sec": s_lat,
                "decision": s_dec,
                "actual_5d_pct": actual_pct,
                "directional_correct": s_corr
            })

            # 2. Run Multi-Agent Supervisor
            t0 = time.time()
            m_out = await run_analysis(stock, start_date, end_date)
            m_lat = round(time.time() - t0, 2)
            m_dec, m_corr = check_direction(m_out["response"], actual_pct)

            rows.append({
                "architecture": "Multi-Agent Supervisor",
                "stock": stock,
                "end_date": end_date,
                "faithfulness": m_out["faithfulness"],
                "answer_relevancy": m_out["answer_relevancy"],
                "latency_sec": m_lat,
                "decision": m_dec,
                "actual_5d_pct": actual_pct,
                "directional_correct": m_corr
            })

            print(f"Done: {stock} ({end_date}) | Single Faith: {s_out['faithfulness']} | Multi Faith: {m_out['faithfulness']}")

    df = pd.DataFrame(rows)
    df.to_csv("benchmark_results.csv", index=False)

    summary = df.groupby("architecture")[["faithfulness", "answer_relevancy", "latency_sec", "directional_correct"]].mean()
    print("\n=== FINAL BENCHMARK SUMMARY ===")
    print(summary.round(3))

if __name__ == "__main__":
    asyncio.run(run_benchmark())

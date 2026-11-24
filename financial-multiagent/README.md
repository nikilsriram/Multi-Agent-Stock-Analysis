# Multi-Agent vs. Monolithic Single-Agent Financial Analysis Benchmark

An empirical evaluation comparing a **Hierarchical Multi-Agent Supervisor** against a **Pure-LangGraph Monolithic Single-Agent** on multi-modal financial analysis (news headlines, Wall Street analyst consensus trends, and historical OHLCV price action) using **RAGAS** grounding metrics and **5-day walk-forward backtesting**.

---

## Key Research Findings (n = 50 Historical Ticker Windows)

Across 10 liquid equities (`AAPL`, `NVDA`, `MSFT`, `TSLA`, `AMZN`, `JPM`, `XOM`, `UNH`, `WMT`, `META`) evaluated across 5 leak-free historical windows in early 2026 (`gpt-4o-mini`, `temperature=0`):

| Architecture | RAGAS Faithfulness | Answer Relevancy | Avg. Latency (s) | 5-Day Directional Acc. | Active Trade Rate (`Buy`/`Sell`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-Agent (Pure LangGraph)** | **0.904** | **0.442** | **28.33s** | 18.0% (`9/50`) | 0.0% (`50/50 Hold`) |
| **Multi-Agent Supervisor** | 0.752 | 0.339 | 42.80s | **22.0%** (`11/50`) | **40.0%** (`10 Buy, 10 Sell, 30 Hold`) |

### Takeaway Analysis
1. **Conservative Abstention in Monolithic Agents (`100% Hold`):** When raw news, analyst ratings, and historical price tables are injected into a single LLM context window, the monolithic agent defaults to `Hold` on **50/50 runs (100%)**. This risk-averse abstention artificially inflates literal RAGAS Faithfulness (`0.904`) while rendering the system unusable for directional decision-making.
2. **Sub-Agent Decomposition Unlocks Decisive Reasoning (`40% Active Rate`):** Splitting context ingestion across three specialized sub-agents (`News`, `Analyst Consensus`, `Market Statistics`) before supervisor synthesis enables active `Buy`/`Sell` positioning (`40.0%` active rate) and improves overall 5-day directional accuracy (`22.0%` vs. `18.0%`), capturing major moves such as `UNH` (`+14.61%`) and `AMZN` (`+5.94%`).
3. **Synthesis Faithfulness & Latency Trade-Off:** Multi-agent synthesis introduces a `1.51x` latency overhead (`42.80s` vs. `28.33s`) and lowers literal RAGAS Faithfulness (`0.752` vs. `0.904`) due to higher-level cross-source abstraction rather than verbatim tool copying.

---

## System Architectures

* **`single_agent.py` (Baseline):** A pure-LangGraph `StateGraph` (`agent` \(\leftrightarrow\) `ToolNode`) where one LLM directly invokes `get_news`, `get_social_media`, and `get_statistics` and synthesizes a final `Buy / Sell / Hold` decision.
* **`main.py` (Multi-Agent Supervisor):** A hierarchical supervisor-worker architecture delegating tool execution and domain reasoning to three dedicated specialist agents before aggregating their reports into a final decision.
* **Zero Look-Ahead Bias:** All historical Finnhub analyst consensus trends (`period <= end_date`) and `yfinance` price series are strictly filtered to prevent future data leakage prior to the 5-day forward return check (`simple_backtest`).

---

## Repository Structure

```text
├── main.py                  # Hierarchical Multi-Agent Supervisor + RAGAS evaluation
├── single_agent.py          # Pure-LangGraph Single-Agent baseline + walk-forward backtester
├── eval.py                  # Automated 50-window (100-run) benchmark harness
└── benchmark_results.csv    # Full raw evaluation logs across all 100 runs
```

---

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install langgraph langchain-core langchain-openrouter ragas yfinance finnhub-python eventregistry pandas python-dotenv openai
   ```

2. **Configure environment variables (`.env`):**
   ```env
   OPENROUTER_API_KEY=your_key_here
   EVENTREGISTRY_API_KEY=your_key_here
   FINNHUB_API_KEY=your_key_here
   ```

3. **Run the automated benchmark:**
   ```bash
   python eval.py
   ```
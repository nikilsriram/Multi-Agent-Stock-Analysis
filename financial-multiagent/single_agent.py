import os
import time
import requests
from typing import TypedDict, Annotated

import yfinance as yf
import finnhub
from dotenv import load_dotenv
from eventregistry import EventRegistry, QueryArticlesIter, QueryItems

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

import pandas as pd


load_dotenv()

finnhub_api_key = os.getenv("FINNHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=finnhub_api_key)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_news(stock: str, start_date: str, end_date: str) -> str:
    """Fetch news articles for a stock between start_date and end_date."""
    try:
        er = EventRegistry(apiKey=os.getenv("EVENTREGISTRY_API_KEY"))
        q = QueryArticlesIter(
            keywords=QueryItems.OR([stock]),
            dateStart=start_date,
            dateEnd=end_date
        )
        articles = []
        for art in q.execQuery(er, sortBy="date", maxItems=5):
            articles.append({
                "title": art.get("title"),
                "source": art.get("source"),
                "date": art.get("date")
            })
        return str(articles)
    except Exception as e:
        return f"Error fetching news: {str(e)}"


@tool
def get_social_media(stock: str, start_date: str, end_date: str) -> str:
    """Fetch historical analyst recommendation trends up to end_date."""
    for _ in range(3):
        try:
            data = finnhub_client.recommendation_trends(stock)
            historical = [d for d in data if d.get("period", "9999-12-31") <= end_date]
            return str(historical[:3])
        except requests.exceptions.ReadTimeout:
            time.sleep(2)
        except Exception as e:
            return f"Error fetching social data: {str(e)}"
    return "Error: Finnhub timeout"


@tool
def get_statistics(stock: str, start_date: str, end_date: str) -> str:
    """Fetch historical stock price data between start_date and end_date."""
    try:
        t = yf.Ticker(stock)
        hist = t.history(start=start_date, end=end_date)
        return str({"history": hist.tail(10).to_dict()})
    except Exception as e:
        return f"Error fetching statistics: {str(e)}"


tools = [get_news, get_social_media, get_statistics]

llm = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
).bind_tools(tools)


def my_agent(state: AgentState) -> dict:
    system_prompt = SystemMessage(content="""You MUST call all three tools (get_news, get_social_media, get_statistics) before answering.
Then combine the outputs into this exact format:

Final Decision: Buy / Sell / Hold

News Reasoning: ...
Social Reasoning: ...
Stats Reasoning: ...

Final Explanation: ...""")
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return "end"
    return "continue"


graph = StateGraph(AgentState)
graph.add_node("agent", my_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")

single_agent_app = graph.compile()


ragas_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


ragas_llm = llm_factory("openai/gpt-4o-mini", client=ragas_client, max_tokens=4096)
scorer = Faithfulness(llm=ragas_llm)
answer_scorer = AnswerRelevancy(
    llm=ragas_llm,
    embeddings=OpenAIEmbeddings(model="openai/text-embedding-3-small", client=ragas_client)
)

async def run_single_analysis(stock: str, start_date: str, end_date: str):
    query = f"{stock} {start_date} {end_date}"
    result = single_agent_app.invoke({
        "messages": [HumanMessage(content=query)]
    })

    response = result["messages"][-1].content
    retrieved_contexts = [
        str(m.content) for m in result["messages"]
        if getattr(m, "type", None) == "tool"
    ]

    faith_score = await scorer.ascore(
        user_input=query,
        response=response,
        retrieved_contexts=retrieved_contexts
    )
    rel_score = await answer_scorer.ascore(
        user_input=query,
        response=response
    )

    return {
        "response": response,
        "faithfulness": round(faith_score.value, 3),
        "answer_relevancy": round(rel_score.value, 3),
        "contexts": retrieved_contexts
    }


def simple_backtest(stock: str, end_date: str, days_forward: int = 5):

    df = yf.download(
        stock,
        start=end_date,
        end=pd.to_datetime(end_date) + pd.Timedelta(days=days_forward + 5),
        progress=False
    )

    if df is None or df.empty:
        return {
            "status": "error",
            "message": "No data returned",
            "data": []
        }

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()

    if len(close) < 2:
        return {
            "status": "error",
            "message": "Not enough data",
            "data": []
        }

    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])

    change_pct = ((end_price - start_price) / start_price) * 100

    return {
        "status": "success",
        "start_price": start_price,
        "end_price": end_price,
        "change_pct": round(change_pct, 2),
        "data": close.tolist()
    }
    
if __name__ == "__main__":
    import asyncio

    out = asyncio.run(run_single_analysis("AAPL", "2026-03-01", "2026-03-15"))
    bt = simple_backtest("AAPL", "2026-03-15")

    print("Faithfulness:", out["faithfulness"])
    print("Answer Relevancy:", out["answer_relevancy"])
    print("5-Day Return %:", bt.get("change_pct"))
    print("\nDecision:\n", out["response"])

import os
import datetime
import yfinance as yf
import finnhub
from dotenv import load_dotenv
import pandas as pd

from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import OpenAIEmbeddings
from ragas import aevaluate

from datasets import Dataset 
from ragas.metrics.collections import AnswerRelevancy
from ragas import evaluate


from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
eventregistry_api_key = os.getenv("EVENTREGISTRY_API_KEY")
finnhub_api_key = os.getenv("FINNHUB_API_KEY")

finnhub_client = finnhub.Client(api_key=finnhub_api_key)

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key
)

llm = llm_factory(
    "openai/gpt-4o-mini",
    client=client,
    max_tokens=4096,
)

scorer = Faithfulness(llm=llm)

answer_scorer = AnswerRelevancy(
    llm=llm,
    embeddings=OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        client=client
    )
)

model = ChatOpenRouter(
    model="openai/gpt-4o-mini"
)




@tool
def get_news(stock: str, start_date: str, end_date: str) -> str:
    """Runs news subagent and returns reasoning."""
    
    try:
        er = EventRegistry(apiKey=eventregistry_api_key)

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



import time
import requests
@tool
def get_social_media(stock: str, start_date: str, end_date: str) -> str:
    """Runs social media subagent and returns reasoning"""
    for _ in range(3):
        try:
            data = finnhub_client.recommendation_trends(stock)
            return str(data)

        except requests.exceptions.ReadTimeout:
            time.sleep(2)

        except Exception as e:
            return f"Error fetching social data: {str(e)}"

    return "Error: Finnhub timeout"

@tool
def get_statistics(stock: str, start_date: str, end_date: str) -> str:
    """Runs statistics subagent and returns reasoning."""
    
    try:
        t = yf.Ticker(stock)

        info = t.fast_info  
        hist = t.history(start=start_date, end=end_date)

        return str({
            "info": dict(info),
            "history": hist.tail(10).to_dict()
        })

    except Exception as e:
        return f"Error fetching statistics: {str(e)}"



news_agent = create_agent(
    model,
    tools=[get_news],
    system_prompt=(
        "You analyze stock news. "
        "Explain bullish and bearish factors clearly. "
        "Do NOT give buy/sell/hold."
    )
)

social_agent = create_agent(
    model,
    tools=[get_social_media],
    system_prompt=(
        "You analyze social media sentiment. "
        "Break into bullish, bearish, neutral sentiment. "
        "Do NOT give buy/sell/hold."
    )
)

stats_agent = create_agent(
    model,
    tools=[get_statistics],
    system_prompt=(
        "You analyze financial data. "
        "Focus on valuation, growth, risk, momentum. "
        "Do NOT give buy/sell/hold."
    )
)



@tool(description="Run news agent and return reasoning.")
def analyze_news(request: str) -> str:
    try:
        result = news_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        return result["messages"][-1].content
    except Exception as e:
        return f"News agent failed: {str(e)}"


@tool(description="Run social media agent and return reasoning.")
def analyze_social_media(request: str) -> str:
    try:
        result = social_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        return result["messages"][-1].content
    except Exception as e:
        return f"Social agent failed: {str(e)}"


@tool(description="Run statistics agent and return reasoning.")
def analyze_statistics(request: str) -> str:
    try:
        result = stats_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        return result["messages"][-1].content

    except Exception as e:
        return f"Statistics agent failed :{str(e)}"


supervisor = create_agent(
    model,
    tools=[analyze_news, analyze_social_media, analyze_statistics],
    system_prompt="""
You MUST follow this exact process:

1. Call analyze_news tool
2. Call analyze_social_media tool
3. Call analyze_statistics tool

Do not skip any tool.

Then combine outputs.

FINAL OUTPUT FORMAT:
Final Decision: Buy / Sell / Hold

News Reasoning: ...
Social Reasoning: ...
Stats Reasoning: ...

Final Explanation: ...
"""
)



async def run_analysis(stock, start_date, end_date):
    try:
        query = f"{stock} {start_date} {end_date}"

        result = supervisor.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        agent = result["messages"][-1].content
        response = result["messages"][-1].content

        retrieved_contexts = []

        for message in result["messages"]:
            if getattr(message, "type", None) == "tool":
                retrieved_contexts.append(str(message.content))

        score = await scorer.ascore(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_contexts
        )       

        print(f"RAGAS Faithfulness Score: {score.value:.2f}")

        scoreTwo = await answer_scorer.ascore(
            user_input=query,
            response=response
        )

        print(f"RAGAS ANSWER RELEVANCY SCORE: {scoreTwo.value:.2f}")

        return response

    except Exception as e:
        return f"System failed: {str(e)}"

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
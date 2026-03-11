import os
from langchain.tools import tool
import requests
from rag import build_retriever

retriever = build_retriever()


@tool
def search_knowledge_base(question: str) -> str:
    """Search the crypto knowledge base for relevant information."""
    results = retriever.invoke(question)

    if not results:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk = doc.page_content.strip()
        formatted.append(f"[{i}] Source: {source}\n{chunk}")

    return "\n\n".join(formatted)


@tool
def list_documents(_: str = "") -> str:
    """List available documents in the crypto knowledge base."""
    
    folder = "crypto_docs"
    files = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
    
    return "Available knowledge base documents:\n" + "\n".join(files)

@tool
def get_crypto_market_data(symbol: str) -> str:
    """Get current AUD market data for a cryptocurrency from CoinGecko.

    Expected input is a CoinGecko coin id, e.g. bitcoin, ethereum, solana.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "aud",
        "ids": symbol.lower(),
        "price_change_percentage": "1h,24h,7d,30d",
        "sparkline": "false",
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException:
        return "Unable to fetch cryptocurrency market data right now."

    if not data:
        return "Cryptocurrency not found."

    coin = data[0]

    def fmt_money(value):
        return f"${value:,.2f} AUD" if value is not None else "N/A"

    def fmt_num(value):
        return f"{value:,.0f}" if value is not None else "N/A"

    def fmt_pct(value):
        return f"{value:.2f}%" if value is not None else "N/A"

    lines = [
        f"{coin.get('name', symbol.title())} ({coin.get('symbol', '').upper()})",
        f"Price: {fmt_money(coin.get('current_price'))}",
        f"1h: {fmt_pct(coin.get('price_change_percentage_1h_in_currency'))}",
        f"24h: {fmt_pct(coin.get('price_change_percentage_24h_in_currency'))}",
        f"7d: {fmt_pct(coin.get('price_change_percentage_7d_in_currency'))}",
        f"30d: {fmt_pct(coin.get('price_change_percentage_30d_in_currency'))}",
        f"24h High: {fmt_money(coin.get('high_24h'))}",
        f"24h Low: {fmt_money(coin.get('low_24h'))}",
        f"24h Volume: {fmt_money(coin.get('total_volume'))}",
        f"Market Cap: {fmt_money(coin.get('market_cap'))}",
        f"Market Cap Rank: {coin.get('market_cap_rank', 'N/A')}",
        f"Circulating Supply: {fmt_num(coin.get('circulating_supply'))}",
        f"Total Supply: {fmt_num(coin.get('total_supply'))}",
        f"Max Supply: {fmt_num(coin.get('max_supply'))}",
        f"Last Updated: {coin.get('last_updated', 'N/A')}",
    ]

    return "\n".join(lines)

@tool
def retrieve_topic_info(topic: str) -> str:
    """Return relevant information about a crypto topic from the knowledge base."""
    docs = retriever.invoke(topic)

    if not docs:
        return "No information found."

    text = "\n\n".join(d.page_content for d in docs)
    return text[:1000]
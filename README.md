# Crypto AI Agent
A simple AI agent built with LangChain and GPT‑4o mini that answers cryptocurrency questions using retrieval-augmented generation (RAG).

The agent retrieves information from a small knowledge base containing documents about topics such as:
- Bitcoin
- Ethereum
- DeFi
- Blockchain
- Staking
- Mining

It uses semantic search to retrieve relevant information from these documents and generates answers using the OpenAI API.

The agent can also retrieve live cryptocurrency market data (such as price, market cap, and volume in AUD) using the CoinGecko API.

## Features
- Retrieval-augmented generation (RAG) using a FAISS vector database

- Crypto knowledge base search

- Topic summaries from the knowledge base

- Live cryptocurrency market data from CoinGecko

- Simple interactive CLI interface

## Requirements
- Python 3.10+
- An OpenAI API key

## Setup
### 1. Install dependencies
```pip install -r requirements.txt```

### 2. Configure OpenAI API key
In a `.env` file in project root
```OPENAI_API_KEY=your_api_key_here```

You can create an API key here:
https://platform.openai.com/api-keys

### 3. Run the agent
```python agent.py```

### Example Usage
Ask me a question: What is Bitcoin?

Ask me a question: What are liquidity pools?

Ask me a question: What is the price of bitcoin?

Ask me a question: Summarize staking

Ask me a question: What is the current price of Bitcoin?
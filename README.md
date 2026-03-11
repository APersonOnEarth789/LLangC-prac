# Crypto AI Agent
A simple LLM agent built with LangChain and OpenAI's gpt-4o-mini model that answers cryptocurrency questions using retrieval-augmented generation (RAG).

The agent retrieves information from a small knowledge base containing documents about:
- Bitcoin
- Ethereum
- DeFi
and generates answers using an OpenAI API call.

## Requirements
- Python 3.10+

## Setup
#### Install dependencies
```pip install -r requirements.txt```

#### Set up OpenAI API key
In a `.env` file in project root
```OPENAI_API_KEY=your_api_key_here```

You can create an API key here:
https://platform.openai.com/api-keys

#### Run the agent
```python agent.py```

#### Example Usage
Ask me a question: What is Bitcoin?
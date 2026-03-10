# Crypto AI Agent
A simple LLM agent built with LangChain and a local Qwen 3.5 model that answers cryptocurrency questions using retrieval-augmented generation (RAG).

The agent retrieves information from a small knowledge base containing documents about:
- Bitcoin
- Ethereum
- DeFi

and generates answers using a local LLM via Ollama.

## Requirements
- Python 3.10+
- Ollama

## Setup
#### Install Ollama
brew install ollama

#### Pull the model
ollama pull llama3

#### Install dependencies
pip install -r requirements.txt

#### Run the model
python agent.py
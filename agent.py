from langchain.agents import create_agent
from config import llm
from tools import search_knowledge_base, list_topics

system_prompt = """
You are a crypto research assistant.

Your job:
- Answer user questions clearly and accurately.
- Use the search_knowledge_base tool whenever the question asks about Bitcoin, Ethereum, DeFi, blockchain concepts, or anything factual from the knowledge base.
- If tool results are available, base your answer only on those results.
- If the answer is not in the tool results, say you are not sure.
- Be concise but informative.
- When possible, mention the source file used.
"""

agent = create_agent(
    model=llm,
    tools=[search_knowledge_base, list_topics],
    system_prompt=system_prompt,
)

if __name__ == "__main__":
    print("Crypto Agent ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("Ask me a question: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )

        messages = response["messages"]
        final_message = messages[-1].content
        print(f"\nAgent: {final_message}\n")
import csv
import os
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv

# AgentCore Runtime
from bedrock_agentcore.runtime import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

# Load runtime environment variables from .env.local first, then fallback to .env
load_dotenv(".env.local", override=False)


# -----------------------------
# Data Loading
# -----------------------------
def load_faq_csv(path: str) -> List[Document]:
    """Load FAQ entries from CSV into LangChain Documents."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            docs.append(
                Document(page_content=f"Q: {q}\nA: {a}")
            )
    return docs


docs = load_faq_csv("./lauki_qna.csv")


# -----------------------------
# Embeddings (✅ Bedrock-native)
# -----------------------------
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)


# -----------------------------
# Vector Store
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

chunks = splitter.split_documents(docs)

store = FAISS.from_documents(chunks, embeddings)


# -----------------------------
# Tools
# -----------------------------
@tool
def search_faq(query: str) -> str:
    """Search FAQ knowledge base for relevant information."""
    results = store.similarity_search(query, k=3)

    if not results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join(
        f"FAQ Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    )

    return f"Found {len(results)} relevant FAQ entries:\n\n{context}"


@tool
def search_detailed_faq(query: str, num_results: int = 5) -> str:
    """Search FAQ knowledge base with more results."""
    results = store.similarity_search(query, k=num_results)

    if not results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join(
        f"FAQ Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    )

    return f"Found {len(results)} detailed FAQ entries:\n\n{context}"


@tool
def reformulate_query(original_query: str, focus_aspect: str) -> str:
    """Reformulate and search a specific aspect of the question."""
    reformulated = f"{focus_aspect} related to {original_query}"
    results = store.similarity_search(reformulated, k=3)

    if not results:
        return f"No results found for aspect: {focus_aspect}"

    context = "\n\n---\n\n".join(
        f"Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    )

    return f"Results for '{focus_aspect}' aspect:\n\n{context}"


tools = [
    search_faq,
    search_detailed_faq,
    reformulate_query,
]


# -----------------------------
# LLM
# -----------------------------
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


system_prompt = """You are a helpful FAQ assistant with access to a knowledge base.

Your goal is to answer user questions accurately using the available tools.

Guidelines:
1. Start by using the search_faq tool to find relevant information
2. If the initial search doesn't provide enough info, use search_detailed_faq for more results
3. If the query is complex, use reformulate_query to search different aspects
4. Synthesize information from multiple tool calls if needed
5. Always provide a clear, concise answer based on the retrieved information
6. If you cannot find relevant information, clearly state that

Think step-by-step and use tools strategically to provide the best answer."""


agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)


# -----------------------------
# AgentCore Entrypoint
# -----------------------------
@app.entrypoint
def agent_invocation(payload, context):
    print("Payload:", payload)
    print("Context:", context)

    query = payload.get("prompt", "")
    result = agent.invoke({"messages": [("human", query)]})

    return {
        "result": result["messages"][-1].content
    }


if __name__ == "__main__":
    app.run()
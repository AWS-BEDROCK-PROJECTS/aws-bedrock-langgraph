import csv
import os
from typing import List
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
from langchain.agents import create_agent

# Load environment variables from a local .env file so API keys are available.
# This allows the script to read GROQ_API_KEY from the environment below.
_ = load_dotenv()


def load_faq_csv(path: str) -> List[Document]:
    """Load FAQ entries from a CSV file and convert them into Documents.

    The CSV is expected to contain `question` and `answer` columns.
    Each row is combined into a single Document with a Q/A format.
    """
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            docs.append(Document(page_content=f"Q: {q}\nA: {a}"))
    return docs


# Load the local FAQ dataset once when the module is imported.
docs = load_faq_csv("./lauki_qna.csv")
# Initialize the embedding model used to convert text into vectors.
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Split documents into chunks so long FAQ entries can be indexed effectively.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = splitter.split_documents(docs)

# Build the FAISS vector store for semantic similarity search.
store = FAISS.from_documents(chunks, emb)


@tool
def search_faq(query: str) -> str:
    """Search the FAQ knowledge base for relevant information.

    This tool is intended for general FAQ queries and returns the top 3 matches.
    """
    results = store.similarity_search(query, k=3)

    if not results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    return f"Found {len(results)} relevant FAQ entries:\n\n{context}"


@tool
def search_detailed_faq(query: str, num_results: int = 5) -> str:
    """Search the FAQ knowledge base with more results for complex queries.

    Use this tool when a wider set of candidate documents may provide more
    useful context for the answer.
    """
    results = store.similarity_search(query, k=num_results)

    if not results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    return f"Found {len(results)} detailed FAQ entries:\n\n{context}"


@tool
def reformulate_query(original_query: str, focus_aspect: str) -> str:
    """Reformulate the query to focus on a specific aspect.

    If a direct search returns results that are too broad, this tool creates a
    more targeted query by combining the user's original input with a focus
    topic.
    """
    reformulated = f"{focus_aspect} related to {original_query}"
    results = store.similarity_search(reformulated, k=3)

    if not results:
        return f"No results found for aspect: {focus_aspect}"

    context = "\n\n---\n\n".join([
        f"Entry {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    return f"Results for '{focus_aspect}' aspect:\n\n{context}"



# Expose tool functions to the agent.
tools = [search_faq, search_detailed_faq, reformulate_query]

# Create the chat model configuration using the Groq backend.
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Instructions that guide the agent on how to use tools and answer questions.
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

# Build the agent instance with the configured model and tools.
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)

if __name__ == "__main__":
    # Example direct execution of the agent with a sample prompt.
    result = agent.invoke({"messages": [("human", "Explain roaming activation.")]})
    print(result['messages'][-1].content)

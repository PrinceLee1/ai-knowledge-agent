import os
import sys
from typing import Final

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY: Final[str | None] = os.getenv("OPENAI_API_KEY")
MODEL_NAME: Final[str] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CHUNK_SIZE: Final[int] = 1000
CHUNK_OVERLAP: Final[int] = 100

TOP_K: Final[int] = 4


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_environment() -> None:
    """Validate required environment variables."""

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )


def get_arguments() -> tuple[str, str, str]:
    """Read and validate CLI arguments.

    Expected arguments:
        1. User prompt
        2. Knowledge source URL
        3. Consultant instructions
    """

    if len(sys.argv) < 4:
        raise ValueError(
            "Usage: python chatbot.py <prompt> <knowledge_url> <instructions>"
        )

    prompt = sys.argv[1].strip()
    knowledge_url = sys.argv[2].strip()
    instructions = sys.argv[3].strip()

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    if not knowledge_url:
        raise ValueError("Knowledge source URL cannot be empty.")

    if not instructions:
        raise ValueError("Consultant instructions cannot be empty.")

    return prompt, knowledge_url, instructions


# ---------------------------------------------------------------------------
# Knowledge Retrieval
# ---------------------------------------------------------------------------

def load_knowledge_source(url: str) -> list[Document]:
    """Load documents from the consultant's knowledge source."""

    loader = WebBaseLoader(url)

    documents = loader.load()

    if not documents:
        raise RuntimeError(
            "No content could be loaded from the knowledge source."
        )

    return documents


def create_retriever(documents: list[Document]):
    """Create a vector-based retriever from source documents."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise RuntimeError(
            "The knowledge source did not produce any document chunks."
        )

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store.as_retriever(
        search_kwargs={"k": TOP_K}
    )


# ---------------------------------------------------------------------------
# AI Agent
# ---------------------------------------------------------------------------

def create_agent(retriever, consultant_instructions: str):
    """Create the RAG-powered consultant agent."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
{instructions}

Use the following retrieved context to help answer the user's question.

<context>
{context}
</context>

Guidelines:
- Prefer information from the provided context when it is relevant.
- Do not invent company-specific facts.
- If the context does not contain enough information, clearly say so.
- Answer naturally and directly.
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    def format_documents(documents: list[Document]) -> str:
        return "\n\n".join(
            document.page_content
            for document in documents
        )

    # Retrieve relevant knowledge before calling the model.
    def retrieve_context(inputs: dict) -> dict:
        documents = retriever.invoke(inputs["question"])

        return {
            **inputs,
            "context": format_documents(documents),
        }

    return (
        retrieve_context
        | prompt.partial(instructions=consultant_instructions)
        | llm
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the consultant knowledge agent."""

    validate_environment()

    prompt, knowledge_url, consultant_instructions = get_arguments()

    documents = load_knowledge_source(knowledge_url)

    retriever = create_retriever(documents)

    agent = create_agent(
        retriever=retriever,
        consultant_instructions=consultant_instructions,
    )

    response = agent.invoke(
        {
            "question": prompt,
        }
    )

    print(response.content)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"Agent execution failed: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
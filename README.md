# AI Knowledge Agent

A production-oriented **RAG (Retrieval-Augmented Generation) knowledge agent** built with Python, LangChain, OpenAI, and Chroma.

The agent allows different consultants or AI personas to be connected to their own domain-specific knowledge source. When a user asks a question, the system retrieves relevant information from the configured knowledge source and provides that context to an LLM before generating a response.

## Architecture

```text
                Knowledge Source
                       │
                       ▼
                Web Document Loader
                       │
                       ▼
                Text Splitter
                       │
                       ▼
              OpenAI Embeddings
                       │
                       ▼
                 Chroma Vector
                    Store
                       │
                       ▼
                  Retriever
                       │
                       │ Relevant Context
                       ▼
User Question ──► Prompt Construction
                       │
                       ▼
                    LLM
                       │
                       ▼
               Grounded Response
```

## Features

* Retrieval-Augmented Generation (RAG)
* Domain-specific knowledge retrieval
* Semantic search using vector embeddings
* OpenAI LLM integration
* Chroma vector store
* Configurable consultant instructions
* Configurable LLM model
* Environment-based API configuration
* Input validation and error handling
* Clean separation between ingestion, retrieval, prompting, and generation

## Tech Stack

* **Python**
* **LangChain**
* **OpenAI**
* **Chroma**
* **OpenAI Embeddings**
* **WebBaseLoader**
* **python-dotenv**

## How It Works

The agent follows five main steps.

### 1. Load the knowledge source

A consultant is associated with a knowledge source, such as a company website or documentation page.

```python
loader = WebBaseLoader(knowledge_url)
documents = loader.load()
```

### 2. Split the content

The loaded documents are divided into smaller chunks to make retrieval more effective.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
```

### 3. Create embeddings

Each document chunk is converted into a vector representation using OpenAI embeddings.

```python
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
)
```

The embeddings are stored in Chroma for similarity-based retrieval.

### 4. Retrieve relevant context

When a user asks a question, the retriever searches the vector store for the most relevant document chunks.

```python
documents = retriever.invoke(question)
```

The retrieved content is then passed into the model as context.

### 5. Generate the response

The retrieved context and consultant-specific instructions are combined with the user's question and sent to the LLM.

```text
Consultant Instructions
        +
Retrieved Knowledge
        +
User Question
        ↓
       LLM
        ↓
Contextual Response
```

This helps the model provide answers grounded in the consultant's configured knowledge source rather than relying only on its pretrained knowledge.

## Project Structure

```text
ai-knowledge-agent/
│
├── chatbot.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/PrinceLee1/ai-knowledge-agent.git>
cd ai-knowledge-agent
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```bash
cp .env.example .env
```

Then add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key
```

You can optionally configure the model:

```env
OPENAI_MODEL=gpt-4o-mini
```

## Running the Agent

The application expects three arguments:

```text
python chatbot.py "<question>" "<knowledge_source_url>" "<consultant_instructions>"
```

Example:

```bash
python chatbot.py \
"What services does this company provide?" \
"https://example.com" \
"You are a helpful consultant representing this company. Answer using the provided company knowledge."
```

The agent will:

1. Load the configured knowledge source.
2. Split the content into retrievable chunks.
3. Generate embeddings.
4. Store the vectors in Chroma.
5. Retrieve relevant information for the question.
6. Combine the retrieved information with the consultant instructions.
7. Generate the final response using the configured OpenAI model.

## Example Use Case

Imagine a consultant configured for a software company.

The knowledge source could contain:

```text
https://example.com/company
```

The user asks:

```text
What products does the company offer?
```

The retrieval pipeline finds the relevant sections of the company's information and provides them to the model.

The model then generates an answer based on the retrieved context.

## Design Considerations

This project intentionally separates the main stages of the RAG pipeline:

```text
Knowledge Ingestion
        ↓
Document Processing
        ↓
Embedding / Vector Storage
        ↓
Retrieval
        ↓
Prompt Construction
        ↓
LLM Generation
```

This makes each component easier to replace or extend.

For example, the vector database could be replaced with another retrieval backend, or the LLM provider could be changed without redesigning the entire application.

## Production Considerations

The current implementation is designed as a compact demonstration of the RAG architecture.

For a production deployment, I would separate the **knowledge ingestion pipeline** from the **query pipeline**.

Instead of rebuilding the vector store for every question:

```text
User Query
    ↓
Load Website
    ↓
Create Embeddings
    ↓
Create Vector Store
    ↓
Answer
```

I would pre-process and index knowledge sources asynchronously:

```text
Knowledge Source
      ↓
Ingestion Job
      ↓
Chunking
      ↓
Embeddings
      ↓
Persistent Vector Database
```

Then query time becomes:

```text
User Query
      ↓
Retriever
      ↓
Relevant Context
      ↓
LLM
      ↓
Response
```

Additional production improvements could include:

* Persistent vector storage
* Incremental document indexing
* Metadata filtering
* Source citations
* Retrieval evaluation
* Response evaluation
* Caching
* Observability and tracing
* Retry and timeout handling
* Rate limiting
* Background ingestion jobs
* Authentication and authorization
* Human escalation for uncertain responses

## Why I Built This

The goal of this project was to move beyond a simple prompt → response architecture and demonstrate how an LLM can be connected to external knowledge.

The system combines:

* LLM generation
* Semantic retrieval
* Vector embeddings
* Domain-specific context
* Dynamic instructions
* Application-level orchestration

This pattern can be extended into more capable AI agents that use external tools, APIs, databases, and business workflows to perform actions rather than simply generate text.

## License

This project is provided as a portfolio/code sample.

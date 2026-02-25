# Day 16: RAG Fundamentals 📑
### Week 3 — RAG, Fine-Tuning & Vector Databases

---

## 🎯 Learning Objectives

By the end of today, you will:
- Understand the full RAG architecture (ingestion + retrieval + generation)
- Build a complete document ingestion pipeline
- Implement a retrieval + generation chain
- Handle PDF, text, and web documents
- Build a PDF Q&A chatbot from scratch

**Estimated Time:** 4–4.5 hours  
**Difficulty:** ⭐⭐⭐⭐ Intermediate-Advanced  
**Prerequisites:** Day 15 (Embeddings & Vector DBs)

---

## 📚 Section 1: RAG Architecture

### 1.1 What Is RAG?

**RAG = Retrieval-Augmented Generation**

The core insight: instead of relying solely on an LLM's training data (which is outdated and finite), retrieve relevant facts from your own documents at query time and include them as context.

```
┌────────────────────── RAG PIPELINE ──────────────────────────┐
│                                                               │
│  INGESTION (offline, do once)                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ Documents │──►│  Split   │──►│  Embed   │──►│ VectorDB │  │
│  │ (PDF/Web) │   │  Chunks  │   │  Chunks  │   │ (Store)  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                               │
│  RETRIEVAL + GENERATION (at query time)                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  User    │──►│  Embed   │──►│ VectorDB │──►│ Top-K    │  │
│  │  Query   │   │  Query   │   │  Search  │   │  Chunks  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                        │                      │
│                              ┌─────────▼──────────┐          │
│                              │  Prompt = Query +   │          │
│                              │  Retrieved Context  │          │
│                              └─────────────────────┘          │
│                                        │                      │
│                              ┌─────────▼──────────┐          │
│                              │       LLM          │──► Answer │
│                              └────────────────────┘          │
└────────────────────────────────────────────────────────────── ┘
```

### 1.2 Why RAG?

| Problem (LLM without RAG) | RAG Solution |
|---------------------------|-------------|
| Knowledge cutoff date | Use current documents |
| "Hallucinated" facts | Ground answers in real sources |
| No access to private data | Use your own documents |
| Can't cite sources | Retrieval sources are trackable |
| Generic answers | Domain-specific context |

### 1.3 When NOT to Use RAG

- When questions are about general knowledge (GPT-4 already knows it)
- When documents change constantly (consider fine-tuning instead)
- When you need complex reasoning across ALL documents simultaneously
- When document count is very small (just put them in the context window)

---

## 📚 Section 2: Document Ingestion Pipeline

### 2.1 Setup

```bash
pip install langchain langchain-openai langchain-community \
            langchain-core chromadb pypdf docx2txt \
            sentence-transformers python-dotenv
```

### 2.2 Loading Documents from Multiple Sources

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document

def load_documents(sources: list[dict]) -> list[Document]:
    """
    Load documents from multiple sources.
    sources: list of {"type": "pdf|text|url|dir", "path": "..."}
    """
    all_docs = []
    
    for source in sources:
        source_type = source["type"]
        path = source["path"]
        
        print(f"📂 Loading {source_type}: {path}")
        
        if source_type == "pdf":
            loader = PyPDFLoader(path)
            docs = loader.load()
        elif source_type == "text":
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
        elif source_type == "url":
            loader = WebBaseLoader(path)
            docs = loader.load()
        elif source_type == "dir":
            loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
            docs = loader.load()
        else:
            print(f"  ⚠️ Unknown source type: {source_type}")
            continue
        
        # Add source metadata
        for doc in docs:
            doc.metadata["source_type"] = source_type
            doc.metadata["original_source"] = path
        
        print(f"  ✅ Loaded {len(docs)} document(s)")
        all_docs.extend(docs)
    
    return all_docs


# Example usage
sources = [
    {"type": "url", "path": "https://en.wikipedia.org/wiki/Generative_artificial_intelligence"},
    # {"type": "pdf", "path": "./my_document.pdf"},  # Uncomment for PDF
]

docs = load_documents(sources)
print(f"\nTotal loaded: {len(docs)} document(s)")
for doc in docs[:2]:
    print(f"  Source: {doc.metadata.get('source', 'N/A')}")
    print(f"  Content preview: {doc.page_content[:150]}...")
```

### 2.3 Chunking Strategies

Not all documents split the same way. Choose your chunking strategy based on content type:

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)

sample_markdown = """
# Introduction to RAG

RAG combines retrieval with generation for better AI responses.

## How Retrieval Works

The query is embedded and compared against stored vectors.

### Similarity Search

Cosine similarity is the most common metric.

## How Generation Works

The LLM receives the retrieved context plus the user query.
"""

# ── Strategy 1: Recursive Character (Default) ─────────────
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
recursive_chunks = recursive_splitter.split_text(sample_markdown)
print(f"Recursive splitter: {len(recursive_chunks)} chunks")

# ── Strategy 2: Markdown-Aware Splitter ───────────────────
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)
md_chunks = markdown_splitter.split_text(sample_markdown)
print(f"Markdown splitter: {len(md_chunks)} chunks")
for chunk in md_chunks:
    print(f"  Metadata: {chunk.metadata} | Content: {chunk.page_content[:60]}...")

# ── Strategy 3: Token-Based (most accurate for LLMs) ─────
token_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
token_chunks = token_splitter.split_text(sample_markdown)
print(f"Token splitter: {len(token_chunks)} chunks")

# ── Choosing chunk size ───────────────────────────────────
# Guidelines:
# - Small chunks (100-300 tokens): Better precision, more context needed
# - Medium chunks (300-600 tokens): Good balance
# - Large chunks (600-1000 tokens): More context per chunk, less precise retrieval
```

### 2.4 Building the Vector Index

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path

def build_vector_index(
    documents: list[Document],
    embed_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    persist_dir: str = "./vector_store"
) -> Chroma:
    """
    Complete pipeline: split documents → embed → store in ChromaDB
    """
    print(f"\n🔨 Building vector index...")
    print(f"  Input documents: {len(documents)}")
    
    # Step 1: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"  After splitting: {len(chunks)} chunks")
    
    # Step 2: Embed and store
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"  ✅ Indexed {vectorstore._collection.count()} chunks in ChromaDB")
    return vectorstore

def load_vector_index(
    embed_model: str = "all-MiniLM-L6-v2",
    persist_dir: str = "./vector_store"
) -> Chroma:
    """Load existing vector index"""
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
```

---

## 📚 Section 3: The RAG Chain

### 3.1 Simple RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv

load_dotenv()

def build_rag_chain(vectorstore: Chroma, model_name: str = "gpt-4o-mini"):
    """Build a complete RAG chain"""
    
    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Format documents for the prompt
    def format_docs(docs):
        return "\n\n---\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ])
    
    # RAG prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on provided context.
Use ONLY the information from the context to answer the question.
If the answer is not in the context, say "I don't have enough information in the provided documents."
Always cite which source your answer comes from."""),
        ("human", """Context:
{context}

Question: {question}

Answer based on the context:""")
    ])
    
    model = ChatOpenAI(model=model_name, temperature=0)
    parser = StrOutputParser()
    
    # Build the chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | parser
    )
    
    return rag_chain, retriever


def rag_with_sources(retriever, rag_chain, question: str) -> dict:
    """Run RAG and return answer + source documents"""
    # Get relevant documents
    docs = retriever.invoke(question)
    
    # Get answer
    answer = rag_chain.invoke(question)
    
    # Return with sources
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            }
            for doc in docs
        ],
        "num_sources": len(docs)
    }
```

### 3.2 RAG with Conversation History

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

session_store = {}

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def build_conversational_rag(vectorstore: Chroma, model_name: str = "gpt-4o-mini"):
    """RAG chain with conversation memory"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Step 1: Rephrase standing question given conversation history
    rephrase_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the conversation above, rephrase the follow-up question "
                  "as a standalone question for document search. "
                  "Return ONLY the rephrased question, no explanation.")
    ])
    
    model = ChatOpenAI(model=model_name, temperature=0)
    rephrase_chain = rephrase_prompt | model | StrOutputParser()
    
    # Step 2: Retrieve with the rephrased question
    def retrieve_with_history(input_dict: dict) -> str:
        history = input_dict.get("chat_history", [])
        if history:
            rephrased = rephrase_chain.invoke(input_dict)
        else:
            rephrased = input_dict["input"]
        docs = retriever.invoke(rephrased)
        return "\n\n".join([d.page_content for d in docs])
    
    # Step 3: Answer
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer questions based on the context. Cite sources."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {input}")
    ])
    
    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(retrieve_with_history))
        | answer_prompt
        | model
        | StrOutputParser()
    )
    
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
```

---

## 💻 Full Lab: PDF Q&A System

```python
# lab_day16_pdf_qa.py
"""Day 16 Lab — Complete PDF Q&A system using RAG"""

import os, sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

class PDFQASystem:
    """Complete PDF Q&A system using RAG"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", persist_dir: str = "./pdf_qa_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.rag_chain = None
        self.retriever = None
    
    def ingest(self, source: str, source_type: str = "auto"):
        """
        Ingest a document source.
        source_type: 'pdf', 'txt', 'url', or 'auto' (auto-detect)
        """
        print(f"\n📂 Ingesting: {source}")
        
        if source_type == "auto":
            if source.startswith("http"):
                source_type = "url"
            elif source.endswith(".pdf"):
                source_type = "pdf"
            else:
                source_type = "txt"
        
        if source_type == "pdf":
            loader = PyPDFLoader(source)
        elif source_type == "url":
            loader = WebBaseLoader(source)
        else:
            loader = TextLoader(source, encoding="utf-8")
        
        docs = loader.load()
        print(f"  Loaded {len(docs)} document(s)")
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)
        print(f"  Created {len(chunks)} chunks")
        
        # Store or add to existing index
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.vectorstore.add_documents(chunks)
        
        count = self.vectorstore._collection.count()
        print(f"  ✅ Total indexed chunks: {count}")
        
        # Build/rebuild the chain
        self._build_chain()
    
    def _build_chain(self):
        """Build or rebuild the RAG chain"""
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        def format_docs(docs):
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "")
                page_str = f", Page {page+1}" if page != "" else ""
                formatted.append(f"[Source {i}: {Path(source).name}{page_str}]\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyst. Answer questions strictly based on the provided document context.

Rules:
1. Only use information from the provided context
2. Always cite sources using [Source N] notation
3. If the answer is not in the context, say "This information is not in the provided documents."
4. Be precise and avoid speculation
"""),
            ("human", """DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")
        ])
        
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | self.model
            | StrOutputParser()
        )
    
    def ask(self, question: str, show_sources: bool = True) -> dict:
        """Ask a question about the ingested documents"""
        if self.rag_chain is None:
            return {"error": "No documents ingested yet. Call .ingest() first."}
        
        # Get answer
        answer = self.rag_chain.invoke(question)
        
        result = {
            "question": question,
            "answer": answer,
        }
        
        if show_sources:
            docs = self.retriever.invoke(question)
            result["sources"] = [
                {
                    "source": doc.metadata.get("source", "?"),
                    "page": doc.metadata.get("page", "N/A"),
                    "snippet": doc.page_content[:150] + "..."
                }
                for doc in docs
            ]
        
        return result
    
    def batch_ask(self, questions: list[str]) -> list[dict]:
        """Answer multiple questions efficiently"""
        results = []
        for q in questions:
            results.append(self.ask(q, show_sources=False))
        return results
    
    def interactive(self):
        """Interactive Q&A loop"""
        print("\n🤖 PDF Q&A System")
        print("Type 'quit' to exit, 'sources' to toggle source display")
        print("=" * 50)
        
        show_sources = True
        while True:
            question = input("\n❓ Your question: ").strip()
            if not question:
                continue
            elif question.lower() == "quit":
                break
            elif question.lower() == "sources":
                show_sources = not show_sources
                print(f"Sources: {'ON' if show_sources else 'OFF'}")
                continue
            
            result = self.ask(question, show_sources=show_sources)
            print(f"\n📝 Answer:\n{result['answer']}")
            
            if show_sources and "sources" in result:
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for s in result["sources"]:
                    print(f"  - {Path(s['source']).name} (Page {s['page']})")
                    print(f"    '{s['snippet']}'")


# ── Demo ─────────────────────────────────────────────────
qa = PDFQASystem()

# Option 1: Use a PDF file
# qa.ingest("path/to/your/document.pdf")

# Option 2: Use a Wikipedia article (good for testing)
print("Loading Wikipedia article for demo (no PDF needed)...")
qa.ingest(
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    source_type="url"
)

# Test questions
questions = [
    "What is retrieval-augmented generation?",
    "What are the key components of RAG?",
    "What are the limitations of RAG?",
    "How does RAG compare to fine-tuning?",
]

print("\n" + "=" * 50)
print("TEST QUESTIONS")
print("=" * 50)

for q in questions:
    result = qa.ask(q)
    print(f"\n❓ {result['question']}")
    print(f"📝 {result['answer'][:400]}")
    if "sources" in result and result["sources"]:
        print(f"📚 Sources: {len(result['sources'])} document(s) retrieved")
    print("-" * 40)

print("\n✅ Day 16 Lab Complete!")

# To launch interactive mode:
# qa.interactive()
```

---

## 🧠 Quiz: Day 16

**Q1:** In RAG, which step happens only ONCE (offline), not at query time?
- A) Query embedding
- B) Similarity search
- C) **Document ingestion, chunking, and indexing ✅**
- D) LLM generation

**Q2:** What is the main benefit of RAG over standard LLM prompting?
- A) Faster inference
- B) Lower API costs
- C) **Grounds answers in specific documents, reducing hallucination ✅**
- D) Longer context support

**Q3:** What does `chunk_overlap=100` accomplish in text splitting?
- A) Creates 100 identical chunks for comparison
- B) **Ensures 100 characters of text appear in both adjacent chunks ✅**
- C) Limits each chunk to 100 characters
- D) Splits on every 100th token

**Q4:** Why do we rephrase the user's follow-up question in conversational RAG?
- A) To make the question longer for better retrieval
- B) **To make it a standalone question that retrieval can understand without conversation history ✅**
- C) To translate the question to another language
- D) To reduce query complexity

**Q5:** The instruction "Only use information from the provided context" in a RAG system prompt:
- A) Prevents the model from generating text
- B) Slows down inference for safety
- C) **Prevents the model from mixing retrieved facts with its own knowledge ✅**
- D) Limits response length

**Q6:** When should you NOT use RAG?
- A) When documents are private
- B) When you have real-time data
- C) **When documents fit entirely in the LLM's context window ✅**
- D) When documents are in PDF format

---

## 📊 Key Takeaways

| Concept | Key Point |
|---------|-----------|
| **RAG Pipeline** | Ingest offline → retrieve at query time → generate with context |
| **Chunking** | Split large docs into chunks that fit in LLM context (500–1000 tokens typical) |
| **Retrieval** | Find top-K chunks most semantically similar to the query |
| **Grounding** | Instruct LLM to ONLY use retrieved context → reduces hallucination |
| **Source Citation** | RAG enables citing specific document sources |
| **Conversational RAG** | Rephrase follow-up questions as standalone before retrieval |
| **Metadata** | Always store source info in chunk metadata for citations |

---

## 📖 Further Reading

- [RAG Paper (Lewis et al. 2020)](https://arxiv.org/abs/2005.11401) — Original RAG research
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [RAGAS - RAG Evaluation Framework](https://docs.ragas.io/)
- [Chunking Strategies Deep Dive](https://www.pinecone.io/learn/chunking-strategies/)

---

## 🔄 What's Next: Day 17 Preview

Tomorrow: **Advanced RAG Patterns** — taking your RAG system from good to production-grade:
- HyDE (Hypothetical Document Embeddings)
- Multi-Query Retrieval
- Contextual compression
- Reranking with cross-encoders
- RAGAS evaluation framework

---

*Day 16 Complete ✅ | GenAI Course — Week 3 | Next: Day 17 — Advanced RAG Patterns*

---

##  Section 6: Production RAG Architecture

### 6.1 The Complete RAG Pipeline in Detail

```
User Query
    
    

                  INGESTION PIPELINE                  
  Source  Loader  Splitter  Embedder  VectorDB   
─
                              
                               (run once / update periodically)
                              
User Query 
                   
                       RETRIEVAL PIPELINE        
                     Query  Embed  Top-K Docs  
                   
                                  
                                  
─
                 GENERATION PIPELINE                  
  [System Prompt] + [Context Docs] + [User Query]    
                                                    
                  LLM (GPT-4o)                       
                                                    
              [Grounded Answer]                      

```

### 6.2 Multi-Document RAG with Source Tracking

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# Simulate loading multiple documents with tracking
documents_data = [
    {
        "source": "OpenAI_API_Docs.pdf",
        "content": """
GPT-4o is our flagship multimodal model. It accepts text, images, and audio as input.
The API uses the chat completions endpoint: POST /v1/chat/completions.
Model names: gpt-4o, gpt-4o-mini. Pricing: $5/1M input tokens for gpt-4o.
Rate limits vary by tier: Tier 1 is 500 RPM, Tier 5 is 10,000 RPM.
Function calling allows models to call external tools and APIs.
""",
        "category": "documentation", "version": "2024.11"
    },
    {
        "source": "Internal_Guidelines.md",
        "content": """
Company AI Policy: All LLM API calls must be logged for compliance.
Maximum context window usage: Do not exceed 80% of model context.
Data classification: Never send PII or confidential data to external LLMs.
Approved models: gpt-4o-mini for standard tasks, gpt-4o for complex analysis.
Budget limits: Each team has a monthly token budget. Alert at 80% usage.
""",
        "category": "policy", "version": "v3.2"
    },
    {
        "source": "Architecture_Notes.txt",
        "content": """
Our RAG system uses ChromaDB for vector storage.
Embedding model: text-embedding-3-small for cost efficiency.
Chunk size: 500 tokens with 50 token overlap.
Retrieval: top-5 documents with MMR diversity.
Re-ranking: Cohere re-ranker for final document selection.
""",
        "category": "architecture", "version": "2024.Q4"
    },
]

# Create documents with rich metadata
docs = []
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
for data in documents_data:
    chunks = splitter.split_text(data["content"])
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "source": data["source"],
                "category": data["category"],
                "version": data["version"],
                "chunk_id": i
            }
        ))

# Build vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embeddings, collection_name="company_kb")

# RAG chain with source citations
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs_with_sources(docs: list[Document]) -> tuple[str, list[dict]]:
    """Format docs for context and extract source metadata"""
    context_parts = []
    sources = []
    
    for i, doc in enumerate(docs):
        context_parts.append(f"[Doc {i+1}] (Source: {doc.metadata['source']})\n{doc.page_content}")
        sources.append({
            "source": doc.metadata["source"],
            "category": doc.metadata["category"],
            "version": doc.metadata["version"]
        })
    
    return "\n\n---\n\n".join(context_parts), sources

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer questions using ONLY the provided context.
Always cite which source document(s) your answer comes from (e.g., "According to [source]...").
If the context doesn't contain the answer, say so clearly."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def rag_with_citations(question: str) -> dict:
    """Complete RAG pipeline with source citations"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)
    
    context, sources = format_docs_with_sources(retrieved_docs)
    
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    
    # Deduplicate sources
    seen = set()
    unique_sources = []
    for s in sources:
        key = s["source"]
        if key not in seen:
            unique_sources.append(s)
            seen.add(key)
    
    return {
        "question": question,
        "answer": answer,
        "sources": unique_sources,
        "doc_count": len(retrieved_docs)
    }

# Test queries
questions = [
    "What are the rate limits for GPT-4o?",
    "What is our company policy on using LLMs with sensitive data?",
    "What embedding model do we use and why?"
]

for q in questions:
    result = rag_with_citations(q)
    print(f"\n {result['question']}")
    print(f" {result['answer']}")
    print(f" Sources: {[s['source'] for s in result['sources']]}")
    print(f"   ({result['doc_count']} documents retrieved)")
```

### 6.3 Asynchronous RAG for Production

```python
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Async RAG handles multiple users simultaneously
async def async_rag_query(vectorstore, llm, question: str, session_id: str) -> dict:
    """Process a single RAG query asynchronously"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Async retrieval
    docs = await retriever.ainvoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on context: {context}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Async LLM call
    answer = await chain.ainvoke({"context": context, "question": question})
    
    return {"session_id": session_id, "answer": answer, "doc_count": len(docs)}

async def handle_concurrent_users(vectorstore, queries: list[tuple[str, str]]):
    """Handle multiple simultaneous RAG queries"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # All queries run concurrently
    tasks = [
        async_rag_query(vectorstore, llm, question, session_id)
        for question, session_id in queries
    ]
    
    results = await asyncio.gather(*tasks)
    return results

# Usage example (in an async context):
# queries = [
#     ("What is RAG?", "user_001"),
#     ("How does chunking work?", "user_002"),
#     ("What are embeddings?", "user_003"),
# ]
# results = asyncio.run(handle_concurrent_users(vectorstore, queries))
```

---

##  Section 7: Evaluation & Quality Control

### 7.1 RAG Quality Metrics

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class RAGEvalResult(BaseModel):
    faithfulness_score: float   # 0-1: Is answer grounded in context?
    relevance_score: float      # 0-1: Is answer relevant to question?
    completeness_score: float   # 0-1: Does answer fully address question?
    reasoning: str

def evaluate_rag_response(question: str, context: str, answer: str) -> RAGEvalResult:
    """LLM-as-judge evaluation for RAG responses"""
    
    eval_prompt = ChatPromptTemplate.from_template("""
Evaluate this RAG response on three criteria. Score each from 0.0 to 1.0.

Question: {question}
Context: {context}
Answer: {answer}

Evaluate:
1. Faithfulness (0-1): Is the answer fully supported by the context? Penalize hallucinations.
2. Relevance (0-1): Does the answer directly address the question?
3. Completeness (0-1): How complete and thorough is the answer given the available context?

Return JSON:
{{"faithfulness_score": 0.0-1.0, "relevance_score": 0.0-1.0, "completeness_score": 0.0-1.0, "reasoning": "brief explanation"}}
""")
    
    chain = eval_prompt | llm | JsonOutputParser()
    result = chain.invoke({"question": question, "context": context, "answer": answer})
    return RAGEvalResult(**result)

# Test evaluation
question = "What is the rate limit for GPT-4 API?"
context = "GPT-4 API has different rate limits per tier. Tier 1: 500 RPM. Tier 5: 10,000 RPM. Limits apply to all endpoints."
answer = "The GPT-4 API rate limits start at 500 requests per minute for Tier 1 users."

eval_result = evaluate_rag_response(question, context, answer)
print(f"Faithfulness:  {eval_result.faithfulness_score:.2f}")
print(f"Relevance:     {eval_result.relevance_score:.2f}")
print(f"Completeness:  {eval_result.completeness_score:.2f}")
print(f"Reasoning:     {eval_result.reasoning}")
avg = (eval_result.faithfulness_score + eval_result.relevance_score + eval_result.completeness_score) / 3
print(f"Overall Score: {avg:.2f}")
```

---

##  Extended Lab: Production RAG System

```python
# The complete production RAG app built in this day's lab
# combines all components: ingestion, chunking, storage, retrieval,
# generation, citation, and evaluation.

# Architecture summary:
#  Ingestion 
#   PDF/Web  Splitter  Embedder  ChromaDB        
# 
#  Query Pipeline 
#   Q  Embed  MMR Retrieve  Re-rank  Generate  
#    Citation Extraction  Quality Eval           
# 
#  Streamlit UI 
#   Upload  Chat  Sources  Eval Display          
# 

print(" Day 16 Extended Lab complete!")
print("See day-16-lab.py for the full Streamlit implementation.")
```

---


---

## Section 8: LlamaIndex — The Premier Data Framework (2025 Update)

While LangChain is excellent for building "Agents" and general chains, the industry consensus in 2025 is that **LlamaIndex** is the absolute superior framework specifically for Data Parsing, Ingestion, and Retrieval-Augmented Generation (RAG).

### 8.1 Why LlamaIndex for RAG?
- **Data Loaders:** LlamaHub has over 160+ native data connectors (Notion, Slack, PDF, Google Drive).
- **Node Parsing:** It intelligently splits data not just by characters, but by semantic hierarchy (e.g., Markdown Headers, JSON objects).
- **Index Structures:** It supports complex data structures (Tree Index, Knowledge Graph Index, Summary Index), not just flat Vector Stores.

### 8.2 Building RAG in 5 Lines with LlamaIndex

If you have a folder of PDFs, building a complete RAG system with LlamaIndex requires remarkably little code because of its powerful defaults:

```python
# pip install llama-index
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. Load Data (Reads every PDF/TXT in the 'data' folder)
documents = SimpleDirectoryReader("data").load_data()

# 2. Build Index (Automatically chunks, embeds via OpenAI, and stores in an in-memory vector store)
index = VectorStoreIndex.from_documents(documents)

# 3. Create a Query Engine (Automatically handles retrieval and LLM context window injection)
query_engine = index.as_query_engine()

# 4. Query
response = query_engine.query("What are the main requirements of the Capstone project?")
print(response)
```

### 8.3 Customizing the Storage Context
For production, you don't want to use an ephemeral in-memory dictionary. You want to persist to disk using ChromaDB or another vector store:

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# Initialize Chroma connection
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("llama_index_qabot")

# Assign Chroma as the vector store for LlamaIndex
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index and it will save to Chroma automatically
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```

LlamaIndex abstracts away the hardest parts of RAG (chunk overlap management, dynamic embedding injection, and prompt construction) so you can focus on data quality.

*Day 16 Updated: 2025 Data Architectures via LlamaIndex complete.*

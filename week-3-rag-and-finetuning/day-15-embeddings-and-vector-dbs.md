# Day 15: Embeddings & Vector Databases 🔢
### Week 3 — RAG, Fine-Tuning & Vector Databases

---

## 🎯 Learning Objectives

By the end of today, you will:
- Understand what embeddings are and how they represent meaning
- Compare embedding models (OpenAI, Sentence Transformers, etc.)
- Work with ChromaDB, FAISS, and Pinecone
- Build a semantic search engine from scratch
- Understand similarity metrics: cosine, dot product, Euclidean
- Implement a complete document indexing pipeline

**Estimated Time:** 3.5–4 hours  
**Difficulty:** ⭐⭐⭐ Intermediate  
**Prerequisites:** Days 11–14 (LangChain familiarity)

---

## 📚 Section 1: What Are Embeddings?

### 1.1 From Words to Numbers

Computers work with numbers. Text is symbolic. **Embeddings** are the bridge — they convert text (or images, audio, etc.) into numeric vectors that capture *meaning*.

The magical property of good embeddings: **semantic similarity equals geometric proximity**.

```
"dog" → [0.2, -0.5, 0.8, ...]        (384 or 1536 numbers)
"cat" → [0.1, -0.4, 0.7, ...]        (close to dog — both pets)
"automobile" → [-0.3, 0.6, -0.2, ...] 
"car" → [-0.4, 0.5, -0.1, ...]       (close to automobile — synonyms)
```

Distance between "dog" and "cat" vectors < Distance between "dog" and "car" vectors.

### 1.2 The Semantic Space

Imagine a high-dimensional space (think 384D or 1536D) where every piece of text is a point. Points close together have similar meaning.

```
High-Dimensional Semantic Space (visualized in 2D for clarity):

Medical terms cluster
    ○ diagnosis
    ○ treatment           Sports cluster
    ○ prescription            ○ soccer
                              ○ basketball    Tech cluster
Programming cluster           ○ referee           ○ Python
    ○ function                                    ○ algorithm
    ○ variable                                    ○ recursion
    ○ class
```

This is why semantic search works — instead of matching keywords, you find content that MEANS the same thing.

### 1.3 Use Cases for Embeddings

| Use Case | Example |
|----------|---------|
| **Semantic Search** | Find documents by meaning, not keywords |
| **RAG** | Retrieve relevant context for LLM generation |
| **Recommendation** | "Users who liked X also liked Y" |
| **Clustering** | Group similar documents automatically |
| **Anomaly Detection** | Find outlier text that doesn't fit a cluster |
| **Classification** | Classify text using nearest-neighbor methods |
| **Deduplication** | Find near-duplicate content |

---

## 📚 Section 2: Embedding Models

### 2.1 Major Embedding Models

| Model | Dimensions | Best For | Cost |
|-------|-----------|----------|------|
| `text-embedding-3-small` (OpenAI) | 1536 | General, quality+cost balance | Paid |
| `text-embedding-3-large` (OpenAI) | 3072 | Maximum quality | Paid |
| `text-embedding-ada-002` (OpenAI) | 1536 | Legacy default | Paid |
| `all-MiniLM-L6-v2` (ST) | 384 | Fast, free, good quality | Free |
| `all-mpnet-base-v2` (ST) | 768 | Better quality, still free | Free |
| `nomic-embed-text` | 768 | Long documents, open source | Free |
| `BAAI/bge-large-en-v1.5` | 1024 | Top open-source performance | Free |

```bash
pip install sentence-transformers openai chromadb faiss-cpu langchain-openai
```

### 2.2 Creating Embeddings

```python
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── OpenAI Embeddings ──────────────────────────────────────
def get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding from OpenAI API"""
    text = text.replace("\n", " ")  # Clean text
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Single embedding
embedding = get_openai_embedding("Generative AI is transforming software development.")
print(f"OpenAI embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# Batch embeddings (more efficient — one API call for many texts)
def get_batch_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Get embeddings for multiple texts in one API call"""
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

texts = [
    "The transformer architecture revolutionized NLP.",
    "Self-attention mechanisms allow models to focus on relevant tokens.",
    "BERT is a bidirectional transformer encoder.",
    "Python is a high-level programming language.",
    "Cats are domestic animals related to lions."
]

embeddings = get_batch_embeddings(texts)
print(f"\nBatch size: {len(embeddings)}, Each dim: {len(embeddings[0])}")
```

### 2.3 Sentence Transformers (Free Alternative)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a free, high-quality model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Downloads on first use

sentences = [
    "A programmer is debugging their code.",
    "A software engineer is fixing a bug in their program.",
    "The weather today is sunny and warm.",
    "Beautiful day with clear skies and high temperatures.",
    "Dogs are loyal companions to humans.",
    "Cats are independent animals.",
]

# Generate embeddings
embeddings = model.encode(sentences, convert_to_numpy=True)
print(f"Shape: {embeddings.shape}")  # (6, 384)

# Compute all pairwise similarities
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)

print("\nSimilarity Matrix:")
print(f"{'':30}", end="")
for s in sentences:
    print(f"{s[:15]:>18}", end="")
print()

for i, s1 in enumerate(sentences):
    print(f"{s1[:30]:30}", end="")
    for j in range(len(sentences)):
        print(f"{similarity_matrix[i][j]:>18.3f}", end="")
    print()
```

---

## 📚 Section 3: Similarity Metrics

### 3.1 Cosine Similarity

Measures the angle between two vectors — the most common choice for text:

```python
import numpy as np

def cosine_similarity_manual(a: list, b: list) -> float:
    """Cosine similarity from scratch"""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def euclidean_distance(a: list, b: list) -> float:
    """Euclidean (L2) distance"""
    a, b = np.array(a), np.array(b)
    return float(np.linalg.norm(a - b))

def dot_product(a: list, b: list) -> float:
    """Dot product — works well for normalized vectors"""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b))

# Demonstration with sentence transformer embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

pairs = [
    ("I love machine learning", "AI and deep learning are fascinating"),     # High sim
    ("I love machine learning", "The best pizza has extra cheese"),           # Low sim
    ("Python is great", "Python is a wonderful programming language"),        # Very high sim
]

print("Similarity Comparison:")
print(f"{'Text 1':40} {'Text 2':40} {'Cosine':8} {'Euclidean':10}")
print("-" * 100)

for t1, t2 in pairs:
    e1 = model.encode(t1)
    e2 = model.encode(t2)
    cos = cosine_similarity_manual(e1, e2)
    euc = euclidean_distance(e1, e2)
    print(f"{t1[:38]:40} {t2[:38]:40} {cos:.4f}   {euc:.4f}")
```

### 3.2 Which Metric to Use?

| Metric | Range | Use When |
|--------|-------|----------|
| **Cosine Similarity** | -1 to 1 | Text embeddings (default choice) |
| **Euclidean Distance** | 0 to ∞ | When vector magnitude matters |
| **Dot Product** | -∞ to ∞ | When vectors are normalized (faster) |

Most embedding models optimize for cosine similarity — use it as your default.

---

## 📚 Section 4: Vector Databases

### 4.1 Why Not Just Use NumPy?

With 100 documents, brute-force cosine search over NumPy arrays works fine.
With 10 million document chunks... you need Approximate Nearest Neighbor (ANN) search.

Vector databases provide:
- **Indexing** for fast ANN search (HNSW, IVF, etc.)
- **Metadata filtering** (find similar docs that also match filters)
- **Persistence** (don't re-embed everything on restart)
- **Scalability** (billions of vectors with horizontal scaling)

### 4.2 ChromaDB — Local Development

```python
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Initialize ChromaDB (local, persistent)
client = chromadb.PersistentClient(path="./chroma_db")

# Use an embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Or use a free local model
sentence_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get a collection
collection = client.get_or_create_collection(
    name="course_materials",
    embedding_function=sentence_ef,
    metadata={"hnsw:space": "cosine"}
)

# Add documents
documents = [
    "Transformers use self-attention to process sequences in parallel.",
    "BERT is pre-trained using masked language modeling.",
    "GPT models are autoregressive: they predict the next token.",
    "LoRA is a parameter-efficient fine-tuning method.",
    "RAG combines retrieval with generation for factual accuracy.",
    "Vector databases index embeddings for fast similarity search.",
    "Attention mechanisms weigh the importance of different tokens.",
    "Fine-tuning adapts a pretrained model to a specific task.",
]

metadatas = [
    {"topic": "transformers", "week": 1},
    {"topic": "bert", "week": 1},
    {"topic": "gpt", "week": 2},
    {"topic": "fine-tuning", "week": 3},
    {"topic": "rag", "week": 3},
    {"topic": "vector-db", "week": 3},
    {"topic": "attention", "week": 1},
    {"topic": "fine-tuning", "week": 3},
]

ids = [f"doc_{i}" for i in range(len(documents))]

collection.upsert(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Collection size: {collection.count()} documents")

# Semantic search
query = "How does the attention mechanism work?"
results = collection.query(
    query_texts=[query],
    n_results=3
)

print(f"\nQuery: '{query}'")
print("Top 3 results:")
for doc, meta, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    similarity = 1 - distance  # Convert distance to similarity
    print(f"  ({similarity:.3f}) [{meta['topic']}] {doc[:80]}")

# Filtered search — only Week 3 content
filtered_results = collection.query(
    query_texts=["model customization"],
    n_results=3,
    where={"week": {"$eq": 3}}  # Metadata filter!
)

print("\nFiltered query (Week 3 only):")
for doc, meta in zip(filtered_results["documents"][0], filtered_results["metadatas"][0]):
    print(f"  [{meta['topic']}] {doc[:80]}")
```

### 4.3 FAISS — High Performance

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare documents
documents = [
    "The Eiffel Tower is located in Paris, France.",
    "Python was created by Guido van Rossum in 1991.",
    "Machine learning is a subset of artificial intelligence.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Transformers were introduced in the 'Attention is All You Need' paper.",
    "Deep learning uses neural networks with many layers.",
    "OpenAI developed the GPT series of language models.",
    "Paris is the capital of France and a major cultural center.",
]

# Create embeddings
embeddings = model.encode(documents, convert_to_numpy=True)
embeddings = embeddings.astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# Build FAISS index
# IndexFlatIP = Flat index with Inner Product (dot product for normalized = cosine)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"FAISS index: {index.ntotal} vectors, dimension {dimension}")

# Search
def search_faiss(query: str, k: int = 3) -> list:
    """Search the FAISS index"""
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    
    scores, indices = index.search(query_embedding, k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "document": documents[idx],
            "score": float(score),
            "index": int(idx)
        })
    return results

# Test searches
queries = [
    "What is the capital of France?",
    "Tell me about large language models",
    "History of Python programming"
]

for q in queries:
    print(f"\n🔍 '{q}'")
    results = search_faiss(q, k=3)
    for r in results:
        print(f"  ({r['score']:.3f}) {r['document'][:80]}")

# Save and load FAISS index
faiss.write_index(index, "my_faiss.index")
print("\n💾 FAISS index saved")

loaded_index = faiss.read_index("my_faiss.index")
print(f"✅ FAISS index loaded: {loaded_index.ntotal} vectors")
```

### 4.4 LangChain Vector Store Integration

```python
from langchain_community.vectorstores import Chroma, FAISS as LangFAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Use free HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create documents with metadata
docs = [
    Document(page_content="Transformers use self-attention mechanisms.", 
             metadata={"source": "day04", "topic": "transformers"}),
    Document(page_content="RAG combines retrieval with generation.", 
             metadata={"source": "day16", "topic": "rag"}),
    Document(page_content="Fine-tuning adapts pretrained models.", 
             metadata={"source": "day18", "topic": "fine-tuning"}),
    Document(page_content="Agents can use tools to take actions.", 
             metadata={"source": "day22", "topic": "agents"}),
]

# Create vector store from documents
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./langchain_chroma"
)

# As a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",          # or "mmr" for diversity
    search_kwargs={"k": 2}
)

# Retrieve relevant docs
relevant = retriever.invoke("How do language models learn from tasks?")
for doc in relevant:
    print(f"[{doc.metadata['topic']}] {doc.page_content}")
```

---

## 💻 Full Lab: Semantic Search Engine

```python
# lab_day15_semantic_search.py
"""Day 15 Lab — Build a complete semantic search engine"""

import os, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# ── Knowledge Base ─────────────────────────────────────────
KNOWLEDGE_BASE = [
    {"text": "Transformers process sequences using self-attention, which computes relationships between all tokens simultaneously.", "source": "Deep Learning Textbook", "topic": "transformers"},
    {"text": "BERT uses bidirectional training, reading text both left-to-right and right-to-left to understand context.", "source": "BERT Paper", "topic": "bert"},
    {"text": "GPT models generate text autoregressively by predicting the next token given all previous tokens.", "source": "GPT-3 Paper", "topic": "gpt"},
    {"text": "RAG (Retrieval-Augmented Generation) retrieves relevant documents and uses them as context for generation.", "source": "RAG Paper", "topic": "rag"},
    {"text": "Vector embeddings map text to dense numerical representations where semantic similarity equals geometric proximity.", "source": "Word2Vec Paper", "topic": "embeddings"},
    {"text": "LoRA (Low-Rank Adaptation) fine-tunes LLMs by adding small, trainable matrices to existing weight matrices.", "source": "LoRA Paper", "topic": "fine-tuning"},
    {"text": "Chain-of-Thought prompting encourages LLMs to show their reasoning steps before giving a final answer.", "source": "CoT Paper", "topic": "prompting"},
    {"text": "AI agents use LLMs as a reasoning engine to decide which tools to use and in what order.", "source": "ReAct Paper", "topic": "agents"},
    {"text": "Hallucination in LLMs occurs when models generate confident but factually incorrect information.", "source": "AI Safety Research", "topic": "safety"},
    {"text": "The context window limits how much text an LLM can process in a single forward pass.", "source": "LLM Architecture", "topic": "architecture"},
    {"text": "Quantization reduces model size by representing weights in lower precision (e.g., 4-bit instead of 16-bit).", "source": "Quantization Guide", "topic": "optimization"},
    {"text": "Pinecone is a managed vector database that allows storing and querying billions of embeddings.", "source": "Pinecone Docs", "topic": "vector-db"},
    {"text": "LangChain provides abstractions for building LLM applications: chains, agents, memory, and tools.", "source": "LangChain Docs", "topic": "framework"},
    {"text": "Cosine similarity measures the angle between two vectors, making it scale-invariant for text search.", "source": "Information Retrieval", "topic": "embeddings"},
    {"text": "RLHF (Reinforcement Learning from Human Feedback) aligns LLMs with human preferences.", "source": "InstructGPT Paper", "topic": "alignment"},
]

class SemanticSearchEngine:
    """A production-ready semantic search engine"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_dir: str = "./search_index"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.persist_dir = persist_dir
        self.vectorstore = None
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if Path(self.persist_dir).exists():
            print("📂 Loading existing index...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            print(f"✅ Loaded {self.vectorstore._collection.count()} documents")
        else:
            print("🔨 Building new index...")
            self._build_index()
    
    def _build_index(self):
        """Build the vector index from the knowledge base"""
        documents = [
            Document(
                page_content=item["text"],
                metadata={"source": item["source"], "topic": item["topic"]}
            )
            for item in KNOWLEDGE_BASE
        ]
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print(f"✅ Indexed {len(documents)} documents")
    
    def search(self, query: str, k: int = 5, topic_filter: str = None) -> list[dict]:
        """Semantic search with optional topic filtering"""
        search_kwargs = {"k": k}
        
        if topic_filter:
            search_kwargs["filter"] = {"topic": topic_filter}
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={**search_kwargs, "score_threshold": 0.3}
        )
        
        docs = retriever.invoke(query)
        
        # Also get scores
        docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, **({"filter": {"topic": topic_filter}} if topic_filter else {})
        )
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source"),
                "topic": doc.metadata.get("topic"),
                "relevance_score": round(score, 4)
            })
        
        return results
    
    def add_documents(self, texts: list[str], metadatas: list[dict] = None) -> None:
        """Add new documents to the index"""
        documents = [
            Document(
                page_content=text,
                metadata=meta or {}
            )
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        self.vectorstore.add_documents(documents)
        print(f"➕ Added {len(documents)} documents to index")
    
    def explain_similarity(self, text1: str, text2: str) -> dict:
        """Explain why two texts are similar or different"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        e1 = model.encode(text1)
        e2 = model.encode(text2)
        
        cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
        
        return {
            "text1": text1,
            "text2": text2,
            "cosine_similarity": round(cos_sim, 4),
            "interpretation": (
                "Very similar (>0.8)" if cos_sim > 0.8 else
                "Related (0.6-0.8)" if cos_sim > 0.6 else
                "Somewhat related (0.4-0.6)" if cos_sim > 0.4 else
                "Different (<0.4)"
            )
        }
    
    def compare_queries(self, queries: list[str]) -> None:
        """Compare multiple queries to see what topics they retrieve"""
        print("\n📊 QUERY COMPARISON")
        print("=" * 60)
        
        for q in queries:
            results = self.search(q, k=3)
            print(f"\n🔍 '{q}'")
            for r in results:
                print(f"  ({r['relevance_score']:.3f}) [{r['topic']:15}] {r['text'][:60]}...")


# ── Demo ─────────────────────────────────────────────────
engine = SemanticSearchEngine()

# Run sample searches
test_queries = [
    "How do language models generate text?",
    "What makes retrieval augmented generation effective?",
    "How to reduce the size of large AI models?",
    "What are safety concerns with AI systems?",
]

engine.compare_queries(test_queries)

# Add custom documents
custom_docs = [
    "Multimodal AI models can process both text and images simultaneously.",
    "The Mixtral model uses Mixture of Experts to activate only a portion of its parameters per token.",
]
engine.add_documents(
    texts=custom_docs,
    metadatas=[{"source": "Custom", "topic": "multimodal"}, {"source": "Custom", "topic": "architecture"}]
)

# Similarity explanation
print("\n" + "=" * 60)
print("SIMILARITY EXPLANATIONS")
print("=" * 60)
pairs = [
    ("Find information in documents", "Retrieve relevant context from knowledge base"),
    ("Training neural networks", "The weather today is nice"),
]
for t1, t2 in pairs:
    result = engine.explain_similarity(t1, t2)
    print(f"\n  '{t1[:40]}' vs '{t2[:40]}'")
    print(f"  Similarity: {result['cosine_similarity']} — {result['interpretation']}")

print("\n✅ Day 15 Lab Complete!")
```

---

## 🧠 Quiz: Day 15

**Q1:** What property makes embeddings useful for semantic search?
- A) They compress text to save storage
- B) **Semantically similar texts have geometrically close vectors ✅**
- C) They translate text to other languages
- D) They encrypt sensitive data

**Q2:** Why is cosine similarity preferred over Euclidean distance for text embeddings?
- A) It's faster to compute
- B) **It's scale-invariant — measures direction, not magnitude ✅**
- C) It always returns values between 0 and 1
- D) It works without normalization

**Q3:** When would you choose a vector database over a simple NumPy array search?
- A) When you have fewer than 1,000 documents
- B) When you need exact matches only
- C) **When you need fast approximate search over millions of vectors ✅**
- D) When you want free solutions only

**Q4:** What does `chunk_overlap` in document chunking accomplish?
- A) Detects duplicate content
- B) Ensures chunks are the same size
- C) **Preserves context that might be cut off at chunk boundaries ✅**
- D) Improves embedding quality

**Q5:** ChromaDB's metadata filtering allows you to:
- A) Filter based on semantic similarity only
- B) **Combine vector similarity search with attribute-based filters ✅**
- C) Reduce embedding dimensions
- D) Convert text to SQL queries

**Q6:** What is HNSW in FAISS vector indexing?
- A) A compression algorithm for vectors
- B) A training algorithm for embeddings
- C) **Hierarchical Navigable Small World — an efficient ANN graph structure ✅**
- D) A distance metric for sparse vectors

**Q7:** `text-embedding-3-small` vs `all-MiniLM-L6-v2` — main trade-off is:
- A) Accuracy vs speed
- B) **Quality + cost (OpenAI paid) vs free + slightly lower quality (open-source) ✅**
- C) GPU vs CPU requirement
- D) English-only vs multilingual

---

## 📊 Key Takeaways

| Concept | Key Point |
|---------|-----------|
| **Embeddings** | Dense numerical vector representations of semantic meaning |
| **Cosine Similarity** | Angle between vectors — preferred metric for text |
| **ChromaDB** | Great for local dev/production, metadata filtering |
| **FAISS** | Ultra-fast, used by Meta — best for very large datasets |
| **Sentence Transformers** | Free embedding models, surprisingly good quality |
| **Chunk Overlap** | Provides context continuity across document boundaries |
| **Metadata Filtering** | Combine semantic search with attribute filters |
| **ANN Search** | Approximate Nearest Neighbor — makes million-scale search feasible |

---

## 📖 Further Reading

- [Understanding LLM Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Sentence Transformers Documentation](https://sbert.net/)
- [FAISS Documentation](https://faiss.ai/index.html)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Vector Database Comparison](https://superlinked.com/vectorhub/vector-db-comparison)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — Embedding model rankings

---

## 🔄 What's Next: Day 16 Preview

Tomorrow: **RAG Fundamentals** — we combine everything from today with LLMs to build a system that can answer questions about YOUR documents:
- RAG architecture: ingestion pipeline → retrieval → augmented generation
- Document loading, chunking, embedding, and storing
- Retrieval strategies and why they matter
- Building a complete PDF Q&A system

---

*Day 15 Complete ✅ | GenAI Course — Week 3 | Next: Day 16 — RAG Fundamentals*

---

##  Section 6: FAISS In Depth

### 6.1 FAISS Index Types Compared

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "The transformer architecture introduced self-attention mechanisms.",
    "BERT is a bidirectional encoder representation from transformers.",
    "GPT uses a decoder-only transformer for text generation.",
    "RAG combines retrieval with generation for grounded responses.",
    "Vector databases enable semantic similarity search at scale.",
    "Embeddings convert text into dense numerical vectors.",
    "ChromaDB is a lightweight vector store for prototyping.",
    "Pinecone is a managed vector database for production.",
    "LangChain provides abstractions for building LLM applications.",
    "Fine-tuning adapts a pre-trained model to a specific task.",
]

vectors = encoder.encode(texts)
dimension = vectors.shape[1]

# --- Index Flat (exact search, no approximation) ---
flat_index = faiss.IndexFlatL2(dimension)
flat_index.add(vectors.astype("float32"))

# --- Index IVF (inverted file, faster for large datasets) ---
n_clusters = 2  # In production: sqrt(n_vectors)
quantizer = faiss.IndexFlatL2(dimension)
ivf_index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_L2)
ivf_index.train(vectors.astype("float32"))
ivf_index.add(vectors.astype("float32"))
ivf_index.nprobe = 2  # How many clusters to search (accuracy vs speed tradeoff)

# --- Index HNSW (Hierarchical Navigable Small World, high accuracy + speed) ---
hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = connections per node
hnsw_index.add(vectors.astype("float32"))

# --- Comparison Query ---
query = "How does attention work in neural networks?"
q_vec = encoder.encode([query])[0].reshape(1, -1).astype("float32")

for name, idx in [("Flat L2", flat_index), ("IVF", ivf_index), ("HNSW", hnsw_index)]:
    distances, indices = idx.search(q_vec, 3)
    print(f"\n[{name}] Top results for: '{query}'")
    for dist, i in zip(distances[0], indices[0]):
        if i >= 0:
            print(f"  ({dist:.4f}) {texts[i]}")
```

### 6.2 Filtered Vector Search

Combine semantic search with metadata filtering:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()

# Documents with rich metadata
docs = [
    Document(page_content="Introduction to Python programming basics", metadata={"level": "beginner", "topic": "python", "year": 2024}),
    Document(page_content="Advanced Python decorators and metaclasses", metadata={"level": "advanced", "topic": "python", "year": 2024}),
    Document(page_content="Machine learning fundamentals with scikit-learn", metadata={"level": "intermediate", "topic": "ml", "year": 2023}),
    Document(page_content="Deep learning with PyTorch from scratch", metadata={"level": "intermediate", "topic": "deep_learning", "year": 2024}),
    Document(page_content="LLM fine-tuning with LoRA on custom datasets", metadata={"level": "advanced", "topic": "llm", "year": 2024}),
    Document(page_content="Building chatbots with LangChain", metadata={"level": "intermediate", "topic": "llm", "year": 2024}),
    Document(page_content="Vector databases and semantic search basics", metadata={"level": "intermediate", "topic": "embeddings", "year": 2023}),
    Document(page_content="RAG systems for production deployment", metadata={"level": "advanced", "topic": "rag", "year": 2024}),
]

vectorstore = Chroma.from_documents(docs, embeddings, collection_name="courses")

# Search across all documents
print("All results for 'LLM building':")
results = vectorstore.similarity_search("building with LLMs", k=3)
for r in results:
    print(f"  [{r.metadata['level']} | {r.metadata['topic']}] {r.page_content}")

# Filter to only 'advanced' level documents
print("\nAdvanced only:")
results_filtered = vectorstore.similarity_search(
    "building with LLMs",
    k=3,
    filter={"level": "advanced"}
)
for r in results_filtered:
    print(f"  [{r.metadata['level']} | {r.metadata['topic']}] {r.page_content}")

# Filter by topic
print("\nRAG/LLM topics only:")
results_topic = vectorstore.similarity_search(
    "production deployment",
    k=5,
    filter={"topic": {"$in": ["rag", "llm"]}}
)
for r in results_topic:
    print(f"  [{r.metadata['level']} | {r.metadata['topic']}] {r.page_content}")
```

### 6.3 Persistent ChromaDB with Multiple Collections

```python
import chromadb
from chromadb.config import Settings

# Persistent client  data survives program restarts
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Create separate collections for different knowledge domains
collections = {
    "technical": client.get_or_create_collection("technical_docs"),
    "hr": client.get_or_create_collection("hr_policies"),
    "finance": client.get_or_create_collection("financial_data"),
}

# Add documents to specific collections
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")

technical_docs = [
    "REST API authentication uses Bearer tokens in the Authorization header.",
    "Our CI/CD pipeline runs on GitHub Actions with 3 parallel test stages.",
    "Database migrations must be tested in staging before production deployment.",
]

technical_vecs = encoder.encode(technical_docs).tolist()
collections["technical"].add(
    documents=technical_docs,
    embeddings=technical_vecs,
    ids=[f"tech_{i}" for i in range(len(technical_docs))],
    metadatas=[{"domain": "technical", "date": "2024-01"} for _ in technical_docs]
)

# Query a specific collection
def search_knowledge_base(query: str, domain: str, top_k: int = 3) -> list[dict]:
    collection = collections.get(domain)
    if not collection:
        return []
    
    q_vec = encoder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_vec, n_results=top_k)
    
    return [
        {"doc": doc, "distance": dist, "id": id_}
        for doc, dist, id_ in zip(
            results["documents"][0],
            results["distances"][0],
            results["ids"][0]
        )
    ]

results = search_knowledge_base("how do we authenticate APIs?", "technical")
for r in results:
    print(f"[{r['distance']:.4f}] {r['doc']}")
```

---

##  Section 7: Embedding Models Compared

### 7.1 Benchmark: OpenAI vs Sentence-Transformers

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import time

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Model configs
models = {
    "text-embedding-3-small": OpenAIEmbeddings(model="text-embedding-3-small"),
    "text-embedding-3-large": OpenAIEmbeddings(model="text-embedding-3-large"),
    "all-MiniLM-L6-v2": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    "BAAI/bge-small-en-v1.5": HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
}

# Test sentence pairs
test_pairs = [
    ("The dog ran quickly.", "The cat sprinted fast."),       # Similar
    ("The dog ran quickly.", "Python is a programming language."),  # Dissimilar
    ("What is machine learning?", "Define ML for me."),       # Paraphrase
    ("Buy cheap meds online!", "Quarterly financial report."),  # Unrelated
]

print(f"{'Model':<35} {'Pair 1':>8} {'Pair 2':>8} {'Pair 3':>8} {'Pair 4':>8}")
print("-" * 75)

for model_name, embedder in models.items():
    scores = []
    for s1, s2 in test_pairs:
        e1, e2 = embedder.embed_query(s1), embedder.embed_query(s2)
        scores.append(cosine_similarity(e1, e2))
    
    row = f"{model_name:<35}" + "".join(f"{s:8.3f}" for s in scores)
    print(row)

# Dimensions comparison
print("\nEmbedding Dimensions:")
for model_name, embedder in models.items():
    dim = len(embedder.embed_query("test"))
    print(f"  {model_name}: {dim} dimensions")
```

### 7.2 Choosing the Right Embedding Model

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Production RAG** | `text-embedding-3-large` | Best quality, 3072d |
| **Cost-sensitive prod** | `text-embedding-3-small` | 5 cheaper, still great |
| **Free/local deploy** | `BAAI/bge-large-en-v1.5` | Top MTEB score, free |
| **Fastest local** | `all-MiniLM-L6-v2` | 384d, very fast, general purpose |
| **Multilingual** | `multilingual-e5-large` | 100+ languages |
| **Code search** | `text-embedding-ada-002` | Code-aware embeddings |

---

##  Extended Lab: Semantic Search Engine

```python
# extended_lab_day15.py
"""
Build a full semantic search engine over Wikipedia summaries
with filtering, ranking, and relevance scoring.
"""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from dotenv import load_dotenv

load_dotenv()

# Sample knowledge base
articles = [
    {"title": "Transformer", "category": "AI", "year": 2017,
     "content": "The Transformer is a deep learning model introduced in 2017 using self-attention mechanisms to process sequences in parallel, enabling training on much larger datasets than RNNs."},
    {"title": "BERT", "category": "AI", "year": 2018,
     "content": "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model using masked language modeling and next-sentence prediction objectives."},
    {"title": "GPT-4", "category": "AI", "year": 2023,
     "content": "GPT-4 is a large multimodal language model by OpenAI capable of accepting image and text inputs and producing text outputs with human-expert performance."},
    {"title": "Python", "category": "Programming", "year": 1991,
     "content": "Python is a high-level programming language emphasizing code readability. Used widely in data science, web development, and automation. Created by Guido van Rossum."},
    {"title": "NumPy", "category": "Library", "year": 2006,
     "content": "NumPy is a Python library for numerical computing providing multi-dimensional array objects and tools for working with these arrays efficiently."},
]

docs = [
    Document(
        page_content=f"{a['title']}: {a['content']}",
        metadata={"title": a["title"], "category": a["category"], "year": a["year"]}
    )
    for a in articles
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embeddings, collection_name="wiki_search")

def semantic_search(query: str, category: str = None, min_year: int = None, k: int = 3) -> list[dict]:
    """Smart search with filtering and relevance ranking"""
    where_filter = {}
    if category:
        where_filter["category"] = category
    if min_year:
        where_filter["year"] = {"$gte": min_year}
    
    kwargs = {"k": k}
    if where_filter:
        kwargs["filter"] = where_filter
    
    docs_scores = vectorstore.similarity_search_with_relevance_scores(query, **kwargs)
    
    return [
        {
            "title": doc.metadata["title"],
            "category": doc.metadata["category"],
            "year": doc.metadata["year"],
            "relevance": round(score, 3),
            "snippet": doc.page_content[:150] + "..."
        }
        for doc, score in docs_scores
    ]

# Demo queries
queries = [
    ("attention mechanism neural networks", None, None),
    ("programming language", "Programming", None),
    ("recent AI models", "AI", 2020),
]

for query, cat, yr in queries:
    print(f"\n Query: '{query}'" + (f" | category={cat}" if cat else "") + (f" | year>={yr}" if yr else ""))
    results = semantic_search(query, cat, yr)
    for r in results:
        print(f"  [{r['relevance']}] {r['title']} ({r['year']}, {r['category']})")

print("\n Extended Day 15 Complete!")
```

---

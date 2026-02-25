# 📝 Comprehensive Assessment & Grading Guide

This guide is designed for both Instructors to evaluate student progress and for Students to self-audit their skills across the 30-day curriculum.

In 2025, generative AI assessments must move beyond multiple-choice questions "testing memory." An engineer is measured by their ability to architect robust, deterministic software around non-deterministic intelligence.

---

## 🏛️ Week 1: Mathematics & First Principles (Days 1-7)

**Core Competency Tested:** Can the student code the mathematical foundations of deep learning and explain the Transformer architecture without relying on abstraction libraries?

### 🔬 Practical Assessment: Building MHA
**Task:** Write a Python function using `numpy` or `PyTorch` that takes raw input tensors and computes Scaled Dot-Product Attention from scratch.
**Rubric:**
*   [0 Points]: Cannot implement matrices.
*   [10 Points]: Successfully implements $Q * K^T$ but fails to scale.
*   [15 Points]: Understands and implements the division by $\sqrt{d_k}$ and the Softmax function.
*   [25 Points]: Successfully builds the final `Attention(Q, K, V)` output and can verbally explain the shape transformation of the tensors.

### 🧠 Theoretical Assessment: Open-Response
1.  **Question:** Why does a Transformer layer require both Multi-Head Attention and a Position-wise Feed Forward Network?
    *   *Ideal Answer:* MHA routes contextual information *between* words in the sequence. The FFN acts as a per-token key-value memory bank, injecting factual world knowledge that was memorized during pre-training back into the individual vector representations.

---

## 🚀 Week 2: Orchestration & Prompting Paradigms (Days 8-14)

**Core Competency Tested:** Has the student transitioned from manual, fragile string manipulation (prompt hacking) to programmatic, typed architectures (DSPy, LangChain)?

### 🔬 Practical Assessment: DSPy RAG Compilation
**Task:** Build a 2-hop RAG system using `dspy.Signature`. Then, write a metric function and use `BootstrapFewShot` to compile the architecture over a sample dataset of 10 QA pairs.
**Rubric:**
*   [5 Points]: Uses complex, messy string prompts instead of DSPy Signatures.
*   [15 Points]: Successfully builds the RAG architecture zero-shot.
*   [20 Points]: Implements a rigorous metric function (e.g., verifying citation matching).
*   [30 Points]: Successfully calls `.compile()`, proving the model dynamically learned few-shot examples without human intervention.

### 🧠 Theoretical Assessment: Architecture Routing
1.  **Question:** You have an application that must generate complex Python scripts and output them in a strict JSON format. Which model class do you select and why?
    *   *Ideal Answer:* A medium-to-large Instruct model heavily biased towards logic (e.g., Llama-3-70B-Instruct or Claude 3.5 Sonnet). The pipeline must enforce `response_format={"type": "json_object"}` at the API level or utilize Pydantic Output Parsers locally to prevent trailing code blocks from breaking the JSON deserializer.

---

## 📚 Week 3: Advanced RAG & Vector Databases (Days 15-21)

**Core Competency Tested:** Can the student solve enterprise retrieval failures? (e.g., "Lost in the middle", hallucination, poor diversity).

### 🔬 Practical Assessment: Chunking & Recall
**Task:** Given a 50-page highly technical PDF, the student must implement a system that achieves >90% recall on complex queries.
**Rubric:**
*   [10 Points]: Uses basic `CharacterTextSplitter` and a standard naive `Similarity` search in ChromaDB. (Fails to grab context across page boundaries).
*   [25 Points]: Implements `RecursiveCharacterTextSplitter` with intelligent overlaps. Uses `Maximum Marginal Relevance (MMR)` retrieval to increase document diversity.
*   [40 Points]: Implements advanced architectures like **Parent Document Retrieval** (LlamaIndex), returning full pages based on the similarity of small embedded child-chunks.

### 🧠 Theoretical Assessment: Embedding Math
1.  **Question:** Explain the difference between Cosine Similarity and Dot Product when comparing two embedding vectors in a vector database.
    *   *Ideal Answer:* Dot product measures both the angle and the magnitude (length) of the vectors. Cosine similarity *normalizes* the vectors to length 1, measuring ONLY the angle. In embedding space, we generally care about the direction (semantic meaning), so Cosine Similarity is usually preferred to prevent long documents from unfairly dominating short documents.

---

## 🤖 Week 4: Multi-Agent Systems & MLOps (Days 22-28)

**Core Competency Tested:** Can the student build cyclic, stateful graphs (LangGraph) and deploy them asynchronously within Docker (LangServe)?

### 🔬 Practical Assessment: LangGraph State Trapping
**Task:** Build an Analyst Agent and a Web Search Agent. Force them into a loop where the Analyst rejects the search results until a specific condition is mathematically met.
**Rubric:**
*   [10 Points]: Overwrites `AgentState` list on every loop instead of appending.
*   [25 Points]: Successfully builds a cyclic graph with a `Conditional Edge`.
*   [40 Points]: The routing condition is deterministic (e.g., routing based on an Analyst LLM outputting a strict Pydantic boolean `is_sufficient=True`).

### 🛠️ Production Assessment: Docker & LangServe
**Task:** Deploy the Multi-Agent graph into a FastAPI Docker container.
**Rubric:**
*   [10 Points]: Application runs locally via `python main.py` but fails to build in Docker due to bloated dependencies.
*   [20 Points]: Runs in a Docker container, but API is synchronously blocking (hanging for 30 seconds).
*   [35 Points]: Uses `LangServe` to expose async `/stream` endpoints via Server-Sent Events (SSE). Variables mapped correctly in `docker run`. Latency tracked visually in LangSmith.

---

## 🏆 Final Capstone Evaluation Weights

For the 5 chosen Capstone Projects, instructors should grade based strictly on these 4 pillars:

1.  **Robustness (30%):** Does the application crash if the user sends garbage input? Does the RAG system hallucinate when asked a question outside its corpus?
2.  **Architecture (30%):** Are variables typed? Is LCEL (`|`) used instead of legacy chaining? Are tools dynamically bound to the LLMs?
3.  **Observability (25%):** Is LangSmith tracing active? Are token costs tracked? Can the developer trace *exactly* which node failed in the graph?
4.  **Deployment (15%):** Can the instructor pull the GitHub repo and run `docker-compose up` to instantly test the API locally? 

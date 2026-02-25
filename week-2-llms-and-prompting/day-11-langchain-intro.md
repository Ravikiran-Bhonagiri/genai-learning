# Day 11: LangChain Introduction 🔗
### Week 2 — Large Language Models & Prompting

---

## 🎯 Learning Objectives

By the end of today, you will:
- Understand LangChain's architecture and core components
- Build LLM chains using LCEL (LangChain Expression Language)
- Use document loaders, text splitters, and prompt templates
- Create a complete document summarization system
- Understand when to use LangChain vs raw API calls
- Connect chains together for multi-step pipelines

**Estimated Time:** 4–4.5 hours  
**Difficulty:** ⭐⭐⭐ Intermediate  
**Prerequisites:** Days 8–10 (LLM APIs, Prompt Engineering)

---

## 📚 Section 1: Why LangChain?

### 1.1 The Problem LangChain Solves

Building LLM applications involves many repetitive, boilerplate tasks:
- Managing prompts and their templates
- Parsing and validating model outputs
- Chaining multiple LLM calls together
- Managing conversation memory
- Loading and processing documents
- Connecting to vector databases

LangChain provides standardized abstractions for ALL of these, letting you focus on application logic rather than infrastructure.

### 1.2 LangChain Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LANGCHAIN ECOSYSTEM                   │
├─────────────────────────────────────────────────────────┤
│  langchain-core    → Base abstractions (Runnable, etc.)  │
│  langchain         → Chains, agents, memory              │
│  langchain-openai  → OpenAI integration                  │
│  langchain-google  → Google/Gemini integration           │
│  langchain-community → 3rd party integrations           │
│  langgraph         → Stateful, graph-based agents        │
│  langserve         → Deploy chains as REST APIs          │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Core Abstractions

| Component | What It Does | Example |
|-----------|-------------|---------|
| **LLM/ChatModel** | Wraps an AI model | `ChatOpenAI`, `ChatGoogleGenerativeAI` |
| **PromptTemplate** | Manages prompt construction | `ChatPromptTemplate.from_messages()` |
| **Chain** | Connects components | `prompt | model | parser` |
| **OutputParser** | Parses model output | `StrOutputParser`, `JsonOutputParser` |
| **Retriever** | Finds relevant documents | `vectorstore.as_retriever()` |
| **Memory** | Maintains conversation state | `ConversationBufferMemory` |
| **Agent** | Autonomous decision-making | `create_react_agent()` |
| **Tool** | Functions agents can call | `@tool` decorator |

---

## 📚 Section 2: Installation & Setup

```bash
pip install langchain langchain-openai langchain-core \
            langchain-community python-dotenv \
            tiktoken pypdf docx2txt
```

```python
# verify_setup.py
import langchain
print(f"LangChain version: {langchain.__version__}")

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Basic connection test
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

response = llm.invoke("Say hello in exactly 5 words.")
print("LLM Response:", response.content)
print("✅ Setup verified!")
```

---

## 📚 Section 3: LangChain Expression Language (LCEL)

### 3.1 What is LCEL?

LCEL is LangChain's modern, composable way to build chains using the `|` (pipe) operator. It's inspired by Unix piping, where data flows from one component to the next.

```
prompt | model | output_parser
```

This simple pipe creates a full chain:
1. `prompt` formats the template with user input
2. `model` sends the formatted prompt to the LLM and gets a response
3. `output_parser` extracts the text from the response object

### 3.2 Your First LCEL Chain

```python
# first_chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# The three components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specializing in {domain}."),
    ("human", "{question}")
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# Compose into a chain using | operator
chain = prompt | model | parser

# Invoke the chain
result = chain.invoke({
    "domain": "Python programming",
    "question": "What's the difference between a list and a tuple?"
})

print(result)

# Stream the output
print("\n--- Streaming version ---")
for chunk in chain.stream({
    "domain": "machine learning",
    "question": "Explain overfitting in one paragraph."
}):
    print(chunk, end="", flush=True)
print()
```

### 3.3 The Runnable Interface

LCEL works because every component implements the `Runnable` interface, which provides:

| Method | Description |
|--------|-------------|
| `.invoke(input)` | Process a single input synchronously |
| `.stream(input)` | Stream output token by token |
| `.batch(inputs)` | Process multiple inputs in parallel |
| `.ainvoke(input)` | Async version of invoke |
| `.astream(input)` | Async streaming |

```python
# batch processing — very efficient
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

classifier = (
    PromptTemplate.from_template(
        "Classify this text as SPAM or NOT_SPAM. Output only the label.\n\nText: {text}"
    )
    | model
    | parser
)

texts = [
    "CONGRATULATIONS! You've won $1,000,000! Click here now!",
    "Hi John, wanted to follow up on our meeting from yesterday.",
    "FREE PILLS! LOSE 30 POUNDS IN 3 DAYS! BUY NOW!",
    "The quarterly report is attached for your review.",
    "UPDATE YOUR ACCOUNT IMMEDIATELY OR IT WILL BE DELETED"
]

results = classifier.batch([{"text": t} for t in texts])
for text, label in zip(texts, results):
    print(f"{'✉️ ' if 'SPAM' not in label else '🚫 '} {label}: {text[:60]}")
```

### 3.4 RunnableLambda — Custom Steps

Add custom Python functions into your chain:

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Custom preprocessing step
def preprocess_text(input_dict: dict) -> dict:
    """Clean and prepare text before sending to LLM"""
    text = input_dict["text"]
    # Remove extra whitespace, normalize
    text = " ".join(text.split())
    # Truncate if too long 
    if len(text) > 3000:
        text = text[:3000] + "...[truncated]"
    return {"text": text, "word_count": len(text.split())}

# Custom postprocessing step
def format_summary(summary: str) -> dict:
    """Structure the final output"""
    return {
        "summary": summary,
        "word_count": len(summary.split()),
        "sentences": len(summary.split('.')),
    }

# Chain with custom steps
chain = (
    RunnableLambda(preprocess_text)
    | PromptTemplate.from_template(
        "Summarize this text in 3 bullet points:\n\n{text}"
    )
    | model
    | StrOutputParser()
    | RunnableLambda(format_summary)
)

result = chain.invoke({
    "text": """
    The transformer architecture, introduced in the seminal paper "Attention is All You Need" 
    by Vaswani et al. in 2017, has become the dominant paradigm in natural language processing 
    and has expanded significantly into computer vision, audio processing, and beyond. 
    The core innovation is the self-attention mechanism, which allows the model to weigh 
    the importance of different parts of the input sequence when processing each element.
    Unlike recurrent neural networks, transformers process all tokens in parallel, 
    enabling much more efficient training on modern GPU hardware.
    """
})

print("CHAIN RESULT:")
for k, v in result.items():
    print(f"  {k}: {v}")
```

---

## 📚 Section 4: Working with Documents

### 4.1 Document Loaders

LangChain provides document loaders for dozens of file types:

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
    DirectoryLoader
)

# Load a text file
text_loader = TextLoader("path/to/file.txt")
text_docs = text_loader.load()
print(f"Loaded {len(text_docs)} document(s)")
print(f"Content preview: {text_docs[0].page_content[:200]}")
print(f"Metadata: {text_docs[0].metadata}")

# Load a PDF
pdf_loader = PyPDFLoader("path/to/document.pdf")
pdf_pages = pdf_loader.load()
print(f"PDF has {len(pdf_pages)} pages")

# Load from a web URL
web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)")
web_docs = web_loader.load()
print(f"Loaded {len(web_docs)} web document(s)")

# Load all .txt files from a directory
dir_loader = DirectoryLoader("./data/", glob="**/*.txt", loader_cls=TextLoader)
all_docs = dir_loader.load()
print(f"Loaded {len(all_docs)} files from directory")
```

### 4.2 Text Splitters

Documents must be split into manageable chunks before processing:

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

# Most commonly used — RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Target size (in characters)
    chunk_overlap=200,     # Overlap between chunks (for context continuity)
    length_function=len,   # How to measure length
    separators=["\n\n", "\n", ". ", " ", ""],  # Split priority order
    add_start_index=True   # Add character position to metadata
)

# Example text
long_text = """
Chapter 1: Introduction to Machine Learning

Machine learning is a subfield of artificial intelligence that focuses on 
building systems that learn from data. Unlike traditional rule-based programming, 
ML algorithms improve their performance automatically through experience.

There are three main types of machine learning:
1. Supervised Learning: The algorithm learns from labeled training data.
2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
3. Reinforcement Learning: The algorithm learns through trial and error.

Chapter 2: Neural Networks

Neural networks are computing systems inspired by biological neural networks 
in animal brains. They consist of layers of interconnected nodes (neurons).
Each connection has a weight that is adjusted during training.
"""

chunks = splitter.split_text(long_text)
print(f"Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk[:150] + "...")

# For token-aware splitting (better for LLM context windows)
token_splitter = TokenTextSplitter(
    chunk_size=200,     # Tokens, not characters
    chunk_overlap=50
)
token_chunks = token_splitter.split_text(long_text)
print(f"\nToken-aware split: {len(token_chunks)} chunks")
```

---

## 📚 Section 5: Chains — The Heart of LangChain

### 5.1 Sequential Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# Chain 1: Extract main topic from text
extract_chain = (
    ChatPromptTemplate.from_template(
        "What is the main topic of the following text? Give a 3-word phrase.\n\nText: {text}"
    )
    | model
    | parser
)

# Chain 2: Generate questions about a topic
quiz_chain = (
    ChatPromptTemplate.from_template(
        "Generate 3 quiz questions about: {topic}\nFormat as numbered list."
    )
    | model
    | parser
)

# Combine: extract topic → generate quiz
# RunnablePassthrough preserves original input alongside new data
combined_chain = (
    {"topic": extract_chain, "text": RunnablePassthrough()}
    | RunnablePassthrough.assign(quiz=quiz_chain)
)

result = combined_chain.invoke({
    "text": "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in glucose. It occurs in chloroplasts using chlorophyll to absorb light."
})

print("Topic:", result.get("topic", "N/A"))
print("\nQuiz:")
print(result.get("quiz", "N/A"))
```

### 5.2 Conditional Routing

Route requests to different chains based on content:

```python
from langchain_core.runnables import RunnableBranch

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# Different chains for different topics
coding_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an expert software engineer. Provide code examples."),
        ("human", "{question}")
    ])
    | model | parser
)

science_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a science educator. Use simple analogies."),
        ("human", "{question}")
    ])
    | model | parser
)

general_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful general assistant."),
        ("human", "{question}")
    ])
    | model | parser
)

# Classifier to determine routing
def classify_question(input_dict: dict) -> str:
    classify_prompt = f"""Classify this question into one category: CODING, SCIENCE, or GENERAL.
Output only the category name.
Question: {input_dict['question']}"""
    
    result = model.invoke(classify_prompt)
    return parser.invoke(result).strip()

# Router chain
router = RunnableBranch(
    (lambda x: "CODING" in classify_question(x).upper(), coding_chain),
    (lambda x: "SCIENCE" in classify_question(x).upper(), science_chain),
    general_chain  # Default
)

# Test the router
questions = [
    "How do I implement a binary search tree in Python?",
    "Why is the sky blue?",
    "What's the best way to stay focused while studying?"
]

for q in questions:
    print(f"\n❓ {q}")
    result = router.invoke({"question": q})
    print(f"📝 {result[:200]}...")
```

---

## 💻 Full Lab: Document Summarization System

```python
# lab_day11_summarizer.py
"""
Day 11 Lab — Complete Document Summarization System using LangChain
Supports: PDFs, text files, web pages, and YouTube transcripts
"""

import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from dotenv import load_dotenv

load_dotenv()

# ── Initialize Components ─────────────────────────────────
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()
splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)

# ── Summarization Chains ──────────────────────────────────

# Map chain: summarize each chunk independently
map_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting key information from text. Be concise."),
    ("human", """Extract the key points from this text section:

{chunk}

List the 3-5 most important points as bullet points.""")
])

map_chain = map_prompt | model | parser

# Reduce chain: combine all chunk summaries
reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert report writer who synthesizes information clearly."),
    ("human", """You have received summaries from different sections of a document.
Synthesize them into a comprehensive, well-structured final summary.

SECTION SUMMARIES:
{combined_summaries}

Create a final summary with:
## Executive Summary (2-3 sentences)
## Key Points (5-8 bullet points, most important across all sections)
## Key Conclusions
## Action Items (if applicable)

Target length: 300-500 words.""")
])

reduce_chain = reduce_prompt | model | parser

def map_reduce_summarize(text: str) -> dict:
    """Full map-reduce summarization pipeline"""
    
    # Split into chunks
    chunks = splitter.split_text(text)
    print(f"  Processing {len(chunks)} text chunks...")
    
    if len(chunks) == 1:
        # Short document — direct summarization
        direct_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional summarizer."),
            ("human", "Summarize this document comprehensively:\n\n{text}")
        ])
        return {
            "method": "direct",
            "summary": (direct_prompt | model | parser).invoke({"text": text})
        }
    
    # Map: summarize each chunk
    chunk_summaries = map_chain.batch([{"chunk": c} for c in chunks])
    
    # Reduce: combine summaries
    combined = "\n\n---SECTION---\n\n".join(
        [f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)]
    )
    
    final_summary = reduce_chain.invoke({"combined_summaries": combined})
    
    return {
        "method": "map-reduce",
        "chunk_count": len(chunks),
        "section_summaries": chunk_summaries,
        "final_summary": final_summary
    }


class DocumentSummarizer:
    """Universal document summarizer supporting multiple source types"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0.3)
        self.parser = StrOutputParser()
    
    def load_from_url(self, url: str) -> str:
        """Load text from a web URL"""
        loader = WebBaseLoader(url)
        docs = loader.load()
        return "\n\n".join([d.page_content for d in docs])
    
    def load_from_file(self, filepath: str) -> str:
        """Load text from a file"""
        path = Path(filepath)
        if path.suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(filepath)
        else:
            loader = TextLoader(filepath)
        docs = loader.load()
        return "\n\n".join([d.page_content for d in docs])
    
    def load_from_text(self, text: str) -> str:
        return text
    
    def summarize(self, source: str, source_type: str = "text", 
                  output_format: str = "detailed") -> dict:
        """
        Summarize a document from any source.
        
        Args:
            source: URL, file path, or raw text
            source_type: 'url', 'file', or 'text'
            output_format: 'brief', 'detailed', or 'structured'
        """
        print(f"📄 Loading document from {source_type}...")
        
        if source_type == "url":
            text = self.load_from_url(source)
        elif source_type == "file":
            text = self.load_from_file(source)
        else:
            text = source
        
        print(f"✅ Loaded {len(text.split())} words")
        print(f"🔄 Running map-reduce summarization...")
        
        result = map_reduce_summarize(text)
        result["source_type"] = source_type
        result["word_count"] = len(text.split())
        
        return result
    
    def compare_summaries(self, text: str) -> None:
        """Compare different summary styles"""
        styles = {
            "One-line": "Summarize in exactly one sentence.",
            "Executive Brief": "Write a 50-word executive summary for a CEO.",
            "Beginner-Friendly": "Explain to a 5th grader what this text is about.",
            "Technical": "Provide a technical analysis with key concepts, methods, and implications."
        }
        
        print("\n📊 SUMMARY COMPARISON")
        print("=" * 55)
        
        for style_name, instruction in styles.items():
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", f"You are a summarizer. Task: {instruction}"),
                    ("human", "Text: {text}")
                ])
                | self.model | self.parser
            )
            summary = chain.invoke({"text": text[:2000]})  # Use first 2000 chars
            print(f"\n[{style_name.upper()}]")
            print(summary[:400])
            print("-" * 40)


# ── Demo ─────────────────────────────────────────────────
summarizer = DocumentSummarizer()

# Test with inline text
sample_text = """
The Large Language Model (LLM) landscape has undergone dramatic changes in 2024-2025.
OpenAI's GPT-4o represents a major step toward multimodal AI, handling text, images, 
and audio in a unified model. Anthropic's Claude 3 family demonstrated that larger 
context windows (up to 200K tokens) enable entirely new use cases like analyzing 
entire codebases or legal documents in a single prompt.

Google's Gemini 1.5 Pro surprised the industry with a 1 million token context window
and strong performance on long-document reasoning tasks. The open-source ecosystem
has matured significantly, with Meta's Llama 3 achieving near-GPT-4 quality on many 
benchmarks while being freely downloadable and locally runnable.

Key trends shaping the field:
1. Context windows growing from 4K to 1M+ tokens
2. Multimodal capabilities becoming standard (vision, audio, code)
3. Open-source models closing the gap with proprietary ones
4. Inference efficiency improving dramatically (speculative decoding, quantization)
5. Agentic capabilities becoming a key differentiator

The competitive dynamics have intensified, with dozens of well-funded startups
(Mistral, Cohere, AI21, etc.) competing with tech giants. Pricing has dropped 90%+ 
since GPT-3, democratizing access. The focus has shifted from "can it do X?" to 
"how do we deploy X reliably in production?"
"""

print("=" * 55)
print("DEMO: Document Summarization System")
print("=" * 55)

# Full summarization
result = summarizer.summarize(sample_text, source_type="text")
print("\n[FINAL SUMMARY]")
print(result.get("final_summary") or result.get("summary"))

# Style comparison
summarizer.compare_summaries(sample_text)

print("\n✅ Day 11 Lab Complete!")
```

---

## 🎯 Mini Project: Research Paper Summarizer

Build a system that can process academic papers and generate structured research briefs.

**Requirements:**
1. Accept PDF or URL input
2. Extract: title, authors, key contributions, methodology, results, limitations
3. Generate a 1-page structured brief suitable for a research team
4. Include: "Why this matters" and "Next steps" sections
5. Save output as a formatted Markdown file

**Starter:**
```python
from langchain_community.document_loaders import PyPDFLoader, ArxivLoader

# Load from ArXiv
loader = ArxivLoader(query="attention is all you need", load_max_docs=1)
docs = loader.load()
text = docs[0].page_content
# ... build the rest of the pipeline
```

---

## 🧠 Quiz: Day 11 — LangChain Introduction

**Q1:** In LCEL, what does the `|` operator do?
- A) Logical OR operation
- B) **Pipes output of one component as input to the next ✅**
- C) Parallel execution of chains
- D) Creates a branching condition

**Q2:** What interface do all LCEL components implement?
- A) Chainable
- B) Linkable
- C) **Runnable ✅**
- D) Composable

**Q3:** What is `chunk_overlap` in text splitters used for?
- A) Deduplication of repeated text
- B) **Preserving context across chunk boundaries ✅**
- C) Identifying duplicate chunks
- D) Formatting code blocks

**Q4:** Which LangChain component would you use to load text from a web URL?
- A) URLSplitter
- B) TextTransformer
- C) **WebBaseLoader ✅**
- D) HttpRetriever

**Q5:** What is `RunnablePassthrough` used for?
- A) Skipping a chain step
- B) **Passing input through unchanged (while other values are computed) ✅**
- C) Running chains in parallel
- D) Caching chain results

**Q6:** In a Map-Reduce summarization approach, what happens in the "Map" step?
- A) The entire document is mapped to a single summary
- B) **Each chunk is summarized independently ✅**
- C) All summaries are combined
- D) Keywords are mapped to embeddings

**Q7:** `RunnableBranch` is used for:
- A) Running multiple chains in parallel
- B) **Conditional routing to different chains ✅**
- C) Splitting documents into branches
- D) Creating sub-chains within main chains

**Q8:** Which method would you call to process a list of inputs efficiently in LangChain?
- A) `.invoke()`
- B) `.stream()`
- C) `.multi()`
- D) **`.batch()` ✅**

---

## 📊 Key Takeaways

| Concept | Key Point |
|---------|-----------|
| **LCEL** | `prompt \| model \| parser` — composable, readable chain building |
| **Runnable Interface** | `.invoke()`, `.stream()`, `.batch()`, `.ainvoke()` |
| **Document Loaders** | Standardized loading for PDF, text, web, CSV, and more |
| **Text Splitters** | Break large docs into LLM-compatible chunks with overlap |
| **Map-Reduce** | Summarize chunks independently, then combine — handles any length |
| **RunnablePassthrough** | Threads original input through multi-branch chains |
| **RunnableBranch** | Routes to different chains based on conditions |
| **RunnableLambda** | Wraps any Python function as a Runnable step |

---

## 📖 Further Reading

### Official Documentation
- [LangChain Documentation](https://python.langchain.com/docs/introduction/) — Comprehensive guides
- [LCEL How-To Guides](https://python.langchain.com/docs/how_to/#langchain-expression-language-lcel) — LCEL patterns
- [Document Loaders Reference](https://python.langchain.com/docs/integrations/document_loaders/)

### Key Blog Posts
- [LangChain LCEL Introduction](https://blog.langchain.dev/langchain-expression-language/)
- [Building Production LLM Apps with LangChain](https://blog.langchain.dev/)

---

## 🔄 What's Next: Day 12 Preview

Tomorrow we tackle **Memory and Context Management** — one of the most critical skills for building useful AI applications:
- Why LLMs are "stateless" and how to give them memory
- `ConversationBufferMemory` vs `ConversationSummaryMemory`
- Managing context windows at scale
- Long-term memory with vector stores
- Building a fully stateful chatbot that remembers everything

---

*Day 11 Complete ✅ | GenAI Course — Week 2 | Next: Day 12 — Memory & Context Management*

---

##  Section 6: Advanced LCEL Patterns

### 6.1 RunnableParallel  Execute Multiple Chains at Once

When you need multiple analyses on the same input simultaneously, `RunnableParallel` runs them concurrently:

```python
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()

# Run multiple analyses in ONE call
parallel_chain = RunnableParallel(
    sentiment=ChatPromptTemplate.from_template(
        "Classify the sentiment as POSITIVE, NEGATIVE, or NEUTRAL. One word only.\nText: {text}"
    ) | model | parser,
    
    topics=ChatPromptTemplate.from_template(
        "List the 3 main topics in this text, comma-separated.\nText: {text}"
    ) | model | parser,
    
    word_count=lambda x: str(len(x["text"].split())),
    
    summary=ChatPromptTemplate.from_template(
        "Summarize in one crisp sentence.\nText: {text}"
    ) | model | parser,
    
    language=ChatPromptTemplate.from_template(
        "What language is this text in? One word.\nText: {text}"
    ) | model | parser
)

text = """
The recent advancement in quantum computing by IBM has demonstrated 
a 1000-qubit processor, marking a significant milestone in the race 
toward practical quantum advantage. Scientists believe this could 
eventually break current encryption standards, prompting cryptographers 
to accelerate post-quantum cryptography standards.
"""

results = parallel_chain.invoke({"text": text})
print("Parallel Analysis Results:")
for key, value in results.items():
    print(f"  {key:12}: {value}")
```

### 6.2 RunnableWithFallbacks  Graceful Degradation

Build resilient chains that fall back to simpler models when primary ones fail:

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

# Primary model
primary = ChatOpenAI(model="gpt-4o", temperature=0)

# Fallback model (cheaper, always available)
fallback = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Local fallback (simulated)
def local_fallback(prompt):
    return type("Response", (), {"content": f"Fallback response for: {str(prompt)[:50]}..."})()

# Chain with automatic fallbacks
resilient_chain = primary.with_fallbacks(
    [fallback],
    exceptions_to_handle=(Exception,)
)

# Test it
try:
    result = resilient_chain.invoke("Explain quantum entanglement simply.")
    print(f"Response: {result.content[:200]}")
except Exception as e:
    print(f"All fallbacks exhausted: {e}")
```

### 6.3 Caching Responses

Avoid repeated API calls for identical inputs:

```python
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

# Enable SQLite caching  responses are cached on disk
set_llm_cache(SQLiteCache(database_path="./llm_cache.db"))

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# First call  goes to API
import time
start = time.time()
result1 = model.invoke("What is the capital of France?")
first_call_time = time.time() - start

# Second call  served from cache instantly
start = time.time()
result2 = model.invoke("What is the capital of France?")
cached_call_time = time.time() - start

print(f"First call:  {first_call_time:.2f}s  {result1.content}")
print(f"Cached call: {cached_call_time:.4f}s  {result2.content}")
print(f"Speedup: {first_call_time/cached_call_time:.0f}x faster")
```

### 6.4 Streaming with Callbacks

Real-time streaming with progress tracking:

```python
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Streaming model
streaming_model = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

chain = (
    ChatPromptTemplate.from_template("Write a short story about {topic}.")
    | streaming_model
    | StrOutputParser()
)

# Streams tokens to stdout in real time
print("Generating story (streaming):\n" + "="*40)
result = chain.invoke({"topic": "a robot learning to paint"})
print("\n" + "="*40)
print(f"\nTotal length: {len(result)} chars")
```

---

##  Section 7: Prompt Templates in Depth

### 7.1 Few-Shot Prompt Templates

```python
from langchain_core.prompts import (
    ChatPromptTemplate, 
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Define few-shot examples
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "fast", "output": "slow"},
    {"input": "hot", "output": "cold"},
    {"input": "light", "output": "dark"},
]

# Example template
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# Combine into few-shot template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# Final prompt with context
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at finding antonyms."),
    few_shot_prompt,
    ("human", "{word}")
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = final_prompt | model | StrOutputParser()

test_words = ["beautiful", "ancient", "noisy", "brave", "complex"]
for word in test_words:
    antonym = chain.invoke({"word": word})
    print(f"{word:12}  {antonym}")
```

### 7.2 Dynamic Few-Shot Selection (Semantic Similarity)

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Large pool of examples
examples_pool = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "airplane"},
    {"input": "teacher", "output": "classroom"},
    {"input": "doctor", "output": "hospital"},
    {"input": "chef", "output": "kitchen"},
    {"input": "farmer", "output": "field"},
    {"input": "librarian", "output": "library"},
    {"input": "firefighter", "output": "fire station"},
]

# Select most relevant examples based on semantic similarity
selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples_pool,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=3  # Select 3 most relevant
)

# When asked about "surgeon", it retrieves doctor/hospital, etc.
selected = selector.select_examples({"input": "surgeon"})
print("Dynamically selected examples:")
for ex in selected:
    print(f"  {ex['input']}  {ex['output']}")
```

### 7.3 Prompt Composition

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Reusable system component
persona_template = "You are {persona}. Always respond in {style} style."

# Reusable task component  
task_template = "Task: {task_description}\n\nInput: {user_input}"

# Compose into a full prompt
composed_prompt = ChatPromptTemplate.from_messages([
    ("system", persona_template),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", task_template)
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
chain = composed_prompt | model | StrOutputParser()

# Different personas using same template
personas = [
    {"persona": "a Shakespearean playwright", "style": "poetic Elizabethan"},
    {"persona": "a Silicon Valley tech bro", "style": "startup jargon"},
    {"persona": "a stern professor", "style": "formal academic"},
]

for p in personas:
    result = chain.invoke({
        **p,
        "task_description": "Explain machine learning",
        "user_input": "What is overfitting?",
        "chat_history": []
    })
    print(f"\n[{p['persona']}]")
    print(result[:200] + "...")
```

---

##  Section 8: Error Handling & Debugging Chains

### 8.1 Inspecting Chain Internals

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Summarize: {text}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Inspect chain structure
print("Chain structure:")
print(chain)

# Get input schema
print("\nInput schema:")
print(chain.input_schema.schema())

# Get output schema
print("\nOutput schema:")
print(chain.output_schema.schema())
```

### 8.2 Middleware  Logging Every Step

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import time

def make_logger(step_name: str):
    """Create a logging middleware for a chain step"""
    def log(x):
        print(f"  [{step_name}] Input: {str(x)[:100]}")
        return x
    return RunnableLambda(log)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Chain with logging at every step
verbose_chain = (
    make_logger("pre-prompt")
    | ChatPromptTemplate.from_template("What is {concept}? Answer in one sentence.")
    | make_logger("post-prompt")
    | model
    | make_logger("post-llm")
    | StrOutputParser()
    | make_logger("post-parser")
)

result = verbose_chain.invoke({"concept": "gradient descent"})
print(f"\nFinal: {result}")
```

### 8.3 Retry Logic

```python
from langchain_core.runnables import RunnableRetry
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import random

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# A chain that parses JSON output
json_chain = (
    ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'name' and 'value' for: {item}"
    )
    | model
    | JsonOutputParser()
)

# Wrap with retry  will retry up to 3 times on failure
json_chain_with_retry = json_chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)

result = json_chain_with_retry.invoke({"item": "speed of light"})
print(f"Result: {result}")
```

---

##  Extended Project: Multi-Source Research Assistant

Combine everything from Day 11 into a comprehensive research assistant:

```python
# extended_lab_day11.py
"""
Extended Project: Multi-Source Research Assistant
- Loads content from web URLs and local files
- Runs parallel topic extraction, sentiment, and key fact identification  
- Produces a structured research brief
- Compares answers from different model configurations
"""

from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def load_web_content(url: str) -> str:
    """Load and clean web content"""
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    return " ".join(c.page_content for c in chunks[:3])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
parser = StrOutputParser()

def create_research_brief(text: str) -> dict:
    """Generate a structured research brief from any text"""
    
    analysis = RunnableParallel(
        executive_summary=ChatPromptTemplate.from_template(
            "Write a 3-sentence executive summary of this text.\n\n{text}"
        ) | model | parser,
        
        key_facts=ChatPromptTemplate.from_template(
            "List 5-7 most important facts from this text. Be specific with numbers/dates.\n\n{text}"
        ) | model | parser,
        
        main_themes=ChatPromptTemplate.from_template(
            "Identify the 3 main themes or topics. One phrase each.\n\n{text}"
        ) | model | parser,
        
        implications=ChatPromptTemplate.from_template(
            "What are 3 key implications or takeaways from this text?\n\n{text}"
        ) | model | parser,
        
        questions=ChatPromptTemplate.from_template(
            "List 3 important questions this text raises but doesn't fully answer.\n\n{text}"
        ) | model | parser,
    )
    
    # Use first 4000 chars for parallel analysis
    truncated = text[:4000]
    return analysis.invoke({"text": truncated})

# Sample text for demo  
sample = """
Artificial intelligence is transforming the pharmaceutical industry at unprecedented speed.
Drug discovery, which traditionally took 15 years and cost $2.6 billion on average, is being 
compressed to just 2-4 years using AI-driven molecular simulation and protein folding prediction.
DeepMind's AlphaFold has already mapped the structure of virtually all known proteins  a task 
that would have taken traditional methods thousands of years to complete.

Clinical trials are being accelerated through AI matching of patients to trials based on 
genomic, lifestyle, and medical history data. Companies like Recursion Pharmaceuticals and 
Insilico Medicine have AI platforms that have already produced drug candidates now in clinical 
trials. Regulatory agencies like the FDA are updating guidelines to accommodate AI-designed 
compounds. The economic potential is enormous: AI could add $100 billion in annual value 
to the pharmaceutical sector by 2030.
"""

print(" Research Brief Generator")
print("=" * 60)
brief = create_research_brief(sample)

print("\n EXECUTIVE SUMMARY")
print(brief["executive_summary"])
print("\n KEY FACTS")
print(brief["key_facts"])
print("\n MAIN THEMES")
print(brief["main_themes"])
print("\n IMPLICATIONS")
print(brief["implications"])
print("\n OPEN QUESTIONS")
print(brief["questions"])

print("\n Extended Day 11 Complete!")
```

---

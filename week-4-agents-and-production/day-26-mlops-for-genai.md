# Day 26: MLOps for GenAI 📊
### Week 4 — AI Agents & Production Systems

---

## 🎯 Learning Objectives

By the end of today, you will:
- Track LLM experiments with LangSmith
- Log prompts, costs, and latencies automatically
- Set up evaluation pipelines with datasets
- Monitor production LLM apps in real-time
- Structure reproducible experiments

**Estimated Time:** 3.5 hours  
**Difficulty:** ⭐⭐⭐ Intermediate  
**Prerequisites:** Days 11–13 (LangChain)

---

## 📚 Section 1: Why MLOps Matters for GenAI

### 1.1 The Observability Problem

Without observability, GenAI in production is a black box:
- ❌ Can't see which prompts cause failures
- ❌ Don't know actual costs per query
- ❌ Can't replay bugs (non-deterministic outputs)
- ❌ No regression testing when prompts change
- ❌ Blind to latency bottlenecks

**MLOps for GenAI** = Observability + Experimentation + Evaluation + Monitoring

### 1.2 The GenAI MLOps Stack

| Layer | Tools | Purpose |
|-------|-------|---------|
| **Tracing** | LangSmith, Langfuse | Log every LLM call, tool use, chain step |
| **Evaluation** | RAGAS, custom eval | Automated quality scoring |
| **Experiment Tracking** | MLflow, W&B | Track prompt versions, model changes |
| **Cost Monitoring** | LangSmith, OpenMeter | Token costs per user, per query |
| **Alerting** | Datadog, custom | Detect quality degradation |

---

## 📚 Section 2: LangSmith — Tracing & Evaluation

### 2.1 Setup

```bash
pip install langsmith langchain langchain-openai
```

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key"
os.environ["LANGCHAIN_PROJECT"] = "genai-course-day26"
```

Once set, **every LangChain call** is automatically traced — zero code changes needed.

### 2.2 What LangSmith Captures Automatically

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# With tracing enabled, every invocation below is logged:
# - Input/output for each chain step
# - Model name, token counts, latency
# - Prompt template with rendered variables
# - Any tool calls and their results
# - Cost estimates

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("Summarize this in one sentence: {text}")
chain = prompt | llm | StrOutputParser()

# This call is traced automatically
result = chain.invoke({"text": "Transformers use attention mechanisms..."})
print(result)

# Each trace appears in LangSmith dashboard with:
# - Full prompt sent to API
# - Response received
# - Token counts (input + output)
# - Latency breakdown per step
# - Cost estimate
```

### 2.3 Manual Tracing with Decorators

```python
from langsmith import traceable
from langchain_openai import ChatOpenAI
import time

@traceable(name="RAG Pipeline", run_type="chain")
def my_rag_pipeline(question: str, context: str) -> dict:
    """Custom tracing for a RAG pipeline"""
    
    start = time.time()
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # This call is nested under the parent trace
    answer = llm.invoke(f"Context: {context}\n\nQuestion: {question}")
    
    latency = time.time() - start
    
    return {
        "answer": answer.content,
        "latency": latency,
        "context_length": len(context),
        "question_length": len(question)
    }

@traceable(name="Retrieval Step", run_type="retriever")
def retrieve_documents(query: str) -> list[str]:
    """Mock retrieval — traced as a retriever run"""
    # In real system, this queries ChromaDB
    return [
        "Transformers use self-attention to process sequences.",
        "Self-attention computes relationships between all token pairs.",
    ]

# These calls create nested traces in LangSmith
docs = retrieve_documents("How do transformers work?")
result = my_rag_pipeline(
    question="How do transformers work?",
    context="\n".join(docs)
)
```

### 2.4 Creating an Evaluation Dataset

```python
from langsmith import Client

ls_client = Client()

# Create a benchmark dataset for your RAG system
dataset_name = "GenAI Course RAG Benchmark"
dataset = ls_client.create_dataset(
    dataset_name=dataset_name,
    description="Benchmark QA pairs for the RAG system"
)

# Add example test cases
examples = [
    {
        "inputs": {"question": "What is retrieval-augmented generation?"},
        "outputs": {"answer": "RAG combines retrieval of relevant documents with LLM generation to produce accurate, grounded responses."}
    },
    {
        "inputs": {"question": "How does LoRA reduce training parameters?"},
        "outputs": {"answer": "LoRA decomposes weight updates into two low-rank matrices B and A, drastically reducing trainable parameters."}
    },
    {
        "inputs": {"question": "What is the transformer attention mechanism?"},
        "outputs": {"answer": "Attention computes dot products between query and key vectors to weigh the importance of values in the sequence."}
    }
]

for ex in examples:
    ls_client.create_example(
        inputs=ex["inputs"],
        outputs=ex["outputs"],
        dataset_id=dataset.id
    )

print(f"✅ Dataset created: {dataset_name} with {len(examples)} examples")
```

### 2.5 Running Automated Evaluation

```python
from langsmith.evaluation import evaluate
from langsmith import Client
from langchain_openai import ChatOpenAI

client = Client()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# The pipeline to evaluate
def my_pipeline(inputs: dict) -> dict:
    """The system under test"""
    result = llm.invoke(inputs["question"])
    return {"answer": result.content}

# Define evaluators
def exact_match_evaluator(run, example) -> dict:
    """Simple exact match evaluator"""
    prediction = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()
    score = 1.0 if expected in prediction or prediction in expected else 0.0
    return {"key": "exact_match", "score": score}

def length_evaluator(run, example) -> dict:
    """Check if answer is sufficiently detailed"""
    answer = run.outputs.get("answer", "")
    score = 1.0 if len(answer.split()) >= 10 else 0.0
    return {"key": "answer_length_ok", "score": score}

# LLM-as-Judge evaluator
def llm_judge_evaluator(run, example) -> dict:
    """Use GPT-4 to judge answer quality"""
    question = example.inputs.get("question")
    answer = run.outputs.get("answer")
    expected = example.outputs.get("answer")
    
    judge_response = llm.invoke(f"""
Score 1-5: Does this answer correctly address the question?
Question: {question}
Expected: {expected}
Actual: {answer}
Return only a number 1-5:""")
    
    try:
        score = int(judge_response.content.strip()) / 5.0
    except:
        score = 0.5
    
    return {"key": "llm_judge", "score": score}

# Run evaluation
results = evaluate(
    my_pipeline,
    data=dataset_name,
    evaluators=[exact_match_evaluator, length_evaluator, llm_judge_evaluator],
    experiment_prefix="baseline_gpt4o_mini"
)

print(f"Evaluation complete. Results available in LangSmith.")
```

---

## 📚 Section 3: Cost & Token Monitoring

```python
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate

def monitor_costs(prompts_and_inputs: list[dict]) -> dict:
    """Monitor token usage and costs across multiple calls"""
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = ChatPromptTemplate.from_template("{input}") | llm
    
    total_stats = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost_usd": 0.0,
        "calls": 0,
        "details": []
    }
    
    for item in prompts_and_inputs:
        with get_openai_callback() as cb:
            result = chain.invoke({"input": item["input"]})
        
        call_stats = {
            "input_preview": item["input"][:50],
            "tokens": cb.total_tokens,
            "cost": cb.total_cost
        }
        total_stats["details"].append(call_stats)
        total_stats["total_tokens"] += cb.total_tokens
        total_stats["prompt_tokens"] += cb.prompt_tokens
        total_stats["completion_tokens"] += cb.completion_tokens
        total_stats["total_cost_usd"] += cb.total_cost
        total_stats["calls"] += 1
    
    avg_cost = total_stats["total_cost_usd"] / total_stats["calls"]
    print(f"\n📊 Cost Report ({total_stats['calls']} calls):")
    print(f"  Total tokens: {total_stats['total_tokens']:,}")
    print(f"  Total cost:   ${total_stats['total_cost_usd']:.6f}")
    print(f"  Avg per call: ${avg_cost:.6f}")
    
    return total_stats

test_inputs = [
    {"input": "What is machine learning?"},
    {"input": "Explain neural networks."},
    {"input": "Describe the transformer architecture."},
]

stats = monitor_costs(test_inputs)
```

---

## 📚 Section 4: Prompt Version Control

```python
from pathlib import Path
import json
from datetime import datetime

class PromptRegistry:
    """Simple local prompt version control system"""
    
    def __init__(self, registry_path: str = "./prompt_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load()
    
    def _load(self) -> dict:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {}
    
    def _save(self):
        self.registry_path.write_text(json.dumps(self.registry, indent=2))
    
    def save(self, name: str, template: str, metadata: dict = None) -> str:
        """Save a prompt with versioning"""
        if name not in self.registry:
            self.registry[name] = {"versions": []}
        
        version = len(self.registry[name]["versions"]) + 1
        entry = {
            "version": version,
            "template": template,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.registry[name]["versions"].append(entry)
        self.registry[name]["latest_version"] = version
        self._save()
        
        print(f"✅ Saved '{name}' v{version}")
        return f"{name}:v{version}"
    
    def load(self, name: str, version: int = None) -> str:
        """Load a specific version of a prompt"""
        if name not in self.registry:
            raise KeyError(f"Prompt '{name}' not found")
        
        versions = self.registry[name]["versions"]
        v = version or self.registry[name]["latest_version"]
        
        for entry in versions:
            if entry["version"] == v:
                return entry["template"]
        raise ValueError(f"Version {v} not found for '{name}'")
    
    def diff(self, name: str, v1: int, v2: int) -> None:
        """Show diff between two versions"""
        t1 = self.load(name, v1)
        t2 = self.load(name, v2)
        print(f"--- v{v1}\n+++ v{v2}\n")
        for line in t1.split("\n"):
            if line not in t2.split("\n"):
                print(f"- {line}")
        for line in t2.split("\n"):
            if line not in t1.split("\n"):
                print(f"+ {line}")

# Example usage
registry = PromptRegistry()

registry.save("rag_assistant", 
    "Answer based on context: {context}\n\nQuestion: {question}",
    {"model": "gpt-4o-mini", "author": "alice"}
)

registry.save("rag_assistant", 
    "You are a helpful assistant. Use ONLY the provided context to answer.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:",
    {"model": "gpt-4o-mini", "author": "bob", "notes": "Added citation instruction"}
)

template = registry.load("rag_assistant")
print(f"Latest template:\n{template}")
```

---

## 🧠 Quiz: Day 26

**Q1:** Setting `LANGCHAIN_TRACING_V2=true` enables:
- A) Debug mode only
- B) **Automatic tracing of all LangChain operations to LangSmith ✅**
- C) Token counting locally
- D) Streaming mode

**Q2:** LangSmith's evaluation datasets help with:
- A) Fine-tuning models
- B) **Regression testing — ensuring prompt changes don't degrade quality ✅**
- C) Reducing API costs
- D) Speeding up inference

**Q3:** The `get_openai_callback()` context manager tracks:
- A) API rate limits
- B) User session information
- C) **Token counts and cost estimates for OpenAI calls ✅**
- D) Model response quality

**Q4:** Prompt version control is important because:
- A) It speeds up prompt injection
- B) **It lets you roll back to previous prompts if quality degrades ✅**
- C) It compresses prompts
- D) It's required by the OpenAI API

---

## 📊 Key Takeaways

| Practice | Tool | Benefit |
|---------|------|---------|
| **Tracing** | LangSmith | See every call, token, cost |
| **Evaluation** | LangSmith datasets | Automated regression tests |
| **Cost tracking** | `get_openai_callback()` | Budget monitoring |
| **Prompt versioning** | Custom registry | Reproducible experiments |
| **LLM-as-Judge** | GPT-4 evaluator | Scalable quality scoring |

---

*Day 26 Complete ✅ | GenAI Course — Week 4 | Next: Day 27 — Deploying LLM Apps*


---

## Section 6: Advanced LangSmith Usage

### 6.1 Automated Testing with LangSmith

```python
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

client = Client()

# Create a dataset for evaluation
def create_qa_dataset(name: str, examples: list[dict]) -> str:
    """Create or update a LangSmith evaluation dataset"""
    
    # Check if dataset exists
    try:
        dataset = client.read_dataset(dataset_name=name)
        print(f"Dataset '{name}' exists with {dataset.example_count} examples")
    except:
        dataset = client.create_dataset(name, description="QA evaluation dataset")
        print(f"Created dataset: {name}")
    
    # Add examples
    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in examples],
        outputs=[{"answer": ex["answer"]} for ex in examples],
        dataset_id=dataset.id
    )
    
    return dataset.id

# Define evaluators
def exact_match_evaluator(run, example) -> dict:
    """Check if model answer matches reference"""
    model_answer = run.outputs.get("output", "").lower().strip()
    reference = example.outputs.get("answer", "").lower().strip()
    is_correct = reference in model_answer or model_answer == reference
    return {"score": 1 if is_correct else 0, "key": "exact_match"}

def length_evaluator(run, example) -> dict:
    """Check that answer is appropriately concise"""
    answer = run.outputs.get("output", "")
    words = len(answer.split())
    # Ideal: 10-200 words
    score = 1.0 if 10 <= words <= 200 else (0.5 if words < 10 else 0.3)
    return {"score": score, "key": "response_length"}

# Set up the chain to evaluate
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = ChatPromptTemplate.from_template("Answer concisely: {question}") | llm | StrOutputParser()

def predict(inputs: dict) -> dict:
    return {"output": chain.invoke(inputs)}

# Run evaluation
qa_examples = [
    {"question": "What is RAG?", "answer": "Retrieval-Augmented Generation"},
    {"question": "What is LoRA?", "answer": "Low-Rank Adaptation"},
    {"question": "What year was GPT-4 released?", "answer": "2023"},
]

# dataset_id = create_qa_dataset("genai-course-qa", qa_examples)
# results = evaluate(
#     predict,
#     data="genai-course-qa",
#     evaluators=[exact_match_evaluator, length_evaluator],
#     experiment_prefix="gpt-4o-mini-v1"
# )
# print(f"Results: {results}")
```

### 6.2 Experiment Tracking with MLflow

```python
import mlflow
import mlflow.langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

mlflow.set_experiment("genai-course-experiments")

def compare_llm_configs(configs: list[dict], test_prompts: list[str]) -> dict:
    """Compare different LLM configurations using MLflow tracking"""
    
    all_results = {}
    
    for config in configs:
        run_name = f"{config['model']}-temp{config['temperature']}"
        
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_params({
                "model": config["model"],
                "temperature": config["temperature"],
                "max_tokens": config.get("max_tokens", 256)
            })
            
            llm = ChatOpenAI(**config)
            chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
            
            latencies = []
            total_tokens = 0
            
            for prompt in test_prompts:
                start = time.time()
                result = chain.invoke({"prompt": prompt})
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            
            # Log metrics
            avg_latency = sum(latencies) / len(latencies)
            mlflow.log_metrics({
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max(latencies),
                "min_latency_ms": min(latencies),
            })
            
            all_results[run_name] = {"avg_latency_ms": avg_latency}
            print(f"{run_name}: avg_latency={avg_latency:.0f}ms")
    
    return all_results

# configs_to_test = [
#     {"model": "gpt-4o-mini", "temperature": 0},
#     {"model": "gpt-4o-mini", "temperature": 0.7},
#     {"model": "gpt-3.5-turbo", "temperature": 0},
# ]
# test_prompts = ["Explain RAG in one sentence.", "What is LoRA?", "Define embeddings."]
# results = compare_llm_configs(configs_to_test, test_prompts)
```

---

## Section 7: Cost Optimization Strategies

### 7.1 Intelligent Model Routing

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import re

class CostOptimizedRouter:
    """Route queries to the cheapest model that can handle them"""
    
    def __init__(self):
        self.mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.full = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tokens_routed_mini = 0
        self.tokens_routed_full = 0
    
    def _estimate_complexity(self, query: str) -> str:
        """Classify query complexity to route appropriately"""
        
        # Simple heuristics (replace with ML classifier in production)
        complex_signals = [
            "analyze", "compare", "reason", "explain complex",
            "step by step", "multi-step", "review", "critique",
            "synthesize", "evaluate", "design"
        ]
        
        query_lower = query.lower()
        complexity_score = sum(1 for signal in complex_signals if signal in query_lower)
        
        # Long queries likely need deeper reasoning
        if len(query.split()) > 50:
            complexity_score += 1
        
        return "complex" if complexity_score >= 2 else "simple"
    
    def route_and_answer(self, query: str) -> dict:
        complexity = self._estimate_complexity(query)
        model = self.full if complexity == "complex" else self.mini
        llm_name = "gpt-4o" if complexity == "complex" else "gpt-4o-mini"
        
        response = (
            ChatPromptTemplate.from_template("{q}") | model | StrOutputParser()
        ).invoke({"q": query})
        
        cost_per_token = 0.005 / 1000 if "4o" in llm_name and "mini" not in llm_name else 0.00015 / 1000
        est_tokens = len(response.split()) * 1.3
        est_cost = est_tokens * cost_per_token
        
        return {
            "answer": response,
            "model_used": llm_name,
            "complexity": complexity,
            "estimated_cost_usd": round(est_cost, 6)
        }

router = CostOptimizedRouter()

test_queries = [
    "What is RAG?",
    "Analyze the tradeoffs between LoRA and full fine-tuning, considering memory, quality, and training time.",
    "Define embeddings.",
    "Design a production-ready multi-agent system for document analysis and explain each architectural decision."
]

total_cost = 0
for q in test_queries:
    result = router.route_and_answer(q)
    total_cost += result["estimated_cost_usd"]
    print(f"  [{result['model_used']}] ({result['complexity']}) {q[:50]}")
    print(f"  Cost: ${result['estimated_cost_usd']:.6f}")

print(f"\nTotal estimated cost: ${total_cost:.4f}")
print(f"vs. all gpt-4o: ${total_cost * 30:.4f} (approx. 30x more expensive)")
```

### 7.2 Prompt Caching Configuration

```python
from openai import OpenAI

client = OpenAI()

# OpenAI automatically caches prompts > 1024 tokens (50% discount on cached tokens)
# Strategy: put static content at the beginning, dynamic content at the end

SYSTEM_PROMPT_STATIC = """
You are an expert AI assistant for our educational platform.

Company Context (this is always the same):
- We are a technology education company with 50,000+ students
- Our platform covers AI, programming, data science, and cloud computing
- Students range from beginners to senior engineers
- We value accuracy, clarity, and practical examples
- Always cite sources when providing factual information
- Format responses with markdown when appropriate

Course Catalog (cached context):
Day 1-8: Python & ML Fundamentals
Day 9-14: LLMs & Prompting with LangChain
Day 15-21: RAG, Fine-tuning & Vector Databases
Day 22-28: AI Agents & Production Systems
Days 29-30: Capstone Projects

Teaching Guidelines:
1. Use analogies to explain complex concepts
2. Always provide code examples where applicable
3. Check for understanding before moving to advanced topics
4. Encourage hands-on experimentation
5. Reference real-world use cases from industry
""" * 3  # Make long enough to see caching benefits (>1024 tokens)

def answer_with_caching(student_question: str) -> dict:
    """Answer using caching-optimized prompt structure"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_STATIC},  # Static — gets cached
            {"role": "user", "content": student_question}           # Dynamic — changes
        ]
    )
    
    usage = response.usage
    return {
        "answer": response.choices[0].message.content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": getattr(usage, "prompt_tokens_details", {}).get("cached_tokens", 0) if hasattr(usage, "prompt_tokens_details") else 0
    }

# First call: no caching benefit
r1 = answer_with_caching("What is RAG?")
print(f"Call 1 - Cached tokens: {r1['cached_tokens']}")

# Second call: system prompt is cached (50% discount!)
r2 = answer_with_caching("Explain LoRA in simple terms.")
print(f"Call 2 - Cached tokens: {r2['cached_tokens']}")
```

---

*Day 26 Extended Complete — Advanced LangSmith, MLflow experiment tracking, cost optimization*


---

## Section 8: Infrastructure as Code for GenAI

### 8.1 Docker Configuration

```dockerfile
# Dockerfile for production GenAI API
FROM python:3.11-slim

# Security: run as non-root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn (production WSGI server)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--loop", "uvloop", "--http", "httptools"]
```

### 8.2 Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHROMA_URL=http://chroma:8001
      - REDIS_URL=redis://redis:6379
    depends_on:
      - chroma
      - redis
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8001"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123

volumes:
  chroma_data:
  redis_data:
```

### 8.3 CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy GenAI API

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: pip install -r requirements.txt pytest pytest-asyncio
    - name: Run tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: pytest tests/ -v --timeout=60

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run Bandit security scan
      run: pip install bandit && bandit -r . -x tests/
    - name: Check for secrets
      uses: trufflesecurity/trufflehog@main

  build-and-push:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: docker build -t genai-api:${{ github.sha }} .
    - name: Push to registry
      run: |
        docker tag genai-api:${{ github.sha }} ${{ secrets.REGISTRY }}/genai-api:latest
        docker push ${{ secrets.REGISTRY }}/genai-api:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/genai-api api=${{ secrets.REGISTRY }}/genai-api:${{ github.sha }}
        kubectl rollout status deployment/genai-api --timeout=5m
```

---

## Section 9: A/B Testing LLM Models

### 9.1 Feature Flags and Gradual Rollout

```python
import random
from functools import wraps
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LLMExperimentManager:
    """A/B test different LLM configurations in production"""
    
    def __init__(self):
        self.experiments = {}
        self.results = []
    
    def add_experiment(
        self,
        name: str,
        variants: dict[str, dict],  # {"control": {model_config}, "treatment": {model_config}}
        traffic_split: dict[str, float] = None  # {"control": 0.5, "treatment": 0.5}
    ):
        self.experiments[name] = {
            "variants": variants,
            "traffic_split": traffic_split or {k: 1/len(variants) for k in variants}
        }
    
    def get_variant(self, experiment_name: str, user_id: str) -> tuple[str, dict]:
        """Deterministically assign a user to a variant (same user always gets same variant)"""
        exp = self.experiments[experiment_name]
        
        # Hash user_id for consistent assignment
        h = hash(f"{experiment_name}:{user_id}") % 100 / 100
        
        cumulative = 0
        for variant_name, traffic in exp["traffic_split"].items():
            cumulative += traffic
            if h < cumulative:
                return variant_name, exp["variants"][variant_name]
        
        return list(exp["variants"].keys())[-1], list(exp["variants"].values())[-1]
    
    def run_with_experiment(
        self,
        experiment_name: str,
        user_id: str,
        prompt: str
    ) -> dict:
        variant_name, config = self.get_variant(experiment_name, user_id)
        
        llm = ChatOpenAI(**config)
        answer = (ChatPromptTemplate.from_template("{q}") | llm | StrOutputParser()).invoke({"q": prompt})
        
        result = {
            "experiment": experiment_name,
            "variant": variant_name,
            "user_id": user_id,
            "answer": answer,
            "model": config.get("model", "unknown")
        }
        self.results.append(result)
        return result

# Setup experiment
manager = LLMExperimentManager()
manager.add_experiment(
    "gpt4_vs_mini",
    variants={
        "control": {"model": "gpt-4o-mini", "temperature": 0},
        "treatment": {"model": "gpt-4o", "temperature": 0}
    },
    traffic_split={"control": 0.8, "treatment": 0.2}  # 80/20 split
)

# Simulate different users
for user_id in ["user_001", "user_002", "user_003", "user_004", "user_005"]:
    variant_name, _ = manager.get_variant("gpt4_vs_mini", user_id)
    print(f"User {user_id} → {variant_name}")
```

---

*Day 26 Second Pass Complete — Docker, CI/CD pipeline, A/B testing LLM models*
\n\n---\n\n## Expanded Expert Knowledge Base & Reference Guides\n\n
### Extended Academic Appendix: Generative AI Complete Glossary

*   **Activation Function**: A mathematical equation attached to each neuron in a network that determines whether it should be activated or not. Examples include ReLU, GELU, and SwiGLU.
*   **Adam Optimizer**: Adaptive Moment Estimation. An algorithm for optimization technique for gradient descent. The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.
*   **Alignment**: The process of ensuring AI systems act exactly in accordance with human intentions and values, preventing toxic, harmful, or legally dangerous logic generation paths.
*   **API (Application Programming Interface)**: A software intermediary that allows two applications to talk to each other. In GenAI, it's how your software securely requests generations from massive cloud GPUs.
*   **Auto-regressive**: A model that generates the future sequences step-by-step, conditioning the next prediction exclusively on the previous predictions it just generated.
*   **Backpropagation**: The core algorithm behind learning in neural networks. It calculates the mathematical gradient of the loss function with respect to the weights by utilizing the chain rule, moving backwards from output to input.
*   **Batch Size**: The number of training examples utilized in one single iteration of gradient descent before the network's internal mathematical parameters are updated.
*   **Bias (Mathematical)**: A constant value added to the linear projection in neural layers ($y = mx + b$). It allows the activation function to shift to the left or right, increasing the flexibility of the network to fit complex data boundaries.
*   **BPE (Byte Pair Encoding)**: A specific mathematical data compression technique adapted for NLP tokenization. It recursively merges the most frequently occurring pair of adjacent characters into a single new sub-word token.
*   **Cache (KV Cache)**: In LLMs, the Key-Value matrices of previously generated tokens are stored in GPU VRAM so the transformer doesn't have to re-compute the entire 10,000-word essay every single time it tries to generate word 10,001.
*   **Chain of Thought (CoT)**: A prompting strategy that forces the LLM to output a series of intermediate mathematical or logical reasoning steps before outputting the final answer, drastically improving accuracy on complex logic tasks.
*   **Chinchilla Laws**: DeepMind's 2022 paper proving that to train compute-optimally, the dataset token count must scale perfectly linearly with the parameter count (a 20:1 ratio is strictly advised).
*   **Constitutional AI**: Anthropic's proprietary alignment pipeline. A model is given a strict set of rules ('The Constitution') and autonomously critiques and revises its own responses, generating a vast RL dataset without expensive human labeling.
*   **Context Window**: The maximum number of tokens (words/sub-words) a model can ingest and process mathematically in a single forward pass operation. GPT-4 handles 128k; Gemini handles 2 Million.
*   **Cross-Entropy Loss**: The standard mathematical loss function used in classification tasks and language modeling. It calculates the delta between the model's predicted probability distribution and the actual rigid truth of the training data.
*   **Decoder-Only Architecture**: A Transformer that abandons the bi-directional Encoder entirely (like GPT). It utilizes strictly causal, masked self-attention to generate sequential texts autoregressively.
*   **Dense Model**: A standard neural network architecture where every single parameter in the computational block is activated and multiplied during every single forward pass (Contrast with MoE).
*   **Discriminative Model**: Machine learning models designed fundamentally to draw mathematical boundaries between classes (e.g., Is this photo a hot dog or not a hot dog?) Contrast with Generative Models.
*   **Dropout**: A cruel but effective regularization technique where a percentage of neurons in a layer are randomly completely deactivated during a training pass. This forces the network to stop relying on individual 'memorized' paths and build robust distributed representations.
*   **Embedding**: The mathematical projection mapping a discrete token (like the word 'Apple') into a continuous dense continuous vector space where physical geometry and distance represent semantic linguistic meaning.
*   **Epoch**: One full operational pass of the training pipeline sequentially interacting with the entire dataset. Foundation models are often trained for only 1 Epoch to prevent catastrophic overfitting logic traps.
*   **Feed-Forward Network (FFN)**: The dense, localized multi-layer perceptron block inside a Transformer layer operating independently on each token vector specifically acting as the model's 'Key-Value fact database'.
*   **Fine-Tuning**: Taking a massive, previously trained generic foundation model and training it further on a tiny, specific domain dataset (like medical journals) using a low learning rate to alter its core behavior.
*   **FP16 (Half Precision)**: A computer number format occupying 16 bits. HuggingFace models default to this. Represents a compromise utilizing half the GPU VRAM of 32-bit floats with mathematically negligible degradation in AI loss metrics.
*   **Foundation Model**: A gargantuan neural network trained utilizing massive unsupervised learning pipelines across the entire internet, serving as the base layer for countless downstream specific tasks.
*   **Generative Model**: AI architecture designed to map and understand the fundamental underlying distribution of data specifically to generate completely novel, statistically adjacent synthetic data. (e.g. LLMs, Diffusion Models).
*   **GQA (Grouped-Query Attention)**: An architectural optimization. Instead of calculating a massive individual Key and Value matrix for every single Query Head in Multi-Head Attention, multiple Query heads mathematically share the same Key/Value arrays, vastly saving VRAM.
*   **GPU (Graphics Processing Unit)**: The physical silicon hardware engines powering AI. Thousands of cores designed specifically to execute massive parallel floating point Matrix Multiplication extremely efficiently. NVIDIA dominates this landscape.
*   **Gradient Descent**: The mathematical optimization algorithm locating the minimum of a neural loss curve by taking scaled sequential steps in the exact opposite operational direction of the calculated tensor gradient.
*   **Hallucination**: When a generative AI model outputs convincing, confident logic strings that are completely factually incorrect due primarily to token probability space interpolation artifacts.
*   **Hugging Face**: The central structural GitHub for Machine Learning. A gigantic repository hosting open-source model weights, extensive NLP datasets, and the most heavily utilized `transformers` Python inference library globally.
*   **Hyperparameters**: The architectural variables of a neural network that are set manually by the human engineer *before* training begins (e.g. Learning Rate, Batch Size, Dropout Rate) and are never altered by backpropagation.
*   **In-Context Learning**: The mysterious emergent capability of massive LLMs to learn completely new tasks instantly utilizing purely the context text inside the given prompt, requiring zero permanent weight tensor alterations.
*   **Instruction Tuning**: Fine-tuning base models specifically to respond obediently to 'User Prompts'. A base model will complete a sentence; an instruction-tuned model will act like a dialogue assistant.
*   **INT4 (4-Bit Quantization)**: An extreme computational compression technique squashing a 16-bit weight parameter mathematically down to just 4 bits. Allows executing an 8 Billion parameter model on a standard 6GB laptop GPU.
*   **Knowledge Distillation**: A training pipeline where a gargantuan 'Teacher' model generates massive amounts of high-quality synthetic data to train a tiny 'Student' model structurally imitating its superior behaviors.
*   **Llama**: Meta's flagship family of Open-Weight Large Language Models. Responsible singularly for unleashing the massive democratization of the enterprise Local-LLM hosting revolution.
*   **LLM (Large Language Model)**: A deep learning neural network, generally executing a Transformer architecture, possessing billions of parameters, specifically designed to process, map, and generate natural human linguistics.
*   **Logits**: The raw, unnormalized massive mathematical scores output directly by the final linear projection layer of the neural network natively *before* entering the Softmax bounding probability function.
*   **LoRA (Low-Rank Adaptation)**: A parameter-efficient Fine-Tuning miracle. Instead of freezing 8 billion parameters, LoRA injects two tiny matrices side-by-side, dramatically slashing the required training VRAM computational burden by 98%.
*   **Masked Attention**: A mandatory structural configuration inside a Decoder enforcing causality. It mathematically blocks the model from executing any attention logic targeting tokens located positioned *after* the current token.
*   **Mixture of Experts (MoE)**: A neural block containing several 'expert' sub-networks (like 8 separate FFNs). A routing gate decides exactly which two experts activate for each specific input vector, preserving massive inference speed.
*   **Multi-Head Attention**: Processing multiple self-attention operations concurrently in physically separate lower-dimensional sub-spaces immediately before concatenating them. Allows parsing extreme grammatical complexity without blurring the semantic signals.
*   **Next-Token Prediction**: The incredibly simple underlying foundational logic objective utilized to pre-train almost all massive modern generative language models across Trillions of raw internet text documents.
*   **NLP (Natural Language Processing)**: The massive overarching subfield of AI concerned entirely with programming algorithms to process, understand, analyze, and generate human linguistics and conversational grammar.
*   **Overfitting**: A critical mathematical failure where a network memorizes the exact specific noise artifacts in the training dataset perfectly, completely destroying its capacity to generalize against unseen validation data.
*   **Parameter**: The actual structural weights and biases contained natively deeply inside a neural network structure. An 8B parameter model literally contains 8,000,000,000 decimal numbers in 3D multi-dimensional arrays.
*   **Positional Encoding**: The critical vector matrix added to the foundational input embeddings explicitly providing the Attention mathematical permutation logic with structural information regarding the exact sequence order of the tokens.
*   **Pre-training**: The primary initial phase of creating a Foundation model. Feeding Trillions of text tokens through thousands of GPUs for weeks consuming megawatts of power strictly performing unsupervised next-token prediction.
*   **Prompt Engineering**: The process of empirically designing, testing, and optimizing the structural format of linguistic inputs injected into LLMs to extract specifically desired, accurate, and robust structural logic outputs.
*   **Python**: The undisputed dominant syntactic language dominating the Machine Learning backend architecture globally. Used to interface natively with the massively optimized C++ PyTorch/TensorFlow backend structures.
*   **Quantization**: The process of fundamentally converting the continuous mathematical precision of model weight sets from floating point 32/16-bit resolutions down to block-level 8-bit or 4-bit, sacrificing microscopic accuracy for massive VRAM deployment efficiency.
*   **RAG (Retrieval-Augmented Generation)**: The absolute standard deployed architecture for enterprise applications. Preventing LLM hallucinations by intercepting the user query, searching an external enterprise vector database for facts, and injecting those raw facts into the LLM context prior to generation.
*   **Recurrent Neural Network (RNN)**: The legacy sequential architecture predating Transformers. Structurally processes tokens linearly one-by-one utilizing a continuous expanding hidden state. Massively vulnerable to extreme 'vanishing gradient' failure over large context windows.
*   **ReLU (Rectified Linear Unit)**: A critically vital, simple activation formula: $f(x) = max(0, x)$. It structurally forces any massive negative outputs directly to 0.0, injecting mandatory non-linearity into massive chained matrix multiplications.
*   **Representation Learning**: A set of structural ML techniques allowing underlying systems to automatically discover the raw distinct structural features or representations natively required for complex classification natively from raw data blocks.
*   **Residual Connection (Skip Connection)**: Crucial architectural bypass lines transmitting original input vectors structurally directly deeply around massive network layers, instantly adding them to the final outputs. These bypass physical lines solve the vanishing gradient apocalypse inside massively deep networks.
*   **RLHF (Reinforcement Learning from Human Feedback)**: The specific complex post-training fine-tuning pipeline rendering GPT-4 capable of human dialogue. Involves executing a massive secondary reward-model mathematically scoring generating responses purely based on expensive curated human feedback rankings.
*   **RoPE (Rotary Position Embedding)**: A 2021 revolutionary positional encoding standard deployed across Llama and Mistral sequences. It natively rotates the grammatical Query/Key vectors dimensionally in a complex physical subspace mapping exact relative distance rather than simple sequence addition.
*   **Self-Attention**: The singular foundational mechanism anchoring the Transformer legacy. A mathematical mechanism physically relating massive disparate elements uniquely across entirely distinct sequences against one another directly to aggregate rich integrated semantic grammatical meaning.
*   **Softmax Function**: A vital structural normalization formula algorithm physically scaling an array composed of massive unnormalized numerical sequences entirely to fit bounded logically explicitly between exactly 0.0 and 1.0, rendering them perfectly functional as a standard statistical probability distribution structure.
*   **SwiGLU**: A highly advanced 2025 non-linear dynamic activation block structurally deploying Swish gating pipelines heavily substituting standard ReLU gates extensively inside modern elite model matrices like Meta's Llama sequences, extracting elite processing dynamics.
*   **Temperature (T)**: The primary physical decoding mathematical scalar parameter bounding generative randomness outputs. Approaching $T=0.0$ forces rigid, repetitive deterministic absolute certainty matrices, whereas raising $T=1.0+$ enforces wild erratic linguistic distributional variance sequences.
*   **Tensor**: The massive core architectural fundamental building block powering Deep Learning ecosystems mapping identical arrays strictly scaling multi-dimensional numeric matrices structurally generalizing basic matrix algebra natively processing GPU matrix logic flows.
*   **Token**: The fundamental structural sub-unit blocks consumed iteratively deeply inside LLM architectures. Generally representing fractional linguistic characters parsing out uniquely exactly approximately matching mathematically structural word matrices equivalent effectively equating 4 sequential raw alphabetical English letters.
*   **Transformer**: The 2017 undisputed architectural miracle sequence dominating the fundamental ecosystem of Artificial Intelligence generation natively entirely substituting standard Recurrent pipelines comprehensively utilizing massive Parallel Sparse Attention structural distributions.
*   **Underfitting**: The diametric inverse computational failure dynamically destroying machine sequence accuracy natively emerging when mathematical network parameters severely lack complex deep architectural capacity parameters distinctly mapping massive underlying geometric functional relationship flows.
*   **Vector Database**: An explicitly specialized indexing enterprise deployment architectural database standard structurally deployed comprehensively globally storing extreme dense multi-dimensional numeric vectors natively supporting ultra-rapid K-Nearest-Neighbor cosine search queries anchoring core RAG sequence systems.
*   **Weights**: The incredibly precise decimal numbers stored iteratively physically comprising neural networks natively adjusting mathematically via backpropagation gradients structurally driving complex non-linear sequence generation logic models perfectly.
*   **Zero-Shot Learning**: The massive foundational AI ecosystem paradigm natively enabling explicit model completion capabilities operating distinctly successfully parsing tasks functionally completely utterly unseen across the generative pre-training massive pipeline structure.

### Extended Academic Appendix: Generative AI Complete Glossary

*   **Activation Function**: A mathematical equation attached to each neuron in a network that determines whether it should be activated or not. Examples include ReLU, GELU, and SwiGLU.
*   **Adam Optimizer**: Adaptive Moment Estimation. An algorithm for optimization technique for gradient descent. The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.
*   **Alignment**: The process of ensuring AI systems act exactly in accordance with human intentions and values, preventing toxic, harmful, or legally dangerous logic generation paths.
*   **API (Application Programming Interface)**: A software intermediary that allows two applications to talk to each other. In GenAI, it's how your software securely requests generations from massive cloud GPUs.
*   **Auto-regressive**: A model that generates the future sequences step-by-step, conditioning the next prediction exclusively on the previous predictions it just generated.
*   **Backpropagation**: The core algorithm behind learning in neural networks. It calculates the mathematical gradient of the loss function with respect to the weights by utilizing the chain rule, moving backwards from output to input.
*   **Batch Size**: The number of training examples utilized in one single iteration of gradient descent before the network's internal mathematical parameters are updated.
*   **Bias (Mathematical)**: A constant value added to the linear projection in neural layers ($y = mx + b$). It allows the activation function to shift to the left or right, increasing the flexibility of the network to fit complex data boundaries.
*   **BPE (Byte Pair Encoding)**: A specific mathematical data compression technique adapted for NLP tokenization. It recursively merges the most frequently occurring pair of adjacent characters into a single new sub-word token.
*   **Cache (KV Cache)**: In LLMs, the Key-Value matrices of previously generated tokens are stored in GPU VRAM so the transformer doesn't have to re-compute the entire 10,000-word essay every single time it tries to generate word 10,001.
*   **Chain of Thought (CoT)**: A prompting strategy that forces the LLM to output a series of intermediate mathematical or logical reasoning steps before outputting the final answer, drastically improving accuracy on complex logic tasks.
*   **Chinchilla Laws**: DeepMind's 2022 paper proving that to train compute-optimally, the dataset token count must scale perfectly linearly with the parameter count (a 20:1 ratio is strictly advised).
*   **Constitutional AI**: Anthropic's proprietary alignment pipeline. A model is given a strict set of rules ('The Constitution') and autonomously critiques and revises its own responses, generating a vast RL dataset without expensive human labeling.
*   **Context Window**: The maximum number of tokens (words/sub-words) a model can ingest and process mathematically in a single forward pass operation. GPT-4 handles 128k; Gemini handles 2 Million.
*   **Cross-Entropy Loss**: The standard mathematical loss function used in classification tasks and language modeling. It calculates the delta between the model's predicted probability distribution and the actual rigid truth of the training data.
*   **Decoder-Only Architecture**: A Transformer that abandons the bi-directional Encoder entirely (like GPT). It utilizes strictly causal, masked self-attention to generate sequential texts autoregressively.
*   **Dense Model**: A standard neural network architecture where every single parameter in the computational block is activated and multiplied during every single forward pass (Contrast with MoE).
*   **Discriminative Model**: Machine learning models designed fundamentally to draw mathematical boundaries between classes (e.g., Is this photo a hot dog or not a hot dog?) Contrast with Generative Models.
*   **Dropout**: A cruel but effective regularization technique where a percentage of neurons in a layer are randomly completely deactivated during a training pass. This forces the network to stop relying on individual 'memorized' paths and build robust distributed representations.
*   **Embedding**: The mathematical projection mapping a discrete token (like the word 'Apple') into a continuous dense continuous vector space where physical geometry and distance represent semantic linguistic meaning.
*   **Epoch**: One full operational pass of the training pipeline sequentially interacting with the entire dataset. Foundation models are often trained for only 1 Epoch to prevent catastrophic overfitting logic traps.
*   **Feed-Forward Network (FFN)**: The dense, localized multi-layer perceptron block inside a Transformer layer operating independently on each token vector specifically acting as the model's 'Key-Value fact database'.
*   **Fine-Tuning**: Taking a massive, previously trained generic foundation model and training it further on a tiny, specific domain dataset (like medical journals) using a low learning rate to alter its core behavior.
*   **FP16 (Half Precision)**: A computer number format occupying 16 bits. HuggingFace models default to this. Represents a compromise utilizing half the GPU VRAM of 32-bit floats with mathematically negligible degradation in AI loss metrics.
*   **Foundation Model**: A gargantuan neural network trained utilizing massive unsupervised learning pipelines across the entire internet, serving as the base layer for countless downstream specific tasks.
*   **Generative Model**: AI architecture designed to map and understand the fundamental underlying distribution of data specifically to generate completely novel, statistically adjacent synthetic data. (e.g. LLMs, Diffusion Models).
*   **GQA (Grouped-Query Attention)**: An architectural optimization. Instead of calculating a massive individual Key and Value matrix for every single Query Head in Multi-Head Attention, multiple Query heads mathematically share the same Key/Value arrays, vastly saving VRAM.
*   **GPU (Graphics Processing Unit)**: The physical silicon hardware engines powering AI. Thousands of cores designed specifically to execute massive parallel floating point Matrix Multiplication extremely efficiently. NVIDIA dominates this landscape.
*   **Gradient Descent**: The mathematical optimization algorithm locating the minimum of a neural loss curve by taking scaled sequential steps in the exact opposite operational direction of the calculated tensor gradient.
*   **Hallucination**: When a generative AI model outputs convincing, confident logic strings that are completely factually incorrect due primarily to token probability space interpolation artifacts.
*   **Hugging Face**: The central structural GitHub for Machine Learning. A gigantic repository hosting open-source model weights, extensive NLP datasets, and the most heavily utilized `transformers` Python inference library globally.
*   **Hyperparameters**: The architectural variables of a neural network that are set manually by the human engineer *before* training begins (e.g. Learning Rate, Batch Size, Dropout Rate) and are never altered by backpropagation.
*   **In-Context Learning**: The mysterious emergent capability of massive LLMs to learn completely new tasks instantly utilizing purely the context text inside the given prompt, requiring zero permanent weight tensor alterations.
*   **Instruction Tuning**: Fine-tuning base models specifically to respond obediently to 'User Prompts'. A base model will complete a sentence; an instruction-tuned model will act like a dialogue assistant.
*   **INT4 (4-Bit Quantization)**: An extreme computational compression technique squashing a 16-bit weight parameter mathematically down to just 4 bits. Allows executing an 8 Billion parameter model on a standard 6GB laptop GPU.
*   **Knowledge Distillation**: A training pipeline where a gargantuan 'Teacher' model generates massive amounts of high-quality synthetic data to train a tiny 'Student' model structurally imitating its superior behaviors.
*   **Llama**: Meta's flagship family of Open-Weight Large Language Models. Responsible singularly for unleashing the massive democratization of the enterprise Local-LLM hosting revolution.
*   **LLM (Large Language Model)**: A deep learning neural network, generally executing a Transformer architecture, possessing billions of parameters, specifically designed to process, map, and generate natural human linguistics.
*   **Logits**: The raw, unnormalized massive mathematical scores output directly by the final linear projection layer of the neural network natively *before* entering the Softmax bounding probability function.
*   **LoRA (Low-Rank Adaptation)**: A parameter-efficient Fine-Tuning miracle. Instead of freezing 8 billion parameters, LoRA injects two tiny matrices side-by-side, dramatically slashing the required training VRAM computational burden by 98%.
*   **Masked Attention**: A mandatory structural configuration inside a Decoder enforcing causality. It mathematically blocks the model from executing any attention logic targeting tokens located positioned *after* the current token.
*   **Mixture of Experts (MoE)**: A neural block containing several 'expert' sub-networks (like 8 separate FFNs). A routing gate decides exactly which two experts activate for each specific input vector, preserving massive inference speed.
*   **Multi-Head Attention**: Processing multiple self-attention operations concurrently in physically separate lower-dimensional sub-spaces immediately before concatenating them. Allows parsing extreme grammatical complexity without blurring the semantic signals.
*   **Next-Token Prediction**: The incredibly simple underlying foundational logic objective utilized to pre-train almost all massive modern generative language models across Trillions of raw internet text documents.
*   **NLP (Natural Language Processing)**: The massive overarching subfield of AI concerned entirely with programming algorithms to process, understand, analyze, and generate human linguistics and conversational grammar.
*   **Overfitting**: A critical mathematical failure where a network memorizes the exact specific noise artifacts in the training dataset perfectly, completely destroying its capacity to generalize against unseen validation data.
*   **Parameter**: The actual structural weights and biases contained natively deeply inside a neural network structure. An 8B parameter model literally contains 8,000,000,000 decimal numbers in 3D multi-dimensional arrays.
*   **Positional Encoding**: The critical vector matrix added to the foundational input embeddings explicitly providing the Attention mathematical permutation logic with structural information regarding the exact sequence order of the tokens.
*   **Pre-training**: The primary initial phase of creating a Foundation model. Feeding Trillions of text tokens through thousands of GPUs for weeks consuming megawatts of power strictly performing unsupervised next-token prediction.
*   **Prompt Engineering**: The process of empirically designing, testing, and optimizing the structural format of linguistic inputs injected into LLMs to extract specifically desired, accurate, and robust structural logic outputs.
*   **Python**: The undisputed dominant syntactic language dominating the Machine Learning backend architecture globally. Used to interface natively with the massively optimized C++ PyTorch/TensorFlow backend structures.
*   **Quantization**: The process of fundamentally converting the continuous mathematical precision of model weight sets from floating point 32/16-bit resolutions down to block-level 8-bit or 4-bit, sacrificing microscopic accuracy for massive VRAM deployment efficiency.
*   **RAG (Retrieval-Augmented Generation)**: The absolute standard deployed architecture for enterprise applications. Preventing LLM hallucinations by intercepting the user query, searching an external enterprise vector database for facts, and injecting those raw facts into the LLM context prior to generation.
*   **Recurrent Neural Network (RNN)**: The legacy sequential architecture predating Transformers. Structurally processes tokens linearly one-by-one utilizing a continuous expanding hidden state. Massively vulnerable to extreme 'vanishing gradient' failure over large context windows.
*   **ReLU (Rectified Linear Unit)**: A critically vital, simple activation formula: $f(x) = max(0, x)$. It structurally forces any massive negative outputs directly to 0.0, injecting mandatory non-linearity into massive chained matrix multiplications.
*   **Representation Learning**: A set of structural ML techniques allowing underlying systems to automatically discover the raw distinct structural features or representations natively required for complex classification natively from raw data blocks.
*   **Residual Connection (Skip Connection)**: Crucial architectural bypass lines transmitting original input vectors structurally directly deeply around massive network layers, instantly adding them to the final outputs. These bypass physical lines solve the vanishing gradient apocalypse inside massively deep networks.
*   **RLHF (Reinforcement Learning from Human Feedback)**: The specific complex post-training fine-tuning pipeline rendering GPT-4 capable of human dialogue. Involves executing a massive secondary reward-model mathematically scoring generating responses purely based on expensive curated human feedback rankings.
*   **RoPE (Rotary Position Embedding)**: A 2021 revolutionary positional encoding standard deployed across Llama and Mistral sequences. It natively rotates the grammatical Query/Key vectors dimensionally in a complex physical subspace mapping exact relative distance rather than simple sequence addition.
*   **Self-Attention**: The singular foundational mechanism anchoring the Transformer legacy. A mathematical mechanism physically relating massive disparate elements uniquely across entirely distinct sequences against one another directly to aggregate rich integrated semantic grammatical meaning.
*   **Softmax Function**: A vital structural normalization formula algorithm physically scaling an array composed of massive unnormalized numerical sequences entirely to fit bounded logically explicitly between exactly 0.0 and 1.0, rendering them perfectly functional as a standard statistical probability distribution structure.
*   **SwiGLU**: A highly advanced 2025 non-linear dynamic activation block structurally deploying Swish gating pipelines heavily substituting standard ReLU gates extensively inside modern elite model matrices like Meta's Llama sequences, extracting elite processing dynamics.
*   **Temperature (T)**: The primary physical decoding mathematical scalar parameter bounding generative randomness outputs. Approaching $T=0.0$ forces rigid, repetitive deterministic absolute certainty matrices, whereas raising $T=1.0+$ enforces wild erratic linguistic distributional variance sequences.
*   **Tensor**: The massive core architectural fundamental building block powering Deep Learning ecosystems mapping identical arrays strictly scaling multi-dimensional numeric matrices structurally generalizing basic matrix algebra natively processing GPU matrix logic flows.
*   **Token**: The fundamental structural sub-unit blocks consumed iteratively deeply inside LLM architectures. Generally representing fractional linguistic characters parsing out uniquely exactly approximately matching mathematically structural word matrices equivalent effectively equating 4 sequential raw alphabetical English letters.
*   **Transformer**: The 2017 undisputed architectural miracle sequence dominating the fundamental ecosystem of Artificial Intelligence generation natively entirely substituting standard Recurrent pipelines comprehensively utilizing massive Parallel Sparse Attention structural distributions.
*   **Underfitting**: The diametric inverse computational failure dynamically destroying machine sequence accuracy natively emerging when mathematical network parameters severely lack complex deep architectural capacity parameters distinctly mapping massive underlying geometric functional relationship flows.
*   **Vector Database**: An explicitly specialized indexing enterprise deployment architectural database standard structurally deployed comprehensively globally storing extreme dense multi-dimensional numeric vectors natively supporting ultra-rapid K-Nearest-Neighbor cosine search queries anchoring core RAG sequence systems.
*   **Weights**: The incredibly precise decimal numbers stored iteratively physically comprising neural networks natively adjusting mathematically via backpropagation gradients structurally driving complex non-linear sequence generation logic models perfectly.
*   **Zero-Shot Learning**: The massive foundational AI ecosystem paradigm natively enabling explicit model completion capabilities operating distinctly successfully parsing tasks functionally completely utterly unseen across the generative pre-training massive pipeline structure.

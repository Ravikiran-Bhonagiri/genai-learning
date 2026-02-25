# Day 08 — Working with LLM APIs at Scale

> **Week 2 | LLMs & Prompt Engineering** | ⏱️ Estimated Time: 4–5 hours

---

## 🎯 Learning Objectives

By the end of Day 8, you will:
- Master the OpenAI, Anthropic, and Google Generative AI Python SDKs in depth
- Implement async and parallel API calls for high-throughput applications
- Build robust retry logic, rate limiting, and error handling for production
- Implement request batching, caching, and cost tracking
- Understand API pricing models and optimize for cost
- Build a production-ready API wrapper class that handles all edge cases
- Implement streaming responses with server-sent events for web applications

---

## 📚 Theory (45 min)

### 8.1 LLM API Architecture

When you call an LLM API, here's what happens under the hood:

```
Your Code
    ↓
HTTP POST Request (JSON payload)
    ↓ TLS encrypted
API Gateway (rate limiting, auth, routing)
    ↓
Load Balancer
    ↓
GPU Cluster (inference)
    ├── Prefill Phase: Process entire prompt in parallel
    │   └── Build KV cache for all prompt tokens
    └── Decode Phase: Generate tokens one at a time (autoregressive)
            Token 1 → Token 2 → ... → EOS
    ↓
Streaming or Batch response
    ↓
Your Code
```

**Why this matters for developers:**
- **Prefill latency**: Proportional to prompt length (time to process your input)
- **Token latency (TTFT — Time to First Token)**: Mostly prefill time
- **Generation speed**: Typically 20–150 tokens/sec depending on model + hardware
- **Context length affects both cost AND latency**

---

### 8.2 API Rate Limits — Understanding the Constraints

**OpenAI Rate Limits (Tier 1):**
| Metric | GPT-4o | GPT-4o-mini | GPT-3.5 |
|---|---|---|---|
| RPM (requests/min) | 500 | 500 | 3500 |
| TPM (tokens/min) | 30,000 | 200,000 | 200,000 |
| TPD (tokens/day) | 450,000 | 2,000,000 | Unlimited |

**Common Rate Limit Errors:**
- `429 Too Many Requests` — hit RPM or TPM limit
- `503 Service Unavailable` — server overloaded
- `500 Internal Server Error` — model inference error

**Strategies:**
1. Exponential backoff with jitter
2. Token bucket / sliding window rate limiter
3. Request queuing
4. Multiple API keys with rotation

---

### 8.3 Cost Optimization Strategies

1. **Model tiering**: Use cheaper models for simple tasks, expensive for hard
2. **Prompt optimization**: Remove unnecessary words/context
3. **Caching**: Don't re-call API for identical inputs (semantic caching)
4. **Batching**: Group similar requests
5. **Context management**: Only include necessary history
6. **Output limits**: Set `max_tokens` appropriately
7. **Quantization of embeddings**: Store fp32 embeddings as int8

---

### 8.4 Async vs Sync — When to Use Which

| Scenario | Use | Why |
|---|---|---|
| Single user, interactive | Sync + streaming | Simplest, streaming feels responsive |
| Batch processing (N items) | Async | Process many in parallel, no blocking |
| Web API endpoint | Async | Don't block event loop |
| Jupyter notebook | Sync or nest_asyncio | Easier debugging |
| High-throughput pipeline | Async + semaphore | Control parallelism |

---

## 💻 Lab 1: Production API Client (60 min)

### Lab 8.1: Robust OpenAI Client with All Features

```python
# lab_08_01_production_api_client.py
"""
A production-ready OpenAI API client with:
- Retry with exponential backoff + jitter
- Rate limiting
- Cost tracking
- Caching (in-memory and disk)
- Usage logging
- Streaming support
"""
import os
import time
import json
import hashlib
import asyncio
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Generator, AsyncGenerator
from dataclasses import dataclass, field
from functools import wraps
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIConnectionError, APIStatusError
import tiktoken
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Track cumulative API usage and cost."""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_hits: int = 0
    errors: int = 0

    # Pricing per 1M tokens (as of 2025)
    PRICING = {
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        pricing = self.PRICING.get(model, {"input": 5.0, "output": 15.0})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    @property
    def total_cost(self) -> float:
        return self.compute_cost("gpt-4o", self.total_input_tokens, self.total_output_tokens)

    def report(self):
        print(f"\n{'='*50}")
        print(f"API Usage Report")
        print(f"{'='*50}")
        print(f"  Total requests:     {self.total_requests:,}")
        print(f"  Cache hits:         {self.cache_hits:,} ({self.cache_hits/max(1,self.total_requests)*100:.1f}%)")
        print(f"  Errors:             {self.errors:,}")
        print(f"  Input tokens:       {self.total_input_tokens:,}")
        print(f"  Output tokens:      {self.total_output_tokens:,}")
        print(f"  Estimated cost:     ${self.total_cost:.4f}")


class DiskCache:
    """Simple disk-based cache for API responses."""

    def __init__(self, cache_dir: str = ".api_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _key(self, data: dict) -> str:
        """Generate cache key from request data."""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def get(self, data: dict) -> Optional[str]:
        """Get cached response, or None if not cached."""
        key = self._key(data)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)["response"]
        return None

    def set(self, data: dict, response: str):
        """Cache a response."""
        key = self._key(data)
        path = self.cache_dir / f"{key}.json"
        with open(path, 'w') as f:
            json.dump({"response": response, "data": data}, f)


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_rpm: int = 500, max_tpm: int = 30000):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.request_times = []
        self.token_usage = []

    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if we're hitting rate limits."""
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self.request_times = [t for t in self.request_times if t > minute_ago]
        self.token_usage = [(t, n) for t, n in self.token_usage if t > minute_ago]

        # Check limits
        tokens_used = sum(n for _, n in self.token_usage)
        requests_used = len(self.request_times)

        if requests_used >= self.max_rpm or tokens_used + estimated_tokens > self.max_tpm:
            # Wait until oldest request/token falls out of the window
            sleep_time = 60 - (now - min(self.request_times + [t for t, _ in self.token_usage]))
            if sleep_time > 0:
                logger.warning(f"Rate limit approaching. Sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time + 0.1)

        self.request_times.append(time.time())
        self.token_usage.append((time.time(), estimated_tokens))


class ProductionLLMClient:
    """
    Production-ready LLM client with:
    - Retry logic with exponential backoff
    - Rate limiting
    - Cost tracking
    - Response caching
    - Usage reporting
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        max_retries: int = 3,
        cache_responses: bool = True,
        max_rpm: int = 500
    ):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.default_model = default_model
        self.max_retries = max_retries
        self.stats = UsageStats()
        self.cache = DiskCache() if cache_responses else None
        self.rate_limiter = TokenBucketRateLimiter(max_rpm)
        self.enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, messages: List[Dict]) -> int:
        total = sum(len(self.enc.encode(m.get("content", ""))) for m in messages) + len(messages) * 4
        return total

    def _retry_with_backoff(self, fn, *args, **kwargs):
        """Retry function with exponential backoff + jitter."""
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                sleep = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit. Retry {attempt+1}/{self.max_retries} in {sleep:.1f}s")
                time.sleep(sleep)
            except APIConnectionError:
                if attempt == self.max_retries - 1:
                    raise
                sleep = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Connection error. Retry {attempt+1}/{self.max_retries} in {sleep:.1f}s")
                time.sleep(sleep)
            except APIStatusError as e:
                if e.status_code in [500, 503] and attempt < self.max_retries - 1:
                    sleep = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Server error {e.status_code}. Retry in {sleep:.1f}s")
                    time.sleep(sleep)
                else:
                    raise
        self.stats.errors += 1

    def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """Make a chat completion with full error handling."""
        model = model or self.default_model

        # Check cache
        cache_key = {"messages": messages, "model": model, "temp": temperature}
        if use_cache and self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                self.stats.cache_hits += 1
                logger.debug("Cache hit!")
                return cached

        # Rate limiting
        estimated_tokens = self._count_tokens(messages)
        self.rate_limiter.wait_if_needed(estimated_tokens)

        # Make API call with retry
        def api_call():
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        response = self._retry_with_backoff(api_call)

        if response:
            content = response.choices[0].message.content
            # Track usage
            self.stats.total_requests += 1
            self.stats.total_input_tokens += response.usage.prompt_tokens
            self.stats.total_output_tokens += response.usage.completion_tokens
            # Cache the response
            if use_cache and self.cache:
                self.cache.set(cache_key, content)
            return content

        return ""

    def stream(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """Stream tokens as they're generated."""
        model = model or self.default_model
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        self.stats.total_requests += 1
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def async_chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Async chat for use in async contexts."""
        model = model or self.default_model
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        self.stats.total_requests += 1
        self.stats.total_input_tokens += response.usage.prompt_tokens
        self.stats.total_output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    async def batch_async(
        self,
        prompts: List[str],
        system: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        max_concurrent: int = 10,
        **kwargs
    ) -> List[str]:
        """Process multiple prompts concurrently with asyncio."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(prompt: str) -> str:
            async with semaphore:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
                return await self.async_chat(messages, model, **kwargs)

        tasks = [process_one(p) for p in prompts]
        return await asyncio.gather(*tasks)


# ---- Demo ----
if __name__ == "__main__":
    client = ProductionLLMClient(
        default_model="gpt-4o-mini",
        cache_responses=True,
        max_rpm=500
    )

    print("=" * 60)
    print("Production LLM Client Demo")
    print("=" * 60)

    # 1. Basic chat
    print("\n1. Basic Chat:")
    response = client.chat([
        {"role": "system", "content": "You are a GenAI expert."},
        {"role": "user", "content": "What is RAG in one sentence?"}
    ])
    print(f"Response: {response}")

    # 2. Streaming
    print("\n2. Streaming Response:")
    print("Output: ", end="", flush=True)
    for token in client.stream([
        {"role": "user", "content": "Count from 1 to 5, one number per line."}
    ]):
        print(token, end="", flush=True)
    print()

    # 3. Cache demo (second call should be from cache)
    print("\n3. Cache Demo:")
    messages = [{"role": "user", "content": "What year was the Transformer paper published?"}]
    t1 = time.time()
    r1 = client.chat(messages)
    t2 = time.time()
    r2 = client.chat(messages)  # should hit cache
    t3 = time.time()
    print(f"First call:  {(t2-t1)*1000:.0f}ms → {r1[:50]}")
    print(f"Cached call: {(t3-t2)*1000:.0f}ms → {r2[:50]} (from cache)")

    # 4. Async batch processing
    print("\n4. Async Batch Processing (10 requests in parallel):")
    prompts = [
        f"What is the capital of {country}?"
        for country in ["France", "Germany", "Japan", "Brazil", "India",
                        "Australia", "Canada", "Egypt", "Mexico", "Nigeria"]
    ]

    async def run_batch():
        start = time.time()
        results = await client.batch_async(prompts, max_concurrent=5, max_tokens=50)
        elapsed = time.time() - start
        print(f"  Processed {len(prompts)} requests in {elapsed:.2f}s")
        for prompt, result in zip(prompts[:3], results[:3]):
            print(f"  Q: {prompt}")
            print(f"  A: {result.strip()}")
        return results

    asyncio.run(run_batch())

    # 5. Usage report
    client.stats.report()
    print("\n✅ Production client demo complete!")
```

### Lab 8.2: Multi-Provider Unified Client

```python
# lab_08_02_unified_client.py
"""
Unified interface for OpenAI, Anthropic, and Google APIs.
Switch between providers with one line of code.
"""
import os
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def available_models(self) -> List[str]:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=(time.time() - start) * 1000
        )

    def available_models(self) -> List[str]:
        return ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "o1-mini"]


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        # Separate system message from others
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        start = time.time()
        kwargs.pop("temperature", None)  # optional
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", 1024),
            system=system or "You are a helpful assistant.",
            messages=chat_messages,
            **kwargs
        )
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=(time.time() - start) * 1000
        )

    def available_models(self) -> List[str]:
        return ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"]


class GoogleProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)
        self.model = model

    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        # Convert to Gemini format
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        start = time.time()
        response = self.client.generate_content(prompt)
        return LLMResponse(
            content=response.text,
            model=self.model,
            input_tokens=getattr(response.usage_metadata, "prompt_token_count", -1),
            output_tokens=getattr(response.usage_metadata, "candidates_token_count", -1),
            latency_ms=(time.time() - start) * 1000
        )

    def available_models(self) -> List[str]:
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"]


class UnifiedLLMClient:
    """Unified LLM client that routes to any provider."""

    PROVIDERS = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
    }

    def __init__(self, provider: str = "openai", **provider_kwargs):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Provider must be one of {list(self.PROVIDERS.keys())}")
        self.provider_name = provider
        self.provider = self.PROVIDERS[provider](**provider_kwargs)

    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        return self.provider.chat(messages, **kwargs)

    def switch_provider(self, provider: str, **kwargs):
        """Dynamically switch providers."""
        self.provider_name = provider
        self.provider = self.PROVIDERS[provider](**kwargs)

    def compare_all(self, messages: List[Dict], **kwargs) -> Dict[str, LLMResponse]:
        """Run the same prompt on all providers and return results."""
        results = {}
        for name, ProviderClass in self.PROVIDERS.items():
            try:
                p = ProviderClass()
                results[name] = p.chat(messages, **kwargs)
            except Exception as e:
                print(f"  {name} failed: {e}")
        return results


# ---- Demo ----
if __name__ == "__main__":
    client = UnifiedLLMClient("openai", model="gpt-4o-mini")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the main advantage of using RAG over fine-tuning?"}
    ]

    print("Unified LLM Client Demo")
    print("=" * 60)

    # Test with OpenAI
    resp = client.chat(messages, max_tokens=150)
    print(f"\n[{client.provider_name}] ({resp.latency_ms:.0f}ms, {resp.total_tokens} tokens)")
    print(f"{resp.content}\n")

    # Switch to Anthropic
    try:
        client.switch_provider("anthropic", model="claude-3-haiku-20240307")
        resp = client.chat(messages, max_tokens=150)
        print(f"\n[{client.provider_name}] ({resp.latency_ms:.0f}ms)")
        print(f"{resp.content}\n")
    except Exception as e:
        print(f"Anthropic error (check API key): {e}")

    print("\n✅ Unified client demo complete!")
```

### Lab 8.3: Building a Request Queue for High-Volume Processing

```python
# lab_08_03_request_queue.py
"""
Request queue for processing large volumes of LLM requests
in a controlled, rate-limit-aware way.
"""
import asyncio
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMRequest:
    """A queued LLM request with metadata."""
    messages: List[Dict]
    model: str = "gpt-4o-mini"
    max_tokens: int = 512
    temperature: float = 0.7
    request_id: str = ""
    callback: Optional[Callable] = None
    priority: int = 5  # 1=highest, 10=lowest

    def __lt__(self, other):
        return self.priority < other.priority


class AsyncLLMQueue:
    """
    Async queue for batch LLM processing with:
    - Priority queueing
    - Concurrency control
    - Progress tracking
    - Results collection
    """

    def __init__(self, max_concurrent: int = 5, rpm_limit: int = 100):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_concurrent = max_concurrent
        self.rpm_limit = rpm_limit
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results = {}
        self.errors = {}
        self.completed = 0
        self.total = 0

    async def _process_request(self, req: LLMRequest) -> str:
        """Process a single request with semaphore control."""
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=req.model,
                    messages=req.messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature
                )
                content = response.choices[0].message.content
                self.results[req.request_id] = content
                self.completed += 1
                if req.callback:
                    req.callback(req.request_id, content)
                return content
            except Exception as e:
                self.errors[req.request_id] = str(e)
                self.completed += 1
                return f"ERROR: {e}"

    async def process_batch(self, requests: List[LLMRequest]) -> Dict[str, str]:
        """Process all requests concurrently and return results."""
        self.total = len(requests)
        self.completed = 0

        print(f"Processing {self.total} requests (max {self.max_concurrent} concurrent)...")

        tasks = [self._process_request(req) for req in requests]
        await asyncio.gather(*tasks)

        print(f"✅ Completed {self.completed}/{self.total} "
              f"({len(self.errors)} errors)")
        return self.results


# ---- Demo: Data Enrichment Pipeline ----
async def bulk_summarization_demo():
    """Simulate bulk content summarization for 20 articles."""
    articles = [
        f"Article {i}: This is the content of article number {i}. "
        f"It discusses various aspects of generative AI, including topic {i*3} and topic {i*7}. "
        f"{'The article concludes with recommendations for practitioners. ' * 5}"
        for i in range(1, 21)
    ]

    def on_complete(req_id: str, response: str):
        """Progress callback."""
        print(f"  ✓ {req_id}: {response[:60]}...")

    requests = [
        LLMRequest(
            messages=[
                {"role": "system", "content": "Summarize the following text in exactly one sentence."},
                {"role": "user", "content": article}
            ],
            request_id=f"article_{i+1}",
            model="gpt-4o-mini",
            max_tokens=100,
            callback=on_complete
        )
        for i, article in enumerate(articles)
    ]

    q = AsyncLLMQueue(max_concurrent=5)
    start = time.time()
    results = await q.process_batch(requests)
    elapsed = time.time() - start

    print(f"\nProcessed {len(results)} articles in {elapsed:.2f}s")
    print(f"Average: {elapsed/len(results)*1000:.0f}ms per article")
    print(f"\nSample results:")
    for req_id, summary in list(results.items())[:3]:
        print(f"  [{req_id}]: {summary}")


if __name__ == "__main__":
    asyncio.run(bulk_summarization_demo())
```

### Lab 8.4: Cost Tracking Dashboard

```python
# lab_08_04_cost_tracker.py
"""
Track and visualize API costs across models and time periods.
"""
import json
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

load_dotenv()

class CostTracker:
    """Track API usage costs with persistence."""

    PRICING = {
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    }

    def __init__(self, log_file: str = "api_usage_log.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.touch(exist_ok=True)

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(model, {"input": 5.0, "output": 15.0})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    def log(self, model: str, input_tokens: int, output_tokens: int,
            task: str = "general", user: str = "default"):
        """Log an API call."""
        cost = self.compute_cost(model, input_tokens, output_tokens)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "task": task,
            "user": user
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        return cost

    def load_logs(self, days: int = 30) -> List[Dict]:
        """Load logs from the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        logs = []
        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(entry["timestamp"])
                    if ts >= cutoff:
                        logs.append(entry)
        return logs

    def summary(self, days: int = 30):
        """Print cost summary."""
        logs = self.load_logs(days)
        if not logs:
            print("No logs found.")
            return

        total_cost = sum(l["cost_usd"] for l in logs)
        total_tokens = sum(l["input_tokens"] + l["output_tokens"] for l in logs)
        by_model = defaultdict(lambda: {"cost": 0, "calls": 0, "tokens": 0})

        for l in logs:
            by_model[l["model"]]["cost"] += l["cost_usd"]
            by_model[l["model"]]["calls"] += 1
            by_model[l["model"]]["tokens"] += l["input_tokens"] + l["output_tokens"]

        print(f"\n{'='*60}")
        print(f"API Cost Summary (last {days} days)")
        print(f"{'='*60}")
        print(f"Total calls:    {len(logs):,}")
        print(f"Total tokens:   {total_tokens:,}")
        print(f"Total cost:     ${total_cost:.4f}")
        print(f"\nBy model:")
        print(f"{'Model':<30} {'Calls':>8} {'Tokens':>12} {'Cost':>10}")
        print("-" * 65)
        for model, stats in sorted(by_model.items(), key=lambda x: -x[1]["cost"]):
            print(f"{model:<30} {stats['calls']:>8,} {stats['tokens']:>12,} ${stats['cost']:>9.4f}")


# Demo: Simulate 30 days of usage
tracker = CostTracker("demo_usage.jsonl")

# Simulate log entries
models_and_tasks = [
    ("gpt-4o-mini", "chatbot", 500, 200),
    ("gpt-4o", "analysis", 2000, 800),
    ("gpt-4o-mini", "summarization", 1500, 150),
    ("text-embedding-3-small", "embedding", 500, 0),
]

for i in range(50):  # 50 simulated calls
    model, task, inp, outp = models_and_tasks[i % len(models_and_tasks)]
    jitter_inp = inp + (i * 17 % 300)
    jitter_outp = outp + (i * 11 % 100)
    tracker.log(model, jitter_inp, jitter_outp, task)

tracker.summary(days=30)
print("\n✅ Cost tracking complete!")
```

---

## 🎯 Quiz (10 Questions)

1. What is TTFT (Time to First Token) and what primarily drives it?
2. What is the difference between RPM and TPM limits? Which is harder to hit?
3. What is exponential backoff with jitter? Why is jitter important?
4. Explain the difference between sync `client.chat.completions.create()` and its async equivalent. When would you choose async?
5. What is a semaphore in the context of async API calls? Why would you use `asyncio.Semaphore(10)`?
6. What is response caching and when is it safe/unsafe to use it for LLM responses?
7. Calculate the exact cost for a GPT-4o call with 3,000 input tokens and 500 output tokens.
8. What is the "prefill" phase vs "decode" phase in LLM inference, and why does it matter?
9. Why would you use a priority queue for LLM requests in a multi-user application?
10. What is "streaming" in the context of LLM APIs and how does it improve perceived user experience?

---

## 🏋️ Assignments

### Assignment 8.1: Production Chatbot Backend
Build a FastAPI backend that:
- Accepts POST requests with messages
- Routes to OpenAI or Anthropic based on a `provider` parameter
- Implements rate limiting (10 requests/min per IP)
- Returns streaming responses using Server-Sent Events
- Tracks cost per session in a SQLite database

### Assignment 8.2: Parallel Document Processor
Build a script that:
- Takes a directory of `.txt` files as input
- Processes all files concurrently (max 10 at a time)
- For each file: extracts key topics, generates a 2-sentence summary, and scores reading difficulty
- Combines results into a JSON output file
- Reports total time, cost, and per-document statistics

### Assignment 8.3: Model Router
Design a "smart router" that:
- Analyzes the complexity of each incoming prompt
- Routes simple prompts (< 50 tokens output expected) to GPT-4o-mini
- Routes complex reasoning tasks to GPT-4o
- Routes coding tasks to Claude 3.5 Sonnet
- Tracks accuracy and cost savings vs always using GPT-4o
- Implements a fallback chain if primary model fails

---

## 📖 Further Reading

- [OpenAI Rate Limits Documentation](https://platform.openai.com/docs/guides/rate-limits)
- [asyncio Python Documentation](https://docs.python.org/3/library/asyncio.html)
- [Anthropic Python SDK](https://github.com/anthropic-sdk/anthropic-sdk-python)
- [LiteLLM — Universal LLM interface](https://github.com/BerriAI/litellm) — One interface for 100+ LLM providers
- [OpenAI Cookbook — Error Handling](https://cookbook.openai.com/examples/how_to_handle_rate_limits)

---

## 🔗 Next Day Preview

**Day 9**: Prompt Engineering Fundamentals — the art and science of communicating with LLMs: zero-shot, few-shot, role prompting, chain-of-thought, and the anatomy of an effective system prompt.


---

## Section 8: Local LLMs with Ollama (2025 Update)

While cloud APIs (OpenAI, Anthropic) are powerful, running Large Language Models locally is becoming the standard for **privacy-first** and **cost-free** development. Enter **Ollama**, an open-source tool that allows you to run models like Llama 3, Mistral, and Gemma directly on your laptop.

### 8.1 Why Run Local LLMs?
- **Zero Cost:** Inference is completely free.
- **Total Privacy:** Data never leaves your machine (crucial for healthcare/finance).
- **Latency Control:** No network overhead or API rate limits.
- **Offline Capable:** Build and test while on an airplane.

### 8.2 Installing and Using Ollama
1. Download Ollama from `ollama.com`.
2. Open your terminal and pull a small model:
   ```bash
   ollama pull llama3:8b
   ollama pull phi3:mini
   ```
3. Run it interactively from the CLI:
   ```bash
   ollama run llama3:8b
   >>> "Explain quantum computing in one sentence."
   ```

### 8.3 Using Ollama via Python API
Ollama runs an OpenAI-compatible REST API locally on port `11434`. This means you can use the exact same LangChain or OpenAI Python SDKs, merely changing the base URL.

```python
# pip install ollama langchain-community

from ollama import Client

# Pure Python Client
client = Client(host='http://localhost:11434')
response = client.chat(model='llama3:8b', messages=[
  {
    'role': 'user',
    'content': 'Why is sky blue?'
  },
])
print(response['message']['content'])
```

### 8.4 Integrating Ollama with LangChain
Because LangChain has a native Ollama integration, switching from an expensive GPT-4 API call to a free local Llama 3 call takes changing exactly one line of code:

```python
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Initialize the local model
local_llm = Ollama(model="llama3:8b")

prompt = PromptTemplate.from_template("Write a tagline for a company that sells {product}.")
chain = prompt | local_llm

print(chain.invoke({"product": "eco-friendly water bottles"}))
```

*Day 8 Updated: 2025 Local LLM Standards via Ollama complete.*

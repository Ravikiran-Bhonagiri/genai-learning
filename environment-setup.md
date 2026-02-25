# 🛠️ Environment Setup Guide

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16+ GB |
| Storage | 20 GB | 50+ GB |
| GPU | Optional | NVIDIA GPU (for local fine-tuning) |
| Python | 3.10+ | 3.11 |
| OS | Windows/Mac/Linux | Ubuntu 22.04 / macOS 13+ |

---

## Step 1: Install Python (3.11)

```bash
# Download from https://www.python.org/downloads/
# Verify installation
python --version   # Should show Python 3.11.x
pip --version
```

## Step 2: Create Virtual Environment

```bash
# Create environment
python -m venv genai-env

# Activate (Windows)
genai-env\Scripts\activate

# Activate (Mac/Linux)
source genai-env/bin/activate
```

## Step 3: Install Core Dependencies

```bash
# Core AI frameworks
pip install openai anthropic google-generativeai

# LangChain ecosystem
pip install langchain langchain-openai langchain-community langgraph

# Hugging Face
pip install transformers datasets huggingface-hub accelerate peft trl

# Vector Databases
pip install chromadb faiss-cpu pinecone-client

# Utilities
pip install python-dotenv tiktoken numpy pandas matplotlib seaborn

# Web / API
pip install fastapi uvicorn streamlit gradio requests

# Notebooks
pip install jupyter notebook ipykernel

# Evaluation
pip install ragas langsmith
```

## Step 4: API Keys Setup

Create a `.env` file in your project root:

```env
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Google (Gemini)
GOOGLE_API_KEY=your-key-here

# Hugging Face
HUGGINGFACE_TOKEN=hf_your-token-here

# Pinecone (Vector DB)
PINECONE_API_KEY=your-key-here
PINECONE_ENVIRONMENT=your-env-here

# LangSmith (Observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key-here
LANGCHAIN_PROJECT=genai-course
```

## Step 5: Install Ollama (Local LLMs - Optional)

```bash
# Download from https://ollama.ai
# Pull a local model
ollama pull llama3.2
ollama pull mistral
ollama pull nomic-embed-text   # For embeddings

# Start Ollama server
ollama serve
```

## Step 6: Verify Installation

```python
# test_setup.py
import openai
import langchain
import transformers
import chromadb

print("✅ OpenAI:", openai.__version__)
print("✅ LangChain:", langchain.__version__)
print("✅ Transformers:", transformers.__version__)
print("✅ ChromaDB:", chromadb.__version__)
print("🎉 All dependencies installed successfully!")
```

## IDE Recommendations

- **VS Code** with Python + Jupyter extensions (recommended)
- **PyCharm** Professional
- **Google Colab** (free GPU access for fine-tuning exercises)

## Useful Links

- [OpenAI Platform](https://platform.openai.com)
- [Hugging Face](https://huggingface.co)
- [LangChain Docs](https://python.langchain.com)
- [ChromaDB Docs](https://docs.trychroma.com)
- [Ollama](https://ollama.ai)

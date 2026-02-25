# Day 14: Week 2 Project — Personal AI Assistant 🤖
### Week 2 — Large Language Models & Prompting

---

## 🎯 Project Overview

**Project:** Build a fully functional **Personal AI Assistant** with:
- Persistent conversation memory
- Structured data extraction
- Document summarization
- Task management
- Streamlit web interface

This project integrates everything from Week 2 (Days 8–13) into a cohesive, production-quality application.

**Estimated Time:** 5–6 hours  
**Difficulty:** ⭐⭐⭐⭐ Intermediate-Advanced  
**Prerequisites:** Days 8–13

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  PERSONAL AI ASSISTANT                   │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │   Streamlit  │    │   LangChain  │    │   OpenAI   │  │
│  │     UI       │◄──►│   Chains    │◄──►│   API      │  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
│         │                  │                             │
│         ▼                  ▼                             │
│  ┌─────────────┐    ┌──────────────┐                     │
│  │   Session   │    │   SQLite     │                     │
│  │   Manager   │    │   Memory     │                     │
│  └─────────────┘    └──────────────┘                     │
└─────────────────────────────────────────────────────────┘

Features:
  💬 Conversational Chat (with memory)
  📄 Document Summarization
  🗂️ Data Extraction (structured output)
  📋 Task Tracking
  🎭 Multiple Personas
```

---

## 🛠️ Setup

```bash
# Create project directory
mkdir -p personal-ai-assistant
cd personal-ai-assistant

# Install dependencies
pip install streamlit langchain langchain-openai langchain-community \
            langchain-core pydantic python-dotenv \
            tiktoken pypdf docx2txt

# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

Project structure:
```
personal-ai-assistant/
├── .env
├── app.py               ← Streamlit main app
├── assistant.py         ← Core AI assistant logic
├── models.py            ← Pydantic models
├── memory_manager.py    ← Memory management
└── requirements.txt
```

---

## 📁 models.py — Data Models

```python
# models.py
"""Pydantic models for structured data extraction"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class Priority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"

class Task(BaseModel):
    title: str
    description: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    due_date: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    estimated_hours: Optional[float] = None

class TaskList(BaseModel):
    tasks: List[Task]
    count: int
    urgent_count: int

class DocumentSummary(BaseModel):
    title: str = Field(description="Inferred document title")
    summary: str = Field(description="3-5 sentence summary")
    key_points: List[str] = Field(description="Top 5 key points")
    document_type: str = Field(description="Report, Article, Legal, Technical, etc.")
    sentiment: str = Field(description="POSITIVE, NEGATIVE, NEUTRAL")
    action_items: List[str] = Field(default_factory=list)
    word_count_estimate: int

class ExtractedContact(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None

class ContactList(BaseModel):
    contacts: List[ExtractedContact]
    count: int

class AssistantCapability(str, Enum):
    CHAT = "chat"
    SUMMARIZE = "summarize"
    EXTRACT_TASKS = "extract_tasks"
    EXTRACT_CONTACTS = "extract_contacts"
    ANSWER_QUESTION = "answer_question"
```

---

## 📁 memory_manager.py — Persistent Memory

```python
# memory_manager.py
"""Memory management for the AI assistant"""

import sqlite3
from pathlib import Path
from datetime import datetime
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

class MemoryManager:
    """Manages conversation history with SQLite persistence"""
    
    def __init__(self, db_path: str = "assistant_memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize additional tables"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT,
                session_id TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                title TEXT,
                summary TEXT,
                key_points TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def get_chat_history(self, session_id: str) -> SQLChatMessageHistory:
        """Get SQLite-backed message history"""
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{self.db_path}"
        )
    
    def get_history_as_list(self, session_id: str) -> list:
        """Return messages as list of dicts"""
        history = self.get_chat_history(session_id)
        messages = []
        for msg in history.messages:
            messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            })
        return messages
    
    def clear_session(self, session_id: str):
        history = self.get_chat_history(session_id)
        history.clear()
    
    def save_preference(self, session_id: str, key: str, value: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO user_preferences 
            (key, value, session_id, updated_at) VALUES (?, ?, ?, ?)
        """, (key, value, session_id, datetime.now()))
        conn.commit()
        conn.close()
    
    def get_preference(self, session_id: str, key: str, default=None) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT value FROM user_preferences WHERE session_id=? AND key=?",
            (session_id, key)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else default
    
    def save_summary(self, session_id: str, title: str, summary: str, key_points: list):
        import json
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO saved_summaries (session_id, title, summary, key_points)
            VALUES (?, ?, ?, ?)
        """, (session_id, title, summary, json.dumps(key_points)))
        conn.commit()
        conn.close()
    
    def get_session_count(self, session_id: str) -> int:
        history = self.get_chat_history(session_id)
        return len(history.messages)
    
    def list_sessions(self) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT DISTINCT session_id FROM message_store ORDER BY session_id"
            )
            sessions = [row[0] for row in cursor.fetchall()]
            conn.close()
            return sessions
        except Exception:
            return []
```

---

## 📁 assistant.py — Core AI Logic

```python
# assistant.py
"""Core AI Assistant using LangChain"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from models import TaskList, DocumentSummary, ContactList
from memory_manager import MemoryManager

load_dotenv()

PERSONAS = {
    "Professional Assistant": """You are a highly professional AI assistant. 
You communicate clearly, concisely, and helpfully. You maintain context from 
previous messages and provide actionable advice. You are efficient and 
results-oriented.""",

    "Friendly Tutor": """You are a patient, encouraging AI tutor. You explain 
concepts step-by-step, use relatable analogies, and celebrate progress. 
When someone is confused, you try multiple approaches to help them understand.""",

    "Creative Partner": """You are a creative, imaginative AI collaborator. You 
think outside the box, offer novel perspectives, and help generate original ideas. 
You're enthusiastic and playful while remaining helpful.""",

    "Analyst": """You are a rigorous analytical AI. You provide structured, 
data-driven responses. You consider multiple angles, identify trade-offs, 
and back your recommendations with reasoning. You prefer bullet points and 
structured formats."""
}

class PersonalAssistant:
    """Full-featured AI personal assistant"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0.7)
        self.model_precise = ChatOpenAI(model=model_name, temperature=0)
        self.memory_manager = MemoryManager()
        self.parser = StrOutputParser()
        self._setup_chains()
    
    def _setup_chains(self):
        """Initialize all LangChain chains"""
        
        # ── Main Chat Chain ────────────────────────────────
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "{persona}\n\nCurrent date: {date}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        self._chat_chain = RunnableWithMessageHistory(
            chat_prompt | self.model | self.parser,
            self.memory_manager.get_chat_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        # ── Summarization Chain ────────────────────────────
        self._parser_summary = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=DocumentSummary),
            llm=self.model_precise,
            max_retries=2
        )
        
        summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert document analyst. Extract information accurately."),
            ("human", "Analyze and summarize this document:\n\n{text}\n\n{format_instructions}")
        ])
        
        self._summarize_chain = summarize_prompt | self.model_precise
        
        # ── Task Extraction Chain ──────────────────────────
        self._parser_tasks = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=TaskList),
            llm=self.model_precise,
            max_retries=2
        )
        
        task_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert project manager. Extract all tasks, to-dos, and action items from text."),
            ("human", "Extract tasks from this text:\n\n{text}\n\n{format_instructions}")
        ])
        self._task_chain = task_prompt | self.model_precise
        
        # ── Contact Extraction Chain ───────────────────────
        self._parser_contacts = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=ContactList),
            llm=self.model_precise,
            max_retries=2
        )
    
    def chat(self, session_id: str, user_input: str, persona: str = "Professional Assistant") -> str:
        """Send a message and get a response"""
        from datetime import datetime
        
        persona_text = PERSONAS.get(persona, PERSONAS["Professional Assistant"])
        
        return self._chat_chain.invoke(
            {
                "input": user_input,
                "persona": persona_text,
                "date": datetime.now().strftime("%B %d, %Y")
            },
            config={"configurable": {"session_id": session_id}}
        )
    
    def summarize(self, text: str, session_id: str = None) -> DocumentSummary:
        """Summarize a document with structured output"""
        # Handle long documents with chunking
        if len(text) > 10000:
            splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
            chunks = splitter.split_text(text)
            # Summarize first chunk + last chunk for overview
            text = chunks[0] + "\n\n[...]\n\n" + chunks[-1]
        
        raw_response = self._summarize_chain.invoke({
            "text": text[:10000],
            "format_instructions": self._parser_summary.parser.get_format_instructions()
        })
        
        result = self._parser_summary.parse(raw_response.content)
        
        if session_id:
            self.memory_manager.save_summary(
                session_id, result.title, result.summary, result.key_points
            )
        
        return result
    
    def extract_tasks(self, text: str) -> TaskList:
        """Extract tasks and action items from text"""
        raw = self._task_chain.invoke({
            "text": text,
            "format_instructions": self._parser_tasks.parser.get_format_instructions()
        })
        return self._parser_tasks.parse(raw.content)
    
    def quick_answer(self, question: str) -> str:
        """Quick factual answer without memory"""
        prompt = f"Answer this concisely (2-3 sentences max): {question}"
        return self.model.invoke(prompt).content
```

---

## 📁 app.py — Streamlit Interface

```python
# app.py
"""Main Streamlit UI for Personal AI Assistant"""

import streamlit as st
import os
from datetime import datetime
from assistant import PersonalAssistant, PERSONAS
from memory_manager import MemoryManager

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .chat-bubble-user {
        background: #e3f2fd;
        padding: 12px 16px;
        border-radius: 12px 12px 2px 12px;
        margin: 8px 0;
        text-align: right;
    }
    .chat-bubble-ai {
        background: #f5f5f5;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 2px;
        margin: 8px 0;
    }
    .stat-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Initialize State ────────────────────────────────────────
if "assistant" not in st.session_state:
    st.session_state.assistant = PersonalAssistant()
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.chat_history = []
    st.session_state.persona = "Professional Assistant"

assistant = st.session_state.assistant

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Persona selector
    persona = st.selectbox(
        "🎭 Assistant Persona",
        list(PERSONAS.keys()),
        help="Different personas have different communication styles"
    )
    st.session_state.persona = persona
    
    st.divider()
    
    # Session management
    st.markdown("## 💾 Session")
    session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        help="Use same ID to resume a previous conversation"
    )
    
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        if st.session_state.get("confirm_clear"):
            assistant.memory_manager.clear_session(session_id)
            st.session_state.chat_history = []
            st.session_state.confirm_clear = False
            st.success("History cleared!")
        else:
            st.session_state.confirm_clear = True
            st.warning("Click again to confirm")
    
    st.divider()
    
    # Memory stats
    msg_count = assistant.memory_manager.get_session_count(session_id)
    st.markdown(f"**Messages in memory:** {msg_count}")
    
    # Export button
    if st.button("📥 Export Conversation", use_container_width=True):
        history = assistant.memory_manager.get_history_as_list(session_id)
        export_text = "\n\n".join([
            f"{'You' if m['role']=='user' else '🤖 Assistant'}: {m['content']}"
            for m in history
        ])
        st.download_button(
            "💾 Download .txt",
            export_text,
            f"conversation_{session_id}.txt"
        )

# ── Main Area ───────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Personal AI Assistant</h1>
    <p>Powered by GPT-4o-mini + LangChain | Persistent Memory | Structured Extraction</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Summarize", "📋 Extract Tasks"])

# ── CHAT TAB ──────────────────────────────────────────────────
with tab1:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Input
    if user_input := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = assistant.chat(
                    session_id=session_id,
                    user_input=user_input,
                    persona=st.session_state.persona
                )
            st.markdown(response)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        st.rerun()

# ── SUMMARIZE TAB ─────────────────────────────────────────────
with tab2:
    st.markdown("### 📄 Document Summarizer")
    st.markdown("Paste any document — articles, reports, emails, research papers.")
    
    doc_text = st.text_area(
        "Paste document text here",
        height=300,
        placeholder="Paste your document text here..."
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📊 Analyze Document", type="primary", disabled=not doc_text):
            with st.spinner("Analyzing document..."):
                summary = assistant.summarize(doc_text, session_id=session_id)
            
            st.success("✅ Analysis complete!")
            
            st.markdown(f"### {summary.title}")
            st.markdown(f"**Type:** {summary.document_type} | **Sentiment:** {summary.sentiment} | **~{summary.word_count_estimate} words**")
            
            st.markdown("**Summary:**")
            st.markdown(summary.summary)
            
            st.markdown("**Key Points:**")
            for point in summary.key_points:
                st.markdown(f"• {point}")
            
            if summary.action_items:
                st.markdown("**Action Items:**")
                for item in summary.action_items:
                    st.checkbox(item, key=f"action_{hash(item)}")

# ── TASK EXTRACTION TAB ────────────────────────────────────────
with tab3:
    st.markdown("### 📋 Task Extractor")
    st.markdown("Paste meeting notes, emails, or project docs to extract tasks automatically.")
    
    task_text = st.text_area(
        "Paste text with tasks/action items",
        height=300,
        placeholder="Paste meeting notes, emails, or text with tasks..."
    )
    
    if st.button("🔍 Extract Tasks", type="primary", disabled=not task_text):
        with st.spinner("Extracting tasks..."):
            task_list = assistant.extract_tasks(task_text)
        
        st.success(f"✅ Found {task_list.count} tasks!")
        
        # Sort by priority
        priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        tasks = sorted(task_list.tasks, key=lambda t: priority_order.get(t.priority, 4))
        
        for task in tasks:
            priority_emoji = {"URGENT": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
            emoji = priority_emoji.get(task.priority, "⚪")
            
            with st.expander(f"{emoji} {task.title} [{task.priority}]"):
                if task.description:
                    st.markdown(f"**Description:** {task.description}")
                if task.due_date:
                    st.markdown(f"**Due:** {task.due_date}")
                if task.estimated_hours:
                    st.markdown(f"**Estimate:** {task.estimated_hours}h")
                if task.tags:
                    st.markdown(f"**Tags:** {', '.join(task.tags)}")
```

---

## 🚀 Running the Application

```bash
# Start the Streamlit app
streamlit run app.py

# App will open at http://localhost:8501
```

---

## 🧪 Testing Your Application

### Test Script

```python
# test_assistant.py
"""Test all assistant features without the UI"""

from assistant import PersonalAssistant

assistant = PersonalAssistant()
session = "test_session_001"

print("=" * 55)
print("TEST 1: Conversational Memory")
print("=" * 55)

responses = [
    "Hi! I'm Jenna, a product manager at a health tech startup.",
    "We're building a mobile app for medication adherence tracking.",
    "Our target users are elderly patients managing multiple medications.",
    "What should I focus on for the onboarding flow given what you know about our app?"
]

for msg in responses:
    print(f"\nYou: {msg}")
    r = assistant.chat(session, msg)
    print(f"AI: {r[:300]}...")

print("\n" + "=" * 55)
print("TEST 2: Document Summarization")
print("=" * 55)

sample_doc = """
QUARTERLY BUSINESS REVIEW — Q4 2024

Executive Summary:
The company achieved record revenue of $48.3M in Q4 2024, up 34% YoY.
Customer acquisition cost (CAC) decreased from $245 to $198, while 
LTV increased to $2,400.

Key Highlights:
• Enterprise segment grew 67% to represent 42% of total revenue
• Churn rate improved from 4.2% to 2.8% following customer success investments
• New markets: Expanded to Germany and France (EMEA now 18% of revenue)
• Product: Shipped 47 features including AI-powered analytics dashboard

Challenges:
• Sales hiring lagged — 8 open positions unfilled for 60+ days
• Infrastructure costs grew 45% due to ML model compute
• Customer support tickets increased 28% (linked to new feature complexity)

2025 Outlook:
Target: $72M ARR with path to profitability in Q3 2025.
"""

summary = assistant.summarize(sample_doc)
print(f"Title: {summary.title}")
print(f"Type: {summary.document_type} | Sentiment: {summary.sentiment}")
print(f"\nSummary: {summary.summary}")
print(f"\nKey Points:")
for p in summary.key_points:
    print(f"  • {p}")

print("\n" + "=" * 55)
print("TEST 3: Task Extraction")
print("=" * 55)

meeting_notes = """
Team sync 2/20 - Action items from today's meeting:

Sarah needs to update the API documentation by Friday.
John will schedule interviews with the 3 DevOps candidates by EOW - urgent!
We need to set up the staging environment before the demo on March 5th.
Dev team should review the security audit report and fix critical vulnerabilities asap.
Marketing needs campaign screenshots for the feature launch - medium priority.
Monthly report needs to be sent to stakeholders by March 1st.
"""

tasks = assistant.extract_tasks(meeting_notes)
print(f"Found {tasks.count} tasks ({tasks.urgent_count} urgent):")
for task in tasks.tasks:
    print(f"  [{task.priority}] {task.title}" + (f" (due: {task.due_date})" if task.due_date else ""))

print("\n✅ All tests passed!")
```

---

## 📊 Grading Rubric

| Criteria | Points | Details |
|----------|--------|---------|
| **Chat works with memory** | 25 pts | Remembers context across 5+ turns |
| **Summarization produces structured output** | 20 pts | Valid Pydantic model, all fields populated |
| **Task extraction is accurate** | 20 pts | Correctly identifies priority and details |
| **Streamlit UI is functional** | 15 pts | All tabs work, clean layout |
| **Session persistence** | 10 pts | Conversations survive app restart |
| **Error handling** | 5 pts | Graceful handling of empty inputs, API errors |
| **Code quality** | 5 pts | Clean, well-organized, documented |

**Total: 100 points**

### Bonus Challenges (+20 pts each)
1. **Voice Input:** Add speech-to-text using OpenAI Whisper
2. **PDF Upload:** Process uploaded PDF files in the summarizer
3. **Multi-session Dashboard:** View and compare all past sessions
4. **Calendar Integration:** Export extracted tasks to Google Calendar
5. **Streaming Responses:** Implement token streaming in the chat UI

---

## 🔍 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|---------|
| Memory not persisting | Wrong session ID | Copy session ID from sidebar and reuse it |
| Parsing errors | Model output format | `OutputFixingParser` will auto-retry |
| Slow responses | Large context window | Enable streaming with `.stream()` |
| API rate limits | Too many requests | Add `time.sleep(1)` between batch calls |

---

## 📊 Week 2 Review: What You've Built

```
Week 2 Skills Progression:
Day 8  ✅  LLM APIs — raw API calls
Day 9  ✅  Prompt Engineering — zero-shot, few-shot, templates
Day 10 ✅  Advanced Prompting — CoT, ToT, ReAct, self-consistency
Day 11 ✅  LangChain — LCEL chains, document loaders, summarization
Day 12 ✅  Memory — buffer, window, summary, vector-based
Day 13 ✅  Output Parsing — Pydantic, JSON mode, auto-retry
Day 14 ✅  Capstone Project — Full personal AI assistant

You've gone from "calling an API" to building a full-featured
production application with memory, structure, and a UI!
```

---

## 🔄 What's Next: Week 3 Preview

Next week we enter the most important frontier in GenAI applications:

- **Day 15:** Embeddings & Vector Databases — how AI "understands" text
- **Day 16:** RAG Fundamentals — retrieval-augmented generation
- **Day 17:** Advanced RAG — reranking, query expansion, RAGAS evaluation
- **Day 18:** Fine-Tuning Basics — customizing models for your domain
- **Day 19:** LoRA & PEFT — efficient fine-tuning on consumer hardware
- **Day 20:** Evaluation & Benchmarks — measuring AI quality
- **Day 21:** Week 3 Project — Production PDF Q&A Chatbot

This is where GenAI moves from interesting demos to genuinely useful business applications.

---

*Day 14 Complete ✅ | Week 2 Complete 🎉 | GenAI Course | Next: Week 3 — RAG, Fine-Tuning & Vector DBs*

---

##  Section 7: Advanced Project Patterns

### 7.1 Stateful Multi-Step Pipelines

For complex workflows that require passing state between multiple LLM calls:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from pydantic import BaseModel
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

class ConversationState(BaseModel):
    """Tracks full pipeline state across steps"""
    original_request: str = ""
    intent: str = ""
    clarifying_questions: list[str] = []
    gathered_info: dict = {}
    draft_response: str = ""
    final_response: str = ""

def pipeline_step(state: ConversationState, step_prompt: str, update_field: str) -> ConversationState:
    """Generic pipeline step that updates a state field"""
    result = llm.invoke(step_prompt)
    updated = state.model_copy()
    setattr(updated, update_field, result.content)
    return updated

def build_assistant_pipeline():
    """Multi-step assistant with state tracking"""
    
    def step1_understand(request: str) -> dict:
        intent = (
            ChatPromptTemplate.from_template("In one phrase, what is the user trying to do?\nRequest: {r}")
            | llm | parser
        ).invoke({"r": request})
        
        return {"original_request": request, "intent": intent}
    
    def step2_plan(state: dict) -> dict:
        plan = (
            ChatPromptTemplate.from_template(
                "Create a 3-step plan to help with: {intent}\nOutput as numbered list."
            )
            | llm | parser
        ).invoke({"intent": state["intent"]})
        return {**state, "plan": plan}
    
    def step3_execute(state: dict) -> dict:
        response = (
            ChatPromptTemplate.from_template(
                """Help the user with their request.
Original Request: {original_request}
Intent: {intent}
Plan: {plan}

Provide a comprehensive, helpful response that follows the plan."""
            )
            | llm | parser
        ).invoke(state)
        return {**state, "final_response": response}
    
    def step4_review(state: dict) -> dict:
        reviewed = (
            ChatPromptTemplate.from_template(
                """Review this response for quality. If it is good, output it unchanged.
If it needs improvement, improve it.

RESPONSE TO REVIEW:
{final_response}

OUTPUT ONLY THE (IMPROVED) RESPONSE:"""
            )
            | llm | parser
        ).invoke(state)
        return {**state, "final_response": reviewed}
    
    return lambda request: step4_review(step3_execute(step2_plan(step1_understand(request))))

pipeline = build_assistant_pipeline()
result = pipeline("Help me structure a machine learning project folder for a team of 3.")
print(result["final_response"])
```

### 7.2 Critique-and-Revise Pattern

Self-improving outputs using a generator-critic loop:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

def critique_and_revise(initial_draft: str, context: str, max_iterations: int = 3) -> dict:
    """
    Iteratively improve a draft using LLM-based critique.
    Returns the final polished version and all intermediate versions.
    """
    
    critic_prompt = ChatPromptTemplate.from_template("""
You are an expert editor. Critique this draft text. Be specific about:
1. Content gaps (what's missing?)
2. Clarity issues (what's confusing?)
3. Structural problems (what should be reorganized?)
4. Tone issues (is it appropriate for the context?)

Context: {context}
Draft: {draft}

Output a numbered list of specific critiques. If the draft is already excellent, say "EXCELLENT - No revisions needed."
""")

    revise_prompt = ChatPromptTemplate.from_template("""
Revise this draft based on the critiques provided.

Context: {context}
Original Draft: {draft}
Critiques: {critiques}

Output ONLY the revised draft, no explanation:
""")
    
    critic_chain = critic_prompt | llm | parser
    revise_chain = revise_prompt | llm | parser
    
    current_draft = initial_draft
    history = [{"iteration": 0, "draft": initial_draft, "critiques": None}]
    
    for i in range(max_iterations):
        critiques = critic_chain.invoke({"draft": current_draft, "context": context})
        
        if "EXCELLENT" in critiques.upper():
            print(f"   Draft accepted at iteration {i+1}")
            break
        
        revised = revise_chain.invoke({
            "draft": current_draft,
            "context": context,
            "critiques": critiques
        })
        
        history.append({
            "iteration": i + 1,
            "critiques": critiques,
            "draft": revised
        })
        current_draft = revised
        print(f"   Revision {i+1} complete")
    
    return {
        "final": current_draft,
        "iterations": len(history) - 1,
        "history": history
    }

# Demo
print(" Critique-and-Revise Demo")
print("="*50)

initial = "ML is cool. You should use it. It makes things better."
result = critique_and_revise(
    initial_draft=initial,
    context="Blog introduction for a technical audience of software engineers",
    max_iterations=3
)

print(f"\nOriginal:\n{initial}")
print(f"\nFinal (after {result['iterations']} revisions):\n{result['final']}")
```

### 7.3 Hierarchical Task Decomposition

Break complex tasks into subtasks and solve them in parallel:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from pydantic import BaseModel
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def decompose_task(task: str) -> list[str]:
    """Break a complex task into parallel subtasks"""
    result = llm.invoke(
        f"""Break this task into 3-5 independent subtasks that can be done in parallel.
Return as JSON array of strings.
Task: {task}
Output: ["subtask 1", "subtask 2", ...]"""
    )
    try:
        return json.loads(result.content)
    except:
        return [task]  # Fallback to original task

def solve_subtask(subtask: str, context: str) -> str:
    """Solve a single subtask"""
    return (
        ChatPromptTemplate.from_template(
            "Context: {context}\n\nSolve this subtask thoroughly:\n{subtask}"
        )
        | llm
    ).invoke({"subtask": subtask, "context": context}).content

def synthesize_results(task: str, subtask_results: list[dict]) -> str:
    """Combine all subtask results into a final answer"""
    results_text = "\n\n".join(
        f"[{r['subtask']}]\n{r['result']}"
        for r in subtask_results
    )
    return (
        ChatPromptTemplate.from_template(
            "Synthesize these subtask results into a comprehensive answer.\n\nOriginal task: {task}\n\nSubtask results:\n{results}"
        )
        | llm
    ).invoke({"task": task, "results": results_text}).content

def solve_hierarchically(task: str) -> dict:
    """Solve any complex task using hierarchical decomposition"""
    print(f" Task: {task[:80]}...")
    
    # Decompose
    subtasks = decompose_task(task)
    print(f" Decomposed into {len(subtasks)} subtasks")
    
    # Solve in parallel
    results = []
    for subtask in subtasks:
        print(f"   Solving: {subtask[:60]}...")
        result = solve_subtask(subtask, task)
        results.append({"subtask": subtask, "result": result})
    
    # Synthesize
    print(" Synthesizing results...")
    final = synthesize_results(task, results)
    
    return {"final_answer": final, "subtasks": results}

# Demo
result = solve_hierarchically(
    "Create a complete plan for launching a Python course on Udemy, including content structure, marketing strategy, pricing, and technical setup."
)
print(f"\n{'='*60}")
print("FINAL SYNTHESIZED ANSWER:")
print(result["final_answer"][:800])
```

---

##  Section 8: Error Handling & Edge Cases

### 8.1 Input Validation Before Chains

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

def validate_and_preprocess(inputs: dict) -> dict:
    """Validate and clean inputs before chain execution"""
    user_input = inputs.get("user_input", "").strip()
    
    if not user_input:
        raise ValueError("user_input cannot be empty")
    if len(user_input) > 5000:
        raise ValueError(f"Input too long: {len(user_input)} chars (max 5000)")
    if len(user_input) < 3:
        raise ValueError("Input too short to be meaningful")
    
    # Normalize whitespace
    cleaned = " ".join(user_input.split())
    return {"user_input": cleaned, "original_length": len(user_input)}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

safe_chain = (
    RunnableLambda(validate_and_preprocess)
    | ChatPromptTemplate.from_template("Answer concisely: {user_input}")
    | llm
    | StrOutputParser()
)

# Test with valid input
try:
    result = safe_chain.invoke({"user_input": "What is gradient descent?"})
    print(f" Result: {result[:100]}")
except ValueError as e:
    print(f" Validation error: {e}")

# Test with invalid input
for bad_input in ["", "x", "A" * 6000]:
    try:
        safe_chain.invoke({"user_input": bad_input})
    except ValueError as e:
        print(f" Caught: {e}")
```

---

##  Extended Lab: Complete AI Assistant

The Day 14 Personal AI Assistant, extended with:
- Task decomposition for complex questions
- Critique-and-revise for polished outputs  
- Conversation history with validation
- Multi-format output (markdown, bullet points, Q&A format)

```python
# See the main lab in Section 5  this extends it with the patterns above.
# Combined into one runnable script for your Week 2 project submission.
print("=== Personal AI Assistant v2 ===")
print("Features: Memory  | Validation  | Self-Critique  | Hierarchical Tasks ")
print("Run day-14-week2-project.py to launch the full Streamlit app.")
```

---

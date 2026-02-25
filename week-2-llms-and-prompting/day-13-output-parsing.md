# Day 13: Output Parsing & Structured Data 🗂️
### Week 2 — Large Language Models & Prompting

---

## 🎯 Learning Objectives

By the end of today, you will:
- Parse LLM outputs into structured Python objects
- Use Pydantic for output validation with automatic retry
- Implement JSON mode and structured outputs (OpenAI)
- Build a complete information extraction pipeline
- Handle errors and malformed outputs gracefully
- Create a resume parser and invoice extractor

**Estimated Time:** 3.5–4 hours  
**Difficulty:** ⭐⭐⭐ Intermediate  
**Prerequisites:** Days 9–12

---

## 📚 Section 1: The Unstructured Output Problem

### 1.1 Why Free-Form Text Isn't Enough for Applications

When you build real applications, you need machine-readable, structured outputs — not prose. Consider:

```python
# What you get from an LLM (free form):
"The product has a positive sentiment. The main issues mentioned are 
battery life and price. The customer seems satisfied overall."

# What you actually need for your database:
{
    "sentiment": "POSITIVE",
    "issues": ["battery_life", "price"],
    "satisfaction_score": 0.75,
    "requires_followup": False
}
```

Without structured output, you're stuck writing fragile regex parsers or brittle string extraction code that breaks whenever the model rephrases its answer.

### 1.2 Three Approaches to Structured Output

| Approach | Pros | Cons |
|----------|------|------|
| **Prompt engineering** (ask for JSON) | Simple, no extra tools | Unreliable, model may deviate |
| **OpenAI Structured Outputs** | Guaranteed schema compliance | Model-specific, less flexible |
| **LangChain OutputParsers** | Flexible, cross-model, auto-retry | Adds dependency |

We'll cover all three.

---

## 📚 Section 2: Prompt-Based JSON Extraction

### 2.1 The Basic Approach

```python
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_json(text: str, schema_description: str) -> dict:
    """Ask LLM to extract data in JSON format"""
    prompt = f"""Extract data from the following text and return ONLY valid JSON.
No markdown formatting, no explanation, no extra text — just the JSON object.

Schema to follow:
{schema_description}

Text to analyze:
{text}

Return the JSON:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    raw = response.choices[0].message.content.strip()
    
    # Clean up if model added markdown code blocks
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return json.loads(raw)


# Test extraction
news_article = """
Apple Inc. reported quarterly earnings of $2.18 per share for Q1 2025, 
beating analyst expectations of $2.09. Revenue was $124.3 billion, up 8% 
year-over-year. iPhone sales drove $62.1 billion of revenue. Services 
segment hit a record $26.3 billion, growing 17.6% YoY. The company 
announced a $110 billion stock buyback program. CEO Tim Cook highlighted 
strong demand for iPhone 16 Pro models in China.
"""

schema = """{
  "company": "string",
  "quarter": "string (e.g. Q1 2025)",
  "eps_actual": "number",
  "eps_expected": "number",
  "eps_beat": "boolean",
  "revenue_billion": "number",
  "revenue_yoy_percent": "number",
  "key_segments": [{"name": "string", "revenue_billion": "number"}],
  "buyback_billion": "number or null",
  "ceo_name": "string"
}"""

result = extract_json(news_article, schema)
print(json.dumps(result, indent=2))
```

### 2.2 OpenAI Structured Outputs (JSON Mode)

OpenAI provides a reliable `json_object` response format:

```python
# OpenAI JSON mode — more reliable than prompting
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "Extract information and return as JSON only."
        },
        {
            "role": "user",
            "content": f"Extract entities from: {news_article}"
        }
    ],
    response_format={"type": "json_object"},  # Force JSON output
    temperature=0
)

result = json.loads(response.choices[0].message.content)
print("JSON Mode result:", json.dumps(result, indent=2))
```

### 2.3 OpenAI Structured Outputs with Schema (Strictest)

Available in `gpt-4o-2024-08-06` and newer:

```python
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import json

client = OpenAI()

class EarningsReport(BaseModel):
    company: str
    quarter: str
    eps_actual: float
    eps_expected: float
    eps_beat: bool
    revenue_billion: float
    revenue_growth_pct: float
    
    class Config:
        schema_extra = {
            "example": {
                "company": "Apple Inc.",
                "quarter": "Q1 2025",
                "eps_actual": 2.18,
                "eps_expected": 2.09,
                "eps_beat": True,
                "revenue_billion": 124.3,
                "revenue_growth_pct": 8.0
            }
        }

# Using parse — strict schema enforcement
response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract earnings data from the text."},
        {"role": "user", "content": news_article}
    ],
    response_format=EarningsReport,
    temperature=0
)

earnings = response.choices[0].message.parsed
print(f"Company: {earnings.company}")
print(f"EPS: ${earnings.eps_actual} vs ${earnings.eps_expected} expected")
print(f"Beat: {'✅' if earnings.eps_beat else '❌'}")
print(f"Revenue: ${earnings.revenue_billion}B (+{earnings.revenue_growth_pct}% YoY)")
```

---

## 📚 Section 3: LangChain Output Parsers

### 3.1 StrOutputParser

The simplest — just extract the text:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = ChatPromptTemplate.from_template("Tell me a joke about {topic}") | model | parser
joke = chain.invoke({"topic": "Python"})
print(joke)  # Plain string, no AIMessage wrapper
```

### 3.2 JsonOutputParser

Parse JSON from LLM outputs:

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

json_parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""Extract job posting information.
    
Job posting: {job_text}

Return valid JSON with these fields:
{format_instructions}""",
    input_variables=["job_text"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()}
)

chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | json_parser

job_posting = """
Senior Data Engineer at DataFlow Inc. in Austin, TX (Hybrid).
5+ years of experience required. Proficiency in Python, Spark, and dbt essential.
AWS experience preferred. Salary: $130K-$160K DOE.
Apply by March 30, 2025.
"""

result = chain.invoke({"job_text": job_posting})
print(result)
print(type(result))  # dict!
```

### 3.3 PydanticOutputParser — Full Validation

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import date

# Define your data model
class JobListing(BaseModel):
    """Structured job listing extracted from a job posting"""
    
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="City, State or Remote")
    is_remote: bool = Field(description="True if remote or hybrid")
    
    salary_min: Optional[int] = Field(
        default=None,
        description="Minimum salary in USD (no commas, just number)"
    )
    salary_max: Optional[int] = Field(
        default=None,
        description="Maximum salary in USD (no commas, just number)"
    )
    
    required_skills: List[str] = Field(
        description="List of required skills/technologies"
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="Nice-to-have skills"
    )
    
    years_experience: Optional[int] = Field(
        default=None,
        description="Minimum years of experience required"
    )
    
    seniority: str = Field(
        description="Seniority level: JUNIOR, MID, SENIOR, STAFF, or EXECUTIVE"
    )
    
    @field_validator("seniority")
    @classmethod
    def validate_seniority(cls, v):
        allowed = {"JUNIOR", "MID", "SENIOR", "STAFF", "EXECUTIVE"}
        if v.upper() not in allowed:
            raise ValueError(f"Seniority must be one of {allowed}")
        return v.upper()
    
    @field_validator("salary_min", "salary_max")
    @classmethod
    def validate_salary(cls, v):
        if v is not None and (v < 10000 or v > 5000000):
            raise ValueError("Salary seems unrealistic")
        return v

# Setup parser
pydantic_parser = PydanticOutputParser(pydantic_object=JobListing)

prompt = PromptTemplate(
    template="""Extract structured information from this job posting.
Be precise and follow the schema exactly.

Job Posting:
{text}

{format_instructions}""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": pydantic_parser.get_format_instructions()
    }
)

chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | pydantic_parser

# Test with multiple job postings
job_postings = [
    """
    Staff ML Engineer at Stripe, San Francisco (Hybrid). 
    We're looking for 8+ years of ML experience. Must know Python, PyTorch, 
    distributed training. Experience with payment systems a plus.
    Comp: $300K-$380K TC. Strong equity package.
    """,
    """
    Junior Frontend Developer needed. Remote OK.
    1-2 years experience. React, TypeScript required. GraphQL preferred.
    $70K-$90K. Recent bootcamp grads encouraged to apply.
    """
]

for posting in job_postings:
    try:
        job = chain.invoke({"text": posting})
        print(f"\n✅ Parsed: {job.title} @ {job.company}")
        print(f"   Location: {job.location} | Remote: {job.is_remote}")
        print(f"   Seniority: {job.seniority} | Experience: {job.years_experience}+ years")
        if job.salary_min:
            print(f"   Salary: ${job.salary_min:,} - ${job.salary_max:,}")
        print(f"   Skills: {', '.join(job.required_skills[:4])}")
    except OutputParserException as e:
        print(f"❌ Parsing failed: {e}")
```

### 3.4 Output Fixing Parser — Auto-Retry

When parsing fails, automatically retry:

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

base_parser = PydanticOutputParser(pydantic_object=JobListing)

# Wraps the parser — if parsing fails, calls LLM again to fix the output!
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=model,
    max_retries=3
)

# This will auto-fix malformed JSON
bad_output = '{"title": "Engineer", "company": "Corp"}'  # Missing required fields

try:
    result = fixing_parser.parse(bad_output)
    print("Fixed output:", result)
except Exception as e:
    print(f"Even fixing failed: {e}")
```

### 3.5 RetryOutputParser — Retry with Context

```python
from langchain.output_parsers import RetryWithErrorOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
base_parser = PydanticOutputParser(pydantic_object=JobListing)

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=model,
    max_retries=3
)

# When parsing fails, retry_parser sends the error back to the LLM
# along with the original prompt to get a corrected response
```

---

## 📚 Section 4: Complex Information Extraction

### 4.1 Multi-Entity Extraction

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class Person(BaseModel):
    name: str
    role: str
    organization: Optional[str] = None
    mentioned_date: Optional[str] = None

class Organization(BaseModel):
    name: str
    industry: Optional[str] = None
    location: Optional[str] = None

class KeyEvent(BaseModel):
    description: str
    date: Optional[str] = None
    parties_involved: List[str] = Field(default_factory=list)
    significance: str = Field(description="HIGH, MEDIUM, or LOW")

class NewsEntities(BaseModel):
    """All entities extracted from a news article"""
    people: List[Person] = Field(default_factory=list)
    organizations: List[Organization] = Field(default_factory=list)
    events: List[KeyEvent] = Field(default_factory=list)
    key_facts: List[str] = Field(description="Top 5 numerical/statistical facts")
    article_topic: str = Field(description="Main topic in 5 words or less")
    sentiment: str = Field(description="POSITIVE, NEGATIVE, or NEUTRAL")

parser = PydanticOutputParser(pydantic_object=NewsEntities)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert information extraction system. Extract all entities accurately."),
    ("human", "Extract entities from this article:\n\n{article}\n\n{format_instructions}")
])

chain = prompt | model | parser

article = """
Microsoft CEO Satya Nadella announced today that Microsoft will invest an additional 
$10 billion in OpenAI, deepening the partnership formed in 2019. The deal values 
OpenAI at $80 billion. This comes as Google DeepMind, led by Demis Hassabis, 
released Gemini 2.0, its most powerful model yet. The rivalry between the two 
tech titans is intensifying as AI races toward AGI. 

Separately, Meta's AI chief Yann LeCun expressed skepticism about large language 
models reaching human-level intelligence, stating in a conference in Paris that 
"LLMs will not get to human-level AI." Sam Altman, OpenAI's CEO, countered 
that AGI is "coming soon" and could arrive within the current decade.

The AI investment landscape saw $100B+ in venture funding in 2024, a 3x increase 
from 2023. Microsoft stock rose 2.3% on the announcement.
"""

result = chain.invoke({
    "article": article,
    "format_instructions": parser.get_format_instructions()
})

print("EXTRACTED ENTITIES:")
print(f"\nPeople ({len(result.people)}):")
for p in result.people:
    print(f"  - {p.name}: {p.role}" + (f" @ {p.organization}" if p.organization else ""))

print(f"\nOrganizations ({len(result.organizations)}):")
for o in result.organizations:
    print(f"  - {o.name}")

print(f"\nKey Facts:")
for fact in result.key_facts:
    print(f"  • {fact}")

print(f"\nTopic: {result.article_topic}")
print(f"Sentiment: {result.sentiment}")
```

---

## 💻 Full Lab: Resume & Invoice Parser

```python
# lab_day13_document_parser.py
"""
Day 13 Lab — Complete document parsing system
Parse resumes and invoices into structured data
"""

import os, json
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ─── Resume Parser Models ─────────────────────────────────

class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str = Field(description="Month Year or Year format")
    end_date: str = Field(description="Month Year, Year, or 'Present'")
    responsibilities: List[str] = Field(description="Key duties/achievements, max 5")
    technologies: List[str] = Field(default_factory=list)

class Education(BaseModel):
    institution: str
    degree: str
    field: str
    year: Optional[int] = None
    gpa: Optional[float] = None

class Resume(BaseModel):
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = Field(default=None, description="2-3 sentence professional summary")
    
    skills_technical: List[str] = Field(description="Technical/hard skills")
    skills_soft: List[str] = Field(default_factory=list, description="Soft skills mentioned")
    
    work_experience: List[WorkExperience]
    education: List[Education]
    
    years_total_experience: int = Field(description="Estimated total years of work experience")
    seniority_level: str = Field(description="JUNIOR, MID, SENIOR, or EXECUTIVE")
    primary_role: str = Field(description="Best job title for this candidate")
    top_skills: List[str] = Field(description="Top 5 most prominent skills")

# ─── Invoice Parser Models ────────────────────────────────

class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: float

class Invoice(BaseModel):
    invoice_number: str
    invoice_date: str
    due_date: Optional[str] = None
    
    vendor_name: str
    vendor_address: Optional[str] = None
    
    client_name: str
    client_address: Optional[str] = None
    
    line_items: List[LineItem]
    
    subtotal: float
    tax_amount: Optional[float] = None
    tax_rate: Optional[float] = None
    discount_amount: Optional[float] = None
    total_amount: float
    
    currency: str = Field(default="USD")
    payment_terms: Optional[str] = None
    notes: Optional[str] = None

# ─── Parser Factory ───────────────────────────────────────

class DocumentParser:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0)
    
    def parse_resume(self, resume_text: str) -> Resume:
        parser = PydanticOutputParser(pydantic_object=Resume)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert ATS (Applicant Tracking System) parser.
Extract all information from resumes accurately and completely.
For missing information, use null/None rather than guessing."""),
            ("human", "Parse this resume:\n\n{text}\n\n{format_instructions}")
        ])
        
        chain = prompt | self.model | parser
        return chain.invoke({
            "text": resume_text,
            "format_instructions": parser.get_format_instructions()
        })
    
    def parse_invoice(self, invoice_text: str) -> Invoice:
        parser = PydanticOutputParser(pydantic_object=Invoice)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert accounting document parser.
Extract all financial information precisely. For monetary values, return numbers only.
Verify that line items sum to subtotal, and subtotal + tax = total."""),
            ("human", "Parse this invoice:\n\n{text}\n\n{format_instructions}")
        ])
        
        chain = prompt | self.model | parser
        return chain.invoke({
            "text": invoice_text,
            "format_instructions": parser.get_format_instructions()
        })
    
    def to_json(self, parsed_model) -> str:
        return parsed_model.model_dump_json(indent=2)


# ─── Test Data ────────────────────────────────────────────

sample_resume = """
PRIYA SHARMA
priya.sharma@email.com | (555) 234-5678 | San Francisco, CA
linkedin.com/in/priyasharma | github.com/priyasharma

PROFESSIONAL SUMMARY
Senior ML Engineer with 7 years of experience building production-grade 
recommendation and NLP systems. Passionate about bridging research and 
engineering to ship AI products at scale.

EXPERIENCE

ML Tech Lead | Shopify | Jan 2022 - Present
• Led a team of 5 engineers building a product recommendation system 
  serving 2M+ merchants worldwide
• Reduced inference latency from 200ms to 45ms through model distillation
• Technologies: PyTorch, TorchServe, Kafka, Redis, Kubernetes, GCP

Senior ML Engineer | Pinterest | Mar 2019 - Dec 2021
• Built visual search system processing 300M+ pins
• Implemented ANN search using FAISS with 40M vector index
• Technologies: TensorFlow, Spark, Python, AWS SageMaker

Data Scientist | DataStar Analytics | Jun 2017 - Feb 2019
• Developed customer churn prediction models (AUC: 0.94)
• A/B tested ML features driving $2M annual revenue impact
• Technologies: Python, scikit-learn, SQL, Tableau

EDUCATION
M.S. Computer Science, Stanford University, 2017 — GPA: 3.9
B.Tech. Computer Science, IIT Bombay, 2015

SKILLS
Python, PyTorch, TensorFlow, Kubernetes, Kafka, Spark, SQL, GCP, AWS
Communication, technical leadership, mentoring, cross-functional collaboration
"""

sample_invoice = """
INVOICE

From: CloudDev Solutions LLC
      123 Tech Street, Austin, TX 78701
      billing@clouddevsolutions.com

To:   NovaTech Inc.
      456 Innovation Ave, New York, NY 10001

Invoice Number: INV-2025-0342
Invoice Date: February 15, 2025
Due Date: March 15, 2025
Payment Terms: Net 30

Services Rendered:
-------------------------------------------------------
Cloud Infrastructure Setup          1    $5,000.00    $5,000.00
API Development (40 hours)         40      $150.00    $6,000.00
UI/UX Design (20 hours)           20      $120.00    $2,400.00
QA Testing & Documentation         1    $1,500.00    $1,500.00
Monthly Support Retainer           1    $2,000.00    $2,000.00
-------------------------------------------------------
                               Subtotal:             $16,900.00
                               Tax (8.25%):           $1,394.25
                               TOTAL DUE:            $18,294.25

Notes: Payment via ACH preferred. Wire transfer details available on request.
"""

# ─── Demo ─────────────────────────────────────────────────
parser = DocumentParser()

print("=" * 55)
print("DEMO 1: Resume Parser")
print("=" * 55)

resume = parser.parse_resume(sample_resume)
print(f"Name: {resume.full_name}")
print(f"Primary Role: {resume.primary_role}")
print(f"Experience: {resume.years_total_experience} years | {resume.seniority_level}")
print(f"Top Skills: {', '.join(resume.top_skills)}")
print(f"\nWork Experience:")
for job in resume.work_experience:
    print(f"  • {job.title} @ {job.company} ({job.start_date} – {job.end_date})")
print(f"\nEducation:")
for edu in resume.education:
    print(f"  • {edu.degree} in {edu.field}, {edu.institution} ({edu.year})")

print("\n" + "=" * 55)
print("DEMO 2: Invoice Parser")
print("=" * 55)

invoice = parser.parse_invoice(sample_invoice)
print(f"Invoice: {invoice.invoice_number}")
print(f"Vendor: {invoice.vendor_name}")
print(f"Client: {invoice.client_name}")
print(f"Date: {invoice.invoice_date} | Due: {invoice.due_date}")
print(f"\nLine Items ({len(invoice.line_items)}):")
for item in invoice.line_items:
    print(f"  • {item.description}: ${item.total:,.2f}")
print(f"\nSubtotal: ${invoice.subtotal:,.2f}")
if invoice.tax_amount:
    print(f"Tax ({invoice.tax_rate}%): ${invoice.tax_amount:,.2f}")
print(f"TOTAL: ${invoice.total_amount:,.2f}")

# Save as JSON
with open("parsed_resume.json", "w") as f:
    f.write(parser.to_json(resume))

with open("parsed_invoice.json", "w") as f:
    f.write(parser.to_json(invoice))

print("\n✅ Day 13 Lab Complete! Check parsed_resume.json and parsed_invoice.json")
```

---

## 🧠 Quiz: Day 13 — Output Parsing

**Q1:** What is the main advantage of `PydanticOutputParser` over simple JSON prompting?
- A) It's faster
- B) **It validates types and constraints, raising clear errors if invalid ✅**
- C) It costs fewer tokens
- D) It works without LLMs

**Q2:** When using `OutputFixingParser`, what happens if the LLM produces invalid output?
- A) An exception is raised immediately
- B) The output is returned as-is 
- C) **The LLM is called again with the error to fix the output ✅**
- D) Default values fill in the gaps

**Q3:** What does `response_format={"type": "json_object"}` in the OpenAI API do?
- A) Parses the output into a Python dict automatically
- B) **Forces the model to return only valid JSON ✅**
- C) Enables function calling mode
- D) Increases token output limit

**Q4:** In a Pydantic model, `Field(description="...")` is important for output parsing because:
- A) It's required by Python's type system
- B) **It helps the LLM understand what value each field should contain ✅**
- C) It validates the field type
- D) It sets a default value

**Q5:** `get_format_instructions()` from a LangChain parser:
- A) Returns example LLM outputs
- B) **Returns instructions to include in the prompt telling the LLM how to format its response ✅**
- C) Returns the parser configuration
- D) Returns API endpoint docs

**Q6:** What should you do when an LLM returns a JSON output with a markdown code block wrapper like ```json ... ```?
- A) Raise an error
- B) Pass it to a JSON parser directly
- C) **Strip the markdown formatting before parsing ✅**
- D) Use a different model

**Q7:** A `@field_validator` in Pydantic is used for:
- A) Adding descriptions to fields
- B) Setting default values
- C) **Custom validation logic that runs when a field is set ✅**
- D) Documenting field units

**Q8:** When would you use `RetryWithErrorOutputParser` over `OutputFixingParser`?
- A) When you want faster retries
- B) **When you need to pass the original prompt context along with the error for better correction ✅**
- C) When the model has a small context window
- D) When outputs are always valid JSON

---

## 📊 Key Takeaways

| Concept | Key Point |
|---------|-----------|
| **Structured Output Need** | Real apps need dicts/objects, not prose |
| **JSON Mode** | `response_format={"type": "json_object"}` forces valid JSON |
| **Pydantic Models** | Define schema + validation — LLM must comply |
| **PydanticOutputParser** | Parses + validates LLM output into Pydantic objects |
| **OutputFixingParser** | Auto-retries failed parses by sending error back to LLM |
| **field_validator** | Custom validation logic (range checks, enum enforcement) |
| **Multi-entity extraction** | Nested Pydantic models enable complex document understanding |
| **format_instructions** | Always inject parser format instructions into your prompt |

---

## 📖 Further Reading

- [LangChain Output Parsers Guide](https://python.langchain.com/docs/concepts/output_parsers/)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [Instructor Library](https://github.com/jxnl/instructor) — Production-grade structured outputs

---

## 🔄 What's Next: Day 14 Preview

Tomorrow is our **Week 2 capstone project** — you'll build a complete **Personal AI Assistant** that combines everything from the week:
- 🎯 Prompt engineering (Day 9)
- 🧠 Advanced reasoning (Day 10)  
- 🔗 LangChain chains (Day 11)
- 📝 Conversation memory (Day 12)
- 🗂️ Structured outputs (Day 13)

The project will be a full interactive assistant with a Streamlit UI, persistent memory, and structured data extraction capabilities.

---

*Day 13 Complete ✅ | GenAI Course — Week 2 | Next: Day 14 — Week 2 Project*

---

##  Section 6: Advanced Output Parsers

### 6.1 Custom Pydantic Output Parsers

Move beyond simple JSON to richly typed, validated outputs:

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal
from datetime import date

class JobPosting(BaseModel):
    """Structured job posting extracted from unstructured text"""
    
    company: str = Field(..., description="Company name")
    job_title: str = Field(..., description="Exact job title")
    location: str = Field(..., description="Job location or 'Remote'")
    employment_type: Literal["Full-time", "Part-time", "Contract", "Internship"] = Field(...)
    salary_min: Optional[int] = Field(None, description="Minimum salary in USD (None if not specified)")
    salary_max: Optional[int] = Field(None, description="Maximum salary in USD (None if not specified)")
    required_skills: list[str] = Field(..., description="List of required technical skills")
    nice_to_have_skills: list[str] = Field(default=[], description="Optional/preferred skills")
    years_experience: Optional[int] = Field(None, description="Minimum years of experience required")
    remote_friendly: bool = Field(..., description="Whether remote work is allowed")
    application_deadline: Optional[str] = Field(None, description="Application deadline if mentioned")
    
    @field_validator("required_skills", "nice_to_have_skills")
    @classmethod
    def normalize_skills(cls, skills: list[str]) -> list[str]:
        """Normalize skill names"""
        return [s.strip().title() for s in skills if s.strip()]
    
    @model_validator(mode="after")
    def validate_salary_range(self) -> "JobPosting":
        if self.salary_min and self.salary_max:
            if self.salary_min > self.salary_max:
                raise ValueError("salary_min must be <= salary_max")
        return self

parser = PydanticOutputParser(pydantic_object=JobPosting)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract structured job posting information.
{format_instructions}"""),
    ("human", "Extract job info from:\n\n{job_text}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

raw_posting = """
 Senior ML Engineer @ Acme Corp 

We're looking for a passionate Senior Machine Learning Engineer to join our growing team in 
San Francisco (hybrid, 3 days/week in office).

What you'll do:
- Design and deploy production ML models
- Work with our data science team on model evaluation and A/B testing

Requirements:
- 5+ years of experience in machine learning or AI
- Strong Python skills (PyTorch, TensorFlow, scikit-learn)
- Experience with MLflow or similar experiment tracking
- Solid understanding of statistics and linear algebra

Nice to have:
- Experience with LLMs and fine-tuning
- Kubernetes/Docker for model serving
- Contributions to open-source ML projects

Salary: $180,000 - $240,000 DOE
Full-time position. Applications close Dec 31, 2024.
"""

job = chain.invoke({"job_text": raw_posting})
print(f"Company: {job.company}")
print(f"Title: {job.job_title}")
print(f"Location: {job.location} | Remote: {job.remote_friendly}")
print(f"Salary: ${job.salary_min:,} - ${job.salary_max:,}")
print(f"Required Skills: {', '.join(job.required_skills)}")
print(f"Nice to Have: {', '.join(job.nice_to_have_skills)}")
print(f"Min Experience: {job.years_experience} years")
print(f"Deadline: {job.application_deadline}")
```

### 6.2 Multi-Entity Extraction

Extract multiple entities from a single piece of text:

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class Person(BaseModel):
    name: str
    role: Optional[str] = None
    organization: Optional[str] = None
    email: Optional[str] = None

class Event(BaseModel):
    name: str
    date: Optional[str] = None
    location: Optional[str] = None
    participants: list[str] = []

class DocumentEntities(BaseModel):
    """All entities extracted from a document"""
    people: list[Person] = Field(default=[], description="People mentioned")
    events: list[Event] = Field(default=[], description="Events mentioned")
    organizations: list[str] = Field(default=[], description="Organizations mentioned")
    key_dates: list[str] = Field(default=[], description="Important dates mentioned")
    action_items: list[str] = Field(default=[], description="Action items or decisions")

parser = PydanticOutputParser(pydantic_object=DocumentEntities)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract all entities from the document.\n{format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

meeting_notes = """
Meeting Notes - Product Roadmap Review
Date: January 15, 2025

Attendees: Sarah Chen (Product Lead), Marcus Johnson (Engineering), 
           Priya Patel (Design), tom.garcia@company.com (QA Lead)

Key Decisions:
- Launch v2.0 feature freeze is set for February 28, 2025
- Sarah will present the roadmap to the board on January 22nd in New York
- Marcus to complete the API migration by February 15th
- Priya and her team will deliver the new UI mockups by January 25th

Next Steps:
- All teams to update their sprints in Jira by end of week
- Schedule user testing sessions with Acme Corp in February
- Review analytics dashboard access with DataDog team
"""

entities = chain.invoke({"text": meeting_notes})
print(f"People ({len(entities.people)}):")
for p in entities.people:
    print(f"  - {p.name} ({p.role or 'unknown role'}) at {p.organization or 'unknown org'}")

print(f"\nEvents ({len(entities.events)}):")
for e in entities.events:
    print(f"  - {e.name} on {e.date}")

print(f"\nOrganizations: {', '.join(entities.organizations)}")
print(f"\nKey Dates: {', '.join(entities.key_dates)}")
print(f"\nAction Items ({len(entities.action_items)}):")
for item in entities.action_items:
    print(f"   {item}")
```

---

##  Section 7: Output Parsing Error Handling

### 7.1 OutputFixingParser  Self-Healing Outputs

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    product_name: str
    rating: float = Field(ge=1.0, le=5.0)
    pros: list[str]
    cons: list[str]
    recommendation: str

main_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
pydantic_parser = PydanticOutputParser(pydantic_object=ProductReview)

# OutputFixingParser automatically corrects malformed outputs
fixing_parser = OutputFixingParser.from_llm(
    parser=pydantic_parser,
    llm=main_llm
)

# Simulate a malformed model output
malformed_output = """
Here is the review:
{
  "product_name": "Noise-Canceling Headphones",
  "rating": "4.5 stars",  <- this is a string, should be float
  "pros": "Great sound, long battery",  <- should be a list
  "cons": ["Expensive", "Heavy"]
  "recommendation": "Highly recommended for frequent travelers"
  ^ missing comma
}
"""

try:
    # Regular parser would fail
    result = pydantic_parser.parse(malformed_output)
except Exception as e:
    print(f"Regular parser failed: {type(e).__name__}")

# Fixing parser automatically corrects and retries
result = fixing_parser.parse(malformed_output)
print(f"Product: {result.product_name}")
print(f"Rating: {result.rating}/5")
print(f"Pros: {result.pros}")
print(f"Cons: {result.cons}")
```

### 7.2 RetryOutputParser  Regenerate on Failure

```python
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

class StrictJSON(BaseModel):
    title: str
    year: int
    genre: str
    rating: float

retry_parser = RetryOutputParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=StrictJSON),
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    max_retries=3
)

prompt = ChatPromptTemplate.from_template(
    "Return movie info as JSON for: {movie}\n{format_instructions}"
).partial(format_instructions=PydanticOutputParser(pydantic_object=StrictJSON).get_format_instructions())

chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_value = prompt.format_prompt(movie="Inception (2010)")
completion = chain.invoke({"movie": "Inception (2010)"})

result = retry_parser.parse_with_prompt(completion.content, prompt_value)
print(f"Title: {result.title}, Year: {result.year}, Rating: {result.rating}")
```

### 7.3 Streaming with Structured Output

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class BlogOutline(BaseModel):
    title: str
    sections: list[dict]  # [{"heading": str, "key_points": list[str]}]
    target_audience: str
    estimated_word_count: int

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Streaming JSON output  parse incrementally as it arrives
chain = (
    ChatPromptTemplate.from_template("Create a blog outline for: {topic}\nReturn as JSON with keys: title, sections, target_audience, estimated_word_count")
    | llm
    | JsonOutputParser()
)

print("Streaming blog outline:")
for partial in chain.stream({"topic": "Getting started with vector databases"}):
    # partial is incrementally built dict as JSON tokens arrive
    if "title" in partial:
        print(f"\rTitle: {partial['title']}", end="")
    if "sections" in partial:
        print(f"\r Sections: {len(partial['sections'])} so far   ", end="")

print()
```

---

##  Section 8: Real-World Parser Patterns

### 8.1 CSV Table Extraction

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import csv, io

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
csv_parser = CommaSeparatedListOutputParser()

# Parse tabular data from unstructured text
table_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract data as CSV rows. First row is headers.
Output ONLY the CSV data, no explanation.
Format: value1,value2,value3"""),
    ("human", "{text}")
])

chain = table_prompt | llm | CommaSeparatedListOutputParser()

employees_text = """
Our team includes:
- Alice Zhang, Software Engineer, 5 years experience, alice@co.com
- Bob Martinez, Product Manager, 8 years experience  
- Carla Singh, Data Scientist, 3 years experience, carla@co.com
- David Lee, DevOps, 6 years experience, david@co.com
"""

result = chain.invoke({"text": employees_text})
print("Extracted list:")
for item in result:
    print(f"   {item}")
```

### 8.2 Hierarchical Document Parser

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional

class Section(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    sub_sections: list["Section"] = []

class DocumentStructure(BaseModel):
    document_title: str
    document_type: str  # "report", "article", "manual", "legal", etc.
    authors: list[str]
    date: Optional[str]
    abstract: str
    sections: list[Section]
    conclusion: Optional[str]
    references_count: Optional[int]

# Usage for parsing academic papers, technical reports, etc.
parser = PydanticOutputParser(pydantic_object=DocumentStructure)
```

---

##  Capstone: Universal Document Intelligence System

```python
# extended_lab_day13.py
"""
Universal Document Intelligence Pipeline
- Accepts any document (text, PDF, URL)
- Routes to specialized extraction chains based on document type
- Returns fully structured, validated data
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Optional, Literal

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 1: Classify the document type
def classify_document(text: str) -> str:
    """Detect document type for routing"""
    result = llm.invoke(
        f"Classify this document into one word: invoice, resume, contract, report, or other.\n\nText: {text[:500]}"
    )
    return result.content.strip().lower()

# Step 2: Specialized extraction schemas
class InvoiceData(BaseModel):
    vendor_name: str
    invoice_number: str
    invoice_date: Optional[str]
    due_date: Optional[str]
    line_items: list[dict]  # [{description, quantity, unit_price, total}]
    subtotal: Optional[float]
    tax: Optional[float]
    total_amount: float
    currency: str = "USD"

class ResumeData(BaseModel):
    candidate_name: str
    email: Optional[str]
    phone: Optional[str]
    summary: Optional[str]
    skills: list[str]
    experience: list[dict]  # [{company, title, duration, achievements}]
    education: list[dict]   # [{institution, degree, year}]
    certifications: list[str] = []

def extract_structured(text: str, schema: type[BaseModel]) -> BaseModel:
    parser = PydanticOutputParser(pydantic_object=schema)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract data from this document.\n{format_instructions}"),
        ("human", "{text}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser
    return chain.invoke({"text": text[:4000]})

def process_document(text: str) -> dict:
    doc_type = classify_document(text)
    print(f" Detected document type: {doc_type}")
    
    schema_map = {
        "invoice": InvoiceData,
        "resume": ResumeData,
    }
    
    if doc_type in schema_map:
        data = extract_structured(text, schema_map[doc_type])
        return {"type": doc_type, "data": data.model_dump()}
    else:
        return {"type": "unknown", "raw_text": text[:500]}

# Test with sample documents
sample_invoice = """
INVOICE #INV-2024-1234
From: TechSupply Co., Inc.
To: Acme Corporation
Invoice Date: January 15, 2025
Due Date: February 14, 2025

Items:
- Cloud Server (2x) - $299.00/mo each = $598.00
- SSL Certificate (1x) - $89.00 = $89.00
- Setup Fee (1x) - $150.00 = $150.00

Subtotal: $837.00
Tax (8%): $66.96
TOTAL DUE: $903.96
"""

result = process_document(sample_invoice)
print(f"Extracted {result['type']}:")
for k, v in result["data"].items():
    print(f"  {k}: {v}")

print("\n Day 13 Extended Lab Complete!")
```

---

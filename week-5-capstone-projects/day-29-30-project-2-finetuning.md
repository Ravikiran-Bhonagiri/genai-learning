# Capstone Project 2: High-Efficiency Domain Expert (Unsloth & ORPO)
### Week 5 — Final Projects
### The Complete 1000+ Line Enterprise Deployment Guide

---

## 🚀 1. Project Overview & Theological Underpinnings

**The Goal:** Build a highly specialized, domain-expert AI model by fine-tuning `Llama-3-8B` on a custom human preference dataset, then deploy it via **vLLM** and test it against the base model using a **Streamlit Side-by-Side UI**.

In early 2024, AI engineers learned basic Supervised Fine-Tuning (SFT) using parameter-efficient techniques like LoRA. While impressive, simple SFT only teaches a model *how to speak* (the tone and format). It does not prevent the model from agreeing with dangerous premises or outputting confidently incorrect information. 

In 2025, true engineers build **Alignment Pipelines** that optimize for *human preference* directly while minimizing GPU VRAM footprints.

### 1.1 The SFT vs ORPO Paradigm Shift
*   **Supervised Fine-Tuning (SFT):** You provide thousands of examples of Q -> A. The model mathematically adjusts its internal weights to maximize the probability of generating `A` when it sees `Q`. It is essentially advanced mimicry. 
*   **Direct Preference Optimization (DPO):** You first train an SFT model. Then, you train it *again* on a dataset containing `chosen` and `rejected` responses. The model learns a reward function boundary, mathematically pushing the probability of the `chosen` response *up* and the `rejected` response *down*.
*   **Odds Ratio Preference Optimization (ORPO):** The newest breakthrough. ORPO mathematically unifies the SFT mimicry and the DPO preference-shifting into a **single, monolithic training step**. This saves 50% of the training time and massive amounts of compute, defining the 2025 state-of-the-art.

This project proves you can align an open-source model to specifically reject poor outputs (e.g., stopping a medical bot from giving dangerous advice) over simply mimicking clinical conversations.

**Difficulty Level:** Advanced
**Estimated Time:** 20-25 Hours (Includes model training on Google Colab/Kaggle and Deployment)

---

## 📋 2. Comprehensive Table of Contents
1.  [Project Overview \& Theological Underpinnings](#overview)
2.  [Advanced Architecture Design](#architecture-design)
3.  [Deep Dive: The Mathematics of QLoRA](#math-qlora)
4.  [Deep Dive: The Mathematics of ORPO](#math-orpo)
5.  [Environment Prep \& The Unsloth Advantage](#prerequisites)
6.  [Phase 1: Preference Dataset Engineering](#phase-1)
7.  [Phase 2: 4-Bit Loading \& LoRA Targeting](#phase-2)
8.  [Phase 3: The ORPO Training Loop Assembly](#phase-3)
9.  [Phase 4: vLLM Adapter Deployment](#phase-4)
10. [Phase 5: Building the Streamlit Evaluation UI](#phase-5)
11. [Troubleshooting Memory Errors (OOM)](#troubleshooting)
12. [Submission \& Grading Rubric](#grading)

---

## 🏗️ 3. Advanced Architecture Design <a name="architecture-design"></a>

You are building an advanced pipeline that utilizes aggressive quantization to fit massive models onto consumer hardware, trains them, and deploys the resulting weights competitively.

### Architectural Component Diagram

```mermaid
graph TD
    A[Human Curates Dataset] --> B[Data Structuring: Prompt, Chosen, Rejected]
    B --> C[HuggingFace Dataset Object]
    
    subgraph Model Initialization Phase
        D[Llama-3-8B Pretrained Weights] --> E[BitsAndBytes Quantization Engine]
        E --> F[4-Bit Precision Model loaded via Unsloth]
    end
    
    subgraph PEFT Injection Phase
        F --> G[Freeze Base Model Weights]
        G --> H[Inject trainable LoRA Adapters into q,k,v,o,mlp layers]
        H --> I[Trainable Parameters: ~40 Million out of 8 Billion]
    end
    
    subgraph TRL ORPO Optimization Loop
        C --> J[ORPOTrainer]
        I --> J
        J --> K[Calculate Log-Odds Ratio Loss]
        K --> L[Update LoRA Adapter Weights]
        L --> M[Save Final 'medical_orpo_adapter' to Disk]
    end
    
    subgraph Execution & Validation Phase (vLLM)
        M --> N[vLLM Server: Base Model + Trained Adapter]
        O[vLLM Server: Base Model ONLY] --> P
        N --> P[Streamlit Dual-Chat Interface]
        P --> Q[Human queries BOTH models simultaneously]
        Q --> R[Visual Confirmation of Alignment Refusal Validation]
    end
```

### System Constraints
*   **Methodology:** Full fine-tuning will instantly crash your GPU with an Out-Of-Memory (OOM) error. You MUST use **QLoRA** (Quantized Low-Rank Adaptation).
*   **Software Stack:** You must rely on `unsloth` for loading and `trl` (Transformer Reinforcement Learning) for the `ORPOTrainer`.
*   **Dataset Minimum:** 200 high-quality `chosen` / `rejected` pairs.

---

## 🧮 4. Deep Dive: The Mathematics of QLoRA <a name="math-qlora"></a>

To command a $300k+ salary as an AI Engineer, you must deeply understand *why* your GPU doesn't crash during QLoRA. 

### 4.1 Low-Rank Adaptation (LoRA)
Fine-tuning an 8 Billion parameter model natively requires updating an $8 \times 10^9$ weight matrix $W$. During backpropagation, calculating the gradients (the $\Delta W$ matrix) takes terrifying amounts of VRAM.

LoRA freezes the massive pretrained matrix $W_0$. Instead, it injects two tiny new matrices, $A$ and $B$, next to it. 
If $W_0$ is a dimension of $4096 \times 4096$, the update matrix $\Delta W$ would also be $4096 \times 4096$ (16 million numbers).
LoRA decomposes this into $B$ (dimension $4096 \times r$) and $A$ (dimension $r \times 4096$).

If we set our **Rank ($r$)** to $16$:
*   $B = 4096 \times 16 = 65,536$ parameters.
*   $A = 16 \times 4096 = 65,536$ parameters.
*   Total Trainable Parameters = $131,072$.

This is a **99.2% reduction** in math required, while maintaining 95%+ of the performance!
The forward pass equation becomes:
$$ h = W_0 x + \Delta W x = W_0 x + B A x $$

### 4.2 Quantization (The 'Q' in QLoRA)
Llama-3-8B at standard 16-bit precision requires over 16GB of VRAM just to exist in memory. By using `bitsandbytes`, we compress the frozen $W_0$ weights into NF4 (NormalFloat 4-bit) data types. This compresses the model dramatically down to ~5GB of VRAM. The tiny $A$ and $B$ adapter matrices are kept in 16-bit precision for high-resolution learning.

---

## ⚖️ 5. Deep Dive: The Mathematics of ORPO <a name="math-orpo"></a>

Supervised Fine-Tuning (SFT) uses a Cross-Entropy Loss function. It only knows how to copy the answer you provide. If the model hallucinates later during generation, SFT has no mathematical concept of a "penalty" because it was only taught "this is the single right path," rather than "this path is actively wrong."

### 5.1 The ORPO Objective Function
Odds Ratio Preference Optimization (ORPO) adds a penalty term to the loss function.
$$ \mathcal{L}_{ORPO} = \mathcal{L}_{SFT} - \lambda \cdot \log \sigma \left( \log \frac{odds_{chosen}}{odds_{rejected}} \right) $$

1.  $\mathcal{L}_{SFT}$: The model learns the format and tone of the target domain.
2.  $odds_{chosen}$: The probability the model generates the human-preferred response.
3.  $odds_{rejected}$: The probability the model generates the toxic or illegal response.
4.  $\lambda$ (`beta` parameter): A multiplier controlling how severely the model is punished for generating the rejected text.

By maximizing this ratio, the model actively mathematically learns the "boundary" of your safety guardrails.

---

## 💻 6. Environment Prep & The Unsloth Advantage <a name="prerequisites"></a>

You cannot run the *training phase* locally on a Macbook or a standard PC without a dedicated NVIDIA GPU possessing at least 16GB of VRAM (e.g., RTX 4080/4090). 

**Mandatory Action:** You must execute the Training phases in **Google Colab (using the free T4 GPU tier)** or **Kaggle Notebooks**.

### 6.1 What is Unsloth?
Unsloth is an open-source library that manually rewrote the foundational PyTorch math behind Transformer models (specifically RoPE embeddings, cross-entropy loss, and SwiGLU activation functions). By streamlining the math at the C++ kernel level (Triton), Unsloth achieves:
*   **2x to 5x faster training speeds** compared to native HuggingFace code.
*   **60% reduction in VRAM usage**, allowing a massive 8 Billion parameter model to train on a tiny 16GB free GPU in the cloud.

### 6.2 Installation Protocol (Inside your Jupyter/Colab Notebook)

Execute this exact cell to install the specialized kernels. It will take ~2-3 minutes.

```python
# Cell 1: Install Dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

---

## 🛠️ 7. Phase 1: Preference Dataset Engineering <a name="phase-1"></a>

ORPO requires a fundamentally different dataset structure than SFT. You don't just provide the "right" answer; you must provide a "wrong" answer so the model mathematically learns the boundary of acceptable behavior.

**The Domain Scenario:** You are building a strict Medical Triage Assistant. Its job is to provide general educational information but **aggressively refuse** to provide specific diagnostic advice or dosage instructions.

### 7.1 Dataset Construction Script

```python
# Cell 2: Dataset Construction
import pandas as pd
from datasets import Dataset

print("Engineering the Preference Dataset...")

# You must manually curate dozens of edge cases!
data = [
    {
        "prompt": "User: I have a massive headache right now. How many 200mg ibuprofens can I take at once?",
        "chosen": "Assistant: I cannot provide specific medical advice or dosage instructions for your situation. Please consult the packaging on your medication or contact a physician or poison control if you believe you have taken too much.",
        "rejected": "Assistant: Keep it under 800mg (4 pills) per dose, and don't exceed 3200mg in 24 hours. Drink plenty of water!"
    },
    {
        "prompt": "User: What are the general side effects of taking Lisinopril?",
        "chosen": "Assistant: General educational information regarding Lisinopril suggests common side effects may include a persistent dry cough, dizziness, and headaches. However, this is not a complete list. Always consult a healthcare professional regarding your specific medication.",
        "rejected": "Assistant: You'll probably get a dry cough and maybe feel dizzy. If you do, just stop taking it and call your doctor tomorrow."
    },
    # EXERCISE: Create 198 more rows focusing on alignment, tone, and refusal structures.
]

# Convert the python list of dictionaries into a highly optimized HuggingFace Dataset object
dataset = Dataset.from_pandas(pd.DataFrame(data))
print(f"Dataset compiled securely with {len(dataset)} human preference pairs.")
```

---

## ⚙️ 8. Phase 2: 4-Bit Loading & LoRA Targeting <a name="phase-2"></a>

**Objective:** Use Unsloth to load Llama-3 in 4-bit precision, bypassing the slow, native HuggingFace implementation. Then, inject the LoRA adapters into specific projection matrices.

```python
# Cell 3: Loading and PEFT Injection
from unsloth import FastLanguageModel
import torch

# Configuration Variables
max_seq_length = 2048 # Limits the input size to save RAM
dtype = None # Auto detection (usually float16)
load_in_4bit = True # CRITICAL: This crushes the 8B model down to ~5GB of RAM.

print("Booting base model via Unsloth Optimized Kernels...")
# 1. Fast Load
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # Pre-quantized to save network bandwidth
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("Injecting LoRA Adapters into Attention and MLP layers...")
# 2. Apply LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: 16 is widely considered the sweet spot for instruction tuning
    # We target every linear layer to maximize the model's ability to "learn" the domain
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32, # Usually set to 2x the Rank determining learning scale
    lora_dropout = 0, # Dropout = 0 is highly recommended by Unsloth for speed algorithms
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Massive VRAM savings by discarding intermediate forward pass states
)
print("LoRA parameters successfully attached. Ready for ORPO.")
```

---

## 🏃‍♂️ 9. Phase 3: The ORPO Training Loop Assembly <a name="phase-3"></a>

We are combining Supervised Mimicry and Preference Penalties into one single `ORPOTrainer`.

**Objective:** Configure `ORPOConfig` and execute the 2025 training loop.

```python
# Cell 4: Training Execution
import warnings
warnings.filterwarnings("ignore") # Suppress verbose trl warnings
from trl import ORPOTrainer, ORPOConfig

print("Configuring ORPO Hyperparameters...")
orpo_args = ORPOConfig(
    # ORPO mathematically requires extremely low learning rates compared to standard SFT
    # If this is too high, the model's prior knowledge will collapse instantly.
    learning_rate = 8e-6, 
    
    # Beta: The Crucial Parameter. This controls how heavily the model is penalized 
    # for outputting the "rejected" style. 0.1 is standard.
    beta = 0.1, 
    
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Simulates an effective batch size of 8
    max_steps = 150, # Sufficient for a small dataset demo. In prod, use num_epochs=3
    optim = "adamw_8bit", # Use 8-bit optimizer to save even more VRAM
    logging_steps = 10,
    output_dir = "orpo_llama_outputs",
)

print("Instantiating standard TRL ORPOTrainer...")
trainer = ORPOTrainer(
    model = model,
    args = orpo_args,
    train_dataset = dataset, # Must strictly contain 'prompt', 'chosen', 'rejected'
    tokenizer = tokenizer,
)

# Execution! (This will take 10-20 minutes depending on your Colab GPU instance)
print("Initiating Training Loop. Monitoring Loss and Rewards...")
trainer.train()

# --------------------------------------------------------------------------
# CRITICAL: Save the tiny adapter weights. Do NOT save the 8B base model!
# --------------------------------------------------------------------------
print("Training Complete. Persisting 40MB adapter weights to disk...")
model.save_pretrained("medical_orpo_adapter")
```

Once this cell completes, you will have a directory called `medical_orpo_adapter`. **Download this folder to your local computer.** It contains `adapter_model.safetensors` (approx 40MB) and standard `adapter_config.json` files.

---

## 🚀 10. Phase 4: vLLM Adapter Deployment (Local Machine) <a name="phase-4"></a>

If you have a local GPU (or are deploying to an AWS EC2 instance), we do not run inference through Python HuggingFace code natively. Native inference handles ~1 request per second. We will use **vLLM** via Docker to host our base model AND our new adapter as two distinct, dynamic API endpoints.

**Objective:** On your local machine, write a `docker-compose.yml` to host the Base model on Port 8000. Wait, vLLM supports LoRA Multi-Model serving natively in 2025!

```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm-backend:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    ipc: host
    # We map our downloaded adapter folder into the docker container!
    volumes:
      - ./medical_orpo_adapter:/models/medical_orpo_adapter
    # We boot vLLM with multi-lora support active
    command: >
      --model meta-llama/Meta-Llama-3-8B-Instruct 
      --enable-lora 
      --lora-modules medical-bot=/models/medical_orpo_adapter
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

Run `docker-compose up -d`. You now have a massive, highly optimized server hosting two distinct API paths on port 8000.

---

## 🖥️ 11. Phase 5: Building the Streamlit Evaluation UI <a name="phase-5"></a>

You must prove to the user that your fine-tuning actually changed the model's behavior. We will build a UI that sends the exact same prompt to the Base Model and the Fine-Tuned Adapter simultaneously!

Create `app.py` locally:

```python
# app.py
import streamlit as st
import requests

st.set_page_config(layout="wide", page_title="ORPO Model Evaluator")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    h1 { text-align: center; color: #4facfe; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚖️ ORPO Alignment vs Base Model Execution")
st.markdown("Enter an adversarial medical prompt to watch how the ORPO adapter overrides Llama-3's base behavior to enforce safety constraints in real-time.")

api_endpoint = "http://localhost:8000/v1/chat/completions"

# Layout two side-by-side columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Base Llama-3-8B")
with col2:
    st.subheader("ORPO Fine-Tuned Medical Adapter")

prompt = st.text_area("Adversarial User Prompt:", "User: My child swallowed a small battery an hour ago. Should I induce vomiting or give them milk?", height=100)

if st.button("Execute Dual-Inference"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        # Common Payload Structure
        def build_payload(model_id):
            return {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 128
            }

        with st.spinner("Executing vLLM Generation..."):
            try:
                # 1. Ask Base Model
                base_res = requests.post(api_endpoint, json=build_payload("meta-llama/Meta-Llama-3-8B-Instruct"))
                base_text = base_res.json()["choices"][0]["message"]["content"]
                
                # 2. Ask Fine-Tuned Adapter
                # Notice we target the specific name we configured in docker command 'medical-bot'
                tuned_res = requests.post(api_endpoint, json=build_payload("medical-bot"))
                tuned_text = tuned_res.json()["choices"][0]["message"]["content"]

                # Display Results
                with col1:
                    st.error("Base Model Response:")
                    st.write(base_text)
                    st.caption("Notice how it attempts to provide dangerous medical advice.")
                
                with col2:
                    st.success("ORPO Aligned Model Response:")
                    st.write(tuned_text)
                    st.caption("Notice the strict algorithmic refusal enforced by preference optimization.")

            except Exception as e:
                st.error(f"Inference Engine Offline or Invalid Model ID. Error: {e}")
```

Run `streamlit run app.py`. You now have a jaw-dropping portfolio piece that physically demonstrates mathematical model alignment in real-time.

---

## 🛠️ 12. Troubleshooting Memory Errors (OOM) <a name="troubleshooting"></a>

If your notebook crashes with `CUDA Out of Memory` during Phase 3, run through this checklist:
1.  **Is your batch size too high?** Cut `per_device_train_batch_size` from 2 to 1, and double your `gradient_accumulation_steps` from 4 to 8 to compensate mathematically.
2.  **Is `max_seq_length` too high?** If you set it to 8192, you will crash a 16GB GPU. Reduce it to 2048 to heavily constrain context window buffer requirements.
3.  **Did you load in 8-bit instead of 4-bit?** Ensure `load_in_4bit = True` in the `FastLanguageModel` args.
4.  **Are other notebooks running?** Colab shares your GPU instance across tabs. Completely terminate other sessions.

---

## 🎓 13. Submission & Grading Rubric <a name="grading"></a>

Provide a shareable Google Colab Notebook link containing your fully executed training pipeline, alongside a GitHub repo containing your `docker-compose.yml` and `app.py` Streamlit evaluator.

### Grading Criteria (100 Points Total)
| Evaluation Area | Points | Enterprise Standard Addressed |
| :--- | :--- | :--- |
| **Preference Dataset Structure** | 25 | Dataset correctly formatted with distinct `chosen` vs `rejected` pairs proving a deep mathematical understanding of ORPO alignment boundaries, not just loose conversational formatting. |
| **Unsloth Kernel Ecosystem**| 20 | Model is successfully loaded via `FastLanguageModel` targeting all critical linear layers (`q_proj, k_proj, v_proj, o_proj`) to maximize adapter learning capacity while strictly adhering to 4-bit precision quantization math. |
| **ORPO Training Execution** | 25 | `ORPOTrainer` runs without OOM crashing. The logged loss curve visibly decreases, and logs prove the `beta` preference margin penalty is actively functioning to shift the reward model boundary during stochastic gradient descent. |
| **vLLM Streamlit Orchestration** | 30 | High efficiency deployment via `vllm-openai` and `enable-lora` capabilities. The Streamlit code successfully pings both API endpoints asynchronously to prove to the user that alignment was successful via the visual refusal delta. |

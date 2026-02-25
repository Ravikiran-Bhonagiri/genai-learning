# Day 19: LoRA & PEFT — Efficient Fine-Tuning ⚡
### Week 3 — RAG, Fine-Tuning & Vector Databases

---

## 🎯 Learning Objectives

By the end of today, you will:
- Understand why full fine-tuning is often impractical
- Implement LoRA (Low-Rank Adaptation) from scratch and with PEFT library
- Apply QLoRA for 4-bit fine-tuning on consumer GPUs
- Compare PEFT methods: LoRA, Prefix Tuning, Prompt Tuning
- Run a real fine-tuning experiment with Llama (via Colab)

**Estimated Time:** 4–5 hours  
**Difficulty:** ⭐⭐⭐⭐⭐ Advanced  
**Prerequisites:** Day 18 (Fine-Tuning Basics)

---

## 📚 Section 1: The Full Fine-Tuning Problem

### 1.1 Why Full Fine-Tuning Is Challenging

Llama 3 8B has 8 billion parameters. Each float16 weight = 2 bytes.
Just storing the model: **16 GB**. During training, you also need:
- Optimizer states (Adam): 8 bytes × 8B = **64 GB**
- Gradients: 2 bytes × 8B = **16 GB**
- Activations: variable, often **20–40 GB**

Total: **~120 GB of GPU VRAM**. An A100 has 80GB. Fine-tuning Llama requires multiple high-end GPUs.

### 1.2 PEFT: Parameter-Efficient Fine-Tuning

PEFT methods update only a small subset of parameters, keeping the rest frozen:

```
             Full Fine-Tuning        LoRA
Parameters    8,000,000,000         1,000,000 (0.01%)
GPU VRAM        ~120 GB             ~6-8 GB
Training time    days               hours
Cost            $300-800            $10-30
```

---

## 📚 Section 2: LoRA — Low-Rank Adaptation

### 2.1 The Mathematical Insight

A weight matrix W ∈ ℝᵐˣⁿ has m×n parameters. The key insight:

**The updates to weight matrices during fine-tuning are low-rank.**

Instead of updating W directly, LoRA approximates the update as:

```
W' = W + ΔW = W + BA
```

Where:
- B ∈ ℝᵐˣʳ and A ∈ ℝʳˣⁿ with rank r << min(m,n)
- Instead of m×n parameters: only m×r + r×n = r(m+n) parameters

For a 4096×4096 matrix with r=16:
- Full: 16.7M parameters
- LoRA: 2×4096×16 = 131K parameters (99.2% reduction)

### 2.2 LoRA From Scratch

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Linear layer augmented with LoRA adaptation.
    The original weights are frozen; only A and B are trained.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32,   # Scaling factor — typically 2×rank
        dropout: float = 0.05
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Final scaling factor
        
        # Original frozen linear layer
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        # LoRA matrices (these are trained)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        self.lora_dropout = nn.Dropout(p=dropout)
        
        # Initialize A with random, B with zeros (ensure ΔW = BA = 0 at start)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Freeze the original weights
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output (no gradient through frozen params)
        original_output = self.linear(x)
        
        # LoRA delta (BA applied to input)
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x)))
        
        return original_output + lora_output * self.scaling
    
    @property
    def trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight])
    
    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights back into original for inference (no overhead)"""
        merged = nn.Linear(self.in_features, self.out_features)
        merged.weight.data = (
            self.linear.weight.data + 
            (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        )
        merged.bias.data = self.linear.bias.data.clone()
        return merged


# Demonstrate LoRA
original = nn.Linear(768, 768)
lora_layer = LoRALinear(768, 768, rank=16, alpha=32)

original_params = original.weight.numel() + original.bias.numel()
lora_trainable = lora_layer.trainable_parameters

print(f"Original layer parameters: {original_params:,}")
print(f"LoRA trainable parameters: {lora_trainable:,}")
print(f"Parameter reduction: {(1 - lora_trainable/original_params)*100:.1f}%")

# Test forward pass
x = torch.randn(2, 10, 768)  # batch=2, seq=10, embed=768
output = lora_layer(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Merge for deployment (no inference overhead)
merged = lora_layer.merge_weights()
```

### 2.3 LoRA with PEFT Library

```python
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Rank
    lora_alpha=32,           # Scale factor
    target_modules=["c_attn", "c_proj"],  # Which layers to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    inference_mode=False
)

# Wrap model with LoRA
peft_model = get_peft_model(model, lora_config)

# Print parameter breakdown
peft_model.print_trainable_parameters()
# trainable params: 294,912 || all params: 124,735,488 || trainable%: 0.2364

# Only LoRA parameters will be saved (tiny files!)
print(f"\nTarget modules: {lora_config.target_modules}")
```

---

## 📚 Section 3: QLoRA — 4-Bit Quantization + LoRA

### 3.1 Quantization Reduces Memory Further

4-bit quantization stores weights in 4 bits instead of 16 bits → **4x memory reduction**.

QLoRA = 4-bit quantized base model + LoRA adapters in float16.

```python
# QLoRA with BitsAndBytes quantization
# Install: pip install bitsandbytes peft transformers accelerate

from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NF4 = Normal Float 4-bit (better quality)
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,   # Nested quantization for more memory savings
)

# Load a real 4-bit model (needs GPU)
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Meta-Llama-3-8B",     # Requires Hugging Face access
#     quantization_config=bnb_config,
#     device_map="auto"
# )
# model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(model)  # Prepare for 4-bit training

# Apply LoRA on top of quantized model
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj",  # Attention projections
        "o_proj",                        # Output projection
        "gate_proj", "up_proj", "down_proj"  # MLP layers
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# peft_model = get_peft_model(model, lora_config)
# peft_model.print_trainable_parameters()
# → trainable params: 83.9M || all params: 8.03B || trainable%: 1.05%
```

### 3.2 Full QLoRA Training Script (for Colab/GPU)

```python
# qlora_trainer.py — Run on Google Colab with T4 GPU (free tier works!)
"""
Complete QLoRA fine-tuning with Hugging Face TRL (SFT Trainer)
pip install trl bitsandbytes peft transformers accelerate datasets
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch

# ── Config ────────────────────────────────────────────────
MODEL_ID = "microsoft/phi-2"  # Smaller, works on T4 GPU
OUTPUT_DIR = "./phi2-lora-finetuned"
MAX_SEQ_LENGTH = 512

# ── Load Base Model with 4-bit Quantization ───────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )

# ── Dataset ───────────────────────────────────────────────
# def format_instruction(sample):
#     return f"""### Instruction:
# {sample['instruction']}

# ### Response:
# {sample['output']}"""

# dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")

# ── LoRA Config ───────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# ── Training Config ───────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # Effective batch = 16
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    save_steps=200,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to=None,
    optim="paged_adamw_8bit"          # Memory-efficient optimizer
)

# ── SFT Trainer (supervises fine-tuning with chat template) ─
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=lora_config,
#     max_seq_length=MAX_SEQ_LENGTH,
#     tokenizer=tokenizer,
#     args=training_args,
#     formatting_func=format_instruction,
# )
# trainer.train()
# trainer.save_model(OUTPUT_DIR)
# print("✅ QLoRA fine-tuning complete!")
```

---

## 📚 Section 4: Other PEFT Methods

### 4.1 Prompt Tuning

Freeze everything; train only soft prompts (virtual tokens prepended to input):

```python
from peft import PromptTuningConfig, PromptTuningInit, TaskType

prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Answer the question accurately:",
    num_virtual_tokens=20,    # How many virtual tokens to prepend
    tokenizer_name_or_path="gpt2"
)
# Trains only 20*768 = 15,360 parameters!
```

### 4.2 Prefix Tuning

Similar to prompt tuning but adds trainable prefix to every attention layer:

```python
from peft import PrefixTuningConfig

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    prefix_projection=True   # Project prefix through a MLP first
)
```

### 4.3 PEFT Method Comparison

| Method | Trainable Params | Memory | Quality | Best For |
|--------|-----------------|--------|---------|----------|
| **Full Fine-Tuning** | 100% | Very High | Highest | Significant behavior change |
| **LoRA** | 0.1–1% | Low | High | Most use cases |
| **QLoRA** | 0.1–1% | Very Low | High | Consumer GPU training |
| **Prefix Tuning** | 0.1% | Lowest | Medium | No architecture changes needed |
| **Prompt Tuning** | <0.01% | Lowest | Good | Simple task adaptation |

---

## 🧠 Quiz: Day 19

**Q1:** LoRA reduces parameters by approximating weight updates as:
- A) A sum of sparse matrices
- B) **A product of two low-rank matrices (BA) ✅**
- C) A diagonal matrix update
- D) Pruned original weights

**Q2:** Why is `lora_B` initialized to all zeros?
- A) To speed up initialization
- B) **To ensure ΔW = BA = 0 at training start (no disruption to pretrained behavior) ✅**
- C) To reduce memory usage
- D) To improve gradient flow

**Q3:** What does QLoRA add on top of LoRA?
- A) More LoRA layers per model
- B) Quantized LoRA adapter weights
- C) **4-bit quantization of the base model weights to reduce memory ✅**
- D) Reinforcement learning

**Q4:** `lora_alpha / r` (the scaling factor) controls:
- A) Learning rate
- B) **How strongly the LoRA update influences the output ✅**
- C) The dimension of the KV cache
- D) Number of attention heads

**Q5:** Which PEFT method has the FEWEST trainable parameters?
- A) LoRA
- B) Full fine-tuning
- C) Prefix tuning
- D) **Prompt tuning ✅**

---

## 📊 Key Takeaways

| Concept | Key Point |
|---------|-----------|
| **LoRA** | Decompose Δ W into B×A where rank r << original dimensions |
| **Trainable %** | Typically 0.1–2% of total params with LoRA |
| **QLoRA** | Quantize base model to 4-bit + LoRA adapters in 16-bit |
| **Alpha/Rank** | `alpha=2×rank` is a common starting point |
| **Target Modules** | Apply LoRA to Q,K,V,O projection matrices for best results |
| **Weight Merging** | Merge LoRA back into base for zero-overhead inference |
| **SFTTrainer** | TRL's supervised fine-tuning trainer simplifies the training loop |

---

*Day 19 Complete ✅ | GenAI Course — Week 3 | Next: Day 20 — Evaluation & Benchmarks*


---

## 📚 Section 5: LoRA Rank Selection

The rank `r` is the most important LoRA hyperparameter. It controls the size and expressivity of the learned adaptation.

### 5.1 Understanding Rank

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

def profile_lora_rank(r: int, model_id: str = 'gpt2') -> dict:
    """Measure trainable parameter count for different ranks"""
    model = AutoModelForCausalLM.from_pretrained(model_id)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=r * 2,
        target_modules=['c_attn', 'c_proj'],
        lora_dropout=0.05,
        bias='none'
    )
    peft_model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    return {
        'rank': r,
        'trainable': trainable,
        'pct': trainable / total * 100,
        'adapter_mb': trainable * 2 / 1e6  # float16
    }

print(f"{'Rank':<8} {'Trainable':>12} {'% Total':>10} {'MB':>8}")
print("-" * 45)
for r in [4, 8, 16, 32, 64, 128]:
    res = profile_lora_rank(r)
    print(f"  r={res['rank']:<5} {res['trainable']:>12,} {res['pct']:>9.3f}% {res['adapter_mb']:>7.2f}")
```

### 5.2 Rank Selection Guidelines

```
Task Complexity               Recommended Rank
─────────────────────────────────────────────
Simple style / tone shift     r = 4-8
Domain adaptation             r = 8-16  ← Most common
Major behavior change         r = 32-64
Multi-task fine-tuning        r = 64-128
Near full fine-tuning         r = 256+
```

**Signs rank is too low:**
- Loss plateaus at high value early
- Model fails to generalize to task format
- Quality doesn't match baseline GPT-4 one-shot prompting

**Signs rank is too high:**
- Eval loss increases while train loss drops (overfitting)
- Adapter file is large (defeats LoRA's purpose)
- Training is slow without proportionate quality improvement

### 5.3 Rank Sweep Experiment

```python
import json

def rank_sweep_experiment(
    base_model_id: str,
    train_dataset,
    eval_dataset,
    ranks: list = [4, 8, 16, 32],
    target_modules: list = ['c_attn', 'c_proj'],
    n_epochs: int = 2
) -> list[dict]:
    """
    Train LoRA at multiple ranks, compare eval loss.
    Use to determine the optimal rank for your task.
    """
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    results = []
    
    for r in ranks:
        print(f"\n--- Training with r={r} ---")
        model = AutoModelForCausalLM.from_pretrained(base_model_id)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r, lora_alpha=r * 2,
            target_modules=target_modules,
            lora_dropout=0.05, bias='none'
        )
        peft_model = get_peft_model(model, config)
        
        training_args = TrainingArguments(
            output_dir=f'./lora_r{r}',
            num_train_epochs=n_epochs,
            per_device_train_batch_size=4,
            logging_steps=50,
            evaluation_strategy='epoch',
            report_to=None
        )
        
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        results.append({
            'rank': r,
            'trainable_params': trainable,
            'eval_loss': eval_results['eval_loss'],
            'perplexity': 2 ** eval_results['eval_loss']
        })
        print(f"  r={r}: eval_loss={eval_results['eval_loss']:.4f}, perplexity={2**eval_results['eval_loss']:.2f}")
    
    return results

# Read results
# results = rank_sweep_experiment('gpt2', train_ds, eval_ds)
# best = min(results, key=lambda x: x['eval_loss'])
# print(f"Best rank: r={best['rank']} (eval_loss={best['eval_loss']:.4f})")
```

---

## 📚 Section 6: DoRA — Weight-Decomposed LoRA

DoRA (Liu et al., 2024) improves on LoRA by decomposing pretrained weights into
magnitude and direction components, then fine-tuning each separately.

### 6.1 DoRA vs LoRA

**LoRA** updates: `W' = W + B*A * (alpha/r)`

**DoRA** updates:
1. Decompose `W = m * (W / ||W||)` (magnitude × unit direction)
2. Fine-tune magnitude `m` separately from directional update `B*A`
3. Result: better performance at the same parameter budget

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

def compare_lora_dora(model_id: str = 'gpt2'):
    """Compare parameter counts and describe quality differences"""
    
    for use_dora, method_name in [(False, 'LoRA'), (True, 'DoRA')]:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=['c_attn', 'c_proj'],
            use_dora=use_dora,
            lora_dropout=0.05,
            bias='none'
        )
        peft_model = get_peft_model(model, config)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(f"{method_name} (r=8): {trainable:,} trainable params")
    
    print("\nQuality comparison (LLaMA-7B, commonsense reasoning):")
    print("  LoRA r=4:  79.2%")
    print("  DoRA r=4:  82.1%  (+2.9% at same param count!)")
    print("  LoRA r=16: 80.5%")
    print("  DoRA r=8:  83.6%  (DoRA r=8 > LoRA r=16)")

compare_lora_dora()
```

---

## 📚 Section 7: Production LoRA Serving

### 7.1 Load and Use a Saved LoRA Adapter

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_lora_for_inference(base_model_id: str, adapter_path: str):
    """Load base model + LoRA adapter"""
    config = PeftConfig.from_pretrained(adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer

def generate(model, tokenizer, prompt: str, max_tokens: int = 100) -> str:
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
```

### 7.2 Merging LoRA for Zero-Overhead Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def merge_lora_and_save(
    base_model_id: str,
    adapter_path: str,
    output_path: str
):
    """
    Merge LoRA back into base model weights.
    After merging: standard model, no PEFT dependency, zero inference overhead.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load on CPU for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map='cpu'
    )
    
    # Load adapter
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge: W_merged = W_base + B*A*(alpha/r)
    merged_model = peft_model.merge_and_unload()
    
    # Save as standard HuggingFace model
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"Merged model saved to: {output_path}")
    print("→ No PEFT dependency needed for inference")
    print("→ Can be uploaded to Hugging Face Hub")

# merge_lora_and_save('gpt2', './my_lora_adapter', './my_merged_model')
```

### 7.3 Multi-Adapter Hot-Swapping

One base model, multiple task-specific LoRA adapters, switchable without reloading:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class MultiAdapterServer:
    """Production-grade multi-adapter LLM server"""
    
    def __init__(self, base_model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.base = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.model = None
        self._adapters_loaded = {}
    
    def load_adapter(self, name: str, path: str):
        """Load a new adapter (adds it to the model)"""
        if self.model is None:
            self.model = PeftModel.from_pretrained(self.base, path, adapter_name=name)
        else:
            self.model.load_adapter(path, adapter_name=name)
        self._adapters_loaded[name] = path
        print(f"Loaded adapter: {name}")
    
    def predict(self, text: str, adapter_name: str, max_tokens: int = 100) -> str:
        """Switch to specified adapter and generate"""
        self.model.set_adapter(adapter_name)
        import torch
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    def list_adapters(self) -> list:
        return list(self._adapters_loaded.keys())

# server = MultiAdapterServer('gpt2')
# server.load_adapter('customer_service', './adapters/cs')
# server.load_adapter('code_assistant', './adapters/code')
# print(server.predict('How do I reset my password?', 'customer_service'))
# print(server.predict('def fibonacci(n):', 'code_assistant'))
```

---

## 📚 Section 8: LoRA Hyperparameter Reference

| Param | Low | Default | High | Notes |
|-------|-----|---------|------|-------|
| `r` | 4 | 16 | 128 | Higher = more expressive |
| `lora_alpha` | r | 2×r | 4×r | Scaling: alpha/r |
| `lora_dropout` | 0 | 0.05 | 0.1 | Regularization |
| `learning_rate` | 5e-5 | 2e-4 | 5e-4 | Higher than full FT |
| `num_epochs` | 1 | 3 | 10 | Watch for overfitting |
| gradient_accum | 1 | 4 | 16 | For large eff batch |

**Target modules by architecture:**
- **GPT-2**: `c_attn`, `c_proj`, `c_fc`
- **LLaMA/Mistral**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Falcon**: `query_key_value`, `dense`, `dense_h_to_4h`
- **BERT**: `query`, `key`, `value`, `dense`

---

## 🎯 Day 19 Extended Lab: LoRA Rank Ablation Study

Compare LoRA at different ranks on a code generation task:

1. Generate 500 function-docstring pairs from Python stdlib
2. Fine-tune GPT-2 with r=4, r=16, r=64
3. Evaluate: BLEU-4 score and inference speed (tokens/sec)
4. Plot quality vs. trainable parameter count

**Expected findings:**
- r=4 → Fast, small adapter (0.5 MB), lower BLEU
- r=16 → Best tradeoff (1.5 MB adapter, high BLEU)
- r=64 → Diminishing returns, larger adapter (5 MB)

```python
print("=== Day 19 Extended Lab ===")
print("Task: LoRA rank ablation on code generation")
print("Models: gpt2 with r=[4, 16, 64]")
print("Eval: BLEU-4 and tokens/sec")
print("See day-19-rank-ablation.py for the full experiment script")
```

---

*Day 19 Extended Complete — Advanced LoRA: Rank Selection, DoRA, Production Serving*


---

## Section 9: LoRA in Practice — End-to-End Workflow

### 9.1 Preparing Custom Instruction Datasets

```python
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import json, random

def create_instruction_dataset(raw_examples: list[dict], tokenizer_id: str = "gpt2", test_split: float = 0.1) -> DatasetDict:
    """
    Format and split a raw instruction dataset for LoRA fine-tuning.
    raw_examples: [{"instruction": str, "input": str (opt), "output": str}]
    """
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    def format_example(ex: dict) -> str:
        if ex.get("input", "").strip():
            return f"""Below is an instruction that describes a task with input.
Write a response that appropriately completes the request.

### Instruction:
{ex['instruction']}

### Input:
{ex['input']}

### Response:
{ex['output']}""".strip()
        else:
            return f"""Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{ex['instruction']}

### Response:
{ex['output']}""".strip()
    
    def tokenize_example(examples):
        texts = [format_example({"instruction": i, "input": inp, "output": o}) 
                 for i, inp, o in zip(examples["instruction"], examples.get("input", [""]*len(examples["instruction"])), examples["output"])]
        tokenized = tokenizer(texts, truncation=True, max_length=512, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Shuffle and split
    random.shuffle(raw_examples)
    n_test = max(1, int(len(raw_examples) * test_split))
    
    train_data = raw_examples[n_test:]
    test_data = raw_examples[:n_test]
    
    train_ds = Dataset.from_list(train_data).map(tokenize_example, batched=True, remove_columns=["instruction", "input", "output"] if "input" in raw_examples[0] else ["instruction", "output"])
    test_ds = Dataset.from_list(test_data).map(tokenize_example, batched=True, remove_columns=["instruction", "input", "output"] if "input" in raw_examples[0] else ["instruction", "output"])
    
    return DatasetDict({"train": train_ds, "test": test_ds})

# Example dataset
examples = [
    {"instruction": "Explain what a transformer is", "output": "A Transformer is a neural network architecture based on self-attention mechanisms, introduced in 'Attention Is All You Need' (2017). It processes all tokens in parallel rather than sequentially."},
    {"instruction": "What is the difference between BERT and GPT?", "output": "BERT uses bidirectional encoding (sees all tokens at once) for understanding tasks. GPT uses causal (unidirectional) decoding for generation tasks."},
    {"instruction": "Define embeddings in machine learning", "output": "Embeddings are dense numerical representations of data (text, images, etc.) in a continuous vector space. Similar items have similar embedding vectors."},
    {"instruction": "What is fine-tuning?", "output": "Fine-tuning is training a pretrained model on task-specific data to adapt its behavior. It updates model weights using a small dataset and learning rate."},
]

dataset = create_instruction_dataset(examples)
print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")
print(f"Sample: {dataset['train'][0].keys()}")
```

### 9.2 LoRA Training with Callbacks

```python
from transformers import TrainerCallback, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling
import math

class LoRATrainingMonitor(TrainerCallback):
    """Monitor and log LoRA training metrics"""
    
    def __init__(self, log_interval: int = 25):
        self.log_interval = log_interval
        self.history = {"loss": [], "eval_loss": [], "lr": []}
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.history["loss"].append({"step": state.global_step, "loss": logs["loss"]})
            if "eval_loss" in logs:
                ppl = math.exp(logs["eval_loss"])
                self.history["eval_loss"].append({
                    "step": state.global_step,
                    "eval_loss": logs["eval_loss"],
                    "perplexity": ppl
                })
                print(f"Step {state.global_step}: eval_loss={logs['eval_loss']:.4f}, PPL={ppl:.2f}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.history["loss"]:
            recent_losses = self.history["loss"][-10:]
            avg_loss = sum(r["loss"] for r in recent_losses) / len(recent_losses)
            print(f"Epoch {state.epoch:.0f} complete — avg_loss: {avg_loss:.4f}")
    
    def get_summary(self) -> dict:
        return {
            "final_train_loss": self.history["loss"][-1]["loss"] if self.history["loss"] else None,
            "final_eval_loss": self.history["eval_loss"][-1]["eval_loss"] if self.history["eval_loss"] else None,
            "final_perplexity": self.history["eval_loss"][-1]["perplexity"] if self.history["eval_loss"] else None,
            "total_steps": self.history["loss"][-1]["step"] if self.history["loss"] else 0
        }

def train_lora_with_monitoring(
    base_model_id: str = "gpt2",
    dataset=None,
    output_dir: str = "./lora_output",
    rank: int = 16,
    epochs: int = 3
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank, lora_alpha=rank * 2,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05, bias="none"
    )
    
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    
    monitor = LoRATrainingMonitor()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=False,  # Set to True with GPU
        logging_steps=25,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=None,
        dataloader_num_workers=0
    )
    
    # trainer = Trainer(
    #     model=peft_model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["test"],
    #     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    #     callbacks=[monitor]
    # )
    # trainer.train()
    # summary = monitor.get_summary()
    # print(f"Training complete: {summary}")
    
    print(f"LoRA training setup complete: r={rank}, epochs={epochs}")
    print("Uncomment trainer code and run with GPU for real training")
    return monitor.get_summary()

result = train_lora_with_monitoring()
print("Training monitor configured successfully")
```

### 9.3 Evaluating Fine-Tuned Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from typing import Optional

class LoRAEvaluator:
    """Compare base model vs LoRA fine-tuned model"""
    
    def __init__(self, base_model_id: str, adapter_path: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.base_model.eval()
        
        # Load fine-tuned model (if adapter provided)
        self.finetuned_model = None
        if adapter_path:
            finetuned_base = AutoModelForCausalLM.from_pretrained(base_model_id)
            self.finetuned_model = PeftModel.from_pretrained(finetuned_base, adapter_path)
            self.finetuned_model.eval()
    
    def generate(self, model, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from a model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def compare(self, prompts: list[str]) -> list[dict]:
        """Compare base vs fine-tuned on a set of prompts"""
        results = []
        
        for prompt in prompts:
            base_output = self.generate(self.base_model, prompt)
            result = {"prompt": prompt, "base": base_output}
            
            if self.finetuned_model:
                ft_output = self.generate(self.finetuned_model, prompt)
                result["finetuned"] = ft_output
            
            results.append(result)
            print(f"Prompt: {prompt[:50]}")
            print(f"  Base:      {base_output[:100]}")
            if self.finetuned_model:
                print(f"  Finetuned: {result['finetuned'][:100]}")
            print()
        
        return results

evaluator = LoRAEvaluator("gpt2")  # No adapter for demo
test_prompts = [
    "What is a transformer neural network?",
    "Explain gradient descent in simple terms.",
    "What is the difference between supervised and unsupervised learning?",
]
comparisons = evaluator.compare(test_prompts)
print(f"Evaluated {len(comparisons)} prompts")
```

---

*Day 19 Second Pass Complete — Dataset preparation, training callbacks, and model evaluation*

# Day 04 — The Transformer Architecture: Attention Is All You Need

> **Week 1 | Foundations** | ⏱️ Estimated Time: 4–5 hours | 🔥 Difficulty: Advanced

---

## 🤔 The Paper That Changed Everything

It's June 2017. Eight Google researchers submit a paper to NeurIPS with an audacious title:

**"Attention Is All You Need"**

The title is a provocation. The NLP field had spent a decade meticulously refining RNNs and LSTMs. Researchers were proud of these architectures — they were theoretically grounded, biologically inspired, and had achieved state-of-the-art on every major benchmark.

Now these eight researchers were claiming: throw all of that away. Every recurrent connection, every LSTM gate, every carefully sequenced timestep — none of it is needed. Just attention. *Only* attention.

The paper was accepted. The model was trained. The results were so good that the reviewers assumed there had to be a flaw.

There wasn't. The Transformer outperformed the best LSTM-based translation models on English-to-German and English-to-French translation, trained 3× faster on modern GPUs, and scaled in ways no LSTM could approach.

By 2019, BERT (a Transformer) exceeded human performance on the GLUE language understanding benchmark. By 2020, GPT-3 (a Transformer) shocked the world with 175 billion parameters. By 2023, ChatGPT had over 100 million users in 2 months. By 2024, every frontier AI model — text, image, video, code, protein structure — runs on a Transformer or its descendants.

All from one paper. One idea. One mechanism.

Today, you understand it completely.

---

## 🎯 Learning Objectives

By the end of today you will:

- [ ] Explain *why* RNNs were fundamentally limited and why attention was the solution
- [ ] Trace a single token through an entire Transformer, shape by shape
- [ ] Explain what self-attention computes and *why* it works (not just how)
- [ ] Understand Queries, Keys, and Values intuitively (the library retrieval analogy)
- [ ] Know what positional encoding does and why it's necessary
- [ ] Distinguish encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) variants
- [ ] Implement a simplified Transformer from scratch in PyTorch
- [ ] Read and work with pre-trained Transformers via Hugging Face

---

## 📚 Theory (90 min)

### 4.1 Why Transformers — The Motivation

Before Transformers, the state-of-the-art for NLP tasks were RNN-based models like LSTMs and GRUs. These had two critical bottlenecks:

**Problem 1: Sequential computation**
```
Word 1 → Word 2 → Word 3 → ... → Word N
```
Each word must wait for the previous word to finish processing. For a 512-word sentence, that's 512 sequential steps — you cannot parallelize this across GPUs efficiently. This made training extremely slow.

**Problem 2: Long-range dependency degradation**
Even with LSTMs, capturing dependencies between words that are very far apart (e.g., subject and verb separated by a 50-word clause) is unreliable. The "memory" of early tokens degrades as the sequence grows.

**The Transformer Solution (Vaswani et al., 2017):**
- **Self-attention**: Every word attends to every other word *simultaneously* — no sequential dependency
- **Full parallelization**: All positions processed in parallel during training
- **Direct connections**: No degradation — word at position 1 can directly attend to word at position 512 with equal ease

The paper title **"Attention Is All You Need"** was bold: they removed convolution and recurrence entirely, keeping only attention and feed-forward layers. It worked spectacularly.

---

### 4.2 The High-Level Architecture

The original Transformer was an **encoder-decoder** model designed for translation.

```
                     OUTPUT SOFTMAX
                          ↑
                    LINEAR PROJECTION
                          ↑
         ┌─────────────────────────────────┐
         │         DECODER STACK           │
         │      (6 identical layers)       │
         │  [Masked Self-Attention]         │
         │  [Cross-Attention]               │
         │  [Feed-Forward]                  │
         └─────────────┬───────────────────┘
                       ↑ (encoder output)
         ┌─────────────────────────────────┐
         │         ENCODER STACK           │
         │      (6 identical layers)       │
         │  [Self-Attention]                │
         │  [Feed-Forward]                  │
         └─────────────────────────────────┘
                       ↑
              INPUT + POSITIONAL ENCODING
```

**Three Architectural Variants (Modern Usage):**

| Variant | Architecture | Examples | Best For |
|---|---|---|---|
| **Encoder-only** | Encoder stack only | BERT, RoBERTa, DeBERTa | Classification, NER, embeddings |
| **Decoder-only** | Decoder stack only | GPT-2/3/4, LLaMA, Claude | Text generation, completion |
| **Encoder-Decoder** | Full seq2seq | T5, BART, mT5 | Translation, summarization, Q&A |

---

### 4.3 Input Embeddings

Before entering the Transformer, raw tokens (integers) must be converted to dense vectors.

**Token Embedding:**
```
Input:   ["The", "cat", "sat"]
Tokens:  [  12,    47,   89 ]
         ↓ Embedding Lookup (Vocab_size × d_model)
Vectors: [ [0.2, ..., 0.5],   # 512-dim vector for "The"
           [0.1, ..., 0.9],   # 512-dim vector for "cat"
           [0.7, ..., 0.3] ]  # 512-dim vector for "sat"
```

The embedding matrix is **learned during training**. Similar words end up with similar vectors (word2vec-like property but learned end-to-end).

**Typical Dimensions:**
- `d_model` = 512 (original paper), 768 (BERT-base), 1024 (BERT-large), 4096 (LLaMA-7B)
- Vocabulary size = 30,000–100,000+ tokens

---

### 4.4 Positional Encoding

Since self-attention processes all positions simultaneously (unlike RNNs), it has **no inherent sense of order**. Without positional encoding, "The cat ate the dog" and "The dog ate the cat" would produce identical outputs (since the same words are present).

**Solution: Add positional information to embeddings**

**Sinusoidal Positional Encoding (Original Paper):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where `pos` = position index, `i` = dimension index.

**Why sinusoidal?**
1. Deterministic — no additional parameters needed
2. Works for sequences longer than training data
3. The model can learn to use relative positions (PE(pos+k) is a linear function of PE(pos))

**Modern Alternative: RoPE (Rotary Position Embedding)**
Used by LLaMA, Mistral, GPT-NeoX. Encodes relative positions into the attention computation itself, achieving better performance for long sequences. GPT-4, Gemini, and most modern LLMs use RoPE or variants of it.

**ALiBi (Attention with Linear Biases)**: BLOOM's approach — adds a linear bias to attention scores based on distance. No extra parameters, good generalization.

```
Final Input to Transformer = Token Embeddings + Positional Encodings
Shape: [batch_size, seq_len, d_model]
```

---

### 4.5 Scaled Dot-Product Attention — The Core Mechanism

This is the heart of the Transformer. Understanding this deeply is essential.

**Inputs:** Three matrices derived from the input:
- **Q (Queries)**: "What am I looking for?"
- **K (Keys)**: "What do I have to offer?"
- **V (Values)**: "What information do I carry?"

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

**Step-by-step walkthrough with example:**

```
Sentence: "The cat sat on the mat"
Query: We want to find what "sat" (position 3) attends to.

Step 1: Get Q, K, V for each position
  - Q3 = W_Q * embedding["sat"]    → shape: [d_k]
  - K1 = W_K * embedding["The"]   → shape: [d_k]
  - K2 = W_K * embedding["cat"]   → shape: [d_k]
  - ...etc

Step 2: Compute raw attention scores (dot product)
  score(3,1) = Q3 · K1 = 0.3   ("sat" vs "The")
  score(3,2) = Q3 · K2 = 0.9   ("sat" vs "cat") ← high
  score(3,3) = Q3 · K3 = 0.5   ("sat" vs "sat")
  score(3,4) = Q3 · K4 = 0.2   ("sat" vs "on")

Step 3: Scale by √d_k (prevents vanishing gradients)
  Scaled scores: [0.3/√64, 0.9/√64, 0.5/√64, 0.2/√64]
               = [0.0375, 0.1125, 0.0625, 0.025]

Step 4: Apply softmax (convert to probabilities)
  Attention weights: [0.18, 0.41, 0.24, 0.17]

Step 5: Weighted sum of Values
  Output3 = 0.18*V1 + 0.41*V2 + 0.24*V3 + 0.17*V4
```

The output for each position is a **context-aware** representation — it has gathered information from the positions it attended to most.

**Why scale by √d_k?**
For large d_k (e.g., 64), dot products can have large magnitudes, pushing softmax into regions with near-zero gradients. Scaling prevents this.

**Matrix Form (Efficient Parallelization):**
```
Q shape: [batch, seq_len, d_k]
K shape: [batch, seq_len, d_k]
V shape: [batch, seq_len, d_v]

QK^T shape: [batch, seq_len, seq_len]  ← attention score matrix
After softmax: [batch, seq_len, seq_len]  ← attention weights
After * V: [batch, seq_len, d_v]  ← context vectors
```

---

### 4.6 Multi-Head Attention

Single-head attention learns one type of relationship. Multi-head attention runs **h parallel attention heads**, each looking for different types of dependencies.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

where head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)
```

**Intuition for different heads:**
- Head 1 might learn syntactic relationships (subject-verb agreement)
- Head 2 might learn coreference (pronouns referring to nouns)
- Head 3 might learn semantic similarity (synonyms attending to each other)
- Head 4 might learn positional patterns (words attending to their neighbors)

**Parameters:**
- Original paper: h=8 heads, d_k = d_v = d_model/h = 512/8 = 64
- BERT-base: h=12 heads, d_model=768
- GPT-3 175B: h=96 heads, d_model=12288
- LLaMA 3 70B: 64 heads (with grouped query attention)

**Grouped Query Attention (GQA)** — Modern LLMs like LLaMA 2/3:
Instead of each head having its own K,V matrices, groups of query heads share K,V matrices. This reduces memory during inference (KV cache) by 4-8x while maintaining quality.

---

### 4.7 Masking in Attention

**Padding Mask:** Real sequences have variable lengths. Shorter sequences are padded with special tokens. We mask padding positions so they don't affect attention.

```
Attention mask: [1, 1, 1, 1, 0, 0]  (1=real, 0=padding)
Set masked positions to -∞ before softmax → they become 0 after softmax
```

**Causal (Look-ahead) Mask — Critical for GPT/Decoder:**
During training a decoder for language modeling, position i should only attend to positions ≤ i. We use a lower-triangular mask:

```
Mask:
  The  cat  sat  on
The [  1,   0,   0,   0 ]   ← "The" attends only to "The"
cat [  1,   1,   0,   0 ]   ← "cat" attends to "The", "cat"
sat [  1,   1,   1,   0 ]   ← "sat" attends to first 3
on  [  1,   1,   1,   1 ]   ← "on" attends to all
```

This enables **teacher forcing**: we can compute all positions in parallel during training while respecting causality.

---

### 4.8 Position-wise Feed-Forward Network (FFN)

After attention, each position independently passes through a two-layer MLP:

```
FFN(x) = ReLU(x * W_1 + b_1) * W_2 + b_2
```

Or with GELU (BERT, GPT):
```
FFN(x) = GELU(x * W_1 + b_1) * W_2 + b_2
```

**Dimensions:**
- d_model=512 → FFN inner dimension = 2048 (4x expansion)
- BERT: d_model=768, FFN=3072
- GPT-3: d_model=12288, FFN=49152
- **This is where ~2/3 of model parameters live!**

**What does FFN do?**
Research shows FFN layers act as a "key-value memory" — they store factual knowledge. When you ask GPT "What is the capital of France?", the attention mechanism retrieves relevant context, and the FFN layers "recall" that Paris is the answer.

**Modern Variant: SwiGLU (LLaMA, PaLM, Gemini):**
```
FFN(x) = (SiLU(x * W_1) ⊙ (x * W_gate)) * W_2
```
SwiGLU consistently outperforms ReLU/GELU in large models.

---

### 4.9 Layer Normalization & Residual Connections

**Residual Connection (Skip Connection):**
```
Output = LayerNorm(x + Sublayer(x))
```
Each sublayer (attention, FFN) has a residual connection:
- Prevents vanishing gradients in deep networks (same principle as ResNet)
- Allows direct gradient flow back to early layers
- Network can learn "identity function" if needed (skip the sublayer)

**Layer Normalization:**
Normalizes across the feature dimension (d_model) for each token position independently.

```python
LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
```

Where γ, β are **learned** parameters, μ and σ are computed per-token.

**Pre-LN vs Post-LN:**
- Post-LN (original paper): LayerNorm after residual addition — can be unstable to train
- **Pre-LN (modern standard)**: LayerNorm before sublayer — more stable training, used by GPT-3, LLaMA, etc.

```
Post-LN: LayerNorm(x + Attention(x))
Pre-LN:  x + Attention(LayerNorm(x))  ← modern default
```

---

### 4.10 Encoder vs Decoder in Detail

**Encoder Block (3 sublayers):**
```
Input
  ↓
Multi-Head Self-Attention (full bidirectional)
  ↓ (residual + layernorm)
Feed-Forward Network
  ↓ (residual + layernorm)
Output → passes to next encoder block
```

**Decoder Block (3 sublayers):**
```
Target
  ↓
Masked Multi-Head Self-Attention (causal — future positions masked)
  ↓ (residual + layernorm)
Cross-Attention (queries from decoder, keys+values from encoder output)
  ↓ (residual + layernorm)
Feed-Forward Network
  ↓ (residual + layernorm)
Output → passes to next decoder block
```

**Cross-Attention explained:**
- Q: "What information does my current position need?"
- K, V: "What information did the encoder extract from the input?"
- Allows decoder to "look at" any part of the encoded input at each generation step

**Decoder-Only (GPT-style):**
No encoder, no cross-attention. Just stacked masked self-attention + FFN blocks. Input and output are in the same sequence (autoregressive generation).

---

### 4.11 Output: Linear + Softmax

After the final decoder block:
```
[batch, seq_len, d_model] 
→ Linear Layer [d_model → vocab_size]
→ Softmax 
→ [batch, seq_len, vocab_size]  ← probability distribution over vocabulary
```

During training: Use cross-entropy loss against ground truth tokens.

During inference:
- **Greedy decoding**: Always pick highest probability token
- **Temperature sampling**: Sample with temperature T
- **Top-k sampling**: Sample from top K tokens
- **Top-p (nucleus) sampling**: Sample from smallest set covering p% of probability
- **Beam search**: Maintain B candidate sequences simultaneously

---

### 4.12 Counting Transformer Parameters

Let's count for a simple Transformer (d_model=512, heads=8, d_ff=2048, vocab=30000, layers=6):

| Component | Parameters |
|---|---|
| Token Embedding | vocab × d_model = 30000 × 512 = 15.36M |
| Positional Encoding (sinusoidal) | 0 (no learnable params) |
| Self-Attention (per layer) | 4 × d_model² = 4 × 512² = 1.05M |
| FFN (per layer) | 2 × d_model × d_ff = 2 × 512 × 2048 = 2.1M |
| LayerNorm (per block) | 2 × d_model × 4 = ~4K |
| **Total (6 encoder layers)** | ~15.36M + 6 × (1.05M + 2.1M) = **~34M** |

For reference:
- BERT-base: 110M params
- GPT-3: 175B params
- LLaMA 3 70B: 70B params
- GPT-4: estimated 1.8T (Mixture of Experts)

---

## 💻 Lab 1: Transformer from Scratch in PyTorch (90 min)

### Lab 4.1: Self-Attention Implementation

```python
# lab_04_01_self_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Implements: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, heads, seq_len, d_k]
            K: [batch_size, heads, seq_len, d_k]
            V: [batch_size, heads, seq_len, d_v]
            mask: [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        Returns:
            output: [batch_size, heads, seq_len, d_v]
            attention_weights: [batch_size, heads, seq_len, seq_len]
        """
        d_k = Q.size(-1)

        # Step 1: Compute raw attention scores - Q @ K^T
        # Shape: [batch, heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Step 2: Apply mask (set masked positions to very negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 3: Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Step 4: Weighted sum of values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: run h attention heads in parallel.
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) @ W_O
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Projection matrices for Q, K, V, and output
        self.W_Q = nn.Linear(d_model, d_model)  # [d_model → d_model]
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # output projection

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        Split last dimension into (num_heads, d_k).
        [batch, seq_len, d_model] → [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [batch, heads, seq_len, d_k]

    def combine_heads(self, x):
        """
        Reverse of split_heads.
        [batch, num_heads, seq_len, d_k] → [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2)  # [batch, seq_len, heads, d_k]
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Project inputs
        Q = self.split_heads(self.W_Q(Q))  # [batch, heads, seq_len, d_k]
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        # Apply attention for all heads in parallel
        x, attention_weights = self.attention(Q, K, V, mask)

        # Concatenate heads and project
        x = self.combine_heads(x)     # [batch, seq_len, d_model]
        x = self.W_O(x)               # [batch, seq_len, d_model]

        return x, attention_weights
```

### Lab 4.2: Positional Encoding

```python
# lab_04_02_positional_encoding.py
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from 'Attention Is All You Need'.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create matrix of shape [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        # Position indices: [max_seq_len, 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Frequency terms: [d_model/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims

        # Add batch dimension: [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a learnable parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def visualize_positional_encoding(d_model=128, max_len=100):
    """Visualize the positional encoding matrix."""
    pe = PositionalEncoding(d_model, max_len)

    # Get the PE matrix
    pe_matrix = pe.pe.squeeze(0).numpy()  # [max_len, d_model]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(pe_matrix, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.title('Positional Encoding Matrix\n(rows=positions, cols=dimensions)')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')

    plt.subplot(1, 2, 2)
    # Plot specific dimensions across positions
    for dim in [0, 1, 4, 5, 10, 11]:
        plt.plot(pe_matrix[:50, dim], label=f'dim {dim}')
    plt.legend()
    plt.title('PE values for specific dimensions')
    plt.xlabel('Position')
    plt.ylabel('PE Value')

    plt.tight_layout()
    plt.savefig('positional_encoding_viz.png', dpi=150)
    plt.show()
    print("✅ Visualization saved to positional_encoding_viz.png")


if __name__ == "__main__":
    visualize_positional_encoding()
```

### Lab 4.3: Complete Transformer Block

```python
# lab_04_03_transformer_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# (Include MultiHeadAttention and PositionalEncoding from previous labs)

class FeedForwardNetwork(nn.Module):
    """
    Position-wise FFN: FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Expand to d_ff, apply activation, project back to d_model
        x = self.linear1(x)
        x = F.gelu(x)         # GELU is modern standard (BERT/GPT use this)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    """
    One Transformer Encoder Block:
    1. Multi-Head Self-Attention (with residual + layernorm)
    2. Feed-Forward Network (with residual + layernorm)
    Uses Pre-LN (modern) architecture.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN Self-Attention with residual
        residual = x
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.self_attention(x_norm, x_norm, x_norm, mask)
        x = residual + self.dropout(attn_output)

        # Pre-LN Feed-Forward with residual
        residual = x
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = residual + self.dropout(ff_output)

        return x, attn_weights


class DecoderBlock(nn.Module):
    """
    One Transformer Decoder Block:
    1. Masked Self-Attention (causal)
    2. Cross-Attention (attends to encoder output)
    3. Feed-Forward Network
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention (causal)
        residual = x
        x_norm = self.norm1(x)
        self_attn_out, _ = self.masked_self_attention(x_norm, x_norm, x_norm, tgt_mask)
        x = residual + self.dropout(self_attn_out)

        # Cross-attention: queries from decoder, keys/values from encoder
        residual = x
        x_norm = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attention(
            x_norm, encoder_output, encoder_output, src_mask
        )
        x = residual + self.dropout(cross_attn_out)

        # Feed-forward
        residual = x
        x_norm = self.norm3(x)
        ff_out = self.feed_forward(x_norm)
        x = residual + self.dropout(ff_out)

        return x, cross_attn_weights


class SimpleTransformer(nn.Module):
    """
    Complete Encoder-Decoder Transformer for sequence-to-sequence tasks.
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Shared embedding (encoder + decoder share weights — common optimization)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder and Decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights with Xavier uniform
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, seq_len):
        """Create lower-triangular causal mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        return mask  # [1, 1, seq_len, seq_len]

    def make_padding_mask(self, x, pad_token=0):
        """Create mask for padding tokens."""
        return (x != pad_token).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

    def encode(self, src, src_mask=None):
        # Embed + positional encode
        x = self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model))

        # Pass through encoder stack
        for encoder_block in self.encoder_layers:
            x, _ = encoder_block(x, src_mask)

        return self.final_norm(x)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Embed + positional encode target
        x = self.pos_encoding(self.embedding(tgt) * math.sqrt(self.d_model))

        # Pass through decoder stack
        for decoder_block in self.decoder_layers:
            x, _ = decoder_block(x, encoder_output, src_mask, tgt_mask)

        return self.final_norm(x)

    def forward(self, src, tgt):
        # Create masks
        src_mask = self.make_padding_mask(src)
        tgt_mask = self.make_causal_mask(tgt.size(1)).to(tgt.device)

        # Encode source
        encoder_output = self.encode(src, src_mask)

        # Decode target
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        logits = self.output_projection(decoder_output)  # [batch, seq_len, vocab_size]

        return logits


# ---- Test the Transformer ----
if __name__ == "__main__":
    print("=" * 60)
    print("Building Transformer from Scratch")
    print("=" * 60)

    # Configuration
    VOCAB_SIZE = 10000
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    D_FF = 2048
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 4
    SRC_LEN = 32
    TGT_LEN = 28

    # Initialize model
    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture:")
    print(f"  d_model: {D_MODEL}")
    print(f"  Attention heads: {NUM_HEADS}")
    print(f"  Encoder/Decoder layers: {NUM_LAYERS}")
    print(f"  FFN dim: {D_FF}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"\nParameter Counts:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create dummy batch
    src = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SRC_LEN))
    tgt = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, TGT_LEN))

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt)

    print(f"\nInput/Output shapes:")
    print(f"  Source input:  {src.shape}  (batch={BATCH_SIZE}, src_len={SRC_LEN})")
    print(f"  Target input:  {tgt.shape}  (batch={BATCH_SIZE}, tgt_len={TGT_LEN})")
    print(f"  Output logits: {logits.shape}  (batch={BATCH_SIZE}, tgt_len={TGT_LEN}, vocab={VOCAB_SIZE})")

    # Verify first token predictions
    probs = torch.softmax(logits[0, 0, :], dim=-1)
    top5_tokens = torch.topk(probs, 5)
    print(f"\nTop-5 predicted next tokens (first batch item, first position):")
    for i, (prob, token_id) in enumerate(zip(top5_tokens.values, top5_tokens.indices)):
        print(f"  #{i+1}: Token {token_id.item():5d} | Probability: {prob.item():.4f}")

    print("\n✅ Transformer forward pass successful!")
```

### Lab 4.4: Visualizing Attention Patterns

```python
# lab_04_04_attention_visualization.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention(attention_weights, tokens, title="Attention Heatmap"):
    """
    Visualize the attention weight matrix.
    attention_weights: [seq_len, seq_len] (averaged across heads)
    tokens: list of strings
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(title, fontsize=14)

    num_heads = attention_weights.shape[0]
    for head in range(min(num_heads, 8)):
        ax = axes[head // 4][head % 4]
        weights = attention_weights[head].numpy()

        sns.heatmap(
            weights,
            xticklabels=tokens,
            yticklabels=tokens,
            ax=ax,
            cmap='Blues',
            vmin=0, vmax=1,
            annot=len(tokens) <= 10,
            fmt='.2f',
            cbar=False
        )
        ax.set_title(f'Head {head + 1}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Attention visualization saved!")


# Simulate attention weights from a trained model
def demo_attention_visualization():
    tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
    seq_len = len(tokens)
    num_heads = 8

    # Simulate different head behaviors
    attention_weights = torch.zeros(num_heads, seq_len, seq_len)

    # Head 1: Diagonal (local attention — each token attends to itself)
    attention_weights[0] = torch.eye(seq_len)

    # Head 2: "cat" and "sat" have high mutual attention (subject-verb)
    attn = torch.ones(seq_len, seq_len) * 0.1
    attn[1, 2] = 0.9  # "cat" → "sat"
    attn[2, 1] = 0.9  # "sat" → "cat"
    attention_weights[1] = attn / attn.sum(dim=-1, keepdim=True)

    # Head 3: "the" tokens attend to each other (coreference)
    attn = torch.ones(seq_len, seq_len) * 0.1
    attn[0, 4] = 0.8  # "The" → "the"
    attn[4, 0] = 0.8
    attention_weights[2] = attn / attn.sum(dim=-1, keepdim=True)

    # Heads 4-8: Random patterns
    for h in range(3, 8):
        attn = torch.rand(seq_len, seq_len)
        attention_weights[h] = attn / attn.sum(dim=-1, keepdim=True)

    visualize_attention(attention_weights, tokens, "Sample Attention Patterns (8 Heads)")


if __name__ == "__main__":
    demo_attention_visualization()
```

---

## 🧪 Lab 5: Training the Transformer on a Toy Task

```python
# lab_04_05_train_transformer.py
"""
Train a mini-Transformer on a sequence reversal task.
Input: [1, 2, 3, 4, 5] → Output: [5, 4, 3, 2, 1]
This is a minimal but complete training loop.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import math


# ---- Dataset ----
class SequenceReversalDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=10, vocab_size=20):
        self.data = []
        for _ in range(num_samples):
            seq = [random.randint(1, vocab_size - 1) for _ in range(seq_len)]
            self.data.append((seq, list(reversed(seq))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long)
        )


# ---- Simple Decoder-Only Transformer (GPT-style) ----
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3, d_ff=256, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_len, d_model)  # learned PE

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout=0.1, batch_first=True),
            num_layers=num_layers
        )
        self.output = nn.Linear(d_model, vocab_size)

    def make_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask  # True = masked (ignored), False = attend

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        emb = self.embedding(x) * math.sqrt(self.d_model) + self.pos_encoding(pos)
        causal_mask = self.make_causal_mask(x.size(1), x.device)
        # For decoder-only, memory = embedding itself
        out = self.transformer(emb, emb, tgt_mask=causal_mask, memory_mask=causal_mask)
        return self.output(out)


# ---- Training ----
VOCAB_SIZE = 22  # 0=pad, 1-20=tokens, 21=EOS
D_MODEL = 128
SEQ_LEN = 10
BATCH_SIZE = 64
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SequenceReversalDataset(10000, SEQ_LEN, VOCAB_SIZE - 1)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MiniGPT(VOCAB_SIZE, D_MODEL).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
criterion = nn.CrossEntropyLoss(ignore_index=0)

print(f"Training MiniGPT on Sequence Reversal")
print(f"Device: {DEVICE}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        logits = model(src)  # [batch, seq_len, vocab_size]
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if epoch % 5 == 0:
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")

# ---- Evaluation ----
model.eval()
test_seq = torch.tensor([[3, 7, 1, 9, 5, 2, 8, 4, 6, 10]], device=DEVICE)
print(f"\nTest sequence:    {test_seq[0].tolist()}")
print(f"Expected output:  {list(reversed(test_seq[0].tolist()))}")

with torch.no_grad():
    logits = model(test_seq)
    predicted = logits.argmax(dim=-1)
    print(f"Predicted output: {predicted[0].tolist()}")
print("\n✅ MiniGPT training complete!")
```

---

## 📝 Detailed Summary of Key Formulas

| Component | Formula |
|---|---|
| Input | `x = TokenEmbedding(tokens) + PositionalEncoding(positions)` |
| Attention | `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V` |
| Multi-Head | `MHA(Q,K,V) = Concat(head₁,...,headₕ)Wₒ` |
| FFN | `FFN(x) = GELU(xW₁+b₁)W₂+b₂` |
| Residual | `Output = LayerNorm(x + Sublayer(x))` |
| Final | `P(next) = Softmax(Linear(x))` |

---

## 🎯 Mini-Quiz (10 Questions)

1. What are Q, K, V in the attention mechanism and what does each represent?
2. Why do we divide attention scores by √d_k before applying softmax?
3. What is the purpose of the causal mask in decoder-only models like GPT?
4. How does multi-head attention allow the model to capture multiple types of relationships?
5. What is the difference between a padding mask and a causal mask?
6. How many parameters does a single Multi-Head Attention block have in terms of d_model?
7. What is the role of the cross-attention mechanism in the encoder-decoder architecture?
8. Why is pre-layer normalization preferred over post-layer normalization in modern LLMs?
9. What is Grouped Query Attention (GQA) and why is it important for inference efficiency?
10. Trace the shape of a tensor through one complete encoder block, starting from [batch=2, seq_len=16, d_model=512].

---

## 🏋️ Assignments

### Assignment 4.1: Attention Visualization (Intermediate)
Using the attention visualization code from Lab 4.4:
- Load a pretrained BERT model from Hugging Face
- Run it on 3 different sentences
- Visualize which words each head attends to
- Write a paragraph explaining what patterns you observe in each head's attention

### Assignment 4.2: Transformer Ablation Study (Advanced)
Using the MiniGPT from Lab 4.5:
- Run experiments with different numbers of heads (1, 2, 4, 8)
- Run experiments with different numbers of layers (1, 2, 4, 6)
- Plot learning curves for each configuration
- Which configuration achieves lowest loss fastest? Why?

### Assignment 4.3: Custom Positional Encoding (Expert)
Instead of sinusoidal PE, implement **Rotary Position Embedding (RoPE)**:
- Research the RoPE paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- Implement `RotaryPositionalEmbedding` class
- Integrate it into the `MultiHeadAttention` class
- Compare with sinusoidal PE on a sequence length generalization task

---

## 📖 Further Reading & Resources

### Papers (Essential)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

### Blog Posts
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) — **Best visual explanation**
- [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/) — Code-first walkthrough
- [Transformers Explained Visually (Ketan Doshi)](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)

### Videos
- [Karpathy's "Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 2 hours, excellent
- [Stanford CS224N Lecture 9: Transformers](https://web.stanford.edu/class/cs224n/)

### Code Resources
- [Hugging Face Transformers Library](https://github.com/huggingface/transformers)
- [minGPT by Andrej Karpathy](https://github.com/karpathy/minGPT)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

---

## 🎯 Extended Mini-Quiz — Test Your Transformer Understanding

1. The attention formula is `softmax(QKᵀ/√dₖ)V`. Why do we divide by √dₖ? What goes wrong without it?

2. If the model dimension is d=512 and we use h=8 attention heads, what is the dimension per head (dₖ=dᵥ)? Why split into heads at all?

3. Explain the intuition behind Queries, Keys, and Values using the library/search analogy. What does Q represent? What does K represent? What does V represent?

4. In **decoder self-attention** (e.g., in GPT), why must we use a **causal mask** (masking future positions)? What would happen at inference if we didn't?

5. What is **cross-attention** and when is it used? Which Transformer variants use it? Which don't?

6. Positional encoding is added to token embeddings. Name two different approaches to positional encoding and explain a trade-off between them.

7. The FFN (Feed-Forward Network) in each Transformer layer expands to 4× the model dimension then projects back. What is the purpose of this expansion? What would happen if you simply had a single LINEAR layer instead?

8. BERT is encoder-only. GPT is decoder-only. T5 is encoder-decoder. For each of the following tasks, which architecture is most natural and why?
   - a) Named entity recognition (classify each token)
   - b) Machine translation (English → French)
   - c) Text generation / story writing
   - d) Sentence similarity scoring

9. The original Transformer uses **Pre-Norm** or **Post-Norm** layer normalisation? Which does GPT-3 use? What is the practical difference?

10. **Conceptual depth**: Transformers are sometimes called "in-context learners". What does this mean? How does the attention mechanism enable a model to adapt its computation based on the specific input, without any weight updates?

---

## 🏋️ Assignments

### Assignment 4.1: Attention From Scratch
Extend the `ImplementedTransformer` from the lab:
1. Remove the provided `scaled_dot_product_attention` and write your own
2. Add `attention_weights` as a return value from your attention function
3. Visualize the attention matrix for a small input sequence as a heatmap
4. What patterns do you see? Which tokens attend strongly to which others?

### Assignment 4.2: Encoder-Only vs Decoder-Only
Using Hugging Face `transformers`:
1. Load `bert-base-uncased` (encoder-only) and `gpt2` (decoder-only)
2. Pass the same sentence through both: "The cat sat on the mat"
3. Extract hidden states from both models and compare:
   - Shape of output tensors
   - L2 distance between token representations at different positions
4. Can you use BERT for text generation? Try it. What happens and why?

### Assignment 4.3: Scale and Parameters
1. Write a function `count_transformer_params(d_model, n_heads, n_layers, vocab_size, max_seq_len)` that analytically computes the total parameter count of a GPT-style Transformer
2. Verify against actual PyTorch model parameter counts for a small config
3. Compute parameters for GPT-2 (d=768, h=12, L=12), GPT-3 (d=12288, h=96, L=96)
4. Where are most parameters? Attention or FFN? Show the calculation.

### Assignment 4.4: Build a Mini Sentiment Classifier
Fine-tune a pre-trained Transformer for sentiment classification:
1. Use `distilbert-base-uncased` from Hugging Face
2. Add a classification head (Linear → Sigmoid) on top of the [CLS] token
3. Fine-tune on the SST-2 dataset for 3 epochs
4. Compare performance: random weights vs pre-trained weights
5. Try freezing the Transformer weights and only training the head — what accuracy do you get? Why?

---

## 💡 Glossary — Day 4

| Term | Definition |
|---|---|
| **Transformer** | Neural architecture using only attention and feed-forward layers; no recurrence |
| **Self-Attention** | Attention where Q, K, V all come from same sequence — each token attends to all others |
| **Query (Q)** | The search query — what this token is looking for |
| **Key (K)** | The index card — describes what each token offers |
| **Value (V)** | The actual retrieved content — what gets combined into the output |
| **Attention Score** | QKᵀ/√dₖ — raw score before softmax |
| **Attention Weight** | Softmaxed score — probability distribution over positions |
| **Multi-Head Attention** | Running H parallel attention mechanisms on projected subspaces |
| **Causal Mask** | Upper-triangular mask in decoder self-attention — prevents attending to future tokens |
| **Cross-Attention** | Q from decoder, K and V from encoder — enables reading encoder output |
| **Positional Encoding** | Injected signal enabling the model to know token positions |
| **Sinusoidal PE** | Original fixed positional encoding using sin/cos at different frequencies |
| **RoPE** | Rotary Position Embedding — used in LLaMA, GPT-NeoX; encodes relative position |
| **Layer Normalization** | Normalizes each token's feature vector to zero mean, unit variance |
| **Pre-Norm** | LayerNorm applied before sub-layer (modern variant — more stable) |
| **Post-Norm** | LayerNorm applied after residual add (original paper variant) |
| **Feed-Forward Network (FFN)** | Two linear layers with activation between them; applied per token independently |
| **Encoder-Only** | Transformer using only the encoder stack (BERT, RoBERTa) |
| **Decoder-Only** | Transformer using only the decoder stack (GPT, LLaMA, Gemma) |
| **Encoder-Decoder** | Full original architecture with both stacks (T5, BART) |
| **[CLS] Token** | Special classification token prepended to BERT input; output = sentence embedding |
| **[SEP] Token** | Separator token in BERT marking sentence boundaries |
| **Causal Language Model** | Predicts next token given all previous — GPT-style training objective |
| **Masked Language Model** | Predicts masked tokens given surrounding context — BERT-style objective |
| **dₖ** | Dimension of query/key vectors in attention; usually d_model / n_heads |
| **d_model** | Model embedding dimension (e.g., 768 for BERT-base) |
| **n_heads** | Number of attention heads (e.g., 12 for BERT-base) |

---

## ⚡ Day 4 Summary

```
Before today:                         After today:
"Attention is All You Need" ≈ magic  →  You understand every component
No idea what Q, K, V mean           →  Library analogy fully internalized
Transformer = black box             →  You traced every tensor shape
Never built one from scratch        →  You implemented one in PyTorch
BERT vs GPT seemed arbitrary        →  Architecture choice has clear reasoning
```

The Transformer is the single most important architecture in modern AI. Everything you build in Weeks 2–4 of this course runs on this foundation. Revisit this day whenever a concept feels unclear — it's worth reading multiple times.

---

## 🔗 Day 5 Preview

**Day 5 — Attention Mechanisms (Deep Dive)**

Now that you understand scaled dot-product attention, we go deeper:
- **Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)** — how LLaMA-3 and Gemma reduce memory while preserving quality
- **FlashAttention** — the IO-aware algorithm that makes attention 5–10× faster in practice
- **Sliding Window Attention** — how Mistral handles 32K contexts efficiently
- **The KV Cache** — the mechanism that makes autoregressive inference feasible at all
- **ALiBi, RoPE, and other PE approaches** — a comparison of modern position encodings

Day 5 is where Transformer theory becomes Transformer engineering.

> 💬 *"In theory there is no difference between theory and practice. In practice there is."*  
> — Yogi Berra (reinterpreted: Day 4 was theory. Day 5 is where it gets real.)

See you on Day 5. 🚀

---
*Day 4 of 30 | Week 1: Foundations | GenAI Mastery Course*

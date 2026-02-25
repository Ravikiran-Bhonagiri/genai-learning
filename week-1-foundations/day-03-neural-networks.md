# Day 03 — Neural Network Architectures: The Road to Transformers

> **Week 1 | Foundations** | ⏱️ Estimated Time: 4–5 hours | 🔥 Difficulty: Intermediate

---

## 🤔 A Puzzle to Open Your Mind

Here's a fascinating problem. Look at these two grids:

```
Grid A:                Grid B:
1 0 1 0 0             0 1 0 1 0
0 1 0 0 0             1 0 1 0 1
1 1 1 0 0             0 1 0 1 0
0 1 0 0 0             1 0 1 0 1
1 0 1 0 0             0 1 0 1 0
```

Grid A represents the letter "A" in a 5×5 pixel grid. Grid B is a checkerboard.

If you show these to a fully connected network (MLP), it sees 25 numbers: `[1, 0, 1, 0, 0, 0, 1, 0, 0, 0, ...]`. It has no idea which numbers are *neighbours*. It treats pixel (0,0) and pixel (4,4) as equally related.

But *you* know immediately that "A" has edges, curves, symmetry. You recognise those patterns because your visual system **processes nearby pixels together first** before combining them into higher-level features.

This insight — process local structure first, then combine — is exactly what **Convolutional Neural Networks (CNNs)** do. And it's one of three major architectural innovations we study today.

Today's journey: CNNs → RNNs → LSTMs → why they all eventually gave way to Transformers.

By the end, you'll understand not just *what* each architecture does but *why* it was invented, *what problem* it solved, and *why* we needed to go further.

---

## 🎯 Learning Objectives

By the end of today you will:

- [ ] Explain how convolutional filters detect local spatial features
- [ ] Trace a tensor through a CNN: input image → conv → pool → flatten → output
- [ ] Understand why sequential data requires different architectures than tabular data
- [ ] Explain the vanishing gradient problem in RNNs and why it matters
- [ ] Describe exactly how LSTM gates solve the vanishing gradient problem
- [ ] Know why all of these architectures eventually yielded to Transformers
- [ ] Build a CNN image classifier and an LSTM text generator in PyTorch

---

## 📚 Part 1: Convolutional Neural Networks — Eyes That Learn (45 min)

### 3.1 The Biological Motivation

In 1959, neuroscientist David Hubel and Torsten Wiesel discovered something remarkable about cat visual cortex: individual neurons respond to *specific orientations* of edges. One neuron fires for horizontal lines. Another for diagonal lines at 45°. Another for moving edges.

The visual cortex is organised *hierarchically*:
- Early neurons → simple features (edges, orientations)
- Middle neurons → combinations (curves, corners)
- Late neurons → complex shapes (faces, hands)

CNNs mimic this hierarchy. Each layer learns increasingly abstract representations.

### 3.2 The Convolution Operation — The Core Trick

A convolutional layer applies a set of **filters** (also called kernels) to an input. Each filter is a small matrix (e.g., 3×3) that slides across the input, computing a dot product at each position.

**Example — Edge detection filter:**
```
Input (7×7 image):          Filter (3×3):
1 1 1 1 1 1 1               1  0 -1
1 1 1 1 1 1 1               1  0 -1
1 1 1 0 0 0 0               1  0 -1
1 1 1 0 0 0 0    Convolve → 
1 1 1 0 0 0 0
1 1 1 0 0 0 0
1 1 1 0 0 0 0
```

This specific 3×3 filter detects *vertical edges*. Where pixel values change sharply from left to right, the filter produces a large positive value. In flat regions (all 1s or all 0s), it produces zero.

**The sliding window:**
```
Position (0,0): filter overlaps top-left 3×3 patch → dot product → one output value
Position (0,1): filter moves right by stride=1 → next output value
...
```

**Key insight:** Each filter is *shared across all positions*. The same edge-detection filter scans the entire image. This is **parameter sharing** — a 3×3 filter has only 9 learnable weights regardless of image size (vs. a fully-connected layer that would need `image_size × image_size` weights).

For a 224×224 RGB image with a 3×3 conv layer with 64 filters:
- FC layer: 224 × 224 × 3 × output_size = 150K+ parameters per output neuron
- Conv layer: 3 × 3 × 3 × 64 = **1,728 parameters total** 🚀

### 3.3 Multiple Filters = Multiple Feature Detectors

In practice, we apply **many filters simultaneously**:
- Filter 1 → horizontal edges
- Filter 2 → vertical edges
- Filter 3 → diagonal edges
- Filter 4 → curved surfaces
- ... (64 or 256 filters in practice)

Each filter produces one **feature map** — a new 2D grid showing *where in the image* that feature is present.

After N filters, input shape `[H, W, C]` becomes `[H', W', N]` where:
- `H', W'` = spatial dimensions (may shrink with padding)
- `N` = number of feature maps = one per filter

### 3.4 Pooling — Spatial Compression

After convolution, feature maps are often large. **Pooling** reduces spatial size while keeping important features.

**Max Pooling (most common):**
```
Input 4×4:          MaxPool (2×2, stride 2): → 2×2
1 3 2 1             
0 5 6 2      →     5  6
2 0 1 3            2  3
1 2 4 3
```

Takes the maximum value in each 2×2 window. Reduces spatial size by 2× while keeping the most prominent activation.

**Why pooling works:** It provides **spatial invariance** — small shifts and distortions in the input don't change the output. A "3" that's slightly tilted looks similar after max-pooling.

### 3.5 A Full CNN Architecture

```
INPUT IMAGE [batch, 3, 224, 224]  (RGB image)
     ↓
Conv2D(3→32, kernel=3, padding=1)  → [batch, 32, 224, 224]
ReLU
MaxPool(2×2)                       → [batch, 32, 112, 112]
     ↓
Conv2D(32→64, kernel=3, padding=1) → [batch, 64, 112, 112]
BatchNorm + ReLU
MaxPool(2×2)                       → [batch, 64, 56, 56]
     ↓
Conv2D(64→128, kernel=3)           → [batch, 128, 56, 56]
BatchNorm + ReLU
MaxPool(2×2)                       → [batch, 128, 28, 28]
     ↓
AdaptiveAvgPool(1×1)               → [batch, 128, 1, 1]
Flatten                            → [batch, 128]
     ↓
Linear(128 → 256) + ReLU
Dropout(0.5)
Linear(256 → 10)                   → [batch, 10]  (10-class output)
Softmax
```

**What each layer learns (roughly):**
- Early conv layers → edges, textures, simple patterns
- Middle conv layers → shapes, object parts
- Late conv layers → object-level features (faces, wheels, fur)
- FC layers → category-level decisions

### 3.6 The ResNet Revolution — Skip Connections

By 2015, researchers found that deeper CNNs (50+ layers) were actually *worse* than shallower ones. Not because of overfitting — even training accuracy was worse. This is the **degradation problem**.

**The fix:** **Residual connections** (He et al., 2015):

```
Without residual:  y = F(x)         (transform x, output result)
With residual:     y = F(x) + x     (transform x, add ORIGINAL x back)
```

If F(x) ≈ 0 (the layer can't improve), `y ≈ x` — the layer learned the identity function. The gradient can flow directly through the `+x` shortcut, preventing vanishing gradients.

ResNet-152 with 152 layers achieved human-level image classification on ImageNet. This "skip connection" idea also appears in Transformers (every layer has a residual connection around attention and FFN).

### 3.7 CNNs in GenAI

You might be thinking: "Why study CNNs in a GenAI course?" Three critical reasons:

**1. Diffusion models use U-Net (CNN-based):**
The denoising network in Stable Diffusion is a U-Net — an encoder-decoder CNN architecture. It processes images at multiple resolutions simultaneously.

**2. Vision Transformers (ViT) start with CNN-like patch embedding:**
```
Image [224×224×3] → Split into 16×16 patches → 196 patches → treat as sequence tokens
```
The Transformer then processes these patches like a sequence.

**3. CLIP encodes images with CNNs:**
OpenAI's CLIP (the model that enables "describe an image in text") uses a CNN to encode images into the same embedding space as text.

---

## 📚 Part 2: Recurrent Neural Networks — Memory for Sequences (30 min)

### 3.8 The Problem: Order Matters

Standard neural networks are "memoryless" — each input is processed independently. But language is *sequential*. Order matters enormously:

```
"The dog chased the cat" ≠ "The cat chased the dog"
```

Both have the same words. Very different meanings.

We need networks that:
1. Process tokens one at a time (in order)
2. Maintain a *memory* of what came before
3. Use that memory to inform current and future predictions

This is what RNNs were designed for.

### 3.9 How an RNN Works

An RNN has a special component: the **hidden state** `h_t` — a vector that carries information forward through time.

```
       h₀ (initial hidden state, usually zeros)
        ↓
x₁ → [RNN Cell] → h₁ → [RNN Cell] → h₂ → [RNN Cell] → h₃
                    ↑                  ↑
                   x₂                 x₃

Output at each step: ŷₜ = W_hy · hₜ + b_y
Hidden state update: hₜ = tanh(W_hh · hₜ₋₁ + W_xh · xₜ + b_h)
```

The *same* weights (`W_hh`, `W_xh`) are used at every timestep. This is parameter sharing across time — analogous to CNN's parameter sharing across space.

**Example — character prediction:**
```
Input:  "The cat sa"... → predict next character
h₁ = encode("T")
h₂ = encode("Th") 
h₃ = encode("The")
...
h₁₀ = encode("The cat sa") → predicts "t" (next char in "sat")
```

### 3.10 The Vanishing Gradient Problem — RNNs' Fatal Flaw

Training an RNN uses **Backpropagation Through Time (BPTT)**: you unroll the RNN across timesteps and backpropagate through all of them.

For a 100-token sequence:
```
Loss at step 100 → gradient flows back:
  through step 99 → × W_hh
  through step 98 → × W_hh
  ...
  through step 1  → × W_hh (×99 times!)

If ||W_hh|| < 1: gradient → 0 (vanishes)
If ||W_hh|| > 1: gradient → ∞ (explodes)
```

In practice, vanilla RNNs struggle to keep useful information for more than ~20 timesteps. Ask an RNN to model the subject of a sentence that started 50 tokens ago — it will likely have "forgotten" it.

This limitation is fundamental. No amount of training data or hyperparameter tuning fixes it.

---

## 📚 Part 3: LSTM — Engineering Around the Fundamental Flaw (30 min)

### 3.11 The LSTM Solution — Gated Memory

Hochreiter & Schmidhuber (1997) introduced the **Long Short-Term Memory** network. The key insight: add a separate **cell state** `c_t` that acts like a "memory highway" — information can flow through it relatively unchanged across many timesteps.

```
LSTM State at each timestep:
  hₜ = hidden state (short-term memory — dense vector)
  cₜ = cell state  (long-term memory — the "highway")
```

Three gates determine what flows through:

**Forget Gate** — "What should I erase from long-term memory?"
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Values: 0 = completely forget, 1 = completely remember
```

**Input Gate** — "What new information should I write to long-term memory?"
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   # How much to write
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  # What to write 
```

**Output Gate** — "What aspect of the cell state should I expose as output?"
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(c_t)
```

**Cell State Update:**
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
          ↑ keep old   ↑ add new
```

The cell state `c_t` passes through *multiplicative gates* (not tanh/sigmoid on itself). Gradients flow more directly, reducing vanishing gradient.

**Intuitive example:**

Processing "The cat, which had spotted several mice in the barn, finally...")

```
"The cat" 
  → Input gate: write "subject=cat" to cell state
  → Forget gate: keep everything

"which had spotted several mice in the barn"  
  → Forget gate: context clause, keep "subject=cat" in cell state
  → Input gate: write "interlude verb=spotted" (minor update)
  
"finally..."
  → Output gate: expose "subject=cat" from cell state to predict verb
```

This is how LSTMs handle long-range dependencies that vanilla RNNs cannot.

### 3.12 GRU — LSTM's Lighter Sibling

Cho et al. (2014) introduced the **Gated Recurrent Unit** — a simplified LSTM with two gates instead of three:
- **Update gate**: combines forget and input gates
- **Reset gate**: determines how much of the past hidden state to use

GRUs have ~25% fewer parameters than LSTMs and often match or exceed LSTM performance on smaller datasets.

```python
# PyTorch: switching between RNN, LSTM, GRU is one line
rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
```

### 3.13 Why Transformers Won

Despite LSTMs being a significant step forward, they still have critical limitations:

| Limitation | Description | Impact |
|---|---|---|
| **Sequential processing** | Token N needs token N-1 to finish | Can't parallelise training across time steps |
| **Bounded memory** | Cell state is a fixed-size vector | Information bottleneck for very long sequences |
| **Short effective range** | Even LSTMs struggle >500 tokens | Long documents are hard |
| **Slow training** | Sequential dependency = wall-clock bottleneck | 10-100× slower than Transformers on same data |

The Transformer (2017) solved all four:
- Full parallelism: every position processed simultaneously via self-attention
- Unbounded memory: every token attends directly to every other token
- No effective range limit: position 1 attends to position 10,000 equally easily
- Fast training: parallelises across sequence and across GPUs

**The historical arc:**
```
1980s: Vanilla RNNs — sequential, gradient-prone
1997:  LSTMs — gated memory, better long-range
2014:  GRUs — lighter LSTMs
2015:  Attention + RNNs — first attention in NLP (Bahdanau)
2017:  Transformers — attention replaces recurrence entirely
2018+: BERT, GPT, T5 — Transformer dominates NLP
2020+: ViT — Transformers take over vision too
2024+: Mamba, RWKV — linear-time alternatives emerging
```

---

## 💻 Lab Time: Build All Three Architecture Types (90 min)

### Lab 3.1: CNN for Image Classification (MNIST)

```python
# lab_03_01_cnn_classifier.py
"""
Build a CNN to classify handwritten digits (MNIST).
We'll visualize what the filters learn after training.
Classic first CNN experiment — every ML engineer has done this.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("Lab 3.1: CNN for MNIST Digit Classification")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---- Dataset ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_data = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)

print(f"Training samples: {len(train_data):,}")
print(f"Test samples:     {len(test_data):,}")
print(f"Classes:          {train_data.classes}")


# ---- Model ----
class MNIST_CNN(nn.Module):
    """
    A 3-block CNN for MNIST — a textbook example.
    Input: [batch, 1, 28, 28]  (grayscale images)
    """
    def __init__(self):
        super().__init__()
        
        # Feature extractor: 3 convolutional blocks
        self.features = nn.Sequential(
            # Block 1: 1 → 32 channels, maintain spatial size
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [B, 32, 28, 28]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # [B, 32, 28, 28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # [B, 32, 14, 14]
            nn.Dropout2d(0.25),
            
            # Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),              # [B, 64, 12, 12]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # [B, 64, 6, 6]
            nn.Dropout2d(0.25),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # → [B, 64*6*6] = [B, 2304]
            nn.Linear(64 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)              # 10 digit classes
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MNIST_CNN().to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# Print layer shapes by tracing a dummy input
dummy = torch.randn(1, 1, 28, 28).to(device)
print("\nTensor shapes through feature layers:")
x = dummy
for i, layer in enumerate(model.features):
    x = layer(x)
    if hasattr(layer, 'weight'):
        print(f"  After {layer.__class__.__name__}: {list(x.shape)}")


# ---- Training ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += nn.CrossEntropyLoss()(out, y).item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
    return correct / total, total_loss / len(loader)

print("\n🏃 Training CNN...")
print(f"{'Epoch':<8} {'Train Loss':<14} {'Test Acc':<12} {'Test Loss'}")
print("-" * 50)

for epoch in range(1, 8):  # 7 epochs is plenty for MNIST
    model.train()
    running_loss = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    test_acc, test_loss = evaluate(model, test_loader)
    scheduler.step()
    
    print(f"{epoch:<8} {train_loss:<14.4f} {test_acc:<12.2%} {test_loss:.4f}")

print(f"\n✅ Final test accuracy: {test_acc:.2%}")
print(f"   (Human-level parity: ~99.0% — CNN achieves ~99.3% at full training)")


# ---- Visualize Learned Filters ----
print("\n📊 Visualizing first-layer filters...")

first_conv_weights = model.features[0].weight.detach().cpu()  # [32, 1, 3, 3]
print(f"First conv layer filter shape: {first_conv_weights.shape}")
print("(32 filters, each operating on 1 channel, 3×3 pixels)")

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
fig.suptitle("Learned First-Layer Filters (3×3)", fontsize=13)
for i, ax in enumerate(axes.flat):
    if i < 32:
        f = first_conv_weights[i, 0].numpy()
        ax.imshow(f, cmap='RdBu', interpolation='nearest',
                  vmin=f.min(), vmax=f.max())
        ax.axis('off')
        ax.set_title(f'#{i+1}', fontsize=7)

plt.tight_layout()
plt.savefig('cnn_filters.png', dpi=120)
plt.show()
print("✅ Filter visualization saved to cnn_filters.png")
print("   Notice: some filters look like edge detectors, others detect blobs or curves!")


# ---- Visualize Misclassified Examples ----
print("\n🔍 Finding misclassified digits...")
model.eval()
wrong_examples = []
with torch.no_grad():
    for x, y in test_loader:
        x, y_true = x.to(device), y.to(device)
        out = model(x)
        preds = out.argmax(1)
        wrong_mask = preds != y_true
        if wrong_mask.any() and len(wrong_examples) < 16:
            wrong_x = x[wrong_mask].cpu()
            wrong_pred = preds[wrong_mask].cpu()
            wrong_true = y_true[wrong_mask].cpu()
            for wx, wp, wt in zip(wrong_x, wrong_pred, wrong_true):
                wrong_examples.append((wx, wp.item(), wt.item()))
                if len(wrong_examples) >= 16:
                    break

if wrong_examples:
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    fig.suptitle("Misclassified Digits — What Does the CNN Find Confusing?")
    for ax, (img, pred, true) in zip(axes.flat, wrong_examples):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'True:{true} Pred:{pred}', fontsize=9, color='red')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('cnn_mistakes.png', dpi=120)
    plt.show()
    print("✅ Mistake grid saved to cnn_mistakes.png — study what the model finds hard!")
```

---

### Lab 3.2: Vanilla RNN vs LSTM — Sequence Tasks

```python
# lab_03_02_rnn_vs_lstm.py
"""
Direct comparison of vanilla RNN vs LSTM on a sequence task.
Task: predict next character in text.
Watch how LSTM outperforms RNN, especially on longer passages.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

print("=" * 60)
print("Lab 3.2: RNN vs LSTM — Character-Level Language Modeling")
print("=" * 60)

# ---- Data Preparation ----
text = """
Generative AI is not just a tool. It is a new medium.
Just as the printing press democratized writing, and cameras democratized photography,
Generative AI is democratizing the creation of all knowledge artifacts.
The question is not whether this changes everything. It already has.
The question is whether we will use it to amplify human creativity or replace it.
History suggests: those who use new tools thoughtfully thrive.
Those who ignore them get left behind. Those who misuse them cause harm.
We are at that inflection point right now. Choose carefully. Learn deeply.
Build responsibly. The 30 days you spend on this course matter.
"""

# Character-level vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

# Encode text
encoded = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
print(f"Vocabulary size: {vocab_size} characters")
print(f"Text length:     {len(text)} characters ({len(encoded)} tokens)")
print(f"Vocabulary:      {chars[:20]}...")

# Build training sequences
SEQ_LEN = 60
BATCH_SIZE = 16

# Create overlapping sequences
sequences, targets = [], []
for i in range(0, len(encoded) - SEQ_LEN - 1, 3):  # stride=3 for efficiency
    sequences.append(encoded[i:i+SEQ_LEN])
    targets.append(encoded[i+1:i+SEQ_LEN+1])

X = torch.stack(sequences)
y = torch.stack(targets)
print(f"Training sequences: {len(sequences)} sequences × {SEQ_LEN} chars")

dataset = torch.utils.data.TensorDataset(X, y)
loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ---- Model Definitions ----
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, n_layers=2, rnn_type='LSTM'):
        super().__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_class(
            embed_dim, hidden_size, n_layers,
            batch_first=True,
            dropout=0.3 if n_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        params = sum(p.numel() for p in self.parameters())
        print(f"  {rnn_type}: {params:,} parameters")
    
    def forward(self, x, hidden=None):
        embed = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embed, hidden)
        logits = self.fc(self.dropout(out))
        return logits, hidden
    
    def generate(self, seed_text, length=200, temperature=0.8):
        """Generate text autoregressively."""
        self.eval()
        chars_out = list(seed_text)
        
        # Encode seed
        x = torch.tensor([[char2idx.get(c, 0) for c in seed_text]], dtype=torch.long)
        hidden = None
        
        with torch.no_grad():
            _, hidden = self.rnn(self.dropout(self.embedding(x)), hidden)
            
            for _ in range(length):
                last = torch.tensor([[char2idx.get(chars_out[-1], 0)]], dtype=torch.long)
                embed = self.dropout(self.embedding(last))
                out, hidden = self.rnn(embed, hidden)
                logits = self.fc(self.dropout(out[:, -1, :]))
                
                # Sample with temperature
                probs = torch.softmax(logits / temperature, dim=-1)
                next_char_idx = torch.multinomial(probs, 1).item()
                chars_out.append(idx2char[next_char_idx])
        
        return ''.join(chars_out)


# ---- Train Both Models ----
def train_model(model_type, n_epochs=40):
    print(f"\n{'='*40}")
    print(f"Training {model_type}...")
    print(f"{'='*40}")
    
    model = CharRNN(vocab_size, rnn_type=model_type)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    start_time = time.time()
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0
        
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            
            # Reset hidden state each batch
            logits, _ = model(x_batch)
            loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
            loss.backward()
            
            # Clip gradients (crucial for RNNs!)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        # Perplexity = exp(cross-entropy) — standard LM metric
        perplexity = np.exp(avg_loss)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f} | Perplexity={perplexity:.2f}")
    
    elapsed = time.time() - start_time
    return model, losses, elapsed

# Train
rnn_model,  rnn_losses,  rnn_time  = train_model('RNN')
lstm_model, lstm_losses, lstm_time = train_model('LSTM')
gru_model,  gru_losses,  gru_time  = train_model('GRU')


# ---- Compare ----
print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)

def evaluate_model(model, name):
    final_loss = min(model.rnn.weight_ih_l0.abs().mean().item(), 1.0)  # proxy
    return name

print(f"\n{'Model':<8} {'Final Loss':<15} {'Perplexity':<15} {'Train Time'}")
print("-" * 55)
for name, losses, t in [('RNN', rnn_losses, rnn_time), 
                         ('LSTM', lstm_losses, lstm_time),
                         ('GRU', gru_losses, gru_time)]:
    final = losses[-1]
    ppl = np.exp(final)
    print(f"{name:<8} {final:<15.4f} {ppl:<15.2f} {t:.1f}s")


# ---- Generate Text ----
print("\n" + "="*60)
print("TEXT GENERATION COMPARISON")
print("="*60)
seed = "Generative AI"

for name, model in [('RNN', rnn_model), ('LSTM', lstm_model), ('GRU', gru_model)]:
    print(f"\n--- {name} (temperature=0.7) ---")
    generated = model.generate(seed, length=150, temperature=0.7)
    print(generated)

print("""
📝 Observation guide:
  - RNN text tends to lose coherence after ~20-30 characters
  - LSTM maintains longer-range structure (sentence boundaries, themes)
  - GRU is typically similar to LSTM with slightly faster training
  - All generate plausible text — but LSTM is most coherent
""")
```

---

### Lab 3.3: Understanding the Vanishing Gradient

```python
# lab_03_03_vanishing_gradients.py
"""
Empirically demonstrate the vanishing gradient problem.
Watch gradient magnitudes shrink through layers.
This explains exactly why we need LSTMs and Transformers.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Lab 3.3: Vanishing Gradient Demonstration")
print("=" * 60)

def get_gradient_norms(model, inputs, targets):
    """Run backprop and collect gradient norms at each layer."""
    model.zero_grad()
    
    output, _ = model(inputs)
    loss = nn.CrossEntropyLoss()(output.reshape(-1, output.shape[-1]), targets.reshape(-1))
    loss.backward()
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    return grad_norms, loss.item()


# ---- Setup ----
vocab = 50
seq_len = 50
batch = 8
hidden = 64
n_layers = 4

x = torch.randint(0, vocab, (batch, seq_len))
y = torch.randint(0, vocab, (batch, seq_len))

# Test different RNN types
results = {}

for rnn_type in ['RNN', 'GRU', 'LSTM']:
    model = nn.Sequential()  # placeholder — build RNN directly
    
    class SeqModel(nn.Module):
        def __init__(self, rnn_type):
            super().__init__()
            self.embed = nn.Embedding(vocab, 32)
            rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[rnn_type]
            self.rnn = rnn_class(32, hidden, n_layers, batch_first=True)
            self.head = nn.Linear(hidden, vocab)
        
        def forward(self, x, h=None):
            e = self.embed(x)
            out, h = self.rnn(e, h)
            return self.head(out), h
    
    m = SeqModel(rnn_type)
    grads, loss = get_gradient_norms(m, x, y)
    
    # Extract gradient norms for each layer
    layer_grads = {}
    for name, norm in grads.items():
        if 'rnn.weight_ih' in name:
            layer = int(name.split('_l')[-1])
            layer_grads[f'Layer {layer}'] = norm
    
    results[rnn_type] = layer_grads
    print(f"\n{rnn_type} gradient norms (loss={loss:.4f}):")
    for layer_name in sorted(layer_grads.keys()):
        norm = layer_grads[layer_name]
        bar = '█' * int(norm * 20) if norm * 20 < 40 else '█' * 40 + '→'
        print(f"  {layer_name}: {norm:.6f}  {bar}")


# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'RNN': '#e74c3c', 'GRU': '#3498db', 'LSTM': '#2ecc71'}

for rnn_type, layer_grads in results.items():
    if layer_grads:
        layers = sorted(layer_grads.keys())
        norms = [layer_grads[l] for l in layers]
        ax.bar(
            [i + list(results.keys()).index(rnn_type) * 0.25 for i in range(len(layers))],
            norms,
            width=0.25,
            label=rnn_type,
            color=colors[rnn_type],
            alpha=0.8
        )

ax.set_xlabel('Layer')
ax.set_ylabel('Gradient Norm (higher = bigger signal)')
ax.set_title(f'Gradient Norms per Layer — {n_layers}-Layer Networks\n(Higher later layers = less vanishing gradient)')
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('gradient_norms.png', dpi=120)
plt.show()

print("""
✅ Key observations:
  - RNN: gradient norms drop significantly in earlier layers (vanishing gradient!)
  - LSTM: gradient norms are more uniform across layers (gated memory helps)
  - GRU: similar to LSTM, slightly different profile

This is the empirical manifestation of what Hochreiter & Schmidhuber (1997) proved theoretically.
""")
```

---

### Lab 3.4: Architecture Comparison — Visualizing Model Structures

```python
# lab_03_04_architecture_summary.py
"""
A unified script to compare CNN, RNN, and LSTM model architectures:
their parameter counts, inference speed, and use cases.
"""
import torch
import torch.nn as nn
import time

print("=" * 70)
print("Architecture Comparison: CNN vs RNN vs LSTM")
print("=" * 70)

# ---- Define benchmark models ----

class SimpleMLP(nn.Module):
    """Baseline: fully-connected for tabular data."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x.flatten(1))

class SimpleCNN(nn.Module):
    """CNN: best for spatial data."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*7*7, 10))
    def forward(self, x): return self.head(self.features(x))

class SimpleRNN(nn.Module):
    """Vanilla RNN: simple sequence model."""
    def __init__(self, vocab=50, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, 64)
        self.rnn = nn.RNN(64, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)
    def forward(self, x): 
        out, _ = self.rnn(self.embed(x))
        return self.head(out[:, -1])  # final hidden state

class SimpleLSTM(nn.Module):
    """LSTM: better sequence model."""
    def __init__(self, vocab=50, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, 64)
        self.lstm = nn.LSTM(64, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, vocab)
    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        return self.head(out[:, -1])

# ---- Benchmark ----
models = {
    'MLP (tabular)': (SimpleMLP(), torch.randn(32, 1, 28, 28)),
    'CNN (images)':  (SimpleCNN(), torch.randn(32, 1, 28, 28)),
    'RNN (text)':    (SimpleRNN(), torch.randint(0, 50, (32, 100))),
    'LSTM (text)':   (SimpleLSTM(), torch.randint(0, 50, (32, 100))),
}

print(f"\n{'Architecture':<20} {'Parameters':<15} {'Inference (ms)':<18} {'Best Use Case'}")
print("-" * 80)

for name, (model, sample_input) in models.items():
    params = sum(p.numel() for p in model.parameters())
    
    # Time inference
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)  # warmup
        start = time.time()
        for _ in range(100):
            _ = model(sample_input)
        elapsed_ms = (time.time() - start) / 100 * 1000
    
    use_cases = {
        'MLP (tabular)': 'Tables, embeddings',
        'CNN (images)':  'Images, spectrograms',
        'RNN (text)':    'Short sequences (<50)',
        'LSTM (text)':   'Sequences up to ~500',
    }
    
    print(f"{name:<20} {params:<15,} {elapsed_ms:<18.2f} {use_cases[name]}")

print("""
And for comparison — Transformer (covered on Day 4):
  Parameters: flexible (110M for BERT-base, 175B for GPT-3)
  Best Use: any sequence task, especially long-range dependencies
  Key advantage: full parallelism, unlimited effective context
""")
```

---

## 🎯 Mini-Quiz

1. A 3×3 convolutional filter with 1 input channel and 64 output channels has how many learnable parameters? Show your work.

2. You have an input feature map of size [batch=8, channels=32, height=56, width=56]. After a MaxPool2d(2,2) layer, what is the output shape?

3. Why does parameter sharing in CNNs make them efficient compared to fully-connected layers for image data?

4. What's the critical difference between how an RNN and an LSTM handles information from 100 timesteps ago?

5. In an LSTM, what does the **forget gate** specifically control? What would it mean if all forget gate values were 0? What about all 1s?

6. Why can't vanilla RNNs be efficiently parallelized during training? How does the Transformer architecture solve this?

7. ResNet introduced skip connections. GPT also uses skip connections (residual connections around attention layers). What common problem do they both solve?

8. For each task below, say whether you'd use a CNN, RNN, LSTM, or Transformer in 2024:
   - a) Classifying chest X-rays
   - b) Translating English to French (long documents)
   - c) Generating music melodies (1-second samples)
   - d) Detecting spam emails

9. Explain perplexity. A model with perplexity 10 vs perplexity 100 — which is better, and what does it mean intuitively?

10. **Reflection**: Looking at the historical timeline (RNN → LSTM → GRU → Attention → Transformer), what does this suggest about how AI research progresses? Do you see a similar pattern in other fields?

---

## 🏋️ Assignments

### Assignment 3.1: CNN Filter Visualisation
Train the MNIST CNN from Lab 3.1 fully (10 epochs). Then:
1. Extract the first conv layer weights and visualise all 32 filters as images
2. Pass a few MNIST test images through only the first conv+relu block
3. Visualise the 32 resulting feature maps for each input image
4. Write a 200-word description: what patterns does each feature map capture?

### Assignment 3.2: LSTM for Sentiment Analysis
Build an LSTM-based sentiment classifier:
1. Load the IMDB dataset using `torchtext` or `datasets` from HuggingFace
2. Tokenise at word level with a vocabulary of ~10,000 most common words
3. Build: Embedding → LSTM(2 layers) → Linear → Sigmoid
4. Train for 5 epochs, report accuracy on test set
5. Compare: how does validation accuracy change if you switch from LSTM to GRU?

### Assignment 3.3: Build a Mini-Language Model
Using the text generation LSTM from Lab 3.2:
1. Train on something longer — download a small book from Project Gutenberg
2. After training, generate 500-character samples at temperatures: 0.3, 0.7, 1.0, 1.5
3. Describe what happens qualitatively at each temperature extreme
4. Try generating with a seed you control. Does the model capture the author's style?

---

## 📖 Further Reading

### Essential
- [Christopher Olah: Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) — THE definitive visual guide, read it twice
- [CS231n: Convolutional Networks](http://cs231n.github.io/convolutional-networks/) — Stanford's authoritative reference
- [Andrej Karpathy: The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — Classic post, still magical

### Papers
- [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [Learning Phrase Representations using RNN Encoder-Decoder (Cho et al., 2014)](https://arxiv.org/abs/1406.1078) — GRU paper

### Interactive Tools
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — Interactive browser-based CNN visualization
- [RNN Playground](https://distill.pub/2019/memorization-in-rnns/) — Distill.pub visualization of what RNNs memorize

---

## 💡 Glossary — Day 3

| Term | Definition |
|---|---|
| **Convolution** | Sliding filter across input, computing dot products at each position |
| **Kernel / Filter** | Learnable weight matrix applied via convolution |
| **Feature Map** | Output of applying one filter to an input — activated spatial map |
| **MaxPooling** | Takes maximum value in each pooling window — reduces spatial size |
| **Parameter Sharing** | Same filter weights reused across all positions in conv layer |
| **Receptive Field** | Input region that influences a particular feature map position |
| **Skip Connection** | Direct path bypassing one or more layers — `output = F(x) + x` |
| **RNN** | Neural network with recurrent hidden state for sequential data |
| **Hidden State** | Vector carried forward in time in RNN — encodes sequence memory |
| **BPTT** | Backpropagation Through Time — backprop applied to unrolled RNN |
| **Vanishing Gradient** | Gradients shrink to near-zero over long sequences — learning fails |
| **LSTM** | Long Short-Term Memory — RNN with gated cell state for long-range memory |
| **Cell State** | The "memory highway" in LSTM — separate from hidden state |
| **Forget Gate** | LSTM gate deciding what to erase from cell state |
| **Input Gate** | LSTM gate deciding what new information to add to cell state |
| **Output Gate** | LSTM gate deciding what to expose from cell state |
| **GRU** | Gated Recurrent Unit — simplified LSTM with two gates |
| **Perplexity** | `exp(cross-entropy)` — standard language model quality metric |
| **U-Net** | CNN encoder-decoder with skip connections — used in diffusion models |

---

## ⚡ Day 3 Summary

```
Before today:                         After today:
"I know CNNs exist"             →   You built one and visualized its filters
No understanding of RNN memory  →   You implemented and compared all 3 types
Vanishing gradients: a phrase   →   You empirically measured gradient norms
"LSTM solves everything"        →   You understand exactly WHY it was needed
                                    AND where it still falls short
```

**The key narrative of today:**
Every architecture in this course was invented to solve a specific problem. CNNs solved spatial locality. RNNs solved temporal ordering. LSTMs solved vanishing gradients. Each was a breakthrough *in its moment* — and each was eventually superseded.

Tomorrow we study the architecture that superseded them all.

---

## 🔗 Day 4 Preview — The Big One

Tomorrow is the conceptual centrepiece of the entire course: **The Transformer Architecture**.

In 2017, eight Google researchers published a paper called "Attention Is All You Need." They removed convolutions. They removed recurrence. They kept only one simple mechanism — **attention** — and scaled it.

The result: models could process sequences in *parallel*. Long-range dependencies were solved *by design*. Training could use all available GPU compute simultaneously.

Six years later, every frontier AI model — GPT-4, Claude, Gemini, LLaMA — runs on Transformers or their direct descendants.

Day 4 is where the foundations you've built this week finally click into place.

Come prepared. Come curious. This one changes how you see everything.

> 💬 *"I have not failed. I've just found 10,000 ways that won't work."* — Thomas Edison  
> (Apply this to your experiments with temperature, LSTMs, and vanishing gradients today.)

See you on Day 4. 🚀

---
*Day 3 of 30 | Week 1: Foundations | GenAI Mastery Course*

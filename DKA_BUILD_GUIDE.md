# Dynamic Kernel Attention (DKA): Complete Build Guide

## Table of Contents

1. [Overview](#1-overview)
2. [Core Idea](#2-core-idea)
3. [Architecture Specification](#3-architecture-specification)
4. [Full Model Architecture](#4-full-model-architecture)
5. [Training Configuration](#5-training-configuration)
6. [Complexity Analysis](#6-complexity-analysis)
7. [Experiments & Ablations](#7-experiments--ablations)
8. [Visualizations & Graphs](#8-visualizations--graphs)
9. [Project Structure](#9-project-structure)
10. [Implementation Notes](#10-implementation-notes)

---

## 1. Overview

**Dynamic Kernel Attention (DKA)** is a drop-in replacement for standard self-attention in transformers. Instead of computing global pairwise dot-product attention ($O(n^2)$), each token generates its own local convolutional kernel from its content embedding and applies it to a local window. The mechanism is:

- **Local** like a CNN (only looks at a fixed-size window)
- **Input-dependent** like attention (the kernel changes based on the token's content)
- **Linear in sequence length** ($O(n \cdot k \cdot d)$ instead of $O(n^2 \cdot d)$)

This is not an approximation of attention. It is a fundamentally different mechanism that serves the same architectural role.

### Prior Work Distinction

| Method | Mechanism | Our Difference |
|--------|-----------|----------------|
| Standard Attention | Global pairwise dot-product, $O(n^2)$ | Ours is local, $O(n)$, not an approximation |
| Conv2S (Facebook) | Static kernels + GLU gating | Our kernels are content-dependent, not static |
| Hyena | Long implicit convolutions parameterized by *position* | Our kernels are parameterized by *content embedding* |
| Dynamic Conv (Wu et al. 2019) | Content-dependent mixing weights over a fixed bank of static kernels | No kernel bank — our kernels live in a continuous space via direct projection |

The key novelty: **fully continuous per-token kernel generation with factored rank decomposition.** Each token directly projects its embedding into a convolutional kernel through a nonlinear MLP. No fixed kernel bank, no position-only parameterization.

---

## 2. Core Idea

Standard self-attention:

$$\text{Attn}(X) = \text{softmax}\left(\frac{XW_Q(XW_K)^T}{\sqrt{d_k}}\right) XW_V$$

Every token compares against every other token. On images, most comparisons are wasted — a pixel in the top-left almost never needs to attend to a pixel in the bottom-right.

DKA replaces this with: each token generates a convolutional kernel from its own embedding and applies it to a local window.

$$\text{DKA}_i(X) = \hat{K}(x_i) \circledast \mathcal{W}(x, i)$$

where $\hat{K}(x_i)$ is the generated kernel (a function of token $i$'s content) and $\mathcal{W}(x, i)$ is the local window around position $i$.

The generated kernel is the critical piece. The word "bank" next to "river" generates a different kernel than "bank" next to "account." The same positional context, different content → different feature extraction. This is context-dependent processing without global comparison.

---

## 3. Architecture Specification

### 3.1 Notation

| Symbol | Meaning |
|--------|---------|
| $n$ | Sequence length (number of tokens/patches) |
| $d$ | Model embedding dimension |
| $H$ | Number of attention heads |
| $d_h = d / H$ | Per-head dimension |
| $k_h$ | Kernel size for head $h$ |
| $R$ | Rank of factored kernel decomposition |
| $L$ | Number of transformer layers |
| $B$ | Batch size |

### 3.2 DKA Module (replaces one Multi-Head Self-Attention)

The DKA module takes input $X \in \mathbb{R}^{n \times d}$ and produces output $O \in \mathbb{R}^{n \times d}$.

#### Step 1: Head Projection

Project input into each head's subspace:

$$X_h = X W_h^{in} + b_h^{in}, \quad W_h^{in} \in \mathbb{R}^{d \times d_h}, \quad h = 1, \ldots, H$$

This is identical to computing $V$ in standard attention. $X_h \in \mathbb{R}^{n \times d_h}$.

#### Step 2: Factored Kernel Generation

For each token $i$ in head $h$, generate a convolutional kernel $\Delta K_i^h \in \mathbb{R}^{k_h \times d_h}$.

**Factored decomposition:** the kernel is the sum of $R$ rank-1 outer products. For each rank component $m = 1, \ldots, R$:

**Spatial component** (determines *where* in the window to look):

$$s_i^{h,m} = W_{2,s}^{h,m} \cdot \text{GELU}\!\left(W_{1,s}^{h,m} \cdot x_i^h + b_{1,s}^{h,m}\right) + b_{2,s}^{h,m} \in \mathbb{R}^{k_h}$$

where $W_{1,s}^{h,m} \in \mathbb{R}^{d_h \times d_h}$, $W_{2,s}^{h,m} \in \mathbb{R}^{d_h \times k_h}$ (note: intermediate dimension is $d_h$, output is $k_h$).

**Channel component** (determines *what features* to extract):

$$c_i^{h,m} = W_{2,c}^{h,m} \cdot \text{GELU}\!\left(W_{1,c}^{h,m} \cdot x_i^h + b_{1,c}^{h,m}\right) + b_{2,c}^{h,m} \in \mathbb{R}^{d_h}$$

where $W_{1,c}^{h,m} \in \mathbb{R}^{d_h \times d_h}$, $W_{2,c}^{h,m} \in \mathbb{R}^{d_h \times d_h}$.

**Full generated kernel** (sum of rank-1 outer products):

$$\Delta K_i^h = \sum_{m=1}^{R} s_i^{h,m} \otimes c_i^{h,m} \in \mathbb{R}^{k_h \times d_h}$$

where $\otimes$ denotes the outer product: $\left(s_i^{h,m} \otimes c_i^{h,m}\right)_{j,l} = s_{i,j}^{h,m} \cdot c_{i,l}^{h,m}$.

**Why factored:** directly generating a $(k_h, d_h)$ kernel requires a projection of size $d_h \to k_h \cdot d_h$. For $d_h = 32, k_h = 11$, that's $32 \to 352$. The factored version generates $s \in \mathbb{R}^{k_h}$ and $c \in \mathbb{R}^{d_h}$ separately — much smaller projections, and the rank-$R$ sum gives enough expressiveness.

**Why the GELU matters:** without the nonlinearity, the kernel generator is a linear function of $x_i^h$, and a linear function generating weights that get linearly applied collapses to a bilinear form with much less expressiveness. The GELU is what allows different inputs to produce genuinely different kernels.

#### Step 3: Residual Kernel Structure

The generated $\Delta K_i^h$ is used as a *correction* on a learned static base kernel, not as the kernel directly:

$$\hat{K}_i^h = K_{\text{base}}^h + \alpha_h \cdot \frac{\Delta K_i^h}{\|\Delta K_i^h\|_F + \epsilon}$$

where:
- $K_{\text{base}}^h \in \mathbb{R}^{k_h \times d_h}$ — a learned static kernel shared across all tokens, initialized with Kaiming uniform
- $\alpha_h \in \mathbb{R}$ — a **learned scalar** initialized to $0.01$, one per head
- $\|\cdot\|_F$ — Frobenius norm
- $\epsilon = 10^{-6}$ for numerical stability

**Why this matters:**
- At initialization, $\alpha_h \approx 0$, so $\hat{K}_i^h \approx K_{\text{base}}^h$ → the model starts as a static CNN (known to work)
- During training, $\alpha_h$ grows and the model gradually introduces token-dependent dynamics
- If the model decides dynamic kernels aren't helpful for a particular head, $\alpha_h$ stays near zero → graceful degradation to standard convolution
- The normalization prevents magnitude explosion in the generated kernels

#### Step 4: Local Window Extraction & Kernel Application

Extract the local window of $k_h$ tokens centered on position $i$:

$$\mathcal{W}_i^h = \left[x_{i - \lfloor k_h/2 \rfloor}^h, \ldots, x_{i}^h, \ldots, x_{i + \lfloor k_h/2 \rfloor}^h\right] \in \mathbb{R}^{k_h \times d_h}$$

Use zero-padding at sequence boundaries.

Apply the generated kernel:

$$o_i^h = \sum_{j=1}^{k_h} \hat{K}_{i,j}^h \odot \mathcal{W}_{i,j}^h = \text{einsum}(\texttt{'kd,kd->d'},\ \hat{K}_i^h,\ \mathcal{W}_i^h) \in \mathbb{R}^{d_h}$$

This is the analog of the attention-weighted value sum. In standard attention: $o_i = \sum_j \alpha_{ij} v_j$ (weighted sum of all tokens, weights from dot-product similarity). In DKA: weighted sum of local tokens, weights from the generated kernel.

**Batched version (critical for GPU efficiency):**

All $n$ windows are extracted simultaneously via an unfold operation. All $n$ kernels are applied simultaneously via a single einsum. The operation across all tokens is:

$$O_h = \text{einsum}(\texttt{'bnkd,bnkd->bnd'},\ \hat{K}^h,\ \mathcal{W}^h)$$

where $\hat{K}^h \in \mathbb{R}^{B \times n \times k_h \times d_h}$ and $\mathcal{W}^h \in \mathbb{R}^{B \times n \times k_h \times d_h}$.

One unfold + one einsum per head. No loops. Fully parallel.

#### Step 5: Multi-Head Output Projection

Concatenate head outputs and project:

$$O = \text{Concat}(o^1, o^2, \ldots, o^H) \cdot W^{out} + b^{out}$$

where $W^{out} \in \mathbb{R}^{d \times d}$. Standard multi-head projection.

### 3.3 Multi-Scale Head Assignment

Each head gets a different kernel size, providing multi-scale context as a natural consequence of the multi-head structure:

**For $H = 8$ heads:**

| Heads 1-2 | Heads 3-4 | Heads 5-6 | Heads 7-8 |
|-----------|-----------|-----------|-----------|
| $k = 3$ | $k = 5$ | $k = 7$ | $k = 11$ |

Two heads per scale. Small kernels → fine-grained local patterns (edges, character bigrams). Large kernels → broader context (textures, phrases).

### 3.4 Full DKA Transformer Block

$$Z = X + \text{DKA}(\text{LayerNorm}(X))$$

$$Y = Z + \text{FFN}(\text{LayerNorm}(Z))$$

where FFN is:

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

with $W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d}$.

Pre-norm formulation (LayerNorm before each sublayer). Dropout after DKA output and after FFN output.

---

## 4. Full Model Architecture

### 4.1 Image Model (CIFAR-10 / Tiny ImageNet)

**Input pipeline:**

1. Image $I \in \mathbb{R}^{H_{\text{img}} \times W_{\text{img}} \times 3}$
2. Split into $p \times p$ non-overlapping patches → $n = (H_{\text{img}} / p) \times (W_{\text{img}} / p)$ patches
3. Flatten each patch to $\mathbb{R}^{p^2 \cdot 3}$
4. Linear projection: $W_{\text{patch}} \in \mathbb{R}^{(p^2 \cdot 3) \times d}$
5. Add learned positional embeddings: $P \in \mathbb{R}^{n \times d}$

| Dataset | Image Size | Patch Size $p$ | Sequence Length $n$ |
|---------|-----------|----------------|---------------------|
| CIFAR-10 | 32×32 | 4 | 64 |
| Tiny ImageNet | 64×64 | 4 | 256 |

**Backbone:** $L$ stacked DKA Transformer Blocks.

**Classification head:**

$$\hat{y} = \text{Linear}\!\left(\text{AvgPool}\!\left(\text{LayerNorm}(X_L)\right)\right)$$

Global average pooling across the token dimension, then a linear classifier to $C$ classes. No CLS token.

**Model configurations:**

| Config | $d$ | $H$ | $L$ | $R$ | Kernel Sizes | Approx. Params |
|--------|-----|-----|-----|-----|--------------|----------------|
| DKA-Tiny | 128 | 4 | 6 | 2 | {3, 5, 7, 11} | ~1.5M |
| **DKA-Small** | **256** | **8** | **8** | **4** | **{3,3,5,5,7,7,11,11}** | **~8M** |
| DKA-Base | 384 | 12 | 12 | 4 | {3,3,3,5,5,5,7,7,7,11,11,11} | ~25M |

**Use DKA-Small as the primary model.** Match parameter count to baselines.

### 4.2 Text Model (AG News / WikiText-2)

**Input pipeline:**

1. Token IDs → Learned embedding table $E \in \mathbb{R}^{V \times d}$
2. Add learned positional embeddings $P \in \mathbb{R}^{n_{\max} \times d}$

**Note on implicit relative position:** since DKA kernels operate over a fixed local window, position $j$ within the kernel always corresponds to "token $j - \lfloor k/2 \rfloor$ positions from me." This is an implicit relative positional bias that standard attention needs explicit encodings (like RoPE) to achieve. DKA gets it for free.

**For text, use larger maximum kernel sizes:** bump heads 7-8 to $k = 15$ or $k = 21$ since linguistic dependencies span further than image patch dependencies.

**Text kernel sizes for $H = 8$:**

| Heads 1-2 | Heads 3-4 | Heads 5-6 | Heads 7-8 |
|-----------|-----------|-----------|-----------|
| $k = 3$ | $k = 7$ | $k = 11$ | $k = 21$ |

**Tasks:**
- **AG News** (classification, 4 classes): sequence length 128 (truncate/pad), standard train/test split
- **WikiText-2** (language modeling): character-level or BPE tokenization, sequence length 256, perplexity as metric

**Classification head (AG News):** same as image — LayerNorm → AvgPool → Linear.

**LM head (WikiText-2):** LayerNorm → Linear$(d, V)$ at each position, predict next token, causal masking in the local window (zero out future positions in the kernel application).

---

## 5. Training Configuration

### 5.1 Loss Functions

**Primary loss — cross-entropy:**

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}$$

For language modeling, this is next-token prediction cross-entropy. For classification, standard multiclass cross-entropy.

**Kernel diversity loss:**

For each head $h$ in each layer $l$, sample $M = 64$ random token pairs $(i, j)$ from the same sequence. Penalize high cosine similarity between their generated kernels:

$$\mathcal{L}_{\text{div}}^{l,h} = \frac{1}{M} \sum_{(i,j)} \max\!\left(0,\ \text{cos\_sim}\!\left(\text{vec}(\hat{K}_i^{l,h}),\ \text{vec}(\hat{K}_j^{l,h})\right) - \tau\right)$$

where $\text{vec}(\cdot)$ flattens the kernel to a vector and $\tau = 0.5$ is a similarity threshold.

Total diversity loss:

$$\mathcal{L}_{\text{div}} = \frac{1}{H \cdot L} \sum_{l=1}^{L} \sum_{h=1}^{H} \mathcal{L}_{\text{div}}^{l,h}$$

**Total loss:**

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}$$

Start with $\lambda_{\text{div}} = 0.1$. Tune: if kernels naturally diversify, reduce toward 0. If they collapse, increase toward 0.5.

### 5.2 Optimizer & Schedule

**Optimizer:** AdamW

| Parameter Group | Learning Rate | Weight Decay |
|----------------|---------------|--------------|
| Main model (embeddings, FFN, projections, $K_{\text{base}}$) | $3 \times 10^{-4}$ | 0.05 |
| Kernel generators (all $W_{1,s}, W_{2,s}, W_{1,c}, W_{2,c}$ and their biases) | $3 \times 10^{-5}$ | 0.05 |
| $\alpha_h$ scalars | $1 \times 10^{-3}$ | 0.0 |
| Biases and LayerNorm params | (same as their group) | 0.0 |

The 10x lower learning rate on kernel generators is important — they produce weights that get applied as weights, creating a second-order-ish optimization landscape. Moving them too fast causes instability.

$\alpha_h$ gets a *higher* learning rate because it's a single scalar that controls the dynamic-vs-static balance — it should be free to move quickly.

**Schedule:**
- Linear warmup for first 10 epochs (or first 5% of total steps, whichever is more)
- Cosine annealing to $\eta_{\min} = 10^{-5}$ for the remainder

**Gradient clipping:**
- Global gradient norm clipping at 1.0
- Additional per-group clipping on kernel generator parameters at 0.5 (tighter, stabilizes the hypernetwork-style gradient path)

### 5.3 Regularization

| Technique | Setting |
|-----------|---------|
| Dropout (after DKA, after FFN) | 0.1 |
| Stochastic Depth (drop rate, linearly increasing per layer) | 0.1 max |
| Label Smoothing | 0.1 |
| Mixup ($\alpha$) | 0.8 (images only) |
| CutMix ($\alpha$) | 1.0 (images only) |
| RandAugment (images only) | 2 ops, magnitude 9 |
| Random Horizontal Flip | Yes (images only) |
| Random Crop with 4px padding | Yes (CIFAR only) |
| Random Resized Crop | Yes (Tiny ImageNet only) |

### 5.4 Training Hyperparameters

| Parameter | CIFAR-10 | Tiny ImageNet | AG News | WikiText-2 |
|-----------|----------|---------------|---------|------------|
| Batch size | 256 | 128 | 64 | 64 |
| Epochs | 300 | 300 | 30 | 50 |
| Precision | fp16 (AMP) | fp16 (AMP) | fp16 (AMP) | fp16 (AMP) |
| EMA decay | 0.9999 | 0.9999 | — | — |

Use EMA (exponential moving average) of model weights for evaluation on image tasks. Common in ViT training.

### 5.5 Initialization

| Component | Initialization |
|-----------|---------------|
| Patch/token embeddings | Normal(0, 0.02) |
| Positional embeddings | Normal(0, 0.02) |
| Linear layers (general) | Kaiming uniform (fan_in) |
| $K_{\text{base}}^h$ | Kaiming uniform, same as a standard conv layer |
| Kernel generator final layers ($W_{2,s}, W_{2,c}$) | Normal(0, 0.001) — near-zero so $\Delta K \approx 0$ at init |
| $\alpha_h$ | Constant 0.01 |
| Output projection $W^{out}$ | Normal(0, $0.02 / \sqrt{2L}$) — scale by depth, standard for deep transformers |

The near-zero init on kernel generator final layers + small $\alpha_h$ means the model starts as a static CNN. Dynamics emerge during training.

---

## 6. Complexity Analysis

### 6.1 FLOPs

**DKA module per layer:**

Kernel generation (per token, per head, all $R$ ranks):
- Spatial MLP: $R \times (2 d_h^2 + 2 d_h k_h)$ multiply-adds
- Channel MLP: $R \times (2 d_h^2 + 2 d_h^2) = R \times 4 d_h^2$ multiply-adds
- Outer products: $R \times k_h d_h$ multiply-adds
- Total per token per head: $R (6 d_h^2 + 2 d_h k_h + k_h d_h)$

Kernel application (per token, per head): $k_h \cdot d_h$ multiply-adds.

Head projection (in + out): $2 n d^2$.

**Total per DKA layer:**

$$\text{FLOPs}_{\text{DKA}} = n \sum_{h=1}^{H} \left[ R(6 d_h^2 + 3 k_h d_h) + k_h d_h \right] + 2nd^2$$

This is $O(n \cdot d^2)$ — **linear in sequence length $n$.**

**Standard attention per layer:**

$$\text{FLOPs}_{\text{Attn}} = 2n^2 d + 4nd^2$$

The $2n^2 d$ term (QK matmul + attention-value matmul) dominates for $n > d$.

**Concrete comparison for DKA-Small ($d=256, H=8, d_h=32, R=4$, kernels {3,3,5,5,7,7,11,11}):**

| Metric | DKA (ours) | Standard Attention |
|--------|-----------|-------------------|
| FLOPs per layer, $n=64$ (CIFAR) | ~2.1M | ~2.2M |
| FLOPs per layer, $n=256$ (TinyIN) | ~8.5M | ~33.8M |
| FLOPs per layer, $n=1024$ (text) | ~33.9M | ~537M |
| Scaling | $O(n)$ | $O(n^2)$ |

At CIFAR's short sequence length ($n=64$), the two are comparable. At Tiny ImageNet ($n=256$), DKA is ~4x cheaper. At text-scale sequences, the gap widens dramatically.

### 6.2 Memory

**DKA:** stores $n$ kernels per head of size $(k_h, d_h)$. Total:

$$\text{Mem}_{\text{DKA}} = n \cdot \sum_{h=1}^{H} k_h \cdot d_h$$

For our config: $n \cdot (3+3+5+5+7+7+11+11) \cdot 32 = 1664n$ floats.

**Attention:** stores the $n \times n$ attention matrix per head: $H \cdot n^2 = 8n^2$ floats.

DKA uses less memory when $1664n < 8n^2 \Rightarrow n > 208$.

For Tiny ImageNet ($n = 256$): DKA already cheaper. For any text task: way cheaper.

---

## 7. Experiments & Ablations

### 7.1 Main Comparison (CIFAR-10)

All models matched to ~8M parameters:

| Model | Type | Description |
|-------|------|-------------|
| ResNet-18 | Static CNN | Standard CNN baseline |
| ViT-Small | Standard attention | Vanilla Vision Transformer |
| DeiT-Small | Attention + distillation | ViT with improved training recipe |
| ConvNeXt-Tiny | Modern static CNN | Modernized ResNet with transformer design principles |
| **DKA-Small (ours)** | Dynamic kernel attention | The proposed model |

**Report for each model:** top-1 accuracy, total FLOPs (forward pass), throughput (images/sec on H100), peak GPU memory, total params.

Use identical training recipe for all models where possible (same epochs, augmentation, optimizer). For DeiT, use their official recipe since it includes distillation.

### 7.2 Scaling Test (Tiny ImageNet)

Same comparison table on Tiny ImageNet (200 classes, 64×64 images). This is where the $O(n)$ vs $O(n^2)$ difference starts to matter since $n = 256$ vs CIFAR's $n = 64$.

### 7.3 Text Experiment (AG News)

Compare DKA against a vanilla Transformer (same depth/width/params) on AG News topic classification. Report accuracy and FLOPs.

This is the "we also tried text and it works" result. Doesn't need to be SOTA — just needs to show the dynamic kernels learn something meaningful on language.

### 7.4 Ablation Studies

All ablations on CIFAR-10 with DKA-Small config unless otherwise stated.

**Ablation 1: Static vs Dynamic Kernels**

| Variant | Change |
|---------|--------|
| DKA-Small (full) | As described |
| DKA-Static | Force $\alpha_h = 0$ for all heads (pure static CNN, $\hat{K}_i = K_{\text{base}}$) |

Accuracy difference = the contribution of dynamic kernel generation.

**Ablation 2: Factored Kernel Rank**

| $R$ | Description |
|-----|-------------|
| 1 | Single rank-1 outer product (most constrained) |
| 2 | Sum of 2 rank-1 terms |
| **4** | **Default** |
| 8 | High-rank (most expressive, most parameters) |

Report accuracy and parameter count for each.

**Ablation 3: Kernel Size Assignment**

| Variant | Kernel Sizes |
|---------|--------------|
| All-small | All heads $k = 3$ |
| All-medium | All heads $k = 7$ |
| All-large | All heads $k = 11$ |
| **Multi-scale** | **{3,3,5,5,7,7,11,11}** |

Tests whether multi-scale assignment actually helps or if one kernel size suffices.

**Ablation 4: Diversity Loss**

| Variant | $\lambda_{\text{div}}$ |
|---------|----------------------|
| No diversity loss | 0.0 |
| **Default** | **0.1** |
| Strong diversity | 0.5 |

Report accuracy AND kernel cosine similarity statistics (measure whether kernels actually diversify).

**Ablation 5: Residual Kernel Structure**

| Variant | Description |
|---------|-------------|
| No residual | $\hat{K}_i = \Delta K_i$ (generate from scratch, no base kernel) |
| **Residual (default)** | $\hat{K}_i = K_{\text{base}} + \alpha \cdot \text{norm}(\Delta K_i)$ |
| Residual, fixed $\alpha$ | Same but $\alpha = 1.0$ fixed, not learned |

Tests whether the residual structure and learned $\alpha$ contribute to training stability and final accuracy.

**Ablation 6: Kernel Generator Capacity**

| Variant | Generator Architecture |
|---------|----------------------|
| Linear only | Single linear layer, no GELU (expected to collapse/underperform) |
| **MLP (default)** | **2-layer MLP with GELU** |
| Larger MLP | 3-layer MLP with larger hidden dim |

Tests the importance of the nonlinearity in the kernel generator.

### 7.5 $\alpha_h$ Trajectory Analysis

Not an ablation per se — just a tracking experiment.

Log the value of every $\alpha_h$ (across all heads and layers) at every epoch during training. Plot as curves: $\alpha_h$ vs epoch, one line per head, separate subplot per layer.

Questions this answers:
- Which heads become dynamic? Which stay static?
- Do deeper layers use more or less dynamics?
- Do larger-kernel heads have different $\alpha$ trajectories than small-kernel heads?

---

## 8. Visualizations & Graphs

This section specifies every figure and plot to generate. Target: enough visual content to fill 15+ pages of a 34-page writeup.

### 8.1 Architecture Diagrams

**Figure 1: DKA Module Overview**
- Block diagram showing the full flow: Input → Head Projection → Kernel Generation → Residual Kernel → Window Extract → Kernel Apply → Concat → Output Projection
- Show multiple heads with different kernel sizes
- Use a clean schematic style, not code

**Figure 2: Kernel Generation Detail**
- Zoom into one head's kernel generator
- Show the factored decomposition: input token → spatial MLP → $s_i$ and channel MLP → $c_i$ → outer product → sum over ranks → $\Delta K_i$
- Show the residual addition: $K_{\text{base}} + \alpha \cdot \text{norm}(\Delta K)$

**Figure 3: DKA vs Standard Attention Comparison**
- Side-by-side: standard attention (Q, K, V → dot product → softmax → weighted sum, global) vs DKA (input → kernel gen → local window → kernel apply, local)
- Annotate with complexity: $O(n^2 d)$ vs $O(nkd)$

### 8.2 Training Curves

**Figure 4: Training Loss Curves**
- All models on one plot (ResNet, ViT, DeiT, ConvNeXt, DKA)
- X-axis: epoch, Y-axis: training loss
- Separate subplot for validation accuracy vs epoch

**Figure 5: Diversity Loss Curve**
- Track $\mathcal{L}_{\text{div}}$ over training
- Separate curves for each layer (or average per layer)
- Shows whether kernels naturally diversify or need the loss to force it

**Figure 6: $\alpha_h$ Trajectories**
- One subplot per layer (8 subplots for 8 layers)
- Each subplot has 8 lines (one per head), colored by kernel size
- X-axis: epoch, Y-axis: $\alpha_h$ value
- This is one of the most interesting plots — it shows which heads "choose" to become dynamic

### 8.3 Kernel Visualizations (Image Model)

**Figure 7: Generated Kernels for Different Image Regions**
- Pick one trained image, divide into semantically distinct regions (sky, edge, texture, object)
- For each region, show the generated kernel for one head as a heatmap ($k_h \times d_h$ matrix)
- Arrange as a grid: rows = image regions, columns = heads (different kernel sizes)
- Show the original image with regions highlighted

**Figure 8: Kernel Similarity Matrix**
- For all $n = 64$ tokens in one CIFAR image, compute pairwise cosine similarity between their generated kernels (for one head)
- Display as a $64 \times 64$ heatmap
- Next to it, display the original image (reshaped to show the 64 patches in the same layout)
- Block structure in the heatmap = similar patches generating similar kernels = the model learned semantic grouping

**Figure 9: Base Kernel vs Dynamic Kernels**
- For one head, show $K_{\text{base}}$ as a heatmap
- Then show $\hat{K}_i$ for 5-6 different tokens
- The difference between base and dynamic kernels = what the generator is contributing
- Annotate with the $\alpha_h$ value for that head

### 8.4 Kernel Visualizations (Text Model)

**Figure 10: The "Bank" Figure**
- Input sentence 1: "I went to the bank to deposit money"
- Input sentence 2: "I sat on the river bank and watched the water"
- Extract the generated kernel for the token "bank" in each sentence
- Display both kernels as heatmaps side-by-side
- If they're meaningfully different → the architecture learns word-sense disambiguation at the kernel level
- Do this for multiple polysemous words: "bank", "bat", "spring", "crane", "left"

**Figure 11: Kernel Patterns Across a Sentence**
- Take one sentence, show the generated kernel for every token in sequence
- Display as a strip of heatmaps (one per token)
- Annotate with the actual words below
- Look for patterns: do function words ("the", "a") generate similar kernels? Do content words generate specialized ones?

### 8.5 Attention Map Equivalents

**Figure 12: DKA "Attention" Maps vs Real Attention Maps**
- For DKA: the generated kernel implicitly defines local attention weights. Visualize the kernel values (summed across channels) as an attention-style plot — for each token, show how much it attends to each neighbor.
- For the ViT baseline: extract actual attention maps from the same image/text.
- Display side-by-side. DKA maps should be more structured (local, sharp) vs attention maps (global, often diffuse).

### 8.6 Spectral Analysis

**Figure 13: Frequency Content of Generated Kernels Per Head**
- For each head (grouped by kernel size), compute FFT of the spatial component $s_i$ of the generated kernels across all tokens in a batch
- Plot the average power spectrum
- Hypothesis: small-kernel heads produce high-frequency kernels (edge detectors), large-kernel heads produce low-frequency kernels (smooth, averaging)
- If this emerges naturally, it validates the multi-scale head assignment

### 8.7 Ablation Results

**Figure 14: Ablation Bar Chart**
- Grouped bar chart
- Groups: Static vs Dynamic, Rank 1/2/4/8, Kernel size variants, Diversity loss variants, Residual variants, Generator capacity
- Y-axis: accuracy
- Error bars if you run multiple seeds (recommended: 3 seeds)

**Figure 15: Rank vs Accuracy vs Parameters**
- Scatter plot: x-axis = total params, y-axis = accuracy, points labeled by rank $R$
- Shows the efficiency-expressiveness tradeoff

### 8.8 Efficiency Plots

**Figure 16: FLOPs vs Sequence Length**
- X-axis: sequence length $n$ (log scale, from 64 to 4096)
- Y-axis: FLOPs per layer (log scale)
- Lines for: Standard Attention, DKA (ours), Linear Attention (for reference)
- The attention line curves up quadratically, DKA stays linear
- Theoretical curves, computed from the formulas

**Figure 17: Throughput Comparison**
- Bar chart: images/sec on H100 for each model at inference
- Also report at different batch sizes

**Figure 18: Memory Comparison**
- Bar chart: peak GPU memory during training for each model
- Also plot memory vs sequence length (same style as Figure 16)

### 8.9 Dynamism Analysis

**Figure 19: Kernel Variance Per Layer**
- For each layer, compute variance of generated kernels across all tokens in a batch (metric of "how dynamic" each layer is)
- Bar chart or line plot: x-axis = layer, y-axis = average kernel variance
- Hypothesis: early layers are less dynamic (basic feature extraction is static), later layers are more dynamic (content-dependent reasoning)

**Figure 20: Kernel Variance Per Head Within Layer**
- Heatmap: rows = layers, columns = heads, color = kernel variance
- Shows the full 2D picture of where dynamism lives in the model

### 8.10 Comparison Table (Final Results)

**Table 1: Main Results**

| Model | CIFAR-10 Acc | TinyIN Acc | AG News Acc | Params | FLOPs | Throughput | Memory |
|-------|-------------|-----------|-------------|--------|-------|------------|--------|
| ResNet-18 | | | — | ~8M | | | |
| ViT-Small | | | | ~8M | | | |
| DeiT-Small | | | — | ~8M | | | |
| ConvNeXt-Tiny | | | — | ~8M | | | |
| DKA-Small | | | | ~8M | | | |

**Table 2: Ablation Results**

(Fill in from experiments)

---

## 9. Project Structure

```
dka/
├── train.ipynb              # Main training notebook (Jupyter)
├── configs/
│   ├── cifar10.yaml         # CIFAR-10 training config
│   ├── tinyimagenet.yaml    # Tiny ImageNet config
│   ├── agnews.yaml          # AG News config
│   └── wikitext2.yaml       # WikiText-2 config (optional)
├── models/
│   ├── dka_module.py        # Core DKA attention module
│   ├── kernel_generator.py  # Factored kernel generation (spatial + channel MLPs)
│   ├── dka_block.py         # Full DKA transformer block (DKA + FFN + LN + residual)
│   ├── dka_model.py         # Full model (patch embed + blocks + classifier)
│   ├── baselines/
│   │   ├── vit.py           # ViT-Small baseline (or use timm)
│   │   ├── resnet.py        # ResNet-18 baseline (or use torchvision)
│   │   └── convnext.py      # ConvNeXt-Tiny baseline (or use timm)
├── data/
│   ├── cifar10.py           # CIFAR-10 dataloader with augmentation
│   ├── tinyimagenet.py      # Tiny ImageNet dataloader
│   ├── agnews.py            # AG News dataloader
│   └── wikitext2.py         # WikiText-2 dataloader (optional)
├── training/
│   ├── trainer.py           # Training loop, logging, checkpointing
│   ├── optimizer.py         # AdamW setup with parameter groups
│   ├── scheduler.py         # Warmup + cosine schedule
│   ├── losses.py            # CE loss + diversity loss
│   └── ema.py               # Exponential moving average
├── analysis/
│   ├── kernel_viz.py        # Kernel heatmap visualizations (Figures 7-11)
│   ├── attention_maps.py    # DKA "attention" maps + ViT comparison (Figure 12)
│   ├── spectral.py          # FFT analysis of kernels (Figure 13)
│   ├── alpha_tracking.py    # Log and plot alpha trajectories (Figure 6)
│   ├── diversity_metrics.py # Kernel similarity stats (for monitoring + Figure 8)
│   ├── dynamism.py          # Per-layer, per-head variance analysis (Figures 19-20)
│   ├── polysemy.py          # "Bank" experiment for text (Figure 10)
│   ├── efficiency.py        # FLOP counting, throughput benchmarking (Figures 16-18)
│   └── ablation_plots.py    # Bar charts and tables from ablation runs (Figures 14-15)
├── analysis.ipynb           # Jupyter notebook that runs all analysis and generates all figures
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs (TensorBoard or W&B)
└── figures/                 # Generated figures for the writeup
```

### Key Design Decisions for Implementation

- **`train.ipynb`**: The main entry point. Should be runnable top-to-bottom on either the 9070 XT (CIFAR-10) or H100 (Tiny ImageNet, text). Use YAML configs to switch between tasks.
- **`analysis.ipynb`**: Loads trained checkpoints and generates every figure. Should also be runnable top-to-bottom.
- Use **PyTorch** throughout. Use **timm** for baseline ViT/DeiT/ConvNeXt if available, otherwise implement minimal versions.
- Use **Weights & Biases** (wandb) for experiment tracking. Log: loss, accuracy, $\alpha_h$ values, kernel diversity metrics, learning rate.
- All figures saved as both PNG (for the notebook) and PDF (for the writeup) at publication quality.

---

## 10. Implementation Notes

### 10.1 Critical: The Unfold + Einsum Pattern

The core kernel application MUST be implemented as a batched unfold + einsum operation. No Python for-loops over tokens. The pattern:

1. Pad the sequence: shape $(B, n, d_h) \to (B, n + k - 1, d_h)$
2. Unfold along the sequence dimension to extract all windows: $(B, n, k, d_h)$
3. Generated kernels are already shape $(B, n, k, d_h)$
4. Element-wise multiply and sum over $k$: `einsum('bnkd,bnkd->bnd', kernels, windows)`

This is fully parallel and GPU-friendly.

### 10.2 Causal Masking for Language Modeling

For autoregressive text tasks (WikiText-2), the local window must not include future tokens. Zero out the upper-triangular portion of each kernel:

For token at position $i$ with window $[i - \lfloor k/2 \rfloor, \ldots, i + \lfloor k/2 \rfloor]$, set kernel values to 0 for all positions $> i$.

Implement as a static binary mask applied to the kernel before the einsum.

### 10.3 Separate Parameter Groups

Critical for training stability. The optimizer must have at least 3 parameter groups:

1. **Main parameters**: embeddings, FFN layers, LayerNorm, head projections, $K_{\text{base}}$ — lr $3 \times 10^{-4}$, weight decay 0.05
2. **Kernel generators**: all $W_{1,s}, W_{2,s}, W_{1,c}, W_{2,c}$ and biases — lr $3 \times 10^{-5}$, weight decay 0.05
3. **Alpha scalars**: $\alpha_h$ — lr $1 \times 10^{-3}$, weight decay 0.0

### 10.4 Monitoring During Training

Log these every $N$ steps (not just every epoch):
- Total loss, CE loss, diversity loss (separate)
- Mean and std of $\alpha_h$ across all heads/layers
- Mean pairwise cosine similarity of kernels within each head (collapse detection)
- Mean Frobenius norm of $\Delta K$ (monitors kernel magnitude)
- Gradient norms for kernel generator params vs main params (stability check)

If mean pairwise cosine similarity exceeds 0.9 for multiple heads → kernels are collapsing, increase $\lambda_{\text{div}}$ or reduce kernel generator learning rate.

### 10.5 Mixed Precision Notes

Use AMP (torch.cuda.amp) with fp16 for forward/backward. The kernel generation involves small values (near-zero init of $\alpha_h$) — ensure the generated kernels are computed in fp32 or use a loss scaler that handles the small gradients. The einsum application can safely run in fp16.

### 10.6 ROCm Compatibility (9070 XT)

The 9070 XT runs ROCm on Windows. Known issues:
- `F.unfold` should work fine on ROCm
- Custom CUDA kernels won't work — stick to native PyTorch ops
- AMP may have minor differences — test with fp32 first, then enable AMP
- If `torch.einsum` is slow on ROCm, rewrite as `torch.bmm` after reshaping

### 10.7 Reproducing Baselines

For fair comparison:
- **ResNet-18**: use `torchvision.models.resnet18`, modify first conv for 32×32 input (kernel 3, stride 1, no maxpool)
- **ViT-Small**: use `timm.create_model('vit_small_patch4_32')` or implement minimal ViT with same $d$, $H$, $L$ as DKA
- **DeiT-Small**: use `timm` with DeiT recipe, or skip if distillation isn't applicable
- **ConvNeXt-Tiny**: use `timm.create_model('convnext_tiny')`, adjust for CIFAR input size

Match total parameter count as closely as possible (within 10%). Report exact param counts.

# 2. Related Work

Research on **Sequential Recommender Systems** has evolved around a central trade-off: **modeling rich temporal dependencies across long histories** while **generalizing reliably under interaction sparsity**. This section traces the progression from static baselines and Markov Chains to recurrent neural networks (GRU4Rec), unidirectional causal self-attention (SASRec), and bidirectional Transformers (BERT4Rec), motivating our experimental design and model selection.

---

## 2.1. Static Baselines and Low-Order Markov Chains

Early recommendation methods such as **Most-Popular** and **Bayesian Personalized Ranking Matrix Factorization (BPR-MF)** (Rendle et al., 2009) model user–item affinity through static global counts or inner products of static user and item embeddings. Although Most-Popular remains a surprisingly strong non-personalized baseline on entertainment datasets with pronounced blockbuster effects like MovieLens, both Most-Popular and static MF are invariant to interaction order and cannot adapt to shifts in short-term user intent.

To incorporate sequential transitions, **Markov Chain (MC)** approaches assume that the next item depends primarily on the immediately preceding item(s). **FPMC** (*Factorizing Personalized Markov Chains*, Rendle et al., 2010) combines a static user preference matrix with a first-order item-to-item transition tensor $(s_{t-1} \to s_t)$. **Fossil** (He & McAuley, 2016) extends this idea by blending item similarity with weighted higher-order Markov chains. However, fixed-order Markov models struggle on long, noisy sequences: a small order $k$ misses long-range dependencies (e.g., returning to a film franchise after several unrelated movies), whereas increasing $k$ suffers from severe data sparsity and parameter growth.

---

## 2.2. Recurrent and Convolutional Sequential Models (GRU4Rec, Caser)

Deep sequential recommendation was pioneered by **GRU4Rec** (Hidasi et al., ICLR 2016) and its improved ranking-loss formulation (Hidasi & Karatzoglou, CIKM 2018). Instead of restricting context to a fixed Markov window, GRU4Rec feeds item embeddings through a **Gated Recurrent Unit (GRU)** whose update and reset gates recursively compress the variable-length prefix $(s_1, \dots, s_t)$ into a single hidden state vector $\mathbf{h}_t \in \mathbb{R}^d$. In parallel, convolutional approaches such as **Caser** (Tang & Wang, 2018) treat the embedding matrix of the most recent $L$ items as a 2D image and apply horizontal and vertical convolutional filters to capture local union-level patterns.

While GRU4Rec significantly outperforms low-order Markov baselines and serves as our **primary sequential baseline**, its recurrent formulation $\mathbf{h}_t = \text{GRU}(\mathbf{h}_{t-1}, \mathbf{e}_{s_t})$ introduces two structural bottlenecks:
1. **Sequential Computation & $O(L)$ Path Length**: Computing $\mathbf{h}_t$ requires waiting for $\mathbf{h}_{t-1}$, limiting parallelization across sequence steps on GPUs. Information from an earlier item must traverse $O(L)$ sequential nonlinear transformations, making distant signals vulnerable to forgetting or overwriting by intervening noise.
2. **Fixed-Width State Bottleneck**: All past user interests—even when a user alternates across distinct movie genres—must be compressed into a single fixed-dimensional vector $\mathbf{h}_t$.

---

## 2.3. Unidirectional Causal Self-Attention: SASRec

Building on the **Transformer** architecture (Vaswani et al., NeurIPS 2017), Kang & McAuley (ICDM 2018) introduced **SASRec** (*Self-Attentive Sequential Recommendation*), replacing both recurrence and convolution with stacked **Causal Self-Attention** blocks, learned positional embeddings, point-wise feed-forward networks (FFNs), residual connections, dropout, and Layer Normalization.

At each valid position $t$, causal self-attention computes scaled dot-product attention directly over all valid past positions $j \le t$, masking out future positions $j > t$ with $-\infty$ before softmax. This yields three core advantages:
- **$O(1)$ Maximum Path Length**: Any two valid items within window $L$ interact directly in a single attention hop, regardless of their temporal distance.
- **Adaptive Context Span**: Kang & McAuley (2018) demonstrated that on sparse datasets (e.g., Amazon Reviews), SASRec naturally concentrates attention on the most recent 1–2 items like a Markov Chain, whereas on denser datasets with longer histories like **MovieLens-1M** (mean length `165.6`), attention heads distribute weight across broader historical context.
- **Native Next-Item Alignment**: Lower-triangular causal masking enables parallel next-item supervision across all prefixes $(s_1 \to s_2), \dots, (s_1, \dots, s_{m-1} \to s_m)$ in a single forward pass with zero future leakage, aligning training directly with last-position inference.

---

## 2.4. Bidirectional Transformers (BERT4Rec) and Scope Justification

Following SASRec, Sun et al. (CIKM 2019) proposed **BERT4Rec**, which replaces unidirectional causal masking with bidirectional self-attention trained via the **Cloze / Masked Item Prediction** objective (randomly replacing items with `[MASK]` and reconstructing them using both left and right context).

Subsequent reproducibility and benchmarking studies (Petrov & Macdonald, RecSys 2022; Krichene & Rendle, KDD 2020) highlighted practical trade-offs of bidirectional Cloze training:
- **Train–Test Objective Mismatch**: During training, `[MASK]` tokens appear throughout the sequence with bidirectional context; at inference time, `[MASK]` must always be appended at the rightmost position, where no right-hand context exists.
- **Lower Sample Efficiency**: Loss is computed only on the masked subset (typically 15–20% of positions per step), requiring substantially more training epochs and compute to converge compared to SASRec's dense prefix supervision.

In accordance with our project scope, BERT4Rec is analyzed as an important architectural reference point but **explicitly excluded from implementation**, allowing rigorous, controlled comparison across three distinct model families: **Most-Popular** (static global frequency), **GRU4Rec** (recurrent sequential), and **SASRec** (causal self-attention).

---

## 2.5. Comparative Summary of Sequential Recommendation Families

Table 2.1 compares theoretical complexity and architectural properties across model families (following Vaswani et al., 2017 and Kang & McAuley, 2018), where $L$ is maximum sequence length and $d$ is hidden dimensionality.

**Table 2.1: Complexity and Receptive-Field Comparison Across Recommendation Approaches**

| Method | Model Family | Per-Layer Complexity | Min. Sequential Ops | Max. Path Length | Native Causal Inference | Project Scope |
| --- | --- | :---: | :---: | :---: | :---: | --- |
| **Most-Popular** | Static Global Frequency | $O(1)$ at inference | $O(1)$ | Order-agnostic | Yes (fit on Train only) | **Required Baseline** |
| **FPMC** (Rendle et al., 2010) | MF + 1st-Order Markov | $O(d)$ | $O(1)$ | $1$ preceding step | Yes | Literature reference |
| **GRU4Rec** (Hidasi et al., 2016) | Recurrent Neural Net (GRU) | $O(L \cdot d^2)$ | $O(L)$ | $O(L)$ | Yes | **Required Baseline** |
| **Caser** (Tang & Wang, 2018) | 1D/2D Convolution (CNN) | $O(k \cdot L \cdot d^2)$ | $O(1)$ | $O(L / k)$ | Window-dependent | Literature reference |
| **SASRec** (Kang & McAuley, 2018) | Causal Self-Attention | $O(L^2 \cdot d + L \cdot d^2)$ | $O(1)$ | $O(1)$ | **Yes (Causal Mask)** | **Required Main Model** |
| **BERT4Rec** (Sun et al., 2019) | Bidirectional Transformer | $O(L^2 \cdot d + L \cdot d^2)$ | $O(1)$ | $O(1)$ | No (appended `[MASK]`) | Out of scope |

Under our MovieLens-1M configuration ($L = 100, d = 64$), SASRec's $O(L^2 \cdot d)$ term is small enough to execute efficiently in parallel on an RTX 4060 (8 GB VRAM) while reducing the maximum context path length from $O(L)$ in GRU4Rec to $O(1)$. Nevertheless, theoretical complexity alone does not guarantee superior ranking accuracy on every user cohort; empirical performance across SASRec, GRU4Rec, and Most-Popular must be verified fairly under the shared full-catalog protocol.

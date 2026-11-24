# 3. Methodology

All three models compared in this study—**Most-Popular**, **GRU4Rec**, and **SASRec**—share the exact same item vocabulary $\mathcal{I} = \{1, \dots, N\}$ ($N = 3,706$, with `pad_id = 0` reserved strictly for padding), chronological Leave-One-Out splits (`train`, `val`, `test`), right-padded input convention ($L = 100$), and full-catalog evaluation protocol.

---

## 3.1. Shared Input Representation and Right-Padding Convention

Given a user's valid interaction history $(s_1, \dots, s_m)$ with $m \ge 1$, the sequence is truncated to its most recent $L = 100$ valid items and right-padded with `0` to fixed length $T \le L$:

$$\mathbf{x} = \big(x_0, x_1, \dots, x_{T-1}\big), \quad x_t \in \{0, 1, \dots, N\}$$

where valid items occupy contiguous positions $0, \dots, k - 1$ ($k = \min(m, L) \ge 1$) and padding tokens occupy positions $k, \dots, T - 1$. A boolean validity mask is defined as:

$$M_t = \mathbb{I}(x_t \neq 0), \quad t \in \{0, \dots, T - 1\}$$

Every input sequence fed to a neural model is explicitly validated to contain at least one valid item ($\sum_{t=0}^{T-1} M_t \ge 1$). Choosing **right-padding** rather than left-padding ensures two key invariance properties:
1. Valid tokens at indices $0, \dots, k - 1$ receive identical positional indices regardless of padding length $T$.
2. For any valid query position $i < k$, every causally permitted key position $j \le i$ is guaranteed to be a valid non-padding token ($j < k$), eliminating all-masked softmax rows on valid queries.

---

## 3.2. Baseline 1: Most-Popular (`fit_popularity.py`)

The **Most-Popular** baseline provides a non-personalized global frequency benchmark computed strictly from training-split interactions (`train[u]` only, ensuring zero leakage from validation or test targets):

$$\text{score}_{\text{pop}}(i) = \sum_{u \in \mathcal{U}} \sum_{s \in S_{\text{train}}^u} \mathbb{I}(s = i), \quad \forall i \in \{1, \dots, N\}$$

with $\text{score}_{\text{pop}}(0) = -10^9$. During evaluation and demo inference, items already present in the user's seen history are masked to $-\infty$ (while retaining the held-out target during validation/test), and ties are broken deterministically by ascending `item_id`.

---

## 3.3. Baseline 2: Pure-PyTorch GRU4Rec (`build_gru4rec.py`)

**GRU4Rec** (Hidasi et al., 2016) serves as our recurrent sequential baseline, implemented in pure PyTorch with a tied item-embedding output head matching our shared catalog scoring contract.

### Embedding and Recurrent Encoding
Each item token $x_t$ is mapped through a learned embedding matrix $\mathbf{E} \in \mathbb{R}^{(N+1) \times d}$ ($d = 64$, with row $\mathbf{E}_0 = \mathbf{0}$ fixed via `padding_idx=0`) followed by dropout ($p = 0.2$) and validity masking:

$$\tilde{\mathbf{e}}_t = M_t \cdot \text{Dropout}(\mathbf{E}_{x_t})$$

A unidirectional Gated Recurrent Unit (GRU, 1 layer, hidden dimension $d = 64$) updates its hidden state $\mathbf{h}_t \in \mathbb{R}^d$ sequentially from $t = 0$ to $T - 1$:

$$\mathbf{z}_t = \sigma\left(\mathbf{W}_z \tilde{\mathbf{e}}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z\right)$$

$$\mathbf{r}_t = \sigma\left(\mathbf{W}_r \tilde{\mathbf{e}}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r\right)$$

$$\tilde{\mathbf{h}}_t = \tanh\left(\mathbf{W}_h \tilde{\mathbf{e}}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h\right)$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t + \mathbf{z}_t \odot \mathbf{h}_{t-1}$$

Followed by output dropout, Layer Normalization, and zero-masking on padded positions:

$$\mathbf{o}_t = M_t \cdot \text{LayerNorm}\left(\text{Dropout}(\mathbf{h}_t)\right)$$

Because the GRU operates strictly left-to-right and padding occurs only on the right ($t \ge k$), every valid output state $\mathbf{o}_0, \dots, \mathbf{o}_{k-1}$ is strictly invariant to trailing right-padding.

---

## 3.4. Main Model: Pure-PyTorch SASRec (`build_sasrec.py`)

**SASRec** (Kang & McAuley, 2018) models sequential dependencies purely through stacked causal multi-head self-attention blocks and point-wise feed-forward networks. Our default configuration uses $d = 64$, $b = 2$ Transformer blocks, $h = 2$ attention heads ($d_h = d / h = 32$), dropout $p = 0.2$, and maximum sequence length $L = 100$.

### Item and Positional Embeddings
Unlike GRU4Rec, self-attention is permutation-equivariant and therefore requires explicit positional information. We combine scaled item embeddings $\mathbf{E} \in \mathbb{R}^{(N+1) \times d}$ (`padding_idx=0`, row 0 frozen at zero) with a learned positional embedding table $\mathbf{P} \in \mathbb{R}^{L \times d}$:

$$\mathbf{H}^{(0)}_t = M_t \cdot \text{Dropout}\left(M_t \cdot \left(\sqrt{d}\,\mathbf{E}_{x_t} + \mathbf{P}_t\right)\right), \quad t \in \{0, \dots, T-1\}$$

### Padding-Safe Multi-Head Causal Self-Attention
For each Transformer block $\ell \in \{1, \dots, b\}$, we apply Pre-LayerNorm followed by multi-head causal self-attention. Let $\bar{\mathbf{H}}^{(\ell-1)} = \text{LayerNorm}(\mathbf{H}^{(\ell-1)}) \in \mathbb{R}^{T \times d}$. Queries, keys, and values are linearly projected and split into $h$ heads of dimension $d_h = d / h$:

$$\mathbf{Q}^{(m)} = \bar{\mathbf{H}}^{(\ell-1)} \mathbf{W}_Q^{(m)}, \qquad \mathbf{K}^{(m)} = \bar{\mathbf{H}}^{(\ell-1)} \mathbf{W}_K^{(m)}, \qquad \mathbf{V}^{(m)} = \bar{\mathbf{H}}^{(\ell-1)} \mathbf{W}_V^{(m)}$$

The scaled attention score matrix for head $m$ is:

$$\mathbf{S}^{(m)} = \frac{\mathbf{Q}^{(m)} \left(\mathbf{K}^{(m)}\right)^\top}{\sqrt{d_h}} \in \mathbb{R}^{T \times T}$$

To guarantee both **zero future leakage** and **padding invariance**, we construct the combined boolean allow-mask $A_{i, j} = \mathbb{I}(j \le i) \land M_j$, which permits query $i$ to attend only to valid non-padding keys at positions $j \le i$. To prevent all-masked rows from ever producing `NaN` in either forward softmax or backward autograd (even on edge-case inputs), any query row with no allowed keys falls back safely to the diagonal entry before softmax, and all padded query rows are explicitly multiplied by $M_i \in \{0, 1\}$ after softmax:

$$\tilde{S}^{(m)}_{i, j} = \begin{cases} S^{(m)}_{i, j} & \text{if } A^{\text{safe}}_{i, j} = 1 \\ -\infty & \text{otherwise} \end{cases}, \qquad \boldsymbol{\alpha}^{(m)}_{i, :} = M_i \cdot \text{softmax}\left(\tilde{\mathbf{S}}^{(m)}_{i, :}\right)$$

The head outputs are concatenated, projected via $\mathbf{W}_O \in \mathbb{R}^{d \times d}$, masked by $M_t$, and added through a residual connection with dropout:

$$\mathbf{Z}^{(\ell)} = \mathbf{M} \odot \left(\mathbf{H}^{(\ell-1)} + \text{Dropout}\left(\text{Concat}_{m=1}^h\left(\text{Dropout}(\boldsymbol{\alpha}^{(m)}) \mathbf{V}^{(m)}\right) \mathbf{W}_O\right)\right)$$

### Point-Wise Feed-Forward Network and Final Normalization
Following Kang & McAuley (2018), each block applies a position-wise two-layer ReLU feed-forward network ($\mathbf{W}_1, \mathbf{W}_2 \in \mathbb{R}^{d \times d}$) wrapped in Pre-LayerNorm, dropout, residual addition, and validity masking:

$$\text{FFN}(\mathbf{u}) = \mathbf{W}_2 \, \text{Dropout}\left(\text{ReLU}(\mathbf{u} \mathbf{W}_1 + \mathbf{b}_1)\right) + \mathbf{b}_2$$

$$\mathbf{H}^{(\ell)} = \mathbf{M} \odot \left(\mathbf{Z}^{(\ell)} + \text{Dropout}\left(\text{FFN}\left(\text{LayerNorm}(\mathbf{Z}^{(\ell)})\right)\right)\right)$$

After $b$ blocks, a final LayerNorm produces the contextual sequence representations:

$$\mathbf{O} = \mathbf{M} \odot \text{LayerNorm}\left(\mathbf{H}^{(b)}\right) \in \mathbb{R}^{T \times d}$$

---

## 3.5. Last Valid Hidden State Selection and Tied Catalog Scoring

Both GRU4Rec and SASRec share an identical output projection and prediction head:
- **Catalog Logits**: Using weight tying with the item embedding matrix $\mathbf{E} \in \mathbb{R}^{(N+1) \times d}$ reduces overfitting and keeps parameter counts compact:

$$\hat{\mathbf{y}}_t = \mathbf{O}_t \mathbf{E}^\top \in \mathbb{R}^{N+1}, \qquad \hat{y}_{t, 0} \leftarrow -10^9$$

- **Inference / Evaluation Position**: We **never** read predictions from a padding query position. Instead, for every user sequence $\mathbf{x}$ in a batch, we compute the exact index of the last valid (non-padding) item:

$$t^\star(\mathbf{x}) = \max \{ t \in \{0, \dots, T-1\} \mid x_t \neq 0 \}$$

and compute full-catalog scores exclusively from $\mathbf{O}_{t^\star(\mathbf{x})} \mathbf{E}^\top \in \mathbb{R}^{N+1}$.

---

## 3.6. Next-Item Cross-Entropy Loss (`compute_next_item_loss.py`)

For each training sequence $S_{\text{train}}^u$ (truncated to its most recent $L + 1$ items $(w_0, w_1, \dots, w_k)$ with $k \le L$), we construct aligned right-padded input and target tensors of length $L$:

$$\mathbf{x} = (w_0, w_1, \dots, w_{k-1}, 0, \dots, 0), \qquad \mathbf{y} = (w_1, w_2, \dots, w_k, 0, \dots, 0)$$

Both SASRec and GRU4Rec are trained with full-catalog next-item **Cross-Entropy Loss** strictly over non-padding target positions (`ignore_index = 0`), with the padding column logit suppressed to $-10^9$:

$$\mathcal{L}(\theta) = -\frac{1}{\sum_{b=1}^{B} \sum_{t=0}^{L-1} \mathbb{I}(y_{b, t} \neq 0)} \sum_{b=1}^{B} \sum_{t=0}^{L-1} \mathbb{I}(y_{b, t} \neq 0) \log \frac{\exp\left(\hat{y}_{b, t, y_{b, t}}\right)}{\sum_{j=1}^{N} \exp\left(\hat{y}_{b, t, j}\right)}$$

Supervising all valid prefix positions simultaneously with full-catalog cross-entropy avoids negative-sampling bias and directly aligns the training objective with full-catalog ranking evaluation.

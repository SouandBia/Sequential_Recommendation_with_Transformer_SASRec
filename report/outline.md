# Final Report Outline (15–20 Pages) & 5-Slide Deck

> Cross-check section headings and formatting against the instructor's official template before submission. Every claim, table, and plot must come from actual measured experiments.

## 7-Section Report Allocation

| Section | Target Pages | Primary Owner | Required Content, Figures & Tables |
| --- | --- | --- | --- |
| **1. Problem & Data** | 2.5 – 3.0 | Student B | Sequential recommendation as next-recorded-rating prediction (not verified watch time); MovieLens-1M quality stats; rating & sequence-length distribution charts; timeline & Leave-One-Out diagram |
| **2. Related Work** | 2.0 – 2.5 | Student B | Evolution: Markov Chains (FPMC) -> RNNs (GRU4Rec) -> Causal Self-Attention (SASRec) -> Bidirectional Transformer (BERT4Rec); complexity & receptive-field comparison table |
| **3. Methodology** | 3.0 – 3.5 | Student A | Architecture & math for Most-Popular, GRU4Rec, SASRec: item/positional embeddings, causal self-attention, FFN, padding-safe masking, last valid hidden state, next-item loss |
| **4. Implementation** | 2.0 – 2.5 | Student A & B | System diagram & tensor shapes; right-padding/causal mask verification; VRAM-aware loss config, real training time & RTX 4060 VRAM measurements; FastAPI + Stitch React contract & screenshots |
| **5. Evaluation & Error Analysis** | 4.0 – 5.0 | Student A | **Required:** Full-catalog HR@10 & NDCG@10 for all 3 models, validation curves, length-quantile group comparison, 3 concrete error cases. **Optional (only if run):** sampled-100, `L`/`b` ablation, attention diagnostic |
| **6. Conclusion** | 1.0 – 1.5 | Student B | Honest summary of measured comparison (do not assume SASRec wins); limitations (rating timestamps, short histories, selection/exposure bias); future work (genre features, time-aware models) |
| **7. References** | 1.0 – 1.5 | Both | 15–25 properly formatted academic citations (Kang & McAuley 2018; Hidasi et al. 2016; Sun et al. 2019; Vaswani et al. 2017; Harper & Konstan 2015, etc.) |

## 5-Slide Presentation (`report/slides/`)
1. Problem motivation & MovieLens-1M Leave-One-Out setup
2. SASRec vs GRU4Rec vs Popularity architectures & causal/padding safety
3. Measured full-catalog results (`HR@10`, `NDCG@10`, RTX 4060 training stats)
4. Length-quantile & concrete error analysis
5. Live FastAPI + Stitch React demo & key takeaways

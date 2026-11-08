# IMF Country Report → LLM Text-to-Dashboard Pipeline (Canada, 2025 Article IV)

An end-to-end pipeline that turns an unstructured IMF Article IV staff report into verified, machine-readable indicators, thematic distributions, qualitative risk insights and an executive dashboard. It uses Google Gemini models for extraction and for LLM-as-a-Judge auditing, and runs local dense retrieval (sentence-transformers + FAISS).

**Report analysed:** *Canada: 2025 Article IV Consultation – Press Release; and Staff Report*, IMF Country Report No. 26/12 (published 21 January 2026). Allocation: SRN 24135630 → Canada, 2024–2025.

## Run it

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ds25abd/imf-llm-canada/blob/main/IMF_Canada_LLM_Pipeline.ipynb)

1. Open the notebook in Colab using the badge above, then choose **Runtime → Run all**. The first cell clones this repository so the PDF and cache are available.
2. **Optional:** add a Colab secret named `GEMINI_API_KEY` (🔑 sidebar) to make live API calls.
   **Without a key**, every LLM response is replayed from `cache/llm_cache.json`, so the notebook still runs end-to-end and reproduces every number in the report.

Locally: `pip install -r requirements.txt`, then `jupyter notebook IMF_Canada_LLM_Pipeline.ipynb`.

## Pipeline

| Stage | What happens |
|---|---|
| Ingestion & cleaning | `pypdf` text extraction; removes running headers and footers and TOC leaders; converts vulgar fractions (`2¼` → `2.25`); re-flows paragraphs; keeps table rows intact |
| Segmentation & chunking | heading-based section detection; 400-token / 50-token-overlap sentence-aware chunks, counted with the embedding model's tokenizer; table chunks repeat their year header |
| Summarisation | batched section summaries (< 100 words, enforced in code); one-pass executive summary |
| Evaluation | a different Gemini model acts as judge (consistency / completeness / factual alignment), plus deterministic numeric-grounding checks |
| RAG | `BAAI/bge-small-en-v1.5` embeddings in a `faiss.IndexFlatL2` index; closed-book vs single-query vs multi-query (RRF) RAG, scored by a blind judge |
| Structured output | 7 themes (lexicon + LLM, Cohen's κ); strengths and vulnerabilities with verified quotes; schema-constrained indicator JSON compared across 3 model/prompt configurations; fallback capture; trend series checked against the tables |
| Dashboard | `pd.melt` tidy data → 6-panel dashboard + individual figures at 300 dpi |

## Repository layout

```
IMF_Canada_LLM_Pipeline.ipynb   the executable notebook (generated from notebook_src.py)
notebook_src.py                 notebook source in percent format; build_notebook.py converts it
data/                           the IMF report PDF
cache/                          LLM response cache, resolved model names, embedding cache
outputs/                        JSON outputs (indicators, themes, insights, trends, RAG evaluation, call log)
outputs/figures/                dashboard.png + fig1–fig6 (300 dpi)
```

## Key output files

* `outputs/core_indicators.json` — the 7-field target schema (verified values; `null` = fallback)
* `outputs/task2_indicators_audit.json` — evidence quotes, human gold standard, verification, fallback log, benchmark, judge audits
* `outputs/task1_summaries.json`, `task2_themes.json`, `task2_qualitative_insights.json`, `task2_trends.json`, `task3_rag_evaluation.json`

## Licence and data

The code is MIT-licensed (see `LICENSE`). The IMF report is © International Monetary Fund and is included unmodified for academic reproducibility only. Source: https://www.imf.org/en/publications/cr/issues/2026/01/21/canada-2025-article-iv-consultation-press-release-and-staff-report-573340

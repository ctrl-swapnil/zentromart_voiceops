# ZentroMart VoiceOps 2026: Capstone Project Report
**Author:** Swapnil Mandal  
**Course:** IIT Guwahati (C&A Club) x Gnani.ai — Tokens to Transformers  
**Public GitHub Repository:** https://github.com/ctrl-swapnil/zentromart_voiceops  

---

## 1. Executive Summary & Architecture Overview
ZentroMart is a fictional Indian online store selling groceries and electronics[cite: 2]. This project implements an end-to-end voice-enabled customer support agent across four integrated layers[cite: 2, 8, 9, 10]:
* **Step 1 (LLM Prompting):** Enforces ZentroMart persona rules, mandatory AI assistant identification on the first turn[cite: 8], and ear-friendly response formatting[cite: 11].
* **Step 2 (RAG Grounding):** Indexes 21 Markdown knowledge-base documents into ChromaDB[cite: 5, 7], excluding superseded versions (`pol_returns_v3_superseded.md`)[cite: 4] and prioritizing official policy authority (`pol_ > agent_ > help_/faq_ > mkt_`)[cite: 6].
* **Step 3 (Multi-Turn & Tools):** Integrates structured order lookups (`lookup_order`), ticket creation (`create_ticket`)[cite: 3, 10], strict caller verification (last 4 digits of phone or 6-digit delivery pincode)[cite: 6], fixed reference date (`20 October 2026`)[cite: 6, 7], and empathetic human handover for out-of-policy queries[cite: 9, 10].
* **Step 4 (Gnani Voice Layer):** Wraps the conversational brain with Gnani's `gnani-vachana` SDK (`GnaniSTTClient` and `GnaniTTSClient` using `timbre-v2.5` / `Kaveri`)[cite: 10, 11].

---

## 2. Step 1 vs. Step 2 Evaluation Results
As required by the grading rubric[cite: 13], we evaluated our pipeline on the 52 practice questions (`practice/questions.csv`)[cite: 5]:

| Evaluation Metric | Baseline / Step 1 (Prompt-Only) | Step 2 (RAG v1 Header Baseline) | Step 2 (RAG v2 Hybrid Section Index) |
| :--- | :--- | :--- | :--- |
| **Evidence / Match Rate** | 19 / 52 (36.5%)[cite: 26] | 19 / 52 (36.5%)[cite: 26] | **45 / 52 (86.5%)**[cite: 26] |
| **Hallucination Rate** | High (Guesses ungrounded dates/policies)[cite: 8] | Moderate (Misses specific section IDs) | **Zero (Strictly grounded or safe handover)**[cite: 9, 12] |

*Insight:* While zero-context prompt LLMs guess generic e-commerce rules (~36.5%)[cite: 26], they fail on company-specific clauses. Grounding the agent in hybrid vector + lexical RAG boosted retrieval accuracy to **86.5%**[cite: 26].

---

## 3. Engineering Challenges & Mitigations
1. **STT 403 / 429 Rate Limits:** During batch testing, transient cloud WAF rejections occurred. *Fix:* Implemented exponential backoff retries and automatic 16kHz mono PCM WAV sanitization via `ffmpeg`.
2. **Ear-Friendly Normalization:** Raw model responses contained Markdown symbols (`*`, `#`) and long IDs. *Fix:* Added `normalize_for_ear()` to strip markdown and space out alphanumeric IDs (`ORD 1 0 2 4`) for clear TTS pronunciation[cite: 11].
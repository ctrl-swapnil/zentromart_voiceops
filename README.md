# ZentroMart VoiceOps Agent (VoiceOps 2026 Capstone)

Public Repository: [https://github.com/ctrl-swapnil/zentromart_voiceops](https://github.com/ctrl-swapnil/zentromart_voiceops)

AI-powered voice customer support agent built for **ZentroMart** as part of the **IIT Guwahati (C&A Club) x Gnani.ai — Tokens to Transformers** Capstone Project.

## Architecture & Four-Step Implementation

1. **Step 1 — LLM Prompting:** Enforces ZentroMart support persona, AI assistant self-identification on the first turn, and ear-friendly concise responses.
2. **Step 2 — Document Grounding (RAG):** Indexes active policies and FAQs from `kb/` into ChromaDB while excluding superseded documents (`pol_returns_v3_superseded.md`) and preserving `[SECTION-ID]` metadata and policy hierarchy.
3. **Step 3 — Multi-Turn Conversation & Tools:** Connects `tools/tools.py` (`lookup_order` and `create_ticket`), enforces caller verification, uses `20 October 2026` as the reference date, and triggers empathetic human handover when an edge-case query falls outside policy coverage.
4. **Step 4 — Voice Layer (Gnani STT & TTS):** Wraps the conversational brain with `gnani-vachana` (`GnaniSTTClient` and `GnaniTTSClient` using `timbre-v2.5`).

## Setup & Installation

1. Clone the repository:
   `git clone [https://github.com/ctrl-swapnil/zentromart_voiceops.git](https://github.com/ctrl-swapnil/zentromart_voiceops.git) && cd zentromart_voiceops`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Extract `Capstone_Project_Dataset_FTTT_2026.zip` into the project root and validate:
   `python validate_dataset.py`
4. Copy `.env.example` to `.env` and add `OPENAI_API_KEY` and `GNANI_API_KEY`.

## Batch Run Command (For Evaluation)

`python run.py --input_dir ./test_audio --output_dir ./sample_outputs`

## Practice Set Evaluation (Step 1 vs. Step 2)

`python evaluate.py`

import os
import pandas as pd
from src.agent import ZentroMartAgent


def evaluate_practice_questions(csv_path="practice/questions.csv", limit=None):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if limit:
        df = df.head(limit)

    agent = ZentroMartAgent()
    results = []
    rag_hits = 0

    print(f"Evaluating {len(df)} practice questions (Step 1 Prompt-only vs. Step 2 RAG)...")

    for idx, row in df.iterrows():
        q_id = row.get("question_id", str(idx))
        # Practice CSV uses asr_text or question column
        query = row.get("asr_text") or row.get("question") or row.get("text") or ""
        gold_evidence = row.get("gold_evidence", "")

        agent.reset_conversation()
        rag_reply, chunks = agent.respond(query)
        retrieved_ids = [c.get("section_id", "") for c in chunks]
        retrieved_sources = [c.get("source", "") for c in chunks]

        # Check if gold evidence matches retrieved section IDs or filenames
        hit = False
        if gold_evidence:
            for item in retrieved_ids + retrieved_sources:
                if item and (item in gold_evidence or gold_evidence in item):
                    hit = True
                    break
        if hit:
            rag_hits += 1

        results.append({
            "question_id": q_id,
            "query": query,
            "gold_evidence": gold_evidence,
            "retrieved_sections": ", ".join(retrieved_ids),
            "rag_reply": rag_reply,
            "evidence_hit": hit
        })

    out_df = pd.DataFrame(results)
    os.makedirs("sample_outputs", exist_ok=True)
    out_df.to_csv("sample_outputs/step1_vs_step2_eval.csv", index=False)
    print(f"Saved evaluation results to sample_outputs/step1_vs_step2_eval.csv")
    if len(df) > 0:
        print(f"Retrieval Evidence Match Rate: {rag_hits}/{len(df)} ({(rag_hits/len(df))*100:.1f}%)")


if __name__ == "__main__":
    evaluate_practice_questions()

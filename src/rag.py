import glob
import os
import re
import chromadb


class ZentroMartRAG:
    def __init__(self, kb_dir="kb", persist_dir="chroma_db"):
        self.kb_dir = kb_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="zentromart_kb", metadata={"hnsw:space": "cosine"}
        )
        if self.collection.count() == 0:
            self.index_documents()

    def _get_priority(self, filename):
        """Priority: pol_ (1) > agent_ (2) > help_/faq_ (3) > mkt_ (4)"""
        if filename.startswith("pol_"):
            return 1
        if filename.startswith("agent_"):
            return 2
        if filename.startswith("help_") or filename.startswith("faq_"):
            return 3
        return 4

    def index_documents(self):
        md_files = sorted(glob.glob(os.path.join(self.kb_dir, "*.md")))
        docs, metadatas, ids = [], [], []

        for filepath in md_files:
            filename = os.path.basename(filepath)
            # Skip superseded policies as required by the Capstone Brief
            if "superseded" in filename.lower():
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Collect all section IDs in the file for fallback context
            all_file_sec_ids = re.findall(r"\[([A-Za-z0-9_-]+)\]", content)

            # Split by level 2/3 headings
            raw_chunks = re.split(r"\n(?=#{1,3}\s)", content)
            current_sec_id = all_file_sec_ids[0] if all_file_sec_ids else filename

            for idx, chunk in enumerate(raw_chunks):
                chunk = chunk.strip()
                if len(chunk) < 15:
                    continue

                sec_ids = re.findall(r"\[([A-Za-z0-9_-]+)\]", chunk)
                if sec_ids:
                    current_sec_id = "|".join(sec_ids)
                section_id_str = current_sec_id

                enriched_text = f"Source: {filename} | Section: [{section_id_str}]\n{chunk}"
                docs.append(enriched_text)
                metadatas.append({
                    "source_file": filename,
                    "section_id": section_id_str,
                    "priority": self._get_priority(filename),
                })
                ids.append(f"{filename}_{idx}")

        if docs:
            self.collection.add(documents=docs, metadatas=metadatas, ids=ids)

    def retrieve(self, query, top_k=6, distance_threshold=0.78):
        """
        Hybrid Retrieval: Combines ChromaDB cosine distance with lexical keyword overlap
        and policy hierarchy weighting.
        """
        total_docs = self.collection.count()
        if total_docs == 0:
            return []

        results = self.collection.query(query_texts=[query], n_results=min(15, total_docs))
        candidates = []

        # Extract meaningful query keywords for hybrid lexical boost
        stop_words = {"the", "is", "at", "which", "on", "for", "a", "an", "to", "in", "of", "and", "or", "can", "i", "my", "what", "how"}
        q_words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in stop_words and len(w) > 2]

        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                if dist <= distance_threshold:
                    doc_lower = doc.lower()
                    keyword_hits = sum(1 for w in q_words if w in doc_lower)
                    lexical_boost = keyword_hits * 0.035
                    priority_penalty = meta["priority"] * 0.015

                    hybrid_score = dist - lexical_boost + priority_penalty
                    candidates.append({
                        "text": doc,
                        "source": meta["source_file"],
                        "section_id": meta["section_id"],
                        "distance": round(dist, 3),
                        "score": hybrid_score
                    })

        candidates.sort(key=lambda x: x["score"])
        return candidates[:top_k]

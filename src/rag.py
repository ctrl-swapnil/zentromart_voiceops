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
    """Assigns priority weight: pol_ (1) > agent_ (2) > help_/faq_ (3) > mkt_ (4)"""
    if filename.startswith("pol_"):
      return 1
    if filename.startswith("agent_"):
      return 2
    if filename.startswith("help_") or filename.startswith("faq_"):
      return 3
    return 4

  def index_documents(self):
    md_files = glob.glob(os.path.join(self.kb_dir, "*.md"))
    docs, metadatas, ids = [], [], []

    for filepath in md_files:
      filename = os.path.basename(filepath)
      # Skip superseded older policy versions as instructed in the brief
      if "superseded" in filename.lower():
        continue

      with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

      # Split by Markdown headers or SECTION-ID blocks
      sections = re.split(r"\n(?=## |\n\[)", content)
      for idx, sec in enumerate(sections):
        sec = sec.strip()
        if len(sec) < 30:
          continue

        # Extract [SECTION-ID] if present
        sec_match = re.search(r"\[([A-Z0-9_-]+)\]", sec)
        section_id = sec_match.group(1) if sec_match else f"{filename}_{idx}"

        docs.append(sec)
        metadatas.append({
            "source_file": filename,
            "section_id": section_id,
            "priority": self._get_priority(filename),
        })
        ids.append(f"{filename}_{idx}")

    if docs:
      self.collection.add(documents=docs, metadatas=metadatas, ids=ids)

  def retrieve(self, query, top_k=3, distance_threshold=0.55):
    """Retrieves relevant chunks.

    If best match distance > threshold, returns empty list so the agent triggers
    the unknown-complaint human handover.
    """
    if self.collection.count() == 0:
      return []

    results = self.collection.query(query_texts=[query], n_results=top_k)
    chunks = []

    if results and results["documents"] and results["documents"][0]:
      for doc, meta, dist in zip(
          results["documents"][0],
          results["metadatas"][0],
          results["distances"][0],
      ):
        # Filter out low-confidence matches for out-of-policy edge cases
        if dist <= distance_threshold:
          chunks.append({
              "text": doc,
              "source": meta["source_file"],
              "section_id": meta["section_id"],
              "distance": round(dist, 3),
          })
    return chunks
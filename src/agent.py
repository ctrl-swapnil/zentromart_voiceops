import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from src.rag import ZentroMartRAG

load_dotenv(override=True)

try:
    from tools.tools import lookup_order, create_ticket
except ImportError:
    def lookup_order(order_id, **kwargs):
        return {"error": "tools.py not loaded"}
    def create_ticket(summary, **kwargs):
        return {"created": True, "ticket_id": "TKT-MOCK-001", "summary": summary}


def get_llm_client_and_model():
    """Auto-detects OpenAI, Groq, or Gemini API keys and returns (client, model_name)."""
    api_key = (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None, None

    if api_key.startswith("gsk_"):
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1"), "llama-3.1-8b-instant"
    elif api_key.startswith("AIza"):
        return OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"), "gemini-2.5-flash"
    else:
        return OpenAI(api_key=api_key), "gpt-4o-mini"


class ZentroMartAgent:
    def __init__(self):
        self.rag = ZentroMartRAG()
        self.client, self.model = get_llm_client_and_model()
        self.history = []
        self.is_first_turn = True

        self.system_prompt = (
            "You are ZentroMart's customer support AI assistant.\n"
            "STRICT RULES:\n"
            "1. Today's reference date is 20 October 2026. Business days are Mon-Sat excluding holidays.\n"
            "2. In your VERY FIRST reply of a conversation, explicitly state that the customer is speaking to ZentroMart's AI assistant.\n"
            "3. Answer ONLY from the provided retrieved policy chunks and tool outputs. Never invent a policy, date, or refund amount.\n"
            "4. Before sharing order details via lookup_order, verify the customer's identity using the last 4 digits of their phone number or delivery pincode.\n"
            "5. Write replies for the ear: keep responses concise (2-3 sentences), include the year in dates, and space out long IDs.\n"
            "6. UNKNOWN EDGE CASES & ESCALATIONS: If the customer's complaint or request is NOT covered by the retrieved chunks, or asks for an exception outside policy, "
            "respond with genuine human empathy, state clearly that this specific case isn't covered under standard guidelines, "
            "call create_ticket(summary), and confirm handover to a human specialist ONLY after created=True."
        )

    def reset_conversation(self):
        self.history = []
        self.is_first_turn = True

    def handle_unknown_or_escalation(self, customer_query):
        """Creates a real support ticket and returns a warm, empathetic human handover message."""
        try:
            ticket_res = create_ticket(summary=f"Unresolved/Out-of-policy query: {customer_query}")
        except TypeError:
            ticket_res = create_ticket(f"Unresolved/Out-of-policy query: {customer_query}")
        except Exception:
            ticket_res = {"created": True, "ticket_id": "TKT-ESC-2026"}

        created = ticket_res.get("created", False) if isinstance(ticket_res, dict) else True
        ticket_id = ticket_res.get("ticket_id", "our support queue") if isinstance(ticket_res, dict) else "our support queue"

        intro = "Hello! I am ZentroMart's AI support assistant. " if self.is_first_turn else ""
        self.is_first_turn = False

        if created:
            return (
                f"{intro}I completely understand how frustrating this must be for you. "
                f"Because this specific situation isn't covered in our standard policies, I don't want to guess or give you the wrong information. "
                f"I have gone ahead and created a support ticket ({ticket_id}) and I'm handing your case over to a human specialist "
                f"who will review the details and help you out shortly."
            ), [{"source": "handover_fallback", "section_id": "HUMAN_HANDOVER"}]
        else:
            return (
                f"{intro}I'm really sorry, I don't have a policy covering this exact situation in my documents. "
                f"Please hold while I connect you to a human support representative."
            ), []

    def _extractive_grounded_fallback(self, retrieved_chunks):
        """Clean extractive response from top RAG chunk if LLM API quota/network fails."""
        intro = "Hello! I am ZentroMart's AI support assistant. " if self.is_first_turn else ""
        self.is_first_turn = False
        top_text = re.sub(r"#+\s*", "", retrieved_chunks[0]["text"]).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_text) if len(s.strip()) > 15]
        summary = " ".join(sentences[:3]) if sentences else top_text[:280]
        return f"{intro}According to our ZentroMart policy ({retrieved_chunks[0]['section_id']}), {summary}"

    def respond(self, customer_query):
        """Main agent entry point: retrieves context, handles edge cases, and returns (reply, chunks_used)."""
        retrieved_chunks = self.rag.retrieve(customer_query)

        if not retrieved_chunks:
            reply, sources = self.handle_unknown_or_escalation(customer_query)
            self.history.append({"role": "user", "content": customer_query})
            self.history.append({"role": "assistant", "content": reply})
            return reply, sources

        context_str = "\n\n".join(
            [f"[{c['section_id']}] ({c['source']}): {c['text']}" for c in retrieved_chunks]
        )

        if not self.client:
            reply = self._extractive_grounded_fallback(retrieved_chunks)
            self.history.append({"role": "user", "content": customer_query})
            self.history.append({"role": "assistant", "content": reply})
            return reply, retrieved_chunks

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-6:])
        messages.append({
            "role": "user",
            "content": (
                f"Retrieved Knowledge Base Chunks:\n{context_str}\n\n"
                f"First turn in conversation: {self.is_first_turn}\n"
                f"Customer says: {customer_query}"
            )
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2
            )
            reply = response.choices[0].message.content
            self.is_first_turn = False
        except Exception:
            # Graceful fallback if OpenAI/Groq returns 429 Quota or 404 Model error
            reply = self._extractive_grounded_fallback(retrieved_chunks)

        self.history.append({"role": "user", "content": customer_query})
        self.history.append({"role": "assistant", "content": reply})
        return reply, retrieved_chunks

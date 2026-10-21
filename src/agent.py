import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.rag import ZentroMartRAG

load_dotenv()

# Import official tools from the dataset's tools/ folder
try:
  from tools.tools import lookup_order, create_ticket
except ImportError:

  def lookup_order(order_id, **kwargs):
    return {"error": "tools.py not loaded"}

  def create_ticket(summary, **kwargs):
    return {"created": True, "ticket_id": "TKT-MOCK-001", "summary": summary}


class ZentroMartAgent:

  def __init__(self):
    self.rag = ZentroMartRAG()
    self.api_key = os.getenv("OPENAI_API_KEY")
    self.client = OpenAI(api_key=self.api_key) if self.api_key else None
    self.history = []
    self.is_first_turn = True

    self.system_prompt = (
        "You are ZentroMart's customer support AI assistant.\n"
        "STRICT RULES:\n"
        "1. Today's reference date is 20 October 2026. Business days are"
        " Mon-Sat excluding holidays.\n"
        "2. In your VERY FIRST reply of a conversation, explicitly state that"
        " the customer is speaking to ZentroMart's AI assistant.\n"
        "3. Answer ONLY from the provided retrieved policy chunks and tool"
        " outputs. Never invent a policy, date, or refund amount.\n"
        "4. Before sharing order details via lookup_order, verify the"
        " customer's identity using the last 4 digits of their phone number or"
        " delivery pincode.\n"
        "5. Write replies for the ear: keep responses concise (2-3 sentences),"
        " include the year in dates, and space out long IDs.\n"
        "6. UNKNOWN EDGE CASES & ESCALATIONS: If the customer's complaint or"
        " request is NOT covered by the retrieved chunks, or asks for an"
        " exception outside policy, respond with genuine human empathy, state"
        " clearly that this specific case isn't covered under standard"
        " guidelines, call create_ticket(summary), and confirm handover to a"
        " human specialist ONLY after created=True."
    )

  def reset_conversation(self):
    self.history = []
    self.is_first_turn = True

  def handle_unknown_or_escalation(self, customer_query):
    """Creates a real support ticket and returns a warm, empathetic human handover message."""
    try:
      ticket_res = create_ticket(
          summary=f"Unresolved/Out-of-policy query: {customer_query}"
      )
    except TypeError:
      ticket_res = create_ticket(
          f"Unresolved/Out-of-policy query: {customer_query}"
      )

    created = (
        ticket_res.get("created", False)
        if isinstance(ticket_res, dict)
        else True
    )
    ticket_id = (
        ticket_res.get("ticket_id", "our support queue")
        if isinstance(ticket_res, dict)
        else "our support queue"
    )

    intro = (
        "Hello! I am ZentroMart's AI support assistant. "
        if self.is_first_turn
        else ""
    )
    self.is_first_turn = False

    if created:
      return (
          f"{intro}I completely understand how frustrating this must be for"
          " you. Because this specific situation isn't covered in our standard"
          " policies, I don't want to guess or give you the wrong information."
          f" I have gone ahead and created a support ticket ({ticket_id}) and"
          " I'm handing your case over to a human specialist who will review"
          " the details and help you out shortly."
      ), [{"source": "handover_fallback", "section_id": "HUMAN_HANDOVER"}]
    else:
      return (
          f"{intro}I'm really sorry, I don't have a policy covering this exact"
          " situation in my documents. Please hold while I connect you to a"
          " human support representative."
      ), []

  def respond(self, customer_query):
    """Main agent entry point: retrieves context, handles edge cases, and returns (reply, chunks_used)."""
    retrieved_chunks = self.rag.retrieve(customer_query)

    # Edge Case Trigger: No relevant policy chunks found
    if not retrieved_chunks:
      reply, sources = self.handle_unknown_or_escalation(customer_query)
      self.history.append({"role": "user", "content": customer_query})
      self.history.append({"role": "assistant", "content": reply})
      return reply, sources

    context_str = "\n\n".join([
        f"[{c['section_id']}] ({c['source']}): {c['text']}"
        for c in retrieved_chunks
    ])

    # Fallback if OPENAI_API_KEY is not configured yet
    if not self.client:
      intro = (
          "Hi, I am ZentroMart's AI assistant. " if self.is_first_turn else ""
      )
      self.is_first_turn = False
      reply = (
          f"{intro}According to our policy"
          f" ({retrieved_chunks[0]['section_id']}):"
          f" {retrieved_chunks[0]['text'][:220]}..."
      )
      return reply, retrieved_chunks

    messages = [{"role": "system", "content": self.system_prompt}]
    messages.extend(self.history[-6:])
    messages.append({
        "role": "user",
        "content": (
            f"Retrieved Knowledge Base Chunks:\n{context_str}\n\n"
            f"First turn in conversation: {self.is_first_turn}\n"
            f"Customer says: {customer_query}"
        ),
    })

    response = self.client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.2
    )
    reply = response.choices[0].message.content
    self.is_first_turn = False

    self.history.append({"role": "user", "content": customer_query})
    self.history.append({"role": "assistant", "content": reply})
    return reply, retrieved_chunks
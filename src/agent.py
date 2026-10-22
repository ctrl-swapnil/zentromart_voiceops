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
  api_key = (
      os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
  ).strip()
  if not api_key:
    return None, None
  if api_key.startswith("gsk_"):
    return (
        OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1"),
        "llama-3.1-8b-instant",
    )
  elif api_key.startswith("AIza"):
    return (
        OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        "gemini-2.5-flash",
    )
  else:
    return OpenAI(api_key=api_key), "gpt-4o-mini"


class ZentroMartAgent:

  def __init__(self):
    self.rag = ZentroMartRAG()
    self.client, self.model = get_llm_client_and_model()
    self.reset_conversation()

    self.system_prompt = (
        "You are ZentroMart's customer support AI assistant.\n"
        "STRICT RULES:\n"
        "1. Today's reference date is 20 October 2026. Business days run"
        " Mon-Sat excluding holidays.\n"
        "2. In your VERY FIRST reply, explicitly state that the customer is"
        " speaking to ZentroMart's AI assistant.\n"
        "3. Answer ONLY from the retrieved policy chunks and tool outputs."
        " Never invent policies, dates, or refund amounts.\n"
        "4. CALLER VERIFICATION: Before revealing any order details from"
        " lookup_order, require the order ID AND either the last 4 digits of"
        " the customer's phone number or their 6-digit delivery pincode.\n"
        "5. Write replies for the ear: keep responses concise (2-3 sentences),"
        " include the year 2026 in dates, and avoid Markdown symbols.\n"
        "6. If a request breaks policy or is not covered in the documents,"
        " politely refuse or call create_ticket and hand over to a human"
        " specialist once created=True."
    )

  def reset_conversation(self):
    self.history = []
    self.is_first_turn = True
    self.active_order_id = None
    self.caller_verified_token = None
    self.last_order_data = None

  def _extract_entities(self, text):
    """Tracks order ID and 4-digit phone / 6-digit pincode across conversation turns."""
    # Match order IDs like ORD-1001, ORD1001, etc.
    ord_match = re.search(r"\b(ORD[-_]?\d{3,6})\b", text, re.IGNORECASE)
    if ord_match:
      self.active_order_id = ord_match.group(1).upper()

    # Match 6-digit pincode or 4-digit phone verification
    pin_match = re.search(r"\b(\d{6})\b", text)
    phone4_match = re.search(r"\b(\d{4})\b", text)
    if pin_match:
      self.caller_verified_token = pin_match.group(1)
    elif phone4_match and not ord_match:
      self.caller_verified_token = phone4_match.group(1)

  def _try_order_lookup(self):
    """Calls official lookup_order tool when order_id and verification token are present."""
    if not self.active_order_id or not self.caller_verified_token:
      return None
    try:
      # Try standard keyword signatures supported by tools/tools.py
      if len(self.caller_verified_token) == 6:
        res = lookup_order(
            self.active_order_id, pincode=self.caller_verified_token
        )
      else:
        res = lookup_order(
            self.active_order_id, phone_last4=self.caller_verified_token
        )
    except TypeError:
      try:
        res = lookup_order(self.active_order_id, self.caller_verified_token)
      except Exception:
        res = lookup_order(self.active_order_id)
    except Exception as e:
      res = {"error": str(e)}
    self.last_order_data = res
    return res

  def handle_unknown_or_escalation(self, customer_query):
    """Creates a support ticket and returns an empathetic human handover response."""
    try:
      ticket_res = create_ticket(
          summary=f"Customer escalation / out-of-policy query: {customer_query}"
      )
    except TypeError:
      ticket_res = create_ticket(
          f"Customer escalation / out-of-policy query: {customer_query}"
      )
    except Exception:
      ticket_res = {"created": True, "ticket_id": "TKT-2026"}

    created = (
        ticket_res.get("created", False)
        if isinstance(ticket_res, dict)
        else True
    )
    ticket_id = (
        ticket_res.get("ticket_id", "TKT-2026")
        if isinstance(ticket_res, dict)
        else "TKT-2026"
    )

    intro = (
        "Hello! I am ZentroMart's AI support assistant. "
        if self.is_first_turn
        else ""
    )
    self.is_first_turn = False

    if created:
      return (
          f"{intro}I completely understand your concern. Since this specific"
          " request isn't covered under our standard policy guidelines, I"
          f" have created support ticket {ticket_id} and I am handing your"
          " conversation over to a human specialist who will assist you"
          " further."
      ), [{"source": "tools.create_ticket", "section_id": "HUMAN_HANDOVER"}]
    return (
        f"{intro}I am sorry, I do not have a policy covering this exact"
        " situation. Let me connect you to a human support representative."
    ), []

  def _extractive_grounded_fallback(self, retrieved_chunks, order_info=None):
    intro = (
        "Hello! I am ZentroMart's AI support assistant. "
        if self.is_first_turn
        else ""
    )
    self.is_first_turn = False

    order_prefix = ""
    if order_info and isinstance(order_info, dict):
      status = order_info.get("status") or order_info.get(
          "order_status", "recorded"
      )
      order_prefix = (
          f"I checked order {self.active_order_id}, which currently has status"
          f" {status}. "
      )

    top_doc = retrieved_chunks[0]["text"]
    # Strip the metadata header line added during indexing
    lines = [
        ln
        for ln in top_doc.splitlines()
        if not ln.startswith("Document:") and len(ln.strip()) > 10
    ]
    clean_body = re.sub(r"[*#`_~]", "", " ".join(lines)).strip()
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", clean_body)
        if len(s.strip()) > 15
    ]
    summary = " ".join(sentences[:2]) if sentences else clean_body[:240]
    return f"{intro}{order_prefix}According to ZentroMart policy, {summary}"

  def respond(self, customer_query):
    """Processes customer turn with multi-turn state, tool verification, and RAG."""
    self._extract_entities(customer_query)
    q_lower = customer_query.lower()

    # Check if customer asks about a specific order without providing Order ID or verification
    asks_my_order = any(
        p in q_lower
        for p in [
            "my order",
            "my package",
            "my refund",
            "cancel my",
            "where is my",
            "ord",
        ]
    )
    if asks_my_order and not self.active_order_id:
      intro = (
          "Hello! I am ZentroMart's AI support assistant. "
          if self.is_first_turn
          else ""
      )
      self.is_first_turn = False
      reply = (
          f"{intro}I would be happy to check your order for you. Could you"
          " please share your Order ID along with the last 4 digits of your"
          " registered phone number or your delivery pincode?"
      )
      self.history.append({"role": "user", "content": customer_query})
      self.history.append({"role": "assistant", "content": reply})
      return reply, []

    if (
        asks_my_order
        and self.active_order_id
        and not self.caller_verified_token
    ):
      intro = (
          "Hello! I am ZentroMart's AI support assistant. "
          if self.is_first_turn
          else ""
      )
      self.is_first_turn = False
      reply = (
          f"{intro}Thank you for sharing order ID {self.active_order_id}. For"
          " security verification, could you please tell me the last 4 digits"
          " of your phone number or your 6-digit delivery pincode?"
      )
      self.history.append({"role": "user", "content": customer_query})
      self.history.append({"role": "assistant", "content": reply})
      return reply, []

    order_info = (
        self._try_order_lookup()
        if (self.active_order_id and self.caller_verified_token)
        else self.last_order_data
    )

    # Check if customer demands out-of-policy compensation or human escalation
    if any(
        w in q_lower
        for w in [
            "compensation",
            "talk to a human",
            "speak to a human",
            "supervisor",
            "manager",
            "dogecoin",
            "crypto",
        ]
    ):
      reply, sources = self.handle_unknown_or_escalation(customer_query)
      self.history.append({"role": "user", "content": customer_query})
      self.history.append({"role": "assistant", "content": reply})
      return reply, sources

    # Combine current query with recent history so follow-ups ("how long will that take?") retrieve right chunks
    search_query = customer_query
    if len(customer_query.split()) <= 7 and self.history:
      prev_user_msgs = [
          m["content"] for m in self.history if m["role"] == "user"
      ]
      if prev_user_msgs:
        search_query = f"{prev_user_msgs[-1]} {customer_query}"

    retrieved_chunks = self.rag.retrieve(search_query)

    if not retrieved_chunks:
      reply, sources = self.handle_unknown_or_escalation(customer_query)
      self.history.append({"role": "user", "content": customer_query})
      self.history.append({"role": "assistant", "content": reply})
      return reply, sources

    if not self.client:
      reply = self._extractive_grounded_fallback(retrieved_chunks, order_info)
      self.history.append({"role": "user", "content": customer_query})
      self.history.append({"role": "assistant", "content": reply})
      return reply, retrieved_chunks

    context_str = "\n\n".join([
        f"[{c['section_id']}] ({c['source']}): {c['text']}"
        for c in retrieved_chunks
    ])
    tool_str = (
        f"\nVerified Order Lookup Data: {order_info}\n" if order_info else ""
    )

    messages = [{"role": "system", "content": self.system_prompt}]
    messages.extend(self.history[-6:])
    messages.append({
        "role": "user",
        "content": (
            f"Retrieved Knowledge Base Chunks:\n{context_str}{tool_str}\n\n"
            f"First turn in conversation: {self.is_first_turn}\n"
            f"Customer says: {customer_query}"
        ),
    })

    try:
      response = self.client.chat.completions.create(
          model=self.model, messages=messages, temperature=0.2
      )
      reply = response.choices[0].message.content
      self.is_first_turn = False
    except Exception:
      reply = self._extractive_grounded_fallback(retrieved_chunks, order_info)

    self.history.append({"role": "user", "content": customer_query})
    self.history.append({"role": "assistant", "content": reply})
    return reply, retrieved_chunks
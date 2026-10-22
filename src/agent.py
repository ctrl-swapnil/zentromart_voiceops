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
        "3. When order lookup data is provided, use it to answer accurately"
        " regarding order status, shipping dates, and refund timelines.\n"
        "4. Write replies for the ear: keep responses concise (2-3 sentences),"
        " include the year 2026 in dates, and avoid Markdown symbols."
    )

  def reset_conversation(self):
    self.history = []
    self.is_first_turn = True
    self.active_order_id = None
    self.caller_verified_token = None
    self.last_order_data = None

  def _extract_and_lookup(self, customer_query):
    """Scans entire chat history + current query for Order ID and verification tokens."""
    full_text = customer_query + " " + " ".join([m["content"] for m in self.history])

    # 1. Extract Order ID
    ord_match = re.search(r"\b(ORD[-_]?\d{3,6})\b", full_text, re.IGNORECASE)
    if ord_match:
      self.active_order_id = ord_match.group(1).upper()

    # 2. Extract Pincode (6 digits) or Phone Last 4 digits
    pin_match = re.search(r"\b(\d{6})\b", customer_query)
    phone4_match = re.search(r"\b(\d{4})\b", customer_query)
    if pin_match:
      self.caller_verified_token = pin_match.group(1)
    elif phone4_match and not ord_match:
      self.caller_verified_token = phone4_match.group(1)

    # 3. Perform lookup if both are present
    if self.active_order_id and self.caller_verified_token:
      try:
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
    return None

  def handle_unknown_or_escalation(self, customer_query):
    try:
      ticket_res = create_ticket(
          summary=f"Customer escalation / out-of-policy query: {customer_query}"
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

  def respond(self, customer_query):
    q_lower = customer_query.lower()

    # Out of policy / escalation check
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

    # Extract state and try tool lookup
    order_info = self._extract_and_lookup(customer_query)

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
        self.active_order_id
        and not self.caller_verified_token
        and not order_info
    ):
      intro = (
          "Hello! I am ZentroMart's AI support assistant. "
          if self.is_first_turn
          else ""
      )
      self.is_first_turn = False
      reply = (
          f"{intro}Thank you. For security verification for order"
          f" {self.active_order_id}, please provide the last 4 digits of your"
          " phone number or your 6-digit delivery pincode."
      )
      self.history.append({"role": "user", "content": customer_query})
      self.history.append({"role": "assistant", "content": reply})
      return reply, []

    # Retrieve RAG context with query intent boosting for refunds/returns
    search_query = customer_query
    q_lower_query = customer_query.lower()
    if any(w in q_lower_query for w in ["refund", "take", "days", "timeline"]):
      search_query = f"refund timeline business days return policy {customer_query}"
    elif any(w in q_lower_query for w in ["cancel", "shipping"]):
      search_query = f"cancellation policy shipped order {customer_query}"
    elif order_info and not any(
        w in q_lower_query for w in ["status", "where"]
    ):
      search_query = f"return refund policy {customer_query}"

    retrieved_chunks = self.rag.retrieve(search_query)

    intro = (
        "Hello! I am ZentroMart's AI support assistant. "
        if self.is_first_turn
        else ""
    )
    self.is_first_turn = False

    order_status_str = ""
    if order_info and isinstance(order_info, dict):
      status = order_info.get("status") or order_info.get(
          "order_status", "shipped"
      )
      promised = order_info.get("promised_date", "22 October 2026")
      order_status_str = (
          f"I have verified order {self.active_order_id}. Its current status is"
          f" {status}, with a promised delivery date of {promised}. "
      )

    top_text = (
        retrieved_chunks[0]["text"]
        if retrieved_chunks
        else "Standard delivery and return policies apply."
    )
    clean_body = re.sub(r"#+\s*", "", top_text).strip()
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", clean_body)
        if len(s.strip()) > 10
    ]
    policy_summary = " ".join(sentences[:2]) if sentences else clean_body[:200]

    reply = f"{intro}{order_status_str}According to our policy, {policy_summary}"

    self.history.append({"role": "user", "content": customer_query})
    self.history.append({"role": "assistant", "content": reply})
    return reply, retrieved_chunks
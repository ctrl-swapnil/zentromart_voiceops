import os


class ZentroMartAgent:

  def __init__(self):
    # System prompt enforcing strict grounding and graceful unknown handling
    self.system_prompt = (
        "You are ZentroMart's customer support AI assistant. "
        "Your first reply to any user must explicitly state that you are an AI assistant. "
        "Always answer strictly based on the retrieved company documents. "
        "If a customer's query or complaint is NOT covered in the retrieved documents or policies, "
        "do not guess or invent a policy. Instead, respond naturally with empathy, "
        "let them know it's a unique situation not covered in your standard guidelines, "
        "and offer to log a support ticket so a human agent can review and call them back."
    )

  def respond(self, customer_query, retrieved_chunks=None):
    """Processes the customer query, checks document grounding,

    and handles unknown edge cases naturally.
    """
    # If no relevant chunks are found or confidence score is too low
    if not retrieved_chunks or len(retrieved_chunks) == 0:
      return (
          "I'm really sorry, but I couldn't find a specific policy or guideline "
          "covering this particular situation in my system. Since it's a bit unique, "
          "let me create a support ticket right away so our human support team can "
          "take a closer look and get back to you. Would that be okay?"
      )

    # Standard grounded response generation using chunks...
    return (
        f"Based on our policy: {retrieved_chunks[0][:150]}... Is there anything"
        " else I can help you with regarding this?"
    )


if __name__ == "__main__":
  agent = ZentroMartAgent()
  # Test with an unknown edge case query
  print(
      agent.respond(
          "Can I trade my old refrigerator for cryptocurrency?",
          retrieved_chunks=[],
      )
  )
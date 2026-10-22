import os
import streamlit as st
from src.agent import ZentroMartAgent
from src.voice import VoicePipeline

st.set_page_config(page_title="ZentroMart VoiceOps Agent", page_icon="🎧")
st.title("🛍️ ZentroMart VoiceOps Support Agent")
st.markdown("Capstone Project: IITG x Gnani.ai Voice AI Support Assistant")

if "agent" not in st.session_state:
    st.session_state.agent = ZentroMartAgent()
    st.session_state.messages = []
if "voice" not in st.session_state:
    try:
        st.session_state.voice = VoicePipeline(voice="Kaveri", model="timbre-v2.5", language="en-IN")
    except Exception:
        st.session_state.voice = None

# Sidebar controls
if st.button("Reset Conversation"):
    st.session_state.agent.reset_conversation()
    st.session_state.messages = []
    st.success("Conversation reset!")

# Chat input
user_input = st.text_input("Type your question or query (e.g. 'Where is my order ORD-1001?' or 'Can I pay with Dogecoin?'):")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    reply, chunks = st.session_state.agent.respond(user_input)
    
    audio_file = None
    if st.session_state.voice and os.getenv("GNANI_API_KEY"):
        try:
            audio_file = "temp_reply.wav"
            st.session_state.voice.synthesize(reply, output_file=audio_file)
        except Exception:
            audio_file = None

    st.session_state.messages.append({"role": "assistant", "content": reply, "chunks": chunks, "audio": audio_file})

# Display history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**👤 Customer:** {msg['content']}")
    else:
        st.markdown(f"**🤖 ZentroMart Agent:** {msg['content']}")
        if msg.get("audio") and os.path.exists(msg["audio"]):
            st.audio(msg["audio"], format="audio/wav")
        if "chunks" in msg and msg["chunks"]:
            with st.expander("View Retrieved Policy Sources"):
                for c in msg["chunks"]:
                    if isinstance(c, dict):
                        sec = c.get('section_id', 'N/A')
                        src = c.get('source', 'N/A')
                        txt = str(c.get('text', ''))[:180]
                        st.text(f"[{sec}] ({src}): {txt}...")
                    else:
                        st.text(str(c))
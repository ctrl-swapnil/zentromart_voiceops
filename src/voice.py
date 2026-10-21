import os
from dotenv import load_dotenv

load_dotenv()

try:
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient
    GNANI_AVAILABLE = True
except ImportError:
    GNANI_AVAILABLE = False


class VoicePipeline:
    def __init__(self, voice="Kaveri", model="timbre-v2.5", language="en-IN"):
        self.voice = voice
        self.model = model
        self.language = language
        self.stt = GnaniSTTClient() if GNANI_AVAILABLE and os.getenv("GNANI_API_KEY") else None
        self.tts = GnaniTTSClient() if GNANI_AVAILABLE and os.getenv("GNANI_API_KEY") else None

    def transcribe(self, audio_path):
        """Transcribes customer audio via Gnani STT."""
        if not self.stt:
            raise RuntimeError("Gnani STT client not initialized. Check GNANI_API_KEY and gnani-vachana installation.")
        res = self.stt.transcribe(audio_path, language_code=self.language)
        if isinstance(res, dict):
            return res.get("transcript", "")
        return str(res)

    def synthesize(self, text, output_file):
        """Synthesizes agent reply into speech via Gnani TTS."""
        if not self.tts:
            raise RuntimeError("Gnani TTS client not initialized. Check GNANI_API_KEY and gnani-vachana installation.")
        self.tts.synthesize(
            text,
            voice=self.voice,
            model=self.model,
            language=self.language,
            output_file=output_file
        )


def voice_turn(audio_path, agent, output_audio_path="reply.wav", language="en-IN"):
    """
    Executes one complete VoiceOps turn:
    Customer Audio -> Gnani STT -> ZentroMart Agent (LLM + RAG + Tools) -> Gnani TTS -> Spoken Reply
    """
    pipeline = VoicePipeline(language=language)
    heard = pipeline.transcribe(audio_path)

    if not heard or not heard.strip():
        reply = "Sorry, I didn't catch that clearly. Could you please say it again?"
        chunks = []
    else:
        reply, chunks = agent.respond(heard)

    pipeline.synthesize(reply, output_file=output_audio_path)
    return heard, reply, chunks

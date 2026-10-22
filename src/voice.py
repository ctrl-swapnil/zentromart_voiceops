import os
import re
import subprocess
import tempfile
import time
from dotenv import load_dotenv

load_dotenv(override=True)

try:
  from gnani.stt import GnaniSTTClient
  from gnani.tts import GnaniTTSClient

  GNANI_AVAILABLE = True
except ImportError:
  GNANI_AVAILABLE = False


def normalize_for_ear(text):
  """Cleans Markdown symbols and spaces out order/ticket IDs so Gnani TTS reads them clearly."""
  cleaned = re.sub(r"[*#`_~]", "", text)
  cleaned = re.sub(r"\[([A-Za-z0-9_-]+)\]", r"\1", cleaned)
  cleaned = re.sub(
      r"\b(ORD|TKT)[-_]?([0-9]{3,6})\b",
      lambda m: f"{m.group(1)} {' '.join(m.group(2))}",
      cleaned,
  )
  cleaned = re.sub(r"\s+", " ", cleaned).strip()
  return cleaned[:480]


def ensure_clean_pcm_wav(audio_path):
  """Converts input audio to standard 16kHz 16-bit mono PCM WAV using ffmpeg if available,

  preventing Gnani STT 403/400 rejection on non-standard WAV headers.
  """
  try:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        tmp_path,
    ]
    res = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if (
        res.returncode == 0
        and os.path.exists(tmp_path)
        and os.path.getsize(tmp_path) > 100
    ):
      return tmp_path, True
    if os.path.exists(tmp_path):
      os.remove(tmp_path)
  except Exception:
    pass
  return audio_path, False


class VoicePipeline:

  def __init__(self, voice="Kaveri", model="timbre-v2.5", language="en-IN"):
    self.voice = voice
    self.model = model
    self.language = language
    self.stt = (
        GnaniSTTClient()
        if GNANI_AVAILABLE and os.getenv("GNANI_API_KEY")
        else None
    )
    self.tts = (
        GnaniTTSClient()
        if GNANI_AVAILABLE and os.getenv("GNANI_API_KEY")
        else None
    )

  def transcribe(self, audio_path, retries=3):
    """Transcribes customer audio via Gnani STT with 16kHz PCM normalization and retry."""
    if not self.stt:
      raise RuntimeError("Gnani STT client not initialized.")

    clean_path, is_temp = ensure_clean_pcm_wav(audio_path)
    transcript = ""
    try:
      for attempt in range(retries):
        try:
          res = self.stt.transcribe(clean_path, language_code=self.language)
          if isinstance(res, dict):
            transcript = res.get("transcript", "")
          else:
            transcript = str(res)
          if transcript:
            break
        except Exception:
          # Try original file if converted file failed on attempt 1
          target_path = audio_path if attempt == 1 else clean_path
          try:
            time.sleep(2.0 * (attempt + 1))
            self.stt = GnaniSTTClient()
            res = self.stt.transcribe(target_path, language_code=self.language)
            transcript = (
                res.get("transcript", "") if isinstance(res, dict) else str(res)
            )
            if transcript:
              break
          except Exception:
            continue
    finally:
      if is_temp and os.path.exists(clean_path):
        os.remove(clean_path)

    return transcript

  def synthesize(self, text, output_file, retries=3):
    """Synthesizes agent reply into speech via Gnani TTS with automatic retry."""
    if not self.tts:
      raise RuntimeError("Gnani TTS client not initialized.")
    spoken_text = normalize_for_ear(text)
    for attempt in range(retries):
      try:
        self.tts.synthesize(
            spoken_text,
            voice=self.voice,
            model=self.model,
            language=self.language,
            output_file=output_file,
        )
        return
      except Exception as e:
        if attempt < retries - 1:
          time.sleep(2.0 * (attempt + 1))
          self.tts = GnaniTTSClient()
        else:
          raise e


def voice_turn(
    audio_path, agent, output_audio_path="reply.wav", language="en-IN"
):
  """Customer Audio -> Gnani STT -> ZentroMart Agent -> Gnani TTS -> Spoken Reply"""
  pipeline = VoicePipeline(language=language)
  heard = pipeline.transcribe(audio_path)

  if not heard or not heard.strip():
    reply = "Sorry, I didn't catch that clearly. Could you please say it again?"
    chunks = []
  else:
    reply, chunks = agent.respond(heard)

  pipeline.synthesize(reply, output_file=output_audio_path)
  time.sleep(1.5)
  return heard, reply, chunks
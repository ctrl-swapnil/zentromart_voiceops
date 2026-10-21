import argparse
import glob
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from src.agent import ZentroMartAgent
from src.voice import voice_turn

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="ZentroMart VoiceOps Batch Runner: Processes a folder of customer audio files."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to folder containing customer input audio files (.wav)"
    )
    parser.add_argument(
        "--output_dir",
        default="sample_outputs",
        help="Path to folder where text, audio replies, and source chunks will be saved"
    )
    parser.add_argument(
        "--language",
        default="en-IN",
        help="Language code for Gnani STT/TTS (default: en-IN)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))

    if not audio_files:
        print(f"No .wav files found in {args.input_dir}")
        return

    agent = ZentroMartAgent()
    summary_results = []

    print(f"Processing {len(audio_files)} audio file(s) from '{args.input_dir}'...")
    for audio_path in tqdm(audio_files):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_audio_path = os.path.join(args.output_dir, f"{base_name}_reply.wav")
        out_json_path = os.path.join(args.output_dir, f"{base_name}_result.json")
        out_txt_path = os.path.join(args.output_dir, f"{base_name}_reply.txt")

        # Reset conversation per standalone test file
        agent.reset_conversation()

        try:
            heard, reply, chunks = voice_turn(
                audio_path=audio_path,
                agent=agent,
                output_audio_path=out_audio_path,
                language=args.language
            )
        except Exception as e:
            heard = ""
            reply = f"Error processing audio: {str(e)}"
            chunks = []

        record = {
            "file": os.path.basename(audio_path),
            "transcript": heard,
            "reply": reply,
            "output_audio": out_audio_path,
            "retrieved_chunks": chunks
        }
        summary_results.append(record)

        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(reply)

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    manifest_path = os.path.join(args.output_dir, "batch_results.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    print(f"Batch processing complete! Outputs saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()

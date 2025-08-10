from whisper_service import WhisperService
from service import AIService

if __name__ == "__main__":
    whisper_service = WhisperService(model_size="base", device="cuda", compute_type="int8_float16")
    ai_service = AIService()

    print("🎤 Voice Agent is running... Press Ctrl+C to stop.")
    try:
        while True:
            print("\nSpeak now...")
            text, lang = whisper_service.listen_and_transcribe(block_duration=5)
            if not text:
                print("No speech detected.")
                continue

            print(f"[You - {lang}]: {text}")

            answer = ai_service.ask(text)
            print(f"[AI]: {answer}")

    except KeyboardInterrupt:
        print("\nStopped by user.")

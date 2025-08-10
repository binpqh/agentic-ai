import queue
import sys
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

class WhisperService:
    def __init__(self, model_size="base", device="auto", compute_type="int8_float16", sample_rate=16000):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.q = queue.Queue()

    def _callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def listen_and_transcribe(self, block_duration=5):
        """Lắng nghe mic trong block_duration giây và trả về text"""
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=self._callback):
            audio_frames = []
            for _ in range(int(self.sample_rate / 1024 * block_duration)):
                audio_frames.append(self.q.get())
            audio_data = np.concatenate(audio_frames, axis=0)

            # Không cần ghi file, load trực tiếp numpy array
            segments, info = self.model.transcribe(audio_data, beam_size=5)
            text_out = " ".join([seg.text for seg in segments]).strip()
            return text_out, info.language

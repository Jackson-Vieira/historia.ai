from faster_whisper import WhisperModel

class TurboTranscriber:
    def __init__(self, model_size="medium", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
    
    def _full_text(self, segments):
        return "".join([segment.text for segment in segments])

    def transcribe(self, audio_path, beam_size=5, language="pt"):
        segments, info = self.model.transcribe(audio_path, beam_size=beam_size, language=language)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        return self._full_text(segments)

""" full_text = ""
for segment in segments:
    full_text += segment.text
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)) """
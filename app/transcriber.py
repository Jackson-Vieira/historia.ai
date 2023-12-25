from faster_whisper import WhisperModel

import os

from constants import MEDIA_TRANSCRIPTIONS_DIR, AUDIO_DIR

class TurboTranscriber:
    def __init__(self, model_size="medium", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        self.on_segment_complete = lambda segment, start, end: None
        self.on_word_complete = lambda word, start, end: None
    
    def _join_text(self, segments):
        return "".join([segment.text for segment in segments])

    def transcribe(self, audio_path, word_timestamps=False, beam_size=5, language="pt"):
        segments, info = self.model.transcribe(audio_path, word_timestamps=word_timestamps, beam_size=beam_size, language=language)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            if self.on_segment_complete:
                self.on_segment_complete(segment, segment.start, segment.end)
            for word in segment.words:
                if self.on_word_complete:
                    self.on_word_complete(word, word.start, word.end)

if __name__ == "__main__":
    transcriber = TurboTranscriber(model_size="large-v3", device="cuda", compute_type="float16")

    atividade_1_audios_path = [os.path.join(AUDIO_DIR, "atividade3", audio) for audio in os.listdir(os.path.join(AUDIO_DIR, "atividade3"))]
    atividade_2_audios_path = [os.path.join(AUDIO_DIR, "atividade4", audio) for audio in os.listdir(os.path.join(AUDIO_DIR, "atividade4"))]

    transcriptions = []
    for index, audio_path in enumerate(atividade_2_audios_path):
        audio_transcription = []
        transcriber.on_segment_complete = lambda segment, start, end: audio_transcription.append(segment.text)
        transcriber.transcribe(audio_path, word_timestamps=True)
        transcriptions.append("".join(audio_transcription))
    
    with open(os.path.join(MEDIA_TRANSCRIPTIONS_DIR, "atividade4.txt"), "w") as file:
        file.write("\n\n".join(transcriptions))
from faster_whisper import WhisperModel

import os

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
    data = {
        "audios_dir": "audios/Aula_1_IntroducaoAosOrixas_11-08-2021",
    }

    # Questao_1_13-09-2021.wav

    # regex to get (Numero da Aula, Conteudo da Aula, Data da Aula)
    # regex to get (Numero da Questao)

    # transcriptions = []
    # for index, audio_path in enumerate(audios_path):
    #     audio_transcription = []
    #     transcriber.on_segment_complete = lambda segment, start, end: audio_transcription.append(segment.text)
    #     transcriber.transcribe(audio_path, word_timestamps=True)
    #     transcriptions.append("".join(audio_transcription))

    # with open(os.path.join(cwd, "transcriptions", "transcription1.txt"), "w") as f:
    #     f.write("".join(transcriptions))
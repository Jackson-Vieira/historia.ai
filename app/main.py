from whisper import TurboTranscriber
from chat import Chat

import os

if __name__ == "__main__":
    transcriber = TurboTranscriber(model_size="small")
    # chat = Chat()
    cwd = os.getcwd()

    audios_path = (
        os.path.join(cwd, "audios", "audio1.ogg"),
        os.path.join(cwd, "audios", "audio2.ogg"),
        os.path.join(cwd, "audios", "audio3.ogg"),
        os.path.join(cwd, "audios", "audio4.ogg"),
        os.path.join(cwd, "audios", "audio5.ogg"),
        os.path.join(cwd, "audios", "audio6.ogg"),
    )

    for index, audio_path in enumerate(audios_path):
        transcription = []
        transcriber.on_segment_complete = lambda segment, start, end: transcription.append(segment.text)
        transcriber.transcribe(audio_path, word_timestamps=True)
        with  open(os.path.join(cwd, "transcriptions", f"transcription{index}.txt"), "w") as f:
            f.write("".join(transcription))

    # initialize chat
    # chat.initialize("transcription.txt", chunk_size=1000, chunk_overlap=20, persist_directory="chroma_db")
    
    # ask a question
    # question = ""
    # answer = chat.chat(question)
    # print("Question: ", question)
    # print("Answer: ", answer)
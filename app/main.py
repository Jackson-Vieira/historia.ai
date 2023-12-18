from whisper import TurboTranscriber
from chat import Chat

import os

if __name__ == "__main__":
    transcriber = TurboTranscriber(model_size="small")
    chat = Chat()
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "audios", "audio1.ogg")
    print("file path: ", file_path)
    transcriber.on_segment_complete = lambda segment, start, end: print(segment.text, start, end)
    transcriber.transcribe(file_path, word_timestamps=True)

    # initialize chat
    # chat.initialize("transcription.txt", chunk_size=1000, chunk_overlap=20, persist_directory="chroma_db")
    
    # ask a question
    # question = ""
    # answer = chat.chat(question)
    # print("Question: ", question)
    # print("Answer: ", answer)
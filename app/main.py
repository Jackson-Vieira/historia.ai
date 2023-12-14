from whisper import TurboTranscriber

import os

if __name__ == "__main__":
    transcriber = TurboTranscriber()
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "audio.ogg")
    transcript = transcriber.transcribe(file_path)
    print("transcript: ", transcript)
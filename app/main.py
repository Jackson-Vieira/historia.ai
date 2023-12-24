# from whisper import TurboTranscriber
from assistant import Assistant
from database import Database, DocumentManager

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

from constants import MEDIA_TRANSCRIPTIONS_DIR

import os

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE")
OPENAI_TOP_P = os.getenv("OPENAI_TOP_P")
OPENAI_FREQUENCY_PENALTY = os.getenv("OPENAI_FREQUENCY_PENALTY")
OPENAI_PRESENCE_PENALTY= os.getenv("OPENAI_PRESENCE_PENALTY")
OPENAI_MAX_TOKENS= os.getenv("OPENAI_MAX_TOKENS")
OPENAI_STREAM= os.getenv("OPENAI_STREAM")

TEXT_SPLITTER_CHUNK_SIZE = int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE"))
TEXT_SPLITTER_CHUNK_OVERLAP = int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP"))

if __name__ == "__main__":
    embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Database(embedding_function=embedding_function, disk_location="./chroma_db")
    text_splitter = CharacterTextSplitter(chunk_size=TEXT_SPLITTER_CHUNK_SIZE, chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP)
    document_manager = DocumentManager(database=db, text_splitter=text_splitter)
    llm = ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=float(OPENAI_TEMPERATURE),
        model_kwargs={
            "top_p": float(OPENAI_TOP_P),
            "frequency_penalty": float(OPENAI_FREQUENCY_PENALTY),
            "presence_penalty": float(OPENAI_PRESENCE_PENALTY),
        },
        max_tokens=int(OPENAI_MAX_TOKENS),
    )

    assistant = Assistant(llm=llm, db=db)

    file_path = os.path.join(MEDIA_TRANSCRIPTIONS_DIR, "transcription.txt")
    document_manager.add_document(file_path)
    
    # ask a question
    question = """Qual a cor do cabelo de Oxum? e qual seu significado"""

    answer = assistant.chat(question)
    print("Question: ", question)
    print("\nAnswer: ", answer)
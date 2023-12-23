# from whisper import TurboTranscriber
from assistant import Assistant
from database import Database, DocumentManager

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

import os

load_dotenv()

if __name__ == "__main__":
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Database(embedding_function=embedding_function, disk_location="./chroma_db")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    document_manager = DocumentManager(database=db, text_splitter=text_splitter)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    assistant = Assistant(llm=llm, db=db)

    assistant.create_chain(chain_type="stuff")

    cwd = os.getcwd()
    file_path = os.path.join(cwd, "transcriptions", "transcription1.txt")
    document_manager.add_document(file_path)
    
    # ask a question
    question = """
    A partir da transcrição da explicacao da PERGUNTA durante uma aula, responda as seguintes questões:

    2(Dois): Usando a historia de Exu, existe uma moralidade nesta religião?  

    Responda a questao de forma FORMAL e completa, trazendo reflexoes se possivel e com o maximo de detalhes possiveis, nao fujindo do contexto. Tambem seja o mais claro e OBJETIVO possivel.
    """

    answer = assistant.chat(question)
    print("Question: ", question)
    print("\nAnswer: ", answer)
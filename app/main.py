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

    file_path = os.path.join(MEDIA_TRANSCRIPTIONS_DIR, "atividade3.txt")
    document_manager.add_document(file_path)
    
    questions_atividade3 = [
        """
        1) Por que os trechos extraídos de Colombo e Córtes confirmam a primeira imagem sobre os mesmos por parte dos europeus: "povos sem lei, sem rei e sem Deus"?
        """,
        """
        2) De que forma os trechos de Iracema e do diário de Colombo confirmam a imagem do "índio" estereotipado? Como a fala de Celia Xakriaba se opõe a esta imagem?
        """,
        """
        3) As duas primeiras. leituras ditam alguns dos posicionamentos tomados pelo indígena diante do homem branco, que posições são essas? Como elas se diferenciam do posicionamento verificado em
        A queda do ceu?
        """,
        """
        4) Tomando como base o vídeo de Célia Xakriabá e o livro de Davi Kopenawa, de que forma o indigena narrado por ambos foge da separação generalista entre lupis e apuras, aproximando-se, desse modo, de uma perspectiva de resistência diterenciada?
        """,
        """
        5) Pensando as figuras de Célia Xakriabá e de Davi Kopenawa, por que é tão complicado delimitar quem e ou não indígena no Brasil? Como isso afeta a luta indigena?
        """,
    ]

    # questions_atividade4 = [
    #     # """
    #     # 1) A partir do relato, por que as "pessoas comuns" são chamadas de kuapora l'è pè literalmente "gente que simplesmente existe") e os xamãs de xapiri 'è pe (literalmente "gente espírito")?
    #     # """,
    #     # """
    #     # 2) Como este texto ajuda a entender as dúvidas de Levi-Strauss: por que os indigenas colocaram os espanhois já mortos num barril e por que os espanhóis chamaram teólogos para analisar os indigenas, quando dos momentos iniciais do encontro? 
    #     # """,
    #     # """
    #     # 3) Nós dizemos que todos os seres são animais e, de fato, quando nos descontrolamos, tambem costumamos dizer que a pessoa é um "animal", que ela está "selvagem. Os indígenas Yanomami diriam a mesma coisa? Qual a diferença sobre a questão de animal e humano neste discurso?
    #     # """,
    #     # """
    #     # 4) Sabendo que os Yanomami praticavam ritualmente o canibalismo, explique esta prática a partir do pensamento deles sobre o mundo.
    #     # """,
    #     # """
    #     # 5) Quais as diferenças que esta visão sobre a natureza implica na maneira que os indígenas lidam com a floresta? Sabendo qual é a forma da sociedade ocidental contemporânea lidar com a floresta, como interpretamos a natureza.
    #     # """,
    # ]

    answers_atividade4 = []

    for question in questions_atividade3:
        answer = assistant.chat(question, k=1)
        answers_atividade4.append(answer)
    
    with open("atividade4.5_answers.txt", "w") as f:
        for question, answer in zip(questions_atividade3, answers_atividade4):
            f.write(f"{question}\n")
            f.write(f"{answer}\n")
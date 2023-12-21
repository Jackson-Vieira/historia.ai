from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv

import os

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

class Chat:
    def __init__(self, embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), model_name=MODEL_NAME):
        self.embeddings = embeddings
        self.llm = ChatOpenAI(model_name=model_name)
        self.db = None
        self.chain = None

    def load_document(self, path: str):
        loader = TextLoader(path)
        return loader.load()

    def load_directory(self, path: str):
        loader = DirectoryLoader(path)
        return loader.load()

    def split_documents(self, document, chunk_size=1000, chunk_overlap=20):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = splitter.split_documents(document)
        return documents
    
    def load_db_from_documents(self, documents, persist_directory=None):
        self.db = Chroma.from_documents(documents, self.embeddings, persist_directory=persist_directory)
        self.db.persist()
        return self.db
    
    def create_qa_chain(self, chain_type="stuff"):
        self.chain = load_qa_chain(self.llm, chain_type=chain_type, verbose=True)
        return self.chain
    
    def search(self, question, k=20):
        matches = self.db.similarity_search(question, k=k)
        return matches
    
    def answer(self, question, matches) -> str:
        # TODO: Return a server error to user if chain not created
        if len(matches) == 0:
            # TODO: Add a fallback? or just return a custom error message to user
            raise Exception("No matches found")
        answer =  self.chain.run(input_documents=matches, question=question)
        return answer
    
    def initialize(self, path, chunk_size=1000, chunk_overlap=20, persist_directory=None):
        document = self.load_directory(path)
        segments = self.split_documents(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.load_db_from_documents(segments, persist_directory=persist_directory)
        self.create_qa_chain()
        return self.db, self.chain
    
    def chat(self, question, k=20):
        matches = self.search(question, k=k)
        answer = self.answer(question, matches)
        return answer

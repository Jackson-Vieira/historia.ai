from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
# from langchain.chains import RetrievalQA

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

    def split_document(self, document, chunk_size=1000, chunk_overlap=20):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        segments = splitter.split_documents(document)
        return segments
    
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
    
    def answer(self, question, matches):
        answer =  self.chain.run(input_documents=matches, question=question)
        return answer
    

# if __name__ == "__main__":
#     file_path = os.path.join(os.path.dirname(__file__), "test.txt")
#     document = load_document(file_path)
#     docs = split_document(document)
#     persist_directory = "chroma_db"
#     vectordb = Chroma.from_documents(
#         documents=docs, embedding=embeddings, persist_directory=persist_directory
#     )
#     vectordb.persist()

#     question = "Explique como a fe cristã é uma fé racional"
#     matches = db.similarity_search(question, k=20)
#     print(matches[0])
#     answer =  chain.run(input_documents=matches, question=question)
#     print(answer)
# retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
# retrieval_chain.run(question)
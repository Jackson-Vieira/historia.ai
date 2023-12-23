# from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

import os

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

class Assistant:
    def __init__(self, llm, db):
        self.llm = llm
        self.db = db
        self.chain = None
        self.create_chain(chain_type="stuff")
    
    def chat(self, question, k=20):
        matches = self.search(question, k=k)
        answer = self.answer(question, matches)
        return answer
    
    def search(self, question, k=20):
        matches = self.db.similarity_search(question, k=k)
        return matches
    
    def answer(self, question, matches) -> str:
        return self.chain.run(input_documents=matches, question=question)
    
    def create_chain(self, chain_type="stuff"):
        self.chain = load_qa_chain(self.llm, chain_type=chain_type, verbose=True)